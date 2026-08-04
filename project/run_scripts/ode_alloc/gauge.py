"""Frozen-basis factor-Gram energy gauge for ODE-Alloc."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .contracts import ODEAllocContractError, finite, positive


class BasisEligibilityError(ODEAllocContractError):
    code = "INELIGIBLE_BASIS_ENERGY"


class QDomainError(ODEAllocContractError):
    code = "Q_CAP_HIT"


@dataclass(frozen=True, slots=True)
class FactorPair:
    layer: int
    left: torch.Tensor
    right: torch.Tensor

    def __post_init__(self) -> None:
        if isinstance(self.layer, bool) or not isinstance(self.layer, int):
            raise ODEAllocContractError("factor layer must be an integer")
        if self.left.ndim != 2 or self.right.ndim != 2:
            raise ODEAllocContractError("low-rank factors must be matrices")
        if self.left.shape[1] != self.right.shape[1] or self.left.shape[1] == 0:
            raise ODEAllocContractError("factor ranks differ or are empty")
        if not torch.isfinite(self.left).all() or not torch.isfinite(self.right).all():
            raise ODEAllocContractError("factor entries must be finite")


def factor_gram_energy(pair: FactorPair) -> float:
    """Compute ||L R^T||_F^2 in float64 without forming the dense update."""

    left = pair.left.detach().to(device="cpu", dtype=torch.float64)
    right = pair.right.detach().to(device="cpu", dtype=torch.float64)
    left_gram = left.transpose(0, 1) @ left
    right_gram = right.transpose(0, 1) @ right
    energy = torch.sum(left_gram * right_gram)
    result = float(energy)
    if not torch.isfinite(energy) or result < 0.0:
        raise ODEAllocContractError("factor-Gram energy is invalid")
    return result


@dataclass(frozen=True, slots=True)
class GaugeReading:
    layers: tuple[int, ...]
    centered_q: torch.Tensor
    ratios: torch.Tensor
    energies: torch.Tensor
    weights: torch.Tensor
    excluded_layers: tuple[int, ...]

    @property
    def energy_before(self) -> torch.Tensor:
        return self.energies.sum()

    @property
    def energy_after(self) -> torch.Tensor:
        return torch.sum(self.energies * self.ratios.square())

    def ratio_by_layer(self) -> dict[int, torch.Tensor]:
        return dict(zip(self.layers, self.ratios, strict=True))


class FixedEnergyGauge:
    """A zero-mean log-ratio chart with no global-strength degree of freedom."""

    def __init__(
        self,
        factors: Mapping[int, FactorPair],
        *,
        basis_energy_epsilon: float,
        max_abs_centered_q: float,
        quantized_zero_by_layer: Mapping[int, bool] | None = None,
    ) -> None:
        self.basis_energy_epsilon = positive(
            "basis energy epsilon", basis_energy_epsilon
        )
        self.max_abs_centered_q = positive(
            "maximum centered q", max_abs_centered_q
        )
        if not factors:
            raise ODEAllocContractError("frozen basis is empty")
        observed: list[tuple[int, float]] = []
        for layer, pair in sorted(factors.items()):
            if layer != pair.layer:
                raise ODEAllocContractError("factor mapping and embedded layer differ")
            observed.append((layer, factor_gram_energy(pair)))
        zero_attestation = dict(quantized_zero_by_layer or {})
        exact_zero_layers = tuple(layer for layer, energy in observed if energy == 0.0)
        if set(zero_attestation) - set(exact_zero_layers):
            raise BasisEligibilityError(
                "INELIGIBLE_BASIS_ENERGY: zero attestation names a nonzero layer"
            )
        for layer in exact_zero_layers:
            if zero_attestation.get(layer) is not True:
                raise BasisEligibilityError(
                    "INELIGIBLE_BASIS_ENERGY: exact-zero Gram layer lacks exact "
                    "quantized Native zero attestation"
                )
        near_zero_layers = tuple(
            layer
            for layer, energy in observed
            if 0.0 < energy <= self.basis_energy_epsilon
        )
        if near_zero_layers:
            raise BasisEligibilityError(
                "INELIGIBLE_BASIS_ENERGY: positive near-zero factor-Gram energy"
            )
        self.excluded_layers = exact_zero_layers
        active = tuple(
            (layer, energy)
            for layer, energy in observed
            if energy > self.basis_energy_epsilon
        )
        if len(active) < 2:
            raise BasisEligibilityError(
                "INELIGIBLE_BASIS_ENERGY: fewer than two active layers"
            )
        self.layers = tuple(layer for layer, _ in active)
        self.energies = torch.tensor(
            [energy for _, energy in active], dtype=torch.float64
        )
        total = self.energies.sum()
        if not torch.isfinite(total) or float(total) <= self.basis_energy_epsilon:
            raise ODEAllocContractError("active basis energy is degenerate")
        self.weights = self.energies / total

    @property
    def dimension(self) -> int:
        return len(self.layers) - 1

    def evaluate(self, q_by_layer: Mapping[int, torch.Tensor | float]) -> GaugeReading:
        if tuple(sorted(q_by_layer)) != self.layers:
            raise ODEAllocContractError("q keys differ from the active frozen basis")
        raw_values = [q_by_layer[layer] for layer in self.layers]
        tensors = [
            value
            if isinstance(value, torch.Tensor)
            else torch.tensor(finite("q", value), dtype=torch.float64)
            for value in raw_values
        ]
        device = tensors[0].device
        if any(item.ndim != 0 or item.device != device for item in tensors):
            raise ODEAllocContractError("q values must be scalar tensors on one device")
        q = torch.stack([item.to(dtype=torch.float64) for item in tensors])
        if not torch.isfinite(q).all():
            raise ODEAllocContractError("q must be finite")
        centered = q - q.mean()
        if float(torch.max(torch.abs(centered)).detach().cpu()) > self.max_abs_centered_q:
            raise QDomainError("Q_CAP_HIT: centered q exceeds the fixed numerical domain")
        weights = self.weights.to(device=device)
        energies = self.energies.to(device=device)
        # sum(w)=1 gives the equivalent denominator.  expm1 makes q=0 yield
        # denominator==1 and therefore r==1 exactly while retaining gradients.
        denominator_sq = 1.0 + torch.sum(weights * torch.expm1(2.0 * centered))
        if not torch.isfinite(denominator_sq) or float(denominator_sq.detach().cpu()) <= 0.0:
            raise ODEAllocContractError("gauge normalization is non-finite")
        ratios = torch.exp(centered) / torch.sqrt(denominator_sq)
        return GaugeReading(
            layers=self.layers,
            centered_q=centered,
            ratios=ratios,
            energies=energies,
            weights=weights,
            excluded_layers=self.excluded_layers,
        )

    def zeros(self, *, requires_grad: bool = False) -> dict[int, torch.Tensor]:
        return {
            layer: torch.zeros((), dtype=torch.float64, requires_grad=requires_grad)
            for layer in self.layers
        }
