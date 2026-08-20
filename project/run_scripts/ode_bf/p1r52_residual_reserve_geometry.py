"""Pure residual-reserve geometry for the P1R52 sequential writer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .contracts import ODEBFContractError, canonical_hash


RESIDUAL_RESERVE_LAYER_COUNT = 5
RESIDUAL_RESERVE_RHO_REF = (4.0 / 5.0) ** RESIDUAL_RESERVE_LAYER_COUNT
RESIDUAL_RESERVE_H_REF = 1.0 / 8.0
RESIDUAL_RESERVE_KAPPA = -math.log(RESIDUAL_RESERVE_RHO_REF) / RESIDUAL_RESERVE_H_REF
FP32_SIMPLEX_SUM_TOLERANCE = (
    RESIDUAL_RESERVE_LAYER_COUNT * torch.finfo(torch.float32).eps
)


@dataclass(frozen=True, slots=True)
class ResidualReserveGeometryReceipt:
    """Immutable raw-free certificate for one residual-reserve allocation."""

    h: float
    kappa: float
    rho: float
    mass: float
    pi: tuple[float, ...]
    omega: tuple[float, ...]
    suffix_retention: tuple[float, ...]
    beta: tuple[float, ...]
    simplex_sum_residual: float
    contribution_sum_residual: float
    maximum_beta: float
    last_beta: float

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": "ode-edit-s05-p1r52-residual-reserve-geometry/v1",
            "layer_count": RESIDUAL_RESERVE_LAYER_COUNT,
            "h": self.h,
            "h_ref": RESIDUAL_RESERVE_H_REF,
            "rho_ref": RESIDUAL_RESERVE_RHO_REF,
            "kappa": self.kappa,
            "rho": self.rho,
            "mass": self.mass,
            "pi": list(self.pi),
            "omega": list(self.omega),
            "suffix_retention": list(self.suffix_retention),
            "beta": list(self.beta),
            "simplex_sum_residual": self.simplex_sum_residual,
            "contribution_sum_residual": self.contribution_sum_residual,
            "maximum_beta": self.maximum_beta,
            "last_beta": self.last_beta,
            "tensor_dtype": "torch.float32",
            "simplex_sum_tolerance": FP32_SIMPLEX_SUM_TOLERANCE,
            "no_debt_lag_catch_up_state": True,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


@dataclass(frozen=True, slots=True)
class ResidualReserveGeometry:
    """FP32 tensors and their immutable raw-free receipt."""

    pi: torch.Tensor
    omega: torch.Tensor
    suffix_retention: torch.Tensor
    beta: torch.Tensor
    rho: torch.Tensor
    mass: torch.Tensor
    receipt: ResidualReserveGeometryReceipt


def _validated_fp32_vector(name: str, value: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype is not torch.float32
        or value.ndim != 1
        or value.numel() != RESIDUAL_RESERVE_LAYER_COUNT
        or not bool(torch.isfinite(value).all())
    ):
        raise ODEBFContractError(f"residual-reserve {name} contract differs")
    return value.detach().clone().contiguous()


def build_residual_reserve_geometry(
    pi: torch.Tensor,
    *,
    h: float = RESIDUAL_RESERVE_H_REF,
) -> ResidualReserveGeometry:
    """Build the fixed-contraction layer geometry without model state."""

    step = float(h)
    if not math.isfinite(step) or step <= 0.0:
        raise ODEBFContractError("residual-reserve step is not positive finite")
    allocation = _validated_fp32_vector("simplex", pi)
    if bool((allocation < 0.0).any()):
        raise ODEBFContractError("residual-reserve simplex is negative")
    simplex_sum = allocation.sum(dtype=torch.float32)
    simplex_residual = abs(float(simplex_sum) - 1.0)
    if simplex_residual > FP32_SIMPLEX_SUM_TOLERANCE:
        raise ODEBFContractError("residual-reserve simplex sum differs")

    rho = torch.tensor(
        math.exp(-RESIDUAL_RESERVE_KAPPA * step),
        dtype=torch.float32,
        device=allocation.device,
    )
    mass = torch.ones((), dtype=torch.float32, device=allocation.device) - rho
    if (
        not bool(torch.isfinite(rho))
        or not bool(torch.isfinite(mass))
        or not (0.0 < float(rho) < 1.0)
        or not (0.0 < float(mass) < 1.0)
    ):
        raise ODEBFContractError("residual-reserve contraction differs")

    omega = (mass * allocation).contiguous()
    suffix = torch.flip(
        torch.cumsum(torch.flip(omega, dims=(0,)), dim=0, dtype=torch.float32),
        dims=(0,),
    )
    suffix_retention = (rho + suffix).contiguous()
    beta = (omega / suffix_retention).contiguous()
    if not bool(torch.isfinite(beta).all()) or bool((suffix_retention <= 0.0).any()):
        raise ODEBFContractError("residual-reserve suffix geometry is invalid")
    maximum_beta = beta.max()
    permitted_maximum = torch.nextafter(
        mass, torch.tensor(float("inf"), dtype=torch.float32, device=mass.device)
    )
    if bool(maximum_beta > permitted_maximum):
        raise ODEBFContractError("residual-reserve last-dump certificate differs")
    contribution_residual = abs(float(omega.sum(dtype=torch.float32) - mass))
    if contribution_residual > FP32_SIMPLEX_SUM_TOLERANCE:
        raise ODEBFContractError("residual-reserve contribution sum differs")

    def values(tensor: torch.Tensor) -> tuple[float, ...]:
        return tuple(float(item) for item in tensor.detach().to(device="cpu"))

    receipt = ResidualReserveGeometryReceipt(
        h=step,
        kappa=RESIDUAL_RESERVE_KAPPA,
        rho=float(rho),
        mass=float(mass),
        pi=values(allocation),
        omega=values(omega),
        suffix_retention=values(suffix_retention),
        beta=values(beta),
        simplex_sum_residual=simplex_residual,
        contribution_sum_residual=contribution_residual,
        maximum_beta=float(maximum_beta),
        last_beta=float(beta[-1]),
    )
    return ResidualReserveGeometry(
        allocation,
        omega,
        suffix_retention,
        beta,
        rho,
        mass,
        receipt,
    )


def current_prefix_residual(
    target_proposal: torch.Tensor,
    current_terminal: torch.Tensor,
) -> torch.Tensor:
    """Return the complete current defect; no carried writer state exists."""

    if (
        not isinstance(target_proposal, torch.Tensor)
        or not isinstance(current_terminal, torch.Tensor)
        or target_proposal.dtype is not torch.float32
        or current_terminal.dtype is not torch.float32
        or target_proposal.shape != current_terminal.shape
        or target_proposal.device != current_terminal.device
        or not bool(torch.isfinite(target_proposal).all())
        or not bool(torch.isfinite(current_terminal).all())
    ):
        raise ODEBFContractError("residual-reserve current-prefix input differs")
    residual = (target_proposal.detach() - current_terminal.detach()).contiguous()
    if residual.dtype is not torch.float32 or not bool(torch.isfinite(residual).all()):
        raise ODEBFContractError("residual-reserve current-prefix result differs")
    return residual


def ideal_layer_contributions(
    entry_residual: torch.Tensor,
    geometry: ResidualReserveGeometry,
) -> torch.Tensor:
    """Return the ideal contribution ``m*pi*E_entry`` for all five layers."""

    if (
        not isinstance(entry_residual, torch.Tensor)
        or entry_residual.dtype is not torch.float32
        or not bool(torch.isfinite(entry_residual).all())
        or entry_residual.device != geometry.omega.device
    ):
        raise ODEBFContractError("residual-reserve ideal residual differs")
    shape = (RESIDUAL_RESERVE_LAYER_COUNT,) + (1,) * entry_residual.ndim
    result = geometry.omega.reshape(shape) * entry_residual.detach().unsqueeze(0)
    if result.dtype is not torch.float32 or not bool(torch.isfinite(result).all()):
        raise ODEBFContractError("residual-reserve ideal contribution differs")
    return result.contiguous()


def ideal_final_residual(
    entry_residual: torch.Tensor,
    geometry: ResidualReserveGeometry,
) -> torch.Tensor:
    """Return the ideal final residual ``rho*E_entry``."""

    if (
        not isinstance(entry_residual, torch.Tensor)
        or entry_residual.dtype is not torch.float32
        or not bool(torch.isfinite(entry_residual).all())
        or entry_residual.device != geometry.rho.device
    ):
        raise ODEBFContractError("residual-reserve ideal final input differs")
    result = geometry.rho * entry_residual.detach()
    if result.dtype is not torch.float32 or not bool(torch.isfinite(result).all()):
        raise ODEBFContractError("residual-reserve ideal final residual differs")
    return result.contiguous()


__all__ = [
    "FP32_SIMPLEX_SUM_TOLERANCE",
    "RESIDUAL_RESERVE_H_REF",
    "RESIDUAL_RESERVE_KAPPA",
    "RESIDUAL_RESERVE_LAYER_COUNT",
    "RESIDUAL_RESERVE_RHO_REF",
    "ResidualReserveGeometry",
    "ResidualReserveGeometryReceipt",
    "build_residual_reserve_geometry",
    "current_prefix_residual",
    "ideal_final_residual",
    "ideal_layer_contributions",
]
