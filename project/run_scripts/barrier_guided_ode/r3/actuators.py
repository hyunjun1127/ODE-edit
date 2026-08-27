"""Normalized native-derived AlphaEdit actuator coordinates."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import torch

from project.run_scripts.ode_edit_motivation.contracts import LowRankFactor, MemitFactorProposal

from .errors import ActuatorBoundary, R3ScientificBoundary


def _factor_norm_fp64(factor: LowRankFactor) -> float:
    left = factor.left.detach().cpu().to(dtype=torch.float64)
    right = factor.right.detach().cpu().to(dtype=torch.float64)
    value = torch.sum((left.transpose(0, 1) @ left) * (right.transpose(0, 1) @ right))
    if not bool(torch.isfinite(value)) or float(value) <= 0.0:
        raise ActuatorBoundary("AlphaEdit factor has zero or invalid Frobenius norm")
    result = float(torch.sqrt(value).cpu().item())
    if not math.isfinite(result) or result <= 0.0:
        raise ActuatorBoundary("AlphaEdit factor norm conversion failed")
    return result


def _factor_sha(factor: LowRankFactor) -> str:
    digest = hashlib.sha256()
    for label, value in (("left", factor.left), ("right", factor.right)):
        tensor = value.detach().cpu().contiguous()
        digest.update(label.encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class NormalizedActuatorBasis:
    normalized_factors: tuple[LowRankFactor, ...]
    raw_scales: torch.Tensor
    raw_factor_sha256: tuple[str, ...]

    @classmethod
    def from_factors(cls, factors: Sequence[LowRankFactor]) -> "NormalizedActuatorBasis":
        raw = tuple(factors)
        if not raw or len({factor.weight_name for factor in raw}) != len(raw):
            raise ActuatorBoundary("actuator factor inventory is empty or duplicated")
        scales = tuple(_factor_norm_fp64(factor) for factor in raw)
        normalized: list[LowRankFactor] = []
        for factor, scale in zip(raw, scales, strict=True):
            normalized.append(
                LowRankFactor(
                    weight_name=factor.weight_name,
                    left=(factor.left.detach().float() / scale).contiguous(),
                    right=factor.right.detach().float().contiguous(),
                    expected_weight_sha256=factor.expected_weight_sha256,
                    native_update_transposed=factor.native_update_transposed,
                )
            )
        return cls(
            normalized_factors=tuple(normalized),
            raw_scales=torch.tensor(scales, dtype=torch.float64),
            raw_factor_sha256=tuple(_factor_sha(factor) for factor in raw),
        )

    def raw_coefficients_fp32(self, normalized_coefficients: torch.Tensor) -> torch.Tensor:
        if (
            normalized_coefficients.dtype != torch.float64
            or normalized_coefficients.shape != self.raw_scales.shape
            or normalized_coefficients.requires_grad
            or not bool(torch.isfinite(normalized_coefficients).all())
        ):
            raise R3ScientificBoundary("normalized coefficients must be detached finite FP64")
        result = normalized_coefficients / self.raw_scales
        if not bool(torch.isfinite(result).all()):
            raise ActuatorBoundary("raw coefficient conversion failed")
        return result.to(dtype=torch.float32).contiguous()

    def normalized_proposal(self, raw: MemitFactorProposal) -> MemitFactorProposal:
        if len(raw.factors) != len(self.normalized_factors):
            raise R3ScientificBoundary("normalized proposal factor inventory differs")
        if tuple(factor.weight_name for factor in raw.factors) != tuple(
            factor.weight_name for factor in self.normalized_factors
        ):
            raise R3ScientificBoundary("normalized proposal factor order differs")
        return MemitFactorProposal(
            snapshot=raw.snapshot,
            factors=self.normalized_factors,
            semantics=raw.semantics,
            solver_name=f"{raw.solver_name}/bgode-r3-frobenius-normalized",
            residual_denominator=raw.residual_denominator,
        )

    def block_frobenius(self, normalized_coefficients: torch.Tensor) -> float:
        if normalized_coefficients.dtype != torch.float64 or normalized_coefficients.shape != self.raw_scales.shape:
            raise R3ScientificBoundary("normalized coefficient geometry differs")
        return float(torch.linalg.vector_norm(normalized_coefficients).cpu().item())

    def ordered_prefix_mismatch(self, coefficient: torch.Tensor) -> tuple[dict[str, float], ...]:
        if coefficient.dtype != torch.float64 or coefficient.shape != self.raw_scales.shape:
            raise R3ScientificBoundary("ordered mismatch requires the applied beta vector")
        rows: list[dict[str, float]] = []
        for index in range(len(self.normalized_factors)):
            raw_prefix = self.raw_scales[:index]
            applied_prefix = coefficient[:index]
            rows.append(
                {
                    "unit_prefix_frobenius": float(torch.linalg.vector_norm(raw_prefix).cpu().item()),
                    "applied_prefix_frobenius": float(torch.linalg.vector_norm(applied_prefix).cpu().item()),
                    "unit_applied_mismatch_frobenius": float(
                        torch.linalg.vector_norm(applied_prefix - raw_prefix).cpu().item()
                    ),
                }
            )
        return tuple(rows)


@dataclass(slots=True)
class NodeBuildLedger:
    ordered_build_count: int = 0
    validation_call_count: int = 0
    live_factor_count: int = 0
    retained_factor_count: int = 0

    def publish(self, basis: NormalizedActuatorBasis, *, validation_call_count: int) -> None:
        if self.live_factor_count != 0 or validation_call_count != 1:
            raise R3ScientificBoundary("R3 node build publication boundary failed")
        self.ordered_build_count += 1
        self.validation_call_count += validation_call_count
        self.live_factor_count = len(basis.normalized_factors)

    def release_node(self) -> None:
        if self.live_factor_count <= 0:
            raise R3ScientificBoundary("R3 node factor release has no live build")
        self.live_factor_count = 0
        self.retained_factor_count = 0


__all__ = ["NodeBuildLedger", "NormalizedActuatorBasis"]
