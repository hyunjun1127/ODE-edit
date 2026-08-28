"""Capture one immutable, normalized Official AlphaEdit proposal basis at W0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from project.run_scripts.barrier_guided_ode.r3.actuators import NormalizedActuatorBasis
from project.run_scripts.ode_edit_motivation.contracts import MemitFactorProposal


class FixedBasisBoundary(RuntimeError):
    """The fixed native proposal basis is invalid or has drifted."""


@dataclass(frozen=True, slots=True)
class FixedAlphaEditBasis:
    raw_proposal: MemitFactorProposal
    normalized_proposal: MemitFactorProposal
    theta_ae: torch.Tensor
    factor_sha256: tuple[str, ...]
    captured_build_count: int = 1

    @classmethod
    def capture(cls, raw: MemitFactorProposal, *, factor_sha256: tuple[str, ...]) -> "FixedAlphaEditBasis":
        if len(raw.factors) != 5 or len(factor_sha256) != 5:
            raise FixedBasisBoundary("fixed AlphaEdit basis must contain the ordered five layers")
        normalized = NormalizedActuatorBasis.from_factors(raw.factors)
        proposal = normalized.normalized_proposal(raw)
        theta = normalized.raw_scales.detach().clone().to(dtype=torch.float64).contiguous()
        if theta.shape != (5,) or not bool(torch.isfinite(theta).all()) or not bool((theta > 0).all()):
            raise FixedBasisBoundary("native AlphaEdit coefficient vector is invalid")
        return cls(
            raw_proposal=raw,
            normalized_proposal=proposal,
            theta_ae=theta,
            factor_sha256=tuple(factor_sha256),
        )

    def raw_coefficients(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.dtype != torch.float64 or theta.shape != self.theta_ae.shape:
            raise FixedBasisBoundary("fixed-basis theta must be FP64 with five coordinates")
        value = theta / self.theta_ae
        if not bool(torch.isfinite(value).all()):
            raise FixedBasisBoundary("normalized-to-raw coefficient conversion is non-finite")
        return value.to(dtype=torch.float32).contiguous()

    def receipt(self) -> dict[str, Any]:
        return {
            "basis_semantics": "W0_OFFICIAL_ORDERED_ALPHAEDIT_ONE_FACTOR_PER_LAYER",
            "captured_build_count": self.captured_build_count,
            "factor_count": 5,
            "factor_sha256": list(self.factor_sha256),
            "raw_scales": [float(value) for value in self.theta_ae.tolist()],
            "theta_ae_norm": float(torch.linalg.vector_norm(self.theta_ae)),
            "virtual_factor_rebuild_count": 0,
            "z_recompute_count": 0,
        }
