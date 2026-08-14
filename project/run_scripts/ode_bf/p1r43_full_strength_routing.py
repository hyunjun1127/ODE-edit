"""P1R43 full-strength Neutral/Soft routing totality wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import ODEBFContractError, canonical_hash
from .fixed_e8_soft_routing import FixedE8Arm, RoutingProblem
from .progress_simplex_routing import SIMPLEX_PRIMAL_TOLERANCE
from .p1r24_atomic_strength import (
    P1R24RoutingResult,
    P1R24RoutingStatus,
    solve_p1r24_matched_routing,
)
from .p1r43_rho_free_target import P1R43_INSTRUCTION_ID, P1R43_METHOD_ID


class P1R43SemanticNoPositiveDirection(RuntimeError):
    """Typed scientific boundary for a positive demand without a writer route."""


@dataclass(frozen=True, slots=True)
class P1R43RoutingResult:
    requested_arm: FixedE8Arm
    selected: P1R24RoutingResult
    fallback_to_neutral: bool
    fallback_reason: str | None
    identity_sha256: str

    def __getattr__(self, name: str) -> Any:
        return getattr(self.selected, name)

    @property
    def arm(self) -> FixedE8Arm:
        return self.requested_arm

    @property
    def alpha_req(self) -> float:
        return self.selected.rho_write

    @property
    def alpha_apply(self) -> float:
        return self.selected.predicted_progress

    @property
    def coverage(self) -> float:
        return 1.0 if self.alpha_req == 0.0 else self.alpha_apply / self.alpha_req

    def raw_free_payload(self) -> dict[str, Any]:
        payload = self.selected.raw_free_payload()
        payload.update(
            {
                "schema": "ode-edit-s05-p1r43-full-strength-routing/v1",
                "instruction_id": P1R43_INSTRUCTION_ID,
                "method_id": P1R43_METHOD_ID,
                "requested_arm": self.requested_arm.value,
                "executed_arm": self.selected.arm.value,
                "writer_demand_name": "alpha_req",
                "legacy_rho_write_alias_only": True,
                "alpha_req": self.alpha_req,
                "alpha_apply": self.alpha_apply,
                "alpha_apply_over_req": self.coverage,
                "fallback_to_neutral": self.fallback_to_neutral,
                "fallback_reason": self.fallback_reason,
                "p_capacity_strength_attenuation_count": 0,
                "hard_p_budget_influence_count": 0,
                "quarter_floor_influence_count": 0,
                "minimum_alpha_attenuation_count": 0,
                "identity_sha256": self.identity_sha256,
            }
        )
        return payload


def solve_p1r43_full_strength_routing(
    problem: RoutingProblem,
    *,
    arm: FixedE8Arm | str,
    alpha_req: float,
) -> P1R43RoutingResult:
    requested = FixedE8Arm(arm)
    neutral = solve_p1r24_matched_routing(
        problem, arm=FixedE8Arm.NEUTRAL, rho_write=alpha_req
    )
    if alpha_req > 0.0 and neutral.status is not P1R24RoutingStatus.JOINT_WRITE:
        raise P1R43SemanticNoPositiveDirection(neutral.status.value)
    selected = neutral
    fallback = False
    fallback_reason: str | None = None
    if requested is FixedE8Arm.SOFT and alpha_req > 0.0:
        try:
            candidate = solve_p1r24_matched_routing(
                problem, arm=FixedE8Arm.SOFT, rho_write=alpha_req
            )
            if (
                candidate.status is not P1R24RoutingStatus.JOINT_WRITE
                or abs(candidate.predicted_progress - alpha_req)
                > SIMPLEX_PRIMAL_TOLERANCE
            ):
                fallback = True
                fallback_reason = "SOFT_FULL_STRENGTH_UNCERTIFIED"
            else:
                selected = candidate
        except ODEBFContractError as exc:
            text = str(exc)
            if not any(
                marker in text
                for marker in (
                    "Structural-P certificate failed",
                    "capacity certificate failed",
                    "selected strength differs",
                )
            ):
                raise
            fallback = True
            fallback_reason = "SOFT_CERTIFICATE_FAILURE_NEUTRAL_FULL_STRENGTH_FALLBACK"
    residual = abs(selected.predicted_progress - alpha_req)
    if alpha_req > 0.0 and residual > SIMPLEX_PRIMAL_TOLERANCE:
        raise ODEBFContractError("P1R43 selected writer strength differs")
    identity = canonical_hash(
        {
            "requested_arm": requested.value,
            "selected_identity_sha256": selected.identity_sha256,
            "alpha_req": alpha_req,
            "alpha_apply": selected.predicted_progress,
            "fallback": fallback,
            "fallback_reason": fallback_reason,
        }
    )
    return P1R43RoutingResult(requested, selected, fallback, fallback_reason, identity)


__all__ = [
    "P1R43RoutingResult",
    "P1R43SemanticNoPositiveDirection",
    "solve_p1r43_full_strength_routing",
]
