"""P2R4 single-variable clamp-off policy for the exact P2R1 target field."""

from __future__ import annotations

from typing import Any

import torch

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .p1r24_atomic_strength import P1R24AliasTargetLock
from .p2r1_rms_tangent_target import (
    P2R1RMSState,
    P2R1TargetUpdate,
    p2r1_target_update,
)


P2R4_INSTRUCTION_ID = "ODEEDIT-S05-P2R4-P2R1-CLAMP-ON-OFF-CAUSAL-ABLATION-V1"
P2R4_METHOD_ID = "P2R4-P2R1-CUMULATIVE-RMS-TANGENT-CLAMP-OFF-V1"
P2R4_METHOD = "P2R4-P2R1-CLAMP-OFF-TARGET-ONLY"
P2R4_CLAMP_POLICY = "IDENTITY_OFF_WITH_FROZEN_ON_WOULD_HIT_OBSERVATION"


def _ratios(numerator: torch.Tensor, denominator: torch.Tensor) -> list[float]:
    if numerator.shape != denominator.shape or bool(torch.any(denominator <= 0.0)):
        raise ODEBFContractError("P2R4 target/origin ratio geometry differs")
    return [float(item) for item in numerator / denominator]


def p2r4_clamp_off_target_update(
    current_target: torch.Tensor,
    target_origin: torch.Tensor,
    semantic_gradient: torch.Tensor,
    preservation_gradient: torch.Tensor,
    state: P2R1RMSState,
    *,
    alias: str,
    microstep_index: int,
    lock: P1R24AliasTargetLock,
) -> P2R1TargetUpdate:
    """Use the exact P2R1 proposal and replace only Pi_clamp by identity."""

    parent = p2r1_target_update(
        current_target,
        target_origin,
        semantic_gradient,
        preservation_gradient,
        state,
        alias=alias,
        microstep_index=microstep_index,
        lock=lock,
    )
    current64 = current_target.detach().to(device="cpu", dtype=torch.float64)
    origin64 = target_origin.detach().to(device="cpu", dtype=torch.float64)
    field64 = parent.field.detach().to(device="cpu", dtype=torch.float64)
    eta_m = float(parent.receipt["eta_m"])
    proposal64 = current64 + eta_m * field64
    if not bool(torch.isfinite(proposal64).all()):
        raise ODEBFContractError("P2R4 clamp-off proposal is nonfinite")
    target_next = proposal64.to(dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(target_next).all()):
        raise ODEBFContractError("P2R4 clamp-off target is nonfinite")

    origin_norm = torch.linalg.vector_norm(origin64, dim=0)
    target_before_norm = torch.linalg.vector_norm(current64, dim=0)
    target_next64 = target_next.to(torch.float64)
    target_next_norm = torch.linalg.vector_norm(target_next64, dim=0)
    before_origin_displacement = torch.linalg.vector_norm(current64 - origin64, dim=0)
    after_origin_displacement = torch.linalg.vector_norm(target_next64 - origin64, dim=0)
    step_displacement = torch.linalg.vector_norm(target_next64 - current64, dim=0)
    would_hit = [bool(item) for item in parent.receipt["clamp_hit"]]
    reference_ratio = [float(item) for item in parent.receipt["clamp_ratio"]]
    if len(would_hit) != current_target.shape[1] or len(reference_ratio) != len(would_hit):
        raise ODEBFContractError("P2R4 frozen clamp observation geometry differs")

    receipt: dict[str, Any] = {
        **dict(parent.receipt),
        "schema": "ode-edit-s05-p2r4-clamp-off-target-microstep/v1",
        "instruction_id": P2R4_INSTRUCTION_ID,
        "method_id": P2R4_METHOD_ID,
        "parent_p2r1_receipt_identity_sha256": parent.receipt["identity_sha256"],
        "parent_p2r1_clamped_target_sha256": tensor_sha256(parent.target_next),
        "clamp_policy": P2R4_CLAMP_POLICY,
        "clamp_reference_on_ratio": reference_ratio,
        "clamp_would_hit": would_hit,
        "clamp_would_hit_count": sum(would_hit),
        "clamp_hit": [False for _ in would_hit],
        "clamp_hit_count": 0,
        "clamp_application_count": 0,
        "identity_projection_application_count": 1,
        "clamp_decision_influence_count": 0,
        "adaptive_projection_count": 0,
        "norm_rescale_count": 0,
        "velocity_clip_count": 0,
        "finite_large_norm_cutoff_count": 0,
        "target_before_norm": [float(item) for item in target_before_norm],
        "target_next_norm": [float(item) for item in target_next_norm],
        "target_before_to_origin_norm_ratio": _ratios(target_before_norm, origin_norm),
        "target_next_to_origin_norm_ratio": _ratios(target_next_norm, origin_norm),
        "target_before_origin_displacement_norm": [
            float(item) for item in before_origin_displacement
        ],
        "target_next_origin_displacement_norm": [
            float(item) for item in after_origin_displacement
        ],
        "target_before_origin_displacement_ratio": _ratios(
            before_origin_displacement, origin_norm
        ),
        "target_next_origin_displacement_ratio": _ratios(
            after_origin_displacement, origin_norm
        ),
        "pre_clamp_displacement_norm": [float(item) for item in step_displacement],
        "post_clamp_displacement_norm": [float(item) for item in step_displacement],
        "proposal_finite": True,
        "proposal_max_abs": float(torch.max(torch.abs(proposal64))),
        "proposal_sha256": tensor_sha256(proposal64),
        "target_next_finite": True,
        "target_next_max_abs": float(torch.max(torch.abs(target_next64))),
        "target_next_sha256": tensor_sha256(target_next),
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    return P2R1TargetUpdate(target_next, parent.state_next, parent.field, receipt)


__all__ = [
    "P2R4_CLAMP_POLICY",
    "P2R4_INSTRUCTION_ID",
    "P2R4_METHOD",
    "P2R4_METHOD_ID",
    "p2r4_clamp_off_target_update",
]
