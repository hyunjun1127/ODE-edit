"""P2R1 cumulative-RMS semantic-first target flow.

This module is deliberately tensor-only.  It owns no model, evaluator,
writer, router, key/factor, or materialization call.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .p1r24_atomic_strength import (
    P1R24AliasTargetLock,
    P1R24_NUMERICAL_EPSILON,
)


P2R1_INSTRUCTION_ID = (
    "ODEEDIT-S05-P2R1-BASELINE-CALIBRATED-RMS-TANGENT-TARGET-FLOW-V1"
)
P2R1_METHOD_ID = "P2R1-CUMULATIVE-RMS-SEMANTIC-TANGENT-TARGET-ONLY-V1"
P2R1_TARGET_MICROSTEP_COUNT = 24
P2R1_OUTER_STATE_COUNT = 8
P2R1_MICROSTEPS_PER_OUTER_STATE = 3
P2R1_FP32_EPSILON = float(torch.finfo(torch.float32).eps)
P2R1_FP32_TINY = float(torch.finfo(torch.float32).tiny)


@dataclass(frozen=True, slots=True)
class P2R1RMSState:
    running_squared_gradient: torch.Tensor | None
    entry_gradient_rms: tuple[float, ...] | None
    completed_microsteps: int

    @classmethod
    def zero(cls) -> "P2R1RMSState":
        return cls(None, None, 0)


@dataclass(frozen=True, slots=True)
class P2R1TargetUpdate:
    target_next: torch.Tensor
    state_next: P2R1RMSState
    field: torch.Tensor
    receipt: Mapping[str, Any]


def p2r1_preservation_gradient(
    current_target: torch.Tensor,
    target_origin: torch.Tensor,
    kl_gradient: torch.Tensor,
    lock: P1R24AliasTargetLock,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-request KL+decay gradient and raw component telemetry."""

    if (
        current_target.ndim != 2
        or target_origin.shape != current_target.shape
        or kl_gradient.shape != current_target.shape
    ):
        raise ODEBFContractError("P2R1 preservation geometry differs")
    current64 = current_target.detach().to(device="cpu", dtype=torch.float64)
    origin64 = target_origin.detach().to(device="cpu", dtype=torch.float64)
    kl64 = kl_gradient.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(current64).all() and torch.isfinite(kl64).all()):
        raise ODEBFContractError("P2R1 preservation input is nonfinite")
    origin_norm = torch.linalg.vector_norm(origin64, dim=0)
    if bool(torch.any(origin_norm <= 0.0)):
        raise ODEBFContractError("P2R1 decay origin is degenerate")
    displacement = current64 - origin64
    displacement_norm = torch.linalg.vector_norm(displacement, dim=0)
    decay_denominator = torch.square(origin_norm) + P1R24_NUMERICAL_EPSILON
    decay_values = lock.decay_factor * displacement_norm / decay_denominator
    decay_gradient = torch.zeros_like(displacement)
    nonzero = displacement_norm > 0.0
    decay_gradient[:, nonzero] = (
        lock.decay_factor
        * displacement[:, nonzero]
        / displacement_norm[nonzero].unsqueeze(0)
        / decay_denominator[nonzero].unsqueeze(0)
    )
    preservation = lock.kl_factor * kl64 + decay_gradient
    if not bool(torch.isfinite(preservation).all()):
        raise ODEBFContractError("P2R1 preservation gradient is nonfinite")
    return preservation, decay_values, decay_gradient


def p2r1_target_update(
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
    """Apply one exact cumulative-RMS semantic-tangent target microstep."""

    if alias != lock.alias:
        raise ODEBFContractError("P2R1 alias/target lock differs")
    if microstep_index != state.completed_microsteps:
        raise ODEBFContractError("P2R1 target microstep state differs")
    if microstep_index < 0 or microstep_index >= P2R1_TARGET_MICROSTEP_COUNT:
        raise ODEBFContractError("P2R1 target microstep index differs")
    if (
        current_target.ndim != 2
        or target_origin.shape != current_target.shape
        or semantic_gradient.shape != current_target.shape
        or preservation_gradient.shape != current_target.shape
    ):
        raise ODEBFContractError("P2R1 target tensor geometry differs")

    current64 = current_target.detach().to(device="cpu", dtype=torch.float64)
    origin64 = target_origin.detach().to(device="cpu", dtype=torch.float64)
    semantic = semantic_gradient.detach().to(device="cpu", dtype=torch.float64)
    preservation = preservation_gradient.detach().to(
        device="cpu", dtype=torch.float64
    )
    if not bool(
        torch.isfinite(current64).all()
        and torch.isfinite(semantic).all()
        and torch.isfinite(preservation).all()
    ):
        raise ODEBFContractError("P2R1 target field input is nonfinite")

    if state.running_squared_gradient is None:
        if state.entry_gradient_rms is not None or microstep_index != 0:
            raise ODEBFContractError("P2R1 RMS entry state differs")
        previous = torch.zeros_like(semantic)
        entry_rms_tensor = torch.sqrt(torch.mean(torch.square(semantic), dim=0))
        entry_rms = tuple(float(item) for item in entry_rms_tensor)
    else:
        previous = state.running_squared_gradient.detach().to(
            device="cpu", dtype=torch.float64
        )
        if previous.shape != semantic.shape or state.entry_gradient_rms is None:
            raise ODEBFContractError("P2R1 cumulative RMS state differs")
        entry_rms = state.entry_gradient_rms
        entry_rms_tensor = torch.tensor(entry_rms, dtype=torch.float64)

    n = microstep_index
    running = (n * previous + torch.square(semantic)) / float(n + 1)
    epsilon = math.sqrt(P2R1_FP32_EPSILON) * torch.clamp(
        entry_rms_tensor, min=P2R1_FP32_TINY
    )
    diagonal = torch.rsqrt(running + torch.square(epsilon).unsqueeze(0))
    inverse_diagonal = torch.reciprocal(diagonal)
    q = torch.sum(semantic * diagonal * semantic, dim=0)
    semantic_velocity = -diagonal * semantic
    preservation_velocity = -diagonal * preservation

    tangent = torch.zeros_like(preservation_velocity)
    gamma = torch.zeros_like(q)
    tangent_norm = torch.zeros_like(q)
    semantic_nonzero = q > 0.0
    if bool(torch.any(semantic_nonzero)):
        gp = torch.sum(semantic * preservation_velocity, dim=0)
        tangent[:, semantic_nonzero] = (
            preservation_velocity[:, semantic_nonzero]
            - (
                gp[semantic_nonzero] / q[semantic_nonzero]
            ).unsqueeze(0)
            * diagonal[:, semantic_nonzero]
            * semantic[:, semantic_nonzero]
        )
        tangent_norm[semantic_nonzero] = torch.sqrt(
            torch.clamp(
                torch.sum(
                    torch.square(tangent[:, semantic_nonzero])
                    * inverse_diagonal[:, semantic_nonzero],
                    dim=0,
                ),
                min=0.0,
            )
        )
        positive_tangent = tangent_norm > 0.0
        gamma[positive_tangent] = torch.minimum(
            torch.ones_like(q[positive_tangent]),
            torch.sqrt(q[positive_tangent]) / tangent_norm[positive_tangent],
        )
    field = semantic_velocity + gamma.unsqueeze(0) * tangent
    tangent_inner = torch.sum(semantic * tangent, dim=0)
    field_inner = torch.sum(semantic * field, dim=0)
    tangent_residual = torch.abs(tangent_inner)
    semantic_residual = torch.abs(field_inner + q)
    if (
        float(torch.max(tangent_residual)) > P1R24_NUMERICAL_EPSILON
        or float(torch.max(semantic_residual)) > P1R24_NUMERICAL_EPSILON
    ):
        raise ODEBFContractError("P2R1 semantic tangent certificate differs")

    eta_m = 0.1 if alias == "llama3-8b-inst" else 0.5
    pre_clamp = current64 + eta_m * field
    origin_norm = torch.linalg.vector_norm(origin64, dim=0)
    if bool(torch.any(origin_norm <= 0.0)):
        raise ODEBFContractError("P2R1 clamp origin is degenerate")
    candidate = pre_clamp.clone()
    clamp_ratio: list[float] = []
    clamp_hit: list[bool] = []
    for request_index in range(current_target.shape[1]):
        displacement = pre_clamp[:, request_index] - origin64[:, request_index]
        norm = float(torch.linalg.vector_norm(displacement))
        maximum = lock.clamp_factor * float(origin_norm[request_index])
        ratio = 1.0 if norm == 0.0 or norm <= maximum else maximum / norm
        candidate[:, request_index] = (
            origin64[:, request_index] + ratio * displacement
        )
        clamp_ratio.append(ratio)
        clamp_hit.append(ratio < 1.0)
    target_next = candidate.to(dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(target_next).all()):
        raise ODEBFContractError("P2R1 target update is nonfinite")

    actual_displacement = target_next.to(torch.float64) - current64
    semantic_velocity_norm = torch.linalg.vector_norm(semantic_velocity, dim=0)
    field_norm = torch.linalg.vector_norm(field, dim=0)
    cosine_denominator = semantic_velocity_norm * field_norm
    cosine = torch.zeros_like(q)
    nonzero_cosine = cosine_denominator > 0.0
    cosine[nonzero_cosine] = (
        torch.sum(semantic_velocity * field, dim=0)[nonzero_cosine]
        / cosine_denominator[nonzero_cosine]
    )
    next_state = P2R1RMSState(
        running.to(dtype=torch.float64).contiguous(),
        entry_rms,
        microstep_index + 1,
    )
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p2r1-rms-tangent-target-microstep/v1",
        "instruction_id": P2R1_INSTRUCTION_ID,
        "method_id": P2R1_METHOD_ID,
        "alias": alias,
        "microstep_index": microstep_index,
        "outer_state_index": microstep_index // P2R1_MICROSTEPS_PER_OUTER_STATE,
        "within_outer_microstep": microstep_index % P2R1_MICROSTEPS_PER_OUTER_STATE,
        "eta_m": eta_m,
        "request_count": current_target.shape[1],
        "entry_gradient_rms": list(entry_rms),
        "epsilon_by_request": [float(item) for item in epsilon],
        "semantic_gradient_norm": [
            float(item) for item in torch.linalg.vector_norm(semantic, dim=0)
        ],
        "preservation_gradient_norm": [
            float(item) for item in torch.linalg.vector_norm(preservation, dim=0)
        ],
        "q_by_request": [float(item) for item in q],
        "tangent_norm_by_request": [float(item) for item in tangent_norm],
        "gamma_by_request": [float(item) for item in gamma],
        "tangent_inner_by_request": [float(item) for item in tangent_inner],
        "semantic_field_inner_by_request": [float(item) for item in field_inner],
        "semantic_tangent_max_abs_residual": float(torch.max(tangent_residual)),
        "semantic_rate_max_abs_residual": float(torch.max(semantic_residual)),
        "field_cosine_with_semantic_nominal": [float(item) for item in cosine],
        "pre_clamp_displacement_norm": [
            float(item)
            for item in torch.linalg.vector_norm(pre_clamp - current64, dim=0)
        ],
        "post_clamp_displacement_norm": [
            float(item) for item in torch.linalg.vector_norm(actual_displacement, dim=0)
        ],
        "clamp_ratio": clamp_ratio,
        "clamp_hit": clamp_hit,
        "clamp_hit_count": sum(clamp_hit),
        "running_squared_gradient_sha256": tensor_sha256(running),
        "semantic_gradient_sha256": tensor_sha256(semantic),
        "preservation_gradient_sha256": tensor_sha256(preservation),
        "tangent_sha256": tensor_sha256(tangent),
        "field_sha256": tensor_sha256(field),
        "target_before_sha256": tensor_sha256(current64),
        "target_next_sha256": tensor_sha256(target_next),
        "target_update_count": 1,
        "writer_update_count": 0,
        "writer_materialization_count": 0,
        "native_endpoint_access_count": 0,
        "target_h_application_count": 0,
        "weight_h_application_count": 0,
        "retry_count": 0,
        "hold_count": 0,
        "semantic_debt_input_count": 0,
        "hard_p_budget_influence_count": 0,
        "early_stop_count": 0,
        "heldout_controller_access_count": 0,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    return P2R1TargetUpdate(target_next, next_state, field, receipt)


__all__ = [
    "P2R1_INSTRUCTION_ID",
    "P2R1_METHOD_ID",
    "P2R1_OUTER_STATE_COUNT",
    "P2R1RMSState",
    "P2R1_TARGET_MICROSTEP_COUNT",
    "P2R1TargetUpdate",
    "p2r1_preservation_gradient",
    "p2r1_target_update",
]
