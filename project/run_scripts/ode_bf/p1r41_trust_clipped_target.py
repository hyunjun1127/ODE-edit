"""P1R41 request-local trust-clipped composite-gradient target flow.

The component owns tensor/state arithmetic only.  It reuses the frozen P1R39
composite objective, instantaneous hold, finite semantic selector, cumulative
clamp, and full-current-residual writer coordinate without model calls.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import torch

from .contracts import ODEBFContractError, canonical_hash
from .fixed_e8_soft_routing import FIXED_E8_NORMALIZATION_EPSILON
from .functional import tensor_sha256
from .p1r24_atomic_strength import (
    P1R24AliasTargetLock,
    P1R24_FREEZE_THRESHOLD,
    P1R24_H,
    P1R24_K,
    P1R24_NUMERICAL_EPSILON,
    P1R24KLResult,
    P1R24TargetStep,
)
from .p1r39_normalized_gradient_target import _full_current_residual_step, _summary
from .scalable_batched_model import ScalableObjectiveResult


P1R41_INSTRUCTION_ID = "ODEEDIT-S05-P1R41-P1R39-TRUST-CLIPPED-GRADIENT-FLOW-B10X10-V1"
P1R41_METHOD_ID = "P1R39-TRUST-CLIPPED-ENTRY-CALIBRATED-COMPOSITE-GRADIENT-P1R35-V1"
P1R41_DIRECTION_POLICY = "PER_REQUEST_TRUST_CLIPPED_ENTRY_CALIBRATED_COMPOSITE_GRADIENT"
P1R41_NORMALIZATION_EPSILON = FIXED_E8_NORMALIZATION_EPSILON
P1R41_SEMANTIC_EPSILON = P1R24_NUMERICAL_EPSILON
P1R41_RADIUS_MIN_DIVISOR = 16.0
P1R41_RADIUS_MAX_MULTIPLIER = 2.0
P1R41_RADIUS_SHRINK = 0.5
P1R41_RADIUS_EXPAND = 2.0
P1R41_RHO_SHRINK = 0.25
P1R41_RHO_EXPAND = 0.75


@dataclass(frozen=True, slots=True)
class P1R41ControllerState:
    previous_instantaneous_held: tuple[bool, ...]
    entry_gradient_norm: tuple[float, ...]
    trust_radius: tuple[float, ...]
    consecutive_reject_count: tuple[int, ...]

    @classmethod
    def zero(cls, target: torch.Tensor) -> "P1R41ControllerState":
        if target.ndim != 2 or target.shape[1] <= 0:
            raise ODEBFContractError("P1R41 target geometry differs")
        count = target.shape[1]
        return cls(
            tuple(False for _ in range(count)),
            (),
            (),
            tuple(0 for _ in range(count)),
        )


@dataclass(frozen=True, slots=True)
class P1R41TargetProposal:
    trial_step: P1R24TargetStep
    active_mask: tuple[bool, ...]
    held_mask: tuple[bool, ...]
    reactivated_mask: tuple[bool, ...]
    active_to_held_mask: tuple[bool, ...]
    entry_gradient_norm: tuple[float, ...]
    radius_before: tuple[float, ...]
    trust_boundary_hit: tuple[bool, ...]
    pure_request_gradient: torch.Tensor
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class P1R41SelectedTarget:
    target_step: P1R24TargetStep
    selected_endpoint: ScalableObjectiveResult
    next_state: P1R41ControllerState
    receipt: Mapping[str, Any]


def _finite_tuple(values: Sequence[float], *, count: int, name: str) -> tuple[float, ...]:
    output = tuple(float(item) for item in values)
    if len(output) != count or not all(math.isfinite(item) for item in output):
        raise ODEBFContractError(f"P1R41 {name} differs")
    return output


def prepare_p1r41_target_proposal(
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    z0: torch.Tensor,
    nll: ScalableObjectiveResult,
    kl: P1R24KLResult,
    lock: P1R24AliasTargetLock,
    state: P1R41ControllerState,
    *,
    alias: str,
    step_index: int,
    shared_speed: float,
) -> P1R41TargetProposal:
    """Build one entry-calibrated, request-local trust-clipped proposal."""

    if alias not in ("llama3-8b-inst", "qwen2.5-7b-inst"):
        raise ODEBFContractError("P1R41 alias differs")
    if nll.target_gradient is None or kl.gradient is None:
        raise ODEBFContractError("P1R41 target gradients are absent")
    if step_index < 0 or step_index >= P1R24_K:
        raise ODEBFContractError("P1R41 target step index differs")
    shape = current_target.shape
    count = shape[1] if current_target.ndim == 2 else 0
    if (
        count <= 0
        or current_terminal.shape != shape
        or z0.shape != shape
        or len(nll.per_request_values) != count
        or len(kl.per_request_values) != count
        or len(state.previous_instantaneous_held) != count
        or len(state.consecutive_reject_count) != count
        or not math.isfinite(shared_speed)
        or shared_speed <= 0.0
    ):
        raise ODEBFContractError("P1R41 target state geometry differs")

    current64 = current_target.detach().to(device="cpu", dtype=torch.float64)
    terminal64 = current_terminal.detach().to(device="cpu", dtype=torch.float64)
    z064 = z0.detach().to(device="cpu", dtype=torch.float64)
    delta_from_origin = current64 - z064
    z0_norm = torch.linalg.vector_norm(z064, dim=0)
    delta_norm = torch.linalg.vector_norm(delta_from_origin, dim=0)
    if bool(torch.any(z0_norm <= 0.0)):
        raise ODEBFContractError("P1R41 target decay origin is degenerate")
    decay_values = lock.decay_factor * delta_norm / torch.square(z0_norm)
    decay_gradient = torch.zeros_like(delta_from_origin)
    nonzero = delta_norm > 0.0
    decay_gradient[:, nonzero] = (
        lock.decay_factor
        * delta_from_origin[:, nonzero]
        / delta_norm[nonzero].unsqueeze(0)
        / torch.square(z0_norm[nonzero]).unsqueeze(0)
        / count
    )
    nll_gradient = nll.target_gradient.detach().to(device="cpu", dtype=torch.float64)
    kl_gradient = kl.gradient.detach().to(device="cpu", dtype=torch.float64)
    aggregate_gradient = nll_gradient + lock.kl_factor * kl_gradient + decay_gradient
    composite_request_gradient = aggregate_gradient * count
    pure_request_gradient = nll_gradient * count
    composite_norm = torch.linalg.vector_norm(composite_request_gradient, dim=0)
    pure_norm = torch.linalg.vector_norm(pure_request_gradient, dim=0)

    if step_index == 0:
        if state.entry_gradient_norm or state.trust_radius:
            raise ODEBFContractError("P1R41 entry state is not reset")
        entry_norm = tuple(float(value + P1R41_NORMALIZATION_EPSILON) for value in composite_norm)
        nominal_radius = P1R24_H * shared_speed
        radius_before = tuple(nominal_radius for _ in range(count))
    else:
        entry_norm = _finite_tuple(state.entry_gradient_norm, count=count, name="entry gradient state")
        radius_before = _finite_tuple(state.trust_radius, count=count, name="trust radius state")
    radius_min = tuple(P1R24_H * shared_speed / P1R41_RADIUS_MIN_DIVISOR for _ in range(count))
    radius_max = tuple(P1R24_H * shared_speed * P1R41_RADIUS_MAX_MULTIPLIER for _ in range(count))
    if any(
        entry_norm[index] <= 0.0
        or radius_before[index] < radius_min[index] - P1R24_NUMERICAL_EPSILON
        or radius_before[index] > radius_max[index] + P1R24_NUMERICAL_EPSILON
        for index in range(count)
    ):
        raise ODEBFContractError("P1R41 entry/radius bounds differ")

    phi = tuple(
        float(nll.per_request_values[index] + lock.kl_factor * kl.per_request_values[index] + decay_values[index])
        for index in range(count)
    )
    active = tuple(value >= P1R24_FREEZE_THRESHOLD for value in phi)
    held = tuple(not item for item in active)
    reactivated = tuple(state.previous_instantaneous_held[index] and active[index] for index in range(count))
    active_to_held = tuple((not state.previous_instantaneous_held[index]) and held[index] for index in range(count))

    raw_delta = torch.zeros_like(current64)
    post_trust = torch.zeros_like(current64)
    trust_scale: list[float] = []
    boundary_hit: list[bool] = []
    for index in range(count):
        if not active[index]:
            trust_scale.append(0.0)
            boundary_hit.append(False)
            continue
        raw_delta[:, index] = (
            -P1R24_H
            * shared_speed
            * composite_request_gradient[:, index]
            / entry_norm[index]
        )
        raw_norm = float(torch.linalg.vector_norm(raw_delta[:, index]))
        scale = min(1.0, radius_before[index] / (raw_norm + P1R41_NORMALIZATION_EPSILON))
        post_trust[:, index] = scale * raw_delta[:, index]
        trust_scale.append(scale)
        boundary_hit.append(scale < 1.0)

    raw_norm = torch.linalg.vector_norm(raw_delta, dim=0)
    post_trust_norm = torch.linalg.vector_norm(post_trust, dim=0)
    if any(float(post_trust_norm[index]) > radius_before[index] + P1R24_NUMERICAL_EPSILON for index in range(count)):
        raise ODEBFContractError("P1R41 trust radius certificate differs")

    candidate = current64 + post_trust
    cumulative_ratio: list[float] = []
    cumulative_hit: list[bool] = []
    for index in range(count):
        if held[index]:
            candidate[:, index] = current64[:, index]
            cumulative_ratio.append(0.0)
            cumulative_hit.append(False)
            continue
        proposed = candidate[:, index] - z064[:, index]
        norm = float(torch.linalg.vector_norm(proposed))
        maximum = lock.clamp_factor * float(z0_norm[index])
        ratio = 1.0 if norm <= maximum or norm == 0.0 else maximum / norm
        candidate[:, index] = z064[:, index] + ratio * proposed
        cumulative_ratio.append(ratio)
        cumulative_hit.append(ratio < 1.0)

    target_next = candidate.to(dtype=torch.float32).contiguous()
    post_clamp_norm = torch.linalg.vector_norm(target_next.to(dtype=torch.float64) - current64, dim=0)
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r41-trust-clipped-target-proposal/v1",
        "instruction_id": P1R41_INSTRUCTION_ID,
        "method_id": P1R41_METHOD_ID,
        "target_direction_policy": P1R41_DIRECTION_POLICY,
        "k": step_index,
        "alias": alias,
        "request_count": count,
        "phi_by_request": list(phi),
        "active_mask": list(active),
        "held_mask": list(held),
        "satisfied_now": list(held),
        "prior_instantaneous_held_mask": list(state.previous_instantaneous_held),
        "reactivated_mask": list(reactivated),
        "active_to_frozen_mask": list(active_to_held),
        "active_count": sum(active),
        "held_count": sum(held),
        "reactivated_count": sum(reactivated),
        "active_to_frozen_count": sum(active_to_held),
        "hold_mode": "CURRENT_STATE_INSTANTANEOUS_NO_CARRY",
        "persistent_mask_input_count": 0,
        "persistent_mask_decision_influence_count": 0,
        "carried_frozen_input_count": 0,
        "carried_hold_decision_influence_count": 0,
        "entry_composite_gradient_norm": list(entry_norm),
        "current_composite_gradient_norm": [float(item) for item in composite_norm],
        "current_pure_nll_gradient_norm": [float(item) for item in pure_norm],
        "raw_displacement_norm": [float(item) for item in raw_norm],
        "trust_radius_before": list(radius_before),
        "trust_radius_min": list(radius_min),
        "trust_radius_max": list(radius_max),
        "trust_scale": trust_scale,
        "trust_boundary_hit": boundary_hit,
        "post_trust_displacement_norm": [float(item) for item in post_trust_norm],
        "post_clamp_displacement_norm": [float(item) for item in post_clamp_norm],
        "cumulative_clamp_ratio": cumulative_ratio,
        "cumulative_clamp_hit": cumulative_hit,
        "normalization_epsilon": P1R41_NORMALIZATION_EPSILON,
        "normalization_epsilon_source": "FIXED_E8_NORMALIZATION_EPSILON",
        "shared_speed": shared_speed,
        "target_new_nll_current_by_request": list(nll.per_request_values),
        "target_new_nll_current_summary": _summary(nll.per_request_values),
        "combined_gradient_sha256": tensor_sha256(aggregate_gradient),
        "per_request_composite_gradient_sha256": tensor_sha256(composite_request_gradient),
        "per_request_pure_gradient_sha256": tensor_sha256(pure_request_gradient),
        "target_trial_count": 1,
        "radius_same_step_retry_count": 0,
        "radius_applies_next_global_step_only": True,
        "objective_alignment_access_count": 0,
        "semantic_velocity_decay_access_count": 0,
        "adam_state_access_count": 0,
        "adam_learning_rate_access_count": 0,
        "adam_bias_correction_access_count": 0,
        "request_raw_cap_access_count": 0,
        "request_raw_cap_application_count": 0,
        "per_request_python_model_call_count": 0,
        "added_model_forward_count": 0,
        "added_backward_count": 0,
        "added_materialization_count": 0,
        "global_frobenius_normalization_count": 0,
    }
    trial_step = _full_current_residual_step(
        target_next=target_next,
        current_target=current_target,
        current_terminal=current_terminal,
        nll_gradient=nll_gradient,
        aggregate_gradient=aggregate_gradient,
        held=held,
        receipt=receipt,
    )
    return P1R41TargetProposal(
        trial_step,
        active,
        held,
        reactivated,
        active_to_held,
        entry_norm,
        radius_before,
        tuple(boundary_hit),
        pure_request_gradient,
        trial_step.receipt,
    )


def select_p1r41_target_proposal(
    proposal: P1R41TargetProposal,
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    current_nll: ScalableObjectiveResult,
    trial_endpoint: ScalableObjectiveResult,
    previous_state: P1R41ControllerState,
    *,
    step_index: int,
) -> P1R41SelectedTarget:
    """Apply the frozen semantic selector and update radius for the next step."""

    shape = current_target.shape
    count = shape[1] if current_target.ndim == 2 else 0
    if (
        count <= 0
        or current_terminal.shape != shape
        or proposal.trial_step.target_next.shape != shape
        or len(current_nll.per_request_values) != count
        or len(trial_endpoint.per_request_values) != count
        or len(previous_state.consecutive_reject_count) != count
        or current_nll.request_order_sha256 != trial_endpoint.request_order_sha256
        or current_nll.target_span_sha256 != trial_endpoint.target_span_sha256
        or trial_endpoint.target_gradient is not None
        or trial_endpoint.backward_count != 0
    ):
        raise ODEBFContractError("P1R41 semantic selection geometry differs")
    current_values = tuple(float(item) for item in current_nll.per_request_values)
    trial_values = tuple(float(item) for item in trial_endpoint.per_request_values)
    semantic_accept = tuple(
        trial <= current + P1R41_SEMANTIC_EPSILON
        for current, trial in zip(current_values, trial_values, strict=True)
    )
    proposal_accept = tuple(proposal.active_mask[index] and semantic_accept[index] for index in range(count))
    selected_values = tuple(trial_values[index] if proposal_accept[index] else current_values[index] for index in range(count))
    trial_delta = (
        proposal.trial_step.target_next.detach().to(device="cpu", dtype=torch.float64)
        - current_target.detach().to(device="cpu", dtype=torch.float64)
    )
    q_values: list[float] = []
    actual_gain: list[float] = []
    rho_values: list[float | None] = []
    rho_status: list[str] = []
    radius_after: list[float] = []
    update_reason: list[str] = []
    consecutive_after: list[int] = []
    nominal = P1R24_H * float(proposal.receipt["shared_speed"])
    radius_min = nominal / P1R41_RADIUS_MIN_DIVISOR
    radius_max = nominal * P1R41_RADIUS_MAX_MULTIPLIER
    for index in range(count):
        q_value = float(-torch.sum(proposal.pure_request_gradient[:, index] * trial_delta[:, index]))
        gain = current_values[index] - trial_values[index]
        rho = gain / (q_value + P1R41_NORMALIZATION_EPSILON) if q_value > 0.0 else None
        q_values.append(q_value)
        actual_gain.append(gain)
        rho_values.append(rho)
        rho_status.append("FINITE_TRUSTED" if rho is not None and math.isfinite(rho) else "UNTRUSTED_Q_NONPOSITIVE")
        before = proposal.radius_before[index]
        if proposal.held_mask[index]:
            after = before
            reason = "HELD_KEEP"
            consecutive = previous_state.consecutive_reject_count[index]
        elif not proposal_accept[index]:
            after = max(radius_min, P1R41_RADIUS_SHRINK * before)
            reason = "SEMANTIC_REJECT_SHRINK"
            consecutive = previous_state.consecutive_reject_count[index] + 1
        elif rho is not None and math.isfinite(rho) and rho < P1R41_RHO_SHRINK:
            after = max(radius_min, P1R41_RADIUS_SHRINK * before)
            reason = "LOW_TRUST_RATIO_SHRINK"
            consecutive = 0
        elif (
            proposal_accept[index]
            and rho is not None
            and math.isfinite(rho)
            and rho > P1R41_RHO_EXPAND
            and proposal.trust_boundary_hit[index]
            and current_values[index] > P1R24_FREEZE_THRESHOLD
        ):
            after = min(radius_max, P1R41_RADIUS_EXPAND * before)
            reason = "ACCEPT_BOUNDARY_HIGH_RATIO_EXPAND"
            consecutive = 0
        else:
            after = before
            reason = "KEEP"
            consecutive = 0
        radius_after.append(after)
        update_reason.append(reason)
        consecutive_after.append(consecutive)
    if any(value < radius_min - P1R24_NUMERICAL_EPSILON or value > radius_max + P1R24_NUMERICAL_EPSILON for value in radius_after):
        raise ODEBFContractError("P1R41 updated radius bounds differ")

    selected64 = current_target.detach().to(device="cpu", dtype=torch.float64).clone()
    trial64 = proposal.trial_step.target_next.detach().to(device="cpu", dtype=torch.float64)
    accept_tensor = torch.tensor(proposal_accept, dtype=torch.bool)
    selected64[:, accept_tensor] = trial64[:, accept_tensor]
    selected = selected64.to(dtype=torch.float32).contiguous()
    endpoint_payload = {
        "schema": "ode-edit-s05-p1r41-selected-finite-endpoint/v1",
        "trial_endpoint_sha256": trial_endpoint.identity_sha256,
        "current_objective_sha256": current_nll.identity_sha256,
        "request_order_sha256": current_nll.request_order_sha256,
        "accept_mask": list(proposal_accept),
        "per_request_values": list(selected_values),
        "loss": math.fsum(selected_values) / count,
        "selection_model_forward_count": 0,
        "selection_backward_count": 0,
        "trial_forward_reuse_count": trial_endpoint.model_forward_count,
    }
    endpoint_payload["identity_sha256"] = canonical_hash(endpoint_payload)
    selected_endpoint = replace(
        trial_endpoint,
        loss=float(endpoint_payload["loss"]),
        per_request_values=selected_values,
        identity_sha256=endpoint_payload["identity_sha256"],
    )
    receipt = {
        **dict(proposal.receipt),
        "schema": "ode-edit-s05-p1r41-trust-clipped-target-selection/v1",
        "semantic_epsilon": P1R41_SEMANTIC_EPSILON,
        "semantic_epsilon_source": "P1R24_NUMERICAL_EPSILON",
        "target_new_nll_trial_by_request": list(trial_values),
        "target_new_nll_selected_by_request": list(selected_values),
        "target_new_nll_trial_summary": _summary(trial_values),
        "target_new_nll_selected_summary": _summary(selected_values),
        "semantic_accept_mask": list(semantic_accept),
        "proposal_accept_mask": list(proposal_accept),
        "accept_count": sum(proposal_accept),
        "reject_count": sum(proposal.active_mask[index] and not proposal_accept[index] for index in range(count)),
        "predicted_semantic_gain_q": q_values,
        "actual_finite_gain_a": actual_gain,
        "trust_ratio_rho": rho_values,
        "trust_ratio_status": rho_status,
        "trust_radius_after": radius_after,
        "radius_update_reason": update_reason,
        "consecutive_reject_count_before": list(previous_state.consecutive_reject_count),
        "consecutive_reject_count_after": consecutive_after,
        "selected_endpoint": endpoint_payload,
        "selection_added_model_forward_count": 0,
        "selection_added_backward_count": 0,
        "per_request_python_model_call_count": 0,
        "current_state_instantaneous_no_carry": True,
        "persistent_mask_decision_influence_count": 0,
        "carried_frozen_input_count": 0,
        "same_step_retry_count": 0,
        "radius_update_applies_next_global_step_only": True,
    }
    selected_step = _full_current_residual_step(
        target_next=selected,
        current_target=current_target,
        current_terminal=current_terminal,
        nll_gradient=proposal.trial_step.nll_gradient,
        aggregate_gradient=proposal.trial_step.combined_gradient,
        held=proposal.held_mask,
        receipt=receipt,
    )
    next_state = P1R41ControllerState(
        proposal.held_mask,
        proposal.entry_gradient_norm,
        tuple(radius_after),
        tuple(consecutive_after),
    )
    return P1R41SelectedTarget(selected_step, selected_endpoint, next_state, selected_step.receipt)


__all__ = [
    "P1R41ControllerState",
    "P1R41TargetProposal",
    "P1R41SelectedTarget",
    "P1R41_INSTRUCTION_ID",
    "P1R41_METHOD_ID",
    "P1R41_DIRECTION_POLICY",
    "prepare_p1r41_target_proposal",
    "select_p1r41_target_proposal",
]
