"""P1R42 semantic-primary move/accept/hold target policy.

The module reuses the P1R39 fixed normalized target step and the existing
finite endpoint.  It performs tensor arithmetic only and owns no model,
evaluator, materialization, trust-radius, Adam, or velocity-decay action.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import torch

from .contracts import ODEBFContractError, canonical_hash
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
from .p1r39_normalized_gradient_target import (
    P1R39_NORMALIZATION_EPSILON,
    _full_current_residual_step,
    _summary,
)
from .scalable_batched_model import ScalableObjectiveResult


P1R42_INSTRUCTION_ID = (
    "ODEEDIT-S05-P1R42-P1R39-MOVE-ACCEPT-HOLD-OBJECTIVE-ALIGNMENT-B10X10-V1"
)
P1R42_METHOD_ID = "P1R39-SEMANTIC-PRIMARY-OBJECTIVE-ALIGNED-TARGET-V1"
P1R42_DIRECTION_POLICY = "SEMANTIC_PRIMARY_PROJECTED_PRESERVATION_FIXED_NORMALIZED"
P1R42_HOLD_POLICY = "CURRENT_STATE_SEMANTIC_V_INSTANTANEOUS_NO_CARRY"
P1R42_SEMANTIC_EPSILON = P1R24_NUMERICAL_EPSILON
P1R42_ZERO_GRADIENT_EPSILON = P1R39_NORMALIZATION_EPSILON


@dataclass(frozen=True, slots=True)
class P1R42ControllerState:
    previous_semantic_held: tuple[bool, ...]

    @classmethod
    def zero(cls, target: torch.Tensor) -> "P1R42ControllerState":
        if target.ndim != 2 or target.shape[1] <= 0:
            raise ODEBFContractError("P1R42 target geometry differs")
        return cls(tuple(False for _ in range(target.shape[1])))


@dataclass(frozen=True, slots=True)
class P1R42TargetProposal:
    trial_step: P1R24TargetStep
    active_mask: tuple[bool, ...]
    semantic_held_mask: tuple[bool, ...]
    semantic_zero_gradient_mask: tuple[bool, ...]
    reactivated_mask: tuple[bool, ...]
    active_to_held_mask: tuple[bool, ...]
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class P1R42SelectedTarget:
    target_step: P1R24TargetStep
    selected_endpoint: ScalableObjectiveResult
    next_state: P1R42ControllerState
    receipt: Mapping[str, Any]


def prepare_p1r42_target_proposal(
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    z0: torch.Tensor,
    nll: ScalableObjectiveResult,
    kl: P1R24KLResult,
    lock: P1R24AliasTargetLock,
    state: P1R42ControllerState,
    *,
    alias: str,
    step_index: int,
    shared_speed: float,
) -> P1R42TargetProposal:
    """Build one semantic-primary fixed-normalized proposal without model calls."""

    if alias not in ("llama3-8b-inst", "qwen2.5-7b-inst"):
        raise ODEBFContractError("P1R42 alias differs")
    if nll.target_gradient is None or kl.gradient is None:
        raise ODEBFContractError("P1R42 target gradients are absent")
    if step_index < 0 or step_index >= P1R24_K:
        raise ODEBFContractError("P1R42 target step index differs")
    shape = current_target.shape
    if (
        current_target.ndim != 2
        or current_terminal.shape != shape
        or z0.shape != shape
        or len(nll.per_request_values) != shape[1]
        or len(kl.per_request_values) != shape[1]
        or len(state.previous_semantic_held) != shape[1]
        or not math.isfinite(shared_speed)
        or shared_speed <= 0.0
    ):
        raise ODEBFContractError("P1R42 target state geometry differs")

    current64 = current_target.detach().to(device="cpu", dtype=torch.float64)
    z064 = z0.detach().to(device="cpu", dtype=torch.float64)
    delta_from_origin = current64 - z064
    z0_norm = torch.linalg.vector_norm(z064, dim=0)
    delta_norm = torch.linalg.vector_norm(delta_from_origin, dim=0)
    if bool(torch.any(z0_norm <= 0.0)):
        raise ODEBFContractError("P1R42 target decay origin is degenerate")

    decay_values = lock.decay_factor * delta_norm / torch.square(z0_norm)
    decay_gradient = torch.zeros_like(delta_from_origin)
    nonzero_decay = delta_norm > 0.0
    decay_gradient[:, nonzero_decay] = (
        lock.decay_factor
        * delta_from_origin[:, nonzero_decay]
        / delta_norm[nonzero_decay].unsqueeze(0)
        / torch.square(z0_norm[nonzero_decay]).unsqueeze(0)
        / shape[1]
    )

    nll_gradient = nll.target_gradient.detach().to(device="cpu", dtype=torch.float64)
    kl_gradient = kl.gradient.detach().to(device="cpu", dtype=torch.float64)
    request_demean_factor = shape[1]
    semantic_gradient = nll_gradient * request_demean_factor
    preservation_gradient = (
        lock.kl_factor * kl_gradient + decay_gradient
    ) * request_demean_factor

    semantic_norm = torch.linalg.vector_norm(semantic_gradient, dim=0)
    preservation_norm = torch.linalg.vector_norm(preservation_gradient, dim=0)
    semantic_direction = -semantic_gradient
    preservation_direction = -preservation_gradient
    raw_conflict = torch.sum(semantic_gradient * preservation_direction, dim=0)
    semantic_norm_sq = torch.sum(torch.square(semantic_gradient), dim=0)
    projection_coefficient = torch.clamp(raw_conflict, min=0.0) / (
        semantic_norm_sq + P1R39_NORMALIZATION_EPSILON
    )
    removed_component = semantic_gradient * projection_coefficient.unsqueeze(0)
    safe_preservation = preservation_direction - removed_component
    safe_inner = torch.sum(semantic_gradient * safe_preservation, dim=0)
    removed_norm = torch.linalg.vector_norm(removed_component, dim=0)

    semantic_zero = semantic_norm <= P1R42_ZERO_GRADIENT_EPSILON
    final_direction = semantic_direction + safe_preservation
    final_direction[:, semantic_zero] = 0.0
    final_direction_norm = torch.linalg.vector_norm(final_direction, dim=0)
    final_zero = final_direction_norm <= P1R42_ZERO_GRADIENT_EPSILON

    if bool(torch.any(safe_inner > P1R24_NUMERICAL_EPSILON)):
        raise ODEBFContractError("P1R42 safe preservation semantic certificate differs")
    if bool(
        torch.any(
            (raw_conflict <= 0.0)
            & (removed_norm > P1R24_NUMERICAL_EPSILON)
        )
    ):
        raise ODEBFContractError("P1R42 nonconflicting preservation was removed")

    semantic_v = tuple(float(item) for item in nll.per_request_values)
    phi = tuple(
        float(
            semantic_v[index]
            + lock.kl_factor * kl.per_request_values[index]
            + decay_values[index]
        )
        for index in range(shape[1])
    )
    semantic_held = tuple(value < P1R24_FREEZE_THRESHOLD for value in semantic_v)
    counterfactual_old_held = tuple(
        value < P1R24_FREEZE_THRESHOLD for value in phi
    )
    zero_mask = tuple(bool(item) for item in semantic_zero)
    final_zero_mask = tuple(bool(item) for item in final_zero)
    active = tuple(
        (not semantic_held[index])
        and (not zero_mask[index])
        and (not final_zero_mask[index])
        for index in range(shape[1])
    )
    reactivated = tuple(
        state.previous_semantic_held[index] and not semantic_held[index]
        for index in range(shape[1])
    )
    active_to_held = tuple(
        (not state.previous_semantic_held[index]) and semantic_held[index]
        for index in range(shape[1])
    )
    old_active_but_semantic_held = tuple(
        semantic_held[index] and not counterfactual_old_held[index]
        for index in range(shape[1])
    )

    active_tensor = torch.tensor(active, dtype=torch.bool)
    velocity = torch.zeros_like(current64)
    if bool(torch.any(active_tensor)):
        velocity[:, active_tensor] = (
            shared_speed
            * final_direction[:, active_tensor]
            / (
                final_direction_norm[active_tensor]
                + P1R39_NORMALIZATION_EPSILON
            ).unsqueeze(0)
        )
    nominal_delta = P1R24_H * velocity
    nominal_norm = torch.linalg.vector_norm(nominal_delta, dim=0)
    expected_norm = P1R24_H * shared_speed

    candidate = current64 + nominal_delta
    cumulative_ratio: list[float] = []
    cumulative_hit: list[bool] = []
    for index in range(shape[1]):
        if not active[index]:
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
    post_clamp_norm = torch.linalg.vector_norm(
        target_next.to(dtype=torch.float64) - current64, dim=0
    )
    active_unclamped = [
        index
        for index in range(shape[1])
        if active[index] and not cumulative_hit[index]
    ]
    norm_residual = max(
        (
            abs(float(nominal_norm[index]) - expected_norm)
            for index in active_unclamped
        ),
        default=0.0,
    )
    if norm_residual > P1R24_NUMERICAL_EPSILON:
        raise ODEBFContractError("P1R42 active normalized displacement norm differs")

    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r42-objective-aligned-target-proposal/v1",
        "instruction_id": P1R42_INSTRUCTION_ID,
        "method_id": P1R42_METHOD_ID,
        "target_direction_policy": P1R42_DIRECTION_POLICY,
        "hold_mode": P1R42_HOLD_POLICY,
        "k": step_index,
        "alias": alias,
        "request_count": shape[1],
        "semantic_v_by_request": list(semantic_v),
        "kl_by_request": [float(item) for item in kl.per_request_values],
        "decay_by_request": [float(item) for item in decay_values],
        "old_composite_phi_by_request": list(phi),
        "semantic_held_mask": list(semantic_held),
        "counterfactual_old_composite_held_mask": list(counterfactual_old_held),
        "old_active_but_semantic_held_mask": list(old_active_but_semantic_held),
        "semantic_zero_gradient_mask": list(zero_mask),
        "final_direction_zero_mask": list(final_zero_mask),
        "active_mask": list(active),
        "held_mask": list(semantic_held),
        "satisfied_now": list(semantic_held),
        "prior_semantic_held_mask": list(state.previous_semantic_held),
        "reactivated_mask": list(reactivated),
        "active_to_held_mask": list(active_to_held),
        "active_count": sum(active),
        "held_count": sum(semantic_held),
        "reactivated_count": sum(reactivated),
        "active_to_held_count": sum(active_to_held),
        "old_active_but_semantic_held_count": sum(old_active_but_semantic_held),
        "request_demean_factor": request_demean_factor,
        "semantic_gradient_norm": [float(item) for item in semantic_norm],
        "preservation_gradient_norm": [float(item) for item in preservation_norm],
        "raw_preservation_conflict_inner_product": [
            float(item) for item in raw_conflict
        ],
        "projection_coefficient": [float(item) for item in projection_coefficient],
        "removed_conflicting_component_norm": [
            float(item) for item in removed_norm
        ],
        "safe_preservation_inner_product": [float(item) for item in safe_inner],
        "safe_inner_product_max": float(torch.max(safe_inner)),
        "safe_inner_product_tolerance": P1R24_NUMERICAL_EPSILON,
        "final_direction_norm": [float(item) for item in final_direction_norm],
        "nominal_displacement_norm": [float(item) for item in nominal_norm],
        "pre_clamp_displacement_norm": [float(item) for item in nominal_norm],
        "post_clamp_displacement_norm": [float(item) for item in post_clamp_norm],
        "active_unclamped_expected_displacement_norm": expected_norm,
        "active_unclamped_norm_max_abs_residual": norm_residual,
        "normalization_epsilon": P1R39_NORMALIZATION_EPSILON,
        "normalization_epsilon_source": "P1R39_NORMALIZATION_EPSILON",
        "semantic_zero_gradient_epsilon": P1R42_ZERO_GRADIENT_EPSILON,
        "semantic_hold_threshold": P1R24_FREEZE_THRESHOLD,
        "shared_speed": shared_speed,
        "cumulative_clamp_ratio": cumulative_ratio,
        "cumulative_clamp_hit": cumulative_hit,
        "target_new_nll_current_by_request": list(semantic_v),
        "target_new_nll_current_summary": _summary(semantic_v),
        "semantic_gradient_sha256": tensor_sha256(semantic_gradient),
        "preservation_gradient_sha256": tensor_sha256(preservation_gradient),
        "safe_preservation_direction_sha256": tensor_sha256(safe_preservation),
        "final_direction_sha256": tensor_sha256(final_direction),
        "aggregate_gradient_sha256": tensor_sha256(
            nll_gradient + lock.kl_factor * kl_gradient + decay_gradient
        ),
        "target_trial_count": 1,
        "persistent_mask_input_count": 0,
        "persistent_mask_decision_influence_count": 0,
        "carried_frozen_input_count": 0,
        "carried_hold_decision_influence_count": 0,
        "preservation_only_target_movement_count": 0,
        "request_local_trust_radius_access_count": 0,
        "semantic_velocity_decay_access_count": 0,
        "gamma_application_count": 0,
        "adam_state_access_count": 0,
        "adam_learning_rate_access_count": 0,
        "adam_bias_correction_access_count": 0,
        "request_raw_cap_access_count": 0,
        "per_request_python_model_call_count": 0,
        "added_model_forward_count": 0,
        "added_backward_count": 0,
        "added_materialization_count": 0,
        "global_frobenius_normalization_count": 0,
    }
    return P1R42TargetProposal(
        _full_current_residual_step(
            target_next=target_next,
            current_target=current_target,
            current_terminal=current_terminal,
            nll_gradient=nll_gradient,
            aggregate_gradient=(
                nll_gradient + lock.kl_factor * kl_gradient + decay_gradient
            ),
            held=semantic_held,
            receipt=receipt,
        ),
        active,
        semantic_held,
        zero_mask,
        reactivated,
        active_to_held,
        receipt,
    )


def select_p1r42_target_proposal(
    proposal: P1R42TargetProposal,
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    current_nll: ScalableObjectiveResult,
    trial_endpoint: ScalableObjectiveResult,
    *,
    step_index: int,
) -> P1R42SelectedTarget:
    """Select request columns with the inherited finite semantic endpoint."""

    shape = current_target.shape
    if (
        current_terminal.shape != shape
        or proposal.trial_step.target_next.shape != shape
        or len(current_nll.per_request_values) != shape[1]
        or len(trial_endpoint.per_request_values) != shape[1]
        or current_nll.request_order_sha256 != trial_endpoint.request_order_sha256
        or current_nll.target_span_sha256 != trial_endpoint.target_span_sha256
        or trial_endpoint.target_gradient is not None
        or trial_endpoint.backward_count != 0
        or step_index < 0
        or step_index >= P1R24_K
    ):
        raise ODEBFContractError("P1R42 semantic selection geometry differs")

    current_values = tuple(float(item) for item in current_nll.per_request_values)
    trial_values = tuple(float(item) for item in trial_endpoint.per_request_values)
    semantic_accept = tuple(
        trial <= current + P1R42_SEMANTIC_EPSILON
        for current, trial in zip(current_values, trial_values, strict=True)
    )
    proposal_accept = tuple(
        proposal.active_mask[index] and semantic_accept[index]
        for index in range(shape[1])
    )
    selected_values = tuple(
        trial_values[index] if proposal_accept[index] else current_values[index]
        for index in range(shape[1])
    )
    selected64 = current_target.detach().to(device="cpu", dtype=torch.float64).clone()
    trial64 = proposal.trial_step.target_next.detach().to(
        device="cpu", dtype=torch.float64
    )
    accept_tensor = torch.tensor(proposal_accept, dtype=torch.bool)
    selected64[:, accept_tensor] = trial64[:, accept_tensor]
    selected = selected64.to(dtype=torch.float32).contiguous()

    endpoint_payload = {
        "schema": "ode-edit-s05-p1r42-selected-finite-endpoint/v1",
        "trial_endpoint_sha256": trial_endpoint.identity_sha256,
        "current_objective_sha256": current_nll.identity_sha256,
        "request_order_sha256": current_nll.request_order_sha256,
        "accept_mask": list(proposal_accept),
        "per_request_values": list(selected_values),
        "loss": math.fsum(selected_values) / len(selected_values),
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
    residual_norm = torch.linalg.vector_norm(
        selected.to(dtype=torch.float64)
        - current_terminal.detach().to(device="cpu", dtype=torch.float64),
        dim=0,
    )
    held_residual_norm = [
        float(residual_norm[index])
        if proposal.semantic_held_mask[index]
        else None
        for index in range(shape[1])
    ]
    receipt = {
        **dict(proposal.receipt),
        "schema": "ode-edit-s05-p1r42-objective-aligned-target-selection/v1",
        "semantic_epsilon": P1R42_SEMANTIC_EPSILON,
        "semantic_epsilon_source": "P1R24_NUMERICAL_EPSILON",
        "target_new_nll_trial_by_request": list(trial_values),
        "target_new_nll_selected_by_request": list(selected_values),
        "target_new_nll_trial_summary": _summary(trial_values),
        "target_new_nll_selected_summary": _summary(selected_values),
        "semantic_accept_mask": list(semantic_accept),
        "proposal_accept_mask": list(proposal_accept),
        "accept_count": sum(proposal_accept),
        "reject_count": sum(
            proposal.active_mask[index] and not proposal_accept[index]
            for index in range(shape[1])
        ),
        "held_writer_residual_norm_by_request": held_residual_norm,
        "held_writer_continues_full_residual": True,
        "selected_endpoint": endpoint_payload,
        "selection_added_model_forward_count": 0,
        "selection_added_backward_count": 0,
        "selection_added_materialization_count": 0,
        "per_request_python_model_call_count": 0,
        "current_state_semantic_hold_no_carry": True,
        "persistent_mask_decision_influence_count": 0,
        "carried_frozen_input_count": 0,
    }
    selected_step = _full_current_residual_step(
        target_next=selected,
        current_target=current_target,
        current_terminal=current_terminal,
        nll_gradient=proposal.trial_step.nll_gradient,
        aggregate_gradient=proposal.trial_step.combined_gradient,
        held=proposal.semantic_held_mask,
        receipt=receipt,
    )
    return P1R42SelectedTarget(
        selected_step,
        selected_endpoint,
        P1R42ControllerState(proposal.semantic_held_mask),
        selected_step.receipt,
    )


__all__ = [
    "P1R42ControllerState",
    "P1R42TargetProposal",
    "P1R42SelectedTarget",
    "P1R42_DIRECTION_POLICY",
    "P1R42_HOLD_POLICY",
    "P1R42_INSTRUCTION_ID",
    "P1R42_METHOD_ID",
    "P1R42_SEMANTIC_EPSILON",
    "P1R42_ZERO_GRADIENT_EPSILON",
    "prepare_p1r42_target_proposal",
    "select_p1r42_target_proposal",
]
