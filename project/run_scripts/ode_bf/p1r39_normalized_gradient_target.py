"""P1R39 per-request normalized-composite-gradient target policy.

This module changes only the target proposal direction inherited from P1R38.
It performs tensor arithmetic only and owns no model/evaluator calls.
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
from .scalable_batched_model import ScalableObjectiveResult


P1R39_INSTRUCTION_ID = "ODEEDIT-S05-P1R39-PR-P1R38-PERREQUEST-NORMALIZED-GRADIENT-NEUTRAL-B10X10-V1"
P1R39_METHOD_ID = "P1R38-PERREQUEST-NORMALIZED-COMPOSITE-GRADIENT-P1R35-NEUTRAL-V1"
P1R39_DIRECTION_POLICY = "PER_REQUEST_NORMALIZED_COMPOSITE_GRADIENT"
P1R39_SEMANTIC_EPSILON = P1R24_NUMERICAL_EPSILON
P1R39_NORMALIZATION_EPSILON = FIXED_E8_NORMALIZATION_EPSILON


@dataclass(frozen=True, slots=True)
class P1R39ControllerState:
    previous_instantaneous_held: tuple[bool, ...]

    @classmethod
    def zero(cls, target: torch.Tensor) -> "P1R39ControllerState":
        if target.ndim != 2 or target.shape[1] <= 0:
            raise ODEBFContractError("P1R39 target geometry differs")
        return cls(tuple(False for _ in range(target.shape[1])))


@dataclass(frozen=True, slots=True)
class P1R39TargetProposal:
    trial_step: P1R24TargetStep
    active_mask: tuple[bool, ...]
    held_mask: tuple[bool, ...]
    reactivated_mask: tuple[bool, ...]
    active_to_held_mask: tuple[bool, ...]
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class P1R39SelectedTarget:
    target_step: P1R24TargetStep
    selected_endpoint: ScalableObjectiveResult
    next_state: P1R39ControllerState
    receipt: Mapping[str, Any]


def _summary(values: Sequence[float]) -> dict[str, float]:
    tensor = torch.tensor(tuple(float(item) for item in values), dtype=torch.float64)
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise ODEBFContractError("P1R39 request summary differs")
    return {
        "mean": float(torch.mean(tensor)),
        "median": float(torch.median(tensor)),
        "p90": float(torch.quantile(tensor, 0.9)),
        "worst": float(torch.max(tensor)),
    }


def _full_current_residual_step(
    *,
    target_next: torch.Tensor,
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    nll_gradient: torch.Tensor,
    aggregate_gradient: torch.Tensor,
    held: tuple[bool, ...],
    receipt: Mapping[str, Any],
) -> P1R24TargetStep:
    current64 = current_target.detach().to(device="cpu", dtype=torch.float64)
    terminal64 = current_terminal.detach().to(device="cpu", dtype=torch.float64)
    next64 = target_next.detach().to(device="cpu", dtype=torch.float64)
    displacement64 = next64 - current64
    required64 = next64 - terminal64
    required_model = required64.to(dtype=torch.float32).contiguous()
    write_velocity = (required_model / P1R24_H).contiguous()
    identity_residual = float(torch.max(torch.abs(P1R24_H * write_velocity - required_model)))
    full_residual_residual = float(torch.max(torch.abs(required64 - (next64 - terminal64))))
    if identity_residual > P1R24_NUMERICAL_EPSILON or full_residual_residual != 0.0:
        raise ODEBFContractError("P1R39 full-current-residual identity differs")
    nll64 = nll_gradient.detach().to(device="cpu", dtype=torch.float64)
    rho_signed = float(-torch.sum(nll64 * required64))
    alpha_target_signed = float(-torch.sum(nll64 * displacement64))
    payload = {
        **dict(receipt),
        "target_next_sha256": tensor_sha256(target_next),
        "target_displacement_sha256": tensor_sha256(displacement64),
        "required_displacement_sha256": tensor_sha256(required_model),
        "write_velocity_sha256": tensor_sha256(write_velocity),
        "full_current_residual_identity_max_abs": full_residual_residual,
        "required_model_norm": float(torch.linalg.vector_norm(required64)),
        "target_displacement_norm": float(torch.linalg.vector_norm(displacement64)),
        "z_minus_y_norm": float(torch.linalg.vector_norm(current64 - terminal64)),
        "remaining_horizon_division_count": 0,
        "fresh_component_division_count": 0,
        "lag_component_division_count": 0,
        "semantic_debt_input_count": 0,
        "physical_h_application_count": 1,
        "second_h_application_count": 0,
        "identity_max_abs_residual": identity_residual,
        "rho_write_signed": rho_signed,
        "rho_write": max(rho_signed, 0.0),
        "alpha_target_signed": alpha_target_signed,
    }
    payload.pop("identity_sha256", None)
    payload["identity_sha256"] = canonical_hash(payload)
    return P1R24TargetStep(
        target_next,
        displacement64.to(dtype=torch.float32).contiguous(),
        required_model,
        write_velocity,
        nll64,
        aggregate_gradient.detach().to(device="cpu", dtype=torch.float64),
        rho_signed,
        max(rho_signed, 0.0),
        alpha_target_signed,
        held,
        payload,
    )


def prepare_p1r39_target_proposal(
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    z0: torch.Tensor,
    nll: ScalableObjectiveResult,
    kl: P1R24KLResult,
    lock: P1R24AliasTargetLock,
    state: P1R39ControllerState,
    *,
    alias: str,
    step_index: int,
    shared_speed: float,
) -> P1R39TargetProposal:
    """Build one vectorized normalized-gradient proposal without model calls."""

    if alias not in ("llama3-8b-inst", "qwen2.5-7b-inst"):
        raise ODEBFContractError("P1R39 alias differs")
    if nll.target_gradient is None or kl.gradient is None:
        raise ODEBFContractError("P1R39 target gradients are absent")
    if step_index < 0 or step_index >= P1R24_K:
        raise ODEBFContractError("P1R39 target step index differs")
    shape = current_target.shape
    if (
        current_target.ndim != 2
        or current_terminal.shape != shape
        or z0.shape != shape
        or len(nll.per_request_values) != shape[1]
        or len(kl.per_request_values) != shape[1]
        or len(state.previous_instantaneous_held) != shape[1]
        or not math.isfinite(shared_speed)
        or shared_speed <= 0.0
    ):
        raise ODEBFContractError("P1R39 target state geometry differs")

    current64 = current_target.detach().to(device="cpu", dtype=torch.float64)
    terminal64 = current_terminal.detach().to(device="cpu", dtype=torch.float64)
    z064 = z0.detach().to(device="cpu", dtype=torch.float64)
    delta_from_origin = current64 - z064
    z0_norm = torch.linalg.vector_norm(z064, dim=0)
    delta_norm = torch.linalg.vector_norm(delta_from_origin, dim=0)
    if bool(torch.any(z0_norm <= 0.0)):
        raise ODEBFContractError("P1R39 target decay origin is degenerate")
    decay_values = lock.decay_factor * delta_norm / torch.square(z0_norm)
    decay_gradient = torch.zeros_like(delta_from_origin)
    nonzero = delta_norm > 0.0
    decay_gradient[:, nonzero] = (
        lock.decay_factor
        * delta_from_origin[:, nonzero]
        / delta_norm[nonzero].unsqueeze(0)
        / torch.square(z0_norm[nonzero]).unsqueeze(0)
        / shape[1]
    )
    nll_gradient = nll.target_gradient.detach().to(device="cpu", dtype=torch.float64)
    kl_gradient = kl.gradient.detach().to(device="cpu", dtype=torch.float64)
    aggregate_gradient = nll_gradient + lock.kl_factor * kl_gradient + decay_gradient
    per_request_gradient = aggregate_gradient * shape[1]
    gradient_norm = torch.linalg.vector_norm(per_request_gradient, dim=0)
    phi = tuple(
        float(nll.per_request_values[index] + lock.kl_factor * kl.per_request_values[index] + decay_values[index])
        for index in range(shape[1])
    )
    active = tuple(value >= P1R24_FREEZE_THRESHOLD for value in phi)
    held = tuple(not item for item in active)
    reactivated = tuple(state.previous_instantaneous_held[i] and active[i] for i in range(shape[1]))
    active_to_held = tuple((not state.previous_instantaneous_held[i]) and held[i] for i in range(shape[1]))
    active_tensor = torch.tensor(active, dtype=torch.bool)

    velocity = torch.zeros_like(current64)
    if bool(torch.any(active_tensor)):
        velocity[:, active_tensor] = (
            -shared_speed
            * per_request_gradient[:, active_tensor]
            / (gradient_norm[active_tensor] + P1R39_NORMALIZATION_EPSILON).unsqueeze(0)
        )
    nominal_delta = P1R24_H * velocity
    nominal_norm = torch.linalg.vector_norm(nominal_delta, dim=0)
    expected_norm = P1R24_H * shared_speed

    candidate = current64 + nominal_delta
    cumulative_ratio: list[float] = []
    cumulative_hit: list[bool] = []
    for index in range(shape[1]):
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
    active_unclamped = [i for i in range(shape[1]) if active[i] and not cumulative_hit[i]]
    norm_residual = max((abs(float(nominal_norm[i]) - expected_norm) for i in active_unclamped), default=0.0)
    if norm_residual > P1R24_NUMERICAL_EPSILON:
        raise ODEBFContractError("P1R39 active normalized displacement norm differs")
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r39-normalized-gradient-target-proposal/v1",
        "instruction_id": P1R39_INSTRUCTION_ID,
        "method_id": P1R39_METHOD_ID,
        "target_direction_policy": P1R39_DIRECTION_POLICY,
        "k": step_index,
        "alias": alias,
        "request_count": shape[1],
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
        "normalized_composite_gradient_norm": [float(item) for item in gradient_norm],
        "nominal_displacement_norm": [float(item) for item in nominal_norm],
        "pre_clamp_displacement_norm": [float(item) for item in nominal_norm],
        "post_clamp_displacement_norm": [float(item) for item in post_clamp_norm],
        "active_unclamped_expected_displacement_norm": expected_norm,
        "active_unclamped_norm_max_abs_residual": norm_residual,
        "normalization_epsilon": P1R39_NORMALIZATION_EPSILON,
        "normalization_epsilon_source": "FIXED_E8_NORMALIZATION_EPSILON",
        "shared_speed": shared_speed,
        "cumulative_clamp_ratio": cumulative_ratio,
        "cumulative_clamp_hit": cumulative_hit,
        "target_new_nll_current_by_request": list(nll.per_request_values),
        "target_new_nll_current_summary": _summary(nll.per_request_values),
        "combined_gradient_sha256": tensor_sha256(aggregate_gradient),
        "per_request_gradient_sha256": tensor_sha256(per_request_gradient),
        "nll_gradient_sha256": tensor_sha256(nll_gradient),
        "semantic_velocity_decay_access_count": 0,
        "adam_state_access_count": 0,
        "adam_learning_rate_access_count": 0,
        "adam_bias_correction_access_count": 0,
        "request_raw_cap_access_count": 0,
        "request_raw_cap_application_count": 0,
        "per_request_python_model_call_count": 0,
        "added_model_forward_count": 0,
        "added_backward_count": 0,
        "global_frobenius_normalization_count": 0,
    }
    return P1R39TargetProposal(
        _full_current_residual_step(
            target_next=target_next,
            current_target=current_target,
            current_terminal=current_terminal,
            nll_gradient=nll_gradient,
            aggregate_gradient=aggregate_gradient,
            held=held,
            receipt=receipt,
        ),
        active,
        held,
        reactivated,
        active_to_held,
        receipt,
    )


def select_p1r39_target_proposal(
    proposal: P1R39TargetProposal,
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    current_nll: ScalableObjectiveResult,
    trial_endpoint: ScalableObjectiveResult,
    *,
    step_index: int,
) -> P1R39SelectedTarget:
    """Select request columns using the inherited existing endpoint forward."""

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
    ):
        raise ODEBFContractError("P1R39 semantic selection geometry differs")
    current_values = tuple(float(item) for item in current_nll.per_request_values)
    trial_values = tuple(float(item) for item in trial_endpoint.per_request_values)
    semantic_accept = tuple(trial <= current + P1R39_SEMANTIC_EPSILON for current, trial in zip(current_values, trial_values, strict=True))
    proposal_accept = tuple(proposal.active_mask[i] and semantic_accept[i] for i in range(shape[1]))
    selected_values = tuple(trial_values[i] if proposal_accept[i] else current_values[i] for i in range(shape[1]))
    selected64 = current_target.detach().to(device="cpu", dtype=torch.float64).clone()
    trial64 = proposal.trial_step.target_next.detach().to(device="cpu", dtype=torch.float64)
    accept_tensor = torch.tensor(proposal_accept, dtype=torch.bool)
    selected64[:, accept_tensor] = trial64[:, accept_tensor]
    selected = selected64.to(dtype=torch.float32).contiguous()

    endpoint_payload = {
        "schema": "ode-edit-s05-p1r39-selected-finite-endpoint/v1",
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
    selected_endpoint = replace(trial_endpoint, loss=float(endpoint_payload["loss"]), per_request_values=selected_values, identity_sha256=endpoint_payload["identity_sha256"])
    receipt = {
        **dict(proposal.receipt),
        "schema": "ode-edit-s05-p1r39-normalized-gradient-target-selection/v1",
        "semantic_epsilon": P1R39_SEMANTIC_EPSILON,
        "semantic_epsilon_source": "P1R24_NUMERICAL_EPSILON",
        "target_new_nll_trial_by_request": list(trial_values),
        "target_new_nll_selected_by_request": list(selected_values),
        "target_new_nll_trial_summary": _summary(trial_values),
        "target_new_nll_selected_summary": _summary(selected_values),
        "semantic_accept_mask": list(semantic_accept),
        "proposal_accept_mask": list(proposal_accept),
        "accept_count": sum(proposal_accept),
        "reject_count": sum(proposal.active_mask[i] and not proposal_accept[i] for i in range(shape[1])),
        "selected_endpoint": endpoint_payload,
        "selection_added_model_forward_count": 0,
        "selection_added_backward_count": 0,
        "per_request_python_model_call_count": 0,
        "current_state_instantaneous_no_carry": True,
        "persistent_mask_decision_influence_count": 0,
        "carried_frozen_input_count": 0,
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
    return P1R39SelectedTarget(
        selected_step,
        selected_endpoint,
        P1R39ControllerState(proposal.held_mask),
        selected_step.receipt,
    )


__all__ = [
    "P1R39ControllerState",
    "P1R39TargetProposal",
    "P1R39SelectedTarget",
    "P1R39_INSTRUCTION_ID",
    "P1R39_METHOD_ID",
    "P1R39_DIRECTION_POLICY",
    "prepare_p1r39_target_proposal",
    "select_p1r39_target_proposal",
]
