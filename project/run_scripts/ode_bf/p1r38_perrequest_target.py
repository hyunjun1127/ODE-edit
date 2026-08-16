"""P1R38 vectorized per-request target controller on the frozen P1R35 writer.

The module owns only target proposal/selection tensor arithmetic.  It does not
build writer fields, route layers, materialize weights, or evaluate held-out
examples.
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
from .scalable_batched_model import ScalableObjectiveResult


P1R38_INSTRUCTION_ID = "ODEEDIT-S05-P1R38-PR-P1R35-PERREQUEST-TARGET-ATOMIC-V1"
P1R38_METHOD_ID = "P1R35-PERREQUEST-REVERSIBLE-HOLD-ADAM-SEMANTIC-SELECTION-V1"
P1R38_BETA1 = 0.9
P1R38_BETA2 = 0.999
P1R38_ADAM_EPSILON = 1.0e-8
P1R38_SEMANTIC_EPSILON = P1R24_NUMERICAL_EPSILON
P1R38_LR_BY_ALIAS = {
    "llama3-8b-inst": 0.1,
    "qwen2.5-7b-inst": 0.5,
}


@dataclass(frozen=True, slots=True)
class P1R38AdamState:
    m: torch.Tensor
    v: torch.Tensor
    t: torch.Tensor
    previous_instantaneous_held: tuple[bool, ...]

    @classmethod
    def zero(cls, target: torch.Tensor) -> "P1R38AdamState":
        if target.ndim != 2 or target.shape[1] <= 0:
            raise ODEBFContractError("P1R38 Adam target geometry differs")
        shape = target.shape
        return cls(
            torch.zeros(shape, dtype=torch.float64, device="cpu"),
            torch.zeros(shape, dtype=torch.float64, device="cpu"),
            torch.zeros(shape[1], dtype=torch.int64, device="cpu"),
            tuple(False for _ in range(shape[1])),
        )


@dataclass(frozen=True, slots=True)
class P1R38TargetProposal:
    trial_step: P1R24TargetStep
    m_candidate: torch.Tensor
    v_candidate: torch.Tensor
    t_candidate: torch.Tensor
    active_mask: tuple[bool, ...]
    held_mask: tuple[bool, ...]
    reactivated_mask: tuple[bool, ...]
    current_phi: tuple[float, ...]
    raw_delta_norm: tuple[float, ...]
    applied_delta_norm: tuple[float, ...]
    cap_hit: tuple[bool, ...]
    cumulative_clamp_hit: tuple[bool, ...]
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class P1R38SelectedTarget:
    target_step: P1R24TargetStep
    selected_endpoint: ScalableObjectiveResult
    next_state: P1R38AdamState
    receipt: Mapping[str, Any]


def _percentile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.detach().to(dtype=torch.float64), q))


def _summary(values: Sequence[float]) -> dict[str, float]:
    tensor = torch.tensor(tuple(float(item) for item in values), dtype=torch.float64)
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise ODEBFContractError("P1R38 request summary differs")
    return {
        "mean": float(torch.mean(tensor)),
        "median": float(torch.median(tensor)),
        "p90": _percentile(tensor, 0.9),
        "worst": float(torch.max(tensor)),
    }


def _validate_state(state: P1R38AdamState, shape: torch.Size) -> None:
    if (
        state.m.shape != shape
        or state.v.shape != shape
        or state.t.shape != (shape[1],)
        or len(state.previous_instantaneous_held) != shape[1]
        or state.m.device.type != "cpu"
        or state.v.device.type != "cpu"
        or state.t.device.type != "cpu"
        or state.m.dtype != torch.float64
        or state.v.dtype != torch.float64
        or state.t.dtype != torch.int64
        or not torch.isfinite(state.m).all()
        or not torch.isfinite(state.v).all()
        or bool(torch.any(state.v < 0.0))
        or bool(torch.any(state.t < 0))
    ):
        raise ODEBFContractError("P1R38 Adam state differs")


def prepare_p1r38_target_proposal(
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    z0: torch.Tensor,
    nll: ScalableObjectiveResult,
    kl: P1R24KLResult,
    lock: P1R24AliasTargetLock,
    state: P1R38AdamState,
    *,
    alias: str,
    step_index: int,
    request_cap_radius: float,
) -> P1R38TargetProposal:
    """Build one vectorized per-request Adam proposal with no model call."""

    if alias not in P1R38_LR_BY_ALIAS:
        raise ODEBFContractError("P1R38 alias differs")
    if nll.target_gradient is None or kl.gradient is None:
        raise ODEBFContractError("P1R38 target gradients are absent")
    if step_index < 0 or step_index >= P1R24_K:
        raise ODEBFContractError("P1R38 target step index differs")
    shape = current_target.shape
    if (
        current_target.ndim != 2
        or current_terminal.shape != shape
        or z0.shape != shape
        or len(nll.per_request_values) != shape[1]
        or len(kl.per_request_values) != shape[1]
    ):
        raise ODEBFContractError("P1R38 target state geometry differs")
    _validate_state(state, shape)
    if not math.isfinite(request_cap_radius) or request_cap_radius <= 0.0:
        raise ODEBFContractError("P1R38 request cap radius differs")

    current64 = current_target.detach().to(device="cpu", dtype=torch.float64)
    terminal64 = current_terminal.detach().to(device="cpu", dtype=torch.float64)
    z064 = z0.detach().to(device="cpu", dtype=torch.float64)
    delta_from_origin = current64 - z064
    z0_norm = torch.linalg.vector_norm(z064, dim=0)
    delta_norm = torch.linalg.vector_norm(delta_from_origin, dim=0)
    if bool(torch.any(z0_norm <= 0.0)):
        raise ODEBFContractError("P1R38 target decay origin is degenerate")
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
    # Each column in the existing batched objective is scaled by the global
    # request mean.  Undo only that common factor before request-local Adam.
    per_request_gradient = aggregate_gradient * shape[1]
    phi = tuple(
        float(
            nll.per_request_values[index]
            + lock.kl_factor * kl.per_request_values[index]
            + decay_values[index]
        )
        for index in range(shape[1])
    )
    active = tuple(value >= P1R24_FREEZE_THRESHOLD for value in phi)
    held = tuple(not item for item in active)
    reactivated = tuple(
        bool(state.previous_instantaneous_held[index] and active[index])
        for index in range(shape[1])
    )
    active_tensor = torch.tensor(active, dtype=torch.bool)

    # Held requests have no proposal and their stale optimizer state is reset.
    prior_m = state.m.clone()
    prior_v = state.v.clone()
    prior_t = state.t.clone()
    prior_m[:, ~active_tensor] = 0.0
    prior_v[:, ~active_tensor] = 0.0
    prior_t[~active_tensor] = 0
    m_candidate = P1R38_BETA1 * prior_m + (1.0 - P1R38_BETA1) * per_request_gradient
    v_candidate = P1R38_BETA2 * prior_v + (1.0 - P1R38_BETA2) * torch.square(
        per_request_gradient
    )
    t_candidate = prior_t + active_tensor.to(dtype=torch.int64)
    m_candidate[:, ~active_tensor] = 0.0
    v_candidate[:, ~active_tensor] = 0.0
    t_candidate[~active_tensor] = 0

    raw_delta = torch.zeros_like(current64)
    if bool(torch.any(active_tensor)):
        t64 = t_candidate[active_tensor].to(dtype=torch.float64)
        m_hat = m_candidate[:, active_tensor] / (
            1.0 - torch.pow(torch.tensor(P1R38_BETA1, dtype=torch.float64), t64)
        ).unsqueeze(0)
        v_hat = v_candidate[:, active_tensor] / (
            1.0 - torch.pow(torch.tensor(P1R38_BETA2, dtype=torch.float64), t64)
        ).unsqueeze(0)
        raw_delta[:, active_tensor] = (
            -P1R38_LR_BY_ALIAS[alias]
            * m_hat
            / (torch.sqrt(v_hat) + P1R38_ADAM_EPSILON)
        )
    raw_norm = torch.linalg.vector_norm(raw_delta, dim=0)
    cap_scale = torch.clamp(
        request_cap_radius / (raw_norm + P1R38_ADAM_EPSILON), max=1.0
    )
    applied_delta = raw_delta * cap_scale.unsqueeze(0)
    applied_delta[:, ~active_tensor] = 0.0
    applied_norm = torch.linalg.vector_norm(applied_delta, dim=0)
    cap_hit = tuple(bool(item < 1.0) for item in cap_scale)

    candidate = current64 + applied_delta
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
    displacement64 = target_next.to(dtype=torch.float64) - current64
    lag64 = current64 - terminal64
    remaining = P1R24_K - step_index
    required64 = displacement64 + lag64 / remaining
    required_model = required64.to(dtype=torch.float32).contiguous()
    write_velocity = (required_model / P1R24_H).contiguous()
    identity_residual = float(
        torch.max(torch.abs(P1R24_H * write_velocity - required_model))
    )
    if identity_residual > P1R24_NUMERICAL_EPSILON:
        raise ODEBFContractError("P1R38 inherited target coordinate differs")
    rho_signed = float(-torch.sum(nll_gradient * required64))
    alpha_target_signed = float(-torch.sum(nll_gradient * displacement64))
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r38-perrequest-target-proposal/v1",
        "instruction_id": P1R38_INSTRUCTION_ID,
        "method_id": P1R38_METHOD_ID,
        "k": step_index,
        "alias": alias,
        "request_count": shape[1],
        "phi_by_request": list(phi),
        "active_mask": list(active),
        "held_mask": list(held),
        "satisfied_now": list(held),
        "prior_instantaneous_held_mask": list(state.previous_instantaneous_held),
        "reactivated_mask": list(reactivated),
        "active_count": sum(active),
        "held_count": sum(held),
        "reactivated_count": sum(reactivated),
        "persistent_mask_input_count": 0,
        "carried_hold_decision_influence_count": 0,
        "hold_mode": "CURRENT_STATE_INSTANTANEOUS_NO_CARRY",
        "adam_beta1": P1R38_BETA1,
        "adam_beta2": P1R38_BETA2,
        "adam_epsilon": P1R38_ADAM_EPSILON,
        "learning_rate": P1R38_LR_BY_ALIAS[alias],
        "adam_t_proposed": [int(item) for item in t_candidate],
        "adam_m_norm": [float(item) for item in torch.linalg.vector_norm(m_candidate, dim=0)],
        "adam_v_norm": [float(item) for item in torch.linalg.vector_norm(v_candidate, dim=0)],
        "raw_delta_norm": [float(item) for item in raw_norm],
        "applied_delta_norm": [float(item) for item in applied_norm],
        "request_cap_radius": request_cap_radius,
        "request_cap_radius_source": "P1R24_H*metric.shared_speed",
        "per_request_cap_hit": list(cap_hit),
        "cumulative_clamp_ratio": cumulative_ratio,
        "cumulative_clamp_hit": cumulative_hit,
        "global_frobenius_normalization_count": 0,
        "per_request_python_model_call_count": 0,
        "added_model_forward_count": 0,
        "added_backward_count": 0,
        "target_new_nll_current_by_request": list(nll.per_request_values),
        "target_new_nll_current_summary": _summary(nll.per_request_values),
        "target_next_sha256": tensor_sha256(target_next),
        "target_displacement_sha256": tensor_sha256(displacement64),
        "required_displacement_sha256": tensor_sha256(required_model),
        "write_velocity_sha256": tensor_sha256(write_velocity),
        "combined_gradient_sha256": tensor_sha256(aggregate_gradient),
        "per_request_gradient_sha256": tensor_sha256(per_request_gradient),
        "nll_gradient_sha256": tensor_sha256(nll_gradient),
        "rho_write_signed": rho_signed,
        "rho_write": max(rho_signed, 0.0),
        "alpha_target_signed": alpha_target_signed,
        "identity_max_abs_residual": identity_residual,
        "physical_h_application_count": 1,
        "second_remaining_division_count": 0,
        "semantic_debt_input_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    step = P1R24TargetStep(
        target_next,
        displacement64.to(dtype=torch.float32).contiguous(),
        required_model,
        write_velocity,
        nll_gradient,
        aggregate_gradient,
        rho_signed,
        max(rho_signed, 0.0),
        alpha_target_signed,
        held,
        payload,
    )
    return P1R38TargetProposal(
        step,
        m_candidate,
        v_candidate,
        t_candidate,
        active,
        held,
        reactivated,
        phi,
        tuple(float(item) for item in raw_norm),
        tuple(float(item) for item in applied_norm),
        cap_hit,
        tuple(cumulative_hit),
        payload,
    )


def select_p1r38_target_proposal(
    proposal: P1R38TargetProposal,
    state: P1R38AdamState,
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    current_nll: ScalableObjectiveResult,
    trial_endpoint: ScalableObjectiveResult,
    *,
    step_index: int,
) -> P1R38SelectedTarget:
    """Select request columns using existing current/trial semantic values."""

    shape = current_target.shape
    _validate_state(state, shape)
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
        raise ODEBFContractError("P1R38 semantic selection geometry differs")
    current_values = tuple(float(item) for item in current_nll.per_request_values)
    trial_values = tuple(float(item) for item in trial_endpoint.per_request_values)
    semantic_accept = tuple(
        trial <= current + P1R38_SEMANTIC_EPSILON
        for current, trial in zip(current_values, trial_values, strict=True)
    )
    proposal_accept = tuple(
        bool(proposal.active_mask[index] and semantic_accept[index])
        for index in range(shape[1])
    )
    selected_values = tuple(
        trial_values[index] if proposal_accept[index] else current_values[index]
        for index in range(shape[1])
    )
    selected64 = current_target.detach().to(device="cpu", dtype=torch.float64).clone()
    trial64 = proposal.trial_step.target_next.detach().to(device="cpu", dtype=torch.float64)
    accept_tensor = torch.tensor(proposal_accept, dtype=torch.bool)
    selected64[:, accept_tensor] = trial64[:, accept_tensor]
    selected = selected64.to(dtype=torch.float32).contiguous()

    next_m = torch.zeros_like(state.m)
    next_v = torch.zeros_like(state.v)
    next_t = torch.zeros_like(state.t)
    next_m[:, accept_tensor] = proposal.m_candidate[:, accept_tensor]
    next_v[:, accept_tensor] = proposal.v_candidate[:, accept_tensor]
    next_t[accept_tensor] = proposal.t_candidate[accept_tensor]
    next_state = P1R38AdamState(
        next_m,
        next_v,
        next_t,
        proposal.held_mask,
    )

    endpoint_payload = {
        "schema": "ode-edit-s05-p1r38-selected-finite-endpoint/v1",
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

    current64 = current_target.detach().to(device="cpu", dtype=torch.float64)
    terminal64 = current_terminal.detach().to(device="cpu", dtype=torch.float64)
    displacement64 = selected.to(dtype=torch.float64) - current64
    lag64 = current64 - terminal64
    remaining = P1R24_K - step_index
    required64 = displacement64 + lag64 / remaining
    required_model = required64.to(dtype=torch.float32).contiguous()
    write_velocity = (required_model / P1R24_H).contiguous()
    nll_gradient = proposal.trial_step.nll_gradient.detach().to(
        device="cpu", dtype=torch.float64
    )
    rho_signed = float(-torch.sum(nll_gradient * required64))
    alpha_target_signed = float(-torch.sum(nll_gradient * displacement64))
    receipt: dict[str, Any] = {
        **dict(proposal.receipt),
        "schema": "ode-edit-s05-p1r38-perrequest-target-selection/v1",
        "proposal_receipt_sha256": proposal.receipt["identity_sha256"],
        "semantic_epsilon": P1R38_SEMANTIC_EPSILON,
        "semantic_epsilon_source": "P1R24_NUMERICAL_EPSILON",
        "target_new_nll_trial_by_request": list(trial_values),
        "target_new_nll_selected_by_request": list(selected_values),
        "target_new_nll_trial_summary": _summary(trial_values),
        "target_new_nll_selected_summary": _summary(selected_values),
        "semantic_accept_mask": list(semantic_accept),
        "proposal_accept_mask": list(proposal_accept),
        "accept_count": sum(proposal_accept),
        "reject_count": sum(
            bool(proposal.active_mask[index] and not proposal_accept[index])
            for index in range(shape[1])
        ),
        "held_count": sum(proposal.held_mask),
        "reactivated_count": sum(proposal.reactivated_mask),
        "adam_t_committed": [int(item) for item in next_t],
        "satisfied_or_rejected_momentum_reset_count": shape[1] - sum(proposal_accept),
        "target_next_sha256": tensor_sha256(selected),
        "target_displacement_sha256": tensor_sha256(displacement64),
        "required_displacement_sha256": tensor_sha256(required_model),
        "write_velocity_sha256": tensor_sha256(write_velocity),
        "selected_endpoint": endpoint_payload,
        "selection_added_model_forward_count": 0,
        "selection_added_backward_count": 0,
        "per_request_python_model_call_count": 0,
        "current_state_instantaneous_no_carry": True,
        "persistent_mask_decision_influence_count": 0,
        "rho_write_signed": rho_signed,
        "rho_write": max(rho_signed, 0.0),
        "alpha_target_signed": alpha_target_signed,
    }
    receipt.pop("identity_sha256", None)
    receipt["identity_sha256"] = canonical_hash(receipt)
    selected_step = replace(
        proposal.trial_step,
        target_next=selected,
        target_displacement=displacement64.to(dtype=torch.float32).contiguous(),
        required_displacement=required_model,
        write_velocity=write_velocity,
        rho_write_signed=rho_signed,
        rho_write=max(rho_signed, 0.0),
        alpha_target_signed=alpha_target_signed,
        frozen_mask=proposal.held_mask,
        receipt=receipt,
    )
    return P1R38SelectedTarget(selected_step, selected_endpoint, next_state, receipt)


__all__ = [
    "P1R38AdamState",
    "P1R38TargetProposal",
    "P1R38SelectedTarget",
    "P1R38_INSTRUCTION_ID",
    "P1R38_METHOD_ID",
    "P1R38_SEMANTIC_EPSILON",
    "prepare_p1r38_target_proposal",
    "select_p1r38_target_proposal",
]
