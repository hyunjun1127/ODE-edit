"""P1R43 rho-free semantic-first target flow.

This reusable policy performs tensor-only target proposal/corrector work.  It
owns no model, evaluator, key, factor, routing, or materialization call.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import torch

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .p1r24_atomic_strength import (
    P1R24_H,
    P1R24_K,
    P1R24_NUMERICAL_EPSILON,
    P1R24TargetStep,
)
from .p1r39_normalized_gradient_target import (
    P1R39_NORMALIZATION_EPSILON,
    _full_current_residual_step,
    _summary,
)
from .scalable_batched_model import ScalableObjectiveResult


P1R43_INSTRUCTION_ID = (
    "ODEEDIT-S05-P1R43-RHO-FREE-SEMANTIC-FIRST-STRENGTH-RECOVERY-V1"
)
P1R43_METHOD_ID = "P1R42-RHO-FREE-SEMANTIC-FIRST-FULL-STRENGTH-V1"
P1R43_DIRECTION_POLICY = "ENTRY_CALIBRATED_PURE_SEMANTIC_GRADIENT"
P1R43_CORRECTOR_POLICY = "ONE_SCALAR_ACTUAL_NLL_CORRECTOR"
P1R43_SEMANTIC_EPSILON = P1R24_NUMERICAL_EPSILON


@dataclass(frozen=True, slots=True)
class P1R43ControllerState:
    entry_semantic_gradient_norm: tuple[float, ...] | None

    @classmethod
    def zero(cls, target: torch.Tensor) -> "P1R43ControllerState":
        if target.ndim != 2 or target.shape[1] <= 0:
            raise ODEBFContractError("P1R43 target geometry differs")
        return cls(None)


@dataclass(frozen=True, slots=True)
class P1R43TargetProposal:
    primary_step: P1R24TargetStep
    next_state: P1R43ControllerState
    semantic_gradient: torch.Tensor
    primary_delta: torch.Tensor
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class P1R43RescueProposal:
    rescue_step: P1R24TargetStep | None
    eligible_mask: tuple[bool, ...]
    alpha_by_request: tuple[float | None, ...]
    curvature_by_request: tuple[float | None, ...]
    directional_derivative_by_request: tuple[float, ...]
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class P1R43SelectedTarget:
    target_step: P1R24TargetStep
    selected_endpoint: ScalableObjectiveResult
    next_state: P1R43ControllerState
    receipt: Mapping[str, Any]


def _validate_objective_pair(
    current: ScalableObjectiveResult,
    endpoint: ScalableObjectiveResult,
    request_count: int,
) -> None:
    if (
        len(current.per_request_values) != request_count
        or len(endpoint.per_request_values) != request_count
        or current.request_order_sha256 != endpoint.request_order_sha256
        or current.target_span_sha256 != endpoint.target_span_sha256
        or endpoint.target_gradient is not None
        or endpoint.backward_count != 0
    ):
        raise ODEBFContractError("P1R43 semantic endpoint geometry differs")


def prepare_p1r43_target_proposal(
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    nll: ScalableObjectiveResult,
    state: P1R43ControllerState,
    *,
    alias: str,
    step_index: int,
    shared_speed: float,
) -> P1R43TargetProposal:
    """Build the entry-calibrated pure-semantic primary proposal."""

    if alias not in ("llama3-8b-inst", "qwen2.5-7b-inst"):
        raise ODEBFContractError("P1R43 alias differs")
    if nll.target_gradient is None:
        raise ODEBFContractError("P1R43 pure semantic gradient is absent")
    if step_index < 0 or step_index >= P1R24_K:
        raise ODEBFContractError("P1R43 target step index differs")
    if (
        current_target.ndim != 2
        or current_terminal.shape != current_target.shape
        or len(nll.per_request_values) != current_target.shape[1]
        or not math.isfinite(shared_speed)
        or shared_speed <= 0.0
    ):
        raise ODEBFContractError("P1R43 target state geometry differs")

    current64 = current_target.detach().to(device="cpu", dtype=torch.float64)
    aggregate = nll.target_gradient.detach().to(device="cpu", dtype=torch.float64)
    semantic = aggregate * current_target.shape[1]
    current_norm = torch.linalg.vector_norm(semantic, dim=0)
    if not bool(torch.isfinite(semantic).all()):
        raise ODEBFContractError("P1R43 semantic gradient is nonfinite")
    if state.entry_semantic_gradient_norm is None:
        entry_norm = current_norm + P1R39_NORMALIZATION_EPSILON
        next_state = P1R43ControllerState(
            tuple(float(item) for item in entry_norm)
        )
    else:
        if len(state.entry_semantic_gradient_norm) != current_target.shape[1]:
            raise ODEBFContractError("P1R43 entry gradient state differs")
        entry_norm = torch.tensor(
            state.entry_semantic_gradient_norm, dtype=torch.float64
        )
        next_state = state
    if bool(torch.any(~torch.isfinite(entry_norm))) or bool(torch.any(entry_norm <= 0.0)):
        raise ODEBFContractError("P1R43 entry gradient norm is invalid")

    delta = -P1R24_H * shared_speed * semantic / entry_norm.unsqueeze(0)
    if not bool(torch.isfinite(delta).all()):
        raise ODEBFContractError("P1R43 primary target displacement is nonfinite")
    target_next = (current64 + delta).to(dtype=torch.float32).contiguous()
    actual_delta = target_next.to(dtype=torch.float64) - current64
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r43-rho-free-primary-target/v1",
        "instruction_id": P1R43_INSTRUCTION_ID,
        "method_id": P1R43_METHOD_ID,
        "target_direction_policy": P1R43_DIRECTION_POLICY,
        "corrector_policy": P1R43_CORRECTOR_POLICY,
        "k": step_index,
        "alias": alias,
        "request_count": current_target.shape[1],
        "shared_speed": shared_speed,
        "h": P1R24_H,
        "target_new_nll_current_by_request": [
            float(item) for item in nll.per_request_values
        ],
        "target_new_nll_current_summary": _summary(nll.per_request_values),
        "entry_semantic_gradient_norm_by_request": [
            float(item) for item in entry_norm
        ],
        "current_semantic_gradient_norm_by_request": [
            float(item) for item in current_norm
        ],
        "primary_displacement_norm_by_request": [
            float(item) for item in torch.linalg.vector_norm(actual_delta, dim=0)
        ],
        "entry_norm_calibration_count": 1 if step_index == 0 else 0,
        "entry_norm_frozen_after_k0": True,
        "semantic_gradient_sha256": tensor_sha256(semantic),
        "primary_delta_sha256": tensor_sha256(actual_delta),
        "target_next_sha256": tensor_sha256(target_next),
        "pure_semantic_decision_influence_count": 1,
        "target_kl_decision_influence_count": 0,
        "target_decay_decision_influence_count": 0,
        "target_preservation_decision_influence_count": 0,
        "trust_rho_decision_influence_count": 0,
        "trust_threshold_access_count": 0,
        "radius_state_access_count": 0,
        "radius_update_count": 0,
        "reject_counter_size_influence_count": 0,
        "semantic_hold_threshold_access_count": 0,
        "persistent_freeze_input_count": 0,
        "semantic_debt_input_count": 0,
        "remaining_horizon_decision_influence_count": 0,
        "same_step_retry_count": 0,
        "clamp_decision_influence_count": 0,
        "nonfinite_overflow_clamp_count": 0,
        "added_model_forward_count": 0,
        "added_backward_count": 0,
        "added_key_factor_refresh_count": 0,
        "added_materialization_count": 0,
    }
    step = _full_current_residual_step(
        target_next=target_next,
        current_target=current_target,
        current_terminal=current_terminal,
        nll_gradient=aggregate,
        aggregate_gradient=aggregate,
        held=tuple(False for _ in range(current_target.shape[1])),
        receipt=receipt,
    )
    return P1R43TargetProposal(step, next_state, semantic, actual_delta, step.receipt)


def prepare_p1r43_rescue_proposal(
    proposal: P1R43TargetProposal,
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    current_nll: ScalableObjectiveResult,
    primary_endpoint: ScalableObjectiveResult,
    *,
    step_index: int,
) -> P1R43RescueProposal:
    """Prepare at most one mixed-batch rescue target without a model call."""

    request_count = current_target.shape[1]
    _validate_objective_pair(current_nll, primary_endpoint, request_count)
    current_values = tuple(float(item) for item in current_nll.per_request_values)
    primary_values = tuple(float(item) for item in primary_endpoint.per_request_values)
    directional = torch.sum(
        proposal.semantic_gradient * proposal.primary_delta, dim=0
    )
    eligible: list[bool] = []
    alphas: list[float | None] = []
    curvatures: list[float | None] = []
    rescue64 = current_target.detach().to(device="cpu", dtype=torch.float64).clone()
    for index, (current, primary) in enumerate(
        zip(current_values, primary_values, strict=True)
    ):
        d = float(directional[index])
        c = 2.0 * (primary - current - d)
        primary_worsens = primary > current + P1R43_SEMANTIC_EPSILON
        alpha = (-d / c) if d < 0.0 and math.isfinite(c) and c > 0.0 else None
        valid = bool(
            primary_worsens
            and alpha is not None
            and math.isfinite(alpha)
            and 0.0 < alpha < 1.0
        )
        eligible.append(valid)
        alphas.append(float(alpha) if valid else None)
        curvatures.append(c if math.isfinite(c) else None)
        if valid:
            rescue64[:, index] = (
                current_target[:, index].detach().to(device="cpu", dtype=torch.float64)
                + float(alpha) * proposal.primary_delta[:, index]
            )
    rescue_step = None
    if any(eligible):
        rescue_target = rescue64.to(dtype=torch.float32).contiguous()
        rescue_step = _full_current_residual_step(
            target_next=rescue_target,
            current_target=current_target,
            current_terminal=current_terminal,
            nll_gradient=proposal.primary_step.nll_gradient,
            aggregate_gradient=proposal.primary_step.combined_gradient,
            held=tuple(not item for item in eligible),
            receipt={
                **dict(proposal.receipt),
                "schema": "ode-edit-s05-p1r43-scalar-rescue-target/v1",
                "rescue_target_sha256": tensor_sha256(rescue_target),
            },
        )
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r43-scalar-rescue-plan/v1",
        "instruction_id": P1R43_INSTRUCTION_ID,
        "method_id": P1R43_METHOD_ID,
        "k": step_index,
        "current_nll_by_request": list(current_values),
        "primary_nll_by_request": list(primary_values),
        "directional_derivative_by_request": [float(item) for item in directional],
        "curvature_by_request": curvatures,
        "alpha_rescue_by_request": alphas,
        "rescue_eligible_mask": eligible,
        "rescue_eligible_count": sum(eligible),
        "rescue_forward_required": any(eligible),
        "maximum_rescue_forward_count": 1,
        "rescue_backward_count": 0,
        "rescue_key_factor_refresh_count": 0,
        "rescue_materialization_count": 0,
        "rescue_physical_euler_count": 0,
        "same_state_retry_count": 0,
        "trust_rho_decision_influence_count": 0,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    return P1R43RescueProposal(
        rescue_step,
        tuple(eligible),
        tuple(alphas),
        tuple(curvatures),
        tuple(float(item) for item in directional),
        receipt,
    )


def select_p1r43_target_proposal(
    proposal: P1R43TargetProposal,
    rescue: P1R43RescueProposal,
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    current_nll: ScalableObjectiveResult,
    primary_endpoint: ScalableObjectiveResult,
    rescue_endpoint: ScalableObjectiveResult | None,
    *,
    step_index: int,
) -> P1R43SelectedTarget:
    """Select current/primary/rescue by actual pure semantic NLL."""

    request_count = current_target.shape[1]
    _validate_objective_pair(current_nll, primary_endpoint, request_count)
    if any(rescue.eligible_mask) != (rescue_endpoint is not None):
        raise ODEBFContractError("P1R43 rescue endpoint presence differs")
    if rescue_endpoint is not None:
        _validate_objective_pair(current_nll, rescue_endpoint, request_count)
    current_values = tuple(float(item) for item in current_nll.per_request_values)
    primary_values = tuple(float(item) for item in primary_endpoint.per_request_values)
    rescue_values = (
        tuple(float(item) for item in rescue_endpoint.per_request_values)
        if rescue_endpoint is not None
        else tuple(math.inf for _ in range(request_count))
    )
    current64 = current_target.detach().to(device="cpu", dtype=torch.float64)
    primary64 = proposal.primary_step.target_next.detach().to(
        device="cpu", dtype=torch.float64
    )
    rescue64 = (
        rescue.rescue_step.target_next.detach().to(device="cpu", dtype=torch.float64)
        if rescue.rescue_step is not None
        else current64
    )
    selected64 = current64.clone()
    selection: list[str] = []
    selected_values: list[float] = []
    for index in range(request_count):
        if primary_values[index] <= current_values[index] + P1R43_SEMANTIC_EPSILON:
            selection.append("PRIMARY")
            selected64[:, index] = primary64[:, index]
            selected_values.append(primary_values[index])
        elif (
            rescue.eligible_mask[index]
            and rescue_values[index] < current_values[index]
            and rescue_values[index] <= primary_values[index]
        ):
            selection.append("RESCUE")
            selected64[:, index] = rescue64[:, index]
            selected_values.append(rescue_values[index])
        else:
            selection.append("CURRENT")
            selected_values.append(current_values[index])
    selected = selected64.to(dtype=torch.float32).contiguous()
    held = tuple(item == "CURRENT" for item in selection)
    endpoint_payload = {
        "schema": "ode-edit-s05-p1r43-selected-finite-endpoint/v1",
        "current_objective_sha256": current_nll.identity_sha256,
        "primary_endpoint_sha256": primary_endpoint.identity_sha256,
        "rescue_endpoint_sha256": (
            None if rescue_endpoint is None else rescue_endpoint.identity_sha256
        ),
        "request_order_sha256": current_nll.request_order_sha256,
        "selection_by_request": selection,
        "semantic_held_mask": list(held),
        "target_hold_mask": list(held),
        "per_request_values": selected_values,
        "loss": math.fsum(selected_values) / request_count,
        "primary_forward_reuse_count": primary_endpoint.model_forward_count,
        "rescue_forward_reuse_count": (
            0 if rescue_endpoint is None else rescue_endpoint.model_forward_count
        ),
        "selection_model_forward_count": 0,
        "selection_backward_count": 0,
    }
    endpoint_payload["identity_sha256"] = canonical_hash(endpoint_payload)
    selected_endpoint = replace(
        primary_endpoint,
        loss=float(endpoint_payload["loss"]),
        per_request_values=tuple(selected_values),
        model_forward_count=(
            primary_endpoint.model_forward_count
            + (0 if rescue_endpoint is None else rescue_endpoint.model_forward_count)
        ),
        processed_token_count=(
            primary_endpoint.processed_token_count
            + (0 if rescue_endpoint is None else rescue_endpoint.processed_token_count)
        ),
        padded_token_count=(
            primary_endpoint.padded_token_count
            + (0 if rescue_endpoint is None else rescue_endpoint.padded_token_count)
        ),
        identity_sha256=endpoint_payload["identity_sha256"],
    )
    receipt = {
        **dict(proposal.receipt),
        "schema": "ode-edit-s05-p1r43-rho-free-target-selection/v1",
        "rescue_plan": dict(rescue.receipt),
        "current_nll_by_request": list(current_values),
        "primary_nll_by_request": list(primary_values),
        "rescue_nll_by_request": [
            None if not math.isfinite(item) else item for item in rescue_values
        ],
        "selected_nll_by_request": selected_values,
        "selected_nll_summary": _summary(selected_values),
        "selection_by_request": selection,
        "semantic_held_mask": list(held),
        "target_hold_mask": list(held),
        "primary_accept_count": selection.count("PRIMARY"),
        "rescue_accept_count": selection.count("RESCUE"),
        "current_hold_count": selection.count("CURRENT"),
        "primary_improving_low_rho_accept_is_unconditional": True,
        "trust_rho_observation_only": True,
        "trust_rho_decision_influence_count": 0,
        "radius_state_access_count": 0,
        "same_step_scalar_corrector_forward_count": (
            0 if rescue_endpoint is None else rescue_endpoint.model_forward_count
        ),
        "same_step_scalar_corrector_backward_count": 0,
        "same_step_scalar_corrector_physical_k_count": 0,
        "target_hold_persistent_count": 0,
        "writer_processes_full_residual_when_target_holds": True,
        "selected_endpoint": endpoint_payload,
    }
    selected_step = _full_current_residual_step(
        target_next=selected,
        current_target=current_target,
        current_terminal=current_terminal,
        nll_gradient=proposal.primary_step.nll_gradient,
        aggregate_gradient=proposal.primary_step.combined_gradient,
        held=held,
        receipt=receipt,
    )
    return P1R43SelectedTarget(
        selected_step, selected_endpoint, proposal.next_state, selected_step.receipt
    )


__all__ = [
    "P1R43ControllerState",
    "P1R43_CORRECTOR_POLICY",
    "P1R43_DIRECTION_POLICY",
    "P1R43_INSTRUCTION_ID",
    "P1R43_METHOD_ID",
    "P1R43RescueProposal",
    "P1R43SelectedTarget",
    "P1R43TargetProposal",
    "prepare_p1r43_rescue_proposal",
    "prepare_p1r43_target_proposal",
    "select_p1r43_target_proposal",
]
