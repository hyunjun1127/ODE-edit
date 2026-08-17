"""P1R51 request-wise semantic allocation over the frozen P1R43 field.

The policy is tensor-only.  It preserves the batch total nominal P1R43
velocity energy and redistributes that energy across requests in proportion
to the current request-wise target-new NLL.  P1R43 owns the unchanged
primary/rescue/current endpoint selection and full-current-residual writer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping

import torch

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .p1r24_atomic_strength import (
    P1R24_H,
    P1R24_K,
    P1R24_NUMERICAL_EPSILON,
)
from .p1r39_normalized_gradient_target import (
    P1R39_NORMALIZATION_EPSILON,
    _full_current_residual_step,
    _summary,
)
from .p1r43_rho_free_target import (
    P1R43RescueProposal,
    P1R43SelectedTarget,
    prepare_p1r43_rescue_proposal,
    select_p1r43_target_proposal,
)
from .scalable_batched_model import ScalableObjectiveResult


P1R51_INSTRUCTION_ID = (
    "ODEEDIT-S05-P1R51-P1R43-REQUESTWISE-SEMANTIC-ALLOCATION-A1-V1"
)
P1R51_METHOD_ID = "P1R43-RSA-A1"
P1R51_DIRECTION_POLICY = "REQUESTWISE_NLL_PROPORTIONAL_ENERGY_ALLOCATION"
P1R51_NUMERICAL_EPSILON = P1R39_NORMALIZATION_EPSILON


@dataclass(frozen=True, slots=True)
class P1R51ControllerState:
    entry_semantic_gradient_norm: tuple[float, ...] | None
    cumulative_accepted_activation_path: tuple[float, ...]

    @classmethod
    def zero(cls, target: torch.Tensor) -> "P1R51ControllerState":
        if target.ndim != 2 or target.shape[1] <= 0:
            raise ODEBFContractError("P1R51 target geometry differs")
        return cls(None, tuple(0.0 for _ in range(target.shape[1])))


@dataclass(frozen=True, slots=True)
class P1R51TargetProposal:
    primary_step: Any
    next_state: P1R51ControllerState
    semantic_gradient: torch.Tensor
    primary_delta: torch.Tensor
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class P1R51SelectedTarget:
    target_step: Any
    selected_endpoint: ScalableObjectiveResult
    next_state: P1R51ControllerState
    receipt: Mapping[str, Any]


def _entropy(shares: torch.Tensor) -> float:
    positive = shares[shares > 0.0]
    return 0.0 if positive.numel() == 0 else float(-torch.sum(positive * torch.log(positive)))


def prepare_p1r51_target_proposal(
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    nll: ScalableObjectiveResult,
    state: P1R51ControllerState,
    *,
    alias: str,
    step_index: int,
    shared_speed: float,
) -> P1R51TargetProposal:
    """Build the allocation-only RSA primary proposal without model calls."""

    if alias not in ("llama3-8b-inst", "qwen2.5-7b-inst"):
        raise ODEBFContractError("P1R51 alias differs")
    if nll.target_gradient is None:
        raise ODEBFContractError("P1R51 pure semantic gradient is absent")
    if step_index < 0 or step_index >= P1R24_K:
        raise ODEBFContractError("P1R51 target step index differs")
    request_count = current_target.shape[1] if current_target.ndim == 2 else 0
    if (
        request_count <= 0
        or current_terminal.shape != current_target.shape
        or len(nll.per_request_values) != request_count
        or len(state.cumulative_accepted_activation_path) != request_count
        or not math.isfinite(shared_speed)
        or shared_speed <= 0.0
    ):
        raise ODEBFContractError("P1R51 target state geometry differs")

    current64 = current_target.detach().to(device="cpu", dtype=torch.float64)
    aggregate = nll.target_gradient.detach().to(device="cpu", dtype=torch.float64)
    semantic = aggregate * request_count
    current_norm = torch.linalg.vector_norm(semantic, dim=0)
    nll_values = torch.tensor(nll.per_request_values, dtype=torch.float64)
    if not bool(torch.isfinite(semantic).all()):
        raise ODEBFContractError("P1R51 semantic gradient is nonfinite")
    if not bool(torch.isfinite(nll_values).all()) or bool(torch.any(nll_values < 0.0)):
        raise ODEBFContractError("P1R51 request-wise NLL is nonfinite or negative")

    if state.entry_semantic_gradient_norm is None:
        entry_norm = current_norm + P1R39_NORMALIZATION_EPSILON
        next_state = P1R51ControllerState(
            tuple(float(item) for item in entry_norm),
            state.cumulative_accepted_activation_path,
        )
    else:
        if len(state.entry_semantic_gradient_norm) != request_count:
            raise ODEBFContractError("P1R51 entry gradient state differs")
        entry_norm = torch.tensor(
            state.entry_semantic_gradient_norm, dtype=torch.float64
        )
        next_state = state
    if bool(torch.any(~torch.isfinite(entry_norm))) or bool(torch.any(entry_norm <= 0.0)):
        raise ODEBFContractError("P1R51 entry gradient norm is invalid")

    nominal_velocity = -shared_speed * semantic / entry_norm.unsqueeze(0)
    nominal_norm = torch.linalg.vector_norm(nominal_velocity, dim=0)
    nominal_energy = float(torch.sum(torch.square(nominal_norm)))
    if not math.isfinite(nominal_energy):
        raise ODEBFContractError("P1R51 nominal P1R43 energy is nonfinite")

    active = current_norm > P1R51_NUMERICAL_EPSILON
    direction = torch.zeros_like(semantic)
    if bool(torch.any(active)):
        direction[:, active] = -semantic[:, active] / current_norm[active].unsqueeze(0)
    active_nll = torch.where(active, nll_values, torch.zeros_like(nll_values))
    allocation_denominator = float(torch.linalg.vector_norm(active_nll))
    amplitude = torch.zeros_like(nll_values)
    if nominal_energy > 0.0 and bool(torch.any(active)):
        amplitude[active] = (
            math.sqrt(nominal_energy)
            * nll_values[active]
            / (allocation_denominator + P1R51_NUMERICAL_EPSILON)
        )
    velocity = direction * amplitude.unsqueeze(0)
    rsa_energy = float(torch.sum(torch.square(velocity)))
    energy_relative_error = abs(rsa_energy - nominal_energy) / (
        nominal_energy + P1R51_NUMERICAL_EPSILON
    )
    if bool(torch.any(active)) and energy_relative_error > P1R24_NUMERICAL_EPSILON:
        raise ODEBFContractError("P1R51 target velocity energy is not conserved")

    nominal_delta = P1R24_H * velocity
    target_next = (current64 + nominal_delta).to(dtype=torch.float32).contiguous()
    actual_delta = target_next.to(dtype=torch.float64) - current64
    if not bool(torch.isfinite(actual_delta).all()):
        raise ODEBFContractError("P1R51 primary target displacement is nonfinite")

    shares = (
        torch.square(amplitude) / rsa_energy
        if rsa_energy > 0.0
        else torch.zeros_like(amplitude)
    )
    sorted_shares = torch.sort(shares, descending=True).values
    entropy = _entropy(shares)
    status = (
        "ACTIVE"
        if bool(torch.any(active)) and nominal_energy > 0.0
        else "ZERO_NOMINAL_ENERGY"
        if nominal_energy == 0.0
        else "FLAT_GRADIENT"
    )
    parent_delta = P1R24_H * nominal_velocity
    parent_target_next = (current64 + parent_delta).to(dtype=torch.float32).contiguous()
    b1_parent_direction_cosine = None
    if request_count == 1:
        flat_parent = parent_delta.flatten()
        flat_rsa = nominal_delta.flatten()
        denominator = float(torch.linalg.vector_norm(flat_parent) * torch.linalg.vector_norm(flat_rsa))
        b1_parent_direction_cosine = (
            1.0 if denominator == 0.0 else float(torch.dot(flat_parent, flat_rsa) / denominator)
        )
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r51-rsa-a1-primary-target/v1",
        "instruction_id": P1R51_INSTRUCTION_ID,
        "method_id": P1R51_METHOD_ID,
        "target_direction_policy": P1R51_DIRECTION_POLICY,
        "parent_target_policy": "ENTRY_CALIBRATED_PURE_SEMANTIC_GRADIENT",
        "k": step_index,
        "alias": alias,
        "request_count": request_count,
        "shared_speed": shared_speed,
        "h": P1R24_H,
        "allocation_status": status,
        "numerical_epsilon": P1R51_NUMERICAL_EPSILON,
        "numerical_epsilon_source": "P1R39_NORMALIZATION_EPSILON",
        "target_new_nll_current_by_request": [float(item) for item in nll_values],
        "target_new_nll_current_summary": _summary(nll.per_request_values),
        "semantic_gradient_norm_by_request": [float(item) for item in current_norm],
        "entry_semantic_gradient_norm_by_request": [float(item) for item in entry_norm],
        "p1r43_nominal_velocity_norm_by_request": [float(item) for item in nominal_norm],
        "p1r43_nominal_velocity_energy": nominal_energy,
        "rsa_direction_norm_by_request": [
            float(item) for item in torch.linalg.vector_norm(direction, dim=0)
        ],
        "semantic_urgency_by_request": [float(item) for item in nll_values],
        "allocation_amplitude_by_request": [float(item) for item in amplitude],
        "allocation_energy_share_by_request": [float(item) for item in shares],
        "proposed_target_velocity_norm_by_request": [
            float(item) for item in torch.linalg.vector_norm(velocity, dim=0)
        ],
        "proposed_target_displacement_norm_by_request": [
            float(item) for item in torch.linalg.vector_norm(nominal_delta, dim=0)
        ],
        "actual_target_displacement_norm_by_request": [
            float(item) for item in torch.linalg.vector_norm(actual_delta, dim=0)
        ],
        "rsa_total_velocity_energy": rsa_energy,
        "energy_relative_error": energy_relative_error,
        "active_gradient_mask": [bool(item) for item in active],
        "flat_gradient_mask": [not bool(item) for item in active],
        "active_gradient_count": int(torch.sum(active)),
        "flat_gradient_count": int(request_count - torch.sum(active)),
        "top1_allocation_share": float(sorted_shares[0]),
        "top3_allocation_share": float(torch.sum(sorted_shares[: min(3, request_count)])),
        "allocation_entropy": entropy,
        "effective_request_support": math.exp(entropy),
        "entry_norm_calibration_count": 1 if step_index == 0 else 0,
        "entry_norm_frozen_after_k0": True,
        "cumulative_accepted_activation_path_before_by_request": list(
            state.cumulative_accepted_activation_path
        ),
        "semantic_gradient_sha256": tensor_sha256(semantic),
        "p1r43_nominal_velocity_sha256": tensor_sha256(nominal_velocity),
        "rsa_velocity_sha256": tensor_sha256(velocity),
        "primary_delta_sha256": tensor_sha256(actual_delta),
        "target_next_sha256": tensor_sha256(target_next),
        "b1_parent_direction_cosine": b1_parent_direction_cosine,
        "b1_parent_velocity_norm_abs_residual": (
            abs(float(nominal_norm[0]) - float(torch.linalg.vector_norm(velocity[:, 0])))
            if request_count == 1
            else None
        ),
        "b1_parent_euler_delta_norm_abs_residual": (
            abs(float(torch.linalg.vector_norm(parent_delta)) - float(torch.linalg.vector_norm(nominal_delta)))
            if request_count == 1
            else None
        ),
        "b1_parent_target_endpoint_max_abs_residual": (
            float(torch.max(torch.abs(parent_target_next - target_next)))
            if request_count == 1
            else None
        ),
        "b1_parent_target_endpoint_hash_equal": (
            tensor_sha256(parent_target_next) == tensor_sha256(target_next)
            if request_count == 1
            else None
        ),
        "b1_parent_actual_target_nll_endpoint_reuse_allowed": request_count == 1,
        "b1_parent_selection_outcome_equal_if_endpoint_hash_equal": request_count == 1,
        "allocation_only_decision_influence_count": 1,
        "total_target_velocity_energy_increase_count": 0,
        "target_kl_decision_influence_count": 0,
        "target_decay_decision_influence_count": 0,
        "target_preservation_decision_influence_count": 0,
        "target_hit_threshold_access_count": 0,
        "persistent_freeze_input_count": 0,
        "semantic_debt_input_count": 0,
        "remaining_horizon_decision_influence_count": 0,
        "same_step_retry_count": 0,
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
        held=tuple(False for _ in range(request_count)),
        receipt=receipt,
    )
    return P1R51TargetProposal(step, next_state, semantic, actual_delta, step.receipt)


def prepare_p1r51_rescue_proposal(
    proposal: P1R51TargetProposal,
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    current_nll: ScalableObjectiveResult,
    primary_endpoint: ScalableObjectiveResult,
    *,
    step_index: int,
) -> P1R43RescueProposal:
    """Apply the unchanged P1R43 scalar rescue plan to an RSA proposal."""

    parent = prepare_p1r43_rescue_proposal(
        proposal,
        current_target,
        current_terminal,
        current_nll,
        primary_endpoint,
        step_index=step_index,
    )
    receipt = {
        **dict(parent.receipt),
        "schema": "ode-edit-s05-p1r51-rsa-a1-scalar-rescue-plan/v1",
        "instruction_id": P1R51_INSTRUCTION_ID,
        "method_id": P1R51_METHOD_ID,
        "parent_corrector_byte_semantics": "P1R43_ONE_SCALAR_ACTUAL_NLL_CORRECTOR",
    }
    receipt.pop("identity_sha256", None)
    receipt["identity_sha256"] = canonical_hash(receipt)
    rescue_step = parent.rescue_step
    if rescue_step is not None:
        step_receipt = {
            **dict(rescue_step.receipt),
            "schema": "ode-edit-s05-p1r51-rsa-a1-scalar-rescue-target/v1",
            "instruction_id": P1R51_INSTRUCTION_ID,
            "method_id": P1R51_METHOD_ID,
        }
        step_receipt.pop("identity_sha256", None)
        step_receipt["identity_sha256"] = canonical_hash(step_receipt)
        rescue_step = replace(rescue_step, receipt=step_receipt)
    return P1R43RescueProposal(
        rescue_step,
        parent.eligible_mask,
        parent.alpha_by_request,
        parent.curvature_by_request,
        parent.directional_derivative_by_request,
        receipt,
    )


def select_p1r51_target_proposal(
    proposal: P1R51TargetProposal,
    rescue: P1R43RescueProposal,
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    current_nll: ScalableObjectiveResult,
    primary_endpoint: ScalableObjectiveResult,
    rescue_endpoint: ScalableObjectiveResult | None,
    *,
    step_index: int,
) -> P1R51SelectedTarget:
    """Reuse P1R43 selection and append accepted-path allocation telemetry."""

    selected: P1R43SelectedTarget = select_p1r43_target_proposal(
        proposal,
        rescue,
        current_target,
        current_terminal,
        current_nll,
        primary_endpoint,
        rescue_endpoint,
        step_index=step_index,
    )
    selected_delta = (
        selected.target_step.target_next.detach().to(device="cpu", dtype=torch.float64)
        - current_target.detach().to(device="cpu", dtype=torch.float64)
    )
    accepted_norm = torch.linalg.vector_norm(selected_delta, dim=0)
    cumulative = tuple(
        before + float(increment)
        for before, increment in zip(
            proposal.next_state.cumulative_accepted_activation_path,
            accepted_norm,
            strict=True,
        )
    )
    current_values = tuple(float(item) for item in current_nll.per_request_values)
    selected_values = tuple(
        float(item) for item in selected.selected_endpoint.per_request_values
    )
    receipt = {
        **dict(selected.receipt),
        "schema": "ode-edit-s05-p1r51-rsa-a1-target-selection/v1",
        "instruction_id": P1R51_INSTRUCTION_ID,
        "method_id": P1R51_METHOD_ID,
        "parent_selection_byte_semantics": "P1R43_PRIMARY_RESCUE_CURRENT",
        "accepted_target_nll_improvement_by_request": [
            current - endpoint
            for current, endpoint in zip(current_values, selected_values, strict=True)
        ],
        "accepted_activation_path_increment_by_request": [
            float(item) for item in accepted_norm
        ],
        "cumulative_accepted_activation_path_by_request": list(cumulative),
    }
    receipt.pop("identity_sha256", None)
    target_step = _full_current_residual_step(
        target_next=selected.target_step.target_next,
        current_target=current_target,
        current_terminal=current_terminal,
        nll_gradient=proposal.primary_step.nll_gradient,
        aggregate_gradient=proposal.primary_step.combined_gradient,
        held=selected.target_step.frozen_mask,
        receipt=receipt,
    )
    next_state = P1R51ControllerState(
        proposal.next_state.entry_semantic_gradient_norm,
        cumulative,
    )
    return P1R51SelectedTarget(
        target_step,
        selected.selected_endpoint,
        next_state,
        target_step.receipt,
    )


__all__ = [
    "P1R51ControllerState",
    "P1R51_DIRECTION_POLICY",
    "P1R51_INSTRUCTION_ID",
    "P1R51_METHOD_ID",
    "P1R51_NUMERICAL_EPSILON",
    "P1R51SelectedTarget",
    "P1R51TargetProposal",
    "prepare_p1r51_rescue_proposal",
    "prepare_p1r51_target_proposal",
    "select_p1r51_target_proposal",
]
