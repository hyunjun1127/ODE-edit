"""P1R52 R42-safe KDC direction in the frozen P1R51 RSA amplitude path.

This module is tensor-only.  It consumes the already-computed P1R51 target
NLL and P1R24 KL results, adds the source-bound AlphaEdit decay, performs the
one-sided semantic-safe preservation projection, and applies the origin
clamp before the FP32 endpoint cast.  P1R43 continues to own the unchanged
primary/rescue/current selection and the full-current-residual writer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping

import torch

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .p1r24_atomic_strength import (
    P1R24AliasTargetLock,
    P1R24_H,
    P1R24_K,
    P1R24_NUMERICAL_EPSILON,
    P1R24KLResult,
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
from .p1r51_requestwise_semantic_allocation import P1R51ControllerState
from .scalable_batched_model import ScalableObjectiveResult


P1R52_INSTRUCTION_ID = "ODEEDIT-S05-P1R52-RSA-R42SAFEKDC-M1-V1"
P1R52_METHOD_ID = "P1R52-RSA-R42SAFEKDC-M1"
P1R52_DIRECTION_POLICY = "P1R51_RSA_R42_ONE_SIDED_SAFE_KL_DECAY_ORIGIN_CLAMP"
P1R52_NUMERICAL_EPSILON = P1R39_NORMALIZATION_EPSILON
P1R52_REPAIR_REVISION = "R1"
P1R52_REPAIR_REASON = "EXACT_ACTIVE_UNIT_NORMALIZATION_AND_METHOD_CONTRACT_REPAIR"
P1R52_SUPERSEDES_SOURCE_HEAD = "c35ebe5c299c919e68e496cbc2f0d77512c73f7a"


class P1R52ActiveDirectionContractError(ODEBFContractError):
    """An active semantic column has no finite nonzero KDC direction."""


class P1R52SemanticDescentContractError(ODEBFContractError):
    """An active KDC direction is not a strict semantic descent direction."""


@dataclass(frozen=True, slots=True)
class P1R52TargetProposal:
    primary_step: Any
    next_state: P1R51ControllerState
    semantic_gradient: torch.Tensor
    kdc_direction: torch.Tensor
    raw_velocity: torch.Tensor
    primary_delta: torch.Tensor
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class P1R52SelectedTarget:
    target_step: Any
    selected_endpoint: ScalableObjectiveResult
    next_state: P1R51ControllerState
    receipt: Mapping[str, Any]


def _entropy(shares: torch.Tensor) -> float:
    positive = shares[shares > 0.0]
    return 0.0 if positive.numel() == 0 else float(-torch.sum(positive * torch.log(positive)))


def _column_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    numerator = torch.sum(left * right, dim=0)
    denominator = torch.linalg.vector_norm(left, dim=0) * torch.linalg.vector_norm(right, dim=0)
    return torch.where(
        denominator > P1R52_NUMERICAL_EPSILON,
        numerator / denominator,
        torch.zeros_like(numerator),
    )


def _origin_relative_clamp(
    candidate: torch.Tensor,
    origin: torch.Tensor,
    clamp_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if candidate.shape != origin.shape or candidate.ndim != 2 or clamp_factor <= 0.0:
        raise ODEBFContractError("P1R52 clamp geometry differs")
    relative = candidate - origin
    relative_norm = torch.linalg.vector_norm(relative, dim=0)
    maximum = clamp_factor * torch.linalg.vector_norm(origin, dim=0)
    ratio = torch.ones_like(relative_norm)
    positive = relative_norm > 0.0
    ratio[positive] = torch.minimum(
        torch.ones_like(relative_norm[positive]),
        maximum[positive] / relative_norm[positive],
    )
    return origin + ratio.unsqueeze(0) * relative, ratio, maximum


def prepare_p1r52_target_proposal(
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    target_origin: torch.Tensor,
    nll: ScalableObjectiveResult,
    kl: P1R24KLResult,
    lock: P1R24AliasTargetLock,
    state: P1R51ControllerState,
    *,
    alias: str,
    step_index: int,
    shared_speed: float,
    kl_teacher_input_sha256: str,
) -> P1R52TargetProposal:
    """Build one P1R52 primary proposal without any model call."""

    if alias not in ("llama3-8b-inst", "qwen2.5-7b-inst") or lock.alias != alias:
        raise ODEBFContractError("P1R52 alias/target lock differs")
    if nll.target_gradient is None or kl.gradient is None:
        raise ODEBFContractError("P1R52 target gradients are absent")
    if step_index < 0 or step_index >= P1R24_K:
        raise ODEBFContractError("P1R52 target step index differs")
    shape = current_target.shape
    request_count = shape[1] if current_target.ndim == 2 else 0
    if (
        request_count <= 0
        or current_terminal.shape != shape
        or target_origin.shape != shape
        or len(nll.per_request_values) != request_count
        or len(kl.per_request_values) != request_count
        or len(state.cumulative_accepted_activation_path) != request_count
        or not math.isfinite(shared_speed)
        or shared_speed <= 0.0
        or len(kl_teacher_input_sha256) != 64
    ):
        raise ODEBFContractError("P1R52 target state geometry differs")

    current64 = current_target.detach().to(device="cpu", dtype=torch.float64)
    origin64 = target_origin.detach().to(device="cpu", dtype=torch.float64)
    aggregate_semantic = nll.target_gradient.detach().to(device="cpu", dtype=torch.float64)
    aggregate_kl = kl.gradient.detach().to(device="cpu", dtype=torch.float64)
    semantic = aggregate_semantic * request_count
    current_norm = torch.linalg.vector_norm(semantic, dim=0)
    nll_values = torch.tensor(nll.per_request_values, dtype=torch.float64)
    if (
        not bool(torch.isfinite(current64).all())
        or not bool(torch.isfinite(origin64).all())
        or not bool(torch.isfinite(semantic).all())
        or not bool(torch.isfinite(aggregate_kl).all())
    ):
        raise ODEBFContractError("P1R52 semantic/preservation gradient is nonfinite")
    if not bool(torch.isfinite(nll_values).all()) or bool(torch.any(nll_values < 0.0)):
        raise ODEBFContractError("P1R52 request-wise NLL is nonfinite or negative")

    if state.entry_semantic_gradient_norm is None:
        entry_norm = current_norm + P1R52_NUMERICAL_EPSILON
        next_state = P1R51ControllerState(
            tuple(float(item) for item in entry_norm),
            state.cumulative_accepted_activation_path,
        )
    else:
        if len(state.entry_semantic_gradient_norm) != request_count:
            raise ODEBFContractError("P1R52 entry gradient state differs")
        entry_norm = torch.tensor(state.entry_semantic_gradient_norm, dtype=torch.float64)
        next_state = state
    if bool(torch.any(~torch.isfinite(entry_norm))) or bool(torch.any(entry_norm <= 0.0)):
        raise ODEBFContractError("P1R52 entry gradient norm is invalid")

    # Frozen P1R51 reference energy and request-wise amplitude.
    nominal_parent_velocity = -shared_speed * semantic / entry_norm.unsqueeze(0)
    parent_norm = torch.linalg.vector_norm(nominal_parent_velocity, dim=0)
    reference_energy = float(torch.sum(torch.square(parent_norm)))
    if not math.isfinite(reference_energy):
        raise ODEBFContractError("P1R52 reference energy is nonfinite")
    active = current_norm > P1R52_NUMERICAL_EPSILON
    active_nll = torch.where(active, nll_values, torch.zeros_like(nll_values))
    allocation_denominator = float(torch.linalg.vector_norm(active_nll))
    amplitude = torch.zeros_like(nll_values)
    if reference_energy > 0.0 and bool(torch.any(active)):
        amplitude[active] = (
            math.sqrt(reference_energy)
            * nll_values[active]
            / (allocation_denominator + P1R52_NUMERICAL_EPSILON)
        )

    # AlphaEdit-compatible request-wise decay at the frozen origin.
    displacement_from_origin = current64 - origin64
    origin_norm = torch.linalg.vector_norm(origin64, dim=0)
    displacement_norm = torch.linalg.vector_norm(displacement_from_origin, dim=0)
    if bool(torch.any(origin_norm <= 0.0)):
        raise ODEBFContractError("P1R52 target decay origin is degenerate")
    decay_values = lock.decay_factor * displacement_norm / torch.square(origin_norm)
    decay_gradient = torch.zeros_like(displacement_from_origin)
    nonzero_decay = displacement_norm > 0.0
    decay_gradient[:, nonzero_decay] = (
        lock.decay_factor
        * displacement_from_origin[:, nonzero_decay]
        / displacement_norm[nonzero_decay].unsqueeze(0)
        / torch.square(origin_norm[nonzero_decay]).unsqueeze(0)
        / request_count
    )

    # Batch B and each scientific coefficient are applied exactly once.
    kl_gradient = aggregate_kl * request_count
    kl_component = lock.kl_factor * kl_gradient
    decay_component = decay_gradient * request_count
    preservation = kl_component + decay_component
    if not bool(torch.isfinite(preservation).all()):
        raise ODEBFContractError("P1R52 scaled preservation gradient is nonfinite")
    kl_gradient_norm = torch.linalg.vector_norm(kl_gradient, dim=0)
    decay_gradient_norm = torch.linalg.vector_norm(decay_component, dim=0)
    preservation_norm = torch.linalg.vector_norm(preservation, dim=0)
    preservation_direction = -preservation
    raw_conflict = torch.sum(semantic * preservation_direction, dim=0)
    semantic_norm_sq = torch.sum(torch.square(semantic), dim=0)
    projection_coefficient = torch.clamp(raw_conflict, min=0.0) / (
        semantic_norm_sq + P1R52_NUMERICAL_EPSILON
    )
    removed_component = semantic * projection_coefficient.unsqueeze(0)
    safe_preservation = preservation_direction - removed_component
    safe_inner = torch.sum(semantic * safe_preservation, dim=0)
    if bool(torch.any(safe_inner > P1R24_NUMERICAL_EPSILON)):
        raise ODEBFContractError("P1R52 one-sided semantic safety certificate differs")
    if bool(torch.any((raw_conflict <= 0.0) & (torch.linalg.vector_norm(removed_component, dim=0) > P1R24_NUMERICAL_EPSILON))):
        raise ODEBFContractError("P1R52 nonconflicting preservation was removed")

    combined_direction = -semantic + safe_preservation
    combined_norm = torch.linalg.vector_norm(combined_direction, dim=0)
    if not bool(torch.isfinite(combined_direction).all()):
        raise P1R52ActiveDirectionContractError(
            "P1R52 combined direction is nonfinite"
        )
    if bool(torch.any(active & (~torch.isfinite(combined_norm)))):
        raise P1R52ActiveDirectionContractError(
            "P1R52 active combined direction is nonfinite"
        )
    if bool(torch.any(active & (combined_norm <= P1R52_NUMERICAL_EPSILON))):
        raise P1R52ActiveDirectionContractError(
            "P1R52 active combined direction does not exceed epsilon_a"
        )
    combined_semantic_inner = torch.sum(semantic * combined_direction, dim=0)
    if bool(torch.any(active & (combined_semantic_inner >= 0.0))):
        raise P1R52SemanticDescentContractError(
            "P1R52 active combined direction is not strict semantic descent"
        )
    kdc_direction = torch.zeros_like(combined_direction)
    if bool(torch.any(active)):
        kdc_direction[:, active] = (
            combined_direction[:, active] / combined_norm[active].unsqueeze(0)
        )
    kdc_direction_norm = torch.linalg.vector_norm(kdc_direction, dim=0)
    unit_norm_residual = torch.where(
        active,
        torch.abs(kdc_direction_norm - 1.0),
        torch.zeros_like(kdc_direction_norm),
    )
    if float(torch.max(unit_norm_residual)) > P1R24_NUMERICAL_EPSILON:
        raise P1R52ActiveDirectionContractError(
            "P1R52 active KDC direction is not exact unit norm"
        )
    final_semantic_inner = torch.sum(semantic * kdc_direction, dim=0)
    if bool(torch.any(active & (final_semantic_inner >= 0.0))):
        raise P1R52SemanticDescentContractError(
            "P1R52 active unit direction is not strict semantic descent"
        )
    semantic_direction = torch.zeros_like(semantic)
    if bool(torch.any(active)):
        semantic_direction[:, active] = -semantic[:, active] / current_norm[active].unsqueeze(0)

    velocity = kdc_direction * amplitude.unsqueeze(0)
    allocation_energy = float(torch.sum(torch.square(amplitude)))
    raw_energy_by_request = torch.square(torch.linalg.vector_norm(velocity, dim=0))
    raw_kdc_energy = float(torch.sum(raw_energy_by_request))
    if not math.isfinite(raw_kdc_energy):
        raise ODEBFContractError("P1R52 raw target velocity energy is nonfinite")
    raw_energy_relative_error = abs(raw_kdc_energy - reference_energy) / (
        reference_energy + P1R52_NUMERICAL_EPSILON
    )
    if bool(torch.any(active)) and raw_energy_relative_error > P1R24_NUMERICAL_EPSILON:
        raise ODEBFContractError("P1R52 raw target velocity energy is not conserved")
    nominal_delta = P1R24_H * velocity
    candidate = current64 + nominal_delta
    clamped64, clamp_ratio, clamp_maximum = _origin_relative_clamp(
        candidate, origin64, lock.clamp_factor
    )
    post_clamp_pre_cast_delta = clamped64 - current64
    target_next = clamped64.to(dtype=torch.float32).contiguous()
    actual_delta = target_next.to(dtype=torch.float64) - current64
    if not bool(torch.isfinite(actual_delta).all()):
        raise ODEBFContractError("P1R52 post-clamp target displacement is nonfinite")
    post_cast_origin_norm = torch.linalg.vector_norm(
        target_next.to(dtype=torch.float64) - origin64, dim=0
    )
    clamp_bound_residual = torch.clamp(post_cast_origin_norm - clamp_maximum, min=0.0)
    # FP32 rounding is covered by the inherited numerical tolerance.
    if float(torch.max(clamp_bound_residual)) > P1R24_NUMERICAL_EPSILON:
        raise ODEBFContractError("P1R52 origin-relative clamp bound differs")

    post_clamp_pre_cast_energy_by_request = torch.square(
        torch.linalg.vector_norm(post_clamp_pre_cast_delta / P1R24_H, dim=0)
    )
    post_cast_energy_by_request = torch.square(
        torch.linalg.vector_norm(actual_delta / P1R24_H, dim=0)
    )
    clamp_only_removed_energy_by_request = (
        raw_energy_by_request - post_clamp_pre_cast_energy_by_request
    )
    if float(torch.min(clamp_only_removed_energy_by_request)) < -P1R24_NUMERICAL_EPSILON:
        raise ODEBFContractError("P1R52 clamp increased target velocity energy")
    fp32_cast_energy_delta_by_request = (
        post_cast_energy_by_request - post_clamp_pre_cast_energy_by_request
    )
    post_clamp_pre_cast_energy = float(torch.sum(post_clamp_pre_cast_energy_by_request))
    post_cast_energy = float(torch.sum(post_cast_energy_by_request))
    rsa_shares = (
        torch.square(amplitude) / float(torch.sum(torch.square(amplitude)))
        if float(torch.sum(torch.square(amplitude))) > 0.0
        else torch.zeros_like(amplitude)
    )
    sorted_shares = torch.sort(rsa_shares, descending=True).values
    semantic_slope_ratio = -torch.sum(semantic * kdc_direction, dim=0) / (
        current_norm + P1R52_NUMERICAL_EPSILON
    )
    direction_cosine = _column_cosine(semantic_direction, kdc_direction)
    clamp_hits = clamp_ratio < 1.0

    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r52-rsa-r42safekdc-primary-target-repair-r1/v1",
        "instruction_id": P1R52_INSTRUCTION_ID,
        "method_id": P1R52_METHOD_ID,
        "repair_revision": P1R52_REPAIR_REVISION,
        "repair_reason": P1R52_REPAIR_REASON,
        "supersedes_source_head": P1R52_SUPERSEDES_SOURCE_HEAD,
        "target_direction_policy": P1R52_DIRECTION_POLICY,
        "parent_amplitude_policy": "P1R51_REQUESTWISE_NLL_PROPORTIONAL_ENERGY_ALLOCATION",
        "parent_selection_policy": "P1R43_PRIMARY_RESCUE_CURRENT",
        "preservation_projection": "R42_ONE_SIDED_SEMANTIC_SAFE",
        "kl_direction": "KL_CURRENT_TO_TEACHER_W0",
        "kl_teacher_capture_count_per_case": 1,
        "kl_teacher_refresh_count": 0,
        "kl_teacher_input_sha256": kl_teacher_input_sha256,
        "kl_teacher_hash_entry_equals_current": True,
        "k": step_index,
        "alias": alias,
        "request_count": request_count,
        "h": P1R24_H,
        "shared_speed": shared_speed,
        "native_compatible_lock": lock.raw_free_payload(),
        "numerical_epsilon": P1R52_NUMERICAL_EPSILON,
        "target_new_nll_current_by_request": [float(item) for item in nll_values],
        "target_new_nll_current_summary": _summary(nll.per_request_values),
        "kl_by_request": [float(item) for item in kl.per_request_values],
        "decay_by_request": [float(item) for item in decay_values],
        "semantic_gradient_norm_by_request": [float(item) for item in current_norm],
        "kl_gradient_norm_by_request": [float(item) for item in kl_gradient_norm],
        "decay_gradient_norm_by_request": [float(item) for item in decay_gradient_norm],
        "preservation_gradient_norm_by_request": [float(item) for item in preservation_norm],
        "preservation_to_semantic_norm_ratio_by_request": [
            float(preservation_norm[index] / (current_norm[index] + P1R52_NUMERICAL_EPSILON))
            for index in range(request_count)
        ],
        "raw_preservation_conflict_inner_product_by_request": [float(item) for item in raw_conflict],
        "projection_coefficient_by_request": [float(item) for item in projection_coefficient],
        "removed_conflicting_component_norm_by_request": [
            float(item) for item in torch.linalg.vector_norm(removed_component, dim=0)
        ],
        "safe_preservation_inner_product_by_request": [float(item) for item in safe_inner],
        "safe_preservation_certificate_max": float(torch.max(safe_inner)),
        "safe_preservation_certificate_tolerance": P1R24_NUMERICAL_EPSILON,
        "p1r51_semantic_to_kdc_direction_cosine_by_request": [float(item) for item in direction_cosine],
        "semantic_slope_ratio_by_request": [float(item) for item in semantic_slope_ratio],
        "combined_direction_norm_by_request": [float(item) for item in combined_norm],
        "final_semantic_inner_product_by_request": [float(item) for item in final_semantic_inner],
        "final_semantic_descent_pass_by_request": [
            bool((not bool(active[index])) or final_semantic_inner[index] < 0.0)
            for index in range(request_count)
        ],
        "kdc_direction_norm_by_request": [float(item) for item in kdc_direction_norm],
        "kdc_unit_norm_max_abs_residual": float(torch.max(unit_norm_residual)),
        "entry_semantic_gradient_norm_by_request": [float(item) for item in entry_norm],
        "p1r43_nominal_velocity_norm_by_request": [float(item) for item in parent_norm],
        "reference_energy": reference_energy,
        "allocation_energy": allocation_energy,
        "raw_kdc_energy": raw_kdc_energy,
        "raw_energy_relative_error": raw_energy_relative_error,
        "allocation_amplitude_by_request": [float(item) for item in amplitude],
        "allocation_energy_share_by_request": [float(item) for item in rsa_shares],
        "raw_energy_by_request": [float(item) for item in raw_energy_by_request],
        "post_clamp_pre_cast_energy_by_request": [
            float(item) for item in post_clamp_pre_cast_energy_by_request
        ],
        "post_cast_energy_by_request": [float(item) for item in post_cast_energy_by_request],
        "clamp_only_removed_energy_by_request": [
            float(item) for item in clamp_only_removed_energy_by_request
        ],
        "fp32_cast_energy_delta_by_request": [
            float(item) for item in fp32_cast_energy_delta_by_request
        ],
        "pre_clamp_energy_by_request": [float(item) for item in raw_energy_by_request],
        "post_clamp_energy_by_request": [float(item) for item in post_cast_energy_by_request],
        "clamp_removed_energy_by_request": [
            float(item) for item in clamp_only_removed_energy_by_request
        ],
        "pre_clamp_energy": raw_kdc_energy,
        "post_clamp_pre_cast_energy": post_clamp_pre_cast_energy,
        "post_cast_energy": post_cast_energy,
        "post_clamp_energy": post_cast_energy,
        "clamp_only_removed_energy": float(torch.sum(clamp_only_removed_energy_by_request)),
        "fp32_cast_energy_delta": float(torch.sum(fp32_cast_energy_delta_by_request)),
        "clamp_removed_energy": float(torch.sum(clamp_only_removed_energy_by_request)),
        "clamp_hit_by_request": [bool(item) for item in clamp_hits],
        "clamp_hit_count": int(torch.sum(clamp_hits)),
        "clamp_ratio_by_request": [float(item) for item in clamp_ratio],
        "origin_clamp_bound_by_request": [float(item) for item in clamp_maximum],
        "origin_clamp_post_cast_norm_by_request": [float(item) for item in post_cast_origin_norm],
        "origin_clamp_bound_max_abs_excess": float(torch.max(clamp_bound_residual)),
        "nominal_target_displacement_norm_by_request": [
            float(item) for item in torch.linalg.vector_norm(nominal_delta, dim=0)
        ],
        "post_cast_actual_delta_norm_by_request": [
            float(item) for item in torch.linalg.vector_norm(actual_delta, dim=0)
        ],
        "active_gradient_mask": [bool(item) for item in active],
        "active_gradient_count": int(torch.sum(active)),
        "top1_allocation_share": float(sorted_shares[0]),
        "top3_allocation_share": float(torch.sum(sorted_shares[: min(3, request_count)])),
        "allocation_entropy": _entropy(rsa_shares),
        "entry_norm_calibration_count": 1 if step_index == 0 else 0,
        "entry_norm_frozen_after_k0": True,
        "cumulative_accepted_activation_path_before_by_request": list(state.cumulative_accepted_activation_path),
        "semantic_gradient_sha256": tensor_sha256(semantic),
        "preservation_gradient_sha256": tensor_sha256(preservation),
        "safe_preservation_direction_sha256": tensor_sha256(safe_preservation),
        "kdc_direction_sha256": tensor_sha256(kdc_direction),
        "nominal_delta_sha256": tensor_sha256(nominal_delta),
        "primary_delta_sha256": tensor_sha256(actual_delta),
        "target_next_sha256": tensor_sha256(target_next),
        "rescue_directional_derivative_delta_source": "POST_CLAMP_POST_FP32_CAST_ACTUAL_DELTA",
        "clamp_energy_redistribution_count": 0,
        "p1r51_energy_identity_scope": "PRE_CLAMP_NOMINAL_FIELD_ONLY",
        "target_kl_decision_influence_count": int(bool(torch.any(kl_gradient_norm > 0.0))),
        "target_decay_decision_influence_count": int(bool(torch.any(decay_gradient_norm > 0.0))),
        "target_preservation_decision_influence_count": int(bool(torch.any(preservation_norm > 0.0))),
        "target_hit_threshold_access_count": 0,
        "persistent_freeze_input_count": 0,
        "semantic_debt_input_count": 0,
        "remaining_horizon_decision_influence_count": 0,
        "same_step_retry_count": 0,
        "added_kl_model_forward_count": 0,
        "added_kl_backward_count": 0,
        "added_model_forward_count": 0,
        "added_backward_count": 0,
        "added_key_factor_refresh_count": 0,
        "added_materialization_count": 0,
        "kl_result_identity_sha256": kl.identity_sha256,
    }
    step = _full_current_residual_step(
        target_next=target_next,
        current_target=current_target,
        current_terminal=current_terminal,
        nll_gradient=aggregate_semantic,
        aggregate_gradient=aggregate_semantic + lock.kl_factor * aggregate_kl + decay_gradient,
        held=tuple(False for _ in range(request_count)),
        receipt=receipt,
    )
    return P1R52TargetProposal(
        step,
        next_state,
        semantic,
        kdc_direction,
        velocity,
        actual_delta,
        step.receipt,
    )


def prepare_p1r52_rescue_proposal(
    proposal: P1R52TargetProposal,
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    current_nll: ScalableObjectiveResult,
    primary_endpoint: ScalableObjectiveResult,
    *,
    step_index: int,
) -> P1R43RescueProposal:
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
        "schema": "ode-edit-s05-p1r52-rsa-r42safekdc-scalar-rescue-plan-repair-r1/v1",
        "instruction_id": P1R52_INSTRUCTION_ID,
        "method_id": P1R52_METHOD_ID,
        "repair_revision": P1R52_REPAIR_REVISION,
        "directional_derivative_delta_source": "POST_CLAMP_POST_FP32_CAST_ACTUAL_DELTA",
    }
    receipt.pop("identity_sha256", None)
    receipt["identity_sha256"] = canonical_hash(receipt)
    rescue_step = parent.rescue_step
    if rescue_step is not None:
        step_receipt = {
            **dict(rescue_step.receipt),
            "schema": "ode-edit-s05-p1r52-rsa-r42safekdc-scalar-rescue-target-repair-r1/v1",
            "instruction_id": P1R52_INSTRUCTION_ID,
            "method_id": P1R52_METHOD_ID,
            "repair_revision": P1R52_REPAIR_REVISION,
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


def select_p1r52_target_proposal(
    proposal: P1R52TargetProposal,
    rescue: P1R43RescueProposal,
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    current_nll: ScalableObjectiveResult,
    primary_endpoint: ScalableObjectiveResult,
    rescue_endpoint: ScalableObjectiveResult | None,
    *,
    step_index: int,
) -> P1R52SelectedTarget:
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
    selected_values = tuple(float(item) for item in selected.selected_endpoint.per_request_values)
    selection = tuple(str(item) for item in selected.receipt["selection_by_request"])
    post_cast_energy_by_request = torch.tensor(
        proposal.receipt["post_cast_energy_by_request"], dtype=torch.float64
    )
    accepted_energy_by_request = torch.square(accepted_norm / P1R24_H)
    rescue_contraction_energy_by_request = torch.tensor(
        [
            float(post_cast_energy_by_request[index] - accepted_energy_by_request[index])
            if choice == "RESCUE"
            else 0.0
            for index, choice in enumerate(selection)
        ],
        dtype=torch.float64,
    )
    current_selection_unused_energy_by_request = torch.tensor(
        [
            float(post_cast_energy_by_request[index]) if choice == "CURRENT" else 0.0
            for index, choice in enumerate(selection)
        ],
        dtype=torch.float64,
    )
    selection_unused_energy_by_request = (
        post_cast_energy_by_request - accepted_energy_by_request
    )
    raw_energy_by_request = torch.tensor(
        proposal.receipt["raw_energy_by_request"], dtype=torch.float64
    )
    total_unused_energy_by_request = raw_energy_by_request - accepted_energy_by_request
    clamp_only_removed = torch.tensor(
        proposal.receipt["clamp_only_removed_energy_by_request"], dtype=torch.float64
    )
    fp32_delta = torch.tensor(
        proposal.receipt["fp32_cast_energy_delta_by_request"], dtype=torch.float64
    )
    energy_decomposition_residual = (
        total_unused_energy_by_request
        - (clamp_only_removed - fp32_delta + selection_unused_energy_by_request)
    )
    receipt = {
        **dict(selected.receipt),
        "schema": "ode-edit-s05-p1r52-rsa-r42safekdc-target-selection-repair-r1/v1",
        "instruction_id": P1R52_INSTRUCTION_ID,
        "method_id": P1R52_METHOD_ID,
        "repair_revision": P1R52_REPAIR_REVISION,
        "repair_reason": P1R52_REPAIR_REASON,
        "supersedes_source_head": P1R52_SUPERSEDES_SOURCE_HEAD,
        "parent_selection_byte_semantics": "P1R43_PRIMARY_RESCUE_CURRENT",
        "rescue_directional_derivative_delta_source": "POST_CLAMP_POST_FP32_CAST_ACTUAL_DELTA",
        "accepted_target_nll_improvement_by_request": [
            current - endpoint
            for current, endpoint in zip(current_values, selected_values, strict=True)
        ],
        "accepted_activation_path_increment_by_request": [float(item) for item in accepted_norm],
        "cumulative_accepted_activation_path_by_request": list(cumulative),
        "accepted_energy_by_request": [float(item) for item in accepted_energy_by_request],
        "accepted_energy": float(torch.sum(accepted_energy_by_request)),
        "rescue_contraction_energy_by_request": [
            float(item) for item in rescue_contraction_energy_by_request
        ],
        "rescue_contraction_energy": float(torch.sum(rescue_contraction_energy_by_request)),
        "current_selection_unused_energy_by_request": [
            float(item) for item in current_selection_unused_energy_by_request
        ],
        "current_selection_unused_energy": float(
            torch.sum(current_selection_unused_energy_by_request)
        ),
        "total_unused_energy_by_request": [float(item) for item in total_unused_energy_by_request],
        "total_unused_energy": float(torch.sum(total_unused_energy_by_request)),
        "energy_decomposition_max_abs_residual": float(
            torch.max(torch.abs(energy_decomposition_residual))
        ),
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
    return P1R52SelectedTarget(target_step, selected.selected_endpoint, next_state, target_step.receipt)


__all__ = [
    "P1R52ActiveDirectionContractError",
    "P1R52_DIRECTION_POLICY",
    "P1R52_INSTRUCTION_ID",
    "P1R52_METHOD_ID",
    "P1R52_NUMERICAL_EPSILON",
    "P1R52_REPAIR_REASON",
    "P1R52_REPAIR_REVISION",
    "P1R52_SUPERSEDES_SOURCE_HEAD",
    "P1R52SemanticDescentContractError",
    "P1R52SelectedTarget",
    "P1R52TargetProposal",
    "prepare_p1r52_rescue_proposal",
    "prepare_p1r52_target_proposal",
    "select_p1r52_target_proposal",
]
