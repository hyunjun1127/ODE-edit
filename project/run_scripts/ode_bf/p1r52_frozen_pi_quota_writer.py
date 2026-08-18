"""Current-layer-only sequential writer policies for P1R52-FPiQ.

The target controller and entry Soft router stay authoritative.  This module
only replaces the physical writer after the entry route has been certified.
It deliberately owns the residual-velocity coordinate so that ``pi`` and
Euler ``h`` each enter exactly once.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .alpha_backend import W64_CAST_DTYPE
from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import (
    CachedBF16FunctionalTrial,
    WaypointFactor,
    assemble_effective_bf16,
    tensor_sha256,
)
from .p1_backend import (
    FULL_CURRENT_RESIDUAL_DIVISOR,
    FULL_CURRENT_RESIDUAL_VELOCITY_DEFINITION,
    P1DynamicField,
    P1LayerField,
    PinnedCovarianceRegistry,
)
from .p1r24_atomic_strength import P1R24_Q_EPSILON
from .progress_simplex_routing import SIMPLEX_PRIMAL_TOLERANCE
from .scalable_batched_model import (
    ScalableCapturePlan,
    ScalableObjectivePlan,
    ScalableObjectiveResult,
    _parameter_inventory_sha256,
    capture_scalable_physical_state,
    evaluate_scalable_target_new_objective,
)
from .scalable_batched_runtime import P1R23_H, P1R23_LAYER_ORDER
from .woodbury import ProjectorCertificate, solve_alpha_woodbury


P1R52_FPIQ_INSTRUCTION_ID = (
    "ODEEDIT-S05-P1R52-FROZEN-PI-SEQUENTIAL-QUOTA-WRITER-V1"
)
P1R52_FPIQ_METHOD_ID = "P1R52-FROZEN-PI-SEQUENTIAL-QUOTA-WRITER-V1"
P1R52_FPIQ_CONTRACT_SHA256 = (
    "e5c767cdd6cfd498376dacad39155d08d0bc79c89496dbf629725a9b928b4436"
)


class P1R52WriterPolicy(str, Enum):
    J0 = "J0"
    SV = "SV"
    FPIQ = "FPIQ"


SEQUENTIAL_POLICIES = (P1R52WriterPolicy.SV, P1R52WriterPolicy.FPIQ)


@dataclass(frozen=True, slots=True)
class SequentialWriterResult:
    policy: P1R52WriterPolicy
    increment: Mapping[str, WaypointFactor]
    velocity: tuple[float, ...]
    layer_fields: tuple[P1LayerField, ...]
    predicted_progress: float
    final_virtual_objective: ScalableObjectiveResult
    expected_bf16_sha256: Mapping[str, str]
    receipt: Mapping[str, Any]


def sequential_quota_decision(
    *,
    policy: P1R52WriterPolicy | str,
    alpha_star: float,
    current_nll: float,
    endpoint_nll: float,
    entry_pi: Sequence[float],
    entry_velocity: Sequence[float],
    applied_slope: float,
    layer_ordinal: int,
) -> dict[str, Any]:
    """Pure quota/velocity algebra shared by runtime tests and execution."""

    selected = P1R52WriterPolicy(policy)
    pi0 = tuple(float(item) for item in entry_pi)
    velocity0 = tuple(float(item) for item in entry_velocity)
    if (
        selected not in SEQUENTIAL_POLICIES
        or len(pi0) != len(P1R23_LAYER_ORDER)
        or len(velocity0) != len(P1R23_LAYER_ORDER)
        or layer_ordinal < 0
        or layer_ordinal >= len(pi0)
        or any(
            not math.isfinite(item)
            for item in (
                alpha_star,
                current_nll,
                endpoint_nll,
                applied_slope,
                *pi0,
                *velocity0,
            )
        )
        or alpha_star < 0.0
        or any(item < 0.0 for item in (*pi0, *velocity0))
    ):
        raise ODEBFContractError("P1R52-FPiQ quota input differs")
    alpha_remaining = min(
        max(float(current_nll) - float(endpoint_nll), 0.0),
        float(alpha_star),
    )
    suffix_mass = math.fsum(pi0[layer_ordinal:])
    relative_share = (
        pi0[layer_ordinal] / suffix_mass
        if suffix_mass > P1R24_Q_EPSILON
        else 0.0
    )
    quota = alpha_remaining * relative_share
    velocity = (
        velocity0[layer_ordinal]
        if selected is P1R52WriterPolicy.SV
        else quota / applied_slope
        if applied_slope > P1R24_Q_EPSILON
        else 0.0
    )
    return {
        "alpha_remaining": alpha_remaining,
        "suffix_mass": suffix_mass,
        "relative_share": relative_share,
        "quota": quota,
        "velocity": velocity,
        "support_exhausted": bool(
            suffix_mass <= P1R24_Q_EPSILON
            and alpha_remaining > P1R24_Q_EPSILON
        ),
        "nonpositive_slope_zero_velocity": bool(
            selected is P1R52WriterPolicy.FPIQ
            and applied_slope <= P1R24_Q_EPSILON
        ),
    }


def _parameter_state(
    model: torch.nn.Module, weight_names: Sequence[str]
) -> tuple[tuple[str, int, int, str], ...]:
    parameters = dict(model.named_parameters())
    return tuple(
        (
            name,
            parameters[name].data_ptr(),
            parameters[name]._version,
            canonical_hash(
                {
                    "dtype": str(parameters[name].dtype),
                    "shape": list(parameters[name].shape),
                    "requires_grad": bool(parameters[name].requires_grad),
                }
            ),
        )
        for name in weight_names
    )


def _merge_factors(
    base: Mapping[str, Sequence[WaypointFactor]],
    increment: Mapping[str, WaypointFactor],
) -> dict[str, tuple[WaypointFactor, ...]]:
    # The inherited inventory names all five touched weights even at k0.
    # Functional trials accept only endpoints with at least one factor, so a
    # future prefix layer remains absent until its factor is planned.
    result = {
        name: tuple(items) for name, items in base.items() if tuple(items)
    }
    for name, factor in increment.items():
        values = result.get(name, ()) + (factor,)
        if len({item.order_key for item in values}) != len(values):
            raise ODEBFContractError("P1R52-FPiQ factor order repeats")
        result[name] = tuple(sorted(values, key=lambda item: item.order_key))
    return result


def _velocity_residual(
    target_state: torch.Tensor,
    current_terminal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    target = target_state.detach().to(device="cpu", dtype=torch.float32)
    current = current_terminal.detach().to(device="cpu", dtype=torch.float32)
    if target.shape != current.shape or target.ndim != 2:
        raise ODEBFContractError("P1R52-FPiQ residual geometry differs")
    residual = (target - current).contiguous()
    velocity_residual = (residual / float(P1R23_H)).contiguous()
    identity = float(
        torch.max(
            torch.abs(float(P1R23_H) * velocity_residual - residual)
        ).item()
    )
    if not torch.isfinite(velocity_residual).all() or identity != 0.0:
        raise ODEBFContractError("P1R52-FPiQ full residual/h identity differs")
    return residual, velocity_residual, identity


def _layer_field(
    model: torch.nn.Module,
    hparams: Any,
    projector: torch.Tensor,
    covariance_registry: PinnedCovarianceRegistry,
    *,
    layer: int,
    key: torch.Tensor,
    target_state: torch.Tensor,
    current_terminal: torch.Tensor,
    step_index: int,
    factor_ordinal: int,
    projector_sha256: str,
    residual_tolerance: float,
) -> tuple[P1LayerField, float]:
    layers = tuple(int(item) for item in hparams.layers)
    if layers != P1R23_LAYER_ORDER or layer not in layers:
        raise ODEBFContractError("P1R52-FPiQ layer inventory differs")
    request_count = target_state.shape[1]
    if key.ndim != 2 or key.shape[1] != request_count:
        raise ODEBFContractError("P1R52-FPiQ current key geometry differs")
    residual, velocity_residual, identity = _velocity_residual(
        target_state, current_terminal
    )
    device = next(model.parameters()).device
    layer_index = layers.index(layer)
    p_device = projector[layer_index].to(device=device, dtype=torch.float32)
    k_device = key.detach().to(device=device, dtype=torch.float32)
    empty_history = torch.empty(
        (key.shape[0], 0), device=device, dtype=torch.float32
    )
    solved = solve_alpha_woodbury(
        p_device,
        k_device,
        history_keys=empty_history,
        regularization=float(hparams.L2),
        projector_certificate=ProjectorCertificate(
            projector_sha256,
            1.0,
            1.0,
            "artifact-unverified",
            1.0e-10,
        ),
        residual_tolerance=residual_tolerance,
        expected_batch_size=request_count,
    )
    if (
        not solved.certificate.passed
        or solved.certificate.alpha_linear_residual > residual_tolerance
    ):
        raise ODEBFContractError("P1R52-FPiQ current-layer W64 solve failed")
    q = solved.q.detach().to(device="cpu", dtype=W64_CAST_DTYPE).contiguous()
    projected = (p_device @ k_device).detach().to(
        device="cpu", dtype=torch.float32
    )
    covariance_action, covariance_gram, covariance_receipt = (
        covariance_registry.action(
            layer, q, expected_batch_size=request_count
        )
    )
    right_gram = q.T.to(dtype=torch.float64) @ q.to(dtype=torch.float64)
    left_gram = (
        velocity_residual.T.to(dtype=torch.float64)
        @ velocity_residual.to(dtype=torch.float64)
    )
    factor_energy = float(torch.sum(left_gram * right_gram))
    if not math.isfinite(factor_energy) or factor_energy < 0.0:
        raise ODEBFContractError("P1R52-FPiQ current-layer energy differs")
    weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
    parameter = dict(model.named_parameters()).get(weight_name)
    if parameter is None or tuple(parameter.shape) != (
        velocity_residual.shape[0],
        q.shape[0],
    ):
        raise ODEBFContractError("P1R52-FPiQ factor/parameter geometry differs")
    factor = WaypointFactor(
        weight_name,
        layer,
        0,
        step_index,
        factor_ordinal,
        1.0,
        velocity_residual.clone(),
        q.clone(),
        global_batch_size=request_count,
    )
    field = P1LayerField(
        layer,
        weight_name,
        key.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        projected,
        velocity_residual,
        FULL_CURRENT_RESIDUAL_VELOCITY_DEFINITION,
        FULL_CURRENT_RESIDUAL_DIVISOR,
        q,
        factor,
        factor_energy,
        covariance_action,
        covariance_gram,
        covariance_receipt,
        solved.certificate,
        torch.empty((velocity_residual.shape[0], 0), dtype=torch.float64),
    )
    del p_device, k_device, empty_history, solved
    return field, identity


def _entry_layer_field(
    entry: P1LayerField,
    *,
    target_state: torch.Tensor,
    current_terminal: torch.Tensor,
    step_index: int,
    factor_ordinal: int,
) -> tuple[P1LayerField, float]:
    residual, velocity_residual, _identity = _velocity_residual(
        target_state, current_terminal
    )
    entry_residual = entry.residual.detach().to(
        device="cpu", dtype=torch.float32
    )
    entry_identity = float(
        torch.max(
            torch.abs(float(P1R23_H) * entry_residual - residual)
        ).item()
    )
    if (
        float(torch.max(torch.abs(velocity_residual - entry_residual)).item())
        > SIMPLEX_PRIMAL_TOLERANCE / float(P1R23_H)
        or entry_identity > SIMPLEX_PRIMAL_TOLERANCE
    ):
        raise ODEBFContractError("P1R52-FPiQ layer4 entry residual differs")
    del residual, velocity_residual, entry_residual, step_index, factor_ordinal
    return entry, entry_identity


def _current_slope(
    model: torch.nn.Module,
    objective_plan: ScalableObjectivePlan,
    field: P1LayerField,
) -> tuple[float, float, ScalableObjectiveResult]:
    coefficient = torch.zeros(
        1,
        device=next(model.parameters()).device,
        dtype=torch.float32,
        requires_grad=True,
    )
    observed = evaluate_scalable_target_new_objective(
        model,
        objective_plan,
        coefficient_layers=(field,),
        coefficients=coefficient,
    )
    if (
        observed.coefficient_gradient is None
        or observed.coefficient_gradient.numel() != 1
    ):
        raise ODEBFContractError("P1R52-FPiQ current slope is absent")
    raw = -float(observed.coefficient_gradient.item())
    applied = float(P1R23_H) * raw
    if not math.isfinite(raw) or not math.isfinite(applied):
        raise ODEBFContractError("P1R52-FPiQ current slope is non-finite")
    return raw, applied, observed


def _factor_for_velocity(
    field: P1LayerField,
    velocity: float,
    *,
    step_index: int,
    factor_ordinal: int,
) -> WaypointFactor:
    if not math.isfinite(velocity) or velocity < 0.0:
        raise ODEBFContractError("P1R52-FPiQ velocity differs")
    return WaypointFactor(
        field.weight_name,
        field.layer,
        0,
        step_index,
        factor_ordinal,
        float(P1R23_H) * velocity,
        field.residual.clone(),
        field.q.clone(),
        global_batch_size=field.factor.global_batch_size,
    )


def plan_sequential_writer(
    model: torch.nn.Module,
    *,
    policy: P1R52WriterPolicy | str,
    hparams: Any,
    projector: torch.Tensor,
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    residual_tolerance: float,
    objective_plan: ScalableObjectivePlan,
    capture_plan: ScalableCapturePlan,
    base_values: Mapping[str, torch.Tensor],
    current_factors: Mapping[str, Sequence[WaypointFactor]],
    entry_field: P1DynamicField,
    entry_applied_slopes: Sequence[float],
    entry_pi: Sequence[float],
    entry_velocity: Sequence[float],
    target_state: torch.Tensor,
    endpoint_nll: float,
    entry_nll: float,
    entry_per_request_nll: Sequence[float],
    step_index: int,
) -> SequentialWriterResult:
    selected = P1R52WriterPolicy(policy)
    if selected not in SEQUENTIAL_POLICIES:
        raise ODEBFContractError("P1R52-FPiQ sequential policy differs")
    layers = tuple(int(item) for item in hparams.layers)
    slopes0 = tuple(float(item) for item in entry_applied_slopes)
    pi0 = tuple(float(item) for item in entry_pi)
    velocity0 = tuple(float(item) for item in entry_velocity)
    if (
        layers != P1R23_LAYER_ORDER
        or tuple(item.layer for item in entry_field.layers) != layers
        or any(len(values) != len(layers) for values in (slopes0, pi0, velocity0))
        or any(not math.isfinite(item) for item in (*slopes0, *pi0, *velocity0))
        or any(item < 0.0 for item in (*pi0, *velocity0))
        or abs(math.fsum(pi0) - (1.0 if max(velocity0, default=0.0) > 0.0 else 0.0))
        > SIMPLEX_PRIMAL_TOLERANCE
    ):
        raise ODEBFContractError("P1R52-FPiQ entry routing identity differs")
    alpha_star = max(float(entry_nll) - float(endpoint_nll), 0.0)
    if not math.isfinite(alpha_star):
        raise ODEBFContractError("P1R52-FPiQ finite demand differs")
    weight_names = tuple(item.weight_name for item in entry_field.layers)
    before_inventory = _parameter_inventory_sha256(model)
    before_state = _parameter_state(model, weight_names)
    planned: dict[str, WaypointFactor] = {}
    selected_fields: list[P1LayerField] = []
    velocities: list[float] = []
    layer_receipts: list[dict[str, Any]] = []
    current_nll = float(entry_nll)
    current_per_request = tuple(float(item) for item in entry_per_request_nll)
    if len(current_per_request) != objective_plan.request_count:
        raise ODEBFContractError("P1R52-FPiQ entry request NLL geometry differs")
    capture_count = 0
    capture_forward_count = 0
    capture_processed_tokens = 0
    capture_padded_tokens = 0
    slope_group_count = 0
    slope_backward_count = 0
    slope_forward_count = 0
    slope_processed_tokens = 0
    slope_padded_tokens = 0
    for ordinal, layer in enumerate(layers):
        if ordinal == 0:
            field, residual_identity = _entry_layer_field(
                entry_field.layers[0],
                target_state=target_state,
                current_terminal=entry_field.current_z,
                step_index=step_index,
                factor_ordinal=ordinal,
            )
            applied_slope = slopes0[0]
            raw_slope = applied_slope / float(P1R23_H)
            key_capture_sha256 = None
            slope_receipt_sha256 = None
            current_terminal = entry_field.current_z
        else:
            candidate = _merge_factors(current_factors, planned)
            entry_subset = {name: base_values[name] for name in candidate}
            with CachedBF16FunctionalTrial(
                model, entry_subset, candidate, row_block=64
            ):
                physical = capture_scalable_physical_state(
                    model, capture_plan, hparams
                )
                capture_count += 1
                capture_forward_count += physical.physical_forward_count
                capture_processed_tokens += physical.processed_token_count
                capture_padded_tokens += physical.padded_token_count
                current_terminal = physical.terminal_z
                field, residual_identity = _layer_field(
                    model,
                    hparams,
                    projector,
                    covariance_registry,
                    layer=layer,
                    key=physical.keys_by_layer[layer],
                    target_state=target_state,
                    current_terminal=current_terminal,
                    step_index=step_index,
                    factor_ordinal=ordinal,
                    projector_sha256=projector_sha256,
                    residual_tolerance=residual_tolerance,
                )
                raw_slope, applied_slope, observed = _current_slope(
                    model, objective_plan, field
                )
            slope_group_count += 1
            slope_backward_count += observed.backward_count
            slope_forward_count += observed.model_forward_count
            slope_processed_tokens += observed.processed_token_count
            slope_padded_tokens += observed.padded_token_count
            key_capture_sha256 = physical.identity_sha256
            slope_receipt_sha256 = observed.identity_sha256
            previous = layer_receipts[-1]
            previous["prefix_next_w_only_nll"] = float(observed.loss)
            previous["prefix_actual_nll_progress"] = (
                float(previous["current_w_only_nll"]) - float(observed.loss)
            )
            previous_values = tuple(
                float(item) for item in previous["current_w_only_nll_by_request"]
            )
            per_request_progress = tuple(
                source - destination
                for source, destination in zip(
                    previous_values, observed.per_request_values, strict=True
                )
            )
            previous["prefix_actual_nll_progress_by_request"] = list(
                per_request_progress
            )
            previous["negative_request_progress_count"] = sum(
                item < 0.0 for item in per_request_progress
            )
            previous["negative_aggregate_progress"] = bool(
                previous["prefix_actual_nll_progress"] < 0.0
            )
            current_nll = float(observed.loss)
            current_per_request = tuple(observed.per_request_values)
        decision = sequential_quota_decision(
            policy=selected,
            alpha_star=alpha_star,
            current_nll=current_nll,
            endpoint_nll=float(endpoint_nll),
            entry_pi=pi0,
            entry_velocity=velocity0,
            applied_slope=applied_slope,
            layer_ordinal=ordinal,
        )
        alpha_remaining = float(decision["alpha_remaining"])
        suffix_mass = float(decision["suffix_mass"])
        relative_share = float(decision["relative_share"])
        quota = float(decision["quota"])
        velocity = float(decision["velocity"])
        support_exhausted = bool(decision["support_exhausted"])
        if ordinal == 0 and selected is P1R52WriterPolicy.FPIQ:
            if abs(velocity - velocity0[0]) > SIMPLEX_PRIMAL_TOLERANCE:
                raise ODEBFContractError("P1R52-FPiQ layer4 velocity identity differs")
        quota_residual = (
            abs(applied_slope * velocity - quota)
            if selected is P1R52WriterPolicy.FPIQ
            and applied_slope > P1R24_Q_EPSILON
            else None
        )
        if quota_residual is not None and quota_residual > SIMPLEX_PRIMAL_TOLERANCE:
            raise ODEBFContractError("P1R52-FPiQ semantic quota identity differs")
        factor = _factor_for_velocity(
            field,
            velocity,
            step_index=step_index,
            factor_ordinal=ordinal,
        )
        planned[field.weight_name] = factor
        selected_fields.append(field)
        velocities.append(velocity)
        ratio = (
            velocity / velocity0[ordinal]
            if velocity0[ordinal] > P1R24_Q_EPSILON
            else None
        )
        layer_receipts.append(
            {
                "layer": layer,
                "current_residual_norm": float(
                    torch.linalg.norm(
                        (target_state.detach().cpu().float() - current_terminal).double()
                    )
                ),
                "velocity_residual_norm": float(
                    torch.linalg.norm(field.residual.double())
                ),
                "full_residual_h_identity_max_abs": residual_identity,
                "current_w_only_nll": current_nll,
                "current_w_only_nll_by_request": list(current_per_request),
                "frozen_endpoint_nll": float(endpoint_nll),
                "alpha_remaining": alpha_remaining,
                "entry_pi": pi0[ordinal],
                "suffix_pi_mass": suffix_mass,
                "relative_pi_share": relative_share,
                "raw_slope": raw_slope,
                "applied_slope": applied_slope,
                "slope_source": (
                    "ENTRY_ROUTING_PROBLEM_APPLIED_STEP"
                    if ordinal == 0
                    else "CURRENT_LAYER_RAW_COEFFICIENT_TIMES_H"
                ),
                "slope_h_application_count": 1,
                "semantic_quota": quota,
                "selected_velocity": velocity,
                "entry_velocity": velocity0[ordinal],
                "velocity_over_entry": ratio,
                "theta": factor.theta,
                "pi_application_count": 1,
                "physical_h_application_count": 1,
                "second_h_application_count": 0,
                "quota_identity_residual": quota_residual,
                "key_sha256": tensor_sha256(field.key),
                "q_sha256": tensor_sha256(field.q),
                "field_sha256": field.arm_identity(),
                "key_capture_sha256": key_capture_sha256,
                "slope_receipt_sha256": slope_receipt_sha256,
                "factor_energy": float(factor.theta * factor.theta)
                * float(field.factor_frobenius_sq),
                "support_status": (
                    "FROZEN_PI_SUPPORT_EXHAUSTED"
                    if support_exhausted
                    else "AVAILABLE"
                ),
                "nonpositive_slope_zero_velocity": bool(
                    decision["nonpositive_slope_zero_velocity"]
                ),
                "prefix_next_w_only_nll": None,
                "prefix_actual_nll_progress": None,
                "prefix_actual_nll_progress_by_request": None,
                "negative_request_progress_count": None,
                "negative_aggregate_progress": None,
            }
        )
    final_candidate = _merge_factors(current_factors, planned)
    final_subset = {name: base_values[name] for name in final_candidate}
    with CachedBF16FunctionalTrial(
        model, final_subset, final_candidate, row_block=64
    ):
        final_objective = evaluate_scalable_target_new_objective(
            model, objective_plan, target_gradient_required=False
        )
    previous = layer_receipts[-1]
    previous["prefix_next_w_only_nll"] = float(final_objective.loss)
    previous["prefix_actual_nll_progress"] = (
        float(previous["current_w_only_nll"]) - float(final_objective.loss)
    )
    previous_values = tuple(
        float(item) for item in previous["current_w_only_nll_by_request"]
    )
    per_request_progress = tuple(
        source - destination
        for source, destination in zip(
            previous_values, final_objective.per_request_values, strict=True
        )
    )
    previous["prefix_actual_nll_progress_by_request"] = list(per_request_progress)
    previous["negative_request_progress_count"] = sum(
        item < 0.0 for item in per_request_progress
    )
    previous["negative_aggregate_progress"] = bool(
        previous["prefix_actual_nll_progress"] < 0.0
    )
    expected_hashes: dict[str, str] = {}
    for name, factors in sorted(final_candidate.items()):
        effective, stats = assemble_effective_bf16(
            base_values[name].detach().to(
                device=next(model.parameters()).device,
                dtype=torch.bfloat16,
            ),
            factors,
            row_block=64,
        )
        expected_hashes[name] = stats.effective_bf16_sha256
        del effective
    after_inventory = _parameter_inventory_sha256(model)
    after_state = _parameter_state(model, weight_names)
    if before_inventory != after_inventory or before_state != after_state:
        raise ODEBFStateError("P1R52-FPiQ virtual prefix mutated live parameters")
    predicted = math.fsum(
        item["applied_slope"] * item["selected_velocity"]
        for item in layer_receipts
    )
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r52-frozen-pi-sequential-writer/v1",
        "instruction_id": P1R52_FPIQ_INSTRUCTION_ID,
        "method_id": P1R52_FPIQ_METHOD_ID,
        "policy": selected.value,
        "step_index": step_index,
        "layer_order": list(layers),
        "alpha_star": alpha_star,
        "alpha_star_source": "finite_demand.rho_write=[L0-Lz]+",
        "target_step_rho_write_authority_count": 0,
        "entry_nll": float(entry_nll),
        "endpoint_nll": float(endpoint_nll),
        "entry_pi": list(pi0),
        "entry_velocity": list(velocity0),
        "selected_velocity": list(velocities),
        "layers": layer_receipts,
        "predicted_progress": predicted,
        "final_virtual_w_only_nll": float(final_objective.loss),
        "writer_coverage": (
            (float(entry_nll) - float(final_objective.loss))
            / (alpha_star + P1R24_Q_EPSILON)
        ),
        "negative_prefix_aggregate_count": sum(
            bool(item["negative_aggregate_progress"]) for item in layer_receipts
        ),
        "negative_prefix_request_count": sum(
            int(item["negative_request_progress_count"])
            for item in layer_receipts
        ),
        "support_exhausted_count": sum(
            item["support_status"] == "FROZEN_PI_SUPPORT_EXHAUSTED"
            for item in layer_receipts
        ),
        "layer4_capture_count": 0,
        "layer4_added_backward_count": 0,
        "prefix_capture_count": capture_count,
        "prefix_capture_forward_count": capture_forward_count,
        "prefix_capture_processed_tokens": capture_processed_tokens,
        "prefix_capture_padded_tokens": capture_padded_tokens,
        "additional_slope_group_count": slope_group_count,
        "additional_slope_backward_count": slope_backward_count,
        "additional_slope_forward_count": slope_forward_count,
        "additional_slope_processed_tokens": slope_processed_tokens,
        "additional_slope_padded_tokens": slope_padded_tokens,
        "expected_additional_slope_groups_per_outer": 4,
        "final_virtual_objective_forward_count": final_objective.model_forward_count,
        "final_virtual_objective_backward_count": final_objective.backward_count,
        "live_parameter_pointer_version_hash_change_count": 0,
        "final_virtual_bf16_sha256": expected_hashes,
        "pi_application_count_per_layer": 1,
        "physical_h_application_count_per_layer": 1,
        "second_h_application_count": 0,
        "hard_p_h_gate_influence_count": 0,
        "cap_influence_count": 0,
        "retry_count": 0,
        "backtracking_count": 0,
        "contraction_count": 0,
        "functional_veto_count": 0,
        "semantic_debt_carry_count": 0,
        "heldout_writer_decision_access_count": 0,
        "request_layer_router_count": 0,
    }
    if capture_count != 4 or slope_group_count != 4:
        raise ODEBFContractError("P1R52-FPiQ optimized prefix compute differs")
    receipt["identity_sha256"] = canonical_hash(receipt)
    return SequentialWriterResult(
        selected,
        dict(planned),
        tuple(velocities),
        tuple(selected_fields),
        predicted,
        final_objective,
        expected_hashes,
        receipt,
    )


def post_commit_identity(
    result: SequentialWriterResult,
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    observed = materialization.get("effective_bf16_sha256")
    if observed != dict(result.expected_bf16_sha256):
        raise ODEBFContractError("P1R52-FPiQ virtual/physical BF16 identity differs")
    payload = {
        "policy": result.policy.value,
        "final_virtual_bf16_sha256": dict(result.expected_bf16_sha256),
        "post_commit_bf16_sha256": dict(observed),
        "exact_hash_identity": True,
        "physical_commit_count": 1,
        "prefix_physical_commit_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "P1R52_FPIQ_CONTRACT_SHA256",
    "P1R52_FPIQ_INSTRUCTION_ID",
    "P1R52_FPIQ_METHOD_ID",
    "P1R52WriterPolicy",
    "SEQUENTIAL_POLICIES",
    "SequentialWriterResult",
    "plan_sequential_writer",
    "post_commit_identity",
    "sequential_quota_decision",
]
