"""P1R22 accepted-physical-state atomic BG Neutral/Soft runtime."""

from __future__ import annotations

import time
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .accounting import ComputeLedger
from .atomic_runtime_optimization import (
    AcceptedPhysicalStateMaterializer,
    P1R22_INSTRUCTION_ID,
    P1R22_METHOD_ID,
    capture_physical_state,
)
from .common_cold_coordinate import (
    CommonColdScale,
    CommonColdScaleMetric,
    common_cold_bootstrap,
    common_terminal_residual_input,
)
from .contracts import BATCH_SIZE, ODEBFContractError, ODEBFStateError, canonical_hash
from .fixed_e8_soft_routing import FIXED_E8_GRID_COUNT, FIXED_E8_H, FixedE8Clock
from .fixed_e8_soft_routing import structural_contribution_vector
from .functional import WaypointFactor, assemble_effective_bf16, tensor_sha256
from .layer_routing_telemetry import build_layer_routing_telemetry
from .p0_runtime import ModelForwardCounter
from .p1_adaptive import fraction_payload
from .p1_adaptive_runtime import (
    _controller_replay_entry,
    _evaluate_stepwise_state,
    _factor_map,
    _factor_state,
    _functional_trial,
    _merge_factors,
    _parameter_contract_sha256,
    _risk_payload,
)
from .p1_backend import PinnedCovarianceRegistry, build_p1_dynamic_field
from .p1_controller import AcceptedLayerContribution, P1ControllerLock
from .p1_evaluator import load_counterfact_cases_after_freeze
from .p1_replay import (
    OuterEntryPretrainedCache,
    Theta0TeacherCache,
    build_outer_entry_pretrained_cache,
)
from .p1_state import ArmWeightSnapshot, P1Arm, P1HistoryLedger
from .p1_stepwise import StepwiseActionFreeze
from .request_digest import ordered_request_digest_v1
from .sampling import StatelessReplaySchedule
from .strength_preserving_routing import (
    STRENGTH_COVERAGE_EPSILON,
    solve_strength_preserving_routing,
    target_probe_requested_strength,
)
from .target_new_nll import (
    RoutingObjective,
    RoutingObjectiveBatchPlan,
    build_target_new_objective_batch_plan,
    evaluate_routing_objective,
)
from .cold_start_target import cold_lookup_positions, capture_cold_z_base
from .common_coldcoord_fixed_e8_runtime import CommonColdArm
from . import common_coldcoord_fixed_e8_runtime as common
from . import fixed_e8_runtime as legacy


P1R22_SCHEMA = "ode-edit-s05-atomic-runtime-optimization-p1r22"


def _physical_parameter_identity(
    touched: Mapping[str, torch.nn.Parameter],
) -> str:
    return canonical_hash(
        {
            name: {
                "sha256": tensor_sha256(parameter),
                "pointer": int(parameter.data_ptr()),
                "requires_grad": bool(parameter.requires_grad),
                "grad_sha256": (
                    None
                    if parameter.grad is None
                    else tensor_sha256(parameter.grad)
                ),
            }
            for name, parameter in sorted(touched.items())
        }
    )


def _allocation_payload(routing: Any) -> dict[str, Any]:
    values = np.asarray(routing.contribution_share, dtype=np.float64)
    positive = np.maximum(values, 0.0)
    total = float(positive.sum())
    distribution = positive / total if total > 0.0 else positive
    hhi = float(distribution @ distribution)
    return {
        "velocity": [float(item) for item in routing.velocity],
        "contribution_share": values.tolist(),
        "hhi": hhi,
        "effective_layer_count": 0.0 if hhi == 0.0 else 1.0 / hhi,
        "active_layer_count": int(routing.active_layer_count),
        "feasible_allocation_dimension": int(
            routing.feasible_allocation_dimension
        ),
        "selected_bound_fixed_count": int(routing.selected_bound_fixed_count),
    }


def _local_soft_payload(routing: Any) -> dict[str, Any]:
    selected_scores = {item.label: float(item.score) for item in routing.scores}
    differences = dict(routing.risk_difference_soft_minus_neutral)
    if routing.arm.value == "SOFT":
        soft_scores = selected_scores
        neutral_scores = {
            label: value - float(differences[label])
            for label, value in selected_scores.items()
        }
    else:
        neutral_scores = selected_scores
        soft_scores = {
            label: value + float(differences[label])
            for label, value in selected_scores.items()
        }
    selected_risk = max(selected_scores.values(), default=0.0)
    neutral_risk = max(neutral_scores.values(), default=0.0)
    soft_risk = max(soft_scores.values(), default=0.0)
    return {
        "selected_arm": routing.arm.value,
        "selected_normalized_soft_risk": selected_risk,
        "neutral_shadow_normalized_soft_risk": neutral_risk,
        "soft_shadow_normalized_soft_risk": soft_risk,
        "local_soft_regret": soft_risk - neutral_risk,
        "neutral_soft_coefficient_distance": float(
            routing.neutral_soft_coefficient_distance
        ),
        "risk_difference_soft_minus_neutral": [
            [label, float(value)]
            for label, value in routing.risk_difference_soft_minus_neutral
        ],
        "model_forward_count": 0,
        "backward_count": 0,
    }


def _optimized_capacity_payload(
    capture: legacy.FixedE8EntryCapture,
    field: Any,
    increment: Mapping[str, WaypointFactor],
    *,
    problem: Any,
    routing: Any,
    signed: Any,
    step_index: int,
    state_sha256: str,
    target_sha256: str,
    factor_state_sha256: str,
    candidate_sha256: str,
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    """Build routing/capacity telemetry from the authoritative physical copy.

    No factor endpoint is assembled here.  Step and cumulative BF16 energy,
    plus effective hashes, come from the one accepted materialization.
    """

    requested: list[float] = []
    realized: list[float] = []
    cumulative: list[float] = []
    effective: dict[str, str] = {}
    for layer in field.layers:
        name = layer.weight_name
        theta = float(increment[name].theta)
        requested.append(theta**2 * layer.factor_frobenius_sq)
        realized.append(float(materialization["realized_bf16_step_energy"][name]))
        cumulative.append(float(materialization["cumulative_bf16_capacity"][name]))
        effective[name] = str(materialization["effective_bf16_sha256"][name])
    velocity = np.asarray(routing.velocity, dtype=np.float64)
    applied = np.asarray(routing.applied_coefficient, dtype=np.float64)
    raw_slopes = np.asarray(signed.signed_progress, dtype=np.float64)
    structural_h = structural_contribution_vector(problem.historical, velocity)
    structural_p = structural_contribution_vector(problem.pretrained, velocity)
    trust = velocity * (problem.trust_metric @ velocity)
    layer_routing = build_layer_routing_telemetry(
        step_index=step_index,
        stage="ACCEPTED_PHYSICAL",
        layer_ids=common.COMMON_COLD_LAYER_ORDER,
        signed_efficiency=raw_slopes,
        raw_velocity=velocity,
        bf_velocity=velocity,
        applied_coefficient=applied,
        active_direction_mask=raw_slopes > 0.0,
        raw_cap_bound_mask=np.isclose(
            velocity, problem.layer_caps, rtol=0.0, atol=1.0e-12
        ),
        bf_cap_bound_mask=np.isclose(
            velocity, problem.layer_caps, rtol=0.0, atol=1.0e-12
        ),
        raw_zero_bound_mask=np.isclose(velocity, 0.0, rtol=0.0, atol=1.0e-12),
        bf_zero_bound_mask=np.isclose(velocity, 0.0, rtol=0.0, atol=1.0e-12),
        predicted_progress_contribution=raw_slopes * applied,
        prequantized_update_energy=requested,
        realized_bf16_update_energy=realized,
        cumulative_bf16_capacity=cumulative,
        bf16_capacity_contribution=realized,
        structural_h_contribution=structural_h,
        structural_p_contribution=structural_p,
        trust_contribution=trust,
        field_sha256=field.identity_sha256,
        state_sha256=state_sha256,
        target_z_sha256=target_sha256,
        factor_state_sha256=factor_state_sha256,
        raw_solver_certificate_sha256=canonical_hash(
            [item.raw_free_payload() for item in routing.certificates]
        ),
        bf_solver_certificate_sha256=routing.identity_sha256,
        candidate_sha256=candidate_sha256,
    )
    payload = {
        "layer_order": list(common.COMMON_COLD_LAYER_ORDER),
        "velocity": list(routing.velocity),
        "applied_coefficient": list(routing.applied_coefficient),
        "requested_frobenius_energy": requested,
        "realized_bf16_step_energy": realized,
        "cumulative_bf16_capacity": cumulative,
        "structural_h_contribution": structural_h.tolist(),
        "structural_p_contribution": structural_p.tolist(),
        "trust_contribution": trust.tolist(),
        "effective_bf16_sha256": dict(sorted(effective.items())),
        "streaming_receipts": {},
        "layer_routing": layer_routing,
        "h_applied_exactly_once": True,
        "accepted_materialization_reuse": True,
        "additional_dense_assembly_count": 0,
        "additional_full_weight_hash_count": 0,
        "dense_fp64_full_delta_live": 0,
        "dense_fp32_full_delta_live": 0,
        "effective_bf16_weight_peak_live": 1,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _conformance_field_receipt(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    arm: CommonColdArm,
    step_index: int,
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    metric: CommonColdScaleMetric,
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    lock: P1ControllerLock,
    history: P1HistoryLedger,
    accepted_by_layer: Mapping[int, Sequence[AcceptedLayerContribution]],
    signed_batched: Any,
    field_batched: Any,
    inventory: Any,
    target_velocity_batched: torch.Tensor,
    target_receipt_batched: Mapping[str, Any],
    routing_batched: Any,
    lookup_positions: Sequence[int],
    target_layer_name: str,
    request_microbatch_size: int,
    objective_batch_plan: RoutingObjectiveBatchPlan,
) -> dict[str, Any]:
    """Model-backed C1 comparison at one already accepted physical state."""

    audit_ledger = ComputeLedger()
    solve_history = common._history_keys(
        history, common.COMMON_COLD_LAYER_ORDER, risk=False
    )
    risk_history = common._history_keys(
        history, common.COMMON_COLD_LAYER_ORDER, risk=True
    )
    residual_input = common_terminal_residual_input(
        current_target,
        current_terminal,
        ordered_request_digest_v1(
            [str(item["request_sha256"]) for item in requests]
        ),
    )
    field_reference = build_p1_dynamic_field(
        model,
        tokenizer,
        requests,
        hparams,
        projector,
        contexts,
        target_state=current_target,
        accepted_waypoint=step_index,
        cumulative_factors_by_weight={},
        history_solve_keys_by_layer=solve_history,
        history_risk_keys_by_layer=risk_history,
        covariance_registry=covariance_registry,
        projector_sha256=projector_sha256,
        residual_tolerance=lock.residual_tolerance,
        ledger=audit_ledger,
        residual_policy=common.SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1,
        allow_zero_capacity=False,
        shared_terminal_residual=residual_input,
        allow_inner_empty_cache=False,
    )
    field_pairs = zip(
        field_batched.layers, field_reference.layers, strict=True
    )
    key_q_parity = []
    for optimized, reference in field_pairs:
        record = {
            "layer": int(optimized.layer),
            "key_exact": tensor_sha256(optimized.key)
            == tensor_sha256(reference.key),
            "q_exact": tensor_sha256(optimized.q)
            == tensor_sha256(reference.q),
            "residual_exact": tensor_sha256(optimized.residual)
            == tensor_sha256(reference.residual),
        }
        if not all(value for key, value in record.items() if key != "layer"):
            raise ODEBFContractError("P1R22 C1 key/q/residual parity differs")
        key_q_parity.append(record)
    signed_reference = legacy._fixed_e8_signed_progress_gradient(
        model,
        tokenizer,
        requests,
        field_reference,
        cumulative_factors_by_weight={},
        contexts=contexts,
        ledger=audit_ledger,
        request_microbatch_size=1,
    )
    slopes_reference = np.asarray(signed_reference.signed_progress, dtype=np.float64)
    slopes_batched = np.asarray(signed_batched.signed_progress, dtype=np.float64)
    slope_delta = slopes_batched - slopes_reference
    slope_relative_l2 = float(
        np.linalg.norm(slope_delta)
        / max(np.linalg.norm(slopes_reference), 1.0e-30)
    )
    nonnegligible = np.abs(slopes_reference) > 1.0e-8
    if (
        slope_relative_l2 > 1.0e-4
        or not np.array_equal(
            np.signbit(slopes_batched[nonnegligible]),
            np.signbit(slopes_reference[nonnegligible]),
        )
        or tuple(np.argsort(slopes_batched, kind="stable"))
        != tuple(np.argsort(slopes_reference, kind="stable"))
    ):
        raise ODEBFContractError("P1R22 C1 nohook slope parity differs")
    singleton_objective = evaluate_routing_objective(
        model,
        tokenizer,
        requests,
        objective=RoutingObjective.TARGET_NEW_NLL,
        contexts=contexts,
        request_microbatch_size=1,
    )
    batched_objective = evaluate_routing_objective(
        model,
        tokenizer,
        requests,
        objective=RoutingObjective.TARGET_NEW_NLL,
        contexts=contexts,
        request_microbatch_size=request_microbatch_size,
        batch_plan=objective_batch_plan,
    )
    per_request_delta = (
        batched_objective.per_request_values.detach().cpu().double()
        - singleton_objective.per_request_values.detach().cpu().double()
    )
    nll_max_abs = float(per_request_delta.abs().max())
    mean_nll_abs = abs(
        float(batched_objective.loss.detach().cpu())
        - float(singleton_objective.loss.detach().cpu())
    )
    if (
        nll_max_abs > 1.0e-5
        or mean_nll_abs > 1.0e-5
        or batched_objective.target_span_sha256
        != singleton_objective.target_span_sha256
    ):
        raise ODEBFContractError("P1R22 C1 objective batching parity differs")
    target_velocity_reference, target_receipt_reference = (
        common.write_aware_common_target_velocity(
            model,
            tokenizer,
            requests,
            contexts,
            target_layer_name=target_layer_name,
            lookup_positions=lookup_positions,
            field=field_reference,
            velocity_coefficients=(0.0,) * len(common.COMMON_COLD_LAYER_ORDER),
            cumulative_factors_by_weight={},
            metric=metric,
            ledger=audit_ledger,
            request_microbatch_size=1,
        )
    )
    target_delta = (
        target_velocity_batched.detach().cpu().double()
        - target_velocity_reference.detach().cpu().double()
    )
    target_reference_norm = float(target_velocity_reference.double().norm())
    target_relative_l2 = float(target_delta.norm()) / max(
        target_reference_norm, 1.0e-30
    )
    target_cosine = float(
        torch.nn.functional.cosine_similarity(
            target_velocity_batched.detach().cpu().double().flatten(),
            target_velocity_reference.detach().cpu().double().flatten(),
            dim=0,
        )
    )
    if target_cosine < 0.99999 or target_relative_l2 > 1.0e-4:
        raise ODEBFContractError("P1R22 C1 target gradient parity differs")
    reference_problem = legacy._build_fixed_e8_problem(
        field_reference,
        signed_reference,
        accepted_by_layer=accepted_by_layer,
        committed_load_by_layer=history.cumulative_load(),
        lock=lock,
        current_history_action_by_layer=common._history_actions(
            field_reference, history
        ),
    )
    alpha_reference, _ = target_probe_requested_strength(
        target_receipt_reference
    )
    routing_reference = solve_strength_preserving_routing(
        reference_problem.problem,
        inventory,
        arm=arm.routing_arm,
        alpha_req=alpha_reference,
    )
    routing_max_abs = float(
        np.max(
            np.abs(
                np.asarray(routing_batched.velocity)
                - np.asarray(routing_reference.velocity)
            )
        )
    )
    alpha_apply_abs = abs(
        float(routing_batched.alpha_apply)
        - float(routing_reference.alpha_apply)
    )
    alpha_max_abs = abs(
        float(routing_batched.alpha_max) - float(routing_reference.alpha_max)
    )
    if (
        routing_max_abs > 1.0e-4
        or alpha_apply_abs > 1.0e-5
        or alpha_max_abs > 1.0e-5
    ):
        raise ODEBFContractError("P1R22 C1 routing parity differs")
    payload = {
        "schema": f"{P1R22_SCHEMA}-c1-field-parity/v1",
        "step_index": step_index,
        "request_microbatch_size": request_microbatch_size,
        "key_q_residual": key_q_parity,
        "nll_per_request_max_abs": nll_max_abs,
        "nll_mean_abs": mean_nll_abs,
        "slope_relative_l2": slope_relative_l2,
        "slope_rank_exact": True,
        "target_gradient_cosine": target_cosine,
        "target_gradient_relative_l2": target_relative_l2,
        "routing_coefficient_max_abs": routing_max_abs,
        "alpha_apply_abs": alpha_apply_abs,
        "alpha_max_abs": alpha_max_abs,
        "batched_target_receipt_sha256": target_receipt_batched[
            "identity_sha256"
        ],
        "reference_target_receipt_sha256": target_receipt_reference[
            "identity_sha256"
        ],
        "audit_compute": audit_ledger.raw_free_payload(),
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _conformance_materialization_receipt(
    model: torch.nn.Module,
    base_values: Mapping[str, torch.Tensor],
    candidate_factors: Mapping[str, Sequence[WaypointFactor]],
    materialization: Mapping[str, Any],
    *,
    step_index: int,
) -> dict[str, Any]:
    parameters = dict(model.named_parameters())
    parity: dict[str, bool] = {}
    for name in sorted(base_values):
        expected, stats = assemble_effective_bf16(
            base_values[name], candidate_factors[name], row_block=64
        )
        parity[name] = bool(
            stats.effective_bf16_sha256
            == materialization["effective_bf16_sha256"][name]
            == tensor_sha256(parameters[name])
            == tensor_sha256(expected)
        )
        del expected
    if not all(parity.values()):
        raise ODEBFContractError("P1R22 C0 materialization parity differs")
    payload = {
        "schema": f"{P1R22_SCHEMA}-c0-materialization-parity/v1",
        "step_index": step_index,
        "entry_relative_rounding": True,
        "incremental_bf16_accumulation_count": 0,
        "effective_candidate_sha_exact": parity,
        "parameter_pointer_inventory_preserved": True,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _run_arm(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    arm: CommonColdArm,
    bootstrap_target: torch.Tensor,
    z_base: torch.Tensor,
    metric: CommonColdScaleMetric,
    lookup_positions: Sequence[int],
    target_layer_name: str,
    capture: legacy.FixedE8EntryCapture,
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    lock: P1ControllerLock,
    arm_state: Any,
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: StatelessReplaySchedule,
    outer_entry_p_cache: OuterEntryPretrainedCache,
    theta0_cache: Theta0TeacherCache,
    touched: Mapping[str, torch.nn.Parameter],
    base_values: Mapping[str, torch.Tensor],
    recorder: legacy.FixedE8ReceiptRecorder,
    stages: Any,
    request_microbatch_size: int,
    objective_batch_plan: RoutingObjectiveBatchPlan,
    conformance_steps: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    if arm not in (CommonColdArm.BG_NEUTRAL, CommonColdArm.BG_SOFT):
        raise ODEBFContractError("P1R22 live arm differs")
    clock = FixedE8Clock()
    history = arm_state.history
    ledger = arm_state.ledger
    materializer = AcceptedPhysicalStateMaterializer(model, base_values)
    current_target = bootstrap_target.detach().cpu().float().clone().contiguous()
    current_factors: dict[str, tuple[WaypointFactor, ...]] = {}
    accepted_by_layer: dict[int, list[AcceptedLayerContribution]] = {
        layer: [] for layer in common.COMMON_COLD_LAYER_ORDER
    }
    physical = capture_physical_state(
        model, tokenizer, requests, hparams, contexts, lookup_positions
    )
    if tensor_sha256(physical.terminal_z) != tensor_sha256(z_base):
        raise ODEBFContractError("P1R22 W0 state-capture parity differs")
    current_terminal = physical.terminal_z.clone()
    accepted: list[dict[str, Any]] = []
    delayed_hashes: list[str] = []
    pending: dict[str, Any] | None = None
    initial_identity: dict[str, Any] | None = None
    terminal_objective_count = 0
    c0_receipts: list[dict[str, Any]] = []
    c1_receipts: list[dict[str, Any]] = []
    edit_started = time.perf_counter()
    restored = False
    try:
        for step_index in range(FIXED_E8_GRID_COUNT):
            point = clock.begin_field()
            replay_entry = _controller_replay_entry(
                model,
                tokenizer,
                alias=alias,
                arm_state=arm_state,
                sample_waypoint=step_index + 1,
                factors={},
                request_by_sha256=request_by_sha256,
                population_by_sha256=population_by_sha256,
                schedule=schedule,
                outer_entry_p_cache=outer_entry_p_cache,
            )
            field_started = time.perf_counter()
            (
                field,
                signed,
                built,
                inventory,
                routing,
                target_velocity,
                target_receipt,
                field_receipt,
                probe,
                _slope_comparison,
            ) = common._build_common_field(
                model,
                tokenizer,
                requests,
                alias=alias,
                arm=arm,
                metric=metric,
                step_index=step_index,
                current_factors=current_factors,
                current_target=current_target,
                current_terminal=current_terminal,
                capture=capture,
                hparams=hparams,
                projector=projector,
                contexts=contexts,
                covariance_registry=covariance_registry,
                projector_sha256=projector_sha256,
                lock=lock,
                history=history,
                accepted_by_layer=accepted_by_layer,
                ledger=ledger,
                replay_entry=replay_entry,
                theta0_cache=theta0_cache,
                touched=touched,
                schedule=schedule,
                recorder=recorder,
                lookup_positions=lookup_positions,
                target_layer_name=target_layer_name,
                typed_zero_positive=True,
                strength_preserving=True,
                atomic_runtime_optimized=True,
                physical_capture=physical,
                trial_entry_weights=base_values,
                request_microbatch_size=request_microbatch_size,
                objective_batch_plan=objective_batch_plan,
            )
            ledger.add_time(
                "p1r22_field_target_route",
                wall_seconds=time.perf_counter() - field_started,
            )
            if target_velocity is None or signed.mean_objective_value is None:
                raise ODEBFContractError("P1R22 field did not produce a transition")
            if step_index in conformance_steps:
                c1_receipts.append(
                    _conformance_field_receipt(
                        model,
                        tokenizer,
                        requests,
                        arm=arm,
                        step_index=step_index,
                        current_target=current_target,
                        current_terminal=current_terminal,
                        metric=metric,
                        hparams=hparams,
                        projector=projector,
                        contexts=contexts,
                        covariance_registry=covariance_registry,
                        projector_sha256=projector_sha256,
                        lock=lock,
                        history=history,
                        accepted_by_layer=accepted_by_layer,
                        signed_batched=signed,
                        field_batched=field,
                        inventory=inventory,
                        target_velocity_batched=target_velocity,
                        target_receipt_batched=target_receipt,
                        routing_batched=routing,
                        lookup_positions=lookup_positions,
                        target_layer_name=target_layer_name,
                        request_microbatch_size=request_microbatch_size,
                        objective_batch_plan=objective_batch_plan,
                    )
                )
            current_nll = float(signed.mean_objective_value)
            if pending is not None:
                actual = float(pending["source_nll"] - current_nll)
                delayed = {
                    "schema": f"{P1R22_SCHEMA}-delayed-progress/v1",
                    "transition_index": int(pending["step_index"]) + 1,
                    "source_mean_target_new_nll": float(pending["source_nll"]),
                    "next_field_mean_target_new_nll": current_nll,
                    "predicted": float(pending["predicted"]),
                    "actual": actual,
                    "completion": "NEXT_FIELD_NOHOOK_NLL_REUSE",
                    "realization_ratio": actual
                    / (
                        float(pending["alpha_apply"])
                        + STRENGTH_COVERAGE_EPSILON
                    ),
                    "alpha_apply": float(pending["alpha_apply"]),
                    "candidate_objective_inner_count": 0,
                    "next_field_receipt_sha256": field_receipt.identity_sha256,
                }
                delayed["identity_sha256"] = canonical_hash(delayed)
                delayed_hashes.append(
                    recorder.write_once(
                        recorder.root
                        / f"delayed-progress-k{int(pending['step_index']) + 1}.json",
                        delayed,
                    )
                )
            if step_index == 0:
                initial_identity = {
                    "bootstrap_target_sha256": tensor_sha256(bootstrap_target),
                    "physical_capture_sha256": physical.identity_sha256,
                    "field_semantic_sha256": field_receipt.field_semantic_sha256,
                    "signed_progress": [float(item) for item in signed.signed_progress],
                    "signed_progress_sha256": canonical_hash(
                        [float(item) for item in signed.signed_progress]
                    ),
                    "alpha_req": float(routing.alpha_req),
                    "alpha_max": float(routing.alpha_max),
                    "alpha_apply": float(routing.alpha_apply),
                    "problem_sha256": built.problem.identity(),
                    "functional_inventory_sha256": inventory.raw_free_payload()[
                        "identity_sha256"
                    ],
                }
                initial_identity["identity_sha256"] = canonical_hash(initial_identity)
            increment = legacy.fixed_e8_waypoint_factors(
                field, routing.velocity, step_index=step_index
            )
            candidate_factors = common._merge_factors(current_factors, increment)
            target_trial = (
                current_target
                + float(FIXED_E8_H)
                * target_velocity.detach().to(device="cpu", dtype=torch.float32)
            ).contiguous()
            materialize_started = time.perf_counter()
            materialization = materializer.materialize(
                candidate_factors,
                transition_index=step_index + 1,
            )
            candidate_physical = capture_physical_state(
                model, tokenizer, requests, hparams, contexts, lookup_positions
            )
            if step_index in conformance_steps:
                c0_receipts.append(
                    _conformance_materialization_receipt(
                        model,
                        base_values,
                        candidate_factors,
                        materialization,
                        step_index=step_index,
                    )
                )
            ledger.add_time(
                "p1r22_materialize_and_capture",
                wall_seconds=time.perf_counter() - materialize_started,
            )
            candidate_terminal = candidate_physical.terminal_z
            candidate_residual = common_terminal_residual_input(
                target_trial,
                candidate_terminal,
                field.request_order_sha256,
            ).residual
            predicted = float(
                built.problem.signed_progress @ np.asarray(routing.velocity)
            )
            progress: dict[str, Any] = {
                "predicted": predicted,
                "actual": None,
                "completion": "DELAYED_TO_NEXT_FIELD",
                "alpha_req": float(routing.alpha_req),
                "alpha_max": float(routing.alpha_max),
                "alpha_apply": float(routing.alpha_apply),
                "coverage": float(routing.coverage),
                "equality_residual": float(routing.equality_residual),
                "candidate_objective_inner_count": 0,
                "scientific_rejection_count": 0,
            }
            if step_index == FIXED_E8_GRID_COUNT - 1:
                terminal_objective = evaluate_routing_objective(
                    model,
                    tokenizer,
                    requests,
                    objective=RoutingObjective.TARGET_NEW_NLL,
                    contexts=contexts,
                    request_microbatch_size=request_microbatch_size,
                )
                terminal_objective_count += 1
                terminal_nll = float(terminal_objective.loss.detach().cpu())
                actual = current_nll - terminal_nll
                progress.update(
                    {
                        "actual": actual,
                        "realization_ratio": actual
                        / (float(routing.alpha_apply) + STRENGTH_COVERAGE_EPSILON),
                        "completion": "TERMINAL_W_ONLY_OBJECTIVE_ONCE",
                    }
                )
            structural_h = built.problem.historical.value(
                np.asarray(routing.velocity)
            )
            structural_p = built.problem.pretrained.value(
                np.asarray(routing.velocity)
            )
            capacity = _optimized_capacity_payload(
                capture,
                field,
                increment,
                problem=built.problem,
                routing=routing,
                signed=signed,
                step_index=step_index,
                state_sha256=_parameter_contract_sha256(touched),
                target_sha256=tensor_sha256(current_target),
                factor_state_sha256=_factor_state(
                    capture.entry_sha256, current_factors, current_target
                ),
                candidate_sha256=_factor_state(
                    capture.entry_sha256, candidate_factors, target_trial
                ),
                materialization=materialization,
            )
            target_realization = legacy.fixed_e8_target_write_realization(
                current_target,
                target_trial,
                current_terminal,
                candidate_terminal,
            )
            payload = {
                "schema": f"{P1R22_SCHEMA}-accepted-transition/v1",
                "arm": arm.value,
                "accepted_index": step_index + 1,
                "tau_before": fraction_payload(point.tau_before),
                "tau_after": fraction_payload(point.tau_after),
                "field_receipt_sha256": field_receipt.identity_sha256,
                "physical_capture_sha256": physical.identity_sha256,
                "next_physical_capture_sha256": candidate_physical.identity_sha256,
                "authoritative_slope": "PHYSICAL_W_ONLY_NOHOOK",
                "overlay_slope_model_forward_count": 0,
                "overlay_slope_backward_count": 0,
                "overlay_slope_decision_influence_count": 0,
                "routing": routing.raw_free_payload(),
                "progress": progress,
                "structural_functional": {
                    "historical": {
                        "value": float(structural_h),
                        "status": "INACTIVE_EMPTY_HISTORY",
                        "decision_influence_count": 0,
                    },
                    "pretrained": {
                        "value": float(structural_p),
                        "status": "OBSERVATION_ONLY",
                        "decision_influence_count": 0,
                    },
                    "functional_h": {
                        "status": "NOT_EVALUATED_INNER_ATOMIC_H0",
                        "decision_influence_count": 0,
                    },
                    "functional_p": {
                        "status": "ROUTING_BASIS_ONLY_NO_SELECTED_ENDPOINT_REPLAY",
                        "decision_influence_count": 0,
                    },
                },
                "functional_basis": inventory.raw_free_payload(),
                "functional_probe_receipt_sha256": probe["identity_sha256"],
                "local_soft_objective": _local_soft_payload(routing),
                "allocation": _allocation_payload(routing),
                "target_velocity": target_receipt,
                "target_write_realization": target_realization,
                "materialization": materialization,
                "candidate_residual_sha256": tensor_sha256(candidate_residual),
                "capacity": capacity,
                "inner_step_heldout_evaluation_count": 0,
                "inner_step_functional_selected_endpoint_replay_count": 0,
                "first_hit_evaluation_count": 0,
                "retry_count": 0,
                "backtracking_count": 0,
            }
            payload["identity_sha256"] = canonical_hash(payload)
            receipt_sha = recorder.accepted(payload)
            transition = legacy._fixed_e8_advance_grid_transition(
                ledger, clock, point, scientific_observation={}
            )
            recorder.transition(
                {
                    **transition,
                    "accepted_receipt_sha256": receipt_sha,
                    "materialization_receipt_sha256": materialization[
                        "identity_sha256"
                    ],
                }
            )
            for layer in field.layers:
                accepted_by_layer[layer.layer].append(
                    AcceptedLayerContribution.from_field(
                        layer,
                        increment[layer.weight_name],
                        history_action=layer.history_action,
                    )
                )
            ledger.record_accepted_step(
                accepted_dt=float(FIXED_E8_H), completed_k_total=step_index + 1
            )
            accepted.append(payload)
            pending = (
                None
                if step_index == FIXED_E8_GRID_COUNT - 1
                else {
                    "step_index": step_index,
                    "source_nll": current_nll,
                    "predicted": predicted,
                    "alpha_apply": float(routing.alpha_apply),
                }
            )
            current_factors = _factor_map(candidate_factors)
            current_target = target_trial
            current_terminal = candidate_terminal.clone()
            physical = candidate_physical
            stages.record(
                f"post_{arm.value.lower().replace('-', '_')}_step_{step_index + 1}",
                {
                    "arm": arm.value,
                    "step": step_index + 1,
                    "tau": float(point.tau_after),
                    "field_receipt_sha256": field_receipt.identity_sha256,
                    "accepted_receipt_sha256": receipt_sha,
                },
            )
        if clock.tau != Fraction(1, 1) or len(accepted) != FIXED_E8_GRID_COUNT:
            raise ODEBFStateError("P1R22 K8/tau1 invariant differs")
        if pending is not None or len(delayed_hashes) != 7:
            raise ODEBFStateError("P1R22 delayed progress accounting differs")
        edit_core_time = time.perf_counter() - edit_started
        terminal_functional_started = time.perf_counter()
        terminal_replay_entry = _controller_replay_entry(
            model,
            tokenizer,
            alias=alias,
            arm_state=arm_state,
            sample_waypoint=FIXED_E8_GRID_COUNT,
            factors={},
            request_by_sha256=request_by_sha256,
            population_by_sha256=population_by_sha256,
            schedule=schedule,
            outer_entry_p_cache=outer_entry_p_cache,
        )
        terminal_functional = _functional_trial(
            model,
            tokenizer,
            alias=alias,
            entry=terminal_replay_entry,
            theta0_cache=theta0_cache,
            factors={},
            lock=lock,
            ledger=ledger,
        )
        terminal_functional_time = time.perf_counter() - terminal_functional_started
        result: dict[str, Any] = {
            "arm": arm.value,
            "status": "ATOMIC_OPTIMIZED_K8_COMPLETE",
            "accepted_update_count": len(accepted),
            "tau_final": float(clock.tau),
            "initial_identity": initial_identity,
            "accepted": accepted,
            "delayed_progress_receipt_sha256": delayed_hashes,
            "terminal_target_sha256": tensor_sha256(current_target),
            "terminal_physical_capture_sha256": physical.identity_sha256,
            "terminal_factors": _factor_map(current_factors),
            "terminal_functional": {
                "historical": _risk_payload(terminal_functional.historical),
                "pretrained": _risk_payload(terminal_functional.pretrained),
                "receipt_sha256": terminal_functional.identity_sha256,
            },
            "materializer": materializer.raw_free_payload(),
            "candidate_objective_inner_k_count": 0,
            "candidate_progress_delayed_reuse_count": 7,
            "terminal_w_only_objective_count": terminal_objective_count,
            "key_capture_forward_count": 9,
            "inner_field_empty_cache_count": 0,
            "overlay_slope_count": 0,
            "edit_core_time_seconds": edit_core_time,
            "terminal_functional_probe_time_seconds": terminal_functional_time,
            "compute": ledger.raw_free_payload(),
            "c0_materialization_parity": c0_receipts,
            "c1_field_parity": c1_receipts,
        }
        result["rollout_sha256"] = canonical_hash(
            {key: value for key, value in result.items() if key != "terminal_factors"}
        )
        restore = materializer.restore()
        restored = True
        result["w0_restore"] = restore
        return result
    finally:
        if not restored:
            materializer.restore()


def _public_rollout(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "terminal_factors"}


def run_p1r22_atomic_optimization(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    destination: Path,
    raw_root: Path,
    stages: Any,
    source_head: str,
    requests: Sequence[Mapping[str, Any]],
    stream: Mapping[str, Any],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    controller_lock: P1ControllerLock,
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: StatelessReplaySchedule,
    theta0_cache: Theta0TeacherCache,
    dataset_path: Path,
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    artifact_guard: Any,
    artifact_receipt: Any,
    numerical_sha256: str,
    context_sha256: str,
    cuda_runtime_receipt: Mapping[str, Any],
    job_ledger: ComputeLedger,
    write_once: Any,
    request_microbatch_size: int,
    optimization_lock: Mapping[str, Any],
    optimization_lock_sha256: str,
    conformance_only: bool = False,
) -> dict[str, Any]:
    from .p1_runtime import ArmRuntimeState, _entry_parameter_snapshot_sha256

    started = time.perf_counter()
    request_order = ordered_request_digest_v1(
        [str(item["request_sha256"]) for item in requests]
    )
    if (
        len(requests) != BATCH_SIZE
        or request_order != stream["batch_ordered_request_digest_v1"][0]
        or optimization_lock["stage_a_seal_root"]
        != "3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628"
    ):
        raise ODEBFContractError("P1R22 seal/order contract differs")
    base_contract = _physical_parameter_identity(touched)
    pre_counter = ModelForwardCounter(model, job_ledger)
    try:
        z_base = capture_cold_z_base(model, tokenizer, requests, hparams)
        lookup_positions = cold_lookup_positions(
            tokenizer, requests, contexts, fact_token_strategy=hparams.fact_token
        )
    finally:
        pre_counter.close()
    target_layer_name = hparams.layer_module_tmp.format(int(hparams.layers[-1]))
    objective_batch_plan = build_target_new_objective_batch_plan(
        model,
        tokenizer,
        requests,
        contexts=contexts,
        request_microbatch_size=request_microbatch_size,
    )
    objective_batch_plan_sha = write_once(
        raw_root / "atomic-runtime" / "objective-batch-plan.json",
        objective_batch_plan.raw_free_payload(),
    )
    metric = CommonColdScaleMetric.from_z_base(
        z_base, request_order, CommonColdScale.BATCH_GLOBAL
    )
    bootstrap_counter = ModelForwardCounter(model, job_ledger)
    try:
        bg_target, bg_bootstrap = common_cold_bootstrap(
            model,
            tokenizer,
            requests,
            contexts,
            target_layer_name=target_layer_name,
            lookup_positions=lookup_positions,
            z_base=z_base,
            metric=metric,
            ledger=job_ledger,
        )
    finally:
        bootstrap_counter.close()
    bootstrap_sha = write_once(
        raw_root / "atomic-runtime" / "bootstrap-bg.json", bg_bootstrap
    )
    outer_population = tuple(
        population_by_sha256[item] for item in theta0_cache.request_order
    )
    outer_snapshot = _entry_parameter_snapshot_sha256(
        model, dict(base_receipt.parameter_sha256)
    )
    outer_counter = ModelForwardCounter(model, job_ledger)
    try:
        outer_cache = build_outer_entry_pretrained_cache(
            model,
            tokenizer,
            outer_population,
            theta0_cache,
            outer_entry_snapshot_sha256=outer_snapshot,
        )
    finally:
        outer_counter.close()
    capture = legacy.FixedE8EntryCapture(
        {name: value.detach().cpu().clone() for name, value in base_values.items()},
        dict(base_receipt.parameter_sha256),
    )
    rollouts: dict[str, dict[str, Any]] = {}
    arm_states: dict[str, Any] = {}
    live_arms = (
        (CommonColdArm.BG_NEUTRAL,)
        if conformance_only
        else (CommonColdArm.BG_NEUTRAL, CommonColdArm.BG_SOFT)
    )
    for arm in live_arms:
        if _physical_parameter_identity(touched) != base_contract:
            raise ODEBFStateError("P1R22 arm W0 entry differs")
        arm_state = ArmRuntimeState(
            P1Arm.R_BF,
            P1HistoryLedger(
                layer_order=common.COMMON_COLD_LAYER_ORDER, maximum_records=40
            ),
            ComputeLedger(),
            ArmWeightSnapshot(
                P1Arm.R_BF,
                0,
                base_receipt.parameter_sha256,
                canonical_hash({"arm": arm.value, "weights": base_receipt.parameter_sha256}),
            ),
            dict(base_values),
        )
        recorder = legacy.FixedE8ReceiptRecorder(raw_root, arm, write_once)
        counter = ModelForwardCounter(model, arm_state.ledger)
        try:
            rollouts[arm.value] = _run_arm(
                model,
                tokenizer,
                requests,
                alias=alias,
                arm=arm,
                bootstrap_target=bg_target,
                z_base=z_base,
                metric=metric,
                lookup_positions=lookup_positions,
                target_layer_name=target_layer_name,
                capture=capture,
                hparams=hparams,
                projector=projector,
                contexts=contexts,
                covariance_registry=covariance_registry,
                projector_sha256=projector_sha256,
                lock=controller_lock,
                arm_state=arm_state,
                request_by_sha256=request_by_sha256,
                population_by_sha256=population_by_sha256,
                schedule=schedule,
                outer_entry_p_cache=outer_cache,
                theta0_cache=theta0_cache,
                touched=touched,
                base_values=base_values,
                recorder=recorder,
                stages=stages,
                request_microbatch_size=request_microbatch_size,
                objective_batch_plan=objective_batch_plan,
                conformance_steps=(
                    frozenset((1, 6)) if conformance_only else frozenset()
                ),
            )
            arm_states[arm.value] = arm_state
        finally:
            counter.close()
        if _physical_parameter_identity(touched) != base_contract:
            raise ODEBFStateError("P1R22 arm W0 restore differs")
    if conformance_only:
        rollout = rollouts[CommonColdArm.BG_NEUTRAL.value]
        if (
            len(rollout["c0_materialization_parity"]) != 2
            or len(rollout["c1_field_parity"]) != 2
        ):
            raise ODEBFContractError("P1R22 C0/C1 receipt count differs")
        conformance = {
            "schema": f"{P1R22_SCHEMA}-c0c1-terminal/v1",
            "instruction_id": P1R22_INSTRUCTION_ID,
            "alias": alias,
            "source_head": source_head,
            "optimization_lock_sha256": optimization_lock_sha256,
            "request_microbatch_size": request_microbatch_size,
            "objective_batch_plan_sha256": objective_batch_plan_sha,
            "steps": [1, 6],
            "c0": rollout["c0_materialization_parity"],
            "c1": rollout["c1_field_parity"],
            "rollout_sha256": rollout["rollout_sha256"],
            "final_w0_restored": True,
            "heldout_evaluation_count": 0,
            "model_forward_result_action": "CONFORMANCE_ONLY_NO_SCIENTIFIC_ENDPOINT",
        }
        conformance["identity_sha256"] = canonical_hash(conformance)
        terminal_sha = write_once(destination / "terminal.json", conformance)
        manifest_sha = write_once(
            destination / "manifest.json",
            {
                "schema": f"{P1R22_SCHEMA}-c0c1-manifest/v1",
                "terminal_sha256": terminal_sha,
                "source_head": source_head,
                "optimization_lock_sha256": optimization_lock_sha256,
            },
        )
        return {
            "status": "P1R22_C0_C1_CONFORMANCE_PASS",
            "terminal_sha256": terminal_sha,
            "manifest_sha256": manifest_sha,
        }
    left = dict(rollouts[CommonColdArm.BG_NEUTRAL.value]["initial_identity"])
    right = dict(rollouts[CommonColdArm.BG_SOFT.value]["initial_identity"])
    if left != right:
        raise ODEBFContractError("P1R22 Neutral/Soft initial state differs")
    action_freeze = {
        "schema": f"{P1R22_SCHEMA}-action-freeze/v1",
        "instruction_id": P1R22_INSTRUCTION_ID,
        "request_order_sha256": request_order,
        "rollout_sha256": {
            key: value["rollout_sha256"] for key, value in sorted(rollouts.items())
        },
        "actions_frozen_before_heldout": True,
        "inner_step_heldout_access_count": 0,
    }
    action_freeze_sha = write_once(
        raw_root / "atomic-runtime" / "action-freeze.json", action_freeze
    )
    cases, cases_sha = load_counterfact_cases_after_freeze(
        dataset_path,
        requests,
        request_order_sha256=request_order,
        action_freeze_sha256=action_freeze_sha,
    )
    freeze = StepwiseActionFreeze(
        P1R22_INSTRUCTION_ID,
        request_order,
        action_freeze_sha,
        True,
    )
    eval_ledger = ComputeLedger()
    w0_counter = ModelForwardCounter(model, eval_ledger)
    try:
        w0_receipt, w0_compute = _evaluate_stepwise_state(
            model,
            tokenizer,
            cases,
            alias=alias,
            freeze=freeze,
            ledger=eval_ledger,
            touched=touched,
        )
    finally:
        w0_counter.close()
    w0_sha = write_once(
        raw_root / "atomic-runtime" / "W0-terminal-reference.json",
        {
            "action_freeze_sha256": action_freeze_sha,
            "cases_sha256": cases_sha,
            "receipt": w0_receipt.raw_free_payload(),
            "compute": w0_compute,
            "shared_by_neutral_soft": True,
        },
    )
    endpoints: dict[str, dict[str, Any]] = {}
    for arm in (CommonColdArm.BG_NEUTRAL, CommonColdArm.BG_SOFT):
        factors = rollouts[arm.value]["terminal_factors"]
        endpoint_materializer = AcceptedPhysicalStateMaterializer(model, base_values)
        endpoint_materialization = endpoint_materializer.materialize(
            factors, transition_index=FIXED_E8_GRID_COUNT
        )
        endpoint_counter = ModelForwardCounter(model, eval_ledger)
        try:
            receipt, compute = _evaluate_stepwise_state(
                model,
                tokenizer,
                cases,
                alias=alias,
                freeze=freeze,
                ledger=eval_ledger,
                touched=touched,
            )
        finally:
            endpoint_counter.close()
        restore = endpoint_materializer.restore()
        endpoints[arm.value] = {
            "receipt": receipt.raw_free_payload(),
            "compute": compute,
            "endpoint_materialization": endpoint_materialization,
            "w0_restore": restore,
        }
        endpoints[arm.value]["identity_sha256"] = canonical_hash(
            endpoints[arm.value]
        )
        write_once(
            raw_root / "atomic-runtime" / f"endpoint-{arm.value}.json",
            endpoints[arm.value],
        )
    artifact_guard.assert_unchanged()
    if _physical_parameter_identity(touched) != base_contract:
        raise ODEBFStateError("P1R22 final W0 restore differs")
    public_rollouts = {
        key: _public_rollout(value) for key, value in sorted(rollouts.items())
    }
    terminal = {
        "schema": f"{P1R22_SCHEMA}-terminal/v1",
        "instruction_id": P1R22_INSTRUCTION_ID,
        "method_id": P1R22_METHOD_ID,
        "status": "ATOMIC_RUNTIME_OPTIMIZATION_BG_NEUTRAL_SOFT_COMPLETE",
        "alias": alias,
        "source_head": source_head,
        "request_order_sha256": request_order,
        "optimization_lock_sha256": optimization_lock_sha256,
        "numerical_lock_sha256": numerical_sha256,
        "context_sha256": context_sha256,
        "bootstrap_sha256": bootstrap_sha,
        "action_freeze_sha256": action_freeze_sha,
        "w0_reference_sha256": w0_sha,
        "rollouts": public_rollouts,
        "endpoints": endpoints,
        "request_microbatch_size": request_microbatch_size,
        "objective_batch_plan_sha256": objective_batch_plan_sha,
        "artifact_receipt": asdict(artifact_receipt),
        "cuda_preflight": dict(cuda_runtime_receipt),
        "production_compute": {
            key: arm_states[key].ledger.raw_free_payload()
            for key in sorted(arm_states)
        },
        "terminal_evaluation_compute": eval_ledger.raw_free_payload(),
        "job_compute": job_ledger.raw_free_payload(),
        "total_runtime_wall_seconds": time.perf_counter() - started,
        "hot_hook_dense_assembly_count": 0,
        "hot_hook_full_weight_hash_count": 0,
        "overlay_slope_model_forward_count": 0,
        "overlay_slope_backward_count": 0,
        "inner_field_empty_cache_count": 0,
        "candidate_objective_inner_k_count": 0,
        "candidate_progress_delayed_reuse_count": 14,
        "terminal_w_only_objective_count": 2,
        "inner_step_heldout_evaluation_count": 0,
        "scientific_promotion_authorized": False,
        "final_w0_restored": True,
        "persistent_commit_count": 0,
    }
    terminal_sha = write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": f"{P1R22_SCHEMA}-manifest/v1",
        "terminal_sha256": terminal_sha,
        "source_head": source_head,
        "request_order_sha256": request_order,
        "optimization_lock_sha256": optimization_lock_sha256,
        "action_freeze_sha256": action_freeze_sha,
        "w0_reference_sha256": w0_sha,
        "endpoint_sha256": {
            key: value["identity_sha256"] for key, value in sorted(endpoints.items())
        },
    }
    manifest_sha = write_once(destination / "manifest.json", manifest)
    return {
        "status": terminal["status"],
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "cells": list(sorted(endpoints)),
    }


__all__ = ["run_p1r22_atomic_optimization"]
