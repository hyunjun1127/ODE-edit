"""R10 common-terminal-coordinate fixed-E8 runtime.

This module reuses the verified R8 transaction, functional-observation and
post-freeze evaluator helpers while replacing only the cold coordinate,
bootstrap, target scale, and three-arm panel locked by R10.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .accounting import ComputeLedger
from .benchmark import (
    COUNTERFACT_EVALUATOR_RELATIVE,
    OFFICIAL_AGGREGATOR_RELATIVE,
    PINNED_SOURCE_SHA256,
)
from .common_cold_coordinate import (
    COMMON_COLD_H,
    COMMON_COLD_INSTRUCTION_ID,
    COMMON_COLD_LAYER_ORDER,
    CommonColdScale,
    CommonColdScaleMetric,
    RequestResidualActivationOverlay,
    common_cold_bootstrap,
    common_cold_source_contract,
    common_terminal_residual_input,
    write_aware_common_target_velocity,
)
from .contracts import BATCH_SIZE, ODEBFContractError, ODEBFStateError, canonical_hash
from .first_hit import FeasibilityVerdict
from .fixed_e8_soft_routing import (
    FIXED_E8_GRID_COUNT,
    FIXED_E8_H,
    FIXED_E8_KAPPA,
    FixedE8Arm,
    FixedE8Clock,
    FixedE8Stage2FailurePolicy,
    FixedE8StepMode,
    solve_fixed_e8_routing,
)
from .functional import WaypointFactor, tensor_sha256
from .p0_runtime import ModelForwardCounter
from .p1_adaptive import fraction_payload
from .p1_adaptive_runtime import (
    FunctionalPDecisionPolicy,
    _controller_replay_entry,
    _evaluate_rewrite,
    _factor_map,
    _factor_state,
    _functional_trial,
    _history_actions,
    _history_keys,
    _merge_factors,
    _omega_state,
    _parameter_contract_sha256,
    _postfreeze_stepwise_panel,
    _risk_payload,
    _terminal_confirm_snapshots,
)
from .p1_backend import (
    SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1,
    PinnedCovarianceRegistry,
    SignedProgressReceipt,
    _virtual_context,
    build_p1_dynamic_field,
    capture_p1_native_entry,
    evaluate_routing_progress,
)
from .p1_controller import AcceptedLayerContribution, P1ControllerLock
from .p1_replay import OuterEntryPretrainedCache, Theta0TeacherCache, build_outer_entry_pretrained_cache
from .p1_state import ArmWeightSnapshot, P1Arm, P1HistoryLedger
from .p1_common_coldcoord_fixed_e8_panel import (
    COMMON_COLD_SCHEMA_NAMESPACE,
    WARM_ALLOFF_ROOTS,
)
from .request_digest import ordered_request_digest_v1
from .routing import PreservationConstraintPolicy
from .sampling import StatelessReplaySchedule
from .target_new_nll import RoutingObjective, evaluate_routing_objective
from .cold_start_target import (
    _unwrap_layer_output,
    cold_lookup_positions,
    capture_cold_z_base,
)
from . import fixed_e8_runtime as legacy


class CommonColdArm(str, Enum):
    RS_NEUTRAL = "RS-NEUTRAL"
    RS_SOFT = "RS-SOFT"
    BG_NEUTRAL = "BG-NEUTRAL"

    @property
    def scale(self) -> CommonColdScale:
        return (
            CommonColdScale.BATCH_GLOBAL
            if self is CommonColdArm.BG_NEUTRAL
            else CommonColdScale.ROBUST_SHARED
        )

    @property
    def routing_arm(self) -> FixedE8Arm:
        return (
            FixedE8Arm.SOFT
            if self is CommonColdArm.RS_SOFT
            else FixedE8Arm.NEUTRAL
        )


def _common_initial_contract(
    *,
    bootstrap_target: torch.Tensor,
    field: Any,
    field_receipt: legacy.FixedE8FieldReceipt,
    routing: Any,
    metric: CommonColdScaleMetric,
) -> dict[str, Any]:
    payload = {
        "schema": "ode-edit-common-cold-initial-contract/v1",
        "bootstrap_target_sha256": tensor_sha256(bootstrap_target),
        "field_semantic_sha256": field_receipt.field_semantic_sha256,
        "shared_terminal_residual_sha256": tensor_sha256(
            field.layers[0].residual
        ),
        "signed_slopes": [float(item) for item in routing.signed_slopes],
        "p_max": float(routing.p_max),
        "requested_progress": float(routing.requested_progress),
        "pre_soft_velocity": [
            float(item) for item in routing.pre_soft_velocity
        ],
        "target_velocity_scale_sha256": metric.identity_sha256,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _common_field_semantic_receipt(
    field: Any,
    *,
    solve_history: Mapping[int, torch.Tensor],
    risk_history: Mapping[int, torch.Tensor],
) -> dict[str, Any]:
    layers = tuple(int(item.layer) for item in field.layers)
    residual_hashes = tuple(tensor_sha256(item.residual) for item in field.layers)
    key_hashes = tuple(tensor_sha256(item.key) for item in field.layers)
    q_hashes = tuple(tensor_sha256(item.q) for item in field.layers)
    weight_hashes = tuple(
        hashlib.sha256(item.weight_name.encode("utf-8")).hexdigest()
        for item in field.layers
    )
    if (
        layers != COMMON_COLD_LAYER_ORDER
        or len(set(residual_hashes)) != 1
        or len(set(key_hashes)) != len(layers)
        or len(set(q_hashes)) != len(layers)
        or len(set(weight_hashes)) != len(layers)
    ):
        raise ODEBFContractError("common cold field layer identities differ")
    semantic_layers: list[dict[str, Any]] = []
    cost: list[dict[str, Any]] = []
    for item in field.layers:
        covariance = asdict(item.covariance_receipt)
        wall = covariance.pop("wall_seconds")
        semantic_layers.append(
            {
                "layer": item.layer,
                "weight_name_sha256": hashlib.sha256(
                    item.weight_name.encode("utf-8")
                ).hexdigest(),
                "key_sha256": tensor_sha256(item.key),
                "projected_key_sha256": tensor_sha256(item.projected_key),
                "residual_sha256": tensor_sha256(item.residual),
                "residual_definition": item.residual_definition,
                "residual_divisor": item.residual_divisor,
                "q_sha256": tensor_sha256(item.q),
                "factor_sha256": item.factor_identity(),
                "covariance_receipt": covariance,
                "covariance_action_sha256": tensor_sha256(item.covariance_action),
                "covariance_gram_sha256": tensor_sha256(item.covariance_gram),
                "woodbury": asdict(item.woodbury_certificate),
                "history_solve_sha256": tensor_sha256(solve_history[item.layer]),
                "history_risk_sha256": tensor_sha256(risk_history[item.layer]),
                "history_action_sha256": tensor_sha256(item.history_action),
            }
        )
        cost.append({"layer": item.layer, "covariance_wall_seconds": wall})
    scientific = {
        "schema": "ode-edit-common-cold-field-semantic/v1",
        "accepted_waypoint": field.accepted_waypoint,
        "request_order_sha256": field.request_order_sha256,
        "target_state_sha256": tensor_sha256(field.target_state),
        "terminal_current_z_sha256": tensor_sha256(field.current_z),
        "residual_policy": SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1,
        "residual_hash_unique_count": len(set(residual_hashes)),
        "key_hash_unique_count": len(set(key_hashes)),
        "q_hash_unique_count": len(set(q_hashes)),
        "writer_identity_unique_count": len(set(weight_hashes)),
        "layers": semantic_layers,
    }
    return {
        "schema": "ode-edit-common-cold-field-semantic-receipt/v1",
        "semantic_identity_sha256": canonical_hash(scientific),
        "scientific_content": scientific,
        "cost_telemetry": cost,
        "full_field_receipt_identity_sha256": field.identity_sha256,
    }


def _objective_with_residual(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    factors: Mapping[str, Sequence[WaypointFactor]],
    contexts: Sequence[Sequence[str]],
    target_layer_name: str,
    lookup_positions: Sequence[int],
    residual: torch.Tensor,
) -> tuple[Any, dict[str, Any]]:
    overlay = RequestResidualActivationOverlay(
        model, target_layer_name, residual, lookup_positions
    )
    with overlay:
        receipt = evaluate_routing_progress(
            model,
            tokenizer,
            requests,
            cumulative_factors_by_weight=factors,
            objective=RoutingObjective.TARGET_NEW_NLL,
            contexts=contexts,
        )
    return receipt, overlay.raw_free_payload()


def _signed_progress_with_residual(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    field: Any,
    *,
    factors: Mapping[str, Sequence[WaypointFactor]],
    contexts: Sequence[Sequence[str]],
    target_layer_name: str,
    lookup_positions: Sequence[int],
    residual: torch.Tensor,
    ledger: ComputeLedger,
) -> tuple[SignedProgressReceipt, dict[str, Any]]:
    overlay = RequestResidualActivationOverlay(
        model, target_layer_name, residual, lookup_positions
    )
    with overlay:
        signed = legacy._fixed_e8_signed_progress_gradient(
            model,
            tokenizer,
            requests,
            field,
            cumulative_factors_by_weight=factors,
            contexts=contexts,
            ledger=ledger,
        )
    return signed, overlay.raw_free_payload()


def _build_common_field(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    arm: CommonColdArm,
    metric: CommonColdScaleMetric,
    step_index: int,
    current_factors: Mapping[str, Sequence[WaypointFactor]],
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    capture: legacy.FixedE8EntryCapture,
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    lock: P1ControllerLock,
    history: P1HistoryLedger,
    accepted_by_layer: Mapping[int, Sequence[AcceptedLayerContribution]],
    ledger: ComputeLedger,
    replay_entry: Any,
    theta0_cache: Theta0TeacherCache,
    touched: Mapping[str, torch.nn.Parameter],
    schedule: StatelessReplaySchedule,
    recorder: legacy.FixedE8ReceiptRecorder,
    lookup_positions: Sequence[int],
    target_layer_name: str,
) -> tuple[Any, Any, Any, Any, Any, torch.Tensor, Mapping[str, Any], Any, Mapping[str, Any]]:
    before = _parameter_contract_sha256(touched)
    before_history = history.snapshot().digest
    before_sampler = schedule.state_digest
    before_rng = legacy._rng_identity()
    solve_history = _history_keys(history, COMMON_COLD_LAYER_ORDER, risk=False)
    risk_history = _history_keys(history, COMMON_COLD_LAYER_ORDER, risk=True)
    residual_input = common_terminal_residual_input(
        current_target, current_terminal, ordered_request_digest_v1(
            [str(item["request_sha256"]) for item in requests]
        )
    )
    field = build_p1_dynamic_field(
        model,
        tokenizer,
        requests,
        hparams,
        projector,
        contexts,
        target_state=current_target,
        accepted_waypoint=step_index,
        cumulative_factors_by_weight=current_factors,
        history_solve_keys_by_layer=solve_history,
        history_risk_keys_by_layer=risk_history,
        covariance_registry=covariance_registry,
        projector_sha256=projector_sha256,
        residual_tolerance=lock.residual_tolerance,
        ledger=ledger,
        residual_policy=SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1,
        allow_zero_capacity=False,
        shared_terminal_residual=residual_input,
    )
    signed, signed_overlay = _signed_progress_with_residual(
        model,
        tokenizer,
        requests,
        field,
        factors=current_factors,
        contexts=contexts,
        target_layer_name=target_layer_name,
        lookup_positions=lookup_positions,
        residual=residual_input.residual,
        ledger=ledger,
    )
    semantic = _common_field_semantic_receipt(
        field, solve_history=solve_history, risk_history=risk_history
    )
    built = legacy._build_fixed_e8_problem(
        field,
        signed,
        accepted_by_layer=accepted_by_layer,
        committed_load_by_layer=history.cumulative_load(),
        lock=lock,
        current_history_action_by_layer=_history_actions(field, history),
    )
    factor_state = _factor_state(capture.entry_sha256, current_factors, current_target)
    inventory, probe = legacy._fixed_e8_functional_basis_probe(
        model,
        tokenizer,
        alias=alias,
        field=field,
        step_index=step_index,
        factors=current_factors,
        target_state=current_target,
        capture=capture,
        replay_entry=replay_entry,
        theta0_cache=theta0_cache,
        lock=lock,
        ledger=ledger,
        touched=touched,
        history=history,
        schedule=schedule,
        factor_state_sha256=factor_state,
        field_semantic_sha256=semantic["semantic_identity_sha256"],
    )
    observer, observed = legacy._fixed_e8_solver_receipt_observer(
        recorder=recorder,
        step_index=step_index,
        arm=arm,
        field_sha256=field.identity_sha256,
        field_semantic_sha256=semantic["semantic_identity_sha256"],
        problem_sha256=built.problem.identity(),
        functional_inventory_sha256=inventory.raw_free_payload()["identity_sha256"],
    )
    routing = solve_fixed_e8_routing(
        built.problem,
        inventory,
        arm=arm.routing_arm,
        certificate_observer=observer,
        stage2_failure_policy=(
            FixedE8Stage2FailurePolicy.CERTIFIED_STAGE1_FALLBACK
            if arm is CommonColdArm.RS_SOFT
            else FixedE8Stage2FailurePolicy.FAIL_CLOSED
        ),
    )
    if tuple(observed) != routing.certificates:
        raise ODEBFStateError("common cold solver receipt sequence differs")
    if routing.mode is not FixedE8StepMode.JOINT_WRITE or routing.p_max <= 0.0:
        raise ODEBFContractError("BOOTSTRAP_READINESS_FAILED: p_max is nonpositive")
    ledger.increment("qp_solve", len(routing.certificates))
    ledger.increment("qp_certificate", len(routing.certificates))
    target_velocity, target_receipt = write_aware_common_target_velocity(
        model,
        tokenizer,
        requests,
        contexts,
        target_layer_name=target_layer_name,
        lookup_positions=lookup_positions,
        field=field,
        velocity_coefficients=routing.velocity,
        cumulative_factors_by_weight=current_factors,
        metric=metric,
        ledger=ledger,
    )
    if (
        _parameter_contract_sha256(touched) != before
        or history.snapshot().digest != before_history
        or schedule.state_digest != before_sampler
        or legacy._rng_identity() != before_rng
    ):
        raise ODEBFStateError("common cold field build mutated state")
    field_payload = {
        "method": common_cold_source_contract(),
        "arm": arm.value,
        "step_index": step_index,
        "field": field.raw_free_payload(),
        "field_semantic": semantic,
        "shared_terminal_residual_input": residual_input.raw_free_payload(),
        "signed_progress": asdict(signed),
        "signed_progress_overlay": signed_overlay,
        "problem_receipt_sha256": built.identity_sha256,
        "functional_basis": inventory.raw_free_payload(),
        "probe_receipt": probe,
        "routing": routing.raw_free_payload(),
        "target_velocity": target_receipt,
        "target_probe_and_write_h_equal": True,
        "h": float(FIXED_E8_H),
        "scientific_retry_count": 0,
    }
    persisted = recorder.field(field_payload)
    receipt_payload = {
        "field_sha256": field.identity_sha256,
        "field_semantic_sha256": semantic["semantic_identity_sha256"],
        "signed_progress_sha256": canonical_hash(list(signed.signed_progress)),
        "problem_sha256": built.problem.identity(),
        "functional_inventory_sha256": inventory.raw_free_payload()["identity_sha256"],
        "routing_sha256": routing.identity_sha256,
        "target_velocity_sha256": target_receipt["velocity_sha256"],
        "persisted_field_receipt_sha256": persisted,
    }
    field_receipt = legacy.FixedE8FieldReceipt(
        receipt_payload["field_sha256"],
        receipt_payload["field_semantic_sha256"],
        receipt_payload["signed_progress_sha256"],
        receipt_payload["problem_sha256"],
        receipt_payload["functional_inventory_sha256"],
        receipt_payload["routing_sha256"],
        receipt_payload["target_velocity_sha256"],
        canonical_hash(receipt_payload),
    )
    return field, signed, built, inventory, routing, target_velocity, target_receipt, field_receipt, probe


def _run_common_arm(
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
    recorder: legacy.FixedE8ReceiptRecorder,
    stages: Any,
    expected_initial_contract: Mapping[str, Any] | None = None,
    initial_contract_sink: dict[str, Any] | None = None,
) -> tuple[legacy.FixedE8Rollout, dict[str, Any]]:
    clock = FixedE8Clock()
    history = arm_state.history
    ledger = arm_state.ledger
    before_model = _parameter_contract_sha256(touched)
    before_history = history.snapshot().digest
    before_sampler = schedule.state_digest
    before_rng = legacy._rng_identity()
    current_target = bootstrap_target.clone()
    current_terminal = z_base.clone()
    current_residual = common_terminal_residual_input(
        current_target,
        current_terminal,
        ordered_request_digest_v1([str(item["request_sha256"]) for item in requests]),
    ).residual
    current_factors: dict[str, tuple[WaypointFactor, ...]] = {}
    accepted_by_layer: dict[int, list[AcceptedLayerContribution]] = {
        layer: [] for layer in COMMON_COLD_LAYER_ORDER
    }
    entry_omega = _omega_state(accepted_by_layer)
    current_objective, _ = _objective_with_residual(
        model,
        tokenizer,
        requests,
        factors=current_factors,
        contexts=contexts,
        target_layer_name=target_layer_name,
        lookup_positions=lookup_positions,
        residual=current_residual,
    )
    entry_eval = _evaluate_rewrite(
        model, tokenizer, requests, alias=alias, factors=current_factors, ledger=ledger
    )
    snapshots: list[legacy.FixedE8Snapshot] = []
    first_hit = legacy.FixedE8HitTracker()
    total_qp = total_backend = total_solver_fallback = total_stage1_fallback = 0
    initial_contract: dict[str, Any] | None = None
    for step_index in range(FIXED_E8_GRID_COUNT):
        point = clock.begin_field()
        replay_entry = _controller_replay_entry(
            model,
            tokenizer,
            alias=alias,
            arm_state=arm_state,
            sample_waypoint=step_index + 1,
            factors=current_factors,
            request_by_sha256=request_by_sha256,
            population_by_sha256=population_by_sha256,
            schedule=schedule,
            outer_entry_p_cache=outer_entry_p_cache,
        )
        field, signed, built, inventory, routing, target_velocity, target_receipt, field_receipt, probe = _build_common_field(
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
        )
        solver_accounting = legacy._fixed_e8_solver_accounting(routing)
        total_qp += int(solver_accounting["actual_logical_qp_certificate_count"])
        total_backend += int(solver_accounting["actual_optimizer_backend_invocation_count"])
        total_solver_fallback += int(solver_accounting["actual_fallback_backend_invocation_count"])
        total_stage1_fallback += routing.stage1_selection_fallback_count
        if step_index == 0:
            initial_contract = _common_initial_contract(
                bootstrap_target=bootstrap_target,
                field=field,
                field_receipt=field_receipt,
                routing=routing,
                metric=metric,
            )
            if initial_contract_sink is not None:
                initial_contract_sink.clear()
                initial_contract_sink.update(initial_contract)
            if (
                expected_initial_contract is not None
                and initial_contract != dict(expected_initial_contract)
            ):
                raise ODEBFContractError(
                    "RS Neutral/Soft common initial contract differs"
                )
        increment = legacy.fixed_e8_waypoint_factors(
            field, routing.velocity, step_index=step_index
        )
        target_write_identity = legacy.fixed_e8_target_write_coefficient_identity(
            field,
            routing,
            increment,
            target_probe_coefficients=target_receipt[
                "target_probe_scientific_coefficient"
            ],
            target_probe_effective_coefficients=target_receipt[
                "target_probe_effective_coefficient"
            ],
            target_probe_effective_dtype=target_receipt[
                "target_probe_effective_dtype"
            ],
            target_probe_effective_device_type=target_receipt[
                "target_probe_effective_device_type"
            ],
            physical_write_effective_device_type=next(
                model.parameters()
            ).device.type,
        )
        candidate_factors = _merge_factors(current_factors, increment)
        target_trial = (
            current_target
            + float(FIXED_E8_H)
            * target_velocity.detach().to(device="cpu", dtype=torch.float32)
        ).contiguous()
        if not torch.isfinite(target_trial).all():
            raise ODEBFContractError("common cold target trial is non-finite")
        realization_before = (
            _parameter_contract_sha256(touched),
            history.snapshot().digest,
            schedule.state_digest,
            legacy._rng_identity(),
        )
        forward_before = ledger.counters["model_forward"]
        with _virtual_context(model, candidate_factors):
            candidate_terminal = capture_cold_z_base(model, tokenizer, requests, hparams)
        forward_delta = ledger.counters["model_forward"] - forward_before
        if (
            forward_delta != 1
            or realization_before
            != (
                _parameter_contract_sha256(touched),
                history.snapshot().digest,
                schedule.state_digest,
                legacy._rng_identity(),
            )
        ):
            raise ODEBFStateError("common cold realization capture mutated state")
        candidate_residual = common_terminal_residual_input(
            target_trial,
            candidate_terminal,
            field.request_order_sha256,
        ).residual
        candidate_objective, candidate_overlay = _objective_with_residual(
            model,
            tokenizer,
            requests,
            factors=candidate_factors,
            contexts=contexts,
            target_layer_name=target_layer_name,
            lookup_positions=lookup_positions,
            residual=candidate_residual,
        )
        actual_progress = float(current_objective.value - candidate_objective.value)
        predicted_progress = float(
            built.problem.signed_progress @ np.asarray(routing.velocity)
        )
        rho = actual_progress / max(predicted_progress, lock.minimum_progress)
        functional = _functional_trial(
            model,
            tokenizer,
            alias=alias,
            entry=replay_entry,
            theta0_cache=theta0_cache,
            factors=candidate_factors,
            lock=lock,
            ledger=ledger,
        )
        evaluation = _evaluate_rewrite(
            model,
            tokenizer,
            requests,
            alias=alias,
            factors=candidate_factors,
            ledger=ledger,
        )
        snapshot_sha = _factor_state(capture.entry_sha256, candidate_factors, target_trial)
        factor_state = _factor_state(capture.entry_sha256, current_factors, current_target)
        capacity = legacy._fixed_capacity_payload(
            capture,
            field,
            current_factors,
            candidate_factors,
            increment,
            problem=built.problem,
            routing=routing,
            signed=signed,
            step_index=step_index,
            state_sha256=_parameter_contract_sha256(touched),
            target_sha256=tensor_sha256(current_target),
            factor_state_sha256=factor_state,
            candidate_sha256=snapshot_sha,
        )
        structural_h_value = built.problem.historical.value(np.asarray(routing.velocity))
        structural_p_value = built.problem.pretrained.value(np.asarray(routing.velocity))
        trust_value = float(
            np.asarray(routing.velocity)
            @ built.problem.trust_metric
            @ np.asarray(routing.velocity)
        )
        structural_payload = {
            "historical": {
                "value": structural_h_value,
                "passed": structural_h_value <= built.problem.historical.budget + 1.0e-8,
                "status": "INACTIVE_EMPTY_HISTORY",
                "role": "UNCALIBRATED_OBSERVATION_ONLY",
                "decision_influence_count": 0,
            },
            "pretrained": {
                "value": structural_p_value,
                "passed": structural_p_value <= built.problem.pretrained.budget + 1.0e-8,
                "status": "ACTIVE_OBSERVATION_ONLY",
                "role": "UNCALIBRATED_OBSERVATION_ONLY",
                "decision_influence_count": 0,
            },
            "trust": {
                "value": trust_value,
                "passed": trust_value <= built.problem.trust_radius**2 + 1.0e-8,
                "radius_squared": built.problem.trust_radius**2,
                "role": "TECHNICAL_INTEGRATION_BOUND",
                "decision_influence_count": 1,
            },
            "functional_h": {
                **_risk_payload(functional.historical),
                "status": "INACTIVE_EMPTY_HISTORY",
                "role": "UNCALIBRATED_OBSERVATION_ONLY",
                "decision_influence_count": 0,
            },
            "functional_p": {
                **_risk_payload(functional.pretrained),
                "status": "ACTIVE_OBSERVATION_ONLY",
                "role": "UNCALIBRATED_OBSERVATION_ONLY",
                "decision_influence_count": 0,
            },
        }
        feasibility, feasibility_payload = (
            legacy._fixed_e8_factual_online_feasibility(
                structural_payload,
                history_item_count=inventory.history_item_count,
            )
        )
        structural_payload["online_feasibility_observation"] = feasibility_payload
        target_realization = legacy.fixed_e8_target_write_realization(
            current_target,
            target_trial,
            current_terminal,
            candidate_terminal,
        )
        progress_payload = {
            "predicted": predicted_progress,
            "actual": actual_progress,
            "rho": rho,
            "p_max": routing.p_max,
            "requested_progress": routing.requested_progress,
            "negative_actual_progress_observation_only": actual_progress <= 0.0,
            "rho_below_point_one_observation_only": rho < 0.1,
            "scientific_rejection_count": 0,
        }
        routing_payload = {
            "method": common_cold_source_contract(),
            "scale": metric.raw_free_payload(),
            "field_receipt_sha256": field_receipt.identity_sha256,
            "field_sha256": field.identity_sha256,
            "field_semantic_sha256": field_receipt.field_semantic_sha256,
            "routing": routing.raw_free_payload(),
            "functional_basis": inventory.raw_free_payload(),
            "probe_receipt_sha256": probe["identity_sha256"],
            "target_velocity": target_receipt,
            "target_write_coefficient_identity": target_write_identity,
            "target_write_realization": target_realization,
            "candidate_residual_sha256": tensor_sha256(candidate_residual),
            "candidate_objective_overlay": candidate_overlay,
            "h": float(FIXED_E8_H),
            "theta_equals_h_times_v": True,
        }
        trial_sha = recorder.trial(
            {
                "step_index": step_index,
                "tau_before": fraction_payload(point.tau_before),
                "tau_after": fraction_payload(point.tau_after),
                "routing": routing_payload,
                "progress": progress_payload,
                "structural_functional": structural_payload,
                "official_success": evaluation.batch_success.raw_free_payload(),
                "snapshot_sha256": snapshot_sha,
                "capacity_sha256": capacity["identity_sha256"],
                "accepted": True,
                "scientific_candidate_veto_count": 0,
            }
        )
        transition = legacy._fixed_e8_advance_grid_transition(
            ledger, clock, point, scientific_observation={}
        )
        transition_sha = recorder.transition(
            {
                **transition,
                "trial_receipt_sha256": trial_sha,
                "snapshot_sha256": snapshot_sha,
                "scientific_observations_do_not_gate": True,
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
        ledger.record_accepted_step(accepted_dt=float(FIXED_E8_H), completed_k_total=step_index + 1)
        hit = legacy.FixedE8HitRecord(
            step_index + 1,
            point.tau_after,
            snapshot_sha,
            evaluation.batch_success.numerator,
            feasibility.all_pass,
        )
        first_hit.append(hit)
        first_observed = first_hit.first_online is hit
        accepted_sha = recorder.accepted(
            {
                "accepted_index": step_index + 1,
                "tau_after": fraction_payload(point.tau_after),
                "transition_sha256": transition_sha,
                "snapshot_sha256": snapshot_sha,
                "official_success": evaluation.batch_success.raw_free_payload(),
                "structural_functional": structural_payload,
                "progress": progress_payload,
                "routing": routing_payload,
                "capacity": capacity,
                "first_hit_observed": first_observed,
                "first_hit_decision_influence_count": 0,
            }
        )
        if first_observed:
            recorder.first_hit(
                {
                    "accepted_index": step_index + 1,
                    "tau": fraction_payload(point.tau_after),
                    "snapshot_sha256": snapshot_sha,
                    "accepted_receipt_sha256": accepted_sha,
                    "observation_only": True,
                }
            )
        snapshots.append(
            legacy.FixedE8Snapshot(
                step_index + 1,
                point.tau_after,
                FIXED_E8_H,
                _factor_map(candidate_factors),
                target_trial.clone(),
                snapshot_sha,
                evaluation,
                feasibility,
                structural_payload,
                routing_payload,
                progress_payload,
                capacity,
                field.identity_sha256,
                canonical_hash(list(routing.velocity)),
                routing.identity_sha256,
                accepted_sha,
                {
                    "policy": "OBSERVATION_ONLY",
                    "observation": _risk_payload(functional.pretrained),
                    "decision_influence_count": 0,
                },
            )
        )
        current_factors = _factor_map(candidate_factors)
        current_target = target_trial.clone()
        current_terminal = candidate_terminal.clone()
        current_residual = candidate_residual.clone()
        current_objective = candidate_objective
        stages.record(
            f"post_{arm.value.lower().replace('-', '_')}_step_{step_index + 1}",
            {
                "arm": arm.value,
                "step": step_index + 1,
                "tau": float(point.tau_after),
                "success_count": evaluation.batch_success.numerator,
                "p_max": routing.p_max,
                "selected_solution_source": routing.selected_solution_source,
                "stage1_fallback_count": routing.stage1_selection_fallback_count,
                "snapshot_sha256": snapshot_sha,
            },
        )
    if len(snapshots) != 8 or clock.tau != Fraction(1, 1) or clock.field_count != 8:
        raise ODEBFStateError("common cold terminal grid invariant differs")
    if ledger.completed_correction_cycles == 0:
        ledger.finish_cycle(0)
    terminal_confirmations = _terminal_confirm_snapshots(
        model,
        tokenizer,
        alias=alias,
        arm_state=arm_state,
        snapshots=snapshots,
        request_by_sha256=request_by_sha256,
        population_by_sha256=population_by_sha256,
        schedule=schedule,
        outer_entry_p_cache=outer_entry_p_cache,
        theta0_cache=theta0_cache,
        lock=lock,
        ledger=ledger,
        recorder=recorder,
        trajectory_complete=True,
        functional_p_policy=FunctionalPDecisionPolicy.OBSERVATION_ONLY,
        preservation_policy=PreservationConstraintPolicy.OBSERVATION_ONLY,
    )
    if (
        _parameter_contract_sha256(touched) != before_model
        or history.snapshot().digest != before_history
        or schedule.state_digest != before_sampler
        or legacy._rng_identity() != before_rng
        or history.version != 0
        or _omega_state(accepted_by_layer) == entry_omega
    ):
        raise ODEBFStateError("common cold rollout state/purity differs")
    rollout_payload = {
        "method": common_cold_source_contract(),
        "variant": arm.value,
        "scale": metric.raw_free_payload(),
        "status": "COMMON_COLD_FIXED_E8_TAU_COMPLETE",
        "accepted_t": fraction_payload(clock.tau),
        "k_acc": 8,
        "n_trial": 8,
        "n_reject": 0,
        "field_build_count": 8,
        "operation_accounting": {
            "actual_logical_qp_certificate_count": total_qp,
            "actual_optimizer_backend_invocation_count": total_backend,
            "actual_solver_backend_fallback_count": total_solver_fallback,
            "actual_certified_stage1_selection_fallback_count": total_stage1_fallback,
            "functional_basis_endpoint_count": 48,
            "candidate_count": 8,
            "field_count": 8,
        },
        "clock": clock.terminal_receipt(),
        "entry_success": entry_eval.batch_success.raw_free_payload(),
        "online_first_hit": None if first_hit.first_online is None else {
            "accepted_index": first_hit.first_online.accepted_index,
            "tau": fraction_payload(first_hit.first_online.tau),
            "snapshot_sha256": first_hit.first_online.snapshot_sha256,
        },
        "first_hit_observation_only": True,
        "joint_zero_write_count": 0,
        "scientific_retry_count": 0,
        "scientific_rejection_count": 0,
        "native_or_direct_z_cold_access_count": 0,
        "receipt_links": recorder.links(),
        "compute": ledger.raw_free_payload(),
    }
    rollout_sha = canonical_hash(rollout_payload)
    recorder.terminal({"rollout_summary": rollout_payload, "rollout_sha256": rollout_sha})
    if initial_contract is None:
        raise ODEBFStateError("common cold initial contract is absent")
    return legacy.FixedE8Rollout(
        arm,
        "COMMON_COLD_FIXED_E8_TAU_COMPLETE",
        "COMMON_COLD_FIXED_E8_TAU_COMPLETE",
        SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1,
        Fraction(1, 1),
        8,
        8,
        0,
        8,
        snapshots,
        first_hit,
        recorder,
        ledger,
        entry_eval.batch_success.raw_free_payload(),
        terminal_confirmations,
        rollout_sha,
        arm.value,
        RoutingObjective.TARGET_NEW_NLL.value,
        "OBSERVATION_ONLY",
        "STRUCTFUNC_SOFT_OR_NEUTRAL",
        "BOUNDED_BASIS_SOFT",
    ), initial_contract


def _warm_context_identity_receipt(
    *,
    alias: str,
    requests: Sequence[Mapping[str, Any]],
    stream: Mapping[str, Any],
    contexts: Sequence[Sequence[str]],
    context_sha256: str,
) -> dict[str, Any]:
    """Prove current context surfaces against the immutable Warm receipts."""

    if alias not in WARM_ALLOFF_ROOTS:
        raise ODEBFContractError("common cold Warm context alias differs")
    if stream.get("panel_kind") != "REUSED_WARMUP_SEAL_CAUSAL_REGRESSION":
        raise ODEBFContractError("common cold reused Warm panel identity differs")
    flattened = tuple(value for group in contexts for value in group)
    if tuple(len(group) for group in contexts) != (1, 5) or len(flattened) != 6:
        raise ODEBFContractError("common cold context inventory differs")
    current_template_hashes = tuple(
        canonical_hash({"template": value}) for value in flattened
    )
    warm = WARM_ALLOFF_ROOTS[alias]
    expected_template_hashes = tuple(warm["context_template_sha256"])
    template_equal = tuple(
        observed == expected
        for observed, expected in zip(
            current_template_hashes, expected_template_hashes, strict=True
        )
    )
    request_order = ordered_request_digest_v1(
        [str(item["request_sha256"]) for item in requests]
    )
    request_equal = request_order == stream["batch_ordered_request_digest_v1"][0]
    rendered_hashes: list[list[str]] = []
    rendered_equal: list[list[bool]] = []
    for request in requests:
        prompt = str(request["prompt"])
        subject = str(request["subject"])
        request_hash = str(request["request_sha256"])
        row_hashes: list[str] = []
        row_equal: list[bool] = []
        for ordinal, template in enumerate(flattened):
            try:
                rendered = template.format(prompt).format(subject)
            except (IndexError, KeyError, ValueError) as exc:
                raise ODEBFContractError(
                    "common cold context rendering differs"
                ) from exc
            if not rendered:
                raise ODEBFContractError("common cold rendered context is empty")
            row_hashes.append(
                canonical_hash(
                    {
                        "request_sha256": request_hash,
                        "context_ordinal": ordinal,
                        "rendered": rendered,
                    }
                )
            )
            # Warm and current raw requests are byte-identical under the seal;
            # deterministic formatting therefore reduces exact rendered-byte
            # equality to exact source-template equality.
            row_equal.append(bool(request_equal and template_equal[ordinal]))
        rendered_hashes.append(row_hashes)
        rendered_equal.append(row_equal)
    payload = {
        "schema": "ode-edit-r10-reused-warm-context-identity/v1",
        "alias": alias,
        "panel_kind": stream["panel_kind"],
        "warm_context_file_sha256": warm["context_file_sha256"],
        "warm_context_sha256": warm["context_sha256"],
        "current_context_sha256": context_sha256,
        "context_inventory_exact_equal": context_sha256 == warm["context_sha256"],
        "warm_template_sha256": list(expected_template_hashes),
        "current_template_sha256": list(current_template_hashes),
        "template_byte_exact_equal": list(template_equal),
        "request_order_sha256": request_order,
        "warm_request_order_sha256": stream["batch_ordered_request_digest_v1"][0],
        "request_bytes_and_order_exact_equal": request_equal,
        "rendered_context_sha256": rendered_hashes,
        "rendered_context_byte_exact_equal": rendered_equal,
        "rendered_context_all_exact_equal": all(
            all(row) for row in rendered_equal
        ),
        "rendered_identity_proof": (
            "EXACT_RAW_REQUEST_HASH_AND_ORDER_PLUS_EXACT_TEMPLATE_BYTES_"
            "UNDER_DETERMINISTIC_TWO_STAGE_FORMAT"
        ),
        "intentional_context_render_change": not all(template_equal),
        "controller_policy": "COMMON_VALIDATED_CANONICAL_PLUS_5",
        "selection_regeneration_clip_rescue_count": 0,
        "decision_influence_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _warm_evaluator_parity_receipt(
    receipts: Mapping[tuple[str, int], Any],
    *,
    alias: str,
    request_order_sha256: str,
) -> dict[str, Any]:
    if alias not in WARM_ALLOFF_ROOTS:
        raise ODEBFContractError("common cold Warm evaluator alias differs")
    warm = WARM_ALLOFF_ROOTS[alias]
    expected_evaluator = PINNED_SOURCE_SHA256[COUNTERFACT_EVALUATOR_RELATIVE]
    expected_aggregator = PINNED_SOURCE_SHA256[OFFICIAL_AGGREGATOR_RELATIVE]
    if not receipts:
        raise ODEBFContractError("common cold post-freeze evaluator receipts are absent")
    rows: list[dict[str, Any]] = []
    for (variant, index), receipt in sorted(receipts.items()):
        primary = receipt.primary
        row = {
            "variant": variant,
            "snapshot_index": int(index),
            "request_order_sha256": primary.request_order_sha256,
            "evaluator_source_sha256": primary.evaluator_source_sha256,
            "aggregator_source_sha256": primary.aggregator_source_sha256,
            "generation_call_count": primary.generation_call_count,
            "boundary_touched": primary.boundary_touched,
        }
        if (
            primary.request_order_sha256 != request_order_sha256
            or primary.evaluator_source_sha256 != expected_evaluator
            or primary.aggregator_source_sha256 != expected_aggregator
            or primary.evaluation_case_identity_sha256
            != warm["evaluation_case_identity_sha256"]
            or primary.target_span_sha256 != warm["target_span_sha256"]
            or primary.generation_call_count != 0
        ):
            raise ODEBFContractError("common cold Warm evaluator parity differs")
        rows.append(row)
    payload = {
        "schema": "ode-edit-r10-reused-warm-evaluator-parity/v1",
        "request_order_sha256": request_order_sha256,
        "alias": alias,
        "evaluator_source_sha256": expected_evaluator,
        "aggregator_source_sha256": expected_aggregator,
        "warm_evaluation_case_identity_sha256": warm[
            "evaluation_case_identity_sha256"
        ],
        "warm_target_span_sha256": warm["target_span_sha256"],
        "receipt_count": len(rows),
        "receipts": rows,
        "outcome_tie_count": sum(bool(row["boundary_touched"]) for row in rows),
        "outcome_tie_role": "OBSERVATION_ONLY_EXACT_NLL_TIE",
        "outcome_tie_parity_influence_count": 0,
        "pinned_success_inequality_unchanged": True,
        "generation_call_count": 0,
        "heldout_open_after_action_freeze": True,
        "warm_metric_definition_exact": True,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _singleton_joint_capture_comparison(
    singleton: torch.Tensor,
    joint_column: torch.Tensor,
) -> dict[str, Any]:
    if (
        not isinstance(singleton, torch.Tensor)
        or not isinstance(joint_column, torch.Tensor)
        or singleton.shape != joint_column.shape
        or singleton.dtype != joint_column.dtype
        or singleton.device != joint_column.device
        or singleton.numel() == 0
        or not torch.isfinite(singleton).all()
        or not torch.isfinite(joint_column).all()
    ):
        raise ODEBFContractError(
            "prior ordinal6 singleton/joint numeric comparison differs"
        )
    left = singleton.detach().to(device="cpu", dtype=torch.float64).contiguous()
    right = joint_column.detach().to(
        device="cpu", dtype=torch.float64
    ).contiguous()
    difference = left - right
    mismatch = torch.ne(singleton, joint_column)
    mismatch_count = int(mismatch.sum().detach().cpu())
    count = int(singleton.numel())
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    denominator = left_norm * right_norm
    cosine = (
        1.0
        if mismatch_count == 0
        else (
            float(torch.sum(left * right)) / denominator
            if denominator > 0.0
            else 0.0
        )
    )
    payload = {
        "schema": "ode-edit-r10-singleton-joint-capture-comparison/v1",
        "canonical_runtime_source": "JOINT_B10_CAPTURE",
        "singleton_role": "OBSERVATION_ONLY_DIAGNOSTIC",
        "shape": list(singleton.shape),
        "dtype": str(singleton.dtype),
        "device": singleton.device.type,
        "element_count": count,
        "exact_equal": mismatch_count == 0,
        "exact_mismatch_count": mismatch_count,
        "exact_mismatch_fraction": mismatch_count / count,
        "maximum_absolute_difference": float(torch.max(torch.abs(difference))),
        "l2_difference": float(torch.linalg.vector_norm(difference)),
        "cosine": cosine,
        "singleton_sha256": tensor_sha256(singleton),
        "joint_column_sha256": tensor_sha256(joint_column),
        "status": (
            "EXACT_SINGLETON_JOINT_IDENTITY"
            if mismatch_count == 0
            else "FINITE_BATCH_KERNEL_NUMERIC_DIFFERENCE_OBSERVED"
        ),
        "controller_selection_endpoint_influence_count": 0,
        "tolerance_or_rescue_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _prior_qwen_ordinal6_audit(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    requests: Sequence[Mapping[str, Any]],
    stream: Mapping[str, Any],
    contexts: Sequence[Sequence[str]],
    hparams: Any,
) -> dict[str, Any]:
    if alias != "qwen2.5-7b-inst":
        payload = {
            "schema": "ode-edit-r10-prior-qwen-ordinal6-rca/v1",
            "alias": alias,
            "status": "NOT_APPLICABLE_NON_QWEN_SIBLING",
            "common_path_policy": True,
            "scientific_action_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload
    if (
        stream.get("panel_kind") != "REUSED_WARMUP_SEAL_CAUSAL_REGRESSION"
        or len(requests) != BATCH_SIZE
    ):
        raise ODEBFContractError("prior ordinal6 reused Warm seal differs")
    prior_requests = tuple(requests)
    ordinal = 6
    request = prior_requests[ordinal]
    from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main

    capture_module_name = hparams.layer_module_tmp.format(int(hparams.layers[-1]))
    capture_module = model.get_submodule(capture_module_name)

    def capture_layout(call: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        shapes: list[list[int]] = []
        dtypes: list[str] = []

        def observe(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            activation = _unwrap_layer_output(output)
            shapes.append([int(item) for item in activation.shape])
            dtypes.append(str(activation.dtype))

        handle = capture_module.register_forward_hook(observe)
        try:
            value = call()
        finally:
            handle.remove()
        if not shapes or len(shapes) != len(dtypes):
            raise ODEBFContractError("prior ordinal6 capture hook accounting differs")
        return value, {
            "hook_call_count": len(shapes),
            "activation_layout": shapes,
            "activation_dtype": dtypes,
            "capture_module_sha256": hashlib.sha256(
                capture_module_name.encode("utf-8")
            ).hexdigest(),
            "capture_side": "MODULE_OUTPUT_POST_HOOK",
        }

    joint, joint_layout = capture_layout(
        lambda: capture_cold_z_base(model, tokenizer, prior_requests, hparams)
    )

    def singleton_capture() -> torch.Tensor:
        return alpha_main.get_module_input_output_at_words(
            model,
            tokenizer,
            int(hparams.layers[-1]),
            context_templates=[str(request["prompt"])],
            words=[str(request["subject"])],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T.detach().to(device="cpu", dtype=torch.float32).contiguous()

    singleton, singleton_layout = capture_layout(singleton_capture)
    if singleton.shape != joint[:, ordinal : ordinal + 1].shape:
        raise ODEBFContractError("prior ordinal6 singleton/joint layout differs")
    capture_comparison = _singleton_joint_capture_comparison(
        singleton, joint[:, ordinal : ordinal + 1]
    )
    capture_equal = bool(capture_comparison["exact_equal"])
    prompt = str(request["prompt"])
    subject = str(request["subject"])
    target = str(request["target_new"])
    canonical = prompt.format(subject)
    original_padding = tokenizer.padding_side
    tokenization_by_padding: dict[str, dict[str, Any]] = {}
    try:
        for padding_side in ("left", "right"):
            tokenizer.padding_side = padding_side
            target_tokens = tuple(
                int(item) for item in tokenizer(f" {target}")["input_ids"]
            )
            encoded = tokenizer(
                [f"{canonical} {target}"], padding=True, return_tensors="pt"
            )
            input_ids = tuple(int(item) for item in encoded["input_ids"][0].tolist())
            attention = tuple(
                int(item) for item in encoded["attention_mask"][0].tolist()
            )
            suffix_start = len(input_ids) - len(target_tokens)
            suffix_exact = tuple(input_ids[suffix_start:]) == target_tokens
            if not target_tokens or suffix_start <= 0 or not suffix_exact:
                raise ODEBFContractError(
                    "prior ordinal6 target suffix/token boundary differs"
                )
            tokenization_by_padding[padding_side] = {
                "input_ids_sha256": canonical_hash(list(input_ids)),
                "attention_mask_sha256": canonical_hash(list(attention)),
                "target_suffix_ids_sha256": canonical_hash(list(target_tokens)),
                "target_suffix_token_count": len(target_tokens),
                "input_token_count": len(input_ids),
                "attention_nonpad_count": sum(attention),
                "suffix_start_index": suffix_start,
                "combined_suffix_exact": True,
                "next_token_logit_start_index": suffix_start - 1,
            }
    finally:
        tokenizer.padding_side = original_padding
    canonical_lookup = int(
        alpha_main.find_fact_lookup_idx(
            prompt,
            subject,
            tokenizer,
            hparams.fact_token,
            verbose=False,
        )
    )
    subject_token_ids = tuple(
        int(item)
        for item in tokenizer(subject, add_special_tokens=False)["input_ids"]
    )
    if not subject_token_ids:
        raise ODEBFContractError("prior ordinal6 subject tokenization is empty")
    lookup_positions = cold_lookup_positions(
        tokenizer, prior_requests, contexts, fact_token_strategy=hparams.fact_token
    )
    request_lookup = lookup_positions[
        ordinal * 6 : (ordinal + 1) * 6
    ]
    flattened_contexts = tuple(item for group in contexts for item in group)
    render_hashes = [
        canonical_hash({"surface": template.format(prompt).format(subject)})
        for template in flattened_contexts
    ]
    norms = torch.linalg.vector_norm(joint.double(), dim=0).square()
    payload = {
        "schema": "ode-edit-r10-prior-qwen-ordinal6-rca/v1",
        "alias": alias,
        "status": "ORDINAL6_CANONICAL_JOINT_CAPTURE_RETAINED",
        "request_sha256": str(request["request_sha256"]),
        "ordinal": ordinal,
        "request_order_sha256": stream["batch_ordered_request_digest_v1"][0],
        "reused_warm_source_stream_root_digest": stream["warm_source"][
            "source_stream_root_digest"
        ],
        "reused_warm_ordinal_case_id": int(request["case_id"]),
        "singleton_capture_sha256": tensor_sha256(singleton),
        "joint_column_capture_sha256": tensor_sha256(joint[:, ordinal : ordinal + 1]),
        "singleton_joint_exact_equal": capture_equal,
        "singleton_joint_numeric_comparison": capture_comparison,
        "exact_mismatch_is_observation_only": True,
        "capture_bug_inferred_from_bitwise_mismatch": False,
        "joint_shape": list(joint.shape),
        "singleton_shape": list(singleton.shape),
        "capture_module_sha256": hashlib.sha256(
            capture_module_name.encode("utf-8")
        ).hexdigest(),
        "capture_side": "MODULE_OUTPUT_POST_HOOK",
        "joint_capture_layout": joint_layout,
        "singleton_capture_layout": singleton_layout,
        "joint_to_singleton_transpose_contract": "[batch,hidden].T->[hidden,batch]",
        "lookup_positions": list(request_lookup),
        "lookup_position_count": len(request_lookup),
        "canonical_subject_lookup_index": canonical_lookup,
        "subject_token_ids_sha256": canonical_hash(list(subject_token_ids)),
        "subject_token_count": len(subject_token_ids),
        "fact_token_strategy": str(hparams.fact_token),
        "subject_sha256": canonical_hash({"subject": subject}),
        "canonical_prompt_sha256": canonical_hash({"prompt": canonical}),
        "controller_render_sha256": render_hashes,
        "tokenization_by_padding": tokenization_by_padding,
        "runtime_controller_padding_side": "left",
        "padding_restored_exact": tokenizer.padding_side == original_padding,
        "z_base_norm_squared": float(norms[ordinal]),
        "z_base_norm_squared_quantiles": [
            float(item)
            for item in torch.quantile(
                norms, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
            )
        ],
        "z_base_norm_squared_max_to_median_ratio": float(
            norms.max() / torch.median(norms)
        ),
        "request_column_mapping_policy": "ORDINAL_PRESERVING_REQUEST_MAJOR_6_CONTEXTS",
        "clip_delete_regenerate_rescue_count": 0,
        "native_or_direct_z_access_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _context_overlay_gate(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    contexts: Sequence[Sequence[str]],
    *,
    target_layer_name: str,
    lookup_positions: Sequence[int],
    z_base: torch.Tensor,
) -> dict[str, Any]:
    plain = evaluate_routing_objective(
        model,
        tokenizer,
        requests,
        objective=RoutingObjective.TARGET_NEW_NLL,
        contexts=contexts,
    )
    zero = torch.zeros_like(z_base, dtype=torch.float32, device="cpu").contiguous()
    zero_overlay = RequestResidualActivationOverlay(
        model, target_layer_name, zero, lookup_positions
    )
    with zero_overlay:
        hooked = evaluate_routing_objective(
            model,
            tokenizer,
            requests,
            objective=RoutingObjective.TARGET_NEW_NLL,
            contexts=contexts,
        )
    zero_receipt = zero_overlay.raw_free_payload()
    if (
        not torch.equal(plain.per_request_values, hooked.per_request_values)
        or plain.target_span_sha256 != hooked.target_span_sha256
        or plain.context_sha256 != hooked.context_sha256
        or zero_receipt["maximum_exact_delta_error"] != 0.0
        or zero_receipt["maximum_authoritative_assignment_error"] != 0.0
    ):
        raise ODEBFContractError("common cold zero-residual context parity failed")
    synthetic = torch.ones_like(z_base, dtype=torch.float32, device="cpu").contiguous()
    synthetic_overlay = RequestResidualActivationOverlay(
        model, target_layer_name, synthetic, lookup_positions
    )
    with synthetic_overlay:
        evaluate_routing_objective(
            model,
            tokenizer,
            requests,
            objective=RoutingObjective.TARGET_NEW_NLL,
            contexts=contexts,
        )
    synthetic_receipt = synthetic_overlay.raw_free_payload()
    if synthetic_receipt["maximum_authoritative_assignment_error"] != 0.0:
        raise ODEBFContractError("common cold synthetic additive overlay failed")
    payload = {
        "schema": "ode-edit-common-cold-context-overlay-gate/v1",
        "request_context_count": 60,
        "zero_residual": zero_receipt,
        "zero_residual_logit_nll_parity": True,
        "nonzero_synthetic_residual": synthetic_receipt,
        "same_request_residual_column_exact": True,
        "authoritative_assignment_exact": True,
        "realized_delta_error_observation_only": True,
        "realized_delta_error_decision_influence_count": 0,
        "absolute_replacement_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def run_common_coldcoord_fixed_e8_diagnostic(
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
    mutation_lock: Any,
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
) -> dict[str, Any]:
    from .p1_runtime import ArmRuntimeState, _entry_parameter_snapshot_sha256, _evaluate_native_rewrite, _observed_memory

    del mutation_lock
    request_order = ordered_request_digest_v1([str(item["request_sha256"]) for item in requests])
    if len(requests) != BATCH_SIZE or request_order != stream["batch_ordered_request_digest_v1"][0]:
        raise ODEBFContractError("common cold request/seal order differs")
    base_bytes = {name: tensor_sha256(value) for name, value in sorted(touched.items())}
    base_contract = _parameter_contract_sha256(touched)
    # The reused-Warm ordinal-6 RCA is completed before any target capture,
    # bootstrap, or controller action in this run.
    pre_action_counter = ModelForwardCounter(model, job_ledger)
    try:
        ordinal_audit = _prior_qwen_ordinal6_audit(
            model,
            tokenizer,
            alias=alias,
            requests=requests,
            stream=stream,
            contexts=contexts,
            hparams=hparams,
        )
    finally:
        pre_action_counter.close()
    ordinal_audit_sha = write_once(raw_root / "common-cold" / "prior-qwen-ordinal6-rca.json", ordinal_audit)
    stages.record(
        "post_prior_qwen_ordinal6_rca",
        {
            "receipt_sha256": ordinal_audit_sha,
            "status": ordinal_audit["status"],
            "completed_before_reused_seal_scientific_action": True,
        },
    )
    context_identity = _warm_context_identity_receipt(
        alias=alias,
        requests=requests,
        stream=stream,
        contexts=contexts,
        context_sha256=context_sha256,
    )
    context_identity_sha = write_once(
        raw_root / "common-cold" / "reused-warm-context-identity.json",
        context_identity,
    )
    stages.record(
        "post_reused_warm_context_identity",
        {
            "receipt_sha256": context_identity_sha,
            "context_inventory_exact_equal": context_identity[
                "context_inventory_exact_equal"
            ],
            "rendered_context_all_exact_equal": context_identity[
                "rendered_context_all_exact_equal"
            ],
            "intentional_context_render_change": context_identity[
                "intentional_context_render_change"
            ],
        },
    )
    pre_action_counter = ModelForwardCounter(model, job_ledger)
    try:
        z_base = capture_cold_z_base(model, tokenizer, requests, hparams)
        lookup_positions = cold_lookup_positions(
            tokenizer, requests, contexts, fact_token_strategy=hparams.fact_token
        )
    finally:
        pre_action_counter.close()
    target_layer_name = hparams.layer_module_tmp.format(int(hparams.layers[-1]))
    terminal_capture_contract = {
        "schema": "ode-edit-r10-canonical-terminal-capture-contract/v1",
        "module_name_sha256": hashlib.sha256(
            target_layer_name.encode("utf-8")
        ).hexdigest(),
        "module_layer": int(hparams.layers[-1]),
        "capture_side": "MODULE_OUTPUT_POST_HOOK",
        "lookup_token_strategy": str(hparams.fact_token),
        "lookup_position_sha256": canonical_hash(list(lookup_positions)),
        "lookup_position_count": len(lookup_positions),
        "request_order_sha256": request_order,
        "shape": list(z_base.shape),
        "dtype": str(z_base.dtype),
        "device": z_base.device.type,
        "z_base_sha256": tensor_sha256(z_base),
        "capture_count": 1,
        "native_or_direct_z_access_count": 0,
    }
    if (
        terminal_capture_contract["shape"]
        != [int(model.config.hidden_size), BATCH_SIZE]
        or terminal_capture_contract["dtype"] != "torch.float32"
        or terminal_capture_contract["device"] != "cpu"
        or terminal_capture_contract["lookup_position_count"] != 60
    ):
        raise ODEBFContractError("common cold canonical terminal capture differs")
    terminal_capture_contract["identity_sha256"] = canonical_hash(
        terminal_capture_contract
    )
    terminal_capture_sha = write_once(
        raw_root / "common-cold" / "canonical-terminal-capture-contract.json",
        terminal_capture_contract,
    )
    pre_zero = common_terminal_residual_input(z_base, z_base, request_order)
    if torch.count_nonzero(pre_zero.residual) != 0:
        raise ODEBFContractError("common cold pre-bootstrap residual is not zero")
    pre_action_counter = ModelForwardCounter(model, job_ledger)
    try:
        overlay_gate = _context_overlay_gate(
            model,
            tokenizer,
            requests,
            contexts,
            target_layer_name=target_layer_name,
            lookup_positions=lookup_positions,
            z_base=z_base,
        )
    finally:
        pre_action_counter.close()
    overlay_gate_sha = write_once(raw_root / "common-cold" / "context-overlay-gate.json", overlay_gate)
    context_audit = legacy.fixed_e8_context_degeneracy_audit(
        tokenizer, contexts, context_sha256=context_sha256
    )
    context_audit_sha = write_once(
        raw_root / "common-cold" / "context-degeneracy-audit.json",
        context_audit,
    )
    metrics = {
        scale: CommonColdScaleMetric.from_z_base(z_base, request_order, scale)
        for scale in (CommonColdScale.ROBUST_SHARED, CommonColdScale.BATCH_GLOBAL)
    }
    bootstrap_counter = ModelForwardCounter(model, job_ledger)
    try:
        rs_target, rs_bootstrap = common_cold_bootstrap(
            model,
            tokenizer,
            requests,
            contexts,
            target_layer_name=target_layer_name,
            lookup_positions=lookup_positions,
            z_base=z_base,
            metric=metrics[CommonColdScale.ROBUST_SHARED],
            ledger=job_ledger,
        )
        bg_target, bg_bootstrap = common_cold_bootstrap(
            model,
            tokenizer,
            requests,
            contexts,
            target_layer_name=target_layer_name,
            lookup_positions=lookup_positions,
            z_base=z_base,
            metric=metrics[CommonColdScale.BATCH_GLOBAL],
            ledger=job_ledger,
        )
    finally:
        bootstrap_counter.close()
    rs_sha = write_once(raw_root / "common-cold" / "bootstrap-rs.json", rs_bootstrap)
    bg_sha = write_once(raw_root / "common-cold" / "bootstrap-bg.json", bg_bootstrap)
    stages.record("post_bootstrap_contract", {
        "pre_residual_exact_zero": True,
        "rs_bootstrap_sha256": rs_sha,
        "bg_bootstrap_sha256": bg_sha,
        "rs_joint_entry_target_sha256": tensor_sha256(rs_target),
        "bg_joint_entry_target_sha256": tensor_sha256(bg_target),
        "tau_joint": 0.0,
    })
    outer_population = tuple(population_by_sha256[item] for item in theta0_cache.request_order)
    outer_snapshot = _entry_parameter_snapshot_sha256(model, dict(base_receipt.parameter_sha256))
    outer_counter = ModelForwardCounter(model, job_ledger)
    try:
        outer_cache = build_outer_entry_pretrained_cache(
            model, tokenizer, outer_population, theta0_cache, outer_entry_snapshot_sha256=outer_snapshot
        )
    finally:
        outer_counter.close()
    capture = legacy.FixedE8EntryCapture(
        {name: value.detach().cpu().clone() for name, value in base_values.items()},
        dict(base_receipt.parameter_sha256),
    )
    rollouts: dict[str, legacy.FixedE8Rollout] = {}
    failures: dict[str, dict[str, Any]] = {}
    initial_contracts: dict[str, dict[str, Any]] = {}
    for arm in CommonColdArm:
        if _parameter_contract_sha256(touched) != base_contract:
            raise ODEBFStateError("common cold arm W0 entry differs")
        arm_state = ArmRuntimeState(
            P1Arm.R_BF,
            P1HistoryLedger(layer_order=COMMON_COLD_LAYER_ORDER, maximum_records=40),
            ComputeLedger(),
            ArmWeightSnapshot(P1Arm.R_BF, 0, base_receipt.parameter_sha256, canonical_hash({"arm": arm.value, "weights": base_receipt.parameter_sha256})),
            dict(base_values),
        )
        recorder = legacy.FixedE8ReceiptRecorder(raw_root, arm, write_once)
        counter = ModelForwardCounter(model, arm_state.ledger)
        initial_capture: dict[str, Any] = {}
        try:
            expected_initial = (
                initial_contracts.get(CommonColdArm.RS_NEUTRAL.value)
                if arm is CommonColdArm.RS_SOFT
                else None
            )
            rollout, initial_contract = _run_common_arm(
                model,
                tokenizer,
                requests,
                alias=alias,
                arm=arm,
                bootstrap_target=(bg_target if arm is CommonColdArm.BG_NEUTRAL else rs_target),
                z_base=z_base,
                metric=metrics[arm.scale],
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
                recorder=recorder,
                stages=stages,
                expected_initial_contract=expected_initial,
                initial_contract_sink=initial_capture,
            )
            rollouts[arm.value] = rollout
            initial_contracts[arm.value] = initial_contract
        except ODEBFContractError as exc:
            if initial_capture:
                initial_contracts[arm.value] = dict(initial_capture)
            failures[arm.value] = {
                "status": "ARM_LOCAL_TECHNICAL_FAILURE",
                "exception_type": type(exc).__name__,
                "message_sha256": hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
                "completed_snapshot_count": len(recorder.accepted_hashes),
                "receipt_links": recorder.links(),
            }
            write_once(raw_root / "common-cold" / f"failure-{arm.value}.json", failures[arm.value])
            stages.record(f"post_{arm.value.lower().replace('-', '_')}_failure", failures[arm.value])
        finally:
            counter.close()
            _observed_memory(arm_state.ledger)
        if _parameter_contract_sha256(touched) != base_contract:
            raise ODEBFStateError("common cold arm restore differs")
    if CommonColdArm.RS_NEUTRAL.value in initial_contracts and CommonColdArm.RS_SOFT.value in initial_contracts:
        left = dict(initial_contracts[CommonColdArm.RS_NEUTRAL.value])
        right = dict(initial_contracts[CommonColdArm.RS_SOFT.value])
        if left != right:
            raise ODEBFContractError("RS Neutral/Soft initial bootstrap/field differs")
    action_freeze = {
        "schema": f"{COMMON_COLD_SCHEMA_NAMESPACE}-action-freeze/v1",
        "instruction_id": COMMON_COLD_INSTRUCTION_ID,
        "request_order_sha256": request_order,
        "rollout_sha256": {key: value.rollout_sha256 for key, value in rollouts.items()},
        "arm_failures": failures,
        "actions_frozen_before_native_and_heldout": True,
        "native_or_direct_z_cold_access_count": 0,
    }
    action_freeze_sha = write_once(raw_root / "common-cold" / "action-freeze.json", action_freeze)
    native_ledger = ComputeLedger()
    native_counter = ModelForwardCounter(model, native_ledger)
    try:
        native_capture = capture_p1_native_entry(
            model,
            tokenizer,
            requests,
            hparams,
            projector,
            contexts,
            history_keys_by_layer=_history_keys(P1HistoryLedger(layer_order=COMMON_COLD_LAYER_ORDER, maximum_records=40), COMMON_COLD_LAYER_ORDER, risk=False),
            mutation_lock=__import__("threading").RLock(),
            ledger=native_ledger,
            residual_tolerance=controller_lock.residual_tolerance,
        )
        native_online = _evaluate_native_rewrite(
            model, tokenizer, requests, alias=alias, candidates=native_capture.native_candidates, ledger=native_ledger
        )
    finally:
        native_counter.close()
    native_sha = write_once(raw_root / "common-cold" / "N32_NATIVE-postfreeze.json", {
        "action_freeze_sha256": action_freeze_sha,
        "capture": native_capture.raw_free_payload(),
        "official_success": native_online.batch_success.raw_free_payload(),
        "compute": native_ledger.raw_free_payload(),
        "opened_after_action_freeze": True,
    })
    panel, step_receipts = _postfreeze_stepwise_panel(
        model,
        tokenizer,
        alias=alias,
        requests=requests,
        dataset_path=dataset_path,
        capture=native_capture,
        rollouts=rollouts,
        raw_root=raw_root,
        write_once=write_once,
        touched=touched,
        instruction_id=COMMON_COLD_INSTRUCTION_ID,
        schema_namespace=COMMON_COLD_SCHEMA_NAMESPACE,
    )
    panel_sha = write_once(raw_root / "stepwise" / "common-cold-panel.json", panel)
    evaluator_parity = _warm_evaluator_parity_receipt(
        step_receipts,
        alias=alias,
        request_order_sha256=request_order,
    )
    evaluator_parity_sha = write_once(
        raw_root / "stepwise" / "reused-warm-evaluator-parity.json",
        evaluator_parity,
    )
    artifact_guard.assert_unchanged()
    if {name: tensor_sha256(value) for name, value in sorted(touched.items())} != base_bytes:
        raise ODEBFStateError("common cold final W0 restore differs")
    terminal = {
        "schema": f"{COMMON_COLD_SCHEMA_NAMESPACE}-terminal/v1",
        "instruction_id": COMMON_COLD_INSTRUCTION_ID,
        "status": "COMMON_COLD_FIXED_E8_DIAGNOSTIC_TERMINAL",
        "alias": alias,
        "source_head": source_head,
        "method": common_cold_source_contract(),
        "panel_kind": "REUSED_WARMUP_SEAL_CAUSAL_REGRESSION",
        "outcome_selection_bias": "BEST_COMPLETED_WARM_ALLOFF_PANEL_REUSE",
        "unseen_or_fresh_sample_claim_authorized": False,
        "request_order_sha256": request_order,
        "stream_root_digest": stream["root_digest"],
        "warm_source_stream_root_digest": stream["warm_source"][
            "source_stream_root_digest"
        ],
        "warm_source_terminal_sha256": {
            key: value["terminal_sha256"]
            for key, value in stream["warm_source"]["source_roots"].items()
        },
        "historical_warm_alloff_reference": {
            "arm": "FR-A8-NEWNLL-ALLOFF",
            "use": "READ_ONLY_MATCHED_REQUEST_REFERENCE_NOT_CURRENT_BASELINE_SUBSTITUTE",
            "efficacy": {"numerator": 10, "denominator": 10},
            "generalization": {"numerator": 20, "denominator": 20},
            "locality": {"numerator": 80, "denominator": 100},
            "current_w0_and_native_executed": True,
        },
        "prior_qwen_ordinal6_rca_sha256": ordinal_audit_sha,
        "canonical_terminal_capture_contract_sha256": terminal_capture_sha,
        "reused_warm_context_identity_sha256": context_identity_sha,
        "context_overlay_gate_sha256": overlay_gate_sha,
        "context_degeneracy_audit_sha256": context_audit_sha,
        "bootstrap_sha256": {"RS": rs_sha, "BG": bg_sha},
        "action_freeze_sha256": action_freeze_sha,
        "n32_postfreeze_sha256": native_sha,
        "stepwise_panel_sha256": panel_sha,
        "reused_warm_evaluator_parity_sha256": evaluator_parity_sha,
        "variant_status": {arm.value: (rollouts[arm.value].status if arm.value in rollouts else failures[arm.value]["status"]) for arm in CommonColdArm},
        "cold_native_or_direct_z_access_count": 0,
        "scientific_retry_count": 0,
        "scientific_rejection_count": 0,
        "hard_h_p_budget_influence_count": 0,
        "first_hit_observation_only": True,
        "numerical_lock_sha256": numerical_sha256,
        "artifact_receipt": asdict(artifact_receipt),
        "context_sha256": context_sha256,
        "cuda_preflight": dict(cuda_runtime_receipt),
        "job_compute": job_ledger.raw_free_payload(),
        "final_w0_restored": True,
        "persistent_endpoint_commit_count": 0,
        "history_append_count": 0,
        "heldout_controller_access_count": 0,
        "scientific_promotion_authorized": False,
    }
    terminal_sha = write_once(destination / "terminal.json", terminal)
    manifest_sha = write_once(destination / "manifest.json", {
        "schema": f"{COMMON_COLD_SCHEMA_NAMESPACE}-manifest/v1",
        "instruction_id": COMMON_COLD_INSTRUCTION_ID,
        "status": terminal["status"],
        "alias": alias,
        "source_head": source_head,
        "terminal_sha256": terminal_sha,
        "stepwise_panel_sha256": panel_sha,
        "retry_submission_count": 0,
    })
    return {"status": terminal["status"], "alias": alias, "terminal_sha256": terminal_sha, "manifest_sha256": manifest_sha, "final_w0_restored": True}


__all__ = ["CommonColdArm", "run_common_coldcoord_fixed_e8_diagnostic"]
