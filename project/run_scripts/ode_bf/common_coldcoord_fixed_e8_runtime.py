"""R10 common-terminal-coordinate fixed-E8 runtime.

This module reuses the verified R8 transaction, functional-observation and
post-freeze evaluator helpers while replacing only the cold coordinate,
bootstrap, target scale, and three-arm panel locked by R10.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, replace
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
    _maximum_progress,
    solve_fixed_e8_routing,
)
from .functional import WaypointFactor, tensor_sha256
from .p0_runtime import ModelForwardCounter
from .p1_adaptive import fraction_payload
from .p1_adaptive_runtime import (
    FunctionalPDecisionPolicy,
    _controller_replay_entry,
    _evaluate_rewrite,
    _evaluate_stepwise_state,
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
from .p1_evaluator import load_counterfact_cases_after_freeze
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
from .p1_stepwise import (
    StepwiseActionFreeze,
    evaluate_counterfact_stepwise_primary,
)
from .p1_common_coldcoord_fixed_e8_panel import (
    COMMON_COLD_SCHEMA_NAMESPACE,
    WARM_ALLOFF_ROOTS,
)
from .request_digest import ordered_request_digest_v1
from .routing import PreservationConstraintPolicy
from .sampling import StatelessReplaySchedule
from .target_new_nll import RoutingObjective, evaluate_routing_objective
from .strength_preserving_routing import (
    STRENGTH_COVERAGE_EPSILON,
    STRENGTH_PRESERVING_AMENDMENT_ID,
    STRENGTH_PRESERVING_INSTRUCTION_ID,
    STRENGTH_PRESERVING_METHOD_ID,
    route_independent_target_write_identity,
    solve_strength_preserving_routing,
    target_probe_requested_strength,
)
from .cold_start_target import (
    _unwrap_layer_output,
    cold_lookup_positions,
    capture_cold_z_base,
)
from . import fixed_e8_runtime as legacy


class CommonColdArm(str, Enum):
    RS_NEUTRAL = "RS-NEUTRAL"
    RS_NEUTRAL_TARGET_HOLD = "RS-NEUTRAL-TARGET-HOLD"
    RS_SOFT = "RS-SOFT"
    RS_SOFT_TARGET_HOLD = "RS-SOFT-TARGET-HOLD"
    BG_NEUTRAL = "BG-NEUTRAL"
    BG_NEUTRAL_TARGET_HOLD = "BG-NEUTRAL-TARGET-HOLD"
    BG_SOFT = "BG-SOFT"
    BG_SOFT_TARGET_HOLD = "BG-SOFT-TARGET-HOLD"

    @property
    def scale(self) -> CommonColdScale:
        return (
            CommonColdScale.BATCH_GLOBAL
            if self
            in (
                CommonColdArm.BG_NEUTRAL,
                CommonColdArm.BG_NEUTRAL_TARGET_HOLD,
                CommonColdArm.BG_SOFT,
                CommonColdArm.BG_SOFT_TARGET_HOLD,
            )
            else CommonColdScale.ROBUST_SHARED
        )

    @property
    def routing_arm(self) -> FixedE8Arm:
        return (
            FixedE8Arm.SOFT
            if self
            in (
                CommonColdArm.RS_SOFT,
                CommonColdArm.RS_SOFT_TARGET_HOLD,
                CommonColdArm.BG_SOFT,
                CommonColdArm.BG_SOFT_TARGET_HOLD,
            )
            else FixedE8Arm.NEUTRAL
        )

    @property
    def target_hold(self) -> bool:
        return self in (
            CommonColdArm.RS_NEUTRAL_TARGET_HOLD,
            CommonColdArm.RS_SOFT_TARGET_HOLD,
            CommonColdArm.BG_NEUTRAL_TARGET_HOLD,
            CommonColdArm.BG_SOFT_TARGET_HOLD,
        )


R10_COMMON_COLD_ARMS = (
    CommonColdArm.RS_NEUTRAL,
    CommonColdArm.RS_SOFT,
    CommonColdArm.BG_NEUTRAL,
)


def _routing_objective_raw_free_payload(
    result: Any,
    *,
    role: str,
    action_frozen_before_evaluation: bool,
) -> dict[str, Any]:
    """Serialize one six-context objective without request text or token IDs."""

    raw_values = result.per_request_values
    if isinstance(raw_values, torch.Tensor):
        values = tuple(
            float(item)
            for item in raw_values.detach().to(
                device="cpu", dtype=torch.float64
            ).tolist()
        )
        raw_mean = result.loss
        if not isinstance(raw_mean, torch.Tensor) or raw_mean.ndim != 0:
            raise ODEBFContractError(
                "six-context routing objective mean differs"
            )
        mean_target_new_nll = float(
            raw_mean.detach().to(device="cpu", dtype=torch.float64)
        )
    elif isinstance(raw_values, tuple):
        values = tuple(float(item) for item in raw_values)
        raw_mean = result.value
        if isinstance(raw_mean, bool) or not isinstance(raw_mean, (int, float)):
            raise ODEBFContractError(
                "six-context routing objective mean differs"
            )
        mean_target_new_nll = float(raw_mean)
    else:
        raise ODEBFContractError(
            "six-context routing objective request values differ"
        )
    objective = result.objective
    objective_name = (
        objective.value
        if isinstance(objective, RoutingObjective)
        else objective
        if type(objective) is str
        else getattr(objective, "value", None)
    )
    if (
        objective_name != RoutingObjective.TARGET_NEW_NLL.value
        or len(values) != BATCH_SIZE
        or not all(math.isfinite(item) for item in values)
        or not math.isfinite(mean_target_new_nll)
        or not math.isclose(
            mean_target_new_nll,
            math.fsum(values) / BATCH_SIZE,
            rel_tol=1.0e-6,
            abs_tol=1.0e-7,
        )
        or not isinstance(result.request_order_sha256, str)
        or len(result.request_order_sha256) != 64
        or not isinstance(result.target_span_sha256, str)
        or len(result.target_span_sha256) != 64
        or not isinstance(result.context_sha256, str)
        or len(result.context_sha256) != 64
        or tuple(result.context_group_sizes) != (1, 5)
        or result.context_count != 6
        or result.model_forward_count != BATCH_SIZE * 6
        or result.processed_token_count <= 0
        or result.generation_call_count != 0
        or getattr(result, "target_old_access_count", 0) != 0
    ):
        raise ODEBFContractError("six-context routing objective differs")
    payload = {
        "schema": "ode-edit-s05-bg-soft-six-context-objective/v1",
        "role": role,
        "objective": objective_name,
        "mean_target_new_nll": mean_target_new_nll,
        "per_request_target_new_nll": list(values),
        "request_order_sha256": result.request_order_sha256,
        "target_span_sha256": result.target_span_sha256,
        "context_sha256": result.context_sha256,
        "context_group_sizes": list(result.context_group_sizes),
        "context_count": result.context_count,
        "model_forward_count": result.model_forward_count,
        "processed_token_count": result.processed_token_count,
        "generation_call_count": result.generation_call_count,
        "target_old_access_count": 0,
        "action_frozen_before_evaluation": action_frozen_before_evaluation,
        "controller_decision_influence_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


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


def _bg_soft_initial_contract(
    *,
    bootstrap_target: torch.Tensor,
    field: Any,
    field_receipt: legacy.FixedE8FieldReceipt,
    routing: Any,
    metric: CommonColdScaleMetric,
    built: Any,
    inventory: Any,
    probe: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze every common pre-Soft input against R10 BG-Neutral."""

    base = _common_initial_contract(
        bootstrap_target=bootstrap_target,
        field=field,
        field_receipt=field_receipt,
        routing=routing,
        metric=metric,
    )
    base.pop("identity_sha256")
    required_probe_keys = (
        "schema",
        "field_semantic_sha256",
        "factor_state_sha256",
        "controller_p_sample_order_sha256",
        "controller_p_baseline_identity_sha256",
        "controller_p_schedule_sha256",
        "inventory",
        "baseline",
        "probe_endpoints",
        "exact_basis_endpoint_count",
        "baseline_endpoint_count",
        "layer_probe_endpoint_count",
        "functional_h_additional_probe_endpoint_count",
        "historical_soft_status",
        "legacy_one_e_minus_three_hinge_access_count",
        "routing_functional_p_definition",
        "functional_p_floor_correction_routing_influence_count",
        "hard_functional_decision_influence_count",
        "counter_delta",
    )
    if any(key not in probe for key in required_probe_keys):
        raise ODEBFContractError("BG-Soft functional probe schema differs")
    functional_probe_semantic_sha256 = canonical_hash(
        {key: probe[key] for key in required_probe_keys}
    )
    base.update(
        {
            "schema": "ode-edit-s05-bg-soft-initial-contract/v1",
            "problem_sha256": built.problem.identity(),
            "functional_inventory_sha256": inventory.raw_free_payload()[
                "identity_sha256"
            ],
            "functional_probe_semantic_sha256": (
                functional_probe_semantic_sha256
            ),
            "timing_bearing_field_and_probe_receipts_excluded": True,
            "selected_routing_velocity_excluded_from_identity": True,
            "first_allowed_semantic_difference": "SELECTED_ROUTING_VELOCITY",
        }
    )
    base["identity_sha256"] = canonical_hash(base)
    return base


def _partial_fixed_e8_clock_receipt(clock: FixedE8Clock) -> dict[str, Any]:
    payload = {
        "schema": "ode-edit-fixed-e8-partial-clock/v1",
        "grid_count": clock.step_index,
        "field_count": clock.field_count,
        "h": float(FIXED_E8_H),
        "tau_final": float(clock.tau),
        "scientific_retry_count": clock.scientific_retry_count,
        "scientific_rejection_count": 0,
        "adaptive_clock_access_count": 0,
        "backtracking_count": 0,
        "termination_label": "ZERO_POSITIVE_DIRECTION",
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _bg_soft_d1_operation_accounting(
    *,
    routing_qp: int,
    routing_backend: int,
    nohook_qp: int,
    nohook_backend: int,
    nohook_fallback: int,
    ledger: ComputeLedger,
) -> dict[str, Any]:
    """Reconcile D1 diagnostic QPs with the arm compute ledger."""

    values = {
        "routing_qp": routing_qp,
        "routing_backend": routing_backend,
        "nohook_qp": nohook_qp,
        "nohook_backend": nohook_backend,
        "nohook_fallback": nohook_fallback,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values.values()
    ):
        raise ODEBFContractError("BG-Soft D1 operation count differs")
    all_qp = routing_qp + nohook_qp
    if (
        ledger.counters["qp_solve"] != all_qp
        or ledger.counters["qp_certificate"] != all_qp
    ):
        raise ODEBFStateError("BG-Soft D1 QP accounting differs")
    payload = {
        "actual_routing_logical_qp_certificate_count": routing_qp,
        "actual_d1_nohook_logical_qp_certificate_count": nohook_qp,
        "actual_all_logical_qp_certificate_count": all_qp,
        "actual_routing_optimizer_backend_invocation_count": (
            routing_backend
        ),
        "actual_d1_nohook_optimizer_backend_invocation_count": (
            nohook_backend
        ),
        "actual_all_optimizer_backend_invocation_count": (
            routing_backend + nohook_backend
        ),
        "actual_d1_nohook_fallback_backend_invocation_count": (
            nohook_fallback
        ),
        "d1_nohook_qp_role": "OBSERVATION_ONLY",
        "d1_nohook_controller_decision_influence_count": 0,
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
    typed_zero_positive: bool = False,
    observability_contract: Mapping[str, Any] | None = None,
    strength_preserving: bool = False,
) -> tuple[
    Any,
    Any,
    Any,
    Any,
    Any,
    torch.Tensor | None,
    Mapping[str, Any] | None,
    Any,
    Mapping[str, Any],
    Mapping[str, Any],
]:
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
    nohook_signed = (
        legacy._fixed_e8_signed_progress_gradient(
            model,
            tokenizer,
            requests,
            field,
            cumulative_factors_by_weight=current_factors,
            contexts=contexts,
            ledger=ledger,
        )
        if typed_zero_positive or strength_preserving
        else None
    )
    semantic = _common_field_semantic_receipt(
        field, solve_history=solve_history, risk_history=risk_history
    )
    authoritative_signed = nohook_signed if strength_preserving else signed
    if authoritative_signed is None:
        raise ODEBFStateError("strength-preserving nohook slope is absent")
    built = legacy._build_fixed_e8_problem(
        field,
        authoritative_signed,
        accepted_by_layer=accepted_by_layer,
        committed_load_by_layer=history.cumulative_load(),
        lock=lock,
        current_history_action_by_layer=_history_actions(field, history),
    )
    nohook_values = (
        np.asarray(nohook_signed.signed_progress, dtype=np.float64)
        if nohook_signed is not None
        else np.asarray([], dtype=np.float64)
    )
    nohook_active = np.flatnonzero(nohook_values > 0.0)
    nohook_certificate_payload: Mapping[str, Any] | None
    if strength_preserving:
        nohook_p_max = None
        nohook_certificate_payload = None
    elif nohook_signed is not None and nohook_active.size:
        nohook_certificates: list[Any] = []

        def observe_nohook(certificate: Any) -> None:
            nohook_certificates.append(certificate)
            recorder.solver(
                {
                    "step_index": step_index,
                    "arm": arm.value,
                    "certificate_sequence_index": len(nohook_certificates) - 1,
                    "persistence_point": (
                        "D1_NOHOOK_OBSERVATION_BEFORE_OVERLAY_POSITIVE_GUARD"
                    ),
                    "status": (
                        "D1_NOHOOK_MAXIMUM_PROGRESS_CERTIFIED"
                        if certificate.passed
                        else "D1_NOHOOK_MAXIMUM_PROGRESS_UNCERTIFIED"
                    ),
                    "field_sha256": field.identity_sha256,
                    "field_semantic_sha256": semantic[
                        "semantic_identity_sha256"
                    ],
                    "problem_sha256": built.problem.identity(),
                    "certificate": certificate.raw_free_payload(),
                    "authority_role": "OBSERVATION_ONLY_D1_NOHOOK",
                    "controller_decision_influence_count": 0,
                    "scientific_retry_count": 0,
                }
            )

        nohook_problem = replace(
            built.problem,
            signed_progress=nohook_values,
        )
        _, nohook_p_max, nohook_certificate = _maximum_progress(
            nohook_problem,
            nohook_active,
            certificate_observer=observe_nohook,
            fail_closed=False,
        )
        if nohook_certificates != [nohook_certificate]:
            raise ODEBFStateError(
                "BG-Soft D1 nohook certificate sequence differs"
            )
        nohook_certificate_payload = nohook_certificate.raw_free_payload()
        ledger.increment("qp_solve", 1)
        ledger.increment("qp_certificate", 1)
    elif nohook_signed is not None:
        nohook_p_max = 0.0
        nohook_certificate_payload = None
    else:
        nohook_p_max = None
        nohook_certificate_payload = None
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
    target_demand_receipt: Mapping[str, Any] | None = None
    if strength_preserving:
        target_started = time.perf_counter()
        target_velocity, target_receipt = write_aware_common_target_velocity(
            model,
            tokenizer,
            requests,
            contexts,
            target_layer_name=target_layer_name,
            lookup_positions=lookup_positions,
            field=field,
            velocity_coefficients=(0.0,) * len(COMMON_COLD_LAYER_ORDER),
            cumulative_factors_by_weight=current_factors,
            metric=metric,
            ledger=ledger,
        )
        target_wall = time.perf_counter() - target_started
        alpha_req, target_demand_receipt = target_probe_requested_strength(
            target_receipt
        )
        router_counter_before = dict(ledger.counters)
        solve_started = time.perf_counter()
        routing = solve_strength_preserving_routing(
            built.problem,
            inventory,
            arm=arm.routing_arm,
            alpha_req=alpha_req,
        )
        solve_wall = time.perf_counter() - solve_started
        router_counter_delta = {
            key: int(ledger.counters[key] - router_counter_before.get(key, 0))
            for key in sorted(ledger.counters)
        }
        forbidden_router_counter_delta = {
            key: value
            for key, value in router_counter_delta.items()
            if value != 0
        }
        if forbidden_router_counter_delta:
            raise ODEBFStateError(
                "strength router added model/accounting work"
            )
        for sequence_index, certificate in enumerate(routing.certificates):
            recorder.solver(
                {
                    "step_index": step_index,
                    "arm": arm.value,
                    "certificate_sequence_index": sequence_index,
                    "persistence_point": "EXACT_STRENGTH_SOLVER_BEFORE_WRITE",
                    "status": "SOLVER_CERTIFICATE_PASSED",
                    "field_sha256": field.identity_sha256,
                    "field_semantic_sha256": semantic[
                        "semantic_identity_sha256"
                    ],
                    "problem_sha256": built.problem.identity(),
                    "functional_inventory_sha256": inventory.raw_free_payload()[
                        "identity_sha256"
                    ],
                    "certificate": certificate.raw_free_payload(),
                    "authority_role": (
                        "AUTHORITATIVE"
                        if certificate in routing.authoritative_certificates
                        else "DIAGNOSTIC_MATCHED_ARM"
                    ),
                    "scientific_retry_count": 0,
                }
            )
        ledger.increment("qp_solve", len(routing.certificates))
        ledger.increment("qp_certificate", len(routing.certificates))
    else:
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
                if arm.routing_arm is FixedE8Arm.SOFT
                else FixedE8Stage2FailurePolicy.FAIL_CLOSED
            ),
        )
        if tuple(observed) != routing.certificates:
            raise ODEBFStateError("common cold solver receipt sequence differs")
        target_wall = 0.0
        solve_wall = 0.0
        router_counter_delta = {}
    slope_comparison: Mapping[str, Any] | None = None
    if nohook_signed is not None and (
        nohook_p_max is not None or strength_preserving
    ):
        from .bg_soft_diagnostics import signed_slope_comparison_receipt

        slope_comparison = signed_slope_comparison_receipt(
            signed.signed_progress,
            nohook_signed.signed_progress,
            overlay_p_max=(
                float(FIXED_E8_H)
                * float(
                    np.maximum(
                        np.asarray(signed.signed_progress, dtype=np.float64),
                        0.0,
                    )
                    @ np.minimum(built.problem.layer_caps, 1.0)
                )
                if strength_preserving
                else float(routing.p_max)
            ),
            no_hook_p_max=(
                float(routing.alpha_max)
                if strength_preserving
                else float(nohook_p_max)
            ),
            layer_order=COMMON_COLD_LAYER_ORDER,
        )
        slope_comparison = {
            **slope_comparison,
            "overlay_signed_progress_receipt_sha256": (
                signed.objective_receipt_sha256
            ),
            "nohook_signed_progress_receipt_sha256": (
                nohook_signed.objective_receipt_sha256
            ),
            "nohook_maximum_progress_certificate": (
                nohook_certificate_payload
            ),
            "nohook_controller_decision_influence_count": 0,
            "nohook_numerical_certificate_required": bool(
                nohook_active.size and not strength_preserving
            ),
            "nohook_numerical_certificate_status": (
                "NOT_APPLICABLE_NO_POSITIVE_DIRECTION"
                if not nohook_active.size
                else "AUTHORITATIVE_STRENGTH_CERTIFIED"
                if strength_preserving
                else (
                    "PASS"
                    if nohook_certificate_payload is not None
                    and nohook_certificate_payload["passed"]
                    else "FAIL"
                )
            ),
            "nohook_numerical_certificate_passed": (
                None
                if not nohook_active.size
                else True
                if strength_preserving
                else bool(
                    nohook_certificate_payload is not None
                    and nohook_certificate_payload["passed"]
                )
            ),
        }
        if (
            slope_comparison["nohook_numerical_certificate_required"]
            and not slope_comparison[
                "nohook_numerical_certificate_passed"
            ]
        ):
            slope_comparison = {
                **slope_comparison,
                "classification": "NOHOOK_NUMERIC_UNAVAILABLE",
                "classification_decision_influence_count": 0,
            }
    zero_positive = bool(
        not strength_preserving
        and (
            routing.mode is not FixedE8StepMode.JOINT_WRITE
            or routing.p_max <= 0.0
        )
    )
    if zero_positive and not typed_zero_positive:
        raise ODEBFContractError("BOOTSTRAP_READINESS_FAILED: p_max is nonpositive")
    if not strength_preserving:
        ledger.increment("qp_solve", len(routing.certificates))
        ledger.increment("qp_certificate", len(routing.certificates))
    if zero_positive:
        target_velocity = None
        target_receipt = {
            "schema": "ode-edit-common-cold-target-velocity-not-computed/v1",
            "status": "ZERO_POSITIVE_DIRECTION",
            "velocity_sha256": canonical_hash(
                {
                    "status": "ZERO_POSITIVE_DIRECTION",
                    "field_sha256": field.identity_sha256,
                    "step_index": step_index,
                }
            ),
            "model_forward_count": 0,
            "target_backward_count": 0,
            "candidate_count": 0,
            "clock_advance_count": 0,
            "decision": "PRESERVE_VALID_PREFIX_AND_TERMINATE_ARM",
        }
    elif not strength_preserving:
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
        "authoritative_physical_nohook_signed_progress": (
            None if nohook_signed is None else asdict(nohook_signed)
        ),
        "signed_progress_overlay": signed_overlay,
        "problem_receipt_sha256": built.identity_sha256,
        "functional_basis": inventory.raw_free_payload(),
        "probe_receipt": probe,
        "routing": routing.raw_free_payload(),
        "target_velocity": target_receipt,
        "target_demand": target_demand_receipt,
        "target_probe_and_write_h_equal": True,
        "h": float(FIXED_E8_H),
        "scientific_retry_count": 0,
        "strength_preserving": strength_preserving,
        "router_added_counter_delta": router_counter_delta,
        "router_added_model_forward_count": router_counter_delta.get(
            "model_forward", 0
        ),
        "router_added_backward_count": router_counter_delta.get("backward", 0),
        "router_added_target_backward_count": router_counter_delta.get(
            "target_backward", 0
        ),
        "router_added_processed_token_count": router_counter_delta.get(
            "processed_tokens", 0
        ),
        "phase_wall_seconds": {
            "target_gradient_common_zero_probe": target_wall,
            "detached_route_solve": solve_wall,
            "functional_probe": float(probe.get("wall_seconds", 0.0)),
        },
    }
    if typed_zero_positive:
        field_payload.update(
            {
                "typed_zero_positive_enabled": True,
                "termination_candidate": (
                    "ZERO_POSITIVE_DIRECTION" if zero_positive else None
                ),
                "field_persisted_before_positive_guard": True,
            }
        )
    if (
        (typed_zero_positive or strength_preserving)
        and nohook_signed is not None
        and slope_comparison is not None
    ):
        field_payload.update(
            {
                "nohook_signed_progress": asdict(nohook_signed),
                "overlay_vs_nohook_signed_slopes": slope_comparison,
            }
        )
    if observability_contract is not None:
        field_payload.update(
            {
                "observability_contract": dict(observability_contract),
                "observability_total_decision_influence_count": 0,
            }
        )
    persisted = recorder.field(field_payload)
    receipt_payload = {
        "field_sha256": field.identity_sha256,
        "field_semantic_sha256": semantic["semantic_identity_sha256"],
        "signed_progress_sha256": canonical_hash(
            list(authoritative_signed.signed_progress)
        ),
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
    return (
        field,
        authoritative_signed,
        built,
        inventory,
        routing,
        target_velocity,
        target_receipt,
        field_receipt,
        probe,
        slope_comparison,
    )


def _single_write_audit_point(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    hparams: Any,
    state: Mapping[str, Any],
    touched: Mapping[str, torch.nn.Parameter],
    history: P1HistoryLedger,
    schedule: StatelessReplaySchedule,
) -> dict[str, Any]:
    """Measure one-layer and atomic BF16 lookup displacement, state-pure."""

    from .bg_soft_diagnostics import single_write_audit_summary

    field = state["field"]
    increment = state["increment"]
    source_factors = state["source_factors"]
    source_terminal = state["source_terminal"]
    intended_delta = state["intended_delta"]

    def controller_signature() -> str:
        factor_payload: dict[str, list[dict[str, Any]]] = {}
        for weight_name, factors in sorted(source_factors.items()):
            factor_payload[
                hashlib.sha256(weight_name.encode("utf-8")).hexdigest()
            ] = [
                {
                    "order_key": list(item.order_key),
                    "theta": float(item.theta),
                    "left_sha256": tensor_sha256(item.left),
                    "right_sha256": tensor_sha256(item.right),
                }
                for item in factors
            ]
        increment_payload = [
            {
                "layer": int(layer.layer),
                "theta": float(increment[layer.weight_name].theta),
                "left_sha256": tensor_sha256(
                    increment[layer.weight_name].left
                ),
                "right_sha256": tensor_sha256(
                    increment[layer.weight_name].right
                ),
            }
            for layer in field.layers
        ]
        return canonical_hash(
            {
                "field_sha256": field.identity_sha256,
                "source_target_sha256": tensor_sha256(
                    state["source_target"]
                ),
                "source_terminal_sha256": tensor_sha256(source_terminal),
                "intended_delta_sha256": tensor_sha256(intended_delta),
                "source_factors": factor_payload,
                "selected_increment": increment_payload,
            }
        )

    controller_before = controller_signature()
    source_signature = (
        _parameter_contract_sha256(touched),
        history.snapshot().digest,
        schedule.state_digest,
        legacy._rng_identity(),
        controller_before,
    )
    layer_deltas: list[torch.Tensor] = []
    layer_receipts: list[dict[str, Any]] = []
    for layer in field.layers:
        factor = increment[layer.weight_name]
        candidate = _merge_factors(
            source_factors, {layer.weight_name: factor}
        )
        with _virtual_context(model, candidate):
            endpoint = capture_cold_z_base(
                model, tokenizer, requests, hparams
            )
        if source_signature != (
            _parameter_contract_sha256(touched),
            history.snapshot().digest,
            schedule.state_digest,
            legacy._rng_identity(),
            controller_signature(),
        ):
            raise ODEBFStateError(
                "BG-Soft single-write audit mutated source state"
            )
        delta = (endpoint - source_terminal).contiguous()
        layer_deltas.append(delta)
        layer_receipts.append(
            {
                "layer": int(layer.layer),
                "weight_name_sha256": hashlib.sha256(
                    layer.weight_name.encode("utf-8")
                ).hexdigest(),
                "coefficient": float(factor.theta),
                "factor_left_sha256": tensor_sha256(factor.left),
                "factor_right_sha256": tensor_sha256(factor.right),
                "bf16_endpoint_sha256": tensor_sha256(endpoint),
                "lookup_delta_sha256": tensor_sha256(delta),
            }
        )
    all_candidate = _merge_factors(source_factors, increment)
    with _virtual_context(model, all_candidate):
        all_endpoint = capture_cold_z_base(model, tokenizer, requests, hparams)
    if source_signature != (
        _parameter_contract_sha256(touched),
        history.snapshot().digest,
        schedule.state_digest,
        legacy._rng_identity(),
        controller_signature(),
    ):
        raise ODEBFStateError(
            "BG-Soft all-layer write audit mutated source state"
        )
    all_delta = (all_endpoint - source_terminal).contiguous()
    if tensor_sha256(all_endpoint) != tensor_sha256(
        state["online_candidate_terminal"]
    ):
        raise ODEBFStateError(
            "BG-Soft all-layer audit endpoint differs from online candidate"
        )
    summary = single_write_audit_summary(
        intended_delta,
        tuple(layer_deltas),
        all_delta,
        layer_order=COMMON_COLD_LAYER_ORDER,
    )
    payload = {
        "schema": "ode-edit-s05-bg-soft-single-write-audit/v1",
        "arm": str(state["arm"]),
        "audit_role": str(state["audit_role"]),
        "source_step_index": int(state["step_index"]),
        "source_field_sha256": field.identity_sha256,
        "source_target_sha256": tensor_sha256(state["source_target"]),
        "source_terminal_sha256": tensor_sha256(source_terminal),
        "intended_delta_sha256": tensor_sha256(intended_delta),
        "single_layer": layer_receipts,
        "all_layer": {
            "bf16_endpoint_sha256": tensor_sha256(all_endpoint),
            "online_candidate_endpoint_sha256": tensor_sha256(
                state["online_candidate_terminal"]
            ),
            "online_candidate_endpoint_exact": True,
            "lookup_delta_sha256": tensor_sha256(all_delta),
        },
        "summary": summary,
        "parameter_history_rng_controller_before_sha256": canonical_hash(
            list(source_signature)
        ),
        "parameter_history_rng_controller_after_sha256": canonical_hash(
            list(source_signature)
        ),
        "controller_state_sha256": controller_before,
        "restoration_exact_after_every_probe": True,
        "single_layer_probe_forward_count": len(COMMON_COLD_LAYER_ORDER),
        "all_layer_probe_forward_count": 1,
        "total_probe_forward_count": len(COMMON_COLD_LAYER_ORDER) + 1,
        "controller_decision_influence_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


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
    pair_initial_contract_sink: dict[str, Any] | None = None,
    typed_zero_positive: bool = False,
    record_six_context_objectives: bool = False,
    observability_contract: Mapping[str, Any] | None = None,
    strength_preserving: bool = False,
) -> tuple[legacy.FixedE8Rollout, dict[str, Any]]:
    clock = FixedE8Clock()
    history = arm_state.history
    ledger = arm_state.ledger
    before_model = _parameter_contract_sha256(touched)
    before_history = history.snapshot().digest
    before_sampler = schedule.state_digest
    before_rng = legacy._rng_identity()
    current_target = bootstrap_target.clone()
    entry_target_sha256 = tensor_sha256(current_target)
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
    entry_target_assisted = (
        _routing_objective_raw_free_payload(
            current_objective,
            role="TARGET_ASSISTED_RESIDUAL_OVERLAY_OBJECTIVE",
            action_frozen_before_evaluation=False,
        )
        if record_six_context_objectives
        else None
    )
    entry_eval = _evaluate_rewrite(
        model, tokenizer, requests, alias=alias, factors=current_factors, ledger=ledger
    )
    snapshots: list[legacy.FixedE8Snapshot] = []
    first_hit = legacy.FixedE8HitTracker()
    total_qp = total_backend = total_solver_fallback = total_stage1_fallback = 0
    total_d1_nohook_qp = 0
    total_d1_nohook_backend = 0
    total_d1_nohook_fallback = 0
    initial_contract: dict[str, Any] | None = None
    zero_positive_step: int | None = None
    zero_positive_field_receipt: str | None = None
    zero_positive_payload: dict[str, Any] | None = None
    entry_audit_state: dict[str, Any] | None = None
    boundary_audit_state: dict[str, Any] | None = None
    authoritative_weight_write_count = 0
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
        field_wall_started = time.perf_counter()
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
            slope_comparison,
        ) = _build_common_field(
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
            typed_zero_positive=typed_zero_positive,
            observability_contract=observability_contract,
            strength_preserving=strength_preserving,
        )
        field_wall = time.perf_counter() - field_wall_started
        if strength_preserving:
            functional_basis_wall = float(probe.get("wall_seconds", 0.0))
            ledger.add_time(
                "p1r19_edit_core_field_target_route",
                wall_seconds=max(field_wall - functional_basis_wall, 0.0),
            )
            ledger.add_time(
                "p1r19_functional_probe",
                wall_seconds=functional_basis_wall,
            )
        solver_accounting = (
            {
                "actual_logical_qp_certificate_count": len(routing.certificates),
                "actual_optimizer_backend_invocation_count": len(routing.certificates),
                "actual_fallback_backend_invocation_count": 0,
            }
            if strength_preserving
            else legacy._fixed_e8_solver_accounting(routing)
        )
        total_qp += int(solver_accounting["actual_logical_qp_certificate_count"])
        total_backend += int(solver_accounting["actual_optimizer_backend_invocation_count"])
        total_solver_fallback += int(solver_accounting["actual_fallback_backend_invocation_count"])
        total_stage1_fallback += routing.stage1_selection_fallback_count
        if slope_comparison is not None:
            nohook_certificate = slope_comparison.get(
                "nohook_maximum_progress_certificate"
            )
            if nohook_certificate is not None:
                total_d1_nohook_qp += 1
                total_d1_nohook_backend += int(
                    nohook_certificate["optimizer_pass_count"]
                )
                total_d1_nohook_fallback += int(
                    nohook_certificate["fallback_invocation_count"]
                )
        if step_index == 0:
            initial_contract = (
                _bg_soft_initial_contract(
                    bootstrap_target=bootstrap_target,
                    field=field,
                    field_receipt=field_receipt,
                    routing=routing,
                    metric=metric,
                    built=built,
                    inventory=inventory,
                    probe=probe,
                )
                if typed_zero_positive
                else _common_initial_contract(
                    bootstrap_target=bootstrap_target,
                    field=field,
                    field_receipt=field_receipt,
                    routing=routing,
                    metric=metric,
                )
            )
            if initial_contract_sink is not None:
                initial_contract_sink.clear()
                initial_contract_sink.update(initial_contract)
            if pair_initial_contract_sink is not None:
                if slope_comparison is None:
                    raise ODEBFStateError(
                        "BG-Soft initial D1 receipt is absent"
                    )
                pair_contract = {
                    "schema": (
                        "ode-edit-s05-bg-soft-main-target-hold-"
                        "initial-contract/v1"
                    ),
                    "bootstrap_target_sha256": tensor_sha256(
                        bootstrap_target
                    ),
                    "field_semantic_sha256": (
                        field_receipt.field_semantic_sha256
                    ),
                    "problem_sha256": built.problem.identity(),
                    "functional_inventory_sha256": (
                        inventory.raw_free_payload()["identity_sha256"]
                    ),
                    "probe_semantic_sha256": initial_contract[
                        "functional_probe_semantic_sha256"
                    ],
                    "overlay_vs_nohook_signed_slopes_sha256": (
                        slope_comparison["identity_sha256"]
                    ),
                    "selected_velocity": [
                        float(item) for item in routing.velocity
                    ],
                    "selected_velocity_sha256": canonical_hash(
                        [float(item) for item in routing.velocity]
                    ),
                    "target_velocity_sha256": target_receipt[
                        "velocity_sha256"
                    ],
                    "first_allowed_difference": (
                        "MAIN_TARGET_ADVANCEMENT_AFTER_K1"
                    ),
                }
                pair_contract["identity_sha256"] = canonical_hash(
                    pair_contract
                )
                pair_initial_contract_sink.clear()
                pair_initial_contract_sink.update(pair_contract)
            if (
                expected_initial_contract is not None
                and initial_contract != dict(expected_initial_contract)
            ):
                raise ODEBFContractError(
                    "common cold pre-routing initial contract differs"
                )
        if target_velocity is None:
            if not typed_zero_positive:
                raise ODEBFStateError(
                    "common cold zero-positive target velocity is absent"
                )
            zero_positive_step = step_index
            if slope_comparison is None:
                raise ODEBFStateError(
                    "BG-Soft zero-positive D1 receipt is absent"
                )
            zero_positive_field_receipt = (
                field_receipt.identity_sha256
            )
            zero_positive_payload = {
                "status": "ZERO_POSITIVE_DIRECTION",
                "step_index": step_index,
                "tau": fraction_payload(clock.tau),
                "field_sha256": field.identity_sha256,
                "field_semantic_sha256": field_receipt.field_semantic_sha256,
                "field_receipt_sha256": field_receipt.identity_sha256,
                "ordered_signed_slopes": [
                    float(item) for item in routing.signed_slopes
                ],
                "overlay_vs_nohook_signed_slopes": slope_comparison,
                "classification": slope_comparison["classification"],
                "positive_mask": [
                    float(item) > 0.0 for item in routing.signed_slopes
                ],
                "positive_count": sum(
                    float(item) > 0.0 for item in routing.signed_slopes
                ),
                "p_max": float(routing.p_max),
                "mode": routing.mode.value,
                "candidate_count": 0,
                "clock_advance_count": 0,
                "valid_prefix_snapshot_sha256": [
                    item.snapshot_sha256 for item in snapshots
                ],
            }
            zero_positive_payload["identity_sha256"] = canonical_hash(
                zero_positive_payload
            )
            stages.record(
                f"post_{arm.value.lower().replace('-', '_')}_zero_positive",
                {
                    "arm": arm.value,
                    "step": step_index,
                    "tau": float(clock.tau),
                    "p_max": float(routing.p_max),
                    "positive_count": sum(
                        float(item) > 0.0 for item in routing.signed_slopes
                    ),
                    "field_receipt_sha256": field_receipt.identity_sha256,
                    "candidate_count": 0,
                    "clock_advance_count": 0,
                },
            )
            break
        increment = legacy.fixed_e8_waypoint_factors(
            field, routing.velocity, step_index=step_index
        )
        target_write_identity = (
            route_independent_target_write_identity(
                routing,
                [
                    increment[item.weight_name].theta
                    for item in field.layers
                ],
                target_receipt,
            )
            if strength_preserving
            else legacy.fixed_e8_target_write_coefficient_identity(
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
        )
        target_only_zero_write = bool(
            strength_preserving
            and routing.alpha_apply == 0.0
        )
        candidate_factors = (
            _factor_map(current_factors)
            if target_only_zero_write
            else _merge_factors(current_factors, increment)
        )
        proposed_target_trial = (
            current_target
            + float(FIXED_E8_H)
            * target_velocity.detach().to(device="cpu", dtype=torch.float32)
        ).contiguous()
        target_trial = (
            current_target.clone()
            if arm.target_hold
            else proposed_target_trial
        )
        if not torch.isfinite(target_trial).all():
            raise ODEBFContractError("common cold target trial is non-finite")
        candidate_core_started = time.perf_counter()
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
        if strength_preserving:
            if signed.mean_objective_value is None:
                raise ODEBFStateError(
                    "strength physical W-only current objective is absent"
                )
            with _virtual_context(model, candidate_factors):
                candidate_weight_objective = evaluate_routing_objective(
                    model,
                    tokenizer,
                    requests,
                    objective=RoutingObjective.TARGET_NEW_NLL,
                    contexts=contexts,
                )
            candidate_objective = None
            candidate_overlay = {
                "status": "NOT_EVALUATED_P1R19_PHYSICAL_WONLY_AUTHORITY",
                "controller_decision_influence_count": 0,
            }
            actual_progress = float(
                signed.mean_objective_value
                - float(candidate_weight_objective.loss.detach().cpu())
            )
        else:
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
            candidate_weight_objective = None
            actual_progress = float(
                current_objective.value - candidate_objective.value
            )
        predicted_progress = float(
            built.problem.signed_progress @ np.asarray(routing.velocity)
        )
        rho = actual_progress / max(
            predicted_progress, STRENGTH_COVERAGE_EPSILON
            if strength_preserving
            else lock.minimum_progress,
        )
        candidate_core_wall = time.perf_counter() - candidate_core_started
        if strength_preserving:
            ledger.add_time(
                "p1r19_edit_core_candidate_write_realization",
                wall_seconds=candidate_core_wall,
            )
        functional_started = time.perf_counter()
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
        functional_trial_wall = time.perf_counter() - functional_started
        if strength_preserving:
            ledger.add_time(
                "p1r19_functional_probe",
                wall_seconds=functional_trial_wall,
            )
        online_eval_started = time.perf_counter()
        evaluation = _evaluate_rewrite(
            model,
            tokenizer,
            requests,
            alias=alias,
            factors=candidate_factors,
            ledger=ledger,
        )
        online_eval_wall = time.perf_counter() - online_eval_started
        if strength_preserving:
            ledger.add_time(
                "p1r19_online_efficacy_observation",
                wall_seconds=online_eval_wall,
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
                "role": (
                    "OBSERVATION_ONLY_UNCALIBRATED"
                    if strength_preserving
                    else "TECHNICAL_INTEGRATION_BOUND"
                ),
                "decision_influence_count": 0 if strength_preserving else 1,
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
        if strength_preserving:
            structural_payload["online_feasibility_observation"]["components"][
                "write_trust"
            ].update(
                {
                    "status": "OBSERVATION_ONLY_UNCALIBRATED",
                    "decision_influence_count": 0,
                }
            )
            structural_payload.update(
                {
                    "hard_structural_h_budget_influence_count": 0,
                    "hard_structural_p_budget_influence_count": 0,
                    "hard_functional_h_budget_influence_count": 0,
                    "hard_functional_p_budget_influence_count": 0,
                    "trust_threshold_influence_count": 0,
                    "functional_veto_count": 0,
                }
            )
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
        if strength_preserving:
            q_values = np.asarray(
                routing.contribution_share, dtype=np.float64
            )
            positive_q = np.maximum(q_values, 0.0)
            q_total = float(positive_q.sum())
            q_distribution = (
                positive_q / q_total
                if q_total > STRENGTH_COVERAGE_EPSILON
                else np.zeros_like(positive_q)
            )
            nonzero_q = q_distribution[q_distribution > 0.0]
            entropy = float(
                -np.sum(nonzero_q * np.log(nonzero_q))
                / math.log(len(COMMON_COLD_LAYER_ORDER))
            ) if nonzero_q.size else 0.0
            hhi = float(q_distribution @ q_distribution)
            progress_payload.update(
                {
                    "actual_definition": "PHYSICAL_BF16_WONLY_TARGET_NEW_NLL_PROGRESS",
                    "actual_is_weight_only_progress": True,
                    "alpha_req": routing.alpha_req,
                    "alpha_max": routing.alpha_max,
                    "alpha_apply": routing.alpha_apply,
                    "coverage": routing.coverage,
                    "equality_residual": routing.equality_residual,
                    "realization_ratio": actual_progress
                    / (routing.alpha_apply + STRENGTH_COVERAGE_EPSILON),
                    "predicted_actual_same_sign": (
                        predicted_progress == 0.0
                        or actual_progress == 0.0
                        or math.copysign(1.0, predicted_progress)
                        == math.copysign(1.0, actual_progress)
                    ),
                    "per_layer_predicted_progress": [
                        float(slope * coefficient)
                        for slope, coefficient in zip(
                            routing.signed_slopes,
                            routing.velocity,
                            strict=True,
                        )
                    ],
                    "contribution_share": q_values.tolist(),
                    "contribution_share_sum": float(q_values.sum()),
                    "top_layer": (
                        int(COMMON_COLD_LAYER_ORDER[int(np.argmax(q_distribution))])
                        if q_total > STRENGTH_COVERAGE_EPSILON
                        else None
                    ),
                    "top_layer_share": float(q_distribution.max(initial=0.0)),
                    "normalized_entropy": entropy,
                    "hhi": hhi,
                    "effective_layer_count": 0.0 if hhi == 0.0 else 1.0 / hhi,
                    "active_layer_count": routing.active_layer_count,
                    "equality_rank": routing.equality_rank,
                    "feasible_allocation_dimension": routing.feasible_allocation_dimension,
                    "selected_bound_fixed_count": routing.selected_bound_fixed_count,
                    "neutral_soft_coefficient_distance": routing.neutral_soft_coefficient_distance,
                    "routing_status": routing.status.value,
                    "same_state_repair_count": 0,
                    "candidate_replay_count": 0,
                }
            )
        if typed_zero_positive and not strength_preserving:
            progress_payload.update(
                {
                    "actual_definition": "RESIDUAL_OVERLAY_OBJECTIVE_PROGRESS",
                    "actual_is_weight_only_progress": False,
                }
            )
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
        if strength_preserving:
            routing_payload.update(
                {
                    "method_id": STRENGTH_PRESERVING_METHOD_ID,
                    "instruction_id": STRENGTH_PRESERVING_INSTRUCTION_ID,
                    "amendment_id": STRENGTH_PRESERVING_AMENDMENT_ID,
                    "physical_weight_only_current_mean_target_new_nll": (
                        signed.mean_objective_value
                    ),
                    "physical_weight_only_candidate_mean_target_new_nll": float(
                        candidate_weight_objective.loss.detach().cpu()
                    ),
                    "physical_weight_only_objective_receipt": (
                        _routing_objective_raw_free_payload(
                            candidate_weight_objective,
                            role="AUTHORITATIVE_PHYSICAL_BF16_WONLY_CANDIDATE",
                            action_frozen_before_evaluation=False,
                        )
                    ),
                    "router_added_compute": {
                        "logical_forward_groups": 0,
                        "model_forward_count": 0,
                        "backward_count": 0,
                        "target_backward_count": 0,
                        "processed_token_count": 0,
                    },
                    "structural_p_layer_cost": {
                        "ordered_layers": list(COMMON_COLD_LAYER_ORDER),
                        "c_p": [
                            float(item) for item in built.pretrained_self_risk
                        ],
                        "definition": "tr(B_l*C0_l*B_l^T)",
                        "pinned_wikipedia_second_moment_only": True,
                        "covariance_receipt_inventory_sha256": canonical_hash(
                            [
                                asdict(item.covariance_receipt)
                                for item in field.layers
                            ]
                        ),
                        "statistic_recompute_count": 0,
                        "technical_feasible_set_influence_count": 0,
                        "hard_threshold_influence_count": 0,
                    },
                    "progress_floor": 0.0,
                    "kappa": 0.0,
                    "requested_progress_legacy": None,
                    "hard_h_p_budget_influence_count": 0,
                    "functional_veto_count": 0,
                    "retry_count": 0,
                }
            )
        if typed_zero_positive:
            routing_payload.update(
                {
                    "target_hold": {
                        "active": arm.target_hold,
                        "target_clock_advance_count": 0 if arm.target_hold else 1,
                        "weight_clock_advance_count": 1,
                        "proposed_target_trial_sha256": tensor_sha256(
                            proposed_target_trial
                        ),
                        "applied_target_trial_sha256": tensor_sha256(
                            target_trial
                        ),
                        "target_state_invariant": (
                            tensor_sha256(target_trial) == entry_target_sha256
                            if arm.target_hold
                            else False
                        ),
                        "first_allowed_main_hold_difference": (
                            "MAIN_TARGET_ADVANCEMENT_AFTER_K1"
                        ),
                    },
                    "target_assisted_six_context_objective": (
                        _routing_objective_raw_free_payload(
                            candidate_objective,
                            role="TARGET_ASSISTED_RESIDUAL_OVERLAY_OBJECTIVE",
                            action_frozen_before_evaluation=False,
                        )
                        if record_six_context_objectives
                        else None
                    ),
                    "weight_only_six_context_objective": (
                        "DEFERRED_UNTIL_ACTION_FREEZE"
                        if record_six_context_objectives
                        else None
                    ),
                }
            )
            if slope_comparison is not None:
                routing_payload["overlay_vs_nohook_signed_slopes"] = (
                    slope_comparison
                )
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
        if not target_only_zero_write:
            for layer in field.layers:
                accepted_by_layer[layer.layer].append(
                    AcceptedLayerContribution.from_field(
                        layer,
                        increment[layer.weight_name],
                        history_action=layer.history_action,
                    )
                )
            authoritative_weight_write_count += 1
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
        if typed_zero_positive:
            audit_state = {
                "arm": arm.value,
                "step_index": step_index,
                "field": field,
                "increment": dict(increment),
                "source_factors": _factor_map(current_factors),
                "source_target": current_target.clone(),
                "source_terminal": current_terminal.clone(),
                "online_candidate_terminal": candidate_terminal.clone(),
                "intended_delta": (
                    target_trial - current_target
                ).contiguous(),
            }
            if entry_audit_state is None:
                entry_audit_state = dict(audit_state)
            boundary_audit_state = dict(audit_state)
        current_factors = _factor_map(candidate_factors)
        current_target = target_trial.clone()
        current_terminal = candidate_terminal.clone()
        current_residual = candidate_residual.clone()
        if not strength_preserving:
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
    trajectory_complete = zero_positive_step is None
    if trajectory_complete:
        if (
            len(snapshots) != 8
            or clock.tau != Fraction(1, 1)
            or clock.field_count != 8
        ):
            raise ODEBFStateError("common cold terminal grid invariant differs")
    elif (
        zero_positive_payload is None
        or zero_positive_field_receipt is None
        or len(snapshots) != clock.step_index
        or clock.field_count != len(snapshots) + 1
    ):
        raise ODEBFStateError("common cold partial grid invariant differs")
    target_hold_clock: Mapping[str, Any] | None = None
    if arm.target_hold:
        target_hashes = [tensor_sha256(item.target_state) for item in snapshots]
        if any(item != entry_target_sha256 for item in target_hashes):
            raise ODEBFStateError("BG-Soft target-hold target state changed")
        if trajectory_complete:
            from .bg_soft_diagnostics import target_hold_clock_receipt

            target_hold_clock = target_hold_clock_receipt(
                bootstrap_target,
                [item.target_state for item in snapshots],
                weight_step_count=len(snapshots),
                h=float(FIXED_E8_H),
                tau_w=float(clock.tau),
                target_advance_count=0,
                scientific_retry_count=0,
            )
        else:
            target_hold_clock = {
                "schema": (
                    "ode-edit-s05-bg-soft-target-hold-partial-clock/v1"
                ),
                "status": "TARGET_HOLD_PREFIX_PRESERVED",
                "bootstrap_target_sha256": entry_target_sha256,
                "target_sha256_by_weight_step": target_hashes,
                "target_hash_invariant": True,
                "weight_step_count": len(snapshots),
                "weight_h": float(FIXED_E8_H),
                "tau_W": float(clock.tau),
                "target_advance_count": 0,
                "scientific_retry_count": 0,
                "termination_label": "ZERO_POSITIVE_DIRECTION",
            }
            target_hold_clock["identity_sha256"] = canonical_hash(
                target_hold_clock
            )
    if ledger.completed_correction_cycles == 0:
        ledger.finish_cycle(0)
    single_write_audit_receipts: dict[str, str] = {}
    if typed_zero_positive and entry_audit_state is not None:
        entry_payload = _single_write_audit_point(
            model,
            tokenizer,
            requests,
            hparams=hparams,
            state={**entry_audit_state, "audit_role": "ENTRY_FIELD_K0"},
            touched=touched,
            history=history,
            schedule=schedule,
        )
        single_write_audit_receipts["entry"] = recorder.write_once(
            recorder.root / "single-write-audit-entry.json",
            entry_payload,
        )
    if typed_zero_positive and boundary_audit_state is not None:
        boundary_payload = _single_write_audit_point(
            model,
            tokenizer,
            requests,
            hparams=hparams,
            state={
                **boundary_audit_state,
                "audit_role": (
                    "K7_SOURCE_FIELD_FOR_TAU1_COMPLETION"
                    if trajectory_complete
                    else "LAST_ACCEPTED_TRANSITION_SOURCE_BEFORE_COLLAPSE"
                ),
            },
            touched=touched,
            history=history,
            schedule=schedule,
        )
        single_write_audit_receipts["boundary_or_final"] = (
            recorder.write_once(
                recorder.root / "single-write-audit-boundary.json",
                boundary_payload,
            )
        )
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
        trajectory_complete=trajectory_complete,
        functional_p_policy=FunctionalPDecisionPolicy.OBSERVATION_ONLY,
        preservation_policy=PreservationConstraintPolicy.OBSERVATION_ONLY,
    )
    omega_changed = _omega_state(accepted_by_layer) != entry_omega
    if (
        _parameter_contract_sha256(touched) != before_model
        or history.snapshot().digest != before_history
        or schedule.state_digest != before_sampler
        or legacy._rng_identity() != before_rng
        or history.version != 0
        or omega_changed != bool(authoritative_weight_write_count)
    ):
        raise ODEBFStateError("common cold rollout state/purity differs")
    status = (
        "COMMON_COLD_FIXED_E8_TAU_COMPLETE"
        if trajectory_complete
        else "ZERO_POSITIVE_DIRECTION"
    )
    clock_payload = (
        clock.terminal_receipt()
        if trajectory_complete
        else _partial_fixed_e8_clock_receipt(clock)
    )
    operation_accounting = {
        "actual_logical_qp_certificate_count": total_qp,
        "actual_optimizer_backend_invocation_count": total_backend,
        "actual_solver_backend_fallback_count": total_solver_fallback,
        "actual_certified_stage1_selection_fallback_count": total_stage1_fallback,
        "functional_basis_endpoint_count": 6 * clock.field_count,
        "candidate_count": len(snapshots),
        "field_count": clock.field_count,
    }
    if typed_zero_positive:
        d1_accounting = _bg_soft_d1_operation_accounting(
            routing_qp=total_qp,
            routing_backend=total_backend,
            nohook_qp=total_d1_nohook_qp,
            nohook_backend=total_d1_nohook_backend,
            nohook_fallback=total_d1_nohook_fallback,
            ledger=ledger,
        )
    if typed_zero_positive:
        operation_accounting.update(
            {
                "actual_logical_qp_certificate_count": d1_accounting[
                    "actual_all_logical_qp_certificate_count"
                ],
                "actual_optimizer_backend_invocation_count": d1_accounting[
                    "actual_all_optimizer_backend_invocation_count"
                ],
                "actual_solver_backend_fallback_count": (
                    total_solver_fallback + total_d1_nohook_fallback
                ),
                "routing_only_logical_qp_certificate_count": total_qp,
                "routing_only_optimizer_backend_invocation_count": (
                    total_backend
                ),
                "routing_only_solver_backend_fallback_count": (
                    total_solver_fallback
                ),
                "d1_observation_accounting": d1_accounting,
            }
        )
    rollout_payload = {
        "method": common_cold_source_contract(),
        "variant": arm.value,
        "scale": metric.raw_free_payload(),
        "status": status,
        "accepted_t": fraction_payload(clock.tau),
        "k_acc": len(snapshots),
        "n_trial": len(snapshots),
        "n_reject": 0,
        "field_build_count": clock.field_count,
        "operation_accounting": operation_accounting,
        "clock": clock_payload,
        "entry_success": entry_eval.batch_success.raw_free_payload(),
        "online_first_hit": None if first_hit.first_online is None else {
            "accepted_index": first_hit.first_online.accepted_index,
            "tau": fraction_payload(first_hit.first_online.tau),
            "snapshot_sha256": first_hit.first_online.snapshot_sha256,
        },
        "first_hit_observation_only": True,
        "joint_zero_write_count": len(snapshots) - authoritative_weight_write_count,
        "authoritative_weight_write_count": authoritative_weight_write_count,
        "target_only_transition_count": (
            len(snapshots) - authoritative_weight_write_count
        ),
        "scientific_retry_count": 0,
        "scientific_rejection_count": 0,
        "native_or_direct_z_cold_access_count": 0,
        "receipt_links": recorder.links(),
        "compute": ledger.raw_free_payload(),
    }
    if typed_zero_positive:
        rollout_payload.update(
            {
                "termination_label": status,
                "entry_target_assisted_six_context_objective": (
                    entry_target_assisted
                    if record_six_context_objectives
                    else None
                ),
                "target_hold": {
                    "active": arm.target_hold,
                    "bootstrap_target_sha256": entry_target_sha256,
                    "terminal_target_sha256": tensor_sha256(current_target),
                    "target_hash_invariant": (
                        tensor_sha256(current_target) == entry_target_sha256
                        if arm.target_hold
                        else False
                    ),
                    "target_clock_advance_count": (
                        0 if arm.target_hold else len(snapshots)
                    ),
                    "weight_clock_advance_count": len(snapshots),
                    "weight_tau": fraction_payload(clock.tau),
                    "clock_receipt": target_hold_clock,
                },
                "zero_positive_direction": zero_positive_payload,
                "single_write_audit_receipts": single_write_audit_receipts,
                "single_write_audit_unavailable_reason": (
                    None
                    if single_write_audit_receipts
                    else "NO_SUCCESSFULLY_SELECTED_WRITE_FIELD"
                ),
                "valid_prefix_preserved": not trajectory_complete,
            }
        )
    if observability_contract is not None:
        rollout_payload.update(
            {
                "observability_contract": dict(observability_contract),
                "observability_total_decision_influence_count": 0,
            }
        )
    rollout_sha = canonical_hash(rollout_payload)
    recorder.terminal({"rollout_summary": rollout_payload, "rollout_sha256": rollout_sha})
    if initial_contract is None:
        raise ODEBFStateError("common cold initial contract is absent")
    return legacy.FixedE8Rollout(
        arm,
        status,
        status,
        SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1,
        clock.tau,
        len(snapshots),
        len(snapshots),
        0,
        clock.field_count,
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


def _heldout_additive_lookup_geometry(
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    cases: Sequence[Any],
    *,
    fact_token_strategy: str,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], dict[str, Any]]:
    """Resolve held-out rewrite/paraphrase lookup rows without serializing text."""

    from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main

    if len(requests) != BATCH_SIZE or len(cases) != BATCH_SIZE:
        raise ODEBFContractError("BG-Soft heldout lookup B10 differs")
    evaluation_padding_side = tokenizer.padding_side
    if evaluation_padding_side not in ("left", "right"):
        raise ODEBFContractError(
            "BG-Soft heldout evaluator padding policy differs"
        )
    positions_by_request: list[tuple[int, ...]] = []
    patched_rows: list[int] = []
    geometry_rows: list[dict[str, Any]] = []
    for request_index, (request, case) in enumerate(
        zip(requests, cases, strict=True)
    ):
        if str(request["request_sha256"]) != case.request_sha256:
            raise ODEBFContractError(
                "BG-Soft heldout lookup request order differs"
            )
        subject = str(request["subject"])
        prefixes = (case.rewrite_prompt,) + case.paraphrase_prompts
        all_prefixes = prefixes + case.neighborhood_prompts
        templates: list[str] = []
        raw_positions: list[int] = []
        prefix_lengths: list[int] = []
        for prefix in prefixes:
            if prefix.count(subject) != 1:
                raise ODEBFContractError(
                    "BG-Soft heldout subject surface is ambiguous"
                )
            template = prefix.replace(subject, "{}", 1)
            templates.append(template)
            raw_positions.append(
                int(
                    alpha_main.find_fact_lookup_idx(
                        template,
                        subject,
                        tokenizer,
                        fact_token_strategy,
                        verbose=False,
                    )
                )
            )
            prefix_lengths.append(len(tokenizer(prefix)["input_ids"]))
        rows = [
            f"{prefix} {suffix}"
            for prefix in all_prefixes
            for suffix in (case.target_new, case.target_true)
        ]
        encoded = tokenizer(rows, padding=True, return_tensors="pt")
        attention = encoded["attention_mask"]
        row_positions: list[int] = []
        for row_index in range(2 * len(prefixes)):
            prefix_index = row_index // 2
            pad_count = int(
                attention.shape[1] - attention[row_index].sum()
            )
            left_pad = (
                pad_count if evaluation_padding_side == "left" else 0
            )
            raw = raw_positions[prefix_index]
            position = (
                left_pad + prefix_lengths[prefix_index] - 1
                if raw == -1
                else left_pad + raw
            )
            if position < 0 or position >= int(attention.shape[1]):
                raise ODEBFContractError(
                    "BG-Soft heldout lookup position is out of range"
                )
            row_positions.append(position)
        row_positions.extend(0 for _ in range(len(rows) - len(row_positions)))
        positions_by_request.append(tuple(row_positions))
        patched_rows.append(2 * len(prefixes))
        geometry_rows.append(
            {
                "request_index": request_index,
                "request_sha256": case.request_sha256,
                "patched_prefix_count": len(prefixes),
                "patched_row_count": 2 * len(prefixes),
                "locality_nohook_row_count": 2
                * len(case.neighborhood_prompts),
                "row_position_sha256": canonical_hash(row_positions),
                "template_sha256": canonical_hash(
                    [
                        hashlib.sha256(item.encode("utf-8")).hexdigest()
                        for item in templates
                    ]
                ),
            }
        )
    if tokenizer.padding_side != evaluation_padding_side:
        raise ODEBFStateError(
            "BG-Soft heldout lookup geometry mutated tokenizer padding"
        )
    payload = {
        "schema": "ode-edit-s05-bg-soft-heldout-additive-lookup/v1",
        "request_count": BATCH_SIZE,
        "patched_rows_per_request": patched_rows,
        "rows": geometry_rows,
        "padding_side_during_geometry": evaluation_padding_side,
        "padding_side_matches_evaluator": True,
        "padding_side_unchanged": True,
        "absolute_z_replacement_count": 0,
        "heldout_controller_decision_influence_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return tuple(positions_by_request), tuple(patched_rows), payload


def _postfreeze_bg_soft_prefix_panel(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    requests: Sequence[Mapping[str, Any]],
    contexts: Sequence[Sequence[str]],
    hparams: Any,
    target_layer_name: str,
    dataset_path: Path,
    rollouts: Mapping[str, legacy.FixedE8Rollout],
    raw_root: Path,
    write_once: Any,
    touched: Mapping[str, torch.nn.Parameter],
    request_order_sha256: str,
    action_freeze_sha256: str,
    frozen_reference: Mapping[str, Any],
    receipt_schema_prefix: str = "ode-edit-s05-bg-soft-missing-cell",
    instruction_id: str = (
        "ODEEDIT-S05-ODE-BF-BG-SOFT-MISSING-CELL-P1R12-V1"
    ),
    amendment_id: str = (
        "ODEEDIT-S05-ODE-BF-BG-SOFT-MISSING-CELL-P1R12-V1-A1"
    ),
    output_token: str = "bg-soft",
    observability_contract: Mapping[str, Any] | None = None,
    bootstrap_targets: Mapping[str, torch.Tensor] | None = None,
    controller_lookup_positions: Sequence[int] | None = None,
    progress_actual_semantics: str = (
        "RESIDUAL_OVERLAY_OBJECTIVE_NOT_WEIGHT_ONLY"
    ),
) -> dict[str, Any]:
    """Evaluate MAIN and TARGET-HOLD prefixes only after action freeze."""

    if not rollouts:
        raise ODEBFContractError("BG-Soft post-freeze rollouts are absent")
    if bootstrap_targets is not None and controller_lookup_positions is None:
        raise ODEBFContractError(
            "universal D2 controller lookup positions are absent"
        )

    freeze_payload = {
        "schema": f"{receipt_schema_prefix}-action-freeze/v1",
        "instruction_id": instruction_id,
        "amendment_id": amendment_id,
        "upstream_action_freeze_sha256": action_freeze_sha256,
        "action_frozen_before_open": True,
        "request_order_sha256": request_order_sha256,
        "rollouts": {
            variant: {
                "rollout_sha256": rollout.rollout_sha256,
                "status": rollout.status,
                "accepted_snapshot_sha256": [
                    item.snapshot_sha256 for item in rollout.snapshots
                ],
            }
            for variant, rollout in sorted(rollouts.items())
        },
        "heldout_controller_access_count": 0,
        "model_generate_call_count": 0,
        "w0_native_bg_neutral_rerun_count": 0,
    }
    if observability_contract is not None:
        freeze_payload.update(
            {
                "observability_contract": dict(observability_contract),
                "d1_d2_d3_observation_only": True,
                "observability_total_decision_influence_count": 0,
            }
        )
    freeze_sha256 = write_once(
        raw_root / "stepwise" / f"{output_token}-action-freeze.json",
        freeze_payload,
    )
    ledger = ComputeLedger()
    receipt_hashes: dict[str, dict[str, str]] = {}
    first_rollout = (
        next(iter(rollouts.values()), None)
        if bootstrap_targets is not None
        else next((item for item in rollouts.values() if item.snapshots), None)
    )
    if first_rollout is not None:
        first = (
            first_rollout.snapshots[0]
            if first_rollout.snapshots
            else None
        )
        loader_snapshot_sha256 = (
            first.snapshot_sha256
            if first is not None
            else canonical_hash(
                {
                    "variant": first_rollout.variant_label,
                    "entry": 0,
                    "action_freeze_sha256": action_freeze_sha256,
                }
            )
        )
        loader_freeze = StepwiseActionFreeze(
            variant=first_rollout.variant_label,
            request_order_sha256=request_order_sha256,
            rollout_sha256=first_rollout.rollout_sha256,
            snapshot_sha256=loader_snapshot_sha256,
            snapshot_index=(0 if first is None else first.accepted_index),
            accepted_snapshot_count=(
                len(first_rollout.snapshots)
                + (1 if bootstrap_targets is not None else 0)
            ),
            rejected_retry_count=0,
            trajectory_status=first_rollout.status,
        )
        cases = load_counterfact_cases_after_freeze(
            dataset_path, requests, loader_freeze
        )
        (
            heldout_lookup_positions,
            heldout_patched_rows,
            heldout_lookup_receipt,
        ) = _heldout_additive_lookup_geometry(
            tokenizer,
            requests,
            cases,
            fact_token_strategy=hparams.fact_token,
        )
        heldout_lookup_sha256 = write_once(
            raw_root / "stepwise" / "heldout-additive-lookup.json",
            heldout_lookup_receipt,
        )
        counter = ModelForwardCounter(model, ledger)
        try:
            if bootstrap_targets is not None:
                for variant, rollout in sorted(rollouts.items()):
                    if variant not in bootstrap_targets:
                        raise ODEBFContractError(
                            "universal D2 entry target is absent"
                        )
                    entry_target = bootstrap_targets[variant]
                    entry_snapshot_sha256 = canonical_hash(
                        {
                            "variant": variant,
                            "entry_index": 0,
                            "bootstrap_target_sha256": tensor_sha256(
                                entry_target
                            ),
                            "rollout_sha256": rollout.rollout_sha256,
                            "action_freeze_sha256": action_freeze_sha256,
                        }
                    )
                    freeze = StepwiseActionFreeze(
                        variant=variant,
                        request_order_sha256=request_order_sha256,
                        rollout_sha256=rollout.rollout_sha256,
                        snapshot_sha256=entry_snapshot_sha256,
                        snapshot_index=0,
                        accepted_snapshot_count=len(rollout.snapshots) + 1,
                        rejected_retry_count=0,
                        trajectory_status=rollout.status,
                    )
                    observed, compute = _evaluate_stepwise_state(
                        model,
                        tokenizer,
                        cases,
                        alias=alias,
                        freeze=freeze,
                        ledger=ledger,
                        touched=touched,
                    )
                    before_oracle = _parameter_contract_sha256(touched)
                    entry_terminal = capture_cold_z_base(
                        model, tokenizer, requests, hparams
                    )
                    entry_residual = common_terminal_residual_input(
                        entry_target,
                        entry_terminal,
                        request_order_sha256,
                    )
                    from .bg_soft_diagnostics import (
                        HeldoutRequestResidualActivationOverlay,
                    )

                    z_oracle_overlay = (
                        HeldoutRequestResidualActivationOverlay(
                            model,
                            target_layer_name,
                            entry_residual.residual,
                            heldout_lookup_positions,
                            heldout_patched_rows,
                        )
                    )
                    oracle_started = time.perf_counter()
                    with z_oracle_overlay:
                        z_oracle = evaluate_counterfact_stepwise_primary(
                            model,
                            tokenizer,
                            cases,
                            model_alias=alias,
                            freeze=freeze,
                        )
                    oracle_wall = time.perf_counter() - oracle_started
                    z_oracle_overlay_payload = (
                        z_oracle_overlay.raw_free_payload()
                    )
                    after_oracle = _parameter_contract_sha256(touched)
                    if before_oracle != after_oracle:
                        raise ODEBFStateError(
                            "universal D2 entry z-oracle mutated state"
                        )
                    left = observed.primary
                    right = z_oracle.primary
                    if (
                        left.request_order_sha256
                        != right.request_order_sha256
                        or left.evaluation_case_identity_sha256
                        != right.evaluation_case_identity_sha256
                        or left.target_span_sha256
                        != right.target_span_sha256
                        or left.evaluator_source_sha256
                        != right.evaluator_source_sha256
                        or left.aggregator_source_sha256
                        != right.aggregator_source_sha256
                        or right.generation_call_count != 0
                        or left.locality != right.locality
                    ):
                        raise ODEBFContractError(
                            "universal D2 entry evaluator parity differs"
                        )
                    ledger.increment(
                        "evaluator_forward", right.model_forward_count
                    )
                    ledger.increment(
                        "evaluator_tokens", right.processed_token_count
                    )
                    ledger.add_time(
                        "postfreeze_z_oracle", wall_seconds=oracle_wall
                    )
                    target_assisted, _ = _objective_with_residual(
                        model,
                        tokenizer,
                        requests,
                        factors={},
                        contexts=contexts,
                        target_layer_name=target_layer_name,
                        lookup_positions=controller_lookup_positions,
                        residual=entry_residual.residual,
                    )
                    weight_only = evaluate_routing_objective(
                        model,
                        tokenizer,
                        requests,
                        objective=RoutingObjective.TARGET_NEW_NLL,
                        contexts=contexts,
                    )
                    target_assisted_payload = (
                        _routing_objective_raw_free_payload(
                            target_assisted,
                            role=(
                                "TARGET_ASSISTED_RESIDUAL_OVERLAY_OBJECTIVE"
                            ),
                            action_frozen_before_evaluation=True,
                        )
                    )
                    weight_only_payload = (
                        _routing_objective_raw_free_payload(
                            weight_only,
                            role="WEIGHT_ONLY_NO_RESIDUAL_HOOK_OBJECTIVE",
                            action_frozen_before_evaluation=True,
                        )
                    )
                    ledger.increment(
                        "evaluator_forward",
                        target_assisted.model_forward_count
                        + weight_only.model_forward_count,
                    )
                    ledger.increment(
                        "evaluator_tokens",
                        target_assisted.processed_token_count
                        + weight_only.processed_token_count,
                    )
                    entry_receipt = {
                        "schema": (
                            f"{receipt_schema_prefix}-step-receipt/v1"
                        ),
                        "variant": variant,
                        "accepted_index": 0,
                        "tau": fraction_payload(Fraction(0, 1)),
                        "delta_tau": fraction_payload(Fraction(0, 1)),
                        "snapshot_sha256": entry_snapshot_sha256,
                        "freeze_sha256": freeze.identity(),
                        "primary": observed.raw_free_payload(),
                        "w_only_primary": observed.raw_free_payload(),
                        "z_oracle_primary": z_oracle.raw_free_payload(),
                        "z_oracle_additive_overlay": (
                            z_oracle_overlay_payload
                        ),
                        "z_oracle_residual": (
                            entry_residual.raw_free_payload()
                        ),
                        "heldout_additive_lookup_sha256": (
                            heldout_lookup_sha256
                        ),
                        "z_oracle_semantics": (
                            "REQUEST_SPECIFIC_ADDITIVE_TERMINAL_RESIDUAL_"
                            "AT_HELDOUT_SUBJECT_LOOKUP"
                        ),
                        "absolute_z_replacement_count": 0,
                        "z_oracle_locality_rows_unhooked_exact": True,
                        "w_only_z_oracle_prompt_span_order_exact": True,
                        "z_oracle_compute": {
                            "wall_seconds": oracle_wall,
                            "model_forward_count": (
                                right.model_forward_count + 1
                            ),
                            "processed_token_count": (
                                right.processed_token_count
                            ),
                            "terminal_recapture_forward_count": 1,
                        },
                        "target_assisted_six_context_objective": (
                            target_assisted_payload
                        ),
                        "weight_only_six_context_objective": (
                            weight_only_payload
                        ),
                        "progress_actual_semantics": progress_actual_semantics,
                        "routing": None,
                        "capacity": None,
                        "compute": compute,
                        "action_frozen_before_open": True,
                        "controller_heldout_access_count": 0,
                        "entry_snapshot": True,
                    }
                    if observability_contract is not None:
                        entry_receipt["observability_contract"] = dict(
                            observability_contract
                        )
                    receipt_hashes.setdefault(variant, {})["0"] = (
                        write_once(
                            raw_root
                            / "stepwise"
                            / variant
                            / "accepted-0000.json",
                            entry_receipt,
                        )
                    )
            for variant, rollout in sorted(rollouts.items()):
                variant_hashes: dict[str, str] = dict(
                    receipt_hashes.get(variant, {})
                )
                for snapshot in rollout.snapshots:
                    freeze = StepwiseActionFreeze(
                        variant=variant,
                        request_order_sha256=request_order_sha256,
                        rollout_sha256=rollout.rollout_sha256,
                        snapshot_sha256=snapshot.snapshot_sha256,
                        snapshot_index=snapshot.accepted_index,
                        accepted_snapshot_count=(
                            len(rollout.snapshots)
                            + (1 if bootstrap_targets is not None else 0)
                        ),
                        rejected_retry_count=0,
                        trajectory_status=rollout.status,
                    )
                    observed, compute = _evaluate_stepwise_state(
                        model,
                        tokenizer,
                        cases,
                        alias=alias,
                        freeze=freeze,
                        ledger=ledger,
                        touched=touched,
                        factors=snapshot.factors,
                    )
                    before_oracle = _parameter_contract_sha256(touched)
                    with _virtual_context(model, snapshot.factors):
                        snapshot_terminal = capture_cold_z_base(
                            model, tokenizer, requests, hparams
                        )
                    snapshot_residual = common_terminal_residual_input(
                        snapshot.target_state,
                        snapshot_terminal,
                        request_order_sha256,
                    )
                    from .bg_soft_diagnostics import (
                        HeldoutRequestResidualActivationOverlay,
                    )

                    z_oracle_overlay = (
                        HeldoutRequestResidualActivationOverlay(
                            model,
                            target_layer_name,
                            snapshot_residual.residual,
                            heldout_lookup_positions,
                            heldout_patched_rows,
                        )
                    )
                    oracle_started = time.perf_counter()
                    with _virtual_context(model, snapshot.factors):
                        with z_oracle_overlay:
                            z_oracle = evaluate_counterfact_stepwise_primary(
                                model,
                                tokenizer,
                                cases,
                                model_alias=alias,
                                freeze=freeze,
                            )
                    oracle_wall = time.perf_counter() - oracle_started
                    z_oracle_overlay_payload = (
                        z_oracle_overlay.raw_free_payload()
                    )
                    after_oracle = _parameter_contract_sha256(touched)
                    if before_oracle != after_oracle:
                        raise ODEBFStateError(
                            "BG-Soft z-oracle evaluator mutated state"
                        )
                    left = observed.primary
                    right = z_oracle.primary
                    if (
                        left.request_order_sha256
                        != right.request_order_sha256
                        or left.evaluation_case_identity_sha256
                        != right.evaluation_case_identity_sha256
                        or left.target_span_sha256
                        != right.target_span_sha256
                        or left.evaluator_source_sha256
                        != right.evaluator_source_sha256
                        or left.aggregator_source_sha256
                        != right.aggregator_source_sha256
                        or right.generation_call_count != 0
                        or left.locality != right.locality
                    ):
                        raise ODEBFContractError(
                            "BG-Soft W-only/z-oracle evaluator parity differs"
                        )
                    ledger.increment(
                        "evaluator_forward", right.model_forward_count
                    )
                    ledger.increment(
                        "evaluator_tokens", right.processed_token_count
                    )
                    ledger.add_time(
                        "postfreeze_z_oracle",
                        wall_seconds=oracle_wall,
                    )
                    with _virtual_context(model, snapshot.factors):
                        weight_only = evaluate_routing_objective(
                            model,
                            tokenizer,
                            requests,
                            objective=RoutingObjective.TARGET_NEW_NLL,
                            contexts=contexts,
                        )
                    weight_only_payload = _routing_objective_raw_free_payload(
                        weight_only,
                        role="WEIGHT_ONLY_NO_RESIDUAL_HOOK_OBJECTIVE",
                        action_frozen_before_evaluation=True,
                    )
                    ledger.increment(
                        "evaluator_forward", weight_only.model_forward_count
                    )
                    ledger.increment(
                        "evaluator_tokens", weight_only.processed_token_count
                    )
                    relative = (
                        Path("stepwise")
                        / variant
                        / f"accepted-{snapshot.accepted_index:04d}.json"
                    )
                    variant_hashes[str(snapshot.accepted_index)] = write_once(
                        raw_root / relative,
                        {
                        "schema": (
                            f"{receipt_schema_prefix}-step-receipt/v1"
                        ),
                        "variant": variant,
                        "accepted_index": snapshot.accepted_index,
                        "tau": fraction_payload(snapshot.tau),
                        "delta_tau": fraction_payload(snapshot.delta_tau),
                        "snapshot_sha256": snapshot.snapshot_sha256,
                        "freeze_sha256": freeze.identity(),
                        "primary": observed.raw_free_payload(),
                        "w_only_primary": observed.raw_free_payload(),
                        "z_oracle_primary": z_oracle.raw_free_payload(),
                        "z_oracle_additive_overlay": (
                            z_oracle_overlay_payload
                        ),
                        "z_oracle_residual": (
                            snapshot_residual.raw_free_payload()
                        ),
                        "heldout_additive_lookup_sha256": (
                            heldout_lookup_sha256
                        ),
                        "z_oracle_semantics": (
                            "REQUEST_SPECIFIC_ADDITIVE_TERMINAL_RESIDUAL_"
                            "AT_HELDOUT_SUBJECT_LOOKUP"
                        ),
                        "absolute_z_replacement_count": 0,
                        "z_oracle_locality_rows_unhooked_exact": True,
                        "w_only_z_oracle_prompt_span_order_exact": True,
                        "z_oracle_compute": {
                            "wall_seconds": oracle_wall,
                            "model_forward_count": (
                                right.model_forward_count + 1
                            ),
                            "processed_token_count": (
                                right.processed_token_count
                            ),
                            "terminal_recapture_forward_count": 1,
                        },
                        "target_assisted_six_context_objective": (
                            snapshot.routing_payload[
                                "target_assisted_six_context_objective"
                            ]
                        ),
                        "weight_only_six_context_objective": weight_only_payload,
                        "progress_actual_semantics": progress_actual_semantics,
                        "routing": snapshot.routing_payload,
                        "capacity": snapshot.capacity_payload,
                        "compute": compute,
                        "action_frozen_before_open": True,
                        "controller_heldout_access_count": 0,
                        **(
                            {
                                "observability_contract": dict(
                                    observability_contract
                                )
                            }
                            if observability_contract is not None
                            else {}
                        ),
                        },
                    )
                receipt_hashes[variant] = dict(
                    sorted(variant_hashes.items())
                )
        finally:
            counter.close()
    from .p1_runtime import _observed_memory

    _observed_memory(ledger)
    panel = {
        "schema": f"{receipt_schema_prefix}-stepwise-panel/v1",
        "action_freeze_sha256": freeze_sha256,
        "request_order_sha256": request_order_sha256,
        "rollout_sha256": {
            key: value.rollout_sha256
            for key, value in sorted(rollouts.items())
        },
        "trajectory_status": {
            key: value.status for key, value in sorted(rollouts.items())
        },
        "accepted_snapshot_count": {
            key: len(value.snapshots)
            for key, value in sorted(rollouts.items())
        },
        "receipt_sha256": dict(sorted(receipt_hashes.items())),
        "frozen_r10_reference": dict(frozen_reference),
        "w0_native_bg_neutral_rerun_count": 0,
        "heldout_controller_access_count": 0,
        "heldout_additive_lookup_sha256": (
            heldout_lookup_sha256
            if first_rollout is not None
            else None
        ),
        "generation_call_count": 0,
        "evaluation_compute": ledger.raw_free_payload(),
    }
    if observability_contract is not None:
        panel.update(
            {
                "instruction_id": instruction_id,
                "amendment_id": amendment_id,
                "observability_contract": dict(observability_contract),
                "d1_d2_d3_observation_only": True,
                "observability_total_decision_influence_count": 0,
                "evaluated_snapshot_count_including_entry_k0": {
                    key: len(value.snapshots) + 1
                    for key, value in sorted(rollouts.items())
                },
            }
        )
    panel["panel_sha256"] = canonical_hash(panel)
    return panel


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
    bg_soft_missing_cell_mode: bool = False,
    bg_soft_reference_lock: Mapping[str, Any] | None = None,
    bg_soft_reference_lock_sha256: str | None = None,
    universal_observability_cell: str | None = None,
    universal_observability_lock: Mapping[str, Any] | None = None,
    universal_observability_lock_sha256: str | None = None,
    universal_frozen_r12_reference: Mapping[str, Any] | None = None,
    strength_preserving_cell: str | None = None,
    strength_preserving_lock: Mapping[str, Any] | None = None,
    strength_preserving_lock_sha256: str | None = None,
) -> dict[str, Any]:
    from .p1_runtime import ArmRuntimeState, _entry_parameter_snapshot_sha256, _evaluate_native_rewrite, _observed_memory

    runtime_wall_started = time.perf_counter()
    del mutation_lock
    request_order = ordered_request_digest_v1([str(item["request_sha256"]) for item in requests])
    if len(requests) != BATCH_SIZE or request_order != stream["batch_ordered_request_digest_v1"][0]:
        raise ODEBFContractError("common cold request/seal order differs")
    universal_observability_mode = universal_observability_cell is not None
    strength_preserving_mode = strength_preserving_cell is not None
    if sum(
        (
            int(bg_soft_missing_cell_mode),
            int(universal_observability_mode),
            int(strength_preserving_mode),
        )
    ) > 1:
        raise ODEBFContractError(
            "common cold special modes are mutually exclusive"
        )
    if strength_preserving_mode:
        from .p1_strength_preserving_router_panel import (
            STRENGTH_PRESERVING_SCHEMA_NAMESPACE,
            validate_strength_preserving_lock,
        )

        if (
            strength_preserving_lock is None
            or strength_preserving_lock_sha256 is None
        ):
            raise ODEBFContractError("strength-preserving lock is absent")
        validate_strength_preserving_lock(strength_preserving_lock)
        strength_live_arm = CommonColdArm(str(strength_preserving_cell))
        if strength_live_arm.target_hold:
            raise ODEBFContractError("strength-preserving Hold arm is forbidden")
        strength_live_arms = (strength_live_arm,)
        expected_bg_initial = None
        frozen_bg_reference = None
        runtime_instruction_id = STRENGTH_PRESERVING_INSTRUCTION_ID
        runtime_amendment_id = STRENGTH_PRESERVING_AMENDMENT_ID
        runtime_schema_namespace = STRENGTH_PRESERVING_SCHEMA_NAMESPACE
        observability_contract = None
        universal_live_arms = ()
    elif bg_soft_missing_cell_mode:
        from .p1_bg_soft_missing_cell_panel import (
            BG_SOFT_AMENDMENT_ID,
            BG_SOFT_INSTRUCTION_ID,
            BG_SOFT_SCHEMA_NAMESPACE,
            bg_soft_frozen_reference,
            bg_soft_initial_contract,
        )

        if (
            bg_soft_reference_lock is None
            or bg_soft_reference_lock_sha256 is None
        ):
            raise ODEBFContractError("BG-Soft reference lock is absent")
        expected_bg_initial = bg_soft_initial_contract(
            bg_soft_reference_lock, alias
        )
        frozen_bg_reference = bg_soft_frozen_reference(
            bg_soft_reference_lock, alias
        )
        runtime_instruction_id = BG_SOFT_INSTRUCTION_ID
        runtime_amendment_id = BG_SOFT_AMENDMENT_ID
        runtime_schema_namespace = BG_SOFT_SCHEMA_NAMESPACE
        observability_contract = None
        universal_live_arms: tuple[CommonColdArm, ...] = ()
        strength_live_arms = ()
    elif universal_observability_mode:
        from .ode_bf_observability import (
            paired_arm_ids,
            universal_observability_contract_receipt,
        )
        from .p1_universal_observability_panel import (
            UNIVERSAL_OBS_AMENDMENT_ID,
            UNIVERSAL_OBS_INSTRUCTION_ID,
            UNIVERSAL_OBS_SCHEMA_NAMESPACE,
        )

        if (
            universal_observability_lock is None
            or universal_observability_lock_sha256 is None
            or universal_frozen_r12_reference is None
        ):
            raise ODEBFContractError(
                "universal observability lock/reference is absent"
            )
        dynamic_id, hold_id = paired_arm_ids(universal_observability_cell)
        universal_live_arms = (
            CommonColdArm(dynamic_id),
            CommonColdArm(hold_id),
        )
        observability_contract = universal_observability_contract_receipt(
            instruction_id=UNIVERSAL_OBS_INSTRUCTION_ID,
            amendment_id=UNIVERSAL_OBS_AMENDMENT_ID,
            cell_id=universal_observability_cell,
        )
        expected_bg_initial = None
        frozen_bg_reference = None
        runtime_instruction_id = UNIVERSAL_OBS_INSTRUCTION_ID
        runtime_amendment_id = UNIVERSAL_OBS_AMENDMENT_ID
        runtime_schema_namespace = UNIVERSAL_OBS_SCHEMA_NAMESPACE
        strength_live_arms = ()
    else:
        expected_bg_initial = None
        frozen_bg_reference = None
        runtime_instruction_id = COMMON_COLD_INSTRUCTION_ID
        runtime_amendment_id = None
        runtime_schema_namespace = COMMON_COLD_SCHEMA_NAMESPACE
        observability_contract = None
        universal_live_arms = ()
        strength_live_arms = ()
    base_bytes = {name: tensor_sha256(value) for name, value in sorted(touched.items())}
    base_contract = _parameter_contract_sha256(touched)
    # The reused-Warm ordinal-6 RCA is completed before any target capture,
    # bootstrap, or controller action in this run.
    if strength_preserving_mode:
        ordinal_audit = {
            "status": "NOT_APPLICABLE_P1R19_R13_BASE",
            "model_forward_count": 0,
            "controller_decision_influence_count": 0,
        }
    else:
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
        if (
            strength_preserving_mode
            and strength_live_arms[0].scale is CommonColdScale.BATCH_GLOBAL
        ) or bg_soft_missing_cell_mode or (
            universal_observability_mode
            and universal_live_arms[0].scale
            is CommonColdScale.BATCH_GLOBAL
        ):
            rs_target = None
            rs_bootstrap = None
        else:
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
        if (
            strength_preserving_mode
            and strength_live_arms[0].scale is CommonColdScale.ROBUST_SHARED
        ) or (
            universal_observability_mode
            and universal_live_arms[0].scale
            is CommonColdScale.ROBUST_SHARED
        ):
            bg_target = None
            bg_bootstrap = None
        else:
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
    rs_sha = (
        None
        if rs_bootstrap is None
        else write_once(
            raw_root / "common-cold" / "bootstrap-rs.json", rs_bootstrap
        )
    )
    bg_sha = (
        None
        if bg_bootstrap is None
        else write_once(
            raw_root / "common-cold" / "bootstrap-bg.json", bg_bootstrap
        )
    )
    bootstrap_stage_payload = {
        "pre_residual_exact_zero": True,
        "rs_bootstrap_sha256": rs_sha,
        "bg_bootstrap_sha256": bg_sha,
        "rs_joint_entry_target_sha256": (
            None if rs_target is None else tensor_sha256(rs_target)
        ),
        "bg_joint_entry_target_sha256": (
            None if bg_target is None else tensor_sha256(bg_target)
        ),
        "tau_joint": 0.0,
    }
    if bg_soft_missing_cell_mode:
        bootstrap_stage_payload[
            "bg_soft_reuses_frozen_bg_neutral_bootstrap_contract"
        ] = True
    if universal_observability_mode:
        bootstrap_stage_payload.update(
            {
                "observability_contract": observability_contract,
                "live_cell": universal_observability_cell,
                "bootstrap_count": 1,
                "dynamic_hold_bootstrap_exact_shared": True,
            }
        )
    stages.record("post_bootstrap_contract", bootstrap_stage_payload)
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
    pair_initial_contracts: dict[str, dict[str, Any]] = {}
    live_arms = (
        strength_live_arms
        if strength_preserving_mode
        else universal_live_arms
        if universal_observability_mode
        else (
            CommonColdArm.BG_SOFT,
            CommonColdArm.BG_SOFT_TARGET_HOLD,
        )
        if bg_soft_missing_cell_mode
        else R10_COMMON_COLD_ARMS
    )
    for arm in live_arms:
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
        pair_initial_capture: dict[str, Any] = {}
        try:
            expected_initial = (
                initial_contracts.get(universal_live_arms[0].value)
                if universal_observability_mode and arm.target_hold
                else expected_bg_initial
                if arm is CommonColdArm.BG_SOFT
                else initial_contracts.get(CommonColdArm.BG_SOFT.value)
                if arm is CommonColdArm.BG_SOFT_TARGET_HOLD
                else initial_contracts.get(CommonColdArm.RS_NEUTRAL.value)
                if arm is CommonColdArm.RS_SOFT
                else None
            )
            rollout, initial_contract = _run_common_arm(
                model,
                tokenizer,
                requests,
                alias=alias,
                arm=arm,
                bootstrap_target=(
                    bg_target
                    if arm.scale is CommonColdScale.BATCH_GLOBAL
                    else rs_target
                ),
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
                pair_initial_contract_sink=(
                    pair_initial_capture if bg_soft_missing_cell_mode else None
                    if not universal_observability_mode
                    else pair_initial_capture
                ),
                typed_zero_positive=(
                    bg_soft_missing_cell_mode
                    or universal_observability_mode
                    or strength_preserving_mode
                ),
                record_six_context_objectives=(
                    bg_soft_missing_cell_mode
                    or universal_observability_mode
                ),
                observability_contract=(
                    observability_contract
                    if universal_observability_mode
                    else None
                ),
                strength_preserving=strength_preserving_mode,
            )
            rollouts[arm.value] = rollout
            initial_contracts[arm.value] = initial_contract
            if pair_initial_capture:
                pair_initial_contracts[arm.value] = dict(
                    pair_initial_capture
                )
        except ODEBFContractError as exc:
            if initial_capture:
                initial_contracts[arm.value] = dict(initial_capture)
            if pair_initial_capture:
                pair_initial_contracts[arm.value] = dict(
                    pair_initial_capture
                )
            failures[arm.value] = {
                "status": "ARM_LOCAL_TECHNICAL_FAILURE",
                "exception_type": type(exc).__name__,
                "message_sha256": hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
                "completed_snapshot_count": len(recorder.accepted_hashes),
                "receipt_links": recorder.links(),
            }
            write_once(raw_root / "common-cold" / f"failure-{arm.value}.json", failures[arm.value])
            stages.record(f"post_{arm.value.lower().replace('-', '_')}_failure", failures[arm.value])
        except ODEBFStateError as exc:
            if not universal_observability_mode:
                raise
            if initial_capture:
                initial_contracts[arm.value] = dict(initial_capture)
            if pair_initial_capture:
                pair_initial_contracts[arm.value] = dict(
                    pair_initial_capture
                )
            failures[arm.value] = {
                "status": "ARM_LOCAL_TECHNICAL_FAILURE",
                "exception_type": type(exc).__name__,
                "message_sha256": hashlib.sha256(
                    str(exc).encode("utf-8")
                ).hexdigest(),
                "completed_snapshot_count": len(recorder.accepted_hashes),
                "receipt_links": recorder.links(),
            }
            write_once(
                raw_root
                / "common-cold"
                / f"failure-{arm.value}.json",
                failures[arm.value],
            )
            stages.record(
                f"post_{arm.value.lower().replace('-', '_')}_failure",
                failures[arm.value],
            )
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
    if bg_soft_missing_cell_mode and (
        initial_contracts.get(CommonColdArm.BG_SOFT.value)
        != expected_bg_initial
    ):
        raise ODEBFContractError(
            "BG-Neutral/BG-Soft frozen initial contract differs"
        )
    if bg_soft_missing_cell_mode and (
        initial_contracts.get(CommonColdArm.BG_SOFT_TARGET_HOLD.value)
        != initial_contracts.get(CommonColdArm.BG_SOFT.value)
    ):
        raise ODEBFContractError(
            "BG-Soft main/target-hold initial contract differs"
        )
    if bg_soft_missing_cell_mode and (
        pair_initial_contracts.get(CommonColdArm.BG_SOFT.value)
        != pair_initial_contracts.get(
            CommonColdArm.BG_SOFT_TARGET_HOLD.value
        )
    ):
        raise ODEBFContractError(
            "BG-Soft main/target-hold selected k0 contract differs"
        )
    if universal_observability_mode:
        dynamic_arm, hold_arm = universal_live_arms
        if (
            dynamic_arm.value in initial_contracts
            and hold_arm.value in initial_contracts
            and initial_contracts.get(dynamic_arm.value)
            != initial_contracts.get(hold_arm.value)
        ):
            raise ODEBFContractError(
                "universal dynamic/hold initial contract differs"
            )
        if (
            dynamic_arm.value in pair_initial_contracts
            and hold_arm.value in pair_initial_contracts
            and pair_initial_contracts.get(dynamic_arm.value)
            != pair_initial_contracts.get(hold_arm.value)
        ):
            raise ODEBFContractError(
                "universal dynamic/hold selected k0 contract differs"
            )
    action_freeze = {
        "schema": f"{runtime_schema_namespace}-action-freeze/v1",
        "instruction_id": runtime_instruction_id,
        "request_order_sha256": request_order,
        "rollout_sha256": {key: value.rollout_sha256 for key, value in rollouts.items()},
        "arm_failures": failures,
        "actions_frozen_before_native_and_heldout": True,
        "native_or_direct_z_cold_access_count": 0,
    }
    if bg_soft_missing_cell_mode:
        action_freeze.update(
            {
                "amendment_id": runtime_amendment_id,
                "w0_native_bg_neutral_rerun_count": 0,
            }
        )
    if universal_observability_mode:
        action_freeze.update(
            {
                "amendment_id": runtime_amendment_id,
                "observability_contract": observability_contract,
                "live_cell": universal_observability_cell,
                "d1_d2_d3_observation_only": True,
                "observability_total_decision_influence_count": 0,
            }
        )
    action_freeze_sha = write_once(raw_root / "common-cold" / "action-freeze.json", action_freeze)
    if strength_preserving_mode:
        if len(live_arms) != 1:
            raise ODEBFContractError(
                "strength-preserving live trajectory count differs"
            )
        strength_arm = live_arms[0]
        if strength_arm.value not in rollouts:
            raise ODEBFContractError(
                "strength-preserving trajectory produced no evaluable prefix"
            )
        native_ledger = ComputeLedger()
        native_counter = ModelForwardCounter(model, native_ledger)
        try:
            native_edit_started = time.perf_counter()
            native_capture = capture_p1_native_entry(
                model,
                tokenizer,
                requests,
                hparams,
                projector,
                contexts,
                history_keys_by_layer=_history_keys(
                    P1HistoryLedger(
                        layer_order=COMMON_COLD_LAYER_ORDER,
                        maximum_records=40,
                    ),
                    COMMON_COLD_LAYER_ORDER,
                    risk=False,
                ),
                mutation_lock=__import__("threading").RLock(),
                ledger=native_ledger,
                residual_tolerance=controller_lock.residual_tolerance,
            )
            native_edit_wall = time.perf_counter() - native_edit_started
            native_eval_started = time.perf_counter()
            native_online = _evaluate_native_rewrite(
                model,
                tokenizer,
                requests,
                alias=alias,
                candidates=native_capture.native_candidates,
                ledger=native_ledger,
            )
            native_eval_wall = time.perf_counter() - native_eval_started
        finally:
            native_counter.close()
        native_sha = write_once(
            raw_root / "common-cold" / "N32_NATIVE-postfreeze.json",
            {
                "action_freeze_sha256": action_freeze_sha,
                "capture": native_capture.raw_free_payload(),
                "official_success": native_online.batch_success.raw_free_payload(),
                "compute": native_ledger.raw_free_payload(),
                "opened_after_action_freeze": True,
            },
        )
        selected_bootstrap = (
            bg_target
            if strength_arm.scale is CommonColdScale.BATCH_GLOBAL
            else rs_target
        )
        if selected_bootstrap is None:
            raise ODEBFContractError(
                "strength-preserving selected bootstrap is absent"
            )
        panel = _postfreeze_bg_soft_prefix_panel(
            model,
            tokenizer,
            alias=alias,
            requests=requests,
            contexts=contexts,
            hparams=hparams,
            target_layer_name=target_layer_name,
            dataset_path=dataset_path,
            rollouts={strength_arm.value: rollouts[strength_arm.value]},
            raw_root=raw_root,
            write_once=write_once,
            touched=touched,
            request_order_sha256=request_order,
            action_freeze_sha256=action_freeze_sha,
            frozen_reference={
                "base_checkpoint": strength_preserving_lock["base_checkpoint"],
                "base_tree": strength_preserving_lock["base_tree"],
                "stage_a_seal_root": strength_preserving_lock[
                    "stage_a_seal_root"
                ],
                "request_order_sha256": strength_preserving_lock[
                    "request_order_sha256"
                ],
            },
            receipt_schema_prefix=runtime_schema_namespace,
            instruction_id=STRENGTH_PRESERVING_INSTRUCTION_ID,
            amendment_id=STRENGTH_PRESERVING_AMENDMENT_ID,
            output_token="strength-preserving",
            bootstrap_targets={strength_arm.value: selected_bootstrap},
            controller_lookup_positions=lookup_positions,
            progress_actual_semantics=(
                "PHYSICAL_BF16_WONLY_TARGET_NEW_NLL_PROGRESS"
            ),
        )
        panel_sha = write_once(
            raw_root / "stepwise" / "strength-preserving-panel.json", panel
        )
        artifact_guard.assert_unchanged()
        if {
            name: tensor_sha256(value)
            for name, value in sorted(touched.items())
        } != base_bytes:
            raise ODEBFStateError(
                "strength-preserving final W0 restore differs"
            )
        rollout = rollouts[strength_arm.value]
        rollout_compute = rollout.ledger.raw_free_payload()
        rollout_components = rollout_compute["component_wall_seconds"]
        edit_core_time = sum(
            float(value)
            for key, value in rollout_components.items()
            if key.startswith("p1r19_edit_core_")
        )
        functional_probe_time = float(
            rollout_components.get("p1r19_functional_probe", 0.0)
        )
        stepwise_compute = panel.get("evaluation_compute", {})
        stepwise_eval_time = sum(
            float(value)
            for value in stepwise_compute.get(
                "component_wall_seconds", {}
            ).values()
        )
        job_compute = job_ledger.raw_free_payload()
        artifact_model_load_time = float(
            job_compute["component_wall_seconds"].get("model_load", 0.0)
        )
        common_runtime_time = time.perf_counter() - runtime_wall_started
        accounted_total_wall_time = common_runtime_time + sum(
            float(value)
            for value in job_compute["component_wall_seconds"].values()
        )
        terminal = {
            "schema": f"{runtime_schema_namespace}-terminal/v1",
            "instruction_id": STRENGTH_PRESERVING_INSTRUCTION_ID,
            "amendment_id": STRENGTH_PRESERVING_AMENDMENT_ID,
            "status": "STRENGTH_PRESERVING_DYNAMIC_TERMINAL",
            "alias": alias,
            "source_head": source_head,
            "method_id": STRENGTH_PRESERVING_METHOD_ID,
            "parent_method": common_cold_source_contract(),
            "live_cell": strength_arm.value,
            "allocation": strength_arm.scale.value,
            "routing": (
                "STRENGTH_PRESERVING_SOFT"
                if strength_arm.routing_arm is FixedE8Arm.SOFT
                else "STRENGTH_PRESERVING_NEUTRAL"
            ),
            "target_dynamics": "DYNAMIC_TARGET",
            "hold_arm_count": 0,
            "request_order_sha256": request_order,
            "stream_root_digest": stream["root_digest"],
            "strength_preserving_lock_sha256": strength_preserving_lock_sha256,
            "strength_preserving_lock_root_digest": strength_preserving_lock[
                "root_digest"
            ],
            "initial_contract": initial_contracts[strength_arm.value],
            "bootstrap_sha256": {
                strength_arm.scale.value: (
                    bg_sha
                    if strength_arm.scale is CommonColdScale.BATCH_GLOBAL
                    else rs_sha
                )
            },
            "action_freeze_sha256": action_freeze_sha,
            "n32_postfreeze_sha256": native_sha,
            "stepwise_panel_sha256": panel_sha,
            "accepted_snapshot_count": len(rollout.snapshots),
            "accepted_update_count": rollout.k_acc,
            "tau_final": float(rollout.accepted_t),
            "field_build_count": rollout.field_build_count,
            "scientific_retry_count": rollout.n_reject,
            "backtracking_count": 0,
            "first_hit_observation_only": True,
            "progress_floor": 0.0,
            "kappa": 0.0,
            "requested_progress_legacy": None,
            "hard_h_p_budget_influence_count": 0,
            "functional_veto_count": 0,
            "native_or_direct_z_controller_access_count": 0,
            "router_added_model_forward_count": 0,
            "router_added_backward_count": 0,
            "router_added_target_backward_count": 0,
            "router_added_processed_token_count": 0,
            "timing_decomposition": {
                "edit_core_time_seconds": edit_core_time,
                "functional_probe_time_seconds": functional_probe_time,
                "stepwise_eval_time_seconds": stepwise_eval_time,
                "artifact_model_load_time_seconds": artifact_model_load_time,
                "common_runtime_wall_time_seconds": common_runtime_time,
                "accounted_total_wall_time_seconds": accounted_total_wall_time,
                "scheduler_total_wall_time": "DEFERRED_TO_SACCT_TERMINAL",
                "rollout_compute": rollout_compute,
                "stepwise_eval_compute": stepwise_compute,
                "artifact_model_load_compute": job_compute,
                "native_reference": {
                    "edit_core_time_seconds": native_edit_wall,
                    "evaluation_time_seconds": native_eval_wall,
                    "end_to_end_time_seconds": native_edit_wall
                    + native_eval_wall,
                    "compute": native_ledger.raw_free_payload(),
                },
            },
            "native_overhead_reported_separately": True,
            "numerical_lock_sha256": numerical_sha256,
            "artifact_receipt": asdict(artifact_receipt),
            "context_sha256": context_sha256,
            "cuda_preflight": dict(cuda_runtime_receipt),
            "job_compute": job_compute,
            "final_w0_restored": True,
            "persistent_endpoint_commit_count": 0,
            "history_append_count": 0,
            "heldout_controller_access_count": 0,
            "scientific_promotion_authorized": False,
        }
        terminal_sha = write_once(destination / "terminal.json", terminal)
        manifest_sha = write_once(
            destination / "manifest.json",
            {
                "schema": f"{runtime_schema_namespace}-manifest/v1",
                "instruction_id": STRENGTH_PRESERVING_INSTRUCTION_ID,
                "amendment_id": STRENGTH_PRESERVING_AMENDMENT_ID,
                "status": terminal["status"],
                "alias": alias,
                "source_head": source_head,
                "live_cell": strength_arm.value,
                "terminal_sha256": terminal_sha,
                "stepwise_panel_sha256": panel_sha,
                "retry_submission_count": 0,
            },
        )
        return {
            "status": terminal["status"],
            "alias": alias,
            "terminal_sha256": terminal_sha,
            "manifest_sha256": manifest_sha,
            "final_w0_restored": True,
        }
    if universal_observability_mode:
        universal_rollouts = {
            arm.value: rollouts[arm.value]
            for arm in universal_live_arms
            if arm.value in rollouts
        }
        if not universal_rollouts:
            raise ODEBFContractError(
                "universal observability arms produced no evaluable prefix"
            )
        dynamic_arm, hold_arm = universal_live_arms
        selected_bootstrap = (
            bg_target
            if dynamic_arm.scale is CommonColdScale.BATCH_GLOBAL
            else rs_target
        )
        if selected_bootstrap is None:
            raise ODEBFContractError(
                "universal observability selected bootstrap is absent"
            )
        bootstrap_targets = {
            key: selected_bootstrap for key in universal_rollouts
        }
        output_token = (
            "universal-"
            + str(universal_observability_cell).lower().replace("-", "_")
        )
        panel = _postfreeze_bg_soft_prefix_panel(
            model,
            tokenizer,
            alias=alias,
            requests=requests,
            contexts=contexts,
            hparams=hparams,
            target_layer_name=target_layer_name,
            dataset_path=dataset_path,
            rollouts=universal_rollouts,
            raw_root=raw_root,
            write_once=write_once,
            touched=touched,
            request_order_sha256=request_order,
            action_freeze_sha256=action_freeze_sha,
            frozen_reference=universal_frozen_r12_reference,
            receipt_schema_prefix=(
                f"{runtime_schema_namespace}-observability"
            ),
            instruction_id=runtime_instruction_id,
            amendment_id=str(runtime_amendment_id),
            output_token=output_token,
            observability_contract=observability_contract,
            bootstrap_targets=bootstrap_targets,
            controller_lookup_positions=lookup_positions,
        )
        panel_sha = write_once(
            raw_root
            / "stepwise"
            / f"{output_token}-factorial-cell-panel.json",
            panel,
        )
        artifact_guard.assert_unchanged()
        if {
            name: tensor_sha256(value)
            for name, value in sorted(touched.items())
        } != base_bytes:
            raise ODEBFStateError(
                "universal observability final W0 restore differs"
            )
        variant_status = {
            arm.value: (
                universal_rollouts[arm.value].status
                if arm.value in universal_rollouts
                else failures[arm.value]["status"]
            )
            for arm in universal_live_arms
        }
        terminal = {
            "schema": f"{runtime_schema_namespace}-terminal/v1",
            "instruction_id": runtime_instruction_id,
            "amendment_id": runtime_amendment_id,
            "status": (
                "UNIVERSAL_OBSERVABILITY_FACTORIAL_CELL_TERMINAL"
                if not failures
                else (
                    "UNIVERSAL_OBSERVABILITY_FACTORIAL_CELL_"
                    "PARTIAL_TECHNICAL_TERMINAL"
                )
            ),
            "alias": alias,
            "source_head": source_head,
            "parent_method": common_cold_source_contract(),
            "live_cell": universal_observability_cell,
            "live_arms": [arm.value for arm in universal_live_arms],
            "factorial_shape": [2, 2, 2],
            "factorial_axes": observability_contract["factorial_axes"],
            "observability_contract": observability_contract,
            "d1_d2_d3_observation_only": True,
            "observability_total_decision_influence_count": 0,
            "scale": dynamic_arm.scale.value,
            "routing": dynamic_arm.routing_arm.value,
            "target_dynamics": [
                "DYNAMIC_TARGET",
                "TARGET_HOLD",
            ],
            "panel_kind": (
                "REUSED_R10_OUTCOME_SELECTED_SEAL_2X2X2_"
                "MECHANISTIC_FACTORIAL"
            ),
            "scientific_promotion_authorized": False,
            "request_order_sha256": request_order,
            "stream_root_digest": stream["root_digest"],
            "universal_numerical_lock_sha256": (
                universal_observability_lock_sha256
            ),
            "universal_numerical_lock_root_digest": (
                universal_observability_lock["root_digest"]
            ),
            "frozen_r12_reference": dict(
                universal_frozen_r12_reference
            ),
            "initial_contract": {
                key: value
                for key, value in sorted(initial_contracts.items())
                if key in {arm.value for arm in universal_live_arms}
            },
            "dynamic_target_hold_pair_initial_contract": (
                pair_initial_contracts.get(dynamic_arm.value)
            ),
            "bootstrap_sha256": {
                dynamic_arm.scale.value: (
                    bg_sha
                    if dynamic_arm.scale is CommonColdScale.BATCH_GLOBAL
                    else rs_sha
                )
            },
            "action_freeze_sha256": action_freeze_sha,
            "stepwise_panel_sha256": panel_sha,
            "variant_status": variant_status,
            "arm_failures": failures,
            "arm_failure_count": len(failures),
            "completed_arm_count": len(universal_rollouts),
            "all_live_arms_scientifically_evaluable": not failures,
            "accepted_snapshot_count": {
                arm.value: (
                    len(universal_rollouts[arm.value].snapshots)
                    if arm.value in universal_rollouts
                    else 0
                )
                for arm in universal_live_arms
            },
            "postfreeze_entry_snapshot_count": len(universal_rollouts),
            "cold_native_or_direct_z_access_count": 0,
            "scientific_retry_count": 0,
            "scientific_rejection_count": 0,
            "hard_h_p_budget_influence_count": 0,
            "first_hit_observation_only": True,
            "w0_native_rerun_count": 0,
            "numerical_lock_sha256": numerical_sha256,
            "artifact_receipt": asdict(artifact_receipt),
            "context_sha256": context_sha256,
            "cuda_preflight": dict(cuda_runtime_receipt),
            "job_compute": job_ledger.raw_free_payload(),
            "final_w0_restored": True,
            "persistent_endpoint_commit_count": 0,
            "history_append_count": 0,
            "heldout_controller_access_count": 0,
        }
        terminal_sha = write_once(destination / "terminal.json", terminal)
        manifest_sha = write_once(
            destination / "manifest.json",
            {
                "schema": f"{runtime_schema_namespace}-manifest/v1",
                "instruction_id": runtime_instruction_id,
                "amendment_id": runtime_amendment_id,
                "status": terminal["status"],
                "alias": alias,
                "source_head": source_head,
                "live_cell": universal_observability_cell,
                "terminal_sha256": terminal_sha,
                "stepwise_panel_sha256": panel_sha,
                "observability_contract_identity_sha256": (
                    observability_contract["identity_sha256"]
                ),
                "retry_submission_count": 0,
            },
        )
        return {
            "status": terminal["status"],
            "alias": alias,
            "terminal_sha256": terminal_sha,
            "manifest_sha256": manifest_sha,
            "final_w0_restored": True,
        }
    if bg_soft_missing_cell_mode:
        bg_rollouts = {
            arm.value: rollouts[arm.value]
            for arm in (
                CommonColdArm.BG_SOFT,
                CommonColdArm.BG_SOFT_TARGET_HOLD,
            )
            if arm.value in rollouts
        }
        if not bg_rollouts or frozen_bg_reference is None:
            raise ODEBFContractError(
                "BG-Soft arms did not produce an action-frozen rollout"
            )
        panel = _postfreeze_bg_soft_prefix_panel(
            model,
            tokenizer,
            alias=alias,
            requests=requests,
            contexts=contexts,
            hparams=hparams,
            target_layer_name=target_layer_name,
            dataset_path=dataset_path,
            rollouts=bg_rollouts,
            raw_root=raw_root,
            write_once=write_once,
            touched=touched,
            request_order_sha256=request_order,
            action_freeze_sha256=action_freeze_sha,
            frozen_reference=frozen_bg_reference,
        )
        panel_sha = write_once(
            raw_root / "stepwise" / "bg-soft-missing-cell-panel.json",
            panel,
        )
        artifact_guard.assert_unchanged()
        if {
            name: tensor_sha256(value) for name, value in sorted(touched.items())
        } != base_bytes:
            raise ODEBFStateError("BG-Soft final W0 restore differs")
        pair_terminal_status = (
            "BG_SOFT_MISSING_CELL_DIAGNOSTIC_TERMINAL"
            if not failures
            else "BG_SOFT_MISSING_CELL_DIAGNOSTIC_PARTIAL_TECHNICAL_TERMINAL"
        )
        variant_status = {
            arm.value: (
                bg_rollouts[arm.value].status
                if arm.value in bg_rollouts
                else failures[arm.value]["status"]
            )
            for arm in (
                CommonColdArm.BG_SOFT,
                CommonColdArm.BG_SOFT_TARGET_HOLD,
            )
        }
        terminal = {
            "schema": f"{runtime_schema_namespace}-terminal/v1",
            "instruction_id": runtime_instruction_id,
            "amendment_id": runtime_amendment_id,
            "status": pair_terminal_status,
            "alias": alias,
            "source_head": source_head,
            "parent_method": common_cold_source_contract(),
            "live_arms": [
                CommonColdArm.BG_SOFT.value,
                CommonColdArm.BG_SOFT_TARGET_HOLD.value,
            ],
            "scale": CommonColdScale.BATCH_GLOBAL.value,
            "routing": FixedE8Arm.SOFT.value,
            "panel_kind": "REUSED_R10_OUTCOME_SELECTED_SEAL_CAUSAL_DIAGNOSTIC",
            "scientific_promotion_authorized": False,
            "request_order_sha256": request_order,
            "stream_root_digest": stream["root_digest"],
            "reference_lock_sha256": bg_soft_reference_lock_sha256,
            "reference_lock_root_digest": bg_soft_reference_lock[
                "root_digest"
            ],
            "frozen_r10_reference": frozen_bg_reference,
            "initial_contract": {
                key: value
                for key, value in sorted(initial_contracts.items())
                if key in (
                    CommonColdArm.BG_SOFT.value,
                    CommonColdArm.BG_SOFT_TARGET_HOLD.value,
                )
            },
            "main_target_hold_pair_initial_contract": (
                pair_initial_contracts.get(CommonColdArm.BG_SOFT.value)
            ),
            "bootstrap_sha256": {"BG": bg_sha},
            "action_freeze_sha256": action_freeze_sha,
            "stepwise_panel_sha256": panel_sha,
            "variant_status": variant_status,
            "arm_failures": failures,
            "arm_failure_count": len(failures),
            "completed_arm_count": len(bg_rollouts),
            "all_live_arms_scientifically_evaluable": not failures,
            "accepted_snapshot_count": {
                arm.value: (
                    len(bg_rollouts[arm.value].snapshots)
                    if arm.value in bg_rollouts
                    else 0
                )
                for arm in (
                    CommonColdArm.BG_SOFT,
                    CommonColdArm.BG_SOFT_TARGET_HOLD,
                )
            },
            "cold_native_or_direct_z_access_count": 0,
            "scientific_retry_count": 0,
            "scientific_rejection_count": 0,
            "hard_h_p_budget_influence_count": 0,
            "first_hit_observation_only": True,
            "w0_native_bg_neutral_rerun_count": 0,
            "numerical_lock_sha256": numerical_sha256,
            "artifact_receipt": asdict(artifact_receipt),
            "context_sha256": context_sha256,
            "cuda_preflight": dict(cuda_runtime_receipt),
            "job_compute": job_ledger.raw_free_payload(),
            "final_w0_restored": True,
            "persistent_endpoint_commit_count": 0,
            "history_append_count": 0,
            "heldout_controller_access_count": 0,
        }
        terminal_sha = write_once(destination / "terminal.json", terminal)
        manifest_sha = write_once(
            destination / "manifest.json",
            {
                "schema": f"{runtime_schema_namespace}-manifest/v1",
                "instruction_id": runtime_instruction_id,
                "amendment_id": runtime_amendment_id,
                "status": terminal["status"],
                "alias": alias,
                "source_head": source_head,
                "terminal_sha256": terminal_sha,
                "stepwise_panel_sha256": panel_sha,
                "frozen_r10_reference": frozen_bg_reference,
                "retry_submission_count": 0,
            },
        )
        return {
            "status": terminal["status"],
            "alias": alias,
            "terminal_sha256": terminal_sha,
            "manifest_sha256": manifest_sha,
            "final_w0_restored": True,
        }
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
        "variant_status": {arm.value: (rollouts[arm.value].status if arm.value in rollouts else failures[arm.value]["status"]) for arm in R10_COMMON_COLD_ARMS},
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
