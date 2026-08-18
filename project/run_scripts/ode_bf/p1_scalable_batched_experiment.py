"""End-to-end P1R23 scalable streaming ODE-BF and Native controls."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .accounting import ComputeLedger
from .artifacts import load_rooted_json
from .atomic_runtime_optimization import AcceptedPhysicalStateMaterializer
from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .fixed_e8_runtime import (
    FixedE8EntryCapture,
    _fixed_e8_functional_basis_probe,
    fixed_e8_target_write_realization,
    fixed_e8_waypoint_factors,
)
from .fixed_e8_soft_routing import FixedE8Arm
from .functional import WaypointFactor, tensor_sha256
from .p0_runtime import ModelForwardCounter
from .p1_adaptive_runtime import (
    _controller_replay_entry,
    _factor_map,
    _factor_state,
    _functional_trial,
    _merge_factors,
    _parameter_contract_sha256,
    _risk_payload,
)
from .p1_backend import PinnedCovarianceRegistry
from .p1_controller import AcceptedLayerContribution, P1ControllerLock
from .p1_evaluator import CounterFactEvaluationCase
from .p1_replay import (
    Theta0TeacherCache,
    build_outer_entry_pretrained_cache,
)
from .p1_scalable_batched_runtime_panel import (
    P1R23_BG_ROUTING_ARMS,
    P1R23_RS_ROUTING_ARMS,
    P1R23_SIMPLEX_BG_ROUTING_ARMS,
    P1R23_SIMPLEX_RS_ROUTING_ARMS,
)
from .p1_state import ArmWeightSnapshot, P1Arm, P1HistoryLedger
from .sampling import StatelessReplaySchedule
from .scalable_batched_evaluator import (
    added_ninety_payload,
    evaluate_scalable_primary,
    load_scalable_cases_after_freeze,
)
from .scalable_batched_field import (
    ScalableBatchGlobalMetric,
    build_scalable_dynamic_field,
    build_scalable_routing_problem,
    scalable_metric_from_allocation,
    scalable_physical_signed_progress,
    target_update_from_existing_gradient,
)
from .scalable_batched_model import (
    ScalableCapturePlan,
    ScalableObjectivePlan,
    build_scalable_capture_plan,
    build_scalable_objective_plan,
    capture_scalable_physical_state,
    evaluate_scalable_target_new_objective,
)
from .scalable_batched_native import (
    capture_optimized_native_k1,
    materialize_native_candidates_once,
    restore_native_entry,
    run_official_native_apply,
)
from .scalable_batched_runtime import (
    DynamicRefreshLedger,
    P1R23_GRID_COUNT,
    P1R23_H,
    P1R23_INSTRUCTION_ID,
    P1R23_LAYER_ORDER,
    P1R23_METHOD_ID,
    ScalableComputeLedger,
    initial_target_from_capture,
    scalable_ordered_request_digest,
)
from .strength_preserving_routing import (
    STRENGTH_COVERAGE_EPSILON,
    solve_strength_preserving_routing,
)
from .progress_simplex_routing import (
    PROGRESS_SIMPLEX_INSTRUCTION_ID,
    PROGRESS_SIMPLEX_METHOD_ID,
    ProgressSimplexStatus,
    SIMPLEX_PRIMAL_TOLERANCE,
    progress_simplex_waypoint_factors,
    solve_progress_simplex_routing,
)
from .p1r24_atomic_strength import (
    P1R24AliasTargetLock,
    P1R24_INSTRUCTION_ID,
    P1R24_METHOD_ID,
    P1R24RoutingStatus,
    build_p1r24_kl_plan,
    evaluate_p1r24_kl,
    p1r24_cumulative_p_receipt,
    p1r24_disable_historical,
    p1r24_target_step,
    solve_p1r24_matched_routing,
    verify_p1r24_alphaedit_geometry,
)
from .p1r34_w_anchored_finite_demand import (
    P1R34DemandStatus,
    P1R34NonSemanticTargetMove,
    P1R34_INSTRUCTION_ID,
    P1R34_METHOD_ID,
    build_p1r34_finite_demand,
)
from .p1r35_full_current_residual import (
    P1R35_INSTRUCTION_ID,
    P1R35_METHOD_ID,
    apply_p1r35_full_current_residual,
)
from .p1r38_perrequest_target import (
    P1R38AdamState,
    P1R38_INSTRUCTION_ID,
    P1R38_METHOD_ID,
    prepare_p1r38_target_proposal,
    select_p1r38_target_proposal,
)
from .p1r39_normalized_gradient_target import (
    P1R39ControllerState,
    P1R39_METHOD_ID,
    prepare_p1r39_target_proposal,
    select_p1r39_target_proposal,
)
from .p1r42_objective_aligned_target import (
    P1R42ControllerState,
    P1R42_INSTRUCTION_ID,
    P1R42_METHOD_ID,
    prepare_p1r42_target_proposal,
    select_p1r42_target_proposal,
)
from .p1r43_rho_free_target import (
    P1R43ControllerState,
    P1R43_INSTRUCTION_ID,
    P1R43_METHOD_ID,
    prepare_p1r43_rescue_proposal,
    prepare_p1r43_target_proposal,
    select_p1r43_target_proposal,
)
from .p1r43_full_strength_routing import solve_p1r43_full_strength_routing
from .p1r51_requestwise_semantic_allocation import (
    P1R51ControllerState,
    P1R51_INSTRUCTION_ID,
    P1R51_METHOD_ID,
    prepare_p1r51_rescue_proposal,
    prepare_p1r51_target_proposal,
    select_p1r51_target_proposal,
)
from .p1r52_r42_safe_kdc import (
    P1R52_INSTRUCTION_ID,
    P1R52_METHOD_ID,
    prepare_p1r52_rescue_proposal,
    prepare_p1r52_target_proposal,
    select_p1r52_target_proposal,
)
from .p1r52_frozen_pi_quota_writer import (
    P1R52_FPIQ_INSTRUCTION_ID,
    P1R52_FPIQ_METHOD_ID,
    P1R52WriterPolicy,
    SEQUENTIAL_POLICIES,
    plan_sequential_writer,
    post_commit_identity,
)


P1R23_SCHEMA = "ode-edit-s05-p1r23-scalable-batched-runtime"


def _key_identity(keys: Mapping[int, torch.Tensor]) -> str:
    return canonical_hash(
        {
            str(layer): tensor_sha256(keys[layer])
            for layer in P1R23_LAYER_ORDER
        }
    )


def _model_w0_contract(touched: Mapping[str, torch.nn.Parameter]) -> str:
    return canonical_hash(
        {
            name: {
                "sha256": tensor_sha256(value),
                "pointer": int(value.data_ptr()),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
            for name, value in sorted(touched.items())
        }
    )


def _phase_add_capture(
    ledger: ScalableComputeLedger,
    phase: str,
    capture: Any,
) -> None:
    ledger.increment(
        phase,
        logical_forward_groups=1,
        model_forward_calls=int(capture.physical_forward_count),
        physical_microbatch_graphs=int(capture.physical_forward_count),
        processed_tokens=int(capture.processed_token_count),
        padded_tokens=int(capture.padded_token_count),
        capture_forward_calls=int(capture.physical_forward_count),
    )


def _phase_add_objective(
    ledger: ScalableComputeLedger,
    phase: str,
    result: Any,
    *,
    target: bool = False,
    slope: bool = False,
) -> None:
    ledger.increment(
        phase,
        logical_forward_groups=1,
        model_forward_calls=int(result.model_forward_count),
        physical_microbatch_graphs=int(result.model_forward_count),
        autograd_invocations=int(result.backward_count),
        backward_calls=int(result.backward_count),
        processed_tokens=int(result.processed_token_count),
        padded_tokens=int(result.padded_token_count),
        target_backward_calls=(int(result.backward_count) if target else 0),
        slope_backward_calls=(int(result.backward_count) if slope else 0),
    )


def _terminal_functional_payload(value: Any) -> dict[str, Any]:
    return {
        "historical": _risk_payload(value.historical),
        "pretrained": _risk_payload(value.pretrained),
        "pretrained_receipt_sha256": value.pretrained_trial_receipt.value_sha256,
        "historical_receipt_sha256": value.historical_trial_receipt.value_sha256,
    }


def _run_ode_arm(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    arm: FixedE8Arm,
    allocation: str,
    capture_plan: ScalableCapturePlan,
    objective_plan: ScalableObjectivePlan,
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    controller_lock: P1ControllerLock,
    arm_state: Any,
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: StatelessReplaySchedule,
    outer_entry_p_cache: Any,
    theta0_cache: Theta0TeacherCache,
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    raw_root: Path,
    write_once: Any,
    progress_simplex: bool = False,
    p1r24: bool = False,
    p1r34: bool = False,
    p1r35: bool = False,
    p1r38: bool = False,
    p1r39: bool = False,
    p1r42: bool = False,
    p1r43: bool = False,
    p1r51: bool = False,
    p1r52: bool = False,
    p1r52_writer_policy: P1R52WriterPolicy | str | None = None,
) -> dict[str, Any]:
    if arm not in (FixedE8Arm.NEUTRAL, FixedE8Arm.SOFT):
        raise ODEBFContractError("P1R23 ODE routing arm differs")
    request_count = len(requests)
    if request_count not in ((1, 10, 100) if p1r52 else (1, 10) if p1r24 else (10, 100)):
        raise ODEBFContractError("P1R23 ODE request count differs")
    if allocation not in ("RS", "BG"):
        raise ODEBFContractError("P1R23 ODE target allocation differs")
    if p1r34 and not p1r24:
        raise ODEBFContractError("P1R34 requires the P1R24 science path")
    if p1r35 and not p1r34:
        raise ODEBFContractError("P1R35 requires the frozen P1R34 science path")
    if p1r38 and not p1r35:
        raise ODEBFContractError("P1R38 requires the frozen P1R35 writer path")
    if p1r38 and allocation not in ("RS",):
        raise ODEBFContractError("P1R38 has no RS/BG target factorial")
    if p1r39 and (
        not p1r35
        or p1r38
        or allocation not in ("RS",)
        or arm is not FixedE8Arm.NEUTRAL
    ):
        raise ODEBFContractError("P1R39 normalized-gradient Neutral path differs")
    if p1r42 and (
        not p1r35
        or p1r38
        or p1r39
        or allocation not in ("RS",)
        or arm not in (FixedE8Arm.NEUTRAL, FixedE8Arm.SOFT)
    ):
        raise ODEBFContractError("P1R42 objective-aligned path differs")
    if p1r43 and (
        not p1r35
        or p1r38
        or p1r39
        or p1r42
        or allocation not in ("RS",)
        or arm not in (FixedE8Arm.NEUTRAL, FixedE8Arm.SOFT)
    ):
        raise ODEBFContractError("P1R43 rho-free semantic-first path differs")
    if p1r51 and (
        not p1r35
        or p1r38
        or p1r39
        or p1r42
        or p1r43
        or allocation not in ("RS",)
        or arm not in (FixedE8Arm.NEUTRAL, FixedE8Arm.SOFT)
    ):
        raise ODEBFContractError("P1R51 request-wise allocation path differs")
    if p1r52 and (
        not p1r35
        or p1r38
        or p1r39
        or p1r42
        or p1r43
        or p1r51
        or allocation not in ("RS",)
        or arm not in (FixedE8Arm.NEUTRAL, FixedE8Arm.SOFT)
    ):
        raise ODEBFContractError("P1R52 R42-safe KDC path differs")
    writer_policy = (
        None
        if p1r52_writer_policy is None
        else P1R52WriterPolicy(p1r52_writer_policy)
    )
    if writer_policy is not None and (
        not p1r52 or arm is not FixedE8Arm.SOFT
    ):
        raise ODEBFContractError("P1R52-FPiQ writer policy path differs")
    arm_label = (
        f"P1R52-FPIQ-{writer_policy.value}"
        if writer_policy is not None
        else f"P1R52-RSA-R42SAFEKDC-M1-{'NEUTRAL' if arm is FixedE8Arm.NEUTRAL else 'SOFT'}"
        if p1r52
        else f"P1R43-RSA-A1-{'NEUTRAL' if arm is FixedE8Arm.NEUTRAL else 'SOFT'}"
        if p1r51
        else f"PR-P1R43-RHO-FREE-SEMANTIC-FIRST-{'NEUTRAL' if arm is FixedE8Arm.NEUTRAL else 'SOFT'}"
        if p1r43
        else f"PR-P1R42-OBJECTIVE-ALIGNED-{'NEUTRAL' if arm is FixedE8Arm.NEUTRAL else 'SOFT'}"
        if p1r42
        else "PR-P1R39-NORMALIZED-GRADIENT-NEUTRAL"
        if p1r39
        else f"PR-P1R35-{'NEUTRAL' if arm is FixedE8Arm.NEUTRAL else 'SOFT'}"
        if p1r38
        else f"{allocation}-P1R35-FULL-CURRENT-RESIDUAL-{'NEUTRAL' if arm is FixedE8Arm.NEUTRAL else 'SOFT'}"
        if p1r35
        else f"{allocation}-P1R34-W-FINITE-DEMAND-{'NEUTRAL' if arm is FixedE8Arm.NEUTRAL else 'SOFT'}"
        if p1r34
        else f"{allocation}-P1R24-{'NEUTRAL' if arm is FixedE8Arm.NEUTRAL else 'SOFT'}"
        if p1r24
        else
        f"{allocation}-SIMPLEX-"
        f"{'NEUTRAL' if arm is FixedE8Arm.NEUTRAL else 'SOFT'}"
        if progress_simplex
        else f"{allocation}-{'NEUTRAL' if arm is FixedE8Arm.NEUTRAL else 'SOFT'}"
    )
    history = arm_state.history
    legacy_ledger = arm_state.ledger
    compute = ScalableComputeLedger()
    refresh = DynamicRefreshLedger()
    entry_capture = FixedE8EntryCapture(
        {name: value.detach().cpu().clone() for name, value in base_values.items()},
        dict(base_receipt.parameter_sha256),
    )
    materializer = AcceptedPhysicalStateMaterializer(model, base_values)
    accepted_by_layer: dict[int, list[AcceptedLayerContribution]] = {
        layer: [] for layer in P1R23_LAYER_ORDER
    }
    current_factors: dict[str, tuple[WaypointFactor, ...]] = {
        name: () for name in touched
    }
    started = time.perf_counter()
    physical_started = time.perf_counter()
    physical = capture_scalable_physical_state(model, capture_plan, hparams)
    compute.add_wall("initial_physical_capture", time.perf_counter() - physical_started)
    _phase_add_capture(compute, "initial_physical_capture", physical)
    initial = initial_target_from_capture(physical)
    metric = scalable_metric_from_allocation(
        initial.target_z, objective_plan.request_order_sha256, allocation
    )
    current_target = initial.target_z.clone()
    current_terminal = initial.current_terminal_z.clone()
    target_origin = current_target.clone()
    frozen_mask: tuple[bool, ...] = tuple(False for _ in range(request_count))
    p1r38_state = P1R38AdamState.zero(current_target) if p1r38 else None
    p1r39_state = P1R39ControllerState.zero(current_target) if p1r39 else None
    p1r42_state = P1R42ControllerState.zero(current_target) if p1r42 else None
    p1r43_state = P1R43ControllerState.zero(current_target) if p1r43 else None
    p1r51_state = P1R51ControllerState.zero(current_target) if p1r51 else None
    p1r52_state = P1R51ControllerState.zero(current_target) if p1r52 else None
    p1r24_target_lock = P1R24AliasTargetLock.for_alias(alias) if p1r24 else None
    p1r24_alpha_geometry = (
        verify_p1r24_alphaedit_geometry(
            hparams,
            p1r24_target_lock,
            easyedit_root=Path("/mnt/raid5/janghj/EasyEdit"),
        )
        if p1r24 and p1r24_target_lock is not None
        else None
    )
    p1r24_kl_plan = (
        build_p1r24_kl_plan(
            tokenizer,
            requests,
            request_order_sha256=objective_plan.request_order_sha256,
            request_microbatch_size=min(objective_plan.request_microbatch_size, request_count),
            fact_token_strategy=hparams.fact_token,
        )
        if p1r24
        else None
    )
    p1r24_kl_teacher: tuple[torch.Tensor, ...] | None = None
    p1r24_kl_teacher_sha256: str | None = None
    if p1r24:
        assert p1r24_kl_plan is not None
        teacher_result, p1r24_kl_teacher = evaluate_p1r24_kl(
            model, p1r24_kl_plan, teacher_log_probs=None
        )
        p1r24_kl_teacher_sha256 = canonical_hash(
            {
                "teacher_tensor_sha256": [
                    tensor_sha256(value) for value in p1r24_kl_teacher
                ],
                "request_order_sha256": objective_plan.request_order_sha256,
                "capture_count": 1,
            }
        )
        compute.increment(
            "outer_entry_kl_teacher",
            logical_forward_groups=1,
            model_forward_calls=teacher_result.model_forward_count,
            physical_microbatch_graphs=teacher_result.model_forward_count,
            processed_tokens=teacher_result.processed_token_count,
        )
    initial_w0 = _model_w0_contract(touched)
    accepted: list[dict[str, Any]] = []
    delayed: list[dict[str, Any]] = []
    request_realization_sha256: list[str] = []
    pending: dict[str, Any] | None = None
    terminal_objective_count = 0
    terminal_functional: dict[str, Any] | None = None
    restored = False
    counter = ModelForwardCounter(model, legacy_ledger)
    try:
        for step_index in range(P1R23_GRID_COUNT):
            state_before = _parameter_contract_sha256(touched)
            replay_entry = (
                None
                if p1r24
                else _controller_replay_entry(
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
            )
            target_started = time.perf_counter()
            target_variable = (
                current_target.detach()
                .to(device=next(model.parameters()).device, dtype=torch.float32)
                .clone()
                .requires_grad_(True)
            )
            target_result = evaluate_scalable_target_new_objective(
                model,
                objective_plan,
                target_state=target_variable,
                current_terminal=current_terminal,
                target_layer_name=hparams.layer_module_tmp.format(
                    int(hparams.layers[-1])
                ),
            )
            if target_result.target_gradient is None:
                raise ODEBFContractError("P1R23 target gradient is absent")
            if p1r24:
                assert p1r24_kl_plan is not None and p1r24_kl_teacher is not None
                assert p1r24_target_lock is not None
                kl_result, _ = evaluate_p1r24_kl(
                    model,
                    p1r24_kl_plan,
                    teacher_log_probs=p1r24_kl_teacher,
                    target_state=target_variable,
                    current_terminal=current_terminal,
                    target_layer_name=hparams.layer_module_tmp.format(int(hparams.layers[-1])),
                )
                _phase_add_objective(compute, "target_kl_gradient", kl_result, target=True)
                if p1r52:
                    assert p1r52_state is not None
                    assert p1r24_kl_teacher_sha256 is not None
                    proposal52 = prepare_p1r52_target_proposal(
                        current_target,
                        current_terminal,
                        target_origin,
                        target_result,
                        kl_result,
                        p1r24_target_lock,
                        p1r52_state,
                        alias=alias,
                        step_index=step_index,
                        shared_speed=float(metric.shared_speed),
                        kl_teacher_input_sha256=p1r24_kl_teacher_sha256,
                    )
                    endpoint_started = time.perf_counter()
                    primary_state = proposal52.primary_step.target_next.detach().to(
                        device=next(model.parameters()).device, dtype=torch.float32
                    )
                    primary_endpoint = evaluate_scalable_target_new_objective(
                        model,
                        objective_plan,
                        target_state=primary_state,
                        current_terminal=current_terminal,
                        target_layer_name=hparams.layer_module_tmp.format(
                            int(hparams.layers[-1])
                        ),
                        target_gradient_required=False,
                    )
                    compute.add_wall(
                        "finite_demand_endpoint_forward",
                        time.perf_counter() - endpoint_started,
                    )
                    _phase_add_objective(
                        compute, "finite_demand_endpoint_forward", primary_endpoint
                    )
                    rescue52 = prepare_p1r52_rescue_proposal(
                        proposal52,
                        current_target,
                        current_terminal,
                        target_result,
                        primary_endpoint,
                        step_index=step_index,
                    )
                    rescue_endpoint = None
                    if rescue52.rescue_step is not None:
                        rescue_started = time.perf_counter()
                        rescue_state = rescue52.rescue_step.target_next.detach().to(
                            device=next(model.parameters()).device,
                            dtype=torch.float32,
                        )
                        rescue_endpoint = evaluate_scalable_target_new_objective(
                            model,
                            objective_plan,
                            target_state=rescue_state,
                            current_terminal=current_terminal,
                            target_layer_name=hparams.layer_module_tmp.format(
                                int(hparams.layers[-1])
                            ),
                            target_gradient_required=False,
                        )
                        compute.add_wall(
                            "scalar_corrector_endpoint_forward",
                            time.perf_counter() - rescue_started,
                        )
                        _phase_add_objective(
                            compute,
                            "scalar_corrector_endpoint_forward",
                            rescue_endpoint,
                        )
                    selected52 = select_p1r52_target_proposal(
                        proposal52,
                        rescue52,
                        current_target,
                        current_terminal,
                        target_result,
                        primary_endpoint,
                        rescue_endpoint,
                        step_index=step_index,
                    )
                    p1r52_state = selected52.next_state
                    target_step = selected52.target_step
                    finite_endpoint = selected52.selected_endpoint
                elif p1r51:
                    assert p1r51_state is not None
                    proposal51 = prepare_p1r51_target_proposal(
                        current_target,
                        current_terminal,
                        target_result,
                        p1r51_state,
                        alias=alias,
                        step_index=step_index,
                        shared_speed=float(metric.shared_speed),
                    )
                    endpoint_started = time.perf_counter()
                    primary_state = proposal51.primary_step.target_next.detach().to(
                        device=next(model.parameters()).device, dtype=torch.float32
                    )
                    primary_endpoint = evaluate_scalable_target_new_objective(
                        model,
                        objective_plan,
                        target_state=primary_state,
                        current_terminal=current_terminal,
                        target_layer_name=hparams.layer_module_tmp.format(
                            int(hparams.layers[-1])
                        ),
                        target_gradient_required=False,
                    )
                    compute.add_wall(
                        "finite_demand_endpoint_forward",
                        time.perf_counter() - endpoint_started,
                    )
                    _phase_add_objective(
                        compute, "finite_demand_endpoint_forward", primary_endpoint
                    )
                    rescue51 = prepare_p1r51_rescue_proposal(
                        proposal51,
                        current_target,
                        current_terminal,
                        target_result,
                        primary_endpoint,
                        step_index=step_index,
                    )
                    rescue_endpoint = None
                    if rescue51.rescue_step is not None:
                        rescue_started = time.perf_counter()
                        rescue_state = rescue51.rescue_step.target_next.detach().to(
                            device=next(model.parameters()).device,
                            dtype=torch.float32,
                        )
                        rescue_endpoint = evaluate_scalable_target_new_objective(
                            model,
                            objective_plan,
                            target_state=rescue_state,
                            current_terminal=current_terminal,
                            target_layer_name=hparams.layer_module_tmp.format(
                                int(hparams.layers[-1])
                            ),
                            target_gradient_required=False,
                        )
                        compute.add_wall(
                            "scalar_corrector_endpoint_forward",
                            time.perf_counter() - rescue_started,
                        )
                        _phase_add_objective(
                            compute,
                            "scalar_corrector_endpoint_forward",
                            rescue_endpoint,
                        )
                    selected51 = select_p1r51_target_proposal(
                        proposal51,
                        rescue51,
                        current_target,
                        current_terminal,
                        target_result,
                        primary_endpoint,
                        rescue_endpoint,
                        step_index=step_index,
                    )
                    p1r51_state = selected51.next_state
                    target_step = selected51.target_step
                    finite_endpoint = selected51.selected_endpoint
                elif p1r43:
                    assert p1r43_state is not None
                    proposal43 = prepare_p1r43_target_proposal(
                        current_target,
                        current_terminal,
                        target_result,
                        p1r43_state,
                        alias=alias,
                        step_index=step_index,
                        shared_speed=float(metric.shared_speed),
                    )
                    endpoint_started = time.perf_counter()
                    primary_state = proposal43.primary_step.target_next.detach().to(
                        device=next(model.parameters()).device, dtype=torch.float32
                    )
                    primary_endpoint = evaluate_scalable_target_new_objective(
                        model,
                        objective_plan,
                        target_state=primary_state,
                        current_terminal=current_terminal,
                        target_layer_name=hparams.layer_module_tmp.format(
                            int(hparams.layers[-1])
                        ),
                        target_gradient_required=False,
                    )
                    compute.add_wall(
                        "finite_demand_endpoint_forward",
                        time.perf_counter() - endpoint_started,
                    )
                    _phase_add_objective(
                        compute, "finite_demand_endpoint_forward", primary_endpoint
                    )
                    rescue43 = prepare_p1r43_rescue_proposal(
                        proposal43,
                        current_target,
                        current_terminal,
                        target_result,
                        primary_endpoint,
                        step_index=step_index,
                    )
                    rescue_endpoint = None
                    if rescue43.rescue_step is not None:
                        rescue_started = time.perf_counter()
                        rescue_state = rescue43.rescue_step.target_next.detach().to(
                            device=next(model.parameters()).device,
                            dtype=torch.float32,
                        )
                        rescue_endpoint = evaluate_scalable_target_new_objective(
                            model,
                            objective_plan,
                            target_state=rescue_state,
                            current_terminal=current_terminal,
                            target_layer_name=hparams.layer_module_tmp.format(
                                int(hparams.layers[-1])
                            ),
                            target_gradient_required=False,
                        )
                        compute.add_wall(
                            "scalar_corrector_endpoint_forward",
                            time.perf_counter() - rescue_started,
                        )
                        _phase_add_objective(
                            compute,
                            "scalar_corrector_endpoint_forward",
                            rescue_endpoint,
                        )
                    selected43 = select_p1r43_target_proposal(
                        proposal43,
                        rescue43,
                        current_target,
                        current_terminal,
                        target_result,
                        primary_endpoint,
                        rescue_endpoint,
                        step_index=step_index,
                    )
                    p1r43_state = selected43.next_state
                    target_step = selected43.target_step
                    finite_endpoint = selected43.selected_endpoint
                elif p1r42:
                    assert p1r42_state is not None
                    proposal42 = prepare_p1r42_target_proposal(
                        current_target,
                        current_terminal,
                        target_origin,
                        target_result,
                        kl_result,
                        p1r24_target_lock,
                        p1r42_state,
                        alias=alias,
                        step_index=step_index,
                        shared_speed=float(metric.shared_speed),
                    )
                    endpoint_started = time.perf_counter()
                    endpoint_state = (
                        current_terminal.detach().to(
                            device=next(model.parameters()).device,
                            dtype=torch.float32,
                        )
                        + proposal42.trial_step.required_displacement.detach().to(
                            device=next(model.parameters()).device,
                            dtype=torch.float32,
                        )
                    ).contiguous()
                    finite_endpoint = evaluate_scalable_target_new_objective(
                        model,
                        objective_plan,
                        target_state=endpoint_state,
                        current_terminal=current_terminal,
                        target_layer_name=hparams.layer_module_tmp.format(
                            int(hparams.layers[-1])
                        ),
                        target_gradient_required=False,
                    )
                    compute.add_wall(
                        "finite_demand_endpoint_forward",
                        time.perf_counter() - endpoint_started,
                    )
                    _phase_add_objective(
                        compute, "finite_demand_endpoint_forward", finite_endpoint
                    )
                    selected42 = select_p1r42_target_proposal(
                        proposal42,
                        current_target,
                        current_terminal,
                        target_result,
                        finite_endpoint,
                        step_index=step_index,
                    )
                    p1r42_state = selected42.next_state
                    target_step = selected42.target_step
                    finite_endpoint = selected42.selected_endpoint
                elif p1r39:
                    assert p1r39_state is not None
                    proposal39 = prepare_p1r39_target_proposal(
                        current_target,
                        current_terminal,
                        target_origin,
                        target_result,
                        kl_result,
                        p1r24_target_lock,
                        p1r39_state,
                        alias=alias,
                        step_index=step_index,
                        shared_speed=float(metric.shared_speed),
                    )
                    endpoint_started = time.perf_counter()
                    endpoint_state = (
                        current_terminal.detach().to(
                            device=next(model.parameters()).device, dtype=torch.float32
                        )
                        + proposal39.trial_step.required_displacement.detach().to(
                            device=next(model.parameters()).device, dtype=torch.float32
                        )
                    ).contiguous()
                    finite_endpoint = evaluate_scalable_target_new_objective(
                        model,
                        objective_plan,
                        target_state=endpoint_state,
                        current_terminal=current_terminal,
                        target_layer_name=hparams.layer_module_tmp.format(
                            int(hparams.layers[-1])
                        ),
                        target_gradient_required=False,
                    )
                    compute.add_wall(
                        "finite_demand_endpoint_forward",
                        time.perf_counter() - endpoint_started,
                    )
                    _phase_add_objective(
                        compute, "finite_demand_endpoint_forward", finite_endpoint
                    )
                    selected39 = select_p1r39_target_proposal(
                        proposal39,
                        current_target,
                        current_terminal,
                        target_result,
                        finite_endpoint,
                        step_index=step_index,
                    )
                    p1r39_state = selected39.next_state
                    target_step = selected39.target_step
                    finite_endpoint = selected39.selected_endpoint
                elif p1r38:
                    assert p1r38_state is not None
                    proposal = prepare_p1r38_target_proposal(
                        current_target,
                        current_terminal,
                        target_origin,
                        target_result,
                        kl_result,
                        p1r24_target_lock,
                        p1r38_state,
                        alias=alias,
                        step_index=step_index,
                        request_cap_radius=P1R23_H * float(metric.shared_speed),
                    )
                    trial_step = apply_p1r35_full_current_residual(
                        proposal.trial_step,
                        current_target=current_target,
                        current_terminal=current_terminal,
                        step_index=step_index,
                    )
                    endpoint_started = time.perf_counter()
                    endpoint_state = (
                        current_terminal.detach()
                        .to(device=next(model.parameters()).device, dtype=torch.float32)
                        + trial_step.required_displacement.detach().to(
                            device=next(model.parameters()).device, dtype=torch.float32
                        )
                    ).contiguous()
                    finite_endpoint = evaluate_scalable_target_new_objective(
                        model,
                        objective_plan,
                        target_state=endpoint_state,
                        current_terminal=current_terminal,
                        target_layer_name=hparams.layer_module_tmp.format(
                            int(hparams.layers[-1])
                        ),
                        target_gradient_required=False,
                    )
                    compute.add_wall(
                        "finite_demand_endpoint_forward",
                        time.perf_counter() - endpoint_started,
                    )
                    _phase_add_objective(
                        compute, "finite_demand_endpoint_forward", finite_endpoint
                    )
                    selected = select_p1r38_target_proposal(
                        proposal,
                        p1r38_state,
                        current_target,
                        current_terminal,
                        target_result,
                        finite_endpoint,
                        step_index=step_index,
                    )
                    p1r38_state = selected.next_state
                    target_step = apply_p1r35_full_current_residual(
                        selected.target_step,
                        current_target=current_target,
                        current_terminal=current_terminal,
                        step_index=step_index,
                    )
                    finite_endpoint = selected.selected_endpoint
                else:
                    target_step = p1r24_target_step(
                        current_target,
                        current_terminal,
                        target_origin,
                        target_result,
                        kl_result,
                        metric,
                        p1r24_target_lock,
                        step_index=step_index,
                        frozen_mask=frozen_mask,
                    )
                    if p1r35:
                        target_step = apply_p1r35_full_current_residual(
                            target_step,
                            current_target=current_target,
                            current_terminal=current_terminal,
                            step_index=step_index,
                        )
                    if p1r34:
                        endpoint_started = time.perf_counter()
                        endpoint_state = (
                            current_terminal.detach()
                            .to(device=next(model.parameters()).device, dtype=torch.float32)
                            + target_step.required_displacement.detach().to(
                                device=next(model.parameters()).device, dtype=torch.float32
                            )
                        ).contiguous()
                        finite_endpoint = evaluate_scalable_target_new_objective(
                            model,
                            objective_plan,
                            target_state=endpoint_state,
                            current_terminal=current_terminal,
                            target_layer_name=hparams.layer_module_tmp.format(
                                int(hparams.layers[-1])
                            ),
                            target_gradient_required=False,
                        )
                        compute.add_wall(
                            "finite_demand_endpoint_forward",
                            time.perf_counter() - endpoint_started,
                        )
                        _phase_add_objective(
                            compute, "finite_demand_endpoint_forward", finite_endpoint
                        )
                target_next = target_step.target_next
                target_velocity = target_step.write_velocity
                alpha_req = target_step.rho_write
                target_receipt = dict(target_step.receipt)
                frozen_mask = target_step.frozen_mask
            else:
                target_next, target_velocity, alpha_req, target_receipt = (
                    target_update_from_existing_gradient(
                        current_target,
                        target_result.target_gradient,
                        metric,
                    )
                )
            compute.add_wall("target_gradient", time.perf_counter() - target_started)
            _phase_add_objective(
                compute, "target_gradient", target_result, target=True
            )
            field_started = time.perf_counter()
            field = build_scalable_dynamic_field(
                model,
                tokenizer,
                requests,
                hparams,
                projector,
                contexts,
                target_state=(
                    current_terminal + target_velocity
                    if p1r24
                    else target_next
                ),
                current_terminal=current_terminal,
                captured_keys_by_layer=physical.keys_by_layer,
                accepted_waypoint=step_index,
                covariance_registry=covariance_registry,
                projector_sha256=projector_sha256,
                residual_tolerance=controller_lock.residual_tolerance,
                ledger=legacy_ledger,
            )
            signed, slope_result = scalable_physical_signed_progress(
                model, objective_plan, field
            )
            compute.add_wall("field_and_physical_slope", time.perf_counter() - field_started)
            _phase_add_objective(
                compute, "physical_slope", slope_result, slope=True
            )
            current_nll = float(slope_result.loss)
            finite_demand = None
            if p1r34:
                finite_demand = build_p1r34_finite_demand(
                    step_index=step_index,
                    l_base=current_nll,
                    endpoint=finite_endpoint,
                    target_step=target_step,
                    current_target=current_target,
                    current_terminal=current_terminal,
                )
                target_receipt = {
                    **target_receipt,
                    "rho_old_signed": target_step.rho_write_signed,
                    "rho_old": target_step.rho_write,
                    "rho_finite_signed": finite_demand.rho_finite_signed,
                    "rho_write": finite_demand.rho_write,
                    "writer_demand": finite_demand.receipt,
                }
                target_receipt["identity_sha256"] = canonical_hash(target_receipt)
                if finite_demand.status is P1R34DemandStatus.NON_SEMANTIC_TARGET_MOVE:
                    write_once(
                        raw_root
                        / "scientific-boundaries"
                        / f"non-semantic-target-move-k{step_index}.json",
                        finite_demand.receipt,
                    )
                    raise P1R34NonSemanticTargetMove("NON_SEMANTIC_TARGET_MOVE")
                assert finite_demand.rho_write is not None
                alpha_req = finite_demand.rho_write
            if pending is not None:
                actual = float(pending["source_nll"] - current_nll)
                item = {
                    "transition_index": int(pending["step_index"]) + 1,
                    "source_mean_target_new_nll": float(pending["source_nll"]),
                    "next_field_mean_target_new_nll": current_nll,
                    "actual": actual,
                    "predicted": float(pending["predicted"]),
                    "realization_ratio": actual
                    / (float(pending["alpha_apply"]) + STRENGTH_COVERAGE_EPSILON),
                    "completion": "NEXT_REFRESHED_FIELD_W_ONLY_NLL_REUSE",
                    "candidate_objective_inner_count": 0,
                    "linearization_error": actual - float(pending["predicted"]),
                }
                if p1r42 or p1r43 or p1r51 or p1r52:
                    source_values = tuple(
                        float(value)
                        for value in pending["source_per_request_nll"]
                    )
                    next_values = tuple(
                        float(value) for value in slope_result.per_request_values
                    )
                    held_mask = tuple(
                        bool(value) for value in pending["semantic_held_mask"]
                    )
                    if not (
                        len(source_values)
                        == len(next_values)
                        == len(held_mask)
                        == request_count
                    ):
                        raise ODEBFStateError(
                            "requestwise delayed realization geometry differs"
                        )
                    actual_by_request = tuple(
                        source - next_value
                        for source, next_value in zip(
                            source_values, next_values, strict=True
                        )
                    )
                    realization = {
                        "schema": (
                            "ode-edit-s05-p1r52-rsa-r42safekdc-requestwise-w-only-realization/v1"
                            if p1r52
                            else "ode-edit-s05-p1r51-rsa-a1-requestwise-w-only-realization/v1"
                            if p1r51
                            else "ode-edit-s05-p1r43-requestwise-w-only-realization/v1"
                            if p1r43
                            else "ode-edit-s05-p1r42-requestwise-w-only-realization/v1"
                        ),
                        "instruction_id": (
                            P1R52_INSTRUCTION_ID
                            if p1r52
                            else P1R51_INSTRUCTION_ID
                            if p1r51
                            else P1R43_INSTRUCTION_ID
                            if p1r43
                            else P1R42_INSTRUCTION_ID
                        ),
                        "method_id": (
                            P1R52_METHOD_ID
                            if p1r52
                            else P1R51_METHOD_ID
                            if p1r51
                            else P1R43_METHOD_ID
                            if p1r43
                            else P1R42_METHOD_ID
                        ),
                        "transition_index": int(pending["step_index"]) + 1,
                        "source_w_only_target_new_nll_by_request": list(
                            source_values
                        ),
                        "next_w_only_target_new_nll_by_request": list(next_values),
                        "actual_w_only_progress_by_request": list(actual_by_request),
                        "semantic_held_mask": list(held_mask),
                        "held_actual_w_only_progress_by_request": [
                            actual_by_request[index] if held_mask[index] else None
                            for index in range(request_count)
                        ],
                        "held_request_count": sum(held_mask),
                        "completion": "NEXT_REFRESHED_FIELD_W_ONLY_NLL_REUSE",
                        "added_model_forward_count": 0,
                        "added_backward_count": 0,
                        "added_materialization_count": 0,
                    }
                    realization["identity_sha256"] = canonical_hash(realization)
                    realization_sha = write_once(
                        raw_root
                        / "ode"
                        / arm_label.lower()
                        / f"requestwise-realization-k{int(pending['step_index']) + 1}.json",
                        realization,
                    )
                    request_realization_sha256.append(realization_sha)
                    item["requestwise_realization_sha256"] = realization_sha
                item["identity_sha256"] = canonical_hash(item)
                delayed.append(item)
            problem_receipt = build_scalable_routing_problem(
                field,
                signed,
                accepted_by_layer=accepted_by_layer,
                committed_load_by_layer=history.cumulative_load(),
                lock=controller_lock,
            )
            routing_problem = (
                p1r24_disable_historical(problem_receipt.problem)
                if p1r24
                else problem_receipt.problem
            )
            field_semantic = canonical_hash(
                {
                    "field_sha256": field.identity_sha256,
                    "target_next_sha256": tensor_sha256(target_next),
                    "physical_capture_sha256": physical.identity_sha256,
                    "request_order_sha256": objective_plan.request_order_sha256,
                }
            )
            functional_started = time.perf_counter()
            if not p1r24:
                inventory, functional_probe = _fixed_e8_functional_basis_probe(
                model,
                tokenizer,
                alias=alias,
                field=field,
                step_index=step_index,
                factors=current_factors,
                target_state=target_next,
                capture=entry_capture,
                replay_entry=replay_entry,
                theta0_cache=theta0_cache,
                lock=controller_lock,
                ledger=legacy_ledger,
                touched=touched,
                history=history,
                schedule=schedule,
                factor_state_sha256=_factor_state(
                    entry_capture.entry_sha256, current_factors, target_next
                ),
                field_semantic_sha256=field_semantic,
                trial_entry_weights=base_values,
                physical_materialized=True,
                )
                if inventory.history_item_count != 0:
                    raise ODEBFStateError("P1R23 atomic replay-H inventory is active")
            else:
                inventory = None
                functional_probe = {
                    "status": "NOT_EVALUATED_INNER_P1R24",
                    "model_forward_count": 0,
                    "backward_count": 0,
                    "identity_sha256": canonical_hash(
                        {"step": step_index, "status": "NOT_EVALUATED_INNER_P1R24"}
                    ),
                }
            compute.add_wall("functional_preservation_basis", time.perf_counter() - functional_started)
            route_started = time.perf_counter()
            routing = (
                solve_p1r43_full_strength_routing(
                    routing_problem,
                    arm=arm,
                    alpha_req=alpha_req,
                )
                if p1r43 or p1r51 or p1r52
                else solve_p1r24_matched_routing(
                    routing_problem,
                    arm=arm,
                    rho_write=alpha_req,
                )
                if p1r24
                else
                solve_progress_simplex_routing(
                    problem_receipt.problem,
                    inventory,
                    arm=arm,
                    alpha_req=alpha_req,
                )
                if progress_simplex
                else solve_strength_preserving_routing(
                    problem_receipt.problem,
                    inventory,
                    arm=arm,
                    alpha_req=alpha_req,
                )
            )
            compute.add_wall("routing_solve", time.perf_counter() - route_started)
            if (
                progress_simplex
                and not p1r24
                and routing.status is ProgressSimplexStatus.NO_POSITIVE_DIRECTION
            ):
                raise ODEBFStateError("NO_POSITIVE_DIRECTION")
            if p1r24 and routing.status in (
                P1R24RoutingStatus.NO_POSITIVE_DIRECTION,
                P1R24RoutingStatus.Q_NUMERICAL_DEGENERACY,
            ):
                raise ODEBFStateError(routing.status.value)
            sequential_writer = None
            if writer_policy in SEQUENTIAL_POLICIES:
                if finite_demand is None or finite_endpoint is None:
                    raise ODEBFContractError(
                        "P1R52-FPiQ authoritative finite demand is absent"
                    )
                sequential_started = time.perf_counter()
                sequential_writer = plan_sequential_writer(
                    model,
                    policy=writer_policy,
                    hparams=hparams,
                    projector=projector,
                    covariance_registry=covariance_registry,
                    projector_sha256=projector_sha256,
                    residual_tolerance=controller_lock.residual_tolerance,
                    objective_plan=objective_plan,
                    capture_plan=capture_plan,
                    base_values=base_values,
                    current_factors=current_factors,
                    entry_field=field,
                    entry_applied_slopes=routing_problem.signed_progress,
                    entry_pi=routing.pi,
                    entry_velocity=routing.velocity,
                    target_state=target_next,
                    endpoint_nll=float(finite_endpoint.loss),
                    entry_nll=current_nll,
                    entry_per_request_nll=slope_result.per_request_values,
                    step_index=step_index,
                )
                if (
                    abs(
                        float(sequential_writer.receipt["alpha_star"])
                        - float(alpha_req)
                    )
                    > SIMPLEX_PRIMAL_TOLERANCE
                ):
                    raise ODEBFContractError(
                        "P1R52-FPiQ finite demand authority differs"
                    )
                compute.add_wall(
                    "sequential_writer_prefix",
                    time.perf_counter() - sequential_started,
                )
                sequential_compute = sequential_writer.receipt
                compute.increment(
                    "sequential_writer_prefix_capture",
                    logical_forward_groups=int(
                        sequential_compute["prefix_capture_count"]
                    ),
                    model_forward_calls=int(
                        sequential_compute["prefix_capture_forward_count"]
                    ),
                    physical_microbatch_graphs=int(
                        sequential_compute["prefix_capture_forward_count"]
                    ),
                    processed_tokens=int(
                        sequential_compute["prefix_capture_processed_tokens"]
                    ),
                    padded_tokens=int(
                        sequential_compute["prefix_capture_padded_tokens"]
                    ),
                    capture_forward_calls=int(
                        sequential_compute["prefix_capture_forward_count"]
                    ),
                )
                compute.increment(
                    "sequential_writer_current_slope",
                    logical_forward_groups=int(
                        sequential_compute["additional_slope_group_count"]
                    ),
                    model_forward_calls=int(
                        sequential_compute["additional_slope_forward_count"]
                    ),
                    physical_microbatch_graphs=int(
                        sequential_compute["additional_slope_forward_count"]
                    ),
                    autograd_invocations=int(
                        sequential_compute["additional_slope_backward_count"]
                    ),
                    backward_calls=int(
                        sequential_compute["additional_slope_backward_count"]
                    ),
                    slope_backward_calls=int(
                        sequential_compute["additional_slope_backward_count"]
                    ),
                    processed_tokens=int(
                        sequential_compute["additional_slope_processed_tokens"]
                    ),
                    padded_tokens=int(
                        sequential_compute["additional_slope_padded_tokens"]
                    ),
                )
                _phase_add_objective(
                    compute,
                    "sequential_writer_final_virtual_objective",
                    sequential_writer.final_virtual_objective,
                )
                increment = dict(sequential_writer.increment)
                writer_velocity = tuple(sequential_writer.velocity)
                writer_fields = tuple(sequential_writer.layer_fields)
                writer_slopes = tuple(
                    float(item["applied_slope"])
                    for item in sequential_writer.receipt["layers"]
                )
            else:
                increment = (
                    progress_simplex_waypoint_factors(
                        field, routing.velocity, step_index=step_index
                    )
                    if progress_simplex or p1r24
                    else fixed_e8_waypoint_factors(
                        field, routing.velocity, step_index=step_index
                    )
                )
                writer_velocity = tuple(float(item) for item in routing.velocity)
                writer_fields = tuple(field.layers)
                writer_slopes = tuple(
                    float(item) for item in routing_problem.signed_progress
                )
            candidate_factors = _merge_factors(current_factors, increment)
            predicted = (
                float(sequential_writer.predicted_progress)
                if sequential_writer is not None
                else float(
                    routing_problem.signed_progress
                    @ np.asarray(routing.velocity, dtype=np.float64)
                )
            )
            refresh.record(
                step_index=step_index,
                accepted_state_sha256=state_before,
                target_sha256=tensor_sha256(target_next),
                key_inventory_sha256=_key_identity(physical.keys_by_layer),
                slope_sha256=canonical_hash(list(signed.signed_progress)),
                field_sha256=field.identity_sha256,
                field_invocation_index=step_index + 1,
            )
            materialize_started = time.perf_counter()
            materialization = materializer.materialize(
                candidate_factors, transition_index=step_index + 1
            )
            sequential_bf16_identity = (
                post_commit_identity(sequential_writer, materialization)
                if sequential_writer is not None
                else None
            )
            compute.add_wall("physical_materialization", time.perf_counter() - materialize_started)
            compute.increment("physical_materialization", materialization_count=1)
            next_capture_started = time.perf_counter()
            next_physical = capture_scalable_physical_state(
                model, capture_plan, hparams
            )
            compute.add_wall("accepted_state_refresh", time.perf_counter() - next_capture_started)
            _phase_add_capture(compute, "accepted_state_refresh", next_physical)
            progress: dict[str, Any] = {
                "predicted": predicted,
                "actual": None,
                "completion": "DELAYED_TO_NEXT_REFRESHED_FIELD",
                "alpha_req": routing.alpha_req,
                "alpha_max": routing.alpha_max,
                "alpha_apply": (
                    predicted if sequential_writer is not None else routing.alpha_apply
                ),
                "coverage": (
                    1.0
                    if routing.alpha_req == 0.0
                    else predicted / routing.alpha_req
                )
                if sequential_writer is not None
                else routing.coverage,
                "equality_residual": (
                    abs(predicted - routing.alpha_req)
                    if sequential_writer is not None
                    else routing.equality_residual
                ),
                "entry_router_alpha_apply": routing.alpha_apply,
                "candidate_objective_inner_count": 0,
            }
            if step_index == P1R23_GRID_COUNT - 1:
                terminal_started = time.perf_counter()
                terminal_objective = (
                    sequential_writer.final_virtual_objective
                    if sequential_writer is not None
                    else evaluate_scalable_target_new_objective(
                        model, objective_plan
                    )
                )
                terminal_objective_count += 1
                compute.add_wall("terminal_objective", time.perf_counter() - terminal_started)
                if sequential_writer is None:
                    _phase_add_objective(compute, "terminal_objective", terminal_objective)
                actual = current_nll - float(terminal_objective.loss)
                progress.update(
                    {
                        "actual": actual,
                        "terminal_mean_target_new_nll": float(terminal_objective.loss),
                        "completion": "TERMINAL_W_ONLY_OBJECTIVE_ONCE",
                        "realization_ratio": actual
                        / (
                            float(progress["alpha_apply"])
                            + STRENGTH_COVERAGE_EPSILON
                        ),
                        "linearization_error": actual - predicted,
                    }
                )
                if p1r42 or p1r43 or p1r51 or p1r52:
                    source_values = tuple(
                        float(value) for value in slope_result.per_request_values
                    )
                    next_values = tuple(
                        float(value)
                        for value in terminal_objective.per_request_values
                    )
                    held_mask = tuple(
                        bool(value)
                        for value in target_receipt["semantic_held_mask"]
                    )
                    if not (
                        len(source_values)
                        == len(next_values)
                        == len(held_mask)
                        == request_count
                    ):
                        raise ODEBFStateError(
                            "terminal requestwise realization geometry differs"
                        )
                    actual_by_request = tuple(
                        source - next_value
                        for source, next_value in zip(
                            source_values, next_values, strict=True
                        )
                    )
                    realization = {
                        "schema": (
                            "ode-edit-s05-p1r52-rsa-r42safekdc-requestwise-w-only-realization/v1"
                            if p1r52
                            else "ode-edit-s05-p1r51-rsa-a1-requestwise-w-only-realization/v1"
                            if p1r51
                            else "ode-edit-s05-p1r43-requestwise-w-only-realization/v1"
                            if p1r43
                            else "ode-edit-s05-p1r42-requestwise-w-only-realization/v1"
                        ),
                        "instruction_id": (
                            P1R52_INSTRUCTION_ID
                            if p1r52
                            else P1R51_INSTRUCTION_ID
                            if p1r51
                            else P1R43_INSTRUCTION_ID
                            if p1r43
                            else P1R42_INSTRUCTION_ID
                        ),
                        "method_id": (
                            P1R52_METHOD_ID
                            if p1r52
                            else P1R51_METHOD_ID
                            if p1r51
                            else P1R43_METHOD_ID
                            if p1r43
                            else P1R42_METHOD_ID
                        ),
                        "transition_index": P1R23_GRID_COUNT,
                        "source_w_only_target_new_nll_by_request": list(
                            source_values
                        ),
                        "next_w_only_target_new_nll_by_request": list(next_values),
                        "actual_w_only_progress_by_request": list(actual_by_request),
                        "semantic_held_mask": list(held_mask),
                        "held_actual_w_only_progress_by_request": [
                            actual_by_request[index] if held_mask[index] else None
                            for index in range(request_count)
                        ],
                        "held_request_count": sum(held_mask),
                        "completion": "TERMINAL_W_ONLY_OBJECTIVE_ONCE",
                        "added_model_forward_count": 0,
                        "added_backward_count": 0,
                        "added_materialization_count": 0,
                    }
                    realization["identity_sha256"] = canonical_hash(realization)
                    realization_sha = write_once(
                        raw_root
                        / "ode"
                        / arm_label.lower()
                        / f"requestwise-realization-k{P1R23_GRID_COUNT}.json",
                        realization,
                    )
                    request_realization_sha256.append(realization_sha)
                    progress["requestwise_realization_sha256"] = realization_sha
            target_realization = fixed_e8_target_write_realization(
                current_target,
                target_next,
                current_terminal,
                next_physical.terminal_z,
            )
            contribution = [
                float(a * v)
                for a, v in zip(
                    writer_slopes,
                    writer_velocity,
                    strict=True,
                )
            ]
            factor_list_hashes = {
                layer: canonical_hash(
                    [
                        {
                            "weight_name": item.factor.weight_name,
                            "order_key": list(item.factor.order_key),
                            "theta": item.factor.theta,
                            "left_sha256": tensor_sha256(item.factor.left),
                            "right_sha256": tensor_sha256(item.factor.right),
                        }
                        for item in accepted_by_layer[layer]
                    ]
                )
                for layer in P1R23_LAYER_ORDER
            }
            cumulative_p = (
                p1r24_cumulative_p_receipt(
                    routing_problem,
                    writer_velocity,
                    step_index=step_index,
                    factor_list_hashes=factor_list_hashes,
                )
                if p1r24
                else None
            )
            payload = {
                "schema": f"{P1R23_SCHEMA}-accepted-transition/v1",
                "arm": arm_label,
                "accepted_index": step_index + 1,
                "tau_before": step_index * P1R23_H,
                "tau_after": (step_index + 1) * P1R23_H,
                "physical_capture_sha256": physical.identity_sha256,
                "next_physical_capture_sha256": next_physical.identity_sha256,
                "target_objective": target_result.raw_free_payload(),
                "target_update": target_receipt,
                "finite_writer_demand": (
                    finite_demand.receipt if finite_demand is not None else None
                ),
                "field_sha256": field.identity_sha256,
                "physical_slope": asdict(signed),
                "routing_problem_sha256": routing_problem.identity(),
                "routing": routing.raw_free_payload(),
                "entry_routing_role": "FROZEN_ENTRY_ALLOCATION",
                "sequential_writer": (
                    sequential_writer.receipt
                    if sequential_writer is not None
                    else None
                ),
                "sequential_virtual_physical_identity": sequential_bf16_identity,
                "functional_basis": (
                    inventory.raw_free_payload()
                    if inventory is not None
                    else {
                        "status": "NOT_EVALUATED_INNER_P1R24",
                        "functional_p_layer_basis_count": 0,
                        "functional_h_count": 0,
                    }
                ),
                "functional_probe_sha256": functional_probe["identity_sha256"],
                "progress": progress,
                "per_layer_applied_progress": contribution,
                "structural_h": 0.0 if p1r24 else routing_problem.historical.value(
                    np.asarray(writer_velocity)
                ),
                "structural_p": routing_problem.pretrained.value(
                    np.asarray(writer_velocity)
                ),
                "cumulative_atomic_structural_p": cumulative_p,
                "target_write_realization": target_realization,
                "materialization": materialization,
                "authoritative_slope": "PHYSICAL_W_ONLY_NOHOOK",
                "overlay_forward_backward_count": 0,
                "inner_step_heldout_evaluation_count": 0,
                "retry_backtracking_reject_count": 0,
                "routing_method": (
                    P1R52_FPIQ_METHOD_ID
                    if writer_policy is not None
                    else P1R52_METHOD_ID
                    if p1r52
                    else P1R51_METHOD_ID
                    if p1r51
                    else P1R43_METHOD_ID
                    if p1r43
                    else P1R42_METHOD_ID
                    if p1r42
                    else P1R39_METHOD_ID
                    if p1r39
                    else P1R38_METHOD_ID
                    if p1r38
                    else P1R35_METHOD_ID
                    if p1r35
                    else P1R34_METHOD_ID
                    if p1r34
                    else P1R24_METHOD_ID
                    if p1r24
                    else
                    PROGRESS_SIMPLEX_METHOD_ID
                    if progress_simplex
                    else "P1R23_STRENGTH_PRESERVING"
                ),
                "progress_simplex_decision_influence_count": (
                    1 if progress_simplex or p1r24 else 0
                ),
                "persistent_historical_ledger_count": 0,
                "historical_h_decision_influence_count": 0,
                "per_request_target_controller": (
                    P1R52_METHOD_ID
                    if p1r52
                    else P1R51_METHOD_ID
                    if p1r51
                    else P1R43_METHOD_ID
                    if p1r43
                    else P1R42_METHOD_ID
                    if p1r42
                    else P1R39_METHOD_ID
                    if p1r39
                    else P1R38_METHOD_ID
                    if p1r38
                    else None
                ),
            }
            payload["identity_sha256"] = canonical_hash(payload)
            write_once(
                raw_root
                / "ode"
                / arm_label.lower()
                / f"accepted-k{step_index + 1}.json",
                payload,
            )
            if p1r24:
                for layer in writer_fields:
                    accepted_by_layer[layer.layer].append(
                        AcceptedLayerContribution.from_field(
                            layer,
                            increment[layer.weight_name],
                            history_action=torch.zeros_like(layer.history_action),
                        )
                    )
            elif any(accepted_by_layer[layer] for layer in P1R23_LAYER_ORDER):
                raise ODEBFStateError("P1R23 atomic H state is not empty")
            accepted.append(payload)
            pending = (
                None
                if step_index == P1R23_GRID_COUNT - 1
                else {
                    "step_index": step_index,
                    "source_nll": current_nll,
                    "predicted": predicted,
                    "alpha_apply": float(progress["alpha_apply"]),
                    **(
                        {
                            "source_per_request_nll": list(
                                slope_result.per_request_values
                            ),
                            "semantic_held_mask": list(
                                target_receipt["semantic_held_mask"]
                            ),
                        }
                        if p1r42 or p1r43 or p1r51 or p1r52
                        else {}
                    ),
                }
            )
            current_factors = _factor_map(candidate_factors)
            current_target = target_next
            current_terminal = next_physical.terminal_z.clone()
            physical = next_physical
            legacy_ledger.record_accepted_step(
                accepted_dt=P1R23_H, completed_k_total=step_index + 1
            )
        if pending is not None or len(accepted) != 8 or len(delayed) != 7:
            raise ODEBFStateError("P1R23 K8 delayed accounting differs")
        if (p1r42 or p1r43 or p1r51 or p1r52) and len(request_realization_sha256) != P1R23_GRID_COUNT:
            raise ODEBFStateError("requestwise realization ledger differs")
        if not (p1r42 or p1r43 or p1r51 or p1r52) and request_realization_sha256:
            raise ODEBFStateError("unexpected requestwise realization ledger is active")
        observed_sequential_slope_groups = sum(
            int(item["sequential_writer"]["additional_slope_group_count"])
            for item in accepted
            if item["sequential_writer"] is not None
        )
        expected_sequential_slope_groups = (
            32 if writer_policy in SEQUENTIAL_POLICIES else 0
        )
        if observed_sequential_slope_groups != expected_sequential_slope_groups:
            raise ODEBFStateError(
                "P1R52-FPiQ K8 additional slope group accounting differs"
            )
        terminal_replay = _controller_replay_entry(
            model,
            tokenizer,
            alias=alias,
            arm_state=arm_state,
            sample_waypoint=8,
            factors={},
            request_by_sha256=request_by_sha256,
            population_by_sha256=population_by_sha256,
            schedule=schedule,
            outer_entry_p_cache=outer_entry_p_cache,
        )
        terminal_value = _functional_trial(
            model,
            tokenizer,
            alias=alias,
            entry=terminal_replay,
            theta0_cache=theta0_cache,
            factors={},
            lock=controller_lock,
            ledger=legacy_ledger,
        )
        terminal_functional = _terminal_functional_payload(terminal_value)
        terminal_oracle: dict[str, Any] | None = None
        if p1r24:
            oracle_variable = (
                current_target.detach()
                .to(device=next(model.parameters()).device, dtype=torch.float32)
                .clone()
                .requires_grad_(True)
            )
            oracle_result = evaluate_scalable_target_new_objective(
                model,
                objective_plan,
                target_state=oracle_variable,
                current_terminal=current_terminal,
                target_layer_name=hparams.layer_module_tmp.format(int(hparams.layers[-1])),
            )
            _phase_add_objective(compute, "terminal_z8_oracle", oracle_result, target=True)
            terminal_oracle = {
                "target_new_nll": oracle_result.loss,
                "per_request_values": list(oracle_result.per_request_values),
                "receipt_sha256": oracle_result.identity_sha256,
                "model_forward_count": oracle_result.model_forward_count,
                "backward_count": oracle_result.backward_count,
            }
        factors_for_endpoint = _factor_map(current_factors)
        target_for_endpoint = current_target.clone()
        physical_for_endpoint = physical
        p1r52_teacher_hashes = (
            [
                str(item["target_update"]["kl_teacher_input_sha256"])
                for item in accepted
            ]
            if p1r52
            else []
        )
        if p1r52 and (
            len(p1r52_teacher_hashes) != P1R23_GRID_COUNT
            or len(set(p1r52_teacher_hashes)) != 1
            or p1r52_teacher_hashes[0] != p1r24_kl_teacher_sha256
        ):
            raise ODEBFContractError("P1R52 immutable KL teacher hash differs across K8")
        rollout_payload = {
            "arm": arm_label,
            "status": (
                f"P1R52_FPIQ_{writer_policy.value}_K8_COMPLETE"
                if writer_policy is not None
                else "P1R52_RSA_R42SAFEKDC_M1_K8_COMPLETE"
                if p1r52
                else "P1R51_RSA_A1_K8_COMPLETE"
                if p1r51
                else "P1R43_RHO_FREE_SEMANTIC_FIRST_K8_COMPLETE"
                if p1r43
                else "P1R42_OBJECTIVE_ALIGNED_P1R35_K8_COMPLETE"
                if p1r42
                else "P1R39_NORMALIZED_GRADIENT_P1R35_K8_COMPLETE"
                if p1r39
                else "P1R38_PR_P1R35_K8_COMPLETE"
                if p1r38
                else "P1R35_FULL_CURRENT_RESIDUAL_K8_COMPLETE"
                if p1r35
                else "P1R34_W_ANCHORED_FINITE_DEMAND_K8_COMPLETE"
                if p1r34
                else "P1R24_ATOMIC_STRENGTH_RECOVERY_K8_COMPLETE"
                if p1r24
                else "PROGRESS_SIMPLEX_DYNAMIC_K8_COMPLETE"
                if progress_simplex
                else "SCALABLE_DYNAMIC_K8_COMPLETE"
            ),
            "request_count": request_count,
            "accepted_update_count": len(accepted),
            "tau_final": 1.0,
            "initial": initial.raw_free_payload(),
            "metric": metric.raw_free_payload(),
            "accepted_receipt_sha256": [
                item["identity_sha256"] for item in accepted
            ],
            "delayed_progress": delayed,
            "terminal_target_sha256": tensor_sha256(target_for_endpoint),
            "terminal_physical_capture_sha256": physical_for_endpoint.identity_sha256,
            "terminal_functional": terminal_functional,
            "terminal_z8_oracle": terminal_oracle,
            "kl_teacher_input_sha256": p1r24_kl_teacher_sha256 if p1r52 else None,
            "kl_teacher_hash_by_k": p1r52_teacher_hashes,
            "kl_teacher_hash_k8_constant": bool(
                p1r52 and len(set(p1r52_teacher_hashes)) == 1
            ),
            "requestwise_realization_sha256": (
                request_realization_sha256 if p1r42 or p1r43 or p1r51 or p1r52 else []
            ),
            "terminal_cumulative_structural_p": (
                accepted[-1]["cumulative_atomic_structural_p"]
                if p1r24 and accepted
                else None
            ),
            "dynamic_refresh": refresh.finalize(),
            "compute": compute.raw_free_payload(),
            "legacy_compute": legacy_ledger.raw_free_payload(),
            "terminal_objective_count": terminal_objective_count,
            "writer_policy": (
                writer_policy.value if writer_policy is not None else None
            ),
            "sequential_writer_additional_slope_group_count": sum(
                int(item["sequential_writer"]["additional_slope_group_count"])
                for item in accepted
                if item["sequential_writer"] is not None
            ),
            "sequential_writer_expected_additional_slope_group_count": (
                32 if writer_policy in SEQUENTIAL_POLICIES else 0
            ),
            "edit_core_wall_seconds": time.perf_counter() - started,
            "materializer": materializer.raw_free_payload(),
            "initial_w0_sha256": initial_w0,
            "alphaedit_target_geometry": p1r24_alpha_geometry,
            "instruction_id": (
                P1R52_FPIQ_INSTRUCTION_ID
                if writer_policy is not None
                else P1R52_INSTRUCTION_ID
                if p1r52
                else P1R51_INSTRUCTION_ID
                if p1r51
                else P1R43_INSTRUCTION_ID
                if p1r43
                else P1R42_INSTRUCTION_ID
                if p1r42
                else P1R35_INSTRUCTION_ID
                if p1r35
                else P1R34_INSTRUCTION_ID
                if p1r34
                else P1R24_INSTRUCTION_ID
                if p1r24
                else
                PROGRESS_SIMPLEX_INSTRUCTION_ID
                if progress_simplex
                else P1R23_INSTRUCTION_ID
            ),
            "method_id": (
                P1R52_FPIQ_METHOD_ID
                if writer_policy is not None
                else P1R52_METHOD_ID
                if p1r52
                else P1R51_METHOD_ID
                if p1r51
                else P1R43_METHOD_ID
                if p1r43
                else P1R42_METHOD_ID
                if p1r42
                else P1R35_METHOD_ID
                if p1r35
                else P1R34_METHOD_ID
                if p1r34
                else P1R24_METHOD_ID
                if p1r24
                else
                PROGRESS_SIMPLEX_METHOD_ID
                if progress_simplex
                else P1R23_METHOD_ID
            ),
        }
        rollout_payload["identity_sha256"] = canonical_hash(rollout_payload)
        return {
            "public": rollout_payload,
            "terminal_factors": factors_for_endpoint,
            "terminal_target": target_for_endpoint,
            "terminal_physical": physical_for_endpoint,
        }
    finally:
        counter.close()
        restore = materializer.restore()
        restored = True
        if _model_w0_contract(touched) != initial_w0:
            raise ODEBFStateError("P1R23 ODE arm did not restore W0")
        if not restored:
            materializer.restore()


def _evaluate_frozen_state(
    model: torch.nn.Module,
    tokenizer: Any,
    cases: Sequence[CounterFactEvaluationCase],
    *,
    alias: str,
    freeze_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    receipt = evaluate_scalable_primary(
        model,
        tokenizer,
        cases,
        model_alias=alias,
        freeze_payload=freeze_payload,
    )
    elapsed = time.perf_counter() - started
    return {
        "receipt": receipt.raw_free_payload(),
        "added_90": added_ninety_payload(receipt),
        "wall_seconds": elapsed,
    }, elapsed


def _action_frozen_cases(
    dataset_path: Path,
    requests: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    selected_snapshot_sha256: str,
    fixed_budget_slots_completed: int,
) -> tuple[tuple[CounterFactEvaluationCase, ...], dict[str, Any]]:
    return load_scalable_cases_after_freeze(
        dataset_path,
        requests,
        arm=arm,
        selected_snapshot_sha256=selected_snapshot_sha256,
        fixed_budget_slots_completed=fixed_budget_slots_completed,
    )


def _paired_initial_semantic_gate(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    left_metric: Mapping[str, Any],
    right_metric: Mapping[str, Any],
    *,
    request_order_sha256: str,
    objective_plan_sha256: str,
    capture_plan_sha256: str,
    allocation: str,
) -> dict[str, Any]:
    if allocation not in ("RS", "BG"):
        raise ODEBFContractError("P1R23 paired allocation schema differs")
    initial_schema = "ode-edit-s05-p1r23-initial-target/v1"
    if any(
        item.get("schema") != initial_schema
        or item.get("residual_exact_zero") is not True
        for item in (left, right)
    ):
        raise ODEBFContractError("P1R23 paired initial coordinate differs")
    metric_schema = (
        "ode-edit-s05-p1r23-robust-shared-scale/v1"
        if allocation == "RS"
        else "ode-edit-s05-p1r23-batch-global-scale/v1"
    )
    expected_scale = (
        "ROBUST_SHARED_REQUEST_SCALE_V1" if allocation == "RS" else None
    )
    for metric in (left_metric, right_metric):
        if (
            metric.get("schema") != metric_schema
            or metric.get("request_order_sha256") != request_order_sha256
            or metric.get("shared_speed_definition") != "median_i_l2_norm_z0_i"
            or (allocation == "RS" and metric.get("scale") != expected_scale)
        ):
            raise ODEBFContractError("P1R23 paired allocation schema differs")
    identities = (
        request_order_sha256,
        objective_plan_sha256,
        capture_plan_sha256,
    )
    if any(not isinstance(item, str) or len(item) != 64 for item in identities):
        raise ODEBFContractError("P1R23 paired operator identity differs")
    payload = {
        "schema": f"{P1R23_SCHEMA}-paired-initial-semantic-gate/v1",
        "request_order_sha256": request_order_sha256,
        "objective_plan_sha256": objective_plan_sha256,
        "capture_plan_sha256": capture_plan_sha256,
        "target_allocation": allocation,
        "allocation_schema": metric_schema,
        "initial_operator_schema": initial_schema,
        "left_r0_exact_zero": True,
        "right_r0_exact_zero": True,
        "left_finite_validated_by_operator": True,
        "right_finite_validated_by_operator": True,
        "exact_value_hash_equality": left == right,
        "exact_metric_hash_equality": left_metric == right_metric,
        "exact_value_hash_equality_decision_influence_count": 0,
        "coordinate_order_operator_schema_gate_influence_count": 1,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _run_ode_pair(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    destination: Path,
    raw_root: Path,
    source_head: str,
    requests: Sequence[Mapping[str, Any]],
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
    job_ledger: ComputeLedger,
    write_once: Any,
    request_microbatch_size: int,
    allocation: str,
    progress_simplex: bool = False,
    p1r24: bool = False,
    p1r34: bool = False,
    p1r35: bool = False,
    neutral_only: bool = False,
    technical_smoke: bool = False,
) -> dict[str, Any]:
    from .p1_runtime import ArmRuntimeState, _entry_parameter_snapshot_sha256

    request_order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    effective_microbatch_size = min(request_microbatch_size, len(requests))
    objective_plan = build_scalable_objective_plan(
        model,
        tokenizer,
        requests,
        contexts=contexts,
        request_microbatch_size=effective_microbatch_size,
        fact_token_strategy=hparams.fact_token,
    )
    capture_plan = build_scalable_capture_plan(
        tokenizer,
        requests,
        contexts=contexts,
        request_microbatch_size=effective_microbatch_size,
        fact_token_strategy=hparams.fact_token,
    )
    write_once(raw_root / "plans" / "objective.json", objective_plan.raw_free_payload())
    write_once(raw_root / "plans" / "capture.json", capture_plan.raw_free_payload())
    outer_population = tuple(
        population_by_sha256[item] for item in theta0_cache.request_order
    )
    outer_snapshot = _entry_parameter_snapshot_sha256(
        model, dict(base_receipt.parameter_sha256)
    )
    counter = ModelForwardCounter(model, job_ledger)
    try:
        outer_cache = build_outer_entry_pretrained_cache(
            model,
            tokenizer,
            outer_population,
            theta0_cache,
            outer_entry_snapshot_sha256=outer_snapshot,
        )
    finally:
        counter.close()
    w0_contract = _model_w0_contract(touched)
    routing_arms = (
        (
            f"{allocation}-P1R35-FULL-CURRENT-RESIDUAL-NEUTRAL",
            f"{allocation}-P1R35-FULL-CURRENT-RESIDUAL-SOFT",
        )
        if p1r35
        else (
            f"{allocation}-P1R34-W-FINITE-DEMAND-NEUTRAL",
            f"{allocation}-P1R34-W-FINITE-DEMAND-SOFT",
        )
        if p1r34
        else (
            f"{allocation}-P1R24-NEUTRAL",
            f"{allocation}-P1R24-SOFT",
        )
        if p1r24
        else
        (
            P1R23_SIMPLEX_BG_ROUTING_ARMS
            if allocation == "BG"
            else P1R23_SIMPLEX_RS_ROUTING_ARMS
        )
        if progress_simplex
        else (
            P1R23_BG_ROUTING_ARMS
            if allocation == "BG"
            else P1R23_RS_ROUTING_ARMS
        )
    )
    if allocation not in ("RS", "BG"):
        raise ODEBFContractError("P1R23 paired target allocation differs")
    rollouts: dict[str, dict[str, Any]] = {}
    selected_arms = (
        (FixedE8Arm.NEUTRAL,)
        if neutral_only
        else (FixedE8Arm.NEUTRAL, FixedE8Arm.SOFT)
    )
    selected_labels = routing_arms[: len(selected_arms)]
    for selected, label in zip(
        selected_arms, selected_labels, strict=True
    ):
        if _model_w0_contract(touched) != w0_contract:
            raise ODEBFStateError("P1R23 paired arm W0 differs")
        arm_state = ArmRuntimeState(
            P1Arm.R_BF,
            P1HistoryLedger(layer_order=P1R23_LAYER_ORDER, maximum_records=40),
            ComputeLedger(),
            ArmWeightSnapshot(
                P1Arm.R_BF,
                0,
                base_receipt.parameter_sha256,
                canonical_hash({"arm": label, "weights": base_receipt.parameter_sha256}),
            ),
            dict(base_values),
        )
        rollouts[label] = _run_ode_arm(
            model,
            tokenizer,
            requests,
            alias=alias,
            arm=selected,
            allocation=allocation,
            capture_plan=capture_plan,
            objective_plan=objective_plan,
            hparams=hparams,
            projector=projector,
            contexts=contexts,
            covariance_registry=covariance_registry,
            projector_sha256=projector_sha256,
            controller_lock=controller_lock,
            arm_state=arm_state,
            request_by_sha256=request_by_sha256,
            population_by_sha256=population_by_sha256,
            schedule=schedule,
            outer_entry_p_cache=outer_cache,
            theta0_cache=theta0_cache,
            touched=touched,
            base_receipt=base_receipt,
            base_values=base_values,
            raw_root=raw_root,
            write_once=write_once,
            progress_simplex=progress_simplex,
            p1r24=p1r24,
            p1r34=p1r34,
            p1r35=p1r35,
        )
    left = rollouts[selected_labels[0]]["public"]["initial"]
    left_metric = rollouts[selected_labels[0]]["public"]["metric"]
    paired_initial = (
        {
            "schema": f"{P1R23_SCHEMA}-single-initial-semantic-gate/v1",
            "request_order_sha256": request_order,
            "objective_plan_sha256": objective_plan.identity_sha256,
            "capture_plan_sha256": capture_plan.identity_sha256,
            "target_allocation": allocation,
            "initial": left,
            "metric": left_metric,
            "neutral_only_smoke": True,
        }
        if neutral_only
        else _paired_initial_semantic_gate(
            left,
            rollouts[selected_labels[1]]["public"]["initial"],
            left_metric,
            rollouts[selected_labels[1]]["public"]["metric"],
            request_order_sha256=request_order,
            objective_plan_sha256=objective_plan.identity_sha256,
            capture_plan_sha256=capture_plan.identity_sha256,
            allocation=allocation,
        )
    )
    if "identity_sha256" not in paired_initial:
        paired_initial["identity_sha256"] = canonical_hash(paired_initial)
    paired_initial_sha = write_once(
        raw_root / "paired-initial-semantic-gate.json", paired_initial
    )
    action_freeze = {
        "schema": f"{P1R23_SCHEMA}-paired-action-freeze/v1",
        "instruction_id": (
            P1R35_INSTRUCTION_ID
            if p1r35
            else P1R34_INSTRUCTION_ID
            if p1r34
            else P1R24_INSTRUCTION_ID
            if p1r24
            else
            PROGRESS_SIMPLEX_INSTRUCTION_ID
            if progress_simplex
            else P1R23_INSTRUCTION_ID
        ),
        "method_id": (
            P1R35_METHOD_ID
            if p1r35
            else P1R34_METHOD_ID
            if p1r34
            else P1R24_METHOD_ID
            if p1r24
            else
            PROGRESS_SIMPLEX_METHOD_ID
            if progress_simplex
            else P1R23_METHOD_ID
        ),
        "request_order_sha256": request_order,
        "target_allocation": allocation,
        "paired_initial_semantic_gate_sha256": paired_initial_sha,
        "rollout_sha256": {
            label: rollouts[label]["public"]["identity_sha256"]
            for label in selected_labels
        },
        "actions_frozen_before_heldout": True,
        "inner_step_heldout_access_count": 0,
        "atomic_joint_batch": True,
        "persistent_history_append_count": 0,
        "replay_h_decision_influence_count": 0,
        "sequential_controller_influence_count": 0,
    }
    action_freeze["identity_sha256"] = canonical_hash(action_freeze)
    action_sha = write_once(raw_root / "action-freeze.json", action_freeze)
    if neutral_only or technical_smoke:
        terminal = {
            "schema": (
                "ode-edit-s05-p1r35-b1-smoke-terminal/v1"
                if p1r35
                else "ode-edit-s05-p1r34-b1-smoke-terminal/v1"
                if p1r34
                else "ode-edit-s05-p1r24-b1-smoke-terminal/v1"
            ),
            "instruction_id": P1R35_INSTRUCTION_ID if p1r35 else P1R34_INSTRUCTION_ID if p1r34 else P1R24_INSTRUCTION_ID,
            "method_id": P1R35_METHOD_ID if p1r35 else P1R34_METHOD_ID if p1r34 else P1R24_METHOD_ID,
            "source_head": source_head,
            "alias": alias,
            "request_count": 1,
            "request_order_sha256": request_order,
            "target_allocation": allocation,
            "routing_arms": list(selected_labels),
            "rollouts": {label: rollouts[label]["public"] for label in selected_labels},
            "action_freeze_sha256": action_sha,
            "scientific_score_gate_influence_count": 0,
            "heldout_evaluator_access_count": 0,
            "persistent_history_append_count": 0,
            "W0_restored": _model_w0_contract(touched) == w0_contract,
            "status": (
                "P1R35_B1_TECHNICAL_SMOKE_COMPLETE"
                if p1r35
                else "P1R34_B1_TECHNICAL_SMOKE_COMPLETE"
                if p1r34
                else "P1R24_B1_TECHNICAL_SMOKE_COMPLETE"
            ),
        }
        terminal["identity_sha256"] = canonical_hash(terminal)
        terminal_sha = write_once(destination / "terminal.json", terminal)
        manifest = {
            "schema": (
                "ode-edit-s05-p1r35-b1-smoke-manifest/v1"
                if p1r35
                else "ode-edit-s05-p1r34-b1-smoke-manifest/v1"
                if p1r34
                else "ode-edit-s05-p1r24-b1-smoke-manifest/v1"
            ),
            "source_head": source_head,
            "terminal_sha256": terminal_sha,
            "action_freeze_sha256": action_sha,
            "W0_restored": terminal["W0_restored"],
            "status": terminal["status"],
        }
        manifest["identity_sha256"] = canonical_hash(manifest)
        manifest_sha = write_once(destination / "manifest.json", manifest)
        return {
            "status": terminal["status"],
            "terminal_sha256": terminal_sha,
            "manifest_sha256": manifest_sha,
        }
    cases, freeze = _action_frozen_cases(
        dataset_path,
        requests,
        arm=(
            f"P1R35-{allocation}-FULL-CURRENT-RESIDUAL-PAIR"
            if p1r35
            else f"P1R34-{allocation}-W-ANCHORED-FINITE-DEMAND-PAIR"
            if p1r34
            else f"P1R24-{allocation}-ATOMIC-STRENGTH-RECOVERY-PAIR"
            if p1r24
            else f"P1R23-{allocation}-PROGRESS-SIMPLEX-PAIR"
            if progress_simplex
            else f"P1R23-{allocation}-ODE-PAIR"
        ),
        selected_snapshot_sha256=action_sha,
        fixed_budget_slots_completed=8,
    )
    endpoints: dict[str, Any] = {}
    w0, _ = _evaluate_frozen_state(
        model, tokenizer, cases, alias=alias, freeze_payload=freeze
    )
    w0_sha = write_once(raw_root / "W0-endpoint.json", w0)
    for label in selected_labels:
        endpoint_materializer = AcceptedPhysicalStateMaterializer(model, base_values)
        materialization = endpoint_materializer.materialize(
            rollouts[label]["terminal_factors"], transition_index=8
        )
        endpoint, _ = _evaluate_frozen_state(
            model, tokenizer, cases, alias=alias, freeze_payload=freeze
        )
        restore = endpoint_materializer.restore()
        endpoint.update(
            {
                "endpoint_materialization": materialization,
                "restore": restore,
                "rollout_sha256": rollouts[label]["public"]["identity_sha256"],
            }
        )
        endpoint["identity_sha256"] = canonical_hash(endpoint)
        endpoints[label] = endpoint
        write_once(raw_root / "endpoints" / f"{label.lower()}.json", endpoint)
    terminal = {
        "schema": f"{P1R23_SCHEMA}-ode-pair-terminal/v1",
        "instruction_id": (
            P1R35_INSTRUCTION_ID
            if p1r35
            else P1R34_INSTRUCTION_ID
            if p1r34
            else P1R24_INSTRUCTION_ID
            if p1r24
            else
            PROGRESS_SIMPLEX_INSTRUCTION_ID
            if progress_simplex
            else P1R23_INSTRUCTION_ID
        ),
        "method_id": (
            P1R35_METHOD_ID
            if p1r35
            else P1R34_METHOD_ID
            if p1r34
            else P1R24_METHOD_ID
            if p1r24
            else
            PROGRESS_SIMPLEX_METHOD_ID
            if progress_simplex
            else P1R23_METHOD_ID
        ),
        "source_head": source_head,
        "alias": alias,
        "request_count": len(requests),
        "request_order_sha256": request_order,
        "request_microbatch_size": effective_microbatch_size,
        "target_allocation": allocation,
        "routing_arms": list(selected_labels),
        "objective_plan_sha256": objective_plan.identity_sha256,
        "capture_plan_sha256": capture_plan.identity_sha256,
        "rollouts": {label: rollouts[label]["public"] for label in selected_labels},
        "W0_endpoint_sha256": w0_sha,
        "W0_shared_by_neutral_soft": True,
        "endpoints": endpoints,
        "action_freeze_sha256": action_sha,
        "paired_initial_semantic_gate": paired_initial,
        "official_endpoint_evaluation_count_per_state": 1,
        "scientific_invalid_count": 0,
        "estimand": "ATOMIC",
        "persistent_commit_count": 0,
        "persistent_history_append_count": 0,
        "replay_h_decision_influence_count": 0,
        "sequential_round_count": 0,
        "p1r20_access_count": 0,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": f"{P1R23_SCHEMA}-manifest/v1",
        "source_head": source_head,
        "terminal_sha256": terminal_sha,
        "request_count": len(requests),
        "role": (
            "P1R35_B10_RS_PAIR"
            if p1r35
            else "P1R34_B10_RS_PAIR"
            if p1r34
            else (
                "PROGRESS_SIMPLEX_BG_PAIR"
                if allocation == "BG"
                else "PROGRESS_SIMPLEX_RS_PAIR"
            )
            if progress_simplex
            else (
                "ODE_BF_K8_PAIR"
                if allocation == "BG"
                else "ODE_BF_K8_RS_PAIR"
            )
        ),
        "W0_restored": _model_w0_contract(touched) == w0_contract,
        "estimand": "ATOMIC",
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = write_once(destination / "manifest.json", manifest)
    return {
        "status": (
            "P1R35_FULL_CURRENT_RESIDUAL_PAIR_COMPLETE"
            if p1r35
            else "P1R34_W_ANCHORED_FINITE_DEMAND_PAIR_COMPLETE"
            if p1r34
            else "P1R24_ATOMIC_STRENGTH_RECOVERY_PAIR_COMPLETE"
            if p1r24
            else "P1R23_PROGRESS_SIMPLEX_PAIR_COMPLETE"
            if progress_simplex
            else "P1R23_ODE_PAIR_COMPLETE"
        ),
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
    }


def _run_native_role(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    role: str,
    destination: Path,
    raw_root: Path,
    source_head: str,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    dataset_path: Path,
    touched: Mapping[str, torch.nn.Parameter],
    base_values: Mapping[str, torch.Tensor],
    request_microbatch_size: int,
    mutation_lock: threading.RLock,
    job_ledger: ComputeLedger,
    write_once: Any,
) -> dict[str, Any]:
    entry = {name: value.detach().clone() for name, value in touched.items()}
    w0_contract = _model_w0_contract(touched)
    counter = ModelForwardCounter(model, job_ledger)
    try:
        if role == "OPTIMIZED_NATIVE_K1":
            capture = capture_optimized_native_k1(
                model,
                tokenizer,
                requests,
                hparams,
                projector,
                contexts,
                touched=touched,
                request_microbatch_size=request_microbatch_size,
                mutation_lock=mutation_lock,
                ledger=job_ledger,
            )
            action = capture.raw_free_payload()
            materialization = materialize_native_candidates_once(
                touched, capture.candidates, capture.entry_sha256
            )
            edit_core = capture.edit_core_wall_seconds
            originals = entry
        elif role == "OFFICIAL_NATIVE":
            action, originals = run_official_native_apply(
                model,
                tokenizer,
                requests,
                hparams,
                touched=touched,
            )
            materialization = {
                "schema": "ode-edit-s05-p1r23-official-native-materialization/v1",
                "accepted_materialization_count": 1,
                "parameter_sha256": {
                    name: tensor_sha256(value) for name, value in touched.items()
                },
                "identity_sha256": canonical_hash(
                    {name: tensor_sha256(value) for name, value in touched.items()}
                ),
            }
            edit_core = float(action["edit_core_wall_seconds"])
        else:
            raise ODEBFContractError("P1R23 Native role differs")
    finally:
        counter.close()
    action_freeze = {
        "schema": f"{P1R23_SCHEMA}-native-action-freeze/v1",
        "role": role,
        "action_sha256": action["identity_sha256"],
        "materialization_sha256": materialization["identity_sha256"],
        "actions_frozen_before_heldout": True,
        "atomic_joint_batch": True,
        "persistent_history_append_count": 0,
        "replay_h_decision_influence_count": 0,
    }
    action_freeze["identity_sha256"] = canonical_hash(action_freeze)
    freeze_sha = write_once(raw_root / "action-freeze.json", action_freeze)
    cases, freeze = _action_frozen_cases(
        dataset_path,
        requests,
        arm=role,
        selected_snapshot_sha256=freeze_sha,
        fixed_budget_slots_completed=0,
    )
    endpoint, evaluator_time = _evaluate_frozen_state(
        model, tokenizer, cases, alias=alias, freeze_payload=freeze
    )
    restore = restore_native_entry(touched, originals)
    if _model_w0_contract(touched) != w0_contract:
        raise ODEBFStateError("P1R23 Native role did not restore W0")
    terminal = {
        "schema": f"{P1R23_SCHEMA}-native-terminal/v1",
        "instruction_id": P1R23_INSTRUCTION_ID,
        "source_head": source_head,
        "alias": alias,
        "role": role,
        "request_count": len(requests),
        "request_microbatch_size": request_microbatch_size,
        "action": action,
        "materialization": materialization,
        "endpoint": endpoint,
        "restore": restore,
        "edit_core_wall_seconds": edit_core,
        "terminal_evaluator_wall_seconds": evaluator_time,
        "compute": job_ledger.raw_free_payload(),
        "method_semantic_delta_count": 0,
        "estimand": "ATOMIC",
        "persistent_commit_count": 0,
        "persistent_history_append_count": 0,
        "replay_h_decision_influence_count": 0,
        "sequential_round_count": 0,
        "p1r20_access_count": 0,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": f"{P1R23_SCHEMA}-manifest/v1",
        "source_head": source_head,
        "terminal_sha256": terminal_sha,
        "request_count": len(requests),
        "role": role,
        "W0_restored": True,
        "estimand": "ATOMIC",
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = write_once(destination / "manifest.json", manifest)
    return {
        "status": f"P1R23_{role}_COMPLETE",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
    }


def _cosine_relative_l2(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    a = left.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    b = right.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ODEBFContractError("P1R23 calibration tensor geometry differs")
    a_norm = torch.linalg.vector_norm(a)
    b_norm = torch.linalg.vector_norm(b)
    if float(a_norm) == 0.0 or float(b_norm) == 0.0:
        cosine = 1.0 if torch.equal(a, b) else 0.0
    else:
        cosine = float(torch.dot(a, b) / (a_norm * b_norm))
    return {
        "max_abs": float(torch.max(torch.abs(a - b))),
        "relative_l2": float(
            torch.linalg.vector_norm(a - b)
            / torch.clamp(a_norm, min=torch.finfo(torch.float64).tiny)
        ),
        "cosine": cosine,
    }


def _run_calibration(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    destination: Path,
    raw_root: Path,
    source_head: str,
    requests: Sequence[Mapping[str, Any]],
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
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    job_ledger: ComputeLedger,
    write_once: Any,
) -> dict[str, Any]:
    """Outcome-free W0/W1 batching calibration for m=1 and production m=2.

    W1 is the single Neutral technical transition selected from each
    partition's W0 field.  It is never evaluated on held-out prompts and is
    restored before the next partition.
    """

    from .p1_runtime import ArmRuntimeState, _entry_parameter_snapshot_sha256

    if len(requests) != 10:
        raise ODEBFContractError("P1R23 calibration must use atomic B10")
    outer_population = tuple(
        population_by_sha256[item] for item in theta0_cache.request_order
    )
    outer_snapshot = _entry_parameter_snapshot_sha256(
        model, dict(base_receipt.parameter_sha256)
    )
    counter = ModelForwardCounter(model, job_ledger)
    try:
        outer_cache = build_outer_entry_pretrained_cache(
            model,
            tokenizer,
            outer_population,
            theta0_cache,
            outer_entry_snapshot_sha256=outer_snapshot,
        )
    finally:
        counter.close()
    w0_contract = _model_w0_contract(touched)
    internal: dict[int, list[dict[str, Any]]] = {}
    public: dict[str, Any] = {}

    for microbatch_size in (1, 2):
        if _model_w0_contract(touched) != w0_contract:
            raise ODEBFStateError("P1R23 calibration W0 differs")
        objective_plan = build_scalable_objective_plan(
            model,
            tokenizer,
            requests,
            contexts=contexts,
            request_microbatch_size=microbatch_size,
            fact_token_strategy=hparams.fact_token,
        )
        capture_plan = build_scalable_capture_plan(
            tokenizer,
            requests,
            contexts=contexts,
            request_microbatch_size=microbatch_size,
            fact_token_strategy=hparams.fact_token,
        )
        arm_state = ArmRuntimeState(
            P1Arm.R_BF,
            P1HistoryLedger(layer_order=P1R23_LAYER_ORDER, maximum_records=40),
            ComputeLedger(),
            ArmWeightSnapshot(
                P1Arm.R_BF,
                0,
                base_receipt.parameter_sha256,
                canonical_hash(
                    {
                        "role": "P1R23-CALIBRATION",
                        "microbatch_size": microbatch_size,
                        "weights": base_receipt.parameter_sha256,
                    }
                ),
            ),
            dict(base_values),
        )
        materializer = AcceptedPhysicalStateMaterializer(model, base_values)
        entry_capture = FixedE8EntryCapture(
            {name: value.detach().cpu().clone() for name, value in base_values.items()},
            dict(base_receipt.parameter_sha256),
        )
        current_factors: dict[str, tuple[WaypointFactor, ...]] = {
            name: () for name in touched
        }
        try:
            physical = capture_scalable_physical_state(model, capture_plan, hparams)
            initial = initial_target_from_capture(physical)
            metric = ScalableBatchGlobalMetric.from_z0(
                initial.target_z, objective_plan.request_order_sha256
            )
            states: list[dict[str, Any]] = []
            target = initial.target_z.clone()
            terminal = initial.current_terminal_z.clone()
            for state_index in (0, 1):
                replay_entry = _controller_replay_entry(
                    model,
                    tokenizer,
                    alias=alias,
                    arm_state=arm_state,
                    sample_waypoint=state_index + 1,
                    factors={},
                    request_by_sha256=request_by_sha256,
                    population_by_sha256=population_by_sha256,
                    schedule=schedule,
                    outer_entry_p_cache=outer_cache,
                )
                target_result = evaluate_scalable_target_new_objective(
                    model,
                    objective_plan,
                    target_state=target.detach()
                    .to(device=next(model.parameters()).device, dtype=torch.float32)
                    .clone()
                    .requires_grad_(True),
                    current_terminal=terminal,
                    target_layer_name=hparams.layer_module_tmp.format(
                        int(hparams.layers[-1])
                    ),
                )
                if target_result.target_gradient is None:
                    raise ODEBFContractError("P1R23 calibration target gradient absent")
                target_next, _velocity, alpha_req, target_receipt = (
                    target_update_from_existing_gradient(
                        target, target_result.target_gradient, metric
                    )
                )
                field = build_scalable_dynamic_field(
                    model,
                    tokenizer,
                    requests,
                    hparams,
                    projector,
                    contexts,
                    target_state=target_next,
                    current_terminal=terminal,
                    captured_keys_by_layer=physical.keys_by_layer,
                    accepted_waypoint=state_index,
                    covariance_registry=covariance_registry,
                    projector_sha256=projector_sha256,
                    residual_tolerance=controller_lock.residual_tolerance,
                    ledger=arm_state.ledger,
                )
                signed, slope_result = scalable_physical_signed_progress(
                    model, objective_plan, field
                )
                problem = build_scalable_routing_problem(
                    field,
                    signed,
                    accepted_by_layer={layer: () for layer in P1R23_LAYER_ORDER},
                    committed_load_by_layer={layer: 0.0 for layer in P1R23_LAYER_ORDER},
                    lock=controller_lock,
                )
                field_semantic = canonical_hash(
                    {
                        "field_sha256": field.identity_sha256,
                        "target_next_sha256": tensor_sha256(target_next),
                        "physical_capture_sha256": physical.identity_sha256,
                        "request_order_sha256": objective_plan.request_order_sha256,
                    }
                )
                inventory, functional_probe = _fixed_e8_functional_basis_probe(
                    model,
                    tokenizer,
                    alias=alias,
                    field=field,
                    step_index=state_index,
                    factors=current_factors,
                    target_state=target_next,
                    capture=entry_capture,
                    replay_entry=replay_entry,
                    theta0_cache=theta0_cache,
                    lock=controller_lock,
                    ledger=arm_state.ledger,
                    touched=touched,
                    history=arm_state.history,
                    schedule=schedule,
                    factor_state_sha256=_factor_state(
                        entry_capture.entry_sha256, current_factors, target_next
                    ),
                    field_semantic_sha256=field_semantic,
                    trial_entry_weights=base_values,
                    physical_materialized=True,
                )
                if inventory.history_item_count != 0:
                    raise ODEBFStateError("P1R23 calibration replay-H is active")
                neutral = solve_strength_preserving_routing(
                    problem.problem,
                    inventory,
                    arm=FixedE8Arm.NEUTRAL,
                    alpha_req=alpha_req,
                )
                soft = solve_strength_preserving_routing(
                    problem.problem,
                    inventory,
                    arm=FixedE8Arm.SOFT,
                    alpha_req=alpha_req,
                )
                slopes = tuple(float(item) for item in signed.signed_progress)
                state_public = {
                    "state_index": state_index,
                    "mean_target_new_nll": float(slope_result.loss),
                    "target_objective": target_result.raw_free_payload(),
                    "target_update": target_receipt,
                    "physical_capture_sha256": physical.identity_sha256,
                    "terminal_sha256": tensor_sha256(physical.terminal_z),
                    "key_inventory_sha256": _key_identity(physical.keys_by_layer),
                    "field_sha256": field.identity_sha256,
                    "signed_slopes": list(slopes),
                    "slope_sign": [int(np.sign(item)) for item in slopes],
                    "slope_rank_desc_stable": np.argsort(
                        -np.asarray(slopes), kind="stable"
                    ).tolist(),
                    "neutral": neutral.raw_free_payload(),
                    "soft": soft.raw_free_payload(),
                    "functional_inventory": inventory.raw_free_payload(),
                    "functional_probe_sha256": functional_probe["identity_sha256"],
                    "atomic_history_item_count": 0,
                }
                state_public["identity_sha256"] = canonical_hash(state_public)
                states.append(
                    {
                        "public": state_public,
                        "gradient": target_result.target_gradient.detach().cpu().clone(),
                        "terminal": physical.terminal_z.detach().cpu().clone(),
                        "neutral_velocity": torch.tensor(neutral.velocity),
                        "soft_velocity": torch.tensor(soft.velocity),
                    }
                )
                if state_index == 0:
                    increment = fixed_e8_waypoint_factors(
                        field, neutral.velocity, step_index=0
                    )
                    current_factors = _merge_factors(current_factors, increment)
                    materializer.materialize(current_factors, transition_index=1)
                    physical = capture_scalable_physical_state(
                        model, capture_plan, hparams
                    )
                    target = target_next
                    terminal = physical.terminal_z.clone()
            internal[microbatch_size] = states
            public[str(microbatch_size)] = {
                "objective_plan": objective_plan.raw_free_payload(),
                "capture_plan": capture_plan.raw_free_payload(),
                "initial_r0_nonzero_count": int(
                    torch.count_nonzero(initial.residual).item()
                ),
                "states": [item["public"] for item in states],
                "W1_is_single_neutral_technical_transition": True,
                "heldout_evaluator_access_count": 0,
                "persistent_history_append_count": 0,
            }
        finally:
            materializer.restore()
        if _model_w0_contract(touched) != w0_contract:
            raise ODEBFStateError("P1R23 calibration did not restore W0")

    comparisons: dict[str, Any] = {}
    reference = internal[1]
    for microbatch_size in (2,):
        observed = internal[microbatch_size]
        state_comparisons: list[dict[str, Any]] = []
        for state_index in (0, 1):
            left = reference[state_index]
            right = observed[state_index]
            left_public = left["public"]
            right_public = right["public"]
            state_comparisons.append(
                {
                    "state_index": state_index,
                    "target_gradient": _cosine_relative_l2(
                        left["gradient"], right["gradient"]
                    ),
                    "terminal_activation": _cosine_relative_l2(
                        left["terminal"], right["terminal"]
                    ),
                    "objective_mean_abs": abs(
                        float(left_public["mean_target_new_nll"])
                        - float(right_public["mean_target_new_nll"])
                    ),
                    "slope_sign_exact": (
                        left_public["slope_sign"] == right_public["slope_sign"]
                    ),
                    "slope_rank_exact": (
                        left_public["slope_rank_desc_stable"]
                        == right_public["slope_rank_desc_stable"]
                    ),
                    "neutral_velocity": _cosine_relative_l2(
                        left["neutral_velocity"], right["neutral_velocity"]
                    ),
                    "soft_velocity": _cosine_relative_l2(
                        left["soft_velocity"], right["soft_velocity"]
                    ),
                }
            )
        comparisons[f"m1_vs_m{microbatch_size}"] = state_comparisons

    terminal = {
        "schema": f"{P1R23_SCHEMA}-calibration-terminal/v1",
        "instruction_id": P1R23_INSTRUCTION_ID,
        "source_head": source_head,
        "alias": alias,
        "estimand": "ATOMIC_TECHNICAL_CALIBRATION",
        "request_count": 10,
        "microbatch_sizes": [1, 2],
        "partitions": public,
        "comparisons": comparisons,
        "endpoint_outcome_tuning_access_count": 0,
        "heldout_evaluator_access_count": 0,
        "persistent_history_append_count": 0,
        "replay_h_decision_influence_count": 0,
        "W0_restored": _model_w0_contract(touched) == w0_contract,
        "compute": job_ledger.raw_free_payload(),
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": f"{P1R23_SCHEMA}-manifest/v1",
        "source_head": source_head,
        "terminal_sha256": terminal_sha,
        "role": "CALIBRATION",
        "request_count": 10,
        "W0_restored": True,
        "estimand": "ATOMIC",
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = write_once(destination / "manifest.json", manifest)
    return {
        "status": "P1R23_CALIBRATION_COMPLETE",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
    }


def run_p1r23_scalable_batched(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    role: str,
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
    mutation_lock: threading.RLock,
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    job_ledger: ComputeLedger,
    write_once: Any,
    request_microbatch_size: int,
    numerical_lock: Mapping[str, Any],
    numerical_lock_sha256: str,
) -> dict[str, Any]:
    del stages
    p1r24_role = role.startswith(("P1R24_", "P1R34_", "P1R35_"))
    p1r35_role = role.startswith("P1R35_")
    p1r34_role = role.startswith("P1R34_") or p1r35_role
    progress_simplex_role = role.startswith("PROGRESS_SIMPLEX_")
    simplex_lock_sha256: str | None = None
    if progress_simplex_role:
        simplex_lock, simplex_lock_sha256 = load_rooted_json(
            Path(__file__).parent
            / "locks/numerical_lock_s05_progress_simplex_router.json",
            expected_schema="ode-edit-s05-p1r23-progress-simplex-router-lock/v1",
        )
        if (
            simplex_lock.get("instruction_id") != PROGRESS_SIMPLEX_INSTRUCTION_ID
            or simplex_lock.get("method_id") != PROGRESS_SIMPLEX_METHOD_ID
            or simplex_lock.get("layer_order") != [4, 5, 6, 7, 8]
            or simplex_lock.get("execution", {}).get("B100_access_count") != 0
        ):
            raise ODEBFContractError("progress-simplex execution lock differs")
    request_order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    if (
        len(requests) not in ((1, 10) if p1r24_role else (10, 100))
        or (
            not p1r24_role
            and request_order != stream["batch_ordered_request_digest_v1"][0]
        )
        or numerical_lock["native_controls"]["optimized_native_target"]
        != "SCIENTIFICALLY_IDENTICAL_DIRECT_Z_SOLVE"
        or numerical_lock["atomic_only"]["result_label"] != "ATOMIC"
        or numerical_lock["atomic_only"]["persistent_history_append_count"] != 0
        or numerical_lock["atomic_only"]["replay_h_decision_influence_count"] != 0
    ):
        raise ODEBFContractError("P1R23 execution lock differs")
    preflight = {
        "schema": f"{P1R23_SCHEMA}-execution-preflight/v1",
        "source_head": source_head,
        "role": role,
        "alias": alias,
        "request_count": len(requests),
        "request_order_sha256": request_order,
        "stream_root_digest": stream["root_digest"],
        "numerical_lock_sha256": numerical_lock_sha256,
        "progress_simplex_numerical_lock_sha256": simplex_lock_sha256,
        "request_microbatch_size": request_microbatch_size,
        "direct_z_role": role in ("OPTIMIZED_NATIVE_K1", "OFFICIAL_NATIVE"),
        "one_gradient_role": role in (
            "ODE_BF_K8_PAIR",
            "ODE_BF_K8_RS_PAIR",
            "PROGRESS_SIMPLEX_BG_PAIR",
            "PROGRESS_SIMPLEX_RS_PAIR",
            "P1R24_B1_RS_NEUTRAL",
            "P1R24_B10_RS_PAIR",
            "P1R24_B10_BG_PAIR",
            "P1R34_B1_RS_PAIR",
            "P1R34_B10_RS_PAIR",
            "P1R35_B1_RS_PAIR",
            "P1R35_B1_RS_PAIR_TECH_R1",
            "P1R35_B10_RS_PAIR",
        ),
        "estimand": "ATOMIC",
        "joint_batch_application_count": 1,
        "persistent_history_append_count": 0,
        "replay_h_decision_influence_count": 0,
        "sequential_controller_influence_count": 0,
        "p1r20_access_count": 0,
    }
    preflight["identity_sha256"] = canonical_hash(preflight)
    write_once(raw_root / "execution-preflight.json", preflight)
    if role in (
        "ODE_BF_K8_PAIR",
        "ODE_BF_K8_RS_PAIR",
        "PROGRESS_SIMPLEX_BG_PAIR",
        "PROGRESS_SIMPLEX_RS_PAIR",
        "P1R24_B1_RS_NEUTRAL",
        "P1R24_B10_RS_PAIR",
        "P1R24_B10_BG_PAIR",
        "P1R34_B1_RS_PAIR",
        "P1R34_B10_RS_PAIR",
        "P1R35_B1_RS_PAIR",
        "P1R35_B1_RS_PAIR_TECH_R1",
        "P1R35_B10_RS_PAIR",
    ):
        progress_simplex = progress_simplex_role
        return _run_ode_pair(
            model,
            tokenizer,
            alias=alias,
            destination=destination,
            raw_root=raw_root,
            source_head=source_head,
            requests=requests,
            hparams=hparams,
            projector=projector,
            contexts=contexts,
            covariance_registry=covariance_registry,
            projector_sha256=projector_sha256,
            controller_lock=controller_lock,
            request_by_sha256=request_by_sha256,
            population_by_sha256=population_by_sha256,
            schedule=schedule,
            theta0_cache=theta0_cache,
            dataset_path=dataset_path,
            touched=touched,
            base_receipt=base_receipt,
            base_values=base_values,
            job_ledger=job_ledger,
            write_once=write_once,
            request_microbatch_size=request_microbatch_size,
            allocation=(
                "RS"
                if role in (
                    "ODE_BF_K8_RS_PAIR",
                    "PROGRESS_SIMPLEX_RS_PAIR",
                    "P1R24_B1_RS_NEUTRAL",
                    "P1R24_B10_RS_PAIR",
                    "P1R34_B1_RS_PAIR",
                    "P1R34_B10_RS_PAIR",
                    "P1R35_B1_RS_PAIR",
                    "P1R35_B1_RS_PAIR_TECH_R1",
                    "P1R35_B10_RS_PAIR",
                )
                else "BG"
            ),
            progress_simplex=progress_simplex,
            p1r24=p1r24_role,
            p1r34=p1r34_role,
            p1r35=p1r35_role,
            neutral_only=role == "P1R24_B1_RS_NEUTRAL",
            technical_smoke=role in (
                "P1R34_B1_RS_PAIR",
                "P1R35_B1_RS_PAIR",
                "P1R35_B1_RS_PAIR_TECH_R1",
            ),
        )
    if role == "CALIBRATION":
        return _run_calibration(
            model,
            tokenizer,
            alias=alias,
            destination=destination,
            raw_root=raw_root,
            source_head=source_head,
            requests=requests,
            hparams=hparams,
            projector=projector,
            contexts=contexts,
            covariance_registry=covariance_registry,
            projector_sha256=projector_sha256,
            controller_lock=controller_lock,
            request_by_sha256=request_by_sha256,
            population_by_sha256=population_by_sha256,
            schedule=schedule,
            theta0_cache=theta0_cache,
            touched=touched,
            base_receipt=base_receipt,
            base_values=base_values,
            job_ledger=job_ledger,
            write_once=write_once,
        )
    if role in ("OPTIMIZED_NATIVE_K1", "OFFICIAL_NATIVE"):
        return _run_native_role(
            model,
            tokenizer,
            alias=alias,
            role=role,
            destination=destination,
            raw_root=raw_root,
            source_head=source_head,
            requests=requests,
            hparams=hparams,
            projector=projector,
            contexts=contexts,
            dataset_path=dataset_path,
            touched=touched,
            base_values=base_values,
            request_microbatch_size=request_microbatch_size,
            mutation_lock=mutation_lock,
            job_ledger=job_ledger,
            write_once=write_once,
        )
    raise ODEBFContractError("P1R23 execution role differs")


__all__ = ["run_p1r23_scalable_batched"]
