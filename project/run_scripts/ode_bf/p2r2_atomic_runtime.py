"""Independent P2R2 Atomic runtime: P2R1 target flow plus residual transport."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from .accounting import ComputeLedger
from .alpha_backend import seed_all
from .atomic_runtime_optimization import AcceptedPhysicalStateMaterializer
from .bg_soft_diagnostics import HeldoutRequestResidualActivationOverlay
from .common_cold_coordinate import common_terminal_residual_input
from .common_coldcoord_fixed_e8_runtime import _heldout_additive_lookup_geometry
from .contracts import BATCH_SIZE, COMMON_SEED, ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import WaypointFactor, tensor_sha256
from .p0_runtime import ModelForwardCounter
from .p1_adaptive_runtime import _merge_factors
from .p1_backend import PinnedCovarianceRegistry
from .p1_controller import AcceptedLayerContribution, P1ControllerLock
from .p1_runtime import _atomic_write_once
from .p1_scalable_batched_experiment import (
    _action_frozen_cases,
    _evaluate_frozen_state,
    _model_w0_contract,
)
from .p1_state import ArmWeightSnapshot
from .p1_stepwise import StepwiseActionFreeze, evaluate_counterfact_stepwise_primary
from .p1r24_atomic_strength import (
    P1R24AliasTargetLock,
    build_p1r24_kl_plan,
    evaluate_p1r24_kl,
    p1r24_disable_historical,
    verify_p1r24_alphaedit_geometry,
)
from .progress_simplex_routing import progress_simplex_waypoint_factors
from .p1r36_independent_b10x10_runtime import _hashes, _restore_exact_w0
from .p2r1_rms_tangent_target import (
    P2R1RMSState,
    P2R1_MICROSTEPS_PER_OUTER_STATE,
    P2R1_TARGET_MICROSTEP_COUNT,
    p2r1_preservation_gradient,
    p2r1_target_update,
)
from .p2r1_target_only_runtime import _panel_metric, _write_private_target_trajectory_once
from .p2r2_residual_transport_writer import (
    P2R2_INSTRUCTION_ID,
    P2R2_METHOD_ID,
    build_proposal_quadratics,
    measure_request_layer_response,
    p2r2_forbidden_influence_receipt,
    p2r2_waypoint_factors,
    solve_p2r2_routing,
)
from .p2r7_shared_writer import (
    P2R7_INSTRUCTION_ID,
    P2R7_METHOD_ID,
    P2R7SharedWriterNoPositiveDirection,
    build_p2r7_deficit_weighting,
    p2r7_actual_bf16_common_metrics,
    p2r7_forbidden_influence_receipt,
    p2r7_weighted_physical_signed_progress,
    solve_p2r7_shared_routing,
)
from .scalable_batched_field import build_scalable_dynamic_field
from .scalable_batched_field import build_scalable_routing_problem
from .scalable_batched_model import (
    build_scalable_capture_plan,
    build_scalable_objective_plan,
    capture_scalable_physical_state,
    evaluate_scalable_target_new_objective,
)
from .scalable_batched_runtime import initial_target_from_capture, scalable_ordered_request_digest


METHOD = "P2R2-SEMANTIC-CONSERVING-RESIDUAL-TRANSPORT-PAIRED"
ARMS = ("NEUTRAL", "SOFTP")
CASE_COUNTS = (1, 10)
STREAM_ROOT = "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6"
STREAM_ORDER = "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c"


def expected_p2r2_result_name(
    alias: str, *, case_count: int, attempt_suffix: str | None = None
) -> str:
    if alias not in ("llama3-8b-inst", "qwen2.5-7b-inst") or case_count not in CASE_COUNTS:
        raise ODEBFContractError("P2R2 result identity differs")
    phase = "sealed-b10-smoke" if case_count == 1 else "b10x10"
    suffix = f"-{attempt_suffix}" if attempt_suffix else ""
    return f"s05-p2r2-semantic-conserving-residual-transport-{phase}-{alias}-paired{suffix}-v1"


def _factor_sha(factor: WaypointFactor) -> str:
    return canonical_hash(
        {
            "weight_name_sha256": hashlib.sha256(factor.weight_name.encode()).hexdigest(),
            "layer": factor.layer,
            "order": list(factor.order_key),
            "theta": factor.theta,
            "left_sha256": tensor_sha256(factor.left),
            "right_sha256": tensor_sha256(factor.right),
        }
    )


def _evaluate_terminal_z_panel(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    cases: Sequence[Any],
    *,
    alias: str,
    request_order_sha256: str,
    action_sha256: str,
    hparams: Any,
    terminal_target: torch.Tensor,
    terminal_physical: Any,
    writer_state_sha256: str,
    variant: str = METHOD,
) -> tuple[dict[str, Any], float]:
    lookup_positions, patched_rows, lookup_receipt = _heldout_additive_lookup_geometry(
        tokenizer, requests, cases, fact_token_strategy=hparams.fact_token
    )
    residual = common_terminal_residual_input(
        terminal_target, terminal_physical.terminal_z, request_order_sha256
    )
    freeze = StepwiseActionFreeze(
        variant=variant,
        request_order_sha256=request_order_sha256,
        rollout_sha256=action_sha256,
        snapshot_sha256=canonical_hash(
            {
                "terminal_target_sha256": tensor_sha256(terminal_target),
                "terminal_physical_sha256": terminal_physical.identity_sha256,
                "writer_state_sha256": writer_state_sha256,
            }
        ),
        snapshot_index=8,
        accepted_snapshot_count=9,
        rejected_retry_count=0,
        trajectory_status="ACTION_FROZEN_P2R2_K8",
    )
    overlay = HeldoutRequestResidualActivationOverlay(
        model,
        hparams.layer_module_tmp.format(int(hparams.layers[-1])),
        residual.residual,
        lookup_positions,
        patched_rows,
    )
    started = time.perf_counter()
    with overlay:
        result = evaluate_counterfact_stepwise_primary(
            model, tokenizer, cases, model_alias=alias, freeze=freeze
        )
    elapsed = time.perf_counter() - started
    raw = result.raw_free_payload()
    payload = {
        "schema": "ode-edit-s05-p2r2-terminal-z-panel/v1",
        "action_freeze_sha256": freeze.identity(),
        "z_inject": raw,
        "eff_z_inject": _panel_metric(raw, "efficacy"),
        "gen_z_inject": _panel_metric(raw, "generalization"),
        "heldout_lookup": lookup_receipt,
        "z_overlay": overlay.raw_free_payload(),
        "terminal_residual": residual.raw_free_payload(),
        "heldout_controller_access_count": 0,
        "inner_step_heldout_evaluation_count": 0,
        "wall_seconds": elapsed,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload, elapsed


def _run_arm_case(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    arm: str,
    case_index: int,
    case_root: Path,
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    controller_lock: P1ControllerLock,
    dataset_path: Path,
    mutation_lock: Any,
    touched: Mapping[str, torch.nn.Parameter],
    base_values: Mapping[str, torch.Tensor],
    expected_w0: str,
    request_microbatch_size: int,
    job_ledger: ComputeLedger,
    p2r7_mode: str | None = None,
) -> dict[str, Any]:
    p2r7 = p2r7_mode is not None
    if (
        (not p2r7 and arm not in ARMS)
        or (
            p2r7
            and (
                p2r7_mode not in ("P1DW", "P1AGG")
                or arm not in ("NEUTRAL", "SOFT")
            )
        )
        or len(requests) != BATCH_SIZE
    ):
        raise ODEBFContractError("P2R2 arm/case inventory differs")
    request_order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    objective_plan = build_scalable_objective_plan(
        model,
        tokenizer,
        requests,
        contexts=contexts,
        request_microbatch_size=min(request_microbatch_size, BATCH_SIZE),
        fact_token_strategy=hparams.fact_token,
    )
    capture_plan = build_scalable_capture_plan(
        tokenizer,
        requests,
        contexts=contexts,
        request_microbatch_size=min(request_microbatch_size, BATCH_SIZE),
        fact_token_strategy=hparams.fact_token,
    )
    if objective_plan.request_order_sha256 != request_order or capture_plan.request_order_sha256 != request_order:
        raise ODEBFContractError("P2R2 plan order differs")
    _atomic_write_once(case_root / "raw" / "objective-plan.json", objective_plan.raw_free_payload())
    _atomic_write_once(case_root / "raw" / "capture-plan.json", capture_plan.raw_free_payload())
    forbidden = (
        p2r7_forbidden_influence_receipt()
        if p2r7
        else p2r2_forbidden_influence_receipt()
    )
    _atomic_write_once(case_root / "raw" / "forbidden-influence.json", forbidden)

    lock = P1R24AliasTargetLock.for_alias(alias)
    alpha_geometry = verify_p1r24_alphaedit_geometry(
        hparams, lock, easyedit_root=Path("/mnt/raid5/janghj/EasyEdit")
    )
    counter = ModelForwardCounter(model, job_ledger)
    materializer = AcceptedPhysicalStateMaterializer(model, base_values)
    compute = {
        "target_forward_count": 0,
        "target_backward_count": 0,
        "kl_forward_count": 0,
        "kl_backward_count": 0,
        "physical_capture_forward_count": 0,
        "physical_response_forward_count": 0,
        "physical_response_batched_vjp_count": 0,
        "post_write_objective_forward_count": 0,
        "candidate_forward_count": 0,
        "candidate_materialization_count": 0,
        "writer_materialization_count": 0,
    }
    started = time.perf_counter()
    cumulative_factors: dict[str, tuple[WaypointFactor, ...]] = {
        name: () for name in touched
    }
    accepted_by_layer: dict[int, list[AcceptedLayerContribution]] = {
        int(layer): [] for layer in hparams.layers
    }
    prior_structural_p = 0.0
    target_receipt_sha: list[str] = []
    writer_receipt_sha: list[str] = []
    k_states: list[torch.Tensor] = []
    step_payloads: list[dict[str, Any]] = []
    try:
        physical = capture_scalable_physical_state(model, capture_plan, hparams)
        compute["physical_capture_forward_count"] += physical.physical_forward_count
        initial = initial_target_from_capture(physical)
        current_target = initial.target_z.clone()
        target_origin = initial.target_z.clone()
        k_states.append(current_target.clone())
        kl_plan = build_p1r24_kl_plan(
            tokenizer,
            requests,
            request_order_sha256=request_order,
            request_microbatch_size=min(request_microbatch_size, BATCH_SIZE),
            fact_token_strategy=hparams.fact_token,
        )
        teacher_result, teacher = evaluate_p1r24_kl(
            model, kl_plan, teacher_log_probs=None
        )
        compute["kl_forward_count"] += teacher_result.model_forward_count
        state = P2R1RMSState.zero()
        current_w_objective = None
        target_wall = 0.0
        writer_wall = 0.0
        for outer in range(8):
            current_terminal = physical.terminal_z.clone()
            for inner in range(P2R1_MICROSTEPS_PER_OUTER_STATE):
                microstep = outer * P2R1_MICROSTEPS_PER_OUTER_STATE + inner
                target_started = time.perf_counter()
                variable = (
                    current_target.detach()
                    .to(device=next(model.parameters()).device, dtype=torch.float32)
                    .clone()
                    .requires_grad_(True)
                )
                semantic = evaluate_scalable_target_new_objective(
                    model,
                    objective_plan,
                    target_state=variable,
                    current_terminal=current_terminal,
                    target_layer_name=hparams.layer_module_tmp.format(int(hparams.layers[-1])),
                )
                kl_result, _ = evaluate_p1r24_kl(
                    model,
                    kl_plan,
                    teacher_log_probs=teacher,
                    target_state=variable,
                    current_terminal=current_terminal,
                    target_layer_name=hparams.layer_module_tmp.format(int(hparams.layers[-1])),
                )
                if semantic.target_gradient is None or kl_result.gradient is None:
                    raise ODEBFContractError("P2R2 target gradient is absent")
                compute["target_forward_count"] += semantic.model_forward_count
                compute["target_backward_count"] += semantic.backward_count
                compute["kl_forward_count"] += kl_result.model_forward_count
                compute["kl_backward_count"] += kl_result.backward_count
                preservation, decay_values, decay_gradient = p2r1_preservation_gradient(
                    current_target,
                    target_origin,
                    kl_result.gradient * BATCH_SIZE,
                    lock,
                )
                update = p2r1_target_update(
                    current_target,
                    target_origin,
                    semantic.target_gradient * BATCH_SIZE,
                    preservation,
                    state,
                    alias=alias,
                    microstep_index=microstep,
                    lock=lock,
                )
                target_wall += time.perf_counter() - target_started
                target_receipt = {
                    **dict(update.receipt),
                    "outer_step": outer,
                    "within_outer_microstep": inner,
                    "current_physical_state_sha256": physical.identity_sha256,
                    "current_terminal_sha256": tensor_sha256(current_terminal),
                    "target_new_nll_by_request": list(semantic.per_request_values),
                    "kl_by_request": list(kl_result.per_request_values),
                    "decay_by_request": [float(item) for item in decay_values],
                    "decay_gradient_sha256": tensor_sha256(decay_gradient),
                    "rms_reset_count": 0,
                }
                target_receipt["identity_sha256"] = canonical_hash(target_receipt)
                target_receipt_sha.append(
                    _atomic_write_once(
                        case_root / "raw" / "target" / f"microstep-{microstep:02d}.json",
                        target_receipt,
                    )
                )
                current_target = update.target_next
                state = update.state_next

            writer_started = time.perf_counter()
            weighting = None
            weighted_slope = None
            common_metrics = None
            route_payload: Mapping[str, Any]
            quadratics_payload: Mapping[str, Any] | None
            response_payload: Mapping[str, Any]
            before_weight_state: dict[str, torch.Tensor] | None = None
            field = build_scalable_dynamic_field(
                model,
                tokenizer,
                requests,
                hparams,
                projector,
                contexts,
                target_state=(
                    (
                        current_terminal
                        + (current_target - current_terminal) / 0.125
                    ).contiguous()
                    if p2r7
                    else current_target
                ),
                current_terminal=current_terminal,
                captured_keys_by_layer=physical.keys_by_layer,
                accepted_waypoint=outer,
                covariance_registry=covariance_registry,
                projector_sha256=projector_sha256,
                residual_tolerance=controller_lock.residual_tolerance,
                ledger=job_ledger,
                allow_zero_capacity=True,
            )
            if p2r7:
                if current_w_objective is None:
                    current_w_objective = evaluate_scalable_target_new_objective(
                        model, objective_plan, target_gradient_required=False
                    )
                    compute["post_write_objective_forward_count"] += (
                        current_w_objective.model_forward_count
                    )
                z_objective = evaluate_scalable_target_new_objective(
                    model,
                    objective_plan,
                    target_state=current_target.to(
                        device=next(model.parameters()).device, dtype=torch.float32
                    ),
                    current_terminal=current_terminal,
                    target_layer_name=hparams.layer_module_tmp.format(
                        int(hparams.layers[-1])
                    ),
                    target_gradient_required=False,
                )
                compute["target_forward_count"] += z_objective.model_forward_count
                weighting = build_p2r7_deficit_weighting(
                    current_w_objective.per_request_values,
                    z_objective.per_request_values,
                    mode=str(p2r7_mode),
                )
                signed, weighted_slope = p2r7_weighted_physical_signed_progress(
                    model, objective_plan, field, weighting.omega
                )
                compute["physical_response_forward_count"] += (
                    weighted_slope.model_forward_count
                )
                compute["physical_response_batched_vjp_count"] += 1
                raw_positive = tuple(value > 0.0 for value in signed.signed_progress)
                if weighting.rho_omega > 0.0 and not any(raw_positive):
                    raise P2R7SharedWriterNoPositiveDirection(
                        "SCIENTIFIC_SHARED_WRITER_NO_POSITIVE_DIRECTION"
                    )
                if weighting.no_semantic_deficit:
                    route = None
                    route_velocity = tuple(0.0 for _ in field.layers)
                    predicted_scalar = 0.0
                    route_payload = {
                        "schema": "ode-edit-s05-p2r7-no-semantic-deficit-route/v1",
                        "status": "NO_SEMANTIC_DEFICIT",
                        "arm": arm,
                        "rho_omega": 0.0,
                        "velocity": list(route_velocity),
                        "predicted_progress": 0.0,
                        "fallback_to_neutral": False,
                        "fallback_reason": None,
                    }
                    route_payload["identity_sha256"] = canonical_hash(route_payload)
                    increment = {}
                else:
                    problem_receipt = build_scalable_routing_problem(
                        field,
                        signed,
                        accepted_by_layer=accepted_by_layer,
                        committed_load_by_layer={
                            int(layer): 0.0 for layer in hparams.layers
                        },
                        lock=controller_lock,
                    )
                    routing_problem = p1r24_disable_historical(
                        problem_receipt.problem
                    )
                    route = solve_p2r7_shared_routing(
                        routing_problem,
                        arm=arm,
                        rho_omega=weighting.rho_omega,
                    )
                    route_velocity = route.velocity
                    predicted_scalar = route.predicted_progress
                    route_payload = route.raw_free_payload()
                    increment = progress_simplex_waypoint_factors(
                        field, route_velocity, step_index=outer
                    )
                    prior_structural_p = float(route.selected_p)
                quadratics_payload = None
                response_payload = {
                    "schema": "ode-edit-s05-p2r7-five-shared-slope/v1",
                    "signed_progress": list(signed.signed_progress),
                    "weighted_objective": weighted_slope.raw_free_payload(),
                    "routing_variable_count": 5,
                    "request_layer_response_matrix_count": 0,
                    "batched_vjp_count": 1,
                    "identity_sha256": canonical_hash(
                        {
                            "signed_progress": list(signed.signed_progress),
                            "objective_sha256": weighted_slope.identity_sha256,
                        }
                    ),
                }
                before_weight_state = {
                    name: parameter.detach().to(device="cpu", dtype=torch.float32).clone()
                    for name, parameter in touched.items()
                }
            else:
                response = measure_request_layer_response(model, objective_plan, field)
                compute["physical_response_forward_count"] += response.model_forward_count
                compute["physical_response_batched_vjp_count"] += response.batched_vjp_count
                quadratics = build_proposal_quadratics(
                    field,
                    cumulative_factors,
                    prior_structural_p=prior_structural_p,
                )
                route = solve_p2r2_routing(response.response, quadratics, arm=arm)
                increment = p2r2_waypoint_factors(field, route, outer_step=outer)
                route_velocity = tuple(float(item) for item in route.alpha.reshape(-1))
                predicted_scalar = float(torch.sum(route.predicted_progress))
                route_payload = route.raw_free_payload()
                quadratics_payload = quadratics.raw_free_payload()
                response_payload = response.raw_free_payload()
            cumulative_factors = _merge_factors(cumulative_factors, increment)
            materialization = materializer.materialize(
                cumulative_factors, transition_index=outer + 1
            )
            compute["writer_materialization_count"] += 1
            if p2r7:
                assert before_weight_state is not None
                common_metrics = p2r7_actual_bf16_common_metrics(
                    before_weight_state, touched, field
                )
            next_physical = capture_scalable_physical_state(model, capture_plan, hparams)
            compute["physical_capture_forward_count"] += next_physical.physical_forward_count
            post = evaluate_scalable_target_new_objective(
                model, objective_plan, target_gradient_required=False
            )
            compute["post_write_objective_forward_count"] += post.model_forward_count
            before_values = torch.tensor(
                (
                    current_w_objective.per_request_values
                    if p2r7
                    else response.current_nll_by_request
                ),
                dtype=torch.float64,
            )
            after_values = torch.tensor(post.per_request_values, dtype=torch.float64)
            actual = before_values - after_values
            predicted = (
                torch.tensor(
                    [
                        float(weight) * predicted_scalar
                        for weight in weighting.omega
                    ],
                    dtype=torch.float64,
                )
                if p2r7
                else route.predicted_progress
            )
            realization = actual / torch.clamp(
                torch.tensor(weighting.deficit, dtype=torch.float64)
                if p2r7
                else torch.abs(predicted),
                min=1.0e-12,
            )
            if p2r7:
                current_w_objective = post
            else:
                p_after = prior_structural_p + route.marginal_structural_p
                if p_after < -1.0e-8 or not math.isfinite(p_after):
                    raise ODEBFContractError("P2R2 cumulative Structural-P certificate failed")
                prior_structural_p = max(0.0, p_after)
            writer_wall += time.perf_counter() - writer_started
            writer_payload = {
                "schema": (
                    "ode-edit-s05-p2r7-shared-writer-step/v1"
                    if p2r7
                    else "ode-edit-s05-p2r2-writer-step/v1"
                ),
                "outer_step": outer,
                "arm": arm,
                "target_state_sha256": tensor_sha256(current_target),
                "current_terminal_sha256": tensor_sha256(current_terminal),
                "full_current_residual_sha256": tensor_sha256(current_target - current_terminal),
                "full_current_residual_norm_by_request": [
                    float(item)
                    for item in torch.linalg.vector_norm(
                        (current_target - current_terminal).double(), dim=0
                    )
                ],
                "field": field.raw_free_payload(),
                "response": response_payload,
                "quadratics": quadratics_payload,
                "route": route_payload,
                "factor_sha256": {
                    name: _factor_sha(value) for name, value in sorted(increment.items())
                },
                "materialization": materialization,
                "current_nll_by_request": before_values.tolist(),
                "next_nll_by_request": list(post.per_request_values),
                "predicted_progress_by_request": (
                    "NOT_RECORDED_SHARED_SCALAR_ONLY"
                    if p2r7
                    else predicted.tolist()
                ),
                "predicted_weighted_progress": (
                    predicted_scalar if p2r7 else None
                ),
                "actual_weighted_progress": (
                    float(
                        torch.tensor(weighting.omega, dtype=torch.float64)
                        @ actual
                    )
                    if p2r7
                    else None
                ),
                "actual_progress_by_request": actual.tolist(),
                "realization_by_request": realization.tolist(),
                "weighted_realization": (
                    float(
                        (
                            torch.tensor(weighting.omega, dtype=torch.float64)
                            @ actual
                        )
                        / max(weighting.rho_omega, 1.0e-12)
                    )
                    if p2r7
                    else None
                ),
                "w_z_gap_by_request": (
                    [
                        float(w_value - z_value)
                        for w_value, z_value in zip(
                            post.per_request_values,
                            weighting.ell_z,
                            strict=True,
                        )
                    ]
                    if p2r7
                    else None
                ),
                "deficit_weighting": (
                    weighting.raw_free_payload() if p2r7 else None
                ),
                "actual_bf16_common_metrics": common_metrics,
                "negative_actual_count": int(torch.sum(actual < 0.0)),
                "cumulative_structural_p": prior_structural_p,
                "routing_variable_count": 5 if p2r7 else 50,
                "request_layer_response_matrix_count": 0 if p2r7 else 1,
                "residual_inverse_h_count": 1 if p2r7 else 0,
                "physical_h_application_count": 1 if p2r7 else 0,
                "second_h_application_count": 0,
                "target_microstep_count_before_write": state.completed_microsteps,
                "one_joint_materialization_count": 1,
                "current_state_refresh_count": 1,
                "retry_count": 0,
                "backtracking_count": 0,
            }
            writer_payload["identity_sha256"] = canonical_hash(writer_payload)
            writer_receipt_sha.append(
                _atomic_write_once(
                    case_root / "raw" / "writer" / f"step-{outer:02d}.json",
                    writer_payload,
                )
            )
            step_payloads.append(writer_payload)
            if p2r7 and increment:
                for layer in field.layers:
                    accepted_by_layer[layer.layer].append(
                        AcceptedLayerContribution.from_field(
                            layer,
                            increment[layer.weight_name],
                            history_action=torch.zeros_like(layer.history_action),
                        )
                    )
            physical = next_physical
            k_states.append(current_target.clone())

        if (
            state.completed_microsteps != P2R1_TARGET_MICROSTEP_COUNT
            or len(k_states) != 9
            or compute["writer_materialization_count"] != 8
        ):
            raise ODEBFStateError("P2R2 K8/microstep/materialization count differs")
        final_target = evaluate_scalable_target_new_objective(
            model,
            objective_plan,
            target_state=current_target.to(
                device=next(model.parameters()).device, dtype=torch.float32
            ),
            current_terminal=physical.terminal_z,
            target_layer_name=hparams.layer_module_tmp.format(int(hparams.layers[-1])),
            target_gradient_required=False,
        )
        compute["target_forward_count"] += final_target.model_forward_count
        trajectory_file_sha, trajectory_tensor_sha = _write_private_target_trajectory_once(
            case_root / "private-target" / "kstate-targets.pt",
            k_states=k_states,
            request_order_sha256=request_order,
        )
        action = {
            "schema": (
                "ode-edit-s05-p2r7-action-freeze/v1"
                if p2r7
                else "ode-edit-s05-p2r2-action-freeze/v1"
            ),
            "instruction_id": P2R7_INSTRUCTION_ID if p2r7 else P2R2_INSTRUCTION_ID,
            "method_id": P2R7_METHOD_ID if p2r7 else P2R2_METHOD_ID,
            "alias": alias,
            "arm": arm,
            "case_index": case_index,
            "request_order_sha256": request_order,
            "target_microstep_receipt_sha256": target_receipt_sha,
            "writer_step_receipt_sha256": writer_receipt_sha,
            "target_microstep_count": state.completed_microsteps,
            "writer_transition_count": 8,
            "writer_materialization_count": 8,
            "terminal_target_sha256": tensor_sha256(current_target),
            "terminal_physical_sha256": physical.identity_sha256,
            "trajectory_tensor_sha256": trajectory_tensor_sha,
            "writer_state_sha256": canonical_hash(
                {
                    name: [_factor_sha(factor) for factor in factors]
                    for name, factors in sorted(cumulative_factors.items())
                }
            ),
            "actions_frozen_before_evaluator": True,
            "heldout_controller_access_count": 0,
            "stepwise_heldout_evaluation_count": 0,
            "retry_count": 0,
            "W0_sha256": expected_w0,
        }
        action["identity_sha256"] = canonical_hash(action)
        action_sha = _atomic_write_once(case_root / "action-freeze.json", action)
        cases, evaluator_freeze = _action_frozen_cases(
            dataset_path,
            requests,
            arm=(f"P2R7-{p2r7_mode}-{arm}" if p2r7 else f"P2R2-{arm}"),
            selected_snapshot_sha256=action_sha,
            fixed_budget_slots_completed=8,
        )
        endpoint, w_evaluator_wall = _evaluate_frozen_state(
            model,
            tokenizer,
            cases,
            alias=alias,
            freeze_payload=evaluator_freeze,
        )
        z_panel, z_evaluator_wall = _evaluate_terminal_z_panel(
            model,
            tokenizer,
            requests,
            cases,
            alias=alias,
            request_order_sha256=request_order,
            action_sha256=action["identity_sha256"],
            hparams=hparams,
            terminal_target=current_target,
            terminal_physical=physical,
            writer_state_sha256=action["writer_state_sha256"],
            variant=(f"P2R7-{p2r7_mode}-{arm}" if p2r7 else METHOD),
        )
    finally:
        counter.close()
        materializer_restore = materializer.restore()
    if _model_w0_contract(touched) != expected_w0:
        raise ODEBFStateError("P2R2 materializer did not restore W0")
    restore = _restore_exact_w0(
        touched,
        base_values,
        mutation_lock=mutation_lock,
        expected_contract=expected_w0,
    )
    terminal = {
        "schema": (
            "ode-edit-s05-p2r7-case-terminal/v1"
            if p2r7
            else "ode-edit-s05-p2r2-case-terminal/v1"
        ),
        "instruction_id": P2R7_INSTRUCTION_ID if p2r7 else P2R2_INSTRUCTION_ID,
        "method_id": P2R7_METHOD_ID if p2r7 else P2R2_METHOD_ID,
        "alias": alias,
        "arm": arm,
        "writer_weighting_mode": p2r7_mode,
        "case_index": case_index,
        "request_count": BATCH_SIZE,
        "request_order_sha256": request_order,
        "objective_plan_sha256": objective_plan.identity_sha256,
        "capture_plan_sha256": capture_plan.identity_sha256,
        "alpha_geometry": alpha_geometry,
        "action_freeze_sha256": action_sha,
        "terminal_target_sha256": tensor_sha256(current_target),
        "target_trajectory_file_sha256": trajectory_file_sha,
        "target_trajectory_tensor_sha256": trajectory_tensor_sha,
        "terminal_full_six_target_new_nll": final_target.loss,
        "terminal_full_six_target_new_nll_by_request": list(final_target.per_request_values),
        "terminal_w_panel": endpoint,
        "terminal_z_panel": z_panel,
        "target_microstep_count": state.completed_microsteps,
        "writer_transition_count": 8,
        "writer_materialization_count": 8,
        "response_batched_vjp_count": compute["physical_response_batched_vjp_count"],
        "routing_variable_count": 5 if p2r7 else 50,
        "request_layer_response_matrix_count": 0 if p2r7 else 8,
        "terminal_cumulative_structural_p": prior_structural_p,
        "compute": compute,
        "target_wall_seconds": target_wall,
        "writer_wall_seconds": writer_wall,
        "terminal_w_evaluator_wall_seconds": w_evaluator_wall,
        "terminal_z_evaluator_wall_seconds": z_evaluator_wall,
        "total_wall_seconds": time.perf_counter() - started,
        "materializer": materializer.raw_free_payload(),
        "materializer_restore": materializer_restore,
        "W0_restore": restore,
        "W0_restored": True,
        "forbidden_influence": forbidden,
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(case_root / "terminal.json", terminal)
    manifest = {
        "schema": (
            "ode-edit-s05-p2r7-case-manifest/v1"
            if p2r7
            else "ode-edit-s05-p2r2-case-manifest/v1"
        ),
        "alias": alias,
        "arm": arm,
        "case_index": case_index,
        "terminal_sha256": terminal_sha,
        "action_freeze_sha256": action_sha,
        "target_microstep_count": 24,
        "writer_materialization_count": 8,
        "W0_restored": True,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(case_root / "manifest.json", manifest)
    return {
        "status": "P2R2_CASE_COMPLETE",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "arm": arm,
        "case_index": case_index,
    }


def run_p2r2_atomic(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    destination: Path,
    raw_root: Path,
    stages: Any,
    source_head: str,
    stream_batches: Sequence[Sequence[Mapping[str, Any]]],
    stream: Mapping[str, Any],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    controller_lock: P1ControllerLock,
    dataset_path: Path,
    mutation_lock: Any,
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    job_ledger: ComputeLedger,
    request_microbatch_size: int,
    case_count: int,
) -> dict[str, Any]:
    if case_count not in CASE_COUNTS or len(stream_batches) != 10:
        raise ODEBFContractError("P2R2 case inventory differs")
    if stream.get("root_digest") != STREAM_ROOT or stream.get("all_request_order_sha256") != STREAM_ORDER:
        raise ODEBFContractError("P2R2 stream identity differs")
    if any(len(batch) != BATCH_SIZE for batch in stream_batches):
        raise ODEBFContractError("P2R2 stream is not B10x10")
    expected_w0 = _model_w0_contract(touched)
    if _hashes(touched) != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("P2R2 entry W0 differs")
    started = time.perf_counter()
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for case_index, requests in enumerate(stream_batches[:case_count], start=1):
        for arm in ARMS:
            seed_all(COMMON_SEED)
            if _model_w0_contract(touched) != expected_w0:
                raise ODEBFStateError("P2R2 cross-arm/case W0 leak detected")
            case_root = raw_root / "cases" / f"case-{case_index:02d}" / arm.lower()
            try:
                completed.append(
                    _run_arm_case(
                        model,
                        tokenizer,
                        requests,
                        alias=alias,
                        arm=arm,
                        case_index=case_index,
                        case_root=case_root,
                        hparams=hparams,
                        projector=projector,
                        contexts=contexts,
                        covariance_registry=covariance_registry,
                        projector_sha256=projector_sha256,
                        controller_lock=controller_lock,
                        dataset_path=dataset_path,
                        mutation_lock=mutation_lock,
                        touched=touched,
                        base_values=base_values,
                        expected_w0=expected_w0,
                        request_microbatch_size=request_microbatch_size,
                        job_ledger=job_ledger,
                    )
                )
            except Exception as exc:
                restore = _restore_exact_w0(
                    touched,
                    base_values,
                    mutation_lock=mutation_lock,
                    expected_contract=expected_w0,
                )
                failure = {
                    "schema": "ode-edit-s05-p2r2-case-failure/v1",
                    "alias": alias,
                    "arm": arm,
                    "case_index": case_index,
                    "classification": "TECHNICAL_FAIL",
                    "exception_class": type(exc).__name__,
                    "exception_message_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
                    "retry_count": 0,
                    "W0_restore": restore,
                    "next_arm_or_case_continues": True,
                }
                observability = getattr(exc, "raw_free_receipt", None)
                if isinstance(observability, Mapping):
                    failure["technical_observability"] = dict(observability)
                failure["identity_sha256"] = canonical_hash(failure)
                _atomic_write_once(case_root / "failure.json", failure)
                failed.append(failure)
            if _model_w0_contract(touched) != expected_w0:
                raise ODEBFStateError("P2R2 post-arm W0 differs")
            stages.record(
                f"post_p2r2_case_{case_index}_{arm.lower()}",
                {
                    "case_index": case_index,
                    "arm": arm,
                    "complete_count": len(completed),
                    "failure_count": len(failed),
                    "W0_restored": True,
                },
            )
    terminal = {
        "schema": "ode-edit-s05-p2r2-job-terminal/v1",
        "instruction_id": P2R2_INSTRUCTION_ID,
        "method_id": P2R2_METHOD_ID,
        "source_head": source_head,
        "alias": alias,
        "case_count_per_arm": case_count,
        "arm_count": 2,
        "attempt_count": case_count * 2,
        "request_attempt_count": case_count * 2 * BATCH_SIZE,
        "completed_case_arm_count": len(completed),
        "failed_case_arm_count": len(failed),
        "completed": completed,
        "failed": failed,
        "W0_restored": _model_w0_contract(touched) == expected_w0,
        "job_compute": job_ledger.raw_free_payload(),
        "total_wall_seconds": time.perf_counter() - started,
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p2r2-job-manifest/v1",
        "source_head": source_head,
        "alias": alias,
        "case_count_per_arm": case_count,
        "terminal_sha256": terminal_sha,
        "completed_case_arm_count": len(completed),
        "failed_case_arm_count": len(failed),
        "W0_restored": terminal["W0_restored"],
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    return {
        "status": "P2R2_ATOMIC_TERMINAL",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "completed_case_arm_count": len(completed),
        "failed_case_arm_count": len(failed),
        "W0_restored": terminal["W0_restored"],
    }


__all__ = ["ARMS", "CASE_COUNTS", "expected_p2r2_result_name", "run_p2r2_atomic"]
