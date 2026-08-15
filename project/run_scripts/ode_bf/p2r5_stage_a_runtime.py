"""P2R5 Stage-A Atomic pilot over the sealed P2R4 Clamp-ON runtime."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

import torch

from .accounting import ComputeLedger
from .alpha_backend import seed_all
from .atomic_runtime_optimization import AcceptedPhysicalStateMaterializer
from .contracts import BATCH_SIZE, COMMON_SEED, ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import WaypointFactor, tensor_sha256
from .p0_runtime import ModelForwardCounter
from .p1_adaptive_runtime import _merge_factors
from .p1_backend import PinnedCovarianceRegistry
from .p1_controller import P1ControllerLock
from .p1_runtime import _atomic_write_once
from .p1_scalable_batched_experiment import (
    _action_frozen_cases,
    _evaluate_frozen_state,
    _model_w0_contract,
)
from .p1_state import ArmWeightSnapshot
from .p1r24_atomic_strength import (
    P1R24AliasTargetLock,
    build_p1r24_kl_plan,
    evaluate_p1r24_kl,
    verify_p1r24_alphaedit_geometry,
)
from .p1r36_independent_b10x10_runtime import _hashes, _restore_exact_w0
from .p2r1_rms_tangent_target import (
    P2R1RMSState,
    P2R1_MICROSTEPS_PER_OUTER_STATE,
    P2R1_TARGET_MICROSTEP_COUNT,
    p2r1_preservation_gradient,
    p2r1_target_update,
)
from .p2r1_target_only_runtime import _write_private_target_trajectory_once
from .p2r2_residual_transport_writer import (
    build_proposal_quadratics,
    measure_request_layer_response,
    p2r2_waypoint_factors,
)
from .p2r4_phaseb_atomic_runtime import (
    _evaluate_terminal_z_panel,
    _factor_sha,
    _normalize_clamp_receipt,
)
from .p2r4_phaseb_receipts import committed_outer_h_receipt
from .p2r5_sdrt_writer import (
    P2R5_ARMS,
    P2R5_INSTRUCTION_ID,
    P2R5_METHOD_ID,
    build_sdrt_quadratics,
    clamp_safe_semantic_deficit,
    p2r5_forbidden_influence_receipt,
    pooled_nonnegative_realization_calibration,
    solve_sdrt_routing,
)
from .scalable_batched_field import build_scalable_dynamic_field
from .scalable_batched_model import (
    build_scalable_capture_plan,
    build_scalable_objective_plan,
    capture_scalable_physical_state,
    evaluate_scalable_target_new_objective,
)
from .scalable_batched_runtime import initial_target_from_capture, scalable_ordered_request_digest


METHOD = "P2R5-SDRT-STAGE-A-PAIRED"
P2R4_PARENT_HEAD = "96f23e804be66569ad3fcf98bfcdd8a02ff3f7fb"
P2R2_TECH_R4_ANCESTOR = "c97e8619b42da7954ce0e824c215a8a82d70589a"
P2R1_ANCESTOR = "8f817e13167289dac190fe74bfa42e2b3e01372d"
STREAM_ROOT = "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6"
STREAM_ORDER = "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c"
STAGE_A_CASES = {
    "llama3-8b-inst": (3, 5),
    "qwen2.5-7b-inst": (1, 4),
}


@dataclass(frozen=True, slots=True)
class P2AtomicArmRuntimePolicy:
    """Thin policy hooks around the sealed P2 target/write runtime."""

    instruction_id: str
    method_id: str
    receipt_namespace: str
    arm_label_prefix: str
    allowed_arms: tuple[str, ...]
    allowed_cases: Mapping[str, tuple[int, ...]]
    route_solver: Callable[..., Any]
    forbidden_receipt_builder: Callable[[], Mapping[str, Any]]
    shadow_solver: Callable[..., Any] | None = None


def _p2r5_route_adapter(
    response: torch.Tensor,
    deficit: torch.Tensor,
    entry_deficit: torch.Tensor,
    calibration: Any,
    quadratics: Any,
    *,
    arm: str,
) -> Any:
    del entry_deficit
    return solve_sdrt_routing(
        response,
        deficit,
        calibration,
        quadratics,
        arm=arm,
    )


P2R5_RUNTIME_POLICY = P2AtomicArmRuntimePolicy(
    instruction_id=P2R5_INSTRUCTION_ID,
    method_id=P2R5_METHOD_ID,
    receipt_namespace="p2r5-sdrt",
    arm_label_prefix="P2R5",
    allowed_arms=P2R5_ARMS,
    allowed_cases=STAGE_A_CASES,
    route_solver=_p2r5_route_adapter,
    forbidden_receipt_builder=p2r5_forbidden_influence_receipt,
)


def expected_p2r5_stage_a_result_name(
    alias: str,
    *,
    case_index: int,
    attempt_suffix: str | None = None,
) -> str:
    if alias not in STAGE_A_CASES or case_index not in STAGE_A_CASES[alias]:
        raise ODEBFContractError("P2R5 Stage-A result identity differs")
    suffix = f"-{attempt_suffix}" if attempt_suffix else ""
    return f"s05-p2r5-sdrt-stage-a-{alias}-case-{case_index:02d}-paired{suffix}-v1"


def _model_capacity(
    touched: Mapping[str, torch.nn.Parameter],
    base_values: Mapping[str, torch.Tensor],
) -> float:
    total = 0.0
    for name, parameter in touched.items():
        difference = parameter.detach().to(torch.float64).cpu() - base_values[name].to(torch.float64).cpu()
        total += float(torch.sum(difference.square()))
    if not math.isfinite(total) or total < 0.0:
        raise ODEBFStateError("P2R5 BF16 endpoint capacity is invalid")
    return total


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left64 = left.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    right64 = right.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    denominator = float(torch.linalg.vector_norm(left64) * torch.linalg.vector_norm(right64))
    return 0.0 if denominator == 0.0 else float(torch.dot(left64, right64)) / denominator


def _proposal_difference_norm(field: Any, entry_field: Any) -> float:
    if len(field.layers) != len(entry_field.layers):
        raise ODEBFContractError("P2R5 proposal refresh layer inventory differs")
    total = 0.0
    for current, entry in zip(field.layers, entry_field.layers, strict=True):
        if current.weight_name != entry.weight_name:
            raise ODEBFContractError("P2R5 proposal refresh layer order differs")
        r = current.residual.detach().to(torch.float64)
        q = current.q.detach().to(torch.float64)
        r0 = entry.residual.detach().to(torch.float64)
        q0 = entry.q.detach().to(torch.float64)
        value = (
            torch.linalg.vector_norm(r, dim=0).square()
            * torch.linalg.vector_norm(q, dim=0).square()
            + torch.linalg.vector_norm(r0, dim=0).square()
            * torch.linalg.vector_norm(q0, dim=0).square()
            - 2.0 * torch.sum(r * r0, dim=0) * torch.sum(q * q0, dim=0)
        )
        total += float(torch.sum(torch.clamp(value, min=0.0)))
    return math.sqrt(max(total, 0.0))


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
    runtime_policy: P2AtomicArmRuntimePolicy = P2R5_RUNTIME_POLICY,
) -> dict[str, Any]:
    if (
        arm not in runtime_policy.allowed_arms
        or alias not in runtime_policy.allowed_cases
        or case_index not in runtime_policy.allowed_cases[alias]
        or len(requests) != BATCH_SIZE
    ):
        raise ODEBFContractError("P2 Atomic arm/case inventory differs")
    instruction_id = runtime_policy.instruction_id
    method_id = runtime_policy.method_id
    receipt_namespace = runtime_policy.receipt_namespace
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
        raise ODEBFContractError("P2 Atomic plan order differs")
    _atomic_write_once(case_root / "raw" / "objective-plan.json", objective_plan.raw_free_payload())
    _atomic_write_once(case_root / "raw" / "capture-plan.json", capture_plan.raw_free_payload())
    forbidden = dict(runtime_policy.forbidden_receipt_builder())
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
        "clamped_z_deficit_forward_count": 0,
        "clamped_z_deficit_backward_count": 0,
        "physical_capture_forward_count": 0,
        "physical_response_forward_count": 0,
        "physical_response_batched_vjp_count": 0,
        "post_write_objective_forward_count": 0,
        "candidate_forward_count": 0,
        "candidate_materialization_count": 0,
        "writer_materialization_count": 0,
        "nnls_model_pass_count": 0,
        "routing_model_pass_count": 0,
    }
    started = time.perf_counter()
    cumulative_factors: dict[str, tuple[WaypointFactor, ...]] = {
        name: () for name in touched
    }
    prior_structural_p = 0.0
    predicted_history: list[torch.Tensor] = []
    actual_history: list[torch.Tensor] = []
    target_receipt_sha: list[str] = []
    writer_receipt_sha: list[str] = []
    k_states: list[torch.Tensor] = []
    step_payloads: list[dict[str, Any]] = []
    allocations: list[list[list[float]]] = []
    shadow_receipt_sha: list[str] = []
    entry_deficit: torch.Tensor | None = None
    entry_target_field: torch.Tensor | None = None
    entry_writer_field: Any | None = None
    entry_allocation: torch.Tensor | None = None
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
        teacher_result, teacher = evaluate_p1r24_kl(model, kl_plan, teacher_log_probs=None)
        compute["kl_forward_count"] += teacher_result.model_forward_count
        state = P2R1RMSState.zero()
        target_wall = 0.0
        writer_wall = 0.0
        for outer in range(8):
            current_terminal = physical.terminal_z.clone()
            outer_target_receipts: list[dict[str, Any]] = []
            last_target_field: torch.Tensor | None = None
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
                    raise ODEBFContractError("P2 Atomic target gradient is absent")
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
                target_receipt = _normalize_clamp_receipt(
                    {
                        **dict(update.receipt),
                        "instruction_id": instruction_id,
                        "outer_step": outer,
                        "within_outer_microstep": inner,
                        "current_physical_state_sha256": physical.identity_sha256,
                        "current_terminal_sha256": tensor_sha256(current_terminal),
                        "target_new_nll_by_request": list(semantic.per_request_values),
                        "kl_by_request": list(kl_result.per_request_values),
                        "decay_by_request": [float(item) for item in decay_values],
                        "decay_gradient_sha256": tensor_sha256(decay_gradient),
                        "rms_reset_count": 0,
                        "clamp_on_decision_active": True,
                    },
                    clamp_policy="ON",
                )
                target_receipt_sha.append(
                    _atomic_write_once(
                        case_root / "raw" / "target" / f"microstep-{microstep:02d}.json",
                        target_receipt,
                    )
                )
                outer_target_receipts.append(target_receipt)
                current_target = update.target_next
                state = update.state_next
                last_target_field = update.field.detach().cpu()
            if last_target_field is None:
                raise ODEBFStateError("P2 Atomic target field is absent")
            if entry_target_field is None:
                entry_target_field = last_target_field.clone()

            writer_started = time.perf_counter()
            z_objective = evaluate_scalable_target_new_objective(
                model,
                objective_plan,
                target_state=current_target.to(
                    device=next(model.parameters()).device, dtype=torch.float32
                ),
                current_terminal=current_terminal,
                target_layer_name=hparams.layer_module_tmp.format(int(hparams.layers[-1])),
                target_gradient_required=False,
            )
            compute["clamped_z_deficit_forward_count"] += z_objective.model_forward_count
            field = build_scalable_dynamic_field(
                model,
                tokenizer,
                requests,
                hparams,
                projector,
                contexts,
                target_state=current_target,
                current_terminal=current_terminal,
                captured_keys_by_layer=physical.keys_by_layer,
                accepted_waypoint=outer,
                covariance_registry=covariance_registry,
                projector_sha256=projector_sha256,
                residual_tolerance=controller_lock.residual_tolerance,
                ledger=job_ledger,
                allow_zero_capacity=True,
            )
            response = measure_request_layer_response(model, objective_plan, field)
            compute["physical_response_forward_count"] += response.model_forward_count
            compute["physical_response_batched_vjp_count"] += response.batched_vjp_count
            deficit = clamp_safe_semantic_deficit(
                response.current_nll_by_request, z_objective.per_request_values
            )
            if entry_deficit is None:
                entry_deficit = deficit.clone()
            calibration = pooled_nonnegative_realization_calibration(
                predicted_history, actual_history
            )
            base_quadratics = build_proposal_quadratics(
                field,
                cumulative_factors,
                prior_structural_p=prior_structural_p,
            )
            quadratics = build_sdrt_quadratics(field, cumulative_factors, base_quadratics)
            shadow_panel = None
            if runtime_policy.shadow_solver is not None:
                shadow_panel = runtime_policy.shadow_solver(
                    response.response,
                    deficit,
                    entry_deficit,
                    calibration,
                    quadratics,
                )
                shadow_payload = shadow_panel.raw_free_payload()
                if (
                    shadow_payload.get("model_forward_count") != 0
                    or shadow_payload.get("model_backward_count") != 0
                    or shadow_payload.get("materialization_count") != 0
                ):
                    raise ODEBFContractError("P2 shadow solve added model/materialization work")
                shadow_receipt_sha.append(
                    _atomic_write_once(
                        case_root / "raw" / "shadow" / f"step-{outer:02d}.json",
                        shadow_payload,
                    )
                )
            if shadow_panel is not None and arm in tuple(
                item.arm for item in shadow_panel.routes
            ):
                route = shadow_panel.route(arm)
            else:
                route = runtime_policy.route_solver(
                    response.response,
                    deficit,
                    entry_deficit,
                    calibration,
                    quadratics,
                    arm=arm,
                )
            increment = p2r2_waypoint_factors(field, route, outer_step=outer)
            cumulative_factors = _merge_factors(cumulative_factors, increment)
            materialization = materializer.materialize(
                cumulative_factors, transition_index=outer + 1
            )
            compute["writer_materialization_count"] += 1
            actual_capacity = _model_capacity(touched, base_values)
            next_physical = capture_scalable_physical_state(model, capture_plan, hparams)
            compute["physical_capture_forward_count"] += next_physical.physical_forward_count
            post = evaluate_scalable_target_new_objective(
                model, objective_plan, target_gradient_required=False
            )
            compute["post_write_objective_forward_count"] += post.model_forward_count
            before_values = torch.tensor(response.current_nll_by_request, dtype=torch.float64)
            after_values = torch.tensor(post.per_request_values, dtype=torch.float64)
            actual = before_values - after_values
            raw_predicted = route.raw_predicted_response
            calibrated_predicted = route.calibrated_predicted_response
            calibrated_realization = actual / torch.clamp(
                torch.abs(calibrated_predicted), min=1.0e-12
            )
            raw_realization = actual / torch.clamp(torch.abs(raw_predicted), min=1.0e-12)
            predicted_history.append(raw_predicted.clone())
            actual_history.append(actual.clone())
            prior_structural_p = route.cumulative_structural_p
            if entry_writer_field is None:
                entry_writer_field = field
            if entry_allocation is None:
                entry_allocation = route.allocation.clone()
            writer_wall += time.perf_counter() - writer_started
            outer_h_receipt = committed_outer_h_receipt(committed_joint_transition=True)
            field_refresh = {
                "target_field_cosine_to_entry": _cosine(last_target_field, entry_target_field),
                "full_residual_cosine_to_entry": _cosine(
                    current_target - current_terminal,
                    entry_writer_field.target_state - entry_writer_field.current_z,
                ),
                "proposal_difference_norm_to_entry": _proposal_difference_norm(
                    field, entry_writer_field
                ),
                "allocation_difference_norm_to_entry": float(
                    torch.linalg.vector_norm(route.allocation - entry_allocation)
                ),
            }
            writer_payload = {
                "schema": f"ode-edit-s05-{receipt_namespace}-writer-step/v1",
                "instruction_id": instruction_id,
                "method_id": method_id,
                "p2r4_parent_head": P2R4_PARENT_HEAD,
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
                "clamped_z_nll_by_request": list(z_objective.per_request_values),
                "current_w_nll_by_request": list(response.current_nll_by_request),
                "semantic_deficit_by_request": deficit.tolist(),
                "calibration": calibration.raw_free_payload(),
                "target_clamp_microsteps": [
                    {
                        "pre_clamp_displacement_norm": item["pre_clamp_displacement_norm"],
                        "post_clamp_displacement_norm": item["post_clamp_displacement_norm"],
                        "clamp_ratio": item["clamp_ratio"],
                        "clamp_hit_count": item["clamp_hit_count"],
                    }
                    for item in outer_target_receipts
                ],
                "field": field.raw_free_payload(),
                "response": response.raw_free_payload(),
                "p2r2_quadratics": base_quadratics.raw_free_payload(),
                "sdrt_quadratics": quadratics.raw_free_payload(),
                "route": route.raw_free_payload(),
                "shadow_panel_identity_sha256": (
                    shadow_panel.identity_sha256 if shadow_panel is not None else "NOT_APPLICABLE"
                ),
                "shadow_model_forward_count": 0,
                "shadow_model_backward_count": 0,
                "shadow_materialization_count": 0,
                "factor_sha256": {
                    name: _factor_sha(value) for name, value in sorted(increment.items())
                },
                "materialization": materialization,
                "next_w_nll_by_request": list(post.per_request_values),
                "raw_predicted_response_by_request": raw_predicted.tolist(),
                "calibrated_predicted_response_by_request": calibrated_predicted.tolist(),
                "actual_progress_by_request": actual.tolist(),
                "actual_over_raw_predicted_by_request": raw_realization.tolist(),
                "actual_over_calibrated_predicted_by_request": calibrated_realization.tolist(),
                "negative_actual_count": int(torch.sum(actual < 0.0)),
                "predicted_cumulative_structural_p": route.cumulative_structural_p,
                "predicted_cumulative_capacity": route.cumulative_capacity,
                "actual_bf16_cumulative_capacity": actual_capacity,
                "field_refresh": field_refresh,
                **outer_h_receipt,
                "outer_h_receipt_identity_sha256": outer_h_receipt["identity_sha256"],
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
            allocations.append(route.allocation.tolist())
            physical = next_physical
            k_states.append(current_target.clone())

        if state.completed_microsteps != P2R1_TARGET_MICROSTEP_COUNT or len(k_states) != 9 or compute["writer_materialization_count"] != 8:
            raise ODEBFStateError("P2 Atomic K8/microstep/materialization count differs")
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
            "schema": f"ode-edit-s05-{receipt_namespace}-action-freeze/v1",
            "instruction_id": instruction_id,
            "method_id": method_id,
            "alias": alias,
            "arm": arm,
            "case_index": case_index,
            "request_order_sha256": request_order,
            "target_microstep_receipt_sha256": target_receipt_sha,
            "writer_step_receipt_sha256": writer_receipt_sha,
            "shadow_step_receipt_sha256": shadow_receipt_sha,
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
            "retry_count": 0,
            "W0_sha256": expected_w0,
        }
        action["identity_sha256"] = canonical_hash(action)
        action_sha = _atomic_write_once(case_root / "action-freeze.json", action)
        cases, evaluator_freeze = _action_frozen_cases(
            dataset_path,
            requests,
            arm=f"{runtime_policy.arm_label_prefix}-{arm}",
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
            clamp_policy="ON",
        )
    finally:
        counter.close()
        materializer_restore = materializer.restore()
    if _model_w0_contract(touched) != expected_w0:
        raise ODEBFStateError("P2 Atomic materializer did not restore W0")
    restore = _restore_exact_w0(
        touched,
        base_values,
        mutation_lock=mutation_lock,
        expected_contract=expected_w0,
    )
    terminal = {
        "schema": f"ode-edit-s05-{receipt_namespace}-case-terminal/v1",
        "instruction_id": instruction_id,
        "method_id": method_id,
        "p2r4_parent_head": P2R4_PARENT_HEAD,
        "alias": alias,
        "arm": arm,
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
        "terminal_cumulative_structural_p": prior_structural_p,
        "terminal_actual_bf16_capacity": step_payloads[-1]["actual_bf16_cumulative_capacity"],
        "terminal_predicted_capacity": step_payloads[-1]["predicted_cumulative_capacity"],
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
        "schema": f"ode-edit-s05-{receipt_namespace}-case-manifest/v1",
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
        "status": f"{runtime_policy.arm_label_prefix}_CASE_COMPLETE",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "arm": arm,
        "case_index": case_index,
        "allocations_by_step": allocations,
    }


def run_p2r5_stage_a(
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
    case_index: int,
    arms: Sequence[str] | None = None,
) -> dict[str, Any]:
    if alias not in STAGE_A_CASES or case_index not in STAGE_A_CASES[alias] or len(stream_batches) != 10:
        raise ODEBFContractError("P2R5 Stage-A case inventory differs")
    if stream.get("root_digest") != STREAM_ROOT or stream.get("all_request_order_sha256") != STREAM_ORDER:
        raise ODEBFContractError("P2R5 stream identity differs")
    if any(len(batch) != BATCH_SIZE for batch in stream_batches):
        raise ODEBFContractError("P2R5 stream is not B10x10")
    selected_arms = tuple(P2R5_ARMS if arms is None else arms)
    if (
        not selected_arms
        or len(set(selected_arms)) != len(selected_arms)
        or any(arm not in P2R5_ARMS for arm in selected_arms)
        or selected_arms != tuple(arm for arm in P2R5_ARMS if arm in selected_arms)
    ):
        raise ODEBFContractError("P2R5 Stage-A selected arm inventory differs")
    expected_w0 = _model_w0_contract(touched)
    if _hashes(touched) != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("P2R5 entry W0 differs")
    started = time.perf_counter()
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    requests = stream_batches[case_index - 1]
    for arm in selected_arms:
        seed_all(COMMON_SEED)
        if _model_w0_contract(touched) != expected_w0:
            raise ODEBFStateError("P2R5 cross-arm W0 leak detected")
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
                "schema": "ode-edit-s05-p2r5-sdrt-case-failure/v1",
                "alias": alias,
                "arm": arm,
                "case_index": case_index,
                "classification": "TECHNICAL_FAIL",
                "exception_class": type(exc).__name__,
                "exception_message_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
                "retry_count": 0,
                "W0_restore": restore,
                "next_arm_continues": True,
            }
            observability = getattr(exc, "raw_free_receipt", None)
            if isinstance(observability, Mapping):
                failure["technical_observability"] = dict(observability)
            failure["technical_location"] = [
                {
                    "file_identity_sha256": hashlib.sha256(frame.filename.encode()).hexdigest(),
                    "line_number": frame.lineno,
                    "symbol": frame.name,
                }
                for frame in traceback.extract_tb(exc.__traceback__)[-4:]
            ]
            failure["identity_sha256"] = canonical_hash(failure)
            _atomic_write_once(case_root / "failure.json", failure)
            failed.append(failure)
        if _model_w0_contract(touched) != expected_w0:
            raise ODEBFStateError("P2R5 post-arm W0 differs")
        stages.record(
            f"post_p2r5_stage_a_case_{case_index}_{arm.lower()}",
            {
                "case_index": case_index,
                "arm": arm,
                "complete_count": len(completed),
                "failure_count": len(failed),
                "W0_restored": True,
            },
        )
    paired_allocation_distance: list[float] | str = "NOT_RECORDED"
    if selected_arms == P2R5_ARMS and len(completed) == 2:
        cap = completed[0]["allocations_by_step"]
        structp = completed[1]["allocations_by_step"]
        paired_allocation_distance = [
            float(
                torch.linalg.vector_norm(
                    torch.tensor(right, dtype=torch.float64)
                    - torch.tensor(left, dtype=torch.float64)
                )
            )
            for left, right in zip(cap, structp, strict=True)
        ]
    terminal = {
        "schema": "ode-edit-s05-p2r5-sdrt-stage-a-job-terminal/v1",
        "instruction_id": P2R5_INSTRUCTION_ID,
        "method_id": P2R5_METHOD_ID,
        "p2r4_parent_head": P2R4_PARENT_HEAD,
        "source_head": source_head,
        "alias": alias,
        "case_index": case_index,
        "arm_count": len(selected_arms),
        "selected_arms": list(selected_arms),
        "attempt_count": len(selected_arms),
        "request_attempt_count": BATCH_SIZE * len(selected_arms),
        "completed_case_arm_count": len(completed),
        "failed_case_arm_count": len(failed),
        "completed": completed,
        "failed": failed,
        "paired_allocation_distance_by_step": paired_allocation_distance,
        "W0_restored": _model_w0_contract(touched) == expected_w0,
        "job_compute": job_ledger.raw_free_payload(),
        "total_wall_seconds": time.perf_counter() - started,
        "stage_b_status": "CLOSED_PENDING_GH_STAGE_A_REVIEW",
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p2r5-sdrt-stage-a-job-manifest/v1",
        "source_head": source_head,
        "alias": alias,
        "case_index": case_index,
        "terminal_sha256": terminal_sha,
        "completed_case_arm_count": len(completed),
        "failed_case_arm_count": len(failed),
        "W0_restored": terminal["W0_restored"],
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    return {
        "status": "P2R5_STAGE_A_TERMINAL",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "completed_case_arm_count": len(completed),
        "failed_case_arm_count": len(failed),
        "W0_restored": terminal["W0_restored"],
    }


# Public reusable hook; legacy P2R5 calls the same function with the frozen
# default policy, while later controllers inject only allocation policy.
run_p2_atomic_arm_case = _run_arm_case


__all__ = [
    "METHOD",
    "P2AtomicArmRuntimePolicy",
    "P2R5_RUNTIME_POLICY",
    "P2R4_PARENT_HEAD",
    "STAGE_A_CASES",
    "expected_p2r5_stage_a_result_name",
    "run_p2_atomic_arm_case",
    "run_p2r5_stage_a",
]
