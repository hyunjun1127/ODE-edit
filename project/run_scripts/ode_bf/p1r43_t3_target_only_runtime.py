"""Fixed-W0 24-update replay of the exact P1R43 target decision operator."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from .accounting import ComputeLedger
from .analysis import PrimaryMetric
from .alpha_backend import seed_all
from .bg_soft_diagnostics import HeldoutRequestResidualActivationOverlay
from .common_cold_coordinate import common_terminal_residual_input
from .common_coldcoord_fixed_e8_runtime import _heldout_additive_lookup_geometry
from .contracts import (
    BATCH_SIZE,
    COMMON_SEED,
    ODEBFContractError,
    ODEBFStateError,
    canonical_hash,
)
from .functional import tensor_sha256
from .evaluator import _EvaluationStateGuard, _is_llama, _model_device
from .p0_runtime import ModelForwardCounter
from .p1_evaluator import _evaluate_prefixes_pinned, _metric_receipt
from .p1_runtime import _atomic_write_once
from .p1_scalable_batched_experiment import (
    _action_frozen_cases,
    _model_w0_contract,
)
from .p1_state import ArmWeightSnapshot
from .p1_stepwise import StepwiseActionFreeze, evaluate_counterfact_stepwise_primary
from .p1r24_atomic_strength import P1R24_H
from .p1r36_independent_b10x10_runtime import _hashes, _restore_exact_w0
from .p1r43_independent_b10x10_runtime import STREAM_ORDER, STREAM_ROOT
from .p1r43_rho_free_target import (
    P1R43ControllerState,
    prepare_p1r43_rescue_proposal,
    prepare_p1r43_target_proposal,
    select_p1r43_target_proposal,
)
from .scalable_batched_field import ScalableRobustSharedMetric
from .scalable_batched_model import (
    build_scalable_capture_plan,
    build_scalable_objective_plan,
    capture_scalable_physical_state,
    evaluate_scalable_target_new_objective,
)
from .scalable_batched_runtime import initial_target_from_capture, scalable_ordered_request_digest


P1R43_T3_INSTRUCTION_ID = (
    "ODEEDIT-S05-P1R43-T3-FIXED-W0-TARGET-HORIZON-ABLATION-V1"
)
P1R43_T3_METHOD_ID = "P1R43-T3-FIXED-W0-TARGET-HORIZON-ABLATION-V1"
P1R43_T3_TARGET_UPDATE_COUNT = 24
P1R43_T3_ENDPOINT_UPDATES = (8, 24)
METHOD = "P1R43-T3-FIXED-W0-TARGET-HORIZON"
CASE_COUNTS = (1, 10)


def expected_p1r43_t3_target_result_name(
    alias: str,
    *,
    case_count: int,
    attempt_suffix: str | None = None,
) -> str:
    if alias not in ("llama3-8b-inst", "qwen2.5-7b-inst") or case_count not in CASE_COUNTS:
        raise ODEBFContractError("P1R43-T3 target result identity differs")
    phase = "first-b10" if case_count == 1 else "b10x10"
    suffix = f"-{attempt_suffix}" if attempt_suffix else ""
    return f"s05-p1r43-t3-fixed-w0-target-horizon-{phase}-{alias}{suffix}-v1"


def _write_private_target_trajectory_once(
    path: Path, *, target_states: Sequence[torch.Tensor], request_order_sha256: str
) -> tuple[str, str]:
    if len(target_states) != P1R43_T3_TARGET_UPDATE_COUNT + 1:
        raise ODEBFContractError("P1R43-T3 target state inventory differs")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(
                {
                    "schema": "ode-edit-s05-p1r43-t3-private-target-trajectory/v1",
                    "request_order_sha256": request_order_sha256,
                    "h": P1R24_H,
                    "target_update_count": P1R43_T3_TARGET_UPDATE_COUNT,
                    "endpoint_update_indices": list(P1R43_T3_ENDPOINT_UPDATES),
                    "targets": torch.stack(
                        [item.detach().cpu().to(torch.float32) for item in target_states]
                    ),
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        raise
    raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest(), tensor_sha256(torch.stack(target_states))


def _panel_metric(payload: Mapping[str, Any], metric: str) -> dict[str, Any]:
    if metric not in ("efficacy", "generalization"):
        raise ODEBFContractError("P1R43-T3 terminal panel metric differs")
    try:
        value = payload["primary"]["metrics"][metric]
    except (KeyError, TypeError) as exc:
        raise ODEBFContractError("P1R43-T3 terminal panel schema differs") from exc
    if not isinstance(value, dict):
        raise ODEBFContractError("P1R43-T3 terminal panel value differs")
    return value


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
    target_update_index: int,
    endpoint_label: str,
) -> tuple[dict[str, Any], float]:
    if target_update_index not in P1R43_T3_ENDPOINT_UPDATES or endpoint_label not in (
        "P1R43_T8",
        "P1R43_T24",
    ):
        raise ODEBFContractError("P1R43-T3 endpoint identity differs")
    lookup_positions, patched_rows, lookup_receipt = _heldout_additive_lookup_geometry(
        tokenizer, requests, cases, fact_token_strategy=hparams.fact_token
    )
    residual = common_terminal_residual_input(
        terminal_target, terminal_physical.terminal_z, request_order_sha256
    )
    snapshot_sha = canonical_hash(
        {
            "terminal_target_sha256": tensor_sha256(terminal_target),
            "terminal_physical_sha256": terminal_physical.identity_sha256,
            "writer_state": "ABSENT_TARGET_ONLY",
        }
    )
    freeze = StepwiseActionFreeze(
        variant=METHOD,
        request_order_sha256=request_order_sha256,
        rollout_sha256=action_sha256,
        snapshot_sha256=snapshot_sha,
        snapshot_index=8,
        accepted_snapshot_count=9,
        rejected_retry_count=0,
        trajectory_status=f"ACTION_FROZEN_{endpoint_label}",
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
        "schema": "ode-edit-s05-p1r43-t3-target-only-z-panel/v1",
        "instruction_id": P1R43_T3_INSTRUCTION_ID,
        "method_id": P1R43_T3_METHOD_ID,
        "action_freeze_sha256": freeze.identity(),
        "endpoint_label": endpoint_label,
        "target_update_index": target_update_index,
        "target_update_count": P1R43_T3_TARGET_UPDATE_COUNT,
        "target_horizon_tau": target_update_index * P1R24_H,
        "z_inject": raw,
        "eff_z_inject": _panel_metric(raw, "efficacy"),
        "gen_z_inject": _panel_metric(raw, "generalization"),
        "heldout_lookup": lookup_receipt,
        "z_overlay": overlay.raw_free_payload(),
        "terminal_residual": residual.raw_free_payload(),
        "heldout_gen_controller_access_count": 0,
        "heldout_efficacy_controller_access_count": 0,
        "terminal_only_evaluator": True,
        "inner_step_heldout_evaluation_count": 0,
        "native_endpoint_runtime_access_count": 0,
        "wall_seconds": elapsed,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload, elapsed


def _evaluate_t8_eff_gen_only(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    cases: Sequence[Any],
    *,
    alias: str,
    request_order_sha256: str,
    action_sha256: str,
    hparams: Any,
    target: torch.Tensor,
    physical: Any,
) -> tuple[dict[str, Any], float]:
    """Evaluate T8 efficacy/generalization without repeating W0 locality."""

    lookup_positions, patched_rows, lookup_receipt = _heldout_additive_lookup_geometry(
        tokenizer, requests, cases, fact_token_strategy=hparams.fact_token
    )
    residual = common_terminal_residual_input(
        target, physical.terminal_z, request_order_sha256
    )
    snapshot_sha = canonical_hash(
        {
            "terminal_target_sha256": tensor_sha256(target),
            "terminal_physical_sha256": physical.identity_sha256,
            "writer_state": "ABSENT_TARGET_ONLY",
            "endpoint_label": "P1R43_T8",
        }
    )
    freeze = StepwiseActionFreeze(
        variant=METHOD,
        request_order_sha256=request_order_sha256,
        rollout_sha256=action_sha256,
        snapshot_sha256=snapshot_sha,
        snapshot_index=8,
        accepted_snapshot_count=9,
        rejected_retry_count=0,
        trajectory_status="ACTION_FROZEN_P1R43_T8",
    )
    overlay = HeldoutRequestResidualActivationOverlay(
        model,
        hparams.layer_module_tmp.format(int(hparams.layers[-1])),
        residual.residual,
        lookup_positions,
        patched_rows,
    )
    llama = _is_llama(model, alias)
    device = _model_device(model)
    efficacy = []
    generalization = []
    numeric: dict[str, list[list[dict[str, float | int]]]] = {
        "efficacy": [],
        "generalization": [],
    }
    processed_tokens = 0
    started = time.perf_counter()
    with overlay, _EvaluationStateGuard(model), torch.no_grad():
        for case_index, case in enumerate(cases):
            prefixes = (case.rewrite_prompt,) + case.paraphrase_prompts
            pairs, tokens, _ = _evaluate_prefixes_pinned(
                model,
                tokenizer,
                prefixes,
                case.target_new,
                case.target_true,
                llama=llama,
                device=device,
            )
            eff = pairs[:1]
            gen = pairs[1:]
            efficacy.append(eff)
            generalization.append(gen)
            for name, values in (("efficacy", eff), ("generalization", gen)):
                numeric[name].append(
                    [
                        {
                            "request_index": case_index,
                            "prompt_index": prompt_index,
                            "nll_new": pair.target_new_nll,
                            "nll_old": pair.target_true_nll,
                            "margin": pair.target_true_nll - pair.target_new_nll,
                        }
                        for prompt_index, pair in enumerate(values)
                    ]
                )
            processed_tokens += tokens
    elapsed = time.perf_counter() - started
    eff_receipt = _metric_receipt(PrimaryMetric.EFFICACY, efficacy)
    gen_receipt = _metric_receipt(PrimaryMetric.GENERALIZATION, generalization)
    payload = {
        "schema": "ode-edit-s05-p1r43-t3-target-only-z-panel/v1",
        "instruction_id": P1R43_T3_INSTRUCTION_ID,
        "method_id": P1R43_T3_METHOD_ID,
        "action_freeze_sha256": freeze.identity(),
        "endpoint_label": "P1R43_T8",
        "target_update_index": 8,
        "target_update_count": P1R43_T3_TARGET_UPDATE_COUNT,
        "target_horizon_tau": 1.0,
        "eff_z_inject": eff_receipt.raw_free_payload(),
        "gen_z_inject": gen_receipt.raw_free_payload(),
        "numeric_vectors": numeric,
        "numeric_vectors_sha256": canonical_hash(numeric),
        "heldout_lookup": lookup_receipt,
        "z_overlay": overlay.raw_free_payload(),
        "terminal_residual": residual.raw_free_payload(),
        "heldout_gen_controller_access_count": 0,
        "heldout_efficacy_controller_access_count": 0,
        "terminal_only_evaluator": True,
        "inner_step_heldout_evaluation_count": 0,
        "native_endpoint_runtime_access_count": 0,
        "locality_evaluation_count": 0,
        "model_forward_count": BATCH_SIZE,
        "processed_token_count": processed_tokens,
        "wall_seconds": elapsed,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload, elapsed


def _run_target_case(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    case_index: int,
    case_root: Path,
    hparams: Any,
    contexts: Sequence[Sequence[str]],
    dataset_path: Path,
    mutation_lock: Any,
    touched: Mapping[str, torch.nn.Parameter],
    base_values: Mapping[str, torch.Tensor],
    expected_w0: str,
    request_microbatch_size: int,
    job_ledger: ComputeLedger,
) -> dict[str, Any]:
    if len(requests) != BATCH_SIZE:
        raise ODEBFContractError("P1R43_T3 target-only case is not B10")
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
    if (
        objective_plan.request_order_sha256 != request_order
        or capture_plan.request_order_sha256 != request_order
    ):
        raise ODEBFContractError("P1R43_T3 target-only plan order differs")
    _atomic_write_once(case_root / "raw" / "objective-plan.json", objective_plan.raw_free_payload())
    _atomic_write_once(case_root / "raw" / "capture-plan.json", capture_plan.raw_free_payload())

    counter = ModelForwardCounter(model, job_ledger)
    compute = {
        "semantic_target_forward_count": 0,
        "semantic_target_backward_count": 0,
        "semantic_target_processed_tokens": 0,
        "primary_endpoint_forward_count": 0,
        "primary_endpoint_backward_count": 0,
        "primary_endpoint_processed_tokens": 0,
        "rescue_endpoint_forward_count": 0,
        "rescue_endpoint_backward_count": 0,
        "rescue_endpoint_processed_tokens": 0,
        "endpoint_measurement_forward_count": 0,
        "endpoint_measurement_backward_count": 0,
        "endpoint_measurement_processed_tokens": 0,
        "kl_forward_count": 0,
        "kl_backward_count": 0,
        "kl_processed_tokens": 0,
        "terminal_evaluator_forward_count": "RECORDED_IN_EVALUATOR_RECEIPT",
        "writer_forward_count": 0,
        "writer_backward_count": 0,
        "materialization_count": 0,
    }
    started = time.perf_counter()
    try:
        physical = capture_scalable_physical_state(model, capture_plan, hparams)
        initial = initial_target_from_capture(physical)
        current_target = initial.target_z.clone()
        current_terminal = initial.current_terminal_z.clone()
        target_states = [current_target.clone()]
        metric = ScalableRobustSharedMetric.from_z0(current_target, request_order)
        state = P1R43ControllerState.zero(current_target)
        receipt_sha: list[str] = []
        endpoint_targets: dict[int, torch.Tensor] = {}
        entry_norm_calibration_count = 0
        selection_early = {"PRIMARY": 0, "RESCUE": 0, "CURRENT": 0}
        selection_late = {"PRIMARY": 0, "RESCUE": 0, "CURRENT": 0}
        post_t8_path_length = torch.zeros(
            current_target.shape[1], dtype=torch.float64
        )
        target_wall = 0.0
        for microstep in range(P1R43_T3_TARGET_UPDATE_COUNT):
            step_started = time.perf_counter()
            variable = (
                current_target.detach()
                .to(device=next(model.parameters()).device, dtype=torch.float32)
                .clone()
                .requires_grad_(True)
            )
            semantic_result = evaluate_scalable_target_new_objective(
                model,
                objective_plan,
                target_state=variable,
                current_terminal=current_terminal,
                target_layer_name=hparams.layer_module_tmp.format(int(hparams.layers[-1])),
            )
            if semantic_result.target_gradient is None:
                raise ODEBFContractError("P1R43-T3 semantic gradient is absent")
            compute["semantic_target_forward_count"] += semantic_result.model_forward_count
            compute["semantic_target_backward_count"] += semantic_result.backward_count
            compute["semantic_target_processed_tokens"] += semantic_result.processed_token_count
            proposal = prepare_p1r43_target_proposal(
                current_target,
                current_terminal,
                semantic_result,
                state,
                alias=alias,
                step_index=microstep,
                shared_speed=float(metric.shared_speed),
                max_target_updates=P1R43_T3_TARGET_UPDATE_COUNT,
            )
            primary_state = proposal.primary_step.target_next.detach().to(
                device=next(model.parameters()).device, dtype=torch.float32
            )
            primary_endpoint = evaluate_scalable_target_new_objective(
                model,
                objective_plan,
                target_state=primary_state,
                current_terminal=current_terminal,
                target_layer_name=hparams.layer_module_tmp.format(int(hparams.layers[-1])),
                target_gradient_required=False,
            )
            compute["primary_endpoint_forward_count"] += primary_endpoint.model_forward_count
            compute["primary_endpoint_backward_count"] += primary_endpoint.backward_count
            compute["primary_endpoint_processed_tokens"] += primary_endpoint.processed_token_count
            rescue = prepare_p1r43_rescue_proposal(
                proposal,
                current_target,
                current_terminal,
                semantic_result,
                primary_endpoint,
                step_index=microstep,
            )
            rescue_endpoint = None
            if rescue.rescue_step is not None:
                rescue_state = rescue.rescue_step.target_next.detach().to(
                    device=next(model.parameters()).device, dtype=torch.float32
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
                compute["rescue_endpoint_forward_count"] += (
                    rescue_endpoint.model_forward_count
                )
                compute["rescue_endpoint_backward_count"] += rescue_endpoint.backward_count
                compute["rescue_endpoint_processed_tokens"] += (
                    rescue_endpoint.processed_token_count
                )
            selected = select_p1r43_target_proposal(
                proposal,
                rescue,
                current_target,
                current_terminal,
                semantic_result,
                primary_endpoint,
                rescue_endpoint,
                step_index=microstep,
            )
            target_wall += time.perf_counter() - step_started
            selected_target = selected.target_step.target_next
            selected_delta = selected_target.to(dtype=torch.float64) - current_target.to(
                dtype=torch.float64
            )
            if microstep >= 8:
                post_t8_path_length += torch.linalg.vector_norm(selected_delta, dim=0)
            request_count = len(requests)
            entry_norm = torch.tensor(
                selected.next_state.entry_semantic_gradient_norm, dtype=torch.float64
            )
            current_norm = torch.tensor(
                proposal.receipt["current_semantic_gradient_norm_by_request"],
                dtype=torch.float64,
            )
            selections = list(selected.receipt["selection_by_request"])
            bucket = selection_early if microstep < 8 else selection_late
            for value in selections:
                bucket[value] += 1
            entry_norm_calibration_count += int(
                proposal.receipt["entry_norm_calibration_count"]
            )
            receipt = {
                **dict(selected.receipt),
                "schema": "ode-edit-s05-p1r43-t3-target-update/v1",
                "instruction_id": P1R43_T3_INSTRUCTION_ID,
                "method_id": P1R43_T3_METHOD_ID,
                "target_update_index": microstep,
                "target_update_number": microstep + 1,
                "target_update_count": P1R43_T3_TARGET_UPDATE_COUNT,
                "target_horizon_tau_after_update": (microstep + 1) * P1R24_H,
                "h": P1R24_H,
                "current_full_six_target_nll_by_request": list(
                    semantic_result.per_request_values
                ),
                "current_full_six_target_nll_mean": semantic_result.loss,
                "primary_endpoint_nll_by_request": list(
                    primary_endpoint.per_request_values
                ),
                "rescue_endpoint_nll_by_request": (
                    None
                    if rescue_endpoint is None
                    else list(rescue_endpoint.per_request_values)
                ),
                "selected_endpoint_nll_by_request": list(
                    selected.selected_endpoint.per_request_values
                ),
                "current_entry_gradient_norm_ratio_by_request": [
                    float(item) for item in current_norm / entry_norm
                ],
                "selected_displacement_norm_by_request": [
                    float(item)
                    for item in torch.linalg.vector_norm(selected_delta, dim=0)
                ],
                "semantic_target_forward_count": semantic_result.model_forward_count,
                "semantic_target_backward_count": semantic_result.backward_count,
                "primary_endpoint_forward_count": primary_endpoint.model_forward_count,
                "rescue_endpoint_forward_count": (
                    0 if rescue_endpoint is None else rescue_endpoint.model_forward_count
                ),
                "logical_batch_request_count": request_count,
                "request_independent_gradient_unscale_factor": request_count,
                "controller_state_reset_count": 1 if microstep == 0 else 0,
                "controller_state_reset_at_8_count": 0,
                "controller_state_reset_at_16_count": 0,
                "fixed_W0_target_state": True,
                "writer_update_count": 0,
                "writer_materialization_count": 0,
                "virtual_W_update_count": 0,
                "native_endpoint_runtime_access_count": 0,
                "heldout_controller_access_count": 0,
                "p2r1_rms_influence_count": 0,
                "p2r1_tangent_influence_count": 0,
                "target_kl_decision_influence_count": 0,
                "target_decay_decision_influence_count": 0,
                "target_gamma_decision_influence_count": 0,
                "target_clamp_decision_influence_count": 0,
            }
            receipt["identity_sha256"] = canonical_hash(receipt)
            receipt_sha.append(
                _atomic_write_once(
                    case_root / "raw" / "target" / f"microstep-{microstep:02d}.json",
                    receipt,
                )
            )
            current_target = selected_target
            state = selected.next_state
            target_states.append(current_target.clone())
            if microstep + 1 in P1R43_T3_ENDPOINT_UPDATES:
                endpoint_targets[microstep + 1] = current_target.clone()
        if (
            len(target_states) != P1R43_T3_TARGET_UPDATE_COUNT + 1
            or set(endpoint_targets) != set(P1R43_T3_ENDPOINT_UPDATES)
            or entry_norm_calibration_count != 1
            or state.entry_semantic_gradient_norm is None
        ):
            raise ODEBFStateError("P1R43-T3 target trajectory state differs")
        if tensor_sha256(endpoint_targets[8]) == tensor_sha256(endpoint_targets[24]):
            raise ODEBFStateError("P1R43-T3 T8/T24 endpoint hashes do not differ")
        endpoint_objectives: dict[int, Any] = {}
        for endpoint_index in P1R43_T3_ENDPOINT_UPDATES:
            endpoint_objective = evaluate_scalable_target_new_objective(
                model,
                objective_plan,
                target_state=endpoint_targets[endpoint_index].detach().to(
                    device=next(model.parameters()).device, dtype=torch.float32
                ),
                current_terminal=current_terminal,
                target_layer_name=hparams.layer_module_tmp.format(
                    int(hparams.layers[-1])
                ),
                target_gradient_required=False,
            )
            compute["endpoint_measurement_forward_count"] += (
                endpoint_objective.model_forward_count
            )
            compute["endpoint_measurement_backward_count"] += endpoint_objective.backward_count
            compute["endpoint_measurement_processed_tokens"] += (
                endpoint_objective.processed_token_count
            )
            endpoint_objectives[endpoint_index] = endpoint_objective
        trajectory_file_sha, trajectory_tensor_sha = _write_private_target_trajectory_once(
            case_root / "private-target" / "target-states-0-24.pt",
            target_states=target_states,
            request_order_sha256=request_order,
        )
        action = {
            "schema": "ode-edit-s05-p1r43-t3-target-only-action-freeze/v1",
            "instruction_id": P1R43_T3_INSTRUCTION_ID,
            "method_id": P1R43_T3_METHOD_ID,
            "case_index": case_index,
            "alias": alias,
            "request_order_sha256": request_order,
            "microstep_receipt_sha256": receipt_sha,
            "target_update_count": P1R43_T3_TARGET_UPDATE_COUNT,
            "target_update_indices": list(range(P1R43_T3_TARGET_UPDATE_COUNT)),
            "target_snapshot_indices": [0, 8, 24],
            "target_endpoint_sha256": {
                "P1R43_T8": tensor_sha256(endpoint_targets[8]),
                "P1R43_T24": tensor_sha256(endpoint_targets[24]),
            },
            "target_endpoint_horizon_tau": {"P1R43_T8": 1.0, "P1R43_T24": 3.0},
            "evaluator_snapshot_index": 8,
            "evaluator_accepted_snapshot_count": 9,
            "evaluator_fixed_budget_slots_completed": 8,
            "h": P1R24_H,
            "entry_gradient_norm_calibration_count": entry_norm_calibration_count,
            "controller_state_reset_count": 1,
            "controller_reset_at_8_count": 0,
            "controller_reset_at_16_count": 0,
            "writer_update_count": 0,
            "writer_materialization_count": 0,
            "virtual_W_update_count": 0,
            "terminal_target_sha256": tensor_sha256(endpoint_targets[24]),
            "trajectory_tensor_sha256": trajectory_tensor_sha,
            "W0_sha256": _model_w0_contract(touched),
            "actions_frozen_before_evaluator": True,
            "native_endpoint_runtime_access_count": 0,
            "heldout_controller_access_count": 0,
            "retry_count": 0,
            "persistent_freeze_count": 0,
            "P2R1_operator_influence_count": 0,
            "target_KL_decay_RMS_tangent_gamma_clamp_influence_count": 0,
        }
        action["identity_sha256"] = canonical_hash(action)
        freeze_sha = _atomic_write_once(case_root / "action-freeze.json", action)
        cases, _ = _action_frozen_cases(
            dataset_path,
            requests,
            arm=METHOD,
            selected_snapshot_sha256=freeze_sha,
            fixed_budget_slots_completed=8,
        )
        endpoint_panels: dict[int, dict[str, Any]] = {}
        evaluator_wall = 0.0
        panel8, wall8 = _evaluate_t8_eff_gen_only(
            model,
            tokenizer,
            requests,
            cases,
            alias=alias,
            request_order_sha256=request_order,
            action_sha256=action["identity_sha256"],
            hparams=hparams,
            target=endpoint_targets[8],
            physical=physical,
        )
        panel24, wall24 = _evaluate_terminal_z_panel(
            model,
            tokenizer,
            requests,
            cases,
            alias=alias,
            request_order_sha256=request_order,
            action_sha256=action["identity_sha256"],
            hparams=hparams,
            terminal_target=endpoint_targets[24],
            terminal_physical=physical,
            target_update_index=24,
            endpoint_label="P1R43_T24",
        )
        panel24["locality_evaluation_count"] = 1
        panel24["identity_sha256"] = canonical_hash(
            {key: value for key, value in panel24.items() if key != "identity_sha256"}
        )
        endpoint_panels = {8: panel8, 24: panel24}
        evaluator_wall = wall8 + wall24
    finally:
        counter.close()
    if _model_w0_contract(touched) != expected_w0:
        raise ODEBFStateError("P1R43-T3 target-only path mutated W0")
    restore = _restore_exact_w0(
        touched,
        base_values,
        mutation_lock=mutation_lock,
        expected_contract=expected_w0,
    )
    terminal = {
        "schema": "ode-edit-s05-p1r43-t3-target-only-case-terminal/v1",
        "instruction_id": P1R43_T3_INSTRUCTION_ID,
        "method_id": P1R43_T3_METHOD_ID,
        "case_index": case_index,
        "alias": alias,
        "request_count": BATCH_SIZE,
        "request_order_sha256": request_order,
        "objective_plan_sha256": objective_plan.identity_sha256,
        "capture_plan_sha256": capture_plan.identity_sha256,
        "action_freeze_sha256": freeze_sha,
        "target_endpoint_sha256": {
            "P1R43_T8": tensor_sha256(endpoint_targets[8]),
            "P1R43_T24": tensor_sha256(endpoint_targets[24]),
        },
        "target_trajectory_file_sha256": trajectory_file_sha,
        "target_trajectory_tensor_sha256": trajectory_tensor_sha,
        "endpoint_full_six_target_new_nll": {
            "P1R43_T8": endpoint_objectives[8].loss,
            "P1R43_T24": endpoint_objectives[24].loss,
        },
        "endpoint_full_six_target_new_nll_by_request": {
            "P1R43_T8": list(endpoint_objectives[8].per_request_values),
            "P1R43_T24": list(endpoint_objectives[24].per_request_values),
        },
        "endpoint_z_panels": {
            "P1R43_T8": endpoint_panels[8],
            "P1R43_T24": endpoint_panels[24],
        },
        "target_update_count": P1R43_T3_TARGET_UPDATE_COUNT,
        "target_snapshot_indices": [0, 8, 24],
        "h": P1R24_H,
        "entry_gradient_norm_calibration_count": entry_norm_calibration_count,
        "controller_state_reset_count": 1,
        "controller_reset_at_8_count": 0,
        "controller_reset_at_16_count": 0,
        "selection_counts_updates_0_7": selection_early,
        "selection_counts_updates_8_23": selection_late,
        "post_t8_path_length_by_request": [
            float(item) for item in post_t8_path_length
        ],
        "z24_minus_z8_norm_by_request": [
            float(item)
            for item in torch.linalg.vector_norm(
                endpoint_targets[24].to(dtype=torch.float64)
                - endpoint_targets[8].to(dtype=torch.float64),
                dim=0,
            )
        ],
        "writer_update_count": 0,
        "writer_materialization_count": 0,
        "virtual_W_update_count": 0,
        "native_endpoint_runtime_access_count": 0,
        "heldout_controller_access_count": 0,
        "retry_count": 0,
        "persistent_freeze_count": 0,
        "P2R1_operator_influence_count": 0,
        "target_KL_decay_RMS_tangent_gamma_clamp_influence_count": 0,
        "target_kl_forward_count": 0,
        "target_kl_backward_count": 0,
        "target_decay_access_count": 0,
        "target_clamp_count": 0,
        "compute": compute,
        "target_wall_seconds": target_wall,
        "terminal_evaluator_wall_seconds": evaluator_wall,
        "total_wall_seconds": time.perf_counter() - started,
        "W0_restore": restore,
        "W0_restored": True,
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(case_root / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p1r43-t3-target-only-case-manifest/v1",
        "case_index": case_index,
        "alias": alias,
        "terminal_sha256": terminal_sha,
        "action_freeze_sha256": freeze_sha,
        "target_trajectory_file_sha256": trajectory_file_sha,
        "target_update_count": P1R43_T3_TARGET_UPDATE_COUNT,
        "target_snapshot_indices": [0, 8, 24],
        "writer_materialization_count": 0,
        "W0_restored": True,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(case_root / "manifest.json", manifest)
    return {
        "status": "P1R43_T3_TARGET_ONLY_CASE_COMPLETE",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "target_endpoint_sha256": {
            "P1R43_T8": tensor_sha256(endpoint_targets[8]),
            "P1R43_T24": tensor_sha256(endpoint_targets[24]),
        },
    }


def run_p1r43_t3_target_only(
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
    contexts: Sequence[Sequence[str]],
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
        raise ODEBFContractError("P1R43-T3 target-only case inventory differs")
    if stream.get("root_digest") != STREAM_ROOT or stream.get("all_request_order_sha256") != STREAM_ORDER:
        raise ODEBFContractError("P1R43-T3 frozen stream identity differs")
    if any(len(batch) != BATCH_SIZE for batch in stream_batches):
        raise ODEBFContractError("P1R43-T3 target-only stream is not B10x10")
    expected_w0 = _model_w0_contract(touched)
    if _hashes(touched) != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("P1R43-T3 entry W0 differs")
    started = time.perf_counter()
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for case_index, requests in enumerate(stream_batches[:case_count], start=1):
        seed_all(COMMON_SEED)
        if _model_w0_contract(touched) != expected_w0:
            raise ODEBFStateError("P1R43-T3 cross-case W0 state leak detected")
        case_root = raw_root / "cases" / f"case-{case_index:02d}"
        try:
            completed.append(
                {
                    "case_index": case_index,
                    **_run_target_case(
                        model,
                        tokenizer,
                        requests,
                        alias=alias,
                        case_index=case_index,
                        case_root=case_root,
                        hparams=hparams,
                        contexts=contexts,
                        dataset_path=dataset_path,
                        mutation_lock=mutation_lock,
                        touched=touched,
                        base_values=base_values,
                        expected_w0=expected_w0,
                        request_microbatch_size=request_microbatch_size,
                        job_ledger=job_ledger,
                    ),
                }
            )
        except Exception as exc:
            restore = _restore_exact_w0(
                touched,
                base_values,
                mutation_lock=mutation_lock,
                expected_contract=expected_w0,
            )
            failure = {
                "schema": "ode-edit-s05-p1r43-t3-target-only-case-failure/v1",
                "case_index": case_index,
                "alias": alias,
                "classification": "TECHNICAL_FAIL",
                "exception_class": type(exc).__name__,
                "exception_message_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
                "retry_count": 0,
                "W0_restore": restore,
                "next_case_continues": case_count == 10,
            }
            failure["identity_sha256"] = canonical_hash(failure)
            _atomic_write_once(case_root / "failure.json", failure)
            failed.append(failure)
        if _model_w0_contract(touched) != expected_w0:
            raise ODEBFStateError("P1R43-T3 post-case W0 differs")
        stages.record(
            f"post_p1r43_t3_target_only_case_{case_index}",
            {
                "case_index": case_index,
                "complete_count": len(completed),
                "failure_count": len(failed),
                "W0_restored": True,
                "writer_update_count": 0,
                "writer_materialization_count": 0,
            },
        )
    terminal = {
        "schema": "ode-edit-s05-p1r43-t3-target-only-job-terminal/v1",
        "instruction_id": P1R43_T3_INSTRUCTION_ID,
        "method_id": P1R43_T3_METHOD_ID,
        "source_head": source_head,
        "alias": alias,
        "method": METHOD,
        "case_count": case_count,
        "request_attempt_count": case_count * BATCH_SIZE,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "completed": completed,
        "failed_case_identity_sha256": [item["identity_sha256"] for item in failed],
        "target_update_count_per_complete_case": P1R43_T3_TARGET_UPDATE_COUNT,
        "target_snapshot_indices": [0, 8, 24],
        "h": P1R24_H,
        "writer_update_count": 0,
        "writer_materialization_count": 0,
        "native_endpoint_runtime_access_count": 0,
        "heldout_controller_access_count": 0,
        "retry_count": 0,
        "W0_restored": _model_w0_contract(touched) == expected_w0,
        "job_compute": job_ledger.raw_free_payload(),
        "total_wall_seconds": time.perf_counter() - started,
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p1r43-t3-target-only-job-manifest/v1",
        "source_head": source_head,
        "alias": alias,
        "case_count": case_count,
        "terminal_sha256": terminal_sha,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "W0_restored": terminal["W0_restored"],
        "writer_materialization_count": 0,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    return {
        "status": "P1R43_T3_TARGET_ONLY_TERMINAL",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "W0_restored": terminal["W0_restored"],
    }


__all__ = [
    "CASE_COUNTS",
    "METHOD",
    "expected_p1r43_t3_target_result_name",
    "run_p1r43_t3_target_only",
]
