"""Independent target-only Gate A runtime for P2R1."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from .accounting import ComputeLedger
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
from .p0_runtime import ModelForwardCounter
from .p1_runtime import _atomic_write_once
from .p1_scalable_batched_experiment import (
    _action_frozen_cases,
    _model_w0_contract,
)
from .p1_state import ArmWeightSnapshot
from .p1_stepwise import StepwiseActionFreeze, evaluate_counterfact_stepwise_primary
from .p1r24_atomic_strength import (
    P1R24AliasTargetLock,
    build_p1r24_kl_plan,
    evaluate_p1r24_kl,
    verify_p1r24_alphaedit_geometry,
)
from .p1r36_independent_b10x10_runtime import _hashes, _restore_exact_w0
from .p1r43_independent_b10x10_runtime import STREAM_ORDER, STREAM_ROOT
from .p2r1_rms_tangent_target import (
    P2R1_INSTRUCTION_ID,
    P2R1_METHOD_ID,
    P2R1RMSState,
    P2R1_TARGET_MICROSTEP_COUNT,
    p2r1_preservation_gradient,
    p2r1_target_update,
)
from .scalable_batched_model import (
    build_scalable_capture_plan,
    build_scalable_objective_plan,
    capture_scalable_physical_state,
    evaluate_scalable_target_new_objective,
)
from .scalable_batched_runtime import initial_target_from_capture, scalable_ordered_request_digest


METHOD = "P2R1-RMS-TANGENT-TARGET-ONLY"
CASE_COUNTS = (1, 10)


def expected_p2r1_target_result_name(
    alias: str,
    *,
    case_count: int,
    attempt_suffix: str | None = None,
) -> str:
    if alias not in ("llama3-8b-inst", "qwen2.5-7b-inst") or case_count not in CASE_COUNTS:
        raise ODEBFContractError("P2R1 target result identity differs")
    phase = "first-b10" if case_count == 1 else "b10x10"
    suffix = f"-{attempt_suffix}" if attempt_suffix else ""
    return f"s05-p2r1-rms-tangent-target-only-{phase}-{alias}{suffix}-v1"


def _write_private_target_trajectory_once(
    path: Path, *, k_states: Sequence[torch.Tensor], request_order_sha256: str
) -> tuple[str, str]:
    if len(k_states) != 9:
        raise ODEBFContractError("P2R1 K-state target inventory differs")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(
                {
                    "schema": "ode-edit-s05-p2r1-private-kstate-target/v1",
                    "request_order_sha256": request_order_sha256,
                    "targets": torch.stack(
                        [item.detach().cpu().to(torch.float32) for item in k_states]
                    ),
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        raise
    raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest(), tensor_sha256(torch.stack(k_states))


def _panel_metric(payload: Mapping[str, Any], metric: str) -> dict[str, Any]:
    if metric not in ("efficacy", "generalization"):
        raise ODEBFContractError("P2R1 terminal panel metric differs")
    try:
        value = payload["primary"]["metrics"][metric]
    except (KeyError, TypeError) as exc:
        raise ODEBFContractError("P2R1 terminal panel schema differs") from exc
    if not isinstance(value, dict):
        raise ODEBFContractError("P2R1 terminal panel value differs")
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
) -> tuple[dict[str, Any], float]:
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
        snapshot_index=P2R1_TARGET_MICROSTEP_COUNT,
        accepted_snapshot_count=P2R1_TARGET_MICROSTEP_COUNT + 1,
        rejected_retry_count=0,
        trajectory_status="ACTION_FROZEN_P2R1_TARGET_N24",
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
        "schema": "ode-edit-s05-p2r1-target-only-terminal-z-panel/v1",
        "instruction_id": P2R1_INSTRUCTION_ID,
        "method_id": P2R1_METHOD_ID,
        "action_freeze_sha256": freeze.identity(),
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
        raise ODEBFContractError("P2R1 target-only case is not B10")
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
        raise ODEBFContractError("P2R1 target-only plan order differs")
    _atomic_write_once(case_root / "raw" / "objective-plan.json", objective_plan.raw_free_payload())
    _atomic_write_once(case_root / "raw" / "capture-plan.json", capture_plan.raw_free_payload())

    lock = P1R24AliasTargetLock.for_alias(alias)
    alpha_geometry = verify_p1r24_alphaedit_geometry(
        hparams, lock, easyedit_root=Path("/mnt/raid5/janghj/EasyEdit")
    )
    counter = ModelForwardCounter(model, job_ledger)
    compute = {
        "target_forward_count": 0,
        "target_backward_count": 0,
        "target_processed_tokens": 0,
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
        target_origin = initial.target_z.clone()
        current_terminal = initial.current_terminal_z.clone()
        k_states = [current_target.clone()]
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
        compute["kl_processed_tokens"] += teacher_result.processed_token_count
        state = P2R1RMSState.zero()
        receipt_sha: list[str] = []
        clamp_hits = 0
        target_wall = 0.0
        for microstep in range(P2R1_TARGET_MICROSTEP_COUNT):
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
            kl_result, _ = evaluate_p1r24_kl(
                model,
                kl_plan,
                teacher_log_probs=teacher,
                target_state=variable,
                current_terminal=current_terminal,
                target_layer_name=hparams.layer_module_tmp.format(int(hparams.layers[-1])),
            )
            if semantic_result.target_gradient is None or kl_result.gradient is None:
                raise ODEBFContractError("P2R1 target gradients are absent")
            compute["target_forward_count"] += semantic_result.model_forward_count
            compute["target_backward_count"] += semantic_result.backward_count
            compute["target_processed_tokens"] += semantic_result.processed_token_count
            compute["kl_forward_count"] += kl_result.model_forward_count
            compute["kl_backward_count"] += kl_result.backward_count
            compute["kl_processed_tokens"] += kl_result.processed_token_count
            request_count = len(requests)
            semantic_gradient = semantic_result.target_gradient * request_count
            kl_gradient = kl_result.gradient * request_count
            preservation, decay_values, decay_gradient = p2r1_preservation_gradient(
                current_target, target_origin, kl_gradient, lock
            )
            update = p2r1_target_update(
                current_target,
                target_origin,
                semantic_gradient,
                preservation,
                state,
                alias=alias,
                microstep_index=microstep,
                lock=lock,
            )
            target_wall += time.perf_counter() - step_started
            receipt = {
                **dict(update.receipt),
                "target_new_nll_by_request": list(semantic_result.per_request_values),
                "target_new_nll_mean": semantic_result.loss,
                "kl_by_request": list(kl_result.per_request_values),
                "kl_mean": kl_result.loss,
                "decay_by_request": [float(item) for item in decay_values],
                "decay_gradient_sha256": tensor_sha256(decay_gradient),
                "target_objective_forward_count": semantic_result.model_forward_count,
                "target_objective_backward_count": semantic_result.backward_count,
                "kl_forward_count": kl_result.model_forward_count,
                "kl_backward_count": kl_result.backward_count,
                "logical_batch_request_count": request_count,
                "request_independent_gradient_unscale_factor": request_count,
            }
            receipt["identity_sha256"] = canonical_hash(receipt)
            receipt_sha.append(
                _atomic_write_once(
                    case_root / "raw" / "target" / f"microstep-{microstep:02d}.json",
                    receipt,
                )
            )
            clamp_hits += int(receipt["clamp_hit_count"])
            current_target = update.target_next
            state = update.state_next
            if (microstep + 1) % 3 == 0:
                k_states.append(current_target.clone())
        if state.completed_microsteps != P2R1_TARGET_MICROSTEP_COUNT or len(k_states) != 9:
            raise ODEBFStateError("P2R1 target trajectory count differs")
        final_target_state = current_target.detach().to(
            device=next(model.parameters()).device, dtype=torch.float32
        )
        final_objective = evaluate_scalable_target_new_objective(
            model,
            objective_plan,
            target_state=final_target_state,
            current_terminal=current_terminal,
            target_layer_name=hparams.layer_module_tmp.format(int(hparams.layers[-1])),
            target_gradient_required=False,
        )
        compute["target_forward_count"] += final_objective.model_forward_count
        compute["target_processed_tokens"] += final_objective.processed_token_count
        trajectory_file_sha, trajectory_tensor_sha = _write_private_target_trajectory_once(
            case_root / "private-target" / "kstate-targets.pt",
            k_states=k_states,
            request_order_sha256=request_order,
        )
        action = {
            "schema": "ode-edit-s05-p2r1-target-only-action-freeze/v1",
            "instruction_id": P2R1_INSTRUCTION_ID,
            "method_id": P2R1_METHOD_ID,
            "case_index": case_index,
            "alias": alias,
            "request_order_sha256": request_order,
            "microstep_receipt_sha256": receipt_sha,
            "target_microstep_count": P2R1_TARGET_MICROSTEP_COUNT,
            "outer_state_count": 8,
            "writer_update_count": 0,
            "writer_materialization_count": 0,
            "terminal_target_sha256": tensor_sha256(current_target),
            "trajectory_tensor_sha256": trajectory_tensor_sha,
            "W0_sha256": _model_w0_contract(touched),
            "actions_frozen_before_evaluator": True,
            "native_endpoint_runtime_access_count": 0,
            "heldout_controller_access_count": 0,
            "retry_count": 0,
            "hold_count": 0,
        }
        action["identity_sha256"] = canonical_hash(action)
        freeze_sha = _atomic_write_once(case_root / "action-freeze.json", action)
        cases, _ = _action_frozen_cases(
            dataset_path,
            requests,
            arm=METHOD,
            selected_snapshot_sha256=freeze_sha,
            fixed_budget_slots_completed=P2R1_TARGET_MICROSTEP_COUNT,
        )
        terminal_panel, evaluator_wall = _evaluate_terminal_z_panel(
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
        )
    finally:
        counter.close()
    if _model_w0_contract(touched) != expected_w0:
        raise ODEBFStateError("P2R1 target-only path mutated W0")
    restore = _restore_exact_w0(
        touched,
        base_values,
        mutation_lock=mutation_lock,
        expected_contract=expected_w0,
    )
    terminal = {
        "schema": "ode-edit-s05-p2r1-target-only-case-terminal/v1",
        "instruction_id": P2R1_INSTRUCTION_ID,
        "method_id": P2R1_METHOD_ID,
        "case_index": case_index,
        "alias": alias,
        "request_count": BATCH_SIZE,
        "request_order_sha256": request_order,
        "objective_plan_sha256": objective_plan.identity_sha256,
        "capture_plan_sha256": capture_plan.identity_sha256,
        "alpha_geometry": alpha_geometry,
        "action_freeze_sha256": freeze_sha,
        "terminal_target_sha256": tensor_sha256(current_target),
        "target_trajectory_file_sha256": trajectory_file_sha,
        "target_trajectory_tensor_sha256": trajectory_tensor_sha,
        "terminal_full_six_target_new_nll": final_objective.loss,
        "terminal_full_six_target_new_nll_by_request": list(final_objective.per_request_values),
        "terminal_z_panel": terminal_panel,
        "target_microstep_count": P2R1_TARGET_MICROSTEP_COUNT,
        "outer_state_count": 8,
        "writer_update_count": 0,
        "writer_materialization_count": 0,
        "clamp_hit_count": clamp_hits,
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
        "schema": "ode-edit-s05-p2r1-target-only-case-manifest/v1",
        "case_index": case_index,
        "alias": alias,
        "terminal_sha256": terminal_sha,
        "action_freeze_sha256": freeze_sha,
        "target_trajectory_file_sha256": trajectory_file_sha,
        "target_microstep_count": P2R1_TARGET_MICROSTEP_COUNT,
        "writer_materialization_count": 0,
        "W0_restored": True,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(case_root / "manifest.json", manifest)
    return {
        "status": "P2R1_TARGET_ONLY_CASE_COMPLETE",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "terminal_target_sha256": tensor_sha256(current_target),
    }


def run_p2r1_target_only(
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
        raise ODEBFContractError("P2R1 target-only case inventory differs")
    if stream.get("root_digest") != STREAM_ROOT or stream.get("all_request_order_sha256") != STREAM_ORDER:
        raise ODEBFContractError("P2R1 frozen stream identity differs")
    if any(len(batch) != BATCH_SIZE for batch in stream_batches):
        raise ODEBFContractError("P2R1 target-only stream is not B10x10")
    expected_w0 = _model_w0_contract(touched)
    if _hashes(touched) != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("P2R1 entry W0 differs")
    started = time.perf_counter()
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for case_index, requests in enumerate(stream_batches[:case_count], start=1):
        seed_all(COMMON_SEED)
        if _model_w0_contract(touched) != expected_w0:
            raise ODEBFStateError("P2R1 cross-case W0 state leak detected")
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
                "schema": "ode-edit-s05-p2r1-target-only-case-failure/v1",
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
            raise ODEBFStateError("P2R1 post-case W0 differs")
        stages.record(
            f"post_p2r1_target_only_case_{case_index}",
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
        "schema": "ode-edit-s05-p2r1-target-only-job-terminal/v1",
        "instruction_id": P2R1_INSTRUCTION_ID,
        "method_id": P2R1_METHOD_ID,
        "source_head": source_head,
        "alias": alias,
        "method": METHOD,
        "case_count": case_count,
        "request_attempt_count": case_count * BATCH_SIZE,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "completed": completed,
        "failed_case_identity_sha256": [item["identity_sha256"] for item in failed],
        "target_microstep_count_per_complete_case": P2R1_TARGET_MICROSTEP_COUNT,
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
        "schema": "ode-edit-s05-p2r1-target-only-job-manifest/v1",
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
        "status": "P2R1_TARGET_ONLY_TERMINAL",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "W0_restored": terminal["W0_restored"],
    }


__all__ = [
    "CASE_COUNTS",
    "METHOD",
    "expected_p2r1_target_result_name",
    "run_p2r1_target_only",
]
