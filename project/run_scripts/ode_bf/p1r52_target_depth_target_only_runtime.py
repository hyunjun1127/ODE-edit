"""P1R52 IL1/IL3-FULL frozen-W target-only causal runtime."""

from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from .accounting import ComputeLedger
from .alpha_backend import seed_all
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
    _factor_inventory_identity,
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
from .p1r43_independent_b10x10_runtime import STREAM_ORDER, STREAM_ROOT
from .p1r51_requestwise_semantic_allocation import P1R51ControllerState
from .p1r52_target_depth import (
    P1R52_TARGET_DEPTH_INSTRUCTION_ID,
    P1R52_TARGET_DEPTH_METHOD_ID,
    P1R52TargetDepth,
    run_p1r52_target_depth_scheduler,
)
from .p2r1_target_only_runtime import (
    _evaluate_terminal_z_panel,
    _write_private_target_trajectory_once,
)
from .scalable_batched_field import scalable_metric_from_allocation
from .scalable_batched_model import (
    build_scalable_capture_plan,
    build_scalable_objective_plan,
    capture_scalable_physical_state,
    evaluate_scalable_target_new_objective,
)
from .scalable_batched_runtime import initial_target_from_capture, scalable_ordered_request_digest


CASE_COUNTS = (1, 10)


def expected_p1r52_target_depth_result_name(
    alias: str,
    *,
    depth: P1R52TargetDepth | str,
    case_count: int,
    attempt_suffix: str | None = None,
) -> str:
    policy = depth if isinstance(depth, P1R52TargetDepth) else P1R52TargetDepth(depth)
    if alias not in ("llama3-8b-inst", "qwen2.5-7b-inst") or case_count not in CASE_COUNTS:
        raise ODEBFContractError("P1R52 target-depth result identity differs")
    phase = "b1" if case_count == 1 else "b10x10"
    depth_token = policy.value.lower().replace("-", "")
    suffix = f"-{attempt_suffix}" if attempt_suffix else ""
    return f"s05-p1r52-target-depth-target-only-{phase}-{alias}-{depth_token}{suffix}-v1"


def _run_target_case(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    depth: P1R52TargetDepth,
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
        raise ODEBFContractError("P1R52 target-depth target-only case is not B10")
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
        raise ODEBFContractError("P1R52 target-depth target-only plan order differs")
    _atomic_write_once(
        case_root / "raw" / "objective-plan.json", objective_plan.raw_free_payload()
    )
    _atomic_write_once(
        case_root / "raw" / "capture-plan.json", capture_plan.raw_free_payload()
    )

    lock = P1R24AliasTargetLock.for_alias(alias)
    alpha_geometry = verify_p1r24_alphaedit_geometry(
        hparams, lock, easyedit_root=Path("/mnt/raid5/janghj/EasyEdit")
    )
    compute: dict[str, int | float] = {
        "semantic_forward_count": 0,
        "semantic_backward_count": 0,
        "semantic_processed_tokens": 0,
        "kl_forward_count": 0,
        "kl_backward_count": 0,
        "kl_processed_tokens": 0,
        "primary_endpoint_forward_count": 0,
        "rescue_endpoint_forward_count": 0,
        "inner_writer_materialization_count": 0,
        "outer_writer_materialization_count": 0,
        "inner_heldout_access_count": 0,
        "entry_norm_calibration_count": 0,
    }
    started = time.perf_counter()
    counter = ModelForwardCounter(model, job_ledger)
    try:
        physical = capture_scalable_physical_state(model, capture_plan, hparams)
        initial = initial_target_from_capture(physical)
        current_target = initial.target_z.clone()
        target_origin = current_target.clone()
        current_terminal = initial.current_terminal_z.clone()
        metric = scalable_metric_from_allocation(
            current_target, request_order, "RS"
        )
        state = P1R51ControllerState.zero(current_target)
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
        teacher_sha = canonical_hash(
            {
                "teacher_tensor_sha256": [tensor_sha256(value) for value in teacher],
                "request_order_sha256": request_order,
                "capture_count": 1,
            }
        )
        compute["kl_forward_count"] += teacher_result.model_forward_count
        compute["kl_processed_tokens"] += teacher_result.processed_token_count
        empty_factor_sha = _factor_inventory_identity({})
        history_cache_sha = canonical_hash(
            {"history": "OFF", "cache": "TARGET_ONLY_ABSENT"}
        )
        outer_receipts: list[str] = []
        inner_receipts: list[str] = []
        target_wall = 0.0
        for outer_step in range(8):
            outer_started = time.perf_counter()

            def evaluate_target(target: torch.Tensor) -> Any:
                variable = (
                    target.detach()
                    .to(device=next(model.parameters()).device, dtype=torch.float32)
                    .clone()
                    .requires_grad_(True)
                )
                result = evaluate_scalable_target_new_objective(
                    model,
                    objective_plan,
                    target_state=variable,
                    current_terminal=current_terminal,
                    target_layer_name=hparams.layer_module_tmp.format(
                        int(hparams.layers[-1])
                    ),
                )
                if result.target_gradient is None:
                    raise ODEBFContractError("P1R52 target-depth semantic gradient is absent")
                compute["semantic_forward_count"] += result.model_forward_count
                compute["semantic_backward_count"] += result.backward_count
                compute["semantic_processed_tokens"] += result.processed_token_count
                return result

            def evaluate_kl(target: torch.Tensor) -> Any:
                variable = (
                    target.detach()
                    .to(device=next(model.parameters()).device, dtype=torch.float32)
                    .clone()
                    .requires_grad_(True)
                )
                result, _ = evaluate_p1r24_kl(
                    model,
                    kl_plan,
                    teacher_log_probs=teacher,
                    target_state=variable,
                    current_terminal=current_terminal,
                    target_layer_name=hparams.layer_module_tmp.format(
                        int(hparams.layers[-1])
                    ),
                )
                compute["kl_forward_count"] += result.model_forward_count
                compute["kl_backward_count"] += result.backward_count
                compute["kl_processed_tokens"] += result.processed_token_count
                return result

            def evaluate_endpoint(target: torch.Tensor, role: str) -> Any:
                result = evaluate_scalable_target_new_objective(
                    model,
                    objective_plan,
                    target_state=target.detach().to(
                        device=next(model.parameters()).device, dtype=torch.float32
                    ),
                    current_terminal=current_terminal,
                    target_layer_name=hparams.layer_module_tmp.format(
                        int(hparams.layers[-1])
                    ),
                    target_gradient_required=False,
                )
                key = (
                    "primary_endpoint_forward_count"
                    if role == "PRIMARY"
                    else "rescue_endpoint_forward_count"
                )
                compute[key] += result.model_forward_count
                return result

            def fixed_state() -> tuple[str, str, str]:
                return (
                    _model_w0_contract(touched),
                    history_cache_sha,
                    empty_factor_sha,
                )

            outer = run_p1r52_target_depth_scheduler(
                outer_step_index=outer_step,
                depth=depth,
                current_target=current_target,
                current_terminal=current_terminal,
                target_origin=target_origin,
                state=state,
                lock=lock,
                alias=alias,
                shared_speed=float(metric.shared_speed),
                teacher_sha256=teacher_sha,
                evaluate_target=evaluate_target,
                evaluate_kl=evaluate_kl,
                evaluate_endpoint=evaluate_endpoint,
                fixed_state_identities=fixed_state,
            )
            target_wall += time.perf_counter() - outer_started
            for inner in outer.inner_steps:
                inner_payload = {
                    "schema": "ode-edit-s05-p1r52-target-depth-target-only-inner/v1",
                    "outer_step_index": outer_step,
                    "inner_index": inner.inner_index,
                    "global_target_update_ordinal": inner.global_target_update_ordinal,
                    "target_update": dict(inner.target_step.receipt),
                    "selected_endpoint": inner.selected_endpoint.raw_free_payload(),
                    "writer_materialization_count": 0,
                    "heldout_access_count": 0,
                }
                inner_payload["identity_sha256"] = canonical_hash(inner_payload)
                inner_receipts.append(
                    _atomic_write_once(
                        case_root
                        / "raw"
                        / "target"
                        / f"outer-{outer_step:02d}-inner-{inner.inner_index:02d}.json",
                        inner_payload,
                    )
                )
            outer_payload = dict(outer.receipt)
            outer_payload["identity_sha256"] = canonical_hash(
                {key: value for key, value in outer_payload.items() if key != "identity_sha256"}
            )
            outer_receipts.append(
                _atomic_write_once(
                    case_root / "raw" / "target" / f"outer-{outer_step:02d}.json",
                    outer_payload,
                )
            )
            compute["entry_norm_calibration_count"] += int(
                outer.receipt["entry_norm_calibration_count"]
            )
            current_target = outer.target_step.target_next
            state = outer.next_state
            k_states.append(current_target.clone())
        if len(k_states) != 9 or compute["entry_norm_calibration_count"] != 1:
            raise ODEBFStateError("P1R52 target-depth target-only trajectory differs")
        final_objective = evaluate_scalable_target_new_objective(
            model,
            objective_plan,
            target_state=current_target.detach().to(
                device=next(model.parameters()).device, dtype=torch.float32
            ),
            current_terminal=current_terminal,
            target_layer_name=hparams.layer_module_tmp.format(int(hparams.layers[-1])),
            target_gradient_required=False,
        )
        compute["semantic_forward_count"] += final_objective.model_forward_count
        compute["semantic_processed_tokens"] += final_objective.processed_token_count
        trajectory_file_sha, trajectory_tensor_sha = _write_private_target_trajectory_once(
            case_root / "private-target" / "kstate-targets.pt",
            k_states=k_states,
            request_order_sha256=request_order,
            schema="ode-edit-s05-p1r52-target-depth-private-kstate-target/v1",
        )
        action = {
            "schema": "ode-edit-s05-p1r52-target-depth-target-only-action-freeze/v1",
            "instruction_id": P1R52_TARGET_DEPTH_INSTRUCTION_ID,
            "method_id": P1R52_TARGET_DEPTH_METHOD_ID,
            "depth_policy": depth.value,
            "case_index": case_index,
            "alias": alias,
            "request_order_sha256": request_order,
            "outer_receipt_sha256": outer_receipts,
            "inner_receipt_sha256": inner_receipts,
            "outer_step_count": 8,
            "configured_inner_count": depth.inner_count,
            "writer_update_count": 0,
            "writer_materialization_count": 0,
            "terminal_target_sha256": tensor_sha256(current_target),
            "trajectory_tensor_sha256": trajectory_tensor_sha,
            "W0_sha256": _model_w0_contract(touched),
            "actions_frozen_before_evaluator": True,
            "inner_heldout_access_count": 0,
            "retry_count": 0,
        }
        action["identity_sha256"] = canonical_hash(action)
        freeze_sha = _atomic_write_once(case_root / "action-freeze.json", action)
        cases, _ = _action_frozen_cases(
            dataset_path,
            requests,
            arm=f"P1R52-TARGET-DEPTH-{depth.value}",
            selected_snapshot_sha256=freeze_sha,
            fixed_budget_slots_completed=8,
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
            method=f"P1R52-TARGET-DEPTH-{depth.value}",
            instruction_id=P1R52_TARGET_DEPTH_INSTRUCTION_ID,
            method_id=P1R52_TARGET_DEPTH_METHOD_ID,
            schema="ode-edit-s05-p1r52-target-depth-target-only-terminal-z-panel/v1",
            trajectory_status=f"ACTION_FROZEN_P1R52_TARGET_DEPTH_{depth.value}_K8",
        )
    finally:
        counter.close()
    if _model_w0_contract(touched) != expected_w0:
        raise ODEBFStateError("P1R52 target-depth target-only path mutated W0")
    restore = _restore_exact_w0(
        touched,
        base_values,
        mutation_lock=mutation_lock,
        expected_contract=expected_w0,
    )
    terminal = {
        "schema": "ode-edit-s05-p1r52-target-depth-target-only-case-terminal/v1",
        "instruction_id": P1R52_TARGET_DEPTH_INSTRUCTION_ID,
        "method_id": P1R52_TARGET_DEPTH_METHOD_ID,
        "depth_policy": depth.value,
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
        "terminal_full_six_target_new_nll_by_request": list(
            final_objective.per_request_values
        ),
        "terminal_z_panel": terminal_panel,
        "outer_step_count": 8,
        "configured_inner_count": depth.inner_count,
        "executed_inner_count": len(inner_receipts),
        "writer_update_count": 0,
        "writer_materialization_count": 0,
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
        "schema": "ode-edit-s05-p1r52-target-depth-target-only-case-manifest/v1",
        "case_index": case_index,
        "alias": alias,
        "depth_policy": depth.value,
        "terminal_sha256": terminal_sha,
        "action_freeze_sha256": freeze_sha,
        "target_trajectory_file_sha256": trajectory_file_sha,
        "outer_step_count": 8,
        "writer_materialization_count": 0,
        "W0_restored": True,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(case_root / "manifest.json", manifest)
    return {
        "status": "P1R52_TARGET_DEPTH_TARGET_ONLY_CASE_COMPLETE",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "terminal_target_sha256": tensor_sha256(current_target),
    }


def run_p1r52_target_depth_target_only(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    depth: P1R52TargetDepth | str,
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
    policy = depth if isinstance(depth, P1R52TargetDepth) else P1R52TargetDepth(depth)
    if case_count not in CASE_COUNTS or len(stream_batches) != 10:
        raise ODEBFContractError("P1R52 target-depth case inventory differs")
    if (
        stream.get("root_digest") != STREAM_ROOT
        or stream.get("all_request_order_sha256") != STREAM_ORDER
    ):
        raise ODEBFContractError("P1R52 target-depth frozen stream identity differs")
    if any(len(batch) != BATCH_SIZE for batch in stream_batches):
        raise ODEBFContractError("P1R52 target-depth stream is not B10x10")
    expected_w0 = _model_w0_contract(touched)
    if _hashes(touched) != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("P1R52 target-depth entry W0 differs")
    started = time.perf_counter()
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for case_index, requests in enumerate(stream_batches[:case_count], start=1):
        seed_all(COMMON_SEED)
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
                        depth=policy,
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
                "schema": "ode-edit-s05-p1r52-target-depth-target-only-case-failure/v1",
                "case_index": case_index,
                "alias": alias,
                "depth_policy": policy.value,
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
            raise ODEBFStateError("P1R52 target-depth post-case W0 differs")
        stages.record(
            f"post_p1r52_target_depth_{policy.value}_case_{case_index}",
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
        "schema": "ode-edit-s05-p1r52-target-depth-target-only-job-terminal/v1",
        "instruction_id": P1R52_TARGET_DEPTH_INSTRUCTION_ID,
        "method_id": P1R52_TARGET_DEPTH_METHOD_ID,
        "source_head": source_head,
        "alias": alias,
        "depth_policy": policy.value,
        "case_count": case_count,
        "request_attempt_count": case_count * BATCH_SIZE,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "completed": completed,
        "failed_case_identity_sha256": [item["identity_sha256"] for item in failed],
        "writer_update_count": 0,
        "writer_materialization_count": 0,
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
        "schema": "ode-edit-s05-p1r52-target-depth-target-only-job-manifest/v1",
        "source_head": source_head,
        "alias": alias,
        "depth_policy": policy.value,
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
        "status": "P1R52_TARGET_DEPTH_TARGET_ONLY_TERMINAL",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "W0_restored": terminal["W0_restored"],
    }


__all__ = [
    "CASE_COUNTS",
    "expected_p1r52_target_depth_result_name",
    "run_p1r52_target_depth_target_only",
]
