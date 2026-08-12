"""Ten independent whole-B10 P1R23 Atomic experiments from identical W0."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .accounting import ComputeLedger
from .atomic_runtime_optimization import AcceptedPhysicalStateMaterializer
from .contracts import BATCH_SIZE, ODEBFContractError, ODEBFStateError, canonical_hash
from .fixed_e8_soft_routing import FixedE8Arm
from .functional import tensor_sha256
from .p0_runtime import ModelForwardCounter
from .p1_replay import build_outer_entry_pretrained_cache
from .p1_runtime import ArmRuntimeState, _atomic_write_once, _entry_parameter_snapshot_sha256
from .p1_scalable_batched_experiment import (
    _action_frozen_cases,
    _evaluate_frozen_state,
    _model_w0_contract,
    _run_ode_arm,
)
from .p1_state import ArmWeightSnapshot, P1Arm, P1HistoryLedger
from .progress_simplex_routing import PROGRESS_SIMPLEX_METHOD_ID
from .scalable_batched_model import build_scalable_capture_plan, build_scalable_objective_plan
from .scalable_batched_native import restore_native_entry, run_official_native_apply
from .scalable_batched_runtime import (
    P1R23_GRID_COUNT,
    P1R23_LAYER_ORDER,
    scalable_ordered_request_digest,
)


INSTRUCTION_ID = "ODEEDIT-S05-P1R23-PROGRESS-SIMPLEX-INDEPENDENT-B10X10-V1"
METHODS = (
    "BG-PROGRESS-SIMPLEX-NEUTRAL",
    "BG-PROGRESS-SIMPLEX-SOFT",
    "RS-PROGRESS-SIMPLEX-NEUTRAL",
    "RS-PROGRESS-SIMPLEX-SOFT",
    "OFFICIAL-ALPHAEDIT",
)
CASE_COUNT = 10
HISTORY_MODE = "OFF"


def expected_historical_h0_result_name(alias: str, method: str) -> str:
    if method not in METHODS:
        raise ODEBFContractError("independent B10 method differs")
    return (
        "s05-p1r23-progress-simplex-independent-b10x10-"
        f"{method.lower().replace('-', '_')}-{alias}-v1"
    )


def _hashes(parameters: Mapping[str, torch.nn.Parameter]) -> dict[str, str]:
    return {name: tensor_sha256(value) for name, value in sorted(parameters.items())}


def _restore_exact_w0(
    touched: Mapping[str, torch.nn.Parameter],
    base_values: Mapping[str, torch.Tensor],
    *,
    mutation_lock: Any,
    expected_contract: str,
) -> dict[str, Any]:
    pointers = {name: int(value.data_ptr()) for name, value in touched.items()}
    with mutation_lock, torch.no_grad():
        for name, parameter in sorted(touched.items()):
            parameter.copy_(base_values[name].to(device=parameter.device, dtype=parameter.dtype))
    if (
        _model_w0_contract(touched) != expected_contract
        or any(int(touched[name].data_ptr()) != pointer for name, pointer in pointers.items())
    ):
        raise ODEBFStateError("independent B10 W0 restore differs")
    payload = {
        "schema": "ode-edit-s05-p1r23-independent-b10-w0-restore/v1",
        "pointer_restored_exact": True,
        "byte_restored_exact": True,
        "parameter_sha256": _hashes(touched),
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _history_off_receipt() -> dict[str, Any]:
    payload = {
        "schema": "ode-edit-s05-p1r23-independent-b10-history-off/v1",
        "mode": HISTORY_MODE,
        "router_visible_history_item_count": 0,
        "history_append_count": 0,
        "raw_historical_request_replay_count": 0,
        "projected_key_historical_sketch_construction_count": 0,
        "functional_h_status": "INACTIVE_BY_HISTORY_MODE_OFF",
        "functional_h_input_count": 0,
        "functional_h_decision_influence_count": 0,
        "structural_h_status": "INACTIVE_BY_HISTORY_MODE_OFF",
        "structural_h_input_count": 0,
        "structural_h_decision_influence_count": 0,
        "cross_case_weight_state_count": 0,
        "cross_case_controller_state_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _case_failure(
    case_root: Path,
    exc: BaseException,
    *,
    case_index: int,
    method: str,
    w0_restore: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": "ode-edit-s05-p1r23-independent-b10-case-failure/v1",
        "instruction_id": INSTRUCTION_ID,
        "case_index": case_index,
        "method": method,
        "status": "TYPED_CASE_FAILURE_NO_RETRY_NO_IMPUTATION",
        "exception_class": type(exc).__name__,
        "exception_message_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
        "w0_restore": dict(w0_restore),
        "retry_count": 0,
        "next_case_continues": True,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    _atomic_write_once(case_root / "failure.json", payload)
    return payload


def _case_freeze(
    *,
    case_index: int,
    alias: str,
    method: str,
    request_order_sha256: str,
    action_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema": "ode-edit-s05-p1r23-independent-b10-action-freeze/v1",
        "instruction_id": INSTRUCTION_ID,
        "case_index": case_index,
        "alias": alias,
        "method": method,
        "request_count": BATCH_SIZE,
        "request_order_sha256": request_order_sha256,
        "action_sha256": action_sha256,
        "actions_frozen_before_evaluator": True,
        "inner_k_evaluator_access_count": 0,
        "future_batch_access_count": 0,
        "history_mode": _history_off_receipt(),
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _run_ode_case(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    method: str,
    case_index: int,
    case_root: Path,
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: Any,
    projector_sha256: str,
    controller_lock: Any,
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: Any,
    theta0_cache: Any,
    outer_entry_p_cache: Any,
    dataset_path: Path,
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    if len(requests) != BATCH_SIZE:
        raise ODEBFContractError("independent Atomic case is not B10")
    request_order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    objective_plan = build_scalable_objective_plan(
        model,
        tokenizer,
        requests,
        contexts=contexts,
        request_microbatch_size=BATCH_SIZE,
        fact_token_strategy=hparams.fact_token,
    )
    capture_plan = build_scalable_capture_plan(
        tokenizer,
        requests,
        contexts=contexts,
        request_microbatch_size=BATCH_SIZE,
        fact_token_strategy=hparams.fact_token,
    )
    if (
        objective_plan.context_ordinals != tuple(range(6))
        or objective_plan.request_order_sha256 != request_order
        or capture_plan.request_order_sha256 != request_order
    ):
        raise ODEBFContractError("independent B10 full-six plan differs")
    _atomic_write_once(case_root / "raw" / "objective-plan.json", objective_plan.raw_free_payload())
    _atomic_write_once(case_root / "raw" / "capture-plan.json", capture_plan.raw_free_payload())
    allocation = "BG" if method.startswith("BG-") else "RS"
    arm = FixedE8Arm.NEUTRAL if method.endswith("-NEUTRAL") else FixedE8Arm.SOFT
    state = ArmRuntimeState(
        P1Arm.R_BF,
        P1HistoryLedger(layer_order=P1R23_LAYER_ORDER, maximum_records=40),
        ComputeLedger(),
        ArmWeightSnapshot(
            P1Arm.R_BF,
            0,
            base_receipt.parameter_sha256,
            canonical_hash({"case": case_index, "method": method, "w0": base_receipt.parameter_sha256}),
        ),
        dict(base_values),
    )
    rollout = _run_ode_arm(
        model,
        tokenizer,
        requests,
        alias=alias,
        arm=arm,
        allocation=allocation,
        capture_plan=capture_plan,
        objective_plan=objective_plan,
        hparams=hparams,
        projector=projector,
        contexts=contexts,
        covariance_registry=covariance_registry,
        projector_sha256=projector_sha256,
        controller_lock=controller_lock,
        arm_state=state,
        request_by_sha256=request_by_sha256,
        population_by_sha256=population_by_sha256,
        schedule=schedule,
        outer_entry_p_cache=outer_entry_p_cache,
        theta0_cache=theta0_cache,
        touched=touched,
        base_receipt=base_receipt,
        base_values=base_values,
        raw_root=case_root / "raw",
        write_once=_atomic_write_once,
        progress_simplex=True,
    )
    public = rollout["public"]
    if (
        public["status"] != "PROGRESS_SIMPLEX_DYNAMIC_K8_COMPLETE"
        or public["request_count"] != BATCH_SIZE
        or len(public["accepted_receipt_sha256"]) != P1R23_GRID_COUNT
        or public["tau_final"] != 1.0
        or len(state.history.snapshot().active_records) != 0
    ):
        raise ODEBFStateError("independent B10 Atomic rollout differs")
    freeze = _case_freeze(
        case_index=case_index,
        alias=alias,
        method=method,
        request_order_sha256=request_order,
        action_sha256=public["identity_sha256"],
    )
    freeze_sha = _atomic_write_once(case_root / "action-freeze.json", freeze)
    cases, evaluator_freeze = _action_frozen_cases(
        dataset_path,
        requests,
        arm=method,
        selected_snapshot_sha256=freeze_sha,
        fixed_budget_slots_completed=P1R23_GRID_COUNT,
    )
    w0, w0_eval_seconds = _evaluate_frozen_state(
        model, tokenizer, cases, alias=alias, freeze_payload=evaluator_freeze
    )
    materializer = AcceptedPhysicalStateMaterializer(model, base_values)
    materialized = False
    try:
        materialization = materializer.materialize(rollout["terminal_factors"], transition_index=8)
        materialized = True
        endpoint, evaluator_seconds = _evaluate_frozen_state(
            model, tokenizer, cases, alias=alias, freeze_payload=evaluator_freeze
        )
    finally:
        restore = materializer.restore()
    if not materialized or _model_w0_contract(touched) != public["initial_w0_sha256"]:
        raise ODEBFStateError("independent B10 endpoint W0 restore differs")
    history_off = _history_off_receipt()
    terminal = {
        "schema": "ode-edit-s05-p1r23-independent-b10-ode-terminal/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": PROGRESS_SIMPLEX_METHOD_ID,
        "case_index": case_index,
        "alias": alias,
        "method": method,
        "allocation": allocation,
        "arm": arm.value,
        "request_count": BATCH_SIZE,
        "request_order_sha256": request_order,
        "objective_plan_sha256": objective_plan.identity_sha256,
        "capture_plan_sha256": capture_plan.identity_sha256,
        "rollout": public,
        "W0_endpoint": w0,
        "endpoint": endpoint,
        "endpoint_materialization": materialization,
        "endpoint_restore": restore,
        "history_mode": history_off,
        "action_freeze_sha256": freeze_sha,
        "W0_evaluator_wall_seconds": w0_eval_seconds,
        "endpoint_evaluator_wall_seconds": evaluator_seconds,
        "retry_count": 0,
        "cross_case_state_count": 0,
        "W0_restored": True,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(case_root / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p1r23-independent-b10-case-manifest/v1",
        "case_index": case_index,
        "method": method,
        "terminal_sha256": terminal_sha,
        "action_freeze_sha256": freeze_sha,
        "W0_restored": True,
        "history_mode": HISTORY_MODE,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(case_root / "manifest.json", manifest)
    return {"status": "CASE_COMPLETE", "terminal_sha256": terminal_sha, "manifest_sha256": manifest_sha}


def _run_alpha_case(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    case_index: int,
    case_root: Path,
    hparams: Any,
    dataset_path: Path,
    touched: Mapping[str, torch.nn.Parameter],
    base_values: Mapping[str, torch.Tensor],
    job_ledger: ComputeLedger,
) -> dict[str, Any]:
    if len(requests) != BATCH_SIZE:
        raise ODEBFContractError("Official AlphaEdit case is not B10")
    request_order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    counter = ModelForwardCounter(model, job_ledger)
    try:
        action, originals = run_official_native_apply(
            model, tokenizer, requests, hparams, touched=touched
        )
    finally:
        counter.close()
    freeze = _case_freeze(
        case_index=case_index,
        alias=alias,
        method="OFFICIAL-ALPHAEDIT",
        request_order_sha256=request_order,
        action_sha256=action["identity_sha256"],
    )
    freeze_sha = _atomic_write_once(case_root / "action-freeze.json", freeze)
    cases, evaluator_freeze = _action_frozen_cases(
        dataset_path,
        requests,
        arm="OFFICIAL-ALPHAEDIT",
        selected_snapshot_sha256=freeze_sha,
        fixed_budget_slots_completed=0,
    )
    endpoint, evaluator_seconds = _evaluate_frozen_state(
        model, tokenizer, cases, alias=alias, freeze_payload=evaluator_freeze
    )
    restore = restore_native_entry(touched, originals)
    w0, w0_eval_seconds = _evaluate_frozen_state(
        model, tokenizer, cases, alias=alias, freeze_payload=evaluator_freeze
    )
    terminal = {
        "schema": "ode-edit-s05-p1r23-independent-b10-alpha-terminal/v1",
        "instruction_id": INSTRUCTION_ID,
        "case_index": case_index,
        "alias": alias,
        "method": "OFFICIAL-ALPHAEDIT",
        "request_count": BATCH_SIZE,
        "request_order_sha256": request_order,
        "action": action,
        "endpoint": endpoint,
        "W0_endpoint": w0,
        "restore": restore,
        "history_mode": _history_off_receipt(),
        "action_freeze_sha256": freeze_sha,
        "endpoint_evaluator_wall_seconds": evaluator_seconds,
        "W0_evaluator_wall_seconds": w0_eval_seconds,
        "retry_count": 0,
        "cross_case_state_count": 0,
        "W0_restored": True,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(case_root / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p1r23-independent-b10-case-manifest/v1",
        "case_index": case_index,
        "method": "OFFICIAL-ALPHAEDIT",
        "terminal_sha256": terminal_sha,
        "action_freeze_sha256": freeze_sha,
        "W0_restored": True,
        "history_mode": HISTORY_MODE,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(case_root / "manifest.json", manifest)
    return {"status": "CASE_COMPLETE", "terminal_sha256": terminal_sha, "manifest_sha256": manifest_sha}


def run_historical_h0_sequential_trajectory(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    method: str,
    destination: Path,
    raw_root: Path,
    stages: Any,
    source_head: str,
    stream_batches: Sequence[Sequence[Mapping[str, Any]]],
    stream: Mapping[str, Any],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: Any,
    projector_sha256: str,
    controller_lock: Any,
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    collision_by_request: Mapping[str, str],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: Any,
    theta0_cache: Any,
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
) -> dict[str, Any]:
    del collision_by_request, artifact_guard, artifact_receipt, numerical_sha256, context_sha256, cuda_runtime_receipt
    if method not in METHODS or len(stream_batches) != CASE_COUNT:
        raise ODEBFContractError("independent B10 matrix/count differs")
    if any(len(batch) != BATCH_SIZE for batch in stream_batches):
        raise ODEBFContractError("independent population is not ten B10 batches")
    if stream.get("all_request_order_sha256") != "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c":
        raise ODEBFContractError("independent B10 frozen order differs")
    expected_w0 = _model_w0_contract(touched)
    if _hashes(touched) != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("independent B10 entry W0 differs")
    started = time.perf_counter()
    outer_cache = None
    if method != "OFFICIAL-ALPHAEDIT":
        outer_population = tuple(population_by_sha256[item] for item in theta0_cache.request_order)
        snapshot = _entry_parameter_snapshot_sha256(model, dict(base_receipt.parameter_sha256))
        counter = ModelForwardCounter(model, job_ledger)
        try:
            outer_cache = build_outer_entry_pretrained_cache(
                model,
                tokenizer,
                outer_population,
                theta0_cache,
                outer_entry_snapshot_sha256=snapshot,
            )
        finally:
            counter.close()
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for case_index, requests in enumerate(stream_batches, start=1):
        case_root = raw_root / "cases" / f"case-{case_index:02d}"
        if _model_w0_contract(touched) != expected_w0:
            raise ODEBFStateError("cross-case W0 state leak detected")
        try:
            if method == "OFFICIAL-ALPHAEDIT":
                result = _run_alpha_case(
                    model,
                    tokenizer,
                    requests,
                    alias=alias,
                    case_index=case_index,
                    case_root=case_root,
                    hparams=hparams,
                    dataset_path=dataset_path,
                    touched=touched,
                    base_values=base_values,
                    job_ledger=job_ledger,
                )
            else:
                if outer_cache is None:
                    raise ODEBFStateError("independent B10 pretrained cache is absent")
                result = _run_ode_case(
                    model,
                    tokenizer,
                    requests,
                    alias=alias,
                    method=method,
                    case_index=case_index,
                    case_root=case_root,
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
                    outer_entry_p_cache=outer_cache,
                    dataset_path=dataset_path,
                    touched=touched,
                    base_receipt=base_receipt,
                    base_values=base_values,
                )
            completed.append({"case_index": case_index, **result})
        except Exception as exc:
            restore = _restore_exact_w0(
                touched,
                base_values,
                mutation_lock=mutation_lock,
                expected_contract=expected_w0,
            )
            failed.append(_case_failure(case_root, exc, case_index=case_index, method=method, w0_restore=restore))
        if _model_w0_contract(touched) != expected_w0:
            raise ODEBFStateError("independent B10 post-case W0 differs")
        stages.record(
            f"post_independent_b10_case_{case_index}",
            {
                "case_index": case_index,
                "method": method,
                "complete_count": len(completed),
                "failure_count": len(failed),
                "W0_restored": True,
                "history_mode": HISTORY_MODE,
            },
        )
    terminal = {
        "schema": "ode-edit-s05-p1r23-independent-b10x10-terminal/v1",
        "instruction_id": INSTRUCTION_ID,
        "source_head": source_head,
        "alias": alias,
        "method": method,
        "case_count": CASE_COUNT,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "completed": completed,
        "failed_case_identity_sha256": [item["identity_sha256"] for item in failed],
        "history_mode": _history_off_receipt(),
        "cross_case_state_count": 0,
        "retry_count": 0,
        "W0_restored": _model_w0_contract(touched) == expected_w0,
        "total_wall_seconds": time.perf_counter() - started,
        "job_compute": job_ledger.raw_free_payload(),
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p1r23-independent-b10x10-manifest/v1",
        "source_head": source_head,
        "terminal_sha256": terminal_sha,
        "case_count": CASE_COUNT,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "W0_restored": terminal["W0_restored"],
        "history_mode": HISTORY_MODE,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    return {
        "status": "P1R23_INDEPENDENT_B10X10_TERMINAL",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "W0_restored": terminal["W0_restored"],
    }


__all__ = ["CASE_COUNT", "INSTRUCTION_ID", "METHODS", "expected_historical_h0_result_name", "run_historical_h0_sequential_trajectory"]
