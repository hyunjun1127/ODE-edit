"""Ten independent whole-B10 P1R35 Atomic edits per full-matrix cell."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .accounting import ComputeLedger
from .alpha_backend import seed_all
from .atomic_runtime_optimization import AcceptedPhysicalStateMaterializer
from .contracts import BATCH_SIZE, COMMON_SEED, ODEBFContractError, ODEBFStateError, canonical_hash
from .fixed_e8_soft_routing import FixedE8Arm
from .functional import tensor_sha256
from .p0_runtime import ModelForwardCounter
from .p1_replay import build_outer_entry_pretrained_cache
from .p1_backend import _virtual_context
from .p1_runtime import ArmRuntimeState, _atomic_write_once, _entry_parameter_snapshot_sha256
from .p1_scalable_batched_experiment import (
    _action_frozen_cases,
    _evaluate_frozen_state,
    _model_w0_contract,
    _run_ode_arm,
)
from .p1_state import ArmWeightSnapshot, P1Arm, P1HistoryLedger
from .p1r34_w_anchored_finite_demand import P1R34NonSemanticTargetMove
from .p1r35_full_current_residual import P1R35_METHOD_ID
from .p1r38_perrequest_target import P1R38_METHOD_ID
from .p1r40_semantic_deficit_velocity_decay import (
    P1R40_INSTRUCTION_ID,
    P1R40_METHOD_ID,
    P1R40VelocityMechanismError,
)
from .scalable_batched_model import (
    build_scalable_capture_plan,
    build_scalable_objective_plan,
)
from .scalable_batched_runtime import (
    P1R23_GRID_COUNT,
    P1R23_LAYER_ORDER,
    scalable_ordered_request_digest,
)


INSTRUCTION_ID = "ODEEDIT-S05-P1R36-P1R35-INDEPENDENT-B10X10-FULL-MATRIX-V1"
METHODS = tuple(
    f"{allocation}-P1R35-{arm}"
    for allocation in ("RS", "BG")
    for arm in ("NEUTRAL", "SOFT")
)
CASE_COUNT = 10
HISTORY_MODE = "OFF"


def expected_p1r36_independent_result_name(alias: str, method: str) -> str:
    if method not in METHODS:
        raise ODEBFContractError("independent B10 method differs")
    token = method.lower().replace("-p1r35-", "-")
    return f"s05-p1r36-p1r35-independent-b10x10-{alias}-{token}-v1"


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
        "schema": "ode-edit-s05-p1r35-independent-b10-w0-restore/v1",
        "pointer_restored_exact": True,
        "byte_restored_exact": True,
        "parameter_sha256": _hashes(touched),
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _history_off_receipt() -> dict[str, Any]:
    payload = {
        "schema": "ode-edit-s05-p1r35-independent-b10-history-off/v1",
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
        "atomic_trajectory_ledger_entry_count_at_case_entry": 0,
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
    p1r38: bool = False,
    p1r40: bool = False,
) -> dict[str, Any]:
    classification = (
        "SCIENTIFIC_FAIL"
        if isinstance(exc, (P1R34NonSemanticTargetMove, P1R40VelocityMechanismError))
        else "TECHNICAL_FAIL"
    )
    prefixes = sorted(
        case_root.glob("raw/ode/*/accepted-k*.json"),
        key=lambda value: value.name,
    )
    payload = {
        "schema": (
            "ode-edit-s05-p1r40-independent-b10-case-failure/v1"
            if p1r40
            else
            "ode-edit-s05-p1r38-perrequest-independent-b10-case-failure/v1"
            if p1r38
            else "ode-edit-s05-p1r35-independent-b10-case-failure/v1"
        ),
        "instruction_id": (
            P1R40_INSTRUCTION_ID
            if p1r40
            else "ODEEDIT-S05-P1R38-PR-P1R35-PERREQUEST-TARGET-ATOMIC-V1"
            if p1r38
            else INSTRUCTION_ID
        ),
        "case_index": case_index,
        "method": method,
        "status": "TYPED_CASE_FAILURE_NO_RETRY_NO_IMPUTATION",
        "classification": classification,
        "exception_class": type(exc).__name__,
        "exception_message_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
        "w0_restore": dict(w0_restore),
        "last_valid_prefix_count": len(prefixes),
        "last_valid_prefix_sha256": [
            hashlib.sha256(path.read_bytes()).hexdigest() for path in prefixes
        ],
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
    p1r38: bool = False,
    p1r40: bool = False,
) -> dict[str, Any]:
    payload = {
        "schema": (
            "ode-edit-s05-p1r40-independent-b10-action-freeze/v1"
            if p1r40
            else
            "ode-edit-s05-p1r38-perrequest-independent-b10-action-freeze/v1"
            if p1r38
            else "ode-edit-s05-p1r35-independent-b10-action-freeze/v1"
        ),
        "instruction_id": (
            P1R40_INSTRUCTION_ID
            if p1r40
            else "ODEEDIT-S05-P1R38-PR-P1R35-PERREQUEST-TARGET-ATOMIC-V1"
            if p1r38
            else INSTRUCTION_ID
        ),
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
    dataset_path: Path,
    mutation_lock: Any,
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    request_microbatch_size: int,
    job_ledger: ComputeLedger,
    p1r38: bool = False,
    p1r40: bool = False,
    technical_smoke: bool = False,
) -> dict[str, Any]:
    if len(requests) != (1 if technical_smoke else BATCH_SIZE):
        raise ODEBFContractError("independent Atomic case is not B10")
    request_order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    objective_plan = build_scalable_objective_plan(
        model,
        tokenizer,
        requests,
        contexts=contexts,
        request_microbatch_size=min(request_microbatch_size, len(requests)),
        fact_token_strategy=hparams.fact_token,
    )
    capture_plan = build_scalable_capture_plan(
        tokenizer,
        requests,
        contexts=contexts,
        request_microbatch_size=min(request_microbatch_size, len(requests)),
        fact_token_strategy=hparams.fact_token,
    )
    objective_payload = objective_plan.raw_free_payload()
    capture_payload = capture_plan.raw_free_payload()
    capture_ordinals = sorted(
        {
            int(ordinal)
            for batch in capture_plan.batches
            for ordinal in batch.row_context_ordinals
        }
    )
    if (
        objective_payload.get("context_count") != 6
        or capture_ordinals != list(range(6))
        or objective_plan.request_order_sha256 != request_order
        or capture_plan.request_order_sha256 != request_order
    ):
        raise ODEBFContractError("independent B10 full-six plan differs")
    outer_population = tuple(
        population_by_sha256[item] for item in theta0_cache.request_order
    )
    snapshot = _entry_parameter_snapshot_sha256(
        model, dict(base_receipt.parameter_sha256)
    )
    counter = ModelForwardCounter(model, job_ledger)
    try:
        outer_entry_p_cache = build_outer_entry_pretrained_cache(
            model,
            tokenizer,
            outer_population,
            theta0_cache,
            outer_entry_snapshot_sha256=snapshot,
        )
    finally:
        counter.close()
    _atomic_write_once(case_root / "raw" / "objective-plan.json", objective_payload)
    _atomic_write_once(case_root / "raw" / "capture-plan.json", capture_payload)
    p1r38_methods = ("PR-P1R35-NEUTRAL", "PR-P1R35-SOFT")
    p1r40_methods = ("SDVD-P1R38-NEUTRAL", "SDVD-P1R38-SOFT")
    if p1r38 and p1r40:
        raise ODEBFContractError("independent target policies are exclusive")
    if method not in (p1r40_methods if p1r40 else p1r38_methods if p1r38 else METHODS):
        raise ODEBFContractError("independent method differs")
    allocation = "RS" if p1r38 or p1r40 else method.split("-", 1)[0]
    arm = (
        FixedE8Arm.NEUTRAL
        if method.endswith("-NEUTRAL")
        else FixedE8Arm.SOFT
    )
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
        p1r24=True,
        p1r34=True,
        p1r35=True,
        p1r38=p1r38,
        p1r40=p1r40,
    )
    public = rollout["public"]
    if (
        public["status"]
        != (
            "P1R40_SEMANTIC_DEFICIT_VELOCITY_DECAY_K8_COMPLETE"
            if p1r40
            else "P1R38_PR_P1R35_K8_COMPLETE"
            if p1r38
            else "P1R35_FULL_CURRENT_RESIDUAL_K8_COMPLETE"
        )
        or public["request_count"] != (1 if technical_smoke else BATCH_SIZE)
        or len(public["accepted_receipt_sha256"]) != P1R23_GRID_COUNT
        or public["tau_final"] != 1.0
        or len(state.history.snapshot().active_records) != 0
    ):
        raise ODEBFStateError("independent B10 Atomic rollout differs")
    if technical_smoke:
        restore = _restore_exact_w0(
            touched,
            base_values,
            mutation_lock=mutation_lock,
            expected_contract=public["initial_w0_sha256"],
        )
        terminal = {
            "schema": (
                "ode-edit-s05-p1r40-b1-technical-terminal/v1"
                if p1r40
                else "ode-edit-s05-p1r38-perrequest-b1-technical-terminal/v1"
            ),
            "instruction_id": P1R40_INSTRUCTION_ID if p1r40 else "ODEEDIT-S05-P1R38-PR-P1R35-PERREQUEST-TARGET-ATOMIC-V1",
            "method_id": P1R40_METHOD_ID if p1r40 else P1R38_METHOD_ID,
            "case_index": case_index,
            "alias": alias,
            "method": method,
            "request_count": 1,
            "rollout": public,
            "endpoint_score_gate_influence_count": 0,
            "heldout_evaluator_access_count": 0,
            "W0_restored": True,
            "endpoint_restore": restore,
        }
        terminal["identity_sha256"] = canonical_hash(terminal)
        terminal_sha = _atomic_write_once(case_root / "terminal.json", terminal)
        manifest = {
            "schema": (
                "ode-edit-s05-p1r40-b1-technical-manifest/v1"
                if p1r40
                else "ode-edit-s05-p1r38-perrequest-b1-technical-manifest/v1"
            ),
            "terminal_sha256": terminal_sha,
            "W0_restored": True,
            "K8": True,
        }
        manifest["identity_sha256"] = canonical_hash(manifest)
        manifest_sha = _atomic_write_once(case_root / "manifest.json", manifest)
        return {
            "status": "B1_TECHNICAL_COMPLETE",
            "terminal_sha256": terminal_sha,
            "manifest_sha256": manifest_sha,
        }
    accepted_terminal = json.loads(
        (
            case_root
            / "raw"
            / "ode"
            / public["arm"].lower()
            / "accepted-k8.json"
        ).read_text(encoding="utf-8")
    )
    freeze = _case_freeze(
        case_index=case_index,
        alias=alias,
        method=method,
        request_order_sha256=request_order,
        action_sha256=public["identity_sha256"],
        p1r38=p1r38,
        p1r40=p1r40,
    )
    freeze_sha = _atomic_write_once(case_root / "action-freeze.json", freeze)
    cases, evaluator_freeze = _action_frozen_cases(
        dataset_path,
        requests,
        arm=method,
        selected_snapshot_sha256=freeze_sha,
        fixed_budget_slots_completed=P1R23_GRID_COUNT,
    )
    with _virtual_context(model, rollout["terminal_factors"]):
        endpoint, evaluator_wall = _evaluate_frozen_state(
            model,
            tokenizer,
            cases,
            alias=alias,
            freeze_payload=evaluator_freeze,
        )
    if _model_w0_contract(touched) != public["initial_w0_sha256"]:
        raise ODEBFStateError("independent B10 terminal evaluator W0 differs")
    restore = _restore_exact_w0(
        touched,
        base_values,
        mutation_lock=mutation_lock,
        expected_contract=public["initial_w0_sha256"],
    )
    history_off = _history_off_receipt()
    terminal = {
        "schema": (
            "ode-edit-s05-p1r40-independent-b10-ode-terminal/v1"
            if p1r40
            else
            "ode-edit-s05-p1r38-perrequest-independent-b10-ode-terminal/v1"
            if p1r38
            else "ode-edit-s05-p1r35-independent-b10-ode-terminal/v1"
        ),
        "instruction_id": (
            P1R40_INSTRUCTION_ID
            if p1r40
            else "ODEEDIT-S05-P1R38-PR-P1R35-PERREQUEST-TARGET-ATOMIC-V1"
            if p1r38
            else INSTRUCTION_ID
        ),
        "method_id": (
            P1R40_METHOD_ID
            if p1r40
            else P1R38_METHOD_ID
            if p1r38
            else P1R35_METHOD_ID
        ),
        "case_index": case_index,
        "alias": alias,
        "method": method,
        "allocation": (
            "NOT_AN_EXPERIMENT_FACTOR" if p1r38 or p1r40 else allocation
        ),
        "inherited_writer_router_family": (
            "P1R35_RS" if p1r38 or p1r40 else allocation
        ),
        "arm": arm.value,
        "request_count": BATCH_SIZE,
        "request_order_sha256": request_order,
        "objective_plan_sha256": objective_plan.identity_sha256,
        "capture_plan_sha256": capture_plan.identity_sha256,
        "rollout": public,
        "endpoint": endpoint,
        "terminal_z8_oracle": public["terminal_z8_oracle"],
        "terminal_w8_full_six_target_new_nll": accepted_terminal["progress"]["terminal_mean_target_new_nll"],
        "stepwise_heldout_status": "NOT_RECORDED",
        "endpoint_restore": restore,
        "history_mode": history_off,
        "action_freeze_sha256": freeze_sha,
        "terminal_evaluator_wall_seconds": evaluator_wall,
        "retry_count": 0,
        "cross_case_state_count": 0,
        "W0_restored": bool(restore["pointer_restored_exact"] and restore["byte_restored_exact"]),
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(case_root / "terminal.json", terminal)
    manifest = {
        "schema": (
            "ode-edit-s05-p1r40-independent-b10-case-manifest/v1"
            if p1r40
            else
            "ode-edit-s05-p1r38-perrequest-independent-b10-case-manifest/v1"
            if p1r38
            else "ode-edit-s05-p1r35-independent-b10-case-manifest/v1"
        ),
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


def run_p1r36_independent_b10x10(
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
    request_microbatch_size: int,
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
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for case_index, requests in enumerate(stream_batches, start=1):
        case_root = raw_root / "cases" / f"case-{case_index:02d}"
        seed_all(COMMON_SEED)
        if _model_w0_contract(touched) != expected_w0:
            raise ODEBFStateError("cross-case W0 state leak detected")
        try:
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
                dataset_path=dataset_path,
                mutation_lock=mutation_lock,
                touched=touched,
                base_receipt=base_receipt,
                base_values=base_values,
                request_microbatch_size=request_microbatch_size,
                job_ledger=job_ledger,
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
        "schema": "ode-edit-s05-p1r35-independent-b10x10-terminal/v1",
        "instruction_id": INSTRUCTION_ID,
        "source_head": source_head,
        "alias": alias,
        "method": method,
        "case_count": CASE_COUNT,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "scientific_failed_case_count": sum(
            item["classification"] == "SCIENTIFIC_FAIL" for item in failed
        ),
        "technical_failed_case_count": sum(
            item["classification"] == "TECHNICAL_FAIL" for item in failed
        ),
        "completed": completed,
        "failed_case_identity_sha256": [item["identity_sha256"] for item in failed],
        "history_mode": _history_off_receipt(),
        "cross_case_state_count": 0,
        "retry_count": 0,
        "rng_reset_count": CASE_COUNT,
        "cross_case_mutable_cache_count": 0,
        "W0_restored": _model_w0_contract(touched) == expected_w0,
        "total_wall_seconds": time.perf_counter() - started,
        "job_compute": job_ledger.raw_free_payload(),
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p1r35-independent-b10x10-manifest/v1",
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
        "status": "P1R36_P1R35_INDEPENDENT_B10X10_TERMINAL",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "W0_restored": terminal["W0_restored"],
    }


__all__ = ["CASE_COUNT", "INSTRUCTION_ID", "METHODS", "expected_p1r36_independent_result_name", "run_p1r36_independent_b10x10"]
