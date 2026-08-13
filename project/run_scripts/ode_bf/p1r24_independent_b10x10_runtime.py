"""Ten independent whole-B10 P1R24 RS-Soft Atomic edits from identical W0."""

from __future__ import annotations

import hashlib
import time
from contextlib import ExitStack, nullcontext
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
from .p1_stepwise import StepwiseActionFreeze, evaluate_counterfact_stepwise_primary
from .common_cold_coordinate import common_terminal_residual_input
from .common_coldcoord_fixed_e8_runtime import _heldout_additive_lookup_geometry
from .bg_soft_diagnostics import HeldoutRequestResidualActivationOverlay
from .p1_runtime import ArmRuntimeState, _atomic_write_once, _entry_parameter_snapshot_sha256
from .p1_scalable_batched_experiment import (
    _action_frozen_cases,
    _evaluate_frozen_state,
    _model_w0_contract,
    _run_ode_arm,
)
from .p1_state import ArmWeightSnapshot, P1Arm, P1HistoryLedger
from .p1r24_atomic_strength import P1R24_METHOD_ID
from .scalable_batched_model import (
    build_scalable_capture_plan,
    build_scalable_objective_plan,
    evaluate_scalable_target_new_objective,
)
from .scalable_batched_runtime import (
    P1R23_GRID_COUNT,
    P1R23_LAYER_ORDER,
    scalable_ordered_request_digest,
)


INSTRUCTION_ID = "ODEEDIT-S05-P1R31-P1R24-INDEPENDENT-B10X10-FULL-MATRIX-DETAILED-V1"
METHODS = tuple(
    f"{allocation}-P1R24-{arm}"
    for allocation in ("RS", "BG")
    for arm in ("NEUTRAL", "SOFT")
)
CASE_COUNT = 10
HISTORY_MODE = "OFF"


def expected_p1r24_independent_result_name(alias: str, method: str) -> str:
    if method not in METHODS:
        raise ODEBFContractError("independent B10 method differs")
    token = method.lower().replace("-p1r24-", "-")
    return f"s05-p1r31-p1r24-independent-b10x10-detailed-{alias}-{token}-v1"


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
        "schema": "ode-edit-s05-p1r24-independent-b10-w0-restore/v1",
        "pointer_restored_exact": True,
        "byte_restored_exact": True,
        "parameter_sha256": _hashes(touched),
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _history_off_receipt() -> dict[str, Any]:
    payload = {
        "schema": "ode-edit-s05-p1r24-independent-b10-history-off/v1",
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
) -> dict[str, Any]:
    payload = {
        "schema": "ode-edit-s05-p1r24-independent-b10-case-failure/v1",
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
        "schema": "ode-edit-s05-p1r24-independent-b10-action-freeze/v1",
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


def _postfreeze_stepwise_panel(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    cases: Sequence[Any],
    *,
    alias: str,
    method: str,
    case_root: Path,
    request_order_sha256: str,
    action_sha256: str,
    objective_plan: Any,
    hparams: Any,
    trajectory: Sequence[Mapping[str, Any]],
    touched: Mapping[str, torch.nn.Parameter],
) -> dict[str, Any]:
    """Evaluate the immutable k0..k8 trajectory after action freeze only."""

    if len(trajectory) != P1R23_GRID_COUNT + 1:
        raise ODEBFContractError("P1R31 frozen trajectory length differs")
    lookup_positions, patched_rows, lookup_receipt = (
        _heldout_additive_lookup_geometry(
            tokenizer,
            requests,
            cases,
            fact_token_strategy=hparams.fact_token,
        )
    )
    lookup_sha = _atomic_write_once(
        case_root / "stepwise" / "heldout-additive-lookup.json",
        lookup_receipt,
    )
    entry_w0 = _model_w0_contract(touched)
    hashes: list[str] = []
    step_rows: list[dict[str, Any]] = []
    wall_total = 0.0
    for snapshot in trajectory:
        index = int(snapshot["accepted_index"])
        factors = snapshot["factors"]
        snapshot_sha = canonical_hash(
            {
                "accepted_index": index,
                "action_sha256": action_sha256,
                "target_state_sha256": tensor_sha256(snapshot["target_state"]),
                "physical_terminal_sha256": tensor_sha256(
                    snapshot["physical_terminal"]
                ),
                "physical_capture_sha256": snapshot["physical_capture_sha256"],
                "factor_order": {
                    name: [list(item.order_key) for item in values]
                    for name, values in sorted(factors.items())
                },
            }
        )
        freeze = StepwiseActionFreeze(
            variant=method,
            request_order_sha256=request_order_sha256,
            rollout_sha256=action_sha256,
            snapshot_sha256=snapshot_sha,
            snapshot_index=index,
            accepted_snapshot_count=P1R23_GRID_COUNT + 1,
            rejected_retry_count=0,
            trajectory_status="ACTION_FROZEN_P1R24_K8",
        )
        started = time.perf_counter()
        with ExitStack() as stack:
            if any(factors.values()):
                stack.enter_context(_virtual_context(model, factors))
            weight_only = evaluate_counterfact_stepwise_primary(
                model, tokenizer, cases, model_alias=alias, freeze=freeze
            )
            weight_objective = evaluate_scalable_target_new_objective(
                model, objective_plan
            )
            target_variable = (
                snapshot["target_state"]
                .detach()
                .to(device=next(model.parameters()).device, dtype=torch.float32)
                .clone()
                .requires_grad_(True)
            )
            z_objective = evaluate_scalable_target_new_objective(
                model,
                objective_plan,
                target_state=target_variable,
                current_terminal=snapshot["physical_terminal"],
                target_layer_name=hparams.layer_module_tmp.format(
                    int(hparams.layers[-1])
                ),
            )
            residual = common_terminal_residual_input(
                snapshot["target_state"],
                snapshot["physical_terminal"],
                request_order_sha256,
            )
            overlay = HeldoutRequestResidualActivationOverlay(
                model,
                hparams.layer_module_tmp.format(int(hparams.layers[-1])),
                residual.residual,
                lookup_positions,
                patched_rows,
            )
            with overlay:
                z_oracle = evaluate_counterfact_stepwise_primary(
                    model, tokenizer, cases, model_alias=alias, freeze=freeze
                )
        wall = time.perf_counter() - started
        wall_total += wall
        if _model_w0_contract(touched) != entry_w0:
            raise ODEBFStateError("P1R31 stepwise evaluator mutated W0")
        row = {
            "schema": "ode-edit-s05-p1r31-p1r24-stepwise-observation/v1",
            "instruction_id": INSTRUCTION_ID,
            "method": method,
            "accepted_index": index,
            "snapshot_sha256": snapshot_sha,
            "action_freeze_sha256": freeze.identity(),
            "weight_only": weight_only.raw_free_payload(),
            "z_oracle": z_oracle.raw_free_payload(),
            "full_six_weight_target_new": weight_objective.raw_free_payload(),
            "full_six_z_target_new": z_objective.raw_free_payload(),
            "full_six_target_old_status": "NOT_RECORDED_BY_FROZEN_TARGET_NEW_PLAN",
            "z_oracle_overlay": overlay.raw_free_payload(),
            "heldout_lookup_sha256": lookup_sha,
            "evaluation_only_materialization_count": int(index > 0),
            "scientific_materialization_count": 0,
            "controller_decision_influence_count": 0,
            "heldout_access_after_action_freeze": True,
            "wall_seconds": wall,
        }
        row["identity_sha256"] = canonical_hash(row)
        hashes.append(
            _atomic_write_once(
                case_root / "stepwise" / f"accepted-k{index}.json", row
            )
        )
        step_rows.append(row)
    panel = {
        "schema": "ode-edit-s05-p1r31-p1r24-stepwise-panel/v1",
        "method": method,
        "snapshot_count": len(step_rows),
        "snapshot_receipt_sha256": hashes,
        "evaluation_only_materialization_count": P1R23_GRID_COUNT,
        "scientific_materialization_count": 0,
        "controller_decision_influence_count": 0,
        "heldout_opened_after_action_freeze": True,
        "wall_seconds": wall_total,
    }
    panel["identity_sha256"] = canonical_hash(panel)
    panel_sha = _atomic_write_once(case_root / "stepwise" / "panel.json", panel)
    return {
        "panel_sha256": panel_sha,
        "snapshot_receipt_sha256": hashes,
        "W0_weight_only": step_rows[0]["weight_only"],
        "W0_z_oracle": step_rows[0]["z_oracle"],
        "terminal_weight_only": step_rows[-1]["weight_only"],
        "terminal_z_oracle": step_rows[-1]["z_oracle"],
        "wall_seconds": wall_total,
    }


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
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    request_microbatch_size: int,
    job_ledger: ComputeLedger,
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
        request_microbatch_size=request_microbatch_size,
        fact_token_strategy=hparams.fact_token,
    )
    capture_plan = build_scalable_capture_plan(
        tokenizer,
        requests,
        contexts=contexts,
        request_microbatch_size=request_microbatch_size,
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
    if method not in METHODS:
        raise ODEBFContractError("P1R24 independent method differs")
    allocation = method.split("-", 1)[0]
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
        retain_postfreeze_trajectory=True,
    )
    public = rollout["public"]
    if (
        public["status"] != "P1R24_ATOMIC_STRENGTH_RECOVERY_K8_COMPLETE"
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
    panel = _postfreeze_stepwise_panel(
        model,
        tokenizer,
        requests,
        cases,
        alias=alias,
        method=method,
        case_root=case_root,
        request_order_sha256=request_order,
        action_sha256=public["identity_sha256"],
        objective_plan=objective_plan,
        hparams=hparams,
        trajectory=rollout["postfreeze_trajectory"],
        touched=touched,
    )
    if _model_w0_contract(touched) != public["initial_w0_sha256"]:
        raise ODEBFStateError("independent B10 stepwise W0 restore differs")
    history_off = _history_off_receipt()
    terminal = {
        "schema": "ode-edit-s05-p1r24-independent-b10-ode-terminal/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": P1R24_METHOD_ID,
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
        "W0_endpoint": panel["W0_weight_only"],
        "W0_z_oracle": panel["W0_z_oracle"],
        "endpoint": panel["terminal_weight_only"],
        "terminal_z_oracle": panel["terminal_z_oracle"],
        "stepwise_panel_sha256": panel["panel_sha256"],
        "stepwise_snapshot_receipt_sha256": panel["snapshot_receipt_sha256"],
        "endpoint_restore": {
            "pointer_restored_exact": True,
            "byte_restored_exact": True,
        },
        "history_mode": history_off,
        "action_freeze_sha256": freeze_sha,
        "W0_evaluator_wall_seconds": None,
        "stepwise_evaluator_wall_seconds": panel["wall_seconds"],
        "retry_count": 0,
        "cross_case_state_count": 0,
        "W0_restored": True,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(case_root / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p1r24-independent-b10-case-manifest/v1",
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


def run_p1r24_independent_b10x10(
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
        "schema": "ode-edit-s05-p1r24-independent-b10x10-terminal/v1",
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
        "schema": "ode-edit-s05-p1r24-independent-b10x10-manifest/v1",
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
        "status": "P1R31_P1R24_INDEPENDENT_B10X10_DETAILED_TERMINAL",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "completed_case_count": len(completed),
        "failed_case_count": len(failed),
        "W0_restored": terminal["W0_restored"],
    }


__all__ = ["CASE_COUNT", "INSTRUCTION_ID", "METHODS", "expected_p1r24_independent_result_name", "run_p1r24_independent_b10x10"]
