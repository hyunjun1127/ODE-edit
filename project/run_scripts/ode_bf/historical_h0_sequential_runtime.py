"""Exact P1R23 Progress-Simplex sequential validation with History disabled."""

from __future__ import annotations

import hashlib
import math
import resource
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .accounting import ComputeLedger
from .contracts import BATCH_SIZE, ODEBFContractError, ODEBFStateError, canonical_hash
from .compute_progress_simplex_runtime import (
    COMPUTE_TOKEN_BUDGET,
    FixedRankHistoricalSketch,
)
from .fixed_e8_soft_routing import FixedE8Arm
from .functional import tensor_sha256
from .p0_runtime import ModelForwardCounter
from .p1_backend import capture_p1_native_entry
from .p1_evaluator import load_counterfact_cases_after_freeze
from .p1_replay import (
    build_outer_entry_pretrained_cache,
    evaluate_next_token_log_probs,
    samplewise_teacher_kl,
)
from .p1_runtime import (
    ArmRuntimeState,
    _assemble_candidates,
    _atomic_write_once,
    _entry_parameter_snapshot_sha256,
    _finalize_prepared_history,
    _history_keys,
    _observed_memory,
    _prepare_history_batch,
)
from .p1_scalable_batched_experiment import _model_w0_contract, _run_ode_arm
from .scalable_batched_model import (
    build_scalable_capture_plan,
    build_scalable_objective_plan,
)
from .scalable_batched_runtime import (
    P1R23_GRID_COUNT,
    P1R23_H,
    P1R23_LAYER_ORDER,
    scalable_ordered_request_digest,
)
from .p1_state import ArmWeightSnapshot, P1Arm, P1HistoryLedger
from .p1_stepwise import StepwiseActionFreeze, evaluate_counterfact_stepwise_primary
from .request_digest import ordered_request_digest_v1
from .transaction import AtomicBatchTransaction


INSTRUCTION_ID = "ODEEDIT-S05-P1R23-PROGRESS-SIMPLEX-SEQUENTIAL-NOH-V1"
METHODS = (
    "BG-PROGRESS-SIMPLEX-NEUTRAL",
    "BG-PROGRESS-SIMPLEX-SOFT",
    "RS-PROGRESS-SIMPLEX-NEUTRAL",
    "RS-PROGRESS-SIMPLEX-SOFT",
)
ROUND_COUNT = 10
EVALUATION_ROUNDS = (1, 5, 10)


def expected_historical_h0_result_name(alias: str, method: str) -> str:
    if method not in METHODS:
        raise ODEBFContractError("Compute-A1 Historical method identity differs")
    return (
        "s05-p1r23-progress-simplex-sequential-noh-"
        f"{method.lower().replace('-', '_')}-{alias}-v1"
    )


def _weight_values(parameters: Mapping[str, torch.nn.Parameter]) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in sorted(parameters.items())
    }


def _weight_hashes(parameters: Mapping[str, torch.nn.Parameter]) -> dict[str, str]:
    return {name: tensor_sha256(value) for name, value in sorted(parameters.items())}


def _ledger_snapshot(ledger: ComputeLedger) -> dict[str, Any]:
    return {
        "counters": dict(ledger.counters),
        "wall": dict(ledger.component_wall_seconds),
        "gpu": dict(ledger.component_gpu_seconds),
    }


def _ledger_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "counters": {
            key: int(after["counters"].get(key, 0) - before["counters"].get(key, 0))
            for key in sorted(set(before["counters"]) | set(after["counters"]))
        },
        "wall_seconds": {
            key: float(after["wall"].get(key, 0.0) - before["wall"].get(key, 0.0))
            for key in sorted(set(before["wall"]) | set(after["wall"]))
        },
        "gpu_seconds": {
            key: float(after["gpu"].get(key, 0.0) - before["gpu"].get(key, 0.0))
            for key in sorted(set(before["gpu"]) | set(after["gpu"]))
        },
    }


_HISTORY_OFF_ZERO_FIELDS = (
    "router_visible_history_item_count",
    "raw_historical_request_replay_count",
    "projected_key_historical_sketch_construction_count",
    "functional_h_controller_input_count",
    "functional_h_decision_influence_count",
    "functional_h_model_forward_count",
    "structural_h_controller_input_count",
    "structural_h_decision_influence_count",
    "structural_h_model_forward_count",
)


def _validate_history_off_edit_receipt(edit: Mapping[str, Any]) -> None:
    if (
        edit.get("history_mode") != "OFF"
        or edit.get("functional_h_status") != "INACTIVE_BY_HISTORY_MODE_OFF"
        or edit.get("structural_h_status") != "INACTIVE_BY_HISTORY_MODE_OFF"
        or any(int(edit.get(key, -1)) != 0 for key in _HISTORY_OFF_ZERO_FIELDS)
    ):
        raise ODEBFStateError("Sequential-NoH H firewall differs")


def _layer_load(
    entry: Mapping[str, torch.Tensor],
    current: Mapping[str, torch.nn.Parameter],
    hparams: Any,
) -> dict[int, float]:
    result: dict[int, float] = {}
    for layer in hparams.layers:
        name = f"{hparams.rewrite_module_tmp.format(int(layer))}.weight"
        delta = (
            current[name].detach().to(device="cpu", dtype=torch.float64)
            - entry[name].to(dtype=torch.float64)
        )
        value = float(torch.sum(delta.square()))
        if not math.isfinite(value) or value < 0.0:
            raise ODEBFContractError("P1R20 layer load is non-finite")
        result[int(layer)] = value
    return result


def _commit_candidates(
    touched: Mapping[str, torch.nn.Parameter],
    candidates: Mapping[str, torch.Tensor],
    *,
    transaction_id: str,
    mutation_lock: Any,
) -> dict[str, Any]:
    transaction = AtomicBatchTransaction(
        touched, transaction_id=transaction_id, mutation_lock=mutation_lock
    )
    expected = {name: tensor_sha256(value) for name, value in candidates.items()}
    for name, value in sorted(candidates.items()):
        transaction.stage(name, value.to(dtype=torch.bfloat16, device="cpu"))
    receipt = transaction.commit(post_commit_verify=lambda: _weight_hashes(touched) == expected)
    return asdict(receipt)


def _endpoint_freeze(
    *, method: str, round_index: int, request_order: str, state_sha256: str
) -> StepwiseActionFreeze:
    rollout = canonical_hash(
        {
            "instruction_id": INSTRUCTION_ID,
            "method": method,
            "round": round_index,
            "state": state_sha256,
        }
    )
    return StepwiseActionFreeze(
        variant=method,
        request_order_sha256=request_order,
        rollout_sha256=rollout,
        snapshot_sha256=state_sha256,
        snapshot_index=round_index,
        accepted_snapshot_count=round_index,
        rejected_retry_count=0,
        trajectory_status="ROUND_ENDPOINT_ACTION_FROZEN",
    )


def _evaluate_batches(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    method: str,
    round_index: int,
    batches: Sequence[Sequence[Mapping[str, Any]]],
    dataset_path: Path,
    state_sha256: str,
    ledger: ComputeLedger,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    evaluator_started = time.perf_counter()
    counter = ModelForwardCounter(model, ledger)
    try:
        for batch_index, requests in enumerate(batches):
            request_order = ordered_request_digest_v1(
                [str(item["request_sha256"]) for item in requests]
            )
            freeze = _endpoint_freeze(
                method=method,
                round_index=round_index,
                request_order=request_order,
                state_sha256=canonical_hash(
                    {"state": state_sha256, "evaluation_batch": batch_index}
                ),
            )
            cases = load_counterfact_cases_after_freeze(dataset_path, requests, freeze)
            observed = evaluate_counterfact_stepwise_primary(
                model, tokenizer, cases, model_alias=alias, freeze=freeze
            )
            ledger.increment("evaluator_forward", observed.primary.model_forward_count)
            ledger.increment("evaluator_tokens", observed.primary.processed_token_count)
            receipts.append(
                {
                    "batch_index": batch_index,
                    "freeze_sha256": freeze.identity(),
                    **observed.raw_free_payload(),
                }
            )
    finally:
        counter.close()
    elapsed = time.perf_counter() - evaluator_started
    ledger.add_time("historical_endpoint_evaluation", wall_seconds=elapsed)
    counts = {
        metric: sum(
            int(item["primary"]["metrics"][metric]["numerator"])
            for item in receipts
        )
        for metric in ("efficacy", "generalization", "locality-preservation")
    }
    denominators = {
        metric: sum(
            int(item["primary"]["metrics"][metric]["denominator"])
            for item in receipts
        )
        for metric in counts
    }
    return receipts, {
        "batch_count": len(receipts),
        "counts": counts,
        "denominators": denominators,
        "receipt_identity_sha256": canonical_hash(receipts),
        "wall_seconds": elapsed,
        "controller_access_count": 0,
    }


def _teacher_kl(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    theta0_cache: Any,
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    ledger: ComputeLedger,
) -> dict[str, Any]:
    identities = tuple(theta0_cache.request_order[:BATCH_SIZE])
    anchors = tuple(population_by_sha256[item] for item in identities)
    counter = ModelForwardCounter(model, ledger)
    try:
        observed, receipt = evaluate_next_token_log_probs(model, tokenizer, anchors)
    finally:
        counter.close()
    values = samplewise_teacher_kl(theta0_cache.select(identities), observed)
    ledger.increment("evaluator_forward", receipt.model_forward_count)
    ledger.increment("evaluator_tokens", receipt.processed_token_count)
    return {
        "anchor_order_sha256": receipt.request_order_sha256,
        "teacher_cache_sha256": theta0_cache.receipt_sha256,
        "sample_count": len(identities),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "maximum": float(values.max()),
        "values_sha256": tensor_sha256(values),
        "model_forward_count": receipt.model_forward_count,
        "processed_token_count": receipt.processed_token_count,
        "decision_influence_count": 0,
    }


def _run_alpha_round(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    state: ArmRuntimeState,
    touched: Mapping[str, torch.nn.Parameter],
    mutation_lock: Any,
    residual_tolerance: float,
    round_index: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    before = _weight_hashes(touched)
    capture = capture_p1_native_entry(
        model,
        tokenizer,
        requests,
        hparams,
        projector,
        contexts,
        history_keys_by_layer=_history_keys(
            state.history, tuple(int(item) for item in hparams.layers), risk=False
        ),
        mutation_lock=mutation_lock,
        ledger=state.ledger,
        residual_tolerance=residual_tolerance,
    )
    if _weight_hashes(touched) != before:
        raise ODEBFStateError("P1R20 Alpha capture mutated entry")
    transaction = _commit_candidates(
        touched,
        capture.native_candidates,
        transaction_id=canonical_hash(
            {"method": "ALPHAEDIT", "round": round_index, "entry": before}
        ),
        mutation_lock=mutation_lock,
    )
    load = _layer_load(capture.entry_weights, touched, hparams)
    return {
        "backend": "PINNED_ALPHAEDIT_N32_JOINT_B10",
        "capture": capture.raw_free_payload(),
        "transaction": transaction,
        "retry_count": 0,
    }, load


def _run_ours_round(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    method: str,
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: Any,
    projector_sha256: str,
    controller_lock: Any,
    state: ArmRuntimeState,
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: Any,
    theta0_cache: Any,
    touched: Mapping[str, torch.nn.Parameter],
    raw_root: Path,
    stages: Any,
    mutation_lock: Any,
    round_index: int,
    outer_entry_p_cache: Any,
    job_ledger: ComputeLedger,
    request_microbatch_size: int = BATCH_SIZE,
) -> tuple[dict[str, Any], dict[int, float]]:
    request_order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    entry_values = _weight_values(touched)
    entry_hashes = _weight_hashes(touched)
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
    round_root = raw_root / f"round-{round_index:02d}"
    _atomic_write_once(
        round_root / "plans" / "objective.json", objective_plan.raw_free_payload()
    )
    _atomic_write_once(
        round_root / "plans" / "capture.json", capture_plan.raw_free_payload()
    )
    if (
        objective_plan.context_ordinals != tuple(range(6))
        or objective_plan.request_order_sha256 != request_order
        or capture_plan.request_order_sha256 != request_order
    ):
        raise ODEBFContractError("Sequential-NoH full-six plan identity differs")
    allocation = "BG" if method.startswith("BG-") else "RS"
    arm = FixedE8Arm.NEUTRAL if method.endswith("-NEUTRAL") else FixedE8Arm.SOFT
    round_state = ArmRuntimeState(
        P1Arm.R_BF,
        P1HistoryLedger(layer_order=P1R23_LAYER_ORDER, maximum_records=40),
        ComputeLedger(),
        ArmWeightSnapshot(
            P1Arm.R_BF,
            0,
            entry_hashes,
            canonical_hash(
                {"method": method, "round": round_index, "entry": entry_hashes}
            ),
        ),
        dict(entry_values),
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
        arm_state=round_state,
        request_by_sha256=request_by_sha256,
        population_by_sha256=population_by_sha256,
        schedule=schedule,
        outer_entry_p_cache=outer_entry_p_cache,
        theta0_cache=theta0_cache,
        touched=touched,
        base_receipt=round_state.snapshot_receipt,
        base_values=entry_values,
        raw_root=round_root,
        write_once=_atomic_write_once,
        progress_simplex=True,
        terminal_functional_audit=round_index in EVALUATION_ROUNDS,
        full_six_slope=True,
        history_mode_off=True,
    )
    public = rollout["public"]
    if (
        public["status"] != "PROGRESS_SIMPLEX_DYNAMIC_K8_COMPLETE"
        or len(public["accepted_receipt_sha256"]) != P1R23_GRID_COUNT
        or public["tau_final"] != 1.0
        or public["historical_h_decision_influence_count"] != 0
        or public["historical_h_controller_input_count"] != 0
    ):
        raise ODEBFStateError("Progress-Simplex Sequential-NoH K8 contract differs")
    candidates, assembly = _assemble_candidates(
        touched, entry_hashes, rollout["terminal_factors"]
    )
    transaction = _commit_candidates(
        touched,
        candidates,
        transaction_id=canonical_hash(
            {
                "method": method,
                "round": round_index,
                "rollout": public["identity_sha256"],
            }
        ),
        mutation_lock=mutation_lock,
    )
    load = _layer_load(entry_values, touched, hparams)
    for name, amount in round_state.ledger.counters.items():
        job_ledger.increment(name, int(amount))
    for name, amount in round_state.ledger.component_wall_seconds.items():
        job_ledger.add_time(name, wall_seconds=float(amount))
    job_ledger.observe_memory(
        allocated_bytes=round_state.ledger.peak_allocated_bytes,
        reserved_bytes=round_state.ledger.peak_reserved_bytes,
        maxrss_kib=round_state.ledger.host_maxrss_kib,
    )
    return {
        "backend": "P1R23_EXACT_PROGRESS_SIMPLEX_FULL6_K8_HISTORY_OFF",
        "variant": method,
        "rollout": public,
        "rollout_sha256": public["identity_sha256"],
        "rollout_status": public["status"],
        "accepted_k": len(public["accepted_receipt_sha256"]),
        "tau": float(public["tau_final"]),
        "transaction": transaction,
        "assembly": assembly,
        "history_mode": "OFF",
        "router_visible_history_item_count": 0,
        "raw_historical_request_replay_count": 0,
        "projected_key_historical_sketch_construction_count": 0,
        "functional_h_status": "INACTIVE_BY_HISTORY_MODE_OFF",
        "functional_h_controller_input_count": 0,
        "functional_h_decision_influence_count": 0,
        "functional_h_model_forward_count": 0,
        "structural_h_status": "INACTIVE_BY_HISTORY_MODE_OFF",
        "structural_h_controller_input_count": 0,
        "structural_h_decision_influence_count": 0,
        "structural_h_model_forward_count": 0,
        "inner_k_heldout_evaluation_count": 0,
        "inner_k_functional_p_probe_endpoint_count": P1R23_GRID_COUNT * 6,
        "full_six_context_ordinals": list(range(6)),
        "full_six_context_sha256": objective_plan.context_sha256,
        "full_six_microbatch_partition": [
            list(item.prepared.request_ordinals) for item in objective_plan.batches
        ],
        "functional_p_decision_influence_count": 1 if arm is FixedE8Arm.SOFT else 0,
        "functional_h_decision_influence_count": 0,
        "retry_count": 0,
    }, load


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
    if method not in METHODS or len(stream_batches) != ROUND_COUNT:
        raise ODEBFContractError("P1R20 trajectory matrix/round count differs")
    if any(len(batch) != BATCH_SIZE for batch in stream_batches):
        raise ODEBFContractError("P1R20 stream is not ten B10 rounds")
    if stream.get("root_digest") is None or len(stream.get("requests", ())) != 100:
        raise ODEBFContractError("P1R20 fresh seal identity differs")

    started = time.perf_counter()
    initial_hashes = _weight_hashes(touched)
    if initial_hashes != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("P1R20 W0 entry differs")
    state = ArmRuntimeState(
        P1Arm.R_BF,
        P1HistoryLedger(layer_order=tuple(int(item) for item in hparams.layers), maximum_records=100),
        ComputeLedger(),
        ArmWeightSnapshot(P1Arm.R_BF, 0, base_receipt.parameter_sha256, canonical_hash({"method": method, "w0": initial_hashes})),
        dict(base_values),
    )
    is_ours = True
    outer_entry_p_cache = None
    if is_ours:
        outer_population = tuple(
            population_by_sha256[item] for item in theta0_cache.request_order
        )
        outer_snapshot = _entry_parameter_snapshot_sha256(
            model, dict(base_receipt.parameter_sha256)
        )
        outer_counter = ModelForwardCounter(model, state.ledger)
        try:
            outer_entry_p_cache = build_outer_entry_pretrained_cache(
                model,
                tokenizer,
                outer_population,
                theta0_cache,
                outer_entry_snapshot_sha256=outer_snapshot,
            )
        finally:
            outer_counter.close()
    w0_payload = {
        "schema": "ode-edit-s05-p1r23-full6-historical-w0/v1",
        "status": "NOT_EVALUATED_55_POSTCOMMIT_B10_CONTRACT",
        "method": method,
        "history_count": 0,
        "parameter_sha256": initial_hashes,
        "model_forward_count": 0,
        "evaluator_access_count": 0,
    }
    w0_sha = _atomic_write_once(
        raw_root / "round-00-endpoint.json",
        w0_payload,
    )
    round_hashes: list[str] = []
    round_summaries: list[dict[str, Any]] = []
    historical_nondegenerate_rounds: list[int] = []
    historical_influence_rounds: list[int] = []
    for round_index, requests in enumerate(stream_batches, start=1):
        history_count_at_entry = len(state.history.snapshot().active_records)
        if history_count_at_entry != 0:
            raise ODEBFStateError("Sequential-NoH router history is not empty")
        round_started = time.perf_counter()
        before_values = _weight_values(touched)
        before_hashes = _weight_hashes(touched)
        if round_index > 1 and before_hashes == initial_hashes:
            raise ODEBFStateError("Sequential-NoH did not accumulate prior round weights")
        phase_entry = _ledger_snapshot(state.ledger)
        if is_ours:
            if outer_entry_p_cache is None:
                raise ODEBFStateError("Sequential-NoH functional-P cache is absent")
            edit, load = _run_ours_round(
                model,
                tokenizer,
                requests,
                alias=alias,
                method=method,
                hparams=hparams,
                projector=projector,
                contexts=contexts,
                covariance_registry=covariance_registry,
                projector_sha256=projector_sha256,
                controller_lock=controller_lock,
                state=state,
                request_by_sha256=request_by_sha256,
                population_by_sha256=population_by_sha256,
                schedule=schedule,
                theta0_cache=theta0_cache,
                touched=touched,
                raw_root=raw_root,
                stages=stages,
                mutation_lock=mutation_lock,
                round_index=round_index,
                outer_entry_p_cache=outer_entry_p_cache,
                job_ledger=state.ledger,
            )
        else:
            counter = ModelForwardCounter(model, state.ledger)
            try:
                edit, load = _run_alpha_round(
                    model,
                    tokenizer,
                    requests,
                    hparams=hparams,
                    projector=projector,
                    contexts=contexts,
                    state=state,
                    touched=touched,
                    mutation_lock=mutation_lock,
                    residual_tolerance=controller_lock.residual_tolerance,
                    round_index=round_index,
                )
            finally:
                counter.close()
        phase_after_edit = _ledger_snapshot(state.ledger)
        after_hashes = _weight_hashes(touched)
        if after_hashes == before_hashes:
            raise ODEBFStateError("P1R20 round produced no persistent edit")

        terminal_event = canonical_hash(
            {
                "instruction_id": INSTRUCTION_ID,
                "method": method,
                "round": round_index,
                "entry": before_hashes,
                "endpoint": after_hashes,
            }
        )
        history_receipt = {
            "schema": "ode-edit-s05-p1r23-sequential-noh-history-off/v1",
            "status": "INACTIVE_BY_HISTORY_MODE_OFF",
            "append_call_count": 0,
            "router_visible_history_item_count": 0,
            "raw_historical_request_replay_count": 0,
            "projected_key_sketch_construction_count": 0,
            "functional_h_input_influence_cost_count": 0,
            "structural_h_input_influence_cost_count": 0,
            "terminal_event_sha256": terminal_event,
        }
        history_receipt["identity_sha256"] = canonical_hash(history_receipt)
        history_count_after_commit = len(state.history.snapshot().active_records)
        if history_count_after_commit != 0:
            raise ODEBFStateError("Sequential-NoH appended router history")
        phase_after_history = _ledger_snapshot(state.ledger)
        _validate_history_off_edit_receipt(edit)

        evaluated, evaluation_summary = _evaluate_batches(
            model,
            tokenizer,
            alias=alias,
            method=method,
            round_index=round_index,
            batches=stream_batches[:round_index],
            dataset_path=dataset_path,
            state_sha256=canonical_hash(after_hashes),
            ledger=state.ledger,
        )
        phase_after_historical_eval = _ledger_snapshot(state.ledger)
        kl = (
            _teacher_kl(
                model,
                tokenizer,
                theta0_cache=theta0_cache,
                population_by_sha256=population_by_sha256,
                ledger=state.ledger,
            )
            if round_index in EVALUATION_ROUNDS
            else {
                "status": "NOT_EVALUATED_OUTER_CHECKPOINT_SCHEDULE",
                "model_forward_count": 0,
                "processed_token_count": 0,
                "decision_influence_count": 0,
            }
        )
        phase_after_general_eval = _ledger_snapshot(state.ledger)
        elapsed = time.perf_counter() - round_started
        payload = {
            "schema": "ode-edit-s05-p1r23-progress-simplex-sequential-noh-round-endpoint/v1",
            "instruction_id": INSTRUCTION_ID,
            "alias": alias,
            "method": method,
            "round": round_index,
            "history_count_at_entry": history_count_at_entry,
            "history_count_after_commit": history_count_after_commit,
            "edit": edit,
            "history": history_receipt,
            "evaluation": evaluated,
            "evaluation_summary": evaluation_summary,
            "teacher_kl": kl,
            "entry_parameter_sha256": before_hashes,
            "endpoint_parameter_sha256": after_hashes,
            "terminal_event_sha256": terminal_event,
            "current_batch_enters_history_after_endpoint_commit": True,
            "future_batch_controller_access_count": 0,
            "inner_k_heldout_evaluation_count": 0,
            "retry_count": 0,
            "first_hit_evaluation_count": 0,
            "phase_compute": {
                "edit_core_k8_refresh_or_baseline": _ledger_delta(
                    phase_entry, phase_after_edit
                ),
                "history_routing_and_append": _ledger_delta(
                    phase_after_edit, phase_after_history
                ),
                "historical_current_all_edited_evaluation": _ledger_delta(
                    phase_after_history, phase_after_historical_eval
                ),
                "loc_teacher_kl_downstream_evaluation": _ledger_delta(
                    phase_after_historical_eval, phase_after_general_eval
                ),
                "downstream_status": (
                    "NOT_RECORDED_NO_PINNED_P1R19_DOWNSTREAM_EVALUATOR"
                ),
                "saved_inner_evaluation_contract": {
                    "inner_k_heldout_evaluation_count": 0,
                    "inner_k_functional_probe_count": 0,
                    "called_early_stopping": False,
                },
            },
            "round_wall_seconds": elapsed,
            "host_maxrss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
        round_sha = _atomic_write_once(
            raw_root / f"round-{round_index:02d}-endpoint.json", payload
        )
        round_hashes.append(round_sha)
        round_summaries.append(
            {
                "round": round_index,
                "receipt_sha256": round_sha,
                "summary": evaluation_summary,
                "teacher_kl": kl,
                "wall_seconds": elapsed,
            }
        )
        stages.record(
            f"post_sequential_round_{round_index}",
            {
                "method": method,
                "round": round_index,
                "receipt_sha256": round_sha,
                "history_count": round_index * BATCH_SIZE,
                "endpoint_parameter_sha256": canonical_hash(after_hashes),
            },
        )
        del before_values
        _observed_memory(state.ledger)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    terminal_weights = _weight_hashes(touched)
    persistent_h_inactivity = True
    with torch.no_grad():
        for name, parameter in touched.items():
            parameter.copy_(base_values[name].to(device=parameter.device))
    restored = _weight_hashes(touched) == initial_hashes
    if not restored:
        raise ODEBFStateError("P1R20 terminal W0 restore differs")
    artifact_guard.assert_unchanged()
    _observed_memory(state.ledger)
    terminal = {
        "schema": "ode-edit-s05-p1r23-progress-simplex-sequential-noh-terminal/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "PROGRESS_SIMPLEX_SEQUENTIAL_NOH_T10_COMPLETE",
        "alias": alias,
        "method": method,
        "source_head": source_head,
        "fresh_seal_root": stream["root_digest"],
        "all_request_order_sha256": stream["all_request_order_sha256"],
        "batch_order_sha256": stream["batch_ordered_request_digest_v1"],
        "w0_endpoint_receipt_sha256": w0_sha,
        "round_receipt_sha256": round_hashes,
        "round_summaries": round_summaries,
        "terminal_edited_parameter_sha256": terminal_weights,
        "final_w0_restored": restored,
        "persistent_commit_count": ROUND_COUNT,
        "history_mode": "OFF",
        "history_append_count": 0,
        "history_final_count": 0,
        "historical_sketch": {"status": "INACTIVE_BY_HISTORY_MODE_OFF"},
        "raw_historical_request_replay_count": 0,
        "projected_key_historical_sketch_construction_count": 0,
        "functional_h_input_influence_cost_count": 0,
        "structural_h_input_influence_cost_count": 0,
        "historical_h_nondegenerate_rounds": [],
        "historical_h_decision_influence_rounds": [],
        "historical_h_persistent_inactivity": persistent_h_inactivity,
        "postcommit_cumulative_b10_evaluation_count": sum(range(1, 11)),
        "wide_locality_teacher_kl_checkpoint_rounds": list(EVALUATION_ROUNDS),
        "future_batch_controller_access_count": 0,
        "inner_k_heldout_evaluation_count": 0,
        "retry_count": 0,
        "backtracking_count": 0,
        "first_hit_evaluation_count": 0,
        "compute": state.ledger.raw_free_payload(),
        "job_initialization_compute": job_ledger.raw_free_payload(),
        "elapsed_seconds": time.perf_counter() - started,
        "artifact_receipt": asdict(artifact_receipt),
        "numerical_lock_sha256": numerical_sha256,
        "context_sha256": context_sha256,
        "cuda_runtime_receipt": dict(cuda_runtime_receipt),
        "scientific_promotion_authorized": False,
    }
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p1r23-progress-simplex-sequential-noh-manifest/v1",
        "status": terminal["status"],
        "alias": alias,
        "method": method,
        "source_head": source_head,
        "terminal_sha256": terminal_sha,
        "round_receipt_sha256": round_hashes,
        "final_w0_restored": restored,
        "scientific_promotion_authorized": False,
    }
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    return {
        "status": terminal["status"],
        "alias": alias,
        "method": method,
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "final_w0_restored": restored,
    }


__all__ = [
    "INSTRUCTION_ID",
    "METHODS",
    "expected_historical_h0_result_name",
    "run_historical_h0_sequential_trajectory",
]
