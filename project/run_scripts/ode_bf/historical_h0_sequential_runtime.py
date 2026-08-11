"""P1R20 ten-round historical-H0 sequential validation runtime.

This is an additive production path.  It reuses the accepted P1R19 BG writer
and strength-preserving router, but removes all inner-k held-out probes.  A
round is action-frozen before its single weight transaction; the committed
batch enters history only after that transaction verifies.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import math
import resource
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .accounting import ComputeLedger
from .common_cold_coordinate import (
    CommonColdScale,
    CommonColdScaleMetric,
    common_cold_bootstrap,
)
from .common_coldcoord_fixed_e8_runtime import (
    CommonColdArm,
    _parameter_contract_sha256,
    _run_common_arm,
)
from .contracts import BATCH_SIZE, ODEBFContractError, ODEBFStateError, canonical_hash
from .cold_start_target import capture_cold_z_base, cold_lookup_positions
from .fixed_e8_runtime import FixedE8EntryCapture, FixedE8ReceiptRecorder
from .functional import tensor_sha256
from .p0_runtime import ModelForwardCounter
from .p1_backend import capture_p1_native_entry
from .p1_evaluator import load_counterfact_cases_after_freeze
from .p1_replay import evaluate_next_token_log_probs, samplewise_teacher_kl
from .p1_runtime import (
    ArmRuntimeState,
    ComponentTimer,
    _assemble_candidates,
    _atomic_write_once,
    _finalize_prepared_history,
    _history_keys,
    _observed_memory,
    _prepare_history_batch,
)
from .p1_state import ArmWeightSnapshot, P1Arm, P1HistoryLedger
from .p1_stepwise import StepwiseActionFreeze, evaluate_counterfact_stepwise_primary
from .request_digest import ordered_request_digest_v1
from .transaction import AtomicBatchTransaction


INSTRUCTION_ID = "ODEEDIT-S05-ODE-BF-HISTORICAL-H0-BG-SEQUENTIAL-P1R20-V1"
METHODS = ("BG-NEUTRAL", "BG-SOFT-H", "MEMIT", "ALPHAEDIT")
ROUND_COUNT = 10
HISTORY_COUNTS = tuple(range(0, 100, 10))


def expected_historical_h0_result_name(alias: str, method: str) -> str:
    if method not in METHODS:
        raise ODEBFContractError("P1R20 method identity differs")
    return (
        "s05-p1r20-historical-h0-bg-sequential-"
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


def _normalize_easyedit_requests(
    requests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(list(requests))
    for item in normalized:
        target = str(item["target_new"])
        item["target_new"] = target if target.startswith(" ") else " " + target
        prompt = str(item["prompt"])
        subject = str(item["subject"])
        if "{}" not in prompt:
            if subject not in prompt:
                raise ODEBFContractError("P1R20 baseline subject is absent")
            item["prompt"] = prompt.replace(subject, "{}")
    return normalized


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


def _run_memit_round(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    memit_hparams: Any,
    touched: Mapping[str, torch.nn.Parameter],
) -> tuple[dict[str, Any], dict[int, float]]:
    from easyeditor.models.memit import memit_main
    layer_stats_module = importlib.import_module("easyeditor.models.rome.layer_stats")

    before_values = _weight_values(touched)
    before_hashes = _weight_hashes(touched)
    original_load_dataset = layer_stats_module.load_dataset

    def forbid_dataset(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise ODEBFContractError("P1R20 MEMIT cache miss requested dataset access")

    started = time.perf_counter()
    try:
        layer_stats_module.load_dataset = forbid_dataset
        with contextlib.redirect_stdout(__import__("io").StringIO()), contextlib.redirect_stderr(
            __import__("io").StringIO()
        ):
            returned, originals = memit_main.apply_memit_to_model(
                model,
                tokenizer,
                _normalize_easyedit_requests(requests),
                memit_hparams,
                copy=False,
                return_orig_weights=True,
                cache_template=None,
            )
    finally:
        layer_stats_module.load_dataset = original_load_dataset
    if returned is not model or set(originals) != set(touched):
        raise ODEBFContractError("P1R20 MEMIT touched inventory differs")
    if any(tensor_sha256(originals[name]) != before_hashes[name] for name in touched):
        raise ODEBFContractError("P1R20 MEMIT entry snapshot differs")
    after_hashes = _weight_hashes(touched)
    if after_hashes == before_hashes:
        raise ODEBFStateError("P1R20 MEMIT produced no physical write")
    load = _layer_load(before_values, touched, memit_hparams)
    return {
        "backend": "PINNED_EASYEDIT_MEMIT_JOINT_B10",
        "entry_parameter_sha256": before_hashes,
        "endpoint_parameter_sha256": after_hashes,
        "wall_seconds": time.perf_counter() - started,
        "covariance_recompute_count": 0,
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
) -> tuple[dict[str, Any], dict[int, float]]:
    request_order = ordered_request_digest_v1(
        [str(item["request_sha256"]) for item in requests]
    )
    entry_values = _weight_values(touched)
    entry_hashes = _weight_hashes(touched)
    z_base = capture_cold_z_base(model, tokenizer, requests, hparams)
    lookup_positions = cold_lookup_positions(
        tokenizer, requests, contexts, fact_token_strategy=hparams.fact_token
    )
    layer_name = hparams.layer_module_tmp.format(int(hparams.layers[-1]))
    metric = CommonColdScaleMetric.from_z_base(
        z_base, request_order, CommonColdScale.BATCH_GLOBAL
    )
    bootstrap_target, bootstrap = common_cold_bootstrap(
        model,
        tokenizer,
        requests,
        contexts,
        target_layer_name=layer_name,
        lookup_positions=lookup_positions,
        z_base=z_base,
        metric=metric,
        ledger=state.ledger,
    )
    arm = CommonColdArm.BG_NEUTRAL if method == "BG-NEUTRAL" else CommonColdArm.BG_SOFT
    recorder = FixedE8ReceiptRecorder(raw_root / f"round-{round_index:02d}", arm, _atomic_write_once)
    capture = FixedE8EntryCapture(entry_values, entry_hashes)
    rollout, initial_contract = _run_common_arm(
        model,
        tokenizer,
        requests,
        alias=alias,
        arm=arm,
        bootstrap_target=bootstrap_target,
        z_base=z_base,
        metric=metric,
        lookup_positions=lookup_positions,
        target_layer_name=layer_name,
        capture=capture,
        hparams=hparams,
        projector=projector,
        contexts=contexts,
        covariance_registry=covariance_registry,
        projector_sha256=projector_sha256,
        lock=controller_lock,
        arm_state=state,
        request_by_sha256=request_by_sha256,
        population_by_sha256=population_by_sha256,
        schedule=schedule,
        outer_entry_p_cache=None,
        theta0_cache=theta0_cache,
        touched=touched,
        recorder=recorder,
        stages=stages,
        typed_zero_positive=False,
        record_six_context_objectives=False,
        strength_preserving=True,
        sequential_lightweight=True,
    )
    if rollout.k_acc != 8 or float(rollout.accepted_t) != 1.0 or not rollout.snapshots:
        raise ODEBFStateError("P1R20 Ours did not complete K8/tau1")
    final = rollout.snapshots[-1]
    candidates, assembly = _assemble_candidates(touched, entry_hashes, final.factors)
    transaction = _commit_candidates(
        touched,
        candidates,
        transaction_id=canonical_hash(
            {
                "method": method,
                "round": round_index,
                "rollout": rollout.rollout_sha256,
            }
        ),
        mutation_lock=mutation_lock,
    )
    load = {
        int(layer): float(value)
        for layer, value in zip(
            hparams.layers,
            final.capacity_payload["cumulative_bf16_capacity"],
            strict=True,
        )
    }
    stepwise = [
        {
            "accepted_index": item.accepted_index,
            "tau": float(item.tau),
            "routing": item.routing_payload["routing"],
            "progress": item.progress_payload,
            "capacity": item.capacity_payload,
            "structural_functional": item.structural_payload,
        }
        for item in rollout.snapshots
    ]
    historical_influence = any(
        any(
            str(score.get("label")) == "structural_historical"
            and int(score.get("influence_count", 0)) > 0
            for score in item["routing"].get("scores", ())
        )
        for item in stepwise
    )
    return {
        "backend": "P1R19_BG_STRENGTH_PRESERVING_K8",
        "variant": arm.value,
        "bootstrap": bootstrap,
        "initial_contract": initial_contract,
        "rollout_sha256": rollout.rollout_sha256,
        "rollout_status": rollout.status,
        "accepted_k": rollout.k_acc,
        "tau": float(rollout.accepted_t),
        "transaction": transaction,
        "assembly": assembly,
        "receipt_links": recorder.links(),
        "stepwise": stepwise,
        "historical_h_decision_influence": historical_influence,
        "historical_bf_classification": (
            "HISTORICAL_H_ACTIVE"
            if historical_influence
            else "NON_HISTORICAL_BF"
            if method == "BG-SOFT-H" and round_index >= 2
            else "NOT_REQUIRED_OR_EMPTY_HISTORY"
        ),
        "inner_k_heldout_evaluation_count": 0,
        "inner_k_functional_probe_count": 0,
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
    memit_hparams: Any | None = None,
) -> dict[str, Any]:
    if method not in METHODS or len(stream_batches) != ROUND_COUNT:
        raise ODEBFContractError("P1R20 trajectory matrix/round count differs")
    if any(len(batch) != BATCH_SIZE for batch in stream_batches):
        raise ODEBFContractError("P1R20 stream is not ten B10 rounds")
    if method == "MEMIT" and memit_hparams is None:
        raise ODEBFContractError("P1R20 MEMIT hparams are absent")
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
    w0_evaluations, w0_summary = _evaluate_batches(
        model,
        tokenizer,
        alias=alias,
        method=method,
        round_index=0,
        batches=stream_batches,
        dataset_path=dataset_path,
        state_sha256=canonical_hash(initial_hashes),
        ledger=state.ledger,
    )
    w0_sha = _atomic_write_once(
        raw_root / "round-00-endpoint.json",
        {
            "schema": "ode-edit-s05-p1r20-round-endpoint/v1",
            "method": method,
            "round": 0,
            "evaluation": w0_evaluations,
            "summary": w0_summary,
            "history_count": 0,
            "action_frozen": True,
        },
    )
    round_hashes: list[str] = []
    round_summaries: list[dict[str, Any]] = []
    first_edited_metrics: dict[int, dict[str, Any]] = {}
    for round_index, requests in enumerate(stream_batches, start=1):
        if len(state.history.snapshot().active_records) != HISTORY_COUNTS[round_index - 1]:
            raise ODEBFStateError("P1R20 round-entry history count differs")
        round_started = time.perf_counter()
        before_values = _weight_values(touched)
        before_hashes = _weight_hashes(touched)
        phase_entry = _ledger_snapshot(state.ledger)
        counter = ModelForwardCounter(model, state.ledger)
        try:
            if method in ("BG-NEUTRAL", "BG-SOFT-H"):
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
                )
            elif method == "ALPHAEDIT":
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
            else:
                edit, load = _run_memit_round(
                    model,
                    tokenizer,
                    requests,
                    memit_hparams=memit_hparams,
                    touched=touched,
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
        history_counter = ModelForwardCounter(model, state.ledger)
        try:
            prepared = _prepare_history_batch(
                model,
                tokenizer,
                arm_state=state,
                requests=requests,
                collision_by_request=collision_by_request,
                hparams=hparams,
                projector=projector,
                contexts=contexts,
                terminal_event_sha256=terminal_event,
                load_increment_by_layer=load,
            )
        finally:
            history_counter.close()
        history_receipt = _finalize_prepared_history(state, prepared)
        history_receipt["load_increment_by_layer"] = {
            str(layer): float(value) for layer, value in sorted(load.items())
        }
        history_receipt["cumulative_load_by_layer"] = {
            str(layer): float(value)
            for layer, value in sorted(state.history.cumulative_load().items())
        }
        if len(state.history.snapshot().active_records) != round_index * BATCH_SIZE:
            raise ODEBFStateError("P1R20 history append count differs")
        phase_after_history = _ledger_snapshot(state.ledger)

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
        first_edited_metrics.setdefault(
            round_index, evaluated[-1]["primary"]["metrics"]
        )
        phase_after_historical_eval = _ledger_snapshot(state.ledger)
        kl = _teacher_kl(
            model,
            tokenizer,
            theta0_cache=theta0_cache,
            population_by_sha256=population_by_sha256,
            ledger=state.ledger,
        )
        phase_after_general_eval = _ledger_snapshot(state.ledger)
        elapsed = time.perf_counter() - round_started
        payload = {
            "schema": "ode-edit-s05-p1r20-round-endpoint/v1",
            "instruction_id": INSTRUCTION_ID,
            "alias": alias,
            "method": method,
            "round": round_index,
            "history_count_at_entry": (round_index - 1) * BATCH_SIZE,
            "history_count_after_commit": round_index * BATCH_SIZE,
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
    with torch.no_grad():
        for name, parameter in touched.items():
            parameter.copy_(base_values[name].to(device=parameter.device))
    restored = _weight_hashes(touched) == initial_hashes
    if not restored:
        raise ODEBFStateError("P1R20 terminal W0 restore differs")
    artifact_guard.assert_unchanged()
    _observed_memory(state.ledger)
    terminal = {
        "schema": "ode-edit-s05-p1r20-terminal/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "HISTORICAL_H0_SEQUENTIAL_T10_COMPLETE",
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
        "history_append_count": ROUND_COUNT,
        "history_final_count": len(state.history.snapshot().active_records),
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
        "schema": "ode-edit-s05-p1r20-manifest/v1",
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
