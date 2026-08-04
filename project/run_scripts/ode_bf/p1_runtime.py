"""Scientific P1: four isolated arms over four sequential joint-B10 edits."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import resource
import subprocess
import threading
import time
import traceback
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from .accounting import ComputeLedger
from .alpha_backend import fresh_contexts_twice, load_original_bf16, seed_all
from .artifacts import ODEBFArtifactGuard, load_rooted_json, sha256_file
from .barriers import FunctionalHPVerdict, functional_replay_risk
from .benchmark import BatchSuccessReceipt
from .contracts import (
    BATCH_SIZE,
    COMMON_SEED,
    FIXED_K,
    MODEL_ALIASES,
    ODEBFContractError,
    ODEBFStateError,
    canonical_hash,
)
from .evaluator import ModelEvaluationReceipt, evaluate_counterfact_rewrite_batch
from .first_hit import FeasibilityVerdict
from .functional import (
    CumulativeBF16FunctionalTrial,
    WaypointFactor,
    assemble_effective_bf16,
    tensor_sha256,
)
from .p0_runtime import ModelForwardCounter
from .p1_backend import (
    CandidateBF16FunctionalTrial,
    P1DynamicField,
    P1NativeCapture,
    PinnedCovarianceRegistry,
    build_p1_dynamic_field,
    build_p1_frozen_field_from_capture,
    capture_committed_history_key_views,
    capture_p1_native_entry,
    evaluate_controller_margin,
    signed_progress_gradient,
    write_aware_target_velocity,
)
from .p1_controller import (
    AcceptedLayerContribution,
    MatchedRawVelocity,
    P1ControllerLock,
    P1RoutingBuild,
    build_p1_routing_problem,
    project_matched_bf_velocity,
    project_shared_raw_velocity,
    rebind_shared_problem,
    scaled_waypoint_factors,
    solve_matched_raw_velocity,
)
from .p1_evaluator import (
    CounterFactPrimaryReceipt,
    EndpointActionFreeze,
    PairedPrimaryFloorReceipt,
    evaluate_counterfact_primary_batch,
    load_counterfact_cases_after_freeze,
    pair_primary_native_floor,
)
from .p1_replay import (
    FunctionalReplayPair,
    Theta0TeacherCache,
    build_theta0_teacher_cache,
    capture_pretrained_entry_kl,
    evaluate_functional_replay_pair,
    evaluate_next_token_log_probs,
    evaluate_target_new_nlls,
    samplewise_teacher_kl,
)
from .p1_selection import (
    load_p1_population_requests,
    load_p1_stream_batches,
    verify_p1_population_seal,
    verify_p1_stream_seal,
)
from .p1_state import (
    P1_ARM_ORDER,
    ArmWeightSnapshot,
    FixedK8P1Selector,
    P1Arm,
    P1EndpointSelection,
    P1HistoryLedger,
    P1HistoryRecord,
    P1Waypoint,
    ProspectiveP1HistoryBatch,
    evaluate_p1a_resource_gate,
    restore_arm_snapshot,
    snapshot_touched_weights,
)
from .request_digest import ordered_request_digest_v1
from .routing import RoutingProblem, RoutingStatus, verify_backtracked_candidate
from .sampling import SampleLineage, StatelessReplaySchedule, load_p1_sampling_seal
from .transaction import AtomicBatchTransaction


INSTRUCTION_ID = "ODEEDIT-S04-ODE-BF-SEQUENTIAL-B10-NATIVE-FLOOR-P1-V1"
EXPECTED_BASE = "a5b7a60237c85432cbded8487ff04602cf4094e6"
RESULT_TOKEN = "seqb10-native-floor-p1-v1"
SEQUENTIAL_BATCHES = 4


def expected_p1_result_name(alias: str) -> str:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("P1 result alias differs")
    return f"s04-p1-seqb10-native-floor-{alias}-v1"


def _atomic_write_once(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError("P1 receipt is create-once")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("P1 temporary receipt exists")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


class P1StageRecorder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.sequence = 0
        self.last_stage: str | None = None

    def record(self, stage: str, payload: Mapping[str, Any]) -> str:
        self.sequence += 1
        digest = _atomic_write_once(
            self.root / f"stage-{self.sequence:03d}-{stage}.json",
            {
                "schema": "ode-edit-s04-ode-bf-p1-stage/v1",
                "sequence": self.sequence,
                "stage": stage,
                "payload": dict(payload),
            },
        )
        self.last_stage = stage
        return digest


class ComponentTimer:
    def __init__(self, ledger: ComputeLedger) -> None:
        self.ledger = ledger

    @contextlib.contextmanager
    def measure(self, component: str) -> Iterator[None]:
        if torch.cuda.is_available():
            torch.cuda.synchronize(0)
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
        else:
            begin = end = None
        started = time.perf_counter()
        try:
            yield
        finally:
            wall = time.perf_counter() - started
            gpu = 0.0
            if begin is not None and end is not None:
                end.record()
                torch.cuda.synchronize(0)
                gpu = float(begin.elapsed_time(end)) / 1000.0
            self.ledger.add_time(component, wall_seconds=wall, gpu_seconds=gpu)


@dataclass(slots=True)
class ArmRuntimeState:
    arm: P1Arm
    history: P1HistoryLedger
    ledger: ComputeLedger
    snapshot_receipt: ArmWeightSnapshot
    snapshot_values: dict[str, torch.Tensor]
    completed_batches: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RetentionFloorReceipt:
    prior_batch_count: int
    native_correct_count: int
    ours_correct_count: int
    denominator: int
    native_bits_sha256: str
    ours_bits_sha256: str
    loss_bits_sha256: str
    loss_count: int
    passed: bool


@dataclass(slots=True)
class ReplayEntryState:
    history_requests: tuple[Mapping[str, Any], ...]
    history_entry_nll: torch.Tensor
    pretrained_requests: tuple[Mapping[str, Any], ...]
    pretrained_entry_kl: torch.Tensor
    history_entry_sha256: str
    pretrained_entry_sha256: str
    schedule_sha256: str


@dataclass(frozen=True, slots=True)
class PreparedHistoryBatch:
    prospective: ProspectiveP1HistoryBatch
    key_identity_sha256: str
    before_state_sha256: str
    load_increment_by_layer: tuple[tuple[int, float], ...]


@dataclass(slots=True)
class FrozenMatchedBundle:
    field: P1DynamicField
    signed_progress: Any
    source_build: P1RoutingBuild
    raw: MatchedRawVelocity
    target_state: torch.Tensor
    source_arm: P1Arm
    source_compute_delta: dict[str, int]
    source_wall_seconds: float


def _assert_arm_batch_transition(
    state: ArmRuntimeState,
    payload: Mapping[str, Any],
    *,
    sequential_batch: int,
    history_before_sha256: str,
) -> None:
    """Fail close on partial B10 persistence or an untracked arm transition."""

    status = payload.get("status")
    snapshot = state.history.snapshot()
    if status == "COMMITTED":
        if (
            state.snapshot_receipt.sequential_batch_completed
            != sequential_batch + 1
            or snapshot.version != sequential_batch + 1
            or len(snapshot.active_records) != (sequential_batch + 1) * BATCH_SIZE
            or snapshot.audit_obsolete_records
        ):
            raise ODEBFStateError("P1 committed arm transition is not an exact B10 chain")
        if snapshot.digest == history_before_sha256:
            raise ODEBFStateError("P1 committed B10 did not advance history")
        return
    if status == "R_BF_NATIVE_PRIMARY_FLOOR_FAIL":
        if (
            state.arm is not P1Arm.R_BF
            or state.snapshot_receipt.sequential_batch_completed != sequential_batch
            or snapshot.digest != history_before_sha256
        ):
            raise ODEBFStateError("P1 failed R_BF floor mutated persistent arm state")
        return
    raise ODEBFStateError("P1 arm batch ended without a canonical transition status")


def _source_freeze(repo_root: Path, source_head: str) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    if head != source_head:
        raise ODEBFContractError("P1 execution HEAD differs")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    if status:
        raise ODEBFContractError("P1 execution source is not clean")


def _virtual_context(
    model: torch.nn.Module,
    factors: Mapping[str, Sequence[WaypointFactor]],
) -> AbstractContextManager[Any]:
    return (
        CumulativeBF16FunctionalTrial(model, factors, row_block=64)
        if factors
        else nullcontext()
    )


def _factor_state(
    entry_sha256: Mapping[str, str],
    factors: Mapping[str, Sequence[WaypointFactor]],
) -> str:
    payload: dict[str, Any] = {"entry": dict(sorted(entry_sha256.items())), "factors": {}}
    for name, values in sorted(factors.items()):
        payload["factors"][name] = [
            {
                "layer": item.layer,
                "cycle": item.correction_cycle,
                "step": item.step_in_cycle,
                "ordinal": item.factor_ordinal,
                "theta": item.theta,
                "left": tensor_sha256(item.left),
                "right": tensor_sha256(item.right),
                "rank": item.left.shape[1],
            }
            for item in sorted(values, key=lambda value: value.order_key)
        ]
    return canonical_hash(payload)


def _merge_factors(
    base: Mapping[str, Sequence[WaypointFactor]],
    increment: Mapping[str, WaypointFactor],
) -> dict[str, tuple[WaypointFactor, ...]]:
    result = {name: tuple(values) for name, values in base.items()}
    for name, factor in increment.items():
        values = result.get(name, ()) + (factor,)
        if len({item.order_key for item in values}) != len(values):
            raise ODEBFContractError("P1 cumulative factor order repeats")
        result[name] = tuple(sorted(values, key=lambda item: item.order_key))
    return result


def _evaluate_rewrite(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    factors: Mapping[str, Sequence[WaypointFactor]],
    ledger: ComputeLedger,
) -> ModelEvaluationReceipt:
    with _virtual_context(model, factors):
        receipt = evaluate_counterfact_rewrite_batch(
            model, tokenizer, requests, model_alias=alias
        )
    ledger.increment("evaluator_forward", receipt.model_forward_count)
    ledger.increment("evaluator_tokens", receipt.processed_token_count)
    return receipt


def _evaluate_native_rewrite(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    candidates: Mapping[str, torch.Tensor],
    ledger: ComputeLedger,
) -> ModelEvaluationReceipt:
    with CandidateBF16FunctionalTrial(model, candidates):
        receipt = evaluate_counterfact_rewrite_batch(
            model, tokenizer, requests, model_alias=alias
        )
    ledger.increment("evaluator_forward", receipt.model_forward_count)
    ledger.increment("evaluator_tokens", receipt.processed_token_count)
    return receipt


def _evaluate_primary(
    model: torch.nn.Module,
    tokenizer: Any,
    cases: Sequence[Any],
    *,
    alias: str,
    freeze: EndpointActionFreeze,
    factors: Mapping[str, Sequence[WaypointFactor]],
    ledger: ComputeLedger,
) -> CounterFactPrimaryReceipt:
    with _virtual_context(model, factors):
        receipt = evaluate_counterfact_primary_batch(
            model, tokenizer, cases, model_alias=alias, freeze=freeze
        )
    ledger.increment("evaluator_forward", receipt.model_forward_count)
    ledger.increment("evaluator_tokens", receipt.processed_token_count)
    return receipt


def _evaluate_native_primary(
    model: torch.nn.Module,
    tokenizer: Any,
    cases: Sequence[Any],
    *,
    alias: str,
    freeze: EndpointActionFreeze,
    candidates: Mapping[str, torch.Tensor],
    ledger: ComputeLedger,
) -> CounterFactPrimaryReceipt:
    with CandidateBF16FunctionalTrial(model, candidates):
        receipt = evaluate_counterfact_primary_batch(
            model, tokenizer, cases, model_alias=alias, freeze=freeze
        )
    ledger.increment("evaluator_forward", receipt.model_forward_count)
    ledger.increment("evaluator_tokens", receipt.processed_token_count)
    return receipt


def _primary_payload(receipt: CounterFactPrimaryReceipt) -> dict[str, Any]:
    return receipt.raw_free_payload()


def _paired_floor_payload(receipt: PairedPrimaryFloorReceipt) -> dict[str, Any]:
    return {
        "all_primary_pass": receipt.all_primary_pass,
        "pairwise_miss_review": receipt.pairwise_miss_review,
        "request_span_parity": receipt.request_span_parity,
        "verdicts": [asdict(item) for item in receipt.table.verdicts],
        "paired": [asdict(item) for item in receipt.paired],
    }


def _observed_memory(ledger: ComputeLedger) -> None:
    allocated = int(torch.cuda.max_memory_allocated(0)) if torch.cuda.is_available() else 0
    reserved = int(torch.cuda.max_memory_reserved(0)) if torch.cuda.is_available() else 0
    ledger.observe_memory(
        allocated_bytes=allocated,
        reserved_bytes=reserved,
        maxrss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )


def _history_keys(ledger: P1HistoryLedger, layers: Sequence[int], *, risk: bool) -> dict[int, torch.Tensor]:
    return {
        int(layer): (
            ledger.risk_keys(int(layer)) if risk else ledger.solve_keys(int(layer))
        )
        for layer in layers
    }


def _arm_history_actions(
    field: P1DynamicField,
    history: P1HistoryLedger,
) -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}
    for layer_field in field.layers:
        keys = history.risk_keys(layer_field.layer)
        if keys.shape[0] == 0:
            keys = torch.empty((layer_field.q.shape[0], 0), dtype=torch.float32)
        if keys.shape[0] != layer_field.q.shape[0]:
            raise ODEBFContractError("P1 arm-local history risk-key geometry differs")
        result[layer_field.layer] = (
            layer_field.residual.double()
            @ (layer_field.q.double().T @ keys.double())
            if keys.shape[1]
            else torch.empty((layer_field.residual.shape[0], 0), dtype=torch.float64)
        )
    return result


def _success_payload(receipt: BatchSuccessReceipt) -> dict[str, Any]:
    return receipt.raw_free_payload()


def _replay_entry(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    arm_state: ArmRuntimeState,
    sequential_batch: int,
    waypoint: int,
    factors: Mapping[str, Sequence[WaypointFactor]],
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: StatelessReplaySchedule,
    theta0_cache: Theta0TeacherCache,
) -> ReplayEntryState:
    history_records = arm_state.history.rotating_records(
        sequential_batch=sequential_batch,
        waypoint=waypoint,
    )
    history_requests = tuple(
        request_by_sha256[item.request_sha256] for item in history_records
    )
    replay_batch = schedule.batch(
        SampleLineage.CONTROLLER,
        outer_batch_index=sequential_batch,
        correction_cycle=0,
        waypoint=waypoint,
        replay_batch_id=0,
    )
    pretrained_requests = tuple(
        population_by_sha256[item] for item in replay_batch.item_sha256
    )
    with _virtual_context(model, factors):
        history_entry, history_receipt = evaluate_target_new_nlls(
            model,
            tokenizer,
            history_requests,
            model_alias=alias,
        )
        pretrained_entry, pretrained_receipt = capture_pretrained_entry_kl(
            model,
            tokenizer,
            pretrained_requests,
            theta0_cache,
        )
    return ReplayEntryState(
        history_requests,
        history_entry,
        pretrained_requests,
        pretrained_entry,
        history_receipt.value_sha256,
        pretrained_receipt.value_sha256,
        replay_batch.schedule_digest,
    )


def _functional_trial(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    entry: ReplayEntryState,
    theta0_cache: Theta0TeacherCache,
    factors: Mapping[str, Sequence[WaypointFactor]],
    lock: P1ControllerLock,
    ledger: ComputeLedger,
) -> FunctionalReplayPair:
    pair = evaluate_functional_replay_pair(
        model,
        tokenizer,
        model_alias=alias,
        history_requests=entry.history_requests,
        history_entry_nll=entry.history_entry_nll,
        pretrained_requests=entry.pretrained_requests,
        pretrained_entry_kl=entry.pretrained_entry_kl,
        theta0_cache=theta0_cache,
        trial_factors_by_weight=factors,
        historical_budget=lock.functional_h_budget_nats,
        pretrained_budget=lock.functional_p_budget_nats,
        smoothmax_temperature=lock.smoothmax_temperature,
    )
    ledger.increment("functional_h_replay")
    ledger.increment("functional_p_replay")
    return pair


def _terminal_replay_entry(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    arm_state: ArmRuntimeState,
    sequential_batch: int,
    selected_stage: int,
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: StatelessReplaySchedule,
    theta0_cache: Theta0TeacherCache,
) -> ReplayEntryState:
    history_records = arm_state.history.snapshot().active_records
    history_requests = tuple(
        request_by_sha256[item.request_sha256] for item in history_records
    )
    replay_batch = schedule.batch(
        SampleLineage.TERMINAL_CONFIRMATION,
        outer_batch_index=sequential_batch,
        correction_cycle=0,
        waypoint=FIXED_K,
        replay_batch_id=0,
    )
    pretrained_requests = tuple(
        population_by_sha256[item] for item in replay_batch.item_sha256
    )
    history_entry, history_receipt = evaluate_target_new_nlls(
        model,
        tokenizer,
        history_requests,
        model_alias=alias,
    )
    pretrained_entry, pretrained_receipt = capture_pretrained_entry_kl(
        model,
        tokenizer,
        pretrained_requests,
        theta0_cache,
    )
    return ReplayEntryState(
        history_requests,
        history_entry,
        pretrained_requests,
        pretrained_entry,
        history_receipt.value_sha256,
        pretrained_receipt.value_sha256,
        replay_batch.schedule_digest,
    )


def _report_only_p_diagnostic(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    sequential_batch: int,
    factors: Mapping[str, Sequence[WaypointFactor]],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: StatelessReplaySchedule,
    theta0_cache: Theta0TeacherCache,
    lock: P1ControllerLock,
    ledger: ComputeLedger,
) -> dict[str, Any]:
    batch = schedule.batch(
        SampleLineage.REPORT_ONLY,
        outer_batch_index=sequential_batch,
        correction_cycle=0,
        waypoint=FIXED_K,
        replay_batch_id=0,
    )
    requests = tuple(population_by_sha256[item] for item in batch.item_sha256)
    entry_kl, entry_receipt = capture_pretrained_entry_kl(
        model, tokenizer, requests, theta0_cache
    )
    with _virtual_context(model, factors):
        observed, observed_receipt = evaluate_next_token_log_probs(
            model, tokenizer, requests
        )
    trial_kl = samplewise_teacher_kl(theta0_cache.select(batch.item_sha256), observed)
    risk = functional_replay_risk(
        barrier="pretrained-theta0-teacher",
        entry_values=entry_kl.tolist(),
        trial_values=trial_kl.tolist(),
        sample_sha256=batch.item_sha256,
        budget=lock.functional_p_budget_nats,
        smooth_max_temperature=lock.smoothmax_temperature,
    )
    ledger.increment("functional_p_replay")
    return {
        "lineage": SampleLineage.REPORT_ONLY.value,
        "schedule_sha256": batch.schedule_digest,
        "entry_value_sha256": entry_receipt.value_sha256,
        "trial_value_sha256": tensor_sha256(trial_kl),
        "observed_log_prob_sha256": observed_receipt.value_sha256,
        "risk": asdict(risk),
        "decision_input": False,
    }


def _retention_floor(
    model: torch.nn.Module,
    tokenizer: Any,
    prior_batches: Sequence[Sequence[Mapping[str, Any]]],
    *,
    alias: str,
    native_candidates: Mapping[str, torch.Tensor],
    ours_factors: Mapping[str, Sequence[WaypointFactor]],
    ledger: ComputeLedger,
) -> RetentionFloorReceipt:
    if not prior_batches:
        empty = canonical_hash([])
        return RetentionFloorReceipt(0, 0, 0, 0, empty, empty, empty, 0, True)
    native_bits: list[int] = []
    ours_bits: list[int] = []
    for batch in prior_batches:
        native = _evaluate_native_rewrite(
            model,
            tokenizer,
            batch,
            alias=alias,
            candidates=native_candidates,
            ledger=ledger,
        )
        ours = _evaluate_rewrite(
            model,
            tokenizer,
            batch,
            alias=alias,
            factors=ours_factors,
            ledger=ledger,
        )
        if native.request_order_sha256 != ours.request_order_sha256:
            raise ODEBFContractError("P1 retention Native/ours order differs")
        native_bits.extend(native.batch_success.request_success_vector)
        ours_bits.extend(ours.batch_success.request_success_vector)
    losses = [int(native == 1 and ours == 0) for native, ours in zip(native_bits, ours_bits)]
    denominator = len(native_bits)
    return RetentionFloorReceipt(
        len(prior_batches),
        sum(native_bits),
        sum(ours_bits),
        denominator,
        canonical_hash(native_bits),
        canonical_hash(ours_bits),
        canonical_hash(losses),
        sum(losses),
        sum(ours_bits) >= sum(native_bits),
    )


def _assemble_candidates(
    parameters: Mapping[str, torch.nn.Parameter],
    entry_sha256: Mapping[str, str],
    factors: Mapping[str, Sequence[WaypointFactor]],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if set(parameters) != set(entry_sha256):
        raise ODEBFContractError("P1 candidate parameter/entry inventory differs")
    candidates: dict[str, torch.Tensor] = {}
    stats_payload: dict[str, Any] = {}
    for name in sorted(parameters):
        parameter = parameters[name]
        if tensor_sha256(parameter) != entry_sha256[name]:
            raise ODEBFContractError("P1 candidate assembly entry bytes differ")
        if name not in factors or not factors[name]:
            candidate = parameter.detach().to(device="cpu").clone()
            stats_payload[name] = {
                "factor_count": 0,
                "rank_columns_total": 0,
                "maximum_fp32_block_elements": 0,
                "dense_fp32_full_delta_live": 0,
                "effective_bf16_sha256": tensor_sha256(candidate),
            }
        else:
            effective, stats = assemble_effective_bf16(
                parameter,
                factors[name],
                row_block=64,
            )
            candidate = effective.detach().to(device="cpu").clone()
            stats_payload[name] = asdict(stats)
            del effective
        candidates[name] = candidate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return candidates, stats_payload


def _prepare_history_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    arm_state: ArmRuntimeState,
    requests: Sequence[Mapping[str, Any]],
    collision_by_request: Mapping[str, str],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    terminal_event_sha256: str,
    load_increment_by_layer: Mapping[int, float],
) -> PreparedHistoryBatch:
    before = arm_state.history.snapshot().digest
    solve_keys, risk_keys, key_identity = capture_committed_history_key_views(
        model,
        tokenizer,
        requests,
        hparams,
        projector,
        contexts,
        ledger=arm_state.ledger,
    )
    expected_version = arm_state.history.version
    records = tuple(
        P1HistoryRecord(
            str(request["request_sha256"]),
            int(request["case_id"]),
            collision_by_request[str(request["request_sha256"])],
            hashlib.sha256(str(request["target_new"]).encode("utf-8")).hexdigest(),
            terminal_event_sha256,
            expected_version + 1,
        )
        for request in requests
    )
    transaction_id = canonical_hash(
        {
            "arm": arm_state.arm.value,
            "version": expected_version + 1,
            "requests": [item.request_sha256 for item in records],
            "terminal_event": terminal_event_sha256,
        }
    )
    prospective = arm_state.history.prospective(
        transaction_id=transaction_id,
        expected_version=expected_version,
        records=records,
        solve_keys_by_layer=solve_keys,
        risk_keys_by_layer=risk_keys,
    )
    if arm_state.history.snapshot().digest != before:
        raise ODEBFStateError("P1 prospective history preparation mutated state")
    return PreparedHistoryBatch(
        prospective,
        key_identity,
        before,
        tuple(
            (int(layer), float(load_increment_by_layer[layer]))
            for layer in sorted(load_increment_by_layer)
        ),
    )


def _finalize_prepared_history(
    arm_state: ArmRuntimeState,
    prepared: PreparedHistoryBatch,
) -> dict[str, Any]:
    if arm_state.history.snapshot().digest != prepared.before_state_sha256:
        raise ODEBFStateError("P1 history changed between prepare and commit")
    finalized = arm_state.history.finalize(
        prepared.prospective,
        post_commit_verified=True,
        load_increment_by_layer=dict(prepared.load_increment_by_layer),
    )
    if finalized.appended_count != BATCH_SIZE or finalized.idempotent_replay:
        raise ODEBFContractError("P1 successful transaction did not append exact B10")
    return {
        "prospective_sha256": prepared.prospective.payload_sha256,
        "finalize": asdict(finalized),
        "append_call_count": 1,
        "idempotency_cpu_gate_required": True,
        "key_identity_sha256": prepared.key_identity_sha256,
        "history_state_sha256": arm_state.history.snapshot().digest,
    }


def _make_matched_bundle(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    projector: torch.Tensor,
    capture: P1NativeCapture,
    *,
    arm_state: ArmRuntimeState,
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    lock: P1ControllerLock,
) -> FrozenMatchedBundle:
    source_counter_before = dict(arm_state.ledger.counters)
    source_started = time.perf_counter()
    layers = tuple(int(layer) for layer in hparams.layers)
    target_state = torch.stack(capture.direct_z, dim=1).to(dtype=torch.float32)
    timer = ComponentTimer(arm_state.ledger)
    with timer.measure("frozen_field_build"):
        field = build_p1_frozen_field_from_capture(
            model,
            hparams,
            projector,
            capture,
            target_state=target_state,
            accepted_waypoint=0,
            history_solve_keys_by_layer=_history_keys(arm_state.history, layers, risk=False),
            history_risk_keys_by_layer=_history_keys(arm_state.history, layers, risk=True),
            covariance_registry=covariance_registry,
            projector_sha256=projector_sha256,
            residual_tolerance=lock.residual_tolerance,
        )
    with timer.measure("signed_progress_backward"):
        signed = signed_progress_gradient(
            model,
            tokenizer,
            requests,
            field,
            cumulative_factors_by_weight={},
            ledger=arm_state.ledger,
        )
    with timer.measure("raw_velocity_qcqp"):
        source = build_p1_routing_problem(
            field,
            signed,
            accepted_by_layer={layer: () for layer in layers},
            committed_load_by_layer={layer: 0.0 for layer in layers},
            lock=lock,
        )
        raw = solve_matched_raw_velocity(source)
    arm_state.ledger.increment("qp_solve", 2)
    arm_state.ledger.increment("qp_certificate", 2)
    return FrozenMatchedBundle(
        field,
        signed,
        source,
        raw,
        target_state,
        arm_state.arm,
        {
            name: arm_state.ledger.counters[name] - source_counter_before[name]
            for name in sorted(arm_state.ledger.counters)
        },
        time.perf_counter() - source_started,
    )


def _factor_loads(
    layers: Sequence[int],
    factors: Mapping[str, Sequence[WaypointFactor]],
) -> dict[int, float]:
    result = {int(layer): 0.0 for layer in layers}
    for values in factors.values():
        for factor in values:
            left_gram = factor.left.T.double() @ factor.left.double()
            right_gram = factor.right.T.double() @ factor.right.double()
            result[factor.layer] += factor.theta**2 * float(torch.sum(left_gram * right_gram))
    return result


def _candidate_capacity(
    entry: Mapping[str, torch.Tensor],
    candidates: Mapping[str, torch.Tensor],
    *,
    row_block: int = 256,
) -> tuple[float, dict[str, float]]:
    if set(entry) != set(candidates):
        raise ODEBFContractError("P1 capacity endpoint inventories differ")
    per_weight: dict[str, float] = {}
    for name in sorted(entry):
        if entry[name].shape != candidates[name].shape:
            raise ODEBFContractError("P1 capacity endpoint shape differs")
        total = 0.0
        for start in range(0, entry[name].shape[0], row_block):
            end = min(start + row_block, entry[name].shape[0])
            difference = candidates[name][start:end].float() - entry[name][start:end].float()
            total += float(torch.sum(difference.double() * difference.double()))
            del difference
        per_weight[name] = total
    return sum(per_weight.values()), per_weight


def _routing_problem_for_stage(
    arm: P1Arm,
    bundle: FrozenMatchedBundle,
    barrier_build: P1RoutingBuild,
    arm_state: ArmRuntimeState,
) -> tuple[RoutingProblem, np.ndarray, dict[str, Any]]:
    if arm is P1Arm.F_G:
        problem = rebind_shared_problem(bundle.source_build, barrier_build)
        velocity = bundle.raw.values.copy()
        projection_payload = {
            "barrier_projector": "identity",
            "raw_velocity_sha256": bundle.raw.velocity_sha256,
            "projection_status": "NOT_APPLIED_GENERIC",
            "projection_distance": 0.0,
        }
    elif arm is P1Arm.F_BF:
        projected = project_shared_raw_velocity(
            bundle.source_build, barrier_build, bundle.raw
        )
        arm_state.ledger.increment("qp_solve", 2)
        arm_state.ledger.increment("qp_certificate", 2)
        problem = projected.problem
        velocity = (
            np.zeros_like(bundle.raw.values)
            if projected.result.status is RoutingStatus.PROGRESS_INFEASIBLE
            else np.asarray(projected.result.values, dtype=np.float64)
        )
        projection_payload = {
            "barrier_projector": "cbf",
            "raw_velocity_sha256": projected.raw_velocity_sha256,
            "projection_status": projected.result.status.value,
            "projection_distance": projected.result.projection_distance,
            "maximum_feasible_progress": projected.result.maximum_feasible_progress,
            "certificate": asdict(projected.result.certificate),
        }
    elif arm is P1Arm.R_BF:
        projected = project_matched_bf_velocity(barrier_build, bundle.raw)
        arm_state.ledger.increment("qp_solve", 2)
        arm_state.ledger.increment("qp_certificate", 2)
        problem = barrier_build.problem
        velocity = (
            np.zeros_like(bundle.raw.values)
            if projected.status is RoutingStatus.PROGRESS_INFEASIBLE
            else np.asarray(projected.values, dtype=np.float64)
        )
        projection_payload = {
            "barrier_projector": "cbf",
            "raw_velocity_sha256": projected.raw_velocity_identity,
            "projection_status": projected.status.value,
            "projection_distance": projected.projection_distance,
            "maximum_feasible_progress": projected.maximum_feasible_progress,
            "certificate": asdict(projected.certificate),
        }
    else:
        raise ODEBFContractError("P1 rollout received Native arm")
    return problem, velocity, projection_payload


def _run_nonnative_rollout(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    sequential_batch: int,
    arm_state: ArmRuntimeState,
    capture: P1NativeCapture,
    matched_bundle: FrozenMatchedBundle | None,
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    lock: P1ControllerLock,
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    collision_by_request: Mapping[str, str],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: StatelessReplaySchedule,
    theta0_cache: Theta0TeacherCache,
    dataset_path: Path,
    mutation_lock: threading.RLock,
) -> tuple[dict[str, Any], FrozenMatchedBundle | None, bool]:
    arm = arm_state.arm
    if arm not in (P1Arm.F_G, P1Arm.F_BF, P1Arm.R_BF):
        raise ODEBFContractError("P1 non-Native rollout arm differs")
    layers = tuple(int(layer) for layer in hparams.layers)
    if arm is P1Arm.F_G:
        if matched_bundle is not None:
            raise ODEBFContractError("F_G matched bundle was pre-populated")
        matched_bundle = _make_matched_bundle(
            model,
            tokenizer,
            requests,
            hparams,
            projector,
            capture,
            arm_state=arm_state,
            covariance_registry=covariance_registry,
            projector_sha256=projector_sha256,
            lock=lock,
        )
    elif arm is P1Arm.F_BF:
        if matched_bundle is None or matched_bundle.source_arm is not P1Arm.F_G:
            raise ODEBFContractError("F_BF lacks its exact F_G matched bundle")
        if matched_bundle.field.request_order_sha256 != capture.request_order_sha256:
            raise ODEBFContractError("F_G/F_BF matched request order differs")
    else:
        if matched_bundle is not None:
            raise ODEBFContractError("R_BF cannot import a frozen matched bundle")
        matched_bundle = _make_matched_bundle(
            model,
            tokenizer,
            requests,
            hparams,
            projector,
            capture,
            arm_state=arm_state,
            covariance_registry=covariance_registry,
            projector_sha256=projector_sha256,
            lock=lock,
        )
    assert matched_bundle is not None
    if capture.request_order_sha256 != ordered_request_digest_v1(
        [str(item["request_sha256"]) for item in requests]
    ):
        raise ODEBFContractError("P1 N32 counterfactual request order differs")

    rollout_counter_before = dict(arm_state.ledger.counters)
    rollout_wall_started = time.perf_counter()
    timer = ComponentTimer(arm_state.ledger)

    current_factors: dict[str, tuple[WaypointFactor, ...]] = {}
    accepted_by_layer: dict[int, list[AcceptedLayerContribution]] = {
        layer: [] for layer in layers
    }
    target_state = matched_bundle.target_state.clone()
    native_target = torch.stack(capture.direct_z, dim=1).to(dtype=torch.float32)
    target_base = capture.current_z_by_layer[layers[-1]].clone()
    active_bundle = matched_bundle
    refresh_pending = False
    selector = FixedK8P1Selector()
    snapshots: dict[int, dict[str, tuple[WaypointFactor, ...]]] = {0: {}}
    current_snapshot = _factor_state(capture.entry_sha256, current_factors)
    current_eval = _evaluate_rewrite(
        model,
        tokenizer,
        requests,
        alias=alias,
        factors=current_factors,
        ledger=arm_state.ledger,
    )
    current_feasibility = FeasibilityVerdict(True, True, True, True, True, True)
    selector.append(
        P1Waypoint(
            0,
            False,
            current_eval.batch_success,
            current_feasibility,
            current_snapshot,
            active_bundle.field.identity_sha256,
            active_bundle.raw.velocity_sha256,
            0.0,
            False,
        )
    )
    waypoint_payloads: list[dict[str, Any]] = [
        {
            "stage": 0,
            "accepted": False,
            "success": _success_payload(current_eval.batch_success),
            "snapshot_sha256": current_snapshot,
            "field_sha256": active_bundle.field.identity_sha256,
            "raw_velocity_sha256": active_bundle.raw.velocity_sha256,
        }
    ]
    for stage in range(1, FIXED_K + 1):
        if arm is P1Arm.R_BF and refresh_pending:
            refresh_counter_before = dict(arm_state.ledger.counters)
            refresh_started = time.perf_counter()
            with timer.measure("refresh_field_build"):
                field = build_p1_dynamic_field(
                    model,
                    tokenizer,
                    requests,
                    hparams,
                    projector,
                    contexts,
                    target_state=target_state,
                    accepted_waypoint=stage - 1,
                    cumulative_factors_by_weight=current_factors,
                    history_solve_keys_by_layer=_history_keys(
                        arm_state.history, layers, risk=False
                    ),
                    history_risk_keys_by_layer=_history_keys(
                        arm_state.history, layers, risk=True
                    ),
                    covariance_registry=covariance_registry,
                    projector_sha256=projector_sha256,
                    residual_tolerance=lock.residual_tolerance,
                    ledger=arm_state.ledger,
                )
            with timer.measure("refresh_signed_progress_backward"):
                signed = signed_progress_gradient(
                    model,
                    tokenizer,
                    requests,
                    field,
                    cumulative_factors_by_weight=current_factors,
                    ledger=arm_state.ledger,
                )
            with timer.measure("refresh_raw_velocity_qcqp"):
                source = build_p1_routing_problem(
                    field,
                    signed,
                    accepted_by_layer={
                        layer: tuple(accepted_by_layer[layer]) for layer in layers
                    },
                    committed_load_by_layer=arm_state.history.cumulative_load(),
                    lock=lock,
                )
                raw = solve_matched_raw_velocity(source)
            arm_state.ledger.increment("qp_solve", 2)
            arm_state.ledger.increment("qp_certificate", 2)
            active_bundle = FrozenMatchedBundle(
                field,
                signed,
                source,
                raw,
                target_state.clone(),
                P1Arm.R_BF,
                {
                    name: arm_state.ledger.counters[name]
                    - refresh_counter_before[name]
                    for name in sorted(arm_state.ledger.counters)
                },
                time.perf_counter() - refresh_started,
            )
            refresh_pending = False

        history_action_by_layer = _arm_history_actions(
            active_bundle.field, arm_state.history
        )
        with timer.measure("barrier_projection_qcqp"):
            barrier_build = build_p1_routing_problem(
                active_bundle.field,
                active_bundle.signed_progress,
                accepted_by_layer={
                    layer: tuple(accepted_by_layer[layer]) for layer in layers
                },
                committed_load_by_layer=arm_state.history.cumulative_load(),
                lock=lock,
                current_history_action_by_layer=history_action_by_layer,
            )
            problem, velocity, projection = _routing_problem_for_stage(
                arm, active_bundle, barrier_build, arm_state
            )
        target_velocity = None
        target_velocity_receipt = None
        if arm is P1Arm.R_BF and np.any(velocity > 0.0):
            with timer.measure("target_velocity_backward"):
                target_velocity, target_velocity_receipt = write_aware_target_velocity(
                    model,
                    tokenizer,
                    requests,
                    active_bundle.field,
                    velocity,
                    cumulative_factors_by_weight=current_factors,
                    target_base=target_base,
                    native_target=native_target,
                    trust_fraction=lock.target_trust_fraction,
                    ledger=arm_state.ledger,
                )
        with timer.measure("controller_margin_entry"):
            current_margin = evaluate_controller_margin(
                model,
                tokenizer,
                requests,
                cumulative_factors_by_weight=current_factors,
            )
        with timer.measure("functional_entry_replay"):
            replay_entry = _replay_entry(
                model,
                tokenizer,
                alias=alias,
                arm_state=arm_state,
                sequential_batch=sequential_batch,
                waypoint=stage,
                factors=current_factors,
                request_by_sha256=request_by_sha256,
                population_by_sha256=population_by_sha256,
                schedule=schedule,
                theta0_cache=theta0_cache,
            )
        trial_payloads: list[dict[str, Any]] = []
        accepted_choice: tuple[
            float,
            dict[str, tuple[WaypointFactor, ...]],
            dict[str, WaypointFactor],
            ModelEvaluationReceipt,
            FeasibilityVerdict,
        ] | None = None
        for beta in lock.backtracking:
            increment = scaled_waypoint_factors(
                active_bundle.field,
                velocity,
                stage=stage,
                beta=beta,
                lock=lock,
            )
            candidate_factors = _merge_factors(current_factors, increment)
            with timer.measure("candidate_margin"):
                margin = evaluate_controller_margin(
                    model,
                    tokenizer,
                    requests,
                    cumulative_factors_by_weight=candidate_factors,
                )
            actual_progress = current_margin.value - margin.value
            with timer.measure("functional_trial_replay"):
                functional = _functional_trial(
                    model,
                    tokenizer,
                    alias=alias,
                    entry=replay_entry,
                    theta0_cache=theta0_cache,
                    factors=candidate_factors,
                    lock=lock,
                    ledger=arm_state.ledger,
                )
            with timer.measure("authoritative_bf16_rewrite_verdict"):
                evaluation = _evaluate_rewrite(
                    model,
                    tokenizer,
                    requests,
                    alias=alias,
                    factors=candidate_factors,
                    ledger=arm_state.ledger,
                )
            with timer.measure("backtracking_certificate"):
                verdict = verify_backtracked_candidate(
                    problem,
                    velocity,
                    beta=beta,
                    actual_signed_progress=actual_progress,
                    functional_h_pass=functional.historical.passed,
                    functional_p_pass=functional.pretrained.passed,
                    authoritative_bf16_pass=True,
                )
            feasibility = FeasibilityVerdict(
                verdict.structural_h_pass,
                verdict.structural_p_pass,
                verdict.trust_pass,
                functional.historical.passed,
                functional.pretrained.passed,
                True,
            )
            arm_state.ledger.increment("trial")
            trial_payloads.append(
                {
                    "beta": beta,
                    "candidate_sha256": _factor_state(
                        capture.entry_sha256, candidate_factors
                    ),
                    "actual_signed_progress": actual_progress,
                    "requested_progress": problem.requested_progress,
                    "margin_entry_sha256": current_margin.value_sha256,
                    "margin_trial_sha256": margin.value_sha256,
                    "historical": asdict(functional.historical),
                    "pretrained": asdict(functional.pretrained),
                    "feasibility": asdict(feasibility),
                    "accepted": verdict.accepted,
                    "success": _success_payload(evaluation.batch_success),
                    "functional_identity_sha256": functional.functional_identity_sha256,
                }
            )
            if verdict.accepted and accepted_choice is None:
                accepted_choice = (
                    beta,
                    candidate_factors,
                    increment,
                    evaluation,
                    feasibility,
                )

        if accepted_choice is None:
            arm_state.ledger.increment("reject")
            accepted = False
            accepted_beta = 0.0
            snapshot = current_snapshot
            stage_eval = current_eval
            stage_feasibility = current_feasibility
        else:
            accepted = True
            accepted_beta, current_factors, increment, stage_eval, stage_feasibility = accepted_choice
            for layer_field in active_bundle.field.layers:
                accepted_by_layer[layer_field.layer].append(
                    AcceptedLayerContribution.from_field(
                        layer_field,
                        increment[layer_field.weight_name],
                        history_action=history_action_by_layer[layer_field.layer],
                    )
                )
            if target_velocity is not None:
                target_state = target_state + (
                    lock.eta * accepted_beta * target_velocity
                )
            current_snapshot = _factor_state(capture.entry_sha256, current_factors)
            snapshot = current_snapshot
            current_eval = stage_eval
            current_feasibility = stage_feasibility
            snapshots[stage] = {
                name: tuple(values) for name, values in current_factors.items()
            }
            refresh_pending = arm is P1Arm.R_BF
        if not accepted:
            snapshots[stage] = {
                name: tuple(values) for name, values in current_factors.items()
            }
        arm_state.ledger.record_accepted_step(
            accepted_dt=lock.eta * accepted_beta,
            completed_k_total=arm_state.ledger.completed_k_total + 1,
        )
        selector.append(
            P1Waypoint(
                stage,
                accepted,
                stage_eval.batch_success,
                stage_feasibility,
                snapshot,
                active_bundle.field.identity_sha256,
                active_bundle.raw.velocity_sha256,
                accepted_beta,
                not accepted,
            )
        )
        waypoint_payloads.append(
            {
                "stage": stage,
                "accepted": accepted,
                "accepted_beta": accepted_beta,
                "snapshot_sha256": snapshot,
                "field_sha256": active_bundle.field.identity_sha256,
                "raw_velocity_sha256": active_bundle.raw.velocity_sha256,
                "projection": projection,
                "controller_schedule_sha256": replay_entry.schedule_sha256,
                "target_velocity": (
                    None if target_velocity_receipt is None else asdict(target_velocity_receipt)
                ),
                "success": _success_payload(stage_eval.batch_success),
                "feasibility": asdict(stage_feasibility),
                "trials": trial_payloads,
                "rejection_reused_field": (not accepted),
            }
        )
    arm_state.ledger.finish_cycle(sequential_batch)
    selection = selector.finalize()
    selected_factors = snapshots[selection.selected_stage]
    selected_record = selector.records[selection.selected_stage]

    with timer.measure("terminal_functional_replay"):
        terminal_entry = _terminal_replay_entry(
            model,
            tokenizer,
            alias=alias,
            arm_state=arm_state,
            sequential_batch=sequential_batch,
            selected_stage=selection.selected_stage,
            request_by_sha256=request_by_sha256,
            population_by_sha256=population_by_sha256,
            schedule=schedule,
            theta0_cache=theta0_cache,
        )
        terminal_replay = _functional_trial(
            model,
            tokenizer,
            alias=alias,
            entry=terminal_entry,
            theta0_cache=theta0_cache,
            factors=selected_factors,
            lock=lock,
            ledger=arm_state.ledger,
        )
    terminal_feasibility = FeasibilityVerdict(
        selected_record.feasibility.structural_h,
        selected_record.feasibility.structural_p,
        selected_record.feasibility.trust,
        terminal_replay.historical.passed,
        terminal_replay.pretrained.passed,
        True,
    )
    if not terminal_feasibility.all_pass:
        raise ODEBFContractError("P1 all-history/terminal-P verifier failed")

    with timer.measure("selected_endpoint_rewrite_verdict"):
        selected_eval = _evaluate_rewrite(
            model,
            tokenizer,
            requests,
            alias=alias,
            factors=selected_factors,
            ledger=arm_state.ledger,
        )
    if selected_eval.batch_success.raw_free_payload() != selected_record.success.raw_free_payload():
        raise ODEBFContractError("P1 selected virtual efficacy event differs")
    freeze = EndpointActionFreeze(
        arm.value,
        sequential_batch,
        capture.request_order_sha256,
        selection.selected_snapshot_sha256,
        FIXED_K,
    )
    cases = load_counterfact_cases_after_freeze(dataset_path, requests, freeze)
    with timer.measure("heldout_primary_native_and_ours"):
        native_primary = _evaluate_native_primary(
            model,
            tokenizer,
            cases,
            alias=alias,
            freeze=freeze,
            candidates=capture.native_candidates,
            ledger=arm_state.ledger,
        )
        ours_primary = _evaluate_primary(
            model,
            tokenizer,
            cases,
            alias=alias,
            freeze=freeze,
            factors=selected_factors,
            ledger=arm_state.ledger,
        )
    paired_floor = pair_primary_native_floor(native_primary, ours_primary)
    with timer.measure("same_entry_n32_rewrite_verdict"):
        native_rewrite = _evaluate_native_rewrite(
            model,
            tokenizer,
            requests,
            alias=alias,
            candidates=capture.native_candidates,
            ledger=arm_state.ledger,
        )
    if (
        tuple(item[0] for item in native_primary.efficacy.per_case_bits)
        != native_rewrite.batch_success.request_success_vector
        or tuple(item[0] for item in ours_primary.efficacy.per_case_bits)
        != selected_eval.batch_success.request_success_vector
    ):
        raise ODEBFContractError("P1 official rewrite/primary efficacy parity differs")
    prior_batches = tuple(
        request_by_sha256[item.request_sha256]
        for item in arm_state.history.snapshot().active_records
    )
    grouped_prior = tuple(
        prior_batches[index : index + BATCH_SIZE]
        for index in range(0, len(prior_batches), BATCH_SIZE)
    )
    with timer.measure("previous_edit_retention"):
        retention = _retention_floor(
            model,
            tokenizer,
            grouped_prior,
            alias=alias,
            native_candidates=capture.native_candidates,
            ours_factors=selected_factors,
            ledger=arm_state.ledger,
        )
    with timer.measure("report_only_pretrained_replay"):
        report_only_p = _report_only_p_diagnostic(
            model,
            tokenizer,
            sequential_batch=sequential_batch,
            factors=selected_factors,
            population_by_sha256=population_by_sha256,
            schedule=schedule,
            theta0_cache=theta0_cache,
            lock=lock,
            ledger=arm_state.ledger,
        )
    primary_floor_pass = paired_floor.all_primary_pass and retention.passed
    hard_floor_stop = arm is P1Arm.R_BF and not primary_floor_pass

    batch_payload: dict[str, Any] = {
        "schema": "ode-edit-s04-ode-bf-p1-arm-batch/v1",
        "instruction_id": INSTRUCTION_ID,
        "alias": alias,
        "arm": arm.value,
        "sequential_batch": sequential_batch,
        "cumulative_edits": (sequential_batch + 1) * BATCH_SIZE,
        "request_order_sha256": capture.request_order_sha256,
        "joint_rank": BATCH_SIZE,
        "n32_same_entry": capture.raw_free_payload(),
        "field_policy": (
            "F_G-shared-entry-frozen"
            if arm in (P1Arm.F_G, P1Arm.F_BF)
            else "R_BF-accepted-state-refresh"
        ),
        "field_sha256": matched_bundle.field.identity_sha256,
        "raw_velocity_sha256": matched_bundle.raw.velocity_sha256,
        "shared_source_compute": {
            "actual_owner": matched_bundle.source_arm.value,
            "normalized_counter_delta": matched_bundle.source_compute_delta,
            "actual_owner_wall_seconds": matched_bundle.source_wall_seconds,
        },
        "waypoints": waypoint_payloads,
        "selection": asdict(selection),
        "terminal_replay": {
            "historical": asdict(terminal_replay.historical),
            "pretrained": asdict(terminal_replay.pretrained),
            "identity_sha256": terminal_replay.functional_identity_sha256,
        },
        "native_primary": _primary_payload(native_primary),
        "ours_primary": _primary_payload(ours_primary),
        "paired_native_floor": _paired_floor_payload(paired_floor),
        "previous_retention_floor": asdict(retention),
        "report_only_pretrained": report_only_p,
        "primary_floor_pass": primary_floor_pass,
        "pairwise_miss_review": paired_floor.pairwise_miss_review or retention.loss_count > 0,
        "hard_floor_stop": hard_floor_stop,
        "controller_sample_lineage": "controller",
        "terminal_sample_lineage": "terminal-confirmation",
        "heldout_opened_after_action_freeze": True,
        "controller_heldout_access_count": 0,
        "generation_call_count": 0,
        "rollout_compute_delta": {
            name: arm_state.ledger.counters[name] - rollout_counter_before[name]
            for name in sorted(arm_state.ledger.counters)
        },
        "rollout_wall_seconds": time.perf_counter() - rollout_wall_started,
    }
    if hard_floor_stop:
        batch_payload["status"] = "R_BF_NATIVE_PRIMARY_FLOOR_FAIL"
        return batch_payload, matched_bundle if arm is P1Arm.F_G else None, True

    with timer.measure("candidate_assembly"):
        candidates, assembly = _assemble_candidates(
            {
                name: dict(model.named_parameters())[name]
                for name in sorted(capture.entry_sha256)
            },
            capture.entry_sha256,
            selected_factors,
        )
    candidate_sha = {name: tensor_sha256(value) for name, value in candidates.items()}
    load_increment = _factor_loads(layers, selected_factors)
    capacity, capacity_by_weight = _candidate_capacity(
        capture.entry_weights, candidates
    )
    with timer.measure("history_key_prepare"):
        with _virtual_context(model, selected_factors):
            prepared_history = _prepare_history_batch(
                model,
                tokenizer,
                arm_state=arm_state,
                requests=requests,
                collision_by_request=collision_by_request,
                hparams=hparams,
                projector=projector,
                contexts=contexts,
                terminal_event_sha256=canonical_hash(
                    selected_eval.batch_success.raw_free_payload()
                ),
                load_increment_by_layer=load_increment,
            )
    touched = {
        name: dict(model.named_parameters())[name] for name in sorted(candidates)
    }
    callback: dict[str, Any] = {}
    history_box: dict[str, dict[str, Any]] = {}
    transaction = AtomicBatchTransaction(
        touched,
        transaction_id=canonical_hash(
            {"arm": arm.value, "batch": sequential_batch, "selection": asdict(selection)}
        ),
        mutation_lock=mutation_lock,
    )
    for name in sorted(candidates):
        transaction.stage(name, candidates[name])

    def verify_commit() -> bool:
        committed = evaluate_counterfact_rewrite_batch(
            model, tokenizer, requests, model_alias=alias
        )
        committed_replay = evaluate_functional_replay_pair(
            model,
            tokenizer,
            model_alias=alias,
            history_requests=terminal_entry.history_requests,
            history_entry_nll=terminal_entry.history_entry_nll,
            pretrained_requests=terminal_entry.pretrained_requests,
            pretrained_entry_kl=terminal_entry.pretrained_entry_kl,
            theta0_cache=theta0_cache,
            trial_factors_by_weight={},
            historical_budget=lock.functional_h_budget_nats,
            pretrained_budget=lock.functional_p_budget_nats,
            smoothmax_temperature=lock.smoothmax_temperature,
        )
        arm_state.ledger.increment(
            "evaluator_forward", committed.model_forward_count
        )
        arm_state.ledger.increment(
            "evaluator_tokens", committed.processed_token_count
        )
        arm_state.ledger.increment("functional_h_replay")
        arm_state.ledger.increment("functional_p_replay")
        bytes_exact = all(tensor_sha256(touched[name]) == candidate_sha[name] for name in touched)
        logits_exact = (
            committed.target_full_vocabulary_logits_sha256
            == selected_eval.target_full_vocabulary_logits_sha256
        )
        event_exact = (
            committed.batch_success.raw_free_payload()
            == selected_eval.batch_success.raw_free_payload()
        )
        callback.update(
            {
                "bytes_exact": bytes_exact,
                "logits_exact": logits_exact,
                "event_exact": event_exact,
                "historical_pass": committed_replay.historical.passed,
                "pretrained_pass": committed_replay.pretrained.passed,
                "committed_event": _success_payload(committed.batch_success),
            }
        )
        verified = all(
            (
                bytes_exact,
                logits_exact,
                event_exact,
                committed_replay.historical.passed,
                committed_replay.pretrained.passed,
            )
        )
        if not verified:
            return False
        selector.verify_post_commit(
            committed_snapshot_sha256=selection.selected_snapshot_sha256,
            success=committed.batch_success,
            feasibility=FeasibilityVerdict(
                terminal_feasibility.structural_h,
                terminal_feasibility.structural_p,
                terminal_feasibility.trust,
                committed_replay.historical.passed,
                committed_replay.pretrained.passed,
                True,
            ),
        )
        history_box["payload"] = _finalize_prepared_history(
            arm_state, prepared_history
        )
        return True

    with timer.measure("atomic_commit_and_postverify"):
        commit = transaction.commit(post_commit_verify=verify_commit)
    arm_state.ledger.increment("commit")
    if "payload" not in history_box:
        raise ODEBFStateError("P1 commit completed without atomic history finalization")
    history = history_box["payload"]
    total_load = sum(load_increment.values())
    batch_payload.update(
        {
            "status": "COMMITTED",
            "assembly": assembly,
            "commit": asdict(commit),
            "virtual_commit": callback,
            "history": history,
            "net_bf16_capacity": capacity,
            "capacity_by_weight_sha256": canonical_hash(capacity_by_weight),
            "layer_load": {str(layer): load_increment[layer] for layer in layers},
            "layer_concentration": (
                0.0 if total_load == 0.0 else max(load_increment.values()) / total_load
            ),
        }
    )
    return batch_payload, matched_bundle if arm is P1Arm.F_G else None, False


def _run_native_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    sequential_batch: int,
    arm_state: ArmRuntimeState,
    capture: P1NativeCapture,
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    collision_by_request: Mapping[str, str],
    dataset_path: Path,
    mutation_lock: threading.RLock,
) -> dict[str, Any]:
    if arm_state.arm is not P1Arm.N32_NATIVE:
        raise ODEBFContractError("P1 Native batch received another arm")
    timer = ComponentTimer(arm_state.ledger)
    snapshot_sha = canonical_hash(
        {
            "entry": capture.entry_sha256,
            "native": {
                name: tensor_sha256(value)
                for name, value in sorted(capture.native_candidates.items())
            },
        }
    )
    with timer.measure("native_rewrite_verdict"):
        rewrite = _evaluate_native_rewrite(
            model,
            tokenizer,
            requests,
            alias=alias,
            candidates=capture.native_candidates,
            ledger=arm_state.ledger,
        )
    freeze = EndpointActionFreeze(
        arm_state.arm.value,
        sequential_batch,
        capture.request_order_sha256,
        snapshot_sha,
        0,
    )
    cases = load_counterfact_cases_after_freeze(dataset_path, requests, freeze)
    with timer.measure("native_heldout_primary"):
        primary = _evaluate_native_primary(
            model,
            tokenizer,
            cases,
            alias=alias,
            freeze=freeze,
            candidates=capture.native_candidates,
            ledger=arm_state.ledger,
        )
    if tuple(item[0] for item in primary.efficacy.per_case_bits) != (
        rewrite.batch_success.request_success_vector
    ):
        raise ODEBFContractError("P1 Native official efficacy parity differs")
    prior_records = arm_state.history.snapshot().active_records
    prior_batches = tuple(
        tuple(
            request_by_sha256[item.request_sha256]
            for item in prior_records[index : index + BATCH_SIZE]
        )
        for index in range(0, len(prior_records), BATCH_SIZE)
    )
    retention_bits: list[int] = []
    with timer.measure("native_previous_edit_retention"):
        for batch in prior_batches:
            retained = _evaluate_native_rewrite(
                model,
                tokenizer,
                batch,
                alias=alias,
                candidates=capture.native_candidates,
                ledger=arm_state.ledger,
            )
            retention_bits.extend(retained.batch_success.request_success_vector)

    touched = {
        name: dict(model.named_parameters())[name]
        for name in sorted(capture.native_candidates)
    }
    capacity, per_weight = _candidate_capacity(
        capture.entry_weights, capture.native_candidates
    )
    load_increment: dict[int, float] = {}
    for layer in hparams.layers:
        name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        load_increment[int(layer)] = per_weight[name]
    with timer.measure("native_history_key_prepare"):
        with CandidateBF16FunctionalTrial(model, capture.native_candidates):
            prepared_history = _prepare_history_batch(
                model,
                tokenizer,
                arm_state=arm_state,
                requests=requests,
                collision_by_request=collision_by_request,
                hparams=hparams,
                projector=projector,
                contexts=contexts,
                terminal_event_sha256=canonical_hash(
                    rewrite.batch_success.raw_free_payload()
                ),
                load_increment_by_layer=load_increment,
            )
    callback: dict[str, Any] = {}
    history_box: dict[str, dict[str, Any]] = {}
    transaction = AtomicBatchTransaction(
        touched,
        transaction_id=canonical_hash(
            {
                "arm": arm_state.arm.value,
                "batch": sequential_batch,
                "snapshot": snapshot_sha,
            }
        ),
        mutation_lock=mutation_lock,
    )
    for name in sorted(capture.native_candidates):
        transaction.stage(name, capture.native_candidates[name])

    def verify_commit() -> bool:
        committed = evaluate_counterfact_rewrite_batch(
            model, tokenizer, requests, model_alias=alias
        )
        bytes_exact = all(
            tensor_sha256(touched[name])
            == tensor_sha256(capture.native_candidates[name])
            for name in touched
        )
        logits_exact = (
            committed.target_full_vocabulary_logits_sha256
            == rewrite.target_full_vocabulary_logits_sha256
        )
        event_exact = (
            committed.batch_success.raw_free_payload()
            == rewrite.batch_success.raw_free_payload()
        )
        arm_state.ledger.increment(
            "evaluator_forward", committed.model_forward_count
        )
        arm_state.ledger.increment(
            "evaluator_tokens", committed.processed_token_count
        )
        callback.update(
            {
                "bytes_exact": bytes_exact,
                "logits_exact": logits_exact,
                "event_exact": event_exact,
                "committed_event": _success_payload(committed.batch_success),
            }
        )
        if not (bytes_exact and logits_exact and event_exact):
            return False
        history_box["payload"] = _finalize_prepared_history(
            arm_state, prepared_history
        )
        return True

    with timer.measure("native_atomic_commit_and_postverify"):
        commit = transaction.commit(post_commit_verify=verify_commit)
    arm_state.ledger.increment("commit")
    if "payload" not in history_box:
        raise ODEBFStateError("P1 Native commit lacked atomic history finalization")
    history = history_box["payload"]
    total_load = sum(load_increment.values())
    return {
        "schema": "ode-edit-s04-ode-bf-p1-arm-batch/v1",
        "instruction_id": INSTRUCTION_ID,
        "alias": alias,
        "arm": arm_state.arm.value,
        "sequential_batch": sequential_batch,
        "cumulative_edits": (sequential_batch + 1) * BATCH_SIZE,
        "status": "COMMITTED",
        "request_order_sha256": capture.request_order_sha256,
        "joint_rank": BATCH_SIZE,
        "native_capture": capture.raw_free_payload(),
        "fixed_k_slots": 0,
        "endpoint_status": (
            "EXACT_BATCH_HIT"
            if rewrite.batch_success.joint_exact_success
            else "NO_EXACT_BATCH_HIT"
        ),
        "rewrite": _success_payload(rewrite.batch_success),
        "primary": _primary_payload(primary),
        "prior_retention": {
            "prior_batch_count": len(prior_batches),
            "correct_count": sum(retention_bits),
            "denominator": len(retention_bits),
            "bits_sha256": canonical_hash(retention_bits),
        },
        "commit": asdict(commit),
        "virtual_commit": callback,
        "history": history,
        "net_bf16_capacity": capacity,
        "capacity_by_weight_sha256": canonical_hash(per_weight),
        "layer_load": {str(layer): load_increment[layer] for layer in sorted(load_increment)},
        "layer_concentration": (
            0.0 if total_load == 0.0 else max(load_increment.values()) / total_load
        ),
        "heldout_opened_after_action_freeze": True,
        "controller_heldout_access_count": 0,
        "generation_call_count": 0,
    }


def _assert_matched_frozen_pair(
    generic: Mapping[str, Any],
    bf: Mapping[str, Any],
) -> dict[str, Any]:
    if generic["arm"] != P1Arm.F_G.value or bf["arm"] != P1Arm.F_BF.value:
        raise ODEBFContractError("P1 matched frozen arm labels differ")
    if (
        generic["request_order_sha256"] != bf["request_order_sha256"]
        or generic["field_sha256"] != bf["field_sha256"]
        or generic["raw_velocity_sha256"] != bf["raw_velocity_sha256"]
        or generic["shared_source_compute"] != bf["shared_source_compute"]
    ):
        raise ODEBFContractError("F_G/F_BF field/raw/request identity differs")
    generic_waypoints = generic["waypoints"]
    bf_waypoints = bf["waypoints"]
    if len(generic_waypoints) != FIXED_K + 1 or len(bf_waypoints) != FIXED_K + 1:
        raise ODEBFContractError("F_G/F_BF fixed K8 waypoint count differs")
    for stage in range(1, FIXED_K + 1):
        left = generic_waypoints[stage]
        right = bf_waypoints[stage]
        if (
            left["stage"] != right["stage"]
            or left["field_sha256"] != right["field_sha256"]
            or left["raw_velocity_sha256"] != right["raw_velocity_sha256"]
            or left["controller_schedule_sha256"]
            != right["controller_schedule_sha256"]
            or [item["beta"] for item in left["trials"]]
            != [item["beta"] for item in right["trials"]]
            or len(left["trials"]) != len(right["trials"])
        ):
            raise ODEBFContractError("F_G/F_BF waypoint budget/schedule differs")
        if left["projection"]["barrier_projector"] != "identity":
            raise ODEBFContractError("F_G unexpectedly applied a BF projector")
        if right["projection"]["barrier_projector"] != "cbf":
            raise ODEBFContractError("F_BF omitted its sole projector difference")
        for left_trial, right_trial in zip(left["trials"], right["trials"]):
            if (
                left_trial["historical"]["sample_order_sha256"]
                != right_trial["historical"]["sample_order_sha256"]
                or left_trial["pretrained"]["sample_order_sha256"]
                != right_trial["pretrained"]["sample_order_sha256"]
            ):
                raise ODEBFContractError("F_G/F_BF replay sample identity differs")
    required_equal = (
        "model_forward",
        "processed_tokens",
        "backward",
        "target_backward",
        "trial",
        "functional_h_replay",
        "functional_p_replay",
        "evaluator_forward",
        "evaluator_tokens",
    )
    mismatches = {
        name: (
            generic["rollout_compute_delta"][name],
            bf["rollout_compute_delta"][name],
        )
        for name in required_equal
        if generic["rollout_compute_delta"][name]
        != bf["rollout_compute_delta"][name]
    }
    if mismatches:
        raise ODEBFContractError("F_G/F_BF matched rollout accounting differs")
    return {
        "field_sha256": generic["field_sha256"],
        "raw_velocity_sha256": generic["raw_velocity_sha256"],
        "shared_source_compute": generic["shared_source_compute"],
        "fixed_k": FIXED_K,
        "backtracking_trials_per_slot": len(generic_waypoints[1]["trials"]),
        "required_equal_counters": {
            name: generic["rollout_compute_delta"][name] for name in required_equal
        },
        "projector_only_difference": True,
    }


def _trajectory_summary(
    batches_by_arm: Mapping[P1Arm, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in P1_ARM_ORDER:
        batches = tuple(batches_by_arm.get(arm, ()))
        points: list[dict[str, Any]] = []
        for batch in batches:
            primary = batch["primary"] if arm is P1Arm.N32_NATIVE else batch["ours_primary"]
            metrics = primary["metrics"]
            if arm is P1Arm.N32_NATIVE:
                retention = batch["prior_retention"]
                retention_value = (
                    1.0
                    if retention["denominator"] == 0
                    else retention["correct_count"] / retention["denominator"]
                )
                endpoint_status = batch["endpoint_status"]
                reject_count = 0
            else:
                retention = batch["previous_retention_floor"]
                retention_value = (
                    1.0
                    if retention["denominator"] == 0
                    else retention["ours_correct_count"] / retention["denominator"]
                )
                endpoint_status = batch["selection"]["status"]
                reject_count = batch["selection"]["rejected_slots"]
            points.append(
                {
                    "sequential_batch": batch["sequential_batch"],
                    "cumulative_edits": batch["cumulative_edits"],
                    "efficacy": metrics["efficacy"]["official_aggregate"],
                    "generalization": metrics["generalization"]["official_aggregate"],
                    "locality": metrics["locality-preservation"]["official_aggregate"],
                    "prior_retention": retention_value,
                    "endpoint_status": endpoint_status,
                    "reject_count": reject_count,
                    "net_bf16_capacity": batch.get("net_bf16_capacity", 0.0),
                    "layer_concentration": batch.get("layer_concentration", 0.0),
                }
            )
        curves: dict[str, Any] = {}
        for metric in ("efficacy", "generalization", "locality", "prior_retention"):
            values = [float(point[metric]) for point in points]
            slope = 0.0 if len(values) < 2 else (values[-1] - values[0]) / (len(values) - 1)
            auc = (
                0.0
                if not values
                else values[0]
                if len(values) == 1
                else sum((values[index] + values[index + 1]) / 2.0 for index in range(len(values) - 1))
            )
            curves[metric] = {"values": values, "slope_per_b10": slope, "trapezoid_auc": auc}
        result[arm.value] = {"points": points, "curves": curves}
    return result


def run_p1(
    *,
    repo_root: Path,
    alias: str,
    output_root: Path,
    source_head: str,
) -> dict[str, Any]:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("P1 alias differs")
    _source_freeze(repo_root, source_head)
    expected_parent = (repo_root / "local" / "odebf" / "results").resolve(strict=False)
    destination = output_root.resolve(strict=False)
    if destination.parent != expected_parent or destination.name != expected_p1_result_name(alias):
        raise ODEBFContractError("P1 output namespace differs")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("P1 result root is create-once")
    destination.mkdir(mode=0o700, parents=True)
    raw_root = destination / "raw"
    raw_root.mkdir(mode=0o700)
    stages = P1StageRecorder(raw_root)
    started = time.time()

    package = repo_root / "project" / "run_scripts" / "ode_bf"
    locks = package / "locks"
    artifact_guard = ODEBFArtifactGuard(
        repo_root,
        locks / "p0_artifact_lock.json",
        alias,
    )
    artifact_receipt = artifact_guard.preflight()
    stream_value = json.loads(
        (locks / "p1_seqb10_stream_seal.json").read_text(encoding="utf-8")
    )
    stream = verify_p1_stream_seal(stream_value)
    population_value = json.loads(
        (locks / "p1_p_population_seal.json").read_text(encoding="utf-8")
    )
    population = verify_p1_population_seal(population_value, stream=stream)
    numerical, numerical_sha256 = load_rooted_json(
        locks / "numerical_lock_p1.json",
        expected_schema="ode-edit-s04-ode-bf-p1-numerical-lock/v1",
    )
    controller_lock = P1ControllerLock()
    if (
        numerical.get("instruction_id") != INSTRUCTION_ID
        or numerical.get("expected_base") != EXPECTED_BASE
        or numerical.get("controller_identity_sha256") != controller_lock.identity()
        or numerical.get("stream_root_digest") != stream["root_digest"]
        or numerical.get("p_population_root_digest") != population["root_digest"]
        or numerical.get("edit_batch_size") != BATCH_SIZE
        or numerical.get("sequential_batch_count") != SEQUENTIAL_BATCHES
        or numerical.get("arms") != [arm.value for arm in P1_ARM_ORDER]
    ):
        raise ODEBFContractError("P1 numerical/seal lock differs")
    dataset = artifact_guard.base_guard.dataset
    stream_batches = load_p1_stream_batches(dataset, stream)
    population_requests = load_p1_population_requests(
        dataset, population, stream=stream
    )
    sampling_seal = load_p1_sampling_seal(
        locks / "p1_p_population_seal.json",
        stream_path=locks / "p1_seqb10_stream_seal.json",
    )
    schedule = StatelessReplaySchedule(sampling_seal)
    request_by_sha256 = {
        str(item["request_sha256"]): item
        for batch in stream_batches
        for item in batch
    }
    population_by_sha256 = {
        str(item["request_sha256"]): item for item in population_requests
    }
    collision_by_request = {
        str(item["request_sha256"]): str(item["collision_sha256"])
        for item in stream["requests"]
    }
    if set(request_by_sha256) != set(collision_by_request):
        raise ODEBFContractError("P1 stream request/collision map differs")
    stages.record(
        "post_preflight",
        {
            "source_head": source_head,
            "artifact_lock_sha256": artifact_receipt.lock_sha256,
            "numerical_lock_sha256": numerical_sha256,
            "controller_identity_sha256": controller_lock.identity(),
            "stream_root_digest": stream["root_digest"],
            "population_root_digest": population["root_digest"],
            "stream_batch_digests": stream["batch_ordered_request_digest_v1"],
            "stream_request_count": len(request_by_sha256),
            "population_count": len(population_by_sha256),
        },
    )

    seed_all(COMMON_SEED)
    torch.cuda.reset_peak_memory_stats(0)
    job_ledger = ComputeLedger()
    load_timer = ComponentTimer(job_ledger)
    with load_timer.measure("model_load"):
        model, tokenizer, hparams = load_original_bf16(artifact_guard)
    initialization_counter = ModelForwardCounter(model, job_ledger)
    mutation_lock = threading.RLock()
    try:
        with load_timer.measure("context_generation_twice"):
            contexts, context_sha256 = fresh_contexts_twice(
                model, tokenizer, seed=COMMON_SEED
            )
        _atomic_write_once(
            raw_root / "context_templates.json",
            {
                "schema": "ode-edit-s04-ode-bf-p1-raw-contexts/v1",
                "contexts": contexts,
            },
        )
        with load_timer.measure("theta0_teacher_population"):
            theta0_cache = build_theta0_teacher_cache(
                model, tokenizer, population_requests, chunk_size=BATCH_SIZE
            )
    finally:
        initialization_counter.close()
    if theta0_cache.population_sha256 != canonical_hash(
        [str(item["request_sha256"]) for item in population_requests]
    ):
        raise ODEBFContractError("P1 theta0 population identity differs")
    stages.record(
        "post_model_context_teacher",
        {
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "parameter_dtype": "torch.bfloat16",
            "context_sha256": context_sha256,
            "context_generation_count": 2,
            "theta0_population_sha256": theta0_cache.population_sha256,
            "theta0_receipt_sha256": theta0_cache.receipt_sha256,
            "theta0_forward_count": theta0_cache.model_forward_count,
            "theta0_processed_tokens": theta0_cache.processed_token_count,
        },
    )

    projector = torch.load(
        artifact_guard.projector,
        map_location="cpu",
        weights_only=True,
    )
    layers = tuple(int(layer) for layer in hparams.layers)
    if (
        not isinstance(projector, torch.Tensor)
        or projector.dtype is not torch.float32
        or projector.ndim != 3
        or projector.shape[0] != len(layers)
    ):
        raise ODEBFContractError("P1 pinned projector contract differs")
    covariance_registry = PinnedCovarianceRegistry(
        easyedit_root=artifact_guard.easyedit_root,
        covariance_spec=artifact_guard.base_guard.spec["covariance"],
    )
    touched = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": dict(
            model.named_parameters()
        )[f"{hparams.rewrite_module_tmp.format(layer)}.weight"]
        for layer in layers
    }
    if any(value.dtype is not torch.bfloat16 for value in touched.values()):
        raise ODEBFContractError("P1 touched parameter dtype differs")
    base_receipt, base_values = snapshot_touched_weights(
        P1Arm.N32_NATIVE, 0, touched
    )
    arm_states: dict[P1Arm, ArmRuntimeState] = {}
    for arm in P1_ARM_ORDER:
        receipt = ArmWeightSnapshot(
            arm,
            0,
            base_receipt.parameter_sha256,
            canonical_hash(
                {
                    "arm": arm.value,
                    "batch": 0,
                    "weights": base_receipt.parameter_sha256,
                }
            ),
        )
        arm_states[arm] = ArmRuntimeState(
            arm,
            P1HistoryLedger(layer_order=layers, maximum_records=40),
            ComputeLedger(),
            receipt,
            base_values,
        )
    w0_sha256 = dict(base_receipt.parameter_sha256)
    batch_receipt_hashes: dict[str, str] = {}
    batches_by_arm: dict[P1Arm, list[dict[str, Any]]] = {
        arm: [] for arm in P1_ARM_ORDER
    }
    matched_gates: list[dict[str, Any]] = []
    resource_gate_payload: dict[str, Any] | None = None
    terminal_status = "PASS_SHORT_SEQUENTIAL_MOTIVATION_ONLY"
    stop_reason: str | None = None
    first_batch_started = time.perf_counter()
    completed_sequential_batches = 0

    try:
        for sequential_batch, requests in enumerate(stream_batches):
            matched_bundle: FrozenMatchedBundle | None = None
            payload_by_arm: dict[P1Arm, dict[str, Any]] = {}
            rbf_floor_stop = False
            for arm in P1_ARM_ORDER:
                state = arm_states[arm]
                restore_arm_snapshot(touched, state.snapshot_values, state.snapshot_receipt)
                before_history = state.history.snapshot().digest
                counter = ModelForwardCounter(model, state.ledger)
                try:
                    with ComponentTimer(state.ledger).measure(
                        "same_entry_n32_capture"
                    ):
                        capture = capture_p1_native_entry(
                            model,
                            tokenizer,
                            requests,
                            hparams,
                            projector,
                            contexts,
                            history_keys_by_layer=_history_keys(
                                state.history, layers, risk=False
                            ),
                            mutation_lock=mutation_lock,
                            ledger=state.ledger,
                            residual_tolerance=controller_lock.residual_tolerance,
                        )
                    if capture.request_order_sha256 != stream[
                        "batch_ordered_request_digest_v1"
                    ][sequential_batch]:
                        raise ODEBFContractError("P1 capture/seal request digest differs")
                    if arm is P1Arm.N32_NATIVE:
                        payload = _run_native_batch(
                            model,
                            tokenizer,
                            requests,
                            alias=alias,
                            sequential_batch=sequential_batch,
                            arm_state=state,
                            capture=capture,
                            hparams=hparams,
                            projector=projector,
                            contexts=contexts,
                            request_by_sha256=request_by_sha256,
                            collision_by_request=collision_by_request,
                            dataset_path=dataset,
                            mutation_lock=mutation_lock,
                        )
                        hard_stop = False
                    else:
                        incoming_bundle = (
                            matched_bundle
                            if arm is P1Arm.F_BF
                            else None
                        )
                        payload, returned_bundle, hard_stop = _run_nonnative_rollout(
                            model,
                            tokenizer,
                            requests,
                            alias=alias,
                            sequential_batch=sequential_batch,
                            arm_state=state,
                            capture=capture,
                            matched_bundle=incoming_bundle,
                            hparams=hparams,
                            projector=projector,
                            contexts=contexts,
                            covariance_registry=covariance_registry,
                            projector_sha256=artifact_guard.spec["projector_sha256"],
                            lock=controller_lock,
                            request_by_sha256=request_by_sha256,
                            collision_by_request=collision_by_request,
                            population_by_sha256=population_by_sha256,
                            schedule=schedule,
                            theta0_cache=theta0_cache,
                            dataset_path=dataset,
                            mutation_lock=mutation_lock,
                        )
                        if arm is P1Arm.F_G:
                            if returned_bundle is None:
                                raise ODEBFContractError("F_G did not expose matched bundle")
                            matched_bundle = returned_bundle
                        elif returned_bundle is not None:
                            raise ODEBFContractError("non-F_G arm exported matched bundle")
                    _observed_memory(state.ledger)
                finally:
                    counter.close()
                if hard_stop:
                    if state.history.snapshot().digest != before_history:
                        raise ODEBFStateError("failed R_BF floor mutated history")
                    rbf_floor_stop = True
                else:
                    receipt, values = snapshot_touched_weights(
                        arm, sequential_batch + 1, touched
                    )
                    state.snapshot_receipt = receipt
                    state.snapshot_values = values
                _assert_arm_batch_transition(
                    state,
                    payload,
                    sequential_batch=sequential_batch,
                    history_before_sha256=before_history,
                )
                payload["compute"] = state.ledger.raw_free_payload()
                receipt_name = f"batch-{sequential_batch + 1}-{arm.value}.json"
                receipt_sha = _atomic_write_once(raw_root / receipt_name, payload)
                payload["receipt_sha256"] = receipt_sha
                batch_receipt_hashes[receipt_name] = receipt_sha
                batches_by_arm[arm].append(payload)
                if payload.get("status") == "COMMITTED":
                    state.completed_batches.append(payload)
                payload_by_arm[arm] = payload
                del capture
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            matched_gate = _assert_matched_frozen_pair(
                payload_by_arm[P1Arm.F_G], payload_by_arm[P1Arm.F_BF]
            )
            matched_gates.append(matched_gate)
            stages.record(
                f"post_batch_{sequential_batch + 1}",
                {
                    "sequential_batch": sequential_batch,
                    "receipts": {
                        arm.value: payload_by_arm[arm]["receipt_sha256"]
                        for arm in P1_ARM_ORDER
                    },
                    "matched_frozen_gate": matched_gate,
                    "r_bf_primary_floor_pass": payload_by_arm[P1Arm.R_BF][
                        "primary_floor_pass"
                    ],
                    "r_bf_hard_floor_stop": rbf_floor_stop,
                },
            )
            completed_sequential_batches = sequential_batch + 1
            if sequential_batch == 0:
                elapsed_first = time.perf_counter() - first_batch_started
                peak_reserved = max(
                    state.ledger.peak_reserved_bytes for state in arm_states.values()
                )
                maxrss = max(
                    state.ledger.host_maxrss_kib for state in arm_states.values()
                )
                resource_gate = evaluate_p1a_resource_gate(
                    peak_reserved_bytes=peak_reserved,
                    process_maxrss_kib=maxrss,
                    elapsed_first_batch_all_arms_seconds=elapsed_first,
                )
                resource_gate_payload = asdict(resource_gate)
                stages.record("post_p1a_resource_gate", resource_gate_payload)
                if rbf_floor_stop:
                    terminal_status = "R_BF_NATIVE_PRIMARY_FLOOR_FAIL"
                    stop_reason = "R_BF fell below same-entry N32 primary aggregate"
                    break
                if not resource_gate.passed:
                    terminal_status = "P1A_RESOURCE_HOLD"
                    stop_reason = "outcome-blind first-B10 resource forecast failed"
                    break
            if rbf_floor_stop:
                terminal_status = "R_BF_NATIVE_PRIMARY_FLOOR_FAIL"
                stop_reason = "R_BF fell below same-entry N32 primary aggregate"
                break
            del matched_bundle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        restore_arm_snapshot(
            touched,
            base_values,
            ArmWeightSnapshot(
                P1Arm.N32_NATIVE,
                0,
                base_receipt.parameter_sha256,
                base_receipt.state_sha256,
            ),
        )
    final_w0_restored = all(
        tensor_sha256(touched[name]) == w0_sha256[name] for name in touched
    )
    if not final_w0_restored:
        raise ODEBFStateError("P1 final W0 restore differs")
    artifact_guard.assert_unchanged()
    _observed_memory(job_ledger)
    trajectories = _trajectory_summary(batches_by_arm)
    elapsed = time.time() - started
    terminal = {
        "schema": "ode-edit-s04-ode-bf-p1-terminal/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": terminal_status,
        "claim_scope": "SHORT_SEQUENTIAL_MOTIVATION_ONLY",
        "alias": alias,
        "source_head": source_head,
        "edit_batch_size": BATCH_SIZE,
        "sequential_batch_count_locked": SEQUENTIAL_BATCHES,
        "sequential_batch_count_completed": completed_sequential_batches,
        "logical_edits_completed": completed_sequential_batches * BATCH_SIZE,
        "arms": [arm.value for arm in P1_ARM_ORDER],
        "stop_reason": stop_reason,
        "strict_technical_gate": terminal_status != "TECHNICAL_FAIL",
        "claim_promotion_authorized": False,
        "broad_lifelong_claim_authorized": False,
        "resource_gate": resource_gate_payload,
        "matched_frozen_gates": matched_gates,
        "trajectories": trajectories,
        "batch_receipts": batch_receipt_hashes,
        "controller_identity_sha256": controller_lock.identity(),
        "numerical_lock_sha256": numerical_sha256,
        "stream_root_digest": stream["root_digest"],
        "p_population_root_digest": population["root_digest"],
        "artifact_receipt": asdict(artifact_receipt),
        "theta0_teacher_receipt_sha256": theta0_cache.receipt_sha256,
        "context_sha256": context_sha256,
        "job_initialization_compute": job_ledger.raw_free_payload(),
        "arm_compute": {
            arm.value: arm_states[arm].ledger.raw_free_payload()
            for arm in P1_ARM_ORDER
        },
        "final_w0_restored": final_w0_restored,
        "persistent_arm_endpoint_count": sum(
            len(state.completed_batches) for state in arm_states.values()
        ),
        "history_record_count": {
            arm.value: len(arm_states[arm].history.snapshot().active_records)
            for arm in P1_ARM_ORDER
        },
        "scientific_outcome_count": sum(
            1 for values in batches_by_arm.values() for _ in values
        ),
        "heldout_controller_access_count": 0,
        "generation_call_count": 0,
        "retry_count": 0,
        "elapsed_seconds": elapsed,
    }
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    summary = {
        "schema": "ode-edit-s04-ode-bf-p1-summary/v1",
        "status": terminal_status,
        "alias": alias,
        "terminal_sha256": terminal_sha,
        "completed_sequential_batches": completed_sequential_batches,
        "logical_edits_completed": completed_sequential_batches * BATCH_SIZE,
        "r_bf_native_floor_pass_by_batch": [
            bool(item.get("primary_floor_pass"))
            for item in batches_by_arm[P1Arm.R_BF]
        ],
        "resource_gate": resource_gate_payload,
        "final_w0_restored": final_w0_restored,
        "claim_promotion_authorized": False,
        "elapsed_seconds": elapsed,
    }
    summary_sha = _atomic_write_once(destination / "summary.json", summary)
    manifest = {
        "schema": "ode-edit-s04-ode-bf-p1-manifest/v1",
        "status": terminal_status,
        "alias": alias,
        "source_head": source_head,
        "terminal_sha256": terminal_sha,
        "summary_sha256": summary_sha,
        "batch_receipts": batch_receipt_hashes,
        "retry_count": 0,
    }
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    return {
        **summary,
        "summary_sha256": summary_sha,
        "manifest_sha256": manifest_sha,
    }


def _sanitized_failure(
    exc: BaseException,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    for frame in traceback.extract_tb(exc.__traceback__):
        try:
            relative = Path(frame.filename).resolve().relative_to(repo_root).as_posix()
        except (ValueError, OSError):
            continue
        if relative.startswith("project/run_scripts/ode_bf/") or relative.startswith(
            "project/run_scripts/session04_ode_bf_"
        ):
            frames.append(
                {"file": relative, "function": frame.name, "line": frame.lineno}
            )
    return {
        "exception_class": type(exc).__name__,
        "exception_message_sha256": hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
        "allowlisted_frames": frames,
    }


def write_p1_failure_once(
    output_root: Path,
    exc: BaseException,
    *,
    repo_root: Path,
) -> tuple[str, dict[str, Any]]:
    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    raw = output_root / "raw"
    raw.mkdir(mode=0o700, exist_ok=True)
    stage_files = sorted(raw.glob("stage-*.json"))
    last_stage = None
    if stage_files:
        last_stage = json.loads(stage_files[-1].read_text(encoding="utf-8"))["stage"]
    failure = {
        "schema": "ode-edit-s04-ode-bf-p1-failure/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "FAIL_CLOSED_NO_RETRY",
        "last_completed_stage": last_stage,
        **_sanitized_failure(exc, repo_root=repo_root),
    }
    digest = _atomic_write_once(output_root / "failure.json", failure)
    return digest, failure
