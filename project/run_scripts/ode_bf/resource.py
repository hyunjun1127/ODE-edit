"""Outcome-free B10 memory forecast and node-local Slurm accounting."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import BATCH_SIZE, FIXED_K, MODEL_ALIASES, ODEBFContractError, canonical_hash
from .p1_adaptive import N_TRIAL_CAP


GPU_MEMORY_REQUEST_MIB = 60_416
MEMORY_CAP_MIB_PER_GPU = 66_017
GPU_CAP_SERVER2 = 3
CANONICAL_NODE = "server2"
PROJECT_JOB_PREFIXES = ("odebf_", "odealloc_", "odeedit_")

# AlphaEdit down-projection geometries are fixed by the pinned hparams/model
# revisions.  They are deliberately explicit so a silent model-shape change
# fails before allocation rather than changing the forecast post outcome.
MODEL_GEOMETRY = {
    "llama3-8b-inst": (4_096, 14_336, 5),
    "qwen2.5-7b-inst": (3_584, 18_944, 5),
}

MODEL_VOCABULARY_SIZE = {
    "llama3-8b-inst": 128_256,
    "qwen2.5-7b-inst": 152_064,
}


def _mib(byte_count: int) -> int:
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise ODEBFContractError("memory byte count is invalid")
    return math.ceil(byte_count / (1024 * 1024))


def _snapshot_upper_bound_bytes(base_lock: Mapping[str, Any], alias: str) -> int:
    files = base_lock["models"][alias]["snapshot_files"]
    sizes = [int(value[1]) for value in files.values()]
    if not sizes or any(size <= 0 for size in sizes):
        raise ODEBFContractError("pinned snapshot size inventory is invalid")
    return sum(sizes)


@dataclass(frozen=True, slots=True)
class P0MemoryForecast:
    alias: str
    edit_batch_size: int
    layer_count: int
    factor_rank_per_waypoint: int
    k_total: int
    snapshot_upper_mib: int
    all_projectors_host_mib: int
    native_zero_history_cache_host_mib: int
    max_dense_solver_gpu_mib: int
    native_entry_clone_gpu_mib: int
    cumulative_rank10_factors_mib: int
    host_entry_candidate_journal_mib: int
    activation_context_replay_reserve_mib: int
    gpu_runtime_safety_reserve_mib: int
    host_runtime_safety_reserve_mib: int
    forecast_gpu_peak_mib: int
    forecast_host_peak_mib: int
    gpu_limit_mib: int
    host_limit_mib: int
    one_live_effective_bf16_weight: bool
    dense_full_history_matrix: bool

    def __post_init__(self) -> None:
        if self.alias not in MODEL_ALIASES:
            raise ODEBFContractError("memory forecast alias is not locked")
        if self.edit_batch_size != BATCH_SIZE or self.factor_rank_per_waypoint != BATCH_SIZE:
            raise ODEBFContractError("memory forecast is not a genuine B10 trajectory")
        if self.k_total != FIXED_K or self.layer_count != 5:
            raise ODEBFContractError("memory forecast rollout/layer contract differs")
        if not self.one_live_effective_bf16_weight or self.dense_full_history_matrix:
            raise ODEBFContractError("memory forecast permits a forbidden materialization")
        if self.forecast_gpu_peak_mib > self.gpu_limit_mib:
            raise ODEBFContractError("locked B10 path exceeds one-GPU memory request")
        if self.forecast_host_peak_mib > self.host_limit_mib:
            raise ODEBFContractError("locked B10 path exceeds per-job host memory request")

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)

    def identity(self) -> str:
        return canonical_hash(self.raw_free_payload())


def forecast_p0_b10_memory(
    artifact_lock_path: Path,
    base_model_lock_path: Path,
    alias: str,
) -> P0MemoryForecast:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("memory forecast alias is not locked")
    artifact = json.loads(artifact_lock_path.read_text(encoding="utf-8"))
    base = json.loads(base_model_lock_path.read_text(encoding="utf-8"))
    out_features, in_features, layer_count = MODEL_GEOMETRY[alias]
    if artifact["models"][alias]["layers"] != [4, 5, 6, 7, 8]:
        raise ODEBFContractError("memory forecast AlphaEdit layers differ")

    snapshot = _snapshot_upper_bound_bytes(base, alias)
    projector = int(artifact["models"][alias]["projector_size"])
    dense_elements = in_features * in_features
    touched_elements = out_features * in_features

    # At the Native identity solve, the pinned FP32 projector, key outer
    # product, Alpha system, solve output/workspace, and expression temporaries
    # can overlap.  Six full dxd FP32 matrices is a deliberately conservative
    # upper bound for that source-order phase; no such matrix is retained.
    dense_solver_gpu = 6 * dense_elements * 4
    cumulative_factors = (
        layer_count
        * FIXED_K
        * BATCH_SIZE
        * (in_features + out_features)
        * 8
    )
    # Entry + Native + WB + staged candidate + rollback journal, all BF16 and
    # host-resident across every touched layer at the worst transaction point.
    host_candidates = 5 * layer_count * touched_elements * 2

    snapshot_mib = _mib(snapshot)
    projector_mib = _mib(projector)
    native_history_mib = _mib(layer_count * dense_elements * 4)
    dense_mib = _mib(dense_solver_gpu)
    factor_mib = _mib(cumulative_factors)
    candidates_mib = _mib(host_candidates)

    # Common, outcome-free reserves cover two fresh context generations,
    # direct-z autograd/KV state, authoritative replay logits, CUDA libraries,
    # allocator fragmentation, and CPU tokenizer/runtime state.  They are the
    # same for both aliases and intentionally broad rather than fitted.
    activation_replay_reserve_mib = 18_000
    gpu_runtime_safety_reserve_mib = 7_000
    host_runtime_safety_reserve_mib = 12_000
    max_effective_weight_mib = _mib(touched_elements * 2)
    native_entry_clone_mib = _mib(layer_count * touched_elements * 2)

    gpu_peak = (
        snapshot_mib
        + dense_mib
        + factor_mib
        + max_effective_weight_mib
        + native_entry_clone_mib
        + activation_replay_reserve_mib
        + gpu_runtime_safety_reserve_mib
    )
    host_peak = (
        snapshot_mib
        + projector_mib
        + native_history_mib
        + candidates_mib
        + factor_mib
        + host_runtime_safety_reserve_mib
    )
    return P0MemoryForecast(
        alias=alias,
        edit_batch_size=BATCH_SIZE,
        layer_count=layer_count,
        factor_rank_per_waypoint=BATCH_SIZE,
        k_total=FIXED_K,
        snapshot_upper_mib=snapshot_mib,
        all_projectors_host_mib=projector_mib,
        native_zero_history_cache_host_mib=native_history_mib,
        max_dense_solver_gpu_mib=dense_mib,
        native_entry_clone_gpu_mib=native_entry_clone_mib,
        cumulative_rank10_factors_mib=factor_mib,
        host_entry_candidate_journal_mib=candidates_mib,
        activation_context_replay_reserve_mib=activation_replay_reserve_mib,
        gpu_runtime_safety_reserve_mib=gpu_runtime_safety_reserve_mib,
        host_runtime_safety_reserve_mib=host_runtime_safety_reserve_mib,
        forecast_gpu_peak_mib=gpu_peak,
        forecast_host_peak_mib=host_peak,
        gpu_limit_mib=GPU_MEMORY_REQUEST_MIB,
        host_limit_mib=GPU_MEMORY_REQUEST_MIB,
        one_live_effective_bf16_weight=True,
        dense_full_history_matrix=False,
    )


@dataclass(frozen=True, slots=True)
class P1MemoryForecast:
    alias: str
    edit_batch_size: int
    sequential_batch_count: int
    arm_count: int
    base_p0_gpu_peak_mib: int
    base_p0_host_peak_mib: int
    persistent_arm_snapshots_host_mib: int
    theta0_teacher_cache_host_mib: int
    all_arm_history_keys_host_mib: int
    evaluator_state_host_reserve_mib: int
    refreshed_field_overlap_gpu_mib: int
    replay_logits_gpu_mib: int
    forecast_gpu_peak_mib: int
    forecast_host_peak_mib: int
    allocation_limit_mib: int
    runtime_reserved_hold_limit_mib: int
    runtime_rss_hold_limit_mib: int
    one_live_effective_bf16_weight: bool
    dense_fp64_full_delta: bool

    def __post_init__(self) -> None:
        if self.alias not in MODEL_ALIASES:
            raise ODEBFContractError("P1 memory forecast alias differs")
        if (
            self.edit_batch_size != BATCH_SIZE
            or self.sequential_batch_count != 4
            or self.arm_count != 4
        ):
            raise ODEBFContractError("P1 memory forecast panel geometry differs")
        if not self.one_live_effective_bf16_weight or self.dense_fp64_full_delta:
            raise ODEBFContractError("P1 memory forecast permits forbidden materialization")
        if (
            self.forecast_gpu_peak_mib > self.allocation_limit_mib
            or self.forecast_host_peak_mib > self.allocation_limit_mib
        ):
            raise ODEBFContractError("P1 forecast exceeds the one-job allocation")

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)

    def identity(self) -> str:
        return canonical_hash(self.raw_free_payload())


def forecast_p1_b10_memory(
    artifact_lock_path: Path,
    base_model_lock_path: Path,
    alias: str,
) -> P1MemoryForecast:
    """Outcome-free upper bound for four arms and four retained B10 states."""

    base = forecast_p0_b10_memory(artifact_lock_path, base_model_lock_path, alias)
    out_features, in_features, layer_count = MODEL_GEOMETRY[alias]
    vocabulary = MODEL_VOCABULARY_SIZE[alias]
    touched_bf16_bytes = layer_count * out_features * in_features * 2
    arm_snapshots = 4 * touched_bf16_bytes
    theta0_cache = 160 * vocabulary * 4
    # Four arm-local ledgers retain solve and risk key views for at most 40
    # requests/layer. Keys are FP32; no dense historical covariance is stored.
    history_keys = 4 * 2 * layer_count * in_features * 40 * 4
    # A refreshed field may briefly overlap its predecessor, but consists only
    # of rank-10 FP32 arms and certificates. Full effective weights stay one-at-a-time.
    refreshed_overlap = 2 * layer_count * BATCH_SIZE * (in_features + out_features) * 4
    replay_logits = BATCH_SIZE * vocabulary * 4
    evaluator_state_reserve_mib = 1_024
    gpu_peak = (
        base.forecast_gpu_peak_mib
        + _mib(refreshed_overlap)
        + _mib(replay_logits)
    )
    host_peak = (
        base.forecast_host_peak_mib
        + _mib(arm_snapshots)
        + _mib(theta0_cache)
        + _mib(history_keys)
        + evaluator_state_reserve_mib
    )
    return P1MemoryForecast(
        alias,
        BATCH_SIZE,
        4,
        4,
        base.forecast_gpu_peak_mib,
        base.forecast_host_peak_mib,
        _mib(arm_snapshots),
        _mib(theta0_cache),
        _mib(history_keys),
        evaluator_state_reserve_mib,
        _mib(refreshed_overlap),
        _mib(replay_logits),
        gpu_peak,
        host_peak,
        GPU_MEMORY_REQUEST_MIB,
        52_000,
        48_000,
        True,
        False,
    )


@dataclass(frozen=True, slots=True)
class AdaptiveP1MemoryForecast:
    alias: str
    edit_batch_size: int
    variant_count: int
    maximum_unique_accepted_snapshots: int
    maximum_factors_per_endpoint_layer: int
    maximum_retained_factors_per_layer: int
    base_p1_gpu_peak_mib: int
    base_p1_host_peak_mib: int
    cumulative_rank10_factor_host_mib: int
    accepted_snapshot_metadata_host_mib: int
    postfreeze_one_state_at_a_time: bool
    forecast_gpu_peak_mib: int
    forecast_host_peak_mib: int
    allocation_limit_mib: int
    one_live_effective_bf16_weight: bool
    dense_fp64_full_delta: bool

    def __post_init__(self) -> None:
        if self.alias not in MODEL_ALIASES:
            raise ODEBFContractError("adaptive P1 memory alias differs")
        if (
            self.edit_batch_size != BATCH_SIZE
            or self.variant_count != 4
            or self.maximum_unique_accepted_snapshots != 136
            or self.maximum_factors_per_endpoint_layer != 64
            or self.maximum_retained_factors_per_layer != 136
        ):
            raise ODEBFContractError("adaptive P1 memory geometry differs")
        if (
            not self.postfreeze_one_state_at_a_time
            or not self.one_live_effective_bf16_weight
            or self.dense_fp64_full_delta
        ):
            raise ODEBFContractError("adaptive P1 memory materialization differs")
        if (
            self.forecast_gpu_peak_mib > self.allocation_limit_mib
            or self.forecast_host_peak_mib > self.allocation_limit_mib
        ):
            raise ODEBFContractError("adaptive P1 forecast exceeds allocation")

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)

    def identity(self) -> str:
        return canonical_hash(self.raw_free_payload())


def forecast_p1_adaptive_b10_memory(
    artifact_lock_path: Path,
    base_model_lock_path: Path,
    alias: str,
) -> AdaptiveP1MemoryForecast:
    """Worst-case one-at-a-time memory bound for the four adaptive variants."""

    base = forecast_p1_b10_memory(
        artifact_lock_path, base_model_lock_path, alias
    )
    out_features, in_features, layer_count = MODEL_GEOMETRY[alias]
    maximum_factors_per_endpoint = 64
    maximum_retained_factors = 8 + 32 + 32 + 64
    factor_bytes = (
        layer_count
        * maximum_retained_factors
        * BATCH_SIZE
        * (in_features + out_features)
        * 4
    )
    # Snapshots retain immutable factor tuple references and hashes, not dense
    # weights or copied factor tensors.  This broad reserve covers Python/JSON
    # metadata for all 8+32+32+64 accepted states.
    snapshot_metadata_mib = 256
    host_peak = (
        base.forecast_host_peak_mib
        + _mib(factor_bytes)
        + snapshot_metadata_mib
    )
    gpu_peak = base.forecast_gpu_peak_mib
    return AdaptiveP1MemoryForecast(
        alias,
        BATCH_SIZE,
        4,
        136,
        maximum_factors_per_endpoint,
        maximum_retained_factors,
        base.forecast_gpu_peak_mib,
        base.forecast_host_peak_mib,
        _mib(factor_bytes),
        snapshot_metadata_mib,
        True,
        gpu_peak,
        host_peak,
        GPU_MEMORY_REQUEST_MIB,
        True,
        False,
    )


@dataclass(frozen=True, slots=True)
class AdaptiveP1TimeForecast:
    """Outcome-blind worst-case wall forecast for one model lane."""

    maximum_trial_count: int
    maximum_field_build_count: int
    maximum_terminal_replay_count: int
    maximum_postfreeze_state_count: int
    seconds_per_trial_bound: int
    seconds_per_field_build_bound: int
    seconds_per_terminal_replay_bound: int
    seconds_per_postfreeze_state_bound: int
    fixed_initialization_and_cleanup_seconds: int
    pre_reserve_seconds: int
    reserve_fraction: float
    forecast_seconds: int
    allocation_seconds: int
    immutable_runtime_calibration_max_seconds: int
    outcome_metric_used: bool

    def __post_init__(self) -> None:
        if (
            self.maximum_trial_count != 24 + 3 * N_TRIAL_CAP
            or self.maximum_field_build_count != 8 + 32 + 32 + 64
            or self.maximum_terminal_replay_count != 136
            or self.maximum_postfreeze_state_count != 2 + 136
            or self.reserve_fraction != 0.10
            or self.immutable_runtime_calibration_max_seconds != 2931
            or self.outcome_metric_used
        ):
            raise ODEBFContractError("adaptive P1 time geometry/provenance differs")
        if self.forecast_seconds != math.ceil(
            self.pre_reserve_seconds * (1.0 + self.reserve_fraction)
        ):
            raise ODEBFContractError("adaptive P1 time reserve arithmetic differs")
        if self.forecast_seconds > self.allocation_seconds:
            raise ODEBFContractError("adaptive P1 forecast exceeds 24-hour allocation")

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)

    def identity(self) -> str:
        return canonical_hash(self.raw_free_payload())


def forecast_p1_adaptive_b10_time() -> AdaptiveP1TimeForecast:
    """Conservative structural bound, calibrated only by immutable job timing.

    Jobs 16681/16682 establish a raw-free lane wall maximum of 2931 seconds.
    Their scientific outcomes are not inputs.  Each trial/field/evaluation bound
    below is at least twice the largest corresponding immutable wall component,
    then an additional ten-percent whole-lane reserve is applied.
    """

    maximum_trials = 24 + 3 * N_TRIAL_CAP
    maximum_fields = 8 + 32 + 32 + 64
    maximum_terminal = 136
    maximum_postfreeze = 2 + 136
    seconds_per_trial = 60
    seconds_per_field = 60
    seconds_per_terminal = 10
    seconds_per_postfreeze = 60
    fixed_seconds = 1800
    pre_reserve = (
        maximum_trials * seconds_per_trial
        + maximum_fields * seconds_per_field
        + maximum_terminal * seconds_per_terminal
        + maximum_postfreeze * seconds_per_postfreeze
        + fixed_seconds
    )
    reserve = 0.10
    forecast = math.ceil(pre_reserve * (1.0 + reserve))
    return AdaptiveP1TimeForecast(
        maximum_trials,
        maximum_fields,
        maximum_terminal,
        maximum_postfreeze,
        seconds_per_trial,
        seconds_per_field,
        seconds_per_terminal,
        seconds_per_postfreeze,
        fixed_seconds,
        pre_reserve,
        reserve,
        forecast,
        24 * 60 * 60,
        2931,
        False,
    )


def gpu_count_from_tres(tres: str) -> int | None:
    matches = re.findall(r"gpu(?::[^,=():]+)?:([0-9]+)", tres)
    if matches:
        return sum(int(value) for value in matches)
    matches = re.findall(r"gpu[^,=]*=([0-9]+)", tres)
    if matches:
        return sum(int(value) for value in matches)
    return None


@dataclass(frozen=True, slots=True)
class SchedulerJob:
    job_id: str
    job_name: str
    state: str
    gpu_count: int
    allocated_nodes: tuple[str, ...]
    requested_nodes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.job_id.isdigit() or not self.job_name:
            raise ODEBFContractError("scheduler job identity is invalid")
        if self.state not in ("RUNNING", "PENDING") or self.gpu_count <= 0:
            raise ODEBFContractError("scheduler job state/resource is invalid")

    @property
    def pending_unconstrained(self) -> bool:
        return self.state == "PENDING" and not self.requested_nodes


def node_local_gpu_totals(
    records: Sequence[SchedulerJob],
    *,
    node: str = CANONICAL_NODE,
) -> tuple[int, int]:
    project = [
        record
        for record in records
        if record.job_name.startswith(PROJECT_JOB_PREFIXES)
    ]
    cluster = sum(record.gpu_count for record in project)
    local = sum(
        record.gpu_count
        for record in project
        if node in record.allocated_nodes
        or node in record.requested_nodes
        or record.pending_unconstrained
    )
    return local, cluster


def assert_pair_capacity(
    records: Sequence[SchedulerJob],
    *,
    new_gpu_count: int = 2,
) -> tuple[int, int]:
    if new_gpu_count != 2:
        raise ODEBFContractError("technical P0 must remain an exact model pair")
    local, cluster = assert_node_local_capacity(records, new_gpu_count=new_gpu_count)
    if 2 * GPU_MEMORY_REQUEST_MIB > 2 * MEMORY_CAP_MIB_PER_GPU:
        raise ODEBFContractError("pair host-memory request exceeds the server2 cap")
    return local, cluster


def assert_node_local_capacity(
    records: Sequence[SchedulerJob],
    *,
    new_gpu_count: int,
) -> tuple[int, int]:
    if (
        isinstance(new_gpu_count, bool)
        or not isinstance(new_gpu_count, int)
        or new_gpu_count <= 0
    ):
        raise ODEBFContractError("new node-local GPU request is invalid")
    local, cluster = node_local_gpu_totals(records)
    if local + new_gpu_count > GPU_CAP_SERVER2:
        raise ODEBFContractError("server2 node-local ODE project GPU cap would be exceeded")
    return local, cluster
