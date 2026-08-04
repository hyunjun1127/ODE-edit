"""Outcome-free B10 memory forecast and node-local Slurm accounting."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import BATCH_SIZE, FIXED_K, MODEL_ALIASES, ODEBFContractError, canonical_hash


GPU_MEMORY_REQUEST_MIB = 65_000
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
