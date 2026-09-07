"""Pure CPU resource admission and transparent historical-cost estimates.

This module never queries or changes a scheduler. A fresh caller-supplied
scheduler snapshot and a separately user-authorized hour cap are mandatory.
Historical elapsed-time extrapolation is an estimate, never spending authority.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

PUBLICATION = "0d0a0131e4a6a2a645dfa6530377d420a084d136"
RUNTIME = "77358b1546d1baf83b3e251afcce663b08d7bfd7"
PACKAGE = "experiment-reports/servers/server2/alpha-native-response-v31-sequential-routing-2026-09-06-v1/main-four-terminal-v1"
ALLOCATED_STATES = frozenset({"RUNNING", "COMPLETING", "CONFIGURING"})
TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"})


class ResourceBoundary(ValueError):
    """Malformed or unresolved resource evidence; no permissive defaults."""


def _positive(value: float, name: str, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResourceBoundary(name)
    number = float(value)
    if not math.isfinite(number) or number < 0 or (not zero and number == 0):
        raise ResourceBoundary(name)
    return number


def _count(value: int, name: str, *, zero: bool = False) -> int:
    _positive(value, name, zero=zero)
    if not isinstance(value, int):
        raise ResourceBoundary(name)
    return value


@dataclass(frozen=True)
class ResourcePolicy:
    project_gpu_cap: int = 2
    mem_mib_per_gpu: int = 182272
    gpus_per_process: int = 1
    authority_id: str = "ODEEDIT-SH1-DIAG-SWEEP-20260907-R1"

    def __post_init__(self):
        if not 1 <= _count(self.project_gpu_cap, "GPU_CAP") <= 2:
            raise ResourceBoundary("CAP_EXCEEDS_USER_CEILING")
        if not 1 <= _count(self.mem_mib_per_gpu, "MEMORY") <= 182272:
            raise ResourceBoundary("MEMORY_EXCEEDS_TASK_CEILING")
        if self.gpus_per_process != 1 or not self.authority_id:
            raise ResourceBoundary("ONE_GPU_PROCESS_OR_AUTHORITY")


@dataclass(frozen=True)
class BudgetAuthority:
    gpu_hours: float | None = None
    user_authority_id: str | None = None
    authority_sha256: str | None = None

    def __post_init__(self):
        if self.gpu_hours is None:
            if self.user_authority_id is not None or self.authority_sha256 is not None:
                raise ResourceBoundary("UNASSIGNED_BUDGET_WITH_AUTHORITY")
            return
        _positive(self.gpu_hours, "GPU_HOUR_CAP")
        if not self.user_authority_id or not self.authority_sha256 or len(self.authority_sha256) != 64:
            raise ResourceBoundary("USER_BUDGET_AUTHORITY_REQUIRED")
        try:
            bytes.fromhex(self.authority_sha256)
        except ValueError as exc:
            raise ResourceBoundary("BUDGET_AUTHORITY_SHA256") from exc


@dataclass(frozen=True)
class Allocation:
    allocation_id: str
    state: str
    gpus: int | None
    process_exit_confirmed: bool = False
    grandfathered: bool = False

    def occupied_gpus(self) -> int:
        if not self.allocation_id:
            raise ResourceBoundary("ALLOCATION_ID")
        if self.state in ALLOCATED_STATES or (self.state in TERMINAL_STATES and not self.process_exit_confirmed):
            if self.gpus is None:
                raise ResourceBoundary("UNCLEARED_ALLOCATION_GPU_COUNT_UNRESOLVED")
            return _count(self.gpus, "ALLOCATION_GPUS", zero=True)
        if self.state == "PENDING" or (self.state in TERMINAL_STATES and self.process_exit_confirmed):
            return 0
        raise ResourceBoundary("ALLOCATION_STATE_UNRESOLVED")


@dataclass(frozen=True)
class SchedulerSnapshot:
    query_ok: bool
    allocation_query_complete: bool
    captured_at_seconds: float
    evidence_sha256: str
    allocations: tuple[Allocation, ...] = ()


@dataclass(frozen=True)
class UsageInterval:
    """Incremental charged allocation interval, including failed attempts."""
    allocation_id: str
    attempt_id: str
    track: str
    category: str
    start_seconds: float
    end_seconds: float
    gpus: int
    evidence_sha256: str

    @property
    def gpu_seconds(self) -> float:
        start = _positive(self.start_seconds, "USAGE_START", zero=True)
        end = _positive(self.end_seconds, "USAGE_END", zero=True)
        if end < start or not all((self.allocation_id, self.attempt_id, self.evidence_sha256)):
            raise ResourceBoundary("USAGE_IDENTITY_OR_INTERVAL")
        if self.track not in {"D", "S"} or self.category not in {"MODEL_LOAD", "FIDELITY", "DIAGNOSTIC", "EVALUATION", "REPLAY", "PRIMARY", "TECHNICAL_ATTEMPT", "ALLOCATED_RESIDENCY"}:
            raise ResourceBoundary("USAGE_CATEGORY")
        return (end - start) * _count(self.gpus, "USAGE_GPUS")


@dataclass(frozen=True)
class UsageLedger:
    intervals: tuple[UsageInterval, ...] = ()

    def append(self, item: UsageInterval) -> "UsageLedger":
        _ = item.gpu_seconds
        for prior in self.intervals:
            if prior.allocation_id == item.allocation_id and max(prior.start_seconds, item.start_seconds) < min(prior.end_seconds, item.end_seconds):
                raise ResourceBoundary("DOUBLE_CHARGED_ALLOCATION_INTERVAL")
        candidate = UsageLedger(self.intervals + (item,))
        _ = candidate.gpu_seconds
        return candidate

    @property
    def gpu_seconds(self) -> float:
        # Direct construction is validated too, not only append().
        for i, item in enumerate(self.intervals):
            for other in self.intervals[:i]:
                if other.allocation_id == item.allocation_id and max(other.start_seconds, item.start_seconds) < min(other.end_seconds, item.end_seconds):
                    raise ResourceBoundary("DOUBLE_CHARGED_ALLOCATION_INTERVAL")
        return math.fsum(item.gpu_seconds for item in self.intervals)

    def receipt(self) -> dict:
        payload = {"intervals": [asdict(x) for x in self.intervals], "charged_gpu_seconds": self.gpu_seconds,
                   "technical_attempts_excluded_from_budget_count": 0}
        payload["identity_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return payload


def assess_admission(*, policy: ResourcePolicy, budget: BudgetAuthority,
                     snapshot: SchedulerSnapshot, ledger: UsageLedger,
                     new_processes: int, reserved_wall_seconds_per_process: float,
                     now_seconds: float, freshness_seconds: float = 30,
                     other_reserved_gpu_seconds: float = 0) -> dict:
    """Return a fail-closed CPU decision; this is not a scheduler reservation.

    The caller must include already charged residency in ``ledger`` and future
    uncharged active/pending task reservations in ``other_reserved_gpu_seconds``.
    Immediately before submission the repository live resource helper is also
    required. Budget authority absence is independent of D/S fixture readiness.
    """
    _count(new_processes, "NEW_PROCESSES")
    duration = _positive(reserved_wall_seconds_per_process, "RESERVED_WALL_SECONDS")
    now = _positive(now_seconds, "NOW", zero=True)
    fresh = _positive(freshness_seconds, "FRESHNESS")
    other = _positive(other_reserved_gpu_seconds, "OTHER_RESERVED", zero=True)
    charged = ledger.gpu_seconds
    reasons = []
    active = None
    if not snapshot.query_ok or not snapshot.allocation_query_complete:
        reasons.append("SCHEDULER_QUERY_UNRESOLVED")
    elif not snapshot.evidence_sha256 or not math.isfinite(snapshot.captured_at_seconds) or not 0 <= now - snapshot.captured_at_seconds <= fresh:
        reasons.append("SCHEDULER_EVIDENCE_STALE_OR_MISSING")
    else:
        try:
            ids = [x.allocation_id for x in snapshot.allocations]
            if len(ids) != len(set(ids)):
                raise ResourceBoundary("DUPLICATE_ALLOCATION_ID")
            active = sum(x.occupied_gpus() for x in snapshot.allocations)
        except ResourceBoundary as exc:
            reasons.append(str(exc))
    requested_gpu_seconds = new_processes * duration
    if active is not None and active + new_processes > policy.project_gpu_cap:
        reasons.append("PROJECT_GPU_CAP_EXCEEDED")
    if budget.gpu_hours is None:
        reasons.append("GPU_HOUR_BUDGET_UNASSIGNED")
        remaining = None
    else:
        remaining = 3600 * budget.gpu_hours - charged - other
        if requested_gpu_seconds > remaining:
            reasons.append("GPU_HOUR_BUDGET_EXCEEDED")
    return {"status": "ALLOW_NEW_SUBMISSION" if not reasons else "WAITING_FOR_ISOLATED_RESOURCE",
            "blocking_reasons": reasons, "active_project_gpus": active,
            "new_processes": new_processes, "policy": asdict(policy),
            "budget": asdict(budget), "charged_gpu_seconds": charged,
            "other_reserved_gpu_seconds": other, "remaining_gpu_seconds": remaining,
            "requested_gpu_seconds": requested_gpu_seconds,
            "scheduler_evidence_sha256": snapshot.evidence_sha256,
            "ledger_identity_sha256": ledger.receipt()["identity_sha256"],
            "is_reservation": False, "existing_job_mutation_count": 0,
            "requires_immediate_repository_live_resource_check": True}


def validate_sbatch_memory(text: str, *, expected_mib: int = 182272) -> dict:
    import re
    # All Slurm memory forms count; hidden --mem-per-* forms are not allowed.
    directives = [line for line in text.splitlines() if re.match(r"^\s*#SBATCH(?:\s|$)", line)]
    memory = [(line, re.findall(r"(?<!\S)--mem(?:-per-cpu|-per-gpu)?(?:=|\s+)\S+", line)) for line in directives]
    options = [option for _, matches in memory for option in matches]
    exact = re.fullmatch(r"--mem(?:=|\s+)([0-9]+)M", options[0]) if len(options) == 1 else None
    if exact is None or int(exact.group(1)) != expected_mib or expected_mib > 182272:
        raise ResourceBoundary("EXACTLY_ONE_EXPLICIT_JOB_TOTAL_MEM_182272M_REQUIRED")
    return {"explicit_mem_count": 1, "job_total_mem_mib": expected_mib}


def estimate_from_publication(repo: str | Path) -> dict:
    """Recompute cost proxies from immutable CSV bytes, never live artifacts.

    JV four-node write-minus-evaluator includes some fixed lifecycle cost; a
    34/4 multiplier is explicitly a proxy, not an isolated per-node timer.
    Unknown components are never imputed as zero or counted as an authorized cap.
    """
    bindings, tables = {}, {}
    for name in ("compute_accounting.csv", "run_registry.csv"):
        raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{PUBLICATION}:{PACKAGE}/{name}"])
        bindings[name] = {"publication_head": PUBLICATION, "path": f"{PACKAGE}/{name}",
                          "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        tables[name] = list(csv.DictReader(io.StringIO(raw.decode())))
    rows = tables["compute_accounting.csv"]
    if len(rows) != 40 or len({(x["alias"], x["arm"], x["batch"]) for x in rows}) != 40:
        raise ResourceBoundary("HISTORICAL_COST_DENOMINATOR")
    models = {}
    for alias in ("llama3-8b-inst", "qwen2.5-7b-inst"):
        jv = sorted((x for x in rows if x["alias"] == alias and x["arm"] == "JV_NATIVE"), key=lambda x: int(x["batch"]))
        official = sorted((x for x in rows if x["alias"] == alias and x["arm"] == "O_NATIVE"), key=lambda x: int(x["batch"]))
        if [int(x["batch"]) for x in jv] != list(range(1, 11)) or [int(x["batch"]) for x in official] != list(range(1, 11)):
            raise ResourceBoundary("HISTORICAL_MODEL_BATCH_IDENTITY")
        wall = lambda row, key: _positive(float(json.loads(row[key])["wall"]), key, zero=True)
        core = [wall(x, "write_including_endpoint") - float(x["endpoint_evaluation_seconds"]) for x in jv]
        oc = [wall(x, "write_including_endpoint") - float(x["endpoint_evaluation_seconds"]) for x in official]
        ev = [float(x["endpoint_evaluation_seconds"]) for x in jv + official]
        z = [wall(x, "compute_z") for x in jv]
        load = [float(x["model_load_and_context_setup_seconds"]) for x in tables["run_registry.csv"] if x["alias"] == alias]
        components = {"target_b100_once_seconds": z, "jv_four_node_core_proxy_seconds": core,
                      "official_core_proxy_seconds": oc, "endpoint_b100_evaluation_seconds": ev,
                      "model_load_and_context_setup_seconds": load}
        distributions = {name: {"min": min(values), "mean": statistics.mean(values), "max": max(values)} for name, values in components.items()}
        proxy = {name: (reducer(z) + 34 / 4 * reducer(core) + reducer(oc) + 11 * reducer(ev) + reducer(load)) / 3600
                 for name, reducer in (("min_component_proxy", min), ("mean_component_proxy", statistics.mean), ("max_component_proxy", max))}
        d_proxy = (statistics.mean(z) + 2 * statistics.mean(core) + statistics.mean(oc) + 4 * statistics.mean(ev) + statistics.mean(load)) / 3600
        replay = sum(wall(x, "full_batch") for x in jv if 6 <= int(x["batch"]) <= 9) / 3600
        models[alias] = {"historical_component_seconds": distributions,
                         "S_recorded_component_proxy_gpu_hours": proxy,
                         "D_main_current100_recorded_component_proxy_gpu_hours": d_proxy if alias.startswith("llama") else "NOT_APPLICABLE_D_MAIN_IS_LLAMA",
                         "D_optional_W5_to_W9_exact_replay_historical_gpu_hours": replay if alias.startswith("llama") else "NOT_AUTHORIZED_REPLAY_TARGET",
                         "nominal_S_paths": 7, "nominal_S_nodes": 34, "nominal_S_main_target_JVP": 170,
                         "S_JV_endpoints": 9, "S_official_endpoints": 1, "S_entry_evaluations": 1}
    return {"status": "ESTIMATE_ONLY_GPU_HOUR_BUDGET_UNASSIGNED", "gpu_hour_cap": None,
            "runtime_head": RUNTIME, "input_bindings": bindings, "models": models,
            "S_both_models_recorded_component_proxy_gpu_hours": {
                key: sum(m["S_recorded_component_proxy_gpu_hours"][key] for m in models.values())
                for key in ("min_component_proxy", "mean_component_proxy", "max_component_proxy")},
            "formula_S": "load + one B100 z + (34/4)*(historical JV N4 writer minus endpoint evaluator) + Official writer minus evaluator + 11 current-B100 evaluations",
            "formula_D_main_proxy": "load + one B100 z + 2*(historical JV N4 writer minus evaluator) + Official writer minus evaluator + 4 current-B100 evaluations",
            "formula_D_replay": "sum historical JV full_batch elapsed seconds B6..B9 / 3600; one optional replay, separate from D main",
            "not_separately_recorded_or_not_comparable": [
                "new first-entry/nonzero-node FD fidelity", "D repeated-capture and requestwise raw-bundle observations",
                "D four old900 rewrite evaluations and cross-objective observations", "D optional C-B1/C-B6 control observations",
                "new repeated Family/endpoint restoration overhead beyond historical proxy", "technical-attempt reservation",
                "server2-to-server1 hardware/contention and cold-vs-historical-state speed ratio",
            ],
            "limitations": ["min/max component proxies are not confidence intervals or hard upper bounds",
                            "per-node attribution is not resolved by a mixed writer lifecycle timer",
                            "stored terminal_compute.wall is a monotonic clock reading, not elapsed time and is not used",
                            "no GPU budget or scheduling reservation is inferred from these estimates"],
            "scientific_promotion": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(estimate_from_publication(args.repo), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
