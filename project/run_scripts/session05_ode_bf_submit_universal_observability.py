#!/usr/bin/env python3
"""Create-once, A2-bound SH1 submitter for R13 universal-observability cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts import (
    session05_ode_bf_universal_observability_dry_plan as dry,
)
from project.run_scripts.ode_bf.artifacts import (
    ODEBFArtifactGuard,
    load_rooted_json,
    sha256_file,
)
from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.ode_bf_observability import paired_arm_ids
from project.run_scripts.ode_bf.p1_universal_observability_panel import (
    UNIVERSAL_OBS_AMENDMENT_ID,
    UNIVERSAL_OBS_BASE_RUNTIME,
    UNIVERSAL_OBS_INSTRUCTION_ID,
    UNIVERSAL_OBS_SCHEMA_NAMESPACE,
    UNIVERSAL_OBS_SESSION_SOURCE_PATHS,
    UNIVERSAL_OBS_SOURCE_MANIFEST_FILE,
    common_cold_schedule,
    expected_universal_observability_result_name,
    load_and_validate_r12_frozen_reference,
    verify_common_cold_case_seal,
)
from project.run_scripts.ode_bf.resource import gpu_count_from_tres
from project.run_scripts.ode_bf.sampling import load_p1_sampling_seal


LLAMA_ALIAS = dry.LLAMA_ALIAS
CELL_ORDER = dry.CELL_ORDER
SERVER1_PROJECT_GPU_CAP = dry.SERVER1_PROJECT_GPU_CAP
SESSION_ID = dry.SESSION_ID
EXECUTION_BRANCH = dry.EXECUTION_BRANCH
EXECUTION_PARENT = dry.EXECUTION_PARENT
EXECUTION_AMENDMENT = dry.EXECUTION_AMENDMENT
EXECUTION_SCHEMA_NAMESPACE = f"{UNIVERSAL_OBS_SCHEMA_NAMESPACE}-a2-sh1"
APPROVAL_ENV = "ODEEDIT_S05_UNIVERSAL_OBSERVABILITY_P1R13_A2_SH1_RUN_APPROVAL"
SUBMISSION_NAMESPACE = "s05-universal-obs-rs-bg-targethold-p1r13-a2-sh1-llama-v1"
SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_universal_observability.sbatch"
SOURCE_MANIFEST = (
    REPO_ROOT / "project/run_scripts/ode_bf/locks" / UNIVERSAL_OBS_SOURCE_MANIFEST_FILE
)
DEFAULT_STATE_ROOT = REPO_ROOT / "local/odebf/state"


def _run(
    args: Sequence[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=REPO_ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _valid_source_head(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _write_once(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _execution_provenance_gate(source_head: str) -> dict[str, str | bool]:
    """Require the supplied current HEAD and its exact A2 ancestry."""

    if not _valid_source_head(source_head):
        raise ODEBFContractError("universal-observability source head differs")
    expected_approval = f"{EXECUTION_AMENDMENT}:{source_head}"
    if os.environ.get(APPROVAL_ENV) != expected_approval:
        raise ODEBFContractError(
            "universal-observability checkpoint-bound approval is absent"
        )
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    parent = _run(["git", "rev-parse", "HEAD^"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(
        ["git", "status", "--porcelain", "--untracked-files=no"]
    ).stdout
    ancestor = _run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            UNIVERSAL_OBS_BASE_RUNTIME,
            head,
        ],
        check=False,
    )
    if (
        head != source_head
        or parent != EXECUTION_PARENT
        or branch != EXECUTION_BRANCH
        or ancestor.returncode != 0
        or dirty
    ):
        raise ODEBFContractError("universal-observability execution provenance differs")
    return {
        "checkpoint_bound_approval": expected_approval,
        "execution_amendment": EXECUTION_AMENDMENT,
        "execution_head": head,
        "exact_execution_parent": parent,
        "base_runtime_is_ancestor": True,
        "branch": branch,
        "tracked_and_index_clean": True,
    }


def _safe_directory(path: Path, *, label: str) -> None:
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise ODEBFContractError(f"universal-observability {label} parent differs")


def _state_root_gate(state_root: Path) -> Path:
    expected = DEFAULT_STATE_ROOT.resolve()
    observed = state_root.expanduser().resolve()
    if observed != expected:
        raise ODEBFContractError("universal-observability state root differs")
    _safe_directory(observed, label="state")
    return observed


def _result_root_for(job: Mapping[str, Any]) -> Path:
    return REPO_ROOT / "local/odebf/results" / expected_universal_observability_result_name(
        LLAMA_ALIAS, str(job["cell_id"])
    )


def _held_job_path(state_root: Path, cell_id: str) -> Path:
    return state_root / f"{SUBMISSION_NAMESPACE}.{cell_id}.held-job.json"


def _wave_token(jobs: Sequence[Mapping[str, Any]]) -> str:
    cells = [str(job["cell_id"]) for job in jobs]
    if (
        not cells
        or any(cell not in CELL_ORDER for cell in cells)
        or len(cells) != len(set(cells))
        or cells != [cell for cell in CELL_ORDER if cell in cells]
    ):
        raise ODEBFContractError("universal-observability wave cell inventory differs")
    return "-".join(cell.lower().replace("-", "_") for cell in cells)


def _wave_state_paths(
    state_root: Path, jobs: Sequence[Mapping[str, Any]]
) -> dict[str, Path]:
    token = _wave_token(jobs)
    prefix = state_root / f"{SUBMISSION_NAMESPACE}.wave-{token}"
    return {
        "intent": prefix.with_name(prefix.name + ".intent.json"),
        "submission_receipt": prefix.with_name(
            prefix.name + ".submission-receipt.json"
        ),
        "submission_failure": prefix.with_name(
            prefix.name + ".submission-failure.json"
        ),
    }


def _load_prior_cell_ownership(
    state_root: Path,
    jobs: Sequence[Mapping[str, Any]],
    *,
    source_head: str,
) -> dict[str, dict[str, Any]]:
    """Load the immutable held-job receipts that own previously submitted cells."""

    expected_paths = {
        _held_job_path(state_root, str(job["cell_id"])): job for job in jobs
    }
    observed_paths = set(state_root.glob(f"{SUBMISSION_NAMESPACE}.*.held-job.json"))
    if not observed_paths <= set(expected_paths):
        raise ODEBFContractError("universal-observability held-job state path differs")
    owned: dict[str, dict[str, Any]] = {}
    for path, job in expected_paths.items():
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file():
            raise ODEBFContractError("universal-observability held-job state differs")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ODEBFContractError(
                "universal-observability held-job state is unreadable"
            ) from exc
        cell_id = str(job["cell_id"])
        job_id = value.get("job_id") if isinstance(value, Mapping) else None
        if (
            not isinstance(value, Mapping)
            or value.get("schema") != f"{EXECUTION_SCHEMA_NAMESPACE}-held-job/v1"
            or value.get("instruction_id") != UNIVERSAL_OBS_INSTRUCTION_ID
            or value.get("amendment_id") != UNIVERSAL_OBS_AMENDMENT_ID
            or value.get("execution_amendment") != EXECUTION_AMENDMENT
            or value.get("source_head") != source_head
            or value.get("alias") != LLAMA_ALIAS
            or value.get("cell_id") != cell_id
            or value.get("job_name") != job["job_name"]
            or value.get("run_token") != job["run_token"]
            or value.get("submission_namespace") != SUBMISSION_NAMESPACE
            or not isinstance(value.get("wave_token"), str)
            or not value["wave_token"]
            or value.get("scheduler_hold") is not True
            or not isinstance(job_id, str)
            or not job_id.isdigit()
        ):
            raise ODEBFContractError("universal-observability held-job receipt differs")
        owned[cell_id] = dict(value)
    return owned


def _validate_prior_wave_receipts(
    state_root: Path,
    ownership: Mapping[str, Mapping[str, Any]],
    *,
    source_head: str,
) -> None:
    """Bind every prior held receipt to one immutable accepted-wave receipt."""

    receipt_paths = sorted(
        state_root.glob(f"{SUBMISSION_NAMESPACE}.wave-*.submission-receipt.json")
    )
    observed: dict[str, tuple[str, str]] = {}
    for path in receipt_paths:
        if path.is_symlink() or not path.is_file():
            raise ODEBFContractError(
                "universal-observability submission receipt state differs"
            )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ODEBFContractError(
                "universal-observability submission receipt is unreadable"
            ) from exc
        jobs = value.get("jobs") if isinstance(value, Mapping) else None
        wave_token = value.get("wave_token") if isinstance(value, Mapping) else None
        submitted_cells = (
            value.get("submitted_cells") if isinstance(value, Mapping) else None
        )
        expected_cells = (
            [cell_id for cell_id in CELL_ORDER if isinstance(jobs, Mapping) and cell_id in jobs]
        )
        if (
            not isinstance(value, Mapping)
            or value.get("schema")
            != f"{EXECUTION_SCHEMA_NAMESPACE}-submission-receipt/v1"
            or value.get("instruction_id") != UNIVERSAL_OBS_INSTRUCTION_ID
            or value.get("amendment_id") != UNIVERSAL_OBS_AMENDMENT_ID
            or value.get("execution_amendment") != EXECUTION_AMENDMENT
            or value.get("source_head") != source_head
            or value.get("alias") != LLAMA_ALIAS
            or value.get("submission_namespace") != SUBMISSION_NAMESPACE
            or not isinstance(wave_token, str)
            or not wave_token
            or not isinstance(jobs, Mapping)
            or submitted_cells != expected_cells
            or wave_token
            != _wave_token([{"cell_id": cell_id} for cell_id in expected_cells])
            or path.name
            != f"{SUBMISSION_NAMESPACE}.wave-{wave_token}.submission-receipt.json"
            or value.get("all_accepted_while_held") is not True
        ):
            raise ODEBFContractError(
                "universal-observability submission receipt differs"
            )
        for cell_id, job_id in jobs.items():
            if (
                not isinstance(cell_id, str)
                or cell_id not in CELL_ORDER
                or not isinstance(job_id, str)
                or not job_id.isdigit()
                or cell_id in observed
            ):
                raise ODEBFContractError(
                    "universal-observability submission receipt ownership differs"
                )
            observed[cell_id] = (str(wave_token), job_id)
    if set(observed) != set(ownership):
        raise ODEBFContractError(
            "universal-observability held/submission ownership differs"
        )
    for cell_id, held in ownership.items():
        wave_token, job_id = observed[cell_id]
        if held.get("wave_token") != wave_token or held.get("job_id") != job_id:
            raise ODEBFContractError(
                "universal-observability held/submission receipt differs"
            )


def _reconcile_cell_ownership(
    state_root: Path,
    jobs: Sequence[Mapping[str, Any]],
    ownership: Mapping[str, Mapping[str, Any]],
    scheduler: Sequence[Mapping[str, Any]],
) -> None:
    """Fail closed on every local/scheduler namespace ambiguity.

    A prior receipt may correspond to its one live Slurm job or a completed job
    with a create-once result root.  A cell without a receipt must have neither.
    """

    del state_root  # Ownership paths were checked while loading their receipts.
    result_parent = REPO_ROOT / "local/odebf/results"
    log_parent = REPO_ROOT / "local/odebf/logs"
    _safe_directory(result_parent, label="result")
    _safe_directory(log_parent, label="log")
    scheduler_by_name: dict[str, list[Mapping[str, Any]]] = {}
    for row in scheduler:
        name = row.get("job_name")
        if isinstance(name, str) and name in dry.JOB_NAMES.values():
            scheduler_by_name.setdefault(name, []).append(row)
    for job in jobs:
        cell_id = str(job["cell_id"])
        name = str(job["job_name"])
        rows = scheduler_by_name.get(name, [])
        if len(rows) > 1:
            raise ODEBFContractError("universal-observability scheduler job duplicates")
        root = _result_root_for(job)
        root_exists = root.exists() or root.is_symlink()
        if root_exists and (root.is_symlink() or not root.is_dir()):
            raise ODEBFContractError("universal-observability result root differs")
        logs = [] if not log_parent.exists() else list(log_parent.glob(f"{name}-*"))
        if any(path.is_symlink() or not path.is_file() for path in logs):
            raise ODEBFContractError("universal-observability log namespace differs")
        receipt = ownership.get(cell_id)
        if receipt is None:
            if rows:
                raise ODEBFContractError(
                    "universal-observability scheduler job lacks held-job ownership"
                )
            if root_exists or logs:
                raise ODEBFContractError(
                    "universal-observability namespace lacks held-job ownership"
                )
            continue
        if rows:
            row = rows[0]
            if row.get("job_id") != receipt["job_id"] or row.get("gpu") != 1:
                raise ODEBFContractError(
                    "universal-observability scheduler receipt differs"
                )
        elif not root_exists:
            raise ODEBFContractError(
                "universal-observability held-job is missing scheduler/result state"
            )


def _source_manifest_gate(source_head: str) -> str:
    value, raw_sha256 = load_rooted_json(
        SOURCE_MANIFEST,
        expected_schema=f"{UNIVERSAL_OBS_SCHEMA_NAMESPACE}-source-manifest/v1",
    )
    entries = value.get("entries")
    if (
        value.get("instruction_id") != UNIVERSAL_OBS_INSTRUCTION_ID
        or value.get("amendment_id") != UNIVERSAL_OBS_AMENDMENT_ID
        or value.get("base_runtime") != UNIVERSAL_OBS_BASE_RUNTIME
        or value.get("execution_branch") != EXECUTION_BRANCH
        or value.get("execution_head_policy") != "runtime-git-head"
        or not isinstance(entries, list)
        or not entries
    ):
        raise ODEBFContractError("universal-observability source manifest header differs")
    observed: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ODEBFContractError("universal-observability source manifest entry differs")
        relative = str(entry["path"])
        path = REPO_ROOT / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry.get("size")
            or sha256_file(path) != entry.get("sha256")
        ):
            raise ODEBFContractError("universal-observability source manifest content differs")
        observed.append(relative)
    if observed != sorted(observed) or len(observed) != len(set(observed)):
        raise ODEBFContractError("universal-observability source manifest ordering differs")
    manifest_relative = SOURCE_MANIFEST.relative_to(REPO_ROOT).as_posix()
    tracked = set(
        _run(["git", "ls-tree", "-r", "--name-only", source_head]).stdout.splitlines()
    )
    expected = {
        relative
        for relative in tracked
        if (
            relative.startswith("project/run_scripts/ode_bf/")
            or relative in UNIVERSAL_OBS_SESSION_SOURCE_PATHS
        )
        and relative != manifest_relative
    }
    if set(observed) != expected:
        raise ODEBFContractError("universal-observability source manifest path set differs")
    return raw_sha256


def _scheduler_snapshot() -> list[dict[str, Any]]:
    user = os.environ.get("USER")
    if not user:
        raise ODEBFContractError("universal-observability scheduler user is unavailable")
    output = _run(
        ["squeue", "-h", "-u", user, "-t", "R,PD", "-o", "%i|%j|%T|%b"]
    ).stdout
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("|", 3)
        if len(fields) != 4:
            raise ODEBFContractError("universal-observability scheduler row differs")
        job_id, name, state, tres = fields
        gpu = gpu_count_from_tres(tres)
        project_job = name.startswith(("odebf_", "odealloc_", "odeedit_"))
        if project_job and (gpu is None or gpu <= 0):
            raise ODEBFContractError(
                "universal-observability project GPU TRES is ambiguous"
            )
        rows.append(
            {
                "job_id": job_id,
                "job_name": name,
                "state": state,
                "gpu": 0 if gpu is None else gpu,
                "project_job": project_job,
            }
        )
    return rows


def _host_memory_gate() -> int:
    total = next(
        (
            line
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
            if line.startswith("MemTotal:")
        ),
        None,
    )
    if total is None:
        raise ODEBFContractError("universal-observability host memory unavailable")
    fields = total.split()
    if len(fields) != 3 or fields[2] != "kB" or not fields[1].isdigit():
        raise ODEBFContractError("universal-observability host memory units differ")
    total_mib = int(fields[1]) // 1024
    if total_mib < 65_000:
        raise ODEBFContractError("universal-observability host memory is insufficient")
    return total_mib


def _validate_plan(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != len(CELL_ORDER):
        raise ODEBFContractError("universal-observability dry-plan inventory differs")
    if (
        plan.get("llama_sh1_only") is not True
        or plan.get("qwen_submission_authorized") is not False
        or plan.get("session_id") != SESSION_ID
        or plan.get("execution_branch") != EXECUTION_BRANCH
        or plan.get("execution_parent") != EXECUTION_PARENT
        or plan.get("execution_head_policy") != "runtime-git-head"
        or plan.get("execution_amendment") != EXECUTION_AMENDMENT
        or plan.get("approval_token")
        != f"{EXECUTION_AMENDMENT}:{plan.get('source_head')}"
        or plan.get("execution_checkpoint_bound") is not True
        or plan.get("execution_authorized") is not True
        or plan.get("execution_hold") is not False
        or plan.get("held_atomic_release_required") is not True
        or plan.get("server1_project_gpu_cap") != SERVER1_PROJECT_GPU_CAP
        or plan.get("active_plus_new_rule")
        != "active_project_gpu + selected_new_gpu <= 4"
        or plan.get("fast_launch_cell_order") != list(CELL_ORDER)
    ):
        raise ODEBFContractError("universal-observability dry-plan policy differs")
    validated: list[Mapping[str, Any]] = []
    for expected_cell, job in zip(CELL_ORDER, jobs, strict=True):
        if not isinstance(job, Mapping):
            raise ODEBFContractError("universal-observability dry-plan job differs")
        dynamic_arm, target_hold_arm = paired_arm_ids(expected_cell)
        if (
            job.get("alias") != LLAMA_ALIAS
            or job.get("cell_id") != expected_cell
            or job.get("job_name") != dry.JOB_NAMES[expected_cell]
            or job.get("result_name")
            != expected_universal_observability_result_name(LLAMA_ALIAS, expected_cell)
            or job.get("live_arms") != [dynamic_arm, target_hold_arm]
            or job.get("runtime_kwargs")
            != {"universal_observability_cell": expected_cell}
            or job.get("one_alias_one_cell") is not True
            or job.get("gpu") != 1
            or job.get("cpu") != 8
            or job.get("memory_mib") != 65_000
            or job.get("time") != "23:59:00"
            or job.get("node") != "server1"
        ):
            raise ODEBFContractError("universal-observability dry-plan allocation differs")
        forecast = job.get("forecast")
        if not isinstance(forecast, Mapping) or forecast.get("fits_envelope") is not True:
            raise ODEBFContractError("universal-observability forecast exceeds allocation")
        validated.append(job)
    return validated


def _select_fast_launch_jobs(
    jobs: Sequence[Mapping[str, Any]], *, active_project_gpu: int
) -> list[Mapping[str, Any]]:
    if type(active_project_gpu) is not int or active_project_gpu < 0:
        raise ODEBFContractError("universal-observability active GPU count differs")
    available = SERVER1_PROJECT_GPU_CAP - active_project_gpu
    if available <= 0:
        raise ODEBFContractError("universal-observability server1 project GPU cap is full")
    selected = list(jobs[:available])
    selected_gpu = sum(int(job["gpu"]) for job in selected)
    if (
        not selected
        or active_project_gpu + selected_gpu > SERVER1_PROJECT_GPU_CAP
    ):
        raise ODEBFContractError("universal-observability server1 GPU cap would be exceeded")
    return selected


def _artifact_gate() -> None:
    guard = ODEBFArtifactGuard(
        REPO_ROOT,
        REPO_ROOT / "project/run_scripts/ode_bf/locks/p0_artifact_lock.json",
        LLAMA_ALIAS,
        require_held_ode_alloc=False,
    )
    guard.preflight()
    guard.assert_unchanged()


def _r12_reference_gate() -> dict[str, Any]:
    locks = REPO_ROOT / "project/run_scripts/ode_bf/locks"
    seal = verify_common_cold_case_seal(
        json.loads((locks / "p1r10_common_coldcoord_cf_b10_seal.json").read_text(encoding="utf-8"))
    )
    schedule = common_cold_schedule(
        load_p1_sampling_seal(
            locks / "p1r2_p_population_seal.json",
            stream_path=locks / "p1r2_seqb10_stream_seal.json",
        )
    )
    from project.run_scripts.ode_bf.p1_controller import P1ControllerLock

    reference, reference_sha256 = load_and_validate_r12_frozen_reference(
        locks,
        controller_identity_sha256=P1ControllerLock().identity(),
        case_root_digest=seal["root_digest"],
        schedule=schedule,
        alias=LLAMA_ALIAS,
    )
    return {
        "r12_reference_lock_sha256": reference_sha256,
        "r12_reference_keys": sorted(reference),
        "r10_case_root_digest": seal["root_digest"],
    }


def _namespace_gates(
    state_root: Path, jobs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Reserve only the never-submitted cells selected for this wave."""

    result_parent = REPO_ROOT / "local/odebf/results"
    log_parent = REPO_ROOT / "local/odebf/logs"
    _safe_directory(result_parent, label="result")
    _safe_directory(log_parent, label="log")
    result_roots: dict[str, str] = {}
    for job in jobs:
        cell_id = str(job["cell_id"])
        root = _result_root_for(job)
        if root.exists() or root.is_symlink():
            raise ODEBFContractError("universal-observability result root collides")
        name = str(job["job_name"])
        if log_parent.exists() and list(log_parent.glob(f"{name}-*")):
            raise ODEBFContractError("universal-observability log namespace collides")
        result_roots[cell_id] = str(root)
    wave_paths = _wave_state_paths(state_root, jobs)
    for path in wave_paths.values():
        if path.exists() or path.is_symlink():
            raise ODEBFContractError("universal-observability state namespace collides")
    for job in jobs:
        cell_id = str(job["cell_id"])
        held = _held_job_path(state_root, cell_id)
        if held.exists() or held.is_symlink():
            raise ODEBFContractError("universal-observability held-job state collides")
    return {
        "result_roots": result_roots,
        "log_parent": str(log_parent),
        "state_root": str(state_root),
        "wave_token": _wave_token(jobs),
        "wave_state_paths": {key: str(path) for key, path in wave_paths.items()},
    }


def _pre_submit(source_head: str, *, state_root: Path) -> dict[str, Any]:
    # Provenance is first so no source/scheduler/state action precedes A2 binding.
    provenance = _execution_provenance_gate(source_head)
    _run(["scripts/check-session-boundary.sh", SESSION_ID])
    manifest_sha256 = _source_manifest_gate(source_head)
    plan = dry.build_plan(source_head)
    jobs = _validate_plan(plan)
    _artifact_gate()
    r12_reference = _r12_reference_gate()
    scheduler = _scheduler_snapshot()
    ownership = _load_prior_cell_ownership(
        state_root, jobs, source_head=source_head
    )
    _validate_prior_wave_receipts(
        state_root, ownership, source_head=source_head
    )
    _reconcile_cell_ownership(state_root, jobs, ownership, scheduler)
    remaining = [job for job in jobs if str(job["cell_id"]) not in ownership]
    if not remaining:
        raise ODEBFContractError(
            "universal-observability all live cells already have immutable ownership"
        )
    active_project_gpu = sum(row["gpu"] for row in scheduler if row["project_job"])
    selected = _select_fast_launch_jobs(
        remaining, active_project_gpu=active_project_gpu
    )
    namespaces = _namespace_gates(state_root, selected)
    selected_cells = [str(job["cell_id"]) for job in selected]
    deferred_cells = [
        str(job["cell_id"])
        for job in remaining
        if str(job["cell_id"]) not in selected_cells
    ]
    return {
        "execution_provenance": provenance,
        "session_gate": SESSION_ID,
        "access_gate": APPROVAL_ENV,
        "source_manifest_sha256": manifest_sha256,
        "artifact_gate": True,
        "r12_reference_gate": r12_reference,
        "prior_owned_cells": list(ownership),
        "remaining_unsubmitted_cells": [str(job["cell_id"]) for job in remaining],
        "active_project_gpu": active_project_gpu,
        "selected_new_gpu": sum(int(job["gpu"]) for job in selected),
        "server1_project_gpu_cap": SERVER1_PROJECT_GPU_CAP,
        "active_plus_new_gpu": active_project_gpu
        + sum(int(job["gpu"]) for job in selected),
        "fast_launch_selected_cells": selected_cells,
        "deferred_cells_due_to_capacity": deferred_cells,
        "wave_token": _wave_token(selected),
        "host_total_mib": _host_memory_gate(),
        "namespace_gates": namespaces,
        "jobs": [dict(job) for job in selected],
        "plan": dict(plan),
    }


def _submit_one(*, job: Mapping[str, Any], source_head: str) -> str:
    cell_id = str(job["cell_id"])
    name = str(job["job_name"])
    root = _result_root_for(job)
    logs = REPO_ROOT / "local/odebf/logs"
    logs.mkdir(mode=0o700, parents=True, exist_ok=True)
    result = _run(
        [
            "sbatch",
            "--parsable",
            "--hold",
            f"--job-name={name}",
            f"--output={logs / (name + '-%j.out')}",
            f"--error={logs / (name + '-%j.err')}",
            f"--chdir={REPO_ROOT}",
            str(SBATCH),
            LLAMA_ALIAS,
            str(root),
            source_head,
            str(job["run_token"]),
            cell_id,
        ]
    )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("universal-observability Slurm job ID differs")
    return job_id


def _cancel_held(jobs: Mapping[str, str]) -> dict[str, bool]:
    cancelled: dict[str, bool] = {}
    for cell_id, job_id in jobs.items():
        completed = _run(["scancel", job_id], check=False)
        cancelled[cell_id] = completed.returncode == 0
    return cancelled


def _submit_held_cells(
    *, source_head: str, state_root: Path, preflight: Mapping[str, Any], intent_sha256: str
) -> tuple[dict[str, str], str]:
    jobs = preflight["jobs"]
    if not isinstance(jobs, list) or not all(isinstance(job, Mapping) for job in jobs):
        raise ODEBFContractError("universal-observability selected job inventory differs")
    wave_token = _wave_token(jobs)
    if preflight.get("wave_token") != wave_token:
        raise ODEBFContractError("universal-observability preflight wave differs")
    wave_paths = _wave_state_paths(state_root, jobs)
    accepted: dict[str, str] = {}
    try:
        for job in jobs:
            cell_id = str(job["cell_id"])
            job_id = _submit_one(job=job, source_head=source_head)
            accepted[cell_id] = job_id
            _write_once(
                _held_job_path(state_root, cell_id),
                {
                    "schema": f"{EXECUTION_SCHEMA_NAMESPACE}-held-job/v1",
                    "instruction_id": UNIVERSAL_OBS_INSTRUCTION_ID,
                    "amendment_id": UNIVERSAL_OBS_AMENDMENT_ID,
                    "execution_amendment": EXECUTION_AMENDMENT,
                    "source_head": source_head,
                    "alias": LLAMA_ALIAS,
                    "cell_id": cell_id,
                    "job_name": job["job_name"],
                    "run_token": job["run_token"],
                    "job_id": job_id,
                    "scheduler_hold": True,
                    "submission_namespace": SUBMISSION_NAMESPACE,
                    "wave_token": wave_token,
                },
            )
        receipt_sha256 = _write_once(
            wave_paths["submission_receipt"],
            {
                "schema": f"{EXECUTION_SCHEMA_NAMESPACE}-submission-receipt/v1",
                "instruction_id": UNIVERSAL_OBS_INSTRUCTION_ID,
                "amendment_id": UNIVERSAL_OBS_AMENDMENT_ID,
                "execution_amendment": EXECUTION_AMENDMENT,
                "source_head": source_head,
                "intent_sha256": intent_sha256,
                "alias": LLAMA_ALIAS,
                "submission_namespace": SUBMISSION_NAMESPACE,
                "wave_token": wave_token,
                "jobs": accepted,
                "submitted_cells": list(accepted),
                "deferred_cells_due_to_capacity": preflight[
                    "deferred_cells_due_to_capacity"
                ],
                "all_accepted_while_held": True,
                "retry_or_resubmit_authorized": False,
            },
        )
        _run(["scontrol", "release", *accepted.values()])
        return accepted, receipt_sha256
    except BaseException as exc:
        cancelled = _cancel_held(accepted)
        _write_once(
            wave_paths["submission_failure"],
            {
                "schema": f"{EXECUTION_SCHEMA_NAMESPACE}-submission-failure/v1",
                "instruction_id": UNIVERSAL_OBS_INSTRUCTION_ID,
                "amendment_id": UNIVERSAL_OBS_AMENDMENT_ID,
                "execution_amendment": EXECUTION_AMENDMENT,
                "source_head": source_head,
                "intent_sha256": intent_sha256,
                "wave_token": wave_token,
                "accepted_held_jobs": accepted,
                "cancelled_before_release": cancelled,
                "exception_class": type(exc).__name__,
                "exception_message_sha256": hashlib.sha256(
                    str(exc).encode("utf-8")
                ).hexdigest(),
                "released": False,
            },
        )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    args = parser.parse_args(argv)
    state_root = _state_root_gate(args.state_root)
    preflight = _pre_submit(args.source_head, state_root=state_root)
    selected = preflight.get("jobs")
    if not isinstance(selected, list) or not all(
        isinstance(job, Mapping) for job in selected
    ):
        raise ODEBFContractError("universal-observability selected job inventory differs")
    wave_paths = _wave_state_paths(state_root, selected)
    intent_sha256 = _write_once(
        wave_paths["intent"],
        {
            "schema": f"{EXECUTION_SCHEMA_NAMESPACE}-submit-intent/v1",
            "instruction_id": UNIVERSAL_OBS_INSTRUCTION_ID,
            "amendment_id": UNIVERSAL_OBS_AMENDMENT_ID,
            "execution_amendment": EXECUTION_AMENDMENT,
            "source_head": args.source_head,
            "alias": LLAMA_ALIAS,
            "wave_token": _wave_token(selected),
            "preflight": preflight,
            "retry_or_resubmit_authorized": False,
        },
    )
    jobs, receipt_sha256 = _submit_held_cells(
        source_head=args.source_head,
        state_root=state_root,
        preflight=preflight,
        intent_sha256=intent_sha256,
    )
    print(
        json.dumps(
            {
                "status": "LLAMA_SH1_A2_CELLS_SUBMITTED",
                "instruction_id": UNIVERSAL_OBS_INSTRUCTION_ID,
                "amendment_id": UNIVERSAL_OBS_AMENDMENT_ID,
                "execution_amendment": EXECUTION_AMENDMENT,
                "source_head": args.source_head,
                "jobs": jobs,
                "intent_sha256": intent_sha256,
                "submission_receipt_sha256": receipt_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
