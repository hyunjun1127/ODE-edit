#!/usr/bin/env python3
"""Create-once server1 submitter for the exact S05 causal pair."""

from __future__ import annotations

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

from project.run_scripts import session05_ode_bf_target_new_nll_dry_plan as dry
from project.run_scripts.ode_bf.artifacts import (
    ODEBFArtifactGuard,
    load_rooted_json,
    sha256_file,
)
from project.run_scripts.ode_bf.contracts import MODEL_ALIASES, ODEBFContractError
from project.run_scripts.ode_bf.p1_target_new_panel import (
    TARGET_NEW_INSTRUCTION_ID,
    TARGET_NEW_RESULT_TOKEN,
    expected_target_new_result_name,
)
from project.run_scripts.ode_bf.resource import (
    MEMORY_CAP_MIB_PER_GPU,
    gpu_count_from_tres,
)


SESSION_ID = "019fc63e-5217-7250-9c22-c5b2ec4248f0"
HANDOFF_BASE = "18d13fbea2d5f58ef665f50fcc9fa255d01097e5"
EXECUTION_BRANCH = "codex/odeeditsh1-s05-target-new-nll-v1"
SERVER1_GPU_CAP = 3
AUTHORIZATION_TOKEN = "target-new-nll-routing-p1-v1"
SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_target_new_nll.sbatch"
SOURCE_MANIFEST = (
    REPO_ROOT
    / "project/run_scripts/ode_bf/locks/source_manifest_s05_target_new_nll.json"
)
HANDOFF_ROOT = Path(
    "/mnt/raid5/janghj/ODE-edit/local/source-handoff/odeedit-s05-sh1-18d13fbea2d5-v1"
)
HANDOFF_SHA256 = {
    "source.bundle": "1393bdca2a30605152ae545291d5620a48e56e3b1b16cea2a04aaab5702e45c1",
    "source-path-manifest.json": "8aaf32cb5e697e28913f4053e23bced58d7ef0ea5bcd7c4b678a515b900a9dd3",
    "handoff-receipt.json": "223094727bb29e8faac0d3118b0e452f7c039e21a472aafc7660151730ba91a7",
}


def _run(
    args: Sequence[str],
    *,
    check: bool = True,
    cwd: Path = REPO_ROOT,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _write_once(path: Path, value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
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


def _verify_handoff() -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, expected in HANDOFF_SHA256.items():
        path = HANDOFF_ROOT / name
        if path.is_symlink() or not path.is_file():
            raise ODEBFContractError("S05 source handoff path differs")
        digest = sha256_file(path)
        if digest != expected:
            raise ODEBFContractError("S05 source handoff digest differs")
        observed[name] = digest
    verified = _run(["git", "bundle", "verify", str(HANDOFF_ROOT / "source.bundle")])
    if HANDOFF_BASE not in verified.stdout + verified.stderr:
        raise ODEBFContractError("S05 source handoff commit differs")
    heads = _run(
        ["git", "bundle", "list-heads", str(HANDOFF_ROOT / "source.bundle")]
    ).stdout.splitlines()
    if heads != [f"{HANDOFF_BASE} refs/heads/odeedit-sh1-source-18d13fb"]:
        raise ODEBFContractError("S05 source handoff ref differs")
    manifest = json.loads(
        (HANDOFF_ROOT / "source-path-manifest.json").read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (HANDOFF_ROOT / "handoff-receipt.json").read_text(encoding="utf-8")
    )
    entries = manifest.get("entries")
    if (
        manifest.get("schema") != "odeedit-source-path-manifest-v1"
        or manifest.get("target_commit") != HANDOFF_BASE
        or not isinstance(entries, list)
        or len(entries) != 660
        or len({item.get("path") for item in entries if isinstance(item, dict)})
        != 660
        or receipt.get("schema") != "odeedit-source-handoff-receipt-v1"
        or receipt.get("target_commit") != HANDOFF_BASE
        or receipt.get("contained_ref")
        != "refs/heads/odeedit-sh1-source-18d13fb"
        or receipt.get("path_count") != 660
        or receipt.get("bundle_sha256") != HANDOFF_SHA256["source.bundle"]
        or receipt.get("path_manifest_sha256")
        != HANDOFF_SHA256["source-path-manifest.json"]
        or receipt.get("admin_commit_excluded") is not True
        or receipt.get("uncommitted_state_excluded") is not True
        or receipt.get("local_artifacts_excluded") is not True
    ):
        raise ODEBFContractError("S05 source handoff manifest contract differs")
    return observed


def _source_manifest_gate(source_head: str) -> str:
    value, raw_sha256 = load_rooted_json(
        SOURCE_MANIFEST,
        expected_schema="ode-edit-s05-target-new-nll-source-manifest/v1",
    )
    entries = value.get("entries")
    if (
        value.get("instruction_id") != TARGET_NEW_INSTRUCTION_ID
        or value.get("expected_parent") != HANDOFF_BASE
        or value.get("execution_head_policy") != "runtime-git-head"
        or not isinstance(entries, list)
        or not entries
    ):
        raise ODEBFContractError("S05 source manifest header differs")
    observed_paths: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ODEBFContractError("S05 source manifest entry differs")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative.startswith("project/run_scripts/"):
            raise ODEBFContractError("S05 source manifest path differs")
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ODEBFContractError("S05 source manifest file differs")
        if path.stat().st_size != entry.get("size") or sha256_file(path) != entry.get("sha256"):
            raise ODEBFContractError("S05 source manifest content differs")
        observed_paths.append(relative)
    if observed_paths != sorted(observed_paths) or len(observed_paths) != len(
        set(observed_paths)
    ):
        raise ODEBFContractError("S05 source manifest path repeats")
    tracked = set(
        _run(["git", "ls-tree", "-r", "--name-only", source_head])
        .stdout.splitlines()
    )
    manifest_relative = SOURCE_MANIFEST.relative_to(REPO_ROOT).as_posix()
    expected_paths = {
        relative
        for relative in tracked
        if (
            relative.startswith("project/run_scripts/ode_bf/")
            or relative.startswith("project/run_scripts/session05_ode_bf_")
        )
        and relative != manifest_relative
    }
    if set(observed_paths) != expected_paths:
        raise ODEBFContractError("S05 source manifest path set differs")
    changed = set(
        _run(["git", "diff", "--name-only", HANDOFF_BASE, source_head]).stdout.splitlines()
    )
    if not changed <= set(observed_paths) | {manifest_relative}:
        raise ODEBFContractError("S05 source diff escapes manifest")
    return raw_sha256


def _scheduler_snapshot() -> list[dict[str, Any]]:
    user = os.environ.get("USER")
    if not user:
        raise ODEBFContractError("scheduler user identity is unavailable")
    output = _run(
        [
            "squeue",
            "-h",
            "-u",
            user,
            "-t",
            "R,PD",
            "-o",
            "%i|%j|%T|%b",
        ]
    ).stdout
    values: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("|", 3)
        if len(fields) != 4:
            raise ODEBFContractError("scheduler row differs")
        job_id, name, state, tres = fields
        gpu = gpu_count_from_tres(tres)
        project_name = name.startswith(("odebf_", "odealloc_", "odeedit_"))
        if project_name and (gpu is None or gpu <= 0):
            raise ODEBFContractError("project scheduler GPU TRES is ambiguous")
        if gpu is None:
            gpu = 0
        values.append(
            {"job_id": job_id, "job_name": name, "state": state, "gpu": gpu}
        )
    return values


def _pre_submit(source_head: str) -> dict[str, Any]:
    if os.environ.get("ODEEDIT_S05_SUBMISSION_AUTHORIZED") != AUTHORIZATION_TOKEN:
        raise ODEBFContractError("S05 submission token differs")
    if _run(["git", "rev-parse", "HEAD"]).stdout.strip() != source_head:
        raise ODEBFContractError("S05 execution head differs")
    if (
        _run(["git", "rev-parse", "HEAD^"]).stdout.strip() != HANDOFF_BASE
        or _run(["git", "branch", "--show-current"]).stdout.strip()
        != EXECUTION_BRANCH
    ):
        raise ODEBFContractError("S05 execution ancestry/branch differs")
    if _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout:
        raise ODEBFContractError("S05 tracked source is not clean")
    _run(["scripts/check-session-boundary.sh", SESSION_ID])
    handoff = _verify_handoff()
    manifest_sha256 = _source_manifest_gate(source_head)
    plan = dry.build_plan(source_head)
    for job in plan["jobs"]:
        if (
            job["forecast"]["conservative_gpu_peak_mib"] > 65_000
            or job["forecast"]["conservative_host_peak_mib"] > 65_000
            or job["forecast"]["conservative_time_seconds"] > 86_400
        ):
            raise ODEBFContractError("S05 forecast exceeds allocation")
    for alias in MODEL_ALIASES:
        guard = ODEBFArtifactGuard(
            REPO_ROOT,
            REPO_ROOT / "project/run_scripts/ode_bf/locks/p0_artifact_lock.json",
            alias,
            require_held_ode_alloc=False,
        )
        guard.preflight()
        guard.assert_unchanged()
    scheduler = _scheduler_snapshot()
    project = [
        row
        for row in scheduler
        if row["state"] in ("RUNNING", "PENDING")
        and row["job_name"].startswith(("odebf_", "odealloc_", "odeedit_"))
    ]
    active_gpu = sum(int(row["gpu"]) for row in project)
    if active_gpu + 2 > SERVER1_GPU_CAP:
        raise ODEBFContractError("server1 ODE project GPU cap would be exceeded")
    if any(row["job_name"] in dry.JOB_NAMES.values() for row in scheduler):
        raise ODEBFContractError("S05 scheduler job name collides")
    if 2 * 65_000 > 2 * MEMORY_CAP_MIB_PER_GPU:
        raise ODEBFContractError("S05 pair host memory exceeds locked capacity")
    result_parent = REPO_ROOT / "local/odebf/results"
    log_parent = REPO_ROOT / "local/odebf/logs"
    for alias in MODEL_ALIASES:
        if (result_parent / expected_target_new_result_name(alias)).exists():
            raise ODEBFContractError("S05 result root collides")
        if list(log_parent.glob(f"{dry.JOB_NAMES[alias]}-*")):
            raise ODEBFContractError("S05 log namespace collides")
    return {
        "handoff": handoff,
        "source_manifest_sha256": manifest_sha256,
        "active_project_gpu": active_gpu,
        "server1_gpu_cap": SERVER1_GPU_CAP,
        "plan": plan,
    }


def _submit_one(*, alias: str, source_head: str) -> str:
    root = REPO_ROOT / "local/odebf/results" / expected_target_new_result_name(alias)
    logs = REPO_ROOT / "local/odebf/logs"
    logs.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = dry.JOB_NAMES[alias]
    completed = _run(
        [
            "sbatch",
            "--parsable",
            "--hold",
            f"--job-name={name}",
            f"--output={logs / (name + '-%j.out')}",
            f"--error={logs / (name + '-%j.err')}",
            f"--chdir={REPO_ROOT}",
            str(SBATCH),
            alias,
            str(root),
            source_head,
            TARGET_NEW_RESULT_TOKEN,
        ]
    )
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("S05 Slurm job ID differs")
    return job_id


def _cancel_held_jobs(jobs: Mapping[str, str]) -> dict[str, bool]:
    cancelled: dict[str, bool] = {}
    for alias, job_id in jobs.items():
        completed = _run(["scancel", job_id], check=False)
        cancelled[alias] = completed.returncode == 0
    return cancelled


def _submit_held_pair(
    *,
    source_head: str,
    state: Path,
    intent_sha256: str,
) -> tuple[dict[str, str], str]:
    """Accept both held jobs before releasing either one to execute."""

    jobs: dict[str, str] = {}
    failure = state / "s05-target-new-nll-routing-p1-v1.submission-failure.json"
    try:
        for alias in MODEL_ALIASES:
            job_id = _submit_one(alias=alias, source_head=source_head)
            jobs[alias] = job_id
            _write_once(
                state
                / f"s05-target-new-nll-routing-p1-v1.{alias}.held-job.json",
                {
                    "schema": "ode-edit-s05-target-new-nll-held-job/v1",
                    "instruction_id": TARGET_NEW_INSTRUCTION_ID,
                    "source_head": source_head,
                    "alias": alias,
                    "job_id": job_id,
                    "scheduler_hold": True,
                },
            )
        receipt = state / "s05-target-new-nll-routing-p1-v1.submission-receipt.json"
        receipt_sha256 = _write_once(
            receipt,
            {
                "schema": "ode-edit-s05-target-new-nll-submission-receipt/v1",
                "instruction_id": TARGET_NEW_INSTRUCTION_ID,
                "source_head": source_head,
                "intent_sha256": intent_sha256,
                "jobs": jobs,
                "pair_accepted_while_held": True,
                "release_command_job_count": 2,
                "retry_or_resubmit_authorized": False,
            },
        )
        _run(["scontrol", "release", *jobs.values()])
        return jobs, receipt_sha256
    except BaseException as exc:
        cancelled = _cancel_held_jobs(jobs)
        _write_once(
            failure,
            {
                "schema": "ode-edit-s05-target-new-nll-submission-failure/v1",
                "instruction_id": TARGET_NEW_INSTRUCTION_ID,
                "source_head": source_head,
                "intent_sha256": intent_sha256,
                "accepted_held_jobs": jobs,
                "cancelled_before_pair_release": cancelled,
                "exception_class": type(exc).__name__,
                "exception_message_sha256": hashlib.sha256(
                    str(exc).encode("utf-8")
                ).hexdigest(),
                "pair_released": False,
            },
        )
        raise


def main() -> int:
    source_head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    preflight = _pre_submit(source_head)
    state = REPO_ROOT / "local/odebf/state"
    intent = state / "s05-target-new-nll-routing-p1-v1.submission-intent.json"
    intent_sha256 = _write_once(
        intent,
        {
            "schema": "ode-edit-s05-target-new-nll-submission-intent/v1",
            "instruction_id": TARGET_NEW_INSTRUCTION_ID,
            "source_head": source_head,
            "preflight": preflight,
        },
    )
    jobs, receipt_sha256 = _submit_held_pair(
        source_head=source_head,
        state=state,
        intent_sha256=intent_sha256,
    )
    print(
        json.dumps(
            {"jobs": jobs, "intent_sha256": intent_sha256, "receipt_sha256": receipt_sha256},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
