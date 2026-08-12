#!/usr/bin/env python3
"""Held-inspect-release submitter for the two P1R28 model jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts import session05_ode_bf_p1r28_corrected_coupling_dry_plan as dry
from project.run_scripts.ode_bf.contracts import ODEBFContractError

SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_p1r28_corrected_coupling.sbatch"
STATE_ROOT = REPO_ROOT / "local/odebf/state/p1r28-corrected-coupling"
RESULT_PARENT = REPO_ROOT / "local/odebf/results"
BRANCH = "codex/p1r28-corrected-coupling-v1"
PROJECT_GPU_CAP = 4
TASK_GPU_LIMIT = 2


def _run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=REPO_ROOT, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _write_once(path: Path, value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def submit(source_head: str, phase: str, *, attempt: str = "v1") -> dict[str, object]:
    if attempt not in ("v1", "tech-r1", "tech-r2", "tech-r3", "tech-r4"):
        raise ODEBFContractError("P1R28 submission attempt differs")
    if (
        source_head != _run(["git", "rev-parse", "HEAD"]).stdout.strip()
        or _run(["git", "branch", "--show-current"]).stdout.strip() != BRANCH
        or _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    ):
        raise ODEBFContractError("P1R28 execution source differs")
    plan = dry.build_plan(source_head, phase)
    attempt_result_parent = (
        RESULT_PARENT if attempt == "v1" else RESULT_PARENT / f"p1r28-{attempt}"
    )
    if any(
        (attempt_result_parent / str(job["result_name"])).exists()
        for job in plan["jobs"]
    ):
        raise ODEBFContractError("P1R28 result namespace exists")
    rows = _run(["squeue", "-h", "-u", "janghj", "-w", "devbox", "-t", "RUNNING", "-o", "%i|%j|%b"]).stdout.splitlines()
    active_rows = [row for row in rows if "gpu" in row.casefold() and "ode" in row.casefold()]
    active = len(active_rows)
    if active + TASK_GPU_LIMIT > PROJECT_GPU_CAP:
        raise ODEBFContractError("P1R28 server1 GPU cap differs")
    namespace = f"s05-p1r28-{phase}-{source_head[:12]}-{attempt}"
    intent_path = STATE_ROOT / f"{namespace}.intent.json"
    receipt_path = STATE_ROOT / f"{namespace}.submission-receipt.json"
    if any(path.exists() or path.is_symlink() for path in (intent_path, receipt_path)):
        raise ODEBFContractError("P1R28 submission namespace exists")
    log_root = REPO_ROOT / f"local/odebf/logs/p1r28-{phase}-{source_head[:12]}-{attempt}"
    log_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    intent = {
        "schema": "ode-edit-s05-p1r28-submission-intent/v1",
        "source_head": source_head, "phase": phase, "attempt": attempt,
        "result_parent": str(attempt_result_parent),
        "dry_plan": plan,
        "active_devbox_ode_gpu_jobs": active,
        "active_job_rows_sha256": hashlib.sha256("\n".join(active_rows).encode()).hexdigest(),
        "project_gpu_cap": PROJECT_GPU_CAP, "actual_task_gpu_limit": TASK_GPU_LIMIT,
        "array": "0-1%2", "held_then_atomic_release": True, "callback_job_count": 0,
    }
    intent_sha = _write_once(intent_path, intent)
    submitted = _run([
        "sbatch", "--hold", "--parsable", "--array", "0-1%2",
        "--chdir", str(REPO_ROOT), "--nodelist", "devbox",
        "--job-name", f"odeedit_s05_p1r28_{phase}",
        "--output", str(log_root / "%A_%a.out"), "--error", str(log_root / "%A_%a.err"),
        str(SBATCH), source_head, str(attempt_result_parent), phase,
    ])
    job_id = submitted.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("P1R28 scheduler ID differs")
    observed = _run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
    required = ("JobState=PENDING", "Reason=JobHeldUser", "ReqNodeList=devbox", "TRES=cpu=8,mem=65000M,node=1,billing=8,gres/gpu=1")
    if not all(item in observed for item in required):
        _run(["scancel", job_id], check=False)
        raise ODEBFContractError("P1R28 held scheduler contract differs")
    receipt = {
        "schema": "ode-edit-s05-p1r28-submission/v1", "source_head": source_head,
        "phase": phase, "attempt": attempt, "job_id": job_id,
        "result_parent": str(attempt_result_parent),
        "array": "0-1%2", "job_count": 2,
        "trajectory_count": plan["trajectory_count"],
        "corrected_trajectory_count": plan["corrected_trajectory_count"],
        "anchor_trajectory_count": plan["anchor_trajectory_count"],
        "max_concurrent_gpu": 2, "project_gpu_cap": 4,
        "intent_sha256": intent_sha,
        "held_inspection_sha256": hashlib.sha256(observed.encode()).hexdigest(),
        "held_then_atomic_release": True, "callback_job_count": 0,
    }
    receipt_sha = _write_once(receipt_path, receipt)
    _run(["scontrol", "release", job_id])
    return {**receipt, "submission_receipt_sha256": receipt_sha}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--phase", required=True, choices=("smoke", "production"))
    parser.add_argument(
        "--attempt",
        default="v1",
        choices=("v1", "tech-r1", "tech-r2", "tech-r3", "tech-r4"),
    )
    args = parser.parse_args()
    print(json.dumps(submit(args.source_head, args.phase, attempt=args.attempt), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
