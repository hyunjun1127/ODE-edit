#!/usr/bin/env python3
"""Held-inspect-release submitter for four Compute-A1 paired jobs."""

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

from project.run_scripts import session05_ode_bf_compute_progress_simplex_dry_plan as dry
from project.run_scripts.ode_bf.contracts import ODEBFContractError


SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_compute_progress_simplex.sbatch"
STATE_ROOT = REPO_ROOT / "local/odebf/state"
RESULT_PARENT = REPO_ROOT / "local/odebf/results"
BRANCH = "codex/p1r23-progress-simplex-compute-a1"


def _run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=REPO_ROOT, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _write_once(path: Path, value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _active_project_gpu() -> int:
    lines = _run(["squeue", "-h", "-u", "janghj", "-t", "RUNNING,PENDING", "-o", "%i|%T|%b"]).stdout.splitlines()
    return sum(1 for line in lines if "gpu" in line.lower())


def submit(source_head: str) -> dict[str, object]:
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    if source_head != head or branch != BRANCH or dirty:
        raise ODEBFContractError("Compute-A1 execution source differs")
    plan = dry.build_plan(source_head)
    result_paths = [RESULT_PARENT / str(job["result_name"]) for job in plan["jobs"]]
    if any(path.exists() or path.is_symlink() for path in result_paths):
        raise ODEBFContractError("Compute-A1 result namespace exists")
    active = _active_project_gpu()
    if active + 4 > 4:
        raise ODEBFContractError("Compute-A1 project GPU cap differs")
    namespace = f"s05-p1r23-compute-a1-{source_head[:12]}-v1"
    intent_path = STATE_ROOT / f"{namespace}.intent.json"
    receipt_path = STATE_ROOT / f"{namespace}.submission-receipt.json"
    if any(path.exists() or path.is_symlink() for path in (intent_path, receipt_path)):
        raise ODEBFContractError("Compute-A1 submission namespace exists")
    log_root = REPO_ROOT / f"local/odebf/logs/{namespace}"
    log_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    intent = {
        "schema": "ode-edit-s05-p1r23-compute-a1-intent/v1",
        "source_head": source_head,
        "dry_plan": plan,
        "active_project_gpu": active,
        "project_gpu_cap": 4,
        "array": "0-3%4",
        "held_then_atomic_release": True,
    }
    intent_sha = _write_once(intent_path, intent)
    submitted = _run([
        "sbatch", "--hold", "--parsable", "--array", "0-3%4",
        "--chdir", str(REPO_ROOT), "--nodelist", "devbox",
        "--job-name", "odeedit_s05_p1r23_compute_a1",
        "--output", str(log_root / "%A_%a.out"),
        "--error", str(log_root / "%A_%a.err"),
        str(SBATCH), source_head, str(RESULT_PARENT),
    ])
    job_id = submitted.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("Compute-A1 scheduler ID differs")
    observed = _run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
    required = ("JobState=PENDING", "Reason=JobHeldUser", "ReqNodeList=devbox", "ArrayTaskThrottle=4", "gres/gpu=1", "mem=65000M")
    if not all(item in observed for item in required):
        _run(["scancel", job_id], check=False)
        raise ODEBFContractError("Compute-A1 held scheduler contract differs")
    receipt = {
        "schema": "ode-edit-s05-p1r23-compute-a1-submission/v1",
        "source_head": source_head,
        "job_id": job_id,
        "array": "0-3%4",
        "job_count": 4,
        "trajectory_count": 8,
        "max_concurrent_gpu": 4,
        "intent_sha256": intent_sha,
        "held_inspection_sha256": hashlib.sha256(observed.encode()).hexdigest(),
        "result_names": [str(job["result_name"]) for job in plan["jobs"]],
    }
    receipt_sha = _write_once(receipt_path, receipt)
    _run(["scontrol", "release", job_id])
    return {**receipt, "submission_receipt_sha256": receipt_sha}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    args = parser.parse_args()
    print(json.dumps(submit(args.source_head), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
