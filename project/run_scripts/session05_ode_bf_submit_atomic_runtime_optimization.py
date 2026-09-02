#!/usr/bin/env python3
"""Held-inspect-release submitter for the two P1R22 model jobs."""

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

from project.run_scripts import session05_ode_bf_atomic_runtime_optimization_dry_plan as dry
from project.run_scripts.ode_bf.contracts import ODEBFContractError


SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_atomic_runtime_optimization.sbatch"
STATE_ROOT = REPO_ROOT / "local/odebf/state"
LOG_ROOT = REPO_ROOT / "local/odebf/logs/p1r22-atomic-runtime-optimization"
RESULT_PARENT = REPO_ROOT / "local/odebf/results"
NAMESPACE = "s05-atomic-runtime-optimization-p1r22-sh2-array-v1"


def _run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _write_once(path: Path, value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _active_user_gpu_jobs() -> tuple[str, ...]:
    observed = _run(
        ["squeue", "-h", "-u", os.environ.get("USER", "janghj"), "-t", "R", "-o", "%i|%b"]
    ).stdout.splitlines()
    return tuple(line for line in observed if "gpu" in line.lower())


def submit(source_head: str) -> dict[str, object]:
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    if (
        source_head != head
        or branch != "codex/odeeditsh2-s05-atomic-runtime-optimization-p1r22-v1"
        or dirty
    ):
        raise ODEBFContractError("P1R22 execution source differs")
    plan = dry.build_plan(source_head)
    active = _active_user_gpu_jobs()
    if len(active) + 2 > 4:
        raise ODEBFContractError("P1R22 server2 janghj running GPU cap differs")
    if any((RESULT_PARENT / str(job["result_name"])).exists() for job in plan["jobs"]):
        raise ODEBFContractError("P1R22 result namespace exists")
    intent_path = STATE_ROOT / f"{NAMESPACE}.intent.json"
    receipt_path = STATE_ROOT / f"{NAMESPACE}.submission-receipt.json"
    if any(path.exists() or path.is_symlink() for path in (intent_path, receipt_path)):
        raise ODEBFContractError("P1R22 submission namespace exists")
    LOG_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    intent = {
        "schema": "ode-edit-s05-atomic-runtime-optimization-p1r22-intent/v1",
        "source_head": source_head,
        "dry_plan": plan,
        "server2_janghj_running_gpu_job_count": len(active),
        "server2_janghj_gpu_cap": 4,
        "held_then_atomic_release": True,
        "array": "0-1%2",
    }
    intent_sha = _write_once(intent_path, intent)
    submitted = _run(
        [
            "sbatch", "--hold", "--parsable", "--chdir", str(REPO_ROOT),
            "--job-name", "odeedit_s05_p1r22_atomic_runtime", "--output",
            str(LOG_ROOT / "%A_%a.out"), "--error", str(LOG_ROOT / "%A_%a.err"),
            str(SBATCH), source_head, str(RESULT_PARENT),
        ]
    )
    job_id = submitted.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("P1R22 scheduler ID differs")
    observed = _run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
    required = (
        "JobState=PENDING", "Reason=JobHeldUser", "ArrayTaskThrottle=2",
        "TRES=cpu=8,mem=60416M,node=1,billing=8,gres/gpu=1",
    )
    if not all(item in observed for item in required):
        raise ODEBFContractError("P1R22 held scheduler contract differs")
    receipt = {
        "schema": "ode-edit-s05-atomic-runtime-optimization-p1r22-submission/v1",
        "source_head": source_head,
        "job_id": job_id,
        "array": "0-1%2",
        "model_job_count": 2,
        "trajectory_count": 4,
        "max_concurrent_gpu": 2,
        "intent_sha256": intent_sha,
        "held_inspection_sha256": hashlib.sha256(observed.encode()).hexdigest(),
        "held_then_atomic_release": True,
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
