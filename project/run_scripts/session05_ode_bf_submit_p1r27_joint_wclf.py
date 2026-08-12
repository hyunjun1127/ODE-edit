#!/usr/bin/env python3
"""Held-inspect-atomic-release submitter for P1R27 arrays."""

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

from project.run_scripts import session05_ode_bf_p1r27_joint_wclf_dry_plan as dry
from project.run_scripts.ode_bf.contracts import ODEBFContractError


SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_p1r27_joint_wclf.sbatch"
STATE_ROOT = REPO_ROOT / "local/odebf/state/p1r27-joint-wclf"
RESULT_PARENT = REPO_ROOT / "local/odebf/results"
BRANCH = "codex/p1r27-joint-wclf-v1"
PROJECT_GPU_CAP = 4


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


def _active_gpu_jobs() -> tuple[int, list[str]]:
    lines = _run(["squeue", "-h", "-u", "janghj", "-w", "devbox", "-t", "RUNNING", "-o", "%i|%j|%b"]).stdout.splitlines()
    active = [line for line in lines if "gpu" in line.casefold() and "ode" in line.casefold()]
    return len(active), active


def submit(source_head: str, phase: str, *, attempt: str = "original") -> dict[str, object]:
    if attempt not in ("original", "tech-r1"):
        raise ODEBFContractError("P1R27 submission attempt differs")
    if (
        source_head != _run(["git", "rev-parse", "HEAD"]).stdout.strip()
        or _run(["git", "branch", "--show-current"]).stdout.strip() != BRANCH
        or _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    ):
        raise ODEBFContractError("P1R27 execution source differs")
    plan = dry.build_plan(source_head, phase)
    if any((RESULT_PARENT / str(job["result_name"])).exists() for job in plan["jobs"]):
        raise ODEBFContractError("P1R27 result namespace exists")
    active, active_rows = _active_gpu_jobs()
    new = int(plan["array_max_concurrent_gpu"])
    if active + new > PROJECT_GPU_CAP:
        raise ODEBFContractError("P1R27 server1 project GPU cap differs")
    namespace = f"s05-p1r27-{phase}-{attempt}-{source_head[:12]}-v1"
    intent_path = STATE_ROOT / f"{namespace}.intent.json"
    receipt_path = STATE_ROOT / f"{namespace}.submission-receipt.json"
    if any(path.exists() or path.is_symlink() for path in (intent_path, receipt_path)):
        raise ODEBFContractError("P1R27 submission namespace exists")
    log_root = REPO_ROOT / f"local/odebf/logs/p1r27-{phase}-{attempt}-{source_head[:12]}"
    log_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    array = "0-3%4"
    intent = {
        "schema": "ode-edit-s05-p1r27-submission-intent/v1",
        "source_head": source_head,
        "phase": phase,
        "attempt": attempt,
        "dry_plan": plan,
        "active_devbox_ode_gpu_jobs": active,
        "active_job_rows_sha256": hashlib.sha256("\n".join(active_rows).encode()).hexdigest(),
        "project_gpu_cap": PROJECT_GPU_CAP,
        "held_then_atomic_release": True,
        "array": array,
        "callback_job_count": 0,
    }
    intent_sha = _write_once(intent_path, intent)
    submitted = _run([
        "sbatch", "--hold", "--parsable", "--array", array,
        "--chdir", str(REPO_ROOT), "--nodelist", "devbox",
        "--job-name", f"odeedit_s05_p1r27_{phase}",
        "--output", str(log_root / "%A_%a.out"),
        "--error", str(log_root / "%A_%a.err"),
        str(SBATCH), source_head, str(RESULT_PARENT), phase,
    ])
    job_id = submitted.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("P1R27 scheduler ID differs")
    observed = _run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
    required = ("JobState=PENDING", "Reason=JobHeldUser", "ReqNodeList=devbox", "TRES=cpu=8,mem=65000M,node=1,billing=8,gres/gpu=1")
    if not all(item in observed for item in required):
        _run(["scancel", job_id], check=False)
        raise ODEBFContractError("P1R27 held scheduler contract differs")
    receipt = {
        "schema": "ode-edit-s05-p1r27-submission/v1",
        "source_head": source_head,
        "phase": phase,
        "attempt": attempt,
        "job_id": job_id,
        "array": array,
        "job_count": 4,
        "trajectory_count": 8,
        "max_concurrent_gpu": 4,
        "project_gpu_cap": 4,
        "intent_sha256": intent_sha,
        "held_inspection_sha256": hashlib.sha256(observed.encode()).hexdigest(),
        "held_then_atomic_release": True,
        "callback_job_count": 0,
    }
    receipt_sha = _write_once(receipt_path, receipt)
    _run(["scontrol", "release", job_id])
    return {**receipt, "submission_receipt_sha256": receipt_sha}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--phase", required=True, choices=("smoke", "production"))
    parser.add_argument("--attempt", choices=("original", "tech-r1"), default="original")
    args = parser.parse_args()
    print(json.dumps(submit(args.source_head, args.phase, attempt=args.attempt), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
