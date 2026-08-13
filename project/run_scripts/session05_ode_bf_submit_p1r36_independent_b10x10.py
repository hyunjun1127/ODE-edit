#!/usr/bin/env python3
"""Held-inspect-release submitter for P1R36 detailed full matrix."""

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

from project.run_scripts import session05_ode_bf_p1r36_independent_b10x10_dry_plan as dry
from project.run_scripts.ode_bf.contracts import ODEBFContractError


SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_p1r36_independent_b10x10.sbatch"
STATE_ROOT = REPO_ROOT / "local/odebf/state/p1r36-p1r35-independent-b10x10"
RESULT_PARENT = REPO_ROOT / "local/odebf/results"
LOG_ROOT = REPO_ROOT / "local/odebf/logs/p1r36-p1r35-independent-b10x10"
BRANCH = "codex/p1r36-p1r35-independent-b10x10-full-matrix-v1"
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


def _active_gpu_jobs() -> int:
    lines = _run(["squeue", "-h", "-u", "janghj", "-w", "devbox", "-t", "RUNNING", "-o", "%i|%b"]).stdout.splitlines()
    return sum(1 for line in lines if "gpu" in line.casefold())


def submit(source_head: str) -> dict[str, object]:
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    parent = _run(["git", "rev-parse", "HEAD^"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    if source_head != head or branch != BRANCH or dirty:
        raise ODEBFContractError("P1R35 independent execution source differs")
    plan = dry.build_plan(source_head)
    if any((RESULT_PARENT / str(job["result_name"])).exists() for job in plan["jobs"]):
        raise ODEBFContractError("P1R35 independent result namespace exists")
    active = _active_gpu_jobs()
    new = int(plan["array_max_concurrent_gpu"])
    if active + new > PROJECT_GPU_CAP:
        raise ODEBFContractError("P1R35 independent server1 project GPU cap differs")
    namespace = f"s05-p1r36-p1r35-independent-b10x10-{source_head[:12]}-v1"
    intent_path = STATE_ROOT / f"{namespace}.intent.json"
    receipt_path = STATE_ROOT / f"{namespace}.submission-receipt.json"
    if any(path.exists() or path.is_symlink() for path in (intent_path, receipt_path)):
        raise ODEBFContractError("P1R35 independent submission namespace exists")
    LOG_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    intent = {
        "schema": "ode-edit-s05-p1r36-p1r35-independent-b10x10-detailed-intent/v1",
        "source_head": source_head,
        "source_parent": parent,
        "dry_plan": plan,
        "active_devbox_gpu_jobs": active,
        "new_max_concurrent_gpu": new,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "held_then_atomic_release": True,
        "array": "0-7%4",
    }
    intent_sha = _write_once(intent_path, intent)
    submitted = _run([
        "sbatch", "--hold", "--parsable", "--array", "0-7%4",
        "--chdir", str(REPO_ROOT), "--nodelist", "devbox",
        "--job-name", "odeedit_s05_p1r36_p1r35_independent_b10x10",
        "--output", str(LOG_ROOT / "%A_%a.out"),
        "--error", str(LOG_ROOT / "%A_%a.err"),
        str(SBATCH), source_head, str(RESULT_PARENT),
    ])
    job_id = submitted.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("P1R35 independent scheduler ID differs")
    observed = _run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "ReqNodeList=devbox",
        "TRES=cpu=8,mem=65000M,node=1,billing=8,gres/gpu=1",
    )
    if not all(item in observed for item in required):
        _run(["scancel", job_id], check=False)
        raise ODEBFContractError("P1R35 independent held scheduler contract differs")
    receipt = {
        "schema": "ode-edit-s05-p1r36-p1r35-independent-b10x10-detailed-submission/v1",
        "source_head": source_head,
        "source_parent": parent,
        "job_id": job_id,
        "array": "0-7%4",
        "job_count": 8,
        "case_count": 80,
        "max_concurrent_gpu": new,
        "project_gpu_cap": PROJECT_GPU_CAP,
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
