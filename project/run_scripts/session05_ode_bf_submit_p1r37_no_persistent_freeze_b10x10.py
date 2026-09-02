#!/usr/bin/env python3
"""Held-inspect-release submitter for the P1R37 eight-cell ablation."""

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

from project.run_scripts import (
    session05_ode_bf_p1r37_no_persistent_freeze_b10x10_dry_plan as dry,
)
from project.run_scripts.ode_bf.contracts import ODEBFContractError


SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_p1r37_no_persistent_freeze_b10x10.sbatch"
STATE_ROOT = REPO_ROOT / "local/odebf/state/p1r37-no-persistent-freeze-b10x10"
RESULT_PARENT = REPO_ROOT / "local/odebf/results"
LOG_ROOT = REPO_ROOT / "local/odebf/logs/p1r37-no-persistent-freeze-b10x10"
BRANCH = "codex/p1r37-p1r36-no-persistent-freeze-independent-b10x10-v1"
SERVER2_PROJECT_GPU_CAP = 4
ARRAY_MAX_CONCURRENT_GPU = 4
P1R36_JOB_ID = "19472"
EXPECTED_SESSION = "019fe491-954b-70a0-8ba8-0588e9f8d741"
SESSION_BOUNDARY = REPO_ROOT / "servers/local/session-boundary.env"


def _run(
    args: list[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
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


def _gpu_allocation_count(lines: list[str]) -> int:
    return sum(1 for line in lines if "gpu" in line.casefold())


def _server2_active_gpu_allocations() -> tuple[int, list[str]]:
    lines = _run(
        [
            "squeue",
            "-h",
            "-u",
            "janghj",
            "-w",
            "server2",
            "-t",
            "RUNNING,CONFIGURING",
            "-o",
            "%i|%T|%b",
        ]
    ).stdout.splitlines()
    return _gpu_allocation_count(lines), lines


def _p1r36_active_gpu_allocations() -> tuple[int, list[str]]:
    lines = _run(
        [
            "squeue",
            "-h",
            "-j",
            P1R36_JOB_ID,
            "-t",
            "RUNNING,CONFIGURING",
            "-o",
            "%i|%T|%b",
        ],
        check=False,
    ).stdout.splitlines()
    return _gpu_allocation_count(lines), lines


def _session_boundary_gate() -> str:
    if not SESSION_BOUNDARY.is_file() or SESSION_BOUNDARY.is_symlink():
        raise ODEBFContractError("P1R37 local session boundary is absent")
    mode = SESSION_BOUNDARY.stat().st_mode & 0o777
    if mode != 0o600:
        raise ODEBFContractError("P1R37 local session boundary mode differs")
    checked = _run(["scripts/check-session-boundary.sh", EXPECTED_SESSION])
    if not checked.stdout.startswith("PASS "):
        raise ODEBFContractError("P1R37 local session boundary differs")
    return hashlib.sha256(SESSION_BOUNDARY.read_bytes()).hexdigest()


def submit(source_head: str) -> dict[str, object]:
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    parent = _run(["git", "rev-parse", "HEAD^"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(["git", "status", "--porcelain"]).stdout
    if source_head != head or branch != BRANCH or dirty:
        raise ODEBFContractError("P1R37 execution source differs")
    session_boundary_sha256 = _session_boundary_gate()
    plan = dry.build_plan(source_head)
    if any(
        (RESULT_PARENT / str(job["result_name"])).exists()
        for job in plan["jobs"]
    ):
        raise ODEBFContractError("P1R37 result namespace exists")
    p1r36_active, p1r36_lines = _p1r36_active_gpu_allocations()
    if p1r36_active > 2:
        raise ODEBFContractError("P1R37 P1R36 release allocation hold remains")
    server2_active, server2_lines = _server2_active_gpu_allocations()
    if server2_active + ARRAY_MAX_CONCURRENT_GPU > SERVER2_PROJECT_GPU_CAP:
        raise ODEBFContractError("P1R37 server2 project GPU cap differs")
    namespace = f"s05-p1r37-no-persistent-freeze-{source_head[:12]}-v1"
    intent_path = STATE_ROOT / f"{namespace}.intent.json"
    receipt_path = STATE_ROOT / f"{namespace}.submission-receipt.json"
    if any(path.exists() or path.is_symlink() for path in (intent_path, receipt_path)):
        raise ODEBFContractError("P1R37 submission namespace exists")
    LOG_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    intent = {
        "schema": "ode-edit-s05-p1r37-no-persistent-freeze-intent/v1",
        "source_head": source_head,
        "source_parent": parent,
        "dry_plan": plan,
        "p1r36_job_id": P1R36_JOB_ID,
        "p1r36_active_gpu_allocations": p1r36_active,
        "p1r36_scheduler_fact_sha256": hashlib.sha256(
            "\n".join(p1r36_lines).encode()
        ).hexdigest(),
        "server2_active_gpu_allocations": server2_active,
        "server2_scheduler_fact_sha256": hashlib.sha256(
            "\n".join(server2_lines).encode()
        ).hexdigest(),
        "new_max_concurrent_gpu": ARRAY_MAX_CONCURRENT_GPU,
        "server2_project_gpu_cap": SERVER2_PROJECT_GPU_CAP,
        "session_boundary_sha256": session_boundary_sha256,
        "expected_session": EXPECTED_SESSION,
        "held_then_atomic_release": True,
        "array": "0-7%4",
    }
    intent_sha = _write_once(intent_path, intent)
    submitted = _run(
        [
            "sbatch",
            "--hold",
            "--parsable",
            "--array",
            "0-7%4",
            "--chdir",
            str(REPO_ROOT),
            "--nodelist",
            "server2",
            "--job-name",
            "odeedit_s05_p1r37_no_persistent_freeze_b10x10",
            "--output",
            str(LOG_ROOT / "%A_%a.out"),
            "--error",
            str(LOG_ROOT / "%A_%a.err"),
            str(SBATCH),
            source_head,
            str(RESULT_PARENT),
        ]
    )
    job_id = submitted.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("P1R37 scheduler ID differs")
    observed = _run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "ReqNodeList=server2",
        "TRES=cpu=8,mem=60416M,node=1,billing=8,gres/gpu=1",
        "ArrayTaskThrottle=4",
    )
    if not all(item in observed for item in required):
        _run(["scancel", job_id], check=False)
        raise ODEBFContractError("P1R37 held scheduler contract differs")
    receipt = {
        "schema": "ode-edit-s05-p1r37-no-persistent-freeze-submission/v1",
        "source_head": source_head,
        "source_parent": parent,
        "job_id": job_id,
        "array": "0-7%4",
        "job_count": 8,
        "case_count": 80,
        "max_concurrent_gpu": ARRAY_MAX_CONCURRENT_GPU,
        "server2_project_gpu_cap": SERVER2_PROJECT_GPU_CAP,
        "p1r36_active_gpu_allocations_at_release": p1r36_active,
        "session_boundary_sha256": session_boundary_sha256,
        "expected_session": EXPECTED_SESSION,
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
