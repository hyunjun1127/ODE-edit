#!/usr/bin/env python3
"""Held-inspect-release submitter for the P1R52 IL5 B1000 cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1r52_target_depth_sequential_il5 import (
    JOB_NAME,
    RESULT_NAME,
)
from project.run_scripts import (
    session05_ode_bf_p1r52_target_depth_il5_sequential_b100x10_dry_plan as dry,
)


SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_p1r52_target_depth_il5_sequential_b100x10.sbatch"
BRANCH = "codex/p1r52-llama-j0-il5-sequential-10xb100-v1"
PROJECT_GPU_CAP = 3


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


def _gpu_jobs() -> tuple[int, list[dict[str, object]]]:
    lines = _run(
        [
            "squeue", "-h", "-u", "janghj", "-w", "server2",
            "-t", "RUNNING,CONFIGURING,PENDING", "-o", "%i|%T|%r|%b",
        ]
    ).stdout.splitlines()
    allocated = 0
    jobs: list[dict[str, object]] = []
    for line in lines:
        job_id, state, reason, gres = (line.split("|", 3) + [""] * 4)[:4]
        gpu = sum(
            int(value)
            for value in re.findall(r"(?:gres/)?gpu(?::[^,:=|]+)*[:=](\d+)", gres)
        )
        if gpu and state in {"RUNNING", "CONFIGURING"}:
            allocated += gpu
        if gpu:
            jobs.append(
                {
                    "job_id": job_id,
                    "state": state,
                    "reason": reason,
                    "gres": gres,
                    "gpu": gpu,
                }
            )
    return allocated, jobs


def _host_memory_available_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise ODEBFContractError("host memory availability is absent")


def submit(source_head: str) -> dict[str, object]:
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    if source_head != head or branch != BRANCH or dirty:
        raise ODEBFContractError("P1R52 IL5 execution source differs")
    output_root = REPO_ROOT / "local/odebf/results" / RESULT_NAME
    state_root = REPO_ROOT / (
        "local/odebf/state/"
        "p1r52-il5-sequential-10xb100-postenergy-warn-r1-tech-r2-v1"
    )
    log_root = REPO_ROOT / (
        "local/odebf/logs/"
        "p1r52-il5-sequential-10xb100-postenergy-warn-r1-tech-r2-v1"
    )
    if output_root.exists() or output_root.is_symlink():
        raise ODEBFContractError("P1R52 IL5 result namespace exists")
    allocated, jobs = _gpu_jobs()
    if allocated + 1 > PROJECT_GPU_CAP:
        raise ODEBFContractError("P1R52 IL5 project GPU cap differs")
    memory_mib = _host_memory_available_mib()
    if memory_mib < 60416:
        raise ODEBFContractError("P1R52 IL5 host memory is below request")
    plan = dry.build_plan(source_head)
    namespace = (
        f"p1r52-il5-sequential-postenergy-warn-r1-tech-r2-"
        f"{source_head[:12]}-v1"
    )
    intent_path = state_root / f"{namespace}.intent.json"
    receipt_path = state_root / f"{namespace}.submission-receipt.json"
    if any(path.exists() or path.is_symlink() for path in (intent_path, receipt_path)):
        raise ODEBFContractError("P1R52 IL5 submission namespace exists")
    log_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    intent = {
        "schema": "ode-edit-s05-p1r52-il5-sequential-submission-intent/v1",
        "source_head": source_head,
        "branch": branch,
        "dry_plan": plan,
        "active_gpu_allocations": allocated,
        "active_or_pending_gpu_jobs": jobs,
        "host_memory_available_mib": memory_mib,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "held_then_release": True,
    }
    intent_sha = _write_once(intent_path, intent)
    submitted = _run(
        [
            "sbatch", "--hold", "--parsable", "--chdir", str(REPO_ROOT),
            "--nodelist", "server2", "--job-name", JOB_NAME,
            "--output", str(log_root / "%j.out"),
            "--error", str(log_root / "%j.err"),
            str(SBATCH), source_head, str(output_root),
        ]
    )
    job_id = submitted.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("P1R52 IL5 scheduler ID differs")
    observed = _run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "ReqNodeList=server2",
        "TRES=cpu=8,mem=60416M,node=1,billing=8,gres/gpu=1",
    )
    if not all(item in observed for item in required):
        _run(["scancel", job_id], check=False)
        raise ODEBFContractError("P1R52 IL5 held scheduler contract differs")
    receipt = {
        "schema": "ode-edit-s05-p1r52-il5-sequential-submission/v1",
        "source_head": source_head,
        "branch": branch,
        "job_id": job_id,
        "request_count": 1000,
        "scientific_amendment": "POSTSOLVE_ENERGY_WARN_RECORD_ONLY",
        "postsolve_energy_warn_enabled": True,
        "postsolve_energy_decision_influence_count": 0,
        "postsolve_energy_tolerance": 1e-12,
        "technical_attempt": "TECH_R2_HELDOUT_LITERAL_BRACE_LOOKUP",
        "superseded_scheduler_job_id": "22170",
        "result_root": str(output_root),
        "active_gpu_allocations_before_release": allocated,
        "host_memory_available_mib": memory_mib,
        "intent_sha256": intent_sha,
        "held_inspection_sha256": hashlib.sha256(observed.encode()).hexdigest(),
        "job_map": plan["jobs"],
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
