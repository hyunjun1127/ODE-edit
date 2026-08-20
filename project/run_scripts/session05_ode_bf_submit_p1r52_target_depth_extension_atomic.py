#!/usr/bin/env python3
"""Held-inspect-release submitter for the three-cell target-depth extension."""

from __future__ import annotations

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
from project.run_scripts.ode_bf.p1r52_target_depth_extension import PROJECT_GPU_CAP
from project.run_scripts.ode_bf.p1r52_target_depth_extension_panel import (
    verify_immutable_references,
)
from project.run_scripts.session05_ode_bf_p1r52_target_depth_extension_atomic_dry_plan import (
    build_plan,
)


SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_p1r52_target_depth_extension_atomic.sbatch"
STATE_ROOT = REPO_ROOT / "local/odebf/state/p1r52-target-depth-inner-telemetry-r1-v1"
RESULT_PARENT = REPO_ROOT / "local/odebf/results"
LOG_ROOT = REPO_ROOT / "local/odebf/logs/p1r52-target-depth-inner-telemetry-r1-v1"
BRANCH = "codex/p1r52-target-depth-il8-il10-il15-v1"
MEMORY_MIB_PER_CELL = 65000


def run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def write_once(path: Path, value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def active_gpus() -> tuple[int, list[dict[str, object]]]:
    lines = run(
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
            "%i|%T|%b|%j",
        ]
    ).stdout.splitlines()
    total = 0
    jobs = []
    for line in lines:
        job, state, gres, name = line.split("|", 3)
        count = sum(
            int(value)
            for value in re.findall(r"(?:gres/)?gpu(?::[^,:=|]+)*[:=](\d+)", gres)
        )
        if count:
            total += count
            jobs.append(
                {
                    "job_id": job,
                    "state": state,
                    "gres": gres,
                    "name": name,
                    "gpu": count,
                }
            )
    return total, jobs


def available_memory_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise ODEBFContractError("server2 MemAvailable is not recorded")


def submit(source_head: str) -> dict[str, object]:
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    if source_head != head or branch != BRANCH or dirty:
        raise ODEBFContractError("P1R52 target-depth extension execution source differs")
    plan = build_plan(source_head)
    verify_immutable_references(REPO_ROOT)
    if any(
        (RESULT_PARENT / str(job["result_name"])).exists()
        for job in plan["new_gpu_cells"]
    ):
        raise ODEBFContractError("P1R52 target-depth extension result namespace exists")
    active, jobs = active_gpus()
    concurrency = min(3, PROJECT_GPU_CAP - active)
    memory_available = available_memory_mib()
    if (
        concurrency <= 0
        or active + concurrency > PROJECT_GPU_CAP
        or concurrency * MEMORY_MIB_PER_CELL > memory_available
    ):
        raise ODEBFContractError("P1R52 target-depth extension resource cap unavailable")
    namespace = f"s05-p1r52-target-depth-il8-il10-il15-{source_head[:12]}-v1"
    intent_path = STATE_ROOT / f"{namespace}.intent.json"
    receipt_path = STATE_ROOT / f"{namespace}.submission-receipt.json"
    if any(path.exists() or path.is_symlink() for path in (intent_path, receipt_path)):
        raise ODEBFContractError("P1R52 target-depth extension submission namespace exists")
    LOG_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    array = f"0-2%{concurrency}"
    intent = {
        "schema": "ode-edit-s05-p1r52-target-depth-extension-intent/v1",
        "source_head": source_head,
        "dry_plan": plan,
        "active_server2_gpu_allocations": active,
        "active_server2_gpu_jobs": jobs,
        "new_max_concurrent_gpu": concurrency,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "host_mem_available_mib": memory_available,
        "new_mem_request_mib": concurrency * MEMORY_MIB_PER_CELL,
        "array": array,
        "held_then_atomic_release": True,
    }
    intent_sha = write_once(intent_path, intent)
    submitted = run(
        [
            "sbatch",
            "--hold",
            "--parsable",
            "--array",
            array,
            "--chdir",
            str(REPO_ROOT),
            "--nodelist",
            "server2",
            "--job-name",
            "odeedit_s05_p1r52_target_depth_extension",
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
    observed = run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
    expected_scheduler = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "ReqNodeList=server2",
        "TRES=cpu=8,mem=65000M,node=1,billing=8,gres/gpu=1",
    )
    if not job_id.isdigit() or not all(value in observed for value in expected_scheduler):
        run(["scancel", job_id], check=False)
        raise ODEBFContractError("P1R52 target-depth extension held contract differs")
    receipt = {
        "schema": "ode-edit-s05-p1r52-target-depth-extension-submission/v1",
        "source_head": source_head,
        "job_id": job_id,
        "array": array,
        "cell_count": 3,
        "case_count": 30,
        "request_attempt_count": 300,
        "active_gpu_allocations_before_release": active,
        "new_max_concurrent_gpu": concurrency,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "host_mem_available_mib": memory_available,
        "intent_sha256": intent_sha,
        "held_inspection_sha256": hashlib.sha256(observed.encode()).hexdigest(),
        "held_then_atomic_release": True,
    }
    receipt_sha = write_once(receipt_path, receipt)
    run(["scontrol", "release", job_id])
    return {**receipt, "submission_receipt_sha256": receipt_sha}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: submitter SOURCE_HEAD")
    print(json.dumps(submit(sys.argv[1]), sort_keys=True, separators=(",", ":")))
