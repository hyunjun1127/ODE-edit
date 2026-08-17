#!/usr/bin/env python3
"""Held-inspect-release submitter for the P1R52 sequential 10xB100 panel."""

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

from project.run_scripts import session05_ode_bf_p1r52_sequential_b100x10_fourcell_dry_plan as dry
from project.run_scripts.ode_bf.contracts import ODEBFContractError


SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_p1r52_sequential_b100x10_fourcell.sbatch"
RESULT_PARENT = REPO_ROOT / "local/odebf/results"
BRANCH = "codex/p1r52-llama-seq-10xb100-fourarm-r1-v1"
PROJECT_GPU_CAP = 4


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
            "squeue",
            "-h",
            "-u",
            "janghj",
            "-w",
            "devbox",
            "-t",
            "RUNNING,CONFIGURING,PENDING",
            "-o",
            "%i|%T|%r|%b",
        ]
    ).stdout.splitlines()
    allocated = 0
    observed: list[dict[str, object]] = []
    for line in lines:
        job_id, state, reason, gres = (line.split("|", 3) + [""] * 4)[:4]
        count = sum(
            int(value)
            for value in re.findall(r"(?:gres/)?gpu(?::[^,:=|]+)*[:=](\d+)", gres)
        )
        if count:
            if state in {"RUNNING", "CONFIGURING"}:
                allocated += count
            observed.append(
                {"job_id": job_id, "state": state, "reason": reason, "gres": gres, "gpu": count}
            )
    return allocated, observed


def _host_memory_available_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise ODEBFContractError("host memory availability is absent")


def submit(source_head: str, *, attempt_suffix: str) -> dict[str, object]:
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    if source_head != head or branch != BRANCH or dirty:
        raise ODEBFContractError("P1R52 B100x10 execution source differs")
    r52_only = attempt_suffix == "tech-r2"
    if attempt_suffix not in ("tech-r1", "tech-r2"):
        raise ODEBFContractError("P1R52 B100x10 submission attempt differs")
    stage_gpu_max = 2 if r52_only else 4
    state_root = REPO_ROOT / (
        f"local/odebf/state/p1r52-llama-sequential-10xb100-fourcell-{attempt_suffix}-v1"
    )
    log_root = REPO_ROOT / (
        f"local/odebf/logs/p1r52-llama-sequential-10xb100-fourcell-{attempt_suffix}-v1"
    )
    plan = dry.build_plan(
        source_head, attempt_suffix=attempt_suffix, r52_only=r52_only
    )
    if any((RESULT_PARENT / str(job["result_name"])).exists() for job in plan["jobs"]):
        raise ODEBFContractError("P1R52 B100x10 result namespace exists")
    allocated, gpu_jobs = _gpu_jobs()
    if r52_only:
        allowed = all(
            str(item["job_id"]).startswith("20453_") and item["state"] == "RUNNING"
            for item in gpu_jobs
        )
        if not allowed or allocated != 2:
            raise ODEBFContractError("P1R52 B100x10 TECH-R2 healthy-cell queue differs")
    elif gpu_jobs:
        raise ODEBFContractError("P1R52 B100x10 requires an empty server1 project GPU queue")
    if allocated + stage_gpu_max > PROJECT_GPU_CAP:
        raise ODEBFContractError("P1R52 B100x10 project GPU cap differs")
    memory_mib = _host_memory_available_mib()
    if memory_mib < stage_gpu_max * 65000:
        raise ODEBFContractError("P1R52 B100x10 host memory is below the stage request")
    namespace = f"s05-p1r52-sequential-b100x10-{attempt_suffix}-{source_head[:12]}-v1"
    intent_path = state_root / f"{namespace}.intent.json"
    receipt_path = state_root / f"{namespace}.submission-receipt.json"
    if any(path.exists() or path.is_symlink() for path in (intent_path, receipt_path)):
        raise ODEBFContractError("P1R52 B100x10 submission namespace exists")
    log_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    intent = {
        "schema": "ode-edit-s05-p1r52-sequential-b100x10-submission-intent/v1",
        "source_head": source_head,
        "branch": branch,
        "dry_plan": plan,
        "active_server1_gpu_allocations": allocated,
        "active_or_pending_server1_gpu_jobs": gpu_jobs,
        "host_memory_available_mib": memory_mib,
        "new_max_concurrent_gpu": stage_gpu_max,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "held_then_atomic_release": True,
    }
    intent_sha = _write_once(intent_path, intent)
    command = [
        "sbatch",
        "--hold",
        "--parsable",
        "--array",
        "0-1%2" if r52_only else "0-3%4",
        "--chdir",
        str(REPO_ROOT),
        "--nodelist",
        "devbox",
        "--job-name",
        f"odeedit_s05_p1r52_seq_b100x10_{attempt_suffix.replace('-', '_')}",
        "--output",
        str(log_root / "%A_%a.out"),
        "--error",
        str(log_root / "%A_%a.err"),
        str(SBATCH),
        source_head,
        str(RESULT_PARENT),
        attempt_suffix,
    ]
    submitted = _run(command)
    job_id = submitted.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("P1R52 B100x10 scheduler ID differs")
    observed = _run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "ReqNodeList=devbox",
        "TRES=cpu=8,mem=65000M,node=1,billing=8,gres/gpu=1",
    )
    if not all(item in observed for item in required):
        _run(["scancel", job_id], check=False)
        raise ODEBFContractError("P1R52 B100x10 held scheduler contract differs")
    receipt = {
        "schema": "ode-edit-s05-p1r52-sequential-b100x10-submission/v1",
        "source_head": source_head,
        "branch": branch,
        "job_id": job_id,
        "array": "0-1%2" if r52_only else "0-3%4",
        "job_count": len(plan["jobs"]),
        "endpoint_attempt_count": len(plan["jobs"]),
        "request_count_per_cell": 1000,
        "total_edit_count": len(plan["jobs"]) * 1000,
        "attempt_suffix": attempt_suffix,
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
    parser.add_argument("--attempt-suffix", choices=("tech-r1", "tech-r2"), default="tech-r2")
    args = parser.parse_args()
    print(
        json.dumps(
            submit(args.source_head, attempt_suffix=args.attempt_suffix),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
