#!/usr/bin/env python3
"""Held-inspect-release submitter for three P1R52-PIR cells."""

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

from project.run_scripts import session05_ode_bf_p1r52_pir_atomic_b10x10_dry_plan as dry
from project.run_scripts.ode_bf.contracts import ODEBFContractError


SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_p1r52_pir_atomic_b10x10.sbatch"
STATE_ROOT = REPO_ROOT / "local/odebf/state/p1r52-pir-atomic-b10x10-v1"
RESULT_PARENT = REPO_ROOT / "local/odebf/results"
LOG_ROOT = REPO_ROOT / "local/odebf/logs/p1r52-pir-atomic-b10x10-v1"
BRANCH = "codex/p1r52-pir-atomic-b10x10-v1"
PROJECT_GPU_CAP = 4
STAGE_GPU_MAX = 3


def _run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=REPO_ROOT, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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


def _active_gpu_allocations() -> tuple[int, list[dict[str, object]]]:
    lines = _run([
        "squeue", "-h", "-u", "janghj", "-w", "devbox",
        "-t", "RUNNING,CONFIGURING", "-o", "%i|%T|%b",
    ]).stdout.splitlines()
    observed: list[dict[str, object]] = []
    total = 0
    for line in lines:
        job_id, state, gres = (line.split("|", 2) + [""])[:3]
        count = sum(int(value) for value in re.findall(r"(?:gres/)?gpu(?::[^,:=|]+)*[:=](\d+)", gres))
        if count:
            observed.append({"job_id": job_id, "state": state, "gres": gres, "gpu": count})
            total += count
    return total, observed


def _memory_snapshot() -> dict[str, int]:
    fields: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            fields[key] = int(value.strip().split()[0])
    return fields


def submit(source_head: str, *, attempt_suffix: str | None = None) -> dict[str, object]:
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    if source_head != head or branch != BRANCH or dirty:
        raise ODEBFContractError("P1R52-PIR execution source differs")
    plan = dry.build_plan(source_head, attempt_suffix=attempt_suffix)
    if any((RESULT_PARENT / str(job["result_name"])).exists() for job in plan["jobs"]):
        raise ODEBFContractError("P1R52-PIR result namespace exists")
    active, active_jobs = _active_gpu_allocations()
    if active + STAGE_GPU_MAX > PROJECT_GPU_CAP:
        raise ODEBFContractError("P1R52-PIR server1 GPU cap would differ")
    memory = _memory_snapshot()
    if memory.get("MemAvailable", 0) < 65000 * 1024 * STAGE_GPU_MAX:
        raise ODEBFContractError("P1R52-PIR host memory gate differs")
    namespace = f"s05-p1r52-pir-{source_head[:12]}-{attempt_suffix or 'primary'}-v1"
    intent_path = STATE_ROOT / f"{namespace}.intent.json"
    receipt_path = STATE_ROOT / f"{namespace}.submission-receipt.json"
    if any(path.exists() or path.is_symlink() for path in (intent_path, receipt_path)):
        raise ODEBFContractError("P1R52-PIR submission namespace exists")
    LOG_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    intent = {
        "schema": "ode-edit-s05-p1r52-pir-submission-intent/v1",
        "source_head": source_head,
        "branch": BRANCH,
        "dry_plan": plan,
        "active_server1_gpu_allocations": active,
        "active_server1_gpu_jobs": active_jobs,
        "host_memory_kib": memory,
        "new_max_concurrent_gpu": STAGE_GPU_MAX,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "held_then_atomic_release": True,
        "array": "0-2%3",
    }
    intent_sha = _write_once(intent_path, intent)
    command = [
        "sbatch", "--hold", "--parsable", "--array", "0-2%3",
        "--chdir", str(REPO_ROOT), "--nodelist", "devbox",
        "--job-name", "odeedit_s05_p1r52_pir",
        "--output", str(LOG_ROOT / "%A_%a.out"),
        "--error", str(LOG_ROOT / "%A_%a.err"),
        str(SBATCH), source_head, str(RESULT_PARENT),
    ]
    if attempt_suffix:
        command.append(attempt_suffix)
    job_id = _run(command).stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("P1R52-PIR scheduler ID differs")
    observed = _run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
    required = (
        "JobState=PENDING", "Reason=JobHeldUser", "ReqNodeList=devbox",
        "TRES=cpu=8,mem=65000M,node=1,billing=8,gres/gpu=1",
    )
    if not all(item in observed for item in required):
        _run(["scancel", job_id], check=False)
        raise ODEBFContractError("P1R52-PIR held scheduler contract differs")
    receipt = {
        "schema": "ode-edit-s05-p1r52-pir-submission/v1",
        "source_head": source_head,
        "branch": BRANCH,
        "job_id": job_id,
        "array": "0-2%3",
        "job_count": 3,
        "endpoint_attempt_count": 30,
        "active_gpu_allocations_before_release": active,
        "new_max_concurrent_gpu": STAGE_GPU_MAX,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "intent_sha256": intent_sha,
        "held_inspection_sha256": hashlib.sha256(observed.encode()).hexdigest(),
        "held_then_atomic_release": True,
        "attempt_suffix": attempt_suffix,
    }
    receipt_sha = _write_once(receipt_path, receipt)
    _run(["scontrol", "release", job_id])
    return {**receipt, "submission_receipt_sha256": receipt_sha}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--attempt-suffix")
    args = parser.parse_args()
    print(json.dumps(submit(args.source_head, attempt_suffix=args.attempt_suffix), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
