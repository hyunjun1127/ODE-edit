#!/usr/bin/env python3
"""Held-inspect-release submitter for joint-P/C pilot and production."""

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

from project.run_scripts import session05_ode_bf_p1r52_joint_pc_b100_dry_plan as dry
from project.run_scripts.ode_bf.contracts import ODEBFContractError


SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_p1r52_joint_pc_b100.sbatch"
RESULT_PARENT = REPO_ROOT / "local/odebf/results"
BRANCH = "codex/p1r52-joint-pc-c1-c2-independent-b100-v1"
PROJECT_GPU_CAP = 3


def _run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=REPO_ROOT, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _write_once(path: Path, value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _gpu_jobs() -> tuple[int, list[dict[str, object]]]:
    lines = _run(["squeue", "-h", "-u", "janghj", "-w", "devbox", "-t", "RUNNING,CONFIGURING,PENDING", "-o", "%i|%T|%r|%b"]).stdout.splitlines()
    allocated = 0
    jobs: list[dict[str, object]] = []
    for line in lines:
        job_id, state, reason, gres = (line.split("|", 3) + [""] * 4)[:4]
        count = sum(int(value) for value in re.findall(r"(?:gres/)?gpu(?::[^,:=|]+)*[:=](\d+)", gres))
        if count and state in {"RUNNING", "CONFIGURING"}:
            allocated += count
        if count:
            jobs.append({"job_id": job_id, "state": state, "reason": reason, "gres": gres, "gpu": count})
    return allocated, jobs


def _available_memory_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise ODEBFContractError("host memory availability is absent")


def submit(source_head: str, *, stage: str) -> dict[str, object]:
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    if source_head != head or branch != BRANCH or dirty:
        raise ODEBFContractError("joint P/C execution source differs")
    plan = dry.build_plan(source_head, stage=stage)
    if any((RESULT_PARENT / str(job["result_name"])).exists() for job in plan["jobs"]):
        raise ODEBFContractError("joint P/C result namespace exists")
    stage_max = 1 if stage.startswith("pilot") else 3
    allocated, gpu_jobs = _gpu_jobs()
    if allocated + stage_max > PROJECT_GPU_CAP:
        raise ODEBFContractError("joint P/C GPU cap differs")
    memory_mib = _available_memory_mib()
    if memory_mib < stage_max * 65000:
        raise ODEBFContractError("joint P/C host memory is below request")
    state_root = REPO_ROOT / f"local/odebf/state/p1r52-joint-pc-c1-c2-{stage}-v1"
    log_root = REPO_ROOT / f"local/odebf/logs/p1r52-joint-pc-c1-c2-{stage}-v1"
    namespace = f"s05-p1r52-joint-pc-c1-c2-{stage}-{source_head[:12]}-v1"
    intent_path = state_root / f"{namespace}.intent.json"
    receipt_path = state_root / f"{namespace}.submission-receipt.json"
    if any(path.exists() or path.is_symlink() for path in (intent_path, receipt_path)):
        raise ODEBFContractError("joint P/C submission namespace exists")
    log_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    intent = {
        "schema": "ode-edit-s05-p1r52-joint-pc-c1-c2-submission-intent/v1",
        "source_head": source_head,
        "branch": branch,
        "stage": stage,
        "dry_plan": plan,
        "active_gpu_allocations": allocated,
        "active_or_pending_gpu_jobs": gpu_jobs,
        "host_memory_available_mib": memory_mib,
        "new_max_concurrent_gpu": stage_max,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "held_inspect_release": True,
    }
    intent_sha = _write_once(intent_path, intent)
    command = [
        "sbatch", "--hold", "--parsable", "--chdir", str(REPO_ROOT),
        "--nodelist", "devbox", "--job-name", f"odeedit_s05_p1r52_joint_pc_{stage}",
        "--output", str(log_root / "%A_%a.out" if stage == "production" else log_root / "%j.out"),
        "--error", str(log_root / "%A_%a.err" if stage == "production" else log_root / "%j.err"),
    ]
    if stage == "production":
        command.extend(["--array", "0-9%3"])
    command.extend([str(SBATCH), source_head, str(RESULT_PARENT), stage])
    job_id = _run(command).stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("joint P/C scheduler ID differs")
    observed = _run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
    required = ("JobState=PENDING", "Reason=JobHeldUser", "ReqNodeList=devbox", "TRES=cpu=8,mem=65000M,node=1,billing=8,gres/gpu=1")
    if not all(item in observed for item in required):
        _run(["scancel", job_id], check=False)
        raise ODEBFContractError("joint P/C held scheduler contract differs")
    receipt = {
        "schema": "ode-edit-s05-p1r52-joint-pc-c1-c2-submission/v1",
        "source_head": source_head,
        "branch": branch,
        "stage": stage,
        "job_id": job_id,
        "array": None if stage == "pilot" else "0-9%3",
        "job_count": len(plan["jobs"]),
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
    parser.add_argument("--stage", choices=("pilot", "pilot-tech-r1", "production"), required=True)
    args = parser.parse_args()
    print(json.dumps(submit(args.source_head, stage=args.stage), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
