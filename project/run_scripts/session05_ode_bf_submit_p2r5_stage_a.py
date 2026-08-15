#!/usr/bin/env python3
"""Held-inspect-release submitter for the four P2R5 Stage-A case pairs."""

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

from project.run_scripts import session05_ode_bf_p2r5_stage_a_dry_plan as dry
from project.run_scripts.ode_bf.contracts import ODEBFContractError


SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_p2r5_stage_a.sbatch"
STATE_ROOT = REPO_ROOT / "local/odebf/state/p2r5-sdrt-stage-a"
RESULT_PARENT = REPO_ROOT / "local/odebf/results"
LOG_ROOT = REPO_ROOT / "local/odebf/logs/p2r5-sdrt-stage-a"
BRANCH = "codex/p2r5-semantic-deficit-constrained-residual-transport-v1"
PROJECT_GPU_CAP = 4
STAGE_GPU_MAX = 4


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


def _active_gpu_allocations() -> tuple[int, list[dict[str, object]]]:
    lines = _run([
        "squeue", "-h", "-u", "janghj", "-w", "devbox",
        "-t", "RUNNING,CONFIGURING", "-o", "%i|%T|%b",
    ]).stdout.splitlines()
    total = 0
    observed: list[dict[str, object]] = []
    for line in lines:
        job_id, state, gres = (line.split("|", 2) + [""])[:3]
        count = sum(int(value) for value in re.findall(r"(?:gres/)?gpu(?::[^,:=|]+)*[:=](\d+)", gres))
        if count:
            observed.append({"job_id": job_id, "state": state, "gres": gres, "gpu": count})
            total += count
    return total, observed


def submit(source_head: str, *, attempt_suffix: str | None = None) -> dict[str, object]:
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    parent = _run(["git", "rev-parse", "HEAD^"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    if source_head != head or branch != BRANCH or dirty:
        raise ODEBFContractError("P2R5 execution source differs")
    if attempt_suffix is not None and not attempt_suffix.replace("-", "").isalnum():
        raise ODEBFContractError("P2R5 attempt suffix differs")
    plan = dry.build_plan(source_head, attempt_suffix=attempt_suffix)
    if any((RESULT_PARENT / str(job["result_name"])).exists() for job in plan["jobs"]):
        raise ODEBFContractError("P2R5 result namespace exists")
    active, active_jobs = _active_gpu_allocations()
    available = PROJECT_GPU_CAP - active
    if available <= 0:
        raise ODEBFContractError("P2R5 server1 project GPU cap has no free allocation")
    stage_concurrency = min(STAGE_GPU_MAX, available, 4)
    if active + stage_concurrency > PROJECT_GPU_CAP:
        raise ODEBFContractError("P2R5 server1 GPU cap differs")
    namespace_tail = f"{attempt_suffix}-{source_head[:12]}" if attempt_suffix else source_head[:12]
    namespace = f"s05-p2r5-sdrt-stage-a-{namespace_tail}-v1"
    intent_path = STATE_ROOT / f"{namespace}.intent.json"
    receipt_path = STATE_ROOT / f"{namespace}.submission-receipt.json"
    if any(path.exists() or path.is_symlink() for path in (intent_path, receipt_path)):
        raise ODEBFContractError("P2R5 submission namespace exists")
    LOG_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    array = f"0-3%{stage_concurrency}"
    node_receipt = _run(["scontrol", "show", "node", "-o", "devbox"]).stdout.strip()
    intent = {
        "schema": "ode-edit-s05-p2r5-sdrt-stage-a-submission-intent/v1",
        "source_head": source_head,
        "source_parent": parent,
        "dry_plan": plan,
        "active_server1_gpu_allocations": active,
        "active_server1_gpu_jobs": active_jobs,
        "new_max_concurrent_gpu": stage_concurrency,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "stage_gpu_max": STAGE_GPU_MAX,
        "node_resource_receipt_sha256": hashlib.sha256(node_receipt.encode()).hexdigest(),
        "held_then_atomic_release": True,
        "array": array,
        "stage_b_status": "CLOSED_PENDING_GH_STAGE_A_REVIEW",
    }
    intent_sha = _write_once(intent_path, intent)
    command = [
        "sbatch", "--hold", "--parsable", "--array", array,
        "--chdir", str(REPO_ROOT), "--nodelist", "devbox",
        "--job-name", "odeedit_s05_p2r5_sdrt_stage_a",
        "--output", str(LOG_ROOT / "%A_%a.out"),
        "--error", str(LOG_ROOT / "%A_%a.err"),
        str(SBATCH), source_head, str(RESULT_PARENT),
    ]
    if attempt_suffix:
        command.append(attempt_suffix)
    submitted = _run(command)
    job_id = submitted.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("P2R5 scheduler ID differs")
    observed = _run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
    required = (
        "JobState=PENDING", "Reason=JobHeldUser", "ReqNodeList=devbox",
        "TRES=cpu=8,mem=65000M,node=1,billing=8,gres/gpu=1",
    )
    if not all(item in observed for item in required):
        _run(["scancel", job_id], check=False)
        raise ODEBFContractError("P2R5 held scheduler contract differs")
    receipt = {
        "schema": "ode-edit-s05-p2r5-sdrt-stage-a-submission/v1",
        "source_head": source_head,
        "source_parent": parent,
        "job_id": job_id,
        "array": array,
        "job_count": 4,
        "endpoint_count": 8,
        "request_attempt_count": 80,
        "active_gpu_allocations_before_release": active,
        "new_max_concurrent_gpu": stage_concurrency,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "intent_sha256": intent_sha,
        "held_inspection_sha256": hashlib.sha256(observed.encode()).hexdigest(),
        "held_then_atomic_release": True,
        "attempt_suffix": attempt_suffix,
        "stage_b_status": "CLOSED_PENDING_GH_STAGE_A_REVIEW",
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
