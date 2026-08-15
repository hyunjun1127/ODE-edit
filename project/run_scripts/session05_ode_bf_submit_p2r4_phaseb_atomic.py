#!/usr/bin/env python3
"""Held-inspect-release submitter for P2R4 Phase-B paired writer arms."""

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

from project.run_scripts import session05_ode_bf_p2r4_phaseb_atomic_dry_plan as dry
from project.run_scripts.ode_bf.contracts import ODEBFContractError


SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_p2r4_phaseb_atomic.sbatch"
STATE_ROOT = REPO_ROOT / "local/odebf/state/p2r4-phaseb-clamp-on-off-writer"
RESULT_PARENT = REPO_ROOT / "local/odebf/results"
LOG_ROOT = REPO_ROOT / "local/odebf/logs/p2r4-phaseb-clamp-on-off-writer"
BRANCH = "codex/odeeditsh2-s05-p2r4-phaseb-clamp-on-off-writer-v1"
PROJECT_GPU_CAP = 4
STAGE_GPU_MAX = 2


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


def submit(
    source_head: str,
    *,
    case_count: int,
    attempt_suffix: str | None = None,
    array_task: int | None = None,
) -> dict[str, object]:
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    parent = _run(["git", "rev-parse", "HEAD^"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    if source_head != head or branch != BRANCH or dirty:
        raise ODEBFContractError("P2R4 Phase-B execution source differs")
    if case_count not in (1, 10):
        raise ODEBFContractError("P2R4 Phase-B case count differs")
    if attempt_suffix is not None and not attempt_suffix.replace("-", "").isalnum():
        raise ODEBFContractError("P2R4 Phase-B attempt suffix differs")
    if array_task not in (None, 0, 1, 2, 3):
        raise ODEBFContractError("P2R4 Phase-B selected array task differs")
    plan = dry.build_plan(source_head, case_count=case_count, attempt_suffix=attempt_suffix)
    selected_jobs = [
        job
        for job in plan["jobs"]
        if array_task is None or int(job["array_index"]) == array_task
    ]
    if any((RESULT_PARENT / str(job["result_name"])).exists() for job in selected_jobs):
        raise ODEBFContractError("P2R4 Phase-B result namespace exists")
    active, active_jobs = _active_gpu_allocations()
    available = PROJECT_GPU_CAP - active
    if available <= 0:
        raise ODEBFContractError("P2R4 Phase-B server2 project GPU cap has no free allocation")
    stage_concurrency = min(STAGE_GPU_MAX, available, len(selected_jobs))
    if active + stage_concurrency > PROJECT_GPU_CAP:
        raise ODEBFContractError("P2R4 Phase-B server2 GPU cap differs")
    phase = "sealed-b10-smoke" if case_count == 1 else "b10x10"
    namespace_tail = f"{attempt_suffix}-{source_head[:12]}" if attempt_suffix else source_head[:12]
    namespace = f"s05-p2r4-phaseb-{phase}-{namespace_tail}-v1"
    intent_path = STATE_ROOT / f"{namespace}.intent.json"
    receipt_path = STATE_ROOT / f"{namespace}.submission-receipt.json"
    if any(path.exists() or path.is_symlink() for path in (intent_path, receipt_path)):
        raise ODEBFContractError("P2R4 Phase-B submission namespace exists")
    LOG_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    array = f"0-3%{stage_concurrency}" if array_task is None else str(array_task)
    node_receipt = _run(["scontrol", "show", "node", "-o", "devbox"]).stdout.strip()
    intent = {
        "schema": "ode-edit-s05-p2r4-phaseb-submission-intent/v1",
        "source_head": source_head,
        "source_parent": parent,
        "phase": phase,
        "case_count_per_model_clamp_arm": case_count,
        "dry_plan": plan,
        "active_server2_gpu_allocations": active,
        "active_server2_gpu_jobs": active_jobs,
        "new_max_concurrent_gpu": stage_concurrency,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "stage_gpu_max": STAGE_GPU_MAX,
        "node_resource_receipt_sha256": hashlib.sha256(node_receipt.encode()).hexdigest(),
        "held_then_atomic_release": True,
        "array": array,
        "selected_array_task": array_task,
    }
    intent_sha = _write_once(intent_path, intent)
    command = [
        "sbatch", "--hold", "--parsable", "--array", array,
        "--chdir", str(REPO_ROOT), "--nodelist", "devbox",
        "--job-name", f"odeedit_s05_p2r4_phaseb_{phase.replace('-', '_')}",
        "--output", str(LOG_ROOT / "%A_%a.out"),
        "--error", str(LOG_ROOT / "%A_%a.err"),
        str(SBATCH), source_head, str(RESULT_PARENT), str(case_count),
    ]
    if attempt_suffix:
        command.append(attempt_suffix)
    submitted = _run(command)
    job_id = submitted.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("P2R4 Phase-B scheduler ID differs")
    observed = _run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
    required = (
        "JobState=PENDING", "Reason=JobHeldUser", "ReqNodeList=devbox",
        "TRES=cpu=8,mem=65000M,node=1,billing=8,gres/gpu=1",
    )
    if not all(item in observed for item in required):
        _run(["scancel", job_id], check=False)
        raise ODEBFContractError("P2R4 Phase-B held scheduler contract differs")
    receipt = {
        "schema": "ode-edit-s05-p2r4-phaseb-submission/v1",
        "source_head": source_head,
        "source_parent": parent,
        "phase": phase,
        "job_id": job_id,
        "array": array,
        "job_count": len(selected_jobs),
        "cell_count": 2 * len(selected_jobs),
        "case_cell_count": 2 * len(selected_jobs) * case_count,
        "request_attempt_count": 20 * len(selected_jobs) * case_count,
        "active_gpu_allocations_before_release": active,
        "new_max_concurrent_gpu": stage_concurrency,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "intent_sha256": intent_sha,
        "held_inspection_sha256": hashlib.sha256(observed.encode()).hexdigest(),
        "held_then_atomic_release": True,
        "attempt_suffix": attempt_suffix,
        "selected_array_task": array_task,
    }
    receipt_sha = _write_once(receipt_path, receipt)
    _run(["scontrol", "release", job_id])
    return {**receipt, "submission_receipt_sha256": receipt_sha}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--case-count", required=True, type=int, choices=(1, 10))
    parser.add_argument("--attempt-suffix")
    parser.add_argument("--array-task", type=int, choices=(0, 1, 2, 3))
    args = parser.parse_args()
    print(json.dumps(submit(
        args.source_head,
        case_count=args.case_count,
        attempt_suffix=args.attempt_suffix,
        array_task=args.array_task,
    ), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
