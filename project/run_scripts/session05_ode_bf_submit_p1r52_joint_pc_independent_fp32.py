#!/usr/bin/env python3
"""Held-inspect-release submitter for independent full-FP32 Joint-P/C."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.session05_ode_bf_p1r52_joint_pc_independent_fp32_dry_plan import build_plan


SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_p1r52_joint_pc_independent_fp32.sbatch"
RESULT_PARENT = REPO_ROOT / "local/odebf/results"
STATE_PARENT = REPO_ROOT / "local/odebf/state"
LOG_PARENT = REPO_ROOT / "local/odebf/logs"
BRANCH = "codex/p1r52-joint-pc-fullfp32-independent-10xb100-v1"
ARRAY = "0-4%4"
PROJECT_GPU_CAP = 4
LOGICAL_SERVER = "server1"
SCHEDULER_NODE = "devbox"


def run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=REPO_ROOT, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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
    lines = run([
        "squeue", "-h", "-u", "janghj", "-w", SCHEDULER_NODE,
        "-t", "RUNNING,CONFIGURING", "-o", "%i|%T|%b|%j",
    ]).stdout.splitlines()
    total = 0
    jobs: list[dict[str, object]] = []
    for line in lines:
        job, state, gres, name = line.split("|", 3)
        count = sum(
            int(value)
            for value in re.findall(r"(?:gres/)?gpu(?::[^,:=|]+)*[:=](\d+)", gres)
        )
        if count:
            total += count
            jobs.append({
                "job_id": job, "state": state, "gres": gres,
                "name": name, "gpu": count,
            })
    return total, jobs


def host_snapshot() -> dict[str, object]:
    memory = run(["free", "-b"]).stdout
    gpu = run([
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]).stdout
    return {
        "logical_server": LOGICAL_SERVER,
        "scheduler_node": SCHEDULER_NODE,
        "free_b_sha256": hashlib.sha256(memory.encode()).hexdigest(),
        "free_b_lines": memory.splitlines(),
        "nvidia_smi_sha256": hashlib.sha256(gpu.encode()).hexdigest(),
        "nvidia_smi_rows": gpu.splitlines(),
    }


def submit(source_head: str, source_tree: str) -> dict[str, object]:
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    tree = run(["git", "rev-parse", "HEAD^{tree}"]).stdout.strip()
    branch = run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    if source_head != head or source_tree != tree or branch != BRANCH or dirty:
        raise ODEBFContractError("independent FP32 execution source differs")
    plan = build_plan(source_head, source_tree)
    if any(
        (RESULT_PARENT / str(cell["result_name"])).exists()
        for cell in plan["cells"]
    ):
        raise ODEBFContractError("independent FP32 result namespace exists")
    active, jobs = active_gpus()
    if active != 0 or active + 4 > PROJECT_GPU_CAP:
        raise ODEBFContractError("independent FP32 server1 GPU cap unavailable")
    namespace = f"p1r52-joint-pc-independent-fp32-{source_head[:12]}-v1"
    state_root = STATE_PARENT / namespace
    log_root = LOG_PARENT / namespace
    if state_root.exists() or state_root.is_symlink() or log_root.exists() or log_root.is_symlink():
        raise ODEBFContractError("independent FP32 state/log namespace exists")
    state_root.mkdir(mode=0o700, parents=True)
    log_root.mkdir(mode=0o700, parents=True)
    snapshot = host_snapshot()
    intent = {
        "schema": "ode-edit-s05-p1r52-joint-pc-independent-fp32-intent/v1",
        "source_head": source_head,
        "source_tree": source_tree,
        "dry_plan": plan,
        "active_server1_gpu_allocations": active,
        "active_server1_gpu_jobs": jobs,
        "host_gpu_memory_snapshot": snapshot,
        "new_max_concurrent_gpu": 4,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "array": ARRAY,
        "held_then_atomic_release": True,
    }
    intent_sha = write_once(state_root / "intent.json", intent)
    submitted = run([
        "sbatch", "--hold", "--parsable", "--array", ARRAY,
        "--chdir", str(REPO_ROOT), "--nodelist", SCHEDULER_NODE,
        "--job-name", "odeedit_p1r52_joint_pc_ind_fp32",
        "--output", str(log_root / "%A_%a.out"),
        "--error", str(log_root / "%A_%a.err"),
        str(SBATCH), source_head, source_tree, str(RESULT_PARENT),
    ])
    job_id = submitted.stdout.strip().split(";", 1)[0]
    observed = run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
    required = (
        "JobState=PENDING", "Reason=JobHeldUser", f"ReqNodeList={SCHEDULER_NODE}",
        "TRES=cpu=8,mem=65000M,node=1,billing=8,gres/gpu=1",
    )
    if not job_id.isdigit() or not all(value in observed for value in required):
        run(["scancel", job_id], check=False)
        raise ODEBFContractError("independent FP32 held scheduler contract differs")
    receipt = {
        "schema": "ode-edit-s05-p1r52-joint-pc-independent-fp32-submission/v1",
        "source_head": source_head,
        "source_tree": source_tree,
        "job_id": job_id,
        "array": ARRAY,
        "cell_count": 5,
        "method_count": 6,
        "independent_case_count_per_method": 10,
        "request_attempt_count": 6000,
        "active_gpu_allocations_before_release": active,
        "new_max_concurrent_gpu": 4,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "logical_server": LOGICAL_SERVER,
        "scheduler_node": SCHEDULER_NODE,
        "intent_sha256": intent_sha,
        "held_inspection_sha256": hashlib.sha256(observed.encode()).hexdigest(),
        "held_then_atomic_release": True,
        "state_root": str(state_root),
        "log_root": str(log_root),
    }
    receipt_sha = write_once(state_root / "submission-receipt.json", receipt)
    run(["scontrol", "release", job_id])
    return {**receipt, "submission_receipt_sha256": receipt_sha}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: submitter SOURCE_HEAD SOURCE_TREE")
    print(json.dumps(submit(sys.argv[1], sys.argv[2]), sort_keys=True, separators=(",", ":")))
