#!/usr/bin/env python3
"""Held-inspect-release submitter for the one-shot P2R6 capture."""

from __future__ import annotations

import argparse
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

from project.run_scripts.session05_ode_bf_p2r6_capture_a1_dry_plan import build_plan
from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p2r6_replay_capture import TECH_R8_SCIENTIFIC_HEAD


SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_p2r6_capture_a1.sbatch"
STATE_ROOT = REPO_ROOT / "local/odebf/state/p2r6-red-r1-capture-a1"
RESULT_PARENT = REPO_ROOT / "local/odebf/results"
RESULT_NAME = (
    "s05-p2r6-semantic-region-capture-a1-qwen2.5-7b-inst-"
    "case-01-paired-red-r1-v1"
)
LOG_ROOT = REPO_ROOT / "local/odebf/logs/p2r6-red-r1-capture-a1"
BRANCH = "codex/p2r6-red-r1-capture-a1"
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
    os.chmod(path.parent, 0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _active_gpu_allocations() -> tuple[int, list[dict[str, object]]]:
    lines = _run(
        [
            "squeue",
            "-h",
            "-u",
            "janghj",
            "-w",
            "devbox",
            "-t",
            "RUNNING,CONFIGURING",
            "-o",
            "%i|%T|%b",
        ]
    ).stdout.splitlines()
    total = 0
    jobs: list[dict[str, object]] = []
    for line in lines:
        job_id, state, gres = (line.split("|", 2) + [""])[:3]
        count = sum(
            int(value)
            for value in re.findall(r"(?:gres/)?gpu(?::[^,:=|]+)*[:=](\d+)", gres)
        )
        if count:
            total += count
            jobs.append({"job_id": job_id, "state": state, "gres": gres, "gpu": count})
    return total, jobs


def submit(source_head: str) -> dict[str, object]:
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    parent = _run(["git", "rev-parse", "HEAD^"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    if source_head != head or branch != BRANCH or dirty or parent != TECH_R8_SCIENTIFIC_HEAD:
        raise ODEBFContractError("P2R6 capture source/parent differs")
    plan = build_plan(source_head)
    result_root = RESULT_PARENT / RESULT_NAME
    intent_path = STATE_ROOT / "capture-a1.intent.json"
    receipt_path = STATE_ROOT / "capture-a1.submission-receipt.json"
    if any(
        path.exists() or path.is_symlink()
        for path in (result_root, intent_path, receipt_path)
    ):
        raise ODEBFContractError("P2R6 one-shot capture namespace exists")
    active, active_jobs = _active_gpu_allocations()
    if active + 1 > PROJECT_GPU_CAP:
        raise ODEBFContractError("P2R6 capture server1 GPU cap differs")
    LOG_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(LOG_ROOT, 0o700)
    node_receipt = _run(["scontrol", "show", "node", "-o", "devbox"]).stdout.strip()
    intent = {
        "schema": "ode-edit-s05-p2r6-red-r1-capture-submission-intent/v1",
        "source_head": source_head,
        "scientific_parent": parent,
        "dry_plan": plan,
        "active_server1_gpu_allocations": active,
        "active_server1_gpu_jobs": active_jobs,
        "new_max_concurrent_gpu": 1,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "node_resource_receipt_sha256": hashlib.sha256(node_receipt.encode()).hexdigest(),
        "capture_invocation_budget": 1,
        "held_then_atomic_release": True,
        "scientific_endpoint_count": 0,
    }
    intent_sha = _write_once(intent_path, intent)
    command = [
        "sbatch",
        "--hold",
        "--parsable",
        "--chdir",
        str(REPO_ROOT),
        "--nodelist",
        "devbox",
        "--job-name",
        "odeedit_s05_p2r6_capture_a1",
        "--output",
        str(LOG_ROOT / "%j.out"),
        "--error",
        str(LOG_ROOT / "%j.err"),
        str(SBATCH),
        source_head,
        str(RESULT_PARENT),
    ]
    submitted = _run(command)
    job_id = submitted.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("P2R6 capture scheduler ID differs")
    observed = _run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "ReqNodeList=devbox",
        "TRES=cpu=8,mem=65000M,node=1,billing=8,gres/gpu=1",
    )
    if not all(item in observed for item in required):
        _run(["scancel", job_id], check=False)
        raise ODEBFContractError("P2R6 capture held scheduler contract differs")
    receipt = {
        "schema": "ode-edit-s05-p2r6-red-r1-capture-submission/v1",
        "source_head": source_head,
        "scientific_parent": parent,
        "job_id": job_id,
        "job_count": 1,
        "gpu_count": 1,
        "active_gpu_allocations_before_release": active,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "intent_sha256": intent_sha,
        "held_inspection_sha256": hashlib.sha256(observed.encode()).hexdigest(),
        "held_then_atomic_release": True,
        "capture_invocation_budget": 1,
        "scientific_endpoint_count": 0,
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
