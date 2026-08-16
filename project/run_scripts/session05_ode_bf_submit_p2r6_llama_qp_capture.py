#!/usr/bin/env python3
"""Held-inspect-release submitter for the one-shot P2R6 Llama QP capture."""

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

from project.run_scripts import session05_ode_bf_p2r6_llama_qp_capture_dry_plan as dry
from project.run_scripts.ode_bf.contracts import ODEBFContractError


SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_p2r6_llama_qp_capture.sbatch"
STATE_ROOT = REPO_ROOT / "local/odebf/state/p2r6-llama-outer1-qp-capture-v1"
RESULT_PARENT = REPO_ROOT / "local/odebf/results"
LOG_ROOT = REPO_ROOT / "local/odebf/logs/p2r6-llama-outer1-qp-capture-v1"
BRANCH = "codex/p2r6-llama-outer1-qp-capture-v1"
PROJECT_GPU_CAP = 4
ATTEMPT_SUFFIX = "a1"


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
    observed: list[dict[str, object]] = []
    for line in lines:
        job_id, state, gres = (line.split("|", 2) + [""])[:3]
        count = sum(
            int(value)
            for value in re.findall(r"(?:gres/)?gpu(?::[^,:=|]+)*[:=](\d+)", gres)
        )
        if count:
            total += count
            observed.append({"job_id": job_id, "state": state, "gres": gres, "gpu": count})
    return total, observed


def submit(source_head: str) -> dict[str, object]:
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    parent = _run(["git", "rev-parse", "HEAD^"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    if source_head != head or branch != BRANCH or dirty:
        raise ODEBFContractError("P2R6 capture execution source differs")
    plan = dry.build_plan(source_head)
    result_name = (
        "s05-p2r6-llama-outer1-qp-capture-llama3-8b-inst-"
        "case-05-ar-cap-a1-v1"
    )
    if (RESULT_PARENT / result_name).exists():
        raise ODEBFContractError("P2R6 capture result namespace exists")
    active, active_jobs = _active_gpu_allocations()
    if active + 1 > PROJECT_GPU_CAP:
        raise ODEBFContractError("P2R6 capture server1 GPU cap differs")
    namespace = f"s05-p2r6-llama-outer1-qp-capture-{source_head[:12]}-a1-v1"
    intent_path = STATE_ROOT / f"{namespace}.intent.json"
    receipt_path = STATE_ROOT / f"{namespace}.submission-receipt.json"
    if any(path.exists() or path.is_symlink() for path in (intent_path, receipt_path)):
        raise ODEBFContractError("P2R6 capture submission namespace exists")
    LOG_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    node_receipt = _run(["scontrol", "show", "node", "-o", "devbox"]).stdout.strip()
    intent = {
        "schema": "ode-edit-s05-p2r6-llama-outer1-qp-capture-intent/v1",
        "source_head": source_head,
        "source_parent": parent,
        "dry_plan": plan,
        "active_server1_gpu_allocations": active,
        "active_server1_gpu_jobs": active_jobs,
        "new_gpu_count": 1,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "node_resource_receipt_sha256": hashlib.sha256(node_receipt.encode()).hexdigest(),
        "held_inspect_release": True,
        "capture_invocation_budget": 1,
        "phase1_status": "NOT_EXECUTED_CAPTURE_ONLY",
        "phase2_status": "CLOSED",
    }
    intent_sha = _write_once(intent_path, intent)
    submitted = _run(
        [
            "sbatch",
            "--hold",
            "--parsable",
            "--chdir",
            str(REPO_ROOT),
            "--nodelist",
            "devbox",
            "--job-name",
            "odeedit_s05_p2r6_llama_qp_capture",
            "--output",
            str(LOG_ROOT / "%j.out"),
            "--error",
            str(LOG_ROOT / "%j.err"),
            str(SBATCH),
            source_head,
            str(RESULT_PARENT),
            ATTEMPT_SUFFIX,
        ]
    )
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
        "schema": "ode-edit-s05-p2r6-llama-outer1-qp-capture-submission/v1",
        "source_head": source_head,
        "source_parent": parent,
        "job_id": job_id,
        "model": "llama3-8b-inst",
        "case_index": 5,
        "arm": "AR-CAP",
        "active_gpu_allocations_before_release": active,
        "new_gpu_count": 1,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "intent_sha256": intent_sha,
        "held_inspection_sha256": hashlib.sha256(observed.encode()).hexdigest(),
        "held_inspect_release": True,
        "capture_invocation_budget": 1,
        "scientific_endpoint_count": 0,
        "phase2_status": "CLOSED",
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
