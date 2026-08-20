#!/usr/bin/env python3
"""Held-inspect-release submitter for target + Official AlphaEdit writer A1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1r52_target_official_alphaedit_writer import PHASE_A_TECH_R1_RESULT_NAME
from project.run_scripts import session05_ode_bf_p1r52_target_official_alphaedit_writer_a1_dry_plan as dry
from project.run_scripts.session05_ode_bf_submit_p1r52_piru_sequential_b100x10 import (
    PROJECT_GPU_CAP,
    _gpu_jobs,
    _host_memory_available_mib,
    _run,
    _write_once,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_p1r52_target_official_alphaedit_writer_a1.sbatch"
BRANCH = "codex/p1r52-target-official-alphaedit-writer-a1-v1"


def submit(source_head: str) -> dict[str, object]:
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(["git", "status", "--porcelain"]).stdout
    if source_head != head or branch != BRANCH or dirty:
        raise ODEBFContractError("target/Official writer execution source differs")
    output_root = REPO_ROOT / "local/odebf/results" / PHASE_A_TECH_R1_RESULT_NAME
    state_root = REPO_ROOT / "local/odebf/state/p1r52-target-official-alphaedit-writer-a1-tech-r1-v1"
    log_root = REPO_ROOT / "local/odebf/logs/p1r52-target-official-alphaedit-writer-a1-tech-r1-v1"
    if any(path.exists() or path.is_symlink() for path in (output_root, state_root, log_root)):
        raise ODEBFContractError("target/Official writer namespace exists")
    allocated, jobs = _gpu_jobs()
    if allocated + 1 > PROJECT_GPU_CAP:
        raise ODEBFContractError("target/Official writer project GPU cap differs")
    memory_mib = _host_memory_available_mib()
    if memory_mib < 65000:
        raise ODEBFContractError("target/Official writer host memory is below request")
    plan = dry.build_plan(source_head)
    intent_path = state_root / "submission-intent.json"
    receipt_path = state_root / "submission-receipt.json"
    log_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    intent = {
        "schema": "ode-edit-s05-p1r52-target-official-alphaedit-writer-submission-intent/v1",
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
            "sbatch",
            "--hold",
            "--parsable",
            "--chdir",
            str(REPO_ROOT),
            "--nodelist",
            "devbox",
            "--job-name",
            "ode_p1r52_target_official_writer_a1",
            "--output",
            str(log_root / "%j.out"),
            "--error",
            str(log_root / "%j.err"),
            str(SBATCH),
            source_head,
            str(output_root),
        ]
    )
    job_id = submitted.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("target/Official writer scheduler ID differs")
    observed = _run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "ReqNodeList=devbox",
        "TRES=cpu=8,mem=65000M,node=1,billing=8,gres/gpu=1",
    )
    if not all(item in observed for item in required):
        _run(["scancel", job_id], check=False)
        raise ODEBFContractError("target/Official writer held contract differs")
    receipt = {
        "schema": "ode-edit-s05-p1r52-target-official-alphaedit-writer-submission/v1",
        "source_head": source_head,
        "branch": branch,
        "job_id": job_id,
        "phase_a_pilot_request_count": 100,
        "phase_a_conditional_request_count": 1000,
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
