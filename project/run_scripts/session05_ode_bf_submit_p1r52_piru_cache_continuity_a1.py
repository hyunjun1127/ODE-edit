#!/usr/bin/env python3
"""Held-inspect-release submitter for PIR-U cache continuity A1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts import session05_ode_bf_p1r52_piru_cache_continuity_a1_dry_plan as dry
from project.run_scripts.session05_ode_bf_submit_p1r52_piru_sequential_b100x10 import (
    PROJECT_GPU_CAP,
    _gpu_jobs,
    _host_memory_available_mib,
    _run,
    _write_once,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_p1r52_piru_cache_continuity_a1.sbatch"
BRANCH = "codex/p1r52-piru-cache-continuity-a1-v1"


def submit(source_head: str, stage: str) -> dict[str, object]:
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(["git", "status", "--porcelain"]).stdout
    if source_head != head or branch != BRANCH or dirty:
        raise ODEBFContractError("PIR-U cache-continuity execution source differs")
    plan = dry.build_plan(source_head, stage)
    allocated, jobs = _gpu_jobs()
    if allocated + int(plan["job_count"]) > PROJECT_GPU_CAP:
        raise ODEBFContractError("PIR-U cache-continuity project GPU cap differs")
    memory_mib = _host_memory_available_mib()
    if memory_mib < 65000 * int(plan["job_count"]):
        raise ODEBFContractError("PIR-U cache-continuity host memory is below request")
    token = "p1r52-piru-cache-continuity-a1-stage-a-b10-v1" if stage == "A-B10" else "p1r52-piru-cache-continuity-a1-stage-b-v1"
    state_root = REPO_ROOT / "local/odebf/state" / token
    log_root = REPO_ROOT / "local/odebf/logs" / token
    roots = [REPO_ROOT / "local/odebf/results" / item["result_name"] for item in plan["jobs"]]
    if any(path.exists() or path.is_symlink() for path in (*roots, state_root, log_root)):
        raise ODEBFContractError("PIR-U cache-continuity namespace exists")
    log_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    intent = {
        "schema": "ode-edit-s05-p1r52-piru-cache-continuity-submission-intent/v1",
        "source_head": source_head,
        "branch": branch,
        "stage": stage,
        "dry_plan": plan,
        "active_gpu_allocations": allocated,
        "active_or_pending_gpu_jobs": jobs,
        "host_memory_available_mib": memory_mib,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "held_then_release": True,
    }
    intent_sha = _write_once(state_root / "submission-intent.json", intent)
    ids: list[str] = []
    inspections: list[str] = []
    try:
        for item, root in zip(plan["jobs"], roots, strict=True):
            submitted = _run([
                "sbatch", "--hold", "--parsable", "--chdir", str(REPO_ROOT),
                "--nodelist", "devbox", "--job-name", f"ode_piru_cache_a1_{item['array_index']}",
                "--output", str(log_root / f"{item['array_index']}-%j.out"),
                "--error", str(log_root / f"{item['array_index']}-%j.err"),
                str(SBATCH), source_head, str(root), str(item["arm"]), stage,
            ])
            job_id = submitted.stdout.strip().split(";", 1)[0]
            observed = _run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
            if not job_id.isdigit() or not all(value in observed for value in (
                "JobState=PENDING", "Reason=JobHeldUser", "ReqNodeList=devbox",
                "TRES=cpu=8,mem=65000M,node=1,billing=8,gres/gpu=1",
            )):
                raise ODEBFContractError("PIR-U cache-continuity held contract differs")
            ids.append(job_id)
            inspections.append(hashlib.sha256(observed.encode()).hexdigest())
        for job_id in ids:
            _run(["scontrol", "release", job_id])
    except Exception:
        for job_id in ids:
            _run(["scancel", job_id], check=False)
        raise
    receipt = {
        "schema": "ode-edit-s05-p1r52-piru-cache-continuity-submission/v1",
        "source_head": source_head,
        "branch": branch,
        "stage": stage,
        "job_ids": ids,
        "job_map": plan["jobs"],
        "active_gpu_allocations_before_release": allocated,
        "host_memory_available_mib": memory_mib,
        "intent_sha256": intent_sha,
        "held_inspection_sha256": inspections,
    }
    receipt_sha = _write_once(state_root / "submission-receipt.json", receipt)
    return {**receipt, "submission_receipt_sha256": receipt_sha}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--stage", required=True, choices=("A-B10", "B"))
    args = parser.parse_args()
    print(json.dumps(submit(args.source_head, args.stage), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
