#!/usr/bin/env python3
"""Deterministic no-model Phase1/Phase2 plan for P2R6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.p2r6_pilot_panel import LOCK_FILE, load_and_validate_lock
from project.run_scripts.ode_bf.p2r6_pilot_runtime import (
    PHASE1_CASES,
    PHASE2_CASES,
    expected_p2r6_result_name,
    p2r6_phase_arms,
)
from project.run_scripts.ode_bf.p2r6_semantic_region_controller import P2R6_INSTRUCTION_ID


def build_plan(
    source_head: str,
    *,
    phase: str,
    selected_controller: str | None = None,
    attempt_suffix: str | None = None,
) -> dict[str, object]:
    lock, lock_sha = load_and_validate_lock(
        REPO_ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
    )
    arms = p2r6_phase_arms(phase, selected_controller)
    cases = PHASE1_CASES if phase == "phase1" else PHASE2_CASES
    cells = [
        (alias, case_index)
        for alias in ("llama3-8b-inst", "qwen2.5-7b-inst")
        for case_index in cases[alias]
    ]
    jobs = [
        {
            "array_index": index,
            "model": alias,
            "case_index": case_index,
            "arms": list(arms),
            "endpoint_count": len(arms),
            "request_attempt_count": 10 * len(arms),
            "result_name": expected_p2r6_result_name(
                alias,
                phase=phase,
                case_index=case_index,
                selected_controller=selected_controller,
                attempt_suffix=attempt_suffix,
            ),
        }
        for index, (alias, case_index) in enumerate(cells)
    ]
    stage_gpu_max = 2 if phase == "phase1" else 4
    return {
        "schema": "ode-edit-s05-p2r6-semantic-region-pilot-dry-plan/v1",
        "instruction_id": P2R6_INSTRUCTION_ID,
        "source_head": source_head,
        "phase": phase,
        "selected_controller": selected_controller or "NOT_SELECTED_PHASE1",
        "numerical_lock_sha256": lock_sha,
        "numerical_lock_root": lock["root_digest"],
        "jobs": jobs,
        "job_count": len(jobs),
        "endpoint_attempt_count": sum(int(item["endpoint_count"]) for item in jobs),
        "request_attempt_count": sum(int(item["request_attempt_count"]) for item in jobs),
        "array": f"0-{len(jobs) - 1}%{min(stage_gpu_max, len(jobs))}",
        "project_gpu_cap_server1": 4,
        "stage_gpu_max": stage_gpu_max,
        "cpu_per_task": 8,
        "memory_mib_per_task": 65000,
        "gpu_per_task": 1,
        "clamp_policy": "ON_DECISION_ACTIVE_EVERY_TARGET_MICROSTEP",
        "shadow_added_model_forward_backward_materialization": [0, 0, 0],
        "b10x10_status": "NOT_AUTHORIZED",
        "attempt_suffix": attempt_suffix,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head")
    parser.add_argument("--phase", required=True, choices=("phase1", "phase2"))
    parser.add_argument("--selected-controller", choices=("AR", "AS"))
    parser.add_argument("--attempt-suffix")
    args = parser.parse_args()
    source_head = args.source_head or subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    print(
        json.dumps(
            build_plan(
                source_head,
                phase=args.phase,
                selected_controller=args.selected_controller,
                attempt_suffix=args.attempt_suffix,
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
