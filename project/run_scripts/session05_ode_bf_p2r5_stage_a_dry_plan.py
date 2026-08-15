#!/usr/bin/env python3
"""Deterministic no-model Stage-A plan for P2R5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.p2r5_sdrt_writer import P2R5_ARMS, P2R5_INSTRUCTION_ID
from project.run_scripts.ode_bf.p2r5_stage_a_panel import LOCK_FILE, load_and_validate_lock
from project.run_scripts.ode_bf.p2r5_stage_a_runtime import (
    STAGE_A_CASES,
    expected_p2r5_stage_a_result_name,
)


def build_plan(source_head: str, *, attempt_suffix: str | None = None) -> dict[str, object]:
    lock, lock_sha = load_and_validate_lock(
        REPO_ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
    )
    cells = [
        ("llama3-8b-inst", 3),
        ("llama3-8b-inst", 5),
        ("qwen2.5-7b-inst", 1),
        ("qwen2.5-7b-inst", 4),
    ]
    tech_r3_arms = (
        [("SDRT-STRUCTP",), P2R5_ARMS, P2R5_ARMS, ("SDRT-CAP",)]
        if attempt_suffix == "tech-r3" else [P2R5_ARMS] * 4
    )
    jobs = [
        {
            "array_index": index,
            "model": alias,
            "case_index": case_index,
            "arms": list(tech_r3_arms[index]),
            "endpoint_count": len(tech_r3_arms[index]),
            "request_attempt_count": 10 * len(tech_r3_arms[index]),
            "result_name": expected_p2r5_stage_a_result_name(
                alias,
                case_index=case_index,
                attempt_suffix=attempt_suffix,
            ),
        }
        for index, (alias, case_index) in enumerate(cells)
    ]
    return {
        "schema": "ode-edit-s05-p2r5-sdrt-stage-a-dry-plan/v1",
        "instruction_id": P2R5_INSTRUCTION_ID,
        "source_head": source_head,
        "numerical_lock_sha256": lock_sha,
        "numerical_lock_root": lock["root_digest"],
        "stage_a_cases": {key: list(value) for key, value in STAGE_A_CASES.items()},
        "jobs": jobs,
        "job_count": 4,
        "endpoint_attempt_count": sum(int(item["endpoint_count"]) for item in jobs),
        "request_attempt_count": sum(int(item["request_attempt_count"]) for item in jobs),
        "array": "0-3%4",
        "project_gpu_cap_server1": 4,
        "stage_gpu_max": 4,
        "cpu_per_task": 8,
        "memory_mib_per_task": 65000,
        "gpu_per_task": 1,
        "clamp_policy": "ON_DECISION_ACTIVE_EVERY_TARGET_MICROSTEP",
        "stage_b_status": "CLOSED_PENDING_GH_STAGE_A_REVIEW",
        "attempt_suffix": attempt_suffix,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head")
    parser.add_argument("--attempt-suffix")
    args = parser.parse_args()
    source_head = args.source_head or subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    print(json.dumps(build_plan(source_head, attempt_suffix=args.attempt_suffix), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
