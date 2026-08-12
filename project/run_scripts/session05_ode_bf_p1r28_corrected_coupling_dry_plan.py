#!/usr/bin/env python3
"""No-model P1R28 B1/Atomic dry plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.p1r24_atomic_strength_panel import (
    expected_p1r24_result_name,
    forecast_p1r24_panel,
)
from project.run_scripts.ode_bf.p1r28_corrected_coupling import (
    P1R28_INSTRUCTION_ID,
    P1R28_METHOD_ID,
)


def build_plan(source_head: str, phase: str, *, repository_root: Path = REPO_ROOT) -> dict[str, object]:
    if phase not in ("smoke", "production"):
        raise ValueError("P1R28 dry phase differs")
    role = "P1R28_B1_RS_PAIR" if phase == "smoke" else "P1R28_B10_RS_PAIR"
    jobs: list[dict[str, object]] = []
    locks = repository_root / "project/run_scripts/ode_bf/locks"
    for alias in ("llama3-8b-inst", "qwen2.5-7b-inst"):
        forecast = forecast_p1r24_panel(
            locks / "p0_artifact_lock.json",
            repository_root / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json",
            alias,
        )
        anchor_runs = int(phase == "smoke" and alias == "llama3-8b-inst")
        jobs.append({
            "array_index": len(jobs),
            "alias": alias,
            "role": role,
            "request_count": 1 if phase == "smoke" else 10,
            "trajectory_count": 2 + anchor_runs,
            "corrected_trajectory_count": 2,
            "anchor_trajectory_count": anchor_runs,
            "anchor_reuse": (
                "SAME_JOB_EXACT_CE8C6C3_P1R24_RERUN"
                if anchor_runs
                else "EXACT_IMMUTABLE_P1R24_RS_NEUTRAL"
            ),
            "result_name": expected_p1r24_result_name(alias, role),
            "gpu": 1,
            "cpu": 8,
            "memory_mib": 65_000,
            "time": "23:59:00",
            "forecast": forecast.raw_free_payload(),
        })
    return {
        "schema": "ode-edit-s05-p1r28-dry-plan/v1",
        "instruction_id": P1R28_INSTRUCTION_ID,
        "method_id": P1R28_METHOD_ID,
        "source_head": source_head,
        "phase": phase,
        "job_count": 2,
        "trajectory_count": 5 if phase == "smoke" else 4,
        "corrected_trajectory_count": 4,
        "anchor_trajectory_count": 1 if phase == "smoke" else 0,
        "project_gpu_cap": 4,
        "actual_concurrent_gpu_limit": 2,
        "array_max_concurrent_gpu": 2,
        "seal_root": "3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628",
        "request_order": "984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b",
        "callback_job_count": 0,
        "history_access_count": 0,
        "soft_bg_access_count": 0,
        "jobs": jobs,
        "model_load": False,
        "gpu_use": False,
        "slurm_submit": False,
        "result_root_creation": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--phase", required=True, choices=("smoke", "production"))
    args = parser.parse_args()
    print(json.dumps(build_plan(args.source_head, args.phase), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
