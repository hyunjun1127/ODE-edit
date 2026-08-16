#!/usr/bin/env python3
"""No-model P1R35 B1/production dry plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.p1r35_full_current_residual import (
    P1R35_INSTRUCTION_ID,
    P1R35_METHOD_ID,
)
from project.run_scripts.ode_bf.p1r35_full_current_residual_panel import (
    expected_p1r35_result_name,
    forecast_p1r35_panel,
)

ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")


def build_plan(
    source_head: str, phase: str, *, repository_root: Path = REPO_ROOT
) -> dict[str, object]:
    if phase not in ("smoke", "production"):
        raise ValueError("P1R35 dry phase differs")
    role = "P1R35_B1_RS_PAIR" if phase == "smoke" else "P1R35_B10_RS_PAIR"
    locks = repository_root / "project/run_scripts/ode_bf/locks"
    jobs = []
    for index, alias in enumerate(ALIASES):
        forecast = forecast_p1r35_panel(
            locks / "p0_artifact_lock.json",
            repository_root / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json",
            alias,
        )
        jobs.append(
            {
                "array_index": index,
                "alias": alias,
                "role": role,
                "request_count": 1 if phase == "smoke" else 10,
                "trajectory_count": 2,
                "result_name": expected_p1r35_result_name(alias, role),
                "gpu": 1,
                "cpu": 8,
                "memory_mib": 65000,
                "time": "23:59:00",
                "forecast": forecast.raw_free_payload(),
            }
        )
    return {
        "schema": "ode-edit-s05-p1r35-dry-plan/v1",
        "instruction_id": P1R35_INSTRUCTION_ID,
        "method_id": P1R35_METHOD_ID,
        "source_head": source_head,
        "phase": phase,
        "job_count": 2,
        "trajectory_count": 4,
        "server1_project_gpu_cap": 4,
        "stage_max_concurrent_gpu": 2,
        "array": "0-1%2",
        "seal_root": "3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628",
        "request_order": "984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b",
        "scientific_parent": "ac321d86a634a20888cb57003a65467d40aa5ddd",
        "coordinate": "u=d+lag=target_next-current_terminal",
        "remaining_horizon_division_count": 0,
        "history_count": 0,
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
