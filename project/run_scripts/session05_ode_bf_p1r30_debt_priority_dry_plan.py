#!/usr/bin/env python3
"""No-model dry plan for the ordered P1R30 atomic stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.p1r30_debt_priority import (
    P1R30_INSTRUCTION_ID,
    P1R30_METHOD_ID,
)
from project.run_scripts.ode_bf.p1r30_debt_priority_panel import (
    expected_p1r30_result_name,
    forecast_p1r30_panel,
)


ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")
PHASE_JOBS = {
    "b1": (
        (ALIASES[0], "P1R30_B1_RS_REFERENCE_SOFT", 2),
        (ALIASES[1], "P1R30_B1_RS_REFERENCE_SOFT", 2),
    ),
    "b10-neutral": (
        (ALIASES[0], "P1R30_B10_RS_DEBT_NEUTRAL", 1),
        (ALIASES[1], "P1R30_B10_RS_DEBT_NEUTRAL", 1),
    ),
    "full-matrix-completion": (
        (ALIASES[0], "P1R30_B10_RS_DEBT_SOFT", 1),
        (ALIASES[1], "P1R30_B10_RS_DEBT_SOFT", 1),
        (ALIASES[0], "P1R30_B10_BG_PAIR", 2),
        (ALIASES[1], "P1R30_B10_BG_PAIR", 2),
    ),
}
CONTRACT_SHA256 = (
    "e0b7fcc3d81963475e57f876f77c4987cbb872fa7ab583e812cd2e9588f60611"
)


def build_plan(
    source_head: str,
    phase: str,
    *,
    repository_root: Path = REPO_ROOT,
) -> dict[str, object]:
    if phase not in PHASE_JOBS:
        raise ValueError("P1R30 dry phase differs")
    locks = repository_root / "project/run_scripts/ode_bf/locks"
    jobs: list[dict[str, object]] = []
    for alias, role, trajectory_count in PHASE_JOBS[phase]:
        forecast = forecast_p1r30_panel(
            locks / "p0_artifact_lock.json",
            repository_root / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json",
            alias,
        )
        jobs.append(
            {
                "array_index": len(jobs),
                "alias": alias,
                "role": role,
                "allocation": "BG" if role == "P1R30_B10_BG_PAIR" else "RS",
                "request_count": 1 if phase == "b1" else 10,
                "trajectory_count": trajectory_count,
                "result_name": expected_p1r30_result_name(alias, role),
                "gpu": 1,
                "cpu": 8,
                "memory_mib": 65_000,
                "time": "23:59:00",
                "forecast": forecast.raw_free_payload(),
            }
        )
    stage_cap = 4 if phase == "full-matrix-completion" else 2
    return {
        "schema": "ode-edit-s05-p1r30-debt-priority-dry-plan/v1",
        "instruction_id": P1R30_INSTRUCTION_ID,
        "method_id": P1R30_METHOD_ID,
        "contract_sha256": CONTRACT_SHA256,
        "source_head": source_head,
        "phase": phase,
        "job_count": len(jobs),
        "trajectory_count": sum(int(item["trajectory_count"]) for item in jobs),
        "server2_janghj_gpu_cap": 4,
        "stage_max_concurrent_gpu": stage_cap,
        "full_matrix_distinct_b10_cell_count": 8,
        "original_b10_seal_root": (
            "3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628"
        ),
        "original_b10_request_order": (
            "984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b"
        ),
        "p1r27_p1r29_controller_inheritance_count": 0,
        "sequential_history_access_count": 0,
        "callback_job_count": 0,
        "jobs": jobs,
        "model_load": False,
        "gpu_use": False,
        "slurm_submit": False,
        "result_root_creation": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--phase", required=True, choices=tuple(PHASE_JOBS))
    args = parser.parse_args()
    print(
        json.dumps(
            build_plan(args.source_head, args.phase),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
