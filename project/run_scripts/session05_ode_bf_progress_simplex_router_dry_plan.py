#!/usr/bin/env python3
"""No-model dry plan for the four paired progress-simplex B10 jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.artifacts import load_rooted_json
from project.run_scripts.ode_bf.p1_scalable_batched_runtime_panel import (
    expected_p1r23_result_name,
    forecast_p1r23_panel,
)


ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")
ROLES = ("PROGRESS_SIMPLEX_BG_PAIR", "PROGRESS_SIMPLEX_RS_PAIR")


def build_plan(source_head: str, *, repository_root: Path = REPO_ROOT) -> dict[str, object]:
    locks = repository_root / "project/run_scripts/ode_bf/locks"
    numerical, numerical_sha = load_rooted_json(
        locks / "numerical_lock_s05_progress_simplex_router.json",
        expected_schema="ode-edit-s05-p1r23-progress-simplex-router-lock/v1",
    )
    jobs: list[dict[str, object]] = []
    for alias in ALIASES:
        forecast = forecast_p1r23_panel(
            locks / "p0_artifact_lock.json",
            repository_root / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json",
            alias,
        )
        for role in ROLES:
            jobs.append(
                {
                    "array_index": len(jobs),
                    "alias": alias,
                    "batch_size": 10,
                    "role": role,
                    "result_name": expected_p1r23_result_name(
                        alias, batch_size=10, role=role
                    ),
                    "request_microbatch_size": 2,
                    "gpu": 1,
                    "cpu": 8,
                    "memory_mib": 60_416,
                    "time": "23:59:00",
                    "forecast": forecast.raw_free_payload(),
                }
            )
    return {
        "schema": "ode-edit-s05-p1r23-progress-simplex-dry-plan/v1",
        "source_head": source_head,
        "instruction_id": numerical["instruction_id"],
        "method_id": numerical["method_id"],
        "numerical_lock_sha256": numerical_sha,
        "numerical_lock_root_digest": numerical["root_digest"],
        "batch_size": 10,
        "job_count": 4,
        "trajectory_count": 8,
        "array_max_concurrent_gpu": 4,
        "native_control_count": 0,
        "B100_access_count": 0,
        "persistent_history_append_count": 0,
        "jobs": jobs,
        "model_load": False,
        "gpu_use": False,
        "slurm_submit": False,
        "result_root_creation": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_plan(args.source_head),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
