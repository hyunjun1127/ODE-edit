#!/usr/bin/env python3
"""No-model P1R26 A1 RS+BG paired matrix plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.p1r26_asdc import P1R26_INSTRUCTION_ID, P1R26_METHOD_ID
from project.run_scripts.ode_bf.p1r26_asdc_panel import expected_p1r26_result_name, forecast_p1r26_panel


ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")
ALLOCATIONS = ("RS", "BG")


def build_plan(source_head: str, phase: str, *, repository_root: Path = REPO_ROOT) -> dict[str, object]:
    if phase not in ("smoke", "production"):
        raise ValueError("P1R26 dry phase differs")
    locks = repository_root / "project/run_scripts/ode_bf/locks"
    jobs: list[dict[str, object]] = []
    for alias in ALIASES:
        forecast = forecast_p1r26_panel(
            locks / "p0_artifact_lock.json",
            repository_root / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json",
            alias,
        )
        for allocation in ALLOCATIONS:
            role = f"P1R26_{'B1' if phase == 'smoke' else 'B10X10'}_{allocation}_PAIR"
            jobs.append(
                {
                    "array_index": len(jobs),
                    "alias": alias,
                    "allocation": allocation,
                    "role": role,
                    "request_count_per_atomic_case": 1 if phase == "smoke" else 10,
                    "case_count": 1 if phase == "smoke" else 10,
                    "trajectory_count": 2 if phase == "smoke" else 20,
                    "result_name": expected_p1r26_result_name(alias, role),
                    "gpu": 1,
                    "cpu": 8,
                    "memory_mib": 65000,
                    "time": "23:59:00",
                    "forecast": forecast.raw_free_payload(),
                }
            )
    return {
        "schema": "ode-edit-s05-p1r26-asdc-dry-plan/v1",
        "instruction_id": P1R26_INSTRUCTION_ID,
        "method_id": P1R26_METHOD_ID,
        "source_head": source_head,
        "phase": phase,
        "job_count": 4,
        "trajectory_count": sum(int(item["trajectory_count"]) for item in jobs),
        "project_gpu_cap_server1": 4,
        "array_max_concurrent_gpu": 4,
        "stream_root": "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6" if phase == "production" else None,
        "stream_order": "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c" if phase == "production" else None,
        "history_access_count": 0,
        "callback_job_count": 0,
        "native_rerun_count": 0,
        "model_load": False,
        "gpu_use": False,
        "slurm_submit": False,
        "result_root_creation": False,
        "jobs": jobs,
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
