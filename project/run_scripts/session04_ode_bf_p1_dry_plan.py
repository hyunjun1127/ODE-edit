#!/usr/bin/env python3
"""Deterministic no-model plan for the sealed sequential-B10 P1 pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import MODEL_ALIASES
from project.run_scripts.ode_bf.p1_runtime import expected_p1_result_name
from project.run_scripts.ode_bf.resource import forecast_p1_b10_memory


JOB_NAMES = {
    "llama3-8b-inst": "odebf_s04_p1_seqb10_llama",
    "qwen2.5-7b-inst": "odebf_s04_p1_seqb10_qwen",
}
PACKAGE_ROOT = Path(__file__).resolve().parent / "ode_bf"


def build_plan(
    source_head: str,
    *,
    repository_root: Path = REPO_ROOT,
) -> dict[str, object]:
    artifact = PACKAGE_ROOT / "locks/p0_artifact_lock.json"
    base = repository_root / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json"
    jobs = []
    for alias in MODEL_ALIASES:
        forecast = forecast_p1_b10_memory(artifact, base, alias)
        jobs.append(
            {
                "alias": alias,
                "job_name": JOB_NAMES[alias],
                "result_name": expected_p1_result_name(alias),
                "gpu": 1,
                "cpu": 8,
                "memory_mib": 65_000,
                "time": "24:00:00",
                "node": "server2",
                "memory_forecast": forecast.raw_free_payload(),
                "memory_forecast_identity": forecast.identity(),
            }
        )
    return {
        "schema": "ode-edit-s04-ode-bf-p1-dry-plan/v1",
        "instruction_id": "ODEEDIT-S04-ODE-BF-SEQUENTIAL-B10-NATIVE-FLOOR-P1-V1",
        "source_head": source_head,
        "benchmark": "CounterFact",
        "edit_batch_size": 10,
        "sequential_batch_count": 4,
        "logical_edits_per_arm": 40,
        "arms": ["N32_NATIVE", "F_G", "F_BF", "R_BF"],
        "k_resolution": 8,
        "correction_cycles_per_batch": 1,
        "model_load": False,
        "gpu_use": False,
        "slurm_submit": False,
        "retry_or_resubmit": False,
        "post_b10_1_resource_gate": {
            "reserved_gpu_mib_strict_less_than": 52_000,
            "process_rss_mib_strict_less_than": 48_000,
            "total_lane_seconds_with_reserve_strict_less_than": 86_400,
            "reserve_fraction": 0.10,
        },
        "jobs": jobs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    print(
        json.dumps(
            build_plan(args.source_head, repository_root=args.repository_root),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
