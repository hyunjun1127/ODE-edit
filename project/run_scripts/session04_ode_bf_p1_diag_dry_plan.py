#!/usr/bin/env python3
"""Deterministic no-model plan for the P1R4 full-residual diagnostic pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import MODEL_ALIASES
from project.run_scripts.ode_bf.p1_runtime import (
    expected_p1r4_diagnostic_result_name,
)
from project.run_scripts.ode_bf.resource import forecast_p1_b10_memory


JOB_NAMES = {
    "llama3-8b-inst": "odebf_s04_p1r4full_llama",
    "qwen2.5-7b-inst": "odebf_s04_p1r4full_qwen",
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
                "result_name": expected_p1r4_diagnostic_result_name(alias),
                "gpu": 1,
                "cpu": 8,
                "memory_mib": 60_416,
                "time": "04:00:00",
                "node": "server2",
                "memory_forecast": forecast.raw_free_payload(),
                "memory_forecast_identity": forecast.identity(),
            }
        )
    return {
        "schema": "ode-edit-s04-ode-bf-p1r4diag-dry-plan/v1",
        "instruction_id": (
            "ODEEDIT-S04-ODE-BF-FULL-RESIDUAL-ARMS-P1R4-V1"
        ),
        "authorized_attempt": "DISTINCT_B10_1_FULL_RESIDUAL_ARM_DIAG",
        "source_head": source_head,
        "benchmark": "CounterFact",
        "scientific_sample_reused_for_diagnostic": True,
        "edit_batch_size": 10,
        "sequential_batch_count": 1,
        "arms": ["N32_NATIVE", "F_G", "F_BF", "R_BF"],
        "functional_p_baseline_kind": "outer_entry",
        "residual_definition": "full_current",
        "residual_divisor": 1,
        "arm_local_infeasibility": True,
        "causal_diagnostic_only": True,
        "scientific_promotion_authorized": False,
        "k_resolution": 8,
        "trials_per_slot": 3,
        "persistent_endpoint_commit_count": 0,
        "history_append_count": 0,
        "heldout_access_count": 0,
        "model_load": False,
        "gpu_use": False,
        "slurm_submit": False,
        "retry_or_resubmit": False,
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
