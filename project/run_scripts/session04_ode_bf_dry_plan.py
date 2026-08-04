#!/usr/bin/env python3
"""Deterministic, no-model ODE-BF technical P0 plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import MODEL_ALIASES
from project.run_scripts.ode_bf.p0_runtime import expected_result_name
from project.run_scripts.ode_bf.resource import forecast_p0_b10_memory


JOB_NAMES = {
    "llama3-8b-inst": "odebf_s04_p0_llama",
    "qwen2.5-7b-inst": "odebf_s04_p0_qwen",
}
PACKAGE_ROOT = Path(__file__).resolve().parent / "ode_bf"


def build_plan(
    source_head: str,
    *,
    repository_root: Path = REPO_ROOT,
) -> dict[str, object]:
    artifact = PACKAGE_ROOT / "locks/p0_artifact_lock.json"
    base = repository_root / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json"
    return {
        "schema": "ode-edit-s04-ode-bf-p0-dry-plan/v1",
        "instruction_id": "ODEEDIT-S04-ODE-BF-V1P1-EXACT-FIRST-HIT-CPU-P0-V1-A3",
        "source_head": source_head,
        "edit_batch_size": 10,
        "joint_editor_invocations_per_job": 1,
        "k_resolution": 8,
        "correction_cycles": 1,
        "model_load": False,
        "gpu_use": False,
        "slurm_submit": False,
        "retry_or_resubmit": False,
        "jobs": [
            {
                "alias": alias,
                "job_name": JOB_NAMES[alias],
                "result_name": expected_result_name(alias),
                "gpu": 1,
                "cpu": 8,
                "memory_mib": 65_000,
                "time": "04:00:00",
                "node": "server2",
                "memory_forecast": forecast_p0_b10_memory(
                    artifact,
                    base,
                    alias,
                ).raw_free_payload(),
            }
            for alias in MODEL_ALIASES
        ],
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
