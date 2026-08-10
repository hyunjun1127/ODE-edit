#!/usr/bin/env python3
"""No-model dry plan for the eight P1R19 Dynamic trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.p1_common_coldcoord_fixed_e8_panel import (
    COMMON_COLD_CASE_SEAL_FILE,
    common_cold_schedule,
    verify_common_cold_case_seal,
)
from project.run_scripts.ode_bf.p1_strength_preserving_router_panel import (
    STRENGTH_PRESERVING_CELLS,
    STRENGTH_PRESERVING_LOCK_FILE,
    STRENGTH_PRESERVING_RESULT_TOKEN,
    expected_strength_preserving_result_name,
    forecast_strength_preserving_panel,
    load_and_validate_strength_preserving_lock,
)
from project.run_scripts.ode_bf.sampling import load_p1_sampling_seal


ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")


def build_plan(source_head: str, *, repository_root: Path = REPO_ROOT) -> dict[str, object]:
    locks = repository_root / "project/run_scripts/ode_bf/locks"
    seal = verify_common_cold_case_seal(
        json.loads((locks / COMMON_COLD_CASE_SEAL_FILE).read_text(encoding="utf-8"))
    )
    schedule = common_cold_schedule(
        load_p1_sampling_seal(
            locks / "p1r2_p_population_seal.json",
            stream_path=locks / "p1r2_seqb10_stream_seal.json",
        )
    )
    numerical, numerical_sha = load_and_validate_strength_preserving_lock(
        locks / STRENGTH_PRESERVING_LOCK_FILE,
        case_root_digest=seal["root_digest"],
        request_order_sha256=seal["batch_ordered_request_digest_v1"][0],
        schedule=schedule,
    )
    jobs: list[dict[str, object]] = []
    for alias in ALIASES:
        forecast = forecast_strength_preserving_panel(
            locks / "p0_artifact_lock.json",
            repository_root / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json",
            alias,
        )
        for cell in STRENGTH_PRESERVING_CELLS:
            jobs.append(
                {
                    "array_index": len(jobs),
                    "alias": alias,
                    "cell": cell,
                    "result_name": expected_strength_preserving_result_name(alias, cell),
                    "run_token": STRENGTH_PRESERVING_RESULT_TOKEN,
                    "gpu": 1,
                    "cpu": 8,
                    "memory_mib": 65_000,
                    "time": "23:59:00",
                    "forecast": forecast.raw_free_payload(),
                }
            )
    return {
        "schema": "ode-edit-s05-strength-preserving-router-p1r19-a2-dry-plan/v1",
        "source_head": source_head,
        "case_root_digest": seal["root_digest"],
        "request_order_sha256": seal["batch_ordered_request_digest_v1"][0],
        "numerical_lock_sha256": numerical_sha,
        "numerical_lock_root_digest": numerical["root_digest"],
        "trajectory_count": 8,
        "array_max_concurrent_gpu": 4,
        "dynamic_only": True,
        "hold_count": 0,
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
    print(json.dumps(build_plan(args.source_head), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
