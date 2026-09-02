#!/usr/bin/env python3
"""No-model atomic dry plan for P1R23 calibration/B10/B100 stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.p1_scalable_batched_runtime_panel import (
    P1R23_LOCK_FILE,
    expected_p1r23_result_name,
    forecast_p1r23_panel,
    load_and_validate_p1r23_lock,
)
from project.run_scripts.ode_bf.p1r23_b100_seal import (
    verify_p1r23_b100_seal,
)


ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")
STAGES = ("calibration", "b10", "rs-b10", "repair-b10", "b100")


def _roles(stage: str) -> tuple[int, tuple[str, ...]]:
    if stage == "calibration":
        return 10, ("CALIBRATION",)
    if stage == "b10":
        return 10, ("ODE_BF_K8_PAIR", "OPTIMIZED_NATIVE_K1", "OFFICIAL_NATIVE")
    if stage == "rs-b10":
        return 10, ("ODE_BF_K8_RS_PAIR",)
    if stage == "repair-b10":
        return 10, ("ODE_BF_K8_PAIR", "ODE_BF_K8_RS_PAIR", "OFFICIAL_NATIVE")
    if stage == "b100":
        return 100, (
            "ODE_BF_K8_PAIR",
            "ODE_BF_K8_RS_PAIR",
            "OPTIMIZED_NATIVE_K1",
            "OFFICIAL_NATIVE",
        )
    raise ValueError("P1R23 dry stage differs")


def build_plan(
    source_head: str,
    stage: str,
    *,
    repository_root: Path = REPO_ROOT,
) -> dict[str, object]:
    locks = repository_root / "project/run_scripts/ode_bf/locks"
    numerical, numerical_sha = load_and_validate_p1r23_lock(
        locks / P1R23_LOCK_FILE
    )
    batch_size, roles = _roles(stage)
    if batch_size == 100:
        seal = verify_p1r23_b100_seal(
            json.loads(
                (locks / "p1r23_b100_prefix_canonical90_seal.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        b100 = {
            "seal_root": seal["seal_root"],
            "order_root": seal["order_root"],
            "atomic_request_order_sha256": numerical["b100_selection"]
            ["atomic_request_order_sha256"],
        }
    else:
        b100 = None
    jobs: list[dict[str, object]] = []
    for alias in ALIASES:
        forecast = forecast_p1r23_panel(
            locks / "p0_artifact_lock.json",
            repository_root / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json",
            alias,
        )
        for role in roles:
            jobs.append(
                {
                    "array_index": len(jobs),
                    "alias": alias,
                    "batch_size": batch_size,
                    "role": role,
                    "result_name": expected_p1r23_result_name(
                        alias, batch_size=batch_size, role=role
                    ),
                    "request_microbatch_size": numerical["microbatch_accumulation"]
                    ["request_microbatch_size"][alias],
                    "gpu": 1,
                    "cpu": 8,
                    "memory_mib": 60_416,
                    "time": "23:59:00",
                    "forecast": forecast.raw_free_payload(),
                }
            )
    return {
        "schema": "ode-edit-s05-p1r23-scalable-batched-dry-plan/v1",
        "source_head": source_head,
        "stage": stage,
        "estimand": "ATOMIC",
        "batch_size": batch_size,
        "numerical_lock_sha256": numerical_sha,
        "numerical_lock_root_digest": numerical["root_digest"],
        "b100": b100,
        "job_count": len(jobs),
        "trajectory_count": {
            "calibration": 2,
            "b10": 4,
            "rs-b10": 4,
            "repair-b10": 8,
            "b100": 8,
        }[stage],
        "native_control_count": (
            4 if stage in ("b10", "b100") else 2 if stage == "repair-b10" else 0
        ),
        "array_max_concurrent_gpu": min(4, len(jobs)),
        "persistent_history_append_count": 0,
        "sequential_round_count": 0,
        "p1r20_access_count": 0,
        "jobs": jobs,
        "model_load": False,
        "gpu_use": False,
        "slurm_submit": False,
        "result_root_creation": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    args = parser.parse_args()
    print(
        json.dumps(
            build_plan(args.source_head, args.stage),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
