#!/usr/bin/env python3
"""Raw-free three-cell P1R52-FPiQ Atomic production dry plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1r52_independent_runtime import (
    expected_p1r52_result_name,
)


POLICIES = ("j0", "sv", "fpiq")


def build_plan(
    source_head: str, *, attempt_suffix: str | None = None
) -> dict[str, object]:
    if len(source_head) != 40:
        raise ODEBFContractError("P1R52-FPiQ dry-plan source identity differs")
    jobs = [
        {
            "array_index": index,
            "alias": "llama3-8b-inst",
            "arm": policy,
            "method": f"P1R52-FPIQ-{policy.upper()}",
            "result_name": expected_p1r52_result_name(
                "llama3-8b-inst", policy, attempt_suffix=attempt_suffix
            ),
            "gpu": 1,
            "cpu": 8,
            "memory_mib": 65000,
        }
        for index, policy in enumerate(POLICIES)
    ]
    return {
        "schema": "ode-edit-s05-p1r52-fpiq-atomic-dry-plan/v1",
        "source_head": source_head,
        "jobs": jobs,
        "job_count": 3,
        "case_count_per_job": 10,
        "request_count_per_case": 10,
        "total_endpoint_attempts": 30,
        "array": "0-2%3",
        "server": "server1/devbox",
        "project_gpu_cap": 4,
        "stage_gpu_max": 3,
        "terminal_evaluator_only": True,
        "stepwise_heldout_evaluator_count": 0,
        "alpha_cache": "ON",
        "structural_h": "OFF",
        "history_mode": "OFF",
        "retry_count": 0,
        "sequential_historical_stage": "NOT_AUTHORIZED",
        "scientific_promotion": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--attempt-suffix")
    args = parser.parse_args()
    print(
        json.dumps(
            build_plan(args.source_head, attempt_suffix=args.attempt_suffix),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
