#!/usr/bin/env python3
"""Raw-free four-cell P1R52 production dry plan."""

from __future__ import annotations

import argparse
import json

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1r52_independent_runtime import expected_p1r52_result_name


CELLS = (
    ("llama3-8b-inst", "neutral"),
    ("llama3-8b-inst", "soft"),
    ("qwen2.5-7b-inst", "neutral"),
    ("qwen2.5-7b-inst", "soft"),
)


def build_plan(source_head: str, *, attempt_suffix: str | None = None) -> dict[str, object]:
    if len(source_head) != 40:
        raise ODEBFContractError("P1R52 dry-plan source identity differs")
    if attempt_suffix is not None and not attempt_suffix.replace("-", "").isalnum():
        raise ODEBFContractError("P1R52 dry-plan attempt suffix differs")
    jobs = [
        {
            "array_index": index,
            "alias": alias,
            "arm": arm,
            "method": f"P1R52-RSA-R42SAFEKDC-M1-{arm.upper()}",
            "result_name": expected_p1r52_result_name(alias, arm, attempt_suffix=attempt_suffix),
            "gpu": 1,
            "cpu": 8,
            "memory_mib": 65000,
        }
        for index, (alias, arm) in enumerate(CELLS)
    ]
    return {
        "schema": "ode-edit-s05-p1r52-rsa-r42safekdc-dry-plan/v1",
        "source_head": source_head,
        "jobs": jobs,
        "job_count": 4,
        "case_count_per_job": 10,
        "request_count_per_case": 10,
        "total_endpoint_attempts": 40,
        "array": "0-3%4",
        "server": "server1/devbox",
        "project_gpu_cap": 4,
        "stage_gpu_max": 4,
        "terminal_evaluator_only": True,
        "stepwise_heldout_evaluator_count": 0,
        "history_mode": "OFF",
        "retry_count": 0,
        "scientific_promotion": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--attempt-suffix")
    args = parser.parse_args()
    print(json.dumps(build_plan(args.source_head, attempt_suffix=args.attempt_suffix), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
