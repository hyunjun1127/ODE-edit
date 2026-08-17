#!/usr/bin/env python3
"""Raw-free P1R51 phase dry plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1r51_independent_runtime import (
    METHOD_BY_PHASE,
    PHASES,
    expected_p1r51_result_name,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_plan(
    source_head: str,
    phase: str,
    *,
    repository_root: Path = REPO_ROOT,
    attempt_suffix: str | None = None,
) -> dict[str, object]:
    del repository_root
    if len(source_head) != 40 or phase not in PHASES:
        raise ODEBFContractError("P1R51 dry-plan identity differs")
    if attempt_suffix is not None and not attempt_suffix.replace("-", "").isalnum():
        raise ODEBFContractError("P1R51 dry-plan attempt suffix differs")
    jobs = [
        {
            "array_index": index,
            "alias": alias,
            "phase": phase,
            "method": METHOD_BY_PHASE[phase],
            "result_name": expected_p1r51_result_name(
                alias, phase=phase, attempt_suffix=attempt_suffix
            ),
            "gpu": 1,
            "cpu": 8,
            "memory_mib": 65000,
        }
        for index, alias in enumerate(("llama3-8b-inst", "qwen2.5-7b-inst"))
    ]
    case_count = 1 if phase in ("b1", "pilot-neutral") else 10
    request_count = 1 if phase == "b1" else 10
    return {
        "schema": "ode-edit-s05-p1r51-rsa-a1-dry-plan/v1",
        "source_head": source_head,
        "phase": phase,
        "jobs": jobs,
        "job_count": 2,
        "case_count_per_job": case_count,
        "request_count_per_case": request_count,
        "total_endpoint_attempts": 2 * case_count,
        "array": "0-1%2",
        "server": "server1/devbox",
        "project_gpu_cap": 4,
        "stage_gpu_max": 2,
        "terminal_evaluator_only": phase != "b1",
        "stepwise_heldout_evaluator_count": 0,
        "history_mode": "OFF",
        "retry_count": 0,
        "scientific_promotion": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--attempt-suffix")
    args = parser.parse_args()
    print(json.dumps(build_plan(args.source_head, args.phase, attempt_suffix=args.attempt_suffix), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
