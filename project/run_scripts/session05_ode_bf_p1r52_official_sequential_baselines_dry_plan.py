#!/usr/bin/env python3
"""Model-free dry plan for two Official sequential baseline cells."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.p1r52_sequential_contract import HISTORY_COUNTS
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    MEMIT_ROLE,
    NATIVE_CORRECTED_ROLE,
    expected_p1r52_sequential_result_name,
)


def build_plan(source_head: str) -> dict[str, object]:
    roles = (NATIVE_CORRECTED_ROLE, MEMIT_ROLE)
    return {
        "schema": "ode-edit-s05-p1r52-official-sequential-baselines-dry-plan/v1",
        "source_head": source_head,
        "jobs": [
            {
                "array_index": index,
                "role": role,
                "alias": "llama3-8b-inst",
                "result_name": expected_p1r52_sequential_result_name("llama3-8b-inst", role),
                "gpu": 1,
                "cpu": 8,
                "memory_mib": 65000,
            }
            for index, role in enumerate(roles)
        ],
        "job_count": 2,
        "array": "0-1%2",
        "request_count_per_job": 100,
        "alphaedit_reset_cache_sequence": [True] + [False] * 9,
        "alphaedit_cache_history_width_at_entry": list(HISTORY_COUNTS),
        "memit_dynamic_request_history_width": [0] * 10,
        "physical_weight_persistence": True,
        "same_stream_order_context_evaluator": True,
        "retry_backtracking_imputation": [0, 0, 0],
        "held_then_atomic_release": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    args = parser.parse_args()
    print(json.dumps(build_plan(args.source_head), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
