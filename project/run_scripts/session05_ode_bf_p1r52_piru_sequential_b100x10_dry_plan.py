#!/usr/bin/env python3
"""Raw-free one-cell PIR-U sequential 10xB100 dry plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1r52_piru_sequential_adapter import (
    P1R52_PIRU_SEQUENTIAL_INSTRUCTION_ID,
    P1R52_PIRU_SEQUENTIAL_RESULT_NAME,
    P1R52_PIRU_SEQUENTIAL_ROLE,
)


def build_plan(source_head: str) -> dict[str, object]:
    if len(source_head) != 40:
        raise ODEBFContractError("P1R52 PIR-U sequential dry-plan source differs")
    return {
        "schema": "ode-edit-s05-p1r52-piru-sequential-10xb100-dry-plan/v1",
        "instruction_id": P1R52_PIRU_SEQUENTIAL_INSTRUCTION_ID,
        "source_head": source_head,
        "jobs": [
            {
                "array_index": 0,
                "role": P1R52_PIRU_SEQUENTIAL_ROLE,
                "alias": "llama3-8b-inst",
                "result_name": P1R52_PIRU_SEQUENTIAL_RESULT_NAME,
                "gpu": 1,
                "cpu": 8,
                "memory_mib": 65000,
            }
        ],
        "job_count": 1,
        "round_count": 10,
        "batch_size": 100,
        "request_count": 1000,
        "k": 8,
        "h": 0.125,
        "writer": "PIR-U",
        "alpha_cache": "ON",
        "structural_h": "ON",
        "batch_entry_evaluator_count": 0,
        "terminal_evaluator_only": True,
        "retry_backtracking_line_search": [0, 0, 0],
        "physical_weight_persistence": True,
        "interbatch_w0_restore_count": 0,
        "terminal_w0_restore_count": 1,
        "held_then_release": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    args = parser.parse_args()
    print(json.dumps(build_plan(args.source_head), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
