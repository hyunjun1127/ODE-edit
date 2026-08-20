#!/usr/bin/env python3
"""Raw-free Stage A/Stage B dry plan for PIR-U cache continuity A1."""

from __future__ import annotations

import argparse
import json

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1r52_piru_cache_continuity import (
    CACHE_COMPLETE_ROLE,
    INSTRUCTION_ID,
    LEGACY_ROLE,
    METHOD_ID,
    RESULT_NAMES,
    STAGE_A_RESULT_NAMES,
)


def build_plan(source_head: str, stage: str) -> dict[str, object]:
    if len(source_head) != 40 or stage not in {"A-B10", "B"}:
        raise ODEBFContractError("PIR-U cache-continuity dry plan differs")
    arms = (
        [("cache-complete", CACHE_COMPLETE_ROLE, STAGE_A_RESULT_NAMES[CACHE_COMPLETE_ROLE])]
        if stage == "A-B10"
        else [
            ("legacy", LEGACY_ROLE, RESULT_NAMES[LEGACY_ROLE]),
            ("cache-complete", CACHE_COMPLETE_ROLE, RESULT_NAMES[CACHE_COMPLETE_ROLE]),
        ]
    )
    return {
        "schema": "ode-edit-s05-p1r52-piru-cache-continuity-a1-dry-plan/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "source_head": source_head,
        "stage": stage,
        "jobs": [
            {
                "array_index": index,
                "arm": arm,
                "role": role,
                "alias": "llama3-8b-inst",
                "result_name": result_name,
                "gpu": 1,
                "cpu": 8,
                "memory_mib": 65000,
            }
            for index, (arm, role, result_name) in enumerate(arms)
        ],
        "job_count": len(arms),
        "round_count": 10,
        "batch_size": 100,
        "request_count": 1000,
        "k": 8,
        "h": 0.125,
        "stage_a_cache_complete_rounds": [10] if stage == "A-B10" else [],
        "stage_b_cache_complete_rounds": list(range(1, 11)) if stage == "B" else [],
        "writer": "PIR-U",
        "coefficient_delta_count": 0,
        "additional_target_slope_forward_backward": [0, 0],
        "batch_entry_evaluator_count": 0,
        "postsolve_energy_residual": "WARN_CONTINUE",
        "retry_backtracking_line_search": [0, 0, 0],
        "terminal_w0_restore_count": 1,
        "held_then_release": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--stage", required=True, choices=("A-B10", "B"))
    args = parser.parse_args()
    print(json.dumps(build_plan(args.source_head, args.stage), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
