#!/usr/bin/env python3
"""Raw-free one-cell dry plan for PIR-U post-energy-WARN R1."""

from __future__ import annotations

import argparse
import json

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1r52_piru_postenergy_warn import (
    INSTRUCTION_ID,
    METHOD_ID,
    RESULT_NAME,
)
from project.run_scripts.ode_bf.p1r52_piru_sequential_adapter import (
    P1R52_PIRU_SEQUENTIAL_ROLE,
)


def build_plan(source_head: str) -> dict[str, object]:
    if len(source_head) != 40:
        raise ODEBFContractError("PIR-U post-energy-WARN source differs")
    return {
        "schema": "ode-edit-s05-p1r52-piru-postenergy-warn-dry-plan/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "source_head": source_head,
        "jobs": [{
            "array_index": 0,
            "role": P1R52_PIRU_SEQUENTIAL_ROLE,
            "alias": "llama3-8b-inst",
            "result_name": RESULT_NAME,
            "gpu": 1,
            "cpu": 8,
            "memory_mib": 65000,
        }],
        "job_count": 1,
        "round_count": 10,
        "batch_size": 100,
        "request_count": 1000,
        "k": 8,
        "h": 0.125,
        "writer": "PIR-U",
        "alpha_cache": "ON",
        "structural_h_optimizer_energy_constraint": "UNCHANGED_ACTIVE",
        "postsolve_energy_residual": "WARN_CONTINUE",
        "batch_entry_evaluator_enabled": False,
        "batch_entry_evaluator_count": 0,
        "accepted_z_observation": "INLINE_AFTER_ACTION_FREEZE",
        "accepted_z_added_backward_generation_action": [0, 0, 0],
        "duplicate_physical_w_evaluator_forward_count": 0,
        "actual_concentration_primary": "SQRT_REALIZED_BF16_STEP_ENERGY_SHARE",
        "retry_backtracking_line_search": [0, 0, 0],
        "physical_weight_persistence": True,
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
