#!/usr/bin/env python3
"""Raw-free one-cell dry plan for P1R52 IL5 sequential B1000."""

from __future__ import annotations

import argparse
import json

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1r52_target_depth_sequential_il5 import (
    INSTRUCTION_ID,
    POLICY,
    RESULT_NAME,
)


def build_plan(source_head: str) -> dict[str, object]:
    if len(source_head) != 40:
        raise ODEBFContractError("P1R52 IL5 dry-plan source differs")
    return {
        "schema": "ode-edit-s05-p1r52-il5-sequential-dry-plan/v1",
        "instruction_id": INSTRUCTION_ID,
        "source_head": source_head,
        "jobs": [
            {
                "cell": 0,
                "alias": "llama3-8b-inst",
                "role": POLICY.role,
                "target_depth": POLICY.depth.value,
                "result_name": RESULT_NAME,
                "gpu": 1,
                "cpu": 8,
                "memory_mib": 65000,
            }
        ],
        "job_count": 1,
        "target_depth": POLICY.depth.value,
        "round_count": POLICY.round_count,
        "batch_size": POLICY.batch_size,
        "request_count": POLICY.request_count,
        "outer_k": POLICY.outer_count,
        "inner_count_per_outer": POLICY.depth.inner_count,
        "inner_h": POLICY.inner_h,
        "inner_rows_per_batch": POLICY.expected_inner_rows_per_batch,
        "request_inner_rows_per_batch": POLICY.expected_request_inner_rows_per_batch,
        "writer": "P1R52-J0",
        "alpha_cache": "ON",
        "structural_h": "ON",
        "postsolve_energy_warn_enabled": POLICY.postsolve_energy_warn_enabled,
        "postsolve_energy_decision_influence_count": 0,
        "postsolve_energy_tolerance": 1e-12,
        "postsolve_coefficient_shrink_count": 0,
        "postsolve_retry_count": 0,
        "batch_entry_evaluator_count": 0,
        "inner_writer_materialization_count": 0,
        "inner_heldout_controller_access_count": 0,
        "outer_writer_materialization_count": 1,
        "physical_weight_persistence": True,
        "retry_backtracking_imputation": [0, 0, 0],
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
