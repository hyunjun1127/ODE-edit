#!/usr/bin/env python3
"""Raw-free dry plan for P1R52 B100 accepted-z observation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1r52_accepted_z_observation_panel import (
    INSTRUCTION_ID,
    ROLES,
    expected_result_name,
    sealed_reference_root,
)


def build_plan(source_head: str) -> dict[str, object]:
    if len(source_head) != 40:
        raise ODEBFContractError("accepted-z observation dry-plan source differs")
    jobs = [
        {
            "array_index": index,
            "role": role,
            "alias": "llama3-8b-inst",
            "result_name": expected_result_name(role),
            "sealed_reference_root": str(sealed_reference_root(role)),
            "gpu": 1,
            "cpu": 8,
            "memory_mib": 65000,
        }
        for index, role in enumerate(ROLES)
    ]
    return {
        "schema": "ode-edit-s05-p1r52-b100-accepted-z-observation-dry-plan/v1",
        "instruction_id": INSTRUCTION_ID,
        "source_head": source_head,
        "jobs": jobs,
        "job_count": 4,
        "array": "0-3%4",
        "round_count": 10,
        "batch_size": 100,
        "request_count_per_job": 1000,
        "total_observed_request_count": 4000,
        "batch_entry_W_metrics_count": 0,
        "added_backward_count": 0,
        "added_generation_call_count": 0,
        "action_influence_count": 0,
        "duplicate_W_evaluator_model_forward_count": 0,
        "physical_edit_replay_required": True,
        "sealed_commit_hash_gate_per_batch": True,
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
