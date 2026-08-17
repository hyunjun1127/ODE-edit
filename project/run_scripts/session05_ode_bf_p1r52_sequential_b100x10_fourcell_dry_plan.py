#!/usr/bin/env python3
"""Raw-free four-cell dry plan for P1R52 sequential 10xB100."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1r52_sequential_b100x10_panel import ROLES
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    expected_p1r52_sequential_result_name,
)
from project.run_scripts.ode_bf.p1r52_sequential_scale import (
    B100X10_INSTRUCTION_ID,
    P1R52_B100X10_SCALE,
)


def build_plan(source_head: str) -> dict[str, object]:
    if len(source_head) != 40:
        raise ODEBFContractError("P1R52 B100x10 dry-plan source differs")
    jobs = [
        {
            "array_index": index,
            "role": role,
            "alias": "llama3-8b-inst",
            "result_name": expected_p1r52_sequential_result_name(
                "llama3-8b-inst", role, scale=P1R52_B100X10_SCALE
            ),
            "gpu": 1,
            "cpu": 8,
            "memory_mib": 65000,
        }
        for index, role in enumerate(ROLES)
    ]
    return {
        "schema": "ode-edit-s05-p1r52-sequential-b100x10-fourcell-dry-plan/v1",
        "instruction_id": B100X10_INSTRUCTION_ID,
        "source_head": source_head,
        "jobs": jobs,
        "job_count": 4,
        "array": "0-3%4",
        "round_count": P1R52_B100X10_SCALE.round_count,
        "batch_size": P1R52_B100X10_SCALE.batch_size,
        "request_count_per_job": P1R52_B100X10_SCALE.request_count,
        "total_endpoint_count": 4,
        "total_edit_count": 4 * P1R52_B100X10_SCALE.request_count,
        "history_width_at_entry": list(P1R52_B100X10_SCALE.history_counts),
        "alphaedit_reset_cache_sequence": [True] + [False] * 9,
        "alphaedit_static_projector_separate": True,
        "memit_official_entrypoint": True,
        "physical_weight_persistence": True,
        "interbatch_w0_restore_count": 0,
        "terminal_w0_restore_count": 1,
        "retry_backtracking_imputation": [0, 0, 0],
        "held_then_atomic_release": True,
        "monitor_cadence_after_initial_gate_seconds": 3600,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    args = parser.parse_args()
    print(json.dumps(build_plan(args.source_head), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
