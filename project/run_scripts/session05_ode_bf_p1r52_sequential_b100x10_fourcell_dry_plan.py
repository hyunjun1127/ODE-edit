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
from project.run_scripts.ode_bf.p1r52_sequential_runtime import R52_CONTROL_ROLE, R52_H_ROLE
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    expected_p1r52_sequential_result_name,
)
from project.run_scripts.ode_bf.p1r52_sequential_scale import (
    B100X10_INSTRUCTION_ID,
    P1R52_B100X10_SCALE,
)


def build_plan(
    source_head: str,
    *,
    attempt_suffix: str = "tech-r1",
    r52_only: bool = False,
) -> dict[str, object]:
    if len(source_head) != 40:
        raise ODEBFContractError("P1R52 B100x10 dry-plan source differs")
    roles = (R52_H_ROLE, R52_CONTROL_ROLE) if r52_only else ROLES
    if (attempt_suffix, r52_only) not in (
        ("tech-r1", False),
        ("tech-r2", True),
        ("tech-r3", True),
        ("tech-r3-release-r1", True),
    ):
        raise ODEBFContractError("P1R52 B100x10 dry-plan repair scope differs")
    jobs = [
        {
            "array_index": index,
            "role": role,
            "alias": "llama3-8b-inst",
            "result_name": expected_p1r52_sequential_result_name(
                "llama3-8b-inst",
                role,
                scale=P1R52_B100X10_SCALE,
                attempt_suffix=attempt_suffix,
            ),
            "gpu": 1,
            "cpu": 8,
            "memory_mib": 65000,
        }
        for index, role in enumerate(roles)
    ]
    return {
        "schema": "ode-edit-s05-p1r52-sequential-b100x10-fourcell-dry-plan/v1",
        "instruction_id": B100X10_INSTRUCTION_ID,
        "source_head": source_head,
        "jobs": jobs,
        "job_count": len(jobs),
        "array": "0-1%2" if r52_only else "0-3%4",
        "round_count": P1R52_B100X10_SCALE.round_count,
        "batch_size": P1R52_B100X10_SCALE.batch_size,
        "request_count_per_job": P1R52_B100X10_SCALE.request_count,
        "total_endpoint_count": len(jobs),
        "total_edit_count": len(jobs) * P1R52_B100X10_SCALE.request_count,
        "attempt_suffix": attempt_suffix,
        "repair_scope": "R52_INVALID_CELLS_ONLY" if r52_only else "FOUR_CELL_INITIAL",
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
    parser.add_argument(
        "--attempt-suffix",
        choices=("tech-r1", "tech-r2", "tech-r3", "tech-r3-release-r1"),
        default="tech-r1",
    )
    parser.add_argument("--r52-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build_plan(
                args.source_head,
                attempt_suffix=args.attempt_suffix,
                r52_only=args.r52_only,
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
