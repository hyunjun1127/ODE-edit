#!/usr/bin/env python3
"""Dry plan for the P1R52 joint-P/C pilot or production."""

from __future__ import annotations

import argparse
import json

from project.run_scripts.ode_bf.p1r52_joint_pc_execution import INSTRUCTION_ID
from project.run_scripts.ode_bf.p1r52_joint_pc_runtime import (
    PILOT_ROLE,
    PILOT_TECH_R1_ROLE,
    PILOT_TECH_R2_ROLE,
    PILOT_TECH_R3_ROLE,
    PILOT_TECH_R4_ROLE,
    STREAM_ORDER,
    STREAM_ROOT,
    expected_result_name,
    production_role,
)


def build_plan(source_head: str, *, stage: str) -> dict[str, object]:
    roles = (
        (PILOT_ROLE,)
        if stage == "pilot"
        else (PILOT_TECH_R1_ROLE,)
        if stage == "pilot-tech-r1"
        else (PILOT_TECH_R2_ROLE,)
        if stage == "pilot-tech-r2"
        else (PILOT_TECH_R3_ROLE,)
        if stage == "pilot-tech-r3"
        else (PILOT_TECH_R4_ROLE,)
        if stage == "pilot-tech-r4"
        else tuple(production_role(i) for i in range(1, 11))
    )
    return {
        "schema": "ode-edit-s05-p1r52-joint-pc-c1-c2-b100-dry-plan/v1",
        "instruction_id": INSTRUCTION_ID,
        "source_head": source_head,
        "stage": stage,
        "model": "llama3-8b-inst",
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "case_count": len(roles),
        "request_count_per_case": 100,
        "target_compute_count_per_case": 1,
        "writer_arms_per_case": ["C0-PIRU-CONTROL", "C1-JOINT-PC-REMAINING", "C2-JOINT-PC-FIXED-QUOTA"],
        "jobs": [
            {
                "array_index": index,
                "role": role,
                "case_index": 1 if role in (PILOT_ROLE, PILOT_TECH_R1_ROLE, PILOT_TECH_R2_ROLE, PILOT_TECH_R3_ROLE, PILOT_TECH_R4_ROLE) else index + 1,
                "result_name": expected_result_name(role),
            }
            for index, role in enumerate(roles)
        ],
        "max_concurrent_gpu": 1 if stage.startswith("pilot") else 3,
        "gpu_per_task": 1,
        "callback_count": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--stage", choices=("pilot", "pilot-tech-r1", "pilot-tech-r2", "pilot-tech-r3", "pilot-tech-r4", "production"), required=True)
    args = parser.parse_args()
    print(json.dumps(build_plan(args.source_head, stage=args.stage), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
