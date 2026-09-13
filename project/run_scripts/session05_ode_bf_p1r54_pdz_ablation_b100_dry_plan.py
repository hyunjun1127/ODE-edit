#!/usr/bin/env python3
"""No-model dry plan for the P1R54 PDZ ablation array."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1r54_pdz_ablation import PDZAblationArm, schedule_for_arm
from project.run_scripts.ode_bf.p1r54_pdz_ablation_b100 import (
    arm_for_role,
    expected_result_name,
    role_for_cell,
)


def build_plan(*, concurrency: int = 3) -> dict[str, object]:
    if concurrency not in (1, 2, 3):
        raise ValueError("P1R54 PDZ ablation concurrency differs")
    cells = []
    for index in range(4):
        role = role_for_cell(index)
        arm = arm_for_role(role)
        schedule = schedule_for_arm(arm)
        cells.append(
            {
                "cell": index,
                "role": role,
                "arm": arm.value,
                "result_name": expected_result_name(role),
                "outer_count": 8,
                "microsteps_per_outer": schedule.microsteps_per_outer,
                "target_dt": float(schedule.target_dt),
                "target_horizon": float(schedule.target_horizon),
                "field_evaluations": schedule.total_field_evaluations,
                "writer_calls": 8,
                "writer_layer_applies": 40,
            }
        )
    payload: dict[str, object] = {
        "schema": "ode-edit-s05-p1r54-pdz-ablation-dry-plan/v1",
        "array": "0-3",
        "array_concurrency": concurrency,
        "gpu_per_task": 1,
        "selected_batch": "B1",
        "request_count_per_cell": 100,
        "cells": cells,
        "reference_execution_count": 0,
        "model_load_count": 0,
        "project_gpu_cap": 3,
        "scientific_submit_hold": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(build_plan(), sort_keys=True))
