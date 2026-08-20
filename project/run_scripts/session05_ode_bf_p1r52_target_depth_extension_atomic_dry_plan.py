#!/usr/bin/env python3
"""Raw-free dry plan for the IL8/IL10/IL15 Llama Soft extension."""

from __future__ import annotations

import json
import subprocess

from project.run_scripts.ode_bf.p1r52_target_depth_extension import (
    INSTRUCTION_ID,
    PROJECT_GPU_CAP,
    extension_cells,
    validate_extension_depths,
)


def build_plan(source_head: str) -> dict[str, object]:
    cells = extension_cells()
    validate_extension_depths(str(cell["depth"]) for cell in cells)
    return {
        "schema": "ode-edit-s05-p1r52-target-depth-extension-dry-plan/v1",
        "instruction_id": INSTRUCTION_ID,
        "source_head": source_head,
        "comparison_il1": "IMMUTABLE_P1R52_REPAIR_R1_SAME_STREAM",
        "comparison_il3": "IMMUTABLE_P1R52_PHASE1B_LLAMA_SOFT_10_OF_10",
        "new_gpu_cells": list(cells),
        "nominal_array": "0-2%3",
        "project_gpu_cap": PROJECT_GPU_CAP,
        "stage_gpu_cell_count": 3,
        "writer": "P1R52_ORIGINAL_J0_FROZEN",
        "inner_h": 0.125,
        "h_rescale_count": 0,
        "il5_action_count": 0,
        "inner_telemetry": "ACCEPTED_Z_REWRITE_REPHRASE_OBSERVATION_ONLY",
        "inner_telemetry_added_backward_count": 0,
        "inner_telemetry_added_generation_count": 0,
        "inner_telemetry_action_influence_count": 0,
        "duplicate_evaluation_count": 0,
        "scientific_promotion": False,
    }


if __name__ == "__main__":
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    print(json.dumps(build_plan(head), sort_keys=True, separators=(",", ":")))
