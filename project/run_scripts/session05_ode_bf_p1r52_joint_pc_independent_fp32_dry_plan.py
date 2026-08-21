#!/usr/bin/env python3
"""Raw-free dry plan for the five independent full-FP32 scheduler cells."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]

from project.run_scripts.ode_bf.p1r52_b100x10_stream import (
    SEAL_FILE,
    verify_p1r52_b100x10_stream,
)
from project.run_scripts.ode_bf.p1r52_joint_pc_independent_fp32_runtime import (
    CELL_LABELS,
    RESULT_NAMES,
    ROLES,
)


def build_plan(source_head: str, source_tree: str) -> dict[str, object]:
    seal = verify_p1r52_b100x10_stream(json.loads(
        (REPO_ROOT / "project/run_scripts/ode_bf/locks" / SEAL_FILE).read_text(
            encoding="utf-8"
        )
    ))
    cells = [
        {
            "array_task": index,
            "cell": label,
            "role": role,
            "methods": ["OFFICIAL-ALPHAEDIT", "OFFICIAL-MEMIT"]
            if index == 0 else [label],
            "independent_case_count": 10,
            "requests_per_case": 100,
            "result_name": RESULT_NAMES[role],
            "model": "llama3-8b-inst",
            "requested_dtype": "torch.float32",
        }
        for index, (label, role) in enumerate(zip(CELL_LABELS, ROLES, strict=True))
    ]
    return {
        "schema": "ode-edit-s05-p1r52-joint-pc-independent-fp32-dry-plan/v1",
        "source_head": source_head,
        "source_tree": source_tree,
        "base_main_head": "0d63ad4ec4978be6d04aabb640e17917bd1348d7",
        "base_main_tree": "652b84489825371558002e24b381dcb777327a7f",
        "stream_root": seal["root_digest"],
        "stream_order": seal["all_request_order_sha256"],
        "batch_order_sha256": list(seal["batch_ordered_request_digest_v1"]),
        "cells": cells,
        "array": "0-4%4",
        "cell_count": 5,
        "method_count": 6,
        "independent_case_count_per_method": 10,
        "scientific_request_count": 6000,
        "baseline_W0_isolation": True,
        "cross_case_sequential_continuity_count": 0,
        "full_fp32_required": True,
        "autocast_bf16_fp16_quantization_allowed": False,
        "project_server1_gpu_cap": 4,
        "logical_server": "server1",
        "scheduler_node": "devbox",
        "timing_observation_only": True,
        "timing_additional_model_forward_backward_evaluator_count": 0,
    }


if __name__ == "__main__":
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True
    ).strip()
    print(json.dumps(build_plan(head, tree), sort_keys=True, separators=(",", ":")))
