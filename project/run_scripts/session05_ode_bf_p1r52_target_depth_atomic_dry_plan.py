#!/usr/bin/env python3
"""Raw-free Phase-1B IL3-FULL Atomic dry plan."""

from __future__ import annotations

import json


def build_plan(source_head: str) -> dict[str, object]:
    jobs = []
    for task, (model, arm) in enumerate((
        ("llama3-8b-inst", "neutral"),
        ("llama3-8b-inst", "soft"),
        ("qwen2.5-7b-inst", "neutral"),
        ("qwen2.5-7b-inst", "soft"),
    )):
        jobs.append({
            "array_task": task,
            "model": model,
            "arm": arm,
            "depth": "IL3-FULL",
            "case_count": 10,
            "request_attempt_count": 100,
            "result_name": f"s05-p1r52-target-depth-atomic-b10x10-{model}-{arm}-il3full-v1",
        })
    return {
        "schema": "ode-edit-s05-p1r52-target-depth-phase1b-dry-plan/v1",
        "source_head": source_head,
        "comparison_il1": "IMMUTABLE_P1R52_REPAIR_R1_GLOBAL_REPORT",
        "new_gpu_cells": jobs,
        "array": "0-3%4",
        "project_gpu_cap": 4,
        "stage_gpu_max": 4,
        "writer": "P1R52_ORIGINAL_J0_FROZEN",
        "scientific_promotion": False,
    }


if __name__ == "__main__":
    import subprocess
    head = subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True, stdout=subprocess.PIPE).stdout.strip()
    print(json.dumps(build_plan(head), sort_keys=True, separators=(",", ":")))
