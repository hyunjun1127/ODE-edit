#!/usr/bin/env python3
"""Deterministic no-model plan for the one-shot P2R6 capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.p2r6_replay_capture import (
    CAPTURE_ALIAS,
    CAPTURE_ARM,
    CAPTURE_CASE_INDEX,
    CAPTURE_INSTRUCTION_ID,
    CAPTURE_OUTER_STEP,
    CAPTURE_SHADOW_ARM,
    TECH_R8_SCIENTIFIC_HEAD,
)


def build_plan(source_head: str) -> dict[str, object]:
    return {
        "schema": "ode-edit-s05-p2r6-red-r1-capture-a1-dry-plan/v1",
        "instruction_id": CAPTURE_INSTRUCTION_ID,
        "scientific_source_head": TECH_R8_SCIENTIFIC_HEAD,
        "instrumentation_source_head": source_head,
        "model": CAPTURE_ALIAS,
        "case_index": CAPTURE_CASE_INDEX,
        "selected_arm": CAPTURE_ARM,
        "failing_shadow_arm": CAPTURE_SHADOW_ARM,
        "capture_outer_step": CAPTURE_OUTER_STEP,
        "job_count": 1,
        "gpu_count": 1,
        "cpu_count": 8,
        "memory_mib": 65000,
        "node": "devbox",
        "project_gpu_cap_server1": 4,
        "capture_invocation_budget": 1,
        "red_amendment_influence_count": 0,
        "terminal_evaluator_access_count": 0,
        "scientific_endpoint_count": 0,
        "intentional_stop_before_shadow_solve": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head")
    args = parser.parse_args()
    head = args.source_head or subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    print(json.dumps(build_plan(head), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
