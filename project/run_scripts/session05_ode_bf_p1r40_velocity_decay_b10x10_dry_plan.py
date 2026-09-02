#!/usr/bin/env python3
"""No-model dry plan for P1R40 four-cell independent B10x10."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.p1r24_independent_b10x10_selection import (
    verify_historical_h0_fresh_seal,
)
from project.run_scripts.ode_bf.p1r40_independent_b10x10_panel import (
    LOCK_FILE,
    expected_result_name,
    load_and_validate_lock,
)


CELLS = (
    ("llama3-8b-inst", "SDVD-P1R38-NEUTRAL"),
    ("llama3-8b-inst", "SDVD-P1R38-SOFT"),
    ("qwen2.5-7b-inst", "SDVD-P1R38-NEUTRAL"),
    ("qwen2.5-7b-inst", "SDVD-P1R38-SOFT"),
)


def build_plan(
    source_head: str,
    *,
    repository_root: Path = REPO_ROOT,
    attempt_suffix: str | None = None,
) -> dict[str, object]:
    locks = repository_root / "project/run_scripts/ode_bf/locks"
    lock, lock_sha = load_and_validate_lock(locks / LOCK_FILE)
    seal = verify_historical_h0_fresh_seal(
        json.loads(
            (locks / "p1r24_independent_b10x10_stream_seal.json").read_text(
                encoding="utf-8"
            )
        )
    )
    if (
        lock["fresh_stream_root"] != seal["root_digest"]
        or lock["all_request_order_sha256"]
        != seal["all_request_order_sha256"]
    ):
        raise RuntimeError("P1R40 lock/seal differs")
    jobs = [
        {
            "array_index": index,
            "alias": alias,
            "method": method,
            "result_name": expected_result_name(
                alias, method, attempt_suffix=attempt_suffix
            ),
            "case_count": 10,
            "request_count_per_case": 10,
            "grid_count": 8,
            "gpu": 1,
            "cpu": 8,
            "memory_mib": 60416,
            "time": "23:59:00",
        }
        for index, (alias, method) in enumerate(CELLS)
    ]
    return {
        "schema": "ode-edit-s05-p1r40-semantic-deficit-velocity-decay-dry-plan/v1",
        "source_head": source_head,
        "attempt_suffix": attempt_suffix,
        "p1r40_lock_sha256": lock_sha,
        "p1r40_lock_root": lock["root_digest"],
        "fresh_stream_root": seal["root_digest"],
        "all_request_order_sha256": seal["all_request_order_sha256"],
        "job_count": 4,
        "cell_count": 4,
        "independent_atomic_b10_case_count": 40,
        "request_attempt_count": 400,
        "maximum_accepted_step_count": 320,
        "allocation_factorial": "NONE",
        "history_mode": "OFF",
        "project_gpu_cap": 4,
        "array_max_concurrent_gpu": 4,
        "model_load": False,
        "gpu_use": False,
        "slurm_submit": False,
        "result_root_creation": False,
        "jobs": jobs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_plan(args.source_head), sort_keys=True, separators=(",", ":")
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
