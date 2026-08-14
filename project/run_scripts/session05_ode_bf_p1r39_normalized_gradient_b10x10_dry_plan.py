#!/usr/bin/env python3
"""No-model dry plan for the two-cell P1R39-A1 Soft extension."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.p1r24_independent_b10x10_selection import verify_historical_h0_fresh_seal
from project.run_scripts.ode_bf.p1r39_independent_b10x10_panel import LOCK_FILE, expected_result_name, load_and_validate_lock


CELLS = (
    ("llama3-8b-inst", "PR-P1R39-NORMALIZED-GRADIENT-SOFT"),
    ("qwen2.5-7b-inst", "PR-P1R39-NORMALIZED-GRADIENT-SOFT"),
)


def build_plan(source_head: str, *, repository_root: Path = REPO_ROOT) -> dict[str, object]:
    locks = repository_root / "project/run_scripts/ode_bf/locks"
    lock, lock_sha = load_and_validate_lock(locks / LOCK_FILE)
    seal = verify_historical_h0_fresh_seal(
        json.loads((locks / "p1r24_independent_b10x10_stream_seal.json").read_text(encoding="utf-8"))
    )
    if lock["fresh_stream_root"] != seal["root_digest"] or lock["all_request_order_sha256"] != seal["all_request_order_sha256"]:
        raise RuntimeError("P1R39 lock/seal differs")
    jobs = [
        {
            "array_index": index,
            "alias": alias,
            "method": method,
            "result_name": expected_result_name(alias, method),
            "case_count": 10,
            "request_count_per_case": 10,
            "gpu": 1,
            "cpu": 8,
            "memory_mib": 65000,
            "time": "23:59:00",
        }
        for index, (alias, method) in enumerate(CELLS)
    ]
    return {
        "schema": "ode-edit-s05-p1r39-a1-normalized-gradient-soft-b10x10-dry-plan/v1",
        "source_head": source_head,
        "p1r39_lock_sha256": lock_sha,
        "p1r39_lock_root": lock["root_digest"],
        "fresh_stream_root": seal["root_digest"],
        "all_request_order_sha256": seal["all_request_order_sha256"],
        "job_count": 2,
        "cell_count": 2,
        "independent_atomic_b10_case_count": 20,
        "request_attempt_count": 200,
        "routing_arm": "SOFT",
        "target_or_demand_attenuation_count": 0,
        "history_mode": "OFF",
        "project_gpu_cap": 4,
        "array_max_concurrent_gpu": 2,
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
    print(json.dumps(build_plan(args.source_head), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
