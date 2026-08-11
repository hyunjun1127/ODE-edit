#!/usr/bin/env python3
"""No-model dry plan for eight exact Progress-Simplex Sequential-NoH cells."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.historical_h0_sequential_runtime import METHODS, expected_historical_h0_result_name
from project.run_scripts.ode_bf.historical_h0_sequential_selection import verify_historical_h0_fresh_seal
from project.run_scripts.ode_bf.p1_historical_h0_sequential_panel import load_and_validate_historical_h0_lock


ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")


def build_plan(source_head: str, *, repository_root: Path = REPO_ROOT) -> dict[str, object]:
    locks = repository_root / "project/run_scripts/ode_bf/locks"
    seal = verify_historical_h0_fresh_seal(json.loads((locks / "p1r20_historical_h0_fresh_cf_b100_seal.json").read_text(encoding="utf-8")))
    numerical, numerical_sha = load_and_validate_historical_h0_lock(locks / "numerical_lock_s05_historical_h0_sequential.json")
    if numerical["fresh_seal_root"] != seal["root_digest"] or numerical["all_request_order_sha256"] != seal["all_request_order_sha256"]:
        raise RuntimeError("P1R20 lock/seal differs")
    jobs = []
    for alias in ALIASES:
        for method in METHODS:
            jobs.append({"array_index": len(jobs), "alias": alias, "method": method, "result_name": expected_historical_h0_result_name(alias, method), "gpu": 1, "cpu": 8, "memory_mib": 65000, "time": "23:59:00"})
    return {"schema": "ode-edit-s05-p1r23-progress-simplex-sequential-noh-dry-plan/v1", "source_head": source_head, "fresh_seal_root": seal["root_digest"], "all_request_order_sha256": seal["all_request_order_sha256"], "numerical_lock_sha256": numerical_sha, "numerical_lock_root": numerical["root_digest"], "trajectory_count": 8, "ode_trajectory_count": 8, "model_level_alphaedit_rerun_count": 0, "array_max_concurrent_gpu": 4, "sequential_round_count": 10, "postcommit_cumulative_b10_evaluation_count_per_trajectory": 55, "history_mode": "OFF", "jobs": jobs, "model_load": False, "gpu_use": False, "slurm_submit": False, "result_root_creation": False}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    args = parser.parse_args()
    print(json.dumps(build_plan(args.source_head), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
