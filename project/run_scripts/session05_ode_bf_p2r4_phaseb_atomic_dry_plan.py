#!/usr/bin/env python3
"""No-model P2R4 Phase-B clamp ON/OFF paired-writer dry plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.p1r24_independent_b10x10_selection import verify_historical_h0_fresh_seal
from project.run_scripts.ode_bf.p2r4_phaseb_atomic_panel import LOCK_FILE, load_and_validate_lock
from project.run_scripts.ode_bf.p2r4_phaseb_atomic_runtime import expected_p2r4_phaseb_result_name


ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")
CLAMP_POLICIES = ("ON", "OFF")


def build_plan(
    source_head: str,
    *,
    case_count: int,
    attempt_suffix: str | None = None,
    repository_root: Path = REPO_ROOT,
) -> dict[str, object]:
    locks = repository_root / "project/run_scripts/ode_bf/locks"
    lock, lock_sha = load_and_validate_lock(locks / LOCK_FILE)
    seal = verify_historical_h0_fresh_seal(
        json.loads((locks / "p1r24_independent_b10x10_stream_seal.json").read_text())
    )
    if (
        case_count not in (1, 10)
        or lock["stream_root"] != seal["root_digest"]
        or lock["all_request_order_sha256"] != seal["all_request_order_sha256"]
    ):
        raise RuntimeError("P2R4 Phase-B dry-plan input differs")
    jobs = [
        {
            "array_index": index,
            "alias": alias,
            "clamp_policy": clamp_policy,
            "arms": ["NEUTRAL", "SOFTP"],
            "result_name": expected_p2r4_phaseb_result_name(
                alias,
                clamp_policy=clamp_policy,
                case_count=case_count,
                attempt_suffix=attempt_suffix,
            ),
            "case_count_per_arm": case_count,
            "request_count_per_case": 10,
            "target_microstep_count_per_case_arm": 24,
            "writer_transition_count_per_case_arm": 8,
            "gpu": 1,
            "cpu": 8,
            "memory_mib": 65000,
            "time": "23:59:00",
        }
        for index, (alias, clamp_policy) in enumerate(
            (alias, clamp_policy)
            for alias in ALIASES
            for clamp_policy in CLAMP_POLICIES
        )
    ]
    return {
        "schema": "ode-edit-s05-p2r4-phaseb-atomic-dry-plan/v1",
        "source_head": source_head,
        "case_count_per_alias_arm": case_count,
        "lock_sha256": lock_sha,
        "lock_root": lock["root_digest"],
        "stream_root": seal["root_digest"],
        "all_request_order_sha256": seal["all_request_order_sha256"],
        "job_count": 4,
        "arm_count": 8,
        "attempt_count": 8 * case_count,
        "request_attempt_count": 80 * case_count,
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
    parser.add_argument("--case-count", required=True, type=int, choices=(1, 10))
    parser.add_argument("--attempt-suffix")
    args = parser.parse_args()
    print(json.dumps(build_plan(args.source_head, case_count=args.case_count, attempt_suffix=args.attempt_suffix), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
