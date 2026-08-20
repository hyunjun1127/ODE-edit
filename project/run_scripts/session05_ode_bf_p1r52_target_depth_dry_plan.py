#!/usr/bin/env python3
"""Deterministic P1R52 target-depth target-only dry plan."""

from __future__ import annotations

import argparse
import json

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1r52_target_depth import P1R52TargetDepth
from project.run_scripts.ode_bf.p1r52_target_depth_target_only_runtime import (
    expected_p1r52_target_depth_result_name,
)


ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")
DEPTHS = (P1R52TargetDepth.IL1, P1R52TargetDepth.IL3_FULL)


def build_plan(
    source_head: str, *, case_count: int, attempt_suffix: str | None = None
) -> dict[str, object]:
    jobs = []
    task_id = 0
    for alias in ALIASES:
        for depth in DEPTHS:
            jobs.append(
                {
                    "task_id": task_id,
                    "alias": alias,
                    "depth": depth.value,
                    "case_count": case_count,
                    "result_name": expected_p1r52_target_depth_result_name(
                        alias,
                        depth=depth,
                        case_count=case_count,
                        attempt_suffix=attempt_suffix,
                    ),
                }
            )
            task_id += 1
    value: dict[str, object] = {
        "schema": "ode-edit-s05-p1r52-target-depth-dry-plan/v1",
        "source_head": source_head,
        "case_count_per_cell": case_count,
        "cell_count": 4,
        "request_attempt_count": 4 * case_count * 10,
        "jobs": jobs,
        "stream_root": "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6",
        "all_request_order_sha256": "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c",
        "writer_materialization_count": 0,
        "attempt_suffix": attempt_suffix,
    }
    value["identity_sha256"] = canonical_hash(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--case-count", required=True, type=int, choices=(1, 10))
    parser.add_argument("--attempt-suffix")
    args = parser.parse_args()
    print(
        json.dumps(
            build_plan(
                args.source_head,
                case_count=args.case_count,
                attempt_suffix=args.attempt_suffix,
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
