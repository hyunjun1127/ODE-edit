#!/usr/bin/env python3
"""Raw-free two-job P1R52 sequential dry plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1r52_sequential_contract import dry_plan
from project.run_scripts.ode_bf.p1r52_sequential_runtime import ROLES, expected_p1r52_sequential_result_name


def build_plan(source_head: str) -> dict[str, object]:
    if len(source_head) != 40:
        raise ODEBFContractError("P1R52 sequential dry-plan source differs")
    plan = dry_plan()
    jobs = [
        {
            "array_index": index,
            "role": role,
            "alias": "llama3-8b-inst",
            "result_name": expected_p1r52_sequential_result_name("llama3-8b-inst", role),
            "gpu": 1,
            "cpu": 8,
            "memory_mib": 65000,
        }
        for index, role in enumerate(ROLES)
    ]
    return {
        **plan,
        "schema": "ode-edit-s05-p1r52-sequential-dry-plan/v1",
        "source_head": source_head,
        "jobs": jobs,
        "job_count": 2,
        "array": "0-1%2",
        "held_then_atomic_release": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    args = parser.parse_args()
    print(json.dumps(build_plan(args.source_head), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
