#!/usr/bin/env python3
"""Fail-closed entry for P1R23 Compute-A1 atomic paired cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.artifacts import load_rooted_json
from project.run_scripts.ode_bf.compute_progress_simplex_runtime import (
    COMPUTE_PROGRESS_SIMPLEX_INSTRUCTION_ID,
    COMPUTE_PROGRESS_SIMPLEX_METHOD_ID,
)
from project.run_scripts.ode_bf.contracts import MODEL_ALIASES, ODEBFContractError
from project.run_scripts.ode_bf.p1_runtime import (
    P1OutputRootCollision,
    run_p1,
    write_p1_failure_once,
)
from project.run_scripts.ode_bf.p1_scalable_batched_runtime_panel import (
    P1R23_COMPUTE_A1_ATTEMPT_NAMESPACES,
)


RUN_TOKEN = "p1r23-compute-progress-simplex-a1-v1"
ROLES = ("COMPUTE_PROGRESS_SIMPLEX_BG_PAIR", "COMPUTE_PROGRESS_SIMPLEX_RS_PAIR")
LOCK = REPO_ROOT / "project/run_scripts/ode_bf/locks/numerical_lock_s05_compute_progress_simplex.json"


def _validate_lock() -> str:
    value, sha256 = load_rooted_json(
        LOCK, expected_schema="ode-edit-s05-p1r23-compute-progress-simplex-lock/v1"
    )
    if (
        value.get("instruction_id") != COMPUTE_PROGRESS_SIMPLEX_INSTRUCTION_ID
        or value.get("method_id") != COMPUTE_PROGRESS_SIMPLEX_METHOD_ID
        or value.get("layer_order") != [4, 5, 6, 7, 8]
        or value.get("execution", {}).get("trajectory_count") != 8
        or value.get("execution", {}).get("B100_access_count") != 0
    ):
        raise ODEBFContractError("compute-progress-simplex numerical lock differs")
    return sha256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    parser.add_argument(
        "--attempt-namespace",
        choices=P1R23_COMPUTE_A1_ATTEMPT_NAMESPACES,
    )
    args = parser.parse_args(argv)
    if len(args.source_head) != 40:
        return 2
    try:
        lock_sha = _validate_lock()
        result = run_p1(
            repo_root=REPO_ROOT,
            alias=args.model,
            output_root=args.output_root,
            source_head=args.source_head,
            scalable_batched_role=args.role,
            scalable_batched_batch_size=10,
            scalable_batched_attempt_namespace=args.attempt_namespace,
        )
        result = {**result, "compute_progress_simplex_lock_sha256": lock_sha}
    except P1OutputRootCollision as exc:
        print(json.dumps({
            "status": "FAIL_CLOSED_OUTPUT_ROOT_COLLISION",
            "exception_message_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
        }, sort_keys=True), file=sys.stderr)
        return 1
    except Exception as exc:
        failure_sha, failure = write_p1_failure_once(
            args.output_root,
            exc,
            repo_root=REPO_ROOT,
            instruction_id=COMPUTE_PROGRESS_SIMPLEX_INSTRUCTION_ID,
            failure_schema="ode-edit-s05-p1r23-compute-progress-simplex-failure/v1",
        )
        print(json.dumps({
            "status": "FAIL_CLOSED",
            "model": args.model,
            "role": args.role,
            "exception_class": failure["exception_class"],
            "failure_sha256": failure_sha,
        }, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
