#!/usr/bin/env python3
"""Fail-closed entry for one atomic P1R23 execution role."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import MODEL_ALIASES
from project.run_scripts.ode_bf.p1_runtime import (
    P1OutputRootCollision,
    run_p1,
    write_p1_failure_once,
)
from project.run_scripts.ode_bf.p1_scalable_batched_runtime_panel import (
    P1R23_BATCH_SIZES,
    P1R23_EXECUTION_ROLES,
)
from project.run_scripts.ode_bf.scalable_batched_runtime import (
    P1R23_INSTRUCTION_ID,
)


P1R23_RUN_TOKEN = "scalable-batched-runtime-p1r23-v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--batch-size", required=True, type=int, choices=P1R23_BATCH_SIZES)
    parser.add_argument("--role", required=True, choices=P1R23_EXECUTION_ROLES)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True, choices=(P1R23_RUN_TOKEN,))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if len(args.source_head) != 40 or any(
        item not in "0123456789abcdef" for item in args.source_head
    ):
        return 2
    try:
        result = run_p1(
            repo_root=REPO_ROOT,
            alias=args.model,
            output_root=args.output_root,
            source_head=args.source_head,
            scalable_batched_role=args.role,
            scalable_batched_batch_size=args.batch_size,
        )
    except P1OutputRootCollision as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED_OUTPUT_ROOT_COLLISION",
                    "instruction_id": P1R23_INSTRUCTION_ID,
                    "exception_message_sha256": hashlib.sha256(
                        str(exc).encode("utf-8")
                    ).hexdigest(),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        failure_sha, failure = write_p1_failure_once(
            args.output_root,
            exc,
            repo_root=REPO_ROOT,
            instruction_id=P1R23_INSTRUCTION_ID,
            failure_schema="ode-edit-s05-p1r23-scalable-batched-failure/v1",
        )
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "model": args.model,
                    "batch_size": args.batch_size,
                    "role": args.role,
                    "exception_class": failure["exception_class"],
                    "exception_message_sha256": failure[
                        "exception_message_sha256"
                    ],
                    "failure_sha256": failure_sha,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
