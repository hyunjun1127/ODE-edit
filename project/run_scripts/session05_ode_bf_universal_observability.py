#!/usr/bin/env python3
"""Fail-closed entry point for one R13 universal-observability cell pair."""

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
from project.run_scripts.ode_bf.ode_bf_observability import R13_LIVE_CELL_IDS
from project.run_scripts.ode_bf.p1_runtime import (
    P1OutputRootCollision,
    run_p1,
    write_p1_failure_once,
)
from project.run_scripts.ode_bf.p1_universal_observability_panel import (
    UNIVERSAL_OBS_INSTRUCTION_ID,
    UNIVERSAL_OBS_RESULT_TOKEN,
)


def _valid_source_head(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="session05-ode-bf-universal-observability",
        allow_abbrev=False,
    )
    parser.add_argument("--model", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--cell", required=True, choices=R13_LIVE_CELL_IDS)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument(
        "--run-token", required=True, choices=(UNIVERSAL_OBS_RESULT_TOKEN,)
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not _valid_source_head(args.source_head):
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED_INVALID_SOURCE_HEAD",
                    "instruction_id": UNIVERSAL_OBS_INSTRUCTION_ID,
                    "model_alias": args.model,
                    "cell_id": args.cell,
                    "failure_receipt_written": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    try:
        # The runtime expands this one cell into its locked Dynamic/Hold pair.
        result = run_p1(
            repo_root=REPO_ROOT,
            alias=args.model,
            output_root=args.output_root,
            source_head=args.source_head,
            universal_observability_cell=args.cell,
        )
    except P1OutputRootCollision as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED_OUTPUT_ROOT_COLLISION",
                    "instruction_id": UNIVERSAL_OBS_INSTRUCTION_ID,
                    "model_alias": args.model,
                    "cell_id": args.cell,
                    "exception_class": type(exc).__name__,
                    "exception_message_sha256": hashlib.sha256(
                        str(exc).encode("utf-8")
                    ).hexdigest(),
                    "failure_receipt_written": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    except Exception as exc:
        failure_sha256, failure = write_p1_failure_once(
            args.output_root,
            exc,
            repo_root=REPO_ROOT,
            instruction_id=UNIVERSAL_OBS_INSTRUCTION_ID,
            failure_schema=(
                "ode-edit-s05-universal-obs-rs-bg-targethold-p1r13-a2-failure/v1"
            ),
        )
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "instruction_id": UNIVERSAL_OBS_INSTRUCTION_ID,
                    "model_alias": args.model,
                    "cell_id": args.cell,
                    "exception_class": failure["exception_class"],
                    "exception_message_sha256": failure[
                        "exception_message_sha256"
                    ],
                    "last_completed_stage": failure["last_completed_stage"],
                    "failure_sha256": failure_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
