#!/usr/bin/env python3
"""Fail-closed entry point for the S05 R12 BG-SOFT missing-cell run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.p1_bg_soft_missing_cell_panel import (
    BG_SOFT_AMENDMENT_ID,
    BG_SOFT_INSTRUCTION_ID,
    BG_SOFT_RESULT_TOKEN,
)
from project.run_scripts.ode_bf.p1_runtime import (
    P1OutputRootCollision,
    run_p1,
    write_p1_failure_once,
)


LLAMA_ALIAS = "llama3-8b-inst"


def _valid_source_head(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=(LLAMA_ALIAS,))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True, choices=(BG_SOFT_RESULT_TOKEN,))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not _valid_source_head(args.source_head):
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED_INVALID_SOURCE_HEAD",
                    "instruction_id": BG_SOFT_INSTRUCTION_ID,
                    "amendment_id": BG_SOFT_AMENDMENT_ID,
                    "model_alias": args.model,
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
        result = run_p1(
            repo_root=REPO_ROOT,
            alias=args.model,
            output_root=args.output_root,
            source_head=args.source_head,
            bg_soft_missing_cell_mode=True,
        )
    except P1OutputRootCollision as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED_OUTPUT_ROOT_COLLISION",
                    "instruction_id": BG_SOFT_INSTRUCTION_ID,
                    "model_alias": args.model,
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
            instruction_id=BG_SOFT_INSTRUCTION_ID,
            failure_schema=(
                "ode-edit-s05-bg-soft-missing-cell-p1r12-a1-failure/v1"
            ),
        )
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "instruction_id": BG_SOFT_INSTRUCTION_ID,
                    "amendment_id": BG_SOFT_AMENDMENT_ID,
                    "model_alias": args.model,
                    "exception_class": failure["exception_class"],
                    "exception_message_sha256": failure["exception_message_sha256"],
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
