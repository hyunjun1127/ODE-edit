#!/usr/bin/env python3
"""Fail-closed R1 entry for the adaptive-tau P1R4 causal diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import MODEL_ALIASES
from project.run_scripts.ode_bf.p1_adaptive import ADAPTIVE_INSTRUCTION_ID
from project.run_scripts.ode_bf.p1_adaptive_runtime import ADAPTIVE_RESULT_TOKEN
from project.run_scripts.ode_bf.p1_runtime import run_p1, write_p1_failure_once


REPAIR_INSTRUCTION_ID = "ODEEDIT-S04-ODE-BF-P1R4-HISTORY-VIEW-R1-V1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="session04-ode-bf-p1r4-adaptive-tau-r1", allow_abbrev=False
    )
    parser.add_argument("--model", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument(
        "--run-token", required=True, choices=(ADAPTIVE_RESULT_TOKEN,)
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_p1(
            repo_root=REPO_ROOT,
            alias=args.model,
            output_root=args.output_root,
            source_head=args.source_head,
            adaptive_mode=True,
        )
    except Exception as exc:
        failure_sha256, failure = write_p1_failure_once(
            args.output_root,
            exc,
            repo_root=REPO_ROOT,
        )
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED_NO_RETRY",
                    "instruction_id": REPAIR_INSTRUCTION_ID,
                    "scientific_instruction_id": ADAPTIVE_INSTRUCTION_ID,
                    "model_alias": args.model,
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
