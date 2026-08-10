#!/usr/bin/env python3
"""Fail-closed entry for one P1R17 Dynamic FullDrive trajectory."""

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
from project.run_scripts.ode_bf.full_drive_dynamic_runtime import FullDriveCell
from project.run_scripts.ode_bf.p1_full_drive_dynamic_panel import (
    FULL_DRIVE_INSTRUCTION_ID,
    FULL_DRIVE_RESULT_TOKEN,
)
from project.run_scripts.ode_bf.p1_runtime import (
    P1OutputRootCollision,
    run_p1,
    write_p1_failure_once,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--cell", required=True, choices=tuple(item.value for item in FullDriveCell))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True, choices=(FULL_DRIVE_RESULT_TOKEN,))
    args = parser.parse_args(argv)
    if len(args.source_head) != 40 or any(char not in "0123456789abcdef" for char in args.source_head):
        return 2
    try:
        result = run_p1(
            repo_root=REPO_ROOT,
            alias=args.model,
            output_root=args.output_root,
            source_head=args.source_head,
            full_drive_cell=args.cell,
        )
    except P1OutputRootCollision:
        return 2
    except Exception as exc:
        failure_sha256, failure = write_p1_failure_once(
            args.output_root,
            exc,
            repo_root=REPO_ROOT,
            instruction_id=FULL_DRIVE_INSTRUCTION_ID,
            failure_schema="ode-edit-s05-p1r17-full-drive-failure/v1",
        )
        print(json.dumps({
            "status": "FAIL_CLOSED",
            "exception_class": failure["exception_class"],
            "exception_message_sha256": failure["exception_message_sha256"],
            "last_completed_stage": failure["last_completed_stage"],
            "failure_sha256": failure_sha256,
        }, sort_keys=True, separators=(",", ":")), file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
