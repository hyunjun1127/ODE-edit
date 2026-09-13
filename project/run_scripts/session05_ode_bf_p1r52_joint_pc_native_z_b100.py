#!/usr/bin/env python3
"""Fail-closed same-B100 Official native-z diagnostic runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.artifacts import sha256_file
from project.run_scripts.ode_bf.p1_runtime import (
    P1OutputRootCollision,
    run_p1,
    write_p1_failure_once,
)
from project.run_scripts.ode_bf.p1r52_b100x10_stream import (
    SEAL_FILE,
    verify_p1r52_b100x10_stream,
)
from project.run_scripts.ode_bf.p1r52_joint_pc_native_z_b100 import INSTRUCTION_ID
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    JOINT_PC_ALPHAEDIT_NATIVE_Z_ROLE,
    JOINT_PC_MEMIT_NATIVE_Z_ROLE,
    expected_p1r52_sequential_result_name,
)
from project.run_scripts.ode_bf.p1r52_sequential_scale import P1R52_B100X10_SCALE


RUN_TOKEN = "p1r52-joint-pc-native-z-b100-v1"
ROLES = (JOINT_PC_ALPHAEDIT_NATIVE_Z_ROLE, JOINT_PC_MEMIT_NATIVE_Z_ROLE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    args = parser.parse_args(argv)
    try:
        stream_path = REPO_ROOT / "project/run_scripts/ode_bf/locks" / SEAL_FILE
        stream = verify_p1r52_b100x10_stream(
            json.loads(stream_path.read_text(encoding="utf-8"))
        )
        result = run_p1(
            repo_root=REPO_ROOT,
            alias="llama3-8b-inst",
            output_root=args.output_root,
            source_head=args.source_head,
            p1r52_sequential_role=args.role,
            p1r52_sequential_scale="b100x10",
            p1r52_batch_entry_evaluator_enabled=False,
        )
        result.update(
            {
                "instruction_id": INSTRUCTION_ID,
                "stream_seal_sha256": sha256_file(stream_path),
                "stream_root": stream["root_digest"],
                "stream_order": stream["all_request_order_sha256"],
                "expected_result_name": expected_p1r52_sequential_result_name(
                    "llama3-8b-inst", args.role, scale=P1R52_B100X10_SCALE
                ),
            }
        )
    except P1OutputRootCollision as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED_OUTPUT_ROOT_COLLISION",
                    "message_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
                }
            ),
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        failure_sha, failure = write_p1_failure_once(
            args.output_root,
            exc,
            repo_root=REPO_ROOT,
            instruction_id=INSTRUCTION_ID,
            failure_schema="ode-edit-s05-p1r52-joint-pc-native-z-b100-failure/v1",
        )
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "role": args.role,
                    "exception_class": failure["exception_class"],
                    "exception_message_sha256": failure["exception_message_sha256"],
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
