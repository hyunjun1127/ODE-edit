#!/usr/bin/env python3
"""Fail-closed P3R1 one-case launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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
from project.run_scripts.ode_bf.p3r1_fixed_m_target import INSTRUCTION_ID
from project.run_scripts.ode_bf.p3r1_runtime import (
    case_role,
    expected_result_name,
    expected_result_parent,
)


SOURCE_FILES = (
    "project/run_scripts/ode_bf/p1_runtime.py",
    "project/run_scripts/ode_bf/p1r24_atomic_strength.py",
    "project/run_scripts/ode_bf/p1r52_accepted_z_observation.py",
    "project/run_scripts/ode_bf/p1r52_sequential_runtime.py",
    "project/run_scripts/ode_bf/p3r1_fixed_m_target.py",
    "project/run_scripts/ode_bf/p3r1_finite_horizon.py",
    "project/run_scripts/ode_bf/p3r1_runtime.py",
    "project/run_scripts/session05_ode_bf_p3r1_fastz_fh.py",
    "project/run_scripts/session05_ode_bf_p3r1_fastz_fh.sbatch",
    "project/run_scripts/session05_submit_p3r1_fastz_fh.py",
)


def source_manifest() -> dict[str, object]:
    entries = [
        {
            "path": name,
            "bytes": (REPO_ROOT / name).stat().st_size,
            "sha256": sha256_file(REPO_ROOT / name),
        }
        for name in SOURCE_FILES
    ]
    payload: dict[str, object] = {
        "schema": "ode-edit-s05-p3r1-source-manifest/v1",
        "instruction_id": INSTRUCTION_ID,
        "entries": entries,
    }
    payload["identity_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--alias", required=True, choices=("llama3-8b-inst", "qwen2.5-7b-inst"))
    parser.add_argument("--case-index", required=True, type=int, choices=range(1, 11))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    args = parser.parse_args(argv)
    role = case_role(args.case_index)
    try:
        expected_output = expected_result_parent(REPO_ROOT) / expected_result_name(
            args.alias, role
        )
        if args.output_root.resolve(strict=False) != expected_output:
            raise ValueError("P3R1 TECH-R1 output namespace differs")
        observed = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        if observed != args.source_head:
            raise ValueError("P3R1 source head differs")
        stream = verify_p1r52_b100x10_stream(
            json.loads(
                (REPO_ROOT / "project/run_scripts/ode_bf/locks" / SEAL_FILE).read_text(
                    encoding="utf-8"
                )
            )
        )
        result = run_p1(
            repo_root=REPO_ROOT,
            alias=args.alias,
            output_root=args.output_root,
            source_head=args.source_head,
            p1r52_sequential_role=role,
            p1r52_sequential_scale="b100x10",
            p1r52_batch_entry_evaluator_enabled=False,
        )
        result.update(
            {
                "stream_root": stream["root_digest"],
                "stream_order": stream["all_request_order_sha256"],
                "expected_result_name": expected_result_name(args.alias, role),
                "source_manifest": source_manifest(),
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
            failure_schema="ode-edit-s05-p3r1-case-failure/v1",
        )
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "exception_class": failure["exception_class"],
                    "failure_sha256": failure_sha,
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
