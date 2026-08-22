#!/usr/bin/env python3
"""Fail-closed P1R52 Llama J0 IL5 sequential 10xB100 runner."""

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
from project.run_scripts.ode_bf.p1r52_target_depth_sequential_il5 import (
    ATTEMPT_SUFFIX,
    INSTRUCTION_ID,
    POLICY,
    RUN_TOKEN,
)
from project.run_scripts.ode_bf.p1r52_target_depth_sequential_il5_panel import (
    PARENT,
    load_and_validate_lock,
    verify_source_manifest,
)


def _source_ancestry(source_head: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PARENT, source_head],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("P1R52 IL5 sequential source ancestry differs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    args = parser.parse_args(argv)
    try:
        _source_ancestry(args.source_head)
        lock, lock_sha, inherited_lock_sha = load_and_validate_lock(REPO_ROOT)
        manifest_sha = verify_source_manifest(REPO_ROOT, args.source_head)
        stream_path = REPO_ROOT / "project/run_scripts/ode_bf/locks" / SEAL_FILE
        stream = verify_p1r52_b100x10_stream(
            json.loads(stream_path.read_text(encoding="utf-8"))
        )
        result = run_p1(
            repo_root=REPO_ROOT,
            alias="llama3-8b-inst",
            output_root=args.output_root,
            source_head=args.source_head,
            p1r52_sequential_role=POLICY.role,
            p1r52_sequential_scale="b100x10",
            p1r52_attempt_suffix=ATTEMPT_SUFFIX,
            p1r52_sequential_target_depth=POLICY.depth.value,
            p1r52_sequential_target_depth_inner_telemetry=True,
            p1r52_batch_entry_evaluator_enabled=False,
            p1r52_postsolve_energy_warn_enabled=(
                POLICY.postsolve_energy_warn_enabled
            ),
        )
        result.update(
            {
                "il5_lock_sha256": lock_sha,
                "il5_lock_root": lock["root_digest"],
                "inherited_b100x10_lock_sha256": inherited_lock_sha,
                "source_manifest_sha256": manifest_sha,
                "stream_seal_sha256": sha256_file(stream_path),
                "stream_root": stream["root_digest"],
                "stream_order": stream["all_request_order_sha256"],
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
            failure_schema="ode-edit-s05-p1r52-il5-sequential-job-failure/v1",
        )
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
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
