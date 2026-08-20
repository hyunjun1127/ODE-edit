#!/usr/bin/env python3
"""Fail-closed P1R52 IL1/IL3-FULL target-only runner."""

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

from project.run_scripts.ode_bf.artifacts import load_rooted_json, sha256_file
from project.run_scripts.ode_bf.contracts import MODEL_ALIASES
from project.run_scripts.ode_bf.p1_runtime import (
    P1OutputRootCollision,
    run_p1,
    write_p1_failure_once,
)
from project.run_scripts.ode_bf.p1r52_target_depth import (
    P1R52_TARGET_DEPTH_INSTRUCTION_ID,
    P1R52TargetDepth,
)
from project.run_scripts.ode_bf.p1r52_target_depth_panel import (
    LOCK_FILE,
    SOURCE_PARENT,
    load_and_validate_lock,
)


RUN_TOKEN = "p1r52-target-depth-il1-il3full-v1"
SOURCE_MANIFEST = "source_manifest_s05_p1r52_target_depth_il1_il3full.json"


def _source_gate(source_head: str) -> str:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_PARENT, source_head],
        cwd=REPO_ROOT,
        check=False,
    ).returncode != 0:
        raise ValueError("P1R52 target-depth source ancestry differs")
    manifest, raw_sha = load_rooted_json(
        REPO_ROOT / "project/run_scripts/ode_bf/locks" / SOURCE_MANIFEST,
        expected_schema="ode-edit-s05-p1r52-target-depth-source-manifest/v1",
    )
    entries = manifest.get("entries")
    if (
        manifest.get("instruction_id") != P1R52_TARGET_DEPTH_INSTRUCTION_ID
        or manifest.get("source_parent") != SOURCE_PARENT
        or not isinstance(entries, list)
        or not entries
    ):
        raise ValueError("P1R52 target-depth source manifest header differs")
    observed: list[str] = []
    for entry in entries:
        relative = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(relative, str):
            raise ValueError("P1R52 target-depth source manifest entry differs")
        path = REPO_ROOT / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry.get("size")
            or sha256_file(path) != entry.get("sha256")
        ):
            raise ValueError("P1R52 target-depth source bytes differ")
        observed.append(relative)
    if observed != sorted(set(observed)):
        raise ValueError("P1R52 target-depth source manifest ordering differs")
    return raw_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=MODEL_ALIASES)
    parser.add_argument(
        "--depth", required=True, choices=tuple(item.value for item in P1R52TargetDepth)
    )
    parser.add_argument("--case-count", required=True, type=int, choices=(1, 10))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    parser.add_argument("--attempt-suffix")
    args = parser.parse_args(argv)
    try:
        lock, lock_sha = load_and_validate_lock(
            REPO_ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        source_manifest_sha = _source_gate(args.source_head)
        result = run_p1(
            repo_root=REPO_ROOT,
            alias=args.model,
            output_root=args.output_root,
            source_head=args.source_head,
            p1r52_target_depth=args.depth,
            p1r52_target_depth_case_count=args.case_count,
            p1r52_target_depth_attempt_suffix=args.attempt_suffix,
        )
        result = {
            **result,
            "p1r52_target_depth_lock_sha256": lock_sha,
            "p1r52_target_depth_lock_root": lock["root_digest"],
            "source_manifest_sha256": source_manifest_sha,
        }
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
            instruction_id=P1R52_TARGET_DEPTH_INSTRUCTION_ID,
            failure_schema="ode-edit-s05-p1r52-target-depth-job-failure/v1",
        )
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "model": args.model,
                    "depth": args.depth,
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
