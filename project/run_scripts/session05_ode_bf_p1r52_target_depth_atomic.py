#!/usr/bin/env python3
"""Fail-closed P1R52 IL3-FULL Atomic J0 runner."""

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
from project.run_scripts.ode_bf.p1_runtime import P1OutputRootCollision, run_p1, write_p1_failure_once
from project.run_scripts.ode_bf.p1r52_independent_runtime import ARMS
from project.run_scripts.ode_bf.p1r52_target_depth import P1R52_TARGET_DEPTH_INSTRUCTION_ID, P1R52TargetDepth
from project.run_scripts.ode_bf.p1r52_target_depth_panel import LOCK_FILE, SOURCE_PARENT, load_and_validate_lock


RUN_TOKEN = "p1r52-target-depth-atomic-il3full-v1"
SOURCE_MANIFEST = "source_manifest_s05_p1r52_target_depth_il1_il3full.json"


def _source_gate(source_head: str) -> str:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_PARENT, source_head],
        cwd=REPO_ROOT,
        check=False,
    ).returncode != 0:
        raise ValueError("P1R52 target-depth Atomic source ancestry differs")
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
        raise ValueError("P1R52 target-depth Atomic source manifest header differs")
    for entry in entries:
        path = REPO_ROOT / str(entry["path"])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry["size"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError("P1R52 target-depth Atomic source bytes differ")
    return raw_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--depth", required=True, choices=(P1R52TargetDepth.IL3_FULL.value,))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    args = parser.parse_args(argv)
    try:
        lock, lock_sha = load_and_validate_lock(REPO_ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE)
        source_manifest_sha = _source_gate(args.source_head)
        result = run_p1(
            repo_root=REPO_ROOT,
            alias=args.model,
            output_root=args.output_root,
            source_head=args.source_head,
            p1r52_arm=args.arm,
            p1r52_atomic_target_depth=args.depth,
        )
        result = {
            **result,
            "target_depth_lock_sha256": lock_sha,
            "target_depth_lock_root": lock["root_digest"],
            "source_manifest_sha256": source_manifest_sha,
        }
    except P1OutputRootCollision as exc:
        print(json.dumps({"status": "FAIL_CLOSED_OUTPUT_ROOT_COLLISION", "message_sha256": hashlib.sha256(str(exc).encode()).hexdigest()}), file=sys.stderr)
        return 1
    except Exception as exc:
        failure_sha, failure = write_p1_failure_once(
            args.output_root,
            exc,
            repo_root=REPO_ROOT,
            instruction_id=P1R52_TARGET_DEPTH_INSTRUCTION_ID,
            failure_schema="ode-edit-s05-p1r52-target-depth-atomic-job-failure/v1",
        )
        print(json.dumps({"status": "FAIL_CLOSED", "model": args.model, "arm": args.arm, "exception_class": failure["exception_class"], "exception_message_sha256": failure["exception_message_sha256"], "failure_sha256": failure_sha}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
