#!/usr/bin/env python3
"""Fail-closed Llama Soft Atomic runner for IL8/IL10/IL15."""

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
from project.run_scripts.ode_bf.p1_runtime import (
    P1OutputRootCollision,
    run_p1,
    write_p1_failure_once,
)
from project.run_scripts.ode_bf.p1r52_target_depth_extension import (
    ARM,
    ATTEMPT_SUFFIX,
    EXTENSION_DEPTHS,
    INSTRUCTION_ID,
    MODEL,
)
from project.run_scripts.ode_bf.p1r52_target_depth_extension_panel import (
    LOCK_FILE,
    SOURCE_PARENT,
    load_and_validate_lock,
    verify_immutable_references,
)


RUN_TOKEN = "p1r52-target-depth-atomic-extension-il8-il10-il15-v1"
TECHNICAL_PARENT = "023e192277662f05b10fa3945846b42d21c59aa4"
SOURCE_MANIFEST = (
    "source_manifest_s05_p1r52_target_depth_il8_il10_il15_tech_r2.json"
)


def source_gate(source_head: str) -> str:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", TECHNICAL_PARENT, source_head],
        cwd=REPO_ROOT,
        check=False,
    ).returncode != 0:
        raise ValueError("P1R52 target-depth extension ancestry differs")
    manifest, raw_sha = load_rooted_json(
        REPO_ROOT / "project/run_scripts/ode_bf/locks" / SOURCE_MANIFEST,
        expected_schema=(
            "ode-edit-s05-p1r52-target-depth-extension-source-manifest-tech-r2/v1"
        ),
    )
    entries = manifest.get("entries")
    if (
        manifest.get("instruction_id") != INSTRUCTION_ID
        or manifest.get("source_parent") != SOURCE_PARENT
        or manifest.get("technical_parent") != TECHNICAL_PARENT
        or not isinstance(entries, list)
        or not entries
    ):
        raise ValueError("P1R52 target-depth extension manifest header differs")
    for entry in entries:
        path = REPO_ROOT / str(entry["path"])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry["size"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError("P1R52 target-depth extension source bytes differ")
    return raw_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=(MODEL,))
    parser.add_argument("--arm", required=True, choices=(ARM,))
    parser.add_argument(
        "--depth", required=True, choices=tuple(item.value for item in EXTENSION_DEPTHS)
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    args = parser.parse_args(argv)
    try:
        lock, lock_sha = load_and_validate_lock(
            REPO_ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        source_manifest_sha = source_gate(args.source_head)
        reference_identities = verify_immutable_references(REPO_ROOT)
        result = run_p1(
            repo_root=REPO_ROOT,
            alias=args.model,
            output_root=args.output_root,
            source_head=args.source_head,
            p1r52_arm=args.arm,
            p1r52_atomic_target_depth=args.depth,
            p1r52_attempt_suffix=ATTEMPT_SUFFIX,
        )
        result = {
            **result,
            "target_depth_extension_lock_sha256": lock_sha,
            "target_depth_extension_lock_root": lock["root_digest"],
            "source_manifest_sha256": source_manifest_sha,
            "immutable_reference_file_count": len(reference_identities),
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
            instruction_id=INSTRUCTION_ID,
            failure_schema="ode-edit-s05-p1r52-target-depth-extension-job-failure/v1",
        )
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "model": args.model,
                    "arm": args.arm,
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
