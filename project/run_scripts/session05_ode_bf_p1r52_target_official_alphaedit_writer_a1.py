#!/usr/bin/env python3
"""Fail-closed P1R52 target + Official AlphaEdit writer Phase-A runner."""

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
from project.run_scripts.ode_bf.p1_runtime import P1OutputRootCollision, run_p1, write_p1_failure_once
from project.run_scripts.ode_bf.p1r52_target_official_alphaedit_writer import (
    INSTRUCTION_ID,
    PHASE_A_ROLE,
)
from project.run_scripts.ode_bf.p1r52_target_official_alphaedit_writer_panel import (
    LOCK_FILE,
    PARENT,
    SOURCE_MANIFEST_FILE,
    SOURCE_MANIFEST_SCHEMA,
    load_and_validate_lock,
)


RUN_TOKEN = "p1r52-target-official-alphaedit-writer-a1-phase-a-v1"


def _source_gate(source_head: str) -> str:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PARENT, source_head],
        cwd=REPO_ROOT,
        check=False,
    ).returncode != 0:
        raise ValueError("target/Official writer source ancestry differs")
    manifest, raw_sha = load_rooted_json(
        REPO_ROOT / "project/run_scripts/ode_bf/locks" / SOURCE_MANIFEST_FILE,
        expected_schema=SOURCE_MANIFEST_SCHEMA,
    )
    if (
        manifest.get("instruction_id") != INSTRUCTION_ID
        or manifest.get("parent") != PARENT
        or not isinstance(manifest.get("entries"), list)
        or not manifest["entries"]
    ):
        raise ValueError("target/Official writer source manifest header differs")
    observed: list[str] = []
    for entry in manifest["entries"]:
        relative = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(relative, str):
            raise ValueError("target/Official writer source entry differs")
        path = REPO_ROOT / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry.get("size")
            or sha256_file(path) != entry.get("sha256")
        ):
            raise ValueError("target/Official writer source bytes differ")
        observed.append(relative)
    if observed != sorted(set(observed)):
        raise ValueError("target/Official writer source manifest ordering differs")
    return raw_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    args = parser.parse_args(argv)
    try:
        lock, lock_sha = load_and_validate_lock(
            REPO_ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        source_manifest_sha = _source_gate(args.source_head)
        result = run_p1(
            repo_root=REPO_ROOT,
            alias="llama3-8b-inst",
            output_root=args.output_root,
            source_head=args.source_head,
            p1r52_sequential_role=PHASE_A_ROLE,
            p1r52_sequential_scale="b100x10",
            p1r52_batch_entry_evaluator_enabled=False,
        )
        result.update(
            {
                "numerical_lock_sha256": lock_sha,
                "numerical_lock_root": lock["root_digest"],
                "source_manifest_sha256": source_manifest_sha,
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
            failure_schema="ode-edit-s05-p1r52-target-official-alphaedit-writer-a1-failure/v1",
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
