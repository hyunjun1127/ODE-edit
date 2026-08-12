#!/usr/bin/env python3
"""Fail-closed P1R26 ASDC entry."""

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
from project.run_scripts.ode_bf.p1r26_asdc import P1R26_INSTRUCTION_ID
from project.run_scripts.ode_bf.p1r26_asdc_panel import (
    LOCK_FILE,
    P1R26_ROLES,
    load_and_validate_p1r26_lock,
)


RUN_TOKEN = "p1r26-asdc-strong-z-atomic-v1"
SOURCE_MANIFEST = "source_manifest_s05_p1r26_asdc.json"


def _source_manifest_gate(source_head: str) -> str:
    manifest_path = REPO_ROOT / "project/run_scripts/ode_bf/locks" / SOURCE_MANIFEST
    manifest, raw_sha = load_rooted_json(
        manifest_path,
        expected_schema="ode-edit-s05-p1r26-asdc-source-manifest/v1",
    )
    entries = manifest.get("entries")
    if (
        manifest.get("instruction_id") != P1R26_INSTRUCTION_ID
        or manifest.get("scientific_base")
        != "ce8c6c36348752f1407f7d713d30e6b5c727379b"
        or not isinstance(entries, list)
        or not entries
    ):
        raise ValueError("P1R26 source manifest header differs")
    observed: list[str] = []
    for entry in entries:
        relative = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(relative, str):
            raise ValueError("P1R26 source manifest entry differs")
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("P1R26 source path differs")
        if path.stat().st_size != entry.get("size") or sha256_file(path) != entry.get("sha256"):
            raise ValueError("P1R26 source bytes differ")
        observed.append(relative)
    manifest_relative = manifest_path.relative_to(REPO_ROOT).as_posix()
    changed = set(
        subprocess.run(
            ["git", "diff", "--name-only", manifest["scientific_base"], source_head],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.splitlines()
    )
    if observed != sorted(set(observed)) or changed != set(observed) | {manifest_relative}:
        raise ValueError("P1R26 source closure differs")
    return raw_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--role", required=True, choices=P1R26_ROLES)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    args = parser.parse_args(argv)
    try:
        _lock, lock_sha = load_and_validate_p1r26_lock(
            REPO_ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        source_manifest_sha = _source_manifest_gate(args.source_head)
        result = run_p1(
            repo_root=REPO_ROOT,
            alias=args.model,
            output_root=args.output_root,
            source_head=args.source_head,
            p1r26_asdc_role=args.role,
        )
        result = {
            **result,
            "p1r26_numerical_lock_sha256": lock_sha,
            "p1r26_source_manifest_sha256": source_manifest_sha,
        }
    except P1OutputRootCollision as exc:
        print(json.dumps({"status": "FAIL_CLOSED_OUTPUT_ROOT_COLLISION", "message_sha256": hashlib.sha256(str(exc).encode()).hexdigest()}), file=sys.stderr)
        return 1
    except Exception as exc:
        failure_sha, failure = write_p1_failure_once(
            args.output_root,
            exc,
            repo_root=REPO_ROOT,
            instruction_id=P1R26_INSTRUCTION_ID,
            failure_schema="ode-edit-s05-p1r26-asdc-failure/v1",
        )
        print(json.dumps({"status": "FAIL_CLOSED", "model": args.model, "role": args.role, "exception_class": failure["exception_class"], "exception_message_sha256": failure["exception_message_sha256"], "failure_sha256": failure_sha}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
