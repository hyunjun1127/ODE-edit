#!/usr/bin/env python3
"""Fail-closed P1R31 P1R24 full-matrix independent whole-B10 runner."""

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
from project.run_scripts.ode_bf.p1r24_independent_b10x10_panel import (
    LOCK_FILE,
    PARENT,
    load_and_validate_lock,
)
from project.run_scripts.ode_bf.p1r24_independent_b10x10_runtime import INSTRUCTION_ID
from project.run_scripts.ode_bf.p1r24_independent_b10x10_runtime import METHODS


RUN_TOKEN = "p1r31-p1r24-independent-b10x10-detailed-v1"
SOURCE_MANIFEST = "source_manifest_s05_p1r31_p1r24_independent_b10x10_detailed.json"
PROTECTED_SCIENTIFIC_SHA256 = {
    "project/run_scripts/ode_bf/p1r24_atomic_strength.py": "95ef3aff1e0b2d82df08453492a01be4d69d7e1c8795edd72c7d098979f54b0f",
    "project/run_scripts/ode_bf/p1r24_atomic_strength_panel.py": "0c46b34a0b6f998ce4c0f8b52c18b9e829d44ce102a41ec37d14a22cc8604c64",
    "project/run_scripts/ode_bf/locks/numerical_lock_s05_p1r24_atomic_strength.json": "b7b0d1903a461c75c8d6497e5b7cddb788500e9c009dd6fd5628e3f13842ab6d",
}


def _source_gate(source_head: str) -> str:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PARENT, source_head],
        cwd=REPO_ROOT,
        check=False,
    ).returncode != 0:
        raise ValueError("P1R31 P1R24 scientific ancestry differs")
    for relative, expected in PROTECTED_SCIENTIFIC_SHA256.items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise ValueError("P1R24 protected scientific source differs")
    manifest_path = REPO_ROOT / "project/run_scripts/ode_bf/locks" / SOURCE_MANIFEST
    manifest, raw_sha = load_rooted_json(
        manifest_path,
        expected_schema="ode-edit-s05-p1r31-p1r24-independent-b10x10-detailed-source-manifest/v1",
    )
    entries = manifest.get("entries")
    if (
        manifest.get("instruction_id") != INSTRUCTION_ID
        or manifest.get("parent_checkpoint") != PARENT
        or not isinstance(entries, list)
        or not entries
    ):
        raise ValueError("P1R24 independent source manifest header differs")
    observed: list[str] = []
    for entry in entries:
        relative = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(relative, str):
            raise ValueError("P1R24 independent source entry differs")
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("P1R24 independent source path differs")
        if path.stat().st_size != entry.get("size") or sha256_file(path) != entry.get("sha256"):
            raise ValueError("P1R24 independent source bytes differ")
        observed.append(relative)
    manifest_relative = manifest_path.relative_to(REPO_ROOT).as_posix()
    changed = set(
        subprocess.run(
            ["git", "diff", "--name-only", PARENT, source_head],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.splitlines()
    )
    if observed != sorted(set(observed)) or changed != set(observed) | {manifest_relative}:
        raise ValueError("P1R24 independent source closure differs")
    return raw_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    args = parser.parse_args(argv)
    try:
        lock, lock_sha = load_and_validate_lock(
            REPO_ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        if sha256_file(
            REPO_ROOT / "project/run_scripts/ode_bf/locks/numerical_lock_s05_p1r24_atomic_strength.json"
        ) != lock["p1r24_numerical_lock_sha256"]:
            raise ValueError("P1R24 frozen numerical lock differs")
        source_manifest_sha = _source_gate(args.source_head)
        result = run_p1(
            repo_root=REPO_ROOT,
            alias=args.model,
            output_root=args.output_root,
            source_head=args.source_head,
            p1r24_independent_b10x10_method=args.method,
        )
        result = {
            **result,
            "p1r24_independent_lock_sha256": lock_sha,
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
            instruction_id=INSTRUCTION_ID,
            failure_schema="ode-edit-s05-p1r24-rs-soft-independent-b10x10-job-failure/v1",
        )
        print(json.dumps({"status": "FAIL_CLOSED", "model": args.model, "exception_class": failure["exception_class"], "exception_message_sha256": failure["exception_message_sha256"], "failure_sha256": failure_sha}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
