#!/usr/bin/env python3
"""Fail-closed P1R38 four-cell independent whole-B10 runner."""

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
from project.run_scripts.ode_bf.p1r38_independent_b10x10_panel import LOCK_FILE, PARENT, load_and_validate_lock
from project.run_scripts.ode_bf.p1r38_independent_b10x10_runtime import B1_METHODS, INSTRUCTION_ID, METHODS


RUN_TOKEN = "p1r38-pr-p1r35-perrequest-b10x10-v1"
SOURCE_MANIFEST = "source_manifest_s05_p1r38_pr_p1r35_perrequest_target.json"
TECHNICAL_PARENT = "87efd168fc9680cabde878cd3b595e98db66d8fa"
PROTECTED_P1R35_SHA256 = {
    "project/run_scripts/ode_bf/p1r35_full_current_residual.py": "539425cc1e6bbe66cc873f4c177e8587d3e5dc955fdb081b1d909b0dc8ce617b",
    "project/run_scripts/ode_bf/p1r34_w_anchored_finite_demand.py": "b2c59ac75798b9d70fbbc370c6d9f805d33cbeb8f683be81bb9effe7d1dbfe53",
    "project/run_scripts/ode_bf/atomic_runtime_optimization.py": "672f700f3f8fcaac230d4f5857b22d305786e776bc9474737b69b931f39d3cd0",
}


def _source_gate(source_head: str) -> str:
    for ancestor in (PARENT, TECHNICAL_PARENT):
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, source_head],
            cwd=REPO_ROOT,
            check=False,
        ).returncode != 0:
            raise ValueError("P1R38 source ancestry differs")
    for relative, expected in PROTECTED_P1R35_SHA256.items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise ValueError("P1R38 protected P1R35 source differs")
    manifest_path = REPO_ROOT / "project/run_scripts/ode_bf/locks" / SOURCE_MANIFEST
    manifest, raw_sha = load_rooted_json(
        manifest_path,
        expected_schema="ode-edit-s05-p1r38-pr-p1r35-perrequest-source-manifest/v1",
    )
    entries = manifest.get("entries")
    if (
        manifest.get("instruction_id") != INSTRUCTION_ID
        or manifest.get("technical_parent") != TECHNICAL_PARENT
        or not isinstance(entries, list)
        or not entries
    ):
        raise ValueError("P1R38 source manifest header differs")
    observed: list[str] = []
    for entry in entries:
        relative = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(relative, str):
            raise ValueError("P1R38 source entry differs")
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("P1R38 source path differs")
        if path.stat().st_size != entry.get("size") or sha256_file(path) != entry.get("sha256"):
            raise ValueError("P1R38 source bytes differ")
        observed.append(relative)
    manifest_relative = manifest_path.relative_to(REPO_ROOT).as_posix()
    changed = set(
        subprocess.run(
            ["git", "diff", "--name-only", TECHNICAL_PARENT, source_head],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.splitlines()
    )
    if observed != sorted(set(observed)) or changed != set(observed) | {manifest_relative}:
        raise ValueError("P1R38 source closure differs")
    return raw_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--method", required=True, choices=METHODS + B1_METHODS)
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
            p1r38_independent_b10x10_method=args.method,
            p1r38_attempt_suffix=args.attempt_suffix,
        )
        result = {
            **result,
            "p1r38_lock_sha256": lock_sha,
            "p1r38_lock_root": lock["root_digest"],
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
            failure_schema="ode-edit-s05-p1r38-perrequest-b10x10-job-failure/v1",
        )
        print(json.dumps({"status": "FAIL_CLOSED", "model": args.model, "exception_class": failure["exception_class"], "exception_message_sha256": failure["exception_message_sha256"], "failure_sha256": failure_sha}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
