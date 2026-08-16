#!/usr/bin/env python3
"""Fail-closed P1R35 full-current-residual Atomic entry."""

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
from project.run_scripts.ode_bf.p1r35_full_current_residual import (
    P1R35_INSTRUCTION_ID,
)
from project.run_scripts.ode_bf.p1r35_full_current_residual_panel import P1R35_ROLES

RUN_TOKEN = "p1r35-full-current-residual-v1"
BASE = "ac321d86a634a20888cb57003a65467d40aa5ddd"


def _source_manifest_gate(source_head: str) -> str:
    path = REPO_ROOT / "project/run_scripts/ode_bf/locks/source_manifest_s05_p1r35_full_current_residual.json"
    manifest, raw_sha = load_rooted_json(
        path,
        expected_schema="ode-edit-s05-p1r35-full-current-residual-source-manifest/v1",
    )
    entries = manifest.get("entries")
    if (
        manifest.get("instruction_id") != P1R35_INSTRUCTION_ID
        or manifest.get("base_checkpoint") != BASE
        or not entries
    ):
        raise ValueError("P1R35 source manifest header differs")
    observed: list[str] = []
    for entry in entries:
        relative = str(entry.get("path"))
        candidate = REPO_ROOT / relative
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.stat().st_size != entry.get("size")
            or sha256_file(candidate) != entry.get("sha256")
        ):
            raise ValueError("P1R35 source manifest entry differs")
        observed.append(relative)
    if observed != sorted(set(observed)):
        raise ValueError("P1R35 source manifest order differs")
    changed = set(
        subprocess.run(
            ["git", "diff", "--name-only", BASE, source_head],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.splitlines()
    )
    if changed != set(observed) | {path.relative_to(REPO_ROOT).as_posix()}:
        raise ValueError("P1R35 source manifest closure differs")
    return raw_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--role", required=True, choices=P1R35_ROLES)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    args = parser.parse_args(argv)
    try:
        lock, lock_sha = load_rooted_json(
            REPO_ROOT / "project/run_scripts/ode_bf/locks/numerical_lock_s05_p1r35_full_current_residual.json",
            expected_schema="ode-edit-s05-p1r35-full-current-residual-lock/v1",
        )
        if lock.get("instruction_id") != P1R35_INSTRUCTION_ID:
            raise ValueError("P1R35 numerical lock differs")
        manifest_sha = _source_manifest_gate(args.source_head)
        result = run_p1(
            repo_root=REPO_ROOT,
            alias=args.model,
            output_root=args.output_root,
            source_head=args.source_head,
            atomic_strength_recovery_role=args.role,
        )
        result = {
            **result,
            "p1r35_numerical_lock_sha256": lock_sha,
            "p1r35_source_manifest_sha256": manifest_sha,
        }
    except P1OutputRootCollision:
        print(json.dumps({"status": "FAIL_CLOSED_OUTPUT_ROOT_COLLISION"}), file=sys.stderr)
        return 1
    except Exception as exc:
        failure_sha, failure = write_p1_failure_once(
            args.output_root,
            exc,
            repo_root=REPO_ROOT,
            instruction_id=P1R35_INSTRUCTION_ID,
            failure_schema="ode-edit-s05-p1r35-full-current-residual-failure/v1",
        )
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "model": args.model,
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
