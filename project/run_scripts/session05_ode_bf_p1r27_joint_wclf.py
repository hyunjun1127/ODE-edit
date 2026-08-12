#!/usr/bin/env python3
"""Fail-closed P1R27 joint W-CLF entry."""

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
from project.run_scripts.ode_bf.p1r27_joint_wclf import P1R27_INSTRUCTION_ID
from project.run_scripts.ode_bf.p1r27_joint_wclf_panel import P1R27_ROLES


RUN_TOKEN = "p1r27-joint-wclf-ztracking-semantic-bf-v1"


def _source_manifest_gate(source_head: str) -> str:
    path = REPO_ROOT / "project/run_scripts/ode_bf/locks/source_manifest_s05_p1r27_joint_wclf.json"
    value, raw_sha = load_rooted_json(
        path,
        expected_schema="ode-edit-s05-p1r27-joint-wclf-source-manifest/v1",
    )
    entries = value.get("entries")
    if (
        value.get("instruction_id") != P1R27_INSTRUCTION_ID
        or value.get("base_checkpoint")
        != "ce8c6c36348752f1407f7d713d30e6b5c727379b"
        or not isinstance(entries, list)
        or not entries
    ):
        raise ValueError("P1R27 source manifest header differs")
    observed: list[str] = []
    for entry in entries:
        relative = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(relative, str):
            raise ValueError("P1R27 source manifest entry differs")
        source = REPO_ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError("P1R27 manifested source differs")
        if source.stat().st_size != entry.get("size") or sha256_file(source) != entry.get("sha256"):
            raise ValueError("P1R27 manifested source bytes differ")
        observed.append(relative)
    if observed != sorted(set(observed)):
        raise ValueError("P1R27 source manifest ordering differs")
    changed = set(
        subprocess.run(
            ["git", "diff", "--name-only", value["base_checkpoint"], source_head],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.splitlines()
    )
    manifest_relative = path.relative_to(REPO_ROOT).as_posix()
    if changed != set(observed) | {manifest_relative}:
        raise ValueError("P1R27 source manifest closure differs")
    return raw_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--role", required=True, choices=P1R27_ROLES)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    args = parser.parse_args(argv)
    try:
        numerical, numerical_sha = load_rooted_json(
            REPO_ROOT / "project/run_scripts/ode_bf/locks/numerical_lock_s05_p1r27_joint_wclf.json",
            expected_schema="ode-edit-s05-p1r27-joint-wclf-lock/v1",
        )
        if numerical.get("instruction_id") != P1R27_INSTRUCTION_ID:
            raise ValueError("P1R27 numerical lock differs")
        source_sha = _source_manifest_gate(args.source_head)
        result = run_p1(
            repo_root=REPO_ROOT,
            alias=args.model,
            output_root=args.output_root,
            source_head=args.source_head,
            joint_wclf_role=args.role,
        )
        result = {
            **result,
            "p1r27_numerical_lock_sha256": numerical_sha,
            "p1r27_source_manifest_sha256": source_sha,
        }
    except P1OutputRootCollision as exc:
        print(json.dumps({"status": "FAIL_CLOSED_OUTPUT_ROOT_COLLISION", "message_sha256": hashlib.sha256(str(exc).encode()).hexdigest()}), file=sys.stderr)
        return 1
    except Exception as exc:
        failure_sha, failure = write_p1_failure_once(
            args.output_root,
            exc,
            repo_root=REPO_ROOT,
            instruction_id=P1R27_INSTRUCTION_ID,
            failure_schema="ode-edit-s05-p1r27-joint-wclf-failure/v1",
        )
        print(json.dumps({"status": "FAIL_CLOSED", "model": args.model, "role": args.role, "exception_class": failure["exception_class"], "exception_message_sha256": failure["exception_message_sha256"], "failure_sha256": failure_sha}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
