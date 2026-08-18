#!/usr/bin/env python3
"""Fail-closed four-cell P1R52 B100 accepted-z observation runner."""

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
from project.run_scripts.ode_bf.p1r52_accepted_z_observation_panel import (
    INSTRUCTION_ID,
    LOCK_FILE,
    MANIFEST_FILE,
    MANIFEST_SCHEMA,
    REFERENCE_TERMINAL_SHA256,
    ROLES,
    SOURCE_PARENT,
    load_and_validate_lock,
    sealed_reference_root,
)
from project.run_scripts.ode_bf.p1r52_b100x10_stream import (
    SEAL_FILE,
    verify_p1r52_b100x10_stream,
)
from project.run_scripts.ode_bf.p1r52_official_sequential_baseline_panel import (
    verify_memit_artifacts,
)
from project.run_scripts.ode_bf.p1r52_sequential_runtime import MEMIT_ROLE


RUN_TOKEN = "p1r52-b100-accepted-z-rephrase-observation-v1"


def _source_gate(source_head: str) -> str:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_PARENT, source_head],
        cwd=REPO_ROOT,
        check=False,
    ).returncode != 0:
        raise ValueError("accepted-z observation source ancestry differs")
    manifest, raw_sha = load_rooted_json(
        REPO_ROOT / "project/run_scripts/ode_bf/locks" / MANIFEST_FILE,
        expected_schema=MANIFEST_SCHEMA,
    )
    if manifest.get("source_parent") != SOURCE_PARENT:
        raise ValueError("accepted-z observation source manifest parent differs")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("accepted-z observation source manifest is empty")
    for entry in entries:
        path = REPO_ROOT / str(entry["path"])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry["size"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError("accepted-z observation source manifest member differs")
    return raw_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True)
    args = parser.parse_args(argv)
    if args.run_token != RUN_TOKEN:
        parser.error("accepted-z observation run token differs")
    try:
        lock_path = REPO_ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        lock, lock_sha = load_and_validate_lock(lock_path)
        source_manifest_sha = _source_gate(args.source_head)
        stream_path = REPO_ROOT / "project/run_scripts/ode_bf/locks" / SEAL_FILE
        stream = verify_p1r52_b100x10_stream(
            json.loads(stream_path.read_text(encoding="utf-8"))
        )
        reference_root = sealed_reference_root(args.role)
        terminal_path = reference_root / "terminal.json"
        if (
            not terminal_path.is_file()
            or sha256_file(terminal_path) != REFERENCE_TERMINAL_SHA256[args.role]
        ):
            raise ValueError("accepted-z sealed reference terminal differs")
        memit_artifacts = (
            verify_memit_artifacts(REPO_ROOT, hash_covariances=False)
            if args.role == MEMIT_ROLE
            else None
        )
        result = run_p1(
            repo_root=REPO_ROOT,
            alias="llama3-8b-inst",
            output_root=args.output_root,
            source_head=args.source_head,
            p1r52_sequential_role=args.role,
            p1r52_sequential_scale="b100x10",
            p1r52_attempt_suffix="accepted-z-rephrase-obs",
            p1r52_accepted_z_observation=True,
            p1r52_accepted_z_reference_root=reference_root,
        )
        result.update(
            {
                "accepted_z_observation_instruction_id": INSTRUCTION_ID,
                "accepted_z_observation_lock_sha256": lock_sha,
                "accepted_z_observation_lock_root": lock["root_digest"],
                "source_manifest_sha256": source_manifest_sha,
                "stream_seal_sha256": sha256_file(stream_path),
                "stream_root": stream["root_digest"],
                "stream_order": stream["all_request_order_sha256"],
                "sealed_reference_root": str(reference_root),
                "sealed_reference_terminal_sha256": sha256_file(terminal_path),
                "memit_artifacts": memit_artifacts,
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
            failure_schema="ode-edit-s05-p1r52-b100-accepted-z-observation-job-failure/v1",
        )
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
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
