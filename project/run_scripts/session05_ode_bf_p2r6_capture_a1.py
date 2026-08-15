#!/usr/bin/env python3
"""Fail-close one-shot P2R6 Qwen k5 replay-capsule runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.artifacts import load_rooted_json, sha256_file
from project.run_scripts.ode_bf.p1_runtime import (
    P1OutputRootCollision,
    run_p1,
    write_p1_failure_once,
)
from project.run_scripts.ode_bf.p2r6_replay_capture import (
    CAPTURE_ALIAS,
    CAPTURE_INSTRUCTION_ID,
    TECH_R8_SCIENTIFIC_HEAD,
)


RUN_TOKEN = "p2r6-red-r1-capture-a1"
SOURCE_MANIFEST = "source_manifest_s05_p2r6_red_r1_capture_a1.json"


def _source_gate(source_head: str) -> str:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", TECH_R8_SCIENTIFIC_HEAD, source_head],
        cwd=REPO_ROOT,
        check=False,
    ).returncode != 0:
        raise ValueError("P2R6 capture TECH-R8 ancestry differs")
    manifest, raw_sha = load_rooted_json(
        REPO_ROOT / "project/run_scripts/ode_bf/locks" / SOURCE_MANIFEST,
        expected_schema="ode-edit-s05-p2r6-red-r1-capture-source-manifest/v1",
    )
    entries = manifest.get("entries")
    if (
        manifest.get("instruction_id") != CAPTURE_INSTRUCTION_ID
        or manifest.get("scientific_source_head") != TECH_R8_SCIENTIFIC_HEAD
        or not isinstance(entries, list)
        or not entries
    ):
        raise ValueError("P2R6 capture source manifest header differs")
    observed: list[str] = []
    for entry in entries:
        relative = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(relative, str):
            raise ValueError("P2R6 capture source manifest entry differs")
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("P2R6 capture source path differs")
        if path.stat().st_size != entry.get("size") or sha256_file(path) != entry.get("sha256"):
            raise ValueError("P2R6 capture source bytes differ")
        observed.append(relative)
    if observed != sorted(set(observed)):
        raise ValueError("P2R6 capture source manifest ordering differs")
    return raw_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=(CAPTURE_ALIAS,))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    args = parser.parse_args(argv)
    try:
        source_manifest_sha = _source_gate(args.source_head)
        result = run_p1(
            repo_root=REPO_ROOT,
            alias=args.model,
            output_root=args.output_root,
            source_head=args.source_head,
            p2r6_phase="capture-a1",
            p2r6_case_index=1,
            p2r6_attempt_suffix="red-r1",
        )
        result = {**result, "source_manifest_sha256": source_manifest_sha}
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
            instruction_id=CAPTURE_INSTRUCTION_ID,
            failure_schema="ode-edit-s05-p2r6-red-r1-capture-job-failure/v1",
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
