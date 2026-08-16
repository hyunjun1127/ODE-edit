#!/usr/bin/env python3
"""One-shot P2R6 Llama outer1 pre-QP private capture runner."""

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
from project.run_scripts.ode_bf.p2r6_llama_qp_capture import (
    CAPTURE_ALIAS,
    CAPTURE_CASE_INDEX,
    CAPTURE_INSTRUCTION_ID,
)
from project.run_scripts.ode_bf.p2r6_pilot_panel import LOCK_FILE, load_and_validate_lock


RUN_TOKEN = "p2r6-llama-outer1-qp-capture-v1"
SOURCE_MANIFEST = "source_manifest_s05_p2r6_llama_outer1_qp_capture_v1.json"
CAPTURE_LOCK = "numerical_lock_s05_p2r6_llama_outer1_qp_capture_v1.json"
FAILED_SCIENTIFIC_SOURCE_HEAD = "1246e5047bdac2e59d5cc0642ebd8d7104c6d066"
FAILED_SCIENTIFIC_SOURCE_TREE = "ed7c236066afe1c8140ac8b306d5be1785cc08c5"


def _source_gate(source_head: str) -> str:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", FAILED_SCIENTIFIC_SOURCE_HEAD, source_head],
        cwd=REPO_ROOT,
        check=False,
    ).returncode != 0:
        raise ValueError("P2R6 capture scientific ancestry differs")
    manifest, raw_sha = load_rooted_json(
        REPO_ROOT / "project/run_scripts/ode_bf/locks" / SOURCE_MANIFEST,
        expected_schema="ode-edit-s05-p2r6-llama-outer1-qp-capture-source-manifest/v1",
    )
    entries = manifest.get("entries")
    if (
        manifest.get("instruction_id") != CAPTURE_INSTRUCTION_ID
        or manifest.get("failed_scientific_source_head") != FAILED_SCIENTIFIC_SOURCE_HEAD
        or manifest.get("failed_scientific_source_tree") != FAILED_SCIENTIFIC_SOURCE_TREE
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


def _capture_lock_gate() -> tuple[dict[str, object], str]:
    lock, raw_sha = load_rooted_json(
        REPO_ROOT / "project/run_scripts/ode_bf/locks" / CAPTURE_LOCK,
        expected_schema="ode-edit-s05-p2r6-llama-outer1-qp-capture-numerical-lock/v1",
    )
    if (
        lock.get("instruction_id") != CAPTURE_INSTRUCTION_ID
        or lock.get("failed_scientific_source_head") != FAILED_SCIENTIFIC_SOURCE_HEAD
        or lock.get("failed_scientific_source_tree") != FAILED_SCIENTIFIC_SOURCE_TREE
        or lock.get("capture_invocation_budget") != 1
        or lock.get("selected_outer1_qp_solve_count") != 0
        or lock.get("terminal_evaluator_count") != 0
        or lock.get("phase1_scientific_endpoint_count") != 0
        or lock.get("w0_pointer_and_bytes_restore_required") is not True
        or lock.get("phase2_status")
        != "CLOSED_PENDING_NEW_PHASE1_AND_EXPLICIT_GH_RELEASE"
    ):
        raise ValueError("P2R6 capture numerical lock differs")
    return lock, raw_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=(CAPTURE_ALIAS,))
    parser.add_argument("--case-index", required=True, type=int, choices=(CAPTURE_CASE_INDEX,))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    parser.add_argument("--attempt-suffix", required=True, choices=("a1", "a2"))
    args = parser.parse_args(argv)
    try:
        lock, lock_sha = load_and_validate_lock(
            REPO_ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        capture_lock, capture_lock_sha = _capture_lock_gate()
        source_manifest_sha = _source_gate(args.source_head)
        result = run_p1(
            repo_root=REPO_ROOT,
            alias=args.model,
            output_root=args.output_root,
            source_head=args.source_head,
            p2r6_phase="capture",
            p2r6_case_index=args.case_index,
            p2r6_attempt_suffix=args.attempt_suffix,
        )
        result = {
            **result,
            "p2r6_numerical_lock_sha256": lock_sha,
            "p2r6_numerical_lock_root": lock["root_digest"],
            "capture_lock_sha256": capture_lock_sha,
            "capture_lock_root": capture_lock["root_digest"],
            "capture_source_manifest_sha256": source_manifest_sha,
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
            instruction_id=CAPTURE_INSTRUCTION_ID,
            failure_schema="ode-edit-s05-p2r6-llama-outer1-qp-capture-job-failure/v1",
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
