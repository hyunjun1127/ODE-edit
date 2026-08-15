#!/usr/bin/env python3
"""Fail-close P2R6 semantic-region focused pilot runner."""

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
from project.run_scripts.ode_bf.p2r6_pilot_panel import (
    LOCK_FILE,
    PARENT,
    load_and_validate_lock,
)
from project.run_scripts.ode_bf.p2r6_pilot_runtime import (
    PHASE1_CASES,
    PHASE2_CASES,
    p2r6_phase_arms,
)
from project.run_scripts.ode_bf.p2r6_semantic_region_controller import (
    P2R6_INSTRUCTION_ID,
)


RUN_TOKEN = "p2r6-red-r2-final-scientific-run-v1"
SOURCE_MANIFEST = "source_manifest_s05_p2r6_red_r2_final.json"
RED_R2_CLEAN_PARENT = "e7c86a5598a7507f4f981068ba1fa8db4df13df9"
RED_R2_CLEAN_PARENT_TREE = "9012ccb2f31e90f49d870a395188466ee60f4319"


def _source_gate(source_head: str) -> str:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PARENT, source_head],
        cwd=REPO_ROOT,
        check=False,
    ).returncode != 0:
        raise ValueError("P2R6 P2R5 source ancestry differs")
    manifest, raw_sha = load_rooted_json(
        REPO_ROOT / "project/run_scripts/ode_bf/locks" / SOURCE_MANIFEST,
        expected_schema="ode-edit-s05-p2r6-red-r2-final-source-manifest/v1",
    )
    entries = manifest.get("entries")
    if (
        manifest.get("instruction_id") != P2R6_INSTRUCTION_ID
        or manifest.get("exact_clean_parent") != RED_R2_CLEAN_PARENT
        or manifest.get("exact_clean_parent_tree") != RED_R2_CLEAN_PARENT_TREE
        or not isinstance(entries, list)
        or not entries
    ):
        raise ValueError("P2R6 source manifest header differs")
    observed: list[str] = []
    for entry in entries:
        relative = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(relative, str):
            raise ValueError("P2R6 source manifest entry differs")
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("P2R6 source path differs")
        if path.stat().st_size != entry.get("size") or sha256_file(path) != entry.get("sha256"):
            raise ValueError("P2R6 source bytes differ")
        observed.append(relative)
    if observed != sorted(set(observed)):
        raise ValueError("P2R6 source manifest ordering differs")
    return raw_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--phase", required=True, choices=("phase1", "phase2"))
    parser.add_argument("--case-index", required=True, type=int)
    parser.add_argument("--selected-controller", choices=("AR", "AS"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    parser.add_argument("--attempt-suffix")
    args = parser.parse_args(argv)
    cases = PHASE1_CASES if args.phase == "phase1" else PHASE2_CASES
    if args.case_index not in cases[args.model]:
        parser.error("P2R6 model/phase/case mapping differs")
    p2r6_phase_arms(args.phase, args.selected_controller)
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
            p2r6_phase=args.phase,
            p2r6_case_index=args.case_index,
            p2r6_selected_controller=args.selected_controller,
            p2r6_attempt_suffix=args.attempt_suffix,
        )
        result = {
            **result,
            "p2r6_lock_sha256": lock_sha,
            "p2r6_lock_root": lock["root_digest"],
            "source_manifest_sha256": source_manifest_sha,
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
            instruction_id=P2R6_INSTRUCTION_ID,
            failure_schema="ode-edit-s05-p2r6-pilot-job-failure/v1",
        )
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "model": args.model,
                    "phase": args.phase,
                    "case_index": args.case_index,
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
