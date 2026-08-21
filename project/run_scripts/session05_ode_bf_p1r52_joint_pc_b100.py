#!/usr/bin/env python3
"""Fail-closed runner for one independent B100 C0/C1/C2 panel."""

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
from project.run_scripts.ode_bf.p1r52_b100x10_stream import (
    SEAL_FILE,
    verify_p1r52_b100x10_stream,
)
from project.run_scripts.ode_bf.p1r52_joint_pc_execution import INSTRUCTION_ID
from project.run_scripts.ode_bf.p1r52_joint_pc_panel import (
    LOCK_FILE,
    PARENT,
    SOURCE_MANIFEST_FILE,
    SOURCE_MANIFEST_SCHEMA,
    load_and_validate_lock,
)
from project.run_scripts.ode_bf.p1r52_joint_pc_runtime import (
    PILOT_ROLE,
    expected_result_name,
    production_role,
)


RUN_TOKEN = "p1r52-joint-pc-c1-c2-independent-b100-v1"


def _source_gate(source_head: str) -> str:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PARENT, source_head],
        cwd=REPO_ROOT,
        check=False,
    ).returncode != 0:
        raise ValueError("joint P/C source ancestry differs")
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
        raise ValueError("joint P/C source manifest header differs")
    for entry in manifest["entries"]:
        path = REPO_ROOT / str(entry["path"])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry["size"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError("joint P/C source manifest member differs")
    return raw_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--case-index", type=int, choices=range(1, 11))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    args = parser.parse_args(argv)
    role = PILOT_ROLE if args.pilot else production_role(args.case_index)
    try:
        lock, lock_sha = load_and_validate_lock(
            REPO_ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        source_manifest_sha = _source_gate(args.source_head)
        stream_path = REPO_ROOT / "project/run_scripts/ode_bf/locks" / SEAL_FILE
        stream = verify_p1r52_b100x10_stream(
            json.loads(stream_path.read_text(encoding="utf-8"))
        )
        result = run_p1(
            repo_root=REPO_ROOT,
            alias="llama3-8b-inst",
            output_root=args.output_root,
            source_head=args.source_head,
            p1r52_sequential_role=role,
            p1r52_sequential_scale="b100x10",
            p1r52_batch_entry_evaluator_enabled=False,
        )
        result.update(
            {
                "joint_pc_lock_sha256": lock_sha,
                "joint_pc_lock_root": lock["root_digest"],
                "source_manifest_sha256": source_manifest_sha,
                "stream_seal_sha256": sha256_file(stream_path),
                "stream_root": stream["root_digest"],
                "stream_order": stream["all_request_order_sha256"],
                "expected_result_name": expected_result_name(role),
            }
        )
    except P1OutputRootCollision as exc:
        print(json.dumps({
            "status": "FAIL_CLOSED_OUTPUT_ROOT_COLLISION",
            "message_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
        }), file=sys.stderr)
        return 1
    except Exception as exc:
        failure_sha, failure = write_p1_failure_once(
            args.output_root,
            exc,
            repo_root=REPO_ROOT,
            instruction_id=INSTRUCTION_ID,
            failure_schema="ode-edit-s05-p1r52-joint-pc-c1-c2-b100-failure/v1",
        )
        print(json.dumps({
            "status": "FAIL_CLOSED",
            "role": role,
            "exception_class": failure["exception_class"],
            "exception_message_sha256": failure["exception_message_sha256"],
            "failure_sha256": failure_sha,
        }, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
