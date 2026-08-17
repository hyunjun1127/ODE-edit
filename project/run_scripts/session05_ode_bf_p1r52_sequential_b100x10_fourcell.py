#!/usr/bin/env python3
"""Fail-closed P1R52 Llama sequential 10xB100 four-cell runner."""

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
from project.run_scripts.ode_bf.p1r52_sequential_b100x10_panel import (
    LOCK_FILE,
    PARENT,
    ROLES,
    SOURCE_MANIFEST_TECH_R1,
    SOURCE_MANIFEST_TECH_R2,
    load_and_validate_lock,
    verify_memit_artifacts,
)
from project.run_scripts.ode_bf.p1r52_sequential_runtime import MEMIT_ROLE
from project.run_scripts.ode_bf.p1r52_sequential_scale import B100X10_INSTRUCTION_ID


RUN_TOKENS = {
    "tech-r1": "p1r52-llama-sequential-10xb100-fourcell-tech-r1-v1",
    "tech-r2": "p1r52-llama-sequential-10xb100-r52-tech-r2-v1",
}


def _source_gate(source_head: str, *, attempt_suffix: str = "tech-r2") -> str:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PARENT, source_head],
        cwd=REPO_ROOT,
        check=False,
    ).returncode != 0:
        raise ValueError("P1R52 B100x10 source ancestry differs")
    manifest_name, schema = {
        "tech-r1": (
            SOURCE_MANIFEST_TECH_R1,
            "ode-edit-s05-p1r52-sequential-b100x10-tech-r1-source-manifest/v1",
        ),
        "tech-r2": (
            SOURCE_MANIFEST_TECH_R2,
            "ode-edit-s05-p1r52-sequential-b100x10-r52-tech-r2-source-manifest/v1",
        ),
    }[attempt_suffix]
    manifest, raw_sha = load_rooted_json(
        REPO_ROOT / "project/run_scripts/ode_bf/locks" / manifest_name,
        expected_schema=schema,
    )
    if (
        manifest.get("instruction_id") != B100X10_INSTRUCTION_ID
        or manifest.get("parent") != PARENT
    ):
        raise ValueError("P1R52 B100x10 source manifest header differs")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("P1R52 B100x10 source manifest is empty")
    for entry in entries:
        path = REPO_ROOT / str(entry["path"])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry["size"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError("P1R52 B100x10 source manifest member differs")
    return raw_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--attempt-suffix", required=True, choices=tuple(RUN_TOKENS))
    parser.add_argument("--run-token", required=True)
    args = parser.parse_args(argv)
    if args.run_token != RUN_TOKENS[args.attempt_suffix]:
        parser.error("run token differs from attempt suffix")
    try:
        lock, lock_sha = load_and_validate_lock(
            REPO_ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        source_manifest_sha = _source_gate(
            args.source_head, attempt_suffix=args.attempt_suffix
        )
        stream_path = REPO_ROOT / "project/run_scripts/ode_bf/locks" / SEAL_FILE
        stream = verify_p1r52_b100x10_stream(
            json.loads(stream_path.read_text(encoding="utf-8"))
        )
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
            p1r52_attempt_suffix=args.attempt_suffix,
        )
        result.update(
            {
                "b100x10_lock_sha256": lock_sha,
                "b100x10_lock_root": lock["root_digest"],
                "source_manifest_sha256": source_manifest_sha,
                "stream_seal_sha256": sha256_file(stream_path),
                "stream_root": stream["root_digest"],
                "stream_order": stream["all_request_order_sha256"],
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
            instruction_id=B100X10_INSTRUCTION_ID,
            failure_schema=(
                f"ode-edit-s05-p1r52-sequential-b100x10-{args.attempt_suffix}-job-failure/v1"
            ),
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
