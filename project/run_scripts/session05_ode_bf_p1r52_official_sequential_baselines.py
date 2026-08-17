#!/usr/bin/env python3
"""Fail-closed corrected AlphaEdit and Official MEMIT sequential runners."""

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
from project.run_scripts.ode_bf.p1r52_official_sequential_baseline_panel import (
    LOCK_FILE,
    PARENT,
    SOURCE_MANIFEST,
    load_and_validate_lock,
    verify_memit_artifacts,
)
from project.run_scripts.ode_bf.p1r52_sequential_contract import INSTRUCTION_ID
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    MEMIT_ROLE,
    NATIVE_CORRECTED_ROLE,
)


RUN_TOKEN = "p1r52-official-sequential-baselines-corrected-v1"
ROLES = (NATIVE_CORRECTED_ROLE, MEMIT_ROLE)


def _source_gate(source_head: str) -> str:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PARENT, source_head],
        cwd=REPO_ROOT,
        check=False,
    ).returncode != 0:
        raise ValueError("P1R52 official baseline source ancestry differs")
    manifest, raw_sha = load_rooted_json(
        REPO_ROOT / "project/run_scripts/ode_bf/locks" / SOURCE_MANIFEST,
        expected_schema="ode-edit-s05-p1r52-official-sequential-baselines-source-manifest/v1",
    )
    if manifest.get("instruction_id") != INSTRUCTION_ID or manifest.get("parent") != PARENT:
        raise ValueError("P1R52 official baseline source manifest header differs")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("P1R52 official baseline source manifest is empty")
    for entry in entries:
        path = REPO_ROOT / str(entry["path"])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry["size"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError("P1R52 official baseline source manifest member differs")
    return raw_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    args = parser.parse_args(argv)
    try:
        lock, lock_sha = load_and_validate_lock(
            REPO_ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        source_manifest_sha = _source_gate(args.source_head)
        memit_artifacts = verify_memit_artifacts(
            REPO_ROOT, hash_covariances=False
        ) if args.role == MEMIT_ROLE else None
        result = run_p1(
            repo_root=REPO_ROOT,
            alias="llama3-8b-inst",
            output_root=args.output_root,
            source_head=args.source_head,
            p1r52_sequential_role=args.role,
        )
        result.update(
            {
                "official_baseline_lock_sha256": lock_sha,
                "official_baseline_lock_root": lock["root_digest"],
                "source_manifest_sha256": source_manifest_sha,
                "memit_artifacts": memit_artifacts,
            }
        )
    except P1OutputRootCollision as exc:
        print(json.dumps({"status": "FAIL_CLOSED_OUTPUT_ROOT_COLLISION", "message_sha256": hashlib.sha256(str(exc).encode()).hexdigest()}), file=sys.stderr)
        return 1
    except Exception as exc:
        failure_sha, failure = write_p1_failure_once(
            args.output_root,
            exc,
            repo_root=REPO_ROOT,
            instruction_id=INSTRUCTION_ID,
            failure_schema="ode-edit-s05-p1r52-official-sequential-baseline-job-failure/v1",
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
