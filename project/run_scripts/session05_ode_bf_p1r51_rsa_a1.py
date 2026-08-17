#!/usr/bin/env python3
"""Fail-closed P1R51 RSA-A1 phase runner."""

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
from project.run_scripts.ode_bf.p1r51_independent_panel import LOCK_FILE, PARENT, load_and_validate_lock
from project.run_scripts.ode_bf.p1r51_independent_runtime import INSTRUCTION_ID, PHASES


RUN_TOKEN = "p1r51-p1r43-rsa-a1-v1"
SOURCE_MANIFEST = "source_manifest_s05_p1r51_rsa_a1.json"


def _source_gate(source_head: str) -> str:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PARENT, source_head],
        cwd=REPO_ROOT,
        check=False,
    ).returncode != 0:
        raise ValueError("P1R51 source ancestry differs")
    manifest_path = REPO_ROOT / "project/run_scripts/ode_bf/locks" / SOURCE_MANIFEST
    manifest, raw_sha = load_rooted_json(
        manifest_path,
        expected_schema="ode-edit-s05-p1r51-rsa-a1-source-manifest/v1",
    )
    entries = manifest.get("entries")
    if (
        manifest.get("instruction_id") != INSTRUCTION_ID
        or manifest.get("parent") != PARENT
        or not isinstance(entries, list)
        or not entries
    ):
        raise ValueError("P1R51 source manifest header differs")
    observed: list[str] = []
    for entry in entries:
        relative = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(relative, str):
            raise ValueError("P1R51 source entry differs")
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("P1R51 source path differs")
        if path.stat().st_size != entry.get("size") or sha256_file(path) != entry.get("sha256"):
            raise ValueError("P1R51 source bytes differ")
        observed.append(relative)
    if observed != sorted(set(observed)):
        raise ValueError("P1R51 source manifest ordering differs")
    protected = manifest.get("protected_p1r43_blobs")
    if protected != {
        "project/run_scripts/ode_bf/p1r43_full_strength_routing.py": "f58741eaf04d1523950953eeec88fcb81f30332f",
        "project/run_scripts/ode_bf/p1r43_rho_free_target.py": "63f64e17d96ff8887d4655614f091c63268310ff",
    }:
        raise ValueError("P1R51 protected P1R43 source identities differ")
    for relative, expected_blob in protected.items():
        observed_blob = subprocess.run(
            ["git", "rev-parse", f"{source_head}:{relative}"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        if observed_blob != expected_blob:
            raise ValueError("P1R51 protected P1R43 Git blob differs")
    return raw_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--phase", required=True, choices=PHASES)
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
            p1r51_phase=args.phase,
            p1r51_attempt_suffix=args.attempt_suffix,
        )
        result = {
            **result,
            "p1r51_lock_sha256": lock_sha,
            "p1r51_lock_root": lock["root_digest"],
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
            failure_schema="ode-edit-s05-p1r51-rsa-a1-job-failure/v1",
        )
        print(json.dumps({"status": "FAIL_CLOSED", "model": args.model, "phase": args.phase, "exception_class": failure["exception_class"], "exception_message_sha256": failure["exception_message_sha256"], "failure_sha256": failure_sha}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
