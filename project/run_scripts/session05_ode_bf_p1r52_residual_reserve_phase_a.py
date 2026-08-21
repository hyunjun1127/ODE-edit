#!/usr/bin/env python3
"""Fail-closed FP32 residual-reserve Phase-A B10 cell runner."""

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
from project.run_scripts.ode_bf.p1r52_residual_reserve_phase_a_execution import (
    INSTRUCTION_ID,
    PHASE_A_ARMS,
)


RUN_TOKEN = "p1r52-residual-reserve-phase-a-fp32-b10-v1"
MANIFEST_FILE = "source_manifest_s05_p1r52_residual_reserve_phase_a.json"
MANIFEST_SCHEMA = "ode-edit-s05-p1r52-residual-reserve-phase-a-source-manifest/v1"
NUMERICAL_LOCK = "numerical_lock_s05_p1r52_residual_reserve_phase_a.json"


def _source_gate(source_head: str) -> str:
    observed_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    if observed_head != source_head:
        raise ValueError("residual-reserve source HEAD differs")
    manifest, raw_sha = load_rooted_json(
        REPO_ROOT / "project/run_scripts/ode_bf/locks" / MANIFEST_FILE,
        expected_schema=MANIFEST_SCHEMA,
    )
    if manifest.get("instruction_id") != INSTRUCTION_ID:
        raise ValueError("residual-reserve source manifest identity differs")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("residual-reserve source manifest is empty")
    for entry in entries:
        path = REPO_ROOT / entry["path"]
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry["size"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError("residual-reserve source manifest bytes differ")
    return raw_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--alias", required=True, choices=("llama3-8b-inst", "qwen2.5-7b-inst"))
    parser.add_argument("--arm", required=True, choices=PHASE_A_ARMS)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--attempt-suffix", choices=("tech-r1", "tech-r2"))
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    args = parser.parse_args(argv)
    try:
        source_manifest_sha = _source_gate(args.source_head)
        lock, lock_sha = load_rooted_json(
            REPO_ROOT / "project/run_scripts/ode_bf/locks" / NUMERICAL_LOCK,
            expected_schema="ode-edit-s05-p1r52-residual-reserve-phase-a-lock/v1",
        )
        if (
            lock.get("instruction_id") != INSTRUCTION_ID
            or lock.get("arms") != list(PHASE_A_ARMS)
            or lock.get("floating_model_dtype") != "torch.float32"
            or lock.get("numeric_storage_cast_count") != 0
            or lock.get("stage_gpu_max") != 4
        ):
            raise ValueError("residual-reserve numerical lock differs")
        result = run_p1(
            repo_root=REPO_ROOT,
            alias=args.alias,
            output_root=args.output_root,
            source_head=args.source_head,
            p1r52_residual_reserve_phase_a_arm=args.arm,
            p1r52_attempt_suffix=args.attempt_suffix,
        )
        result = {
            **result,
            "source_manifest_sha256": source_manifest_sha,
            "numerical_lock_sha256": lock_sha,
            "numerical_lock_root": lock["root_digest"],
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
            failure_schema="ode-edit-s05-p1r52-residual-reserve-phase-a-failure/v1",
        )
        print(json.dumps({"status": "FAIL_CLOSED", "alias": args.alias, "arm": args.arm, "exception_class": failure["exception_class"], "exception_message_sha256": failure["exception_message_sha256"], "failure_sha256": failure_sha}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
