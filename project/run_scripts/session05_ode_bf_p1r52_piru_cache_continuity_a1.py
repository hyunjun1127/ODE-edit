#!/usr/bin/env python3
"""Fail-closed PIR-U Alpha-cache continuity A1 runner."""

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
from project.run_scripts.ode_bf.p1_runtime import P1OutputRootCollision, run_p1, write_p1_failure_once
from project.run_scripts.ode_bf.p1r52_b100x10_stream import SEAL_FILE, verify_p1r52_b100x10_stream
from project.run_scripts.ode_bf.p1r52_piru_cache_continuity import (
    ATTEMPT_SUFFIX,
    CACHE_COMPLETE_ROLE,
    INSTRUCTION_ID,
    LEGACY_ROLE,
    STAGE_A_ATTEMPT_SUFFIX,
)
from project.run_scripts.ode_bf.p1r52_piru_cache_continuity_panel import (
    LOCK_FILE,
    PARENT,
    SOURCE_MANIFEST_FILE,
    SOURCE_MANIFEST_SCHEMA,
    load_and_validate_lock,
    verify_sealed_legacy,
)


def _source_gate(source_head: str) -> str:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PARENT, source_head],
        cwd=REPO_ROOT,
        check=False,
    ).returncode != 0:
        raise ValueError("PIR-U cache-continuity source ancestry differs")
    manifest, raw_sha = load_rooted_json(
        REPO_ROOT / "project/run_scripts/ode_bf/locks" / SOURCE_MANIFEST_FILE,
        expected_schema=SOURCE_MANIFEST_SCHEMA,
    )
    if manifest.get("instruction_id") != INSTRUCTION_ID:
        raise ValueError("PIR-U cache-continuity source manifest header differs")
    for entry in manifest.get("entries", []):
        path = REPO_ROOT / entry["path"]
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry["size"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError("PIR-U cache-continuity source bytes differ")
    return raw_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--arm", required=True, choices=("legacy", "cache-complete"))
    parser.add_argument("--stage", required=True, choices=("A-B10", "B"))
    args = parser.parse_args(argv)
    role = LEGACY_ROLE if args.arm == "legacy" else CACHE_COMPLETE_ROLE
    if args.stage == "A-B10" and role != CACHE_COMPLETE_ROLE:
        parser.error("Stage A executes the Cache-Complete B10 probe only")
    attempt_suffix = STAGE_A_ATTEMPT_SUFFIX if args.stage == "A-B10" else ATTEMPT_SUFFIX
    cache_rounds = (10,) if args.stage == "A-B10" else None
    try:
        lock, lock_sha = load_and_validate_lock(
            REPO_ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        source_manifest_sha = _source_gate(args.source_head)
        stream_path = REPO_ROOT / "project/run_scripts/ode_bf/locks" / SEAL_FILE
        stream = verify_p1r52_b100x10_stream(json.loads(stream_path.read_text(encoding="utf-8")))
        sealed_legacy = verify_sealed_legacy()
        result = run_p1(
            repo_root=REPO_ROOT,
            alias="llama3-8b-inst",
            output_root=args.output_root,
            source_head=args.source_head,
            p1r52_sequential_role=role,
            p1r52_sequential_scale="b100x10",
            p1r52_attempt_suffix=attempt_suffix,
            p1r52_batch_entry_evaluator_enabled=False,
            p1r52_accepted_z_observation=True,
            p1r52_accepted_z_sealed_w_reuse=False,
            p1r52_postsolve_energy_warn_enabled=True,
            p1r52_piru_cache_complete_rounds=cache_rounds,
        )
        stage_a_gate = None
        if args.stage == "A-B10":
            stable_fields = (
                "commit_weight_sha256",
                "history_width_at_entry",
            )
            prefix_rows = []
            for round_index in range(1, 10):
                relative = Path("raw/batches") / f"b{round_index:02d}" / "terminal.json"
                observed = json.loads((args.output_root / relative).read_text(encoding="utf-8"))
                reference = json.loads((Path(sealed_legacy["root"]) / relative).read_text(encoding="utf-8"))
                if any(observed[field] != reference[field] for field in stable_fields) or (
                    observed["atomic_or_native"]["terminal_target_sha256"]
                    != reference["atomic_or_native"]["terminal_target_sha256"]
                    or observed["atomic_or_native"]["materializer"]["transition_receipt_sha256"]
                    != reference["atomic_or_native"]["materializer"]["transition_receipt_sha256"]
                    or observed["atomic_or_native"]["field_sha256"]
                    != reference["atomic_or_native"]["field_sha256"]
                    or observed["history_transaction"]["before_version"]
                    != reference["history_transaction"]["before_version"]
                    or observed["history_transaction"]["after_version"]
                    != reference["history_transaction"]["after_version"]
                    or observed["history_transaction"]["appended_count"]
                    != reference["history_transaction"]["appended_count"]
                ):
                    raise ValueError("PIR-U Stage-A sealed Legacy prefix differs")
                prefix_rows.append({"round": round_index, "status": "EXACT"})
            current_b10 = json.loads(
                (args.output_root / "raw/batches/b10/terminal.json").read_text(encoding="utf-8")
            )
            reference_b10 = json.loads(
                (Path(sealed_legacy["root"]) / "raw/batches/b10/terminal.json").read_text(encoding="utf-8")
            )
            if (
                current_b10["entry_weight_sha256"] != reference_b10["entry_weight_sha256"]
                or current_b10["history_width_at_entry"] != 900
                or current_b10["atomic_or_native"]["initial"]["target_z_sha256"]
                != reference_b10["atomic_or_native"]["initial"]["target_z_sha256"]
            ):
                raise ValueError("PIR-U Stage-A B10 immutable entry differs")
            stage_a_gate = {
                "status": "SEALED_HIGH_HISTORY_B10_ENTRY_PASS",
                "prefix_rows": prefix_rows,
                "b10_history_width": 900,
                "b10_entry_weight_sha256": current_b10["entry_weight_sha256"],
                "b10_entry_target_sha256": current_b10["atomic_or_native"]["initial"]["target_z_sha256"],
                "legacy_b10_terminal_sha256": sealed_legacy["b10_terminal_sha256"],
            }
        result.update(
            {
                "cache_continuity_lock_sha256": lock_sha,
                "cache_continuity_lock_root": lock["root_digest"],
                "source_manifest_sha256": source_manifest_sha,
                "stream_seal_sha256": sha256_file(stream_path),
                "stream_root": stream["root_digest"],
                "stream_order": stream["all_request_order_sha256"],
                "sealed_legacy": sealed_legacy,
                "stage_a_gate": stage_a_gate,
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
            failure_schema="ode-edit-s05-p1r52-piru-cache-continuity-job-failure/v1",
        )
        print(json.dumps({"status": "FAIL_CLOSED", "role": role, "exception_class": failure["exception_class"], "exception_message_sha256": failure["exception_message_sha256"], "failure_sha256": failure_sha}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
