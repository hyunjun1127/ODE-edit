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
from project.run_scripts.ode_bf.contracts import canonical_hash
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


def _accepted_scientific_projection(receipt: dict) -> dict:
    """Project an accepted transition onto source/process-independent science.

    Several receipt hashes bind source-version or process-local capture identities.  They
    are useful provenance, but they are not a byte-level scientific-state comparison.
    This projection instead binds the selected target, routing coefficients, current
    key/q geometry, virtual/physical BF16 endpoint, and realized energy.
    """
    target = receipt["target_update"]
    routing = receipt["routing"]
    writer = receipt["sequential_writer"]
    layers = writer["layers"]
    return {
        "target": {
            "target_next_sha256": target["target_next_sha256"],
            "target_displacement_sha256": target["target_displacement_sha256"],
            "selection_by_request": target["selection_by_request"],
            "selected_nll_by_request": target["selected_nll_by_request"],
            "allocation_amplitude_by_request": target["allocation_amplitude_by_request"],
        },
        "routing": {
            "pi": routing["pi"],
            "velocity": routing["velocity"],
            "status": routing["status"],
            "executed_arm": routing["executed_arm"],
        },
        "writer": {
            "alpha_star": writer["alpha_star"],
            "entry_pi": writer["entry_pi"],
            "beta": writer["beta"],
            "gamma": writer["gamma"],
            "selected_velocity": writer["selected_velocity"],
            "layers": [
                {
                    "layer": layer["layer"],
                    "key_sha256": layer["key_sha256"],
                    "q_sha256": layer["q_sha256"],
                    "q_norm": layer["q_norm"],
                    "current_residual_norm": layer["current_residual_norm"],
                    "next_residual_norm": layer["next_residual_norm"],
                    "beta": layer["beta"],
                    "gamma_beta": layer["gamma_beta"],
                    "theta": layer["theta"],
                    "factor_energy": layer["factor_energy"],
                    "actual_bf16_energy_share": layer["actual_bf16_energy_share"],
                }
                for layer in layers
            ],
        },
        "virtual_physical": {
            "final_virtual_bf16_sha256": receipt["sequential_virtual_physical_identity"][
                "final_virtual_bf16_sha256"
            ],
            "post_commit_bf16_sha256": receipt["sequential_virtual_physical_identity"][
                "post_commit_bf16_sha256"
            ],
            "exact_hash_identity": receipt["sequential_virtual_physical_identity"][
                "exact_hash_identity"
            ],
            "effective_bf16_sha256": receipt["materialization"]["effective_bf16_sha256"],
            "realized_bf16_step_energy": receipt["materialization"][
                "realized_bf16_step_energy"
            ],
        },
        "structural_h": receipt["structural_h"],
    }


def _terminal_scientific_projection(receipt: dict) -> dict:
    immediate = receipt["current_batch_immediate_post_summary"]
    transaction = receipt["history_transaction"]
    return {
        "entry_weight_sha256": receipt["entry_weight_sha256"],
        "commit_weight_sha256": receipt["commit_weight_sha256"],
        "history_width_at_entry": receipt["history_width_at_entry"],
        "terminal_target_sha256": receipt["atomic_or_native"]["terminal_target_sha256"],
        "history_transaction": {
            "before_version": transaction["before_version"],
            "after_version": transaction["after_version"],
            "appended_count": transaction["appended_count"],
        },
        "immediate_post": {
            "request_order_sha256": immediate["request_order_sha256"],
            "metrics": immediate["metrics"],
            "locality": immediate["locality"],
        },
    }


def verify_stage_a_result(output_root: Path, sealed_legacy: dict) -> dict:
    """Verify a completed Stage-A result without model/GPU work."""
    reference_root = Path(sealed_legacy["root"])
    prefix_rows = []
    for round_index in range(1, 10):
        relative = Path("raw/batches") / f"b{round_index:02d}"
        observed_terminal = json.loads((output_root / relative / "terminal.json").read_text())
        reference_terminal = json.loads(
            (reference_root / relative / "terminal.json").read_text()
        )
        observed_projection = _terminal_scientific_projection(observed_terminal)
        reference_projection = _terminal_scientific_projection(reference_terminal)
        if observed_projection != reference_projection:
            raise ValueError(
                f"PIR-U Stage-A sealed Legacy terminal science differs at B{round_index}"
            )
        step_hashes = []
        for step_index in range(1, 9):
            step_relative = relative / "raw/ode/p1r52-pir-pir-u" / f"accepted-k{step_index}.json"
            observed_step = json.loads((output_root / step_relative).read_text())
            reference_step = json.loads((reference_root / step_relative).read_text())
            observed_step_projection = _accepted_scientific_projection(observed_step)
            reference_step_projection = _accepted_scientific_projection(reference_step)
            if observed_step_projection != reference_step_projection:
                raise ValueError(
                    "PIR-U Stage-A sealed Legacy transition science differs "
                    f"at B{round_index}-K{step_index}"
                )
            step_hashes.append(canonical_hash(observed_step_projection))
        prefix_rows.append(
            {
                "round": round_index,
                "status": "SCIENTIFIC_EXACT",
                "terminal_projection_sha256": canonical_hash(observed_projection),
                "accepted_step_projection_sha256": step_hashes,
            }
        )

    current_b10 = json.loads(
        (output_root / "raw/batches/b10/terminal.json").read_text(encoding="utf-8")
    )
    reference_b10 = json.loads(
        (reference_root / "raw/batches/b10/terminal.json").read_text(encoding="utf-8")
    )
    if (
        current_b10["entry_weight_sha256"] != reference_b10["entry_weight_sha256"]
        or current_b10["history_width_at_entry"] != 900
        or current_b10["atomic_or_native"]["initial"]["target_z_sha256"]
        != reference_b10["atomic_or_native"]["initial"]["target_z_sha256"]
    ):
        raise ValueError("PIR-U Stage-A B10 immutable entry differs")

    b10_step_rows = []
    for step_index in range(1, 9):
        step_path = (
            output_root
            / "raw/batches/b10/raw/ode/p1r52-pir-pir-u"
            / f"accepted-k{step_index}.json"
        )
        step = json.loads(step_path.read_text(encoding="utf-8"))
        writer = step["sequential_writer"]
        later_layers = writer["layers"][1:]
        if (
            writer["prefix_history_policy"] != "PIRU-CACHE-COMPLETE"
            or writer["prefix_empty_history_solve_count"] != 0
            or writer["prefix_committed_history_solve_count"] != 4
            or writer["current_batch_history_inclusion_count"] != 0
            or writer["current_prefix_history_inclusion_count"] != 0
            or any(layer["solve_history_width"] != 900 for layer in later_layers)
            or any(layer["solve_history_policy"] != "PIRU-CACHE-COMPLETE" for layer in later_layers)
        ):
            raise ValueError(f"PIR-U Stage-A B10 cache-complete mechanism differs at K{step_index}")
        b10_step_rows.append(
            {
                "k": step_index,
                "history_width": 900,
                "empty_history_solve_count": 0,
                "committed_history_solve_count": 4,
                "q_sha256": [layer["q_sha256"] for layer in later_layers],
            }
        )
    return {
        "status": "SEALED_HIGH_HISTORY_B10_PASS",
        "verifier_semantics": "SOURCE_PROCESS_INDEPENDENT_SCIENTIFIC_PROJECTION",
        "excluded_provenance_fields": [
            "atomic_or_native.field_sha256",
            "atomic_or_native.materializer.transition_receipt_sha256",
            "process_local_capture_sha256",
            "source_coupled_identity_sha256",
        ],
        "prefix_rows": prefix_rows,
        "b10_history_width": 900,
        "b10_entry_weight_sha256": current_b10["entry_weight_sha256"],
        "b10_entry_target_sha256": current_b10["atomic_or_native"]["initial"][
            "target_z_sha256"
        ],
        "b10_step_rows": b10_step_rows,
        "legacy_b10_terminal_sha256": sealed_legacy["b10_terminal_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--arm", required=True, choices=("legacy", "cache-complete"))
    parser.add_argument("--stage", required=True, choices=("A-B10", "B"))
    parser.add_argument("--verify-stage-a-only", action="store_true")
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
        if args.verify_stage_a_only:
            if args.stage != "A-B10":
                parser.error("--verify-stage-a-only requires --stage A-B10")
            stage_a_gate = verify_stage_a_result(args.output_root, sealed_legacy)
            print(json.dumps(stage_a_gate, sort_keys=True, separators=(",", ":")))
            return 0
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
            stage_a_gate = verify_stage_a_result(args.output_root, sealed_legacy)
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
