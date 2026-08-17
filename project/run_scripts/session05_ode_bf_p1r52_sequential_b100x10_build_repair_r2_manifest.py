#!/usr/bin/env python3
"""Create-once source manifest for the B100 post-cast clamp TECH-R2 repair."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.artifacts import sha256_file
from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1r52_sequential_b100x10_panel import (
    LOCK_FILE,
    PARENT,
    PARENT_TREE,
    SOURCE_MANIFEST_TECH_R2,
)
from project.run_scripts.ode_bf.p1r52_sequential_scale import B100X10_INSTRUCTION_ID


LOCK_ROOT = REPO_ROOT / "project/run_scripts/ode_bf/locks"
SOURCE_PATHS = (
    "project/run_scripts/ode_bf/p1_backend.py",
    "project/run_scripts/ode_bf/p1_evaluator.py",
    "project/run_scripts/ode_bf/p1_runtime.py",
    "project/run_scripts/ode_bf/p1_scalable_batched_experiment.py",
    "project/run_scripts/ode_bf/p1_state.py",
    "project/run_scripts/ode_bf/p1r29_sequential_preparation.py",
    "project/run_scripts/ode_bf/p1r52_b100x10_stream.py",
    "project/run_scripts/ode_bf/p1r52_official_sequential_baselines.py",
    "project/run_scripts/ode_bf/p1r52_r42_safe_kdc.py",
    "project/run_scripts/ode_bf/p1r52_sequential_b100x10_panel.py",
    "project/run_scripts/ode_bf/p1r52_sequential_contract.py",
    "project/run_scripts/ode_bf/p1r52_sequential_runtime.py",
    "project/run_scripts/ode_bf/p1r52_sequential_scale.py",
    "project/run_scripts/ode_bf/tests/test_p1r52_r42_safe_kdc.py",
    "project/run_scripts/ode_bf/tests/test_p1r52_sequential_b100x10.py",
    "project/run_scripts/session05_ode_bf_p1r52_b100x10_make_stream.py",
    "project/run_scripts/session05_ode_bf_p1r52_sequential_b100x10_build_locks.py",
    "project/run_scripts/session05_ode_bf_p1r52_sequential_b100x10_build_repair_r1_manifest.py",
    "project/run_scripts/session05_ode_bf_p1r52_sequential_b100x10_build_repair_r2_manifest.py",
    "project/run_scripts/session05_ode_bf_p1r52_sequential_b100x10_fourcell.py",
    "project/run_scripts/session05_ode_bf_p1r52_sequential_b100x10_fourcell.sbatch",
    "project/run_scripts/session05_ode_bf_p1r52_sequential_b100x10_fourcell_dry_plan.py",
    "project/run_scripts/session05_ode_bf_submit_p1r52_sequential_b100x10_fourcell.py",
)


def main() -> int:
    entries = []
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        entries.append(
            {"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)}
        )
    value = {
        "schema_version": "ode-edit-s05-p1r52-sequential-b100x10-r52-tech-r2-source-manifest/v1",
        "instruction_id": B100X10_INSTRUCTION_ID,
        "parent": PARENT,
        "parent_tree": PARENT_TREE,
        "supersedes_source_head": "93d0e162efdddf6bee2c77995c923b2aaf1afbd1",
        "repair_revision": "TECH-R2",
        "repair_scope": "POST_FP32_CAST_ORIGIN_CLAMP_NUMERICAL_CLOSURE_AND_INVALID_R52_CELLS_ONLY",
        "repair_first_false_gate": "P1R52_ORIGIN_RELATIVE_CLAMP_BOUND_DIFFERS_AFTER_FP32_CAST",
        "failed_job": "20453",
        "failed_cell_count": 2,
        "healthy_cell_rerun_count": 0,
        "exception_message_sha256": "c73c26fd95c01ec736460e60de17e1660d45d5f2742c0642330d8670ba4d63a3",
        "failure_sha256": "f225013d88df72d7cdd668572d9b4829cceafc2671969ca1d5f450f2f70a9673",
        "clamp_radius_delta_count": 0,
        "science_formula_tolerance_config_delta_count": 0,
        "added_model_forward_backward_materialization": [0, 0, 0],
        "entries": entries,
        "entry_count": len(entries),
        "numerical_lock_sha256": sha256_file(LOCK_ROOT / LOCK_FILE),
        "stream_seal_sha256": "550600af120070594078b73a5da7d066ad6287d8a3224c55f61babca592e0586",
    }
    payload = {**value, "root_digest": canonical_hash(value)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    destination = LOCK_ROOT / SOURCE_MANIFEST_TECH_R2
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "path": str(destination),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "root": payload["root_digest"],
                "entry_count": len(entries),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
