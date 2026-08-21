#!/usr/bin/env python3
"""Build rooted source/numerical locks for the joint-P/C task."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1r52_joint_pc_execution import INSTRUCTION_ID
from project.run_scripts.ode_bf.p1r52_joint_pc_panel import (
    CONTRACT_SHA256,
    LOCK_FILE,
    LOCK_SCHEMA,
    PARENT,
    PARENT_TREE,
    SOURCE_MANIFEST_FILE,
    SOURCE_MANIFEST_SCHEMA,
)
from project.run_scripts.ode_bf.p1r52_joint_pc_runtime import STREAM_ORDER, STREAM_ROOT


SOURCE_FILES = (
    "project/run_scripts/ode_bf/p1r52_joint_pc_execution.py",
    "project/run_scripts/ode_bf/p1r52_joint_pc_panel.py",
    "project/run_scripts/ode_bf/p1r52_joint_pc_router.py",
    "project/run_scripts/ode_bf/p1r52_joint_pc_runtime.py",
    "project/run_scripts/ode_bf/p1r52_sequential_runtime.py",
    "project/run_scripts/ode_bf/tests/test_p1r52_joint_pc_execution.py",
    "project/run_scripts/ode_bf/tests/test_p1r52_joint_pc_router.py",
    "project/run_scripts/session05_ode_bf_p1r52_joint_pc_b100.py",
    "project/run_scripts/session05_ode_bf_p1r52_joint_pc_b100.sbatch",
    "project/run_scripts/session05_ode_bf_p1r52_joint_pc_b100_build_locks.py",
    "project/run_scripts/session05_ode_bf_p1r52_joint_pc_b100_dry_plan.py",
    "project/run_scripts/session05_ode_bf_submit_p1r52_joint_pc_b100.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict[str, object]) -> None:
    rooted = dict(value)
    rooted["root_digest"] = canonical_hash(rooted)
    payload = json.dumps(rooted, indent=2, sort_keys=True).encode() + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        path.write_bytes(payload)
        os.chmod(path, 0o600)
    else:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)


def main() -> int:
    lock_root = REPO_ROOT / "project/run_scripts/ode_bf/locks"
    numerical = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": INSTRUCTION_ID,
        "parent": PARENT,
        "parent_tree": PARENT_TREE,
        "contract_sha256": CONTRACT_SHA256,
        "model": "llama3-8b-inst",
        "batch_size": 100,
        "case_count": 10,
        "target_k": 8,
        "target_compute_count_per_case": 1,
        "writer_arm_count": 3,
        "layer_order": [4, 5, 6, 7, 8],
        "reference_pi": [0.2] * 5,
        "c1_execution": "REMAINING_RESIDUAL_SUFFIX",
        "c2_execution": "FIXED_ENTRY_RESIDUAL_QUOTA",
        "writer_coordinate": "FINITE_RESIDUAL_DIVIDE_H_THEN_FACTOR_H_ONCE",
        "residual_divide_h_count_per_layer": 1,
        "residual_divide_h_layer_count": 5,
        "factor_h_application_count_per_selected_layer": 1,
        "second_h_application_count": 0,
        "c2_direct_fixed_quota_coordinate": "THETA_PI_LEFT_ENTRY_RESIDUAL",
        "c2_numeric_h_multiplication_count": 0,
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "batch_entry_evaluator_count": 0,
        "retry_count": 0,
        "imputation_count": 0,
        "multi_context_residual_count": 0,
        "structural_h_decision_influence_count": 0,
        "hard_budget_influence_count": 0,
        "current_slope_inverse_count": 0,
        "extra_target_compute_count": 0,
        "inner_heldout_access_count": 0,
        "easyedit_write_count": 0,
    }
    entries = []
    for relative in sorted(SOURCE_FILES):
        path = REPO_ROOT / relative
        entries.append({"path": relative, "size": path.stat().st_size, "sha256": _sha(path)})
    manifest = {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "instruction_id": INSTRUCTION_ID,
        "parent": PARENT,
        "parent_tree": PARENT_TREE,
        "entries": entries,
    }
    _write(lock_root / LOCK_FILE, numerical)
    _write(lock_root / SOURCE_MANIFEST_FILE, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
