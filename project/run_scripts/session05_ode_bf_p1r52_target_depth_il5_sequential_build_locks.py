#!/usr/bin/env python3
"""Create-once numerical lock and source manifest for IL5 sequential."""

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
from project.run_scripts.ode_bf.p1r52_target_depth_sequential_il5 import (
    INSTRUCTION_ID,
    METHOD_ID,
    POLICY,
    STREAM_ORDER,
    STREAM_ROOT,
)
from project.run_scripts.ode_bf.p1r52_target_depth_sequential_il5_panel import (
    LOCK_FILE,
    LOCK_SCHEMA,
    PARENT,
    PARENT_TREE,
    SOURCE_MANIFEST_FILE,
    SOURCE_MANIFEST_SCHEMA,
)


LOCK_ROOT = REPO_ROOT / "project/run_scripts/ode_bf/locks"
SOURCE_PATHS = (
    "project/run_scripts/ode_bf/atomic_runtime_optimization.py",
    "project/run_scripts/ode_bf/bg_soft_diagnostics.py",
    "project/run_scripts/ode_bf/common_cold_coordinate.py",
    "project/run_scripts/ode_bf/common_coldcoord_fixed_e8_runtime.py",
    "project/run_scripts/ode_bf/fixed_e8_soft_routing.py",
    "project/run_scripts/ode_bf/p1_backend.py",
    "project/run_scripts/ode_bf/p1_runtime.py",
    "project/run_scripts/ode_bf/p1_scalable_batched_experiment.py",
    "project/run_scripts/ode_bf/p1_state.py",
    "project/run_scripts/ode_bf/p1_evaluator.py",
    "project/run_scripts/ode_bf/p1r34_w_anchored_finite_demand.py",
    "project/run_scripts/ode_bf/p1r52_b100x10_stream.py",
    "project/run_scripts/ode_bf/p1r52_sequential_b100x10_panel.py",
    "project/run_scripts/ode_bf/p1r52_sequential_contract.py",
    "project/run_scripts/ode_bf/p1r52_sequential_runtime.py",
    "project/run_scripts/ode_bf/p1r52_sequential_scale.py",
    "project/run_scripts/ode_bf/p1r52_r42_safe_kdc.py",
    "project/run_scripts/ode_bf/p1r52_piru_postenergy_warn.py",
    "project/run_scripts/ode_bf/p1r52_target_depth.py",
    "project/run_scripts/ode_bf/p1r52_target_depth_inner_telemetry.py",
    "project/run_scripts/ode_bf/p1r52_target_depth_sequential_il5.py",
    "project/run_scripts/ode_bf/p1r52_target_depth_sequential_il5_panel.py",
    "project/run_scripts/ode_bf/tests/test_p1r52_target_depth.py",
    "project/run_scripts/ode_bf/tests/test_p1r52_target_depth_sequential_il5.py",
    "project/run_scripts/session05_ode_bf_p1r52_target_depth_il5_sequential_b100x10.py",
    "project/run_scripts/session05_ode_bf_p1r52_target_depth_il5_sequential_b100x10.sbatch",
    "project/run_scripts/session05_ode_bf_p1r52_target_depth_il5_sequential_b100x10_dry_plan.py",
    "project/run_scripts/session05_ode_bf_submit_p1r52_target_depth_il5_sequential_b100x10.py",
)


def _write_once(path: Path, value: dict[str, object]) -> str:
    payload = {**value, "root_digest": canonical_hash(value)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    lock = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "exact_parent": PARENT,
        "exact_parent_tree": PARENT_TREE,
        "alias": "llama3-8b-inst",
        "role": POLICY.role,
        "round_count": POLICY.round_count,
        "batch_size": POLICY.batch_size,
        "request_count": POLICY.request_count,
        "outer_k": POLICY.outer_count,
        "target_depth": POLICY.depth.value,
        "target_inner_count": POLICY.depth.inner_count,
        "inner_h": POLICY.inner_h,
        "expected_inner_rows_per_batch": POLICY.expected_inner_rows_per_batch,
        "expected_request_inner_rows_per_batch": POLICY.expected_request_inner_rows_per_batch,
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "writer": "P1R52-J0",
        "allocation": "RS-SOFT",
        "alpha_cache": "ON",
        "structural_h": "ON",
        "physical_weight_persistence": True,
        "batch_entry_evaluator_count": 0,
        "interbatch_w0_restore_count": 0,
        "terminal_w0_restore_count": 1,
        "task_gpu_cap": 3,
        "stage_gpu_max": 1,
        "scientific_amendment": "POSTSOLVE_ENERGY_WARN_RECORD_ONLY",
        "postsolve_energy_warn_enabled": True,
        "postsolve_energy_decision_influence_count": 0,
        "postsolve_energy_tolerance": 1e-12,
        "postsolve_coefficient_shrink_count": 0,
        "postsolve_retry_count": 0,
        "technical_attempt": "TECH_R1_IL5_POSTENERGY_WARN_SCOPE",
        "execution_node": "server2",
        "retry_count": 0,
        "backtracking_count": 0,
        "imputation_count": 0,
        "inner_writer_materialization_count": 0,
        "inner_heldout_controller_access_count": 0,
        "observation_action_influence_count": 0,
        "easyedit_write_count": 0,
    }
    lock_sha = _write_once(LOCK_ROOT / LOCK_FILE, lock)
    entries = []
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        entries.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "instruction_id": INSTRUCTION_ID,
        "parent": PARENT,
        "parent_tree": PARENT_TREE,
        "entries": entries,
        "entry_count": len(entries),
        "numerical_lock_sha256": lock_sha,
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "source_scope": "TARGET_DEPTH_IL5_POSTSOLVE_ENERGY_WARN_RECORD_ONLY_WITH_EXACT_IL5_RUNTIME_SCOPE_TECH_R1",
    }
    manifest_sha = _write_once(LOCK_ROOT / SOURCE_MANIFEST_FILE, manifest)
    print(
        json.dumps(
            {
                "lock": str(LOCK_ROOT / LOCK_FILE),
                "lock_sha256": lock_sha,
                "manifest": str(LOCK_ROOT / SOURCE_MANIFEST_FILE),
                "manifest_sha256": manifest_sha,
                "entry_count": len(entries),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
