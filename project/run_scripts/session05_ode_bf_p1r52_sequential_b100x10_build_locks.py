#!/usr/bin/env python3
"""Create-once numerical lock and source manifest for P1R52 10xB100."""

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
    LOCK_SCHEMA,
    PARENT,
    PARENT_TREE,
    ROLES,
    SOURCE_MANIFEST,
)
from project.run_scripts.ode_bf.p1r52_sequential_scale import B100X10_INSTRUCTION_ID


LOCK_ROOT = REPO_ROOT / "project/run_scripts/ode_bf/locks"
SOURCE_PATHS = (
    "project/run_scripts/ode_bf/p1_backend.py",
    "project/run_scripts/ode_bf/p1_evaluator.py",
    "project/run_scripts/ode_bf/p1r29_sequential_preparation.py",
    "project/run_scripts/ode_bf/p1_runtime.py",
    "project/run_scripts/ode_bf/p1_scalable_batched_experiment.py",
    "project/run_scripts/ode_bf/p1_state.py",
    "project/run_scripts/ode_bf/p1r52_b100x10_stream.py",
    "project/run_scripts/ode_bf/p1r52_official_sequential_baselines.py",
    "project/run_scripts/ode_bf/p1r52_sequential_b100x10_panel.py",
    "project/run_scripts/ode_bf/p1r52_sequential_contract.py",
    "project/run_scripts/ode_bf/p1r52_sequential_runtime.py",
    "project/run_scripts/ode_bf/p1r52_sequential_scale.py",
    "project/run_scripts/ode_bf/tests/test_p1r52_sequential_b100x10.py",
    "project/run_scripts/session05_ode_bf_p1r52_b100x10_make_stream.py",
    "project/run_scripts/session05_ode_bf_p1r52_sequential_b100x10_build_locks.py",
    "project/run_scripts/session05_ode_bf_p1r52_sequential_b100x10_fourcell.py",
    "project/run_scripts/session05_ode_bf_p1r52_sequential_b100x10_fourcell.sbatch",
    "project/run_scripts/session05_ode_bf_p1r52_sequential_b100x10_fourcell_dry_plan.py",
    "project/run_scripts/session05_ode_bf_submit_p1r52_sequential_b100x10_fourcell.py",
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
        "instruction_id": B100X10_INSTRUCTION_ID,
        "exact_parent": PARENT,
        "exact_parent_tree": PARENT_TREE,
        "model": "llama3-8b-inst",
        "roles": list(ROLES),
        "round_count": 10,
        "batch_size": 100,
        "request_count_per_role": 1000,
        "k": 8,
        "h": 0.125,
        "history_width_at_entry": list(range(0, 1000, 100)),
        "stream_root": "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a",
        "stream_order": "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3",
        "stream_first_b100_prefixes_p1r24_b10x10": True,
        "stream_seal_sha256": "550600af120070594078b73a5da7d066ad6287d8a3224c55f61babca592e0586",
        "p1r52_atomic_repair_revision": "R1",
        "p1r52_sequential_scientific_formula_delta": "NONE_BATCH_SCALE_ONLY",
        "inherited_r52_sequential_lock_sha256": "d1ff7c3b95c02de636634670b6e75fe80463a970418f32d81ae5b605b499faf4",
        "inherited_official_baseline_lock_sha256": "5da594670caf54a442acf2a0d543eeee57d6be1a4efae47425e7d8a5a83afc16",
        "alphaedit_entrypoint_policy": "OFFICIAL_EASYEDIT_CACHE_C_CONTINUITY_ON_STATIC_P_SEPARATE",
        "memit_entrypoint_policy": "OFFICIAL_EASYEDIT_MEMIT_SEQUENTIAL",
        "r52_structural_h_roles": ["ON", "OFF_WITH_ALPHA_CACHE_ON"],
        "physical_weight_persistence": True,
        "interbatch_w0_restore_count": 0,
        "terminal_w0_restore_count": 1,
        "frozen_evaluator_sha256": "25c3f49039e7bacb58de6d9ab8139f6f24f21532434a08690e9ec71ea939e145",
        "frozen_aggregator_sha256": "64f009b2fb648627a956b95abd801838d86edf04c8e1888d90ade0ec07b745c0",
        "project_gpu_cap": 4,
        "stage_gpu_max": 4,
        "monitor_cadence_after_initial_gate_seconds": 3600,
        "retry_count": 0,
        "backtracking_count": 0,
        "imputation_count": 0,
        "inner_k_step_heldout_evaluation_count": 0,
        "accuracy_extension_added_forward_count": 0,
        "accuracy_extension_added_backward_count": 0,
        "accuracy_extension_added_generation_count": 0,
        "heldout_controller_influence_count": 0,
        "easyedit_write_count": 0,
    }
    lock_sha = _write_once(LOCK_ROOT / LOCK_FILE, lock)
    entries = []
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        entries.append(
            {"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)}
        )
    manifest = {
        "schema_version": "ode-edit-s05-p1r52-sequential-b100x10-source-manifest/v1",
        "instruction_id": B100X10_INSTRUCTION_ID,
        "parent": PARENT,
        "parent_tree": PARENT_TREE,
        "entries": entries,
        "entry_count": len(entries),
        "numerical_lock_sha256": lock_sha,
        "stream_seal_sha256": "550600af120070594078b73a5da7d066ad6287d8a3224c55f61babca592e0586",
        "source_scope": "BATCH_SCALE_ADAPTER_AND_FOUR_CELL_EXECUTION_ONLY",
    }
    manifest_sha = _write_once(LOCK_ROOT / SOURCE_MANIFEST, manifest)
    print(
        json.dumps(
            {
                "lock": str(LOCK_ROOT / LOCK_FILE),
                "lock_sha256": lock_sha,
                "manifest": str(LOCK_ROOT / SOURCE_MANIFEST),
                "manifest_sha256": manifest_sha,
                "entry_count": len(entries),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
