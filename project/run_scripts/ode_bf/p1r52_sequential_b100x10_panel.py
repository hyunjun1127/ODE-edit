"""Closed identities for the P1R52 Llama sequential 10xB100 four-cell panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import ODEBFContractError
from .p1r52_b100x10_stream import REQUEST_COUNT, ROUND_COUNT
from .p1r52_official_sequential_baseline_panel import verify_memit_artifacts
from .p1r52_sequential_runtime import (
    MEMIT_ROLE,
    NATIVE_CORRECTED_ROLE,
    R52_CONTROL_ROLE,
    R52_H_ROLE,
)
from .p1r52_sequential_scale import B100X10_INSTRUCTION_ID, P1R52_B100X10_SCALE


LOCK_SCHEMA = "ode-edit-s05-p1r52-sequential-b100x10-fourcell-lock/v1"
LOCK_FILE = "numerical_lock_s05_p1r52_llama_sequential_10xb100_fourcell.json"
SOURCE_MANIFEST_TECH_R1 = "source_manifest_s05_p1r52_llama_sequential_10xb100_fourcell_tech_r1.json"
SOURCE_MANIFEST_TECH_R2 = "source_manifest_s05_p1r52_llama_sequential_10xb100_r52_tech_r2.json"
SOURCE_MANIFEST_TECH_R3 = "source_manifest_s05_p1r52_llama_sequential_10xb100_r52_tech_r3.json"
SOURCE_MANIFEST = SOURCE_MANIFEST_TECH_R3
PARENT = "329ef063b969e01635f7602e40923936a405c1bd"
PARENT_TREE = "34d5c147bd23cde3549d88cbd6f844f61d07b618"
ROLES = (MEMIT_ROLE, NATIVE_CORRECTED_ROLE, R52_H_ROLE, R52_CONTROL_ROLE)


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": B100X10_INSTRUCTION_ID,
        "exact_parent": PARENT,
        "exact_parent_tree": PARENT_TREE,
        "model": "llama3-8b-inst",
        "roles": list(ROLES),
        "round_count": ROUND_COUNT,
        "batch_size": P1R52_B100X10_SCALE.batch_size,
        "request_count_per_role": REQUEST_COUNT,
        "k": 8,
        "h": 0.125,
        "history_width_at_entry": list(P1R52_B100X10_SCALE.history_counts),
        "stream_root": "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a",
        "stream_order": "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3",
        "stream_first_b100_prefixes_p1r24_b10x10": True,
        "p1r52_atomic_repair_revision": "R1",
        "p1r52_sequential_scientific_formula_delta": "NONE_BATCH_SCALE_ONLY",
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
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("P1R52 B100x10 numerical lock differs")
    zero = (
        "retry_count",
        "backtracking_count",
        "imputation_count",
        "inner_k_step_heldout_evaluation_count",
        "accuracy_extension_added_forward_count",
        "accuracy_extension_added_backward_count",
        "accuracy_extension_added_generation_count",
        "heldout_controller_influence_count",
        "easyedit_write_count",
    )
    if any(value.get(key) != 0 for key in zero):
        raise ODEBFContractError("P1R52 B100x10 forbidden count differs")


def load_and_validate_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_lock(value)
    return value, file_sha


__all__ = [
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "PARENT",
    "PARENT_TREE",
    "ROLES",
    "SOURCE_MANIFEST",
    "SOURCE_MANIFEST_TECH_R1",
    "SOURCE_MANIFEST_TECH_R2",
    "SOURCE_MANIFEST_TECH_R3",
    "load_and_validate_lock",
    "validate_lock",
    "verify_memit_artifacts",
]
