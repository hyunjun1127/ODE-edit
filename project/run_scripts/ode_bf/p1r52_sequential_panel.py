"""Closed identities for P1R52 Llama sequential/Historical execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import ODEBFContractError
from .p1r52_sequential_contract import HISTORY_COUNTS, INSTRUCTION_ID, ROUND_COUNT
from .p1r52_sequential_runtime import ROLES


LOCK_SCHEMA = "ode-edit-s05-p1r52-sequential-historical-lock/v1"
LOCK_FILE = "numerical_lock_s05_p1r52_llama_soft_sequential_historical_10xb10.json"
SOURCE_MANIFEST = "source_manifest_s05_p1r52_llama_soft_sequential_historical_10xb10.json"
PARENT = "8205bcc93230c02754ebf7134491f044ff61abe0"
PARENT_TREE = "ae073cba740b733f5166c7631cc5a95a40169049"


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": INSTRUCTION_ID,
        "contract_sha256": "86415f32b28e590ce474e6feaa663274b68f46f20e67d897e34d9bc777863265",
        "exact_parent": PARENT,
        "exact_parent_tree": PARENT_TREE,
        "model": "llama3-8b-inst",
        "roles": list(ROLES),
        "round_count": ROUND_COUNT,
        "batch_size": 10,
        "request_count": 100,
        "k": 8,
        "h": 0.125,
        "history_width_at_entry": list(HISTORY_COUNTS),
        "stream_root": "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6",
        "stream_order": "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c",
        "p1r52_atomic_repair_revision": "R1",
        "p1r52_atomic_head": "34f0b505fe5f32078aa4b8cffd109a7490d8aaa5",
        "structural_h_policy": "STRENGTH_THEN_H_THEN_P_THEN_CAPACITY",
        "lifetime_anchor_context": "CONTROLLER_LEGAL_FULL_SIX_ONLY",
        "evaluation_checkpoints": list(range(1, 11)),
        "batch_entry_pre_evaluator_count": 10,
        "batch_entry_pre_definition": "CURRENT_B10_AT_EXACT_PHYSICAL_W_E_MINUS_1_BEFORE_TARGET_OR_WRITER",
        "b1_cross_arm_entry_hash_metric_equality_required": True,
        "canonical_metric_aliases": {
            "Eff": "rewrite_success",
            "Gen": "paraphrase_success",
            "rephrase_success": "paraphrase_success",
            "rephrase_acc": "paraphrase_acc",
        },
        "project_gpu_cap": 4,
        "stage_gpu_max": 2,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("P1R52 sequential numerical lock differs")
    zero = (
        "interbatch_w0_restore_count",
        "inner_k_step_heldout_evaluation_count",
        "functional_h_veto_count",
        "hard_h_budget_count",
        "retry_count",
        "backtracking_count",
        "first_hit_stop_count",
        "accuracy_extension_added_forward_count",
        "accuracy_extension_added_backward_count",
        "accuracy_extension_added_generation_count",
        "heldout_controller_influence_count",
        "batch_entry_pre_backward_count",
        "batch_entry_pre_generation_count",
        "batch_entry_pre_decision_influence_count",
    )
    if any(value.get(key) != 0 for key in zero):
        raise ODEBFContractError("P1R52 sequential forbidden influence differs")


def load_and_validate_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_lock(value)
    return value, file_sha


__all__ = [
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "PARENT",
    "PARENT_TREE",
    "SOURCE_MANIFEST",
    "load_and_validate_lock",
    "validate_lock",
]
