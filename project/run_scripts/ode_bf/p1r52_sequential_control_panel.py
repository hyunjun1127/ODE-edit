"""Closed identities for the Alpha-cache-on, Structural-H-off control."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import ODEBFContractError
from .p1r52_sequential_contract import HISTORY_COUNTS, INSTRUCTION_ID, ROUND_COUNT
from .p1r52_sequential_runtime import R52_CONTROL_ROLE


LOCK_SCHEMA = "ode-edit-s05-p1r52-sequential-alphacache-on-structuralh-off-lock/v1"
LOCK_FILE = "numerical_lock_s05_p1r52_sequential_alphacache_on_structuralh_off.json"
SOURCE_MANIFEST = "source_manifest_s05_p1r52_sequential_alphacache_on_structuralh_off.json"
PARENT = "237f3b65bff69b835a7253f5154bba19feaf1fef"
PARENT_TREE = "abe87eec97d5ff90f6eb2f76656f2c0ff922f012"


def validate_control_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": INSTRUCTION_ID,
        "exact_parent": PARENT,
        "exact_parent_tree": PARENT_TREE,
        "model": "llama3-8b-inst",
        "role": R52_CONTROL_ROLE,
        "round_count": ROUND_COUNT,
        "batch_size": 10,
        "request_count": 100,
        "k": 8,
        "h": 0.125,
        "alpha_solve_history_width_at_entry": list(HISTORY_COUNTS),
        "structural_h_decision_history_width": [0] * ROUND_COUNT,
        "stream_root": "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6",
        "stream_order": "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c",
        "p1r52_atomic_repair_revision": "R1",
        "p1r52_atomic_head": "34f0b505fe5f32078aa4b8cffd109a7490d8aaa5",
        "structural_h_policy": "ALPHA_SOLVE_CACHE_ON_STRUCTURAL_H_OFF",
        "risk_anchor_policy": "OBSERVATION_ONLY_TRANSACTION_PARITY",
        "evaluation_checkpoints": list(range(1, 11)),
        "batch_entry_pre_evaluator_count": 10,
        "project_gpu_cap": 4,
        "stage_gpu_max": 1,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("P1R52 sequential control numerical lock differs")
    zero = (
        "structural_h_decision_influence_count",
        "observation_ledger_decision_influence_count",
        "interbatch_w0_restore_count",
        "inner_k_step_heldout_evaluation_count",
        "functional_h_veto_count",
        "hard_h_budget_count",
        "retry_count",
        "backtracking_count",
        "first_hit_stop_count",
        "counterfactual_added_forward_count",
        "counterfactual_added_backward_count",
        "counterfactual_added_materialization_count",
        "accuracy_extension_added_forward_count",
        "accuracy_extension_added_backward_count",
        "accuracy_extension_added_generation_count",
        "heldout_controller_influence_count",
        "batch_entry_pre_backward_count",
        "batch_entry_pre_generation_count",
        "batch_entry_pre_decision_influence_count",
    )
    if any(value.get(key) != 0 for key in zero):
        raise ODEBFContractError("P1R52 sequential control forbidden influence differs")


def load_and_validate_control_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_control_lock(value)
    return value, file_sha


__all__ = [
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "PARENT",
    "PARENT_TREE",
    "SOURCE_MANIFEST",
    "load_and_validate_control_lock",
    "validate_control_lock",
]
