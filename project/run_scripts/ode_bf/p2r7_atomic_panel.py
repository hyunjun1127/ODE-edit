"""Closed P2R7 scientific/source identities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import ODEBFContractError
from .p2r7_atomic_runtime import PILOT_ARMS, PRODUCTION_ARMS


LOCK_SCHEMA = "ode-edit-s05-p2r7-p2target-p1dw-shared-writer-lock/v1"
LOCK_FILE = "numerical_lock_s05_p2r7_p2target_p1dw_shared_writer.json"
PARENT = "c97e8619b42da7954ce0e824c215a8a82d70589a"
PARENT_TREE = "711b33fc736641d1f3067c00bd067ed5e60606fe"


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": "ODEEDIT-S05-P2R7-P2TARGET-P1DW-SHARED-WRITER-V1",
        "contract_sha256": "d58e5c067d5687559e2bd809d9c5da17c4d546efa6e8feb679f3038f7ebaee5b",
        "exact_p2r2_parent": PARENT,
        "exact_p2r2_parent_tree": PARENT_TREE,
        "p1r43_router_parent": "11508b6da11d606521b703037034e1814b70d8a8",
        "models": ["llama3-8b-inst", "qwen2.5-7b-inst"],
        "pilot_arms": [f"{mode}-{arm}" for mode, arm in PILOT_ARMS],
        "production_arms": [f"{mode}-{arm}" for mode, arm in PRODUCTION_ARMS],
        "stream_root": "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6",
        "all_request_order_sha256": "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c",
        "request_count_per_case": 10,
        "context_ordinals": [0, 1, 2, 3, 4, 5],
        "layer_order": [4, 5, 6, 7, 8],
        "routing_variable_count": 5,
        "outer_state_count": 8,
        "target_microsteps_per_outer_state": 3,
        "target_microstep_count": 24,
        "rms_reset_count_per_case": 0,
        "teacher_refresh_count_per_case": 1,
        "writer_transition_count": 8,
        "writer_materialization_count": 8,
        "residual_inverse_h_count_per_step": 1,
        "physical_h_application_count_per_step": 1,
        "deficit_weight": "d_i/sum_i_d_i",
        "deficit_demand": "sum_i_omega_i_d_i",
        "soft_priority": ["WEIGHTED_STRENGTH", "STRUCTURAL_P", "CAPACITY"],
        "pilot_case_indices": [1],
        "production_case_count": 10,
        "project_gpu_cap": 4,
        "scientific_promotion_authorized": False,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("P2R7 numerical/source lock differs")
    zero = (
        "request_layer_response_matrix_count",
        "p2r6_qp_import_count",
        "p2r6_shadow_solve_count",
        "candidate_materialization_count",
        "retry_count",
        "backtracking_count",
        "remaining_horizon_division_count",
        "semantic_debt_input_count",
        "hard_p_budget_influence_count",
        "functional_p_veto_count",
        "history_h_count",
        "stepwise_heldout_evaluation_count",
        "heldout_controller_access_count",
        "second_h_application_count",
    )
    if any(value.get(key) != 0 for key in zero):
        raise ODEBFContractError("P2R7 forbidden influence differs")


def load_and_validate_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_lock(value)
    return value, file_sha


__all__ = [
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "PARENT",
    "PARENT_TREE",
    "load_and_validate_lock",
    "validate_lock",
]
