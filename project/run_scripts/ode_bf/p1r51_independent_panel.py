"""Closed source/numerical identities for P1R51 RSA-A1 execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import ODEBFContractError
from .p1r51_independent_runtime import PHASES


LOCK_SCHEMA = "ode-edit-s05-p1r51-rsa-a1-lock/v1"
LOCK_FILE = "numerical_lock_s05_p1r51_rsa_a1.json"
PARENT = "11508b6da11d606521b703037034e1814b70d8a8"
PARENT_TREE = "a0e71bbb27cf36b3b51bf6cde4c92cdc9a47879b"


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": "ODEEDIT-S05-P1R51-P1R43-REQUESTWISE-SEMANTIC-ALLOCATION-A1-V1",
        "contract_sha256": "e19617806b9f60945291885d5fcc91383ee4f2c5198aaf40f77ab38bb9e1562a",
        "exact_p1r43_parent": PARENT,
        "exact_p1r43_parent_tree": PARENT_TREE,
        "phases": list(PHASES),
        "models": ["llama3-8b-inst", "qwen2.5-7b-inst"],
        "stream_root": "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6",
        "all_request_order_sha256": "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c",
        "case_count_per_full_cell": 10,
        "request_count_per_case": 10,
        "grid_count": 8,
        "h": 0.125,
        "tau_final": 1.0,
        "context_ordinals": list(range(6)),
        "target_direction_policy": "REQUESTWISE_NLL_PROPORTIONAL_ENERGY_ALLOCATION",
        "parent_target_policy": "ENTRY_CALIBRATED_PURE_SEMANTIC_GRADIENT",
        "corrector_policy": "ONE_SCALAR_ACTUAL_NLL_CORRECTOR",
        "writer_coordinate": "FULL_CURRENT_RESIDUAL",
        "writer_demand_name": "alpha_req",
        "routing_arms": ["NEUTRAL", "SOFT"],
        "history_mode": "OFF",
        "numerical_epsilon": 1e-12,
        "numerical_epsilon_source": "P1R39_NORMALIZATION_EPSILON",
        "project_gpu_cap": 4,
        "stage_gpu_max": 2,
        "scientific_promotion_authorized": False,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("P1R51 numerical/source lock differs")
    zero = (
        "adaptive_target_inner_loop_count",
        "p2_operator_access_count",
        "target_hit_threshold_access_count",
        "request_radius_access_count",
        "target_path_budget_access_count",
        "persistent_freeze_input_count",
        "semantic_debt_input_count",
        "remaining_horizon_decision_influence_count",
        "writer_router_change_count",
        "hard_p_budget_influence_count",
        "functional_p_veto_influence_count",
        "historical_sequential_state_count",
        "inner_step_heldout_evaluation_count",
        "retry_count",
        "backtracking_count",
    )
    if any(value.get(key) != 0 for key in zero):
        raise ODEBFContractError("P1R51 forbidden influence differs")


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
