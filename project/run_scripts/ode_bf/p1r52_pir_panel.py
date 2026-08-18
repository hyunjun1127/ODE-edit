"""Closed numerical/source identities for P1R52-PIR Atomic execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import ODEBFContractError


LOCK_FILE = "numerical_lock_s05_p1r52_pir_atomic_b10x10.json"
LOCK_SCHEMA = "ode-edit-s05-p1r52-pir-atomic-lock/v1"
PARENT = "14b0250429bcff993b05fe69783ef808e09a16be"
PARENT_TREE = "33a4eebd788f4410f839afd1cf3cf93c8673fee7"
POLICIES = ("pir-j0", "pir-g", "pir-u")


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": "ODEEDIT-S05-P1R52-PIR-SEQUENTIAL-WRITER-V1",
        "contract_sha256": "60227762fb804d4f52e7501b60c6016014468d112183cc0c2ecf9cc6fbcf730a",
        "parent_head": PARENT,
        "parent_tree": PARENT_TREE,
        "model": "llama3-8b-inst",
        "arms": list(POLICIES),
        "stream_root": "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6",
        "dataset_order_sha256": "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c",
        "case_count_per_cell": 10,
        "request_count_per_case": 10,
        "grid_count": 8,
        "h": 0.125,
        "entry_layer_order": [4, 5, 6, 7, 8],
        "prefix_layer_order": [4, 5, 6, 7, 8],
        "alpha_star_source": "finite_demand.rho_write=[L0-Lz]+",
        "entry_pi_source": "routing.pi",
        "entry_applied_slope_source": "routing_problem.signed_progress",
        "beta_definition": "pi_l/sum_{j>=l}pi_j",
        "pir_g_gamma_definition": "alpha_star/sum_l(a0_l*beta_l)",
        "pir_u_gamma": 1.0,
        "residual_coordinate": "left=(z_selected-current_terminal)/h",
        "theta_coordinate": "theta=h*gamma*beta",
        "physical_h_application_count_per_factor": 1,
        "pi_application_count_per_layer": 1,
        "beta_application_count_per_layer": 1,
        "layer4_capture_count": 0,
        "layer4_added_backward_count": 0,
        "additional_current_slope_backward_count": 0,
        "current_q_solve_count_per_k8_sequential_arm": 32,
        "physical_materialization_count_per_outer": 1,
        "sequential_p_receipt": "ENTRY_FIELD_MIXED_GEOMETRY_PROXY_NOT_COMPARABLE",
        "alpha_cache": "ON",
        "structural_h": "OFF",
        "history_mode": "OFF",
        "stage_gpu_max": 3,
        "project_gpu_cap": 4,
        "scientific_promotion_authorized": False,
        "sequential_historical_stage": "NOT_AUTHORIZED",
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("P1R52-PIR numerical/source lock differs")
    zero = (
        "target_step_rho_authority_count",
        "second_h_application_count",
        "per_layer_current_nll_decision_count",
        "inverse_current_slope_division_count",
        "semantic_quota_carry_debt_count",
        "controller_change_count",
        "hard_p_h_gate_influence_count",
        "functional_veto_count",
        "retry_count",
        "backtracking_count",
        "heldout_writer_decision_access_count",
        "request_layer_router_count",
        "atomic_history_append_count",
    )
    if any(value.get(key) != 0 for key in zero):
        raise ODEBFContractError("P1R52-PIR forbidden influence differs")


def load_and_validate_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_lock(value)
    return value, file_sha


__all__ = [
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "PARENT",
    "PARENT_TREE",
    "POLICIES",
    "load_and_validate_lock",
    "validate_lock",
]
