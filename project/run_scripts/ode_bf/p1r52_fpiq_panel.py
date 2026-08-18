"""Closed numerical/source identities for P1R52-FPiQ Atomic execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import ODEBFContractError


LOCK_FILE = "numerical_lock_s05_p1r52_fpiq_atomic_b10x10.json"
LOCK_SCHEMA = "ode-edit-s05-p1r52-fpiq-atomic-lock/v1"
PARENT = "34f0b505fe5f32078aa4b8cffd109a7490d8aaa5"
PARENT_TREE = "253a177dcd72d2235d0ece500558b357b578f84b"
POLICIES = ("j0", "sv", "fpiq")


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": "ODEEDIT-S05-P1R52-FROZEN-PI-SEQUENTIAL-QUOTA-WRITER-V1",
        "contract_sha256": "e5c767cdd6cfd498376dacad39155d08d0bc79c89496dbf629725a9b928b4436",
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
        "residual_coordinate": "R=(z_selected-current_terminal)/h",
        "slope_coordinate": "a_applied=h*g_raw_coefficient",
        "slope_active_epsilon": 1e-12,
        "slope_active_epsilon_source": "P1R24_Q_EPSILON",
        "physical_h_application_count_per_factor": 1,
        "pi_application_count_per_layer": 1,
        "layer4_capture_count": 0,
        "layer4_added_backward_count": 0,
        "additional_slope_group_count_per_k8_sequential_arm": 32,
        "physical_materialization_count_per_outer": 1,
        "alpha_cache": "ON",
        "structural_h": "OFF",
        "stage_gpu_max": 3,
        "project_gpu_cap": 4,
        "scientific_promotion_authorized": False,
        "sequential_historical_stage": "NOT_AUTHORIZED",
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("P1R52-FPiQ numerical/source lock differs")
    zero = (
        "target_step_rho_authority_count",
        "second_h_application_count",
        "controller_change_count",
        "hard_p_h_gate_influence_count",
        "functional_veto_count",
        "retry_count",
        "backtracking_count",
        "debt_carry_count",
        "heldout_writer_decision_access_count",
        "request_layer_router_count",
    )
    if any(value.get(key) != 0 for key in zero):
        raise ODEBFContractError("P1R52-FPiQ forbidden influence differs")


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
