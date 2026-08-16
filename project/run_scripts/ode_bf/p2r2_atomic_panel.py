"""Closed numerical/source identities for P2R2 Atomic execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import ODEBFContractError
from .p2r2_atomic_runtime import ARMS, CASE_COUNTS, METHOD


LOCK_SCHEMA = "ode-edit-s05-p2r2-residual-transport-writer-lock/v1"
LOCK_FILE = "numerical_lock_s05_p2r2_residual_transport_writer.json"
PARENT = "8f817e13167289dac190fe74bfa42e2b3e01372d"
PARENT_TREE = "abe577756ea82e1fed0847b6329c82dc49184e9f"


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": "ODEEDIT-S05-P2R2-P2R1-SEMANTIC-CONSERVING-RESIDUAL-TRANSPORT-WRITER-V1",
        "contract_sha256": "f519e81fcc40870423774be0c507851332f604fde910dbcadad8faa73e0cf3f6",
        "exact_p2r1_parent": PARENT,
        "exact_p2r1_parent_tree": PARENT_TREE,
        "method": METHOD,
        "models": ["llama3-8b-inst", "qwen2.5-7b-inst"],
        "arms": list(ARMS),
        "stream_root": "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6",
        "all_request_order_sha256": "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c",
        "case_counts": list(CASE_COUNTS),
        "request_count_per_case": 10,
        "context_ordinals": [0, 1, 2, 3, 4, 5],
        "layer_order": [4, 5, 6, 7, 8],
        "request_layer_alpha_shape": [5, 10],
        "physical_response_shape": [10, 50],
        "outer_state_count": 8,
        "target_microsteps_per_outer_state": 3,
        "target_microstep_count": 24,
        "rms_reset_count_per_case": 0,
        "teacher_refresh_count_per_case": 1,
        "residual_divisor": 1,
        "simplex_primal_tolerance": 1.0e-8,
        "structural_p_tie_relative_tolerance": 1.0e-8,
        "outer_joint_materialization_count_per_step": 1,
        "writer_h_application_count": 0,
        "project_gpu_cap": 4,
        "stage_gpu_max": 2,
        "scientific_promotion_authorized": False,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("P2R2 numerical/source lock differs")
    zero_fields = (
        "candidate_forward_count",
        "candidate_materialization_count",
        "semantic_debt_input_count",
        "remaining_horizon_division_count",
        "legacy_writer_access_count",
        "legacy_writer_decision_influence_count",
        "hard_p_budget_influence_count",
        "functional_p_veto_count",
        "retry_count",
        "backtracking_count",
        "heldout_controller_access_count",
    )
    if any(value.get(key) != 0 for key in zero_fields):
        raise ODEBFContractError("P2R2 forbidden influence differs")


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
