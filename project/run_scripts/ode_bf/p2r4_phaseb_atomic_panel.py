"""Closed numerical/source identities for P2R4 Phase-B Atomic execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import ODEBFContractError
from .p2r4_phaseb_atomic_runtime import (
    ARMS,
    CASE_COUNTS,
    CLAMP_POLICIES,
    METHOD,
    P2R2_V2_SOURCE_HEAD,
    P2R4_PHASEB_INSTRUCTION_ID,
)


LOCK_SCHEMA = "ode-edit-s05-p2r4-phaseb-clamp-on-off-writer-lock/v1"
LOCK_FILE = "numerical_lock_s05_p2r4_phaseb_clamp_on_off_writer.json"
P2R1_PARENT = "8f817e13167289dac190fe74bfa42e2b3e01372d"
P2R1_PARENT_TREE = "abe577756ea82e1fed0847b6329c82dc49184e9f"
P2R2_V2_TREE = "711b33fc736641d1f3067c00bd067ed5e60606fe"
PARENT = P2R2_V2_SOURCE_HEAD


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": P2R4_PHASEB_INSTRUCTION_ID,
        "exact_p2r1_parent": P2R1_PARENT,
        "exact_p2r1_parent_tree": P2R1_PARENT_TREE,
        "exact_p2r2_v2_source_head": P2R2_V2_SOURCE_HEAD,
        "exact_p2r2_v2_source_tree": P2R2_V2_TREE,
        "method": METHOD,
        "models": ["llama3-8b-inst", "qwen2.5-7b-inst"],
        "clamp_policies": list(CLAMP_POLICIES),
        "writer_arms": list(ARMS),
        "stream_root": "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6",
        "all_request_order_sha256": "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c",
        "case_counts": list(CASE_COUNTS),
        "request_count_per_case": 10,
        "outer_state_count": 8,
        "target_microsteps_per_outer_state": 3,
        "target_microstep_count": 24,
        "rms_reset_count_per_case": 0,
        "writer_joint_materialization_count_per_outer": 1,
        "outer_h_application_count": 1,
        "applied_coordinate_update_count": 1,
        "writer_h_numeric_multiplication_count": 0,
        "physical_h_application_count": 0,
        "second_h_application_count": 0,
        "residual_presplit_count": 0,
        "remaining_division_count": 0,
        "semantic_debt_input_count": 0,
        "project_gpu_cap": 4,
        "stage_gpu_max": 2,
        "scientific_promotion_authorized": False,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("P2R4 Phase-B numerical/source lock differs")
    zero_fields = (
        "writer_interface_mutation_count",
        "writer_equation_mutation_count",
        "off_fallback_addition_count",
        "target_retry_count",
        "target_backtracking_count",
        "target_early_stop_count",
        "heldout_controller_access_count",
        "candidate_forward_count",
        "candidate_materialization_count",
    )
    if any(value.get(key) != 0 for key in zero_fields):
        raise ODEBFContractError("P2R4 Phase-B forbidden influence differs")


def load_and_validate_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_lock(value)
    return value, file_sha


__all__ = [
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "P2R1_PARENT",
    "P2R1_PARENT_TREE",
    "P2R2_V2_TREE",
    "PARENT",
    "load_and_validate_lock",
    "validate_lock",
]
