"""Closed identities for the P1R52 joint-P/C B100 writer panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import ODEBFContractError
from .p1r52_joint_pc_execution import INSTRUCTION_ID
from .p1r52_joint_pc_runtime import STREAM_ORDER, STREAM_ROOT


PARENT = "6caafd1148498497602876a9693f6760e6b39e85"
PARENT_TREE = "9e2c8aac6d21b3dedc7770d4de894e1b73d3c097"
CONTRACT_SHA256 = "9f24ca3062e91a903f4fb92e80c73c372b8fc03f76e8e12d545ae97c3df4f0b8"
LOCK_FILE = "numerical_lock_s05_p1r52_joint_pc_c1_c2_b100_v1.json"
LOCK_SCHEMA = "ode-edit-s05-p1r52-joint-pc-c1-c2-b100-lock/v1"
SOURCE_MANIFEST_FILE = "source_manifest_s05_p1r52_joint_pc_c1_c2_b100_v1.json"
SOURCE_MANIFEST_SCHEMA = "ode-edit-s05-p1r52-joint-pc-c1-c2-b100-source-manifest/v1"


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": INSTRUCTION_ID,
        "parent": PARENT,
        "parent_tree": PARENT_TREE,
        "contract_sha256": CONTRACT_SHA256,
        "model": "llama3-8b-inst",
        "batch_size": 100,
        "case_count": 10,
        "target_k": 8,
        "target_compute_count_per_case": 1,
        "writer_arm_count": 3,
        "layer_order": [4, 5, 6, 7, 8],
        "reference_pi": [0.2] * 5,
        "c1_execution": "REMAINING_RESIDUAL_SUFFIX",
        "c2_execution": "FIXED_ENTRY_RESIDUAL_QUOTA",
        "writer_coordinate": "FINITE_RESIDUAL_DIVIDE_H_THEN_FACTOR_H_ONCE",
        "residual_divide_h_count_per_layer": 1,
        "residual_divide_h_layer_count": 5,
        "factor_h_application_count_per_selected_layer": 1,
        "second_h_application_count": 0,
        "c2_direct_fixed_quota_coordinate": "THETA_PI_LEFT_ENTRY_RESIDUAL",
        "c2_numeric_h_multiplication_count": 0,
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "batch_entry_evaluator_count": 0,
        "retry_count": 0,
        "imputation_count": 0,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("joint P/C numerical lock differs")
    for name in (
        "multi_context_residual_count",
        "structural_h_decision_influence_count",
        "hard_budget_influence_count",
        "current_slope_inverse_count",
        "extra_target_compute_count",
        "inner_heldout_access_count",
        "easyedit_write_count",
    ):
        if value.get(name) != 0:
            raise ODEBFContractError("joint P/C forbidden influence differs")


def load_and_validate_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_lock(value)
    return value, file_sha


__all__ = [
    "CONTRACT_SHA256",
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "PARENT",
    "PARENT_TREE",
    "SOURCE_MANIFEST_FILE",
    "SOURCE_MANIFEST_SCHEMA",
    "load_and_validate_lock",
]
