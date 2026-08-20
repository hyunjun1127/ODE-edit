"""Frozen numerical/source identities for the P1R52 target-depth task."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import ODEBFContractError


LOCK_SCHEMA = "ode-edit-s05-p1r52-target-depth-il1-il3full-lock/v1"
LOCK_FILE = "numerical_lock_s05_p1r52_target_depth_il1_il3full.json"
SOURCE_PARENT = "d4e3b57d6e5fb0e082d1b51bfe15064458422143"
SOURCE_PARENT_TREE = "d6ead0224011588cec149b6687575c08b7aa7cc0"


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": "ODEEDIT-S05-P1R52-TARGET-DEPTH-IL1-IL3FULL-V1",
        "contract_sha256": "06e65d4a2df4a4610ef09818c2afd96b8dbe86cc3041afdbc5b099ca2358ec63",
        "source_parent": SOURCE_PARENT,
        "source_parent_tree": SOURCE_PARENT_TREE,
        "models": ["llama3-8b-inst", "qwen2.5-7b-inst"],
        "depth_policies": {"IL1": 1, "IL3-FULL": 3},
        "inner_h": 0.125,
        "outer_step_count": 8,
        "stream_root": "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6",
        "all_request_order_sha256": "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c",
        "case_counts": [1, 10],
        "request_count_per_case": 10,
        "context_ordinals": [0, 1, 2, 3, 4, 5],
        "target_operator": "P1R52_RSA_R42SAFEKDC_M1_REPAIR_R1",
        "writer": "P1R52_ORIGINAL_J0_FROZEN",
        "target_only_writer_materialization_count": 0,
        "inner_writer_materialization_count": 0,
        "outer_writer_materialization_count_atomic": 1,
        "entry_norm_calibration_count_per_case": 1,
        "early_stop": "ENTIRE_SELECTED_FP32_TARGET_BYTE_IDENTICAL_ONLY",
        "stage_gpu_max": 4,
        "main_push_count": 0,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("P1R52 target-depth numerical/source lock differs")
    zeros = (
        "h_over_three_count",
        "il8_count",
        "il25_count",
        "inner_writer_count",
        "inner_factor_build_count",
        "inner_weight_mutation_count",
        "inner_history_append_count",
        "inner_covariance_recompute_count",
        "inner_heldout_access_count",
        "inner_finite_demand_count",
        "inner_rho_sum_count",
        "last_inner_gradient_writer_authority_count",
        "writer_router_change_count",
        "retry_count",
        "backtracking_count",
    )
    if any(value.get(key) != 0 for key in zeros):
        raise ODEBFContractError("P1R52 target-depth forbidden influence differs")


def load_and_validate_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_lock(value)
    return value, file_sha


__all__ = [
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "SOURCE_PARENT",
    "SOURCE_PARENT_TREE",
    "load_and_validate_lock",
    "validate_lock",
]
