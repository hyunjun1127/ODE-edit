"""Closed identities for the P1R39-A1 normalized-gradient Soft extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import MODEL_ALIASES, ODEBFContractError


LOCK_SCHEMA = "ode-edit-s05-p1r39-a1-normalized-gradient-soft-b10x10-lock/v1"
LOCK_FILE = "numerical_lock_s05_p1r39_a1_normalized_gradient_soft_b10x10.json"
PARENT = "763457560f2efb177a56310dfd87526772cf8158"
METHODS = ["PR-P1R39-NORMALIZED-GRADIENT-SOFT"]
B1_METHODS = ["PR-P1R39-B1-NORMALIZED-GRADIENT-SOFT"]


def expected_result_name(
    alias: str, method: str, *, attempt_suffix: str | None = None
) -> str:
    if alias not in MODEL_ALIASES or method not in METHODS + B1_METHODS:
        raise ODEBFContractError("P1R39 result identity differs")
    if method in B1_METHODS:
        suffix = f"-{attempt_suffix}" if attempt_suffix else ""
        return f"s05-p1r39-a1-normalized-gradient-b1-{alias}-soft{suffix}-v1"
    return f"s05-p1r39-a1-normalized-gradient-independent-b10x10-{alias}-soft-v1"


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": "ODEEDIT-S05-P1R39-A1-NORMALIZED-GRADIENT-SOFT-B10X10-V1",
        "accepted_p1r39_checkpoint": PARENT,
        "accepted_p1r39_tree": "1f58423b53dfb7e84c8011ab08c2bf7a5ad4ca25",
        "accepted_p1r39_report_sha256": "97d7a6d36818dd5445d29850d03031b19b715c94434c236124c9289e01bd7548",
        "p1r39_target_source_sha256": "98356016cd87d258f9c9e889ccb59f6f14e3f869e0f9284d2c90721e38c6537a",
        "accepted_p1r38_checkpoint": "6f48ac2800b257ceb16368fff5137212dfa6037f",
        "accepted_p1r38_report_sha256": "281582a6bfe95b53bda1bce09e0eb453c8675fcf717c6b957051197faf2bf865",
        "methods": METHODS,
        "models": list(MODEL_ALIASES),
        "fresh_stream_root": "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6",
        "all_request_order_sha256": "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c",
        "case_count_per_cell": 10,
        "request_count_per_case": 10,
        "grid_count": 8,
        "h": 0.125,
        "tau_final": 1.0,
        "context_ordinals": list(range(6)),
        "target_direction_policy": "PER_REQUEST_NORMALIZED_COMPOSITE_GRADIENT",
        "normalization_epsilon": 1.0e-12,
        "normalization_epsilon_source": "FIXED_E8_NORMALIZATION_EPSILON",
        "hold_threshold": 0.05,
        "hold_mode": "CURRENT_STATE_INSTANTANEOUS_NO_CARRY",
        "semantic_epsilon": 1.0e-8,
        "semantic_epsilon_source": "P1R24_NUMERICAL_EPSILON",
        "writer_coordinate": "FULL_CURRENT_RESIDUAL",
        "routing_arm": "SOFT",
        "target_or_demand_attenuation_count": 0,
        "history_mode": "OFF",
        "terminal_evaluator_count_per_successful_case": 1,
        "project_gpu_cap": 4,
        "array_max_concurrent_gpu": 2,
        "scientific_promotion_authorized": False,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("P1R39 numerical/source lock differs")
    zero = (
        "adam_state_access_count",
        "adam_learning_rate_access_count",
        "adam_bias_correction_access_count",
        "request_raw_cap_access_count",
        "semantic_velocity_decay_access_count",
        "remaining_horizon_division_count",
        "semantic_debt_input_count",
        "additional_per_request_model_call_count",
        "additional_backward_count",
        "history_append_count",
        "retry_count",
        "backtracking_count",
        "inner_k_evaluator_access_count",
    )
    if any(value.get(key) != 0 for key in zero):
        raise ODEBFContractError("P1R39 forbidden influence differs")


def load_and_validate_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_lock(value)
    return value, file_sha


__all__ = [
    "B1_METHODS",
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "METHODS",
    "PARENT",
    "expected_result_name",
    "load_and_validate_lock",
    "validate_lock",
]
