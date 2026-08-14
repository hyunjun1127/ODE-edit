"""Closed identities for P1R41 trust-clipped execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import MODEL_ALIASES, ODEBFContractError


LOCK_SCHEMA = "ode-edit-s05-p1r41-trust-clipped-gradient-flow-b10x10-lock/v1"
LOCK_FILE = "numerical_lock_s05_p1r41_trust_clipped_gradient_flow_b10x10.json"
PARENT = "763457560f2efb177a56310dfd87526772cf8158"
METHODS = ["P1R41-TRUST-CLIPPED-NEUTRAL", "P1R41-TRUST-CLIPPED-SOFT"]
B1_METHODS = ["P1R41-B1-TRUST-CLIPPED-NEUTRAL", "P1R41-B1-TRUST-CLIPPED-SOFT"]


def expected_result_name(
    alias: str, method: str, *, attempt_suffix: str | None = None
) -> str:
    if alias not in MODEL_ALIASES or method not in METHODS + B1_METHODS:
        raise ODEBFContractError("P1R41 result identity differs")
    if method in B1_METHODS:
        suffix = f"-{attempt_suffix}" if attempt_suffix else ""
        arm = "neutral" if method.endswith("-NEUTRAL") else "soft"
        return f"s05-p1r41-trust-clipped-b1-{alias}-{arm}{suffix}-v1"
    arm = "neutral" if method.endswith("-NEUTRAL") else "soft"
    return f"s05-p1r41-trust-clipped-independent-b10x10-{alias}-{arm}-v1"


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": "ODEEDIT-S05-P1R41-P1R39-TRUST-CLIPPED-GRADIENT-FLOW-B10X10-V1",
        "contract_sha256": "5ae7074d2280d8975a5735ceca108377d4f270292a293232f7a690efe25344b1",
        "accepted_p1r39_checkpoint": PARENT,
        "accepted_p1r39_tree": "1f58423b53dfb7e84c8011ab08c2bf7a5ad4ca25",
        "accepted_p1r39_numerical_lock_sha256": "c028389177536bacf19063b1019d20522e63b377e5605a42873258e0baf9cd31",
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
        "target_direction_policy": "PER_REQUEST_TRUST_CLIPPED_ENTRY_CALIBRATED_COMPOSITE_GRADIENT",
        "normalization_epsilon": 1.0e-12,
        "normalization_epsilon_source": "FIXED_E8_NORMALIZATION_EPSILON",
        "hold_threshold": 0.05,
        "hold_mode": "CURRENT_STATE_INSTANTANEOUS_NO_CARRY",
        "semantic_epsilon": 1.0e-8,
        "semantic_epsilon_source": "P1R24_NUMERICAL_EPSILON",
        "writer_coordinate": "FULL_CURRENT_RESIDUAL",
        "entry_gradient_reference": "CASE_K1_PER_REQUEST_COMPOSITE_GRADIENT_NORM_PLUS_EPS",
        "initial_radius_factor": 1.0,
        "minimum_radius_divisor": 16.0,
        "maximum_radius_factor": 2.0,
        "radius_shrink_factor": 0.5,
        "radius_expand_factor": 2.0,
        "rho_shrink_threshold": 0.25,
        "rho_expand_threshold": 0.75,
        "radius_update_same_step_influence_count": 0,
        "routing_arms": ["NEUTRAL", "SOFT"],
        "history_mode": "OFF",
        "terminal_evaluator_count_per_successful_case": 1,
        "project_gpu_cap": 4,
        "array_max_concurrent_gpu": 4,
        "scientific_promotion_authorized": False,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("P1R41 numerical/source lock differs")
    zero = (
        "adam_state_access_count",
        "adam_learning_rate_access_count",
        "adam_bias_correction_access_count",
        "request_raw_cap_access_count",
        "semantic_velocity_decay_access_count",
        "objective_alignment_access_count",
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
        raise ODEBFContractError("P1R41 forbidden influence differs")


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
