"""Closed identities for P1R38 per-request P1R35 B10x10 execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import MODEL_ALIASES, ODEBFContractError


LOCK_SCHEMA = "ode-edit-s05-p1r38-pr-p1r35-perrequest-target-lock/v1"
LOCK_FILE = "numerical_lock_s05_p1r38_pr_p1r35_perrequest_target.json"
PARENT = "a625e3d1cded3ced0e9128ef7a44205953041447"
METHODS = ["PR-P1R35-NEUTRAL", "PR-P1R35-SOFT"]


def expected_result_name(alias: str, method: str) -> str:
    if alias not in MODEL_ALIASES or method not in METHODS:
        raise ODEBFContractError("P1R38 result identity differs")
    return (
        f"s05-p1r38-pr-p1r35-independent-b10x10-{alias}-"
        f"{method.rsplit('-', 1)[-1].lower()}-v1"
    )


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": "ODEEDIT-S05-P1R38-PR-P1R35-PERREQUEST-TARGET-ATOMIC-V1",
        "contract_sha256": "5c4343e52de1305c28ee9894c6dad5984e6e04b82a0f737ad645dd6038a6ddc3",
        "accepted_p1r35_scientific_checkpoint": PARENT,
        "accepted_tech_r1_checkpoint": "117ece2ee1132e7277a0b43ec5ac77feebfb2646",
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
        "hold_threshold": 0.05,
        "hold_mode": "CURRENT_STATE_INSTANTANEOUS_NO_CARRY",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1.0e-8,
        "llama_learning_rate": 0.1,
        "qwen_learning_rate": 0.5,
        "semantic_epsilon": 1.0e-8,
        "semantic_epsilon_source": "P1R24_NUMERICAL_EPSILON",
        "request_cap_radius_formula": "P1R24_H*metric.shared_speed",
        "request_cap_mode": "UPPER_ONLY_NO_FLOOR",
        "finite_endpoint_forward_reuse": True,
        "history_mode": "OFF",
        "terminal_evaluator_count_per_successful_case": 1,
        "project_gpu_cap": 4,
        "array_max_concurrent_gpu": 4,
        "scientific_promotion_authorized": False,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("P1R38 numerical/source lock differs")
    zero = (
        "global_frobenius_normalization_count",
        "additional_per_request_model_call_count",
        "additional_backward_count",
        "history_append_count",
        "retry_count",
        "backtracking_count",
        "inner_k_evaluator_access_count",
    )
    if any(value.get(key) != 0 for key in zero):
        raise ODEBFContractError("P1R38 forbidden influence differs")


def load_and_validate_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_lock(value)
    return value, file_sha


__all__ = [
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "METHODS",
    "PARENT",
    "expected_result_name",
    "load_and_validate_lock",
    "validate_lock",
]
