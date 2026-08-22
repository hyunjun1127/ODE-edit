"""Closed identities for P1R40 semantic-deficit velocity decay execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import MODEL_ALIASES, ODEBFContractError


LOCK_SCHEMA = "ode-edit-s05-p1r40-semantic-deficit-velocity-decay-lock/v1"
LOCK_FILE = "numerical_lock_s05_p1r40_semantic_deficit_velocity_decay.json"
PARENT = "6f48ac2800b257ceb16368fff5137212dfa6037f"
METHODS = ["SDVD-P1R38-NEUTRAL", "SDVD-P1R38-SOFT"]


def expected_result_name(
    alias: str, method: str, *, attempt_suffix: str | None = None
) -> str:
    if alias not in MODEL_ALIASES or method not in METHODS:
        raise ODEBFContractError("P1R40 result identity differs")
    arm = method.rsplit("-", 1)[-1].lower()
    suffix = f"-{attempt_suffix}" if attempt_suffix else ""
    return f"s05-p1r40-p1r38-semantic-deficit-velocity-decay-{alias}-{arm}{suffix}-v1"


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": "ODEEDIT-S05-P1R40-P1R38-SEMANTIC-DEFICIT-VELOCITY-DECAY-ATOMIC-V1",
        "contract_sha256": "8dfa8a44724b0b9ad7a360d7a593bc576fe69419867efc6b8ae911e746b0a8a2",
        "accepted_p1r38_scientific_checkpoint": PARENT,
        "p1r38_target_source_sha256": "2ee532a3b4293cafb99116b1a587a18f4eef2489deb26b6e5cb3afcd8e3795a3",
        "p1r35_coordinate_source_sha256": "539425cc1e6bbe66cc873f4c177e8587d3e5dc955fdb081b1d909b0dc8ce617b",
        "p1r34_finite_demand_source_sha256": "b2c59ac75798b9d70fbbc370c6d9f805d33cbeb8f683be81bb9effe7d1dbfe53",
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
        "velocity_epsilon": 1.0e-8,
        "semantic_deficit": "PURE_TARGET_NEW_NLL_NO_KL_DECAY_THRESHOLD_SUBTRACTION",
        "request_gradient_formula": "B*BATCH_MEAN_TARGET_NEW_GRADIENT",
        "q_signed_formula": "-DOT(REQUEST_TARGET_NEW_GRADIENT,NOMINAL_DISPLACEMENT)",
        "gamma_formula": "1_IF_Q_NONPOSITIVE_ELSE_MIN_1_D_OVER_Q_PLUS_EPS",
        "gamma_application": "TARGET_DISPLACEMENT_EXACTLY_ONCE",
        "nll_bin_edges": [0.0, 0.05, 0.25, 1.0],
        "request_cap_radius_formula": "P1R24_H*metric.shared_speed",
        "request_cap_mode": "UPPER_ONLY_NO_FLOOR",
        "writer_coordinate": "FULL_CURRENT_RESIDUAL_TARGET_NEXT_MINUS_CURRENT_TERMINAL",
        "finite_endpoint_forward_reuse": True,
        "history_mode": "OFF",
        "terminal_evaluator_count_per_successful_case": 1,
        "project_gpu_cap": 4,
        "array_max_concurrent_gpu": 4,
        "scientific_promotion_authorized": False,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("P1R40 numerical/source lock differs")
    zero = (
        "additional_per_request_model_call_count",
        "additional_forward_count",
        "additional_backward_count",
        "additional_materialization_count",
        "gamma_target_to_writer_application_count",
        "remaining_horizon_division_count",
        "semantic_debt_input_count",
        "history_append_count",
        "retry_count",
        "backtracking_count",
        "inner_k_evaluator_access_count",
    )
    if any(value.get(key) != 0 for key in zero):
        raise ODEBFContractError("P1R40 forbidden influence differs")


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
