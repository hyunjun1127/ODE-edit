"""Closed numerical/source identities for P1R43 execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import MODEL_ALIASES, ODEBFContractError
from .p1r43_independent_b10x10_runtime import B1_METHODS, METHODS


LOCK_SCHEMA = "ode-edit-s05-p1r43-rho-free-b10x10-lock/v1"
LOCK_FILE = "numerical_lock_s05_p1r43_rho_free_b10x10.json"
PARENT = "ca68a4f459fd7303a4d5abbde2e1bf7aee0d805c"
PARENT_TREE = "4af77417a52c70b99784d64dbfcabba0a91a1960"


def expected_result_name(
    alias: str, method: str, *, attempt_suffix: str | None = None
) -> str:
    if alias not in MODEL_ALIASES or method not in METHODS + B1_METHODS:
        raise ODEBFContractError("P1R43 result identity differs")
    arm = "neutral" if method.endswith("-NEUTRAL") else "soft"
    if method in B1_METHODS:
        suffix = f"-{attempt_suffix}" if attempt_suffix else ""
        return f"s05-p1r43-rho-free-b1-{alias}-{arm}{suffix}-v1"
    suffix = f"-{attempt_suffix}" if attempt_suffix else ""
    return f"s05-p1r43-rho-free-independent-b10x10-{alias}-{arm}{suffix}-v1"


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": "ODEEDIT-S05-P1R43-RHO-FREE-SEMANTIC-FIRST-STRENGTH-RECOVERY-V1",
        "contract_sha256": "ab8a8506d38e9348b9e421d943fc70fc0db09e02eb19c241f7759274065d39b0",
        "exact_p1r42_parent": PARENT,
        "exact_p1r42_parent_tree": PARENT_TREE,
        "p1r39_ancestor": "763457560f2efb177a56310dfd87526772cf8158",
        "methods": list(METHODS),
        "models": list(MODEL_ALIASES),
        "stream_root": "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6",
        "all_request_order_sha256": "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c",
        "case_count_per_cell": 10,
        "request_count_per_case": 10,
        "grid_count": 8,
        "h": 0.125,
        "tau_final": 1.0,
        "context_ordinals": list(range(6)),
        "target_direction_policy": "ENTRY_CALIBRATED_PURE_SEMANTIC_GRADIENT",
        "corrector_policy": "ONE_SCALAR_ACTUAL_NLL_CORRECTOR",
        "writer_coordinate": "FULL_CURRENT_RESIDUAL",
        "writer_demand_name": "alpha_req",
        "soft_totality": "FULL_STRENGTH_OR_NEUTRAL_FULL_STRENGTH_FALLBACK",
        "routing_arms": ["NEUTRAL", "SOFT"],
        "history_mode": "OFF",
        "terminal_panels": ["EFF_Z_INJECT", "GEN_Z_INJECT", "EFF_W", "GEN_W"],
        "project_gpu_cap": 4,
        "array_max_concurrent_gpu": 4,
        "scientific_promotion_authorized": False,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("P1R43 numerical/source lock differs")
    zero = (
        "trust_rho_decision_influence_count",
        "trust_threshold_access_count",
        "radius_state_access_count",
        "radius_update_count",
        "semantic_hold_threshold_access_count",
        "target_kl_decision_influence_count",
        "target_decay_decision_influence_count",
        "target_preservation_decision_influence_count",
        "semantic_debt_input_count",
        "remaining_horizon_decision_influence_count",
        "hard_p_budget_influence_count",
        "inner_k_evaluator_access_count",
        "heldout_gen_controller_access_count",
        "history_append_count",
        "retry_count",
        "backtracking_count",
    )
    if any(value.get(key) != 0 for key in zero):
        raise ODEBFContractError("P1R43 forbidden influence differs")


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
    "PARENT_TREE",
    "expected_result_name",
    "load_and_validate_lock",
    "validate_lock",
]
