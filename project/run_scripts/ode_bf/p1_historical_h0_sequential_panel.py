"""Frozen panel for P1R23 Full-6 structural Historical."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import ODEBFContractError


LOCK_FILE = "numerical_lock_s05_historical_h0_sequential.json"
LOCK_SCHEMA = "ode-edit-s05-p1r23-full6-structural-historical-lock/v1"
SCIENTIFIC_CHECKPOINT = "794885a48c09046b454b88eee7880e08c390902a"
EXECUTION_CHECKPOINT = "f64d0c1221ac0eecc53a1fbb9685a9a23835af0e"
COMPUTE_NUMERICAL_ROOT = "504cd1931365750ea07cfb537335c9c923008122065d6dbe79d37b65223cc731"
FRESH_SEAL_ROOT = "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6"
ALL_REQUEST_ORDER = "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c"


def validate_historical_h0_lock(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": "ODEEDIT-S05-P1R23-FULL6-STRUCTURAL-HISTORICAL-V1-A1-FULL-MATRIX",
        "accepted_compute_a1_execution_checkpoint": EXECUTION_CHECKPOINT,
        "accepted_compute_a1_scientific_checkpoint": SCIENTIFIC_CHECKPOINT,
        "accepted_compute_a1_numerical_lock_root": COMPUTE_NUMERICAL_ROOT,
        "fresh_seal_root": FRESH_SEAL_ROOT,
        "all_request_order_sha256": ALL_REQUEST_ORDER,
        "round_count": 10,
        "batch_size": 10,
        "request_count": 100,
        "models": ["llama3-8b-inst", "qwen2.5-7b-inst"],
        "methods": [
            "BG-COMPUTE-FULL6-NOSOFT",
            "BG-COMPUTE-FULL6-SOFT",
            "RS-COMPUTE-FULL6-NOSOFT",
            "RS-COMPUTE-FULL6-SOFT",
            "ALPHAEDIT",
        ],
        "scientific_promotion_authorized": False,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ODEBFContractError("P1R23 Full-6 Historical lock identity differs")
    ours = value.get("ours", {})
    history = value.get("history", {})
    evaluation = value.get("evaluation", {})
    baseline = value.get("baseline", {})
    if (
        ours.get("grid_count") != 8
        or ours.get("h") != 0.125
        or ours.get("tau_final") != 1.0
        or ours.get("allocations") != ["BG", "RS"]
        or ours.get("target_context_ordinals") != list(range(6))
        or ours.get("physical_slope_context_ordinals") != list(range(6))
        or ours.get("nosoftroute") != "PROGRESS_SIMPLEX_NEUTRAL_V_EQUALS_ONE"
        or ours.get("soft_cost")
        != ["PINNED_STRUCTURAL_P", "FIXED_RANK_HISTORICAL_H"]
        or ours.get("functional_p_h_routing_influence") != 0
        or ours.get("per_layer_upper_cap_count") != 0
        or any(
            ours.get(key) != 0
            for key in ("retry_count", "backtracking_count", "early_stop_count")
        )
        or history.get("round_entry_counts") != list(range(0, 100, 10))
        or history.get("append_after_endpoint_commit") is not True
        or history.get("fixed_rank") != 100
        or history.get("retains_all_committed_cases_in_pilot") is not True
        or history.get("persistent_inactivity_hard_failure") is not True
        or evaluation.get("inner_k_heldout_count") != 0
        or evaluation.get("postcommit_cumulative_b10_evaluation_count") != 55
        or evaluation.get("teacher_kl_wide_locality_rounds") != [1, 5, 10]
        or evaluation.get("future_batch_controller_access_count") != 0
        or baseline.get("methods") != ["ALPHAEDIT"]
        or baseline.get("one_per_model") is not True
        or baseline.get("statistics_recompute_count") != 0
    ):
        raise ODEBFContractError("P1R23 Full-6 Historical method lock differs")


def load_and_validate_historical_h0_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_historical_h0_lock(value)
    return value, file_sha


__all__ = [
    "ALL_REQUEST_ORDER",
    "COMPUTE_NUMERICAL_ROOT",
    "EXECUTION_CHECKPOINT",
    "FRESH_SEAL_ROOT",
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "SCIENTIFIC_CHECKPOINT",
    "load_and_validate_historical_h0_lock",
    "validate_historical_h0_lock",
]
