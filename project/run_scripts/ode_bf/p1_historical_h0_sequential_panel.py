"""Frozen lock validator for independent P1R23 B10x10 Atomic comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import ODEBFContractError


LOCK_FILE = "numerical_lock_s05_historical_h0_sequential.json"
LOCK_SCHEMA = "ode-edit-s05-p1r23-progress-simplex-independent-b10x10-lock/v1"
SCIENTIFIC_CHECKPOINT = "a343d1f6967ef37763009b92d227ade85cd93de0"
FRESH_SEAL_ROOT = "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6"
ALL_REQUEST_ORDER = "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c"


def validate_historical_h0_lock(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": "ODEEDIT-S05-P1R23-PROGRESS-SIMPLEX-INDEPENDENT-B10X10-V1",
        "accepted_progress_simplex_scientific_checkpoint": SCIENTIFIC_CHECKPOINT,
        "fresh_seal_root": FRESH_SEAL_ROOT,
        "all_request_order_sha256": ALL_REQUEST_ORDER,
        "case_count": 10,
        "batch_size": 10,
        "request_count": 100,
        "models": ["llama3-8b-inst", "qwen2.5-7b-inst"],
        "methods": [
            "BG-PROGRESS-SIMPLEX-NEUTRAL",
            "BG-PROGRESS-SIMPLEX-SOFT",
            "RS-PROGRESS-SIMPLEX-NEUTRAL",
            "RS-PROGRESS-SIMPLEX-SOFT",
            "OFFICIAL-ALPHAEDIT",
        ],
        "scientific_promotion_authorized": False,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ODEBFContractError("independent B10 lock identity differs")
    atomic = value.get("atomic", {})
    isolation = value.get("isolation", {})
    evaluation = value.get("evaluation", {})
    if (
        atomic.get("grid_count") != 8
        or atomic.get("h") != 0.125
        or atomic.get("tau_final") != 1.0
        or atomic.get("context_ordinals") != list(range(6))
        or atomic.get("cardinality") != "TEN_INDEPENDENT_WHOLE_B10"
        or atomic.get("progress_simplex_source_byte_identity") != SCIENTIFIC_CHECKPOINT
        or atomic.get("functional_p_routing_influence") is not True
        or any(atomic.get(key) != 0 for key in ("retry_count", "backtracking_count", "early_stop_count"))
        or isolation.get("W0_restore_after_every_case") is not True
        or isolation.get("cross_case_weight_state_count") != 0
        or isolation.get("history_append_count") != 0
        or isolation.get("history_replay_count") != 0
        or isolation.get("history_sketch_count") != 0
        or isolation.get("functional_h_influence_count") != 0
        or isolation.get("structural_h_influence_count") != 0
        or evaluation.get("per_case_action_freeze") is not True
        or evaluation.get("inner_k_evaluator_access_count") != 0
        or evaluation.get("future_batch_access_count") != 0
        or evaluation.get("official_alphaedit_per_model_case_count") != 10
    ):
        raise ODEBFContractError("independent B10 method lock differs")


def load_and_validate_historical_h0_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_historical_h0_lock(value)
    return value, file_sha


__all__ = ["ALL_REQUEST_ORDER", "FRESH_SEAL_ROOT", "LOCK_FILE", "LOCK_SCHEMA", "SCIENTIFIC_CHECKPOINT", "load_and_validate_historical_h0_lock", "validate_historical_h0_lock"]
