"""Closed identities for P1R24 RS-Soft independent B10x10 execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import MODEL_ALIASES, ODEBFContractError


LOCK_SCHEMA = "ode-edit-s05-p1r24-rs-soft-independent-b10x10-lock/v1"
LOCK_FILE = "numerical_lock_s05_p1r24_independent_b10x10.json"
PARENT = "ce8c6c36348752f1407f7d713d30e6b5c727379b"
METHOD = "RS-P1R24-SOFT"


def expected_result_name(alias: str) -> str:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("P1R24 independent alias differs")
    return f"s05-p1r24-rs-soft-independent-b10x10-{alias}-tech-r1-v1"


def validate_lock(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema_version") != LOCK_SCHEMA
        or value.get("instruction_id")
        != "ODEEDIT-S05-P1R24-RS-SOFT-INDEPENDENT-B10X10-V1"
        or value.get("accepted_p1r24_scientific_checkpoint") != PARENT
        or value.get("method") != METHOD
        or value.get("fresh_stream_root")
        != "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6"
        or value.get("all_request_order_sha256")
        != "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c"
        or value.get("models") != list(MODEL_ALIASES)
        or value.get("case_count_per_model") != 10
        or value.get("request_count_per_case") != 10
        or value.get("grid_count") != 8
        or value.get("h") != 0.125
        or value.get("tau_final") != 1.0
        or value.get("context_ordinals") != list(range(6))
        or value.get("history_mode") != "OFF"
        or value.get("project_gpu_cap") != 3
        or value.get("array_max_concurrent_gpu") != 2
        or value.get("scientific_promotion_authorized") is not False
    ):
        raise ODEBFContractError("P1R24 independent lock identity differs")
    zero_keys = (
        "history_append_count",
        "history_replay_count",
        "history_sketch_count",
        "functional_h_influence_count",
        "structural_h_influence_count",
        "cross_case_weight_state_count",
        "retry_count",
        "backtracking_count",
        "early_stop_count",
        "inner_k_evaluator_access_count",
    )
    if any(value.get(key) != 0 for key in zero_keys):
        raise ODEBFContractError("P1R24 independent forbidden influence differs")


def load_and_validate_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_lock(value)
    return value, file_sha


__all__ = ["LOCK_FILE", "LOCK_SCHEMA", "METHOD", "PARENT", "expected_result_name", "load_and_validate_lock", "validate_lock"]
