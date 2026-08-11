"""Frozen panel and numerical lock for P1R20."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import ODEBFContractError


LOCK_FILE = "numerical_lock_s05_historical_h0_sequential.json"
LOCK_SCHEMA = "ode-edit-s05-historical-h0-bg-sequential-p1r20-lock/v1"
SCIENTIFIC_CHECKPOINT = "657dafd40de50bb396fc9cfaf3862de1d07dd816"
SCIENTIFIC_TREE = "028c49948665f9ebfbd2e713b9fd2b220c57bf49"
P1R19_NUMERICAL_ROOT = "1b0f7c5b9896e9cf688c5fc06ff0ef74cc2475584fa327dfc68fe17a5b27a7d9"
FRESH_SEAL_ROOT = "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6"
ALL_REQUEST_ORDER = "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c"


def validate_historical_h0_lock(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": "ODEEDIT-S05-ODE-BF-HISTORICAL-H0-BG-SEQUENTIAL-P1R20-V1",
        "scientific_checkpoint": SCIENTIFIC_CHECKPOINT,
        "scientific_tree": SCIENTIFIC_TREE,
        "launcher_only_provenance": "d604953a046b6af4da7386dd6c64c20fc848f625",
        "accepted_p1r19_numerical_lock_root": P1R19_NUMERICAL_ROOT,
        "accepted_p1r19_method_id": "R13-DYNAMIC-PHYSICAL-WONLY-STRENGTH-PRESERVING-ROUTER-V1",
        "fresh_seal_root": FRESH_SEAL_ROOT,
        "all_request_order_sha256": ALL_REQUEST_ORDER,
        "round_count": 10,
        "batch_size": 10,
        "request_count": 100,
        "models": ["llama3-8b-inst", "qwen2.5-7b-inst"],
        "methods": ["BG-NEUTRAL", "BG-SOFT-H", "MEMIT", "ALPHAEDIT"],
        "scientific_promotion_authorized": False,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ODEBFContractError("P1R20 numerical lock identity differs")
    ours = value.get("ours", {})
    history = value.get("history", {})
    evaluation = value.get("evaluation", {})
    baseline = value.get("baseline", {})
    if (
        ours.get("grid_count") != 8
        or ours.get("h") != 0.125
        or ours.get("tau_final") != 1.0
        or ours.get("allocation") != "BG"
        or ours.get("neutral_p_h_routing_influence") != 0
        or ours.get("soft_h_cost") != ["PINNED_STRUCTURAL_P", "HISTORICAL_H"]
        or ours.get("functional_p_h_routing_influence") != 0
        or any(ours.get(key) != 0 for key in ("retry_count", "backtracking_count", "early_stop_count"))
        or history.get("round_entry_counts") != list(range(0, 100, 10))
        or history.get("append_after_endpoint_commit") is not True
        or history.get("maximum_records") != 100
        or evaluation.get("inner_k_heldout_count") != 0
        or evaluation.get("future_batch_controller_access_count") != 0
        or baseline.get("fresh_sequential") is not True
        or baseline.get("joint_p1r18_reuse_count") != 0
        or baseline.get("statistics_recompute_count") != 0
    ):
        raise ODEBFContractError("P1R20 method/evaluation lock differs")


def load_and_validate_historical_h0_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_historical_h0_lock(value)
    return value, file_sha


__all__ = [
    "ALL_REQUEST_ORDER",
    "FRESH_SEAL_ROOT",
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "SCIENTIFIC_CHECKPOINT",
    "SCIENTIFIC_TREE",
    "load_and_validate_historical_h0_lock",
    "validate_historical_h0_lock",
]
