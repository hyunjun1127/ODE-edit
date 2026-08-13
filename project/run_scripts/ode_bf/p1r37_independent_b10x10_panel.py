"""Closed identities for the P1R37 no-persistent-freeze ablation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import MODEL_ALIASES, ODEBFContractError
from .p1r36_independent_b10x10_runtime import METHODS
from .p1r37_independent_b10x10_runtime import (
    expected_p1r37_independent_result_name,
)
from .p1r37_instantaneous_freeze import FREEZE_POLICY, INSTRUCTION_ID


LOCK_SCHEMA = "ode-edit-s05-p1r37-no-persistent-freeze-independent-b10x10-lock/v1"
LOCK_FILE = "numerical_lock_s05_p1r37_no_persistent_freeze_independent_b10x10.json"
P1R36_PARENT = "87efd168fc9680cabde878cd3b595e98db66d8fa"
P1R36_TREE = "0df5a4318b7886145e6b083d48db982de56b37ed"
P1R35_PARENT = "a625e3d1cded3ced0e9128ef7a44205953041447"


def expected_result_name(alias: str, method: str) -> str:
    return expected_p1r37_independent_result_name(alias, method)


def validate_lock(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema_version") != LOCK_SCHEMA
        or value.get("instruction_id") != INSTRUCTION_ID
        or value.get("p1r36_source_head") != P1R36_PARENT
        or value.get("p1r36_source_tree") != P1R36_TREE
        or value.get("p1r35_scientific_parent") != P1R35_PARENT
        or value.get("models") != list(MODEL_ALIASES)
        or value.get("methods") != list(METHODS)
        or value.get("fresh_stream_root")
        != "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6"
        or value.get("all_request_order_sha256")
        != "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c"
        or value.get("case_count_per_cell") != 10
        or value.get("request_count_per_case") != 10
        or value.get("grid_count") != 8
        or value.get("h") != 0.125
        or value.get("tau_final") != 1.0
        or value.get("history_mode") != "OFF"
        or value.get("freeze_threshold") != 0.05
        or value.get("freeze_policy") != FREEZE_POLICY
        or value.get("predeclared_sentinel")
        != "P1R36_LLAMA_RS_SOFT_CASE02_K7"
        or value.get("server2_project_gpu_cap") != 4
        or value.get("array_max_concurrent_gpu") != 2
        or value.get("scientific_promotion") is not False
    ):
        raise ODEBFContractError("P1R37 numerical/source identity differs")
    zero_keys = (
        "persistent_mask_decision_influence_count",
        "carried_frozen_input_count",
        "history_append_count",
        "history_replay_count",
        "retry_count",
        "backtracking_count",
        "early_stop_count",
        "inner_k_evaluator_access_count",
    )
    if any(value.get(key) != 0 for key in zero_keys):
        raise ODEBFContractError("P1R37 forbidden influence differs")


def load_and_validate_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_lock(value)
    return value, file_sha


__all__ = [
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "P1R35_PARENT",
    "P1R36_PARENT",
    "P1R36_TREE",
    "expected_result_name",
    "load_and_validate_lock",
    "validate_lock",
]
