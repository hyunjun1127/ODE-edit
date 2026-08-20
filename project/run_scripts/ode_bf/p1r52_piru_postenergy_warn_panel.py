"""Closed identities for the PIR-U post-energy-WARN sequential cell."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import ODEBFContractError
from .p1r52_piru_postenergy_warn import INSTRUCTION_ID
from .p1r52_piru_sequential_adapter import P1R52_PIRU_SEQUENTIAL_ROLE
from .p1r52_piru_sequential_panel import (
    STREAM_ORDER,
    STREAM_ROOT,
    verify_reference_results,
)


PARENT = "3f873f70a341b4d49f1552b55d132c43508c0609"
PARENT_TREE = "111be0fc080d5e0a5d8c5c43b689cfe6bcccde36"
LOCK_FILE = "numerical_lock_s05_p1r52_piru_postenergy_warn_r1_10xb100.json"
LOCK_SCHEMA = "ode-edit-s05-p1r52-piru-postenergy-warn-r1-10xb100-lock/v1"
SOURCE_MANIFEST_FILE = (
    "source_manifest_s05_p1r52_piru_postenergy_warn_r1_10xb100.json"
)
SOURCE_MANIFEST_SCHEMA = (
    "ode-edit-s05-p1r52-piru-postenergy-warn-r1-10xb100-source-manifest/v1"
)


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": INSTRUCTION_ID,
        "parent": PARENT,
        "parent_tree": PARENT_TREE,
        "model": "llama3-8b-inst",
        "role": P1R52_PIRU_SEQUENTIAL_ROLE,
        "round_count": 10,
        "batch_size": 100,
        "request_count": 1000,
        "k": 8,
        "h": 0.125,
        "writer_policy": "PIR-U",
        "gamma": 1,
        "alpha_cache": "ON",
        "structural_h": "ON_POSTSOLVE_ENERGY_WARN",
        "optimizer_energy_constraint": "UNCHANGED_ACTIVE",
        "postsolve_energy_residual_policy": "WARN_CONTINUE_IF_POSITIVE",
        "postsolve_energy_tolerance": 1.0e-12,
        "strength_p_nonnegative_solver_stage_policy": "UNCHANGED_HARD_FAIL",
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "batch_entry_evaluator_count": 0,
        "accepted_z_observation": "INLINE_AFTER_ACTION_FREEZE",
        "duplicate_physical_w_evaluator_forward_count": 0,
        "actual_layer_concentration_primary": (
            "SQRT_REALIZED_BF16_STEP_ENERGY_SHARE"
        ),
        "terminal_w0_restore_count": 1,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("PIR-U post-energy-WARN numerical lock differs")
    for name in (
        "replacement_energy_threshold_count",
        "polish_count",
        "coefficient_shrink_count",
        "retry_count",
        "backtracking_count",
        "line_search_count",
        "batch_entry_evaluator_count",
        "heldout_inner_access_count",
        "easyedit_write_count",
    ):
        if value.get(name) != 0:
            raise ODEBFContractError("PIR-U post-energy-WARN forbidden count differs")


def load_and_validate_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_lock(value)
    return value, file_sha


__all__ = [
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "PARENT",
    "PARENT_TREE",
    "SOURCE_MANIFEST_FILE",
    "SOURCE_MANIFEST_SCHEMA",
    "STREAM_ORDER",
    "STREAM_ROOT",
    "load_and_validate_lock",
    "validate_lock",
    "verify_reference_results",
]
