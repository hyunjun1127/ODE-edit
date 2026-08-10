"""Frozen panel identity for the P1R19 strength-preserving router."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import MODEL_ALIASES, ODEBFContractError
from .fixed_e8_soft_routing import (
    FIXED_E8_KKT_TOLERANCE,
    FIXED_E8_PRIMAL_TOLERANCE,
    FIXED_E8_SOLVER_FTOL,
    FIXED_E8_XI_TIE_TOLERANCE,
)
from .p1_common_coldcoord_fixed_e8_panel import (
    common_cold_schedule_receipt,
    forecast_common_cold_panel as _forecast_common_cold_panel,
)
from .strength_preserving_routing import (
    STRENGTH_PRESERVING_AMENDMENT_ID,
    STRENGTH_PRESERVING_INSTRUCTION_ID,
    STRENGTH_PRESERVING_METHOD_ID,
)


STRENGTH_PRESERVING_SCHEMA_NAMESPACE = (
    "ode-edit-s05-strength-preserving-router-p1r19-a2"
)
STRENGTH_PRESERVING_LOCK_FILE = (
    "numerical_lock_s05_strength_preserving_router.json"
)
STRENGTH_PRESERVING_BASE = "e6facd2d5dfae12d3c094b51981ad99951174109"
STRENGTH_PRESERVING_BASE_TREE = "a9b40eba586d9f54c67f5968ecdd395471d5a531"
STRENGTH_PRESERVING_SEAL_ROOT = (
    "3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628"
)
STRENGTH_PRESERVING_REQUEST_ORDER = (
    "984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b"
)
STRENGTH_PRESERVING_CELLS = (
    "RS-NEUTRAL",
    "RS-SOFT",
    "BG-NEUTRAL",
    "BG-SOFT",
)
STRENGTH_PRESERVING_RESULT_TOKEN = (
    "strength-preserving-router-p1r19-a2-tech-r1-v1"
)
STRENGTH_PRESERVING_FORECAST_SECONDS = 43_200
STRENGTH_PRESERVING_ALLOCATION_SECONDS = 86_340


def expected_strength_preserving_result_name(alias: str, cell: str) -> str:
    if alias not in MODEL_ALIASES or cell not in STRENGTH_PRESERVING_CELLS:
        raise ODEBFContractError("strength-preserving result identity differs")
    return (
        "s05-strength-preserving-router-p1r19-a2-"
        f"{cell.lower().replace('-', '_')}-{alias}-tech-r1-v1"
    )


def forecast_strength_preserving_panel(
    artifact_lock_path: Path,
    base_model_lock_path: Path,
    alias: str,
) -> Any:
    parent = _forecast_common_cold_panel(
        artifact_lock_path, base_model_lock_path, alias
    )
    return replace(
        parent,
        arm_count=1,
        bootstrap_count=1,
        target_backward_batches_per_arm=8,
        conservative_time_seconds=STRENGTH_PRESERVING_FORECAST_SECONDS,
        allocation_time_seconds=STRENGTH_PRESERVING_ALLOCATION_SECONDS,
        fits_envelope=bool(
            parent.conservative_gpu_peak_mib
            <= parent.allocatable_calibration_bytes // (1024 * 1024)
            and parent.conservative_host_peak_mib <= 65_000
            and STRENGTH_PRESERVING_FORECAST_SECONDS
            <= STRENGTH_PRESERVING_ALLOCATION_SECONDS
        ),
    )


def validate_strength_preserving_lock(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": f"{STRENGTH_PRESERVING_SCHEMA_NAMESPACE}-lock/v1",
        "instruction_id": STRENGTH_PRESERVING_INSTRUCTION_ID,
        "amendment_ids": [
            STRENGTH_PRESERVING_AMENDMENT_ID,
            "ODEEDIT-S05-ODE-BF-STRENGTH-PRESERVING-ROUTER-P1R19-V1-A2",
        ],
        "method_id": STRENGTH_PRESERVING_METHOD_ID,
        "base_checkpoint": STRENGTH_PRESERVING_BASE,
        "base_tree": STRENGTH_PRESERVING_BASE_TREE,
        "stage_a_seal_root": STRENGTH_PRESERVING_SEAL_ROOT,
        "request_order_sha256": STRENGTH_PRESERVING_REQUEST_ORDER,
        "cells": list(STRENGTH_PRESERVING_CELLS),
        "models": ["llama3-8b-inst", "qwen2.5-7b-inst"],
        "grid_count": 8,
        "h": 0.125,
        "tau_final": 1.0,
        "equality_tolerance": FIXED_E8_PRIMAL_TOLERANCE,
        "xi_tie_tolerance": FIXED_E8_XI_TIE_TOLERANCE,
        "solver_ftol": FIXED_E8_SOLVER_FTOL,
        "solver_kkt_tolerance": FIXED_E8_KKT_TOLERANCE,
        "target_probe_velocity_coefficients": [0.0] * 5,
        "authoritative_slope": "PHYSICAL_BF16_WONLY_TARGET_NEW_NLL_APPLIED_STEP",
        "structural_p_definition": (
            "tr(B_l*C0_l*B_l^T)_PINNED_WIKIPEDIA_SECOND_MOMENT"
        ),
        "router_added_model_forward_count": 0,
        "router_added_backward_count": 0,
        "router_added_target_backward_count": 0,
        "router_added_processed_token_count": 0,
        "hard_h_p_budget_influence_count": 0,
        "functional_veto_count": 0,
        "retry_count": 0,
        "backtracking_count": 0,
        "hold_arm_count": 0,
        "scientific_promotion_authorized": False,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ODEBFContractError("strength-preserving numerical lock differs")


def load_and_validate_strength_preserving_lock(
    path: Path,
    *,
    case_root_digest: str,
    request_order_sha256: str,
    schedule: Any,
) -> tuple[dict[str, Any], str]:
    value, file_sha256 = load_rooted_json(
        path,
        expected_schema=f"{STRENGTH_PRESERVING_SCHEMA_NAMESPACE}-lock/v1",
    )
    validate_strength_preserving_lock(value)
    if (
        case_root_digest != STRENGTH_PRESERVING_SEAL_ROOT
        or request_order_sha256 != STRENGTH_PRESERVING_REQUEST_ORDER
        or value.get("schedule_identity_sha256")
        != common_cold_schedule_receipt(schedule)["identity_sha256"]
    ):
        raise ODEBFContractError("strength-preserving seal/schedule differs")
    return value, file_sha256


__all__ = [
    "STRENGTH_PRESERVING_BASE",
    "STRENGTH_PRESERVING_BASE_TREE",
    "STRENGTH_PRESERVING_CELLS",
    "STRENGTH_PRESERVING_LOCK_FILE",
    "STRENGTH_PRESERVING_REQUEST_ORDER",
    "STRENGTH_PRESERVING_RESULT_TOKEN",
    "STRENGTH_PRESERVING_SCHEMA_NAMESPACE",
    "STRENGTH_PRESERVING_SEAL_ROOT",
    "expected_strength_preserving_result_name",
    "forecast_strength_preserving_panel",
    "load_and_validate_strength_preserving_lock",
    "validate_strength_preserving_lock",
]
