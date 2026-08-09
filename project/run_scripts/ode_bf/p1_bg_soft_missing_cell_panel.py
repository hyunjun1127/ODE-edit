"""Contracts for the R10 BG-Soft missing-cell diagnostic."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import MODEL_ALIASES, ODEBFContractError
from .p1_common_coldcoord_fixed_e8_panel import (
    COMMON_COLD_CASE_SEAL_FILE,
    COMMON_COLD_NUMERICAL_LOCK_FILE,
    common_cold_schedule,
    common_cold_schedule_receipt,
    forecast_common_cold_panel as _forecast_common_cold_panel,
    load_common_cold_requests,
    validate_common_cold_runtime_gpu_capacity,
    verify_common_cold_case_seal,
)


BG_SOFT_INSTRUCTION_ID = (
    "ODEEDIT-S05-ODE-BF-BG-SOFT-MISSING-CELL-P1R12-V1"
)
BG_SOFT_AMENDMENT_ID = (
    "ODEEDIT-S05-ODE-BF-BG-SOFT-MISSING-CELL-P1R12-V1-A1"
)
BG_SOFT_SCHEMA_NAMESPACE = "ode-edit-s05-bg-soft-missing-cell-p1r12-a1"
BG_SOFT_PARENT_HEAD = "38592cd14eddbfd40cd063b1891c75588ddcb7fa"
BG_SOFT_EXECUTION_REPAIR_PARENT = (
    "8d0e8c80e3a100fccd4c295c3c3dbab5153602be"
)
BG_SOFT_RESULT_TOKEN = "bg-soft-missing-cell-p1r12-a1-v1"
BG_SOFT_REFERENCE_LOCK_FILE = "p1r12_bg_soft_reference_lock.json"
BG_SOFT_SOURCE_MANIFEST_FILE = (
    "source_manifest_s05_bg_soft_missing_cell_a1.json"
)
BG_SOFT_R10_CASE_ROOT = (
    "3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628"
)
BG_SOFT_R10_NUMERICAL_ROOT = (
    "8d9da5691882db02adc4965d3fe0ac91a1f06f67fa3d7b302d22d729d1a62c9f"
)
BG_SOFT_R10_NUMERICAL_SHA256 = (
    "7e09a1e75542c32f240e8591cd58db15c735e3aeb68d3edb3c694ca06de848ea"
)
BG_SOFT_A1_FORECAST_SECONDS = 82_800
BG_SOFT_A1_ALLOCATION_SECONDS = 86_340


def forecast_common_cold_panel(
    artifact_lock_path: Path,
    base_model_lock_path: Path,
    alias: str,
) -> Any:
    """Conservative two-arm A1 forecast including D1-D4."""

    parent = _forecast_common_cold_panel(
        artifact_lock_path, base_model_lock_path, alias
    )
    fits = bool(
        parent.conservative_gpu_peak_mib
        <= parent.allocatable_calibration_bytes // (1024 * 1024)
        and parent.conservative_host_peak_mib <= 65_000
        and BG_SOFT_A1_FORECAST_SECONDS <= BG_SOFT_A1_ALLOCATION_SECONDS
    )
    return replace(
        parent,
        arm_count=2,
        bootstrap_count=1,
        target_backward_batches_per_arm=16,
        conservative_time_seconds=BG_SOFT_A1_FORECAST_SECONDS,
        allocation_time_seconds=BG_SOFT_A1_ALLOCATION_SECONDS,
        fits_envelope=fits,
    )


def expected_bg_soft_result_name(alias: str) -> str:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("BG-Soft result alias differs")
    return f"s05-bg-soft-missing-cell-p1r12-a1-{alias}-v1"


def load_and_validate_bg_soft_reference_lock(
    path: Path,
    *,
    controller_identity_sha256: str,
    case_root_digest: str,
    schedule: Any,
) -> tuple[dict[str, Any], str]:
    value, file_sha256 = load_rooted_json(
        path,
        expected_schema=f"{BG_SOFT_SCHEMA_NAMESPACE}-reference-lock/v1",
    )
    expected = {
        "instruction_id": BG_SOFT_INSTRUCTION_ID,
        "amendment_id": BG_SOFT_AMENDMENT_ID,
        "parent_checkpoint": BG_SOFT_PARENT_HEAD,
        "case_root_digest": BG_SOFT_R10_CASE_ROOT,
        "r10_numerical_root_digest": BG_SOFT_R10_NUMERICAL_ROOT,
        "r10_numerical_file_sha256": BG_SOFT_R10_NUMERICAL_SHA256,
        "controller_geometry_identity_sha256": controller_identity_sha256,
        "schedule_identity_sha256": common_cold_schedule_receipt(schedule)[
            "identity_sha256"
        ],
        "live_arms": ["BG-SOFT", "BG-SOFT-TARGET-HOLD"],
        "scale": "MATCHED_BATCH_GLOBAL_SCALE_V1",
        "routing": "E8-SOFT",
        "joint_grid_count": 8,
        "joint_h": 0.125,
        "joint_tau_final": 1.0,
        "hard_h_p_budget_influence_count": 0,
        "first_hit_observation_only": True,
        "native_or_direct_z_cold_access_count": 0,
        "w0_native_bg_neutral_rerun_count": 0,
        "scientific_promotion_authorized": False,
        "diagnostics": {
            "D1": "OVERLAY_VS_NOHOOK_SIGNED_SLOPES",
            "D2": "POSTFREEZE_W_ONLY_VS_ADDITIVE_Z_ORACLE",
            "D3": "ENTRY_AND_BOUNDARY_SINGLE_WRITE_AUDIT",
            "D4": "TARGET_HOLD_CATCH_UP_KILL_TEST",
        },
    }
    if case_root_digest != BG_SOFT_R10_CASE_ROOT or any(
        value.get(key) != expected_value
        for key, expected_value in expected.items()
    ):
        raise ODEBFContractError("BG-Soft reference lock differs")
    aliases = value.get("aliases")
    if not isinstance(aliases, Mapping) or set(aliases) != set(MODEL_ALIASES):
        raise ODEBFContractError("BG-Soft alias references differ")
    for alias in MODEL_ALIASES:
        item = aliases[alias]
        if (
            not isinstance(item, Mapping)
            or item.get("initial_contract", {}).get("schema")
            != "ode-edit-s05-bg-soft-initial-contract/v1"
            or item.get("initial_contract", {}).get(
                "first_allowed_semantic_difference"
            )
            != "SELECTED_ROUTING_VELOCITY"
            or item.get("frozen_r10_reference", {}).get("alias") != alias
        ):
            raise ODEBFContractError("BG-Soft alias contract differs")
    return value, file_sha256


def bg_soft_initial_contract(
    lock: Mapping[str, Any], alias: str
) -> dict[str, Any]:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("BG-Soft initial alias differs")
    return dict(lock["aliases"][alias]["initial_contract"])


def bg_soft_frozen_reference(
    lock: Mapping[str, Any], alias: str
) -> dict[str, Any]:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("BG-Soft reference alias differs")
    return dict(lock["aliases"][alias]["frozen_r10_reference"])


__all__ = [
    "BG_SOFT_AMENDMENT_ID",
    "BG_SOFT_A1_ALLOCATION_SECONDS",
    "BG_SOFT_A1_FORECAST_SECONDS",
    "BG_SOFT_INSTRUCTION_ID",
    "BG_SOFT_EXECUTION_REPAIR_PARENT",
    "BG_SOFT_PARENT_HEAD",
    "BG_SOFT_REFERENCE_LOCK_FILE",
    "BG_SOFT_RESULT_TOKEN",
    "BG_SOFT_SCHEMA_NAMESPACE",
    "BG_SOFT_SOURCE_MANIFEST_FILE",
    "COMMON_COLD_CASE_SEAL_FILE",
    "COMMON_COLD_NUMERICAL_LOCK_FILE",
    "bg_soft_frozen_reference",
    "bg_soft_initial_contract",
    "common_cold_schedule",
    "expected_bg_soft_result_name",
    "forecast_common_cold_panel",
    "load_and_validate_bg_soft_reference_lock",
    "load_common_cold_requests",
    "validate_common_cold_runtime_gpu_capacity",
    "verify_common_cold_case_seal",
]
