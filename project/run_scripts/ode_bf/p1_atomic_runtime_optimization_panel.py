"""Frozen panel and numerical-lock identity for P1R22."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .atomic_runtime_optimization import P1R22_INSTRUCTION_ID, P1R22_METHOD_ID
from .contracts import MODEL_ALIASES, ODEBFContractError
from .p1_common_coldcoord_fixed_e8_panel import (
    common_cold_schedule_receipt,
    forecast_common_cold_panel,
)
from .p1_strength_preserving_router_panel import (
    STRENGTH_PRESERVING_BASE,
    STRENGTH_PRESERVING_BASE_TREE,
    STRENGTH_PRESERVING_REQUEST_ORDER,
    STRENGTH_PRESERVING_SEAL_ROOT,
)


P1R22_SCHEMA = "ode-edit-s05-atomic-runtime-optimization-p1r22"
P1R22_LOCK_FILE = "numerical_lock_s05_atomic_runtime_optimization.json"
P1R22_RESULT_TOKEN = "atomic-runtime-optimization-p1r22-v1"
P1R22_FORECAST_SECONDS = 21_600
P1R22_ALLOCATION_SECONDS = 86_340


def expected_p1r22_result_name(alias: str) -> str:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("P1R22 result alias differs")
    return f"s05-atomic-runtime-optimization-p1r22-{alias}-v1"


def expected_p1r22_conformance_result_name(alias: str) -> str:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("P1R22 conformance result alias differs")
    return f"s05-atomic-runtime-optimization-p1r22-c0c1-{alias}-tech-r3-v1"


def forecast_p1r22_panel(
    artifact_lock_path: Path,
    base_model_lock_path: Path,
    alias: str,
) -> Any:
    parent = forecast_common_cold_panel(
        artifact_lock_path, base_model_lock_path, alias
    )
    return replace(
        parent,
        arm_count=2,
        bootstrap_count=1,
        fields_per_arm=8,
        candidates_per_arm=8,
        functional_basis_endpoints_per_arm=40,
        target_backward_batches_per_arm=8,
        conservative_time_seconds=P1R22_FORECAST_SECONDS,
        allocation_time_seconds=P1R22_ALLOCATION_SECONDS,
        fits_envelope=bool(
            parent.conservative_gpu_peak_mib
            <= parent.allocatable_calibration_bytes // (1024 * 1024)
            and parent.conservative_host_peak_mib <= 65_000
            and P1R22_FORECAST_SECONDS <= P1R22_ALLOCATION_SECONDS
        ),
    )


def validate_p1r22_lock(value: Mapping[str, Any]) -> None:
    microbatch = value.get("request_microbatch_size")
    expected = {
        "schema_version": f"{P1R22_SCHEMA}-lock/v1",
        "instruction_id": P1R22_INSTRUCTION_ID,
        "method_id": P1R22_METHOD_ID,
        "scientific_base_checkpoint": STRENGTH_PRESERVING_BASE,
        "scientific_base_tree": STRENGTH_PRESERVING_BASE_TREE,
        "p1r19_scientific_checkpoint": (
            "657dafd40de50bb396fc9cfaf3862de1d07dd816"
        ),
        "p1r19_execution_checkpoint": (
            "d604953a046b6af4da7386dd6c64c20fc848f625"
        ),
        "stage_a_seal_root": STRENGTH_PRESERVING_SEAL_ROOT,
        "request_order_sha256": STRENGTH_PRESERVING_REQUEST_ORDER,
        "models": ["llama3-8b-inst", "qwen2.5-7b-inst"],
        "cells": ["BG-NEUTRAL", "BG-SOFT"],
        "grid_count": 8,
        "h": 0.125,
        "tau_final": 1.0,
        "hot_hook_dense_assembly_count": 0,
        "hot_hook_full_weight_hash_count": 0,
        "overlay_slope_model_forward_count": 0,
        "overlay_slope_backward_count": 0,
        "inner_field_empty_cache_count": 0,
        "inner_step_heldout_evaluation_count": 0,
        "candidate_objective_inner_k_count": 0,
        "candidate_progress_delayed_reuse_count_per_arm": 7,
        "terminal_w_only_objective_count_per_arm": 1,
        "w0_evaluation_count_per_model": 1,
        "authoritative_slope": "PHYSICAL_W_ONLY_NOHOOK",
        "scientific_promotion_authorized": False,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ODEBFContractError("P1R22 numerical lock differs")
    if (
        not isinstance(microbatch, Mapping)
        or set(microbatch) != set(MODEL_ALIASES)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in microbatch.values()
        )
        or any(60 % int(item) != 0 for item in microbatch.values())
    ):
        raise ODEBFContractError("P1R22 frozen request microbatch differs")


def load_and_validate_p1r22_lock(
    path: Path,
    *,
    case_root_digest: str,
    request_order_sha256: str,
    schedule: Any,
) -> tuple[dict[str, Any], str]:
    value, file_sha256 = load_rooted_json(
        path, expected_schema=f"{P1R22_SCHEMA}-lock/v1"
    )
    validate_p1r22_lock(value)
    if (
        case_root_digest != STRENGTH_PRESERVING_SEAL_ROOT
        or request_order_sha256 != STRENGTH_PRESERVING_REQUEST_ORDER
        or value.get("schedule_identity_sha256")
        != common_cold_schedule_receipt(schedule)["identity_sha256"]
    ):
        raise ODEBFContractError("P1R22 seal/schedule differs")
    return value, file_sha256


__all__ = [
    "P1R22_LOCK_FILE",
    "P1R22_RESULT_TOKEN",
    "expected_p1r22_result_name",
    "expected_p1r22_conformance_result_name",
    "forecast_p1r22_panel",
    "load_and_validate_p1r22_lock",
    "validate_p1r22_lock",
]
