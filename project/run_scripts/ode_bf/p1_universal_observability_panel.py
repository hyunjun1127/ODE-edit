"""Locks and planning helpers for the R13 universal-observability factorial."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import MODEL_ALIASES, ODEBFContractError
from .ode_bf_observability import (
    ODE_BF_OBSERVABILITY_CONTRACT_V1,
    R13_LIVE_CELL_IDS,
    paired_arm_ids,
    universal_observability_contract_receipt,
)
from .p1_bg_soft_missing_cell_panel import (
    BG_SOFT_REFERENCE_LOCK_FILE,
    bg_soft_frozen_reference,
    common_cold_schedule,
    forecast_common_cold_panel as _forecast_common_cold_panel,
    load_and_validate_bg_soft_reference_lock,
)
from .p1_common_coldcoord_fixed_e8_panel import (
    COMMON_COLD_CASE_SEAL_FILE,
    load_common_cold_requests,
    validate_common_cold_runtime_gpu_capacity,
    verify_common_cold_case_seal,
)


UNIVERSAL_OBS_INSTRUCTION_ID = (
    "ODEEDIT-S05-ODE-BF-UNIVERSAL-OBS-RS-BG-TARGETHOLD-P1R13-V1"
)
UNIVERSAL_OBS_AMENDMENT_ID = (
    "ODEEDIT-S05-ODE-BF-UNIVERSAL-OBS-RS-BG-TARGETHOLD-P1R13-V1-A1"
)
UNIVERSAL_OBS_SCHEMA_NAMESPACE = (
    "ode-edit-s05-universal-obs-rs-bg-targethold-p1r13-a1"
)
UNIVERSAL_OBS_BASE_RUNTIME = (
    "bfd77ddbf046333c59a8e589429d426fd5eee4c4"
)
UNIVERSAL_OBS_R10_CASE_ROOT = (
    "3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628"
)
UNIVERSAL_OBS_R10_NUMERICAL_ROOT = (
    "8d9da5691882db02adc4965d3fe0ac91a1f06f67fa3d7b302d22d729d1a62c9f"
)
UNIVERSAL_OBS_R12_REFERENCE_SHA256 = (
    "0f227472b92244861b48d9bd01561a3df282d0332ae03587f1c3897dbf2c5a5c"
)
UNIVERSAL_OBS_R12_PORTABLE_PACKAGE_ID = "BGSOFT_R10_FROZEN_BUNDLE_A1_R1"
UNIVERSAL_OBS_R12_PORTABLE_ARCHIVE_SHA256 = (
    "9defda0d6a21ea05a8bdcef8357249548740ad3eead21fc17709df376ea28295"
)
UNIVERSAL_OBS_R12_PORTABLE_MANIFEST_SHA256 = (
    "3e38ea8b48e91dca7f161ba2ee2015ba025c5f58d1c293c15086221a4ffd0b48"
)
UNIVERSAL_OBS_R12_PORTABLE_RECEIPT_SHA256 = (
    "84ea98a2dc4122f0e8369d3988487e7b195ecbb66f55dbd455e368371e643c76"
)
UNIVERSAL_OBS_R12_PORTABLE_MANIFEST_ROOT = (
    "8b67746e3595f32aaa6975452e8cea795b1b652cbac15054f7b01991a64f9ea6"
)
UNIVERSAL_OBS_R12_PORTABLE_RECEIPT_ROOT = (
    "d59024624d9f3c6bcb3e03e77150255aec086c95ec828710c506efd4b3453f4a"
)
UNIVERSAL_OBS_R12_PORTABLE_NORMALIZED_TREE = (
    "76e3ad535583d578194ecdcc33a8498c4d65e46302b314aa446fb0da7980d66f"
)
UNIVERSAL_OBS_R12_REFERENCE_CLOSURE_ROOT = (
    "127e48c7bd1af5b9b99c3e234bca690e24eb38c426987af8efcbb918eeb9975c"
)
UNIVERSAL_OBS_LOCK_FILE = "numerical_lock_s05_universal_observability.json"
UNIVERSAL_OBS_SOURCE_MANIFEST_FILE = (
    "source_manifest_s05_universal_observability.json"
)
UNIVERSAL_OBS_SESSION_SOURCE_PATHS = (
    "project/run_scripts/session05_ode_bf_submit_universal_observability.py",
    "project/run_scripts/session05_ode_bf_universal_observability.py",
    "project/run_scripts/session05_ode_bf_universal_observability.sbatch",
    "project/run_scripts/session05_ode_bf_universal_observability_dry_plan.py",
    "project/run_scripts/session05_ode_bf_universal_observability_package.py",
)
UNIVERSAL_OBS_RESULT_TOKEN = "universal-obs-rs-bg-targethold-p1r13-a1-v1"
UNIVERSAL_OBS_FORECAST_SECONDS = 82_800
UNIVERSAL_OBS_ALLOCATION_SECONDS = 86_340


def expected_universal_observability_result_name(
    alias: str, cell_id: str
) -> str:
    if alias not in MODEL_ALIASES or cell_id not in R13_LIVE_CELL_IDS:
        raise ODEBFContractError("universal observability result identity differs")
    slug = cell_id.lower().replace("-", "_")
    return f"s05-universal-obs-p1r13-a1-{slug}-{alias}-v1"


def forecast_universal_observability_panel(
    artifact_lock_path: Path,
    base_model_lock_path: Path,
    alias: str,
) -> Any:
    """Conservative one-pair forecast with D1--D4 attached to both cells."""

    parent = _forecast_common_cold_panel(
        artifact_lock_path, base_model_lock_path, alias
    )
    fits = bool(
        parent.conservative_gpu_peak_mib
        <= parent.allocatable_calibration_bytes // (1024 * 1024)
        and parent.conservative_host_peak_mib <= 65_000
        and UNIVERSAL_OBS_FORECAST_SECONDS <= UNIVERSAL_OBS_ALLOCATION_SECONDS
    )
    return replace(
        parent,
        arm_count=2,
        bootstrap_count=1,
        target_backward_batches_per_arm=16,
        conservative_time_seconds=UNIVERSAL_OBS_FORECAST_SECONDS,
        allocation_time_seconds=UNIVERSAL_OBS_ALLOCATION_SECONDS,
        fits_envelope=fits,
    )


def load_and_validate_universal_observability_lock(
    path: Path,
    *,
    controller_identity_sha256: str,
    case_root_digest: str,
    population_root_digest: str,
    schedule: Any,
    cell_id: str,
) -> tuple[dict[str, Any], str]:
    value, file_sha256 = load_rooted_json(
        path,
        expected_schema=f"{UNIVERSAL_OBS_SCHEMA_NAMESPACE}-lock/v1",
    )
    contract = universal_observability_contract_receipt(
        instruction_id=UNIVERSAL_OBS_INSTRUCTION_ID,
        amendment_id=UNIVERSAL_OBS_AMENDMENT_ID,
        cell_id=cell_id,
    )
    from .p1_common_coldcoord_fixed_e8_panel import (
        common_cold_schedule_receipt,
    )

    expected = {
        "instruction_id": UNIVERSAL_OBS_INSTRUCTION_ID,
        "amendment_id": UNIVERSAL_OBS_AMENDMENT_ID,
        "base_runtime": UNIVERSAL_OBS_BASE_RUNTIME,
        "case_root_digest": UNIVERSAL_OBS_R10_CASE_ROOT,
        "r10_numerical_root_digest": UNIVERSAL_OBS_R10_NUMERICAL_ROOT,
        "controller_geometry_identity_sha256": controller_identity_sha256,
        "population_root_digest": population_root_digest,
        "schedule_identity_sha256": common_cold_schedule_receipt(schedule)[
            "identity_sha256"
        ],
        "observability_contract_identity": (
            ODE_BF_OBSERVABILITY_CONTRACT_V1
        ),
        "factorial_shape": [2, 2, 2],
        "live_cells": list(R13_LIVE_CELL_IDS),
        "reused_r12_cell": "BG-SOFT",
        "r12_reference_lock_sha256": UNIVERSAL_OBS_R12_REFERENCE_SHA256,
        "r12_portable_package_id": UNIVERSAL_OBS_R12_PORTABLE_PACKAGE_ID,
        "r12_portable_archive_sha256": UNIVERSAL_OBS_R12_PORTABLE_ARCHIVE_SHA256,
        "r12_portable_manifest_sha256": UNIVERSAL_OBS_R12_PORTABLE_MANIFEST_SHA256,
        "r12_portable_receipt_sha256": UNIVERSAL_OBS_R12_PORTABLE_RECEIPT_SHA256,
        "r12_portable_manifest_root": UNIVERSAL_OBS_R12_PORTABLE_MANIFEST_ROOT,
        "r12_portable_receipt_root": UNIVERSAL_OBS_R12_PORTABLE_RECEIPT_ROOT,
        "r12_portable_normalized_tree": UNIVERSAL_OBS_R12_PORTABLE_NORMALIZED_TREE,
        "r12_reference_closure_root": UNIVERSAL_OBS_R12_REFERENCE_CLOSURE_ROOT,
        "joint_grid_count": 8,
        "joint_h": 0.125,
        "joint_tau_final": 1.0,
        "hard_h_p_budget_influence_count": 0,
        "first_hit_observation_only": True,
        "native_or_direct_z_cold_access_count": 0,
        "scientific_promotion_authorized": False,
    }
    if (
        cell_id not in R13_LIVE_CELL_IDS
        or case_root_digest != UNIVERSAL_OBS_R10_CASE_ROOT
        or any(value.get(key) != expected_value for key, expected_value in expected.items())
        or value.get("factorial_arm_registry") != contract["registered_arm_ids"]
        or value.get("cell_live_arms", {}).get(cell_id)
        != list(paired_arm_ids(cell_id))
    ):
        raise ODEBFContractError("universal observability numerical lock differs")
    return value, file_sha256


def load_and_validate_r12_frozen_reference(
    locks: Path,
    *,
    controller_identity_sha256: str,
    case_root_digest: str,
    schedule: Any,
    alias: str,
) -> tuple[dict[str, Any], str]:
    lock, file_sha256 = load_and_validate_bg_soft_reference_lock(
        locks / BG_SOFT_REFERENCE_LOCK_FILE,
        controller_identity_sha256=controller_identity_sha256,
        case_root_digest=case_root_digest,
        schedule=schedule,
    )
    if file_sha256 != UNIVERSAL_OBS_R12_REFERENCE_SHA256:
        raise ODEBFContractError("universal observability R12 reference bytes differ")
    return bg_soft_frozen_reference(lock, alias), file_sha256


__all__ = [
    "BG_SOFT_REFERENCE_LOCK_FILE",
    "COMMON_COLD_CASE_SEAL_FILE",
    "UNIVERSAL_OBS_ALLOCATION_SECONDS",
    "UNIVERSAL_OBS_AMENDMENT_ID",
    "UNIVERSAL_OBS_BASE_RUNTIME",
    "UNIVERSAL_OBS_FORECAST_SECONDS",
    "UNIVERSAL_OBS_INSTRUCTION_ID",
    "UNIVERSAL_OBS_LOCK_FILE",
    "UNIVERSAL_OBS_R12_PORTABLE_ARCHIVE_SHA256",
    "UNIVERSAL_OBS_R12_PORTABLE_MANIFEST_ROOT",
    "UNIVERSAL_OBS_R12_PORTABLE_MANIFEST_SHA256",
    "UNIVERSAL_OBS_R12_PORTABLE_NORMALIZED_TREE",
    "UNIVERSAL_OBS_R12_PORTABLE_PACKAGE_ID",
    "UNIVERSAL_OBS_R12_PORTABLE_RECEIPT_ROOT",
    "UNIVERSAL_OBS_R12_PORTABLE_RECEIPT_SHA256",
    "UNIVERSAL_OBS_R12_REFERENCE_CLOSURE_ROOT",
    "UNIVERSAL_OBS_RESULT_TOKEN",
    "UNIVERSAL_OBS_SCHEMA_NAMESPACE",
    "UNIVERSAL_OBS_SESSION_SOURCE_PATHS",
    "UNIVERSAL_OBS_SOURCE_MANIFEST_FILE",
    "common_cold_schedule",
    "expected_universal_observability_result_name",
    "forecast_universal_observability_panel",
    "load_and_validate_r12_frozen_reference",
    "load_and_validate_universal_observability_lock",
    "load_common_cold_requests",
    "validate_common_cold_runtime_gpu_capacity",
    "verify_common_cold_case_seal",
]
