"""Locked panel and resource contracts for the P1R11 Alpha actuator test."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .canonical_alpha_posfield import (
    AlphaActuator,
    CANONICAL_ALPHA_POSFIELD_INSTRUCTION_ID,
    CANONICAL_ALPHA_POSFIELD_SCHEMA_NAMESPACE,
    canonical_alpha_posfield_source_contract,
)
from .common_cold_coordinate import COMMON_COLD_H, CommonColdScale
from .contracts import MODEL_ALIASES, ODEBFContractError, canonical_hash
from .p1_common_coldcoord_fixed_e8_panel import (
    COMMON_COLD_CASE_SEAL_FILE,
    COMMON_COLD_NUMERICAL_LOCK_FILE,
    forecast_common_cold_panel,
)
from .sampling import StatelessReplaySchedule


CANONICAL_ALPHA_PARENT_HEAD = "38592cd14eddbfd40cd063b1891c75588ddcb7fa"
CANONICAL_ALPHA_RESULT_TOKEN = "canonical-alpha-posfield-p1r11-v1"
CANONICAL_ALPHA_NUMERICAL_LOCK_FILE = (
    "numerical_lock_s05_canonical_alpha_posfield.json"
)
CANONICAL_ALPHA_NUMERICAL_ROOT = (
    "500525c392affd635b0925ef7a2f395c1b5f3e737d098f5768728aa1d9daca52"
)
CANONICAL_ALPHA_SOURCE_MANIFEST_FILE = (
    "source_manifest_s05_canonical_alpha_posfield.json"
)
CANONICAL_ALPHA_CASE_ROOT = (
    "3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628"
)
CANONICAL_ALPHA_R10_NUMERICAL_ROOT = (
    "8d9da5691882db02adc4965d3fe0ac91a1f06f67fa3d7b302d22d729d1a62c9f"
)
CANONICAL_ALPHA_R10_NUMERICAL_SHA256 = (
    "7e09a1e75542c32f240e8591cd58db15c735e3aeb68d3edb3c694ca06de848ea"
)


def parse_alpha_actuator(value: AlphaActuator | str) -> AlphaActuator:
    try:
        return AlphaActuator(value)
    except (TypeError, ValueError) as exc:
        raise ODEBFContractError("canonical Alpha actuator differs") from exc


def alpha_actuator_slug(value: AlphaActuator | str) -> str:
    return parse_alpha_actuator(value).value.lower().replace("-", "_")


def expected_canonical_alpha_result_name(
    alias: str,
    actuator: AlphaActuator | str,
) -> str:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("canonical Alpha result alias differs")
    return (
        "s05-canonical-alpha-posfield-p1r11-"
        f"{alias}-{alpha_actuator_slug(actuator)}-v1"
    )


@dataclass(frozen=True, slots=True)
class CanonicalAlphaResourceForecast:
    alias: str
    actuator: str
    arm_count: int
    bootstrap_count: int
    fields_per_arm: int
    candidates_per_arm: int
    functional_basis_endpoints_per_arm: int
    target_backward_batches_per_arm: int
    conservative_gpu_peak_mib: int
    conservative_host_peak_mib: int
    conservative_time_seconds: int
    allocation_time_seconds: int
    fits_envelope: bool
    parent_forecast_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


def forecast_canonical_alpha_panel(
    artifact_lock_path: Path,
    base_model_lock_path: Path,
    alias: str,
    actuator: AlphaActuator | str,
) -> CanonicalAlphaResourceForecast:
    selected = parse_alpha_actuator(actuator)
    parent = forecast_common_cold_panel(
        artifact_lock_path, base_model_lock_path, alias
    )
    # Keep the measured three-arm R10 envelope as a conservative one-arm cap.
    return CanonicalAlphaResourceForecast(
        alias=alias,
        actuator=selected.value,
        arm_count=1,
        bootstrap_count=1,
        fields_per_arm=8,
        candidates_per_arm=8,
        functional_basis_endpoints_per_arm=48,
        target_backward_batches_per_arm=8,
        conservative_gpu_peak_mib=parent.conservative_gpu_peak_mib,
        conservative_host_peak_mib=parent.conservative_host_peak_mib,
        conservative_time_seconds=parent.conservative_time_seconds,
        allocation_time_seconds=parent.allocation_time_seconds,
        fits_envelope=parent.fits_envelope,
        parent_forecast_sha256=canonical_hash(parent.raw_free_payload()),
    )


def validate_canonical_alpha_lock(
    value: Mapping[str, Any],
    *,
    controller_identity_sha256: str,
    case_root_digest: str,
    population_root_digest: str,
    schedule: StatelessReplaySchedule,
    actuator: AlphaActuator | str,
) -> dict[str, Any]:
    from .p1_common_coldcoord_fixed_e8_panel import common_cold_schedule_receipt

    selected = parse_alpha_actuator(actuator)
    payload = dict(value)
    root = payload.pop("root_digest", None)
    if root != CANONICAL_ALPHA_NUMERICAL_ROOT or root != canonical_hash(payload):
        raise ODEBFContractError("canonical Alpha numerical lock root differs")
    reproduction = payload.get("current_shared_r10_reproduction_by_alias")
    if (
        not isinstance(reproduction, Mapping)
        or set(reproduction) != set(MODEL_ALIASES)
        or any(
            not isinstance(reproduction[alias], Mapping)
            or reproduction[alias].get("accepted_snapshot_count")
            != len(reproduction[alias].get("snapshot_sha256", ()))
            or reproduction[alias].get("accepted_snapshot_count")
            != len(reproduction[alias].get("field_semantic_sha256", ()))
            or reproduction[alias].get("terminal_status")
            not in {
                "COMMON_COLD_FIXED_E8_TAU_COMPLETE",
                "ZERO_POSITIVE_DIRECTION",
            }
            for alias in MODEL_ALIASES
        )
    ):
        raise ODEBFContractError("canonical Alpha R10 reproduction lock differs")
    expected = {
        "schema": f"{CANONICAL_ALPHA_POSFIELD_SCHEMA_NAMESPACE}-numerical-lock/v1",
        "instruction_id": CANONICAL_ALPHA_POSFIELD_INSTRUCTION_ID,
        "parent_checkpoint": CANONICAL_ALPHA_PARENT_HEAD,
        "controller_geometry_identity_sha256": controller_identity_sha256,
        "case_root_digest": case_root_digest,
        "population_root_digest": population_root_digest,
        "schedule_identity_sha256": common_cold_schedule_receipt(schedule)[
            "identity_sha256"
        ],
        "r10_numerical_lock_file": COMMON_COLD_NUMERICAL_LOCK_FILE,
        "r10_numerical_lock_root": CANONICAL_ALPHA_R10_NUMERICAL_ROOT,
        "r10_numerical_lock_sha256": CANONICAL_ALPHA_R10_NUMERICAL_SHA256,
        "case_seal_file": COMMON_COLD_CASE_SEAL_FILE,
        "actuators": [item.value for item in AlphaActuator],
        "scale": CommonColdScale.ROBUST_SHARED.value,
        "routing_arm": "RS-NEUTRAL",
        "target_objective": "TARGET_NEW_NLL",
        "bootstrap_count": 1,
        "bootstrap_h": COMMON_COLD_H,
        "joint_grid_count": 8,
        "joint_h": COMMON_COLD_H,
        "joint_tau_target": 1.0,
        "first_hit_observation_only": True,
        "hard_h_p_budget_influence_count": 0,
        "scientific_retry_count": 0,
        "scientific_rejection_count": 0,
        "native_or_direct_z_cold_access_count": 0,
        "current_residual_policy": (
            "SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1"
        ),
        "current_shared_r10_reproduction_by_alias": reproduction,
        "canonical_residual_policy": (
            "CANONICAL_ORDERED_ALPHA_REMAINING_RESIDUAL_V1"
        ),
        "canonical_shadow_e8_transition_count": 0,
        "canonical_shadow_scientific_commit_count": 0,
        "method_semantic_identity_sha256": (
            canonical_alpha_posfield_source_contract()["identity_sha256"]
        ),
        "scientific_promotion_authorized": False,
    }
    # The same immutable lock serves both independent actuator jobs.
    if selected.value not in expected["actuators"] or payload != expected:
        raise ODEBFContractError("canonical Alpha numerical lock differs")
    payload["root_digest"] = root
    return payload


def load_and_validate_canonical_alpha_lock(
    path: Path,
    **kwargs: Any,
) -> tuple[dict[str, Any], str]:
    value, raw_sha = load_rooted_json(path)
    return validate_canonical_alpha_lock(value, **kwargs), raw_sha


__all__ = [
    "CANONICAL_ALPHA_CASE_ROOT",
    "CANONICAL_ALPHA_NUMERICAL_LOCK_FILE",
    "CANONICAL_ALPHA_NUMERICAL_ROOT",
    "CANONICAL_ALPHA_PARENT_HEAD",
    "CANONICAL_ALPHA_RESULT_TOKEN",
    "CANONICAL_ALPHA_SOURCE_MANIFEST_FILE",
    "alpha_actuator_slug",
    "expected_canonical_alpha_result_name",
    "forecast_canonical_alpha_panel",
    "load_and_validate_canonical_alpha_lock",
    "parse_alpha_actuator",
    "validate_canonical_alpha_lock",
]
