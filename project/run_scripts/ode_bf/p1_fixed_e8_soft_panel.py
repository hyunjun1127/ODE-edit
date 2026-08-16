"""Outcome-free panel, lock, and resource contracts for fixed E8 R8."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import MODEL_ALIASES, ODEBFContractError, canonical_hash
from .fixed_e8_soft_routing import (
    FIXED_E8_GRID_COUNT,
    FIXED_E8_H,
    FIXED_E8_KAPPA,
    FIXED_E8_METHOD_ID,
    FixedE8Arm,
    FixedE8OperationCeiling,
    fixed_e8_semantic_receipt,
)
from .p1_cold_structp_softp_noveto_panel import (
    PRIOR_P1R5_TECHNICAL_GPU,
    SERVER1_GPU_ALLOCATABLE_CALIBRATION_BYTES,
    SERVER1_GPU_PHYSICAL_TOTAL_BYTES,
    load_cold_requests,
    verify_cold_case_seal,
)
from .resource import forecast_p1_adaptive_b10_memory
from .sampling import LineageSeal, SamplingSeal, StatelessReplaySchedule


FIXED_E8_INSTRUCTION_ID = (
    "ODEEDIT-S05-ODE-BF-FIXED-E8-SOLVER-ISOLATION-CERT-R8-V1"
)
# These identities deliberately retain the already-sealed P1R7 scientific
# sample/order contract.  R8 changes solver ownership/certification only.
FIXED_E8_SCHEDULE_SEED_INSTRUCTION_ID = (
    "ODEEDIT-S05-ODE-BF-COLD-FIXED-E8-STRUCTFUNC-SOFT-P1R7-V1"
)
FIXED_E8_CASE_PROVENANCE_INSTRUCTION_ID = (
    "ODEEDIT-S05-ODE-BF-COLD-FIXED-E8-STRUCTFUNC-SOFT-P1R7-V1"
)
FIXED_E8_PARENT_HEAD = "dfdfd703cc4b580cf2c11fb84946cefbb5f30bb7"
FIXED_E8_EXECUTION_PARENT_HEAD = "26f0d4a9bc60378850d017e0beee262e70ddee94"
FIXED_E8_R8_SOLVER_PARENT_HEAD = "a90756b89e32edaa225aa4eb13c9f6e23b865339"
FIXED_E8_RESULT_TOKEN = "fixed-e8-solver-isolation-cert-r8-r2-v1"
FIXED_E8_SCHEMA_NAMESPACE = "ode-edit-s05-fixed-e8-solver-isolation-cert-r8"
FIXED_E8_PANEL_LABELS = tuple(item.value for item in FixedE8Arm)
FIXED_E8_CASE_SEAL_FILE = "p1r6_cold_cf_b10_seal.json"
FIXED_E8_CASE_PROVENANCE_FILE = "p1r7_fixed_e8_case_provenance.json"
FIXED_E8_ALLOCATION_SECONDS = 86_400
FIXED_E8_FORECAST_SECONDS = 43_200


def expected_fixed_e8_result_name(alias: str) -> str:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("fixed E8 result alias differs")
    return f"s05-fixed-e8-solver-isolation-cert-r8-r2-{alias}-v1"


def fixed_e8_schedule(base: SamplingSeal) -> StatelessReplaySchedule:
    lineages: list[LineageSeal] = []
    for lineage in base.lineages:
        seed = int(
            hashlib.sha256(
                f"{FIXED_E8_SCHEDULE_SEED_INSTRUCTION_ID}|{lineage.lineage.value}".encode(
                    "utf-8"
                )
            ).hexdigest()[:15],
            16,
        )
        lineages.append(
            LineageSeal(
                lineage.lineage,
                lineage.population_sha256,
                seed,
                lineage.sample_count,
                lineage.allowed_item_sha256,
                lineage.strata,
            )
        )
    return StatelessReplaySchedule(
        SamplingSeal(base.population_sha256, base.items, tuple(lineages))
    )


def fixed_e8_schedule_receipt(
    schedule: StatelessReplaySchedule,
) -> dict[str, Any]:
    payload = {
        "schema": f"{FIXED_E8_SCHEMA_NAMESPACE}-replay-schedule/v1",
        "instruction_id": FIXED_E8_INSTRUCTION_ID,
        "population_sha256": schedule.seal.population_sha256,
        "lineages": {
            item.lineage.value: {
                "seed": item.seed,
                "seal_sha256": item.identity(),
                "pool_count": len(item.allowed_item_sha256),
                "sample_count": item.sample_count,
            }
            for item in schedule.seal.lineages
        },
        "controller_terminal_distinct": True,
        "arm_identity_in_schedule": False,
        "model_identity_in_schedule": False,
        "state_digest": schedule.state_digest,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def verify_fixed_e8_case_provenance(
    value: Mapping[str, Any], *, source_seal: Mapping[str, Any]
) -> dict[str, Any]:
    payload = dict(value)
    observed_root = payload.pop("root_digest", None)
    if observed_root != canonical_hash(payload):
        raise ODEBFContractError("fixed E8 case provenance root differs")
    expected = {
        "schema": "ode-edit-s05-fixed-e8-p1r7-case-provenance/v1",
        "instruction_id": FIXED_E8_CASE_PROVENANCE_INSTRUCTION_ID,
        "source_seal_relative_path": (
            "project/run_scripts/ode_bf/locks/p1r6_cold_cf_b10_seal.json"
        ),
        "source_seal_instruction_id": source_seal["instruction_id"],
        "source_seal_root_digest": source_seal["root_digest"],
        "batch_ordered_request_digest_v1": source_seal[
            "batch_ordered_request_digest_v1"
        ],
        "request_count": len(source_seal["requests"]),
        "ordered_seal_reused_exactly": True,
        "fresh_selection_performed": False,
        "outcome_or_metric_access_count": 0,
        "scientific_promotion_authorized": False,
    }
    if payload != expected:
        raise ODEBFContractError("fixed E8 case provenance differs")
    payload["root_digest"] = observed_root
    return payload


def load_fixed_e8_case_provenance(
    path: Path, *, source_seal: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    """Load the rooted artifact whose canonical schema field is ``schema``."""

    value, raw_sha256 = load_rooted_json(path)
    return (
        verify_fixed_e8_case_provenance(value, source_seal=source_seal),
        raw_sha256,
    )


@dataclass(frozen=True, slots=True)
class FixedE8ResourceForecast:
    alias: str
    arm_count: int
    fields_per_arm: int
    candidates_per_arm: int
    functional_basis_endpoints_per_arm: int
    controller_candidate_functional_endpoints_per_arm: int
    terminal_audit_functional_endpoints_per_arm: int
    field_backward_batches_per_arm: int
    target_backward_batches_per_arm: int
    target_write_realization_forwards_per_arm: int
    qp_solves_per_arm: int
    qp_backend_invocations_per_arm: int
    action_rewrite_evaluations_per_arm: int
    postfreeze_unique_state_evaluations: int
    prior_technical_terminal_sha256: str
    prior_technical_peak_reserved_bytes: int
    streaming_incremental_gpu_mib: int
    runtime_safety_reserve_mib: int
    conservative_gpu_peak_mib: int
    conservative_host_peak_mib: int
    conservative_time_seconds: int
    physical_total_bytes: int
    allocatable_calibration_bytes: int
    host_allocation_memory_mib: int
    allocation_time_seconds: int
    fits_envelope: bool
    parent_forecast_sha256: str
    scientific_outcome_metric_used: bool

    def __post_init__(self) -> None:
        if (
            self.alias not in MODEL_ALIASES
            or self.arm_count != 2
            or self.fields_per_arm != 8
            or self.candidates_per_arm != 8
            or self.functional_basis_endpoints_per_arm != 48
            or self.controller_candidate_functional_endpoints_per_arm != 8
            or self.terminal_audit_functional_endpoints_per_arm != 8
            or self.field_backward_batches_per_arm != 8
            or self.target_backward_batches_per_arm != 8
            or self.target_write_realization_forwards_per_arm != 8
            or self.qp_solves_per_arm != 32
            or self.qp_backend_invocations_per_arm != 64
            or self.action_rewrite_evaluations_per_arm != 9
            or self.postfreeze_unique_state_evaluations != 18
            or len(self.prior_technical_terminal_sha256) != 64
            or self.prior_technical_peak_reserved_bytes <= 0
            or self.streaming_incremental_gpu_mib != 512
            or self.runtime_safety_reserve_mib != 8_192
            or self.physical_total_bytes != SERVER1_GPU_PHYSICAL_TOTAL_BYTES
            or self.allocatable_calibration_bytes
            != SERVER1_GPU_ALLOCATABLE_CALIBRATION_BYTES
            or self.conservative_gpu_peak_mib
            > self.allocatable_calibration_bytes // (1024 * 1024)
            or self.conservative_host_peak_mib > self.host_allocation_memory_mib
            or self.conservative_time_seconds > self.allocation_time_seconds
            or not self.fits_envelope
            or len(self.parent_forecast_sha256) != 64
            or self.scientific_outcome_metric_used
        ):
            raise ODEBFContractError("fixed E8 resource forecast differs")

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


def forecast_fixed_e8_panel(
    artifact_lock_path: Path,
    base_model_lock_path: Path,
    alias: str,
) -> FixedE8ResourceForecast:
    parent = forecast_p1_adaptive_b10_memory(
        artifact_lock_path, base_model_lock_path, alias
    )
    technical = PRIOR_P1R5_TECHNICAL_GPU[alias]
    gpu = (
        math.ceil(int(technical["peak_reserved_bytes"]) / (1024 * 1024))
        + 512
        + 8_192
    )
    host = max(parent.forecast_host_peak_mib + 2_048, 30_000)
    return FixedE8ResourceForecast(
        alias,
        2,
        8,
        8,
        48,
        8,
        8,
        8,
        8,
        8,
        32,
        64,
        9,
        18,
        str(technical["terminal_sha256"]),
        int(technical["peak_reserved_bytes"]),
        512,
        8_192,
        gpu,
        host,
        FIXED_E8_FORECAST_SECONDS,
        SERVER1_GPU_PHYSICAL_TOTAL_BYTES,
        SERVER1_GPU_ALLOCATABLE_CALIBRATION_BYTES,
        65_000,
        FIXED_E8_ALLOCATION_SECONDS,
        gpu
        <= SERVER1_GPU_ALLOCATABLE_CALIBRATION_BYTES // (1024 * 1024)
        and host <= 65_000
        and FIXED_E8_FORECAST_SECONDS <= FIXED_E8_ALLOCATION_SECONDS,
        parent.identity(),
        False,
    )


def validate_fixed_e8_runtime_gpu_capacity(
    forecast: FixedE8ResourceForecast,
    *,
    device_property_total_bytes: int,
    allocatable_total_bytes: int,
    free_bytes: int,
) -> dict[str, Any]:
    values = (device_property_total_bytes, allocatable_total_bytes, free_bytes)
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in values):
        raise ODEBFContractError("fixed E8 GPU capacity receipt differs")
    if device_property_total_bytes != forecast.allocatable_calibration_bytes:
        raise ODEBFContractError("fixed E8 stable GPU device identity differs")
    if (
        allocatable_total_bytes > device_property_total_bytes
        or free_bytes > allocatable_total_bytes
    ):
        raise ODEBFContractError("fixed E8 allocatable GPU capacity differs")
    # The conservative peak already includes the locked 8 GiB runtime reserve.
    # Do not charge that reserve a second time at the live-capacity gate.
    required = forecast.conservative_gpu_peak_mib * 1024 * 1024
    if free_bytes < required or allocatable_total_bytes < required:
        raise ODEBFContractError("fixed E8 runtime GPU capacity is insufficient")
    payload = {
        "stable_identity_api": "torch.cuda.get_device_properties.total_memory",
        "stable_device_total_bytes": device_property_total_bytes,
        "physical_inventory_total_bytes": forecast.physical_total_bytes,
        "physical_inventory_role": "LOCKED_FORECAST_PROVENANCE_ONLY",
        "runtime_capacity_api": "torch.cuda.mem_get_info",
        "allocatable_total_bytes": allocatable_total_bytes,
        "free_bytes": free_bytes,
        "required_free_bytes": required,
        "physical_and_allocatable_semantics_separate": True,
        "passed": True,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def validate_fixed_e8_lock(
    value: Mapping[str, Any],
    *,
    controller_identity_sha256: str,
    case_root_digest: str,
    population_root_digest: str,
    schedule: StatelessReplaySchedule,
) -> dict[str, Any]:
    payload = dict(value)
    observed_root = payload.pop("root_digest", None)
    if observed_root != canonical_hash(payload):
        raise ODEBFContractError("fixed E8 numerical lock root differs")
    expected = {
        "schema": f"{FIXED_E8_SCHEMA_NAMESPACE}-numerical-lock/v1",
        "instruction_id": FIXED_E8_INSTRUCTION_ID,
        "scientific_parent_checkpoint": FIXED_E8_PARENT_HEAD,
        "execution_parent_checkpoint": FIXED_E8_EXECUTION_PARENT_HEAD,
        "r8_solver_parent_checkpoint": FIXED_E8_R8_SOLVER_PARENT_HEAD,
        "method_id": FIXED_E8_METHOD_ID,
        "controller_geometry_identity_sha256": controller_identity_sha256,
        "case_root_digest": case_root_digest,
        "population_root_digest": population_root_digest,
        "schedule_identity_sha256": fixed_e8_schedule_receipt(schedule)[
            "identity_sha256"
        ],
        "arms": list(FIXED_E8_PANEL_LABELS),
        "grid_count": FIXED_E8_GRID_COUNT,
        "h": float(FIXED_E8_H),
        "tau_final": 1.0,
        "kappa": FIXED_E8_KAPPA,
        "functional_basis_endpoints_per_field": 6,
        "functional_basis_endpoints_per_arm": 48,
        "scientific_retry_count": 0,
        "scientific_rejection_count": 0,
        "hard_structural_h_budget_influence_count": 0,
        "hard_structural_p_budget_influence_count": 0,
        "functional_candidate_veto_influence_count": 0,
        "native_or_direct_z_cold_access_count": 0,
        "first_hit_observation_only": True,
        "persistent_commit_count": 0,
        "history_append_count": 0,
        "scientific_promotion_authorized": False,
        "method_semantic_identity_sha256": fixed_e8_semantic_receipt()[
            "identity_sha256"
        ],
        "operation_ceiling": FixedE8OperationCeiling().raw_free_payload(),
    }
    if payload != expected:
        raise ODEBFContractError("fixed E8 numerical lock differs")
    payload["root_digest"] = observed_root
    return payload


def load_and_validate_fixed_e8_lock(
    path: Path,
    *,
    controller_identity_sha256: str,
    case_root_digest: str,
    population_root_digest: str,
    schedule: StatelessReplaySchedule,
) -> tuple[dict[str, Any], str]:
    """Load the rooted numerical lock whose canonical field is ``schema``."""

    value, raw_sha256 = load_rooted_json(path)
    return (
        validate_fixed_e8_lock(
            value,
            controller_identity_sha256=controller_identity_sha256,
            case_root_digest=case_root_digest,
            population_root_digest=population_root_digest,
            schedule=schedule,
        ),
        raw_sha256,
    )


__all__ = [
    "FIXED_E8_CASE_SEAL_FILE",
    "FIXED_E8_CASE_PROVENANCE_FILE",
    "FIXED_E8_INSTRUCTION_ID",
    "FIXED_E8_EXECUTION_PARENT_HEAD",
    "FIXED_E8_R8_SOLVER_PARENT_HEAD",
    "FIXED_E8_PANEL_LABELS",
    "FIXED_E8_PARENT_HEAD",
    "FIXED_E8_RESULT_TOKEN",
    "expected_fixed_e8_result_name",
    "fixed_e8_schedule",
    "fixed_e8_schedule_receipt",
    "forecast_fixed_e8_panel",
    "load_cold_requests",
    "validate_fixed_e8_lock",
    "validate_fixed_e8_runtime_gpu_capacity",
    "verify_cold_case_seal",
    "verify_fixed_e8_case_provenance",
    "load_fixed_e8_case_provenance",
    "load_and_validate_fixed_e8_lock",
]
