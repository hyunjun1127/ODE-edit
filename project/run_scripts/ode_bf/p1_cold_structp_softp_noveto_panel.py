"""Fresh-seal contracts for the S05 cold Structural-P/soft-P pilot."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import BATCH_SIZE, MODEL_ALIASES, ODEBFContractError, canonical_hash
from .p1_adaptive import AdaptiveVariant, adaptive_lock
from .p1_selection import _load_canonical_request_map, _sha256_file
from .request_digest import ordered_request_digest_v1
from .resource import forecast_p1_adaptive_b10_memory
from .sampling import LineageSeal, SamplingSeal, SampleLineage, StatelessReplaySchedule


COLD_INSTRUCTION_ID = (
    "ODEEDIT-S05-ODE-BF-COLDSTART-STRUCTP-SOFTP-NOVETO-FULLTAU-P1R6-V1"
)
COLD_SCHEMA_NAMESPACE = "ode-edit-s05-cold-structp-softp-noveto-p1r6"
COLD_RESULT_TOKEN = "cold-structp-softp-noveto-fulltau-p1r6-v1"
COLD_CASE_SALT = "ODEEDIT-S05-P1R6-CF-B10-V1"
COLD_PANEL_LABELS = (
    "COLD-FR-A8-NEWNLL-NOSOFT-NOVETO",
    "COLD-FR-A8-NEWNLL-SOFT-NOVETO",
)
COLD_PARENT_HEAD = "b43c25c57804353ad67c10185d49db6ad0e22aec"


def expected_cold_result_name(alias: str) -> str:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("cold result alias differs")
    return f"s05-cold-structp-softp-noveto-p1r6-{alias}-v1"


def _verify_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ODEBFContractError(f"cold {label} is not SHA-256")
    return value


def verify_cold_case_seal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    observed_root = payload.pop("root_digest", None)
    if observed_root != canonical_hash(payload):
        raise ODEBFContractError("cold case seal root differs")
    requests = payload.get("requests")
    expected_static = {
        "schema_version": "ode-edit-s05-cold-p1r6-fresh-cf-b10-seal/v1",
        "instruction_id": COLD_INSTRUCTION_ID,
        "status": "SEALED_BEFORE_MODEL_LOAD",
        "benchmark": "counterfact",
        "salt": COLD_CASE_SALT,
        "rank_formula": "sha256(utf8(salt + '|' + decimal_case_id))",
        "edit_batch_size": BATCH_SIZE,
        "sequential_batch_count": 1,
        "logical_edit_count": BATCH_SIZE,
        "heldout_fields_opened_during_selection": 0,
        "scientific_promotion_authorized": False,
    }
    if any(payload.get(key) != expected for key, expected in expected_static.items()):
        raise ODEBFContractError("cold case seal policy differs")
    if not isinstance(requests, list) or len(requests) != BATCH_SIZE:
        raise ODEBFContractError("cold case seal count differs")
    if [item.get("ordinal") for item in requests] != list(range(BATCH_SIZE)):
        raise ODEBFContractError("cold case order differs")
    case_ids = [item.get("case_id") for item in requests]
    request_ids = [item.get("request_sha256") for item in requests]
    collision_ids = [item.get("collision_sha256") for item in requests]
    ranks = [item.get("rank_sha256") for item in requests]
    if any(
        len(set(values)) != BATCH_SIZE
        for values in (case_ids, request_ids, collision_ids, ranks)
    ):
        raise ODEBFContractError("cold selected requests are not distinct")
    if not all(
        isinstance(case_id, int) and not isinstance(case_id, bool) and case_id >= 0
        for case_id in case_ids
    ):
        raise ODEBFContractError("cold case identity differs")
    for values, label in (
        (request_ids, "request identity"),
        (collision_ids, "collision identity"),
        (ranks, "rank identity"),
    ):
        for item in values:
            _verify_sha256(item, label)
    expected_ranks = [
        hashlib.sha256(f"{COLD_CASE_SALT}|{case_id}".encode("utf-8")).hexdigest()
        for case_id in case_ids
    ]
    if ranks != expected_ranks or ranks != sorted(ranks):
        raise ODEBFContractError("cold rank/order differs")
    ordered = ordered_request_digest_v1(tuple(request_ids))
    if payload.get("batch_ordered_request_digest_v1") != [ordered]:
        raise ODEBFContractError("cold ordered request digest differs")
    exclusion = payload.get("prior_exclusion")
    if (
        not isinstance(exclusion, Mapping)
        or exclusion.get("prior_case_id_count") != 431
        or exclusion.get("prior_request_sha256_count") != 434
        or exclusion.get("local_result_payload_open_count") != 0
        or exclusion.get("heldout_metric_open_count") != 0
    ):
        raise ODEBFContractError("cold prior exclusion contract differs")
    for key in (
        "prior_case_ids_sha256",
        "prior_request_sha256_sha256",
        "tracked_source_manifest_sha256",
        "exclusion_digest",
    ):
        _verify_sha256(exclusion.get(key), key)
    source = payload.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("relative_contract")
        != "EasyEdit/data/counterfact/counterfact.json"
        or source.get("size_bytes") != 45_108_470
        or source.get("row_count") != 21_919
        or source.get("sha256")
        != "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
    ):
        raise ODEBFContractError("cold dataset source differs")
    payload["root_digest"] = observed_root
    return payload


def load_cold_requests(
    dataset_path: Path, seal: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    value = verify_cold_case_seal(seal)
    dataset = dataset_path.resolve(strict=True)
    source = value["source"]
    if dataset.stat().st_size != source["size_bytes"] or _sha256_file(dataset) != source["sha256"]:
        raise ODEBFContractError("cold dataset bytes differ")
    approved = {
        str(item["request_sha256"]): (
            int(item["case_id"]),
            str(item["collision_sha256"]),
        )
        for item in value["requests"]
    }
    loaded = _load_canonical_request_map(dataset, approved)
    ordered = tuple(loaded[str(item["request_sha256"])] for item in value["requests"])
    if ordered_request_digest_v1(
        [str(item["request_sha256"]) for item in ordered]
    ) != value["batch_ordered_request_digest_v1"][0]:
        raise ODEBFContractError("cold loaded request order differs")
    return ordered


def cold_schedule(base: SamplingSeal) -> StatelessReplaySchedule:
    """Reuse the pinned population/pools with instruction-derived fresh seeds."""

    lineages: list[LineageSeal] = []
    for lineage in base.lineages:
        seed = int(
            hashlib.sha256(
                f"{COLD_INSTRUCTION_ID}|{lineage.lineage.value}".encode("utf-8")
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


def cold_schedule_receipt(schedule: StatelessReplaySchedule) -> dict[str, Any]:
    values = {
        item.lineage.value: {
            "seed": item.seed,
            "seal_sha256": item.identity(),
            "pool_count": len(item.allowed_item_sha256),
            "sample_count": item.sample_count,
        }
        for item in schedule.seal.lineages
    }
    return {
        "schema": "ode-edit-s05-cold-p1r6-replay-schedule/v1",
        "instruction_id": COLD_INSTRUCTION_ID,
        "population_sha256": schedule.seal.population_sha256,
        "lineages": values,
        "controller_terminal_distinct": True,
        "arm_identity_in_schedule": False,
        "model_identity_in_schedule": False,
        "state_digest": schedule.state_digest,
    }


@dataclass(frozen=True, slots=True)
class ColdResourceForecast:
    alias: str
    variant_count: int
    bootstrap_trial_cap: int
    joint_trial_cap_per_arm: int
    joint_field_cap_per_arm: int
    probes_per_field: int
    prior_technical_terminal_sha256: str
    prior_technical_peak_reserved_bytes: int
    cold_streaming_incremental_gpu_mib: int
    gpu_runtime_safety_reserve_mib: int
    conservative_gpu_peak_mib: int
    conservative_host_peak_mib: int
    conservative_time_seconds: int
    gpu_physical_total_bytes: int
    gpu_allocatable_calibration_bytes: int
    host_allocation_memory_mib: int
    allocation_time_seconds: int
    fits_envelope: bool
    parent_forecast_sha256: str
    scientific_outcome_metric_used: bool

    def __post_init__(self) -> None:
        if (
            self.alias not in MODEL_ALIASES
            or self.variant_count != 2
            or self.bootstrap_trial_cap != 32
            or self.joint_trial_cap_per_arm != 128
            or self.joint_field_cap_per_arm != 32
            or self.probes_per_field != 6
            or len(self.prior_technical_terminal_sha256) != 64
            or self.prior_technical_peak_reserved_bytes <= 0
            or self.cold_streaming_incremental_gpu_mib != 512
            or self.gpu_runtime_safety_reserve_mib != 8_192
            or self.gpu_physical_total_bytes != SERVER1_GPU_PHYSICAL_TOTAL_BYTES
            or self.gpu_allocatable_calibration_bytes
            != SERVER1_GPU_ALLOCATABLE_CALIBRATION_BYTES
            or self.conservative_gpu_peak_mib
            > self.gpu_allocatable_calibration_bytes // (1024 * 1024)
            or self.conservative_host_peak_mib > self.host_allocation_memory_mib
            or self.conservative_time_seconds > self.allocation_time_seconds
            or not self.fits_envelope
            or len(self.parent_forecast_sha256) != 64
            or self.scientific_outcome_metric_used
        ):
            raise ODEBFContractError("cold resource forecast differs")

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


SERVER1_GPU_PHYSICAL_TOTAL_BYTES = 51_527_024_640
SERVER1_GPU_ALLOCATABLE_CALIBRATION_BYTES = 50_899_386_368
PRIOR_P1R5_TECHNICAL_GPU = {
    "llama3-8b-inst": {
        "terminal_sha256": (
            "9b19aa025d27eaf54841ba7b0e99e3ef9ba778fc919dc34d524522a5ec920d88"
        ),
        "peak_reserved_bytes": 22_575_841_280,
    },
    "qwen2.5-7b-inst": {
        "terminal_sha256": (
            "2bd1404e2211098f9a51e4e7c92abf8894da5a860b9709d30e1b9b2869b9abba"
        ),
        "peak_reserved_bytes": 25_889_341_440,
    },
}


def forecast_cold_panel(
    artifact_lock_path: Path,
    base_model_lock_path: Path,
    alias: str,
) -> ColdResourceForecast:
    parent = forecast_p1_adaptive_b10_memory(
        artifact_lock_path, base_model_lock_path, alias
    )
    technical = PRIOR_P1R5_TECHNICAL_GPU[alias]
    # The immutable P1R5 terminal peak is a technical runtime calibration, not
    # a scientific outcome.  Cold target tensors/gradients and the shared
    # bootstrap state are streamed and bounded here by 512 MiB; an additional
    # common 8 GiB reserve covers allocator/library variance.  GPU capacity is
    # deliberately separate from Slurm's 65,000 MiB host-memory request.
    gpu = (
        math.ceil(int(technical["peak_reserved_bytes"]) / (1024 * 1024))
        + 512
        + 8_192
    )
    host = max(parent.forecast_host_peak_mib + 2_048, 30_000)
    # Worst-case source cap: 32 bootstrap trials plus two 128-trial arms and
    # 64 field probe sets, bounded by the authorized 24 h allocation.
    seconds = 82_800
    return ColdResourceForecast(
        alias,
        2,
        32,
        128,
        32,
        6,
        str(technical["terminal_sha256"]),
        int(technical["peak_reserved_bytes"]),
        512,
        8_192,
        gpu,
        host,
        seconds,
        SERVER1_GPU_PHYSICAL_TOTAL_BYTES,
        SERVER1_GPU_ALLOCATABLE_CALIBRATION_BYTES,
        65_000,
        86_400,
        gpu
        <= SERVER1_GPU_ALLOCATABLE_CALIBRATION_BYTES // (1024 * 1024)
        and host <= 65_000
        and seconds <= 86_400,
        parent.identity(),
        False,
    )


def validate_cold_runtime_gpu_capacity(
    forecast: ColdResourceForecast,
    *,
    physical_total_bytes: int,
    allocatable_total_bytes: int,
    free_bytes: int,
) -> dict[str, Any]:
    """Fail closed on stable device identity and live allocatable capacity."""

    values = (physical_total_bytes, allocatable_total_bytes, free_bytes)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ODEBFContractError("cold GPU capacity units differ")
    if physical_total_bytes != forecast.gpu_physical_total_bytes:
        raise ODEBFContractError("cold GPU physical identity differs")
    if (
        allocatable_total_bytes <= 0
        or free_bytes <= 0
        or free_bytes > allocatable_total_bytes
        or allocatable_total_bytes > physical_total_bytes
    ):
        raise ODEBFContractError("cold GPU allocatable capacity is invalid")
    required_bytes = forecast.conservative_gpu_peak_mib * 1024 * 1024
    if allocatable_total_bytes < required_bytes or free_bytes < required_bytes:
        raise ODEBFContractError("cold GPU free capacity is below forecast")
    return {
        "schema": "ode-edit-s05-cold-gpu-capacity/v1",
        "alias": forecast.alias,
        "physical_total_bytes": physical_total_bytes,
        "allocatable_total_bytes": allocatable_total_bytes,
        "free_bytes": free_bytes,
        "required_forecast_bytes": required_bytes,
        "physical_identity_stable": True,
        "runtime_capacity_pass": True,
        "host_memory_request_used_as_gpu_capacity": False,
    }


def validate_cold_lock(
    value: Mapping[str, Any],
    *,
    controller_identity_sha256: str,
    case_root_digest: str,
    population_root_digest: str,
    schedule: StatelessReplaySchedule,
) -> None:
    payload = dict(value)
    observed = payload.pop("root_digest", None)
    if observed != canonical_hash(payload):
        raise ODEBFContractError("cold numerical lock root differs")
    expected = {
        "schema_version": "ode-edit-s05-cold-structp-softp-noveto-p1r6-lock/v1",
        "instruction_id": COLD_INSTRUCTION_ID,
        "parent_source_head": COLD_PARENT_HEAD,
        "status": "FRESH_B10_PRE_SUBMIT_REVIEW_HOLD",
        "benchmark": "counterfact",
        "edit_batch_size": BATCH_SIZE,
        "sequential_batch_count": 1,
        "case_root_digest": case_root_digest,
        "p_population_root_digest": population_root_digest,
        "replay_schedule_state_digest": schedule.state_digest,
        "controller_identity_sha256": controller_identity_sha256,
        "panel_labels": list(COLD_PANEL_LABELS),
        "clock_variant": AdaptiveVariant.FR_A8.value,
        "clock_lock_sha256": adaptive_lock(AdaptiveVariant.FR_A8).identity(),
        "routing_objective": "TARGET_NEW_NLL",
        "target_state_objective": "COLD_TARGET_NEW_NLL_G_UNIT",
        "functional_p_candidate_veto_influence_count": 0,
        "structural_p_hard": True,
        "structural_h_hard": True,
        "numerical_trust_hard": True,
        "first_hit_observation_only": True,
        "full_horizon_tau": 1.0,
        "no_p_trust_factorial": True,
        "scientific_promotion_authorized": False,
    }
    if any(payload.get(key) != expected_value for key, expected_value in expected.items()):
        raise ODEBFContractError("cold numerical lock differs")
    bootstrap = payload.get("cold_bootstrap")
    if bootstrap != {
        "z_initialization": "canonical-W0-z-base",
        "metric": "per-request-I-over-z-base-norm-squared-fp64",
        "norm_squared_floor": 1.0e-12,
        "lambda_z": 0.0,
        "objective": "uniform-B10-target-new-suffix-NLL-over-1+5-contexts",
        "target_old_access_count": 0,
        "native_or_direct_z_access_count": 0,
        "delta_tau_initial_max": 0.125,
        "delta_tau_min": 0.0078125,
        "same_state_additional_retry_cap": 4,
        "accepted_step_cap": 8,
        "aggregate_trial_cap": 32,
        "per_request_g_radius": 0.25,
        "minimum_actual_decrease": 1.0e-8,
        "rho_accept": 0.1,
        "exit_joint_rank_min_exclusive": 1,
        "exit_nonzero_layer_count": 5,
        "exit_minimum_feasible_progress": 1.0e-8,
        "native_fallback": False,
    }:
        raise ODEBFContractError("cold bootstrap numerical contract differs")
    joint = payload.get("joint_cold_ode")
    if joint != {
        "full_current_residual": True,
        "target_metric_fixed_from_outer_z_base": True,
        "target_weight_shared_delta_tau": True,
        "delta_tau_max": 0.125,
        "delta_tau_min": 0.0078125,
        "same_state_additional_retry_cap": 4,
        "accepted_step_cap": 32,
        "aggregate_trial_cap": 128,
        "minimum_actual_progress": 1.0e-8,
        "rho_accept": 0.1,
        "functional_p_observed_every_candidate": True,
        "functional_p_candidate_veto": False,
        "soft_hinge_reference": 0.001,
        "structural_p_hard": True,
        "first_hit_observation_only": True,
        "persistent_endpoint_commit_count": 0,
        "history_append_count": 0,
    }:
        raise ODEBFContractError("cold joint numerical contract differs")


def cold_source_manifest_paths() -> tuple[str, ...]:
    return (
        "project/run_scripts/ode_bf/cold_start_target.py",
        "project/run_scripts/ode_bf/p1_cold_structp_softp_noveto_panel.py",
        "project/run_scripts/ode_bf/p1_adaptive_runtime.py",
        "project/run_scripts/ode_bf/p1_runtime.py",
        "project/run_scripts/ode_bf/functional_p_secant.py",
        "project/run_scripts/ode_bf/target_new_nll.py",
        "project/run_scripts/ode_bf/locks/p1r6_cold_cf_b10_seal.json",
        "project/run_scripts/ode_bf/locks/numerical_lock_s05_cold_structp_softp_noveto.json",
        "project/run_scripts/ode_bf/locks/p0_artifact_lock.json",
        "project/run_scripts/ode_bf/tests/test_cold_start_target.py",
        "project/run_scripts/session05_ode_bf_cold_structp_softp_noveto.py",
        "project/run_scripts/session05_ode_bf_cold_structp_softp_noveto_dry_plan.py",
        "project/run_scripts/session05_ode_bf_cold_structp_softp_noveto.sbatch",
        "project/run_scripts/session05_ode_bf_submit_cold_structp_softp_noveto.py",
    )
