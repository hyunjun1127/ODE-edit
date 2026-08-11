"""Panel and lock validation for P1R23 scalable batched ODE-BF."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import MODEL_ALIASES, ODEBFContractError, canonical_hash
from .p1_common_coldcoord_fixed_e8_panel import forecast_common_cold_panel
from .scalable_batched_runtime import (
    P1R23_GRID_COUNT,
    P1R23_H,
    P1R23_INSTRUCTION_ID,
    P1R23_LAYER_ORDER,
    P1R23_METHOD_ID,
)


P1R23_SCHEMA = "ode-edit-s05-scalable-batched-runtime-p1r23"
P1R23_LOCK_FILE = "numerical_lock_s05_scalable_batched_runtime.json"
P1R23_B10_RESULT_TOKEN = "scalable-batched-runtime-p1r23-b10-v1"
P1R23_B100_RESULT_TOKEN = "scalable-batched-runtime-p1r23-b100-v1"
P1R23_BATCH_SIZES = (10, 100)
P1R23_BG_ROUTING_ARMS = ("BG-NEUTRAL", "BG-SOFT")
P1R23_RS_ROUTING_ARMS = ("RS-NEUTRAL", "RS-SOFT")
P1R23_SIMPLEX_BG_ROUTING_ARMS = (
    "BG-SIMPLEX-NEUTRAL",
    "BG-SIMPLEX-SOFT",
)
P1R23_SIMPLEX_RS_ROUTING_ARMS = (
    "RS-SIMPLEX-NEUTRAL",
    "RS-SIMPLEX-SOFT",
)
P1R23_COMPUTE_SIMPLEX_BG_ROUTING_ARMS = (
    "BG-COMPUTE-SIMPLEX-NEUTRAL",
    "BG-COMPUTE-SIMPLEX-SOFT",
)
P1R23_COMPUTE_SIMPLEX_RS_ROUTING_ARMS = (
    "RS-COMPUTE-SIMPLEX-NEUTRAL",
    "RS-COMPUTE-SIMPLEX-SOFT",
)
P1R23_ROUTING_ARMS = P1R23_RS_ROUTING_ARMS + P1R23_BG_ROUTING_ARMS
P1R23_EXECUTION_ROLES = (
    "ODE_BF_K8_PAIR",
    "ODE_BF_K8_RS_PAIR",
    "PROGRESS_SIMPLEX_BG_PAIR",
    "PROGRESS_SIMPLEX_RS_PAIR",
    "COMPUTE_PROGRESS_SIMPLEX_BG_PAIR",
    "COMPUTE_PROGRESS_SIMPLEX_RS_PAIR",
    "OPTIMIZED_NATIVE_K1",
    "OFFICIAL_NATIVE",
    "CALIBRATION",
)
P1R23_FORECAST_SECONDS = 86_000
P1R23_ALLOCATION_SECONDS = 86_340


def forecast_p1r23_panel(
    artifact_lock_path: Path,
    base_model_lock_path: Path,
    alias: str,
) -> Any:
    """Peak-memory forecast for streaming P1R23 jobs.

    B100 changes the number of physical microbatches, not the live request
    microbatch geometry.  This intentionally keeps the conservative common
    cold peak while reserving the full allocation for the slowest role.
    """

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
        conservative_time_seconds=P1R23_FORECAST_SECONDS,
        allocation_time_seconds=P1R23_ALLOCATION_SECONDS,
        fits_envelope=bool(
            parent.conservative_gpu_peak_mib
            <= parent.allocatable_calibration_bytes // (1024 * 1024)
            and parent.conservative_host_peak_mib <= 65_000
            and P1R23_FORECAST_SECONDS <= P1R23_ALLOCATION_SECONDS
        ),
    )


def expected_p1r23_result_name(
    alias: str,
    *,
    batch_size: int,
    role: str,
    routing_arm: str | None = None,
) -> str:
    if alias not in MODEL_ALIASES or batch_size not in P1R23_BATCH_SIZES:
        raise ODEBFContractError("P1R23 result identity differs")
    if role in (
        "ODE_BF_K8_PAIR",
        "ODE_BF_K8_RS_PAIR",
        "PROGRESS_SIMPLEX_BG_PAIR",
        "PROGRESS_SIMPLEX_RS_PAIR",
        "COMPUTE_PROGRESS_SIMPLEX_BG_PAIR",
        "COMPUTE_PROGRESS_SIMPLEX_RS_PAIR",
    ):
        if routing_arm is not None or (
            (
                role.startswith("PROGRESS_SIMPLEX_")
                or role.startswith("COMPUTE_PROGRESS_SIMPLEX_")
            )
            and batch_size != 10
        ):
            raise ODEBFContractError("P1R23 paired ODE job forbids one-arm dispatch")
        suffix = {
            "ODE_BF_K8_PAIR": "ode-bf-k8-neutral-soft-pair",
            "ODE_BF_K8_RS_PAIR": "ode-bf-k8-rs-neutral-soft-pair",
            "PROGRESS_SIMPLEX_BG_PAIR": "progress-simplex-bg-neutral-soft-pair",
            "PROGRESS_SIMPLEX_RS_PAIR": "progress-simplex-rs-neutral-soft-pair",
            "COMPUTE_PROGRESS_SIMPLEX_BG_PAIR": "compute-progress-simplex-bg-neutral-soft-pair",
            "COMPUTE_PROGRESS_SIMPLEX_RS_PAIR": "compute-progress-simplex-rs-neutral-soft-pair",
        }[role]
    elif role in ("OPTIMIZED_NATIVE_K1", "OFFICIAL_NATIVE"):
        if routing_arm is not None:
            raise ODEBFContractError("P1R23 Native must not be arm-duplicated")
        suffix = role.lower().replace("_", "-")
    elif role == "CALIBRATION":
        if batch_size != 10 or routing_arm is not None:
            raise ODEBFContractError("P1R23 calibration identity differs")
        suffix = "calibration"
    else:
        raise ODEBFContractError("P1R23 result role differs")
    revision = (
        "compute-a1-v1"
        if role.startswith("COMPUTE_PROGRESS_SIMPLEX_")
        else "progress-simplex-v1"
        if role.startswith("PROGRESS_SIMPLEX_")
        else "tech-r2-v1"
    )
    return f"s05-p1r23-b{batch_size}-{alias}-{suffix}-{revision}"


def validate_p1r23_lock(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": f"{P1R23_SCHEMA}-lock/v1",
        "instruction_id": P1R23_INSTRUCTION_ID,
        "method_id": P1R23_METHOD_ID,
        "layer_order": list(P1R23_LAYER_ORDER),
        "stage_b_material_count": 0,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ODEBFContractError("P1R23 numerical lock identity differs")
    grid = value.get("scientific_grid")
    if not isinstance(grid, Mapping) or (
        grid.get("K") != P1R23_GRID_COUNT
        or grid.get("h") != P1R23_H
        or grid.get("tau_final") != 1.0
        or grid.get("retry_count") != 0
        or grid.get("backtracking_count") != 0
        or grid.get("static_split_count") != 0
    ):
        raise ODEBFContractError("P1R23 K8 grid lock differs")
    microbatch = value.get("microbatch_accumulation")
    if not isinstance(microbatch, Mapping) or (
        microbatch.get("mean_of_means_count") != 0
        or microbatch.get("global_divisor") != "request_count_exactly_once"
        or microbatch.get("mid_field_weight_write_count") != 0
        or microbatch.get("oom_retry_count") != 0
    ):
        raise ODEBFContractError("P1R23 microbatch lock differs")
    physical = value.get("physical_state")
    if not isinstance(physical, Mapping) or (
        physical.get("entry_relative_cumulative_reconstruction") is not True
        or physical.get("incremental_bf16_accumulation_count") != 0
        or physical.get("hot_hook_dense_assembly_count") != 0
        or physical.get("dynamic_target_key_slope_route_refresh_count") != 8
    ):
        raise ODEBFContractError("P1R23 physical-state lock differs")
    native = value.get("native_controls")
    if not isinstance(native, Mapping) or (
        native.get("optimized_native_target")
        != "SCIENTIFICALLY_IDENTICAL_DIRECT_Z_SOLVE"
        or native.get("optimized_native_one_gradient_substitution_count") != 0
        or native.get("optimized_native_accepted_materialization_count") != 1
    ):
        raise ODEBFContractError("P1R23 Native denominator lock differs")
    atomic = value.get("atomic_only")
    if not isinstance(atomic, Mapping) or (
        atomic.get("amendment_id")
        != "ODEEDIT-S05-ODE-BF-SCALABLE-BATCHED-RUNTIME-P1R23-V1-A3-ATOMIC-FIRST"
        or atomic.get("result_label") != "ATOMIC"
        or atomic.get("joint_batch_application_count") != 1
        or atomic.get("b100_sequential_b10_round_count") != 0
        or atomic.get("persistent_history_append_count") != 0
        or atomic.get("replay_h_decision_influence_count") != 0
        or atomic.get("sequential_controller_influence_count") != 0
        or atomic.get("p1r20_source_job_result_access_count") != 0
        or atomic.get("action_freeze_before_evaluation") is not True
        or atomic.get("exact_w0_restore_required") is not True
    ):
        raise ODEBFContractError("P1R23 atomic-only lock differs")
    b100 = value.get("b100_selection")
    if not isinstance(b100, Mapping) or (
        b100.get("atomic_request_order_sha256")
        != "3c8227494a9fcb5141f33993be0406e16c21aca1863bfd341ebede31a2a5ed47"
    ):
        raise ODEBFContractError("P1R23 B100 atomic order differs")


def load_and_validate_p1r23_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha256 = load_rooted_json(
        path, expected_schema=f"{P1R23_SCHEMA}-lock/v1"
    )
    validate_p1r23_lock(value)
    if value["root_digest"] != canonical_hash(
        {key: item for key, item in value.items() if key != "root_digest"}
    ):
        raise ODEBFContractError("P1R23 lock canonical digest differs")
    return value, file_sha256


__all__ = [
    "P1R23_B100_RESULT_TOKEN",
    "P1R23_B10_RESULT_TOKEN",
    "P1R23_BATCH_SIZES",
    "P1R23_BG_ROUTING_ARMS",
    "P1R23_EXECUTION_ROLES",
    "P1R23_LOCK_FILE",
    "P1R23_ROUTING_ARMS",
    "P1R23_RS_ROUTING_ARMS",
    "P1R23_SIMPLEX_BG_ROUTING_ARMS",
    "P1R23_SIMPLEX_RS_ROUTING_ARMS",
    "P1R23_COMPUTE_SIMPLEX_BG_ROUTING_ARMS",
    "P1R23_COMPUTE_SIMPLEX_RS_ROUTING_ARMS",
    "P1R23_SCHEMA",
    "expected_p1r23_result_name",
    "forecast_p1r23_panel",
    "load_and_validate_p1r23_lock",
    "validate_p1r23_lock",
]
