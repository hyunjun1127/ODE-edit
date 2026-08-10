"""Planning and lock contracts for the P1R17 full-drive 2x2 panel."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import MODEL_ALIASES, ODEBFContractError, canonical_hash
from .full_drive_dynamic_runtime import (
    FULL_DRIVE_CELL_ORDER,
    FullDriveCell,
    full_drive_runtime_contract,
)
from .full_drive_soft_routing import (
    FULL_DRIVE_LAMBDA_GRID,
    FULL_DRIVE_METHOD_ID,
    LambdaParetoLock,
    full_drive_numerical_contract,
)


FULL_DRIVE_INSTRUCTION_ID = (
    "ODEEDIT-S05-ODE-BF-FULLDRIVE-SOFT-DYNAMIC2X2-P1R17-V1"
)
FULL_DRIVE_SCHEMA_NAMESPACE = (
    "ode-edit-s05-ode-bf-fulldrive-soft-dynamic2x2-p1r17-v1"
)
FULL_DRIVE_BASE_CHECKPOINT = "e6facd2d5dfae12d3c094b51981ad99951174109"
FULL_DRIVE_BASE_TREE = "a9b40eba586d9f54c67f5968ecdd395471d5a531"
FULL_DRIVE_STAGE_A_SEAL_ROOT = (
    "3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628"
)
FULL_DRIVE_REQUEST_ORDER = (
    "984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b"
)
FULL_DRIVE_TASK_GPU_CAP = 4
FULL_DRIVE_LAMBDA_LOCK_FILE = "p1r17_lambda_pareto_lock.json"
FULL_DRIVE_MODEL_LAMBDA_LOCK_FILE = "p1r17_model_lambda_lock.json"
FULL_DRIVE_NUMERICAL_LOCK_FILE = "numerical_lock_s05_p1r17_full_drive.json"
FULL_DRIVE_SOURCE_MANIFEST_FILE = "source_manifest_s05_p1r17_full_drive.json"
FULL_DRIVE_RESULT_TOKEN = "full-drive-soft-dynamic2x2-p1r17-tech-r1-v1"
FULL_DRIVE_ALLOCATION_SECONDS = 86_340


@dataclass(frozen=True, slots=True)
class FullDriveResourceForecast:
    alias: str
    cell: str
    arm_count: int
    field_count: int
    physical_edit_objective_endpoints_per_field: int
    functional_basis_endpoints_per_field: int
    target_velocity_backward_batches_per_field: int
    postfreeze_snapshot_count: int
    requested_gpu_count: int
    requested_cpu_count: int
    requested_memory_mib: int
    allocation_seconds: int
    conservative_time_seconds: int
    conservative_gpu_peak_mib: int
    conservative_host_peak_mib: int
    fits_envelope: bool
    evidence: Mapping[str, Any]
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        identity = payload.pop("identity_sha256")
        if canonical_hash(payload) != identity:
            raise ODEBFContractError("full-drive forecast identity differs")
        payload["identity_sha256"] = identity
        return payload


def expected_full_drive_result_name(alias: str, cell: FullDriveCell | str) -> str:
    selected = FullDriveCell(cell)
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("full-drive alias differs")
    return (
        "s05-p1r17-full-drive-"
        f"{selected.value.lower().replace('-', '_')}-{alias}-tech-r1-v1"
    )


def build_full_drive_forecast(
    alias: str,
    cell: FullDriveCell | str,
    *,
    prior_observed_peak_mib: int,
    prior_observed_host_mib: int,
    prior_observed_seconds: int,
) -> FullDriveResourceForecast:
    """Build a conservative one-cell forecast from immutable R13 resources.

    P1R17 runs one trajectory per job rather than R13's paired Dynamic/Hold
    job.  The wall projection keeps the full immutable R13 paired-cell time
    as its ceiling instead of dividing it, while adding no memory discount.
    """

    selected = FullDriveCell(cell)
    values = (
        prior_observed_peak_mib,
        prior_observed_host_mib,
        prior_observed_seconds,
    )
    if alias not in MODEL_ALIASES or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in values
    ):
        raise ODEBFContractError("full-drive forecast evidence differs")
    conservative_seconds = max(prior_observed_seconds, 82_800)
    gpu_peak = prior_observed_peak_mib
    host_peak = max(prior_observed_host_mib, 32_000)
    evidence = {
        "source": "IMMUTABLE_R13_PAIRED_DYNAMIC_HOLD_JOB_RESOURCE_CEILING",
        "one_cell_one_trajectory_per_job": True,
        "time_discount_from_paired_parent": 0,
        "memory_discount_from_paired_parent": 0,
        "physical_slope_endpoint_count": 6,
        "functional_basis_endpoint_count": 6,
        "terminal_evaluation_separate": True,
    }
    payload = {
        "alias": alias,
        "cell": selected.value,
        "arm_count": 1,
        "field_count": 8,
        "physical_edit_objective_endpoints_per_field": 6,
        "functional_basis_endpoints_per_field": 6,
        "target_velocity_backward_batches_per_field": 10,
        "postfreeze_snapshot_count": 9,
        "requested_gpu_count": 1,
        "requested_cpu_count": 8,
        "requested_memory_mib": 65_000,
        "allocation_seconds": FULL_DRIVE_ALLOCATION_SECONDS,
        "conservative_time_seconds": conservative_seconds,
        "conservative_gpu_peak_mib": gpu_peak,
        "conservative_host_peak_mib": host_peak,
        "fits_envelope": (
            conservative_seconds <= FULL_DRIVE_ALLOCATION_SECONDS
            and gpu_peak <= 65_000
            and host_peak <= 65_000
        ),
        "evidence": evidence,
    }
    return FullDriveResourceForecast(
        alias,
        selected.value,
        1,
        8,
        6,
        6,
        10,
        9,
        1,
        8,
        65_000,
        FULL_DRIVE_ALLOCATION_SECONDS,
        conservative_seconds,
        gpu_peak,
        host_peak,
        bool(payload["fits_envelope"]),
        evidence,
        canonical_hash(payload),
    )


def validate_full_drive_lambda_lock(
    path: Path,
    *,
    expected_closure_root: str,
) -> tuple[dict[str, Any], str]:
    value, file_sha256 = load_rooted_json(
        path,
        expected_schema="ode-edit-s05-p1r17-lambda-pareto-lock/v1",
    )
    required = {
        "lambda_grid": list(FULL_DRIVE_LAMBDA_GRID),
        "outcome_model_access_count": 0,
        "model_forward_count": 0,
        "gpu_action_count": 0,
        "weight_write_count": 0,
    }
    source = value.get("source")
    if (
        any(value.get(key) != expected for key, expected in required.items())
        or not isinstance(source, Mapping)
        or source.get("alias") != "llama3-8b-inst"
        or source.get("allocation") != "RS"
        or source.get("routing") != "NEUTRAL"
        or source.get("target_dynamics") != "DYNAMIC_TARGET"
        or source.get("step_index") != 0
        or source.get("closure_root_digest") != expected_closure_root
        or value.get("selected_lambda") in (
            FULL_DRIVE_LAMBDA_GRID[0],
            FULL_DRIVE_LAMBDA_GRID[-1],
        )
    ):
        raise ODEBFContractError("full-drive lambda lock differs")
    return value, file_sha256


def validate_model_lambda_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha256 = load_rooted_json(
        path,
        expected_schema=(
            "ode-edit-s05-p1r17-model-calibrated-lambda-lock/v1"
        ),
    )
    lambdas = value.get("lambda_by_alias")
    replays = value.get("model_replays")
    if (
        value.get("policy") != "MODEL_CALIBRATED_LAMBDA"
        or set(lambdas or {}) != set(MODEL_ALIASES)
        or set(replays or {}) != set(MODEL_ALIASES)
        or value.get("selection_algorithm_identical_across_models") is not True
        or value.get("fixed_across_allocation_and_k8_within_model") is not True
        or value.get("outcome_access_count") != 0
        or value.get("production_model_forward_count") != 0
        or value.get("production_weight_write_count") != 0
    ):
        raise ODEBFContractError("full-drive model lambda lock differs")
    for alias in MODEL_ALIASES:
        selected = lambdas[alias]
        replay = replays[alias]
        if (
            not isinstance(selected, (int, float))
            or isinstance(selected, bool)
            or selected < 0.0
            or replay.get("selected_lambda") != selected
            or replay.get("source", {}).get("alias") != alias
            or replay.get("source", {}).get("allocation") != "RS"
            or replay.get("source", {}).get("routing") != "NEUTRAL"
            or replay.get("source", {}).get("target_dynamics")
            != "DYNAMIC_TARGET"
            or replay.get("source", {}).get("step_index") != 0
            or replay.get("outcome_model_access_count") != 0
            or replay.get("production_outcome_access_count") != 0
        ):
            raise ODEBFContractError("full-drive model lambda replay differs")
    return value, file_sha256


def full_drive_panel_contract(lambda_lock: LambdaParetoLock) -> dict[str, Any]:
    payload = {
        "instruction_id": FULL_DRIVE_INSTRUCTION_ID,
        "method_id": FULL_DRIVE_METHOD_ID,
        "base_checkpoint": FULL_DRIVE_BASE_CHECKPOINT,
        "base_tree": FULL_DRIVE_BASE_TREE,
        "stage_a_seal_root": FULL_DRIVE_STAGE_A_SEAL_ROOT,
        "request_order": FULL_DRIVE_REQUEST_ORDER,
        "cell_order": [item.value for item in FULL_DRIVE_CELL_ORDER],
        "target_dynamics": ["DYNAMIC_TARGET"],
        "factorial_shape": [2, 2, 1],
        "lambda_lock_sha256": lambda_lock.identity_sha256,
        "selected_lambda": lambda_lock.selected_lambda,
        "runtime_contract": full_drive_runtime_contract(),
        "numerical_contract": full_drive_numerical_contract(),
        "task_gpu_cap": FULL_DRIVE_TASK_GPU_CAP,
        "scientific_promotion_authorized": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "FULL_DRIVE_ALLOCATION_SECONDS",
    "FULL_DRIVE_BASE_CHECKPOINT",
    "FULL_DRIVE_BASE_TREE",
    "FULL_DRIVE_INSTRUCTION_ID",
    "FULL_DRIVE_LAMBDA_LOCK_FILE",
    "FULL_DRIVE_MODEL_LAMBDA_LOCK_FILE",
    "FULL_DRIVE_NUMERICAL_LOCK_FILE",
    "FULL_DRIVE_REQUEST_ORDER",
    "FULL_DRIVE_RESULT_TOKEN",
    "FULL_DRIVE_SCHEMA_NAMESPACE",
    "FULL_DRIVE_SOURCE_MANIFEST_FILE",
    "FULL_DRIVE_STAGE_A_SEAL_ROOT",
    "FULL_DRIVE_TASK_GPU_CAP",
    "FullDriveResourceForecast",
    "build_full_drive_forecast",
    "expected_full_drive_result_name",
    "full_drive_panel_contract",
    "validate_full_drive_lambda_lock",
    "validate_model_lambda_lock",
]
