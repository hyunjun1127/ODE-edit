"""Sealed Llama B1/B100 runtime binding for the P1R54 PDZ ablation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p1_runtime import _atomic_write_once
from .p1r36_independent_b10x10_runtime import _hashes
from .p1r52_target_timescale import reconstruct_target_subcycle_final_selected
from .p1r52_target_timescale_b100 import run_target_timescale_b100
from .p1r54_pdz_ablation import (
    ClockedPDZAmplitudePolicy,
    INSTRUCTION_ID,
    METHOD_ID,
    PDZAblationArm,
    P1R54ScientificBoundaryNTSM,
    run_pdz_ablation_subcycle_scheduler,
    schedule_for_arm,
)


ROLE_PREFIX = "r54-pdz-ablation-b100-"
ROLES = (
    f"{ROLE_PREFIX}t2-prc",
    f"{ROLE_PREFIX}t3-prc",
    f"{ROLE_PREFIX}t5-prc",
    f"{ROLE_PREFIX}t1-direct",
)
ROLE_TO_ARM = {
    ROLES[0]: PDZAblationArm.PDZ_T2_PRC,
    ROLES[1]: PDZAblationArm.PDZ_T3_PRC,
    ROLES[2]: PDZAblationArm.PDZ_T5_PRC,
    ROLES[3]: PDZAblationArm.PDZ_T1_DIRECT,
}
TECHNICAL_ATTEMPT_SUFFIX = "tech-r1"
RESULT_NAMES = {
    ROLES[0]: "s05-p1r54-pdz-ablation-llama-b100-t2-prc-tech-r1-v1",
    ROLES[1]: "s05-p1r54-pdz-ablation-llama-b100-t3-prc-tech-r1-v1",
    ROLES[2]: "s05-p1r54-pdz-ablation-llama-b100-t5-prc-tech-r1-v1",
    ROLES[3]: "s05-p1r54-pdz-ablation-llama-b100-t1-direct-tech-r1-v1",
}
HELDOUT_K_INDICES = (7,)


def role_for_cell(cell: int) -> str:
    if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell < 4:
        raise ODEBFContractError("P1R54 PDZ ablation array cell differs")
    return ROLES[cell]


def arm_for_role(role: str) -> PDZAblationArm:
    try:
        return ROLE_TO_ARM[role]
    except KeyError as exc:
        raise ODEBFContractError("P1R54 PDZ ablation role differs") from exc


def expected_result_name(role: str) -> str:
    try:
        return RESULT_NAMES[role]
    except KeyError as exc:
        raise ODEBFContractError("P1R54 PDZ ablation result role differs") from exc


def is_p1r54_pdz_ablation_role(role: str | None) -> bool:
    return role in ROLES


def _scientific_boundary_result(
    *,
    destination: Path,
    arm: PDZAblationArm,
    source_head: str,
    boundary: Mapping[str, Any],
    touched: Mapping[str, Any],
    base_receipt: Any,
    entry_pointers: Mapping[str, int],
) -> dict[str, Any]:
    hashes = _hashes(touched)
    if hashes != dict(base_receipt.parameter_sha256) or any(
        int(touched[name].data_ptr()) != entry_pointers[name] for name in touched
    ):
        raise ODEBFStateError("P1R54 NTSM W0 rollback differs")
    boundary_sha = _atomic_write_once(
        destination / "scientific-boundary-ntsm.json", dict(boundary)
    )
    terminal: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-pdz-ablation-cell-terminal/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "status": "SCIENTIFIC_BOUNDARY_NTSM",
        "source_head": source_head,
        "arm": arm.value,
        "request_count": 100,
        "boundary_sha256": boundary_sha,
        "boundary_identity": boundary["identity_sha256"],
        "boundary_outer_index": boundary["clock"]["outer_index"],
        "boundary_global_ordinal": boundary["clock"]["global_ordinal"],
        "writer_update_count_at_boundary": 0,
        "terminal_W0_restore": True,
        "W0_restored": True,
        "fallback_retry_zero_write_imputation_count": 0,
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-pdz-ablation-cell-manifest/v1",
        "instruction_id": INSTRUCTION_ID,
        "source_head": source_head,
        "arm": arm.value,
        "status": "SCIENTIFIC_BOUNDARY_NTSM",
        "terminal_sha256": terminal_sha,
        "boundary_sha256": boundary_sha,
        "W0_restored": True,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    return {
        "status": "P1R54_PDZ_ABLATION_SCIENTIFIC_BOUNDARY_NTSM",
        "arm": arm.value,
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "W0_restored": True,
    }


def run_p1r54_pdz_ablation_b100(
    model: Any,
    tokenizer: Any,
    *,
    role: str,
    destination: Path,
    source_head: str,
    touched: Mapping[str, Any],
    base_receipt: Any,
    easyedit_root: Path = Path("/mnt/raid5/janghj/EasyEdit"),
    **kwargs: Any,
) -> dict[str, Any]:
    arm = arm_for_role(role)
    schedule = schedule_for_arm(arm)
    policy = ClockedPDZAmplitudePolicy(arm, request_count=100)
    entry_pointers = {name: int(value.data_ptr()) for name, value in touched.items()}
    metadata = {
        "p1r54_ablation_arm": arm.value,
        "p1r54_target_h": 0.125,
        "p1r54_target_microsteps_per_outer": schedule.microsteps_per_outer,
        "p1r54_total_target_time": float(schedule.target_horizon),
        "p1r54_selection_policy": (
            "P1R43_PRIMARY_RESCUE_CURRENT"
            if arm is not PDZAblationArm.PDZ_T1_DIRECT
            else "DIRECT_EULER"
        ),
        "p1r54_expected_field_gradient_count": schedule.total_field_evaluations,
        "p1r54_expected_primary_evaluation_count": schedule.total_field_evaluations,
        "p1r54_expected_selector_count": (
            schedule.total_field_evaluations
            if arm is not PDZAblationArm.PDZ_T1_DIRECT
            else 0
        ),
        "p1r54_expected_finite_demand_builder_count": 8,
        "p1r54_expected_writer_call_count": 8,
        "p1r54_expected_writer_layer_apply_count": 40,
        "p1r54_direct_rescue_selector_current_count": 0,
        "p1r54_inner_writer_materialization_history_cache_append_count": 0,
        "p1r54_heldout_k8_only": True,
        "p1r54_scientific_promotion": False,
    }
    try:
        return run_target_timescale_b100(
            model,
            tokenizer,
            role=role,
            destination=destination,
            source_head=source_head,
            target_schedule_override=schedule,
            amplitude_policy=policy,
            target_subcycle_runner=run_pdz_ablation_subcycle_scheduler,
            heldout_k_indices=HELDOUT_K_INDICES,
            experiment_instruction_id=INSTRUCTION_ID,
            experiment_method_id=METHOD_ID,
            terminal_schema="ode-edit-s05-p1r54-pdz-ablation-cell-terminal/v1",
            manifest_schema="ode-edit-s05-p1r54-pdz-ablation-cell-manifest/v1",
            terminal_status="P1R54_PDZ_ABLATION_TERMINAL",
            stage_prefix="p1r54_pdz_ablation",
            amplitude_writer_layer_apply_count_key="p1r54_writer_layer_apply_count",
            experiment_metadata=metadata,
            touched=touched,
            base_receipt=base_receipt,
            easyedit_root=easyedit_root,
            **kwargs,
        )
    except P1R54ScientificBoundaryNTSM as exc:
        return _scientific_boundary_result(
            destination=destination,
            arm=arm,
            source_head=source_head,
            boundary=exc.receipt,
            touched=touched,
            base_receipt=base_receipt,
            entry_pointers=entry_pointers,
        )


__all__ = [
    "HELDOUT_K_INDICES",
    "RESULT_NAMES",
    "ROLES",
    "TECHNICAL_ATTEMPT_SUFFIX",
    "arm_for_role",
    "expected_result_name",
    "is_p1r54_pdz_ablation_role",
    "role_for_cell",
    "run_p1r54_pdz_ablation_b100",
]
