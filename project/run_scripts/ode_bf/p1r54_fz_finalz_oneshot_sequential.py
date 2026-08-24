"""P1R54 FZ target flow with one final C3 write per B100 block."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .p1r52_c_writer_kstep_cache_sequential import (
    Phase3SequentialExperimentBinding,
    run_phase3,
)
from .p1r52_target_timescale import (
    TargetTimescaleCell,
    reconstruct_target_subcycle_final_selected,
    schedule_for_cell,
)
from .p1r54_energyfree_localz import EnergyFreeLocalZArm, EnergyFreeLocalZPolicy
from .p1r54_fz_c3_sequential import STREAM_ORDER, STREAM_ROOT
from .writer_cadence import WriterCadence


INSTRUCTION_ID = "ODEEDIT-S05-P1R54-FZ-FINAL-Z-ONESHOT-SEQUENTIAL-10XB100-V1"
METHOD_ID = "P1R54-FZ-FINAL-Z-ONESHOT-C3-SEQUENTIAL-FULL-FP32"
ROLE = "r54-fz-final-z-oneshot-cache-sequential-full-fp32"
RESULT_NAME = "s05-p1r54-fz-final-z-oneshot-sequential-10xb100-tech-r1-v1"
HELDOUT_K_INDICES = (7,)
CONTROL_TERMINAL_SHA256 = (
    "7507d44f453fe701e67e76c758f578ef658ab00081dc189afc84e99d695fdb88"
)
CONTROL_MANIFEST_SHA256 = (
    "b4efbcb2e7cade61935eb6a745fc935d5a8fe9e3b0343129d28628d61c94d295"
)
TECH_R0_FAILURE_SHA256 = (
    "1e75dbf99f49ed15cd630b60fb9756553568efa7449df69d0ed28d21ecc0e40f"
)


def is_p1r54_fz_finalz_oneshot_role(role: str | None) -> bool:
    return role == ROLE


def expected_result_name(role: str) -> str:
    if role != ROLE:
        raise ODEBFContractError("P1R54 FZ final-z one-shot role differs")
    return RESULT_NAME


def _amplitude_policy_factory(batch_index: int, request_count: int) -> EnergyFreeLocalZPolicy:
    if (
        isinstance(batch_index, bool)
        or not 1 <= batch_index <= 10
        or request_count != 100
    ):
        raise ODEBFContractError("P1R54 FZ final-z batch geometry differs")
    return EnergyFreeLocalZPolicy(
        EnergyFreeLocalZArm.FZ,
        request_count=request_count,
        outer_count=8,
    )


def control_source_equivalence() -> Mapping[str, Any]:
    schedule = schedule_for_cell(TargetTimescaleCell.Z0_COARSE)
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-fz-final-z-control-source-equivalence/v1",
        "instruction_id": INSTRUCTION_ID,
        "control_method": "P1R54-FZ-C3-KSTEP-ALPHA-CACHE-SEQUENTIAL-FULL-FP32",
        "control_terminal_sha256": CONTROL_TERMINAL_SHA256,
        "control_manifest_sha256": CONTROL_MANIFEST_SHA256,
        "new_method": METHOD_ID,
        "same_target_policy": "F_i,k=m_i,k*||z0_i||*d_i,k",
        "same_target_schedule_identity": schedule.raw_free_payload()["identity_sha256"],
        "same_target_dt": float(schedule.target_dt),
        "same_outer_count": 8,
        "single_variable": "writer_cadence",
        "control_writer_cadence": WriterCadence.KSTEP_EACH_OUTER.value,
        "new_writer_cadence": WriterCadence.FINAL_Z_ONESHOT.value,
        "control_writer_calls_per_block": 8,
        "new_writer_calls_per_block": 1,
        "control_layer_applies_per_block": 40,
        "new_layer_applies_per_block": 5,
        "target_proposal_difference_count_at_k1": 0,
        "reference_execution_count": 0,
        "technical_attempt": "TECH-R1",
        "excluded_technical_failure_sha256": TECH_R0_FAILURE_SHA256,
        "scientific_change_count": 0,
        "outcome_decision_influence_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def validate_finalz_batch_rows(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(rows) != 10:
        raise ODEBFStateError("final-z block count differs")
    for index, row in enumerate(rows, start=1):
        if (
            row.get("batch_index") != index
            or row.get("writer_call_count") != 1
            or row.get("layer_apply_count") != 5
            or row.get("field_evaluation_count") != 8
            or row.get("rho_capture_count") != 1
            or row.get("rho_refresh_count") != 0
            or row.get("no_write_prefix_count") != 7
            or row.get("cache_entry_width") != (index - 1) * 100
            or row.get("cache_exit_width") != index * 100
        ):
            raise ODEBFStateError("final-z block clock/cache counts differ")
        if index > 1 and rows[index - 2].get("commit_weight_sha256") != row.get(
            "entry_weight_sha256"
        ):
            raise ODEBFStateError("final-z sequential W chain differs")
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-fz-final-z-block-chain/v1",
        "batch_count": 10,
        "field_evaluation_count": 80,
        "writer_call_count": 10,
        "layer_apply_count": 50,
        "no_write_prefix_count": 70,
        "commit_to_next_entry_match_count": 9,
        "cache_entry_widths": list(range(0, 1000, 100)),
        "cache_exit_widths": list(range(100, 1100, 100)),
        "sample_duplication_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def build_binding(
    *, easyedit_root: Path = Path("/data/janghj/EasyEdit")
) -> Phase3SequentialExperimentBinding:
    return Phase3SequentialExperimentBinding(
        role=ROLE,
        writer_arm="C3-KSTEP-CACHE",
        instruction_id=INSTRUCTION_ID,
        method_id=METHOD_ID,
        batch_schema="ode-edit-s05-p1r54-fz-final-z-oneshot-batch/v1",
        final_schema="ode-edit-s05-p1r54-fz-final-z-oneshot-final-w10/v1",
        terminal_schema="ode-edit-s05-p1r54-fz-final-z-oneshot-terminal/v1",
        manifest_schema="ode-edit-s05-p1r54-fz-final-z-oneshot-manifest/v1",
        terminal_status="TERMINAL_VALID",
        return_status="P1R54_FZ_FINAL_Z_ONESHOT_SEQUENTIAL_TERMINAL",
        stage_prefix="p1r54_fz_finalz_oneshot_sequential",
        selected_target_resolver=reconstruct_target_subcycle_final_selected,
        heldout_step_indices=HELDOUT_K_INDICES,
        target_subcycle_schedule=schedule_for_cell(TargetTimescaleCell.Z0_COARSE),
        amplitude_policy_factory=_amplitude_policy_factory,
        easyedit_root=easyedit_root,
        writer_cadence=WriterCadence.FINAL_Z_ONESHOT,
        metadata={
            "p1r54_fz_finalz_oneshot_sequential": True,
            "model_alias": "llama3-8b-inst",
            "formula": "F_i,k=m_i,k*||z0_i||*d_i,k",
            "stream_root": STREAM_ROOT,
            "stream_order": STREAM_ORDER,
            "sample_duplication_count": 0,
            "batch_size": 100,
            "sequential_batch_count": 10,
            "target_dt": 0.125,
            "target_microsteps_per_outer": 1,
            "target_horizon_per_batch": 1.0,
            "outer_count_per_batch": 8,
            "writer_policy": "C3_OFFICIAL_ALPHAEDIT_FINAL_Z_ONESHOT",
            "writer_cadence": WriterCadence.FINAL_Z_ONESHOT.value,
            "writer_calls_per_batch": 1,
            "writer_calls_total": 10,
            "writer_layer_applies_per_batch": 5,
            "writer_layer_applies_total": 50,
            "intermediate_target_writer_influence_count": 0,
            "finite_demand_builder_count_per_batch": 1,
            "materialization_count_per_batch": 1,
            "cache_policy": "ALPHA_CACHE_APPEND_ONCE_AFTER_FINAL_K8_WRITE",
            "rho_policy": "BATCH_ENTRY_REQUEST_LOCAL_IMMUTABLE_K1_K8",
            "reference_execution_count": 0,
            "pdz_execution_count": 0,
            "qwen_execution_count": 0,
            "technical_attempt": "TECH-R1",
            "excluded_technical_failure_sha256": TECH_R0_FAILURE_SHA256,
            "scientific_change_count": 0,
        },
    )


def run_p1r54_fz_finalz_oneshot_sequential(
    model: Any,
    tokenizer: Any,
    *,
    role: str,
    easyedit_root: Path = Path("/data/janghj/EasyEdit"),
    **kwargs: Any,
) -> dict[str, Any]:
    if role != ROLE:
        raise ODEBFContractError("P1R54 FZ final-z execution role differs")
    return run_phase3(
        model,
        tokenizer,
        role=role,
        experiment_binding=build_binding(easyedit_root=easyedit_root),
        **kwargs,
    )


__all__ = [
    "CONTROL_MANIFEST_SHA256",
    "CONTROL_TERMINAL_SHA256",
    "HELDOUT_K_INDICES",
    "INSTRUCTION_ID",
    "METHOD_ID",
    "RESULT_NAME",
    "ROLE",
    "TECH_R0_FAILURE_SHA256",
    "build_binding",
    "control_source_equivalence",
    "expected_result_name",
    "is_p1r54_fz_finalz_oneshot_role",
    "run_p1r54_fz_finalz_oneshot_sequential",
    "validate_finalz_batch_rows",
]
