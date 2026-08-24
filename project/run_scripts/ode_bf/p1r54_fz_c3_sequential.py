"""Thin P1R54 FZ binding for the existing Phase-3 C3 transaction.

The module owns no target direction, writer, cache, evaluator, or rollback
implementation.  It binds the frozen FZ amplitude policy and Z0 target clock
to the reusable B1->B10 Phase-3 sequential transaction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .contracts import ODEBFContractError, canonical_hash
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


INSTRUCTION_ID = "ODEEDIT-S05-P1R54-FZ-C3-SEQUENTIAL-10XB100-V1"
METHOD_ID = "P1R54-FZ-C3-KSTEP-ALPHA-CACHE-SEQUENTIAL-FULL-FP32"
ROLE = "r54-fz-c3-kstep-cache-sequential-full-fp32"
RESULT_NAME = "s05-p1r54-fz-c3-sequential-10xb100-v1"
STREAM_ROOT = "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
STREAM_ORDER = "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"
HELDOUT_K_INDICES = (7,)


def is_p1r54_fz_sequential_role(role: str | None) -> bool:
    return role == ROLE


def expected_result_name(role: str) -> str:
    if role != ROLE:
        raise ODEBFContractError("P1R54 FZ sequential role differs")
    return RESULT_NAME


def _amplitude_policy_factory(batch_index: int, request_count: int) -> EnergyFreeLocalZPolicy:
    if (
        isinstance(batch_index, bool)
        or not 1 <= batch_index <= 10
        or request_count != 100
    ):
        raise ODEBFContractError("P1R54 FZ sequential batch geometry differs")
    return EnergyFreeLocalZPolicy(
        EnergyFreeLocalZArm.FZ,
        request_count=request_count,
        outer_count=8,
    )


def atomic_b1_source_equivalence() -> Mapping[str, Any]:
    """Outcome-free receipt for the shared atomic/sequential B1 science path."""

    schedule = schedule_for_cell(TargetTimescaleCell.Z0_COARSE)
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-fz-sequential-atomic-b1-source-equivalence/v1",
        "instruction_id": INSTRUCTION_ID,
        "atomic_role": "r54-energyfree-localz-b100-fz",
        "sequential_role": ROLE,
        "amplitude_policy_class": "EnergyFreeLocalZPolicy",
        "amplitude_arm": "FZ",
        "formula": "F_i,k=m_i,k*||z0_i||*d_i,k",
        "target_schedule_identity": schedule.raw_free_payload()["identity_sha256"],
        "target_dt": float(schedule.target_dt),
        "microsteps_per_outer": int(schedule.microsteps_per_outer),
        "target_horizon": float(schedule.target_horizon),
        "outer_count": 8,
        "writer_arm": "C3-KSTEP-CACHE",
        "heldout_k_indices": list(HELDOUT_K_INDICES),
        "atomic_b1_entry_cache_width": 0,
        "sequential_b1_entry_cache_width": 0,
        "different_scientific_input_count": 0,
        "outcome_decision_influence_count": 0,
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
        batch_schema="ode-edit-s05-p1r54-fz-c3-sequential-batch/v1",
        final_schema="ode-edit-s05-p1r54-fz-c3-sequential-final-w10/v1",
        terminal_schema="ode-edit-s05-p1r54-fz-c3-sequential-terminal/v1",
        manifest_schema="ode-edit-s05-p1r54-fz-c3-sequential-manifest/v1",
        terminal_status="TERMINAL_VALID",
        return_status="P1R54_FZ_C3_SEQUENTIAL_TERMINAL",
        stage_prefix="p1r54_fz_c3_sequential",
        selected_target_resolver=reconstruct_target_subcycle_final_selected,
        heldout_step_indices=HELDOUT_K_INDICES,
        target_subcycle_schedule=schedule_for_cell(TargetTimescaleCell.Z0_COARSE),
        amplitude_policy_factory=_amplitude_policy_factory,
        easyedit_root=easyedit_root,
        metadata={
            "p1r54_fz_sequential": True,
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
            "writer_policy": "C3_OFFICIAL_ALPHAEDIT_KSTEP",
            "cache_policy": "ALPHA_CACHE_BATCH_ENTRY_SNAPSHOT_APPEND_AFTER_K8",
            "rho_policy": "BATCH_ENTRY_REQUEST_LOCAL_IMMUTABLE_K1_K8",
            "reference_execution_count": 0,
            "pdz_execution_count": 0,
            "qwen_execution_count": 0,
        },
    )


def run_p1r54_fz_c3_sequential(
    model: Any,
    tokenizer: Any,
    *,
    role: str,
    easyedit_root: Path = Path("/data/janghj/EasyEdit"),
    **kwargs: Any,
) -> dict[str, Any]:
    if role != ROLE:
        raise ODEBFContractError("P1R54 FZ sequential execution role differs")
    return run_phase3(
        model,
        tokenizer,
        role=role,
        experiment_binding=build_binding(easyedit_root=easyedit_root),
        **kwargs,
    )


__all__ = [
    "HELDOUT_K_INDICES",
    "INSTRUCTION_ID",
    "METHOD_ID",
    "RESULT_NAME",
    "ROLE",
    "atomic_b1_source_equivalence",
    "build_binding",
    "expected_result_name",
    "is_p1r54_fz_sequential_role",
    "run_p1r54_fz_c3_sequential",
]
