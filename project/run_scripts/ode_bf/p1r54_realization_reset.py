"""Four-cell P1R54 post-write realization-reset experiment binding."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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
from .p1r52_target_timescale_b100 import run_target_timescale_b100
from .p1r54_energyfree_localz import EnergyFreeLocalZArm, EnergyFreeLocalZPolicy
from .p1r54_fz_c3_sequential import STREAM_ORDER, STREAM_ROOT
from .p1r54_realization_policy import RealizationPolicy, RealizationTransitionController


INSTRUCTION_ID = "ODEEDIT-S05-P1R54-POSTWRITE-REALIZATION-RESET-FOUR-RUNS-V1"
METHOD_ID = "P1R54-ENERGYFREE-LOCALZ-POSTWRITE-W-REALIZATION-RESET"
HELDOUT_K_INDICES = (7,)


class ResetExecutionScope(str, Enum):
    INDEPENDENT_B100 = "INDEPENDENT_B100"
    SEQUENTIAL_10XB100 = "SEQUENTIAL_10XB100"


@dataclass(frozen=True, slots=True)
class ResetCell:
    cell: int
    arm: EnergyFreeLocalZArm
    scope: ResetExecutionScope
    role: str
    result_name: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "cell": self.cell,
            "arm": self.arm.value,
            "scope": self.scope.value,
            "role": self.role,
            "result_name": self.result_name,
            "realization_policy": RealizationPolicy.POST_WRITE_W_REALIZATION_RESET.value,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


CELLS = (
    ResetCell(
        0,
        EnergyFreeLocalZArm.FZ,
        ResetExecutionScope.INDEPENDENT_B100,
        "r54-realization-reset-independent-b100-fz",
        "s05-p1r54-realization-reset-independent-b100-fz-v1",
    ),
    ResetCell(
        1,
        EnergyFreeLocalZArm.PDZ,
        ResetExecutionScope.INDEPENDENT_B100,
        "r54-realization-reset-independent-b100-pdz-t1",
        "s05-p1r54-realization-reset-independent-b100-pdz-t1-v1",
    ),
    ResetCell(
        2,
        EnergyFreeLocalZArm.FZ,
        ResetExecutionScope.SEQUENTIAL_10XB100,
        "r54-realization-reset-sequential-10xb100-fz",
        "s05-p1r54-realization-reset-sequential-10xb100-fz-v1",
    ),
    ResetCell(
        3,
        EnergyFreeLocalZArm.PDZ,
        ResetExecutionScope.SEQUENTIAL_10XB100,
        "r54-realization-reset-sequential-10xb100-pdz-t1",
        "s05-p1r54-realization-reset-sequential-10xb100-pdz-t1-v1",
    ),
)
ROLES = tuple(item.role for item in CELLS)
RESULT_NAMES = {item.role: item.result_name for item in CELLS}


def cell_config(cell: int) -> ResetCell:
    if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell < len(CELLS):
        raise ODEBFContractError("P1R54 realization-reset array cell differs")
    return CELLS[cell]


def cell_for_role(role: str) -> ResetCell:
    for item in CELLS:
        if item.role == role:
            return item
    raise ODEBFContractError("P1R54 realization-reset role differs")


def expected_result_name(role: str) -> str:
    return cell_for_role(role).result_name


def is_p1r54_realization_reset_role(role: str | None) -> bool:
    return role in ROLES


def _amplitude_factory(arm: EnergyFreeLocalZArm):
    def factory(batch_index: int, request_count: int) -> EnergyFreeLocalZPolicy:
        if (
            isinstance(batch_index, bool)
            or not 1 <= batch_index <= 10
            or request_count != 100
        ):
            raise ODEBFContractError("P1R54 reset amplitude batch geometry differs")
        return EnergyFreeLocalZPolicy(arm, request_count=request_count, outer_count=8)

    return factory


def _realization_factory(batch_index: int, request_count: int) -> RealizationTransitionController:
    if (
        isinstance(batch_index, bool)
        or not 1 <= batch_index <= 10
        or request_count != 100
    ):
        raise ODEBFContractError("P1R54 reset transition batch geometry differs")
    return RealizationTransitionController(
        RealizationPolicy.POST_WRITE_W_REALIZATION_RESET,
        request_count=request_count,
        outer_count=8,
    )


def source_equivalence_receipt(cell: int) -> Mapping[str, Any]:
    config = cell_config(cell)
    schedule = schedule_for_cell(TargetTimescaleCell.Z0_COARSE)
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-realization-reset-source-equivalence/v1",
        "instruction_id": INSTRUCTION_ID,
        "cell": config.raw_free_payload(),
        "amplitude_policy_class": "EnergyFreeLocalZPolicy",
        "amplitude_arm": config.arm.value,
        "target_schedule_identity": schedule.raw_free_payload()["identity_sha256"],
        "target_dt": float(schedule.target_dt),
        "microsteps_per_outer": int(schedule.microsteps_per_outer),
        "outer_count": 8,
        "writer": "C3_OFFICIAL_ALPHAEDIT_KSTEP",
        "writer_call_count_per_b100": 8,
        "writer_layer_apply_count_per_b100": 40,
        "single_changed_transition": "NEXT_CONTROLLER_ANCHOR_ONLY",
        "K1_proposal_command_writer_post_W_source_equivalent": True,
        "divergence_first_allowed_outer": 2,
        "new_continuation_execution_count": 0,
        "reset_added_model_forward_count": 0,
        "reset_added_backward_count": 0,
        "reset_added_materialization_count": 0,
        "reset_added_evaluator_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def build_sequential_binding(
    config: ResetCell,
    *,
    easyedit_root: Path = Path("/data/janghj/EasyEdit"),
) -> Phase3SequentialExperimentBinding:
    if config.scope is not ResetExecutionScope.SEQUENTIAL_10XB100:
        raise ODEBFContractError("P1R54 reset sequential binding scope differs")
    arm_tag = config.arm.value.lower()
    return Phase3SequentialExperimentBinding(
        role=config.role,
        writer_arm="C3-KSTEP-CACHE",
        instruction_id=INSTRUCTION_ID,
        method_id=f"{METHOD_ID}-{config.arm.value}-SEQUENTIAL",
        batch_schema=f"ode-edit-s05-p1r54-realization-reset-{arm_tag}-sequential-batch/v1",
        final_schema=f"ode-edit-s05-p1r54-realization-reset-{arm_tag}-sequential-final/v1",
        terminal_schema=f"ode-edit-s05-p1r54-realization-reset-{arm_tag}-sequential-terminal/v1",
        manifest_schema=f"ode-edit-s05-p1r54-realization-reset-{arm_tag}-sequential-manifest/v1",
        terminal_status="TERMINAL_VALID",
        return_status="P1R54_REALIZATION_RESET_SEQUENTIAL_TERMINAL",
        stage_prefix=f"p1r54_realization_reset_{arm_tag}_sequential",
        selected_target_resolver=reconstruct_target_subcycle_final_selected,
        heldout_step_indices=HELDOUT_K_INDICES,
        target_subcycle_schedule=schedule_for_cell(TargetTimescaleCell.Z0_COARSE),
        amplitude_policy_factory=_amplitude_factory(config.arm),
        easyedit_root=easyedit_root,
        metadata={
            "cell": config.raw_free_payload(),
            "stream_root": STREAM_ROOT,
            "stream_order": STREAM_ORDER,
            "sample_duplication_count": 0,
            "sequential_batch_count": 10,
            "request_count": 1000,
            "realization_policy": RealizationPolicy.POST_WRITE_W_REALIZATION_RESET.value,
            "continuation_gpu_execution_count": 0,
            "source_equivalence": source_equivalence_receipt(config.cell),
            "scientific_promotion": False,
        },
        realization_controller_factory=_realization_factory,
    )


def run_p1r54_realization_reset(
    model: Any,
    tokenizer: Any,
    *,
    role: str,
    easyedit_root: Path = Path("/data/janghj/EasyEdit"),
    **kwargs: Any,
) -> dict[str, Any]:
    config = cell_for_role(role)
    if config.scope is ResetExecutionScope.SEQUENTIAL_10XB100:
        return run_phase3(
            model,
            tokenizer,
            role=role,
            experiment_binding=build_sequential_binding(
                config, easyedit_root=easyedit_root
            ),
            **kwargs,
        )
    policy = EnergyFreeLocalZPolicy(config.arm, request_count=100, outer_count=8)
    controller = RealizationTransitionController(
        RealizationPolicy.POST_WRITE_W_REALIZATION_RESET,
        request_count=100,
        outer_count=8,
    )
    arm_tag = config.arm.value.lower()
    return run_target_timescale_b100(
        model,
        tokenizer,
        role=role,
        target_schedule_override=schedule_for_cell(TargetTimescaleCell.Z0_COARSE),
        amplitude_policy=policy,
        realization_controller=controller,
        heldout_k_indices=HELDOUT_K_INDICES,
        experiment_instruction_id=INSTRUCTION_ID,
        experiment_method_id=f"{METHOD_ID}-{config.arm.value}-INDEPENDENT",
        terminal_schema=f"ode-edit-s05-p1r54-realization-reset-{arm_tag}-independent-terminal/v1",
        manifest_schema=f"ode-edit-s05-p1r54-realization-reset-{arm_tag}-independent-manifest/v1",
        terminal_status="P1R54_REALIZATION_RESET_INDEPENDENT_TERMINAL",
        stage_prefix=f"p1r54_realization_reset_{arm_tag}_independent",
        amplitude_writer_layer_apply_count_key="writer_layer_apply_count",
        experiment_metadata={
            "cell": config.raw_free_payload(),
            "canonical_batch": "B1",
            "request_count": 100,
            "stream_root": STREAM_ROOT,
            "stream_order": STREAM_ORDER,
            "realization_policy": RealizationPolicy.POST_WRITE_W_REALIZATION_RESET.value,
            "continuation_gpu_execution_count": 0,
            "source_equivalence": source_equivalence_receipt(config.cell),
            "scientific_promotion": False,
        },
        easyedit_root=easyedit_root,
        **kwargs,
    )


__all__ = [
    "CELLS",
    "HELDOUT_K_INDICES",
    "INSTRUCTION_ID",
    "METHOD_ID",
    "RESULT_NAMES",
    "ROLES",
    "ResetCell",
    "ResetExecutionScope",
    "build_sequential_binding",
    "cell_config",
    "cell_for_role",
    "expected_result_name",
    "is_p1r54_realization_reset_role",
    "run_p1r54_realization_reset",
    "source_equivalence_receipt",
]
