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
from .p1r54_fz_c3_independent import bind_independent_batch_view
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
        "s05-p1r54-realization-reset-independent-b100-fz-tech-r3-v1",
    ),
    ResetCell(
        1,
        EnergyFreeLocalZArm.PDZ,
        ResetExecutionScope.INDEPENDENT_B100,
        "r54-realization-reset-independent-b100-pdz-t1",
        "s05-p1r54-realization-reset-independent-b100-pdz-t1-tech-r3-v1",
    ),
    ResetCell(
        2,
        EnergyFreeLocalZArm.FZ,
        ResetExecutionScope.SEQUENTIAL_10XB100,
        "r54-realization-reset-sequential-10xb100-fz",
        "s05-p1r54-realization-reset-sequential-10xb100-fz-tech-r3-v1",
    ),
    ResetCell(
        3,
        EnergyFreeLocalZArm.PDZ,
        ResetExecutionScope.SEQUENTIAL_10XB100,
        "r54-realization-reset-sequential-10xb100-pdz-t1",
        "s05-p1r54-realization-reset-sequential-10xb100-pdz-t1-tech-r3-v1",
    ),
)
ROLES = tuple(item.role for item in CELLS)
RESULT_NAMES = {item.role: item.result_name for item in CELLS}


@dataclass(frozen=True, slots=True)
class ResetIndependentBatchCell:
    cell: int
    arm: EnergyFreeLocalZArm
    batch_index: int
    role: str
    result_name: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "cell": self.cell,
            "arm": self.arm.value,
            "batch_index": self.batch_index,
            "batch_label": f"B{self.batch_index}",
            "scope": ResetExecutionScope.INDEPENDENT_B100.value,
            "role": self.role,
            "result_name": self.result_name,
            "realization_policy": RealizationPolicy.POST_WRITE_W_REALIZATION_RESET.value,
            "weight_entry": "W0",
            "alpha_cache_entry": "COLD",
            "cross_batch_state_consumption_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


INDEPENDENT_EXPANSION_CELLS = tuple(
    ResetIndependentBatchCell(
        cell,
        arm,
        batch_index,
        f"r54-realization-reset-independent-b{batch_index:02d}-{arm.value.lower()}",
        (
            "s05-p1r54-realization-reset-independent-"
            f"b{batch_index:02d}-b100-{arm.value.lower()}-v1"
        ),
    )
    for cell, (arm, batch_index) in enumerate(
        (arm, batch_index)
        for arm in (EnergyFreeLocalZArm.FZ, EnergyFreeLocalZArm.PDZ)
        for batch_index in range(2, 11)
    )
)
INDEPENDENT_EXPANSION_ROLES = tuple(item.role for item in INDEPENDENT_EXPANSION_CELLS)
ALL_ROLES = (*ROLES, *INDEPENDENT_EXPANSION_ROLES)


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
    for item in INDEPENDENT_EXPANSION_CELLS:
        if item.role == role:
            return item.result_name
    return cell_for_role(role).result_name


def is_p1r54_realization_reset_role(role: str | None) -> bool:
    return role in ALL_ROLES


def independent_expansion_cell(cell: int) -> ResetIndependentBatchCell:
    if (
        isinstance(cell, bool)
        or not isinstance(cell, int)
        or not 0 <= cell < len(INDEPENDENT_EXPANSION_CELLS)
    ):
        raise ODEBFContractError("P1R54 reset independent expansion cell differs")
    return INDEPENDENT_EXPANSION_CELLS[cell]


def independent_expansion_for_role(role: str) -> ResetIndependentBatchCell:
    for item in INDEPENDENT_EXPANSION_CELLS:
        if item.role == role:
            return item
    raise ODEBFContractError("P1R54 reset independent expansion role differs")


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


def _experiment_metadata(config: ResetCell) -> Mapping[str, Any]:
    """Return one collision-safe terminal metadata namespace.

    The reused target-timescale and sequential terminal schemas already own
    generic keys such as ``cell``, ``request_count`` and
    ``scientific_promotion``.  Keeping reset deployment metadata under one
    experiment-specific key preserves those legacy terminal contracts.
    """

    request_count = (
        100
        if config.scope is ResetExecutionScope.INDEPENDENT_B100
        else 1000
    )
    batch_count = (
        1
        if config.scope is ResetExecutionScope.INDEPENDENT_B100
        else 10
    )
    payload: dict[str, Any] = {
        "cell": config.raw_free_payload(),
        "canonical_batch": (
            "B1"
            if config.scope is ResetExecutionScope.INDEPENDENT_B100
            else "B1-B10"
        ),
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "sample_duplication_count": 0,
        "sequential_batch_count": batch_count,
        "request_count": request_count,
        "realization_policy": RealizationPolicy.POST_WRITE_W_REALIZATION_RESET.value,
        "continuation_gpu_execution_count": 0,
        "source_equivalence": source_equivalence_receipt(config.cell),
        "scientific_promotion": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return {"p1r54_realization_reset": payload}


def _independent_expansion_metadata(
    config: ResetIndependentBatchCell,
    *,
    canonical_batch_order_sha256: str,
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "cell": config.raw_free_payload(),
        "canonical_batch": f"B{config.batch_index}",
        "canonical_batch_order_sha256": canonical_batch_order_sha256,
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "sample_duplication_count": 0,
        "sequential_batch_count": 1,
        "request_count": 100,
        "realization_policy": RealizationPolicy.POST_WRITE_W_REALIZATION_RESET.value,
        "continuation_gpu_execution_count": 0,
        "cross_batch_W_cache_history_factor_consumption_count": 0,
        "scientific_promotion": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return {"p1r54_realization_reset": payload}


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
        metadata=_experiment_metadata(config),
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
    if role in INDEPENDENT_EXPANSION_ROLES:
        config = independent_expansion_for_role(role)
        stream_batches = kwargs.pop("stream_batches")
        stream = kwargs.pop("stream")
        selected_view = bind_independent_batch_view(
            stream_batches, stream, batch_index=config.batch_index
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
            stream_batches=selected_view,
            stream=stream,
            target_schedule_override=schedule_for_cell(TargetTimescaleCell.Z0_COARSE),
            amplitude_policy=policy,
            realization_controller=controller,
            heldout_k_indices=HELDOUT_K_INDICES,
            experiment_instruction_id=INSTRUCTION_ID,
            experiment_method_id=(
                f"{METHOD_ID}-{config.arm.value}-INDEPENDENT-B{config.batch_index:02d}"
            ),
            terminal_schema=(
                "ode-edit-s05-p1r54-realization-reset-"
                f"{arm_tag}-independent-b100-terminal/v1"
            ),
            manifest_schema=(
                "ode-edit-s05-p1r54-realization-reset-"
                f"{arm_tag}-independent-b100-manifest/v1"
            ),
            terminal_status="P1R54_REALIZATION_RESET_INDEPENDENT_TERMINAL",
            stage_prefix=(
                f"p1r54_realization_reset_{arm_tag}_independent_b{config.batch_index:02d}"
            ),
            amplitude_writer_layer_apply_count_key="writer_layer_apply_count",
            experiment_metadata=_independent_expansion_metadata(
                config,
                canonical_batch_order_sha256=str(
                    stream["batch_ordered_request_digest_v1"][config.batch_index - 1]
                ),
            ),
            easyedit_root=easyedit_root,
            **kwargs,
        )
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
        experiment_metadata=_experiment_metadata(config),
        easyedit_root=easyedit_root,
        **kwargs,
    )


__all__ = [
    "ALL_ROLES",
    "CELLS",
    "HELDOUT_K_INDICES",
    "INSTRUCTION_ID",
    "INDEPENDENT_EXPANSION_CELLS",
    "INDEPENDENT_EXPANSION_ROLES",
    "METHOD_ID",
    "RESULT_NAMES",
    "ROLES",
    "ResetCell",
    "ResetExecutionScope",
    "ResetIndependentBatchCell",
    "build_sequential_binding",
    "cell_config",
    "cell_for_role",
    "expected_result_name",
    "independent_expansion_cell",
    "independent_expansion_for_role",
    "is_p1r54_realization_reset_role",
    "run_p1r54_realization_reset",
    "source_equivalence_receipt",
]
