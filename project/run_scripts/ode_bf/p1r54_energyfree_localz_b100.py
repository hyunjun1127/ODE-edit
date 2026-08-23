"""Thin sealed-B1 runtime binding for the two P1R54 energy-free arms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import ODEBFContractError
from .p1r52_target_timescale import TargetTimescaleCell, schedule_for_cell
from .p1r52_target_timescale_b100 import run_target_timescale_b100
from .p1r54_energyfree_localz import (
    EnergyFreeLocalZArm,
    EnergyFreeLocalZPolicy,
    INSTRUCTION_ID,
    METHOD_ID,
)


ROLE_PREFIX = "r54-energyfree-localz-b100-"
ROLES = (
    f"{ROLE_PREFIX}fz",
    f"{ROLE_PREFIX}pdz",
)
ROLE_TO_ARM = {
    ROLES[0]: EnergyFreeLocalZArm.FZ,
    ROLES[1]: EnergyFreeLocalZArm.PDZ,
}
TECHNICAL_ATTEMPT_SUFFIX = "tech-r1"
RESULT_NAMES = {
    ROLES[0]: f"s05-p1r54-energyfree-localz-llama-b100-fz-{TECHNICAL_ATTEMPT_SUFFIX}-v1",
    ROLES[1]: f"s05-p1r54-energyfree-localz-llama-b100-pdz-{TECHNICAL_ATTEMPT_SUFFIX}-v1",
}
HELDOUT_K_INDICES = (7,)


def role_for_cell(cell: int) -> str:
    if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell < len(ROLES):
        raise ODEBFContractError("P1R54 array cell differs")
    return ROLES[cell]


def arm_for_role(role: str) -> EnergyFreeLocalZArm:
    try:
        return ROLE_TO_ARM[role]
    except KeyError as exc:
        raise ODEBFContractError("P1R54 role differs") from exc


def expected_result_name(role: str) -> str:
    try:
        return RESULT_NAMES[role]
    except KeyError as exc:
        raise ODEBFContractError("P1R54 result role differs") from exc


def is_p1r54_role(role: str | None) -> bool:
    return role in ROLES


def run_p1r54_energyfree_localz_b100(
    model: Any,
    tokenizer: Any,
    *,
    role: str,
    easyedit_root: Path = Path("/data/janghj/EasyEdit"),
    **kwargs: Any,
) -> dict[str, Any]:
    arm = arm_for_role(role)
    policy = EnergyFreeLocalZPolicy(arm, request_count=100, outer_count=8)
    return run_target_timescale_b100(
        model,
        tokenizer,
        role=role,
        target_schedule_override=schedule_for_cell(TargetTimescaleCell.Z0_COARSE),
        amplitude_policy=policy,
        heldout_k_indices=HELDOUT_K_INDICES,
        experiment_instruction_id=INSTRUCTION_ID,
        experiment_method_id=METHOD_ID,
        terminal_schema="ode-edit-s05-p1r54-energyfree-localz-cell-terminal/v1",
        manifest_schema="ode-edit-s05-p1r54-energyfree-localz-cell-manifest/v1",
        terminal_status="P1R54_ENERGYFREE_LOCALZ_TERMINAL",
        stage_prefix="p1r54_energyfree_localz",
        amplitude_writer_layer_apply_count_key="p1r54_writer_layer_apply_count",
        experiment_metadata={
            "p1r54_arm": arm.value,
            "p1r54_amplitude_policy": arm.policy_name,
            "p1r54_reference_schedule": "P1R52_Z0_COARSE",
            "p1r54_target_h": 0.125,
            "p1r54_target_microsteps_per_outer": 1,
            "p1r54_total_target_time": 1.0,
            "p1r54_writer_policy": "C3_OFFICIAL_ALPHAEDIT_KSTEP",
            "p1r54_writer_call_expected_count": 8,
            "p1r54_writer_layer_apply_expected_count": 40,
            "p1r54_heldout_k1_k7_full_eval_count": 0,
            "p1r54_heldout_k8_only": True,
            "p1r54_global_p1r52_execution_count": 0,
            "p1r54_p1r53_execution_count": 0,
            "p1r54_native_execution_count": 0,
            "p1r54_qwen_execution_count": 0,
        },
        easyedit_root=easyedit_root,
        **kwargs,
    )


__all__ = [
    "HELDOUT_K_INDICES",
    "RESULT_NAMES",
    "ROLES",
    "TECHNICAL_ATTEMPT_SUFFIX",
    "arm_for_role",
    "expected_result_name",
    "is_p1r54_role",
    "role_for_cell",
    "run_p1r54_energyfree_localz_b100",
]
