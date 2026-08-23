"""Thin sealed-B1 runtime binding for the two P1R53 speed arms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import ODEBFContractError
from .p1r52_target_timescale import TargetTimescaleCell, schedule_for_cell
from .p1r52_target_timescale_b100 import run_target_timescale_b100
from .p1r53_request_local_speed import (
    INSTRUCTION_ID,
    METHOD_ID,
    RequestLocalSpeedArm,
    RequestLocalSpeedPolicy,
)


ROLE_PREFIX = "r53-request-local-speed-b100-"
ROLES = (
    f"{ROLE_PREFIX}lp-s",
    f"{ROLE_PREFIX}lfd-e",
)
ROLE_TO_ARM = {
    ROLES[0]: RequestLocalSpeedArm.LP_S,
    ROLES[1]: RequestLocalSpeedArm.LFD_E,
}
TECHNICAL_ATTEMPT_SUFFIX = "tech-r1"
RESULT_NAMES = {
    ROLES[0]: f"s05-p1r53-request-local-speed-llama-b100-lp-s-{TECHNICAL_ATTEMPT_SUFFIX}-v1",
    ROLES[1]: f"s05-p1r53-request-local-speed-llama-b100-lfd-e-{TECHNICAL_ATTEMPT_SUFFIX}-v1",
}
HELDOUT_K_INDICES = (7,)


def role_for_cell(cell: int) -> str:
    if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell < len(ROLES):
        raise ODEBFContractError("P1R53 array cell differs")
    return ROLES[cell]


def arm_for_role(role: str) -> RequestLocalSpeedArm:
    try:
        return ROLE_TO_ARM[role]
    except KeyError as exc:
        raise ODEBFContractError("P1R53 role differs") from exc


def expected_result_name(role: str) -> str:
    try:
        return RESULT_NAMES[role]
    except KeyError as exc:
        raise ODEBFContractError("P1R53 result role differs") from exc


def is_p1r53_role(role: str | None) -> bool:
    return role in ROLES


def run_p1r53_request_local_speed_b100(
    model: Any,
    tokenizer: Any,
    *,
    role: str,
    easyedit_root: Path = Path("/data/janghj/EasyEdit"),
    **kwargs: Any,
) -> dict[str, Any]:
    arm = arm_for_role(role)
    policy = RequestLocalSpeedPolicy(arm, request_count=100, outer_count=8)
    return run_target_timescale_b100(
        model,
        tokenizer,
        role=role,
        target_schedule_override=schedule_for_cell(TargetTimescaleCell.Z0_COARSE),
        amplitude_policy=policy,
        heldout_k_indices=HELDOUT_K_INDICES,
        experiment_instruction_id=INSTRUCTION_ID,
        experiment_method_id=METHOD_ID,
        terminal_schema="ode-edit-s05-p1r53-request-local-speed-cell-terminal/v1",
        manifest_schema="ode-edit-s05-p1r53-request-local-speed-cell-manifest/v1",
        terminal_status="P1R53_REQUEST_LOCAL_SPEED_TERMINAL",
        stage_prefix="p1r53_request_local_speed",
        experiment_metadata={
            "p1r53_arm": arm.value,
            "p1r53_amplitude_policy": arm.policy_name,
            "p1r53_reference_schedule": "P1R52_Z0_COARSE",
            "p1r53_target_h": 0.125,
            "p1r53_target_microsteps_per_outer": 1,
            "p1r53_total_target_time": 1.0,
            "p1r53_writer_policy": "C3_OFFICIAL_ALPHAEDIT_KSTEP",
            "p1r53_writer_call_expected_count": 8,
            "p1r53_writer_layer_apply_expected_count": 40,
            "p1r53_heldout_k1_k7_full_eval_count": 0,
            "p1r53_heldout_k8_only": True,
            "p1r53_global_p1r52_execution_count": 0,
            "p1r53_native_execution_count": 0,
            "p1r53_qwen_execution_count": 0,
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
    "is_p1r53_role",
    "role_for_cell",
    "run_p1r53_request_local_speed_b100",
]
