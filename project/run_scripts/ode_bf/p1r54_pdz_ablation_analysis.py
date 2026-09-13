"""Raw-free schema validation for P1R54 PDZ ablation terminals."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import ODEBFContractError, canonical_hash
from .p1r54_pdz_ablation import PDZAblationArm


def group_microsteps_by_outer(
    arm: PDZAblationArm | str,
    microsteps: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    selected = arm if isinstance(arm, PDZAblationArm) else PDZAblationArm(arm)
    width = {
        PDZAblationArm.PDZ_T2_PRC: 2,
        PDZAblationArm.PDZ_T3_PRC: 3,
        PDZAblationArm.PDZ_T5_PRC: 5,
        PDZAblationArm.PDZ_T1_DIRECT: 1,
    }[selected]
    if len(microsteps) != 8 * width:
        raise ODEBFContractError("P1R54 analysis microstep denominator differs")
    groups: list[tuple[Mapping[str, Any], ...]] = []
    for outer in range(8):
        group = tuple(microsteps[outer * width : (outer + 1) * width])
        for micro, item in enumerate(group):
            if (
                item.get("outer_step_index") != outer
                or item.get("microstep_index") != micro
                or item.get("global_field_evaluation_ordinal")
                != outer * width + micro
            ):
                raise ODEBFContractError("P1R54 analysis clock differs")
        if any(
            group[micro].get("entry_target_sha256")
            != group[micro - 1].get("selected_target_sha256")
            for micro in range(1, width)
        ):
            raise ODEBFContractError("P1R54 analysis microstep state chain differs")
        groups.append(group)
    return tuple(groups)


def validate_terminal_schema(terminal: Mapping[str, Any]) -> Mapping[str, Any]:
    identity = terminal.get("identity_sha256")
    payload = dict(terminal)
    payload.pop("identity_sha256", None)
    if identity != canonical_hash(payload):
        raise ODEBFContractError("P1R54 analysis terminal identity differs")
    status = terminal.get("status")
    if status == "SCIENTIFIC_BOUNDARY_NTSM":
        if (
            terminal.get("arm") != PDZAblationArm.PDZ_T1_DIRECT.value
            or terminal.get("writer_update_count_at_boundary") != 0
            or terminal.get("W0_restored") is not True
        ):
            raise ODEBFContractError("P1R54 analysis NTSM terminal differs")
        return terminal
    if status != "TERMINAL_VALID":
        raise ODEBFContractError("P1R54 analysis terminal status differs")
    arm = PDZAblationArm(str(terminal["p1r54_ablation_arm"]))
    microsteps = terminal.get("microstep_trajectory")
    executions = terminal.get("kstep_executions")
    if (
        not isinstance(microsteps, list)
        or not isinstance(executions, list)
        or len(executions) != 8
        or terminal.get("writer_call_count") != 8
        or terminal.get("p1r54_writer_layer_apply_count") != 40
        or terminal.get("W0_restored") is not True
    ):
        raise ODEBFContractError("P1R54 analysis writer denominator differs")
    group_microsteps_by_outer(arm, microsteps)
    return terminal


__all__ = ["group_microsteps_by_outer", "validate_terminal_schema"]
