"""Typed same-horizon/longer-time target subcycles for P1R52.

Only the target clock is varied here.  The existing P1R52 field,
PRIMARY/RESCUE/CURRENT selector, origin clamp, and controller state remain the
scientific authorities.  Physical weights, the KL teacher, history/cache, and
writer factors are fixed for every target microstep inside one outer K.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, Callable, Mapping

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p1r24_atomic_strength import P1R24AliasTargetLock, P1R24KLResult
from .p1r39_normalized_gradient_target import _full_current_residual_step
from .p1r51_requestwise_semantic_allocation import P1R51ControllerState
from .p1r52_r42_safe_kdc import (
    P1R52AmplitudePolicy,
    P1R52SelectedTarget,
    prepare_p1r52_rescue_proposal,
    prepare_p1r52_target_proposal,
    select_p1r52_target_proposal,
)
from .scalable_batched_model import ScalableObjectiveResult


INSTRUCTION_ID = "ODEEDIT-S05-P1R52-TARGET-TIMESCALE-B100-V1"
METHOD_ID = "P1R52-TARGET-TIMESCALE-B100-C3-KSTEP"


class TargetTimescaleCell(str, Enum):
    Z0_COARSE = "Z0-COARSE"
    Z1_REFINE = "Z1-REFINE"
    Z15 = "Z15"
    Z20 = "Z20"
    Z30 = "Z30"


@dataclass(frozen=True, slots=True)
class TargetSubcycleSchedule:
    cell: TargetTimescaleCell
    target_horizon: Fraction
    microsteps_per_outer: int
    target_dt: Fraction
    outer_count: int = 8

    def __post_init__(self) -> None:
        if (
            isinstance(self.microsteps_per_outer, bool)
            or self.microsteps_per_outer <= 0
            or self.outer_count != 8
            or self.target_horizon <= 0
            or self.target_horizon > 3
            or self.target_dt <= 0
            or self.microsteps_per_outer * self.target_dt
            != self.target_horizon / self.outer_count
        ):
            raise ODEBFContractError("P1R52 target-subcycle schedule algebra differs")

    @property
    def total_field_evaluations(self) -> int:
        return self.outer_count * self.microsteps_per_outer

    def raw_free_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": "ode-edit-s05-p1r52-target-subcycle-schedule/v1",
            "instruction_id": INSTRUCTION_ID,
            "cell": self.cell.value,
            "target_horizon": float(self.target_horizon),
            "microsteps_per_outer": self.microsteps_per_outer,
            "target_dt": float(self.target_dt),
            "outer_count": self.outer_count,
            "total_field_evaluations": self.total_field_evaluations,
            "outer_target_time": float(
                self.microsteps_per_outer * self.target_dt
            ),
            "schedule_identity": "8*m*target_dt=Tz; m*target_dt=Tz/8",
            "shared_speed_override_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


SCHEDULES: tuple[TargetSubcycleSchedule, ...] = (
    TargetSubcycleSchedule(TargetTimescaleCell.Z0_COARSE, Fraction(1), 1, Fraction(1, 8)),
    TargetSubcycleSchedule(TargetTimescaleCell.Z1_REFINE, Fraction(1), 2, Fraction(1, 16)),
    TargetSubcycleSchedule(TargetTimescaleCell.Z15, Fraction(3, 2), 3, Fraction(1, 16)),
    TargetSubcycleSchedule(TargetTimescaleCell.Z20, Fraction(2), 4, Fraction(1, 16)),
    TargetSubcycleSchedule(TargetTimescaleCell.Z30, Fraction(3), 6, Fraction(1, 16)),
)


def schedule_for_cell(cell: TargetTimescaleCell | str) -> TargetSubcycleSchedule:
    policy = cell if isinstance(cell, TargetTimescaleCell) else TargetTimescaleCell(cell)
    return next(item for item in SCHEDULES if item.cell is policy)


@dataclass(frozen=True, slots=True)
class TargetSubcycleMicrostep:
    microstep_index: int
    global_field_evaluation_ordinal: int
    entry_objective: ScalableObjectiveResult
    kl_result: P1R24KLResult
    selected: P1R52SelectedTarget
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TargetSubcycleOuter:
    target_step: Any
    selected_endpoint: ScalableObjectiveResult
    next_state: P1R51ControllerState
    microsteps: tuple[TargetSubcycleMicrostep, ...]
    target_results: tuple[ScalableObjectiveResult, ...]
    kl_results: tuple[P1R24KLResult, ...]
    final_selected: P1R52SelectedTarget
    outer_entry_target: torch.Tensor
    current_terminal: torch.Tensor
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TargetSubcycleFinalBridgeReceipt:
    status: str
    step_index: int
    cell: str
    configured_microstep_count: int
    executed_microstep_count: int
    final_selected_target_sha256: str
    writer_target_sha256: str
    outer_entry_target_sha256: str
    current_terminal_sha256: str
    final_full_current_residual_sha256: str
    outer_reassembly_receipt_identity: str
    final_endpoint_objective_identity: str
    final_selected_endpoint_objective_match: bool
    stale_target_rejection_enabled: bool
    intermediate_writer_authority_count: int
    selected_target_recompute_count: int

    def raw_free_payload(self) -> dict[str, object]:
        payload = {
            "status": self.status,
            "step_index": self.step_index,
            "cell": self.cell,
            "configured_microstep_count": self.configured_microstep_count,
            "executed_microstep_count": self.executed_microstep_count,
            "final_selected_target_sha256": self.final_selected_target_sha256,
            "writer_target_sha256": self.writer_target_sha256,
            "outer_entry_target_sha256": self.outer_entry_target_sha256,
            "current_terminal_sha256": self.current_terminal_sha256,
            "final_full_current_residual_sha256": self.final_full_current_residual_sha256,
            "outer_reassembly_receipt_identity": self.outer_reassembly_receipt_identity,
            "final_endpoint_objective_identity": self.final_endpoint_objective_identity,
            "final_selected_endpoint_objective_match": self.final_selected_endpoint_objective_match,
            "stale_target_rejection_enabled": self.stale_target_rejection_enabled,
            "intermediate_writer_authority_count": self.intermediate_writer_authority_count,
            "selected_target_recompute_count": self.selected_target_recompute_count,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return str(self.raw_free_payload()["identity_sha256"])


def run_target_subcycle_scheduler(
    *,
    outer_step_index: int,
    schedule: TargetSubcycleSchedule,
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    target_origin: torch.Tensor,
    state: P1R51ControllerState,
    lock: P1R24AliasTargetLock,
    alias: str,
    shared_speed: float,
    teacher_sha256: str,
    evaluate_target: Callable[[torch.Tensor], ScalableObjectiveResult],
    evaluate_kl: Callable[[torch.Tensor], P1R24KLResult],
    evaluate_endpoint: Callable[[torch.Tensor, str], ScalableObjectiveResult],
    fixed_state_identities: Callable[[], tuple[str, str, str]],
    first_target_result: ScalableObjectiveResult | None = None,
    first_kl_result: P1R24KLResult | None = None,
    amplitude_policy: P1R52AmplitudePolicy | None = None,
) -> TargetSubcycleOuter:
    """Execute every configured target microstep at one fixed physical W_k."""

    if not isinstance(schedule, TargetSubcycleSchedule):
        raise ODEBFContractError("P1R52 target-subcycle schedule type differs")
    if not 0 <= outer_step_index < schedule.outer_count:
        raise ODEBFContractError("P1R52 target-subcycle outer index differs")
    if (first_target_result is None) != (first_kl_result is None):
        raise ODEBFContractError("P1R52 first subcycle result pair differs")
    physical_before, history_before, factors_before = fixed_state_identities()
    origin_before = tensor_sha256(target_origin)
    scratch_target = current_target
    scratch_state = state
    target_results: list[ScalableObjectiveResult] = []
    kl_results: list[P1R24KLResult] = []
    microsteps: list[TargetSubcycleMicrostep] = []
    entry_nll_gradient: torch.Tensor | None = None
    entry_aggregate_gradient: torch.Tensor | None = None
    target_dt = float(schedule.target_dt)

    for microstep_index in range(schedule.microsteps_per_outer):
        if microstep_index == 0 and first_target_result is not None:
            target_result = first_target_result
            assert first_kl_result is not None
            kl_result = first_kl_result
        else:
            target_result = evaluate_target(scratch_target)
            kl_result = evaluate_kl(scratch_target)
        target_results.append(target_result)
        kl_results.append(kl_result)
        proposal = prepare_p1r52_target_proposal(
            scratch_target,
            current_terminal,
            target_origin,
            target_result,
            kl_result,
            lock,
            scratch_state,
            alias=alias,
            step_index=outer_step_index,
            shared_speed=shared_speed,
            kl_teacher_input_sha256=teacher_sha256,
            target_dt=target_dt,
            amplitude_policy=amplitude_policy,
        )
        if microstep_index == 0:
            entry_nll_gradient = proposal.primary_step.nll_gradient
            entry_aggregate_gradient = proposal.primary_step.combined_gradient
        primary_endpoint = evaluate_endpoint(
            proposal.primary_step.target_next, "PRIMARY"
        )
        rescue = prepare_p1r52_rescue_proposal(
            proposal,
            scratch_target,
            current_terminal,
            target_result,
            primary_endpoint,
            step_index=outer_step_index,
        )
        rescue_endpoint = (
            None
            if rescue.rescue_step is None
            else evaluate_endpoint(rescue.rescue_step.target_next, "RESCUE")
        )
        selected = select_p1r52_target_proposal(
            proposal,
            rescue,
            scratch_target,
            current_terminal,
            target_result,
            primary_endpoint,
            rescue_endpoint,
            step_index=outer_step_index,
            target_dt=target_dt,
        )
        selected_observation = None
        if amplitude_policy is not None:
            observer = getattr(amplitude_policy, "observe_selected", None)
            if observer is None:
                raise ODEBFContractError(
                    "P1R52 external amplitude selected observer is absent"
                )
            selected_observation = observer(
                step_index=outer_step_index,
                current_nll=tuple(float(item) for item in target_result.per_request_values),
                selected_nll=tuple(
                    float(item) for item in selected.selected_endpoint.per_request_values
                ),
                field_receipt=proposal.receipt,
                selection=tuple(str(item) for item in selected.receipt["selection_by_request"]),
                target_dt=target_dt,
            )
        physical_now, history_now, factors_now = fixed_state_identities()
        if (
            physical_now != physical_before
            or history_now != history_before
            or factors_now != factors_before
            or tensor_sha256(target_origin) != origin_before
        ):
            raise ODEBFStateError("P1R52 fixed state changed inside target subcycle")
        micro_receipt: dict[str, Any] = {
            "schema": "ode-edit-s05-p1r52-target-subcycle-microstep/v1",
            "instruction_id": INSTRUCTION_ID,
            "cell": schedule.cell.value,
            "outer_step_index": outer_step_index,
            "microstep_index": microstep_index,
            "global_field_evaluation_ordinal": (
                outer_step_index * schedule.microsteps_per_outer
                + microstep_index
            ),
            "target_time_before": float(
                Fraction(outer_step_index * schedule.microsteps_per_outer + microstep_index)
                * schedule.target_dt
            ),
            "target_time_after": float(
                Fraction(outer_step_index * schedule.microsteps_per_outer + microstep_index + 1)
                * schedule.target_dt
            ),
            "target_dt": target_dt,
            "entry_target_sha256": tensor_sha256(scratch_target),
            "last_micro_entry_objective_identity": target_result.identity_sha256,
            "final_selected_endpoint_objective_identity": selected.selected_endpoint.identity_sha256,
            "last_micro_entry_objective": target_result.raw_free_payload(),
            "final_selected_endpoint_objective": selected.selected_endpoint.raw_free_payload(),
            "selected_target_sha256": tensor_sha256(selected.target_step.target_next),
            "selection_by_request": list(selected.receipt["selection_by_request"]),
            "primary_receipt_identity": proposal.receipt["identity_sha256"],
            "selected_receipt_identity": selected.receipt["identity_sha256"],
            "field_receipt": dict(proposal.receipt),
            "selection_receipt": dict(selected.receipt),
            "target_energy_delta_over_target_dt": True,
            "writer_authority_count": 0,
            "materialization_count": 0,
            "history_append_count": 0,
            "early_break_count": 0,
        }
        if selected_observation is not None:
            micro_receipt["external_amplitude_selected_observation"] = dict(
                selected_observation
            )
        micro_receipt["identity_sha256"] = canonical_hash(micro_receipt)
        microsteps.append(
            TargetSubcycleMicrostep(
                microstep_index,
                outer_step_index * schedule.microsteps_per_outer + microstep_index,
                target_result,
                kl_result,
                selected,
                micro_receipt,
            )
        )
        scratch_target = selected.target_step.target_next
        scratch_state = selected.next_state

    assert entry_nll_gradient is not None and entry_aggregate_gradient is not None
    final_selected = microsteps[-1].selected
    target_star = final_selected.target_step.target_next
    outer_no_move = tuple(
        bool(torch.equal(
            current_target.detach().cpu().contiguous()[:, request_index],
            target_star.detach().cpu().contiguous()[:, request_index],
        ))
        for request_index in range(current_target.shape[1])
    )
    full_residual = (
        target_star.detach().to(device="cpu", dtype=torch.float64)
        - current_terminal.detach().to(device="cpu", dtype=torch.float64)
    )
    physical_after, history_after, factors_after = fixed_state_identities()
    if (physical_after, history_after, factors_after) != (
        physical_before,
        history_before,
        factors_before,
    ):
        raise ODEBFStateError("P1R52 target-subcycle terminal freeze differs")
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r52-target-subcycle-outer/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "schedule": schedule.raw_free_payload(),
        "outer_step_index": outer_step_index,
        "configured_microstep_count": schedule.microsteps_per_outer,
        "executed_microstep_count": len(microsteps),
        "target_dt": target_dt,
        "outer_target_time": float(schedule.microsteps_per_outer * schedule.target_dt),
        "target_time_before": float(
            outer_step_index * schedule.microsteps_per_outer * schedule.target_dt
        ),
        "target_time_after": float(
            (outer_step_index + 1) * schedule.microsteps_per_outer * schedule.target_dt
        ),
        "last_micro_entry_objective_identity": target_results[-1].identity_sha256,
        "final_selected_endpoint_objective_identity": final_selected.selected_endpoint.identity_sha256,
        "last_micro_entry_objective_is_primary_count": 0,
        "final_selected_endpoint_objective_is_primary_count": 1,
        "microstep_receipts": [dict(item.receipt) for item in microsteps],
        "selection_by_request": list(final_selected.receipt["selection_by_request"]),
        "semantic_held_mask": list(outer_no_move),
        "target_hold_mask": list(outer_no_move),
        "outer_entry_target_sha256": tensor_sha256(current_target),
        "final_selected_target_sha256": tensor_sha256(target_star),
        "current_terminal_sha256": tensor_sha256(current_terminal),
        "final_full_current_residual_sha256": tensor_sha256(full_residual),
        "final_full_current_residual_norm": float(torch.linalg.vector_norm(full_residual)),
        "final_residual_rebuilt_from_final_z_minus_y_Wk": True,
        "all_configured_microsteps_executed": True,
        "early_break_count": 0,
        "zero_motion_time_advance_enabled": True,
        "intermediate_writer_authority_count": 0,
        "inner_writer_materialization_count": 0,
        "inner_weight_mutation_count": 0,
        "inner_history_append_count": 0,
        "inner_factor_build_count": 0,
        "inner_heldout_access_count": 0,
        "outer_writer_materialization_expected_count": 1,
        "cache_entry_reuse_expected_count": 1,
        "teacher_sha256": teacher_sha256,
        "kl_teacher_input_sha256": teacher_sha256,
        "physical_state_sha256": physical_before,
        "history_cache_sha256": history_before,
        "factor_inventory_sha256": factors_before,
        "target_origin_sha256": origin_before,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    target_step = _full_current_residual_step(
        target_next=target_star,
        current_target=current_target,
        current_terminal=current_terminal,
        nll_gradient=entry_nll_gradient,
        aggregate_gradient=entry_aggregate_gradient,
        held=outer_no_move,
        receipt=receipt,
    )
    return TargetSubcycleOuter(
        target_step,
        final_selected.selected_endpoint,
        final_selected.next_state,
        tuple(microsteps),
        tuple(target_results),
        tuple(kl_results),
        final_selected,
        current_target,
        current_terminal,
        target_step.receipt,
    )


def reconstruct_target_subcycle_final_selected(
    outer: TargetSubcycleOuter,
    *,
    step_index: int,
) -> tuple[P1R52SelectedTarget, TargetSubcycleFinalBridgeReceipt]:
    """Fail-close bridge exposing only the final microstep endpoint to C3."""

    if (
        not isinstance(outer, TargetSubcycleOuter)
        or isinstance(step_index, bool)
        or not 0 <= step_index < 8
        or outer.receipt.get("outer_step_index") != step_index
        or outer.receipt.get("executed_microstep_count") != len(outer.microsteps)
        or not outer.microsteps
        or outer.target_step.receipt is not outer.receipt
    ):
        raise ODEBFContractError("P1R52 target-subcycle final bridge input differs")
    final_sha = tensor_sha256(outer.final_selected.target_step.target_next)
    writer_sha = tensor_sha256(outer.target_step.target_next)
    if final_sha != writer_sha:
        raise ODEBFStateError("P1R52 stale target reached writer bridge")
    endpoint_identity = outer.selected_endpoint.identity_sha256
    if endpoint_identity != outer.final_selected.selected_endpoint.identity_sha256:
        raise ODEBFStateError("P1R52 final endpoint objective binding differs")
    residual = (
        outer.target_step.target_next.detach().to(device="cpu", dtype=torch.float64)
        - outer.current_terminal.detach().to(device="cpu", dtype=torch.float64)
    )
    residual_sha = tensor_sha256(residual)
    if residual_sha != outer.receipt.get("final_full_current_residual_sha256"):
        raise ODEBFStateError("P1R52 final writer residual is stale")
    selected = P1R52SelectedTarget(
        outer.target_step,
        outer.selected_endpoint,
        outer.next_state,
        outer.receipt,
    )
    bridge = TargetSubcycleFinalBridgeReceipt(
        "TARGET_SUBCYCLE_FINAL_SELECTED_BRIDGE_PASS",
        step_index,
        str(outer.receipt["schedule"]["cell"]),
        int(outer.receipt["configured_microstep_count"]),
        len(outer.microsteps),
        final_sha,
        writer_sha,
        str(outer.receipt["outer_entry_target_sha256"]),
        str(outer.receipt["current_terminal_sha256"]),
        residual_sha,
        str(outer.receipt["identity_sha256"]),
        endpoint_identity,
        True,
        True,
        0,
        0,
    )
    return selected, bridge


__all__ = [
    "INSTRUCTION_ID",
    "METHOD_ID",
    "SCHEDULES",
    "TargetSubcycleFinalBridgeReceipt",
    "TargetSubcycleMicrostep",
    "TargetSubcycleOuter",
    "TargetSubcycleSchedule",
    "TargetTimescaleCell",
    "reconstruct_target_subcycle_final_selected",
    "run_target_subcycle_scheduler",
    "schedule_for_cell",
]
