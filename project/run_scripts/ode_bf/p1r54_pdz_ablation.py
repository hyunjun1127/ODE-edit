"""Clocked P1R54 PDZ target transitions for the target-time sweep.

The module keeps the sealed P1R54 amplitude implementation and the P1R52
direction, preservation, origin-clamp, and FP32-closure implementation as the
scientific authorities.  It adds only a global target-microstep clock and the
explicit selector-free DIRECT_EULER transition required by the ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    P1R52AmplitudeContext,
    P1R52AmplitudeDecision,
    P1R52AmplitudePolicy,
    P1R52SelectedTarget,
    P1R52_NUMERICAL_EPSILON,
    prepare_p1r52_target_proposal,
)
from .p1r52_target_timescale import (
    TargetSubcycleMicrostep,
    TargetSubcycleOuter,
    TargetSubcycleSchedule,
    run_target_subcycle_scheduler,
)
from .p1r54_energyfree_localz import EnergyFreeLocalZArm, EnergyFreeLocalZPolicy
from .scalable_batched_model import ScalableObjectiveResult


INSTRUCTION_ID = "ODEEDIT-S05-P1R54-PDZ-PRC-TIME-SWEEP-T2-T3-T5-DIRECT-V1"
METHOD_ID = "P1R54-PDZ-PRC-TIME-SWEEP-T2-T3-T5-VS-T1-DIRECT-C3-KSTEP"


class PDZAblationArm(str, Enum):
    PDZ_T2_PRC = "PDZ-T2-PRC"
    PDZ_T3_PRC = "PDZ-T3-PRC"
    PDZ_T5_PRC = "PDZ-T5-PRC"
    PDZ_T1_DIRECT = "PDZ-T1-DIRECT"


class PDZAblationScheduleCell(str, Enum):
    PDZ_T2_PRC = "PDZ-T2-PRC"
    PDZ_T3_PRC = "PDZ-T3-PRC"
    PDZ_T5_PRC = "PDZ-T5-PRC"
    PDZ_T1_DIRECT = "PDZ-T1-DIRECT"


class PDZAblationSelection(str, Enum):
    DIRECT_EULER = "DIRECT_EULER"


class P1R54ScientificBoundaryNTSM(RuntimeError):
    """A selector-free Primary endpoint worsened beyond the inherited epsilon."""

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        super().__init__("SCIENTIFIC_BOUNDARY_NTSM")
        self.receipt = dict(receipt)


@dataclass(frozen=True, slots=True)
class TargetMicrostepClock:
    outer_index: int
    microstep_index: int
    global_ordinal: int
    microsteps_per_outer: int
    target_dt: Fraction

    def __post_init__(self) -> None:
        if (
            isinstance(self.outer_index, bool)
            or not 0 <= self.outer_index < 8
            or isinstance(self.microstep_index, bool)
            or not 0 <= self.microstep_index < self.microsteps_per_outer
            or self.microsteps_per_outer not in (1, 2, 3, 5)
            or self.global_ordinal
            != self.outer_index * self.microsteps_per_outer + self.microstep_index
            or self.target_dt != Fraction(1, 8)
        ):
            raise ODEBFContractError("P1R54 ablation target clock differs")

    @property
    def time_before(self) -> Fraction:
        return self.global_ordinal * self.target_dt

    @property
    def time_after(self) -> Fraction:
        return (self.global_ordinal + 1) * self.target_dt

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "outer_index": self.outer_index,
            "microstep_index": self.microstep_index,
            "global_ordinal": self.global_ordinal,
            "microsteps_per_outer": self.microsteps_per_outer,
            "target_dt": float(self.target_dt),
            "time_before": float(self.time_before),
            "time_after": float(self.time_after),
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


def schedule_for_arm(arm: PDZAblationArm | str) -> TargetSubcycleSchedule:
    selected = arm if isinstance(arm, PDZAblationArm) else PDZAblationArm(arm)
    geometry = {
        PDZAblationArm.PDZ_T2_PRC: (2, PDZAblationScheduleCell.PDZ_T2_PRC),
        PDZAblationArm.PDZ_T3_PRC: (3, PDZAblationScheduleCell.PDZ_T3_PRC),
        PDZAblationArm.PDZ_T5_PRC: (5, PDZAblationScheduleCell.PDZ_T5_PRC),
        PDZAblationArm.PDZ_T1_DIRECT: (1, PDZAblationScheduleCell.PDZ_T1_DIRECT),
    }
    microsteps, cell = geometry[selected]
    horizon = Fraction(microsteps)
    return TargetSubcycleSchedule(cell, horizon, microsteps, Fraction(1, 8))  # type: ignore[arg-type]


class ClockedPDZAmplitudePolicy:
    """Reuse sealed PDZ amplitude bytes under an explicit global clock."""

    def __init__(self, arm: PDZAblationArm | str, *, request_count: int) -> None:
        self.arm = arm if isinstance(arm, PDZAblationArm) else PDZAblationArm(arm)
        self.request_count = request_count
        self.microsteps_per_outer = schedule_for_arm(self.arm).microsteps_per_outer
        if isinstance(request_count, bool) or request_count <= 0:
            raise ODEBFContractError("P1R54 ablation request count differs")
        self._call_count = 0
        self._rho: torch.Tensor | None = None
        self._rho_sha256: str | None = None
        self._decision_identities: list[str] = []
        self._observation_identities: list[str] = []

    @property
    def call_count(self) -> int:
        return self._call_count

    def _clock(self) -> TargetMicrostepClock:
        return TargetMicrostepClock(
            self._call_count // self.microsteps_per_outer,
            self._call_count % self.microsteps_per_outer,
            self._call_count,
            self.microsteps_per_outer,
            Fraction(1, 8),
        )

    def __call__(self, context: P1R52AmplitudeContext) -> P1R52AmplitudeDecision:
        clock = self._clock()
        if context.step_index != clock.outer_index:
            raise ODEBFContractError("P1R54 ablation amplitude outer clock differs")
        rho = context.target_origin_norm
        if rho is None:
            raise ODEBFContractError("P1R54 ablation immutable rho is absent")
        rho = rho.detach().clone().contiguous()
        rho_sha = tensor_sha256(rho)
        if self._call_count == 0:
            self._rho = rho
            self._rho_sha256 = rho_sha
        elif (
            self._rho is None
            or self._rho_sha256 != tensor_sha256(self._rho)
            or rho_sha != self._rho_sha256
            or not torch.equal(rho, self._rho)
        ):
            raise ODEBFStateError("P1R54 ablation rho refreshed after n=0")

        # The sealed policy remains the numerical amplitude authority.  A fresh
        # one-call instance avoids changing its legacy 8-outer clock contract.
        sealed = EnergyFreeLocalZPolicy(EnergyFreeLocalZArm.PDZ, request_count=self.request_count)
        decision = sealed(replace(context, step_index=0))
        receipt = dict(decision.receipt)
        receipt.pop("p1r54_amplitude_receipt_identity", None)
        receipt.update(
            {
                "p1r54_schema": "ode-edit-s05-p1r54-pdz-ablation-amplitude-decision/v1",
                "p1r54_instruction_id": INSTRUCTION_ID,
                "p1r54_method_id": METHOD_ID,
                "p1r54_ablation_arm": self.arm.value,
                "p1r54_outer_index": clock.outer_index,
                "p1r54_microstep_index": clock.microstep_index,
                "p1r54_global_ordinal": clock.global_ordinal,
                "p1r54_target_microsteps_per_outer": self.microsteps_per_outer,
                "p1r54_total_target_time": float(
                    8 * self.microsteps_per_outer * clock.target_dt
                ),
                "p1r54_clock": clock.raw_free_payload(),
                "p1r54_rho_calibration_count_this_call": (
                    self.request_count if clock.global_ordinal == 0 else 0
                ),
                "p1r54_rho_calibration_count_total": self.request_count,
                "p1r54_rho_capture_event_count": 1,
                "p1r54_rho_refresh_count": 0,
                "p1r54_sealed_pdz_amplitude_call_count": 1,
            }
        )
        receipt["p1r54_amplitude_receipt_identity"] = canonical_hash(receipt)
        self._decision_identities.append(
            str(receipt["p1r54_amplitude_receipt_identity"])
        )
        self._call_count += 1
        return P1R52AmplitudeDecision(
            decision.amplitude.detach().clone().contiguous(),
            decision.policy_name,
            decision.enforce_reference_energy_identity,
            receipt,
        )

    def observe_selected(
        self,
        *,
        step_index: int,
        current_nll: tuple[float, ...],
        selected_nll: tuple[float, ...],
        field_receipt: Mapping[str, Any],
        selection: tuple[str, ...],
        target_dt: float,
    ) -> Mapping[str, Any]:
        ordinal = len(self._observation_identities)
        clock = TargetMicrostepClock(
            ordinal // self.microsteps_per_outer,
            ordinal % self.microsteps_per_outer,
            ordinal,
            self.microsteps_per_outer,
            Fraction(1, 8),
        )
        allowed = (
            {"PRIMARY", "RESCUE", "CURRENT"}
            if self.arm is not PDZAblationArm.PDZ_T1_DIRECT
            else {PDZAblationSelection.DIRECT_EULER.value}
        )
        if (
            step_index != clock.outer_index
            or self._call_count != ordinal + 1
            or target_dt != 0.125
            or len(current_nll) != self.request_count
            or len(selected_nll) != self.request_count
            or len(selection) != self.request_count
            or any(item not in allowed for item in selection)
        ):
            raise ODEBFContractError("P1R54 ablation selected observation differs")
        current = torch.tensor(current_nll, dtype=torch.float64)
        selected = torch.tensor(selected_nll, dtype=torch.float64)
        post_clamp = torch.tensor(
            field_receipt["post_cast_energy_by_request"], dtype=torch.float64
        )
        selected_delta = current - selected
        payload: dict[str, Any] = {
            "schema": "ode-edit-s05-p1r54-pdz-ablation-selected-observation/v1",
            "instruction_id": INSTRUCTION_ID,
            "arm": self.arm.value,
            "clock": clock.raw_free_payload(),
            "current_target_new_nll_by_request": [float(item) for item in current],
            "selected_target_new_nll_by_request": [float(item) for item in selected],
            "selected_nll_decrease_by_request": [float(item) for item in selected_delta],
            "negative_progress_by_request": [bool(item < 0.0) for item in selected_delta],
            "negative_progress_count": int(torch.sum(selected_delta < 0.0)),
            "raw_energy_by_request": list(field_receipt["raw_energy_by_request"]),
            "postclamp_primary_energy_by_request": [float(item) for item in post_clamp],
            "selected_direct_energy_by_request": (
                [float(item) for item in post_clamp]
                if self.arm is PDZAblationArm.PDZ_T1_DIRECT
                else None
            ),
            "clamp_hit_by_request": list(field_receipt["clamp_hit_by_request"]),
            "selection_by_request": list(selection),
            "selected_observation_decision_influence_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        self._observation_identities.append(str(payload["identity_sha256"]))
        return payload

    def terminal_receipt(self) -> Mapping[str, Any]:
        expected = 8 * self.microsteps_per_outer
        if (
            self._call_count != expected
            or len(self._observation_identities) != expected
            or self._rho is None
            or self._rho_sha256 != tensor_sha256(self._rho)
        ):
            raise ODEBFStateError("P1R54 ablation amplitude terminal clock differs")
        payload: dict[str, Any] = {
            "schema": "ode-edit-s05-p1r54-pdz-ablation-amplitude-terminal/v1",
            "instruction_id": INSTRUCTION_ID,
            "method_id": METHOD_ID,
            "arm": self.arm.value,
            "outer_count": 8,
            "microsteps_per_outer": self.microsteps_per_outer,
            "amplitude_call_count": self._call_count,
            "field_gradient_count": self._call_count,
            "primary_evaluation_count": self._call_count,
            "selector_count": (
                self._call_count
                if self.arm is not PDZAblationArm.PDZ_T1_DIRECT
                else 0
            ),
            "direct_euler_count": (
                self._call_count
                if self.arm is PDZAblationArm.PDZ_T1_DIRECT
                else 0
            ),
            "rho_calibration_count": self.request_count,
            "rho_capture_event_count": 1,
            "rho_refresh_count": 0,
            "rho_sha256": self._rho_sha256,
            "decision_receipt_identities": list(self._decision_identities),
            "selected_observation_identities": list(self._observation_identities),
            "rescue_proposal_count": 0 if self.arm is PDZAblationArm.PDZ_T1_DIRECT else "PARENT_PRC",
            "rescue_evaluation_count": 0 if self.arm is PDZAblationArm.PDZ_T1_DIRECT else "PARENT_PRC",
            "current_selection_count": 0 if self.arm is PDZAblationArm.PDZ_T1_DIRECT else "PARENT_PRC",
            "hold_rollback_retry_line_search_adaptive_count": 0,
            "heldout_decision_influence_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


def _direct_selected(
    *,
    proposal: Any,
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    current_nll: ScalableObjectiveResult,
    primary_endpoint: ScalableObjectiveResult,
    outer_step_index: int,
) -> P1R52SelectedTarget:
    selected_target = proposal.primary_step.target_next
    delta = (
        selected_target.detach().to(device="cpu", dtype=torch.float64)
        - current_target.detach().to(device="cpu", dtype=torch.float64)
    )
    accepted_norm = torch.linalg.vector_norm(delta, dim=0)
    cumulative = tuple(
        before + float(increment)
        for before, increment in zip(
            proposal.next_state.cumulative_accepted_activation_path,
            accepted_norm,
            strict=True,
        )
    )
    current_values = tuple(float(item) for item in current_nll.per_request_values)
    endpoint_values = tuple(
        float(item) for item in primary_endpoint.per_request_values
    )
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-pdz-direct-euler-selection/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "arm": PDZAblationArm.PDZ_T1_DIRECT.value,
        "outer_step_index": outer_step_index,
        "selection_type": PDZAblationSelection.DIRECT_EULER.value,
        "selection_by_request": [
            PDZAblationSelection.DIRECT_EULER.value for _ in current_values
        ],
        "accepted_target_nll_improvement_by_request": [
            left - right
            for left, right in zip(current_values, endpoint_values, strict=True)
        ],
        "accepted_activation_path_increment_by_request": [
            float(item) for item in accepted_norm
        ],
        "cumulative_accepted_activation_path_by_request": list(cumulative),
        "rescue_proposal_count": 0,
        "rescue_evaluation_count": 0,
        "selector_call_count": 0,
        "current_candidate_count": 0,
        "hold_count": 0,
        "rollback_count": 0,
        "retry_count": 0,
        "line_search_count": 0,
        "adaptive_count": 0,
        "semantic_acceptance_helper_count": 0,
        "primary_endpoint_forward_count": 1,
        "primary_endpoint_decision_influence_count": 0,
    }
    target_step = _full_current_residual_step(
        target_next=selected_target,
        current_target=current_target,
        current_terminal=current_terminal,
        nll_gradient=proposal.primary_step.nll_gradient,
        aggregate_gradient=proposal.primary_step.combined_gradient,
        held=tuple(False for _ in current_values),
        receipt=receipt,
    )
    next_state = P1R51ControllerState(
        proposal.next_state.entry_semantic_gradient_norm,
        cumulative,
    )
    return P1R52SelectedTarget(
        target_step, primary_endpoint, next_state, target_step.receipt
    )


def run_direct_euler_subcycle_scheduler(
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
    """Run one selector-free PDZ Euler transition at fixed physical W_k."""

    expected = schedule_for_arm(PDZAblationArm.PDZ_T1_DIRECT)
    if (
        schedule.raw_free_payload() != expected.raw_free_payload()
        or not isinstance(amplitude_policy, ClockedPDZAmplitudePolicy)
        or amplitude_policy.arm is not PDZAblationArm.PDZ_T1_DIRECT
        or (first_target_result is None) != (first_kl_result is None)
    ):
        raise ODEBFContractError("P1R54 direct scheduler activation differs")
    clock = TargetMicrostepClock(
        outer_step_index, 0, outer_step_index, 1, Fraction(1, 8)
    )
    physical_before, history_before, factors_before = fixed_state_identities()
    origin_before = tensor_sha256(target_origin)
    target_result = (
        evaluate_target(current_target)
        if first_target_result is None
        else first_target_result
    )
    kl_result = (
        evaluate_kl(current_target) if first_kl_result is None else first_kl_result
    )
    proposal = prepare_p1r52_target_proposal(
        current_target,
        current_terminal,
        target_origin,
        target_result,
        kl_result,
        lock,
        state,
        alias=alias,
        step_index=outer_step_index,
        shared_speed=shared_speed,
        kl_teacher_input_sha256=teacher_sha256,
        target_dt=0.125,
        amplitude_policy=amplitude_policy,
    )
    primary_endpoint = evaluate_endpoint(
        proposal.primary_step.target_next, "PRIMARY"
    )
    worsening = float(primary_endpoint.loss - target_result.loss)
    physical_now, history_now, factors_now = fixed_state_identities()
    if (
        (physical_now, history_now, factors_now)
        != (physical_before, history_before, factors_before)
        or tensor_sha256(target_origin) != origin_before
    ):
        raise ODEBFStateError("P1R54 direct fixed state changed before selection")
    if worsening > P1R52_NUMERICAL_EPSILON:
        current_values = tuple(float(item) for item in target_result.per_request_values)
        endpoint_values = tuple(
            float(item) for item in primary_endpoint.per_request_values
        )
        boundary: dict[str, Any] = {
            "schema": "ode-edit-s05-p1r54-pdz-direct-scientific-boundary-ntsm/v1",
            "instruction_id": INSTRUCTION_ID,
            "method_id": METHOD_ID,
            "status": "SCIENTIFIC_BOUNDARY_NTSM",
            "arm": PDZAblationArm.PDZ_T1_DIRECT.value,
            "clock": clock.raw_free_payload(),
            "current_target_sha256": tensor_sha256(current_target),
            "direct_target_sha256": tensor_sha256(
                proposal.primary_step.target_next
            ),
            "current_mean_target_new_nll": float(target_result.loss),
            "direct_mean_target_new_nll": float(primary_endpoint.loss),
            "aggregate_worsening": worsening,
            "numerical_tolerance": P1R52_NUMERICAL_EPSILON,
            "request_worsening_by_request": [
                right - left
                for left, right in zip(
                    current_values, endpoint_values, strict=True
                )
            ],
            "field_receipt_identity": proposal.receipt["identity_sha256"],
            "raw_delta_sha256": proposal.receipt["nominal_delta_sha256"],
            "postclamp_primary_sha256": tensor_sha256(
                proposal.primary_step.target_next
            ),
            "selection_type": PDZAblationSelection.DIRECT_EULER.value,
            "writer_update_count": 0,
            "fallback_zero_write_rescue_contraction_exclusion_skip_count": 0,
            "retry_count": 0,
            "decision_influence_count": 0,
        }
        boundary["identity_sha256"] = canonical_hash(boundary)
        raise P1R54ScientificBoundaryNTSM(boundary)

    selected = _direct_selected(
        proposal=proposal,
        current_target=current_target,
        current_terminal=current_terminal,
        current_nll=target_result,
        primary_endpoint=primary_endpoint,
        outer_step_index=outer_step_index,
    )
    observed = amplitude_policy.observe_selected(
        step_index=outer_step_index,
        current_nll=tuple(float(item) for item in target_result.per_request_values),
        selected_nll=tuple(
            float(item) for item in primary_endpoint.per_request_values
        ),
        field_receipt=proposal.receipt,
        selection=tuple(
            PDZAblationSelection.DIRECT_EULER.value
            for _ in target_result.per_request_values
        ),
        target_dt=0.125,
    )
    micro_receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-pdz-direct-microstep/v1",
        "instruction_id": INSTRUCTION_ID,
        "arm": PDZAblationArm.PDZ_T1_DIRECT.value,
        "outer_step_index": outer_step_index,
        "microstep_index": 0,
        "global_field_evaluation_ordinal": outer_step_index,
        "target_time_before": float(clock.time_before),
        "target_time_after": float(clock.time_after),
        "target_dt": 0.125,
        "entry_target_sha256": tensor_sha256(current_target),
        "last_micro_entry_objective_identity": target_result.identity_sha256,
        "final_selected_endpoint_objective_identity": primary_endpoint.identity_sha256,
        "last_micro_entry_objective": target_result.raw_free_payload(),
        "final_selected_endpoint_objective": primary_endpoint.raw_free_payload(),
        "selected_target_sha256": tensor_sha256(selected.target_step.target_next),
        "selection_by_request": [
            PDZAblationSelection.DIRECT_EULER.value
            for _ in target_result.per_request_values
        ],
        "primary_receipt_identity": proposal.receipt["identity_sha256"],
        "selected_receipt_identity": selected.receipt["identity_sha256"],
        "field_receipt": dict(proposal.receipt),
        "selection_receipt": dict(selected.receipt),
        "external_amplitude_selected_observation": dict(observed),
        "target_energy_delta_over_target_dt": True,
        "writer_authority_count": 0,
        "materialization_count": 0,
        "history_append_count": 0,
        "early_break_count": 0,
    }
    micro_receipt["identity_sha256"] = canonical_hash(micro_receipt)
    micro = TargetSubcycleMicrostep(
        0,
        outer_step_index,
        target_result,
        kl_result,
        selected,
        micro_receipt,
    )
    target_star = selected.target_step.target_next
    full_residual = (
        target_star.detach().to(device="cpu", dtype=torch.float64)
        - current_terminal.detach().to(device="cpu", dtype=torch.float64)
    )
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-pdz-direct-outer/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "schedule": schedule.raw_free_payload(),
        "outer_step_index": outer_step_index,
        "configured_microstep_count": 1,
        "executed_microstep_count": 1,
        "target_dt": 0.125,
        "outer_target_time": 0.125,
        "target_time_before": float(clock.time_before),
        "target_time_after": float(clock.time_after),
        "last_micro_entry_objective_identity": target_result.identity_sha256,
        "final_selected_endpoint_objective_identity": primary_endpoint.identity_sha256,
        "last_micro_entry_objective_is_primary_count": 0,
        "final_selected_endpoint_objective_is_primary_count": 1,
        "microstep_receipts": [dict(micro_receipt)],
        "selection_by_request": micro_receipt["selection_by_request"],
        "semantic_held_mask": [False for _ in target_result.per_request_values],
        "target_hold_mask": [False for _ in target_result.per_request_values],
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
        "direct_euler_count": len(target_result.per_request_values),
        "selector_rescue_current_count": 0,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    target_step = _full_current_residual_step(
        target_next=target_star,
        current_target=current_target,
        current_terminal=current_terminal,
        nll_gradient=proposal.primary_step.nll_gradient,
        aggregate_gradient=proposal.primary_step.combined_gradient,
        held=tuple(False for _ in target_result.per_request_values),
        receipt=receipt,
    )
    return TargetSubcycleOuter(
        target_step,
        primary_endpoint,
        selected.next_state,
        (micro,),
        (target_result,),
        (kl_result,),
        selected,
        current_target,
        current_terminal,
        target_step.receipt,
    )


def run_pdz_ablation_subcycle_scheduler(**kwargs: Any) -> TargetSubcycleOuter:
    schedule = kwargs.get("schedule")
    if not isinstance(schedule, TargetSubcycleSchedule):
        raise ODEBFContractError("P1R54 ablation schedule type differs")
    cell = schedule.cell.value
    if cell in {
        PDZAblationScheduleCell.PDZ_T2_PRC.value,
        PDZAblationScheduleCell.PDZ_T3_PRC.value,
        PDZAblationScheduleCell.PDZ_T5_PRC.value,
    }:
        return run_target_subcycle_scheduler(**kwargs)
    if cell == PDZAblationScheduleCell.PDZ_T1_DIRECT.value:
        return run_direct_euler_subcycle_scheduler(**kwargs)
    raise ODEBFContractError("P1R54 ablation schedule cell differs")


__all__ = [
    "ClockedPDZAmplitudePolicy",
    "INSTRUCTION_ID",
    "METHOD_ID",
    "PDZAblationArm",
    "PDZAblationScheduleCell",
    "PDZAblationSelection",
    "P1R54ScientificBoundaryNTSM",
    "TargetMicrostepClock",
    "run_direct_euler_subcycle_scheduler",
    "run_pdz_ablation_subcycle_scheduler",
    "schedule_for_arm",
]
