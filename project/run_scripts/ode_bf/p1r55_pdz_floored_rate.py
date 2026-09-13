"""Request-local PDZ and PDZ-floored adaptive-rate amplitudes."""

from __future__ import annotations

from enum import Enum
import math
from typing import Any, Mapping

import torch

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .p1r52_r42_safe_kdc import P1R52AmplitudeContext, P1R52AmplitudeDecision
from .p1r55_objective_risk import INSTRUCTION_ID, ObjectiveRiskPolicy


TARGET_DT = 0.125


class AmplitudePolicy(str, Enum):
    PDZ = "PDZ"
    PDZ_FLOORED_RATE = "PDZ_FLOORED_RATE"


class P1R55ScientificInvalid(ODEBFContractError):
    """A scientific slope/rate invariant failed without fallback."""


def solve_pdz_floored_rate(
    *,
    risk: torch.Tensor,
    rho: torch.Tensor,
    slope: torch.Tensor,
    active_mask: torch.Tensor,
    target_dt: float = TARGET_DT,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    """Solve the bounded request-local rate problem without unsafe division."""

    if (
        risk.ndim != 1
        or rho.shape != risk.shape
        or slope.shape != risk.shape
        or active_mask.shape != risk.shape
        or risk.dtype is not torch.float64
        or rho.dtype is not torch.float64
        or slope.dtype is not torch.float64
        or active_mask.dtype is not torch.bool
        or any(item.device.type != "cpu" for item in (risk, rho, slope, active_mask))
        or not isinstance(target_dt, float)
        or target_dt != TARGET_DT
    ):
        raise ODEBFContractError("P1R55 rate-solver geometry differs")
    if (
        not torch.isfinite(risk).all()
        or not torch.isfinite(rho).all()
        or bool(torch.any(risk < 0.0))
        or bool(torch.any(rho < 0.0))
        or bool(torch.any(active_mask & (rho <= 0.0)))
    ):
        raise P1R55ScientificInvalid("P1R55 risk/rho is invalid")
    if bool(torch.any(active_mask & ((~torch.isfinite(slope)) | (slope <= 0.0)))):
        raise P1R55ScientificInvalid("P1R55 active semantic slope is invalid")
    if bool(torch.any((~active_mask) & (~torch.isfinite(slope)))):
        raise P1R55ScientificInvalid("P1R55 inactive semantic slope is nonfinite")

    deficit = -torch.expm1(-risk)
    pdz = rho * deficit
    demand = -math.expm1(-target_dt) * risk
    capacity = target_dt * rho * slope
    cap_saturated = active_mask & (capacity <= demand)
    feasible = active_mask & (~cap_saturated)
    amplitude = torch.zeros_like(risk)
    rate = torch.full_like(risk, torch.nan)
    amplitude[cap_saturated] = rho[cap_saturated]
    if bool(torch.any(feasible)):
        # Division is reached only after the positive-finite slope and
        # capacity>demand gates prove the raw rate lies below rho.
        rate[feasible] = demand[feasible] / (target_dt * slope[feasible])
        amplitude[feasible] = torch.maximum(pdz[feasible], rate[feasible])
    if (
        not torch.isfinite(amplitude).all()
        or bool(torch.any(amplitude < 0.0))
        or bool(torch.any(active_mask & (amplitude < pdz)))
        or bool(torch.any(amplitude > rho))
        or bool(torch.any((~active_mask) & (amplitude != 0.0)))
        or bool(torch.any(feasible & ((~torch.isfinite(rate)) | (rate > rho))))
    ):
        raise P1R55ScientificInvalid("P1R55 bounded rate solution differs")
    lower_bound = active_mask & torch.isclose(amplitude, pdz, rtol=0.0, atol=0.0)
    kappa: list[float | None] = []
    for index in range(risk.numel()):
        kappa.append(
            None
            if float(demand[index]) == 0.0
            else float(capacity[index] / demand[index])
        )
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r55-pdz-floored-rate-solve/v1",
        "instruction_id": INSTRUCTION_ID,
        "target_dt": target_dt,
        "risk_by_request": [float(item) for item in risk],
        "rho_by_request": [float(item) for item in rho],
        "slope_by_request": [float(item) for item in slope],
        "pdz_amplitude_by_request": [float(item) for item in pdz],
        "rate_amplitude_by_request": [
            None if not math.isfinite(float(item)) else float(item) for item in rate
        ],
        "rate_to_pdz_ratio_by_request": [
            None
            if not math.isfinite(float(rate[index])) or float(pdz[index]) == 0.0
            else float(rate[index] / pdz[index])
            for index in range(risk.numel())
        ],
        "amplitude_by_request": [float(item) for item in amplitude],
        "amplitude_to_rho_ratio_by_request": [
            None if float(rho[index]) == 0.0 else float(amplitude[index] / rho[index])
            for index in range(risk.numel())
        ],
        "demand_by_request": [float(item) for item in demand],
        "capacity_by_request": [float(item) for item in capacity],
        "capacity_demand_ratio_by_request": kappa,
        "lower_bound_active_by_request": [bool(item) for item in lower_bound],
        "upper_cap_saturated_by_request": [bool(item) for item in cap_saturated],
        "rate_division_by_request": [bool(item) for item in feasible],
        "rate_division_count": int(feasible.sum()),
        "unsafe_predivision_count": 0,
        "pdz_lower_bound_violation_count": 0,
        "rho_upper_cap_violation_count": 0,
        "cross_request_decision_count": 0,
        "amplitude_sha256": tensor_sha256(amplitude),
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    return amplitude.contiguous(), receipt


class RequestLocalPDZPolicy:
    """Vectorized 2x2 P1R55 amplitude policy with immutable entry rho."""

    def __init__(
        self,
        *,
        risk_policy: ObjectiveRiskPolicy | str,
        amplitude_policy: AmplitudePolicy | str,
        request_count: int,
        outer_count: int = 8,
        microsteps_per_outer: int = 1,
    ) -> None:
        self.risk_policy = (
            risk_policy
            if isinstance(risk_policy, ObjectiveRiskPolicy)
            else ObjectiveRiskPolicy(risk_policy)
        )
        self.amplitude_policy = (
            amplitude_policy
            if isinstance(amplitude_policy, AmplitudePolicy)
            else AmplitudePolicy(amplitude_policy)
        )
        if (
            isinstance(request_count, bool)
            or request_count <= 0
            or outer_count != 8
            or isinstance(microsteps_per_outer, bool)
            or microsteps_per_outer not in (1, 2)
        ):
            raise ODEBFContractError("P1R55 amplitude policy geometry differs")
        self.request_count = request_count
        self.outer_count = outer_count
        self.microsteps_per_outer = microsteps_per_outer
        self._call_count = 0
        self._rho: torch.Tensor | None = None
        self._rho_sha256: str | None = None
        self._decision_ids: list[str] = []
        self._observation_ids: list[str] = []

    @property
    def arm(self) -> str:
        return f"{self.risk_policy.value}+{self.amplitude_policy.value}"

    def __call__(self, context: P1R52AmplitudeContext) -> P1R52AmplitudeDecision:
        expected_outer = self._call_count // self.microsteps_per_outer
        if (
            not isinstance(context, P1R52AmplitudeContext)
            or context.step_index != expected_outer
            or self._call_count >= self.outer_count * self.microsteps_per_outer
        ):
            raise ODEBFContractError("P1R55 amplitude clock differs")
        required = (
            context.semantic_gradient,
            context.target_new_nll,
            context.active_mask,
            context.kdc_direction,
            context.target_origin_norm,
        )
        if (
            context.target_origin_norm is None
            or context.semantic_gradient.ndim != 2
            or context.semantic_gradient.shape[1] != self.request_count
            or context.target_new_nll.shape != (self.request_count,)
            or context.active_mask.shape != (self.request_count,)
            or context.kdc_direction.shape != context.semantic_gradient.shape
            or context.target_origin_norm.shape != (self.request_count,)
            or any(item.device.type != "cpu" for item in required if item is not None)
            or any(
                item.dtype is not dtype
                for item, dtype in zip(
                    required,
                    (torch.float64, torch.float64, torch.bool, torch.float64, torch.float64),
                    strict=True,
                )
                if item is not None
            )
            or any(
                not torch.isfinite(item).all()
                for item in required
                if item is not None and item.dtype is not torch.bool
            )
        ):
            raise ODEBFContractError("P1R55 amplitude input differs")
        rho_current = context.target_origin_norm.detach().clone().contiguous()
        if self._call_count == 0:
            self._rho = rho_current
            self._rho_sha256 = tensor_sha256(rho_current)
        elif (
            self._rho is None
            or self._rho_sha256 != tensor_sha256(self._rho)
            or not torch.equal(rho_current, self._rho)
        ):
            raise ODEBFContractError("P1R55 immutable request-local rho differs")
        assert self._rho is not None and self._rho_sha256 is not None
        semantic_inner = torch.sum(
            context.semantic_gradient * context.kdc_direction, dim=0
        )
        slope = -semantic_inner
        if bool(torch.any(context.active_mask & (semantic_inner >= 0.0))):
            raise P1R55ScientificInvalid("P1R55 direction is not semantic descent")
        pdz = self._rho * (-torch.expm1(-context.target_new_nll))
        rate_receipt: Mapping[str, Any] | None = None
        if self.amplitude_policy is AmplitudePolicy.PDZ:
            amplitude = torch.where(
                context.active_mask, pdz, torch.zeros_like(pdz)
            )
        else:
            amplitude, rate_receipt = solve_pdz_floored_rate(
                risk=context.target_new_nll,
                rho=self._rho,
                slope=slope,
                active_mask=context.active_mask,
            )
        if (
            not torch.isfinite(amplitude).all()
            or bool(torch.any(amplitude < 0.0))
            or bool(torch.any(amplitude > self._rho))
        ):
            raise P1R55ScientificInvalid("P1R55 amplitude is invalid")
        gradient_norm = torch.linalg.vector_norm(context.semantic_gradient, dim=0)
        gamma = torch.where(
            gradient_norm > 0.0,
            slope / gradient_norm,
            torch.zeros_like(slope),
        )
        # This receipt is flattened into the established P1R52 target receipt.
        # Keep every key in one experiment-owned namespace so provenance can be
        # merged without replacing the parent target science or telemetry.
        receipt: dict[str, Any] = {
            "p1r55_schema": "ode-edit-s05-p1r55-request-local-amplitude-decision/v1",
            "p1r55_instruction_id": INSTRUCTION_ID,
            "p1r55_arm": self.arm,
            "p1r55_risk_policy": self.risk_policy.value,
            "p1r55_amplitude_policy": self.amplitude_policy.value,
            "p1r55_outer_index": context.step_index,
            "p1r55_microstep_index": self._call_count % self.microsteps_per_outer,
            "p1r55_global_microstep_ordinal": self._call_count,
            "p1r55_target_dt": TARGET_DT,
            "p1r55_rho_source": "IMMUTABLE_CASE_ENTRY_TARGET_ORIGIN_NORM",
            "p1r55_rho_by_request": [float(item) for item in self._rho],
            "p1r55_rho_sha256": self._rho_sha256,
            "p1r55_rho_capture_count_this_call": 1 if self._call_count == 0 else 0,
            "p1r55_rho_capture_count_total": 1,
            "p1r55_rho_refresh_count": 0,
            "p1r55_risk_by_request": [float(item) for item in context.target_new_nll],
            "p1r55_semantic_gradient_sha256": tensor_sha256(
                context.semantic_gradient.contiguous()
            ),
            "p1r55_kdc_direction_sha256": tensor_sha256(
                context.kdc_direction.contiguous()
            ),
            "p1r55_semantic_gradient_norm_by_request": [
                float(item) for item in gradient_norm
            ],
            "p1r55_semantic_slope_by_request": [float(item) for item in slope],
            "p1r55_semantic_direction_efficiency_by_request": [
                float(item) for item in gamma
            ],
            "p1r55_pdz_amplitude_by_request": [float(item) for item in pdz],
            "p1r55_selected_amplitude_by_request": [float(item) for item in amplitude],
            "p1r55_rate_solver_receipt": (
                None if rate_receipt is None else dict(rate_receipt)
            ),
            "p1r55_cross_request_nll_decision_access_count": 0,
            "p1r55_cross_request_gradient_reduction_decision_count": 0,
            "p1r55_batch_energy_normalization_count": 0,
            "p1r55_unused_energy_redistribution_count": 0,
            "p1r55_batch_percentile_decision_count": 0,
            "p1r55_per_request_backward_loop_count": 0,
            "p1r55_extra_model_forward_count": 0,
            "p1r55_extra_model_backward_count": 0,
            "p1r55_heldout_decision_access_count": 0,
            "p1r55_request_cohort_gain_count": 0,
            "p1r55_learned_threshold_count": 0,
            "p1r55_retry_line_search_count": 0,
            "p1r55_writer_debt_count": 0,
            "p1r55_shared_energy_decision_count": 0,
            "p1r55_amplitude_sha256": tensor_sha256(amplitude),
        }
        receipt["p1r55_identity_sha256"] = canonical_hash(receipt)
        self._decision_ids.append(str(receipt["p1r55_identity_sha256"]))
        self._call_count += 1
        return P1R52AmplitudeDecision(
            amplitude.detach().clone().contiguous(),
            self.amplitude_policy.value,
            False,
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
        expected_call = len(self._observation_ids)
        if (
            step_index != expected_call // self.microsteps_per_outer
            or self._call_count != expected_call + 1
            or len(current_nll) != self.request_count
            or len(selected_nll) != self.request_count
            or len(selection) != self.request_count
            or target_dt != TARGET_DT
        ):
            raise ODEBFContractError("P1R55 selected observation clock differs")
        payload: dict[str, Any] = {
            "schema": "ode-edit-s05-p1r55-selected-risk-observation/v1",
            "instruction_id": INSTRUCTION_ID,
            "arm": self.arm,
            "outer_index": step_index,
            "microstep_index": expected_call % self.microsteps_per_outer,
            "global_microstep_ordinal": expected_call,
            "current_risk_by_request": list(current_nll),
            "selected_risk_by_request": list(selected_nll),
            "actual_decrease_by_request": [
                float(left - right)
                for left, right in zip(current_nll, selected_nll, strict=True)
            ],
            "selection_by_request": list(selection),
            "clamp_hit_by_request": list(field_receipt["clamp_hit_by_request"]),
            "selected_observation_decision_influence_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        self._observation_ids.append(str(payload["identity_sha256"]))
        return payload

    def terminal_receipt(self) -> Mapping[str, Any]:
        expected = self.outer_count * self.microsteps_per_outer
        if (
            self._call_count != expected
            or len(self._decision_ids) != expected
            or len(self._observation_ids) != expected
            or self._rho is None
            or self._rho_sha256 != tensor_sha256(self._rho)
        ):
            raise ODEBFContractError("P1R55 amplitude terminal state differs")
        payload: dict[str, Any] = {
            "schema": "ode-edit-s05-p1r55-request-local-amplitude-terminal/v1",
            "instruction_id": INSTRUCTION_ID,
            "arm": self.arm,
            "risk_policy": self.risk_policy.value,
            "amplitude_policy": self.amplitude_policy.value,
            "request_count": self.request_count,
            "outer_count": self.outer_count,
            "microsteps_per_outer": self.microsteps_per_outer,
            "amplitude_call_count": self._call_count,
            "rho_calibration_count": 1,
            "rho_refresh_count": 0,
            "rho_sha256": self._rho_sha256,
            "decision_receipt_identities": list(self._decision_ids),
            "selected_observation_count": len(self._observation_ids),
            "selected_observation_identities": list(self._observation_ids),
            "vectorized_request_axis": True,
            "per_request_backward_loop_count": 0,
            "additional_model_forward_count": 0,
            "additional_backward_count": 0,
            "forbidden_decision_access_count": 0,
            "retry_fallback_adaptive_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


__all__ = [
    "AmplitudePolicy",
    "P1R55ScientificInvalid",
    "RequestLocalPDZPolicy",
    "TARGET_DT",
    "solve_pdz_floored_rate",
]
