"""Typed request-local amplitude policies for the frozen P1R52 field.

The P1R52 semantic/preservation direction, clamp, and selector remain the
authorities.  This module only supplies one vectorized nonnegative amplitude
per request.  It imports no model, writer, evaluator, or optimizer code.
"""

from __future__ import annotations

from enum import Enum
import math
from typing import Any, Mapping

import torch

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .p1r52_r42_safe_kdc import (
    P1R52AmplitudeContext,
    P1R52AmplitudeDecision,
    P1R52_NUMERICAL_EPSILON,
)


INSTRUCTION_ID = "ODEEDIT-S05-P1R53-REQUEST-LOCAL-TARGET-SPEED-B100-V1"
METHOD_ID = "P1R53-REQUEST-LOCAL-TARGET-SPEED-C3-KSTEP"
REPORTING_EPSILON = 1.0e-30


class RequestLocalSpeedArm(str, Enum):
    LP_S = "LP-S"
    LFD_E = "LFD-E"

    @property
    def policy_name(self) -> str:
        return {
            RequestLocalSpeedArm.LP_S:
                "REQUEST_LOCAL_PARENT_AMPLITUDE_WITH_SHARED_BATCH_SCALE",
            RequestLocalSpeedArm.LFD_E:
                "ENTRY_MATCHED_LOCAL_FINITE_DEMAND",
        }[self]


class P1R53ScientificInvalid(ODEBFContractError):
    """A finite-demand scientific precondition failed without fallback."""


class RequestLocalSpeedPolicy:
    """One arm-local, vectorized amplitude policy with a frozen K0 state."""

    def __init__(
        self,
        arm: RequestLocalSpeedArm | str,
        *,
        request_count: int,
        outer_count: int = 8,
    ) -> None:
        self.arm = arm if isinstance(arm, RequestLocalSpeedArm) else RequestLocalSpeedArm(arm)
        if (
            isinstance(request_count, bool)
            or request_count <= 0
            or isinstance(outer_count, bool)
            or outer_count != 8
        ):
            raise ODEBFContractError("P1R53 amplitude policy geometry differs")
        self.request_count = request_count
        self.outer_count = outer_count
        self._call_count = 0
        self._kappa: torch.Tensor | None = None
        self._kappa_sha256: str | None = None
        self._kappa_calibration_count = 0
        self._kappa_refresh_count = 0
        self._k0_lp_amplitude_sha256: str | None = None
        self._k0_selected_amplitude_sha256: str | None = None
        self._receipt_identities: list[str] = []
        self._selected_observation_identities: list[str] = []

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def kappa_calibration_count(self) -> int:
        return self._kappa_calibration_count

    @property
    def kappa_refresh_count(self) -> int:
        return self._kappa_refresh_count

    def __call__(self, context: P1R52AmplitudeContext) -> P1R52AmplitudeDecision:
        if not isinstance(context, P1R52AmplitudeContext):
            raise ODEBFContractError("P1R53 amplitude context type differs")
        if context.step_index != self._call_count or self._call_count >= self.outer_count:
            raise ODEBFContractError("P1R53 amplitude clock differs")
        tensors = (
            context.semantic_gradient,
            context.semantic_gradient_norm,
            context.entry_semantic_gradient_norm,
            context.target_new_nll,
            context.active_mask,
            context.kdc_direction,
            context.local_parent_amplitude,
        )
        if (
            context.semantic_gradient.ndim != 2
            or context.semantic_gradient.shape[1] != self.request_count
            or any(item.device.type != "cpu" for item in tensors)
            or context.semantic_gradient.dtype != torch.float64
            or context.semantic_gradient_norm.shape != (self.request_count,)
            or context.entry_semantic_gradient_norm.shape != (self.request_count,)
            or context.target_new_nll.shape != (self.request_count,)
            or context.active_mask.shape != (self.request_count,)
            or context.active_mask.dtype != torch.bool
            or context.kdc_direction.shape != context.semantic_gradient.shape
            or context.local_parent_amplitude.shape != (self.request_count,)
            or any(
                item.dtype != torch.float64
                for item in tensors
                if item is not context.active_mask
            )
        ):
            raise ODEBFContractError("P1R53 amplitude tensor geometry differs")
        numeric = tuple(item for item in tensors if item is not context.active_mask)
        if any(not bool(torch.isfinite(item).all()) for item in numeric):
            raise P1R53ScientificInvalid("P1R53 pre-clamp field input is nonfinite")
        if not math.isfinite(context.shared_speed) or context.shared_speed <= 0.0:
            raise ODEBFContractError("P1R53 frozen shared batch scale differs")

        active = context.active_mask
        local_parent = torch.where(
            active,
            context.local_parent_amplitude,
            torch.zeros_like(context.local_parent_amplitude),
        )
        sigma = -torch.sum(
            context.semantic_gradient * context.kdc_direction,
            dim=0,
        )
        if bool(torch.any(active & (sigma <= 0.0))):
            raise P1R53ScientificInvalid("P1R53 active semantic slope is not positive")
        if bool(torch.any(active & (context.target_new_nll <= 0.0))):
            raise P1R53ScientificInvalid("P1R53 active target-new NLL is not positive")

        if self.arm is RequestLocalSpeedArm.LP_S:
            amplitude = local_parent
            enforce_energy_identity = True
            kappa = torch.zeros_like(amplitude)
            kappa_source = "NOT_APPLICABLE_LP_S"
            calibration_now = 0
        else:
            if self._call_count == 0:
                kappa = torch.zeros_like(local_parent)
                kappa[active] = (
                    local_parent[active]
                    * sigma[active]
                    / context.target_new_nll[active]
                )
                if not bool(torch.isfinite(kappa).all()) or bool(torch.any(active & (kappa <= 0.0))):
                    raise P1R53ScientificInvalid("P1R53 K0 local kappa is invalid")
                self._kappa = kappa.detach().clone().contiguous()
                self._kappa_sha256 = tensor_sha256(self._kappa)
                self._kappa_calibration_count = self.request_count
                calibration_now = self.request_count
            else:
                if self._kappa is None or self._kappa_sha256 != tensor_sha256(self._kappa):
                    raise ODEBFContractError("P1R53 frozen local kappa state differs")
                kappa = self._kappa
                calibration_now = 0
            assert self._kappa is not None
            amplitude = torch.zeros_like(local_parent)
            if self._call_count == 0:
                # The K0 scientific lock is entry-field identity, not merely an
                # algebraically equivalent round trip through multiplication.
                amplitude.copy_(local_parent)
            else:
                amplitude[active] = (
                    self._kappa[active]
                    * context.target_new_nll[active]
                    / sigma[active]
                )
            if not bool(torch.isfinite(amplitude).all()):
                raise P1R53ScientificInvalid("P1R53 LFD-E pre-clamp amplitude is nonfinite")
            enforce_energy_identity = False
            kappa = self._kappa
            kappa_source = "K0_LOCAL_PARENT_ENTRY_RATE"

        velocity = context.kdc_direction * amplitude.unsqueeze(0)
        lp_velocity = context.kdc_direction * local_parent.unsqueeze(0)
        nominal_inner = torch.sum(context.semantic_gradient * velocity, dim=0)
        nominal_expected = -kappa * context.target_new_nll
        nominal_residual = torch.where(
            active,
            nominal_inner - nominal_expected,
            torch.zeros_like(nominal_inner),
        )
        local_energy = float(torch.sum(torch.square(local_parent)))
        reference_energy = context.counterfactual_global_reference_energy
        relative_identity_error = abs(local_energy - reference_energy) / (
            reference_energy + P1R52_NUMERICAL_EPSILON
        )
        fd_to_lp = amplitude / (local_parent + REPORTING_EPSILON)
        receipt: dict[str, Any] = {
            "p1r53_schema": "ode-edit-s05-p1r53-request-local-amplitude-decision/v1",
            "p1r53_instruction_id": INSTRUCTION_ID,
            "p1r53_method_id": METHOD_ID,
            "p1r53_arm": self.arm.value,
            "p1r53_amplitude_policy": self.arm.policy_name,
            "p1r53_outer_index": context.step_index,
            "p1r53_outer_count": self.outer_count,
            "p1r53_target_dt": 0.125,
            "p1r53_target_microsteps_per_outer": 1,
            "p1r53_total_target_time": 1.0,
            "p1r53_local_parent_speed_by_request": [float(item) for item in local_parent],
            "p1r53_local_fd_speed_by_request": [float(item) for item in amplitude],
            "p1r53_fd_to_lp_speed_ratio_by_request": [float(item) for item in fd_to_lp],
            "p1r53_local_semantic_slope_by_request": [float(item) for item in sigma],
            "p1r53_local_kappa_by_request": [float(item) for item in kappa],
            "p1r53_nominal_nll_decrease_rate_by_request": [
                float(item) for item in -nominal_inner
            ],
            "p1r53_nominal_rate_identity_max_abs_residual": (
                float(torch.max(torch.abs(nominal_residual)))
                if self.arm is RequestLocalSpeedArm.LFD_E
                else None
            ),
            "p1r53_local_parent_energy": local_energy,
            "p1r53_counterfactual_global_reference_energy": reference_energy,
            "p1r53_relative_energy_identity_error": relative_identity_error,
            "p1r53_cross_request_nll_decision_count": 0,
            "p1r53_cross_request_gradient_reduction_decision_count": 0,
            "p1r53_batch_nll_denominator_decision_count": 0,
            "p1r53_global_energy_sum_decision_count": 0,
            "p1r53_unused_energy_redistribution_count": 0,
            "p1r53_shared_batch_scale_access_count": 1,
            "p1r53_counterfactual_reference_energy_observation_count": 1,
            "p1r53_kappa_source": kappa_source,
            "p1r53_kappa_calibration_count_this_call": calibration_now,
            "p1r53_kappa_calibration_count_total": self._kappa_calibration_count,
            "p1r53_kappa_refresh_count": self._kappa_refresh_count,
            "p1r53_angle_gate_count": 0,
            "p1r53_trust_gate_count": 0,
            "p1r53_line_search_count": 0,
            "p1r53_retry_count": 0,
            "p1r53_fallback_optimizer_count": 0,
            "p1r53_adam_state_count": 0,
            "p1r53_momentum_state_count": 0,
            "p1r53_per_request_backward_loop_count": 0,
            "p1r53_added_model_forward_count": 0,
            "p1r53_added_backward_count": 0,
            "p1r53_reporting_epsilon": REPORTING_EPSILON,
            "p1r53_reporting_epsilon_decision_influence_count": 0,
            "p1r53_local_parent_amplitude_sha256": tensor_sha256(local_parent),
            "p1r53_selected_amplitude_sha256": tensor_sha256(amplitude),
            "p1r53_selected_velocity_sha256": tensor_sha256(velocity),
            "p1r53_local_parent_velocity_sha256": tensor_sha256(lp_velocity),
            "p1r53_kappa_sha256": None if self._kappa is None else self._kappa_sha256,
        }
        receipt["p1r53_amplitude_receipt_identity"] = canonical_hash(receipt)
        if self._call_count == 0:
            self._k0_lp_amplitude_sha256 = tensor_sha256(local_parent)
            self._k0_selected_amplitude_sha256 = tensor_sha256(amplitude)
        self._receipt_identities.append(str(receipt["p1r53_amplitude_receipt_identity"]))
        self._call_count += 1
        return P1R52AmplitudeDecision(
            amplitude.detach().clone().contiguous(),
            self.arm.policy_name,
            enforce_energy_identity,
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
        """Bind post-selector demand telemetry without influencing the action."""

        if (
            step_index != len(self._selected_observation_identities)
            or self._call_count != step_index + 1
            or len(current_nll) != self.request_count
            or len(selected_nll) != self.request_count
            or len(selection) != self.request_count
            or target_dt != 0.125
        ):
            raise ODEBFContractError("P1R53 selected observation clock differs")
        current = torch.tensor(current_nll, dtype=torch.float64)
        selected = torch.tensor(selected_nll, dtype=torch.float64)
        actual = current - selected
        kappa = torch.tensor(
            field_receipt["p1r53_local_kappa_by_request"], dtype=torch.float64
        )
        nominal = target_dt * kappa * current
        post_clamp_speed = torch.sqrt(
            torch.tensor(
                field_receipt["post_cast_energy_by_request"], dtype=torch.float64
            )
        )
        if self.arm is RequestLocalSpeedArm.LFD_E:
            realized = actual / (nominal + REPORTING_EPSILON)
            unmet = nominal - actual
            nominal_values: list[float] | None = [float(item) for item in nominal]
            realized_values: list[float] | None = [float(item) for item in realized]
            unmet_values: list[float] | None = [float(item) for item in unmet]
        else:
            nominal_values = None
            realized_values = None
            unmet_values = None
        payload: dict[str, Any] = {
            "schema": "ode-edit-s05-p1r53-request-local-selected-observation/v1",
            "instruction_id": INSTRUCTION_ID,
            "arm": self.arm.value,
            "outer_index": step_index,
            "target_dt": target_dt,
            "current_target_new_nll_by_request": [float(item) for item in current],
            "selected_target_new_nll_by_request": [float(item) for item in selected],
            "actual_selected_nll_decrease_by_request": [float(item) for item in actual],
            "nominal_nll_decrease_by_request": nominal_values,
            "realized_demand_ratio_by_request": realized_values,
            "unmet_demand_by_request": unmet_values,
            "pre_clamp_velocity_norm_by_request": list(
                field_receipt["allocation_amplitude_by_request"]
            ),
            "post_clamp_actual_velocity_norm_by_request": [
                float(item) for item in post_clamp_speed
            ],
            "clamp_hit_by_request": list(field_receipt["clamp_hit_by_request"]),
            "selection_by_request": list(selection),
            "reporting_epsilon": REPORTING_EPSILON,
            "reporting_epsilon_decision_influence_count": 0,
            "selected_observation_decision_influence_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        self._selected_observation_identities.append(str(payload["identity_sha256"]))
        return payload

    def terminal_receipt(self) -> Mapping[str, Any]:
        if (
            self._call_count != self.outer_count
            or len(self._selected_observation_identities) != self.outer_count
        ):
            raise ODEBFContractError("P1R53 amplitude policy terminal clock differs")
        if self.arm is RequestLocalSpeedArm.LFD_E:
            if self._kappa is None or self._kappa_calibration_count != self.request_count:
                raise ODEBFContractError("P1R53 LFD-E terminal kappa state differs")
        elif self._kappa is not None or self._kappa_calibration_count != 0:
            raise ODEBFContractError("P1R53 LP-S unexpectedly owns kappa state")
        payload: dict[str, Any] = {
            "schema": "ode-edit-s05-p1r53-request-local-amplitude-policy-terminal/v1",
            "instruction_id": INSTRUCTION_ID,
            "method_id": METHOD_ID,
            "arm": self.arm.value,
            "policy_name": self.arm.policy_name,
            "request_count": self.request_count,
            "outer_count": self.outer_count,
            "amplitude_call_count": self._call_count,
            "kappa_calibration_count": self._kappa_calibration_count,
            "kappa_refresh_count": self._kappa_refresh_count,
            "kappa_sha256": self._kappa_sha256,
            "k0_local_parent_amplitude_sha256": self._k0_lp_amplitude_sha256,
            "k0_selected_amplitude_sha256": self._k0_selected_amplitude_sha256,
            "k0_lp_lfd_amplitude_identity": (
                self._k0_lp_amplitude_sha256 == self._k0_selected_amplitude_sha256
                if self.arm is RequestLocalSpeedArm.LFD_E
                else None
            ),
            "decision_receipt_identities": list(self._receipt_identities),
            "selected_observation_count": len(self._selected_observation_identities),
            "selected_observation_identities": list(self._selected_observation_identities),
            "vectorized_request_axis": True,
            "per_request_backward_loop_count": 0,
            "additional_model_forward_count": 0,
            "additional_backward_count": 0,
            "kappa_sweep_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


__all__ = [
    "INSTRUCTION_ID",
    "METHOD_ID",
    "P1R53ScientificInvalid",
    "REPORTING_EPSILON",
    "RequestLocalSpeedArm",
    "RequestLocalSpeedPolicy",
]
