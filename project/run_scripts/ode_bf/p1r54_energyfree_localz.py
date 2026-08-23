"""Energy-free request-local target amplitudes for the frozen P1R52 field.

The module imports no model, writer, evaluator, or optimizer code.  It only
maps the existing request-local origin norm and current target-new NLL to one
nonnegative amplitude per request.  Direction, preservation, clamp, selector,
and writer authority remain in their existing P1R52/C3 implementations.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

import torch

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .p1r52_r42_safe_kdc import P1R52AmplitudeContext, P1R52AmplitudeDecision


INSTRUCTION_ID = "ODEEDIT-S05-P1R54-ENERGYFREE-LOCALZ-B100-V1"
METHOD_ID = "P1R54-ENERGYFREE-LOCALZ-C3-KSTEP"


class EnergyFreeLocalZArm(str, Enum):
    FZ = "FZ"
    PDZ = "PDZ"

    @property
    def policy_name(self) -> str:
        return {
            EnergyFreeLocalZArm.FZ: "FIXED_ORIGIN_NORM_LOCAL_Z",
            EnergyFreeLocalZArm.PDZ: "NLL_PACED_ORIGIN_NORM_LOCAL_Z",
        }[self]


class P1R54ScientificInvalid(ODEBFContractError):
    """A P1R54 scientific invariant failed without retry or fallback."""


class EnergyFreeLocalZPolicy:
    """Vectorized FZ/PDZ amplitude with immutable request-local ``rho``."""

    def __init__(
        self,
        arm: EnergyFreeLocalZArm | str,
        *,
        request_count: int,
        outer_count: int = 8,
    ) -> None:
        self.arm = arm if isinstance(arm, EnergyFreeLocalZArm) else EnergyFreeLocalZArm(arm)
        if (
            isinstance(request_count, bool)
            or request_count <= 0
            or isinstance(outer_count, bool)
            or outer_count != 8
        ):
            raise ODEBFContractError("P1R54 amplitude policy geometry differs")
        self.request_count = request_count
        self.outer_count = outer_count
        self._call_count = 0
        self._rho: torch.Tensor | None = None
        self._rho_sha256: str | None = None
        self._receipt_identities: list[str] = []
        self._selected_observation_identities: list[str] = []

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def rho_sha256(self) -> str | None:
        return self._rho_sha256

    def __call__(self, context: P1R52AmplitudeContext) -> P1R52AmplitudeDecision:
        if not isinstance(context, P1R52AmplitudeContext):
            raise ODEBFContractError("P1R54 amplitude context type differs")
        if context.step_index != self._call_count or self._call_count >= self.outer_count:
            raise ODEBFContractError("P1R54 amplitude clock differs")
        rho_input = context.target_origin_norm
        if (
            rho_input is None
            or context.semantic_gradient.ndim != 2
            or context.semantic_gradient.shape[1] != self.request_count
            or context.semantic_gradient.device.type != "cpu"
            or context.semantic_gradient.dtype != torch.float64
            or context.target_new_nll.shape != (self.request_count,)
            or context.target_new_nll.device.type != "cpu"
            or context.target_new_nll.dtype != torch.float64
            or context.active_mask.shape != (self.request_count,)
            or context.active_mask.device.type != "cpu"
            or context.active_mask.dtype != torch.bool
            or context.kdc_direction.shape != context.semantic_gradient.shape
            or context.kdc_direction.device.type != "cpu"
            or context.kdc_direction.dtype != torch.float64
            or rho_input.shape != (self.request_count,)
            or rho_input.device.type != "cpu"
            or rho_input.dtype != torch.float64
        ):
            raise ODEBFContractError("P1R54 amplitude tensor geometry differs")
        if any(
            not bool(torch.isfinite(item).all())
            for item in (context.semantic_gradient, context.target_new_nll, context.kdc_direction, rho_input)
        ):
            raise P1R54ScientificInvalid("P1R54 pre-clamp field input is nonfinite")
        if bool(torch.any(context.target_new_nll < 0.0)):
            raise P1R54ScientificInvalid("P1R54 target-new NLL is negative")

        active = context.active_mask
        rho_current = rho_input.detach().clone().contiguous()
        if bool(torch.any(active & (rho_current <= 0.0))) or bool(torch.any(rho_current < 0.0)):
            raise P1R54ScientificInvalid("P1R54 request-local origin norm is invalid")
        if self._call_count == 0:
            self._rho = rho_current
            self._rho_sha256 = tensor_sha256(self._rho)
        elif (
            self._rho is None
            or self._rho_sha256 != tensor_sha256(self._rho)
            or not torch.equal(rho_current, self._rho)
        ):
            raise ODEBFContractError("P1R54 immutable request-local rho differs")
        assert self._rho is not None and self._rho_sha256 is not None

        semantic_inner = torch.sum(context.semantic_gradient * context.kdc_direction, dim=0)
        if bool(torch.any(active & (semantic_inner >= 0.0))):
            raise P1R54ScientificInvalid("P1R54 active direction is not semantic descent")
        direction_norm = torch.linalg.vector_norm(context.kdc_direction, dim=0)
        if bool(torch.any(active & (torch.abs(direction_norm - 1.0) > 1.0e-12))):
            raise P1R54ScientificInvalid("P1R54 active direction is not exact unit norm")

        if self.arm is EnergyFreeLocalZArm.FZ:
            pacing = torch.ones_like(self._rho)
            deficit: torch.Tensor | None = None
        else:
            deficit = -torch.expm1(-context.target_new_nll)
            if (
                not bool(torch.isfinite(deficit).all())
                or bool(torch.any(deficit < 0.0))
                or bool(torch.any(deficit > 1.0))
            ):
                raise P1R54ScientificInvalid("P1R54 PDZ deficit is invalid")
            pacing = deficit
        amplitude = torch.where(active, self._rho * pacing, torch.zeros_like(self._rho))
        if (
            not bool(torch.isfinite(amplitude).all())
            or bool(torch.any(amplitude < 0.0))
            or bool(torch.any(amplitude > self._rho))
        ):
            raise P1R54ScientificInvalid("P1R54 energy-free amplitude is invalid")
        velocity = context.kdc_direction * amplitude.unsqueeze(0)
        velocity_norm = torch.linalg.vector_norm(velocity, dim=0)
        norm_residual = torch.where(active, velocity_norm - amplitude, torch.zeros_like(amplitude))
        if float(torch.max(torch.abs(norm_residual))) > 1.0e-12:
            raise P1R54ScientificInvalid("P1R54 velocity/amplitude norm identity differs")

        receipt: dict[str, Any] = {
            "p1r54_schema": "ode-edit-s05-p1r54-energyfree-localz-amplitude-decision/v1",
            "p1r54_instruction_id": INSTRUCTION_ID,
            "p1r54_method_id": METHOD_ID,
            "p1r54_arm": self.arm.value,
            "p1r54_amplitude_policy": self.arm.policy_name,
            "p1r54_outer_index": context.step_index,
            "p1r54_outer_count": self.outer_count,
            "p1r54_target_dt": 0.125,
            "p1r54_target_microsteps_per_outer": 1,
            "p1r54_total_target_time": 1.0,
            "p1r54_rho_source": "IMMUTABLE_CASE_ENTRY_TARGET_ORIGIN_NORM",
            "p1r54_rho_by_request": [float(item) for item in self._rho],
            "p1r54_rho_sha256": self._rho_sha256,
            "p1r54_rho_calibration_count_this_call": self.request_count if self._call_count == 0 else 0,
            "p1r54_rho_calibration_count_total": self.request_count,
            "p1r54_rho_refresh_count": 0,
            "p1r54_pacing_factor_by_request": [float(item) for item in pacing],
            "p1r54_deficit_by_request": (
                None if deficit is None else [float(item) for item in deficit]
            ),
            "p1r54_amplitude_by_request": [float(item) for item in amplitude],
            "p1r54_velocity_norm_by_request": [float(item) for item in velocity_norm],
            "p1r54_semantic_inner_product_by_request": [float(item) for item in semantic_inner],
            "p1r54_direction_norm_by_request": [float(item) for item in direction_norm],
            "p1r54_velocity_amplitude_max_abs_residual": float(torch.max(torch.abs(norm_residual))),
            "p1r54_shared_scale_decision_access_count": 0,
            "p1r54_global_reference_energy_decision_access_count": 0,
            "p1r54_batch_nll_denominator_decision_access_count": 0,
            "p1r54_cross_request_nll_decision_access_count": 0,
            "p1r54_cross_request_gradient_reduction_decision_access_count": 0,
            "p1r54_entry_gradient_denominator_decision_access_count": 0,
            "p1r54_gradient_ratio_decision_access_count": 0,
            "p1r54_kappa_decision_access_count": 0,
            "p1r54_inverse_slope_decision_access_count": 0,
            "p1r54_unused_energy_redistribution_count": 0,
            "p1r54_new_threshold_count": 0,
            "p1r54_hold_reject_count": 0,
            "p1r54_line_search_count": 0,
            "p1r54_retry_count": 0,
            "p1r54_adaptive_h_count": 0,
            "p1r54_speed_floor_cap_count": 0,
            "p1r54_fallback_count": 0,
            "p1r54_per_request_backward_loop_count": 0,
            "p1r54_added_model_forward_count": 0,
            "p1r54_added_backward_count": 0,
            "p1r54_heldout_decision_influence_count": 0,
            "p1r54_kdc_direction_sha256": tensor_sha256(context.kdc_direction),
            "p1r54_amplitude_sha256": tensor_sha256(amplitude),
            "p1r54_velocity_sha256": tensor_sha256(velocity),
        }
        receipt["p1r54_amplitude_receipt_identity"] = canonical_hash(receipt)
        self._receipt_identities.append(str(receipt["p1r54_amplitude_receipt_identity"]))
        self._call_count += 1
        return P1R52AmplitudeDecision(
            amplitude.detach().clone().contiguous(),
            self.arm.policy_name,
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
        """Record post-selector facts without influencing target decisions."""

        if (
            step_index != len(self._selected_observation_identities)
            or self._call_count != step_index + 1
            or len(current_nll) != self.request_count
            or len(selected_nll) != self.request_count
            or len(selection) != self.request_count
            or target_dt != 0.125
        ):
            raise ODEBFContractError("P1R54 selected observation clock differs")
        current = torch.tensor(current_nll, dtype=torch.float64)
        selected = torch.tensor(selected_nll, dtype=torch.float64)
        post_clamp_speed = torch.sqrt(
            torch.tensor(field_receipt["post_cast_energy_by_request"], dtype=torch.float64)
        )
        payload: dict[str, Any] = {
            "schema": "ode-edit-s05-p1r54-energyfree-localz-selected-observation/v1",
            "instruction_id": INSTRUCTION_ID,
            "arm": self.arm.value,
            "outer_index": step_index,
            "target_dt": target_dt,
            "current_target_new_nll_by_request": [float(item) for item in current],
            "selected_target_new_nll_by_request": [float(item) for item in selected],
            "actual_selected_nll_decrease_by_request": [float(item) for item in current - selected],
            "pre_clamp_velocity_norm_by_request": list(field_receipt["allocation_amplitude_by_request"]),
            "post_clamp_actual_velocity_norm_by_request": [float(item) for item in post_clamp_speed],
            "clamp_hit_by_request": list(field_receipt["clamp_hit_by_request"]),
            "selection_by_request": list(selection),
            "selected_observation_decision_influence_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        self._selected_observation_identities.append(str(payload["identity_sha256"]))
        return payload

    def terminal_receipt(self) -> Mapping[str, Any]:
        if (
            self._call_count != self.outer_count
            or len(self._selected_observation_identities) != self.outer_count
            or self._rho is None
            or self._rho_sha256 != tensor_sha256(self._rho)
        ):
            raise ODEBFContractError("P1R54 amplitude policy terminal state differs")
        payload: dict[str, Any] = {
            "schema": "ode-edit-s05-p1r54-energyfree-localz-amplitude-policy-terminal/v1",
            "instruction_id": INSTRUCTION_ID,
            "method_id": METHOD_ID,
            "arm": self.arm.value,
            "policy_name": self.arm.policy_name,
            "request_count": self.request_count,
            "outer_count": self.outer_count,
            "amplitude_call_count": self._call_count,
            "rho_calibration_count": self.request_count,
            "rho_refresh_count": 0,
            "rho_sha256": self._rho_sha256,
            "decision_receipt_identities": list(self._receipt_identities),
            "selected_observation_count": len(self._selected_observation_identities),
            "selected_observation_identities": list(self._selected_observation_identities),
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
    "EnergyFreeLocalZArm",
    "EnergyFreeLocalZPolicy",
    "INSTRUCTION_ID",
    "METHOD_ID",
    "P1R54ScientificInvalid",
]
