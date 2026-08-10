"""Common terminal cold coordinate for the fixed-E8 R10 diagnostic.

The controller coordinate is a request-wise additive residual at the canonical
target-z capture site.  One residual column belongs to one request and is
shared across that request's six controller contexts and all five writer
layers.  Native/direct-z values are deliberately absent from this module.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import torch

from .accounting import ComputeLedger
from .contracts import BATCH_SIZE, ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import WaypointFactor, tensor_sha256
from .p1_backend import (
    FULL_CURRENT_RESIDUAL_DIVISOR,
    P1DynamicField,
    SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1,
    SharedTerminalResidualInput,
    _virtual_context,
)
from .target_new_nll import RoutingObjective, evaluate_routing_objective
from .cold_start_target import (
    _rewrap_layer_output,
    _rng_identity,
    _unwrap_layer_output,
)


COMMON_COLD_INSTRUCTION_ID = (
    "ODEEDIT-S05-ODE-BF-COMMON-COLDCOORD-FIXED-E8-P1R10-V1"
)
COMMON_COLD_LAYER_ORDER = (4, 5, 6, 7, 8)
COMMON_COLD_CONTEXTS_PER_REQUEST = 6
COMMON_COLD_H = 1.0 / 8.0
COMMON_COLD_EPS = 1.0e-12
COMMON_COLD_TARGET_STEP_SEMANTICS = (
    "STEP_INDEPENDENT_FINITE_UNIT_FIELD_LOOKAHEAD"
)
COMMON_COLD_TARGET_OVERLAY_DEFINITION = "r_k+(z-z_k)"


class CommonColdScale(str, Enum):
    ROBUST_SHARED = "ROBUST_SHARED_REQUEST_SCALE_V1"
    BATCH_GLOBAL = "MATCHED_BATCH_GLOBAL_SCALE_V1"


@dataclass(frozen=True, slots=True)
class CommonColdScaleMetric:
    scale: CommonColdScale
    shared_speed: float
    z_base_sha256: str
    request_order_sha256: str
    per_request_z_base_norm: tuple[float, ...]
    identity_sha256: str

    @classmethod
    def from_z_base(
        cls,
        z_base: torch.Tensor,
        request_order_sha256: str,
        scale: CommonColdScale | str,
    ) -> "CommonColdScaleMetric":
        selected = CommonColdScale(scale)
        if (
            not isinstance(z_base, torch.Tensor)
            or z_base.ndim != 2
            or z_base.shape[1] != BATCH_SIZE
            or not torch.isfinite(z_base).all()
            or len(request_order_sha256) != 64
        ):
            raise ODEBFContractError("common cold scale entry differs")
        norms = torch.linalg.vector_norm(
            z_base.detach().to(device="cpu", dtype=torch.float64), dim=0
        )
        speed = float(torch.median(norms))
        if not math.isfinite(speed) or speed <= COMMON_COLD_EPS:
            raise ODEBFContractError("common cold shared scale is degenerate")
        payload = {
            "schema": "ode-edit-common-cold-scale/v1",
            "scale": selected.value,
            "shared_speed": speed,
            "shared_speed_definition": "median_i_l2_norm_z_base_i",
            "z_base_sha256": tensor_sha256(z_base),
            "request_order_sha256": request_order_sha256,
            "per_request_z_base_norm": [float(item) for item in norms],
            "metric_evaluation_dtype": "torch.float64",
            "epsilon": COMMON_COLD_EPS,
            "native_or_direct_z_access_count": 0,
            "clipping_or_rescue_count": 0,
        }
        return cls(
            selected,
            speed,
            payload["z_base_sha256"],
            request_order_sha256,
            tuple(float(item) for item in norms),
            canonical_hash(payload),
        )

    def velocity(
        self, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if (
            not isinstance(gradient, torch.Tensor)
            or gradient.ndim != 2
            or gradient.shape[1] != BATCH_SIZE
            or not torch.isfinite(gradient).all()
        ):
            raise ODEBFContractError("common cold target gradient differs")
        value = gradient.detach().to(device="cpu", dtype=torch.float64).contiguous()
        per_request = torch.linalg.vector_norm(value, dim=0)
        zero_mask = per_request <= COMMON_COLD_EPS
        if self.scale is CommonColdScale.ROBUST_SHARED:
            denominator = torch.clamp(per_request, min=COMMON_COLD_EPS)
            velocity64 = -self.shared_speed * value / denominator.unsqueeze(0)
            velocity64[:, zero_mask] = 0.0
            allocation = "PER_REQUEST_EQUAL_NONZERO_EUCLIDEAN_SPEED"
        else:
            frobenius = float(torch.linalg.vector_norm(value))
            denominator_value = max(frobenius, COMMON_COLD_EPS)
            velocity64 = (
                -self.shared_speed
                * math.sqrt(BATCH_SIZE)
                * value
                / denominator_value
            )
            if frobenius <= COMMON_COLD_EPS:
                velocity64.zero_()
            allocation = "BATCH_GLOBAL_MATCHED_TOTAL_FROBENIUS_SPEED"
        velocity = velocity64.to(dtype=torch.float32).contiguous()
        if not torch.isfinite(velocity).all():
            raise ODEBFContractError("common cold velocity is non-finite")
        receipt = {
            "schema": "ode-edit-common-cold-velocity/v1",
            "scale_identity_sha256": self.identity_sha256,
            "scale": self.scale.value,
            "allocation": allocation,
            "shared_speed": self.shared_speed,
            "gradient_sha256": tensor_sha256(value),
            "gradient_norm_by_request": [float(item) for item in per_request],
            "zero_gradient_by_request": [bool(item) for item in zero_mask],
            "zero_gradient_count": int(zero_mask.sum()),
            "gradient_frobenius_norm": float(torch.linalg.vector_norm(value)),
            "velocity_sha256": tensor_sha256(velocity),
            "velocity_norm_by_request": [
                float(item)
                for item in torch.linalg.vector_norm(velocity64, dim=0)
            ],
            "velocity_frobenius_norm": float(torch.linalg.vector_norm(velocity64)),
            "native_or_direct_z_access_count": 0,
            "clip_or_rescue_count": 0,
        }
        receipt["identity_sha256"] = canonical_hash(receipt)
        return velocity, receipt

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-common-cold-scale/v1",
            "scale": self.scale.value,
            "shared_speed": self.shared_speed,
            "shared_speed_definition": "median_i_l2_norm_z_base_i",
            "z_base_sha256": self.z_base_sha256,
            "request_order_sha256": self.request_order_sha256,
            "per_request_z_base_norm": list(self.per_request_z_base_norm),
            "metric_evaluation_dtype": "torch.float64",
            "epsilon": COMMON_COLD_EPS,
            "native_or_direct_z_access_count": 0,
            "clipping_or_rescue_count": 0,
            "identity_sha256": self.identity_sha256,
        }


def _authoritative_additive_assignment_errors(
    before: torch.Tensor,
    residual: torch.Tensor,
    assigned: torch.Tensor,
) -> tuple[float, float]:
    if (
        not all(isinstance(value, torch.Tensor) for value in (before, residual, assigned))
        or before.shape != residual.shape
        or before.shape != assigned.shape
        or before.dtype != residual.dtype
        or before.dtype != assigned.dtype
        or before.device != residual.device
        or before.device != assigned.device
        or before.numel() == 0
        or not all(
            bool(torch.isfinite(value).all())
            for value in (before, residual, assigned)
        )
    ):
        raise ODEBFContractError("common cold additive assignment receipt differs")
    expected = before + residual
    assignment_error = float(
        torch.max(torch.abs(assigned.float() - expected.float())).detach().cpu()
    )
    realized_delta_error = float(
        torch.max(
            torch.abs((assigned - before).float() - residual.float())
        ).detach().cpu()
    )
    if not math.isfinite(assignment_error) or not math.isfinite(realized_delta_error):
        raise ODEBFContractError("common cold additive assignment is non-finite")
    return assignment_error, realized_delta_error


class RequestResidualActivationOverlay:
    """Add one request residual at the canonical terminal lookup site."""

    def __init__(
        self,
        model: torch.nn.Module,
        layer_name: str,
        residual: torch.Tensor,
        lookup_positions: Sequence[int],
        *,
        contexts_per_request: int = COMMON_COLD_CONTEXTS_PER_REQUEST,
    ) -> None:
        positions = tuple(int(item) for item in lookup_positions)
        if (
            residual.ndim != 2
            or residual.shape[1] != BATCH_SIZE
            or contexts_per_request != COMMON_COLD_CONTEXTS_PER_REQUEST
            or len(positions) != BATCH_SIZE * contexts_per_request
            or not torch.isfinite(residual).all()
        ):
            raise ODEBFContractError("common cold additive overlay geometry differs")
        self.model = model
        self.layer_name = layer_name
        self.residual = residual
        self.lookup_positions = positions
        self.contexts_per_request = contexts_per_request
        self.calls = 0
        self.request_ordinals: list[int] = []
        self.maximum_exact_delta_error = 0.0
        self.maximum_authoritative_assignment_error = 0.0
        self._handle: Any = None

    def _hook(self, _module: torch.nn.Module, _inputs: Any, output: Any) -> Any:
        if self.calls >= len(self.lookup_positions):
            raise ODEBFContractError("common cold overlay received extra forward")
        activation = _unwrap_layer_output(output)
        request_index = self.calls // self.contexts_per_request
        raw_position = self.lookup_positions[self.calls]
        residual = self.residual[:, request_index].to(
            device=activation.device, dtype=activation.dtype
        )
        if activation.ndim != 3 or activation.shape[-1] != residual.numel():
            raise ODEBFContractError("common cold overlay layout differs")
        patched = activation.clone()
        if activation.shape[0] == 1:
            position = raw_position if raw_position >= 0 else activation.shape[1] + raw_position
            if position < 0 or position >= activation.shape[1]:
                raise ODEBFContractError("common cold overlay lookup is out of range")
            before = activation[0, position, :]
            patched_expected = before + residual
            patched[0, position, :] = patched_expected
            assigned = patched[0, position, :]
            observed = patched[0, position, :] - before
        elif activation.shape[1] == 1:
            position = raw_position if raw_position >= 0 else activation.shape[0] + raw_position
            if position < 0 or position >= activation.shape[0]:
                raise ODEBFContractError("common cold overlay lookup is out of range")
            before = activation[position, 0, :]
            patched_expected = before + residual
            patched[position, 0, :] = patched_expected
            assigned = patched[position, 0, :]
            observed = patched[position, 0, :] - before
        else:
            raise ODEBFContractError("common cold overlay batch layout differs")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (before, residual, patched_expected, assigned, observed)
        ):
            raise ODEBFContractError("common cold overlay contains non-finite values")
        assignment_error, error = _authoritative_additive_assignment_errors(
            before, residual, assigned
        )
        self.maximum_exact_delta_error = max(self.maximum_exact_delta_error, error)
        self.maximum_authoritative_assignment_error = max(
            self.maximum_authoritative_assignment_error, assignment_error
        )
        self.request_ordinals.append(request_index)
        self.calls += 1
        return _rewrap_layer_output(output, patched)

    def __enter__(self) -> "RequestResidualActivationOverlay":
        self._handle = self.model.get_submodule(self.layer_name).register_forward_hook(
            self._hook
        )
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        del exc_type, exc, traceback
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def assert_complete(self) -> None:
        expected = tuple(
            ordinal
            for ordinal in range(BATCH_SIZE)
            for _ in range(self.contexts_per_request)
        )
        if (
            self._handle is not None
            or self.calls != len(self.lookup_positions)
            or tuple(self.request_ordinals) != expected
        ):
            raise ODEBFStateError("common cold overlay call/cleanup contract differs")

    def raw_free_payload(self) -> dict[str, Any]:
        self.assert_complete()
        payload = {
            "policy_identity": SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1,
            "mode": "REQUEST_WISE_ADDITIVE_TERMINAL_RESIDUAL",
            "request_count": BATCH_SIZE,
            "contexts_per_request": self.contexts_per_request,
            "hook_call_count": self.calls,
            "request_ordinal_sha256": canonical_hash(self.request_ordinals),
            "residual_sha256": tensor_sha256(self.residual),
            "maximum_exact_delta_error": self.maximum_exact_delta_error,
            "realized_delta_error_role": "BF16_ROUNDING_OBSERVATION_ONLY",
            "realized_delta_error_decision_influence_count": 0,
            "authoritative_assignment_definition": "patched_expected=before+residual",
            "maximum_authoritative_assignment_error": (
                self.maximum_authoritative_assignment_error
            ),
            "absolute_replacement_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


@dataclass(frozen=True, slots=True)
class CommonColdObjectiveReceipt:
    value: float
    per_request_values: tuple[float, ...]
    residual_sha256: str
    target_span_sha256: str
    context_sha256: str
    lookup_plan_sha256: str
    model_forward_count: int
    processed_token_count: int
    backward_count: int
    gradient: torch.Tensor | None
    overlay: dict[str, Any]
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-common-cold-objective/v1",
            "value": self.value,
            "per_request_values": list(self.per_request_values),
            "residual_sha256": self.residual_sha256,
            "target_span_sha256": self.target_span_sha256,
            "context_sha256": self.context_sha256,
            "lookup_plan_sha256": self.lookup_plan_sha256,
            "model_forward_count": self.model_forward_count,
            "processed_token_count": self.processed_token_count,
            "backward_count": self.backward_count,
            "gradient_present": self.gradient is not None,
            "overlay": dict(self.overlay),
            "target_old_access_count": 0,
            "native_or_direct_z_access_count": 0,
            "identity_sha256": self.identity_sha256,
        }


def evaluate_common_cold_objective(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    contexts: Sequence[Sequence[str]],
    *,
    layer_name: str,
    lookup_positions: Sequence[int],
    residual: torch.Tensor,
    require_gradient: bool,
) -> CommonColdObjectiveReceipt:
    device = next(model.parameters()).device
    flat = residual.to(device=device, dtype=torch.float32).contiguous().view(-1)
    if require_gradient:
        flat = flat.detach().requires_grad_(True)
    view = flat.view(residual.shape)
    before_rng = _rng_identity()
    overlay = RequestResidualActivationOverlay(
        model, layer_name, view, lookup_positions
    )
    with overlay:
        result = evaluate_routing_objective(
            model,
            tokenizer,
            requests,
            objective=RoutingObjective.TARGET_NEW_NLL,
            contexts=contexts,
            gradient_input=flat if require_gradient else None,
        )
    overlay_payload = overlay.raw_free_payload()
    if _rng_identity() != before_rng:
        raise ODEBFStateError("common cold objective changed RNG")
    gradient = (
        None
        if result.input_gradient is None
        else result.input_gradient.view(residual.shape)
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
    )
    if require_gradient != (gradient is not None):
        raise ODEBFContractError("common cold objective gradient accounting differs")
    values = tuple(
        float(item)
        for item in result.per_request_values.detach().to(
            device="cpu", dtype=torch.float64
        )
    )
    payload = {
        "schema": "ode-edit-common-cold-objective/v1",
        "objective": RoutingObjective.TARGET_NEW_NLL.value,
        "value": float(result.loss.detach().to(device="cpu", dtype=torch.float64)),
        "per_request_values": list(values),
        "residual_sha256": tensor_sha256(residual),
        "target_span_sha256": result.target_span_sha256,
        "context_sha256": result.context_sha256,
        "lookup_plan_sha256": canonical_hash(list(lookup_positions)),
        "model_forward_count": result.model_forward_count,
        "processed_token_count": result.processed_token_count,
        "backward_count": result.backward_count,
        "overlay_identity_sha256": overlay_payload["identity_sha256"],
        "target_old_access_count": 0,
        "native_or_direct_z_access_count": 0,
    }
    return CommonColdObjectiveReceipt(
        payload["value"],
        values,
        payload["residual_sha256"],
        result.target_span_sha256,
        result.context_sha256,
        payload["lookup_plan_sha256"],
        result.model_forward_count,
        result.processed_token_count,
        result.backward_count,
        gradient,
        overlay_payload,
        canonical_hash(payload),
    )


def common_terminal_residual_input(
    target_state: torch.Tensor,
    terminal_current_z: torch.Tensor,
    request_order_sha256: str,
) -> SharedTerminalResidualInput:
    target = target_state.detach().to(device="cpu", dtype=torch.float32).contiguous()
    current = (
        terminal_current_z.detach().to(device="cpu", dtype=torch.float32).contiguous()
    )
    if target.shape != current.shape or target.shape[1] != BATCH_SIZE:
        raise ODEBFContractError("common terminal residual geometry differs")
    residual = (target - current).contiguous()
    return SharedTerminalResidualInput(
        residual=residual,
        terminal_current_z=current,
        request_order_sha256=request_order_sha256,
    )


class CommonSharedResidualTargetFieldOverlay:
    """Differentiate the same shared terminal residual through five writers."""

    DEFINITION = COMMON_COLD_TARGET_OVERLAY_DEFINITION
    STEP_SEMANTICS = COMMON_COLD_TARGET_STEP_SEMANTICS

    def __init__(
        self,
        model: torch.nn.Module,
        field: P1DynamicField,
        applied_coefficients: torch.Tensor,
        target_state_variable: torch.Tensor,
    ) -> None:
        self.model = model
        self.field = field
        self.applied_coefficients = applied_coefficients
        self.target_state_variable = target_state_variable
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _shared_left(self, device: torch.device) -> torch.Tensor:
        anchor = self.field.target_state.to(device=device, dtype=torch.float32)
        residual = self.field.layers[0].residual.to(
            device=device, dtype=torch.float32
        )
        if self.target_state_variable.shape != anchor.shape or residual.shape != anchor.shape:
            raise ODEBFContractError("common target-field overlay shape differs")
        return residual + (
            self.target_state_variable.to(device=device, dtype=torch.float32) - anchor
        )

    def _hook(self, index: int):
        layer = self.field.layers[index]

        def apply(_module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            if (
                not inputs
                or not isinstance(inputs[0], torch.Tensor)
                or not isinstance(output, torch.Tensor)
            ):
                raise ODEBFContractError("common target-field Linear contract differs")
            hidden = inputs[0]
            right = layer.q.to(device=hidden.device, dtype=torch.float32)
            left = self._shared_left(hidden.device)
            if (
                hidden.shape[-1] != right.shape[0]
                or right.shape[1] != left.shape[1]
                or output.shape[-1] != left.shape[0]
            ):
                raise ODEBFContractError("common target-field geometry differs")
            perturbation = (hidden.float() @ right) @ left.T
            return (
                output.float()
                + self.applied_coefficients[index] * perturbation
            ).to(dtype=output.dtype)

        return apply

    def __enter__(self) -> "CommonSharedResidualTargetFieldOverlay":
        layers = tuple(int(item.layer) for item in self.field.layers)
        residual_hashes = tuple(tensor_sha256(item.residual) for item in self.field.layers)
        if (
            layers != COMMON_COLD_LAYER_ORDER
            or len(set(residual_hashes)) != 1
            or self.applied_coefficients.shape != (len(COMMON_COLD_LAYER_ORDER),)
            or self.target_state_variable.shape != self.field.target_state.shape
            or not self.target_state_variable.requires_grad
        ):
            raise ODEBFContractError("common target-field inventory differs")
        try:
            for index, layer in enumerate(self.field.layers):
                if (
                    layer.residual_definition
                    != SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1
                    or layer.residual_divisor != FULL_CURRENT_RESIDUAL_DIVISOR
                ):
                    raise ODEBFContractError("common target-field policy differs")
                module_name = layer.weight_name[: -len(".weight")]
                module = self.model.get_submodule(module_name)
                if type(module) is not torch.nn.Linear:
                    raise ODEBFContractError("common target-field target is not Linear")
                self._handles.append(module.register_forward_hook(self._hook(index)))
        except BaseException:
            for handle in reversed(self._handles):
                handle.remove()
            self._handles.clear()
            raise
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        del exc_type, exc, traceback
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        return False


def write_aware_common_target_velocity(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    contexts: Sequence[Sequence[str]],
    *,
    target_layer_name: str,
    lookup_positions: Sequence[int],
    field: P1DynamicField,
    velocity_coefficients: Sequence[float],
    cumulative_factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
    metric: CommonColdScaleMetric,
    ledger: ComputeLedger,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute F_z at the exact Euler weight probe ``h*v`` once."""

    device = next(model.parameters()).device
    velocity = tuple(float(item) for item in velocity_coefficients)
    if len(velocity) != len(COMMON_COLD_LAYER_ORDER) or any(
        not math.isfinite(item) for item in velocity
    ):
        raise ODEBFContractError("common target velocity coefficient differs")
    applied = torch.tensor(
        tuple(COMMON_COLD_H * item for item in velocity),
        device=device,
        dtype=torch.float32,
    )
    scientific_applied = tuple(float(COMMON_COLD_H * item) for item in velocity)
    flat = field.target_state.to(device=device, dtype=torch.float32).contiguous().view(-1)
    flat = flat.detach().requires_grad_(True)
    target = flat.view(field.target_state.shape)
    base_residual = field.layers[0].residual.to(device=device, dtype=torch.float32)
    residual_variable = base_residual + target - field.target_state.to(
        device=device, dtype=torch.float32
    )
    before_rng = _rng_identity()
    terminal_overlay = RequestResidualActivationOverlay(
        model, target_layer_name, residual_variable, lookup_positions
    )
    with _virtual_context(model, cumulative_factors_by_weight):
        with CommonSharedResidualTargetFieldOverlay(
            model, field, applied, target
        ):
            with terminal_overlay:
                result = evaluate_routing_objective(
                    model,
                    tokenizer,
                    requests,
                    objective=RoutingObjective.TARGET_NEW_NLL,
                    contexts=contexts,
                    gradient_input=flat,
                )
    overlay_payload = terminal_overlay.raw_free_payload()
    if _rng_identity() != before_rng:
        raise ODEBFStateError("common target velocity changed RNG")
    if result.input_gradient is None or result.backward_count != BATCH_SIZE:
        raise ODEBFContractError("common target velocity gradient differs")
    gradient = result.input_gradient.view(field.target_state.shape).to(
        device="cpu", dtype=torch.float64
    )
    target_velocity, scale_receipt = metric.velocity(gradient)
    applied_target_step64 = (
        float(COMMON_COLD_H)
        * target_velocity.detach().to(device="cpu", dtype=torch.float64)
    )
    target_new_nll_applied_step_reduction = float(
        -torch.sum(gradient * applied_target_step64)
    )
    if (
        not math.isfinite(target_new_nll_applied_step_reduction)
        or target_new_nll_applied_step_reduction < 0.0
    ):
        raise ODEBFContractError(
            "common target velocity applied-step reduction differs"
        )
    ledger.increment("backward", result.backward_count)
    ledger.increment("target_backward", result.backward_count)
    payload = {
        "schema": "ode-edit-common-cold-write-aware-target-velocity/v1",
        "field_sha256": field.identity_sha256,
        "field_target_state_sha256": tensor_sha256(field.target_state),
        "residual_sha256": tensor_sha256(field.layers[0].residual),
        "velocity_coefficient": list(velocity),
        "target_probe_applied_coefficient": [float(item) for item in applied.cpu()],
        "target_probe_coefficient": [float(item) for item in applied.cpu()],
        "target_probe_scientific_coefficient": list(scientific_applied),
        "target_probe_effective_coefficient": [
            float(item) for item in applied.cpu()
        ],
        "target_probe_effective_dtype": str(applied.dtype),
        "target_probe_effective_device_type": applied.device.type,
        "target_probe_float64_to_float32_cast_delta": [
            float(applied[index].item()) - scientific_applied[index]
            for index in range(len(scientific_applied))
        ],
        "target_probe_cast_decision_influence_count": 0,
        "velocity_sha256": tensor_sha256(target_velocity),
        "target_probe_coefficient_definition": "h*v",
        "h": COMMON_COLD_H,
        "h_application_count": 1,
        "scale_velocity": scale_receipt,
        "target_velocity_sha256": tensor_sha256(target_velocity),
        "target_new_nll_applied_step_reduction": (
            target_new_nll_applied_step_reduction
        ),
        "target_new_nll_applied_step_reduction_definition": (
            "-dot(existing_target_gradient,h*model_facing_F_z)"
        ),
        "target_gradient_reuse_count": 1,
        "additional_target_graph_count": 0,
        "target_backward_count": result.backward_count,
        "processed_token_count": result.processed_token_count,
        "terminal_additive_overlay": overlay_payload,
        "target_overlay_definition": COMMON_COLD_TARGET_OVERLAY_DEFINITION,
        "target_velocity_step_semantics": COMMON_COLD_TARGET_STEP_SEMANTICS,
        "candidate_coupled": False,
        "native_or_direct_z_access_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return target_velocity, payload


def common_cold_bootstrap(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    contexts: Sequence[Sequence[str]],
    *,
    target_layer_name: str,
    lookup_positions: Sequence[int],
    z_base: torch.Tensor,
    metric: CommonColdScaleMetric,
    ledger: ComputeLedger,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Perform exactly one target-only Euler step with W fixed at W0."""

    zero = torch.zeros_like(z_base, dtype=torch.float32, device="cpu").contiguous()
    before_rng = _rng_identity()
    before = evaluate_common_cold_objective(
        model,
        tokenizer,
        requests,
        contexts,
        layer_name=target_layer_name,
        lookup_positions=lookup_positions,
        residual=zero,
        require_gradient=True,
    )
    if before.gradient is None:
        raise ODEBFContractError("common cold bootstrap gradient is absent")
    velocity, velocity_receipt = metric.velocity(before.gradient)
    residual_after = (COMMON_COLD_H * velocity).contiguous()
    after = evaluate_common_cold_objective(
        model,
        tokenizer,
        requests,
        contexts,
        layer_name=target_layer_name,
        lookup_positions=lookup_positions,
        residual=residual_after,
        require_gradient=False,
    )
    if _rng_identity() != before_rng:
        raise ODEBFStateError("common cold bootstrap changed RNG")
    decrease = float(before.value - after.value)
    if not math.isfinite(decrease) or decrease <= 0.0:
        raise ODEBFContractError("BOOTSTRAP_READINESS_FAILED: target NLL did not decrease")
    joint_target = (z_base.to(dtype=torch.float32, device="cpu") + residual_after).contiguous()
    ledger.increment("target_backward", before.backward_count)
    ledger.increment("backward", before.backward_count)
    payload = {
        "schema": "ode-edit-common-cold-bootstrap/v1",
        "scale": metric.scale.value,
        "h_boot": COMMON_COLD_H,
        "transition_count": 1,
        "retry_count": 0,
        "writer_count": 0,
        "weight_mutation_count": 0,
        "pre_bootstrap": {
            "target_sha256": tensor_sha256(z_base),
            "residual_sha256": tensor_sha256(zero),
            "residual_exact_zero": bool(torch.count_nonzero(zero) == 0),
            "objective": before.raw_free_payload(),
        },
        "joint_entry": {
            "target_sha256": tensor_sha256(joint_target),
            "residual_sha256": tensor_sha256(residual_after),
            "residual_nonzero": bool(torch.count_nonzero(residual_after) > 0),
            "objective": after.raw_free_payload(),
        },
        "velocity": velocity_receipt,
        "mean_target_new_nll_decrease": decrease,
        "tau_joint_after_bootstrap": 0.0,
        "native_or_direct_z_access_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return joint_target, payload


def common_cold_source_contract() -> dict[str, Any]:
    payload = {
        "instruction_id": COMMON_COLD_INSTRUCTION_ID,
        "residual_policy": SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1,
        "residual_definition": "z_k-stop_gradient(H_T(W_k,x_canonical))",
        "controller_overlay": "H_T(W_k,x_i_c)+r_i_k",
        "request_specific": True,
        "contexts_per_request": COMMON_COLD_CONTEXTS_PER_REQUEST,
        "shared_across_writer_layers": list(COMMON_COLD_LAYER_ORDER),
        "remaining_layer_divisor": 1,
        "bootstrap_transition_count": 1,
        "joint_grid_count": 8,
        "h": COMMON_COLD_H,
        "native_or_direct_z_access_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "COMMON_COLD_CONTEXTS_PER_REQUEST",
    "COMMON_COLD_H",
    "COMMON_COLD_INSTRUCTION_ID",
    "COMMON_COLD_LAYER_ORDER",
    "CommonColdScale",
    "CommonColdScaleMetric",
    "CommonColdObjectiveReceipt",
    "CommonSharedResidualTargetFieldOverlay",
    "RequestResidualActivationOverlay",
    "common_cold_bootstrap",
    "common_cold_source_contract",
    "common_terminal_residual_input",
    "evaluate_common_cold_objective",
    "write_aware_common_target_velocity",
]
