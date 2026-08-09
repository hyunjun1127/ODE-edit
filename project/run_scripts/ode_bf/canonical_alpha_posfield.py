"""Canonical-ordered Alpha proposal control for the P1R11 fixed-E8 panel."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .accounting import ComputeLedger
from .common_cold_coordinate import (
    COMMON_COLD_H,
    COMMON_COLD_LAYER_ORDER,
    CommonColdScaleMetric,
    RequestResidualActivationOverlay,
)
from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import WaypointFactor, tensor_sha256
from .p1_backend import (
    CANONICAL_ORDERED_ALPHA_REMAINING_RESIDUAL_V1,
    P1DynamicField,
    _virtual_context,
)
from .target_new_nll import RoutingObjective, evaluate_routing_objective


CANONICAL_ALPHA_POSFIELD_INSTRUCTION_ID = (
    "ODEEDIT-S05-ODE-BF-CANONICAL-ALPHA-POSFIELD-P1R11-V1"
)
CANONICAL_ALPHA_POSFIELD_SCHEMA_NAMESPACE = (
    "ode-edit-s05-canonical-alpha-posfield-p1r11"
)


class AlphaActuator(str, Enum):
    CURRENT_SHARED_AE = "CURRENT-SHARED-AE"
    CANONICAL_ORDERED_AE = "CANONICAL-ORDERED-AE"


@dataclass(slots=True)
class CanonicalAlphaPostfreezeCapture:
    native_candidates: dict[str, torch.Tensor]
    entry_weights: dict[str, torch.Tensor]
    entry_sha256: dict[str, str]
    request_order_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "reference": "Native AlphaEdit original-BF16 canonical-FP32-solve",
            "entry_sha256": dict(sorted(self.entry_sha256.items())),
            "native_candidate_sha256": {
                name: tensor_sha256(value)
                for name, value in sorted(self.native_candidates.items())
            },
            "request_order_sha256": self.request_order_sha256,
        }


class CanonicalAlphaReceiptRecorder:
    """Create-once R11 receipts without changing legacy fixed-E8 metadata."""

    def __init__(
        self,
        root: Path,
        actuator: AlphaActuator,
        write_once: Any,
    ) -> None:
        self.root = root / "canonical-alpha-posfield" / actuator.value
        self.arm = actuator
        self.variant_label = actuator.value
        self.write_once = write_once
        self.field_hashes: list[str] = []
        self.solver_hashes: list[str] = []
        self.trial_hashes: list[str] = []
        self.transition_hashes: list[str] = []
        self.accepted_hashes: list[str] = []
        self.first_hit_hashes: list[str] = []
        self.terminal_hashes: list[str] = []

    def _append(
        self,
        category: str,
        hashes: list[str],
        payload: Mapping[str, Any],
    ) -> str:
        ordinal = len(hashes)
        digest = self.write_once(
            self.root / f"{category}-{ordinal:04d}.json",
            {
                "schema": f"{CANONICAL_ALPHA_POSFIELD_SCHEMA_NAMESPACE}-receipt/v1",
                "instruction_id": CANONICAL_ALPHA_POSFIELD_INSTRUCTION_ID,
                "actuator": self.arm.value,
                "category": category,
                "ordinal": ordinal,
                **dict(payload),
            },
        )
        hashes.append(digest)
        return digest

    def field(self, payload: Mapping[str, Any]) -> str:
        return self._append("field", self.field_hashes, payload)

    def solver(self, payload: Mapping[str, Any]) -> str:
        return self._append("solver", self.solver_hashes, payload)

    def trial(self, payload: Mapping[str, Any]) -> str:
        return self._append("trial", self.trial_hashes, payload)

    def transition(self, payload: Mapping[str, Any]) -> str:
        return self._append("transition", self.transition_hashes, payload)

    def accepted(self, payload: Mapping[str, Any]) -> str:
        return self._append("accepted", self.accepted_hashes, payload)

    def first_hit(self, payload: Mapping[str, Any]) -> str:
        return self._append("first-hit", self.first_hit_hashes, payload)

    def terminal(self, payload: Mapping[str, Any]) -> str:
        return self._append("terminal", self.terminal_hashes, payload)

    def links(self) -> dict[str, list[str]]:
        return {
            "field": list(self.field_hashes),
            "solver": list(self.solver_hashes),
            "trial": list(self.trial_hashes),
            "transition": list(self.transition_hashes),
            "accepted": list(self.accepted_hashes),
            "first_hit": list(self.first_hit_hashes),
            "terminal": list(self.terminal_hashes),
        }


def canonical_alpha_posfield_source_contract() -> dict[str, Any]:
    payload = {
        "instruction_id": CANONICAL_ALPHA_POSFIELD_INSTRUCTION_ID,
        "panel": [item.value for item in AlphaActuator],
        "primary_claim_scope": (
            "ORDERED_RESIDUAL_REFRESH_PLUS_REMAINING_WRITER_DIVISOR_PLUS_"
            "PROPOSAL_STATE_SEMANTICS_WITHIN_ALPHA_WB"
        ),
        "canonical_ordered_claim_boundary": (
            "PROPOSAL_CONTROL_NOT_UPSTREAM_ONE_SHOT_OR_SAME_STATE_JOINT_FIELD"
        ),
        "fixed_grid": {"k": 8, "h": COMMON_COLD_H, "tau_target": 1.0},
        "routing": "TARGET_NEW_NLL_NEUTRAL_QCQP",
        "cold_native_or_direct_z_access_count": 0,
        "history_append_count": 0,
        "persistent_commit_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _rng_identity() -> str:
    payload: dict[str, Any] = {"cpu": tensor_sha256(torch.get_rng_state())}
    if torch.cuda.is_available():
        payload["cuda"] = [
            tensor_sha256(torch.cuda.get_rng_state(index))
            for index in range(torch.cuda.device_count())
        ]
    return canonical_hash(payload)


class CanonicalOrderedTargetFieldOverlay:
    """Differentiate canonical remaining-residual left factors at one anchor.

    Keys, projected directions and ordered current activations are frozen at
    the field anchor.  For layer ``l`` with remaining-writer divisor ``d_l``,
    the differentiable left factor is ``R_l(z_k) + (z-z_k)/d_l``.
    """

    DEFINITION = "R_l(z_k)+(z-z_k)/remaining_writer_count_l"
    STEP_SEMANTICS = "STEP_INDEPENDENT_FINITE_UNIT_ORDERED_ALPHA_FIELD_LOOKAHEAD"

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

    def _left(self, index: int, device: torch.device) -> torch.Tensor:
        layer = self.field.layers[index]
        anchor = self.field.target_state.to(device=device, dtype=torch.float32)
        residual = layer.residual.to(device=device, dtype=torch.float32)
        if (
            self.target_state_variable.shape != anchor.shape
            or residual.shape != anchor.shape
            or layer.residual_divisor <= 0
        ):
            raise ODEBFContractError("canonical ordered target overlay shape differs")
        return residual + (
            self.target_state_variable.to(device=device, dtype=torch.float32) - anchor
        ) / float(layer.residual_divisor)

    def _hook(self, index: int):
        layer = self.field.layers[index]

        def apply(_module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            if (
                not inputs
                or not isinstance(inputs[0], torch.Tensor)
                or not isinstance(output, torch.Tensor)
            ):
                raise ODEBFContractError("canonical ordered target Linear contract differs")
            hidden = inputs[0]
            right = layer.q.to(device=hidden.device, dtype=torch.float32)
            left = self._left(index, hidden.device)
            if (
                hidden.shape[-1] != right.shape[0]
                or right.shape[1] != left.shape[1]
                or output.shape[-1] != left.shape[0]
            ):
                raise ODEBFContractError("canonical ordered target geometry differs")
            perturbation = (hidden.float() @ right) @ left.T
            return (
                output.float()
                + self.applied_coefficients[index] * perturbation
            ).to(dtype=output.dtype)

        return apply

    def __enter__(self) -> "CanonicalOrderedTargetFieldOverlay":
        layers = tuple(int(item.layer) for item in self.field.layers)
        divisors = tuple(int(item.residual_divisor) for item in self.field.layers)
        if (
            layers != COMMON_COLD_LAYER_ORDER
            or divisors != tuple(range(len(layers), 0, -1))
            or self.applied_coefficients.shape != (len(layers),)
            or self.target_state_variable.shape != self.field.target_state.shape
            or not self.target_state_variable.requires_grad
        ):
            raise ODEBFContractError("canonical ordered target inventory differs")
        try:
            for index, layer in enumerate(self.field.layers):
                if (
                    layer.residual_definition
                    != CANONICAL_ORDERED_ALPHA_REMAINING_RESIDUAL_V1
                ):
                    raise ODEBFContractError("canonical ordered target policy differs")
                module = self.model.get_submodule(
                    layer.weight_name[: -len(".weight")]
                )
                if type(module) is not torch.nn.Linear:
                    raise ODEBFContractError("canonical ordered target is not Linear")
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


def write_aware_canonical_ordered_target_velocity(
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
    controller_terminal_residual: torch.Tensor,
    metric: CommonColdScaleMetric,
    ledger: ComputeLedger,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute F_z with canonical ordered proposal factors and exact h*v."""

    device = next(model.parameters()).device
    velocity = tuple(float(item) for item in velocity_coefficients)
    if len(velocity) != len(COMMON_COLD_LAYER_ORDER) or any(
        not math.isfinite(item) for item in velocity
    ):
        raise ODEBFContractError("canonical ordered target coefficient differs")
    applied = torch.tensor(
        tuple(COMMON_COLD_H * item for item in velocity),
        device=device,
        dtype=torch.float32,
    )
    scientific_applied = tuple(COMMON_COLD_H * item for item in velocity)
    flat = field.target_state.to(device=device, dtype=torch.float32).contiguous().view(-1)
    flat = flat.detach().requires_grad_(True)
    target = flat.view(field.target_state.shape)
    terminal_residual = controller_terminal_residual.to(
        device=device, dtype=torch.float32
    )
    if terminal_residual.shape != field.target_state.shape:
        raise ODEBFContractError("canonical ordered controller residual differs")
    residual_variable = terminal_residual + target - field.target_state.to(
        device=device, dtype=torch.float32
    )
    before_rng = _rng_identity()
    terminal_overlay = RequestResidualActivationOverlay(
        model, target_layer_name, residual_variable, lookup_positions
    )
    with _virtual_context(model, cumulative_factors_by_weight):
        with CanonicalOrderedTargetFieldOverlay(model, field, applied, target):
            with terminal_overlay:
                result = evaluate_routing_objective(
                    model,
                    tokenizer,
                    requests,
                    objective=RoutingObjective.TARGET_NEW_NLL,
                    contexts=contexts,
                    gradient_input=flat,
                )
    if _rng_identity() != before_rng:
        raise ODEBFStateError("canonical ordered target velocity changed RNG")
    if result.input_gradient is None or result.backward_count != len(requests):
        raise ODEBFContractError("canonical ordered target gradient differs")
    gradient = result.input_gradient.view(field.target_state.shape).to(
        device="cpu", dtype=torch.float64
    )
    target_velocity, scale_receipt = metric.velocity(gradient)
    ledger.increment("backward", result.backward_count)
    ledger.increment("target_backward", result.backward_count)
    payload = {
        "schema": "ode-edit-canonical-ordered-write-aware-target-velocity/v1",
        "field_sha256": field.identity_sha256,
        "field_target_state_sha256": tensor_sha256(field.target_state),
        "controller_terminal_residual_sha256": tensor_sha256(
            controller_terminal_residual
        ),
        "layer_residual_sha256": [
            tensor_sha256(item.residual) for item in field.layers
        ],
        "layer_residual_divisor": [item.residual_divisor for item in field.layers],
        "velocity_coefficient": list(velocity),
        "target_probe_scientific_coefficient": list(scientific_applied),
        "target_probe_effective_coefficient": [float(item) for item in applied.cpu()],
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
        "target_backward_count": result.backward_count,
        "processed_token_count": result.processed_token_count,
        "terminal_additive_overlay": terminal_overlay.raw_free_payload(),
        "target_overlay_definition": CanonicalOrderedTargetFieldOverlay.DEFINITION,
        "target_velocity_step_semantics": CanonicalOrderedTargetFieldOverlay.STEP_SEMANTICS,
        "candidate_coupled": False,
        "native_or_direct_z_access_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return target_velocity, payload


__all__ = [
    "AlphaActuator",
    "CANONICAL_ALPHA_POSFIELD_INSTRUCTION_ID",
    "CANONICAL_ALPHA_POSFIELD_SCHEMA_NAMESPACE",
    "CanonicalAlphaPostfreezeCapture",
    "CanonicalAlphaReceiptRecorder",
    "CanonicalOrderedTargetFieldOverlay",
    "canonical_alpha_posfield_source_contract",
    "write_aware_canonical_ordered_target_velocity",
]
