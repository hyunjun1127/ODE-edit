"""Arbitrary-B dynamic field algebra for P1R23.

This module is deliberately thin: it reuses the accepted P1 Woodbury,
covariance, five-dimensional problem, and strength-preserving solvers while
removing only their legacy joint-B10 shape assumption.  All model work enters
through :mod:`scalable_batched_model`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .accounting import ComputeLedger
from .common_cold_coordinate import (
    COMMON_COLD_EPS,
    SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1,
)
from .contracts import ODEBFContractError, canonical_hash
from .fixed_e8_runtime import (
    FixedE8ProblemReceipt,
    _build_fixed_e8_problem,
)
from .functional import tensor_sha256
from .p1_backend import (
    P1DynamicField,
    PinnedCovarianceRegistry,
    SignedProgressReceipt,
    build_p1_dynamic_field,
)
from .p1_controller import AcceptedLayerContribution, P1ControllerLock
from .scalable_batched_runtime import scalable_ordered_request_digest
from .scalable_batched_model import (
    ScalableObjectivePlan,
    ScalableObjectiveResult,
    evaluate_scalable_target_new_objective,
)
from .scalable_batched_runtime import (
    P1R23_CONTEXTS_PER_REQUEST,
    P1R23_H,
    P1R23_LAYER_ORDER,
)


@dataclass(frozen=True, slots=True)
class ScalableSharedTerminalResidual:
    residual: torch.Tensor
    terminal_current_z: torch.Tensor
    request_order_sha256: str
    global_batch_size: int
    policy_identity: str = SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1

    def __post_init__(self) -> None:
        if (
            isinstance(self.global_batch_size, bool)
            or not isinstance(self.global_batch_size, int)
            or self.global_batch_size <= 0
            or len(self.request_order_sha256) != 64
            or self.policy_identity
            != SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1
            or self.residual.ndim != 2
            or self.residual.shape
            != self.terminal_current_z.shape
            or self.residual.shape[1] != self.global_batch_size
            or self.residual.dtype != torch.float32
            or self.terminal_current_z.dtype != torch.float32
            or self.residual.device.type != "cpu"
            or self.terminal_current_z.device.type != "cpu"
            or not self.residual.is_contiguous()
            or not self.terminal_current_z.is_contiguous()
            or not torch.isfinite(self.residual).all()
            or not torch.isfinite(self.terminal_current_z).all()
        ):
            raise ODEBFContractError("P1R23 shared terminal residual differs")

    @property
    def residual_sha256(self) -> str:
        return tensor_sha256(self.residual)

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "policy_identity": self.policy_identity,
            "global_batch_size": self.global_batch_size,
            "request_order_sha256": self.request_order_sha256,
            "residual_shape": list(self.residual.shape),
            "residual_sha256": self.residual_sha256,
            "terminal_current_z_sha256": tensor_sha256(self.terminal_current_z),
            "backend_recapture_count": 0,
            "remaining_layer_divisor": 1,
        }


def scalable_terminal_residual(
    target_state: torch.Tensor,
    terminal_current_z: torch.Tensor,
    request_order_sha256: str,
) -> ScalableSharedTerminalResidual:
    target = target_state.detach().to(device="cpu", dtype=torch.float32).contiguous()
    current = terminal_current_z.detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    if target.shape != current.shape or target.ndim != 2:
        raise ODEBFContractError("P1R23 target/terminal geometry differs")
    return ScalableSharedTerminalResidual(
        (target - current).contiguous(),
        current,
        request_order_sha256,
        target.shape[1],
    )


@dataclass(frozen=True, slots=True)
class ScalableBatchGlobalMetric:
    shared_speed: float
    z0_sha256: str
    request_order_sha256: str
    global_batch_size: int
    per_request_z0_norm: tuple[float, ...]
    identity_sha256: str

    @classmethod
    def from_z0(
        cls, z0: torch.Tensor, request_order_sha256: str
    ) -> "ScalableBatchGlobalMetric":
        if (
            not isinstance(z0, torch.Tensor)
            or z0.ndim != 2
            or z0.shape[1] <= 0
            or not torch.isfinite(z0).all()
            or len(request_order_sha256) != 64
        ):
            raise ODEBFContractError("P1R23 BG metric entry differs")
        value = z0.detach().to(device="cpu", dtype=torch.float64)
        norms = torch.linalg.vector_norm(value, dim=0)
        speed = float(torch.median(norms))
        if not math.isfinite(speed) or speed <= COMMON_COLD_EPS:
            raise ODEBFContractError("P1R23 BG metric is degenerate")
        payload = {
            "schema": "ode-edit-s05-p1r23-batch-global-scale/v1",
            "shared_speed": speed,
            "shared_speed_definition": "median_i_l2_norm_z0_i",
            "global_batch_size": z0.shape[1],
            "z0_sha256": tensor_sha256(z0),
            "request_order_sha256": request_order_sha256,
            "per_request_z0_norm": [float(item) for item in norms],
            "native_or_direct_z_access_count": 0,
        }
        return cls(
            speed,
            payload["z0_sha256"],
            request_order_sha256,
            z0.shape[1],
            tuple(float(item) for item in norms),
            canonical_hash(payload),
        )

    def velocity(
        self, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if (
            gradient.ndim != 2
            or gradient.shape[1] != self.global_batch_size
            or not torch.isfinite(gradient).all()
        ):
            raise ODEBFContractError("P1R23 target gradient geometry differs")
        value = gradient.detach().to(device="cpu", dtype=torch.float64).contiguous()
        frobenius = float(torch.linalg.vector_norm(value))
        if not math.isfinite(frobenius):
            raise ODEBFContractError("P1R23 target gradient is non-finite")
        if frobenius <= COMMON_COLD_EPS:
            velocity64 = torch.zeros_like(value)
        else:
            velocity64 = (
                -self.shared_speed
                * math.sqrt(self.global_batch_size)
                * value
                / frobenius
            )
        velocity = velocity64.to(dtype=torch.float32).contiguous()
        payload = {
            "schema": "ode-edit-s05-p1r23-batch-global-target-velocity/v1",
            "metric_sha256": self.identity_sha256,
            "global_batch_size": self.global_batch_size,
            "gradient_sha256": tensor_sha256(value),
            "gradient_frobenius_norm": frobenius,
            "gradient_norm_by_request": [
                float(item) for item in torch.linalg.vector_norm(value, dim=0)
            ],
            "velocity_sha256": tensor_sha256(velocity),
            "velocity_frobenius_norm": float(torch.linalg.vector_norm(velocity64)),
            "allocation": "BATCH_GLOBAL_MATCHED_TOTAL_FROBENIUS_SPEED",
            "zero_gradient": frobenius <= COMMON_COLD_EPS,
            "clip_retry_rescue_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return velocity, payload

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p1r23-batch-global-scale/v1",
            "shared_speed": self.shared_speed,
            "shared_speed_definition": "median_i_l2_norm_z0_i",
            "global_batch_size": self.global_batch_size,
            "z0_sha256": self.z0_sha256,
            "request_order_sha256": self.request_order_sha256,
            "per_request_z0_norm": list(self.per_request_z0_norm),
            "identity_sha256": self.identity_sha256,
        }


def target_update_from_existing_gradient(
    current_target: torch.Tensor,
    gradient: torch.Tensor,
    metric: ScalableBatchGlobalMetric,
) -> tuple[torch.Tensor, torch.Tensor, float, dict[str, Any]]:
    velocity, scale_receipt = metric.velocity(gradient)
    displacement64 = float(P1R23_H) * velocity.to(dtype=torch.float64)
    alpha_req = float(
        -torch.sum(gradient.to(dtype=torch.float64) * displacement64)
    )
    if not math.isfinite(alpha_req) or alpha_req < 0.0:
        raise ODEBFContractError("P1R23 target requested strength differs")
    target_next = (
        current_target.detach().to(device="cpu", dtype=torch.float32)
        + float(P1R23_H) * velocity
    ).contiguous()
    payload = {
        "schema": "ode-edit-s05-p1r23-one-gradient-target-update/v1",
        "current_target_sha256": tensor_sha256(current_target),
        "gradient_sha256": tensor_sha256(gradient),
        "velocity_sha256": tensor_sha256(velocity),
        "target_next_sha256": tensor_sha256(target_next),
        "alpha_req": alpha_req,
        "alpha_req_definition": "-dot(global_mean_target_gradient,h*F_z)",
        "h": float(P1R23_H),
        "h_application_count": 1,
        "target_backward_reuse_count": 1,
        "additional_target_graph_count": 0,
        "scale": scale_receipt,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return target_next, velocity, alpha_req, payload


def build_scalable_dynamic_field(
    model: Any,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    *,
    target_state: torch.Tensor,
    current_terminal: torch.Tensor,
    captured_keys_by_layer: Mapping[int, torch.Tensor],
    accepted_waypoint: int,
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    residual_tolerance: float,
    ledger: ComputeLedger,
) -> P1DynamicField:
    request_count = len(requests)
    layers = tuple(int(item) for item in hparams.layers)
    if layers != P1R23_LAYER_ORDER or set(captured_keys_by_layer) != set(layers):
        raise ODEBFContractError("P1R23 field layer inventory differs")
    order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    residual = scalable_terminal_residual(target_state, current_terminal, order)
    empty_solve = {
        layer: torch.empty(
            (captured_keys_by_layer[layer].shape[0], 0), dtype=torch.float32
        )
        for layer in layers
    }
    field = build_p1_dynamic_field(
        model,
        tokenizer,
        requests,
        hparams,
        projector,
        contexts,
        target_state=target_state,
        accepted_waypoint=accepted_waypoint,
        cumulative_factors_by_weight={},
        history_solve_keys_by_layer=empty_solve,
        history_risk_keys_by_layer=empty_solve,
        covariance_registry=covariance_registry,
        projector_sha256=projector_sha256,
        residual_tolerance=residual_tolerance,
        ledger=ledger,
        residual_policy=SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1,
        allow_zero_capacity=False,
        shared_terminal_residual=residual,
        captured_keys_by_layer=captured_keys_by_layer,
        allow_inner_empty_cache=False,
        expected_batch_size=request_count,
    )
    if field.model_forward_count != 0 or field.current_z.shape[1] != request_count:
        raise ODEBFContractError("P1R23 field performed duplicate capture")
    return field


def scalable_physical_signed_progress(
    model: torch.nn.Module,
    plan: ScalableObjectivePlan,
    field: P1DynamicField,
) -> tuple[SignedProgressReceipt, ScalableObjectiveResult]:
    device = next(model.parameters()).device
    coefficients = torch.zeros(
        len(P1R23_LAYER_ORDER),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    observed = evaluate_scalable_target_new_objective(
        model,
        plan,
        coefficient_layers=field.layers,
        coefficients=coefficients,
    )
    if observed.coefficient_gradient is None:
        raise ODEBFContractError("P1R23 physical slope gradient is absent")
    gradient = observed.coefficient_gradient.detach().to(
        device="cpu", dtype=torch.float64
    )
    progress = tuple(float(item) for item in -gradient)
    if len(progress) != len(P1R23_LAYER_ORDER) or any(
        not math.isfinite(item) for item in progress
    ):
        raise ODEBFContractError("P1R23 physical slope differs")
    receipt = SignedProgressReceipt(
        field.identity_sha256,
        progress,
        tuple(
            layer
            for layer, value in zip(P1R23_LAYER_ORDER, progress, strict=True)
            if value <= 0.0
        ),
        tensor_sha256(gradient),
        observed.model_forward_count,
        observed.processed_token_count,
        True,
        "TARGET_NEW_NLL",
        0,
        observed.identity_sha256,
        plan.context_sha256,
        P1R23_CONTEXTS_PER_REQUEST,
        (1, 5),
        observed.backward_count,
        observed.loss,
    )
    return receipt, observed


def build_scalable_routing_problem(
    field: P1DynamicField,
    signed: SignedProgressReceipt,
    *,
    accepted_by_layer: Mapping[int, Sequence[AcceptedLayerContribution]],
    committed_load_by_layer: Mapping[int, float],
    lock: P1ControllerLock,
) -> FixedE8ProblemReceipt:
    return _build_fixed_e8_problem(
        field,
        signed,
        accepted_by_layer=accepted_by_layer,
        committed_load_by_layer=committed_load_by_layer,
        lock=lock,
        current_history_action_by_layer={
            item.layer: item.history_action for item in field.layers
        },
    )


__all__ = [
    "ScalableBatchGlobalMetric",
    "ScalableSharedTerminalResidual",
    "build_scalable_dynamic_field",
    "build_scalable_routing_problem",
    "scalable_physical_signed_progress",
    "scalable_terminal_residual",
    "target_update_from_existing_gradient",
]
