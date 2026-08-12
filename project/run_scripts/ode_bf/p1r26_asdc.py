"""Request-wise Absolute Semantic Deficit Controller for P1R26.

The controller is deliberately separated from launch/sample-loop plumbing.  It
accepts the already-computed P1R24 nominal target geometry and full-six
target-new gradients, performs no model calls, and returns the selected target
displacement plus a raw-free certificate consumed by the physical writer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .p1r24_atomic_strength import (
    P1R24AliasTargetLock,
    P1R24KLResult,
    P1R24_H,
    P1R24_K,
    P1R24_NUMERICAL_EPSILON,
)
from .scalable_batched_field import ScalableBatchGlobalMetric, ScalableRobustSharedMetric
from .scalable_batched_model import ScalableObjectiveResult


P1R26_INSTRUCTION_ID = "ODEEDIT-S05-P1R26-ABSOLUTE-SEMANTIC-DEFICIT-CONTROLLER-V1"
P1R26_METHOD_ID = "P1R26-ASDC-STRONG-Z-MATCHED-STRENGTH-ATOMIC-P-V1"
P1R26_EPSILON_Z = 0.05
P1R26_PROJECTION_EPSILON = P1R24_NUMERICAL_EPSILON
P1R26_CERTIFICATE_TOLERANCE = P1R24_NUMERICAL_EPSILON


@dataclass(frozen=True, slots=True)
class ASDCProjectionResult:
    displacement: torch.Tensor
    pre_clamp_displacement: torch.Tensor
    nominal_progress: tuple[float, ...]
    required_progress: tuple[float, ...]
    achieved_progress: tuple[float, ...]
    clamp_limited: tuple[bool, ...]
    observation_mask: tuple[bool, ...]
    receipt: Mapping[str, Any]


def project_absolute_semantic_deficit(
    nominal_displacement: torch.Tensor,
    current_target: torch.Tensor,
    target_origin: torch.Tensor,
    target_new_gradient_by_request: torch.Tensor,
    target_new_nll_by_request: Sequence[float],
    clamp_radius_by_request: Sequence[float],
    *,
    remaining_steps: int,
    observation_mask: Sequence[bool],
) -> ASDCProjectionResult:
    """Apply the exact request-wise Euclidean ASDC projection and clamp.

    ``target_new_gradient_by_request`` contains derivatives of each request's
    own six-context mean loss, not the global-B mean gradient used by the
    writer.  No tensor in this routine requires a model graph.
    """

    shape = nominal_displacement.shape
    request_count = shape[1] if nominal_displacement.ndim == 2 else 0
    if (
        request_count == 0
        or current_target.shape != shape
        or target_origin.shape != shape
        or target_new_gradient_by_request.shape != shape
        or len(target_new_nll_by_request) != request_count
        or len(clamp_radius_by_request) != request_count
        or len(observation_mask) != request_count
        or remaining_steps <= 0
    ):
        raise ODEBFContractError("P1R26 ASDC input geometry differs")
    values = tuple(float(item) for item in target_new_nll_by_request)
    radii = tuple(float(item) for item in clamp_radius_by_request)
    if (
        not torch.isfinite(nominal_displacement).all()
        or not torch.isfinite(current_target).all()
        or not torch.isfinite(target_origin).all()
        or not torch.isfinite(target_new_gradient_by_request).all()
        or any(not math.isfinite(item) or item < 0.0 for item in values)
        or any(not math.isfinite(item) or item <= 0.0 for item in radii)
    ):
        raise ODEBFContractError("P1R26 ASDC input is nonfinite or degenerate")

    nominal = nominal_displacement.detach().to(device="cpu", dtype=torch.float64)
    current = current_target.detach().to(device="cpu", dtype=torch.float64)
    origin = target_origin.detach().to(device="cpu", dtype=torch.float64)
    gradients = target_new_gradient_by_request.detach().to(
        device="cpu", dtype=torch.float64
    )
    selected = nominal.clone()
    nominal_progress: list[float] = []
    raw_nominal_progress: list[float] = []
    deficits: list[float] = []
    deadlines: list[float] = []
    required: list[float] = []
    correction_norms: list[float] = []
    gradient_norms: list[float] = []
    for index in range(request_count):
        gradient = gradients[:, index]
        raw = float(-torch.dot(gradient, nominal[:, index]))
        baseline = max(raw, 0.0)
        deficit = max(values[index] - P1R26_EPSILON_Z, 0.0)
        deadline = deficit / remaining_steps
        demand = max(baseline, deadline)
        norm_squared = float(torch.dot(gradient, gradient))
        shortfall = max(demand - raw, 0.0)
        if shortfall > P1R26_CERTIFICATE_TOLERANCE and norm_squared == 0.0:
            raise ODEBFContractError("ASDC_NO_DESCENT_DIRECTION")
        correction = (
            shortfall / (norm_squared + P1R26_PROJECTION_EPSILON)
        ) * gradient
        selected[:, index] = nominal[:, index] - correction
        raw_nominal_progress.append(raw)
        nominal_progress.append(baseline)
        deficits.append(deficit)
        deadlines.append(deadline)
        required.append(demand)
        correction_norms.append(float(torch.linalg.vector_norm(correction)))
        gradient_norms.append(math.sqrt(norm_squared))

    pre_clamp = selected.clone()
    pre_clamp_achieved = tuple(
        float(-torch.dot(gradients[:, index], pre_clamp[:, index]))
        for index in range(request_count)
    )
    clamp_ratios: list[float] = []
    for index in range(request_count):
        proposed = current[:, index] + selected[:, index] - origin[:, index]
        norm = float(torch.linalg.vector_norm(proposed))
        radius = radii[index]
        ratio = 1.0 if norm <= radius or norm == 0.0 else radius / norm
        selected[:, index] = origin[:, index] + ratio * proposed - current[:, index]
        clamp_ratios.append(ratio)
    achieved = tuple(
        float(-torch.dot(gradients[:, index], selected[:, index]))
        for index in range(request_count)
    )
    understrength = tuple(
        achieved[index] + P1R26_CERTIFICATE_TOLERANCE < required[index]
        for index in range(request_count)
    )
    limited = tuple(
        understrength[index] and clamp_ratios[index] < 1.0
        for index in range(request_count)
    )
    silent_understrength = tuple(
        understrength[index] and not limited[index]
        for index in range(request_count)
    )
    if any(silent_understrength):
        raise ODEBFContractError("P1R26 ASDC silently under-strength")

    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r26-asdc-projection/v1",
        "instruction_id": P1R26_INSTRUCTION_ID,
        "method_id": P1R26_METHOD_ID,
        "request_count": request_count,
        "epsilon_z": P1R26_EPSILON_Z,
        "remaining_steps": remaining_steps,
        "target_new_nll_by_request": list(values),
        "semantic_deficit_by_request": deficits,
        "deadline_progress_by_request": deadlines,
        "raw_nominal_progress_by_request": raw_nominal_progress,
        "nominal_progress_by_request": nominal_progress,
        "required_progress_by_request": required,
        "pre_clamp_achieved_progress_by_request": list(pre_clamp_achieved),
        "post_clamp_achieved_progress_by_request": list(achieved),
        "semantic_constraint_clamp_limited": list(limited),
        "clamp_limited_count": sum(limited),
        "combined_freeze_observation_mask": list(observation_mask),
        "combined_freeze_decision_influence_count": 0,
        "gradient_norm_by_request": gradient_norms,
        "nominal_displacement_norm_by_request": [
            float(torch.linalg.vector_norm(nominal[:, index]))
            for index in range(request_count)
        ],
        "selected_displacement_norm_by_request": [
            float(torch.linalg.vector_norm(selected[:, index]))
            for index in range(request_count)
        ],
        "correction_norm_by_request": correction_norms,
        "clamp_ratio_by_request": clamp_ratios,
        "nominal_displacement_sha256": tensor_sha256(nominal),
        "pre_clamp_displacement_sha256": tensor_sha256(pre_clamp),
        "selected_displacement_sha256": tensor_sha256(selected),
        "target_new_gradient_by_request_sha256": tensor_sha256(gradients),
        "additional_model_forward_count": 0,
        "additional_backward_count": 0,
        "additional_materialization_count": 0,
        "candidate_trial_count": 0,
        "retry_count": 0,
        "silent_understrength_count": 0,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    return ASDCProjectionResult(
        selected.to(dtype=torch.float32).contiguous(),
        pre_clamp.to(dtype=torch.float32).contiguous(),
        tuple(nominal_progress),
        tuple(required),
        achieved,
        limited,
        tuple(bool(item) for item in observation_mask),
        receipt,
    )


@dataclass(frozen=True, slots=True)
class P1R26TargetStep:
    target_next: torch.Tensor
    target_displacement: torch.Tensor
    required_displacement: torch.Tensor
    write_velocity: torch.Tensor
    nll_gradient: torch.Tensor
    combined_gradient: torch.Tensor
    rho_write_signed: float
    rho_write: float
    alpha_target_signed: float
    frozen_mask: tuple[bool, ...]
    receipt: Mapping[str, Any]


def p1r26_target_step(
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    z0: torch.Tensor,
    nll: ScalableObjectiveResult,
    kl: P1R24KLResult,
    metric: ScalableBatchGlobalMetric | ScalableRobustSharedMetric,
    lock: P1R24AliasTargetLock,
    *,
    step_index: int,
    frozen_mask: Sequence[bool],
) -> P1R26TargetStep:
    """Build P1R24 full-speed d0, then apply ASDC and writer lag once."""

    if nll.target_gradient is None or kl.gradient is None:
        raise ODEBFContractError("P1R26 target gradients are absent")
    if step_index < 0 or step_index >= P1R24_K:
        raise ODEBFContractError("P1R26 target step index differs")
    shape = current_target.shape
    if current_terminal.shape != shape or z0.shape != shape or shape[1] != len(frozen_mask):
        raise ODEBFContractError("P1R26 target state geometry differs")
    current64 = current_target.detach().to(device="cpu", dtype=torch.float64)
    z064 = z0.detach().to(device="cpu", dtype=torch.float64)
    delta_from_origin = current64 - z064
    z0_norm = torch.linalg.vector_norm(z064, dim=0)
    delta_norm = torch.linalg.vector_norm(delta_from_origin, dim=0)
    if bool(torch.any(z0_norm <= 0.0)):
        raise ODEBFContractError("P1R26 target decay origin is degenerate")
    decay_values = lock.decay_factor * delta_norm / torch.square(z0_norm)
    decay_gradient = torch.zeros_like(delta_from_origin)
    nonzero = delta_norm > 0.0
    decay_gradient[:, nonzero] = (
        lock.decay_factor
        * delta_from_origin[:, nonzero]
        / delta_norm[nonzero].unsqueeze(0)
        / torch.square(z0_norm[nonzero]).unsqueeze(0)
        / shape[1]
    )
    nll_gradient = nll.target_gradient.detach().to(device="cpu", dtype=torch.float64)
    kl_gradient = kl.gradient.detach().to(device="cpu", dtype=torch.float64)
    combined = nll_gradient + lock.kl_factor * kl_gradient + decay_gradient
    combined_loss = tuple(
        float(nll.per_request_values[index] + lock.kl_factor * kl.per_request_values[index] + decay_values[index])
        for index in range(shape[1])
    )
    observation = tuple(
        bool(frozen_mask[index] or combined_loss[index] < P1R26_EPSILON_Z)
        for index in range(shape[1])
    )
    velocity, metric_receipt = metric.velocity(combined.to(dtype=torch.float32))
    nominal_candidate = current64 + P1R24_H * velocity.to(dtype=torch.float64)
    nominal_clamp_ratio: list[float] = []
    clamp_radius = tuple(lock.clamp_factor * float(item) for item in z0_norm)
    for index in range(shape[1]):
        proposed = nominal_candidate[:, index] - z064[:, index]
        norm = float(torch.linalg.vector_norm(proposed))
        radius = clamp_radius[index]
        ratio = 1.0 if norm <= radius or norm == 0.0 else radius / norm
        nominal_candidate[:, index] = z064[:, index] + ratio * proposed
        nominal_clamp_ratio.append(ratio)
    nominal_displacement = (nominal_candidate - current64).contiguous()
    request_gradient = (nll_gradient * shape[1]).contiguous()
    remaining = P1R24_K - step_index
    projected = project_absolute_semantic_deficit(
        nominal_displacement,
        current64,
        z064,
        request_gradient,
        nll.per_request_values,
        clamp_radius,
        remaining_steps=remaining,
        observation_mask=observation,
    )
    displacement = projected.displacement.to(dtype=torch.float64)
    target_next = (current64 + displacement).to(dtype=torch.float32).contiguous()
    lag = (current64 - current_terminal.detach().to(device="cpu", dtype=torch.float64)).contiguous()
    required = (displacement + lag / remaining).contiguous()
    required_model = required.to(dtype=torch.float32).contiguous()
    write_velocity = (required_model / P1R24_H).contiguous()
    identity_residual = float(torch.max(torch.abs(P1R24_H * write_velocity - required_model)))
    if identity_residual > P1R26_CERTIFICATE_TOLERANCE:
        raise ODEBFContractError("P1R26 remaining-step identity differs")
    alpha_target_signed = float(-torch.sum(nll_gradient * displacement))
    rho_signed = float(-torch.sum(nll_gradient * required))
    rho = max(rho_signed, 0.0)
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r26-target-write-coordinate/v1",
        "instruction_id": P1R26_INSTRUCTION_ID,
        "method_id": P1R26_METHOD_ID,
        "k": step_index,
        "tau_before": step_index / P1R24_K,
        "tau_after": (step_index + 1) / P1R24_K,
        "remaining_steps": remaining,
        "target_new_loss": nll.loss,
        "kl_loss": kl.loss,
        "decay_loss_mean": float(torch.mean(decay_values)),
        "combined_loss_by_request": list(combined_loss),
        "combined_freeze_observation_mask": list(observation),
        "combined_freeze_decision_influence_count": 0,
        "nominal_clamp_ratio": nominal_clamp_ratio,
        "asdc": projected.receipt,
        "target_displacement_sha256": tensor_sha256(projected.displacement),
        "write_lag_sha256": tensor_sha256(lag),
        "required_displacement_sha256": tensor_sha256(required_model),
        "write_velocity_sha256": tensor_sha256(write_velocity),
        "target_to_writer_selected_displacement_sha256": tensor_sha256(projected.displacement),
        "identity_max_abs_residual": identity_residual,
        "alpha_target_signed": alpha_target_signed,
        "rho_write_signed": rho_signed,
        "rho_write": rho,
        "field_coordinate": "ACTIVATION_VELOCITY",
        "physical_h_application_count": 1,
        "second_remaining_division_count": 0,
        "combined_gradient_sha256": tensor_sha256(combined),
        "target_new_gradient_sha256": tensor_sha256(nll_gradient),
        "request_target_new_gradient_sha256": tensor_sha256(request_gradient),
        "metric": metric_receipt,
        "p1r25_scalar_pacing_access_count": 0,
        "selected_target_scale": 1.0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return P1R26TargetStep(
        target_next,
        projected.displacement,
        required_model,
        write_velocity,
        nll_gradient.to(dtype=torch.float32),
        combined.to(dtype=torch.float32),
        rho_signed,
        rho,
        alpha_target_signed,
        observation,
        payload,
    )


__all__ = [
    "ASDCProjectionResult",
    "P1R26_CERTIFICATE_TOLERANCE",
    "P1R26_EPSILON_Z",
    "P1R26_INSTRUCTION_ID",
    "P1R26_METHOD_ID",
    "P1R26TargetStep",
    "p1r26_target_step",
    "project_absolute_semantic_deficit",
]
