"""P1R27 joint z-CLF, W-only CLF, tracking, and semantic-first routing.

This module contains only reusable controller mathematics and raw-free
receipts.  Model execution, sample selection, transaction materialization,
and terminal evaluation remain owned by the existing runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .fixed_e8_soft_routing import FixedE8Arm, FIXED_E8_SOLVER_FTOL, FIXED_E8_SOLVER_MAXITER
from .functional import tensor_sha256
from .p1r24_atomic_strength import P1R24AliasTargetLock, P1R24KLResult
from .p1_backend import P1DynamicField, SignedProgressReceipt
from .progress_simplex_routing import (
    SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE,
    SIMPLEX_ENERGY_RELATIVE_TOLERANCE,
    SIMPLEX_PRIMAL_TOLERANCE,
    SIMPLEX_XI_TIE_TOLERANCE,
)
from .routing import RoutingProblem
from .scalable_batched_field import ScalableBatchGlobalMetric, ScalableRobustSharedMetric
from .scalable_batched_model import ScalableObjectiveResult
from .scalable_batched_model import (
    ScalableObjectivePlan,
    evaluate_scalable_target_new_objective,
)
from .scalable_batched_runtime import P1R23_CONTEXTS_PER_REQUEST, P1R23_LAYER_ORDER


P1R27_INSTRUCTION_ID = "ODEEDIT-S05-P1R27-JOINT-WCLF-ZTRACKING-SEMANTIC-PRESERVING-BF-V1"
P1R27_METHOD_ID = "P1R27-JOINT-WCLF-ZTRACKING-SEMANTIC-FIRST-BF-V1"
P1R27_K = 8
P1R27_H = 1.0 / P1R27_K
P1R27_EPSILON = 1.0e-12
P1R27_TELEMETRY_THRESHOLD = 0.05


@dataclass(frozen=True, slots=True)
class PhysicalSlopeBasis:
    semantic_slopes: tuple[float, ...]
    lag_slopes: tuple[float, ...]
    context_sha256: str
    objective: ScalableObjectiveResult
    identity_sha256: str

    def combined(
        self,
        field: P1DynamicField,
        *,
        semantic_scale: float,
        lag_scale: float,
    ) -> SignedProgressReceipt:
        values = tuple(
            semantic_scale * semantic + lag_scale * lag
            for semantic, lag in zip(
                self.semantic_slopes, self.lag_slopes, strict=True
            )
        )
        if any(not math.isfinite(item) for item in values):
            raise ODEBFContractError("P1R27 combined physical slope is nonfinite")
        raw_gradient = torch.tensor(
            [-item for item in values], dtype=torch.float64
        ).contiguous()
        return SignedProgressReceipt(
            field.identity_sha256,
            values,
            tuple(
                layer
                for layer, value in zip(P1R23_LAYER_ORDER, values, strict=True)
                if value <= 0.0
            ),
            tensor_sha256(raw_gradient),
            self.objective.model_forward_count,
            self.objective.processed_token_count,
            True,
            "TARGET_NEW_NLL",
            0,
            self.objective.identity_sha256,
            self.context_sha256,
            P1R23_CONTEXTS_PER_REQUEST,
            (1, 5),
            self.objective.backward_count,
            self.objective.loss,
        )

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p1r27-physical-slope-basis/v1",
            "semantic_slopes": list(self.semantic_slopes),
            "lag_slopes": list(self.lag_slopes),
            "context_sha256": self.context_sha256,
            "model_forward_count": self.objective.model_forward_count,
            "backward_count": self.objective.backward_count,
            "additional_slope_backward_count": 0,
            "objective_sha256": self.objective.identity_sha256,
            "identity_sha256": self.identity_sha256,
        }


def p1r27_physical_slope_basis(
    model: torch.nn.Module,
    plan: ScalableObjectivePlan,
    semantic_field: P1DynamicField,
    lag_field: P1DynamicField | None,
) -> PhysicalSlopeBasis:
    """Obtain semantic and lag field slopes in one shared model VJP."""

    fields = (semantic_field,) if lag_field is None else (semantic_field, lag_field)
    layers = tuple(layer for field in fields for layer in field.layers)
    device = next(model.parameters()).device
    coefficients = torch.zeros(
        len(layers), device=device, dtype=torch.float32, requires_grad=True
    )
    observed = evaluate_scalable_target_new_objective(
        model,
        plan,
        coefficient_layers=layers,
        coefficients=coefficients,
    )
    if observed.coefficient_gradient is None:
        raise ODEBFContractError("P1R27 physical slope basis gradient is absent")
    gradient = observed.coefficient_gradient.detach().to(
        device="cpu", dtype=torch.float64
    )
    width = len(P1R23_LAYER_ORDER)
    semantic = tuple(float(item) for item in -gradient[:width])
    lag = (
        tuple(0.0 for _ in range(width))
        if lag_field is None
        else tuple(float(item) for item in -gradient[width:])
    )
    if len(semantic) != width or len(lag) != width:
        raise ODEBFContractError("P1R27 slope basis geometry differs")
    payload = {
        "semantic_field_sha256": semantic_field.identity_sha256,
        "lag_field_sha256": None if lag_field is None else lag_field.identity_sha256,
        "semantic_slopes": list(semantic),
        "lag_slopes": list(lag),
        "objective_sha256": observed.identity_sha256,
        "context_sha256": plan.context_sha256,
        "shared_vjp_count": 1,
        "additional_slope_backward_count": 0,
    }
    return PhysicalSlopeBasis(
        semantic, lag, plan.context_sha256, observed, _identity(payload)
    )


def _identity(payload: Mapping[str, Any]) -> str:
    return canonical_hash(dict(payload))


@dataclass(frozen=True, slots=True)
class ZeroSeekingZEntry:
    allocation: str
    shared_speed: float
    request_count: int
    entry_gradient_norms: tuple[float, ...]
    entry_frobenius_norm: float
    metric_sha256: str
    gradient_sha256: str
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


def build_zero_seeking_z_entry(
    metric: ScalableBatchGlobalMetric | ScalableRobustSharedMetric,
    gradient: torch.Tensor,
    *,
    allocation: str,
) -> ZeroSeekingZEntry:
    value = gradient.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if (
        allocation not in ("RS", "BG")
        or value.ndim != 2
        or value.shape[1] != metric.global_batch_size
        or not torch.isfinite(value).all()
        or not math.isfinite(metric.shared_speed)
        or metric.shared_speed <= P1R27_EPSILON
    ):
        raise ODEBFContractError("P1R27 z-CLF entry differs")
    norms = tuple(float(item) for item in torch.linalg.vector_norm(value, dim=0))
    frobenius = float(torch.linalg.vector_norm(value))
    payload = {
        "schema": "ode-edit-s05-p1r27-z-clf-entry/v1",
        "instruction_id": P1R27_INSTRUCTION_ID,
        "allocation": allocation,
        "shared_speed": float(metric.shared_speed),
        "shared_speed_definition": "median_i_l2_norm_z_base_i",
        "request_count": value.shape[1],
        "entry_gradient_norms": list(norms),
        "entry_frobenius_norm": frobenius,
        "metric_sha256": metric.identity_sha256,
        "gradient_sha256": tensor_sha256(value),
        "metric": "EUCLIDEAN",
        "entry_denominator_frozen": True,
    }
    return ZeroSeekingZEntry(
        allocation,
        float(metric.shared_speed),
        value.shape[1],
        norms,
        frobenius,
        metric.identity_sha256,
        payload["gradient_sha256"],
        _identity(payload),
    )


@dataclass(frozen=True, slots=True)
class ZeroSeekingZStep:
    nominal_displacement: torch.Tensor
    nll_gradient: torch.Tensor
    nominal_target: torch.Tensor
    receipt: Mapping[str, Any]


def _decay_gradient(
    current: torch.Tensor,
    origin: torch.Tensor,
    lock: P1R24AliasTargetLock,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = current - origin
    origin_norm = torch.linalg.vector_norm(origin, dim=0)
    delta_norm = torch.linalg.vector_norm(delta, dim=0)
    if bool(torch.any(origin_norm <= 0.0)):
        raise ODEBFContractError("P1R27 target decay origin is degenerate")
    values = lock.decay_factor * delta_norm / torch.square(origin_norm)
    gradient = torch.zeros_like(delta)
    nonzero = delta_norm > 0.0
    gradient[:, nonzero] = (
        lock.decay_factor
        * delta[:, nonzero]
        / delta_norm[nonzero].unsqueeze(0)
        / torch.square(origin_norm[nonzero]).unsqueeze(0)
        / current.shape[1]
    )
    return values, gradient


def _scaled_descent(
    gradient: torch.Tensor,
    entry: ZeroSeekingZEntry,
) -> tuple[torch.Tensor, tuple[float, ...], float]:
    current_norms = torch.linalg.vector_norm(gradient, dim=0)
    if entry.allocation == "RS":
        denominator = torch.maximum(
            torch.tensor(entry.entry_gradient_norms, dtype=torch.float64),
            current_norms,
        ).clamp(min=P1R27_EPSILON)
        velocity = -entry.shared_speed * gradient / denominator.unsqueeze(0)
    else:
        current_frobenius = float(torch.linalg.vector_norm(gradient))
        denominator_value = max(
            entry.entry_frobenius_norm, current_frobenius, P1R27_EPSILON
        )
        velocity = (
            -entry.shared_speed
            * math.sqrt(entry.request_count)
            * gradient
            / denominator_value
        )
    return (
        velocity,
        tuple(float(item) for item in current_norms),
        float(torch.linalg.vector_norm(gradient)),
    )


def _scaled_regularizer_descent(
    regularizer_gradient: torch.Tensor,
    nll_gradient: torch.Tensor,
    entry: ZeroSeekingZEntry,
) -> torch.Tensor:
    current_nll_norms = torch.linalg.vector_norm(nll_gradient, dim=0)
    if entry.allocation == "RS":
        denominator = torch.maximum(
            torch.tensor(entry.entry_gradient_norms, dtype=torch.float64),
            current_nll_norms,
        ).clamp(min=P1R27_EPSILON)
        return (
            -entry.shared_speed
            * regularizer_gradient
            / denominator.unsqueeze(0)
        )
    denominator = max(
        entry.entry_frobenius_norm,
        float(torch.linalg.vector_norm(nll_gradient)),
        P1R27_EPSILON,
    )
    return (
        -entry.shared_speed
        * math.sqrt(entry.request_count)
        * regularizer_gradient
        / denominator
    )


def build_zero_seeking_z_step(
    current_target: torch.Tensor,
    target_origin: torch.Tensor,
    nll: ScalableObjectiveResult,
    kl: P1R24KLResult,
    entry: ZeroSeekingZEntry,
    lock: P1R24AliasTargetLock,
    *,
    step_index: int,
) -> ZeroSeekingZStep:
    if nll.target_gradient is None or kl.gradient is None:
        raise ODEBFContractError("P1R27 target gradients are absent")
    if step_index < 0 or step_index >= P1R27_K:
        raise ODEBFContractError("P1R27 target step differs")
    current = current_target.detach().to(device="cpu", dtype=torch.float64)
    origin = target_origin.detach().to(device="cpu", dtype=torch.float64)
    g = nll.target_gradient.detach().to(device="cpu", dtype=torch.float64)
    g_kl = kl.gradient.detach().to(device="cpu", dtype=torch.float64)
    if current.shape != origin.shape or current.shape != g.shape or g.shape != g_kl.shape:
        raise ODEBFContractError("P1R27 target geometry differs")
    if not all(torch.isfinite(item).all() for item in (current, origin, g, g_kl)):
        raise ODEBFContractError("P1R27 target input is nonfinite")

    decay_values, decay_gradient = _decay_gradient(current, origin, lock)
    u_sem, current_norms, current_frobenius = _scaled_descent(g, entry)
    regularizer_gradient = lock.kl_factor * g_kl + decay_gradient
    u_reg = _scaled_regularizer_descent(regularizer_gradient, g, entry)

    if entry.allocation == "RS":
        projected = u_reg.clone()
        chi: list[float] = []
        removed: list[float] = []
        for index in range(entry.request_count):
            dot = float(torch.dot(g[:, index], projected[:, index]))
            amount = max(dot, 0.0)
            projected[:, index] -= (
                amount / (float(torch.dot(g[:, index], g[:, index])) + P1R27_EPSILON)
            ) * g[:, index]
            sem_norm = float(torch.linalg.vector_norm(u_sem[:, index]))
            reg_norm = float(torch.linalg.vector_norm(projected[:, index]))
            factor = min(1.0, sem_norm / (reg_norm + P1R27_EPSILON))
            chi.append(factor)
            removed.append(amount)
            projected[:, index] *= factor
    else:
        dot = float(torch.sum(g * u_reg))
        amount = max(dot, 0.0)
        projected = u_reg - (
            amount / (float(torch.sum(g * g)) + P1R27_EPSILON)
        ) * g
        factor = min(
            1.0,
            float(torch.linalg.vector_norm(u_sem))
            / (float(torch.linalg.vector_norm(projected)) + P1R27_EPSILON),
        )
        projected *= factor
        chi = [factor]
        removed = [amount]
    velocity = u_sem + projected
    displacement = P1R27_H * velocity

    candidate = current + displacement
    origin_norms = torch.linalg.vector_norm(origin, dim=0)
    clamp_ratio: list[float] = []
    for index in range(entry.request_count):
        relative = candidate[:, index] - origin[:, index]
        norm = float(torch.linalg.vector_norm(relative))
        maximum = lock.clamp_factor * float(origin_norms[index])
        ratio = 1.0 if norm <= maximum or norm == 0.0 else maximum / norm
        candidate[:, index] = origin[:, index] + ratio * relative
        clamp_ratio.append(ratio)
    displacement = (candidate - current).contiguous()
    target = candidate.to(dtype=torch.float32).contiguous()
    payload = {
        "schema": "ode-edit-s05-p1r27-zero-seeking-z-clf-step/v1",
        "instruction_id": P1R27_INSTRUCTION_ID,
        "method_id": P1R27_METHOD_ID,
        "step_index": step_index,
        "h": P1R27_H,
        "h_application_count": 1,
        "allocation": entry.allocation,
        "entry_sha256": entry.identity_sha256,
        "target_new_nll": float(nll.loss),
        "kl_loss": float(kl.loss),
        "decay_loss_mean": float(torch.mean(decay_values)),
        "nll_gradient_sha256": tensor_sha256(g),
        "regularizer_gradient_sha256": tensor_sha256(regularizer_gradient),
        "current_gradient_norms": list(current_norms),
        "current_gradient_frobenius_norm": current_frobenius,
        "semantic_velocity_sha256": tensor_sha256(u_sem),
        "regularizer_projected_velocity_sha256": tensor_sha256(projected),
        "regularizer_increasing_component_removed": removed,
        "regularizer_gate_chi": chi,
        "nominal_displacement_sha256": tensor_sha256(displacement),
        "nominal_target_sha256": tensor_sha256(target),
        "native_clamp_ratio": clamp_ratio,
        "native_clamp_application_count": 1,
        "zero_seeking": True,
        "absolute_0p05_telemetry": [
            max(float(item) - P1R27_TELEMETRY_THRESHOLD, 0.0)
            for item in nll.per_request_values
        ],
        "absolute_0p05_decision_influence_count": 0,
        "remaining_steps_decision_influence_count": 0,
        "combined_loss_freeze_decision_influence_count": 0,
        "retry_backtracking_reject_count": 0,
    }
    payload["identity_sha256"] = _identity(payload)
    return ZeroSeekingZStep(
        displacement.to(dtype=torch.float32),
        g.to(dtype=torch.float32),
        target,
        payload,
    )


class WriterCLFStatus(str, Enum):
    ACTIVE = "ACTIVE"
    SEMANTIC_STALL = "WRITER_CLF_SEMANTIC_STALL"
    NUMERICAL_ACTION_METRIC_INVALID = "NUMERICAL_ACTION_METRIC_INVALID"


def _validate_action_metric(
    problem: RoutingProblem,
    active_mask: np.ndarray,
) -> np.ndarray:
    metric = np.asarray(problem.trust_metric, dtype=np.float64) / (P1R27_H**2)
    active = np.asarray(active_mask, dtype=bool)
    if (
        metric.ndim != 2
        or metric.shape[0] != metric.shape[1]
        or active.shape != (metric.shape[0],)
        or not np.all(np.isfinite(metric))
        or not np.allclose(metric, metric.T, rtol=0.0, atol=1.0e-14)
    ):
        raise ODEBFStateError(WriterCLFStatus.NUMERICAL_ACTION_METRIC_INVALID.value)
    if active.any():
        try:
            np.linalg.cholesky(metric[np.ix_(active, active)])
        except np.linalg.LinAlgError as exc:
            raise ODEBFStateError(
                WriterCLFStatus.NUMERICAL_ACTION_METRIC_INVALID.value
            ) from exc
    return metric


def _energy(value: np.ndarray, metric: np.ndarray) -> float:
    return float(value @ metric @ value)


@dataclass(frozen=True, slots=True)
class WriterCLFEntry:
    active0: tuple[bool, ...]
    c_ref0: tuple[float, ...]
    energy_reference: float
    energy_limit: float
    eta_w: float
    v_w0: float
    r_w0: float
    kappa_w: float
    beta_h: float
    status: WriterCLFStatus
    trust_metric_sha256: str
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(frozen=True, slots=True)
class SemanticWriterAction:
    coefficients: tuple[float, ...]
    r_w: float
    active_mask: tuple[bool, ...]
    energy: float
    projection_residual: float
    status: WriterCLFStatus
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


def _project_semantic_action(
    slopes: np.ndarray,
    metric: np.ndarray,
    *,
    eta_w: float,
    energy_limit: float,
) -> SemanticWriterAction:
    active = np.flatnonzero(slopes > 0.0)
    full = np.zeros_like(slopes)
    if active.size == 0 or eta_w == 0.0:
        payload = {
            "coefficients": full.tolist(),
            "r_w": 0.0,
            "active_mask": [bool(item > 0.0) for item in slopes],
            "status": WriterCLFStatus.SEMANTIC_STALL.value,
        }
        return SemanticWriterAction(
            tuple(full), 0.0, tuple(item > 0.0 for item in slopes), 0.0, 0.0,
            WriterCLFStatus.SEMANTIC_STALL, _identity(payload)
        )
    local_metric = metric[np.ix_(active, active)]
    local_slopes = slopes[active]
    try:
        direction = np.linalg.solve(local_metric, local_slopes)
    except np.linalg.LinAlgError as exc:
        raise ODEBFStateError(WriterCLFStatus.NUMERICAL_ACTION_METRIC_INVALID.value) from exc
    desired = eta_w * direction

    def objective(value: np.ndarray) -> float:
        delta = value - desired
        return float(0.5 * delta @ local_metric @ delta)

    def jacobian(value: np.ndarray) -> np.ndarray:
        return local_metric @ (value - desired)

    constraints = ({
        "type": "ineq",
        "fun": lambda value: float(energy_limit - value @ local_metric @ value),
        "jac": lambda value: -2.0 * (local_metric @ value),
    },)
    initial = desired.copy()
    desired_energy = _energy(initial, local_metric)
    if desired_energy > energy_limit and desired_energy > 0.0:
        initial *= math.sqrt(energy_limit / desired_energy)
    result = minimize(
        objective,
        initial,
        jac=jacobian,
        method="SLSQP",
        bounds=tuple((0.0, None) for _ in active),
        constraints=constraints,
        options={"disp": False, "ftol": FIXED_E8_SOLVER_FTOL, "maxiter": FIXED_E8_SOLVER_MAXITER},
    )
    selected = np.asarray(result.x, dtype=np.float64)
    full[active] = selected
    energy = _energy(full, metric)
    negative = max(float(-selected.min(initial=0.0)), 0.0)
    violation = max(energy - energy_limit, 0.0)
    if (
        not result.success
        or not np.all(np.isfinite(full))
        or negative > SIMPLEX_PRIMAL_TOLERANCE
        or violation > SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE
    ):
        raise ODEBFStateError("P1R27 semantic action certificate failed")
    progress = max(float(slopes @ full), 0.0)
    status = (
        WriterCLFStatus.SEMANTIC_STALL
        if progress <= P1R27_EPSILON
        else WriterCLFStatus.ACTIVE
    )
    payload = {
        "coefficients": full.tolist(),
        "r_w": progress,
        "active_mask": [bool(item > 0.0) for item in slopes],
        "energy": energy,
        "energy_limit": energy_limit,
        "projection_residual": objective(selected),
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "status": status.value,
    }
    return SemanticWriterAction(
        tuple(float(item) for item in full), progress,
        tuple(bool(item > 0.0) for item in slopes), energy,
        objective(selected), status, _identity(payload)
    )


def build_writer_clf_entry(
    problem: RoutingProblem,
    *,
    v_w0: float,
) -> tuple[WriterCLFEntry, SemanticWriterAction]:
    slopes = np.asarray(problem.signed_progress, dtype=np.float64)
    active = slopes > 0.0
    metric = _validate_action_metric(problem, active)
    c_ref = np.where(active, P1R27_H, 0.0)
    energy_reference = _energy(c_ref, metric)
    energy_limit = (
        energy_reference * (1.0 + SIMPLEX_ENERGY_RELATIVE_TOLERANCE)
        + SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE
    )
    if not active.any():
        eta = r_w0 = kappa = beta = 0.0
        status = WriterCLFStatus.SEMANTIC_STALL
        semantic = _project_semantic_action(
            slopes, metric, eta_w=0.0, energy_limit=energy_limit
        )
    else:
        local_metric = metric[np.ix_(active, active)]
        try:
            direction = np.linalg.solve(local_metric, slopes[active])
        except np.linalg.LinAlgError as exc:
            raise ODEBFStateError(WriterCLFStatus.NUMERICAL_ACTION_METRIC_INVALID.value) from exc
        reference_norm = math.sqrt(max(_energy(c_ref, metric), 0.0))
        direction_norm = math.sqrt(max(_energy(direction, local_metric), 0.0))
        eta = reference_norm / (direction_norm + P1R27_EPSILON)
        semantic = _project_semantic_action(
            slopes, metric, eta_w=eta, energy_limit=energy_limit
        )
        r_w0 = semantic.r_w
        if r_w0 <= P1R27_EPSILON:
            eta = r_w0 = kappa = beta = 0.0
            status = WriterCLFStatus.SEMANTIC_STALL
            semantic = _project_semantic_action(
                slopes, metric, eta_w=0.0, energy_limit=energy_limit
            )
        else:
            kappa = r_w0 / (P1R27_H * max(float(v_w0), P1R27_EPSILON))
            beta = -math.expm1(-kappa * P1R27_H)
            status = WriterCLFStatus.ACTIVE
    payload = {
        "schema": "ode-edit-s05-p1r27-w-clf-entry/v1",
        "instruction_id": P1R27_INSTRUCTION_ID,
        "active0": active.tolist(),
        "c_ref0": c_ref.tolist(),
        "energy_reference": energy_reference,
        "energy_limit": energy_limit,
        "eta_w": eta,
        "v_w0": float(v_w0),
        "r_w0": r_w0,
        "kappa_w": kappa,
        "beta_h": beta,
        "status": status.value,
        "trust_metric_sha256": canonical_hash(problem.trust_metric.tolist()),
        "coordinate_conversion": "G_num=trust_metric/h^2",
        "coordinate_identity_residual": abs(
            _energy(c_ref, metric)
            - _energy(c_ref / P1R27_H, np.asarray(problem.trust_metric, dtype=np.float64))
        ),
        "per_layer_upper_cap_count": 0,
        "p_h_capacity_influence_count": 0,
    }
    entry = WriterCLFEntry(
        tuple(bool(item) for item in active), tuple(float(item) for item in c_ref),
        energy_reference, energy_limit, eta, float(v_w0), r_w0, kappa, beta,
        status, payload["trust_metric_sha256"], _identity(payload)
    )
    return entry, semantic


def semantic_writer_action(
    problem: RoutingProblem,
    entry: WriterCLFEntry,
) -> SemanticWriterAction:
    slopes = np.asarray(problem.signed_progress, dtype=np.float64)
    metric = _validate_action_metric(problem, slopes > 0.0)
    return _project_semantic_action(
        slopes,
        metric,
        eta_w=entry.eta_w,
        energy_limit=entry.energy_limit,
    )


@dataclass(frozen=True, slots=True)
class BackpressureTracking:
    gamma_z: float
    r_z: float
    r_w: float
    beta_h: float
    selected_displacement: torch.Tensor
    desired_writer_displacement: torch.Tensor
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected_displacement"] = tensor_sha256(self.selected_displacement)
        payload["desired_writer_displacement"] = tensor_sha256(
            self.desired_writer_displacement
        )
        payload.update({
            "schema": "ode-edit-s05-p1r27-z-writer-backpressure/v1",
            "instruction_id": P1R27_INSTRUCTION_ID,
            "p_h_gamma_influence_count": 0,
            "remaining_steps_decision_influence_count": 0,
            "never_accelerates_z": self.gamma_z <= 1.0,
        })
        return payload


def apply_z_writer_backpressure(
    nominal_displacement: torch.Tensor,
    nll_gradient: torch.Tensor,
    lag: torch.Tensor,
    *,
    r_w: float,
    beta_h: float,
) -> BackpressureTracking:
    d0 = nominal_displacement.detach().to(device="cpu", dtype=torch.float64)
    g = nll_gradient.detach().to(device="cpu", dtype=torch.float64)
    e = lag.detach().to(device="cpu", dtype=torch.float64)
    if d0.shape != g.shape or d0.shape != e.shape:
        raise ODEBFContractError("P1R27 tracking geometry differs")
    r_z = max(float(-torch.sum(g * d0)), 0.0)
    gamma = min(1.0, float(r_w) / (r_z + P1R27_EPSILON))
    selected = (gamma * d0).contiguous()
    desired = (selected + float(beta_h) * e).contiguous()
    payload = {
        "schema": "ode-edit-s05-p1r27-z-writer-backpressure/v1",
        "gamma_z": gamma,
        "r_z": r_z,
        "r_w": float(r_w),
        "beta_h": float(beta_h),
        "nominal_displacement_sha256": tensor_sha256(d0),
        "selected_displacement_sha256": tensor_sha256(selected),
        "desired_writer_displacement_sha256": tensor_sha256(desired),
        "p_h_gamma_influence_count": 0,
        "remaining_steps_decision_influence_count": 0,
        "never_accelerates_z": gamma <= 1.0,
    }
    return BackpressureTracking(
        gamma, r_z, float(r_w), float(beta_h),
        selected.to(dtype=torch.float32), desired.to(dtype=torch.float32),
        _identity(payload),
    )


class P1R27RoutingStatus(str, Enum):
    JOINT_WRITE = "JOINT_WRITE"
    SOFT_NO_ROUTING_DOF = "SOFT_NO_ROUTING_DOF"
    WRITER_CLF_SEMANTIC_STALL = "WRITER_CLF_SEMANTIC_STALL"


@dataclass(frozen=True, slots=True)
class P1R27RoutingResult:
    arm: FixedE8Arm
    status: P1R27RoutingStatus
    requested_progress: float
    semantic_slack: float
    neutral_coefficients: tuple[float, ...]
    soft_coefficients: tuple[float, ...]
    coefficients: tuple[float, ...]
    velocity: tuple[float, ...]
    predicted_progress: float
    equality_residual: float
    selected_p: float
    selected_capacity: float
    selected_energy: float
    soft_decision_influence_count: int
    identity_sha256: str

    @property
    def alpha_req(self) -> float:
        return self.requested_progress

    @property
    def alpha_max(self) -> float:
        return self.requested_progress

    @property
    def alpha_apply(self) -> float:
        return self.predicted_progress

    @property
    def coverage(self) -> float:
        return 1.0 if self.requested_progress == 0.0 else self.predicted_progress / (
            self.requested_progress + P1R27_EPSILON
        )

    @property
    def applied_coefficient(self) -> tuple[float, ...]:
        return self.coefficients

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["arm"] = self.arm.value
        payload["status"] = self.status.value
        payload.update({
            "instruction_id": P1R27_INSTRUCTION_ID,
            "method_id": P1R27_METHOD_ID,
            "parameterization": "applied_c=h*v",
            "soft_neutral_requested_r_w_identical": True,
            "hard_p_h_budget_influence_count": 0,
            "functional_p_veto_count": 0,
            "retry_backtracking_reject_count": 0,
            "per_layer_upper_cap_count": 0,
        })
        return payload


def _maximize_progress(
    slopes: np.ndarray, metric: np.ndarray, energy_limit: float
) -> np.ndarray:
    active = np.flatnonzero(slopes > 0.0)
    full = np.zeros_like(slopes)
    if active.size == 0:
        return full
    local_metric = metric[np.ix_(active, active)]
    local_slopes = slopes[active]
    direction = np.linalg.solve(local_metric, local_slopes)
    direction = np.maximum(direction, 0.0)
    if not np.any(direction > 0.0):
        direction = np.ones(active.size, dtype=np.float64)
    direction *= math.sqrt(energy_limit / max(_energy(direction, local_metric), P1R27_EPSILON))
    result = minimize(
        lambda value: -float(local_slopes @ value),
        direction,
        jac=lambda value: -local_slopes,
        method="SLSQP",
        bounds=tuple((0.0, None) for _ in active),
        constraints=({
            "type": "ineq",
            "fun": lambda value: float(energy_limit - value @ local_metric @ value),
            "jac": lambda value: -2.0 * (local_metric @ value),
        },),
        options={"disp": False, "ftol": FIXED_E8_SOLVER_FTOL, "maxiter": FIXED_E8_SOLVER_MAXITER},
    )
    if not result.success:
        raise ODEBFStateError("P1R27 semantic stage1 certificate failed")
    full[active] = np.asarray(result.x, dtype=np.float64)
    return full


def solve_semantic_first_routing(
    problem: RoutingProblem,
    entry: WriterCLFEntry,
    *,
    arm: FixedE8Arm | str,
    requested_progress: float,
    semantic_reference: Sequence[float],
) -> P1R27RoutingResult:
    selected_arm = FixedE8Arm(arm)
    if selected_arm not in (FixedE8Arm.NEUTRAL, FixedE8Arm.SOFT):
        raise ODEBFContractError("P1R27 routing arm differs")
    slopes = np.asarray(problem.signed_progress, dtype=np.float64)
    metric = _validate_action_metric(problem, slopes > 0.0)
    reference = np.asarray(tuple(semantic_reference), dtype=np.float64)
    if reference.shape != slopes.shape or requested_progress < 0.0:
        raise ODEBFContractError("P1R27 routing geometry differs")
    active = np.flatnonzero(slopes > 0.0)
    zero = np.zeros_like(slopes)
    if requested_progress <= P1R27_EPSILON or active.size == 0:
        payload = {
            "status": P1R27RoutingStatus.WRITER_CLF_SEMANTIC_STALL.value,
            "requested_progress": float(requested_progress),
            "slopes": slopes.tolist(),
        }
        return P1R27RoutingResult(
            selected_arm, P1R27RoutingStatus.WRITER_CLF_SEMANTIC_STALL,
            float(requested_progress), float(requested_progress), tuple(zero), tuple(zero),
            tuple(zero), tuple(zero), 0.0, 0.0, 0.0, 0.0, 0.0, 0,
            _identity(payload),
        )

    maximum = _maximize_progress(slopes, metric, entry.energy_limit)
    maximum_progress = float(slopes @ maximum)
    floor = min(float(requested_progress), maximum_progress)
    slack = max(float(requested_progress) - floor, 0.0)
    local_metric = metric[np.ix_(active, active)]
    local_slopes = slopes[active]
    local_reference = np.maximum(reference[active], 0.0)

    def distance(value: np.ndarray) -> float:
        delta = value - local_reference
        return float(0.5 * delta @ local_metric @ delta)

    constraints = (
        {
            "type": "eq",
            "fun": lambda value: float(local_slopes @ value - floor),
            "jac": lambda value: local_slopes,
        },
        {
            "type": "ineq",
            "fun": lambda value: float(entry.energy_limit - value @ local_metric @ value),
            "jac": lambda value: -2.0 * (local_metric @ value),
        },
    )
    initial = maximum[active] * (floor / max(maximum_progress, P1R27_EPSILON))
    neutral_result = minimize(
        distance,
        initial,
        jac=lambda value: local_metric @ (value - local_reference),
        method="SLSQP",
        bounds=tuple((0.0, None) for _ in active),
        constraints=constraints,
        options={"disp": False, "ftol": FIXED_E8_SOLVER_FTOL, "maxiter": FIXED_E8_SOLVER_MAXITER},
    )
    if not neutral_result.success:
        raise ODEBFStateError("P1R27 semantic Neutral certificate failed")
    neutral = zero.copy()
    neutral[active] = np.asarray(neutral_result.x, dtype=np.float64)
    neutral_progress = float(slopes @ neutral)
    soft = neutral.copy()
    influence = 0
    status = P1R27RoutingStatus.JOINT_WRITE
    if selected_arm is FixedE8Arm.SOFT and active.size > 1:
        transform_h = 1.0 / P1R27_H

        def p_value(value: np.ndarray) -> float:
            full = zero.copy()
            full[active] = value
            return float(problem.pretrained.value(full * transform_h))

        def p_grad(value: np.ndarray) -> np.ndarray:
            full = zero.copy()
            full[active] = value
            velocity = full * transform_h
            gradient_v = 2.0 * problem.pretrained.linear + 2.0 * problem.pretrained.gram @ velocity
            return gradient_v[active] * transform_h

        stage_p = minimize(
            p_value,
            neutral[active],
            jac=p_grad,
            method="SLSQP",
            bounds=tuple((0.0, None) for _ in active),
            constraints=constraints,
            options={"disp": False, "ftol": FIXED_E8_SOLVER_FTOL, "maxiter": FIXED_E8_SOLVER_MAXITER},
        )
        if not stage_p.success:
            raise ODEBFStateError("P1R27 Structural-P certificate failed")
        p1 = np.asarray(stage_p.x, dtype=np.float64)
        p_star = p_value(p1)
        p_scale = max(float(np.trace(problem.pretrained.gram)), P1R27_EPSILON)
        p_tie = SIMPLEX_XI_TIE_TOLERANCE * p_scale
        capacity_metric = np.asarray(problem.capacity_metric, dtype=np.float64)

        def capacity(value: np.ndarray) -> float:
            full = zero.copy()
            full[active] = value / P1R27_H
            return float(0.5 * full @ capacity_metric @ full)

        def capacity_grad(value: np.ndarray) -> np.ndarray:
            full = zero.copy()
            full[active] = value / P1R27_H
            return (capacity_metric @ full)[active] / P1R27_H

        constraints2 = (*constraints, {
            "type": "ineq",
            "fun": lambda value: float(p_star + p_tie - p_value(value)),
            "jac": lambda value: -p_grad(value),
        })
        stage_capacity = minimize(
            capacity,
            p1,
            jac=capacity_grad,
            method="SLSQP",
            bounds=tuple((0.0, None) for _ in active),
            constraints=constraints2,
            options={"disp": False, "ftol": FIXED_E8_SOLVER_FTOL, "maxiter": FIXED_E8_SOLVER_MAXITER},
        )
        if not stage_capacity.success:
            raise ODEBFStateError("P1R27 capacity certificate failed")
        soft[active] = np.asarray(stage_capacity.x, dtype=np.float64)
        influence = int(
            np.linalg.norm(soft - neutral, ord=2) > SIMPLEX_PRIMAL_TOLERANCE
        )
        if not influence:
            status = P1R27RoutingStatus.SOFT_NO_ROUTING_DOF
    elif selected_arm is FixedE8Arm.SOFT:
        status = P1R27RoutingStatus.SOFT_NO_ROUTING_DOF

    chosen = neutral if selected_arm is FixedE8Arm.NEUTRAL else soft
    neutral_predicted = float(slopes @ neutral)
    predicted = float(slopes @ chosen)
    equality = abs(predicted - neutral_predicted)
    energy = _energy(chosen, metric)
    if (
        equality > SIMPLEX_PRIMAL_TOLERANCE
        or predicted + SIMPLEX_PRIMAL_TOLERANCE < neutral_predicted
        or energy > entry.energy_limit + SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE
        or not np.all(np.isfinite(chosen))
    ):
        raise ODEBFStateError("P1R27 BF predicted-strength contract violated")
    velocity = chosen / P1R27_H
    selected_p = float(problem.pretrained.value(velocity))
    selected_capacity = float(0.5 * velocity @ problem.capacity_metric @ velocity)
    payload = {
        "arm": selected_arm.value,
        "status": status.value,
        "requested_progress": float(requested_progress),
        "semantic_slack": slack,
        "neutral_coefficients": neutral.tolist(),
        "soft_coefficients": soft.tolist(),
        "coefficients": chosen.tolist(),
        "velocity": velocity.tolist(),
        "predicted_progress": predicted,
        "neutral_predicted_progress": neutral_predicted,
        "equality_residual": equality,
        "selected_p": selected_p,
        "selected_capacity": selected_capacity,
        "selected_energy": energy,
        "soft_decision_influence_count": influence,
    }
    return P1R27RoutingResult(
        selected_arm, status, float(requested_progress), slack,
        tuple(float(item) for item in neutral), tuple(float(item) for item in soft),
        tuple(float(item) for item in chosen), tuple(float(item) for item in velocity),
        predicted, equality, selected_p, selected_capacity, energy, influence,
        _identity(payload),
    )


@dataclass(slots=True)
class AcceptedRealizationLedger:
    entries: list[dict[str, Any]]

    def __init__(self) -> None:
        self.entries = []

    def complete(
        self,
        *,
        source_v_w: float,
        next_v_w: float,
        requested_progress: float,
        predicted_progress: float,
        transition_index: int,
    ) -> dict[str, Any]:
        actual = float(source_v_w - next_v_w)
        payload = {
            "transition_index": int(transition_index),
            "source_v_w": float(source_v_w),
            "next_v_w": float(next_v_w),
            "requested_progress": float(requested_progress),
            "predicted_progress": float(predicted_progress),
            "actual_progress": actual,
            "non_saturated_realization_ratio": actual
            / (float(requested_progress) + P1R27_EPSILON),
            "linearization_error": actual - float(predicted_progress),
            "candidate_trial_count": 0,
            "rollback_retry_count": 0,
        }
        payload["identity_sha256"] = _identity(payload)
        self.entries.append(payload)
        return payload

    def raw_free_payload(self) -> dict[str, Any]:
        payload = {
            "schema": "ode-edit-s05-p1r27-accepted-realization-ledger/v1",
            "entry_count": len(self.entries),
            "entry_sha256": [item["identity_sha256"] for item in self.entries],
            "candidate_trial_count": 0,
            "rollback_retry_count": 0,
        }
        payload["identity_sha256"] = _identity(payload)
        return payload


__all__ = [
    "AcceptedRealizationLedger",
    "BackpressureTracking",
    "P1R27_H",
    "P1R27_INSTRUCTION_ID",
    "P1R27_K",
    "P1R27_METHOD_ID",
    "PhysicalSlopeBasis",
    "P1R27RoutingResult",
    "P1R27RoutingStatus",
    "SemanticWriterAction",
    "WriterCLFEntry",
    "WriterCLFStatus",
    "ZeroSeekingZEntry",
    "ZeroSeekingZStep",
    "apply_z_writer_backpressure",
    "build_writer_clf_entry",
    "build_zero_seeking_z_entry",
    "build_zero_seeking_z_step",
    "semantic_writer_action",
    "p1r27_physical_slope_basis",
    "solve_semantic_first_routing",
]
