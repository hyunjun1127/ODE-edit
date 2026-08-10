"""P1R15 per-request target and physical simplex-transport routing.

This is an additive child of the executed P1R14 implementation.  The target
coordinate is the frozen per-request ``ColdTargetMetric`` and the physical
coordinate remains the unscaled five-layer coefficient ``theta``.  The sole
Euler application is ``theta = h * v``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from scipy import __version__ as SCIPY_VERSION
from scipy.optimize import Bounds, NonlinearConstraint, minimize, nnls

from .contracts import BATCH_SIZE, ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .integrated_physical_writer import (
    PHYSICAL_WRITER_EPSILON_P,
    PHYSICAL_WRITER_GRID_COUNT,
    PHYSICAL_WRITER_H,
    PHYSICAL_WRITER_LAYER_ORDER,
)
from .integrated_physical_writer_solver import (
    PHYSICAL_SOLVER_FALLBACK_BACKEND,
    PHYSICAL_SOLVER_FALLBACK_OPTIONS,
    PHYSICAL_SOLVER_FTOL,
    PHYSICAL_SOLVER_KKT_TOLERANCE,
    PHYSICAL_SOLVER_PRIMARY_BACKEND,
    PHYSICAL_SOLVER_PRIMARY_OPTIONS,
    PHYSICAL_SOLVER_PRIMAL_TOLERANCE,
    PhysicalWriterConstraint,
    PhysicalWriterSolverCertificate,
    physical_writer_technical_constraints,
    solve_certified_physical_qp,
)
from .routing import RoutingProblem


P1R15_INSTRUCTION_ID = (
    "ODEEDIT-S05-ODE-BF-PERREQUEST-TARGET-SIMPLEX-TRANSPORT-P1R15-V1"
)
P1R15_AMENDMENT_ID = f"{P1R15_INSTRUCTION_ID}-A1"
P1R15_METHOD_ID = "FIXED_E8_PERREQUEST_GDUAL_SIMPLEX_TRANSPORT_P_TIEBREAK_V1"
P1R15_SCHEMA = "ode-edit-s05-p1r15-perrequest-simplex-transport/v1"

P1R15_EPS_BASE = 1.0e-12
P1R15_EPS_LOSS = 1.0e-12
P1R15_EPS_DUAL = 1.0e-12
P1R15_EPS_ALPHA = 1.0e-12
P1R15_EPS_SPEED = 1.0e-8
P1R15_EPS_P = PHYSICAL_WRITER_EPSILON_P
P1R15_G_PATH_TOLERANCE = 1.0e-8


class TargetStepStatus(str, Enum):
    TARGET_STEP = "TARGET_STEP"
    TARGET_GOAL_MET = "TARGET_GOAL_MET"
    TARGET_NO_DESCENT_DIRECTION = "TARGET_NO_DESCENT_DIRECTION"


class JointFloorComponent(str, Enum):
    NONE = "NONE"
    EDIT_NO_DIRECTION = "EDIT_NO_DIRECTION"
    TRANSPORT_NO_DIRECTION = "TRANSPORT_NO_DIRECTION"
    JOINT_FLOOR_INFEASIBLE = "JOINT_FLOOR_INFEASIBLE"
    NUMERICAL_CERTIFICATE_FAILED = "NUMERICAL_CERTIFICATE_FAILED"


class SimplexStepStatus(str, Enum):
    JOINT_WRITE = "JOINT_WRITE"
    ZERO_WRITE_GOAL_MET = "ZERO_WRITE_GOAL_MET"
    NO_JOINT_PHYSICAL_TRANSPORT_DIRECTION = (
        "NO_JOINT_PHYSICAL_TRANSPORT_DIRECTION"
    )


@dataclass(frozen=True, slots=True)
class PerRequestTargetStep:
    status: TargetStepStatus
    step_index: int
    target_state_sha256: str
    desired_target_sha256: str
    z_base_sha256: str
    loss_by_request: tuple[float, ...]
    base_norm_by_request: tuple[float, ...]
    gradient_norm_by_request: tuple[float, ...]
    dual_norm_by_request: tuple[float, ...]
    rho_by_request: tuple[float, ...]
    unit_g_norm_by_request: tuple[float, ...]
    step_g_path_by_request: tuple[float, ...]
    cumulative_g_path_before: tuple[float, ...]
    cumulative_g_path_after: tuple[float, ...]
    predicted_reduction_by_request: tuple[float, ...]
    goal_met_mask: tuple[bool, ...]
    no_descent_mask: tuple[bool, ...]
    target_clock_advance_count: int
    target_clock_advance_eligibility_count: int
    h_application_count: int
    old_target_access_count: int
    heldout_access_count: int
    native_or_direct_z_access_count: int
    desired_target: torch.Tensor
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("desired_target")
        payload["status"] = self.status.value
        return payload


def build_per_request_gdual_target_step(
    *,
    target_state: torch.Tensor,
    z_base: torch.Tensor,
    loss_by_request: Sequence[float] | torch.Tensor,
    gradient_by_request: torch.Tensor,
    step_index: int,
    cumulative_g_path_before: Sequence[float],
) -> PerRequestTargetStep:
    """Build the exact A1 G-dual Dynamic target step in FP64/FP32."""

    if step_index < 0 or step_index >= PHYSICAL_WRITER_GRID_COUNT:
        raise ODEBFContractError("P1R15 target step index differs")
    if (
        not isinstance(target_state, torch.Tensor)
        or not isinstance(z_base, torch.Tensor)
        or not isinstance(gradient_by_request, torch.Tensor)
        or target_state.ndim != 2
        or target_state.shape[1] != BATCH_SIZE
        or z_base.shape != target_state.shape
        or gradient_by_request.shape != target_state.shape
    ):
        raise ODEBFContractError("P1R15 target geometry differs")
    target64 = target_state.detach().to(device="cpu", dtype=torch.float64)
    base64 = z_base.detach().to(device="cpu", dtype=torch.float64)
    gradient64 = gradient_by_request.detach().to(
        device="cpu", dtype=torch.float64
    )
    losses = torch.as_tensor(
        tuple(float(item) for item in loss_by_request), dtype=torch.float64
    )
    cumulative_before = torch.as_tensor(
        tuple(float(item) for item in cumulative_g_path_before),
        dtype=torch.float64,
    )
    if (
        losses.shape != (BATCH_SIZE,)
        or cumulative_before.shape != (BATCH_SIZE,)
        or not bool(torch.isfinite(target64).all())
        or not bool(torch.isfinite(base64).all())
        or not bool(torch.isfinite(gradient64).all())
        or not bool(torch.isfinite(losses).all())
        or not bool(torch.isfinite(cumulative_before).all())
        or bool(torch.any(losses < 0.0))
        or bool(torch.any(cumulative_before < 0.0))
    ):
        raise ODEBFContractError("P1R15 target value is malformed")
    base_norm = torch.linalg.vector_norm(base64, dim=0)
    if not bool(torch.isfinite(base_norm).all()) or bool(
        torch.any(base_norm <= P1R15_EPS_BASE)
    ):
        raise ODEBFContractError("P1R15 target base metric is degenerate")
    gradient_norm = torch.linalg.vector_norm(gradient64, dim=0)
    dual_norm = base_norm * gradient_norm
    if not bool(torch.isfinite(dual_norm).all()):
        raise ODEBFContractError("P1R15 target dual norm is non-finite")

    goal_met = losses <= P1R15_EPS_LOSS
    no_descent = (~goal_met) & (dual_norm <= P1R15_EPS_DUAL)
    unit = torch.zeros_like(gradient64)
    rho = torch.zeros(BATCH_SIZE, dtype=torch.float64)
    active = (~goal_met) & (~no_descent)
    remaining = PHYSICAL_WRITER_GRID_COUNT - step_index
    for index in range(BATCH_SIZE):
        if not bool(active[index]):
            continue
        # G_i^-1 = b_i^2 I and d_i = b_i ||g_i||.
        unit[:, index] = (
            -(base_norm[index] ** 2) * gradient64[:, index]
            / dual_norm[index]
        )
        rho[index] = min(
            1.0,
            float(
                losses[index]
                / (PHYSICAL_WRITER_H * remaining * dual_norm[index])
            ),
        )
    unit_g_norm = torch.linalg.vector_norm(unit, dim=0) / base_norm
    if bool(torch.any(torch.abs(unit_g_norm[active] - 1.0) > 1.0e-10)):
        raise ODEBFContractError("P1R15 target G-unit certificate differs")
    if bool(torch.any(unit_g_norm[~active] != 0.0)):
        raise ODEBFContractError("P1R15 inactive target unit differs")

    desired64 = target64 + PHYSICAL_WRITER_H * rho.unsqueeze(0) * unit
    desired = desired64.to(dtype=torch.float32).contiguous()
    # The certificate is authoritative in FP64; the model-facing state is FP32.
    step_path = PHYSICAL_WRITER_H * rho
    cumulative_after = cumulative_before + step_path
    if (
        bool(torch.any(step_path > PHYSICAL_WRITER_H + P1R15_G_PATH_TOLERANCE))
        or bool(torch.any(cumulative_after > 1.0 + P1R15_G_PATH_TOLERANCE))
    ):
        raise ODEBFContractError("P1R15 target G-path trust differs")
    predicted = PHYSICAL_WRITER_H * rho * dual_norm
    status = (
        TargetStepStatus.TARGET_NO_DESCENT_DIRECTION
        if bool(torch.any(no_descent))
        else (
            TargetStepStatus.TARGET_GOAL_MET
            if bool(torch.all(goal_met))
            else TargetStepStatus.TARGET_STEP
        )
    )
    eligibility = (
        0 if status is TargetStepStatus.TARGET_NO_DESCENT_DIRECTION else 1
    )
    payload = {
        "schema": f"{P1R15_SCHEMA}-target-step",
        "status": status.value,
        "step_index": step_index,
        "target_state_sha256": tensor_sha256(target_state),
        "desired_target_sha256": tensor_sha256(desired),
        "z_base_sha256": tensor_sha256(z_base),
        "loss_by_request": [float(item) for item in losses],
        "base_norm_by_request": [float(item) for item in base_norm],
        "gradient_norm_by_request": [float(item) for item in gradient_norm],
        "dual_norm_by_request": [float(item) for item in dual_norm],
        "rho_by_request": [float(item) for item in rho],
        "unit_g_norm_by_request": [float(item) for item in unit_g_norm],
        "step_g_path_by_request": [float(item) for item in step_path],
        "cumulative_g_path_before": [float(item) for item in cumulative_before],
        "cumulative_g_path_after": [float(item) for item in cumulative_after],
        "predicted_reduction_by_request": [float(item) for item in predicted],
        "goal_met_mask": [bool(item) for item in goal_met],
        "no_descent_mask": [bool(item) for item in no_descent],
        # Construction never advances the authoritative joint clock.  The
        # accepted physical transition consumes the eligibility atomically.
        "target_clock_advance_count": 0,
        "target_clock_advance_eligibility_count": eligibility,
        "h_application_count": 1,
        "old_target_access_count": 0,
        "heldout_access_count": 0,
        "native_or_direct_z_access_count": 0,
    }
    converted = {key: value for key, value in payload.items() if key != "schema"}
    converted["status"] = status
    for name in (
        "loss_by_request",
        "base_norm_by_request",
        "gradient_norm_by_request",
        "dual_norm_by_request",
        "rho_by_request",
        "unit_g_norm_by_request",
        "step_g_path_by_request",
        "cumulative_g_path_before",
        "cumulative_g_path_after",
        "predicted_reduction_by_request",
        "goal_met_mask",
        "no_descent_mask",
    ):
        converted[name] = tuple(converted[name])
    converted["desired_target"] = desired
    converted["identity_sha256"] = canonical_hash(payload)
    return PerRequestTargetStep(**converted)


@dataclass(frozen=True, slots=True)
class GammaBackendAttempt:
    backend: str
    backend_version: str
    success: bool
    status: int
    message_sha256: str
    candidate_v: tuple[float, ...]
    gamma: float
    objective_value: float
    ordered_constraint_names: tuple[str, ...]
    ordered_constraint_slacks: tuple[float, ...]
    ordered_kkt_multipliers: tuple[float, ...]
    maximum_primal_violation: float
    stationarity_residual: float
    complementarity_residual: float
    finite: bool
    passed: bool
    reconstruction_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class JointGammaCertificate:
    gamma: float
    candidate_v: tuple[float, ...]
    active_axes: tuple[str, ...]
    optimizer_pass_count: int
    fallback_invocation_count: int
    selected_backend: str
    maximum_primal_violation: float
    stationarity_residual: float
    complementarity_residual: float
    passed: bool
    feasible_full_floors: bool
    attempts: tuple[GammaBackendAttempt, ...]
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["attempts"] = [item.raw_free_payload() for item in self.attempts]
        return payload


@dataclass(frozen=True, slots=True)
class _GammaConstraint:
    name: str
    function: Callable[[np.ndarray], float]
    jacobian: Callable[[np.ndarray], np.ndarray]
    hessian: Callable[[np.ndarray], np.ndarray]


def _gamma_attempt(
    result: Any,
    *,
    backend: str,
    constraints: Sequence[_GammaConstraint],
    lower: np.ndarray,
    upper: np.ndarray,
) -> GammaBackendAttempt:
    raw = np.asarray(result.x, dtype=np.float64)
    value = raw.copy()
    for index in range(value.size):
        if value[index] < lower[index] and lower[index] - value[index] <= PHYSICAL_SOLVER_PRIMAL_TOLERANCE:
            value[index] = lower[index]
        elif value[index] > upper[index] and value[index] - upper[index] <= PHYSICAL_SOLVER_PRIMAL_TOLERANCE:
            value[index] = upper[index]
    gradient = np.zeros(6, dtype=np.float64)
    gradient[5] = -1.0
    names: list[str] = []
    slacks: list[float] = []
    gradients: list[np.ndarray] = []
    for constraint in constraints:
        names.append(constraint.name)
        slacks.append(float(constraint.function(value)))
        gradients.append(np.asarray(constraint.jacobian(value), dtype=np.float64))
    for index in range(6):
        left = np.zeros(6, dtype=np.float64)
        left[index] = 1.0
        names.extend((f"lower_{index}", f"upper_{index}"))
        slacks.extend((float(value[index] - lower[index]), float(upper[index] - value[index])))
        gradients.extend((left, -left))
    slack = np.asarray(slacks, dtype=np.float64)
    matrix = np.stack(gradients, axis=0)
    active = slack <= 10.0 * PHYSICAL_SOLVER_PRIMAL_TOLERANCE
    multipliers = np.zeros_like(slack)
    if np.any(active) and np.all(np.isfinite(gradient)):
        multipliers[active] = nnls(matrix[active].T, gradient)[0]
    finite = bool(
        value.shape == (6,)
        and np.all(np.isfinite(value))
        and np.all(np.isfinite(slack))
        and np.all(np.isfinite(multipliers))
    )
    primal = float(max(0.0, -float(slack.min(initial=0.0))))
    stationarity = float(
        np.linalg.norm(gradient - matrix.T @ multipliers)
        / max(float(np.linalg.norm(gradient)), 1.0)
    )
    complementarity = float(np.max(np.abs(multipliers * slack), initial=0.0))
    passed = bool(
        finite
        and primal <= PHYSICAL_SOLVER_PRIMAL_TOLERANCE
        and stationarity <= PHYSICAL_SOLVER_KKT_TOLERANCE
        and complementarity <= PHYSICAL_SOLVER_KKT_TOLERANCE
    )
    reconstruction = {
        "candidate": value.tolist(),
        "objective": float(-value[5]),
        "constraint_names": names,
        "constraint_slacks": slack.tolist(),
        "multipliers": multipliers.tolist(),
        "primal": primal,
        "stationarity": stationarity,
        "complementarity": complementarity,
        "passed": passed,
    }
    return GammaBackendAttempt(
        backend,
        SCIPY_VERSION,
        bool(getattr(result, "success", False)),
        int(getattr(result, "status", -1)),
        canonical_hash({"message": str(getattr(result, "message", ""))}),
        tuple(float(item) for item in value[:5]),
        float(value[5]),
        float(-value[5]),
        tuple(names),
        tuple(float(item) for item in slack),
        tuple(float(item) for item in multipliers),
        primal,
        stationarity,
        complementarity,
        finite,
        passed,
        canonical_hash(reconstruction),
    )


def solve_joint_gamma_feasibility(
    problem: RoutingProblem,
    *,
    a_edit: np.ndarray,
    a_transport: np.ndarray,
    delta_edit: float,
    delta_transport: float,
) -> JointGammaCertificate:
    """Certify the common normalized fraction of all active floors."""

    edit = np.asarray(a_edit, dtype=np.float64)
    transport = np.asarray(a_transport, dtype=np.float64)
    if edit.shape != (5,) or transport.shape != (5,) or not np.all(
        np.isfinite(np.concatenate((edit, transport)))
    ):
        raise ODEBFContractError("P1R15 gamma slope geometry differs")
    floors = {
        "edit": float(delta_edit),
        "transport": float(delta_transport),
    }
    if any(not math.isfinite(value) or value < 0.0 for value in floors.values()):
        raise ODEBFContractError("P1R15 gamma floor differs")
    active_axes = tuple(name for name, value in floors.items() if value > 0.0)
    if not active_axes:
        raise ODEBFContractError("P1R15 gamma has no active floor")
    caps = np.minimum(problem.layer_caps, 1.0).astype(np.float64)
    lower = np.zeros(6, dtype=np.float64)
    upper = np.concatenate((caps, np.ones(1, dtype=np.float64)))
    constraints: list[_GammaConstraint] = []
    trust = problem.trust_metric.copy()
    trust_hessian = np.zeros((6, 6), dtype=np.float64)
    trust_hessian[:5, :5] = -2.0 * trust
    constraints.append(
        _GammaConstraint(
            "technical_write_trust",
            lambda value: float(problem.trust_radius**2 - value[:5] @ trust @ value[:5]),
            lambda value: np.concatenate((-2.0 * trust @ value[:5], np.zeros(1))),
            lambda value: trust_hessian.copy(),
        )
    )
    zero_hessian = lambda value: np.zeros((6, 6), dtype=np.float64)  # noqa: E731
    for name, slope in (("edit", edit), ("transport", transport)):
        floor = floors[name]
        if floor <= 0.0:
            continue
        constraints.append(
            _GammaConstraint(
                f"{name}_normalized_floor",
                lambda value, a=slope, d=floor: float(
                    PHYSICAL_WRITER_H * a @ value[:5] - value[5] * d
                ),
                lambda value, a=slope, d=floor: np.concatenate(
                    (PHYSICAL_WRITER_H * a, np.asarray([-d]))
                ),
                zero_hessian,
            )
        )
    objective = lambda value: float(-value[5])  # noqa: E731
    gradient = lambda value: np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -1.0])  # noqa: E731
    initial = np.zeros(6, dtype=np.float64)
    primary = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=tuple(zip(lower, upper, strict=True)),
        constraints=[
            {"type": "ineq", "fun": item.function, "jac": item.jacobian}
            for item in constraints
        ],
        options=dict(PHYSICAL_SOLVER_PRIMARY_OPTIONS),
    )
    attempts = [
        _gamma_attempt(
            primary,
            backend=PHYSICAL_SOLVER_PRIMARY_BACKEND,
            constraints=constraints,
            lower=lower,
            upper=upper,
        )
    ]
    if not attempts[-1].passed and attempts[-1].finite:
        seed = np.concatenate(
            (np.asarray(attempts[-1].candidate_v), [attempts[-1].gamma])
        )
        nonlinear = NonlinearConstraint(
            lambda point: np.asarray([item.function(point) for item in constraints]),
            np.full(len(constraints), -PHYSICAL_SOLVER_FTOL),
            np.full(len(constraints), np.inf),
            jac=lambda point: np.stack([item.jacobian(point) for item in constraints]),
            hess=lambda point, multipliers: sum(
                (
                    float(weight) * item.hessian(point)
                    for weight, item in zip(multipliers, constraints, strict=True)
                ),
                np.zeros((6, 6), dtype=np.float64),
            ),
        )
        fallback = minimize(
            objective,
            seed,
            jac=gradient,
            hess=lambda value: np.zeros((6, 6), dtype=np.float64),
            method="trust-constr",
            bounds=Bounds(lower - PHYSICAL_SOLVER_FTOL, upper + PHYSICAL_SOLVER_FTOL),
            constraints=[nonlinear],
            options=dict(PHYSICAL_SOLVER_FALLBACK_OPTIONS),
        )
        attempts.append(
            _gamma_attempt(
                fallback,
                backend=PHYSICAL_SOLVER_FALLBACK_BACKEND,
                constraints=constraints,
                lower=lower,
                upper=upper,
            )
        )
    selected = attempts[-1]
    passed = selected.passed
    feasible = bool(passed and selected.gamma >= 1.0 - PHYSICAL_SOLVER_PRIMAL_TOLERANCE)
    payload = {
        "schema": f"{P1R15_SCHEMA}-joint-gamma-certificate",
        "gamma": selected.gamma,
        "candidate_v": list(selected.candidate_v),
        "active_axes": list(active_axes),
        "optimizer_pass_count": len(attempts),
        "fallback_invocation_count": len(attempts) - 1,
        "selected_backend": selected.backend,
        "maximum_primal_violation": selected.maximum_primal_violation,
        "stationarity_residual": selected.stationarity_residual,
        "complementarity_residual": selected.complementarity_residual,
        "passed": passed,
        "feasible_full_floors": feasible,
        "attempts": [item.raw_free_payload() for item in attempts],
        "thresholds": {
            "primal": PHYSICAL_SOLVER_PRIMAL_TOLERANCE,
            "kkt": PHYSICAL_SOLVER_KKT_TOLERANCE,
        },
    }
    return JointGammaCertificate(
        selected.gamma,
        selected.candidate_v,
        active_axes,
        len(attempts),
        len(attempts) - 1,
        selected.backend,
        selected.maximum_primal_violation,
        selected.stationarity_residual,
        selected.complementarity_residual,
        passed,
        feasible,
        tuple(attempts),
        canonical_hash(payload),
    )


@dataclass(frozen=True, slots=True)
class SimplexTransportRoutingResult:
    status: SimplexStepStatus
    failure_component: JointFloorComponent
    step_index: int
    a_edit: tuple[float, ...]
    a_transport: tuple[float, ...]
    p_edit_max: float
    p_transport_max: float
    deficit_edit: float
    transport_loss_current: float
    delta_edit: float
    delta_transport: float
    velocity: tuple[float, ...]
    applied_theta: tuple[float, ...]
    speed: float
    alpha: tuple[float, ...] | None
    alpha_status: str
    stage1_velocity: tuple[float, ...] | None
    stage2_velocity: tuple[float, ...] | None
    stage1_speed: float | None
    stage2_speed: float | None
    delta_speed_stage1_to_stage2: float | None
    delta_alpha_l1_stage1_to_stage2: float | None
    p_scale: float
    p_bar_star: float | None
    p_active: bool
    p_allocation_influence: bool
    speed_tie_slack: float | None
    p_tie_slack: float | None
    clock_advance_count: int
    candidate_count: int
    gamma_certificate: JointGammaCertificate | None
    solver_certificates: tuple[PhysicalWriterSolverCertificate, ...]
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["failure_component"] = self.failure_component.value
        payload["gamma_certificate"] = (
            None
            if self.gamma_certificate is None
            else self.gamma_certificate.raw_free_payload()
        )
        payload["solver_certificates"] = [
            item.raw_free_payload() for item in self.solver_certificates
        ]
        return payload


def _vector(value: Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(tuple(float(item) for item in value), dtype=np.float64)
    if result.shape != (5,) or not np.all(np.isfinite(result)):
        raise ODEBFContractError(f"P1R15 {label} differs")
    return result


def _matrix(value: Sequence[Sequence[float]], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (
        result.shape != (5, 5)
        or not np.all(np.isfinite(result))
        or not np.allclose(result, result.T, rtol=0.0, atol=1.0e-12)
        or float(np.min(np.linalg.eigvalsh(result))) < -1.0e-10
    ):
        raise ODEBFContractError(f"P1R15 {label} differs")
    return result


def _alpha(value: np.ndarray) -> tuple[tuple[float, ...] | None, str]:
    speed = float(np.sum(value))
    if speed <= P1R15_EPS_ALPHA:
        return None, "UNDEFINED_ZERO_SPEED"
    alpha = value / speed
    if np.any(alpha < 0.0) or not math.isclose(
        float(np.sum(alpha)), 1.0, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ODEBFContractError("P1R15 alpha simplex differs")
    return tuple(float(item) for item in alpha), "DEFINED_POSITIVE_SPEED"


def _constraint(
    name: str,
    function: Callable[[np.ndarray], float],
    jacobian: Callable[[np.ndarray], np.ndarray],
    hessian: Callable[[np.ndarray], np.ndarray],
) -> PhysicalWriterConstraint:
    return PhysicalWriterConstraint(name, function, jacobian, hessian)


def solve_perrequest_simplex_transport(
    problem: RoutingProblem,
    *,
    a_edit: Sequence[float],
    a_transport: Sequence[float],
    current_nohook_loss: float,
    goal_loss: float,
    transport_loss_current: float,
    step_index: int,
    functional_p_derivative: Sequence[float],
    structural_p_matrix_raw: Sequence[Sequence[float]],
    pre_guard_observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> SimplexTransportRoutingResult:
    """Run edit/transport maxima, gamma feasibility, then s -> P -> C."""

    if step_index < 0 or step_index >= PHYSICAL_WRITER_GRID_COUNT:
        raise ODEBFContractError("P1R15 routing step differs")
    scalars = (current_nohook_loss, goal_loss, transport_loss_current)
    if any(not math.isfinite(float(item)) or float(item) < 0.0 for item in scalars):
        raise ODEBFContractError("P1R15 routing loss differs")
    edit = _vector(a_edit, "edit slope")
    transport = _vector(a_transport, "transport slope")
    p_derivative = _vector(functional_p_derivative, "P derivative")
    p_raw = _matrix(structural_p_matrix_raw, "structural P")
    edit_problem = replace(problem, signed_progress=edit)
    transport_problem = replace(problem, signed_progress=transport)
    caps = np.minimum(problem.layer_caps, 1.0)
    lower = np.zeros(5, dtype=np.float64)
    technical = physical_writer_technical_constraints(
        edit_problem, requested_progress=None
    )
    zero_hessian = lambda value: np.zeros((5, 5), dtype=np.float64)  # noqa: E731
    certificates: list[PhysicalWriterSolverCertificate] = []

    max_edit_v, max_edit_cert = solve_certified_physical_qp(
        phase="p1r15-maximum-edit",
        problem=edit_problem,
        objective=lambda value: float(-edit @ value),
        jacobian=lambda value: -edit.copy(),
        hessian=zero_hessian,
        constraints=technical,
        lower=lower,
        upper=caps,
        initial=np.zeros(5, dtype=np.float64),
        p_max=0.0,
        requested_progress=0.0,
        authority_role="AUTHORITATIVE_EDIT_MAXIMUM",
    )
    certificates.append(max_edit_cert)
    p_edit = max(float(edit @ max_edit_v), 0.0)
    max_tr_v, max_tr_cert = solve_certified_physical_qp(
        phase="p1r15-maximum-transport",
        problem=transport_problem,
        objective=lambda value: float(-transport @ value),
        jacobian=lambda value: -transport.copy(),
        hessian=zero_hessian,
        constraints=technical,
        lower=lower,
        upper=caps,
        initial=np.zeros(5, dtype=np.float64),
        p_max=p_edit,
        requested_progress=0.0,
        authority_role="AUTHORITATIVE_TRANSPORT_MAXIMUM",
    )
    certificates.append(max_tr_cert)
    p_transport = max(float(transport @ max_tr_v), 0.0)
    deficit = max(float(current_nohook_loss - goal_loss), 0.0)
    remaining = PHYSICAL_WRITER_GRID_COUNT - step_index
    delta_edit = (
        min(PHYSICAL_WRITER_H * p_edit, deficit / remaining)
        if deficit > 0.0
        else 0.0
    )
    delta_transport = (
        min(
            PHYSICAL_WRITER_H * p_transport,
            float(transport_loss_current) / remaining,
        )
        if transport_loss_current > 0.0
        else 0.0
    )
    pre_guard = {
        "schema": f"{P1R15_SCHEMA}-pre-guard",
        "step_index": step_index,
        "a_edit": edit.tolist(),
        "a_transport": transport.tolist(),
        "p_edit_max": p_edit,
        "p_transport_max": p_transport,
        "deficit_edit": deficit,
        "transport_loss_current": float(transport_loss_current),
        "delta_edit": delta_edit,
        "delta_transport": delta_transport,
        "all_five_layer_domain": True,
        "positive_layer_mask_access_count": 0,
        "overlay_access_count": 0,
        "hard_h_p_influence_count": 0,
    }
    if pre_guard_observer is not None:
        pre_guard_observer(pre_guard)

    failure = JointFloorComponent.NONE
    if deficit > 0.0 and p_edit <= 0.0:
        failure = JointFloorComponent.EDIT_NO_DIRECTION
    elif transport_loss_current > 0.0 and p_transport <= 0.0:
        failure = JointFloorComponent.TRANSPORT_NO_DIRECTION

    gamma: JointGammaCertificate | None = None
    if failure is JointFloorComponent.NONE and (
        delta_edit > 0.0 or delta_transport > 0.0
    ):
        gamma = solve_joint_gamma_feasibility(
            edit_problem,
            a_edit=edit,
            a_transport=transport,
            delta_edit=delta_edit,
            delta_transport=delta_transport,
        )
        if not gamma.passed:
            failure = JointFloorComponent.NUMERICAL_CERTIFICATE_FAILED
        elif not gamma.feasible_full_floors:
            failure = JointFloorComponent.JOINT_FLOOR_INFEASIBLE

    if failure is not JointFloorComponent.NONE:
        zero = np.zeros(5, dtype=np.float64)
        payload = {
            "status": SimplexStepStatus.NO_JOINT_PHYSICAL_TRANSPORT_DIRECTION,
            "failure_component": failure,
            "step_index": step_index,
            "a_edit": tuple(float(item) for item in edit),
            "a_transport": tuple(float(item) for item in transport),
            "p_edit_max": p_edit,
            "p_transport_max": p_transport,
            "deficit_edit": deficit,
            "transport_loss_current": float(transport_loss_current),
            "delta_edit": delta_edit,
            "delta_transport": delta_transport,
            "velocity": tuple(float(item) for item in zero),
            "applied_theta": tuple(float(item) for item in zero),
            "speed": 0.0,
            "alpha": None,
            "alpha_status": "UNDEFINED_ZERO_SPEED",
            "stage1_velocity": None,
            "stage2_velocity": None,
            "stage1_speed": None,
            "stage2_speed": None,
            "delta_speed_stage1_to_stage2": None,
            "delta_alpha_l1_stage1_to_stage2": None,
            "p_scale": 0.0,
            "p_bar_star": None,
            "p_active": False,
            "p_allocation_influence": False,
            "speed_tie_slack": None,
            "p_tie_slack": None,
            "clock_advance_count": 0,
            "candidate_count": 0,
            "gamma_certificate": gamma,
            "solver_certificates": tuple(certificates),
        }
        identity = canonical_hash(
            {
                **{k: (v.value if isinstance(v, Enum) else v) for k, v in payload.items() if k not in ("gamma_certificate", "solver_certificates")},
                "gamma_certificate": None if gamma is None else gamma.raw_free_payload(),
                "solver_certificates": [item.raw_free_payload() for item in certificates],
            }
        )
        return SimplexTransportRoutingResult(**payload, identity_sha256=identity)

    floor_constraints: list[PhysicalWriterConstraint] = []
    if delta_edit > 0.0:
        floor_constraints.append(
            _constraint(
                "physical_edit_floor",
                lambda value: float(PHYSICAL_WRITER_H * edit @ value - delta_edit),
                lambda value: PHYSICAL_WRITER_H * edit.copy(),
                zero_hessian,
            )
        )
    if delta_transport > 0.0:
        floor_constraints.append(
            _constraint(
                "physical_transport_floor",
                lambda value: float(PHYSICAL_WRITER_H * transport @ value - delta_transport),
                lambda value: PHYSICAL_WRITER_H * transport.copy(),
                zero_hessian,
            )
        )
    feasible = [*technical, *floor_constraints]
    initial = (
        np.zeros(5, dtype=np.float64)
        if gamma is None
        else np.asarray(gamma.candidate_v, dtype=np.float64)
    )
    ones = np.ones(5, dtype=np.float64)
    stage1, speed_cert = solve_certified_physical_qp(
        phase="p1r15-stage1-minimum-global-speed",
        problem=edit_problem,
        objective=lambda value: float(ones @ value),
        jacobian=lambda value: ones.copy(),
        hessian=zero_hessian,
        constraints=feasible,
        lower=lower,
        upper=caps,
        initial=initial,
        p_max=p_edit,
        requested_progress=delta_edit,
        authority_role="AUTHORITATIVE_MINIMUM_L1_SPEED",
    )
    certificates.append(speed_cert)
    speed_star = float(np.sum(stage1))
    speed_tie = _constraint(
        "global_speed_tie",
        lambda value: float(speed_star + P1R15_EPS_SPEED - np.sum(value)),
        lambda value: -ones.copy(),
        zero_hessian,
    )
    pi = PHYSICAL_WRITER_H * np.maximum(p_derivative, 0.0)
    m_step = PHYSICAL_WRITER_H**2 * p_raw
    p_scale = float(pi @ caps + caps @ np.abs(m_step) @ caps)
    p_active = p_scale > 0.0

    def p_bar(value: np.ndarray) -> float:
        return 0.0 if not p_active else float((pi @ value + value @ m_step @ value) / p_scale)

    def p_gradient(value: np.ndarray) -> np.ndarray:
        return np.zeros_like(value) if not p_active else (pi + 2.0 * m_step @ value) / p_scale

    def p_hessian(value: np.ndarray) -> np.ndarray:
        del value
        return np.zeros_like(m_step) if not p_active else 2.0 * m_step / p_scale

    stage2, p_cert = solve_certified_physical_qp(
        phase="p1r15-stage2-minimum-P-within-speed-tie",
        problem=edit_problem,
        objective=p_bar,
        jacobian=p_gradient,
        hessian=p_hessian,
        constraints=[*feasible, speed_tie],
        lower=lower,
        upper=caps,
        initial=stage1,
        p_max=p_edit,
        requested_progress=delta_edit,
        authority_role="AUTHORITATIVE_P_TIE_STAGE",
    )
    certificates.append(p_cert)
    p_star = p_bar(stage2)
    p_tie = _constraint(
        "dimensionless_P_tie",
        lambda value: float(p_star + P1R15_EPS_P - p_bar(value)),
        lambda value: -p_gradient(value),
        lambda value: -p_hessian(value),
    )
    capacity = problem.capacity_metric
    stage3, capacity_cert = solve_certified_physical_qp(
        phase="p1r15-stage3-minimum-capacity-within-speed-P-ties",
        problem=edit_problem,
        objective=lambda value: float(0.5 * value @ capacity @ value),
        jacobian=lambda value: capacity @ value,
        hessian=lambda value: capacity.copy(),
        constraints=[*feasible, speed_tie, p_tie],
        lower=lower,
        upper=caps,
        initial=stage2,
        p_max=p_edit,
        requested_progress=delta_edit,
        authority_role="AUTHORITATIVE_CAPACITY_TIE_STAGE",
    )
    certificates.append(capacity_cert)
    speed = float(np.sum(stage3))
    alpha, alpha_status = _alpha(stage3)
    alpha1, _ = _alpha(stage1)
    alpha2, _ = _alpha(stage2)
    delta_alpha = (
        None
        if alpha1 is None or alpha2 is None
        else float(np.sum(np.abs(np.asarray(alpha2) - np.asarray(alpha1))))
    )
    p_allocation_influence = bool(
        p_active
        and delta_alpha is not None
        and delta_alpha > P1R15_EPS_ALPHA
    )
    theta = PHYSICAL_WRITER_H * stage3
    zero_write = bool(np.count_nonzero(stage3) == 0)
    if zero_write and not (
        deficit == 0.0 and float(transport_loss_current) == 0.0
    ):
        raise ODEBFContractError("P1R15 zero-write totality differs")
    status = (
        SimplexStepStatus.ZERO_WRITE_GOAL_MET
        if zero_write
        else SimplexStepStatus.JOINT_WRITE
    )
    payload = {
        "status": status,
        "failure_component": JointFloorComponent.NONE,
        "step_index": step_index,
        "a_edit": tuple(float(item) for item in edit),
        "a_transport": tuple(float(item) for item in transport),
        "p_edit_max": p_edit,
        "p_transport_max": p_transport,
        "deficit_edit": deficit,
        "transport_loss_current": float(transport_loss_current),
        "delta_edit": delta_edit,
        "delta_transport": delta_transport,
        "velocity": tuple(float(item) for item in stage3),
        "applied_theta": tuple(float(item) for item in theta),
        "speed": speed,
        "alpha": alpha,
        "alpha_status": alpha_status,
        "stage1_velocity": tuple(float(item) for item in stage1),
        "stage2_velocity": tuple(float(item) for item in stage2),
        "stage1_speed": speed_star,
        "stage2_speed": float(np.sum(stage2)),
        "delta_speed_stage1_to_stage2": float(np.sum(stage2) - speed_star),
        "delta_alpha_l1_stage1_to_stage2": delta_alpha,
        "p_scale": p_scale,
        "p_bar_star": p_star,
        "p_active": p_active,
        "p_allocation_influence": p_allocation_influence,
        "speed_tie_slack": float(speed_star + P1R15_EPS_SPEED - speed),
        "p_tie_slack": float(p_star + P1R15_EPS_P - p_bar(stage3)),
        "clock_advance_count": 1,
        "candidate_count": 0 if zero_write else 1,
        "gamma_certificate": gamma,
        "solver_certificates": tuple(certificates),
    }
    identity = canonical_hash(
        {
            **{k: (v.value if isinstance(v, Enum) else v) for k, v in payload.items() if k not in ("gamma_certificate", "solver_certificates")},
            "gamma_certificate": None if gamma is None else gamma.raw_free_payload(),
            "solver_certificates": [item.raw_free_payload() for item in certificates],
        }
    )
    return SimplexTransportRoutingResult(**payload, identity_sha256=identity)


def p1r15_source_contract() -> dict[str, Any]:
    payload = {
        "schema": f"{P1R15_SCHEMA}-source-contract",
        "instruction_id": P1R15_INSTRUCTION_ID,
        "amendment_id": P1R15_AMENDMENT_ID,
        "method_id": P1R15_METHOD_ID,
        "grid_count": PHYSICAL_WRITER_GRID_COUNT,
        "h": PHYSICAL_WRITER_H,
        "target_metric": "G_i=I/||z_base_i||_2^2",
        "target_direction": "-G_i^-1*g_i/||g_i||_(G_i^-1)",
        "eps_base": P1R15_EPS_BASE,
        "eps_loss": P1R15_EPS_LOSS,
        "eps_dual": P1R15_EPS_DUAL,
        "eps_alpha": P1R15_EPS_ALPHA,
        "eps_speed": P1R15_EPS_SPEED,
        "eps_P": P1R15_EPS_P,
        "coefficient_coordinate": "W(theta)=W+sum_l(theta_l*B_l)",
        "applied_coefficient": "theta=h*v",
        "speed": "s=sum_l(v_l)",
        "alpha": "v/s_when_s>eps_alpha_else_UNDEFINED_ZERO_SPEED",
        "joint_feasibility": "AUXILIARY_GAMMA_COMMON_NORMALIZED_FLOOR",
        "lexicographic_stages": ["MIN_SPEED", "MIN_P_BAR", "MIN_CAPACITY"],
        "online_logical_groups_per_step": 13,
        "online_logical_groups_k8": 104,
        "model_forwards_per_full_field": 132,
        "backwards_per_full_field": 22,
        "stage_b_material_count": 0,
        "native_or_direct_z_controller_access_count": 0,
        "hard_h_p_veto_retry_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "JointFloorComponent",
    "JointGammaCertificate",
    "P1R15_AMENDMENT_ID",
    "P1R15_EPS_ALPHA",
    "P1R15_EPS_BASE",
    "P1R15_EPS_DUAL",
    "P1R15_EPS_LOSS",
    "P1R15_EPS_P",
    "P1R15_EPS_SPEED",
    "P1R15_INSTRUCTION_ID",
    "P1R15_METHOD_ID",
    "PerRequestTargetStep",
    "SimplexStepStatus",
    "SimplexTransportRoutingResult",
    "TargetStepStatus",
    "build_per_request_gdual_target_step",
    "p1r15_source_contract",
    "solve_joint_gamma_feasibility",
    "solve_perrequest_simplex_transport",
]
