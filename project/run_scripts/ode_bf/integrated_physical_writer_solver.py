"""Neutral independent-certificate solver primitive for P1R14.

This module owns no experiment arm, preservation policy, or target clock.  It
provides only the inherited FP64 numerical backend and independent primal/KKT
certificate used by the integrated physical-writer E, P, and capacity stages.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import numpy as np
from scipy import __version__ as SCIPY_VERSION
from scipy.optimize import Bounds, NonlinearConstraint, minimize, nnls

from .contracts import ODEBFContractError, canonical_hash
from .routing import RoutingProblem


PHYSICAL_SOLVER_FTOL = 1.0e-12
PHYSICAL_SOLVER_PRIMAL_TOLERANCE = 1.0e-8
PHYSICAL_SOLVER_KKT_TOLERANCE = 1.0e-5
PHYSICAL_SOLVER_MAXITER = 1_000
PHYSICAL_SOLVER_NORMALIZATION_EPSILON = 1.0e-12
PHYSICAL_SOLVER_PRIMARY_BACKEND = "scipy-slsqp-float64"
PHYSICAL_SOLVER_FALLBACK_BACKEND = "scipy-trust-constr-float64"
PHYSICAL_SOLVER_PRIMARY_OPTIONS = (
    ("disp", False),
    ("ftol", PHYSICAL_SOLVER_FTOL),
    ("maxiter", PHYSICAL_SOLVER_MAXITER),
)
PHYSICAL_SOLVER_FALLBACK_OPTIONS = (
    ("barrier_tol", PHYSICAL_SOLVER_FTOL),
    ("gtol", PHYSICAL_SOLVER_FTOL),
    ("maxiter", PHYSICAL_SOLVER_MAXITER),
    ("verbose", 0),
    ("xtol", PHYSICAL_SOLVER_FTOL),
)


@dataclass(frozen=True, slots=True)
class PhysicalWriterConstraint:
    name: str
    function: Callable[[np.ndarray], float]
    jacobian: Callable[[np.ndarray], np.ndarray]
    hessian: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class PhysicalWriterBackendAttempt:
    backend: str
    backend_version: str
    backend_options: tuple[tuple[str, Any], ...]
    success: bool
    status: int
    message_sha256: str
    iterations: int
    raw_candidate_vector: tuple[float, ...]
    candidate_vector: tuple[float, ...]
    objective_value: float
    objective_gradient: tuple[float, ...]
    ordered_constraint_names: tuple[str, ...]
    ordered_constraint_slacks: tuple[float, ...]
    ordered_constraint_gradients: tuple[tuple[float, ...], ...]
    ordered_kkt_multipliers: tuple[float, ...]
    maximum_primal_violation: float
    stationarity_residual: float
    complementarity_residual: float
    finite: bool
    first_false_component: str | None
    passed: bool
    reconstruction_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PhysicalWriterSolverCertificate:
    phase: str
    authority_role: str
    optimizer_status_history: tuple[int, ...]
    optimizer_message_sha256_history: tuple[str, ...]
    optimizer_pass_count: int
    fallback_invocation_count: int
    maximum_primal_violation: float
    stationarity_residual: float
    complementarity_residual: float
    signed_progress: float
    trust_value: float
    p_max: float
    requested_progress: float
    first_false_component: str | None
    active_constraints: tuple[str, ...]
    backend_attempts: tuple[PhysicalWriterBackendAttempt, ...]
    selected_backend: str
    independent_certificate_authority: bool
    passed: bool

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "certificate_thresholds": {
                    "primal": PHYSICAL_SOLVER_PRIMAL_TOLERANCE,
                    "stationarity": PHYSICAL_SOLVER_KKT_TOLERANCE,
                    "complementarity": PHYSICAL_SOLVER_KKT_TOLERANCE,
                },
                "backend_success_is_diagnostic_only": True,
                "scientific_retry_count": 0,
                "field_rebuild_count": 0,
                "candidate_evaluation_count": 0,
            }
        )
        return payload


def physical_writer_technical_constraints(
    problem: RoutingProblem,
    *,
    requested_progress: float | None,
) -> list[PhysicalWriterConstraint]:
    dimension = int(problem.signed_progress.size)
    if dimension != 5 or np.any(problem.layer_caps <= 0.0) or np.any(
        problem.layer_caps > 1.0 + PHYSICAL_SOLVER_PRIMAL_TOLERANCE
    ):
        raise ODEBFContractError("physical writer technical domain differs")
    trust = problem.trust_metric
    constraints = [
        PhysicalWriterConstraint(
            "technical_write_trust",
            lambda value, matrix=trust, radius=problem.trust_radius: float(
                radius**2 - value @ matrix @ value
            ),
            lambda value, matrix=trust: -2.0 * matrix @ value,
            lambda value, matrix=trust: -2.0 * matrix,
        )
    ]
    if requested_progress is not None:
        progress = problem.signed_progress.copy()
        target = float(requested_progress)
        constraints.insert(
            0,
            PhysicalWriterConstraint(
                "physical_edit_progress",
                lambda value, a=progress, floor=target: float(a @ value - floor),
                lambda value, a=progress: a.copy(),
                lambda value, n=dimension: np.zeros((n, n), dtype=np.float64),
            ),
        )
    return constraints


def _canonicalize_bounds(
    value: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    candidate = np.asarray(value, dtype=np.float64).copy()
    for index, observed in enumerate(candidate):
        if observed < lower[index] and lower[index] - observed <= PHYSICAL_SOLVER_PRIMAL_TOLERANCE:
            candidate[index] = lower[index]
        elif observed > upper[index] and observed - upper[index] <= PHYSICAL_SOLVER_PRIMAL_TOLERANCE:
            candidate[index] = upper[index]
    return candidate


def _attempt(
    result: Any,
    *,
    backend: str,
    options: tuple[tuple[str, Any], ...],
    objective: Callable[[np.ndarray], float],
    jacobian: Callable[[np.ndarray], np.ndarray],
    constraints: Sequence[PhysicalWriterConstraint],
    lower: np.ndarray,
    upper: np.ndarray,
) -> PhysicalWriterBackendAttempt:
    raw = np.asarray(result.x, dtype=np.float64)
    value = _canonicalize_bounds(raw, lower, upper)
    gradient = np.asarray(jacobian(value), dtype=np.float64)
    names: list[str] = []
    slacks: list[float] = []
    gradients: list[np.ndarray] = []
    for constraint in constraints:
        names.append(constraint.name)
        slacks.append(float(constraint.function(value)))
        gradients.append(np.asarray(constraint.jacobian(value), dtype=np.float64))
    for index in range(value.size):
        left = np.zeros_like(value)
        left[index] = 1.0
        right = -left
        names.extend((f"lower_{index}", f"upper_{index}"))
        slacks.extend((float(value[index] - lower[index]), float(upper[index] - value[index])))
        gradients.extend((left, right))
    slack_array = np.asarray(slacks, dtype=np.float64)
    gradient_matrix = np.stack(gradients, axis=0)
    active = slack_array <= 10.0 * PHYSICAL_SOLVER_PRIMAL_TOLERANCE
    multipliers = np.zeros_like(slack_array)
    if np.any(active) and np.all(np.isfinite(gradient)):
        multipliers[active] = nnls(gradient_matrix[active].T, gradient)[0]
    finite = bool(
        np.all(np.isfinite(value))
        and np.all(np.isfinite(gradient))
        and np.all(np.isfinite(slack_array))
        and np.all(np.isfinite(multipliers))
        and math.isfinite(float(objective(value)))
    )
    primal = float(max(0.0, -float(slack_array.min(initial=0.0))))
    stationarity = float(
        np.linalg.norm(gradient - gradient_matrix.T @ multipliers)
        / max(float(np.linalg.norm(gradient)), 1.0)
    )
    complementarity = float(np.max(np.abs(multipliers * slack_array), initial=0.0))
    passed = bool(
        finite
        and primal <= PHYSICAL_SOLVER_PRIMAL_TOLERANCE
        and stationarity <= PHYSICAL_SOLVER_KKT_TOLERANCE
        and complementarity <= PHYSICAL_SOLVER_KKT_TOLERANCE
    )
    first_false: str | None = None
    if not finite:
        first_false = "finite"
    elif primal > PHYSICAL_SOLVER_PRIMAL_TOLERANCE:
        first_false = "primal"
    elif stationarity > PHYSICAL_SOLVER_KKT_TOLERANCE:
        first_false = "stationarity"
    elif complementarity > PHYSICAL_SOLVER_KKT_TOLERANCE:
        first_false = "complementarity"
    reconstruction = {
        "candidate_vector": value.tolist(),
        "objective_value": float(objective(value)),
        "objective_gradient": gradient.tolist(),
        "ordered_constraint_names": names,
        "ordered_constraint_slacks": slack_array.tolist(),
        "ordered_constraint_gradients": gradient_matrix.tolist(),
        "ordered_kkt_multipliers": multipliers.tolist(),
        "derived": {
            "finite": finite,
            "primal": primal,
            "stationarity": stationarity,
            "complementarity": complementarity,
            "passed": passed,
        },
    }
    return PhysicalWriterBackendAttempt(
        backend,
        SCIPY_VERSION,
        options,
        bool(getattr(result, "success", False)),
        int(getattr(result, "status", -1)),
        canonical_hash({"message": str(getattr(result, "message", ""))}),
        int(getattr(result, "nit", 0)),
        tuple(float(item) for item in raw),
        tuple(float(item) for item in value),
        float(objective(value)),
        tuple(float(item) for item in gradient),
        tuple(names),
        tuple(float(item) for item in slack_array),
        tuple(tuple(float(item) for item in row) for row in gradient_matrix),
        tuple(float(item) for item in multipliers),
        primal,
        stationarity,
        complementarity,
        finite,
        first_false,
        passed,
        canonical_hash(reconstruction),
    )


def solve_certified_physical_qp(
    *,
    phase: str,
    problem: RoutingProblem,
    objective: Callable[[np.ndarray], float],
    jacobian: Callable[[np.ndarray], np.ndarray],
    hessian: Callable[[np.ndarray], np.ndarray],
    constraints: Sequence[PhysicalWriterConstraint],
    lower: np.ndarray,
    upper: np.ndarray,
    initial: np.ndarray,
    p_max: float,
    requested_progress: float,
    authority_role: str,
) -> tuple[np.ndarray, PhysicalWriterSolverCertificate]:
    if any(value.shape != (5,) for value in (lower, upper, initial)):
        raise ODEBFContractError("physical writer solver dimension differs")
    scipy_constraints = [
        {"type": "ineq", "fun": item.function, "jac": item.jacobian}
        for item in constraints
    ]
    primary = minimize(
        objective,
        np.asarray(initial, dtype=np.float64),
        jac=jacobian,
        method="SLSQP",
        bounds=tuple(zip(lower, upper, strict=True)),
        constraints=scipy_constraints,
        options=dict(PHYSICAL_SOLVER_PRIMARY_OPTIONS),
    )
    attempts = [
        _attempt(
            primary,
            backend=PHYSICAL_SOLVER_PRIMARY_BACKEND,
            options=PHYSICAL_SOLVER_PRIMARY_OPTIONS,
            objective=objective,
            jacobian=jacobian,
            constraints=constraints,
            lower=lower,
            upper=upper,
        )
    ]
    if not attempts[-1].passed and attempts[-1].finite:
        seed = np.asarray(attempts[-1].candidate_vector, dtype=np.float64)
        feasible = bool(
            np.all(seed >= lower - PHYSICAL_SOLVER_FTOL)
            and np.all(seed <= upper + PHYSICAL_SOLVER_FTOL)
            and all(item.function(seed) >= -PHYSICAL_SOLVER_FTOL for item in constraints)
        )
        nonlinear = NonlinearConstraint(
            lambda point: np.asarray([item.function(point) for item in constraints]),
            np.full(len(constraints), -PHYSICAL_SOLVER_FTOL),
            np.full(len(constraints), np.inf),
            jac=lambda point: np.stack([item.jacobian(point) for item in constraints]),
            hess=lambda point, multipliers: sum(
                (float(weight) * item.hessian(point) for weight, item in zip(multipliers, constraints, strict=True)),
                np.zeros((5, 5), dtype=np.float64),
            ),
            keep_feasible=feasible,
        )
        fallback = minimize(
            objective,
            seed,
            jac=jacobian,
            hess=hessian,
            method="trust-constr",
            bounds=Bounds(
                lower - PHYSICAL_SOLVER_FTOL,
                upper + PHYSICAL_SOLVER_FTOL,
                keep_feasible=feasible,
            ),
            constraints=[nonlinear],
            options=dict(PHYSICAL_SOLVER_FALLBACK_OPTIONS),
        )
        attempts.append(
            _attempt(
                fallback,
                backend=PHYSICAL_SOLVER_FALLBACK_BACKEND,
                options=PHYSICAL_SOLVER_FALLBACK_OPTIONS,
                objective=objective,
                jacobian=jacobian,
                constraints=constraints,
                lower=lower,
                upper=upper,
            )
        )
    selected = attempts[-1]
    value = np.asarray(selected.candidate_vector, dtype=np.float64)
    certificate = PhysicalWriterSolverCertificate(
        phase,
        authority_role,
        tuple(item.status for item in attempts),
        tuple(item.message_sha256 for item in attempts),
        len(attempts),
        len(attempts) - 1,
        selected.maximum_primal_violation,
        selected.stationarity_residual,
        selected.complementarity_residual,
        float(problem.signed_progress @ value),
        float(value @ problem.trust_metric @ value),
        float(p_max),
        float(requested_progress),
        selected.first_false_component,
        tuple(
            name
            for name, slack in zip(
                selected.ordered_constraint_names,
                selected.ordered_constraint_slacks,
                strict=True,
            )
            if abs(slack) <= 10.0 * PHYSICAL_SOLVER_PRIMAL_TOLERANCE
        ),
        tuple(attempts),
        selected.backend,
        True,
        selected.passed,
    )
    if not certificate.passed:
        raise ODEBFContractError(
            f"physical writer {phase} NUMERIC_QP_UNCERTIFIED: "
            f"{certificate.raw_free_payload()}"
        )
    return value, certificate


__all__ = [
    "PHYSICAL_SOLVER_FTOL",
    "PHYSICAL_SOLVER_KKT_TOLERANCE",
    "PHYSICAL_SOLVER_NORMALIZATION_EPSILON",
    "PHYSICAL_SOLVER_PRIMAL_TOLERANCE",
    "PhysicalWriterConstraint",
    "PhysicalWriterSolverCertificate",
    "physical_writer_technical_constraints",
    "solve_certified_physical_qp",
]
