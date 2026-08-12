"""Lexicographic signed-progress routing and matched Generic/BF velocities."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize, nnls

from .contracts import (
    FieldIdentity,
    ODEBFContractError,
    canonical_hash,
    finite,
    positive,
)


class RoutingStatus(str, Enum):
    FEASIBLE = "FEASIBLE"
    PROGRESS_INFEASIBLE = "PROGRESS_INFEASIBLE"


class PreservationConstraintPolicy(str, Enum):
    """Whether preservation constraints participate in routing decisions.

    ``OBSERVATION_ONLY`` leaves progress, non-negativity, layer caps, and all
    numerical certificates active.  Only structural H/P and the write-trust
    radius are removed from the routing feasible set; their values remain
    available from the unchanged :class:`RoutingProblem` for diagnostics.
    """

    LOCKED = "LOCKED"
    OBSERVATION_ONLY = "ALLOFF"


@dataclass(frozen=True, slots=True)
class QuadraticBarrier:
    label: str
    offset: float
    linear: np.ndarray
    gram: np.ndarray
    budget: float
    approximation: str
    common_functional_space_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.label not in ("historical", "pretrained"):
            raise ODEBFContractError("quadratic barrier label is invalid")
        offset = finite("barrier offset", self.offset)
        budget = finite("barrier budget", self.budget)
        if budget < offset:
            raise ODEBFContractError("barrier is infeasible at the entry state")
        linear = np.asarray(self.linear, dtype=np.float64)
        gram = np.asarray(self.gram, dtype=np.float64)
        if linear.ndim != 1 or gram.shape != (linear.size, linear.size):
            raise ODEBFContractError("quadratic barrier dimensions differ")
        if not np.isfinite(linear).all() or not np.isfinite(gram).all():
            raise ODEBFContractError("quadratic barrier contains non-finite values")
        if not np.allclose(gram, gram.T, rtol=0.0, atol=1.0e-12):
            raise ODEBFContractError("quadratic barrier Gram is not symmetric")
        eigenvalues = np.linalg.eigvalsh(gram)
        if float(eigenvalues.min(initial=0.0)) < -1.0e-10:
            raise ODEBFContractError("quadratic barrier Gram is not PSD")
        if self.approximation not in ("layer-local-diagonal", "common-functional-gram"):
            raise ODEBFContractError("quadratic barrier approximation is unlabeled")
        if self.approximation == "layer-local-diagonal":
            if not np.array_equal(gram, np.diag(np.diag(gram))):
                raise ODEBFContractError("layer-local diagonal barrier is not diagonal")
        elif self.common_functional_space_sha256 is None or len(self.common_functional_space_sha256) != 64:
            raise ODEBFContractError("functional Gram lacks a common-space identity")
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "budget", budget)
        object.__setattr__(self, "linear", linear.copy())
        object.__setattr__(self, "gram", gram.copy())

    def value(self, velocity: np.ndarray) -> float:
        value = np.asarray(velocity, dtype=np.float64)
        return float(self.offset + 2.0 * self.linear @ value + value @ self.gram @ value)

    def slack(self, velocity: np.ndarray) -> float:
        return self.budget - self.value(velocity)

    def slack_gradient(self, velocity: np.ndarray) -> np.ndarray:
        value = np.asarray(velocity, dtype=np.float64)
        return -(2.0 * self.linear + 2.0 * self.gram @ value)


@dataclass(frozen=True, slots=True)
class RoutingProblem:
    signed_progress: np.ndarray
    capacity_metric: np.ndarray
    trust_metric: np.ndarray
    trust_radius: float
    layer_caps: np.ndarray
    requested_progress: float
    minimum_progress: float
    historical: QuadraticBarrier
    pretrained: QuadraticBarrier

    def __post_init__(self) -> None:
        progress = np.asarray(self.signed_progress, dtype=np.float64)
        capacity = np.asarray(self.capacity_metric, dtype=np.float64)
        trust = np.asarray(self.trust_metric, dtype=np.float64)
        caps = np.asarray(self.layer_caps, dtype=np.float64)
        dimension = progress.size
        if progress.ndim != 1 or dimension < 2:
            raise ODEBFContractError("routing needs at least two layer directions")
        if capacity.shape != (dimension, dimension) or trust.shape != (dimension, dimension):
            raise ODEBFContractError("routing metric dimensions differ")
        if caps.shape != (dimension,) or np.any(caps <= 0.0):
            raise ODEBFContractError("routing layer caps are invalid")
        if not all(np.isfinite(item).all() for item in (progress, capacity, trust, caps)):
            raise ODEBFContractError("routing problem contains non-finite values")
        for name, matrix, strictly_positive in (
            ("capacity", capacity, True),
            ("trust", trust, False),
        ):
            if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
                raise ODEBFContractError(f"{name} metric is not symmetric")
            minimum = float(np.min(np.linalg.eigvalsh(matrix)))
            threshold = 1.0e-12 if strictly_positive else -1.0e-10
            if minimum < threshold:
                raise ODEBFContractError(f"{name} metric is not PSD/positive definite")
        if self.historical.linear.size != dimension or self.pretrained.linear.size != dimension:
            raise ODEBFContractError("H/P barrier dimensions differ from routing")
        requested = positive("requested progress", self.requested_progress)
        minimum = positive("minimum progress", self.minimum_progress)
        if requested < minimum:
            raise ODEBFContractError("requested progress is below the locked minimum")
        object.__setattr__(self, "signed_progress", progress.copy())
        object.__setattr__(self, "capacity_metric", capacity.copy())
        object.__setattr__(self, "trust_metric", trust.copy())
        object.__setattr__(self, "layer_caps", caps.copy())
        object.__setattr__(self, "trust_radius", positive("trust radius", self.trust_radius))
        object.__setattr__(self, "requested_progress", requested)
        object.__setattr__(self, "minimum_progress", minimum)

    @property
    def positive_direction_mask(self) -> np.ndarray:
        # Signed derivative; no positive clipping is permitted.
        return self.signed_progress > 0.0

    def identity(self) -> str:
        return canonical_hash(
            {
                "signed_progress": self.signed_progress.tolist(),
                "capacity_metric": self.capacity_metric.tolist(),
                "trust_metric": self.trust_metric.tolist(),
                "trust_radius": self.trust_radius,
                "layer_caps": self.layer_caps.tolist(),
                "requested_progress": self.requested_progress,
                "minimum_progress": self.minimum_progress,
                "historical": {
                    "offset": self.historical.offset,
                    "linear": self.historical.linear.tolist(),
                    "gram": self.historical.gram.tolist(),
                    "budget": self.historical.budget,
                    "approximation": self.historical.approximation,
                },
                "pretrained": {
                    "offset": self.pretrained.offset,
                    "linear": self.pretrained.linear.tolist(),
                    "gram": self.pretrained.gram.tolist(),
                    "budget": self.pretrained.budget,
                    "approximation": self.pretrained.approximation,
                },
            }
        )


@dataclass(frozen=True, slots=True)
class ZeroActionRoutingProblem(RoutingProblem):
    """Exact zero-action geometry for a scientifically totalized transition.

    This type deliberately does not relax :class:`RoutingProblem`: it accepts
    only an exactly zero slope/trust/radius/request tuple.  It exists so a
    controller that has already certified a semantic zero direction can
    serialize and materialize the required zero action without inventing a
    positive trust radius or capacity.
    """

    def __post_init__(self) -> None:
        progress = np.asarray(self.signed_progress, dtype=np.float64)
        capacity = np.asarray(self.capacity_metric, dtype=np.float64)
        trust = np.asarray(self.trust_metric, dtype=np.float64)
        caps = np.asarray(self.layer_caps, dtype=np.float64)
        dimension = progress.size
        if progress.ndim != 1 or dimension < 2:
            raise ODEBFContractError("zero-action routing needs layer directions")
        if capacity.shape != (dimension, dimension) or trust.shape != (
            dimension,
            dimension,
        ):
            raise ODEBFContractError("zero-action routing metric dimensions differ")
        if caps.shape != (dimension,) or np.any(caps <= 0.0):
            raise ODEBFContractError("zero-action routing layer caps are invalid")
        if not all(
            np.isfinite(item).all() for item in (progress, capacity, trust, caps)
        ):
            raise ODEBFContractError("zero-action routing contains non-finite values")
        if (
            not np.array_equal(progress, np.zeros_like(progress))
            or not np.array_equal(trust, np.zeros_like(trust))
            or float(self.trust_radius) != 0.0
            or float(self.requested_progress) != 0.0
            or float(self.minimum_progress) != 0.0
        ):
            raise ODEBFContractError("zero-action routing carries nonzero action geometry")
        if not np.allclose(capacity, capacity.T, rtol=0.0, atol=1.0e-12):
            raise ODEBFContractError("zero-action capacity metric is not symmetric")
        if float(np.min(np.linalg.eigvalsh(capacity))) < 1.0e-12:
            raise ODEBFContractError("zero-action capacity metric is not positive definite")
        if (
            self.historical.linear.size != dimension
            or self.pretrained.linear.size != dimension
        ):
            raise ODEBFContractError("zero-action H/P dimensions differ")
        object.__setattr__(self, "signed_progress", progress.copy())
        object.__setattr__(self, "capacity_metric", capacity.copy())
        object.__setattr__(self, "trust_metric", trust.copy())
        object.__setattr__(self, "layer_caps", caps.copy())
        object.__setattr__(self, "trust_radius", 0.0)
        object.__setattr__(self, "requested_progress", 0.0)
        object.__setattr__(self, "minimum_progress", 0.0)


@dataclass(frozen=True, slots=True)
class SolverCertificate:
    phase: str
    solver: str
    success: bool
    iterations: int
    maximum_primal_violation: float
    stationarity_residual: float
    complementarity_residual: float
    signed_progress: float
    historical_value: float | None
    pretrained_value: float | None
    trust_value: float
    excluded_nonpositive_directions: tuple[int, ...]
    passed: bool


CertificateObserver = Callable[[SolverCertificate], None]


@dataclass(frozen=True, slots=True)
class RawVelocity:
    values: np.ndarray
    problem_identity: str
    velocity_identity: str
    maximum_feasible_progress: float
    certificate: SolverCertificate


@dataclass(frozen=True, slots=True)
class BFProjectionResult:
    status: RoutingStatus
    raw_velocity_identity: str
    maximum_feasible_progress: float
    values: np.ndarray | None
    certificate: SolverCertificate
    projection_distance: float | None


def _constraint_functions(
    problem: RoutingProblem,
    active_indices: np.ndarray,
    *,
    include_barriers: bool,
    requested_progress: float | None,
    preservation_policy: PreservationConstraintPolicy,
) -> list[tuple[str, Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]]:
    progress = problem.signed_progress[active_indices]
    trust = problem.trust_metric[np.ix_(active_indices, active_indices)]
    constraints: list[tuple[str, Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]] = []
    if requested_progress is not None:
        constraints.append(
            (
                "progress",
                lambda value, p=progress, target=requested_progress: float(p @ value - target),
                lambda value, p=progress: p.copy(),
            )
        )
    if preservation_policy is PreservationConstraintPolicy.LOCKED:
        constraints.append(
            (
                "trust",
                lambda value, matrix=trust, radius=problem.trust_radius: float(radius**2 - value @ matrix @ value),
                lambda value, matrix=trust: -2.0 * matrix @ value,
            )
        )
    if (
        include_barriers
        and preservation_policy is PreservationConstraintPolicy.LOCKED
    ):
        for barrier in (problem.historical, problem.pretrained):
            linear = barrier.linear[active_indices]
            gram = barrier.gram[np.ix_(active_indices, active_indices)]
            constraints.append(
                (
                    barrier.label,
                    lambda value, offset=barrier.offset, lin=linear, matrix=gram, budget=barrier.budget: float(
                        budget - (offset + 2.0 * lin @ value + value @ matrix @ value)
                    ),
                    lambda value, lin=linear, matrix=gram: -(2.0 * lin + 2.0 * matrix @ value),
                )
            )
    return constraints


def _certificate(
    *,
    phase: str,
    result: object,
    value: np.ndarray,
    objective_gradient: np.ndarray,
    constraints: Sequence[tuple[str, Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]],
    caps: np.ndarray,
    problem: RoutingProblem,
    active_indices: np.ndarray,
    include_barriers: bool,
    preservation_policy: PreservationConstraintPolicy,
    primal_tolerance: float,
    kkt_tolerance: float,
) -> SolverCertificate:
    slacks: list[float] = []
    active_gradients: list[np.ndarray] = []
    active_slacks: list[float] = []
    for _, function, jacobian in constraints:
        slack = float(function(value))
        slacks.append(slack)
        if slack <= 10.0 * primal_tolerance:
            active_gradients.append(np.asarray(jacobian(value), dtype=np.float64))
            active_slacks.append(slack)
    for index in range(value.size):
        lower_slack = float(value[index])
        upper_slack = float(caps[index] - value[index])
        slacks.extend((lower_slack, upper_slack))
        basis = np.zeros_like(value)
        basis[index] = 1.0
        if lower_slack <= 10.0 * primal_tolerance:
            active_gradients.append(basis)
            active_slacks.append(lower_slack)
        if upper_slack <= 10.0 * primal_tolerance:
            active_gradients.append(-basis)
            active_slacks.append(upper_slack)
    maximum_violation = max(0.0, max((-slack for slack in slacks), default=0.0))
    if active_gradients:
        gradient_matrix = np.stack(active_gradients, axis=1)
        multipliers, stationarity = nnls(gradient_matrix, objective_gradient)
        scale = max(float(np.linalg.norm(objective_gradient)), 1.0)
        stationarity /= scale
        complementarity = max(
            (abs(float(multiplier * slack)) for multiplier, slack in zip(multipliers, active_slacks)),
            default=0.0,
        )
    else:
        stationarity = float(np.linalg.norm(objective_gradient)) / max(float(np.linalg.norm(objective_gradient)), 1.0)
        complementarity = 0.0
    expanded = np.zeros(problem.signed_progress.size, dtype=np.float64)
    expanded[active_indices] = value
    barriers_in_decision = bool(
        include_barriers
        and preservation_policy is PreservationConstraintPolicy.LOCKED
    )
    historical_value = problem.historical.value(expanded) if barriers_in_decision else None
    pretrained_value = problem.pretrained.value(expanded) if barriers_in_decision else None
    trust_value = float(expanded @ problem.trust_metric @ expanded)
    success = bool(getattr(result, "success"))
    passed = (
        success
        and maximum_violation <= primal_tolerance
        and stationarity <= kkt_tolerance
        and complementarity <= kkt_tolerance
    )
    return SolverCertificate(
        phase,
        "scipy-slsqp-float64",
        success,
        int(getattr(result, "nit")),
        maximum_violation,
        float(stationarity),
        float(complementarity),
        float(problem.signed_progress @ expanded),
        historical_value,
        pretrained_value,
        trust_value,
        tuple(int(index) for index in np.flatnonzero(~problem.positive_direction_mask)),
        passed,
    )


def _solve(
    problem: RoutingProblem,
    *,
    phase: str,
    include_barriers: bool,
    requested_progress: float | None,
    initial: np.ndarray | None,
    primal_tolerance: float,
    kkt_tolerance: float,
    preservation_policy: PreservationConstraintPolicy,
    certificate_observer: CertificateObserver | None = None,
) -> tuple[np.ndarray, SolverCertificate]:
    active_indices = np.flatnonzero(problem.positive_direction_mask)
    if active_indices.size == 0:
        raise ODEBFContractError("all signed rewrite directions are nonpositive")
    progress = problem.signed_progress[active_indices]
    capacity = problem.capacity_metric[np.ix_(active_indices, active_indices)]
    caps = problem.layer_caps[active_indices]
    constraints = _constraint_functions(
        problem,
        active_indices,
        include_barriers=include_barriers,
        requested_progress=requested_progress,
        preservation_policy=preservation_policy,
    )
    scipy_constraints = [
        {"type": "ineq", "fun": function, "jac": jacobian}
        for _, function, jacobian in constraints
    ]
    if phase.startswith("maximum-progress"):
        objective = lambda value: float(-progress @ value)
        jacobian = lambda value: -progress.copy()
    else:
        objective = lambda value: float(0.5 * value @ capacity @ value)
        jacobian = lambda value: capacity @ value
    x0 = np.zeros(active_indices.size, dtype=np.float64) if initial is None else np.asarray(initial, dtype=np.float64)
    result = minimize(
        objective,
        x0,
        jac=jacobian,
        method="SLSQP",
        bounds=tuple((0.0, float(cap)) for cap in caps),
        constraints=scipy_constraints,
        options={"ftol": 1.0e-12, "maxiter": 1000, "disp": False},
    )
    value = np.asarray(result.x, dtype=np.float64)
    certificate = _certificate(
        phase=phase,
        result=result,
        value=value,
        objective_gradient=np.asarray(jacobian(value), dtype=np.float64),
        constraints=constraints,
        caps=caps,
        problem=problem,
        active_indices=active_indices,
        include_barriers=include_barriers,
        preservation_policy=preservation_policy,
        primal_tolerance=primal_tolerance,
        kkt_tolerance=kkt_tolerance,
    )
    if certificate_observer is not None:
        certificate_observer(certificate)
    if not certificate.passed:
        raise ODEBFContractError("routing solver certificate failed")
    expanded = np.zeros(problem.signed_progress.size, dtype=np.float64)
    expanded[active_indices] = value
    return expanded, certificate


def _maximum_progress(
    problem: RoutingProblem,
    *,
    include_barriers: bool,
    primal_tolerance: float,
    kkt_tolerance: float,
    preservation_policy: PreservationConstraintPolicy,
    certificate_observer: CertificateObserver | None = None,
) -> tuple[np.ndarray, SolverCertificate]:
    return _solve(
        problem,
        phase=(
            "maximum-progress-with-barriers"
            if include_barriers
            else "maximum-progress-raw"
        ) + (
            ""
            if preservation_policy is PreservationConstraintPolicy.LOCKED
            else "-preservation-observation-only"
        ),
        include_barriers=include_barriers,
        requested_progress=None,
        initial=None,
        primal_tolerance=primal_tolerance,
        kkt_tolerance=kkt_tolerance,
        preservation_policy=preservation_policy,
        certificate_observer=certificate_observer,
    )


def solve_raw_velocity(
    problem: RoutingProblem,
    *,
    primal_tolerance: float = 1.0e-8,
    kkt_tolerance: float = 1.0e-5,
    preservation_policy: PreservationConstraintPolicy = (
        PreservationConstraintPolicy.LOCKED
    ),
    certificate_observer: CertificateObserver | None = None,
) -> RawVelocity:
    maximum, maximum_certificate = _maximum_progress(
        problem,
        include_barriers=False,
        primal_tolerance=primal_tolerance,
        kkt_tolerance=kkt_tolerance,
        preservation_policy=preservation_policy,
        certificate_observer=certificate_observer,
    )
    maximum_progress = float(problem.signed_progress @ maximum)
    if maximum_progress + primal_tolerance < problem.requested_progress:
        raise ODEBFContractError("raw velocity cannot satisfy requested progress")
    scale = problem.requested_progress / maximum_progress
    initial = maximum[problem.positive_direction_mask] * scale
    values, certificate = _solve(
        problem,
        phase="minimum-capacity-raw" + (
            ""
            if preservation_policy is PreservationConstraintPolicy.LOCKED
            else "-preservation-observation-only"
        ),
        include_barriers=False,
        requested_progress=problem.requested_progress,
        initial=initial,
        primal_tolerance=primal_tolerance,
        kkt_tolerance=kkt_tolerance,
        preservation_policy=preservation_policy,
        certificate_observer=certificate_observer,
    )
    payload = {
        "problem": problem.identity(),
        "values": values.tolist(),
        "maximum_progress": maximum_certificate.signed_progress,
        "requested_progress": problem.requested_progress,
    }
    if preservation_policy is not PreservationConstraintPolicy.LOCKED:
        payload["preservation_constraint_policy"] = preservation_policy.value
    return RawVelocity(
        values,
        problem.identity(),
        canonical_hash(payload),
        maximum_progress,
        certificate,
    )


def project_bf_velocity(
    problem: RoutingProblem,
    raw_velocity: RawVelocity,
    *,
    primal_tolerance: float = 1.0e-8,
    kkt_tolerance: float = 1.0e-5,
    preservation_policy: PreservationConstraintPolicy = (
        PreservationConstraintPolicy.LOCKED
    ),
    certificate_observer: CertificateObserver | None = None,
) -> BFProjectionResult:
    if raw_velocity.problem_identity != problem.identity():
        raise ODEBFContractError("BF projector did not receive the matched raw velocity")
    maximum, maximum_certificate = _maximum_progress(
        problem,
        include_barriers=True,
        primal_tolerance=primal_tolerance,
        kkt_tolerance=kkt_tolerance,
        preservation_policy=preservation_policy,
        certificate_observer=certificate_observer,
    )
    maximum_progress = float(problem.signed_progress @ maximum)
    if maximum_progress + primal_tolerance < problem.minimum_progress or maximum_progress + primal_tolerance < problem.requested_progress:
        return BFProjectionResult(
            RoutingStatus.PROGRESS_INFEASIBLE,
            raw_velocity.velocity_identity,
            maximum_progress,
            None,
            maximum_certificate,
            None,
        )
    initial = maximum[problem.positive_direction_mask] * (
        problem.requested_progress / maximum_progress
    )
    values, certificate = _solve(
        problem,
        phase="minimum-capacity-cbf-projection" + (
            ""
            if preservation_policy is PreservationConstraintPolicy.LOCKED
            else "-preservation-observation-only"
        ),
        include_barriers=True,
        requested_progress=problem.requested_progress,
        initial=initial,
        primal_tolerance=primal_tolerance,
        kkt_tolerance=kkt_tolerance,
        preservation_policy=preservation_policy,
        certificate_observer=certificate_observer,
    )
    difference = values - raw_velocity.values
    distance = float(np.sqrt(max(difference @ problem.capacity_metric @ difference, 0.0)))
    return BFProjectionResult(
        RoutingStatus.FEASIBLE,
        raw_velocity.velocity_identity,
        maximum_progress,
        values,
        certificate,
        distance,
    )


@dataclass(frozen=True, slots=True)
class CoupledField:
    identity: FieldIdentity
    write_velocity: np.ndarray
    target_velocity: np.ndarray
    target_backward_count: int

    def __post_init__(self) -> None:
        write = np.asarray(self.write_velocity, dtype=np.float64)
        target = np.asarray(self.target_velocity, dtype=np.float64)
        if write.ndim != 1 or target.ndim != 1 or not np.isfinite(write).all() or not np.isfinite(target).all():
            raise ODEBFContractError("coupled field contains invalid velocities")
        if self.target_backward_count <= 0:
            raise ODEBFContractError("coupled field lacks target-backward accounting")
        object.__setattr__(self, "write_velocity", write.copy())
        object.__setattr__(self, "target_velocity", target.copy())

    def beta_scaled(self, beta: float) -> tuple[np.ndarray, np.ndarray]:
        value = finite("coupled field beta", beta)
        if value <= 0.0 or value > 1.0:
            raise ODEBFContractError("coupled field beta is outside (0,1]")
        return value * self.write_velocity, value * self.target_velocity


@dataclass(frozen=True, slots=True)
class BacktrackingVerdict:
    beta: float
    raw_predicted_progress: float | None
    predicted_beta_progress: float | None
    actual_signed_progress: float
    trust_ratio: float | None
    progress_pass: bool
    structural_h_pass: bool
    structural_p_pass: bool
    trust_pass: bool
    functional_h_pass: bool
    functional_p_pass: bool
    authoritative_bf16_pass: bool

    @property
    def accepted(self) -> bool:
        return all(
            (
                self.progress_pass,
                self.structural_h_pass,
                self.structural_p_pass,
                self.trust_pass,
                self.functional_h_pass,
                self.functional_p_pass,
                self.authoritative_bf16_pass,
            )
        )

    @property
    def first_rejecting_gate(self) -> str | None:
        hierarchy = (
            ("progress", self.progress_pass),
            ("structural_h", self.structural_h_pass),
            ("structural_p", self.structural_p_pass),
            ("trust", self.trust_pass),
            ("functional_h", self.functional_h_pass),
            ("functional_p", self.functional_p_pass),
            ("authoritative_bf16", self.authoritative_bf16_pass),
        )
        return next((name for name, passed in hierarchy if not passed), None)


def verify_backtracked_candidate(
    problem: RoutingProblem,
    velocity: np.ndarray,
    *,
    beta: float,
    actual_signed_progress: float,
    functional_h_pass: bool,
    functional_p_pass: bool,
    authoritative_bf16_pass: bool,
    minimum_progress: float = 1.0e-8,
    rho_accept: float = 0.1,
    certificate_tolerance: float = 1.0e-8,
) -> BacktrackingVerdict:
    scale = finite("backtracking beta", beta)
    if scale <= 0.0 or scale > 1.0:
        raise ODEBFContractError("backtracking beta is outside (0,1]")
    candidate_velocity = np.asarray(velocity, dtype=np.float64)
    if candidate_velocity.ndim != 1 or candidate_velocity.shape != problem.signed_progress.shape:
        raise ODEBFContractError("backtracking velocity dimension differs")
    candidate = scale * candidate_velocity
    actual = finite("actual signed progress", actual_signed_progress)
    epsilon = positive("minimum progress", minimum_progress)
    threshold = positive("trust-ratio acceptance threshold", rho_accept)
    certificate_epsilon = finite(
        "backtracking certificate tolerance", certificate_tolerance
    )
    if certificate_epsilon < 0.0:
        raise ODEBFContractError("backtracking certificate tolerance is negative")
    if threshold > 1.0:
        raise ODEBFContractError("trust-ratio acceptance threshold exceeds one")
    prediction_finite = bool(np.isfinite(candidate_velocity).all())
    raw_prediction: float | None = None
    predicted_beta: float | None = None
    trust_ratio: float | None = None
    if prediction_finite:
        observed_raw = float(problem.signed_progress @ candidate_velocity)
        prediction_finite = math.isfinite(observed_raw)
        if prediction_finite:
            raw_prediction = observed_raw
            predicted_beta = scale * max(observed_raw, 0.0)
            if observed_raw > 0.0:
                candidate_prediction = float(problem.signed_progress @ candidate)
                if not math.isclose(
                    candidate_prediction,
                    predicted_beta,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                ):
                    raise ODEBFContractError("beta-scaled predicted progress differs")
            if predicted_beta > 0.0 and math.isfinite(predicted_beta):
                trust_ratio = actual / max(predicted_beta, epsilon)
    progress_pass = bool(
        predicted_beta is not None
        and predicted_beta > 0.0
        and trust_ratio is not None
        and math.isfinite(trust_ratio)
        and actual >= epsilon
        and trust_ratio >= threshold
    )
    candidate_finite = bool(np.isfinite(candidate).all())
    historical_pass = bool(
        candidate_finite
        and problem.historical.value(candidate)
        <= problem.historical.budget + certificate_epsilon
    )
    pretrained_pass = bool(
        candidate_finite
        and problem.pretrained.value(candidate)
        <= problem.pretrained.budget + certificate_epsilon
    )
    trust_pass = bool(
        candidate_finite
        and float(candidate @ problem.trust_metric @ candidate)
        <= problem.trust_radius**2 + certificate_epsilon
    )
    return BacktrackingVerdict(
        scale,
        raw_prediction,
        predicted_beta,
        actual,
        trust_ratio,
        progress_pass,
        historical_pass,
        pretrained_pass,
        trust_pass,
        bool(functional_h_pass),
        bool(functional_p_pass),
        bool(authoritative_bf16_pass),
    )
