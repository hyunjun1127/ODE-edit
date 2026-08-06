"""Fixed-grid E8 preservation-aware soft routing.

This module is deliberately independent from the adaptive P1R6 controller.
It exposes a fixed eight-interval clock and the parameter-free lexicographic
routing program used by the P1R7 CPU lock.  Scientific quantities never gate
an E8 transition: only malformed inputs or a failed numerical certificate are
fail-closed here.

The :class:`RoutingProblem` supplied by :mod:`p1_controller` is expressed in
*applied-step* coordinates: its signed progress and linear barrier terms have
already been multiplied by ``h`` and its capacity, trust, and Gram matrices by
``h**2``.  Consequently this module evaluates those matrices directly against
the velocity ``v``.  Multiplying them by ``h`` again is a unit error.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize, nnls

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .routing import QuadraticBarrier, RoutingProblem


FIXED_E8_LAYER_ORDER = (4, 5, 6, 7, 8)
FIXED_E8_GRID_COUNT = 8
FIXED_E8_H = Fraction(1, 8)
FIXED_E8_KAPPA = 0.25
FIXED_E8_NORMALIZATION_EPSILON = 1.0e-12
FIXED_E8_XI_TIE_TOLERANCE = 1.0e-8
FIXED_E8_PRIMAL_TOLERANCE = 1.0e-8
FIXED_E8_KKT_TOLERANCE = 1.0e-5
FIXED_E8_SOLVER_FTOL = 1.0e-12
FIXED_E8_SOLVER_MAXITER = 1_000

FIXED_E8_METHOD_ID = "COLD-FR-E8-NEWNLL-STRUCTSOFT-FUNCSOFT-FULLTAU"
FIXED_E8_TARGET_OVERLAY_DEFINITION = "R_l(z_k)+(z-z_k)"
FIXED_E8_TARGET_STEP_SEMANTICS = (
    "STEP_INDEPENDENT_FINITE_UNIT_FIELD_LOOKAHEAD"
)


class FixedE8Arm(str, Enum):
    NEUTRAL = "E8-NEUTRAL"
    SOFT = "E8-SOFT"


class FixedE8StepMode(str, Enum):
    JOINT_WRITE = "JOINT_WRITE"
    ZERO_WRITE_TARGET_RECOVERY = "ZERO_WRITE_TARGET_RECOVERY"


@dataclass(frozen=True, slots=True)
class FixedE8GridPoint:
    step_index: int
    tau_before: Fraction
    tau_after: Fraction
    h: Fraction = FIXED_E8_H

    def __post_init__(self) -> None:
        if (
            self.step_index < 0
            or self.step_index >= FIXED_E8_GRID_COUNT
            or self.h != FIXED_E8_H
            or self.tau_before != Fraction(self.step_index, FIXED_E8_GRID_COUNT)
            or self.tau_after
            != Fraction(self.step_index + 1, FIXED_E8_GRID_COUNT)
        ):
            raise ODEBFContractError("fixed E8 grid point differs")

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "tau_before": {
                "numerator": self.tau_before.numerator,
                "denominator": self.tau_before.denominator,
            },
            "tau_after": {
                "numerator": self.tau_after.numerator,
                "denominator": self.tau_after.denominator,
            },
            "h": {"numerator": self.h.numerator, "denominator": self.h.denominator},
        }


class FixedE8Clock:
    """Eight unconditional scientific grid transitions.

    There is intentionally no reject/retry/backtracking API.  A caller may
    fail closed on a technical exception before :meth:`advance`, but a finite
    scientific observation is always advanced exactly once.
    """

    __slots__ = ("_step", "_field_count", "_scientific_retry_count")

    def __init__(self) -> None:
        self._step = 0
        self._field_count = 0
        self._scientific_retry_count = 0

    @property
    def step_index(self) -> int:
        return self._step

    @property
    def tau(self) -> Fraction:
        return Fraction(self._step, FIXED_E8_GRID_COUNT)

    @property
    def complete(self) -> bool:
        return self._step == FIXED_E8_GRID_COUNT

    @property
    def field_count(self) -> int:
        return self._field_count

    @property
    def scientific_retry_count(self) -> int:
        return self._scientific_retry_count

    def begin_field(self) -> FixedE8GridPoint:
        if self.complete:
            raise ODEBFStateError("fixed E8 clock is already complete")
        if self._field_count != self._step:
            raise ODEBFStateError("fixed E8 field was already built at this grid state")
        self._field_count += 1
        return FixedE8GridPoint(
            self._step,
            Fraction(self._step, FIXED_E8_GRID_COUNT),
            Fraction(self._step + 1, FIXED_E8_GRID_COUNT),
        )

    def advance(
        self,
        point: FixedE8GridPoint,
        *,
        scientific_observation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if point.step_index != self._step or self._field_count != self._step + 1:
            raise ODEBFStateError("fixed E8 transition does not match the grid state")
        # Scientific values are intentionally not inspected.  Merely touching
        # this mapping would create a hidden decision path.
        del scientific_observation
        self._step += 1
        return {
            "mode": "FIXED_E8_GRID_ADVANCE",
            "grid_point": point.raw_free_payload(),
            "grid_count": self._step,
            "field_count": self._field_count,
            "tau_final": float(self.tau),
            "scientific_retry_count": 0,
            "scientific_rejection_count": 0,
            "adaptive_clock_access_count": 0,
            "first_hit_decision_influence_count": 0,
        }

    def terminal_receipt(self) -> dict[str, Any]:
        if not self.complete or self._field_count != FIXED_E8_GRID_COUNT:
            raise ODEBFStateError("fixed E8 trajectory is not complete")
        payload = {
            "schema": "ode-edit-fixed-e8-clock/v1",
            "grid_count": self._step,
            "field_count": self._field_count,
            "h": float(FIXED_E8_H),
            "tau_final": float(self.tau),
            "scientific_retry_count": self._scientific_retry_count,
            "adaptive_clock_access_count": 0,
            "backtracking_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


@dataclass(frozen=True, slots=True)
class FunctionalBasisMetric:
    """One bounded functional soft score measured on the common probe basis."""

    label: str
    baseline: float
    probe_endpoints: tuple[float, ...]
    active: bool
    inactive_reason: str | None = None

    def __post_init__(self) -> None:
        if self.label not in (
            "functional_p",
            "functional_h_mean",
            "functional_h_smoothmax",
        ):
            raise ODEBFContractError("fixed E8 functional metric label differs")
        if len(self.probe_endpoints) != len(FIXED_E8_LAYER_ORDER):
            raise ODEBFContractError("fixed E8 functional basis size differs")
        values = (self.baseline, *self.probe_endpoints)
        if any(not math.isfinite(float(value)) for value in values):
            raise ODEBFContractError("fixed E8 functional basis is non-finite")
        if self.active and self.inactive_reason is not None:
            raise ODEBFContractError("active fixed E8 functional metric has a reason")
        if not self.active and self.inactive_reason != "INACTIVE_EMPTY_HISTORY":
            raise ODEBFContractError("inactive fixed E8 functional metric differs")

    @property
    def signed_increment(self) -> np.ndarray:
        return np.asarray(self.probe_endpoints, dtype=np.float64) - float(
            self.baseline
        )

    @property
    def signed_secant(self) -> np.ndarray:
        return self.signed_increment / float(FIXED_E8_H)

    @property
    def positive_increment(self) -> np.ndarray:
        return np.maximum(self.signed_increment, 0.0)

    @property
    def normalization(self) -> float:
        return max(
            FIXED_E8_NORMALIZATION_EPSILON,
            float(np.sum(self.positive_increment)),
        )

    def score(self, velocity: Sequence[float]) -> float:
        if not self.active:
            return 0.0
        value = np.asarray(tuple(velocity), dtype=np.float64)
        if value.shape != (len(FIXED_E8_LAYER_ORDER),):
            raise ODEBFContractError("fixed E8 functional velocity shape differs")
        return float(self.positive_increment @ value / self.normalization)

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "active": self.active,
            "inactive_reason": self.inactive_reason,
            "baseline": self.baseline,
            "probe_endpoints": list(self.probe_endpoints),
            "raw_increment": self.signed_increment.tolist(),
            "signed_secant": self.signed_secant.tolist(),
            "positive_clipped_increment": self.positive_increment.tolist(),
            "normalization": self.normalization,
            "normalization_definition": (
                "max(epsilon,sum(max(probe_endpoint-baseline,0)))"
            ),
            "old_1e_3_hinge_access_count": 0,
        }


@dataclass(frozen=True, slots=True)
class FixedE8SoftInventory:
    functional_p: FunctionalBasisMetric
    functional_h_mean: FunctionalBasisMetric
    functional_h_smoothmax: FunctionalBasisMetric
    history_item_count: int
    controller_batch_sha256: str
    field_sha256: str

    def __post_init__(self) -> None:
        if (
            self.functional_p.label != "functional_p"
            or self.functional_h_mean.label != "functional_h_mean"
            or self.functional_h_smoothmax.label != "functional_h_smoothmax"
            or self.history_item_count < 0
            or len(self.controller_batch_sha256) != 64
            or len(self.field_sha256) != 64
        ):
            raise ODEBFContractError("fixed E8 functional inventory differs")
        expected_h_active = self.history_item_count > 0
        if (
            self.functional_h_mean.active != expected_h_active
            or self.functional_h_smoothmax.active != expected_h_active
            or not self.functional_p.active
        ):
            raise ODEBFContractError("fixed E8 functional activity differs")

    @property
    def exact_basis_endpoint_count(self) -> int:
        # One shared baseline plus one endpoint for each of five layers.  H
        # reuses these endpoints and never adds its own model evaluations.
        return 1 + len(FIXED_E8_LAYER_ORDER)

    def active_metrics(self) -> tuple[FunctionalBasisMetric, ...]:
        return tuple(
            metric
            for metric in (
                self.functional_p,
                self.functional_h_mean,
                self.functional_h_smoothmax,
            )
            if metric.active
        )

    def raw_free_payload(self) -> dict[str, Any]:
        payload = {
            "history_item_count": self.history_item_count,
            "controller_batch_sha256": self.controller_batch_sha256,
            "field_sha256": self.field_sha256,
            "basis_endpoint_count": self.exact_basis_endpoint_count,
            "functional_h_probe_endpoint_count": 0,
            "common_baseline_and_layer_probe_overlay": True,
            "metrics": [
                metric.raw_free_payload()
                for metric in (
                    self.functional_p,
                    self.functional_h_mean,
                    self.functional_h_smoothmax,
                )
            ],
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


@dataclass(frozen=True, slots=True)
class FixedE8Score:
    label: str
    score: float
    active: bool
    normalization: float
    raw_delta: float
    positive_delta: float
    influence_count: int

    def __post_init__(self) -> None:
        if (
            not self.label
            or any(
                not math.isfinite(value)
                for value in (
                    self.score,
                    self.normalization,
                    self.raw_delta,
                    self.positive_delta,
                )
            )
            or self.normalization <= 0.0
            or self.score < -FIXED_E8_PRIMAL_TOLERANCE
            or self.influence_count not in (0, 1)
        ):
            raise ODEBFContractError("fixed E8 normalized score differs")


@dataclass(frozen=True, slots=True)
class FixedE8SolverCertificate:
    phase: str
    success: bool
    iterations: int
    optimizer_pass_count: int
    maximum_primal_violation: float
    stationarity_residual: float
    complementarity_residual: float
    signed_progress: float
    trust_value: float
    p_max: float
    requested_progress: float
    xi: float | None
    first_false_component: str | None
    active_constraints: tuple[str, ...]
    passed: bool

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        continuation_count = max(self.optimizer_pass_count - 1, 0)
        payload.update(
            {
                "numerical_backend_continuation_count": continuation_count,
                "numerical_backend_continuation_role": (
                    "FIXED_NUMERICAL_BACKEND_CONTINUATION_SAME_QP"
                    if continuation_count
                    else "NONE"
                ),
                "scientific_retry_count": 0,
                "field_rebuild_count": 0,
                "candidate_evaluation_count": 0,
            }
        )
        return payload


FixedE8CertificateObserver = Callable[[FixedE8SolverCertificate], None]


@dataclass(frozen=True, slots=True)
class FixedE8RoutingResult:
    arm: FixedE8Arm
    mode: FixedE8StepMode
    signed_slopes: tuple[float, ...]
    active_direction_mask: tuple[bool, ...]
    p_max_velocity: tuple[float, ...]
    p_max: float
    requested_progress: float
    pre_soft_velocity: tuple[float, ...]
    soft_velocity: tuple[float, ...]
    velocity: tuple[float, ...]
    applied_coefficient: tuple[float, ...]
    scores: tuple[FixedE8Score, ...]
    xi_star: float | None
    certificates: tuple[FixedE8SolverCertificate, ...]
    identity_sha256: str

    def __post_init__(self) -> None:
        dimension = len(FIXED_E8_LAYER_ORDER)
        arrays = (
            self.signed_slopes,
            self.active_direction_mask,
            self.p_max_velocity,
            self.pre_soft_velocity,
            self.soft_velocity,
            self.velocity,
            self.applied_coefficient,
        )
        if any(len(value) != dimension for value in arrays):
            raise ODEBFContractError("fixed E8 routing result dimension differs")
        if any(
            not math.isfinite(value)
            for value in (
                *self.signed_slopes,
                *self.p_max_velocity,
                self.p_max,
                self.requested_progress,
                *self.pre_soft_velocity,
                *self.soft_velocity,
                *self.velocity,
                *self.applied_coefficient,
                *(() if self.xi_star is None else (self.xi_star,)),
            )
        ):
            raise ODEBFContractError("fixed E8 routing result is non-finite")
        expected_theta = tuple(float(FIXED_E8_H) * item for item in self.velocity)
        if not np.allclose(
            np.asarray(self.applied_coefficient),
            np.asarray(expected_theta),
            rtol=0.0,
            atol=1.0e-14,
        ):
            raise ODEBFContractError("fixed E8 applied coefficient used h incorrectly")
        if not all(item.passed for item in self.certificates):
            raise ODEBFContractError("fixed E8 routing certificate failed")
        if len(self.identity_sha256) != 64:
            raise ODEBFContractError("fixed E8 routing identity differs")

    def raw_free_payload(self) -> dict[str, Any]:
        logical = len(self.certificates)
        backend = sum(item.optimizer_pass_count for item in self.certificates)
        continuation = sum(
            max(item.optimizer_pass_count - 1, 0)
            for item in self.certificates
        )
        maximum_schedule = bool(
            self.mode is FixedE8StepMode.JOINT_WRITE
            and logical == 4
            and backend == 5
            and continuation == 1
        )
        return {
            "arm": self.arm.value,
            "mode": self.mode.value,
            "layer_order": list(FIXED_E8_LAYER_ORDER),
            "signed_slopes": list(self.signed_slopes),
            "active_direction_mask": list(self.active_direction_mask),
            "p_max_velocity": list(self.p_max_velocity),
            "p_max": self.p_max,
            "kappa": FIXED_E8_KAPPA,
            "requested_progress": self.requested_progress,
            "pre_soft_velocity": list(self.pre_soft_velocity),
            "soft_velocity": list(self.soft_velocity),
            "velocity": list(self.velocity),
            "applied_coefficient": list(self.applied_coefficient),
            "velocity_definition": "v_l",
            "applied_coefficient_definition": "theta_l=h*v_l",
            "scores": [asdict(item) for item in self.scores],
            "xi_star": self.xi_star,
            "selected_velocity_role": (
                "PRE_SOFT_NEUTRAL"
                if self.arm is FixedE8Arm.NEUTRAL
                else "JOINT_SOFT_LEXICOGRAPHIC"
            ),
            "matched_solver_schedule": maximum_schedule,
            "solver_schedule_kind": (
                "FULL_LEXICOGRAPHIC_MAXIMUM_SCHEDULE"
                if maximum_schedule
                else "ZERO_WRITE_REDUCED_TECHNICAL_SCHEDULE"
            ),
            "actual_logical_qp_count": logical,
            "actual_optimizer_backend_invocation_count": backend,
            "actual_numerical_backend_continuation_count": continuation,
            "static_operation_counts_are_maximum_ceiling": True,
            "soft_shadow_decision_influence_count": (
                0 if self.arm is FixedE8Arm.NEUTRAL else 1
            ),
            "certificates": [item.raw_free_payload() for item in self.certificates],
            "identity_sha256": self.identity_sha256,
        }


Constraint = tuple[
    str,
    Callable[[np.ndarray], float],
    Callable[[np.ndarray], np.ndarray],
]


def _validate_problem(problem: RoutingProblem) -> None:
    if problem.signed_progress.shape != (len(FIXED_E8_LAYER_ORDER),):
        raise ODEBFContractError("fixed E8 routing layer dimension differs")
    if np.any(problem.layer_caps > 1.0 + FIXED_E8_PRIMAL_TOLERANCE):
        raise ODEBFContractError("fixed E8 layer cap exceeds one")
    if np.any(problem.layer_caps <= 0.0):
        raise ODEBFContractError("fixed E8 layer cap is nonpositive")


def _technical_constraints(
    problem: RoutingProblem,
    active: np.ndarray,
    *,
    requested_progress: float | None,
) -> list[Constraint]:
    progress = problem.signed_progress[active]
    trust = problem.trust_metric[np.ix_(active, active)]
    constraints: list[Constraint] = [
        (
            "technical_write_trust",
            lambda value, matrix=trust, radius=problem.trust_radius: float(
                radius**2 - value @ matrix @ value
            ),
            lambda value, matrix=trust: -2.0 * matrix @ value,
        )
    ]
    if requested_progress is not None:
        constraints.insert(
            0,
            (
                "kappa_progress",
                lambda value, a=progress, target=requested_progress: float(
                    a @ value - target
                ),
                lambda value, a=progress: a.copy(),
            ),
        )
    return constraints


def _certificate(
    *,
    phase: str,
    result: Any,
    value: np.ndarray,
    objective_gradient: np.ndarray,
    constraints: Sequence[Constraint],
    lower: np.ndarray,
    upper: np.ndarray,
    expanded_velocity: np.ndarray,
    problem: RoutingProblem,
    p_max: float,
    requested_progress: float,
    xi: float | None,
    optimizer_pass_count: int,
) -> FixedE8SolverCertificate:
    slacks: list[tuple[str, float]] = []
    active_gradients: list[np.ndarray] = []
    active_slacks: list[float] = []
    for name, function, jacobian in constraints:
        slack = float(function(value))
        slacks.append((name, slack))
        if slack <= 10.0 * FIXED_E8_PRIMAL_TOLERANCE:
            active_gradients.append(np.asarray(jacobian(value), dtype=np.float64))
            active_slacks.append(slack)
    for index in range(value.size):
        for name, slack, direction in (
            (f"lower_{index}", float(value[index] - lower[index]), 1.0),
            (f"upper_{index}", float(upper[index] - value[index]), -1.0),
        ):
            slacks.append((name, slack))
            if slack <= 10.0 * FIXED_E8_PRIMAL_TOLERANCE:
                basis = np.zeros_like(value)
                basis[index] = direction
                active_gradients.append(basis)
                active_slacks.append(slack)
    maximum_violation = max(
        0.0, max((-slack for _, slack in slacks), default=0.0)
    )
    if active_gradients:
        matrix = np.stack(active_gradients, axis=1)
        multipliers, stationarity = nnls(matrix, objective_gradient)
        stationarity /= max(float(np.linalg.norm(objective_gradient)), 1.0)
        complementarity = max(
            (
                abs(float(multiplier * slack))
                for multiplier, slack in zip(
                    multipliers, active_slacks, strict=True
                )
            ),
            default=0.0,
        )
    else:
        stationarity = float(np.linalg.norm(objective_gradient)) / max(
            float(np.linalg.norm(objective_gradient)), 1.0
        )
        complementarity = 0.0
    first_false = next(
        (name for name, slack in slacks if slack < -FIXED_E8_PRIMAL_TOLERANCE),
        None,
    )
    success = bool(getattr(result, "success", False))
    passed = bool(
        success
        and maximum_violation <= FIXED_E8_PRIMAL_TOLERANCE
        and stationarity <= FIXED_E8_KKT_TOLERANCE
        and complementarity <= FIXED_E8_KKT_TOLERANCE
    )
    return FixedE8SolverCertificate(
        phase=phase,
        success=success,
        iterations=int(getattr(result, "nit", 0)),
        optimizer_pass_count=optimizer_pass_count,
        maximum_primal_violation=maximum_violation,
        stationarity_residual=float(stationarity),
        complementarity_residual=float(complementarity),
        signed_progress=float(problem.signed_progress @ expanded_velocity),
        trust_value=float(expanded_velocity @ problem.trust_metric @ expanded_velocity),
        p_max=p_max,
        requested_progress=requested_progress,
        xi=xi,
        first_false_component=first_false,
        active_constraints=tuple(
            name
            for name, slack in slacks
            if abs(slack) <= 10.0 * FIXED_E8_PRIMAL_TOLERANCE
        ),
        passed=passed,
    )


def _solve_slsqp(
    *,
    phase: str,
    problem: RoutingProblem,
    active: np.ndarray,
    objective: Callable[[np.ndarray], float],
    jacobian: Callable[[np.ndarray], np.ndarray],
    constraints: Sequence[Constraint],
    lower: np.ndarray,
    upper: np.ndarray,
    initial: np.ndarray,
    p_max: float,
    requested_progress: float,
    xi: float | None,
    polish_once: bool = False,
    certificate_observer: FixedE8CertificateObserver | None = None,
) -> tuple[np.ndarray, FixedE8SolverCertificate]:
    scipy_constraints = [
        {"type": "ineq", "fun": function, "jac": derivative}
        for _, function, derivative in constraints
    ]
    result = minimize(
        objective,
        np.asarray(initial, dtype=np.float64),
        jac=jacobian,
        method="SLSQP",
        bounds=tuple(
            (float(left), float(right))
            for left, right in zip(lower, upper, strict=True)
        ),
        constraints=scipy_constraints,
        options={
            "ftol": FIXED_E8_SOLVER_FTOL,
            "maxiter": FIXED_E8_SOLVER_MAXITER,
            "disp": False,
        },
    )
    optimizer_pass_count = 1
    if polish_once:
        # The lexicographic stage-2 optimum can sit at the intersection of
        # several quadratic xi-tie constraints.  Continue SLSQP once from its
        # own converged point so the independently recomputed KKT certificate
        # reaches the locked tolerance without relaxing that tolerance or any
        # scientific constraint.  This is numerical polishing inside one QP,
        # not an E8 trajectory retry or a new field/candidate evaluation.
        first_iterations = int(getattr(result, "nit", 0))
        result = minimize(
            objective,
            np.asarray(result.x, dtype=np.float64),
            jac=jacobian,
            method="SLSQP",
            bounds=tuple(
                (float(left), float(right))
                for left, right in zip(lower, upper, strict=True)
            ),
            constraints=scipy_constraints,
            options={
                "ftol": FIXED_E8_SOLVER_FTOL,
                "maxiter": FIXED_E8_SOLVER_MAXITER,
                "disp": False,
            },
        )
        result.nit = first_iterations + int(getattr(result, "nit", 0))
        optimizer_pass_count = 2
    value = np.asarray(result.x, dtype=np.float64)
    expanded = np.zeros(problem.signed_progress.size, dtype=np.float64)
    expanded[active] = value[: active.size]
    observed_xi = xi
    if value.size != active.size:
        if value.size != active.size + 1:
            raise ODEBFContractError("fixed E8 solver auxiliary dimension differs")
        observed_xi = float(value[active.size])
    certificate = _certificate(
        phase=phase,
        result=result,
        value=value,
        objective_gradient=np.asarray(jacobian(value), dtype=np.float64),
        constraints=constraints,
        lower=lower,
        upper=upper,
        expanded_velocity=expanded,
        problem=problem,
        p_max=p_max,
        requested_progress=requested_progress,
        xi=observed_xi,
        optimizer_pass_count=optimizer_pass_count,
    )
    if not certificate.passed:
        if certificate_observer is not None:
            certificate_observer(certificate)
        raise ODEBFContractError(
            f"fixed E8 {phase} solver certificate failed: "
            f"{certificate.raw_free_payload()}"
        )
    return value, certificate


def structural_soft_score(
    barrier: QuadraticBarrier,
    velocity: Sequence[float],
    *,
    active: bool,
    influence_count: int,
) -> FixedE8Score:
    value = np.asarray(tuple(velocity), dtype=np.float64)
    if value.shape != barrier.linear.shape:
        raise ODEBFContractError("fixed E8 structural velocity shape differs")
    raw_delta = float(structural_contribution_vector(barrier, value).sum())
    positive_delta = max(raw_delta, 0.0)
    normalization = max(
        FIXED_E8_NORMALIZATION_EPSILON, float(np.trace(barrier.gram))
    )
    return FixedE8Score(
        label=f"structural_{barrier.label}",
        score=0.0 if not active else positive_delta / normalization,
        active=active,
        normalization=normalization,
        raw_delta=raw_delta,
        positive_delta=positive_delta,
        influence_count=influence_count if active else 0,
    )


def structural_contribution_vector(
    barrier: QuadraticBarrier,
    velocity: Sequence[float],
) -> np.ndarray:
    """Per-layer attribution for an already h/h²-scaled quadratic."""

    value = np.asarray(tuple(velocity), dtype=np.float64)
    if value.shape != barrier.linear.shape:
        raise ODEBFContractError("fixed E8 structural contribution shape differs")
    return 2.0 * barrier.linear * value + value * (barrier.gram @ value)


def functional_soft_score(
    metric: FunctionalBasisMetric,
    velocity: Sequence[float],
    *,
    influence_count: int,
) -> FixedE8Score:
    value = np.asarray(tuple(velocity), dtype=np.float64)
    if value.shape != metric.positive_increment.shape:
        raise ODEBFContractError("fixed E8 functional velocity shape differs")
    raw_delta = float(metric.signed_increment @ value)
    positive_delta = float(metric.positive_increment @ value)
    return FixedE8Score(
        label=metric.label,
        score=0.0 if not metric.active else positive_delta / metric.normalization,
        active=metric.active,
        normalization=metric.normalization,
        raw_delta=raw_delta,
        positive_delta=positive_delta,
        influence_count=influence_count if metric.active else 0,
    )


def _structural_score_constraint(
    label: str,
    barrier: QuadraticBarrier,
    active: np.ndarray,
    *,
    xi_index: int,
) -> Constraint:
    linear = barrier.linear[active]
    gram = barrier.gram[np.ix_(active, active)]
    normalization = max(
        FIXED_E8_NORMALIZATION_EPSILON, float(np.trace(barrier.gram))
    )

    def slack(value: np.ndarray) -> float:
        velocity = value[: active.size]
        raw = float(2.0 * linear @ velocity + velocity @ gram @ velocity)
        return float(value[xi_index] - raw / normalization)

    def derivative(value: np.ndarray) -> np.ndarray:
        velocity = value[: active.size]
        result = np.zeros_like(value)
        result[: active.size] = -(2.0 * linear + 2.0 * gram @ velocity) / normalization
        result[xi_index] = 1.0
        return result

    return label, slack, derivative


def _functional_score_constraint(
    metric: FunctionalBasisMetric,
    active: np.ndarray,
    *,
    xi_index: int,
) -> Constraint:
    positive = metric.positive_increment[active]

    def slack(value: np.ndarray) -> float:
        return float(
            value[xi_index]
            - positive @ value[: active.size] / metric.normalization
        )

    def derivative(value: np.ndarray) -> np.ndarray:
        del value
        result = np.zeros(active.size + 1, dtype=np.float64)
        result[: active.size] = -positive / metric.normalization
        result[xi_index] = 1.0
        return result

    return metric.label, slack, derivative


def _maximum_progress(
    problem: RoutingProblem,
    active: np.ndarray,
    *,
    certificate_observer: FixedE8CertificateObserver | None = None,
) -> tuple[np.ndarray, float, FixedE8SolverCertificate]:
    progress = problem.signed_progress[active]
    caps = np.minimum(problem.layer_caps[active], 1.0)
    constraints = _technical_constraints(problem, active, requested_progress=None)
    initial = np.zeros(active.size, dtype=np.float64)
    value, certificate = _solve_slsqp(
        phase="maximum-technical-progress",
        problem=problem,
        active=active,
        objective=lambda item, a=progress: float(-a @ item),
        jacobian=lambda item, a=progress: -a.copy(),
        constraints=constraints,
        lower=np.zeros(active.size, dtype=np.float64),
        upper=caps,
        initial=initial,
        p_max=0.0,
        requested_progress=0.0,
        xi=None,
        certificate_observer=certificate_observer,
    )
    expanded = np.zeros(problem.signed_progress.size, dtype=np.float64)
    expanded[active] = value
    p_max = float(problem.signed_progress @ expanded)
    if not math.isfinite(p_max) or p_max < -FIXED_E8_PRIMAL_TOLERANCE:
        raise ODEBFContractError("fixed E8 maximum progress differs")
    certificate = FixedE8SolverCertificate(
        **{
            **asdict(certificate),
            "p_max": max(p_max, 0.0),
        }
    )
    if certificate_observer is not None:
        certificate_observer(certificate)
    return expanded, max(p_max, 0.0), certificate


def solve_fixed_e8_routing(
    problem: RoutingProblem,
    inventory: FixedE8SoftInventory,
    *,
    arm: FixedE8Arm | str,
    certificate_observer: FixedE8CertificateObserver | None = None,
) -> FixedE8RoutingResult:
    """Solve one fixed-grid E8 routing field.

    Structural and functional risks never appear in the technical feasible
    set.  ``E8-NEUTRAL`` minimizes capacity under the same progress/trust set;
    ``E8-SOFT`` first minimizes the worst normalized soft score and then the
    same capacity objective within the sealed xi tie tolerance.
    """

    selected = FixedE8Arm(arm)
    _validate_problem(problem)
    if inventory.field_sha256 == "0" * 64:
        raise ODEBFContractError("fixed E8 functional inventory field is absent")
    active = np.flatnonzero(problem.signed_progress > 0.0)
    signed = tuple(float(item) for item in problem.signed_progress)
    mask = tuple(bool(item > 0.0) for item in problem.signed_progress)
    if active.size == 0:
        return _zero_write_result(problem, inventory, selected, signed, mask)
    maximum, p_max, maximum_certificate = _maximum_progress(
        problem,
        active,
        certificate_observer=certificate_observer,
    )
    if p_max <= FIXED_E8_NORMALIZATION_EPSILON:
        return _zero_write_result(
            problem,
            inventory,
            selected,
            signed,
            mask,
            maximum=maximum,
            certificate=maximum_certificate,
        )
    requested = FIXED_E8_KAPPA * p_max
    technical = _technical_constraints(
        problem, active, requested_progress=requested
    )
    caps = np.minimum(problem.layer_caps[active], 1.0)
    scale = requested / p_max
    initial_velocity = maximum[active] * scale
    metric = problem.capacity_metric[np.ix_(active, active)]
    certificates: list[FixedE8SolverCertificate] = [maximum_certificate]
    neutral_active, capacity_certificate = _solve_slsqp(
        phase="neutral-minimum-capacity",
        problem=problem,
        active=active,
        objective=lambda value, matrix=metric: float(
            0.5 * value @ matrix @ value
        ),
        jacobian=lambda value, matrix=metric: matrix @ value,
        constraints=technical,
        lower=np.zeros(active.size, dtype=np.float64),
        upper=caps,
        initial=initial_velocity,
        p_max=p_max,
        requested_progress=requested,
        xi=None,
        certificate_observer=certificate_observer,
    )
    if certificate_observer is not None:
        certificate_observer(capacity_certificate)
    certificates.append(capacity_certificate)
    xi_index = active.size
    stage1_constraints: list[Constraint] = []
    for name, function, derivative in technical:
        stage1_constraints.append(
            (
                name,
                lambda value, f=function: f(value[:active.size]),
                lambda value, d=derivative: np.concatenate(
                    (d(value[:active.size]), np.zeros(1, dtype=np.float64))
                ),
            )
        )
    if inventory.history_item_count > 0:
        stage1_constraints.append(
            _structural_score_constraint(
                "soft_structural_historical",
                problem.historical,
                active,
                xi_index=xi_index,
            )
        )
    stage1_constraints.append(
        _structural_score_constraint(
            "soft_structural_pretrained",
            problem.pretrained,
            active,
            xi_index=xi_index,
        )
    )
    for functional in inventory.active_metrics():
        stage1_constraints.append(
            _functional_score_constraint(functional, active, xi_index=xi_index)
        )
    neutral_expanded = _expand(
        active, neutral_active, problem.signed_progress.size
    )
    initial_scores = _score_inventory(
        problem,
        inventory,
        neutral_expanded,
        influence_count=1,
    )
    initial_xi = max(
        (item.score for item in initial_scores if item.active), default=0.0
    )
    initial_stage1 = np.concatenate((neutral_active, [initial_xi]))
    stage1_value, stage1_certificate = _solve_slsqp(
        phase="soft-minimum-worst-normalized-score",
        problem=problem,
        active=active,
        objective=lambda value: float(value[xi_index]),
        jacobian=lambda value: np.eye(
            1, value.size, xi_index, dtype=np.float64
        ).reshape(-1),
        constraints=stage1_constraints,
        lower=np.concatenate((np.zeros(active.size), [0.0])),
        upper=np.concatenate((caps, [np.inf])),
        initial=initial_stage1,
        p_max=p_max,
        requested_progress=requested,
        xi=initial_xi,
        certificate_observer=certificate_observer,
    )
    if certificate_observer is not None:
        certificate_observer(stage1_certificate)
    xi_star = max(float(stage1_value[xi_index]), 0.0)
    certificates.append(stage1_certificate)
    stage2_constraints: list[Constraint] = list(technical)
    xi_cap = xi_star + FIXED_E8_XI_TIE_TOLERANCE

    def append_structural_cap(label: str, barrier: QuadraticBarrier) -> None:
        linear = barrier.linear[active]
        gram = barrier.gram[np.ix_(active, active)]
        normalization = max(
            FIXED_E8_NORMALIZATION_EPSILON,
            float(np.trace(barrier.gram)),
        )
        stage2_constraints.append(
            (
                label,
                lambda value, lin=linear, matrix=gram, scale=normalization: float(
                    xi_cap
                    - (2.0 * lin @ value + value @ matrix @ value) / scale
                ),
                lambda value, lin=linear, matrix=gram, scale=normalization: -(
                    2.0 * lin + 2.0 * matrix @ value
                )
                / scale,
            )
        )

    if inventory.history_item_count > 0:
        append_structural_cap(
            "xi_tie_structural_historical", problem.historical
        )
    append_structural_cap("xi_tie_structural_pretrained", problem.pretrained)
    for functional in inventory.active_metrics():
        positive = functional.positive_increment[active]
        normalization = functional.normalization
        stage2_constraints.append(
            (
                f"xi_tie_{functional.label}",
                lambda value, slope=positive, scale=normalization: float(
                    xi_cap - slope @ value / scale
                ),
                lambda value, slope=positive, scale=normalization: -slope
                / scale,
            )
        )
    soft_active, stage2_certificate = _solve_slsqp(
        phase="soft-minimum-capacity-within-xi-tie",
        problem=problem,
        active=active,
        objective=lambda value, matrix=metric: float(
            0.5 * value @ matrix @ value
        ),
        jacobian=lambda value, matrix=metric: matrix @ value,
        constraints=stage2_constraints,
        lower=np.zeros(active.size, dtype=np.float64),
        upper=caps,
        initial=stage1_value[:active.size],
        p_max=p_max,
        requested_progress=requested,
        xi=xi_star,
        polish_once=True,
        certificate_observer=certificate_observer,
    )
    if certificate_observer is not None:
        certificate_observer(stage2_certificate)
    certificates.append(stage2_certificate)
    pre_soft_velocity = neutral_expanded
    soft_velocity = _expand(
        active, soft_active, problem.signed_progress.size
    )
    velocity = (
        pre_soft_velocity
        if selected is FixedE8Arm.NEUTRAL
        else soft_velocity
    )
    scores = _score_inventory(
        problem,
        inventory,
        velocity,
        influence_count=1 if selected is FixedE8Arm.SOFT else 0,
    )
    payload = {
        "method_id": FIXED_E8_METHOD_ID,
        "arm": selected.value,
        "mode": FixedE8StepMode.JOINT_WRITE.value,
        "problem_sha256": problem.identity(),
        "functional_inventory_sha256": inventory.raw_free_payload()[
            "identity_sha256"
        ],
        "signed_slopes": list(signed),
        "active_direction_mask": list(mask),
        "p_max_velocity": maximum.tolist(),
        "p_max": p_max,
        "kappa": FIXED_E8_KAPPA,
        "requested_progress": requested,
        "pre_soft_velocity": pre_soft_velocity.tolist(),
        "soft_velocity": soft_velocity.tolist(),
        "velocity": velocity.tolist(),
        "applied_coefficient": (float(FIXED_E8_H) * velocity).tolist(),
        "scores": [asdict(item) for item in scores],
        "xi_star": xi_star,
        "certificates": [item.raw_free_payload() for item in certificates],
        "hard_structural_h_budget_influence_count": 0,
        "hard_structural_p_budget_influence_count": 0,
        "functional_candidate_veto_influence_count": 0,
        "write_trust_role": "TECHNICAL_INTEGRATION_BOUND",
        "selected_velocity_role": (
            "PRE_SOFT_NEUTRAL"
            if selected is FixedE8Arm.NEUTRAL
            else "JOINT_SOFT_LEXICOGRAPHIC"
        ),
        "matched_solver_schedule": True,
        "solver_schedule_kind": "FULL_LEXICOGRAPHIC_MAXIMUM_SCHEDULE",
        "actual_logical_qp_count": len(certificates),
        "actual_optimizer_backend_invocation_count": sum(
            item.optimizer_pass_count for item in certificates
        ),
        "actual_numerical_backend_continuation_count": sum(
            max(item.optimizer_pass_count - 1, 0) for item in certificates
        ),
        "static_operation_counts_are_maximum_ceiling": True,
        "soft_shadow_decision_influence_count": (
            0 if selected is FixedE8Arm.NEUTRAL else 1
        ),
    }
    identity = canonical_hash(payload)
    return FixedE8RoutingResult(
        selected,
        FixedE8StepMode.JOINT_WRITE,
        signed,
        mask,
        tuple(float(item) for item in maximum),
        p_max,
        requested,
        tuple(float(item) for item in pre_soft_velocity),
        tuple(float(item) for item in soft_velocity),
        tuple(float(item) for item in velocity),
        tuple(float(FIXED_E8_H) * float(item) for item in velocity),
        scores,
        xi_star,
        tuple(certificates),
        identity,
    )


def _zero_write_result(
    problem: RoutingProblem,
    inventory: FixedE8SoftInventory,
    arm: FixedE8Arm,
    signed: tuple[float, ...],
    mask: tuple[bool, ...],
    *,
    maximum: np.ndarray | None = None,
    certificate: FixedE8SolverCertificate | None = None,
) -> FixedE8RoutingResult:
    zero = np.zeros(problem.signed_progress.size, dtype=np.float64)
    maximum_value = zero if maximum is None else np.asarray(maximum, dtype=np.float64)
    scores = _score_inventory(problem, inventory, zero, influence_count=0)
    certificates = () if certificate is None else (certificate,)
    payload = {
        "method_id": FIXED_E8_METHOD_ID,
        "arm": arm.value,
        "mode": FixedE8StepMode.ZERO_WRITE_TARGET_RECOVERY.value,
        "problem_sha256": problem.identity(),
        "functional_inventory_sha256": inventory.raw_free_payload()[
            "identity_sha256"
        ],
        "signed_slopes": list(signed),
        "active_direction_mask": list(mask),
        "p_max_velocity": maximum_value.tolist(),
        "p_max": 0.0,
        "requested_progress": 0.0,
        "pre_soft_velocity": zero.tolist(),
        "soft_velocity": zero.tolist(),
        "velocity": zero.tolist(),
        "applied_coefficient": zero.tolist(),
        "scores": [asdict(item) for item in scores],
        "xi_star": None,
        "certificates": [item.raw_free_payload() for item in certificates],
        "target_only_recovery_consumes_grid_interval": True,
        "retry_count": 0,
        "matched_solver_schedule": False,
        "solver_schedule_kind": "ZERO_WRITE_REDUCED_TECHNICAL_SCHEDULE",
        "actual_logical_qp_count": len(certificates),
        "actual_optimizer_backend_invocation_count": sum(
            item.optimizer_pass_count for item in certificates
        ),
        "actual_numerical_backend_continuation_count": sum(
            max(item.optimizer_pass_count - 1, 0) for item in certificates
        ),
        "static_operation_counts_are_maximum_ceiling": True,
    }
    return FixedE8RoutingResult(
        arm,
        FixedE8StepMode.ZERO_WRITE_TARGET_RECOVERY,
        signed,
        mask,
        tuple(float(item) for item in maximum_value),
        0.0,
        0.0,
        tuple(0.0 for _ in signed),
        tuple(0.0 for _ in signed),
        tuple(0.0 for _ in signed),
        tuple(0.0 for _ in signed),
        scores,
        None,
        certificates,
        canonical_hash(payload),
    )


def _expand(active: np.ndarray, value: np.ndarray, dimension: int) -> np.ndarray:
    result = np.zeros(dimension, dtype=np.float64)
    result[active] = value
    return result


def _score_inventory(
    problem: RoutingProblem,
    inventory: FixedE8SoftInventory,
    velocity: np.ndarray,
    *,
    influence_count: int,
) -> tuple[FixedE8Score, ...]:
    historical_active = inventory.history_item_count > 0
    return (
        structural_soft_score(
            problem.historical,
            velocity,
            active=historical_active,
            influence_count=influence_count,
        ),
        structural_soft_score(
            problem.pretrained,
            velocity,
            active=True,
            influence_count=influence_count,
        ),
        *tuple(
            functional_soft_score(
                metric, velocity, influence_count=influence_count
            )
            for metric in (
                inventory.functional_p,
                inventory.functional_h_mean,
                inventory.functional_h_smoothmax,
            )
        ),
    )


def expanded_quadratic_delta(
    *,
    linear_scaled_by_h: Sequence[float],
    gram_scaled_by_h2: Sequence[Sequence[float]],
    velocity: Sequence[float],
) -> float:
    """Evaluate an already step-scaled quadratic exactly once."""

    linear = np.asarray(tuple(linear_scaled_by_h), dtype=np.float64)
    gram = np.asarray(tuple(tuple(row) for row in gram_scaled_by_h2), dtype=np.float64)
    value = np.asarray(tuple(velocity), dtype=np.float64)
    if linear.shape != value.shape or gram.shape != (value.size, value.size):
        raise ODEBFContractError("fixed E8 quadratic geometry differs")
    return float(2.0 * linear @ value + value @ gram @ value)


def direct_low_rank_delta(
    *,
    baseline: Sequence[float],
    layer_updates: Sequence[Sequence[float]],
    velocity: Sequence[float],
    h: Fraction = FIXED_E8_H,
) -> float:
    """Direct norm delta using ``theta=h*v`` for unit regression fixtures."""

    base = np.asarray(tuple(baseline), dtype=np.float64)
    updates = np.asarray(tuple(tuple(row) for row in layer_updates), dtype=np.float64)
    value = np.asarray(tuple(velocity), dtype=np.float64)
    if updates.ndim != 2 or updates.shape[0] != value.size or updates.shape[1] != base.size:
        raise ODEBFContractError("fixed E8 direct low-rank geometry differs")
    theta = float(h) * value
    candidate = base + theta @ updates
    return float(candidate @ candidate - base @ base)


def expanded_terms_from_low_rank(
    *,
    baseline: Sequence[float],
    layer_updates: Sequence[Sequence[float]],
    h: Fraction = FIXED_E8_H,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(tuple(baseline), dtype=np.float64)
    updates = np.asarray(tuple(tuple(row) for row in layer_updates), dtype=np.float64)
    if updates.ndim != 2 or updates.shape[1] != base.size:
        raise ODEBFContractError("fixed E8 expanded low-rank geometry differs")
    step = float(h)
    return step * (updates @ base), step**2 * (updates @ updates.T)


@dataclass(frozen=True, slots=True)
class FixedE8OperationCeiling:
    arm_count: int = 2
    fields_per_arm: int = FIXED_E8_GRID_COUNT
    candidates_per_arm: int = FIXED_E8_GRID_COUNT
    functional_basis_endpoints_per_field: int = 6
    functional_h_extra_endpoints_per_field: int = 0
    controller_candidate_functional_endpoints_per_arm: int = 8
    terminal_audit_functional_endpoints_per_arm: int = 8
    action_rewrite_evaluations_per_arm: int = 9
    field_backward_batches_per_arm: int = 8
    target_backward_batches_per_arm: int = 8
    qp_solves_per_field: int = 4
    qp_backend_invocations_per_field: int = 5
    scientific_retries_per_arm: int = 0

    def __post_init__(self) -> None:
        if (
            self.arm_count != 2
            or self.fields_per_arm != 8
            or self.candidates_per_arm != 8
            or self.functional_basis_endpoints_per_field != 6
            or self.functional_h_extra_endpoints_per_field != 0
            or self.controller_candidate_functional_endpoints_per_arm != 8
            or self.terminal_audit_functional_endpoints_per_arm != 8
            or self.action_rewrite_evaluations_per_arm != 9
            or self.field_backward_batches_per_arm != 8
            or self.target_backward_batches_per_arm != 8
            or self.qp_solves_per_field != 4
            or self.qp_backend_invocations_per_field != 5
            or self.scientific_retries_per_arm != 0
        ):
            raise ODEBFContractError("fixed E8 operation ceiling differs")

    def raw_free_payload(self) -> dict[str, Any]:
        per_arm = {
            "field_count": self.fields_per_arm,
            "candidate_count": self.candidates_per_arm,
            "functional_basis_endpoint_count": (
                self.fields_per_arm * self.functional_basis_endpoints_per_field
            ),
            "functional_h_extra_endpoint_count": 0,
            "controller_candidate_functional_endpoint_count": (
                self.controller_candidate_functional_endpoints_per_arm
            ),
            "terminal_audit_functional_endpoint_count": (
                self.terminal_audit_functional_endpoints_per_arm
            ),
            "action_rewrite_evaluation_count": (
                self.action_rewrite_evaluations_per_arm
            ),
            "field_backward_batch_count": self.field_backward_batches_per_arm,
            "target_backward_batch_count": self.target_backward_batches_per_arm,
            "qp_solve_count": self.fields_per_arm * self.qp_solves_per_field,
            "qp_backend_invocation_count": (
                self.fields_per_arm * self.qp_backend_invocations_per_field
            ),
            "stage2_numerical_polish_count": self.fields_per_arm,
            "scientific_retry_count": 0,
        }
        return {
            "per_arm": per_arm,
            "two_arm": {
                key: value * self.arm_count for key, value in per_arm.items()
            },
            "matched_compute_schedule": True,
        }


def fixed_e8_semantic_receipt() -> dict[str, Any]:
    payload = {
        "method_id": FIXED_E8_METHOD_ID,
        "layer_order": list(FIXED_E8_LAYER_ORDER),
        "grid_count": FIXED_E8_GRID_COUNT,
        "h": float(FIXED_E8_H),
        "tau_final": 1.0,
        "kappa": FIXED_E8_KAPPA,
        "normalization_epsilon": FIXED_E8_NORMALIZATION_EPSILON,
        "xi_tie_tolerance": FIXED_E8_XI_TIE_TOLERANCE,
        "structural_problem_coordinates": "already-h-and-h2-scaled",
        "structural_score": "positive(delta_risk)/max(epsilon,trace(gram))",
        "functional_score": (
            "sum(max(probe-baseline,0)*v)/"
            "max(epsilon,sum(max(probe-baseline,0)))"
        ),
        "target_overlay_definition": FIXED_E8_TARGET_OVERLAY_DEFINITION,
        "target_velocity_step_semantics": FIXED_E8_TARGET_STEP_SEMANTICS,
        "candidate_coupled": False,
        "retry_recomputes_target_velocity": False,
        "adaptive_clock_access_count": 0,
        "scientific_rejection_count": 0,
        "hard_structural_h_budget_influence_count": 0,
        "hard_structural_p_budget_influence_count": 0,
        "functional_candidate_veto_influence_count": 0,
        "first_hit_decision_influence_count": 0,
        "persistent_commit_count": 0,
        "history_append_count": 0,
        "native_or_direct_z_cold_access_count": 0,
        "operation_ceiling": FixedE8OperationCeiling().raw_free_payload(),
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload
