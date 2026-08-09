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
from dataclasses import asdict, dataclass, replace
from enum import Enum
from fractions import Fraction
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import __version__ as SCIPY_VERSION
from scipy.optimize import Bounds, NonlinearConstraint, minimize, nnls

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
FIXED_E8_FALLBACK_LIMIT = 1
FIXED_E8_PRIMARY_BACKEND = "scipy-slsqp-float64"
FIXED_E8_FALLBACK_BACKEND = "scipy-trust-constr-float64"
FIXED_E8_PRIMARY_OPTIONS = (
    ("disp", False),
    ("ftol", FIXED_E8_SOLVER_FTOL),
    ("maxiter", FIXED_E8_SOLVER_MAXITER),
)
FIXED_E8_FALLBACK_OPTIONS = (
    ("barrier_tol", FIXED_E8_SOLVER_FTOL),
    ("gtol", FIXED_E8_SOLVER_FTOL),
    ("maxiter", FIXED_E8_SOLVER_MAXITER),
    ("verbose", 0),
    ("xtol", FIXED_E8_SOLVER_FTOL),
)


def _fallback_receipt_options(
    *, keep_feasible_from_seed: bool
) -> tuple[tuple[str, Any], ...]:
    return (
        *FIXED_E8_FALLBACK_OPTIONS,
        ("bound_feasible_padding", FIXED_E8_SOLVER_FTOL),
        ("constraint_feasible_padding", FIXED_E8_SOLVER_FTOL),
        ("keep_feasible_from_seed", keep_feasible_from_seed),
    )

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


class FixedE8Stage2FailurePolicy(str, Enum):
    FAIL_CLOSED = "FAIL_CLOSED"
    CERTIFIED_STAGE1_FALLBACK = "CERTIFIED_STAGE1_FALLBACK"


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
class FixedE8BackendAttempt:
    backend: str
    backend_version: str
    backend_options: tuple[tuple[str, Any], ...]
    backend_options_sha256: str
    success: bool
    status: int
    message_sha256: str
    iterations: int
    raw_candidate_vector: tuple[float, ...]
    candidate_vector: tuple[float, ...]
    bound_canonicalization_indices: tuple[int, ...]
    bound_canonicalization_max_abs: float
    bound_canonicalization_policy: str
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
class FixedE8SolverCertificate:
    phase: str
    authority_role: str
    success: bool
    optimizer_status: int
    optimizer_message_sha256: str
    optimizer_status_history: tuple[int, ...]
    optimizer_message_sha256_history: tuple[str, ...]
    finite: bool
    iterations: int
    optimizer_pass_count: int
    fallback_invocation_count: int
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
    backend_attempts: tuple[FixedE8BackendAttempt, ...]
    selected_backend: str
    independent_certificate_authority: bool
    passed: bool

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        fallback_count = self.fallback_invocation_count
        payload.update(
            {
                "fallback_backend_invocation_count": fallback_count,
                "fallback_backend_role": (
                    "INDEPENDENT_SAME_QP_CERTIFICATE_RECOVERY"
                    if fallback_count
                    else "NONE"
                ),
                "scientific_retry_count": 0,
                "field_rebuild_count": 0,
                "candidate_evaluation_count": 0,
                "backend_success_is_diagnostic_only": True,
                "certificate_thresholds": {
                    "primal": FIXED_E8_PRIMAL_TOLERANCE,
                    "stationarity": FIXED_E8_KKT_TOLERANCE,
                    "complementarity": FIXED_E8_KKT_TOLERANCE,
                },
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
    authoritative_certificates: tuple[FixedE8SolverCertificate, ...]
    diagnostic_shadow_certificates: tuple[FixedE8SolverCertificate, ...]
    soft_shadow_status: str
    identity_sha256: str
    failed_authoritative_certificates: tuple[FixedE8SolverCertificate, ...] = ()
    stage2_failure_policy: FixedE8Stage2FailurePolicy = (
        FixedE8Stage2FailurePolicy.FAIL_CLOSED
    )
    selected_solution_source: str = "LEGACY_DEFAULT"
    stage1_selection_fallback_count: int = 0
    stage1_selection_receipt: Mapping[str, Any] | None = None

    @property
    def certificates(self) -> tuple[FixedE8SolverCertificate, ...]:
        return (
            self.authoritative_certificates
            + self.failed_authoritative_certificates
            + self.diagnostic_shadow_certificates
        )

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
        if not all(item.passed for item in self.authoritative_certificates):
            raise ODEBFContractError("fixed E8 routing certificate failed")
        if any(
            item.authority_role != "AUTHORITATIVE"
            for item in self.authoritative_certificates
        ) or any(
            item.authority_role != "DIAGNOSTIC_SOFT_SHADOW"
            for item in self.diagnostic_shadow_certificates
        ):
            raise ODEBFContractError("fixed E8 certificate ownership differs")
        if any(
            item.authority_role != "AUTHORITATIVE" or item.passed
            for item in self.failed_authoritative_certificates
        ):
            raise ODEBFContractError("fixed E8 failed certificate ownership differs")
        if self.soft_shadow_status not in (
            "AVAILABLE",
            "SOFT_SHADOW_NUMERIC_UNAVAILABLE",
            "AUTHORITATIVE_SOFT_NOT_SHADOW",
            "AUTHORITATIVE_STAGE1_FALLBACK",
            "NOT_APPLICABLE_ZERO_WRITE",
        ):
            raise ODEBFContractError("fixed E8 soft shadow status differs")
        if self.stage2_failure_policy is FixedE8Stage2FailurePolicy.FAIL_CLOSED:
            if (
                self.selected_solution_source != "LEGACY_DEFAULT"
                or self.stage1_selection_fallback_count != 0
                or self.stage1_selection_receipt is not None
                or self.failed_authoritative_certificates
            ):
                raise ODEBFContractError("fixed E8 legacy stage2 policy differs")
        else:
            if self.arm is not FixedE8Arm.SOFT or self.selected_solution_source not in (
                "CERTIFIED_STAGE2",
                "CERTIFIED_STAGE1_FALLBACK",
            ):
                raise ODEBFContractError("fixed E8 stage2 selection source differs")
            fallback = self.selected_solution_source == "CERTIFIED_STAGE1_FALLBACK"
            if (
                self.stage1_selection_fallback_count != int(fallback)
                or (self.stage1_selection_receipt is None) != (not fallback)
                or bool(self.failed_authoritative_certificates) != fallback
            ):
                raise ODEBFContractError("fixed E8 stage1 fallback receipt differs")
        if len(self.identity_sha256) != 64:
            raise ODEBFContractError("fixed E8 routing identity differs")

    def raw_free_payload(self) -> dict[str, Any]:
        logical = len(self.certificates)
        backend = sum(item.optimizer_pass_count for item in self.certificates)
        fallback = sum(
            item.fallback_invocation_count
            for item in self.certificates
        )
        maximum_schedule = bool(
            self.mode is FixedE8StepMode.JOINT_WRITE
            and logical == 4
            and backend >= 4
            and backend <= 8
        )
        payload = {
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
            "actual_fallback_backend_invocation_count": fallback,
            "static_operation_counts_are_maximum_ceiling": True,
            "soft_shadow_decision_influence_count": (
                0 if self.arm is FixedE8Arm.NEUTRAL else 1
            ),
            "authoritative_certificates": [
                item.raw_free_payload() for item in self.authoritative_certificates
            ],
            "diagnostic_shadow_certificates": [
                item.raw_free_payload()
                for item in self.diagnostic_shadow_certificates
            ],
            "soft_shadow_status": self.soft_shadow_status,
            "identity_sha256": self.identity_sha256,
        }
        if self.stage2_failure_policy is not FixedE8Stage2FailurePolicy.FAIL_CLOSED:
            payload.update(
                {
                    "stage2_failure_policy": self.stage2_failure_policy.value,
                    "selected_solution_source": self.selected_solution_source,
                    "stage1_selection_fallback_count": (
                        self.stage1_selection_fallback_count
                    ),
                    "stage1_selection_receipt": (
                        None
                        if self.stage1_selection_receipt is None
                        else dict(self.stage1_selection_receipt)
                    ),
                    "failed_authoritative_certificates": [
                        item.raw_free_payload()
                        for item in self.failed_authoritative_certificates
                    ],
                }
            )
        return payload


@dataclass(frozen=True, slots=True)
class FixedE8Constraint:
    name: str
    function: Callable[[np.ndarray], float]
    jacobian: Callable[[np.ndarray], np.ndarray]
    hessian: Callable[[np.ndarray], np.ndarray]


Constraint = FixedE8Constraint


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
        FixedE8Constraint(
            "technical_write_trust",
            lambda value, matrix=trust, radius=problem.trust_radius: float(
                radius**2 - value @ matrix @ value
            ),
            lambda value, matrix=trust: -2.0 * matrix @ value,
            lambda value, matrix=trust: -2.0 * matrix,
        )
    ]
    if requested_progress is not None:
        constraints.insert(
            0,
            FixedE8Constraint(
                "kappa_progress",
                lambda value, a=progress, target=requested_progress: float(
                    a @ value - target
                ),
                lambda value, a=progress: a.copy(),
                lambda value, dimension=active.size: np.zeros(
                    (dimension, dimension), dtype=np.float64
                ),
            ),
        )
    return constraints


def _certificate_reconstruction_payload(
    *,
    candidate_vector: Sequence[float],
    objective_value: float,
    objective_gradient: Sequence[float],
    ordered_constraint_names: Sequence[str],
    ordered_constraint_slacks: Sequence[float],
    ordered_constraint_gradients: Sequence[Sequence[float]],
    ordered_kkt_multipliers: Sequence[float],
) -> dict[str, Any]:
    return {
        "candidate_vector": list(candidate_vector),
        "objective_value": float(objective_value),
        "objective_gradient": list(objective_gradient),
        "ordered_constraint_names": list(ordered_constraint_names),
        "ordered_constraint_slacks": list(ordered_constraint_slacks),
        "ordered_constraint_gradients": [
            list(item) for item in ordered_constraint_gradients
        ],
        "ordered_kkt_multipliers": list(ordered_kkt_multipliers),
        "certificate_thresholds": {
            "primal": FIXED_E8_PRIMAL_TOLERANCE,
            "stationarity": FIXED_E8_KKT_TOLERANCE,
            "complementarity": FIXED_E8_KKT_TOLERANCE,
            "active_constraint_slack": 10.0 * FIXED_E8_PRIMAL_TOLERANCE,
        },
    }


def reconstruct_fixed_e8_backend_certificate(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the independent certificate from a raw-free attempt receipt."""

    candidate = np.asarray(payload.get("candidate_vector"), dtype=np.float64)
    objective_value = float(payload.get("objective_value"))
    gradient = np.asarray(payload.get("objective_gradient"), dtype=np.float64)
    names = tuple(str(item) for item in payload.get("ordered_constraint_names", ()))
    slacks = np.asarray(payload.get("ordered_constraint_slacks"), dtype=np.float64)
    gradients = np.asarray(
        payload.get("ordered_constraint_gradients"), dtype=np.float64
    )
    multipliers = np.asarray(
        payload.get("ordered_kkt_multipliers"), dtype=np.float64
    )
    if (
        candidate.ndim != 1
        or gradient.shape != candidate.shape
        or slacks.ndim != 1
        or len(names) != slacks.size
        or gradients.shape != (slacks.size, candidate.size)
        or multipliers.shape != slacks.shape
        or np.any(multipliers < -FIXED_E8_PRIMAL_TOLERANCE)
    ):
        raise ODEBFContractError("fixed E8 certificate reconstruction shape differs")
    finite = bool(
        math.isfinite(objective_value)
        and np.all(np.isfinite(candidate))
        and np.all(np.isfinite(gradient))
        and np.all(np.isfinite(slacks))
        and np.all(np.isfinite(gradients))
        and np.all(np.isfinite(multipliers))
    )
    maximum_violation = float(max(0.0, -float(slacks.min(initial=0.0))))
    stationarity = float(
        np.linalg.norm(gradient - gradients.T @ multipliers)
        / max(float(np.linalg.norm(gradient)), 1.0)
    )
    complementarity = float(
        np.max(np.abs(multipliers * slacks), initial=0.0)
    )
    passed = bool(
        finite
        and maximum_violation <= FIXED_E8_PRIMAL_TOLERANCE
        and stationarity <= FIXED_E8_KKT_TOLERANCE
        and complementarity <= FIXED_E8_KKT_TOLERANCE
    )
    first_false: str | None = None
    if not finite:
        first_false = "optimizer_finite"
    elif maximum_violation > FIXED_E8_PRIMAL_TOLERANCE:
        first_false = next(
            (
                name
                for name, slack in zip(names, slacks, strict=True)
                if float(slack) < -FIXED_E8_PRIMAL_TOLERANCE
            ),
            "primal_feasibility",
        )
    elif stationarity > FIXED_E8_KKT_TOLERANCE:
        first_false = "stationarity"
    elif complementarity > FIXED_E8_KKT_TOLERANCE:
        first_false = "complementarity"
    reconstruction = _certificate_reconstruction_payload(
        candidate_vector=candidate,
        objective_value=objective_value,
        objective_gradient=gradient,
        ordered_constraint_names=names,
        ordered_constraint_slacks=slacks,
        ordered_constraint_gradients=gradients,
        ordered_kkt_multipliers=multipliers,
    )
    derived = {
        "finite": finite,
        "maximum_primal_violation": maximum_violation,
        "stationarity_residual": stationarity,
        "complementarity_residual": complementarity,
        "first_false_component": first_false,
        "passed": passed,
    }
    return {
        **derived,
        "reconstruction_sha256": canonical_hash(
            {"inputs": reconstruction, "derived": derived}
        ),
    }


def _canonicalize_primal_tolerance_bound_drift(
    value: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...], float]:
    """Snap only out-of-bound drift already covered by the primal tolerance.

    Numerical backends may return a point infinitesimally outside an exact
    bound while the independent certificate correctly accepts that point at
    ``FIXED_E8_PRIMAL_TOLERANCE``.  Physical factor assembly requires the
    canonical closed interval, so the solver/backend boundary normalizes only
    those exterior coordinates and the independent certificate is rebuilt on
    the normalized point.  Interior values are never rounded or clipped.
    """

    candidate = np.asarray(value, dtype=np.float64)
    left = np.asarray(lower, dtype=np.float64)
    right = np.asarray(upper, dtype=np.float64)
    if (
        candidate.ndim != 1
        or candidate.shape != left.shape
        or candidate.shape != right.shape
    ):
        raise ODEBFContractError("fixed E8 bound canonicalization shape differs")
    canonical = candidate.copy()
    changed: list[int] = []
    maximum = 0.0
    for index, observed in enumerate(candidate):
        low = float(left[index])
        high = float(right[index])
        replacement: float | None = None
        if math.isfinite(low) and observed < low:
            if low - float(observed) <= FIXED_E8_PRIMAL_TOLERANCE:
                replacement = low
        elif math.isfinite(high) and observed > high:
            if float(observed) - high <= FIXED_E8_PRIMAL_TOLERANCE:
                replacement = high
        if replacement is not None:
            changed.append(index)
            maximum = max(maximum, abs(float(observed) - replacement))
            canonical[index] = replacement
    return canonical, tuple(changed), maximum


def _backend_attempt(
    *,
    result: Any,
    backend: str,
    backend_options: tuple[tuple[str, Any], ...],
    raw_value: np.ndarray,
    value: np.ndarray,
    bound_canonicalization_indices: tuple[int, ...],
    bound_canonicalization_max_abs: float,
    objective_value: float,
    objective_gradient: np.ndarray,
    constraints: Sequence[Constraint],
    lower: np.ndarray,
    upper: np.ndarray,
) -> FixedE8BackendAttempt:
    names: list[str] = []
    slacks: list[float] = []
    gradients: list[np.ndarray] = []
    for constraint in constraints:
        names.append(constraint.name)
        slacks.append(float(constraint.function(value)))
        gradients.append(
            np.asarray(constraint.jacobian(value), dtype=np.float64)
        )
    for index in range(value.size):
        for name, bound, slack, direction in (
            (
                f"lower_{index}",
                float(lower[index]),
                float(value[index] - lower[index]),
                1.0,
            ),
            (
                f"upper_{index}",
                float(upper[index]),
                float(upper[index] - value[index]),
                -1.0,
            ),
        ):
            if not math.isfinite(bound):
                continue
            names.append(name)
            slacks.append(slack)
            basis = np.zeros_like(value)
            basis[index] = direction
            gradients.append(basis)
    slack_array = np.asarray(slacks, dtype=np.float64)
    gradient_matrix = np.stack(gradients, axis=0)
    active = slack_array <= 10.0 * FIXED_E8_PRIMAL_TOLERANCE
    multipliers = np.zeros(slack_array.size, dtype=np.float64)
    if (
        np.any(active)
        and np.all(np.isfinite(objective_gradient))
        and np.all(np.isfinite(gradient_matrix[active]))
    ):
        active_multipliers, _ = nnls(
            gradient_matrix[active].T, objective_gradient
        )
        multipliers[active] = active_multipliers
    success = bool(getattr(result, "success", False))
    optimizer_status = int(getattr(result, "status", -1))
    optimizer_message_sha256 = canonical_hash(
        {"optimizer_message": str(getattr(result, "message", ""))}
    )
    reconstruction_input = _certificate_reconstruction_payload(
        candidate_vector=value,
        objective_value=objective_value,
        objective_gradient=objective_gradient,
        ordered_constraint_names=names,
        ordered_constraint_slacks=slack_array,
        ordered_constraint_gradients=gradient_matrix,
        ordered_kkt_multipliers=multipliers,
    )
    reconstructed = reconstruct_fixed_e8_backend_certificate(
        reconstruction_input
    )
    return FixedE8BackendAttempt(
        backend=backend,
        backend_version=SCIPY_VERSION,
        backend_options=backend_options,
        backend_options_sha256=canonical_hash(dict(backend_options)),
        success=success,
        status=optimizer_status,
        message_sha256=optimizer_message_sha256,
        iterations=int(getattr(result, "nit", 0)),
        raw_candidate_vector=tuple(float(item) for item in raw_value),
        candidate_vector=tuple(float(item) for item in value),
        bound_canonicalization_indices=bound_canonicalization_indices,
        bound_canonicalization_max_abs=float(bound_canonicalization_max_abs),
        bound_canonicalization_policy=(
            "EXTERIOR_ONLY_WITHIN_LOCKED_PRIMAL_TOLERANCE_THEN_RECERTIFY"
        ),
        objective_value=float(objective_value),
        objective_gradient=tuple(float(item) for item in objective_gradient),
        ordered_constraint_names=tuple(names),
        ordered_constraint_slacks=tuple(float(item) for item in slack_array),
        ordered_constraint_gradients=tuple(
            tuple(float(item) for item in row) for row in gradient_matrix
        ),
        ordered_kkt_multipliers=tuple(float(item) for item in multipliers),
        maximum_primal_violation=float(
            reconstructed["maximum_primal_violation"]
        ),
        stationarity_residual=float(reconstructed["stationarity_residual"]),
        complementarity_residual=float(
            reconstructed["complementarity_residual"]
        ),
        finite=bool(reconstructed["finite"]),
        first_false_component=reconstructed["first_false_component"],
        passed=bool(reconstructed["passed"]),
        reconstruction_sha256=str(reconstructed["reconstruction_sha256"]),
    )


def _solve_slsqp(
    *,
    phase: str,
    problem: RoutingProblem,
    active: np.ndarray,
    objective: Callable[[np.ndarray], float],
    jacobian: Callable[[np.ndarray], np.ndarray],
    hessian: Callable[[np.ndarray], np.ndarray],
    constraints: Sequence[Constraint],
    lower: np.ndarray,
    upper: np.ndarray,
    initial: np.ndarray,
    p_max: float,
    requested_progress: float,
    xi: float | None,
    authority_role: str,
    fail_closed: bool = True,
    certificate_observer: FixedE8CertificateObserver | None = None,
) -> tuple[np.ndarray, FixedE8SolverCertificate]:
    scipy_constraints = [
        {
            "type": "ineq",
            "fun": constraint.function,
            "jac": constraint.jacobian,
        }
        for constraint in constraints
    ]
    bounds = tuple(
        (float(left), float(right))
        for left, right in zip(lower, upper, strict=True)
    )

    def run_primary(start: np.ndarray) -> Any:
        return minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            jac=jacobian,
            method="SLSQP",
            bounds=bounds,
            constraints=scipy_constraints,
            options=dict(FIXED_E8_PRIMARY_OPTIONS),
        )

    def candidate_parts(
        result: Any,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float | None,
        np.ndarray,
        tuple[int, ...],
        float,
    ]:
        raw_value = np.asarray(result.x, dtype=np.float64)
        value, canonicalized, maximum_adjustment = (
            _canonicalize_primal_tolerance_bound_drift(raw_value, lower, upper)
        )
        expanded = np.zeros(problem.signed_progress.size, dtype=np.float64)
        expanded[active] = value[: active.size]
        observed_xi = xi
        if value.size != active.size:
            if value.size != active.size + 1:
                raise ODEBFContractError(
                    "fixed E8 solver auxiliary dimension differs"
                )
            observed_xi = float(value[active.size])
        return (
            value,
            expanded,
            observed_xi,
            raw_value,
            canonicalized,
            maximum_adjustment,
        )

    primary_result = run_primary(np.asarray(initial, dtype=np.float64))
    (
        value,
        expanded,
        observed_xi,
        raw_value,
        canonicalized,
        maximum_adjustment,
    ) = candidate_parts(primary_result)
    attempts = [
        _backend_attempt(
            result=primary_result,
            backend=FIXED_E8_PRIMARY_BACKEND,
            backend_options=FIXED_E8_PRIMARY_OPTIONS,
            raw_value=raw_value,
            value=value,
            bound_canonicalization_indices=canonicalized,
            bound_canonicalization_max_abs=maximum_adjustment,
            objective_value=float(objective(value)),
            objective_gradient=np.asarray(jacobian(value), dtype=np.float64),
            constraints=constraints,
            lower=lower,
            upper=upper,
        )
    ]
    if not attempts[-1].passed and attempts[-1].finite:
        trust_constraints = [
            NonlinearConstraint(
                lambda point: np.asarray(
                    [item.function(point) for item in constraints],
                    dtype=np.float64,
                ),
                np.full(
                    len(constraints),
                    -FIXED_E8_SOLVER_FTOL,
                    dtype=np.float64,
                ),
                np.full(len(constraints), np.inf, dtype=np.float64),
                jac=lambda point: np.stack(
                    [item.jacobian(point) for item in constraints], axis=0
                ),
                hess=lambda point, multiplier: sum(
                    (
                        float(weight) * item.hessian(point)
                        for weight, item in zip(
                            np.asarray(multiplier).reshape(-1),
                            constraints,
                            strict=True,
                        )
                    ),
                    np.zeros((value.size, value.size), dtype=np.float64),
                ),
                keep_feasible=True,
            )
        ]
        fallback_keep_feasible = bool(
            np.all(value >= lower - FIXED_E8_SOLVER_FTOL)
            and np.all(value <= upper + FIXED_E8_SOLVER_FTOL)
            and all(
                item.function(value) >= -FIXED_E8_SOLVER_FTOL
                for item in constraints
            )
        )
        fallback_result = minimize(
            objective,
            value,
            jac=jacobian,
            hess=hessian,
            method="trust-constr",
            bounds=Bounds(
                lower - FIXED_E8_SOLVER_FTOL,
                upper + FIXED_E8_SOLVER_FTOL,
                keep_feasible=fallback_keep_feasible,
            ),
            constraints=[
                NonlinearConstraint(
                    item.fun,
                    item.lb,
                    item.ub,
                    jac=item.jac,
                    hess=item.hess,
                    keep_feasible=fallback_keep_feasible,
                )
                for item in trust_constraints
            ],
            options=dict(FIXED_E8_FALLBACK_OPTIONS),
        )
        (
            value,
            expanded,
            observed_xi,
            raw_value,
            canonicalized,
            maximum_adjustment,
        ) = candidate_parts(fallback_result)
        attempts.append(
            _backend_attempt(
                result=fallback_result,
                backend=FIXED_E8_FALLBACK_BACKEND,
                backend_options=_fallback_receipt_options(
                    keep_feasible_from_seed=fallback_keep_feasible
                ),
                raw_value=raw_value,
                value=value,
                bound_canonicalization_indices=canonicalized,
                bound_canonicalization_max_abs=maximum_adjustment,
                objective_value=float(objective(value)),
                objective_gradient=np.asarray(jacobian(value), dtype=np.float64),
                constraints=constraints,
                lower=lower,
                upper=upper,
            )
        )
    selected_attempt = attempts[-1]
    certificate = FixedE8SolverCertificate(
        phase=phase,
        authority_role=authority_role,
        success=selected_attempt.success,
        optimizer_status=selected_attempt.status,
        optimizer_message_sha256=selected_attempt.message_sha256,
        optimizer_status_history=tuple(item.status for item in attempts),
        optimizer_message_sha256_history=tuple(
            item.message_sha256 for item in attempts
        ),
        finite=selected_attempt.finite,
        iterations=sum(item.iterations for item in attempts),
        optimizer_pass_count=len(attempts),
        fallback_invocation_count=len(attempts) - 1,
        maximum_primal_violation=selected_attempt.maximum_primal_violation,
        stationarity_residual=selected_attempt.stationarity_residual,
        complementarity_residual=selected_attempt.complementarity_residual,
        signed_progress=float(problem.signed_progress @ expanded),
        trust_value=float(expanded @ problem.trust_metric @ expanded),
        p_max=p_max,
        requested_progress=requested_progress,
        xi=observed_xi,
        first_false_component=selected_attempt.first_false_component,
        active_constraints=tuple(
            name
            for name, slack in zip(
                selected_attempt.ordered_constraint_names,
                selected_attempt.ordered_constraint_slacks,
                strict=True,
            )
            if abs(slack) <= 10.0 * FIXED_E8_PRIMAL_TOLERANCE
        ),
        backend_attempts=tuple(attempts),
        selected_backend=selected_attempt.backend,
        independent_certificate_authority=True,
        passed=selected_attempt.passed,
    )
    if not certificate.passed:
        if fail_closed and certificate_observer is not None:
            certificate_observer(certificate)
        if fail_closed:
            raise ODEBFContractError(
                f"fixed E8 {phase} NUMERIC_QP_UNCERTIFIED: "
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

    def hessian(value: np.ndarray) -> np.ndarray:
        del value
        result = np.zeros((active.size + 1, active.size + 1), dtype=np.float64)
        result[: active.size, : active.size] = -2.0 * gram / normalization
        return result

    return FixedE8Constraint(label, slack, derivative, hessian)


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

    return FixedE8Constraint(
        metric.label,
        slack,
        derivative,
        lambda value: np.zeros((active.size + 1, active.size + 1), dtype=np.float64),
    )


def _maximum_progress(
    problem: RoutingProblem,
    active: np.ndarray,
    *,
    certificate_observer: FixedE8CertificateObserver | None = None,
    fail_closed: bool = True,
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
        hessian=lambda item, dimension=active.size: np.zeros(
            (dimension, dimension), dtype=np.float64
        ),
        constraints=constraints,
        lower=np.zeros(active.size, dtype=np.float64),
        upper=caps,
        initial=initial,
        p_max=0.0,
        requested_progress=0.0,
        xi=None,
        authority_role="AUTHORITATIVE",
        fail_closed=fail_closed,
        certificate_observer=certificate_observer,
    )
    expanded = np.zeros(problem.signed_progress.size, dtype=np.float64)
    expanded[active] = value
    p_max = float(problem.signed_progress @ expanded)
    if not math.isfinite(p_max) or p_max < -FIXED_E8_PRIMAL_TOLERANCE:
        raise ODEBFContractError("fixed E8 maximum progress differs")
    certificate = replace(certificate, p_max=max(p_max, 0.0))
    if certificate_observer is not None:
        certificate_observer(certificate)
    return expanded, max(p_max, 0.0), certificate


def solve_fixed_e8_routing(
    problem: RoutingProblem,
    inventory: FixedE8SoftInventory,
    *,
    arm: FixedE8Arm | str,
    certificate_observer: FixedE8CertificateObserver | None = None,
    stage2_failure_policy: FixedE8Stage2FailurePolicy | str = (
        FixedE8Stage2FailurePolicy.FAIL_CLOSED
    ),
) -> FixedE8RoutingResult:
    """Solve one fixed-grid E8 routing field.

    Structural and functional risks never appear in the technical feasible
    set.  ``E8-NEUTRAL`` minimizes capacity under the same progress/trust set;
    ``E8-SOFT`` first minimizes the worst normalized soft score and then the
    same capacity objective within the sealed xi tie tolerance.
    """

    selected = FixedE8Arm(arm)
    selected_stage2_policy = FixedE8Stage2FailurePolicy(stage2_failure_policy)
    if (
        selected_stage2_policy
        is FixedE8Stage2FailurePolicy.CERTIFIED_STAGE1_FALLBACK
        and selected is not FixedE8Arm.SOFT
    ):
        raise ODEBFContractError("fixed E8 stage1 fallback is Soft-only")
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
    authoritative_certificates: list[FixedE8SolverCertificate] = [
        maximum_certificate
    ]
    diagnostic_shadow_certificates: list[FixedE8SolverCertificate] = []
    failed_authoritative_certificates: list[FixedE8SolverCertificate] = []
    stage1_selection_receipt: dict[str, Any] | None = None
    selected_solution_source = "LEGACY_DEFAULT"
    stage1_selection_fallback_count = 0
    neutral_active, capacity_certificate = _solve_slsqp(
        phase="neutral-minimum-capacity",
        problem=problem,
        active=active,
        objective=lambda value, matrix=metric: float(
            0.5 * value @ matrix @ value
        ),
        jacobian=lambda value, matrix=metric: matrix @ value,
        hessian=lambda value, matrix=metric: matrix.copy(),
        constraints=technical,
        lower=np.zeros(active.size, dtype=np.float64),
        upper=caps,
        initial=initial_velocity,
        p_max=p_max,
        requested_progress=requested,
        xi=None,
        authority_role="AUTHORITATIVE",
        certificate_observer=certificate_observer,
    )
    if certificate_observer is not None:
        certificate_observer(capacity_certificate)
    authoritative_certificates.append(capacity_certificate)
    xi_index = active.size
    stage1_constraints: list[Constraint] = []
    for constraint in technical:
        stage1_constraints.append(
            FixedE8Constraint(
                constraint.name,
                lambda value, item=constraint: item.function(
                    value[:active.size]
                ),
                lambda value, item=constraint: np.concatenate(
                    (
                        item.jacobian(value[:active.size]),
                        np.zeros(1, dtype=np.float64),
                    )
                ),
                lambda value, item=constraint: np.pad(
                    item.hessian(value[:active.size]),
                    ((0, 1), (0, 1)),
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
        hessian=lambda value: np.zeros(
            (value.size, value.size), dtype=np.float64
        ),
        constraints=stage1_constraints,
        lower=np.concatenate((np.zeros(active.size), [0.0])),
        upper=np.concatenate((caps, [np.inf])),
        initial=initial_stage1,
        p_max=p_max,
        requested_progress=requested,
        xi=initial_xi,
        authority_role=(
            "AUTHORITATIVE"
            if selected is FixedE8Arm.SOFT
            else "DIAGNOSTIC_SOFT_SHADOW"
        ),
        fail_closed=selected is FixedE8Arm.SOFT,
        certificate_observer=certificate_observer,
    )
    if certificate_observer is not None:
        certificate_observer(stage1_certificate)
    if selected is FixedE8Arm.SOFT:
        authoritative_certificates.append(stage1_certificate)
    else:
        diagnostic_shadow_certificates.append(stage1_certificate)
    xi_star: float | None = None
    soft_active = neutral_active.copy()
    soft_shadow_status = "SOFT_SHADOW_NUMERIC_UNAVAILABLE"
    if stage1_certificate.passed:
        xi_star = max(float(stage1_value[xi_index]), 0.0)
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
                FixedE8Constraint(
                    label,
                    lambda value, lin=linear, matrix=gram, scale=normalization: float(
                        xi_cap
                        - (2.0 * lin @ value + value @ matrix @ value) / scale
                    ),
                    lambda value, lin=linear, matrix=gram, scale=normalization: -(
                        2.0 * lin + 2.0 * matrix @ value
                    )
                    / scale,
                    lambda value, matrix=gram, scale=normalization: -2.0
                    * matrix
                    / scale,
                ),
            )

        if inventory.history_item_count > 0:
            append_structural_cap(
                "xi_tie_structural_historical", problem.historical
            )
        append_structural_cap(
            "xi_tie_structural_pretrained", problem.pretrained
        )
        for functional in inventory.active_metrics():
            positive = functional.positive_increment[active]
            normalization = functional.normalization
            stage2_constraints.append(
                FixedE8Constraint(
                    f"xi_tie_{functional.label}",
                    lambda value, slope=positive, scale=normalization: float(
                        xi_cap - slope @ value / scale
                    ),
                    lambda value, slope=positive, scale=normalization: -slope
                    / scale,
                    lambda value, dimension=active.size: np.zeros(
                        (dimension, dimension), dtype=np.float64
                    ),
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
            hessian=lambda value, matrix=metric: matrix.copy(),
            constraints=stage2_constraints,
            lower=np.zeros(active.size, dtype=np.float64),
            upper=caps,
            initial=stage1_value[:active.size],
            p_max=p_max,
            requested_progress=requested,
            xi=xi_star,
            authority_role=(
                "AUTHORITATIVE"
                if selected is FixedE8Arm.SOFT
                else "DIAGNOSTIC_SOFT_SHADOW"
            ),
            fail_closed=(
                selected is FixedE8Arm.SOFT
                and selected_stage2_policy
                is FixedE8Stage2FailurePolicy.FAIL_CLOSED
            ),
            certificate_observer=certificate_observer,
        )
        if certificate_observer is not None:
            certificate_observer(stage2_certificate)
        if selected is FixedE8Arm.SOFT:
            if stage2_certificate.passed:
                authoritative_certificates.append(stage2_certificate)
                soft_shadow_status = "AUTHORITATIVE_SOFT_NOT_SHADOW"
                if (
                    selected_stage2_policy
                    is FixedE8Stage2FailurePolicy.CERTIFIED_STAGE1_FALLBACK
                ):
                    selected_solution_source = "CERTIFIED_STAGE2"
            else:
                if (
                    selected_stage2_policy
                    is not FixedE8Stage2FailurePolicy.CERTIFIED_STAGE1_FALLBACK
                ):
                    raise ODEBFContractError(
                        "fixed E8 uncertified stage2 escaped fail-close"
                    )
                reconstructed = reconstruct_fixed_e8_backend_certificate(
                    stage1_certificate.backend_attempts[-1].raw_free_payload()
                )
                preserved_stage1 = np.asarray(
                    stage1_value[:active.size], dtype=np.float64
                ).copy()
                preserved_sha = canonical_hash(preserved_stage1.tolist())
                reconstructed_candidate = np.asarray(
                    stage1_certificate.backend_attempts[-1].candidate_vector,
                    dtype=np.float64,
                )[:active.size]
                if (
                    not stage1_certificate.passed
                    or not bool(reconstructed["passed"])
                    or not np.array_equal(
                        preserved_stage1, reconstructed_candidate
                    )
                ):
                    raise ODEBFContractError(
                        "fixed E8 certified stage1 selection reverify failed"
                    )
                soft_active = preserved_stage1
                applied_sha = canonical_hash(soft_active.tolist())
                if applied_sha != preserved_sha:
                    raise ODEBFContractError(
                        "fixed E8 stage1 fallback vector identity differs"
                    )
                failed_authoritative_certificates.append(stage2_certificate)
                stage1_selection_fallback_count = 1
                selected_solution_source = "CERTIFIED_STAGE1_FALLBACK"
                soft_shadow_status = "AUTHORITATIVE_STAGE1_FALLBACK"
                stage1_selection_receipt = {
                    "selected_solution_source": selected_solution_source,
                    "preserved_stage1_vector": preserved_stage1.tolist(),
                    "preserved_stage1_vector_sha256": preserved_sha,
                    "applied_vector_sha256": applied_sha,
                    "stage1_certificate": stage1_certificate.raw_free_payload(),
                    "stage1_reconstruction": reconstructed,
                    "stage2_failure_component": (
                        stage2_certificate.first_false_component
                    ),
                    "stage2_status": stage2_certificate.optimizer_status,
                    "stage2_message_sha256": (
                        stage2_certificate.optimizer_message_sha256
                    ),
                    "stage2_certificate": stage2_certificate.raw_free_payload(),
                    "fallback_reason": "STAGE2_NUMERIC_CERTIFICATE_MISS",
                    "fallback_count": 1,
                }
                stage1_selection_receipt["identity_sha256"] = canonical_hash(
                    stage1_selection_receipt
                )
        else:
            diagnostic_shadow_certificates.append(stage2_certificate)
            if stage2_certificate.passed:
                soft_shadow_status = "AVAILABLE"
            else:
                soft_active = neutral_active.copy()
                soft_shadow_status = "SOFT_SHADOW_NUMERIC_UNAVAILABLE"
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
        "authoritative_certificates": [
            item.raw_free_payload() for item in authoritative_certificates
        ],
        "diagnostic_shadow_certificates": [
            item.raw_free_payload() for item in diagnostic_shadow_certificates
        ],
        "soft_shadow_status": soft_shadow_status,
        "hard_structural_h_budget_influence_count": 0,
        "hard_structural_p_budget_influence_count": 0,
        "functional_candidate_veto_influence_count": 0,
        "write_trust_role": "TECHNICAL_INTEGRATION_BOUND",
        "selected_velocity_role": (
            "PRE_SOFT_NEUTRAL"
            if selected is FixedE8Arm.NEUTRAL
            else "JOINT_SOFT_LEXICOGRAPHIC"
        ),
        "matched_solver_schedule": (
            len(authoritative_certificates)
            + len(diagnostic_shadow_certificates)
            == 4
        ),
        "solver_schedule_kind": (
            "FULL_LEXICOGRAPHIC_MAXIMUM_SCHEDULE"
            if len(authoritative_certificates)
            + len(diagnostic_shadow_certificates)
            == 4
            else "SOFT_SHADOW_NUMERIC_UNAVAILABLE_REDUCED_DIAGNOSTIC_SCHEDULE"
        ),
        "actual_logical_qp_count": len(authoritative_certificates)
        + len(diagnostic_shadow_certificates),
        "actual_optimizer_backend_invocation_count": sum(
            item.optimizer_pass_count
            for item in (
                *authoritative_certificates,
                *diagnostic_shadow_certificates,
            )
        ),
        "actual_fallback_backend_invocation_count": sum(
            item.fallback_invocation_count
            for item in (
                *authoritative_certificates,
                *diagnostic_shadow_certificates,
            )
        ),
        "static_operation_counts_are_maximum_ceiling": True,
        "soft_shadow_decision_influence_count": (
            0 if selected is FixedE8Arm.NEUTRAL else 1
        ),
    }
    if selected_stage2_policy is not FixedE8Stage2FailurePolicy.FAIL_CLOSED:
        all_certificates = (
            *authoritative_certificates,
            *failed_authoritative_certificates,
            *diagnostic_shadow_certificates,
        )
        payload.update(
            {
                "stage2_failure_policy": selected_stage2_policy.value,
                "selected_solution_source": selected_solution_source,
                "stage1_selection_fallback_count": (
                    stage1_selection_fallback_count
                ),
                "stage1_selection_receipt": stage1_selection_receipt,
                "failed_authoritative_certificates": [
                    item.raw_free_payload()
                    for item in failed_authoritative_certificates
                ],
                "actual_logical_qp_count": len(all_certificates),
                "actual_optimizer_backend_invocation_count": sum(
                    item.optimizer_pass_count for item in all_certificates
                ),
                "actual_fallback_backend_invocation_count": sum(
                    item.fallback_invocation_count for item in all_certificates
                ),
            }
        )
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
        tuple(authoritative_certificates),
        tuple(diagnostic_shadow_certificates),
        soft_shadow_status,
        identity,
        tuple(failed_authoritative_certificates),
        selected_stage2_policy,
        selected_solution_source,
        stage1_selection_fallback_count,
        stage1_selection_receipt,
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
        "authoritative_certificates": [
            item.raw_free_payload() for item in certificates
        ],
        "diagnostic_shadow_certificates": [],
        "soft_shadow_status": "NOT_APPLICABLE_ZERO_WRITE",
        "target_only_recovery_consumes_grid_interval": True,
        "retry_count": 0,
        "matched_solver_schedule": False,
        "solver_schedule_kind": "ZERO_WRITE_REDUCED_TECHNICAL_SCHEDULE",
        "actual_logical_qp_count": len(certificates),
        "actual_optimizer_backend_invocation_count": sum(
            item.optimizer_pass_count for item in certificates
        ),
        "actual_fallback_backend_invocation_count": sum(
            item.fallback_invocation_count for item in certificates
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
        (),
        "NOT_APPLICABLE_ZERO_WRITE",
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
    target_write_realization_forwards_per_arm: int = 8
    qp_solves_per_field: int = 4
    nominal_qp_backend_invocations_per_field: int = 4
    fallback_invocations_per_qp: int = FIXED_E8_FALLBACK_LIMIT
    qp_backend_invocation_failure_ceiling_per_field: int = 8
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
            or self.target_write_realization_forwards_per_arm != 8
            or self.qp_solves_per_field != 4
            or self.nominal_qp_backend_invocations_per_field != 4
            or self.fallback_invocations_per_qp != 1
            or self.qp_backend_invocation_failure_ceiling_per_field != 8
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
            "target_write_realization_forward_count": (
                self.target_write_realization_forwards_per_arm
            ),
            "qp_solve_count": self.fields_per_arm * self.qp_solves_per_field,
            "qp_backend_invocation_count": (
                self.fields_per_arm
                * self.qp_backend_invocation_failure_ceiling_per_field
            ),
            "nominal_qp_backend_invocation_count": (
                self.fields_per_arm
                * self.nominal_qp_backend_invocations_per_field
            ),
            "fallback_backend_invocation_ceiling_count": (
                self.fields_per_arm
                * self.qp_solves_per_field
                * self.fallback_invocations_per_qp
            ),
            "unconditional_stage2_polish_count": 0,
            "scientific_retry_count": 0,
        }
        return {
            "per_arm": per_arm,
            "two_arm": {
                key: value * self.arm_count for key, value in per_arm.items()
            },
            "field_probe_evaluator_schedule_matched": True,
            "solver_backend_schedule_matched_not_required": True,
            "solver_backend_schedule_role": (
                "PRIMARY_PLUS_AT_MOST_ONE_INDEPENDENT_SAME_QP_FALLBACK"
            ),
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
        "fallback_backend_limit_per_qp": FIXED_E8_FALLBACK_LIMIT,
        "fallback_backend_role": "INDEPENDENT_SAME_QP_CERTIFICATE_RECOVERY",
        "primary_backend": FIXED_E8_PRIMARY_BACKEND,
        "fallback_backend": FIXED_E8_FALLBACK_BACKEND,
        "backend_success_is_diagnostic_only": True,
        "certificate_authority": "INDEPENDENT_FINITE_PRIMAL_STATIONARITY_COMPLEMENTARITY",
        "structural_problem_coordinates": "already-h-and-h2-scaled",
        "structural_score": "positive(delta_risk)/max(epsilon,trace(gram))",
        "functional_score": (
            "sum(max(probe-baseline,0)*v)/"
            "max(epsilon,sum(max(probe-baseline,0)))"
        ),
        "target_overlay_definition": FIXED_E8_TARGET_OVERLAY_DEFINITION,
        "target_velocity_step_semantics": FIXED_E8_TARGET_STEP_SEMANTICS,
        "target_probe_displacement_definition": "h*sum_l(v_l*B_l(z))",
        "physical_write_displacement_definition": "h*sum_l(v_l*B_l)",
        "target_probe_equals_physical_trial": True,
        "h_applied_exactly_once_to_target_and_write": True,
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
