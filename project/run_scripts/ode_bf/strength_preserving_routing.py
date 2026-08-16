"""Strength-preserving allocation for the P1R19 Dynamic writer.

The target flow chooses one applied-step semantic strength.  This module only
chooses how the five non-negative physical writer coefficients realize that
strength.  It deliberately has no target, model, retry, H/P budget, or
endpoint API; all inputs are detached FP64 routing geometry.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import __version__ as SCIPY_VERSION
from scipy.optimize import minimize, nnls

from .contracts import ODEBFContractError, canonical_hash
from .fixed_e8_soft_routing import (
    FIXED_E8_H,
    FIXED_E8_KKT_TOLERANCE,
    FIXED_E8_LAYER_ORDER,
    FIXED_E8_NORMALIZATION_EPSILON,
    FIXED_E8_PRIMAL_TOLERANCE,
    FIXED_E8_SOLVER_FTOL,
    FIXED_E8_SOLVER_MAXITER,
    FIXED_E8_XI_TIE_TOLERANCE,
    FixedE8Arm,
    FixedE8Score,
    FixedE8SoftInventory,
    FixedE8StepMode,
    functional_soft_score,
    structural_soft_score,
)
from .routing import RoutingProblem


STRENGTH_PRESERVING_INSTRUCTION_ID = (
    "ODEEDIT-S05-ODE-BF-STRENGTH-PRESERVING-ROUTER-P1R19-V1"
)
STRENGTH_PRESERVING_AMENDMENT_ID = (
    "ODEEDIT-S05-ODE-BF-STRENGTH-PRESERVING-ROUTER-P1R19-V1-A1"
)
STRENGTH_PRESERVING_METHOD_ID = (
    "R13-DYNAMIC-PHYSICAL-WONLY-STRENGTH-PRESERVING-ROUTER-V1"
)
STRENGTH_EQUALITY_TOLERANCE = FIXED_E8_PRIMAL_TOLERANCE
STRENGTH_COVERAGE_EPSILON = FIXED_E8_NORMALIZATION_EPSILON


class StrengthPreservingStatus(str, Enum):
    JOINT_WRITE = "JOINT_WRITE"
    ZERO_DEMAND_TARGET_ONLY = "ZERO_DEMAND_TARGET_ONLY"
    WRITER_UNREACHABLE = "WRITER_UNREACHABLE"
    NO_ROUTING_DOF = "NO_STRENGTH_PRESERVING_ROUTING_DOF"


@dataclass(frozen=True, slots=True)
class StrengthSolverCertificate:
    phase: str
    success: bool
    status: int
    message_sha256: str
    iterations: int
    finite: bool
    maximum_bound_violation: float
    equality_residual: float
    maximum_soft_risk_violation: float
    stationarity_residual: float
    first_false_component: str | None
    passed: bool
    backend: str = "scipy-slsqp-float64"
    backend_version: str = SCIPY_VERSION
    ftol: float = FIXED_E8_SOLVER_FTOL
    primal_tolerance: float = STRENGTH_EQUALITY_TOLERANCE
    kkt_tolerance: float = FIXED_E8_KKT_TOLERANCE

    @property
    def optimizer_pass_count(self) -> int:
        return 1

    @property
    def fallback_invocation_count(self) -> int:
        return 0

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StrengthPreservingRoutingResult:
    arm: FixedE8Arm
    mode: FixedE8StepMode
    status: StrengthPreservingStatus
    signed_slopes: tuple[float, ...]
    active_direction_mask: tuple[bool, ...]
    p_max_velocity: tuple[float, ...]
    p_max: float
    requested_progress: float
    alpha_req: float
    alpha_max: float
    alpha_apply: float
    coverage: float
    pre_soft_velocity: tuple[float, ...]
    soft_velocity: tuple[float, ...]
    velocity: tuple[float, ...]
    applied_coefficient: tuple[float, ...]
    contribution_share: tuple[float, ...]
    scores: tuple[FixedE8Score, ...]
    xi_star: float | None
    authoritative_certificates: tuple[StrengthSolverCertificate, ...]
    diagnostic_shadow_certificates: tuple[StrengthSolverCertificate, ...]
    selected_solution_source: str
    stage1_selection_fallback_count: int
    soft_shadow_status: str
    active_layer_count: int
    equality_rank: int
    feasible_allocation_dimension: int
    selected_bound_fixed_count: int
    neutral_soft_coefficient_distance: float
    equality_residual: float
    risk_difference_soft_minus_neutral: tuple[tuple[str, float], ...]
    identity_sha256: str

    @property
    def certificates(self) -> tuple[StrengthSolverCertificate, ...]:
        return self.authoritative_certificates + self.diagnostic_shadow_certificates

    @property
    def stage2_failure_policy(self) -> str:
        return "FAIL_CLOSED"

    @property
    def failed_authoritative_certificates(self) -> tuple[()]:
        return ()

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "method_id": STRENGTH_PRESERVING_METHOD_ID,
                "instruction_id": STRENGTH_PRESERVING_INSTRUCTION_ID,
                "amendment_id": STRENGTH_PRESERVING_AMENDMENT_ID,
                "progress_floor": 0.0,
                "kappa": 0.0,
                "legacy_requested_progress": None,
                "hard_h_p_budget_influence_count": 0,
                "functional_veto_count": 0,
                "retry_count": 0,
                "backtracking_count": 0,
                "capacity_primary_objective": False,
                "overlay_authoritative_access_count": 0,
                "transport_decision_influence_count": 0,
                "target_probe_arm_dependent_influence_count": 0,
                "h": float(FIXED_E8_H),
                "h_application_count": 1,
            }
        )
        payload["identity_sha256"] = self.identity_sha256
        return payload


def target_probe_requested_strength(
    target_velocity_receipt: Mapping[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Recover ``-g dot (h F_z)`` from the frozen R13 target receipt.

    The R13 scale receipt stores the FP64 gradient norms used to construct its
    unit allocation.  This avoids a second target-gradient call and binds the
    requested strength to the exact zero-coefficient probe.
    """

    velocity = tuple(float(item) for item in target_velocity_receipt.get(
        "velocity_coefficient", ()
    ))
    applied = tuple(float(item) for item in target_velocity_receipt.get(
        "target_probe_applied_coefficient", ()
    ))
    scale = target_velocity_receipt.get("scale_velocity")
    if (
        velocity != (0.0,) * len(FIXED_E8_LAYER_ORDER)
        or applied != (0.0,) * len(FIXED_E8_LAYER_ORDER)
        or not isinstance(scale, Mapping)
    ):
        raise ODEBFContractError("strength target probe is not route independent")
    speed = float(scale["shared_speed"])
    norms = np.asarray(scale["gradient_norm_by_request"], dtype=np.float64)
    frobenius = float(scale["gradient_frobenius_norm"])
    allocation = str(scale["allocation"])
    if (
        speed <= 0.0
        or norms.shape != (10,)
        or not np.all(np.isfinite(norms))
        or np.any(norms < 0.0)
        or not math.isfinite(frobenius)
        or frobenius < 0.0
    ):
        raise ODEBFContractError("strength target probe gradient scale differs")
    if allocation == "PER_REQUEST_EQUAL_NONZERO_EUCLIDEAN_SPEED":
        formula_value = float(FIXED_E8_H) * speed * float(norms.sum())
        formula = "h*shared_speed*sum_i(||gradient_i||_2)"
    elif allocation == "BATCH_GLOBAL_MATCHED_TOTAL_FROBENIUS_SPEED":
        formula_value = (
            float(FIXED_E8_H) * speed * math.sqrt(10.0) * frobenius
        )
        formula = "h*shared_speed*sqrt(B10)*||gradient||_F"
    else:
        raise ODEBFContractError("strength target probe allocation differs")
    alpha_req = float(
        target_velocity_receipt["target_new_nll_applied_step_reduction"]
    )
    if not math.isfinite(alpha_req) or alpha_req < 0.0:
        raise ODEBFContractError("strength target demand is non-finite")
    cast_delta = alpha_req - formula_value
    payload = {
        "schema": "ode-edit-s05-p1r19-common-target-demand/v1",
        "target_velocity_receipt_sha256": target_velocity_receipt["identity_sha256"],
        "target_velocity_sha256": target_velocity_receipt["velocity_sha256"],
        "target_scale_identity_sha256": scale["scale_identity_sha256"],
        "zero_probe_velocity_coefficients": list(velocity),
        "zero_probe_applied_coefficients": list(applied),
        "allocation": allocation,
        "alpha_req": alpha_req,
        "alpha_req_definition": (
            "-dot(existing_target_gradient,h*model_facing_F_z)"
        ),
        "ideal_fp64_norm_formula": formula,
        "ideal_fp64_norm_formula_value": formula_value,
        "model_facing_fp32_cast_delta": cast_delta,
        "additional_target_graph_count": 0,
        "same_b10_six_context_target_new_suffix_nll": True,
        "same_applied_step_units_as_weight_slope": True,
        "target_probe_arm_dependent_influence_count": 0,
        "native_or_direct_z_access_count": 0,
        "h": float(FIXED_E8_H),
        "h_application_count": 1,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return alpha_req, payload


def route_independent_target_write_identity(
    routing: StrengthPreservingRoutingResult,
    increment_theta: Sequence[float],
    target_velocity_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind common zero target probe and the one physical ``h*v`` write."""

    probe_velocity = tuple(
        float(item) for item in target_velocity_receipt["velocity_coefficient"]
    )
    probe_applied = tuple(
        float(item)
        for item in target_velocity_receipt["target_probe_applied_coefficient"]
    )
    physical = tuple(float(item) for item in increment_theta)
    expected = tuple(float(item) for item in routing.applied_coefficient)
    if (
        probe_velocity != (0.0,) * len(FIXED_E8_LAYER_ORDER)
        or probe_applied != (0.0,) * len(FIXED_E8_LAYER_ORDER)
        or len(physical) != len(FIXED_E8_LAYER_ORDER)
        or not np.allclose(physical, expected, rtol=0.0, atol=1.0e-14)
    ):
        raise ODEBFContractError("strength target probe/write identity differs")
    payload = {
        "schema": "ode-edit-s05-p1r19-route-independent-target-write/v1",
        "target_velocity_receipt_sha256": target_velocity_receipt["identity_sha256"],
        "routing_sha256": routing.identity_sha256,
        "zero_target_probe_velocity_coefficient": list(probe_velocity),
        "zero_target_probe_applied_coefficient": list(probe_applied),
        "physical_write_applied_coefficient": list(physical),
        "target_probe_h": float(FIXED_E8_H),
        "physical_write_h": float(FIXED_E8_H),
        "target_probe_h_application_count": 1,
        "physical_write_h_application_count": 1,
        "target_probe_arm_dependent_influence_count": 0,
        "target_probe_mutates_weight_factor_history_count": 0,
        "physical_write_authoritative_candidate_count": 1,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _capacity_value(problem: RoutingProblem, value: np.ndarray) -> float:
    return float(0.5 * value @ problem.capacity_metric @ value)


def _risk_functions(
    problem: RoutingProblem,
    inventory: FixedE8SoftInventory,
) -> tuple[tuple[str, Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]], ...]:
    functions: list[
        tuple[str, Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]
    ] = []

    def structural(label: str, barrier: Any) -> None:
        normalization = max(
            FIXED_E8_NORMALIZATION_EPSILON, float(np.trace(barrier.gram))
        )
        functions.append(
            (
                label,
                lambda value, b=barrier, scale=normalization: float(
                    (2.0 * b.linear @ value + value @ b.gram @ value) / scale
                ),
                lambda value, b=barrier, scale=normalization: (
                    2.0 * b.linear + 2.0 * b.gram @ value
                )
                / scale,
            )
        )

    if inventory.history_item_count > 0:
        structural("structural_historical", problem.historical)
    structural("structural_pretrained", problem.pretrained)
    for metric in inventory.active_metrics():
        positive = metric.positive_increment
        normalization = metric.normalization
        functions.append(
            (
                metric.label,
                lambda value, slope=positive, scale=normalization: float(
                    slope @ value / scale
                ),
                lambda value, slope=positive, scale=normalization: slope / scale,
            )
        )
    return tuple(functions)


def _score_inventory(
    problem: RoutingProblem,
    inventory: FixedE8SoftInventory,
    velocity: np.ndarray,
    *,
    influence_count: int,
) -> tuple[FixedE8Score, ...]:
    return (
        structural_soft_score(
            problem.historical,
            velocity,
            active=inventory.history_item_count > 0,
            influence_count=influence_count,
        ),
        structural_soft_score(
            problem.pretrained,
            velocity,
            active=True,
            influence_count=influence_count,
        ),
        *(
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


def _certificate(
    *,
    phase: str,
    result: Any,
    value: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    slopes: np.ndarray,
    alpha: float,
    objective_gradient: np.ndarray,
    risk_slacks: Sequence[float] = (),
    risk_gradients: Sequence[np.ndarray] = (),
) -> StrengthSolverCertificate:
    finite = bool(
        np.all(np.isfinite(value))
        and np.all(np.isfinite(objective_gradient))
        and all(math.isfinite(float(item)) for item in risk_slacks)
    )
    bound_violation = max(
        float(np.max(lower - value, initial=0.0)),
        float(np.max(value - upper, initial=0.0)),
        0.0,
    )
    equality_residual = abs(float(slopes @ value[: slopes.size]) - alpha)
    risk_violation = max(
        (max(-float(item), 0.0) for item in risk_slacks), default=0.0
    )
    active_columns: list[np.ndarray] = []
    # The equality multiplier is free.  Bounds and risk constraints are
    # included as active signed gradients for a deterministic residual audit.
    equality_column = np.zeros_like(value)
    equality_column[: slopes.size] = slopes
    active_columns.extend((equality_column, -equality_column))
    for index in range(value.size):
        if value[index] <= 10.0 * STRENGTH_EQUALITY_TOLERANCE:
            column = np.zeros_like(value)
            column[index] = -1.0
            active_columns.append(column)
        if upper[index] - value[index] <= 10.0 * STRENGTH_EQUALITY_TOLERANCE:
            column = np.zeros_like(value)
            column[index] = 1.0
            active_columns.append(column)
    for slack, gradient in zip(risk_slacks, risk_gradients, strict=True):
        if float(slack) <= 10.0 * STRENGTH_EQUALITY_TOLERANCE:
            active_columns.append(-np.asarray(gradient, dtype=np.float64))
    if not finite:
        stationarity = math.inf
    elif active_columns:
        matrix = np.stack(active_columns, axis=1)
        multipliers, _ = nnls(matrix, -objective_gradient)
        stationarity = float(
            np.linalg.norm(objective_gradient + matrix @ multipliers, ord=np.inf)
        )
    else:
        stationarity = float(np.linalg.norm(objective_gradient, ord=np.inf))
    success = bool(getattr(result, "success", False))
    first_false: str | None = None
    if not finite or not math.isfinite(stationarity):
        first_false = "finite"
    elif bound_violation > STRENGTH_EQUALITY_TOLERANCE:
        first_false = "bounds"
    elif equality_residual > STRENGTH_EQUALITY_TOLERANCE:
        first_false = "strength_equality"
    elif risk_violation > STRENGTH_EQUALITY_TOLERANCE:
        first_false = "soft_risk_tie"
    elif stationarity > FIXED_E8_KKT_TOLERANCE:
        first_false = "stationarity"
    elif not success:
        first_false = "solver_success"
    passed = first_false is None
    return StrengthSolverCertificate(
        phase=phase,
        success=success,
        status=int(getattr(result, "status", -1)),
        message_sha256=canonical_hash(
            {"optimizer_message": str(getattr(result, "message", ""))}
        ),
        iterations=int(getattr(result, "nit", 0)),
        finite=finite,
        maximum_bound_violation=bound_violation,
        equality_residual=equality_residual,
        maximum_soft_risk_violation=risk_violation,
        stationarity_residual=stationarity,
        first_false_component=first_false,
        passed=passed,
    )


def _solve_neutral(
    problem: RoutingProblem,
    active: np.ndarray,
    alpha: float,
    initial: np.ndarray,
) -> tuple[np.ndarray, StrengthSolverCertificate]:
    slopes = problem.signed_progress[active]
    caps = np.minimum(problem.layer_caps[active], 1.0)
    metric = problem.capacity_metric[np.ix_(active, active)]
    result = minimize(
        lambda value: float(0.5 * value @ metric @ value),
        initial,
        jac=lambda value: metric @ value,
        method="SLSQP",
        bounds=tuple((0.0, float(cap)) for cap in caps),
        constraints=(
            {
                "type": "eq",
                "fun": lambda value: float(slopes @ value - alpha),
                "jac": lambda value: slopes.copy(),
            },
        ),
        options={
            "disp": False,
            "ftol": FIXED_E8_SOLVER_FTOL,
            "maxiter": FIXED_E8_SOLVER_MAXITER,
        },
    )
    value = np.asarray(result.x, dtype=np.float64)
    cert = _certificate(
        phase="strength-neutral-minimum-capacity",
        result=result,
        value=value,
        lower=np.zeros_like(value),
        upper=caps,
        slopes=slopes,
        alpha=alpha,
        objective_gradient=metric @ value,
    )
    if not cert.passed:
        raise ODEBFContractError(
            f"strength Neutral certificate failed: {cert.raw_free_payload()}"
        )
    return value, cert


def _solve_soft(
    problem: RoutingProblem,
    inventory: FixedE8SoftInventory,
    active: np.ndarray,
    alpha: float,
    neutral: np.ndarray,
) -> tuple[np.ndarray, float, tuple[StrengthSolverCertificate, ...]]:
    slopes = problem.signed_progress[active]
    caps = np.minimum(problem.layer_caps[active], 1.0)
    risks = _risk_functions(problem, inventory)
    initial_full = np.zeros(problem.signed_progress.size, dtype=np.float64)
    initial_full[active] = neutral
    initial_xi = max(0.0, *(risk(initial_full) for _, risk, _ in risks))
    initial = np.concatenate((neutral, [initial_xi]))

    def expand(value: np.ndarray) -> np.ndarray:
        full = np.zeros(problem.signed_progress.size, dtype=np.float64)
        full[active] = value[: active.size]
        return full

    constraints: list[dict[str, Any]] = [
        {
            "type": "eq",
            "fun": lambda value: float(slopes @ value[: active.size] - alpha),
            "jac": lambda value: np.concatenate((slopes, [0.0])),
        }
    ]
    for _, risk, gradient in risks:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda value, fn=risk: float(value[-1] - fn(expand(value))),
                "jac": lambda value, grad=gradient: np.concatenate(
                    (-grad(expand(value))[active], [1.0])
                ),
            }
        )
    stage1 = minimize(
        lambda value: float(value[-1]),
        initial,
        jac=lambda value: np.concatenate((np.zeros(active.size), [1.0])),
        method="SLSQP",
        bounds=tuple((0.0, float(cap)) for cap in caps) + ((None, None),),
        constraints=tuple(constraints),
        options={
            "disp": False,
            "ftol": FIXED_E8_SOLVER_FTOL,
            "maxiter": FIXED_E8_SOLVER_MAXITER,
        },
    )
    value1 = np.asarray(stage1.x, dtype=np.float64)
    full1 = expand(value1)
    risk_slacks1 = [float(value1[-1] - fn(full1)) for _, fn, _ in risks]
    risk_gradients1 = [
        np.concatenate((-grad(full1)[active], [1.0])) for _, _, grad in risks
    ]
    cert1 = _certificate(
        phase="strength-soft-minimum-worst-risk",
        result=stage1,
        value=value1,
        lower=np.concatenate((np.zeros(active.size), [-np.inf])),
        upper=np.concatenate((caps, [np.inf])),
        slopes=slopes,
        alpha=alpha,
        objective_gradient=np.concatenate((np.zeros(active.size), [1.0])),
        risk_slacks=risk_slacks1,
        risk_gradients=risk_gradients1,
    )
    if not cert1.passed:
        raise ODEBFContractError(
            f"strength Soft stage1 certificate failed: {cert1.raw_free_payload()}"
        )
    xi_star = max(float(value1[-1]), 0.0)
    metric = problem.capacity_metric[np.ix_(active, active)]
    xi_cap = xi_star + FIXED_E8_XI_TIE_TOLERANCE
    constraints2: list[dict[str, Any]] = [
        {
            "type": "eq",
            "fun": lambda value: float(slopes @ value - alpha),
            "jac": lambda value: slopes.copy(),
        }
    ]
    for _, risk, gradient in risks:
        constraints2.append(
            {
                "type": "ineq",
                "fun": lambda value, fn=risk: float(xi_cap - fn(expand(value))),
                "jac": lambda value, grad=gradient: -grad(expand(value))[active],
            }
        )
    stage2 = minimize(
        lambda value: float(0.5 * value @ metric @ value),
        value1[: active.size],
        jac=lambda value: metric @ value,
        method="SLSQP",
        bounds=tuple((0.0, float(cap)) for cap in caps),
        constraints=tuple(constraints2),
        options={
            "disp": False,
            "ftol": FIXED_E8_SOLVER_FTOL,
            "maxiter": FIXED_E8_SOLVER_MAXITER,
        },
    )
    value2 = np.asarray(stage2.x, dtype=np.float64)
    full2 = np.zeros(problem.signed_progress.size, dtype=np.float64)
    full2[active] = value2
    risk_slacks2 = [float(xi_cap - fn(full2)) for _, fn, _ in risks]
    risk_gradients2 = [-grad(full2)[active] for _, _, grad in risks]
    cert2 = _certificate(
        phase="strength-soft-minimum-capacity-within-xi-tie",
        result=stage2,
        value=value2,
        lower=np.zeros(active.size),
        upper=caps,
        slopes=slopes,
        alpha=alpha,
        objective_gradient=metric @ value2,
        risk_slacks=risk_slacks2,
        risk_gradients=risk_gradients2,
    )
    if not cert2.passed:
        raise ODEBFContractError(
            f"strength Soft stage2 certificate failed: {cert2.raw_free_payload()}"
        )
    return value2, xi_star, (cert1, cert2)


def solve_strength_preserving_routing(
    problem: RoutingProblem,
    inventory: FixedE8SoftInventory,
    *,
    arm: FixedE8Arm | str,
    alpha_req: float,
) -> StrengthPreservingRoutingResult:
    """Solve one exact matched-strength Neutral or Soft allocation."""

    selected = FixedE8Arm(arm)
    if selected not in (FixedE8Arm.NEUTRAL, FixedE8Arm.SOFT):
        raise ODEBFContractError("strength routing arm differs")
    dimension = len(FIXED_E8_LAYER_ORDER)
    slopes = np.asarray(problem.signed_progress, dtype=np.float64)
    caps = np.minimum(np.asarray(problem.layer_caps, dtype=np.float64), 1.0)
    if (
        slopes.shape != (dimension,)
        or caps.shape != (dimension,)
        or problem.capacity_metric.shape != (dimension, dimension)
        or not np.all(np.isfinite(slopes))
        or not np.all(np.isfinite(caps))
        or np.any(caps < 0.0)
        or not math.isfinite(float(alpha_req))
        or alpha_req < 0.0
    ):
        raise ODEBFContractError("strength routing geometry differs")
    if (
        not np.all(np.isfinite(problem.capacity_metric))
        or not np.allclose(
            problem.capacity_metric,
            problem.capacity_metric.T,
            rtol=0.0,
            atol=1.0e-14,
        )
        or float(np.linalg.eigvalsh(problem.capacity_metric).min()) <= 0.0
    ):
        raise ODEBFContractError(
            "strength capacity metric is not strictly positive definite"
        )
    active = np.flatnonzero(slopes > 0.0)
    maximum = np.zeros(dimension, dtype=np.float64)
    maximum[active] = caps[active]
    alpha_max = float(slopes @ maximum)
    alpha_apply = min(float(alpha_req), alpha_max)
    coverage = (
        1.0
        if alpha_req == 0.0
        else alpha_apply / (alpha_req + STRENGTH_COVERAGE_EPSILON)
    )
    neutral_full = np.zeros(dimension, dtype=np.float64)
    soft_full = np.zeros(dimension, dtype=np.float64)
    authoritative: list[StrengthSolverCertificate] = []
    diagnostic: list[StrengthSolverCertificate] = []
    xi_star: float | None = None
    if alpha_apply > 0.0:
        active_slopes = slopes[active]
        active_caps = caps[active]
        initial = active_caps * (alpha_apply / alpha_max)
        neutral_active, neutral_certificate = _solve_neutral(
            problem, active, alpha_apply, initial
        )
        neutral_full[active] = neutral_active
        if selected is FixedE8Arm.NEUTRAL:
            authoritative.append(neutral_certificate)
            try:
                soft_active, xi_star, soft_certificates = _solve_soft(
                    problem, inventory, active, alpha_apply, neutral_active
                )
            except ODEBFContractError as exc:
                # A detached Soft shadow is useful matched-allocation
                # telemetry, but it must never stop the independent Neutral
                # scientific trajectory.
                soft_full[:] = neutral_full
                soft_shadow_status = (
                    "DIAGNOSTIC_UNAVAILABLE_"
                    + canonical_hash({"exception": str(exc)})
                )
            else:
                soft_full[active] = soft_active
                diagnostic.extend(soft_certificates)
                soft_shadow_status = "AVAILABLE"
        else:
            soft_active, xi_star, soft_certificates = _solve_soft(
                problem, inventory, active, alpha_apply, neutral_active
            )
            soft_full[active] = soft_active
            diagnostic.append(neutral_certificate)
            authoritative.extend(soft_certificates)
            soft_shadow_status = "AUTHORITATIVE_SOFT_NOT_SHADOW"
    else:
        soft_shadow_status = "NOT_APPLICABLE_ZERO_APPLIED_STRENGTH"
    selected_value = neutral_full if selected is FixedE8Arm.NEUTRAL else soft_full
    equality_residual = abs(float(slopes @ selected_value) - alpha_apply)
    if equality_residual > STRENGTH_EQUALITY_TOLERANCE:
        raise ODEBFContractError("strength equality certificate differs")
    intrinsic_dimension = (
        0
        if active.size <= 1
        or alpha_apply == 0.0
        or abs(alpha_apply - alpha_max) <= STRENGTH_EQUALITY_TOLERANCE
        else int(active.size - 1)
    )
    status = (
        StrengthPreservingStatus.ZERO_DEMAND_TARGET_ONLY
        if alpha_req == 0.0
        else StrengthPreservingStatus.WRITER_UNREACHABLE
        if alpha_apply < alpha_req
        else StrengthPreservingStatus.NO_ROUTING_DOF
        if intrinsic_dimension == 0
        else StrengthPreservingStatus.JOINT_WRITE
    )
    selected_bound_fixed = int(
        sum(
            value <= STRENGTH_EQUALITY_TOLERANCE
            or cap - value <= STRENGTH_EQUALITY_TOLERANCE
            for value, cap in zip(selected_value[active], caps[active], strict=True)
        )
    )
    selected_progress = float(slopes @ selected_value)
    if alpha_apply > 0.0 and selected_progress <= 0.0:
        raise ODEBFContractError(
            "positive strength was attenuated to a zero writer"
        )
    denominator = alpha_apply + STRENGTH_COVERAGE_EPSILON
    shares = slopes * selected_value / denominator
    neutral_scores = _score_inventory(
        problem, inventory, neutral_full, influence_count=0
    )
    soft_scores = _score_inventory(
        problem,
        inventory,
        soft_full,
        influence_count=1 if selected is FixedE8Arm.SOFT else 0,
    )
    scores = neutral_scores if selected is FixedE8Arm.NEUTRAL else soft_scores
    risk_difference = tuple(
        (left.label, float(right.score - left.score))
        for left, right in zip(neutral_scores, soft_scores, strict=True)
    )
    payload = {
        "method_id": STRENGTH_PRESERVING_METHOD_ID,
        "arm": selected.value,
        "problem_sha256": problem.identity(),
        "functional_inventory_sha256": inventory.raw_free_payload()["identity_sha256"],
        "signed_slopes": slopes.tolist(),
        "active_direction_mask": [bool(item > 0.0) for item in slopes],
        "alpha_req": float(alpha_req),
        "alpha_max": alpha_max,
        "alpha_apply": alpha_apply,
        "coverage": coverage,
        "neutral_velocity": neutral_full.tolist(),
        "soft_velocity": soft_full.tolist(),
        "selected_velocity": selected_value.tolist(),
        "selected_progress": selected_progress,
        "xi_star": xi_star,
        "feasible_allocation_dimension": intrinsic_dimension,
        "certificates": [item.raw_free_payload() for item in (*authoritative, *diagnostic)],
    }
    identity = canonical_hash(payload)
    return StrengthPreservingRoutingResult(
        arm=selected,
        mode=(
            FixedE8StepMode.ZERO_WRITE_TARGET_RECOVERY
            if alpha_apply == 0.0
            else FixedE8StepMode.JOINT_WRITE
        ),
        status=status,
        signed_slopes=tuple(float(item) for item in slopes),
        active_direction_mask=tuple(bool(item > 0.0) for item in slopes),
        p_max_velocity=tuple(float(item) for item in maximum),
        p_max=alpha_max,
        requested_progress=alpha_apply,
        alpha_req=float(alpha_req),
        alpha_max=alpha_max,
        alpha_apply=alpha_apply,
        coverage=coverage,
        pre_soft_velocity=tuple(float(item) for item in neutral_full),
        soft_velocity=tuple(float(item) for item in soft_full),
        velocity=tuple(float(item) for item in selected_value),
        applied_coefficient=tuple(
            float(FIXED_E8_H) * float(item) for item in selected_value
        ),
        contribution_share=tuple(float(item) for item in shares),
        scores=scores,
        xi_star=xi_star,
        authoritative_certificates=tuple(authoritative),
        diagnostic_shadow_certificates=tuple(diagnostic),
        selected_solution_source=(
            "EXACT_STRENGTH_NEUTRAL_CAPACITY"
            if selected is FixedE8Arm.NEUTRAL
            else "EXACT_STRENGTH_SOFT_XI_THEN_CAPACITY"
        ),
        stage1_selection_fallback_count=0,
        soft_shadow_status=soft_shadow_status,
        active_layer_count=int(active.size),
        equality_rank=0 if alpha_apply <= STRENGTH_COVERAGE_EPSILON else 1,
        feasible_allocation_dimension=intrinsic_dimension,
        selected_bound_fixed_count=selected_bound_fixed,
        neutral_soft_coefficient_distance=float(np.linalg.norm(soft_full - neutral_full)),
        equality_residual=equality_residual,
        risk_difference_soft_minus_neutral=risk_difference,
        identity_sha256=identity,
    )


__all__ = [
    "STRENGTH_COVERAGE_EPSILON",
    "STRENGTH_EQUALITY_TOLERANCE",
    "STRENGTH_PRESERVING_AMENDMENT_ID",
    "STRENGTH_PRESERVING_INSTRUCTION_ID",
    "STRENGTH_PRESERVING_METHOD_ID",
    "StrengthPreservingRoutingResult",
    "StrengthPreservingStatus",
    "StrengthSolverCertificate",
    "solve_strength_preserving_routing",
    "route_independent_target_write_identity",
    "target_probe_requested_strength",
]
