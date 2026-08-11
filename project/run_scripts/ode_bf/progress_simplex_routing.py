"""Fixed-progress simplex routing for the P1R23 B10 method revision.

The legacy P1R23 strength router represents the reachable write by a boxed
coefficient vector.  This module instead keeps its *applied-step* physical
progress fixed and represents its distribution over positive no-hook writer
directions by a simplex.  It is intentionally detached from target evolution,
model execution, retries, and endpoint evaluation.

``RoutingProblem.signed_progress`` is already in applied-step coordinates
(``h * B``).  Therefore this module never applies ``h`` while forming ``q``;
the companion factor builder applies it exactly once when constructing the
physical :class:`WaypointFactor` objects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import __version__ as SCIPY_VERSION
from scipy.optimize import minimize

from .contracts import ODEBFContractError, canonical_hash
from .fixed_e8_soft_routing import (
    FIXED_E8_H,
    FIXED_E8_KKT_TOLERANCE,
    FIXED_E8_LAYER_ORDER,
    FIXED_E8_NORMALIZATION_EPSILON,
    FIXED_E8_SOLVER_FTOL,
    FIXED_E8_SOLVER_MAXITER,
    FixedE8Arm,
    FixedE8Score,
    FixedE8SoftInventory,
    FixedE8StepMode,
    functional_soft_score,
    structural_soft_score,
)
from .functional import WaypointFactor
from .routing import RoutingProblem


PROGRESS_SIMPLEX_INSTRUCTION_ID = "ODEEDIT-S05-P1R23-PROGRESS-SIMPLEX-BF-ROUTER-V1"
PROGRESS_SIMPLEX_METHOD_ID = "P1R23-DYNAMIC-PROGRESS-SIMPLEX-PHYSICAL-WONLY-ROUTER-V1"

# These are numerical certificates, frozen before any model outcome.  They
# are deliberately separate from the inherited solver ftol/KKT diagnostics.
SIMPLEX_PRIMAL_TOLERANCE = 1.0e-8
SIMPLEX_ENERGY_RELATIVE_TOLERANCE = 1.0e-8
SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE = 1.0e-12
SIMPLEX_XI_TIE_TOLERANCE = 1.0e-8


class ProgressSimplexStatus(str, Enum):
    """Typed, non-rescue outcomes for one routed physical field."""

    JOINT_WRITE = "JOINT_WRITE"
    NO_POSITIVE_DIRECTION = "NO_POSITIVE_DIRECTION"
    NO_ROUTING_DOF = "NO_ROUTING_DOF"
    NO_SAFE_REDISTRIBUTION = "NO_SAFE_REDISTRIBUTION"


@dataclass(frozen=True, slots=True)
class ProgressSimplexCertificate:
    """Raw-free certificate for a direct-simplex optimization stage.

    KKT is retained as telemetry only for this activation: a large value is
    reported but is not a silently widened feasibility criterion.
    """

    phase: str
    success: bool
    status: int
    message_sha256: str
    iterations: int
    finite: bool
    simplex_sum_residual: float
    simplex_negative_violation: float
    progress_equality_residual: float
    energy_relative_ratio: float
    energy_violation: float
    stationarity_telemetry: float
    first_false_component: str | None
    passed: bool
    backend: str = "scipy-slsqp-float64"
    backend_version: str = SCIPY_VERSION
    ftol: float = FIXED_E8_SOLVER_FTOL
    simplex_tolerance: float = SIMPLEX_PRIMAL_TOLERANCE
    energy_relative_tolerance: float = SIMPLEX_ENERGY_RELATIVE_TOLERANCE
    energy_absolute_tolerance: float = SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE
    kkt_telemetry_tolerance: float = FIXED_E8_KKT_TOLERANCE

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProgressSimplexRoutingResult:
    """The selected arm plus same-state Neutral/Soft routing telemetry."""

    arm: FixedE8Arm
    mode: FixedE8StepMode
    status: ProgressSimplexStatus
    signed_slopes: tuple[float, ...]
    active_direction_mask: tuple[bool, ...]
    q: float
    alpha_req: float
    alpha_max: float
    alpha_apply: float
    coverage: float
    neutral_pi: tuple[float, ...]
    soft_pi: tuple[float, ...]
    pi: tuple[float, ...]
    selected_pi_entropy: float
    selected_pi_effective_layer_count: float
    selected_pi_top1_share: float
    selected_pi_top1_index: int | None
    selected_pi_top1_layer_id: int | None
    neutral_velocity: tuple[float, ...]
    soft_velocity: tuple[float, ...]
    velocity: tuple[float, ...]
    neutral_mapping_max_abs: float
    applied_coefficient: tuple[float, ...]
    predicted_contribution: tuple[float, ...]
    predicted_progress: float
    equality_residual: float
    simplex_sum_residual: float
    simplex_minimum: float
    active_layer_count: int
    equality_rank: int
    feasible_allocation_dimension: int
    maximum_velocity: float
    velocity_above_one_count: int
    neutral_energy: float
    soft_energy: float
    selected_energy: float
    neutral_energy_ratio: float
    soft_energy_ratio: float
    selected_energy_ratio: float
    global_energy_soft_neutral_ratio: float
    neutral_capacity: float
    soft_capacity: float
    selected_capacity: float
    xi_star: float | None
    scores: tuple[FixedE8Score, ...]
    neutral_scores: tuple[FixedE8Score, ...]
    soft_scores: tuple[FixedE8Score, ...]
    neutral_p_h_risk: float
    soft_p_h_risk: float
    selected_p_h_risk: float
    risk_difference_soft_minus_neutral: float
    risk_difference_soft_minus_neutral_by_metric: tuple[tuple[str, float], ...]
    certificates: tuple[ProgressSimplexCertificate, ...]
    neutral_soft_coefficient_l1: float
    neutral_soft_coefficient_l2: float
    neutral_soft_coefficient_cosine: float
    selected_solution_source: str
    soft_objective_decision_influence_count: int
    hard_h_p_budget_influence_count: int
    functional_veto_count: int
    retry_count: int
    backtracking_count: int
    structural_only: bool
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.structural_only:
            from .compute_progress_simplex_runtime import (
                COMPUTE_PROGRESS_SIMPLEX_INSTRUCTION_ID,
                COMPUTE_PROGRESS_SIMPLEX_METHOD_ID,
            )
            instruction_id = COMPUTE_PROGRESS_SIMPLEX_INSTRUCTION_ID
            method_id = COMPUTE_PROGRESS_SIMPLEX_METHOD_ID
        else:
            instruction_id = PROGRESS_SIMPLEX_INSTRUCTION_ID
            method_id = PROGRESS_SIMPLEX_METHOD_ID
        payload.update(
            {
                "instruction_id": instruction_id,
                "method_id": method_id,
                "progress_definition": "q=sum(a_l for a_l>0), applied-step units",
                "parameterization": "v_l=q*pi_l/a_l on active directions",
                "authoritative_slope": "PHYSICAL_W_ONLY_NOHOOK_APPLIED_STEP",
                "per_layer_upper_cap_influence_count": 0,
                "postsolve_per_layer_clipping_count": 0,
                "alpha_req_router_decision_influence_count": 0,
                "target_probe_arm_dependent_influence_count": 0,
                "overlay_authoritative_access_count": 0,
                "transport_decision_influence_count": 0,
                "functional_routing_influence_count": (
                    0 if self.structural_only else 1
                ),
                "selected_pi_entropy_definition": "-sum(pi_l * log(pi_l)) over positive selected pi",
                "selected_pi_top1_index_convention": "zero-based index in locked layer order [4,5,6,7,8]",
                "selected_pi_top1_layer_id_convention": "physical transformer layer identifier",
                "global_energy_soft_neutral_ratio_definition": "soft_energy / neutral_energy",
                "capacity_definition": "0.5 * v^T capacity_metric * v",
                "risk_difference_soft_minus_neutral_definition": "max normalized active P/H score (Soft) minus Neutral",
                "h": float(FIXED_E8_H),
                "h_application_count": 1,
            }
        )
        payload["identity_sha256"] = self.identity_sha256
        return payload


@dataclass(frozen=True, slots=True)
class StructuralOnlyRoutingInventory:
    """Zero-model-call inventory for the compute-aware router.

    P1R23's exact router consumed functional endpoint probes at every field.
    The compute addendum makes only the already-built structural pretrained
    barrier and the fixed-rank historical barrier authoritative.  This small
    receipt prevents a synthetic functional inventory from being mistaken for
    an evaluated one.
    """

    history_item_count: int
    controller_batch_sha256: str
    field_sha256: str
    historical_sketch_sha256: str

    def __post_init__(self) -> None:
        if (
            self.history_item_count < 0
            or any(
                not isinstance(item, str) or len(item) != 64
                for item in (
                    self.controller_batch_sha256,
                    self.field_sha256,
                    self.historical_sketch_sha256,
                )
            )
        ):
            raise ODEBFContractError("compute-aware structural inventory differs")

    def raw_free_payload(self) -> dict[str, Any]:
        payload = {
            "schema": "ode-edit-s05-p1r23-compute-structural-inventory/v1",
            "history_item_count": self.history_item_count,
            "controller_batch_sha256": self.controller_batch_sha256,
            "field_sha256": self.field_sha256,
            "historical_sketch_sha256": self.historical_sketch_sha256,
            "structural_pretrained_active": True,
            "structural_historical_active": self.history_item_count > 0,
            "functional_probe_count": 0,
            "functional_routing_influence_count": 0,
            "model_forward_count": 0,
            "backward_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


def _risk_functions(
    problem: RoutingProblem,
    inventory: FixedE8SoftInventory | StructuralOnlyRoutingInventory,
) -> tuple[
    tuple[str, Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]],
    ...
]:
    """Reuse P1R23's normalized structural/functional P/H semantics.

    The returned quantities are allocation costs only: no budget, veto, or
    candidate rejection is introduced by this module.
    """

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
    if isinstance(inventory, FixedE8SoftInventory):
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
    inventory: FixedE8SoftInventory | StructuralOnlyRoutingInventory,
    velocity: np.ndarray,
    *,
    influence_count: int,
) -> tuple[FixedE8Score, ...]:
    structural_scores = (
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
    )
    if isinstance(inventory, StructuralOnlyRoutingInventory):
        return structural_scores
    return (
        *structural_scores,
        *(
            functional_soft_score(metric, velocity, influence_count=influence_count)
            for metric in (
                inventory.functional_p,
                inventory.functional_h_mean,
                inventory.functional_h_smoothmax,
            )
        ),
    )


def _validate_problem(problem: RoutingProblem, alpha_req: float) -> tuple[np.ndarray, np.ndarray]:
    slopes = np.asarray(problem.signed_progress, dtype=np.float64)
    energy_metric = np.asarray(problem.trust_metric, dtype=np.float64)
    dimension = len(FIXED_E8_LAYER_ORDER)
    if (
        slopes.shape != (dimension,)
        or energy_metric.shape != (dimension, dimension)
        or problem.capacity_metric.shape != (dimension, dimension)
        or not np.all(np.isfinite(slopes))
        or not np.all(np.isfinite(energy_metric))
        or not np.all(np.isfinite(problem.capacity_metric))
        or not math.isfinite(float(alpha_req))
        or alpha_req < 0.0
    ):
        raise ODEBFContractError("progress simplex routing geometry differs")
    if not np.allclose(energy_metric, energy_metric.T, rtol=0.0, atol=1.0e-14):
        raise ODEBFContractError("progress simplex energy metric is not symmetric")
    if float(np.linalg.eigvalsh(energy_metric).min()) < -1.0e-10:
        raise ODEBFContractError("progress simplex energy metric is not PSD")
    if not np.allclose(
        problem.capacity_metric, problem.capacity_metric.T, rtol=0.0, atol=1.0e-14
    ):
        raise ODEBFContractError("progress simplex capacity metric is not symmetric")
    return slopes, energy_metric


def _energy(value: np.ndarray, metric: np.ndarray) -> float:
    return float(value @ metric @ value)


def _capacity(value: np.ndarray, metric: np.ndarray) -> float:
    """The inherited applied-step capacity objective, for receipt only."""

    return float(0.5 * value @ metric @ value)


def _energy_limit(neutral_energy: float) -> float:
    return (
        neutral_energy * (1.0 + SIMPLEX_ENERGY_RELATIVE_TOLERANCE)
        + SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE
    )


def _energy_ratio(value: float, neutral: float) -> float:
    if neutral <= SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE:
        return 1.0 if value <= SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE else math.inf
    return float(value / neutral)


def _full_velocity(
    pi_active: np.ndarray,
    active: np.ndarray,
    slopes: np.ndarray,
    q: float,
) -> np.ndarray:
    value = np.zeros(slopes.size, dtype=np.float64)
    value[active] = q * pi_active / slopes[active]
    return value


def _certificate(
    *,
    phase: str,
    result: Any,
    pi_active: np.ndarray,
    active: np.ndarray,
    slopes: np.ndarray,
    q: float,
    energy_metric: np.ndarray,
    neutral_energy: float,
    objective_gradient: np.ndarray,
) -> ProgressSimplexCertificate:
    full = _full_velocity(pi_active, active, slopes, q) if active.size else np.zeros_like(slopes)
    finite = bool(
        np.all(np.isfinite(pi_active))
        and np.all(np.isfinite(full))
        and np.all(np.isfinite(objective_gradient))
    )
    simplex_sum = abs(float(pi_active.sum()) - 1.0) if active.size else 0.0
    negative = max(float(-np.min(pi_active, initial=0.0)), 0.0)
    progress = float(slopes @ full)
    equality = abs(progress - q)
    selected_energy = _energy(full, energy_metric)
    limit = _energy_limit(neutral_energy)
    energy_violation = max(selected_energy - limit, 0.0)
    ratio = _energy_ratio(selected_energy, neutral_energy)
    stationarity = float(np.linalg.norm(objective_gradient, ord=np.inf))
    success = bool(getattr(result, "success", True))
    first_false: str | None = None
    if not finite:
        first_false = "finite"
    elif simplex_sum > SIMPLEX_PRIMAL_TOLERANCE:
        first_false = "simplex_sum"
    elif negative > SIMPLEX_PRIMAL_TOLERANCE:
        first_false = "simplex_nonnegative"
    elif equality > SIMPLEX_PRIMAL_TOLERANCE:
        first_false = "predicted_progress_equality"
    elif energy_violation > SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE:
        first_false = "neutral_relative_global_energy"
    elif not success:
        first_false = "solver_success"
    return ProgressSimplexCertificate(
        phase=phase,
        success=success,
        status=int(getattr(result, "status", 0)),
        message_sha256=canonical_hash(
            {"optimizer_message": str(getattr(result, "message", "direct"))}
        ),
        iterations=int(getattr(result, "nit", 0)),
        finite=finite,
        simplex_sum_residual=simplex_sum,
        simplex_negative_violation=negative,
        progress_equality_residual=equality,
        energy_relative_ratio=ratio,
        energy_violation=energy_violation,
        stationarity_telemetry=stationarity,
        first_false_component=first_false,
        passed=first_false is None,
    )


def _direct_neutral_certificate(
    *,
    pi_active: np.ndarray,
    active: np.ndarray,
    slopes: np.ndarray,
    q: float,
    energy_metric: np.ndarray,
    neutral_energy: float,
) -> ProgressSimplexCertificate:
    return _certificate(
        phase="simplex-neutral-identity",
        result=SimpleNamespace(success=True, status=0, message="direct-neutral", nit=0),
        pi_active=pi_active,
        active=active,
        slopes=slopes,
        q=q,
        energy_metric=energy_metric,
        neutral_energy=neutral_energy,
        objective_gradient=np.zeros_like(pi_active),
    )


def _solve_soft(
    problem: RoutingProblem,
    inventory: FixedE8SoftInventory | StructuralOnlyRoutingInventory,
    *,
    active: np.ndarray,
    slopes: np.ndarray,
    q: float,
    neutral_pi_active: np.ndarray,
    energy_metric: np.ndarray,
    neutral_energy: float,
) -> tuple[np.ndarray, float, tuple[ProgressSimplexCertificate, ...]]:
    """Minimax P/H risk then capacity tie-break directly on the simplex."""

    risks = _risk_functions(problem, inventory)
    active_energy = energy_metric[np.ix_(active, active)]
    active_capacity = problem.capacity_metric[np.ix_(active, active)]
    transform = q / slopes[active]
    energy_limit = _energy_limit(neutral_energy)

    def expand(pi_active: np.ndarray) -> np.ndarray:
        return _full_velocity(pi_active, active, slopes, q)

    def energy_pi(pi_active: np.ndarray) -> float:
        return float((transform * pi_active) @ active_energy @ (transform * pi_active))

    def energy_grad(pi_active: np.ndarray) -> np.ndarray:
        velocity = transform * pi_active
        return 2.0 * transform * (active_energy @ velocity)

    def risk_grad_pi(gradient: np.ndarray, pi_active: np.ndarray) -> np.ndarray:
        del pi_active
        return transform * gradient[active]

    initial_full = expand(neutral_pi_active)
    initial_xi = max(float(fn(initial_full)) for _, fn, _ in risks)
    initial = np.concatenate((neutral_pi_active, [initial_xi]))
    constraints: list[dict[str, Any]] = [
        {
            "type": "eq",
            "fun": lambda value: float(value[:-1].sum() - 1.0),
            "jac": lambda value: np.concatenate((np.ones(active.size), [0.0])),
        },
        {
            "type": "ineq",
            "fun": lambda value: float(energy_limit - energy_pi(value[:-1])),
            "jac": lambda value: np.concatenate((-energy_grad(value[:-1]), [0.0])),
        },
    ]
    for _, risk, gradient in risks:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda value, fn=risk: float(value[-1] - fn(expand(value[:-1]))),
                "jac": lambda value, grad=gradient: np.concatenate(
                    (-risk_grad_pi(grad(expand(value[:-1])), value[:-1]), [1.0])
                ),
            }
        )
    stage1 = minimize(
        lambda value: float(value[-1]),
        initial,
        jac=lambda value: np.concatenate((np.zeros(active.size), [1.0])),
        method="SLSQP",
        bounds=tuple((0.0, None) for _ in active) + ((None, None),),
        constraints=tuple(constraints),
        options={"disp": False, "ftol": FIXED_E8_SOLVER_FTOL, "maxiter": FIXED_E8_SOLVER_MAXITER},
    )
    value1 = np.asarray(stage1.x, dtype=np.float64)
    pi1 = value1[:-1]
    cert1 = _certificate(
        phase="simplex-soft-minimum-worst-p-h-risk",
        result=stage1,
        pi_active=pi1,
        active=active,
        slopes=slopes,
        q=q,
        energy_metric=energy_metric,
        neutral_energy=neutral_energy,
        objective_gradient=np.zeros_like(pi1),
    )
    if not cert1.passed:
        raise ODEBFContractError(
            f"progress simplex Soft stage1 certificate failed: {cert1.raw_free_payload()}"
        )
    xi_star = float(value1[-1])
    xi_cap = xi_star + SIMPLEX_XI_TIE_TOLERANCE
    constraints2: list[dict[str, Any]] = [
        {
            "type": "eq",
            "fun": lambda value: float(value.sum() - 1.0),
            "jac": lambda value: np.ones(active.size),
        },
        {
            "type": "ineq",
            "fun": lambda value: float(energy_limit - energy_pi(value)),
            "jac": lambda value: -energy_grad(value),
        },
    ]
    for _, risk, gradient in risks:
        constraints2.append(
            {
                "type": "ineq",
                "fun": lambda value, fn=risk: float(xi_cap - fn(expand(value))),
                "jac": lambda value, grad=gradient: -risk_grad_pi(
                    grad(expand(value)), value
                ),
            }
        )

    def capacity(value: np.ndarray) -> float:
        velocity = transform * value
        return float(0.5 * velocity @ active_capacity @ velocity)

    def capacity_grad(value: np.ndarray) -> np.ndarray:
        velocity = transform * value
        return transform * (active_capacity @ velocity)

    stage2 = minimize(
        capacity,
        pi1,
        jac=capacity_grad,
        method="SLSQP",
        bounds=tuple((0.0, None) for _ in active),
        constraints=tuple(constraints2),
        options={"disp": False, "ftol": FIXED_E8_SOLVER_FTOL, "maxiter": FIXED_E8_SOLVER_MAXITER},
    )
    pi2 = np.asarray(stage2.x, dtype=np.float64)
    cert2 = _certificate(
        phase="simplex-soft-minimum-capacity-within-p-h-tie",
        result=stage2,
        pi_active=pi2,
        active=active,
        slopes=slopes,
        q=q,
        energy_metric=energy_metric,
        neutral_energy=neutral_energy,
        objective_gradient=capacity_grad(pi2),
    )
    if not cert2.passed:
        raise ODEBFContractError(
            f"progress simplex Soft stage2 certificate failed: {cert2.raw_free_payload()}"
        )
    return pi2, xi_star, (cert1, cert2)


def solve_progress_simplex_routing(
    problem: RoutingProblem,
    inventory: FixedE8SoftInventory | StructuralOnlyRoutingInventory,
    *,
    arm: FixedE8Arm | str,
    alpha_req: float,
) -> ProgressSimplexRoutingResult:
    """Route a P1R23 applied-step physical field on the fixed-progress simplex.

    ``alpha_req`` is retained exclusively as target-demand/coverage telemetry;
    the routing constraint is instead the all-positive-direction P1R23
    reachable progress ``q``.
    """

    selected = FixedE8Arm(arm)
    if selected not in (FixedE8Arm.NEUTRAL, FixedE8Arm.SOFT):
        raise ODEBFContractError("progress simplex routing arm differs")
    slopes, energy_metric = _validate_problem(problem, float(alpha_req))
    active = np.flatnonzero(slopes > 0.0)
    q = float(slopes[active].sum())
    full_zero = np.zeros_like(slopes)
    neutral_pi = full_zero.copy()
    soft_pi = full_zero.copy()
    selected_pi = full_zero.copy()
    neutral_velocity = full_zero.copy()
    soft_velocity = full_zero.copy()
    selected_velocity = full_zero.copy()
    certificates: tuple[ProgressSimplexCertificate, ...] = ()
    xi_star: float | None = None
    neutral_energy = 0.0
    soft_energy = 0.0
    status: ProgressSimplexStatus
    source: str
    soft_influence = 0
    neutral_mapping_max_abs = 0.0

    if active.size == 0:
        status = ProgressSimplexStatus.NO_POSITIVE_DIRECTION
        source = "NO_POSITIVE_DIRECTION_NO_RESCUE"
    else:
        neutral_pi_active = slopes[active] / q
        neutral_pi[active] = neutral_pi_active
        neutral_velocity[active] = 1.0
        formula_velocity = _full_velocity(neutral_pi_active, active, slopes, q)
        neutral_mapping_max_abs = float(
            np.max(np.abs(formula_velocity - neutral_velocity), initial=0.0)
        )
        if neutral_mapping_max_abs > SIMPLEX_PRIMAL_TOLERANCE:
            raise ODEBFContractError("progress simplex Neutral does not reproduce v=1")
        neutral_energy = _energy(neutral_velocity, energy_metric)
        neutral_certificate = _direct_neutral_certificate(
            pi_active=neutral_pi_active,
            active=active,
            slopes=slopes,
            q=q,
            energy_metric=energy_metric,
            neutral_energy=neutral_energy,
        )
        if not neutral_certificate.passed:
            raise ODEBFContractError(
                f"progress simplex Neutral certificate failed: {neutral_certificate.raw_free_payload()}"
            )
        if active.size == 1:
            soft_pi[:] = neutral_pi
            soft_velocity[:] = neutral_velocity
            selected_pi = neutral_pi.copy()
            selected_velocity = neutral_velocity.copy()
            certificates = (neutral_certificate,)
            status = ProgressSimplexStatus.NO_ROUTING_DOF
            source = "SIMPLEX_UNIQUE_ONE_ACTIVE_DIRECTION"
        else:
            soft_active, xi_star, soft_certificates = _solve_soft(
                problem,
                inventory,
                active=active,
                slopes=slopes,
                q=q,
                neutral_pi_active=neutral_pi_active,
                energy_metric=energy_metric,
                neutral_energy=neutral_energy,
            )
            soft_pi[active] = soft_active
            soft_velocity = _full_velocity(soft_active, active, slopes, q)
            soft_energy = _energy(soft_velocity, energy_metric)
            certificates = (neutral_certificate, *soft_certificates)
            if selected is FixedE8Arm.NEUTRAL:
                selected_pi = neutral_pi.copy()
                selected_velocity = neutral_velocity.copy()
                source = "SIMPLEX_NEUTRAL_IDENTITY_V_EQUALS_ONE"
            else:
                selected_pi = soft_pi.copy()
                selected_velocity = soft_velocity.copy()
                source = "SIMPLEX_SOFT_MINIMAX_P_H_THEN_CAPACITY_TIE"
                soft_influence = 1
            if float(np.linalg.norm(soft_velocity - neutral_velocity, ord=2)) <= SIMPLEX_PRIMAL_TOLERANCE:
                status = ProgressSimplexStatus.NO_SAFE_REDISTRIBUTION
            else:
                status = ProgressSimplexStatus.JOINT_WRITE

    if active.size == 0:
        soft_energy = 0.0
    elif active.size == 1:
        soft_energy = neutral_energy
    selected_energy = _energy(selected_velocity, energy_metric)
    selected_progress = float(slopes @ selected_velocity)
    equality = abs(selected_progress - q)
    simplex_sum = abs(float(selected_pi[active].sum()) - 1.0) if active.size else 0.0
    simplex_minimum = float(selected_pi[active].min()) if active.size else 0.0
    if active.size and (
        equality > SIMPLEX_PRIMAL_TOLERANCE
        or simplex_sum > SIMPLEX_PRIMAL_TOLERANCE
        or simplex_minimum < -SIMPLEX_PRIMAL_TOLERANCE
        or selected_energy > _energy_limit(neutral_energy) + SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE
    ):
        raise ODEBFContractError("progress simplex selected certificate differs")
    coverage = 1.0 if alpha_req == 0.0 else q / (float(alpha_req) + FIXED_E8_NORMALIZATION_EPSILON)
    contributions = slopes * selected_velocity
    neutral_scores = _score_inventory(problem, inventory, neutral_velocity, influence_count=0)
    soft_scores = _score_inventory(problem, inventory, soft_velocity, influence_count=1 if active.size > 1 else 0)
    scores = neutral_scores if selected is FixedE8Arm.NEUTRAL else soft_scores
    if tuple(score.label for score in neutral_scores) != tuple(
        score.label for score in soft_scores
    ):
        raise ODEBFContractError("progress simplex P/H score inventory differs")
    neutral_p_h_risk = max((float(score.score) for score in neutral_scores), default=0.0)
    soft_p_h_risk = max((float(score.score) for score in soft_scores), default=0.0)
    selected_p_h_risk = max((float(score.score) for score in scores), default=0.0)
    risk_difference_by_metric = tuple(
        (neutral_score.label, float(soft_score.score - neutral_score.score))
        for neutral_score, soft_score in zip(neutral_scores, soft_scores, strict=True)
    )
    risk_difference = float(soft_p_h_risk - neutral_p_h_risk)
    capacity_metric = np.asarray(problem.capacity_metric, dtype=np.float64)
    neutral_capacity = _capacity(neutral_velocity, capacity_metric)
    soft_capacity = _capacity(soft_velocity, capacity_metric)
    selected_capacity = _capacity(selected_velocity, capacity_metric)
    if active.size:
        selected_pi_active = selected_pi[active]
        positive_pi = selected_pi_active[selected_pi_active > 0.0]
        selected_pi_entropy = float(-np.sum(positive_pi * np.log(positive_pi)))
        selected_pi_effective_layer_count = float(math.exp(selected_pi_entropy))
        top_active_offset = int(np.argmax(selected_pi_active))
        selected_pi_top1_index = int(active[top_active_offset])
        selected_pi_top1_layer_id = int(FIXED_E8_LAYER_ORDER[selected_pi_top1_index])
        selected_pi_top1_share = float(selected_pi_active[top_active_offset])
    else:
        selected_pi_entropy = 0.0
        selected_pi_effective_layer_count = 0.0
        selected_pi_top1_index = None
        selected_pi_top1_layer_id = None
        selected_pi_top1_share = 0.0
    distance = soft_velocity - neutral_velocity
    neutral_norm = float(np.linalg.norm(neutral_velocity))
    soft_norm = float(np.linalg.norm(soft_velocity))
    cosine = (
        1.0
        if neutral_norm == 0.0 and soft_norm == 0.0
        else 0.0
        if neutral_norm == 0.0 or soft_norm == 0.0
        else float(neutral_velocity @ soft_velocity / (neutral_norm * soft_norm))
    )
    structural_only = isinstance(inventory, StructuralOnlyRoutingInventory)
    if structural_only:
        from .compute_progress_simplex_runtime import (
            COMPUTE_PROGRESS_SIMPLEX_INSTRUCTION_ID,
            COMPUTE_PROGRESS_SIMPLEX_METHOD_ID,
        )
        instruction_id = COMPUTE_PROGRESS_SIMPLEX_INSTRUCTION_ID
        method_id = COMPUTE_PROGRESS_SIMPLEX_METHOD_ID
    else:
        instruction_id = PROGRESS_SIMPLEX_INSTRUCTION_ID
        method_id = PROGRESS_SIMPLEX_METHOD_ID
    payload = {
        "instruction_id": instruction_id,
        "method_id": method_id,
        "arm": selected.value,
        "problem_sha256": problem.identity(),
        "routing_inventory_sha256": inventory.raw_free_payload()["identity_sha256"],
        "functional_routing_influence_count": (
            0 if isinstance(inventory, StructuralOnlyRoutingInventory) else 1
        ),
        "signed_slopes": slopes.tolist(),
        "active_direction_mask": [bool(item > 0.0) for item in slopes],
        "q": q,
        "alpha_req": float(alpha_req),
        "neutral_pi": neutral_pi.tolist(),
        "soft_pi": soft_pi.tolist(),
        "selected_pi": selected_pi.tolist(),
        "neutral_velocity": neutral_velocity.tolist(),
        "soft_velocity": soft_velocity.tolist(),
        "selected_velocity": selected_velocity.tolist(),
        "selected_pi_entropy": selected_pi_entropy,
        "selected_pi_effective_layer_count": selected_pi_effective_layer_count,
        "selected_pi_top1_share": selected_pi_top1_share,
        "selected_pi_top1_index": selected_pi_top1_index,
        "selected_pi_top1_layer_id": selected_pi_top1_layer_id,
        "neutral_mapping_max_abs": neutral_mapping_max_abs,
        "neutral_capacity": neutral_capacity,
        "soft_capacity": soft_capacity,
        "selected_capacity": selected_capacity,
        "neutral_p_h_risk": neutral_p_h_risk,
        "soft_p_h_risk": soft_p_h_risk,
        "selected_p_h_risk": selected_p_h_risk,
        "risk_difference_soft_minus_neutral": risk_difference,
        "risk_difference_soft_minus_neutral_by_metric": risk_difference_by_metric,
        "status": status.value,
        "certificates": [item.raw_free_payload() for item in certificates],
    }
    identity = canonical_hash(payload)
    return ProgressSimplexRoutingResult(
        arm=selected,
        mode=FixedE8StepMode.JOINT_WRITE,
        status=status,
        signed_slopes=tuple(float(item) for item in slopes),
        active_direction_mask=tuple(bool(item > 0.0) for item in slopes),
        q=q,
        alpha_req=float(alpha_req),
        alpha_max=q,
        alpha_apply=q,
        coverage=coverage,
        neutral_pi=tuple(float(item) for item in neutral_pi),
        soft_pi=tuple(float(item) for item in soft_pi),
        pi=tuple(float(item) for item in selected_pi),
        selected_pi_entropy=selected_pi_entropy,
        selected_pi_effective_layer_count=selected_pi_effective_layer_count,
        selected_pi_top1_share=selected_pi_top1_share,
        selected_pi_top1_index=selected_pi_top1_index,
        selected_pi_top1_layer_id=selected_pi_top1_layer_id,
        neutral_velocity=tuple(float(item) for item in neutral_velocity),
        soft_velocity=tuple(float(item) for item in soft_velocity),
        velocity=tuple(float(item) for item in selected_velocity),
        neutral_mapping_max_abs=neutral_mapping_max_abs,
        applied_coefficient=tuple(float(FIXED_E8_H) * float(item) for item in selected_velocity),
        predicted_contribution=tuple(float(item) for item in contributions),
        predicted_progress=selected_progress,
        equality_residual=equality,
        simplex_sum_residual=simplex_sum,
        simplex_minimum=simplex_minimum,
        active_layer_count=int(active.size),
        equality_rank=1 if active.size else 0,
        feasible_allocation_dimension=max(int(active.size) - 1, 0),
        maximum_velocity=float(selected_velocity.max(initial=0.0)),
        velocity_above_one_count=int(np.count_nonzero(selected_velocity > 1.0)),
        neutral_energy=neutral_energy,
        soft_energy=soft_energy,
        selected_energy=selected_energy,
        neutral_energy_ratio=_energy_ratio(neutral_energy, neutral_energy),
        soft_energy_ratio=_energy_ratio(soft_energy, neutral_energy),
        selected_energy_ratio=_energy_ratio(selected_energy, neutral_energy),
        global_energy_soft_neutral_ratio=_energy_ratio(soft_energy, neutral_energy),
        neutral_capacity=neutral_capacity,
        soft_capacity=soft_capacity,
        selected_capacity=selected_capacity,
        xi_star=xi_star,
        scores=scores,
        neutral_scores=neutral_scores,
        soft_scores=soft_scores,
        neutral_p_h_risk=neutral_p_h_risk,
        soft_p_h_risk=soft_p_h_risk,
        selected_p_h_risk=selected_p_h_risk,
        risk_difference_soft_minus_neutral=risk_difference,
        risk_difference_soft_minus_neutral_by_metric=risk_difference_by_metric,
        certificates=certificates,
        neutral_soft_coefficient_l1=float(np.linalg.norm(distance, ord=1)),
        neutral_soft_coefficient_l2=float(np.linalg.norm(distance, ord=2)),
        neutral_soft_coefficient_cosine=cosine,
        selected_solution_source=source,
        soft_objective_decision_influence_count=soft_influence,
        hard_h_p_budget_influence_count=0,
        functional_veto_count=0,
        retry_count=0,
        backtracking_count=0,
        structural_only=structural_only,
        identity_sha256=identity,
    )


def progress_simplex_waypoint_factors(
    field: Any,
    velocity: Sequence[float],
    *,
    step_index: int,
) -> dict[str, WaypointFactor]:
    """Create the one-h-applied physical factors without an upper cap on ``v``.

    ``WaypointFactor`` already enforces finite, non-negative theta and factor
    shape/rank integrity.  This helper deliberately does *not* reuse
    ``fixed_e8_waypoint_factors`` because that historical helper forbids
    ``v > 1``.
    """

    values = tuple(float(item) for item in velocity)
    layers = tuple(int(item.layer) for item in field.layers)
    if (
        layers != FIXED_E8_LAYER_ORDER
        or len(values) != len(FIXED_E8_LAYER_ORDER)
        or step_index < 0
        or step_index >= 8
        or any(not math.isfinite(item) or item < 0.0 for item in values)
    ):
        raise ODEBFContractError("progress simplex waypoint geometry differs")
    return {
        layer.weight_name: WaypointFactor(
            layer.weight_name,
            layer.layer,
            0,
            step_index,
            ordinal,
            float(FIXED_E8_H) * values[ordinal],
            layer.residual.clone(),
            layer.q.clone(),
            global_batch_size=layer.factor.global_batch_size,
        )
        for ordinal, layer in enumerate(field.layers)
    }


__all__ = [
    "PROGRESS_SIMPLEX_INSTRUCTION_ID",
    "PROGRESS_SIMPLEX_METHOD_ID",
    "ProgressSimplexCertificate",
    "ProgressSimplexRoutingResult",
    "ProgressSimplexStatus",
    "SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE",
    "SIMPLEX_ENERGY_RELATIVE_TOLERANCE",
    "SIMPLEX_PRIMAL_TOLERANCE",
    "SIMPLEX_XI_TIE_TOLERANCE",
    "StructuralOnlyRoutingInventory",
    "progress_simplex_waypoint_factors",
    "solve_progress_simplex_routing",
]
