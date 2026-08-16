"""One-sided BF16 functional H/P secants and parameter-free soft routing.

The secants observe the locked functional replay, not gradients.  The
two-stage solver preserves the existing structural feasible set and adds no
tunable coefficient: it first minimizes the single worst normalized H/P
violation, then selects the closest capacity-metric velocity.  Empty history
omits the two inactive H constraints so the numerical program is exactly the
P-only program while retaining explicit zero-valued H receipts.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import minimize, nnls

from .contracts import ODEBFContractError, canonical_hash
from .routing import RoutingProblem


H_REF = 1.0 / 8.0
FUNCTIONAL_P_BUDGET = 1.0e-3
SIGMA_TIE_TOLERANCE = 1.0e-8
PRIMAL_TOLERANCE = 1.0e-8
KKT_TOLERANCE = 1.0e-5


class FunctionalPFieldPolicy(str, Enum):
    """Whether the cached secant changes the structural-BF velocity."""

    NONE = "NONE"
    PROBE_ONLY = "PROBE_ONLY"
    SOFT_HARD = "PSOFT_HARD"


@dataclass(frozen=True, slots=True)
class FunctionalPSecant:
    baseline_damage: float
    probe_damage: tuple[float, ...]
    signed_secant: tuple[float, ...]
    h_ref: float
    controller_p_budget: float
    sample_order_sha256: str
    baseline_identity_sha256: str
    factor_state_sha256: str
    cache_identity_sha256: str
    history_item_count: int = 0
    history_sample_order_sha256: str = ""
    history_baseline_identity_sha256: str = ""
    historical_mean_baseline: float = 0.0
    historical_mean_probe: tuple[float, ...] = ()
    historical_mean_signed_secant: tuple[float, ...] = ()
    historical_smooth_baseline: float = 0.0
    historical_smooth_probe: tuple[float, ...] = ()
    historical_smooth_signed_secant: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        baseline = float(self.baseline_damage)
        probes = tuple(float(value) for value in self.probe_damage)
        secants = tuple(float(value) for value in self.signed_secant)
        if (
            len(probes) < 2
            or len(probes) != len(secants)
            or not math.isfinite(baseline)
            or not all(math.isfinite(value) for value in (*probes, *secants))
            or self.h_ref != H_REF
            or self.controller_p_budget != FUNCTIONAL_P_BUDGET
            or any(
                len(value) != 64
                for value in (
                    self.sample_order_sha256,
                    self.baseline_identity_sha256,
                    self.factor_state_sha256,
                    self.cache_identity_sha256,
                )
            )
        ):
            raise ODEBFContractError("functional-P secant contract differs")
        expected = tuple((value - baseline) / H_REF for value in probes)
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-15)
            for left, right in zip(secants, expected, strict=True)
        ):
            raise ODEBFContractError("functional-P signed secant differs")
        history_vectors = (
            tuple(float(value) for value in self.historical_mean_probe),
            tuple(
                float(value) for value in self.historical_mean_signed_secant
            ),
            tuple(float(value) for value in self.historical_smooth_probe),
            tuple(
                float(value)
                for value in self.historical_smooth_signed_secant
            ),
        )
        history_scalars = (
            float(self.historical_mean_baseline),
            float(self.historical_smooth_baseline),
        )
        if (
            isinstance(self.history_item_count, bool)
            or self.history_item_count < 0
            or not all(math.isfinite(value) for value in history_scalars)
            or not all(
                math.isfinite(value)
                for values in history_vectors
                for value in values
            )
        ):
            raise ODEBFContractError("functional-H secant contract differs")
        if self.history_item_count == 0:
            if (
                self.history_sample_order_sha256 != canonical_hash([])
                or len(self.history_baseline_identity_sha256) != 64
                or any(value != 0.0 for value in history_scalars)
                or any(
                    len(values) != len(probes)
                    or any(value != 0.0 for value in values)
                    for values in history_vectors
                )
            ):
                raise ODEBFContractError("empty functional-H secant differs")
        else:
            if (
                len(self.history_sample_order_sha256) != 64
                or len(self.history_baseline_identity_sha256) != 64
                or any(len(values) != len(probes) for values in history_vectors)
            ):
                raise ODEBFContractError("functional-H secant geometry differs")
            expected_mean = tuple(
                (value - history_scalars[0]) / H_REF
                for value in history_vectors[0]
            )
            expected_smooth = tuple(
                (value - history_scalars[1]) / H_REF
                for value in history_vectors[2]
            )
            if any(
                not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-15)
                for expected_values, observed_values in (
                    (expected_mean, history_vectors[1]),
                    (expected_smooth, history_vectors[3]),
                )
                for left, right in zip(
                    expected_values, observed_values, strict=True
                )
            ):
                raise ODEBFContractError("functional-H signed secant differs")

    @property
    def historical_soft_active(self) -> bool:
        return self.history_item_count > 0

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


def build_functional_p_secant(
    *,
    baseline_damage: float,
    probe_damage: Sequence[float],
    sample_order_sha256: str,
    baseline_identity_sha256: str,
    factor_state_sha256: str,
) -> FunctionalPSecant:
    baseline = float(baseline_damage)
    probes = tuple(float(value) for value in probe_damage)
    if not math.isfinite(baseline) or not all(math.isfinite(value) for value in probes):
        raise ODEBFContractError("functional-P probe damage is non-finite")
    secants = tuple((value - baseline) / H_REF for value in probes)
    payload = {
        "baseline_damage": baseline,
        "probe_damage": list(probes),
        "signed_secant": list(secants),
        "h_ref": H_REF,
        "controller_p_budget": FUNCTIONAL_P_BUDGET,
        "sample_order_sha256": sample_order_sha256,
        "baseline_identity_sha256": baseline_identity_sha256,
        "factor_state_sha256": factor_state_sha256,
    }
    return FunctionalPSecant(
        baseline,
        probes,
        secants,
        H_REF,
        FUNCTIONAL_P_BUDGET,
        sample_order_sha256,
        baseline_identity_sha256,
        factor_state_sha256,
        canonical_hash(payload),
        0,
        canonical_hash([]),
        canonical_hash({"empty_history": True}),
        0.0,
        tuple(0.0 for _ in probes),
        tuple(0.0 for _ in probes),
        0.0,
        tuple(0.0 for _ in probes),
        tuple(0.0 for _ in probes),
    )


def build_functional_replay_secant(
    *,
    pretrained_baseline_damage: float,
    pretrained_probe_damage: Sequence[float],
    pretrained_sample_order_sha256: str,
    pretrained_baseline_identity_sha256: str,
    history_item_count: int,
    history_sample_order_sha256: str,
    history_baseline_identity_sha256: str,
    historical_mean_baseline: float,
    historical_mean_probe: Sequence[float],
    historical_smooth_baseline: float,
    historical_smooth_probe: Sequence[float],
    factor_state_sha256: str,
) -> FunctionalPSecant:
    """Build the generic joint replay secant from one shared BF16 probe set."""

    probes = tuple(float(value) for value in pretrained_probe_damage)
    mean_probe = tuple(float(value) for value in historical_mean_probe)
    smooth_probe = tuple(float(value) for value in historical_smooth_probe)
    if (
        isinstance(history_item_count, bool)
        or history_item_count < 0
        or len(probes) < 2
        or len(mean_probe) != len(probes)
        or len(smooth_probe) != len(probes)
    ):
        raise ODEBFContractError("functional replay secant geometry differs")
    if history_item_count == 0:
        if (
            float(historical_mean_baseline) != 0.0
            or float(historical_smooth_baseline) != 0.0
            or any(value != 0.0 for value in (*mean_probe, *smooth_probe))
        ):
            raise ODEBFContractError("empty functional-H replay is nonzero")
    p_baseline = float(pretrained_baseline_damage)
    p_secant = tuple((value - p_baseline) / H_REF for value in probes)
    h_mean_baseline = float(historical_mean_baseline)
    h_smooth_baseline = float(historical_smooth_baseline)
    h_mean_secant = tuple(
        (value - h_mean_baseline) / H_REF for value in mean_probe
    )
    h_smooth_secant = tuple(
        (value - h_smooth_baseline) / H_REF for value in smooth_probe
    )
    payload = {
        "pretrained_baseline_damage": p_baseline,
        "pretrained_probe_damage": list(probes),
        "pretrained_signed_secant": list(p_secant),
        "pretrained_sample_order_sha256": pretrained_sample_order_sha256,
        "pretrained_baseline_identity_sha256": (
            pretrained_baseline_identity_sha256
        ),
        "history_item_count": history_item_count,
        "history_sample_order_sha256": history_sample_order_sha256,
        "history_baseline_identity_sha256": history_baseline_identity_sha256,
        "historical_mean_baseline": h_mean_baseline,
        "historical_mean_probe": list(mean_probe),
        "historical_mean_signed_secant": list(h_mean_secant),
        "historical_smooth_baseline": h_smooth_baseline,
        "historical_smooth_probe": list(smooth_probe),
        "historical_smooth_signed_secant": list(h_smooth_secant),
        "h_ref": H_REF,
        "controller_p_budget": FUNCTIONAL_P_BUDGET,
        "factor_state_sha256": factor_state_sha256,
    }
    return FunctionalPSecant(
        p_baseline,
        probes,
        p_secant,
        H_REF,
        FUNCTIONAL_P_BUDGET,
        pretrained_sample_order_sha256,
        pretrained_baseline_identity_sha256,
        factor_state_sha256,
        canonical_hash(payload),
        history_item_count,
        history_sample_order_sha256,
        history_baseline_identity_sha256,
        h_mean_baseline,
        mean_probe,
        h_mean_secant,
        h_smooth_baseline,
        smooth_probe,
        h_smooth_secant,
    )


@dataclass(frozen=True, slots=True)
class PSoftSolverCertificate:
    stage: str
    solver: str
    success: bool
    iterations: int
    maximum_primal_violation: float
    stationarity_residual: float
    complementarity_residual: float
    sigma: float
    signed_progress: float
    structural_h_value: float
    structural_p_value: float
    trust_value: float
    predicted_controller_p: float
    predicted_historical_mean: float | None
    predicted_historical_smoothmax: float | None
    active_constraints: tuple[str, ...]
    first_false_component: str | None
    passed: bool

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PSoftHardResult:
    pre_soft_velocity: tuple[float, ...]
    soft_velocity: tuple[float, ...]
    signed_secant: tuple[float, ...]
    q_dot_u: float
    q_dot_v: float
    sigma_star: float
    stage2_distance: float
    predicted_full_step_controller_p: float
    predicted_full_step_historical_mean: float | None
    predicted_full_step_historical_smoothmax: float | None
    normalized_slack_components: tuple[tuple[str, float], ...]
    stage1_certificate: PSoftSolverCertificate
    stage2_certificate: PSoftSolverCertificate
    identity_sha256: str

    def __post_init__(self) -> None:
        if (
            len(self.pre_soft_velocity) != len(self.soft_velocity)
            or len(self.soft_velocity) != len(self.signed_secant)
            or any(
                not math.isfinite(value)
                for value in (
                    *self.pre_soft_velocity,
                    *self.soft_velocity,
                    *self.signed_secant,
                    self.q_dot_u,
                    self.q_dot_v,
                    self.sigma_star,
                    self.stage2_distance,
                    self.predicted_full_step_controller_p,
                    *(
                        ()
                        if self.predicted_full_step_historical_mean is None
                        else (self.predicted_full_step_historical_mean,)
                    ),
                    *(
                        ()
                        if self.predicted_full_step_historical_smoothmax is None
                        else (self.predicted_full_step_historical_smoothmax,)
                    ),
                    *(value for _, value in self.normalized_slack_components),
                )
            )
            or self.sigma_star < -PRIMAL_TOLERANCE
            or self.stage2_distance < -PRIMAL_TOLERANCE
            or not self.stage1_certificate.passed
            or not self.stage2_certificate.passed
            or len(self.identity_sha256) != 64
        ):
            raise ODEBFContractError("P-soft result contract differs")

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "pre_soft_velocity": list(self.pre_soft_velocity),
            "soft_velocity": list(self.soft_velocity),
            "signed_secant": list(self.signed_secant),
            "velocity_delta": [
                right - left
                for left, right in zip(
                    self.pre_soft_velocity, self.soft_velocity, strict=True
                )
            ],
            "q_dot_u": self.q_dot_u,
            "q_dot_v": self.q_dot_v,
            "sigma_star": self.sigma_star,
            "stage2_distance": self.stage2_distance,
            "predicted_full_step_controller_p": (
                self.predicted_full_step_controller_p
            ),
            "predicted_full_step_historical_mean": (
                self.predicted_full_step_historical_mean
            ),
            "predicted_full_step_historical_smoothmax": (
                self.predicted_full_step_historical_smoothmax
            ),
            "normalized_slack_components": dict(
                self.normalized_slack_components
            ),
            "stage1_certificate": self.stage1_certificate.raw_free_payload(),
            "stage2_certificate": self.stage2_certificate.raw_free_payload(),
            "identity_sha256": self.identity_sha256,
        }


Constraint = tuple[str, Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]


def _structural_constraints(
    problem: RoutingProblem,
    active: np.ndarray,
) -> list[Constraint]:
    progress = problem.signed_progress[active]
    trust = problem.trust_metric[np.ix_(active, active)]
    result: list[Constraint] = [
        (
            "rewrite_progress",
            lambda value, a=progress, target=problem.requested_progress: float(
                a @ value - target
            ),
            lambda value, a=progress: a.copy(),
        ),
        (
            "trust",
            lambda value, matrix=trust, radius=problem.trust_radius: float(
                radius**2 - value @ matrix @ value
            ),
            lambda value, matrix=trust: -2.0 * matrix @ value,
        ),
    ]
    for barrier in (problem.historical, problem.pretrained):
        linear = barrier.linear[active]
        gram = barrier.gram[np.ix_(active, active)]
        result.append(
            (
                f"structural_{barrier.label}",
                lambda value, offset=barrier.offset, lin=linear, matrix=gram,
                budget=barrier.budget: float(
                    budget - (offset + 2.0 * lin @ value + value @ matrix @ value)
                ),
                lambda value, lin=linear, matrix=gram: -(
                    2.0 * lin + 2.0 * matrix @ value
                ),
            )
        )
    return result


def _certificate(
    *,
    stage: str,
    result: Any,
    value: np.ndarray,
    objective_gradient: np.ndarray,
    constraints: Sequence[Constraint],
    lower: np.ndarray,
    upper: np.ndarray,
    expanded_velocity: np.ndarray,
    sigma: float,
    problem: RoutingProblem,
    secant: FunctionalPSecant,
) -> PSoftSolverCertificate:
    named_slacks: list[tuple[str, float]] = []
    active_gradients: list[np.ndarray] = []
    active_slacks: list[float] = []
    for name, function, jacobian in constraints:
        slack = float(function(value))
        named_slacks.append((name, slack))
        if slack <= 10.0 * PRIMAL_TOLERANCE:
            active_gradients.append(np.asarray(jacobian(value), dtype=np.float64))
            active_slacks.append(slack)
    for index in range(value.size):
        for name, slack, sign in (
            (f"lower_{index}", float(value[index] - lower[index]), 1.0),
            (f"upper_{index}", float(upper[index] - value[index]), -1.0),
        ):
            named_slacks.append((name, slack))
            if slack <= 10.0 * PRIMAL_TOLERANCE:
                basis = np.zeros_like(value)
                basis[index] = sign
                active_gradients.append(basis)
                active_slacks.append(slack)
    maximum_violation = max(
        0.0, max((-slack for _, slack in named_slacks), default=0.0)
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
        (name for name, slack in named_slacks if slack < -PRIMAL_TOLERANCE),
        None,
    )
    success = bool(result.success)
    passed = bool(
        success
        and maximum_violation <= PRIMAL_TOLERANCE
        and stationarity <= KKT_TOLERANCE
        and complementarity <= KKT_TOLERANCE
    )
    q = np.asarray(secant.signed_secant, dtype=np.float64)
    h_mean = np.asarray(
        secant.historical_mean_signed_secant, dtype=np.float64
    )
    h_smooth = np.asarray(
        secant.historical_smooth_signed_secant, dtype=np.float64
    )
    return PSoftSolverCertificate(
        stage=stage,
        solver="scipy-slsqp-float64",
        success=success,
        iterations=int(result.nit),
        maximum_primal_violation=maximum_violation,
        stationarity_residual=float(stationarity),
        complementarity_residual=float(complementarity),
        sigma=float(sigma),
        signed_progress=float(problem.signed_progress @ expanded_velocity),
        structural_h_value=problem.historical.value(expanded_velocity),
        structural_p_value=problem.pretrained.value(expanded_velocity),
        trust_value=float(
            expanded_velocity @ problem.trust_metric @ expanded_velocity
        ),
        predicted_controller_p=float(
            secant.baseline_damage + H_REF * q @ expanded_velocity
        ),
        predicted_historical_mean=(
            None
            if not secant.historical_soft_active
            else float(
                secant.historical_mean_baseline
                + H_REF * h_mean @ expanded_velocity
            )
        ),
        predicted_historical_smoothmax=(
            None
            if not secant.historical_soft_active
            else float(
                secant.historical_smooth_baseline
                + H_REF * h_smooth @ expanded_velocity
            )
        ),
        active_constraints=tuple(
            name
            for name, slack in named_slacks
            if abs(slack) <= 10.0 * PRIMAL_TOLERANCE
        ),
        first_false_component=first_false,
        passed=passed,
    )


def solve_p_soft_hard(
    problem: RoutingProblem,
    pre_soft_velocity: Sequence[float],
    secant: FunctionalPSecant,
) -> PSoftHardResult:
    """Solve the locked minimum-slack then closest-velocity problem."""

    u = np.asarray(tuple(pre_soft_velocity), dtype=np.float64)
    q = np.asarray(secant.signed_secant, dtype=np.float64)
    if (
        u.shape != problem.signed_progress.shape
        or q.shape != u.shape
        or not np.isfinite(u).all()
        or np.any(u < -PRIMAL_TOLERANCE)
    ):
        raise ODEBFContractError("P-soft input velocity differs")
    active = np.flatnonzero(problem.positive_direction_mask)
    if active.size == 0 or np.any(np.abs(u[~problem.positive_direction_mask]) > PRIMAL_TOLERANCE):
        raise ODEBFContractError("P-soft nonpositive routing support differs")
    ua = u[active]
    qa = q[active]
    caps = problem.layer_caps[active]
    common = _structural_constraints(problem, active)
    metric_specs: list[tuple[str, float, np.ndarray, float]] = [
        (
            "predicted_controller_p",
            float(secant.baseline_damage),
            qa,
            FUNCTIONAL_P_BUDGET,
        )
    ]
    if secant.historical_soft_active:
        metric_specs.extend(
            (
                (
                    "predicted_historical_mean",
                    float(secant.historical_mean_baseline),
                    np.asarray(
                        secant.historical_mean_signed_secant,
                        dtype=np.float64,
                    )[active],
                    FUNCTIONAL_P_BUDGET,
                ),
                (
                    "predicted_historical_smoothmax",
                    float(secant.historical_smooth_baseline),
                    np.asarray(
                        secant.historical_smooth_signed_secant,
                        dtype=np.float64,
                    )[active],
                    FUNCTIONAL_P_BUDGET,
                ),
            )
        )

    def p_slack_stage1(value: np.ndarray) -> float:
        velocity, sigma = value[:-1], float(value[-1])
        return float(
            FUNCTIONAL_P_BUDGET
            - secant.baseline_damage
            + FUNCTIONAL_P_BUDGET * sigma
            - H_REF * qa @ velocity
        )

    def p_jac_stage1(value: np.ndarray) -> np.ndarray:
        del value
        return np.concatenate(
            (-H_REF * qa, np.asarray([FUNCTIONAL_P_BUDGET]))
        )

    stage1_constraints: list[Constraint] = [
        (
            name,
            lambda value, function=function: function(value[:-1]),
            lambda value, jacobian=jacobian: np.concatenate(
                (jacobian(value[:-1]), np.asarray([0.0]))
            ),
        )
        for name, function, jacobian in common
    ]
    stage1_constraints.append(
        ("predicted_controller_p", p_slack_stage1, p_jac_stage1)
    )
    for label, baseline, signed_metric, budget in metric_specs[1:]:
        stage1_constraints.append(
            (
                label,
                lambda value, base=baseline, signed=signed_metric,
                locked_budget=budget: float(
                    locked_budget
                    - base
                    + locked_budget * float(value[-1])
                    - H_REF * signed @ value[:-1]
                ),
                lambda value, signed=signed_metric,
                locked_budget=budget: np.concatenate(
                    (-H_REF * signed, np.asarray([locked_budget]))
                ),
            )
        )
    sigma0 = max(
        0.0,
        *(
            (
                baseline + H_REF * float(signed_metric @ ua) - budget
            )
            / budget
            for _, baseline, signed_metric, budget in metric_specs
        ),
    )
    x0 = np.concatenate((ua, np.asarray([sigma0])))
    stage1 = minimize(
        lambda value: float(value[-1]),
        x0,
        jac=lambda value: np.concatenate(
            (np.zeros(active.size, dtype=np.float64), np.asarray([1.0]))
        ),
        method="SLSQP",
        bounds=tuple((0.0, float(cap)) for cap in caps) + ((0.0, None),),
        constraints=[
            {"type": "ineq", "fun": function, "jac": jacobian}
            for _, function, jacobian in stage1_constraints
        ],
        options={"ftol": 1.0e-12, "maxiter": 1000, "disp": False},
    )
    x1 = np.asarray(stage1.x, dtype=np.float64)
    sigma_star = max(0.0, float(x1[-1]))
    expanded1 = np.zeros_like(u)
    expanded1[active] = x1[:-1]
    certificate1 = _certificate(
        stage="MINIMUM_NORMALIZED_P_VIOLATION",
        result=stage1,
        value=x1,
        objective_gradient=np.concatenate(
            (np.zeros(active.size, dtype=np.float64), np.asarray([1.0]))
        ),
        constraints=stage1_constraints,
        lower=np.zeros(active.size + 1, dtype=np.float64),
        upper=np.concatenate((caps, np.asarray([math.inf]))),
        expanded_velocity=expanded1,
        sigma=sigma_star,
        problem=problem,
        secant=secant,
    )
    if not certificate1.passed:
        raise ODEBFContractError("P-soft stage-1 solver certificate failed")

    predicted_limit = (
        FUNCTIONAL_P_BUDGET
        + FUNCTIONAL_P_BUDGET * (sigma_star + SIGMA_TIE_TOLERANCE)
    )

    def p_slack_stage2(value: np.ndarray) -> float:
        return float(
            predicted_limit
            - secant.baseline_damage
            - H_REF * qa @ value
        )

    stage2_constraints = [*common]
    stage2_constraints.append(
        (
            "predicted_controller_p",
            p_slack_stage2,
            lambda value: -H_REF * qa.copy(),
        )
    )
    for label, baseline, signed_metric, budget in metric_specs[1:]:
        metric_limit = budget * (
            1.0 + sigma_star + SIGMA_TIE_TOLERANCE
        )
        stage2_constraints.append(
            (
                label,
                lambda value, base=baseline, signed=signed_metric,
                limit=metric_limit: float(
                    limit - base - H_REF * signed @ value
                ),
                lambda value, signed=signed_metric: -H_REF * signed.copy(),
            )
        )
    capacity = problem.capacity_metric[np.ix_(active, active)]
    stage2 = minimize(
        lambda value: float(0.5 * (value - ua) @ capacity @ (value - ua)),
        x1[:-1],
        jac=lambda value: capacity @ (value - ua),
        method="SLSQP",
        bounds=tuple((0.0, float(cap)) for cap in caps),
        constraints=[
            {"type": "ineq", "fun": function, "jac": jacobian}
            for _, function, jacobian in stage2_constraints
        ],
        options={"ftol": 1.0e-12, "maxiter": 1000, "disp": False},
    )
    va = np.asarray(stage2.x, dtype=np.float64)
    v = np.zeros_like(u)
    v[active] = va
    normalized_slack_components = tuple(
        (
            label,
            max(
                0.0,
                (
                    baseline
                    + H_REF
                    * float(
                        np.asarray(
                            (
                                secant.signed_secant
                                if label == "predicted_controller_p"
                                else (
                                    secant.historical_mean_signed_secant
                                    if label == "predicted_historical_mean"
                                    else secant.historical_smooth_signed_secant
                                )
                            ),
                            dtype=np.float64,
                        )
                        @ v
                    )
                    - budget
                )
                / budget,
            ),
        )
        for label, baseline, _, budget in metric_specs
    )
    sigma_value = max(value for _, value in normalized_slack_components)
    certificate2 = _certificate(
        stage="CLOSEST_STRUCTURAL_BF_VELOCITY",
        result=stage2,
        value=va,
        objective_gradient=capacity @ (va - ua),
        constraints=stage2_constraints,
        lower=np.zeros(active.size, dtype=np.float64),
        upper=caps,
        expanded_velocity=v,
        sigma=sigma_value,
        problem=problem,
        secant=secant,
    )
    if not certificate2.passed or sigma_value > sigma_star + SIGMA_TIE_TOLERANCE + PRIMAL_TOLERANCE:
        raise ODEBFContractError("P-soft stage-2 solver certificate failed")
    difference = v - u
    distance = float(
        math.sqrt(max(float(difference @ problem.capacity_metric @ difference), 0.0))
    )
    payload = {
        "problem_sha256": problem.identity(),
        "secant_sha256": secant.cache_identity_sha256,
        "pre_soft_velocity": u.tolist(),
        "soft_velocity": v.tolist(),
        "q_dot_u": float(q @ u),
        "q_dot_v": float(q @ v),
        "sigma_star": sigma_star,
        "stage2_distance": distance,
        "predicted_full_step_controller_p": float(
            secant.baseline_damage + H_REF * q @ v
        ),
        "predicted_full_step_historical_mean": (
            None
            if not secant.historical_soft_active
            else float(
                secant.historical_mean_baseline
                + H_REF
                * np.asarray(
                    secant.historical_mean_signed_secant,
                    dtype=np.float64,
                )
                @ v
            )
        ),
        "predicted_full_step_historical_smoothmax": (
            None
            if not secant.historical_soft_active
            else float(
                secant.historical_smooth_baseline
                + H_REF
                * np.asarray(
                    secant.historical_smooth_signed_secant,
                    dtype=np.float64,
                )
                @ v
            )
        ),
        "normalized_slack_components": dict(normalized_slack_components),
        "stage1_certificate": certificate1.raw_free_payload(),
        "stage2_certificate": certificate2.raw_free_payload(),
    }
    return PSoftHardResult(
        tuple(float(value) for value in u),
        tuple(float(value) for value in v),
        tuple(float(value) for value in q),
        float(q @ u),
        float(q @ v),
        sigma_star,
        distance,
        payload["predicted_full_step_controller_p"],
        payload["predicted_full_step_historical_mean"],
        payload["predicted_full_step_historical_smoothmax"],
        normalized_slack_components,
        certificate1,
        certificate2,
        canonical_hash(payload),
    )
