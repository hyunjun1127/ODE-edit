"""Full-strength physical edit routing with an optional Soft-BF objective.

This module is intentionally independent of the R13 ``kappa`` controller.
It reuses only the inherited five-layer technical coefficient set and the
already-defined dimensionless structural/functional Soft risk inventory.
The physical edit slope supplied here is an applied-step endpoint difference:
``Phi(W)-Phi(W+h B_l)``.  Consequently ``a_edit @ v`` already has applied-step
loss units and ``h`` appears only when the selected velocity becomes the
physical coefficient ``theta=h*v``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import ODEBFContractError, canonical_hash
from .fixed_e8_soft_routing import (
    FIXED_E8_H,
    FIXED_E8_KKT_TOLERANCE,
    FIXED_E8_LAYER_ORDER,
    FIXED_E8_NORMALIZATION_EPSILON,
    FIXED_E8_PRIMAL_TOLERANCE,
    FixedE8Constraint,
    FixedE8SoftInventory,
    FixedE8SolverCertificate,
    RoutingProblem,
    _functional_score_constraint,
    _maximum_progress,
    _score_inventory,
    _solve_slsqp,
    _structural_score_constraint,
    _technical_constraints,
)


FULL_DRIVE_METHOD_ID = "FULL_STRENGTH_PHYSICAL_W_ONLY_SOFT_BF_DYNAMIC_V1"
FULL_DRIVE_EPS_J = 1.0e-8
FULL_DRIVE_EPS_C = 1.0e-12
FULL_DRIVE_LAMBDA_GRID = (
    0.0,
    1.0e-4,
    10.0 ** -3.5,
    1.0e-3,
    10.0 ** -2.5,
    1.0e-2,
    10.0 ** -1.5,
    1.0e-1,
    10.0 ** -0.5,
    1.0,
    10.0 ** 0.5,
    10.0,
    10.0 ** 1.5,
    100.0,
)
FULL_DRIVE_DENSE_LOG_MIN = 1.0e-4
FULL_DRIVE_DENSE_LOG_MAX = 1.0e2
FULL_DRIVE_DENSE_LOG_COUNT = 257
FULL_DRIVE_LOG_REFINEMENT_TOLERANCE = 1.0e-6
FULL_DRIVE_EDIT_RETENTION_FLOOR = 0.90
FULL_DRIVE_MATERIAL_DISTANCE = 1.0e-6
FULL_DRIVE_NONZERO_L1 = 1.0e-6
FULL_DRIVE_MAX_TOP1_EXCLUSIVE = 0.80
FULL_DRIVE_MIN_EFFECTIVE_LAYERS_EXCLUSIVE = 1.5


class FullDriveArm(str, Enum):
    NO_SOFT = "FULL-NOSOFT"
    SOFT = "FULL-SOFT"


class FullDriveStepMode(str, Enum):
    JOINT_WRITE = "JOINT_WRITE"
    TARGET_ONLY_PMAX_ZERO = "TARGET_ONLY_PMAX_ZERO"


@dataclass(frozen=True, slots=True)
class LambdaReplaySource:
    alias: str
    allocation: str
    routing: str
    target_dynamics: str
    step_index: int
    source_checkpoint: str
    result_terminal_sha256: str
    field_receipt_sha256: str
    problem_identity_sha256: str
    inventory_identity_sha256: str
    closure_root_digest: str

    def __post_init__(self) -> None:
        if (
            self.alias not in ("llama3-8b-inst", "qwen2.5-7b-inst")
            or self.allocation != "RS"
            or self.routing != "NEUTRAL"
            or self.target_dynamics != "DYNAMIC_TARGET"
            or self.step_index != 0
            or any(
                len(value) != 64
                for value in (
                    self.result_terminal_sha256,
                    self.field_receipt_sha256,
                    self.problem_identity_sha256,
                    self.inventory_identity_sha256,
                    self.closure_root_digest,
                )
            )
            or len(self.source_checkpoint) != 40
        ):
            raise ODEBFContractError("full-drive lambda source identity differs")


@dataclass(frozen=True, slots=True)
class LambdaParetoPoint:
    lambda_p: float
    r_edit: float
    r_ph: float
    velocity: tuple[float, ...]
    share: tuple[float, ...]
    top1_share: float
    effective_layer_count: float
    distance_to_nominal: float
    left_share_l1: float | None
    right_share_l1: float | None
    left_cosine: float | None
    right_cosine: float | None
    eligible: bool
    rejection_reasons: tuple[str, ...]
    routing_identity_sha256: str


@dataclass(frozen=True, slots=True)
class LambdaParetoLock:
    source: LambdaReplaySource
    selected_lambda: float
    points: tuple[LambdaParetoPoint, ...]
    selection_rule: Mapping[str, Any]
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload = {
            "schema": "ode-edit-s05-p1r17-lambda-pareto-lock/v1",
            "source": asdict(self.source),
            "lambda_grid": list(FULL_DRIVE_LAMBDA_GRID),
            "selected_lambda": self.selected_lambda,
            "points": [asdict(item) for item in self.points],
            "selection_rule": dict(self.selection_rule),
            "outcome_model_access_count": 0,
            "model_forward_count": 0,
            "gpu_action_count": 0,
            "weight_write_count": 0,
        }
        if canonical_hash(payload) != self.identity_sha256:
            raise ODEBFContractError("full-drive lambda lock identity differs")
        return payload


@dataclass(frozen=True, slots=True)
class FullDriveRoutingResult:
    arm: FullDriveArm
    mode: FullDriveStepMode
    lambda_p: float
    edit_slopes: tuple[float, ...]
    active_direction_mask: tuple[bool, ...]
    nominal_velocity: tuple[float, ...]
    p_max: float
    velocity: tuple[float, ...]
    applied_coefficient: tuple[float, ...]
    r_edit: float
    r_ph: float
    objective_j: float
    nominal_r_ph: float
    distance_to_nominal: float
    scores: tuple[Mapping[str, Any], ...]
    certificates: tuple[FixedE8SolverCertificate, ...]
    identity_sha256: str

    @property
    def signed_slopes(self) -> tuple[float, ...]:
        return self.edit_slopes

    @property
    def requested_progress(self) -> float:
        return 0.0

    @property
    def pre_soft_velocity(self) -> tuple[float, ...]:
        return self.nominal_velocity

    @property
    def authoritative_certificates(self) -> tuple[FixedE8SolverCertificate, ...]:
        return self.certificates

    @property
    def diagnostic_shadow_certificates(self) -> tuple[FixedE8SolverCertificate, ...]:
        return ()

    @property
    def soft_shadow_status(self) -> str:
        return "AUTHORITATIVE_FULL_DRIVE"

    @property
    def selected_solution_source(self) -> str:
        return (
            "FULL_STRENGTH_NOMINAL"
            if self.arm is FullDriveArm.NO_SOFT
            else "FULL_SOFT_JOINT"
        )

    @property
    def stage1_selection_fallback_count(self) -> int:
        return 0

    def __post_init__(self) -> None:
        dimension = len(FIXED_E8_LAYER_ORDER)
        vectors = (
            self.edit_slopes,
            self.active_direction_mask,
            self.nominal_velocity,
            self.velocity,
            self.applied_coefficient,
        )
        if any(len(item) != dimension for item in vectors):
            raise ODEBFContractError("full-drive routing dimension differs")
        numeric = (
            self.lambda_p,
            *self.edit_slopes,
            *self.nominal_velocity,
            self.p_max,
            *self.velocity,
            *self.applied_coefficient,
            self.r_edit,
            self.r_ph,
            self.objective_j,
            self.nominal_r_ph,
            self.distance_to_nominal,
        )
        if any(not math.isfinite(float(item)) for item in numeric):
            raise ODEBFContractError("full-drive routing is non-finite")
        expected = np.asarray(self.velocity, dtype=np.float64) * float(FIXED_E8_H)
        if not np.allclose(
            expected,
            np.asarray(self.applied_coefficient, dtype=np.float64),
            rtol=0.0,
            atol=1.0e-14,
        ):
            raise ODEBFContractError("full-drive h application differs")
        if any(not item.passed for item in self.certificates):
            raise ODEBFContractError("full-drive numerical certificate failed")

    def raw_free_payload(self) -> dict[str, Any]:
        payload = {
            "method_id": FULL_DRIVE_METHOD_ID,
            "arm": self.arm.value,
            "mode": self.mode.value,
            "lambda_p": self.lambda_p,
            "layer_order": list(FIXED_E8_LAYER_ORDER),
            "edit_slopes": list(self.edit_slopes),
            "active_direction_mask": list(self.active_direction_mask),
            "nominal_velocity": list(self.nominal_velocity),
            "p_max": self.p_max,
            "p_max_role": "NOMINAL_FULL_ODE_FIELD_NOT_HARD_FLOOR",
            "velocity": list(self.velocity),
            "applied_coefficient": list(self.applied_coefficient),
            "applied_coefficient_definition": "theta=h*v",
            "r_edit": self.r_edit,
            "r_ph": self.r_ph,
            "objective_j": self.objective_j,
            "nominal_r_ph": self.nominal_r_ph,
            "distance_to_nominal": self.distance_to_nominal,
            "scores": [dict(item) for item in self.scores],
            "certificates": [item.raw_free_payload() for item in self.certificates],
            "epsilon_j": FULL_DRIVE_EPS_J,
            "epsilon_c": FULL_DRIVE_EPS_C,
            "progress_floor": 0.0,
            "kappa": 0.0,
            "requested_progress": None,
            "hard_h_p_budget_influence_count": 0,
            "functional_veto_count": 0,
            "capacity_primary_objective_count": 0,
            "retry_count": 0,
            "transport_decision_influence_count": 0,
            "native_or_direct_z_controller_access_count": 0,
            "h_application_count": 1,
            "stable_lexicographic_order": list(FIXED_E8_LAYER_ORDER),
            "stable_lexicographic_status": (
                "INACTIVE_UNIQUE_POSITIVE_DEFINITE_DISTANCE"
            ),
        }
        if canonical_hash(payload) != self.identity_sha256:
            raise ODEBFContractError("full-drive routing identity differs")
        return payload


def _expanded(active: np.ndarray, value: np.ndarray) -> np.ndarray:
    result = np.zeros(len(FIXED_E8_LAYER_ORDER), dtype=np.float64)
    result[active] = np.asarray(value[: active.size], dtype=np.float64)
    return result


def _risk_scores(
    problem: RoutingProblem,
    inventory: FixedE8SoftInventory,
    velocity: np.ndarray,
    *,
    influence_count: int,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        asdict(item)
        for item in _score_inventory(
            problem,
            inventory,
            velocity,
            influence_count=influence_count,
        )
    )


def _maximum_active_risk(scores: Sequence[Mapping[str, Any]]) -> float:
    return max(
        (float(item["score"]) for item in scores if bool(item["active"])),
        default=0.0,
    )


def _risk_constraints(
    problem: RoutingProblem,
    inventory: FixedE8SoftInventory,
    active: np.ndarray,
    *,
    xi_index: int,
) -> list[FixedE8Constraint]:
    result: list[FixedE8Constraint] = []
    if inventory.history_item_count > 0:
        result.append(
            _structural_score_constraint(
                "full_drive_structural_historical",
                problem.historical,
                active,
                xi_index=xi_index,
            )
        )
    result.append(
        _structural_score_constraint(
            "full_drive_structural_pretrained",
            problem.pretrained,
            active,
            xi_index=xi_index,
        )
    )
    result.extend(
        _functional_score_constraint(metric, active, xi_index=xi_index)
        for metric in inventory.active_metrics()
    )
    return result


def _lift_constraint(
    constraint: FixedE8Constraint, active_size: int
) -> FixedE8Constraint:
    return FixedE8Constraint(
        constraint.name,
        lambda value, item=constraint: item.function(value[:active_size]),
        lambda value, item=constraint: np.concatenate(
            (item.jacobian(value[:active_size]), np.zeros(1, dtype=np.float64))
        ),
        lambda value, item=constraint: np.pad(
            item.hessian(value[:active_size]), ((0, 1), (0, 1))
        ),
    )


def _objective_constraint(
    *,
    name: str,
    edit: np.ndarray,
    p_max: float,
    lambda_p: float,
    cap: float,
) -> FixedE8Constraint:
    denominator = p_max + FIXED_E8_NORMALIZATION_EPSILON

    def slack(value: np.ndarray) -> float:
        return float(
            cap
            - (
                1.0
                - edit @ value[: edit.size] / denominator
                + lambda_p * value[edit.size]
            )
        )

    gradient = np.concatenate((edit / denominator, [-lambda_p]))
    return FixedE8Constraint(
        name,
        slack,
        lambda value, result=gradient: result.copy(),
        lambda value, dimension=edit.size + 1: np.zeros(
            (dimension, dimension), dtype=np.float64
        ),
    )


def solve_full_drive_routing(
    problem: RoutingProblem,
    inventory: FixedE8SoftInventory,
    *,
    edit_slopes: Sequence[float],
    arm: FullDriveArm | str,
    lambda_p: float,
) -> FullDriveRoutingResult:
    """Select a full nominal field or its Soft-BF counterpart.

    ``edit_slopes`` must be five BF16-authoritative applied-step endpoint
    differences.  P/H risk affects only the Soft objective and never enters
    the technical feasible set.
    """

    selected = FullDriveArm(arm)
    slopes = np.asarray(tuple(edit_slopes), dtype=np.float64)
    if (
        slopes.shape != (len(FIXED_E8_LAYER_ORDER),)
        or not np.all(np.isfinite(slopes))
        or not math.isfinite(lambda_p)
        or lambda_p < 0.0
    ):
        raise ODEBFContractError("full-drive slope/lambda contract differs")
    if problem.signed_progress.shape != slopes.shape:
        raise ODEBFContractError("full-drive problem shape differs")
    if inventory.field_sha256 == "0" * 64:
        raise ODEBFContractError("full-drive functional inventory is absent")
    scientific_problem = RoutingProblem(
        slopes.copy(),
        problem.capacity_metric.copy(),
        problem.trust_metric.copy(),
        problem.trust_radius,
        problem.layer_caps.copy(),
        problem.requested_progress,
        problem.minimum_progress,
        problem.historical,
        problem.pretrained,
    )
    active = np.flatnonzero(slopes > 0.0)
    mask = tuple(bool(item > 0.0) for item in slopes)
    if active.size == 0:
        nominal = selected_velocity = np.zeros_like(slopes)
        p_max = 0.0
        certificates: list[FixedE8SolverCertificate] = []
        mode = FullDriveStepMode.TARGET_ONLY_PMAX_ZERO
    else:
        nominal, p_max, maximum_certificate = _maximum_progress(
            scientific_problem, active
        )
        certificates = [maximum_certificate]
        if p_max <= FIXED_E8_NORMALIZATION_EPSILON:
            nominal = selected_velocity = np.zeros_like(slopes)
            p_max = 0.0
            mode = FullDriveStepMode.TARGET_ONLY_PMAX_ZERO
        elif selected is FullDriveArm.NO_SOFT:
            selected_velocity = nominal.copy()
            mode = FullDriveStepMode.JOINT_WRITE
        else:
            mode = FullDriveStepMode.JOINT_WRITE
            edit = slopes[active]
            caps = np.minimum(problem.layer_caps[active], 1.0)
            technical = _technical_constraints(
                scientific_problem, active, requested_progress=None
            )
            xi_index = active.size
            constraints = [
                _lift_constraint(item, active.size) for item in technical
            ]
            constraints.extend(
                _risk_constraints(
                    scientific_problem,
                    inventory,
                    active,
                    xi_index=xi_index,
                )
            )
            nominal_active = nominal[active]
            nominal_scores = _risk_scores(
                scientific_problem, inventory, nominal, influence_count=1
            )
            initial_xi = _maximum_active_risk(nominal_scores)
            denominator = p_max + FIXED_E8_NORMALIZATION_EPSILON
            stage1, stage1_certificate = _solve_slsqp(
                phase="full-drive-soft-minimum-j",
                problem=scientific_problem,
                active=active,
                objective=lambda value, a=edit, scale=denominator, weight=lambda_p: float(
                    1.0 - a @ value[: a.size] / scale + weight * value[a.size]
                ),
                jacobian=lambda value, a=edit, scale=denominator, weight=lambda_p: np.concatenate(
                    (-a / scale, [weight])
                ),
                hessian=lambda value, size=active.size + 1: np.zeros(
                    (size, size), dtype=np.float64
                ),
                constraints=constraints,
                lower=np.concatenate((np.zeros(active.size), [0.0])),
                upper=np.concatenate((caps, [np.inf])),
                initial=np.concatenate((nominal_active, [initial_xi])),
                p_max=p_max,
                requested_progress=0.0,
                xi=initial_xi,
                authority_role="AUTHORITATIVE",
            )
            certificates.append(stage1_certificate)
            j_star = float(
                1.0
                - edit @ stage1[:active.size] / denominator
                + lambda_p * stage1[xi_index]
            )
            stage2_constraints = [
                *constraints,
                _objective_constraint(
                    name="full_drive_j_tie",
                    edit=edit,
                    p_max=p_max,
                    lambda_p=lambda_p,
                    cap=j_star + FULL_DRIVE_EPS_J,
                ),
            ]
            capacity = problem.capacity_metric[np.ix_(active, active)]
            distance_scale = float(
                nominal_active @ capacity @ nominal_active + FULL_DRIVE_EPS_C
            )
            stage2, stage2_certificate = _solve_slsqp(
                phase="full-drive-soft-minimum-distance-to-nominal",
                problem=scientific_problem,
                active=active,
                objective=lambda value, matrix=capacity, origin=nominal_active, scale=distance_scale: float(
                    (value[: origin.size] - origin)
                    @ matrix
                    @ (value[: origin.size] - origin)
                    / scale
                ),
                jacobian=lambda value, matrix=capacity, origin=nominal_active, scale=distance_scale: np.concatenate(
                    (2.0 * matrix @ (value[: origin.size] - origin) / scale, [0.0])
                ),
                hessian=lambda value, matrix=capacity, scale=distance_scale: np.pad(
                    2.0 * matrix / scale, ((0, 1), (0, 1))
                ),
                constraints=stage2_constraints,
                lower=np.concatenate((np.zeros(active.size), [0.0])),
                upper=np.concatenate((caps, [np.inf])),
                initial=stage1,
                p_max=p_max,
                requested_progress=0.0,
                xi=float(stage1[xi_index]),
                authority_role="AUTHORITATIVE",
            )
            certificates.append(stage2_certificate)
            # The inherited capacity metric is positive definite, making the
            # distance-to-u objective strictly convex.  Its certified Stage-2
            # minimizer is therefore unique.  The stable layer order remains
            # the final (inactive) lexicographic rule without adding a
            # numerically fragile extra scientific objective.
            selected_velocity = _expanded(active, stage2)
    nominal_scores = _risk_scores(
        scientific_problem, inventory, nominal, influence_count=0
    )
    selected_scores = _risk_scores(
        scientific_problem,
        inventory,
        selected_velocity,
        influence_count=1 if selected is FullDriveArm.SOFT else 0,
    )
    nominal_risk = _maximum_active_risk(nominal_scores)
    selected_risk = _maximum_active_risk(selected_scores)
    r_edit = (
        0.0
        if p_max <= FIXED_E8_NORMALIZATION_EPSILON
        else float(slopes @ selected_velocity / (p_max + FIXED_E8_NORMALIZATION_EPSILON))
    )
    objective_j = 1.0 - r_edit + lambda_p * selected_risk
    delta_full = selected_velocity - nominal
    distance_scale_full = float(
        nominal @ problem.capacity_metric @ nominal + FULL_DRIVE_EPS_C
    )
    distance = float(
        delta_full @ problem.capacity_metric @ delta_full / distance_scale_full
    )
    applied = tuple(float(FIXED_E8_H) * float(item) for item in selected_velocity)
    payload = {
        "method_id": FULL_DRIVE_METHOD_ID,
        "arm": selected.value,
        "mode": mode.value,
        "lambda_p": float(lambda_p),
        "layer_order": list(FIXED_E8_LAYER_ORDER),
        "edit_slopes": slopes.tolist(),
        "active_direction_mask": list(mask),
        "nominal_velocity": nominal.tolist(),
        "p_max": float(p_max),
        "p_max_role": "NOMINAL_FULL_ODE_FIELD_NOT_HARD_FLOOR",
        "velocity": selected_velocity.tolist(),
        "applied_coefficient": list(applied),
        "applied_coefficient_definition": "theta=h*v",
        "r_edit": r_edit,
        "r_ph": selected_risk,
        "objective_j": objective_j,
        "nominal_r_ph": nominal_risk,
        "distance_to_nominal": distance,
        "scores": [dict(item) for item in selected_scores],
        "certificates": [item.raw_free_payload() for item in certificates],
        "epsilon_j": FULL_DRIVE_EPS_J,
        "epsilon_c": FULL_DRIVE_EPS_C,
        "progress_floor": 0.0,
        "kappa": 0.0,
        "requested_progress": None,
        "hard_h_p_budget_influence_count": 0,
        "functional_veto_count": 0,
        "capacity_primary_objective_count": 0,
        "retry_count": 0,
        "transport_decision_influence_count": 0,
        "native_or_direct_z_controller_access_count": 0,
        "h_application_count": 1,
        "stable_lexicographic_order": list(FIXED_E8_LAYER_ORDER),
        "stable_lexicographic_status": (
            "INACTIVE_UNIQUE_POSITIVE_DEFINITE_DISTANCE"
        ),
    }
    return FullDriveRoutingResult(
        selected,
        mode,
        float(lambda_p),
        tuple(float(item) for item in slopes),
        mask,
        tuple(float(item) for item in nominal),
        float(p_max),
        tuple(float(item) for item in selected_velocity),
        applied,
        r_edit,
        selected_risk,
        objective_j,
        nominal_risk,
        distance,
        selected_scores,
        tuple(certificates),
        canonical_hash(payload),
    )


def _distribution(value: Sequence[float]) -> tuple[tuple[float, ...], float, float]:
    vector = np.abs(np.asarray(tuple(value), dtype=np.float64))
    total = float(vector.sum())
    share = (
        np.zeros_like(vector) if total <= FULL_DRIVE_EPS_C else vector / total
    )
    hhi = float(share @ share)
    return (
        tuple(float(item) for item in share),
        float(share.max(initial=0.0)),
        0.0 if hhi <= FULL_DRIVE_EPS_C else 1.0 / hhi,
    )


def _continuity(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float]:
    left_value = np.asarray(tuple(left), dtype=np.float64)
    right_value = np.asarray(tuple(right), dtype=np.float64)
    left_share, _, _ = _distribution(left_value)
    right_share, _, _ = _distribution(right_value)
    share_l1 = float(
        np.abs(np.asarray(left_share) - np.asarray(right_share)).sum()
    )
    denominator = float(np.linalg.norm(left_value) * np.linalg.norm(right_value))
    cosine = (
        1.0
        if denominator <= FULL_DRIVE_EPS_C
        and np.linalg.norm(left_value - right_value) <= FULL_DRIVE_EPS_C
        else 0.0
        if denominator <= FULL_DRIVE_EPS_C
        else float(left_value @ right_value / denominator)
    )
    return share_l1, cosine


def replay_and_select_lambda(
    problem: RoutingProblem,
    inventory: FixedE8SoftInventory,
    *,
    edit_slopes: Sequence[float],
    source: LambdaReplaySource,
) -> LambdaParetoLock:
    """Run the predeclared solver-only grid and freeze one common lambda."""

    results = tuple(
        solve_full_drive_routing(
            problem,
            inventory,
            edit_slopes=edit_slopes,
            arm=FullDriveArm.SOFT,
            lambda_p=value,
        )
        for value in FULL_DRIVE_LAMBDA_GRID
    )
    if any(item.mode is not FullDriveStepMode.JOINT_WRITE for item in results):
        raise ODEBFContractError("SOLVER_GEOMETRY_BLOCKED: replay pmax is zero")
    geometry_distinct = any(
        np.linalg.norm(
            np.asarray(results[index].velocity)
            - np.asarray(results[index - 1].velocity)
        )
        >= 1.0e-6
        for index in range(1, len(results))
    )
    if not geometry_distinct:
        raise ODEBFContractError(
            "SOLVER_GEOMETRY_BLOCKED: all-lambda geometry is identical"
        )
    nominal_risk = results[0].nominal_r_ph
    raw_points: list[LambdaParetoPoint] = []
    for index, result in enumerate(results):
        share, top1, effective = _distribution(result.velocity)
        left = (
            (None, None)
            if index == 0
            else _continuity(results[index - 1].velocity, result.velocity)
        )
        right = (
            (None, None)
            if index == len(results) - 1
            else _continuity(result.velocity, results[index + 1].velocity)
        )
        reasons: list[str] = []
        if index == 0 or index == len(results) - 1:
            reasons.append("NOT_STABLE_INTERIOR_GRID_POINT")
        if result.r_edit < 0.90:
            reasons.append("EDIT_RETENTION_BELOW_POINT_NINE")
        if top1 >= 0.80:
            reasons.append("SEVERE_SINGLE_LAYER_TOP1")
        if effective <= 1.5:
            reasons.append("SEVERE_SINGLE_LAYER_EFFECTIVE_COUNT")
        improvement = nominal_risk - result.r_ph
        if (
            improvement < 0.05 * nominal_risk
            if nominal_risk > FIXED_E8_NORMALIZATION_EPSILON
            else improvement < 1.0e-8
        ):
            reasons.append("RISK_IMPROVEMENT_INSUFFICIENT")
        if result.distance_to_nominal < 1.0e-6:
            reasons.append("MAJOR_GEOMETRY_CHANGE_ABSENT")
        for label, continuity in (("LEFT", left), ("RIGHT", right)):
            if continuity[0] is not None and continuity[0] > 0.25:
                reasons.append(f"{label}_SHARE_DISCONTINUITY")
            if continuity[1] is not None and continuity[1] < 0.95:
                reasons.append(f"{label}_COSINE_DISCONTINUITY")
        raw_points.append(
            LambdaParetoPoint(
                float(result.lambda_p),
                result.r_edit,
                result.r_ph,
                result.velocity,
                share,
                top1,
                effective,
                result.distance_to_nominal,
                left[0],
                right[0],
                left[1],
                right[1],
                not reasons,
                tuple(reasons),
                result.identity_sha256,
            )
        )
    eligible = [item for item in raw_points if item.eligible]
    if not eligible:
        raise ODEBFContractError(
            "SOLVER_GEOMETRY_BLOCKED: no stable interior lambda"
        )
    selected = min(eligible, key=lambda item: item.lambda_p)
    rule = {
        "grid": "{0} union {10^(j/2):j=-8..4}",
        "stable_interior_only": True,
        "minimum_r_edit": 0.90,
        "maximum_top1_share_exclusive": 0.80,
        "minimum_effective_layer_count_exclusive": 1.5,
        "minimum_relative_risk_improvement": 0.05,
        "zero_baseline_minimum_absolute_risk_improvement": 1.0e-8,
        "minimum_distance_to_nominal": 1.0e-6,
        "maximum_adjacent_share_l1": 0.25,
        "minimum_adjacent_coefficient_cosine": 0.95,
        "require_two_distinct_geometries": True,
        "selection": "SMALLEST_POSITIVE_ELIGIBLE_LAMBDA",
    }
    payload = {
        "schema": "ode-edit-s05-p1r17-lambda-pareto-lock/v1",
        "source": asdict(source),
        "lambda_grid": list(FULL_DRIVE_LAMBDA_GRID),
        "selected_lambda": selected.lambda_p,
        "points": [asdict(item) for item in raw_points],
        "selection_rule": rule,
        "outcome_model_access_count": 0,
        "model_forward_count": 0,
        "gpu_action_count": 0,
        "weight_write_count": 0,
    }
    return LambdaParetoLock(
        source,
        selected.lambda_p,
        tuple(raw_points),
        rule,
        canonical_hash(payload),
    )


def select_model_lambda_dense(
    problem: RoutingProblem,
    inventory: FixedE8SoftInventory,
    *,
    edit_slopes: Sequence[float],
    source: LambdaReplaySource,
) -> dict[str, Any]:
    """Run the outcome-free A6 dense replay for one model geometry.

    Uncertified solver points remain recorded and ineligible.  They never
    change the sealed numerical certificates or trigger a scientific retry.
    """

    cache: dict[float, FullDriveRoutingResult] = {}
    failures: dict[float, dict[str, str]] = {}

    def solve(value: float) -> FullDriveRoutingResult | None:
        key = float(value)
        if key not in cache and key not in failures:
            try:
                cache[key] = solve_full_drive_routing(
                    problem,
                    inventory,
                    edit_slopes=edit_slopes,
                    arm=FullDriveArm.SOFT,
                    lambda_p=key,
                )
            except ODEBFContractError as exc:
                failures[key] = {
                    "exception_class": type(exc).__name__,
                    "message_sha256": canonical_hash({"message": str(exc)}),
                    "status": "NUMERICAL_CERTIFICATE_FAILED_INELIGIBLE",
                }
        return cache.get(key)

    dense = [
        10.0
        ** (
            math.log10(FULL_DRIVE_DENSE_LOG_MIN)
            + index
            * (
                math.log10(FULL_DRIVE_DENSE_LOG_MAX)
                - math.log10(FULL_DRIVE_DENSE_LOG_MIN)
            )
            / (FULL_DRIVE_DENSE_LOG_COUNT - 1)
        )
        for index in range(FULL_DRIVE_DENSE_LOG_COUNT)
    ]
    for value in (0.0, *dense):
        solve(value)
    certified = sorted(value for value in cache if value > 0.0)
    refinement_values: list[float] = []
    for left, right in zip(certified, certified[1:], strict=False):
        left_result = cache[left]
        right_result = cache[right]
        if (
            (left_result.r_edit - FULL_DRIVE_EDIT_RETENTION_FLOOR)
            * (right_result.r_edit - FULL_DRIVE_EDIT_RETENTION_FLOOR)
            > 0.0
            or left_result.r_edit == right_result.r_edit
        ):
            continue
        lower = math.log10(left)
        upper = math.log10(right)
        for _ in range(64):
            if upper - lower <= FULL_DRIVE_LOG_REFINEMENT_TOLERANCE:
                break
            midpoint = 0.5 * (lower + upper)
            candidate = 10.0**midpoint
            result = solve(candidate)
            if result is None:
                upper = midpoint
                continue
            lower_result = solve(10.0**lower)
            if lower_result is None:
                raise ODEBFContractError(
                    "full-drive dense replay lost its certified bracket"
                )
            if (
                (lower_result.r_edit - FULL_DRIVE_EDIT_RETENTION_FLOOR)
                * (result.r_edit - FULL_DRIVE_EDIT_RETENTION_FLOOR)
                <= 0.0
            ):
                upper = midpoint
            else:
                lower = midpoint
        refinement_values.extend((10.0**lower, 10.0**upper))
        solve(10.0**lower)
        solve(10.0**upper)

    points: list[dict[str, Any]] = []
    eligible: list[tuple[float, float, float, float, FullDriveRoutingResult]] = []
    for value, result in sorted(cache.items()):
        share, top1, effective = _distribution(result.velocity)
        velocity_l1 = float(np.sum(np.abs(np.asarray(result.velocity))))
        reasons: list[str] = []
        if value <= 0.0:
            reasons.append("ZERO_LAMBDA_IDENTITY_CONTROL")
        if result.r_edit < FULL_DRIVE_EDIT_RETENTION_FLOOR:
            reasons.append("EDIT_RETENTION_BELOW_POINT_NINE")
        if result.distance_to_nominal < FULL_DRIVE_MATERIAL_DISTANCE:
            reasons.append("MATERIAL_GEOMETRY_CHANGE_ABSENT")
        if velocity_l1 <= FULL_DRIVE_NONZERO_L1:
            reasons.append("ZERO_OR_NEAR_ZERO_VELOCITY")
        if top1 >= FULL_DRIVE_MAX_TOP1_EXCLUSIVE:
            reasons.append("SEVERE_SINGLE_LAYER_TOP1")
        if effective <= FULL_DRIVE_MIN_EFFECTIVE_LAYERS_EXCLUSIVE:
            reasons.append("SEVERE_SINGLE_LAYER_EFFECTIVE_COUNT")
        values = (
            result.r_edit,
            result.r_ph,
            result.distance_to_nominal,
            *result.velocity,
        )
        if any(not math.isfinite(float(item)) for item in values):
            reasons.append("NONFINITE")
        point = {
            "lambda_p": value,
            "r_edit": result.r_edit,
            "r_ph": result.r_ph,
            "distance_to_nominal": result.distance_to_nominal,
            "velocity": list(result.velocity),
            "velocity_l1": velocity_l1,
            "share": list(share),
            "top1_share": top1,
            "effective_layer_count": effective,
            "routing_identity_sha256": result.identity_sha256,
            "eligible": not reasons,
            "rejection_reasons": reasons,
        }
        points.append(point)
        if not reasons:
            eligible.append(
                (
                    result.r_ph,
                    -result.r_edit,
                    result.distance_to_nominal,
                    value,
                    result,
                )
            )
    if eligible:
        selected_tuple = min(eligible, key=lambda item: item[:4])
        selected_lambda = selected_tuple[3]
        selected_result = selected_tuple[4]
        classification = "MODEL_CALIBRATED_LAMBDA"
    else:
        selected_lambda = 0.0
        selected_result = cache[0.0]
        classification = "FULL_SOFT_DEGENERATES_TO_FULL_NOSOFT_BY_GEOMETRY"
    payload = {
        "schema": "ode-edit-s05-p1r17-model-lambda-replay/v1",
        "source": asdict(source),
        "search": {
            "domain": [FULL_DRIVE_DENSE_LOG_MIN, FULL_DRIVE_DENSE_LOG_MAX],
            "dense_log_count": FULL_DRIVE_DENSE_LOG_COUNT,
            "include_zero_identity": True,
            "refinement_tolerance_log10": (
                FULL_DRIVE_LOG_REFINEMENT_TOLERANCE
            ),
            "maximum_refinement_iterations": 64,
            "refinement_boundary": "R_EDIT_EQUALS_POINT_NINE",
        },
        "eligibility": {
            "minimum_r_edit": FULL_DRIVE_EDIT_RETENTION_FLOOR,
            "minimum_distance_to_nominal": FULL_DRIVE_MATERIAL_DISTANCE,
            "minimum_velocity_l1": FULL_DRIVE_NONZERO_L1,
            "maximum_top1_share_exclusive": FULL_DRIVE_MAX_TOP1_EXCLUSIVE,
            "minimum_effective_layer_count_exclusive": (
                FULL_DRIVE_MIN_EFFECTIVE_LAYERS_EXCLUSIVE
            ),
            "finite_certified_solution_required": True,
        },
        "selection": (
            "MIN_R_PH_THEN_HIGHER_R_EDIT_THEN_SMALLER_"
            "DISTANCE_TO_NOMINAL_THEN_SMALLER_LAMBDA"
        ),
        "selected_lambda": selected_lambda,
        "classification": classification,
        "selected_routing_identity_sha256": selected_result.identity_sha256,
        "eligible_count": len(eligible),
        "points": points,
        "uncertified_points": [
            {"lambda_p": value, **receipt}
            for value, receipt in sorted(failures.items())
        ],
        "refinement_values": sorted(set(refinement_values)),
        "outcome_model_access_count": 0,
        "heldout_access_count": 0,
        "weight_write_count": 0,
        "production_outcome_access_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def full_drive_numerical_contract() -> dict[str, Any]:
    payload = {
        "method_id": FULL_DRIVE_METHOD_ID,
        "layer_order": list(FIXED_E8_LAYER_ORDER),
        "k": 8,
        "h": float(FIXED_E8_H),
        "lambda_grid": list(FULL_DRIVE_LAMBDA_GRID),
        "epsilon_j": FULL_DRIVE_EPS_J,
        "epsilon_c": FULL_DRIVE_EPS_C,
        "normalization_epsilon": FIXED_E8_NORMALIZATION_EPSILON,
        "solver_primal_tolerance": FIXED_E8_PRIMAL_TOLERANCE,
        "solver_kkt_tolerance": FIXED_E8_KKT_TOLERANCE,
        "progress_floor": 0.0,
        "kappa": 0.0,
        "requested_progress": None,
        "retry_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "FULL_DRIVE_EPS_C",
    "FULL_DRIVE_EPS_J",
    "FULL_DRIVE_LAMBDA_GRID",
    "FULL_DRIVE_METHOD_ID",
    "FULL_DRIVE_DENSE_LOG_COUNT",
    "FULL_DRIVE_DENSE_LOG_MAX",
    "FULL_DRIVE_DENSE_LOG_MIN",
    "FULL_DRIVE_EDIT_RETENTION_FLOOR",
    "FULL_DRIVE_LOG_REFINEMENT_TOLERANCE",
    "FullDriveArm",
    "LambdaParetoLock",
    "LambdaParetoPoint",
    "LambdaReplaySource",
    "FullDriveRoutingResult",
    "FullDriveStepMode",
    "full_drive_numerical_contract",
    "replay_and_select_lambda",
    "select_model_lambda_dense",
    "solve_full_drive_routing",
]
