"""Semantic-deficit-constrained residual transport writer.

This module owns only the P2R5 writer mass, realization calibration, semantic
lexicographic solve, and same-semantic-face preservation routing.  Target
construction, physical response measurement, factor construction, materializing
transactions, and evaluators remain in the sealed P2R4/P2R2/P2R1 modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.linalg import null_space
from scipy.optimize import LinearConstraint, linprog, minimize, nnls, root

from .contracts import ODEBFContractError, canonical_hash
from .functional import WaypointFactor, tensor_sha256
from .p1_backend import P1DynamicField
from .p2r2_residual_transport_writer import ProposalQuadratics
from .progress_simplex_routing import (
    SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE,
    SIMPLEX_PRIMAL_TOLERANCE,
    SIMPLEX_XI_TIE_TOLERANCE,
)


P2R5_INSTRUCTION_ID = (
    "ODEEDIT-S05-P2R5-SEMANTIC-DEFICIT-CONSTRAINED-RESIDUAL-TRANSPORT-V1"
)
P2R5_METHOD_ID = "P2R5-SEMANTIC-DEFICIT-CONSTRAINED-RESIDUAL-TRANSPORT-V1"
P2R5_ARMS = ("SDRT-CAP", "SDRT-STRUCTP")
P2R5_NUMERICAL_EPSILON = SIMPLEX_PRIMAL_TOLERANCE


class P2R5RoutingTechnicalError(ODEBFContractError):
    """Fail-close error carrying raw-free numerical observability."""

    def __init__(self, message: str, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.raw_free_receipt = dict(receipt)


@dataclass(frozen=True, slots=True)
class SDRTCalibration:
    eta: float
    prior_step_count: int
    pair_count: int
    numerator: float
    denominator: float
    status: str
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p2r5-pooled-nnls-calibration/v1",
            "eta": self.eta,
            "prior_step_count": self.prior_step_count,
            "pair_count": self.pair_count,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "status": self.status,
            "eta_floor_count": 0,
            "model_forward_count": 0,
            "model_backward_count": 0,
            "identity_sha256": self.identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class SDRTQuadratics:
    capacity_gram: torch.Tensor
    capacity_cross: torch.Tensor
    prior_capacity: float
    structural_p_gram: torch.Tensor
    structural_p_cross: torch.Tensor
    prior_structural_p: float
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p2r5-cumulative-quadratics/v1",
            "capacity_gram_sha256": tensor_sha256(self.capacity_gram),
            "capacity_cross_sha256": tensor_sha256(self.capacity_cross),
            "prior_capacity": self.prior_capacity,
            "structural_p_gram_sha256": tensor_sha256(self.structural_p_gram),
            "structural_p_cross_sha256": tensor_sha256(self.structural_p_cross),
            "prior_structural_p": self.prior_structural_p,
            "identity_sha256": self.identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class SDRTRoutingResult:
    arm: str
    allocation: torch.Tensor
    deficit: torch.Tensor
    raw_predicted_response: torch.Tensor
    calibrated_predicted_response: torch.Tensor
    semantic_response_star: torch.Tensor
    eta: float
    e1_xi: float
    e2_total_normalized_mismatch: float
    cumulative_capacity: float
    marginal_capacity: float
    cumulative_structural_p: float
    marginal_structural_p: float
    coefficient_mass_by_request: tuple[float, ...]
    active_mass_constraints: tuple[int, ...]
    feasible_rank: int
    feasible_nullity: int
    semantic_face_max_abs_residual: float
    solver_receipts: tuple[Mapping[str, Any], ...]
    status: str
    identity_sha256: str

    @property
    def predicted_progress(self) -> torch.Tensor:
        return self.calibrated_predicted_response

    def raw_free_payload(self) -> dict[str, Any]:
        allocation = self.allocation
        return {
            "schema": "ode-edit-s05-p2r5-sdrt-routing/v1",
            "arm": self.arm,
            "allocation_shape": list(allocation.shape),
            "allocation_sha256": tensor_sha256(allocation),
            "allocation_by_layer_request": allocation.tolist(),
            "deficit_by_request": self.deficit.tolist(),
            "raw_predicted_response_by_request": self.raw_predicted_response.tolist(),
            "calibrated_predicted_response_by_request": (
                self.calibrated_predicted_response.tolist()
            ),
            "semantic_response_star_by_request": self.semantic_response_star.tolist(),
            "eta": self.eta,
            "e1_xi": self.e1_xi,
            "e2_total_normalized_mismatch": self.e2_total_normalized_mismatch,
            "cumulative_capacity": self.cumulative_capacity,
            "marginal_capacity": self.marginal_capacity,
            "cumulative_structural_p": self.cumulative_structural_p,
            "marginal_structural_p": self.marginal_structural_p,
            "coefficient_mass_by_request": list(self.coefficient_mass_by_request),
            "coefficient_mass_min": min(self.coefficient_mass_by_request),
            "coefficient_mass_max": max(self.coefficient_mass_by_request),
            "active_mass_constraints": list(self.active_mass_constraints),
            "feasible_rank": self.feasible_rank,
            "feasible_nullity": self.feasible_nullity,
            "semantic_face_max_abs_residual": self.semantic_face_max_abs_residual,
            "solver_receipts": [dict(item) for item in self.solver_receipts],
            "status": self.status,
            "neutral_fallback_count": 0,
            "hard_p_budget_influence_count": 0,
            "functional_p_veto_count": 0,
            "strength_attenuation_by_preservation_count": 0,
            "candidate_forward_count": 0,
            "candidate_materialization_count": 0,
            "identity_sha256": self.identity_sha256,
        }


def clamp_safe_semantic_deficit(
    current_w_nll: Sequence[float] | torch.Tensor,
    clamped_z_nll: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Return exact request-wise ``[L_W-L_z]_+`` in FP64."""

    current = torch.as_tensor(current_w_nll, dtype=torch.float64)
    oracle = torch.as_tensor(clamped_z_nll, dtype=torch.float64)
    if current.ndim != 1 or current.shape != oracle.shape or current.numel() != 10:
        raise ODEBFContractError("P2R5 semantic deficit geometry differs")
    if not bool(torch.isfinite(current).all() and torch.isfinite(oracle).all()):
        raise ODEBFContractError("P2R5 semantic deficit is nonfinite")
    return torch.clamp(current - oracle, min=0.0).contiguous()


def pooled_nonnegative_realization_calibration(
    prior_predicted_by_step: Sequence[torch.Tensor],
    prior_actual_by_step: Sequence[torch.Tensor],
) -> SDRTCalibration:
    """Exact scalar NNLS for accepted prior-step request responses."""

    if len(prior_predicted_by_step) != len(prior_actual_by_step):
        raise ODEBFContractError("P2R5 calibration history axes differ")
    if not prior_predicted_by_step:
        payload = {
            "eta": 1.0,
            "prior_step_count": 0,
            "pair_count": 0,
            "numerator": 0.0,
            "denominator": 0.0,
            "status": "ENTRY_ETA_ONE",
        }
        return SDRTCalibration(1.0, 0, 0, 0.0, 0.0, "ENTRY_ETA_ONE", canonical_hash(payload))

    predicted = torch.cat(
        [item.detach().to(device="cpu", dtype=torch.float64).reshape(-1) for item in prior_predicted_by_step]
    )
    actual = torch.cat(
        [item.detach().to(device="cpu", dtype=torch.float64).reshape(-1) for item in prior_actual_by_step]
    )
    if predicted.shape != actual.shape or predicted.numel() != 10 * len(prior_predicted_by_step):
        raise ODEBFContractError("P2R5 calibration request coverage differs")
    if not bool(torch.isfinite(predicted).all() and torch.isfinite(actual).all()):
        raise ODEBFContractError("P2R5 calibration history is nonfinite")
    numerator = float(torch.dot(predicted, actual))
    denominator = float(torch.dot(predicted, predicted))
    if denominator == 0.0:
        eta = 0.0
        status = "NNLS_ZERO_DESIGN_ETA_ZERO"
    else:
        eta = max(0.0, numerator / denominator)
        status = "POOLED_NNLS_CERTIFIED"
    if not math.isfinite(eta):
        raise ODEBFContractError("P2R5 calibration result is nonfinite")
    payload = {
        "eta": eta,
        "prior_step_count": len(prior_predicted_by_step),
        "pair_count": int(predicted.numel()),
        "numerator": numerator,
        "denominator": denominator,
        "status": status,
    }
    return SDRTCalibration(
        eta,
        len(prior_predicted_by_step),
        int(predicted.numel()),
        numerator,
        denominator,
        status,
        canonical_hash(payload),
    )


def build_sdrt_quadratics(
    field: P1DynamicField,
    prior_factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
    base: ProposalQuadratics,
) -> SDRTQuadratics:
    """Add exact cumulative endpoint-capacity offset/cross to sealed P2R2 terms."""

    request_count = field.current_z.shape[1]
    alpha_count = len(field.layers) * request_count
    if base.capacity_gram.shape != (alpha_count, alpha_count):
        raise ODEBFContractError("P2R5 quadratic geometry differs")
    capacity_cross = torch.zeros(alpha_count, dtype=torch.float64)
    prior_capacity = 0.0
    for layer_ordinal, layer in enumerate(field.layers):
        start = layer_ordinal * request_count
        end = start + request_count
        residual = layer.residual.detach().to(torch.float64)
        q = layer.q.detach().to(torch.float64)
        previous = tuple(prior_factors_by_weight.get(layer.weight_name, ()))
        for factor in previous:
            left = factor.left.detach().to(torch.float64)
            right = factor.right.detach().to(torch.float64)
            theta = float(factor.theta)
            capacity_cross[start:end] += theta * torch.sum(
                (left.T @ residual) * (right.T @ q), dim=0
            )
        for left_index, left_factor in enumerate(previous):
            left_l = left_factor.left.detach().to(torch.float64)
            right_l = left_factor.right.detach().to(torch.float64)
            for right_factor in previous[left_index:]:
                left_r = right_factor.left.detach().to(torch.float64)
                right_r = right_factor.right.detach().to(torch.float64)
                value = (
                    float(left_factor.theta)
                    * float(right_factor.theta)
                    * float(torch.sum((left_l.T @ left_r) * (right_l.T @ right_r)))
                )
                prior_capacity += value if right_factor is left_factor else 2.0 * value
    if not bool(torch.isfinite(capacity_cross).all()) or not math.isfinite(prior_capacity):
        raise ODEBFContractError("P2R5 cumulative capacity terms are nonfinite")
    if prior_capacity < -P2R5_NUMERICAL_EPSILON:
        raise ODEBFContractError("P2R5 prior capacity is negative")
    prior_capacity = max(0.0, prior_capacity)
    payload = {
        "capacity_gram_sha256": tensor_sha256(base.capacity_gram),
        "capacity_cross_sha256": tensor_sha256(capacity_cross),
        "prior_capacity": prior_capacity,
        "structural_p_gram_sha256": tensor_sha256(base.structural_p_gram),
        "structural_p_cross_sha256": tensor_sha256(base.structural_p_cross),
        "prior_structural_p": base.prior_structural_p,
        "base_quadratics_sha256": base.identity_sha256,
    }
    return SDRTQuadratics(
        base.capacity_gram.detach().to(torch.float64).contiguous(),
        capacity_cross.contiguous(),
        prior_capacity,
        base.structural_p_gram.detach().to(torch.float64).contiguous(),
        base.structural_p_cross.detach().to(torch.float64).contiguous(),
        base.prior_structural_p,
        canonical_hash(payload),
    )


def _mass_matrix(layer_count: int, request_count: int) -> np.ndarray:
    matrix = np.zeros((request_count, layer_count * request_count), dtype=np.float64)
    for request in range(request_count):
        matrix[request, request::request_count] = 1.0
    return matrix


def _semantic_optimum(
    calibrated_response: np.ndarray,
    deficit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    request_count, alpha_count = calibrated_response.shape
    layer_count = alpha_count // request_count
    mass = _mass_matrix(layer_count, request_count)

    # E1: minimize worst normalized under-deficit slack xi.
    objective = np.zeros(alpha_count + 1, dtype=np.float64)
    objective[-1] = 1.0
    e1_a_ub = np.concatenate(
        [
            np.concatenate([mass, np.zeros((request_count, 1), dtype=np.float64)], axis=1),
            np.concatenate([-calibrated_response, -deficit[:, None]], axis=1),
        ],
        axis=0,
    )
    e1_b_ub = np.concatenate(
        [np.ones(request_count, dtype=np.float64), -deficit], axis=0
    )
    first = linprog(
        objective,
        A_ub=e1_a_ub,
        b_ub=e1_b_ub,
        bounds=[(0.0, None)] * alpha_count + [(0.0, None)],
        method="highs",
    )
    if not first.success or not np.all(np.isfinite(first.x)):
        raise P2R5RoutingTechnicalError(
            "P2R5 E1 semantic solve failed",
            {
                "schema": "ode-edit-s05-p2r5-routing-technical/v1",
                "stage": "E1_WORST_NORMALIZED_UNDER_DEFICIT",
                "status": int(first.status),
                "message_sha256": canonical_hash(str(first.message)),
            },
        )
    xi = max(0.0, float(first.x[-1]))

    # E2: same E1 optimum, minimum normalized absolute mismatch/overshoot.
    scale = deficit + P2R5_NUMERICAL_EPSILON
    e2_objective = np.concatenate(
        [np.zeros(alpha_count, dtype=np.float64), 1.0 / scale], axis=0
    )
    zeros_t = np.zeros((request_count, request_count), dtype=np.float64)
    identity_t = np.eye(request_count, dtype=np.float64)
    e1_face = (1.0 - xi) * deficit - P2R5_NUMERICAL_EPSILON
    e2_a_ub = np.concatenate(
        [
            np.concatenate([mass, zeros_t], axis=1),
            np.concatenate([-calibrated_response, zeros_t], axis=1),
            np.concatenate([calibrated_response, -identity_t], axis=1),
            np.concatenate([-calibrated_response, -identity_t], axis=1),
        ],
        axis=0,
    )
    e2_b_ub = np.concatenate(
        [
            np.ones(request_count, dtype=np.float64),
            -e1_face,
            deficit,
            -deficit,
        ],
        axis=0,
    )
    second = linprog(
        e2_objective,
        A_ub=e2_a_ub,
        b_ub=e2_b_ub,
        bounds=[(0.0, None)] * (alpha_count + request_count),
        method="highs",
    )
    if not second.success or not np.all(np.isfinite(second.x)):
        raise P2R5RoutingTechnicalError(
            "P2R5 E2 semantic solve failed",
            {
                "schema": "ode-edit-s05-p2r5-routing-technical/v1",
                "stage": "E2_TOTAL_NORMALIZED_MISMATCH",
                "status": int(second.status),
                "message_sha256": canonical_hash(str(second.message)),
                "e1_xi": xi,
            },
        )
    allocation = second.x[:alpha_count]
    response_star = calibrated_response @ allocation
    mismatch = float(np.sum(np.abs(deficit - response_star) / scale))
    return allocation, response_star, xi, mismatch


def _quadratic_value(matrix: np.ndarray, cross: np.ndarray, x: np.ndarray) -> float:
    return float(x @ matrix @ x + 2.0 * cross @ x)


def _same_semantic_face_solve(
    start: np.ndarray,
    objective_matrix: np.ndarray,
    objective_cross: np.ndarray,
    calibrated_response: np.ndarray,
    response_star: np.ndarray,
    *,
    p_matrix: np.ndarray | None = None,
    p_cross: np.ndarray | None = None,
    p_limit: float | None = None,
    stage: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    request_count, alpha_count = calibrated_response.shape
    layer_count = alpha_count // request_count
    mass = _mass_matrix(layer_count, request_count)
    independent: list[int] = []
    current_rank = 0
    for request in range(request_count):
        candidate = calibrated_response[[*independent, request]]
        candidate_rank = int(
            np.linalg.matrix_rank(candidate, tol=P2R5_NUMERICAL_EPSILON)
        )
        if candidate_rank > current_rank:
            independent.append(request)
            current_rank = candidate_rank
    face_matrix = calibrated_response[independent]
    face_value = response_star[independent]
    if independent:
        coordinate_basis = null_space(
            face_matrix,
            rcond=P2R5_NUMERICAL_EPSILON,
        )
    else:
        coordinate_basis = np.eye(alpha_count, dtype=np.float64)
    if (
        coordinate_basis.ndim != 2
        or coordinate_basis.shape[0] != alpha_count
        or not np.all(np.isfinite(coordinate_basis))
    ):
        raise P2R5RoutingTechnicalError(
            f"P2R5 {stage} semantic-face coordinates failed",
            {
                "schema": "ode-edit-s05-p2r5-routing-technical/v3",
                "stage": stage,
                "coordinate_backend": "SCIPY_SVD_SEMANTIC_FACE_NULLSPACE",
                "semantic_rank": int(np.linalg.matrix_rank(face_matrix)),
            },
        )
    semantic_basis_residual = (
        float(np.max(np.abs(face_matrix @ coordinate_basis)))
        if independent and coordinate_basis.shape[1] else 0.0
    )
    reduced_count = int(coordinate_basis.shape[1])

    def expand(value: np.ndarray) -> np.ndarray:
        return start + coordinate_basis @ value

    linear_constraints = [
        LinearConstraint(
            coordinate_basis,
            -start,
            np.full(alpha_count, np.inf, dtype=np.float64),
        ),
        LinearConstraint(
            mass @ coordinate_basis,
            np.full(request_count, -np.inf, dtype=np.float64),
            np.ones(request_count, dtype=np.float64) - mass @ start,
        ),
    ]
    p_hessian: np.ndarray | None = None
    reduced_p_hessian: np.ndarray | None = None
    if p_limit is not None:
        if p_matrix is None or p_cross is None:
            raise ODEBFContractError("P2R5 Structural-P tie is absent")
        p_hessian = p_matrix + p_matrix.T
        reduced_p_hessian = coordinate_basis.T @ p_hessian @ coordinate_basis
    objective_hessian = objective_matrix + objective_matrix.T
    reduced_objective_hessian = (
        coordinate_basis.T @ objective_hessian @ coordinate_basis
    )
    start_semantic_residual = (
        float(np.max(np.abs(face_matrix @ start - face_value)))
        if independent else 0.0
    )
    start_mass_violation = max(0.0, float(np.max(mass @ start - 1.0)))
    start_p_violation = (
        max(0.0, _quadratic_value(p_matrix, p_cross, start) - p_limit)
        if p_limit is not None and p_matrix is not None and p_cross is not None
        else 0.0
    )
    objective_symmetric = 0.5 * (objective_matrix + objective_matrix.T)
    objective_eigenvalues = np.linalg.eigvalsh(objective_symmetric)
    if float(np.min(objective_eigenvalues)) < -P2R5_NUMERICAL_EPSILON:
        raise P2R5RoutingTechnicalError(
            f"P2R5 {stage} objective is nonconvex",
            {
                "schema": "ode-edit-s05-p2r5-routing-technical/v3",
                "stage": stage,
                "objective_min_eigenvalue": float(np.min(objective_eigenvalues)),
                "primal_tolerance": P2R5_NUMERICAL_EPSILON,
            },
        )
    if p_limit is not None and p_matrix is not None:
        p_eigenvalues = np.linalg.eigvalsh(0.5 * (p_matrix + p_matrix.T))
        if float(np.min(p_eigenvalues)) < -P2R5_NUMERICAL_EPSILON:
            raise P2R5RoutingTechnicalError(
                f"P2R5 {stage} P tie is nonconvex",
                {
                    "schema": "ode-edit-s05-p2r5-routing-technical/v3",
                    "stage": stage,
                    "p_min_eigenvalue": float(np.min(p_eigenvalues)),
                    "primal_tolerance": P2R5_NUMERICAL_EPSILON,
                },
            )
    objective = lambda value: _quadratic_value(
        objective_matrix, objective_cross, expand(value)
    )
    objective_gradient = lambda value: coordinate_basis.T @ (
        objective_hessian @ expand(value) + 2.0 * objective_cross
    )
    backend = "SCIPY_TRUST_CONSTR_EXACT_CONSTRAINTS_ANALYTIC_DERIVATIVES"
    try:
        if p_limit is None:
            result = minimize(
                objective,
                np.zeros(reduced_count, dtype=np.float64),
                jac=objective_gradient,
                hess=lambda value: reduced_objective_hessian,
                method="trust-constr",
                constraints=linear_constraints,
                options={
                    "gtol": SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE,
                    "xtol": SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE,
                    "barrier_tol": SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE,
                    "maxiter": 3000,
                    "verbose": 0,
                },
            )
        else:
            backend = "SCIPY_SLSQP_NULLSPACE_CONVEX_P_TIE_ANALYTIC_GRADIENT"
            if p_matrix is None or p_cross is None:
                raise ODEBFContractError("P2R5 Structural-P tie is absent")
            result = minimize(
                objective,
                np.zeros(reduced_count, dtype=np.float64),
                jac=objective_gradient,
                method="SLSQP",
                constraints=(
                    {
                        "type": "ineq",
                        "fun": lambda value: expand(value),
                        "jac": lambda value: coordinate_basis,
                    },
                    {
                        "type": "ineq",
                        "fun": lambda value: np.ones(request_count, dtype=np.float64)
                        - mass @ expand(value),
                        "jac": lambda value: -mass @ coordinate_basis,
                    },
                    {
                        "type": "ineq",
                        "fun": lambda value, m=p_matrix, c=p_cross: float(
                            p_limit - _quadratic_value(m, c, expand(value))
                        ),
                        "jac": lambda value, m=p_matrix, c=p_cross: -coordinate_basis.T
                        @ ((m + m.T) @ expand(value) + 2.0 * c),
                    },
                ),
                options={
                    "ftol": SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE,
                    "maxiter": 3000,
                    "disp": False,
                },
            )
            candidate_y = np.asarray(result.x, dtype=np.float64)
            candidate_x = expand(candidate_y)
            active_zero = tuple(
                int(index)
                for index in np.flatnonzero(
                    candidate_x <= P2R5_NUMERICAL_EPSILON
                )
            )
            candidate_mass_slack = 1.0 - mass @ candidate_x
            active_mass = tuple(
                int(index)
                for index in np.flatnonzero(
                    candidate_mass_slack <= P2R5_NUMERICAL_EPSILON
                )
            )
            candidate_p_slack = float(
                p_limit - _quadratic_value(p_matrix, p_cross, candidate_x)
            )
            active_p = candidate_p_slack <= P2R5_NUMERICAL_EPSILON

            def active_values(value: np.ndarray) -> np.ndarray:
                selected = expand(value)
                values = [float(selected[index]) for index in active_zero]
                values.extend(
                    float(1.0 - mass[index] @ selected)
                    for index in active_mass
                )
                if active_p:
                    values.append(
                        float(
                            p_limit
                            - _quadratic_value(p_matrix, p_cross, selected)
                        )
                    )
                return np.asarray(values, dtype=np.float64)

            def active_jacobian(value: np.ndarray) -> np.ndarray:
                selected = expand(value)
                rows = [coordinate_basis[index] for index in active_zero]
                rows.extend(
                    -(mass @ coordinate_basis)[index] for index in active_mass
                )
                if active_p:
                    rows.append(
                        -coordinate_basis.T
                        @ (
                            (p_matrix + p_matrix.T) @ selected
                            + 2.0 * p_cross
                        )
                    )
                return (
                    np.stack(rows)
                    if rows
                    else np.empty((0, reduced_count), dtype=np.float64)
                )

            initial_active_jacobian = active_jacobian(candidate_y)
            if initial_active_jacobian.shape[0]:
                initial_multiplier, _ = nnls(
                    initial_active_jacobian.T,
                    objective_gradient(candidate_y),
                )

                def kkt_function(value: np.ndarray) -> np.ndarray:
                    reduced = value[:reduced_count]
                    multiplier = value[reduced_count:]
                    jacobian = active_jacobian(reduced)
                    stationarity = (
                        objective_gradient(reduced) - jacobian.T @ multiplier
                    )
                    return np.concatenate((stationarity, active_values(reduced)))

                def kkt_jacobian(value: np.ndarray) -> np.ndarray:
                    reduced = value[:reduced_count]
                    multiplier = value[reduced_count:]
                    jacobian = active_jacobian(reduced)
                    stationarity_hessian = reduced_objective_hessian.copy()
                    if active_p:
                        stationarity_hessian = (
                            stationarity_hessian
                            + multiplier[-1] * reduced_p_hessian
                        )
                    upper = np.concatenate(
                        (stationarity_hessian, -jacobian.T), axis=1
                    )
                    lower = np.concatenate(
                        (
                            jacobian,
                            np.zeros(
                                (jacobian.shape[0], jacobian.shape[0]),
                                dtype=np.float64,
                            ),
                        ),
                        axis=1,
                    )
                    return np.concatenate((upper, lower), axis=0)

                initial_kkt = np.concatenate((candidate_y, initial_multiplier))
                polished = root(
                    kkt_function,
                    initial_kkt,
                    jac=kkt_jacobian,
                    method="hybr",
                    options={
                        "xtol": SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE,
                        "maxfev": 3000,
                    },
                )
                before_residual = float(
                    np.max(np.abs(kkt_function(initial_kkt)))
                )
                after_residual = (
                    float(np.max(np.abs(kkt_function(polished.x))))
                    if np.all(np.isfinite(polished.x)) else math.inf
                )
                if after_residual < before_residual:
                    result.x = np.asarray(polished.x[:reduced_count], dtype=np.float64)
                result.p2r5_active_set_polish_success = bool(polished.success)
                result.p2r5_active_set_polish_before = before_residual
                result.p2r5_active_set_polish_after = after_residual
                result.p2r5_active_set_polish_constraint_count = int(
                    initial_active_jacobian.shape[0]
                )
            else:
                result.p2r5_active_set_polish_success = True
                result.p2r5_active_set_polish_before = 0.0
                result.p2r5_active_set_polish_after = 0.0
                result.p2r5_active_set_polish_constraint_count = 0
    except Exception as exc:
        raise P2R5RoutingTechnicalError(
            f"P2R5 {stage} solve raised",
            {
                "schema": "ode-edit-s05-p2r5-routing-technical/v2",
                "stage": stage,
                "backend": backend,
                "exception_class": type(exc).__name__,
                "message_sha256": canonical_hash(str(exc)),
                "semantic_rank": int(np.linalg.matrix_rank(face_matrix)),
                "semantic_nullity": reduced_count,
                "semantic_basis_max_abs_residual": semantic_basis_residual,
                "start_semantic_max_abs_residual": start_semantic_residual,
                "start_mass_violation": start_mass_violation,
                "start_p_violation": start_p_violation,
                "neutral_fallback_count": 0,
            },
        ) from exc
    reduced_x = np.asarray(result.x, dtype=np.float64)
    result_x = expand(reduced_x)
    finite = bool(np.all(np.isfinite(result_x)))
    negative_violation = (
        max(0.0, -float(np.min(result_x))) if finite else math.inf
    )
    mass_violation = (
        max(0.0, float(np.max(mass @ result_x - 1.0))) if finite else math.inf
    )
    semantic_residual = (
        float(np.max(np.abs(face_matrix @ result_x - face_value)))
        if finite and independent else 0.0
    )
    p_violation = (
        max(0.0, _quadratic_value(p_matrix, p_cross, result_x) - p_limit)
        if finite and p_limit is not None and p_matrix is not None and p_cross is not None
        else 0.0
    )
    if p_limit is None:
        optimality = float(getattr(result, "optimality", math.inf))
        constraint_violation = float(getattr(result, "constr_violation", math.inf))
        kkt_active_constraint_count = -1
    else:
        reduced_gradient = objective_gradient(reduced_x)
        active_gradients: list[np.ndarray] = []
        for index in np.flatnonzero(result_x <= P2R5_NUMERICAL_EPSILON):
            active_gradients.append(coordinate_basis[index])
        mass_slack = 1.0 - mass @ result_x
        for index in np.flatnonzero(mass_slack <= P2R5_NUMERICAL_EPSILON):
            active_gradients.append(-(mass @ coordinate_basis)[index])
        p_slack = float(p_limit - _quadratic_value(p_matrix, p_cross, result_x))
        if p_slack <= P2R5_NUMERICAL_EPSILON:
            active_gradients.append(
                -coordinate_basis.T
                @ ((p_matrix + p_matrix.T) @ result_x + 2.0 * p_cross)
            )
        if active_gradients:
            active_matrix = np.stack(active_gradients)
            multiplier, _ = nnls(active_matrix.T, reduced_gradient)
            kkt_residual = reduced_gradient - active_matrix.T @ multiplier
        else:
            kkt_residual = reduced_gradient
        optimality = (
            float(np.max(np.abs(kkt_residual))) if kkt_residual.size else 0.0
        )
        constraint_violation = max(
            negative_violation,
            mass_violation,
            p_violation,
        )
        kkt_active_constraint_count = len(active_gradients)
    certified = (
        finite
        and negative_violation <= P2R5_NUMERICAL_EPSILON
        and mass_violation <= P2R5_NUMERICAL_EPSILON
        and semantic_residual <= P2R5_NUMERICAL_EPSILON
        and p_violation <= P2R5_NUMERICAL_EPSILON
        and optimality <= P2R5_NUMERICAL_EPSILON
        and constraint_violation <= P2R5_NUMERICAL_EPSILON
    )
    solver_receipt = {
        "schema": "ode-edit-s05-p2r5-semantic-face-solver/v1",
        "stage": stage,
        "backend": backend,
        "coordinate_backend": "SCIPY_SVD_SEMANTIC_FACE_NULLSPACE",
        "semantic_rank": len(independent),
        "semantic_nullity": reduced_count,
        "semantic_basis_max_abs_residual": semantic_basis_residual,
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "message_sha256": canonical_hash(str(result.message)),
        "iterations": int(getattr(result, "nit", -1)),
        "function_evaluations": int(getattr(result, "nfev", -1)),
        "gradient_evaluations": int(getattr(result, "njev", -1)),
        "hessian_evaluations": int(getattr(result, "nhev", -1)),
        "optimality": optimality,
        "kkt_active_constraint_count": kkt_active_constraint_count,
        "active_set_polish_success": bool(
            getattr(result, "p2r5_active_set_polish_success", False)
        ),
        "active_set_polish_before": float(
            getattr(result, "p2r5_active_set_polish_before", 0.0)
        ),
        "active_set_polish_after": float(
            getattr(result, "p2r5_active_set_polish_after", 0.0)
        ),
        "active_set_polish_constraint_count": int(
            getattr(result, "p2r5_active_set_polish_constraint_count", -1)
        ),
        "constraint_violation": constraint_violation,
        "negative_violation": negative_violation,
        "mass_violation": mass_violation,
        "semantic_face_max_abs_residual": semantic_residual,
        "p_tie_violation": p_violation,
        "objective_min_eigenvalue": float(np.min(objective_eigenvalues)),
        "objective_max_eigenvalue": float(np.max(objective_eigenvalues)),
        "primal_tolerance": P2R5_NUMERICAL_EPSILON,
        "non_success_certified_with_inherited_tolerance": bool(
            certified and not result.success
        ),
        "certificate_pass": certified,
    }
    solver_receipt["identity_sha256"] = canonical_hash(solver_receipt)
    if not certified:
        raise P2R5RoutingTechnicalError(
            f"P2R5 {stage} solve failed",
            {
                "schema": "ode-edit-s05-p2r5-routing-technical/v2",
                "stage": stage,
                "backend": backend,
                "status": int(result.status),
                "message_sha256": canonical_hash(str(result.message)),
                "iterations": int(getattr(result, "nit", -1)),
                "function_evaluations": int(getattr(result, "nfev", -1)),
                "gradient_evaluations": int(getattr(result, "njev", -1)),
                "hessian_evaluations": int(getattr(result, "nhev", -1)),
                "semantic_rank": int(np.linalg.matrix_rank(face_matrix)),
                "semantic_nullity": reduced_count,
                "semantic_basis_max_abs_residual": semantic_basis_residual,
                "start_semantic_max_abs_residual": start_semantic_residual,
                "start_mass_violation": start_mass_violation,
                "start_p_violation": start_p_violation,
                "optimality": optimality,
                "kkt_active_constraint_count": kkt_active_constraint_count,
                "active_set_polish_success": bool(
                    getattr(result, "p2r5_active_set_polish_success", False)
                ),
                "active_set_polish_before": float(
                    getattr(result, "p2r5_active_set_polish_before", 0.0)
                ),
                "active_set_polish_after": float(
                    getattr(result, "p2r5_active_set_polish_after", 0.0)
                ),
                "constraint_violation": constraint_violation,
                "negative_violation": negative_violation,
                "mass_violation": mass_violation,
                "semantic_face_max_abs_residual": semantic_residual,
                "p_tie_violation": p_violation,
                "objective_min_eigenvalue": float(np.min(objective_eigenvalues)),
                "objective_max_eigenvalue": float(np.max(objective_eigenvalues)),
                "objective_trace": float(np.trace(objective_symmetric)),
                "primal_tolerance": P2R5_NUMERICAL_EPSILON,
                "neutral_fallback_count": 0,
            },
        )
    return result_x, solver_receipt


def _feasible_face_geometry(
    allocation: np.ndarray,
    calibrated_response: np.ndarray,
) -> tuple[tuple[int, ...], int, int]:
    request_count, alpha_count = calibrated_response.shape
    layer_count = alpha_count // request_count
    mass = _mass_matrix(layer_count, request_count)
    mass_values = mass @ allocation
    active_mass = tuple(
        int(item)
        for item in np.flatnonzero(1.0 - mass_values <= P2R5_NUMERICAL_EPSILON)
    )
    active_zero = np.flatnonzero(allocation <= P2R5_NUMERICAL_EPSILON)
    rows = [row for row in calibrated_response]
    rows.extend(mass[item] for item in active_mass)
    rows.extend(np.eye(alpha_count, dtype=np.float64)[item] for item in active_zero)
    matrix = np.stack(rows) if rows else np.empty((0, alpha_count), dtype=np.float64)
    rank = int(np.linalg.matrix_rank(matrix, tol=P2R5_NUMERICAL_EPSILON))
    return active_mass, rank, alpha_count - rank


def solve_sdrt_routing(
    response: torch.Tensor,
    deficit: torch.Tensor,
    calibration: SDRTCalibration,
    quadratics: SDRTQuadratics,
    *,
    arm: str,
) -> SDRTRoutingResult:
    """Lexicographic E1/E2 semantic solve followed by same-face CAP/STRUCTP."""

    if arm not in P2R5_ARMS:
        raise ODEBFContractError("P2R5 arm differs")
    observed = response.detach().to(device="cpu", dtype=torch.float64).numpy()
    deficit_np = deficit.detach().to(device="cpu", dtype=torch.float64).numpy()
    if observed.shape != (10, 50) or deficit_np.shape != (10,):
        raise ODEBFContractError("P2R5 response/deficit geometry differs")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(deficit_np)):
        raise ODEBFContractError("P2R5 response/deficit is nonfinite")
    calibrated = calibration.eta * observed
    semantic_start, response_star, xi, mismatch = _semantic_optimum(calibrated, deficit_np)

    capacity = quadratics.capacity_gram.numpy()
    capacity_cross = quadratics.capacity_cross.numpy()
    structural = quadratics.structural_p_gram.numpy()
    structural_cross = quadratics.structural_p_cross.numpy()
    if arm == "SDRT-CAP":
        selected, cap_receipt = _same_semantic_face_solve(
            semantic_start,
            capacity,
            capacity_cross,
            calibrated,
            response_star,
            stage="CAPACITY_ON_SEMANTIC_FACE",
        )
        solver_receipts = (cap_receipt,)
        status = "SDRT_CAP_CERTIFIED"
    else:
        p_selected, p_receipt = _same_semantic_face_solve(
            semantic_start,
            structural,
            structural_cross,
            calibrated,
            response_star,
            stage="STRUCTURAL_P_ON_SEMANTIC_FACE",
        )
        p_star = _quadratic_value(structural, structural_cross, p_selected)
        p_tie = SIMPLEX_XI_TIE_TOLERANCE * max(float(np.trace(structural)), 1.0e-12)
        selected, cap_receipt = _same_semantic_face_solve(
            p_selected,
            capacity,
            capacity_cross,
            calibrated,
            response_star,
            p_matrix=structural,
            p_cross=structural_cross,
            p_limit=p_star + p_tie,
            stage="CAPACITY_TIE_ON_STRUCTURAL_P_SEMANTIC_FACE",
        )
        solver_receipts = (p_receipt, cap_receipt)
        status = "SDRT_STRUCTP_CERTIFIED"

    raw_progress = observed @ selected
    calibrated_progress = calibrated @ selected
    mass = _mass_matrix(5, 10) @ selected
    negative = max(0.0, -float(np.min(selected)))
    mass_violation = max(0.0, float(np.max(mass - 1.0)))
    semantic_residual = float(np.max(np.abs(calibrated_progress - response_star)))
    if (
        negative > P2R5_NUMERICAL_EPSILON
        or mass_violation > P2R5_NUMERICAL_EPSILON
        or semantic_residual > P2R5_NUMERICAL_EPSILON * 2.0
    ):
        raise P2R5RoutingTechnicalError(
            "P2R5 routing certificate failed",
            {
                "schema": "ode-edit-s05-p2r5-routing-technical/v1",
                "stage": "FINAL_CERTIFICATE",
                "negative_violation": negative,
                "mass_violation": mass_violation,
                "semantic_face_max_abs_residual": semantic_residual,
                "primal_tolerance": P2R5_NUMERICAL_EPSILON,
            },
        )

    marginal_capacity = _quadratic_value(capacity, capacity_cross, selected)
    cumulative_capacity = quadratics.prior_capacity + marginal_capacity
    marginal_p = _quadratic_value(structural, structural_cross, selected)
    cumulative_p = quadratics.prior_structural_p + marginal_p
    if cumulative_capacity < -P2R5_NUMERICAL_EPSILON or cumulative_p < -P2R5_NUMERICAL_EPSILON:
        raise ODEBFContractError("P2R5 cumulative quadratic certificate failed")
    cumulative_capacity = max(0.0, cumulative_capacity)
    cumulative_p = max(0.0, cumulative_p)
    active_mass, rank, nullity = _feasible_face_geometry(selected, calibrated)
    allocation = torch.from_numpy(selected.reshape(5, 10)).to(torch.float64)
    raw_tensor = torch.from_numpy(raw_progress).to(torch.float64)
    calibrated_tensor = torch.from_numpy(calibrated_progress).to(torch.float64)
    star_tensor = torch.from_numpy(response_star).to(torch.float64)
    payload = {
        "arm": arm,
        "allocation_sha256": tensor_sha256(allocation),
        "deficit_sha256": tensor_sha256(deficit),
        "raw_response_sha256": tensor_sha256(raw_tensor),
        "calibrated_response_sha256": tensor_sha256(calibrated_tensor),
        "semantic_response_star_sha256": tensor_sha256(star_tensor),
        "eta": calibration.eta,
        "e1_xi": xi,
        "e2_mismatch": mismatch,
        "cumulative_capacity": cumulative_capacity,
        "marginal_capacity": marginal_capacity,
        "cumulative_p": cumulative_p,
        "marginal_p": marginal_p,
        "mass": mass.tolist(),
        "active_mass": list(active_mass),
        "rank": rank,
        "nullity": nullity,
        "semantic_residual": semantic_residual,
        "status": status,
    }
    return SDRTRoutingResult(
        arm,
        allocation,
        deficit.detach().to(torch.float64).cpu(),
        raw_tensor,
        calibrated_tensor,
        star_tensor,
        calibration.eta,
        xi,
        mismatch,
        cumulative_capacity,
        marginal_capacity,
        cumulative_p,
        marginal_p,
        tuple(float(item) for item in mass),
        active_mass,
        rank,
        nullity,
        semantic_residual,
        solver_receipts,
        status,
        canonical_hash(payload),
    )


def p2r5_forbidden_influence_receipt() -> dict[str, Any]:
    payload = {
        "schema": "ode-edit-s05-p2r5-forbidden-influence/v1",
        "clamp_on_decision_influence_count_per_target_microstep": 1,
        "clamp_off_access_count": 0,
        "clamp_off_decision_influence_count": 0,
        "neutral_fallback_count": 0,
        "semantic_debt_input_count": 0,
        "explicit_lag_input_count": 0,
        "remaining_horizon_division_count": 0,
        "rho_trust_strength_decision_count": 0,
        "hard_p_budget_influence_count": 0,
        "functional_p_veto_count": 0,
        "candidate_forward_count": 0,
        "candidate_materialization_count": 0,
        "retry_count": 0,
        "backtracking_count": 0,
        "historical_h_influence_count": 0,
        "sequential_state_count": 0,
        "outer_joint_materialization_count_per_step": 1,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "P2R5_ARMS",
    "P2R5_INSTRUCTION_ID",
    "P2R5_METHOD_ID",
    "P2R5RoutingTechnicalError",
    "SDRTCalibration",
    "SDRTQuadratics",
    "SDRTRoutingResult",
    "build_sdrt_quadratics",
    "clamp_safe_semantic_deficit",
    "p2r5_forbidden_influence_receipt",
    "pooled_nonnegative_realization_calibration",
    "solve_sdrt_routing",
]
