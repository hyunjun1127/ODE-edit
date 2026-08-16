"""Certified dense convex QP/QCQP backend for the P2R6 semantic region.

The backend works in the authoritative physical ``alpha`` coordinates.  It
uses a primal-dual predictor-corrector method over the complete inequality set,
so constraints are never frozen into a monotone active set.  The returned
candidate is accepted only after an independent FP64 KKT reconstruction in the
original, unscaled coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from .contracts import ODEBFContractError, canonical_hash


P2R6_QP_EXTERNAL_TOLERANCE = 1.0e-8
P2R6_QP_INTERNAL_TOLERANCE = 1.0e-11
P2R6_QP_MAX_ITERATIONS = 300
P2R6_QP_FRACTION_TO_BOUNDARY = 0.995


class P2R6CertifiedQPError(ODEBFContractError):
    """Typed fail-close error carrying a raw-free numerical receipt."""

    def __init__(self, message: str, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.raw_free_receipt = dict(receipt)


@dataclass(frozen=True, slots=True)
class ConvexQuadraticTie:
    """One optional convex quadratic inequality ``x'Qx + 2c'x <= limit``."""

    matrix: np.ndarray
    cross: np.ndarray
    limit: float
    name: str = "structural_p_tie"


@dataclass(frozen=True, slots=True)
class CertifiedConvexQPResult:
    value: np.ndarray
    dual: np.ndarray
    ordered_slack: np.ndarray
    receipt: Mapping[str, Any]


def _symmetric(value: np.ndarray) -> np.ndarray:
    return 0.5 * (value + value.T)


def _maximum_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(value))) if value.size else 0.0


def _positive_step(value: np.ndarray, direction: np.ndarray) -> float:
    negative = direction < 0.0
    if not np.any(negative):
        return 1.0
    return min(1.0, float(np.min(-value[negative] / direction[negative])))


def _raw_geometry(
    value: np.ndarray,
    linear_matrix: np.ndarray,
    linear_upper: np.ndarray,
    tie: ConvexQuadraticTie | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    constraint = linear_matrix @ value - linear_upper
    jacobian = linear_matrix.copy()
    tie_hessian = None
    if tie is not None:
        matrix = _symmetric(np.asarray(tie.matrix, dtype=np.float64))
        cross = np.asarray(tie.cross, dtype=np.float64)
        constraint = np.concatenate(
            (
                constraint,
                np.asarray(
                    [value @ matrix @ value + 2.0 * cross @ value - tie.limit],
                    dtype=np.float64,
                ),
            )
        )
        jacobian = np.concatenate(
            (jacobian, ((2.0 * matrix @ value + 2.0 * cross)[None, :])), axis=0
        )
        tie_hessian = 2.0 * matrix
    return constraint, jacobian, tie_hessian


def _ordered_names(alpha_count: int, request_count: int, tie: ConvexQuadraticTie | None) -> list[str]:
    names = [f"alpha_nonnegative[{index}]" for index in range(alpha_count)]
    names.extend(f"request_mass_upper[{index}]" for index in range(request_count))
    names.extend(f"semantic_response_lower[{index}]" for index in range(request_count))
    if tie is not None:
        names.append(tie.name)
    return names


def solve_certified_semantic_region_qp(
    start: np.ndarray,
    objective_matrix: np.ndarray,
    objective_cross: np.ndarray,
    response: np.ndarray,
    lower: np.ndarray,
    mass_matrix: np.ndarray,
    *,
    stage: str,
    tie: ConvexQuadraticTie | None = None,
) -> CertifiedConvexQPResult:
    """Solve and independently certify one P2R6 convex QP/QCQP."""

    x = np.asarray(start, dtype=np.float64).copy()
    objective = _symmetric(np.asarray(objective_matrix, dtype=np.float64))
    cross = np.asarray(objective_cross, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    mass = np.asarray(mass_matrix, dtype=np.float64)
    alpha_count = x.size
    request_count = lower.size
    if (
        objective.shape != (alpha_count, alpha_count)
        or cross.shape != (alpha_count,)
        or response.shape != (request_count, alpha_count)
        or mass.shape != (request_count, alpha_count)
    ):
        raise ODEBFContractError("P2R6 certified QP geometry differs")
    finite_inputs = (objective, cross, response, lower, mass, x)
    if not all(np.all(np.isfinite(item)) for item in finite_inputs):
        raise ODEBFContractError("P2R6 certified QP input is nonfinite")
    objective_eigenvalues = np.linalg.eigvalsh(objective)
    if float(np.min(objective_eigenvalues)) < -P2R6_QP_EXTERNAL_TOLERANCE:
        raise P2R6CertifiedQPError(
            "P2R6 certified QP objective is nonconvex",
            {
                "schema": "ode-edit-s05-p2r6-certified-convex-qp/v1",
                "stage": stage,
                "objective_min_eigenvalue": float(np.min(objective_eigenvalues)),
                "certificate_pass": False,
            },
        )
    if tie is not None:
        tie_matrix = _symmetric(np.asarray(tie.matrix, dtype=np.float64))
        tie_cross = np.asarray(tie.cross, dtype=np.float64)
        tie_eigenvalues = np.linalg.eigvalsh(tie_matrix)
        if tie_matrix.shape != objective.shape or tie_cross.shape != cross.shape:
            raise ODEBFContractError("P2R6 quadratic tie geometry differs")
        if not math.isfinite(float(tie.limit)) or not np.all(np.isfinite(tie_matrix)) or not np.all(np.isfinite(tie_cross)):
            raise ODEBFContractError("P2R6 quadratic tie is nonfinite")
        if float(np.min(tie_eigenvalues)) < -P2R6_QP_EXTERNAL_TOLERANCE:
            raise P2R6CertifiedQPError(
                "P2R6 quadratic tie is nonconvex",
                {
                    "schema": "ode-edit-s05-p2r6-certified-convex-qp/v1",
                    "stage": stage,
                    "tie_min_eigenvalue": float(np.min(tie_eigenvalues)),
                    "certificate_pass": False,
                },
            )
    else:
        tie_eigenvalues = np.asarray([], dtype=np.float64)

    linear_matrix = np.concatenate((-np.eye(alpha_count), mass, -response), axis=0)
    scientific_linear_upper = np.concatenate(
        (np.zeros(alpha_count), np.ones(request_count), -lower), axis=0
    )
    # Solve the scientific inequalities themselves.  The external tolerance
    # is a certificate threshold, never a feasible-set translation.
    backend_envelope = 0.0
    linear_upper = scientific_linear_upper
    # Positive row scaling changes only the numerical coordinate of each
    # inequality.  Certificates and multipliers below are reconstructed in the
    # original coordinates.
    row_scale = np.maximum(
        1.0,
        np.maximum(np.linalg.norm(linear_matrix, axis=1), np.abs(linear_upper)),
    )
    if tie is not None:
        tie_scale = max(
            1.0,
            abs(float(tie.limit)),
            float(np.linalg.norm(tie.matrix, ord="fro")),
            float(np.linalg.norm(tie.cross)),
        )
        full_scale = np.concatenate((row_scale, np.asarray([tie_scale])))
    else:
        full_scale = row_scale

    hessian = 2.0 * objective
    gradient_constant = 2.0 * cross

    def scaled_geometry(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        raw_constraint, raw_jacobian, raw_tie_hessian = _raw_geometry(
            value, linear_matrix, linear_upper, tie
        )
        scaled_tie_hessian = (
            raw_tie_hessian / full_scale[-1]
            if raw_tie_hessian is not None
            else None
        )
        return (
            raw_constraint / full_scale,
            raw_jacobian / full_scale[:, None],
            scaled_tie_hessian,
        )

    constraint, jacobian, _ = scaled_geometry(x)
    # E1 supplies a certified feasible start.  Preserve its natural slacks so
    # the Newton system does not manufacture an O(1) infeasibility on a narrow
    # semantic face.
    slack = np.maximum(-constraint, P2R6_QP_INTERNAL_TOLERANCE)
    dual_scaled = np.ones_like(slack)
    iterations = 0
    linear_solve_count = 0
    status = "MAX_ITERATIONS"

    def direction(
        value: np.ndarray,
        local_slack: np.ndarray,
        local_dual: np.ndarray,
        residual_primal: np.ndarray,
        residual_dual: np.ndarray,
        residual_center: np.ndarray,
        local_jacobian: np.ndarray,
        local_tie_hessian: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal linear_solve_count
        lagrangian_hessian = hessian.copy()
        if local_tie_hessian is not None:
            lagrangian_hessian += local_dual[-1] * local_tie_hessian
        constraint_count = local_slack.size
        zero_nm = np.zeros((value.size, constraint_count), dtype=np.float64)
        zero_mm = np.zeros((constraint_count, constraint_count), dtype=np.float64)
        identity_mm = np.eye(constraint_count, dtype=np.float64)
        kkt = np.block(
            [
                [lagrangian_hessian, local_jacobian.T, zero_nm],
                [local_jacobian, zero_mm, identity_mm],
                [zero_nm.T, np.diag(local_slack), np.diag(local_dual)],
            ]
        )
        right = np.concatenate((-residual_dual, -residual_primal, -residual_center))
        try:
            solved_direction = np.linalg.solve(kkt, right)
        except np.linalg.LinAlgError as exc:
            raise P2R6CertifiedQPError(
                "P2R6 certified QP Newton system is singular",
                {
                    "schema": "ode-edit-s05-p2r6-certified-convex-qp/v1",
                    "stage": stage,
                    "iteration": iterations,
                    "kkt_rank": int(np.linalg.matrix_rank(kkt)),
                    "exception_class": type(exc).__name__,
                    "certificate_pass": False,
                },
            ) from exc
        linear_solve_count += 1
        delta_value = solved_direction[: value.size]
        delta_dual = solved_direction[
            value.size : value.size + constraint_count
        ]
        delta_slack = solved_direction[value.size + constraint_count :]
        return delta_value, delta_slack, delta_dual

    for iterations in range(1, P2R6_QP_MAX_ITERATIONS + 1):
        constraint, jacobian, scaled_tie_hessian = scaled_geometry(x)
        residual_primal = constraint + slack
        residual_dual = hessian @ x + gradient_constant + jacobian.T @ dual_scaled
        complementarity = slack * dual_scaled
        if max(
            _maximum_abs(residual_primal),
            _maximum_abs(residual_dual),
            _maximum_abs(complementarity),
        ) <= P2R6_QP_INTERNAL_TOLERANCE:
            status = "CONVERGED"
            break
        mu = float(np.dot(slack, dual_scaled) / slack.size)
        affine = direction(
            x,
            slack,
            dual_scaled,
            residual_primal,
            residual_dual,
            complementarity,
            jacobian,
            scaled_tie_hessian,
        )
        _, affine_slack, affine_dual = affine
        affine_primal_step = _positive_step(slack, affine_slack)
        affine_dual_step = _positive_step(dual_scaled, affine_dual)
        mu_affine = float(
            np.dot(
                slack + affine_primal_step * affine_slack,
                dual_scaled + affine_dual_step * affine_dual,
            )
            / slack.size
        )
        sigma = min(1.0, max(0.0, (mu_affine / max(mu, 1.0e-300)) ** 3))
        corrected_center = (
            complementarity + affine_slack * affine_dual - sigma * mu
        )
        delta_value, delta_slack, delta_dual = direction(
            x,
            slack,
            dual_scaled,
            residual_primal,
            residual_dual,
            corrected_center,
            jacobian,
            scaled_tie_hessian,
        )
        primal_step = min(
            1.0,
            P2R6_QP_FRACTION_TO_BOUNDARY * _positive_step(slack, delta_slack),
        )
        dual_step = min(
            1.0,
            P2R6_QP_FRACTION_TO_BOUNDARY
            * _positive_step(dual_scaled, delta_dual),
        )
        x = x + primal_step * delta_value
        slack = slack + primal_step * delta_slack
        dual_scaled = dual_scaled + dual_step * delta_dual
        if not (
            np.all(np.isfinite(x))
            and np.all(np.isfinite(slack))
            and np.all(np.isfinite(dual_scaled))
            and np.all(slack > 0.0)
            and np.all(dual_scaled > 0.0)
        ):
            status = "NONFINITE_OR_NONPOSITIVE_ITERATE"
            break

    solver_constraint, _solver_jacobian, _ = _raw_geometry(
        x, linear_matrix, linear_upper, tie
    )
    final_scaled_constraint, final_scaled_jacobian, _ = scaled_geometry(x)
    final_internal_r_pri = _maximum_abs(final_scaled_constraint + slack)
    final_internal_r_dual = _maximum_abs(
        hessian @ x + gradient_constant + final_scaled_jacobian.T @ dual_scaled
    )
    final_internal_r_comp = _maximum_abs(slack * dual_scaled)
    raw_constraint, raw_jacobian, _ = _raw_geometry(
        x, linear_matrix, scientific_linear_upper, tie
    )
    ordered_slack = -raw_constraint
    ordered_certificate_slack = -solver_constraint
    dual = dual_scaled / full_scale
    objective_gradient = hessian @ x + gradient_constant
    stationarity = objective_gradient + raw_jacobian.T @ dual
    r_pri = max(0.0, float(np.max(raw_constraint)))
    r_dual = max(0.0, -float(np.min(dual)))
    r_stat = _maximum_abs(stationarity)
    r_comp = _maximum_abs(dual * ordered_slack)
    finite = bool(
        np.all(np.isfinite(x))
        and np.all(np.isfinite(ordered_slack))
        and np.all(np.isfinite(dual))
    )
    objective_value = float(x @ objective @ x + 2.0 * cross @ x)
    names = _ordered_names(alpha_count, request_count, tie)
    independent_residual_pass = bool(
        finite
        and max(r_pri, r_dual, r_stat, r_comp) <= P2R6_QP_EXTERNAL_TOLERANCE
    )
    internal_residual_pass = bool(
        max(final_internal_r_pri, final_internal_r_dual, final_internal_r_comp)
        <= P2R6_QP_EXTERNAL_TOLERANCE
    )
    certificate_pass = bool(independent_residual_pass and internal_residual_pass)
    if certificate_pass and status != "CONVERGED":
        status = "CONVERGED_BY_INDEPENDENT_FP64_KKT"
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p2r6-certified-convex-qp/v1",
        "stage": stage,
        "backend": "DENSE_FP64_PRIMAL_DUAL_PREDICTOR_CORRECTOR_ORIGINAL_ALPHA",
        "constraint_release_capability": "ALL_INEQUALITIES_REMAIN_PRIMAL_DUAL_VARIABLES",
        "alpha_count": alpha_count,
        "request_count": request_count,
        "constraint_count": len(names),
        "ordered_constraint_names": names,
        "final_alpha": x.tolist(),
        "ordered_slacks": ordered_slack.tolist(),
        "ordered_certificate_slacks": ordered_certificate_slack.tolist(),
        "dual_multipliers": dual.tolist(),
        "objective_value": objective_value,
        "objective_identity_residual": abs(
            objective_value - _quadratic_value_for_receipt(objective, cross, x)
        ),
        "r_pri": r_pri,
        "r_dual": r_dual,
        "r_stat": r_stat,
        "r_comp": r_comp,
        "internal_r_pri": final_internal_r_pri,
        "internal_r_dual": final_internal_r_dual,
        "internal_r_comp": final_internal_r_comp,
        "objective_min_eigenvalue": float(np.min(objective_eigenvalues)),
        "objective_max_eigenvalue": float(np.max(objective_eigenvalues)),
        "objective_psd_certificate": bool(
            float(np.min(objective_eigenvalues)) >= -P2R6_QP_EXTERNAL_TOLERANCE
        ),
        "tie_enabled": tie is not None,
        "tie_min_eigenvalue": (
            float(np.min(tie_eigenvalues)) if tie_eigenvalues.size else 0.0
        ),
        "tie_max_eigenvalue": (
            float(np.max(tie_eigenvalues)) if tie_eigenvalues.size else 0.0
        ),
        "iterations": iterations,
        "linear_solve_count": linear_solve_count,
        "internal_tolerance": P2R6_QP_INTERNAL_TOLERANCE,
        "external_tolerance": P2R6_QP_EXTERNAL_TOLERANCE,
        "backend_constraint_envelope": backend_envelope,
        "solver_status": status,
        "certificate_pass": certificate_pass,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    if not certificate_pass:
        raise P2R6CertifiedQPError(
            f"P2R6 {stage} certified convex QP failed", receipt
        )
    return CertifiedConvexQPResult(x, dual, ordered_slack, receipt)


def _quadratic_value_for_receipt(
    matrix: np.ndarray, cross: np.ndarray, value: np.ndarray
) -> float:
    return float(value @ matrix @ value + 2.0 * cross @ value)


__all__ = [
    "CertifiedConvexQPResult",
    "ConvexQuadraticTie",
    "P2R6CertifiedQPError",
    "P2R6_QP_EXTERNAL_TOLERANCE",
    "solve_certified_semantic_region_qp",
]
