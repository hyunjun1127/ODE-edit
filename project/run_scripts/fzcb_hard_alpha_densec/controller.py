"""Matrix-free fixed-z equality and completion-barrier controller."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .contracts import FailureLabel, MethodBoundary, NumericalLock, TrajectoryBoundary
from .matrix_free import LSQRResult, LinearOperator, lsqr, normal_range_operator


@dataclass(frozen=True)
class MinimumAction:
    coefficient: torch.Tensor
    multiplier: torch.Tensor
    action: float
    range_receipt: LSQRResult
    multiplier_receipt: LSQRResult


@dataclass(frozen=True)
class SuffixValue:
    displacement: torch.Tensor
    multiplier: torch.Tensor
    value: float
    remaining: float
    terminal: bool
    range_receipt: LSQRResult | None


@dataclass(frozen=True)
class BarrierCorrection:
    alpha: float
    psi0: float
    guide_norm_sq: float
    psi_min: float
    predicted: float
    active: bool


def _solve_range(operator: LinearOperator, rhs: torch.Tensor, lock: NumericalLock) -> MinimumAction:
    primal = lsqr(
        operator,
        rhs,
        relative_tolerance=lock.fp64_relative_tolerance,
        max_iterations=lock.lsqr_max_iterations,
    )
    if not primal.converged:
        raise MethodBoundary(FailureLabel.ALPHA_TANGENT_RANGE_INFEASIBLE.value)
    normal = normal_range_operator(operator)
    dual = lsqr(
        normal,
        rhs,
        relative_tolerance=lock.fp64_relative_tolerance,
        max_iterations=lock.lsqr_max_iterations,
    )
    if not dual.converged:
        raise MethodBoundary(FailureLabel.ALPHA_TANGENT_RANGE_INFEASIBLE.value)
    multiplier = dual.solution
    dual_match = operator.adjoint(multiplier)
    denominator = torch.linalg.vector_norm(primal.solution).clamp_min(torch.finfo(operator.dtype).tiny)
    if float(torch.linalg.vector_norm(dual_match - primal.solution).div(denominator).item()) > lock.fp64_relative_tolerance * 8:
        raise MethodBoundary(FailureLabel.ALPHA_TANGENT_RANGE_INFEASIBLE.value)
    action = 0.5 * float(torch.dot(primal.solution.reshape(-1), primal.solution.reshape(-1)).item())
    return MinimumAction(primal.solution, multiplier, action, primal, dual)


def minimum_action(operator: LinearOperator, rhs: torch.Tensor, lock: NumericalLock) -> MinimumAction:
    return _solve_range(operator, rhs, lock)


def suffix_value(
    operator: LinearOperator,
    residual: torch.Tensor,
    remaining: float,
    lock: NumericalLock,
    *,
    terminal_relative_residual: float | None = None,
) -> SuffixValue:
    if remaining < 0:
        raise TrajectoryBoundary("negative remaining homotopy time")
    if remaining == 0:
        # Algebraically certified s->1 limit.  No epsilon denominator and no
        # endpoint forcing are introduced.
        closure = float(torch.linalg.vector_norm(residual).item())
        tolerance = 0.0 if terminal_relative_residual is None else terminal_relative_residual
        if closure <= tolerance:
            zero = torch.zeros(operator.domain_shape, dtype=operator.dtype, device=operator.device)
            multiplier = torch.zeros(operator.range_shape, dtype=operator.dtype, device=operator.device)
            return SuffixValue(zero, multiplier, 0.0, 0.0, True, None)
        raise TrajectoryBoundary(FailureLabel.TERMINAL_FIXED_Z_FAILURE.value)
    try:
        solved = _solve_range(operator, residual, lock)
    except MethodBoundary as error:
        raise MethodBoundary(FailureLabel.ALPHA_SUFFIX_RANGE_INFEASIBLE.value) from error
    return SuffixValue(
        solved.coefficient,
        solved.multiplier / remaining,
        solved.action / remaining,
        remaining,
        False,
        solved.range_receipt,
    )


def null_projection(operator: LinearOperator, vector: torch.Tensor, lock: NumericalLock) -> tuple[torch.Tensor, LSQRResult]:
    rhs = operator.forward(vector)
    normal = normal_range_operator(operator)
    solve = lsqr(
        normal,
        rhs,
        relative_tolerance=lock.fp64_relative_tolerance,
        max_iterations=lock.lsqr_max_iterations,
    )
    if not solve.converged:
        raise MethodBoundary(FailureLabel.ALPHA_TANGENT_RANGE_INFEASIBLE.value)
    free = vector - operator.adjoint(solve.solution)
    return free, solve


def analytic_barrier_correction(
    *,
    psi0: float,
    guide_norm_sq: float,
    cbf_tolerance: float,
    guide_tolerance: float,
) -> BarrierCorrection:
    if psi0 <= cbf_tolerance:
        return BarrierCorrection(0.0, psi0, guide_norm_sq, psi0, psi0, False)
    if guide_norm_sq <= guide_tolerance:
        raise MethodBoundary(FailureLabel.ALPHA_ZERO_GUIDE_AUTHORITY.value)
    psi_min = psi0 - 0.5 * guide_norm_sq
    if psi_min > cbf_tolerance:
        raise MethodBoundary(FailureLabel.FZCB_LOCAL_CBF_INFEASIBLE.value)
    discriminant = 1.0 - 2.0 * (psi0 - cbf_tolerance) / guide_norm_sq
    if discriminant < -guide_tolerance:
        raise MethodBoundary(FailureLabel.FZCB_LOCAL_CBF_INFEASIBLE.value)
    discriminant = max(0.0, discriminant)
    alpha = 1.0 - math.sqrt(discriminant)
    predicted = psi0 - alpha * guide_norm_sq + 0.5 * alpha * alpha * guide_norm_sq
    return BarrierCorrection(alpha, psi0, guide_norm_sq, psi_min, predicted, True)


def envelope_directional_derivative(
    *,
    displacement: torch.Tensor,
    metric_apply: callable,
    metric_direction_apply: callable,
    multiplier: torch.Tensor,
    rhs_direction: torch.Tensor,
    operator_direction_displacement: torch.Tensor,
    remaining: float,
    remaining_direction: float,
) -> float:
    if remaining <= 0:
        raise TrajectoryBoundary("envelope derivative is undefined at terminal and must not be called")
    q = displacement.reshape(-1)
    hq = metric_apply(displacement).reshape(-1)
    dhq = metric_direction_apply(displacement).reshape(-1)
    metric_term = 0.5 * torch.dot(q, dhq) / remaining
    constraint_term = torch.dot(
        multiplier.reshape(-1),
        (rhs_direction - operator_direction_displacement).reshape(-1),
    )
    time_term = -0.5 * torch.dot(q, hq) * remaining_direction / (remaining * remaining)
    return float((metric_term + constraint_term + time_term).item())
