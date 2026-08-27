"""Deterministic factor-space equality controllers for BGODE-R3.

Production authority is the event-score factor itself.  The small Fisher Gram
is returned only as a diagnostic and is never inverted or eigendecomposed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import torch

from .errors import EqualityInfeasible, NumericalBoundary, R3ScientificBoundary


class ControllerArm(str, Enum):
    PLAIN = "PLAIN_NORMALIZED_EUCLIDEAN"
    FISHER = "FISHER_FACTOR_SPACE"
    FULL = "FULL_TARGET_EXCLUDED_BARRIER"


def _finite(value: torch.Tensor, *, name: str, ndim: int) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float64
        or value.ndim != ndim
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise R3ScientificBoundary(f"{name} must be detached finite FP64")
    return value


def _vector_norm(value: torch.Tensor) -> float:
    result = float(torch.linalg.vector_norm(value).cpu().item())
    if not math.isfinite(result):
        raise R3ScientificBoundary("numerical norm is non-finite")
    return result


def _cutoff(shape: tuple[int, int], maximum_singular_value: float) -> tuple[float, float]:
    rcond = max(shape) * torch.finfo(torch.float32).eps
    threshold = rcond * maximum_singular_value
    return float(rcond), float(threshold)


def _residual_bound(*, scale: float, dimension: int) -> float:
    if not math.isfinite(scale) or scale < 0.0 or dimension <= 0:
        raise R3ScientificBoundary("residual bound inputs are invalid")
    return 64.0 * torch.finfo(torch.float32).eps * dimension * (1.0 + scale)


@dataclass(frozen=True, slots=True)
class EqualityFactorization:
    particular: torch.Tensor
    null_basis: torch.Tensor
    raw_singular_values: tuple[float, ...]
    equilibrated_singular_values: tuple[float, ...]
    raw_rank: int
    equilibrated_rank: int
    raw_threshold: float
    equilibrated_threshold: float
    rcond: float
    consistency_residual: float
    consistency_tolerance: float


def factor_equality_system(directions: torch.Tensor, rates: torch.Tensor) -> EqualityFactorization:
    c = _finite(directions, name="equality_directions", ndim=2)
    d = _finite(rates, name="equality_rates", ndim=1)
    if c.shape[1] != 2 or c.shape[0] < 2 or d.shape != (2,) or d.tolist() != [1.0, 0.0]:
        raise R3ScientificBoundary("R3 equality requires C[L,2] and d=[1,0]")
    matrix = c.transpose(0, 1).contiguous()
    column_norms = torch.linalg.vector_norm(matrix, dim=0)
    scales = torch.where(column_norms == 0.0, torch.ones_like(column_norms), column_norms)
    equilibrated = matrix / scales[None, :]
    eq_u, eq_s, eq_vh = torch.linalg.svd(equilibrated, full_matrices=True)
    eq_max = float(eq_s.max().cpu().item()) if eq_s.numel() else 0.0
    rcond, eq_threshold = _cutoff(tuple(equilibrated.shape), eq_max)
    eq_rank = int((eq_s > eq_threshold).sum().cpu().item())
    projected = eq_u[:, :eq_rank] @ (eq_u[:, :eq_rank].transpose(0, 1) @ d)
    consistency = _vector_norm(d - projected)
    consistency_tolerance = _residual_bound(
        scale=_vector_norm(d) + _vector_norm(equilibrated),
        dimension=max(equilibrated.shape),
    )
    preliminary = {
        "dtype": "torch.float64",
        "rcond": rcond,
        "equilibrated_singular_values": [float(value) for value in eq_s.cpu().tolist()],
        "equilibrated_rank": eq_rank,
        "equilibrated_threshold": eq_threshold,
        "consistency_residual": consistency,
        "consistency_tolerance": consistency_tolerance,
    }
    if consistency > consistency_tolerance:
        raise EqualityInfeasible("complete equality system is inconsistent", receipt=preliminary)

    raw_u, raw_s, raw_vh = torch.linalg.svd(matrix, full_matrices=True)
    raw_max = float(raw_s.max().cpu().item()) if raw_s.numel() else 0.0
    _, raw_threshold = _cutoff(tuple(matrix.shape), raw_max)
    raw_rank = int((raw_s > raw_threshold).sum().cpu().item())
    raw_projected = raw_u[:, :raw_rank] @ (raw_u[:, :raw_rank].transpose(0, 1) @ d)
    raw_consistency = _vector_norm(d - raw_projected)
    if raw_consistency > consistency_tolerance:
        raise EqualityInfeasible(
            "raw equality system is inconsistent after equilibrated gate",
            receipt={**preliminary, "raw_consistency_residual": raw_consistency},
        )
    particular = raw_vh[:raw_rank].transpose(0, 1) @ (
        (raw_u[:, :raw_rank].transpose(0, 1) @ d) / raw_s[:raw_rank]
    )
    null_basis = raw_vh[raw_rank:].transpose(0, 1).contiguous()
    return EqualityFactorization(
        particular=particular.detach().contiguous(),
        null_basis=null_basis.detach().contiguous(),
        raw_singular_values=tuple(float(value) for value in raw_s.cpu().tolist()),
        equilibrated_singular_values=tuple(float(value) for value in eq_s.cpu().tolist()),
        raw_rank=raw_rank,
        equilibrated_rank=eq_rank,
        raw_threshold=raw_threshold,
        equilibrated_threshold=eq_threshold,
        rcond=rcond,
        consistency_residual=max(consistency, raw_consistency),
        consistency_tolerance=consistency_tolerance,
    )


@dataclass(frozen=True, slots=True)
class FactorSpaceReceipt:
    arm: str
    dimension: int
    event_count: int
    equality: EqualityFactorization
    reduced_singular_values: tuple[float, ...]
    reduced_rank: int
    reduced_threshold: float
    retained_directions: tuple[tuple[float, ...], ...]
    equality_residual: float
    equality_tolerance: float
    stationarity_residual: float
    stationarity_tolerance: float
    objective: float
    diagnostic_fisher: tuple[tuple[float, ...], ...]

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FactorSpaceSolution:
    velocity: torch.Tensor
    retained_directions: torch.Tensor
    receipt: FactorSpaceReceipt

    def coefficient(self, step_size: float) -> torch.Tensor:
        value = float(step_size)
        if not math.isfinite(value) or value <= 0.0:
            raise R3ScientificBoundary("step size must be positive and finite")
        result = self.velocity * value
        if not bool(torch.isfinite(result).all()):
            raise R3ScientificBoundary("finite coefficient is non-finite")
        return result.detach().contiguous()


def solve_factor_space(
    factor: torch.Tensor,
    objective_offset: torch.Tensor,
    equality_directions: torch.Tensor,
    equality_rates: torch.Tensor,
    *,
    arm: ControllerArm,
) -> FactorSpaceSolution:
    x = _finite(factor, name="event_score_factor", ndim=2)
    y = _finite(objective_offset, name="objective_offset", ndim=1)
    if y.shape != (x.shape[0],):
        raise R3ScientificBoundary("factor and objective event widths differ")
    equality = factor_equality_system(equality_directions, equality_rates)
    up = equality.particular
    z = equality.null_basis
    if up.shape != (x.shape[1],) or z.shape[0] != x.shape[1]:
        raise R3ScientificBoundary("equality and factor actuator widths differ")

    if arm is ControllerArm.PLAIN:
        velocity = up
        reduced_values = torch.empty(0, dtype=torch.float64)
        reduced_rank = 0
        reduced_threshold = 0.0
        retained_directions = torch.empty((x.shape[1], 0), dtype=torch.float64)
        residual_vector = velocity
        stationarity = _vector_norm(z.transpose(0, 1) @ velocity) if z.shape[1] else 0.0
        objective_tensor = 0.5 * torch.dot(velocity, velocity)
    else:
        offset = torch.zeros_like(y) if arm is ControllerArm.FISHER else y
        reduced = x @ z
        right = -(x @ up + offset)
        if z.shape[1] == 0:
            correction = torch.empty(0, dtype=torch.float64)
            reduced_values = torch.empty(0, dtype=torch.float64)
            reduced_rank = 0
            reduced_threshold = 0.0
            retained_directions = torch.empty((x.shape[1], 0), dtype=torch.float64)
        else:
            left_u, reduced_values, right_vh = torch.linalg.svd(reduced, full_matrices=False)
            maximum = float(reduced_values.max().cpu().item()) if reduced_values.numel() else 0.0
            _, reduced_threshold = _cutoff(tuple(reduced.shape), maximum)
            retained = reduced_values > reduced_threshold
            reduced_rank = int(retained.sum().cpu().item())
            retained_directions = z @ right_vh[retained].transpose(0, 1)
            if reduced_rank:
                correction = right_vh[retained].transpose(0, 1) @ (
                    (left_u[:, retained].transpose(0, 1) @ right) / reduced_values[retained]
                )
            else:
                correction = torch.zeros(z.shape[1], dtype=torch.float64)
        velocity = up + z @ correction
        residual_vector = x @ velocity + offset
        stationarity = _vector_norm(z.transpose(0, 1) @ (x.transpose(0, 1) @ residual_vector)) if z.shape[1] else 0.0
        objective_tensor = 0.5 * torch.dot(residual_vector, residual_vector)

    c = _finite(equality_directions, name="equality_directions", ndim=2)
    d = _finite(equality_rates, name="equality_rates", ndim=1)
    equality_residual = _vector_norm(c.transpose(0, 1) @ velocity - d)
    equality_tolerance = _residual_bound(
        scale=_vector_norm(d) + _vector_norm(c) * _vector_norm(velocity),
        dimension=max(c.shape),
    )
    stationarity_tolerance = _residual_bound(
        scale=_vector_norm(residual_vector) * (1.0 + _vector_norm(x)),
        dimension=max(x.shape),
    )
    objective = float(objective_tensor.cpu().item())
    fisher = x.transpose(0, 1) @ x
    receipt = FactorSpaceReceipt(
        arm=arm.value,
        dimension=x.shape[1],
        event_count=x.shape[0],
        equality=equality,
        reduced_singular_values=tuple(float(value) for value in reduced_values.cpu().tolist()),
        reduced_rank=reduced_rank,
        reduced_threshold=reduced_threshold,
        retained_directions=tuple(
            tuple(float(value) for value in row) for row in retained_directions.cpu().tolist()
        ),
        equality_residual=equality_residual,
        equality_tolerance=equality_tolerance,
        stationarity_residual=stationarity,
        stationarity_tolerance=stationarity_tolerance,
        objective=objective,
        diagnostic_fisher=tuple(tuple(float(value) for value in row) for row in fisher.cpu().tolist()),
    )
    if (
        not math.isfinite(objective)
        or equality_residual > equality_tolerance
        or stationarity > stationarity_tolerance
    ):
        raise NumericalBoundary("factor-space residual gate failed", receipt=receipt.payload())
    return FactorSpaceSolution(
        velocity=velocity.detach().contiguous(),
        retained_directions=retained_directions.detach().contiguous(),
        receipt=receipt,
    )


@dataclass(frozen=True, slots=True)
class RetainedDirectionValidation:
    direction_count: int
    maximum_absolute_error: float
    absolute_tolerance: float
    relative_tolerance: float
    allclose: bool


def validate_retained_direction_scores(
    jvp_scores: torch.Tensor,
    symmetric_difference_scores: torch.Tensor,
    retained_directions: torch.Tensor,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> RetainedDirectionValidation:
    jvp = _finite(jvp_scores, name="jvp_scores", ndim=2)
    observed = _finite(symmetric_difference_scores, name="symmetric_difference_scores", ndim=2)
    directions = _finite(retained_directions, name="retained_directions", ndim=2)
    if jvp.shape != observed.shape or directions.shape[0] != jvp.shape[1]:
        raise R3ScientificBoundary("retained-direction validation shapes differ")
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise R3ScientificBoundary("absolute validation tolerance is invalid")
    if not math.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise R3ScientificBoundary("relative validation tolerance is invalid")
    predicted = jvp @ directions
    measured = observed @ directions
    allclose = bool(
        torch.allclose(
            predicted,
            measured,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
    )
    error = _vector_norm(predicted - measured) if predicted.numel() else 0.0
    receipt = RetainedDirectionValidation(
        direction_count=directions.shape[1],
        maximum_absolute_error=(
            float(torch.max(torch.abs(predicted - measured)).cpu().item())
            if predicted.numel()
            else 0.0
        ),
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        allclose=allclose,
    )
    if not allclose:
        raise NumericalBoundary(
            "retained singular direction JVP/symmetric-difference gate failed",
            receipt=asdict(receipt) | {"difference_norm": error},
        )
    return receipt


@dataclass(frozen=True, slots=True)
class BarrierAttribution:
    gradient_dot_fisher: float
    gradient_dot_full: float
    correction_fisher_energy: float
    identity_residual: float
    correction_norm: float


def barrier_attribution(
    factor: torch.Tensor,
    objective_offset: torch.Tensor,
    fisher: FactorSpaceSolution,
    full: FactorSpaceSolution,
) -> BarrierAttribution:
    x = _finite(factor, name="event_score_factor", ndim=2)
    y = _finite(objective_offset, name="objective_offset", ndim=1)
    gradient = x.transpose(0, 1) @ y
    delta = full.velocity - fisher.velocity
    first = float(torch.dot(gradient, fisher.velocity).cpu().item())
    second = float(torch.dot(gradient, full.velocity).cpu().item())
    energy = float(torch.dot(x @ delta, x @ delta).cpu().item())
    residual = abs(second - (first - energy))
    return BarrierAttribution(
        gradient_dot_fisher=first,
        gradient_dot_full=second,
        correction_fisher_energy=energy,
        identity_residual=residual,
        correction_norm=_vector_norm(delta),
    )


__all__ = [
    "BarrierAttribution",
    "ControllerArm",
    "EqualityFactorization",
    "FactorSpaceReceipt",
    "FactorSpaceSolution",
    "RetainedDirectionValidation",
    "barrier_attribution",
    "factor_equality_system",
    "solve_factor_space",
    "validate_retained_direction_scores",
]
