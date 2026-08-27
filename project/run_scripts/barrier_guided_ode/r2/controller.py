"""FP64 two-equality Rayleighian and Euclidean plain controller."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .errors import (
    ActuatorEqualityInfeasible,
    NumericalImplementationBoundary,
    R2ScientificBoundary,
)


def _finite_fp64(value: torch.Tensor, *, name: str, ndim: int) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float64
        or value.ndim != ndim
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise R2ScientificBoundary(f"{name} must be detached finite FP64")
    return value


def _norm(value: torch.Tensor) -> float:
    result = float(torch.linalg.vector_norm(value).detach().cpu().item())
    if not math.isfinite(result):
        raise R2ScientificBoundary("numerical norm is non-finite")
    return result


def _backward_tolerance(*, scale: float, dimension: int, multiplier: float) -> float:
    if not math.isfinite(scale) or scale < 0.0 or dimension <= 0 or multiplier <= 0.0:
        raise R2ScientificBoundary("tolerance inputs are invalid")
    return multiplier * torch.finfo(torch.float64).eps * max(1, dimension) * max(1.0, scale)


@dataclass(frozen=True, slots=True)
class R2NumericalReceipt:
    dimension: int
    equality_count: int
    matrix_rank: int
    equality_rank: int
    dtype: str
    eigenvalues: tuple[float, ...]
    schur_eigenvalues: tuple[float, ...]
    matrix_scale: float
    matrix_rank_tolerance: float
    equality_rank_tolerance: float
    schur_rank_tolerance: float
    range_residual_gradient: float
    range_residual_directions: tuple[float, ...]
    range_tolerance_gradient: float
    range_tolerance_directions: tuple[float, ...]
    equality_residual: float
    equality_tolerance: float
    stationarity_residual: float
    stationarity_tolerance: float
    objective: float

    def __post_init__(self) -> None:
        if self.dtype != "torch.float64" or self.equality_count != 2:
            raise R2ScientificBoundary("R2 controller receipt dtype/equality mismatch")
        if self.dimension < 2 or self.matrix_rank < 0 or self.equality_rank < 0:
            raise R2ScientificBoundary("R2 numerical ranks are invalid")
        scalar_names = (
            "matrix_scale",
            "matrix_rank_tolerance",
            "equality_rank_tolerance",
            "schur_rank_tolerance",
            "range_residual_gradient",
            "range_tolerance_gradient",
            "equality_residual",
            "equality_tolerance",
            "stationarity_residual",
            "stationarity_tolerance",
            "objective",
        )
        if any(not math.isfinite(getattr(self, name)) for name in scalar_names):
            raise R2ScientificBoundary("R2 numerical receipt is non-finite")

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class R2RayleighianSolution:
    velocity: torch.Tensor
    multiplier: torch.Tensor
    receipt: R2NumericalReceipt

    def __post_init__(self) -> None:
        if (
            self.velocity.dtype != torch.float64
            or self.velocity.ndim != 1
            or self.velocity.requires_grad
            or not bool(torch.isfinite(self.velocity).all())
            or self.multiplier.dtype != torch.float64
            or self.multiplier.shape != (2,)
            or not bool(torch.isfinite(self.multiplier).all())
        ):
            raise R2ScientificBoundary("R2 solution tensors are invalid")

    def finite_step_fp64(self, step_size: float) -> torch.Tensor:
        if isinstance(step_size, bool) or not isinstance(step_size, (int, float)):
            raise R2ScientificBoundary("step size must be numeric")
        value = float(step_size)
        if not math.isfinite(value) or value <= 0.0:
            raise R2ScientificBoundary("step size must be positive and finite")
        result = self.velocity * value
        if not bool(torch.isfinite(result).all()):
            raise R2ScientificBoundary("finite coefficient is non-finite")
        return result.detach().contiguous()

    def physical_coefficient_fp32(self, step_size: float) -> torch.Tensor:
        """Cast exactly once at the physical assignment boundary."""

        return self.finite_step_fp64(step_size).to(dtype=torch.float32).contiguous()


@dataclass(frozen=True, slots=True)
class PlainMinimumNormSolution:
    velocity: torch.Tensor
    singular_values: tuple[float, ...]
    rank: int
    rank_tolerance: float
    equality_residual: float
    equality_tolerance: float

    def finite_step_fp64(self, step_size: float) -> torch.Tensor:
        if not math.isfinite(float(step_size)) or float(step_size) <= 0.0:
            raise R2ScientificBoundary("step size must be positive and finite")
        return (self.velocity * float(step_size)).detach().contiguous()


def _base_receipt(
    *,
    dimension: int,
    matrix_rank: int,
    equality_rank: int,
    eigenvalues: torch.Tensor,
    schur_eigenvalues: torch.Tensor,
    matrix_scale: float,
    matrix_tolerance: float,
    equality_tolerance: float,
    schur_tolerance: float,
    range_g: float,
    range_a: tuple[float, ...],
    limit_g: float,
    limit_a: tuple[float, ...],
    equality_residual: float = 0.0,
    equality_limit: float = 0.0,
    stationarity_residual: float = 0.0,
    stationarity_limit: float = 0.0,
    objective: float = 0.0,
) -> R2NumericalReceipt:
    return R2NumericalReceipt(
        dimension=dimension,
        equality_count=2,
        matrix_rank=matrix_rank,
        equality_rank=equality_rank,
        dtype="torch.float64",
        eigenvalues=tuple(float(value) for value in eigenvalues.detach().cpu().tolist()),
        schur_eigenvalues=tuple(float(value) for value in schur_eigenvalues.detach().cpu().tolist()),
        matrix_scale=matrix_scale,
        matrix_rank_tolerance=matrix_tolerance,
        equality_rank_tolerance=equality_tolerance,
        schur_rank_tolerance=schur_tolerance,
        range_residual_gradient=range_g,
        range_residual_directions=range_a,
        range_tolerance_gradient=limit_g,
        range_tolerance_directions=limit_a,
        equality_residual=equality_residual,
        equality_tolerance=equality_limit,
        stationarity_residual=stationarity_residual,
        stationarity_tolerance=stationarity_limit,
        objective=objective,
    )


def solve_two_equality_rayleighian(
    fisher: torch.Tensor,
    gradient: torch.Tensor,
    equality_directions: torch.Tensor,
    equality_rates: torch.Tensor,
) -> R2RayleighianSolution:
    matrix = _finite_fp64(fisher, name="fisher", ndim=2)
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise R2ScientificBoundary("fisher must be square with dimension at least two")
    dimension = int(matrix.shape[0])
    vector = _finite_fp64(gradient, name="gradient", ndim=1)
    directions = _finite_fp64(equality_directions, name="equality_directions", ndim=2)
    rates = _finite_fp64(equality_rates, name="equality_rates", ndim=1)
    if vector.shape != (dimension,) or directions.shape != (dimension, 2) or rates.tolist() != [1.0, 0.0]:
        raise R2ScientificBoundary("R2 controller requires A[L,2] and d=[1,0]")

    symmetric = 0.5 * (matrix + matrix.transpose(0, 1))
    matrix_scale = float(torch.linalg.matrix_norm(symmetric).cpu().item())
    symmetry_residual = _norm(matrix - matrix.transpose(0, 1))
    symmetry_limit = _backward_tolerance(scale=matrix_scale, dimension=dimension, multiplier=64.0)
    empty = torch.empty(0, dtype=torch.float64)
    if symmetry_residual > symmetry_limit:
        receipt = _base_receipt(
            dimension=dimension,
            matrix_rank=0,
            equality_rank=0,
            eigenvalues=empty,
            schur_eigenvalues=empty,
            matrix_scale=matrix_scale,
            matrix_tolerance=symmetry_limit,
            equality_tolerance=symmetry_limit,
            schur_tolerance=symmetry_limit,
            range_g=0.0,
            range_a=(0.0, 0.0),
            limit_g=0.0,
            limit_a=(0.0, 0.0),
        )
        raise NumericalImplementationBoundary("fisher symmetry check failed", receipt=receipt.as_payload())

    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    spectral_scale = float(torch.max(torch.abs(eigenvalues)).cpu().item())
    matrix_tolerance = _backward_tolerance(
        scale=spectral_scale,
        dimension=dimension,
        multiplier=64.0,
    )
    if float(eigenvalues.min().cpu().item()) < -matrix_tolerance:
        receipt = _base_receipt(
            dimension=dimension,
            matrix_rank=0,
            equality_rank=0,
            eigenvalues=eigenvalues,
            schur_eigenvalues=empty,
            matrix_scale=spectral_scale,
            matrix_tolerance=matrix_tolerance,
            equality_tolerance=matrix_tolerance,
            schur_tolerance=matrix_tolerance,
            range_g=0.0,
            range_a=(0.0, 0.0),
            limit_g=0.0,
            limit_a=(0.0, 0.0),
        )
        raise NumericalImplementationBoundary("fisher is not PSD in FP64", receipt=receipt.as_payload())
    keep = eigenvalues > matrix_tolerance
    matrix_rank = int(keep.sum().cpu().item())
    if matrix_rank <= 0:
        receipt = _base_receipt(
            dimension=dimension,
            matrix_rank=0,
            equality_rank=0,
            eigenvalues=eigenvalues,
            schur_eigenvalues=empty,
            matrix_scale=spectral_scale,
            matrix_tolerance=matrix_tolerance,
            equality_tolerance=matrix_tolerance,
            schur_tolerance=matrix_tolerance,
            range_g=_norm(vector),
            range_a=tuple(_norm(directions[:, index]) for index in range(2)),
            limit_g=matrix_tolerance,
            limit_a=(matrix_tolerance, matrix_tolerance),
        )
        raise NumericalImplementationBoundary("fisher numerical rank is zero", receipt=receipt.as_payload())
    basis = eigenvectors[:, keep]
    inverse_eigenvalues = 1.0 / eigenvalues[keep]
    inverse = (basis * inverse_eigenvalues[None, :]) @ basis.transpose(0, 1)
    projector = basis @ basis.transpose(0, 1)

    range_g = _norm(vector - projector @ vector)
    range_a = tuple(_norm(directions[:, index] - projector @ directions[:, index]) for index in range(2))
    projected_g = inverse @ vector
    limit_g = _backward_tolerance(
        scale=max(1.0, _norm(vector), matrix_scale * _norm(projected_g)),
        dimension=dimension,
        multiplier=256.0,
    )
    limit_a = tuple(
        _backward_tolerance(
            scale=max(1.0, _norm(directions[:, index]), matrix_scale * _norm(inverse @ directions[:, index])),
            dimension=dimension,
            multiplier=256.0,
        )
        for index in range(2)
    )

    singular_values = torch.linalg.svdvals(directions)
    equality_scale = float(singular_values.max().cpu().item())
    equality_tolerance = _backward_tolerance(
        scale=equality_scale,
        dimension=max(directions.shape),
        multiplier=64.0,
    )
    equality_rank = int((singular_values > equality_tolerance).sum().cpu().item())
    if range_g > limit_g or any(value > limit for value, limit in zip(range_a, limit_a, strict=True)):
        receipt = _base_receipt(
            dimension=dimension,
            matrix_rank=matrix_rank,
            equality_rank=equality_rank,
            eigenvalues=eigenvalues,
            schur_eigenvalues=empty,
            matrix_scale=spectral_scale,
            matrix_tolerance=matrix_tolerance,
            equality_tolerance=equality_tolerance,
            schur_tolerance=matrix_tolerance,
            range_g=range_g,
            range_a=range_a,
            limit_g=limit_g,
            limit_a=limit_a,
        )
        raise NumericalImplementationBoundary(
            "gradient/equality direction lies outside FP64 range(G)",
            receipt=receipt.as_payload(),
        )
    if equality_rank != 2:
        receipt = _base_receipt(
            dimension=dimension,
            matrix_rank=matrix_rank,
            equality_rank=equality_rank,
            eigenvalues=eigenvalues,
            schur_eigenvalues=empty,
            matrix_scale=spectral_scale,
            matrix_tolerance=matrix_tolerance,
            equality_tolerance=equality_tolerance,
            schur_tolerance=matrix_tolerance,
            range_g=range_g,
            range_a=range_a,
            limit_g=limit_g,
            limit_a=limit_a,
        )
        raise ActuatorEqualityInfeasible("two equality directions are rank deficient", receipt=receipt.as_payload())

    unconstrained = -(inverse @ vector)
    schur_raw = directions.transpose(0, 1) @ inverse @ directions
    schur = 0.5 * (schur_raw + schur_raw.transpose(0, 1))
    schur_eigenvalues, schur_eigenvectors = torch.linalg.eigh(schur)
    schur_scale = float(torch.max(torch.abs(schur_eigenvalues)).cpu().item())
    schur_tolerance = _backward_tolerance(
        scale=schur_scale,
        dimension=2,
        multiplier=64.0,
    )
    schur_keep = schur_eigenvalues > schur_tolerance
    if int(schur_keep.sum().cpu().item()) != 2:
        receipt = _base_receipt(
            dimension=dimension,
            matrix_rank=matrix_rank,
            equality_rank=equality_rank,
            eigenvalues=eigenvalues,
            schur_eigenvalues=schur_eigenvalues,
            matrix_scale=spectral_scale,
            matrix_tolerance=matrix_tolerance,
            equality_tolerance=equality_tolerance,
            schur_tolerance=schur_tolerance,
            range_g=range_g,
            range_a=range_a,
            limit_g=limit_g,
            limit_a=limit_a,
        )
        raise ActuatorEqualityInfeasible("two-equality Schur matrix is rank deficient", receipt=receipt.as_payload())
    schur_inverse = (
        schur_eigenvectors[:, schur_keep]
        * (1.0 / schur_eigenvalues[schur_keep])[None, :]
    ) @ schur_eigenvectors[:, schur_keep].transpose(0, 1)
    mismatch = rates - directions.transpose(0, 1) @ unconstrained
    multiplier = schur_inverse @ mismatch
    velocity = unconstrained + inverse @ directions @ multiplier

    equality_residual = _norm(directions.transpose(0, 1) @ velocity - rates)
    stationarity_residual = _norm(symmetric @ velocity + vector - directions @ multiplier)
    equality_limit = _backward_tolerance(
        scale=max(1.0, _norm(rates), _norm(directions) * _norm(velocity)),
        dimension=dimension,
        multiplier=512.0,
    )
    stationarity_limit = _backward_tolerance(
        scale=max(1.0, _norm(vector), matrix_scale * _norm(velocity), _norm(directions) * _norm(multiplier)),
        dimension=dimension,
        multiplier=512.0,
    )
    objective_tensor = torch.dot(vector, velocity) + 0.5 * torch.dot(velocity, symmetric @ velocity)
    objective = float(objective_tensor.cpu().item())
    receipt = _base_receipt(
        dimension=dimension,
        matrix_rank=matrix_rank,
        equality_rank=equality_rank,
        eigenvalues=eigenvalues,
        schur_eigenvalues=schur_eigenvalues,
        matrix_scale=spectral_scale,
        matrix_tolerance=matrix_tolerance,
        equality_tolerance=equality_tolerance,
        schur_tolerance=schur_tolerance,
        range_g=range_g,
        range_a=range_a,
        limit_g=limit_g,
        limit_a=limit_a,
        equality_residual=equality_residual,
        equality_limit=equality_limit,
        stationarity_residual=stationarity_residual,
        stationarity_limit=stationarity_limit,
        objective=objective,
    )
    if not math.isfinite(objective) or equality_residual > equality_limit or stationarity_residual > stationarity_limit:
        raise NumericalImplementationBoundary("FP64 KKT residual failed", receipt=receipt.as_payload())
    return R2RayleighianSolution(
        velocity=velocity.detach().contiguous(),
        multiplier=multiplier.detach().contiguous(),
        receipt=receipt,
    )


def solve_plain_minimum_norm(
    equality_directions: torch.Tensor,
    equality_rates: torch.Tensor,
) -> PlainMinimumNormSolution:
    directions = _finite_fp64(equality_directions, name="equality_directions", ndim=2)
    rates = _finite_fp64(equality_rates, name="equality_rates", ndim=1)
    if directions.shape[1] != 2 or directions.shape[0] < 2 or rates.tolist() != [1.0, 0.0]:
        raise R2ScientificBoundary("Plain requires A[L,2] and d=[1,0]")
    left, singular_values, right_t = torch.linalg.svd(directions, full_matrices=False)
    scale = float(singular_values.max().cpu().item())
    tolerance = _backward_tolerance(
        scale=scale,
        dimension=max(directions.shape),
        multiplier=64.0,
    )
    rank = int((singular_values > tolerance).sum().cpu().item())
    if rank != 2:
        raise ActuatorEqualityInfeasible(
            "Plain two-equality matrix is rank deficient",
            receipt={
                "dtype": "torch.float64",
                "singular_values": [float(value) for value in singular_values.cpu().tolist()],
                "rank": rank,
                "rank_tolerance": tolerance,
            },
        )
    velocity = left @ ((right_t @ rates) / singular_values)
    residual = _norm(directions.transpose(0, 1) @ velocity - rates)
    limit = _backward_tolerance(
        scale=max(1.0, _norm(rates), _norm(directions) * _norm(velocity)),
        dimension=directions.shape[0],
        multiplier=512.0,
    )
    if residual > limit:
        raise NumericalImplementationBoundary(
            "Plain FP64 equality residual failed",
            receipt={
                "dtype": "torch.float64",
                "singular_values": [float(value) for value in singular_values.cpu().tolist()],
                "rank": rank,
                "rank_tolerance": tolerance,
                "equality_residual": residual,
                "equality_tolerance": limit,
            },
        )
    return PlainMinimumNormSolution(
        velocity=velocity.detach().contiguous(),
        singular_values=tuple(float(value) for value in singular_values.cpu().tolist()),
        rank=rank,
        rank_tolerance=tolerance,
        equality_residual=residual,
        equality_tolerance=limit,
    )


__all__ = [
    "PlainMinimumNormSolution",
    "R2NumericalReceipt",
    "R2RayleighianSolution",
    "solve_plain_minimum_norm",
    "solve_two_equality_rayleighian",
]
