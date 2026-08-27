"""Deterministic singular-aware equality Rayleighian solver for BGODE-R1."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .errors import BGODEScientificBoundary, NumericalRankBoundary
from .telemetry import RayleighianNumericalReceipt


def _finite_fp32(value: torch.Tensor, *, name: str, ndim: int) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.ndim != ndim
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise BGODEScientificBoundary(f"{name} must be a finite detached FP32 tensor")
    return value.detach()


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value).detach().cpu().item())


def _rank_tolerance(scale: float, dimension: int) -> float:
    if not math.isfinite(scale) or scale <= 0.0:
        raise NumericalRankBoundary("matrix scale must be finite and positive")
    return 16.0 * torch.finfo(torch.float32).eps * float(max(1, dimension)) * scale


@dataclass(frozen=True, slots=True)
class RayleighianSolution:
    velocity: torch.Tensor
    lagrange_multiplier: torch.Tensor
    receipt: RayleighianNumericalReceipt

    def __post_init__(self) -> None:
        if (
            not isinstance(self.velocity, torch.Tensor)
            or self.velocity.dtype != torch.float32
            or self.velocity.ndim != 1
            or self.velocity.requires_grad
            or not bool(torch.isfinite(self.velocity).all())
        ):
            raise BGODEScientificBoundary("Rayleighian velocity is invalid")
        if (
            not isinstance(self.lagrange_multiplier, torch.Tensor)
            or self.lagrange_multiplier.dtype != torch.float32
            or self.lagrange_multiplier.ndim != 1
            or self.lagrange_multiplier.requires_grad
            or not bool(torch.isfinite(self.lagrange_multiplier).all())
        ):
            raise BGODEScientificBoundary("Rayleighian multiplier is invalid")

    def finite_step(self, step_size: float) -> torch.Tensor:
        """Return the only authorized finite coefficient, ``beta=h*u``."""

        if isinstance(step_size, bool) or not isinstance(step_size, (int, float)):
            raise BGODEScientificBoundary("step size must be a positive finite scalar")
        h = float(step_size)
        if not math.isfinite(h) or h <= 0.0:
            raise BGODEScientificBoundary("step size must be a positive finite scalar")
        beta = self.velocity * h
        if not bool(torch.isfinite(beta).all()):
            raise BGODEScientificBoundary("finite Rayleighian step is non-finite")
        return beta.detach().contiguous()


def solve_equality_rayleighian(
    fisher: torch.Tensor,
    gradient: torch.Tensor,
    equality_directions: torch.Tensor,
    equality_rates: torch.Tensor | float,
) -> RayleighianSolution:
    """Solve ``min g'u + .5 u'Gu`` subject to ``A'u=d``.

    No damping, ridge, active-set rescue, progress penalty, or rank fallback is
    present.  A singular ``G`` is accepted only when ``g`` and every equality
    direction lie in its numerical range and the equality Schur complement is
    full rank.
    """

    g_matrix = _finite_fp32(fisher, name="fisher", ndim=2)
    if g_matrix.shape[0] != g_matrix.shape[1] or g_matrix.shape[0] == 0:
        raise BGODEScientificBoundary("fisher must be a non-empty square matrix")
    dimension = int(g_matrix.shape[0])
    gradient_vector = _finite_fp32(gradient, name="gradient", ndim=1)
    if gradient_vector.shape != (dimension,):
        raise BGODEScientificBoundary("gradient dimension differs from fisher")
    directions = _finite_fp32(
        equality_directions,
        name="equality_directions",
        ndim=equality_directions.ndim if isinstance(equality_directions, torch.Tensor) else 0,
    )
    if directions.ndim == 1:
        directions = directions[:, None]
    if directions.ndim != 2 or directions.shape[0] != dimension or directions.shape[1] == 0:
        raise BGODEScientificBoundary("equality directions must have shape [L,k]")
    equality_count = int(directions.shape[1])
    if equality_count > dimension:
        raise NumericalRankBoundary("more equalities than actuator dimensions")
    if isinstance(equality_rates, torch.Tensor):
        rates = _finite_fp32(equality_rates, name="equality_rates", ndim=equality_rates.ndim)
        if rates.ndim == 0:
            rates = rates.reshape(1)
    else:
        if isinstance(equality_rates, bool) or not isinstance(equality_rates, (int, float)):
            raise BGODEScientificBoundary("equality rate must be finite")
        rates = torch.tensor([float(equality_rates)], dtype=torch.float32)
    if rates.ndim != 1 or rates.shape != (equality_count,) or not bool(torch.isfinite(rates).all()):
        raise BGODEScientificBoundary("equality rates have the wrong shape")

    symmetry_scale = max(1.0, float(torch.linalg.matrix_norm(g_matrix).cpu().item()))
    symmetry_residual = _norm(g_matrix - g_matrix.transpose(0, 1))
    symmetry_limit = 64.0 * torch.finfo(torch.float32).eps * dimension * symmetry_scale
    if symmetry_residual > symmetry_limit:
        raise NumericalRankBoundary("fisher matrix is not symmetric")
    symmetric = 0.5 * (g_matrix + g_matrix.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    scale = float(torch.max(torch.abs(eigenvalues)).cpu().item())
    tolerance = _rank_tolerance(scale, dimension)
    if float(eigenvalues.min().cpu().item()) < -tolerance:
        raise NumericalRankBoundary("fisher matrix is not PSD")
    keep = eigenvalues > tolerance
    rank = int(keep.sum().cpu().item())
    if rank <= 0:
        raise NumericalRankBoundary("fisher pullback has numerical rank zero")
    basis = eigenvectors[:, keep]
    inverse_values = 1.0 / eigenvalues[keep]
    pinv = (basis * inverse_values[None, :]) @ basis.transpose(0, 1)
    projector = basis @ basis.transpose(0, 1)

    range_g = _norm(gradient_vector - projector @ gradient_vector)
    range_a_by_column = tuple(
        _norm(directions[:, index] - projector @ directions[:, index])
        for index in range(equality_count)
    )
    range_a = max(range_a_by_column)
    eps = torch.finfo(torch.float32).eps
    range_limit_g = 256.0 * eps * dimension * max(1.0, _norm(gradient_vector))
    range_limit_a = max(
        256.0 * eps * dimension * max(1.0, _norm(directions[:, index]))
        for index in range(equality_count)
    )
    if range_g > range_limit_g:
        raise NumericalRankBoundary("gradient is outside range(G)")
    if range_a > range_limit_a:
        raise NumericalRankBoundary("equality direction is outside range(G)")

    singular_values = torch.linalg.svdvals(directions)
    direction_scale = float(singular_values.max().cpu().item())
    direction_tolerance = _rank_tolerance(direction_scale, max(directions.shape))
    equality_rank = int((singular_values > direction_tolerance).sum().cpu().item())
    if equality_rank != equality_count:
        raise NumericalRankBoundary("equality directions are rank deficient")

    u0 = -(pinv @ gradient_vector)
    schur_raw = directions.transpose(0, 1) @ pinv @ directions
    schur = 0.5 * (schur_raw + schur_raw.transpose(0, 1))
    schur_eigenvalues, schur_eigenvectors = torch.linalg.eigh(schur)
    schur_scale = float(torch.max(torch.abs(schur_eigenvalues)).cpu().item())
    schur_tolerance = _rank_tolerance(schur_scale, equality_count)
    if float(schur_eigenvalues.min().cpu().item()) < -schur_tolerance:
        raise NumericalRankBoundary("equality Schur complement is not PSD")
    schur_keep = schur_eigenvalues > schur_tolerance
    if int(schur_keep.sum().cpu().item()) != equality_count:
        raise NumericalRankBoundary("equality Schur complement is singular")
    schur_inverse = (
        schur_eigenvectors[:, schur_keep]
        * (1.0 / schur_eigenvalues[schur_keep])[None, :]
    ) @ schur_eigenvectors[:, schur_keep].transpose(0, 1)
    mismatch = rates - directions.transpose(0, 1) @ u0
    multiplier = schur_inverse @ mismatch
    velocity = u0 + pinv @ directions @ multiplier

    equality_residual = _norm(directions.transpose(0, 1) @ velocity - rates)
    stationarity_residual = _norm(symmetric @ velocity + gradient_vector - directions @ multiplier)
    equality_limit = 512.0 * eps * dimension * max(1.0, _norm(rates))
    stationarity_limit = 512.0 * eps * dimension * max(
        1.0, _norm(gradient_vector), scale * _norm(velocity)
    )
    if equality_residual > equality_limit:
        raise NumericalRankBoundary("equality residual exceeds deterministic tolerance")
    if stationarity_residual > stationarity_limit:
        raise NumericalRankBoundary("KKT stationarity residual exceeds tolerance")
    objective_tensor = torch.dot(gradient_vector, velocity) + 0.5 * torch.dot(
        velocity, symmetric @ velocity
    )
    if not bool(torch.isfinite(objective_tensor)):
        raise NumericalRankBoundary("Rayleighian objective is non-finite")
    receipt = RayleighianNumericalReceipt(
        dimension=dimension,
        equality_count=equality_count,
        matrix_rank=rank,
        equality_rank=equality_rank,
        dtype=str(fisher.dtype),
        matrix_scale=scale,
        pinv_tolerance=tolerance,
        range_residual_g=range_g,
        range_residual_a_max=range_a,
        equality_residual=equality_residual,
        stationarity_residual=stationarity_residual,
        objective=float(objective_tensor.detach().cpu().item()),
    )
    return RayleighianSolution(
        velocity=velocity.detach().contiguous(),
        lagrange_multiplier=multiplier.detach().contiguous(),
        receipt=receipt,
    )


__all__ = ["RayleighianSolution", "solve_equality_rayleighian"]
