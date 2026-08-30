"""Outcome-blind rank-one equality tangents and covariance action matching."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable

import torch

from .contracts import NumericalLock, ScientificBoundary


def shape_rank_rtol(shape: tuple[int, int], dtype: torch.dtype = torch.float32) -> float:
    return max(shape) * torch.finfo(dtype).eps


def thin_column_basis(matrix: torch.Tensor) -> tuple[torch.Tensor, dict[str, float | int]]:
    if matrix.ndim != 2:
        raise ScientificBoundary("constraint matrix must be rank two")
    u, s, _ = torch.linalg.svd(matrix, full_matrices=False)
    rtol = shape_rank_rtol(tuple(matrix.shape), matrix.dtype)
    threshold = float(s.max().item()) * rtol if s.numel() else 0.0
    keep = s > threshold
    rank = int(keep.sum().item())
    if rank == 0:
        raise ScientificBoundary("constraint matrix has zero numerical rank")
    return u[:, keep], {
        "rank": rank,
        "effective_rank": rank,
        "rank_rtol": rtol,
        "smallest_retained_singular_value": float(s[keep][-1].item()),
        "largest_singular_value": float(s[0].item()),
    }


def project_input_tangent(
    raw: torch.Tensor,
    constraints: torch.Tensor,
    *,
    projector: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    if raw.ndim != 1 or constraints.shape[0] != raw.numel():
        raise ScientificBoundary("input tangent orientation mismatch")
    allowed = raw if projector is None else projector @ raw
    projected_constraints = constraints if projector is None else projector @ constraints
    basis, rank = thin_column_basis(projected_constraints)
    tangent = allowed - basis @ (basis.T @ allowed)
    norm = tangent.norm()
    if not torch.isfinite(norm) or float(norm.item()) == 0.0:
        raise ScientificBoundary("empty equality tangent")
    tangent = tangent / norm
    residual = torch.linalg.vector_norm(tangent @ constraints)
    denom = tangent.norm() * constraints.norm()
    rank["null_residual_relative"] = float((residual / denom.clamp_min(torch.finfo(tangent.dtype).tiny)).item())
    if projector is not None:
        rank["projector_residual_relative"] = float(
            ((projector @ tangent - tangent).norm() / tangent.norm()).item()
        )
    return tangent, rank


def rank_one_factor(delta: torch.Tensor, tolerance: float) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Factor a scientific single-request final-layer Official delta."""
    if delta.ndim != 2:
        raise ScientificBoundary("weight delta must be rank two")
    row_norms = torch.linalg.vector_norm(delta, dim=1)
    pivot = int(row_norms.argmax().item())
    v = delta[pivot].clone()
    denom = torch.dot(v, v)
    if float(denom.item()) == 0.0:
        raise ScientificBoundary("official final-layer delta is zero")
    u = delta @ v / denom
    residual = torch.linalg.norm(delta - torch.outer(u, v)) / delta.norm()
    value = float(residual.item())
    if value > tolerance:
        raise ScientificBoundary(f"official final-layer delta is not rank one: {value}")
    return u, v, value


def covariance_quadratic(vector: torch.Tensor, matvec: Callable[[torch.Tensor], torch.Tensor]) -> float:
    cv = matvec(vector)
    value = torch.dot(vector.to(torch.float64), cv.to(torch.float64))
    if not torch.isfinite(value) or float(value.item()) <= 0.0:
        raise ScientificBoundary("covariance quadratic is non-positive/non-finite")
    return float(value.item())


@dataclass(frozen=True)
class TangentAxis:
    axis: int
    seed: int
    output: torch.Tensor
    input: torch.Tensor
    gamma: float
    base_action: float
    tangent_unit_action: float
    null_residual_relative: float
    cross_action_relative: float
    rank: dict[str, float | int]

    def correction(self, sign: int) -> torch.Tensor:
        if sign not in {-1, 1}:
            raise ValueError("sign must be +/-1")
        return torch.outer(self.output, self.input).mul_(self.gamma * sign)


def deterministic_seed(namespace: str, model: str, method: str, case_id: str, axis: int) -> int:
    digest = hashlib.sha256(f"{namespace}|{model}|{method}|{case_id}|{axis}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def generate_axes(
    *,
    base_delta: torch.Tensor,
    constraints: torch.Tensor,
    covariance_matvec: Callable[[torch.Tensor], torch.Tensor],
    lock: NumericalLock,
    model_alias: str,
    method: str,
    case_id: str,
    projector: torch.Tensor | None = None,
) -> tuple[list[TangentAxis], dict[str, float]]:
    base_u, base_v, factor_residual = rank_one_factor(
        base_delta, shape_rank_rtol(tuple(base_delta.shape), base_delta.dtype)
    )
    base_v_action = covariance_quadratic(base_v, covariance_matvec)
    base_action = float(torch.dot(base_u, base_u).item()) * base_v_action
    axes: list[TangentAxis] = []
    for axis in range(lock.random_axis_count):
        seed = deterministic_seed(lock.seed_namespace, model_alias, method, case_id, axis)
        generator = torch.Generator(device=base_delta.device)
        generator.manual_seed(seed)
        raw_v = torch.randn(base_delta.shape[1], generator=generator, device=base_delta.device, dtype=base_delta.dtype)
        v, rank = project_input_tangent(raw_v, constraints, projector=projector)
        raw_u = torch.randn(base_delta.shape[0], generator=generator, device=base_delta.device, dtype=base_delta.dtype)
        u = raw_u - base_u * (torch.dot(base_u, raw_u) / torch.dot(base_u, base_u))
        u = u / u.norm()
        v_action = covariance_quadratic(v, covariance_matvec)
        unit_action = float(torch.dot(u, u).item()) * v_action
        gamma = math.sqrt(lock.rho_tangent * base_action / unit_action)
        cross = float(torch.dot(base_u, u).abs().item()) * math.sqrt(base_v_action * v_action)
        cross_rel = cross / math.sqrt(base_action * unit_action)
        axes.append(
            TangentAxis(
                axis=axis,
                seed=seed,
                output=u,
                input=v,
                gamma=gamma,
                base_action=base_action,
                tangent_unit_action=unit_action,
                null_residual_relative=float(rank["null_residual_relative"]),
                cross_action_relative=cross_rel,
                rank=rank,
            )
        )
    return axes, {"base_action": base_action, "rank_one_factor_residual": factor_residual}


def candidate_action(axis: TangentAxis, sign: int) -> float:
    cross_signed = 0.0  # output factor is constructed orthogonal to base output factor
    return axis.base_action + axis.gamma**2 * axis.tangent_unit_action + sign * 2.0 * axis.gamma * cross_signed
