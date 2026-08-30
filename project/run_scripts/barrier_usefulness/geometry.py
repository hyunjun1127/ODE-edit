"""Equality-feasible, actual second-moment orthogonal tangent construction."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable

import torch

from .contracts import BarrierLock, ScientificBoundary


MatVec = Callable[[torch.Tensor], torch.Tensor]


def numerical_rank_basis(matrix: torch.Tensor) -> tuple[torch.Tensor, dict[str, float | int]]:
    if matrix.ndim != 2 or matrix.numel() == 0:
        raise ScientificBoundary("empty/rank-invalid constraint bank")
    u, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
    rtol = max(matrix.shape) * torch.finfo(matrix.dtype).eps
    threshold = float(singular.max().item()) * rtol
    keep = singular > threshold
    rank = int(keep.sum().item())
    if rank == 0:
        raise ScientificBoundary("zero numerical constraint rank")
    return u[:, keep], {
        "rank": rank,
        "rank_rtol": rtol,
        "largest_singular": float(singular[0].item()),
        "smallest_retained": float(singular[keep][-1].item()),
    }


def equality_tangent(raw: torch.Tensor, constraints: torch.Tensor, projector: torch.Tensor | None) -> tuple[torch.Tensor, dict[str, float | int]]:
    allowed = raw if projector is None else projector @ raw
    registered = constraints if projector is None else projector @ constraints
    basis, receipt = numerical_rank_basis(registered)
    tangent = allowed - basis @ (basis.T @ allowed)
    if projector is not None:
        tangent = projector @ tangent
        tangent = tangent - basis @ (basis.T @ tangent)
    norm = tangent.norm()
    if not torch.isfinite(norm) or float(norm.item()) == 0.0:
        raise ScientificBoundary("empty equality/projector intersection")
    tangent = tangent / norm
    floor = torch.finfo(tangent.dtype).tiny
    receipt["NA_relative"] = float((constraints.T @ tangent).norm().div(constraints.norm() * tangent.norm()).clamp_min(floor).item())
    if projector is not None:
        receipt["P_feasible_relative"] = float((projector @ tangent - tangent).norm().div(tangent.norm()).item())
    return tangent, receipt


def actual_quadratic(matrix: torch.Tensor, matvec: MatVec, *, row_chunk: int = 16) -> float:
    total = torch.zeros((), dtype=torch.float64, device=matrix.device)
    for begin in range(0, matrix.shape[0], row_chunk):
        block = matrix[begin : begin + row_chunk]
        cb = torch.stack([matvec(row) for row in block])
        total += (block.double() * cb.double()).sum()
    value = float(total.item())
    if not math.isfinite(value) or value <= 0:
        raise ScientificBoundary("actual registered second-moment action invalid")
    return value


def actual_cross(delta: torch.Tensor, output: torch.Tensor, cv: torch.Tensor) -> float:
    # N=outer(output,v), so <delta,N>_C = output dot (delta @ C v).
    return float(torch.dot(output.double(), (delta.double() @ cv.double())).item())


def deterministic_seed(namespace: str, model: str, method: str, case_id: str, axis: int) -> int:
    digest = hashlib.sha256(f"{namespace}|{model}|{method}|{case_id}|{axis}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True)
class Axis:
    axis_id: str
    seed: int
    output: torch.Tensor
    input: torch.Tensor
    gamma: float
    base_action: float
    unit_action: float
    cross: float
    total_action: float
    equality: dict[str, float | int]

    def correction(self, sign: int) -> torch.Tensor:
        if sign not in (-1, 1):
            raise ValueError("sign must be +/-1")
        return torch.outer(self.output, self.input) * (self.gamma * sign)


def build_axes(
    *,
    base_delta: torch.Tensor,
    constraints: torch.Tensor,
    matvec: MatVec,
    lock: BarrierLock,
    model: str,
    method: str,
    case_id: str,
    count: int,
    projector: torch.Tensor | None,
    base_action: float | None = None,
) -> list[Axis]:
    base_action = actual_quadratic(base_delta, matvec) if base_action is None else float(base_action)
    if not math.isfinite(base_action) or base_action <= 0:
        raise ScientificBoundary("provided base action invalid")
    answer: list[Axis] = []
    for ordinal in range(count):
        seed = deterministic_seed(lock.seed_namespace, model, method, case_id, ordinal)
        generator = torch.Generator(device=base_delta.device).manual_seed(seed)
        raw_v = torch.randn(base_delta.shape[1], generator=generator, device=base_delta.device, dtype=base_delta.dtype)
        v, equality = equality_tangent(raw_v, constraints, projector)
        cv = matvec(v)
        qv = float(torch.dot(v.double(), cv.double()).item())
        if not math.isfinite(qv) or qv <= 0:
            raise ScientificBoundary("candidate input action invalid")
        raw_u = torch.randn(base_delta.shape[0], generator=generator, device=base_delta.device, dtype=base_delta.dtype)
        dc = base_delta @ cv
        denom = torch.dot(dc, dc)
        u = raw_u if float(denom.item()) == 0.0 else raw_u - dc * (torch.dot(dc, raw_u) / denom)
        u = u / u.norm()
        measured_cross = actual_cross(base_delta, u, cv)
        unit_action = float(torch.dot(u.double(), u.double()).item()) * qv
        gamma = math.sqrt(lock.rho_tangent * base_action / unit_action)
        cross_scaled = actual_cross(base_delta, u * gamma, cv)
        # The candidate N is constructed exactly as an outer product, so this
        # is its exact registered action, not a factor approximation of D.
        correction_action = gamma * gamma * unit_action
        total = base_action + 2.0 * cross_scaled + correction_action
        cross_relative = abs(cross_scaled) / math.sqrt(base_action * correction_action)
        action_relative = abs(total / base_action - (1.0 + lock.rho_tangent))
        if max(cross_relative, action_relative, float(equality["NA_relative"])) > lock.fp32_relative_tolerance:
            raise ScientificBoundary("actual cross/equality/action shell gate failed")
        answer.append(Axis(
            axis_id=f"axis-{ordinal:02d}", seed=seed, output=u, input=v, gamma=gamma,
            base_action=base_action, unit_action=unit_action, cross=measured_cross,
            total_action=total, equality=equality,
        ))
    return answer


def projector_probe_audit(projector: torch.Tensor, constraints: torch.Tensor, *, seed: int, probes: int = 8) -> dict[str, object]:
    """Deterministic operator audit without materializing the prohibitive dense P^2."""
    generator = torch.Generator(device=projector.device).manual_seed(seed)
    symmetry = []
    idempotence = []
    for _ in range(probes):
        left = torch.randn(projector.shape[0], generator=generator, device=projector.device, dtype=projector.dtype)
        right = torch.randn(projector.shape[0], generator=generator, device=projector.device, dtype=projector.dtype)
        pl = projector @ left
        pr = projector @ right
        symmetry.append(float(abs(torch.dot(left, pr) - torch.dot(pl, right)).div(left.norm() * right.norm()).item()))
        idempotence.append(float((projector @ pl - pl).norm().div(pl.norm().clamp_min(torch.finfo(projector.dtype).tiny)).item()))
    projected_constraints = projector @ constraints
    _, rank = numerical_rank_basis(projected_constraints)
    return {
        "audit_kind": "deterministic-operator-probe-plus-trace-rank",
        "probe_count": probes,
        "rank_P_trace_rounded": int(round(float(torch.trace(projector).item()))),
        "rank_PA_star": int(rank["rank"]),
        "symmetry_probe_max": max(symmetry),
        "idempotence_probe_max": max(idempotence),
        "full_dense_P2_materialization_count": 0,
    }
