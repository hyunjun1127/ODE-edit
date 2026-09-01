"""Dense-authoritative Official Alpha backend and context incidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .contracts import EngineeringBoundary, FailureLabel


def _relative(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(left.dtype).tiny
    return torch.linalg.vector_norm(left - right) / torch.linalg.vector_norm(right).clamp_min(tiny)


def match_update_to_weight(update: torch.Tensor, weight_shape: torch.Size) -> tuple[torch.Tensor, bool]:
    """The single storage-orientation adapter, matching Official AlphaEdit."""

    if update.shape == weight_shape:
        return update, False
    if update.T.shape == weight_shape:
        return update.T, True
    raise EngineeringBoundary(FailureLabel.PROJECTOR_ORIENTATION_FAILURE.value)


@dataclass(frozen=True)
class DenseEntryFactor:
    projector: torch.Tensor
    history: torch.Tensor
    l2: float
    entry_matrix: torch.Tensor
    lu: torch.Tensor
    pivots: torch.Tensor

    @classmethod
    def build(cls, projector: torch.Tensor, history: torch.Tensor, l2: float) -> "DenseEntryFactor":
        if projector.ndim != 2 or projector.shape[0] != projector.shape[1]:
            raise EngineeringBoundary("projector must be square")
        if history.shape != projector.shape:
            raise EngineeringBoundary("dense cache/projector shape mismatch")
        if projector.dtype not in (torch.float32, torch.float64) or history.dtype != projector.dtype:
            raise EngineeringBoundary("dense backend requires common FP32/FP64 dtype")
        if not torch.isfinite(projector).all() or not torch.isfinite(history).all() or l2 <= 0:
            raise EngineeringBoundary(FailureLabel.CACHE_NONFINITE_OR_CORRUPT.value)
        identity = torch.eye(projector.shape[0], dtype=projector.dtype, device=projector.device)
        entry_matrix = l2 * identity + projector @ history
        # A_H is generally nonsymmetric.  LU is the only authoritative entry factorization.
        lu, pivots, info = torch.linalg.lu_factor_ex(entry_matrix)
        if int(info.max().item()) != 0:
            raise EngineeringBoundary(FailureLabel.DENSE_ALPHA_SOLVE_RESIDUAL_FAILURE.value)
        return cls(projector, history, float(l2), entry_matrix, lu, pivots)

    def woodbury(self, keys: torch.Tensor) -> "DenseSolve":
        if keys.ndim != 2 or keys.shape[0] != self.projector.shape[0] or keys.dtype != self.projector.dtype:
            raise EngineeringBoundary("key shape/dtype mismatch")
        projected_keys = self.projector @ keys
        q = torch.linalg.lu_solve(self.lu, self.pivots, projected_keys)
        small = torch.eye(keys.shape[1], dtype=keys.dtype, device=keys.device) + keys.T @ q
        # D = Q S^{-1}; this right-solve orientation is contract-authoritative.
        d = torch.linalg.solve(small.T, q.T).T
        residual = self.entry_matrix @ d + projected_keys @ (keys.T @ d) - projected_keys
        solve_relative = torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(projected_keys).clamp_min(
            torch.finfo(keys.dtype).tiny
        )
        return DenseSolve(keys, projected_keys, q, small, d, solve_relative)

    def direct(self, keys: torch.Tensor) -> torch.Tensor:
        projected_keys = self.projector @ keys
        full = self.entry_matrix + projected_keys @ keys.T
        return torch.linalg.solve(full, projected_keys)


@dataclass(frozen=True)
class DenseSolve:
    keys: torch.Tensor
    projected_keys: torch.Tensor
    q: torch.Tensor
    small: torch.Tensor
    d: torch.Tensor
    solve_relative: torch.Tensor

    def direct_relative(self, direct: torch.Tensor) -> torch.Tensor:
        return _relative(self.d, direct)

    def update(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.ndim != 2 or residual.shape[1] != self.d.shape[1]:
            raise EngineeringBoundary("residual/context shape mismatch")
        return residual @ self.d.T

    def coefficient_leakage(self, projector: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(projector.shape[0], dtype=projector.dtype, device=projector.device)
        numerator = torch.linalg.vector_norm((identity - projector) @ self.d)
        return numerator / torch.linalg.vector_norm(self.d).clamp_min(torch.finfo(self.d.dtype).tiny)


@dataclass(frozen=True)
class ContextIncidence:
    matrix: torch.Tensor
    request_ids: tuple[int, ...]
    context_request_ids: tuple[int, ...]
    repeat_factor: int | None

    @classmethod
    def from_context_request_ids(
        cls,
        request_ids: Iterable[int],
        context_request_ids: Iterable[int],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "ContextIncidence":
        requests = tuple(int(value) for value in request_ids)
        contexts = tuple(int(value) for value in context_request_ids)
        if not requests or len(set(requests)) != len(requests):
            raise EngineeringBoundary("request order must be nonempty and unique")
        index = {request_id: column for column, request_id in enumerate(requests)}
        if any(request_id not in index for request_id in contexts):
            raise EngineeringBoundary("context references unknown request")
        matrix = torch.zeros((len(contexts), len(requests)), dtype=dtype, device=device)
        for row, request_id in enumerate(contexts):
            matrix[row, index[request_id]] = 1
        counts = matrix.sum(0)
        if not torch.all(counts > 0) or not torch.all(matrix.sum(1) == 1):
            raise EngineeringBoundary("invalid request-context incidence")
        repeat = int(counts[0].item()) if torch.all(counts == counts[0]) else None
        return cls(matrix, requests, contexts, repeat)

    def effective(self, d: torch.Tensor) -> torch.Tensor:
        if d.shape[1] != self.matrix.shape[0]:
            raise EngineeringBoundary("D/context incidence column mismatch")
        return d @ self.matrix

    def repeated_residual(self, request_residual: torch.Tensor) -> torch.Tensor:
        if request_residual.shape[1] != self.matrix.shape[1]:
            raise EngineeringBoundary("request residual/incidence mismatch")
        return request_residual @ self.matrix.T

    def direct_update(self, request_residual: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self.repeated_residual(request_residual) @ d.T

    def effective_update(self, request_residual: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return request_residual @ self.effective(d).T
