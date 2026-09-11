"""Fixed-entry native writer and metric for A, design §§4–7/9.

S = KK' + M + ridge I, J = K' Q (Q'SQ)^-1 Q'.  Only the existing
orthogonal Projector representation is reused; no v2 routing Metric is used.
We factor P(KK'+M)P + ridge I in the *full input dimension*. Projecting both
sides of its inverse is exactly Q(Q'SQ)^-1Q'; excluded coordinates are not
extra actuators. Native metric normalization is trace(S)/din, not trace(H)/rank.

Construction and operator storage default to CPU. Passing storage_device is an
explicit caller decision. Operator outputs retain the caller's device but use
FP64; only ``writer_factor`` and differentiable physical writes are FP32.
"""
from __future__ import annotations

import math
import time
from typing import Any

import torch

from project.run_scripts.l4_two_memory_conflict_routing.geometry import Projector


class NativeGeometryBoundary(RuntimeError):
    """Invalid input or numerical geometry, without a rescue metric."""


def _finite(tensor: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise NativeGeometryBoundary(f"nonfinite {label}")


def _relative(residual: torch.Tensor, reference: torch.Tensor) -> float:
    error, scale = float(torch.linalg.vector_norm(residual)), float(torch.linalg.vector_norm(reference))
    return error / scale if scale else (0. if error == 0 else math.inf)


class NativeWriterMetric:
    """Full-P* fixed-entry writer, metric action and exact native preconditioner.

    ``keys`` are [din,B], ``history`` is [din,din], and RHS R is [dout,B].
    The campaign uses ridge=1. A different positive ridge is accepted solely
    so explicit, consistently scaled raw-sum/mean representations can be
    checked: K'=K/sqrt(B), M'=M/B, ridge'=ridge/B, R'=R/sqrt(B).
    Changing only K is *not* the same writer. No rank ceiling or ridge repair.
    """

    def __init__(self, keys: torch.Tensor, history: torch.Tensor, projector: Projector,
                 *, ridge: float = 1., storage_device: str | torch.device = "cpu"):
        started = time.perf_counter()
        if keys.ndim != 2 or not keys.is_floating_point():
            raise NativeGeometryBoundary("keys must be floating [din,B]")
        dimension, batch = keys.shape
        if dimension <= 0 or history.shape != (dimension, dimension) or not history.is_floating_point():
            raise NativeGeometryBoundary("history shape/dtype mismatch")
        if not math.isfinite(ridge) or ridge <= 0:
            raise NativeGeometryBoundary("native ridge must be positive finite")
        if projector.dimension != dimension or projector.vectors.ndim != 2 or projector.vectors.shape[0] != dimension:
            raise NativeGeometryBoundary("projector dimension mismatch")
        self.dimension, self.batch, self.ridge = dimension, batch, float(ridge)
        self.storage_device = torch.device(storage_device)
        self.keys = keys.detach().to(device=self.storage_device, dtype=torch.float64).clone()
        self.history = history.detach().to(device=self.storage_device, dtype=torch.float64).clone()
        vectors = projector.vectors.detach().to(device=self.storage_device, dtype=torch.float64).clone()
        _finite(self.keys, "keys")
        _finite(self.history, "history")
        _finite(vectors, "projector vectors")
        columns = vectors.shape[1]
        if columns > dimension:
            raise NativeGeometryBoundary("too many projector vectors")
        gram = vectors.T @ vectors
        orth_error = float(torch.linalg.vector_norm(gram - torch.eye(columns, dtype=torch.float64,
                                                                   device=self.storage_device)))
        orth_bound = 64 * dimension * torch.finfo(torch.float64).eps * max(math.sqrt(columns), 1.)
        if orth_error > orth_bound:
            raise NativeGeometryBoundary("projector vectors are not orthonormal")
        self.projector = Projector(vectors, bool(projector.complement), dimension)
        self.rank = dimension - columns if projector.complement else columns
        history_asymmetry = float(torch.linalg.vector_norm(self.history - self.history.T))
        history_norm = float(torch.linalg.vector_norm(self.history))
        history_bound = 8 * dimension * torch.finfo(history.dtype).eps * history_norm
        if history_asymmetry > history_bound:
            raise NativeGeometryBoundary("nonsymmetric native history")
        self.history = (self.history + self.history.T) * .5
        self.mean_eigenvalue = float((self.keys.square().sum() + self.history.diagonal().sum()
                                     + self.ridge * dimension) / dimension)
        if not math.isfinite(self.mean_eigenvalue) or self.mean_eigenvalue <= 0:
            raise NativeGeometryBoundary("nonpositive native mean eigenvalue")

        projected_keys = self.projector.right(self.keys.T).T
        # Each projection is by the complete P*, stored as included or excluded
        # eigenvectors. This is full-dimensional factorization, never Ub100.
        matrix = self.projector.right(self.projector.right(self.history).T)
        matrix.add_(projected_keys @ projected_keys.T)
        projected_trace = float(matrix.diagonal().sum()) + self.ridge * self.rank
        matrix.diagonal().add_(self.ridge)
        matrix = (matrix + matrix.T) * .5
        _finite(matrix, "full projected native matrix")
        try:
            self.cholesky = torch.linalg.cholesky(matrix)
        except torch.linalg.LinAlgError as error:
            raise NativeGeometryBoundary("projected native matrix is not SPD; no ridge repair") from error
        del matrix
        self.writer_factor_raw = self.inverse(self.keys.T)
        self.writer_factor = self.writer_factor_raw.float()
        self.raw_image_metric = self._s_action(self.writer_factor_raw) @ self.writer_factor_raw.T
        physical64 = self.writer_factor.double()
        self.image_metric = self._s_action(physical64) @ physical64.T
        self.raw_image_metric = (self.raw_image_metric + self.raw_image_metric.T) * .5
        self.image_metric = (self.image_metric + self.image_metric.T) * .5
        _finite(self.image_metric, "FP32 writer-image metric")
        _finite(self.writer_factor, "FP32 writer factor")
        raw_stationarity = self.projector.right(self._s_action(self.writer_factor_raw) - self.keys.T)
        fp32_stationarity = self.projector.right(self._s_action(physical64) - self.keys.T)
        projected_rhs = self.projector.right(self.keys.T)
        self.receipt: dict[str, Any] = {
            "definition": "S=KK^T+M+ridgeI; J=K^TQ(Q^TSQ)^-1Q^T",
            "dimension": dimension, "batch": batch, "projector_rank": self.rank,
            "projector_stores_complement": bool(projector.complement),
            "projector_stored_columns": columns, "projector_orthogonality_error": orth_error,
            "projector_orthogonality_bound": orth_bound,
            "history_symmetry_error": history_asymmetry, "history_symmetry_bound": history_bound,
            "ridge": self.ridge, "extra_ridge": 0, "rank_truncation": 0,
            "factorized_dimension": dimension, "factorization": "FP64_FULL_DIMENSION_CHOLESKY",
            "storage_device": str(self.storage_device), "reduction_dtype": "float64",
            "writer_factor_dtype": "float32", "raw_factor_dtype": "float64",
            "mean_eigenvalue": self.mean_eigenvalue, "normalization": "trace(S)/din",
            "projected_trace": projected_trace,
            "projected_mean_eigenvalue": projected_trace / self.rank if self.rank else None,
            "raw_factor_projected_stationarity_relative": _relative(raw_stationarity, projected_rhs),
            "fp32_factor_projected_stationarity_relative": _relative(fp32_stationarity, projected_rhs),
            "raw_factor_projection_relative": _relative(self.writer_factor_raw - self.projector.right(self.writer_factor_raw), self.writer_factor_raw),
            "fp32_factor_projection_relative": _relative(physical64 - self.projector.right(physical64), physical64),
            "factor_cast_relative": _relative(physical64 - self.writer_factor_raw, self.writer_factor_raw),
            "image_metric_cast_relative": _relative(self.image_metric - self.raw_image_metric, self.raw_image_metric),
            "setup_wall_seconds": time.perf_counter() - started,
            "timer_scope": "CPU wall; caller must synchronize any explicit CUDA use",
        }

    def _rows(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.dimension or not x.is_floating_point():
            raise NativeGeometryBoundary("operator input must be [n,din]")
        result = x.to(device=self.storage_device, dtype=torch.float64)
        _finite(result, "operator input")
        return result

    def _s_action(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.keys) @ self.keys.T + x @ self.history + self.ridge * x

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector.right(self._rows(x)).to(x.device)

    def s_action(self, x: torch.Tensor, *, normalized: bool = False,
                 projected: bool = False) -> torch.Tensor:
        """Right multiply by S (or full-P* PSP), never a Frobenius proxy."""
        value = self._rows(x)
        if projected:
            value = self.projector.right(value)
        result = self._s_action(value)
        if projected:
            result = self.projector.right(result)
        if normalized:
            result = result / self.mean_eigenvalue
        _finite(result, "metric action")
        return result.to(x.device)

    def dot(self, x: torch.Tensor, y: torch.Tensor, *, normalized: bool = False,
            projected: bool = False) -> torch.Tensor:
        if x.shape != y.shape:
            raise NativeGeometryBoundary("metric pair shape mismatch")
        action = self.s_action(x, normalized=normalized, projected=projected)
        result = torch.sum(action * y.to(device=action.device, dtype=torch.float64))
        _finite(result, "metric dot")
        return result

    def energy(self, x: torch.Tensor, *, normalized: bool = False,
               projected: bool = False) -> torch.Tensor:
        return self.dot(x, x, normalized=normalized, projected=projected)

    def inverse(self, x: torch.Tensor, *, normalized: bool = False) -> torch.Tensor:
        """Exact projected inverse action; also the native block preconditioner."""
        value = self.projector.right(self._rows(x))
        solved = torch.cholesky_solve(value.T.contiguous(), self.cholesky).T
        result = self.projector.right(solved)
        if normalized:
            result = result * self.mean_eigenvalue
        _finite(result, "projected inverse")
        return result.to(x.device)

    def write(self, rhs: torch.Tensor, *, raw_fp64: bool = False) -> torch.Tensor:
        """Differentiable fixed-entry weight update. No physical model mutation."""
        if rhs.ndim != 2 or rhs.shape[1] != self.batch:
            raise NativeGeometryBoundary("RHS must be [dout,B]")
        dtype = torch.float64 if raw_fp64 else torch.float32
        if rhs.dtype != dtype:
            raise NativeGeometryBoundary(f"RHS must be {dtype}")
        _finite(rhs, "RHS")
        factor = self.writer_factor_raw if raw_fp64 else self.writer_factor
        result = rhs @ factor.to(rhs.device)
        _finite(result, "writer delta")
        return result
