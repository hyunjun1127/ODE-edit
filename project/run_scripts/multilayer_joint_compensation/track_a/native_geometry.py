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
from copy import deepcopy
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

    STATE_SCHEMA = "multilayer-joint-native-writer-metric-v1"

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

    def to_state(self) -> dict[str, Any]:
        """Return plain containers and CPU tensors safe for weights_only=True.

        Values are cloned, including the existing factorization. This does not
        change/recompute any construction output. The caller must SHA-seal the
        serialized artifact together with its source/input identity.
        """
        tensors = {
            "keys": self.keys, "history": self.history,
            "cholesky": self.cholesky, "projector_vectors": self.projector.vectors,
            "writer_factor_raw": self.writer_factor_raw,
            "writer_factor": self.writer_factor, "image_metric": self.image_metric,
            "raw_image_metric": self.raw_image_metric,
        }
        return {
            "schema": self.STATE_SCHEMA, "dimension": self.dimension,
            "batch": self.batch, "rank": self.rank, "ridge": self.ridge,
            "projector_complement": self.projector.complement,
            "mean_eigenvalue": self.mean_eigenvalue,
            "normalization": "trace(S)/din", "receipt": deepcopy(self.receipt),
            **{name: value.detach().to("cpu").clone() for name, value in tensors.items()},
        }

    @classmethod
    def from_state(cls, state: dict[str, Any], *,
                   storage_device: str | torch.device = "cpu") -> "NativeWriterMetric":
        """Reuse a sealed factorization; no full factorization/eigen solve.

        Schema/shape/dtype/finite checks and three deterministic factor-action
        probes detect ordinary malformed/mismatched state. They do not replace
        the caller's whole-file SHA verification. The probes use the existing
        64*dimension*eps_FP64 numerical roundoff scale, not a science threshold.
        """
        tensor_names = ("keys", "history", "cholesky", "projector_vectors",
                        "writer_factor_raw", "writer_factor", "image_metric", "raw_image_metric")
        expected = set(tensor_names) | {"schema", "dimension", "batch", "rank", "ridge",
                                       "projector_complement", "mean_eigenvalue", "normalization", "receipt"}
        if type(state) is not dict or set(state) != expected or state["schema"] != cls.STATE_SCHEMA:
            raise NativeGeometryBoundary("invalid native geometry state schema")
        if state["normalization"] != "trace(S)/din":
            raise NativeGeometryBoundary("state normalization mismatch")
        d, b, rank = state["dimension"], state["batch"], state["rank"]
        if any(type(v) is not int for v in (d, b, rank)) or d <= 0 or b < 0 or not 0 <= rank <= d:
            raise NativeGeometryBoundary("invalid state dimensions")
        if type(state["projector_complement"]) is not bool:
            raise NativeGeometryBoundary("invalid state projector mode")
        for scalar in ("ridge", "mean_eigenvalue"):
            if type(state[scalar]) not in (int, float) or not math.isfinite(state[scalar]) or state[scalar] <= 0:
                raise NativeGeometryBoundary(f"invalid state {scalar}")

        def plain_receipt(value: Any) -> bool:
            if value is None or type(value) in (bool, int, str):
                return True
            if type(value) is float:
                return math.isfinite(value)
            if type(value) in (list, tuple):
                return all(plain_receipt(item) for item in value)
            if type(value) is dict:
                return all(type(key) is str and plain_receipt(item) for key, item in value.items())
            return False

        if type(state["receipt"]) is not dict or not plain_receipt(state["receipt"]):
            raise NativeGeometryBoundary("state receipt is not finite primitive metadata")
        columns = d - rank if state["projector_complement"] else rank
        shapes = {"keys": (d, b), "history": (d, d), "cholesky": (d, d),
                  "projector_vectors": (d, columns), "writer_factor_raw": (b, d),
                  "writer_factor": (b, d), "image_metric": (b, b), "raw_image_metric": (b, b)}
        obj = cls.__new__(cls)
        obj.dimension, obj.batch, obj.rank = d, b, rank
        obj.ridge, obj.mean_eigenvalue = float(state["ridge"]), float(state["mean_eigenvalue"])
        obj.storage_device = torch.device(storage_device)
        for name in tensor_names:
            value = state[name]
            dtype = torch.float32 if name == "writer_factor" else torch.float64
            if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                    or tuple(value.shape) != shapes[name] or value.dtype != dtype):
                raise NativeGeometryBoundary(f"state tensor shape/dtype mismatch: {name}")
            _finite(value, f"state {name}")
            setattr(obj, name, value.detach().to(device=obj.storage_device).clone())
        obj.projector = Projector(obj.projector_vectors, state["projector_complement"], d)
        del obj.projector_vectors
        if not bool((obj.cholesky.diagonal() > 0).all()):
            raise NativeGeometryBoundary("state Cholesky has nonpositive diagonal")
        if not torch.equal(obj.writer_factor, obj.writer_factor_raw.float()):
            raise NativeGeometryBoundary("state physical writer is not the raw FP32 cast")
        mean = float((obj.keys.square().sum() + obj.history.diagonal().sum() + obj.ridge * d) / d)
        eps_scale = 64 * d * torch.finfo(torch.float64).eps
        if abs(mean - obj.mean_eigenvalue) > eps_scale * max(abs(mean), abs(obj.mean_eigenvalue)):
            raise NativeGeometryBoundary("state native normalization does not match S")
        # Three deterministic dense vectors exercise all coordinates without
        # refactorizing or retaining another d×d matrix.
        indices = torch.arange(d, dtype=torch.float64, device=obj.storage_device)
        probes = torch.stack((torch.ones_like(indices), ((indices % 7) - 3) / 4,
                              ((indices % 11) - 5) / 6))
        projected = obj.projector.right(probes)
        expected_action = obj.projector.right(obj._s_action(projected)) + obj.ridge * (probes - projected)
        actual_action = (probes @ obj.cholesky) @ obj.cholesky.T
        residual = actual_action - expected_action
        # Absolute-factor envelope avoids falsely flagging a tiny result that
        # is a cancellation of legitimate terms of the stored SPD operator.
        scale = (probes.abs() @ obj.cholesky.abs()) @ obj.cholesky.T.abs() + expected_action.abs()
        allowed = eps_scale * scale
        if bool((residual.abs() > allowed).any()):
            raise NativeGeometryBoundary("state factor-action probe mismatch")
        obj.receipt = deepcopy(state["receipt"])
        if (obj.receipt.get("dimension") != d or obj.receipt.get("batch") != b
                or obj.receipt.get("projector_rank") != rank or obj.receipt.get("ridge") != obj.ridge
                or obj.receipt.get("normalization") != "trace(S)/din"):
            raise NativeGeometryBoundary("state construction receipt identity mismatch")
        obj.reuse_receipt = {
            "schema": cls.STATE_SCHEMA, "status": "STATE_REUSED",
            "shape_dtype_finite": "PASS", "deterministic_probe_count": 3,
            "factor_action_probe_max_abs": float(residual.abs().max()),
            "factor_action_roundoff_scale": eps_scale,
            "full_refactorization_count": 0,
            "whole_artifact_sha_verification": "CALLER_REQUIRED",
        }
        return obj

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
