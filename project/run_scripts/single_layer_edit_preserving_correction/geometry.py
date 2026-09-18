"""Exact FP64 ENFC geometry, with streamed columns and factored projectors.

No model code, policy selection, tensor persistence, or native P mutation lives
here.  Rank is determined from singular values, never from eigenvalues of K Kᵀ.
TSQR removes only QR's structurally zero rows, not numerically small directions.
Its final SVD contains *all* min(original_shape) singular values.  The original
matrix dimensions, not the compressed factor dimensions, determine tau.

Memory is O(n² + block_columns*n), not O(number_of_tokens²); exact 14336-row
rank diagnostics can nevertheless be expensive.  ``workspace_plan`` describes
factor sizes, not a measured peak, and callers must measure actual CPU cost.
No approximation/sketch or data-dependent rank tolerance is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, Any
import hashlib
import functools
import json
import math
import time

import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits

EPS64 = np.finfo(np.float64).eps
AMBIGUITY_FACTOR = 10.0
PROJECTOR_CEILING = 1e-10


class GeometryError(ValueError):
    """Technical geometry error; must not be relabelled ordinary fallback."""


class RankUnresolved(GeometryError):
    """Typed no-correction rank outcome (not permission to change cutoff)."""


def _bounded_cpu(function):
    @functools.wraps(function)
    def call(*args, **kwargs):
        with threadpool_limits(limits=8):
            return function(*args, **kwargs)
    return call


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        if value.device.type != "cpu":
            raise GeometryError("CPU geometry requires an explicit CPU tensor")
        value = value.detach().numpy()
    value = np.asarray(value)
    if value.ndim != 2 or value.dtype.kind not in "fiu":
        raise GeometryError("expected a real rank-two matrix")
    if not np.isfinite(value).all():
        raise GeometryError("nonfinite geometry input")
    return value


def matrix_sha256(value: Any) -> str:
    """Local convention: JSON shape/dtype header, newline, C-order bytes.

    Deliberately distinct from any prior headerless tensor SHA convention.
    """
    value = _numpy(value)
    h = hashlib.sha256(json.dumps({"shape": list(value.shape),
                                  "dtype": value.dtype.str},
                                 sort_keys=True, separators=(",", ":")).encode())
    h.update(b"\n")
    for start in range(0, value.shape[0], 128):
        h.update(np.ascontiguousarray(value[start:start + 128]).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class ColumnBlocks:
    """Restartable, ordered source of complete key columns, including repeats.

    ``factory`` must yield [rows, block_tokens] arrays, without filtering, and
    reproduce the same entries on every call.  Identity/provenance is supplied
    and sealed by the caller; this class does not infer token identity.
    """
    rows: int
    columns: int
    factory: Callable[[], Iterator[np.ndarray]]
    identity: str = "NOT_BOUND_BY_GEOMETRY"

    def blocks(self) -> Iterator[np.ndarray]:
        count = 0
        if self.rows < 0 or self.columns < 0:
            raise GeometryError("negative source shape")
        for value in self.factory():
            block = _numpy(value)
            if block.shape[0] != self.rows:
                raise GeometryError("column source row mismatch")
            count += block.shape[1]
            if count > self.columns:
                raise GeometryError("column source contains excess columns")
            if block.shape[1]:
                yield np.asarray(block, dtype=np.float64)
        if count != self.columns:
            raise GeometryError("column source cardinality mismatch")

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.columns

    def left(self, transform: np.ndarray, identity: str = "") -> "ColumnBlocks":
        transform = np.asarray(_numpy(transform), dtype=np.float64)
        if transform.shape[1] != self.rows:
            raise GeometryError("left transform shape mismatch")
        return ColumnBlocks(transform.shape[0], self.columns,
                            lambda: (transform @ x for x in self.blocks()),
                            identity or f"left({self.identity})")


def columns(value: Any, block_columns: int = 512) -> ColumnBlocks:
    if isinstance(value, ColumnBlocks):
        return value
    if not isinstance(block_columns, int) or block_columns <= 0:
        raise GeometryError("block_columns must be positive")
    value = _numpy(value)
    return ColumnBlocks(*value.shape,
                        lambda: (value[:, i:i + block_columns]
                                 for i in range(0, value.shape[1], block_columns)))


def workspace_plan(shape: tuple[int, int], block_columns: int = 512) -> dict:
    n, t = shape
    k = min(n, t)
    return {"shape": [n, t], "dtype": "float64", "algorithm": "TSQR_then_thin_SVD",
            "compressed_R_bytes": 8 * n * k,
            "maximum_stacked_QR_input_bytes": 8 * n * min(t, k + block_columns),
            "thin_left_vectors_bytes": 8 * n * k,
            "input_block_bytes": 8 * n * min(t, block_columns),
            "peak_status": "ESTIMATE_COMPONENTS_NOT_MEASURED_PEAK",
            "lapack_workspace_and_input_storage_additional": True}


def _rank_receipt(singular: np.ndarray, shape: tuple[int, int]) -> dict:
    largest = float(singular[0]) if singular.size else 0.0
    tau = max(shape, default=0) * EPS64 * largest
    ambiguous = np.zeros(singular.size, dtype=bool) if largest == 0 else (
        (singular >= tau / AMBIGUITY_FACTOR) & (singular <= tau * AMBIGUITY_FACTOR))
    rank = int(np.count_nonzero(singular > tau)) if largest else 0
    positive = singular[singular > 0]
    return {"shape": list(shape), "rank": rank,
            "status": "RANK_UNRESOLVED" if ambiguous.any() else "RESOLVED",
            "threshold": tau, "sigma_max": largest,
            "singular_values": singular.tolist(),
            "ambiguous_indices": np.flatnonzero(ambiguous).tolist(),
            "ambiguity_band": [tau / AMBIGUITY_FACTOR, tau * AMBIGUITY_FACTOR],
            "condition_full": None if len(positive) != len(singular) or not len(positive)
            else float(largest / positive[-1]),
            "condition_retained": None if rank == 0 else float(largest / singular[rank - 1]),
            "threshold_policy": "max(original_shape)*eps64*sigma_max",
            "algorithm": "TSQR(K.T)_then_SVD(R.T); no_Gram; no_rank_truncation"}


def _svd(value: Any, *, left_vectors: bool, threads: int = 8) -> tuple[dict, np.ndarray | None]:
    source = columns(value)
    if not isinstance(threads, int) or not 1 <= threads <= 8:
        raise GeometryError("CPU thread count must be 1..8")
    started = time.monotonic()
    n, t = source.shape
    r = np.empty((0, n), dtype=np.float64)
    blocks = 0
    with threadpool_limits(limits=threads):
        for block in source.blocks():
            blocks += 1
            stacked = np.concatenate((r, block.T), axis=0)
            # mode='r' does not form Q.  Discard only structural zero rows.
            packed_r = la.qr(stacked, mode="r", overwrite_a=True,
                             check_finite=False)[0]
            r = np.array(packed_r[:min(stacked.shape)], order="F", copy=True)
        if min(n, t) == 0:
            singular = np.empty(0, dtype=np.float64)
            u = np.empty((n, 0), dtype=np.float64) if left_vectors else None
        elif left_vectors:
            u, singular, _ = la.svd(r.T, full_matrices=False, check_finite=False,
                                     lapack_driver="gesdd")
        else:
            singular = la.svdvals(r.T, check_finite=False)
            u = None
    receipt = _rank_receipt(singular, source.shape)
    receipt.update(blocks=blocks, wall_seconds=time.monotonic() - started,
                   cpu_threads=threads, workspace_plan=workspace_plan(source.shape))
    return receipt, u


def rank_diagnostic(value: Any, *, threads: int = 8) -> dict:
    return _svd(value, left_vectors=False, threads=threads)[0]


@dataclass
class RightSpace:
    """Q=V(I−UUᵀ)Vᵀ; never stores the input_dimension² projector.

    U contains resolved *constrained* directions in V coordinates.  No null
    complement is constructed.  A rank-unresolved space cannot be applied.
    Both NumPy and torch projection return FP64 and never mutate their input.
    """
    basis: np.ndarray
    blocked: np.ndarray
    status: str
    diagnostic: dict = field(default_factory=dict)

    @property
    def dimension(self) -> int | None:
        return None if self.status == "RANK_UNRESOLVED" else self.basis.shape[1] - self.blocked.shape[1]

    @_bounded_cpu
    def project(self, value: Any, *, row_block: int = 128) -> Any:
        if self.status == "RANK_UNRESOLVED":
            raise RankUnresolved("unresolved rank: no correction may be constructed")
        if not isinstance(row_block, int) or row_block < 1:
            raise GeometryError("positive row_block required")
        if hasattr(value, "detach"):
            return self._project_torch(value, row_block)
        value = _numpy(value)
        if value.shape[1] != self.basis.shape[0]:
            raise GeometryError("right projection dimension mismatch")
        if self.dimension == 0:
            return np.zeros(value.shape, dtype=np.float64)
        out = np.empty(value.shape, dtype=np.float64)
        for i in range(0, value.shape[0], row_block):
            x = np.asarray(value[i:i + row_block], dtype=np.float64)
            reduced = x @ self.basis
            if self.blocked.shape[1]:
                reduced -= (reduced @ self.blocked) @ self.blocked.T
            out[i:i + row_block] = reduced @ self.basis.T
        if not np.isfinite(out).all():
            raise GeometryError("nonfinite projected direction")
        return out

    def _project_torch(self, value: Any, row_block: int) -> Any:
        import torch
        if value.ndim != 2 or value.shape[1] != self.basis.shape[0]:
            raise GeometryError("right projection dimension mismatch")
        if not bool(torch.isfinite(value).all()):
            raise GeometryError("nonfinite gradient")
        if self.dimension == 0:
            return torch.zeros_like(value, dtype=torch.float64)
        # Geometry factors are constants, not differentiable rank decisions.
        v = torch.as_tensor(self.basis, dtype=torch.float64, device=value.device)
        u = torch.as_tensor(self.blocked, dtype=torch.float64, device=value.device)
        chunks = []
        for x in value.split(row_block, dim=0):
            reduced = x.to(torch.float64) @ v
            if u.shape[1]:
                reduced = reduced - (reduced @ u) @ u.T
            chunks.append(reduced @ v.T)
        out = torch.cat(chunks, dim=0) if chunks else value.to(torch.float64).clone()
        if not bool(torch.isfinite(out).all()):
            raise GeometryError("nonfinite projected direction")
        return out

    def receipt(self) -> dict:
        return {**self.diagnostic, "status": self.status, "dimension": self.dimension,
                "input_dimension": self.basis.shape[0],
                "allowed_dimension": self.basis.shape[1],
                "blocked_dimension": self.blocked.shape[1],
                "basis_sha256": matrix_sha256(self.basis),
                "blocked_sha256": matrix_sha256(self.blocked)}


def _basis(value: Any) -> np.ndarray:
    value = np.asarray(_numpy(value), dtype=np.float64)
    error = float(la.norm(value.T @ value - np.eye(value.shape[1])))
    relative = error / max(1.0, math.sqrt(value.shape[1]))
    if relative > PROJECTOR_CEILING:
        raise GeometryError(f"nonorthonormal basis: relative Gram error {relative}")
    return value


@_bounded_cpu
def allowed_range(raw_projector: Any, *, provenance_basis: Any | None = None,
                  provenance: dict | None = None, threads: int = 8) -> RightSpace:
    """Bind P* while retaining P_raw bytes.

    Provenance path requires exact hashes of the raw matrix and the supplied
    basis under ``matrix_sha256``, plus the declared allowed rank.  It records
    P_raw−P* by streamed columns.  Without provenance, use the contract's
    symmetric-eigenvalue .1/.9 exclusion rule and split at .5; no native write
    ever receives P*.  Eigh's full n² CPU cost is explicit, not a thin-SVD claim.
    """
    raw = _numpy(raw_projector)
    if raw.shape[0] != raw.shape[1]:
        raise GeometryError("P_raw must be square")
    if not isinstance(threads, int) or not 1 <= threads <= 8:
        raise GeometryError("CPU thread count must be 1..8")
    started = time.monotonic()
    raw_hash = matrix_sha256(raw)
    eigen_receipt = None
    if provenance_basis is not None:
        supplied = _numpy(provenance_basis)
        if provenance is None or provenance.get("raw_sha256") != raw_hash or (
                provenance.get("basis_sha256") != matrix_sha256(supplied)):
            raise GeometryError("unbound projector provenance hashes")
        v = _basis(supplied)
        if v.shape[0] != raw.shape[0] or provenance.get("allowed_rank") != v.shape[1]:
            raise GeometryError("projector provenance dimension mismatch")
        method = "SEALED_NATIVE_RANGE_PROVENANCE"
    else:
        with threadpool_limits(limits=threads):
            symmetric = (np.asarray(raw, dtype=np.float64) + raw.T) * 0.5
            eigenvalues, vectors = la.eigh(symmetric, check_finite=False, driver="evd")
        nearest = np.minimum(abs(eigenvalues), abs(eigenvalues - 1))
        if np.any(nearest > 0.1) or np.any((eigenvalues >= 0.1) & (eigenvalues <= 0.9)):
            raise GeometryError("P_raw fails contract eigenvalue recovery criterion")
        v = _basis(vectors[:, eigenvalues > 0.5])
        eigen_receipt = {"eigenvalues": eigenvalues.tolist(),
                         "max_distance_to_zero_or_one": float(nearest.max(initial=0)),
                         "split": 0.5, "ambiguity_interval": [0.1, 0.9]}
        method = "SYMMETRIC_EIGEN_RECOVERY"
    difference2 = raw2 = asymmetry2 = 0.0
    for i in range(0, raw.shape[1], 256):
        block = np.asarray(raw[:, i:i + 256], dtype=np.float64)
        difference2 += float(np.sum((block - v @ v[i:i + 256].T) ** 2))
        raw2 += float(np.sum(block ** 2))
        asymmetry2 += float(np.sum((block - raw[i:i + 256].T) ** 2))
    return RightSpace(v, np.empty((v.shape[1], 0), dtype=np.float64),
                      "REPAIR_SPACE_EMPTY" if not v.shape[1] else "RESOLVED",
                      {"kind": "KL-P", "range_method": method,
                       "raw_sha256": raw_hash, "raw_mutated": False,
                       "provenance": provenance, "eigen_recovery": eigen_receipt,
                       "raw_vs_star_frobenius": math.sqrt(difference2),
                       "raw_vs_star_relative": math.sqrt(difference2 / raw2) if raw2 else 0.,
                       "raw_asymmetry_frobenius": math.sqrt(asymmetry2),
                       "wall_seconds": time.monotonic() - started,
                       "cpu_threads": threads})


def edit_null_space(allowed: RightSpace | np.ndarray, keys: Any, *,
                    kind: str = "EN-F", threads: int = 8) -> RightSpace:
    """Exact allowed-range intersection, used with full or subject-mean keys.

    ``keys`` must already embody the caller's correct token/prefix provenance;
    this function cannot promote subject keys into full-token protection.
    """
    if isinstance(allowed, RightSpace):
        if allowed.blocked.shape[1] or allowed.status == "RANK_UNRESOLVED":
            raise GeometryError("intersection expects an unblocked allowed-range basis")
        v = allowed.basis
    else:
        v = _basis(allowed)
    source = columns(keys)
    if v.shape[0] != source.rows:
        raise GeometryError("edit-key input dimension mismatch")
    diagnostic, u = _svd(source.left(v.T), left_vectors=True, threads=threads)
    rank = diagnostic["rank"]
    status = diagnostic["status"]
    blocked = u[:, :rank] if status == "RESOLVED" else np.empty((v.shape[1], 0))
    if status == "RESOLVED" and rank == v.shape[1]:
        status = "REPAIR_SPACE_EMPTY"
    return RightSpace(v, blocked, status,
                      {"kind": kind, "key_identity": source.identity,
                       "key_shape": list(source.shape), "reduced_rank": diagnostic})


def row_space(right_factor: Any, *, threads: int = 8) -> RightSpace:
    a = _numpy(right_factor)
    diagnostic, u = _svd(a.T, left_vectors=True, threads=threads)
    # The nonzero singular values and max(shape) are invariant to transpose.
    diagnostic["factor_shape"] = list(a.shape)
    status = diagnostic["status"]
    v = u[:, :diagnostic["rank"]] if status == "RESOLVED" else np.empty((a.shape[1], 0))
    if status == "RESOLVED" and not v.shape[1]:
        status = "REPAIR_SPACE_EMPTY"
    return RightSpace(v, np.empty((v.shape[1], 0)), status,
                      {"kind": "CA", "rank_A": diagnostic,
                       "coefficient_gauge_per_output": a.shape[0] - diagnostic["rank"]})


def ca_exact(right_factor: Any, keys: Any, *, output_dimension: int,
             threads: int = 8) -> tuple[RightSpace, dict]:
    if not isinstance(output_dimension, int) or output_dimension < 1:
        raise GeometryError("positive output_dimension required")
    a = _numpy(right_factor)
    ca = row_space(a, threads=threads)
    source = columns(keys)
    raw_ak = rank_diagnostic(source.left(a), threads=threads)
    if ca.status == "RANK_UNRESOLVED":
        return ca, {"status": "RANK_UNRESOLVED", "rank_A": ca.diagnostic["rank_A"],
                    "rank_raw_AK": raw_ak}
    exact = edit_null_space(ca, source, kind="CA-EXACT", threads=threads)
    dimension = exact.dimension
    return exact, {"status": exact.status, "rank_A": ca.diagnostic["rank_A"],
                   "rank_raw_AK": raw_ak, "rank_orthogonal_row_response": exact.diagnostic["reduced_rank"],
                   "coefficient_gauge_per_output": a.shape[0] - ca.basis.shape[1],
                   "physical_dimension_per_output": dimension,
                   "physical_dimension": None if dimension is None else output_dimension * dimension,
                   "raw_AK_rank_is_diagnostic_not_coordinate_invariant_dimension": True}


@_bounded_cpu
def projector_diagnostics(space: RightSpace) -> dict:
    """Deterministic operator residuals in a small exact QR coordinate basis.

    Uses the allowed-rank square core, not an input-dimension square Q or random
    probes. This is O(rank(P*)³) and is a technical-check cost, not a free check.
    """
    if space.status == "RANK_UNRESOLVED":
        return {"status": "RANK_UNRESOLVED", "relative_symmetry": None,
                "relative_idempotence": None}
    v, u = space.basis, space.blocked
    if not v.shape[1]:
        return {"status": "PASS", "relative_symmetry": 0., "relative_idempotence": 0.}
    r = la.qr(v, mode="r", check_finite=False)[0][:v.shape[1]]
    core = r @ (np.eye(v.shape[1]) - u @ u.T) @ r.T
    scale = max(1., float(la.norm(core)))
    sym = float(la.norm(core - core.T)) / scale
    idem = float(la.norm(core @ core - core)) / scale
    return {"status": "PASS" if max(sym, idem) <= PROJECTOR_CEILING else "FAIL",
            "relative_symmetry": sym, "relative_idempotence": idem,
            "denominator": "max(1,Frobenius(Q))", "ceiling": PROJECTOR_CEILING}


@_bounded_cpu
def gradient_diagnostics(gradient: Any, allowed: RightSpace, space: RightSpace,
                         *, native_output_basis: Any | None = None) -> tuple[np.ndarray, dict]:
    g = np.asarray(_numpy(gradient), dtype=np.float64)
    gp, gq = allowed.project(g), space.project(g)
    p2, q2 = float(np.sum(gp * gp)), float(np.sum(gq * gq))
    inner = float(np.sum(g * gq))
    span_fraction = None
    if native_output_basis is not None:
        out_basis = _basis(native_output_basis)
        if out_basis.shape[0] != g.shape[0]:
            raise GeometryError("native output basis shape mismatch")
        span_fraction = None if q2 == 0 else float(np.sum((out_basis.T @ gq) ** 2)) / q2
    return gq, {"gradient_norm_squared": float(np.sum(g * g)),
                "allowed_gradient_norm_squared": p2, "chi": q2,
                "G_dot_projected_G": inner, "orthogonal_chi_absolute_gap": abs(inner - q2),
                "remaining_gradient_fraction": None if p2 == 0 else q2 / p2,
                "native_output_span_fraction": span_fraction,
                "native_output_span_restricted": False}


@_bounded_cpu
def invariant_diagnostics(ideal_delta: Any, actual_delta: Any, native_weight: Any,
                          keys: Any, allowed: RightSpace) -> dict:
    """Stream actual per-token response; do not confuse ideal and rounded D.

    Caller forms actual_delta = float64(Wcandidate)-float64(Wnative). Neither
    logit/NLL invariants nor key stationarity can be established by this helper.
    """
    ideal = np.asarray(_numpy(ideal_delta), dtype=np.float64)
    actual = np.asarray(_numpy(actual_delta), dtype=np.float64)
    native = np.asarray(_numpy(native_weight), dtype=np.float64)
    if ideal.shape != actual.shape or actual.shape != native.shape:
        raise GeometryError("delta/native shape mismatch")
    source = columns(keys)
    if source.rows != actual.shape[1]:
        raise GeometryError("response-key shape mismatch")
    ideal_response2 = key2 = 0.
    maximum = 0.
    count = 0
    for block in source.blocks():
        ideal_response2 += float(np.sum((ideal @ block) ** 2))
        key2 += float(np.sum(block ** 2))
        response_norm = la.norm(actual @ block, axis=0)
        anchor_norm = np.maximum(1., la.norm(native @ block, axis=0))
        maximum = max(maximum, float(np.max(response_norm / anchor_norm, initial=0.)))
        count += block.shape[1]
    ideal_norm = float(la.norm(ideal))
    actual_norm = float(la.norm(actual))
    denom = ideal_norm * math.sqrt(key2)
    residual = math.sqrt(ideal_response2) / denom if denom else 0.
    leakage = float(la.norm(actual - allowed.project(actual))) / actual_norm if actual_norm else 0.
    return {"ideal_response_relative": residual,
            "ideal_response_denominator": "Frobenius(Dideal)*Frobenius(K), zero product -> zero",
            "actual_max_token_normalized_response": maximum,
            "actual_projection_leakage_relative": leakage,
            "ideal_norm": ideal_norm, "actual_norm": actual_norm,
            "rounding_delta_norm": float(la.norm(actual - ideal)),
            "valid_key_columns": count,
            "ideal_pass": residual <= 1e-10,
            "actual_response_pass": maximum <= 1e-5,
            "actual_leakage_pass": leakage <= 1e-5,
            "logits_NLL_ID_stationarity": "NOT_TESTED_BY_GEOMETRY"}
