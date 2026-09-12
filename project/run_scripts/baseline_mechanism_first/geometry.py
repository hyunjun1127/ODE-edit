"""E1 read-only matrix diagnostics; no solver here is a replacement writer.

Adapted equations: C1 canonical_alpha_fp32_solve, C2 dense/factor comparison,
C6 covariance inner product, C7 finite-zero handling, C8 geometry receipts.
Unlike C8, native P and M are NEVER symmetrized in the native-system audit.
All weight directions use explicit [output, input] orientation.
"""
from __future__ import annotations

import math
import time
from typing import Any

import torch


def _matrix(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2 or not value.is_floating_point():
        raise ValueError(f"{name}: expected floating matrix")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name}: NONFINITE")
    return value.detach()


def _norm(value: torch.Tensor) -> float:
    result = float(torch.linalg.vector_norm(value).item())
    if not math.isfinite(result):
        raise ValueError("NONFINITE_DIAGNOSTIC_NORM")
    return result


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else (0.0 if numerator == 0 else None)


@torch.no_grad()
def native_write_diagnostics(
    K: torch.Tensor, R: torch.Tensor, P: torch.Tensor, M: torch.Tensor,
    actual_delta: torch.Tensor, *, ridge: float = 1.0,
    weight_before: torch.Tensor | None = None,
    return_factor: bool = False,
) -> dict[str, Any]:
    """Audit native source-order dense RHS against the observed materialization.

    ``actual_delta`` must be the captured, shape-matched native update, not a
    factor reconstruction. Endpoint subtraction (which includes addition
    rounding) is a separate observation in ``WriterObservation``.
    """
    started = time.perf_counter()
    K, R, P, M, delta = [_matrix(n, x) for n, x in
                        zip(("K", "R", "P", "M", "delta"), (K, R, P, M, actual_delta))]
    d, c = K.shape
    if not c or R.shape[1] != c or P.shape != (d, d) or M.shape != (d, d) or delta.shape != (R.shape[0], d):
        raise ValueError("native writer geometry mismatch")
    if len({x.dtype for x in (K, R, P, M, delta)}) != 1 or K.dtype != torch.float32:
        raise ValueError("native writer operands must retain FP32")
    if len({x.device for x in (K, R, P, M, delta)}) != 1:
        raise ValueError("native operands must share device")
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be positive finite")
    A = P @ (K @ K.T + M) + ridge * torch.eye(d, dtype=K.dtype, device=K.device)
    PK = P @ K
    rhs = PK @ R.T
    # Both are diagnostics. Neither may be applied to a model by this module.
    dense = torch.linalg.solve(A, rhs).T
    F = torch.linalg.solve(A, PK)
    factor = R @ F.T
    if not all(bool(torch.isfinite(value).all()) for value in (dense, F, factor)):
        raise ValueError("NONFINITE_DIAGNOSTIC_SOLVE")
    solve_residual = _norm(A @ delta.T - rhs)
    local_residual = _norm(delta @ K - R)
    dense_norm = _norm(delta)
    result: dict[str, Any] = {
        "status": "FINITE_ZERO" if dense_norm == 0 else "FINITE",
        "direction_source": "ACTUAL_NATIVE_MATERIALIZED_DENSE",
        "writer_replaced": False, "native_history_symmetrized": False,
        "native_projector_symmetrized": False, "key_columns": c,
        "delta_norm": dense_norm,
        "delta_relative_norm": None if weight_before is None else _ratio(dense_norm, _norm(weight_before)),
        "delta_K_minus_R_norm": local_residual,
        "delta_K_minus_R_relative": _ratio(local_residual, _norm(R)),
        "native_solve_residual_norm": solve_residual,
        "native_solve_backward_error": _ratio(solve_residual, _norm(A) * dense_norm + _norm(rhs)),
        "recomputed_dense_vs_actual_norm": _norm(dense - delta),
        "recomputed_dense_vs_actual_bitwise": torch.equal(dense, delta),
        "factor_vs_actual_norm": _norm(factor - delta),
        "factor_vs_actual_relative": _ratio(_norm(factor - delta), dense_norm),
        "factor_vs_actual_max_abs": float((factor - delta).abs().max()),
        "factor_vs_actual_bitwise": torch.equal(factor, delta),
        "native_system_symmetry_error": _ratio(_norm(A - A.T), _norm(A)),
        "wall_seconds": time.perf_counter() - started,
        "timing_kind": "HOST_WALL_INCLUDES_SCALAR_SYNCHRONIZATION",
    }
    if return_factor:
        result["diagnostic_factor_F"] = F
    return result


@torch.no_grad()
def covariance_diagnostics(S: torch.Tensor, delta: torch.Tensor, C0: torch.Tensor) -> dict[str, Any]:
    """Raw covariance quadratics and BOTH cross orders; no clamp or PSD repair."""
    S, delta, C0 = [_matrix(n, v) for n, v in zip(("S", "delta", "C0"), (S, delta, C0))]
    if S.shape != delta.shape or C0.shape != (S.shape[1], S.shape[1]):
        raise ValueError("covariance geometry mismatch")
    SC, DC = S @ C0, delta @ C0
    qs, qd = float((SC * S).sum()), float((DC * delta).sum())
    cross_ds, cross_sd = float((DC * S).sum()), float((SC * delta).sum())
    total = float((((S + delta) @ C0) * (S + delta)).sum())
    if not all(math.isfinite(value) for value in (qs, qd, cross_ds, cross_sd, total)):
        raise ValueError("NONFINITE_COVARIANCE_DIAGNOSTIC")
    cosine = cross_ds / math.sqrt(qs * qd) if qs > 0 and qd > 0 else None
    return {
        "status": "FINITE_ZERO" if qs == qd == 0 else "FINITE",
        "C0_symmetrized": False, "q_C_S": qs, "q_C_delta": qd,
        "signed_delta_C_S": cross_ds, "signed_S_C_delta": cross_sd,
        "q_C_S_plus_delta": total,
        "quadratic_identity_residual": total - (qs + qd + cross_ds + cross_sd),
        "covariance_cosine": cosine,
        "cosine_status": "FINITE" if cosine is not None else "UNDEFINED_ZERO_OR_NONPOSITIVE_NORM",
        "negative_quadratic_observed": qs < 0 or qd < 0,
    }


def _spectrum(matrix: torch.Tensor, maximum_dimension: int | None, *, compute: bool) -> dict[str, Any]:
    """Exact SVD only below an explicitly supplied cost cap; no implied rank."""
    if not compute:
        return {"status": "NOT_MEASURED_YET", "method": None, "rank": None, "condition": None}
    if maximum_dimension is not None and max(matrix.shape, default=0) > maximum_dimension:
        return {"status": "OMITTED_EXPLICIT_DIMENSION_LIMIT", "method": None, "rank": None,
                "condition": None, "explicit_dimension_limit": maximum_dimension}
    values = torch.linalg.svdvals(matrix)
    largest = float(values.max()) if values.numel() else 0.0
    threshold = max(matrix.shape, default=0) * torch.finfo(matrix.dtype).eps * largest
    rank = int((values > threshold).sum())
    smallest = float(values.min()) if values.numel() else 0.0
    condition = largest / smallest if smallest > 0 else None
    total = float(values.sum())
    entropy_rank = 0.0
    if total > 0:
        p = values[values > 0] / total
        entropy_rank = float(torch.exp(-(p * p.log()).sum()))
    return {"status": "FINITE_ZERO" if largest == 0 else "FINITE", "method": "EXACT_SVD",
            "rank": rank, "rank_threshold": threshold, "largest": largest, "smallest": smallest,
            "effective_rank_entropy": entropy_rank, "condition": condition,
            "condition_status": "FINITE" if condition is not None else "SINGULAR_OR_EMPTY"}


@torch.no_grad()
def projected_geometry(P: torch.Tensor, C0: torch.Tensor, M: torch.Tensor, K: torch.Tensor,
                       *, basis: torch.Tensor | None = None, ridge: float = 1.0,
                       compute_spectrum: bool = False,
                       exact_max_dimension: int | None = None) -> dict[str, Any]:
    """Optional U is a diagnostic basis, never a replacement for stored P.

    Spectra of C0/M are raw singular spectra. Symmetric reduced H is explicitly
    labeled; its condition is not reported as the native condition.
    """
    P, C0, M, K = [_matrix(n, v) for n, v in zip(("P", "C0", "M", "K"), (P, C0, M, K))]
    d = K.shape[0]
    if any(v.shape != (d, d) for v in (P, C0, M)):
        raise ValueError("projected geometry mismatch")
    if (exact_max_dimension is not None and exact_max_dimension < 0) or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("invalid diagnostic cost cap or ridge")
    pk = P @ K
    denom = K.square().sum(0)
    ratios = [float(n / den) if float(den) > 0 else None
              for n, den in zip(pk.square().sum(0), denom)]
    A = P @ (K @ K.T + M) + ridge * torch.eye(d, dtype=K.dtype, device=K.device)
    result: dict[str, Any] = {
        "P_symmetry_error": _ratio(_norm(P - P.T), _norm(P)),
        "P_idempotence_error": _ratio(_norm(P @ P - P), _norm(P)),
        "P_spectrum": _spectrum(P, exact_max_dimension, compute=compute_spectrum),
        "C0_raw_singular_spectrum": _spectrum(C0, exact_max_dimension, compute=compute_spectrum),
        "M_symmetry_error": _ratio(_norm(M - M.T), _norm(M)),
        "native_system_spectrum": _spectrum(A, exact_max_dimension, compute=compute_spectrum),
        "Pk_squared_ratios": ratios, "zero_key_count": sum(v is None for v in ratios),
        "reduced_status": "NOT_OBSERVED_NO_BASIS", "writer_inputs_modified": False,
    }
    if basis is not None:
        U = _matrix("U", basis)
        if U.shape[0] != d:
            raise ValueError("U input width mismatch")
        KU, MU, CU = U.T @ K, U.T @ M @ U, U.T @ C0 @ U
        symmetric_MU = (MU + MU.T) * 0.5
        H = KU @ KU.T + symmetric_MU + ridge * torch.eye(U.shape[1], device=U.device, dtype=U.dtype)
        result.update({
            "reduced_status": "DIAGNOSTIC_ONLY",
            "basis_orthogonality_error": _norm(U.T @ U - torch.eye(U.shape[1], device=U.device, dtype=U.dtype)),
            "basis_projector_reconstruction_error": _ratio(_norm(U @ U.T - P), _norm(P)),
            "C_U_raw_singular_spectrum": _spectrum(CU, exact_max_dimension, compute=compute_spectrum),
            "M_U_raw_singular_spectrum": _spectrum(MU, exact_max_dimension, compute=compute_spectrum),
            "reduced_symmetric_diagnostic_H_spectrum": _spectrum(H, exact_max_dimension, compute=compute_spectrum),
            "reduced_history_symmetrization_norm": _norm(symmetric_MU - MU),
        })
    return result
