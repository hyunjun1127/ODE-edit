"""Small FP64 CPU reference for a realization-first linear writer.

This is a numerical specification, not a production/model implementation.
The objective is lexicographic: minimize weighted realization error first,
then choose the minimum A-energy writer among its minimizers. Numerical SVD
truncation restricts that statement to the retained singular subspace.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class WriterResult:
    update: np.ndarray
    diagnostics: dict[str, Any]


def _norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def solve_writer(
    A: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    *,
    cutoff: float | None = None,
    compatibility_rtol: float = 1e-10,
    compatibility_atol: float = 1e-12,
) -> WriterResult:
    """Solve U = Z X^+ L^-1 with no explicit inverse or ridge fallback.

    K has shape (key_width, constraint_count); T is (output_width, count).
    Weights must be positive and sum to one. They affect inconsistent least
    squares; compatible exact constraints do not depend on their values.

    We use the symmetric part of A, record its skew, and require it to be SPD.
    There is no jitter, target scaling, or regularized fallback. Compatibility
    tolerances classify results only; they do not modify the update.
    """
    A, K, T, weights = [np.asarray(a, dtype=np.float64) for a in (A, K, T, weights)]
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    n = A.shape[0]
    if K.ndim != 2 or T.ndim != 2 or K.shape[0] != n or K.shape[1] != T.shape[1]:
        raise ValueError("K and T have incompatible shapes")
    q = K.shape[1]
    if n == 0 or T.shape[0] == 0 or q == 0:
        raise ValueError("All dimensions must be positive")
    if weights.shape != (q,) or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0, rtol=1e-12, atol=1e-14):
        raise ValueError("weights must be positive, length q, and normalized")
    if not all(np.all(np.isfinite(a)) for a in (A, K, T, weights)):
        raise ValueError("Inputs must be finite")
    if compatibility_atol < 0 or compatibility_rtol < 0:
        raise ValueError("Compatibility tolerances must be nonnegative")

    skew = A - A.T
    A_sym = (A + A.T) * 0.5
    # Raises LinAlgError on non-SPD A. Never add unrequested diagonal jitter.
    L = np.linalg.cholesky(A_sym)
    root_w = np.sqrt(weights)
    X = np.linalg.solve(L, K * root_w[None, :])
    Z = T * root_w[None, :]
    left, singular, right_t = np.linalg.svd(X, full_matrices=False)
    sigma_max = float(singular[0]) if singular.size else 0.0
    threshold = (np.finfo(np.float64).eps * max(X.shape) * sigma_max
                 if cutoff is None else float(cutoff))
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("cutoff must be finite and nonnegative")
    keep = singular > threshold
    rank = int(np.count_nonzero(keep))
    if rank:
        right = right_t[keep, :]
        white_update = ((Z @ right.T) / singular[keep][None, :]) @ left[:, keep].T
        # U L = white_update. This is a solve, not an explicit inverse.
        update = np.linalg.solve(L.T, white_update.T).T
        projected_Z = (Z @ right.T) @ right
        X_retained = (left[:, keep] * singular[keep][None, :]) @ right
        retained_condition = float(singular[keep][0] / singular[keep][-1])
    else:
        update = np.zeros((T.shape[0], n), dtype=np.float64)
        projected_Z = np.zeros_like(Z)
        X_retained = np.zeros_like(X)
        retained_condition = None
    residual = (update @ K - T) * root_w[None, :]
    infeasible = Z - projected_Z
    target_norm = _norm(Z)
    compatible = _norm(infeasible) <= compatibility_atol + compatibility_rtol * target_norm
    weighted_residual_norm = _norm(residual)
    projection_error = _norm(residual + infeasible)
    numerical_verified = projection_error <= compatibility_atol + compatibility_rtol * target_norm
    nominal_status = ("COMPATIBLE_EXACT_WITHIN_TOLERANCE" if compatible
                      else "RANGE_PROJECTED_WEIGHTED_LEAST_SQUARES")
    # Least-squares stationarity holds against retained X even after truncation.
    orthogonality = residual @ X_retained.T
    d = {
        "status": nominal_status if numerical_verified else "NUMERICAL_REALIZATION_FAILURE",
        "nominal_geometry_status": nominal_status,
        "numerical_projection_verified": bool(numerical_verified),
        "compatible_with_retained_subspace": bool(compatible),
        "shape": {"key_width": n, "output_width": T.shape[0], "constraint_count": q},
        "rank": rank,
        "svd_cutoff": threshold,
        "cutoff_rule": "eps64 * max(X.shape) * sigma_max" if cutoff is None else "explicit_absolute_cutoff",
        "singular_values": singular.tolist(),
        "retained_condition_number": retained_condition,
        "discarded_singular_count": int(singular.size - rank),
        "positive_singular_values_below_cutoff": int(np.count_nonzero((singular > 0) & ~keep)),
        "numerical_truncation": bool(np.any((singular > 0) & ~keep)),
        "A_skew_frobenius": _norm(skew),
        "A_skew_relative": _norm(skew) / max(_norm(A_sym), np.finfo(np.float64).tiny),
        "weighted_target_norm": target_norm,
        "weighted_residual_norm": weighted_residual_norm,
        "retained_infeasible_target_norm": _norm(infeasible),
        "residual_projection_identity_error": projection_error,
        "retained_LS_orthogonality_norm": _norm(orthogonality),
        "weighted_residual_by_column": np.linalg.norm(residual, axis=0).tolist(),
        "unweighted_residual_by_column": np.linalg.norm(update @ K - T, axis=0).tolist(),
        "energy_A": float(np.sum((update @ A_sym) * update)),
        "update_frobenius": _norm(update),
        "compatibility_rtol": compatibility_rtol,
        "compatibility_atol": compatibility_atol,
    }
    return WriterResult(update=update, diagnostics=d)


def applied_fp32_diagnostics(W_old: np.ndarray, result: WriterResult, K: np.ndarray, T: np.ndarray) -> dict[str, Any]:
    """Distinguish ideal FP64 U, FP32 cast U, and the representable W change."""
    W_old = np.asarray(W_old, dtype=np.float32)
    K, T = np.asarray(K, dtype=np.float64), np.asarray(T, dtype=np.float64)
    if W_old.shape != result.update.shape:
        raise ValueError("W_old and update must have identical orientation/shape")
    U32 = result.update.astype(np.float32)
    W_new = W_old + U32
    U_effective = W_new.astype(np.float64) - W_old.astype(np.float64)
    return {
        "ideal_equality_error": _norm(result.update @ K - T),
        "cast_equality_error": _norm(U32.astype(np.float64) @ K - T),
        "applied_equality_error": _norm(U_effective @ K - T),
        "cast_update_error": _norm(U32.astype(np.float64) - result.update),
        "applied_update_error": _norm(U_effective - result.update),
        "ideal_update_norm": _norm(result.update),
        "applied_update_norm": _norm(U_effective),
        "unchanged_weight_fraction": float(np.mean(W_new == W_old)),
        "weight_and_update_finite": bool(np.isfinite(W_new).all() and np.isfinite(U32).all()),
        "limitation": "Parameter-space diagnostic; native activation matmul/cast order requires model-level parity.",
    }


def repeat_context_targets(
    K_rows: np.ndarray,
    owner_indices: np.ndarray,
    roles: list[str],
    row_weights: np.ndarray,
    D: np.ndarray,
    *,
    included_roles: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Explicit row contract; never infer a zero target merely from a KL role.

    D is output_width x request_count. Rows remain in input order, including
    duplicate keys and zero-target requests. Each included row receives its
    owner's local increment. The caller declares the constraint role policy.
    Exclusion from writer constraints is not exclusion from the native loss.
    """
    K_rows = np.asarray(K_rows, dtype=np.float64)
    owner_indices = np.asarray(owner_indices, dtype=np.int64)
    row_weights = np.asarray(row_weights, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    q = K_rows.shape[1]
    if len(roles) != q or owner_indices.shape != (q,) or row_weights.shape != (q,):
        raise ValueError("row metadata must match K_rows")
    if np.any(owner_indices < 0) or np.any(owner_indices >= D.shape[1]):
        raise ValueError("owner index outside D")
    selected = np.flatnonzero([role in included_roles for role in roles])
    if selected.size == 0 or np.any(row_weights[selected] <= 0):
        raise ValueError("Must select rows with strictly positive weights")
    weights = row_weights[selected] / row_weights[selected].sum()
    return K_rows[:, selected], D[:, owner_indices[selected]], weights, selected
