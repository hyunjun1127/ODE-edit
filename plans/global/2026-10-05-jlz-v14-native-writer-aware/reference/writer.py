"""Small FP64 CPU reference for native ridge and its full key adjoint.

This is an algebra reference, not a production model writer or optimizer.
Columns of K/R are requests, not an execution microbatch. A is fixed within
one candidate evaluation. Input geometry must already use the native metric.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Ridge:
    update: np.ndarray  # output x input
    P: np.ndarray      # input x requests
    M: np.ndarray      # requests x requests
    A: np.ndarray
    K: np.ndarray
    R: np.ndarray

    @property
    def energy(self) -> float:
        return float(np.sum((self.update @ self.A) * self.update))


def ridge(A: np.ndarray, K: np.ndarray, R: np.ndarray, *, woodbury=False) -> Ridge:
    A, K, R = (np.asarray(x, dtype=np.float64) for x in (A, K, R))
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if K.ndim != 2 or R.ndim != 2 or K.shape[0] != A.shape[0] or K.shape[1] != R.shape[1]:
        raise ValueError("K/R dimensions do not match A and request count")
    if not all(np.isfinite(x).all() for x in (A, K, R)):
        raise ValueError("nonfinite geometry")
    if not np.allclose(A, A.T, rtol=1e-12, atol=1e-13):
        raise ValueError("A must be symmetric; metric preparation is explicit")
    np.linalg.cholesky(A)  # no silent ridge/jitter fallback
    if woodbury:
        AinvK = np.linalg.solve(A, K)
        G = K.T @ AinvK
        P = np.linalg.solve((np.eye(G.shape[0]) + G).T, AinvK.T).T
    else:
        P = np.linalg.solve(A + K @ K.T, K)
    return Ridge(R @ P.T, P, K.T @ P, A, K, R)


def vjp(result: Ridge, grad_update: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return dL/dR and dL/dK for L(U), holding A fixed.

    If a loss also uses K directly (e.g. UK), its direct K cotangent must be
    added by the caller. Do not detach K in a causal chain of writers.
    """
    B = np.asarray(grad_update, dtype=np.float64)
    if B.shape != result.update.shape:
        raise ValueError("update cotangent shape mismatch")
    C = B.T @ result.R
    J = np.linalg.solve((result.A + result.K @ result.K.T).T, C)
    grad_R = B @ result.P
    grad_K = J - (J @ result.P.T + result.P @ J.T) @ result.K
    return grad_R, grad_K


def shared_requested_budget_bound(
    residuals: list[np.ndarray], anchors: list[np.ndarray], radius: np.ndarray
) -> dict:
    """Bound summed writer A-energy under sum_l ||R_lr||/a_lr <= b_r.

    The anchors are positive, fixed entry-state norms. The bound is on
    ideal native-writer energy, not task/locality loss, hidden trajectory,
    context leakage, or FP32 deployed updates.
    """
    if not residuals or len(residuals) != len(anchors):
        raise ValueError("one anchor vector per nonempty residual list")
    b = np.asarray(radius, dtype=np.float64)
    if b.ndim != 1 or not np.isfinite(b).all() or np.any(b < 0):
        raise ValueError("radius must be a finite nonnegative request vector")
    rho = []
    anchor_rows = []
    for R, a in zip(residuals, anchors):
        R, a = np.asarray(R, dtype=np.float64), np.asarray(a, dtype=np.float64)
        if R.ndim != 2 or R.shape[1] != b.size or a.shape != b.shape:
            raise ValueError("shared request dimension mismatch")
        if not np.isfinite(R).all() or not np.isfinite(a).all() or np.any(a <= 0):
            raise ValueError("anchors must be positive and all inputs finite")
        rho.append(np.linalg.norm(R, axis=0) / a)
        anchor_rows.append(a)
    usage = np.sum(rho, axis=0)
    if np.any(usage > b + 1e-12):
        raise ValueError("candidate exceeds shared requested budget")
    tight_intermediate = 0.25 * sum(float(np.sum(R * R)) for R in residuals)
    max_anchor_sq = np.max(np.stack(anchor_rows) ** 2, axis=0)
    total = 0.25 * float(np.sum(max_anchor_sq * b * b))
    return {"usage_per_request": usage.tolist(), "quarter_residual_frobenius_sq": tight_intermediate,
            "shared_budget_energy_bound": total}
