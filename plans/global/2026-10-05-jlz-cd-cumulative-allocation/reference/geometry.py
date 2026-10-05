"""FP64 mathematical reference; not a production writer or planner.

All geometry is frozen at batch entry. The retained weighted SVD is used by
the cost AND the subject-response operator. No tolerance or regularization
coefficient here is a prescribed experimental setting.
"""

from dataclasses import dataclass

import numpy as np


def _array(value, name, ndim=2):
    value = np.array(value, dtype=np.float64, copy=True)
    if value.ndim != ndim or not np.isfinite(value).all():
        raise ValueError(f"{name}: require finite {ndim}-D array")
    return value


def _symmetric_psd(value, name, size):
    value = _array(value, name)
    if value.shape != (size, size):
        raise ValueError(f"{name}: wrong square shape")
    if not np.allclose(value, value.T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name}: not symmetric")
    # Check only: do not silently alter eigenvalues, add jitter, or symmetrize.
    scale = max(1.0, float(np.linalg.norm(value, ord=2)))
    if float(np.linalg.eigvalsh(value)[0]) < -1e-11 * scale:
        raise ValueError(f"{name}: not positive semidefinite")
    return value


@dataclass(frozen=True)
class FrozenCD:
    keys: np.ndarray
    owners: np.ndarray
    weights: np.ndarray
    E: np.ndarray
    A: np.ndarray
    C0: np.ndarray
    H: np.ndarray
    L: np.ndarray
    previous_delta: np.ndarray
    lambda_cov: float
    retained_values: np.ndarray
    retained_right: np.ndarray
    rank_threshold: float
    Bmap: np.ndarray
    F: np.ndarray
    J: np.ndarray
    S: np.ndarray

    @property
    def request_count(self):
        return self.E.shape[0]

    @property
    def retained_rank(self):
        return self.retained_values.size


def build_geometry(keys, owners, A, C0, previous_delta, lambda_cov, *,
                   svd_rtol, svd_atol=0.0, weights=None):
    """Construct a frozen weighted retained-SVD writer map.

    Shapes: K=(input_dim, contexts), previous_delta=(output_dim, input_dim),
    owners=(contexts,) contiguous integer IDs 0..B-1. By default each owner's
    contexts share weight 1/(B*q_owner). Positive custom weights are allowed.
    H=A-lambda_cov*C0 must be PSD and A must be SPD.
    """
    K = _array(keys, "keys")
    n, m = K.shape
    if n == 0 or m == 0:
        raise ValueError("keys: empty dimension")
    owner_raw = np.asarray(owners)
    if owner_raw.shape != (m,) or owner_raw.dtype.kind not in "iu":
        raise ValueError("owners: require integer vector of context length")
    owner = owner_raw.astype(np.int64, copy=True)
    if np.any(owner < 0):
        raise ValueError("owners: negative ID")
    count = int(owner.max()) + 1
    counts = np.bincount(owner, minlength=count)
    if np.any(counts == 0):
        raise ValueError("owners: IDs must be contiguous and all present")
    if weights is None:
        w = 1.0 / (count * counts[owner])
    else:
        w = _array(weights, "weights", ndim=1)
        if w.shape != (m,) or np.any(w <= 0):
            raise ValueError("weights: require one positive weight per context")
    if not np.isfinite(lambda_cov) or lambda_cov <= 0:
        raise ValueError("lambda_cov: require positive finite scalar")
    for name, tol in (("svd_rtol", svd_rtol), ("svd_atol", svd_atol)):
        if not np.isfinite(tol) or tol < 0:
            raise ValueError(f"{name}: require nonnegative finite scalar")
    C0 = _symmetric_psd(C0, "C0", n)
    A = _symmetric_psd(A, "A", n)
    L = np.linalg.cholesky(A)  # No hidden diagonal regularization/fallback.
    H = _symmetric_psd(A - lambda_cov * C0, "H", n)
    previous = _array(previous_delta, "previous_delta")
    if previous.shape[1] != n or previous.shape[0] == 0:
        raise ValueError("previous_delta: wrong shape")
    E = np.zeros((count, m), dtype=np.float64)
    E[owner, np.arange(m)] = 1.0
    root_w = np.sqrt(w)
    X = np.linalg.solve(L, K) * root_w
    left, singular, right_t = np.linalg.svd(X, full_matrices=False)
    threshold = max(float(svd_atol), float(svd_rtol) * float(singular[0]))
    keep = singular > threshold
    left = left[:, keep]
    singular = singular[keep]
    right = right_t[keep].T
    C = (E * root_w) @ right
    Z = C / singular  # Empty retained rank is valid; this remains B-by-0.
    Bmap = np.linalg.solve(L.T, (Z @ left.T).T).T
    F = Z @ Z.T
    S = (C @ right.T) / root_w
    J = previous @ C0 @ Bmap.T
    arrays = [K, owner, w, E, A, C0, H, L, previous, singular,
              right, Bmap, F, J, S]
    for array in arrays:
        array.setflags(write=False)
    return FrozenCD(K, owner, w, E, A, C0, H, L, previous,
                    float(lambda_cov), singular, right, threshold,
                    Bmap, F, J, S)


def _target(D, geometry):
    D = _array(D, "D")
    if D.shape != (geometry.previous_delta.shape[0], geometry.request_count):
        raise ValueError("D: wrong output/request shape")
    return D


def projected_action(D, geometry):
    """Entry-writer subject actions; context columns retain native row order."""
    return _target(D, geometry) @ geometry.S


def action_pullback(action_gradient, geometry):
    gradient = _array(action_gradient, "action_gradient")
    expected = (geometry.previous_delta.shape[0], geometry.keys.shape[1])
    if gradient.shape != expected:
        raise ValueError("action_gradient: wrong shape")
    return gradient @ geometry.S.T


def writer_cost(D, geometry, *, alpha, allocation_price=1.0):
    """Q + 2*alpha*lambda_cov*abs(c), and its exact zero-choice subgradient.

    alpha=0 is the energy-only arm, alpha=1 the cumulative arm. A common
    allocation_price is an explicit caller parameter, not a native default.
    Neither B normalization nor microbatch normalization is inserted here.
    """
    D = _target(D, geometry)
    for name, scalar in (("alpha", alpha), ("allocation_price", allocation_price)):
        if not np.isfinite(scalar) or scalar < 0:
            raise ValueError(f"{name}: require nonnegative finite scalar")
    q = float(np.sum((D @ geometry.F) * D))
    cross = float(np.sum(D * geometry.J))
    cumulative = 2.0 * alpha * geometry.lambda_cov * abs(cross)
    gradient = (2.0 * D @ geometry.F
                + 2.0 * alpha * geometry.lambda_cov * np.sign(cross) * geometry.J)
    return {
        "Q": q,
        "cross": cross,
        "cumulative": cumulative,
        "Pi": q + cumulative,
        "weighted_cost": allocation_price * (q + cumulative),
        "gradient": allocation_price * gradient,
    }


def realization_diagnostics(D, geometry):
    """Observe discarded target; do not use a hidden compatibility gate."""
    D = _target(D, geometry)
    target = D @ geometry.E
    action = D @ geometry.S
    weighted_target = target * np.sqrt(geometry.weights)
    weighted_error = (target - action) * np.sqrt(geometry.weights)
    denominator = float(np.linalg.norm(weighted_target))
    error = float(np.linalg.norm(weighted_error))
    return {"weighted_target_norm": denominator,
            "discarded_target_norm": error,
            "discarded_target_relative": error / denominator if denominator else 0.0}
