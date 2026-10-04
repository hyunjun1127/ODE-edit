"""Exact Euclidean projection onto sum_l weights_l * ||block_l|| <= radius.

The returned ``tau_proj`` is the multiplier of this projection subproblem,
not the multiplier of the model editing objective. Float64 arithmetic is used.
"""

import numpy as np


def checked_blocks(blocks, name="blocks"):
    try:
        blocks = list(blocks)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonempty sequence of arrays") from exc
    if not blocks:
        raise ValueError(f"{name} must contain at least one block")
    if any(np.iscomplexobj(b) for b in blocks):
        raise ValueError(f"{name} must contain real arrays")
    try:
        out = [np.asarray(b, dtype=float) for b in blocks]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite arrays") from exc
    if any(b.ndim == 0 or b.size == 0 or not np.isfinite(b).all() for b in out):
        raise ValueError(f"{name} must contain finite, nonempty arrays")
    return out


def nonnegative_scalar(value, name):
    if np.ndim(value) != 0 or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a scalar")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def project(blocks, radius, weights=None):
    """Return (projected copies, tau_proj, exact_nonzero_mask).

    Group shapes may differ. Weights, when supplied, must be strictly positive.
    No diagnostic tolerance is used to prune projected groups.
    """
    blocks = checked_blocks(blocks)
    radius = nonnegative_scalar(radius, "radius")
    if weights is not None and np.iscomplexobj(weights):
        raise ValueError("weights must be real")
    w = np.ones(len(blocks)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(blocks),) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("weights must be a finite positive vector, one per block")
    norms = np.array([np.linalg.norm(b) for b in blocks])
    if not np.isfinite(norms).all():
        raise ValueError("block norms exceed the finite floating-point range")
    if float(w @ norms) <= radius:
        return [b.copy() for b in blocks], 0.0, norms > 0
    if radius == 0:
        return [np.zeros_like(b) for b in blocks], float(np.max(norms / w)), np.zeros(len(blocks), dtype=bool)
    order = np.argsort(-norms / w)
    s_wy, s_ww, tau_proj = 0.0, 0.0, 0.0
    for k, layer in enumerate(order):
        s_wy += w[layer] * norms[layer]
        s_ww += w[layer] ** 2
        tau_proj = (s_wy - radius) / s_ww
        next_break = norms[order[k + 1]] / w[order[k + 1]] if k + 1 < len(order) else -np.inf
        if tau_proj >= next_break:
            break
    keep = np.maximum(0.0, norms - tau_proj * w)
    out = [b * (n / old) if old > 0 else np.zeros_like(b) for b, n, old in zip(blocks, keep, norms)]
    return out, float(tau_proj), keep > 0
