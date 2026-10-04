"""CPU NumPy diagnostics for minimizing F(u) + decay * sum_l ||u_l||.

The constraint is sum_l ||u_l|| <= radius; F is smooth. These diagnostics
neither solve the constrained problem nor certify convergence of an optimizer.
"""

import numpy as np


def _nonnegative_scalar(value, name):
    if np.ndim(value) != 0 or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative scalar") from exc
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return result


def _arrays(values, name):
    try:
        values = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonempty sequence of arrays") from exc
    if not values:
        raise ValueError(f"{name} must be nonempty")
    result = []
    for value in values:
        if np.iscomplexobj(value):
            raise ValueError(f"{name} arrays must be real")
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain finite arrays") from exc
        if array.ndim == 0 or array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain finite, nonempty arrays with ndim > 0")
        result.append(array)
    return result


def kkt_diagnostics(blocks, smooth_grads, radius, decay,
                    active_tol=1e-12, boundary_tol=1e-12):
    """Return JSON-serializable residuals, without modifying either input.

    ``smooth_grads`` excludes the nonsmooth group-norm penalty. Active groups
    have ||u_l|| > active_tol. This is diagnostic classification only: a small
    nonzero group is not pruned, and its norm still contributes to budget_used.
    Entries not applicable to a group's classification are None.

    On the budget boundary, mu is the nonnegative mean of active signed radial
    prices -<grad F_l, u_l/||u_l||> - decay; otherwise it is zero. For radius=0
    and exactly zero blocks, mu=max(0,max_l ||grad F_l||-decay) is an explicit
    degenerate convention. None of these choices is a projection threshold.

    Active vector residuals are grad F_l+(decay+mu)*u_l/||u_l||. Inactive
    inequality excesses are max(0,||grad F_l||-decay-mu). norm_only_residuals
    omit direction and are included solely to expose that weaker diagnostic.
    max_residual is an unscaled maximum, not a scientific acceptance gate.
    """
    radius = _nonnegative_scalar(radius, "radius")
    decay = _nonnegative_scalar(decay, "decay")
    active_tol = _nonnegative_scalar(active_tol, "active_tol")
    boundary_tol = _nonnegative_scalar(boundary_tol, "boundary_tol")
    blocks = _arrays(blocks, "blocks")
    smooth_grads = _arrays(smooth_grads, "smooth_grads")
    if len(blocks) != len(smooth_grads):
        raise ValueError("blocks and smooth_grads must have the same count")
    if any(u.shape != g.shape for u, g in zip(blocks, smooth_grads)):
        raise ValueError("each block and smooth gradient must have the same shape")

    norms = [float(np.linalg.norm(u.ravel())) for u in blocks]
    grad_norms = [float(np.linalg.norm(g.ravel())) for g in smooth_grads]
    budget_used = sum(norms)
    active = [norm > active_tol for norm in norms]
    directions = [u / norm if flag else None
                  for u, norm, flag in zip(blocks, norms, active)]
    prices = [float(-np.sum(g * e) - decay) if flag else None
              for g, e, flag in zip(smooth_grads, directions, active)]
    if radius == 0 and all(np.all(u == 0) for u in blocks):
        mu = max(0.0, max(grad_norms) - decay)
        convention = "zero_radius_degenerate"
    elif abs(budget_used - radius) <= boundary_tol and any(active):
        mu = max(0.0, float(np.mean([p for p in prices if p is not None])))
        convention = "boundary_active_radial_mean"
    else:
        mu = 0.0
        convention = "zero_off_boundary_or_no_active"

    vectors, residual_norms, excesses, norm_only = [], [], [], []
    for g, e, flag, grad_norm in zip(smooth_grads, directions, active, grad_norms):
        residual = g + (decay + mu) * e if flag else None
        vectors.append(residual.tolist() if flag else None)
        residual_norms.append(float(np.linalg.norm(residual.ravel())) if flag else None)
        excesses.append(None if flag else max(0.0, grad_norm - decay - mu))
        norm_only.append(abs(grad_norm - decay - mu) if flag else None)
    primal = max(0.0, budget_used - radius)
    complementarity = abs(mu * (budget_used - radius))
    dual = max(0.0, -mu)
    return {
        "budget_used": budget_used, "radius": radius, "decay": decay,
        "layer_norms": norms, "active": active,
        "near_zero_nonzero": [0 < norm <= active_tol for norm in norms],
        "mu": mu, "mu_convention": convention,
        "signed_radial_prices": prices,
        "active_vector_residuals": vectors,
        "active_residual_norms": residual_norms,
        "inactive_inequality_excess": excesses,
        "norm_only_residuals": norm_only,
        "primal_violation": primal, "complementarity": complementarity,
        "dual_violation": dual,
        "max_residual": max([primal, complementarity, dual]
                            + [v for v in residual_norms + excesses if v is not None]),
    }
