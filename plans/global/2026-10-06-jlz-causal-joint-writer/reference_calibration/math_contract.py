"""Small calibration/proximal identities; no model or calibration fitting."""

import numpy as np


def ridge_scalar(requested, kappa):
    """Single mean key, fixed requested target and positive capacity."""
    if not np.isfinite(requested) or not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("require finite requested value and positive capacity")
    gamma = kappa / (1 + kappa)
    q = requested ** 2 * kappa / (1 + kappa) ** 2
    error = requested ** 2 / (1 + kappa) ** 2
    derivative = requested ** 2 * (1 - kappa) / (1 + kappa) ** 3
    return {"realized": gamma * requested, "Q": q, "E": error,
            "ridge_optimum_value": q + error,
            "dQ_dkappa": derivative, "exact_energy": requested ** 2 / kappa}


def _matrix(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"{name}: require finite output-by-request matrix")
    return value


def radial_price(requested, gradient_actual_task, gradient_native_norm,
                 gradient_actual_Q):
    """Report radial break-even price, never silently repair an invalid price.

    This is only -a+b+lambda*d=0 along t*R at t=1. It does not assert
    optimality, clamp equivalence, per-request balance, or positivity.
    """
    R = _matrix(requested, "R")
    gradients = [_matrix(x, name) for x, name in zip(
        (gradient_actual_task, gradient_native_norm, gradient_actual_Q),
        ("gF", "gN", "gQ"))]
    if any(g.shape != R.shape for g in gradients):
        raise ValueError("gradient shape mismatch")
    gF, gN, gQ = gradients
    a_per = -np.sum(gF * R, axis=0)
    b_per = np.sum(gN * R, axis=0)
    d_per = np.sum(gQ * R, axis=0)
    a, b, d = map(float, (a_per.sum(), b_per.sum(), d_per.sum()))
    result = {"a": a, "b": b, "d": d, "numerator": a - b,
              "lambda_candidate": None, "positive_price": False,
              "a_per_request": a_per.tolist(), "b_per_request": b_per.tolist(),
              "d_per_request": d_per.tolist()}
    if not np.any(R):
        result["status"] = "ZERO_REFERENCE"
    elif d <= 0:
        result["status"] = "NONPOSITIVE_RADIAL_Q_DENOMINATOR"
    else:
        coefficient = (a - b) / d
        if not np.isfinite(coefficient):
            result["status"] = "NONFINITE_PRICE"
            return result
        result["lambda_candidate"] = coefficient
        result["status"] = ("POSITIVE_RADIAL_BREAK_EVEN" if coefficient > 0
                            else "ZERO_RADIAL_PRICE" if coefficient == 0
                            else "NEGATIVE_RADIAL_PRICE")
        result["positive_price"] = coefficient > 0
        total = gF + gN + coefficient * gQ
        result["radial_residual_per_request"] = np.sum(total * R, axis=0).tolist()
        result["aggregate_radial_residual"] = float(np.sum(total * R))
        result["full_gradient_norm"] = float(np.linalg.norm(total))
    return result


def native_norm_u(u, anchors, coefficient=0.5):
    u = _matrix(u, "u")
    anchors = np.asarray(anchors, dtype=np.float64)
    if anchors.shape != (u.shape[1],) or np.any(anchors <= 0) or not np.isfinite(anchors).all():
        raise ValueError("anchors: positive finite per-request values required")
    if not np.isfinite(coefficient) or coefficient < 0:
        raise ValueError("norm coefficient must be finite and nonnegative")
    lengths = np.linalg.norm(u, axis=0)
    prices = coefficient / anchors
    value = float(np.sum(prices * lengths))
    gradient = u / np.where(lengths > 0, lengths, 1)[None, :] * prices[None, :]
    return value, gradient


def prox_native_balls(proposal, anchors, step, radius=0.75, coefficient=0.5):
    """prox of native local norm plus independent per-layer/request balls.

    In u=R/a coordinates the threshold is step*coefficient/a, not /a**2.
    This helper handles one layer; no request/layer is permanently removed.
    """
    proposal = _matrix(proposal, "proposal")
    anchors = np.asarray(anchors, dtype=np.float64)
    if anchors.shape != (proposal.shape[1],) or np.any(anchors <= 0) or not np.isfinite(anchors).all():
        raise ValueError("anchors: positive finite per-request values required")
    if not np.isfinite(step) or step <= 0 or not np.isfinite(radius) or radius < 0:
        raise ValueError("positive step and nonnegative radius required")
    if not np.isfinite(coefficient) or coefficient < 0:
        raise ValueError("nonnegative norm coefficient required")
    lengths = np.linalg.norm(proposal, axis=0)
    threshold = step * coefficient / anchors
    radii = np.minimum(radius, np.maximum(0, lengths - threshold))
    return proposal * (radii / np.where(lengths > 0, lengths, 1))[None, :]


def routed_witness(point):
    """Dropping the upper dependency does not give the original gradient.

    F=.5*(x-1)^2+.5*(y-x)^2+.5*y^2. The route drops the dependence
    of the second term on x, but retains its dependence on y.
    """
    x, y = np.asarray(point, dtype=np.float64)
    value = .5 * (x - 1) ** 2 + .5 * (y - x) ** 2 + .5 * y ** 2
    full = np.array([2 * x - y - 1, 2 * y - x])
    routed = np.array([x - 1, 2 * y - x])
    return float(value), full, routed
