"""CPU mathematical reference; no model loader, runner, or submission code."""
from __future__ import annotations

import math
import numpy as np


def frontier(eigenvalues, mode_energies, *, exact_energy, loss,
             native_norm, native_action, epsilon, group_ends=None):
    """Enumerate nested projected-gradient candidates in ascending eigen order.

    exact_energy must be the directly computed ||G Q_exact||^2. The other
    energies are ||G V u_j||^2; no large-norm subtraction is needed.
    Equal-eigenvalue groups must be supplied as indivisible boundaries if a
    backend treats numerically equal singular values as one group.
    This routine uses exact scalar algebra, not FP32 endpoint guarantees.
    """
    lam = np.asarray(eigenvalues, dtype=np.float64)
    energy = np.asarray(mode_energies, dtype=np.float64)
    if lam.ndim != 1 or lam.shape != energy.shape:
        raise ValueError('eigenvalue/energy shape mismatch')
    if not np.isfinite(lam).all() or not np.isfinite(energy).all():
        raise ValueError('nonfinite spectrum')
    if np.any(lam < 0) or np.any(energy < 0) or np.any(np.diff(lam) < 0):
        raise ValueError('require ascending nonnegative spectrum/energy')
    scalars = (exact_energy, loss, native_norm, native_action, epsilon)
    if any(not math.isfinite(x) or x < 0 for x in scalars):
        raise ValueError('nonnegative finite scalar required')
    if group_ends is None:
        group_ends = [i+1 for i in range(len(lam))
                      if i+1 == len(lam) or lam[i+1] != lam[i]]
    if any(not isinstance(k, (int,np.integer)) or k < 1 or k > len(lam)
           for k in group_ends) or list(group_ends) != sorted(set(group_ends)):
        raise ValueError('invalid group boundaries')
    for k in group_ends:
        if k < len(lam) and lam[k-1] == lam[k]:
            raise ValueError('cannot split a repeated eigenvalue')
    g2 = np.r_[exact_energy, exact_energy + np.cumsum(energy)]
    action2 = np.r_[0., np.cumsum(lam*energy)]
    result = []
    for k in [0, *group_ends]:
        g, action = math.sqrt(g2[k]), math.sqrt(action2[k])
        if g == 0 or loss == 0 or native_norm == 0:
            eta = 0.
            caps = (0., 0., 0.)
        else:
            caps = (loss/g2[k], native_norm/g,
                    epsilon*native_action/action if action else math.inf)
            eta = min(caps)
        result.append(dict(released_modes=int(k),
            eigenvalue_upper=0. if k == 0 else float(lam[k-1]),
            gradient_energy=float(g2[k]), unit_coefficient_action_squared=float(action2[k]),
            epsilon=float(epsilon), eta=float(eta),
            predicted_decrease=float(eta*g2[k]), correction_norm=float(eta*g),
            response_norm=float(eta*action),
            active_caps=[name for name,value in zip(('linear_zero_loss','native_norm','response'),caps)
                         if math.isfinite(value) and value == eta],
            status='PROPOSED' if eta > 0 else 'NO_STEP'))
    return result


def _tied(rows, key, maximize):
    values = [r[key] for r in rows]
    best = (max if maximize else min)(values)
    tol = 64*np.finfo(np.float64).eps*max(1., max(abs(x) for x in values))
    return [r for r in rows if abs(r[key]-best) <= tol]


def choose(rows):
    if not rows:
        raise ValueError('empty frontier')
    remaining = list(rows)
    for key,maximize in [('predicted_decrease',True), ('correction_norm',False),
                         ('response_norm',False)]:
        remaining = _tied(remaining,key,maximize)
    return min(remaining,key=lambda r:r['released_modes'])


def quadratic_second_scale(loss0, slope, loss1):
    """One first-ray endpoint gives a second t in [0,1]; not an optimizer proof."""
    if not all(math.isfinite(x) for x in (loss0,slope,loss1)) or slope >= 0:
        raise ValueError('finite descent slope required')
    curvature = 2*(loss1-loss0-slope)
    if curvature <= 0:
        return dict(scale=None, curvature=curvature, reason='NO_POSITIVE_CURVATURE')
    t = min(1.,max(0.,-slope/curvature))
    if t == 0 or t == 1:
        return dict(scale=None, curvature=curvature, reason='ENDPOINT_ALREADY_AVAILABLE')
    return dict(scale=t, curvature=curvature, reason='QUADRATIC_INTERPOLATION')
