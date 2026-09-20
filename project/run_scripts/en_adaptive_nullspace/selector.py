"""Production scalar selector matching the sealed CPU reference exactly."""
from __future__ import annotations
import math
import numpy as np


def frontier(eigenvalues, mode_energies, *, exact_energy, loss, native_norm,
             native_action, epsilon, group_ends=None, loss_floor=1e-6):
    lam = np.asarray(eigenvalues, dtype=np.float64)
    energy = np.asarray(mode_energies, dtype=np.float64)
    if lam.ndim != 1 or lam.shape != energy.shape or not np.isfinite(lam).all() or not np.isfinite(energy).all():
        raise ValueError('invalid spectrum')
    if np.any(lam < 0) or np.any(energy < 0) or np.any(np.diff(lam) < 0):
        raise ValueError('ascending nonnegative spectrum required')
    if not math.isfinite(loss) or any(not math.isfinite(x) or x < 0 for x in (exact_energy, native_norm, native_action, epsilon, loss_floor)):
        raise ValueError('nonnegative finite scalars required')
    if group_ends is None:
        group_ends = [i+1 for i in range(len(lam)) if i+1 == len(lam) or lam[i+1] != lam[i]]
    if list(group_ends) != sorted(set(group_ends)) or (len(lam) and (not group_ends or group_ends[-1] != len(lam))):
        raise ValueError('frontier must include full allowed space')
    for k in group_ends:
        if not isinstance(k, (int, np.integer)) or not 1 <= k <= len(lam) or (k < len(lam) and lam[k-1] == lam[k]):
            raise ValueError('invalid indivisible singular group')
    g2 = np.r_[exact_energy, exact_energy + np.cumsum(energy)]
    a2 = np.r_[0., np.cumsum(lam * energy)]
    rows = []
    for k in [0, *group_ends]:
        g, action = math.sqrt(g2[k]), math.sqrt(a2[k])
        no_signal = loss <= loss_floor or g == 0 or native_norm == 0
        caps = (0., 0., 0.) if no_signal else (
            loss/g2[k], native_norm/g, epsilon*native_action/action if action else math.inf)
        eta = min(caps)
        rows.append(dict(released_modes=int(k), eigenvalue_upper=0. if not k else float(lam[k-1]),
                         gradient_energy=float(g2[k]), unit_coefficient_action_squared=float(a2[k]),
                         epsilon=float(epsilon), eta=float(eta), predicted_decrease=float(eta*g2[k]),
                         native_signed_loss=float(loss), loss_floor=float(loss_floor), loss_was_clamped=False,
                         correction_norm=float(eta*g), response_norm=float(eta*action),
                         caps={name: (float(value) if math.isfinite(value) else None)
                               for name, value in zip(('linear_zero_loss','native_norm','response'),caps)},
                         active_caps=[name for name, value in zip(('linear_zero_loss','native_norm','response'),caps)
                                      if math.isfinite(value) and value == eta],
                         status='NO_SIGNAL' if no_signal else 'PROPOSED' if eta > 0 else 'NO_STEP'))
    return rows


def choose(rows):
    if not rows:
        raise ValueError('empty frontier')
    remaining = list(rows)
    for key, maximize in [('predicted_decrease', True), ('correction_norm', False), ('response_norm', False)]:
        values = [r[key] for r in remaining]
        best = (max if maximize else min)(values)
        tol = 64*np.finfo(np.float64).eps*max(1., max(abs(x) for x in values))
        remaining = [r for r in remaining if abs(r[key]-best) <= tol]
    return min(remaining, key=lambda r:r['released_modes'])


def select_arms(spectrum, *, epsilon=.05, loss_floor=1e-6):
    args = {k: spectrum[k] for k in ('eigenvalues','mode_energies','exact_energy','loss','native_norm','native_action','group_ends')}
    frontiers = {str(e): frontier(**args, epsilon=e, loss_floor=loss_floor) for e in (.01, .05, .1)}
    rows = frontiers[str(epsilon)] if str(epsilon) in frontiers else frontier(**args, epsilon=epsilon, loss_floor=loss_floor)
    num = spectrum['numerical_released']
    exact = next(row for row in rows if row['released_modes'] == 0)
    numeric = next(row for row in rows if row['released_modes'] == num)
    adaptive = choose(rows)
    selected = {name: dict(row, arm=name, blocked_rank=spectrum['exact_rank']-row['released_modes'])
                for name, row in [('EN_EXACT', exact), ('EN_NUM', numeric), ('EN_ADAPT', adaptive)]}
    return dict(selected=selected, frontiers=frontiers, epsilon_primary=epsilon,
                algebra_only_epsilons=[.01,.1], loss_floor=loss_floor)


def quadratic_second_scale(loss0, slope, loss1):
    if not all(math.isfinite(x) for x in (loss0, slope, loss1)) or slope >= 0:
        raise ValueError('finite descent slope required')
    curvature = 2*(loss1-loss0-slope)
    if curvature <= 0:
        return dict(scale=None, curvature=curvature, reason='NO_POSITIVE_CURVATURE')
    t = min(1., max(0., -slope/curvature))
    if t == 0 or t == 1:
        return dict(scale=None, curvature=curvature, reason='ENDPOINT_ALREADY_AVAILABLE')
    return dict(scale=t, curvature=curvature, reason='QUADRATIC_INTERPOLATION')
