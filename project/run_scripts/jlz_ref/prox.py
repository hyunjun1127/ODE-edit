"""Block proximal spectral gradient for the JLZ composite objective (reference).

The native decay is linear in ||r_lr|| and therefore non-smooth at zero; layers that the objective
does not use must come out exactly zero. Each trial point is prox_t(x - t g) with the exact radial
prox of the decay plus clamp ball (SpaRSA-type acceptance, nonmonotone over 10 points). Bookkeeping
follows the MEMIT-HJ v2 SPG conventions: total-call cap including the return evaluation, statuses
CONVERGED / STALLED_AT_PRECISION (six accepted plateau points) / NOT_CONVERGED / LINESEARCH_FAILED.
"""
import numpy as np


def solve(fun, h, prox, x0, *, tol, cap, max_trials=50, memory=10, sigma=1e-4, eps=np.finfo(float).eps):
    if cap < 2:
        raise ValueError("RETURN_EVALUATION_RESERVE")
    calls, backs = 0, 0

    def evaluate(x):
        nonlocal calls
        calls += 1
        f, g = fun(x)
        if not np.isfinite(f) or not np.all(np.isfinite(g)):
            raise FloatingPointError("NONFINITE_ORACLE")
        return f, g

    def residual(x, g):
        return float(np.linalg.norm(x - prox(x - g, 1.0)))

    x = prox(x0, 0.0)
    f, g = evaluate(x)
    scale = max(1.0, float(np.linalg.norm(g)))
    t = 1.0 / max(residual(x, g), 1e-12)
    hist, status = [f + h(x)], "NOT_CONVERGED"
    while calls < cap - 1:
        if residual(x, g) / scale <= tol:
            status = "CONVERGED"
            break
        ok = False
        for _ in range(max_trials):
            if calls >= cap - 1:
                break
            xn = prox(x - t * g, t)
            fn, gn = evaluate(xn)
            Fn = fn + h(xn)
            if Fn <= max(hist[-memory:]) - sigma / (2.0 * t) * float(np.sum((xn - x) ** 2)):
                ok = True
                break
            backs += 1
            t *= 0.5
        if not ok:
            status = "NOT_CONVERGED" if calls >= cap - 1 else "LINESEARCH_FAILED"
            break
        s, y = xn - x, gn - g
        sy = float(s @ y)
        t = min(1e12, max(1e-12, float(s @ s) / sy)) if sy > 0 else 1e12
        x, f, g = xn, fn, gn
        hist.append(Fn)
        if len(hist) >= 7:
            w = hist[-6:]
            floor = 10 * eps * max(max(map(abs, w)), np.finfo(float).tiny)
            best = [min(w[:i + 1]) for i in range(6)]
            if max(abs(w[i + 1] - w[i]) for i in range(5)) <= floor and best[0] - best[-1] <= floor:
                status = "STALLED_AT_PRECISION"
                break
    f, g = evaluate(x)
    final = residual(x, g) / scale
    if status == "CONVERGED" and final > tol:
        status = "NOT_CONVERGED"
    if status == "NOT_CONVERGED" and final <= tol:
        status = "CONVERGED"
    assert calls <= cap
    return x, g, dict(status=status, calls=calls, value=f + h(x), smooth=f, normalized_residual=final,
                      gradient_scale=scale, backtracks=backs, accepted_steps=len(hist) - 1, final_recomputed=True)


def native_adam(fun, c, rho, d, x0, lr=0.1, steps=25, stop=None, betas=(0.9, 0.999), eps=1e-8):
    """Diagnostic: the native compute_z loop applied blockwise (Adam, rescale each block to its radius)."""
    nb = len(c)
    x, m, v = x0.copy(), np.zeros_like(x0), np.zeros_like(x0)
    hits = np.zeros(nb, dtype=int)
    taken = 0
    for it in range(steps):
        f, g = fun(x)
        F = f + sum(c[i] * np.linalg.norm(x[i * d:(i + 1) * d]) for i in range(nb))
        if stop is not None and F < stop:
            break
        if it == steps - 1:
            break
        for i in range(nb):
            blk = x[i * d:(i + 1) * d]
            n = np.linalg.norm(blk)
            if n > 0:
                g[i * d:(i + 1) * d] += c[i] * blk / n
        taken += 1
        m = betas[0] * m + (1 - betas[0]) * g
        v = betas[1] * v + (1 - betas[1]) * g * g
        x = x - lr * (m / (1 - betas[0] ** taken)) / (np.sqrt(v / (1 - betas[1] ** taken)) + eps)
        for i in range(nb):
            blk = x[i * d:(i + 1) * d]
            n = np.linalg.norm(blk)
            if n > rho[i]:
                hits[i] += 1
                x[i * d:(i + 1) * d] = blk * (rho[i] / n)
    f, _ = fun(x)
    F = f + sum(c[i] * np.linalg.norm(x[i * d:(i + 1) * d]) for i in range(nb))
    ratios = [float(np.linalg.norm(x[i * d:(i + 1) * d]) / rho[i]) for i in range(nb)]
    return x, dict(value=F, adam_steps=taken, clamp_hits=hits.tolist(), norm_over_radius=ratios)
