"""MEMIT-HJ v2 z-step -- solve the original compute_z objective exactly (reference, NumPy).

Objective (unchanged, native compute_z): L(delta) = NLL + kl_factor * KL + v_weight_decay * ||delta|| / ||h||^2
subject to ||delta|| <= clamp_norm_factor * ||h||, with the native zero-step rule (if L(0) < stop, delta = 0).
Only the optimizer changes: the native loop (Adam lr 0.1, 24 steps, rescale after each step) is kept
here as a diagnostic reproduction; the method uses a spectral projected gradient (SPG) that stops on the
projected-gradient norm and reports the clamp multiplier nu_c = -<grad L, delta/r> when the ball is active.
fun_grad(x) -> (value, gradient); the production loss oracle is the parity-tested native loss
(project/run_scripts/single_layer_mechanism_first/z_hook.py) evaluated at the top edit layer.
"""
import numpy as np


def project_ball(x, radius):
    n = float(np.linalg.norm(x))
    return x if n <= radius else x * (radius / n)


def native_adam(fun_grad, x0, radius, lr=0.1, steps=25, stop=5e-2, betas=(0.9, 0.999), eps=1e-8):
    """Reproduce the native loop: evaluate, stop if loss < stop, no step on the last iteration,
    Adam step, then rescale to the clamp radius if exceeded."""
    x = x0.astype(float).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    hits, evals, taken, stopped = 0, 0, 0, "cap"
    for it in range(steps):
        f, g = fun_grad(x)
        evals += 1
        if f < stop:
            stopped = "loss"
            break
        if it == steps - 1:
            break
        taken += 1
        m = betas[0] * m + (1 - betas[0]) * g
        v = betas[1] * v + (1 - betas[1]) * g * g
        mh = m / (1 - betas[0] ** taken)
        vh = v / (1 - betas[1] ** taken)
        x = x - lr * mh / (np.sqrt(vh) + eps)
        if np.linalg.norm(x) > radius:
            hits += 1
            x = x * (radius / np.linalg.norm(x))
    f, _ = fun_grad(x)
    return x, dict(value=float(f), evaluations=evals, adam_steps=taken, clamp_hits=hits, stopped_by=stopped,
                   norm_over_radius=float(np.linalg.norm(x) / radius))


def spg_ball(fun_grad, x0, radius, tol=1e-8, max_iter=1000, memory=10, gamma=1e-4,
             alpha_min=1e-12, alpha_max=1e12):
    """Spectral projected gradient with nonmonotone Armijo line search on {||x|| <= radius}.

    Stops when ||P(x - g) - x|| <= tol * max(1, ||g0||). Returns x and diagnostics including the
    clamp multiplier (zero when the solution is interior).
    """
    x = project_ball(x0.astype(float), radius)
    f, g = fun_grad(x)
    evals = 1
    g0 = max(1.0, float(np.linalg.norm(g)))
    hist = [f]
    pg = float(np.linalg.norm(project_ball(x - g, radius) - x))
    alpha = 1.0 / max(pg, 1e-12)
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        pg = float(np.linalg.norm(project_ball(x - g, radius) - x))
        if pg <= tol * g0:
            converged = True
            break
        d = project_ball(x - alpha * g, radius) - x
        gd = float(g @ d)
        fmax = max(hist[-memory:])
        lam = 1.0
        while True:
            xn = x + lam * d
            fn, gn = fun_grad(xn)
            evals += 1
            if fn <= fmax + gamma * lam * gd or lam < 1e-20:
                break
            lam *= 0.5
        s, y = xn - x, gn - g
        sy = float(s @ y)
        alpha = min(alpha_max, max(alpha_min, float(s @ s) / sy)) if sy > 0 else alpha_max
        x, f, g = xn, fn, gn
        hist.append(f)
    nx = float(np.linalg.norm(x))
    active = nx >= radius * (1 - 1e-9)
    mult = max(0.0, -float(g @ x) / radius) if active else 0.0
    return x, dict(value=float(f), iterations=it, evaluations=evals, converged=converged,
                   projected_gradient=pg, active=bool(active), clamp_multiplier=mult,
                   norm_over_radius=nx / radius)


def exact_z(fun_grad, dim, radius, stop=5e-2, **kw):
    """Native zero-step rule, then SPG from the origin (the native starting point)."""
    x0 = np.zeros(dim)
    f0, _ = fun_grad(x0)
    if f0 < stop:
        return x0, dict(value=float(f0), iterations=0, evaluations=1, converged=True, zero_step=True,
                        active=False, clamp_multiplier=0.0, norm_over_radius=0.0)
    x, info = spg_ball(fun_grad, x0, radius, **kw)
    info["zero_step"] = False
    return x, info
