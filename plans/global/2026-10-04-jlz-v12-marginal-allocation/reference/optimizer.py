"""Request-local EfficiencyAdam heuristic, plus the declared norm subgradient.

Euclidean projection after coordinatewise Adam does not guarantee objective
KKT stationarity. Moments are retained even for projected-zero groups.
"""

import numpy as np

try:
    from .budget import checked_blocks, nonnegative_scalar
except ImportError:  # Direct unittest discovery from this directory.
    from budget import checked_blocks, nonnegative_scalar


def objective_gradients(blocks, smooth_grads, decay):
    """g_J = g_F + decay * u/||u||; choose the norm subgradient 0 at u=0.

    ``smooth_grads`` are the one existing backward pass through F. This helper
    has no model calls. No near-zero cutoff changes the optimization gradient.
    """
    blocks = checked_blocks(blocks)
    smooth_grads = checked_blocks(smooth_grads, "smooth_grads")
    decay = nonnegative_scalar(decay, "decay")
    if len(blocks) != len(smooth_grads) or any(u.shape != g.shape for u, g in zip(blocks, smooth_grads)):
        raise ValueError("smooth_grads must match the block count and shapes")
    return [g + decay * u / norm if (norm := np.linalg.norm(u)) > 0 else g.copy()
            for u, g in zip(blocks, smooth_grads)]


class EfficiencyAdam:
    def __init__(self, shapes, lr, betas=(0.9, 0.999), eps=1e-8, efficiency=True):
        self.shapes = [tuple(shape) for shape in shapes]
        if not self.shapes or any(not shape or any(not isinstance(d, (int, np.integer)) or isinstance(d, bool) or d <= 0 for d in shape) for shape in self.shapes):
            raise ValueError("shapes must contain nonempty tuples of positive integer dimensions")
        self.lr = nonnegative_scalar(lr, "lr")
        self.eps = nonnegative_scalar(eps, "eps")
        if self.eps == 0:
            raise ValueError("eps must be positive")
        if len(betas) != 2 or any(not np.isfinite(b) or not 0 <= b < 1 for b in betas):
            raise ValueError("betas must be two finite values in [0, 1)")
        self.b1, self.b2 = map(float, betas)
        self.efficiency = bool(efficiency)
        self.m = [np.zeros(shape) for shape in self.shapes]
        self.v = [np.zeros(shape) for shape in self.shapes]
        self.s = np.zeros(len(self.shapes))
        self.t = 0

    @classmethod
    def from_native(cls, shapes, anchor_star, native_lr=0.1, native_eps=1e-8, betas=(0.9, 0.999), efficiency=True):
        """Use lr_u=native_lr/a*, eps_u=a*native_eps.

        Delta-coordinate native parity applies only to one declared variable
        block with a_layer=a*, not an arbitrary sole active block of many.
        """
        anchor_star = nonnegative_scalar(anchor_star, "anchor_star")
        if anchor_star == 0:
            raise ValueError("anchor_star must be positive")
        return cls(shapes, native_lr / anchor_star, betas, anchor_star * native_eps, efficiency)

    def step(self, params, grads):
        params = checked_blocks(params, "params")
        grads = checked_blocks(grads, "grads")
        if len(params) != len(self.shapes) or len(grads) != len(self.shapes):
            raise ValueError("params and grads must contain one block per declared shape")
        if any(p.shape != shape or g.shape != shape for p, g, shape in zip(params, grads, self.shapes)):
            raise ValueError("params and grads must match declared shapes")
        t = self.t + 1
        b1t, b2t = 1 - self.b1 ** t, 1 - self.b2 ** t
        m = [self.b1 * old + (1 - self.b1) * g for old, g in zip(self.m, grads)]
        v = [self.b2 * old + (1 - self.b2) * g * g for old, g in zip(self.v, grads)]
        s = self.b2 * self.s + (1 - self.b2) * np.array([np.mean(g * g) for g in grads])
        s_hat = s / b2t
        mean_s = float(np.mean(s_hat))
        gamma = np.sqrt(s_hat / mean_s) if self.efficiency and mean_s > 0 else np.ones(len(params))
        new = [p - self.lr * gl * (ml / b1t) / (np.sqrt(vl / b2t) + self.eps)
               for p, gl, ml, vl in zip(params, gamma, m, v)]
        if not all(np.isfinite(x).all() for x in [*m, *v, s, gamma, *new]):
            raise ValueError("optimizer arithmetic exceeded the finite floating-point range")
        self.m, self.v, self.s, self.t = m, v, s, t
        return new, gamma
