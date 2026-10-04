"""Seven original attachment checks, with their original numerical scope.

The final allocation test exercises ordinary projected gradient, not Adam.
"""
import unittest

import numpy as np
from scipy.optimize import minimize

from budget import project
from optimizer import EfficiencyAdam


def native_adam(param, grads, lr, betas=(0.9, 0.999), eps=1e-8):
    m = np.zeros_like(param); v = np.zeros_like(param); out = []
    for t, g in enumerate(grads, start=1):
        m = betas[0] * m + (1 - betas[0]) * g
        v = betas[1] * v + (1 - betas[1]) * g * g
        param = param - lr * (m / (1 - betas[0] ** t)) / (np.sqrt(v / (1 - betas[1] ** t)) + eps)
        out.append(param.copy())
    return out


class Budget(unittest.TestCase):
    def test_inside_is_identity(self):
        y = [np.array([0.1, 0.0]), np.array([0.0, 0.2])]
        x, tau, _ = project(y, 1.0)
        for a, b in zip(x, y):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(tau, 0.0)

    def test_single_block_is_native_clamp(self):
        y = [np.array([3.0, 4.0])]
        x, _, _ = project(y, 2.0)
        np.testing.assert_allclose(x[0], y[0] * 2.0 / 5.0, atol=1e-12)

    def test_projection_matches_brute_force_and_kkt(self):
        rng = np.random.default_rng(0)
        y = [rng.standard_normal(3) * s for s in (3.0, 2.0, 1.0, 0.5)]
        w = np.array([1.0, 1.5, 0.7, 1.2])
        x, tau, active = project(y, 2.0, w)
        self.assertAlmostEqual(sum(wl * np.linalg.norm(b) for wl, b in zip(w, x)), 2.0, places=10)
        for l in range(4):
            ny, nx = np.linalg.norm(y[l]), np.linalg.norm(x[l])
            if active[l]:
                self.assertAlmostEqual(ny - nx, tau * w[l], places=10)
            else:
                self.assertLessEqual(ny, tau * w[l] + 1e-12)
        flat = np.concatenate(y)
        cons = {'type': 'ineq', 'fun': lambda z: 2.0 - sum(w[l] * np.linalg.norm(z[3 * l:3 * l + 3]) for l in range(4))}
        res = minimize(lambda z: 0.5 * np.sum((z - flat) ** 2), np.zeros(12), constraints=[cons], method='SLSQP',
                       options=dict(ftol=1e-14, maxiter=500))
        np.testing.assert_allclose(np.concatenate(x), res.x, atol=1e-5)


class Optimizer(unittest.TestCase):
    def test_single_layer_is_exactly_native_adam(self):
        rng = np.random.default_rng(1)
        grads = [rng.standard_normal(16) for _ in range(5)]
        ref = native_adam(np.zeros(16), grads, lr=0.1)
        opt = EfficiencyAdam([(16,)], lr=0.1); p = [np.zeros(16)]
        for g, r in zip(grads, ref):
            p, gamma = opt.step(p, [g])
            np.testing.assert_allclose(p[0], r, atol=1e-15)
            self.assertEqual(gamma[0], 1.0)

    def test_first_step_keeps_cross_layer_gradient_ratio(self):
        rng = np.random.default_rng(2)
        g = [3.3 * rng.standard_normal(64), 1.0 * rng.standard_normal(64)]
        opt = EfficiencyAdam([(64,), (64,)], lr=0.1)
        p, gamma = opt.step([np.zeros(64), np.zeros(64)], g)
        ratio = np.linalg.norm(p[0]) / np.linalg.norm(p[1])
        rms = np.sqrt(np.mean(g[0] ** 2)) / np.sqrt(np.mean(g[1] ** 2))
        self.assertAlmostEqual(ratio, rms, places=6)

    def test_lockstep_adam_spreads_budget_but_efficiency_adam_concentrates(self):
        rng = np.random.default_rng(3)
        g = [s * rng.standard_normal(256) for s in (3.3, 2.0, 1.5, 1.2, 1.0)]   # front layer most effective
        zero = [np.zeros(256) for _ in g]
        plain, _ = EfficiencyAdam([(256,)] * 5, lr=0.1, efficiency=False).step(zero, g)
        eff, _ = EfficiencyAdam([(256,)] * 5, lr=0.1).step(zero, g)
        xp, _, ap = project([-x for x in plain], 0.75)
        xe, _, ae = project([-x for x in eff], 0.75)
        sp = np.array([np.linalg.norm(x) for x in xp]); se = np.array([np.linalg.norm(x) for x in xe])
        self.assertTrue(ap.all())                                  # lockstep: every layer equally active
        self.assertLess(np.ptp(sp / sp.sum()), 1e-3)                # uniform shares (the v9-v11 pattern)
        self.assertEqual(int(np.argmax(se)), 0)                     # budget goes to the most effective layer
        self.assertGreater(se[0] / se.sum(), 0.9)


class Allocation(unittest.TestCase):
    def test_kkt_allocation_is_a_fixed_point_of_projected_steps(self):
        """Concave per-layer gains b_l (1 - exp(-<d_l,u_l>)) under the shared budget: active layers equalize
        marginal gain b_l exp(-x_l) = tau, weaker layers stay at zero."""
        b = np.array([3.0, 2.0, 1.2, 1.0, 0.8]); radius = 1.0; d = 4
        dirs = [np.eye(d)[0] for _ in b]
        # closed form: x_l = ln(b_l / tau) on the active set with sum x_l = radius
        for k in range(len(b), 0, -1):
            tau = np.exp((np.sum(np.log(b[:k])) - radius) / k)
            if np.all(b[:k] > tau) and (k == len(b) or b[k] <= tau):
                break
        xs = np.where(b > tau, np.log(np.maximum(b, 1e-300) / tau), 0.0)
        u = [x * dl for x, dl in zip(xs, dirs)]
        grad = [-bl * np.exp(-float(dl @ ul)) * dl for bl, dl, ul in zip(b, dirs, u)]
        for alpha in (1e-3, 1e-1, 1.0):
            moved, _, _ = project([ul - alpha * gl for ul, gl in zip(u, grad)], radius)
            for a, c in zip(moved, u):
                np.testing.assert_allclose(a, c, atol=1e-10)
        self.assertEqual(int(np.sum(xs > 0)), int(np.sum(b > tau)))


if __name__ == '__main__':
    unittest.main()
