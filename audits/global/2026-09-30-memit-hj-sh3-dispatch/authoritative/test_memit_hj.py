"""CPU checks for the MEMIT-HJ reference (NumPy only).

Run: python3 -m unittest project.run_scripts.memit_hj.test_memit_hj -v
"""
import unittest

import numpy as np

from .allocation import (capacity, capacity_from_adj, ainv_k, joint_objective, joint_ols, layer_shares,
                         memit_adj, min_cost_realization, resid_divisor, resid_joint, sequential, update)
from .drift import context_dispersion, rebuild_history, refresh_needed, relative_drift
from .zsolve import exact_z, native_adam, spg_ball


def fixture(seed=0, d_in=12, d_out=5, B=3, n=5, lam=3.0, hist=True):
    rng = np.random.default_rng(seed)
    As, Ks = [], []
    for _ in range(n):
        X = rng.standard_normal((d_in, 3 * d_in))
        C0 = X @ X.T / X.shape[1]
        H = np.zeros((d_in, d_in))
        if hist:
            P = rng.standard_normal((d_in, 4))
            H = P @ P.T
        As.append(lam * C0 + H)
        Ks.append(rng.standard_normal((d_in, B)))
    R = rng.standard_normal((d_out, B))
    return R, Ks, As


class Allocation(unittest.TestCase):
    def test_woodbury_adj_and_capacity(self):
        R, Ks, As = fixture()
        for K, A in zip(Ks, As):
            AK = ainv_k(A, K)
            G = capacity(K, AK)
            adj = memit_adj(A, K)
            np.testing.assert_allclose(adj, AK @ np.linalg.inv(np.eye(G.shape[0]) + G), rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(capacity_from_adj(K, adj), G, rtol=1e-9, atol=1e-12)
            self.assertGreater(np.linalg.eigvalsh(G).min(), 0)

    def test_joint_ols_is_the_minimizer(self):
        R, Ks, As = fixture(1)
        sol = joint_ols(R, Ks, As)
        base = joint_objective(R, Ks, As, sol["deltas"])
        fit = sum(D @ K for D, K in zip(sol["deltas"], Ks)) - R
        for D, K, A in zip(sol["deltas"], Ks, As):  # stationarity
            np.testing.assert_allclose(fit @ K.T + D @ A, 0, atol=1e-10)
        rng = np.random.default_rng(2)
        for _ in range(20):
            pert = [D + 1e-3 * rng.standard_normal(D.shape) for D in sol["deltas"]]
            self.assertGreater(joint_objective(R, Ks, As, pert), base)
        div = sequential(R, Ks, As, mode="divisor")
        self.assertLessEqual(base, joint_objective(R, Ks, As, div["deltas"]) + 1e-12)

    def test_sequential_equals_joint_under_additivity(self):
        R, Ks, As = fixture(3)
        sol = joint_ols(R, Ks, As)
        seq = sequential(R, Ks, As, mode="joint")
        for a, b in zip(seq["deltas"], sol["deltas"]):
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(seq["final_residual"], sol["unrealized"], rtol=1e-9, atol=1e-12)

    def test_unrealized_and_displacements(self):
        R, Ks, As = fixture(4)
        sol = joint_ols(R, Ks, As)
        G = sum(sol["capacities"])
        np.testing.assert_allclose(R - sum(sol["displacements"]), R @ np.linalg.inv(np.eye(G.shape[0]) + G),
                                   rtol=1e-10, atol=1e-12)

    def test_top_layer_is_memit_h(self):
        R, Ks, As = fixture(5)
        G = capacity(Ks[-1], ainv_k(As[-1], Ks[-1]))
        np.testing.assert_allclose(resid_joint(R, G, np.zeros_like(G)), resid_divisor(R, 4, 5), rtol=1e-12)

    def test_divisor_rule_is_the_large_uniform_capacity_limit(self):
        rng = np.random.default_rng(6)
        d_in, B, n = 10, 3, 5
        Q, _ = np.linalg.qr(rng.standard_normal((d_in, B)))
        R = rng.standard_normal((4, B))
        for kappa, tol in ((1.0, None), (1e8, 1e-6)):
            K = np.sqrt(kappa) * Q  # A = I  =>  G = kappa I for every layer
            A = np.eye(d_in)
            G = capacity(K, ainv_k(A, K))
            adj = memit_adj(A, K)
            for i in range(n):
                ours = update(resid_joint(R, G, (n - i - 1) * G), adj)
                memit = update(resid_divisor(R, i, n), adj)
                ratio = np.linalg.norm(ours) / np.linalg.norm(memit)
                expected = (n - i) * (1 + kappa) / (1 + (n - i) * kappa)
                self.assertAlmostEqual(ratio, expected, places=9)
                if tol is not None:
                    self.assertLess(abs(ratio - 1), tol)

    def test_single_request_shares(self):
        R, Ks, As = fixture(7, B=1)
        sol = joint_ols(R, Ks, As)
        kap = np.array([G[0, 0] for G in sol["capacities"]])
        frac = layer_shares(sol["capacities"])["realized_fraction"]
        np.testing.assert_allclose(frac, kap / (1 + kap.sum()), rtol=1e-12)
        for D, f in zip(sol["displacements"], frac):
            np.testing.assert_allclose(D, R * f, rtol=1e-9, atol=1e-12)

    def test_min_cost_realization(self):
        rng = np.random.default_rng(8)
        _, Ks, As = fixture(8, B=1)
        k, A = Ks[0][:, 0], As[0]
        d = rng.standard_normal(5)
        D, kappa = min_cost_realization(d, k, A)
        np.testing.assert_allclose(D @ k, d, rtol=1e-10)
        cost = np.trace(D @ A @ D.T)
        self.assertAlmostEqual(cost, float(d @ d) / kappa, places=9)
        for _ in range(10):  # any other exact realization costs more
            E = rng.standard_normal(D.shape)
            E = E - np.outer(E @ k, k) / float(k @ k)
            self.assertGreater(np.trace((D + E) @ A @ (D + E).T), cost)

    def test_w0_shares_do_not_depend_on_lambda(self):
        R, Ks, As1 = fixture(9, lam=1.0, hist=False)
        _, _, As2 = fixture(9, lam=50.0, hist=False)
        s1 = layer_shares(joint_ols(R, Ks, As1)["capacities"])["capacity_share"]
        s2 = layer_shares(joint_ols(R, Ks, As2)["capacities"])["capacity_share"]
        np.testing.assert_allclose(s1, s2, rtol=1e-10)

    def test_history_moves_load_away(self):
        R, Ks, As = fixture(10, hist=False)
        before = layer_shares(joint_ols(R, Ks, As)["capacities"])["capacity_share"]
        As2 = list(As)
        As2[0] = As[0] + 50.0 * Ks[0] @ Ks[0].T  # layer 4 already stores keys like these
        after = layer_shares(joint_ols(R, Ks, As2)["capacities"])["capacity_share"]
        self.assertLess(after[0], before[0])
        self.assertTrue(np.all(after[1:] > before[1:]))


class ZStep(unittest.TestCase):
    @staticmethod
    def shrink_problem(c, w):
        def fg(x):
            n = np.linalg.norm(x)
            return 0.5 * float((x - c) @ (x - c)) + w * n, (x - c) + (w * x / n if n > 0 else 0 * x)
        return fg

    def test_interior_optimum(self):
        c = np.array([1.0, 2.0, -0.5])
        w, r = 0.3, 10.0
        x, info = spg_ball(self.shrink_problem(c, w), np.zeros(3), r)
        np.testing.assert_allclose(x, c * (1 - w / np.linalg.norm(c)), atol=1e-7)
        self.assertTrue(info["converged"])
        self.assertFalse(info["active"])
        self.assertEqual(info["clamp_multiplier"], 0.0)

    def test_boundary_optimum_multiplier(self):
        c = np.array([3.0, 4.0])
        w, r = 0.5, 2.0
        x, info = spg_ball(self.shrink_problem(c, w), np.zeros(2), r)
        np.testing.assert_allclose(x, c * r / 5.0, atol=1e-7)
        self.assertTrue(info["active"])
        self.assertAlmostEqual(info["clamp_multiplier"], 5.0 - r - w, places=6)

    def test_zero_step_rule(self):
        x, info = exact_z(lambda z: (0.01 + float(z @ z), 2 * z), 4, 1.0)
        self.assertTrue(info["zero_step"])
        self.assertEqual(float(np.linalg.norm(x)), 0.0)

    def test_native_loop_pins_to_clamp_while_optimum_is_interior(self):
        """Toy with the logged scales: NLL(0)~6 nats, ||h||=3.15, decay 0.5||d||/||h||^2, clamp 0.75||h||."""
        rng = np.random.default_rng(0)
        d = 4096
        a = rng.standard_normal(d)
        a = 5.5 * a / np.linalg.norm(a)
        b, h = 6.0, 3.15
        c, r = 0.5 / h ** 2, 0.75 * h

        def fg(x):
            z = b - a @ x
            n = np.linalg.norm(x)
            return (np.logaddexp(0.0, z) + c * n,
                    -a / (1.0 + np.exp(-z)) + (c * x / n if n > 0 else 0 * x))

        xa, ia = native_adam(fg, np.zeros(d), r)
        xs, iz = exact_z(fg, d, r)
        t_star = (b - np.log(c / (5.5 - c))) / 5.5
        self.assertEqual((ia["adam_steps"], ia["clamp_hits"], ia["stopped_by"]), (24, 24, "cap"))
        self.assertAlmostEqual(ia["norm_over_radius"], 1.0, places=9)
        self.assertTrue(iz["converged"] and not iz["active"])
        self.assertAlmostEqual(float(np.linalg.norm(xs)), t_star, places=5)
        self.assertLess(iz["value"], ia["value"])


class History(unittest.TestCase):
    def test_refresh_rule(self):
        rng = np.random.default_rng(11)
        C, d, N = 6, 20, 50
        kbar = rng.standard_normal((d, N))
        ctx = kbar[None] + 0.2 * rng.standard_normal((C, d, N))
        disp = context_dispersion(ctx)
        old = ctx.mean(axis=0)
        small = old + 0.02 * rng.standard_normal((d, N))
        large = old + 0.5 * rng.standard_normal((d, N))
        self.assertFalse(refresh_needed(relative_drift(old, small), disp))
        self.assertTrue(refresh_needed(relative_drift(old, large), disp))

    def test_rebuild_history(self):
        rng = np.random.default_rng(12)
        K = rng.standard_normal((8, 30))
        np.testing.assert_allclose(rebuild_history(K), sum(np.outer(K[:, j], K[:, j]) for j in range(30)),
                                   rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
