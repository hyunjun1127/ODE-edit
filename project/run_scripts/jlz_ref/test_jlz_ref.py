"""CPU checks for the JLZ reference pipeline (NumPy only).

Run: python3 -m unittest project.run_scripts.jlz_ref.test_jlz_ref -v
"""
import unittest

import numpy as np

from .problem import JLZProblem, prox_blocks
from .prox import solve
from .toy import ToyLM
from .writer import write_batch

LAYERS = [2, 3, 4, 5, 6]


def setup(d=48, m=24, V=40, L=9, B=3, seed=0, h0_norm=2.0, tol=1e-8, cap=3000):
    model = ToyLM(d, m, V, L, seed=seed)
    rng = np.random.default_rng(seed + 1)
    X = rng.standard_normal((800, d))
    X *= h0_norm / np.linalg.norm(X, axis=1, keepdims=True)
    cache = model.forward(X)
    C0 = {l: cache["xs"][l].T @ cache["xs"][l] / len(X) for l in LAYERS}
    H = {l: np.zeros((m, m)) for l in LAYERS}
    ctx = np.vstack([np.zeros(d)] + [0.3 * h0_norm * rng.standard_normal(d) / np.sqrt(d) for _ in range(5)])
    batch = []
    for _ in range(B):
        h0 = rng.standard_normal(d)
        h0 *= h0_norm / np.linalg.norm(h0)
        logits = model.forward(h0[None])["logits"][0]
        batch.append(dict(h0=h0, h_kl=h0 + 0.2 * rng.standard_normal(d) / np.sqrt(d),
                          t_new=int(np.argsort(logits)[V // 2])))
    cfg = dict(layers=LAYERS, contexts=ctx, lam=float(m), tol=tol, cap=cap)
    return model, dict(C0=C0, H=H), batch, cfg


def problem(model, state, batch, cfg, **kw):
    return JLZProblem(model, batch, cfg["layers"], cfg["contexts"], state["C0"], state["H"], cfg["lam"], **kw)


class Pipeline(unittest.TestCase):
    def test_gradient_matches_finite_differences(self):
        model, state, batch, cfg = setup()
        prob = problem(model, state, batch, cfg)
        rng = np.random.default_rng(3)
        R = {l: 0.3 * rng.standard_normal((model.d, prob.B)) for l in LAYERS}
        f, G, _ = prob.smooth(R)
        for _ in range(5):
            D = {l: rng.standard_normal(R[l].shape) for l in LAYERS}
            e = 1e-6
            fp = prob.smooth({l: R[l] + e * D[l] for l in LAYERS})[0]
            fm = prob.smooth({l: R[l] - e * D[l] for l in LAYERS})[0]
            fd = (fp - fm) / (2 * e)
            an = sum(float(np.sum(G[l] * D[l])) for l in LAYERS)
            self.assertLess(abs(fd - an), 1e-6 * max(1.0, abs(an)))

    def test_prox_is_the_exact_radial_minimizer(self):
        rng = np.random.default_rng(4)
        d, t = 7, 0.3
        for c, rho in ((0.5, 10.0), (0.1, 0.4), (50.0, 1.0)):
            v = rng.standard_normal(d)
            p = prox_blocks(v, t, np.array([c]), np.array([rho]), d)
            u = v / np.linalg.norm(v)
            grid = np.linspace(0, rho, 20001)
            obj = 0.5 * (grid[:, None] * u - v) ** 2
            best = grid[np.argmin(obj.sum(1) / t + c * grid)]
            self.assertAlmostEqual(float(np.linalg.norm(p)), best, places=3)
            if np.linalg.norm(p) > 0:
                np.testing.assert_allclose(p / np.linalg.norm(p), u, atol=1e-12)

    def test_solver_converges_with_kkt_evidence(self):
        model, state, batch, cfg = setup()
        rec = write_batch(model, state, batch, cfg)
        self.assertEqual(rec["solver"]["status"], "CONVERGED")
        self.assertTrue(rec["solver"]["final_recomputed"])
        for k in rec["kkt"]:
            if k["state"] == "ZERO":
                self.assertLessEqual(k["subgradient_ratio"], 1 + 1e-6)
            else:
                self.assertLess(k["tangential_residual"], 1e-6)
                self.assertLess(k["radial_residual"], 1e-6)
                self.assertLessEqual(k["norm_over_radius"], 1 + 1e-9)
        self.assertTrue(any(k["state"] == "ZERO" for k in rec["kkt"]))  # norm decay yields exact zeros

    def test_commit_reproduces_the_fit_and_appends_history(self):
        model, state, batch, cfg = setup()
        W0 = [w.copy() for w in model.W]
        H0 = {l: state["H"][l].copy() for l in LAYERS}
        rec = write_batch(model, state, batch, cfg)
        self.assertEqual(rec["consistency"], 0.0)
        self.assertEqual(rec["lowest_layer_key_change"], 0.0)
        prob = rec["_problem"]
        for l in LAYERS:
            np.testing.assert_array_equal(model.W[l], W0[l] + rec["_deltas"][l])
        post = model.forward(prob.rw)
        for l in LAYERS:
            K = post["xs"][l].reshape(prob.B, prob.C, -1).mean(axis=1).T
            np.testing.assert_allclose(state["H"][l] - H0[l], K @ K.T, rtol=1e-12, atol=1e-12)

    def test_zero_step_requests_are_fixed_and_excluded(self):
        model, state, batch, cfg = setup()
        probe = problem(model, state, batch, cfg)
        stop = float(np.sort(probe.nll_entry)[0]) + 1e-9  # the easiest request falls under the stop rule
        cfg = dict(cfg, stop=stop)
        prob = problem(model, state, batch, cfg, stop=stop)
        r0 = int(np.argmin(prob.nll_entry))
        self.assertFalse(prob.active[r0])
        self.assertTrue(all(not prob.free[(l, r0)] for l in LAYERS))
        rec = write_batch(model, state, batch, cfg)
        self.assertEqual(rec["zero_step"], [r0])
        for l in LAYERS:
            self.assertEqual(float(np.linalg.norm(rec["_R"][l][:, r0])), 0.0)

    def test_requests_are_coupled_through_the_batch_solve(self):
        model, state, batch, cfg = setup()
        prob = problem(model, state, batch, cfg)
        R = prob.zeros()
        base = prob.terms(prob.deltas(R))[0][0]
        R[4][:, 1] = 0.5 * np.ones(model.d) / np.sqrt(model.d)
        moved = prob.terms(prob.deltas(R))[0][0]
        self.assertGreater(abs(moved - base), 1e-6)  # request 1's target moves request 0's loss

    def test_native_adam_pins_blocks_and_loses_to_prox(self):
        model, state, batch, cfg = setup(d=512, m=64, B=2, seed=3, tol=1e-4)
        ra = write_batch(model, state, batch, cfg, solver="native_adam")
        model2, state2, batch2, cfg2 = setup(d=512, m=64, B=2, seed=3, tol=1e-4)
        rp = write_batch(model2, state2, batch2, cfg2)
        ratios = np.array(ra["solver"]["norm_over_radius"])
        self.assertGreaterEqual(float(np.mean(ratios >= 0.98)), 0.7)
        self.assertGreater(ra["solver"]["value"], rp["solver"]["value"])

    def test_single_layer_reduction(self):
        model, state, batch, cfg = setup()
        rec = write_batch(model, state, batch, cfg, free_layers=[4])
        for l in LAYERS:
            if l != 4:
                self.assertEqual(float(np.linalg.norm(rec["_R"][l])), 0.0)
        model2, state2, batch2, cfg2 = setup()
        rec2 = write_batch(model2, state2, batch2, dict(cfg2, layers=[4]))
        np.testing.assert_allclose(rec["_R"][4], rec2["_R"][4], atol=1e-6)

    def test_solver_statuses_and_call_budget(self):
        model, state, batch, cfg = setup()
        prob = problem(model, state, batch, cfg)
        c, rho = prob.block_params()
        d = prob.d

        def fun(x):
            f, G, _ = prob.smooth(prob.unpack(x))
            return f, prob.pack(G)

        def h(x):
            return prob.decay(prob.unpack(x))

        def P(v, t):
            return prox_blocks(v, t, c, rho, d)

        x0 = np.zeros(len(prob.blocks) * d)
        _, _, info = solve(fun, h, P, x0, tol=1e-12, cap=5)
        self.assertEqual(info["status"], "NOT_CONVERGED")
        self.assertLessEqual(info["calls"], 5)

        def wrong(x):
            f, g = fun(x)
            return f, -g

        _, _, info = solve(wrong, h, P, x0, tol=1e-12, cap=500, max_trials=8)
        self.assertEqual(info["status"], "LINESEARCH_FAILED")


if __name__ == "__main__":
    unittest.main()
