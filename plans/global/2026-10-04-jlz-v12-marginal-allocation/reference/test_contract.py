"""Additional contract and limitation checks; no model or GPU dependencies."""

import unittest

import numpy as np
from scipy.optimize import minimize

from budget import project
from optimizer import EfficiencyAdam, objective_gradients
from witnesses import non_kkt_witness, sparse_first_step_witness


class ProjectionContract(unittest.TestCase):
    def test_heterogeneous_groups_against_independent_solve(self):
        rng = np.random.default_rng(14)
        shapes = [(1,), (3,), (2, 2)]
        blocks = [rng.normal(size=s) for s in shapes]
        weights = np.array([0.8, 1.2, 2.0])
        radius = 1.1
        result, tau, active = project(blocks, radius, weights)
        lengths = [b.size for b in blocks]
        cuts = np.cumsum([0] + lengths)
        budget = lambda x: sum(weights[i] * np.linalg.norm(x[cuts[i]:cuts[i + 1]]) for i in range(len(blocks)))
        norms = np.array([np.linalg.norm(b) for b in blocks])
        # An independent constrained QP in group radii avoids differentiating
        # the full-vector norm at zero groups, where SLSQP can stall.
        independent = minimize(lambda x: 0.5 * np.sum((x - norms) ** 2),
                               np.zeros(len(blocks)), jac=lambda x: x - norms,
                               bounds=[(0, None)] * len(blocks), method="SLSQP",
                               constraints=[{"type": "ineq", "fun": lambda x: radius - weights @ x,
                                             "jac": lambda x: -weights}],
                               options={"ftol": 1e-12, "maxiter": 500})
        self.assertTrue(independent.success, independent.message)
        expected = [b * n / norm for b, n, norm in zip(blocks, independent.x, norms)]
        np.testing.assert_allclose(np.concatenate([b.ravel() for b in result]), np.concatenate([b.ravel() for b in expected]), atol=2e-8)
        self.assertAlmostEqual(budget(np.concatenate([b.ravel() for b in result])), radius)
        self.assertGreater(tau, 0)
        self.assertEqual([b.shape for b in result], shapes)
        self.assertTrue(active.any())

    def test_zero_radius_all_zero_and_no_pruning(self):
        result, tau, active = project([np.array([3., 4.]), np.ones(1)], 0, [2., 1.])
        self.assertEqual(tau, 2.5)
        self.assertFalse(active.any())
        self.assertTrue(all(np.count_nonzero(b) == 0 for b in result))
        zeros = [np.zeros(1), np.zeros((2, 2))]
        result, tau, active = project(zeros, 0)
        self.assertEqual(tau, 0)
        self.assertFalse(active.any())
        tiny = np.array([1e-13])
        result, _, active = project([tiny], 1)
        self.assertTrue(active[0])
        np.testing.assert_array_equal(result[0], tiny)

    def test_invalid_projection_inputs(self):
        cases = [(None, 1, None), ([], 1, None), ([np.zeros(0)], 1, None), ([np.array(1.)], 1, None),
                 ([np.array([1j])], 1, None), ([np.ones(2)], None, None), ([np.ones(2)], 1j, None),
                 ([np.array([np.nan])], 1, None), ([np.array([np.inf])], 1, None),
                 ([np.ones(2)], -1, None), ([np.ones(2)], np.inf, None), ([np.ones(2)], np.nan, None),
                 ([np.ones(2)], 1, [0]), ([np.ones(2)], 1, [-1]), ([np.ones(2)], 1, [np.inf]),
                 ([np.ones(2)], 1, [1, 2]), ([np.ones(2)], 1, [[1]]), ([np.ones(2)], 1, [1j])]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                project(*args)


class OptimizerContract(unittest.TestCase):
    def test_native_delta_parity_including_scaled_epsilon_and_clamp(self):
        rng = np.random.default_rng(9)
        for anchor in (0.2, 1., 5., 20.):
            with self.subTest(anchor=anchor):
                native = EfficiencyAdam([(8,)], lr=0.1, efficiency=False)
                relative = EfficiencyAdam.from_native([(8,)], anchor)
                delta, u = [np.zeros(8)], [np.zeros(8)]
                for scale in (1e-9, 1e-12, 1., 1e-8, 0., 2.):
                    g_delta = rng.normal(size=8) * scale
                    delta, _ = native.step(delta, [g_delta])
                    delta, _, _ = project(delta, .75 * anchor)
                    u, gamma = relative.step(u, [anchor * g_delta])
                    u, _, _ = project(u, .75)
                    np.testing.assert_allclose(anchor * u[0], delta[0], rtol=2e-13, atol=2e-15)
                    self.assertEqual(gamma[0], 1.)

    def test_unscaled_epsilon_really_breaks_parity(self):
        native = EfficiencyAdam([(1,)], lr=.1)
        wrong = EfficiencyAdam([(1,)], lr=.1 / 5)
        delta, _ = native.step([np.zeros(1)], [np.array([1e-9])])
        u, _ = wrong.step([np.zeros(1)], [np.array([5e-9])])
        self.assertGreater(abs(delta[0][0] - 5 * u[0][0]), .02)

    def test_smooth_gradient_plus_analytic_norm_subgradient(self):
        blocks = [np.array([3., 4.]), np.zeros(2), np.array([1e-14])]
        smooth = [np.array([2., -1.]), np.array([1., 3.]), np.array([2.])]
        combined = objective_gradients(blocks, smooth, .5)
        np.testing.assert_allclose(combined[0], [2.3, -.6])
        np.testing.assert_array_equal(combined[1], smooth[1])
        np.testing.assert_allclose(combined[2], [2.5])
        h = 1e-6
        value = lambda u: smooth[0] @ u + .5 * np.linalg.norm(u)
        central = np.array([(value(blocks[0] + h * e) - value(blocks[0] - h * e)) / (2 * h) for e in np.eye(2)])
        np.testing.assert_allclose(combined[0], central, atol=1e-9)
        np.testing.assert_array_equal(blocks[1], np.zeros(2))

    def test_all_zero_gradients_stay_finite(self):
        opt = EfficiencyAdam([(2,), (1,)], lr=.1)
        zero = [np.zeros(2), np.zeros(1)]
        for _ in range(3):
            result, gamma = opt.step(zero, zero)
            np.testing.assert_array_equal(gamma, [1, 1])
            for got, expected in zip(result, zero):
                np.testing.assert_array_equal(got, expected)

    def test_projected_zero_layer_can_reenter_with_retained_moments(self):
        opt = EfficiencyAdam([(1,), (1,)], lr=1.)
        u, _ = opt.step([np.zeros(1), np.zeros(1)], [np.array([10.]), np.array([.001])])
        u, _, active = project(u, .75)
        self.assertFalse(active[1])
        self.assertGreater(opt.m[1][0], 0.)
        next_u, _ = opt.step(u, [np.zeros(1), np.array([100.])])
        next_u, _, active = project(next_u, .75)
        self.assertTrue(active[1])
        self.assertEqual(opt.t, 2)
        self.assertGreater(opt.m[1][0], 10.)

    def test_independent_request_moments_under_interleaved_schedule(self):
        rng = np.random.default_rng(41)
        grads = {r: [[rng.normal(size=2), rng.normal(size=3)] for _ in range(n)] for r, n in [("a", 4), ("b", 2)]}
        def run(schedule):
            optimizers = {r: EfficiencyAdam.from_native([(2,), (3,)], anchor) for r, anchor in [("a", 5.), ("b", 3.)]}
            values = {r: [np.zeros(2), np.zeros(3)] for r in grads}
            for r in schedule:
                opt = optimizers[r]
                values[r], _ = opt.step(values[r], grads[r][opt.t])
                values[r], _, _ = project(values[r], .75)
            return optimizers, values
        sequential, seq_values = run("aaaabb")
        interleaved, int_values = run("ababaa")
        for r in grads:
            self.assertEqual(sequential[r].t, interleaved[r].t)
            for a, b in zip(seq_values[r] + sequential[r].m + sequential[r].v, int_values[r] + interleaved[r].m + interleaved[r].v):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(sequential[r].s, interleaved[r].s)

    def test_invalid_optimizer_and_gradient_inputs(self):
        for kwargs in ({"lr": -1}, {"lr": np.inf}, {"eps": 0}, {"eps": np.nan}, {"betas": (1., .9)}, {"betas": (-.1, .9)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EfficiencyAdam([(2,)], **({"lr": .1} | kwargs))
        for shapes in ([], [()], [(0,)], [(2.5,)]):
            with self.subTest(shapes=shapes), self.assertRaises(ValueError):
                EfficiencyAdam(shapes, .1)
        for anchor in (0, -1, np.inf):
            with self.assertRaises(ValueError):
                EfficiencyAdam.from_native([(2,)], anchor)
        opt = EfficiencyAdam([(2,)], .1)
        for params, grads in [([np.zeros(3)], [np.zeros(3)]), ([np.zeros(2)], []), ([np.zeros(2)], [np.array([np.nan, 0])])]:
            with self.assertRaises(ValueError):
                opt.step(params, grads)
            self.assertEqual(opt.t, 0)
        with self.assertRaises(ValueError):
            objective_gradients([np.zeros(2)], [np.zeros(1)], .1)


class ExpectedLimitations(unittest.TestCase):
    def test_real_efficiency_adam_non_kkt_fixed_point(self):
        result = non_kkt_witness()
        self.assertEqual(result["status"], "EXPECTED_LIMITATION")
        self.assertLess(result["max_displacement_from_initial"], 1e-13)
        self.assertGreater(result["kkt"]["active_residual_norms"][0], .6)
        self.assertGreater(result["objective_gap"], .03)

    def test_sparse_gradient_step_ratio_limitation(self):
        result = sparse_first_step_witness()
        self.assertAlmostEqual(result["step_norm_ratio"], 1 / np.sqrt(2), places=7)


if __name__ == "__main__":
    unittest.main()
