import unittest

import numpy as np

from math_contract import native_norm_u, prox_native_balls, radial_price, ridge_scalar, routed_witness


MEASUREMENTS = {}


class CalibrationMathTests(unittest.TestCase):
    def close(self, actual, expected, atol=1e-10, rtol=1e-10):
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)

    def test_history_can_increase_then_decrease_ridge_Q(self):
        # A=1/7+H, key=1: increasing H makes kappa=1/A decrease.
        kappas = [7., 1., .25]
        rows = [ridge_scalar(1., k) for k in kappas]
        q = [r["Q"] for r in rows]
        self.close(q, [.109375, .25, .16])
        self.assertLess(q[0], q[1]); self.assertGreater(q[1], q[2])
        self.assertGreater(rows[0]["realized"], rows[1]["realized"])
        self.assertGreater(rows[1]["realized"], rows[2]["realized"])
        self.assertLess(rows[0]["ridge_optimum_value"], rows[1]["ridge_optimum_value"])
        self.assertLess(rows[1]["ridge_optimum_value"], rows[2]["ridge_optimum_value"])
        MEASUREMENTS["history_counterexample"] = {"kappa": kappas, "Q": q,
            "E_plus_Q": [r["ridge_optimum_value"] for r in rows],
            "exact_energy": [r["exact_energy"] for r in rows]}

    def test_scalar_ridge_derivative(self):
        errors = []
        for k in (.2, .7, 1., 3., 7.):
            eps = 1e-6
            finite = (ridge_scalar(1.3, k + eps)["Q"] - ridge_scalar(1.3, k - eps)["Q"]) / (2 * eps)
            analytic = ridge_scalar(1.3, k)["dQ_dkappa"]
            self.close(finite, analytic, atol=2e-9)
            errors.append(abs(finite - analytic))
        MEASUREMENTS["ridge_Q_derivative_max_error"] = max(errors)

    def test_fixed_key_euler_identity_and_radial_price(self):
        rng = np.random.default_rng(20261006)
        R = rng.normal(size=(3, 4))
        T = rng.normal(size=(4, 4)); F = T @ T.T
        Q = float(np.sum((R @ F) * R)); gQ = 2 * R @ F
        self.close(np.sum(R * gQ), 2 * Q)
        # Native group norm is degree one, with per-request physical prices.
        lengths = np.linalg.norm(R, axis=0); beta = np.array([.1, .2, .3, .4])
        gN = R / lengths * beta
        N = float(np.sum(beta * lengths)); self.close(np.sum(R * gN), N)
        chosen_price = .37
        gF = -gN - chosen_price * gQ
        result = radial_price(R, gF, gN, gQ)
        self.assertEqual(result["status"], "POSITIVE_RADIAL_BREAK_EVEN")
        self.close(result["lambda_candidate"], chosen_price)
        self.close(result["aggregate_radial_residual"], 0)

    def test_radial_balance_does_not_establish_full_KKT(self):
        R = np.array([[1.], [0.]])
        result = radial_price(R, np.array([[-3.], [1.]]), np.array([[.5], [0.]]), 2 * R)
        self.close(result["lambda_candidate"], 1.25)
        self.close(result["aggregate_radial_residual"], 0)
        self.close(result["full_gradient_norm"], 1.)
        # At this ball boundary the nonzero y component is a feasible tangent.
        MEASUREMENTS["radially_balanced_nonzero_tangent_gradient"] = result["full_gradient_norm"]

    def test_boundary_has_nonzero_KKT_multiplier_before_break_even(self):
        R = np.ones((1, 1)); gF = np.array([[-3.]]); gN = np.array([[.5]]); gQ = 2 * R
        limit = radial_price(R, gF, gN, gQ)["lambda_candidate"]
        # For F=-3r, N=.5r, Q=r^2, r<=1, all prices below the break-even
        # value keep the upper boundary optimal, with a positive multiplier.
        for coefficient in (0., .4, 1.):
            derivative = float((gF + gN + coefficient * gQ)[0, 0])
            self.assertLess(coefficient, limit)
            self.assertLess(derivative, 0)
        self.close((gF + gN + limit * gQ)[0, 0], 0)

    def test_native_subject_reference_can_give_negative_actual_price(self):
        # Native virtual .5*(r-3)^2+.5*r minimizes at clamp r=1.
        # Actual ridge gamma=.1 changes the derivative at that same r.
        R = np.ones((1, 1)); gamma = .1; kappa = gamma / (1 - gamma)
        gF_virtual = np.array([[-2.]])
        gF_actual = np.array([[gamma * (gamma - 3.)]])
        gN = np.array([[.5]])
        q = ridge_scalar(1., kappa)["Q"]; gQ = np.array([[2 * q]])
        virtual = radial_price(R, gF_virtual, gN, gQ)
        actual = radial_price(R, gF_actual, gN, gQ)
        self.assertTrue(virtual["positive_price"])
        self.assertFalse(actual["positive_price"])
        self.assertEqual(actual["status"], "NEGATIVE_RADIAL_PRICE")
        self.close(actual["lambda_candidate"], -7 / 6)
        MEASUREMENTS["native_vs_actual_reference_price_witness"] = {
            "virtual_price": virtual["lambda_candidate"], "actual_price": actual["lambda_candidate"]}

    def test_aggregate_price_does_not_balance_individual_requests(self):
        R = np.ones((1, 2))
        result = radial_price(R, np.array([[-3., -1.]]), np.array([[.5, .5]]), 2 * R)
        self.close(result["lambda_candidate"], .75)
        self.close(result["aggregate_radial_residual"], 0)
        self.close(result["radial_residual_per_request"], [-1., 1.])
        MEASUREMENTS["aggregate_balanced_request_radial_residuals"] = result["radial_residual_per_request"]

    def test_zero_interior_and_degenerate_prices_are_explicit(self):
        z = np.zeros((1, 1)); one = np.ones((1, 1))
        self.assertEqual(radial_price(z, z, z, z)["status"], "ZERO_REFERENCE")
        self.assertEqual(radial_price(one, -one, one, z)["status"], "NONPOSITIVE_RADIAL_Q_DENOMINATOR")
        interior = radial_price(one, -.5 * one, .5 * one, 2 * one)
        self.assertEqual(interior["status"], "ZERO_RADIAL_PRICE")
        self.assertEqual(interior["lambda_candidate"], 0)

    def test_native_norm_coordinate_conversion(self):
        u = np.array([[.3, .2], [.4, -.1]])
        anchors = np.array([2., 4.]); R = u * anchors
        value, gradient = native_norm_u(u, anchors)
        self.close(value, np.sum(.5 * np.linalg.norm(R, axis=0) / anchors ** 2))
        physical_gradient = .5 * R / np.linalg.norm(R, axis=0) / anchors ** 2
        self.close(gradient, physical_gradient * anchors)
        self.close(np.linalg.norm(gradient, axis=0), .5 / anchors)

    def test_proximal_mapping_matches_radial_minimum_and_caps(self):
        proposal = np.array([[.4, 1.2, 0.], [0., 1.6, 0.]])
        anchors = np.array([4., 2., 1.]); step = 1.; cap = .75
        out = prox_native_balls(proposal, anchors, step, cap)
        self.close(np.linalg.norm(out, axis=0), [.275, .75, 0.])
        grid = np.linspace(0, cap, 10001)
        for col in range(proposal.shape[1]):
            length = np.linalg.norm(proposal[:, col]); beta = .5 / anchors[col]
            optimum = np.linalg.norm(out[:, col])
            objective = lambda r: .5 / step * (r - length) ** 2 + beta * r
            self.assertLessEqual(objective(optimum), float(np.min(objective(grid))) + 1e-12)
        # Wrong /a^2 threshold would yield .36875 instead of .275.
        self.assertNotAlmostEqual(out[0, 0], .4 - .5 / 4 ** 2)

    def test_exact_zero_and_reentry_without_layer_pruning(self):
        anchor = np.array([2.]); step = .1
        zero = prox_native_balls(np.array([[.02], [0.]]), anchor, step)
        self.close(zero, 0)  # Norm=.02 below threshold .025.
        next_proposal = zero - step * np.array([[-1.], [0.]])
        reentry = prox_native_balls(next_proposal, anchor, step)
        self.close(reentry, [[.075], [0.]])

    def test_routed_gradient_is_not_full_gradient_or_true_KKT(self):
        point = np.array([1., .5])
        _, full, route = routed_witness(point)
        self.close(route, 0)
        self.close(full, [.5, 0.])
        eps = 1e-6
        fd = (routed_witness(point + [eps, 0])[0] - routed_witness(point - [eps, 0])[0]) / (2 * eps)
        self.close(fd, full[0], atol=1e-9)
        # Routed mixed partials differ: dg_x/dy=0, dg_y/dx=-1.
        droute_dy = (routed_witness(point + [0, eps])[2] - routed_witness(point - [0, eps])[2]) / (2 * eps)
        droute_dx = (routed_witness(point + [eps, 0])[2] - routed_witness(point - [eps, 0])[2]) / (2 * eps)
        self.close(droute_dy[0], 0); self.close(droute_dx[1], -1, atol=1e-9)
        MEASUREMENTS["routed_stationarity_true_gradient_norm"] = float(np.linalg.norm(full))
        MEASUREMENTS["routed_mixed_partials"] = [float(droute_dy[0]), float(droute_dx[1])]


if __name__ == "__main__":
    unittest.main(verbosity=2)
