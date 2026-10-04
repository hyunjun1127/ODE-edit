"""Analytic witnesses for the diagnostic contract; no optimizer or model."""

import json
import unittest

import numpy as np

from kkt import kkt_diagnostics


class KKTDiagnosticsTests(unittest.TestCase):
    def test_exact_boundary_kkt_with_active_and_inactive_groups(self):
        result = kkt_diagnostics(
            [np.array([-0.45, -0.60]), np.zeros(2)],
            [np.array([3.0, 4.0]), np.array([1.0, 2.0])],
            radius=0.75, decay=0.1,
        )
        self.assertAlmostEqual(result["mu"], 4.9)
        self.assertEqual(result["active"], [True, False])
        self.assertEqual(result["signed_radial_prices"], [4.9, None])
        self.assertLess(result["max_residual"], 1e-14)
        np.testing.assert_allclose(result["active_vector_residuals"][0], [0, 0], atol=1e-14)
        self.assertIsNone(result["active_vector_residuals"][1])
        json.dumps(result, allow_nan=False)

    def test_inactive_nonsmooth_inequality_including_equality(self):
        result = kkt_diagnostics(
            [np.zeros(2), np.zeros(2), np.zeros(1)],
            [np.array([0.3, 0.4]), np.array([0.6, 0.8]), np.array([0.2])],
            radius=2, decay=0.5,
        )
        self.assertEqual(result["mu"], 0)
        np.testing.assert_allclose(result["inactive_inequality_excess"], [0, 0.5, 0])
        self.assertAlmostEqual(result["max_residual"], 0.5)

    def test_norm_equality_misses_angular_stationarity(self):
        result = kkt_diagnostics([np.array([1.0, 0.0])],
                                 [np.array([0.0, 1.0])], radius=1, decay=1)
        self.assertEqual(result["norm_only_residuals"], [0.0])
        self.assertEqual(result["signed_radial_prices"], [-1.0])
        np.testing.assert_allclose(result["active_vector_residuals"], [[1, 1]])
        self.assertAlmostEqual(result["active_residual_norms"][0], np.sqrt(2))

    def test_interior_mu_zero_even_with_positive_radial_price(self):
        result = kkt_diagnostics([np.array([0.5])], [np.array([-3.0])],
                                 radius=1, decay=1)
        self.assertEqual(result["mu"], 0)
        self.assertEqual(result["signed_radial_prices"], [2.0])
        self.assertEqual(result["active_vector_residuals"], [[-2.0]])
        exact = kkt_diagnostics([np.array([0.5])], [np.array([-1.0])],
                                radius=1, decay=1)
        self.assertEqual(exact["max_residual"], 0)

    def test_boundary_mu_averages_prices_and_reports_all_active_vectors(self):
        result = kkt_diagnostics([np.array([0.4]), np.array([[0.6]])],
                                 [np.array([-2.0]), np.array([[-4.0]])],
                                 radius=1, decay=0.5)
        self.assertEqual(result["mu"], 2.5)
        self.assertEqual(result["signed_radial_prices"], [1.5, 3.5])
        self.assertEqual(result["active_vector_residuals"], [[1.0], [[-1.0]]])
        self.assertEqual(result["active_residual_norms"], [1.0, 1.0])

    def test_near_zero_classification_does_not_mutate_or_prune(self):
        blocks = [np.array([5e-13, 0.0]), np.array([[0.2]])]
        grads = [np.zeros(2), np.array([[-1.0]])]
        originals = [value.copy() for value in blocks + grads]
        result = kkt_diagnostics(blocks, grads, radius=1, decay=1)
        self.assertEqual(result["active"], [False, True])
        self.assertEqual(result["near_zero_nonzero"], [True, False])
        self.assertEqual(result["layer_norms"][0], 5e-13)
        self.assertEqual(result["budget_used"], 0.2 + 5e-13)
        for actual, original in zip(blocks + grads, originals):
            np.testing.assert_array_equal(actual, original)
        exact_nonzero = kkt_diagnostics(blocks, grads, radius=1, decay=1, active_tol=0)
        self.assertEqual(exact_nonzero["active"], [True, True])
        self.assertEqual(exact_nonzero["active_residual_norms"][0], 1)

    def test_zero_radius_degenerate_convention(self):
        result = kkt_diagnostics([np.zeros(2), np.zeros((1, 1))],
                                 [np.array([3.0, 4.0]), np.array([[2.0]])],
                                 radius=0, decay=0.1)
        self.assertEqual(result["mu_convention"], "zero_radius_degenerate")
        self.assertAlmostEqual(result["mu"], 4.9)
        self.assertEqual(result["max_residual"], 0)
        self.assertEqual(result["signed_radial_prices"], [None, None])

    def test_primal_violation_and_boundary_tolerance(self):
        infeasible = kkt_diagnostics([np.array([2.0])], [np.array([-3.0])],
                                     radius=1, decay=0)
        self.assertEqual(infeasible["primal_violation"], 1)
        self.assertEqual(infeasible["mu"], 0)
        near_boundary = kkt_diagnostics([np.array([1 + 5e-13])], [np.array([-3.0])],
                                        radius=1, decay=0, boundary_tol=1e-12)
        self.assertEqual(near_boundary["mu"], 3)
        self.assertGreater(near_boundary["primal_violation"], 0)
        self.assertAlmostEqual(near_boundary["complementarity"],
                               3 * near_boundary["primal_violation"])
        self.assertEqual(near_boundary["dual_violation"], 0)

    def test_invalid_scalars(self):
        for name in ("radius", "decay", "active_tol", "boundary_tol"):
            for value in (-1, np.nan, np.inf, -np.inf, 1j, [1], None):
                with self.subTest(name=name, value=value):
                    kwargs = dict(radius=1, decay=0, active_tol=1e-12, boundary_tol=1e-12)
                    kwargs[name] = value
                    with self.assertRaises(ValueError):
                        kkt_diagnostics([np.ones(1)], [np.zeros(1)], **kwargs)

    def test_invalid_arrays_counts_and_shapes(self):
        cases = [([], []), ([np.ones(1)], []),
                 ([np.ones(1)], [np.zeros(1), np.zeros(1)]),
                 ([np.ones(2)], [np.zeros((1, 2))]),
                 ([np.array(1.0)], [np.array(0.0)]),
                 ([np.array([])], [np.array([])]),
                 ([np.array([np.nan])], [np.zeros(1)]),
                 ([np.ones(1)], [np.array([np.inf])]),
                 ([np.array([1j])], [np.zeros(1)]),
                 (None, [np.zeros(1)])]
        for blocks, grads in cases:
            with self.subTest(blocks=blocks, grads=grads):
                with self.assertRaises(ValueError):
                    kkt_diagnostics(blocks, grads, radius=1, decay=0)


if __name__ == "__main__":
    unittest.main()
