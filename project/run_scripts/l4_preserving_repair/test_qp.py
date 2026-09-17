"""Small exact and many-row CPU fixtures, not actual-model validation."""
from __future__ import annotations

import json
import unittest
import numpy as np
from scipy.optimize import LinearConstraint, minimize

from .geometry import GeometryError, GeometryPolicy, orthonormal_basis, whiten_gn
from .qp import QPError, QPPolicy, kkt_receipt, solve_box_qp, solve_repair_qp


class GeometryTests(unittest.TestCase):
    def test_full_shape_gram_and_dependency(self):
        rng = np.random.default_rng(1701)
        a, b = rng.normal(size=(2, 17, 23))
        basis, receipt = orthonormal_basis({"B": a, "R": 2 * a, "H": b})
        self.assertEqual(len(basis), 2)
        self.assertEqual(receipt["input_status"][1]["status"], "NUMERICAL_DEPENDENCY")
        flat = np.stack([x.reshape(-1).astype(np.float64) for x in basis])
        np.testing.assert_allclose(flat @ flat.T, np.eye(2), atol=2e-7)
        self.assertEqual(basis[0].dtype, np.float32)

    def test_zero_gradient_and_declared_order(self):
        basis, receipt = orthonormal_basis({"B": np.zeros(6), "R": np.eye(1, 6)[0]})
        self.assertEqual(len(basis), 1)
        self.assertEqual(receipt["input_status"][0]["status"], "NUMERICAL_ZERO")
        np.testing.assert_array_equal(basis[0], [1, 0, 0, 0, 0, 0])

    def test_small_gradient_resolved_and_zero(self):
        basis, _ = orthonormal_basis({"B": np.array([2e-12, 0.0]), "R": np.array([0.0, 5e-13])})
        self.assertEqual(len(basis), 1)

    def test_no_global_gram_cancellation_for_near_dependency(self):
        a = np.ones(128)
        b = a.copy()
        b[0] += 1e-6
        basis, receipt = orthonormal_basis({"B": a, "R": b})
        self.assertEqual(len(basis), 2)
        self.assertGreater(receipt["input_status"][1]["relative_residual"], 1e-8)

    def test_basis_nonfinite_shape_dtype_and_scope(self):
        for values in ({"B": np.array([np.nan])},
                       {"B": np.zeros(2), "R": np.zeros(3)},
                       {"B": np.array([1], dtype=np.int64)},
                       {str(k): np.ones(2) for k in range(4)}):
            with self.assertRaises(GeometryError):
                orthonormal_basis(values)

    def test_chunk_size_does_not_change_span(self):
        rng = np.random.default_rng(17)
        gradients = dict(zip(("B", "R", "H"), rng.normal(size=(3, 7, 31))))
        q1, _ = orthonormal_basis(gradients, GeometryPolicy(chunk_elements=5))
        q2, _ = orthonormal_basis(gradients, GeometryPolicy(chunk_elements=512))
        np.testing.assert_allclose(q1, q2, atol=1e-7)

    def test_minimal_shift_and_whitening(self):
        H = np.diag([0.0, 2.0, 6.0])
        result = whiten_gn(H, [1, 2, 3])
        self.assertAlmostEqual(result["lambda_num"], 6.0 / (1e6 - 1))
        R = np.asarray(result["R"])
        np.testing.assert_allclose(R.T @ (H + result["lambda_num"] * np.eye(3)) @ R, np.eye(3), atol=1e-8)
        self.assertAlmostEqual(result["condition_number"], 1e6)

    def test_no_shift_for_resolved_psd(self):
        result = whiten_gn([[2, 0.3], [0.3, 1]], [1, 0])
        self.assertEqual(result["lambda_num"], 0.0)

    def test_negative_curvature_is_technical(self):
        with self.assertRaisesRegex(GeometryError, "GN_SIGNIFICANT_NEGATIVE"):
            whiten_gn(np.diag([-1e-5, 1]), [1, 0])

    def test_roundoff_negative_preserved_and_shifted(self):
        result = whiten_gn(np.diag([-1e-13, 1]), [1, 0])
        self.assertLess(result["eigenvalues"][0], 0)
        self.assertGreater(result["lambda_num"], 0)

    def test_zero_curvature_significant_gradient_is_error(self):
        with self.assertRaisesRegex(GeometryError, "GN_ZERO_WITH_SIGNIFICANT"):
            whiten_gn(np.zeros((2, 2)), [1, 0])
        self.assertEqual(whiten_gn(np.zeros((2, 2)), [0, 0])["status"], "NUMERICAL_ZERO_GN_AND_GRADIENT")

    def test_asymmetry_and_nonfinite_fail(self):
        with self.assertRaisesRegex(GeometryError, "GN_ASYMMETRIC"):
            whiten_gn([[1, 0.1], [0.2, 1]], [1, 2])
        with self.assertRaisesRegex(GeometryError, "GN_NONFINITE"):
            whiten_gn([[float("inf")]], [1])


class QPTests(unittest.TestCase):
    def test_scalar_box_exact(self):
        result = solve_box_qp([-2], [], [], 0.5)
        np.testing.assert_array_equal(result["u"], [0.5])
        self.assertAlmostEqual(result["predicted_gain"], 0.875)
        self.assertTrue(result["KKT_pass"])

    def test_interior_exact(self):
        result = solve_box_qp([-0.3, 0.4, -0.1], [], [], 1.0)
        np.testing.assert_allclose(result["u"], [0.3, -0.4, 0.1], atol=1e-14)

    def test_zero_slack_conflict_returns_certified_zero(self):
        result = solve_box_qp([-3, 2], [[1, 0], [-1, 0], [0, 1], [0, -1]], [0] * 4, 2.0)
        np.testing.assert_allclose(result["u"], [0, 0], atol=1e-14)
        self.assertAlmostEqual(result["predicted_gain"], 0)
        self.assertTrue(result["KKT_pass"])

    def test_redundant_identical_rows(self):
        A = np.tile([1.0, 0.0, 0.0], (8000, 1))
        result = solve_box_qp([-2, 1, -0.2], A, np.full(8000, 0.1), 1)
        np.testing.assert_allclose(result["u"], [0.1, -1, 0.2], atol=1e-12)
        self.assertLessEqual(len(result["working_set"]), 3)

    def test_zero_rows_are_not_constraints(self):
        result = solve_box_qp([-0.2, 0.3], np.zeros((300, 2)), np.zeros(300), 1)
        np.testing.assert_allclose(result["u"], [0.2, -0.3], atol=1e-14)

    def test_many_random_guards_independent_slsqp_crosscheck(self):
        rng = np.random.default_rng(20260917)
        for case in range(18):
            m = case % 3 + 1
            A = rng.normal(size=(1200, m))
            s = rng.uniform(0.0, 1.0, 1200)
            if case % 2:
                s[:5] = 0.0
            g = rng.normal(size=m) * 3
            result = solve_box_qp(g, A, s, 0.7)
            u = np.asarray(result["u"])
            reference = minimize(lambda x: 0.5 * (x @ x) + g @ x, np.zeros(m),
                                 jac=lambda x: x + g, method="SLSQP", bounds=[(-0.7, 0.7)] * m,
                                 constraints=[LinearConstraint(A, -np.inf, s)],
                                 options={"ftol": 1e-12, "maxiter": 300})
            self.assertTrue(reference.success, reference.message)
            np.testing.assert_allclose(u, reference.x, atol=1e-7)
            self.assertLessEqual(np.max(A @ u - s), 1e-8)

    def test_row_scaling_does_not_change_endpoint(self):
        A = np.array([[1, 0], [0, 1], [-1, 1]], dtype=float)
        s = np.array([0.1, 0.3, 0.2])
        base = solve_box_qp([-2, -1], A, s, 1.0)
        scale = np.array([1e4, 1e-4, 20])
        other = solve_box_qp([-2, -1], A * scale[:, None], s * scale, 1.0)
        np.testing.assert_allclose(base["u"], other["u"], atol=1e-12)
        self.assertEqual(len(other["KKT"]["multipliers_original_units"]), 7)

    def test_negative_slack_is_not_tolerance_clipped(self):
        with self.assertRaisesRegex(QPError, "ZERO_ANCHOR_INFEASIBLE"):
            solve_box_qp([-1], [[1]], [-1e-16], 1)

    def test_nonfinite_badshape_badarm(self):
        for args in (([float("nan")], [], [], 1), ([1], [[float("inf")]], [0], 1),
                     ([1], [], [], 0), ([1], [[1, 2]], [0], 1)):
            with self.assertRaises(QPError):
                solve_box_qp(*args)
        with self.assertRaisesRegex(QPError, "UNAUTHORIZED"):
            solve_repair_qp([1], [[1]], [], [], 1.0, "N4")

    def test_negative_dual_and_large_row_not_hidden(self):
        report = kkt_receipt([0], [[1e12]], [0], [1e-12], [-1e-10], [1e12])
        self.assertEqual(report["raw"]["primal_max_positive_violation"], 1.0)
        self.assertGreater(report["row_normalized"]["dual_max_negative_violation"], 1)
        self.assertLess(report["multipliers_original_units"][0], 0)

    def test_iteration_limit_is_technical_not_fallback(self):
        with self.assertRaisesRegex(QPError, "ITERATION_LIMIT"):
            solve_box_qp([-2], [], [], 1, QPPolicy(max_iterations=1))

    def test_radius_and_same_box_free_safe(self):
        result = solve_repair_qp([-1, -2], [[1, 0], [0, 4]], [[1, 0]], [0], 0.25, "R-QP")
        self.assertEqual(result["radius"], 0.5)
        self.assertLessEqual(result["Gamma_safe"], result["Gamma_free"])
        self.assertAlmostEqual(result["c"][0], 0)
        self.assertLessEqual(result["safe_over_free"], 1)
        json.dumps(result, allow_nan=False)

    def test_gd_no_proposal_guard_but_single_direction(self):
        result = solve_repair_qp([-1], [[1]], [[1]], [0], 0.5, "R-GD")
        self.assertGreater(result["u"][0], 0)
        self.assertFalse(result["proposal_guards_used"])
        with self.assertRaisesRegex(QPError, "DIRECTION_DIMENSION"):
            solve_repair_qp([-1, -1], np.eye(2), [], [], 1, "R-GD")

    def test_normal_off_reason(self):
        zero = solve_repair_qp([], [], [], [], 1, "R-QP")
        small = solve_repair_qp([1], [[1]], [], [], 1e-6, "R-QP")
        self.assertEqual(zero["reason"], "ZERO_DIMENSION")
        self.assertEqual(small["reason"], "BASE_SMALL")
        self.assertIsNone(zero["safe_over_free"])

    def test_small_base_never_hides_nonfinite_or_invalid_model(self):
        with self.assertRaisesRegex(QPError, "MODEL_NONFINITE"):
            solve_repair_qp([float("nan")], [[1]], [], [], 1e-8, "R-GD")
        with self.assertRaisesRegex(QPError, "MODEL_SHAPE"):
            solve_repair_qp([1], [[1, 2]], [], [], 1e-8, "R-GD")

    def test_many_zero_slack_cone_row_order_invariance(self):
        rng = np.random.default_rng(8117)
        A = rng.normal(size=(2000, 3))
        A = np.concatenate((A, -A))
        for order in (np.arange(len(A)), rng.permutation(len(A))):
            result = solve_box_qp([-1, 2, -0.5], A[order], np.zeros(len(A)), 1)
            np.testing.assert_allclose(result["u"], np.zeros(3), atol=1e-10)
            self.assertTrue(result["KKT_pass"])


if __name__ == "__main__":
    unittest.main()
