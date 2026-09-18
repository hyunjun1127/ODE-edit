"""CPU mathematics/regression checks; none is an actual Llama T PASS."""
import unittest
from unittest.mock import patch

import numpy as np
import scipy.linalg as la

from project.run_scripts.single_layer_edit_preserving_correction.geometry import (
    GeometryError, RankUnresolved, ColumnBlocks, allowed_range, edit_null_space,
    row_space, ca_exact, rank_diagnostic, columns, matrix_sha256,
    gradient_diagnostics, invariant_diagnostics, projector_diagnostics,
    workspace_plan,
)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20260918)

    def test_streamed_singular_values_and_original_dimension_tau(self):
        x = self.rng.normal(size=(9, 111))
        got = rank_diagnostic(columns(x, block_columns=7))
        expected = la.svdvals(x)
        np.testing.assert_allclose(got["singular_values"], expected, rtol=1e-13)
        self.assertEqual(got["rank"], 9)
        self.assertEqual(len(got["singular_values"]), 9)
        self.assertAlmostEqual(got["threshold"], 111 * np.finfo(float).eps * expected[0], 25)
        self.assertGreater(got["blocks"], 1)

    def test_tall_thin_and_wide_spectra(self):
        for shape in [(31, 5), (5, 31), (1, 1)]:
            x = self.rng.normal(size=shape)
            result = rank_diagnostic(columns(x, 3))
            np.testing.assert_allclose(result["singular_values"], la.svdvals(x), rtol=1e-13)

    def test_rank_ambiguity_is_not_unlocked(self):
        tau = 3 * np.finfo(float).eps
        x = np.diag([1., tau, 0.])
        diag = rank_diagnostic(x)
        self.assertEqual(diag["status"], "RANK_UNRESOLVED")
        space = edit_null_space(allowed_range(np.eye(3)), x)
        self.assertIsNone(space.dimension)
        with self.assertRaises(RankUnresolved):
            space.project(np.ones((2, 3)))

    def test_zero_matrix_and_zero_allowed_range(self):
        self.assertEqual(rank_diagnostic(np.zeros((5, 21)))["rank"], 0)
        zero = allowed_range(np.zeros((5, 5)))
        space = edit_null_space(zero, np.zeros((5, 21)))
        self.assertEqual(space.status, "REPAIR_SPACE_EMPTY")
        np.testing.assert_array_equal(space.project(np.ones((3, 5))), np.zeros((3, 5)))

    def test_column_cardinality_validation(self):
        broken = ColumnBlocks(2, 5, lambda: iter([np.ones((2, 4))]))
        with self.assertRaises(GeometryError):
            rank_diagnostic(broken)
        extra = ColumnBlocks(2, 1, lambda: iter([np.ones((2, 2))]))
        with self.assertRaises(GeometryError):
            rank_diagnostic(extra)

    def test_native_raw_is_unchanged_and_not_replaced_by_star(self):
        p = np.diag([1.01, 0.99, -.01, .01]).astype(np.float32)
        before = p.copy()
        space = allowed_range(p)
        np.testing.assert_array_equal(p, before)
        self.assertGreater(space.diagnostic["raw_vs_star_frobenius"], 0.)
        np.testing.assert_allclose(space.project(np.eye(4)), np.diag([1., 1., 0., 0.]))

    def test_nonprojector_negative_eigen_recovery(self):
        for values in [[1., .5], [1., .1], [1., .9], [1., -1.], [1., 1.2]]:
            with self.assertRaises(GeometryError):
                allowed_range(np.diag(values))

    def test_bound_provenance_path(self):
        p = np.diag([1., 0., 1.]).astype(np.float32)
        v = np.eye(3)[:, [0, 2]]
        provenance = {"raw_sha256": matrix_sha256(p), "basis_sha256": matrix_sha256(v),
                      "allowed_rank": 2, "source": "fixture"}
        with patch("scipy.linalg.eigh", side_effect=AssertionError("no eigen recomputation")):
            result = allowed_range(p, provenance_basis=v, provenance=provenance)
        self.assertEqual(result.dimension, 2)
        with self.assertRaises(GeometryError):
            allowed_range(p, provenance_basis=v, provenance={**provenance, "raw_sha256": "wrong"})

    def test_intersection_not_sequential_projector_product(self):
        v = np.eye(4)[:, :3]
        p = v @ v.T
        keys = np.array([[1., 0.], [1., 0.], [0., 1.], [1., 0.]])
        space = edit_null_space(allowed_range(p), keys)
        q = space.project(np.eye(4))
        np.testing.assert_allclose(q @ keys, 0., atol=1e-14)
        np.testing.assert_allclose(q @ q, q, atol=1e-14)
        np.testing.assert_allclose(q.T, q, atol=1e-14)
        naive = p @ (np.eye(4) - keys @ np.linalg.pinv(keys))
        self.assertGreater(la.norm(q - naive), .01)
        self.assertEqual(space.dimension, 1)
        self.assertEqual(projector_diagnostics(space)["status"], "PASS")

    def test_no_full_matrices_svd(self):
        original = la.svd
        calls = []
        def checked(*args, **kwargs):
            self.assertFalse(kwargs.get("full_matrices", True))
            calls.append(args[0].shape)
            return original(*args, **kwargs)
        with patch("scipy.linalg.svd", side_effect=checked):
            space = edit_null_space(allowed_range(np.eye(17)), self.rng.normal(size=(17, 3)))
        self.assertEqual(space.dimension, 14)
        self.assertTrue(calls)

    def test_empty_space_exact_zero_not_rounding_direction(self):
        space = edit_null_space(allowed_range(np.eye(4)), self.rng.normal(size=(4, 20)))
        self.assertEqual(space.dimension, 0)
        np.testing.assert_array_equal(space.project(self.rng.normal(size=(5, 4))), np.zeros((5, 4)))

    def test_ca_coordinate_gauge_does_not_create_physical_directions(self):
        a = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [1., 1., 0., 0.]])
        keys = np.eye(4)[:, :1]
        space, diagnostic = ca_exact(a, keys, output_dimension=7)
        self.assertEqual(diagnostic["coefficient_gauge_per_output"], 1)
        self.assertEqual(diagnostic["physical_dimension_per_output"], 1)
        self.assertEqual(diagnostic["physical_dimension"], 7)
        transformed, second = ca_exact(np.diag([2., 4., 8.]) @ a, keys, output_dimension=7)
        self.assertEqual(second["physical_dimension"], 7)
        g = self.rng.normal(size=(7, 4))
        np.testing.assert_allclose(space.project(g), transformed.project(g), atol=1e-13)

    def test_ca_is_weight_euclidean_not_coefficient_gradient(self):
        a = np.array([[100., 0., 0.], [0., .01, 0.]])
        space = row_space(a)
        g = np.array([[2., 3., 4.]])
        np.testing.assert_allclose(space.project(g), [[2., 3., 0.]])
        self.assertGreater(la.norm(space.project(g) - g @ a.T @ a), 1000.)

    def test_gradient_chi_and_output_span_are_diagnostics_only(self):
        allowed = allowed_range(np.eye(4))
        space = edit_null_space(allowed, np.eye(4)[:, :2])
        g = self.rng.normal(size=(3, 4))
        qg, diag = gradient_diagnostics(g, allowed, space, native_output_basis=np.eye(3)[:, :1])
        self.assertAlmostEqual(diag["chi"], float(np.sum(qg * qg)))
        self.assertLess(diag["orthogonal_chi_absolute_gap"], 1e-12)
        self.assertGreater(la.norm(qg[1:]), 0.)
        self.assertFalse(diag["native_output_span_restricted"])

    def test_ideal_and_actual_response_are_separate(self):
        allowed = allowed_range(np.eye(3))
        keys = np.array([[1.], [1.], [0.]])
        native = np.array([[1e6, 1., 0.]], dtype=np.float32)
        ideal = np.array([[.01, -.01, 0.]])
        candidate = (native.astype(np.float64) + ideal).astype(np.float32)
        actual = candidate.astype(np.float64) - native.astype(np.float64)
        diag = invariant_diagnostics(ideal, actual, native, keys, allowed)
        self.assertEqual(diag["ideal_response_relative"], 0.)
        self.assertGreater(diag["actual_max_token_normalized_response"], 0.)
        self.assertGreater(diag["rounding_delta_norm"], 0.)

    def test_torch_cpu_projection_matches_numpy(self):
        import torch
        allowed = allowed_range(np.eye(7))
        space = edit_null_space(allowed, self.rng.normal(size=(7, 3)))
        g = self.rng.normal(size=(4, 7)).astype(np.float32)
        result = space.project(torch.from_numpy(g))
        self.assertEqual(result.dtype, torch.float64)
        np.testing.assert_allclose(result.numpy(), space.project(g), atol=1e-14)

    def test_nonfinite_is_technical_error(self):
        for value in [np.array([[np.nan]]), np.array([[np.inf]])]:
            with self.assertRaises(GeometryError):
                rank_diagnostic(value)

    def test_production_dimension_memory_plan_is_not_toy_full_token_square(self):
        plan = workspace_plan((14336, 100000))
        self.assertEqual(plan["compressed_R_bytes"], 14336 ** 2 * 8)
        self.assertLess(plan["compressed_R_bytes"], 100000 ** 2 * 8)
        self.assertEqual(plan["peak_status"], "ESTIMATE_COMPONENTS_NOT_MEASURED_PEAK")


if __name__ == "__main__":
    unittest.main()
