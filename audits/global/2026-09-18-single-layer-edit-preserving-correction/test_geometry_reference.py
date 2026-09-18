"""CPU-only semantic fixtures. PASS is not neural or production validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

import geometry_reference as ref


class GeometryFixtures(unittest.TestCase):
    def test_exact_intersection_preserves_both_constraints(self):
        p = np.diag([1., 1., 1., 0.])
        k = np.array([[1., 0.], [0., 1.], [1., 1.], [2., -1.]])
        result = ref.intersection(p, k)
        q = result["Q"]
        self.assertEqual(result["null_dimension"], 1)
        np.testing.assert_allclose(q @ k, 0., atol=2e-14)
        np.testing.assert_allclose(q @ p, q, atol=2e-14)
        np.testing.assert_allclose(q @ q, q, atol=2e-14)
        naive = p @ (np.eye(4) - k @ np.linalg.pinv(k))
        self.assertGreater(np.linalg.norm(naive - q), 0.1)
        self.assertGreater(np.linalg.norm(naive @ p - naive), 0.1)

    def test_same_finite_response_survives_nonlinear_suffix(self):
        k = np.array([[1., 0., 1.], [0., 1., 1.], [0., 0., 0.]])
        q = ref.intersection(np.eye(3), k)["Q"]
        w = np.array([[.2, -.5, 1.], [.1, .7, -.3]])
        d = np.array([[3., 2., 4.], [-2., 3., 5.]]) @ q
        np.testing.assert_array_equal((w + d) @ k, w @ k)
        np.testing.assert_array_equal(ref.nonlinear_suffix((w + d) @ k), ref.nonlinear_suffix(w @ k))
        self.assertGreater(np.linalg.norm(d @ np.array([[0.], [0.], [1.]])), 1.)

    def test_subject_only_lock_does_not_lock_contextual_output(self):
        k = np.eye(2)
        q = ref.intersection(np.eye(2), k[:, :1])["Q"]
        d = np.array([[0., 1.]]) @ q
        np.testing.assert_array_equal(d @ k[:, :1], np.zeros((1, 1)))
        self.assertGreater(abs(ref.nonlinear_suffix(d @ k)[0, -1]), .5)

    def test_native_family_uniqueness_and_basis_invariance(self):
        a = np.array([[1., 0., 1.], [0., 1., 1.]])
        k = np.eye(3)[:, :2]
        first = ref.native_family_dimension(a, k, output_dimension=4)
        second = ref.native_family_dimension(np.array([[2., 1.], [1., 1.]]) @ a, k, output_dimension=4)
        self.assertEqual(first["physical_dimension"], 0)
        self.assertEqual(first, second)
        self.assertEqual(ref.intersection(np.eye(3), k)["null_dimension"], 1)
        nonzero = ref.native_family_dimension(a, k[:, :1], output_dimension=4)
        self.assertEqual(nonzero["physical_dimension"], 4)

    def test_coefficient_gauge_is_not_physical_freedom(self):
        a = np.array([[1., 0., 0.], [2., 0., 0.]])
        result = ref.native_family_dimension(a, np.eye(3)[:, :1], output_dimension=2)
        self.assertEqual(result["coefficient_gauge_per_output"], 1)
        self.assertEqual(result["physical_dimension"], 0)
        np.testing.assert_array_equal(np.array([[2., -1.]]) @ a, np.zeros((1, 3)))

    def test_projected_descent_and_gradient_fraction(self):
        p = np.diag([1., 1., 1., 0.])
        q = ref.intersection(p, np.eye(4)[:, :2])["Q"]
        w = np.array([[1., 2., 3., 4.]])
        result = ref.descent_diagnostic(w, p, q)
        self.assertAlmostEqual(result["chi"], 9.)
        self.assertAlmostEqual(result["remaining_gradient_fraction"], 9. / 14.)
        direction = result["direction"]
        self.assertAlmostEqual(float(np.sum(w * direction)), -result["chi"])
        self.assertLess(float(np.sum((w + .1 * direction) ** 2)), float(np.sum(w ** 2)))
        self.assertIsNone(ref.descent_diagnostic(np.zeros_like(w), p, q)["remaining_gradient_fraction"])

    def test_near_rank_policy_does_not_unlock_ambiguous_modes(self):
        eps = np.finfo(np.float64).eps
        ambiguous = ref.intersection(np.eye(2), np.diag([1., 2. * eps]))
        self.assertEqual(ambiguous["status"], "RANK_UNRESOLVED")
        self.assertIsNone(ambiguous["Q"])
        small_but_resolved = ref.intersection(np.eye(2), np.diag([1., 1e-8]))
        self.assertEqual(small_but_resolved["status"], "REPAIR_SPACE_EMPTY")
        zero = ref.intersection(np.eye(2), np.diag([1., 0.]))
        self.assertEqual(zero["null_dimension"], 1)

    def test_native_gram_budget_blocks_nonzero_edit_null_correction(self):
        k = np.array([[1., 0.], [0., 1.], [1., 1.]])
        c = np.array([[2., .2, 0.], [.2, 3., .1], [0., .1, 1.]])
        native = ref.native_ridge(k, np.array([[2., -1.], [1., 3.]]), c)
        q = ref.intersection(np.eye(3), k)["Q"]
        d = np.array([[1., 2., -1.], [2., -1., 3.]]) @ q
        np.testing.assert_allclose(d @ k, 0., atol=2e-14)
        self.assertAlmostEqual(float(np.sum((native @ c) * d)), 0., places=12)
        increase = ref.covariance_energy(native + d, c) - ref.covariance_energy(native, c)
        self.assertAlmostEqual(increase, ref.covariance_energy(d, c), places=12)
        self.assertGreater(increase, 0.)

    def test_all_history_rank_saturation_is_valid_zero_space(self):
        p = np.diag([1., 1., 1., 0.])
        free = [ref.intersection(p, np.eye(4)[:, :count])["null_dimension"] for count in (1, 2, 3)]
        self.assertEqual(free, [2, 1, 0])
        result = ref.intersection(p, np.eye(4)[:, :3])
        self.assertEqual(result["status"], "REPAIR_SPACE_EMPTY")
        np.testing.assert_array_equal(result["Q"], np.zeros((4, 4)))

    def test_aggregate_gram_cannot_identify_a_retired_fact(self):
        first = np.eye(2)
        second = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2.)
        np.testing.assert_allclose(first @ first.T, second @ second.T, atol=1e-15)
        gram_after_first_retirement = first[:, 1:] @ first[:, 1:].T
        gram_after_second_retirement = second[:, 1:] @ second[:, 1:].T
        self.assertGreater(np.linalg.norm(gram_after_first_retirement - gram_after_second_retirement), .5)

    def test_post_native_history_lock_freezes_native_damage(self):
        k_history = np.array([[1.], [0.]])
        accepted = np.array([[1., 0.]])
        native = np.array([[2., 0.]])
        q = ref.intersection(np.eye(2), k_history)["Q"]
        repair = np.array([[4., 3.]]) @ q
        np.testing.assert_array_equal((native + repair) @ k_history, native @ k_history)
        self.assertFalse(np.array_equal((native + repair) @ k_history, accepted @ k_history))


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GeometryFixtures)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    root = Path(__file__).resolve().parent
    source = {}
    for name in ("geometry_reference.py", "test_geometry_reference.py"):
        raw = (root / name).read_bytes()
        source[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    receipt = {
        "scope": "NumPy FP64 small CPU mathematical fixtures ONLY",
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "numpy_version": np.__version__, "dtype": "float64",
        "model_forward_calls": 0, "GPU_calls": 0,
        "production_runner": False,
        "real_model_gradient_validation": "NOT_RUN",
        "real_FP32_materialization_validation": "NOT_RUN",
        "model_rank_or_capacity_claim": False,
        "near_rank_policy": {
            "threshold": "max(shape)*eps64*s_max of captured entries",
            "ambiguity_band": "[threshold/10, 10*threshold]",
            "action": "RANK_UNRESOLVED; no repair projector emitted",
            "provenance": "numerical design choice; not calibrated on a model",
        },
        "sources": source,
        "limitations": [
            "Full SVD is intentionally for small fixtures, not production scalability.",
            "Rank of captured FP32 entries is not intrinsic neural rank or capacity.",
            "The nonlinear suffix is a deterministic toy, not a language-model test.",
            "No dataset, teacher, native target, checkpoint, or GPU was loaded.",
        ],
    }
    (root / "geometry_checks.json").write_text(json.dumps(receipt, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
