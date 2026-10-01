"""CPU synthetic algebra/regression only; no actual GPU/model claims."""
import unittest
import torch

from .burden import (build_burden_matrix, anchor_scale_squared,
                     omega_and_grad, actual_energy_chunked)


class BurdenTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        g = torch.Generator().manual_seed(20261002)
        z = torch.randn(9, 9, dtype=torch.float64, generator=g)
        self.A = z @ z.T + torch.eye(9, dtype=torch.float64)
        self.K = torch.randn(9, 4, dtype=torch.float64, generator=g)
        self.P = torch.linalg.solve(self.A + self.K @ self.K.T, self.K)
        self.R = torch.randn(7, 4, dtype=torch.float64, generator=g)

    def test_small_matches_direct_no_eigen_clip(self):
        M, receipt = build_burden_matrix(self.A, self.K, self.P)
        direct = self.P.T @ self.A @ self.P
        torch.testing.assert_close(M, direct, atol=1e-13, rtol=1e-11)
        self.assertEqual(receipt["original_verdict"], "PASS")
        self.assertEqual(receipt["route"], "S-S2")
        self.assertFalse(receipt["eigenvalue_clipping"])

    def test_finite_mismatch_warns_and_uses_direct(self):
        P = self.P * 1.1
        M, receipt = build_burden_matrix(self.A, self.K, P)
        expected = P.T @ (self.A @ P)
        expected = (expected + expected.T) * .5
        self.assertEqual(receipt["original_verdict"], "FAIL_DIRECT_FALLBACK")
        self.assertEqual(receipt["action"], "RECORD_WARNING_USE_DIRECT")
        self.assertTrue(torch.equal(M, expected))

    def test_negative_values_not_clipped(self):
        A = -torch.eye(9, dtype=torch.float64)
        M, receipt = build_burden_matrix(A, self.K, self.P)
        self.assertLess(float(torch.linalg.eigvalsh(M).min()), 0)
        self.assertFalse(receipt["eigenvalue_clipping"])

    def test_prior_route_requires_explicit_binding(self):
        with self.assertRaisesRegex(ValueError, "NOT_VERIFIED"):
            build_burden_matrix(None, self.K, self.P, verify_direct=False)
        M, receipt = build_burden_matrix(None, self.K, self.P, verify_direct=False,
                                        previously_verified_route={"source": "sha", "dtype": "float64"})
        self.assertFalse(receipt["direct_verified"])
        self.assertEqual(tuple(M.shape), (4, 4))

    def test_omega_matches_dense_and_analytic_gradient(self):
        M, _ = build_burden_matrix(self.A, self.K, self.P)
        omega, grad = omega_and_grad(self.R, M, 2.7)
        delta = self.R @ self.P.T
        direct = .5 * ((delta @ self.A) * delta).sum() / 2.7
        torch.testing.assert_close(omega, direct, atol=1e-12, rtol=1e-12)
        variable = self.R.clone().requires_grad_()
        automatic = .5 * ((variable @ M) * variable).sum() / 2.7
        automatic.backward()
        torch.testing.assert_close(grad, variable.grad, atol=1e-13, rtol=1e-12)
        direction = torch.ones_like(self.R)
        eps = 1e-6
        plus = omega_and_grad(self.R + eps * direction, M, 2.7)[0]
        minus = omega_and_grad(self.R - eps * direction, M, 2.7)[0]
        self.assertAlmostEqual(float((plus - minus) / (2 * eps)), float((grad * direction).sum()), places=8)

    def test_eta_zero_exact_gradient_and_diagnostic_omega_retained(self):
        M, _ = build_burden_matrix(self.A, self.K, self.P)
        value0, grad0 = omega_and_grad(self.R.float(), M, 2.7, eta=0)
        value1, grad1 = omega_and_grad(self.R.float(), M, 2.7, eta=1)
        self.assertEqual(float(value0), float(value1))
        self.assertEqual(int(grad0.count_nonzero()), 0)
        self.assertEqual(grad1.dtype, torch.float32)
        self.assertTrue(torch.equal(grad0 + grad1, grad1))

    def test_anchor_scale_mean_columns(self):
        anchors = torch.tensor([[3., 0.], [4., 2.]])
        self.assertEqual(anchor_scale_squared(anchors), 14.5)
        with self.assertRaisesRegex(ValueError, "ANCHOR_SCALE"):
            anchor_scale_squared(torch.zeros_like(anchors))

    def test_actual_energy_chunk_tail_and_fp32_addition(self):
        entry = torch.full((7, 9), 100., dtype=torch.float32)
        committed = entry + (self.R @ self.P.T).float() * 1e-4
        actual = committed.double() - entry.double()
        direct = float(((actual @ self.A) * actual).sum())
        receipt = actual_energy_chunked(committed, entry, self.A, 2.7, row_chunk=3)
        self.assertAlmostEqual(receipt["actual_raw_energy"], direct, places=18)
        self.assertEqual(receipt["chunks"], 3)
        self.assertEqual(receipt["checkpoint_saved"], False)
        self.assertEqual(receipt["actual_omega"], .5 * receipt["actual_raw_energy"] / 2.7)

    def test_zero_actual_delta(self):
        entry = torch.ones(7, 9)
        receipt = actual_energy_chunked(entry, entry.clone(), self.A, 1.)
        self.assertEqual(receipt["actual_raw_energy"], 0.)
        self.assertEqual(receipt["actual_delta_frobenius"], 0.)

    def test_nonfinite_and_wrong_shapes_remain_structural(self):
        P = self.P.clone()
        P[0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "NONFINITE_P"):
            build_burden_matrix(self.A, self.K, P)
        M, _ = build_burden_matrix(self.A, self.K, self.P)
        bad = M.clone()
        bad[0, 1] += .01
        with self.assertRaisesRegex(ValueError, "NOT_SYMMETRIC"):
            omega_and_grad(self.R, bad, 1.)
        with self.assertRaisesRegex(ValueError, "ETA_OR_SCALE"):
            omega_and_grad(self.R, M, 1., eta=.5)
        with self.assertRaisesRegex(ValueError, "NOT_FP32"):
            actual_energy_chunked(torch.zeros(7, 9).double(), torch.zeros(7, 9), self.A, 1.)


if __name__ == "__main__":
    unittest.main()
