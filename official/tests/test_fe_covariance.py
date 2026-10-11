import types
import unittest

import torch

from official.baselines.fe_covariance import (
    FECovarianceError, checked_ridge_solve, guarded_author_solve,
    prepare_covariance,
)


class CovarianceTests(unittest.TestCase):
    def test_healthy_prior_is_byte_identical(self):
        c = torch.tensor([[2., .25], [.25, 1.]], dtype=torch.float32)
        for policy in ('strict', 'project_roundoff'):
            result, audit = prepare_covariance(c, policy=policy)
            self.assertIs(result, c)
            self.assertFalse(audit['repaired'])

    def test_small_indefinite_prior_requires_explicit_repair(self):
        c = torch.diag(torch.tensor([-1e-8, 1e-9, 1.]))
        before = c.clone()
        with self.assertRaisesRegex(FECovarianceError, 'NOT_POSITIVE_SEMIDEFINITE'):
            prepare_covariance(c)
        result, audit = prepare_covariance(c, policy='project_roundoff')
        self.assertTrue(torch.equal(c, before))
        self.assertEqual(result.dtype, torch.float64)
        self.assertEqual(audit['corrected_eigenvalues'], 2)
        self.assertEqual(audit['repaired_cholesky_info'], 0)
        self.assertAlmostEqual(result[2, 2].item(), 1., places=15)
        self.assertAlmostEqual(result[0, 0].item(), torch.finfo(torch.float32).eps, places=15)
        self.assertGreaterEqual(torch.linalg.eigvalsh(result).min().item(),
                                audit['eigenvalue_floor'] * (1-1e-12))

    def test_singular_psd_prior_is_not_misidentified_as_negative(self):
        c = torch.diag(torch.tensor([0., 1.]))
        result, audit = prepare_covariance(c)
        self.assertIs(result, c)
        self.assertTrue(audit['psd_singular'])

    def test_materially_wrong_cache_is_not_silently_repaired(self):
        with self.assertRaisesRegex(FECovarianceError, 'EXCEEDS_STORAGE_ROUNDOFF'):
            prepare_covariance(torch.diag(torch.tensor([-.01, 1.])),
                               policy='project_roundoff')

    def test_invalid_inputs(self):
        for c in (torch.ones(2, 3), torch.ones(2, 2, dtype=torch.int32),
                  torch.tensor([[float('nan')]]),
                  torch.tensor([[1., 1.], [0., 1.]])):
            with self.assertRaises(FECovarianceError):
                prepare_covariance(c, policy='project_roundoff')
        with self.assertRaises(FECovarianceError):
            prepare_covariance(torch.eye(2), policy='silent_lu')

    def test_zero_clipping_does_not_resolve_unidentifiable_inverse(self):
        c = torch.diag(torch.tensor([-1e-8, 1.]))
        k = torch.tensor([[1e-4], [0.]], dtype=torch.float64)
        zero_clipped = torch.diag(torch.tensor([0., 1.], dtype=torch.float64))
        huge = torch.linalg.solve(15000*zero_clipped+k@k.T, k)
        repaired, _ = prepare_covariance(c, policy='project_roundoff')
        bounded = checked_ridge_solve(15000*repaired+k@k.T, k)
        self.assertGreater(float(huge.norm()), 9000)
        self.assertLess(float(bounded.norm()), 1)

    def test_finite_lu_output_on_indefinite_system_is_rejected(self):
        a = torch.diag(torch.tensor([-1e-12, 1.], dtype=torch.float64))
        k = torch.ones(2, 1, dtype=torch.float64)
        self.assertTrue(torch.isfinite(torch.linalg.solve(a, k)).all())
        with self.assertRaisesRegex(FECovarianceError, 'SYSTEM_NOT_POSITIVE_DEFINITE'):
            checked_ridge_solve(a, k)

    def test_healthy_solve_and_global_torch_are_unchanged(self):
        module = types.SimpleNamespace(torch=torch)
        original = torch.linalg.solve
        a = torch.tensor([[2., .3], [.3, 1.]], dtype=torch.float64)
        b = torch.tensor([[.2], [.8]], dtype=torch.float64)
        expected = original(a, b)
        with guarded_author_solve(module) as audit:
            result = module.torch.linalg.solve(a, b)
            self.assertIs(torch.linalg.solve, original)
            self.assertTrue(torch.equal(result, expected))
        self.assertIs(module.torch, torch)
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]['cholesky_info'], 0)
        with self.assertRaises(FECovarianceError):
            with guarded_author_solve(module):
                module.torch.linalg.solve(-torch.eye(2, dtype=torch.float64), b)
        self.assertIs(module.torch, torch)


if __name__ == '__main__':
    unittest.main()
