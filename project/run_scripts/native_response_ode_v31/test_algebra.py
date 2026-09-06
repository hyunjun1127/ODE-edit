import unittest
import torch
from .algebra import nnls_response, identities, ray_solution, turning, factor_inner, FrozenNormalization


class AlgebraTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260906)
        self.e = torch.randn(13, dtype=torch.float64)
        self.psi = torch.randn(13, 5, dtype=torch.float64)
        self.metric = torch.eye(5, dtype=torch.float64)

    def test_kkt_dissipation_speed(self):
        for lam in (.01, .1, 1.):
            sol = nnls_response(self.e, self.psi, self.metric, lam)
            self.assertTrue(bool((sol.coefficients >= 0).all()))
            self.assertGreaterEqual(float(sol.gradient.min()), -1e-10)
            self.assertLess(sol.complementarity, 1e-10)
            fact = identities(self.e, self.psi, self.metric, sol.coefficients, lam)
            self.assertLess(abs(fact['dissipation_residual']), 1e-10)
            self.assertLessEqual(fact['speed_excess'], 1e-10)

    def test_positive_coordinate_scaling(self):
        s = torch.tensor([.2, .5, 1., 2., 5.], dtype=torch.float64)
        original = nnls_response(self.e, self.psi, self.metric).coefficients
        scaled = nnls_response(self.e, self.psi*s, self.metric*s[:, None]*s).coefficients
        torch.testing.assert_close(scaled*s, original, atol=1e-10, rtol=1e-10)

    def test_layer_permutation(self):
        p = torch.tensor([4, 2, 0, 3, 1])
        a = nnls_response(self.e, self.psi, self.metric).coefficients
        b = nnls_response(self.e, self.psi[:, p], self.metric[p][:, p]).coefficients
        torch.testing.assert_close(a[p], b, atol=1e-10, rtol=1e-10)

    def test_fixed_qref_quota_invariance(self):
        # Positive per-layer raw-field quota changes must cancel in whitening;
        # qref remains fixed, never normalized again at a comparator/node.
        d = torch.randn(20, 5, dtype=torch.float64)
        q = d.square().sum(0) / 7.
        scale = torch.tensor([.2, .4, 1., 3., 5.], dtype=torch.float64)
        torch.testing.assert_close(d/q.sqrt(), (d*scale)/(q*scale.square()).sqrt(), atol=1e-10, rtol=1e-10)

    def test_negative_individual_gain_jointly_useful(self):
        e = torch.tensor([1., 0.], dtype=torch.float64)
        psi = torch.tensor([[1., -.1], [1., -1.]], dtype=torch.float64)
        sol = nnls_response(e, psi, torch.eye(2, dtype=torch.float64))
        self.assertLess(float((psi.T@e)[1]), 0)
        self.assertGreater(float(sol.coefficients[1]), 0)

    def test_affine_euler_barrier_increment(self):
        c = nnls_response(self.e, self.psi, self.metric).coefficients
        q = c@self.metric@c
        p = self.psi@c
        for h in (.25, .5, 1.):
            b = .5*self.e.square().sum() - .5*(self.e-h*p).square().sum() - .1*h*q
            torch.testing.assert_close(b, h*(1-h/2)*p.square().sum(), atol=1e-10, rtol=1e-10)

    def test_ray_and_turning(self):
        c = nnls_response(self.e, self.psi, self.metric).coefficients
        ray, _ = ray_solution(self.e, self.psi, self.metric, torch.ones(5))
        self.assertTrue(bool((ray >= 0).all()))
        self.assertGreaterEqual(turning(c, ray, self.metric)['Rturn'], -1e-14)

    def test_factor_gram_dense_parity(self):
        l, r = torch.randn(4, 2, dtype=torch.float64), torch.randn(7, 2, dtype=torch.float64)
        m = torch.randn(7, 7, dtype=torch.float64); m = m@m.T
        value = factor_inner(l, r, l, r, lambda x: m@x)
        dense = l@r.T
        self.assertAlmostEqual(value, float(torch.trace(dense@m@dense.T)), places=10)

    def test_n0_no_small_anchor_clipping(self):
        z = torch.tensor([[1e-20, 0., 1.], [0., 0., 2.]])
        norm = FrozenNormalization.capture(z, torch.zeros_like(z), 'cpu')
        self.assertEqual(norm.active.tolist(), [True, False, True])
        self.assertGreater(float(norm.scales[0]), 0)

    def test_zero_response_and_invalid_metric(self):
        sol = nnls_response(self.e, torch.zeros_like(self.psi), self.metric)
        self.assertEqual(float(sol.coefficients.abs().sum()), 0.)
        with self.assertRaises(Exception):
            nnls_response(self.e, self.psi, -self.metric)


if __name__ == '__main__':
    unittest.main()
