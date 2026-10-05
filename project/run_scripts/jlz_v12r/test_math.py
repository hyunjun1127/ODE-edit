import unittest
import numpy as np
import torch
from scipy.optimize import brentq
from .projection import project_capped_energy
from .controller import RequestController
from .optimizer import EfficiencyAdamAbs, analytic_norm


class ProjectionTests(unittest.TestCase):
    def test_independent_constrained_reference(self):
        rng = np.random.default_rng(20261006)
        for _ in range(30):
            sizes = [2, 3, 4]
            x = rng.normal(size=sum(sizes))
            caps = rng.uniform(.1, 2, 3)
            rho = float(rng.uniform(.05, 2))
            slices = [slice(0, 2), slice(2, 5), slice(5, 9)]
            blocks = {l: torch.tensor(x[s], dtype=torch.float64)[:, None] for l, s in enumerate(slices)}
            result, receipt = project_capped_energy(blocks, torch.tensor(caps)[:, None], torch.tensor([rho], dtype=torch.float64))
            y = np.concatenate([result[l][:, 0].double().numpy() for l in blocks])
            # Independent monotone dual solve (Brent, not production's sorted
            # active sets), with primal/dual stationarity checks. SLSQP's
            # status8 on a near-optimal constrained result is not a production
            # fault; the first failing reference and source remain preserved.
            norms = np.array([np.linalg.norm(x[s]) for s in slices])
            energy = lambda t: np.minimum(caps, norms / (1+t)) @ np.minimum(caps, norms / (1+t))
            dual = 0.
            if energy(0) > rho*rho:
                hi = 1.
                while energy(hi) > rho*rho:
                    hi *= 2
                dual = brentq(lambda t: energy(t)-rho*rho, 0., hi, xtol=1e-13)
            reference = np.concatenate([x[s] * min(c/n, 1/(1+dual)) for s, c, n in zip(slices, caps, norms)])
            self.assertLess(np.linalg.norm(reference)-rho, 1e-10)
            self.assertLess(abs(.5*((y-x)**2).sum() - .5*((reference-x)**2).sum()), 2e-6)
            self.assertLess(np.max(np.abs(y-reference)), 2e-7)
            self.assertLess(receipt['radius_excess_max'], 1e-6)

    def test_zero_and_single_layer(self):
        R = {4: torch.zeros(7, 3)}
        out, r = project_capped_energy(R, torch.tensor([[1., 2., 0.]]), torch.tensor([0., 3., 1.]))
        self.assertTrue(torch.equal(out[4], R[4]))
        x = torch.tensor([[3.], [4.]])
        out, _ = project_capped_energy({8: x}, torch.tensor([[3.]]), torch.tensor([2.]))
        self.assertTrue(torch.allclose(out[8], .4*x))

    def test_caps_first_not_double_radial(self):
        x = {4: torch.tensor([[10.]]), 8: torch.tensor([[1.]])}
        out, _ = project_capped_energy(x, torch.tensor([[.5], [2.]]), torch.tensor([1.]))
        self.assertAlmostEqual(float(out[4][0, 0]), .5, places=6)
        self.assertAlmostEqual(float(out[8][0, 0]), np.sqrt(.75), places=6)

    def test_nonfinite_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'PROJECTION_INPUT'):
            project_capped_energy({4: torch.tensor([[float('nan')]])}, torch.ones(1, 1), torch.ones(1))


class ControllerTests(unittest.TestCase):
    def controller(self, B=2, **kwargs):
        return RequestController(torch.full((2, B), 3.), torch.full((B,), 6.), [4, 8], **kwargs)

    def test_grace_reactivation_and_satisfied_updates(self):
        c = self.controller()
        active, terminal = c.observe(torch.tensor([.01, .1]), 0)
        self.assertEqual(active.tolist(), [False, True])
        self.assertFalse(terminal)
        for k in range(13):
            if k:
                c.observe(torch.tensor([.1 if k >= 2 else .01, .01 if k == 3 else .1]), k)
            c.before_update(torch.tensor([.1, .1]))
            self.assertEqual(c.expansion[1].item(), 0 if k < 12 else 1)
            c.record_update(c.active)
        self.assertEqual(c.t.tolist(), [11, 13])
        self.assertEqual(c.expansion.tolist(), [0, 1])

    def test_terminal_noexpansion(self):
        c = self.controller(B=1)
        for k in range(25):
            _, terminal = c.observe(torch.tensor([.1]), k)
            if not terminal:
                c.before_update(torch.tensor([.1])); c.record_update(c.active)
        self.assertEqual(c.t.item(), 24)
        self.assertEqual(c.expansion.item(), 4)
        self.assertTrue(torch.equal(c.radii, c.maximum))
        self.assertEqual(c.terminal_states(torch.tensor([.1])), ['UNSATISFIED_MAX'])
        self.assertEqual(c.receipt()['local_caps'], c.caps.tolist())
        with self.assertRaisesRegex(RuntimeError, 'TERMINAL'):
            c.before_update(torch.tensor([.1]))

    def test_noexpand_and_clipped_base(self):
        c = RequestController(torch.tensor([[8., 3.]]), torch.tensor([6., 4.]), [4], n_exp=0, base_multiplier='sqrt2')
        self.assertEqual(c.clipped.tolist(), [True, True])
        self.assertTrue(torch.equal(c.base, c.maximum))
        c.observe(torch.tensor([.1, .1]), 0)
        self.assertTrue(torch.isfinite(c.before_update(torch.tensor([.1, .1]))).all())
        self.assertEqual(c.expansion.tolist(), [0, 0])

    def test_zero_step_terminal(self):
        c = self.controller()
        active, terminal = c.observe(torch.tensor([.01, .049]), 0)
        self.assertFalse(bool(active.any())); self.assertTrue(terminal)
        self.assertEqual(c.terminal_states(torch.tensor([.01, .049])), ['ZERO_STEP', 'ZERO_STEP'])


class OptimizerTests(unittest.TestCase):
    def test_one_layer_native_adam_without_anchor_scaling(self):
        R = {8: torch.zeros(3, 2)}
        opt = EfficiencyAdamAbs(R)
        ref = torch.nn.Parameter(R[8].clone())
        native = torch.optim.Adam([ref], lr=.1, eps=1e-8, betas=(.9, .999), foreach=False)
        for _ in range(3):
            g = torch.tensor([[.2, -.7], [.5, .01], [-1., 2.]])
            ref.grad = g; native.step(); native.zero_grad()
            R, receipt = opt.step(R, {8: g}, torch.ones(2, dtype=torch.bool), torch.full((1, 2), 99.), torch.full((2,), 99.))
            self.assertTrue(torch.allclose(R[8], ref.detach(), atol=3e-7, rtol=2e-6))
            self.assertEqual(receipt['gamma'], [[1., 1.]])

    def test_local_counters_zero_gamma_and_moments(self):
        R = {4: torch.zeros(4, 2), 8: torch.zeros(2, 2)}
        opt = EfficiencyAdamAbs(R)
        zero = {l: torch.zeros_like(v) for l, v in R.items()}
        R, receipt = opt.step(R, zero, torch.tensor([True, False]), torch.ones(2, 2), torch.ones(2))
        self.assertEqual(opt.t.tolist(), [1, 0]); self.assertEqual(receipt['gamma'], [[1., 1.], [1., 1.]])
        g = {l: torch.ones_like(v) for l, v in R.items()}
        R, _ = opt.step(R, g, torch.tensor([True, True]), torch.full((2, 2), .01), torch.full((2,), .01))
        self.assertEqual(opt.t.tolist(), [2, 1])
        self.assertTrue(all(bool((v > 0).all()) for v in opt.m.values()))

    def test_analytic_norm_and_zero(self):
        R = {4: torch.tensor([[3., 0.], [4., 0.]]), 8: torch.zeros(2, 2)}
        loss, grad = analytic_norm(R, torch.tensor([2., 3.]), torch.tensor([True, False]))
        self.assertAlmostEqual(loss[0].item(), .625)
        self.assertTrue(torch.equal(grad[8], torch.zeros_like(grad[8])))
        self.assertTrue(torch.allclose(grad[4][:, 0], torch.tensor([.075, .1], dtype=torch.float64)))


if __name__ == '__main__':
    unittest.main()
