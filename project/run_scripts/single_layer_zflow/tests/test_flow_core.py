"""Small FP64 algebra tests; NOT a Llama checkpoint or performance test."""
import unittest
import math
import torch
from project.run_scripts.single_layer_zflow.flow_core import (
    native_map, preservation_gram, QuadraticGeometry, integrate,
)

DT = torch.float64

def randn(*shape):
    return torch.randn(*shape, dtype=DT)


class TestSLZFlow(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(1)

    def test_native_factorization_with_context_expansion(self):
        din, m, dout = 9, 3, 5
        K = randn(din, 2*m)
        Q, _ = torch.linalg.qr(randn(din, din))
        P = Q[:, :6] @ Q[:, :6].T
        Kh = randn(din, 7)
        M = Kh@Kh.T
        E = torch.eye(m, dtype=DT).repeat_interleave(2, dim=1)
        X = randn(dout, m)
        B = native_map(K, P, M, 1., E)
        N = P@(K@K.T+M)+torch.eye(din, dtype=DT)
        direct = torch.linalg.solve(N, P@K@(X@E).T).T
        torch.testing.assert_close(X@B, direct, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close((X@B)@P, X@B, rtol=1e-10, atol=1e-10)

    def test_cached_affine_forward_and_nonlinear_gradient(self):
        # Whole-sequence downstream attention depends nonlinearly on X.
        dout, din, m, tokens = 6, 9, 3, 8
        W, K, residual, B = randn(dout, din), randn(din, tokens), randn(dout, tokens), randn(m, din)
        H0, A = residual+W@K, B@K
        head, target = randn(dout, 11), torch.randint(0, 11, (tokens,))
        def suffix(h):
            rows = h.T
            attention = torch.softmax(rows@rows.T / dout**0.5, dim=-1)
            logits = torch.tanh(rows+attention@rows)@head
            return torch.nn.functional.cross_entropy(logits, target)
        x1 = randn(dout, m).requires_grad_()
        x2 = x1.detach().clone().requires_grad_()
        l1 = suffix(residual+(W+x1@B)@K)
        l2 = suffix(H0+x2@A)
        g1, = torch.autograd.grad(l1, x1)
        g2, = torch.autograd.grad(l2, x2)
        torch.testing.assert_close(l1, l2, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(g1, g2, rtol=1e-10, atol=1e-10)

    def test_quadratic_cost_and_gradient(self):
        B, X, Kh = randn(3, 8), randn(5, 3), randn(8, 4)
        M = Kh@Kh.T
        S = preservation_gram(B, M, 1.)
        geom = QuadraticGeometry(S)
        dW = X@B
        direct = ((dW@Kh).square().sum()+dW.square().sum())/(2*X.shape[1])
        self.assertAlmostEqual(geom.cost(X), float(direct), places=10)
        leaf = X.clone().requires_grad_()
        loss = ((leaf@S)*leaf).sum()/2
        grad, = torch.autograd.grad(loss, leaf)
        torch.testing.assert_close(geom.cost_grad(X), grad)

    def test_imex_matches_small_matrix_solve(self):
        T = randn(4, 4); S = T@T.T+.2*torch.eye(4, dtype=DT)
        geom = QuadraticGeometry(S)
        X, g, eta, price = randn(5, 4), randn(5, 4), .7, .4
        H = S+geom.epsilon*torch.eye(4, dtype=DT)
        expected = torch.linalg.solve(H+eta*price*S, (X@H-eta*g).T).T
        actual = geom.propose(X, g, eta, price).x
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_discrete_barrier_at_zero_and_segment(self):
        geom = QuadraticGeometry(torch.diag(torch.tensor([.2, 1., 3.], dtype=DT)))
        X, g = torch.zeros((2, 3), dtype=DT), 10*randn(2, 3)
        trial = geom.propose(X, g, 1., .1, budget=.5, kappa=.7)
        self.assertGreater(trial.dual, 0.)
        self.assertLessEqual(trial.cost, trial.allowed_cost+1e-10)
        self.assertGreater(trial.scalar_iterations, 0)
        for alpha in torch.linspace(0, 1, 11):
            self.assertLessEqual(geom.cost(X+alpha*(trial.x-X)), .5+1e-10)

    def test_convergence_and_gradient_carry(self):
        S = torch.diag(torch.tensor([.7, 1., 1.4], dtype=DT))
        geom = QuadraticGeometry(S)
        target = randn(2, 3)
        calls = [0]
        def oracle(x):
            calls[0] += 1
            diff = x-target
            return float(diff.square().sum()/2), diff
        price = .3
        result = integrate(oracle, torch.zeros_like(target), geom, price=price,
                           initial_step=5., max_oracle_calls=400,
                           relative_stationarity=1e-8)
        expected = torch.linalg.solve(torch.eye(3, dtype=DT)+price*S, target.T).T
        self.assertEqual(result.status, 'FIRST_ORDER_STATIONARY')
        torch.testing.assert_close(result.x, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(calls[0], 1+result.accepted_steps+result.rejected_steps)
        self.assertGreater(result.rejected_steps, 0)

    def test_budget_constrained_stationarity(self):
        S = torch.diag(torch.tensor([.7, 1., 1.4], dtype=DT))
        geom = QuadraticGeometry(S)
        target = 3*randn(2, 3)
        def oracle(x):
            diff = x-target
            return float(diff.square().sum()/2), diff
        result = integrate(oracle, torch.zeros_like(target), geom, price=.2, budget=.3,
                           initial_step=1., max_oracle_calls=600,
                           relative_stationarity=1e-7)
        self.assertEqual(result.status, 'FIRST_ORDER_STATIONARY')
        self.assertLessEqual(geom.cost(result.x), .3+1e-9)
        self.assertAlmostEqual(geom.cost(result.x), .3, places=6)
        for event in result.trace:
            if event['accepted']:
                self.assertLessEqual(event['cost'], .3+1e-9)

    def test_resource_stop_is_not_convergence(self):
        geom = QuadraticGeometry(torch.eye(2, dtype=DT))
        def oracle(x):
            d = x-1
            return float(d.square().sum()/2), d
        result = integrate(oracle, torch.zeros((2,2), dtype=DT), geom, price=.1,
                           max_oracle_calls=1)
        self.assertEqual(result.status, 'RESOURCE_STOP')

    def test_invalid_geometry_and_budget(self):
        with self.assertRaises(ValueError):
            QuadraticGeometry(torch.zeros((2,2), dtype=DT))
        with self.assertRaises(ValueError):
            QuadraticGeometry(torch.diag(torch.tensor([-1., 1.], dtype=DT)))
        geom = QuadraticGeometry(torch.eye(2, dtype=DT))
        with self.assertRaises(ValueError):
            geom.propose(torch.ones((2,2), dtype=DT), torch.zeros((2,2),dtype=DT),
                         1., .2, budget=.1)

    def test_tiny_budget_does_not_label_interior_stationary(self):
        geom = QuadraticGeometry(torch.ones((1, 1), dtype=DT))
        def oracle(x):
            diff = x-1
            return float(diff.square().sum()/2), diff
        for budget in (1e-16, 1e-10, .1):
            with self.subTest(budget=budget):
                result = integrate(oracle, torch.zeros((1, 1), dtype=DT), geom,
                                   price=.2, budget=budget, max_oracle_calls=100)
                self.assertEqual(result.status, 'FIRST_ORDER_STATIONARY')
                self.assertGreaterEqual(geom.cost(result.x)/budget, 1-1e-8)
                self.assertAlmostEqual(float(result.x)/math.sqrt(2*budget), 1., places=7)
                self.assertLessEqual(result.terminal_stats['normalized_complementarity'], 1e-7)
        # This is the original counterexample's interior state, at 63% of b.
        budget = 1e-16
        interior = torch.tensor([[math.sqrt(2*budget*(1-math.exp(-1)))]], dtype=DT)
        stats = geom.stationary(interior, oracle(interior)[1], .2, budget)
        self.assertEqual(stats['normal_multiplier'], 0.)
        self.assertGreater(stats['residual'], .9)

    def test_barrier_modes_and_feasibility(self):
        geom = QuadraticGeometry(torch.ones((1, 1), dtype=DT))
        x, g = torch.zeros((1, 1), dtype=DT), torch.tensor([[-10.]], dtype=DT)
        for budget in (1e-16, .1):
            for mode in ('fixed', 'exponential'):
                with self.subTest(budget=budget, mode=mode):
                    proposal = geom.propose(x, g, 1., .2, budget,
                                            barrier_mode=mode)
                    self.assertLessEqual(proposal.cost/budget, 1+1e-12)
                    expected_cap = budget if mode == 'fixed' else (1-math.exp(-1))*budget
                    self.assertAlmostEqual(proposal.allowed_cost/expected_cap, 1.)
                    self.assertAlmostEqual(proposal.cost/expected_cap, 1., places=9)
                    for alpha in (.1, .5, 1.):
                        self.assertLessEqual(geom.cost(alpha*proposal.x)/budget, 1+1e-12)
        off = geom.propose(x, g, 1., .2, .1, barrier_mode='off')
        no_budget = geom.propose(x, g, 1., .2, None, barrier_mode='fixed')
        torch.testing.assert_close(off.x, no_budget.x)
        self.assertTrue(math.isinf(off.allowed_cost))
        self.assertGreater(off.cost, .1)

    def test_bounded_step_recovery_after_rejection(self):
        geom = QuadraticGeometry(torch.eye(2, dtype=DT))
        target = torch.tensor([[1., 2.]], dtype=DT)
        curvature = torch.tensor([[1., 10.]], dtype=DT)
        calls = []
        def oracle(x):
            calls.append(x.clone())
            diff = x-target
            return float((curvature*diff.square()).sum()/2), curvature*diff
        result = integrate(oracle, torch.zeros_like(target), geom, price=.2,
                           initial_step=2., maximum_step=2., max_oracle_calls=150,
                           relative_stationarity=1e-8)
        self.assertGreater(result.rejected_steps, 0)
        self.assertEqual(len(calls), result.oracle_calls)
        self.assertEqual(result.oracle_calls, 1+result.accepted_steps+result.rejected_steps)
        steps = [row['step'] for row in result.trace]
        self.assertLessEqual(max(steps), 2.)
        grew = False
        for index in range(1, len(steps)):
            if steps[index] > steps[index-1]:
                grew = True
                self.assertGreaterEqual(index, 2)
                self.assertEqual(result.trace[index-1]['accepted'], 1.)
                self.assertEqual(result.trace[index-2]['accepted'], 1.)
                self.assertLessEqual(steps[index]/steps[index-1], 1.5+1e-12)
        self.assertTrue(grew)
        expected = curvature*target/(curvature+.2)
        torch.testing.assert_close(result.x, expected, rtol=2e-6, atol=2e-6)

    def test_invalid_configuration_checked_before_oracle(self):
        geom = QuadraticGeometry(torch.ones((1, 1), dtype=DT))
        calls = []
        def oracle(x):
            calls.append(1)
            return 0., torch.zeros_like(x)
        invalid = [
            {'price': float('nan')}, {'price': -1.}, {'budget': 0.},
            {'budget': float('inf')}, {'kappa': float('nan')},
            {'initial_step': 0.}, {'minimum_step': -1.},
            {'maximum_step': .5}, {'step_growth': .5},
            {'step_growth': float('inf')}, {'growth_after_accepts': 0},
            {'growth_after_accepts': 1.5}, {'max_oracle_calls': 0},
            {'max_oracle_calls': 2.5}, {'relative_stationarity': -1.},
            {'absolute_stationarity': float('nan')},
            {'complementarity_tolerance': 0.}, {'feasibility_tolerance': -1.},
            {'active_tolerance': float('inf')}, {'barrier_mode': 'mystery'},
            {'armijo': .5},
        ]
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                options = {'price': .2, **overrides}
                integrate(oracle, torch.zeros((1, 1), dtype=DT), geom, **options)
        with self.assertRaises(ValueError):
            integrate(oracle, torch.tensor([[1e-7]], dtype=DT), geom,
                      price=.2, budget=1e-16)
        self.assertEqual(calls, [])

    def test_geometry_rejects_nonfinite_settings(self):
        for value in (float('nan'), float('inf'), 0.):
            with self.subTest(value=value), self.assertRaises(ValueError):
                QuadraticGeometry(torch.eye(2, dtype=DT), metric_ridge_relative=value)
        geom = QuadraticGeometry(torch.eye(2, dtype=DT))
        with self.assertRaises(ValueError):
            geom.propose(torch.zeros((1, 2), dtype=DT), torch.ones((1, 2), dtype=DT),
                         1., .1, budget=.1, kappa=float('nan'))
        with self.assertRaises(ValueError):
            preservation_gram(torch.ones((1, 2), dtype=DT), torch.eye(2, dtype=DT),
                              float('inf'))

    def test_final_stats_and_resource_stop(self):
        geom = QuadraticGeometry(torch.ones((1, 1), dtype=DT))
        def oracle(x):
            return float(torch.nn.functional.softplus(-x).sum()), -torch.sigmoid(-x)
        result = integrate(oracle, torch.zeros((1, 1), dtype=DT), geom,
                           price=0., max_oracle_calls=3, barrier_mode='off')
        self.assertEqual(result.status, 'RESOURCE_STOP')
        self.assertEqual(result.oracle_calls, 3)
        self.assertEqual(result.accepted_steps, 2)
        L, g = oracle(result.x)
        self.assertEqual(result.terminal_stats['L'], L)
        self.assertEqual(result.terminal_stats['C'], geom.cost(result.x))
        self.assertEqual(result.terminal_stats['F'], L)
        self.assertEqual(result.terminal_stats['residual'], geom.stationary(result.x, g, 0.)['residual'])
        for field in ('slack', 'relative_slack', 'complementarity',
                      'normalized_complementarity', 'normalized_feasibility'):
            self.assertIn(field, result.terminal_stats)

    def test_numerical_stop_and_nonfinite_rejection_accounting(self):
        geom = QuadraticGeometry(torch.ones((1, 1), dtype=DT))
        calls = []
        def oracle(x):
            calls.append(1)
            if len(calls) == 1:
                return 1., -torch.ones_like(x)
            return float('nan'), torch.full_like(x, float('nan'))
        result = integrate(oracle, torch.zeros((1, 1), dtype=DT), geom,
                           price=.1, initial_step=1., minimum_step=.2,
                           max_oracle_calls=20)
        self.assertEqual(result.status, 'NUMERICAL_STOP')
        self.assertEqual(result.accepted_steps, 0)
        self.assertEqual(result.rejected_steps, 3)
        self.assertEqual(result.oracle_calls, len(calls))
        self.assertEqual(result.oracle_calls, 1+result.rejected_steps)
        self.assertEqual(result.terminal_stats['L'], 1.)

if __name__ == '__main__':
    unittest.main(verbosity=2)
