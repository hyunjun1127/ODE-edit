"""Bounded CPU regression for policy changes, accounting and pinned math."""

import json
import unittest

import torch

from project.run_scripts.jlz_pilot.solver import solve as pinned_solve
from .solver import BudgetAccountant, budgeted_oracle_evaluate, prox_blocks, solve


class SolverTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x = torch.zeros((1, 2))
        self.c = torch.tensor([0.0])
        self.rho = torch.tensor([4.0])
        self.mask = torch.tensor([True])

    def run_solver(self, fun, **kwargs):
        return solve(fun, self.x, self.c, self.rho, self.mask, **kwargs)

    @staticmethod
    def quadratic(x):
        target = torch.tensor([[2.0, 3.0]])
        return float(0.5 * (x - target).square().sum()), x - target, {"point": x.clone()}

    def test_pinned_trajectory_and_math_unchanged(self):
        kwargs = dict(tol=1e-8, cap=32, max_trials=50, memory=10, sigma=1e-4)
        actual = self.run_solver(self.quadratic, **kwargs)
        expected = pinned_solve(self.quadratic, self.x, self.c, self.rho, self.mask, **kwargs)
        for name in ("x", "gradient"):
            self.assertTrue(torch.equal(actual[name], expected[name]))
        for name in ("calls", "history", "backtracks", "accepted_steps", "status"):
            self.assertEqual(actual[name], expected[name])

    def test_exact_prox_inactive_and_ball(self):
        value = torch.tensor([[3., 4.], [3., 4.], [2., 0.]])
        actual = prox_blocks(value, 1., torch.tensor([1., 0., 9.]),
                             torch.tensor([2., 20., 4.]), torch.tensor([True, False, True]))
        self.assertTrue(torch.equal(actual, torch.tensor([[1.2, 1.6], [0., 0.], [0., 0.]])))

    def test_nonfinite_trial_rejects_halves_and_keeps_last_point(self):
        seen = []
        def oracle(x):
            seen.append(x.clone())
            if len(seen) == 2:
                raise FloatingPointError("NONFINITE_TRIAL_FIXTURE")
            return self.quadratic(x)
        result = self.run_solver(oracle, cap=4)
        self.assertEqual(result["calls"], 4)
        self.assertEqual(result["nonfinite_trials"], 1)
        self.assertEqual(result["trace"][1]["decision"], "REJECT_NONFINITE_TRIAL")
        torch.testing.assert_close(seen[2], seen[1] * .5, rtol=0, atol=0)
        self.assertTrue(torch.equal(result["x"], seen[2]))
        self.assertTrue(torch.equal(seen[-1], result["x"]))
        self.assertTrue(result["commit_eligible"])

    def test_nonfinite_return_value_and_gradient_are_trial_rejections(self):
        for component in ("value", "gradient", "payload"):
            count = 0
            def oracle(x):
                nonlocal count
                count += 1
                if count == 2:
                    if component == "value":
                        return float("nan"), torch.ones_like(x), {}
                    if component == "gradient":
                        return 1., torch.full_like(x, float("inf")), {}
                    return 1., torch.ones_like(x), {"nll": [float("nan")]}
                return self.quadratic(x)
            result = self.run_solver(oracle, cap=3)
            self.assertEqual(result["nonfinite_trials"], 1)
            self.assertTrue(torch.equal(result["x"], self.x))

    def test_final_reserve_caps32_and120_no_hidden_calls(self):
        for cap in (32, 120):
            seen = []
            def oracle(x):
                seen.append(x.clone())
                return float((x - 2.).square().sum()), 2. * (x - 2.), {}
            # Every trial rejects numerically, stopping at the immutable 50
            # backtrack budget or the call cap, whichever comes first.
            def bad_trial(x):
                if seen:
                    seen.append(x.clone())
                    raise FloatingPointError("TRIAL_ONLY")
                return oracle(x)
            def final(x):
                return oracle(x)
            result = self.run_solver(bad_trial, final_fun=final, cap=cap)
            self.assertEqual(result["calls"], min(cap, 52))
            self.assertEqual(len(seen), result["calls"])
            self.assertEqual(result["budget"]["events"][-1], "final")
            self.assertEqual(result["budget"]["events"].count("final"), 1)
            self.assertTrue(result["commit_eligible"])
            self.assertTrue(torch.equal(result["x"], self.x))

    def test_fresh_original_final_not_fast_payload(self):
        routes = []
        def fast(x):
            routes.append("fast")
            return 0., torch.zeros_like(x), {"route": "fast"}
        def original(x):
            routes.append("original")
            return 2., torch.ones_like(x), {"route": "original", "point": x.clone()}
        result = self.run_solver(fast, final_fun=original)
        self.assertEqual(routes, ["fast", "original"])
        self.assertEqual(result["final_payload"]["route"], "original")
        self.assertEqual(result["status"], "NOT_CONVERGED")
        self.assertEqual(result["convergence_original_verdict"], "FAIL")
        self.assertTrue(result["commit_eligible"])

    def test_full_cap_consumption32_and120_preserves_original_final(self):
        for cap in (32, 120):
            seen = []
            def oracle(point):
                seen.append(point.clone())
                x = point.detach().requires_grad_(True)
                loss = (1 - x[0, 0]) ** 2 + 100 * (x[0, 1] - x[0, 0] ** 2) ** 2
                gradient, = torch.autograd.grad(loss, x)
                return loss, gradient, {"point": x}
            result = solve(oracle, torch.tensor([[-1.2, 1.]]), self.c,
                           torch.tensor([10.]), self.mask, cap=cap, tol=1e-30)
            self.assertEqual(result["calls"], cap)
            self.assertEqual(len(seen), cap)
            self.assertEqual(result["budget"]["events"].count("trial"), cap - 2)
            self.assertEqual(result["budget"]["events"][-1], "final")
            self.assertTrue(torch.equal(seen[-1], result["x"]))
            self.assertEqual(result["status"], "BUDGET_STOP")
            self.assertTrue(result["commit_eligible"])

    def test_precision_stall_retains_failed_numerical_verdict(self):
        curvature = torch.tensor([[1., 3., 13.]])
        target = torch.tensor([[.7, -.31, .27]])
        def oracle(point):
            x = point.detach().requires_grad_(True)
            loss = torch.tensor(1e8) + .5 * (curvature * (x - target).square()).sum()
            gradient, = torch.autograd.grad(loss, x)
            return loss, gradient, {"point": x}
        result = solve(oracle, torch.zeros_like(target), self.c,
                       torch.tensor([10.]), self.mask, tol=1e-30)
        self.assertEqual(result["status"], "STALLED_AT_PRECISION")
        self.assertGreaterEqual(result["accepted_steps"], 6)
        self.assertEqual(result["convergence_original_verdict"], "FAIL")
        self.assertTrue(result["commit_eligible"])

    def test_milestones8_16_32_use_accepted_incumbent_without_oracles(self):
        calls = 0
        def oracle(x):
            nonlocal calls
            calls += 1
            # All nonzero proposals fail Armijo; the initial point is the
            # accepted incumbent throughout. Payload weights must never leak.
            loss = 0. if bool((x == 0).all()) else 1e6
            return loss, torch.ones_like(x), dict(
                nll=[1., 3.], kl=[.1, .3],
                general=[{"case_id": 10, "value": .2}, {"case_id": 11, "value": .4}],
                replay=[], omega=.5, weights={4: torch.ones((2, 2))}, point=x)
        result = solve(oracle, torch.zeros((10, 2)), torch.zeros(10), torch.ones(10),
                       torch.ones(10, dtype=torch.bool), cap=32)
        self.assertEqual(calls, 32)
        self.assertEqual(result["calls"], 32)
        milestones = result["milestone_trace"]
        self.assertEqual([m["actual_calls"] for m in milestones], [8, 16, 32])
        self.assertEqual([m["incumbent_evaluation_call"] for m in milestones], [1, 1, 32])
        self.assertEqual([m["fresh_original_final"] for m in milestones], [False, False, True])
        for item in milestones:
            self.assertEqual(item["total"], 0.)
            self.assertEqual(item["accepted_steps"], 0)
            self.assertEqual(item["loss_summary"]["nll"]["mean"], 2.)
            self.assertAlmostEqual(item["loss_summary"]["general"]["mean"], .3)
            self.assertEqual(item["loss_summary"]["replay"]["count"], 0)
            self.assertEqual(item["loss_summary"]["omega"], .5)
            self.assertEqual([v["R_norm"] for v in item["layer_R_norms"]], [0.] * 5)
            self.assertEqual(item["extra_oracle_calls"], 0)
            self.assertIn("NOT_INDEPENDENT_CAP_RERUN", item["interpretation"])
        serialized = json.dumps(milestones, allow_nan=False)
        self.assertNotIn('"weights"', serialized)
        self.assertNotIn('"point"', serialized)
        self.assertEqual(result["milestone_not_reached"], [])

    def test_nonfinite_milestone_retains_prior_finite_incumbent(self):
        calls = 0
        def trial(x):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise FloatingPointError("NONFINITE_TRIAL")
            return self.quadratic(x)
        result = self.run_solver(trial, final_fun=self.quadratic, cap=32)
        self.assertEqual(result["calls"], 32)
        self.assertEqual(result["milestone_trace"][0]["trigger"], "NONFINITE_TRIAL_REJECTED")
        self.assertEqual(result["milestone_trace"][0]["incumbent_evaluation_call"], 1)
        self.assertEqual(result["milestone_trace"][0]["smooth"], 6.5)

    def test_milestones_preserve_all_pinned_rosenbrock_decisions(self):
        def oracle(point):
            x = point.detach().requires_grad_(True)
            value = (1 - x[0, 0]) ** 2 + 100 * (x[0, 1] - x[0, 0] ** 2) ** 2
            gradient, = torch.autograd.grad(value, x)
            return value, gradient, {"nll": [float(value.detach())], "omega": 0.}
        args = (oracle, torch.tensor([[-1.2, 1.]]), self.c, torch.tensor([10.]), self.mask)
        expected = pinned_solve(*args, tol=1e-30, cap=32)
        result = solve(*args, tol=1e-30, cap=32)
        for key in ("calls", "accepted_steps", "backtracks", "history", "status"):
            self.assertEqual(result[key], expected[key])
        self.assertTrue(torch.equal(result["x"], expected["x"]))
        self.assertEqual([r["actual_calls"] for r in result["milestone_trace"]], [8, 16, 32])

    def test_initial_and_final_nonfinite_are_fatal_with_receipts(self):
        def bad(x):
            return float("nan"), torch.zeros_like(x), {}
        for phase in ("initial", "final"):
            with self.assertRaises(FloatingPointError) as caught:
                self.run_solver(bad if phase == "initial" else self.quadratic,
                                final_fun=bad, cap=2)
            evidence = caught.exception.solver_evidence
            self.assertEqual(evidence["phase"], phase)
            self.assertEqual(evidence["trace"][-1]["outcome"], "ERROR")
            self.assertEqual(evidence["budget"]["actual_calls"], 1 if phase == "initial" else 2)

    def test_cuda_state_and_input_mutation_not_swallowed(self):
        for exception in (RuntimeError("CUDA fixture"), ValueError("STATE fixture")):
            count = 0
            def oracle(x):
                nonlocal count
                count += 1
                if count == 2:
                    raise exception
                return self.quadratic(x)
            with self.assertRaises(type(exception)):
                self.run_solver(oracle)
            self.assertEqual(count, 2)
        def mutating(x):
            x.add_(1.)
            raise FloatingPointError("not merely numerical")
        with self.assertRaisesRegex(RuntimeError, "ORACLE_INPUT_MUTATION"):
            self.run_solver(mutating)

    def test_shared_explicit_rechecks_cannot_consume_final_reserve(self):
        account, trace = BudgetAccountant(3), []
        budgeted_oracle_evaluate(self.quadratic, self.x, account, trace, role="initial")
        budgeted_oracle_evaluate(self.quadratic, self.x, account, trace, role="original_recheck")
        with self.assertRaisesRegex(RuntimeError, "FINAL_RESERVED"):
            budgeted_oracle_evaluate(self.quadratic, self.x, account, trace, role="original_recheck")
        budgeted_oracle_evaluate(self.quadratic, self.x, account, trace, role="final")
        self.assertEqual(account.events, ["initial", "original_recheck", "final"])
        self.assertEqual(len(trace), 3)

    def test_inactive_zero_does_not_drop_initial_final_losses(self):
        result = solve(self.quadratic, self.x, self.c, self.rho, torch.tensor([False]))
        self.assertEqual(result["calls"], 2)
        self.assertEqual(result["smooth"], 6.5)
        self.assertEqual(result["status"], "POLICY_ZERO_STEP")
        self.assertTrue(torch.equal(result["x"], self.x))

    def test_inputs_and_output_detached(self):
        original = self.x.clone().requires_grad_(True)
        result = solve(self.quadratic, original, self.c, self.rho, self.mask)
        self.assertTrue(torch.equal(original, self.x))
        self.assertIsNone(original.grad)
        self.assertFalse(result["x"].requires_grad)
        self.assertFalse(result["gradient"].requires_grad)
        self.assertFalse(result["final_payload"]["point"].requires_grad)


if __name__ == "__main__":
    unittest.main()
