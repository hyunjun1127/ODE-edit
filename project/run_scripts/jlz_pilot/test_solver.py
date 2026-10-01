"""CPU tests using closed-form group lasso and independent KKT conditions."""

import unittest

import torch

from .solver import block_kkt, prox_blocks, solve


class SolverTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    @staticmethod
    def quadratic(target, curvature=None):
        if curvature is None:
            curvature = torch.ones_like(target)

        def oracle(x):
            variable = x.detach().requires_grad_(True)
            loss = (0.5 * curvature * (variable - target).square()).sum()
            gradient, = torch.autograd.grad(loss, variable)
            return loss, gradient, {"point": variable, "call_loss": loss}

        return oracle

    def test_quadratic_known_group_solution_zero_interior_boundary(self):
        target = torch.tensor([[0.2, 0.1], [3.0, 4.0], [0.0, 4.0]])
        c = torch.tensor([0.5, 1.0, 0.5])
        rho = torch.tensor([3.0, 6.0, 1.0])
        active = torch.ones(3, dtype=torch.bool)
        result = solve(self.quadratic(target), torch.zeros_like(target), c, rho, active, tol=1e-6, cap=100)
        expected = torch.tensor([[0.0, 0.0], [2.4, 3.2], [0.0, 1.0]])
        self.assertEqual(result["status"], "CONVERGED")
        torch.testing.assert_close(result["x"], expected, atol=2e-6, rtol=2e-6)
        self.assertEqual([row["state"] for row in result["kkt"]], ["ZERO", "INTERIOR", "BOUNDARY"])
        self.assertLess(max(row["stationarity_violation"] for row in result["kkt"]), 3e-6)
        self.assertLess(result["feasibility_violation"], 2e-7)
        self.assertTrue(result["commit_eligible"])

    def test_inactive_and_zero_radius_remain_exact_zero(self):
        target = torch.tensor([[2.0, 3.0], [100.0, 100.0], [10.0, -3.0]])
        result = solve(self.quadratic(target), torch.ones_like(target), torch.zeros(3),
                       torch.tensor([4.0, 3.0, 0.0]), torch.tensor([True, False, True]), tol=1e-6)
        self.assertEqual(result["status"], "CONVERGED")
        torch.testing.assert_close(result["x"][0], target[0])
        self.assertEqual(int(torch.count_nonzero(result["x"][1:])), 0)
        self.assertEqual([row["state"] for row in result["kkt"]][1:], ["INACTIVE", "FIXED_ZERO"])

    def test_return_payload_is_fresh_at_returned_point_and_detached(self):
        seen = []
        original = self.quadratic(torch.tensor([[2.0, 3.0]]))

        def oracle(x):
            seen.append(x.detach().clone())
            value, grad, payload = original(x)
            payload["call_index"] = len(seen)
            return value, grad, payload

        x0 = torch.zeros((1, 2), requires_grad=True)
        result = solve(oracle, x0, torch.tensor([0.2]), torch.tensor([4.0]), torch.tensor([True]), cap=25)
        self.assertEqual(result["calls"], len(seen))
        self.assertEqual(result["final_payload"]["call_index"], len(seen))
        torch.testing.assert_close(result["final_payload"]["point"], result["x"], rtol=0, atol=0)
        torch.testing.assert_close(seen[-1], result["x"], rtol=0, atol=0)
        self.assertFalse(result["x"].requires_grad)
        self.assertFalse(result["gradient"].requires_grad)
        self.assertFalse(result["final_payload"]["point"].requires_grad)
        self.assertFalse(result["final_payload"]["call_loss"].requires_grad)
        self.assertIsNone(x0.grad)

    def test_call_cap_includes_reserved_return_evaluation(self):
        for cap in (2, 3, 4, 7):
            with self.subTest(cap=cap):
                target = torch.tensor([[2.0, 3.0, 1.0]])
                result = solve(self.quadratic(target, torch.tensor([[1.0, 3.0, 9.0]])),
                               torch.zeros_like(target), torch.tensor([0.01]), torch.tensor([8.0]),
                               torch.tensor([True]), tol=1e-12, cap=cap)
                self.assertLessEqual(result["calls"], cap)
                self.assertTrue(result["final_recomputed"])
                self.assertEqual(result["status"], "BUDGET_STOP")
                self.assertEqual(result["reason"], "CALL_CAP")
                self.assertFalse(result["commit_eligible"])

    def test_time_budget_preserves_return_call(self):
        result = solve(self.quadratic(torch.tensor([[3.0, 4.0]])), torch.zeros((1, 2)),
                       torch.tensor([0.2]), torch.tensor([3.0]), torch.tensor([True]), max_seconds=0)
        self.assertEqual(result["status"], "BUDGET_STOP")
        self.assertEqual(result["reason"], "TIME_LIMIT")
        self.assertEqual(result["calls"], 2)
        self.assertTrue(result["final_recomputed"])

    def test_nonfinite_trial_returns_safe_point_but_disallows_commit(self):
        seen = []

        def oracle(x):
            seen.append(x.clone())
            if len(seen) == 2:
                return float("nan"), torch.ones_like(x), {"bad": True}
            return float((x - 2).square().sum()), 2 * (x - 2), {"point": x.clone()}

        result = solve(oracle, torch.zeros((1, 2)), torch.tensor([0.0]), torch.tensor([3.0]), torch.tensor([True]))
        self.assertEqual(result["status"], "NONFINITE")
        self.assertEqual(result["calls"], 3)
        self.assertTrue(result["final_recomputed"])
        self.assertFalse(result["commit_eligible"])
        torch.testing.assert_close(result["x"], torch.zeros((1, 2)), rtol=0, atol=0)
        self.assertNotIn("bad", result["final_payload"])

    def test_nonfinite_return_clears_payload(self):
        calls = 0

        def oracle(x):
            nonlocal calls
            calls += 1
            if calls == 2:
                return 0.0, torch.full_like(x, float("inf")), {"bad": True}
            return 1.0, torch.ones_like(x), {"point": x}

        result = solve(oracle, torch.zeros((1, 2)), torch.tensor([0.0]), torch.tensor([3.0]), torch.tensor([True]), cap=2)
        self.assertEqual(result["status"], "NONFINITE")
        self.assertFalse(result["final_recomputed"])
        self.assertFalse(result["commit_eligible"])
        self.assertIsNone(result["final_payload"])

    def test_fp32_plateau_is_not_convergence(self):
        curvature = torch.tensor([[1.0, 3.0, 13.0]])
        target = torch.tensor([[0.7, -0.31, 0.27]])

        def oracle(x):
            variable = x.detach().requires_grad_(True)
            value = torch.tensor(1e8) + 0.5 * (curvature * (variable - target).square()).sum()
            gradient, = torch.autograd.grad(value, variable)
            return value, gradient, None

        result = solve(oracle, torch.zeros_like(target), torch.tensor([0.0]), torch.tensor([10.0]),
                       torch.tensor([True]), tol=1e-30, cap=100)
        self.assertEqual(result["status"], "STALLED_AT_PRECISION")
        self.assertGreater(result["normalized_residual"], 1e-30)
        self.assertGreaterEqual(result["accepted_steps"], 6)
        self.assertFalse(result["commit_eligible"])

    def test_boundary_certificate_rejects_positive_radial_gradient(self):
        x = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        gradient = torch.tensor([[0.0, 0.0], [-0.8, 0.0]])
        rows = block_kkt(x, gradient, torch.tensor([0.5, 0.5]), torch.ones(2), torch.ones(2, dtype=torch.bool))
        self.assertAlmostEqual(rows[0]["stationarity_violation"], 0.5)
        self.assertAlmostEqual(rows[1]["stationarity_violation"], 0.0)
        self.assertAlmostEqual(rows[1]["clamp_multiplier"], 0.3, places=6)

    def test_tiny_radius_interior_not_misclassified_as_boundary(self):
        rows = block_kkt(torch.tensor([[5e-9, 0.0]]), torch.zeros((1, 2)), torch.tensor([0.1]),
                         torch.tensor([1e-8]), torch.tensor([True]))
        self.assertEqual(rows[0]["state"], "INTERIOR")

    def test_line_search_failure_keeps_accepted_point(self):
        calls = 0

        def oracle(x):
            nonlocal calls
            calls += 1
            # Discontinuous fixture intentionally rejects every nonzero trial.
            value = 0.0 if bool((x == 0).all()) else 100.0
            return value, torch.ones_like(x), {"point": x.clone()}

        result = solve(oracle, torch.zeros((1, 2)), torch.tensor([0.0]), torch.tensor([4.0]),
                       torch.tensor([True]), max_trials=3, cap=20)
        self.assertEqual(result["status"], "LINESEARCH_FAILED")
        self.assertEqual(result["backtracks"], 3)
        self.assertEqual(result["calls"], 5)
        torch.testing.assert_close(result["x"], torch.zeros((1, 2)), rtol=0, atol=0)

    def test_invalid_inputs_fail_before_oracle(self):
        def forbidden(_):
            self.fail("oracle should not be called")

        args = (forbidden, torch.zeros((1, 2)), torch.tensor([0.1]), torch.tensor([1.0]), torch.tensor([True]))
        with self.assertRaisesRegex(ValueError, "RETURN_EVALUATION_RESERVE"):
            solve(*args, cap=1)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            solve(forbidden, args[1], torch.tensor([-0.1]), args[3], args[4])


if __name__ == "__main__":
    unittest.main()
