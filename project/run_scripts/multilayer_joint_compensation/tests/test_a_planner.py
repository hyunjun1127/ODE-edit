"""Compact A0 CPU tests; no model load, native target, GPU or Slurm."""
import unittest

import torch

from project.run_scripts.multilayer_joint_compensation.track_a.planner import (
    PlannerBoundary, plan_joint_targets,
)


def squared_callback(targets, observations=None):
    def callback(deltas):
        if observations is not None:
            observations.append(tuple(d.clone() for d in deltas))
        errors = tuple(d - t for d, t in zip(deltas, targets))
        # This is already a logical mean; planner must not divide by B again.
        loss = sum(.5 * e.square().mean() for e in errors)
        return loss, tuple(e / e.numel() for e in errors), {"logical_mean": True}
    return callback


class PlannerTests(unittest.TestCase):
    def test_b0_no_callback(self):
        def forbidden(_):
            raise AssertionError("B0 callback")
        result = plan_joint_targets((torch.empty(0, 3), torch.empty(0, 4)),
                                    (torch.empty(0, 0, dtype=torch.float64),) * 2,
                                    (2, 2), forbidden)
        self.assertEqual(result.final["status"], "B0_NOOP")
        self.assertEqual(result.counts["callback_calls"], 0)
        self.assertEqual(tuple(result.deltas[1].shape), (2, 4))
        self.assertEqual(float(result.deltas[1].sum()), 0.)

    def test_zero_entry_fixed_q_and_25_updates(self):
        seen = []
        factors = (torch.eye(2), torch.eye(2))
        metrics = (torch.eye(2, dtype=torch.float64),) * 2
        result = plan_joint_targets(factors, metrics, (2, 2),
                                    squared_callback((torch.ones(2, 2), torch.full((2, 2), 2.)), seen))
        self.assertEqual(len(seen), 26)
        self.assertTrue(all(torch.count_nonzero(d) == 0 for d in seen[0]))
        self.assertEqual(result.counts["optimizer_updates"], 25)
        self.assertEqual(result.counts["joint_gradient_calls"], 25)
        self.assertEqual(result.counts["q_captures"], 1)
        self.assertEqual(result.counts["final_observation_calls"], 1)
        self.assertEqual(result.counts["callback_weight_gradient_sets"], 26)
        self.assertEqual(result.q_raw, (.25, 1.))
        self.assertEqual(result.sigma_e, 2.5)
        self.assertEqual(result.trajectory[0]["writer_metric_energy"], [0., 0.])
        self.assertTrue(all(row["position"] == "pre_update" for row in result.trajectory))
        self.assertTrue(all(torch.equal(d, r @ j) for d, r, j in zip(result.deltas, result.rhs, factors)))

    def test_adam_matches_independent_autograd_reference(self):
        torch.manual_seed(17)
        factors = (torch.randn(3, 4), torch.randn(3, 5) * .02)
        metrics = tuple(j.double() @ j.double().T for j in factors)
        targets = (torch.randn(2, 4), torch.randn(2, 5))
        result = plan_joint_targets(factors, metrics, (2, 2), squared_callback(targets))
        parameters = [torch.zeros(2, 3, requires_grad=True) for _ in range(2)]
        optimizer = torch.optim.Adam(parameters, lr=.1, betas=(.9, .999), eps=1e-8,
                                     foreach=False)
        for _ in range(25):
            loss = sum(.5 * (r @ j - t).square().mean()
                       for r, j, t in zip(parameters, factors, targets))
            penalty = sum(q * torch.sum((r.double() @ metric) * r.double())
                          for q, r, metric in zip(result.q_balanced, parameters, metrics))
            total = loss + .1 * penalty / (2 * result.sigma_e**2)
            total.backward()
            optimizer.step()
            optimizer.zero_grad()
        for got, expected in zip(result.rhs, parameters):
            torch.testing.assert_close(got, expected, atol=3e-7, rtol=3e-6)

    def test_duplicate_rhs_pseudoinverse_and_null(self):
        j = torch.tensor([[1., 2.], [1., 2.], [0., 0.]])
        result = plan_joint_targets((j,), (j.double() @ j.double().T,), (1,),
                                    squared_callback((torch.tensor([[3., 1.]]),)), layer_ids=(4,))
        self.assertEqual(result.metric_receipts[0]["retained_rank"], 1)
        self.assertEqual(result.rhs[0][0, 2], 0.)
        torch.testing.assert_close(result.rhs[0][:, 0], result.rhs[0][:, 1])

    def test_zero_writer_and_exact_zero_q(self):
        j = torch.zeros(7, 3)
        result = plan_joint_targets((j,), (torch.zeros(7, 7, dtype=torch.float64),), (2,),
                                    squared_callback((torch.ones(2, 3),)), layer_ids=(4,))
        self.assertEqual(result.q_raw, (0.,))
        self.assertEqual(result.q_exact_zero, (True,))
        self.assertEqual(result.q_floor, 1e-12)
        self.assertEqual(torch.count_nonzero(result.deltas[0]), 0)
        self.assertEqual(result.counts["optimizer_updates"], 25)

    def test_joint_nonlinear_full_weight_forward(self):
        torch.manual_seed(20260911)
        w4, w8 = torch.randn(3, 4), torch.randn(2, 3)
        x, target = torch.randn(7, 4), torch.randn(7, 2)
        factors = (torch.randn(2, 4), torch.randn(2, 3))
        metrics = tuple(j.double() @ j.double().T for j in factors)
        seen = []

        def callback(deltas):
            ds = [d.detach().clone().requires_grad_(True) for d in deltas]
            output = torch.tanh(x @ (w4 + ds[0]).T) @ (w8 + ds[1]).T
            loss = .5 * (output - target).square().mean()
            gradients = torch.autograd.grad(loss, ds)
            seen.append(float(loss.detach()))
            return loss.detach(), tuple(g.detach() for g in gradients), {}

        result = plan_joint_targets(factors, metrics, (3, 2), callback)
        output = torch.tanh(x @ (w4 + result.deltas[0]).T) @ (w8 + result.deltas[1]).T
        self.assertAlmostEqual(float(.5 * (output - target).square().mean()), result.final["current_nll"])
        self.assertEqual(len(seen), 26)
        # This test checks coupled full forward, not a promised efficacy gate.
        self.assertTrue(torch.count_nonzero(result.deltas[0]) > 0)
        self.assertTrue(torch.count_nonzero(result.deltas[1]) > 0)

    def test_unequal_physical_scales_image_efficiency(self):
        factors = (torch.eye(2) * .001, torch.eye(2) * 20.)
        metrics = tuple(j.double() @ j.double().T for j in factors)
        result = plan_joint_targets(factors, metrics, (1, 1),
                                    squared_callback((torch.ones(1, 2),) * 2))
        self.assertAlmostEqual(result.q_raw[0], result.q_raw[1], places=6)
        self.assertEqual(result.deltas[0].dtype, torch.float32)

    def test_inputs_captured_without_mutation(self):
        j, metric = torch.eye(2), torch.eye(2, dtype=torch.float64)
        before = (j.clone(), metric.clone())
        result = plan_joint_targets((j,), (metric,), (1,),
                                    squared_callback((torch.ones(1, 2),)),
                                    layer_ids=(4,), balance_weight=0.)
        self.assertTrue(torch.equal(j, before[0]))
        self.assertTrue(torch.equal(metric, before[1]))
        self.assertEqual(result.trajectory[12]["balance_weight"], 0.)
        self.assertEqual(result.counts["native_z_calls"], 0)

    def test_fail_closed_input_and_nonfinite(self):
        cases = (torch.diag(torch.tensor([1., -1.], dtype=torch.float64)),
                 torch.tensor([[1., 1.], [0., 1.]], dtype=torch.float64))
        for metric in cases:
            with self.subTest(metric=metric):
                with self.assertRaises(PlannerBoundary):
                    plan_joint_targets((torch.eye(2),), (metric,), (1,),
                                       squared_callback((torch.ones(1, 2),)), layer_ids=(4,))
        with self.assertRaisesRegex(PlannerBoundary, "nonfinite Current"):
            plan_joint_targets((torch.eye(2),), (torch.eye(2, dtype=torch.float64),), (1,),
                               lambda ds: (float("nan"), (torch.zeros_like(ds[0]),), {}), layer_ids=(4,))

    def test_loss_gradient_not_redivided_by_batch(self):
        result = plan_joint_targets((torch.eye(7),), (torch.eye(7, dtype=torch.float64),), (1,),
                                    squared_callback((torch.ones(1, 7),)), layer_ids=(4,), balance_weight=0.)
        # Initial physical mean gradient is -1/7 in seven dimensions => q=1/7.
        self.assertAlmostEqual(result.q_raw[0], 1 / 7, places=7)
        self.assertEqual(result.sigma_e, .5)

    def test_arbitrary_batch_rhs_no_rank100_cutoff(self):
        for batch in (1, 7, 64, 100, 257):
            with self.subTest(batch=batch):
                # Wide B with duplicate columns has the same physical image;
                # no RHS truncation to 100 or 256 is permissible.
                j = torch.ones(batch, 2) / batch
                result = plan_joint_targets((j,), (j.double() @ j.double().T,), (1,),
                                            squared_callback((torch.ones(1, 2),)), layer_ids=(4,))
                self.assertEqual(tuple(result.rhs[0].shape), (1, batch))
                self.assertEqual(result.metric_receipts[0]["retained_rank"], 1)
                self.assertEqual(result.counts["optimizer_updates"], 25)

    def test_logical_mean_microbatch_accumulation(self):
        torch.manual_seed(4)
        x, target = torch.randn(7, 3), torch.randn(7, 2)
        weights = (torch.randn(3, 3), torch.randn(2, 3))
        factors = (torch.randn(2, 3), torch.randn(2, 3))
        metrics = tuple(j.double() @ j.double().T for j in factors)

        def make_callback(microbatch):
            def callback(deltas):
                ds = [d.clone().requires_grad_(True) for d in deltas]
                grads = [torch.zeros_like(d) for d in ds]
                loss_sum = 0.
                for start in range(0, len(x), microbatch):
                    xx, yy = x[start:start + microbatch], target[start:start + microbatch]
                    output = torch.tanh(xx @ (weights[0] + ds[0]).T) @ (weights[1] + ds[1]).T
                    # One common logical denominator, including the uneven tail.
                    loss = .5 * (output - yy).square().sum() / target.numel()
                    loss_sum += float(loss.detach())
                    for acc, grad in zip(grads, torch.autograd.grad(loss, ds)):
                        acc.add_(grad.detach())
                return loss_sum, tuple(grads), {"microbatch": microbatch}
            return callback

        plans = [plan_joint_targets(factors, metrics, (3, 2), make_callback(m)) for m in (1, 2, 4)]
        for candidate in plans[1:]:
            for expected, got in zip(plans[0].deltas, candidate.deltas):
                torch.testing.assert_close(got, expected, atol=3e-6, rtol=3e-5)
            self.assertAlmostEqual(candidate.sigma_e, plans[0].sigma_e, places=6)


if __name__ == "__main__":
    unittest.main()
