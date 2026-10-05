"""Production controller CPU fixtures, not actual-model/GPU qualification."""
import math
import unittest
import weakref

import torch

from .solver import SolverError, SolverLimits, bb_step, mapping, norm_value, prox, solve


class Quadratic:
    def __init__(self, center, hessian=1.):
        self.center = center
        self.hessian = hessian
        self.calls = []; self.reverse_ids = []

    def evaluate(self, u, gradient=False, lambda_Q=None, candidate_id=None):
        frozen = {l: v.detach().clone() for l, v in u.items()}
        payload = dict(u=frozen, candidate_id=candidate_id)
        value = sum(float((.5 * self.hessian * (v - self.center[l]).square()).sum())
                    for l, v in frozen.items())
        result = dict(smooth_sum=value, task_sum=value, Q=0., payload=payload)
        self.calls.append((candidate_id, gradient, frozen))
        if gradient:
            result.update(self.backward(payload, lambda_Q=lambda_Q))
        return result

    def backward(self, payload, lambda_Q=None):
        self.reverse_ids.append(payload["candidate_id"])
        return dict(gradient={l: self.hessian * (v - self.center[l])
                              for l, v in payload["u"].items()})


class SolverTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def close(self, left, right, atol=1e-12, rtol=1e-12):
        torch.testing.assert_close(torch.as_tensor(left, dtype=torch.float64),
                                   torch.as_tensor(right, dtype=torch.float64), atol=atol, rtol=rtol)

    def test_norm_threshold_coordinate_and_independent_layer_caps(self):
        u = {4: torch.tensor([[.4, 1.2, 0.], [0., 1.6, 0.]], dtype=torch.float64),
             8: torch.tensor([[2., 0., 0.]], dtype=torch.float64)}
        anchors = {4: torch.tensor([4., 2., 1.]), 8: torch.ones(3)}
        out = prox(u, {l: torch.zeros_like(v) for l, v in u.items()}, anchors, 1.)
        self.close(out[4].norm(dim=0), [.275, .75, 0.])
        self.close(out[8].norm(dim=0), [.75, 0., 0.])
        # Per-layer cap permits a request's aggregate use above .75.
        self.assertGreater(float(out[4][:, 0].norm() + out[8][:, 0].norm()), .75)
        expected = sum(float((.5 / anchors[l].double() * v.norm(dim=0)).sum()) for l, v in out.items())
        self.close(norm_value(out, anchors), expected)

    def test_zero_norm_and_reentry_no_pruning(self):
        anchors = {4: torch.tensor([2.])}
        small = {4: torch.tensor([[.02], [0.]], dtype=torch.float64)}
        zero = prox(small, {4: torch.zeros_like(small[4])}, anchors, .1)
        self.close(zero[4], torch.zeros(2, 1))
        reentry = prox(zero, {4: torch.tensor([[-1.], [0.]], dtype=torch.float64)}, anchors, .1)
        self.close(reentry[4], [[.075], [0.]])

    def test_zero_initial_gradient_na_and_one_call(self):
        u = {4: torch.zeros(3, 2, dtype=torch.float64)}
        engine = Quadratic(u)
        result = solve(engine, u, {4: torch.ones(2)}, .1)
        self.assertEqual(result.status, "STATIONARY_TOL")
        self.assertIsNone(result.receipt["eta0"]); self.assertIsNone(result.receipt["tau"])
        self.assertEqual(result.receipt["mapping"]["raw_max"], 0.)
        self.assertEqual(len(engine.calls), 1); self.assertEqual(engine.reverse_ids, [0])

    def test_convex_solution_accepted_payload_and_terminal_gradient_reuse(self):
        center = {4: torch.tensor([[.6, -.5]], dtype=torch.float64),
                  8: torch.tensor([[.9, .1]], dtype=torch.float64)}
        u = {l: torch.zeros_like(v) for l, v in center.items()}
        anchors = {l: torch.full((2,), 5.) for l in u}
        engine = Quadratic(center)
        result = solve(engine, u, anchors, .1)
        expected = {4: torch.tensor([[.5, -.4]], dtype=torch.float64),
                    8: torch.tensor([[.75, 0.]], dtype=torch.float64)}
        for l in u:self.close(result.u[l], expected[l], atol=1e-8)
        self.assertEqual(result.status, "STATIONARY_TOL")
        self.assertIs(result.payload["u"][4], engine.calls[result.receipt["accepted_candidate_id"]][2][4])
        self.assertEqual(result.receipt["terminal_extra_gradient"], 0)
        self.assertEqual(result.receipt["full_gradients"], result.receipt["accepted_updates"] + 1)
        accepted_ids = [x["candidate_id"] for x in result.receipt["events"] if x["accepted"]]
        self.assertEqual(engine.reverse_ids, accepted_ids)
        self.assertEqual(result.receipt["tau"], result.receipt["eta0"])

    def test_sum_scaling_no_request_average(self):
        u = {4: torch.zeros(1, 3, dtype=torch.float64)}
        engine = Quadratic({4: torch.ones(1, 3, dtype=torch.float64)})
        result = solve(engine, u, {4: torch.full((3,), 2.)}, .1)
        self.close(result.u[4], [[.75, .75, .75]])
        first = result.receipt["events"][0]
        self.close(first["J_sum"], 1.5); self.close(first["J_mean"], .5)
        self.close(result.receipt["eta0"], .75)

    def test_bb1_nonpositive_curvature_and_safeguards(self):
        old_u = {4: torch.zeros(1, 1, dtype=torch.float64)}
        u = {4: torch.ones(1, 1, dtype=torch.float64)}
        g = {4: torch.zeros(1, 1, dtype=torch.float64)}
        old_g = {4: torch.ones(1, 1, dtype=torch.float64)}
        eta, event = bb_step(u, old_u, g, old_g, .2, .1)
        self.close(eta, .2); self.assertEqual(event["reason"], "LAST_ACCEPTED_ETA")
        eta, event = bb_step(u, old_u, {4: torch.tensor([[1e-12]], dtype=torch.float64)},
                            g, .2, .1)
        self.close(eta, 1e5); self.assertEqual(event["reason"], "POSITIVE_FINITE_BB1")

    def test_rejected_primal_has_no_backward_and_never_returned(self):
        engine = Quadratic({4: torch.tensor([[.01]], dtype=torch.float64)}, hessian=1000.)
        u = {4: torch.zeros(1, 1, dtype=torch.float64)}
        result = solve(engine, u, {4: torch.tensor([10.])}, .1)
        self.assertGreater(result.receipt["rejected_trials"], 0)
        rejected_ids = {e["candidate_id"] for e in result.receipt["events"] if not e["accepted"]}
        self.assertFalse(rejected_ids.intersection(engine.reverse_ids))
        self.assertNotIn(result.receipt["accepted_candidate_id"], rejected_ids)
        accepted = []
        for event in result.receipt["events"]:
            if event["initial"]:accepted.append(event["J_sum"]); continue
            self.close(event["armijo_reference"], max(accepted[-5:]))
            self.assertEqual(event["accepted"], event["J_sum"] <= event["armijo_bound"])
            if event["accepted"]:accepted.append(event["J_sum"])

    def test_50_25_24_budgets_exact_and_finite_budget_commit(self):
        engine = Quadratic({4: torch.tensor([[.01]], dtype=torch.float64)}, hessian=1000.)
        u = {4: torch.zeros(1, 1, dtype=torch.float64)}
        result = solve(engine, u, {4: torch.tensor([10.])}, .1,
                       limits=SolverLimits(max_evaluations=2, max_updates=24, max_gradients=25))
        self.assertEqual(result.status, "LINE_SEARCH_BUDGET")
        self.assertEqual(result.receipt["logical_evaluations"], 2)
        self.assertEqual(result.receipt["full_gradients"], 1)
        self.assertEqual(result.receipt["accepted_candidate_id"], 0)
        self.assertEqual(result.receipt["last_evaluated_candidate_id"], 1)
        self.assertEqual(result.payload["candidate_id"], 0)
        # Update-budget return still has a full accepted gradient, no 26th call.
        engine2 = Quadratic({4: torch.tensor([[.5]], dtype=torch.float64)})
        result2 = solve(engine2, u, {4: torch.tensor([10.])}, .1,
                        limits=SolverLimits(max_updates=0))
        self.assertEqual(result2.status, "BUDGET_NOT_STATIONARY")
        self.assertEqual(len(engine2.calls), 1)
        with self.assertRaises(SolverError):SolverLimits(max_evaluations=51).validate()

    def test_nonfinite_failure_not_budget_status(self):
        class Broken(Quadratic):
            def backward(self, payload, lambda_Q=None):
                return dict(gradient={4: torch.full((1, 1), math.nan)})
        with self.assertRaises(SolverError) as ctx:
            solve(Broken({4: torch.ones(1, 1)}), {4: torch.zeros(1, 1)}, {4: torch.ones(1)}, .1)
        self.assertEqual(ctx.exception.status, "NONFINITE_OR_INVALID_GRADIENT")

    def test_mapping_uses_fixed_tau_and_explicit_denominator(self):
        u = {4: torch.tensor([[.2, 0.]], dtype=torch.float64)}
        g = {4: torch.tensor([[-3., 0.]], dtype=torch.float64)}
        anchors = {4: torch.tensor([2., 4.])}
        value = mapping(u, g, anchors, .1)
        self.close(value["raw_max"], 2.75); self.close(value["normalized_max"], 2.75 / 3)
        self.close(value["denominator_max"], 3.); self.close(value["tau"], .1)

    def test_only_current_and_one_trial_payload_are_retained(self):
        class Payload(dict):pass
        class MemoryQuadratic:
            def __init__(self):
                self.live=[]; self.peak=0
                self.center=torch.tensor([[.5], [.4]],dtype=torch.float64)
                self.diagonal=torch.tensor([[1.], [4.]],dtype=torch.float64)
            def evaluate(self,u,gradient=False,lambda_Q=None,candidate_id=None):
                payload=Payload(u={4:u[4].clone()},candidate_id=candidate_id)
                self.live.append(weakref.ref(payload))
                self.peak=max(self.peak,sum(ref() is not None for ref in self.live))
                delta=payload['u'][4]-self.center
                result=dict(smooth_sum=float((.5*self.diagonal*delta.square()).sum()),payload=payload)
                if gradient:result.update(self.backward(payload,lambda_Q=lambda_Q))
                return result
            def backward(self,payload,lambda_Q=None):
                return dict(gradient={4:self.diagonal*(payload['u'][4]-self.center)})
        engine=MemoryQuadratic()
        result=solve(engine,{4:torch.zeros(2,1,dtype=torch.float64)},
                     {4:torch.tensor([10.])},.1)
        self.assertGreaterEqual(result.receipt['accepted_updates'],3)
        self.assertLessEqual(engine.peak,2)
        self.assertEqual(sum(ref() is not None for ref in engine.live),1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
