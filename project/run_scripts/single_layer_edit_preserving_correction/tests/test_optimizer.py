"""Small CPU controller fixtures; no Llama/numerical T qualification."""
import math
import unittest

import torch

from project.run_scripts.single_layer_edit_preserving_correction.optimizer import (
    optimize, Check, Observation, ControllerFailure, EvidenceWriteFailure,
    TrialNumericalOverflow, tensor_sha256,
)


class IdentitySpace:
    status = "RESOLVED"
    dimension = 1

    def project(self, value):
        return value.clone()


def quadratic(target=0., bias=0.):
    def observe(weight, *, gradient):
        residual = weight.double() - target
        return float(.5 * residual.square().sum()) + bias, residual if gradient else None, []
    return observe


class OptimizerTests(unittest.TestCase):
    def kwargs(self):
        return dict(space=IdentitySpace(), guard=lambda w: True,
                    proposal_check=lambda d: True, invariant=lambda w, d, a: True)

    def test_n4_no_callbacks_and_owned_endpoint(self):
        native = torch.tensor([[2.]], dtype=torch.float32)
        result = optimize("N4", native)
        self.assertEqual(result.stop_reason, "NATIVE_ENDPOINT")
        self.assertEqual(result.counters["gradient_sweeps"], 0)
        self.assertEqual(result.counters["objective_trial_sweeps"], 0)
        self.assertNotEqual(result.weight.data_ptr(), native.data_ptr())
        torch.testing.assert_close(result.actual_delta, torch.zeros((1, 1), dtype=torch.float64))

    def test_one_round_first_accepted_polyak(self):
        native = torch.tensor([[2.]])
        result = optimize("EN-F", native, objective=quadratic(), **self.kwargs())
        torch.testing.assert_close(result.weight, torch.tensor([[1.]]))
        self.assertEqual(result.counters["gradient_sweeps"], 1)
        self.assertEqual(result.counters["objective_trial_sweeps"], 1)
        self.assertEqual(result.counters["accepted_rounds"], 1)
        self.assertEqual(result.trials[0]["p_actual"], -2.)
        self.assertEqual(result.stop_reason, "GRADIENT_BUDGET_EXHAUSTED")

    def test_rejected_guards_keep_native_all_costs_and_exact_eight_limit(self):
        seen = []
        kwargs = self.kwargs()
        kwargs["guard"] = lambda w: Check(False, "PER_SEQUENCE_NLL")
        result = optimize("CA", torch.tensor([[2.]]), objective=quadratic(),
                          event=lambda record, tensors: seen.append(record["event"]), **kwargs)
        torch.testing.assert_close(result.weight, torch.tensor([[2.]]))
        self.assertEqual(result.counters["objective_trial_sweeps"], 8)
        self.assertEqual(result.counters["guard_calls"], 8)
        self.assertEqual(result.counters["rejected_trials"], 8)
        self.assertEqual(seen.count("trial_rejected"), 8)
        self.assertEqual(result.stop_reason, "TRIAL_BUDGET_EXHAUSTED_NO_NEW_ACCEPT")

    def test_enf4_four_gradients_and_ideal_accumulation_not_rounded_delta(self):
        # Bias keeps Polyak proposals non-dyadic; rounded D differs from ideal.
        native = torch.tensor([[100.12345]], dtype=torch.float32)
        observed_weights, saved = [], []
        fn = quadratic(target=100., bias=.000123)
        def objective(weight, *, gradient):
            if gradient:
                observed_weights.append(weight.clone())
            return fn(weight, gradient=gradient)
        def event(record, tensors):
            if record["event"] == "trial_accepted":
                saved.append((tensors["proposed_ideal_delta"].clone(),
                              tensors["candidate_weight"].clone(),
                              tensors["gradient"].clone(), record["eta"]))
        result = optimize("EN-F4", native, objective=objective, event=event, **self.kwargs())
        self.assertGreaterEqual(result.counters["accepted_rounds"], 2)
        self.assertLessEqual(result.counters["gradient_sweeps"], 4)
        self.assertLessEqual(result.counters["objective_trial_sweeps"], 24)
        self.assertTrue(any(not torch.equal(i, w.double()-native.double()) for i, w, _, _ in saved))
        for i in range(1, len(saved)):
            ideal, weight, gradient, eta = saved[i]
            torch.testing.assert_close(ideal, saved[i-1][0]-eta*gradient, rtol=0, atol=0)
            torch.testing.assert_close(observed_weights[i], saved[i-1][1], rtol=0, atol=0)
        torch.testing.assert_close(result.weight, (native.double()+result.ideal_delta).float(), rtol=0, atol=0)

    def test_enf4_first_round_budget_six_then_stop(self):
        kwargs = self.kwargs()
        kwargs["guard"] = lambda w: False
        result = optimize("EN-F4", torch.tensor([[2.]]), objective=quadratic(), **kwargs)
        self.assertEqual(result.counters["attempted_trial_slots"], 6)
        self.assertEqual(result.counters["gradient_sweeps"], 1)

    def test_actual_fp32_prediction_and_no_move(self):
        native = torch.tensor([[1e8]], dtype=torch.float32)
        def objective(weight, *, gradient):
            return Observation(1., torch.ones_like(weight).double() if gradient else None)
        result = optimize("EN-F", native, objective=objective, **self.kwargs())
        self.assertEqual(result.stop_reason, "NO_RESOLVED_STEP")
        self.assertEqual(result.trials[0]["p_actual"], 0.)
        self.assertEqual(result.trials[0]["nominal_prediction"], -1.)
        self.assertEqual(result.counters["objective_trial_sweeps"], 0)

    def test_scale_clipping_uses_actual_step_not_unclipped_prediction(self):
        result = optimize("SCALE", torch.tensor([[2.]]), entry_weight=torch.tensor([[1.]]),
                          objective=quadratic(bias=100.), guard=lambda w: True)
        self.assertEqual(result.scalar_a, 1.)
        torch.testing.assert_close(result.weight, torch.tensor([[1.]]))
        self.assertEqual(result.trials[0]["p_actual"], -2.)
        self.assertEqual(result.trials[0]["nominal_prediction"], -2.)
        self.assertEqual(result.trials[0]["eta0"], 25.5)

    def test_scale_no_feasible_descent_and_zero_gradient(self):
        result = optimize("SCALE", torch.tensor([[2.]]), entry_weight=torch.tensor([[1.]]),
                          objective=quadratic(target=4.), guard=lambda w: True)
        self.assertEqual(result.stop_reason, "NO_FEASIBLE_DESCENT")
        zero = optimize("SCALE", torch.tensor([[2.]]), entry_weight=torch.tensor([[2.]]),
                        objective=quadratic(), guard=lambda w: True)
        self.assertEqual(zero.stop_reason, "ZERO_GRADIENT")

    def test_actual_weight_dedup_reuses_objective_and_guard_without_more_forward(self):
        result = optimize("SCALE", torch.tensor([[2.]]), entry_weight=torch.tensor([[1.]]),
                          objective=quadratic(bias=1000.), guard=lambda w: False)
        self.assertEqual(result.counters["attempted_trial_slots"], 8)
        self.assertEqual(result.counters["objective_trial_sweeps"], 1)
        self.assertEqual(result.counters["guard_calls"], 1)
        self.assertEqual(result.counters["duplicate_trials"], 7)
        self.assertEqual(result.counters["rejected_trials"], 8)

    def test_floor_negative_loss_and_covariance_separate_units(self):
        fn = lambda w, gradient: (5e-7, torch.ones_like(w) if gradient else None)
        result = optimize("KL-P", torch.ones((1, 1)), objective=fn, **self.kwargs())
        self.assertEqual(result.stop_reason, "NUMERICAL_FLOOR")
        bad = lambda w, gradient: (-2e-6, torch.ones_like(w) if gradient else None)
        with self.assertRaises(ControllerFailure):
            optimize("KL-P", torch.ones((1, 1)), objective=bad, **self.kwargs())
        with self.assertRaises(ControllerFailure):
            optimize("EN-COV", torch.ones((1, 1)), objective=fn, **self.kwargs())
        cov = optimize("EN-COV", torch.ones((1, 1)), objective=fn,
                       cov_resolution=1e-6, **self.kwargs())
        self.assertEqual(cov.stop_reason, "NUMERICAL_FLOOR")

    def test_rank_empty_do_not_force_gradient(self):
        for status, expected in [("RANK_UNRESOLVED", "RANK_UNRESOLVED"),
                                 ("REPAIR_SPACE_EMPTY", "REPAIR_SPACE_EMPTY")]:
            space = IdentitySpace()
            space.status = status
            fn = lambda *a, **k: self.fail("gradient must not run")
            kwargs = self.kwargs()
            kwargs["space"] = space
            result = optimize("EN-F", torch.ones((1, 1)), objective=fn, **kwargs)
            self.assertEqual(result.stop_reason, expected)
            self.assertEqual(result.counters["gradient_sweeps"], 0)

    def test_nonfinite_gradient_saved_then_technical_failure(self):
        events = []
        def event(record, payload):
            events.append((record, payload.get("observed_gradient")))
        fn = lambda w, gradient: (1., torch.full_like(w, math.nan))
        with self.assertRaises(ControllerFailure):
            optimize("CA", torch.ones((1, 1)), objective=fn, event=event, **self.kwargs())
        self.assertEqual(events[-1][0]["event"], "technical_failure")
        self.assertTrue(torch.isnan(events[-1][1]).all())
        self.assertFalse(any(e[0]["event"] == "selected_endpoint" for e in events))

    def test_math_null_failure_technical_actual_invariant_failure_rejected(self):
        kwargs = self.kwargs()
        kwargs["proposal_check"] = lambda d: False
        with self.assertRaises(ControllerFailure):
            optimize("EN-F", torch.tensor([[2.]]), objective=quadratic(), **kwargs)
        kwargs = self.kwargs()
        kwargs["invariant"] = lambda *a: Check(False, "LOGIT_MAX")
        result = optimize("EN-F", torch.tensor([[2.]]), objective=quadratic(), **kwargs)
        self.assertEqual(result.counters["accepted_rounds"], 0)
        self.assertTrue(all(t["reason"] == "ACTUAL_INVARIANT_FAILED" for t in result.trials))

    def test_candidate_nonfinite_forward_rejected_oom_technical(self):
        base = quadratic()
        count = 0
        def fn(w, *, gradient):
            nonlocal count
            if not gradient:
                count += 1
                if count == 1:
                    raise TrialNumericalOverflow("finite weight forward overflow")
            return base(w, gradient=gradient)
        result = optimize("CA", torch.tensor([[2.]]), objective=fn, **self.kwargs())
        self.assertEqual(result.counters["nonfinite_objective_trials"], 1)
        self.assertEqual(result.counters["objective_trial_sweeps"], 2)
        def oom(w, *, gradient):
            if not gradient:
                raise RuntimeError("CUDA out of memory")
            return base(w, gradient=gradient)
        with self.assertRaises(ControllerFailure):
            optimize("CA", torch.tensor([[2.]]), objective=oom, **self.kwargs())

    def test_nonfinite_candidate_weight_has_no_forward(self):
        def fn(w, *, gradient):
            self.assertTrue(gradient)
            return 1e308, torch.ones_like(w).double()
        result = optimize("CA", torch.zeros((1, 1)), objective=fn, **self.kwargs())
        self.assertEqual(result.counters["nonfinite_weight_trials"], 8)
        self.assertEqual(result.counters["objective_trial_sweeps"], 0)

    def test_polyak_overflow_is_unresolved_without_inf_halving(self):
        fn = lambda w, gradient: (1e308, torch.full_like(w, 1e-100, dtype=torch.float64))
        result = optimize("CA", torch.ones((1, 1)), objective=fn, **self.kwargs())
        self.assertEqual(result.stop_reason, "STEP_SCALE_UNRESOLVED")
        self.assertEqual(result.counters["attempted_trial_slots"], 0)

    def test_evidence_sink_failure_is_not_success(self):
        def event(record, tensors):
            raise OSError("full disk")
        with self.assertRaises(EvidenceWriteFailure):
            optimize("N4", torch.ones((1, 1)), event=event)

    def test_evidence_and_invariant_mutation_fail_closed(self):
        def event(record, tensors):
            if "current_weight" in tensors:
                tensors["current_weight"].add_(1.)
        with self.assertRaises(EvidenceWriteFailure):
            optimize("N4", torch.ones((1, 1)), event=event)
        def mutation(weight, ideal, actual):
            ideal.add_(1.)
            return True
        kwargs = self.kwargs()
        kwargs["invariant"] = mutation
        with self.assertRaises(ControllerFailure):
            optimize("EN-F", torch.tensor([[2.]]), objective=quadratic(), **kwargs)

    def test_shared_gradient_exact_binding(self):
        native = torch.tensor([[2.]])
        obs = Observation(2., native.double(), [], weight_sha256=tensor_sha256(native),
                          objective_id="teacher/S64/locked")
        result = optimize("CA", native, objective=quadratic(), objective_id=obs.objective_id,
                          initial_observation=obs, **self.kwargs())
        self.assertEqual(result.counters["gradient_sweeps"], 0)
        self.assertEqual(result.counters["shared_initial_gradient_reuses"], 1)
        with self.assertRaises(ControllerFailure):
            optimize("CA", native+1, objective=quadratic(), objective_id=obs.objective_id,
                     initial_observation=obs, **self.kwargs())

    def test_callback_mutation_and_invalid_boolean_fail(self):
        def mutate(w, *, gradient):
            w.add_(1.)
            return quadratic()(w, gradient=gradient)
        with self.assertRaises(ControllerFailure):
            optimize("CA", torch.tensor([[2.]]), objective=mutate, **self.kwargs())
        kwargs = self.kwargs()
        kwargs["guard"] = lambda w: {"passed": "false"}
        with self.assertRaises(ControllerFailure):
            optimize("CA", torch.tensor([[2.]]), objective=quadratic(), **kwargs)

    def test_no_user_tolerance_or_budget_tuning_argument(self):
        with self.assertRaises(TypeError):
            optimize("CA", torch.tensor([[2.]]), objective=quadratic(), epsilon=.15, **self.kwargs())


if __name__ == "__main__":
    unittest.main()
