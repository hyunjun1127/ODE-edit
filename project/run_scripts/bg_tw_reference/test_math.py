"""Small CPU fixtures only; no model-level parity or scientific efficacy claim.

Run: python -m unittest project.run_scripts.bg_tw_reference.test_math -v
"""
import math
import unittest

import torch

from .native_map import FrozenNativeMap, NativeMapError
from .correction import (
    CANDIDATE_ORDER, CorrectionError, NumericalPolicy,
    accumulate_route_gradient, choose_candidate, correct_once,
    full_control_slope, materialize_candidate, relaxed_barrier,
    weighted_chunk_mean,
)


torch.set_num_threads(2)


def fixture_map(repeat=1):
    generator = torch.Generator().manual_seed(20260915)
    keys = torch.randn(5, 3*repeat, generator=generator)*.2
    # Deliberately nonsymmetric; accidentally transposing/Cholesky differs.
    projector = torch.randn(5, 5, generator=generator)*.04
    history_factor = torch.randn(5, 5, generator=generator)*.1
    history = history_factor @ history_factor.T
    result = FrozenNativeMap.from_frozen_kpm(
        keys, projector, history, request_count=3, weight_shape=(4, 5),
        source_identity="cpu-fixture-not-model", l2=1.,
        request_major_repeat_verified=(repeat != 1))
    return result, generator


def numerics():
    return NumericalPolicy(epsilon=1e-12, ball_atol=2e-6,
                           trust_atol=2e-6, screen_tolerance=1e-7,
                           alpha_cap=None)


class NativeMapTests(unittest.TestCase):
    def test_native_direct_rhs_map_repeat_and_orientation(self):
        for repeat in (1, 3):
            operator, generator = fixture_map(repeat)
            residual = torch.randn(4, 3, generator=generator)
            expected = torch.linalg.solve(
                operator.system,
                operator.projected_keys @ residual.repeat_interleave(repeat, dim=1).T).T
            self.assertTrue(torch.equal(operator.direct_native_update(residual), expected))
            torch.testing.assert_close(operator(residual), expected, atol=1e-7, rtol=2e-5)
            self.assertFalse(torch.allclose(operator.system, operator.system.T))
            self.assertEqual(operator.A.shape, (3, 5))

    def test_vjp_and_finite_directional_difference_fp32(self):
        operator, generator = fixture_map(3)
        residual = torch.randn(4, 3, generator=generator).requires_grad_()
        cotangent = torch.randn(4, 5, generator=generator)
        direction = torch.randn(4, 3, generator=generator)
        objective = (operator(residual)*cotangent).sum()
        actual, = torch.autograd.grad(objective, residual)
        torch.testing.assert_close(actual, operator.vjp(cotangent), atol=1e-7, rtol=1e-6)
        step = 1e-3
        finite_difference = ((operator(residual+step*direction)*cotangent).sum() -
                             (operator(residual-step*direction)*cotangent).sum())/(2*step)
        expected = (actual*direction).sum()
        self.assertLess(float((finite_difference-expected).abs()), 4e-5)
        self.assertFalse(operator.A.requires_grad)
        self.assertFalse(operator.system.requires_grad)

    def test_all_tokens_receive_real_downprojection_not_subject_hook(self):
        operator, generator = fixture_map()
        residual = torch.randn(4, 3, generator=generator).requires_grad_()
        parent = torch.randn(4, 5, generator=generator)
        tokens = torch.randn(2, 7, 5, generator=generator)
        endpoint = parent+operator(residual)
        difference = torch.nn.functional.linear(tokens, endpoint)-torch.nn.functional.linear(tokens, parent)
        torch.testing.assert_close(difference, tokens @ operator(residual).T, atol=7e-7, rtol=1e-4)
        self.assertEqual(int((difference.abs().sum(-1) > 0).sum()), 14)
        grad, = torch.autograd.grad(difference.square().sum(), residual)
        self.assertGreater(float(grad.norm()), 0.)

    def test_no_unverified_repeat_and_frozen_state_no_alias(self):
        operator, _ = fixture_map(3)
        with self.assertRaisesRegex(NativeMapError, "REPEAT_IDENTITY"):
            FrozenNativeMap.from_native_system(
                operator.system, operator.projected_keys,
                request_count=3, weight_shape=(4, 5), source_identity="fixture")
        source_system = operator.system.clone()
        copied = FrozenNativeMap.from_native_system(
            source_system, operator.projected_keys, request_count=3,
            weight_shape=(4, 5), source_identity="fixture",
            request_major_repeat_verified=True)
        source_system.add_(1.)
        self.assertFalse(torch.equal(source_system, copied.system))
        copied.A.add_(1.)
        with self.assertRaisesRegex(NativeMapError, "OPERATOR_MUTATED"):
            copied(torch.zeros(4, 3))

    def test_actual_native_vs_map_materialization_receipt(self):
        operator, generator = fixture_map()
        residual = torch.randn(4, 3, generator=generator)
        parent = torch.randn(4, 5, generator=generator)
        captured = parent+operator.direct_native_update(residual)
        comparison = operator.comparison(residual, parent, captured_native_endpoint=captured)
        self.assertTrue(comparison["rhs_captured_endpoint_exact"])
        self.assertEqual(comparison["model_forward_parity"], "NOT_TESTED")
        self.assertGreaterEqual(comparison["map_native_endpoint_max_abs"], 0.)

    def test_fp32_only_no_singular_or_invalid_shape_silencing(self):
        operator, _ = fixture_map()
        with self.assertRaisesRegex(NativeMapError, "FP32"):
            operator(torch.zeros(4, 3, dtype=torch.float64))
        with self.assertRaisesRegex(NativeMapError, "NON_SQUARE"):
            FrozenNativeMap.from_native_system(
                operator.system, operator.projected_keys,
                request_count=3, weight_shape=(5, 5), source_identity="fixture")
        with self.assertRaises(torch.linalg.LinAlgError):
            FrozenNativeMap.from_native_system(
                torch.zeros(5, 5), operator.projected_keys,
                request_count=3, weight_shape=(4, 5), source_identity="fixture")


class BarrierMassTests(unittest.TestCase):
    def test_c2_join_and_negative_slack(self):
        values = []
        for point in (.1-1e-9, .1, .1+1e-9, -.2):
            h = torch.tensor(point, dtype=torch.float64, requires_grad=True)
            value = relaxed_barrier(h)
            first, = torch.autograd.grad(value, h, create_graph=True)
            second, = torch.autograd.grad(first, h)
            self.assertTrue(all(math.isfinite(float(v.detach())) for v in (value, first, second)))
            values.append((float(value.detach()), float(first.detach()), float(second.detach())))
        for value, first, second in values[:3]:
            self.assertAlmostEqual(value, -math.log(.1), delta=2e-8)
            self.assertAlmostEqual(first, -10., delta=2e-6)
            self.assertAlmostEqual(second, 100., delta=3e-6)
        self.assertAlmostEqual(values[-1][1], -40.)
        self.assertAlmostEqual(values[-1][2], 100.)

    def test_full_control_slope_matches_autograd(self):
        policy = numerics()
        for control in (-1e-8, 0., .01, .095, .11, .3):
            d = torch.tensor(control, dtype=torch.float64, requires_grad=True)
            objective = policy.mu*relaxed_barrier((.1-d)/.1)
            expected, = torch.autograd.grad(objective, d)
            self.assertAlmostEqual(full_control_slope(control, .1, policy), float(expected), delta=1e-12)

    def test_weighted_microbatch_gradient_equals_full_objective(self):
        policy = numerics()
        generator = torch.Generator().manual_seed(11)
        residual = torch.randn(2, 3, generator=generator, requires_grad=True)
        current_x = torch.randn(100, 2, 3, generator=generator)
        control_x = torch.randn(64, 2, 3, generator=generator)
        current_loss = ((current_x*residual).sum((1, 2))-.3).square()
        control_loss = .01*((control_x*residual).sum((1, 2))).square()
        d64 = control_loss.mean()
        full = current_loss.mean()+policy.mu*relaxed_barrier((.1-d64)/.1)
        expected, = torch.autograd.grad(full, residual, retain_graph=True)
        current, control = [], []
        for losses, chunk_size, dest in ((current_loss, 17, current), (control_loss, 7, control)):
            for chunk in losses.split(chunk_size):
                gradient, = torch.autograd.grad(chunk.mean(), residual, retain_graph=True)
                dest.append((gradient, chunk.numel()))
        actual, evidence = accumulate_route_gradient(
            current, control, d64=float(d64.detach()), ceiling=.1, policy=policy)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        self.assertFalse(evidence["barrier_applied_to_microbatch_means"])
        self.assertEqual(evidence["document_count"], 64)
        per_chunk_barrier = sum(policy.mu*relaxed_barrier((.1-c.mean())/.1)*len(c)/64
                                for c in control_loss.split(7))
        wrong, = torch.autograd.grad(current_loss.mean()+per_chunk_barrier, residual)
        self.assertGreater(float((wrong-actual).abs().max()), 1e-4)

    def test_true_mass_not_mean_of_microbatch_means_and_missing_rejected(self):
        means = [(torch.tensor(1.), 3), (torch.tensor(7.), 1)]
        self.assertEqual(float(weighted_chunk_mean(means, expected_count=4)), 2.5)
        with self.assertRaisesRegex(CorrectionError, "TOTAL_MASS"):
            weighted_chunk_mean(means, expected_count=5)
        gradient = torch.ones(2, 3)
        with self.assertRaisesRegex(CorrectionError, "TOTAL_MASS"):
            accumulate_route_gradient([(gradient, 99)], [(gradient, 64)],
                                      d64=.01, ceiling=.1, policy=numerics())

    def test_zero_self_kl_has_zero_gradient_not_invalid_reference(self):
        generator = torch.Generator().manual_seed(29)
        logits = torch.randn(2, 5, 11, generator=generator, requires_grad=True)
        logp = logits.log_softmax(-1)
        teacher = logp.detach().clone()
        kl = (teacher.exp()*(teacher-logp)).sum(-1).mean(-1).mean()
        gradient, = torch.autograd.grad(kl, logits)
        self.assertEqual(float(kl), 0.)
        self.assertLess(float(gradient.abs().max()), 2e-8)


class CorrectionCandidateTests(unittest.TestCase):
    def setup_inputs(self):
        operator, generator = fixture_map()
        anchor = torch.randn(4, 3, generator=generator)
        radii = .75*anchor.norm(dim=0)
        direction = torch.randn(4, 3, generator=generator)
        proposal = anchor+.3*radii*direction/direction.norm(dim=0)
        canonical = anchor+torch.randn(4, 3, generator=generator)*.2
        gradient = torch.randn(4, 3, generator=generator)
        return operator, anchor, radii, proposal, canonical, gradient

    def test_projection_and_executable_trust_keep_own_anchor(self):
        operator, anchor, radii, proposal, canonical, gradient = self.setup_inputs()
        corrected, receipt = correct_once(proposal, canonical, anchor, radii,
                                          gradient, operator, policy=numerics())
        rprop = proposal-canonical
        self.assertTrue(receipt["corrected_available"])
        self.assertLessEqual(float(operator(corrected-rprop).double().norm()),
                             receipt["trust_bound"]+2e-6)
        self.assertTrue(bool(((proposal+(corrected-rprop)-anchor).norm(dim=0) <= radii+2e-6).all()))
        self.assertFalse(torch.equal(anchor, canonical))
        self.assertTrue(torch.isfinite(corrected).all())

    def test_zero_gradient_or_native_action_means_raw_only(self):
        operator, anchor, radii, proposal, canonical, gradient = self.setup_inputs()
        for h, g in ((canonical, torch.zeros_like(gradient)), (proposal, gradient)):
            corrected, receipt = correct_once(proposal, h, anchor, radii,
                                              g, operator, policy=numerics())
            self.assertFalse(receipt["corrected_available"])
            self.assertTrue(torch.equal(corrected, proposal-h))

    def test_native_ball_violation_or_nonfinite_is_not_silent_clamping(self):
        operator, anchor, radii, proposal, canonical, gradient = self.setup_inputs()
        with self.assertRaisesRegex(CorrectionError, "NATIVE_PROPOSAL_OUTSIDE"):
            correct_once(anchor+100., canonical, anchor, radii, gradient,
                         operator, policy=numerics())
        gradient[0, 0] = float("nan")
        with self.assertRaisesRegex(CorrectionError, "GRADIENT_NONFINITE"):
            correct_once(proposal, canonical, anchor, radii, gradient,
                         operator, policy=numerics())

    def test_explicit_alpha_cap_not_unreported_tuning(self):
        operator, anchor, radii, proposal, canonical, gradient = self.setup_inputs()
        policy = NumericalPolicy(epsilon=1e-12, ball_atol=2e-6, trust_atol=2e-6,
                                 screen_tolerance=0., alpha_cap=1e-7)
        _, receipt = correct_once(proposal, canonical, anchor, radii, gradient,
                                  operator, policy=policy)
        self.assertTrue(receipt["alpha_cap_hit"])
        self.assertEqual(receipt["alpha"], 1e-7)
        self.assertEqual(receipt["numerical_policy"]["alpha_cap"], 1e-7)

    def test_actual_fp32_materialization_raw_copy_and_fractional_candidate(self):
        parent = torch.tensor([[1., 1e6, -1., -.125], [2., -3., 6., 3.]])
        update = torch.tensor([[1e-7, .03, .003, .234], [.5, -.4, .7, .123]])
        raw = torch.nextafter(parent, torch.full_like(parent, math.inf))
        endpoint, receipt = materialize_candidate(parent, candidate="RAW1", raw_native_endpoint=raw)
        self.assertTrue(torch.equal(endpoint, raw))
        self.assertTrue(receipt["endpoint_copy"])
        for name, eta in (("CORR1", 1.), ("CORR.5", .5), ("CORR.25", .25)):
            endpoint, _ = materialize_candidate(parent, candidate=name,
                                                raw_native_endpoint=raw, corrected_map_update=update)
            full = parent+update
            expected = full if eta == 1. else parent+eta*(full-parent)
            self.assertTrue(torch.equal(endpoint, expected))

    def test_screen_feasible_rank_exact_ties_and_parent(self):
        rows = [dict(candidate=name, D64=.04, Ecur=2., actual_delta_norm=3.,
                     actual_endpoint_sha=f"fixture-{name}") for name in CANDIDATE_ORDER]
        rows[0]["D64"] = .2
        rows[1]["actual_delta_norm"] = 2.
        rows[2]["actual_delta_norm"] = 2.
        result = choose_candidate(rows, ceiling=.1, policy=numerics(), corrected_available=True)
        self.assertEqual(result["selected"], "CORR1")
        rows[2]["Ecur"] = 1.
        self.assertEqual(choose_candidate(rows, ceiling=.1, policy=numerics(), corrected_available=True)["selected"], "CORR.5")
        for row in rows:
            row["D64"] = .2
        result = choose_candidate(rows, ceiling=.1, policy=numerics(), corrected_available=True)
        self.assertEqual(result["selected"], "PARENT")
        self.assertFalse(result["write_accepted"])
        self.assertFalse(result["native_fallback"])

    def test_no_extra_candidates_and_nonfinite_is_technical(self):
        raw = dict(candidate="RAW1", D64=.1, Ecur=2., actual_delta_norm=0.,
                   actual_endpoint_sha="fixture")
        self.assertEqual(choose_candidate([raw], ceiling=.1, policy=numerics(), corrected_available=False)["selected"], "RAW1")
        with self.assertRaisesRegex(CorrectionError, "MENU"):
            choose_candidate([raw, raw], ceiling=.1, policy=numerics(), corrected_available=False)
        raw["Ecur"] = float("nan")
        with self.assertRaisesRegex(CorrectionError, "NONFINITE"):
            choose_candidate([raw], ceiling=.1, policy=numerics(), corrected_available=False)


if __name__ == "__main__":
    unittest.main()
