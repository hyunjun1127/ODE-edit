"""Small CPU-only equation, sign/mask, and observer noninterference oracles."""
import unittest

import torch

from project.run_scripts.baseline_mechanism_first.geometry import (
    covariance_diagnostics, native_write_diagnostics, projected_geometry,
)
from project.run_scripts.baseline_mechanism_first.instrumentation import (
    NativeTargetObserver, ReadOnlyCapture, WriterObservation, exposure_diagnostics,
)
from project.run_scripts.baseline_mechanism_first.signed_response import (
    AllPositionContraction, activation_gradient_contraction, desired_margin,
    finite_difference_audit, teacher_forced_nll,
)


class NativeGeometryTests(unittest.TestCase):
    def test_native_nonsymmetric_dense_factor_and_input_identity(self):
        generator = torch.Generator().manual_seed(7)
        K = torch.randn(5, 3, generator=generator)
        R = torch.randn(4, 3, generator=generator)
        P = torch.eye(5)
        P[0, 1] = .02
        M = torch.eye(5)
        M[1, 0] = .1
        A = P @ (K @ K.T + M) + torch.eye(5)
        delta = torch.linalg.solve(A, P @ K @ R.T).T
        inputs = [v.clone() for v in (K, R, P, M, delta)]
        result = native_write_diagnostics(K, R, P, M, delta)
        self.assertTrue(result["recomputed_dense_vs_actual_bitwise"])
        self.assertLess(result["factor_vs_actual_relative"], 1e-5)
        self.assertFalse(result["native_history_symmetrized"])
        self.assertEqual(result["delta_K_minus_R_norm"], float((delta @ K - R).norm()))
        for before, after in zip(inputs, (K, R, P, M, delta)):
            self.assertTrue(torch.equal(before, after))
        geometry = projected_geometry(P, torch.eye(5), M, K, basis=torch.eye(5), compute_spectrum=True)
        self.assertGreater(geometry["reduced_history_symmetrization_norm"], 0)
        self.assertFalse(geometry["writer_inputs_modified"])

    def test_zero_and_raw_covariance_cross(self):
        S = torch.tensor([[1., 2.], [0., -1.]], dtype=torch.float64)
        delta = -S
        C = torch.tensor([[2., .1], [.3, 1.]], dtype=torch.float64)
        result = covariance_diagnostics(S, delta, C)
        self.assertEqual(result["q_C_S_plus_delta"], 0)
        self.assertLess(result["signed_delta_C_S"], 0)
        self.assertAlmostEqual(result["quadratic_identity_residual"], 0)
        zero = covariance_diagnostics(S, delta, torch.zeros_like(C))
        self.assertEqual(zero["status"], "FINITE_ZERO")
        self.assertIsNone(zero["covariance_cosine"])
        geometry = projected_geometry(torch.eye(2, dtype=C.dtype), C, C, torch.zeros(2, 1, dtype=C.dtype), exact_max_dimension=0)
        self.assertIsNone(geometry["P_spectrum"]["rank"])


class SignedResponseTests(unittest.TestCase):
    def test_raw_margin_and_true_new_all_token_signs(self):
        for category, expected in (("R", 2.), ("P", 2.), ("N", -2.)):
            result = desired_margin(3., 1., category, raw_margin=-9.)
            self.assertEqual(result["desired_margin"], expected)
            self.assertEqual(result["raw_margin"], -9.)
        self.assertFalse(desired_margin(1., 1., "N")["success"])
        generator = torch.Generator().manual_seed(21)
        logits = torch.randn(2, 5, 7, dtype=torch.float64, generator=generator, requires_grad=True)
        ids = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
        mask = torch.tensor([[False, False, True, True, True], [False, False, False, True, True]])
        nll = teacher_forced_nll(logits, ids, mask)
        scalar = desired_margin(nll[0], nll[1], "N")["desired_margin"]
        grad = torch.autograd.grad(scalar, logits)[0]
        self.assertEqual(float(grad[:, 0].abs().sum()), 0)
        self.assertGreater(float(grad[0, 1:4].abs().sum()), 0)
        self.assertLess(float(grad[1, 2, ids[1, 3]]), 0)
        self.assertGreater(float(grad[0, 1, ids[0, 2]]), 0)

    def test_hook_dense_gradient_oracle_full_positions_and_cleanup(self):
        torch.manual_seed(8)
        module = torch.nn.Linear(3, 4, bias=False).double()
        x = torch.randn(2, 5, 3, dtype=torch.float64)
        delta = torch.randn_like(module.weight)
        weight_before = module.weight.detach().clone()
        rng_before = torch.get_rng_state().clone()
        base = module(x)
        scalar = base[0].sin().sum() - base[1].square().mean()
        dense_grad = torch.autograd.grad(scalar, module.weight)[0]
        expected = float((dense_grad * delta).sum())
        mask = torch.ones(2, 5, dtype=torch.bool)
        mask[:, -1] = False
        with AllPositionContraction(module, delta, position_mask=mask) as observer:
            out = module(x)
            result = observer.compute(out[0].sin().sum() - out[1].square().mean())
        self.assertAlmostEqual(result["event_derivative"], expected, places=11)
        self.assertAlmostEqual(result["included_position_derivative"] + result["excluded_position_derivative"], expected, places=11)
        self.assertEqual(result["raw_position_count"], 10)
        self.assertEqual(len(module._forward_hooks), 0)
        self.assertTrue(torch.equal(weight_before, module.weight))
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertIsNone(module.weight.grad)
        self.assertTrue(torch.equal(base, out))

    def test_frozen_graph_and_exception_cleanup(self):
        module = torch.nn.Linear(2, 2, bias=False).double()
        module.requires_grad_(False)
        x = torch.ones(1, 3, 2, dtype=torch.float64)
        delta = torch.zeros_like(module.weight)
        with AllPositionContraction(module, delta) as observer:
            result = observer.compute(module(x).sum())
        self.assertEqual(result["status"], "FINITE_ZERO")
        self.assertFalse(module.weight.requires_grad)
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with AllPositionContraction(module, delta):
                raise RuntimeError("synthetic")
        self.assertEqual(len(module._forward_hooks), 0)

    def test_fd_signed_and_unresolved(self):
        result = finite_difference_audit(-3, .03, 0, -.03, alpha=.01,
                                        forward_noise=1e-10, absolute_tolerance=1e-12)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["event_derivative"], -3)
        zero = finite_difference_audit(0, 2, 2, 2, alpha=.01,
                                      forward_noise=0, absolute_tolerance=0)
        self.assertEqual(zero["status"], "NUMERICALLY_UNRESOLVED")

    def test_factor_derivative_is_separate_from_actual_direction(self):
        x, g = torch.ones(1, 2, 3), torch.ones(1, 2, 2)
        actual = torch.zeros(2, 3)
        result = activation_gradient_contraction(x, g, actual,
                                                  diagnostic_factor=(torch.ones(2, 1), torch.ones(3, 1)))
        self.assertEqual(result["event_derivative"], 0)
        self.assertEqual(result["diagnostic_factor_derivative"], 12)
        self.assertEqual(result["factor_minus_actual_derivative"], 12)


class CaptureTests(unittest.TestCase):
    def test_capture_does_not_change_values_rng_or_inputs(self):
        module = torch.nn.Linear(3, 2)
        x = torch.ones(2, 4, 3)
        expected = module(x)
        state = torch.get_rng_state().clone()
        with ReadOnlyCapture(module) as capture:
            actual = module(x)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(len(module._forward_hooks), 0)
        capture.records[0]["input"].zero_()
        self.assertEqual(float(x.sum()), 24)

    def test_writer_endpoint_and_exposure_exclusions(self):
        K = torch.eye(2)
        R = torch.eye(2)
        delta = R.clone()
        observation = WriterObservation.from_native(K, R, delta, column_metadata=[{}, {}],
                                                     weight_before=torch.zeros(2, 2), weight_after=delta)
        self.assertTrue(observation.materialization_receipt()["endpoint_equals_materialized_bitwise"])
        result = exposure_diagnostics(K, K, delta, torch.ones(3, 2),
                                      position_mask=torch.tensor([True, True, False]),
                                      position_tags=["subject", "prefix", "continuation"])
        self.assertEqual(result["raw"]["position_count"], 3)
        self.assertEqual(result["excluded"]["actual_delta_q_squared_sum"], 2)
        row = NativeTargetObserver().record(case_id=1, source="cached_z")
        self.assertEqual(row["training_nll_status"], "NOT_OBSERVED")


if __name__ == "__main__":
    unittest.main()
