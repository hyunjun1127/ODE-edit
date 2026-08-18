from __future__ import annotations

import inspect
import math
import unittest

import torch

from project.run_scripts.ode_bf.p1r52_pir_writer import (
    PIRTypedBoundary,
    P1R52PIRPolicy,
    _finish_prefix_observation,
    pir_gamma_policy,
    plan_pir_writer,
    remaining_pi_beta,
)
from project.run_scripts.ode_bf.scalable_batched_runtime import P1R23_H


class P1R52PIRWriterTests(unittest.TestCase):
    def test_equal_pi_beta(self) -> None:
        self.assertEqual(
            remaining_pi_beta((0.2, 0.2, 0.2, 0.2, 0.2)),
            (0.2, 0.25, 1.0 / 3.0, 0.5, 1.0),
        )

    def test_positive_tiny_suffix_is_not_epsilon_truncated(self) -> None:
        beta = remaining_pi_beta((1.0, 0.0, 0.0, 0.0, 1e-20))
        self.assertEqual(beta, (1.0, 0.0, 0.0, 0.0, 1.0))

    def test_zero_alpha_zeroes_both_policies(self) -> None:
        for policy in (P1R52PIRPolicy.PIR_G, P1R52PIRPolicy.PIR_U):
            value = pir_gamma_policy(
                policy=policy,
                alpha_star=0.0,
                entry_applied_slopes=(1.0, 2.0, 3.0, 4.0, 5.0),
                beta=(0.2, 0.25, 1.0 / 3.0, 0.5, 1.0),
            )
            self.assertEqual(value["gamma"], 0.0)
            self.assertEqual(value["coefficients"], (0.0,) * 5)

    def test_pir_g_denominator_and_strength_identity(self) -> None:
        beta = remaining_pi_beta((0.1, 0.2, 0.3, 0.2, 0.2))
        value = pir_gamma_policy(
            policy="PIR-G",
            alpha_star=0.75,
            entry_applied_slopes=(0.3, 0.4, 0.5, 0.6, 0.7),
            beta=beta,
        )
        self.assertGreater(value["denominator"], 0.0)
        self.assertLessEqual(value["strength_identity_residual"], 1e-10)
        self.assertAlmostEqual(value["predicted_progress"], 0.75, places=12)
        self.assertTrue(all(0.0 <= item <= 1.0 for item in beta))

    def test_pir_g_nonpositive_denominator_is_typed(self) -> None:
        with self.assertRaises(PIRTypedBoundary):
            pir_gamma_policy(
                policy="PIR-G",
                alpha_star=1.0,
                entry_applied_slopes=(-1.0, -1.0, -1.0, -1.0, -1.0),
                beta=(0.2, 0.25, 1.0 / 3.0, 0.5, 1.0),
            )

    def test_pir_u_gamma_one_and_left_theta_identity(self) -> None:
        beta = remaining_pi_beta((0.2,) * 5)
        value = pir_gamma_policy(
            policy="PIR-U",
            alpha_star=2.0,
            entry_applied_slopes=(0.3,) * 5,
            beta=beta,
        )
        self.assertEqual(value["gamma"], 1.0)
        residual = torch.tensor([[2.0, -3.0]], dtype=torch.float32)
        for share, coefficient in zip(beta, value["coefficients"], strict=True):
            left = residual / P1R23_H
            theta = P1R23_H * coefficient
            self.assertTrue(torch.equal(theta * left, share * residual))

    def test_prefix_observation_is_finite_and_non_mutating(self) -> None:
        target = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
        source = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        destination = torch.tensor([[2.0, 2.0]], dtype=torch.float32)
        copies = (target.clone(), source.clone(), destination.clone())
        receipt: dict[str, object] = {}
        _finish_prefix_observation(
            receipt,
            target_state=target,
            source_terminal=source,
            destination_terminal=destination,
            coefficient=0.5,
        )
        self.assertGreater(float(receipt["residual_reduction"]), 0.0)
        self.assertTrue(math.isfinite(float(receipt["intended_realized_cosine"])))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip((target, source, destination), copies)))

    def test_plan_source_has_q_only_and_no_current_slope_or_nll_decision(self) -> None:
        source = inspect.getsource(plan_pir_writer)
        self.assertIn("q_only=True", source)
        self.assertIn("entry_router_covariance_rebind_count", source)
        self.assertIn("entry_router_covariance_current_writer_decision_influence_count", source)
        self.assertNotIn("_current_slope(", source)
        self.assertNotIn("sequential_quota_decision", source)
        self.assertNotIn("current_nll=", source)
        self.assertIn('"additional_slope_backward_count": 0', source)
        self.assertIn('"applied_slope": slopes0[ordinal]', source)
        self.assertIn('"sequential_p_receipt": PIR_P_RECEIPT_STATUS', source)
        self.assertIn('"atomic_history_append_count": 0', source)


if __name__ == "__main__":
    unittest.main()
