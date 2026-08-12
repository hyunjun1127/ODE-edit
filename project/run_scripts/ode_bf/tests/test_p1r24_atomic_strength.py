from __future__ import annotations

import math
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch

from project.run_scripts.ode_bf.fixed_e8_soft_routing import FixedE8Arm
from project.run_scripts.ode_bf.p1r24_atomic_strength import (
    P1R24AliasTargetLock,
    P1R24RoutingStatus,
    p1r24_cumulative_p_receipt,
    p1r24_disable_historical,
    p1r24_target_step,
    solve_p1r24_matched_routing,
)
from project.run_scripts.ode_bf.routing import QuadraticBarrier, RoutingProblem
from project.run_scripts.ode_bf.scalable_batched_field import ScalableRobustSharedMetric
from project.run_scripts.ode_bf.scalable_batched_runtime import scalable_ordered_request_digest


class _Objective:
    def __init__(self, gradient: torch.Tensor, values: tuple[float, ...]) -> None:
        self.target_gradient = gradient
        self.per_request_values = values
        self.loss = sum(values) / len(values)


class _KL:
    def __init__(self, gradient: torch.Tensor, values: tuple[float, ...]) -> None:
        self.gradient = gradient
        self.per_request_values = values
        self.loss = sum(values) / len(values)


def _barrier(label: str, *, offset: float, linear: np.ndarray, gram: np.ndarray) -> QuadraticBarrier:
    return QuadraticBarrier(label, offset, linear, gram, offset + 1.0e6, "layer-local-diagonal")


def _problem() -> RoutingProblem:
    slopes = np.asarray((2.0, 1.5, 1.0, 0.5, -0.25))
    diagonal = np.asarray((1.0, 1.2, 1.4, 1.6, 1.8))
    return RoutingProblem(
        slopes,
        np.diag(diagonal + 0.5),
        np.diag(diagonal),
        100.0,
        np.ones(5),
        1.0e-12,
        1.0e-12,
        _barrier("historical", offset=5.0, linear=np.ones(5), gram=np.diag(np.ones(5))),
        _barrier("pretrained", offset=3.0, linear=np.asarray((-0.8, 0.4, 0.3, 0.2, 0.1)), gram=np.diag(np.asarray((2.0, 1.0, 3.0, 4.0, 5.0)))),
    )


class P1R24ContractTests(unittest.TestCase):
    def test_remaining_step_8_and_1_post_clamp_and_h_once(self) -> None:
        z0 = torch.ones(4, 1)
        metric = ScalableRobustSharedMetric.from_z0(z0, "a" * 64)
        gradient = -torch.ones(4, 1)
        nll = _Objective(gradient, (1.0,))
        kl = _KL(torch.zeros_like(gradient), (0.0,))
        lock = P1R24AliasTargetLock.for_alias("llama3-8b-inst")
        step0 = p1r24_target_step(z0, z0, z0, nll, kl, metric, lock, step_index=0, frozen_mask=(False,))
        self.assertEqual(step0.receipt["remaining_steps"], 8)
        self.assertEqual(step0.receipt["physical_h_application_count"], 1)
        self.assertEqual(step0.receipt["second_remaining_division_count"], 0)
        self.assertLessEqual(step0.receipt["identity_max_abs_residual"], 1.0e-8)
        lagged_terminal = step0.target_next - 0.75
        step7 = p1r24_target_step(step0.target_next, lagged_terminal, z0, nll, kl, metric, lock, step_index=7, frozen_mask=(False,))
        self.assertEqual(step7.receipt["remaining_steps"], 1)
        expected = step7.target_displacement + (step0.target_next - lagged_terminal)
        torch.testing.assert_close(step7.required_displacement, expected)

    def test_unit_gain_remaining_rule_closes_terminal_lag(self) -> None:
        z = torch.zeros(3)
        y = torch.zeros(3)
        for k in range(8):
            d = torch.ones(3) * 0.125
            required = d + (z - y) / (8 - k)
            z = z + d
            y = y + required
        torch.testing.assert_close(z, y)

    def test_matched_strength_and_neutral_mapping(self) -> None:
        problem = p1r24_disable_historical(_problem())
        rho = 0.75
        with mock.patch(
            "project.run_scripts.ode_bf.p1r24_atomic_strength.minimize",
            side_effect=AssertionError("Neutral must not invoke the Soft optimizer"),
        ):
            result = solve_p1r24_matched_routing(
                problem, arm=FixedE8Arm.NEUTRAL, rho_write=rho
            )
        self.assertEqual(result.status, P1R24RoutingStatus.JOINT_WRITE)
        self.assertAlmostEqual(result.predicted_progress, rho, places=10)
        active = [index for index, value in enumerate(result.signed_slopes) if value > 0]
        expected = rho / sum(result.signed_slopes[index] for index in active)
        for index in active:
            self.assertAlmostEqual(result.velocity[index], expected, places=10)

    def test_soft_matches_strength_and_uses_cumulative_cross(self) -> None:
        problem = p1r24_disable_historical(_problem())
        neutral = solve_p1r24_matched_routing(problem, arm=FixedE8Arm.NEUTRAL, rho_write=0.75)
        soft = solve_p1r24_matched_routing(problem, arm=FixedE8Arm.SOFT, rho_write=0.75)
        self.assertAlmostEqual(neutral.predicted_progress, soft.predicted_progress, places=8)
        self.assertLessEqual(soft.selected_p, neutral.selected_p + 1.0e-7)
        receipt = p1r24_cumulative_p_receipt(problem, soft.velocity, step_index=1, factor_list_hashes={layer: str(layer) * 64 for layer in (4, 5, 6, 7, 8)})
        self.assertEqual(receipt["atomic_contribution_count"], 1)
        self.assertNotEqual(receipt["d_P"], 0.0)
        self.assertNotEqual(receipt["linear_cross_term"], 0.0)
        self.assertLessEqual(receipt["algebra_identity_residual"], 1.0e-8)

    def test_low_rank_quadratic_equals_dense_toy(self) -> None:
        rng = np.random.default_rng(5)
        delta = rng.normal(size=(4, 3))
        candidate = rng.normal(size=(4, 3))
        covariance = rng.normal(size=(3, 3))
        covariance = covariance @ covariance.T
        v = 0.7
        direct = np.trace((delta + v * candidate) @ covariance @ (delta + v * candidate).T)
        offset = np.trace(delta @ covariance @ delta.T)
        cross = 2.0 * v * np.trace(delta @ covariance @ candidate.T)
        self_term = v * v * np.trace(candidate @ covariance @ candidate.T)
        self.assertAlmostEqual(direct, offset + cross + self_term, places=10)

    def test_atomic_h_is_zero_while_p_can_accumulate(self) -> None:
        disabled = p1r24_disable_historical(_problem())
        value = np.ones(5)
        self.assertEqual(disabled.historical.value(value), 0.0)
        self.assertGreater(disabled.pretrained.value(value), 0.0)

    def test_semantic_zero_write_is_total(self) -> None:
        result = solve_p1r24_matched_routing(p1r24_disable_historical(_problem()), arm=FixedE8Arm.SOFT, rho_write=0.0)
        self.assertEqual(result.status, P1R24RoutingStatus.SEMANTIC_NO_WRITE)
        self.assertTrue(all(value == 0.0 for value in result.velocity))

    def test_positive_sub_epsilon_demand_is_not_semantic_zero(self) -> None:
        result = solve_p1r24_matched_routing(
            p1r24_disable_historical(_problem()),
            arm=FixedE8Arm.NEUTRAL,
            rho_write=5.0e-13,
        )
        self.assertEqual(result.status, P1R24RoutingStatus.JOINT_WRITE)
        self.assertAlmostEqual(result.predicted_progress, 5.0e-13, places=24)

    def test_no_budget_veto_retry_or_history_source(self) -> None:
        root = Path(__file__).parents[1]
        source = (root / "p1r24_atomic_strength.py").read_text(encoding="utf-8")
        runtime = (root / "p1_scalable_batched_experiment.py").read_text(encoding="utf-8")
        self.assertIn('"hard_p_h_budget_influence_count": 0', source)
        self.assertIn('"retry_backtracking_veto_count": 0', source)
        self.assertIn('"functional_p_layer_basis_count": 0', runtime)
        self.assertNotIn("alpha_apply=q", source)

    def test_atomic_role_does_not_require_held_sequential_artifact(self) -> None:
        root = Path(__file__).parents[1]
        runtime = (root / "p1_runtime.py").read_text(encoding="utf-8")
        guard = runtime[runtime.index("require_held_ode_alloc=") : runtime.index(
            "# P1R23/P1R24 own distinct atomic seals"
        )]
        self.assertIn("and scalable_batched_role is None", guard)
        self.assertIn("and atomic_strength_recovery_role is None", guard)

    def test_sealed_b1_smoke_digest_is_distinct_and_total(self) -> None:
        value = "a" * 64
        observed = scalable_ordered_request_digest((value,))
        self.assertEqual(len(observed), 64)
        self.assertNotEqual(observed, value)


if __name__ == "__main__":
    unittest.main()
