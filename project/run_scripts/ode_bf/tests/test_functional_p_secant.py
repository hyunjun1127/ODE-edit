from __future__ import annotations

import math
import unittest

import numpy as np

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.functional_p_secant import (
    FUNCTIONAL_P_BUDGET,
    H_REF,
    build_functional_p_secant,
    build_functional_replay_secant,
    solve_p_soft_hard,
)
from project.run_scripts.ode_bf.routing import QuadraticBarrier, RoutingProblem


def _problem(
    *,
    progress: tuple[float, ...] = (1.0, 1.0),
    requested: float = 1.0,
    trust_radius: float = 2.0,
    barrier_budget: float = 10.0,
) -> RoutingProblem:
    dimension = len(progress)
    zero = np.zeros(dimension, dtype=np.float64)
    zero_gram = np.zeros((dimension, dimension), dtype=np.float64)
    return RoutingProblem(
        np.asarray(progress, dtype=np.float64),
        np.eye(dimension, dtype=np.float64),
        np.eye(dimension, dtype=np.float64),
        trust_radius,
        np.ones(dimension, dtype=np.float64),
        requested,
        1.0e-8,
        QuadraticBarrier(
            "historical",
            0.0,
            zero,
            zero_gram,
            barrier_budget,
            "layer-local-diagonal",
        ),
        QuadraticBarrier(
            "pretrained",
            0.0,
            zero,
            zero_gram,
            barrier_budget,
            "layer-local-diagonal",
        ),
    )


def _secant(
    signed: tuple[float, ...],
    *,
    baseline: float = 5.0e-4,
):
    return build_functional_p_secant(
        baseline_damage=baseline,
        probe_damage=tuple(baseline + H_REF * value for value in signed),
        sample_order_sha256="a" * 64,
        baseline_identity_sha256="b" * 64,
        factor_state_sha256="c" * 64,
    )


class FunctionalPSecantTests(unittest.TestCase):
    def test_empty_history_is_exactly_the_p_only_numerical_program(self) -> None:
        problem = _problem(requested=1.0)
        p_only = _secant((4.0e-3, 8.0e-3), baseline=5.0e-4)
        generic = build_functional_replay_secant(
            pretrained_baseline_damage=5.0e-4,
            pretrained_probe_damage=(1.0e-3, 1.5e-3),
            pretrained_sample_order_sha256="a" * 64,
            pretrained_baseline_identity_sha256="b" * 64,
            history_item_count=0,
            history_sample_order_sha256=p_only.history_sample_order_sha256,
            history_baseline_identity_sha256=(
                p_only.history_baseline_identity_sha256
            ),
            historical_mean_baseline=0.0,
            historical_mean_probe=(0.0, 0.0),
            historical_smooth_baseline=0.0,
            historical_smooth_probe=(0.0, 0.0),
            factor_state_sha256="c" * 64,
        )
        left = solve_p_soft_hard(problem, (0.5, 0.5), p_only)
        right = solve_p_soft_hard(problem, (0.5, 0.5), generic)
        self.assertFalse(generic.historical_soft_active)
        self.assertEqual(left.soft_velocity, right.soft_velocity)
        self.assertEqual(left.sigma_star, right.sigma_star)
        self.assertEqual(left.stage2_distance, right.stage2_distance)
        self.assertIsNone(right.predicted_full_step_historical_mean)
        self.assertIsNone(right.predicted_full_step_historical_smoothmax)

    def test_nonempty_history_uses_single_worst_normalized_slack(self) -> None:
        problem = _problem(requested=1.0)
        receipt = build_functional_replay_secant(
            pretrained_baseline_damage=0.0,
            pretrained_probe_damage=(0.0, 0.0),
            pretrained_sample_order_sha256="a" * 64,
            pretrained_baseline_identity_sha256="b" * 64,
            history_item_count=4,
            history_sample_order_sha256="d" * 64,
            history_baseline_identity_sha256="e" * 64,
            historical_mean_baseline=5.0e-4,
            historical_mean_probe=(1.0e-3, 1.5e-3),
            historical_smooth_baseline=5.0e-4,
            historical_smooth_probe=(1.0e-3, 1.5e-3),
            factor_state_sha256="c" * 64,
        )
        result = solve_p_soft_hard(problem, (0.5, 0.5), receipt)
        self.assertTrue(receipt.historical_soft_active)
        self.assertLessEqual(result.sigma_star, 1.0e-8)
        self.assertGreater(result.soft_velocity[0], 0.99)
        self.assertLess(result.soft_velocity[1], 1.0e-6)
        self.assertLessEqual(result.predicted_full_step_historical_mean, 1.0e-3 + 2.0e-8)
        self.assertLessEqual(
            result.predicted_full_step_historical_smoothmax,
            1.0e-3 + 2.0e-8,
        )
        self.assertEqual(
            tuple(name for name, _ in result.normalized_slack_components),
            (
                "predicted_controller_p",
                "predicted_historical_mean",
                "predicted_historical_smoothmax",
            ),
        )

    def test_nonempty_history_keeps_signed_secants_and_both_constraints(self) -> None:
        problem = _problem(requested=1.0)
        receipt = build_functional_replay_secant(
            pretrained_baseline_damage=0.0,
            pretrained_probe_damage=(0.0, 0.0),
            pretrained_sample_order_sha256="a" * 64,
            pretrained_baseline_identity_sha256="b" * 64,
            history_item_count=4,
            history_sample_order_sha256="d" * 64,
            history_baseline_identity_sha256="e" * 64,
            historical_mean_baseline=5.0e-4,
            historical_mean_probe=(1.0e-3, 1.5e-3),
            historical_smooth_baseline=5.0e-4,
            historical_smooth_probe=(1.5e-3, 1.0e-3),
            factor_state_sha256="c" * 64,
        )
        self.assertEqual(
            receipt.historical_mean_signed_secant, (4.0e-3, 8.0e-3)
        )
        self.assertEqual(
            receipt.historical_smooth_signed_secant, (8.0e-3, 4.0e-3)
        )
        result = solve_p_soft_hard(problem, (0.5, 0.5), receipt)
        self.assertAlmostEqual(result.sigma_star, 0.25, places=6)
        np.testing.assert_allclose(
            result.soft_velocity, (0.5, 0.5), rtol=0.0, atol=2.0e-6
        )
        self.assertAlmostEqual(
            result.predicted_full_step_historical_mean, 1.25e-3, places=9
        )
        self.assertAlmostEqual(
            result.predicted_full_step_historical_smoothmax,
            1.25e-3,
            places=9,
        )

        signed = build_functional_replay_secant(
            pretrained_baseline_damage=0.0,
            pretrained_probe_damage=(0.0, 0.0),
            pretrained_sample_order_sha256="a" * 64,
            pretrained_baseline_identity_sha256="b" * 64,
            history_item_count=4,
            history_sample_order_sha256="d" * 64,
            history_baseline_identity_sha256="e" * 64,
            historical_mean_baseline=5.0e-4,
            historical_mean_probe=(2.5e-4, 5.0e-4),
            historical_smooth_baseline=5.0e-4,
            historical_smooth_probe=(5.0e-4, 2.5e-4),
            factor_state_sha256="c" * 64,
        )
        self.assertEqual(
            signed.historical_mean_signed_secant, (-2.0e-3, 0.0)
        )
        self.assertEqual(
            signed.historical_smooth_signed_secant, (0.0, -2.0e-3)
        )

    def test_signed_secant_preserves_negative_and_zero_entries(self) -> None:
        receipt = _secant((-2.0e-3, 0.0, 4.0e-3))
        self.assertEqual(receipt.signed_secant, (-2.0e-3, 0.0, 4.0e-3))
        self.assertEqual(receipt.h_ref, 1.0 / 8.0)
        self.assertEqual(receipt.controller_p_budget, 1.0e-3)
        self.assertEqual(
            receipt.probe_damage,
            tuple(
                receipt.baseline_damage + H_REF * value
                for value in receipt.signed_secant
            ),
        )

    def test_zero_slack_keeps_existing_velocity_when_it_is_closest(self) -> None:
        problem = _problem(requested=0.5)
        before = (0.25, 0.25)
        receipt = _secant((1.0e-3, -1.0e-3), baseline=5.0e-4)
        result = solve_p_soft_hard(problem, before, receipt)
        np.testing.assert_allclose(result.soft_velocity, before, rtol=0.0, atol=1.0e-8)
        self.assertLessEqual(result.sigma_star, 1.0e-8)
        self.assertLessEqual(result.stage2_distance, 1.0e-8)
        self.assertTrue(result.stage1_certificate.passed)
        self.assertTrue(result.stage2_certificate.passed)

    def test_minimum_slack_then_closest_solution_is_exact(self) -> None:
        problem = _problem(requested=1.0)
        before = (0.5, 0.5)
        # Every progress-feasible velocity sums to one and therefore predicts
        # 1.5e-3 damage.  The normalized minimum violation is exactly 0.5.
        receipt = _secant((8.0e-3, 8.0e-3), baseline=5.0e-4)
        result = solve_p_soft_hard(problem, before, receipt)
        self.assertAlmostEqual(result.sigma_star, 0.5, places=7)
        np.testing.assert_allclose(result.soft_velocity, before, rtol=0.0, atol=2.0e-7)
        self.assertAlmostEqual(
            result.predicted_full_step_controller_p,
            1.5e-3,
            places=10,
        )
        self.assertAlmostEqual(
            float(problem.signed_progress @ np.asarray(result.soft_velocity)),
            problem.requested_progress,
            places=8,
        )

    def test_soft_solution_changes_allocation_without_weakening_constraints(self) -> None:
        problem = _problem(requested=1.0)
        receipt = _secant((4.0e-3, 8.0e-3), baseline=5.0e-4)
        result = solve_p_soft_hard(problem, (0.5, 0.5), receipt)
        value = np.asarray(result.soft_velocity, dtype=np.float64)
        self.assertGreater(value[0], 0.99)
        self.assertLess(value[1], 1.0e-6)
        self.assertGreaterEqual(
            float(problem.signed_progress @ value),
            problem.requested_progress - 1.0e-8,
        )
        self.assertLessEqual(
            float(value @ problem.trust_metric @ value),
            problem.trust_radius**2 + 1.0e-8,
        )
        self.assertLessEqual(
            result.predicted_full_step_controller_p,
            FUNCTIONAL_P_BUDGET * (1.0 + result.sigma_star) + 2.0e-8,
        )

    def test_nonpositive_routing_direction_stays_excluded(self) -> None:
        problem = _problem(progress=(1.0, -1.0), requested=0.5)
        receipt = _secant((-1.0e-2, -1.0))
        result = solve_p_soft_hard(problem, (0.5, 0.0), receipt)
        self.assertEqual(result.soft_velocity[1], 0.0)
        with self.assertRaisesRegex(ODEBFContractError, "nonpositive"):
            solve_p_soft_hard(problem, (0.5, 1.0e-3), receipt)

    def test_nonfinite_probe_or_input_fails_closed(self) -> None:
        with self.assertRaisesRegex(ODEBFContractError, "non-finite"):
            build_functional_p_secant(
                baseline_damage=0.0,
                probe_damage=(0.0, math.nan),
                sample_order_sha256="a" * 64,
                baseline_identity_sha256="b" * 64,
                factor_state_sha256="c" * 64,
            )
        with self.assertRaisesRegex(ODEBFContractError, "input velocity"):
            solve_p_soft_hard(
                _problem(requested=0.5),
                (math.inf, 0.0),
                _secant((0.0, 0.0)),
            )


if __name__ == "__main__":
    unittest.main()
