from __future__ import annotations

import hashlib
import unittest

import numpy as np

from project.run_scripts.ode_bf.barriers import (
    FunctionalHPVerdict,
    functional_replay_risk,
)
from project.run_scripts.ode_bf.contracts import (
    Arm,
    ODEBFContractError,
    assert_matched_projector_arms,
)
from project.run_scripts.ode_bf.routing import (
    QuadraticBarrier,
    RoutingProblem,
    verify_backtracked_candidate,
)


def _ids(count: int, prefix: str) -> tuple[str, ...]:
    return tuple(
        hashlib.sha256(f"{prefix}-{index}".encode("utf-8")).hexdigest()
        for index in range(count)
    )


class FunctionalBarrierTests(unittest.TestCase):
    def test_h_small_history_uses_mean_and_smooth_max_with_raw_max_diagnostic(self) -> None:
        receipt = functional_replay_risk(
            barrier="historical-current-teacher",
            entry_values=[1.0, 1.0, 1.0],
            trial_values=[1.0, 1.0005, 1.002],
            sample_sha256=_ids(3, "h"),
            budget=1.0e-3,
        )
        self.assertIsNone(receipt.cvar_diagnostic)
        self.assertFalse(receipt.cvar_decision_enabled)
        self.assertEqual(receipt.decision_rule, "mean-and-smooth-max")
        self.assertTrue(receipt.passed)
        self.assertGreater(receipt.raw_max_positive_damage, receipt.budget)

        rejected = functional_replay_risk(
            barrier="historical-current-teacher",
            entry_values=[0.0, 0.0, 0.0],
            trial_values=[0.0, 0.0, 0.1],
            sample_sha256=_ids(3, "h-outlier"),
            budget=1.0e-3,
        )
        self.assertFalse(rejected.passed)
        self.assertGreater(rejected.smooth_max_positive_damage, rejected.budget)

    def test_p_is_samplewise_positive_part_plus_signed_mean(self) -> None:
        receipt = functional_replay_risk(
            barrier="pretrained-theta0-teacher",
            entry_values=[0.2] * 10,
            trial_values=[0.19, 0.21, 0.22, 0.18, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            sample_sha256=_ids(10, "p"),
            budget=0.03,
        )
        self.assertAlmostEqual(receipt.mean_positive_damage, 0.003)
        self.assertAlmostEqual(receipt.signed_mean_damage, 0.0)
        self.assertIsNotNone(receipt.cvar_diagnostic)
        self.assertFalse(receipt.cvar_decision_enabled)
        self.assertEqual(receipt.decision_rule, "uniform-mean-positive-part")

    def test_p_uniform_mean_is_authoritative_and_outlier_max_is_diagnostic(self) -> None:
        passed = functional_replay_risk(
            barrier="pretrained-theta0-teacher",
            entry_values=[0.0] * 10,
            trial_values=[0.009] + [0.0] * 9,
            sample_sha256=_ids(10, "p-mean-pass"),
            budget=1.0e-3,
        )
        self.assertTrue(passed.passed)
        self.assertLessEqual(passed.mean_positive_damage, passed.budget)
        self.assertGreater(passed.raw_max_positive_damage, passed.budget)
        self.assertGreater(passed.smooth_max_positive_damage, passed.budget)

        failed = functional_replay_risk(
            barrier="pretrained-theta0-teacher",
            entry_values=[0.0] * 10,
            trial_values=[0.011] + [0.0] * 9,
            sample_sha256=_ids(10, "p-mean-fail"),
            budget=1.0e-3,
        )
        self.assertFalse(failed.passed)
        self.assertGreater(failed.mean_positive_damage, failed.budget)

    def test_functional_verifiers_cannot_be_replaced_by_structural_proxy(self) -> None:
        history = functional_replay_risk(
            barrier="historical-current-teacher",
            entry_values=[0.0] * 10,
            trial_values=[0.0] * 9 + [0.1],
            sample_sha256=_ids(10, "history"),
            budget=0.01,
        )
        pretrained = functional_replay_risk(
            barrier="pretrained-theta0-teacher",
            entry_values=[0.0] * 10,
            trial_values=[0.0] * 10,
            sample_sha256=_ids(10, "pretrained"),
            budget=0.01,
        )
        verdict = FunctionalHPVerdict(history, pretrained, True, True, True)
        self.assertFalse(verdict.accepted)

    def test_locked_rca_scalar_fixture_separates_progress_from_mean_p(self) -> None:
        requested = 0.1989646926522255
        actual = 0.05081033706665039
        zero = np.zeros(2, dtype=np.float64)
        barrier = QuadraticBarrier(
            "historical",
            0.0,
            zero,
            np.zeros((2, 2), dtype=np.float64),
            1.0,
            "layer-local-diagonal",
        )
        pretrained_barrier = QuadraticBarrier(
            "pretrained",
            0.0,
            zero,
            np.zeros((2, 2), dtype=np.float64),
            1.0,
            "layer-local-diagonal",
        )
        problem = RoutingProblem(
            np.array([1.0, 0.5], dtype=np.float64),
            np.eye(2, dtype=np.float64),
            np.eye(2, dtype=np.float64),
            1.0,
            np.ones(2, dtype=np.float64),
            requested,
            1.0e-8,
            barrier,
            pretrained_barrier,
        )
        progress = verify_backtracked_candidate(
            problem,
            np.array([requested, 0.0], dtype=np.float64),
            beta=1.0,
            actual_signed_progress=actual,
            functional_h_pass=True,
            functional_p_pass=True,
            authoritative_bf16_pass=True,
        )
        self.assertLess(actual, requested)  # Old unscaled-p rule rejected this.
        self.assertTrue(progress.progress_pass)
        self.assertGreater(progress.trust_ratio or 0.0, 0.1)

        p_mean = 0.0004859965153241738
        p_raw = 0.002074637102356033
        remainder = (10.0 * p_mean - p_raw) / 9.0
        replay = functional_replay_risk(
            barrier="pretrained-theta0-teacher",
            entry_values=[0.0] * 10,
            trial_values=[p_raw] + [remainder] * 9,
            sample_sha256=_ids(10, "pure-rca-p"),
            budget=1.0e-3,
        )
        self.assertTrue(replay.passed)
        self.assertAlmostEqual(replay.mean_positive_damage, p_mean)
        self.assertGreater(replay.raw_max_positive_damage, replay.budget)


class AttributionContractTests(unittest.TestCase):
    def test_later_arm_order_is_exact(self) -> None:
        self.assertEqual(
            tuple(Arm),
            (
                Arm.NATIVE_ALPHA,
                Arm.NATIVE_ALPHA_WB,
                Arm.FIXED_EQUAL_REPLAY,
                Arm.FIXED_GENERIC,
                Arm.FIXED_BF,
                Arm.ODE_Z_STATIC,
                Arm.FULL_ODE_BF,
            ),
        )

    def test_generic_bf_pair_diff_is_projector_only(self) -> None:
        common = {
            "raw_velocity_sha256": "a" * 64,
            "sampling_schedule_sha256": "b" * 64,
            "k_resolution": 8,
            "correction_cycles": 1,
            "backtracking": [1.0, 0.5, 0.25],
            "edit_batch_size": 10,
            "trial_budget": 24,
        }
        assert_matched_projector_arms(
            {**common, "barrier_projector": "identity"},
            {**common, "barrier_projector": "cbf"},
        )
        with self.assertRaisesRegex(ODEBFContractError, "beyond"):
            assert_matched_projector_arms(
                {**common, "barrier_projector": "identity"},
                {**common, "barrier_projector": "cbf", "trial_budget": 23},
            )


if __name__ == "__main__":
    unittest.main()
