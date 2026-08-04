from __future__ import annotations

import hashlib
import unittest

from project.run_scripts.ode_bf.barriers import (
    FunctionalHPVerdict,
    functional_replay_risk,
)
from project.run_scripts.ode_bf.contracts import (
    Arm,
    ODEBFContractError,
    assert_matched_projector_arms,
)


def _ids(count: int, prefix: str) -> tuple[str, ...]:
    return tuple(
        hashlib.sha256(f"{prefix}-{index}".encode("utf-8")).hexdigest()
        for index in range(count)
    )


class FunctionalBarrierTests(unittest.TestCase):
    def test_h_small_history_disables_cvar_and_raw_max_is_authoritative(self) -> None:
        receipt = functional_replay_risk(
            barrier="historical-current-teacher",
            entry_values=[1.0, 1.0, 1.0],
            trial_values=[1.0, 1.0005, 1.002],
            sample_sha256=_ids(3, "h"),
            budget=1.0e-3,
        )
        self.assertIsNone(receipt.cvar_diagnostic)
        self.assertFalse(receipt.cvar_decision_enabled)
        self.assertFalse(receipt.passed)
        self.assertGreater(receipt.raw_max_positive_damage, receipt.budget)

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
