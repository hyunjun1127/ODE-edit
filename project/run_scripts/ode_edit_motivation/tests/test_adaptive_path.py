import unittest

from project.run_scripts.ode_edit_motivation.adaptive_path import (
    choose_refreshed_direction,
    normalized_predicted_advantage,
)
from project.run_scripts.ode_edit_motivation.contracts import ContractError


class AdaptiveDirectionGateTests(unittest.TestCase):
    def test_refresh_must_clear_relative_margin(self):
        selected, advantage = choose_refreshed_direction(102.0, 100.0)
        self.assertFalse(selected)
        self.assertAlmostEqual(advantage, 2.0 / 102.0)
        selected, advantage = choose_refreshed_direction(103.0, 100.0)
        self.assertTrue(selected)
        self.assertAlmostEqual(advantage, 3.0 / 103.0)

    def test_tie_and_small_advantage_prefer_fixed(self):
        self.assertFalse(choose_refreshed_direction(1.0, 1.0)[0])
        self.assertFalse(choose_refreshed_direction(1.01, 1.0)[0])

    def test_negative_scores_are_finite_and_ordered(self):
        selected, advantage = choose_refreshed_direction(-1.0, -2.0)
        self.assertTrue(selected)
        self.assertEqual(advantage, 0.5)

    def test_invalid_values_fail_closed(self):
        with self.assertRaises(ContractError):
            normalized_predicted_advantage(float("nan"), 1.0)
        with self.assertRaises(ContractError):
            choose_refreshed_direction(1.0, 1.0, relative_margin=1.0)


if __name__ == "__main__":
    unittest.main()
