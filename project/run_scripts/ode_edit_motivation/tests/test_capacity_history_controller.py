from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from project.run_scripts.ode_edit_motivation.capacity_history_analysis import (
    BRANCH_ALPHA_NATIVE,
    BRANCH_ALPHA_QP,
    BRANCH_MEMIT_NATIVE,
    BRANCH_MEMIT_QP,
)
from project.run_scripts.ode_edit_motivation.contracts import ContractError
from project.run_scripts.ode_edit_motivation.capacity_history_controller import (
    MIN_TRUST_RATIO,
    accept_round,
    capacity_probe_action_ids,
    is_alpha_branch,
    is_qp_branch,
    policy_parameters,
    sanitized_traceback_frames,
)


def _keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from _keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _keys(item)


class CapacityHistoryControllerTests(unittest.TestCase):
    def test_capacity_probe_contract_is_five_layers_without_uniform(self) -> None:
        actions = tuple(
            SimpleNamespace(action_id=f"layer_{layer}")
            for layer in range(4, 9)
        )
        self.assertEqual(
            capacity_probe_action_ids(actions),
            ("layer_4", "layer_5", "layer_6", "layer_7", "layer_8"),
        )
        with self.assertRaisesRegex(ContractError, "unit-action order"):
            capacity_probe_action_ids(
                (*actions, SimpleNamespace(action_id="uniform"))
            )

    def test_sanitized_traceback_contains_code_location_not_message(self) -> None:
        try:
            raise RuntimeError("raw prompt must not persist")
        except RuntimeError as exc:
            frames = sanitized_traceback_frames(exc)
        self.assertTrue(frames)
        self.assertEqual(
            set(frames[-1]),
            {"source", "function", "line"},
        )
        self.assertNotIn("raw prompt", repr(frames))

    def test_branch_family_and_qp_axes_are_exact(self) -> None:
        self.assertFalse(is_alpha_branch(BRANCH_MEMIT_NATIVE))
        self.assertFalse(is_alpha_branch(BRANCH_MEMIT_QP))
        self.assertTrue(is_alpha_branch(BRANCH_ALPHA_NATIVE))
        self.assertTrue(is_alpha_branch(BRANCH_ALPHA_QP))
        self.assertFalse(is_qp_branch(BRANCH_MEMIT_NATIVE))
        self.assertTrue(is_qp_branch(BRANCH_MEMIT_QP))
        self.assertFalse(is_qp_branch(BRANCH_ALPHA_NATIVE))
        self.assertTrue(is_qp_branch(BRANCH_ALPHA_QP))

    def test_acceptance_is_positive_gain_and_fixed_trust_ratio(self) -> None:
        accepted, ratio = accept_round(rewrite_gain=0.1, predicted_gain=1.0)
        self.assertTrue(accepted)
        self.assertAlmostEqual(ratio, 0.1)
        accepted, ratio = accept_round(
            rewrite_gain=MIN_TRUST_RATIO - 1e-6,
            predicted_gain=1.0,
        )
        self.assertFalse(accepted)
        self.assertLess(ratio, MIN_TRUST_RATIO)
        accepted, ratio = accept_round(rewrite_gain=-0.1, predicted_gain=1.0)
        self.assertFalse(accepted)
        self.assertLess(ratio, 0.0)
        accepted, ratio = accept_round(rewrite_gain=1.0, predicted_gain=0.0)
        self.assertFalse(accepted)
        self.assertTrue(math.isinf(ratio))

    def test_policy_is_model_common_and_contains_no_nfe_field(self) -> None:
        policy = policy_parameters()
        self.assertEqual(policy["seed"], 41)
        self.assertEqual(policy["max_accepted_rounds"], 3)
        self.assertEqual(policy["initial_trust_fraction"], 0.25)
        self.assertEqual(policy["retry_shrink"], 0.5)
        self.assertNotIn("nfe", set(_keys(policy)))


if __name__ == "__main__":
    unittest.main()
