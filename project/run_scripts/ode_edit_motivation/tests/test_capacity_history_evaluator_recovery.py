from __future__ import annotations

import unittest
from unittest.mock import patch

from project.run_scripts.ode_edit_motivation.capacity_history_evaluator_recovery import (
    _ORIGINAL_POLICY_PARAMETERS,
    recovered_policy_parameters,
)
from project.run_scripts.ode_edit_motivation.mv0_fidelity import _safe_payload


class CapacityHistoryEvaluatorRecoveryTests(unittest.TestCase):
    def test_recovery_only_renames_reserved_metadata_key(self) -> None:
        current = _ORIGINAL_POLICY_PARAMETERS()
        legacy = {
            "controller": current["controller"],
            "evaluation": current["metric_protocol"],
        }
        with (
            patch(
                "project.run_scripts.ode_edit_motivation."
                "capacity_history_evaluator_recovery._ORIGINAL_POLICY_PARAMETERS",
                return_value=legacy,
            ),
            patch(
                "project.run_scripts.ode_edit_motivation."
                "capacity_history_evaluator_recovery._file_sha256",
                return_value="0" * 64,
            ),
        ):
            recovered = recovered_policy_parameters()
        self.assertNotIn("evaluation", recovered)
        self.assertEqual(recovered["metric_protocol"], legacy["evaluation"])
        self.assertEqual(recovered["controller"], legacy["controller"])
        self.assertTrue(recovered["technical_recovery"]["metadata_only"])
        self.assertFalse(recovered["technical_recovery"]["sanitizer_relaxation"])
        self.assertFalse(recovered["technical_recovery"]["controller_action_rerun"])
        _safe_payload(recovered)


if __name__ == "__main__":
    unittest.main()
