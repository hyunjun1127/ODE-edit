from __future__ import annotations

import unittest

from project.run_scripts.ode_edit_motivation.capacity_history_evaluator_recovery import (
    _ORIGINAL_POLICY_PARAMETERS,
    recovered_policy_parameters,
)
from project.run_scripts.ode_edit_motivation.mv0_fidelity import _safe_payload


class CapacityHistoryEvaluatorRecoveryTests(unittest.TestCase):
    def test_recovery_only_renames_reserved_metadata_key(self) -> None:
        original = _ORIGINAL_POLICY_PARAMETERS()
        recovered = recovered_policy_parameters()
        self.assertNotIn("evaluation", recovered)
        self.assertEqual(recovered["metric_protocol"], original["evaluation"])
        self.assertEqual(recovered["controller"], original["controller"])
        self.assertTrue(recovered["technical_recovery"]["metadata_only"])
        self.assertFalse(recovered["technical_recovery"]["sanitizer_relaxation"])
        self.assertFalse(recovered["technical_recovery"]["controller_action_rerun"])
        _safe_payload(recovered)


if __name__ == "__main__":
    unittest.main()
