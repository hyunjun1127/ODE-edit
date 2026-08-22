from __future__ import annotations

import inspect
import unittest

from project.run_scripts import session05_ode_bf_p1r52_w0_preedit as preedit


class W0PreEditContractTest(unittest.TestCase):
    def test_one_model_no_writer_plan(self) -> None:
        plan = preedit.build_plan()
        self.assertEqual(plan["model_load_count"], 1)
        self.assertEqual(plan["request_count"], 1000)
        self.assertEqual(plan["pre_edit_evaluator_count"], 10)
        self.assertEqual(plan["physical_W_transition_count"], 0)
        self.assertEqual(plan["writer_call_count"], 0)

    def test_source_has_no_writer_or_target_action(self) -> None:
        source = inspect.getsource(preedit.main)
        self.assertNotIn("run_phase1", source)
        self.assertNotIn("run_official_native_apply", source)
        self.assertNotIn("_fp32_target_and_j0", source)
        self.assertIn("_evaluate_batch_entry", source)
        self.assertIn("W0 pre-edit evaluator mutated model weights", source)


if __name__ == "__main__":
    unittest.main()
