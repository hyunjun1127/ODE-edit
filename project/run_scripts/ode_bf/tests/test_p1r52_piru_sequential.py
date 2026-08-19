from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from project.run_scripts.ode_bf.p1r52_pir_writer import P1R52PIRPolicy
from project.run_scripts.ode_bf.p1r52_piru_sequential_adapter import (
    P1R52_PIRU_SEQUENTIAL_ATTEMPT_SUFFIX,
    P1R52_PIRU_SEQUENTIAL_METHOD_ID,
    P1R52_PIRU_SEQUENTIAL_RESULT_NAME,
    P1R52_PIRU_SEQUENTIAL_ROLE,
    is_piru_structural_h_role,
    pir_policy_for_role,
)
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    R52_H_ROLE,
    R52_ROLES,
    R52_STRUCTURAL_H_ROLES,
    expected_p1r52_sequential_result_name,
    run_p1r52_sequential,
)
from project.run_scripts.ode_bf.p1r52_sequential_scale import P1R52_B100X10_SCALE
from project.run_scripts.ode_bf.p1r52_piru_sequential_panel import (
    LOCK_FILE,
    load_and_validate_lock,
    verify_reference_results,
)
from project.run_scripts import session05_ode_bf_p1r52_piru_sequential_b100x10_dry_plan as dry


class P1R52PIRUSequentialTests(unittest.TestCase):
    def test_piru_role_is_a_structural_h_r52_role(self) -> None:
        self.assertIn(P1R52_PIRU_SEQUENTIAL_ROLE, R52_ROLES)
        self.assertIn(P1R52_PIRU_SEQUENTIAL_ROLE, R52_STRUCTURAL_H_ROLES)
        self.assertIn(R52_H_ROLE, R52_STRUCTURAL_H_ROLES)
        self.assertTrue(is_piru_structural_h_role(P1R52_PIRU_SEQUENTIAL_ROLE))
        self.assertIs(
            pir_policy_for_role(P1R52_PIRU_SEQUENTIAL_ROLE),
            P1R52PIRPolicy.PIR_U,
        )
        self.assertIsNone(pir_policy_for_role(R52_H_ROLE))

    def test_piru_result_identity_is_create_once_and_b100_only(self) -> None:
        self.assertEqual(
            expected_p1r52_sequential_result_name(
                "llama3-8b-inst",
                P1R52_PIRU_SEQUENTIAL_ROLE,
                scale=P1R52_B100X10_SCALE,
                attempt_suffix=P1R52_PIRU_SEQUENTIAL_ATTEMPT_SUFFIX,
            ),
            P1R52_PIRU_SEQUENTIAL_RESULT_NAME,
        )

    def test_piru_dry_plan_has_one_cell_and_no_entry_evaluator(self) -> None:
        plan = dry.build_plan("a" * 40)
        self.assertEqual(plan["job_count"], 1)
        self.assertEqual(plan["request_count"], 1000)
        self.assertEqual(plan["writer"], "PIR-U")
        self.assertEqual(plan["structural_h"], "ON")
        self.assertEqual(plan["batch_entry_evaluator_count"], 0)
        self.assertEqual(plan["retry_backtracking_line_search"], [0, 0, 0])

    def test_piru_runtime_uses_narrow_policy_switch(self) -> None:
        source = inspect.getsource(run_p1r52_sequential)
        self.assertIn("pir_policy = pir_policy_for_role(role)", source)
        self.assertIn("p1r52_pir_policy=pir_policy", source)
        self.assertIn("structural_h_decision_enabled=structural_h_enabled", source)
        self.assertIn("P1R52_PIRU_SEQUENTIAL_METHOD_ID", source)

    def test_numerical_lock_and_sealed_references(self) -> None:
        repo = Path(__file__).resolve().parents[4]
        lock, lock_sha = load_and_validate_lock(
            repo / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        self.assertEqual(len(lock_sha), 64)
        self.assertEqual(lock["root_digest"], "95d2e909d0e59bbf8ce65f181778dfb23b38ef12735e84649674b5e2cfccea67")
        references = verify_reference_results()
        self.assertEqual(len(references["references"]), 3)
        self.assertTrue(all(item["W0_restored"] for item in references["references"].values()))


if __name__ == "__main__":
    unittest.main()
