"""Focused source and coordinate gates for the joint-P/C writer."""

from __future__ import annotations

import ast
import inspect
import unittest

from project.run_scripts.ode_bf.p1r52_joint_pc_execution import (
    JointPCWriterArm,
    plan_c1_writer,
    plan_c2_writer,
)
from project.run_scripts.ode_bf.p1r52_joint_pc_runtime import (
    PILOT_RESULT_NAME,
    PILOT_ROLE,
    PILOT_TECH_R1_RESULT_NAME,
    PILOT_TECH_R1_ROLE,
    PILOT_TECH_R2_RESULT_NAME,
    PILOT_TECH_R2_ROLE,
    PILOT_TECH_R3_RESULT_NAME,
    PILOT_TECH_R3_ROLE,
    PILOT_TECH_R4_RESULT_NAME,
    PILOT_TECH_R4_ROLE,
    PILOT_TECH_R5_RESULT_NAME,
    PILOT_TECH_R5_ROLE,
    STREAM_ORDER,
    STREAM_ROOT,
    expected_result_name,
    production_role,
)
from project.run_scripts.ode_bf.p1r52_pir_writer import remaining_pi_beta
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    expected_p1r52_sequential_result_name,
)
from project.run_scripts.ode_bf.p1r52_sequential_scale import P1R52_B100X10_SCALE


class JointPCExecutionTests(unittest.TestCase):
    def test_c1_reuses_exact_piru_suffix_rule(self) -> None:
        pi = (0.05, 0.1, 0.15, 0.2, 0.5)
        beta = remaining_pi_beta(pi)
        self.assertEqual(beta[-1], 1.0)
        source = inspect.getsource(plan_c1_writer)
        self.assertIn("plan_pir_writer", source)
        self.assertIn("P1R52PIRPolicy.PIR_U", source)
        self.assertIn('"current_slope_inverse_count": 0', source)

    def test_c2_is_fixed_quota_without_suffix_or_retry(self) -> None:
        source = inspect.getsource(plan_c2_writer)
        tree = ast.parse(source)
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("remaining_pi_beta", called)
        self.assertNotIn("plan_pir_writer", called)
        self.assertNotIn("build_velocity_factor", called)
        self.assertIn("float(pi[ordinal]) * entry_residual", source)
        self.assertIn("entry_residual.clone()", source)
        self.assertIn('"numeric_h_multiplication_count": 0', source)
        self.assertIn('"suffix_normalization_count": 0', source)
        self.assertIn('"debt_count": 0', source)
        self.assertIn('"catchup_count": 0', source)
        self.assertIn('"second_pass_count": 0', source)
        self.assertIn('"retry_count": 0', source)
        self.assertIn('"current_slope_inverse_count": 0', source)

    def test_roles_and_frozen_stream_are_exact(self) -> None:
        self.assertEqual(expected_result_name(PILOT_ROLE), PILOT_RESULT_NAME)
        self.assertEqual(expected_result_name(PILOT_TECH_R1_ROLE), PILOT_TECH_R1_RESULT_NAME)
        self.assertEqual(expected_result_name(PILOT_TECH_R2_ROLE), PILOT_TECH_R2_RESULT_NAME)
        self.assertEqual(expected_result_name(PILOT_TECH_R3_ROLE), PILOT_TECH_R3_RESULT_NAME)
        self.assertEqual(expected_result_name(PILOT_TECH_R4_ROLE), PILOT_TECH_R4_RESULT_NAME)
        self.assertEqual(expected_result_name(PILOT_TECH_R5_ROLE), PILOT_TECH_R5_RESULT_NAME)
        self.assertEqual(
            expected_p1r52_sequential_result_name(
                "llama3-8b-inst",
                PILOT_TECH_R4_ROLE,
                scale=P1R52_B100X10_SCALE,
            ),
            PILOT_TECH_R4_RESULT_NAME,
        )
        self.assertEqual(
            expected_p1r52_sequential_result_name(
                "llama3-8b-inst",
                PILOT_TECH_R5_ROLE,
                scale=P1R52_B100X10_SCALE,
            ),
            PILOT_TECH_R5_RESULT_NAME,
        )
        self.assertTrue(expected_result_name(production_role(10)).endswith("case-10-v1"))
        self.assertEqual(
            STREAM_ROOT,
            "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a",
        )
        self.assertEqual(
            STREAM_ORDER,
            "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3",
        )
        self.assertEqual(tuple(item.value for item in JointPCWriterArm), (
            "C0-PIRU-CONTROL",
            "C1-JOINT-PC-REMAINING",
            "C2-JOINT-PC-FIXED-QUOTA",
        ))

    def test_finite_residual_coordinate_adapter_is_explicit(self) -> None:
        from project.run_scripts.ode_bf.p1r52_joint_pc_runtime import (
            _writer_velocity_field,
        )

        source = inspect.getsource(_writer_velocity_field)
        self.assertIn("full_residual / float(P1R23_H)", source)
        self.assertIn("float(P1R23_H) * velocity_residual", source)
        self.assertIn("FULL_CURRENT_RESIDUAL_VELOCITY_DEFINITION", source)
        self.assertIn('"second_h_application_count": 0', source)
        self.assertIn('"model_forward_count": 0', source)
        self.assertIn('"model_backward_count": 0', source)


if __name__ == "__main__":
    unittest.main()
