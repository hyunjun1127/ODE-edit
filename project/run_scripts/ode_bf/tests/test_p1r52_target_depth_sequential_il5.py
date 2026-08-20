from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path

from project.run_scripts import (
    session05_ode_bf_p1r52_target_depth_il5_sequential_b100x10_dry_plan as dry,
)
from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1_scalable_batched_experiment import _run_ode_arm
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    expected_p1r52_sequential_result_name,
    run_p1r52_sequential,
)
from project.run_scripts.ode_bf.p1r52_target_depth import (
    P1R52_SEQUENTIAL_TARGET_DEPTH_INNER_COUNTS,
    P1R52_TARGET_DEPTH_INNER_COUNTS,
    P1R52TargetDepth,
)
from project.run_scripts.ode_bf.p1r52_target_depth_sequential_il5 import (
    ATTEMPT_SUFFIX,
    POLICY,
    RESULT_NAME,
    ROLE,
    STREAM_ORDER,
    STREAM_ROOT,
    validate_runtime_activation,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


class P1R52TargetDepthSequentialIL5Tests(unittest.TestCase):
    def test_il5_is_sequential_only_and_other_depth_registries_are_unchanged(self) -> None:
        self.assertEqual(P1R52_TARGET_DEPTH_INNER_COUNTS, (1, 3, 8, 10, 15))
        self.assertEqual(P1R52_SEQUENTIAL_TARGET_DEPTH_INNER_COUNTS, (5,))
        self.assertEqual(P1R52TargetDepth.from_inner_count(5), P1R52TargetDepth.IL5_FULL)
        self.assertEqual(POLICY.expected_inner_rows_per_batch, 40)
        self.assertEqual(POLICY.expected_request_inner_rows_per_batch, 4000)

    def test_activation_is_exact_and_fail_closed(self) -> None:
        self.assertIs(
            validate_runtime_activation(
                role=ROLE,
                scale_id="b100x10",
                attempt_suffix=ATTEMPT_SUFFIX,
                depth="IL5-FULL",
                inner_telemetry=True,
                batch_entry_evaluator=False,
            ),
            POLICY,
        )
        for changed in (
            {"role": "r52-soft-sequential-alphacache-on-structuralh-off"},
            {"depth": "IL3-FULL"},
            {"inner_telemetry": False},
            {"batch_entry_evaluator": True},
        ):
            values = {
                "role": ROLE,
                "scale_id": "b100x10",
                "attempt_suffix": ATTEMPT_SUFFIX,
                "depth": "IL5-FULL",
                "inner_telemetry": True,
                "batch_entry_evaluator": False,
                **changed,
            }
            with self.assertRaises(ODEBFContractError):
                validate_runtime_activation(**values)

    def test_result_and_dry_plan_bind_one_b1000_cell(self) -> None:
        self.assertEqual(
            expected_p1r52_sequential_result_name(
                "llama3-8b-inst",
                ROLE,
                scale=__import__(
                    "project.run_scripts.ode_bf.p1r52_sequential_scale",
                    fromlist=["P1R52_B100X10_SCALE"],
                ).P1R52_B100X10_SCALE,
                attempt_suffix=ATTEMPT_SUFFIX,
            ),
            RESULT_NAME,
        )
        plan = dry.build_plan("1" * 40)
        self.assertEqual(plan["job_count"], 1)
        self.assertEqual(plan["target_depth"], "IL5-FULL")
        self.assertEqual(plan["inner_count_per_outer"], 5)
        self.assertEqual(plan["inner_rows_per_batch"], 40)
        self.assertEqual(plan["request_inner_rows_per_batch"], 4000)
        self.assertEqual(plan["batch_entry_evaluator_count"], 0)

    def test_runtime_uses_shared_scheduler_and_explicit_sequential_gate(self) -> None:
        scalable = inspect.getsource(_run_ode_arm)
        sequential = inspect.getsource(run_p1r52_sequential)
        scheduler = (
            REPO_ROOT / "project/run_scripts/ode_bf/p1r52_target_depth.py"
        ).read_text(encoding="utf-8")
        self.assertIn("run_p1r52_target_depth_scheduler(", scalable)
        self.assertIn("p1r52_sequential_target_depth", scalable)
        self.assertIn("validate_il5_sequential_activation", sequential)
        self.assertIn("p1r52_sequential_target_depth=", sequential)
        self.assertNotIn("writer_materialize(", scheduler)
        self.assertNotIn("evaluate_heldout", scheduler)

    def test_source_ast_keeps_one_outer_writer_and_no_inner_writer(self) -> None:
        source = inspect.getsource(run_p1r52_sequential)
        tree = ast.parse(source)
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        self.assertIn("target_depth_telemetry_observer", names)
        self.assertIn("target_depth_policy", names)
        self.assertIn("commit_sequential_batch", source)
        self.assertIn('"target_depth_controller_heldout_access_count": 0', source)
        self.assertIn('"target_depth_observation_action_influence_count": 0', source)
        self.assertIn('"batch_entry_pre_evaluator_count"', source)

    def test_stream_and_task_contract_are_exact(self) -> None:
        self.assertEqual(
            STREAM_ROOT,
            "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a",
        )
        self.assertEqual(
            STREAM_ORDER,
            "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3",
        )
        self.assertEqual(POLICY.round_count * POLICY.batch_size, 1000)


if __name__ == "__main__":
    unittest.main()
