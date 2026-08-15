from __future__ import annotations

import ast
import inspect
from pathlib import Path
import unittest

from project.run_scripts import session05_ode_bf_p2r2_atomic_dry_plan as dry
from project.run_scripts.ode_bf.p2r2_atomic_panel import (
    LOCK_FILE,
    load_and_validate_lock,
)
from project.run_scripts.ode_bf.p2r2_atomic_runtime import (
    expected_p2r2_result_name,
)


ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "project/run_scripts/ode_bf"


class P2R2ExecutionContractTest(unittest.TestCase):
    def test_numerical_lock_and_dry_plan(self) -> None:
        lock, _ = load_and_validate_lock(PACKAGE / "locks" / LOCK_FILE)
        plan = dry.build_plan(lock["exact_p2r1_parent"], case_count=10)
        self.assertEqual(plan["job_count"], 2)
        self.assertEqual(plan["arm_count"], 4)
        self.assertEqual(plan["attempt_count"], 40)
        self.assertEqual(plan["request_attempt_count"], 400)
        self.assertEqual(plan["array_max_concurrent_gpu"], 2)
        self.assertFalse(plan["model_load"])
        self.assertFalse(plan["gpu_use"])
        self.assertFalse(plan["slurm_submit"])

    def test_result_names_are_create_once_distinct(self) -> None:
        values = {
            expected_p2r2_result_name(alias, case_count=count)
            for alias in ("llama3-8b-inst", "qwen2.5-7b-inst")
            for count in (1, 10)
        }
        self.assertEqual(len(values), 4)

    def test_runtime_has_outer8_micro3_and_continuous_rms_state(self) -> None:
        source = (PACKAGE / "p2r2_atomic_runtime.py").read_text()
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "P2R1RMSState"
        ]
        # Construction is only P2R1RMSState.zero() at case entry; no reset in the loop.
        self.assertEqual(calls, [])
        self.assertIn("state = P2R1RMSState.zero()", source)
        self.assertIn("for outer in range(8):", source)
        self.assertIn("for inner in range(P2R1_MICROSTEPS_PER_OUTER_STATE):", source)
        self.assertIn("state = update.state_next", source)
        self.assertIn('"rms_reset_count": 0', source)

    def test_execution_path_has_no_legacy_writer_import(self) -> None:
        writer = (PACKAGE / "p2r2_residual_transport_writer.py").read_text()
        runtime = (PACKAGE / "p2r2_atomic_runtime.py").read_text()
        for source in (writer, runtime):
            tree = ast.parse(source)
            imported = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names
            }
            self.assertFalse(any("p1r43" in value for value in imported))
        self.assertNotIn("min(alpha_req", writer)
        self.assertNotIn("alpha_max", writer)

    def test_response_uses_one_batched_vjp_and_no_candidate_forward(self) -> None:
        source = inspect.getsource(
            __import__(
                "project.run_scripts.ode_bf.p2r2_residual_transport_writer",
                fromlist=["measure_request_layer_response"],
            ).measure_request_layer_response
        )
        self.assertIn("is_grads_batched=True", source)
        self.assertNotIn("materialize", source)
        self.assertNotIn("for candidate", source)

    def test_action_freeze_precedes_both_terminal_panels(self) -> None:
        source = (PACKAGE / "p2r2_atomic_runtime.py").read_text()
        freeze = source.index('case_root / "action-freeze.json"')
        w_panel = source.index("endpoint, w_evaluator_wall = _evaluate_frozen_state")
        z_panel = source.index("z_panel, z_evaluator_wall = _evaluate_terminal_z_panel")
        self.assertLess(freeze, w_panel)
        self.assertLess(freeze, z_panel)
        self.assertIn('"heldout_controller_access_count": 0', source)

    def test_launcher_resources_hold_and_cap_are_exact(self) -> None:
        sbatch = (ROOT / "project/run_scripts/session05_ode_bf_p2r2_atomic.sbatch").read_text()
        submit = (ROOT / "project/run_scripts/session05_ode_bf_submit_p2r2_atomic.py").read_text()
        self.assertIn("#SBATCH --cpus-per-task=8", sbatch)
        self.assertIn("#SBATCH --mem=65000M", sbatch)
        self.assertIn("#SBATCH --gres=gpu:1", sbatch)
        self.assertIn("#SBATCH --array=0-1%2", sbatch)
        self.assertIn("PROJECT_GPU_CAP = 4", submit)
        self.assertIn("STAGE_GPU_MAX = 2", submit)
        self.assertIn('"sbatch", "--hold"', submit)
        self.assertIn('"scontrol", "release", job_id', submit)


if __name__ == "__main__":
    unittest.main()
