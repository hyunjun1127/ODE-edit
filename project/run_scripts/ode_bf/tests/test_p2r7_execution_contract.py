from __future__ import annotations

import inspect
from pathlib import Path
import unittest

from project.run_scripts import session05_ode_bf_p2r7_atomic_dry_plan as dry
from project.run_scripts.ode_bf.artifacts import sha256_file
from project.run_scripts.ode_bf.p2r7_atomic_panel import LOCK_FILE, load_and_validate_lock
from project.run_scripts.ode_bf import p1_runtime, p2r2_atomic_runtime, p2r7_atomic_runtime


ROOT = Path(__file__).resolve().parents[4]


class P2R7ExecutionContractTest(unittest.TestCase):
    def test_lock_and_both_dry_plans(self) -> None:
        lock, _ = load_and_validate_lock(
            ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        pilot = dry.build_plan("HEAD", phase="pilot", repository_root=ROOT)
        production = dry.build_plan("HEAD", phase="b10x10", repository_root=ROOT)
        self.assertEqual(lock["routing_variable_count"], 5)
        self.assertEqual(pilot["attempt_count"], 8)
        self.assertEqual(production["attempt_count"], 40)
        self.assertTrue(
            all(
                job["request_layer_response_matrix_count"] == 0
                for job in pilot["jobs"] + production["jobs"]
            )
        )

    def test_target_operator_source_is_exact_p2r2_parent_blob(self) -> None:
        self.assertEqual(
            sha256_file(
                ROOT / "project/run_scripts/ode_bf/p2r1_rms_tangent_target.py"
            ),
            "98591cc1472c23d14178002dd535596916c93751d2d10b95f1306dc7705dac21",
        )
        source = inspect.getsource(p2r2_atomic_runtime._run_arm_case)
        self.assertIn("P2R1RMSState.zero()", source)
        self.assertIn("P2R1_MICROSTEPS_PER_OUTER_STATE", source)
        self.assertIn("current_target = update.target_next", source)
        self.assertNotIn("state = P2R1RMSState.zero()\n            writer", source)

    def test_p2r7_path_is_selected_by_p1_runtime(self) -> None:
        source = inspect.getsource(p1_runtime.run_p1)
        self.assertIn("p2r7_phase", source)
        self.assertIn("run_p2r7_atomic", source)
        self.assertIn("phase=p2r7_phase", source)

    def test_writer_coordinate_and_compute_firewall_are_bound(self) -> None:
        source = inspect.getsource(p2r2_atomic_runtime._run_arm_case)
        self.assertIn("(current_target - current_terminal) / 0.125", source)
        self.assertIn("progress_simplex_waypoint_factors", source)
        self.assertIn('"request_layer_response_matrix_count": 0 if p2r7 else 1', source)
        self.assertIn('"physical_h_application_count": 1 if p2r7 else 0', source)
        wrapper = inspect.getsource(p2r7_atomic_runtime)
        self.assertNotIn("measure_request_layer_response", wrapper)
        self.assertNotIn("solve_p2r2_routing", wrapper)

    def test_launcher_resources_and_hold_release(self) -> None:
        sbatch = (
            ROOT / "project/run_scripts/session05_ode_bf_p2r7_atomic.sbatch"
        ).read_text()
        submitter = (
            ROOT / "project/run_scripts/session05_ode_bf_submit_p2r7_atomic.py"
        ).read_text()
        for token in (
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --mem=65000M",
            "#SBATCH --array=0-1%2",
        ):
            self.assertIn(token, sbatch)
        self.assertIn('"--hold"', submitter)
        self.assertIn('"scontrol", "release"', submitter)
        self.assertIn("PROJECT_GPU_CAP = 4", submitter)


if __name__ == "__main__":
    unittest.main()
