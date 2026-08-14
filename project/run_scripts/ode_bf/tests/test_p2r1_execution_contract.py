from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from project.run_scripts import session05_ode_bf_p2r1_target_only_dry_plan as dry
from project.run_scripts.ode_bf.p2r1_rms_tangent_target import (
    p2r1_preservation_gradient,
    p2r1_target_update,
)
from project.run_scripts.ode_bf.p2r1_target_only_panel import (
    LOCK_FILE,
    PARENT,
    load_and_validate_lock,
)
from project.run_scripts.ode_bf.p2r1_target_only_runtime import (
    expected_p2r1_target_result_name,
)


ROOT = Path(__file__).resolve().parents[4]


class P2R1ExecutionContractTests(unittest.TestCase):
    def test_lock_parent_stream_and_target_only_counts(self) -> None:
        lock, _ = load_and_validate_lock(
            ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        self.assertEqual(PARENT, "11508b6da11d606521b703037034e1814b70d8a8")
        self.assertEqual(lock["target_microstep_count"], 24)
        self.assertEqual(lock["target_microsteps_per_outer_state"], 3)
        self.assertEqual(lock["writer_materialization_count"], 0)
        self.assertEqual(lock["native_endpoint_runtime_access_count"], 0)

    def test_tensor_policy_owns_no_model_or_evaluator(self) -> None:
        source = "\n".join(
            inspect.getsource(item)
            for item in (p2r1_preservation_gradient, p2r1_target_update)
        )
        for forbidden in (
            "model(",
            "evaluate_",
            "backtracking",
            "Native",
            "beta1",
            "beta2",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("semantic_tangent_max_abs_residual", source)
        self.assertIn("semantic_rate_max_abs_residual", source)

    def test_target_runtime_firewall_and_no_writer_dispatch(self) -> None:
        source = (
            ROOT / "project/run_scripts/ode_bf/p2r1_target_only_runtime.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("_run_ode_arm", source)
        self.assertNotIn("AcceptedPhysicalStateMaterializer", source)
        self.assertNotIn("solve_p1r", source)
        self.assertIn('"writer_materialization_count": 0', source)
        self.assertIn('"heldout_controller_access_count": 0', source)

    def test_two_alias_first_b10_and_b10x10_plans(self) -> None:
        for case_count in (1, 10):
            plan = dry.build_plan("0" * 40, case_count=case_count, repository_root=ROOT)
            self.assertEqual(plan["job_count"], 2)
            self.assertEqual(plan["writer_update_count"], 0)
            self.assertEqual(plan["target_microstep_attempt_count"], 2 * case_count * 24)
            self.assertEqual(
                len(
                    {
                        expected_p2r1_target_result_name(alias, case_count=case_count)
                        for alias in ("llama3-8b-inst", "qwen2.5-7b-inst")
                    }
                ),
                2,
            )

    def test_launcher_is_server1_two_gpu_max(self) -> None:
        submitter = (
            ROOT / "project/run_scripts/session05_ode_bf_submit_p2r1_target_only.py"
        ).read_text(encoding="utf-8")
        sbatch = (
            ROOT / "project/run_scripts/session05_ode_bf_p2r1_target_only.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("PROJECT_GPU_CAP = 4", submitter)
        self.assertIn("STAGE_GPU_MAX = 2", submitter)
        self.assertIn("#SBATCH --array=0-1%2", sbatch)
        self.assertIn("devbox", sbatch)
        self.assertNotIn("server2", sbatch)


if __name__ == "__main__":
    unittest.main()
