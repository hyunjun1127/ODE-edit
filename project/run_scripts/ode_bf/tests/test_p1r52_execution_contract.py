from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from project.run_scripts import session05_ode_bf_p1r52_rsa_r42safekdc_m1_dry_plan as dry
from project.run_scripts.ode_bf.p1r52_independent_panel import LOCK_FILE, load_and_validate_lock
from project.run_scripts.ode_bf.p1r52_r42_safe_kdc import prepare_p1r52_target_proposal


ROOT = Path(__file__).resolve().parents[4]


class P1R52ExecutionContractTests(unittest.TestCase):
    def test_lock_and_four_cell_plan(self) -> None:
        lock, _ = load_and_validate_lock(ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE)
        self.assertEqual(lock["additional_kl_forward_count"], 0)
        self.assertEqual(lock["additional_kl_backward_count"], 0)
        self.assertEqual(lock["writer_router_change_count"], 0)
        self.assertEqual(lock["repair_revision"], "R1")
        self.assertEqual(lock["active_direction_denominator_epsilon_count"], 0)
        self.assertEqual(lock["raw_energy_certificate_tolerance"], 1e-8)
        plan = dry.build_plan("0" * 40, attempt_suffix="repair-r1")
        self.assertEqual(plan["job_count"], 4)
        self.assertEqual(plan["array"], "0-3%4")
        self.assertEqual(len({job["result_name"] for job in plan["jobs"]}), 4)
        self.assertEqual({(job["alias"], job["arm"]) for job in plan["jobs"]}, {
            ("llama3-8b-inst", "neutral"), ("llama3-8b-inst", "soft"),
            ("qwen2.5-7b-inst", "neutral"), ("qwen2.5-7b-inst", "soft"),
        })
        self.assertTrue(all("-repair-r1-v1" in job["result_name"] for job in plan["jobs"]))

    def test_policy_is_tensor_only_and_forbidden_controls_absent(self) -> None:
        source = inspect.getsource(prepare_p1r52_target_proposal)
        for forbidden in ("model(", ".backward(", "evaluate_scalable", "P2R", "remaining_steps", "historical", "sequential"):
            self.assertNotIn(forbidden, source)
        self.assertIn("kl: P1R24KLResult", source)
        self.assertIn("POST_CLAMP_POST_FP32_CAST_ACTUAL_DELTA", source)
        self.assertIn("combined_direction[:, active] / combined_norm[active].unsqueeze(0)", source)
        self.assertNotIn("combined_norm[active] + P1R52_NUMERICAL_EPSILON", source)
        self.assertIn("raw_energy_relative_error", source)

    def test_launcher_is_isolated_server1_four_cell_wave(self) -> None:
        sbatch = (ROOT / "project/run_scripts/session05_ode_bf_p1r52_rsa_r42safekdc_m1.sbatch").read_text(encoding="utf-8")
        submitter = (ROOT / "project/run_scripts/session05_ode_bf_submit_p1r52_rsa_r42safekdc_m1.py").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-3%4", sbatch)
        self.assertIn('EXPECTED_BRANCH="codex/p1r52-rsa-r42safekdc-m1-v1"', sbatch)
        self.assertIn("PROJECT_GPU_CAP = 4", submitter)
        self.assertIn("active != 0", submitter)
        self.assertNotIn("server2", sbatch)

    def test_teacher_hash_is_captured_once_and_bound_across_k8(self) -> None:
        source = (
            ROOT / "project/run_scripts/ode_bf/p1_scalable_batched_experiment.py"
        ).read_text(encoding="utf-8")
        self.assertIn("p1r24_kl_teacher_sha256 = canonical_hash", source)
        self.assertIn("kl_teacher_input_sha256=p1r24_kl_teacher_sha256", source)
        self.assertIn("len(set(p1r52_teacher_hashes)) != 1", source)
        self.assertIn('"kl_teacher_hash_k8_constant"', source)


if __name__ == "__main__":
    unittest.main()
