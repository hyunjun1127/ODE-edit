from __future__ import annotations

import json
from pathlib import Path
import unittest

from project.run_scripts.ode_bf.contracts import canonical_hash


REPO_ROOT = Path(__file__).resolve().parents[4]


class P4EulerCalibrationR1FocusedTests(unittest.TestCase):
    def test_approved_lock_exact_prefix_reuse(self) -> None:
        path = (
            REPO_ROOT
            / "project/run_scripts/ode_bf/locks/"
            "numerical_lock_s05_p4_euler_calibration_r1.json"
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        body = dict(value)
        root = body.pop("root_digest")
        self.assertEqual(root, canonical_hash(body))
        self.assertEqual(value["stage_1"]["h_grid"], [0.0625, 0.25, 1.0, 4.0])
        self.assertEqual(value["stage_1"]["prefix_M"], [1, 3, 5, 10])
        self.assertTrue(value["stage_1"]["one_trajectory_per_model_arm_h"])
        self.assertEqual(value["stage_1"]["separate_prefix_trajectory_count"], 0)
        self.assertEqual(value["calibration_boundary"]["slice"], "B1_CASE01")
        self.assertFalse(value["calibration_boundary"]["B1_future_reentry_allowed"])

    def test_model_binding_has_no_gradient_or_optimizer_authority(self) -> None:
        source = (
            REPO_ROOT / "project/run_scripts/ode_bf/p4_euler_model_binding.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("torch.autograd.grad(", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertIn("use_reentrant=False", source)
        self.assertIn("actual_model_forward_count_after_gradient", source)
        self.assertIn("value_device = target_state.device", source)
        self.assertIn("kl_values.to(device=value_device", source)

    def test_stage1_uses_one_m10_trajectory_and_prefix_snapshots(self) -> None:
        source = (
            REPO_ROOT
            / "project/run_scripts/session05_ode_bf_p4_euler_calibration_stage1.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(source.count("run_raw_projected_euler("), 2)
        stage1_body = source.split("def _fixed_endpoint_trajectory(", 1)[0]
        self.assertEqual(stage1_body.count("run_raw_projected_euler("), 1)
        for expected in (
            "microsteps=10",
            "1: pre_states[1]",
            "3: pre_states[3]",
            "5: pre_states[5]",
            "10: result.final_state.detach().clone()",
            '"separate_prefix_trajectory_count": 0',
            '"duplicate_autograd_evaluation_count": 0',
            '"prefix_observation_decision_influence_count": 0',
        ):
            self.assertIn(expected, source)

    def test_endpoint_observation_has_no_duplicate_field_evaluation(self) -> None:
        binding = (
            REPO_ROOT / "project/run_scripts/ode_bf/p4_euler_model_binding.py"
        ).read_text(encoding="utf-8")
        runner = (
            REPO_ROOT
            / "project/run_scripts/session05_ode_bf_p4_euler_calibration_stage1.py"
        ).read_text(encoding="utf-8")
        self.assertIn("@torch.no_grad()", binding)
        self.assertIn('"autograd_grad_call_count": 0', binding)
        self.assertIn("z_M9_PRE_FINAL_UPDATE_NO_DUPLICATE_AUTOGRAD", runner)

    def test_preflight_is_no_model_no_cuda_and_source_bound(self) -> None:
        source = (
            REPO_ROOT
            / "project/run_scripts/session05_ode_bf_p4_euler_calibration_preflight.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("from_pretrained(", source)
        self.assertNotIn("torch.cuda", source)
        self.assertIn('"model_load_count": 0', source)
        self.assertIn('"slurm_submit_count": 0', source)
        self.assertIn("source_manifest", source)

    def test_sbatch_is_two_model_cells_at_cap2(self) -> None:
        source = (
            REPO_ROOT
            / "project/run_scripts/"
            "session05_ode_bf_p4_euler_calibration_stage1_server4.sbatch"
        ).read_text(encoding="utf-8")
        for expected in (
            "#SBATCH --array=0-1%2",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=60416M",
            "#SBATCH --time=48:00:00",
            "#SBATCH --nodelist=server4",
            "export PROJECT_GPU_CAP=2",
            'EXPECTED_BRANCH="codex/server4-p4-euler-calibration-r1"',
        ):
            self.assertIn(expected, source)

    def test_stage2_is_fixed_horizon_m5_m10_and_fail_closed(self) -> None:
        source = (
            REPO_ROOT
            / "project/run_scripts/session05_ode_bf_p4_euler_calibration_stage1.py"
        ).read_text(encoding="utf-8")
        for expected in (
            "SELECTED_H = 0.25",
            "SELECTED_TARGET_HORIZON = 1.25",
            "for microsteps in (5, 10)",
            "numerator = torch.linalg.vector_norm(endpoints[5] - endpoints[10]",
            'summary["median"] <= 0.10',
            'summary["p90"] <= 0.25',
            'summary["max"] <= 0.50',
            'float(row["clamp_fraction"]) < 0.5',
            '"actual_autograd_grad_call_count": 30',
            '"STAGE2_MODEL_CELL_SCIENTIFIC_HOLD"',
        ):
            self.assertIn(expected, source)

    def test_stage2_sbatch_preserves_cap_and_science_lock(self) -> None:
        source = (
            REPO_ROOT
            / "project/run_scripts/"
            "session05_ode_bf_p4_euler_calibration_stage2_server4.sbatch"
        ).read_text(encoding="utf-8")
        for expected in (
            "#SBATCH --array=0-1%2",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=60416M",
            "#SBATCH --time=48:00:00",
            "--stage stage2",
            "export PROJECT_GPU_CAP=2",
        ):
            self.assertIn(expected, source)


if __name__ == "__main__":
    unittest.main()
