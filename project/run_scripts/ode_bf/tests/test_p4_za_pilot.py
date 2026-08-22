from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

from project.run_scripts.ode_bf.p4_target_solver_binding import (
    non_barrier_arm_identity,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


class P4ZAPilotEssentialGateTests(unittest.TestCase):
    def test_non_barrier_identity_has_no_arm_and_binds_writer_zero(self) -> None:
        paired = SimpleNamespace(
            binding={"identity_sha256": "1" * 64},
            new=SimpleNamespace(
                request_order_sha256="2" * 64,
                context_sha256="3" * 64,
            ),
        )
        kl = SimpleNamespace(identity_sha256="4" * 64)
        first = non_barrier_arm_identity(
            paired,
            kl,
            teacher_sha256="5" * 64,
            origin_target_sha256="6" * 64,
            terminal_target_sha256="7" * 64,
            target_layer_name="model.layers.8.mlp.down_proj",
            learning_rate=0.1,
            kl_factor=0.0625,
            decay_factor=0.5,
            clamp_factor=0.75,
        )
        second = non_barrier_arm_identity(
            paired,
            kl,
            teacher_sha256="5" * 64,
            origin_target_sha256="6" * 64,
            terminal_target_sha256="7" * 64,
            target_layer_name="model.layers.8.mlp.down_proj",
            learning_rate=0.1,
            kl_factor=0.0625,
            decay_factor=0.5,
            clamp_factor=0.75,
        )
        self.assertEqual(first, second)
        self.assertNotIn("arm", first)
        self.assertEqual(first["writer_count"], 0)
        self.assertTrue(first["W0_shared"])
        self.assertEqual(first["inner_iterations"], 5)
        self.assertTrue(first["moment_reset_each_outer"])

    def test_target_binding_is_current_request_negative_only(self) -> None:
        source = (
            REPO_ROOT / "project/run_scripts/ode_bf/p4_target_solver_binding.py"
        ).read_text(encoding="utf-8")
        self.assertIn('source.get("target_true")', source)
        self.assertIn('old["target_new"] = target_true', source)
        self.assertIn('"historical_negative_access_count": 0', source)
        self.assertIn("smooth_semantic_logodds_potential", source)
        self.assertIn("evaluate_p1r24_kl", source)
        self.assertIn(".detach().to(target_state.device)", source)

    def test_runner_binds_exact_arm_panel_and_direct_native(self) -> None:
        source = (
            REPO_ROOT / "project/run_scripts/session05_ode_bf_p4_za_pilot.py"
        ).read_text(encoding="utf-8")
        self.assertIn("P4TargetArm.POSITIVE, P4TargetArm.POSITIVE_NEGATIVE", source)
        self.assertIn("from easyeditor.models.alphaedit.compute_z import compute_z", source)
        self.assertIn('"writer_call_count": 0', source)
        self.assertIn("reuse_final_verified_closure=True", source)
        self.assertIn("EXECUTION_MICROBATCH = 1", source)
        self.assertIn("transferred_bf16_plan_decision_influence_count", source)
        self.assertIn("first_target_completion_nonfinite_restore_gate", source)

    def test_final_gate_is_no_model_and_full_fp32_offline(self) -> None:
        source = (
            REPO_ROOT
            / "project/run_scripts/session05_ode_bf_p4_za_final_pre_gpu.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("from_pretrained(", source)
        self.assertIn('"dtype": "torch.float32"', source)
        self.assertIn('"local_files_only": True', source)
        self.assertIn('"hf_duplicate_rehash_count": 0', source)
        self.assertIn('"model_load_count": 0', source)
        self.assertIn('"slurm_submit_count": 0', source)

    def test_sbatch_is_two_cell_cap2_held_ready_shape(self) -> None:
        source = (
            REPO_ROOT
            / "project/run_scripts/session05_ode_bf_p4_za_case01_server4.sbatch"
        ).read_text(encoding="utf-8")
        for expected in (
            "#SBATCH --array=0-1%2",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=65000M",
            "#SBATCH --time=48:00:00",
            "#SBATCH --nodelist=server4",
            "export PROJECT_GPU_CAP=2",
            'EXPECTED_SESSION="codex://threads/01a028a7-9e3c-7541-81ba-efb40555d17d"',
        ):
            self.assertIn(expected, source)


if __name__ == "__main__":
    unittest.main()
