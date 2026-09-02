from __future__ import annotations

import json
from pathlib import Path
import unittest

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p4_euler_orchestration import build_za_phase_contract


REPO_ROOT = Path(__file__).resolve().parents[4]


class P4EulerZALlamaH1EssentialTests(unittest.TestCase):
    def test_lock_binds_user_revision_and_projection_risk(self) -> None:
        path = REPO_ROOT / (
            "project/run_scripts/ode_bf/locks/"
            "numerical_lock_s05_p4_euler_za_llama_h1_b2b10.json"
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        body = dict(value)
        root = body.pop("root_digest")
        self.assertEqual(root, canonical_hash(body))
        self.assertEqual(
            value["policy_provenance"],
            "USER_DIRECTED_POST_CALIBRATION_HYPERPARAMETER_REVISION",
        )
        self.assertEqual(
            value["interpretation"],
            "EXPLORATORY_CONFIRMATION_AFTER_CALIBRATION",
        )
        self.assertEqual(
            value["selected_setting"],
            {
                "h": 1.0,
                "M": 5,
                "T_z": 5.0,
                "formula": "T_z=M*h",
                "under_edit_mitigation_intent": True,
                "model_or_arm_specific_branch_count": 0,
            },
        )
        self.assertEqual(
            value["calibration_evidence_boundary"][
                "h_1_M5_clamp_hit_fraction"
            ]["Z_plus_minus"]["fraction"],
            0.86,
        )
        self.assertFalse(
            value["claim_boundary"]["continuous_ODE_convergence_claim"]
        )

    def test_za_phase_is_one_local_solve_without_writer_or_k8(self) -> None:
        for arm in ("Z+", "Z±"):
            row = build_za_phase_contract(arm=arm, local_solve_count=1)
            self.assertEqual(row["local_M_step_solve_count"], 1)
            self.assertEqual(row["outer_k8_repeat_count"], 0)
            self.assertEqual(row["writer_apply_count"], 0)

    def test_runner_is_raw_euler_h1_m5_and_adam_free(self) -> None:
        source = (
            REPO_ROOT
            / "project/run_scripts/session05_ode_bf_p4_euler_za_llama_h1.py"
        ).read_text(encoding="utf-8")
        for expected in (
            "SELECTED_H = 1.0",
            "SELECTED_M = 5",
            "SELECTED_TARGET_HORIZON = 5.0",
            "run_raw_projected_euler(",
            "P4TargetArm.POSITIVE, P4TargetArm.POSITIVE_NEGATIVE",
            "from easyeditor.models.alphaedit.compute_z import compute_z",
            '"writer_materialization_count": 0',
            '"cache_append_count": 0',
            '"outer_k8_repeat_count": 0',
            '"high_clamp_fraction_stop_rule_count": 0',
            "fixed_budget_slots_completed=0",
            "freeze_snapshot_index=0",
        ):
            self.assertIn(expected, source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("run_fixed_m_target_adam", source)

    def test_runner_binds_exact_B2_B10_and_llama_only(self) -> None:
        source = (
            REPO_ROOT
            / "project/run_scripts/session05_ode_bf_p4_euler_za_llama_h1.py"
        ).read_text(encoding="utf-8")
        self.assertIn("CASE_INDICES = tuple(range(2, 11))", source)
        self.assertIn('MODEL_ALIAS = "llama3-8b-inst"', source)
        self.assertIn('choices=(MODEL_ALIAS,)', source)
        self.assertIn('"qwen_submission_count": 0', source)

    def test_preflight_is_no_model_no_cuda_and_exact_setting(self) -> None:
        source = (
            REPO_ROOT
            / "project/run_scripts/"
            "session05_ode_bf_p4_euler_za_llama_h1_preflight.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("from_pretrained(", source)
        self.assertNotIn("torch.cuda", source)
        for expected in (
            '"model_load_count": 0',
            '"cuda_api_access_count": 0',
            '"slurm_submit_count": 0',
            '"h": SELECTED_H',
            '"M": SELECTED_M',
            '"T_z": SELECTED_TARGET_HORIZON',
            '"B1_excluded": True',
        ):
            self.assertIn(expected, source)

    def test_sbatch_is_llama_B2_B10_cap2(self) -> None:
        source = (
            REPO_ROOT
            / "project/run_scripts/"
            "session05_ode_bf_p4_euler_za_llama_h1_server4.sbatch"
        ).read_text(encoding="utf-8")
        for expected in (
            "#SBATCH --array=2-10%2",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=60416M",
            "#SBATCH --time=48:00:00",
            "#SBATCH --nodelist=server4",
            "export PROJECT_GPU_CAP=2",
            'readonly MODEL="llama3-8b-inst"',
        ):
            self.assertIn(expected, source)

    def test_terminal_panel_defaults_preserve_prior_callers(self) -> None:
        source = (
            REPO_ROOT / "project/run_scripts/ode_bf/p2r1_target_only_runtime.py"
        ).read_text(encoding="utf-8")
        self.assertIn("freeze_snapshot_index: int = 8", source)
        self.assertIn("freeze_accepted_snapshot_count: int = 9", source)
        self.assertIn("native_endpoint_runtime_access_count: int = 0", source)


if __name__ == "__main__":
    unittest.main()
