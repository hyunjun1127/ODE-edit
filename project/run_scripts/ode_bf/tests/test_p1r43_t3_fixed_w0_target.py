from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from project.run_scripts import session05_ode_bf_p1r43_t3_target_only_dry_plan as dry
from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1r43_rho_free_target import (
    P1R43ControllerState,
    prepare_p1r43_target_proposal,
)
from project.run_scripts.ode_bf.p1r43_t3_target_only_panel import (
    LOCK_FILE,
    PARENT,
    PARENT_TREE,
    load_and_validate_lock,
)
from project.run_scripts.ode_bf.p1r43_t3_target_only_runtime import (
    P1R43_T3_ENDPOINT_UPDATES,
    P1R43_T3_TARGET_UPDATE_COUNT,
    _heldout_eff_gen_lookup_geometry,
    expected_p1r43_t3_target_result_name,
)
from project.run_scripts.ode_bf.p1r24_atomic_strength import P1R24_H
from project.run_scripts.ode_bf.scalable_batched_model import ScalableObjectiveResult


ROOT = Path(__file__).resolve().parents[4]


def _objective(gradient: torch.Tensor) -> ScalableObjectiveResult:
    values = (1.0, 2.0)
    return ScalableObjectiveResult(
        sum(values) / len(values),
        values,
        "1" * 64,
        "2" * 64,
        tuple("3" * 64 for _ in values),
        tuple(1 for _ in values),
        1,
        1,
        10,
        10,
        gradient,
        None,
        "4" * 64,
        "5" * 64,
        "6" * 64,
    )


class P1R43T3FixedW0TargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = torch.tensor([[2.0, 3.0], [1.0, 1.0]])
        self.terminal = self.target.clone()
        self.gradient = torch.tensor([[0.25, 0.0], [0.0, 0.5]])

    def _proposal(self, step: int, maximum: int = 8):
        return prepare_p1r43_target_proposal(
            self.target,
            self.terminal,
            _objective(self.gradient),
            P1R43ControllerState.zero(self.target),
            alias="llama3-8b-inst",
            step_index=step,
            shared_speed=0.4,
            max_target_updates=maximum,
        )

    def test_legacy_default_and_explicit_eight_are_byte_semantically_identical(self) -> None:
        default = prepare_p1r43_target_proposal(
            self.target,
            self.terminal,
            _objective(self.gradient),
            P1R43ControllerState.zero(self.target),
            alias="llama3-8b-inst",
            step_index=7,
            shared_speed=0.4,
        )
        explicit = self._proposal(7, 8)
        self.assertTrue(torch.equal(default.primary_step.target_next, explicit.primary_step.target_next))
        self.assertEqual(default.receipt, explicit.receipt)
        self.assertNotIn("max_target_updates", default.receipt)

    def test_indices_zero_through_twenty_three_and_fixed_h(self) -> None:
        first = self._proposal(0, 24)
        last = self._proposal(23, 24)
        self.assertEqual(first.receipt["h"], 0.125)
        self.assertEqual(last.receipt["h"], P1R24_H)
        self.assertEqual(last.receipt["max_target_updates"], 24)
        with self.assertRaises(ODEBFContractError):
            self._proposal(24, 24)

    def test_lock_and_two_alias_plans(self) -> None:
        lock, _ = load_and_validate_lock(
            ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        self.assertEqual(PARENT, "11508b6da11d606521b703037034e1814b70d8a8")
        self.assertEqual(PARENT_TREE, "a0e71bbb27cf36b3b51bf6cde4c92cdc9a47879b")
        self.assertEqual(lock["target_update_count"], 24)
        self.assertEqual(lock["target_snapshot_indices"], [0, 8, 24])
        for case_count in (1, 10):
            plan = dry.build_plan("0" * 40, case_count=case_count, repository_root=ROOT)
            self.assertEqual(plan["job_count"], 2)
            self.assertEqual(plan["target_microstep_attempt_count"], 2 * case_count * 24)
            self.assertEqual(plan["writer_materialization_count"], 0)
            self.assertEqual(
                len(
                    {
                        expected_p1r43_t3_target_result_name(alias, case_count=case_count)
                        for alias in ("llama3-8b-inst", "qwen2.5-7b-inst")
                    }
                ),
                2,
            )

    def test_runtime_is_continuous_fixed_w0_target_only(self) -> None:
        source = (
            ROOT / "project/run_scripts/ode_bf/p1r43_t3_target_only_runtime.py"
        ).read_text(encoding="utf-8")
        self.assertIn("range(P1R43_T3_TARGET_UPDATE_COUNT)", source)
        self.assertIn("endpoint_targets[microstep + 1] = current_target.clone()", source)
        self.assertIn('"controller_reset_at_8_count": 0', source)
        self.assertIn('"controller_reset_at_16_count": 0', source)
        self.assertIn("snapshot_index=8", source)
        self.assertIn("accepted_snapshot_count=9", source)
        self.assertNotIn("microstep % 8", source)
        self.assertNotIn("p2r1_target_update", source)
        self.assertNotIn("AcceptedPhysicalStateMaterializer", source)
        self.assertNotIn("solve_p1r", source)
        self.assertEqual(P1R43_T3_TARGET_UPDATE_COUNT, 24)
        self.assertEqual(P1R43_T3_ENDPOINT_UPDATES, (8, 24))

    def test_launcher_is_server1_two_gpu_max(self) -> None:
        submitter = (
            ROOT / "project/run_scripts/session05_ode_bf_submit_p1r43_t3_target_only.py"
        ).read_text(encoding="utf-8")
        sbatch = (
            ROOT / "project/run_scripts/session05_ode_bf_p1r43_t3_target_only.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("PROJECT_GPU_CAP = 4", submitter)
        self.assertIn("STAGE_GPU_MAX = 2", submitter)
        self.assertIn("#SBATCH --array=0-1%2", sbatch)
        self.assertIn("devbox", sbatch)
        self.assertNotIn("server2", sbatch)

    def test_eff_gen_overlay_geometry_excludes_locality_rows(self) -> None:
        class Tokenizer:
            padding_side = "left"

            def __call__(self, value, **_kwargs):
                if isinstance(value, str):
                    return {"input_ids": list(range(len(value.split()) + 1))}
                lengths = [len(item.split()) + 1 for item in value]
                maximum = max(lengths)
                mask = torch.zeros((len(value), maximum), dtype=torch.int64)
                for row, length in enumerate(lengths):
                    mask[row, maximum - length :] = 1
                return {"input_ids": mask, "attention_mask": mask}

        requests = []
        cases = []
        for index in range(10):
            digest = f"{index:064x}"
            subject = f"subject{index}"
            requests.append({"request_sha256": digest, "subject": subject})
            cases.append(
                SimpleNamespace(
                    request_sha256=digest,
                    rewrite_prompt=f"rewrite {subject}",
                    paraphrase_prompts=(f"paraphrase {subject}",),
                    neighborhood_prompts=(f"locality {subject}",),
                    target_new="new",
                    target_true="old",
                )
            )
        with patch(
            "easyeditor.models.alphaedit.AlphaEdit_main.find_fact_lookup_idx",
            return_value=-1,
        ):
            positions, patched, receipt = _heldout_eff_gen_lookup_geometry(
                Tokenizer(), requests, cases, fact_token_strategy="subject_last"
            )
        self.assertEqual(patched, (4,) * 10)
        self.assertTrue(all(len(item) == 4 for item in positions))
        self.assertEqual(receipt["locality_row_count"], 0)


if __name__ == "__main__":
    unittest.main()
