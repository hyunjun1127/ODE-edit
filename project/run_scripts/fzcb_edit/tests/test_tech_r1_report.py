from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from project.run_scripts.fzcb_edit.build_tech_r1_joint_report import MODELS, build


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class TechR1ReportTests(unittest.TestCase):
    def test_raw_free_joint_package_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_root = root / "results"
            state_root = root / "state"
            output_root = root / "report"
            technical_root = root / "campaign-39d2f67-tech-r4"
            _write(result_root / "campaign-manifest.json", {"job": "1"})
            _write(technical_root / "campaign-manifest.json", {"job": "old"})
            for name in ("focused-gate.json", "pre-gpu.json", "submission-receipt.json"):
                _write(state_root / name, {"status": "PASS"})
            for model in MODELS:
                result = {
                    "model": model,
                    "case_ids": [19795],
                    "typed_conclusion": "SUBSPACE_INCONCLUSIVE",
                    "valid_arm_denominator": 3,
                    "attempted_arm_denominator": 5,
                    "target": {"a0_norm": 1.0, "delta_star_norm": 0.5, "z_star_norm": 1.1, "delta_over_a0": 0.5},
                    "arms": [
                        {"arm": "OFFICIAL_MEMIT", "status": "TERMINAL_VALID", "denominator": {"valid_terminal": 1}, "terminal": {"weight_root": "o"}},
                        {"arm": "TRUE_FROZEN_C_SPLIT", "status": "SCIENTIFIC_CONTROLLER_HOLD", "denominator": {"valid_terminal": 0}, "A0": 1.0, "exception": "closure"},
                        {"arm": "REFRESHED_EQUALITY_ONLY", "status": "TERMINAL_VALID", "denominator": {"valid_terminal": 1}, "A0": 1.0, "terminal": {"spent_action": 0.9, "weight_root": "e"}},
                        {
                            "arm": "FZCB", "status": "SUBSPACE_INCONCLUSIVE",
                            "denominator": {"valid_terminal": 0}, "A0": 1.0,
                            "psi0": 0.1, "psi_min": 0.05, "g_free_norm_squared": 0.1,
                            "fd": {"failure": {"steps": [{
                                "multiplier": 1.0, "epsilon": 0.01,
                                "selected_derivative": 0.2, "repeat_noise": 0.0,
                                "repeats": [{
                                    "repeat": 0, "positive_value": 1.1,
                                    "negative_value": 0.9, "derivative": 0.2,
                                }],
                            }]}},
                            "evidence": {"sensitivity_method": "SKETCH"},
                        },
                        {"arm": "STRONG_STATIC_SAME_OBJECTIVE", "status": "TERMINAL_VALID_STRICT", "denominator": {"valid_terminal": 1}, "A0": 1.0, "terminal": {"spent_action": 0.95, "weight_root": "s"}},
                    ],
                }
                _write(result_root / f"{model}-B1/result.json", result)
                _write(result_root / f"{model}-B1/arm-journal/journal-index.json", {"members": []})
            summary = build(
                result_root=result_root,
                state_root=state_root,
                output_root=output_root,
                source_head="h",
                source_tree="t",
                job_id="1",
                technical_attempt_roots=(technical_root,),
            )
            self.assertEqual(len(summary["summaries"]), 2)
            self.assertTrue(Path(summary["report"]["path"]).is_file())
            self.assertTrue(Path(summary["manifest"]["path"]).is_file())
            receipt = json.loads(Path(summary["receipt"]["path"]).read_text())
            self.assertEqual(len(receipt["member_root"]), 64)
            self.assertEqual(receipt["b10_submit_count"], 0)
            self.assertIn("EQUALITY_FAILURE", (output_root / "fd-axis-sweep.csv").read_text())
            self.assertIn("INCOMPLETE_STRONG_STATIC_FAILURE_RECEIPT", (output_root / "failure-ledger.csv").read_text())


if __name__ == "__main__":
    unittest.main()
