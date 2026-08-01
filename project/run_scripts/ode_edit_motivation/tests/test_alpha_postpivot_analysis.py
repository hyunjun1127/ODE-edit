from __future__ import annotations

import copy
import unittest

from project.run_scripts.ode_edit_motivation.adaptive_gate_analysis import ANALYSIS_SCHEMA
from project.run_scripts.ode_edit_motivation.adaptive_gate_refresh import TRACK_ALPHAEDIT
from project.run_scripts.ode_edit_motivation.alpha_postpivot_analysis import (
    AlphaPostpivotAnalysisError,
    analyze_postpivot_pair,
)


def _analysis(model: str, direction_values: list[float]) -> dict[str, object]:
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "model_alias": model,
        "editor_track": TRACK_ALPHAEDIT,
        "fixed_contract": {"same": True},
        "technical_validity": {"pass": True},
        "effects": {
            "direction_refresh_A4_minus_B4": {
                "mean": sum(direction_values) / len(direction_values)
            },
            "coefficient_refresh_B4_minus_C4": {"mean": 0.2},
        },
        "case_effects": [
            {
                "case_id": str(index),
                "direction_refresh_A4_minus_B4": value,
            }
            for index, value in enumerate(direction_values)
        ],
        "projector": {
            "right_norm_retention_count": 80,
            "mean_right_norm_retention": 0.5,
            "minimum_right_norm_retention": 0.0,
        },
    }


class AlphaPostpivotAnalysisTests(unittest.TestCase):
    def test_pair_signal_requires_both_models(self) -> None:
        llama = _analysis("llama3-8b-inst", [0.2] * 8)
        qwen = _analysis("qwen2.5-7b-inst", [0.1] * 8)
        result = analyze_postpivot_pair(llama, qwen)
        self.assertEqual(result["verdict"], "ALPHA_ALWAYS_REFRESH_TRANSFER_SIGNAL")
        self.assertTrue(result["pair_gate_pass"])

    def test_one_model_no_direction_closes_common_transfer(self) -> None:
        llama = _analysis("llama3-8b-inst", [0.2] * 8)
        qwen = _analysis("qwen2.5-7b-inst", [-0.01] * 8)
        result = analyze_postpivot_pair(llama, qwen)
        self.assertEqual(result["verdict"], "ALPHA_COMMON_TRANSFER_NO_SIGNAL")
        self.assertFalse(result["models"]["qwen2.5-7b-inst"]["direction_pass"])

    def test_casewise_noncollapse_is_not_hidden_by_positive_mean(self) -> None:
        llama = _analysis("llama3-8b-inst", [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        qwen = _analysis("qwen2.5-7b-inst", [0.2] * 8)
        result = analyze_postpivot_pair(llama, qwen)
        self.assertGreater(
            result["models"]["llama3-8b-inst"]["direction_transfer_A4_minus_B4"],
            0.0,
        )
        self.assertFalse(result["models"]["llama3-8b-inst"]["noncollapse_pass"])
        self.assertEqual(result["verdict"], "ALPHA_COMMON_TRANSFER_NO_SIGNAL")

    def test_pair_rejects_contract_drift(self) -> None:
        llama = _analysis("llama3-8b-inst", [0.2] * 8)
        qwen = _analysis("qwen2.5-7b-inst", [0.2] * 8)
        qwen = copy.deepcopy(qwen)
        qwen["fixed_contract"] = {"same": False}
        result = analyze_postpivot_pair(llama, qwen)
        self.assertEqual(result["verdict"], "BLOCK_METHOD_MISMATCH")

    def test_rejects_wrong_projector_count(self) -> None:
        llama = _analysis("llama3-8b-inst", [0.2] * 8)
        qwen = _analysis("qwen2.5-7b-inst", [0.2] * 8)
        llama["projector"]["right_norm_retention_count"] = 79
        with self.assertRaisesRegex(AlphaPostpivotAnalysisError, "retention count"):
            analyze_postpivot_pair(llama, qwen)


if __name__ == "__main__":
    unittest.main()
