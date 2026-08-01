from __future__ import annotations

import copy
import unittest

from project.run_scripts.ode_edit_motivation import adaptive_gate_pair_analysis as pair
from project.run_scripts.ode_edit_motivation.adaptive_gate_analysis import ANALYSIS_SCHEMA
from project.run_scripts.ode_edit_motivation.adaptive_gate_refresh import TRACK_MEMIT


EFFECT_KEYS = (
    "gated_minus_refreshed_G4_minus_A4",
    "gated_minus_coefficient_G4_minus_B4",
    "gated_minus_fixed_G4_minus_C4",
    "native_gap_reduction_abs_A4_native_minus_abs_G4_native",
    "gated_minus_casewise_max_A4_B4",
)


def _analysis(model: str, *, gate: bool = True) -> dict:
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "model_alias": model,
        "editor_track": TRACK_MEMIT,
        "fixed_contract": {
            "case_count": 8,
            "k": 4,
            "gate_relative_margin": 0.02,
        },
        "technical_validity": {"pass": True},
        "lenient_gate": {"model_gate_pass": gate},
        "effects": {
            key: {"mean": 0.1 + 0.01 * index}
            for index, key in enumerate(EFFECT_KEYS)
        },
    }


class AdaptiveGatePairAnalysisTests(unittest.TestCase):
    def test_pair_pass_requires_both_models(self) -> None:
        result = pair.analyze_pair(
            _analysis(pair.LLAMA_ALIAS),
            _analysis(pair.QWEN_ALIAS),
            track=TRACK_MEMIT,
        )
        self.assertTrue(result["same_method_gate"])
        self.assertTrue(result["pair_gate_pass"])
        self.assertEqual(result["verdict"], "LENIENT_PAIR_PASS")

    def test_one_model_failure_is_not_pooled_away(self) -> None:
        result = pair.analyze_pair(
            _analysis(pair.LLAMA_ALIAS),
            _analysis(pair.QWEN_ALIAS, gate=False),
            track=TRACK_MEMIT,
        )
        self.assertFalse(result["pair_gate_pass"])
        self.assertEqual(result["verdict"], "LENIENT_PAIR_PARTIAL_OR_FAIL")

    def test_method_contract_mismatch_blocks_pair(self) -> None:
        qwen = copy.deepcopy(_analysis(pair.QWEN_ALIAS))
        qwen["fixed_contract"]["gate_relative_margin"] = 0.03
        result = pair.analyze_pair(
            _analysis(pair.LLAMA_ALIAS),
            qwen,
            track=TRACK_MEMIT,
        )
        self.assertFalse(result["same_method_gate"])
        self.assertEqual(result["verdict"], "BLOCK_METHOD_MISMATCH")

    def test_wrong_model_identity_is_rejected(self) -> None:
        with self.assertRaises(pair.AdaptiveGatePairAnalysisError):
            pair.analyze_pair(
                _analysis(pair.QWEN_ALIAS),
                _analysis(pair.QWEN_ALIAS),
                track=TRACK_MEMIT,
            )


if __name__ == "__main__":
    unittest.main()
