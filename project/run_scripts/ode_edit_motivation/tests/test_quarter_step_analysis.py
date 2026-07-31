from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from project.run_scripts.ode_edit_motivation import quarter_step_analysis as analysis
from project.run_scripts.ode_edit_motivation.quarter_step_refresh import (
    COEFFICIENT_PREFIX,
    COMMON_STEP_1,
    FIXED_PREFIX,
    NATIVE_ORDERED_FULL,
    NATIVE_ORDERED_SPLIT4,
    NO_OP_REPLAY,
    QSTEP_BRANCH_ORDER,
    QSTEP_STREAM_SCHEMA,
    REFRESHED_PREFIX,
    _checkpoint_arm,
)


def _sha(value: int) -> str:
    return f"{value:064x}"


def _case(index: int) -> dict:
    progress = {
        NO_OP_REPLAY: 0.0,
        COMMON_STEP_1: 0.10,
        _checkpoint_arm(REFRESHED_PREFIX, 2): 0.30,
        _checkpoint_arm(COEFFICIENT_PREFIX, 2): 0.25,
        _checkpoint_arm(FIXED_PREFIX, 2): 0.20,
        _checkpoint_arm(REFRESHED_PREFIX, 3): 0.50,
        _checkpoint_arm(COEFFICIENT_PREFIX, 3): 0.40,
        _checkpoint_arm(FIXED_PREFIX, 3): 0.30,
        _checkpoint_arm(REFRESHED_PREFIX, 4): 0.80,
        _checkpoint_arm(COEFFICIENT_PREFIX, 4): 0.60,
        _checkpoint_arm(FIXED_PREFIX, 4): 0.40,
        NATIVE_ORDERED_FULL: 0.50,
        NATIVE_ORDERED_SPLIT4: 0.50001,
    }
    endpoint_energy = {
        NO_OP_REPLAY: 0.0,
        COMMON_STEP_1: 1.0,
        _checkpoint_arm(REFRESHED_PREFIX, 2): 3.5,
        _checkpoint_arm(COEFFICIENT_PREFIX, 2): 3.8,
        _checkpoint_arm(FIXED_PREFIX, 2): 4.0,
        _checkpoint_arm(REFRESHED_PREFIX, 3): 8.2,
        _checkpoint_arm(COEFFICIENT_PREFIX, 3): 8.5,
        _checkpoint_arm(FIXED_PREFIX, 3): 9.0,
        _checkpoint_arm(REFRESHED_PREFIX, 4): 14.0,
        _checkpoint_arm(COEFFICIENT_PREFIX, 4): 15.0,
        _checkpoint_arm(FIXED_PREFIX, 4): 16.0,
        NATIVE_ORDERED_FULL: 16.0,
        NATIVE_ORDERED_SPLIT4: 16.0,
    }
    arms = {}
    for arm_id in QSTEP_BRANCH_ORDER:
        step = analysis._expected_step(arm_id)
        arms[arm_id] = {
            "progress": progress[arm_id],
            "endpoint_c_energy": endpoint_energy[arm_id],
            "cumulative_path_distance": float(step),
            "step_index": step,
            "nfe": 1,
            "success": True,
        }
    return {
        "model_alias": "llama3-8b-inst",
        "case_id": index,
        "request_id": _sha(index + 1000),
        "pass": True,
        "technical": {
            "exact_panel": True,
            "lineage_exact": True,
            "matched_per_hop_c": True,
            "rollback_exact": True,
            "firewall_pass": True,
            "receipt_before_outcome": True,
            "native_split_control": True,
        },
        "branch_order": list(QSTEP_BRANCH_ORDER),
        "arms": arms,
        "budgets": {
            "native_c_energy": 16.0,
            "per_hop_c_energy": 1.0,
            "native_c_distance": 4.0,
            "hop_c_distance": 1.0,
            "probe_c_distance": 0.0625,
        },
        "lineage": {
            "origin_state_id": _sha(index + 2000),
            "target_token_sha256": _sha(index + 3000),
            "direct_z_tensor_sha256": _sha(index + 4000),
            "context_id": _sha(9999),
            "final_lineage_ids": {
                REFRESHED_PREFIX: _sha(index + 5000),
                COEFFICIENT_PREFIX: _sha(index + 6000),
                FIXED_PREFIX: _sha(index + 7000),
            },
            "native_split_final_lineage_id": _sha(index + 8000),
        },
        "geometry": {
            "w0_to_current_c_cosines": {
                REFRESHED_PREFIX: [],
                COEFFICIENT_PREFIX: [],
                FIXED_PREFIX: [],
            }
        },
        "compute": {
            "controlled_nfe": 98,
            "proposal_build_count": 5,
            "probe_panel_count": 7,
            "wall_seconds": 1.0,
        },
    }


def _records() -> list[dict]:
    return [_case(index) for index in range(12)]


def _rotate(value):
    if isinstance(value, dict):
        items = [(key, _rotate(child)) for key, child in value.items()]
        return dict(items[1:] + items[:1])
    if isinstance(value, list):
        return [_rotate(child) for child in value]
    return value


class QuarterStepAnalysisTests(unittest.TestCase):
    def test_known_answer_direction_and_total_are_clear(self) -> None:
        summary = analysis.analyze_case_records(
            _records(), model_alias="llama3-8b-inst"
        )
        self.assertTrue(summary["technical_validity"]["pass"])
        self.assertEqual(summary["verdict"], "DIRECTION_REFRESH_CLEAR")
        self.assertAlmostEqual(
            summary["effects"]["direction_refresh_A4_minus_B4"]["mean"], 0.2
        )
        self.assertAlmostEqual(
            summary["effects"]["coefficient_refresh_B4_minus_C4"]["mean"], 0.2
        )
        self.assertAlmostEqual(
            summary["effects"]["total_refresh_A4_minus_C4"]["mean"], 0.4
        )
        self.assertEqual(summary["compute"]["controlled_nfe_total"], 1176)

    def test_failed_case_stays_in_itd_denominator(self) -> None:
        records = _records()
        records[0]["pass"] = False
        records[0]["technical"]["rollback_exact"] = False
        records[0]["arms"][_checkpoint_arm(REFRESHED_PREFIX, 4)]["progress"] = 999.0
        summary = analysis.analyze_case_records(
            records, model_alias="llama3-8b-inst"
        )
        self.assertEqual(summary["verdict"], "BLOCK_TECHNICAL_INVALID")
        self.assertEqual(summary["technical_validity"]["itd_denominator"], 12)
        self.assertFalse(summary["case_effects"][0]["analysis_success"])
        self.assertEqual(
            summary["case_effects"][0]["direction_refresh_effect_A4_minus_B4"],
            0.0,
        )

    def test_mapping_key_order_is_irrelevant_but_exact_set_is_required(self) -> None:
        rotated = [_rotate(record) for record in _records()]
        summary = analysis.analyze_case_records(
            rotated, model_alias="llama3-8b-inst"
        )
        self.assertTrue(summary["technical_validity"]["pass"])
        malformed = copy.deepcopy(rotated)
        malformed[0]["technical"]["extra"] = True
        with self.assertRaises(analysis.QuarterStepAnalysisError):
            analysis.analyze_case_records(
                malformed, model_alias="llama3-8b-inst"
            )

    def test_jsonl_loader_uses_sequence_not_key_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analysis_cases.jsonl"
            lines = []
            for sequence, record in enumerate(_records()):
                wrapper = {
                    "schema_version": QSTEP_STREAM_SCHEMA,
                    "run_id": "quarter-test",
                    "sequence": sequence,
                    "recorded_at": "2026-08-02T00:00:00+00:00",
                    "event": "quarter_step_analysis_case",
                    "payload": record,
                }
                lines.append(json.dumps(_rotate(wrapper)))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            loaded = analysis.load_analysis_cases(path)
            self.assertEqual(len(loaded), 12)


if __name__ == "__main__":
    unittest.main()
