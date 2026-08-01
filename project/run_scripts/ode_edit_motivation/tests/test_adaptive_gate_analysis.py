from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from project.run_scripts.ode_edit_motivation import adaptive_gate_analysis as analysis
from project.run_scripts.ode_edit_motivation.adaptive_gate_refresh import (
    AGATE_BRANCH_ORDER,
    AGATE_CASE_COUNT,
    AGATE_CONTROLLED_NFE,
    AGATE_PROBE_PANEL_COUNT,
    COEFFICIENT_FINAL,
    FIXED_FINAL,
    GATE_RELATIVE_MARGIN,
    GATED_FINAL,
    REFRESHED_FINAL,
    STREAM_SCHEMA,
    TRACK_ALPHAEDIT,
    TRACK_MEMIT,
)
from project.run_scripts.ode_edit_motivation.quarter_step_refresh import (
    NATIVE_ORDERED_FULL,
    NATIVE_ORDERED_SPLIT4,
    NO_OP_REPLAY,
)


def _sha(value: int) -> str:
    return f"{value:064x}"


def _case(index: int, *, track: str = TRACK_MEMIT) -> dict:
    progress = {
        NO_OP_REPLAY: 0.0,
        GATED_FINAL: 0.8,
        REFRESHED_FINAL: 0.7,
        COEFFICIENT_FINAL: 0.6,
        FIXED_FINAL: 0.5,
        NATIVE_ORDERED_FULL: 0.65,
        NATIVE_ORDERED_SPLIT4: 0.65001,
    }
    return {
        "model_alias": "llama3-8b-inst",
        "editor_track": track,
        "case_id": str(index),
        "request_id": _sha(index + 100),
        "pass": True,
        "technical": {key: True for key in analysis.TECHNICAL_KEYS},
        "branch_order": list(AGATE_BRANCH_ORDER),
        "arms": {
            arm: {
                "progress": progress[arm],
                "endpoint_c_energy": 0.0 if arm == NO_OP_REPLAY else 16.0,
                "success": True,
            }
            for arm in AGATE_BRANCH_ORDER
        },
        "gate": {
            "relative_margin": GATE_RELATIVE_MARGIN,
            "choices": ["refreshed", "fixed", "refreshed"],
            "normalized_predicted_advantages": [0.03, -0.01, 0.04],
            "refreshed_predicted_scores": [1.0, 0.9, 1.1],
            "fixed_predicted_scores": [0.8, 1.0, 0.7],
        },
        "geometry": {
            "gated_w0_to_refreshed_c_cosines": [
                {f"weight_{layer}": 1.0 - 0.05 * step for layer in range(5)}
                for step in range(4)
            ]
        },
        "budgets": {
            "native_c_energy": 16.0,
            "native_c_distance": 4.0,
            "hop_c_distance": 1.0,
            "per_hop_c_energy": 1.0,
            "probe_c_distance": 0.0625,
        },
        "compute": {
            "controlled_nfe": AGATE_CONTROLLED_NFE,
            "proposal_build_count": 8,
            "probe_panel_count": AGATE_PROBE_PANEL_COUNT,
            "wall_seconds": 12.0,
        },
    }


def _transform(track: str) -> dict:
    if track == TRACK_MEMIT:
        return {}
    return {
        family: {
            "projector_manifest_id": _sha(900 + offset),
            "right_norm_retention": {
                f"weight_{layer}": 0.5 + 0.01 * layer for layer in range(5)
            },
        }
        for offset, family in enumerate(("synchronous", "ordered"))
    }


def _feature(index: int, *, track: str = TRACK_MEMIT) -> dict:
    return {
        "case_id": str(index),
        "request_id": _sha(index + 100),
        "editor_track": track,
        "k": 4,
        "hop_fraction": 0.25,
        "native_c_energy": 16.0,
        "native_c_distance": 4.0,
        "per_hop_c_energy": 1.0,
        "hop_c_distance": 1.0,
        "probe_c_distance": 0.0625,
        "gate_relative_margin": GATE_RELATIVE_MARGIN,
        "target_identity": {},
        "w0_panel": {},
        "w0_layer_weights": {},
        "w0_transform": _transform(track),
        "paths": {},
        "compute_plan": {},
        "feature_hash": _sha(index + 500),
    }


class AdaptiveGateAnalysisTests(unittest.TestCase):
    def test_memit_lenient_model_gate_and_effects(self) -> None:
        result = analysis.analyze_case_records(
            [_case(index) for index in range(AGATE_CASE_COUNT)],
            [_feature(index) for index in range(AGATE_CASE_COUNT)],
            model_alias="llama3-8b-inst",
            track=TRACK_MEMIT,
        )
        self.assertTrue(result["technical_validity"]["pass"])
        self.assertTrue(result["lenient_gate"]["model_gate_pass"])
        self.assertEqual(result["verdict"], "LENIENT_MODEL_PASS")
        self.assertAlmostEqual(
            result["effects"]["gated_minus_refreshed_G4_minus_A4"]["mean"],
            0.1,
        )
        self.assertAlmostEqual(result["gate_behavior"]["refresh_rate"], 2.0 / 3.0)
        self.assertEqual(result["projector"]["right_norm_retention_count"], 0)

    def test_alpha_projector_retention_is_compact(self) -> None:
        result = analysis.analyze_case_records(
            [_case(index, track=TRACK_ALPHAEDIT) for index in range(AGATE_CASE_COUNT)],
            [_feature(index, track=TRACK_ALPHAEDIT) for index in range(AGATE_CASE_COUNT)],
            model_alias="llama3-8b-inst",
            track=TRACK_ALPHAEDIT,
        )
        self.assertTrue(result["lenient_gate"]["alpha_transfer"])
        self.assertEqual(
            result["projector"]["right_norm_retention_count"],
            AGATE_CASE_COUNT * 10,
        )
        self.assertAlmostEqual(result["projector"]["mean_right_norm_retention"], 0.52)

    def test_wrong_gate_margin_is_rejected(self) -> None:
        records = [_case(index) for index in range(AGATE_CASE_COUNT)]
        records[0] = copy.deepcopy(records[0])
        records[0]["gate"]["relative_margin"] = 0.03
        with self.assertRaises(analysis.AdaptiveGateAnalysisError):
            analysis.analyze_case_records(
                records,
                [_feature(index) for index in range(AGATE_CASE_COUNT)],
                model_alias="llama3-8b-inst",
                track=TRACK_MEMIT,
            )

    def test_stream_loader_rejects_wrong_event(self) -> None:
        wrapper = {
            "schema_version": STREAM_SCHEMA,
            "run_id": "run",
            "sequence": 0,
            "recorded_at": "2026-08-01T00:00:00+00:00",
            "event": "wrong",
            "payload": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stream.jsonl"
            path.write_text(json.dumps(wrapper) + "\n", encoding="utf-8")
            with self.assertRaises(analysis.AdaptiveGateAnalysisError):
                analysis.load_stream(path, expected_event=analysis.ANALYSIS_EVENT)


if __name__ == "__main__":
    unittest.main()
