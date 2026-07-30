import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from project.run_scripts.ode_edit_motivation.mv1_pilot_analysis import (
    ANALYSIS_SCHEMA,
    analyze_pilot_run,
)


STREAM_SCHEMA = "ode-edit-mv1-c0/v1"
FRACTIONS = (1.0 / 256.0, 1.0 / 64.0, 1.0 / 16.0)
ACTIONS = ("layer_4", "layer_5", "uniform")
CASES = ("case-0", "case-1", "case-2")


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_file(path):
    return digest_bytes(path.read_bytes())


def fraction_label(value):
    return f"q_1_{round(1.0 / value)}"


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_stream(path, run_id, rows):
    path.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": STREAM_SCHEMA,
                    "run_id": run_id,
                    "sequence": sequence,
                    "recorded_at": "2026-07-30T00:00:00+00:00",
                    "event": event,
                    "payload": payload,
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
            for sequence, (event, payload) in enumerate(rows)
        ),
        encoding="utf-8",
    )


def build_run(parent, *, omit_cell=None):
    run_id = "mv1_synthetic_c0p_v1"
    root = Path(parent) / run_id
    root.mkdir()
    receipt_root = root / "action_receipts"
    receipt_root.mkdir()
    request_ids = {
        case: digest_bytes(f"request:{case}".encode()) for case in CASES
    }
    manifest = {
        "schema_version": "ode-edit-mv1-c0-manifest/v1",
        "run_id": run_id,
        "selected_split": "calibration",
        "selected_case_ids": list(CASES),
        "selected_request_ids": [request_ids[case] for case in CASES],
        "fractions": list(FRACTIONS),
        "action_set": list(ACTIONS),
    }
    write_json(root / "manifest.json", manifest)

    feature_rows = []
    action_rows = []
    outcome_rows = []
    event_rows = []
    action_receipt_hashes = {}
    slopes = {
        "case-0": {"layer_4": 1.00, "layer_5": 0.40, "uniform": 0.70},
        "case-1": {"layer_4": 0.50, "layer_5": 1.10, "uniform": 0.70},
        "case-2": {"layer_4": 0.80, "layer_5": 0.60, "uniform": 0.75},
    }
    for case_index, case_id in enumerate(CASES):
        receipts_for_event = {}
        for fraction in FRACTIONS:
            label = fraction_label(fraction)
            distance = (fraction * 256.0) ** 0.5
            scores = {
                action_id: value * (1.0 + 0.02 * (case_index - 1))
                for action_id, value in slopes[case_id].items()
            }
            feature = {
                "case_id": case_id,
                "request_id": request_ids[case_id],
                "fraction": fraction,
                "fraction_label": label,
                "native_c_energy": 256.0,
                "operational_c_distance": distance,
                "probe_c_distance": distance / 4.0,
                "action_scores": scores,
                "context_action_scores": {
                    action_id: [value] for action_id, value in scores.items()
                },
                "layer_factor_c_norms": {"layer_4": 2.0, "layer_5": 3.0},
                "feature_policy": (
                    "inference-only-central-fd/temperature-1/mean-context-token"
                ),
            }
            feature["feature_hash"] = digest_bytes(canonical(feature).encode())
            selected = max(scores, key=scores.get)
            commitment = {
                "case_id": case_id,
                "request_id": request_ids[case_id],
                "fraction": fraction,
                "fraction_label": label,
                "feature_hash": feature["feature_hash"],
                "action_id": selected,
                "controller": "finite-action-argmax/static-fallback-uniform",
                "near_tie_tolerance": 0.0,
            }
            commitment["commitment_hash"] = digest_bytes(
                canonical(commitment).encode()
            )
            feature_rows.append(("mv1_feature", feature))
            action_rows.append(("mv1_action_commitment", commitment))

            receipt = {
                "schema_version": "ode-edit-mv1-action-receipt/v1",
                "case_id": case_id,
                "request_id": request_ids[case_id],
                "entries": [
                    {
                        "case_id": case_id,
                        "request_id": request_ids[case_id],
                        "fraction": fraction,
                        "feature_hash": feature["feature_hash"],
                        "commitment_hash": commitment["commitment_hash"],
                        "action_id": selected,
                    }
                ],
                "durability": (
                    "features-and-actions-flush+fsync-before-exclusive-receipt"
                ),
            }
            receipt_identity = {
                "case_id": case_id,
                "request_id": request_ids[case_id],
                "fractions": [fraction],
            }
            receipt_name = digest_bytes(
                canonical(receipt_identity).encode()
            ) + ".json"
            receipt_path = receipt_root / receipt_name
            write_json(receipt_path, receipt)
            receipt_sha = digest_file(receipt_path)
            action_receipt_hashes[receipt_name] = receipt_sha
            receipts_for_event[label] = {
                "name": receipt_name,
                "sha256": receipt_sha,
            }

            for action_id in ACTIONS:
                if omit_cell == (case_id, label, action_id):
                    continue
                progress = slopes[case_id][action_id] * distance
                outcome_rows.append(
                    (
                        "mv1_operational_outcome",
                        {
                            "case_id": case_id,
                            "request_id": request_ids[case_id],
                            "fraction": fraction,
                            "fraction_label": label,
                            "action_id": action_id,
                            "feature_hash": feature["feature_hash"],
                            "commitment_hash": commitment["commitment_hash"],
                            "selected_by_controller": action_id == selected,
                            "operational_c_distance": distance,
                            "operational_c_energy": distance * distance,
                            "budget_validation": "scaled-unit-c",
                            "progress": progress,
                            "status": "completed",
                            "rollback_exact": True,
                        },
                    )
                )
            outcome_rows.append(
                (
                    "mv1_contextual_outcome",
                    {
                        "case_id": case_id,
                        "request_id": request_ids[case_id],
                        "fraction": fraction,
                        "fraction_label": label,
                        "action_id": "ordered_global_alpha",
                        "commitment_hash": commitment["commitment_hash"],
                        "operational_c_distance": distance,
                        "operational_c_energy": distance * distance,
                        "budget_validation": (
                            "ordered-energy-times-global-alpha-squared"
                        ),
                        "progress": 0.65 * distance,
                        "status": "completed",
                        "rollback_exact": True,
                    },
                )
            )
        outcome_rows.extend(
            (
                (
                    "mv1_reference_outcome",
                    {
                        "case_id": case_id,
                        "request_id": request_ids[case_id],
                        "action_id": "native_memit_full",
                        "c_distance": 16.0,
                        "progress": 2.0,
                        "status": "completed",
                        "rollback_exact": True,
                    },
                ),
                (
                    "mv1_replay_outcome",
                    {
                        "case_id": case_id,
                        "request_id": request_ids[case_id],
                        "action_id": "no_op_replay",
                        "c_distance": 0.0,
                        "progress": 0.001,
                        "status": "completed",
                        "rollback_exact": True,
                    },
                ),
            )
        )
        event_rows.append(
            (
                "mv1_case",
                {
                    "case_id": case_id,
                    "request_id": request_ids[case_id],
                    "feature_count": len(FRACTIONS),
                    "commitment_count": len(FRACTIONS),
                    "outcome_count": (
                        len(FRACTIONS) * (len(ACTIONS) + 1) + 2
                    ),
                    "action_receipts": receipts_for_event,
                    "rollback_exact": True,
                    "pass": True,
                },
            )
        )

    write_stream(root / "features.jsonl", run_id, feature_rows)
    write_stream(root / "actions.jsonl", run_id, action_rows)
    write_stream(root / "outcomes.jsonl", run_id, outcome_rows)
    write_stream(root / "events.jsonl", run_id, event_rows)
    artifacts = {
        "manifest_sha256": digest_file(root / "manifest.json"),
        "features_sha256": digest_file(root / "features.jsonl"),
        "actions_sha256": digest_file(root / "actions.jsonl"),
        "outcomes_sha256": digest_file(root / "outcomes.jsonl"),
        "events_sha256": digest_file(root / "events.jsonl"),
        "action_receipts": action_receipt_hashes,
    }
    summary = {
        "schema_version": "ode-edit-mv1-c0-summary/v1",
        "run_id": run_id,
        "model_alias": "synthetic-model",
        "run_status": "completed",
        "planned_case_count": len(CASES),
        "attempted_case_count": len(CASES),
        "not_run_due_to_abort_count": 0,
        "pass_count": len(CASES),
        "failure_count": 0,
        "all_pass": True,
        "feature_count": len(feature_rows),
        "commitment_count": len(action_rows),
        "outcome_count": len(outcome_rows),
        "action_receipt_count": len(action_receipt_hashes),
        "artifacts": artifacts,
    }
    write_json(root / "summary.json", summary)
    return root


class MV1PilotAnalysisTests(unittest.TestCase):
    def test_complete_pilot_validates_artifacts_and_reports_descriptive_bounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = build_run(temporary)

            report = analyze_pilot_run(root)

        self.assertEqual(report["schema_version"], ANALYSIS_SCHEMA)
        self.assertEqual(report["claim_status"], "pilot_descriptive_only")
        self.assertEqual(report["analysis_status"], "descriptive_complete")
        self.assertTrue(report["artifact_validation"]["valid"])
        self.assertTrue(
            report["artifact_validation"]["commit_before_outcome"]["valid"]
        )
        self.assertEqual(
            report["pilot_budget_lock_diagnostic"]["largest_passing_fraction"],
            FRACTIONS[-1],
        )
        first = report["fraction_diagnostics"][0]
        derivative = first["derivative_validation"]
        self.assertEqual(derivative["expected_comparison_count"], 9)
        self.assertEqual(derivative["finite_comparison_count"], 9)
        self.assertGreaterEqual(derivative["sign_concordance_itd"], 0.8)
        self.assertLessEqual(derivative["median_relative_error"], 0.25)
        effects = first["preliminary_case_level_effects"]
        self.assertGreater(effects["controller_minus_uniform"]["mean"], 0.0)
        self.assertEqual(
            effects[
                "retrospective_oracle_minus_calibration_best_static"
            ]["planned_case_count"],
            3,
        )
        json.dumps(report, allow_nan=False)
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn('"prompt"', serialized)
        self.assertNotIn('"target_new"', serialized)
        self.assertNotIn('"logits"', serialized)

    def test_missing_outcome_keeps_full_denominator_and_blocks_fraction(self):
        missing = ("case-2", "q_1_16", "layer_5")
        with tempfile.TemporaryDirectory() as temporary:
            root = build_run(temporary, omit_cell=missing)

            report = analyze_pilot_run(root)

        self.assertEqual(report["analysis_status"], "blocked_incomplete_panel")
        diagnostic = report["fraction_diagnostics"][-1]["derivative_validation"]
        self.assertEqual(diagnostic["expected_comparison_count"], 9)
        self.assertEqual(diagnostic["finite_comparison_count"], 8)
        self.assertEqual(diagnostic["failure_or_missing_count"], 1)
        self.assertFalse(diagnostic["locked_criterion_pass"])
        oracle = report["fraction_diagnostics"][-1][
            "preliminary_case_level_effects"
        ]["retrospective_oracle_minus_calibration_best_static"]
        self.assertEqual(oracle["planned_case_count"], 3)
        self.assertGreater(oracle["failure_or_missing_case_count"], 0)

    def test_post_summary_tamper_is_reported_as_invalid_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = build_run(temporary)
            receipt = next((root / "action_receipts").glob("*.json"))
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["durability"] = "tampered"
            write_json(receipt, payload)

            report = analyze_pilot_run(root)

        self.assertEqual(report["analysis_status"], "blocked_invalid_artifacts")
        self.assertFalse(report["artifact_validation"]["valid"])
        self.assertIn(
            "receipt_durability_mismatch",
            report["artifact_validation"]["error_codes"],
        )
        self.assertIn(
            "summary_receipt_hash_map_mismatch",
            report["artifact_validation"]["error_codes"],
        )


if __name__ == "__main__":
    unittest.main()
