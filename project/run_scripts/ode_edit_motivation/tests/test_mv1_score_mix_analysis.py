import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from project.run_scripts.ode_edit_motivation.mv1_score_mix_analysis import (
    ANALYSIS_SCHEMA,
    NUMERIC_COMPARISON_TOLERANCE,
    SCORE_MIX_Q,
    _ACTION_FIELDS as ANALYZER_ACTION_FIELDS,
    _FEATURE_FIELDS as ANALYZER_FEATURE_FIELDS,
    analyze_score_mix_run,
    build_small_summary,
    main,
    render_korean_markdown,
)
from project.run_scripts.ode_edit_motivation.mv1_score_mix import (
    _ACTION_FIELDS as RUNNER_ACTION_FIELDS,
    _FEATURE_FIELDS as RUNNER_FEATURE_FIELDS,
)


STREAM_SCHEMA = "ode-edit-mv1-score-mix/v1"
RUN_ID = "mv1mix_llama_d0_v1"
CASES = tuple(f"case-{index}" for index in range(100))
SELECTED = CASES[3:8]
OUTCOMES = (
    "score_mix",
    "uniform",
    "ordered_global_alpha",
    "native_memit_full",
    "no_op_replay",
)
SINGLES = tuple(f"layer_{layer}" for layer in range(4, 9))


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


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_stream(path, rows):
    path.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": STREAM_SCHEMA,
                    "run_id": RUN_ID,
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


def selection_manifest():
    calibration = list(CASES[:20])
    confirmatory = list(CASES[20:80])
    untouched = list(CASES[80:100])
    payload = {
        "schema_version": "ode-edit-counterfact-selection/v1",
        "seed": "ode-edit-motivation-counterfact-v1",
        "source": {
            "sha256": digest_bytes(b"counterfact"),
            "size": 100,
            "row_count": 100,
        },
        "case_ids": {
            "calibration": calibration,
            "confirmatory": confirmatory,
            "untouched": untouched,
        },
        "order_hash": digest_bytes(canonical(CASES).encode()),
        "split_hash": digest_bytes(
            canonical(
                {
                    "calibration": calibration,
                    "confirmatory": confirmatory,
                    "untouched": untouched,
                }
            ).encode()
        ),
    }
    payload["manifest_id"] = digest_bytes(canonical(payload).encode())
    return payload


def metric_payload(progress):
    return {
        "progress": progress,
        "context_progress": [progress],
        "nll_reduction": progress / 2,
        "context_nll_reduction": [progress / 2],
        "exact_margin_min": progress,
        "exact_satisfied": False,
    }


def build_run(parent, *, omit_outcome=None, score_mix_energy_offset=0.0):
    root = Path(parent) / RUN_ID
    root.mkdir(parents=True)
    receipt_root = root / "action_receipts"
    receipt_root.mkdir()
    requests = {
        case_id: digest_bytes(f"request:{case_id}".encode()) for case_id in SELECTED
    }
    selection = selection_manifest()
    manifest = {
        "schema_version": "ode-edit-mv1-score-mix-manifest/v1",
        "run_id": RUN_ID,
        "ode_edit_git": {
            "commit": "a" * 40,
            "tracked_worktree_clean": True,
        },
        "slurm": {
            "under_slurm": True,
            "job_id": "12345",
            "job_name": "odeedit_mv1mix_d0_pair_v1",
            "node": "devbox",
            "slice_label": "d0",
        },
        "model": {
            "model_alias": "llama3-8b-inst",
            "repository_id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "revision": "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
            "dtype": "torch.float32",
            "device": "cuda:0",
            "offline": True,
        },
        "selection": selection,
        "selected_split": "calibration",
        "selected_slice": {
            "label": "d0",
            "start": 3,
            "count": 5,
            "pilot_case_count_excluded": 3,
        },
        "selected_case_ids": list(SELECTED),
        "selected_request_ids": [requests[case] for case in SELECTED],
        "q": SCORE_MIX_Q,
        "q_label": "q_1_256",
        "probe_ratio": 0.25,
        "probe_action_set": [*SINGLES, "uniform"],
        "controller_input_set": list(SINGLES),
        "outcome_action_set": list(OUTCOMES),
        "expected_counts": {
            "features": 5,
            "commitments": 5,
            "outcomes": 25,
            "receipts": 5,
        },
    }
    write_json(root / "manifest.json", manifest)

    feature_rows = []
    action_rows = []
    outcome_rows = []
    event_rows = []
    receipt_hashes = {}
    # e_m=0; case-0 is within only because 5e-13 < the locked 1e-12
    # numerical tolerance, while case-1 at 2e-12 is above it.
    gains = (5e-13, 2e-12, -1e-4, 0.0, -2e-4)
    for index, (case_id, gain) in enumerate(zip(SELECTED, gains)):
        request_id = requests[case_id]
        single_scores = {
            action: float(position + 1 + index / 10)
            for position, action in enumerate(SINGLES)
        }
        uniform_score = sum(single_scores.values()) / (5.0**0.5)
        scores = {**single_scores, "uniform": uniform_score}
        weights = {
            action: value
            / sum(item * item for item in single_scores.values()) ** 0.5
            for action, value in single_scores.items()
        }
        predicted = sum(
            weights[action] * single_scores[action] for action in SINGLES
        )
        feature = {
            "case_id": case_id,
            "request_id": request_id,
            "q": SCORE_MIX_Q,
            "q_label": "q_1_256",
            "native_c_energy": 256.0,
            "operational_c_distance": 1.0,
            "probe_c_distance": 0.25,
            "action_scores": scores,
            "context_action_scores": {
                action: [value] for action, value in scores.items()
            },
            "feature_policy": (
                "same-snapshot/six-unit-c/central-fd/epsilon=d_over_4/"
                "temperature-1-mean-context-token"
            ),
        }
        feature["feature_hash"] = digest_bytes(canonical(feature).encode())
        action = {
            "case_id": case_id,
            "request_id": request_id,
            "q": SCORE_MIX_Q,
            "q_label": "q_1_256",
            "feature_hash": feature["feature_hash"],
            "action_id": "score_mix",
            "layer_weights": weights,
            "predicted_score": predicted,
            "actual_unit_c_energy": 1.0,
            "controller": "five-single-slopes/relu-l2-else-max-onehot",
            "controller_branch": "positive-relu-l2",
            "tie_policy": "exact-tie-lower-layer",
        }
        action["commitment_hash"] = digest_bytes(canonical(action).encode())
        feature_rows.append(("mv1mix_feature", feature))
        action_rows.append(("mv1mix_action_commitment", action))

        receipt = {
            "schema_version": "ode-edit-mv1-score-mix-receipt/v1",
            "case_id": case_id,
            "request_id": request_id,
            "q": SCORE_MIX_Q,
            "feature_hash": feature["feature_hash"],
            "commitment_hash": action["commitment_hash"],
            "action_id": "score_mix",
            "durability": (
                "feature-and-dynamic-action-write+flush+fsync-before-exclusive-receipt"
            ),
            "outcomes_observed_before_commitment": False,
        }
        receipt_identity = {
            "case_id": case_id,
            "request_id": request_id,
            "q": SCORE_MIX_Q,
            "feature_hash": feature["feature_hash"],
            "commitment_hash": action["commitment_hash"],
        }
        receipt_name = digest_bytes(canonical(receipt_identity).encode()) + ".json"
        receipt_path = receipt_root / receipt_name
        write_json(receipt_path, receipt)
        receipt_hash = digest_file(receipt_path)
        receipt_hashes[receipt_name] = receipt_hash

        progress = {
            "score_mix": 0.1 + gain,
            "uniform": 0.1,
            "ordered_global_alpha": 0.09,
            "native_memit_full": 0.5,
            "no_op_replay": 0.0,
        }
        for outcome_action in OUTCOMES:
            if omit_outcome == (case_id, outcome_action):
                continue
            is_replay = outcome_action == "no_op_replay"
            is_equal = outcome_action in {
                "score_mix",
                "uniform",
                "ordered_global_alpha",
            }
            payload = {
                "case_id": case_id,
                "request_id": request_id,
                "q": SCORE_MIX_Q,
                "q_label": "q_1_256",
                "feature_hash": feature["feature_hash"],
                "commitment_hash": action["commitment_hash"],
                "status": "completed",
                "rollback_exact": True,
                "action_id": outcome_action,
                "selected_by_controller": outcome_action == "score_mix",
                "c_distance": 0.0 if is_replay else (1.0 if is_equal else 16.0),
                "c_energy": (
                    0.0
                    if is_replay
                    else (
                        1.0 + score_mix_energy_offset
                        if outcome_action == "score_mix"
                        else 1.0
                    )
                    if is_equal
                    else 256.0
                ),
                "budget_validation": (
                    "reference-no-op"
                    if is_replay
                    else "unit-c-remeasured-then-scaled/equal-c"
                    if outcome_action in {"score_mix", "uniform"}
                    else (
                        "native-c-remeasured-then-global-alpha-squared/equal-c"
                    )
                    if outcome_action == "ordered_global_alpha"
                    else "reference-native-full"
                ),
                **metric_payload(progress[outcome_action]),
            }
            if is_replay:
                payload["logits_hash_equal"] = True
            outcome_rows.append(("mv1mix_outcome", payload))
        event_rows.append(
            (
                "mv1mix_case",
                {
                    "case_id": case_id,
                    "request_id": request_id,
                    "event_seed": 17 + index,
                    "target_token_count": 1,
                    "native_c_energy": 256.0,
                    "q": SCORE_MIX_Q,
                    "probe_direction_count": 6,
                    "feature_count": 1,
                    "commitment_count": 1,
                    "outcome_count": 5,
                    "direct_z_artifact_sha256": digest_bytes(
                        f"activation:{case_id}".encode()
                    ),
                    "direct_z_artifact_size": 100,
                    "action_receipt": {
                        "name": receipt_name,
                        "sha256": receipt_hash,
                    },
                    "rollback_exact": True,
                    "pass": True,
                },
            )
        )

    write_stream(root / "features.jsonl", feature_rows)
    write_stream(root / "actions.jsonl", action_rows)
    write_stream(root / "outcomes.jsonl", outcome_rows)
    write_stream(root / "events.jsonl", event_rows)
    artifacts = {
        "manifest_sha256": digest_file(root / "manifest.json"),
        "features_sha256": digest_file(root / "features.jsonl"),
        "actions_sha256": digest_file(root / "actions.jsonl"),
        "outcomes_sha256": digest_file(root / "outcomes.jsonl"),
        "events_sha256": digest_file(root / "events.jsonl"),
        "action_receipts": receipt_hashes,
    }
    observed_outcomes = len(outcome_rows)
    summary = {
        "schema_version": "ode-edit-mv1-score-mix-summary/v1",
        "run_id": RUN_ID,
        "model_alias": "llama3-8b-inst",
        "slice": {"label": "d0", "start": 3, "count": 5},
        "slurm": manifest["slurm"],
        "selection_manifest_id": selection["manifest_id"],
        "run_status": "completed",
        "abort_failure_type": None,
        "planned_case_count": 5,
        "attempted_case_count": 5,
        "not_run_due_to_abort_count": 0,
        "pass_count": 5,
        "failure_count": 0,
        "all_pass": True,
        "q": SCORE_MIX_Q,
        "feature_count": 5,
        "commitment_count": 5,
        "outcome_count": observed_outcomes,
        "action_receipt_count": 5,
        "expected_counts_exact": observed_outcomes == 25,
        "all_rollbacks_exact": True,
        "projector_files_loaded": [],
        "artifacts": artifacts,
    }
    write_json(root / "summary.json", summary)
    return root


class ScoreMixAnalysisTests(unittest.TestCase):
    def test_analyzer_feature_and_action_schema_match_runner_exactly(self):
        self.assertEqual(ANALYZER_FEATURE_FIELDS, RUNNER_FEATURE_FIELDS)
        self.assertEqual(ANALYZER_ACTION_FIELDS, RUNNER_ACTION_FIELDS)

    def test_valid_run_and_numeric_gain_tolerance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = build_run(temporary)
            report = analyze_score_mix_run(root)

        self.assertEqual(report["schema_version"], ANALYSIS_SCHEMA)
        self.assertEqual(report["analysis_status"], "d0_descriptive_complete")
        self.assertTrue(report["artifact_validation"]["valid"])
        self.assertEqual(
            report["replay_envelope"]["value"],
            NUMERIC_COMPARISON_TOLERANCE,
        )
        self.assertEqual(
            report["replay_envelope"]["numeric_floor"],
            NUMERIC_COMPARISON_TOLERANCE,
        )
        self.assertTrue(
            report["case_diagnostics"][0][
                "g_mix_leq_envelope_with_numeric_tolerance"
            ]
        )
        self.assertFalse(
            report["case_diagnostics"][1][
                "g_mix_leq_envelope_with_numeric_tolerance"
            ]
        )
        self.assertFalse(
            report["model_early_kill_input"][
                "all_five_g_mix_leq_model_replay_envelope"
            ]
        )

    def test_tampered_receipt_and_hash_are_technical_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = build_run(temporary)
            receipt = next((root / "action_receipts").glob("*.json"))
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["durability"] = "tampered"
            write_json(receipt, payload)
            report = analyze_score_mix_run(root)

        self.assertEqual(
            report["analysis_status"], "technical_block_invalid_artifacts"
        )
        errors = report["artifact_validation"]["error_codes"]
        self.assertTrue(any("receipt_durability_mismatch" in item for item in errors))
        self.assertIn("summary_receipt_hash_map_mismatch", errors)

    def test_tampered_feature_hash_chain_is_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = build_run(temporary)
            rows = (root / "features.jsonl").read_text(encoding="utf-8").splitlines()
            first = json.loads(rows[0])
            first["payload"]["feature_hash"] = "0" * 64
            rows[0] = json.dumps(first, sort_keys=True)
            (root / "features.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
            report = analyze_score_mix_run(root)

        self.assertEqual(
            report["analysis_status"], "technical_block_invalid_artifacts"
        )
        self.assertTrue(
            any(
                code.startswith("feature_hash_mismatch")
                for code in report["artifact_validation"]["error_codes"]
            )
        )

    def test_action_unit_c_and_arm_specific_budget_validation_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = build_run(temporary)
            action_rows = (
                root / "actions.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            first_action = json.loads(action_rows[0])
            first_action["payload"]["actual_unit_c_energy"] = 1.1
            action_rows[0] = json.dumps(first_action, sort_keys=True)
            (root / "actions.jsonl").write_text(
                "\n".join(action_rows) + "\n",
                encoding="utf-8",
            )
            outcome_rows = (
                root / "outcomes.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            ordered_index = next(
                index
                for index, row in enumerate(outcome_rows)
                if json.loads(row)["payload"]["action_id"]
                == "ordered_global_alpha"
            )
            ordered = json.loads(outcome_rows[ordered_index])
            ordered["payload"]["budget_validation"] = (
                "unit-c-remeasured-then-scaled/equal-c"
            )
            outcome_rows[ordered_index] = json.dumps(ordered, sort_keys=True)
            (root / "outcomes.jsonl").write_text(
                "\n".join(outcome_rows) + "\n",
                encoding="utf-8",
            )
            report = analyze_score_mix_run(root)

        errors = report["artifact_validation"]["error_codes"]
        self.assertTrue(
            any(
                code.startswith("action_unit_c_energy_mismatch")
                for code in errors
            )
        )
        self.assertTrue(
            any(
                code.startswith("budget_validation_mismatch")
                for code in errors
            )
        )

    def test_equal_c_energy_uses_runner_tolerance_against_locked_distance(self):
        with tempfile.TemporaryDirectory() as temporary:
            within_root = build_run(
                Path(temporary) / "within",
                score_mix_energy_offset=2.9e-5,
            )
            within_report = analyze_score_mix_run(within_root)
            outside_root = build_run(
                Path(temporary) / "outside",
                score_mix_energy_offset=3.1e-5,
            )
            outside_report = analyze_score_mix_run(outside_root)

        self.assertEqual(
            within_report["analysis_status"],
            "d0_descriptive_complete",
        )
        self.assertFalse(
            any(
                code.startswith("equal_c_energy_mismatch")
                for code in within_report["artifact_validation"]["error_codes"]
            )
        )
        self.assertEqual(
            outside_report["analysis_status"],
            "technical_block_invalid_artifacts",
        )
        self.assertTrue(
            any(
                code.startswith(
                    f"equal_c_energy_mismatch:{SELECTED[0]}:score_mix"
                )
                for code in outside_report["artifact_validation"]["error_codes"]
            )
        )
    def test_outcome_count_and_slice_tamper_are_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = build_run(
                temporary, omit_outcome=(SELECTED[-1], "ordered_global_alpha")
            )
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            manifest["selected_slice"]["start"] = 2
            write_json(root / "manifest.json", manifest)
            report = analyze_score_mix_run(root)

        errors = report["artifact_validation"]["error_codes"]
        self.assertIn("manifest_slice_mismatch", errors)
        self.assertIn("observed_outcome_count_mismatch", errors)
        self.assertIn("outcome_case_action_panel_mismatch", errors)
        self.assertIsNone(
            report["model_early_kill_input"][
                "all_five_g_mix_leq_model_replay_envelope"
            ]
        )

    def test_markdown_and_small_summary_do_not_overclaim_or_emit_activation_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = build_run(temporary)
            report = analyze_score_mix_run(root)
            markdown = render_korean_markdown(report)
            summary = build_small_summary(report)

        self.assertIn("paired D1만 허용", markdown)
        self.assertIn("pair-level early kill은 이 문서에서 계산하지 않는다", markdown)
        self.assertFalse(summary["pair_level_early_kill_computed"])
        serialized = json.dumps(summary, sort_keys=True)
        self.assertNotIn("direct_z", serialized)
        self.assertNotIn("artifact_sha256", serialized)
        self.assertNotIn('"prompt"', serialized)

    def test_cli_writes_korean_markdown_and_small_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = build_run(temporary)
            report_path = Path(temporary) / "report.md"
            summary_path = Path(temporary) / "report.json"
            exit_code = main(
                [
                    str(root),
                    "--report-output",
                    str(report_path),
                    "--summary-output",
                    str(summary_path),
                ]
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            markdown = report_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn("독립 분석", markdown)
        self.assertEqual(summary["analysis_status"], "d0_descriptive_complete")


if __name__ == "__main__":
    unittest.main()
