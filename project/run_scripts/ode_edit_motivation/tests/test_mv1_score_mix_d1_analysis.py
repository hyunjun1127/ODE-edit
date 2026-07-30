import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from project.run_scripts.ode_edit_motivation.mv1_score_mix import (
    _ACTION_FIELDS as RUNNER_ACTION_FIELDS,
    _FEATURE_FIELDS as RUNNER_FEATURE_FIELDS,
)
from project.run_scripts.ode_edit_motivation.mv1_score_mix_analysis import (
    ScoreMixAnalysisError,
    _ACTION_FIELDS as SHARED_ACTION_FIELDS,
    _FEATURE_FIELDS as SHARED_FEATURE_FIELDS,
)
from project.run_scripts.ode_edit_motivation.mv1_score_mix_d1_analysis import (
    ANALYSIS_SCHEMA,
    D0_RUN_IDS,
    D1_RUN_IDS,
    SCORE_MIX_Q,
    SINGLE_ACTIONS,
    STATIC_POLICY_SCHEMA,
    analyze_d1_run,
    fit_frozen_static_policy_from_features,
    main,
    render_korean_markdown,
)
from project.run_scripts.ode_edit_motivation.tests.test_mv1_score_mix_analysis import (
    build_run as build_d0_run,
    canonical,
    digest_bytes,
    digest_file,
    metric_payload,
    selection_manifest,
    write_json,
)


STREAM_SCHEMA = "ode-edit-mv1-score-mix/v1"
RUN_ID = "mv1mix_llama_d1_v1"
OUTCOMES = (
    "score_mix",
    "uniform",
    "ordered_global_alpha",
    "native_memit_full",
    "no_op_replay",
)


def write_stream(path, rows):
    path.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": STREAM_SCHEMA,
                    "run_id": RUN_ID,
                    "sequence": sequence,
                    "recorded_at": "2026-07-31T00:00:00+00:00",
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


def build_d1_run(parent, *, omit_outcome=None):
    root = Path(parent) / RUN_ID
    root.mkdir(parents=True)
    receipt_root = root / "action_receipts"
    receipt_root.mkdir()
    selection = selection_manifest()
    calibration = tuple(selection["case_ids"]["calibration"])
    selected = calibration[8:20]
    requests = {
        case_id: digest_bytes(f"d1-request:{case_id}".encode())
        for case_id in selected
    }
    slurm = {
        "under_slurm": True,
        "job_id": "23456",
        "job_name": "odeedit_mv1mix_d1_pair_v1",
        "node": "devbox",
        "slice_label": "d1",
    }
    manifest = {
        "schema_version": "ode-edit-mv1-score-mix-manifest/v1",
        "run_id": RUN_ID,
        "ode_edit_git": {
            "commit": "b" * 40,
            "tracked_worktree_clean": True,
        },
        "slurm": slurm,
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
            "label": "d1",
            "start": 8,
            "count": 12,
            "pilot_case_count_excluded": 3,
        },
        "selected_case_ids": list(selected),
        "selected_request_ids": [requests[case] for case in selected],
        "q": SCORE_MIX_Q,
        "q_label": "q_1_256",
        "probe_ratio": 0.25,
        "probe_action_set": [*SINGLE_ACTIONS, "uniform"],
        "controller_input_set": list(SINGLE_ACTIONS),
        "outcome_action_set": list(OUTCOMES),
        "expected_counts": {
            "features": 12,
            "commitments": 12,
            "outcomes": 60,
            "receipts": 12,
        },
    }
    write_json(root / "manifest.json", manifest)

    feature_rows = []
    action_rows = []
    outcome_rows = []
    event_rows = []
    receipt_hashes = {}
    for index, case_id in enumerate(selected):
        request_id = requests[case_id]
        single_scores = {
            action: float(position + 1 + index / 20)
            for position, action in enumerate(SINGLE_ACTIONS)
        }
        uniform_score = sum(single_scores.values()) / (5.0**0.5)
        scores = {**single_scores, "uniform": uniform_score}
        norm = sum(value * value for value in single_scores.values()) ** 0.5
        weights = {
            action: value / norm for action, value in single_scores.items()
        }
        predicted = sum(
            weights[action] * single_scores[action] for action in SINGLE_ACTIONS
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

        gain = (index - 4) * 0.001
        progress = {
            "score_mix": 0.2 + gain,
            "uniform": 0.2,
            "ordered_global_alpha": 0.19,
            "native_memit_full": 0.6,
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
                "c_energy": 0.0 if is_replay else (1.0 if is_equal else 256.0),
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
                    "event_seed": 31 + index,
                    "target_token_count": 1,
                    "native_c_energy": 256.0,
                    "q": SCORE_MIX_Q,
                    "probe_direction_count": 6,
                    "feature_count": 1,
                    "commitment_count": 1,
                    "outcome_count": 5,
                    "direct_z_artifact_sha256": hashlib.sha256(
                        f"d1-activation:{case_id}".encode()
                    ).hexdigest(),
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
        "slice": {"label": "d1", "start": 8, "count": 12},
        "slurm": slurm,
        "selection_manifest_id": selection["manifest_id"],
        "run_status": "completed",
        "abort_failure_type": None,
        "planned_case_count": 12,
        "attempted_case_count": 12,
        "not_run_due_to_abort_count": 0,
        "pass_count": 12,
        "failure_count": 0,
        "all_pass": True,
        "q": SCORE_MIX_Q,
        "feature_count": 12,
        "commitment_count": 12,
        "outcome_count": observed_outcomes,
        "action_receipt_count": 12,
        "expected_counts_exact": observed_outcomes == 60,
        "all_rollbacks_exact": True,
        "projector_files_loaded": [],
        "artifacts": artifacts,
    }
    write_json(root / "summary.json", summary)
    return root


def static_feature(case_id, slopes):
    request_id = digest_bytes(f"static-request:{case_id}".encode())
    feature = {
        "case_id": case_id,
        "request_id": request_id,
        "q": SCORE_MIX_Q,
        "q_label": "q_1_256",
        "native_c_energy": 256.0,
        "operational_c_distance": 1.0,
        "probe_c_distance": 0.25,
        "action_scores": {
            **dict(zip(SINGLE_ACTIONS, slopes)),
            "uniform": sum(slopes) / (5.0**0.5),
        },
        "context_action_scores": {
            **{
                action: [value]
                for action, value in zip(SINGLE_ACTIONS, slopes)
            },
            "uniform": [sum(slopes) / (5.0**0.5)],
        },
        "feature_policy": (
            "same-snapshot/six-unit-c/central-fd/epsilon=d_over_4/"
            "temperature-1-mean-context-token"
        ),
    }
    feature["feature_hash"] = digest_bytes(canonical(feature).encode())
    return feature


class ScoreMixD1AnalysisTests(unittest.TestCase):
    def test_shared_analyzer_schema_matches_actual_runner(self):
        self.assertEqual(SHARED_FEATURE_FIELDS, RUNNER_FEATURE_FIELDS)
        self.assertEqual(SHARED_ACTION_FIELDS, RUNNER_ACTION_FIELDS)

    def test_valid_d1_exact_twelve_by_five_panel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = build_d1_run(temporary)
            report = analyze_d1_run(root)

        self.assertEqual(report["schema_version"], ANALYSIS_SCHEMA)
        self.assertEqual(report["analysis_status"], "d1_descriptive_complete")
        self.assertTrue(report["artifact_validation"]["valid"])
        self.assertEqual(
            report["artifact_validation"]["observed_counts"]["outcomes"], 60
        )
        self.assertEqual(len(report["case_diagnostics"]), 12)
        self.assertEqual(
            report["descriptive_d1_effect"]["g_mix"]["finite_case_count"], 12
        )

    def test_receipt_tamper_is_technical_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = build_d1_run(temporary)
            receipt = next((root / "action_receipts").glob("*.json"))
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["durability"] = "tampered"
            write_json(receipt, payload)
            report = analyze_d1_run(root)

        self.assertEqual(
            report["analysis_status"], "technical_block_invalid_artifacts"
        )
        self.assertTrue(
            any(
                code.startswith("receipt_durability_mismatch")
                for code in report["artifact_validation"]["error_codes"]
            )
        )

    def test_slice_overlap_and_missing_outcome_are_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            selection = selection_manifest()
            missing_case = selection["case_ids"]["calibration"][19]
            root = build_d1_run(
                temporary,
                omit_outcome=(missing_case, "ordered_global_alpha"),
            )
            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            manifest["selected_slice"]["start"] = 7
            manifest["selected_case_ids"][0] = manifest["selection"]["case_ids"][
                "calibration"
            ][7]
            write_json(root / "manifest.json", manifest)
            report = analyze_d1_run(root)

        errors = report["artifact_validation"]["error_codes"]
        self.assertIn("manifest_slice_mismatch", errors)
        self.assertIn("selected_slice_ids_mismatch", errors)
        self.assertIn("pilot_or_d0_selection_overlap", errors)
        self.assertIn("observed_outcome_count_mismatch", errors)
        self.assertIn("outcome_case_action_panel_mismatch", errors)

    def test_static_weights_known_answer_and_outcome_mutation_invariance(self):
        slopes = (3.0, 4.0, -2.0, 0.0, -1.0)
        d0 = [static_feature(f"d0-{index}", slopes) for index in range(5)]
        d1 = [static_feature(f"d1-{index}", slopes) for index in range(12)]
        outcomes = [{"progress": 1.0}, {"progress": -2.0}]

        before = fit_frozen_static_policy_from_features(
            model_alias="llama3-8b-inst",
            d0_run_id=D0_RUN_IDS["llama3-8b-inst"],
            d1_run_id=D1_RUN_IDS["llama3-8b-inst"],
            d0_features=d0,
            d1_features=d1,
        )
        outcomes[0]["progress"] = 1e9
        outcomes.append({"progress": -1e9})
        after = fit_frozen_static_policy_from_features(
            model_alias="llama3-8b-inst",
            d0_run_id=D0_RUN_IDS["llama3-8b-inst"],
            d1_run_id=D1_RUN_IDS["llama3-8b-inst"],
            d0_features=d0,
            d1_features=d1,
        )

        self.assertEqual(before, after)
        self.assertEqual(before["schema_version"], STATIC_POLICY_SCHEMA)
        self.assertAlmostEqual(before["layer_weights"]["layer_4"], 0.6)
        self.assertAlmostEqual(before["layer_weights"]["layer_5"], 0.8)
        self.assertEqual(before["layer_weights"]["layer_6"], 0.0)
        self.assertEqual(before["outcome_fields_used"], [])
        self.assertNotIn(
            "outcomes",
            inspect.signature(
                fit_frozen_static_policy_from_features
            ).parameters,
        )
        poisoned_d0 = [dict(feature) for feature in d0]
        poisoned_d0[0]["progress"] = 999.0
        with self.assertRaises(ScoreMixAnalysisError):
            fit_frozen_static_policy_from_features(
                model_alias="llama3-8b-inst",
                d0_run_id=D0_RUN_IDS["llama3-8b-inst"],
                d1_run_id=D1_RUN_IDS["llama3-8b-inst"],
                d0_features=poisoned_d0,
                d1_features=d1,
            )

        tied_slopes = (-1.0, -1.0, -2.0, -3.0, -4.0)
        tied_d0 = [
            static_feature(f"tie-d0-{index}", tied_slopes)
            for index in range(5)
        ]
        tied_d1 = [
            static_feature(f"tie-d1-{index}", tied_slopes)
            for index in range(12)
        ]
        tied = fit_frozen_static_policy_from_features(
            model_alias="llama3-8b-inst",
            d0_run_id=D0_RUN_IDS["llama3-8b-inst"],
            d1_run_id=D1_RUN_IDS["llama3-8b-inst"],
            d0_features=tied_d0,
            d1_features=tied_d1,
        )
        self.assertEqual(
            tied["controller_branch"], "all-nonpositive-max-onehot"
        )
        self.assertEqual(tied["layer_weights"]["layer_4"], 1.0)
        self.assertEqual(tied["layer_weights"]["layer_5"], 0.0)

    def test_cli_writes_report_summary_and_frozen_policy_without_overclaim(self):
        with tempfile.TemporaryDirectory() as temporary:
            d0_root = build_d0_run(Path(temporary) / "d0")
            d1_root = build_d1_run(Path(temporary) / "d1")
            report_path = Path(temporary) / "report.md"
            summary_path = Path(temporary) / "summary.json"
            policy_path = Path(temporary) / "policy.json"
            exit_code = main(
                [
                    str(d1_root),
                    "--d0-run-directory",
                    str(d0_root),
                    "--report-output",
                    str(report_path),
                    "--summary-output",
                    str(summary_path),
                    "--static-policy-output",
                    str(policy_path),
                ]
            )
            markdown = report_path.read_text(encoding="utf-8")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            policy = json.loads(policy_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertIn("D1 calibration", markdown)
        self.assertIn("outcome field 사용: 없음", markdown)
        self.assertIn("허용하지 않는다", markdown)
        self.assertFalse(summary["pair_level_decision_computed"])
        self.assertEqual(policy["feature_case_count"], 17)
        self.assertEqual(policy["outcome_fields_used"], [])
        serialized = json.dumps(
            {"summary": summary, "policy": policy},
            sort_keys=True,
        )
        for raw_key in (
            '"prompt"',
            '"target_new"',
            '"logits"',
            '"progress"',
            "direct_z",
        ):
            self.assertNotIn(raw_key, serialized)


if __name__ == "__main__":
    unittest.main()
