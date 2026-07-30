from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from project.run_scripts.ode_edit_motivation.mv1_score_mix_analysis import (
    MODEL_ENVELOPES,
    PROBE_ACTIONS,
    SCORE_MIX_Q,
    SINGLE_ACTIONS,
    _canonical_json,
    _expected_mix,
    _selection_manifest_id,
    _sha256_bytes,
    _sha256_file,
)
from project.run_scripts.ode_edit_motivation.mv1_score_mix_calibration_forecast import (
    CALIBRATION_METHOD,
    FORECAST_ANALYSIS_SCHEMA,
    FORECAST_POLICY_SCHEMA,
)
from project.run_scripts.ode_edit_motivation.mv1_score_mix_confirmatory_analysis import (
    ADAPTIVE_ACTION,
    CONFIRMATORY_ACTION_FIELDS,
    CONFIRMATORY_ANALYSIS_SCHEMA,
    CONFIRMATORY_CASE_COUNT,
    CONFIRMATORY_FOLD_COUNT,
    CONFIRMATORY_FOLD_SEED,
    CONFIRMATORY_FEATURE_FIELDS,
    CONFIRMATORY_EVENT_FIELDS,
    CONFIRMATORY_JOB_NAME,
    CONFIRMATORY_MANIFEST_SCHEMA,
    CONFIRMATORY_OUTCOME_ACTIONS,
    CONFIRMATORY_RECEIPT_SCHEMA,
    CONFIRMATORY_RUN_IDS,
    CONFIRMATORY_STREAM_SCHEMA,
    CONFIRMATORY_SUMMARY_SCHEMA,
    BUNDLE_ID,
    GLOBAL_ALPHA_ACTION,
    NATIVE_ACTION,
    REPLAY_ACTION,
    STATIC_ACTION,
    UNIFORM_ACTION,
    analyze_confirmatory_run,
)
from project.run_scripts.ode_edit_motivation.mv1_analysis import (
    build_confirmatory_fold_manifest,
)
from project.run_scripts.ode_edit_motivation.mv1_score_mix_d1_analysis import (
    D0_RUN_IDS,
    D1_RUN_IDS,
    STATIC_POLICY_SCHEMA,
)
from project.run_scripts.ode_edit_motivation.mv1_score_mix_confirmatory import (
    CONFIRMATORY_MANIFEST_SCHEMA as RUNNER_MANIFEST_SCHEMA,
    CONFIRMATORY_OUTCOME_ACTIONS as RUNNER_OUTCOME_ACTIONS,
    CONFIRMATORY_RECEIPT_SCHEMA as RUNNER_RECEIPT_SCHEMA,
    CONFIRMATORY_STREAM_SCHEMA as RUNNER_STREAM_SCHEMA,
    CONFIRMATORY_SUMMARY_SCHEMA as RUNNER_SUMMARY_SCHEMA,
    _ACTION_FIELDS as RUNNER_ACTION_FIELDS,
    CONFIRMATORY_EVENT_FIELDS as RUNNER_EVENT_FIELDS,
    _FEATURE_FIELDS as RUNNER_FEATURE_FIELDS,
)


MODEL = "llama3-8b-inst"
RUN_ID = CONFIRMATORY_RUN_IDS[MODEL]


def digest(value: object) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_stream(path: Path, event: str, payloads) -> None:
    lines = []
    for sequence, payload in enumerate(payloads):
        lines.append(
            json.dumps(
                {
                    "schema_version": CONFIRMATORY_STREAM_SCHEMA,
                    "run_id": RUN_ID,
                    "sequence": sequence,
                    "recorded_at": f"2026-07-31T01:{sequence:02d}:00+09:00",
                    "event": event,
                    "payload": payload,
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def static_policy():
    weight = 1.0 / math.sqrt(len(SINGLE_ACTIONS))
    policy = {
        "schema_version": STATIC_POLICY_SCHEMA,
        "model_alias": MODEL,
        "source_runs": {"d0": D0_RUN_IDS[MODEL], "d1": D1_RUN_IDS[MODEL]},
        "source_slices": {
            "d0": {"start": 3, "count": 5},
            "d1": {"start": 8, "count": 12},
        },
        "feature_case_count": 17,
        "feature_panel_sha256": "1" * 64,
        "fit_numeric_inputs": [
            f"action_scores.{action}" for action in SINGLE_ACTIONS
        ],
        "outcome_fields_used": [],
        "sbar_single_layer_slopes": {
            action: 1.0 for action in SINGLE_ACTIONS
        },
        "layer_weights": {action: weight for action in SINGLE_ACTIONS},
        "predicted_mean_score": math.sqrt(len(SINGLE_ACTIONS)),
        "controller": "relu(sbar)/L2-else-max-onehot",
        "controller_branch": "positive-relu-l2",
        "tie_policy": "exact-tie-lower-layer",
        "weight_l2_norm": 1.0,
        "claim_boundary": "synthetic frozen comparator",
    }
    policy["policy_hash"] = digest(policy)
    return policy


def forecast_artifacts(static):
    weight = 1.0 / math.sqrt(len(SINGLE_ACTIONS))
    cases = []
    for index in range(17):
        wave = "d0" if index < 5 else "d1"
        case_id = f"{wave}-forecast-{index:02d}"
        slopes = {action: 1.0 for action in SINGLE_ACTIONS}
        cases.append(
            {
                "wave": wave,
                "case_id": case_id,
                "request_id": digest({"forecast_request": case_id}),
                "feature_hash": digest({"forecast_feature": case_id}),
                "commitment_hash": digest({"forecast_action": case_id}),
                "operational_c_distance": 1.0,
                "slopes": slopes,
                "s_uniform": math.sqrt(len(SINGLE_ACTIONS)),
                "adaptive_weights": {
                    action: weight for action in SINGLE_ACTIONS
                },
                "leave_one_out_static_weights": {
                    action: weight for action in SINGLE_ACTIONS
                },
                "x_AU": 1.0,
                "y_AU": 0.5,
                "x_AS": 0.0,
                "no_op_absolute_progress": 0.02 if index == 0 else 0.0,
            }
        )
    policy = {
        "schema_version": FORECAST_POLICY_SCHEMA,
        "model_alias": MODEL,
        "static_policy_hash": static["policy_hash"],
        "source_runs": {"d0": D0_RUN_IDS[MODEL], "d1": D1_RUN_IDS[MODEL]},
        "beta": 0.5,
        "calibration_method": CALIBRATION_METHOD,
        "calibration_hash": digest(cases),
        "confirmatory_outcomes_used": [],
    }
    policy["policy_hash"] = digest(policy)
    analysis = {
        "schema_version": FORECAST_ANALYSIS_SCHEMA,
        "claim_status": "calibration_forecast_only_not_method_gain",
        "model_alias": MODEL,
        "source_runs": policy["source_runs"],
        "static_policy_hash": static["policy_hash"],
        "forecast_policy_hash": policy["policy_hash"],
        "calibration_hash": policy["calibration_hash"],
        "calibration_method": CALIBRATION_METHOD,
        "calibration_case_count": 17,
        "calibration_replay_envelope": 0.02,
        "beta": {
            "value": policy["beta"],
            "constraint": "nonnegative zero-intercept",
            "eligible_x_AU_case_count": 17,
            "x_floor": 1e-12,
        },
        "adaptive_static_forecast": {
            "mean_feature_only_x_AS": 0.0,
            "expected_realized_gap": 0.0,
            "case_bootstrap_interval_95": [0.0, 0.0],
            "bootstrap_seed": 20260731,
            "bootstrap_replicates": 4000,
            "calibration_residual_envelope": 0.5,
        },
        "feature_only_action_hash": digest({"feature_only": True}),
        "case_diagnostics": cases,
        "information_firewall": {
            "static_weights_use_outcomes": False,
            "adaptive_weights_use_outcomes": False,
            "beta_uses_calibration_adaptive_uniform_outcomes": True,
            "confirmatory_outcomes_used": [],
        },
        "claim_boundary": (
            "D0+D1 calibration-conditional adaptive-minus-static forecast; "
            "not a confirmatory effect, method gain, GO, MV-2, or ODE claim"
        ),
    }
    analysis["analysis_hash"] = digest(analysis)
    return policy, analysis


def selection_manifest():
    calibration = [f"cal-{index:02d}" for index in range(20)]
    confirmatory = [f"conf-{index:02d}" for index in range(60)]
    untouched = [f"unt-{index:02d}" for index in range(20)]
    selection = {
        "schema_version": "ode-edit-counterfact-selection/v1",
        "seed": "synthetic",
        "source_sha256": "3" * 64,
        "source_size": 1,
        "source_row_count": 100,
        "case_ids": {
            "calibration": calibration,
            "confirmatory": confirmatory,
            "untouched": untouched,
        },
        "order_hash": digest(calibration + confirmatory + untouched),
        "split_hash": digest(
            {
                "calibration": calibration,
                "confirmatory": confirmatory,
                "untouched": untouched,
            }
        ),
    }
    selection["manifest_id"] = _selection_manifest_id(selection)
    return selection


def build_run(parent: str | Path):
    static = static_policy()
    forecast, forecast_analysis = forecast_artifacts(static)
    selection = selection_manifest()
    fold_manifest = build_confirmatory_fold_manifest(
        selection["case_ids"]["confirmatory"],
        seed=CONFIRMATORY_FOLD_SEED,
        fold_count=CONFIRMATORY_FOLD_COUNT,
    )
    cases = [
        case_id
        for case_id in selection["case_ids"]["confirmatory"]
        if fold_manifest.fold_for(case_id) == 0
    ]
    requests = [digest({"request": case}) for case in cases]
    root = Path(parent) / RUN_ID
    root.mkdir(parents=True)
    receipt_root = root / "action_receipts"
    receipt_root.mkdir()

    features = []
    actions = []
    outcomes = []
    events = []
    receipt_hashes = {}
    effects = [value / 10.0 for value in range(-2, 10)]
    static_weights = static["layer_weights"]
    for index, (case_id, request_id, effect) in enumerate(
        zip(cases, requests, effects)
    ):
        slopes = {
            action_id: 1.0 + 0.1 * offset
            for offset, action_id in enumerate(SINGLE_ACTIONS)
        }
        slopes[SINGLE_ACTIONS[index % len(SINGLE_ACTIONS)]] += 3.0
        scores = {
            **slopes,
            "uniform": sum(slopes.values()) / math.sqrt(len(SINGLE_ACTIONS)),
        }
        adaptive_weights, adaptive_score, branch = _expected_mix(scores)
        static_score = sum(
            static_weights[action] * slopes[action] for action in SINGLE_ACTIONS
        )
        distance = 1.0
        score_gap = adaptive_score - static_score
        x_as = distance * score_gap
        predicted_gain = forecast["beta"] * x_as
        feature = {
            "case_id": case_id,
            "request_id": request_id,
            "q": SCORE_MIX_Q,
            "q_label": "q_1_256",
            "native_c_energy": 256.0,
            "operational_c_distance": distance,
            "probe_c_distance": 0.25,
            "action_scores": scores,
            "context_action_scores": {
                action: [value] for action, value in scores.items()
            },
            "feature_policy": "synthetic-confirmatory-central-fd",
            "static_policy_hash": static["policy_hash"],
            "forecast_policy_hash": forecast["policy_hash"],
            "adaptive_layer_weights": adaptive_weights,
            "static_layer_weights": static_weights,
            "adaptive_predicted_score": adaptive_score,
            "static_predicted_score": static_score,
            "predicted_adaptive_static_score_gap": score_gap,
            "x_as": x_as,
            "forecast_beta": forecast["beta"],
            "predicted_adaptive_static_gain": predicted_gain,
        }
        feature["feature_hash"] = digest(feature)
        bundle_id = BUNDLE_ID
        action = {
            "case_id": case_id,
            "request_id": request_id,
            "q": SCORE_MIX_Q,
            "q_label": "q_1_256",
            "feature_hash": feature["feature_hash"],
            "bundle_id": bundle_id,
            "adaptive_action_id": ADAPTIVE_ACTION,
            "adaptive_layer_weights": adaptive_weights,
            "adaptive_predicted_score": adaptive_score,
            "adaptive_actual_unit_c_energy": 1.0,
            "adaptive_controller": (
                "five-single-slopes/relu-l2-else-max-onehot"
            ),
            "adaptive_controller_branch": branch,
            "static_action_id": STATIC_ACTION,
            "static_layer_weights": static_weights,
            "static_predicted_score": static_score,
            "static_actual_unit_c_energy": 1.0,
            "static_policy_hash": static["policy_hash"],
            "forecast_policy_hash": forecast["policy_hash"],
            "predicted_adaptive_static_score_gap": score_gap,
            "x_as": x_as,
            "forecast_beta": forecast["beta"],
            "predicted_adaptive_static_gain": predicted_gain,
            "tie_policy": "exact-tie-lower-layer",
        }
        self_check = set(action) | {"commitment_hash"}
        if self_check != CONFIRMATORY_ACTION_FIELDS:
            raise AssertionError("synthetic action schema drift")
        action["commitment_hash"] = digest(action)
        receipt = {
            "schema_version": CONFIRMATORY_RECEIPT_SCHEMA,
            "case_id": case_id,
            "request_id": request_id,
            "q": SCORE_MIX_Q,
            "feature_hash": feature["feature_hash"],
            "commitment_hash": action["commitment_hash"],
            "bundle_id": bundle_id,
            "adaptive_action_id": ADAPTIVE_ACTION,
            "static_action_id": STATIC_ACTION,
            "adaptive_layer_weights": adaptive_weights,
            "static_layer_weights": static_weights,
            "static_policy_hash": static["policy_hash"],
            "forecast_policy_hash": forecast["policy_hash"],
            "predicted_adaptive_static_score_gap": score_gap,
            "x_as": x_as,
            "forecast_beta": forecast["beta"],
            "predicted_adaptive_static_gain": predicted_gain,
            "durability": (
                "feature+adaptive-static-forecast-action-write+flush+fsync-"
                "before-exclusive-receipt"
            ),
            "outcomes_observed_before_commitment": False,
        }
        receipt_identity = {
            key: receipt[key]
            for key in (
                "case_id",
                "request_id",
                "q",
                "feature_hash",
                "commitment_hash",
                "static_policy_hash",
                "forecast_policy_hash",
            )
        }
        receipt_name = digest(receipt_identity) + ".json"
        receipt_path = receipt_root / receipt_name
        write_json(receipt_path, receipt)
        receipt_hashes[receipt_name] = _sha256_file(receipt_path)

        arm_progress = {
            ADAPTIVE_ACTION: 1.0 + effect,
            STATIC_ACTION: 1.0,
            UNIFORM_ACTION: 1.05,
            GLOBAL_ALPHA_ACTION: 0.9,
            NATIVE_ACTION: 2.0,
            REPLAY_ACTION: 0.0,
        }
        for action_id in CONFIRMATORY_OUTCOME_ACTIONS:
            matched = action_id in {
                ADAPTIVE_ACTION,
                STATIC_ACTION,
                UNIFORM_ACTION,
                GLOBAL_ALPHA_ACTION,
            }
            replay = action_id == REPLAY_ACTION
            progress = arm_progress[action_id]
            payload = {
                "case_id": case_id,
                "request_id": request_id,
                "q": SCORE_MIX_Q,
                "q_label": "q_1_256",
                "feature_hash": feature["feature_hash"],
                "commitment_hash": action["commitment_hash"],
                "status": "completed",
                "rollback_exact": True,
                "action_id": action_id,
                "selected_by_controller": action_id == ADAPTIVE_ACTION,
                "c_distance": 1.0 if matched else (0.0 if replay else 16.0),
                "c_energy": 1.0 if matched else (0.0 if replay else 256.0),
                "budget_validation": (
                    "reference-no-op"
                    if replay
                    else "reference-native-full"
                    if action_id == NATIVE_ACTION
                    else "native-c-remeasured-then-global-alpha-squared/equal-c"
                    if action_id == GLOBAL_ALPHA_ACTION
                    else "unit-c-remeasured-then-scaled/equal-c"
                ),
                "progress": progress,
                "context_progress": [progress],
                "nll_reduction": progress,
                "context_nll_reduction": [progress],
                "exact_margin_min": progress,
                "exact_satisfied": progress >= 0.0,
                "static_policy_hash": static["policy_hash"],
                "forecast_policy_hash": forecast["policy_hash"],
            }
            if replay:
                payload["logits_hash_equal"] = True
            outcomes.append(payload)
        event = {
            "case_id": case_id,
            "request_id": request_id,
            "event_seed": index,
            "target_token_count": 1,
            "native_c_energy": 256.0,
            "q": SCORE_MIX_Q,
            "probe_direction_count": len(PROBE_ACTIONS),
            "feature_count": 1,
            "commitment_count": 1,
            "outcome_count": len(CONFIRMATORY_OUTCOME_ACTIONS),
            "direct_z_artifact_sha256": digest({"z": case_id}),
            "direct_z_artifact_size": 1,
            "static_policy_hash": static["policy_hash"],
            "forecast_policy_hash": forecast["policy_hash"],
            "action_receipt": {
                "name": receipt_name,
                "sha256": receipt_hashes[receipt_name],
            },
            "rollback_exact": True,
            "pass": True,
        }
        if set(feature) != CONFIRMATORY_FEATURE_FIELDS:
            raise AssertionError("synthetic feature schema drift")
        features.append(feature)
        actions.append(action)
        events.append(event)

    selected_fold = {
        "label": "c1",
        "fold": 0,
        "count": 12,
        "case_ids": cases,
    }
    summary_slice = {
        "label": "c1",
        "split": "confirmatory",
        "fold": 0,
        "fold_count": 5,
        "count": 12,
        "fold_manifest_id": fold_manifest.manifest_id,
    }
    slurm = {
        "under_slurm": True,
        "job_id": "40001",
        "job_name": CONFIRMATORY_JOB_NAME,
        "node": "devbox",
        "slice_label": "c1",
        "fold": 0,
    }
    manifest = {
        "schema_version": CONFIRMATORY_MANIFEST_SCHEMA,
        "run_id": RUN_ID,
        "ode_edit_git": {
            "commit": "4" * 40,
            "tracked_worktree_clean": True,
        },
        "slurm": slurm,
        "model": {
            "model_alias": MODEL,
            "repository_id": MODEL_ENVELOPES[MODEL]["repository_id"],
            "revision": MODEL_ENVELOPES[MODEL]["revision"],
            "dtype": "torch.float32",
            "device": "cuda:0",
            "offline": True,
        },
        "hparams_relative_path": "hparams.json",
        "selection": selection,
        "selected_split": "confirmatory",
        "confirmatory_folds": fold_manifest.to_dict(),
        "selected_fold": selected_fold,
        "selected_case_ids": cases,
        "selected_request_ids": requests,
        "contexts": {
            "manifest_id": "5" * 64,
            "source": "synthetic",
            "group_sizes": [1],
            "raw_templates_persisted": False,
        },
        "provenance_id": "6" * 64,
        "fixed_files": [],
        "q": SCORE_MIX_Q,
        "q_label": "q_1_256",
        "probe_ratio": 0.25,
        "probe_action_set": list(PROBE_ACTIONS),
        "controller_input_set": list(SINGLE_ACTIONS),
        "outcome_action_set": list(CONFIRMATORY_OUTCOME_ACTIONS),
        "static_policy": {
            "schema_version": STATIC_POLICY_SCHEMA,
            "policy_hash": static["policy_hash"],
            "model_alias": MODEL,
            "source_runs": static["source_runs"],
            "feature_panel_sha256": static["feature_panel_sha256"],
            "layer_weights": static["layer_weights"],
            "path_persisted": False,
        },
        "forecast_policy": {
            "schema_version": FORECAST_POLICY_SCHEMA,
            "policy_hash": forecast["policy_hash"],
            "model_alias": MODEL,
            "static_policy_hash": static["policy_hash"],
            "source_runs": forecast["source_runs"],
            "beta": forecast["beta"],
            "calibration_method": forecast["calibration_method"],
            "calibration_hash": forecast["calibration_hash"],
            "confirmatory_outcomes_used": [],
            "path_persisted": False,
        },
        "decision_policy": (
            "outcome-free adaptive five-single relu(g)/L2 plus "
            "calibration-only frozen-static weights"
        ),
        "forecast_policy_formula": (
            "x_AS=d*(dot(w_adaptive,s)-dot(w_static,s)); "
            "predicted_adaptive_static_gain=beta*x_AS"
        ),
        "action_receipt_policy": (
            "feature+adaptive/static action+forecast JSONL "
            "write/flush/fsync then exclusive receipt before outcomes"
        ),
        "utility_policy": (
            "temperature=1 mean-context-token smooth target margin"
        ),
        "teacher_suffix_policy": (
            "prefix+target single tokenization with exact target suffix"
        ),
        "projector_policy": (
            "full sha256-and-size preflight only; never deserialized"
        ),
        "covariance_policy": (
            "verified-read-only; recompute-and-download-blocked"
        ),
        "direct_z_policy": (
            "computed-once-and-frozen per event below this local run"
        ),
        "artifact_firewall": (
            "structured JSON contains case IDs/full request hashes/"
            "scalars/hashes only; prompts, targets, logits, weights, "
            "generations, and activations are forbidden; "
            "activation-derived direct_z remains local-only"
        ),
        "expected_counts": {
            "features": 12,
            "commitments": 12,
            "outcomes": 72,
            "receipts": 12,
        },
    }
    write_json(root / "manifest.json", manifest)
    write_stream(
        root / "features.jsonl", "mv1mix_confirmatory_feature", features
    )
    write_stream(
        root / "actions.jsonl",
        "mv1mix_confirmatory_action_commitment",
        actions,
    )
    write_stream(
        root / "outcomes.jsonl", "mv1mix_confirmatory_outcome", outcomes
    )
    write_stream(
        root / "events.jsonl", "mv1mix_confirmatory_case", events
    )
    summary = {
        "schema_version": CONFIRMATORY_SUMMARY_SCHEMA,
        "run_id": RUN_ID,
        "model_alias": MODEL,
        "slice": summary_slice,
        "slurm": slurm,
        "provenance_id": "6" * 64,
        "selection_manifest_id": selection["manifest_id"],
        "context_id": "5" * 64,
        "run_status": "completed",
        "abort_failure_type": None,
        "planned_case_count": 12,
        "attempted_case_count": 12,
        "not_run_due_to_abort_count": 0,
        "pass_count": 12,
        "failure_count": 0,
        "all_pass": True,
        "case_results": [
            {"case_id": case, "status": "attempted", "pass": True}
            for case in cases
        ],
        "q": SCORE_MIX_Q,
        "probe_direction_count_per_case": len(PROBE_ACTIONS),
        "feature_count": 12,
        "commitment_count": 12,
        "outcome_count": 72,
        "action_receipt_count": 12,
        "expected_counts_exact": True,
        "all_rollbacks_exact": True,
        "covariance_files_loaded": ["synthetic"],
        "projector_files_loaded": [],
        "direct_z_artifact_count": 12,
        "resource": {"wall_seconds": 1.0},
        "artifacts": {
            "manifest_sha256": _sha256_file(root / "manifest.json"),
            "features_sha256": _sha256_file(root / "features.jsonl"),
            "actions_sha256": _sha256_file(root / "actions.jsonl"),
            "outcomes_sha256": _sha256_file(root / "outcomes.jsonl"),
            "events_sha256": _sha256_file(root / "events.jsonl"),
            "action_receipts": receipt_hashes,
        },
        "git_output_written": False,
        "static_policy_hash": static["policy_hash"],
        "forecast_policy_hash": forecast["policy_hash"],
        "forecast_beta": forecast["beta"],
        "stream_sequences": {
            "features": 12,
            "actions": 12,
            "outcomes": 72,
            "events": 12,
        },
    }
    write_json(root / "summary.json", summary)
    return root, static, forecast, forecast_analysis


class ConfirmatoryAnalysisTests(unittest.TestCase):
    def test_runner_analyzer_exact_schema_parity(self):
        self.assertEqual(CONFIRMATORY_FEATURE_FIELDS, RUNNER_FEATURE_FIELDS)
        self.assertEqual(CONFIRMATORY_ACTION_FIELDS, RUNNER_ACTION_FIELDS)
        self.assertEqual(CONFIRMATORY_EVENT_FIELDS, RUNNER_EVENT_FIELDS)
        self.assertEqual(
            CONFIRMATORY_OUTCOME_ACTIONS, RUNNER_OUTCOME_ACTIONS
        )
        self.assertEqual(CONFIRMATORY_MANIFEST_SCHEMA, RUNNER_MANIFEST_SCHEMA)
        self.assertEqual(CONFIRMATORY_STREAM_SCHEMA, RUNNER_STREAM_SCHEMA)
        self.assertEqual(CONFIRMATORY_RECEIPT_SCHEMA, RUNNER_RECEIPT_SCHEMA)
        self.assertEqual(CONFIRMATORY_SUMMARY_SCHEMA, RUNNER_SUMMARY_SCHEMA)

    def test_exact_fold_six_arm_known_robust_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, static, forecast, forecast_analysis = build_run(temporary)
            report = analyze_confirmatory_run(
                root,
                static_policy=static,
                forecast_policy=forecast,
                forecast_analysis=forecast_analysis,
                bootstrap_seed=7,
                bootstrap_replicates=200,
            )

        self.assertEqual(report["schema_version"], CONFIRMATORY_ANALYSIS_SCHEMA)
        self.assertEqual(
            report["analysis_status"], "confirmatory_single_model_complete"
        )
        self.assertTrue(report["artifact_validation"]["valid"])
        self.assertEqual(report["artifact_validation"]["case_count"], 12)
        self.assertEqual(report["artifact_validation"]["arm_count_per_case"], 6)
        self.assertEqual(report["calibration_replay_envelope"], 0.02)
        self.assertEqual(report["c1_replay_envelope"], 1e-12)
        self.assertEqual(report["replay_envelope"], 0.02)
        primary = report["primary_adaptive_minus_frozen_static"]
        self.assertAlmostEqual(primary["mean"], 0.35)
        self.assertAlmostEqual(primary["trimmed_mean_20pct"], 0.35)
        self.assertAlmostEqual(primary["median"], 0.35)
        self.assertEqual(
            primary["positive_sign_count_above_replay_envelope"], 9
        )
        self.assertTrue(
            report["single_model_gate_inputs"]["clear_continue_input"]
        )
        self.assertFalse(
            report["single_model_gate_inputs"]["pair_level_decision_computed"]
        )

    def test_policy_tamper_fold_and_arm_count_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, static, forecast, forecast_analysis = build_run(temporary)
            tampered_policy = dict(forecast)
            tampered_policy["beta"] = 9.0
            report = analyze_confirmatory_run(
                root,
                static_policy=static,
                forecast_policy=tampered_policy,
                forecast_analysis=forecast_analysis,
                bootstrap_replicates=100,
            )
            self.assertFalse(report["artifact_validation"]["valid"])
            self.assertIn(
                "static_or_forecast_policy_invalid",
                report["artifact_validation"]["error_codes"],
            )

            manifest = json.loads((root / "manifest.json").read_text())
            manifest["selected_case_ids"][0] = "conf-01"
            write_json(root / "manifest.json", manifest)
            report = analyze_confirmatory_run(
                root,
                static_policy=static,
                forecast_policy=forecast,
                forecast_analysis=forecast_analysis,
                bootstrap_replicates=100,
            )
            self.assertFalse(report["artifact_validation"]["valid"])
            self.assertIn(
                "confirmatory_exact_fold_or_disjointness_mismatch",
                report["artifact_validation"]["error_codes"],
            )

    def test_outcome_mutation_does_not_change_committed_actions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, static, forecast, forecast_analysis = build_run(temporary)
            action_before = (root / "actions.jsonl").read_bytes()
            report_before = analyze_confirmatory_run(
                root,
                static_policy=static,
                forecast_policy=forecast,
                forecast_analysis=forecast_analysis,
                bootstrap_replicates=100,
            )

            lines = (root / "outcomes.jsonl").read_text().splitlines()
            for index, line in enumerate(lines):
                row = json.loads(line)
                if row["payload"]["action_id"] == ADAPTIVE_ACTION:
                    row["payload"]["progress"] += 1.0
                    lines[index] = json.dumps(row, sort_keys=True)
            (root / "outcomes.jsonl").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            summary = json.loads((root / "summary.json").read_text())
            summary["artifacts"]["outcomes_sha256"] = _sha256_file(
                root / "outcomes.jsonl"
            )
            write_json(root / "summary.json", summary)
            report_after = analyze_confirmatory_run(
                root,
                static_policy=static,
                forecast_policy=forecast,
                forecast_analysis=forecast_analysis,
                bootstrap_replicates=100,
            )

            self.assertEqual(action_before, (root / "actions.jsonl").read_bytes())
            self.assertTrue(report_after["artifact_validation"]["valid"])
            self.assertNotEqual(
                report_before["primary_adaptive_minus_frozen_static"]["mean"],
                report_after["primary_adaptive_minus_frozen_static"]["mean"],
            )

    def test_forecast_analysis_hash_and_residual_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, static, forecast, forecast_analysis = build_run(temporary)
            tampered = copy.deepcopy(forecast_analysis)
            tampered["adaptive_static_forecast"][
                "calibration_residual_envelope"
            ] = -1.0
            tampered_without_hash = dict(tampered)
            tampered_without_hash.pop("analysis_hash")
            tampered["analysis_hash"] = digest(tampered_without_hash)
            report = analyze_confirmatory_run(
                root,
                static_policy=static,
                forecast_policy=forecast,
                forecast_analysis=tampered,
                bootstrap_replicates=100,
            )
            self.assertFalse(report["artifact_validation"]["valid"])
            self.assertIn(
                "forecast_analysis_parity_mismatch",
                report["artifact_validation"]["error_codes"],
            )

            with self.assertRaises(TypeError):
                analyze_confirmatory_run(
                    root,
                    static_policy=static,
                    forecast_policy=forecast,
                    bootstrap_replicates=100,
                )

    def test_outcome_stream_exact_arm_order_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, static, forecast, forecast_analysis = build_run(temporary)
            path = root / "outcomes.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["payload"], rows[1]["payload"] = (
                rows[1]["payload"],
                rows[0]["payload"],
            )
            path.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                encoding="utf-8",
            )
            summary = json.loads((root / "summary.json").read_text())
            summary["artifacts"]["outcomes_sha256"] = _sha256_file(path)
            write_json(root / "summary.json", summary)
            report = analyze_confirmatory_run(
                root,
                static_policy=static,
                forecast_policy=forecast,
                forecast_analysis=forecast_analysis,
                bootstrap_replicates=100,
            )
            self.assertFalse(report["artifact_validation"]["valid"])
            self.assertIn(
                "outcome_exact_order_mismatch",
                report["artifact_validation"]["error_codes"],
            )

    def test_scientific_kill_uses_spec_oracle_mean_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, static, forecast, forecast_analysis = build_run(temporary)
            path = root / "outcomes.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            case_order = list(
                json.loads((root / "manifest.json").read_text())[
                    "selected_case_ids"
                ]
            )
            for row in rows:
                payload = row["payload"]
                case_index = case_order.index(payload["case_id"])
                if payload["action_id"] in {
                    ADAPTIVE_ACTION,
                    STATIC_ACTION,
                    GLOBAL_ALPHA_ACTION,
                }:
                    payload["progress"] = 1.0
                elif payload["action_id"] == UNIFORM_ACTION:
                    payload["progress"] = 1.11 if case_index < 7 else 1.0
                elif payload["action_id"] == REPLAY_ACTION:
                    payload["progress"] = 0.1
            path.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                encoding="utf-8",
            )
            summary = json.loads((root / "summary.json").read_text())
            summary["artifacts"]["outcomes_sha256"] = _sha256_file(path)
            write_json(root / "summary.json", summary)
            report = analyze_confirmatory_run(
                root,
                static_policy=static,
                forecast_policy=forecast,
                forecast_analysis=forecast_analysis,
                bootstrap_replicates=100,
            )
            self.assertTrue(report["artifact_validation"]["valid"])
            self.assertAlmostEqual(report["replay_envelope"], 0.1)
            oracle = report["finite_panel_oracle_opportunity"]
            self.assertLessEqual(oracle["mean"], report["replay_envelope"])
            self.assertGreater(
                oracle["positive_sign_fraction_above_replay_envelope"],
                0.5,
            )
            self.assertTrue(
                report["single_model_gate_inputs"]["scientific_kill_input"]
            )


if __name__ == "__main__":
    unittest.main()
