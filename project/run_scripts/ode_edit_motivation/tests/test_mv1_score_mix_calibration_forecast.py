from __future__ import annotations

import copy
import math
import unittest

from project.run_scripts.ode_edit_motivation.mv1_score_mix_analysis import (
    OUTCOME_ACTIONS,
    SCORE_MIX_Q,
    SINGLE_ACTIONS,
    _canonical_json,
    _expected_mix,
    _sha256_bytes,
    ScoreMixAnalysisError,
)
from project.run_scripts.ode_edit_motivation.mv1_score_mix_calibration_forecast import (
    CALIBRATION_METHOD,
    FORECAST_ANALYSIS_SCHEMA,
    FORECAST_POLICY_SCHEMA,
    _fit_nonnegative_robust_beta,
    fit_calibration_forecast,
    validate_forecast_policy,
)
from project.run_scripts.ode_edit_motivation.mv1_score_mix_d1_analysis import (
    D0_RUN_IDS,
    D1_RUN_IDS,
)
from project.run_scripts.ode_edit_motivation.mv1_score_mix_confirmatory import (
    FORECAST_POLICY_SCHEMA as RUNNER_FORECAST_POLICY_SCHEMA,
    _FORECAST_POLICY_FIELDS as RUNNER_FORECAST_POLICY_FIELDS,
)


MODEL = "llama3-8b-inst"


def digest(value: object) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def make_case(index: int, wave: str):
    case_id = f"{wave}-{index:02d}"
    request_id = digest({"request": case_id})
    # Rotate a clearly useful layer while retaining positive support elsewhere.
    slopes = {
        action: 1.0 + 0.1 * offset
        for offset, action in enumerate(SINGLE_ACTIONS)
    }
    selected = index % len(SINGLE_ACTIONS)
    slopes[SINGLE_ACTIONS[selected]] += 3.0
    analytic_uniform = sum(slopes.values()) / math.sqrt(len(SINGLE_ACTIONS))
    # The measured uniform probe is intentionally a little below the analytic
    # direct-sum score, creating a positive and known x_AU.
    scores = {**slopes, "uniform": analytic_uniform - 0.25}
    weights, predicted_score, branch = _expected_mix(scores)
    distance = 1.0
    x_au = distance * (
        sum(weights[action] * slopes[action] for action in SINGLE_ACTIONS)
        - scores["uniform"]
    )
    y_au = 2.0 * x_au

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
        "feature_policy": "synthetic-six-unit-c-central-fd",
    }
    feature["feature_hash"] = digest(feature)
    action = {
        "case_id": case_id,
        "request_id": request_id,
        "q": SCORE_MIX_Q,
        "q_label": "q_1_256",
        "feature_hash": feature["feature_hash"],
        "action_id": "score_mix",
        "layer_weights": weights,
        "predicted_score": predicted_score,
        "actual_unit_c_energy": 1.0,
        "controller": "five-single-slopes/relu-l2-else-max-onehot",
        "controller_branch": branch,
        "tie_policy": "exact-tie-lower-layer",
    }
    action["commitment_hash"] = digest(action)

    progress = {
        "score_mix": 1.0 + y_au,
        "uniform": 1.0,
        "ordered_global_alpha": 0.9,
        "native_memit_full": 2.0,
        "no_op_replay": 0.0,
    }
    outcomes = []
    for action_id in OUTCOME_ACTIONS:
        equal_c = action_id in {
            "score_mix",
            "uniform",
            "ordered_global_alpha",
        }
        replay = action_id == "no_op_replay"
        value = progress[action_id]
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
            "selected_by_controller": action_id == "score_mix",
            "c_distance": 1.0 if equal_c else (0.0 if replay else 16.0),
            "c_energy": 1.0 if equal_c else (0.0 if replay else 256.0),
            "budget_validation": (
                "reference-no-op"
                if replay
                else "reference-native-full"
                if action_id == "native_memit_full"
                else "native-c-remeasured-then-global-alpha-squared/equal-c"
                if action_id == "ordered_global_alpha"
                else "unit-c-remeasured-then-scaled/equal-c"
            ),
            "progress": value,
            "context_progress": [value],
            "nll_reduction": value,
            "context_nll_reduction": [value],
            "exact_margin_min": value,
            "exact_satisfied": value >= 0.0,
        }
        if replay:
            payload["logits_hash_equal"] = True
        outcomes.append(payload)
    return feature, action, outcomes


def make_inputs():
    d0_features, d0_actions, d0_outcomes = [], [], []
    d1_features, d1_actions, d1_outcomes = [], [], []
    for index in range(5):
        feature, action, outcomes = make_case(index, "d0")
        d0_features.append(feature)
        d0_actions.append(action)
        d0_outcomes.extend(outcomes)
    for index in range(12):
        feature, action, outcomes = make_case(index + 5, "d1")
        d1_features.append(feature)
        d1_actions.append(action)
        d1_outcomes.extend(outcomes)
    return {
        "model_alias": MODEL,
        "d0_run_id": D0_RUN_IDS[MODEL],
        "d1_run_id": D1_RUN_IDS[MODEL],
        "d0_features": d0_features,
        "d0_actions": d0_actions,
        "d0_outcomes": d0_outcomes,
        "d1_features": d1_features,
        "d1_actions": d1_actions,
        "d1_outcomes": d1_outcomes,
        "bootstrap_seed": 123,
        "bootstrap_replicates": 200,
    }


class CalibrationForecastTests(unittest.TestCase):
    def test_forecast_policy_schema_matches_runner_exactly(self):
        self.assertEqual(FORECAST_POLICY_SCHEMA, RUNNER_FORECAST_POLICY_SCHEMA)
        self.assertEqual(
            RUNNER_FORECAST_POLICY_FIELDS,
            {
                "schema_version",
                "model_alias",
                "static_policy_hash",
                "source_runs",
                "beta",
                "calibration_method",
                "calibration_hash",
                "confirmatory_outcomes_used",
                "policy_hash",
            },
        )

    def test_known_beta_policy_schema_and_forecast(self):
        analysis, policy, static = fit_calibration_forecast(**make_inputs())

        self.assertEqual(analysis["schema_version"], FORECAST_ANALYSIS_SCHEMA)
        self.assertEqual(policy["schema_version"], FORECAST_POLICY_SCHEMA)
        self.assertEqual(set(policy), {
            "schema_version",
            "model_alias",
            "static_policy_hash",
            "source_runs",
            "beta",
            "calibration_method",
            "calibration_hash",
            "confirmatory_outcomes_used",
            "policy_hash",
        })
        self.assertAlmostEqual(policy["beta"], 2.0, places=10)
        self.assertGreater(
            analysis["adaptive_static_forecast"]["expected_realized_gap"],
            0.0,
        )
        self.assertEqual(policy["confirmatory_outcomes_used"], [])
        self.assertEqual(policy["static_policy_hash"], static["policy_hash"])
        self.assertEqual(validate_forecast_policy(policy), policy)
        analysis_without_hash = dict(analysis)
        observed_analysis_hash = analysis_without_hash.pop("analysis_hash")
        self.assertEqual(observed_analysis_hash, digest(analysis_without_hash))
        self.assertEqual(analysis["calibration_replay_envelope"], 1e-12)
        self.assertTrue(
            all(
                row["no_op_absolute_progress"] == 0.0
                for row in analysis["case_diagnostics"]
            )
        )

    def test_calibration_outcome_mutation_cannot_change_actions(self):
        inputs = make_inputs()
        before_analysis, before_policy, before_static = fit_calibration_forecast(
            **inputs
        )
        changed = copy.deepcopy(inputs)
        for outcome in changed["d0_outcomes"] + changed["d1_outcomes"]:
            if outcome["action_id"] == "score_mix":
                outcome["progress"] *= -3.0
        after_analysis, after_policy, after_static = fit_calibration_forecast(
            **changed
        )

        # Calibration outcomes may change beta, but neither adaptive nor static
        # feature-only action is allowed to move.
        self.assertEqual(before_static, after_static)
        self.assertEqual(
            before_analysis["feature_only_action_hash"],
            after_analysis["feature_only_action_hash"],
        )
        self.assertEqual(
            [
                row["adaptive_weights"]
                for row in before_analysis["case_diagnostics"]
            ],
            [
                row["adaptive_weights"]
                for row in after_analysis["case_diagnostics"]
            ],
        )
        self.assertNotEqual(before_policy["beta"], after_policy["beta"])
        self.assertEqual(after_policy["beta"], 0.0)
        self.assertNotEqual(
            before_policy["calibration_hash"],
            after_policy["calibration_hash"],
        )

    def test_beta_zero_negative_and_outlier_robustness(self):
        beta, eligible = _fit_nonnegative_robust_beta(
            [0.0, 1e-12, -1e-12],
            [9.0, 9.0, 9.0],
        )
        self.assertEqual((beta, eligible), (0.0, 0))

        beta, eligible = _fit_nonnegative_robust_beta(
            [-2.0, -1.0, 1.0],
            [-4.0, -2.0, 2.0],
        )
        self.assertEqual((beta, eligible), (2.0, 3))

        beta, eligible = _fit_nonnegative_robust_beta(
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1e12],
        )
        self.assertEqual((beta, eligible), (2.0, 5))

        beta, eligible = _fit_nonnegative_robust_beta(
            [1.0, 2.0, 3.0],
            [-1.0, -2.0, 99.0],
        )
        self.assertEqual((beta, eligible), (0.0, 3))

    def test_calibration_replay_is_case_bound_and_policy_hashed(self):
        inputs = make_inputs()
        replay = next(
            outcome
            for outcome in inputs["d1_outcomes"]
            if outcome["action_id"] == "no_op_replay"
        )
        replay["progress"] = -0.125
        analysis, policy, _static = fit_calibration_forecast(**inputs)

        self.assertEqual(analysis["calibration_replay_envelope"], 0.125)
        matching = [
            row
            for row in analysis["case_diagnostics"]
            if row["case_id"] == replay["case_id"]
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["no_op_absolute_progress"], 0.125)
        self.assertEqual(policy["calibration_hash"], digest(analysis["case_diagnostics"]))

    def test_policy_tamper_and_outcome_panel_failure_are_blocked(self):
        _analysis, policy, _static = fit_calibration_forecast(**make_inputs())
        tampered = dict(policy)
        tampered["beta"] += 1.0
        with self.assertRaises(ScoreMixAnalysisError):
            validate_forecast_policy(tampered)

        wrong_method = dict(policy)
        wrong_method["calibration_method"] = "other-robust-method"
        wrong_method["policy_hash"] = digest(
            {
                key: value
                for key, value in wrong_method.items()
                if key != "policy_hash"
            }
        )
        self.assertNotEqual(wrong_method["calibration_method"], CALIBRATION_METHOD)
        with self.assertRaises(ScoreMixAnalysisError):
            validate_forecast_policy(wrong_method)

        inputs = make_inputs()
        inputs["d1_outcomes"] = inputs["d1_outcomes"][:-1]
        with self.assertRaises(ScoreMixAnalysisError):
            fit_calibration_forecast(**inputs)


if __name__ == "__main__":
    unittest.main()
