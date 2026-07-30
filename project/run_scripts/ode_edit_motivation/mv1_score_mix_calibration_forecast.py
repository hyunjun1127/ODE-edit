"""Calibration-only forecast policy for the MV-1 score-mix confirmatory run.

This module deliberately separates two objects:

* a small runner policy containing only the frozen static-policy identity and
  the outcome-calibrated scalar ``beta``; and
* a descriptive calibration analysis containing the expected
  adaptive-minus-static gap, interval, residual envelope, and case diagnostics.

The adaptive and static actions are functions of feature rows only.  Calibration
outcomes may calibrate ``beta`` but cannot change either action.  Confirmatory
outcomes are neither accepted by the fitting API nor represented in the policy.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .mv1_score_mix_analysis import (
    NUMERIC_COMPARISON_TOLERANCE,
    OUTCOME_ACTIONS,
    PROBE_ACTIONS,
    SCORE_MIX_Q,
    SINGLE_ACTIONS,
    _ACTION_FIELDS,
    _FEATURE_FIELDS,
    _OUTCOME_FIELDS,
    _canonical_json,
    _contains_outcome_field,
    _expected_mix,
    _finite,
    _is_full_hash,
    _payload_hash,
    _read_json_object,
    _read_jsonl,
    _sha256_bytes,
    _validate_rows,
    _write_exclusive,
    analyze_score_mix_run,
    ScoreMixAnalysisError,
)
from .mv1_score_mix_d1_analysis import (
    D0_RUN_IDS,
    D1_RUN_IDS,
    STATIC_POLICY_SCHEMA,
    _analyze_d1_internal,
    _cross_validate_d0_d1,
    fit_frozen_static_policy_from_features,
)


FORECAST_ANALYSIS_SCHEMA = "ode-edit-mv1-score-mix-calibration-forecast/v1"
FORECAST_POLICY_SCHEMA = "ode-edit-mv1-score-mix-forecast-policy/v1"
CALIBRATION_CASE_COUNT = 17
D0_CASE_COUNT = 5
D1_CASE_COUNT = 12
BETA_X_FLOOR = NUMERIC_COMPARISON_TOLERANCE
DEFAULT_BOOTSTRAP_SEED = 20260731
DEFAULT_BOOTSTRAP_REPLICATES = 4000
CALIBRATION_METHOD = (
    "nonnegative-zero-intercept-median-ratio/"
    "eligible:abs(x_AU)>1e-12/"
    "beta=max(0,median(y_AU/x_AU))/"
    "feature-only-17-case-leave-one-out-static/"
    "case-bootstrap-refit-beta-and-mean-x_AS"
)

_POLICY_FIELDS = {
    "schema_version",
    "model_alias",
    "static_policy_hash",
    "source_runs",
    "beta",
    "calibration_method",
    "calibration_hash",
    "confirmatory_outcomes_used",
    "policy_hash",
}


def _close(left: float, right: float, *, tolerance: float = 2e-7) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ScoreMixAnalysisError("mean requires a non-empty finite sequence")
    return statistics.fmean(values)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ScoreMixAnalysisError("percentile requires a non-empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ScoreMixAnalysisError("percentile probability is invalid")
    ordered = sorted(_finite(value, name="bootstrap_value") for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _feature_only_weights(slopes: Mapping[str, float]) -> dict[str, float]:
    if set(slopes) != set(SINGLE_ACTIONS):
        raise ScoreMixAnalysisError("feature-only slope panel mismatch")
    uniform = sum(float(slopes[action]) for action in SINGLE_ACTIONS) / math.sqrt(
        len(SINGLE_ACTIONS)
    )
    weights, _score, _branch = _expected_mix({**slopes, "uniform": uniform})
    return weights


def _dot(weights: Mapping[str, float], slopes: Mapping[str, float]) -> float:
    if set(weights) != set(SINGLE_ACTIONS) or set(slopes) != set(SINGLE_ACTIONS):
        raise ScoreMixAnalysisError("layer dot-product panel mismatch")
    value = sum(
        _finite(weights[action], name=f"weight:{action}")
        * _finite(slopes[action], name=f"slope:{action}")
        for action in SINGLE_ACTIONS
    )
    return _finite(value, name="layer_dot_product")


def _fit_nonnegative_robust_beta(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> tuple[float, int]:
    """Fit the pre-specified robust zero-intercept calibration coefficient.

    The unconstrained robust coefficient is the median of per-case
    zero-intercept ratios for cases with ``abs(x_AU) > 1e-12``.  Projection
    onto the non-negative half-line is explicit:
    ``beta=max(0, median_ratio)``.
    """

    if len(x_values) != len(y_values) or not x_values:
        raise ScoreMixAnalysisError("beta input length mismatch")
    ratios = [
        _finite(y, name="y_AU") / _finite(x, name="x_AU")
        for x, y in zip(x_values, y_values)
        if abs(_finite(x, name="x_AU")) > BETA_X_FLOOR
    ]
    if not ratios:
        return 0.0, 0
    beta = max(0.0, _finite(statistics.median(ratios), name="median_ratio"))
    return beta, len(ratios)


def _validate_feature_action_outcome_wave(
    *,
    wave: str,
    expected_count: int,
    features: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if wave not in {"d0", "d1"}:
        raise ScoreMixAnalysisError("unknown calibration wave")
    if (
        len(features) != expected_count
        or len(actions) != expected_count
        or len(outcomes) != expected_count * len(OUTCOME_ACTIONS)
    ):
        raise ScoreMixAnalysisError("calibration wave exact count mismatch")

    feature_by_case: dict[str, Mapping[str, Any]] = {}
    action_by_case: dict[str, Mapping[str, Any]] = {}
    outcome_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for feature in features:
        if set(feature) != _FEATURE_FIELDS or _contains_outcome_field(feature):
            raise ScoreMixAnalysisError("calibration feature schema/firewall mismatch")
        case_id = feature.get("case_id")
        if not isinstance(case_id, str) or case_id in feature_by_case:
            raise ScoreMixAnalysisError("calibration feature identity mismatch")
        observed, expected = _payload_hash(feature, "feature_hash")
        if observed != expected or not _is_full_hash(feature.get("request_id")):
            raise ScoreMixAnalysisError("calibration feature hash mismatch")
        if (
            feature.get("q") != SCORE_MIX_Q
            or feature.get("q_label") != "q_1_256"
            or not isinstance(feature.get("action_scores"), dict)
            or set(feature["action_scores"]) != set(PROBE_ACTIONS)
        ):
            raise ScoreMixAnalysisError("calibration feature policy mismatch")
        feature_by_case[case_id] = feature

    for action in actions:
        if set(action) != _ACTION_FIELDS or _contains_outcome_field(action):
            raise ScoreMixAnalysisError("calibration action schema/firewall mismatch")
        case_id = action.get("case_id")
        if not isinstance(case_id, str) or case_id in action_by_case:
            raise ScoreMixAnalysisError("calibration action identity mismatch")
        observed, expected = _payload_hash(action, "commitment_hash")
        if observed != expected:
            raise ScoreMixAnalysisError("calibration action hash mismatch")
        feature = feature_by_case.get(case_id)
        if (
            feature is None
            or action.get("request_id") != feature.get("request_id")
            or action.get("feature_hash") != feature.get("feature_hash")
            or action.get("action_id") != "score_mix"
        ):
            raise ScoreMixAnalysisError("calibration feature/action chain mismatch")
        scores = feature["action_scores"]
        expected_weights, expected_score, expected_branch = _expected_mix(scores)
        weights = action.get("layer_weights")
        if not isinstance(weights, dict) or set(weights) != set(SINGLE_ACTIONS):
            raise ScoreMixAnalysisError("adaptive weight panel mismatch")
        if any(
            not _close(
                _finite(weights[key], name=f"adaptive_weight:{key}"),
                expected_weights[key],
            )
            for key in SINGLE_ACTIONS
        ):
            raise ScoreMixAnalysisError("adaptive weights differ from feature policy")
        if (
            not _close(
                _finite(action.get("predicted_score"), name="predicted_score"),
                expected_score,
            )
            or action.get("controller_branch") != expected_branch
            or not _close(
                _finite(
                    action.get("actual_unit_c_energy"),
                    name="actual_unit_c_energy",
                ),
                1.0,
                tolerance=3e-5,
            )
        ):
            raise ScoreMixAnalysisError("adaptive action parity mismatch")
        action_by_case[case_id] = action

    for outcome in outcomes:
        action_id = outcome.get("action_id")
        expected_fields = (
            _OUTCOME_FIELDS | {"logits_hash_equal"}
            if action_id == "no_op_replay"
            else _OUTCOME_FIELDS
        )
        if set(outcome) != expected_fields:
            raise ScoreMixAnalysisError("calibration outcome schema mismatch")
        case_id = outcome.get("case_id")
        key = (case_id, action_id)
        if (
            not isinstance(case_id, str)
            or action_id not in OUTCOME_ACTIONS
            or key in outcome_by_key
        ):
            raise ScoreMixAnalysisError("calibration outcome identity mismatch")
        feature = feature_by_case.get(case_id)
        action = action_by_case.get(case_id)
        if (
            feature is None
            or action is None
            or outcome.get("request_id") != feature.get("request_id")
            or outcome.get("feature_hash") != feature.get("feature_hash")
            or outcome.get("commitment_hash") != action.get("commitment_hash")
            or outcome.get("status") != "completed"
            or outcome.get("rollback_exact") is not True
        ):
            raise ScoreMixAnalysisError("calibration outcome chain/rollback mismatch")
        _finite(outcome.get("progress"), name="outcome.progress")
        outcome_by_key[key] = outcome

    normalized: list[dict[str, Any]] = []
    for feature in features:
        case_id = str(feature["case_id"])
        if {
            action_id for observed_case, action_id in outcome_by_key
            if observed_case == case_id
        } != set(OUTCOME_ACTIONS):
            raise ScoreMixAnalysisError("calibration outcome arm panel mismatch")
        action = action_by_case[case_id]
        distance = _finite(
            feature.get("operational_c_distance"),
            name="operational_c_distance",
        )
        if distance <= 0.0:
            raise ScoreMixAnalysisError("operational C distance must be positive")
        for equal_c_action in (
            "score_mix",
            "uniform",
            "ordered_global_alpha",
        ):
            outcome = outcome_by_key[(case_id, equal_c_action)]
            if (
                not _close(
                    _finite(outcome.get("c_distance"), name="c_distance"),
                    distance,
                    tolerance=3e-5,
                )
                or not _close(
                    _finite(outcome.get("c_energy"), name="c_energy"),
                    distance * distance,
                    tolerance=3e-5,
                )
            ):
                raise ScoreMixAnalysisError("calibration equal-C mismatch")
        replay = outcome_by_key[(case_id, "no_op_replay")]
        if (
            replay.get("logits_hash_equal") is not True
            or _finite(replay.get("c_distance"), name="replay.c_distance") != 0.0
            or _finite(replay.get("c_energy"), name="replay.c_energy") != 0.0
        ):
            raise ScoreMixAnalysisError("calibration replay mismatch")
        slopes = {
            action_id: _finite(
                feature["action_scores"][action_id],
                name=f"score:{action_id}",
            )
            for action_id in SINGLE_ACTIONS
        }
        adaptive_weights = {
            action_id: _finite(
                action["layer_weights"][action_id],
                name=f"adaptive_weight:{action_id}",
            )
            for action_id in SINGLE_ACTIONS
        }
        s_uniform = _finite(
            feature["action_scores"]["uniform"], name="score:uniform"
        )
        x_au = distance * (_dot(adaptive_weights, slopes) - s_uniform)
        y_au = _finite(
            outcome_by_key[(case_id, "score_mix")]["progress"],
            name="adaptive_progress",
        ) - _finite(
            outcome_by_key[(case_id, "uniform")]["progress"],
            name="uniform_progress",
        )
        no_op_absolute_progress = abs(
            _finite(
                outcome_by_key[(case_id, "no_op_replay")]["progress"],
                name="no_op_progress",
            )
        )
        normalized.append(
            {
                "wave": wave,
                "case_id": case_id,
                "request_id": feature["request_id"],
                "feature_hash": feature["feature_hash"],
                "commitment_hash": action["commitment_hash"],
                "operational_c_distance": distance,
                "slopes": slopes,
                "s_uniform": s_uniform,
                "adaptive_weights": adaptive_weights,
                "x_AU": _finite(x_au, name="x_AU"),
                "y_AU": _finite(y_au, name="y_AU"),
                "no_op_absolute_progress": no_op_absolute_progress,
            }
        )
    return normalized


def _attach_leave_one_out_static(
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(cases) != CALIBRATION_CASE_COUNT:
        raise ScoreMixAnalysisError("leave-one-out static requires 17 cases")
    output: list[dict[str, Any]] = []
    for held_out_index, case in enumerate(cases):
        sums = {action: 0.0 for action in SINGLE_ACTIONS}
        for index, other in enumerate(cases):
            if index == held_out_index:
                continue
            for action in SINGLE_ACTIONS:
                sums[action] += _finite(
                    other["slopes"][action],
                    name=f"loo_slope:{action}",
                )
        means = {
            action: sums[action] / (CALIBRATION_CASE_COUNT - 1)
            for action in SINGLE_ACTIONS
        }
        static_weights = _feature_only_weights(means)
        x_as = _finite(case["operational_c_distance"], name="distance") * (
            _dot(case["adaptive_weights"], case["slopes"])
            - _dot(static_weights, case["slopes"])
        )
        output.append(
            {
                **dict(case),
                "leave_one_out_static_weights": static_weights,
                "x_AS": _finite(x_as, name="x_AS"),
            }
        )
    return output


def _bootstrap_forecast(
    cases: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> list[float]:
    if replicates < 100:
        raise ScoreMixAnalysisError("at least 100 bootstrap replicates are required")
    generator = random.Random(seed)
    values: list[float] = []
    for _ in range(replicates):
        sample = [
            cases[generator.randrange(len(cases))] for _ in range(len(cases))
        ]
        beta, _eligible = _fit_nonnegative_robust_beta(
            [float(case["x_AU"]) for case in sample],
            [float(case["y_AU"]) for case in sample],
        )
        values.append(
            beta * _mean([float(case["x_AS"]) for case in sample])
        )
    return values


def validate_forecast_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a normalized exact runner policy."""

    if set(policy) != _POLICY_FIELDS:
        raise ScoreMixAnalysisError("forecast policy exact schema mismatch")
    if (
        policy.get("schema_version") != FORECAST_POLICY_SCHEMA
        or policy.get("model_alias") not in D1_RUN_IDS
        or not _is_full_hash(policy.get("static_policy_hash"))
        or not _is_full_hash(policy.get("calibration_hash"))
        or policy.get("calibration_method") != CALIBRATION_METHOD
        or policy.get("confirmatory_outcomes_used") != []
    ):
        raise ScoreMixAnalysisError("forecast policy identity/firewall mismatch")
    source_runs = policy.get("source_runs")
    model_alias = str(policy["model_alias"])
    if source_runs != {
        "d0": D0_RUN_IDS[model_alias],
        "d1": D1_RUN_IDS[model_alias],
    }:
        raise ScoreMixAnalysisError("forecast policy source run mismatch")
    beta = _finite(policy.get("beta"), name="forecast_policy.beta")
    if beta < 0.0:
        raise ScoreMixAnalysisError("forecast policy beta must be non-negative")
    stripped = dict(policy)
    observed_hash = stripped.pop("policy_hash", None)
    expected_hash = _sha256_bytes(_canonical_json(stripped).encode("utf-8"))
    if observed_hash != expected_hash:
        raise ScoreMixAnalysisError("forecast policy hash mismatch")
    return json.loads(_canonical_json(policy))


def fit_calibration_forecast(
    *,
    model_alias: str,
    d0_run_id: str,
    d1_run_id: str,
    d0_features: Sequence[Mapping[str, Any]],
    d0_actions: Sequence[Mapping[str, Any]],
    d0_outcomes: Sequence[Mapping[str, Any]],
    d1_features: Sequence[Mapping[str, Any]],
    d1_actions: Sequence[Mapping[str, Any]],
    d1_outcomes: Sequence[Mapping[str, Any]],
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Fit forecast analysis, small policy, and feature-only static policy.

    No confirmatory outcome argument exists.  Calibration outcomes affect only
    ``beta`` and descriptive forecast quantities; adaptive and static weights
    are computed before those outcomes are consulted.
    """

    if model_alias not in D1_RUN_IDS:
        raise ScoreMixAnalysisError("forecast model is outside fixed models")
    if (
        d0_run_id != D0_RUN_IDS[model_alias]
        or d1_run_id != D1_RUN_IDS[model_alias]
    ):
        raise ScoreMixAnalysisError("forecast source run envelope mismatch")

    d0_cases = _validate_feature_action_outcome_wave(
        wave="d0",
        expected_count=D0_CASE_COUNT,
        features=d0_features,
        actions=d0_actions,
        outcomes=d0_outcomes,
    )
    d1_cases = _validate_feature_action_outcome_wave(
        wave="d1",
        expected_count=D1_CASE_COUNT,
        features=d1_features,
        actions=d1_actions,
        outcomes=d1_outcomes,
    )
    cases = [*d0_cases, *d1_cases]
    if (
        len({case["case_id"] for case in cases}) != CALIBRATION_CASE_COUNT
        or len({case["request_id"] for case in cases}) != CALIBRATION_CASE_COUNT
    ):
        raise ScoreMixAnalysisError("D0/D1 calibration cases overlap")

    static_policy = fit_frozen_static_policy_from_features(
        model_alias=model_alias,
        d0_run_id=d0_run_id,
        d1_run_id=d1_run_id,
        d0_features=d0_features,
        d1_features=d1_features,
    )
    if static_policy.get("schema_version") != STATIC_POLICY_SCHEMA:
        raise ScoreMixAnalysisError("static policy schema mismatch")
    cases_with_static = _attach_leave_one_out_static(cases)
    x_au = [float(case["x_AU"]) for case in cases_with_static]
    y_au = [float(case["y_AU"]) for case in cases_with_static]
    beta, eligible_count = _fit_nonnegative_robust_beta(x_au, y_au)
    expected_x_as = _mean(
        [float(case["x_AS"]) for case in cases_with_static]
    )
    expected_forecast = beta * expected_x_as
    residuals = [
        y - beta * x for x, y in zip(x_au, y_au)
    ]
    residual_envelope = max(
        NUMERIC_COMPARISON_TOLERANCE,
        statistics.median(abs(value) for value in residuals),
    )
    calibration_replay_envelope = max(
        NUMERIC_COMPARISON_TOLERANCE,
        max(float(case["no_op_absolute_progress"]) for case in cases_with_static),
    )
    bootstrap = _bootstrap_forecast(
        cases_with_static,
        seed=bootstrap_seed,
        replicates=bootstrap_replicates,
    )
    interval = [
        _percentile(bootstrap, 0.025),
        _percentile(bootstrap, 0.975),
    ]
    calibration_payload = [
        {
            "wave": case["wave"],
            "case_id": case["case_id"],
            "request_id": case["request_id"],
            "feature_hash": case["feature_hash"],
            "commitment_hash": case["commitment_hash"],
            "operational_c_distance": case["operational_c_distance"],
            "slopes": case["slopes"],
            "s_uniform": case["s_uniform"],
            "adaptive_weights": case["adaptive_weights"],
            "leave_one_out_static_weights": case[
                "leave_one_out_static_weights"
            ],
            "x_AU": case["x_AU"],
            "y_AU": case["y_AU"],
            "x_AS": case["x_AS"],
            "no_op_absolute_progress": case["no_op_absolute_progress"],
        }
        for case in cases_with_static
    ]
    calibration_hash = _sha256_bytes(
        _canonical_json(calibration_payload).encode("utf-8")
    )
    feature_only_action_hash = _sha256_bytes(
        _canonical_json(
            [
                {
                    "case_id": case["case_id"],
                    "adaptive_weights": case["adaptive_weights"],
                    "leave_one_out_static_weights": case[
                        "leave_one_out_static_weights"
                    ],
                }
                for case in cases_with_static
            ]
        ).encode("utf-8")
    )

    policy: dict[str, Any] = {
        "schema_version": FORECAST_POLICY_SCHEMA,
        "model_alias": model_alias,
        "static_policy_hash": static_policy["policy_hash"],
        "source_runs": {"d0": d0_run_id, "d1": d1_run_id},
        "beta": beta,
        "calibration_method": CALIBRATION_METHOD,
        "calibration_hash": calibration_hash,
        "confirmatory_outcomes_used": [],
    }
    policy["policy_hash"] = _sha256_bytes(
        _canonical_json(policy).encode("utf-8")
    )
    validate_forecast_policy(policy)

    analysis = {
        "schema_version": FORECAST_ANALYSIS_SCHEMA,
        "claim_status": "calibration_forecast_only_not_method_gain",
        "model_alias": model_alias,
        "source_runs": {"d0": d0_run_id, "d1": d1_run_id},
        "static_policy_hash": static_policy["policy_hash"],
        "forecast_policy_hash": policy["policy_hash"],
        "calibration_hash": calibration_hash,
        "calibration_method": CALIBRATION_METHOD,
        "calibration_case_count": CALIBRATION_CASE_COUNT,
        "calibration_replay_envelope": calibration_replay_envelope,
        "beta": {
            "value": beta,
            "constraint": "nonnegative zero-intercept",
            "eligible_x_AU_case_count": eligible_count,
            "x_floor": BETA_X_FLOOR,
        },
        "adaptive_static_forecast": {
            "mean_feature_only_x_AS": expected_x_as,
            "expected_realized_gap": expected_forecast,
            "case_bootstrap_interval_95": interval,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_replicates": bootstrap_replicates,
            "calibration_residual_envelope": residual_envelope,
        },
        "feature_only_action_hash": feature_only_action_hash,
        "case_diagnostics": calibration_payload,
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
    analysis["analysis_hash"] = _sha256_bytes(
        _canonical_json(analysis).encode("utf-8")
    )
    json.dumps(analysis, allow_nan=False)
    return analysis, policy, static_policy


def fit_calibration_forecast_from_runs(
    d0_run_directory: str | Path,
    d1_run_directory: str | Path,
    *,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load two independently validated sanitized runs and fit the forecast."""

    d0_root = Path(d0_run_directory).expanduser().resolve()
    d1_root = Path(d1_run_directory).expanduser().resolve()
    d0_report = analyze_score_mix_run(d0_root)
    d1_report, _d1_features_by_case, d1_manifest = _analyze_d1_internal(d1_root)
    if (
        d0_report.get("analysis_status") != "d0_descriptive_complete"
        or d0_report.get("artifact_validation", {}).get("valid") is not True
        or d0_report.get("artifact_validation", {}).get("panel_complete") is not True
        or d1_report.get("analysis_status") != "d1_descriptive_complete"
        or d1_report.get("artifact_validation", {}).get("valid") is not True
        or d1_report.get("artifact_validation", {}).get("panel_complete") is not True
    ):
        raise ScoreMixAnalysisError("D0/D1 is not independently valid and complete")
    d0_manifest = _read_json_object(d0_root / "manifest.json")
    _cross_validate_d0_d1(
        d0_report=d0_report,
        d0_manifest=d0_manifest,
        d1_report=d1_report,
        d1_manifest=d1_manifest,
    )
    if d0_report.get("model_alias") != d1_report.get("model_alias"):
        raise ScoreMixAnalysisError("D0/D1 model mismatch")

    def payloads(root: Path, name: str, run_id: str) -> list[dict[str, Any]]:
        return [
            row.payload for row in _read_jsonl(root / name, run_id=run_id)
        ]

    return fit_calibration_forecast(
        model_alias=str(d1_report["model_alias"]),
        d0_run_id=str(d0_report["run_id"]),
        d1_run_id=str(d1_report["run_id"]),
        d0_features=payloads(d0_root, "features.jsonl", str(d0_report["run_id"])),
        d0_actions=payloads(d0_root, "actions.jsonl", str(d0_report["run_id"])),
        d0_outcomes=payloads(d0_root, "outcomes.jsonl", str(d0_report["run_id"])),
        d1_features=payloads(d1_root, "features.jsonl", str(d1_report["run_id"])),
        d1_actions=payloads(d1_root, "actions.jsonl", str(d1_report["run_id"])),
        d1_outcomes=payloads(d1_root, "outcomes.jsonl", str(d1_report["run_id"])),
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the D0+D1 calibration-only adaptive-static forecast policy."
        )
    )
    parser.add_argument("d0_run_directory", type=Path)
    parser.add_argument("d1_run_directory", type=Path)
    parser.add_argument("--analysis-output", required=True, type=Path)
    parser.add_argument("--policy-output", required=True, type=Path)
    parser.add_argument(
        "--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        analysis, policy, _static_policy = fit_calibration_forecast_from_runs(
            args.d0_run_directory,
            args.d1_run_directory,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        if args.analysis_output.resolve() == args.policy_output.resolve():
            raise ScoreMixAnalysisError("forecast outputs must differ")
        _write_exclusive(
            args.analysis_output,
            (
                json.dumps(
                    analysis,
                    sort_keys=True,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        )
        _write_exclusive(
            args.policy_output,
            (
                json.dumps(
                    policy,
                    sort_keys=True,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        )
        print(
            json.dumps(
                {
                    "status": "calibration_forecast_complete",
                    "model_alias": policy["model_alias"],
                    "policy_hash": policy["policy_hash"],
                    "confirmatory_outcomes_used": [],
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ScoreMixAnalysisError) as exc:
        print(
            json.dumps(
                {
                    "status": "calibration_forecast_aborted",
                    "error_type": type(exc).__name__,
                    "raw_input_emitted": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
