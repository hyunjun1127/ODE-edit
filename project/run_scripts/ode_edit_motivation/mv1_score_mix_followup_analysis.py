"""Strict single-model analysis for the two precommitted MV-1 follow-ups.

The run analyzer reuses the C1 validator/statistics core with a fixed wave
identity.  The optional fold0+fold1 helper accepts *analysis JSON only*: it
never opens raw run artifacts, and it recomputes the 24-event summaries from
the sanitized case diagnostics rather than averaging fold summaries.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .manifests import DEFAULT_SELECTION_SEED
from .mv1_analysis import build_confirmatory_fold_manifest
from .mv1_score_mix_analysis import (
    NUMERIC_COMPARISON_TOLERANCE,
    _canonical_json,
    _finite,
    _is_full_hash,
    _read_json_object,
    _sha256_bytes,
    _write_exclusive,
    ScoreMixAnalysisError,
)
from .mv1_score_mix_confirmatory_analysis import (
    CONFIRMATORY_ANALYSIS_SCHEMA,
    CONFIRMATORY_CASE_COUNT,
    CONFIRMATORY_FOLD01_IDENTITY_SCHEMA,
    CONFIRMATORY_FOLD01_INPUT_SCHEMA,
    CONFIRMATORY_FOLD_COUNT,
    CONFIRMATORY_FOLD_SEED,
    CONFIRMATORY_OUTCOME_ACTIONS,
    CONFIRMATORY_RUN_IDS,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    ScoreMixAnalysisWaveLock,
    _effect_summary,
    _fmean,
    _json_safe,
    _pearson,
    analyze_score_mix_wave,
)
from .mv1_score_mix_followup import (
    FOLLOWUP_JOB_NAMES,
    FOLLOWUP_MODES,
    FOLLOWUP_RUN_IDS,
)


FOLLOWUP_ANALYSIS_SCHEMA = "ode-edit-mv1-score-mix-followup-analysis/v1"
FOLD01_AGGREGATE_SCHEMA = (
    "ode-edit-mv1-score-mix-fold01-aggregate-analysis/v1"
)
FOLLOWUP_BOOTSTRAP_SEED = DEFAULT_BOOTSTRAP_SEED
FOLLOWUP_BOOTSTRAP_REPLICATES = DEFAULT_BOOTSTRAP_REPLICATES
FOLD01_AGGREGATE_BOOTSTRAP_SEED = 20260731
FOLD01_AGGREGATE_BOOTSTRAP_REPLICATES = 4000
PRIMARY_ESTIMAND = "progress(score_mix)-progress(frozen_static_mix)"

FOLLOWUP_ANALYSIS_WAVE_LOCKS = MappingProxyType(
    {
        "fold1": ScoreMixAnalysisWaveLock(
            label="fold1",
            selected_split="confirmatory",
            fold=1,
            case_count=12,
            job_name=FOLLOWUP_JOB_NAMES["fold1"],
            run_ids=FOLLOWUP_RUN_IDS["fold1"],
            run_seed=17,
            require_canonical_selection_seed=True,
        ),
        "untouched": ScoreMixAnalysisWaveLock(
            label="untouched",
            selected_split="untouched",
            fold=None,
            case_count=20,
            job_name=FOLLOWUP_JOB_NAMES["untouched"],
            run_ids=FOLLOWUP_RUN_IDS["untouched"],
            run_seed=17,
            require_canonical_selection_seed=True,
        ),
    }
)

_BASE_REPORT_FIELDS = {
    "schema_version",
    "claim_status",
    "analysis_status",
    "run_id",
    "model_alias",
    "artifact_validation",
    "replay_envelope",
    "calibration_replay_envelope",
    "primary_adaptive_minus_frozen_static",
    "finite_panel_oracle_opportunity",
    "predicted_vs_realized",
    "calibration_residual_envelope",
    "case_diagnostics",
    "single_model_gate_inputs",
    "claim_boundary",
}
_C1_REPORT_FIELDS = _BASE_REPORT_FIELDS | {"c1_replay_envelope"}
_C1_AGGREGATE_INPUT_FIELDS = {
    "schema_version",
    "claim_status",
    "analysis_status",
    "c1_analysis",
    "aggregate_identity",
    "analysis_hash",
}
_C1_AGGREGATE_IDENTITY_FIELDS = {
    "schema_version",
    "selected_split",
    "fold",
    "fold_count",
    "case_count",
    "selection_seed",
    "confirmatory_case_ids",
    "selection_manifest_id",
    "fold_manifest_id",
    "selected_case_ids_sha256",
    "context_id",
    "provenance_id",
    "q",
    "outcome_action_order",
    "calibration_hash",
    "forecast_beta",
    "static_policy_hash",
    "forecast_policy_hash",
    "primary_estimand",
    "bootstrap_seed",
    "bootstrap_replicates",
    "c1_runtime_seed",
}
_FOLLOWUP_REPORT_FIELDS = _BASE_REPORT_FIELDS | {
    "wave_replay_envelope",
    "mode",
    "execution_lock",
    "model_single_run_gate_inputs",
}
_C1_ARTIFACT_VALIDATION_FIELDS = {
    "valid",
    "panel_complete",
    "error_codes",
    "exact_fold",
    "case_count",
    "arm_count_per_case",
    "static_policy_hash",
    "forecast_policy_hash",
    "outcome_firewall_and_receipts_valid",
    "equal_c_and_rollback_valid",
}
_FOLLOWUP_ARTIFACT_VALIDATION_FIELDS = _C1_ARTIFACT_VALIDATION_FIELDS | {
    "selection_manifest_id",
    "fold_manifest_id",
    "selected_case_ids_sha256",
    "context_id",
    "provenance_id",
    "q",
    "outcome_action_order",
    "calibration_hash",
    "forecast_beta",
    "run_seed",
    "primary_estimand",
    "bootstrap_seed",
    "bootstrap_replicates",
}
_CASE_DIAGNOSTIC_FIELDS = {
    "case_id",
    "adaptive_static_effect",
    "finite_panel_oracle_opportunity",
    "predicted_adaptive_static_gain",
    "prediction_error",
    "effect_above_replay_envelope",
}
_SINGLE_MODEL_GATE_FIELDS = {
    "clear_continue_input",
    "scientific_kill_input",
    "controller_pivot_input",
    "gray_input",
    "architecture_conditional_tolerance_input",
    "pair_level_decision_computed",
}
_MODEL_SINGLE_RUN_INPUT_FIELDS = {
    "replay_envelope",
    "primary",
    "oracle",
    "forecast_calibration",
    "pair_level_decision_computed",
}
_MODEL_FORECAST_CALIBRATION_FIELDS = {
    "predicted_vs_realized",
    "calibration_residual_envelope",
    "calibration_replay_envelope",
}
_PREDICTION_FIELDS = {
    "predicted_mean",
    "realized_mean",
    "mean_error",
    "mean_absolute_error",
    "median_absolute_error",
    "zero_intercept_realized_on_predicted_slope",
    "pearson",
    "positive_direction_concordance",
}


def _exact_int(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _exact_float(value: Any, expected: float) -> bool:
    return type(value) is float and value == expected


def _effect_summary_matches(
    actual: Any,
    expected: Mapping[str, Any],
) -> bool:
    if not isinstance(actual, Mapping) or set(actual) != set(expected):
        return False
    exact_int_fields = {
        "case_count",
        "positive_sign_count_above_replay_envelope",
        "bootstrap_seed",
        "bootstrap_replicates",
    }
    exact_float_fields = {
        "mean",
        "trimmed_mean_20pct",
        "median",
        "positive_sign_fraction_above_replay_envelope",
    }
    if any(
        type(actual.get(field)) is not int for field in exact_int_fields
    ) or any(
        type(actual.get(field)) is not float for field in exact_float_fields
    ):
        return False
    for field in (
        "paired_case_bootstrap_mean_ci_95",
        "observed_range",
    ):
        values = actual.get(field)
        if (
            not isinstance(values, list)
            or len(values) != 2
            or any(type(value) is not float for value in values)
        ):
            return False
    return dict(actual) == dict(expected)


def _prediction_matches(
    actual: Any,
    expected: Mapping[str, Any],
) -> bool:
    if not isinstance(actual, Mapping) or set(actual) != _PREDICTION_FIELDS:
        return False
    if any(
        type(actual.get(field)) is not float
        for field in (
            "predicted_mean",
            "realized_mean",
            "mean_error",
            "mean_absolute_error",
            "median_absolute_error",
            "positive_direction_concordance",
        )
    ):
        return False
    if any(
        actual.get(field) is not None
        and type(actual.get(field)) is not float
        for field in (
            "zero_intercept_realized_on_predicted_slope",
            "pearson",
        )
    ):
        return False
    return dict(actual) == dict(expected)


def _validated_c1_aggregate_input(
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the versioned C1 envelope required for cross-wave parity."""

    _json_safe(source, location="c1_fold01_aggregate_input")
    if (
        not isinstance(source, Mapping)
        or set(source) != _C1_AGGREGATE_INPUT_FIELDS
        or source.get("schema_version") != CONFIRMATORY_FOLD01_INPUT_SCHEMA
        or source.get("claim_status") != "c1_fold01_aggregate_input_only"
        or source.get("analysis_status")
        != "c1_fold01_aggregate_input_complete"
        or not _is_full_hash(source.get("analysis_hash"))
    ):
        raise ScoreMixAnalysisError("C1 aggregate-input envelope mismatch")
    hash_payload = {
        key: value for key, value in source.items() if key != "analysis_hash"
    }
    if source["analysis_hash"] != _sha256_bytes(
        _canonical_json(hash_payload).encode("utf-8")
    ):
        raise ScoreMixAnalysisError("C1 aggregate-input hash mismatch")
    c1_analysis = source.get("c1_analysis")
    identity = source.get("aggregate_identity")
    if not isinstance(c1_analysis, dict) or not isinstance(identity, dict):
        raise ScoreMixAnalysisError("C1 aggregate-input payload mismatch")
    if (
        set(identity) != _C1_AGGREGATE_IDENTITY_FIELDS
        or identity.get("schema_version")
        != CONFIRMATORY_FOLD01_IDENTITY_SCHEMA
        or identity.get("selected_split") != "confirmatory"
        or not _exact_int(identity.get("fold"), 0)
        or not _exact_int(
            identity.get("fold_count"),
            CONFIRMATORY_FOLD_COUNT,
        )
        or not _exact_int(
            identity.get("case_count"),
            CONFIRMATORY_CASE_COUNT,
        )
        or identity.get("selection_seed") != DEFAULT_SELECTION_SEED
        or not _is_full_hash(identity.get("selection_manifest_id"))
        or not _is_full_hash(identity.get("fold_manifest_id"))
        or not _is_full_hash(identity.get("selected_case_ids_sha256"))
        or not _is_full_hash(identity.get("context_id"))
        or not _is_full_hash(identity.get("provenance_id"))
        or not _exact_float(identity.get("q"), 1.0 / 256.0)
        or not isinstance(identity.get("outcome_action_order"), list)
        or tuple(identity["outcome_action_order"])
        != CONFIRMATORY_OUTCOME_ACTIONS
        or not _is_full_hash(identity.get("calibration_hash"))
        or type(identity.get("forecast_beta")) is not float
        or _finite(
            identity.get("forecast_beta"),
            name="c1_identity.forecast_beta",
        )
        < 0.0
        or not _is_full_hash(identity.get("static_policy_hash"))
        or not _is_full_hash(identity.get("forecast_policy_hash"))
        or identity.get("primary_estimand") != PRIMARY_ESTIMAND
        or not _exact_int(
            identity.get("bootstrap_seed"),
            FOLD01_AGGREGATE_BOOTSTRAP_SEED,
        )
        or not _exact_int(
            identity.get("bootstrap_replicates"),
            FOLD01_AGGREGATE_BOOTSTRAP_REPLICATES,
        )
        or identity.get("c1_runtime_seed") is not None
    ):
        raise ScoreMixAnalysisError("C1 aggregate identity mismatch")
    case_rows = c1_analysis.get("case_diagnostics")
    if not isinstance(case_rows, list):
        raise ScoreMixAnalysisError("C1 aggregate case IDs unavailable")
    case_ids = [row.get("case_id") for row in case_rows if isinstance(row, dict)]
    confirmatory_case_ids = identity.get("confirmatory_case_ids")
    if (
        not isinstance(confirmatory_case_ids, list)
        or len(confirmatory_case_ids) != 60
        or len(set(confirmatory_case_ids)) != 60
        or any(
            not isinstance(case_id, str) or not case_id
            for case_id in confirmatory_case_ids
        )
        or len(case_ids) != CONFIRMATORY_CASE_COUNT
        or any(
            not isinstance(case_id, str) or not case_id
            for case_id in case_ids
        )
        or identity["selected_case_ids_sha256"]
        != _sha256_bytes(_canonical_json(case_ids).encode("utf-8"))
    ):
        raise ScoreMixAnalysisError("C1 aggregate case-ID hash mismatch")
    folds = build_confirmatory_fold_manifest(
        confirmatory_case_ids,
        seed=CONFIRMATORY_FOLD_SEED,
        fold_count=CONFIRMATORY_FOLD_COUNT,
    )
    expected_fold0 = [
        case_id
        for case_id in confirmatory_case_ids
        if folds.fold_for(case_id) == 0
    ]
    if (
        folds.manifest_id != identity["fold_manifest_id"]
        or case_ids != expected_fold0
    ):
        raise ScoreMixAnalysisError("C1 aggregate exact fold0 mismatch")
    artifact = c1_analysis.get("artifact_validation")
    if (
        not isinstance(artifact, dict)
        or identity["static_policy_hash"]
        != artifact.get("static_policy_hash")
        or identity["forecast_policy_hash"]
        != artifact.get("forecast_policy_hash")
    ):
        raise ScoreMixAnalysisError("C1 aggregate nested policy mismatch")
    return dict(c1_analysis), dict(identity)


def followup_analysis_wave_lock(mode: str) -> ScoreMixAnalysisWaveLock:
    if not isinstance(mode, str) or mode not in FOLLOWUP_MODES:
        raise ScoreMixAnalysisError(
            "follow-up analysis mode must be exactly fold1 or untouched"
        )
    return FOLLOWUP_ANALYSIS_WAVE_LOCKS[mode]


def analyze_followup_run(
    run_directory: str | Path,
    *,
    mode: str,
    static_policy: Mapping[str, Any],
    forecast_policy: Mapping[str, Any],
    forecast_analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Emit one model's fixed single-run inputs, with no pair decision."""

    try:
        return analyze_score_mix_wave(
            run_directory,
            wave=followup_analysis_wave_lock(mode),
            static_policy=static_policy,
            forecast_policy=forecast_policy,
            forecast_analysis=forecast_analysis,
            bootstrap_seed=FOLLOWUP_BOOTSTRAP_SEED,
            bootstrap_replicates=FOLLOWUP_BOOTSTRAP_REPLICATES,
        )
    except ArithmeticError as exc:
        raise ScoreMixAnalysisError(
            "follow-up arithmetic overflow"
        ) from exc


def _validated_fold_report(
    report: Mapping[str, Any],
    *,
    fold: int,
) -> tuple[str, list[dict[str, Any]]]:
    _json_safe(report, location=f"fold{fold}_analysis")
    expected_fields = _C1_REPORT_FIELDS if fold == 0 else _FOLLOWUP_REPORT_FIELDS
    if set(report) != expected_fields:
        raise ScoreMixAnalysisError(f"fold{fold} analysis exact schema mismatch")
    expected_schema = (
        CONFIRMATORY_ANALYSIS_SCHEMA if fold == 0 else FOLLOWUP_ANALYSIS_SCHEMA
    )
    expected_status = (
        "confirmatory_single_model_complete"
        if fold == 0
        else "followup_single_model_complete"
    )
    expected_claim = (
        "single_model_confirmatory_gate_inputs_only"
        if fold == 0
        else "single_model_followup_gate_inputs_only"
    )
    if (
        report.get("schema_version") != expected_schema
        or report.get("analysis_status") != expected_status
        or report.get("claim_status") != expected_claim
        or (fold == 1 and report.get("mode") != "fold1")
    ):
        raise ScoreMixAnalysisError(f"fold{fold} analysis identity mismatch")
    artifact = report.get("artifact_validation")
    expected_artifact_fields = (
        _C1_ARTIFACT_VALIDATION_FIELDS
        if fold == 0
        else _FOLLOWUP_ARTIFACT_VALIDATION_FIELDS
    )
    if (
        not isinstance(artifact, Mapping)
        or set(artifact) != expected_artifact_fields
        or artifact.get("valid") is not True
        or artifact.get("panel_complete") is not True
        or artifact.get("error_codes") != []
        or not _exact_int(artifact.get("case_count"), 12)
        or not _exact_int(
            artifact.get("arm_count_per_case"),
            len(CONFIRMATORY_OUTCOME_ACTIONS),
        )
        or artifact.get("outcome_firewall_and_receipts_valid") is not True
        or artifact.get("equal_c_and_rollback_valid") is not True
        or not _is_full_hash(artifact.get("static_policy_hash"))
        or not _is_full_hash(artifact.get("forecast_policy_hash"))
    ):
        raise ScoreMixAnalysisError(f"fold{fold} artifact lock mismatch")
    outcome_action_order = artifact.get("outcome_action_order")
    if fold == 1 and (
        not _exact_float(artifact.get("q"), 1.0 / 256.0)
        or not isinstance(outcome_action_order, list)
        or tuple(outcome_action_order) != CONFIRMATORY_OUTCOME_ACTIONS
        or artifact.get("primary_estimand") != PRIMARY_ESTIMAND
        or not _exact_int(artifact.get("run_seed"), 17)
        or not _exact_int(
            artifact.get("bootstrap_seed"),
            FOLLOWUP_BOOTSTRAP_SEED,
        )
        or not _exact_int(
            artifact.get("bootstrap_replicates"),
            FOLLOWUP_BOOTSTRAP_REPLICATES,
        )
        or not _is_full_hash(artifact.get("selection_manifest_id"))
        or not _is_full_hash(artifact.get("fold_manifest_id"))
        or not _is_full_hash(artifact.get("selected_case_ids_sha256"))
        or not _is_full_hash(artifact.get("context_id"))
        or not _is_full_hash(artifact.get("provenance_id"))
        or not _is_full_hash(artifact.get("calibration_hash"))
        or type(artifact.get("forecast_beta")) is not float
        or _finite(
            artifact.get("forecast_beta"),
            name="fold1.forecast_beta",
        )
        < 0.0
    ):
        raise ScoreMixAnalysisError("fold1 extended artifact lock mismatch")
    expected_exact_fold = (
        "confirmatory[0::5]" if fold == 0 else "confirmatory_hash_fold_1"
    )
    if artifact.get("exact_fold") != expected_exact_fold:
        raise ScoreMixAnalysisError(f"fold{fold} exact-fold lock mismatch")
    model_alias = report.get("model_alias")
    if not isinstance(model_alias, str) or model_alias not in CONFIRMATORY_RUN_IDS:
        raise ScoreMixAnalysisError(f"fold{fold} model is outside the lock")
    expected_run_id = (
        CONFIRMATORY_RUN_IDS[model_alias]
        if fold == 0
        else FOLLOWUP_RUN_IDS["fold1"][model_alias]
    )
    if report.get("run_id") != expected_run_id:
        raise ScoreMixAnalysisError(f"fold{fold} run ID mismatch")
    replay_envelope = _finite(
        report.get("replay_envelope"),
        name=f"fold{fold}.replay_envelope",
    )
    if (
        type(report.get("replay_envelope")) is not float
        or replay_envelope < NUMERIC_COMPARISON_TOLERANCE
    ):
        raise ScoreMixAnalysisError(f"fold{fold} replay envelope is invalid")
    wave_replay_field = (
        "c1_replay_envelope" if fold == 0 else "wave_replay_envelope"
    )
    wave_replay = _finite(
        report.get(wave_replay_field),
        name=f"fold{fold}.{wave_replay_field}",
    )
    calibration_replay = _finite(
        report.get("calibration_replay_envelope"),
        name=f"fold{fold}.calibration_replay_envelope",
    )
    calibration_residual = _finite(
        report.get("calibration_residual_envelope"),
        name=f"fold{fold}.calibration_residual_envelope",
    )
    if (
        type(report.get(wave_replay_field)) is not float
        or type(report.get("calibration_replay_envelope")) is not float
        or type(report.get("calibration_residual_envelope")) is not float
        or wave_replay < NUMERIC_COMPARISON_TOLERANCE
        or calibration_replay < NUMERIC_COMPARISON_TOLERANCE
        or calibration_residual < 0.0
        or replay_envelope != max(wave_replay, calibration_replay)
    ):
        raise ScoreMixAnalysisError(f"fold{fold} replay-envelope parity mismatch")
    rows = report.get("case_diagnostics")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ScoreMixAnalysisError(f"fold{fold} case diagnostics mismatch")
    normalized: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    effects: list[float] = []
    oracles: list[float] = []
    predicted_values: list[float] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _CASE_DIAGNOSTIC_FIELDS:
            raise ScoreMixAnalysisError(
                f"fold{fold} case diagnostic exact schema mismatch"
            )
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ScoreMixAnalysisError(f"fold{fold} case identity mismatch")
        case_ids.add(case_id)
        effect = _finite(
            row.get("adaptive_static_effect"),
            name=f"fold{fold}.effect",
        )
        oracle = _finite(
            row.get("finite_panel_oracle_opportunity"),
            name=f"fold{fold}.oracle",
        )
        predicted = _finite(
            row.get("predicted_adaptive_static_gain"),
            name=f"fold{fold}.predicted",
        )
        error = _finite(
            row.get("prediction_error"),
            name=f"fold{fold}.prediction_error",
        )
        if (
            any(
                type(row.get(field)) is not float
                for field in (
                    "adaptive_static_effect",
                    "finite_panel_oracle_opportunity",
                    "predicted_adaptive_static_gain",
                    "prediction_error",
                )
            )
            or type(row.get("effect_above_replay_envelope")) is not bool
            or oracle < -NUMERIC_COMPARISON_TOLERANCE
            or error != effect - predicted
            or row.get("effect_above_replay_envelope")
            is not (effect > replay_envelope)
        ):
            raise ScoreMixAnalysisError(f"fold{fold} case estimand mismatch")
        normalized.append(dict(row))
        effects.append(effect)
        oracles.append(oracle)
        predicted_values.append(predicted)
    if fold == 1 and artifact["selected_case_ids_sha256"] != _sha256_bytes(
        _canonical_json(
            [row["case_id"] for row in normalized]
        ).encode("utf-8")
    ):
        raise ScoreMixAnalysisError("fold1 aggregate case-ID hash mismatch")
    expected_primary = _effect_summary(
        effects,
        expected_case_count=12,
        replay_envelope=replay_envelope,
        bootstrap_seed=FOLLOWUP_BOOTSTRAP_SEED,
        bootstrap_replicates=FOLLOWUP_BOOTSTRAP_REPLICATES,
    )
    expected_oracle = _effect_summary(
        oracles,
        expected_case_count=12,
        replay_envelope=replay_envelope,
        bootstrap_seed=FOLLOWUP_BOOTSTRAP_SEED + 1,
        bootstrap_replicates=FOLLOWUP_BOOTSTRAP_REPLICATES,
    )
    if (
        not _effect_summary_matches(
            report.get("primary_adaptive_minus_frozen_static"),
            expected_primary,
        )
        or not _effect_summary_matches(
            report.get("finite_panel_oracle_opportunity"),
            expected_oracle,
        )
    ):
        raise ScoreMixAnalysisError(f"fold{fold} summary/diagnostic mismatch")
    prediction = report.get("predicted_vs_realized")
    denominator = sum(value * value for value in predicted_values)
    expected_prediction = {
        "predicted_mean": _fmean(
            predicted_values,
            name=f"fold{fold}.predicted_mean",
        ),
        "realized_mean": _fmean(
            effects,
            name=f"fold{fold}.realized_mean",
        ),
        "mean_error": _fmean(
            (
                realized - predicted
                for predicted, realized in zip(predicted_values, effects)
            ),
            name=f"fold{fold}.mean_error",
        ),
        "mean_absolute_error": _fmean(
            (
                abs(realized - predicted)
                for predicted, realized in zip(predicted_values, effects)
            ),
            name=f"fold{fold}.mean_absolute_error",
        ),
        "median_absolute_error": _finite(
            statistics.median(
                abs(realized - predicted)
                for predicted, realized in zip(predicted_values, effects)
            ),
            name=f"fold{fold}.median_absolute_error",
        ),
        "zero_intercept_realized_on_predicted_slope": (
            sum(
                predicted * realized
                for predicted, realized in zip(predicted_values, effects)
            )
            / denominator
            if denominator > NUMERIC_COMPARISON_TOLERANCE
            else None
        ),
        "pearson": _pearson(predicted_values, effects),
        "positive_direction_concordance": sum(
            (predicted > 0.0) == (realized > replay_envelope)
            for predicted, realized in zip(predicted_values, effects)
        )
        / len(effects),
    }
    if not _prediction_matches(prediction, expected_prediction):
        raise ScoreMixAnalysisError(f"fold{fold} forecast calibration mismatch")
    if fold == 0:
        clear_input = bool(
            expected_primary["mean"] > replay_envelope
            and (
                expected_primary["trimmed_mean_20pct"] > replay_envelope
                or expected_primary["median"] > replay_envelope
                or expected_primary[
                    "positive_sign_count_above_replay_envelope"
                ]
                >= 7
            )
        )
        kill_input = bool(
            expected_primary["mean"] <= replay_envelope
            and expected_primary["trimmed_mean_20pct"] <= replay_envelope
            and expected_primary[
                "positive_sign_fraction_above_replay_envelope"
            ]
            <= 0.50
            and expected_oracle["mean"] <= replay_envelope
        )
        pivot_input = bool(
            expected_primary["mean"] <= replay_envelope
            and expected_primary["trimmed_mean_20pct"] <= replay_envelope
            and expected_primary[
                "positive_sign_fraction_above_replay_envelope"
            ]
            <= 0.50
            and expected_oracle["mean"] > replay_envelope
            and (
                expected_oracle["trimmed_mean_20pct"] > replay_envelope
                or expected_oracle["median"] > replay_envelope
                or expected_oracle[
                    "positive_sign_count_above_replay_envelope"
                ]
                >= 7
            )
        )
        gray_input = bool(
            not clear_input and not kill_input and not pivot_input
        )
    else:
        clear_input = False
        kill_input = False
        pivot_input = False
        gray_input = False
    expected_gate = {
        "clear_continue_input": clear_input,
        "scientific_kill_input": kill_input,
        "controller_pivot_input": pivot_input,
        "gray_input": gray_input,
        "architecture_conditional_tolerance_input": calibration_residual,
        "pair_level_decision_computed": False,
    }
    gate = report.get("single_model_gate_inputs")
    if (
        not isinstance(gate, Mapping)
        or set(gate) != _SINGLE_MODEL_GATE_FIELDS
        or any(
            type(gate.get(field)) is not bool
            for field in (
                "clear_continue_input",
                "scientific_kill_input",
                "controller_pivot_input",
                "gray_input",
                "pair_level_decision_computed",
            )
        )
        or type(
            gate.get("architecture_conditional_tolerance_input")
        )
        is not float
        or dict(gate) != expected_gate
    ):
        raise ScoreMixAnalysisError(f"fold{fold} pair-decision firewall mismatch")
    if fold == 1:
        execution = report.get("execution_lock")
        expected_execution = {
            "selected_split": "confirmatory",
            "fold": 1,
            "case_count": 12,
            "run_seed": 17,
            "q": 1.0 / 256.0,
            "outcome_action_order": list(CONFIRMATORY_OUTCOME_ACTIONS),
            "primary_estimand": PRIMARY_ESTIMAND,
            "bootstrap_seed": FOLLOWUP_BOOTSTRAP_SEED,
            "bootstrap_replicates": FOLLOWUP_BOOTSTRAP_REPLICATES,
            "pair_level_decision_computed": False,
        }
        if (
            not isinstance(execution, Mapping)
            or set(execution) != set(expected_execution)
            or not _exact_int(execution.get("fold"), 1)
            or not _exact_int(execution.get("case_count"), 12)
            or not _exact_int(execution.get("run_seed"), 17)
            or not _exact_float(execution.get("q"), 1.0 / 256.0)
            or not _exact_int(
                execution.get("bootstrap_seed"),
                FOLLOWUP_BOOTSTRAP_SEED,
            )
            or not _exact_int(
                execution.get("bootstrap_replicates"),
                FOLLOWUP_BOOTSTRAP_REPLICATES,
            )
            or type(
                execution.get("pair_level_decision_computed")
            )
            is not bool
            or dict(execution) != expected_execution
        ):
            raise ScoreMixAnalysisError("fold1 execution lock mismatch")
        model_inputs = report.get("model_single_run_gate_inputs")
        if (
            not isinstance(model_inputs, Mapping)
            or set(model_inputs) != _MODEL_SINGLE_RUN_INPUT_FIELDS
            or type(model_inputs.get("replay_envelope")) is not float
            or model_inputs.get("replay_envelope") != replay_envelope
            or not _effect_summary_matches(
                model_inputs.get("primary"),
                expected_primary,
            )
            or not _effect_summary_matches(
                model_inputs.get("oracle"),
                expected_oracle,
            )
            or model_inputs.get("pair_level_decision_computed") is not False
        ):
            raise ScoreMixAnalysisError("fold1 model gate-input parity mismatch")
        forecast_inputs = model_inputs.get("forecast_calibration")
        if (
            not isinstance(forecast_inputs, Mapping)
            or set(forecast_inputs) != _MODEL_FORECAST_CALIBRATION_FIELDS
            or not _prediction_matches(
                forecast_inputs.get("predicted_vs_realized"),
                expected_prediction,
            )
            or type(
                forecast_inputs.get("calibration_residual_envelope")
            )
            is not float
            or forecast_inputs.get("calibration_residual_envelope")
            != calibration_residual
            or type(
                forecast_inputs.get("calibration_replay_envelope")
            )
            is not float
            or forecast_inputs.get("calibration_replay_envelope")
            != calibration_replay
        ):
            raise ScoreMixAnalysisError(
                "fold1 forecast gate-input parity mismatch"
            )
    return str(model_alias), normalized


def _aggregate_fold01_analyses_validated(
    c1_analysis: Mapping[str, Any],
    fold1_analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the fixed 24-event model summary from sanitized analyses."""

    c1_report, c1_identity = _validated_c1_aggregate_input(c1_analysis)
    model0, fold0_rows = _validated_fold_report(c1_report, fold=0)
    model1, fold1_rows = _validated_fold_report(fold1_analysis, fold=1)
    if model0 != model1:
        raise ScoreMixAnalysisError("fold0/fold1 model mismatch")
    artifact0 = c1_report["artifact_validation"]
    artifact1 = fold1_analysis["artifact_validation"]
    parity_fields = (
        "static_policy_hash",
        "forecast_policy_hash",
    )
    if any(artifact0.get(field) != artifact1.get(field) for field in parity_fields):
        raise ScoreMixAnalysisError("fold0/fold1 locked identity mismatch")
    cross_wave_identity_fields = (
        "selection_manifest_id",
        "fold_manifest_id",
        "context_id",
        "provenance_id",
        "q",
        "outcome_action_order",
        "calibration_hash",
        "forecast_beta",
        "primary_estimand",
        "bootstrap_seed",
        "bootstrap_replicates",
    )
    if any(
        c1_identity.get(field) != artifact1.get(field)
        for field in cross_wave_identity_fields
    ):
        raise ScoreMixAnalysisError(
            "fold0/fold1 aggregate identity parity mismatch"
        )
    calibration_replay0 = _finite(
        c1_report.get("calibration_replay_envelope"),
        name="fold0.calibration_replay_envelope",
    )
    calibration_replay1 = _finite(
        fold1_analysis.get("calibration_replay_envelope"),
        name="fold1.calibration_replay_envelope",
    )
    calibration_residual0 = _finite(
        c1_report.get("calibration_residual_envelope"),
        name="fold0.calibration_residual_envelope",
    )
    calibration_residual1 = _finite(
        fold1_analysis.get("calibration_residual_envelope"),
        name="fold1.calibration_residual_envelope",
    )
    if (
        calibration_replay0 < NUMERIC_COMPARISON_TOLERANCE
        or calibration_residual0 < 0.0
        or calibration_replay0 != calibration_replay1
        or calibration_residual0 != calibration_residual1
    ):
        raise ScoreMixAnalysisError("fold0/fold1 forecast calibration mismatch")
    fold0_ids = [str(row["case_id"]) for row in fold0_rows]
    fold1_ids = [str(row["case_id"]) for row in fold1_rows]
    confirmatory_folds = build_confirmatory_fold_manifest(
        c1_identity["confirmatory_case_ids"],
        seed=CONFIRMATORY_FOLD_SEED,
        fold_count=CONFIRMATORY_FOLD_COUNT,
    )
    expected_fold1_ids = [
        case_id
        for case_id in c1_identity["confirmatory_case_ids"]
        if confirmatory_folds.fold_for(case_id) == 1
    ]
    if (
        fold1_ids != expected_fold1_ids
        or set(fold0_ids).intersection(fold1_ids)
        or len(set(fold0_ids + fold1_ids)) != 24
    ):
        raise ScoreMixAnalysisError("fold0/fold1 case panels are not disjoint")

    rows = [
        {"fold": fold, **row}
        for fold, source in ((0, fold0_rows), (1, fold1_rows))
        for row in source
    ]
    effects = [
        _finite(row["adaptive_static_effect"], name="aggregate.effect")
        for row in rows
    ]
    oracles = [
        _finite(
            row["finite_panel_oracle_opportunity"],
            name="aggregate.oracle",
        )
        for row in rows
    ]
    predicted = [
        _finite(
            row["predicted_adaptive_static_gain"],
            name="aggregate.predicted",
        )
        for row in rows
    ]
    replay_envelope = max(
        NUMERIC_COMPARISON_TOLERANCE,
        _finite(
            c1_report["replay_envelope"],
            name="aggregate.c1_replay_envelope",
        ),
        _finite(
            fold1_analysis["replay_envelope"],
            name="aggregate.fold1_replay_envelope",
        ),
    )
    for row, effect in zip(rows, effects):
        row["effect_above_replay_envelope"] = (
            effect > replay_envelope
        )
    primary = _effect_summary(
        effects,
        expected_case_count=24,
        replay_envelope=replay_envelope,
        bootstrap_seed=FOLD01_AGGREGATE_BOOTSTRAP_SEED,
        bootstrap_replicates=FOLD01_AGGREGATE_BOOTSTRAP_REPLICATES,
    )
    oracle = _effect_summary(
        oracles,
        expected_case_count=24,
        replay_envelope=replay_envelope,
        bootstrap_seed=FOLD01_AGGREGATE_BOOTSTRAP_SEED + 1,
        bootstrap_replicates=FOLD01_AGGREGATE_BOOTSTRAP_REPLICATES,
    )
    denominator = sum(value * value for value in predicted)
    realized_on_predicted_slope = (
        sum(left * right for left, right in zip(predicted, effects)) / denominator
        if denominator > NUMERIC_COMPARISON_TOLERANCE
        else None
    )
    forecast_calibration = {
        "predicted_mean": _fmean(
            predicted,
            name="aggregate.predicted_mean",
        ),
        "realized_mean": _fmean(
            effects,
            name="aggregate.realized_mean",
        ),
        "mean_error": _fmean(
            (
                realized - forecast
                for forecast, realized in zip(predicted, effects)
            ),
            name="aggregate.mean_error",
        ),
        "mean_absolute_error": _fmean(
            (
                abs(realized - forecast)
                for forecast, realized in zip(predicted, effects)
            ),
            name="aggregate.mean_absolute_error",
        ),
        "median_absolute_error": _finite(
            statistics.median(
                abs(realized - forecast)
                for forecast, realized in zip(predicted, effects)
            ),
            name="aggregate.median_absolute_error",
        ),
        "zero_intercept_realized_on_predicted_slope": realized_on_predicted_slope,
        "pearson": _pearson(predicted, effects),
        "positive_direction_concordance": sum(
            (forecast > 0.0) == (realized > replay_envelope)
            for forecast, realized in zip(predicted, effects)
        )
        / len(effects),
        "calibration_residual_envelope": c1_report[
            "calibration_residual_envelope"
        ],
        "calibration_replay_envelope": calibration_replay0,
    }
    clear_continue_input = bool(
        primary["mean"] > replay_envelope
        and (
            primary["trimmed_mean_20pct"] > replay_envelope
            or primary["median"] > replay_envelope
            or primary["positive_sign_count_above_replay_envelope"] >= 14
        )
    )
    scientific_kill_input = bool(
        primary["mean"] <= replay_envelope
        and primary["trimmed_mean_20pct"] <= replay_envelope
        and primary["positive_sign_count_above_replay_envelope"] <= 12
        and oracle["mean"] <= replay_envelope
    )
    controller_pivot_input = bool(
        primary["mean"] <= replay_envelope
        and primary["trimmed_mean_20pct"] <= replay_envelope
        and primary["positive_sign_count_above_replay_envelope"] <= 12
        and oracle["mean"] > replay_envelope
        and (
            oracle["trimmed_mean_20pct"] > replay_envelope
            or oracle["median"] > replay_envelope
            or oracle["positive_sign_count_above_replay_envelope"] >= 14
        )
    )
    report = {
        "schema_version": FOLD01_AGGREGATE_SCHEMA,
        "claim_status": "single_model_fold01_24_case_gate_inputs_only",
        "analysis_status": "fold01_aggregate_single_model_complete",
        "error_codes": [],
        "model_alias": model0,
        "source_run_ids": {
            "fold0": c1_report["run_id"],
            "fold1": fold1_analysis["run_id"],
        },
        "locked_identity": {
            "static_policy_hash": artifact0["static_policy_hash"],
            "forecast_policy_hash": artifact0["forecast_policy_hash"],
            "selection_manifest_id": c1_identity[
                "selection_manifest_id"
            ],
            "fold_manifest_id": c1_identity["fold_manifest_id"],
            "context_id": c1_identity["context_id"],
            "provenance_id": c1_identity["provenance_id"],
            "q": c1_identity["q"],
            "outcome_action_order": c1_identity[
                "outcome_action_order"
            ],
            "calibration_hash": c1_identity["calibration_hash"],
            "forecast_beta": c1_identity["forecast_beta"],
            "primary_estimand": c1_identity["primary_estimand"],
            "bootstrap_seed": c1_identity["bootstrap_seed"],
            "bootstrap_replicates": c1_identity[
                "bootstrap_replicates"
            ],
            "runtime_seed_evidence": {
                "fold0": c1_identity["c1_runtime_seed"],
                "fold1": artifact1["run_seed"],
            },
            "selected_case_ids_sha256": {
                "fold0": c1_identity["selected_case_ids_sha256"],
                "fold1": artifact1["selected_case_ids_sha256"],
            },
        },
        "case_ids": {
            "fold0": fold0_ids,
            "fold1": fold1_ids,
            "combined_case_count": 24,
        },
        "replay_envelope": replay_envelope,
        "primary_adaptive_minus_frozen_static": primary,
        "finite_panel_oracle_opportunity": oracle,
        "forecast_calibration": forecast_calibration,
        "case_diagnostics": rows,
        "model_gate_inputs": {
            "clear_sign_count_threshold": 14,
            "kill_sign_count_upper_bound": 12,
            "observed_positive_sign_count_above_replay_envelope": primary[
                "positive_sign_count_above_replay_envelope"
            ],
            "clear_continue_input": clear_continue_input,
            "scientific_kill_input": scientific_kill_input,
            "controller_pivot_input": controller_pivot_input,
            "gray_input": bool(
                not clear_continue_input
                and not scientific_kill_input
                and not controller_pivot_input
            ),
            "pair_level_decision_computed": False,
            "cross_model_decision_computed": False,
        },
        "bootstrap_lock": {
            "seed": FOLD01_AGGREGATE_BOOTSTRAP_SEED,
            "oracle_seed": FOLD01_AGGREGATE_BOOTSTRAP_SEED + 1,
            "replicates": FOLD01_AGGREGATE_BOOTSTRAP_REPLICATES,
            "statistical_unit": "24 unique events",
        },
        "claim_boundary": (
            "Fold0+fold1 sanitized single-model gate inputs only. No pair, "
            "cross-model, method-gain, MV-2, or ODE verdict is computed."
        ),
    }
    report["analysis_hash"] = _sha256_bytes(
        _canonical_json(report).encode("utf-8")
    )
    _json_safe(report, location="fold01_aggregate_analysis")
    json.dumps(report, allow_nan=False)
    return report


def aggregate_fold01_analyses(
    c1_analysis: Mapping[str, Any],
    fold1_analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a deterministic technical block for every invalid source pair."""

    try:
        if not isinstance(c1_analysis, Mapping) or not isinstance(
            fold1_analysis,
            Mapping,
        ):
            raise ScoreMixAnalysisError(
                "fold01 sources must be analysis JSON objects"
            )
        return _aggregate_fold01_analyses_validated(
            c1_analysis,
            fold1_analysis,
        )
    except (
        ArithmeticError,
        KeyError,
        RecursionError,
        TypeError,
        UnicodeError,
        ValueError,
        ScoreMixAnalysisError,
    ):
        report = {
            "schema_version": FOLD01_AGGREGATE_SCHEMA,
            "claim_status": "single_model_fold01_24_case_gate_inputs_only",
            "analysis_status": "technical_block_invalid_analysis_json",
            "error_codes": ["fold01_analysis_json_contract_mismatch"],
            "model_alias": None,
            "source_run_ids": None,
            "locked_identity": None,
            "case_ids": None,
            "replay_envelope": None,
            "primary_adaptive_minus_frozen_static": None,
            "finite_panel_oracle_opportunity": None,
            "forecast_calibration": None,
            "case_diagnostics": [],
            "model_gate_inputs": {
                "clear_sign_count_threshold": 14,
                "kill_sign_count_upper_bound": 12,
                "observed_positive_sign_count_above_replay_envelope": None,
                "clear_continue_input": None,
                "scientific_kill_input": None,
                "controller_pivot_input": None,
                "gray_input": None,
                "pair_level_decision_computed": False,
                "cross_model_decision_computed": False,
            },
            "bootstrap_lock": {
                "seed": FOLD01_AGGREGATE_BOOTSTRAP_SEED,
                "oracle_seed": FOLD01_AGGREGATE_BOOTSTRAP_SEED + 1,
                "replicates": FOLD01_AGGREGATE_BOOTSTRAP_REPLICATES,
                "statistical_unit": "24 unique events",
            },
            "claim_boundary": (
                "Invalid fold0/fold1 analysis JSON is a technical block. "
                "No scientific, pair, cross-model, MV-2, or ODE verdict is "
                "computed and no raw input is emitted."
            ),
        }
        report["analysis_hash"] = _sha256_bytes(
            _canonical_json(report).encode("utf-8")
        )
        return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze one fixed fold1/untouched MV-1 follow-up run."
    )
    parser.add_argument("run_directory", nargs="?", type=Path)
    parser.add_argument("--mode", choices=FOLLOWUP_MODES)
    parser.add_argument("--static-policy", type=Path)
    parser.add_argument("--forecast-policy", type=Path)
    parser.add_argument("--forecast-analysis", type=Path)
    parser.add_argument("--c1-analysis", type=Path)
    parser.add_argument("--fold1-analysis", type=Path)
    parser.add_argument("--analysis-output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        aggregate_mode = (
            args.c1_analysis is not None
            or args.fold1_analysis is not None
        )
        if aggregate_mode:
            if (
                args.c1_analysis is None
                or args.fold1_analysis is None
                or args.run_directory is not None
                or args.mode is not None
                or args.static_policy is not None
                or args.forecast_policy is not None
                or args.forecast_analysis is not None
            ):
                raise ScoreMixAnalysisError(
                    "aggregate mode requires only C1 and fold1 analysis JSON"
                )
            try:
                c1_source = _read_json_object(args.c1_analysis)
                fold1_source = _read_json_object(args.fold1_analysis)
            except (OSError, UnicodeError, ScoreMixAnalysisError):
                c1_source = {}
                fold1_source = {}
            report = aggregate_fold01_analyses(c1_source, fold1_source)
        else:
            if (
                args.run_directory is None
                or args.mode is None
                or args.static_policy is None
                or args.forecast_policy is None
                or args.forecast_analysis is None
            ):
                raise ScoreMixAnalysisError(
                    "single-run mode requires run, mode, and all frozen policies"
                )
            report = analyze_followup_run(
                args.run_directory,
                mode=args.mode,
                static_policy=_read_json_object(args.static_policy),
                forecast_policy=_read_json_object(args.forecast_policy),
                forecast_analysis=_read_json_object(args.forecast_analysis),
            )
        _write_exclusive(
            args.analysis_output,
            (
                json.dumps(
                    report,
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
                    "status": report["analysis_status"],
                    "pair_level_decision_computed": False,
                },
                sort_keys=True,
            )
        )
        return (
            0
            if report["analysis_status"]
            in {
                "followup_single_model_complete",
                "fold01_aggregate_single_model_complete",
            }
            else 2
        )
    except (OSError, ScoreMixAnalysisError) as exc:
        print(
            json.dumps(
                {
                    "status": "followup_analysis_aborted",
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
