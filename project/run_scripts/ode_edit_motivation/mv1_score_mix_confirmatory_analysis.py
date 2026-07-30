"""Fail-closed single-model analysis for MV-1 score-mix C1 fold 0.

The analyzer consumes only sanitized scalar/hash artifacts plus the independently
frozen static and calibration-forecast policies.  It verifies the exact
12-case fold, six-arm panel, durable pre-outcome commitment, equal-C primary
arms, rollback, policy parity, and file hashes before computing any effect.

It intentionally emits single-model gate *inputs*.  Cross-model continue,
architecture-conditional, gray, pivot, and kill decisions belong to a separate
pair-level post-run review.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .contracts import ContractError
from .manifests import DEFAULT_SELECTION_SEED
from .mv1_score_mix_analysis import (
    MODEL_ENVELOPES,
    NUMERIC_COMPARISON_TOLERANCE,
    PROBE_ACTIONS,
    SCORE_MIX_Q,
    SINGLE_ACTIONS,
    _FORBIDDEN_KEYS,
    _OUTCOME_FIELDS,
    _canonical_json,
    _contains_outcome_field,
    _expected_mix,
    _finite,
    _is_full_hash,
    _payload_hash,
    _read_json_object,
    _selection_manifest_id,
    _sha256_bytes,
    _sha256_file,
    _write_exclusive,
    ScoreMixAnalysisError,
)
from .mv1_score_mix_calibration_forecast import (
    CALIBRATION_CASE_COUNT,
    CALIBRATION_METHOD,
    FORECAST_ANALYSIS_SCHEMA,
    validate_forecast_policy,
)
from .mv1_score_mix_d1_analysis import (
    D0_RUN_IDS,
    D1_RUN_IDS,
    STATIC_POLICY_SCHEMA,
)
from .mv1_analysis import build_confirmatory_fold_manifest


CONFIRMATORY_ANALYSIS_SCHEMA = "ode-edit-mv1-score-mix-confirmatory-analysis/v1"
CONFIRMATORY_FOLD01_INPUT_SCHEMA = (
    "ode-edit-mv1-score-mix-c1-fold01-aggregate-input/v1"
)
CONFIRMATORY_FOLD01_IDENTITY_SCHEMA = (
    "ode-edit-mv1-score-mix-c1-fold01-aggregate-identity/v1"
)
CONFIRMATORY_MANIFEST_SCHEMA = "ode-edit-mv1-score-mix-confirmatory-manifest/v1"
CONFIRMATORY_SUMMARY_SCHEMA = "ode-edit-mv1-score-mix-confirmatory-summary/v1"
CONFIRMATORY_STREAM_SCHEMA = "ode-edit-mv1-score-mix-confirmatory/v1"
CONFIRMATORY_RECEIPT_SCHEMA = (
    "ode-edit-mv1-score-mix-confirmatory-receipt/v1"
)
CONFIRMATORY_JOB_NAME = "odeedit_mv1mix_c1_pair_v1"
CONFIRMATORY_CASE_COUNT = 12
CONFIRMATORY_FOLD_INDEX = 0
CONFIRMATORY_FOLD_COUNT = 5
CONFIRMATORY_FOLD_SEED = "ode-edit-mv1-score-mix-confirmatory-folds-v1"
BUNDLE_ID = "adaptive-score-mix-vs-frozen-static-mix"
DEFAULT_BOOTSTRAP_SEED = 20260731
DEFAULT_BOOTSTRAP_REPLICATES = 4000

ADAPTIVE_ACTION = "score_mix"
STATIC_ACTION = "frozen_static_mix"
UNIFORM_ACTION = "uniform"
GLOBAL_ALPHA_ACTION = "ordered_global_alpha"
NATIVE_ACTION = "native_memit_full"
REPLAY_ACTION = "no_op_replay"
CONFIRMATORY_OUTCOME_ACTIONS = (
    ADAPTIVE_ACTION,
    STATIC_ACTION,
    UNIFORM_ACTION,
    GLOBAL_ALPHA_ACTION,
    NATIVE_ACTION,
    REPLAY_ACTION,
)
MATCHED_C_ACTIONS = (
    ADAPTIVE_ACTION,
    STATIC_ACTION,
    UNIFORM_ACTION,
    GLOBAL_ALPHA_ACTION,
)
ORACLE_ACTIONS = MATCHED_C_ACTIONS
_BUDGET_VALIDATION_BY_ACTION = {
    ADAPTIVE_ACTION: "unit-c-remeasured-then-scaled/equal-c",
    STATIC_ACTION: "unit-c-remeasured-then-scaled/equal-c",
    UNIFORM_ACTION: "unit-c-remeasured-then-scaled/equal-c",
    GLOBAL_ALPHA_ACTION: (
        "native-c-remeasured-then-global-alpha-squared/equal-c"
    ),
    NATIVE_ACTION: "reference-native-full",
    REPLAY_ACTION: "reference-no-op",
}

CONFIRMATORY_RUN_IDS = {
    "llama3-8b-inst": "mv1mix_llama_c1_v1",
    "qwen2.5-7b-inst": "mv1mix_qwen_c1_v1",
}


@dataclass(frozen=True, slots=True)
class ScoreMixAnalysisWaveLock:
    """Exact artifact identity for C1 or one precommitted follow-up."""

    label: str
    selected_split: str
    fold: int | None
    case_count: int
    job_name: str
    run_ids: Mapping[str, str]
    run_seed: int | None
    require_canonical_selection_seed: bool

    def __post_init__(self) -> None:
        if (
            type(self.label) is not str
            or type(self.selected_split) is not str
            or type(self.case_count) is not int
            or (
                self.fold is not None
                and type(self.fold) is not int
            )
            or (
                self.run_seed is not None
                and type(self.run_seed) is not int
            )
            or type(self.require_canonical_selection_seed) is not bool
        ):
            raise ContractError("analysis wave slice types are invalid")
        normalized_runs = dict(self.run_ids)
        expected = {
            "c1": {
                "slice": ("confirmatory", 0, 12),
                "job_name": "odeedit_mv1mix_c1_pair_v1",
                "run_ids": {
                    "llama3-8b-inst": "mv1mix_llama_c1_v1",
                    "qwen2.5-7b-inst": "mv1mix_qwen_c1_v1",
                },
                "run_seed": None,
                "canonical_seed": False,
            },
            "fold1": {
                "slice": ("confirmatory", 1, 12),
                "job_name": "odeedit_mv1mix_fold1_pair_v1",
                "run_ids": {
                    "llama3-8b-inst": "mv1mix_llama_fold1_v1",
                    "qwen2.5-7b-inst": "mv1mix_qwen_fold1_v1",
                },
                "run_seed": 17,
                "canonical_seed": True,
            },
            "untouched": {
                "slice": ("untouched", None, 20),
                "job_name": "odeedit_mv1mix_untouched_pair_v1",
                "run_ids": {
                    "llama3-8b-inst": "mv1mix_llama_untouched_v1",
                    "qwen2.5-7b-inst": "mv1mix_qwen_untouched_v1",
                },
                "run_seed": 17,
                "canonical_seed": True,
            },
        }.get(self.label)
        if expected is None or (
            (self.selected_split, self.fold, self.case_count)
            != expected["slice"]
            or self.job_name != expected["job_name"]
            or normalized_runs != expected["run_ids"]
            or self.run_seed != expected["run_seed"]
            or self.require_canonical_selection_seed
            is not expected["canonical_seed"]
        ):
            raise ContractError("analysis wave identity differs from the precommit")
        object.__setattr__(
            self,
            "run_ids",
            MappingProxyType(normalized_runs),
        )


CONFIRMATORY_ANALYSIS_WAVE_LOCK = ScoreMixAnalysisWaveLock(
    label="c1",
    selected_split="confirmatory",
    fold=CONFIRMATORY_FOLD_INDEX,
    case_count=CONFIRMATORY_CASE_COUNT,
    job_name=CONFIRMATORY_JOB_NAME,
    run_ids=CONFIRMATORY_RUN_IDS,
    run_seed=None,
    require_canonical_selection_seed=False,
)


def _validated_analysis_wave_lock(
    wave: ScoreMixAnalysisWaveLock,
) -> ScoreMixAnalysisWaveLock:
    """Reject annotation bypasses and revalidate every canonical field."""

    if type(wave) is not ScoreMixAnalysisWaveLock:
        raise ContractError("analysis wave lock has an invalid runtime type")
    revalidated = ScoreMixAnalysisWaveLock(
        label=wave.label,
        selected_split=wave.selected_split,
        fold=wave.fold,
        case_count=wave.case_count,
        job_name=wave.job_name,
        run_ids=wave.run_ids,
        run_seed=wave.run_seed,
        require_canonical_selection_seed=(
            wave.require_canonical_selection_seed
        ),
    )
    if wave != revalidated:
        raise ContractError("analysis wave lock failed canonical revalidation")
    return wave


CONFIRMATORY_FEATURE_FIELDS = {
    "case_id",
    "request_id",
    "q",
    "q_label",
    "native_c_energy",
    "operational_c_distance",
    "probe_c_distance",
    "action_scores",
    "context_action_scores",
    "feature_policy",
    "static_policy_hash",
    "forecast_policy_hash",
    "adaptive_layer_weights",
    "static_layer_weights",
    "adaptive_predicted_score",
    "static_predicted_score",
    "predicted_adaptive_static_score_gap",
    "x_as",
    "forecast_beta",
    "predicted_adaptive_static_gain",
    "feature_hash",
}
CONFIRMATORY_ACTION_FIELDS = {
    "case_id",
    "request_id",
    "q",
    "q_label",
    "feature_hash",
    "bundle_id",
    "adaptive_action_id",
    "adaptive_layer_weights",
    "adaptive_predicted_score",
    "adaptive_actual_unit_c_energy",
    "adaptive_controller",
    "adaptive_controller_branch",
    "static_action_id",
    "static_layer_weights",
    "static_predicted_score",
    "static_actual_unit_c_energy",
    "static_policy_hash",
    "forecast_policy_hash",
    "forecast_beta",
    "predicted_adaptive_static_score_gap",
    "x_as",
    "predicted_adaptive_static_gain",
    "tie_policy",
    "commitment_hash",
}
CONFIRMATORY_EVENT_FIELDS = {
    "case_id",
    "request_id",
    "event_seed",
    "target_token_count",
    "native_c_energy",
    "q",
    "probe_direction_count",
    "feature_count",
    "commitment_count",
    "outcome_count",
    "direct_z_artifact_sha256",
    "direct_z_artifact_size",
    "static_policy_hash",
    "forecast_policy_hash",
    "action_receipt",
    "rollback_exact",
    "pass",
}
CONFIRMATORY_RECEIPT_FIELDS = {
    "schema_version",
    "case_id",
    "request_id",
    "q",
    "feature_hash",
    "commitment_hash",
    "bundle_id",
    "adaptive_action_id",
    "static_action_id",
    "adaptive_layer_weights",
    "static_layer_weights",
    "static_policy_hash",
    "forecast_policy_hash",
    "predicted_adaptive_static_score_gap",
    "x_as",
    "forecast_beta",
    "predicted_adaptive_static_gain",
    "durability",
    "outcomes_observed_before_commitment",
}
CONFIRMATORY_MANIFEST_FIELDS = {
    "schema_version",
    "run_id",
    "ode_edit_git",
    "slurm",
    "model",
    "hparams_relative_path",
    "selection",
    "selected_split",
    "confirmatory_folds",
    "selected_fold",
    "selected_case_ids",
    "selected_request_ids",
    "contexts",
    "provenance_id",
    "fixed_files",
    "q",
    "q_label",
    "probe_ratio",
    "probe_action_set",
    "controller_input_set",
    "outcome_action_set",
    "decision_policy",
    "action_receipt_policy",
    "utility_policy",
    "teacher_suffix_policy",
    "projector_policy",
    "covariance_policy",
    "direct_z_policy",
    "artifact_firewall",
    "static_policy",
    "forecast_policy",
    "forecast_policy_formula",
    "expected_counts",
}
CONFIRMATORY_SUMMARY_FIELDS = {
    "schema_version",
    "run_id",
    "model_alias",
    "slice",
    "slurm",
    "provenance_id",
    "selection_manifest_id",
    "context_id",
    "run_status",
    "abort_failure_type",
    "planned_case_count",
    "attempted_case_count",
    "not_run_due_to_abort_count",
    "pass_count",
    "failure_count",
    "all_pass",
    "case_results",
    "q",
    "probe_direction_count_per_case",
    "feature_count",
    "commitment_count",
    "outcome_count",
    "action_receipt_count",
    "expected_counts_exact",
    "all_rollbacks_exact",
    "covariance_files_loaded",
    "projector_files_loaded",
    "direct_z_artifact_count",
    "resource",
    "artifacts",
    "git_output_written",
    "static_policy_hash",
    "forecast_policy_hash",
    "forecast_beta",
    "stream_sequences",
}

_STATIC_POLICY_FIELDS = {
    "schema_version",
    "model_alias",
    "source_runs",
    "source_slices",
    "feature_case_count",
    "feature_panel_sha256",
    "fit_numeric_inputs",
    "outcome_fields_used",
    "sbar_single_layer_slopes",
    "layer_weights",
    "predicted_mean_score",
    "controller",
    "controller_branch",
    "tie_policy",
    "weight_l2_norm",
    "claim_boundary",
    "policy_hash",
}
_FORECAST_ANALYSIS_FIELDS = {
    "schema_version",
    "claim_status",
    "model_alias",
    "source_runs",
    "static_policy_hash",
    "forecast_policy_hash",
    "calibration_hash",
    "calibration_method",
    "calibration_case_count",
    "calibration_replay_envelope",
    "beta",
    "adaptive_static_forecast",
    "feature_only_action_hash",
    "case_diagnostics",
    "information_firewall",
    "claim_boundary",
    "analysis_hash",
}
_FORECAST_BETA_FIELDS = {
    "value",
    "constraint",
    "eligible_x_AU_case_count",
    "x_floor",
}
_FORECAST_SUMMARY_FIELDS = {
    "mean_feature_only_x_AS",
    "expected_realized_gap",
    "case_bootstrap_interval_95",
    "bootstrap_seed",
    "bootstrap_replicates",
    "calibration_residual_envelope",
}
_FORECAST_CASE_FIELDS = {
    "wave",
    "case_id",
    "request_id",
    "feature_hash",
    "commitment_hash",
    "operational_c_distance",
    "slopes",
    "s_uniform",
    "adaptive_weights",
    "leave_one_out_static_weights",
    "x_AU",
    "y_AU",
    "x_AS",
    "no_op_absolute_progress",
}
_STREAM_FILES = (
    "features.jsonl",
    "actions.jsonl",
    "outcomes.jsonl",
    "events.jsonl",
)
_STREAM_EVENTS = {
    "features.jsonl": "mv1mix_confirmatory_feature",
    "actions.jsonl": "mv1mix_confirmatory_action_commitment",
    "outcomes.jsonl": "mv1mix_confirmatory_outcome",
    "events.jsonl": "mv1mix_confirmatory_case",
}
_HASHED_FILES = ("manifest.json", *_STREAM_FILES)


@dataclass(frozen=True, slots=True)
class ConfirmatoryStreamRow:
    event: str
    payload: dict[str, Any]
    line_number: int


def _json_safe(value: Any, *, location: str = "root") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ScoreMixAnalysisError(f"non-finite scalar at {location}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _json_safe(item, location=f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ScoreMixAnalysisError(f"non-string key at {location}")
            if key.lower() in _FORBIDDEN_KEYS:
                raise ScoreMixAnalysisError(
                    f"forbidden raw field at {location}.{key}"
                )
            _json_safe(item, location=f"{location}.{key}")
        return
    raise ScoreMixAnalysisError(f"non-JSON value at {location}")


def _read_stream(path: Path, *, run_id: str) -> list[ConfirmatoryStreamRow]:
    if path.is_symlink() or not path.is_file():
        raise ScoreMixAnalysisError(f"required stream unavailable: {path.name}")
    rows: list[ConfirmatoryStreamRow] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = json.loads(line)
                _json_safe(raw, location=f"{path.name}:{line_number}")
                if not isinstance(raw, dict) or set(raw) != {
                    "schema_version",
                    "run_id",
                    "sequence",
                    "recorded_at",
                    "event",
                    "payload",
                }:
                    raise ScoreMixAnalysisError("confirmatory stream envelope mismatch")
                if (
                    raw.get("schema_version") != CONFIRMATORY_STREAM_SCHEMA
                    or raw.get("run_id") != run_id
                    or raw.get("sequence") != line_number - 1
                    or raw.get("event") != _STREAM_EVENTS[path.name]
                    or not isinstance(raw.get("recorded_at"), str)
                    or not raw["recorded_at"]
                    or not isinstance(raw.get("payload"), dict)
                ):
                    raise ScoreMixAnalysisError("confirmatory stream identity mismatch")
                rows.append(
                    ConfirmatoryStreamRow(
                        event=str(raw["event"]),
                        payload=dict(raw["payload"]),
                        line_number=line_number,
                    )
                )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScoreMixAnalysisError(
            f"cannot parse confirmatory stream {path.name}"
        ) from exc
    return rows


def _close(left: float, right: float, *, tolerance: float = 3e-5) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ScoreMixAnalysisError("percentile requires finite values")
    ordered = sorted(_finite(value, name="percentile") for value in values)
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return _finite(
        ordered[low] * (1.0 - fraction) + ordered[high] * fraction,
        name="percentile_result",
    )


def _fmean(values: Sequence[float] | Any, *, name: str) -> float:
    """Return a finite mean and normalize finite-input arithmetic overflow."""

    try:
        result = statistics.fmean(values)
    except ArithmeticError as exc:
        raise ScoreMixAnalysisError(f"{name} arithmetic overflow") from exc
    return _finite(result, name=name)


def _trimmed_mean(values: Sequence[float], fraction: float = 0.2) -> float:
    if not values:
        raise ScoreMixAnalysisError("trimmed mean requires finite values")
    ordered = sorted(_finite(value, name="trimmed_value") for value in values)
    trim = math.floor(len(ordered) * fraction)
    retained = ordered[trim : len(ordered) - trim]
    if not retained:
        raise ScoreMixAnalysisError("trimmed mean removed all values")
    return _fmean(retained, name="trimmed_mean")


def _effect_summary(
    values: Sequence[float],
    *,
    expected_case_count: int = CONFIRMATORY_CASE_COUNT,
    replay_envelope: float,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    if len(values) != expected_case_count:
        raise ScoreMixAnalysisError("confirmatory effect denominator mismatch")
    if bootstrap_replicates < 100:
        raise ScoreMixAnalysisError("at least 100 bootstrap replicates are required")
    finite = [_finite(value, name="effect") for value in values]
    generator = random.Random(bootstrap_seed)
    boot = [
        _fmean(
            (
                finite[generator.randrange(len(finite))]
                for _ in finite
            ),
            name="bootstrap_mean",
        )
        for _ in range(bootstrap_replicates)
    ]
    median = _finite(statistics.median(finite), name="effect_median")
    return {
        "case_count": len(finite),
        "mean": _fmean(finite, name="effect_mean"),
        "trimmed_mean_20pct": _trimmed_mean(finite),
        "median": median,
        "positive_sign_count_above_replay_envelope": sum(
            value > replay_envelope for value in finite
        ),
        "positive_sign_fraction_above_replay_envelope": sum(
            value > replay_envelope for value in finite
        )
        / len(finite),
        "paired_case_bootstrap_mean_ci_95": [
            _percentile(boot, 0.025),
            _percentile(boot, 0.975),
        ],
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "observed_range": [min(finite), max(finite)],
    }


def _validate_static_policy(
    policy: Mapping[str, Any],
    *,
    model_alias: str,
) -> dict[str, Any]:
    if set(policy) != _STATIC_POLICY_FIELDS:
        raise ScoreMixAnalysisError("static policy exact schema mismatch")
    if (
        policy.get("schema_version") != STATIC_POLICY_SCHEMA
        or policy.get("model_alias") != model_alias
        or policy.get("source_runs")
        != {"d0": D0_RUN_IDS[model_alias], "d1": D1_RUN_IDS[model_alias]}
        or policy.get("outcome_fields_used") != []
        or policy.get("feature_case_count") != 17
    ):
        raise ScoreMixAnalysisError("static policy identity/firewall mismatch")
    weights = policy.get("layer_weights")
    if not isinstance(weights, dict) or set(weights) != set(SINGLE_ACTIONS):
        raise ScoreMixAnalysisError("static policy weight panel mismatch")
    finite_weights = {
        action: _finite(weights[action], name=f"static_weight:{action}")
        for action in SINGLE_ACTIONS
    }
    if (
        any(value < 0.0 for value in finite_weights.values())
        or not _close(
            sum(value * value for value in finite_weights.values()),
            1.0,
            tolerance=2e-7,
        )
    ):
        raise ScoreMixAnalysisError("static policy weight norm mismatch")
    stripped = dict(policy)
    observed = stripped.pop("policy_hash", None)
    expected = _sha256_bytes(_canonical_json(stripped).encode("utf-8"))
    if observed != expected:
        raise ScoreMixAnalysisError("static policy hash mismatch")
    return json.loads(_canonical_json(policy))


def _validate_forecast_analysis(
    analysis: Mapping[str, Any],
    *,
    model_alias: str,
    static_policy: Mapping[str, Any],
    forecast_policy: Mapping[str, Any],
) -> tuple[dict[str, Any], float, float]:
    """Validate the exact, self-hashed D0+D1 calibration analysis."""

    if set(analysis) != _FORECAST_ANALYSIS_FIELDS:
        raise ScoreMixAnalysisError("forecast analysis exact schema mismatch")
    expected_source_runs = {
        "d0": D0_RUN_IDS[model_alias],
        "d1": D1_RUN_IDS[model_alias],
    }
    if (
        analysis.get("schema_version") != FORECAST_ANALYSIS_SCHEMA
        or analysis.get("claim_status")
        != "calibration_forecast_only_not_method_gain"
        or analysis.get("model_alias") != model_alias
        or analysis.get("source_runs") != expected_source_runs
        or analysis.get("static_policy_hash") != static_policy.get("policy_hash")
        or analysis.get("forecast_policy_hash")
        != forecast_policy.get("policy_hash")
        or analysis.get("calibration_hash")
        != forecast_policy.get("calibration_hash")
        or analysis.get("calibration_method") != CALIBRATION_METHOD
        or analysis.get("calibration_method")
        != forecast_policy.get("calibration_method")
        or analysis.get("calibration_case_count") != CALIBRATION_CASE_COUNT
        or not _is_full_hash(analysis.get("feature_only_action_hash"))
        or analysis.get("information_firewall")
        != {
            "static_weights_use_outcomes": False,
            "adaptive_weights_use_outcomes": False,
            "beta_uses_calibration_adaptive_uniform_outcomes": True,
            "confirmatory_outcomes_used": [],
        }
        or analysis.get("claim_boundary")
        != (
            "D0+D1 calibration-conditional adaptive-minus-static forecast; "
            "not a confirmatory effect, method gain, GO, MV-2, or ODE claim"
        )
    ):
        raise ScoreMixAnalysisError("forecast analysis identity/firewall mismatch")

    beta_payload = analysis.get("beta")
    if not isinstance(beta_payload, dict) or set(beta_payload) != _FORECAST_BETA_FIELDS:
        raise ScoreMixAnalysisError("forecast analysis beta schema mismatch")
    beta = _finite(beta_payload.get("value"), name="forecast_analysis.beta")
    eligible_count = beta_payload.get("eligible_x_AU_case_count")
    if (
        not _close(
            beta,
            _finite(forecast_policy.get("beta"), name="forecast_policy.beta"),
            tolerance=2e-7,
        )
        or beta < 0.0
        or beta_payload.get("constraint") != "nonnegative zero-intercept"
        or not isinstance(eligible_count, int)
        or isinstance(eligible_count, bool)
        or not 0 <= eligible_count <= CALIBRATION_CASE_COUNT
        or not _close(
            _finite(beta_payload.get("x_floor"), name="forecast_analysis.x_floor"),
            NUMERIC_COMPARISON_TOLERANCE,
            tolerance=0.0,
        )
    ):
        raise ScoreMixAnalysisError("forecast analysis beta parity mismatch")

    forecast_summary = analysis.get("adaptive_static_forecast")
    if (
        not isinstance(forecast_summary, dict)
        or set(forecast_summary) != _FORECAST_SUMMARY_FIELDS
    ):
        raise ScoreMixAnalysisError("forecast analysis summary schema mismatch")
    interval = forecast_summary.get("case_bootstrap_interval_95")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ScoreMixAnalysisError("forecast interval schema mismatch")
    interval_values = [
        _finite(value, name="forecast_interval") for value in interval
    ]
    residual_envelope = _finite(
        forecast_summary.get("calibration_residual_envelope"),
        name="calibration_residual_envelope",
    )
    bootstrap_seed = forecast_summary.get("bootstrap_seed")
    bootstrap_replicates = forecast_summary.get("bootstrap_replicates")
    for field in (
        "mean_feature_only_x_AS",
        "expected_realized_gap",
    ):
        _finite(forecast_summary.get(field), name=f"forecast_summary.{field}")
    if (
        interval_values[0] > interval_values[1]
        or residual_envelope < 0.0
        or not isinstance(bootstrap_seed, int)
        or isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_replicates, int)
        or isinstance(bootstrap_replicates, bool)
        or bootstrap_replicates < 100
    ):
        raise ScoreMixAnalysisError("forecast interval/residual semantics mismatch")

    cases = analysis.get("case_diagnostics")
    if not isinstance(cases, list) or len(cases) != CALIBRATION_CASE_COUNT:
        raise ScoreMixAnalysisError("forecast calibration case count mismatch")
    no_op_values: list[float] = []
    case_ids_seen: set[str] = set()
    request_ids: set[str] = set()
    wave_counts = {"d0": 0, "d1": 0}
    for case in cases:
        if not isinstance(case, dict) or set(case) != _FORECAST_CASE_FIELDS:
            raise ScoreMixAnalysisError("forecast calibration case schema mismatch")
        wave = case.get("wave")
        case_id = case.get("case_id")
        request_id = case.get("request_id")
        if (
            wave not in wave_counts
            or not isinstance(case_id, str)
            or not case_id
            or not _is_full_hash(request_id)
            or not _is_full_hash(case.get("feature_hash"))
            or not _is_full_hash(case.get("commitment_hash"))
            or case_id in case_ids_seen
            or request_id in request_ids
        ):
            raise ScoreMixAnalysisError("forecast calibration case identity mismatch")
        case_ids_seen.add(case_id)
        request_ids.add(str(request_id))
        wave_counts[str(wave)] += 1
        if (
            not isinstance(case.get("slopes"), dict)
            or set(case["slopes"]) != set(SINGLE_ACTIONS)
            or not isinstance(case.get("adaptive_weights"), dict)
            or set(case["adaptive_weights"]) != set(SINGLE_ACTIONS)
            or not isinstance(case.get("leave_one_out_static_weights"), dict)
            or set(case["leave_one_out_static_weights"]) != set(SINGLE_ACTIONS)
        ):
            raise ScoreMixAnalysisError("forecast calibration panel mismatch")
        for panel in (
            case["slopes"],
            case["adaptive_weights"],
            case["leave_one_out_static_weights"],
        ):
            for value in panel.values():
                _finite(value, name="forecast_calibration_panel_value")
        distance = _finite(
            case.get("operational_c_distance"),
            name="forecast_calibration_distance",
        )
        if distance <= 0.0:
            raise ScoreMixAnalysisError("forecast calibration distance invalid")
        for field in ("s_uniform", "x_AU", "y_AU", "x_AS"):
            _finite(case.get(field), name=f"forecast_calibration.{field}")
        no_op = _finite(
            case.get("no_op_absolute_progress"),
            name="forecast_calibration.no_op_absolute_progress",
        )
        if no_op < 0.0:
            raise ScoreMixAnalysisError("forecast calibration no-op is not absolute")
        no_op_values.append(no_op)
    if wave_counts != {"d0": 5, "d1": 12}:
        raise ScoreMixAnalysisError("forecast calibration wave count mismatch")

    expected_calibration_hash = _sha256_bytes(
        _canonical_json(cases).encode("utf-8")
    )
    calibration_replay_envelope = max(
        NUMERIC_COMPARISON_TOLERANCE,
        max(no_op_values),
    )
    if (
        analysis.get("calibration_hash") != expected_calibration_hash
        or not _close(
            _finite(
                analysis.get("calibration_replay_envelope"),
                name="calibration_replay_envelope",
            ),
            calibration_replay_envelope,
            tolerance=2e-7,
        )
        or calibration_replay_envelope < 0.0
    ):
        raise ScoreMixAnalysisError("forecast calibration hash/replay mismatch")

    stripped = dict(analysis)
    observed_hash = stripped.pop("analysis_hash", None)
    expected_hash = _sha256_bytes(_canonical_json(stripped).encode("utf-8"))
    if not _is_full_hash(observed_hash) or observed_hash != expected_hash:
        raise ScoreMixAnalysisError("forecast analysis hash mismatch")
    return (
        json.loads(_canonical_json(analysis)),
        residual_envelope,
        calibration_replay_envelope,
    )


def _dot(weights: Mapping[str, Any], scores: Mapping[str, Any]) -> float:
    return sum(
        _finite(weights[action], name=f"weight:{action}")
        * _finite(scores[action], name=f"score:{action}")
        for action in SINGLE_ACTIONS
    )


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = _fmean(left, name="pearson_left_mean")
    mean_right = _fmean(right, name="pearson_right_mean")
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    denominator = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denominator <= NUMERIC_COMPARISON_TOLERANCE:
        return None
    return _finite(
        sum(
            left_value * right_value
            for left_value, right_value in zip(centered_left, centered_right)
        )
        / denominator,
        name="pearson",
    )


def _validate_manifest_and_policies(
    *,
    wave: ScoreMixAnalysisWaveLock,
    root: Path,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    static_policy: Mapping[str, Any],
    forecast_policy: Mapping[str, Any],
    errors: list[str],
) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
    if set(manifest) != CONFIRMATORY_MANIFEST_FIELDS:
        errors.append("manifest_exact_schema_mismatch")
    if set(summary) != CONFIRMATORY_SUMMARY_FIELDS:
        errors.append("summary_exact_schema_mismatch")
    if manifest.get("schema_version") != CONFIRMATORY_MANIFEST_SCHEMA:
        errors.append("manifest_schema_version_mismatch")
    if summary.get("schema_version") != CONFIRMATORY_SUMMARY_SCHEMA:
        errors.append("summary_schema_version_mismatch")

    run_id = manifest.get("run_id")
    model = manifest.get("model")
    model_alias = model.get("model_alias") if isinstance(model, dict) else None
    envelope = MODEL_ENVELOPES.get(model_alias)
    if envelope is None:
        errors.append("model_outside_confirmatory_envelope")
    else:
        if run_id != wave.run_ids[model_alias]:
            errors.append("run_id_outside_confirmatory_envelope")
        if (
            model.get("repository_id") != envelope["repository_id"]
            or model.get("revision") != envelope["revision"]
            or model.get("dtype") != "torch.float32"
            or model.get("device") != "cuda:0"
            or model.get("offline") is not True
        ):
            errors.append("fixed_model_identity_mismatch")
    if not isinstance(run_id, str) or root.name != run_id:
        errors.append("run_directory_identity_mismatch")
    if (
        summary.get("run_id") != run_id
        or summary.get("model_alias") != model_alias
    ):
        errors.append("summary_run_model_mismatch")

    if manifest.get("selected_split") != wave.selected_split:
        errors.append("confirmatory_split_mismatch")
    if (
        manifest.get("q") != SCORE_MIX_Q
        or manifest.get("q_label") != "q_1_256"
        or summary.get("q") != SCORE_MIX_Q
        or manifest.get("probe_ratio") != 0.25
        or tuple(manifest.get("probe_action_set", ())) != PROBE_ACTIONS
        or tuple(manifest.get("controller_input_set", ())) != SINGLE_ACTIONS
        or tuple(manifest.get("outcome_action_set", ()))
        != CONFIRMATORY_OUTCOME_ACTIONS
    ):
        errors.append("confirmatory_action_or_q_policy_mismatch")
    expected_policy_text = {
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
    }
    if any(
        manifest.get(field) != expected
        for field, expected in expected_policy_text.items()
    ):
        errors.append("manifest_firewall_or_execution_policy_mismatch")

    selection = manifest.get("selection")
    case_ids = tuple(manifest.get("selected_case_ids", ()))
    request_ids = tuple(manifest.get("selected_request_ids", ()))
    if not isinstance(selection, dict):
        errors.append("selection_manifest_missing")
    else:
        split = selection.get("case_ids")
        calibration = (
            tuple(split.get("calibration", ())) if isinstance(split, dict) else ()
        )
        confirmatory = (
            tuple(split.get("confirmatory", ())) if isinstance(split, dict) else ()
        )
        untouched = (
            tuple(split.get("untouched", ())) if isinstance(split, dict) else ()
        )
        try:
            fold_manifest = build_confirmatory_fold_manifest(
                confirmatory,
                seed=CONFIRMATORY_FOLD_SEED,
                fold_count=CONFIRMATORY_FOLD_COUNT,
            )
            if wave.selected_split == "confirmatory":
                expected = tuple(
                    case_id
                    for case_id in confirmatory
                    if fold_manifest.fold_for(case_id) == wave.fold
                )
            else:
                expected = untouched
            expected_fold_payload = fold_manifest.to_dict()
            expected_selected_fold = {
                "label": wave.label,
                "fold": wave.fold,
                "count": wave.case_count,
                "case_ids": list(expected),
                **(
                    {"run_seed": wave.run_seed}
                    if wave.run_seed is not None
                    else {}
                ),
            }
            expected_summary_slice = {
                "label": wave.label,
                "split": wave.selected_split,
                "fold": wave.fold,
                "fold_count": CONFIRMATORY_FOLD_COUNT,
                "count": wave.case_count,
                "fold_manifest_id": fold_manifest.manifest_id,
                **(
                    {"run_seed": wave.run_seed}
                    if wave.run_seed is not None
                    else {}
                ),
            }
        except ContractError:
            expected = ()
            expected_fold_payload = None
            expected_selected_fold = None
            expected_summary_slice = None
            errors.append("confirmatory_fold_construction_failure")
        if (
            len(calibration) != 20
            or len(confirmatory) != 60
            or len(untouched) != 20
            or len(expected) != wave.case_count
            or case_ids != expected
            or set(calibration).intersection(confirmatory)
            or set(calibration).intersection(untouched)
            or set(confirmatory).intersection(untouched)
            or (
                wave.selected_split == "confirmatory"
                and set(case_ids).intersection((*calibration, *untouched))
            )
            or (
                wave.selected_split == "untouched"
                and set(case_ids).intersection((*calibration, *confirmatory))
            )
        ):
            errors.append("confirmatory_exact_fold_or_disjointness_mismatch")
        if (
            manifest.get("confirmatory_folds") != expected_fold_payload
            or manifest.get("selected_fold") != expected_selected_fold
            or summary.get("slice") != expected_summary_slice
        ):
            errors.append("confirmatory_fold_manifest_or_summary_mismatch")
        expected_order_hash = _sha256_bytes(
            _canonical_json(list(calibration + confirmatory + untouched)).encode(
                "utf-8"
            )
        )
        expected_split_hash = _sha256_bytes(
            _canonical_json(
                {
                    "calibration": list(calibration),
                    "confirmatory": list(confirmatory),
                    "untouched": list(untouched),
                }
            ).encode("utf-8")
        )
        if (
            selection.get("order_hash") != expected_order_hash
            or selection.get("split_hash") != expected_split_hash
            or selection.get("manifest_id") != _selection_manifest_id(selection)
        ):
            errors.append("selection_manifest_hash_mismatch")
        if (
            wave.require_canonical_selection_seed
            and selection.get("seed") != DEFAULT_SELECTION_SEED
        ):
            errors.append("selection_seed_mismatch")
    if (
        len(case_ids) != wave.case_count
        or len(set(case_ids)) != wave.case_count
        or len(request_ids) != wave.case_count
        or not all(_is_full_hash(value) for value in request_ids)
    ):
        errors.append("confirmatory_case_or_request_count_mismatch")
    contexts = manifest.get("contexts")
    if (
        summary.get("selection_manifest_id")
        != (
            selection.get("manifest_id")
            if isinstance(selection, dict)
            else None
        )
        or not isinstance(contexts, dict)
        or summary.get("context_id") != contexts.get("manifest_id")
        or contexts.get("raw_templates_persisted") is not False
        or summary.get("provenance_id") != manifest.get("provenance_id")
    ):
        errors.append("summary_selection_context_provenance_mismatch")

    if model_alias in wave.run_ids:
        try:
            normalized_static = _validate_static_policy(
                static_policy, model_alias=model_alias
            )
            normalized_forecast = validate_forecast_policy(forecast_policy)
            expected_static_manifest = {
                "schema_version": STATIC_POLICY_SCHEMA,
                "policy_hash": normalized_static["policy_hash"],
                "model_alias": model_alias,
                "source_runs": normalized_static["source_runs"],
                "feature_panel_sha256": normalized_static[
                    "feature_panel_sha256"
                ],
                "layer_weights": normalized_static["layer_weights"],
                "path_persisted": False,
            }
            expected_forecast_manifest = {
                "schema_version": normalized_forecast["schema_version"],
                "policy_hash": normalized_forecast["policy_hash"],
                "model_alias": model_alias,
                "static_policy_hash": normalized_forecast[
                    "static_policy_hash"
                ],
                "source_runs": normalized_forecast["source_runs"],
                "beta": normalized_forecast["beta"],
                "calibration_method": normalized_forecast[
                    "calibration_method"
                ],
                "calibration_hash": normalized_forecast["calibration_hash"],
                "confirmatory_outcomes_used": [],
                "path_persisted": False,
            }
            if (
                normalized_forecast["model_alias"] != model_alias
                or normalized_forecast["static_policy_hash"]
                != normalized_static["policy_hash"]
                or manifest.get("static_policy") != expected_static_manifest
                or manifest.get("forecast_policy") != expected_forecast_manifest
                or manifest.get("forecast_policy_formula")
                != (
                    "x_AS=d*(dot(w_adaptive,s)-dot(w_static,s)); "
                    "predicted_adaptive_static_gain=beta*x_AS"
                )
                or summary.get("static_policy_hash")
                != normalized_static["policy_hash"]
                or summary.get("forecast_policy_hash")
                != normalized_forecast["policy_hash"]
                or not _close(
                    _finite(
                        summary.get("forecast_beta"),
                        name="summary.forecast_beta",
                    ),
                    normalized_forecast["beta"],
                    tolerance=2e-7,
                )
            ):
                errors.append("static_forecast_policy_parity_mismatch")
        except ScoreMixAnalysisError:
            errors.append("static_or_forecast_policy_invalid")

    slurm = manifest.get("slurm")
    if (
        not isinstance(slurm, dict)
        or slurm != summary.get("slurm")
        or slurm.get("under_slurm") is not True
        or slurm.get("job_name") != wave.job_name
        or slurm.get("node") != "devbox"
        or slurm.get("slice_label") != wave.label
        or slurm.get("fold") != wave.fold
        or not isinstance(slurm.get("job_id"), str)
        or not slurm["job_id"].isdigit()
    ):
        errors.append("confirmatory_slurm_identity_mismatch")
    git_state = manifest.get("ode_edit_git")
    if (
        not isinstance(git_state, dict)
        or not isinstance(git_state.get("commit"), str)
        or len(git_state["commit"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in str(git_state.get("commit", ""))
        )
        or git_state.get("tracked_worktree_clean") is not True
    ):
        errors.append("confirmatory_git_identity_mismatch")
    return model_alias, case_ids, request_ids


def _validate_stream_payloads(
    *,
    expected_case_count: int,
    case_ids: Sequence[str],
    request_ids: Sequence[str],
    streams: Mapping[str, Sequence[ConfirmatoryStreamRow]],
    receipts: Mapping[str, Mapping[str, Any]],
    receipt_hashes: Mapping[str, str],
    static_policy: Mapping[str, Any],
    forecast_policy: Mapping[str, Any],
    errors: list[str],
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[tuple[str, str], Mapping[str, Any]],
]:
    expected_request = dict(zip(case_ids, request_ids))

    def index(
        name: str,
        key_function,
    ) -> dict[Any, Mapping[str, Any]]:
        result = {}
        for row in streams.get(name, ()):
            key = key_function(row.payload)
            if key in result:
                errors.append(f"duplicate_{name}_key")
            result[key] = row.payload
        return result

    features = index("features.jsonl", lambda payload: payload.get("case_id"))
    actions = index("actions.jsonl", lambda payload: payload.get("case_id"))
    outcomes = index(
        "outcomes.jsonl",
        lambda payload: (payload.get("case_id"), payload.get("action_id")),
    )
    events = index("events.jsonl", lambda payload: payload.get("case_id"))
    observed_outcome_order = tuple(
        (row.payload.get("case_id"), row.payload.get("action_id"))
        for row in streams.get("outcomes.jsonl", ())
    )
    expected_outcome_order = tuple(
        (case_id, action_id)
        for case_id in case_ids
        for action_id in CONFIRMATORY_OUTCOME_ACTIONS
    )
    if observed_outcome_order != expected_outcome_order:
        errors.append("outcome_exact_order_mismatch")
    if set(features) != set(case_ids):
        errors.append("feature_case_panel_mismatch")
    if set(actions) != set(case_ids):
        errors.append("action_case_panel_mismatch")
    if set(events) != set(case_ids):
        errors.append("event_case_panel_mismatch")
    if set(outcomes) != {
        (case_id, action_id)
        for case_id in case_ids
        for action_id in CONFIRMATORY_OUTCOME_ACTIONS
    }:
        errors.append("outcome_case_action_panel_mismatch")

    static_weights = static_policy.get("layer_weights", {})
    beta = _finite(forecast_policy.get("beta"), name="forecast_beta")
    for case_id in case_ids:
        feature = features.get(case_id)
        action = actions.get(case_id)
        event = events.get(case_id)
        if feature is None or action is None or event is None:
            continue
        if (
            set(feature) != CONFIRMATORY_FEATURE_FIELDS
            or _contains_outcome_field(feature)
        ):
            errors.append(f"feature_schema_or_firewall:{case_id}")
            continue
        if (
            set(action) != CONFIRMATORY_ACTION_FIELDS
            or _contains_outcome_field(action)
        ):
            errors.append(f"action_schema_or_firewall:{case_id}")
            continue
        if set(event) != CONFIRMATORY_EVENT_FIELDS:
            errors.append(f"event_schema_mismatch:{case_id}")
            continue
        if (
            feature.get("request_id") != expected_request.get(case_id)
            or action.get("request_id") != expected_request.get(case_id)
            or event.get("request_id") != expected_request.get(case_id)
        ):
            errors.append(f"request_identity_mismatch:{case_id}")
        observed_feature, expected_feature = _payload_hash(
            feature, "feature_hash"
        )
        observed_action, expected_action = _payload_hash(
            action, "commitment_hash"
        )
        if observed_feature != expected_feature:
            errors.append(f"feature_hash_mismatch:{case_id}")
        if observed_action != expected_action:
            errors.append(f"action_hash_mismatch:{case_id}")
        if (
            feature.get("static_policy_hash") != static_policy.get("policy_hash")
            or feature.get("forecast_policy_hash")
            != forecast_policy.get("policy_hash")
            or action.get("feature_hash") != observed_feature
            or action.get("adaptive_action_id") != ADAPTIVE_ACTION
            or action.get("static_action_id") != STATIC_ACTION
            or action.get("static_policy_hash") != static_policy.get("policy_hash")
            or action.get("forecast_policy_hash")
            != forecast_policy.get("policy_hash")
            or event.get("static_policy_hash")
            != static_policy.get("policy_hash")
            or event.get("forecast_policy_hash")
            != forecast_policy.get("policy_hash")
            or not _close(
                _finite(action.get("forecast_beta"), name="action.beta"),
                beta,
                tolerance=2e-7,
            )
            or action.get("bundle_id") != BUNDLE_ID
        ):
            errors.append(f"action_policy_chain_mismatch:{case_id}")
        scores = feature.get("action_scores")
        if not isinstance(scores, dict) or set(scores) != set(PROBE_ACTIONS):
            errors.append(f"feature_score_panel_mismatch:{case_id}")
            continue
        expected_adaptive, expected_score, expected_branch = _expected_mix(scores)
        adaptive_weights = action.get("adaptive_layer_weights")
        observed_static = action.get("static_layer_weights")
        feature_adaptive = feature.get("adaptive_layer_weights")
        feature_static = feature.get("static_layer_weights")
        if (
            not isinstance(adaptive_weights, dict)
            or set(adaptive_weights) != set(SINGLE_ACTIONS)
            or not isinstance(observed_static, dict)
            or set(observed_static) != set(SINGLE_ACTIONS)
            or not isinstance(feature_adaptive, dict)
            or set(feature_adaptive) != set(SINGLE_ACTIONS)
            or not isinstance(feature_static, dict)
            or set(feature_static) != set(SINGLE_ACTIONS)
        ):
            errors.append(f"action_weight_panel_mismatch:{case_id}")
            continue
        if any(
            not _close(
                _finite(adaptive_weights[key], name=f"adaptive:{key}"),
                expected_adaptive[key],
                tolerance=2e-7,
            )
            for key in SINGLE_ACTIONS
        ) or any(
            not _close(
                _finite(feature_adaptive[key], name=f"feature_adaptive:{key}"),
                expected_adaptive[key],
                tolerance=2e-7,
            )
            for key in SINGLE_ACTIONS
        ):
            errors.append(f"adaptive_feature_parity_mismatch:{case_id}")
        if any(
            not _close(
                _finite(observed_static[key], name=f"static:{key}"),
                _finite(static_weights[key], name=f"policy_static:{key}"),
                tolerance=2e-7,
            )
            for key in SINGLE_ACTIONS
        ) or any(
            not _close(
                _finite(feature_static[key], name=f"feature_static:{key}"),
                _finite(static_weights[key], name=f"policy_static:{key}"),
                tolerance=2e-7,
            )
            for key in SINGLE_ACTIONS
        ):
            errors.append(f"static_policy_weight_parity_mismatch:{case_id}")
        static_score = _dot(observed_static, scores)
        distance = _finite(
            feature.get("operational_c_distance"), name="operational_distance"
        )
        predicted_score_gap = expected_score - static_score
        x_as = distance * predicted_score_gap
        predicted_gain = beta * x_as
        if (
            not _close(
                _finite(
                    action.get("adaptive_predicted_score"),
                    name="adaptive_predicted_score",
                ),
                expected_score,
                tolerance=2e-7,
            )
            or not _close(
                _finite(
                    action.get("static_predicted_score"),
                    name="static_predicted_score",
                ),
                static_score,
                tolerance=2e-7,
            )
            or not _close(
                _finite(
                    action.get("predicted_adaptive_static_score_gap"),
                    name="predicted_score_gap",
                ),
                predicted_score_gap,
                tolerance=2e-7,
            )
            or not _close(
                _finite(
                    action.get("x_as"),
                    name="x_as",
                ),
                x_as,
                tolerance=2e-7,
            )
            or not _close(
                _finite(
                    action.get("predicted_adaptive_static_gain"),
                    name="predicted_gain",
                ),
                predicted_gain,
                tolerance=2e-7,
            )
            or not _close(
                _finite(
                    action.get("adaptive_actual_unit_c_energy"),
                    name="adaptive_unit_energy",
                ),
                1.0,
            )
            or not _close(
                _finite(
                    action.get("static_actual_unit_c_energy"),
                    name="static_unit_energy",
                ),
                1.0,
            )
            or action.get("adaptive_controller")
            != "five-single-slopes/relu-l2-else-max-onehot"
            or action.get("adaptive_controller_branch") != expected_branch
        ):
            errors.append(f"action_prediction_or_unit_c_mismatch:{case_id}")
        feature_scalar_parity = {
            "adaptive_predicted_score": expected_score,
            "static_predicted_score": static_score,
            "predicted_adaptive_static_score_gap": predicted_score_gap,
            "x_as": x_as,
            "forecast_beta": beta,
            "predicted_adaptive_static_gain": predicted_gain,
        }
        if any(
            not _close(
                _finite(feature.get(key), name=f"feature:{key}"),
                value,
                tolerance=2e-7,
            )
            for key, value in feature_scalar_parity.items()
        ):
            errors.append(f"feature_prediction_parity_mismatch:{case_id}")

        receipt_reference = event.get("action_receipt")
        if not isinstance(receipt_reference, dict):
            errors.append(f"receipt_reference_missing:{case_id}")
        else:
            name = receipt_reference.get("name")
            receipt = receipts.get(name) if isinstance(name, str) else None
            if receipt is None or set(receipt) != CONFIRMATORY_RECEIPT_FIELDS:
                errors.append(f"receipt_missing_or_schema:{case_id}")
            else:
                stripped = {
                    key: receipt.get(key)
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
                expected_name = (
                    _sha256_bytes(_canonical_json(stripped).encode("utf-8"))
                    + ".json"
                )
                if (
                    name != expected_name
                    or receipt.get("schema_version")
                    != CONFIRMATORY_RECEIPT_SCHEMA
                    or receipt.get("case_id") != case_id
                    or receipt.get("request_id") != expected_request.get(case_id)
                    or receipt.get("q") != SCORE_MIX_Q
                    or receipt.get("feature_hash") != observed_feature
                    or receipt.get("commitment_hash") != observed_action
                    or receipt.get("bundle_id") != action.get("bundle_id")
                    or receipt.get("adaptive_action_id") != ADAPTIVE_ACTION
                    or receipt.get("static_action_id") != STATIC_ACTION
                    or receipt.get("adaptive_layer_weights")
                    != action.get("adaptive_layer_weights")
                    or receipt.get("static_layer_weights")
                    != action.get("static_layer_weights")
                    or receipt.get("static_policy_hash")
                    != static_policy.get("policy_hash")
                    or receipt.get("forecast_policy_hash")
                    != forecast_policy.get("policy_hash")
                    or not _close(
                        _finite(
                            receipt.get("predicted_adaptive_static_score_gap"),
                            name="receipt_score_gap",
                        ),
                        predicted_score_gap,
                        tolerance=2e-7,
                    )
                    or not _close(
                        _finite(receipt.get("x_as"), name="receipt_x_as"),
                        x_as,
                        tolerance=2e-7,
                    )
                    or not _close(
                        _finite(
                            receipt.get("forecast_beta"),
                            name="receipt_beta",
                        ),
                        beta,
                        tolerance=2e-7,
                    )
                    or not _close(
                        _finite(
                            receipt.get("predicted_adaptive_static_gain"),
                            name="receipt_gain",
                        ),
                        predicted_gain,
                        tolerance=2e-7,
                    )
                    or receipt.get("outcomes_observed_before_commitment") is not False
                    or receipt.get("durability")
                    != (
                        "feature+adaptive-static-forecast-action-write+flush+fsync-"
                        "before-exclusive-receipt"
                    )
                    or receipt_reference.get("sha256")
                    != receipt_hashes.get(str(name))
                ):
                    errors.append(f"receipt_identity_or_hash_mismatch:{case_id}")
        if (
            event.get("feature_count") != 1
            or event.get("commitment_count") != 1
            or event.get("outcome_count") != len(CONFIRMATORY_OUTCOME_ACTIONS)
            or event.get("probe_direction_count") != len(PROBE_ACTIONS)
            or event.get("rollback_exact") is not True
            or event.get("pass") is not True
        ):
            errors.append(f"event_count_or_rollback_mismatch:{case_id}")

        for action_id in CONFIRMATORY_OUTCOME_ACTIONS:
            outcome = outcomes.get((case_id, action_id))
            if outcome is None:
                continue
            expected_fields = (
                _OUTCOME_FIELDS
                | {
                    "static_policy_hash",
                    "forecast_policy_hash",
                    "logits_hash_equal",
                }
                if action_id == REPLAY_ACTION
                else _OUTCOME_FIELDS
                | {"static_policy_hash", "forecast_policy_hash"}
            )
            if set(outcome) != expected_fields:
                errors.append(f"outcome_schema_mismatch:{case_id}:{action_id}")
                continue
            if (
                outcome.get("request_id") != expected_request.get(case_id)
                or outcome.get("feature_hash") != observed_feature
                or outcome.get("commitment_hash") != observed_action
                or outcome.get("q") != SCORE_MIX_Q
                or outcome.get("q_label") != "q_1_256"
                or outcome.get("status") != "completed"
                or outcome.get("rollback_exact") is not True
                or outcome.get("static_policy_hash")
                != static_policy.get("policy_hash")
                or outcome.get("forecast_policy_hash")
                != forecast_policy.get("policy_hash")
                or outcome.get("selected_by_controller")
                != (action_id == ADAPTIVE_ACTION)
                or outcome.get("budget_validation")
                != _BUDGET_VALIDATION_BY_ACTION[action_id]
            ):
                errors.append(f"outcome_chain_or_rollback:{case_id}:{action_id}")
            try:
                _finite(outcome.get("progress"), name="outcome.progress")
                if action_id in MATCHED_C_ACTIONS:
                    if (
                        not _close(
                            _finite(outcome.get("c_distance"), name="c_distance"),
                            distance,
                        )
                        or not _close(
                            _finite(outcome.get("c_energy"), name="c_energy"),
                            distance * distance,
                        )
                    ):
                        errors.append(
                            f"equal_c_mismatch:{case_id}:{action_id}"
                        )
                elif action_id == REPLAY_ACTION:
                    if (
                        outcome.get("logits_hash_equal") is not True
                        or _finite(
                            outcome.get("c_distance"), name="replay_distance"
                        )
                        != 0.0
                        or _finite(
                            outcome.get("c_energy"), name="replay_energy"
                        )
                        != 0.0
                    ):
                        errors.append(f"replay_mismatch:{case_id}")
            except ScoreMixAnalysisError:
                errors.append(f"outcome_nonfinite:{case_id}:{action_id}")
    if len(receipts) != expected_case_count:
        errors.append("receipt_count_mismatch")
    return features, actions, outcomes


def _validate_summary_and_hashes(
    *,
    expected_case_count: int,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    streams: Mapping[str, Sequence[ConfirmatoryStreamRow]],
    receipts: Mapping[str, Mapping[str, Any]],
    observed_hashes: Mapping[str, str],
    errors: list[str],
) -> None:
    expected_counts = {
        "features": expected_case_count,
        "commitments": expected_case_count,
        "outcomes": expected_case_count * len(CONFIRMATORY_OUTCOME_ACTIONS),
        "receipts": expected_case_count,
    }
    if manifest.get("expected_counts") != expected_counts:
        errors.append("manifest_expected_counts_mismatch")
    exact_summary = {
        "planned_case_count": expected_case_count,
        "attempted_case_count": expected_case_count,
        "not_run_due_to_abort_count": 0,
        "pass_count": expected_case_count,
        "failure_count": 0,
        "feature_count": expected_case_count,
        "commitment_count": expected_case_count,
        "outcome_count": expected_case_count * len(CONFIRMATORY_OUTCOME_ACTIONS),
        "action_receipt_count": expected_case_count,
        "probe_direction_count_per_case": len(PROBE_ACTIONS),
    }
    for field, expected in exact_summary.items():
        if summary.get(field) != expected:
            errors.append(f"summary_count_mismatch:{field}")
    if summary.get("stream_sequences") != {
        "features": expected_case_count,
        "actions": expected_case_count,
        "outcomes": expected_case_count * len(CONFIRMATORY_OUTCOME_ACTIONS),
        "events": expected_case_count,
    }:
        errors.append("summary_stream_sequence_mismatch")
    if (
        summary.get("run_status") != "completed"
        or summary.get("abort_failure_type") is not None
        or summary.get("all_pass") is not True
        or summary.get("expected_counts_exact") is not True
        or summary.get("all_rollbacks_exact") is not True
        or summary.get("projector_files_loaded") != []
        or summary.get("git_output_written") is not False
    ):
        errors.append("summary_completion_or_firewall_mismatch")
    if (
        len(streams.get("features.jsonl", ())) != expected_case_count
        or len(streams.get("actions.jsonl", ())) != expected_case_count
        or len(streams.get("outcomes.jsonl", ()))
        != expected_case_count * len(CONFIRMATORY_OUTCOME_ACTIONS)
        or len(streams.get("events.jsonl", ())) != expected_case_count
        or len(receipts) != expected_case_count
    ):
        errors.append("observed_exact_count_mismatch")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("summary_artifact_hashes_missing")
        return
    expected_file_hashes = {
        "manifest_sha256": observed_hashes.get("manifest.json"),
        "features_sha256": observed_hashes.get("features.jsonl"),
        "actions_sha256": observed_hashes.get("actions.jsonl"),
        "outcomes_sha256": observed_hashes.get("outcomes.jsonl"),
        "events_sha256": observed_hashes.get("events.jsonl"),
    }
    for field, expected in expected_file_hashes.items():
        if artifacts.get(field) != expected:
            errors.append(f"summary_artifact_hash_mismatch:{field}")
    actual_receipts = {
        name.removeprefix("action_receipts/"): digest
        for name, digest in observed_hashes.items()
        if name.startswith("action_receipts/")
    }
    if artifacts.get("action_receipts") != actual_receipts:
        errors.append("summary_receipt_hash_map_mismatch")


def analyze_score_mix_wave(
    run_directory: str | Path,
    *,
    wave: ScoreMixAnalysisWaveLock,
    static_policy: Mapping[str, Any],
    forecast_policy: Mapping[str, Any],
    forecast_analysis: Mapping[str, Any],
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    emit_fold01_aggregate_input: bool = False,
) -> dict[str, Any]:
    wave = _validated_analysis_wave_lock(wave)
    if type(emit_fold01_aggregate_input) is not bool:
        raise ScoreMixAnalysisError(
            "fold01 aggregate-input option must be boolean"
        )
    if emit_fold01_aggregate_input and wave.label != "c1":
        raise ScoreMixAnalysisError(
            "fold01 aggregate-input envelope is C1-only"
        )
    if emit_fold01_aggregate_input and (
        type(bootstrap_seed) is not int
        or type(bootstrap_replicates) is not int
        or bootstrap_seed != DEFAULT_BOOTSTRAP_SEED
        or bootstrap_replicates != DEFAULT_BOOTSTRAP_REPLICATES
    ):
        raise ScoreMixAnalysisError(
            "fold01 aggregate-input bootstrap differs from the precommit"
        )
    if wave.label != "c1" and (
        type(bootstrap_seed) is not int
        or type(bootstrap_replicates) is not int
        or bootstrap_seed != DEFAULT_BOOTSTRAP_SEED
        or bootstrap_replicates != DEFAULT_BOOTSTRAP_REPLICATES
    ):
        raise ScoreMixAnalysisError(
            "follow-up bootstrap configuration differs from the precommit"
        )
    root = Path(run_directory).expanduser().resolve()
    if not root.is_dir():
        raise ScoreMixAnalysisError("confirmatory run directory unavailable")
    manifest = _read_json_object(root / "manifest.json")
    summary = _read_json_object(root / "summary.json")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ScoreMixAnalysisError("confirmatory run_id invalid")
    streams = {
        name: _read_stream(root / name, run_id=run_id)
        for name in _STREAM_FILES
    }
    receipt_root = root / "action_receipts"
    if receipt_root.is_symlink() or not receipt_root.is_dir():
        raise ScoreMixAnalysisError("confirmatory receipt directory unavailable")
    receipt_paths = sorted(receipt_root.glob("*.json"))
    receipts = {path.name: _read_json_object(path) for path in receipt_paths}
    observed_hashes = {
        name: _sha256_file(root / name) for name in _HASHED_FILES
    }
    observed_hashes.update(
        {
            f"action_receipts/{path.name}": _sha256_file(path)
            for path in receipt_paths
        }
    )

    errors: list[str] = []
    model_alias, case_ids, request_ids = _validate_manifest_and_policies(
        wave=wave,
        root=root,
        manifest=manifest,
        summary=summary,
        static_policy=static_policy,
        forecast_policy=forecast_policy,
        errors=errors,
    )
    if "static_or_forecast_policy_invalid" in errors:
        features, actions, outcomes = {}, {}, {}
    else:
        features, actions, outcomes = _validate_stream_payloads(
            expected_case_count=wave.case_count,
            case_ids=case_ids,
            request_ids=request_ids,
            streams=streams,
            receipts=receipts,
            receipt_hashes={
                path.name: observed_hashes[f"action_receipts/{path.name}"]
                for path in receipt_paths
            },
            static_policy=static_policy,
            forecast_policy=forecast_policy,
            errors=errors,
        )
    _validate_summary_and_hashes(
        expected_case_count=wave.case_count,
        manifest=manifest,
        summary=summary,
        streams=streams,
        receipts=receipts,
        observed_hashes=observed_hashes,
        errors=errors,
    )
    residual_envelope: float | None = None
    calibration_replay_envelope: float | None = None
    if (
        model_alias in D1_RUN_IDS
        and "static_or_forecast_policy_invalid" not in errors
    ):
        try:
            (
                _normalized_forecast_analysis,
                residual_envelope,
                calibration_replay_envelope,
            ) = _validate_forecast_analysis(
                forecast_analysis,
                model_alias=str(model_alias),
                static_policy=static_policy,
                forecast_policy=forecast_policy,
            )
        except (KeyError, TypeError, ScoreMixAnalysisError):
            errors.append("forecast_analysis_parity_mismatch")
    else:
        errors.append("forecast_analysis_parity_mismatch")
    error_codes = sorted(set(errors))
    artifact_valid = not error_codes

    replay_values: list[float] = []
    c1_replay_envelope: float | None = None
    effects: list[float] = []
    oracle_values: list[float] = []
    predicted_values: list[float] = []
    case_diagnostics: list[dict[str, Any]] = []
    if artifact_valid:
        for case_id in case_ids:
            progress = {
                action_id: _finite(
                    outcomes[(case_id, action_id)]["progress"],
                    name=f"progress:{action_id}",
                )
                for action_id in CONFIRMATORY_OUTCOME_ACTIONS
            }
            replay_values.append(abs(progress[REPLAY_ACTION]))
        c1_replay_envelope = max(
            NUMERIC_COMPARISON_TOLERANCE,
            max(replay_values),
        )
        replay_envelope = max(
            c1_replay_envelope,
            _finite(
                calibration_replay_envelope,
                name="calibration_replay_envelope",
            ),
        )
        for case_id in case_ids:
            progress = {
                action_id: _finite(
                    outcomes[(case_id, action_id)]["progress"],
                    name=f"progress:{action_id}",
                )
                for action_id in CONFIRMATORY_OUTCOME_ACTIONS
            }
            effect = progress[ADAPTIVE_ACTION] - progress[STATIC_ACTION]
            oracle = (
                max(progress[action_id] for action_id in ORACLE_ACTIONS)
                - progress[STATIC_ACTION]
            )
            predicted = _finite(
                actions[case_id]["predicted_adaptive_static_gain"],
                name="predicted_forecast",
            )
            effects.append(effect)
            oracle_values.append(oracle)
            predicted_values.append(predicted)
            case_diagnostics.append(
                {
                    "case_id": case_id,
                    "adaptive_static_effect": effect,
                    "finite_panel_oracle_opportunity": oracle,
                    "predicted_adaptive_static_gain": predicted,
                    "prediction_error": effect - predicted,
                    "effect_above_replay_envelope": effect > replay_envelope,
                }
            )
        primary = _effect_summary(
            effects,
            expected_case_count=wave.case_count,
            replay_envelope=replay_envelope,
            bootstrap_seed=bootstrap_seed,
            bootstrap_replicates=bootstrap_replicates,
        )
        oracle_summary = _effect_summary(
            oracle_values,
            expected_case_count=wave.case_count,
            replay_envelope=replay_envelope,
            bootstrap_seed=bootstrap_seed + 1,
            bootstrap_replicates=bootstrap_replicates,
        )
        denominator = sum(value * value for value in predicted_values)
        realized_on_predicted_slope = (
            sum(
                predicted * realized
                for predicted, realized in zip(predicted_values, effects)
            )
            / denominator
            if denominator > NUMERIC_COMPARISON_TOLERANCE
            else None
        )
        prediction = {
            "predicted_mean": _fmean(
                predicted_values,
                name="predicted_mean",
            ),
            "realized_mean": _fmean(effects, name="realized_mean"),
            "mean_error": _fmean(
                (
                    realized - predicted
                    for predicted, realized in zip(predicted_values, effects)
                ),
                name="prediction_mean_error",
            ),
            "mean_absolute_error": _fmean(
                (
                    abs(realized - predicted)
                    for predicted, realized in zip(predicted_values, effects)
                ),
                name="prediction_mean_absolute_error",
            ),
            "median_absolute_error": _finite(
                statistics.median(
                    abs(realized - predicted)
                    for predicted, realized in zip(predicted_values, effects)
                ),
                name="prediction_median_absolute_error",
            ),
            "zero_intercept_realized_on_predicted_slope": (
                realized_on_predicted_slope
            ),
            "pearson": _pearson(predicted_values, effects),
            "positive_direction_concordance": sum(
                (predicted > 0.0) == (realized > replay_envelope)
                for predicted, realized in zip(predicted_values, effects)
            )
            / len(effects),
        }
        if wave.label == "c1":
            clear_continue = bool(
                primary["mean"] > replay_envelope
                and (
                    primary["trimmed_mean_20pct"] > replay_envelope
                    or primary["median"] > replay_envelope
                    or primary[
                        "positive_sign_count_above_replay_envelope"
                    ]
                    >= 7
                )
            )
            scientific_kill_input = bool(
                primary["mean"] <= replay_envelope
                and primary["trimmed_mean_20pct"] <= replay_envelope
                and primary[
                    "positive_sign_fraction_above_replay_envelope"
                ]
                <= 0.50
                and oracle_summary["mean"] <= replay_envelope
            )
            controller_pivot_input = bool(
                primary["mean"] <= replay_envelope
                and primary["trimmed_mean_20pct"] <= replay_envelope
                and primary[
                    "positive_sign_fraction_above_replay_envelope"
                ]
                <= 0.50
                and oracle_summary["mean"] > replay_envelope
                and (
                    oracle_summary["trimmed_mean_20pct"] > replay_envelope
                    or oracle_summary["median"] > replay_envelope
                    or oracle_summary[
                        "positive_sign_count_above_replay_envelope"
                    ]
                    >= 7
                )
            )
        else:
            clear_continue = False
            scientific_kill_input = False
            controller_pivot_input = False
    else:
        replay_envelope = None
        c1_replay_envelope = None
        primary = None
        oracle_summary = None
        prediction = None
        clear_continue = False
        scientific_kill_input = False
        controller_pivot_input = False

    if not artifact_valid:
        replay_envelope = None
        c1_replay_envelope = None
        calibration_replay_envelope = None
        residual_envelope = None
        primary = None
        oracle_summary = None
        prediction = None
        case_diagnostics = []
        clear_continue = False
        scientific_kill_input = False
        controller_pivot_input = False

    selection = manifest.get("selection")
    fold_payload = manifest.get("confirmatory_folds")
    artifact_validation = {
        "valid": artifact_valid,
        "panel_complete": bool(
            artifact_valid
            and len(features) == wave.case_count
            and len(actions) == wave.case_count
            and len(outcomes)
            == wave.case_count * len(CONFIRMATORY_OUTCOME_ACTIONS)
        ),
        "error_codes": error_codes,
        "exact_fold": (
            "confirmatory[0::5]"
            if wave.label == "c1"
            else f"confirmatory_hash_fold_{wave.fold}"
            if wave.selected_split == "confirmatory"
            else "selection_manifest_untouched_exact"
        ),
        "case_count": wave.case_count,
        "arm_count_per_case": len(CONFIRMATORY_OUTCOME_ACTIONS),
        "static_policy_hash": static_policy.get("policy_hash"),
        "forecast_policy_hash": forecast_policy.get("policy_hash"),
        "outcome_firewall_and_receipts_valid": artifact_valid,
        "equal_c_and_rollback_valid": artifact_valid,
    }
    if wave.label != "c1":
        artifact_validation.update(
            {
                "selection_manifest_id": (
                    selection.get("manifest_id")
                    if isinstance(selection, Mapping)
                    else None
                ),
                "fold_manifest_id": (
                    fold_payload.get("manifest_id")
                    if isinstance(fold_payload, Mapping)
                    else None
                ),
                "selected_case_ids_sha256": _sha256_bytes(
                    _canonical_json(list(case_ids)).encode("utf-8")
                ),
                "context_id": summary.get("context_id"),
                "provenance_id": summary.get("provenance_id"),
                "q": SCORE_MIX_Q,
                "outcome_action_order": list(CONFIRMATORY_OUTCOME_ACTIONS),
                "calibration_hash": forecast_policy.get("calibration_hash"),
                "forecast_beta": forecast_policy.get("beta"),
                "run_seed": wave.run_seed,
                "primary_estimand": (
                    "progress(score_mix)-progress(frozen_static_mix)"
                ),
                "bootstrap_seed": bootstrap_seed,
                "bootstrap_replicates": bootstrap_replicates,
            }
        )
    report = {
        "schema_version": (
            CONFIRMATORY_ANALYSIS_SCHEMA
            if wave.label == "c1"
            else "ode-edit-mv1-score-mix-followup-analysis/v1"
        ),
        "claim_status": (
            "single_model_confirmatory_gate_inputs_only"
            if wave.label == "c1"
            else "single_model_followup_gate_inputs_only"
        ),
        "analysis_status": (
            (
                "confirmatory_single_model_complete"
                if wave.label == "c1"
                else "followup_single_model_complete"
            )
            if artifact_valid
            else "technical_block_invalid_artifacts"
        ),
        "run_id": run_id,
        "model_alias": model_alias,
        "artifact_validation": artifact_validation,
        "replay_envelope": replay_envelope,
        "calibration_replay_envelope": calibration_replay_envelope,
        **(
            {"c1_replay_envelope": c1_replay_envelope}
            if wave.label == "c1"
            else {"wave_replay_envelope": c1_replay_envelope}
        ),
        "primary_adaptive_minus_frozen_static": primary,
        "finite_panel_oracle_opportunity": oracle_summary,
        "predicted_vs_realized": prediction,
        "calibration_residual_envelope": residual_envelope,
        "case_diagnostics": case_diagnostics,
        "single_model_gate_inputs": {
            "clear_continue_input": clear_continue,
            "scientific_kill_input": scientific_kill_input,
            "controller_pivot_input": controller_pivot_input,
            "gray_input": bool(
                artifact_valid
                and wave.label == "c1"
                and not clear_continue
                and not scientific_kill_input
                and not controller_pivot_input
            ),
            "architecture_conditional_tolerance_input": residual_envelope,
            "pair_level_decision_computed": False,
        },
        "claim_boundary": (
            (
                "One fixed-model 12-case C1 fold. Pair-level continuation, "
                "architecture-conditional continuation, gray/kill/pivot, method "
                "gain, MV-2, and ODE decisions are not made here."
            )
            if wave.label == "c1"
            else (
                f"One fixed-model {wave.case_count}-case {wave.label} run. "
                "Only single-run gate inputs are emitted; pair, cross-model, "
                "aggregate-24, method-gain, MV-2, and ODE decisions are not made."
            )
        ),
    }
    if wave.label != "c1":
        report["mode"] = wave.label
        report["execution_lock"] = {
            "selected_split": wave.selected_split,
            "fold": wave.fold,
            "case_count": wave.case_count,
            "run_seed": wave.run_seed,
            "q": SCORE_MIX_Q,
            "outcome_action_order": list(CONFIRMATORY_OUTCOME_ACTIONS),
            "primary_estimand": (
                "progress(score_mix)-progress(frozen_static_mix)"
            ),
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_replicates": bootstrap_replicates,
            "pair_level_decision_computed": False,
        }
        report["model_single_run_gate_inputs"] = {
            "replay_envelope": replay_envelope,
            "primary": primary,
            "oracle": oracle_summary,
            "forecast_calibration": {
                "predicted_vs_realized": prediction,
                "calibration_residual_envelope": residual_envelope,
                "calibration_replay_envelope": calibration_replay_envelope,
            },
            "pair_level_decision_computed": False,
        }
    _json_safe(report, location="confirmatory_analysis")
    json.dumps(report, allow_nan=False)
    if emit_fold01_aggregate_input:
        if (
            not isinstance(selection, Mapping)
            or selection.get("seed") != DEFAULT_SELECTION_SEED
        ):
            raise ScoreMixAnalysisError(
                "fold01 aggregate-input selection seed is not canonical"
            )
        aggregate_identity = None
        if artifact_valid:
            selection_case_ids = selection.get("case_ids")
            confirmatory_case_ids = (
                list(selection_case_ids.get("confirmatory", []))
                if isinstance(selection_case_ids, Mapping)
                else []
            )
            aggregate_identity = {
                "schema_version": CONFIRMATORY_FOLD01_IDENTITY_SCHEMA,
                "selected_split": "confirmatory",
                "fold": CONFIRMATORY_FOLD_INDEX,
                "fold_count": CONFIRMATORY_FOLD_COUNT,
                "case_count": CONFIRMATORY_CASE_COUNT,
                "selection_seed": DEFAULT_SELECTION_SEED,
                "confirmatory_case_ids": confirmatory_case_ids,
                "selection_manifest_id": selection.get("manifest_id"),
                "fold_manifest_id": (
                    fold_payload.get("manifest_id")
                    if isinstance(fold_payload, Mapping)
                    else None
                ),
                "selected_case_ids_sha256": _sha256_bytes(
                    _canonical_json(list(case_ids)).encode("utf-8")
                ),
                "context_id": summary.get("context_id"),
                "provenance_id": summary.get("provenance_id"),
                "q": SCORE_MIX_Q,
                "outcome_action_order": list(
                    CONFIRMATORY_OUTCOME_ACTIONS
                ),
                "calibration_hash": forecast_policy.get("calibration_hash"),
                "forecast_beta": forecast_policy.get("beta"),
                "static_policy_hash": static_policy.get("policy_hash"),
                "forecast_policy_hash": forecast_policy.get("policy_hash"),
                "primary_estimand": (
                    "progress(score_mix)-progress(frozen_static_mix)"
                ),
                "bootstrap_seed": bootstrap_seed,
                "bootstrap_replicates": bootstrap_replicates,
                "c1_runtime_seed": None,
            }
        envelope = {
            "schema_version": CONFIRMATORY_FOLD01_INPUT_SCHEMA,
            "claim_status": "c1_fold01_aggregate_input_only",
            "analysis_status": (
                "c1_fold01_aggregate_input_complete"
                if artifact_valid
                else "technical_block_invalid_artifacts"
            ),
            "c1_analysis": report,
            "aggregate_identity": aggregate_identity,
        }
        envelope["analysis_hash"] = _sha256_bytes(
            _canonical_json(envelope).encode("utf-8")
        )
        _json_safe(envelope, location="c1_fold01_aggregate_input")
        json.dumps(envelope, allow_nan=False)
        return envelope
    return report


def analyze_confirmatory_run(
    run_directory: str | Path,
    *,
    static_policy: Mapping[str, Any],
    forecast_policy: Mapping[str, Any],
    forecast_analysis: Mapping[str, Any],
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    return analyze_score_mix_wave(
        run_directory,
        wave=CONFIRMATORY_ANALYSIS_WAVE_LOCK,
        static_policy=static_policy,
        forecast_policy=forecast_policy,
        forecast_analysis=forecast_analysis,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
    )


def analyze_confirmatory_run_for_fold01(
    run_directory: str | Path,
    *,
    static_policy: Mapping[str, Any],
    forecast_policy: Mapping[str, Any],
    forecast_analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Emit a versioned aggregate-input envelope around unchanged C1 v1."""

    return analyze_score_mix_wave(
        run_directory,
        wave=CONFIRMATORY_ANALYSIS_WAVE_LOCK,
        static_policy=static_policy,
        forecast_policy=forecast_policy,
        forecast_analysis=forecast_analysis,
        bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
        bootstrap_replicates=DEFAULT_BOOTSTRAP_REPLICATES,
        emit_fold01_aggregate_input=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze one MV-1 C1 fold-0 score-mix confirmatory run."
    )
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--static-policy", required=True, type=Path)
    parser.add_argument("--forecast-policy", required=True, type=Path)
    parser.add_argument("--forecast-analysis", required=True, type=Path)
    parser.add_argument("--analysis-output", required=True, type=Path)
    parser.add_argument(
        "--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--fold01-aggregate-input",
        action="store_true",
        help=(
            "emit a versioned C1 envelope with the identities required by "
            "the CPU-only fold0+fold1 aggregate"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        static_policy = _read_json_object(args.static_policy)
        forecast_policy = _read_json_object(args.forecast_policy)
        forecast_analysis = _read_json_object(args.forecast_analysis)
        if args.fold01_aggregate_input:
            if (
                args.bootstrap_seed != DEFAULT_BOOTSTRAP_SEED
                or args.bootstrap_replicates != DEFAULT_BOOTSTRAP_REPLICATES
            ):
                raise ScoreMixAnalysisError(
                    "fold01 aggregate-input bootstrap differs from the precommit"
                )
            report = analyze_confirmatory_run_for_fold01(
                args.run_directory,
                static_policy=static_policy,
                forecast_policy=forecast_policy,
                forecast_analysis=forecast_analysis,
            )
            analyzed_report = report["c1_analysis"]
        else:
            report = analyze_confirmatory_run(
                args.run_directory,
                static_policy=static_policy,
                forecast_policy=forecast_policy,
                forecast_analysis=forecast_analysis,
                bootstrap_seed=args.bootstrap_seed,
                bootstrap_replicates=args.bootstrap_replicates,
            )
            analyzed_report = report
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
                    "artifact_valid": analyzed_report[
                        "artifact_validation"
                    ]["valid"],
                    "pair_level_decision_computed": False,
                },
                sort_keys=True,
            )
        )
        return 0 if analyzed_report["artifact_validation"]["valid"] else 2
    except (OSError, ScoreMixAnalysisError) as exc:
        print(
            json.dumps(
                {
                    "status": "confirmatory_analysis_aborted",
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
