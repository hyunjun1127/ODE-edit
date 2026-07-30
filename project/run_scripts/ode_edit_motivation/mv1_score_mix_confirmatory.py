"""Held-out MV-1 score-mix confirmatory runner.

This module extends the calibrated score-mix diagnostic with one frozen-static
operational comparator.  It deliberately reuses the existing read-only
EasyEdit bridge, direct-z cache, synchronous/ordered proposal construction,
unit-C geometry, branch rollback, and artifact firewall.  Per event, direct-z
and both proposal families are built once; the six central finite-difference
probes are also evaluated once.

The adaptive and frozen-static actions, along with a calibration-only forecast,
are durably committed before any operational outcome is evaluated.  Neither
the static policy nor the forecast policy may contain confirmatory outcomes.
Projectors are provenance-checked but never deserialized, and covariance
moments are loaded only from the pinned read-only bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import torch

from .contracts import (
    ContractError,
    EditRequest,
    MemitFactorProposal,
    canonical_json,
    sha256_bytes,
)
from .easyedit_bridge import CovarianceCacheSpec, EasyEditBridge
from .gpu_runtime import (
    FixedModelRuntime,
    load_fixed_model,
    offline_environment,
    rng_state_hash,
    seed_runtime,
)
from .manifests import (
    DEFAULT_SELECTION_SEED,
    MODEL_SPECS,
    CounterFactSelectionManifest,
    fixed_model_spec,
    generate_counterfact_selection,
    load_counterfact_requests,
    preflight_fixed_artifacts,
)
from .mv0_fidelity import (
    DEFAULT_OUTPUT_ROOT,
    RollbackError,
    SanitizedJsonlWriter,
    _bridge_pins,
    _covariance_specs,
    _event_seed,
    _file_sha256,
    _freeze_contexts,
    _git_runtime_state,
    _load_hparams,
    _load_verified_covariances,
    _local_run_directory,
    _relative_provenance,
    _safe_payload,
    _silence_upstream,
    _weight_hashes,
    _write_json_exclusive,
)
from .mv1_analysis import (
    ConfirmatoryFoldManifest,
    build_confirmatory_fold_manifest,
)
from .mv1_score_mix_calibration_forecast import CALIBRATION_METHOD
from .mv1_calibration import (
    ACTION_UNIFORM,
    ActionDirection,
    MV1Error,
    _evaluate_branch,
    _feature_hash,
    _layer_by_weight,
    _metric_difference,
    _state_identity,
    assert_exact_action_contract,
    build_unit_c_actions,
    proposal_c_energy,
    rewrite_metrics,
    scale_proposal,
    teacher_forced_rewrite_exact,
)
from .mv1_score_mix import (
    EXPECTED_LAYER_COUNT,
    NATIVE_MEMIT_ACTION,
    NO_OP_ACTION,
    ORDERED_GLOBAL_ALPHA_ACTION,
    PROBE_RATIO,
    SCORE_MIX_ACTION,
    SCORE_MIX_Q,
    ScoreMixDecision,
    _assert_outcome_free,
    build_score_mix_direction,
    derive_score_mix_decision,
)


FROZEN_STATIC_ACTION = "frozen_static_mix"
CONFIRMATORY_OUTCOME_ACTIONS = (
    SCORE_MIX_ACTION,
    FROZEN_STATIC_ACTION,
    ACTION_UNIFORM,
    ORDERED_GLOBAL_ALPHA_ACTION,
    NATIVE_MEMIT_ACTION,
    NO_OP_ACTION,
)
CONFIRMATORY_MANIFEST_SCHEMA = (
    "ode-edit-mv1-score-mix-confirmatory-manifest/v1"
)
CONFIRMATORY_STREAM_SCHEMA = "ode-edit-mv1-score-mix-confirmatory/v1"
CONFIRMATORY_RECEIPT_SCHEMA = (
    "ode-edit-mv1-score-mix-confirmatory-receipt/v1"
)
CONFIRMATORY_SUMMARY_SCHEMA = (
    "ode-edit-mv1-score-mix-confirmatory-summary/v1"
)
CONFIRMATORY_MANIFEST_FIELDS = frozenset(
    {
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
        "static_policy",
        "forecast_policy",
        "decision_policy",
        "forecast_policy_formula",
        "action_receipt_policy",
        "utility_policy",
        "teacher_suffix_policy",
        "projector_policy",
        "covariance_policy",
        "direct_z_policy",
        "artifact_firewall",
        "expected_counts",
    }
)
CONFIRMATORY_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "model_alias",
        "slice",
        "slurm",
        "provenance_id",
        "selection_manifest_id",
        "context_id",
        "static_policy_hash",
        "forecast_policy_hash",
        "forecast_beta",
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
        "stream_sequences",
        "expected_counts_exact",
        "all_rollbacks_exact",
        "covariance_files_loaded",
        "projector_files_loaded",
        "direct_z_artifact_count",
        "resource",
        "artifacts",
        "git_output_written",
    }
)
STATIC_POLICY_SCHEMA = "ode-edit-mv1-score-mix-static-policy/v1"
FORECAST_POLICY_SCHEMA = "ode-edit-mv1-score-mix-forecast-policy/v1"
CONFIRMATORY_FOLD_SEED = "ode-edit-mv1-score-mix-confirmatory-folds-v1"
CONFIRMATORY_FOLD_COUNT = 5
CONFIRMATORY_FOLD = 0
CONFIRMATORY_CASE_COUNT = 12
CONFIRMATORY_JOB_NAME = "odeedit_mv1mix_c1_pair_v1"
CONFIRMATORY_LABEL = "c1"
CONFIRMATORY_RUN_IDS = {
    "llama3-8b-inst": "mv1mix_llama_c1_v1",
    "qwen2.5-7b-inst": "mv1mix_qwen_c1_v1",
}


@dataclass(frozen=True, slots=True)
class ScoreMixWaveLock:
    """Outcome-blind execution envelope shared by C1 and its fixed follow-ups."""

    label: str
    selected_split: str
    fold: int | None
    case_count: int
    job_name: str
    run_ids: Mapping[str, str]
    run_seed: int | None = None

    def __post_init__(self) -> None:
        if (
            type(self.label) is not str
            or type(self.selected_split) is not str
            or type(self.case_count) is not int
            or (
                self.fold is not None
                and type(self.fold) is not int
            )
        ):
            raise ContractError("score-mix wave slice types are invalid")
        exact_slice = (
            self.label,
            self.selected_split,
            self.fold,
            self.case_count,
        )
        if exact_slice not in {
            ("c1", "confirmatory", 0, 12),
            ("fold1", "confirmatory", 1, 12),
            ("untouched", "untouched", None, 20),
        }:
            raise ContractError("score-mix wave slice is outside the fixed envelope")
        if (
            not isinstance(self.job_name, str)
            or not self.job_name
            or any(ord(character) < 32 for character in self.job_name)
        ):
            raise ContractError("score-mix wave job name is invalid")
        normalized_runs = dict(self.run_ids)
        if (
            set(normalized_runs) != set(MODEL_SPECS)
            or any(
                not isinstance(run_id, str)
                or not run_id
                or any(ord(character) < 32 for character in run_id)
                for run_id in normalized_runs.values()
            )
            or len(set(normalized_runs.values())) != len(normalized_runs)
        ):
            raise ContractError("score-mix wave run-ID map is invalid")
        if self.run_seed is not None and (
            isinstance(self.run_seed, bool) or not isinstance(self.run_seed, int)
        ):
            raise ContractError("score-mix wave run seed must be an integer")
        expected_identity = {
            "c1": {
                "job_name": "odeedit_mv1mix_c1_pair_v1",
                "run_ids": {
                    "llama3-8b-inst": "mv1mix_llama_c1_v1",
                    "qwen2.5-7b-inst": "mv1mix_qwen_c1_v1",
                },
                "run_seed": None,
            },
            "fold1": {
                "job_name": "odeedit_mv1mix_fold1_pair_v1",
                "run_ids": {
                    "llama3-8b-inst": "mv1mix_llama_fold1_v1",
                    "qwen2.5-7b-inst": "mv1mix_qwen_fold1_v1",
                },
                "run_seed": 17,
            },
            "untouched": {
                "job_name": "odeedit_mv1mix_untouched_pair_v1",
                "run_ids": {
                    "llama3-8b-inst": "mv1mix_llama_untouched_v1",
                    "qwen2.5-7b-inst": "mv1mix_qwen_untouched_v1",
                },
                "run_seed": 17,
            },
        }[self.label]
        if (
            self.job_name != expected_identity["job_name"]
            or normalized_runs != expected_identity["run_ids"]
            or self.run_seed != expected_identity["run_seed"]
        ):
            raise ContractError("score-mix wave identity differs from the precommit")
        object.__setattr__(
            self,
            "run_ids",
            MappingProxyType(normalized_runs),
        )


CONFIRMATORY_WAVE_LOCK = ScoreMixWaveLock(
    label=CONFIRMATORY_LABEL,
    selected_split="confirmatory",
    fold=CONFIRMATORY_FOLD,
    case_count=CONFIRMATORY_CASE_COUNT,
    job_name=CONFIRMATORY_JOB_NAME,
    run_ids=CONFIRMATORY_RUN_IDS,
)


def _validated_score_mix_wave_lock(wave: ScoreMixWaveLock) -> ScoreMixWaveLock:
    """Reject annotation bypasses and revalidate every canonical field."""

    if type(wave) is not ScoreMixWaveLock:
        raise ContractError("score-mix wave lock has an invalid runtime type")
    revalidated = ScoreMixWaveLock(
        label=wave.label,
        selected_split=wave.selected_split,
        fold=wave.fold,
        case_count=wave.case_count,
        job_name=wave.job_name,
        run_ids=wave.run_ids,
        run_seed=wave.run_seed,
    )
    if wave != revalidated:
        raise ContractError("score-mix wave lock failed canonical revalidation")
    return wave


CONFIRMATORY_POLICY_PATHS = {
    "llama3-8b-inst": {
        "static": (
            "experiment-reports/global/"
            "2026-07-31-mv1mix-llama-d1-v1-static-policy.json"
        ),
        "forecast": (
            "experiment-reports/global/"
            "2026-07-31-mv1mix-llama-d0-d1-forecast-v1.policy.json"
        ),
    },
    "qwen2.5-7b-inst": {
        "static": (
            "experiment-reports/global/"
            "2026-07-31-mv1mix-qwen-d1-v1-static-policy.json"
        ),
        "forecast": (
            "experiment-reports/global/"
            "2026-07-31-mv1mix-qwen-d0-d1-forecast-v1.policy.json"
        ),
    },
}
SOURCE_RUN_IDS = {
    "llama3-8b-inst": {
        "d0": "mv1mix_llama_d0_v1",
        "d1": "mv1mix_llama_d1_v1",
    },
    "qwen2.5-7b-inst": {
        "d0": "mv1mix_qwen_d0_v1",
        "d1": "mv1mix_qwen_d1_v1",
    },
}
LAYERS = (4, 5, 6, 7, 8)
LAYER_ACTIONS = tuple(f"layer_{layer}" for layer in LAYERS)
POLICY_FILE_SIZE_LIMIT = 1024 * 1024
BUNDLE_ID = "adaptive-score-mix-vs-frozen-static-mix"
CONFIRMATORY_EVENT_FIELDS = frozenset(
    {
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
)


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
_FORECAST_POLICY_FIELDS = {
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
_FEATURE_FIELDS = {
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
_ACTION_FIELDS = {
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
    "predicted_adaptive_static_score_gap",
    "x_as",
    "forecast_beta",
    "predicted_adaptive_static_gain",
    "tie_policy",
    "commitment_hash",
}


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ContractError(f"{name} must be a finite number")
    return result


def _nonempty_text(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 512
        or any(ord(char) < 32 for char in value)
    ):
        raise ContractError(f"{name} must be a non-empty bounded string")
    return value


def _full_hash(name: str, value: Any) -> str:
    value = _nonempty_text(name, value).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ContractError(f"{name} must be a full SHA-256 digest")
    return value


def _hashed_payload(payload: Mapping[str, Any], hash_field: str) -> str:
    source = dict(payload)
    source.pop(hash_field, None)
    return sha256_bytes(canonical_json(source).encode("utf-8"))


def _strict_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    stat = source.stat()
    if (
        not source.is_file()
        or stat.st_size <= 0
        or stat.st_size > POLICY_FILE_SIZE_LIMIT
    ):
        raise ContractError("policy JSON must be a non-empty bounded regular file")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"policy JSON contains duplicate key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ContractError(f"policy JSON contains non-finite constant: {value}")

    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("policy file is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ContractError("policy JSON must contain one object")
    return payload


def _validate_layer_weights(
    value: Any,
    *,
    name: str,
) -> tuple[float, ...]:
    if not isinstance(value, Mapping) or set(value) != set(LAYER_ACTIONS):
        raise ContractError(f"{name} must contain exactly five layer weights")
    weights = tuple(_finite(f"{name}.{action}", value[action]) for action in LAYER_ACTIONS)
    if any(weight < 0.0 for weight in weights):
        raise ContractError(f"{name} must be non-negative")
    norm = math.sqrt(sum(weight * weight for weight in weights))
    if not math.isclose(norm, 1.0, rel_tol=2e-7, abs_tol=2e-7):
        raise ContractError(f"{name} must have unit L2 norm")
    return weights


@dataclass(frozen=True, slots=True)
class StaticPolicyLock:
    model_alias: str
    source_runs: Mapping[str, str]
    feature_panel_sha256: str
    weights: tuple[float, ...]
    predicted_mean_score: float
    controller_branch: str
    policy_hash: str

    @property
    def layer_weights(self) -> dict[str, float]:
        return dict(zip(LAYER_ACTIONS, self.weights))


@dataclass(frozen=True, slots=True)
class ForecastPolicyLock:
    model_alias: str
    static_policy_hash: str
    source_runs: Mapping[str, str]
    beta: float
    calibration_method: str
    calibration_hash: str
    policy_hash: str


@dataclass(frozen=True, slots=True)
class ForecastTerms:
    adaptive_score: float
    static_score: float
    predicted_score_gap: float
    x_as: float
    predicted_gain: float


def validate_static_policy(
    payload: Mapping[str, Any],
    *,
    model_alias: str,
) -> StaticPolicyLock:
    """Validate the exact slope-only D0+D1 static-policy artifact."""

    if set(payload) != _STATIC_POLICY_FIELDS:
        raise ContractError("static policy schema fields differ from the lock")
    if payload.get("schema_version") != STATIC_POLICY_SCHEMA:
        raise ContractError("static policy schema version mismatch")
    if payload.get("model_alias") != model_alias or model_alias not in SOURCE_RUN_IDS:
        raise ContractError("static policy model mismatch")
    expected_runs = SOURCE_RUN_IDS[model_alias]
    if payload.get("source_runs") != expected_runs:
        raise ContractError("static policy source runs mismatch")
    if payload.get("source_slices") != {
        "d0": {"start": 3, "count": 5},
        "d1": {"start": 8, "count": 12},
    }:
        raise ContractError("static policy source slices mismatch")
    if payload.get("feature_case_count") != 17:
        raise ContractError("static policy feature denominator mismatch")
    feature_panel_sha256 = _full_hash(
        "static feature panel hash",
        payload.get("feature_panel_sha256"),
    )
    if payload.get("fit_numeric_inputs") != [
        f"action_scores.{action}" for action in LAYER_ACTIONS
    ]:
        raise ContractError("static policy fit inputs mismatch")
    if payload.get("outcome_fields_used") != []:
        raise ContractError("static policy must not use outcome fields")
    slopes = payload.get("sbar_single_layer_slopes")
    if not isinstance(slopes, Mapping) or set(slopes) != set(LAYER_ACTIONS):
        raise ContractError("static policy slope panel mismatch")
    normalized_slopes = {
        action: _finite(f"static slope {action}", slopes[action])
        for action in LAYER_ACTIONS
    }
    weights = _validate_layer_weights(
        payload.get("layer_weights"),
        name="static layer_weights",
    )
    observed_norm = _finite("static weight_l2_norm", payload.get("weight_l2_norm"))
    computed_norm = math.sqrt(sum(weight * weight for weight in weights))
    if not math.isclose(
        observed_norm,
        computed_norm,
        rel_tol=2e-7,
        abs_tol=2e-7,
    ):
        raise ContractError("static policy recorded L2 norm mismatch")
    predicted_mean_score = _finite(
        "static predicted_mean_score",
        payload.get("predicted_mean_score"),
    )
    expected_decision = derive_score_mix_decision(
        normalized_slopes,
        layers=LAYERS,
    )
    if (
        any(
            not math.isclose(
                observed,
                expected,
                rel_tol=2e-12,
                abs_tol=2e-12,
            )
            for observed, expected in zip(weights, expected_decision.weights)
        )
        or not math.isclose(
            predicted_mean_score,
            expected_decision.predicted_score,
            rel_tol=2e-12,
            abs_tol=2e-12,
        )
        or payload.get("controller_branch") != expected_decision.controller_branch
    ):
        raise ContractError("static policy weights do not match frozen mean slopes")
    if payload.get("controller") != "relu(sbar)/L2-else-max-onehot":
        raise ContractError("static policy controller mismatch")
    controller_branch = payload.get("controller_branch")
    if controller_branch not in (
        "positive-relu-l2",
        "all-nonpositive-max-onehot",
    ):
        raise ContractError("static policy controller branch mismatch")
    if payload.get("tie_policy") != "exact-tie-lower-layer":
        raise ContractError("static policy tie rule mismatch")
    _nonempty_text("static claim boundary", payload.get("claim_boundary"))
    policy_hash = _full_hash("static policy hash", payload.get("policy_hash"))
    if policy_hash != _hashed_payload(payload, "policy_hash"):
        raise ContractError("static policy hash mismatch")
    return StaticPolicyLock(
        model_alias=model_alias,
        source_runs=dict(expected_runs),
        feature_panel_sha256=feature_panel_sha256,
        weights=weights,
        predicted_mean_score=predicted_mean_score,
        controller_branch=controller_branch,
        policy_hash=policy_hash,
    )


def load_static_policy(
    path: str | Path,
    *,
    model_alias: str,
) -> StaticPolicyLock:
    return validate_static_policy(
        _strict_json_object(path),
        model_alias=model_alias,
    )


def validate_forecast_policy(
    payload: Mapping[str, Any],
    *,
    model_alias: str,
    static_policy_hash: str,
) -> ForecastPolicyLock:
    """Validate the calibration-only forecast contract.

    The loader is deliberately isolated from event execution so a future
    schema revision can be implemented without changing the outcome path.
    """

    if set(payload) != _FORECAST_POLICY_FIELDS:
        raise ContractError("forecast policy schema fields differ from the lock")
    if payload.get("schema_version") != FORECAST_POLICY_SCHEMA:
        raise ContractError("forecast policy schema version mismatch")
    if payload.get("model_alias") != model_alias or model_alias not in SOURCE_RUN_IDS:
        raise ContractError("forecast policy model mismatch")
    expected_runs = SOURCE_RUN_IDS[model_alias]
    if payload.get("source_runs") != expected_runs:
        raise ContractError("forecast policy source runs mismatch")
    observed_static_hash = _full_hash(
        "forecast static policy hash",
        payload.get("static_policy_hash"),
    )
    if observed_static_hash != _full_hash(
        "expected static policy hash",
        static_policy_hash,
    ):
        raise ContractError("forecast/static policy linkage mismatch")
    beta = _finite("forecast beta", payload.get("beta"))
    if beta < 0.0:
        raise ContractError("forecast beta must be non-negative")
    calibration_method = _nonempty_text(
        "forecast calibration method",
        payload.get("calibration_method"),
    )
    if calibration_method != CALIBRATION_METHOD:
        raise ContractError("forecast calibration method differs from the lock")
    calibration_hash = _full_hash(
        "forecast calibration hash",
        payload.get("calibration_hash"),
    )
    if payload.get("confirmatory_outcomes_used") != []:
        raise ContractError("forecast policy must not use confirmatory outcomes")
    policy_hash = _full_hash("forecast policy hash", payload.get("policy_hash"))
    if policy_hash != _hashed_payload(payload, "policy_hash"):
        raise ContractError("forecast policy hash mismatch")
    return ForecastPolicyLock(
        model_alias=model_alias,
        static_policy_hash=observed_static_hash,
        source_runs=dict(expected_runs),
        beta=beta,
        calibration_method=calibration_method,
        calibration_hash=calibration_hash,
        policy_hash=policy_hash,
    )


def load_forecast_policy(
    path: str | Path,
    *,
    model_alias: str,
    static_policy_hash: str,
) -> ForecastPolicyLock:
    return validate_forecast_policy(
        _strict_json_object(path),
        model_alias=model_alias,
        static_policy_hash=static_policy_hash,
    )


def select_score_mix_wave_cases(
    selection: CounterFactSelectionManifest,
    *,
    wave: ScoreMixWaveLock,
) -> tuple[tuple[str, ...], ConfirmatoryFoldManifest]:
    """Return one exact outcome-blind wave in fixed-selection order."""

    wave = _validated_score_mix_wave_lock(wave)
    if (
        selection.seed != DEFAULT_SELECTION_SEED
        or len(selection.calibration) != 20
        or len(selection.confirmatory) != 60
        or len(selection.untouched) != 20
        or set(selection.calibration).intersection(selection.confirmatory)
        or set(selection.calibration).intersection(selection.untouched)
        or set(selection.confirmatory).intersection(selection.untouched)
    ):
        raise ContractError(
            "confirmatory runner requires the canonical seed and fixed 20/60/20 split"
        )
    folds = build_confirmatory_fold_manifest(
        selection.confirmatory,
        seed=CONFIRMATORY_FOLD_SEED,
        fold_count=CONFIRMATORY_FOLD_COUNT,
    )
    if wave.selected_split == "confirmatory":
        selected = tuple(
            case_id
            for case_id in selection.confirmatory
            if folds.fold_for(case_id) == wave.fold
        )
        allowed = set(selection.confirmatory)
    elif wave.selected_split == "untouched":
        selected = tuple(selection.untouched)
        allowed = set(selection.untouched)
    else:  # ScoreMixWaveLock already rejects this; retain a local fail-closed guard.
        raise ContractError("score-mix wave selected split is invalid")
    if (
        len(selected) != wave.case_count
        or len(set(selected)) != wave.case_count
        or not set(selected).issubset(allowed)
        or set(selected).intersection(selection.calibration)
        or (
            wave.selected_split == "untouched"
            and set(selected).intersection(selection.confirmatory)
        )
    ):
        raise ContractError("score-mix wave cases differ from the exact envelope")
    return selected, folds


def select_confirmatory_fold_cases(
    selection: CounterFactSelectionManifest,
) -> tuple[tuple[str, ...], ConfirmatoryFoldManifest]:
    """Return C1 fold 0 in fixed-selection order, using IDs only."""

    return select_score_mix_wave_cases(
        selection,
        wave=CONFIRMATORY_WAVE_LOCK,
    )


def _score_mix_execution_envelope(
    model_alias: str,
    run_id: str,
    *,
    wave: ScoreMixWaveLock,
) -> str:
    wave = _validated_score_mix_wave_lock(wave)
    try:
        expected = wave.run_ids[model_alias]
    except KeyError as exc:
        raise MV1Error("model is outside the confirmatory envelope") from exc
    if run_id != expected:
        raise MV1Error("run ID is outside the confirmatory envelope")
    return wave.label


def _execution_envelope(model_alias: str, run_id: str) -> str:
    return _score_mix_execution_envelope(
        model_alias,
        run_id,
        wave=CONFIRMATORY_WAVE_LOCK,
    )


def _score_mix_slurm_state(
    model_alias: str,
    run_id: str,
    *,
    wave: ScoreMixWaveLock,
) -> dict[str, Any]:
    wave = _validated_score_mix_wave_lock(wave)
    label = _score_mix_execution_envelope(model_alias, run_id, wave=wave)
    values = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURMD_NODENAME"),
    }
    if all(value is None for value in values.values()):
        return {"under_slurm": False}
    if any(value is None for value in values.values()):
        raise MV1Error("partial Slurm identity is forbidden")
    if (
        values["job_name"] != wave.job_name
        or values["node"] != "devbox"
        or not str(values["job_id"]).isdigit()
    ):
        raise MV1Error("Slurm identity is outside the confirmatory envelope")
    return {
        "under_slurm": True,
        "job_id": values["job_id"],
        "job_name": values["job_name"],
        "node": values["node"],
        "slice_label": label,
        "fold": wave.fold,
    }


def _confirmatory_slurm_state(
    model_alias: str,
    run_id: str,
) -> dict[str, Any]:
    return _score_mix_slurm_state(
        model_alias,
        run_id,
        wave=CONFIRMATORY_WAVE_LOCK,
    )


def _validate_execution_mode(
    *,
    slurm_state: Mapping[str, Any],
    model_loader: Callable[[str], FixedModelRuntime],
    event_runner: Callable[..., Mapping[str, Any]],
) -> bool:
    """Return whether this is production, closing both injectable test seams."""

    under_slurm = slurm_state.get("under_slurm")
    if under_slurm is True:
        if (
            model_loader is not load_fixed_model
            or event_runner is not _run_confirmatory_event
        ):
            raise MV1Error("production confirmatory execution forbids injected seams")
        return True
    if under_slurm is not False:
        raise MV1Error("confirmatory Slurm state is malformed")
    if (
        model_loader is load_fixed_model
        or event_runner is _run_confirmatory_event
    ):
        raise MV1Error(
            "non-Slurm confirmatory execution requires both injected test seams"
        )
    return False


def _run_read_only_git(
    repo_root: Path,
    arguments: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
            "LANG": "C",
            "PATH": "/usr/bin:/bin",
        }
    )
    return subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=repo_root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=10,
    )


def _anchor_production_policy_paths(
    *,
    model_alias: str,
    static_policy_path: str | Path,
    forecast_policy_path: str | Path,
) -> tuple[Path, Path]:
    """Bind production policies to exact clean tracked repo files."""

    try:
        expected = CONFIRMATORY_POLICY_PATHS[model_alias]
    except KeyError as exc:
        raise MV1Error("model has no production policy-path lock") from exc
    repo_root = Path(__file__).resolve().parents[3]
    top_level = _run_read_only_git(repo_root, ("rev-parse", "--show-toplevel"))
    if (
        top_level.returncode != 0
        or Path(top_level.stdout.strip()).resolve() != repo_root
    ):
        raise MV1Error("confirmatory production repository identity mismatch")

    supplied = {
        "static": Path(static_policy_path).expanduser().resolve(strict=True),
        "forecast": Path(forecast_policy_path).expanduser().resolve(strict=True),
    }
    anchored: dict[str, Path] = {}
    relative_paths: list[str] = []
    for label in ("static", "forecast"):
        relative = expected[label]
        lexical = repo_root / relative
        if lexical.is_symlink() or not lexical.is_file():
            raise MV1Error(f"{label} policy must be a regular non-symlink file")
        resolved = lexical.resolve(strict=True)
        try:
            resolved.relative_to(repo_root)
        except ValueError as exc:
            raise MV1Error(f"{label} policy escapes the repository") from exc
        if supplied[label] != resolved:
            raise MV1Error(f"{label} policy path differs from the production lock")
        anchored[label] = resolved
        relative_paths.append(relative)

        tracked = _run_read_only_git(
            repo_root,
            (
                "ls-files",
                "--error-unmatch",
                "--stage",
                "--",
                relative,
            ),
        )
        fields = tracked.stdout.strip().split(maxsplit=3)
        if (
            tracked.returncode != 0
            or len(fields) != 4
            or fields[0] != "100644"
            or fields[2] != "0"
        ):
            raise MV1Error(f"{label} policy is not one regular stage-0 tracked file")
        indexed_blob = fields[1]
        observed_blob = _run_read_only_git(
            repo_root,
            ("hash-object", "--path", relative, str(resolved)),
        )
        if (
            observed_blob.returncode != 0
            or observed_blob.stdout.strip() != indexed_blob
        ):
            raise MV1Error(f"{label} policy content differs from the tracked index")

    clean = _run_read_only_git(
        repo_root,
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
            "--",
            *relative_paths,
        ),
    )
    if clean.returncode != 0 or clean.stdout:
        raise MV1Error("production policy files must be tracked and clean")
    return anchored["static"], anchored["forecast"]


def build_frozen_static_direction(
    *,
    synchronous: MemitFactorProposal,
    unit_actions: Sequence[ActionDirection],
    policy: StaticPolicyLock,
    covariance_by_layer: Mapping[int, torch.Tensor],
    layer_by_weight: Mapping[str, int],
) -> ActionDirection:
    """Build the frozen model-level mix from the already computed unit actions."""

    decision = ScoreMixDecision(
        layers=LAYERS,
        slopes=(0.0,) * EXPECTED_LAYER_COUNT,
        weights=policy.weights,
        predicted_score=policy.predicted_mean_score,
        controller_branch=policy.controller_branch,
    )
    mixed = build_score_mix_direction(
        synchronous=synchronous,
        unit_actions=unit_actions,
        decision=decision,
        covariance_by_layer=covariance_by_layer,
        layer_by_weight=layer_by_weight,
    )
    return ActionDirection(
        action_id=FROZEN_STATIC_ACTION,
        proposal=mixed.proposal,
        c_squared_norm=mixed.c_squared_norm,
    )


def compute_forecast_terms(
    *,
    single_layer_scores: Mapping[str, Any],
    adaptive_weights: Mapping[str, Any],
    static_weights: Mapping[str, Any],
    operational_c_distance: Any,
    beta: Any,
) -> ForecastTerms:
    if set(single_layer_scores) != set(LAYER_ACTIONS):
        raise ContractError("forecast requires exactly five single-layer scores")
    adaptive = _validate_layer_weights(
        adaptive_weights,
        name="adaptive forecast weights",
    )
    static = _validate_layer_weights(
        static_weights,
        name="static forecast weights",
    )
    slopes = tuple(
        _finite(f"forecast score {action}", single_layer_scores[action])
        for action in LAYER_ACTIONS
    )
    distance = _finite(
        "forecast operational C distance",
        operational_c_distance,
    )
    if distance <= 0.0:
        raise ContractError("forecast operational C distance must be positive")
    coefficient = _finite("forecast beta", beta)
    if coefficient < 0.0:
        raise ContractError("forecast beta must be non-negative")
    adaptive_score = sum(weight * score for weight, score in zip(adaptive, slopes))
    static_score = sum(weight * score for weight, score in zip(static, slopes))
    gap = adaptive_score - static_score
    x_as = distance * gap
    predicted_gain = coefficient * x_as
    if any(
        not math.isfinite(value)
        for value in (adaptive_score, static_score, gap, x_as, predicted_gain)
    ):
        raise ContractError("forecast terms must be finite")
    return ForecastTerms(
        adaptive_score=adaptive_score,
        static_score=static_score,
        predicted_score_gap=gap,
        x_as=x_as,
        predicted_gain=predicted_gain,
    )


def commit_confirmatory_action(
    *,
    feature_writer: SanitizedJsonlWriter,
    action_writer: SanitizedJsonlWriter,
    receipt_root: Path,
    feature: Mapping[str, Any],
    action: Mapping[str, Any],
) -> tuple[str, str]:
    """Durably commit adaptive/static decisions and forecast before outcomes."""

    if set(feature) != _FEATURE_FIELDS or set(action) != _ACTION_FIELDS:
        raise ContractError("confirmatory feature/action schema differs from the lock")
    _assert_outcome_free(feature)
    _assert_outcome_free(action)
    if (
        feature["case_id"] != action["case_id"]
        or feature["request_id"] != action["request_id"]
        or feature["q"] != SCORE_MIX_Q
        or action["q"] != SCORE_MIX_Q
        or action["feature_hash"] != feature["feature_hash"]
        or action["bundle_id"] != BUNDLE_ID
        or action["adaptive_action_id"] != SCORE_MIX_ACTION
        or action["static_action_id"] != FROZEN_STATIC_ACTION
        or action["static_policy_hash"] != feature["static_policy_hash"]
        or action["forecast_policy_hash"] != feature["forecast_policy_hash"]
        or action["adaptive_layer_weights"] != feature["adaptive_layer_weights"]
        or action["static_layer_weights"] != feature["static_layer_weights"]
        or action["predicted_adaptive_static_score_gap"]
        != feature["predicted_adaptive_static_score_gap"]
        or action["x_as"] != feature["x_as"]
        or action["forecast_beta"] != feature["forecast_beta"]
        or action["predicted_adaptive_static_gain"]
        != feature["predicted_adaptive_static_gain"]
    ):
        raise ContractError("confirmatory feature/action identity mismatch")
    if (
        _feature_hash(
            {key: value for key, value in feature.items() if key != "feature_hash"}
        )
        != feature["feature_hash"]
        or _feature_hash(
            {
                key: value
                for key, value in action.items()
                if key != "commitment_hash"
            }
        )
        != action["commitment_hash"]
    ):
        raise ContractError("confirmatory feature/action hash mismatch")
    _validate_layer_weights(
        feature["adaptive_layer_weights"],
        name="committed adaptive weights",
    )
    _validate_layer_weights(
        feature["static_layer_weights"],
        name="committed static weights",
    )
    for field in (
        "predicted_adaptive_static_score_gap",
        "x_as",
        "forecast_beta",
        "predicted_adaptive_static_gain",
    ):
        _finite(f"committed {field}", feature[field])
    _safe_payload(feature)
    _safe_payload(action)

    feature_writer.write("mv1mix_confirmatory_feature", feature)
    action_writer.write("mv1mix_confirmatory_action_commitment", action)
    feature_writer.sync()
    action_writer.sync()
    receipt_payload = {
        "schema_version": CONFIRMATORY_RECEIPT_SCHEMA,
        "case_id": feature["case_id"],
        "request_id": feature["request_id"],
        "q": SCORE_MIX_Q,
        "feature_hash": feature["feature_hash"],
        "commitment_hash": action["commitment_hash"],
        "bundle_id": BUNDLE_ID,
        "adaptive_action_id": SCORE_MIX_ACTION,
        "static_action_id": FROZEN_STATIC_ACTION,
        "adaptive_layer_weights": action["adaptive_layer_weights"],
        "static_layer_weights": action["static_layer_weights"],
        "static_policy_hash": action["static_policy_hash"],
        "forecast_policy_hash": action["forecast_policy_hash"],
        "predicted_adaptive_static_score_gap": action[
            "predicted_adaptive_static_score_gap"
        ],
        "x_as": action["x_as"],
        "forecast_beta": action["forecast_beta"],
        "predicted_adaptive_static_gain": action[
            "predicted_adaptive_static_gain"
        ],
        "durability": (
            "feature+adaptive-static-forecast-action-write+flush+fsync-"
            "before-exclusive-receipt"
        ),
        "outcomes_observed_before_commitment": False,
    }
    _assert_outcome_free(
        {
            key: value
            for key, value in receipt_payload.items()
            if key != "outcomes_observed_before_commitment"
        }
    )
    receipt_identity = {
        "case_id": feature["case_id"],
        "request_id": feature["request_id"],
        "q": SCORE_MIX_Q,
        "feature_hash": feature["feature_hash"],
        "commitment_hash": action["commitment_hash"],
        "static_policy_hash": action["static_policy_hash"],
        "forecast_policy_hash": action["forecast_policy_hash"],
    }
    receipt_name = (
        hashlib.sha256(canonical_json(receipt_identity).encode("utf-8")).hexdigest()
        + ".json"
    )
    receipt_path = receipt_root / receipt_name
    _write_json_exclusive(receipt_path, receipt_payload)
    return receipt_name, _file_sha256(receipt_path)


def _run_confirmatory_event(
    *,
    runtime: FixedModelRuntime,
    bridge: EasyEditBridge,
    hparams: Any,
    contexts: Any,
    covariance_specs: Sequence[CovarianceCacheSpec],
    covariance_moments: Mapping[int, torch.Tensor],
    direct_z_root: Path,
    receipt_root: Path,
    request: EditRequest,
    seed: int,
    static_policy: StaticPolicyLock,
    forecast_policy: ForecastPolicyLock,
    feature_writer: SanitizedJsonlWriter,
    action_writer: SanitizedJsonlWriter,
    outcome_writer: SanitizedJsonlWriter,
) -> dict[str, Any]:
    event_seed = _event_seed(seed, request.case_id)
    seed_runtime(event_seed)
    if (
        len(request.request_id) != 64
        or any(char not in "0123456789abcdef" for char in request.request_id)
    ):
        raise MV1Error("request_id must be a full canonical SHA-256 digest")
    if (
        static_policy.model_alias != runtime.spec.alias
        or forecast_policy.model_alias != runtime.spec.alias
        or forecast_policy.static_policy_hash != static_policy.policy_hash
    ):
        raise MV1Error("event policy/model linkage mismatch")

    layer_by_weight = _layer_by_weight(hparams)
    if (
        tuple(int(layer) for layer in hparams.layers) != LAYERS
        or len(layer_by_weight) != EXPECTED_LAYER_COUNT
    ):
        raise MV1Error("confirmatory requires the five locked ascending MEMIT layers")
    weight_names = tuple(layer_by_weight)
    base_hashes = _weight_hashes(runtime.model, weight_names)
    state_identity = _state_identity(runtime)
    initial_rng_hash = rng_state_hash()

    baseline_teacher, target_ids = teacher_forced_rewrite_exact(
        runtime,
        request,
        contexts,
    )
    baseline = rewrite_metrics(baseline_teacher, target_ids)
    if (
        _weight_hashes(runtime.model, weight_names) != base_hashes
        or _state_identity(runtime) != state_identity
        or rng_state_hash() != initial_rng_hash
    ):
        raise RollbackError("baseline evaluation changed the frozen runtime state")

    with _silence_upstream():
        direct_z = bridge.load_or_compute_direct_z(
            runtime.model,
            runtime.tokenizer,
            (request,),
            hparams,
            contexts,
            model_id=runtime.spec.snapshot_name,
            local_cache_root=direct_z_root,
            cache_path=f"{request.request_id}.pt",
        )
        synchronous = bridge.propose_synchronous_memit_factors(
            runtime.model,
            runtime.tokenizer,
            (request,),
            hparams,
            contexts,
            direct_z,
            covariance_specs,
            model_id=runtime.spec.snapshot_name,
        )
        ordered = bridge.propose_ordered_memit_factors(
            runtime.model,
            runtime.tokenizer,
            (request,),
            hparams,
            contexts,
            model_id=runtime.spec.snapshot_name,
            direct_z=direct_z,
            covariance_caches=covariance_specs,
        )
    synchronous.assert_same_entry_snapshot(ordered)
    if (
        _weight_hashes(runtime.model, weight_names) != base_hashes
        or _state_identity(runtime) != state_identity
        or rng_state_hash() != initial_rng_hash
    ):
        raise RollbackError("proposal construction changed the frozen runtime state")

    native_energy = proposal_c_energy(
        ordered,
        covariance_moments,
        layer_by_weight,
    )
    unit_actions = build_unit_c_actions(
        synchronous,
        covariance_moments,
        layer_by_weight,
    )
    expected_action_ids = (*LAYER_ACTIONS, ACTION_UNIFORM)
    assert_exact_action_contract(
        synchronous,
        ordered,
        unit_actions,
        expected_factor_names=weight_names,
        expected_action_ids=expected_action_ids,
        covariance_by_layer=covariance_moments,
        layer_by_weight=layer_by_weight,
    )
    distance = math.sqrt(SCORE_MIX_Q * native_energy)
    epsilon = PROBE_RATIO * distance
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise MV1Error("confirmatory probe distance must be finite and positive")

    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(0).clone()
    branch_rng_hash = rng_state_hash()
    probe_cache: dict[tuple[str, float], Any] = {}

    def evaluate_probe(
        action: ActionDirection,
        signed_distance: float,
    ) -> Any:
        key = (action.action_id, round(signed_distance, 15))
        if key not in probe_cache:
            probe_cache[key] = _evaluate_branch(
                runtime=runtime,
                request=request,
                contexts=contexts,
                target_ids=target_ids,
                proposal=scale_proposal(
                    action.proposal,
                    signed_distance,
                    solver_suffix=(
                        f"score-mix-confirmatory-probe-{signed_distance:.12g}"
                    ),
                ),
                exact_application=False,
                base_hashes=base_hashes,
                state_identity=state_identity,
                cpu_rng=cpu_rng,
                cuda_rng=cuda_rng,
            )
        return probe_cache[key]

    scores: dict[str, float] = {}
    context_scores: dict[str, list[float]] = {}
    for action in unit_actions:
        plus = evaluate_probe(action, epsilon)
        minus = evaluate_probe(action, -epsilon)
        scores[action.action_id] = (plus.utility - minus.utility) / (2.0 * epsilon)
        context_scores[action.action_id] = [
            (right - left) / (2.0 * epsilon)
            for left, right in zip(
                minus.context_utility,
                plus.context_utility,
            )
        ]
    if tuple(scores) != expected_action_ids:
        raise MV1Error("confirmatory central-FD panel differs from the lock")

    decision = derive_score_mix_decision(
        {action_id: scores[action_id] for action_id in LAYER_ACTIONS},
        layers=LAYERS,
    )
    adaptive_action = build_score_mix_direction(
        synchronous=synchronous,
        unit_actions=unit_actions,
        decision=decision,
        covariance_by_layer=covariance_moments,
        layer_by_weight=layer_by_weight,
    )
    static_action = build_frozen_static_direction(
        synchronous=synchronous,
        unit_actions=unit_actions,
        policy=static_policy,
        covariance_by_layer=covariance_moments,
        layer_by_weight=layer_by_weight,
    )
    terms = compute_forecast_terms(
        single_layer_scores={
            action_id: scores[action_id] for action_id in LAYER_ACTIONS
        },
        adaptive_weights=decision.layer_weights,
        static_weights=static_policy.layer_weights,
        operational_c_distance=distance,
        beta=forecast_policy.beta,
    )
    if not math.isclose(
        terms.adaptive_score,
        decision.predicted_score,
        rel_tol=2e-12,
        abs_tol=2e-12,
    ):
        raise MV1Error("adaptive score differs from committed forecast term")

    q_label = "q_1_256"
    feature_payload: dict[str, Any] = {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "q": SCORE_MIX_Q,
        "q_label": q_label,
        "native_c_energy": native_energy,
        "operational_c_distance": distance,
        "probe_c_distance": epsilon,
        "action_scores": scores,
        "context_action_scores": context_scores,
        "feature_policy": (
            "same-snapshot/six-unit-c/central-fd/epsilon=d_over_4/"
            "temperature-1-mean-context-token"
        ),
        "static_policy_hash": static_policy.policy_hash,
        "forecast_policy_hash": forecast_policy.policy_hash,
        "adaptive_layer_weights": decision.layer_weights,
        "static_layer_weights": static_policy.layer_weights,
        "adaptive_predicted_score": terms.adaptive_score,
        "static_predicted_score": terms.static_score,
        "predicted_adaptive_static_score_gap": terms.predicted_score_gap,
        "x_as": terms.x_as,
        "forecast_beta": forecast_policy.beta,
        "predicted_adaptive_static_gain": terms.predicted_gain,
    }
    feature_payload["feature_hash"] = _feature_hash(feature_payload)
    action_payload: dict[str, Any] = {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "q": SCORE_MIX_Q,
        "q_label": q_label,
        "feature_hash": feature_payload["feature_hash"],
        "bundle_id": BUNDLE_ID,
        "adaptive_action_id": SCORE_MIX_ACTION,
        "adaptive_layer_weights": decision.layer_weights,
        "adaptive_predicted_score": terms.adaptive_score,
        "adaptive_actual_unit_c_energy": adaptive_action.c_squared_norm,
        "adaptive_controller": "five-single-slopes/relu-l2-else-max-onehot",
        "adaptive_controller_branch": decision.controller_branch,
        "static_action_id": FROZEN_STATIC_ACTION,
        "static_layer_weights": static_policy.layer_weights,
        "static_predicted_score": terms.static_score,
        "static_actual_unit_c_energy": static_action.c_squared_norm,
        "static_policy_hash": static_policy.policy_hash,
        "forecast_policy_hash": forecast_policy.policy_hash,
        "predicted_adaptive_static_score_gap": terms.predicted_score_gap,
        "x_as": terms.x_as,
        "forecast_beta": forecast_policy.beta,
        "predicted_adaptive_static_gain": terms.predicted_gain,
        "tie_policy": "exact-tie-lower-layer",
    }
    action_payload["commitment_hash"] = _feature_hash(action_payload)
    receipt_name, receipt_sha256 = commit_confirmatory_action(
        feature_writer=feature_writer,
        action_writer=action_writer,
        receipt_root=receipt_root,
        feature=feature_payload,
        action=action_payload,
    )

    # No operational outcome branch is reachable before the durable receipt.
    operational_cache: dict[str, Any] = {}

    def evaluate_operational(
        action_id: str,
        proposal: MemitFactorProposal | None,
        *,
        exact_application: bool,
    ) -> Any:
        if action_id not in operational_cache:
            operational_cache[action_id] = _evaluate_branch(
                runtime=runtime,
                request=request,
                contexts=contexts,
                target_ids=target_ids,
                proposal=proposal,
                exact_application=exact_application,
                base_hashes=base_hashes,
                state_identity=state_identity,
                cpu_rng=cpu_rng,
                cuda_rng=cuda_rng,
            )
        return operational_cache[action_id]

    uniform_action = unit_actions[-1]
    adaptive_proposal = scale_proposal(
        adaptive_action.proposal,
        distance,
        solver_suffix="score-mix-confirmatory-operational",
    )
    static_proposal = scale_proposal(
        static_action.proposal,
        distance,
        solver_suffix="frozen-static-confirmatory-operational",
    )
    uniform_proposal = scale_proposal(
        uniform_action.proposal,
        distance,
        solver_suffix="uniform-confirmatory-operational",
    )
    ordered_global_alpha = scale_proposal(
        ordered,
        math.sqrt(SCORE_MIX_Q),
        solver_suffix="ordered-global-alpha-q-1-256-confirmatory",
    )
    remeasured_energies = {
        SCORE_MIX_ACTION: adaptive_action.c_squared_norm * distance * distance,
        FROZEN_STATIC_ACTION: static_action.c_squared_norm * distance * distance,
        ACTION_UNIFORM: uniform_action.c_squared_norm * distance * distance,
        ORDERED_GLOBAL_ALPHA_ACTION: native_energy * SCORE_MIX_Q,
    }
    expected_operational_energy = distance * distance
    if any(
        not math.isclose(
            energy,
            expected_operational_energy,
            rel_tol=3e-5,
            abs_tol=3e-5,
        )
        for energy in remeasured_energies.values()
    ):
        raise MV1Error("confirmatory equal-C proposals failed remeasurement")

    adaptive_after = evaluate_operational(
        SCORE_MIX_ACTION,
        adaptive_proposal,
        exact_application=False,
    )
    static_after = evaluate_operational(
        FROZEN_STATIC_ACTION,
        static_proposal,
        exact_application=False,
    )
    uniform_after = evaluate_operational(
        ACTION_UNIFORM,
        uniform_proposal,
        exact_application=False,
    )
    ordered_after = evaluate_operational(
        ORDERED_GLOBAL_ALPHA_ACTION,
        ordered_global_alpha,
        exact_application=False,
    )
    native_after = evaluate_operational(
        NATIVE_MEMIT_ACTION,
        ordered,
        exact_application=True,
    )
    replay = evaluate_operational(
        NO_OP_ACTION,
        None,
        exact_application=False,
    )
    if replay.logits_hash != baseline.logits_hash:
        raise RollbackError("confirmatory no-op logits differ from baseline")

    common = {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "q": SCORE_MIX_Q,
        "q_label": q_label,
        "feature_hash": feature_payload["feature_hash"],
        "commitment_hash": action_payload["commitment_hash"],
        "static_policy_hash": static_policy.policy_hash,
        "forecast_policy_hash": forecast_policy.policy_hash,
        "status": "completed",
        "rollback_exact": True,
    }
    outcomes = (
        {
            **common,
            "action_id": SCORE_MIX_ACTION,
            "selected_by_controller": True,
            "c_distance": distance,
            "c_energy": remeasured_energies[SCORE_MIX_ACTION],
            "budget_validation": "unit-c-remeasured-then-scaled/equal-c",
            **_metric_difference(adaptive_after, baseline),
        },
        {
            **common,
            "action_id": FROZEN_STATIC_ACTION,
            "selected_by_controller": False,
            "c_distance": distance,
            "c_energy": remeasured_energies[FROZEN_STATIC_ACTION],
            "budget_validation": "unit-c-remeasured-then-scaled/equal-c",
            **_metric_difference(static_after, baseline),
        },
        {
            **common,
            "action_id": ACTION_UNIFORM,
            "selected_by_controller": False,
            "c_distance": distance,
            "c_energy": remeasured_energies[ACTION_UNIFORM],
            "budget_validation": "unit-c-remeasured-then-scaled/equal-c",
            **_metric_difference(uniform_after, baseline),
        },
        {
            **common,
            "action_id": ORDERED_GLOBAL_ALPHA_ACTION,
            "selected_by_controller": False,
            "c_distance": distance,
            "c_energy": remeasured_energies[ORDERED_GLOBAL_ALPHA_ACTION],
            "budget_validation": (
                "native-c-remeasured-then-global-alpha-squared/equal-c"
            ),
            **_metric_difference(ordered_after, baseline),
        },
        {
            **common,
            "action_id": NATIVE_MEMIT_ACTION,
            "selected_by_controller": False,
            "c_distance": math.sqrt(native_energy),
            "c_energy": native_energy,
            "budget_validation": "reference-native-full",
            **_metric_difference(native_after, baseline),
        },
        {
            **common,
            "action_id": NO_OP_ACTION,
            "selected_by_controller": False,
            "c_distance": 0.0,
            "c_energy": 0.0,
            "budget_validation": "reference-no-op",
            "logits_hash_equal": True,
            **_metric_difference(replay, baseline),
        },
    )
    if tuple(item["action_id"] for item in outcomes) != CONFIRMATORY_OUTCOME_ACTIONS:
        raise MV1Error("confirmatory runtime outcome arms differ from the lock")
    for payload in outcomes:
        outcome_writer.write("mv1mix_confirmatory_outcome", payload)

    if (
        _weight_hashes(runtime.model, weight_names) != base_hashes
        or _state_identity(runtime) != state_identity
        or rng_state_hash() != branch_rng_hash
    ):
        raise RollbackError("confirmatory event did not restore frozen state")
    return {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "event_seed": event_seed,
        "target_token_count": baseline.target_token_count,
        "native_c_energy": native_energy,
        "q": SCORE_MIX_Q,
        "probe_direction_count": len(scores),
        "feature_count": 1,
        "commitment_count": 1,
        "outcome_count": len(outcomes),
        "direct_z_artifact_sha256": direct_z.artifact.sha256,
        "direct_z_artifact_size": direct_z.artifact.size,
        "static_policy_hash": static_policy.policy_hash,
        "forecast_policy_hash": forecast_policy.policy_hash,
        "action_receipt": {
            "name": receipt_name,
            "sha256": receipt_sha256,
        },
        "rollback_exact": True,
        "pass": True,
    }


def _validate_confirmatory_event_result(
    result: Mapping[str, Any],
    *,
    request: EditRequest,
    expected_static_policy_hash: str,
    expected_forecast_policy_hash: str,
) -> dict[str, Any]:
    normalized = dict(result)
    if set(normalized) != CONFIRMATORY_EVENT_FIELDS:
        raise ContractError("confirmatory event result fields differ from the lock")
    if (
        normalized["case_id"] != request.case_id
        or normalized["request_id"] != request.request_id
        or normalized["q"] != SCORE_MIX_Q
        or normalized["probe_direction_count"] != len(LAYER_ACTIONS) + 1
        or normalized["feature_count"] != 1
        or normalized["commitment_count"] != 1
        or normalized["outcome_count"] != len(CONFIRMATORY_OUTCOME_ACTIONS)
        or normalized["rollback_exact"] is not True
        or normalized["pass"] is not True
    ):
        raise ContractError("confirmatory event result differs from the contract")
    for field in ("event_seed", "target_token_count", "direct_z_artifact_size"):
        value = normalized[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ContractError(f"confirmatory event {field} must be a positive integer")
    native_energy = _finite(
        "confirmatory event native_c_energy",
        normalized["native_c_energy"],
    )
    if native_energy <= 0.0:
        raise ContractError("confirmatory event native C energy must be positive")
    direct_z_hash = _full_hash(
        "confirmatory event direct-z hash",
        normalized["direct_z_artifact_sha256"],
    )
    static_hash = _full_hash(
        "confirmatory event static policy hash",
        normalized["static_policy_hash"],
    )
    forecast_hash = _full_hash(
        "confirmatory event forecast policy hash",
        normalized["forecast_policy_hash"],
    )
    if (
        static_hash != expected_static_policy_hash
        or forecast_hash != expected_forecast_policy_hash
    ):
        raise ContractError("confirmatory event policy hash differs from the lock")
    receipt = normalized["action_receipt"]
    if not isinstance(receipt, Mapping) or set(receipt) != {"name", "sha256"}:
        raise ContractError("confirmatory event receipt schema mismatch")
    receipt_name = _nonempty_text(
        "confirmatory event receipt name",
        receipt["name"],
    )
    if (
        len(receipt_name) != 69
        or not receipt_name.endswith(".json")
        or _full_hash(
            "confirmatory event receipt identity",
            receipt_name[:-5],
        )
        != receipt_name[:-5]
    ):
        raise ContractError("confirmatory event receipt name is not canonical")
    receipt_hash = _full_hash(
        "confirmatory event receipt hash",
        receipt["sha256"],
    )
    normalized["native_c_energy"] = native_energy
    normalized["direct_z_artifact_sha256"] = direct_z_hash
    normalized["static_policy_hash"] = static_hash
    normalized["forecast_policy_hash"] = forecast_hash
    normalized["action_receipt"] = {
        "name": receipt_name,
        "sha256": receipt_hash,
    }
    return normalized


def run_confirmatory_event_loop(
    *,
    requests: Sequence[EditRequest],
    event_runner: Callable[..., Mapping[str, Any]],
    event_writer: SanitizedJsonlWriter,
    event_kwargs: Mapping[str, Any],
    expected_static_policy_hash: str,
    expected_forecast_policy_hash: str,
) -> tuple[list[dict[str, Any]], str | None]:
    """Run fail-closed events through an injectable, pure orchestration seam."""

    expected_static_policy_hash = _full_hash(
        "expected event static policy hash",
        expected_static_policy_hash,
    )
    expected_forecast_policy_hash = _full_hash(
        "expected event forecast policy hash",
        expected_forecast_policy_hash,
    )
    results: list[dict[str, Any]] = []
    abort_failure_type: str | None = None
    for request in requests:
        try:
            raw_result = event_runner(
                request=request,
                **event_kwargs,
            )
            if not isinstance(raw_result, Mapping):
                raise ContractError("event runner must return a mapping")
            result = _validate_confirmatory_event_result(
                raw_result,
                request=request,
                expected_static_policy_hash=expected_static_policy_hash,
                expected_forecast_policy_hash=expected_forecast_policy_hash,
            )
        except Exception as exc:
            result = {
                "case_id": request.case_id,
                "request_id": request.request_id,
                "failure_type": type(exc).__name__,
                "failure_class": "fatal_confirmatory_contract_or_runtime",
                "feature_count": 0,
                "commitment_count": 0,
                "outcome_count": 0,
                "rollback_exact": False,
                "pass": False,
            }
            abort_failure_type = type(exc).__name__
        results.append(result)
        event_writer.write("mv1mix_confirmatory_case", result)
        if abort_failure_type is not None:
            break
    return results, abort_failure_type


def run_score_mix_wave(
    *,
    wave: ScoreMixWaveLock,
    easyedit_root: str | Path,
    model_alias: str,
    run_id: str,
    static_policy_path: str | Path,
    forecast_policy_path: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    seed: int = 17,
    selection_seed: str = DEFAULT_SELECTION_SEED,
    model_loader: Callable[[str], FixedModelRuntime] = load_fixed_model,
    event_runner: Callable[..., Mapping[str, Any]] = _run_confirmatory_event,
) -> dict[str, Any]:
    """Run one exact C1/follow-up slice through the shared C1 event core."""

    wave = _validated_score_mix_wave_lock(wave)
    started = time.perf_counter()
    _score_mix_execution_envelope(model_alias, run_id, wave=wave)
    if selection_seed != DEFAULT_SELECTION_SEED:
        raise MV1Error("confirmatory selection seed differs from the canonical lock")
    if wave.run_seed is not None and (
        type(seed) is not int or seed != wave.run_seed
    ):
        raise MV1Error("score-mix wave seed differs from the fixed lock")
    slurm_state = _score_mix_slurm_state(model_alias, run_id, wave=wave)
    production = _validate_execution_mode(
        slurm_state=slurm_state,
        model_loader=model_loader,
        event_runner=event_runner,
    )
    if production:
        static_policy_path, forecast_policy_path = (
            _anchor_production_policy_paths(
                model_alias=model_alias,
                static_policy_path=static_policy_path,
                forecast_policy_path=forecast_policy_path,
            )
        )
    static_policy = load_static_policy(
        static_policy_path,
        model_alias=model_alias,
    )
    forecast_policy = load_forecast_policy(
        forecast_policy_path,
        model_alias=model_alias,
        static_policy_hash=static_policy.policy_hash,
    )
    root = Path(easyedit_root).expanduser().resolve(strict=True)
    spec = fixed_model_spec(model_alias)
    git_state = _git_runtime_state()
    provenance = preflight_fixed_artifacts(root, model_alias=model_alias)
    selection = generate_counterfact_selection(root, seed=selection_seed)
    case_ids, fold_manifest = select_score_mix_wave_cases(selection, wave=wave)
    requests = load_counterfact_requests(root, case_ids)
    bridge = EasyEditBridge(root, expected_files=_bridge_pins())
    bridge_provenance = bridge.preflight()
    if not set(record.path for record in bridge_provenance.files).issubset(
        set(record.path for record in provenance.files)
    ):
        raise MV1Error("bridge provenance is outside the fixed full manifest")

    with offline_environment():
        bindings = bridge.load()
        hparams = _load_hparams(root, spec, bindings)
        seed_runtime(seed)
        runtime = model_loader(model_alias)
        if runtime.spec != spec:
            raise MV1Error("model loader returned a different fixed model spec")
        contexts = _freeze_contexts(bridge, runtime, seed=seed)
        destination = _local_run_directory(output_root, run_id)
        manifest_path = destination / "manifest.json"
        features_path = destination / "features.jsonl"
        actions_path = destination / "actions.jsonl"
        outcomes_path = destination / "outcomes.jsonl"
        events_path = destination / "events.jsonl"
        summary_path = destination / "summary.json"
        manifest_payload = {
                "schema_version": CONFIRMATORY_MANIFEST_SCHEMA,
                "run_id": run_id,
                "ode_edit_git": git_state,
                "slurm": slurm_state,
                "model": runtime.metadata(),
                "hparams_relative_path": spec.hparams_path,
                "selection": selection.to_dict(),
                "selected_split": wave.selected_split,
                "confirmatory_folds": fold_manifest.to_dict(),
                "selected_fold": {
                    "label": wave.label,
                    "fold": wave.fold,
                    "count": wave.case_count,
                    "case_ids": list(case_ids),
                    **(
                        {"run_seed": wave.run_seed}
                        if wave.run_seed is not None
                        else {}
                    ),
                },
                "selected_case_ids": list(case_ids),
                "selected_request_ids": [
                    request.request_id for request in requests
                ],
                "contexts": {
                    "manifest_id": contexts.manifest_id,
                    "source": contexts.source,
                    "group_sizes": [
                        len(group) for group in contexts.templates
                    ],
                    "raw_templates_persisted": False,
                },
                "provenance_id": provenance.manifest_id,
                "fixed_files": _relative_provenance(provenance, root),
                "q": SCORE_MIX_Q,
                "q_label": "q_1_256",
                "probe_ratio": PROBE_RATIO,
                "probe_action_set": [*LAYER_ACTIONS, ACTION_UNIFORM],
                "controller_input_set": list(LAYER_ACTIONS),
                "outcome_action_set": list(CONFIRMATORY_OUTCOME_ACTIONS),
                "static_policy": {
                    "schema_version": STATIC_POLICY_SCHEMA,
                    "policy_hash": static_policy.policy_hash,
                    "model_alias": static_policy.model_alias,
                    "source_runs": dict(static_policy.source_runs),
                    "feature_panel_sha256": (
                        static_policy.feature_panel_sha256
                    ),
                    "layer_weights": static_policy.layer_weights,
                    "path_persisted": False,
                },
                "forecast_policy": {
                    "schema_version": FORECAST_POLICY_SCHEMA,
                    "policy_hash": forecast_policy.policy_hash,
                    "model_alias": forecast_policy.model_alias,
                    "static_policy_hash": forecast_policy.static_policy_hash,
                    "source_runs": dict(forecast_policy.source_runs),
                    "beta": forecast_policy.beta,
                    "calibration_method": forecast_policy.calibration_method,
                    "calibration_hash": forecast_policy.calibration_hash,
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
                    "features": wave.case_count,
                    "commitments": wave.case_count,
                    "outcomes": (
                        wave.case_count * len(CONFIRMATORY_OUTCOME_ACTIONS)
                    ),
                    "receipts": wave.case_count,
                },
        }
        if set(manifest_payload) != CONFIRMATORY_MANIFEST_FIELDS:
            raise MV1Error("confirmatory manifest fields differ from the lock")
        _write_json_exclusive(manifest_path, manifest_payload)
        covariance_specs = _covariance_specs(root, spec)
        covariance_moments, loaded_covariances = _load_verified_covariances(
            root=root,
            runtime=runtime,
            specs=covariance_specs,
        )
        direct_z_root = destination / "direct_z"
        receipt_root = destination / "action_receipts"
        receipt_root.mkdir(mode=0o700)
        torch.cuda.reset_peak_memory_stats(0)
        with (
            SanitizedJsonlWriter(
                features_path,
                run_id,
                schema_version=CONFIRMATORY_STREAM_SCHEMA,
            ) as feature_writer,
            SanitizedJsonlWriter(
                actions_path,
                run_id,
                schema_version=CONFIRMATORY_STREAM_SCHEMA,
            ) as action_writer,
            SanitizedJsonlWriter(
                outcomes_path,
                run_id,
                schema_version=CONFIRMATORY_STREAM_SCHEMA,
            ) as outcome_writer,
            SanitizedJsonlWriter(
                events_path,
                run_id,
                schema_version=CONFIRMATORY_STREAM_SCHEMA,
            ) as event_writer,
        ):
            event_results, abort_failure_type = run_confirmatory_event_loop(
                requests=requests,
                event_runner=event_runner,
                event_writer=event_writer,
                expected_static_policy_hash=static_policy.policy_hash,
                expected_forecast_policy_hash=forecast_policy.policy_hash,
                event_kwargs={
                    "runtime": runtime,
                    "bridge": bridge,
                    "hparams": hparams,
                    "contexts": contexts,
                    "covariance_specs": covariance_specs,
                    "covariance_moments": covariance_moments,
                    "direct_z_root": direct_z_root,
                    "receipt_root": receipt_root,
                    "seed": seed,
                    "static_policy": static_policy,
                    "forecast_policy": forecast_policy,
                    "feature_writer": feature_writer,
                    "action_writer": action_writer,
                    "outcome_writer": outcome_writer,
                },
            )
            stream_sequences = {
                "features": feature_writer.sequence,
                "actions": action_writer.sequence,
                "outcomes": outcome_writer.sequence,
                "events": event_writer.sequence,
            }

        elapsed = time.perf_counter() - started
        planned = len(requests)
        attempted = len(event_results)
        pass_count = sum(bool(item["pass"]) for item in event_results)
        feature_count = sum(
            int(item.get("feature_count", 0)) for item in event_results
        )
        commitment_count = sum(
            int(item.get("commitment_count", 0)) for item in event_results
        )
        outcome_count = sum(
            int(item.get("outcome_count", 0)) for item in event_results
        )
        receipt_count = len(tuple(receipt_root.glob("*.json")))
        expected_stream_sequences = {
            "features": planned,
            "actions": planned,
            "outcomes": planned * len(CONFIRMATORY_OUTCOME_ACTIONS),
            "events": planned,
        }
        exact_counts = bool(
            feature_count == planned
            and commitment_count == planned
            and outcome_count
            == planned * len(CONFIRMATORY_OUTCOME_ACTIONS)
            and receipt_count == planned
            and stream_sequences == expected_stream_sequences
        )
        summary = {
            "schema_version": CONFIRMATORY_SUMMARY_SCHEMA,
            "run_id": run_id,
            "model_alias": model_alias,
            "slice": {
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
            },
            "slurm": slurm_state,
            "provenance_id": provenance.manifest_id,
            "selection_manifest_id": selection.manifest_id,
            "context_id": contexts.manifest_id,
            "static_policy_hash": static_policy.policy_hash,
            "forecast_policy_hash": forecast_policy.policy_hash,
            "forecast_beta": forecast_policy.beta,
            "run_status": (
                "aborted" if abort_failure_type is not None else "completed"
            ),
            "abort_failure_type": abort_failure_type,
            "planned_case_count": planned,
            "attempted_case_count": attempted,
            "not_run_due_to_abort_count": planned - attempted,
            "pass_count": pass_count,
            "failure_count": planned - pass_count,
            "all_pass": bool(
                abort_failure_type is None
                and attempted == planned
                and pass_count == planned
                and exact_counts
            ),
            "case_results": [
                {
                    "case_id": item["case_id"],
                    "status": "attempted",
                    "pass": bool(item["pass"]),
                }
                for item in event_results
            ]
            + [
                {
                    "case_id": case_id,
                    "status": "not_run_due_to_abort",
                    "pass": False,
                }
                for case_id in case_ids[attempted:]
            ],
            "q": SCORE_MIX_Q,
            "probe_direction_count_per_case": 6,
            "feature_count": feature_count,
            "commitment_count": commitment_count,
            "outcome_count": outcome_count,
            "action_receipt_count": receipt_count,
            "stream_sequences": stream_sequences,
            "expected_counts_exact": exact_counts,
            "all_rollbacks_exact": bool(
                event_results
                and all(bool(item["rollback_exact"]) for item in event_results)
            ),
            "covariance_files_loaded": list(loaded_covariances),
            "projector_files_loaded": [],
            "direct_z_artifact_count": (
                len(tuple(direct_z_root.glob("*.pt")))
                if direct_z_root.is_dir()
                else 0
            ),
            "resource": {
                "wall_seconds": elapsed,
                "gpu_peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(0)
                ),
                "gpu_peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(0)
                ),
                "host_max_rss_kib": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ),
                "visible_gpu_count": 1,
            },
            "artifacts": {
                "manifest_sha256": _file_sha256(manifest_path),
                "features_sha256": _file_sha256(features_path),
                "actions_sha256": _file_sha256(actions_path),
                "outcomes_sha256": _file_sha256(outcomes_path),
                "events_sha256": _file_sha256(events_path),
                "action_receipts": {
                    path.name: _file_sha256(path)
                    for path in sorted(receipt_root.glob("*.json"))
                },
            },
            "git_output_written": False,
        }
        if set(summary) != CONFIRMATORY_SUMMARY_FIELDS:
            raise MV1Error("confirmatory summary fields differ from the lock")
        _write_json_exclusive(summary_path, summary)
        return {**summary, "output_directory": str(destination)}


def run_mv1_score_mix_confirmatory(
    *,
    easyedit_root: str | Path,
    model_alias: str,
    run_id: str,
    static_policy_path: str | Path,
    forecast_policy_path: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    seed: int = 17,
    selection_seed: str = DEFAULT_SELECTION_SEED,
    model_loader: Callable[[str], FixedModelRuntime] = load_fixed_model,
    event_runner: Callable[..., Mapping[str, Any]] = _run_confirmatory_event,
) -> dict[str, Any]:
    """Run one fixed model on exactly confirmatory fold 0 (12 cases)."""

    return run_score_mix_wave(
        wave=CONFIRMATORY_WAVE_LOCK,
        easyedit_root=easyedit_root,
        model_alias=model_alias,
        run_id=run_id,
        static_policy_path=static_policy_path,
        forecast_policy_path=forecast_policy_path,
        output_root=output_root,
        seed=seed,
        selection_seed=selection_seed,
        model_loader=model_loader,
        event_runner=event_runner,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ode-edit-mv1-score-mix-confirmatory",
        description=(
            "Held-out adaptive score-mix versus frozen-static confirmatory"
        ),
    )
    parser.add_argument("--easyedit-root", required=True)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--static-policy", required=True, type=Path)
    parser.add_argument("--forecast-policy", required=True, type=Path)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_mv1_score_mix_confirmatory(
            easyedit_root=args.easyedit_root,
            model_alias=args.model,
            run_id=args.run_id,
            static_policy_path=args.static_policy,
            forecast_policy_path=args.forecast_policy,
            output_root=args.output_root,
            seed=args.seed,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "aborted",
                    "error_type": type(exc).__name__,
                    "raw_exception_persisted": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
