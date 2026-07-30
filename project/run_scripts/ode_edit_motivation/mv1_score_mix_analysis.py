"""Pure-CPU, precommitted analysis for one MV-1 score-mix D0 run.

Only the sanitized JSON contract emitted by ``mv1_score_mix`` is consumed.
Tensor artifacts are neither opened nor named in emitted results.  D0 is a
five-case calibration/early-kill diagnostic: a positive event can authorize
D1, but it cannot establish a method gain, GO decision, or MV-2 entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ANALYSIS_SCHEMA = "ode-edit-mv1-score-mix-analysis/v1"
SMALL_SUMMARY_SCHEMA = "ode-edit-mv1-score-mix-analysis-summary/v1"
MANIFEST_SCHEMA = "ode-edit-mv1-score-mix-manifest/v1"
SUMMARY_SCHEMA = "ode-edit-mv1-score-mix-summary/v1"
STREAM_SCHEMA = "ode-edit-mv1-score-mix/v1"
RECEIPT_SCHEMA = "ode-edit-mv1-score-mix-receipt/v1"

SCORE_MIX_Q = 1.0 / 256.0
NUMERIC_COMPARISON_TOLERANCE = 1e-12
D0_CASE_COUNT = 5
D0_START = 3
D0_JOB_NAME = "odeedit_mv1mix_d0_pair_v1"
LAYERS = (4, 5, 6, 7, 8)
SINGLE_ACTIONS = tuple(f"layer_{layer}" for layer in LAYERS)
PROBE_ACTIONS = (*SINGLE_ACTIONS, "uniform")
OUTCOME_ACTIONS = (
    "score_mix",
    "uniform",
    "ordered_global_alpha",
    "native_memit_full",
    "no_op_replay",
)

MODEL_ENVELOPES: Mapping[str, Mapping[str, str]] = {
    "llama3-8b-inst": {
        "run_id": "mv1mix_llama_d0_v1",
        "repository_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "revision": "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
    },
    "qwen2.5-7b-inst": {
        "run_id": "mv1mix_qwen_d0_v1",
        "repository_id": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
    },
}

_STREAM_FILES = ("features.jsonl", "actions.jsonl", "outcomes.jsonl", "events.jsonl")
_HASHED_FILES = ("manifest.json", *_STREAM_FILES)
_EXPECTED_STREAM_EVENTS = {
    "features.jsonl": "mv1mix_feature",
    "actions.jsonl": "mv1mix_action_commitment",
    "outcomes.jsonl": "mv1mix_outcome",
    "events.jsonl": "mv1mix_case",
}
_FORBIDDEN_KEYS = {
    "answer",
    "answers",
    "attribute_prompts",
    "evaluation",
    "generation_prompts",
    "ground_truth",
    "input_ids",
    "locality",
    "logits",
    "neighborhood_prompts",
    "paraphrase_prompts",
    "predicted_ids",
    "prompt",
    "prompts",
    "raw_generation",
    "raw_generations",
    "requested_rewrite",
    "subject",
    "target",
    "target_ids",
    "target_new",
    "target_true",
    "token_ids",
}
_OUTCOME_ONLY_KEYS = {
    "budget_validation",
    "c_distance",
    "c_energy",
    "context_nll_reduction",
    "context_progress",
    "exact_margin_min",
    "exact_satisfied",
    "logits_hash_equal",
    "nll_reduction",
    "progress",
    "rollback_exact",
    "selected_by_controller",
    "status",
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
    "feature_hash",
}
_ACTION_FIELDS = {
    "case_id",
    "request_id",
    "q",
    "q_label",
    "feature_hash",
    "action_id",
    "layer_weights",
    "predicted_score",
    "actual_unit_c_energy",
    "controller",
    "controller_branch",
    "tie_policy",
    "commitment_hash",
}
_OUTCOME_FIELDS = {
    "case_id",
    "request_id",
    "q",
    "q_label",
    "feature_hash",
    "commitment_hash",
    "status",
    "rollback_exact",
    "action_id",
    "selected_by_controller",
    "c_distance",
    "c_energy",
    "budget_validation",
    "progress",
    "context_progress",
    "nll_reduction",
    "context_nll_reduction",
    "exact_margin_min",
    "exact_satisfied",
}
_EVENT_FIELDS = {
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
    "action_receipt",
    "rollback_exact",
    "pass",
}
_BUDGET_VALIDATION_BY_ACTION = {
    "score_mix": "unit-c-remeasured-then-scaled/equal-c",
    "uniform": "unit-c-remeasured-then-scaled/equal-c",
    "ordered_global_alpha": (
        "native-c-remeasured-then-global-alpha-squared/equal-c"
    ),
    "native_memit_full": "reference-native-full",
    "no_op_replay": "reference-no-op",
}


class ScoreMixAnalysisError(RuntimeError):
    """Raised when the sanitized input cannot be read safely."""


@dataclass(frozen=True, slots=True)
class StreamRow:
    event: str
    payload: dict[str, Any]
    line_number: int


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _is_full_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ScoreMixAnalysisError(f"{name} is not finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ScoreMixAnalysisError(f"{name} is not finite") from exc
    if not math.isfinite(result):
        raise ScoreMixAnalysisError(f"{name} is not finite")
    return result


def _assert_json_safe(value: Any, *, location: str = "root") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ScoreMixAnalysisError(f"non-finite scalar at {location}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_json_safe(item, location=f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ScoreMixAnalysisError(f"non-string key at {location}")
            if key.lower() in _FORBIDDEN_KEYS:
                raise ScoreMixAnalysisError(f"forbidden raw field at {location}.{key}")
            _assert_json_safe(item, location=f"{location}.{key}")
        return
    raise ScoreMixAnalysisError(f"non-JSON value at {location}")


def _read_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ScoreMixAnalysisError(f"required regular file unavailable: {path.name}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScoreMixAnalysisError(f"cannot parse {path.name}") from exc
    if not isinstance(value, dict):
        raise ScoreMixAnalysisError(f"{path.name} must contain one JSON object")
    _assert_json_safe(value, location=path.name)
    return value


def _read_jsonl(path: Path, *, run_id: str) -> list[StreamRow]:
    if path.is_symlink() or not path.is_file():
        raise ScoreMixAnalysisError(f"required regular file unavailable: {path.name}")
    rows: list[StreamRow] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ScoreMixAnalysisError(
                        f"{path.name}:{line_number} is empty"
                    )
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ScoreMixAnalysisError(
                        f"{path.name}:{line_number} must be an object"
                    )
                _assert_json_safe(raw, location=f"{path.name}:{line_number}")
                if set(raw) != {
                    "schema_version",
                    "run_id",
                    "sequence",
                    "recorded_at",
                    "event",
                    "payload",
                }:
                    raise ScoreMixAnalysisError(
                        f"{path.name}:{line_number} envelope mismatch"
                    )
                if raw.get("schema_version") != STREAM_SCHEMA:
                    raise ScoreMixAnalysisError(
                        f"{path.name}:{line_number} schema mismatch"
                    )
                if raw.get("run_id") != run_id:
                    raise ScoreMixAnalysisError(
                        f"{path.name}:{line_number} run_id mismatch"
                    )
                if raw.get("sequence") != line_number - 1:
                    raise ScoreMixAnalysisError(
                        f"{path.name}:{line_number} sequence mismatch"
                    )
                if not isinstance(raw.get("recorded_at"), str) or not raw["recorded_at"]:
                    raise ScoreMixAnalysisError(
                        f"{path.name}:{line_number} timestamp missing"
                    )
                if not isinstance(raw.get("event"), str):
                    raise ScoreMixAnalysisError(
                        f"{path.name}:{line_number} event invalid"
                    )
                if not isinstance(raw.get("payload"), dict):
                    raise ScoreMixAnalysisError(
                        f"{path.name}:{line_number} payload invalid"
                    )
                rows.append(
                    StreamRow(
                        event=raw["event"],
                        payload=dict(raw["payload"]),
                        line_number=line_number,
                    )
                )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScoreMixAnalysisError(f"cannot parse {path.name}") from exc
    return rows


def _add_error(errors: list[str], condition: bool, code: str) -> None:
    if not condition:
        errors.append(code)


def _payload_hash(payload: Mapping[str, Any], hash_field: str) -> tuple[str | None, str]:
    stripped = dict(payload)
    observed = stripped.pop(hash_field, None)
    expected = _sha256_bytes(_canonical_json(stripped).encode("utf-8"))
    return observed if isinstance(observed, str) else None, expected


def _selection_manifest_id(selection: Mapping[str, Any]) -> str:
    stripped = dict(selection)
    stripped.pop("manifest_id", None)
    return _sha256_bytes(_canonical_json(stripped).encode("utf-8"))


def _contains_outcome_field(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in _OUTCOME_ONLY_KEYS or "outcome" in lowered:
                return True
            if _contains_outcome_field(item):
                return True
    elif isinstance(value, list):
        return any(_contains_outcome_field(item) for item in value)
    return False


def _expected_mix(
    scores: Mapping[str, Any],
) -> tuple[dict[str, float], float, str]:
    single = [_finite(scores.get(action), name=f"score:{action}") for action in SINGLE_ACTIONS]
    positive = [max(value, 0.0) for value in single]
    norm = math.sqrt(sum(value * value for value in positive))
    if norm > 0.0:
        weights = [value / norm for value in positive]
        branch = "positive-relu-l2"
    else:
        best = max(single)
        selected = next(index for index, value in enumerate(single) if value == best)
        weights = [1.0 if index == selected else 0.0 for index in range(len(single))]
        branch = "all-nonpositive-max-onehot"
    mapping = {
        action: weight for action, weight in zip(SINGLE_ACTIONS, weights)
    }
    predicted = sum(weight * score for weight, score in zip(weights, single))
    return mapping, predicted, branch


def _validate_envelope(
    root: Path,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    errors: list[str],
) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
    run_id = manifest.get("run_id")
    model = manifest.get("model")
    model_alias = model.get("model_alias") if isinstance(model, dict) else None
    model_envelope = MODEL_ENVELOPES.get(model_alias)
    _add_error(errors, model_envelope is not None, "model_alias_outside_d0")
    if model_envelope is not None:
        _add_error(
            errors,
            run_id == model_envelope["run_id"],
            "run_id_outside_model_d0_envelope",
        )
        _add_error(
            errors,
            model.get("repository_id") == model_envelope["repository_id"],
            "model_repository_mismatch",
        )
        _add_error(
            errors,
            model.get("revision") == model_envelope["revision"],
            "model_revision_mismatch",
        )
    _add_error(errors, isinstance(run_id, str) and root.name == run_id, "run_directory_mismatch")
    _add_error(errors, summary.get("run_id") == run_id, "summary_run_id_mismatch")
    _add_error(errors, summary.get("model_alias") == model_alias, "summary_model_mismatch")
    if isinstance(model, dict):
        _add_error(errors, model.get("dtype") == "torch.float32", "model_dtype_mismatch")
        _add_error(errors, model.get("device") == "cuda:0", "model_device_mismatch")
        _add_error(errors, model.get("offline") is True, "model_not_offline")

    selected_slice = manifest.get("selected_slice")
    expected_slice = {
        "label": "d0",
        "start": D0_START,
        "count": D0_CASE_COUNT,
        "pilot_case_count_excluded": 3,
    }
    _add_error(errors, selected_slice == expected_slice, "manifest_slice_mismatch")
    _add_error(
        errors,
        summary.get("slice") == {"label": "d0", "start": 3, "count": 5},
        "summary_slice_mismatch",
    )
    _add_error(errors, manifest.get("selected_split") == "calibration", "split_mismatch")
    _add_error(errors, manifest.get("q") == SCORE_MIX_Q, "manifest_q_mismatch")
    _add_error(errors, manifest.get("q_label") == "q_1_256", "q_label_mismatch")
    _add_error(errors, summary.get("q") == SCORE_MIX_Q, "summary_q_mismatch")
    _add_error(errors, manifest.get("probe_ratio") == 0.25, "probe_ratio_mismatch")
    _add_error(errors, tuple(manifest.get("probe_action_set", ())) == PROBE_ACTIONS, "probe_actions_mismatch")
    _add_error(errors, tuple(manifest.get("controller_input_set", ())) == SINGLE_ACTIONS, "controller_inputs_mismatch")
    _add_error(errors, tuple(manifest.get("outcome_action_set", ())) == OUTCOME_ACTIONS, "outcome_actions_mismatch")

    slurm = manifest.get("slurm")
    summary_slurm = summary.get("slurm")
    _add_error(errors, isinstance(slurm, dict) and slurm == summary_slurm, "slurm_identity_mismatch")
    if isinstance(slurm, dict):
        _add_error(errors, slurm.get("under_slurm") is True, "not_slurm_execution")
        _add_error(errors, slurm.get("job_name") == D0_JOB_NAME, "slurm_job_name_mismatch")
        _add_error(errors, slurm.get("node") == "devbox", "slurm_node_mismatch")
        _add_error(
            errors,
            isinstance(slurm.get("job_id"), str) and slurm["job_id"].isdigit(),
            "slurm_job_id_invalid",
        )
        _add_error(errors, slurm.get("slice_label") == "d0", "slurm_slice_mismatch")

    git_state = manifest.get("ode_edit_git")
    _add_error(
        errors,
        isinstance(git_state, dict)
        and isinstance(git_state.get("commit"), str)
        and len(git_state["commit"]) == 40
        and all(
            character in "0123456789abcdef"
            for character in git_state["commit"]
        )
        and git_state.get("tracked_worktree_clean") is True,
        "git_execution_state_invalid",
    )
    case_ids = tuple(manifest.get("selected_case_ids", ()))
    request_ids = tuple(manifest.get("selected_request_ids", ()))
    _add_error(errors, len(case_ids) == D0_CASE_COUNT and len(set(case_ids)) == D0_CASE_COUNT, "selected_case_count_mismatch")
    _add_error(errors, len(request_ids) == D0_CASE_COUNT and all(_is_full_hash(value) for value in request_ids), "selected_request_count_or_hash_mismatch")

    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        errors.append("selection_manifest_missing")
    else:
        case_splits = selection.get("case_ids")
        calibration = (
            tuple(case_splits.get("calibration", ()))
            if isinstance(case_splits, dict)
            else ()
        )
        confirmatory = (
            tuple(case_splits.get("confirmatory", ()))
            if isinstance(case_splits, dict)
            else ()
        )
        untouched = (
            tuple(case_splits.get("untouched", ()))
            if isinstance(case_splits, dict)
            else ()
        )
        _add_error(errors, len(calibration) == 20, "calibration_split_count_mismatch")
        _add_error(errors, len(confirmatory) == 60, "confirmatory_split_count_mismatch")
        _add_error(errors, len(untouched) == 20, "untouched_split_count_mismatch")
        _add_error(errors, case_ids == calibration[3:8], "selected_slice_ids_mismatch")
        _add_error(errors, not set(case_ids).intersection(calibration[:3]), "pilot_selection_leakage")
        expected_order_hash = _sha256_bytes(
            _canonical_json(list(calibration + confirmatory + untouched)).encode("utf-8")
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
        expected_selection_id = _selection_manifest_id(selection)
        _add_error(errors, selection.get("order_hash") == expected_order_hash, "selection_order_hash_mismatch")
        _add_error(errors, selection.get("split_hash") == expected_split_hash, "selection_split_hash_mismatch")
        _add_error(errors, selection.get("manifest_id") == expected_selection_id, "selection_manifest_id_mismatch")
        _add_error(errors, summary.get("selection_manifest_id") == expected_selection_id, "summary_selection_manifest_id_mismatch")
    return model_alias, case_ids, request_ids


def _index_rows(
    rows: Sequence[StreamRow],
    *,
    fields: Sequence[str],
    errors: list[str],
    label: str,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        try:
            key = tuple(row.payload[field] for field in fields)
        except KeyError:
            errors.append(f"{label}_missing_index_field")
            continue
        if key in result:
            errors.append(f"{label}_duplicate_index")
        else:
            result[key] = row.payload
    return result


def _validate_rows(
    *,
    case_ids: Sequence[str],
    request_ids: Sequence[str],
    streams: Mapping[str, Sequence[StreamRow]],
    receipts: Mapping[str, Mapping[str, Any]],
    receipt_hashes: Mapping[str, str],
    errors: list[str],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    expected_request = dict(zip(case_ids, request_ids))
    for name, rows in streams.items():
        expected_event = _EXPECTED_STREAM_EVENTS[name]
        _add_error(
            errors,
            all(row.event == expected_event for row in rows),
            f"{name}_event_mismatch",
        )

    feature_index_raw = _index_rows(
        streams["features.jsonl"],
        fields=("case_id",),
        errors=errors,
        label="feature",
    )
    action_index_raw = _index_rows(
        streams["actions.jsonl"],
        fields=("case_id",),
        errors=errors,
        label="action",
    )
    outcome_index_raw = _index_rows(
        streams["outcomes.jsonl"],
        fields=("case_id", "action_id"),
        errors=errors,
        label="outcome",
    )
    event_index_raw = _index_rows(
        streams["events.jsonl"],
        fields=("case_id",),
        errors=errors,
        label="event",
    )
    features = {str(key[0]): value for key, value in feature_index_raw.items()}
    actions = {str(key[0]): value for key, value in action_index_raw.items()}
    outcomes = {
        (str(key[0]), str(key[1])): value
        for key, value in outcome_index_raw.items()
    }
    events = {str(key[0]): value for key, value in event_index_raw.items()}

    _add_error(errors, set(features) == set(case_ids), "feature_case_panel_mismatch")
    _add_error(errors, set(actions) == set(case_ids), "action_case_panel_mismatch")
    _add_error(errors, set(events) == set(case_ids), "event_case_panel_mismatch")
    _add_error(
        errors,
        set(outcomes)
        == {(case_id, action_id) for case_id in case_ids for action_id in OUTCOME_ACTIONS},
        "outcome_case_action_panel_mismatch",
    )

    for case_id in case_ids:
        feature = features.get(case_id)
        action = actions.get(case_id)
        event = events.get(case_id)
        if feature is None or action is None or event is None:
            continue
        _add_error(errors, set(feature) == _FEATURE_FIELDS, f"feature_schema_mismatch:{case_id}")
        _add_error(errors, set(action) == _ACTION_FIELDS, f"action_schema_mismatch:{case_id}")
        _add_error(errors, set(event) == _EVENT_FIELDS, f"event_schema_mismatch:{case_id}")
        _add_error(errors, not _contains_outcome_field(feature), f"feature_outcome_leakage:{case_id}")
        _add_error(errors, not _contains_outcome_field(action), f"action_outcome_leakage:{case_id}")
        expected_request_id = expected_request.get(case_id)
        for label, payload in (("feature", feature), ("action", action), ("event", event)):
            _add_error(errors, payload.get("request_id") == expected_request_id, f"{label}_request_mismatch:{case_id}")
        _add_error(errors, feature.get("q") == SCORE_MIX_Q and feature.get("q_label") == "q_1_256", f"feature_q_mismatch:{case_id}")
        _add_error(errors, action.get("q") == SCORE_MIX_Q and action.get("q_label") == "q_1_256", f"action_q_mismatch:{case_id}")
        observed_feature_hash, expected_feature_hash = _payload_hash(feature, "feature_hash")
        observed_action_hash, expected_action_hash = _payload_hash(action, "commitment_hash")
        _add_error(errors, observed_feature_hash == expected_feature_hash, f"feature_hash_mismatch:{case_id}")
        _add_error(errors, observed_action_hash == expected_action_hash, f"commitment_hash_mismatch:{case_id}")
        _add_error(errors, action.get("feature_hash") == observed_feature_hash, f"action_feature_chain_mismatch:{case_id}")
        _add_error(errors, action.get("action_id") == "score_mix", f"committed_action_mismatch:{case_id}")
        _add_error(errors, action.get("tie_policy") == "exact-tie-lower-layer", f"tie_policy_mismatch:{case_id}")
        try:
            actual_unit_c_energy = _finite(
                action.get("actual_unit_c_energy"),
                name="actual_unit_c_energy",
            )
            _add_error(
                errors,
                math.isclose(
                    actual_unit_c_energy,
                    1.0,
                    rel_tol=2e-5,
                    abs_tol=2e-5,
                ),
                f"action_unit_c_energy_mismatch:{case_id}",
            )
        except ScoreMixAnalysisError:
            errors.append(f"action_unit_c_energy_nonfinite:{case_id}")

        scores = feature.get("action_scores")
        if not isinstance(scores, dict) or set(scores) != set(PROBE_ACTIONS):
            errors.append(f"feature_score_set_mismatch:{case_id}")
        else:
            try:
                expected_weights, expected_score, expected_branch = _expected_mix(scores)
                observed_weights = action.get("layer_weights")
                weights_valid = (
                    isinstance(observed_weights, dict)
                    and set(observed_weights) == set(SINGLE_ACTIONS)
                    and all(
                        math.isclose(
                            _finite(observed_weights[key], name=f"weight:{key}"),
                            expected_weights[key],
                            rel_tol=1e-10,
                            abs_tol=1e-12,
                        )
                        for key in SINGLE_ACTIONS
                    )
                )
                _add_error(errors, weights_valid, f"controller_weights_mismatch:{case_id}")
                _add_error(
                    errors,
                    math.isclose(
                        _finite(action.get("predicted_score"), name="predicted_score"),
                        expected_score,
                        rel_tol=1e-10,
                        abs_tol=1e-12,
                    ),
                    f"controller_score_mismatch:{case_id}",
                )
                _add_error(errors, action.get("controller_branch") == expected_branch, f"controller_branch_mismatch:{case_id}")
            except ScoreMixAnalysisError:
                errors.append(f"nonfinite_controller_input:{case_id}")

        reference = event.get("action_receipt")
        if not isinstance(reference, dict):
            errors.append(f"event_receipt_missing:{case_id}")
        else:
            name = reference.get("name")
            receipt = receipts.get(name) if isinstance(name, str) else None
            if receipt is None:
                errors.append(f"event_receipt_unknown:{case_id}")
            else:
                expected_receipt_fields = {
                    "schema_version",
                    "case_id",
                    "request_id",
                    "q",
                    "feature_hash",
                    "commitment_hash",
                    "action_id",
                    "durability",
                    "outcomes_observed_before_commitment",
                }
                _add_error(errors, set(receipt) == expected_receipt_fields, f"receipt_schema_mismatch:{case_id}")
                _add_error(errors, receipt.get("schema_version") == RECEIPT_SCHEMA, f"receipt_version_mismatch:{case_id}")
                _add_error(errors, receipt.get("case_id") == case_id and receipt.get("request_id") == expected_request_id, f"receipt_identity_mismatch:{case_id}")
                _add_error(errors, receipt.get("q") == SCORE_MIX_Q and receipt.get("action_id") == "score_mix", f"receipt_action_mismatch:{case_id}")
                _add_error(errors, receipt.get("feature_hash") == observed_feature_hash and receipt.get("commitment_hash") == observed_action_hash, f"receipt_commitment_chain_mismatch:{case_id}")
                _add_error(errors, receipt.get("outcomes_observed_before_commitment") is False, f"receipt_outcome_leakage:{case_id}")
                _add_error(
                    errors,
                    receipt.get("durability")
                    == "feature-and-dynamic-action-write+flush+fsync-before-exclusive-receipt",
                    f"receipt_durability_mismatch:{case_id}",
                )
                identity = {
                    "case_id": case_id,
                    "request_id": expected_request_id,
                    "q": SCORE_MIX_Q,
                    "feature_hash": observed_feature_hash,
                    "commitment_hash": observed_action_hash,
                }
                expected_name = _sha256_bytes(_canonical_json(identity).encode("utf-8")) + ".json"
                _add_error(errors, name == expected_name, f"receipt_filename_mismatch:{case_id}")
                _add_error(errors, reference.get("sha256") == receipt_hashes.get(str(name)), f"event_receipt_hash_mismatch:{case_id}")
        _add_error(errors, event.get("q") == SCORE_MIX_Q, f"event_q_mismatch:{case_id}")
        _add_error(errors, event.get("probe_direction_count") == 6, f"probe_count_mismatch:{case_id}")
        _add_error(errors, event.get("feature_count") == 1 and event.get("commitment_count") == 1 and event.get("outcome_count") == 5, f"event_count_mismatch:{case_id}")
        _add_error(errors, event.get("rollback_exact") is True and event.get("pass") is True, f"event_failure_or_rollback:{case_id}")

        for action_id in OUTCOME_ACTIONS:
            outcome = outcomes.get((case_id, action_id))
            if outcome is None:
                continue
            expected_fields = _OUTCOME_FIELDS | ({"logits_hash_equal"} if action_id == "no_op_replay" else set())
            _add_error(errors, set(outcome) == expected_fields, f"outcome_schema_mismatch:{case_id}:{action_id}")
            _add_error(errors, outcome.get("request_id") == expected_request_id, f"outcome_request_mismatch:{case_id}:{action_id}")
            _add_error(errors, outcome.get("q") == SCORE_MIX_Q and outcome.get("q_label") == "q_1_256", f"outcome_q_mismatch:{case_id}:{action_id}")
            _add_error(errors, outcome.get("feature_hash") == observed_feature_hash and outcome.get("commitment_hash") == observed_action_hash, f"outcome_commitment_chain_mismatch:{case_id}:{action_id}")
            _add_error(errors, outcome.get("status") == "completed" and outcome.get("rollback_exact") is True, f"outcome_failure_or_rollback:{case_id}:{action_id}")
            _add_error(
                errors,
                outcome.get("budget_validation")
                == _BUDGET_VALIDATION_BY_ACTION[action_id],
                f"budget_validation_mismatch:{case_id}:{action_id}",
            )
            try:
                for scalar_name in ("c_distance", "c_energy", "progress", "nll_reduction", "exact_margin_min"):
                    _finite(outcome.get(scalar_name), name=scalar_name)
                for vector_name in ("context_progress", "context_nll_reduction"):
                    vector = outcome.get(vector_name)
                    if not isinstance(vector, list) or not vector:
                        raise ScoreMixAnalysisError("context denominator missing")
                    for value in vector:
                        _finite(value, name=vector_name)
            except ScoreMixAnalysisError:
                errors.append(f"outcome_nonfinite_or_denominator_missing:{case_id}:{action_id}")
            if action_id == "score_mix":
                _add_error(errors, outcome.get("selected_by_controller") is True, f"mix_selection_flag_mismatch:{case_id}")
            else:
                _add_error(errors, outcome.get("selected_by_controller") is False, f"sentinel_selection_flag_mismatch:{case_id}:{action_id}")
            if action_id == "no_op_replay":
                _add_error(errors, outcome.get("logits_hash_equal") is True, f"replay_logits_mismatch:{case_id}")
                _add_error(errors, outcome.get("c_distance") == 0.0 and outcome.get("c_energy") == 0.0, f"replay_budget_mismatch:{case_id}")

        equal_c = [
            outcomes.get((case_id, action_id))
            for action_id in ("score_mix", "uniform", "ordered_global_alpha")
        ]
        if all(item is not None for item in equal_c):
            try:
                locked_distance = _finite(
                    feature.get("operational_c_distance"),
                    name="operational_c_distance",
                )
                expected_energy = locked_distance * locked_distance
                for arm, item in zip(
                    ("score_mix", "uniform", "ordered_global_alpha"),
                    equal_c,
                ):
                    if item is None:
                        continue
                    observed_distance = _finite(
                        item.get("c_distance"),
                        name="c_distance",
                    )
                    observed_energy = _finite(
                        item.get("c_energy"),
                        name="c_energy",
                    )
                    _add_error(
                        errors,
                        math.isclose(
                            observed_distance,
                            locked_distance,
                            rel_tol=3e-5,
                            abs_tol=3e-5,
                        ),
                        f"equal_c_distance_mismatch:{case_id}:{arm}",
                    )
                    _add_error(
                        errors,
                        math.isclose(
                            observed_energy,
                            expected_energy,
                            rel_tol=3e-5,
                            abs_tol=3e-5,
                        ),
                        f"equal_c_energy_mismatch:{case_id}:{arm}",
                    )
            except ScoreMixAnalysisError:
                errors.append(f"equal_c_nonfinite:{case_id}")

    _add_error(errors, len(receipts) == D0_CASE_COUNT, "receipt_count_mismatch")
    return features, actions, outcomes, events


def _validate_summary_and_hashes(
    *,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    streams: Mapping[str, Sequence[StreamRow]],
    receipts: Mapping[str, Mapping[str, Any]],
    observed_hashes: Mapping[str, str],
    errors: list[str],
) -> None:
    expected_counts = {
        "features": D0_CASE_COUNT,
        "commitments": D0_CASE_COUNT,
        "outcomes": D0_CASE_COUNT * len(OUTCOME_ACTIONS),
        "receipts": D0_CASE_COUNT,
    }
    _add_error(errors, manifest.get("expected_counts") == expected_counts, "manifest_expected_counts_mismatch")
    exact_summary = {
        "planned_case_count": D0_CASE_COUNT,
        "attempted_case_count": D0_CASE_COUNT,
        "not_run_due_to_abort_count": 0,
        "pass_count": D0_CASE_COUNT,
        "failure_count": 0,
        "feature_count": D0_CASE_COUNT,
        "commitment_count": D0_CASE_COUNT,
        "outcome_count": D0_CASE_COUNT * len(OUTCOME_ACTIONS),
        "action_receipt_count": D0_CASE_COUNT,
    }
    for key, expected in exact_summary.items():
        _add_error(errors, summary.get(key) == expected, f"summary_count_mismatch:{key}")
    _add_error(errors, summary.get("run_status") == "completed", "summary_run_not_completed")
    _add_error(errors, summary.get("abort_failure_type") is None, "summary_abort_present")
    _add_error(errors, summary.get("all_pass") is True, "summary_all_pass_false")
    _add_error(errors, summary.get("expected_counts_exact") is True, "summary_exact_counts_false")
    _add_error(errors, summary.get("all_rollbacks_exact") is True, "summary_rollback_false")
    _add_error(errors, summary.get("projector_files_loaded") == [], "projector_loaded")
    _add_error(errors, len(streams["features.jsonl"]) == 5, "observed_feature_count_mismatch")
    _add_error(errors, len(streams["actions.jsonl"]) == 5, "observed_action_count_mismatch")
    _add_error(errors, len(streams["outcomes.jsonl"]) == 25, "observed_outcome_count_mismatch")
    _add_error(errors, len(streams["events.jsonl"]) == 5, "observed_event_count_mismatch")

    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("summary_artifacts_missing")
        return
    for file_name in _HASHED_FILES:
        key = file_name.replace(".jsonl", "_sha256").replace(".json", "_sha256")
        _add_error(errors, artifacts.get(key) == observed_hashes.get(file_name), f"summary_artifact_hash_mismatch:{file_name}")
    expected_receipt_hashes = artifacts.get("action_receipts")
    actual_receipt_hashes = {
        name.removeprefix("action_receipts/"): digest
        for name, digest in observed_hashes.items()
        if name.startswith("action_receipts/")
    }
    _add_error(errors, expected_receipt_hashes == actual_receipt_hashes, "summary_receipt_hash_map_mismatch")


def _safe_progress(
    outcomes: Mapping[tuple[str, str], Mapping[str, Any]],
    case_id: str,
    action_id: str,
) -> float | None:
    payload = outcomes.get((case_id, action_id))
    if payload is None or payload.get("status") != "completed":
        return None
    try:
        return _finite(payload.get("progress"), name="progress")
    except ScoreMixAnalysisError:
        return None


def _effect_summary(values: Sequence[float]) -> dict[str, Any]:
    return {
        "planned_case_count": D0_CASE_COUNT,
        "finite_case_count": len(values),
        "failure_or_missing_case_count": D0_CASE_COUNT - len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "observed_case_range": [min(values), max(values)] if values else None,
    }


def analyze_score_mix_run(run_directory: str | Path) -> dict[str, Any]:
    """Validate and descriptively analyze one sanitized D0 model directory."""

    root = Path(run_directory).expanduser().resolve()
    if not root.is_dir():
        raise ScoreMixAnalysisError("run directory does not exist")
    manifest = _read_json_object(root / "manifest.json")
    summary = _read_json_object(root / "summary.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ScoreMixAnalysisError("manifest schema mismatch")
    if summary.get("schema_version") != SUMMARY_SCHEMA:
        raise ScoreMixAnalysisError("summary schema mismatch")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ScoreMixAnalysisError("run_id invalid")
    streams = {
        name: _read_jsonl(root / name, run_id=run_id) for name in _STREAM_FILES
    }
    receipt_root = root / "action_receipts"
    if receipt_root.is_symlink() or not receipt_root.is_dir():
        raise ScoreMixAnalysisError("action_receipts must be a regular directory")
    receipt_paths = sorted(receipt_root.glob("*.json"))
    receipts = {path.name: _read_json_object(path) for path in receipt_paths}
    observed_hashes = {name: _sha256_file(root / name) for name in _HASHED_FILES}
    observed_hashes.update(
        {
            f"action_receipts/{path.name}": _sha256_file(path)
            for path in receipt_paths
        }
    )

    errors: list[str] = []
    model_alias, case_ids, request_ids = _validate_envelope(
        root, manifest, summary, errors
    )
    features, actions, outcomes, _events = _validate_rows(
        case_ids=case_ids,
        request_ids=request_ids,
        streams=streams,
        receipts=receipts,
        receipt_hashes={
            path.name: observed_hashes[f"action_receipts/{path.name}"]
            for path in receipt_paths
        },
        errors=errors,
    )
    _validate_summary_and_hashes(
        manifest=manifest,
        summary=summary,
        streams=streams,
        receipts=receipts,
        observed_hashes=observed_hashes,
        errors=errors,
    )

    replay_values = [
        _safe_progress(outcomes, case_id, "no_op_replay") for case_id in case_ids
    ]
    finite_replay = [abs(value) for value in replay_values if value is not None]
    replay_envelope = (
        max(NUMERIC_COMPARISON_TOLERANCE, max(finite_replay))
        if finite_replay
        else None
    )
    case_rows: list[dict[str, Any]] = []
    gains: list[float] = []
    within_flags: list[bool] = []
    for case_id in case_ids:
        mix = _safe_progress(outcomes, case_id, "score_mix")
        uniform = _safe_progress(outcomes, case_id, "uniform")
        replay = _safe_progress(outcomes, case_id, "no_op_replay")
        feature = features.get(case_id)
        action = actions.get(case_id)
        distance: float | None = None
        predicted_mix: float | None = None
        predicted_uniform: float | None = None
        observed_uniform_probe: float | None = None
        if feature is not None and action is not None:
            try:
                distance = _finite(
                    feature.get("operational_c_distance"),
                    name="operational_c_distance",
                )
                predicted_mix = _finite(
                    action.get("predicted_score"),
                    name="predicted_score",
                )
                scores = feature.get("action_scores")
                if isinstance(scores, dict):
                    singles = [
                        _finite(scores.get(key), name=f"score:{key}")
                        for key in SINGLE_ACTIONS
                    ]
                    predicted_uniform = sum(singles) / math.sqrt(len(singles))
                    observed_uniform_probe = _finite(
                        scores.get("uniform"),
                        name="score:uniform",
                    )
            except ScoreMixAnalysisError:
                distance = None
                predicted_mix = None
                predicted_uniform = None
                observed_uniform_probe = None
        gain = mix - uniform if mix is not None and uniform is not None else None
        within = (
            gain <= replay_envelope
            if gain is not None and replay_envelope is not None
            else None
        )
        if gain is not None:
            gains.append(gain)
        if within is not None:
            within_flags.append(within)
        predicted_gap = (
            predicted_mix - predicted_uniform
            if predicted_mix is not None and predicted_uniform is not None
            else None
        )
        predicted_finite_gain = (
            predicted_gap * distance
            if predicted_gap is not None and distance is not None
            else None
        )
        realized_slope_gap = (
            gain / distance
            if gain is not None and distance is not None and distance > 0
            else None
        )
        case_rows.append(
            {
                "case_id": case_id,
                "p_score_mix": mix,
                "p_uniform": uniform,
                "g_mix": gain,
                "absolute_replay_progress": abs(replay) if replay is not None else None,
                "model_replay_envelope": replay_envelope,
                "numeric_comparison_tolerance": NUMERIC_COMPARISON_TOLERANCE,
                "g_mix_leq_envelope_with_numeric_tolerance": within,
                "predicted_score_mix": predicted_mix,
                "predicted_score_uniform": predicted_uniform,
                "observed_uniform_probe_score": observed_uniform_probe,
                "predicted_slope_gap": predicted_gap,
                "operational_c_distance": distance,
                "predicted_local_linear_gain": predicted_finite_gain,
                "realized_slope_gap": realized_slope_gap,
                "realized_minus_predicted_gain": (
                    gain - predicted_finite_gain
                    if gain is not None and predicted_finite_gain is not None
                    else None
                ),
            }
        )

    integrity_errors = sorted(set(errors))
    panel_complete = (
        len(case_rows) == D0_CASE_COUNT
        and len(gains) == D0_CASE_COUNT
        and len(finite_replay) == D0_CASE_COUNT
        and len(within_flags) == D0_CASE_COUNT
    )
    valid = not integrity_errors
    analyzable = valid and panel_complete
    all_five_nonpositive = all(within_flags) if analyzable else None
    analysis_status = (
        "d0_descriptive_complete"
        if analyzable
        else "technical_block_invalid_artifacts"
        if not valid
        else "technical_block_incomplete_panel"
    )
    report = {
        "schema_version": ANALYSIS_SCHEMA,
        "claim_status": "d0_calibration_descriptive_only",
        "analysis_status": analysis_status,
        "run_id": run_id,
        "model_alias": model_alias,
        "metric_definition": {
            "p_i_v": (
                "runner outcome.progress = after.utility - before.utility; "
                "utility is the temperature-1 mean-context-token smooth target margin"
            ),
            "g_mix": "P_i(V_mix,i) - P_i(V_uniform,i) at matched q and C budget",
            "predicted_slope_gap": (
                "action.predicted_score - sum(five single-layer slopes)/sqrt(5)"
            ),
            "realized_slope_gap": "G_mix / operational_c_distance",
        },
        "analysis_policy": {
            "input_scope": (
                "sanitized manifest, summary, four JSONL streams, and action receipts"
            ),
            "raw_tensor_artifacts_opened": False,
            "activation_artifact_path_or_hash_emitted": False,
            "outcome_or_selection_refit": False,
            "inference_or_confidence_interval": False,
            "d0_role": "calibration and continuous-routing early-kill input only",
            "positive_event_meaning": "permits paired D1 only; not GO or positive claim",
            "analysis_code_sha256": _sha256_file(Path(__file__)),
        },
        "artifact_validation": {
            "valid": valid,
            "error_codes": integrity_errors,
            "panel_complete": panel_complete,
            "expected_counts": {
                "cases": 5,
                "features": 5,
                "commitments": 5,
                "outcomes": 25,
                "events": 5,
                "receipts": 5,
            },
            "observed_counts": {
                "features": len(streams["features.jsonl"]),
                "commitments": len(streams["actions.jsonl"]),
                "outcomes": len(streams["outcomes.jsonl"]),
                "events": len(streams["events.jsonl"]),
                "receipts": len(receipts),
            },
            "commitment_chain": {
                "valid": valid,
                "contract": (
                    "feature hash -> outcome-free action hash -> durable exclusive "
                    "receipt hash -> every outcome/event reference"
                ),
                "cross_file_wall_clock_order_proven": False,
                "runner_contract_evidence_only": True,
            },
            "selection_and_leakage_firewall": {
                "valid": valid,
                "slice": "calibration[3:8]",
                "historical_pilot_excluded": True,
                "action_recomputed_from_feature_only": True,
                "forbidden_raw_fields_checked_recursively": True,
            },
        },
        "failure_denominator": {
            "planned_case_count": D0_CASE_COUNT,
            "finite_gain_count": len(gains),
            "finite_replay_count": len(finite_replay),
            "missing_or_failed_cases_are_not_dropped": True,
        },
        "replay_envelope": {
            "policy": (
                "max(1e-12, max(abs(no_op_replay progress))) across all five "
                "planned D0 cases"
            ),
            "value": replay_envelope,
            "numeric_floor": NUMERIC_COMPARISON_TOLERANCE,
            "comparison": "G_mix <= envelope; continue iff G_mix > envelope",
            "tolerance_meaning": (
                "the 1e-12 absolute utility-unit floor only stabilizes exact-zero "
                "floating-point/JSON arithmetic; it is not added a second time, is "
                "not an empirical effect allowance, and is not added to gains"
            ),
        },
        "case_diagnostics": case_rows,
        "descriptive_effect": {
            "g_mix": _effect_summary(gains),
            "label": (
                "observed D0 calibration values only; not expected method improvement "
                "and not a population estimate"
            ),
        },
        "model_early_kill_input": {
            "all_five_g_mix_leq_model_replay_envelope": all_five_nonpositive,
            "events_above_envelope": (
                sum(not value for value in within_flags) if analyzable else None
            ),
            "scope": "single-model input only",
            "pair_level_early_kill_computed": False,
            "pair_rule": (
                "early kill only if independently validated Llama and Qwen reports "
                "both have all five events within their own model envelope"
            ),
            "if_any_event_above_envelope": (
                "paired D1 calibration may proceed; no GO, method-gain claim, "
                "confirmatory success, or MV-2 entry"
            ),
        },
    }
    _assert_json_safe(report, location="analysis_output")
    json.dumps(report, allow_nan=False)
    return report


def build_small_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the small broadcast-safe subset; no per-case feature vectors."""

    early = report["model_early_kill_input"]
    effect = report["descriptive_effect"]["g_mix"]
    return {
        "schema_version": SMALL_SUMMARY_SCHEMA,
        "claim_status": report["claim_status"],
        "analysis_status": report["analysis_status"],
        "run_id": report["run_id"],
        "model_alias": report["model_alias"],
        "artifact_valid": report["artifact_validation"]["valid"],
        "panel_complete": report["artifact_validation"]["panel_complete"],
        "error_codes": report["artifact_validation"]["error_codes"],
        "replay_envelope": report["replay_envelope"]["value"],
        "numeric_envelope_floor": NUMERIC_COMPARISON_TOLERANCE,
        "finite_gain_count": effect["finite_case_count"],
        "mean_g_mix": effect["mean"],
        "observed_g_mix_range": effect["observed_case_range"],
        "all_five_g_mix_leq_model_replay_envelope": early[
            "all_five_g_mix_leq_model_replay_envelope"
        ],
        "events_above_envelope": early["events_above_envelope"],
        "pair_level_early_kill_computed": False,
        "decision_boundary": (
            "single-model D0 calibration only; positive permits paired D1, never GO"
        ),
    }


def render_korean_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise Korean report from the validated analysis object."""

    validation = report["artifact_validation"]
    replay = report["replay_envelope"]
    early = report["model_early_kill_input"]
    lines = [
        f"# MV-1 score-mix D0 독립 분석: {report['model_alias']}",
        "",
        f"- run ID: `{report['run_id']}`",
        f"- 분석 상태: `{report['analysis_status']}`",
        f"- artifact/panel: `valid={validation['valid']}`, "
        f"`complete={validation['panel_complete']}`",
        "- 해석 경계: D0 calibration 및 continuous-routing early-kill 입력만이다. "
        "양의 event는 paired D1만 허용하며 GO, 방법 개선 claim, confirmatory "
        "success 또는 MV-2 진입을 뜻하지 않는다.",
        "",
        "## Metric과 판정",
        "",
        "- `P_i(V)`는 runner가 기록한 `progress = after.utility - before.utility`다.",
        "- `G_mix = P_i(V_mix) - P_i(V_uniform)`이며 두 arm은 같은 `q=1/256` "
        "및 remeasured equal-`C` budget을 사용한다.",
        f"- model replay envelope `e_m = {replay['value']!r}`는 "
        "`max(1e-12, 5개 no_op_replay의 max(abs(progress)))`다.",
        "- event 판정은 `G_mix <= e_m`, continue는 `G_mix > e_m`다. "
        "`1e-12`는 exact-zero JSON/부동소수점 산술용 절대 floor이며 "
        "효과 허용폭으로 한 번 더 더하지 않는다.",
        "",
        "## Case별 결과",
        "",
        "| case | P(score_mix) | P(uniform) | G_mix | e_m 내 | predicted slope gap | realized slope gap |",
        "| --- | ---: | ---: | ---: | :---: | ---: | ---: |",
    ]

    def number(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.8g}"

    for row in report["case_diagnostics"]:
        within = row["g_mix_leq_envelope_with_numeric_tolerance"]
        within_label = "NA" if within is None else ("yes" if within else "no")
        lines.append(
            f"| `{row['case_id']}` | {number(row['p_score_mix'])} | "
            f"{number(row['p_uniform'])} | {number(row['g_mix'])} | "
            f"{within_label} | {number(row['predicted_slope_gap'])} | "
            f"{number(row['realized_slope_gap'])} |"
        )
    lines.extend(
        [
            "",
            "## 단일 모델 결론",
            "",
            f"- all-five nonpositive-within-envelope: "
            f"`{early['all_five_g_mix_leq_model_replay_envelope']}`",
            f"- envelope 초과 event 수: `{early['events_above_envelope']}`",
            "- pair-level early kill은 이 문서에서 계산하지 않는다. 독립 검증된 "
            "Llama/Qwen 두 analyzer 출력이 모두 all-five 조건을 만족할 때만 GH가 "
            "결합 판정한다.",
        ]
    )
    if validation["error_codes"]:
        lines.extend(
            [
                "",
                "## Technical BLOCK",
                "",
                *[f"- `{code}`" for code in validation["error_codes"]],
            ]
        )
    return "\n".join(lines) + "\n"


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze one sanitized MV-1 score-mix D0 model run."
    )
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Default: REPORT_OUTPUT with '.summary.json' appended.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = analyze_score_mix_run(args.run_directory)
        summary = build_small_summary(report)
        summary_output = args.summary_output
        if summary_output is None:
            summary_output = Path(str(args.report_output) + ".summary.json")
        if args.report_output.resolve() == summary_output.resolve():
            raise ScoreMixAnalysisError("report and summary outputs must differ")
        _write_exclusive(
            args.report_output,
            render_korean_markdown(report).encode("utf-8"),
        )
        _write_exclusive(
            summary_output,
            (
                json.dumps(
                    summary,
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
                    "report_written": True,
                    "summary_written": True,
                    "artifact_valid": report["artifact_validation"]["valid"],
                },
                sort_keys=True,
            )
        )
        return 0 if report["analysis_status"] == "d0_descriptive_complete" else 2
    except (OSError, ScoreMixAnalysisError) as exc:
        print(
            json.dumps(
                {
                    "status": "analysis_aborted",
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
