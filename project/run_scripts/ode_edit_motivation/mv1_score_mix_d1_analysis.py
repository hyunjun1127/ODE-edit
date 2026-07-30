"""Independent D1 analysis and slope-only static-policy freeze for MV-1.

The D1 report is descriptive calibration output.  The frozen static policy is
fit from the five single-layer slope fields in independently valid D0 and D1
feature streams only.  Outcome values are not accepted by the fitting API.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .mv1_score_mix_analysis import (
    MANIFEST_SCHEMA,
    MODEL_ENVELOPES,
    NUMERIC_COMPARISON_TOLERANCE,
    OUTCOME_ACTIONS,
    PROBE_ACTIONS,
    SCORE_MIX_Q,
    SINGLE_ACTIONS,
    STREAM_SCHEMA,
    SUMMARY_SCHEMA,
    ScoreMixAnalysisError,
    StreamRow,
    _FEATURE_FIELDS,
    _HASHED_FILES,
    _STREAM_FILES,
    _add_error,
    _assert_json_safe,
    _canonical_json,
    _contains_outcome_field,
    _expected_mix,
    _finite,
    _index_rows,
    _is_full_hash,
    _payload_hash,
    _read_json_object,
    _read_jsonl,
    _safe_progress,
    _selection_manifest_id,
    _sha256_bytes,
    _sha256_file,
    _validate_rows,
    _write_exclusive,
    analyze_score_mix_run,
)


ANALYSIS_SCHEMA = "ode-edit-mv1-score-mix-d1-analysis/v1"
SMALL_SUMMARY_SCHEMA = "ode-edit-mv1-score-mix-d1-analysis-summary/v1"
STATIC_POLICY_SCHEMA = "ode-edit-mv1-score-mix-static-policy/v1"
D1_START = 8
D1_CASE_COUNT = 12
D1_JOB_NAME = "odeedit_mv1mix_d1_pair_v1"
D0_CASE_COUNT = 5
TOTAL_STATIC_CASE_COUNT = 17

D1_RUN_IDS = {
    "llama3-8b-inst": "mv1mix_llama_d1_v1",
    "qwen2.5-7b-inst": "mv1mix_qwen_d1_v1",
}
D0_RUN_IDS = {
    model_alias: envelope["run_id"]
    for model_alias, envelope in MODEL_ENVELOPES.items()
}


def _validate_d1_envelope(
    root: Path,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    errors: list[str],
) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
    run_id = manifest.get("run_id")
    model = manifest.get("model")
    model_alias = model.get("model_alias") if isinstance(model, dict) else None
    fixed_model = MODEL_ENVELOPES.get(model_alias)
    _add_error(errors, fixed_model is not None, "model_alias_outside_d1")
    if fixed_model is not None:
        _add_error(
            errors,
            run_id == D1_RUN_IDS[model_alias],
            "run_id_outside_model_d1_envelope",
        )
        _add_error(
            errors,
            model.get("repository_id") == fixed_model["repository_id"],
            "model_repository_mismatch",
        )
        _add_error(
            errors,
            model.get("revision") == fixed_model["revision"],
            "model_revision_mismatch",
        )
    _add_error(
        errors,
        isinstance(run_id, str) and root.name == run_id,
        "run_directory_mismatch",
    )
    _add_error(errors, summary.get("run_id") == run_id, "summary_run_id_mismatch")
    _add_error(
        errors,
        summary.get("model_alias") == model_alias,
        "summary_model_mismatch",
    )
    if isinstance(model, dict):
        _add_error(
            errors, model.get("dtype") == "torch.float32", "model_dtype_mismatch"
        )
        _add_error(errors, model.get("device") == "cuda:0", "model_device_mismatch")
        _add_error(errors, model.get("offline") is True, "model_not_offline")

    _add_error(
        errors,
        manifest.get("selected_slice")
        == {
            "label": "d1",
            "start": D1_START,
            "count": D1_CASE_COUNT,
            "pilot_case_count_excluded": 3,
        },
        "manifest_slice_mismatch",
    )
    _add_error(
        errors,
        summary.get("slice")
        == {"label": "d1", "start": D1_START, "count": D1_CASE_COUNT},
        "summary_slice_mismatch",
    )
    _add_error(
        errors,
        manifest.get("selected_split") == "calibration",
        "split_mismatch",
    )
    _add_error(errors, manifest.get("q") == SCORE_MIX_Q, "manifest_q_mismatch")
    _add_error(errors, manifest.get("q_label") == "q_1_256", "q_label_mismatch")
    _add_error(errors, summary.get("q") == SCORE_MIX_Q, "summary_q_mismatch")
    _add_error(
        errors, manifest.get("probe_ratio") == 0.25, "probe_ratio_mismatch"
    )
    _add_error(
        errors,
        tuple(manifest.get("probe_action_set", ())) == PROBE_ACTIONS,
        "probe_actions_mismatch",
    )
    _add_error(
        errors,
        tuple(manifest.get("controller_input_set", ())) == SINGLE_ACTIONS,
        "controller_inputs_mismatch",
    )
    _add_error(
        errors,
        tuple(manifest.get("outcome_action_set", ())) == OUTCOME_ACTIONS,
        "outcome_actions_mismatch",
    )

    slurm = manifest.get("slurm")
    _add_error(
        errors,
        isinstance(slurm, dict) and slurm == summary.get("slurm"),
        "slurm_identity_mismatch",
    )
    if isinstance(slurm, dict):
        _add_error(errors, slurm.get("under_slurm") is True, "not_slurm_execution")
        _add_error(
            errors,
            slurm.get("job_name") == D1_JOB_NAME,
            "slurm_job_name_mismatch",
        )
        _add_error(errors, slurm.get("node") == "devbox", "slurm_node_mismatch")
        _add_error(
            errors,
            isinstance(slurm.get("job_id"), str) and slurm["job_id"].isdigit(),
            "slurm_job_id_invalid",
        )
        _add_error(
            errors, slurm.get("slice_label") == "d1", "slurm_slice_mismatch"
        )

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
    _add_error(
        errors,
        len(case_ids) == D1_CASE_COUNT and len(set(case_ids)) == D1_CASE_COUNT,
        "selected_case_count_mismatch",
    )
    _add_error(
        errors,
        len(request_ids) == D1_CASE_COUNT
        and all(_is_full_hash(value) for value in request_ids),
        "selected_request_count_or_hash_mismatch",
    )

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
        _add_error(
            errors, len(calibration) == 20, "calibration_split_count_mismatch"
        )
        _add_error(
            errors, len(confirmatory) == 60, "confirmatory_split_count_mismatch"
        )
        _add_error(errors, len(untouched) == 20, "untouched_split_count_mismatch")
        _add_error(
            errors,
            case_ids == calibration[D1_START:20],
            "selected_slice_ids_mismatch",
        )
        _add_error(
            errors,
            not set(case_ids).intersection(calibration[:D1_START]),
            "pilot_or_d0_selection_overlap",
        )
        expected_order_hash = _sha256_bytes(
            _canonical_json(
                list(calibration + confirmatory + untouched)
            ).encode("utf-8")
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
        _add_error(
            errors,
            selection.get("order_hash") == expected_order_hash,
            "selection_order_hash_mismatch",
        )
        _add_error(
            errors,
            selection.get("split_hash") == expected_split_hash,
            "selection_split_hash_mismatch",
        )
        _add_error(
            errors,
            selection.get("manifest_id") == expected_selection_id,
            "selection_manifest_id_mismatch",
        )
        _add_error(
            errors,
            summary.get("selection_manifest_id") == expected_selection_id,
            "summary_selection_manifest_id_mismatch",
        )
    return model_alias, case_ids, request_ids


def _validate_d1_summary_and_hashes(
    *,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    streams: Mapping[str, Sequence[StreamRow]],
    receipts: Mapping[str, Mapping[str, Any]],
    observed_hashes: Mapping[str, str],
    errors: list[str],
) -> None:
    expected_counts = {
        "features": D1_CASE_COUNT,
        "commitments": D1_CASE_COUNT,
        "outcomes": D1_CASE_COUNT * len(OUTCOME_ACTIONS),
        "receipts": D1_CASE_COUNT,
    }
    _add_error(
        errors,
        manifest.get("expected_counts") == expected_counts,
        "manifest_expected_counts_mismatch",
    )
    exact_summary = {
        "planned_case_count": D1_CASE_COUNT,
        "attempted_case_count": D1_CASE_COUNT,
        "not_run_due_to_abort_count": 0,
        "pass_count": D1_CASE_COUNT,
        "failure_count": 0,
        "feature_count": D1_CASE_COUNT,
        "commitment_count": D1_CASE_COUNT,
        "outcome_count": D1_CASE_COUNT * len(OUTCOME_ACTIONS),
        "action_receipt_count": D1_CASE_COUNT,
    }
    for key, expected in exact_summary.items():
        _add_error(
            errors,
            summary.get(key) == expected,
            f"summary_count_mismatch:{key}",
        )
    _add_error(
        errors, summary.get("run_status") == "completed", "summary_run_not_completed"
    )
    _add_error(
        errors, summary.get("abort_failure_type") is None, "summary_abort_present"
    )
    _add_error(errors, summary.get("all_pass") is True, "summary_all_pass_false")
    _add_error(
        errors,
        summary.get("expected_counts_exact") is True,
        "summary_exact_counts_false",
    )
    _add_error(
        errors,
        summary.get("all_rollbacks_exact") is True,
        "summary_rollback_false",
    )
    _add_error(
        errors, summary.get("projector_files_loaded") == [], "projector_loaded"
    )
    _add_error(
        errors,
        len(streams["features.jsonl"]) == D1_CASE_COUNT,
        "observed_feature_count_mismatch",
    )
    _add_error(
        errors,
        len(streams["actions.jsonl"]) == D1_CASE_COUNT,
        "observed_action_count_mismatch",
    )
    _add_error(
        errors,
        len(streams["outcomes.jsonl"])
        == D1_CASE_COUNT * len(OUTCOME_ACTIONS),
        "observed_outcome_count_mismatch",
    )
    _add_error(
        errors,
        len(streams["events.jsonl"]) == D1_CASE_COUNT,
        "observed_event_count_mismatch",
    )
    _add_error(
        errors, len(receipts) == D1_CASE_COUNT, "observed_receipt_count_mismatch"
    )

    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("summary_artifacts_missing")
        return
    for file_name in _HASHED_FILES:
        key = file_name.replace(".jsonl", "_sha256").replace(".json", "_sha256")
        _add_error(
            errors,
            artifacts.get(key) == observed_hashes.get(file_name),
            f"summary_artifact_hash_mismatch:{file_name}",
        )
    actual_receipts = {
        name.removeprefix("action_receipts/"): digest
        for name, digest in observed_hashes.items()
        if name.startswith("action_receipts/")
    }
    _add_error(
        errors,
        artifacts.get("action_receipts") == actual_receipts,
        "summary_receipt_hash_map_mismatch",
    )


def _effect_summary(values: Sequence[float]) -> dict[str, Any]:
    return {
        "planned_case_count": D1_CASE_COUNT,
        "finite_case_count": len(values),
        "failure_or_missing_case_count": D1_CASE_COUNT - len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "observed_case_range": [min(values), max(values)] if values else None,
    }


def _analyze_d1_internal(
    run_directory: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    root = Path(run_directory).expanduser().resolve()
    if not root.is_dir():
        raise ScoreMixAnalysisError("D1 run directory does not exist")
    manifest = _read_json_object(root / "manifest.json")
    summary = _read_json_object(root / "summary.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ScoreMixAnalysisError("D1 manifest schema mismatch")
    if summary.get("schema_version") != SUMMARY_SCHEMA:
        raise ScoreMixAnalysisError("D1 summary schema mismatch")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ScoreMixAnalysisError("D1 run_id invalid")
    streams = {
        name: _read_jsonl(root / name, run_id=run_id) for name in _STREAM_FILES
    }
    receipt_root = root / "action_receipts"
    if receipt_root.is_symlink() or not receipt_root.is_dir():
        raise ScoreMixAnalysisError("D1 action_receipts must be a regular directory")
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
    model_alias, case_ids, request_ids = _validate_d1_envelope(
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
    # The shared D0 row validator has one count-only D0 specialization.
    # All receipt identities/chains were already checked above; replace only
    # that five-count error with the exact D1 twelve-count contract.
    errors[:] = [code for code in errors if code != "receipt_count_mismatch"]
    _add_error(
        errors, len(receipts) == D1_CASE_COUNT, "d1_receipt_count_mismatch"
    )
    _validate_d1_summary_and_hashes(
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
    for case_id in case_ids:
        mix = _safe_progress(outcomes, case_id, "score_mix")
        uniform = _safe_progress(outcomes, case_id, "uniform")
        replay = _safe_progress(outcomes, case_id, "no_op_replay")
        gain = mix - uniform if mix is not None and uniform is not None else None
        if gain is not None:
            gains.append(gain)
        feature = features.get(case_id)
        action = actions.get(case_id)
        distance: float | None = None
        predicted_mix: float | None = None
        predicted_uniform: float | None = None
        if feature is not None and action is not None:
            try:
                distance = _finite(
                    feature.get("operational_c_distance"),
                    name="operational_c_distance",
                )
                predicted_mix = _finite(
                    action.get("predicted_score"), name="predicted_score"
                )
                scores = feature.get("action_scores")
                if isinstance(scores, dict):
                    single = [
                        _finite(scores.get(key), name=f"score:{key}")
                        for key in SINGLE_ACTIONS
                    ]
                    predicted_uniform = sum(single) / math.sqrt(len(single))
            except ScoreMixAnalysisError:
                distance = None
                predicted_mix = None
                predicted_uniform = None
        predicted_gap = (
            predicted_mix - predicted_uniform
            if predicted_mix is not None and predicted_uniform is not None
            else None
        )
        case_rows.append(
            {
                "case_id": case_id,
                "p_score_mix": mix,
                "p_uniform": uniform,
                "g_mix": gain,
                "absolute_replay_progress": abs(replay) if replay is not None else None,
                "model_d1_replay_envelope": replay_envelope,
                "g_mix_above_d1_replay_envelope": (
                    gain > replay_envelope
                    if gain is not None and replay_envelope is not None
                    else None
                ),
                "predicted_slope_gap": predicted_gap,
                "realized_slope_gap": (
                    gain / distance
                    if gain is not None and distance is not None and distance > 0
                    else None
                ),
            }
        )

    error_codes = sorted(set(errors))
    panel_complete = (
        len(case_rows) == D1_CASE_COUNT
        and len(gains) == D1_CASE_COUNT
        and len(finite_replay) == D1_CASE_COUNT
    )
    valid = not error_codes
    selection = manifest.get("selection")
    selection_case_ids = (
        selection.get("case_ids") if isinstance(selection, dict) else None
    )
    calibration = (
        tuple(selection_case_ids.get("calibration", ()))
        if isinstance(selection_case_ids, dict)
        else ()
    )
    pilot_d0_disjoint = bool(
        len(calibration) == 20
        and not set(case_ids).intersection(calibration[:D1_START])
    )
    analysis_status = (
        "d1_descriptive_complete"
        if valid and panel_complete
        else "technical_block_invalid_artifacts"
        if not valid
        else "technical_block_incomplete_panel"
    )
    report = {
        "schema_version": ANALYSIS_SCHEMA,
        "claim_status": "d1_calibration_descriptive_only",
        "analysis_status": analysis_status,
        "run_id": run_id,
        "model_alias": model_alias,
        "analysis_policy": {
            "d1_role": "calibration completion and static-policy freeze input only",
            "raw_tensor_artifacts_opened": False,
            "inference_or_confidence_interval": False,
            "positive_result_meaning": (
                "descriptive calibration observation only; not GO, method-gain "
                "claim, confirmatory success, or MV-2 entry"
            ),
            "pair_level_decision_computed": False,
            "analysis_code_sha256": _sha256_file(Path(__file__)),
        },
        "artifact_validation": {
            "valid": valid,
            "panel_complete": panel_complete,
            "error_codes": error_codes,
            "expected_counts": {
                "cases": 12,
                "features": 12,
                "commitments": 12,
                "outcomes": 60,
                "events": 12,
                "receipts": 12,
            },
            "observed_counts": {
                "features": len(streams["features.jsonl"]),
                "commitments": len(streams["actions.jsonl"]),
                "outcomes": len(streams["outcomes.jsonl"]),
                "events": len(streams["events.jsonl"]),
                "receipts": len(receipts),
            },
            "selection_firewall": {
                "slice": "calibration[8:20]",
                "pilot_and_d0_disjoint": pilot_d0_disjoint,
            },
            "commitment_and_outcome_firewall_valid": valid,
        },
        "failure_denominator": {
            "planned_case_count": D1_CASE_COUNT,
            "finite_gain_count": len(gains),
            "finite_replay_count": len(finite_replay),
            "missing_or_failed_cases_are_not_dropped": True,
        },
        "d1_replay_envelope": {
            "policy": "max(1e-12, max(abs(no_op_replay progress))) over D1",
            "value": replay_envelope,
            "numeric_floor": NUMERIC_COMPARISON_TOLERANCE,
            "role": "descriptive D1 near-zero reference; not a GO threshold",
        },
        "case_diagnostics": case_rows,
        "descriptive_d1_effect": {
            "g_mix": _effect_summary(gains),
            "events_above_d1_replay_envelope": (
                sum(
                    bool(row["g_mix_above_d1_replay_envelope"])
                    for row in case_rows
                )
                if valid and panel_complete
                else None
            ),
            "label": (
                "observed D1 calibration values only; no expected improvement "
                "or population claim"
            ),
        },
    }
    _assert_json_safe(report, location="d1_analysis_output")
    json.dumps(report, allow_nan=False)
    return report, features, manifest


def analyze_d1_run(run_directory: str | Path) -> dict[str, Any]:
    """Analyze one model's sanitized D1 run."""

    report, _features, _manifest = _analyze_d1_internal(run_directory)
    return report


def fit_frozen_static_policy_from_features(
    *,
    model_alias: str,
    d0_run_id: str,
    d1_run_id: str,
    d0_features: Sequence[Mapping[str, Any]],
    d1_features: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fit the canonical policy from 17 feature rows; outcomes are not accepted."""

    if model_alias not in D1_RUN_IDS:
        raise ScoreMixAnalysisError("static policy model is outside fixed models")
    if d0_run_id != D0_RUN_IDS[model_alias]:
        raise ScoreMixAnalysisError("static policy D0 run_id mismatch")
    if d1_run_id != D1_RUN_IDS[model_alias]:
        raise ScoreMixAnalysisError("static policy D1 run_id mismatch")
    if len(d0_features) != D0_CASE_COUNT or len(d1_features) != D1_CASE_COUNT:
        raise ScoreMixAnalysisError("static policy requires exact 5+12 features")

    normalized: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    sums = {action: 0.0 for action in SINGLE_ACTIONS}
    for wave, rows in (("d0", d0_features), ("d1", d1_features)):
        for feature in rows:
            if set(feature) != _FEATURE_FIELDS:
                raise ScoreMixAnalysisError("static feature schema mismatch")
            if _contains_outcome_field(feature):
                raise ScoreMixAnalysisError("outcome field reached static fit")
            observed_hash, expected_hash = _payload_hash(feature, "feature_hash")
            if observed_hash != expected_hash:
                raise ScoreMixAnalysisError("static feature hash mismatch")
            if feature.get("q") != SCORE_MIX_Q:
                raise ScoreMixAnalysisError("static feature q mismatch")
            case_id = feature.get("case_id")
            if not isinstance(case_id, str) or case_id in seen_cases:
                raise ScoreMixAnalysisError("static feature case overlap")
            if not _is_full_hash(feature.get("request_id")):
                raise ScoreMixAnalysisError("static feature request hash mismatch")
            seen_cases.add(case_id)
            scores = feature.get("action_scores")
            if not isinstance(scores, dict) or set(scores) != set(PROBE_ACTIONS):
                raise ScoreMixAnalysisError("static single-slope panel mismatch")
            single = {
                action: _finite(scores.get(action), name=f"static:{action}")
                for action in SINGLE_ACTIONS
            }
            for action, value in single.items():
                sums[action] += value
            normalized.append(
                {
                    "wave": wave,
                    "case_id": case_id,
                    "request_id": feature.get("request_id"),
                    "feature_hash": observed_hash,
                    "single_slopes": single,
                }
            )
    if len(seen_cases) != TOTAL_STATIC_CASE_COUNT:
        raise ScoreMixAnalysisError("static feature denominator mismatch")

    sbar = {
        action: sums[action] / TOTAL_STATIC_CASE_COUNT
        for action in SINGLE_ACTIONS
    }
    weights, predicted_score, branch = _expected_mix(
        {**sbar, "uniform": sum(sbar.values()) / math.sqrt(len(sbar))}
    )
    feature_panel_hash = _sha256_bytes(
        _canonical_json(normalized).encode("utf-8")
    )
    policy: dict[str, Any] = {
        "schema_version": STATIC_POLICY_SCHEMA,
        "model_alias": model_alias,
        "source_runs": {"d0": d0_run_id, "d1": d1_run_id},
        "source_slices": {
            "d0": {"start": 3, "count": 5},
            "d1": {"start": 8, "count": 12},
        },
        "feature_case_count": TOTAL_STATIC_CASE_COUNT,
        "feature_panel_sha256": feature_panel_hash,
        "fit_numeric_inputs": [
            f"action_scores.{action}" for action in SINGLE_ACTIONS
        ],
        "outcome_fields_used": [],
        "sbar_single_layer_slopes": sbar,
        "layer_weights": weights,
        "predicted_mean_score": predicted_score,
        "controller": "relu(sbar)/L2-else-max-onehot",
        "controller_branch": branch,
        "tie_policy": "exact-tie-lower-layer",
        "weight_l2_norm": math.sqrt(
            sum(weight * weight for weight in weights.values())
        ),
        "claim_boundary": (
            "frozen calibration comparator only; no positive claim, GO, "
            "confirmatory success, or MV-2 authorization"
        ),
    }
    policy["policy_hash"] = _sha256_bytes(
        _canonical_json(policy).encode("utf-8")
    )
    _assert_json_safe(policy, location="static_policy")
    json.dumps(policy, allow_nan=False)
    return policy


def _load_validated_d0_features(
    run_directory: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    report = analyze_score_mix_run(run_directory)
    if (
        report.get("analysis_status") != "d0_descriptive_complete"
        or report.get("artifact_validation", {}).get("valid") is not True
        or report.get("artifact_validation", {}).get("panel_complete") is not True
    ):
        raise ScoreMixAnalysisError("D0 is not independently valid and complete")
    root = Path(run_directory).expanduser().resolve()
    manifest = _read_json_object(root / "manifest.json")
    run_id = manifest.get("run_id")
    rows = _read_jsonl(root / "features.jsonl", run_id=run_id)
    return report, [row.payload for row in rows], manifest


def _cross_validate_d0_d1(
    *,
    d0_report: Mapping[str, Any],
    d0_manifest: Mapping[str, Any],
    d1_report: Mapping[str, Any],
    d1_manifest: Mapping[str, Any],
) -> None:
    model_alias = d1_report.get("model_alias")
    if d0_report.get("model_alias") != model_alias:
        raise ScoreMixAnalysisError("D0/D1 model mismatch")
    if d0_manifest.get("run_id") != D0_RUN_IDS.get(model_alias):
        raise ScoreMixAnalysisError("D0 run is outside fixed envelope")
    if d1_manifest.get("run_id") != D1_RUN_IDS.get(model_alias):
        raise ScoreMixAnalysisError("D1 run is outside fixed envelope")
    d0_selection = d0_manifest.get("selection")
    d1_selection = d1_manifest.get("selection")
    if (
        not isinstance(d0_selection, dict)
        or not isinstance(d1_selection, dict)
        or d0_selection.get("manifest_id") != d1_selection.get("manifest_id")
    ):
        raise ScoreMixAnalysisError("D0/D1 selection manifest mismatch")
    calibration = tuple(d1_selection["case_ids"]["calibration"])
    d0_cases = tuple(d0_manifest.get("selected_case_ids", ()))
    d1_cases = tuple(d1_manifest.get("selected_case_ids", ()))
    if (
        d0_cases != calibration[3:8]
        or d1_cases != calibration[8:20]
        or set(d0_cases).intersection(d1_cases)
        or set(calibration[:3]).intersection((*d0_cases, *d1_cases))
    ):
        raise ScoreMixAnalysisError("D0/D1 slices overlap or differ from lock")


def build_small_summary(
    report: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    effect = report["descriptive_d1_effect"]["g_mix"]
    return {
        "schema_version": SMALL_SUMMARY_SCHEMA,
        "claim_status": report["claim_status"],
        "analysis_status": report["analysis_status"],
        "run_id": report["run_id"],
        "model_alias": report["model_alias"],
        "artifact_valid": report["artifact_validation"]["valid"],
        "panel_complete": report["artifact_validation"]["panel_complete"],
        "error_codes": report["artifact_validation"]["error_codes"],
        "d1_replay_envelope": report["d1_replay_envelope"]["value"],
        "finite_gain_count": effect["finite_case_count"],
        "mean_g_mix": effect["mean"],
        "observed_g_mix_range": effect["observed_case_range"],
        "events_above_d1_replay_envelope": report["descriptive_d1_effect"][
            "events_above_d1_replay_envelope"
        ],
        "static_policy_hash": policy["policy_hash"],
        "static_policy_feature_case_count": policy["feature_case_count"],
        "pair_level_decision_computed": False,
        "decision_boundary": (
            "D1 descriptive calibration and frozen comparator only; never GO"
        ),
    }


def render_korean_markdown(
    report: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> str:
    validation = report["artifact_validation"]
    lines = [
        f"# MV-1 score-mix D1 독립 분석: {report['model_alias']}",
        "",
        f"- run ID: `{report['run_id']}`",
        f"- 분석 상태: `{report['analysis_status']}`",
        f"- artifact/panel: `valid={validation['valid']}`, "
        f"`complete={validation['panel_complete']}`",
        "- 해석 경계: D1 calibration과 frozen static comparator 생성만 수행한다. "
        "양수 결과도 방법 개선 claim, GO, confirmatory success 또는 MV-2 "
        "진입을 허용하지 않는다.",
        f"- D1 replay envelope: `{report['d1_replay_envelope']['value']}`",
        "",
        "## Case별 descriptive outcome",
        "",
        "| case | P(score_mix) | P(uniform) | G_mix | e_D1 초과 |",
        "| --- | ---: | ---: | ---: | :---: |",
    ]

    def number(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.8g}"

    for row in report["case_diagnostics"]:
        above = row["g_mix_above_d1_replay_envelope"]
        lines.append(
            f"| `{row['case_id']}` | {number(row['p_score_mix'])} | "
            f"{number(row['p_uniform'])} | {number(row['g_mix'])} | "
            f"{'NA' if above is None else ('yes' if above else 'no')} |"
        )
    lines.extend(
        [
            "",
            "## Frozen static policy",
            "",
            "- 입력: independently valid D0 `[3:8]`와 D1 `[8:20]`의 17개 "
            "feature에서 다섯 single-layer slope만 사용했다.",
            "- outcome field 사용: 없음",
            f"- policy hash: `{policy['policy_hash']}`",
            "",
            "| layer action | sbar | frozen weight |",
            "| --- | ---: | ---: |",
        ]
    )
    for action in SINGLE_ACTIONS:
        lines.append(
            f"| `{action}` | "
            f"{number(policy['sbar_single_layer_slopes'][action])} | "
            f"{number(policy['layer_weights'][action])} |"
        )
    lines.extend(
        [
            "",
            "- 이 policy는 confirmatory comparator 후보를 freeze한 것이며 "
            "positive efficacy evidence가 아니다.",
            "- pair-level 또는 confirmatory 결정은 이 단일 모델 문서에서 "
            "계산하지 않는다.",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze one MV-1 D1 run and freeze a D0+D1 slope-only policy."
    )
    parser.add_argument("d1_run_directory", type=Path)
    parser.add_argument("--d0-run-directory", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--static-policy-output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        d1_report, d1_features_by_case, d1_manifest = _analyze_d1_internal(
            args.d1_run_directory
        )
        if (
            d1_report["analysis_status"] != "d1_descriptive_complete"
            or d1_report["artifact_validation"]["valid"] is not True
            or d1_report["artifact_validation"]["panel_complete"] is not True
        ):
            raise ScoreMixAnalysisError("D1 is not independently valid and complete")
        d0_report, d0_features, d0_manifest = _load_validated_d0_features(
            args.d0_run_directory
        )
        _cross_validate_d0_d1(
            d0_report=d0_report,
            d0_manifest=d0_manifest,
            d1_report=d1_report,
            d1_manifest=d1_manifest,
        )
        d1_features = [
            d1_features_by_case[case_id]
            for case_id in d1_manifest["selected_case_ids"]
        ]
        policy = fit_frozen_static_policy_from_features(
            model_alias=d1_report["model_alias"],
            d0_run_id=d0_manifest["run_id"],
            d1_run_id=d1_manifest["run_id"],
            d0_features=d0_features,
            d1_features=d1_features,
        )
        summary = build_small_summary(d1_report, policy)
        outputs = {
            args.report_output.resolve(),
            args.summary_output.resolve(),
            args.static_policy_output.resolve(),
        }
        if len(outputs) != 3:
            raise ScoreMixAnalysisError("all three output paths must differ")
        _write_exclusive(
            args.report_output,
            render_korean_markdown(d1_report, policy).encode("utf-8"),
        )
        _write_exclusive(
            args.summary_output,
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
        _write_exclusive(
            args.static_policy_output,
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
                    "status": d1_report["analysis_status"],
                    "artifact_valid": True,
                    "report_written": True,
                    "summary_written": True,
                    "static_policy_written": True,
                    "static_policy_hash": policy["policy_hash"],
                },
                sort_keys=True,
            )
        )
        return 0
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
