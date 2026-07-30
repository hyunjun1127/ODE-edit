"""Precommitted, pure-CPU analysis for one MV-1 C0 pilot model run.

The analyzer consumes only the sanitized JSON artifacts emitted by
``mv1_calibration``.  It never opens ``direct_z`` or any tensor artifact.  A
three-case result is always descriptive: numerical effects are labelled as
preliminary observed bounds, never as a method-gain estimate.
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
from typing import Any, Iterable, Mapping, Sequence


ANALYSIS_SCHEMA = "ode-edit-mv1-pilot-analysis/v1"
MANIFEST_SCHEMA = "ode-edit-mv1-c0-manifest/v1"
SUMMARY_SCHEMA = "ode-edit-mv1-c0-summary/v1"
STREAM_SCHEMA = "ode-edit-mv1-c0/v1"
RECEIPT_SCHEMA = "ode-edit-mv1-action-receipt/v1"
UNIFORM_ACTION = "uniform"
ORDERED_ACTION = "ordered_global_alpha"
REPLAY_ACTION = "no_op_replay"

MIN_SIGN_CONCORDANCE = 0.80
MAX_MEDIAN_RELATIVE_ERROR = 0.25
MAX_P90_RELATIVE_ERROR = 0.75
NUMERIC_FLOOR = 1e-12

_STREAM_FILES = ("features.jsonl", "actions.jsonl", "outcomes.jsonl", "events.jsonl")
_HASHED_FILES = ("manifest.json", *_STREAM_FILES)
_FORBIDDEN_KEYS = {
    "answer",
    "answers",
    "input_ids",
    "logits",
    "predicted_ids",
    "prompt",
    "prompts",
    "raw_generation",
    "raw_generations",
    "target",
    "target_ids",
    "target_new",
    "token_ids",
}
_EXPECTED_EVENTS = {
    "features.jsonl": {"mv1_feature"},
    "actions.jsonl": {"mv1_action_commitment"},
    "outcomes.jsonl": {
        "mv1_operational_outcome",
        "mv1_contextual_outcome",
        "mv1_reference_outcome",
        "mv1_replay_outcome",
    },
    "events.jsonl": {"mv1_case"},
}


class PilotAnalysisError(RuntimeError):
    """Raised when sanitized artifacts cannot be parsed safely."""


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


def _full_hash(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PilotAnalysisError(f"{name} is not a full lowercase SHA-256 digest")
    return value


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise PilotAnalysisError(f"{name} is not finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PilotAnalysisError(f"{name} is not finite") from exc
    if not math.isfinite(result):
        raise PilotAnalysisError(f"{name} is not finite")
    return result


def _nonempty_identifier(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or any(ord(character) < 32 for character in value)
    ):
        raise PilotAnalysisError(f"{name} is not a valid identifier")
    return value


def _assert_json_safe(value: Any, *, location: str = "root") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PilotAnalysisError(f"non-finite scalar at {location}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_json_safe(item, location=f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PilotAnalysisError(f"non-string key at {location}")
            if key.lower() in _FORBIDDEN_KEYS:
                raise PilotAnalysisError(f"forbidden raw field at {location}.{key}")
            _assert_json_safe(item, location=f"{location}.{key}")
        return
    raise PilotAnalysisError(f"non-JSON value at {location}")


def _read_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PilotAnalysisError(f"required regular file is unavailable: {path.name}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PilotAnalysisError(f"cannot parse {path.name}") from exc
    if not isinstance(value, dict):
        raise PilotAnalysisError(f"{path.name} must contain one JSON object")
    _assert_json_safe(value, location=path.name)
    return value


def _unwrap_stream_record(
    raw: Mapping[str, Any],
    *,
    path: Path,
    line_number: int,
    run_id: str,
) -> StreamRow:
    if raw.get("schema_version") != STREAM_SCHEMA:
        raise PilotAnalysisError(
            f"{path.name}:{line_number} has an unexpected schema"
        )
    if raw.get("run_id") != run_id:
        raise PilotAnalysisError(f"{path.name}:{line_number} run_id mismatch")
    event = _nonempty_identifier(raw.get("event"), name="stream event")
    if "payload" in raw:
        if set(raw) != {
            "schema_version",
            "run_id",
            "sequence",
            "recorded_at",
            "event",
            "payload",
        }:
            raise PilotAnalysisError(
                f"{path.name}:{line_number} has an ambiguous envelope"
            )
        if raw.get("sequence") != line_number - 1:
            raise PilotAnalysisError(
                f"{path.name}:{line_number} sequence is not contiguous"
            )
        _nonempty_identifier(
            raw.get("recorded_at"),
            name=f"{path.name}:{line_number} recorded_at",
        )
        payload = raw["payload"]
        if not isinstance(payload, dict):
            raise PilotAnalysisError(
                f"{path.name}:{line_number} payload must be an object"
            )
        normalized = dict(payload)
    else:
        raise PilotAnalysisError(
            f"{path.name}:{line_number} has no sanitized payload envelope"
        )
    return StreamRow(event=event, payload=normalized, line_number=line_number)


def _read_jsonl(path: Path, *, run_id: str) -> list[StreamRow]:
    if path.is_symlink() or not path.is_file():
        raise PilotAnalysisError(f"required regular file is unavailable: {path.name}")
    rows: list[StreamRow] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise PilotAnalysisError(
                        f"{path.name}:{line_number} is an empty record"
                    )
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise PilotAnalysisError(
                        f"{path.name}:{line_number} must be an object"
                    )
                _assert_json_safe(raw, location=f"{path.name}:{line_number}")
                rows.append(
                    _unwrap_stream_record(
                        raw,
                        path=path,
                        line_number=line_number,
                        run_id=run_id,
                    )
                )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PilotAnalysisError(f"cannot parse {path.name}") from exc
    return rows


def _fraction_label(value: float) -> str:
    denominator = round(1.0 / value)
    if denominator <= 0 or not math.isclose(
        value, 1.0 / denominator, rel_tol=0.0, abs_tol=1e-15
    ):
        raise PilotAnalysisError("fraction is not an exact reciprocal")
    return f"q_1_{denominator}"


def _payload_hash(payload: Mapping[str, Any], *, hash_field: str) -> str:
    stripped = dict(payload)
    observed = _full_hash(stripped.pop(hash_field, None), name=hash_field)
    expected = _sha256_bytes(_canonical_json(stripped).encode("utf-8"))
    if observed != expected:
        raise PilotAnalysisError(f"{hash_field} does not match its payload")
    return observed


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sign(value: float, *, tolerance: float) -> int:
    if abs(value) <= tolerance:
        return 0
    return 1 if value > 0 else -1


def _safe_mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _effect_summary(
    case_effects: Sequence[tuple[str, float | None, str]],
) -> dict[str, Any]:
    finite = [value for _, value, _ in case_effects if value is not None]
    return {
        "planned_case_count": len(case_effects),
        "finite_pair_count": len(finite),
        "failure_or_missing_case_count": len(case_effects) - len(finite),
        "mean": _safe_mean(finite),
        "median": statistics.median(finite) if finite else None,
        "positive_fraction_itd": (
            sum(value is not None and value > 0 for _, value, _ in case_effects)
            / len(case_effects)
            if case_effects
            else None
        ),
        "observed_case_range": (
            [min(finite), max(finite)] if finite else None
        ),
        "case_effects": [
            {"case_id": case_id, "effect": value, "status": status}
            for case_id, value, status in case_effects
        ],
    }


def _add_error(errors: list[str], condition: bool, code: str) -> None:
    if not condition:
        errors.append(code)


def _unique_index(
    rows: Iterable[StreamRow],
    *,
    key_fields: Sequence[str],
    event: str | None = None,
    errors: list[str],
    label: str,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if event is not None and row.event != event:
            continue
        try:
            key = tuple(row.payload[field] for field in key_fields)
        except KeyError:
            errors.append(f"{label}_missing_key")
            continue
        if key in index:
            errors.append(f"{label}_duplicate_key")
            continue
        index[key] = row.payload
    return index


def _progress(
    payload: Mapping[str, Any] | None,
) -> tuple[float | None, str]:
    if payload is None:
        return None, "missing"
    status = payload.get("status")
    if status != "completed":
        return None, f"status:{status}"
    try:
        value = _finite(payload.get("progress"), name="progress")
    except PilotAnalysisError:
        return None, "non_finite"
    return value, "completed"


def _validate_run_artifacts(
    run_directory: Path,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    streams: Mapping[str, Sequence[StreamRow]],
    receipts: Mapping[str, Mapping[str, Any]],
    observed_hashes: Mapping[str, str],
) -> tuple[
    list[str],
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    errors: list[str] = []
    run_id = manifest["run_id"]
    case_ids = tuple(manifest["selected_case_ids"])
    request_ids = tuple(manifest["selected_request_ids"])
    fractions = tuple(float(value) for value in manifest["fractions"])
    fraction_labels = tuple(_fraction_label(value) for value in fractions)
    action_set = tuple(manifest["action_set"])

    _add_error(errors, len(case_ids) == len(set(case_ids)), "duplicate_case_id")
    _add_error(errors, len(request_ids) == len(case_ids), "request_count_mismatch")
    for request_id in request_ids:
        try:
            _full_hash(request_id, name="selected_request_id")
        except PilotAnalysisError:
            errors.append("invalid_selected_request_hash")
    _add_error(errors, len(fractions) == len(set(fractions)), "duplicate_fraction")
    _add_error(errors, action_set and action_set[-1] == UNIFORM_ACTION, "uniform_not_last")
    _add_error(errors, len(action_set) == len(set(action_set)), "duplicate_action")

    for file_name, rows in streams.items():
        _add_error(
            errors,
            all(row.event in _EXPECTED_EVENTS[file_name] for row in rows),
            f"{file_name}_unexpected_event",
        )

    features = _unique_index(
        streams["features.jsonl"],
        key_fields=("case_id", "fraction_label"),
        errors=errors,
        label="feature",
    )
    actions = _unique_index(
        streams["actions.jsonl"],
        key_fields=("case_id", "fraction_label"),
        errors=errors,
        label="action",
    )
    operational = _unique_index(
        streams["outcomes.jsonl"],
        key_fields=("case_id", "fraction_label", "action_id"),
        event="mv1_operational_outcome",
        errors=errors,
        label="operational",
    )
    ordered = _unique_index(
        streams["outcomes.jsonl"],
        key_fields=("case_id", "fraction_label"),
        event="mv1_contextual_outcome",
        errors=errors,
        label="ordered",
    )
    replay_index = _unique_index(
        streams["outcomes.jsonl"],
        key_fields=("case_id",),
        event="mv1_replay_outcome",
        errors=errors,
        label="replay",
    )
    replay = {key[0]: value for key, value in replay_index.items()}
    reference_index = _unique_index(
        streams["outcomes.jsonl"],
        key_fields=("case_id",),
        event="mv1_reference_outcome",
        errors=errors,
        label="reference",
    )
    reference = {key[0]: value for key, value in reference_index.items()}
    events_index = _unique_index(
        streams["events.jsonl"],
        key_fields=("case_id",),
        errors=errors,
        label="event",
    )
    events = {key[0]: value for key, value in events_index.items()}

    for case_id in case_ids:
        _nonempty_identifier(case_id, name="case_id")
    expected_cases = set(case_ids)
    for label, index in (
        ("feature", features),
        ("action", actions),
        ("operational", operational),
        ("ordered", ordered),
    ):
        _add_error(
            errors,
            all(key[0] in expected_cases for key in index),
            f"{label}_case_outside_manifest",
        )
    _add_error(
        errors,
        all(case_id in expected_cases for case_id in replay),
        "replay_case_outside_manifest",
    )

    request_by_case: dict[str, str] = {}
    for payload in [
        *features.values(),
        *actions.values(),
        *operational.values(),
        *ordered.values(),
        *replay.values(),
        *reference.values(),
        *events.values(),
    ]:
        case_id = payload.get("case_id")
        request_id = payload.get("request_id")
        if case_id not in expected_cases:
            continue
        try:
            request_id = _full_hash(request_id, name="request_id")
        except PilotAnalysisError:
            errors.append("row_invalid_request_hash")
            continue
        previous = request_by_case.setdefault(case_id, request_id)
        _add_error(errors, previous == request_id, "case_request_hash_mismatch")
    _add_error(
        errors,
        set(request_by_case.values()).issubset(set(request_ids)),
        "row_request_outside_manifest",
    )

    for key, feature in features.items():
        case_id, fraction_label = key
        _add_error(
            errors,
            fraction_label in fraction_labels,
            "feature_fraction_outside_manifest",
        )
        try:
            fraction = _finite(feature.get("fraction"), name="feature fraction")
            _add_error(
                errors,
                _fraction_label(fraction) == fraction_label,
                "feature_fraction_label_mismatch",
            )
            _payload_hash(feature, hash_field="feature_hash")
        except PilotAnalysisError:
            errors.append("feature_hash_or_fraction_invalid")
        scores = feature.get("action_scores")
        _add_error(
            errors,
            isinstance(scores, dict) and set(scores) == set(action_set),
            "feature_action_set_mismatch",
        )
        if isinstance(scores, dict):
            for value in scores.values():
                try:
                    _finite(value, name="action score")
                except PilotAnalysisError:
                    errors.append("feature_non_finite_score")
        _add_error(
            errors,
            feature.get("case_id") == case_id,
            "feature_case_identity_mismatch",
        )

    for key, action in actions.items():
        feature = features.get(key)
        try:
            commitment_hash = _payload_hash(
                action, hash_field="commitment_hash"
            )
        except PilotAnalysisError:
            commitment_hash = None
            errors.append("commitment_hash_invalid")
        _add_error(
            errors,
            feature is not None
            and action.get("feature_hash") == feature.get("feature_hash"),
            "commitment_feature_hash_mismatch",
        )
        _add_error(
            errors,
            action.get("action_id") in action_set,
            "committed_action_outside_set",
        )
        _add_error(
            errors,
            commitment_hash is not None,
            "commitment_unverifiable",
        )

    receipt_entry_count = 0
    receipt_keys: set[tuple[str, str]] = set()
    for receipt_name, receipt in receipts.items():
        _add_error(
            errors,
            receipt.get("schema_version") == RECEIPT_SCHEMA,
            "receipt_schema_mismatch",
        )
        _add_error(
            errors,
            receipt.get("durability")
            == "features-and-actions-flush+fsync-before-exclusive-receipt",
            "receipt_durability_mismatch",
        )
        entries = receipt.get("entries")
        if not isinstance(entries, list) or not entries:
            errors.append("receipt_entries_invalid")
            continue
        identity_fractions: list[float] = []
        for entry in entries:
            receipt_entry_count += 1
            if not isinstance(entry, dict):
                errors.append("receipt_entry_invalid")
                continue
            try:
                fraction = _finite(entry.get("fraction"), name="receipt fraction")
                fraction_label = _fraction_label(fraction)
            except PilotAnalysisError:
                errors.append("receipt_fraction_invalid")
                continue
            key = (entry.get("case_id"), fraction_label)
            receipt_keys.add(key)
            identity_fractions.append(fraction)
            feature = features.get(key)
            action = actions.get(key)
            _add_error(
                errors,
                feature is not None
                and action is not None
                and entry.get("request_id") == feature.get("request_id")
                and entry.get("feature_hash") == feature.get("feature_hash")
                and entry.get("commitment_hash") == action.get("commitment_hash")
                and entry.get("action_id") == action.get("action_id"),
                "receipt_chain_mismatch",
            )
        identity = {
            "case_id": receipt.get("case_id"),
            "request_id": receipt.get("request_id"),
            "fractions": sorted(identity_fractions),
        }
        expected_name = _sha256_bytes(
            _canonical_json(identity).encode("utf-8")
        ) + ".json"
        _add_error(errors, receipt_name == expected_name, "receipt_name_mismatch")

    _add_error(
        errors,
        receipt_keys == set(actions) == set(features),
        "receipt_coverage_mismatch",
    )

    for key, outcome in operational.items():
        case_id, fraction_label, action_id = key
        commitment = actions.get((case_id, fraction_label))
        feature = features.get((case_id, fraction_label))
        _add_error(
            errors,
            action_id in action_set,
            "operational_action_outside_set",
        )
        _add_error(
            errors,
            commitment is not None
            and outcome.get("commitment_hash")
            == commitment.get("commitment_hash"),
            "operational_commitment_chain_mismatch",
        )
        _add_error(
            errors,
            feature is not None
            and outcome.get("feature_hash") == feature.get("feature_hash"),
            "operational_feature_chain_mismatch",
        )
        _add_error(
            errors,
            outcome.get("rollback_exact") is True,
            "operational_rollback_not_exact",
        )
    for key, outcome in ordered.items():
        commitment = actions.get(key)
        _add_error(
            errors,
            outcome.get("action_id") == ORDERED_ACTION,
            "ordered_action_id_mismatch",
        )
        _add_error(
            errors,
            commitment is not None
            and outcome.get("commitment_hash")
            == commitment.get("commitment_hash"),
            "ordered_commitment_chain_mismatch",
        )
        _add_error(
            errors,
            outcome.get("rollback_exact") is True,
            "ordered_rollback_not_exact",
        )
    for outcome in replay.values():
        _add_error(
            errors,
            outcome.get("action_id") == REPLAY_ACTION,
            "replay_action_id_mismatch",
        )
        _add_error(
            errors,
            outcome.get("rollback_exact") is True,
            "replay_rollback_not_exact",
        )
    for outcome in reference.values():
        _add_error(
            errors,
            outcome.get("action_id") == "native_memit_full",
            "reference_action_id_mismatch",
        )
        _add_error(
            errors,
            outcome.get("rollback_exact") is True,
            "reference_rollback_not_exact",
        )

    for case_id, event in events.items():
        event_receipts = event.get("action_receipts", {})
        if event.get("pass") is True:
            _add_error(
                errors,
                isinstance(event_receipts, dict)
                and set(event_receipts) == set(fraction_labels),
                "event_receipt_coverage_mismatch",
            )
        if isinstance(event_receipts, dict):
            for fraction_label, reference in event_receipts.items():
                if not isinstance(reference, dict):
                    errors.append("event_receipt_reference_invalid")
                    continue
                name = reference.get("name")
                _add_error(
                    errors,
                    fraction_label in fraction_labels
                    and name in receipts
                    and reference.get("sha256") == observed_hashes.get(
                        f"action_receipts/{name}"
                    ),
                    "event_receipt_hash_mismatch",
                )

    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("summary_artifacts_invalid")
        artifacts = {}
    for file_name in _HASHED_FILES:
        summary_key = file_name.replace(".jsonl", "_sha256").replace(
            ".json", "_sha256"
        )
        _add_error(
            errors,
            artifacts.get(summary_key) == observed_hashes.get(file_name),
            f"summary_hash_mismatch:{file_name}",
        )
    expected_receipt_hashes = artifacts.get("action_receipts")
    observed_receipt_hashes = {
        name.removeprefix("action_receipts/"): digest
        for name, digest in observed_hashes.items()
        if name.startswith("action_receipts/")
    }
    _add_error(
        errors,
        expected_receipt_hashes == observed_receipt_hashes,
        "summary_receipt_hash_map_mismatch",
    )

    actual_counts = {
        "feature_count": len(streams["features.jsonl"]),
        "commitment_count": len(streams["actions.jsonl"]),
        "outcome_count": len(streams["outcomes.jsonl"]),
        "action_receipt_count": len(receipts),
    }
    for name, value in actual_counts.items():
        _add_error(
            errors,
            summary.get(name) == value,
            f"summary_count_mismatch:{name}",
        )
    _add_error(
        errors,
        summary.get("planned_case_count") == len(case_ids),
        "summary_planned_case_count_mismatch",
    )
    _add_error(
        errors,
        summary.get("attempted_case_count") == len(events),
        "summary_attempted_case_count_mismatch",
    )
    _add_error(
        errors,
        receipt_entry_count == len(actions),
        "receipt_entry_count_mismatch",
    )
    _add_error(errors, manifest.get("run_id") == summary.get("run_id"), "run_id_mismatch")
    _add_error(
        errors,
        run_directory.name == run_id,
        "run_directory_name_mismatch",
    )
    return errors, features, actions, operational, ordered, replay


def _replay_envelope(
    case_ids: Sequence[str],
    replay: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    values: list[float] = []
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        value, status = _progress(replay.get(case_id))
        if value is not None:
            values.append(abs(value))
        rows.append(
            {
                "case_id": case_id,
                "absolute_progress": abs(value) if value is not None else None,
                "status": status,
            }
        )
    return {
        "policy": "max-absolute-no_op_replay-progress; numeric floor only at exact zero",
        "planned_case_count": len(case_ids),
        "finite_replay_count": len(values),
        "failure_or_missing_count": len(case_ids) - len(values),
        "value": max(values) if values else None,
        "case_rows": rows,
    }


def _diagnose_fraction(
    *,
    fraction: float,
    case_ids: Sequence[str],
    action_set: Sequence[str],
    features: Mapping[tuple[str, str], Mapping[str, Any]],
    actions: Mapping[tuple[str, str], Mapping[str, Any]],
    operational: Mapping[tuple[str, str, str], Mapping[str, Any]],
    ordered: Mapping[tuple[str, str], Mapping[str, Any]],
    replay_envelope: float | None,
) -> dict[str, Any]:
    fraction_label = _fraction_label(fraction)
    comparisons: list[dict[str, Any]] = []
    relative_errors: list[float] = []
    sign_concordant = 0
    expected = len(case_ids) * len(action_set)
    for case_id in case_ids:
        feature = features.get((case_id, fraction_label))
        scores = feature.get("action_scores") if feature is not None else None
        distance: float | None
        try:
            distance = (
                _finite(feature.get("operational_c_distance"), name="distance")
                if feature is not None
                else None
            )
            if distance is not None and distance <= 0:
                distance = None
        except PilotAnalysisError:
            distance = None
        for action_id in action_set:
            outcome = operational.get((case_id, fraction_label, action_id))
            actual_progress, outcome_status = _progress(outcome)
            try:
                score = (
                    _finite(scores.get(action_id), name="score")
                    if isinstance(scores, dict) and action_id in scores
                    else None
                )
            except PilotAnalysisError:
                score = None
            if score is None or actual_progress is None or distance is None:
                comparisons.append(
                    {
                        "case_id": case_id,
                        "action_id": action_id,
                        "derivative_score": score,
                        "actual_slope": None,
                        "sign_concordant": False,
                        "relative_error": None,
                        "status": (
                            "missing_feature_or_distance"
                            if score is None or distance is None
                            else outcome_status
                        ),
                    }
                )
                continue
            actual_slope = actual_progress / distance
            replay_slope = (replay_envelope or 0.0) / distance
            denominator = max(abs(actual_slope), replay_slope, NUMERIC_FLOOR)
            relative_error = abs(score - actual_slope) / denominator
            concordant = _sign(score, tolerance=replay_slope) == _sign(
                actual_slope, tolerance=replay_slope
            )
            sign_concordant += int(concordant)
            relative_errors.append(relative_error)
            comparisons.append(
                {
                    "case_id": case_id,
                    "action_id": action_id,
                    "derivative_score": score,
                    "actual_slope": actual_slope,
                    "sign_concordant": concordant,
                    "relative_error": relative_error,
                    "status": "completed",
                }
            )

    finite_count = len(relative_errors)
    concordance_itd = sign_concordant / expected if expected else None
    concordance_finite = sign_concordant / finite_count if finite_count else None
    median_error = statistics.median(relative_errors) if relative_errors else None
    p90_error = _percentile(relative_errors, 0.90) if relative_errors else None
    complete = finite_count == expected
    criterion = {
        "minimum_sign_concordance": MIN_SIGN_CONCORDANCE,
        "maximum_median_relative_error": MAX_MEDIAN_RELATIVE_ERROR,
        "maximum_p90_relative_error": MAX_P90_RELATIVE_ERROR,
        "complete_panel_required": True,
    }
    locked_pass = bool(
        complete
        and concordance_itd is not None
        and concordance_itd >= MIN_SIGN_CONCORDANCE
        and median_error is not None
        and median_error <= MAX_MEDIAN_RELATIVE_ERROR
        and p90_error is not None
        and p90_error <= MAX_P90_RELATIVE_ERROR
    )

    def action_progress(case_id: str, action_id: str) -> tuple[float | None, str]:
        return _progress(
            operational.get((case_id, fraction_label, action_id))
        )

    action_means: dict[str, float] = {}
    static_complete = True
    for action_id in action_set:
        values: list[float] = []
        for case_id in case_ids:
            value, _ = action_progress(case_id, action_id)
            if value is None:
                static_complete = False
            else:
                values.append(value)
        if len(values) == len(case_ids):
            action_means[action_id] = sum(values) / len(values)
    static_action: str | None = None
    near_tied: list[str] = []
    if static_complete and len(action_means) == len(action_set):
        best = max(action_means.values())
        tie_tolerance = replay_envelope or 0.0
        near_tied = [
            action_id
            for action_id in action_set
            if best - action_means[action_id] <= tie_tolerance
        ]
        static_action = min(
            near_tied,
            key=lambda action_id: (
                action_id == UNIFORM_ACTION,
                action_set.index(action_id),
                action_id,
            ),
        )

    controller_vs_uniform: list[tuple[str, float | None, str]] = []
    controller_vs_static: list[tuple[str, float | None, str]] = []
    controller_vs_ordered: list[tuple[str, float | None, str]] = []
    ordered_vs_static: list[tuple[str, float | None, str]] = []
    oracle_vs_static: list[tuple[str, float | None, str]] = []
    oracle_near_tie_vs_static: list[tuple[str, float | None, str]] = []
    for case_id in case_ids:
        commitment = actions.get((case_id, fraction_label))
        selected = commitment.get("action_id") if commitment is not None else None
        selected_value, selected_status = (
            action_progress(case_id, selected)
            if selected in action_set
            else (None, "missing_commitment")
        )
        uniform_value, uniform_status = action_progress(case_id, UNIFORM_ACTION)
        ordered_value, ordered_status = _progress(
            ordered.get((case_id, fraction_label))
        )
        static_value, static_status = (
            action_progress(case_id, static_action)
            if static_action is not None
            else (None, "static_unavailable")
        )
        panel_values: list[float] = []
        panel_status = "completed"
        for action_id in action_set:
            value, status = action_progress(case_id, action_id)
            if value is None:
                panel_status = status
            else:
                panel_values.append(value)

        def paired(
            left: float | None,
            right: float | None,
            left_status: str,
            right_status: str,
        ) -> tuple[float | None, str]:
            if left is None or right is None:
                return None, f"left={left_status};right={right_status}"
            return left - right, "completed"

        controller_vs_uniform.append(
            (
                case_id,
                *paired(
                    selected_value,
                    uniform_value,
                    selected_status,
                    uniform_status,
                ),
            )
        )
        controller_vs_static.append(
            (
                case_id,
                *paired(
                    selected_value,
                    static_value,
                    selected_status,
                    static_status,
                ),
            )
        )
        controller_vs_ordered.append(
            (
                case_id,
                *paired(
                    selected_value,
                    ordered_value,
                    selected_status,
                    ordered_status,
                ),
            )
        )
        ordered_vs_static.append(
            (
                case_id,
                *paired(
                    ordered_value,
                    static_value,
                    ordered_status,
                    static_status,
                ),
            )
        )
        if (
            len(panel_values) == len(action_set)
            and static_value is not None
        ):
            raw_gap = max(panel_values) - static_value
            adjusted_gap = (
                0.0
                if raw_gap <= (replay_envelope or 0.0)
                else raw_gap
            )
            oracle_vs_static.append((case_id, raw_gap, "completed"))
            oracle_near_tie_vs_static.append(
                (case_id, adjusted_gap, "completed")
            )
        else:
            status = f"panel={panel_status};static={static_status}"
            oracle_vs_static.append((case_id, None, status))
            oracle_near_tie_vs_static.append((case_id, None, status))

    return {
        "fraction": fraction,
        "fraction_label": fraction_label,
        "derivative_validation": {
            "comparison_unit": "case-action",
            "relative_error_definition": (
                "abs(score-actual_slope) / "
                "max(abs(actual_slope), replay_envelope/distance, 1e-12)"
            ),
            "sign_zero_tolerance": "replay_envelope/distance",
            "expected_comparison_count": expected,
            "finite_comparison_count": finite_count,
            "failure_or_missing_count": expected - finite_count,
            "sign_concordant_count": sign_concordant,
            "sign_concordance_itd": concordance_itd,
            "sign_concordance_finite_only": concordance_finite,
            "median_relative_error": median_error,
            "p90_relative_error": p90_error,
            "locked_criterion": criterion,
            "locked_criterion_pass": locked_pass,
            "comparison_rows": comparisons,
        },
        "calibration_best_static": {
            "status": "pilot_retrospective_only",
            "complete_panel": static_complete,
            "action_id": static_action,
            "near_tied_actions": near_tied,
            "action_mean_progress": action_means,
            "tie_rule": "within-replay-envelope->single->lower-manifest-layer->action_id",
        },
        "preliminary_case_level_effects": {
            "controller_minus_uniform": _effect_summary(controller_vs_uniform),
            "controller_minus_calibration_best_static": _effect_summary(
                controller_vs_static
            ),
            "controller_minus_ordered_global_alpha": _effect_summary(
                controller_vs_ordered
            ),
            "ordered_global_alpha_minus_calibration_best_static": _effect_summary(
                ordered_vs_static
            ),
            "retrospective_oracle_minus_calibration_best_static": _effect_summary(
                oracle_vs_static
            ),
            "near_tie_adjusted_oracle_minus_calibration_best_static": _effect_summary(
                oracle_near_tie_vs_static
            ),
        },
    }


def analyze_pilot_run(run_directory: str | Path) -> dict[str, Any]:
    """Analyze one sanitized model-run directory without reading tensor artifacts."""

    root = Path(run_directory).expanduser().resolve()
    if not root.is_dir():
        raise PilotAnalysisError("run directory does not exist")
    manifest = _read_json_object(root / "manifest.json")
    summary = _read_json_object(root / "summary.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise PilotAnalysisError("manifest schema mismatch")
    if summary.get("schema_version") != SUMMARY_SCHEMA:
        raise PilotAnalysisError("summary schema mismatch")
    run_id = _nonempty_identifier(manifest.get("run_id"), name="run_id")

    streams = {
        name: _read_jsonl(root / name, run_id=run_id) for name in _STREAM_FILES
    }
    receipt_root = root / "action_receipts"
    if receipt_root.is_symlink() or not receipt_root.is_dir():
        raise PilotAnalysisError("action_receipts must be a regular directory")
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
    observed_hashes["summary.json"] = _sha256_file(root / "summary.json")

    (
        integrity_errors,
        features,
        actions,
        operational,
        ordered,
        replay,
    ) = _validate_run_artifacts(
        root,
        manifest,
        summary,
        streams,
        receipts,
        observed_hashes,
    )
    case_ids = tuple(manifest["selected_case_ids"])
    action_set = tuple(manifest["action_set"])
    fractions = tuple(float(value) for value in manifest["fractions"])
    replay = _replay_envelope(case_ids, replay)
    replay_value = replay["value"]
    diagnostics = [
        _diagnose_fraction(
            fraction=fraction,
            case_ids=case_ids,
            action_set=action_set,
            features=features,
            actions=actions,
            operational=operational,
            ordered=ordered,
            replay_envelope=replay_value,
        )
        for fraction in fractions
    ]
    passing = [
        item["fraction"]
        for item in diagnostics
        if item["derivative_validation"]["locked_criterion_pass"]
    ]
    expected_panel = {
        "case_count": len(case_ids),
        "fraction_count": len(fractions),
        "action_count": len(action_set),
        "feature_rows": len(case_ids) * len(fractions),
        "commitment_rows": len(case_ids) * len(fractions),
        "operational_rows": len(case_ids) * len(fractions) * len(action_set),
        "ordered_global_alpha_rows": len(case_ids) * len(fractions),
        "native_reference_rows": len(case_ids),
        "replay_rows": len(case_ids),
        "event_rows": len(case_ids),
        "total_outcome_rows": (
            len(case_ids)
            * (len(fractions) * (len(action_set) + 1) + 2)
        ),
    }
    observed_panel = {
        "feature_rows": len(features),
        "commitment_rows": len(actions),
        "operational_rows": len(operational),
        "ordered_global_alpha_rows": len(ordered),
        "native_reference_rows": sum(
            row.event == "mv1_reference_outcome"
            for row in streams["outcomes.jsonl"]
        ),
        "replay_rows": sum(
            row.event == "mv1_replay_outcome"
            for row in streams["outcomes.jsonl"]
        ),
        "event_rows": len(streams["events.jsonl"]),
        "total_outcome_rows": len(streams["outcomes.jsonl"]),
    }
    panel_complete = all(
        observed_panel[name] == expected_panel[name] for name in observed_panel
    )
    integrity_valid = not integrity_errors
    claim_status = "pilot_descriptive_only"
    analysis_status = (
        "descriptive_complete"
        if integrity_valid and panel_complete
        else "blocked_invalid_artifacts"
        if not integrity_valid
        else "blocked_incomplete_panel"
    )
    effect_bounds = {
        item["fraction_label"]: {
            name: {
                "mean": summary_item["mean"],
                "observed_case_range": summary_item["observed_case_range"],
                "planned_case_count": summary_item["planned_case_count"],
                "finite_pair_count": summary_item["finite_pair_count"],
            }
            for name, summary_item in item[
                "preliminary_case_level_effects"
            ].items()
        }
        for item in diagnostics
    }
    report = {
        "schema_version": ANALYSIS_SCHEMA,
        "claim_status": claim_status,
        "analysis_status": analysis_status,
        "run_id": run_id,
        "model_alias": summary.get("model_alias"),
        "analysis_policy": {
            "result_access": (
                "sanitized manifest/features/actions/outcomes/events/summary/"
                "action_receipts only"
            ),
            "raw_tensor_artifacts_opened": False,
            "inference_or_confidence_interval": False,
            "numeric_effect_label": (
                "preliminary observed pilot bounds; not expected method gain"
            ),
            "analysis_code_sha256": _sha256_file(Path(__file__)),
        },
        "artifact_validation": {
            "valid": integrity_valid,
            "error_codes": sorted(set(integrity_errors)),
            "schemas": {
                "manifest": manifest.get("schema_version"),
                "stream": STREAM_SCHEMA,
                "summary": summary.get("schema_version"),
                "receipt": RECEIPT_SCHEMA,
            },
            "observed_sha256": observed_hashes,
            "expected_counts": expected_panel,
            "observed_counts": observed_panel,
            "panel_complete": panel_complete,
            "commit_before_outcome": {
                "valid": integrity_valid
                and set(features) == set(actions)
                and len(receipts) == len(actions),
                "evidence": (
                    "feature/action hashes -> flush+fsync exclusive receipt -> "
                    "outcome commitment hash"
                ),
                "receipt_count": len(receipts),
                "limitation": (
                    "artifact chain proves the committed precursor required by "
                    "the runner contract; separate files carry no wall-clock timestamp"
                ),
            },
        },
        "failure_denominator": {
            "planned_case_count": len(case_ids),
            "summary_run_status": summary.get("run_status"),
            "summary_failure_count": summary.get("failure_count"),
            "summary_not_run_due_to_abort_count": summary.get(
                "not_run_due_to_abort_count"
            ),
            "missing rows are never dropped from locked criteria": True,
        },
        "replay_near_zero_envelope": replay,
        "fraction_diagnostics": diagnostics,
        "pilot_budget_lock_diagnostic": {
            "status": "pilot_candidate_only; C0 calibration lock not authorized",
            "passing_fractions": passing,
            "largest_passing_fraction": max(passing) if passing else None,
            "failure_if_none": "analytic_utility_calibration_failure",
        },
        "preliminary_expected_effect_bounds": {
            "label": (
                "observed three-case range and mean only; retrospective oracle is "
                "an opportunity ceiling, controller contrasts are not confirmatory"
            ),
            "by_fraction": effect_bounds,
            "non_additivity_warning": (
                "controller, oracle, and ordered-global-alpha contrasts must not "
                "be added into a total expected improvement"
            ),
        },
    }
    _assert_json_safe(report, location="analysis_output")
    json.dumps(report, allow_nan=False)
    return report


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
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
        description="Analyze one sanitized MV-1 three-case pilot model run."
    )
    parser.add_argument("run_directory", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional exclusive JSON output path; stdout is always sanitized.",
    )
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = analyze_pilot_run(args.run_directory)
        if args.output is not None:
            _write_exclusive_json(args.output, report)
        print(
            json.dumps(
                report,
                sort_keys=True,
                ensure_ascii=False,
                indent=2 if args.pretty else None,
                allow_nan=False,
            )
        )
        return 0 if report["analysis_status"] == "descriptive_complete" else 2
    except PilotAnalysisError as exc:
        print(
            json.dumps(
                {
                    "schema_version": ANALYSIS_SCHEMA,
                    "claim_status": "pilot_descriptive_only",
                    "analysis_status": "blocked_parse_or_schema_error",
                    "error": str(exc),
                },
                sort_keys=True,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
