"""Deterministic raw-free endpoint and round publication for ORBODE.

The live evaluator deliberately returns prompts, targets, token ids and token
predictions so locality preservation can be computed exactly.  Those values
belong only in ignored result roots.  This module validates the complete B100
shape, computes observation-only factual metrics, and emits a fresh payload
containing only numeric observations and cryptographic identities.

No controller, writer, model or evaluator code is imported here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


EVALUATION_SCHEMA = "orbode.raw-free-evaluation.v1"
ENDPOINT_SCHEMA = "orbode.raw-free-endpoint.v1"
ROUND_SCHEMA = "orbode.raw-free-round.v1"
MECHANISM_SCHEMA = "orbode.raw-free-mechanism-telemetry.v1"
PREAMBLE_SCHEMA = "orbode.raw-free-runtime-preamble.v1"
PRIMARY_ARM_ORDER = ("O", "QCL", "NQFIX", "ORBFH", "JAC")
DERIVED_ARM = "ORBHit"
EVALUATION_KINDS = (
    "rewrite_target_new",
    "rewrite_target_true",
    "rephrase_target_new",
    "rephrase_target_true",
    "locality_target_true",
)
ROWS_PER_REQUEST = {
    "rewrite_target_new": 1,
    "rewrite_target_true": 1,
    "rephrase_target_new": 2,
    "rephrase_target_true": 2,
    "locality_target_true": 10,
}
RAW_EVALUATOR_ROW_FIELDS = frozenset(
    {
        "case_id",
        "kind",
        "prompt_index",
        "prompt",
        "target",
        "target_token_ids",
        "nll",
        "token_predictions",
        "token_correct",
        "all_tokens_correct",
    }
)
PROHIBITED_PUBLIC_FIELDS = frozenset(
    {
        "prompt",
        "target",
        "target_token_ids",
        "token_predictions",
        "token_correct",
        "input_ids",
        "attention_mask",
        "labels",
        "prediction",
        "predictions",
        "token_ids",
    }
)
PUBLIC_ROW_BASE_FIELDS = frozenset(
    {
        "case_id",
        "request_sha256",
        "kind",
        "prompt_index",
        "nll",
        "all_tokens_correct",
        "target_token_count",
        "correct_token_count",
        "input_identity_sha256",
        "observation_identity_sha256",
    }
)
PUBLIC_ROW_ENTRY_FIELDS = frozenset(
    {
        "entry_nll",
        "nll_delta_from_entry",
        "entry_all_tokens_correct",
        "entry_observation_identity_sha256",
    }
)
PUBLIC_ROW_LOCALITY_FIELDS = frozenset(
    {
        "prediction_token_count",
        "prediction_preserved_count",
        "all_predictions_preserved",
    }
)
PRIMARY_TERMINAL_STATUSES = frozenset(
    {
        "TERMINAL_VALID",
        "ENTRY_ALREADY_HIT",
        "FLOW_STALLED_ZERO_ACTION",
        "FLOW_STALLED_TERMINAL_HIT",
        "FLOW_STALLED_AFTER_TRANSIENT_HIT",
        "FLOW_STALLED_TERMINAL_MISS",
        "HORIZON_SEMANTIC_HIT_OBSERVED",
        "HORIZON_TRANSIENT_HIT_TERMINAL_MISS",
        "HORIZON_SEMANTIC_MISS",
        "FIRST_HIT_AUDIT",
    }
)
DERIVED_TERMINAL_STATUSES = frozenset(
    {"FIRST_HIT", "HORIZON_SEMANTIC_MISS", "ENTRY_ALREADY_HIT", "ORBHit_WITHHELD_TECHNICAL"}
)
SEMANTIC_FIELDS = frozenset(
    {
        "all_strict",
        "strict_event_count",
        "event_count",
        "request_strict",
        "request_strict_count",
        "target_logit_mean",
        "target_logit_min",
        "maximum_other_logit_mean",
        "strict_tie_count",
    }
)
STEP_FIELDS = frozenset(
    {
        "arm",
        "sweep",
        "layer",
        "visit_ordinal",
        "built_state_version",
        "resulting_state_version",
        "residual_denominator",
        "command_reference_state_version",
        "build_identity",
        "residual_sha256",
        "keys_sha256",
        "solver_identity",
        "factor_rank",
        "coefficient_u",
        "euler_alpha",
        "g",
        "r",
        "per_request_g",
        "per_request_r",
        "potential_before",
        "potential_after",
        "predicted_reduction",
        "actual_reduction",
        "discretization_defect",
        "entry_sublevel_violation",
        "residual_norm_mean_before",
        "residual_norm_mean_after",
        "response_norm_mean",
        "command_norm_mean",
        "writer_command_norm_mean",
        "predicted_transition_norm_mean",
        "actual_transition_norm_mean",
        "realization_error_norm_mean",
        "model_error_norm_mean",
        "per_request_D_R0",
        "per_request_D_model_R0",
        "per_request_q_res",
        "per_request_writer_command_R0_squared",
        "per_request_response_orthogonal_R0_squared",
        "per_request_zero_command_cross_response_R0_squared",
        "per_request_actual_potential_change",
        "per_request_actual_potential_worsened",
        "per_request_metric_status",
        "direction_frobenius",
        "direction_frobenius_squared",
        "applied_update_frobenius",
        "applied_update_frobenius_squared",
        "resolution_stable_path_increment_frobenius_squared",
        "action_geometry_status",
        "native_creg_action_status",
        "semantic_all_strict",
        "semantic_request_count",
        "semantic_strict_event_count",
        "semantic_event_count",
        "semantic_request_strict_count",
        "semantic_tie_count",
        "virtual_state_identity_sha256",
    }
)
STEP_OPTIONAL_NUMERIC_FIELDS = frozenset(
    {
        "g",
        "r",
        "predicted_reduction",
        "response_norm_mean",
        "predicted_transition_norm_mean",
        "model_error_norm_mean",
    }
)
STEP_REQUEST_OPTIONAL_NUMERIC_FIELDS = (
    "per_request_g",
    "per_request_r",
    "per_request_D_R0",
    "per_request_D_model_R0",
    "per_request_q_res",
    "per_request_writer_command_R0_squared",
    "per_request_response_orthogonal_R0_squared",
    "per_request_zero_command_cross_response_R0_squared",
    "per_request_actual_potential_change",
)
STEP_REQUEST_OPTIONAL_BOOLEAN_FIELDS = ("per_request_actual_potential_worsened",)
STEP_REQUEST_STATUS_VALUES = frozenset(
    {
        "DEFINED",
        "DEFINED_D_R0_ONLY_NO_RESPONSE",
        "SKIPPED_ZERO_COMMAND",
        "ZERO_COMMAND_RESPONSE_OBSERVED",
        "ZERO_COMMAND_WITHOUT_RESPONSE",
        "ZERO_ANCHOR_EXCLUDED_FROM_QRES",
        "ZERO_GLOBAL_ACTION_WITHOUT_RESPONSE",
        "ZERO_GLOBAL_ACTION_RESPONSE_OBSERVED",
        "ZERO_COMMAND_CROSS_RESPONSE_OBSERVED",
        "ZERO_COMMAND_NO_CROSS_RESPONSE",
    }
)


class ArtifactBoundary(ValueError):
    """The raw evaluator or public artifact failed an integrity boundary."""


def canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactBoundary("artifact value is not finite canonical JSON") from exc


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactBoundary(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ArtifactBoundary(f"{label} is non-finite")
    return result


def _count(value: object, label: str) -> int:
    if not _is_int(value) or int(value) < 0:
        raise ArtifactBoundary(f"{label} is not a nonnegative integer")
    return int(value)


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactBoundary(f"{label} is not a lowercase SHA256")
    return value


def _with_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["identity_sha256"] = canonical_hash(result)
    return result


def _assert_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    identity = body.pop("identity_sha256", None)
    if identity != canonical_hash(body):
        raise ArtifactBoundary("artifact identity differs")
    return body


def _assert_raw_free(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in PROHIBITED_PUBLIC_FIELDS:
                raise ArtifactBoundary(f"raw evaluator field escaped publication: {key}")
            _assert_raw_free(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_raw_free(item)


def _canonical_raw_free_clone(value: object, label: str) -> Any:
    """Clone a JSON fact tree only after recursively excluding raw evaluator keys.

    ``allow_nan=False`` in :func:`canonical_json` makes this a recursive finite
    check as well.  The round publication therefore retains the complete
    numeric/hash mechanism ledger without retaining model inputs or literal
    generations.
    """

    _assert_raw_free(value)
    try:
        return json.loads(canonical_json(value))
    except ArtifactBoundary:
        raise
    except Exception as exc:  # pragma: no cover - defensive JSON boundary
        raise ArtifactBoundary(f"{label} is not canonical raw-free JSON") from exc


def _scrub_exception_messages(value: object) -> Any:
    """Keep exception provenance while excluding arbitrary message text."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if key in PROHIBITED_PUBLIC_FIELDS:
                raise ArtifactBoundary(f"raw evaluator field escaped publication: {key}")
            if key == "exception":
                if not isinstance(item, str):
                    raise ArtifactBoundary("runtime preamble exception is not text")
                result["exception_message_sha256"] = hashlib.sha256(
                    item.encode("utf-8")
                ).hexdigest()
                result["exception_message_redacted"] = True
            else:
                result[key] = _scrub_exception_messages(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_scrub_exception_messages(item) for item in value]
    return value


def _validate_semantic_payload(
    value: object, *, request_count: int, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != SEMANTIC_FIELDS:
        raise ArtifactBoundary(f"{label} semantic schema differs")
    if not isinstance(value.get("all_strict"), bool):
        raise ArtifactBoundary(f"{label} all_strict differs")
    request_strict = value.get("request_strict")
    if (
        not isinstance(request_strict, (list, tuple))
        or len(request_strict) != request_count
        or any(not isinstance(item, bool) for item in request_strict)
    ):
        raise ArtifactBoundary(f"{label} request-strict denominator differs")
    strict_request_count = _count(
        value.get("request_strict_count"), f"{label}.request_strict_count"
    )
    strict_event_count = _count(
        value.get("strict_event_count"), f"{label}.strict_event_count"
    )
    event_count = _count(value.get("event_count"), f"{label}.event_count")
    tie_count = _count(value.get("strict_tie_count"), f"{label}.strict_tie_count")
    if (
        event_count < request_count
        or strict_event_count > event_count
        or tie_count > event_count
        or strict_request_count != sum(request_strict)
        or bool(value["all_strict"]) is not all(request_strict)
    ):
        raise ArtifactBoundary(f"{label} semantic counts differ")
    for field in (
        "target_logit_mean",
        "target_logit_min",
        "maximum_other_logit_mean",
    ):
        _finite(value.get(field), f"{label}.{field}")
    return _canonical_raw_free_clone(value, label)


def _validate_step_payload(
    value: object,
    *,
    expected_arm: str,
    request_count: int,
    step_index: int,
) -> dict[str, Any]:
    label = f"mechanism step[{step_index}]"
    if not isinstance(value, Mapping) or set(value) != STEP_FIELDS:
        raise ArtifactBoundary(f"{label} schema differs")
    if value.get("arm") != expected_arm:
        raise ArtifactBoundary(f"{label} arm differs")
    for field in (
        "sweep",
        "visit_ordinal",
        "built_state_version",
        "resulting_state_version",
        "command_reference_state_version",
        "semantic_strict_event_count",
        "semantic_event_count",
        "semantic_request_strict_count",
        "semantic_tie_count",
    ):
        _count(value.get(field), f"{label}.{field}")
    if (
        not _is_int(value.get("layer"))
        or int(value["layer"]) not in (4, 5, 6, 7, 8)
        or not _is_int(value.get("residual_denominator"))
        or int(value["residual_denominator"]) <= 0
        or not _is_int(value.get("factor_rank"))
        or int(value["factor_rank"]) <= 0
        or value.get("semantic_request_count") != request_count
        or int(value["semantic_request_strict_count"]) > request_count
        or int(value["semantic_strict_event_count"]) > int(value["semantic_event_count"])
        or not isinstance(value.get("semantic_all_strict"), bool)
        or bool(value["semantic_all_strict"])
        is not (int(value["semantic_request_strict_count"]) == request_count)
    ):
        raise ArtifactBoundary(f"{label} count/state contract differs")
    for field in ("build_identity", "residual_sha256", "keys_sha256", "solver_identity"):
        _sha256(value.get(field), f"{label}.{field}")
    _sha256(value.get("virtual_state_identity_sha256"), f"{label}.virtual_state_identity")
    for field in STEP_FIELDS - STEP_OPTIONAL_NUMERIC_FIELDS - {
        "arm",
        "sweep",
        "layer",
        "visit_ordinal",
        "built_state_version",
        "resulting_state_version",
        "residual_denominator",
        "command_reference_state_version",
        "build_identity",
        "residual_sha256",
        "keys_sha256",
        "solver_identity",
        "factor_rank",
        "per_request_g",
        "per_request_r",
        "per_request_D_R0",
        "per_request_D_model_R0",
        "per_request_q_res",
        "per_request_writer_command_R0_squared",
        "per_request_response_orthogonal_R0_squared",
        "per_request_zero_command_cross_response_R0_squared",
        "per_request_actual_potential_change",
        "per_request_actual_potential_worsened",
        "per_request_metric_status",
        "action_geometry_status",
        "native_creg_action_status",
        "semantic_all_strict",
        "semantic_request_count",
        "semantic_strict_event_count",
        "semantic_event_count",
        "semantic_request_strict_count",
        "semantic_tie_count",
        "virtual_state_identity_sha256",
    }:
        _finite(value.get(field), f"{label}.{field}")
    for field in STEP_OPTIONAL_NUMERIC_FIELDS:
        if value.get(field) is not None:
            _finite(value[field], f"{label}.{field}")
    coefficient = _finite(value.get("coefficient_u"), f"{label}.coefficient_u")
    alpha = _finite(value.get("euler_alpha"), f"{label}.euler_alpha")
    if not 0.0 <= coefficient <= 1.0 or alpha < 0.0:
        raise ArtifactBoundary(f"{label} coefficient/step differs")
    for field in STEP_REQUEST_OPTIONAL_NUMERIC_FIELDS:
        items = value.get(field)
        if not isinstance(items, (list, tuple)) or len(items) != request_count:
            raise ArtifactBoundary(f"{label}.{field} denominator differs")
        for request_index, item in enumerate(items):
            if item is not None:
                _finite(item, f"{label}.{field}[{request_index}]")
    for field in STEP_REQUEST_OPTIONAL_BOOLEAN_FIELDS:
        items = value.get(field)
        if (
            not isinstance(items, (list, tuple))
            or len(items) != request_count
            or any(item is not None and not isinstance(item, bool) for item in items)
        ):
            raise ArtifactBoundary(f"{label}.{field} denominator/type differs")
    metric_status = value.get("per_request_metric_status")
    if (
        not isinstance(metric_status, (list, tuple))
        or len(metric_status) != request_count
        or any(item not in STEP_REQUEST_STATUS_VALUES for item in metric_status)
    ):
        raise ArtifactBoundary(f"{label}.per_request_metric_status differs")
    if (
        value.get("action_geometry_status") != "FROBENIUS_ONLY_CREG_NOT_BOUND"
        or value.get("native_creg_action_status") != "TELEMETRY_WITHHELD"
    ):
        raise ArtifactBoundary(f"{label} action geometry status differs")
    return _canonical_raw_free_clone(value, label)


def _reduce_mechanism_telemetry(
    value: object, *, expected_arm: str, request_count: int
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactBoundary("mechanism telemetry is not an object")
    source_schema = value.get("schema")
    if source_schema == "orbode.official-bypass.v1":
        for field in ("entry_semantic", "terminal_semantic"):
            _validate_semantic_payload(
                value.get(field), request_count=request_count, label=f"Official.{field}"
            )
        if value.get("native_compute_z_bypassed_with_shared_fixed_z") is not True:
            raise ArtifactBoundary("Official shared fixed-z bypass fact differs")
        for field in (
            "dynamic_layer_visit_count",
            "factor_build_count",
            "physical_write_count",
            "history_append_count",
        ):
            _count(value.get(field), f"Official telemetry.{field}")
        for field in (
            "terminal_net_frobenius",
            "terminal_net_frobenius_squared",
            "sum_step_action_frobenius_squared",
            "resolution_stable_path_frobenius_squared",
        ):
            if _finite(value.get(field), f"Official telemetry.{field}") < 0.0:
                raise ArtifactBoundary(f"Official telemetry.{field} is negative")
        if value.get("native_creg_action_status") != "TELEMETRY_WITHHELD":
            raise ArtifactBoundary("Official native Creg status differs")
        normalized = _canonical_raw_free_clone(value, "Official mechanism telemetry")
    elif source_schema == "orbode.arm-telemetry.v1":
        if value.get("arm") != expected_arm:
            raise ArtifactBoundary("dynamic mechanism telemetry arm differs")
        entry = _validate_semantic_payload(
            value.get("entry_semantic"), request_count=request_count, label="dynamic.entry"
        )
        terminal = _validate_semantic_payload(
            value.get("terminal_semantic"),
            request_count=request_count,
            label="dynamic.terminal",
        )
        steps = value.get("steps")
        if not isinstance(steps, list) or value.get("step_count") != len(steps):
            raise ArtifactBoundary("dynamic mechanism step count differs")
        normalized_steps = [
            _validate_step_payload(
                step,
                expected_arm=expected_arm,
                request_count=request_count,
                step_index=index,
            )
            for index, step in enumerate(steps)
        ]
        for field in (
            "zero_action_visit_count",
            "nonzero_action_visit_count",
            "factor_build_count",
            "key_capture_count",
            "terminal_capture_count",
            "jvp_call_count",
            "materialization_count",
            "physical_write_count",
            "history_append_count",
            "anchor_active_request_count",
            "anchor_zero_request_count",
            "anchor_zero_semantic_miss_count",
        ):
            _count(value.get(field), f"dynamic telemetry.{field}")
        if (
            int(value["factor_build_count"]) != len(steps)
            or int(value["zero_action_visit_count"])
            + int(value["nonzero_action_visit_count"])
            != len(steps)
            or int(value["anchor_active_request_count"])
            + int(value["anchor_zero_request_count"])
            != request_count
            or int(value["anchor_zero_semantic_miss_count"])
            > int(value["anchor_zero_request_count"])
            or value.get("anchor_zero_semantic_miss_policy")
            != "NONBLOCKING_SCIENTIFIC_OBSERVATION"
        ):
            raise ArtifactBoundary("dynamic mechanism visit/build reconciliation differs")
        for field in (
            "sum_step_action_frobenius_squared",
            "resolution_stable_path_frobenius_squared",
            "terminal_net_frobenius",
            "terminal_net_frobenius_squared",
        ):
            if _finite(value.get(field), f"dynamic telemetry.{field}") < 0.0:
                raise ArtifactBoundary(f"dynamic telemetry.{field} is negative")
        if value.get("terminal_status") not in PRIMARY_TERMINAL_STATUSES:
            raise ArtifactBoundary("dynamic mechanism terminal status differs")
        first_hit = value.get("first_hit")
        if first_hit is not None:
            if not isinstance(first_hit, Mapping) or set(first_hit) != {
                "sweep",
                "layer",
                "state_version",
                "prefix_length",
            }:
                raise ArtifactBoundary("dynamic first-hit schema differs")
            for field in first_hit:
                _count(first_hit[field], f"dynamic first_hit.{field}")
        normalized = _canonical_raw_free_clone(value, "dynamic mechanism telemetry")
        normalized["entry_semantic"] = entry
        normalized["terminal_semantic"] = terminal
        normalized["steps"] = normalized_steps
    else:
        raise ArtifactBoundary("mechanism telemetry schema differs")
    payload = {
        "schema": MECHANISM_SCHEMA,
        "arm": expected_arm,
        "request_count": request_count,
        "source_schema": source_schema,
        "source_telemetry_identity_sha256": canonical_hash(value),
        "telemetry": normalized,
        "literal_prompt_target_token_prediction_publication_count": 0,
    }
    return _with_identity(payload)


def _validate_mechanism_publication(
    value: object, *, expected_arm: str, request_count: int
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactBoundary("published mechanism telemetry is not an object")
    _assert_raw_free(value)
    _assert_identity(value)
    if (
        value.get("schema") != MECHANISM_SCHEMA
        or value.get("arm") != expected_arm
        or value.get("request_count") != request_count
        or value.get("literal_prompt_target_token_prediction_publication_count") != 0
        or canonical_hash(value.get("telemetry"))
        != value.get("source_telemetry_identity_sha256")
    ):
        raise ArtifactBoundary("published mechanism telemetry binding differs")
    _sha256(value.get("source_telemetry_identity_sha256"), "source telemetry identity")
    rebuilt = _reduce_mechanism_telemetry(
        value.get("telemetry"), expected_arm=expected_arm, request_count=request_count
    )
    if rebuilt.get("telemetry") != value.get("telemetry"):
        raise ArtifactBoundary("published mechanism telemetry facts differ")
    return {
        "status": "RAW_FREE_MECHANISM_PASS",
        "identity_sha256": value["identity_sha256"],
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ArtifactBoundary("percentile input differs")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _nll_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [_finite(row["nll"], "published row nll") for row in rows]
    correct = sum(int(row["all_tokens_correct"] is True) for row in rows)
    return {
        "row_count": len(rows),
        "nll_mean": math.fsum(values) / len(values),
        "nll_median": _percentile(values, 0.5),
        "nll_p90": _percentile(values, 0.9),
        "nll_max": max(values),
        "all_tokens_correct_count": correct,
        "all_tokens_correct_rate": correct / len(rows),
    }


def _expected_rows(
    case_ids: Sequence[int], kind: str, per_request: int
) -> tuple[tuple[int, str, int], ...]:
    return tuple(
        (case_id, kind, prompt_index)
        for case_id in case_ids
        for prompt_index in range(per_request)
    )


def _request_bindings(
    case_ids: Sequence[int], request_sha256: Sequence[str]
) -> tuple[tuple[int, ...], dict[int, str]]:
    normalized_cases = tuple(case_ids)
    if (
        not normalized_cases
        or any(not _is_int(value) or int(value) < 0 for value in normalized_cases)
        or len(set(normalized_cases)) != len(normalized_cases)
    ):
        raise ArtifactBoundary("expected case identity/order differs")
    if len(request_sha256) != len(normalized_cases):
        raise ArtifactBoundary("request identity denominator differs")
    normalized_requests = tuple(
        _sha256(value, f"request_sha256[{index}]")
        for index, value in enumerate(request_sha256)
    )
    if len(set(normalized_requests)) != len(normalized_requests):
        raise ArtifactBoundary("request SHA identities are not unique")
    return normalized_cases, dict(zip(normalized_cases, normalized_requests, strict=True))


def _validate_raw_row(
    value: object,
    *,
    expected_key: tuple[int, str, int],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != RAW_EVALUATOR_ROW_FIELDS:
        raise ArtifactBoundary("raw evaluator row schema differs")
    case_id, kind, prompt_index = expected_key
    if (
        value.get("case_id") != case_id
        or value.get("kind") != kind
        or value.get("prompt_index") != prompt_index
        or not isinstance(value.get("prompt"), str)
        or not value.get("prompt")
        or not isinstance(value.get("target"), str)
        or not value.get("target")
        or not isinstance(value.get("all_tokens_correct"), bool)
    ):
        raise ArtifactBoundary("raw evaluator row identity differs")
    nll = _finite(value.get("nll"), "raw evaluator nll")
    target_ids = value.get("target_token_ids")
    predictions = value.get("token_predictions")
    correctness = value.get("token_correct")
    if (
        not isinstance(target_ids, list)
        or not target_ids
        or not isinstance(predictions, list)
        or not isinstance(correctness, list)
        or len(target_ids) != len(predictions)
        or len(target_ids) != len(correctness)
        or any(not _is_int(item) or item < 0 for item in target_ids)
        or any(not _is_int(item) or item < 0 for item in predictions)
        or any(not isinstance(item, bool) for item in correctness)
    ):
        raise ArtifactBoundary("raw evaluator token observation differs")
    derived = [left == right for left, right in zip(predictions, target_ids, strict=True)]
    if correctness != derived or bool(value["all_tokens_correct"]) != all(derived):
        raise ArtifactBoundary("raw evaluator correctness is inconsistent")
    input_identity = canonical_hash(
        {
            "case_id": case_id,
            "kind": kind,
            "prompt_index": prompt_index,
            "prompt": value["prompt"],
            "target": value["target"],
            "target_token_ids": target_ids,
        }
    )
    return {
        "raw": dict(value),
        "case_id": case_id,
        "kind": kind,
        "prompt_index": prompt_index,
        "nll": nll,
        "all_tokens_correct": bool(value["all_tokens_correct"]),
        "target_token_count": len(target_ids),
        "correct_token_count": sum(correctness),
        "input_identity_sha256": input_identity,
        "observation_identity_sha256": canonical_hash(value),
    }


def _validate_raw_evaluation(
    evaluation: object,
    *,
    case_ids: tuple[int, ...],
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(evaluation, Mapping) or set(evaluation) != set(EVALUATION_KINDS):
        raise ArtifactBoundary("raw evaluator kind inventory differs")
    result: dict[str, list[dict[str, Any]]] = {}
    for kind in EVALUATION_KINDS:
        raw_rows = evaluation[kind]
        if not isinstance(raw_rows, list):
            raise ArtifactBoundary(f"raw evaluator rows are not a list: {kind}")
        expected = _expected_rows(case_ids, kind, ROWS_PER_REQUEST[kind])
        if len(raw_rows) != len(expected):
            raise ArtifactBoundary(f"raw evaluator row count differs: {kind}")
        result[kind] = [
            _validate_raw_row(row, expected_key=key)
            for row, key in zip(raw_rows, expected, strict=True)
        ]
    for case_position, _case_id in enumerate(case_ids):
        rewrite_new = result["rewrite_target_new"][case_position]["raw"]
        rewrite_true = result["rewrite_target_true"][case_position]["raw"]
        if rewrite_new["prompt"] != rewrite_true["prompt"]:
            raise ArtifactBoundary("rewrite target-new/true prompt join differs")
        new_reference = (rewrite_new["target"], rewrite_new["target_token_ids"])
        true_reference = (rewrite_true["target"], rewrite_true["target_token_ids"])
        rephrase_start = case_position * ROWS_PER_REQUEST["rephrase_target_new"]
        locality_start = case_position * ROWS_PER_REQUEST["locality_target_true"]
        for prompt_index in range(ROWS_PER_REQUEST["rephrase_target_new"]):
            new_row = result["rephrase_target_new"][rephrase_start + prompt_index]["raw"]
            true_row = result["rephrase_target_true"][rephrase_start + prompt_index]["raw"]
            if new_row["prompt"] != true_row["prompt"]:
                raise ArtifactBoundary("rephrase target-new/true prompt join differs")
            if (new_row["target"], new_row["target_token_ids"]) != new_reference:
                raise ArtifactBoundary("rewrite/rephrase target-new identity differs")
            if (true_row["target"], true_row["target_token_ids"]) != true_reference:
                raise ArtifactBoundary("rewrite/rephrase target-true identity differs")
        for prompt_index in range(ROWS_PER_REQUEST["locality_target_true"]):
            locality_row = result["locality_target_true"][locality_start + prompt_index]["raw"]
            if (locality_row["target"], locality_row["target_token_ids"]) != true_reference:
                raise ArtifactBoundary("rewrite/locality target-true identity differs")
    return result


def _preference_payload(
    rows_by_kind: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    prefix: str,
    case_ids: Sequence[int],
) -> dict[str, Any]:
    new_rows = rows_by_kind[f"{prefix}_target_new"]
    true_rows = rows_by_kind[f"{prefix}_target_true"]
    if len(new_rows) != len(true_rows):
        raise ArtifactBoundary(f"{prefix} target-new/true denominator differs")
    bits: list[bool] = []
    by_case: dict[int, list[bool]] = {case_id: [] for case_id in case_ids}
    for new, true in zip(new_rows, true_rows, strict=True):
        key_new = (new["case_id"], new["prompt_index"])
        key_true = (true["case_id"], true["prompt_index"])
        if key_new != key_true:
            raise ArtifactBoundary(f"{prefix} target-new/true row join differs")
        success = float(new["nll"]) < float(true["nll"])
        bits.append(success)
        by_case[int(new["case_id"])].append(success)
    strict = [all(by_case[case_id]) for case_id in case_ids]
    return {
        "prompt_denominator": len(bits),
        "prompt_success_count": sum(bits),
        "prompt_success_rate": sum(bits) / len(bits),
        "strict_request_denominator": len(strict),
        "strict_request_success_count": sum(strict),
        "strict_request_success_rate": sum(strict) / len(strict),
        "tie_is_failure": True,
        "bit_vector_sha256": canonical_hash(bits),
        "strict_bit_vector_sha256": canonical_hash(strict),
    }


def reduce_evaluation_payload(
    evaluation: Mapping[str, Any],
    *,
    case_ids: Sequence[int],
    request_sha256: Sequence[str],
    request_order_sha256: str,
    entry_evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one B100 evaluation and remove every literal evaluator field."""

    cases, request_by_case = _request_bindings(case_ids, request_sha256)
    order_sha = _sha256(request_order_sha256, "request_order_sha256")
    if canonical_hash(list(request_sha256)) != order_sha:
        raise ArtifactBoundary("request order SHA does not bind the provided request identities")
    current = _validate_raw_evaluation(evaluation, case_ids=cases)
    entry = (
        None
        if entry_evaluation is None
        else _validate_raw_evaluation(entry_evaluation, case_ids=cases)
    )
    public_rows: list[dict[str, Any]] = []
    for kind in EVALUATION_KINDS:
        for index, observed in enumerate(current[kind]):
            row = {
                "case_id": observed["case_id"],
                "request_sha256": request_by_case[int(observed["case_id"])],
                "kind": observed["kind"],
                "prompt_index": observed["prompt_index"],
                "nll": observed["nll"],
                "all_tokens_correct": observed["all_tokens_correct"],
                "target_token_count": observed["target_token_count"],
                "correct_token_count": observed["correct_token_count"],
                "input_identity_sha256": observed["input_identity_sha256"],
                "observation_identity_sha256": observed["observation_identity_sha256"],
            }
            if entry is not None:
                prior = entry[kind][index]
                if prior["input_identity_sha256"] != observed["input_identity_sha256"]:
                    raise ArtifactBoundary("entry/current evaluator prompt-target identity differs")
                row.update(
                    {
                        "entry_nll": prior["nll"],
                        "nll_delta_from_entry": observed["nll"] - prior["nll"],
                        "entry_all_tokens_correct": prior["all_tokens_correct"],
                        "entry_observation_identity_sha256": prior[
                            "observation_identity_sha256"
                        ],
                    }
                )
                if kind == "locality_target_true":
                    left = prior["raw"]["token_predictions"]
                    right = observed["raw"]["token_predictions"]
                    if len(left) != len(right):
                        raise ArtifactBoundary("locality token preservation denominator differs")
                    preserved = sum(a == b for a, b in zip(left, right, strict=True))
                    row.update(
                        {
                            "prediction_token_count": len(left),
                            "prediction_preserved_count": preserved,
                            "all_predictions_preserved": preserved == len(left),
                        }
                    )
            public_rows.append(row)

    public_by_kind = {
        kind: [row for row in public_rows if row["kind"] == kind]
        for kind in EVALUATION_KINDS
    }
    locality_rows = public_by_kind["locality_target_true"]
    locality: dict[str, Any] = {
        "target_true_all_tokens_correct_count": sum(
            int(row["all_tokens_correct"]) for row in locality_rows
        ),
        "target_true_all_tokens_correct_denominator": len(locality_rows),
    }
    locality["target_true_all_tokens_correct_rate"] = (
        locality["target_true_all_tokens_correct_count"]
        / locality["target_true_all_tokens_correct_denominator"]
    )
    if entry is not None:
        preservation_numerator = sum(
            int(row["prediction_preserved_count"]) for row in locality_rows
        )
        preservation_denominator = sum(
            int(row["prediction_token_count"]) for row in locality_rows
        )
        if preservation_denominator <= 0:
            raise ArtifactBoundary("locality prediction denominator is empty")
        locality.update(
            {
                "prediction_preservation_numerator": preservation_numerator,
                "prediction_preservation_denominator": preservation_denominator,
                "prediction_preservation_rate": (
                    preservation_numerator / preservation_denominator
                ),
                "all_prompt_predictions_preserved_count": sum(
                    int(row["all_predictions_preserved"]) for row in locality_rows
                ),
            }
        )
    payload = {
        "schema": EVALUATION_SCHEMA,
        "request_count": len(cases),
        "request_order_sha256": order_sha,
        "source_evaluation_identity_sha256": canonical_hash(evaluation),
        "entry_evaluation_identity_sha256": (
            None if entry_evaluation is None else canonical_hash(entry_evaluation)
        ),
        "entry_comparison": entry is not None,
        "row_count": len(public_rows),
        "expected_rows_per_request": dict(ROWS_PER_REQUEST),
        "rows": public_rows,
        "kind_summaries": {
            kind: _nll_summary(rows) for kind, rows in public_by_kind.items()
        },
        "preference": {
            "rewrite": _preference_payload(public_by_kind, prefix="rewrite", case_ids=cases),
            "rephrase": _preference_payload(public_by_kind, prefix="rephrase", case_ids=cases),
        },
        "locality": locality,
        "literal_prompt_target_token_prediction_publication_count": 0,
        "controller_or_writer_influence_count": 0,
    }
    result = _with_identity(payload)
    validate_evaluation_publication(
        result,
        case_ids=cases,
        request_sha256=tuple(request_sha256),
        request_order_sha256=order_sha,
    )
    return result


def _published_rows_by_kind(
    payload: Mapping[str, Any], *, case_ids: tuple[int, ...], request_by_case: Mapping[int, str]
) -> dict[str, list[Mapping[str, Any]]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ArtifactBoundary("published evaluation rows are absent")
    expected_total = len(case_ids) * sum(ROWS_PER_REQUEST.values())
    if len(rows) != expected_total or payload.get("row_count") != expected_total:
        raise ArtifactBoundary("published evaluation row denominator differs")
    result: dict[str, list[Mapping[str, Any]]] = {kind: [] for kind in EVALUATION_KINDS}
    offset = 0
    for kind in EVALUATION_KINDS:
        expected = _expected_rows(case_ids, kind, ROWS_PER_REQUEST[kind])
        for case_id, expected_kind, prompt_index in expected:
            row = rows[offset]
            offset += 1
            expected_fields = set(PUBLIC_ROW_BASE_FIELDS)
            if payload.get("entry_comparison") is True:
                expected_fields.update(PUBLIC_ROW_ENTRY_FIELDS)
                if kind == "locality_target_true":
                    expected_fields.update(PUBLIC_ROW_LOCALITY_FIELDS)
            if (
                not isinstance(row, Mapping)
                or set(row) != expected_fields
                or row.get("case_id") != case_id
                or row.get("kind") != expected_kind
                or row.get("prompt_index") != prompt_index
                or row.get("request_sha256") != request_by_case[case_id]
                or not isinstance(row.get("all_tokens_correct"), bool)
                or not _is_int(row.get("target_token_count"))
                or int(row["target_token_count"]) <= 0
                or not _is_int(row.get("correct_token_count"))
                or not 0 <= int(row["correct_token_count"]) <= int(row["target_token_count"])
                or bool(row["all_tokens_correct"])
                is not (int(row["correct_token_count"]) == int(row["target_token_count"]))
            ):
                raise ArtifactBoundary("published evaluation row shape/order differs")
            _finite(row.get("nll"), "published row nll")
            _sha256(row.get("input_identity_sha256"), "published input identity")
            _sha256(row.get("observation_identity_sha256"), "published observation identity")
            if payload.get("entry_comparison") is True:
                _finite(row.get("entry_nll"), "published entry nll")
                delta = _finite(row.get("nll_delta_from_entry"), "published nll delta")
                if not math.isclose(
                    delta,
                    float(row["nll"]) - float(row["entry_nll"]),
                    rel_tol=0.0,
                    abs_tol=0.0,
                ):
                    raise ArtifactBoundary("published entry/current nll delta differs")
                if not isinstance(row.get("entry_all_tokens_correct"), bool):
                    raise ArtifactBoundary("published entry correctness differs")
                _sha256(
                    row.get("entry_observation_identity_sha256"),
                    "published entry observation identity",
                )
                if kind == "locality_target_true":
                    denominator = _count(
                        row.get("prediction_token_count"), "locality prediction token count"
                    )
                    numerator = _count(
                        row.get("prediction_preserved_count"),
                        "locality prediction preserved count",
                    )
                    if (
                        denominator <= 0
                        or numerator > denominator
                        or row.get("all_predictions_preserved") is not (numerator == denominator)
                    ):
                        raise ArtifactBoundary("published locality preservation row differs")
            result[kind].append(row)
    return result


def validate_evaluation_publication(
    payload: Mapping[str, Any],
    *,
    case_ids: Sequence[int],
    request_sha256: Sequence[str],
    request_order_sha256: str,
) -> dict[str, Any]:
    """Validate one already reduced evaluator artifact without raw inputs."""

    if not isinstance(payload, Mapping):
        raise ArtifactBoundary("published evaluation is not an object")
    _assert_raw_free(payload)
    _assert_identity(payload)
    cases, request_by_case = _request_bindings(case_ids, request_sha256)
    order_sha = _sha256(request_order_sha256, "request_order_sha256")
    if canonical_hash(list(request_sha256)) != order_sha:
        raise ArtifactBoundary("published request order binding differs")
    if (
        payload.get("schema") != EVALUATION_SCHEMA
        or payload.get("request_count") != len(cases)
        or payload.get("request_order_sha256") != order_sha
        or payload.get("expected_rows_per_request") != ROWS_PER_REQUEST
        or payload.get("literal_prompt_target_token_prediction_publication_count") != 0
        or payload.get("controller_or_writer_influence_count") != 0
        or not isinstance(payload.get("entry_comparison"), bool)
    ):
        raise ArtifactBoundary("published evaluation contract differs")
    _sha256(
        payload.get("source_evaluation_identity_sha256"),
        "source evaluation identity",
    )
    entry_identity = payload.get("entry_evaluation_identity_sha256")
    if (payload["entry_comparison"] is True) != (entry_identity is not None):
        raise ArtifactBoundary("published entry comparison identity differs")
    if entry_identity is not None:
        _sha256(entry_identity, "entry evaluation identity")
    rows = _published_rows_by_kind(payload, case_ids=cases, request_by_case=request_by_case)
    expected_summaries = {kind: _nll_summary(values) for kind, values in rows.items()}
    if payload.get("kind_summaries") != expected_summaries:
        raise ArtifactBoundary("published NLL summaries differ")
    expected_preference = {
        "rewrite": _preference_payload(rows, prefix="rewrite", case_ids=cases),
        "rephrase": _preference_payload(rows, prefix="rephrase", case_ids=cases),
    }
    if payload.get("preference") != expected_preference:
        raise ArtifactBoundary("published preference summaries differ")
    locality_rows = rows["locality_target_true"]
    expected_locality: dict[str, Any] = {
        "target_true_all_tokens_correct_count": sum(
            int(row["all_tokens_correct"]) for row in locality_rows
        ),
        "target_true_all_tokens_correct_denominator": len(locality_rows),
    }
    expected_locality["target_true_all_tokens_correct_rate"] = (
        expected_locality["target_true_all_tokens_correct_count"] / len(locality_rows)
    )
    if payload["entry_comparison"] is True:
        numerator = sum(int(row["prediction_preserved_count"]) for row in locality_rows)
        denominator = sum(int(row["prediction_token_count"]) for row in locality_rows)
        expected_locality.update(
            {
                "prediction_preservation_numerator": numerator,
                "prediction_preservation_denominator": denominator,
                "prediction_preservation_rate": numerator / denominator,
                "all_prompt_predictions_preserved_count": sum(
                    int(row["all_predictions_preserved"]) for row in locality_rows
                ),
            }
        )
    if payload.get("locality") != expected_locality:
        raise ArtifactBoundary("published locality summary differs")
    return {
        "status": "RAW_FREE_EVALUATION_PASS",
        "request_count": len(cases),
        "row_count": sum(len(value) for value in rows.values()),
        "identity_sha256": payload["identity_sha256"],
    }


def _numeric_counts(value: object, label: str) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise ArtifactBoundary(f"{label} is not an object")
    result: dict[str, int | float] = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ArtifactBoundary(f"{label} contains a nonnumeric field: {key}")
        number = _finite(item, f"{label}.{key}")
        if isinstance(item, int):
            if item < 0:
                raise ArtifactBoundary(f"{label}.{key} is negative")
            result[str(key)] = int(item)
        else:
            if number < 0.0:
                raise ArtifactBoundary(f"{label}.{key} is negative")
            result[str(key)] = number
    return result


def _reduce_endpoint_facts(
    endpoint: Mapping[str, Any], *, expected_arm: str, request_count: int
) -> dict[str, Any]:
    source = {str(key): item for key, item in endpoint.items() if key != "evaluation"}
    if source.get("arm") != expected_arm or source.get("status") not in PRIMARY_TERMINAL_STATUSES:
        raise ArtifactBoundary("endpoint fact arm/status differs")
    _validate_semantic_payload(
        source.get("semantic_observation"),
        request_count=request_count,
        label=f"{expected_arm}.endpoint",
    )
    normalized = _canonical_raw_free_clone(source, "endpoint facts")
    return _with_identity(
        {
            "schema": "orbode.raw-free-endpoint-facts.v1",
            "arm": expected_arm,
            "source_facts_identity_sha256": canonical_hash(source),
            "facts": normalized,
        }
    )


def _validate_endpoint_facts(
    value: object, *, expected_arm: str, request_count: int
) -> None:
    if not isinstance(value, Mapping):
        raise ArtifactBoundary("published endpoint facts are not an object")
    _assert_raw_free(value)
    _assert_identity(value)
    facts = value.get("facts")
    if (
        value.get("schema") != "orbode.raw-free-endpoint-facts.v1"
        or value.get("arm") != expected_arm
        or not isinstance(facts, Mapping)
        or canonical_hash(facts) != value.get("source_facts_identity_sha256")
    ):
        raise ArtifactBoundary("published endpoint fact binding differs")
    _sha256(value.get("source_facts_identity_sha256"), "endpoint source-facts identity")
    if facts.get("arm") != expected_arm or facts.get("status") not in PRIMARY_TERMINAL_STATUSES:
        raise ArtifactBoundary("published endpoint fact arm/status differs")
    _validate_semantic_payload(
        facts.get("semantic_observation"),
        request_count=request_count,
        label=f"published {expected_arm}.endpoint",
    )


def _reduce_safe_fact_tree(value: object, *, schema: str, label: str) -> dict[str, Any]:
    normalized = _canonical_raw_free_clone(value, label)
    return _with_identity(
        {
            "schema": schema,
            "source_identity_sha256": canonical_hash(value),
            "facts": normalized,
        }
    )


def _validate_safe_fact_tree(value: object, *, schema: str, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ArtifactBoundary(f"published {label} is not an object")
    _assert_raw_free(value)
    _assert_identity(value)
    facts = value.get("facts")
    if (
        value.get("schema") != schema
        or canonical_hash(facts) != value.get("source_identity_sha256")
    ):
        raise ArtifactBoundary(f"published {label} identity differs")
    _sha256(value.get("source_identity_sha256"), f"{label} source identity")


def _reduce_runtime_preamble(value: object, *, round_index: int) -> dict[str, Any] | None:
    if value is None:
        if round_index == 0:
            raise ArtifactBoundary("round0 runtime P0-P3 preamble is absent")
        return None
    if round_index != 0 or not isinstance(value, Mapping):
        raise ArtifactBoundary("runtime preamble round boundary differs")
    if (
        value.get("schema") != "orbode.fast-runtime-preamble.v1"
        or value.get("status") != "FAST_RUNTIME_PREAMBLE_PASS"
        or value.get("fixed_z_request_consumption_count") != 1
        or value.get("fixed_z_recompute_count") != 0
        or value.get("w0_pointer_bytes_restore_pass") is not True
        or value.get("scientific_selection_influence_count") != 0
    ):
        raise ArtifactBoundary("runtime preamble top-level gate differs")
    _sha256(value.get("request_sha256"), "runtime preamble request identity")
    if not isinstance(value.get("request_case_id"), str) or not value["request_case_id"]:
        raise ArtifactBoundary("runtime preamble request case identity differs")
    p0 = value.get("P0_official_scaling")
    stock = p0.get("stock_official_parity") if isinstance(p0, Mapping) else None
    if (
        not isinstance(p0, Mapping)
        or not isinstance(stock, Mapping)
        or stock.get("mode") != "PINNED_STOCK_R_OVER_N_WRAPPER_VS_DIRECT"
        or stock.get("dynamic_qcl_one_pass_used_as_stock_oracle") is not False
        or stock.get("exact_parity") is not True
        or any(
            stock.get(f"{field}_present") is not True
            or stock.get(f"{field}_equal") is not True
            for field in (
                "selected_weight_endpoint_sha256",
                "terminal_activation_sha256",
                "semantic_observation_sha256",
                "evaluation_sha256",
            )
        )
        or p0.get("dynamic_qcl_one_pass_stock_parity_claim")
        != "NOT_APPLICABLE_DISTINCT_NUMERICAL_PATH"
        or p0.get("right_factor_bitwise_identity") is not True
        or _finite(p0.get("residual_scaling_max_abs_error"), "P0 scaling error") != 0.0
    ):
        raise ArtifactBoundary("runtime preamble P0 gate differs")
    for field in (
        "selected_weight_endpoint_sha256",
        "terminal_activation_sha256",
        "semantic_observation_sha256",
        "evaluation_sha256",
    ):
        _sha256(stock.get(f"wrapper_{field}"), f"P0.stock.wrapper_{field}")
        _sha256(stock.get(f"direct_{field}"), f"P0.stock.direct_{field}")
    policy = value.get("outcome_comparison_gate_policy")
    if (
        not isinstance(policy, Mapping)
        or policy.get("schema") != "orbode.nonblocking-outcome-comparison-policy.v1"
        or policy.get("classification") != "NONBLOCKING_TELEMETRY_ONLY"
        or policy.get("blocking_outcome_comparison_count") != 0
        or any(
            policy.get(field) is not False
            for field in (
                "ours_vs_official_gate",
                "ours_vs_ours_gate",
                "resolution_match_monotonicity_convergence_gate",
                "action_magnitude_match_gate",
                "zero_correction_or_official_path_gate",
                "official_nonworse_outcome_gate",
            )
        )
        or policy.get("stock_official_o_wrapper_direct_fidelity_gate") is not True
        or policy.get("technical_integrity_gates_retained") is not True
        or policy.get("scientific_selection_influence_count") != 0
    ):
        raise ArtifactBoundary("runtime preamble outcome-comparison gate policy differs")
    p1 = value.get("P1_jvp")
    if not isinstance(p1, Mapping):
        raise ArtifactBoundary("runtime preamble P1 gate is absent")
    epsilon_grid = p1.get("epsilon_grid")
    if (
        epsilon_grid != [2.0 ** -7, 2.0 ** -8, 2.0 ** -9]
        or p1.get("primary_epsilon") != 2.0 ** -8
        or p1.get("absolute_tolerance_formula")
        != "64*eps_float32*max(1,max_abs(primal))/epsilon"
        or p1.get("relative_tolerance") != 2.0 ** -5
        or p1.get("response_selection_status")
        not in {
            "FIRST_NUMERICALLY_ACTIVE_RESPONSE",
            "ALL_ZERO_WITHIN_NUMERICAL_ENVELOPE_OBSERVED",
        }
        or not _is_int(p1.get("selected_layer"))
        or int(p1["selected_layer"]) not in (4, 5, 6, 7, 8)
        or not isinstance(p1.get("finite_difference"), list)
        or len(p1["finite_difference"]) != 3
        or any(
            not isinstance(item, Mapping)
            or item.get("allclose") is not True
            or item.get("numerical_activity_status")
            not in {
                "NUMERICALLY_ACTIVE_RESPONSE",
                "ALL_ZERO_WITHIN_NUMERICAL_ENVELOPE_OBSERVED",
            }
            for item in p1["finite_difference"]
        )
        or not isinstance(p1.get("virtual_materialized"), Mapping)
        or p1["virtual_materialized"].get("allclose") is not True
        or p1["virtual_materialized"].get("numerical_activity_status")
        not in {
            "NUMERICALLY_ACTIVE_RESPONSE",
            "ALL_ZERO_WITHIN_NUMERICAL_ENVELOPE_OBSERVED",
        }
    ):
        raise ArtifactBoundary("runtime preamble P1 FD/JVP gate differs")
    p2 = value.get("P2_state_transaction")
    if (
        not isinstance(p2, Mapping)
        or p2.get("changed_l5_state_version") != 1
        or p2.get("same_layer_repeated_factor_count") != 2
        or p2.get("inner_persistent_mutation_count") != 0
        or p2.get("algorithmic_retry_count") != 0
        or p2.get("algorithmic_rollback_count") != 0
        or not isinstance(p2.get("terminal_changed_observed"), bool)
        or not isinstance(p2.get("l5_keys_changed_observed"), bool)
        or p2.get("state_effect_comparison_policy")
        != "NONBLOCKING_TELEMETRY_ONLY"
        or p2.get("state_effect_comparison_gate_count") != 0
        or not isinstance(p2.get("overlay"), Mapping)
        or p2["overlay"].get("w0_pointer_version_bytes_unchanged") is not True
        or p2["overlay"].get("physical_inner_write_count") != 0
        or not isinstance(p2.get("terminal_identity"), Mapping)
        or p2["terminal_identity"].get("allclose") is not True
        or not isinstance(p2.get("semantic_logit_identity"), Mapping)
        or p2["semantic_logit_identity"].get("strict_identity") is not True
    ):
        raise ArtifactBoundary("runtime preamble P2 state/transaction gate differs")
    p3 = value.get("P3_resolution_observation_only")
    if (
        not isinstance(p3, list)
        or [item.get("N") if isinstance(item, Mapping) else None for item in p3] != [2, 4, 8]
    ):
        raise ArtifactBoundary("runtime preamble P3 resolution inventory differs")
    for item in p3:
        assert isinstance(item, Mapping)
        n = int(item["N"])
        if (
            _finite(item.get("h"), f"P3.N{n}.h") != 1.0 / n
            or _finite(item.get("T"), f"P3.N{n}.T") != 1.0
            or item.get("status") not in PRIMARY_TERMINAL_STATUSES
            or item.get("selection_influence_count") != 0
        ):
            raise ArtifactBoundary("runtime preamble P3 resolution fact differs")
        _sha256(item.get("selected_weight_endpoint_sha256"), f"P3.N{n}.endpoint")
        _count(item.get("step_count"), f"P3.N{n}.step_count")
        for field in ("terminal_potential", "maximum_discretization_defect", "sum_discretization_defect"):
            if item.get(field) is not None:
                _finite(item[field], f"P3.N{n}.{field}")
    p3_prefix = value.get("P3_orbhit_direct_prefix_identity")
    if not isinstance(p3_prefix, Mapping) or p3_prefix.get("orbfh_primary_gate_influence_count") != 0:
        raise ArtifactBoundary("runtime preamble P3 ORBHit identity gate differs")
    scrubbed = _scrub_exception_messages(value)
    normalized = _canonical_raw_free_clone(scrubbed, "runtime preamble")
    return _with_identity(
        {
            "schema": PREAMBLE_SCHEMA,
            "source_schema": value["schema"],
            "source_preamble_identity_sha256": canonical_hash(value),
            "facts": normalized,
            "literal_prompt_target_token_prediction_publication_count": 0,
        }
    )


def _validate_runtime_preamble(value: object, *, round_index: int) -> None:
    if value is None:
        if round_index == 0:
            raise ArtifactBoundary("published round0 runtime preamble is absent")
        return
    if round_index != 0 or not isinstance(value, Mapping):
        raise ArtifactBoundary("published runtime preamble round boundary differs")
    _assert_raw_free(value)
    _assert_identity(value)
    if (
        value.get("schema") != PREAMBLE_SCHEMA
        or value.get("source_schema") != "orbode.fast-runtime-preamble.v1"
        or value.get("literal_prompt_target_token_prediction_publication_count") != 0
        or not isinstance(value.get("facts"), Mapping)
    ):
        raise ArtifactBoundary("published runtime preamble contract differs")
    _sha256(value.get("source_preamble_identity_sha256"), "source preamble identity")
    # Revalidate all factual gates from the public raw-free facts.  Its source
    # hash cannot be recomputed only when an arbitrary exception message was
    # deliberately redacted, so the enclosing receipt binds that original.
    facts = value["facts"]
    reconstructed = _reduce_runtime_preamble(facts, round_index=round_index)
    if reconstructed.get("facts") != facts:
        raise ArtifactBoundary("published runtime preamble facts differ")


def reduce_endpoint_payload(
    arm_payload: Mapping[str, Any],
    *,
    expected_arm: str,
    case_ids: Sequence[int],
    request_sha256: Sequence[str],
    request_order_sha256: str,
    entry_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Reduce one primary arm endpoint and seal transaction/count facts."""

    if not isinstance(arm_payload, Mapping):
        raise ArtifactBoundary("primary arm payload is not an object")
    status = arm_payload.get("status")
    endpoint = arm_payload.get("endpoint")
    if (
        expected_arm not in PRIMARY_ARM_ORDER
        or arm_payload.get("arm") != expected_arm
        or status not in PRIMARY_TERMINAL_STATUSES
        or arm_payload.get("request_count") != len(case_ids)
        or arm_payload.get("scientific_promotion") is not False
        or not isinstance(endpoint, Mapping)
        or endpoint.get("arm") != expected_arm
        or endpoint.get("status") not in PRIMARY_TERMINAL_STATUSES
        or not isinstance(endpoint.get("evaluation"), Mapping)
    ):
        raise ArtifactBoundary("primary endpoint arm/status/request contract differs")
    selected_sha = _sha256(
        endpoint.get("selected_weight_endpoint_sha256"), "selected weight endpoint identity"
    )
    terminal_sha = _sha256(
        endpoint.get("terminal_activation_sha256"), "terminal activation identity"
    )
    physical_write_count = _count(endpoint.get("physical_write_count"), "physical write count")
    temporary_count = _count(
        endpoint.get("temporary_observation_materialization_count", 0),
        "temporary observation materialization count",
    )
    history_count = _count(endpoint.get("history_append_count"), "history append count")
    if (
        _count(endpoint.get("inner_history_append_count"), "inner history append count") != 0
        or _count(endpoint.get("inner_cache_mutation_count"), "inner cache mutation count") != 0
        or physical_write_count > 1
        or history_count > 1
    ):
        raise ArtifactBoundary("endpoint transaction boundary differs")
    overlay = arm_payload.get("overlay")
    overlay_identity = None
    if overlay is not None:
        if (
            not isinstance(overlay, Mapping)
            or overlay.get("w0_pointer_version_bytes_unchanged") is not True
            or overlay.get("physical_inner_write_count") != 0
        ):
            raise ArtifactBoundary("endpoint overlay W0/inner-write boundary differs")
        overlay_identity = canonical_hash(overlay)
    adapter_counts = _numeric_counts(arm_payload.get("adapter_ledger"), "adapter ledger")
    if (
        adapter_counts.get("inner_history_append_count") != 0
        or adapter_counts.get("inner_cache_mutation_count") != 0
        or adapter_counts.get("dynamic_z_count") != 0
        or adapter_counts.get("fixed_z_recompute_count") != 0
    ):
        raise ArtifactBoundary("endpoint adapter transaction/fixed-z boundary differs")
    jvp_counts = _numeric_counts(arm_payload.get("jvp_ledger"), "JVP ledger")
    mechanism = _reduce_mechanism_telemetry(
        arm_payload.get("telemetry"),
        expected_arm=expected_arm,
        request_count=len(case_ids),
    )
    endpoint_facts = _reduce_endpoint_facts(
        endpoint, expected_arm=expected_arm, request_count=len(case_ids)
    )
    accounting = _reduce_safe_fact_tree(
        arm_payload.get("accounting_reconciliation", {}),
        schema="orbode.raw-free-accounting-reconciliation.v1",
        label="accounting reconciliation",
    )
    dtype_scope = _reduce_safe_fact_tree(
        arm_payload.get("dtype_scope", {}),
        schema="orbode.raw-free-dtype-scope.v1",
        label="dtype scope",
    )
    evaluation = reduce_evaluation_payload(
        endpoint["evaluation"],
        case_ids=case_ids,
        request_sha256=request_sha256,
        request_order_sha256=request_order_sha256,
        entry_evaluation=entry_evaluation,
    )
    payload = {
        "schema": ENDPOINT_SCHEMA,
        "arm": expected_arm,
        "status": status,
        "endpoint_status": endpoint["status"],
        "request_count": len(case_ids),
        "selected_weight_endpoint_sha256": selected_sha,
        "terminal_activation_sha256": terminal_sha,
        "fixed_z_identity_sha256": _sha256(
            arm_payload.get("fixed_z_identity_sha256"), "endpoint fixed-z identity"
        ),
        "physical_write_count": physical_write_count,
        "temporary_observation_materialization_count": temporary_count,
        "history_append_count": history_count,
        "inner_history_append_count": 0,
        "inner_cache_mutation_count": 0,
        "adapter_counts": adapter_counts,
        "jvp_counts": jvp_counts,
        "overlay_identity_sha256": overlay_identity,
        "mechanism_telemetry": mechanism,
        "endpoint_facts": endpoint_facts,
        "accounting_reconciliation": accounting,
        "dtype_scope": dtype_scope,
        "source_endpoint_identity_sha256": canonical_hash(arm_payload),
        "wall_seconds": _finite(arm_payload.get("wall_seconds"), "endpoint wall seconds"),
        "model_forward_invocation_count": _count(
            arm_payload.get("model_forward_invocation_count"),
            "endpoint model forward invocation count",
        ),
        "evaluation": evaluation,
        "scientific_promotion": False,
    }
    result = _with_identity(payload)
    validate_endpoint_publication(
        result,
        expected_arm=expected_arm,
        case_ids=case_ids,
        request_sha256=request_sha256,
        request_order_sha256=request_order_sha256,
    )
    return result


def validate_endpoint_publication(
    payload: Mapping[str, Any],
    *,
    expected_arm: str,
    case_ids: Sequence[int],
    request_sha256: Sequence[str],
    request_order_sha256: str,
) -> dict[str, Any]:
    """Validate one raw-free primary endpoint publication."""

    if not isinstance(payload, Mapping):
        raise ArtifactBoundary("published endpoint is not an object")
    _assert_raw_free(payload)
    _assert_identity(payload)
    if (
        payload.get("schema") != ENDPOINT_SCHEMA
        or payload.get("arm") != expected_arm
        or payload.get("status") not in PRIMARY_TERMINAL_STATUSES
        or payload.get("endpoint_status") not in PRIMARY_TERMINAL_STATUSES
        or payload.get("request_count") != len(case_ids)
        or payload.get("inner_history_append_count") != 0
        or payload.get("inner_cache_mutation_count") != 0
        or payload.get("scientific_promotion") is not False
    ):
        raise ArtifactBoundary("published endpoint contract differs")
    for field in (
        "selected_weight_endpoint_sha256",
        "terminal_activation_sha256",
        "fixed_z_identity_sha256",
        "source_endpoint_identity_sha256",
    ):
        _sha256(payload.get(field), field)
    if payload.get("overlay_identity_sha256") is not None:
        _sha256(payload["overlay_identity_sha256"], "overlay identity")
    _count(payload.get("physical_write_count"), "published physical write count")
    _count(
        payload.get("temporary_observation_materialization_count"),
        "published temporary observation materialization count",
    )
    _count(payload.get("history_append_count"), "published history append count")
    _finite(payload.get("wall_seconds"), "published endpoint wall seconds")
    _count(
        payload.get("model_forward_invocation_count"),
        "published endpoint model forward invocation count",
    )
    _numeric_counts(payload.get("adapter_counts"), "published adapter counts")
    _numeric_counts(payload.get("jvp_counts"), "published JVP counts")
    _validate_mechanism_publication(
        payload.get("mechanism_telemetry"),
        expected_arm=expected_arm,
        request_count=len(case_ids),
    )
    _validate_endpoint_facts(
        payload.get("endpoint_facts"),
        expected_arm=expected_arm,
        request_count=len(case_ids),
    )
    _validate_safe_fact_tree(
        payload.get("accounting_reconciliation"),
        schema="orbode.raw-free-accounting-reconciliation.v1",
        label="accounting reconciliation",
    )
    _validate_safe_fact_tree(
        payload.get("dtype_scope"),
        schema="orbode.raw-free-dtype-scope.v1",
        label="dtype scope",
    )
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("entry_comparison") is not True:
        raise ArtifactBoundary("published endpoint lacks entry comparison")
    validate_evaluation_publication(
        evaluation,
        case_ids=case_ids,
        request_sha256=request_sha256,
        request_order_sha256=request_order_sha256,
    )
    return {
        "status": "RAW_FREE_ENDPOINT_PASS",
        "arm": expected_arm,
        "request_count": len(case_ids),
        "identity_sha256": payload["identity_sha256"],
    }


def _reduce_derived_endpoint(
    value: object,
    *,
    case_ids: Sequence[int],
    request_sha256: Sequence[str],
    request_order_sha256: str,
    entry_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("arm") != DERIVED_ARM:
        raise ArtifactBoundary("ORBFH derived endpoint is absent")
    status = value.get("status")
    if status not in DERIVED_TERMINAL_STATUSES:
        raise ArtifactBoundary("ORBHit status differs")
    endpoint = value.get("endpoint")
    if not isinstance(endpoint, Mapping):
        raise ArtifactBoundary("ORBHit endpoint is not an object")
    payload: dict[str, Any] = {
        "arm": DERIVED_ARM,
        "status": status,
        "source_derived_identity_sha256": canonical_hash(value),
        "first_hit_identity_sha256": canonical_hash(value.get("first_hit")),
        "state_version": _count(value.get("state_version"), "ORBHit state version"),
        "factor_count": _count(value.get("factor_count"), "ORBHit factor count"),
        "withheld_technical": status == "ORBHit_WITHHELD_TECHNICAL",
    }
    if status != "ORBHit_WITHHELD_TECHNICAL":
        if not isinstance(endpoint.get("evaluation"), Mapping):
            raise ArtifactBoundary("valid ORBHit endpoint evaluation is absent")
        payload.update(
            {
                "selected_weight_endpoint_sha256": _sha256(
                    endpoint.get("selected_weight_endpoint_sha256"),
                    "ORBHit selected weight identity",
                ),
                "evaluation": reduce_evaluation_payload(
                    endpoint["evaluation"],
                    case_ids=case_ids,
                    request_sha256=request_sha256,
                    request_order_sha256=request_order_sha256,
                    entry_evaluation=entry_evaluation,
                ),
            }
        )
    return _with_identity(payload)


def reduce_round_payload(round_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce one exact B100 round and validate the primary terminal panel."""

    if not isinstance(round_payload, Mapping):
        raise ArtifactBoundary("round payload is not an object")
    round_index = round_payload.get("round_index")
    case_ids = round_payload.get("case_ids")
    request_sha256 = round_payload.get("request_sha256")
    request_order_sha256 = round_payload.get("canonical_request_order_sha256")
    arms = round_payload.get("arms")
    if (
        not _is_int(round_index)
        or not 0 <= int(round_index) <= 9
        or round_payload.get("request_count") != 100
        or not isinstance(case_ids, list)
        or not isinstance(request_sha256, list)
        or len(case_ids) != 100
        or len(request_sha256) != 100
        or not isinstance(arms, list)
        or len(arms) != len(PRIMARY_ARM_ORDER)
        or round_payload.get("w0_pointer_bytes_restore_pass") is not True
        or not isinstance(round_payload.get("pre_evaluation"), Mapping)
    ):
        raise ArtifactBoundary("round identity/request/W0 contract differs")
    cases, _request_by_case = _request_bindings(case_ids, request_sha256)
    order_sha = _sha256(request_order_sha256, "round request order identity")
    if canonical_hash(request_sha256) != order_sha:
        raise ArtifactBoundary("round request order identity differs")
    observed_arm_order = tuple(
        value.get("arm") if isinstance(value, Mapping) else None for value in arms
    )
    if observed_arm_order != PRIMARY_ARM_ORDER:
        raise ArtifactBoundary("round primary arm order differs")
    entry = reduce_evaluation_payload(
        round_payload["pre_evaluation"],
        case_ids=cases,
        request_sha256=request_sha256,
        request_order_sha256=order_sha,
    )
    endpoints = [
        reduce_endpoint_payload(
            arm,
            expected_arm=expected_arm,
            case_ids=cases,
            request_sha256=request_sha256,
            request_order_sha256=order_sha,
            entry_evaluation=round_payload["pre_evaluation"],
        )
        for arm, expected_arm in zip(arms, PRIMARY_ARM_ORDER, strict=True)
    ]
    derived = _reduce_derived_endpoint(
        arms[3].get("derived_endpoint"),
        case_ids=cases,
        request_sha256=request_sha256,
        request_order_sha256=order_sha,
        entry_evaluation=round_payload["pre_evaluation"],
    )
    fixed_z = round_payload.get("fixed_z")
    semantic = round_payload.get("semantic")
    if (
        not isinstance(fixed_z, Mapping)
        or fixed_z.get("compute_count") != 100
        or fixed_z.get("recompute_count") != 0
        or not isinstance(semantic, Mapping)
        or semantic.get("request_count") != 100
        or not _is_int(semantic.get("event_count"))
        or int(semantic["event_count"]) <= 100
    ):
        raise ArtifactBoundary("round fixed-z/semantic denominator differs")
    runtime_preamble = _reduce_runtime_preamble(
        round_payload.get("runtime_preamble"), round_index=int(round_index)
    )
    payload = {
        "schema": ROUND_SCHEMA,
        "status": "TERMINAL_VALID",
        "round_index": int(round_index),
        "request_count": 100,
        "case_ids": list(cases),
        "request_sha256": list(request_sha256),
        "request_order_sha256": order_sha,
        "sealed_batch_order_digest": _sha256(
            round_payload.get("sealed_batch_order_digest"),
            "sealed batch order digest",
        ),
        "fixed_z_identity_sha256": _sha256(
            fixed_z.get("identity_sha256"), "round fixed-z identity"
        ),
        "fixed_z_target_context_identity_sha256": _sha256(
            fixed_z.get("target_context_identity_sha256"),
            "fixed-z target context identity",
        ),
        "fixed_z_compute_count": 100,
        "fixed_z_recompute_count": 0,
        "semantic_inventory_sha256": _sha256(
            semantic.get("inventory_sha256"), "semantic inventory identity"
        ),
        "semantic_event_count": int(semantic["event_count"]),
        "primary_arm_order": list(PRIMARY_ARM_ORDER),
        "primary_endpoint_count": 500,
        "entry_evaluation": entry,
        "primary_endpoints": endpoints,
        "derived_orbhit": derived,
        "w0_sha256": _sha256(round_payload.get("w0_sha256"), "round W0 identity"),
        "w0_pointer_bytes_restore_pass": True,
        "model_forward_invocation_count": _count(
            round_payload.get("model_forward_invocation_count"),
            "round model forward invocation count",
        ),
        "runtime_preamble": runtime_preamble,
        "source_round_identity_sha256": canonical_hash(round_payload),
        "literal_prompt_target_token_prediction_publication_count": 0,
        "scientific_promotion": False,
    }
    result = _with_identity(payload)
    validate_round_publication(result, expected_round_index=int(round_index))
    return result


def validate_round_publication(
    payload: Mapping[str, Any], *, expected_round_index: int
) -> dict[str, Any]:
    """Validate a raw-free B100 round for runtime/common-gate consumption."""

    if not isinstance(payload, Mapping):
        raise ArtifactBoundary("published round is not an object")
    _assert_raw_free(payload)
    _assert_identity(payload)
    case_ids = payload.get("case_ids")
    request_sha256 = payload.get("request_sha256")
    request_order_sha256 = payload.get("request_order_sha256")
    if (
        payload.get("schema") != ROUND_SCHEMA
        or payload.get("status") != "TERMINAL_VALID"
        or payload.get("round_index") != expected_round_index
        or payload.get("request_count") != 100
        or not isinstance(case_ids, list)
        or not isinstance(request_sha256, list)
        or len(case_ids) != 100
        or len(request_sha256) != 100
        or payload.get("primary_arm_order") != list(PRIMARY_ARM_ORDER)
        or payload.get("primary_endpoint_count") != 500
        or payload.get("fixed_z_compute_count") != 100
        or payload.get("fixed_z_recompute_count") != 0
        or payload.get("w0_pointer_bytes_restore_pass") is not True
        or payload.get("literal_prompt_target_token_prediction_publication_count") != 0
        or payload.get("scientific_promotion") is not False
    ):
        raise ArtifactBoundary("published round contract differs")
    cases, _request_by_case = _request_bindings(case_ids, request_sha256)
    order_sha = _sha256(request_order_sha256, "published round request order identity")
    if canonical_hash(request_sha256) != order_sha:
        raise ArtifactBoundary("published round request order reprojection differs")
    for field in (
        "sealed_batch_order_digest",
        "fixed_z_identity_sha256",
        "fixed_z_target_context_identity_sha256",
        "semantic_inventory_sha256",
        "w0_sha256",
        "source_round_identity_sha256",
    ):
        _sha256(payload.get(field), field)
    _count(payload.get("semantic_event_count"), "semantic event count")
    _count(payload.get("model_forward_invocation_count"), "round model forward count")
    _validate_runtime_preamble(
        payload.get("runtime_preamble"), round_index=expected_round_index
    )
    entry = payload.get("entry_evaluation")
    if not isinstance(entry, Mapping) or entry.get("entry_comparison") is not False:
        raise ArtifactBoundary("published round entry evaluation differs")
    validate_evaluation_publication(
        entry,
        case_ids=cases,
        request_sha256=request_sha256,
        request_order_sha256=order_sha,
    )
    endpoints = payload.get("primary_endpoints")
    if not isinstance(endpoints, list) or len(endpoints) != len(PRIMARY_ARM_ORDER):
        raise ArtifactBoundary("published round endpoint panel differs")
    if tuple(item.get("arm") for item in endpoints if isinstance(item, Mapping)) != PRIMARY_ARM_ORDER:
        raise ArtifactBoundary("published round endpoint arm order differs")
    for endpoint, arm in zip(endpoints, PRIMARY_ARM_ORDER, strict=True):
        validate_endpoint_publication(
            endpoint,
            expected_arm=arm,
            case_ids=cases,
            request_sha256=request_sha256,
            request_order_sha256=order_sha,
        )
    derived = payload.get("derived_orbhit")
    if (
        not isinstance(derived, Mapping)
        or derived.get("arm") != DERIVED_ARM
        or derived.get("status") not in DERIVED_TERMINAL_STATUSES
        or derived.get("withheld_technical") is not (
            derived.get("status") == "ORBHit_WITHHELD_TECHNICAL"
        )
    ):
        raise ArtifactBoundary("published ORBHit endpoint differs")
    _assert_identity(derived)
    _sha256(derived.get("source_derived_identity_sha256"), "derived source identity")
    _sha256(derived.get("first_hit_identity_sha256"), "derived first-hit identity")
    if derived["withheld_technical"] is False:
        validate_evaluation_publication(
            derived.get("evaluation"),
            case_ids=cases,
            request_sha256=request_sha256,
            request_order_sha256=order_sha,
        )
    return {
        "status": "RAW_FREE_ROUND_PASS",
        "round_index": expected_round_index,
        "request_count": 100,
        "primary_endpoint_count": 500,
        "identity_sha256": payload["identity_sha256"],
    }


__all__ = [
    "ArtifactBoundary",
    "ENDPOINT_SCHEMA",
    "EVALUATION_SCHEMA",
    "MECHANISM_SCHEMA",
    "PREAMBLE_SCHEMA",
    "PRIMARY_ARM_ORDER",
    "PROHIBITED_PUBLIC_FIELDS",
    "ROUND_SCHEMA",
    "canonical_hash",
    "reduce_endpoint_payload",
    "reduce_evaluation_payload",
    "reduce_round_payload",
    "validate_endpoint_publication",
    "validate_evaluation_publication",
    "validate_round_publication",
]
