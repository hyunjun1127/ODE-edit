#!/usr/bin/env python3
"""Fail-closed MEMIT/AlphaEdit direct-z cross-track analyzer.

This module joins the two fixed direct-z run families through immutable,
scalar-only provenance.  It reports four distinct paired quantities for every
shared metric: the MEMIT BF-minus-native lift, the AlphaEdit BF-minus-native
lift, their same-case 2x2 interaction, and the absolute AlphaEdit-BF minus
MEMIT-BF endpoint contrast.  The interaction and endpoint contrast are never
conflated.

The cross-bound join is deliberately stricter than either single-track
analysis: source manifest/summary identities, replay lock, ordered
case/request ids, direct-z artifact metadata, W0 lineage, frozen target
digests, cross-equal heldout evaluation identity, and matched C budget must
agree.  Alpha-only context/state/snapshot/parameter/solver fields have no
counterpart in the pre-run replay lock, so they are validated and reported as
self-bound internal provenance rather than mislabelled as externally pinned
cross-track identity.  A contract mismatch produces a BLOCK result.  No
prompt, tensor value, logit, token text, or generation is emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ANALYSIS_SCHEMA = "ode-edit-direct-z-cross-track-analysis/v2"
LOCK_SCHEMA = "ode-edit-direct-z-alpha-replay-lock/v1"
MEMIT_MANIFEST_SCHEMA = "ode-edit-direct-z-possibility-manifest/v1"
MEMIT_SUMMARY_SCHEMA = "ode-edit-direct-z-possibility-summary/v1"
MEMIT_STREAM_SCHEMA = "ode-edit-direct-z-possibility-stream/v1"
ALPHA_MANIFEST_SCHEMA = "ode-edit-direct-z-alpha-paired-manifest/v1"
ALPHA_SUMMARY_SCHEMA = "ode-edit-direct-z-alpha-paired-summary/v1"
ALPHA_STREAM_SCHEMA = "ode-edit-direct-z-alpha-paired-stream/v1"

EXPECTED_REPLAY_LOCK_ID = (
    "8ee67660c167233e95f77cc170011eccf1526d08bd1337f5fd6da612db641f6e"
)
EXPECTED_CASE_ORDER = (
    "21100",
    "1477",
    "20838",
    "18707",
    "14288",
    "16426",
    "19041",
    "17609",
)
EXPECTED_CASE_COUNT = len(EXPECTED_CASE_ORDER)
EXPECTED_RANK_SLICE = (132, 140)
BOOTSTRAP_SEED = 20260801
BOOTSTRAP_RESAMPLES = 4000
STRICT_MAJORITY = 5
C_ENERGY_REL_TOL = 5e-5
C_ENERGY_ABS_TOL = 1e-10
PRESERVATION_ABS_TOL = 1e-12
CLAIM_BOUNDARY = (
    "fixed_atomic_cross_track_geometry_signal_only_not_method_superiority_or_lifelong"
)

MODEL_ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")
RUN_IDS = {
    "llama3-8b-inst": {
        "memit": "dzf_llama_p0_v2",
        "alpha": "dzf_alpha_llama_p0_v1",
    },
    "qwen2.5-7b-inst": {
        "memit": "dzf_qwen_p0_v2",
        "alpha": "dzf_alpha_qwen_p0_v1",
    },
}
MEMIT_ARM_ORDER = (
    "no_op_replay",
    "oracle_do_z",
    "native_ordered_full",
    "native_alpha_c_matched",
    "bf_current_refreshed_k4",
    "sync_z_cone_oracle",
)
ALPHA_ARM_ORDER = (
    "no_op_replay",
    "oracle_do_z",
    "alpha_genuine_ordered_full",
    "alpha_genuine_ordered_c_matched",
    "alpha_posthoc_bp_full",
    "alpha_posthoc_bp_c_matched",
    "alpha_bf_genuine_refreshed_k4",
    "alpha_sync_z_cone_genuine",
)
MEMIT_NATIVE = "native_alpha_c_matched"
MEMIT_BF = "bf_current_refreshed_k4"
ALPHA_NATIVE = "alpha_genuine_ordered_c_matched"
ALPHA_BF = "alpha_bf_genuine_refreshed_k4"
MEMIT_CLAIM_BOUNDARY = "possibility_only_not_method_superiority"
ALPHA_CLAIM_BOUNDARY = (
    "atomic_geometry_signal_only_not_lifelong_or_method_superiority"
)
ALPHA_ARM_CONSTRUCTION = {
    "no_op_replay": "no_weight_write",
    "oracle_do_z": "weight_free_direct_z_ceiling",
    "alpha_genuine_ordered_full": "genuine_alphaedit_isolated_ordered_solve",
    "alpha_genuine_ordered_c_matched": "genuine_alphaedit_isolated_ordered_solve_c_matched",
    "alpha_posthoc_bp_full": "posthoc_unprojected_alpha_base_right_projection",
    "alpha_posthoc_bp_c_matched": "posthoc_unprojected_alpha_base_right_projection_c_matched",
    "alpha_bf_genuine_refreshed_k4": "genuine_alphaedit_refreshed_k4_endpoint_c_matched",
    "alpha_sync_z_cone_genuine": "genuine_alphaedit_sync_cone_diagnostic",
}
ALPHA_GENUINE_ARMS = frozenset(
    {
        "alpha_genuine_ordered_full",
        "alpha_genuine_ordered_c_matched",
        "alpha_bf_genuine_refreshed_k4",
        "alpha_sync_z_cone_genuine",
    }
)
ALPHA_POSTHOC_ARMS = frozenset(
    {"alpha_posthoc_bp_full", "alpha_posthoc_bp_c_matched"}
)

# Positive sign-normalized effects always mean that the left-hand side is
# better on the named axis.  Endpoint C energy is a validity constraint and is
# intentionally excluded from performance effects.
METRIC_DIRECTIONS: dict[str, str] = {
    "z_residual_ratio": "lower_is_better",
    "generated_delta_error_mean": "lower_is_better",
    "generated_delta_error_worst": "lower_is_better",
    "delta_gain": "higher_is_better",
    "delta_cosine": "higher_is_better",
    "off_token_spill_ratio": "lower_is_better",
    "output_progress": "higher_is_better",
    "output_nll_reduction": "higher_is_better",
    "exact_margin_min": "higher_is_better",
    "exact_satisfied": "higher_is_better",
    "paraphrase_nll_reduction": "higher_is_better",
    "heldout_kl": "lower_is_better",
    "endpoint_frobenius_norm": "lower_is_better",
    "nfe": "lower_is_better",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,127}$")
_CASE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_ENVELOPE_KEYS = {
    "schema_version",
    "run_id",
    "sequence",
    "recorded_at",
    "event",
    "payload",
}
_COMMON_OUTCOME_KEYS = {
    "case_id",
    "request_id",
    "arm_id",
    "case_pass",
    "arm_success",
    "technical_pass",
    "rollback_exact",
    "firewall_pass",
    "receipt_before_outcome",
    "z_residual_ratio",
    "generated_delta_error_mean",
    "generated_delta_error_worst",
    "delta_gain",
    "delta_cosine",
    "off_token_spill_ratio",
    "output_progress",
    "output_nll_reduction",
    "exact_margin_min",
    "exact_satisfied",
    "paraphrase_nll_reduction",
    "heldout_kl",
    "preservation_score",
    "endpoint_c_energy",
    "endpoint_frobenius_norm",
    "nfe",
}


class DirectZCrossTrackAnalysisError(ValueError):
    """Raised for malformed or unsafe artifacts that cannot yield a result."""


def _reject_nonstandard_constant(value: str) -> None:
    raise DirectZCrossTrackAnalysisError(
        f"non-finite JSON constant rejected: {value}"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DirectZCrossTrackAnalysisError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DirectZCrossTrackAnalysisError(f"{path}: object required")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    if set(value) != expected:
        raise DirectZCrossTrackAnalysisError(
            f"{path}: exact key mismatch; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _required_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    missing = expected - set(value)
    if missing:
        raise DirectZCrossTrackAnalysisError(
            f"{path}: missing required keys: {sorted(missing)}"
        )


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DirectZCrossTrackAnalysisError(f"{path}: lowercase SHA-256 required")
    return value


def _run_id(value: Any, path: str) -> str:
    if not isinstance(value, str) or _RUN_ID_RE.fullmatch(value) is None:
        raise DirectZCrossTrackAnalysisError(f"{path}: sanitized run id required")
    return value


def _case_id(value: Any, path: str) -> str:
    if type(value) is int and value >= 0:
        return str(value)
    if isinstance(value, str) and _CASE_ID_RE.fullmatch(value) is not None:
        return value
    raise DirectZCrossTrackAnalysisError(f"{path}: sanitized case id required")


def _bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise DirectZCrossTrackAnalysisError(f"{path}: bool required")
    return value


def _int(value: Any, path: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise DirectZCrossTrackAnalysisError(f"{path}: invalid integer")
    return value


def _finite(value: Any, path: str, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float):
        raise DirectZCrossTrackAnalysisError(f"{path}: finite scalar required")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise DirectZCrossTrackAnalysisError(f"{path}: scalar outside contract")
    return result


def _timestamp(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 64
        or any(ord(character) < 32 for character in value)
    ):
        raise DirectZCrossTrackAnalysisError(f"{path}: invalid timestamp")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_identity(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DirectZCrossTrackAnalysisError(
            f"{label}: regular non-symlink file required"
        )
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise DirectZCrossTrackAnalysisError(f"{label}: read failure") from exc
    return {"sha256": digest.hexdigest(), "size": size}


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    _file_identity(path, label)
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonstandard_constant,
            )
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectZCrossTrackAnalysisError(f"{label}: invalid JSON") from exc
    return _mapping(value, label)


def _read_jsonl(path: Path, label: str) -> list[Mapping[str, Any]]:
    _file_identity(path, label)
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    value = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_keys,
                        parse_constant=_reject_nonstandard_constant,
                    )
                except json.JSONDecodeError as exc:
                    raise DirectZCrossTrackAnalysisError(
                        f"{label}[{line_number}]: invalid JSON"
                    ) from exc
                rows.append(_mapping(value, f"{label}[{line_number}]"))
    except OSError as exc:
        raise DirectZCrossTrackAnalysisError(f"{label}: read failure") from exc
    return rows


def _failure(failures: list[str], model_alias: str, case_id: str, field: str) -> None:
    failures.append(f"model={model_alias} case={case_id}: {field} mismatch")


def _check(
    condition: bool,
    failures: list[str],
    model_alias: str,
    case_id: str,
    field: str,
) -> None:
    if not condition:
        _failure(failures, model_alias, case_id, field)


def _load_lock(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path = Path(path).expanduser().resolve(strict=True)
    raw = _read_json_object(lock_path, "replay_lock")
    _exact_keys(
        raw,
        {"schema_version", "selection_sha256", "rank_slice", "case_order", "models"},
        "replay_lock",
    )
    if raw["schema_version"] != LOCK_SCHEMA:
        raise DirectZCrossTrackAnalysisError("replay_lock: schema mismatch")
    selection_sha256 = _sha256(raw["selection_sha256"], "replay_lock.selection_sha256")
    if not isinstance(raw["rank_slice"], list) or tuple(raw["rank_slice"]) != EXPECTED_RANK_SLICE:
        raise DirectZCrossTrackAnalysisError("replay_lock: fixed rank slice mismatch")
    if not isinstance(raw["case_order"], list):
        raise DirectZCrossTrackAnalysisError("replay_lock.case_order: array required")
    case_order = tuple(
        _case_id(value, f"replay_lock.case_order[{index}]")
        for index, value in enumerate(raw["case_order"])
    )
    if case_order != EXPECTED_CASE_ORDER:
        raise DirectZCrossTrackAnalysisError("replay_lock: exact case order mismatch")
    raw_models = _mapping(raw["models"], "replay_lock.models")
    if set(raw_models) != set(MODEL_ALIASES):
        raise DirectZCrossTrackAnalysisError("replay_lock: fixed model set mismatch")
    models: dict[str, Any] = {}
    for model_alias in MODEL_ALIASES:
        model = _mapping(raw_models[model_alias], f"replay_lock.models.{model_alias}")
        _exact_keys(
            model,
            {"source_run_id", "source_manifest", "source_summary", "artifacts"},
            f"replay_lock.models.{model_alias}",
        )
        source_run_id = _run_id(
            model["source_run_id"], f"replay_lock.models.{model_alias}.source_run_id"
        )
        if source_run_id != RUN_IDS[model_alias]["memit"]:
            raise DirectZCrossTrackAnalysisError(
                f"replay_lock.models.{model_alias}: fixed MEMIT run mismatch"
            )
        identities: dict[str, dict[str, Any]] = {}
        for name in ("source_manifest", "source_summary"):
            identity = _mapping(model[name], f"replay_lock.models.{model_alias}.{name}")
            _exact_keys(identity, {"sha256", "size"}, f"replay_lock.models.{model_alias}.{name}")
            identities[name] = {
                "sha256": _sha256(identity["sha256"], f"replay_lock.{model_alias}.{name}.sha256"),
                "size": _int(identity["size"], f"replay_lock.{model_alias}.{name}.size", positive=True),
            }
        raw_artifacts = model["artifacts"]
        if not isinstance(raw_artifacts, list) or len(raw_artifacts) != EXPECTED_CASE_COUNT:
            raise DirectZCrossTrackAnalysisError(
                f"replay_lock.models.{model_alias}.artifacts: exact count required"
            )
        artifacts: list[dict[str, Any]] = []
        for index, item_raw in enumerate(raw_artifacts):
            item = _mapping(item_raw, f"replay_lock.{model_alias}.artifacts[{index}]")
            _exact_keys(
                item,
                {
                    "case_id",
                    "request_id",
                    "artifact_sha256",
                    "artifact_size",
                    "tensor_sha256",
                    "origin_lineage_id",
                    "target_token_sha256",
                    "reference_c_energy",
                },
                f"replay_lock.{model_alias}.artifacts[{index}]",
            )
            case_id = _case_id(item["case_id"], f"replay_lock.{model_alias}.case_id")
            if case_id != EXPECTED_CASE_ORDER[index]:
                raise DirectZCrossTrackAnalysisError(
                    f"replay_lock.models.{model_alias}: artifact case order mismatch"
                )
            artifacts.append(
                {
                    "case_id": case_id,
                    "request_id": _sha256(item["request_id"], "replay_lock.artifact.request_id"),
                    "artifact_sha256": _sha256(item["artifact_sha256"], "replay_lock.artifact.sha256"),
                    "artifact_size": _int(item["artifact_size"], "replay_lock.artifact.size", positive=True),
                    "tensor_sha256": _sha256(item["tensor_sha256"], "replay_lock.artifact.tensor_sha256"),
                    "origin_lineage_id": _sha256(item["origin_lineage_id"], "replay_lock.artifact.origin_lineage_id"),
                    "target_token_sha256": _sha256(item["target_token_sha256"], "replay_lock.artifact.target_token_sha256"),
                    "reference_c_energy": _finite(
                        item["reference_c_energy"],
                        "replay_lock.artifact.reference_c_energy",
                        nonnegative=True,
                    ),
                }
            )
            if artifacts[-1]["reference_c_energy"] <= 0.0:
                raise DirectZCrossTrackAnalysisError(
                    "replay_lock artifact reference C energy must be positive"
                )
        models[model_alias] = {
            "source_run_id": source_run_id,
            **identities,
            "artifacts": artifacts,
        }
    request_orders = {
        tuple(item["request_id"] for item in models[alias]["artifacts"])
        for alias in MODEL_ALIASES
    }
    if len(request_orders) != 1:
        raise DirectZCrossTrackAnalysisError(
            "replay_lock: request order must be identical across models"
        )
    lock_id = _canonical_sha256(raw)
    if lock_id != EXPECTED_REPLAY_LOCK_ID:
        raise DirectZCrossTrackAnalysisError("replay_lock: pinned canonical id mismatch")
    return (
        {
            "schema_version": LOCK_SCHEMA,
            "selection_sha256": selection_sha256,
            "case_order": list(EXPECTED_CASE_ORDER),
            "models": models,
            "lock_id": lock_id,
        },
        _file_identity(lock_path, "replay_lock"),
    )


def _validate_run_root(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise DirectZCrossTrackAnalysisError(f"{label}: symlink directory rejected")
    root = candidate.resolve(strict=True)
    if not root.is_dir():
        raise DirectZCrossTrackAnalysisError(f"{label}: directory required")
    return root


def _validate_manifest_summary(
    root: Path,
    *,
    track: str,
    model_alias: str,
    lock: Mapping[str, Any],
    failures: list[str],
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any]]:
    manifest_path = root / "manifest.json"
    summary_path = root / "summary.json"
    manifest = _read_json_object(manifest_path, f"{model_alias}.{track}.manifest")
    summary = _read_json_object(summary_path, f"{model_alias}.{track}.summary")
    manifest_identity = _file_identity(manifest_path, "manifest")
    summary_identity = _file_identity(summary_path, "summary")
    expected_run = RUN_IDS[model_alias][track]
    expected_arms = MEMIT_ARM_ORDER if track == "memit" else ALPHA_ARM_ORDER
    expected_manifest_schema = (
        MEMIT_MANIFEST_SCHEMA if track == "memit" else ALPHA_MANIFEST_SCHEMA
    )
    expected_summary_schema = MEMIT_SUMMARY_SCHEMA if track == "memit" else ALPHA_SUMMARY_SCHEMA
    _required_keys(
        manifest,
        {
            "schema_version",
            "run_id",
            "model_alias",
            "case_count",
            "arm_order",
            "selection_sha256",
            "config_sha256",
            "bootstrap_seed",
            "bootstrap_resamples",
            "claim_boundary",
        },
        f"{model_alias}.{track}.manifest",
    )
    run_id = _run_id(manifest["run_id"], f"{model_alias}.{track}.manifest.run_id")
    _check(run_id == expected_run, failures, model_alias, "all", f"{track}.run_id")
    _check(manifest["schema_version"] == expected_manifest_schema, failures, model_alias, "all", f"{track}.manifest.schema")
    _check(manifest["model_alias"] == model_alias, failures, model_alias, "all", f"{track}.manifest.model")
    _check(_int(manifest["case_count"], "manifest.case_count") == EXPECTED_CASE_COUNT, failures, model_alias, "all", f"{track}.manifest.case_count")
    _check(isinstance(manifest["arm_order"], list) and tuple(manifest["arm_order"]) == expected_arms, failures, model_alias, "all", f"{track}.manifest.arm_order")
    selection = _sha256(manifest["selection_sha256"], "manifest.selection_sha256")
    _check(selection == lock["selection_sha256"], failures, model_alias, "all", f"{track}.selection")
    _sha256(manifest["config_sha256"], "manifest.config_sha256")
    _check(_int(manifest["bootstrap_seed"], "manifest.bootstrap_seed") == BOOTSTRAP_SEED, failures, model_alias, "all", f"{track}.bootstrap_seed")
    _check(_int(manifest["bootstrap_resamples"], "manifest.bootstrap_resamples") == BOOTSTRAP_RESAMPLES, failures, model_alias, "all", f"{track}.bootstrap_resamples")
    expected_claim_boundary = (
        MEMIT_CLAIM_BOUNDARY if track == "memit" else ALPHA_CLAIM_BOUNDARY
    )
    _check(
        manifest["claim_boundary"] == expected_claim_boundary,
        failures,
        model_alias,
        "all",
        f"{track}.claim_boundary",
    )
    source = lock["models"][model_alias]
    if track == "memit":
        _check(manifest_identity == source["source_manifest"], failures, model_alias, "all", "source_manifest_identity")
        _check(summary_identity == source["source_summary"], failures, model_alias, "all", "source_summary_identity")
    else:
        _required_keys(
            manifest,
            {"paired_memit_run_id", "paired_memit_manifest_sha256", "replay_lock_id"},
            f"{model_alias}.alpha.manifest",
        )
        _check(manifest["paired_memit_run_id"] == source["source_run_id"], failures, model_alias, "all", "alpha.paired_memit_run_id")
        _check(_sha256(manifest["paired_memit_manifest_sha256"], "alpha.manifest.paired_memit_manifest_sha256") == source["source_manifest"]["sha256"], failures, model_alias, "all", "alpha.paired_memit_manifest")
        _check(_sha256(manifest["replay_lock_id"], "alpha.manifest.replay_lock_id") == lock["lock_id"], failures, model_alias, "all", "alpha.replay_lock_id")
    _required_keys(
        summary,
        {
            "schema_version",
            "run_id",
            "model_alias",
            "planned_case_count",
            "attempted_case_count",
            "pass_case_count",
            "failed_case_count",
            "outcome_count",
            "arm_order",
            "selection_sha256",
            "config_sha256",
            "all_rollbacks_exact",
            "firewall_pass",
            "receipt_before_outcome",
        },
        f"{model_alias}.{track}.summary",
    )
    _check(summary["schema_version"] == expected_summary_schema, failures, model_alias, "all", f"{track}.summary.schema")
    _check(summary["run_id"] == expected_run and summary["model_alias"] == model_alias, failures, model_alias, "all", f"{track}.summary.identity")
    expected_outcomes = EXPECTED_CASE_COUNT * len(expected_arms)
    counts_ok = (
        _int(summary["planned_case_count"], "summary.planned_case_count") == EXPECTED_CASE_COUNT
        and _int(summary["attempted_case_count"], "summary.attempted_case_count") == EXPECTED_CASE_COUNT
        and _int(summary["pass_case_count"], "summary.pass_case_count") == EXPECTED_CASE_COUNT
        and _int(summary["failed_case_count"], "summary.failed_case_count") == 0
        and _int(summary["outcome_count"], "summary.outcome_count") == expected_outcomes
    )
    _check(counts_ok, failures, model_alias, "all", f"{track}.summary.counts")
    _check(isinstance(summary["arm_order"], list) and tuple(summary["arm_order"]) == expected_arms, failures, model_alias, "all", f"{track}.summary.arm_order")
    _check(summary["selection_sha256"] == manifest["selection_sha256"] and summary["config_sha256"] == manifest["config_sha256"], failures, model_alias, "all", f"{track}.summary.manifest_identity")
    for flag in ("all_rollbacks_exact", "firewall_pass", "receipt_before_outcome"):
        _check(_bool(summary[flag], f"summary.{flag}"), failures, model_alias, "all", f"{track}.summary.{flag}")
    if track == "memit":
        _required_keys(summary, {"direct_z_once_per_case"}, "memit.summary")
        _check(_bool(summary["direct_z_once_per_case"], "memit.summary.direct_z_once_per_case"), failures, model_alias, "all", "memit.summary.direct_z_once_per_case")
    else:
        required_flags = (
            "frozen_target_replay_exact",
            "frozen_target_load_once_per_case",
            "projector_precomputed_only",
            "projector_integrity_exact",
            "genuine_alpha_ordered_solve_once_per_case",
            "genuine_alpha_bf_all_hops_genuine",
            "posthoc_arms_projection_only",
        )
        _required_keys(
            summary,
            set(required_flags)
            | {
                "direct_z_recompute_count_total",
                "paired_memit_run_id",
                "paired_memit_manifest_sha256",
                "replay_lock_id",
            },
            "alpha.summary",
        )
        for flag in required_flags:
            _check(_bool(summary[flag], f"alpha.summary.{flag}"), failures, model_alias, "all", f"alpha.summary.{flag}")
        _check(_int(summary["direct_z_recompute_count_total"], "alpha.summary.direct_z_recompute_count_total") == 0, failures, model_alias, "all", "alpha.summary.direct_z_recompute_count_total")
        _check(summary["paired_memit_run_id"] == source["source_run_id"], failures, model_alias, "all", "alpha.summary.paired_memit_run_id")
        _check(summary["paired_memit_manifest_sha256"] == source["source_manifest"]["sha256"], failures, model_alias, "all", "alpha.summary.paired_memit_manifest")
        _check(summary["replay_lock_id"] == lock["lock_id"], failures, model_alias, "all", "alpha.summary.replay_lock_id")
    return manifest, summary, {
        "manifest_json": manifest_identity["sha256"],
        "summary_json": summary_identity["sha256"],
    }


def _validate_stream_envelope(
    row: Mapping[str, Any],
    *,
    path: str,
    schema: str,
    run_id: str,
    event: str,
    sequence: int,
) -> Mapping[str, Any]:
    _exact_keys(row, _ENVELOPE_KEYS, path)
    if (
        row["schema_version"] != schema
        or _run_id(row["run_id"], f"{path}.run_id") != run_id
        or _int(row["sequence"], f"{path}.sequence") != sequence
        or row["event"] != event
    ):
        raise DirectZCrossTrackAnalysisError(f"{path}: envelope identity/order mismatch")
    _timestamp(row["recorded_at"], f"{path}.recorded_at")
    return _mapping(row["payload"], f"{path}.payload")


def _validate_features(
    root: Path,
    *,
    track: str,
    model_alias: str,
    lock: Mapping[str, Any],
    failures: list[str],
) -> tuple[list[dict[str, Any]], str]:
    path = root / "features.jsonl"
    rows = _read_jsonl(path, f"{model_alias}.{track}.features")
    if len(rows) != EXPECTED_CASE_COUNT:
        raise DirectZCrossTrackAnalysisError(f"{model_alias}.{track}.features: exact count required")
    stream_schema = MEMIT_STREAM_SCHEMA if track == "memit" else ALPHA_STREAM_SCHEMA
    event_name = "direct_z_possibility_feature" if track == "memit" else "direct_z_alpha_feature"
    artifacts = lock["models"][model_alias]["artifacts"]
    normalized: list[dict[str, Any]] = []
    parameter_identity: str | None = None
    context_id: str | None = None
    model_id_sha256: str | None = None
    solver_identity: tuple[str, str, str] | None = None
    state_ids: set[str] = set()
    snapshot_ids: set[str] = set()
    for index, row in enumerate(rows):
        payload = _validate_stream_envelope(
            row,
            path=f"{model_alias}.{track}.features[{index}]",
            schema=stream_schema,
            run_id=RUN_IDS[model_alias][track],
            event=event_name,
            sequence=index,
        )
        _required_keys(payload, {"case_id", "request_id", "feature_hash", "target_identity"}, f"features[{index}].payload")
        artifact = artifacts[index]
        case_id = _case_id(payload["case_id"], f"features[{index}].case_id")
        request_id = _sha256(payload["request_id"], f"features[{index}].request_id")
        _check(case_id == artifact["case_id"], failures, model_alias, case_id, f"{track}.feature.case_order")
        _check(request_id == artifact["request_id"], failures, model_alias, case_id, f"{track}.feature.request_id")
        feature_hash = _sha256(payload["feature_hash"], f"features[{index}].feature_hash")
        feature_body = dict(payload)
        del feature_body["feature_hash"]
        _check(feature_hash == _canonical_sha256(feature_body), failures, model_alias, case_id, f"{track}.feature_hash")
        target_identity = _mapping(payload["target_identity"], f"features[{index}].target_identity")
        if track == "memit":
            _exact_keys(
                target_identity,
                {
                    "direct_z_artifact_sha256",
                    "direct_z_compute_count",
                    "direct_z_tensor_sha256",
                    "origin_lineage_id",
                    "target_token_sha256",
                },
                f"features[{index}].target_identity",
            )
            _check(_sha256(target_identity["direct_z_artifact_sha256"], "memit.target.artifact") == artifact["artifact_sha256"], failures, model_alias, case_id, "memit.direct_z_artifact")
            _check(_int(target_identity["direct_z_compute_count"], "memit.target.compute_count") == 1, failures, model_alias, case_id, "memit.direct_z_compute_count")
            _check(_sha256(target_identity["direct_z_tensor_sha256"], "memit.target.tensor") == artifact["tensor_sha256"], failures, model_alias, case_id, "memit.direct_z_tensor")
            _check(_sha256(target_identity["origin_lineage_id"], "memit.target.lineage") == artifact["origin_lineage_id"], failures, model_alias, case_id, "memit.W0_lineage")
            _check(_sha256(target_identity["target_token_sha256"], "memit.target.target_digest") == artifact["target_token_sha256"], failures, model_alias, case_id, "memit.target_digest")
            normalized.append(
                {
                    "case_id": case_id,
                    "request_id": request_id,
                    "artifact_sha256": artifact["artifact_sha256"],
                    "tensor_sha256": artifact["tensor_sha256"],
                    "origin_lineage_id": artifact["origin_lineage_id"],
                    "target_digest": artifact["target_token_sha256"],
                }
            )
            continue
        _required_keys(payload, {"target_anchor", "reference_c_energy"}, f"alpha.features[{index}].payload")
        anchor = _mapping(payload["target_anchor"], f"alpha.features[{index}].target_anchor")
        anchor_keys = {
            "alpha_solver_config_id",
            "alpha_solver_yaml_sha256",
            "context_id",
            "cross_solver_frozen_target_anchor",
            "direct_z_artifact_sha256",
            "direct_z_artifact_size",
            "direct_z_compute_count",
            "direct_z_load_count",
            "direct_z_tensor_sha256",
            "memit_target_hparams_sha256",
            "model_id",
            "origin_lineage_id",
            "origin_parameter_hashes",
            "origin_snapshot_id",
            "origin_state_id",
            "request_ids",
            "source_manifest_sha256",
            "source_run_id",
            "source_summary_sha256",
            "target_anchor_id",
            "target_token_sha256",
        }
        _exact_keys(anchor, anchor_keys, f"alpha.features[{index}].target_anchor")
        target_anchor_id = _sha256(anchor["target_anchor_id"], "alpha.anchor.target_anchor_id")
        anchor_body = dict(anchor)
        del anchor_body["target_anchor_id"]
        _check(target_anchor_id == _canonical_sha256(anchor_body), failures, model_alias, case_id, "alpha.target_anchor_hash")
        _exact_keys(
            target_identity,
            {"direct_z_tensor_sha256", "origin_lineage_id", "target_anchor_id", "target_token_sha256"},
            f"alpha.features[{index}].target_identity",
        )
        source = lock["models"][model_alias]
        comparisons = {
            "alpha.direct_z_artifact": _sha256(anchor["direct_z_artifact_sha256"], "alpha.anchor.artifact") == artifact["artifact_sha256"],
            "alpha.direct_z_artifact_size": _int(anchor["direct_z_artifact_size"], "alpha.anchor.artifact_size", positive=True) == artifact["artifact_size"],
            "alpha.direct_z_tensor": _sha256(anchor["direct_z_tensor_sha256"], "alpha.anchor.tensor") == artifact["tensor_sha256"],
            "alpha.W0_lineage": _sha256(anchor["origin_lineage_id"], "alpha.anchor.lineage") == artifact["origin_lineage_id"],
            "alpha.target_digest": _sha256(anchor["target_token_sha256"], "alpha.anchor.target_digest") == artifact["target_token_sha256"],
            "alpha.source_manifest": _sha256(anchor["source_manifest_sha256"], "alpha.anchor.source_manifest") == source["source_manifest"]["sha256"],
            "alpha.source_summary": _sha256(anchor["source_summary_sha256"], "alpha.anchor.source_summary") == source["source_summary"]["sha256"],
            "alpha.source_run": anchor["source_run_id"] == source["source_run_id"],
            "alpha.request_ids": isinstance(anchor["request_ids"], list) and anchor["request_ids"] == [request_id],
            "alpha.cross_solver_anchor": _bool(anchor["cross_solver_frozen_target_anchor"], "alpha.anchor.cross_solver_frozen_target_anchor"),
            "alpha.direct_z_compute_count": _int(anchor["direct_z_compute_count"], "alpha.anchor.compute_count") == 0,
            "alpha.direct_z_load_count": _int(anchor["direct_z_load_count"], "alpha.anchor.load_count") == 1,
            "alpha.target_identity.tensor": _sha256(target_identity["direct_z_tensor_sha256"], "alpha.target_identity.tensor") == artifact["tensor_sha256"],
            "alpha.target_identity.lineage": _sha256(target_identity["origin_lineage_id"], "alpha.target_identity.lineage") == artifact["origin_lineage_id"],
            "alpha.target_identity.target": _sha256(target_identity["target_token_sha256"], "alpha.target_identity.target") == artifact["target_token_sha256"],
            "alpha.target_identity.anchor": _sha256(target_identity["target_anchor_id"], "alpha.target_identity.anchor") == target_anchor_id,
        }
        for field, condition in comparisons.items():
            _check(condition, failures, model_alias, case_id, field)
        reference = _finite(payload["reference_c_energy"], "alpha.feature.reference_c_energy", nonnegative=True)
        _check(math.isclose(reference, artifact["reference_c_energy"], rel_tol=C_ENERGY_REL_TOL, abs_tol=C_ENERGY_ABS_TOL), failures, model_alias, case_id, "alpha.reference_c_energy")
        current_context = _sha256(anchor["context_id"], "alpha.anchor.context_id")
        current_state = _sha256(anchor["origin_state_id"], "alpha.anchor.origin_state_id")
        current_snapshot = _sha256(anchor["origin_snapshot_id"], "alpha.anchor.origin_snapshot_id")
        parameter_hashes = _mapping(anchor["origin_parameter_hashes"], "alpha.anchor.origin_parameter_hashes")
        if not parameter_hashes:
            raise DirectZCrossTrackAnalysisError("alpha.anchor.origin_parameter_hashes must not be empty")
        normalized_parameters = {
            str(name): _sha256(value, f"alpha.anchor.origin_parameter_hashes.{name}")
            for name, value in parameter_hashes.items()
            if isinstance(name, str) and name
        }
        if len(normalized_parameters) != len(parameter_hashes):
            raise DirectZCrossTrackAnalysisError("alpha.anchor origin parameter names invalid")
        current_parameter_identity = _canonical_sha256(normalized_parameters)
        if context_id is None:
            context_id = current_context
            parameter_identity = current_parameter_identity
        _check(current_context == context_id, failures, model_alias, case_id, "alpha.context_id")
        _check(current_parameter_identity == parameter_identity, failures, model_alias, case_id, "alpha.W0_parameter_identity")
        _check(current_state not in state_ids, failures, model_alias, case_id, "alpha.source_state_uniqueness")
        _check(current_snapshot not in snapshot_ids, failures, model_alias, case_id, "alpha.source_snapshot_uniqueness")
        state_ids.add(current_state)
        snapshot_ids.add(current_snapshot)
        if not isinstance(anchor["model_id"], str) or not anchor["model_id"]:
            raise DirectZCrossTrackAnalysisError("alpha.anchor.model_id required")
        current_model_id_sha256 = hashlib.sha256(
            anchor["model_id"].encode("utf-8")
        ).hexdigest()
        current_solver_identity = (
            _sha256(
                anchor["alpha_solver_config_id"],
                "alpha.anchor.alpha_solver_config_id",
            ),
            _sha256(
                anchor["alpha_solver_yaml_sha256"],
                "alpha.anchor.alpha_solver_yaml_sha256",
            ),
            _sha256(
                anchor["memit_target_hparams_sha256"],
                "alpha.anchor.memit_target_hparams_sha256",
            ),
        )
        if model_id_sha256 is None:
            model_id_sha256 = current_model_id_sha256
            solver_identity = current_solver_identity
        _check(
            current_model_id_sha256 == model_id_sha256,
            failures,
            model_alias,
            case_id,
            "alpha.model_id_internal_consistency",
        )
        _check(
            current_solver_identity == solver_identity,
            failures,
            model_alias,
            case_id,
            "alpha.solver_ids_internal_consistency",
        )
        alpha_internal = {
            "schema_version": "ode-edit-direct-z-alpha-self-bound-provenance/v1",
            "model_alias": model_alias,
            "case_id": case_id,
            "request_id": request_id,
            "target_anchor_sha256": target_anchor_id,
            "model_id_sha256": current_model_id_sha256,
            "context_sha256": current_context,
            "source_state_sha256": current_state,
            "source_snapshot_sha256": current_snapshot,
            "origin_parameter_set_sha256": current_parameter_identity,
            "alpha_solver_config_sha256": current_solver_identity[0],
            "alpha_solver_yaml_sha256": current_solver_identity[1],
            "memit_target_hparams_sha256": current_solver_identity[2],
        }
        normalized.append(
            {
                "case_id": case_id,
                "request_id": request_id,
                "artifact_sha256": artifact["artifact_sha256"],
                "artifact_size": artifact["artifact_size"],
                "tensor_sha256": artifact["tensor_sha256"],
                "origin_lineage_id": artifact["origin_lineage_id"],
                "target_digest": artifact["target_token_sha256"],
                "target_anchor_id": target_anchor_id,
                "alpha_self_bound_provenance": alpha_internal,
            }
        )
    return normalized, _file_identity(path, "features")["sha256"]


def _validate_events(
    root: Path,
    *,
    track: str,
    model_alias: str,
    lock: Mapping[str, Any],
    failures: list[str],
) -> tuple[list[dict[str, Any]], str]:
    path = root / "events.jsonl"
    rows = _read_jsonl(path, f"{model_alias}.{track}.events")
    if len(rows) != EXPECTED_CASE_COUNT:
        raise DirectZCrossTrackAnalysisError(f"{model_alias}.{track}.events: exact count required")
    schema = MEMIT_STREAM_SCHEMA if track == "memit" else ALPHA_STREAM_SCHEMA
    event_name = "direct_z_possibility_case" if track == "memit" else "direct_z_alpha_case"
    artifacts = lock["models"][model_alias]["artifacts"]
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        payload = _validate_stream_envelope(
            row,
            path=f"{model_alias}.{track}.events[{index}]",
            schema=schema,
            run_id=RUN_IDS[model_alias][track],
            event=event_name,
            sequence=index,
        )
        required = {
            "schema_version",
            "case_id",
            "request_id",
            "evaluation_payload_sha256",
            "feature_count",
            "outcome_count",
            "pass",
            "technical",
        }
        _required_keys(payload, required, f"events[{index}].payload")
        artifact = artifacts[index]
        case_id = _case_id(payload["case_id"], f"events[{index}].case_id")
        request_id = _sha256(payload["request_id"], f"events[{index}].request_id")
        _check(case_id == artifact["case_id"], failures, model_alias, case_id, f"{track}.event.case_order")
        _check(request_id == artifact["request_id"], failures, model_alias, case_id, f"{track}.event.request_id")
        expected_payload_schema = "ode-edit-direct-z-possibility-event/v2" if track == "memit" else "ode-edit-direct-z-alpha-paired-event/v1"
        _check(payload["schema_version"] == expected_payload_schema, failures, model_alias, case_id, f"{track}.event.schema")
        _check(_int(payload["feature_count"], "event.feature_count") == 1, failures, model_alias, case_id, f"{track}.event.feature_count")
        _check(_int(payload["outcome_count"], "event.outcome_count") == (len(MEMIT_ARM_ORDER) if track == "memit" else len(ALPHA_ARM_ORDER)), failures, model_alias, case_id, f"{track}.event.outcome_count")
        _check(_bool(payload["pass"], "event.pass"), failures, model_alias, case_id, f"{track}.event.pass")
        technical = _mapping(payload["technical"], f"events[{index}].technical")
        for name, value in technical.items():
            if type(value) is bool:
                _check(value, failures, model_alias, case_id, f"{track}.event.technical.{name}")
        if track == "memit":
            _required_keys(payload, {"direct_z_compute_count"}, "memit.event")
            _check(_int(payload["direct_z_compute_count"], "memit.event.direct_z_compute_count") == 1, failures, model_alias, case_id, "memit.event.direct_z_compute_count")
        else:
            _required_keys(payload, {"direct_z_compute_count", "direct_z_load_count"}, "alpha.event")
            _check(_int(payload["direct_z_compute_count"], "alpha.event.direct_z_compute_count") == 0, failures, model_alias, case_id, "alpha.event.direct_z_compute_count")
            _check(_int(payload["direct_z_load_count"], "alpha.event.direct_z_load_count") == 1, failures, model_alias, case_id, "alpha.event.direct_z_load_count")
        normalized.append(
            {
                "case_id": case_id,
                "request_id": request_id,
                "heldout_payload_sha256": _sha256(
                    payload["evaluation_payload_sha256"],
                    f"events[{index}].evaluation_payload_sha256",
                ),
            }
        )
    return normalized, _file_identity(path, "events")["sha256"]


def _normalize_outcome_metrics(payload: Mapping[str, Any], path: str) -> dict[str, float]:
    _required_keys(payload, _COMMON_OUTCOME_KEYS, path)
    heldout_kl = _finite(payload["heldout_kl"], f"{path}.heldout_kl", nonnegative=True)
    preservation = _finite(payload["preservation_score"], f"{path}.preservation_score")
    if not math.isclose(preservation, -heldout_kl, rel_tol=0.0, abs_tol=PRESERVATION_ABS_TOL):
        raise DirectZCrossTrackAnalysisError(f"{path}: preservation_score must equal -heldout_kl")
    cosine = _finite(payload["delta_cosine"], f"{path}.delta_cosine")
    if cosine < -1.0 or cosine > 1.0:
        raise DirectZCrossTrackAnalysisError(f"{path}: delta_cosine outside [-1,1]")
    metrics = {
        "z_residual_ratio": _finite(payload["z_residual_ratio"], f"{path}.z_residual_ratio", nonnegative=True),
        "generated_delta_error_mean": _finite(payload["generated_delta_error_mean"], f"{path}.generated_delta_error_mean", nonnegative=True),
        "generated_delta_error_worst": _finite(payload["generated_delta_error_worst"], f"{path}.generated_delta_error_worst", nonnegative=True),
        "delta_gain": _finite(payload["delta_gain"], f"{path}.delta_gain"),
        "delta_cosine": cosine,
        "off_token_spill_ratio": _finite(payload["off_token_spill_ratio"], f"{path}.off_token_spill_ratio", nonnegative=True),
        "output_progress": _finite(payload["output_progress"], f"{path}.output_progress"),
        "output_nll_reduction": _finite(payload["output_nll_reduction"], f"{path}.output_nll_reduction"),
        "exact_margin_min": _finite(payload["exact_margin_min"], f"{path}.exact_margin_min"),
        "exact_satisfied": float(_bool(payload["exact_satisfied"], f"{path}.exact_satisfied")),
        "paraphrase_nll_reduction": _finite(payload["paraphrase_nll_reduction"], f"{path}.paraphrase_nll_reduction"),
        "heldout_kl": heldout_kl,
        "endpoint_frobenius_norm": _finite(payload["endpoint_frobenius_norm"], f"{path}.endpoint_frobenius_norm", nonnegative=True),
        "nfe": float(_int(payload["nfe"], f"{path}.nfe")),
        "endpoint_c_energy": _finite(payload["endpoint_c_energy"], f"{path}.endpoint_c_energy", nonnegative=True),
    }
    return metrics


def _c_matches(value: float, reference: float) -> bool:
    return math.isclose(
        value,
        reference,
        rel_tol=C_ENERGY_REL_TOL,
        abs_tol=C_ENERGY_ABS_TOL,
    )


def _c_within_cap(value: float, reference: float) -> bool:
    return value <= reference * (1.0 + C_ENERGY_REL_TOL) + C_ENERGY_ABS_TOL


def _validate_outcomes(
    root: Path,
    *,
    track: str,
    model_alias: str,
    lock: Mapping[str, Any],
    alpha_features: Sequence[Mapping[str, Any]] | None,
    failures: list[str],
) -> tuple[list[dict[str, Any]], str]:
    path = root / "outcomes.jsonl"
    rows = _read_jsonl(path, f"{model_alias}.{track}.outcomes")
    arm_order = MEMIT_ARM_ORDER if track == "memit" else ALPHA_ARM_ORDER
    expected_count = EXPECTED_CASE_COUNT * len(arm_order)
    if len(rows) != expected_count:
        raise DirectZCrossTrackAnalysisError(f"{model_alias}.{track}.outcomes: exact count required")
    schema = MEMIT_STREAM_SCHEMA if track == "memit" else ALPHA_STREAM_SCHEMA
    event_name = "direct_z_possibility_outcome" if track == "memit" else "direct_z_alpha_outcome"
    artifacts = lock["models"][model_alias]["artifacts"]
    selected = {MEMIT_NATIVE, MEMIT_BF} if track == "memit" else {ALPHA_NATIVE, ALPHA_BF}
    cases: list[dict[str, Any]] = []
    for case_index, artifact in enumerate(artifacts):
        arms: dict[str, dict[str, float]] = {}
        for arm_index, expected_arm in enumerate(arm_order):
            sequence = case_index * len(arm_order) + arm_index
            payload = _validate_stream_envelope(
                rows[sequence],
                path=f"{model_alias}.{track}.outcomes[{sequence}]",
                schema=schema,
                run_id=RUN_IDS[model_alias][track],
                event=event_name,
                sequence=sequence,
            )
            path_label = f"outcomes[{sequence}].payload"
            _required_keys(payload, _COMMON_OUTCOME_KEYS, path_label)
            case_id = _case_id(payload["case_id"], f"{path_label}.case_id")
            request_id = _sha256(payload["request_id"], f"{path_label}.request_id")
            _check(case_id == artifact["case_id"], failures, model_alias, artifact["case_id"], f"{track}.outcome.case_order")
            _check(request_id == artifact["request_id"], failures, model_alias, artifact["case_id"], f"{track}.outcome.request_id")
            _check(payload["arm_id"] == expected_arm, failures, model_alias, artifact["case_id"], f"{track}.outcome.arm_order")
            for flag in ("case_pass", "arm_success", "technical_pass", "rollback_exact", "firewall_pass", "receipt_before_outcome"):
                _check(_bool(payload[flag], f"{path_label}.{flag}"), failures, model_alias, artifact["case_id"], f"{track}.outcome.{flag}")
            metrics = _normalize_outcome_metrics(payload, path_label)
            if track == "alpha":
                _required_keys(
                    payload,
                    {
                        "target_anchor_id",
                        "reference_c_energy",
                        "arm_construction",
                        "frozen_target_replay_exact",
                        "projector_precomputed_only",
                        "projector_integrity_exact",
                        "genuine_alpha_solver_used",
                        "posthoc_b_at_p_only",
                    },
                    path_label,
                )
                assert alpha_features is not None
                _check(_sha256(payload["target_anchor_id"], f"{path_label}.target_anchor_id") == alpha_features[case_index]["target_anchor_id"], failures, model_alias, artifact["case_id"], "alpha.outcome.target_anchor")
                outcome_reference = _finite(payload["reference_c_energy"], f"{path_label}.reference_c_energy", nonnegative=True)
                _check(_c_matches(outcome_reference, artifact["reference_c_energy"]), failures, model_alias, artifact["case_id"], "alpha.outcome.reference_c_energy")
                _check(
                    payload["arm_construction"]
                    == ALPHA_ARM_CONSTRUCTION[expected_arm],
                    failures,
                    model_alias,
                    artifact["case_id"],
                    "alpha.outcome.arm_construction",
                )
                for flag in (
                    "frozen_target_replay_exact",
                    "projector_precomputed_only",
                    "projector_integrity_exact",
                ):
                    _check(
                        _bool(payload[flag], f"{path_label}.{flag}"),
                        failures,
                        model_alias,
                        artifact["case_id"],
                        f"alpha.outcome.{flag}",
                    )
                _check(
                    _bool(
                        payload["genuine_alpha_solver_used"],
                        f"{path_label}.genuine_alpha_solver_used",
                    )
                    == (expected_arm in ALPHA_GENUINE_ARMS),
                    failures,
                    model_alias,
                    artifact["case_id"],
                    "alpha.outcome.genuine_alpha_solver_used",
                )
                _check(
                    _bool(
                        payload["posthoc_b_at_p_only"],
                        f"{path_label}.posthoc_b_at_p_only",
                    )
                    == (expected_arm in ALPHA_POSTHOC_ARMS),
                    failures,
                    model_alias,
                    artifact["case_id"],
                    "alpha.outcome.posthoc_b_at_p_only",
                )
            if expected_arm in selected:
                reference = artifact["reference_c_energy"]
                _check(_c_matches(metrics["endpoint_c_energy"], reference), failures, model_alias, artifact["case_id"], f"{track}.{expected_arm}.C_energy")
                _check(_c_within_cap(metrics["endpoint_c_energy"], reference), failures, model_alias, artifact["case_id"], f"{track}.{expected_arm}.C_energy_cap")
                arms[expected_arm] = metrics
        if set(arms) != selected:
            raise DirectZCrossTrackAnalysisError(f"{model_alias}.{track}: selected arm missing")
        cases.append(
            {
                "case_id": artifact["case_id"],
                "request_id": artifact["request_id"],
                "arms": arms,
            }
        )
    return cases, _file_identity(path, "outcomes")["sha256"]


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _joint_bootstrap_plan() -> tuple[tuple[int, ...], ...]:
    rng = random.Random(BOOTSTRAP_SEED)
    return tuple(
        tuple(rng.randrange(EXPECTED_CASE_COUNT) for _ in range(EXPECTED_CASE_COUNT))
        for _ in range(BOOTSTRAP_RESAMPLES)
    )


def _effect_summary(
    values: Sequence[float], bootstrap_plan: Sequence[Sequence[int]]
) -> dict[str, Any]:
    if len(values) != EXPECTED_CASE_COUNT or any(not math.isfinite(value) for value in values):
        raise DirectZCrossTrackAnalysisError("effect summary denominator/value mismatch")
    mean = math.fsum(values) / EXPECTED_CASE_COUNT
    positive = sum(value > 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    estimates = sorted(
        math.fsum(values[index] for index in sample) / EXPECTED_CASE_COUNT
        for sample in bootstrap_plan
    )
    return {
        "n_itd": EXPECTED_CASE_COUNT,
        "mean": mean,
        "median": statistics.median(values),
        "positive_count": positive,
        "negative_count": negative,
        "zero_count": EXPECTED_CASE_COUNT - positive - negative,
        "lenient_positive": mean > 0.0 and positive >= STRICT_MAJORITY,
        "lenient_negative": mean < 0.0 and negative >= STRICT_MAJORITY,
        "case_effects": list(values),
        "joint_case_bootstrap_mean_ci95": [
            _quantile(estimates, 0.025),
            _quantile(estimates, 0.975),
        ],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "ci_exclusion_required_for_gate": False,
    }


def _oriented(value: float, direction: str) -> float:
    return value if direction == "higher_is_better" else -value


def _build_effects(
    memit_cases: Sequence[Mapping[str, Any]],
    alpha_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bootstrap_plan = _joint_bootstrap_plan()
    effects: dict[str, Any] = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        memit_lift: list[float] = []
        alpha_lift: list[float] = []
        interaction: list[float] = []
        endpoint: list[float] = []
        for index in range(EXPECTED_CASE_COUNT):
            memit_native = _oriented(
                float(memit_cases[index]["arms"][MEMIT_NATIVE][metric]), direction
            )
            memit_bf = _oriented(
                float(memit_cases[index]["arms"][MEMIT_BF][metric]), direction
            )
            alpha_native = _oriented(
                float(alpha_cases[index]["arms"][ALPHA_NATIVE][metric]), direction
            )
            alpha_bf = _oriented(
                float(alpha_cases[index]["arms"][ALPHA_BF][metric]), direction
            )
            m_lift = memit_bf - memit_native
            a_lift = alpha_bf - alpha_native
            memit_lift.append(m_lift)
            alpha_lift.append(a_lift)
            interaction.append(a_lift - m_lift)
            endpoint.append(alpha_bf - memit_bf)
        effects[metric] = {
            "direction": direction,
            "sign_normalized": True,
            "memit_bf_minus_native_lift": _effect_summary(memit_lift, bootstrap_plan),
            "alpha_bf_minus_native_lift": _effect_summary(alpha_lift, bootstrap_plan),
            "alpha_minus_memit_bf_native_2x2_interaction": _effect_summary(interaction, bootstrap_plan),
            "alpha_bf_minus_memit_bf_endpoint_contrast": _effect_summary(endpoint, bootstrap_plan),
        }
    return effects


def _classification(effects: Mapping[str, Any], valid: bool) -> dict[str, Any]:
    if not valid:
        return {
            "verdict": "BLOCK_CROSS_BOUND_JOIN_ALPHA_SELF_BINDING_OR_C_BUDGET_INVALID",
            "technical_pass": False,
            "axis_level_lenient_positive": {},
        }
    contrast_keys = (
        "memit_bf_minus_native_lift",
        "alpha_bf_minus_native_lift",
        "alpha_minus_memit_bf_native_2x2_interaction",
        "alpha_bf_minus_memit_bf_endpoint_contrast",
    )
    signals = {
        contrast: [
            metric
            for metric in METRIC_DIRECTIONS
            if effects[metric][contrast]["lenient_positive"]
        ]
        for contrast in contrast_keys
    }
    return {
        "verdict": "CROSS_TRACK_AXIS_SIGNALS_REPORTED_NO_SUPERIORITY_DECISION",
        "technical_pass": True,
        "axis_level_lenient_positive": signals,
        "gate": {
            "requires_positive_mean": True,
            "strict_majority": STRICT_MAJORITY,
            "itd_denominator": EXPECTED_CASE_COUNT,
            "requires_ci_exclusion": False,
            "model_pooling": False,
            "method_superiority_decision": False,
        },
    }


def _join_model(
    *,
    model_alias: str,
    memit_root: Path,
    alpha_root: Path,
    lock: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    _, _, memit_hashes = _validate_manifest_summary(
        memit_root,
        track="memit",
        model_alias=model_alias,
        lock=lock,
        failures=failures,
    )
    _, _, alpha_hashes = _validate_manifest_summary(
        alpha_root,
        track="alpha",
        model_alias=model_alias,
        lock=lock,
        failures=failures,
    )
    memit_features, memit_features_hash = _validate_features(
        memit_root,
        track="memit",
        model_alias=model_alias,
        lock=lock,
        failures=failures,
    )
    alpha_features, alpha_features_hash = _validate_features(
        alpha_root,
        track="alpha",
        model_alias=model_alias,
        lock=lock,
        failures=failures,
    )
    memit_events, memit_events_hash = _validate_events(
        memit_root,
        track="memit",
        model_alias=model_alias,
        lock=lock,
        failures=failures,
    )
    alpha_events, alpha_events_hash = _validate_events(
        alpha_root,
        track="alpha",
        model_alias=model_alias,
        lock=lock,
        failures=failures,
    )
    memit_cases, memit_outcomes_hash = _validate_outcomes(
        memit_root,
        track="memit",
        model_alias=model_alias,
        lock=lock,
        alpha_features=None,
        failures=failures,
    )
    alpha_cases, alpha_outcomes_hash = _validate_outcomes(
        alpha_root,
        track="alpha",
        model_alias=model_alias,
        lock=lock,
        alpha_features=alpha_features,
        failures=failures,
    )
    cross_certificates: list[dict[str, Any]] = []
    alpha_internal_certificates: list[dict[str, Any]] = []
    for index, case_id in enumerate(EXPECTED_CASE_ORDER):
        memit_feature = memit_features[index]
        alpha_feature = alpha_features[index]
        memit_event = memit_events[index]
        alpha_event = alpha_events[index]
        artifact = lock["models"][model_alias]["artifacts"][index]
        comparisons = {
            "case_id": memit_feature["case_id"] == alpha_feature["case_id"] == memit_event["case_id"] == alpha_event["case_id"] == artifact["case_id"],
            "request_id": memit_feature["request_id"] == alpha_feature["request_id"] == memit_event["request_id"] == alpha_event["request_id"] == artifact["request_id"],
            "W0_lineage": memit_feature["origin_lineage_id"] == alpha_feature["origin_lineage_id"] == artifact["origin_lineage_id"],
            "direct_z_artifact": memit_feature["artifact_sha256"] == alpha_feature["artifact_sha256"] == artifact["artifact_sha256"],
            "direct_z_tensor": memit_feature["tensor_sha256"] == alpha_feature["tensor_sha256"] == artifact["tensor_sha256"],
            "target_digest": memit_feature["target_digest"] == alpha_feature["target_digest"] == artifact["target_token_sha256"],
            "heldout_identity": memit_event["heldout_payload_sha256"] == alpha_event["heldout_payload_sha256"],
        }
        for field, condition in comparisons.items():
            _check(condition, failures, model_alias, case_id, field)
        certificate_body = {
            "schema_version": "ode-edit-direct-z-cross-bound-join-certificate/v2",
            "model_alias": model_alias,
            "case_id": case_id,
            "request_id": artifact["request_id"],
            "replay_lock_id": lock["lock_id"],
            "source_manifest_sha256": lock["models"][model_alias]["source_manifest"]["sha256"],
            "source_summary_sha256": lock["models"][model_alias]["source_summary"]["sha256"],
            "w0_lineage_sha256": artifact["origin_lineage_id"],
            "direct_z_artifact_sha256": artifact["artifact_sha256"],
            "direct_z_artifact_size": artifact["artifact_size"],
            "direct_z_tensor_sha256": artifact["tensor_sha256"],
            "target_digest_sha256": artifact["target_token_sha256"],
            "cross_equal_heldout_payload_sha256": alpha_event[
                "heldout_payload_sha256"
            ],
        }
        cross_certificates.append(
            {
                "case_id": case_id,
                "request_id": artifact["request_id"],
                "certificate_sha256": _canonical_sha256(certificate_body),
                "identity_digests": {
                    "w0_lineage": artifact["origin_lineage_id"],
                    "direct_z_artifact": artifact["artifact_sha256"],
                    "direct_z_tensor": artifact["tensor_sha256"],
                    "target": artifact["target_token_sha256"],
                    "heldout_payload": alpha_event["heldout_payload_sha256"],
                },
            }
        )
        alpha_internal = alpha_feature["alpha_self_bound_provenance"]
        alpha_internal_certificates.append(
            {
                "case_id": case_id,
                "request_id": artifact["request_id"],
                "certificate_sha256": _canonical_sha256(alpha_internal),
                "identity_digests": {
                    key: value
                    for key, value in alpha_internal.items()
                    if key not in {"schema_version", "model_alias", "case_id", "request_id"}
                },
                "binding_scope": "alpha_internal_self_bound_not_replay_lock_pinned",
            }
        )
    valid = not failures
    effects = _build_effects(memit_cases, alpha_cases) if valid else {}
    return (
        {
            "run_ids": dict(RUN_IDS[model_alias]),
            "artifact_validation": {
                "valid": valid,
                "failures": failures,
                "case_count": EXPECTED_CASE_COUNT,
                "canonical_cross_bound_join_certificate_count": (
                    len(cross_certificates) if valid else 0
                ),
                "alpha_self_bound_provenance_certificate_count": (
                    len(alpha_internal_certificates) if valid else 0
                ),
                "raw_scalar_contract_revalidated": valid,
                "cross_bound_identity_fields": [
                    "case_id",
                    "request_id",
                    "source_manifest_summary",
                    "W0_lineage",
                    "direct_z_artifact_size_and_sha256",
                    "direct_z_tensor_sha256",
                    "target_token_sha256",
                    "cross_equal_heldout_payload_sha256",
                    "reference_C_energy",
                ],
                "alpha_internal_only_fields": [
                    "model_id",
                    "context_id",
                    "origin_state_id",
                    "origin_snapshot_id",
                    "origin_parameter_hashes",
                    "alpha_solver_ids",
                    "target_anchor_id",
                ],
                "heldout_binding_scope": "cross_track_and_cross_model_equality_not_replay_lock_pin",
                "C_energy_rel_tol": C_ENERGY_REL_TOL,
                "C_energy_abs_tol": C_ENERGY_ABS_TOL,
            },
            "canonical_cross_bound_join_certificates": (
                cross_certificates if valid else []
            ),
            "alpha_self_bound_provenance_certificates": (
                alpha_internal_certificates if valid else []
            ),
            "effects": effects,
            "classification": _classification(effects, valid),
            "source_hashes": {
                "memit": {
                    **memit_hashes,
                    "features_jsonl": memit_features_hash,
                    "events_jsonl": memit_events_hash,
                    "outcomes_jsonl": memit_outcomes_hash,
                },
                "alpha": {
                    **alpha_hashes,
                    "features_jsonl": alpha_features_hash,
                    "events_jsonl": alpha_events_hash,
                    "outcomes_jsonl": alpha_outcomes_hash,
                },
            },
        },
        failures,
    )


def analyze_cross_track(
    *,
    replay_lock: str | Path,
    memit_llama_directory: str | Path,
    alpha_llama_directory: str | Path,
    memit_qwen_directory: str | Path,
    alpha_qwen_directory: str | Path,
) -> dict[str, Any]:
    lock, lock_file_identity = _load_lock(replay_lock)
    directory_inputs = {
        "llama3-8b-inst": {
            "memit": memit_llama_directory,
            "alpha": alpha_llama_directory,
        },
        "qwen2.5-7b-inst": {
            "memit": memit_qwen_directory,
            "alpha": alpha_qwen_directory,
        },
    }
    panels: dict[str, Any] = {}
    failures_by_model: dict[str, list[str]] = {}
    for model_alias in MODEL_ALIASES:
        memit_root = _validate_run_root(directory_inputs[model_alias]["memit"], f"{model_alias}.memit")
        alpha_root = _validate_run_root(directory_inputs[model_alias]["alpha"], f"{model_alias}.alpha")
        panel, failures = _join_model(
            model_alias=model_alias,
            memit_root=memit_root,
            alpha_root=alpha_root,
            lock=lock,
        )
        panels[model_alias] = panel
        failures_by_model[model_alias] = failures
    # The panel identities must also describe the same heldout requests across
    # models.  This checks experimental pairing only; metrics remain separate.
    if all(panels[alias]["artifact_validation"]["valid"] for alias in MODEL_ALIASES):
        left = panels[MODEL_ALIASES[0]]["canonical_cross_bound_join_certificates"]
        right = panels[MODEL_ALIASES[1]]["canonical_cross_bound_join_certificates"]
        for index, case_id in enumerate(EXPECTED_CASE_ORDER):
            left_ids = left[index]["identity_digests"]
            right_ids = right[index]["identity_digests"]
            if left[index]["request_id"] != right[index]["request_id"]:
                for alias in MODEL_ALIASES:
                    _failure(failures_by_model[alias], alias, case_id, "cross_model.request_id")
            if left_ids["heldout_payload"] != right_ids["heldout_payload"]:
                for alias in MODEL_ALIASES:
                    _failure(failures_by_model[alias], alias, case_id, "cross_model.heldout_identity")
        for alias in MODEL_ALIASES:
            if failures_by_model[alias]:
                panels[alias]["artifact_validation"]["valid"] = False
                panels[alias]["artifact_validation"]["failures"] = failures_by_model[alias]
                panels[alias]["artifact_validation"][
                    "canonical_cross_bound_join_certificate_count"
                ] = 0
                panels[alias]["artifact_validation"][
                    "alpha_self_bound_provenance_certificate_count"
                ] = 0
                panels[alias]["canonical_cross_bound_join_certificates"] = []
                panels[alias]["alpha_self_bound_provenance_certificates"] = []
                panels[alias]["effects"] = {}
                panels[alias]["classification"] = _classification({}, False)
    valid = all(not failures_by_model[alias] for alias in MODEL_ALIASES)
    analysis: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA,
        "scope": "fixed_two_model_memit_alpha_direct_z_cross_track",
        "claim_boundary": CLAIM_BOUNDARY,
        "fixed_contract": {
            "models": list(MODEL_ALIASES),
            "run_ids": RUN_IDS,
            "case_order": list(EXPECTED_CASE_ORDER),
            "memit_arm_order": list(MEMIT_ARM_ORDER),
            "alpha_arm_order": list(ALPHA_ARM_ORDER),
            "selected_arms": {
                "memit_native": MEMIT_NATIVE,
                "memit_bf": MEMIT_BF,
                "alpha_native": ALPHA_NATIVE,
                "alpha_bf": ALPHA_BF,
            },
            "C_energy_rel_tol": C_ENERGY_REL_TOL,
            "C_energy_abs_tol": C_ENERGY_ABS_TOL,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "joint_case_resampling": True,
            "ci_exclusion_required_for_gate": False,
            "model_pooling": False,
        },
        "replay_lock": {
            "canonical_lock_id": lock["lock_id"],
            "file_sha256": lock_file_identity["sha256"],
            "selection_sha256": lock["selection_sha256"],
        },
        "artifact_validation": {
            "valid": valid,
            "failures_by_model": failures_by_model,
            "verdict": (
                "PASS_CROSS_BOUND_JOIN_ALPHA_SELF_BINDING_AND_C_BUDGET"
                if valid
                else "BLOCK_CROSS_BOUND_JOIN_ALPHA_SELF_BINDING_OR_C_BUDGET_INVALID"
            ),
        },
        "models": panels,
        "classification": {
            "verdict": (
                "CROSS_TRACK_AXIS_SIGNALS_REPORTED_NO_MODEL_POOLING"
                if valid
                else "BLOCK_CROSS_BOUND_JOIN_ALPHA_SELF_BINDING_OR_C_BUDGET_INVALID"
            ),
            "technical_pass": valid,
            "model_pooling": False,
            "interaction_is_distinct_from_endpoint_contrast": True,
            "method_superiority_decision": False,
            "lifelong_or_sequential_claim": False,
        },
        "interpretation_limits": [
            "models are analyzed separately; no pooled effect or pooled gate",
            "the 2x2 interaction measures differential BF lift, not absolute endpoint superiority",
            "the Alpha-BF minus MEMIT-BF endpoint contrast is reported separately",
            "bootstrap intervals are descriptive; positive mean plus 5/8 is the lenient gate",
            "Alpha context/state/snapshot/parameter/model/solver digests are self-bound internal provenance, not replay-lock-pinned cross identities",
            "heldout payload identity is cross-track/cross-model equal but not pinned by the replay lock",
            "atomic direct-z geometry only; no lifelong, collapse-prevention, or downstream guarantee",
            "no raw prompt, tensor value, logit, token text, or generation is serialized",
        ],
        "analysis_code_sha256": _file_identity(Path(__file__).resolve(strict=True), "analysis_code")["sha256"],
    }
    analysis["analysis_sha256"] = _canonical_sha256(analysis)
    return analysis


def render_markdown(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# Direct-z MEMIT × AlphaEdit cross-track analysis",
        "",
        f"- verdict: `{analysis['classification']['verdict']}`",
        f"- cross-bound join: `{analysis['artifact_validation']['verdict']}`",
        f"- replay lock: `{analysis['replay_lock']['canonical_lock_id']}`",
        f"- analysis SHA-256: `{analysis['analysis_sha256']}`",
        "- scope: fixed atomic geometry signal; no model pooling or method-superiority decision",
        "",
    ]
    for model_alias in MODEL_ALIASES:
        panel = analysis["models"][model_alias]
        lines.extend(
            [
                f"## {model_alias}",
                "",
                f"- technical validity: `{panel['artifact_validation']['valid']}`",
                f"- cross-bound join certificates: `{panel['artifact_validation']['canonical_cross_bound_join_certificate_count']}/{EXPECTED_CASE_COUNT}`",
                f"- Alpha self-bound provenance certificates: `{panel['artifact_validation']['alpha_self_bound_provenance_certificate_count']}/{EXPECTED_CASE_COUNT}`",
                f"- verdict: `{panel['classification']['verdict']}`",
                "",
            ]
        )
        if not panel["artifact_validation"]["valid"]:
            for failure in panel["artifact_validation"]["failures"]:
                lines.append(f"- BLOCK: `{failure}`")
            lines.append("")
            continue
        lines.extend(
            [
                "| Metric | MEMIT BF lift | Alpha BF lift | Alpha−MEMIT 2×2 interaction | AlphaBF−MEMITBF endpoint |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for metric in METRIC_DIRECTIONS:
            item = panel["effects"][metric]
            cells = []
            for key in (
                "memit_bf_minus_native_lift",
                "alpha_bf_minus_native_lift",
                "alpha_minus_memit_bf_native_2x2_interaction",
                "alpha_bf_minus_memit_bf_endpoint_contrast",
            ):
                effect = item[key]
                gate = "✓" if effect["lenient_positive"] else ""
                cells.append(f"{effect['mean']:.8g} ({effect['positive_count']}/8){gate}")
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
        lines.extend(
            [
                "",
                "Positive values are sign-normalized improvements. A check mark requires positive mean and at least 5/8 positive cases.",
                "The joint-case bootstrap CI is descriptive and is available in JSON.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "- The 2×2 interaction asks whether BF gains more over its native writer under AlphaEdit than under MEMIT.",
            "- The endpoint contrast separately asks whether the AlphaEdit BF endpoint itself beats the MEMIT BF endpoint.",
            "- A positive interaction does not imply a positive endpoint contrast.",
            "- The replay lock pins the cross-bound target/W0 identities; Alpha-only context/state/snapshot/parameter/solver digests are separately self-bound, not externally pinned by that lock.",
            "- Heldout identity is cross-track and cross-model equal, but is not replay-lock pinned.",
            "- These are fixed-panel atomic signals, not lifelong retention or collapse-prevention evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    analysis: Mapping[str, Any], output_json: str | Path, output_markdown: str | Path
) -> None:
    json_path = Path(output_json).expanduser().resolve()
    markdown_path = Path(output_markdown).expanduser().resolve()
    if json_path == markdown_path:
        raise DirectZCrossTrackAnalysisError("JSON and Markdown outputs must differ")
    for path in (json_path, markdown_path):
        if path.exists() or path.is_symlink():
            raise DirectZCrossTrackAnalysisError("analysis outputs are exclusive")
        if not path.parent.is_dir() or path.parent.is_symlink():
            raise DirectZCrossTrackAnalysisError("output parent must already exist")
    json_text = json.dumps(
        analysis,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    markdown_text = render_markdown(analysis)
    try:
        with json_path.open("x", encoding="utf-8") as handle:
            handle.write(json_text)
        with markdown_path.open("x", encoding="utf-8") as handle:
            handle.write(markdown_text)
    except OSError as exc:
        raise DirectZCrossTrackAnalysisError("exclusive analysis output failed") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ode-edit-direct-z-cross-track-analysis", allow_abbrev=False
    )
    parser.add_argument("--replay-lock", required=True)
    parser.add_argument("--memit-llama-directory", required=True)
    parser.add_argument("--alpha-llama-directory", required=True)
    parser.add_argument("--memit-qwen-directory", required=True)
    parser.add_argument("--alpha-qwen-directory", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        analysis = analyze_cross_track(
            replay_lock=args.replay_lock,
            memit_llama_directory=args.memit_llama_directory,
            alpha_llama_directory=args.alpha_llama_directory,
            memit_qwen_directory=args.memit_qwen_directory,
            alpha_qwen_directory=args.alpha_qwen_directory,
        )
        write_outputs(analysis, args.output_json, args.output_markdown)
    except (DirectZCrossTrackAnalysisError, FileNotFoundError) as exc:
        print(f"direct-z cross-track analysis rejected: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "verdict": analysis["classification"]["verdict"],
                "analysis_sha256": analysis["analysis_sha256"],
                "model_pooling": False,
            },
            sort_keys=True,
        )
    )
    return 0 if analysis["artifact_validation"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
