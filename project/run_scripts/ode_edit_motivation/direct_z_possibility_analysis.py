#!/usr/bin/env python3
"""Fail-closed analyzer for the direct-z possibility diagnostic.

This analyzer deliberately answers a narrow question: whether a frozen
direct-z target exposes an oracle ceiling, whether that target is reachable in
the locked synchronous proposal cone, and whether the current BF controller
leaves a directional objective gap.  It does *not* estimate method
superiority.

Only scalar, sanitized records are accepted.  Failed scientific cases remain
in the intention-to-diagnose denominator and contribute zero to every paired
contrast.  Technical failures block the possibility classification.
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
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "ode-edit-direct-z-possibility-manifest/v1"
SUMMARY_SCHEMA = "ode-edit-direct-z-possibility-summary/v1"
STREAM_SCHEMA = "ode-edit-direct-z-possibility-stream/v1"
STREAM_EVENT = "direct_z_possibility_outcome"
ANALYSIS_SCHEMA = "ode-edit-direct-z-possibility-analysis/v1"

EXPECTED_CASE_COUNT = 8
BOOTSTRAP_SEED = 20260801
BOOTSTRAP_RESAMPLES = 4000
STRICT_MAJORITY = EXPECTED_CASE_COUNT // 2 + 1
CLAIM_BOUNDARY = "possibility_only_not_method_superiority"

MODEL_ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")
ARM_ORDER = (
    "no_op_replay",
    "oracle_do_z",
    "native_ordered_full",
    "native_alpha_c_matched",
    "bf_current_refreshed_k4",
    "sync_z_cone_oracle",
)
EXPECTED_OUTCOME_COUNT = EXPECTED_CASE_COUNT * len(ARM_ORDER)

NO_OP = ARM_ORDER[0]
ORACLE_DO_Z = ARM_ORDER[1]
NATIVE_FULL = ARM_ORDER[2]
NATIVE_ALPHA = ARM_ORDER[3]
BF_CURRENT = ARM_ORDER[4]
SYNC_CONE = ARM_ORDER[5]

C_MATCH_REL_TOL = 3e-5
C_MATCH_ABS_TOL = 3e-5
ZERO_ABS_TOL = 1e-12
PRESERVATION_ABS_TOL = 1e-12

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,127}$")
_UNSAFE_KEY_FRAGMENTS = (
    "prompt",
    "tensor",
    "logit",
    "token",
    "generation",
    "weight",
)

_MANIFEST_FIELDS = {
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
}
_SUMMARY_FIELDS = {
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
    "direct_z_once_per_case",
}
_ENVELOPE_FIELDS = {
    "schema_version",
    "run_id",
    "sequence",
    "recorded_at",
    "event",
    "payload",
}
_OUTCOME_FIELDS = {
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


class DirectZPossibilityAnalysisError(ValueError):
    """The sanitized run violates the precommitted analysis contract."""


def _reject_nonstandard_constant(value: str) -> None:
    raise DirectZPossibilityAnalysisError(f"non-finite JSON constant rejected: {value}")


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    extra = actual - expected
    for key in extra:
        lowered = key.lower()
        if any(fragment in lowered for fragment in _UNSAFE_KEY_FRAGMENTS):
            raise DirectZPossibilityAnalysisError(f"{path}: unsafe key rejected")
    if actual != expected:
        raise DirectZPossibilityAnalysisError(
            f"{path}: exact key set mismatch; "
            f"missing={sorted(expected - actual)}, extra={sorted(extra)}"
        )


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DirectZPossibilityAnalysisError(f"{path}: object required")
    return value


def _bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise DirectZPossibilityAnalysisError(f"{path}: bool required")
    return value


def _int(value: Any, path: str, *, nonnegative: bool = True) -> int:
    if type(value) is not int or (nonnegative and value < 0):
        raise DirectZPossibilityAnalysisError(f"{path}: invalid integer")
    return value


def _finite(value: Any, path: str, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float):
        raise DirectZPossibilityAnalysisError(f"{path}: finite scalar required")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise DirectZPossibilityAnalysisError(f"{path}: scalar outside contract")
    return result


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DirectZPossibilityAnalysisError(f"{path}: lowercase SHA-256 required")
    return value


def _run_id(value: Any, path: str) -> str:
    if not isinstance(value, str) or _RUN_ID_RE.fullmatch(value) is None:
        raise DirectZPossibilityAnalysisError(f"{path}: sanitized run id required")
    return value


def _case_id(value: Any, path: str) -> str:
    if type(value) is int and value >= 0:
        return str(value)
    if isinstance(value, str) and _SAFE_ID_RE.fullmatch(value) is not None:
        return value
    raise DirectZPossibilityAnalysisError(f"{path}: sanitized case id required")


def _safe_timestamp(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 64
        or any(ord(character) < 32 for character in value)
    ):
        raise DirectZPossibilityAnalysisError(f"{path}: invalid recorded_at")
    return value


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DirectZPossibilityAnalysisError(f"{label}: regular non-symlink file required")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=_reject_nonstandard_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectZPossibilityAnalysisError(f"{label}: invalid JSON") from exc
    return _mapping(value, label)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_manifest(raw: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(raw, _MANIFEST_FIELDS, "manifest")
    if raw["schema_version"] != MANIFEST_SCHEMA:
        raise DirectZPossibilityAnalysisError("manifest: schema mismatch")
    model_alias = raw["model_alias"]
    if model_alias not in MODEL_ALIASES:
        raise DirectZPossibilityAnalysisError("manifest: unsupported model")
    arm_order = raw["arm_order"]
    if not isinstance(arm_order, list) or tuple(arm_order) != ARM_ORDER:
        raise DirectZPossibilityAnalysisError("manifest: exact arm order mismatch")
    if _int(raw["case_count"], "manifest.case_count") != EXPECTED_CASE_COUNT:
        raise DirectZPossibilityAnalysisError("manifest: exact case count mismatch")
    if _int(raw["bootstrap_seed"], "manifest.bootstrap_seed") != BOOTSTRAP_SEED:
        raise DirectZPossibilityAnalysisError("manifest: bootstrap seed mismatch")
    if (
        _int(raw["bootstrap_resamples"], "manifest.bootstrap_resamples")
        != BOOTSTRAP_RESAMPLES
    ):
        raise DirectZPossibilityAnalysisError("manifest: bootstrap resamples mismatch")
    if raw["claim_boundary"] != CLAIM_BOUNDARY:
        raise DirectZPossibilityAnalysisError("manifest: claim boundary mismatch")
    return {
        "schema_version": MANIFEST_SCHEMA,
        "run_id": _run_id(raw["run_id"], "manifest.run_id"),
        "model_alias": model_alias,
        "case_count": EXPECTED_CASE_COUNT,
        "arm_order": list(ARM_ORDER),
        "selection_sha256": _sha256(
            raw["selection_sha256"], "manifest.selection_sha256"
        ),
        "config_sha256": _sha256(raw["config_sha256"], "manifest.config_sha256"),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _validate_summary(raw: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(raw, _SUMMARY_FIELDS, "summary")
    if raw["schema_version"] != SUMMARY_SCHEMA:
        raise DirectZPossibilityAnalysisError("summary: schema mismatch")
    run_id = _run_id(raw["run_id"], "summary.run_id")
    model_alias = raw["model_alias"]
    arm_order = raw["arm_order"]
    if (
        run_id != manifest["run_id"]
        or model_alias != manifest["model_alias"]
        or not isinstance(arm_order, list)
        or tuple(arm_order) != ARM_ORDER
        or _sha256(raw["selection_sha256"], "summary.selection_sha256")
        != manifest["selection_sha256"]
        or _sha256(raw["config_sha256"], "summary.config_sha256")
        != manifest["config_sha256"]
    ):
        raise DirectZPossibilityAnalysisError("summary: manifest identity mismatch")
    planned = _int(raw["planned_case_count"], "summary.planned_case_count")
    attempted = _int(raw["attempted_case_count"], "summary.attempted_case_count")
    passed = _int(raw["pass_case_count"], "summary.pass_case_count")
    failed = _int(raw["failed_case_count"], "summary.failed_case_count")
    outcomes = _int(raw["outcome_count"], "summary.outcome_count")
    if (
        planned != EXPECTED_CASE_COUNT
        or attempted != EXPECTED_CASE_COUNT
        or passed + failed != EXPECTED_CASE_COUNT
        or outcomes != EXPECTED_OUTCOME_COUNT
    ):
        raise DirectZPossibilityAnalysisError("summary: denominator/count mismatch")
    return {
        "schema_version": SUMMARY_SCHEMA,
        "run_id": run_id,
        "model_alias": model_alias,
        "planned_case_count": planned,
        "attempted_case_count": attempted,
        "pass_case_count": passed,
        "failed_case_count": failed,
        "outcome_count": outcomes,
        "arm_order": list(ARM_ORDER),
        "selection_sha256": manifest["selection_sha256"],
        "config_sha256": manifest["config_sha256"],
        "all_rollbacks_exact": _bool(
            raw["all_rollbacks_exact"], "summary.all_rollbacks_exact"
        ),
        "firewall_pass": _bool(raw["firewall_pass"], "summary.firewall_pass"),
        "receipt_before_outcome": _bool(
            raw["receipt_before_outcome"], "summary.receipt_before_outcome"
        ),
        "direct_z_once_per_case": _bool(
            raw["direct_z_once_per_case"], "summary.direct_z_once_per_case"
        ),
    }


def _normalize_payload(raw: Mapping[str, Any], path: str) -> dict[str, Any]:
    _exact_keys(raw, _OUTCOME_FIELDS, path)
    arm_id = raw["arm_id"]
    if arm_id not in ARM_ORDER:
        raise DirectZPossibilityAnalysisError(f"{path}: unknown arm")
    delta_cosine = _finite(raw["delta_cosine"], f"{path}.delta_cosine")
    if delta_cosine < -1.0 or delta_cosine > 1.0:
        raise DirectZPossibilityAnalysisError(f"{path}.delta_cosine: outside [-1,1]")
    heldout_kl = _finite(raw["heldout_kl"], f"{path}.heldout_kl", nonnegative=True)
    preservation_score = _finite(
        raw["preservation_score"], f"{path}.preservation_score"
    )
    if not math.isclose(
        preservation_score,
        -heldout_kl,
        rel_tol=0.0,
        abs_tol=PRESERVATION_ABS_TOL,
    ):
        raise DirectZPossibilityAnalysisError(
            f"{path}: preservation_score must equal -heldout_kl"
        )
    endpoint_c_energy = _finite(
        raw["endpoint_c_energy"], f"{path}.endpoint_c_energy", nonnegative=True
    )
    endpoint_frobenius_norm = _finite(
        raw["endpoint_frobenius_norm"],
        f"{path}.endpoint_frobenius_norm",
        nonnegative=True,
    )
    if arm_id in (NO_OP, ORACLE_DO_Z) and (
        abs(endpoint_c_energy) > ZERO_ABS_TOL
        or abs(endpoint_frobenius_norm) > ZERO_ABS_TOL
    ):
        raise DirectZPossibilityAnalysisError(
            f"{path}: non-parameter arm endpoint norms must be zero"
        )
    return {
        "case_id": _case_id(raw["case_id"], f"{path}.case_id"),
        "request_id": _sha256(raw["request_id"], f"{path}.request_id"),
        "arm_id": arm_id,
        "case_pass": _bool(raw["case_pass"], f"{path}.case_pass"),
        "arm_success": _bool(raw["arm_success"], f"{path}.arm_success"),
        "technical_pass": _bool(raw["technical_pass"], f"{path}.technical_pass"),
        "rollback_exact": _bool(raw["rollback_exact"], f"{path}.rollback_exact"),
        "firewall_pass": _bool(raw["firewall_pass"], f"{path}.firewall_pass"),
        "receipt_before_outcome": _bool(
            raw["receipt_before_outcome"], f"{path}.receipt_before_outcome"
        ),
        "z_residual_ratio": _finite(
            raw["z_residual_ratio"], f"{path}.z_residual_ratio", nonnegative=True
        ),
        "generated_delta_error_mean": _finite(
            raw["generated_delta_error_mean"],
            f"{path}.generated_delta_error_mean",
            nonnegative=True,
        ),
        "generated_delta_error_worst": _finite(
            raw["generated_delta_error_worst"],
            f"{path}.generated_delta_error_worst",
            nonnegative=True,
        ),
        "delta_gain": _finite(raw["delta_gain"], f"{path}.delta_gain"),
        "delta_cosine": delta_cosine,
        "off_token_spill_ratio": _finite(
            raw["off_token_spill_ratio"],
            f"{path}.off_token_spill_ratio",
            nonnegative=True,
        ),
        "output_progress": _finite(
            raw["output_progress"], f"{path}.output_progress"
        ),
        "output_nll_reduction": _finite(
            raw["output_nll_reduction"], f"{path}.output_nll_reduction"
        ),
        "exact_margin_min": _finite(
            raw["exact_margin_min"], f"{path}.exact_margin_min"
        ),
        "exact_satisfied": _bool(raw["exact_satisfied"], f"{path}.exact_satisfied"),
        "paraphrase_nll_reduction": _finite(
            raw["paraphrase_nll_reduction"],
            f"{path}.paraphrase_nll_reduction",
        ),
        "heldout_kl": heldout_kl,
        "preservation_score": preservation_score,
        "endpoint_c_energy": endpoint_c_energy,
        "endpoint_frobenius_norm": endpoint_frobenius_norm,
        "nfe": _int(raw["nfe"], f"{path}.nfe"),
    }


def _load_outcomes(path: Path, run_id: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise DirectZPossibilityAnalysisError(
            "outcomes: regular non-symlink file required"
        )
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    wrapper = json.loads(
                        line, parse_constant=_reject_nonstandard_constant
                    )
                except json.JSONDecodeError as exc:
                    raise DirectZPossibilityAnalysisError(
                        f"outcomes line {line_number}: invalid JSON"
                    ) from exc
                wrapper = _mapping(wrapper, f"outcomes[{line_number}]")
                _exact_keys(wrapper, _ENVELOPE_FIELDS, f"outcomes[{line_number}]")
                if (
                    wrapper["schema_version"] != STREAM_SCHEMA
                    or _run_id(wrapper["run_id"], f"outcomes[{line_number}].run_id")
                    != run_id
                    or _int(wrapper["sequence"], f"outcomes[{line_number}].sequence")
                    != len(records)
                    or wrapper["event"] != STREAM_EVENT
                ):
                    raise DirectZPossibilityAnalysisError(
                        f"outcomes line {line_number}: envelope identity/order mismatch"
                    )
                _safe_timestamp(
                    wrapper["recorded_at"], f"outcomes[{line_number}].recorded_at"
                )
                records.append(
                    _normalize_payload(
                        _mapping(wrapper["payload"], f"outcomes[{line_number}].payload"),
                        f"outcomes[{line_number}].payload",
                    )
                )
    except OSError as exc:
        raise DirectZPossibilityAnalysisError("outcomes: read failure") from exc
    if len(records) != EXPECTED_OUTCOME_COUNT:
        raise DirectZPossibilityAnalysisError(
            f"outcomes: expected {EXPECTED_OUTCOME_COUNT}, got {len(records)}"
        )
    return records


def _group_cases(
    records: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    cases: list[dict[str, Any]] = []
    technical_failures: list[str] = []
    for case_index in range(EXPECTED_CASE_COUNT):
        block = records[case_index * len(ARM_ORDER) : (case_index + 1) * len(ARM_ORDER)]
        first = block[0]
        case_id = first["case_id"]
        request_id = first["request_id"]
        if (
            tuple(record["arm_id"] for record in block) != ARM_ORDER
            or {record["case_id"] for record in block} != {case_id}
            or {record["request_id"] for record in block} != {request_id}
            or len({record["case_pass"] for record in block}) != 1
        ):
            raise DirectZPossibilityAnalysisError(
                f"case block {case_index}: identity/arm order mismatch"
            )
        arms = {record["arm_id"]: dict(record) for record in block}
        case_pass = bool(first["case_pass"])
        all_arm_success = all(record["arm_success"] for record in block)
        if case_pass != all_arm_success:
            technical_failures.append(f"case={case_id}: case/arm success mismatch")
        for flag in (
            "technical_pass",
            "rollback_exact",
            "firewall_pass",
            "receipt_before_outcome",
        ):
            if not all(record[flag] for record in block):
                technical_failures.append(f"case={case_id}: {flag}=false")
        if case_pass and not math.isclose(
            arms[NATIVE_ALPHA]["endpoint_c_energy"],
            arms[BF_CURRENT]["endpoint_c_energy"],
            rel_tol=C_MATCH_REL_TOL,
            abs_tol=C_MATCH_ABS_TOL,
        ):
            technical_failures.append(f"case={case_id}: matched-C violation")
        cases.append(
            {
                "case_id": case_id,
                "request_id": request_id,
                "analysis_success": case_pass and all_arm_success,
                "arms": arms,
            }
        )
    if len({case["case_id"] for case in cases}) != EXPECTED_CASE_COUNT:
        raise DirectZPossibilityAnalysisError("case ids must be unique")
    if len({case["request_id"] for case in cases}) != EXPECTED_CASE_COUNT:
        raise DirectZPossibilityAnalysisError("request ids must be unique")
    observed_pass = sum(case["analysis_success"] for case in cases)
    if (
        observed_pass != summary["pass_case_count"]
        or EXPECTED_CASE_COUNT - observed_pass != summary["failed_case_count"]
    ):
        raise DirectZPossibilityAnalysisError("summary/outcomes pass-count mismatch")
    observed_flags = {
        "all_rollbacks_exact": all(record["rollback_exact"] for record in records),
        "firewall_pass": all(record["firewall_pass"] for record in records),
        "receipt_before_outcome": all(
            record["receipt_before_outcome"] for record in records
        ),
    }
    for key, observed in observed_flags.items():
        if observed != summary[key]:
            raise DirectZPossibilityAnalysisError(
                f"summary/outcomes flag mismatch: {key}"
            )
    if not summary["direct_z_once_per_case"]:
        technical_failures.append("summary: direct_z_once_per_case=false")
    return cases, technical_failures


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise DirectZPossibilityAnalysisError("quantile requires values")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _bootstrap_mean_ci(values: Sequence[float], stream: int) -> list[float]:
    if len(values) != EXPECTED_CASE_COUNT:
        raise DirectZPossibilityAnalysisError("bootstrap ITD denominator mismatch")
    rng = random.Random(BOOTSTRAP_SEED + stream)
    estimates = sorted(
        math.fsum(values[rng.randrange(EXPECTED_CASE_COUNT)] for _ in values)
        / EXPECTED_CASE_COUNT
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return [_quantile(estimates, 0.025), _quantile(estimates, 0.975)]


def _effect_summary(values: Sequence[float], stream: int) -> dict[str, Any]:
    if len(values) != EXPECTED_CASE_COUNT or any(not math.isfinite(v) for v in values):
        raise DirectZPossibilityAnalysisError("effect summary denominator/value mismatch")
    mean = math.fsum(values) / EXPECTED_CASE_COUNT
    positive = sum(value > 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    return {
        "n_itd": EXPECTED_CASE_COUNT,
        "mean": mean,
        "median": statistics.median(values),
        "positive_count": positive,
        "negative_count": negative,
        "zero_count": EXPECTED_CASE_COUNT - positive - negative,
        "positive_fraction": positive / EXPECTED_CASE_COUNT,
        "lenient_positive": mean > 0.0 and positive >= STRICT_MAJORITY,
        "lenient_negative": mean < 0.0 and negative >= STRICT_MAJORITY,
        "paired_bootstrap_mean_ci95": _bootstrap_mean_ci(values, stream),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "ci_exclusion_required_for_gate": False,
    }


def _contrast(
    cases: Sequence[Mapping[str, Any]],
    lhs_arm: str,
    rhs_arm: str,
    metric: str,
) -> list[float]:
    return [
        float(case["arms"][lhs_arm][metric])
        - float(case["arms"][rhs_arm][metric])
        if case["analysis_success"]
        else 0.0
        for case in cases
    ]


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = math.fsum(xs) / len(xs)
    mean_y = math.fsum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denominator = math.sqrt(
        math.fsum(value * value for value in centered_x)
        * math.fsum(value * value for value in centered_y)
    )
    if denominator == 0.0:
        return None
    return math.fsum(x * y for x, y in zip(centered_x, centered_y)) / denominator


def _relation_summary(
    grouped_pairs: Sequence[Sequence[tuple[float, float]]], stream: int
) -> dict[str, Any]:
    def flatten(indices: Iterable[int]) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for index in indices:
            for x, y in grouped_pairs[index]:
                xs.append(x)
                ys.append(y)
        return xs, ys

    point_x, point_y = flatten(range(EXPECTED_CASE_COUNT))
    point = _pearson(point_x, point_y)
    rng = random.Random(BOOTSTRAP_SEED + stream)
    bootstrap: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        xs, ys = flatten(
            rng.randrange(EXPECTED_CASE_COUNT) for _ in range(EXPECTED_CASE_COUNT)
        )
        estimate = _pearson(xs, ys)
        if estimate is not None and math.isfinite(estimate):
            bootstrap.append(estimate)
    bootstrap.sort()
    ci: list[float | None]
    if bootstrap:
        ci = [_quantile(bootstrap, 0.025), _quantile(bootstrap, 0.975)]
    else:
        ci = [None, None]
    return {
        "pearson": point,
        "paired_case_bootstrap_ci95": ci,
        "valid_bootstrap_resamples": len(bootstrap),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "descriptive_noncausal": True,
    }


def _build_effects(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stream = 0

    def summarize(values: Sequence[float]) -> dict[str, Any]:
        nonlocal stream
        result = _effect_summary(values, stream)
        stream += 1
        return result

    oracle = {
        "z_residual_gain_noop_minus_oracle": summarize(
            _contrast(cases, NO_OP, ORACLE_DO_Z, "z_residual_ratio")
        ),
        "generated_error_mean_gain_noop_minus_oracle": summarize(
            _contrast(cases, NO_OP, ORACLE_DO_Z, "generated_delta_error_mean")
        ),
        "output_nll_gain_oracle_minus_noop": summarize(
            _contrast(cases, ORACLE_DO_Z, NO_OP, "output_nll_reduction")
        ),
        "output_progress_gain_oracle_minus_noop": summarize(
            _contrast(cases, ORACLE_DO_Z, NO_OP, "output_progress")
        ),
        "exact_margin_gain_oracle_minus_noop": summarize(
            _contrast(cases, ORACLE_DO_Z, NO_OP, "exact_margin_min")
        ),
        "exact_satisfaction_gain_oracle_minus_noop": summarize(
            _contrast(cases, ORACLE_DO_Z, NO_OP, "exact_satisfied")
        ),
        "paraphrase_gain_oracle_minus_noop": summarize(
            _contrast(cases, ORACLE_DO_Z, NO_OP, "paraphrase_nll_reduction")
        ),
    }
    cone = {
        "z_residual_gain_noop_minus_cone": summarize(
            _contrast(cases, NO_OP, SYNC_CONE, "z_residual_ratio")
        ),
        "generated_error_mean_gain_noop_minus_cone": summarize(
            _contrast(cases, NO_OP, SYNC_CONE, "generated_delta_error_mean")
        ),
        "generated_error_worst_gain_noop_minus_cone": summarize(
            _contrast(cases, NO_OP, SYNC_CONE, "generated_delta_error_worst")
        ),
        "delta_gain_cone_minus_noop": summarize(
            _contrast(cases, SYNC_CONE, NO_OP, "delta_gain")
        ),
        "delta_cosine_gain_cone_minus_noop": summarize(
            _contrast(cases, SYNC_CONE, NO_OP, "delta_cosine")
        ),
        "output_nll_gain_cone_minus_noop": summarize(
            _contrast(cases, SYNC_CONE, NO_OP, "output_nll_reduction")
        ),
        "exact_margin_gain_cone_minus_noop": summarize(
            _contrast(cases, SYNC_CONE, NO_OP, "exact_margin_min")
        ),
        "paraphrase_gain_cone_minus_noop": summarize(
            _contrast(cases, SYNC_CONE, NO_OP, "paraphrase_nll_reduction")
        ),
        "preservation_gain_cone_minus_noop": summarize(
            _contrast(cases, SYNC_CONE, NO_OP, "preservation_score")
        ),
        "output_gap_oracle_minus_cone": summarize(
            _contrast(cases, ORACLE_DO_Z, SYNC_CONE, "output_nll_reduction")
        ),
        "z_residual_gap_cone_minus_oracle": summarize(
            _contrast(cases, SYNC_CONE, ORACLE_DO_Z, "z_residual_ratio")
        ),
    }
    current_bf = {
        "z_objective_gap_bf_minus_cone": summarize(
            _contrast(cases, BF_CURRENT, SYNC_CONE, "z_residual_ratio")
        ),
        "generated_error_mean_gap_bf_minus_cone": summarize(
            _contrast(cases, BF_CURRENT, SYNC_CONE, "generated_delta_error_mean")
        ),
        "generated_error_worst_gap_bf_minus_cone": summarize(
            _contrast(cases, BF_CURRENT, SYNC_CONE, "generated_delta_error_worst")
        ),
        "output_objective_gap_cone_minus_bf": summarize(
            _contrast(cases, SYNC_CONE, BF_CURRENT, "output_nll_reduction")
        ),
        "exact_margin_gap_cone_minus_bf": summarize(
            _contrast(cases, SYNC_CONE, BF_CURRENT, "exact_margin_min")
        ),
        "off_token_spill_gap_bf_minus_cone": summarize(
            _contrast(cases, BF_CURRENT, SYNC_CONE, "off_token_spill_ratio")
        ),
        "preservation_difference_bf_minus_cone": summarize(
            _contrast(cases, BF_CURRENT, SYNC_CONE, "preservation_score")
        ),
        "z_gain_bf_vs_native_alpha": summarize(
            _contrast(cases, NATIVE_ALPHA, BF_CURRENT, "z_residual_ratio")
        ),
        "output_nll_gain_bf_vs_native_alpha": summarize(
            _contrast(cases, BF_CURRENT, NATIVE_ALPHA, "output_nll_reduction")
        ),
        "output_progress_gain_bf_vs_native_alpha": summarize(
            _contrast(cases, BF_CURRENT, NATIVE_ALPHA, "output_progress")
        ),
        "exact_margin_gain_bf_vs_native_alpha": summarize(
            _contrast(cases, BF_CURRENT, NATIVE_ALPHA, "exact_margin_min")
        ),
        "paraphrase_gain_bf_vs_native_alpha": summarize(
            _contrast(cases, BF_CURRENT, NATIVE_ALPHA, "paraphrase_nll_reduction")
        ),
        "preservation_gain_bf_vs_native_alpha": summarize(
            _contrast(cases, BF_CURRENT, NATIVE_ALPHA, "preservation_score")
        ),
        "endpoint_c_difference_bf_minus_native_alpha": summarize(
            _contrast(cases, BF_CURRENT, NATIVE_ALPHA, "endpoint_c_energy")
        ),
        "endpoint_frobenius_difference_bf_minus_native_alpha": summarize(
            _contrast(cases, BF_CURRENT, NATIVE_ALPHA, "endpoint_frobenius_norm")
        ),
    }

    relation_arms = (NATIVE_FULL, NATIVE_ALPHA, BF_CURRENT, SYNC_CONE)
    relation_groups: dict[str, list[list[tuple[float, float]]]] = {
        "z_vs_output": [],
        "z_vs_preservation": [],
        "output_vs_preservation": [],
    }
    matched_joint_counts = {
        "z_and_output_bf_better": 0,
        "z_and_preservation_bf_better": 0,
        "output_and_preservation_bf_better": 0,
    }
    for case in cases:
        pairs = {key: [] for key in relation_groups}
        for arm in relation_arms:
            if case["analysis_success"]:
                z_gain = (
                    case["arms"][NO_OP]["z_residual_ratio"]
                    - case["arms"][arm]["z_residual_ratio"]
                )
                output_gain = (
                    case["arms"][arm]["output_nll_reduction"]
                    - case["arms"][NO_OP]["output_nll_reduction"]
                )
                preservation_gain = (
                    case["arms"][arm]["preservation_score"]
                    - case["arms"][NO_OP]["preservation_score"]
                )
            else:
                z_gain = output_gain = preservation_gain = 0.0
            pairs["z_vs_output"].append((z_gain, output_gain))
            pairs["z_vs_preservation"].append((z_gain, preservation_gain))
            pairs["output_vs_preservation"].append((output_gain, preservation_gain))
        for key in relation_groups:
            relation_groups[key].append(pairs[key])
        if case["analysis_success"]:
            z_better = (
                case["arms"][NATIVE_ALPHA]["z_residual_ratio"]
                - case["arms"][BF_CURRENT]["z_residual_ratio"]
                > 0.0
            )
            output_better = (
                case["arms"][BF_CURRENT]["output_nll_reduction"]
                - case["arms"][NATIVE_ALPHA]["output_nll_reduction"]
                > 0.0
            )
            preservation_better = (
                case["arms"][BF_CURRENT]["preservation_score"]
                - case["arms"][NATIVE_ALPHA]["preservation_score"]
                > 0.0
            )
            matched_joint_counts["z_and_output_bf_better"] += int(
                z_better and output_better
            )
            matched_joint_counts["z_and_preservation_bf_better"] += int(
                z_better and preservation_better
            )
            matched_joint_counts["output_and_preservation_bf_better"] += int(
                output_better and preservation_better
            )

    relations = {
        "z_gain_vs_output_gain": _relation_summary(
            relation_groups["z_vs_output"], stream + 0
        ),
        "z_gain_vs_preservation_gain": _relation_summary(
            relation_groups["z_vs_preservation"], stream + 1
        ),
        "output_gain_vs_preservation_gain": _relation_summary(
            relation_groups["output_vs_preservation"], stream + 2
        ),
        "bf_vs_native_alpha_joint_direction_counts": {
            **matched_joint_counts,
            "n_itd": EXPECTED_CASE_COUNT,
        },
        "interpretation": "descriptive relations; no mediation or causality claim",
    }
    return {
        "oracle_ceiling": oracle,
        "cone_feasibility": cone,
        "current_bf_objective_gap": current_bf,
        "z_output_preservation_relations": relations,
    }


def _classify(effects: Mapping[str, Any], technical_pass: bool) -> dict[str, Any]:
    oracle = effects["oracle_ceiling"]
    cone = effects["cone_feasibility"]
    current = effects["current_bf_objective_gap"]
    oracle_clear = bool(
        oracle["z_residual_gain_noop_minus_oracle"]["lenient_positive"]
        and oracle["output_nll_gain_oracle_minus_noop"]["lenient_positive"]
    )
    cone_clear = bool(
        cone["z_residual_gain_noop_minus_cone"]["lenient_positive"]
        and cone["output_nll_gain_cone_minus_noop"]["lenient_positive"]
    )
    bf_gap_clear = bool(
        current["z_objective_gap_bf_minus_cone"]["lenient_positive"]
        or current["generated_error_mean_gap_bf_minus_cone"]["lenient_positive"]
        or current["output_objective_gap_cone_minus_bf"]["lenient_positive"]
    )
    if not technical_pass:
        verdict = "BLOCK_TECHNICAL_INVALID"
    elif not oracle_clear:
        verdict = "NO_ORACLE_CEILING_OBSERVED"
    elif not cone_clear:
        verdict = "ORACLE_ONLY_NO_CONE_FEASIBILITY"
    elif bf_gap_clear:
        verdict = "CONE_FEASIBLE_CURRENT_BF_GAP"
    else:
        verdict = "CONE_FEASIBLE_NO_DIRECTIONAL_BF_GAP_OBSERVED"
    return {
        "verdict": verdict,
        "oracle_ceiling_clear": oracle_clear,
        "sync_cone_feasibility_clear": cone_clear,
        "current_bf_directional_gap_clear": bf_gap_clear,
        "gate": {
            "strict_majority": STRICT_MAJORITY,
            "requires_positive_mean": True,
            "requires_ci_exclusion": False,
            "method_superiority_decision": False,
        },
    }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def analyze_run(run_directory: str | Path) -> dict[str, Any]:
    root = Path(run_directory).expanduser().resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise DirectZPossibilityAnalysisError("run directory must be a regular directory")
    manifest_path = root / "manifest.json"
    summary_path = root / "summary.json"
    outcomes_path = root / "outcomes.jsonl"
    manifest = _validate_manifest(_read_json_object(manifest_path, "manifest"))
    summary = _validate_summary(
        _read_json_object(summary_path, "summary"), manifest
    )
    records = _load_outcomes(outcomes_path, manifest["run_id"])
    cases, technical_failures = _group_cases(records, summary)
    effects = _build_effects(cases)
    technical_pass = not technical_failures
    classification = _classify(effects, technical_pass)
    analysis: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA,
        "run_id": manifest["run_id"],
        "model_alias": manifest["model_alias"],
        "scope": "single_model_direct_z_possibility",
        "claim_boundary": CLAIM_BOUNDARY,
        "fixed_contract": {
            "case_count": EXPECTED_CASE_COUNT,
            "arm_order": list(ARM_ORDER),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "failed_cases_itd_zero_effect": True,
            "ci_exclusion_required_for_gate": False,
        },
        "artifact_validation": {
            "valid": technical_pass,
            "failures": technical_failures,
            "planned_case_count": EXPECTED_CASE_COUNT,
            "successful_case_count": sum(
                bool(case["analysis_success"]) for case in cases
            ),
            "failed_case_count": sum(
                not bool(case["analysis_success"]) for case in cases
            ),
            "itd_denominator": EXPECTED_CASE_COUNT,
            "outcome_count": len(records),
        },
        "effects": effects,
        "classification": classification,
        "compute": {
            "nfe_total_itd": sum(
                int(record["nfe"])
                for case in cases
                for record in case["arms"].values()
            ),
            "endpoint_c_native_alpha_bf_matched": all(
                not case["analysis_success"]
                or math.isclose(
                    case["arms"][NATIVE_ALPHA]["endpoint_c_energy"],
                    case["arms"][BF_CURRENT]["endpoint_c_energy"],
                    rel_tol=C_MATCH_REL_TOL,
                    abs_tol=C_MATCH_ABS_TOL,
                )
                for case in cases
            ),
        },
        "case_diagnostics": [
            {
                "case_id": case["case_id"],
                "analysis_success": case["analysis_success"],
                "oracle_output_gain": (
                    case["arms"][ORACLE_DO_Z]["output_nll_reduction"]
                    - case["arms"][NO_OP]["output_nll_reduction"]
                    if case["analysis_success"]
                    else 0.0
                ),
                "cone_output_gain": (
                    case["arms"][SYNC_CONE]["output_nll_reduction"]
                    - case["arms"][NO_OP]["output_nll_reduction"]
                    if case["analysis_success"]
                    else 0.0
                ),
                "bf_output_objective_gap": (
                    case["arms"][SYNC_CONE]["output_nll_reduction"]
                    - case["arms"][BF_CURRENT]["output_nll_reduction"]
                    if case["analysis_success"]
                    else 0.0
                ),
                "bf_z_objective_gap": (
                    case["arms"][BF_CURRENT]["z_residual_ratio"]
                    - case["arms"][SYNC_CONE]["z_residual_ratio"]
                    if case["analysis_success"]
                    else 0.0
                ),
            }
            for case in cases
        ],
        "source_hashes": {
            "manifest_json": _file_sha256(manifest_path),
            "summary_json": _file_sha256(summary_path),
            "outcomes_jsonl": _file_sha256(outcomes_path),
            "analysis_code": _file_sha256(Path(__file__).resolve(strict=True)),
        },
        "interpretation_limits": [
            "possibility diagnostic only",
            "no method superiority or benchmark gain claim",
            "bootstrap interval is descriptive and not a gate",
            "z/output/preservation relations are noncausal",
            "no sequential retention or downstream generalization claim",
        ],
    }
    analysis["analysis_sha256"] = hashlib.sha256(_canonical_json(analysis)).hexdigest()
    return analysis


def render_markdown(analysis: Mapping[str, Any]) -> str:
    effects = analysis["effects"]
    rows = (
        (
            "Oracle z gain",
            effects["oracle_ceiling"]["z_residual_gain_noop_minus_oracle"],
        ),
        (
            "Oracle output-NLL gain",
            effects["oracle_ceiling"]["output_nll_gain_oracle_minus_noop"],
        ),
        (
            "Cone z gain",
            effects["cone_feasibility"]["z_residual_gain_noop_minus_cone"],
        ),
        (
            "Cone output-NLL gain",
            effects["cone_feasibility"]["output_nll_gain_cone_minus_noop"],
        ),
        (
            "BF z objective gap",
            effects["current_bf_objective_gap"]["z_objective_gap_bf_minus_cone"],
        ),
        (
            "BF output objective gap",
            effects["current_bf_objective_gap"]["output_objective_gap_cone_minus_bf"],
        ),
        (
            "BF vs alpha preservation",
            effects["current_bf_objective_gap"][
                "preservation_gain_bf_vs_native_alpha"
            ],
        ),
    )
    lines = [
        f"# Direct-z possibility analysis — {analysis['model_alias']}",
        "",
        f"- run: `{analysis['run_id']}`",
        f"- verdict: `{analysis['classification']['verdict']}`",
        f"- technical validity: `{analysis['artifact_validation']['valid']}`",
        f"- ITD: `{analysis['artifact_validation']['successful_case_count']}/"
        f"{analysis['artifact_validation']['itd_denominator']}` successful",
        f"- analysis SHA-256: `{analysis['analysis_sha256']}`",
        "- scope: possibility only; this is not a method-superiority analysis",
        "",
        "| Paired descriptive contrast | Mean | Median | Positive | Bootstrap mean CI95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, item in rows:
        lo, hi = item["paired_bootstrap_mean_ci95"]
        lines.append(
            f"| {label} | {item['mean']:.8f} | {item['median']:.8f} | "
            f"{item['positive_count']}/{item['n_itd']} | "
            f"[{lo:.8f}, {hi:.8f}] |"
        )
    lines.extend(
        [
            "",
            "## Lenient possibility gate",
            "",
            f"- oracle ceiling: `{analysis['classification']['oracle_ceiling_clear']}`",
            f"- synchronous cone feasibility: "
            f"`{analysis['classification']['sync_cone_feasibility_clear']}`",
            f"- current BF directional gap: "
            f"`{analysis['classification']['current_bf_directional_gap_clear']}`",
            f"- strict majority: `{STRICT_MAJORITY}/{EXPECTED_CASE_COUNT}` plus positive mean",
            "- CI exclusion is not required; intervals are descriptive.",
            "",
            "## Claim boundary",
            "",
            "- Allowed: fixed-model direct-z possibility classification on this locked panel.",
            "- Forbidden: ODE-Edit superiority, causal mediation, benchmark gain, or sequential robustness.",
            "- `preservation_score` is exactly `-heldout_kl`; higher is better.",
            "- Failed scientific cases remain in the ITD denominator with zero paired effect.",
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
        raise DirectZPossibilityAnalysisError("JSON and Markdown outputs must differ")
    for path in (json_path, markdown_path):
        if path.exists() or path.is_symlink():
            raise DirectZPossibilityAnalysisError("analysis outputs are exclusive")
        if not path.parent.is_dir() or path.parent.is_symlink():
            raise DirectZPossibilityAnalysisError("output parent must already exist")
    json_text = (
        json.dumps(
            analysis,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    markdown_text = render_markdown(analysis)
    try:
        with json_path.open("x", encoding="utf-8") as handle:
            handle.write(json_text)
        with markdown_path.open("x", encoding="utf-8") as handle:
            handle.write(markdown_text)
    except OSError as exc:
        raise DirectZPossibilityAnalysisError("exclusive analysis output failed") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ode-edit-direct-z-possibility-analysis", allow_abbrev=False
    )
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        analysis = analyze_run(args.run_directory)
        write_outputs(analysis, args.output_json, args.output_markdown)
    except DirectZPossibilityAnalysisError as exc:
        print(f"direct-z possibility analysis rejected: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "model_alias": analysis["model_alias"],
                "verdict": analysis["classification"]["verdict"],
                "analysis_sha256": analysis["analysis_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if analysis["artifact_validation"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
