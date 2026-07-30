#!/usr/bin/env python3
"""MV-2 단일 모델 refresh diagnostic의 고정 분석기.

이 모듈의 핵심 진입점은 :func:`analyze_case_records`다. 분석기는 실행기의
내부 클래스나 raw tensor 형식에 의존하지 않고, 아래의 작은 case-level
계약만 받는다.

각 case record는 ``case_id``, ``request_id``, ``pass``, ``technical``,
``arms``, ``budgets``, ``lineage``, ``compute``를 포함한다.
``branch_order``는 ``ARM_ORDER``와 정확히 같고 ``arms``는 같은 여섯
원소를 가져야 한다. 실행 실패 case는 제거하지 않고 direction,
coefficient, secondary total contrast에 각각 0을 기여한다.

bootstrap seed/sample 수, case 수, arm 순서, matched-C tolerance는
preregistered 상수이며 CLI로 바꿀 수 없다.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


EXPECTED_CASES = 12
ARM_ORDER = (
    "no_op_replay",
    "h0_sham",
    "partial_joint",
    "refreshed_direction_refreshed_coefficient",
    "fixed_direction_refreshed_coefficient",
    "fixed_direction_fixed_coefficient",
)

NO_OP = ARM_ORDER[0]
H0_SHAM = ARM_ORDER[1]
PARTIAL = ARM_ORDER[2]
REFRESHED_DIRECTION_REFRESHED_COEFFICIENT = ARM_ORDER[3]
FIXED_DIRECTION_REFRESHED_COEFFICIENT = ARM_ORDER[4]
FIXED_DIRECTION_FIXED_COEFFICIENT = ARM_ORDER[5]

BOOTSTRAP_SEED = 20260801
BOOTSTRAP_SAMPLES = 4000
# Outcome-blind practical floor for one allowed-utility contrast.  This is
# deliberately much larger than float32 replay jitter, so a deterministic
# no-op cannot turn an immaterial 1e-12-scale sign into a method claim.
PRACTICAL_EFFECT_FLOOR = 1e-4
MATCHED_C_REL_TOL = 3e-5
MATCHED_C_ABS_TOL = 3e-5
MODEL_ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")

SCHEMA_VERSION = "ode-edit.mv2.refresh-analysis.v1"
RUN_STREAM_SCHEMA = "ode-edit-mv2-refresh/v1"
RUN_STREAM_EVENT = "mv2_refresh_analysis_case"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MV2AnalysisError(ValueError):
    """입력 schema 또는 수치가 분석 가능한 계약을 위반했음을 뜻한다."""


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MV2AnalysisError(f"{path}: mapping이어야 함")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: Sequence[str], path: str
) -> None:
    actual = tuple(value.keys())
    expected_tuple = tuple(expected)
    if actual != expected_tuple:
        raise MV2AnalysisError(
            f"{path}: exact key/order 위반: expected={expected_tuple}, actual={actual}"
        )


def _require_nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise MV2AnalysisError(f"{path}: 비어 있지 않은 문자열이어야 함")
    return value


def _require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise MV2AnalysisError(f"{path}: bool이어야 함")
    return value


def _require_finite(value: Any, path: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MV2AnalysisError(f"{path}: finite number여야 함")
    result = float(value)
    if not math.isfinite(result):
        raise MV2AnalysisError(f"{path}: non-finite 값 거부")
    if nonnegative and result < 0.0:
        raise MV2AnalysisError(f"{path}: 0 이상이어야 함")
    return result


def _require_nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MV2AnalysisError(f"{path}: 0 이상의 int여야 함")
    return value


def _require_sha256(value: Any, path: str) -> str:
    result = _require_nonempty_string(value, path)
    if _SHA256_RE.fullmatch(result) is None:
        raise MV2AnalysisError(f"{path}: canonical lowercase SHA256가 아님")
    return result


def _trimmed_mean(values: Sequence[float], proportion: float = 0.10) -> float:
    if not values:
        raise MV2AnalysisError("trimmed mean 입력이 비어 있음")
    ordered = sorted(values)
    trim = math.floor(len(ordered) * proportion)
    retained = ordered[trim : len(ordered) - trim] if trim else ordered
    return math.fsum(retained) / len(retained)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise MV2AnalysisError("quantile 입력이 비어 있음")
    if probability <= 0.0:
        return sorted_values[0]
    if probability >= 1.0:
        return sorted_values[-1]
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def _paired_bootstrap_mean(values: Sequence[float], *, stream: int) -> dict[str, Any]:
    """고정 seed에서 같은 case resample index를 쓰는 paired-case bootstrap."""

    if len(values) != EXPECTED_CASES:
        raise MV2AnalysisError(
            f"bootstrap denominator는 정확히 {EXPECTED_CASES}여야 함"
        )
    rng = random.Random(BOOTSTRAP_SEED)
    estimates: list[float] = []
    count = len(values)
    for _ in range(BOOTSTRAP_SAMPLES):
        estimates.append(
            math.fsum(values[rng.randrange(count)] for _ in range(count)) / count
        )
    estimates.sort()
    return {
        "seed": BOOTSTRAP_SEED,
        "stream": stream,
        "samples": BOOTSTRAP_SAMPLES,
        "mean_ci95": [
            _quantile(estimates, 0.025),
            _quantile(estimates, 0.975),
        ],
        "probability_mean_positive": (
            sum(value > 0.0 for value in estimates) / BOOTSTRAP_SAMPLES
        ),
    }


def _effect_summary(
    values: Sequence[float],
    *,
    envelope: float,
    stream: int,
) -> dict[str, Any]:
    mean = math.fsum(values) / len(values)
    trimmed_mean = _trimmed_mean(values)
    median = statistics.median(values)
    positive_count = sum(value > 0.0 for value in values)
    return {
        "n": len(values),
        "mean": mean,
        "trimmed_mean_10pct": trimmed_mean,
        "median": median,
        "positive_count": positive_count,
        "sign_rate": positive_count / len(values),
        "mean_above_envelope": mean > envelope,
        "trimmed_mean_above_envelope": trimmed_mean > envelope,
        "median_above_envelope": median > envelope,
        "sign_at_least_7_of_12": positive_count >= 7,
        "bootstrap": _paired_bootstrap_mean(values, stream=stream),
    }


def _validate_lineage(lineage: Mapping[str, Any], path: str) -> dict[str, str]:
    expected = (
        "origin_state_id",
        "w1_state_id",
        "lineage_id",
        "target_token_sha256",
        "direct_z_tensor_sha256",
        "context_id",
    )
    _require_exact_keys(lineage, expected, path)
    return {
        "origin_state_id": _require_sha256(
            lineage["origin_state_id"], f"{path}.origin_state_id"
        ),
        "w1_state_id": _require_sha256(
            lineage["w1_state_id"], f"{path}.w1_state_id"
        ),
        "lineage_id": _require_sha256(
            lineage["lineage_id"], f"{path}.lineage_id"
        ),
        "target_token_sha256": _require_sha256(
            lineage["target_token_sha256"], f"{path}.target_token_sha256"
        ),
        "direct_z_tensor_sha256": _require_sha256(
            lineage["direct_z_tensor_sha256"],
            f"{path}.direct_z_tensor_sha256",
        ),
        "context_id": _require_sha256(
            lineage["context_id"], f"{path}.context_id"
        ),
    }


def _validate_technical(
    technical: Mapping[str, Any], path: str
) -> tuple[dict[str, bool], list[str]]:
    expected = (
        "exact_panel",
        "lineage_exact",
        "matched_second_c",
        "rollback_exact",
        "firewall_pass",
        "receipt_before_outcome",
    )
    _require_exact_keys(technical, expected, path)
    normalized: dict[str, bool] = {}
    failures: list[str] = []
    for key in expected:
        value = _require_bool(technical[key], f"{path}.{key}")
        normalized[key] = value
        if not value:
            failures.append(f"{path}.{key}=false")
    return normalized, failures


def _validate_arm(arm: Mapping[str, Any], path: str) -> dict[str, Any]:
    expected = ("progress", "c_energy", "nfe", "success")
    _require_exact_keys(arm, expected, path)
    return {
        "progress": _require_finite(arm["progress"], f"{path}.progress"),
        "c_energy": _require_finite(
            arm["c_energy"], f"{path}.c_energy", nonnegative=True
        ),
        "nfe": _require_nonnegative_int(arm["nfe"], f"{path}.nfe"),
        "success": _require_bool(arm["success"], f"{path}.success"),
    }


def _validate_compute(compute: Mapping[str, Any], path: str) -> dict[str, Any]:
    expected = (
        "controlled_nfe",
        "proposal_build_count",
        "probe_panel_count",
        "wall_seconds",
    )
    _require_exact_keys(compute, expected, path)
    return {
        "controlled_nfe": _require_nonnegative_int(
            compute["controlled_nfe"], f"{path}.controlled_nfe"
        ),
        "proposal_build_count": _require_nonnegative_int(
            compute["proposal_build_count"], f"{path}.proposal_build_count"
        ),
        "probe_panel_count": _require_nonnegative_int(
            compute["probe_panel_count"], f"{path}.probe_panel_count"
        ),
        "wall_seconds": _require_finite(
            compute["wall_seconds"], f"{path}.wall_seconds", nonnegative=True
        ),
    }


def _normalize_case(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    path = f"records[{index}]"
    expected = (
        "case_id",
        "request_id",
        "pass",
        "technical",
        "branch_order",
        "arms",
        "budgets",
        "lineage",
        "compute",
    )
    optional = {"model_alias"}
    actual = set(record)
    if set(expected) - actual or actual - set(expected) - optional:
        raise MV2AnalysisError(
            f"{path}: exact case schema 위반: "
            f"missing={sorted(set(expected) - actual)}, "
            f"extra={sorted(actual - set(expected) - optional)}"
        )

    case_id_raw = record["case_id"]
    if isinstance(case_id_raw, bool) or not isinstance(case_id_raw, (str, int)):
        raise MV2AnalysisError(f"{path}.case_id: str 또는 int여야 함")
    case_id = str(case_id_raw)
    if not case_id:
        raise MV2AnalysisError(f"{path}.case_id: 비어 있을 수 없음")
    request_id = _require_sha256(record["request_id"], f"{path}.request_id")
    case_pass = _require_bool(record["pass"], f"{path}.pass")

    technical, technical_failures = _validate_technical(
        _require_mapping(record["technical"], f"{path}.technical"),
        f"{path}.technical",
    )

    branch_order = record["branch_order"]
    if (
        not isinstance(branch_order, list)
        or tuple(branch_order) != ARM_ORDER
    ):
        raise MV2AnalysisError(
            f"{path}.branch_order: exact six-arm order 위반"
        )
    arms_raw = _require_mapping(record["arms"], f"{path}.arms")
    if set(arms_raw) != set(ARM_ORDER):
        raise MV2AnalysisError(f"{path}.arms: exact six-arm set 위반")
    arms = {
        arm_id: _validate_arm(
            _require_mapping(arms_raw[arm_id], f"{path}.arms.{arm_id}"),
            f"{path}.arms.{arm_id}",
        )
        for arm_id in ARM_ORDER
    }

    budgets = _require_mapping(record["budgets"], f"{path}.budgets")
    _require_exact_keys(budgets, ("second_step_c_energy",), f"{path}.budgets")
    second_step_c_energy = _require_finite(
        budgets["second_step_c_energy"],
        f"{path}.budgets.second_step_c_energy",
        nonnegative=True,
    )

    lineage = _validate_lineage(
        _require_mapping(record["lineage"], f"{path}.lineage"),
        f"{path}.lineage",
    )
    compute = _validate_compute(
        _require_mapping(record["compute"], f"{path}.compute"),
        f"{path}.compute",
    )
    if not case_pass:
        technical_failures.append(f"{path}.pass=false")
    elif (
        second_step_c_energy <= 0.0
        or compute["controlled_nfe"] != 43
        or compute["proposal_build_count"] != 4
        or compute["probe_panel_count"] != 3
        or any(arms[arm_id]["nfe"] != 1 for arm_id in ARM_ORDER)
        or any(not arms[arm_id]["success"] for arm_id in ARM_ORDER)
    ):
        technical_failures.append(
            f"{path}: positive-C/one-build/reuse/NFE/success contract 위반"
        )
    if lineage["origin_state_id"] == lineage["w1_state_id"]:
        technical_failures.append(f"{path}.lineage: W1이 W0 descendant가 아님")

    if not math.isclose(
        arms[NO_OP]["c_energy"],
        0.0,
        rel_tol=MATCHED_C_REL_TOL,
        abs_tol=MATCHED_C_ABS_TOL,
    ):
        technical_failures.append(f"{path}.arms.{NO_OP}.c_energy!=0")
    if not math.isclose(
        arms[H0_SHAM]["c_energy"],
        0.0,
        rel_tol=MATCHED_C_REL_TOL,
        abs_tol=MATCHED_C_ABS_TOL,
    ):
        technical_failures.append(f"{path}.arms.{H0_SHAM}.c_energy!=0")
    if not math.isclose(
        arms[PARTIAL]["c_energy"],
        second_step_c_energy,
        rel_tol=MATCHED_C_REL_TOL,
        abs_tol=MATCHED_C_ABS_TOL,
    ):
        technical_failures.append(
            f"{path}.arms.{PARTIAL}.c_energy!=h^2*b0=(1-h)^2*b0"
        )

    for arm_id in (
        REFRESHED_DIRECTION_REFRESHED_COEFFICIENT,
        FIXED_DIRECTION_REFRESHED_COEFFICIENT,
        FIXED_DIRECTION_FIXED_COEFFICIENT,
    ):
        if not math.isclose(
            arms[arm_id]["c_energy"],
            second_step_c_energy,
            rel_tol=MATCHED_C_REL_TOL,
            abs_tol=MATCHED_C_ABS_TOL,
        ):
            technical_failures.append(
                f"{path}.arms.{arm_id}.c_energy!=second_step_c_energy"
            )
    if technical["matched_second_c"] and any(
        "c_energy" in failure for failure in technical_failures
    ):
        technical_failures.append(
            f"{path}.technical.matched_second_c가 measured C와 불일치"
        )

    model_alias: str | None = None
    if "model_alias" in record:
        model_alias = _require_nonempty_string(
            record["model_alias"], f"{path}.model_alias"
        )

    return {
        "case_id": case_id,
        "request_id": request_id,
        "pass": case_pass,
        "technical": technical,
        "technical_failures": technical_failures,
        "branch_order": list(ARM_ORDER),
        "arms": arms,
        "budgets": {"second_step_c_energy": second_step_c_energy},
        "lineage": lineage,
        "compute": compute,
        "model_alias": model_alias,
    }


def analyze_case_records(
    records: Iterable[Mapping[str, Any]],
    *,
    model_alias: str | None = None,
) -> dict[str, Any]:
    """정확히 12개 case record의 단일 모델 MV-2 결과를 분석한다.

    Schema/수치가 malformed이면 :class:`MV2AnalysisError`를 발생시킨다.
    명시적으로 기록된 technical gate 실패나 measured matched-C 실패는
    결과를 버리지 않고 ``BLOCK_TECHNICAL_INVALID`` verdict로 반환한다.
    """

    materialized = list(records)
    if len(materialized) != EXPECTED_CASES:
        raise MV2AnalysisError(
            f"exact case count 위반: expected={EXPECTED_CASES}, "
            f"actual={len(materialized)}"
        )
    normalized = [
        _normalize_case(_require_mapping(record, f"records[{index}]"), index)
        for index, record in enumerate(materialized)
    ]

    case_ids = [record["case_id"] for record in normalized]
    request_ids = [record["request_id"] for record in normalized]
    lineage_ids = [record["lineage"]["lineage_id"] for record in normalized]
    for label, values in (
        ("case_id", case_ids),
        ("request_id", request_ids),
        ("lineage_id", lineage_ids),
        (
            "origin_state_id",
            [record["lineage"]["origin_state_id"] for record in normalized],
        ),
        (
            "w1_state_id",
            [record["lineage"]["w1_state_id"] for record in normalized],
        ),
    ):
        if len(set(values)) != EXPECTED_CASES:
            raise MV2AnalysisError(f"{label}: 12개 case에서 unique해야 함")

    embedded_aliases = {
        record["model_alias"]
        for record in normalized
        if record["model_alias"] is not None
    }
    if len(embedded_aliases) > 1:
        raise MV2AnalysisError("여러 model_alias가 섞인 입력은 거부")
    if model_alias is not None:
        model_alias = _require_nonempty_string(model_alias, "model_alias")
        if embedded_aliases and embedded_aliases != {model_alias}:
            raise MV2AnalysisError("CLI/API model_alias와 record model_alias 불일치")
    elif embedded_aliases:
        model_alias = next(iter(embedded_aliases))
    else:
        model_alias = "single-model-unspecified"
    if model_alias not in MODEL_ALIASES:
        raise MV2AnalysisError("model_alias가 고정 MV-2 backbone 밖임")

    context_ids = {record["lineage"]["context_id"] for record in normalized}
    if len(context_ids) != 1:
        raise MV2AnalysisError("12개 case의 frozen context identity가 다름")

    technical_failures = [
        failure
        for record in normalized
        for failure in record["technical_failures"]
    ]
    failed_case_ids = [record["case_id"] for record in normalized if not record["pass"]]
    failed_arm_counts = {
        arm_id: sum(not record["arms"][arm_id]["success"] for record in normalized)
        for arm_id in ARM_ORDER
    }

    direction_effects: list[float] = []
    coefficient_effects: list[float] = []
    total_refresh_effects: list[float] = []
    refresh_opportunity_oracles: list[float] = []
    generic_continuation_gains: list[float] = []
    case_effects: list[dict[str, Any]] = []
    case_envelopes: list[float] = []

    for record in normalized:
        arms = record["arms"]
        complete_success = record["pass"] and all(
            arms[arm_id]["success"] for arm_id in ARM_ORDER
        )
        if complete_success:
            no_op_progress = arms[NO_OP]["progress"]
            sham_progress = arms[H0_SHAM]["progress"]
            direction = (
                arms[REFRESHED_DIRECTION_REFRESHED_COEFFICIENT]["progress"]
                - arms[FIXED_DIRECTION_REFRESHED_COEFFICIENT]["progress"]
            )
            coefficient = (
                arms[FIXED_DIRECTION_REFRESHED_COEFFICIENT]["progress"]
                - arms[FIXED_DIRECTION_FIXED_COEFFICIENT]["progress"]
            )
            total_refresh = (
                arms[REFRESHED_DIRECTION_REFRESHED_COEFFICIENT]["progress"]
                - arms[FIXED_DIRECTION_FIXED_COEFFICIENT]["progress"]
            )
            refresh_opportunity_oracle = (
                max(
                    arms[REFRESHED_DIRECTION_REFRESHED_COEFFICIENT]["progress"],
                    arms[FIXED_DIRECTION_REFRESHED_COEFFICIENT]["progress"],
                    arms[FIXED_DIRECTION_FIXED_COEFFICIENT]["progress"],
                )
                - arms[FIXED_DIRECTION_FIXED_COEFFICIENT]["progress"]
            )
            generic_continuation_gain = (
                max(
                    arms[REFRESHED_DIRECTION_REFRESHED_COEFFICIENT]["progress"],
                    arms[FIXED_DIRECTION_REFRESHED_COEFFICIENT]["progress"],
                    arms[FIXED_DIRECTION_FIXED_COEFFICIENT]["progress"],
                )
                - arms[PARTIAL]["progress"]
            )
        else:
            # ITD: 실패 case를 제거하거나 관측된 일부 arm으로 rescue하지 않는다.
            no_op_progress = 0.0
            sham_progress = 0.0
            direction = 0.0
            coefficient = 0.0
            total_refresh = 0.0
            refresh_opportunity_oracle = 0.0
            generic_continuation_gain = 0.0

        case_envelope = max(
            PRACTICAL_EFFECT_FLOOR,
            abs(no_op_progress),
            abs(sham_progress - no_op_progress),
        )
        case_envelopes.append(case_envelope)
        direction_effects.append(direction)
        coefficient_effects.append(coefficient)
        total_refresh_effects.append(total_refresh)
        refresh_opportunity_oracles.append(refresh_opportunity_oracle)
        generic_continuation_gains.append(generic_continuation_gain)
        case_effects.append(
            {
                "case_id": record["case_id"],
                "analysis_success": complete_success,
                "replay_sham_envelope": case_envelope,
                "direction_refresh_effect": direction,
                "coefficient_refresh_effect": coefficient,
                "total_refresh_effect": total_refresh,
                "refresh_opportunity_oracle": refresh_opportunity_oracle,
                "generic_continuation_gain": generic_continuation_gain,
            }
        )

    envelope = max(case_envelopes)
    direction_summary = _effect_summary(
        direction_effects, envelope=envelope, stream=0
    )
    coefficient_summary = _effect_summary(
        coefficient_effects, envelope=envelope, stream=1
    )
    total_refresh_summary = _effect_summary(
        total_refresh_effects, envelope=envelope, stream=2
    )
    refresh_opportunity_oracle = {
        "definition": "max(A,B,C)-C",
        "n": EXPECTED_CASES,
        "mean": math.fsum(refresh_opportunity_oracles) / EXPECTED_CASES,
        "trimmed_mean_10pct": _trimmed_mean(refresh_opportunity_oracles),
        "median": statistics.median(refresh_opportunity_oracles),
        "max": max(refresh_opportunity_oracles),
        "all_finite": all(
            math.isfinite(value) for value in refresh_opportunity_oracles
        ),
        "outcome_selected_upper_bound": True,
    }
    generic_continuation_gain = {
        "definition": "max(A,B,C)-partial_joint",
        "n": EXPECTED_CASES,
        "mean": math.fsum(generic_continuation_gains) / EXPECTED_CASES,
        "trimmed_mean_10pct": _trimmed_mean(generic_continuation_gains),
        "median": statistics.median(generic_continuation_gains),
        "max": max(generic_continuation_gains),
        "all_finite": all(
            math.isfinite(value) for value in generic_continuation_gains
        ),
        "refresh_kill_rescue_allowed": False,
    }

    direction_clear = bool(
        direction_summary["mean"] > envelope
        and (
            direction_summary["trimmed_mean_10pct"] > envelope
            or direction_summary["median"] > envelope
            or direction_summary["positive_count"] >= 7
        )
    )
    coefficient_clear = bool(
        coefficient_summary["mean"] > envelope
        and (
            coefficient_summary["trimmed_mean_10pct"] > envelope
            or coefficient_summary["median"] > envelope
            or coefficient_summary["positive_count"] >= 7
        )
    )
    total_clear = bool(
        total_refresh_summary["mean"] > envelope
        and (
            total_refresh_summary["trimmed_mean_10pct"] > envelope
            or total_refresh_summary["median"] > envelope
            or total_refresh_summary["positive_count"] >= 7
        )
    )
    total_nonnegative = bool(
        total_refresh_summary["mean"] >= -envelope
        and total_refresh_summary["trimmed_mean_10pct"] >= -envelope
    )
    direction_null = bool(
        direction_summary["mean"] <= envelope
        and direction_summary["trimmed_mean_10pct"] <= envelope
        and direction_summary["sign_rate"] <= 0.5
    )
    coefficient_null = bool(
        coefficient_summary["mean"] <= envelope
        and coefficient_summary["trimmed_mean_10pct"] <= envelope
        and coefficient_summary["sign_rate"] <= 0.5
    )
    direction_kill = bool(
        direction_null
        and refresh_opportunity_oracle["all_finite"]
        and refresh_opportunity_oracle["mean"] <= envelope
    )

    if technical_failures:
        verdict = "BLOCK_TECHNICAL_INVALID"
        verdict_ko = "기술 유효성 실패로 판정 차단"
    elif direction_clear and total_clear:
        verdict = "DIRECTION_REFRESH_CLEAR"
        verdict_ko = "direction mechanism과 current-method 총 기대효과 모두 명확"
    elif direction_clear:
        verdict = "DIRECTION_MECHANISM_ONLY_REDESIGN_COEFFICIENT"
        verdict_ko = "direction 신호는 있으나 A-C 총 기대효과 불충분"
    elif coefficient_clear:
        verdict = "PIVOT_FIXED_DIRECTION_DYNAMIC_COEFFICIENT"
        verdict_ko = "fixed-direction dynamic coefficient로 전환"
    elif direction_kill and coefficient_null:
        verdict = "PIVOT_STATIC_ROUTING"
        verdict_ko = "두 refresh 신호가 null이므로 static routing으로 전환"
    else:
        verdict = "INCONCLUSIVE_BOUNDED_CONTINUATION"
        verdict_ko = "핵심 신호가 경계적이어서 제한된 추가 진단 필요"

    arm_compute: dict[str, Any] = {}
    for arm_id in ARM_ORDER:
        nfes = [record["arms"][arm_id]["nfe"] for record in normalized]
        arm_compute[arm_id] = {
            "nfe_total": sum(nfes),
            "nfe_mean": math.fsum(nfes) / EXPECTED_CASES,
            "success_count": EXPECTED_CASES - failed_arm_counts[arm_id],
            "failed_count": failed_arm_counts[arm_id],
        }
    compute = {
        "case_count": EXPECTED_CASES,
        "controlled_nfe_total": sum(
            record["compute"]["controlled_nfe"] for record in normalized
        ),
        "proposal_build_count_total": sum(
            record["compute"]["proposal_build_count"] for record in normalized
        ),
        "probe_panel_count_total": sum(
            record["compute"]["probe_panel_count"] for record in normalized
        ),
        "wall_seconds_total": math.fsum(
            record["compute"]["wall_seconds"] for record in normalized
        ),
        "wall_seconds_mean": (
            math.fsum(record["compute"]["wall_seconds"] for record in normalized)
            / EXPECTED_CASES
        ),
        "arms": arm_compute,
        "w1_policy_incremental_cost_per_case": {
            REFRESHED_DIRECTION_REFRESHED_COEFFICIENT: {
                "proposal_build_count": 1,
                "probe_nfe": 12,
            },
            FIXED_DIRECTION_REFRESHED_COEFFICIENT: {
                "proposal_build_count": 0,
                "probe_nfe": 12,
            },
            FIXED_DIRECTION_FIXED_COEFFICIENT: {
                "proposal_build_count": 0,
                "probe_nfe": 0,
            },
        },
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "single_model",
        "model_alias": model_alias,
        "fixed_contract": {
            "expected_cases": EXPECTED_CASES,
            "arm_order": list(ARM_ORDER),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "practical_effect_floor": PRACTICAL_EFFECT_FLOOR,
            "matched_c_rel_tol": MATCHED_C_REL_TOL,
            "matched_c_abs_tol": MATCHED_C_ABS_TOL,
        },
        "technical_validity": {
            "pass": not technical_failures,
            "failures": technical_failures,
            "failed_case_count_in_denominator": len(failed_case_ids),
            "failed_case_ids": failed_case_ids,
            "failed_arm_counts": failed_arm_counts,
        },
        "replay_sham_envelope": {
            "definition": "max(1e-4, abs(no_op), abs(h0_sham-no_op))",
            "value": envelope,
            "per_case": case_envelopes,
        },
        "effects": {
            "direction_refresh": direction_summary,
            "coefficient_refresh": coefficient_summary,
            "total_refresh": total_refresh_summary,
        },
        "refresh_opportunity_oracle": refresh_opportunity_oracle,
        "generic_continuation_gain": generic_continuation_gain,
        "decision": {
            "verdict": verdict,
            "verdict_ko": verdict_ko,
            "direction_clear": direction_clear,
            "coefficient_clear": coefficient_clear,
            "total_clear": total_clear,
            "total_nonnegative": total_nonnegative,
            "direction_null": direction_null,
            "coefficient_null": coefficient_null,
            "direction_kill": direction_kill,
        },
        "compute": compute,
        "case_effects": case_effects,
    }


# JSONL-facing name kept deliberately thin: the scientific API is case-level.
analyze_records = analyze_case_records


def render_markdown(summary: Mapping[str, Any]) -> str:
    """단일 모델 summary를 한국어 독립 분석 보고서로 렌더링한다."""

    technical = summary["technical_validity"]
    envelope = summary["replay_sham_envelope"]["value"]
    direction = summary["effects"]["direction_refresh"]
    coefficient = summary["effects"]["coefficient_refresh"]
    total_refresh = summary["effects"]["total_refresh"]
    refresh_oracle = summary["refresh_opportunity_oracle"]
    generic_gain = summary["generic_continuation_gain"]
    decision = summary["decision"]
    compute = summary["compute"]
    policy_cost = compute["w1_policy_incremental_cost_per_case"]

    def metric_row(label: str, metric: Mapping[str, Any]) -> str:
        ci = metric["bootstrap"]["mean_ci95"]
        return (
            f"| {label} | {metric['mean']:.9g} | "
            f"{metric['trimmed_mean_10pct']:.9g} | {metric['median']:.9g} | "
            f"{metric['positive_count']}/12 | "
            f"[{ci[0]:.9g}, {ci[1]:.9g}] |"
        )

    failure_lines = (
        "\n".join(f"- `{failure}`" for failure in technical["failures"])
        if technical["failures"]
        else "- 없음"
    )
    return "\n".join(
        [
            "# MV-2 단일 모델 refresh 독립 분석",
            "",
            f"- model: `{summary['model_alias']}`",
            f"- verdict: `{decision['verdict']}` — {decision['verdict_ko']}",
            f"- technical validity: `{'PASS' if technical['pass'] else 'BLOCK'}`",
            f"- ITD denominator: `12` (실패 case "
            f"`{technical['failed_case_count_in_denominator']}`건 포함)",
            "",
            "## 고정 분석 계약",
            "",
            "- direction effect: `progress(A) - progress(B)`",
            "- coefficient effect: `progress(B) - progress(C)`",
            "- total refresh 기대효과: `progress(A) - progress(C)`",
            "- 실패 case는 사후 제외하지 않고 세 contrast에 `0`을 기여한다.",
            f"- replay+sham envelope `e={envelope:.9g}`",
            f"- outcome-blind practical floor: `{PRACTICAL_EFFECT_FLOOR:.9g}`",
            f"- bootstrap: seed `{BOOTSTRAP_SEED}`, resamples `{BOOTSTRAP_SAMPLES}`",
            "- hierarchy: direction이 유일한 primary이고 coefficient는 direction 실패 시 conditional, total은 secondary 기대효과다.",
            "",
            "## 효과",
            "",
            "| contrast | mean | 10% trimmed mean | median | positive sign | paired bootstrap mean 95% CI |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            metric_row("direction refresh", direction),
            metric_row("coefficient refresh", coefficient),
            metric_row("total refresh (method 기대효과 proxy)", total_refresh),
            "",
            "## Refresh opportunity oracle",
            "",
            "- 정의: `max(A,B,C) - C`",
            "- outcome-selected upper bound이며 실현 가능한 method 성능으로 해석하지 않는다.",
            f"- mean: `{refresh_oracle['mean']:.9g}`",
            f"- 10% trimmed mean: `{refresh_oracle['trimmed_mean_10pct']:.9g}`",
            f"- median: `{refresh_oracle['median']:.9g}`",
            "",
            "## Generic continuation gain",
            "",
            "- 정의: `max(A,B,C) - partial_joint`",
            "- 일반적인 second-half-step 이득이므로 refresh kill을 구제하지 않는다.",
            f"- mean: `{generic_gain['mean']:.9g}`",
            "",
            "## Compute/NFE",
            "",
            f"- controlled NFE total: `{compute['controlled_nfe_total']}`",
            f"- proposal build total: `{compute['proposal_build_count_total']}`",
            f"- probe panel total: `{compute['probe_panel_count_total']}`",
            f"- wall seconds total: `{compute['wall_seconds_total']:.6g}`",
            "- W1 controller incremental cost/case (diagnostic outcome forward 제외):",
            f"  - A: proposal build `{policy_cost[REFRESHED_DIRECTION_REFRESHED_COEFFICIENT]['proposal_build_count']}`, probe NFE `{policy_cost[REFRESHED_DIRECTION_REFRESHED_COEFFICIENT]['probe_nfe']}`",
            f"  - B: proposal build `{policy_cost[FIXED_DIRECTION_REFRESHED_COEFFICIENT]['proposal_build_count']}`, probe NFE `{policy_cost[FIXED_DIRECTION_REFRESHED_COEFFICIENT]['probe_nfe']}`",
            f"  - C: proposal build `{policy_cost[FIXED_DIRECTION_FIXED_COEFFICIENT]['proposal_build_count']}`, probe NFE `{policy_cost[FIXED_DIRECTION_FIXED_COEFFICIENT]['probe_nfe']}`",
            "",
            "## 기술 gate 실패",
            "",
            failure_lines,
            "",
            "## 해석 경계",
            "",
            "- 이 문서는 한 모델의 사전 고정된 mechanism diagnostic만 판정한다.",
            "- total refresh는 equal-C continuation의 absolute utility proxy이며 downstream 성능 예측이 아니다.",
            "- secondary metric 하나나 bootstrap endpoint 하나만으로 핵심 판정을 뒤집지 않는다.",
            "- technical gate 실패 시 scientific signal과 무관하게 판정을 차단한다.",
            "",
        ]
    )


def load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    """Canonical case JSONL 또는 runner의 sanitized stream을 읽는다."""

    records: list[Mapping[str, Any]] = []
    stream_run_id: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise MV2AnalysisError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(record, Mapping):
                raise MV2AnalysisError(
                    f"{path}:{line_number}: JSON object여야 함"
                )
            if "payload" in record:
                expected_wrapper = {
                    "schema_version",
                    "run_id",
                    "sequence",
                    "recorded_at",
                    "event",
                    "payload",
                }
                if set(record) != expected_wrapper:
                    raise MV2AnalysisError(
                        f"{path}:{line_number}: sanitized wrapper schema 위반"
                    )
                if (
                    record["schema_version"] != RUN_STREAM_SCHEMA
                    or record["event"] != RUN_STREAM_EVENT
                    or record["sequence"] != len(records)
                    or not isinstance(record["run_id"], str)
                    or not record["run_id"]
                    or not isinstance(record["recorded_at"], str)
                    or not record["recorded_at"]
                    or not isinstance(record["payload"], Mapping)
                ):
                    raise MV2AnalysisError(
                        f"{path}:{line_number}: sanitized stream identity/order 위반"
                    )
                if stream_run_id is None:
                    stream_run_id = record["run_id"]
                elif record["run_id"] != stream_run_id:
                    raise MV2AnalysisError(
                        f"{path}:{line_number}: 여러 run_id가 섞임"
                    )
                record = record["payload"]
            records.append(record)
    return records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MV-2 single-model refresh diagnostic analyzer",
        allow_abbrev=False,
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--model-alias", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = analyze_case_records(
            load_jsonl(args.input_jsonl),
            model_alias=args.model_alias,
        )
    except (OSError, MV2AnalysisError) as error:
        print(f"MV-2 analysis rejected: {error}", file=sys.stderr)
        return 2

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_report.write_text(render_markdown(summary), encoding="utf-8")
    return 0 if summary["technical_validity"]["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
