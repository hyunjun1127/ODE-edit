"""Fail-closed analysis for the K=4 native-distance Motivation diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .quarter_step_refresh import (
    COEFFICIENT_PREFIX,
    COMMON_STEP_1,
    FIXED_PREFIX,
    MATCHED_C_ABS_TOL,
    MATCHED_C_REL_TOL,
    NATIVE_ORDERED_FULL,
    NATIVE_ORDERED_SPLIT4,
    NO_OP_REPLAY,
    QSTEP_BOOTSTRAP_RESAMPLES,
    QSTEP_BOOTSTRAP_SEED,
    QSTEP_BRANCH_ORDER,
    QSTEP_CASE_COUNT,
    QSTEP_K,
    QSTEP_PRACTICAL_EFFECT_FLOOR,
    QSTEP_STREAM_SCHEMA,
    REFRESHED_PREFIX,
    _checkpoint_arm,
)


ANALYSIS_SCHEMA = "ode-edit-quarter-step-analysis/v1"
MODEL_ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")


class QuarterStepAnalysisError(ValueError):
    pass


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QuarterStepAnalysisError(f"{path}: mapping이어야 함")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], path: str) -> None:
    expected_set = set(expected)
    if set(value) != expected_set:
        raise QuarterStepAnalysisError(
            f"{path}: exact key set 위반; "
            f"missing={sorted(expected_set - set(value))}, "
            f"extra={sorted(set(value) - expected_set)}"
        )


def _finite(value: Any, path: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise QuarterStepAnalysisError(f"{path}: finite scalar여야 함")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise QuarterStepAnalysisError(f"{path}: finite scalar여야 함") from exc
    if not math.isfinite(normalized) or (nonnegative and normalized < 0.0):
        raise QuarterStepAnalysisError(f"{path}: 유효 범위 밖 scalar")
    return normalized


def _integer(value: Any, path: str, *, nonnegative: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QuarterStepAnalysisError(f"{path}: integer여야 함")
    if nonnegative and value < 0:
        raise QuarterStepAnalysisError(f"{path}: nonnegative integer여야 함")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise QuarterStepAnalysisError(f"{path}: bool이어야 함")
    return value


def _sha256(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise QuarterStepAnalysisError(f"{path}: lowercase SHA-256이어야 함")
    return value


def load_analysis_cases(path: str | Path) -> list[Mapping[str, Any]]:
    source = Path(path).expanduser().resolve(strict=True)
    records: list[Mapping[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                wrapper = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QuarterStepAnalysisError(
                    f"line {line_number}: invalid JSON"
                ) from exc
            wrapper = _mapping(wrapper, f"line[{line_number}]")
            _exact_keys(
                wrapper,
                (
                    "schema_version",
                    "run_id",
                    "sequence",
                    "recorded_at",
                    "event",
                    "payload",
                ),
                f"line[{line_number}]",
            )
            if (
                wrapper["schema_version"] != QSTEP_STREAM_SCHEMA
                or wrapper["event"] != "quarter_step_analysis_case"
                or wrapper["sequence"] != len(records)
            ):
                raise QuarterStepAnalysisError(
                    f"line {line_number}: stream identity/order 위반"
                )
            records.append(_mapping(wrapper["payload"], f"line[{line_number}].payload"))
    return records


def _expected_step(arm_id: str) -> int:
    if arm_id == NO_OP_REPLAY:
        return 0
    if arm_id == COMMON_STEP_1:
        return 1
    if arm_id in (NATIVE_ORDERED_FULL, NATIVE_ORDERED_SPLIT4):
        return QSTEP_K
    for step in range(2, QSTEP_K + 1):
        if arm_id in {
            _checkpoint_arm(REFRESHED_PREFIX, step),
            _checkpoint_arm(COEFFICIENT_PREFIX, step),
            _checkpoint_arm(FIXED_PREFIX, step),
        }:
            return step
    raise QuarterStepAnalysisError(f"unknown arm: {arm_id}")


def _normalize_case(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    path = f"records[{index}]"
    _exact_keys(
        record,
        (
            "model_alias",
            "case_id",
            "request_id",
            "pass",
            "technical",
            "branch_order",
            "arms",
            "budgets",
            "lineage",
            "geometry",
            "compute",
        ),
        path,
    )
    model_alias = record["model_alias"]
    if model_alias not in MODEL_ALIASES:
        raise QuarterStepAnalysisError(f"{path}.model_alias: fixed backbone 밖")
    case_id_raw = record["case_id"]
    if isinstance(case_id_raw, bool) or not isinstance(case_id_raw, (str, int)):
        raise QuarterStepAnalysisError(f"{path}.case_id: str/int여야 함")
    case_id = str(case_id_raw)
    if not case_id:
        raise QuarterStepAnalysisError(f"{path}.case_id: 비어 있음")
    request_id = _sha256(record["request_id"], f"{path}.request_id")
    case_pass = _boolean(record["pass"], f"{path}.pass")

    technical_raw = _mapping(record["technical"], f"{path}.technical")
    technical_keys = (
        "exact_panel",
        "lineage_exact",
        "matched_per_hop_c",
        "rollback_exact",
        "firewall_pass",
        "receipt_before_outcome",
        "native_split_control",
    )
    _exact_keys(technical_raw, technical_keys, f"{path}.technical")
    technical = {
        key: _boolean(technical_raw[key], f"{path}.technical.{key}")
        for key in technical_keys
    }
    failures = [
        f"{path}.technical.{key}=false" for key, value in technical.items() if not value
    ]
    if not case_pass:
        failures.append(f"{path}.pass=false")

    branch_order = record["branch_order"]
    if not isinstance(branch_order, list) or tuple(branch_order) != QSTEP_BRANCH_ORDER:
        raise QuarterStepAnalysisError(f"{path}.branch_order: exact arm order 위반")
    arms_raw = _mapping(record["arms"], f"{path}.arms")
    if set(arms_raw) != set(QSTEP_BRANCH_ORDER):
        raise QuarterStepAnalysisError(f"{path}.arms: exact arm set 위반")
    arms: dict[str, dict[str, Any]] = {}
    for arm_id in QSTEP_BRANCH_ORDER:
        arm = _mapping(arms_raw[arm_id], f"{path}.arms.{arm_id}")
        _exact_keys(
            arm,
            (
                "progress",
                "endpoint_c_energy",
                "cumulative_path_distance",
                "step_index",
                "nfe",
                "success",
            ),
            f"{path}.arms.{arm_id}",
        )
        arms[arm_id] = {
            "progress": _finite(arm["progress"], f"{path}.arms.{arm_id}.progress"),
            "endpoint_c_energy": _finite(
                arm["endpoint_c_energy"],
                f"{path}.arms.{arm_id}.endpoint_c_energy",
                nonnegative=True,
            ),
            "cumulative_path_distance": _finite(
                arm["cumulative_path_distance"],
                f"{path}.arms.{arm_id}.cumulative_path_distance",
                nonnegative=True,
            ),
            "step_index": _integer(
                arm["step_index"], f"{path}.arms.{arm_id}.step_index"
            ),
            "nfe": _integer(arm["nfe"], f"{path}.arms.{arm_id}.nfe"),
            "success": _boolean(arm["success"], f"{path}.arms.{arm_id}.success"),
        }

    budgets_raw = _mapping(record["budgets"], f"{path}.budgets")
    budget_keys = (
        "native_c_energy",
        "per_hop_c_energy",
        "native_c_distance",
        "hop_c_distance",
        "probe_c_distance",
    )
    _exact_keys(budgets_raw, budget_keys, f"{path}.budgets")
    budgets = {
        key: _finite(budgets_raw[key], f"{path}.budgets.{key}", nonnegative=True)
        for key in budget_keys
    }

    lineage = _mapping(record["lineage"], f"{path}.lineage")
    _exact_keys(
        lineage,
        (
            "origin_state_id",
            "target_token_sha256",
            "direct_z_tensor_sha256",
            "context_id",
            "final_lineage_ids",
            "native_split_final_lineage_id",
        ),
        f"{path}.lineage",
    )
    final_lineages = _mapping(
        lineage["final_lineage_ids"], f"{path}.lineage.final_lineage_ids"
    )
    _exact_keys(
        final_lineages,
        (REFRESHED_PREFIX, COEFFICIENT_PREFIX, FIXED_PREFIX),
        f"{path}.lineage.final_lineage_ids",
    )
    normalized_lineage = {
        "origin_state_id": _sha256(
            lineage["origin_state_id"], f"{path}.lineage.origin_state_id"
        ),
        "target_token_sha256": _sha256(
            lineage["target_token_sha256"], f"{path}.lineage.target_token_sha256"
        ),
        "direct_z_tensor_sha256": _sha256(
            lineage["direct_z_tensor_sha256"],
            f"{path}.lineage.direct_z_tensor_sha256",
        ),
        "context_id": _sha256(lineage["context_id"], f"{path}.lineage.context_id"),
        "final_lineage_ids": {
            key: _sha256(value, f"{path}.lineage.final_lineage_ids.{key}")
            for key, value in final_lineages.items()
        },
        "native_split_final_lineage_id": _sha256(
            lineage["native_split_final_lineage_id"],
            f"{path}.lineage.native_split_final_lineage_id",
        ),
    }

    geometry = _mapping(record["geometry"], f"{path}.geometry")
    _exact_keys(geometry, ("w0_to_current_c_cosines",), f"{path}.geometry")
    cosines = _mapping(
        geometry["w0_to_current_c_cosines"],
        f"{path}.geometry.w0_to_current_c_cosines",
    )
    _exact_keys(
        cosines,
        (REFRESHED_PREFIX, COEFFICIENT_PREFIX, FIXED_PREFIX),
        f"{path}.geometry.w0_to_current_c_cosines",
    )

    compute = _mapping(record["compute"], f"{path}.compute")
    _exact_keys(
        compute,
        ("controlled_nfe", "proposal_build_count", "probe_panel_count", "wall_seconds"),
        f"{path}.compute",
    )
    normalized_compute = {
        "controlled_nfe": _integer(
            compute["controlled_nfe"], f"{path}.compute.controlled_nfe"
        ),
        "proposal_build_count": _integer(
            compute["proposal_build_count"], f"{path}.compute.proposal_build_count"
        ),
        "probe_panel_count": _integer(
            compute["probe_panel_count"], f"{path}.compute.probe_panel_count"
        ),
        "wall_seconds": _finite(
            compute["wall_seconds"], f"{path}.compute.wall_seconds", nonnegative=True
        ),
    }

    if case_pass:
        if (
            any(value <= 0.0 for value in budgets.values())
            or normalized_compute["controlled_nfe"] != 98
            or normalized_compute["proposal_build_count"] != 5
            or normalized_compute["probe_panel_count"] != 7
            or any(not arms[arm_id]["success"] for arm_id in QSTEP_BRANCH_ORDER)
            or any(arms[arm_id]["nfe"] != 1 for arm_id in QSTEP_BRANCH_ORDER)
        ):
            failures.append(f"{path}: compute/success contract 위반")
        if not math.isclose(
            budgets["native_c_energy"],
            budgets["native_c_distance"] ** 2,
            rel_tol=MATCHED_C_REL_TOL,
            abs_tol=MATCHED_C_ABS_TOL,
        ) or not math.isclose(
            budgets["per_hop_c_energy"],
            budgets["native_c_energy"] / 16.0,
            rel_tol=MATCHED_C_REL_TOL,
            abs_tol=MATCHED_C_ABS_TOL,
        ):
            failures.append(f"{path}: native/per-hop C budget 위반")
        if not math.isclose(
            budgets["hop_c_distance"],
            budgets["native_c_distance"] / 4.0,
            rel_tol=MATCHED_C_REL_TOL,
            abs_tol=MATCHED_C_ABS_TOL,
        ) or not math.isclose(
            budgets["probe_c_distance"],
            budgets["native_c_distance"] / 64.0,
            rel_tol=MATCHED_C_REL_TOL,
            abs_tol=MATCHED_C_ABS_TOL,
        ):
            failures.append(f"{path}: hop/probe distance 위반")
        for arm_id in QSTEP_BRANCH_ORDER:
            expected_step = _expected_step(arm_id)
            if arms[arm_id]["step_index"] != expected_step:
                failures.append(f"{path}.arms.{arm_id}: step index 위반")
            expected_distance = (
                0.0
                if arm_id == NO_OP_REPLAY
                else expected_step * budgets["hop_c_distance"]
            )
            if not math.isclose(
                arms[arm_id]["cumulative_path_distance"],
                expected_distance,
                rel_tol=MATCHED_C_REL_TOL,
                abs_tol=MATCHED_C_ABS_TOL,
            ):
                failures.append(f"{path}.arms.{arm_id}: path distance 위반")
        for step in range(1, QSTEP_K + 1):
            arm_id = COMMON_STEP_1 if step == 1 else _checkpoint_arm(FIXED_PREFIX, step)
            expected_energy = (step * budgets["hop_c_distance"]) ** 2
            if not math.isclose(
                arms[arm_id]["endpoint_c_energy"],
                expected_energy,
                rel_tol=2e-4,
                abs_tol=2e-4,
            ):
                failures.append(f"{path}.arms.{arm_id}: fixed endpoint C 위반")
        for arm_id in (NATIVE_ORDERED_FULL, NATIVE_ORDERED_SPLIT4):
            if not math.isclose(
                arms[arm_id]["endpoint_c_energy"],
                budgets["native_c_energy"],
                rel_tol=2e-4,
                abs_tol=2e-4,
            ):
                failures.append(f"{path}.arms.{arm_id}: native endpoint C 위반")

    return {
        "model_alias": model_alias,
        "case_id": case_id,
        "request_id": request_id,
        "pass": case_pass,
        "technical": technical,
        "technical_failures": failures,
        "arms": arms,
        "budgets": budgets,
        "lineage": normalized_lineage,
        "compute": normalized_compute,
    }


def _trimmed_mean(values: Sequence[float]) -> float:
    ordered = sorted(values)
    trim = max(1, len(ordered) // 10)
    core = ordered[trim:-trim]
    return math.fsum(core) / len(core)


def _bootstrap_ci(values: Sequence[float], stream: int) -> tuple[float, float]:
    rng = random.Random(QSTEP_BOOTSTRAP_SEED + stream)
    n = len(values)
    samples = sorted(
        math.fsum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(QSTEP_BOOTSTRAP_RESAMPLES)
    )
    lo = samples[int(0.025 * len(samples))]
    hi = samples[min(len(samples) - 1, int(0.975 * len(samples)))]
    return lo, hi


def _summary(values: Sequence[float], *, envelope: float, stream: int) -> dict[str, Any]:
    lo, hi = _bootstrap_ci(values, stream)
    return {
        "n": len(values),
        "mean": math.fsum(values) / len(values),
        "trimmed_mean_10pct": _trimmed_mean(values),
        "median": statistics.median(values),
        "bootstrap_95_ci": [lo, hi],
        "positive_count": sum(value > envelope for value in values),
        "negative_count": sum(value < -envelope for value in values),
        "within_noise_count": sum(abs(value) <= envelope for value in values),
        "practical_envelope": envelope,
    }


def analyze_case_records(
    records: Iterable[Mapping[str, Any]], *, model_alias: str
) -> dict[str, Any]:
    materialized = list(records)
    if len(materialized) != QSTEP_CASE_COUNT:
        raise QuarterStepAnalysisError(
            f"exact case count 위반: expected={QSTEP_CASE_COUNT}, actual={len(materialized)}"
        )
    if model_alias not in MODEL_ALIASES:
        raise QuarterStepAnalysisError("model_alias가 fixed backbone 밖")
    normalized = [
        _normalize_case(_mapping(record, f"records[{index}]"), index)
        for index, record in enumerate(materialized)
    ]
    if {record["model_alias"] for record in normalized} != {model_alias}:
        raise QuarterStepAnalysisError("record와 requested model_alias 불일치")
    for label, values in (
        ("case_id", [record["case_id"] for record in normalized]),
        ("request_id", [record["request_id"] for record in normalized]),
        ("origin_state_id", [record["lineage"]["origin_state_id"] for record in normalized]),
    ):
        if len(set(values)) != QSTEP_CASE_COUNT:
            raise QuarterStepAnalysisError(f"{label}: 12개 case에서 unique해야 함")
    if len({record["lineage"]["context_id"] for record in normalized}) != 1:
        raise QuarterStepAnalysisError("12개 case의 frozen context가 다름")

    technical_failures = [
        failure
        for record in normalized
        for failure in record["technical_failures"]
    ]
    complete = [
        record["pass"]
        and not record["technical_failures"]
        and all(record["arms"][arm]["success"] for arm in QSTEP_BRANCH_ORDER)
        for record in normalized
    ]
    native_split_gaps = [
        (
            record["arms"][NATIVE_ORDERED_SPLIT4]["progress"]
            - record["arms"][NATIVE_ORDERED_FULL]["progress"]
        )
        if ok
        else 0.0
        for record, ok in zip(normalized, complete)
    ]
    replay_values = [
        record["arms"][NO_OP_REPLAY]["progress"] if ok else 0.0
        for record, ok in zip(normalized, complete)
    ]
    envelope = max(
        QSTEP_PRACTICAL_EFFECT_FLOOR,
        *(abs(value) for value in native_split_gaps),
        *(abs(value) for value in replay_values),
    )

    direction: list[float] = []
    coefficient: list[float] = []
    total: list[float] = []
    refreshed_vs_native: list[float] = []
    trajectory: dict[str, dict[str, list[float]]] = {
        policy: {str(step): [] for step in range(1, QSTEP_K + 1)}
        for policy in (REFRESHED_PREFIX, COEFFICIENT_PREFIX, FIXED_PREFIX)
    }
    case_effects: list[dict[str, Any]] = []
    for record, ok in zip(normalized, complete):
        arms = record["arms"]
        a4 = arms[_checkpoint_arm(REFRESHED_PREFIX, 4)]["progress"] if ok else 0.0
        b4 = arms[_checkpoint_arm(COEFFICIENT_PREFIX, 4)]["progress"] if ok else 0.0
        c4 = arms[_checkpoint_arm(FIXED_PREFIX, 4)]["progress"] if ok else 0.0
        native = arms[NATIVE_ORDERED_FULL]["progress"] if ok else 0.0
        d = a4 - b4
        c = b4 - c4
        t = a4 - c4
        direction.append(d)
        coefficient.append(c)
        total.append(t)
        refreshed_vs_native.append(a4 - native)
        for policy in trajectory:
            for step in range(1, QSTEP_K + 1):
                arm_id = COMMON_STEP_1 if step == 1 else _checkpoint_arm(policy, step)
                trajectory[policy][str(step)].append(
                    arms[arm_id]["progress"] if ok else 0.0
                )
        case_effects.append(
            {
                "case_id": record["case_id"],
                "analysis_success": ok,
                "native_split_gap": native_split_gaps[len(case_effects)],
                "direction_refresh_effect_A4_minus_B4": d,
                "coefficient_refresh_effect_B4_minus_C4": c,
                "total_refresh_effect_A4_minus_C4": t,
                "refreshed_path_vs_native_A4_minus_MEMIT": a4 - native,
            }
        )

    effects = {
        "direction_refresh_A4_minus_B4": _summary(direction, envelope=envelope, stream=0),
        "coefficient_refresh_B4_minus_C4": _summary(
            coefficient, envelope=envelope, stream=1
        ),
        "total_refresh_A4_minus_C4": _summary(total, envelope=envelope, stream=2),
        "refreshed_path_vs_native_A4_minus_MEMIT": _summary(
            refreshed_vs_native, envelope=envelope, stream=3
        ),
        "native_split_application_gap_split4_minus_one_shot": _summary(
            native_split_gaps,
            envelope=QSTEP_PRACTICAL_EFFECT_FLOOR,
            stream=4,
        ),
    }
    direction_clear = bool(
        effects["direction_refresh_A4_minus_B4"]["mean"] > envelope
        and effects["direction_refresh_A4_minus_B4"]["positive_count"] >= 7
    )
    coefficient_clear = bool(
        effects["coefficient_refresh_B4_minus_C4"]["mean"] > envelope
        and effects["coefficient_refresh_B4_minus_C4"]["positive_count"] >= 7
    )
    total_clear = bool(
        effects["total_refresh_A4_minus_C4"]["mean"] > envelope
        and effects["total_refresh_A4_minus_C4"]["positive_count"] >= 7
    )
    total_harm = bool(
        effects["total_refresh_A4_minus_C4"]["mean"] < -envelope
        and effects["total_refresh_A4_minus_C4"]["negative_count"] >= 7
    )
    if technical_failures:
        verdict = "BLOCK_TECHNICAL_INVALID"
        verdict_ko = "기술 유효성 실패로 과학 판정을 차단"
    elif total_harm:
        verdict = "KILL_REFRESH_AT_NATIVE_DISTANCE"
        verdict_ko = "native-distance에서 refresh가 일관되게 해로워 중단"
    elif direction_clear and total_clear:
        verdict = "DIRECTION_REFRESH_CLEAR"
        verdict_ko = "방향 재계산 및 총 refresh 기대효과가 명확"
    elif coefficient_clear and not direction_clear:
        verdict = "PIVOT_COEFFICIENT_ONLY"
        verdict_ko = "방향 고정·계수 갱신으로 전환"
    elif not direction_clear and not coefficient_clear and not total_clear:
        verdict = "NO_CLEAR_REFRESH_SIGNAL"
        verdict_ko = "1/4 step에서도 명확한 refresh 신호가 없음"
    else:
        verdict = "INCONCLUSIVE_BOUNDED_CONTINUATION"
        verdict_ko = "효과가 혼재해 제한된 후속 진단만 허용"

    trajectory_summary = {
        policy: {
            step: {
                "mean_progress": math.fsum(values) / len(values),
                "median_progress": statistics.median(values),
            }
            for step, values in by_step.items()
        }
        for policy, by_step in trajectory.items()
    }
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "scope": "single_model",
        "model_alias": model_alias,
        "fixed_contract": {
            "case_count": QSTEP_CASE_COUNT,
            "k": QSTEP_K,
            "branch_order": list(QSTEP_BRANCH_ORDER),
            "bootstrap_seed": QSTEP_BOOTSTRAP_SEED,
            "bootstrap_resamples": QSTEP_BOOTSTRAP_RESAMPLES,
            "practical_effect_floor": QSTEP_PRACTICAL_EFFECT_FLOOR,
            "primary_effects": ["A4-B4", "B4-C4", "A4-C4"],
            "best_hop_selection_forbidden": True,
        },
        "technical_validity": {
            "pass": not technical_failures,
            "failures": technical_failures,
            "successful_case_count": sum(complete),
            "failed_case_count": QSTEP_CASE_COUNT - sum(complete),
            "itd_denominator": QSTEP_CASE_COUNT,
        },
        "practical_envelope": {
            "value": envelope,
            "definition": "max(1e-4, |no-op|, |native split4 - one-shot|)",
        },
        "effects": effects,
        "trajectory_secondary": trajectory_summary,
        "compute": {
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
        },
        "case_effects": case_effects,
        "verdict": verdict,
        "verdict_ko": verdict_ko,
    }


def render_markdown(analysis: Mapping[str, Any]) -> str:
    effects = _mapping(analysis["effects"], "analysis.effects")
    lines = [
        f"# Quarter-step Motivation 분석 — {analysis['model_alias']}",
        "",
        f"- 판정: `{analysis['verdict']}` — {analysis['verdict_ko']}",
        f"- technical validity: `{analysis['technical_validity']['pass']}`",
        f"- practical envelope: `{analysis['practical_envelope']['value']:.8f}`",
        "- 주효과는 사전등록된 step 4만 사용하며 best-hop 선택은 하지 않음",
        "",
        "| 효과 | mean | median | 95% bootstrap CI | + / - / noise |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("방향 A4-B4", "direction_refresh_A4_minus_B4"),
        ("계수 B4-C4", "coefficient_refresh_B4_minus_C4"),
        ("총 refresh A4-C4", "total_refresh_A4_minus_C4"),
        ("A4-native MEMIT", "refreshed_path_vs_native_A4_minus_MEMIT"),
        ("native split4-one-shot", "native_split_application_gap_split4_minus_one_shot"),
    ):
        item = effects[key]
        lo, hi = item["bootstrap_95_ci"]
        lines.append(
            f"| {label} | {item['mean']:.8f} | {item['median']:.8f} | "
            f"[{lo:.8f}, {hi:.8f}] | {item['positive_count']} / "
            f"{item['negative_count']} / {item['within_noise_count']} |"
        )
    lines.extend(
        [
            "",
            "## 해석 경계",
            "",
            "- `A`: 방향과 계수를 매 step 재계산.",
            "- `B`: W0 방향을 고정하고 계수만 매 step 재계산.",
            "- `C`: W0 방향과 계수를 모두 고정.",
            "- native one-shot/split4 차이는 반복 적용 수치 잡음 대조군이며 방법 효과가 아님.",
            "- 이 결과는 Motivation diagnostic이며 최종 ODE-Edit 성능 claim이 아님.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ode-edit-quarter-step-analysis", allow_abbrev=False
    )
    parser.add_argument("--analysis-cases", required=True)
    parser.add_argument("--model", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analysis = analyze_case_records(
        load_analysis_cases(args.analysis_cases), model_alias=args.model
    )
    json_path = Path(args.output_json).expanduser().resolve()
    markdown_path = Path(args.output_markdown).expanduser().resolve()
    if json_path.exists() or markdown_path.exists():
        raise QuarterStepAnalysisError("analysis output must be exclusive")
    json_path.write_text(
        json.dumps(analysis, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(analysis), encoding="utf-8")
    print(json.dumps({"verdict": analysis["verdict"], "model": args.model}))
    return 0 if analysis["technical_validity"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
