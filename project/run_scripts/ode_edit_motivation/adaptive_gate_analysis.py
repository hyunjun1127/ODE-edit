"""Deterministic compact analysis for the Session 01 adaptive-gate pair.

The analyzer consumes only scalar ``analysis_cases.jsonl`` and ``features.jsonl``
streams after a model run is terminal.  It never opens direct-z tensors, raw
prompts, model weights, full logs, or outcome-time controller inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .adaptive_gate_refresh import (
    AGATE_BOOTSTRAP_RESAMPLES,
    AGATE_BOOTSTRAP_SEED,
    AGATE_BRANCH_ORDER,
    AGATE_CASE_COUNT,
    AGATE_PRACTICAL_FLOOR,
    AGATE_PROBE_PANEL_COUNT,
    AGATE_CONTROLLED_NFE,
    COEFFICIENT_FINAL,
    FIXED_FINAL,
    GATE_RELATIVE_MARGIN,
    GATED_FINAL,
    REFRESHED_FINAL,
    STREAM_SCHEMA,
    TRACK_ALPHAEDIT,
    TRACK_MEMIT,
)
from .manifests import MODEL_SPECS
from .quarter_step_refresh import (
    NATIVE_ORDERED_FULL,
    NATIVE_ORDERED_SPLIT4,
    NO_OP_REPLAY,
    QSTEP_HOP_FRACTION,
    QSTEP_K,
    QSTEP_PROBE_FRACTION,
)


ANALYSIS_SCHEMA = "ode-edit-adaptive-gate-analysis/v1"
ANALYSIS_EVENT = "adaptive_gate_analysis_case"
FEATURE_EVENT = "adaptive_gate_feature"
MODEL_ALIASES = tuple(sorted(MODEL_SPECS))
TRACKS = (TRACK_MEMIT, TRACK_ALPHAEDIT)
TECHNICAL_KEYS = (
    "exact_panel",
    "lineage_exact",
    "matched_per_hop_c",
    "rollback_exact",
    "firewall_pass",
    "receipt_before_outcome",
    "native_split_control",
    "precomputed_cache_only",
    "projector_read_only",
)


class AdaptiveGateAnalysisError(ValueError):
    """A scalar adaptive-gate stream differs from the locked contract."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdaptiveGateAnalysisError(f"{path}: mapping required")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], path: str) -> None:
    expected_set = set(expected)
    if set(value) != expected_set:
        raise AdaptiveGateAnalysisError(
            f"{path}: keys differ; missing={sorted(expected_set - set(value))}, "
            f"extra={sorted(set(value) - expected_set)}"
        )


def _finite(value: Any, path: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise AdaptiveGateAnalysisError(f"{path}: finite scalar required")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AdaptiveGateAnalysisError(f"{path}: finite scalar required") from exc
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise AdaptiveGateAnalysisError(f"{path}: invalid finite scalar")
    return result


def _integer(value: Any, path: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdaptiveGateAnalysisError(f"{path}: integer required")
    if value < (1 if positive else 0):
        raise AdaptiveGateAnalysisError(f"{path}: integer outside range")
    return value


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise AdaptiveGateAnalysisError(f"{path}: boolean required")
    return value


def load_stream(
    path: str | Path,
    *,
    expected_event: str,
) -> list[Mapping[str, Any]]:
    source = Path(path).expanduser().resolve(strict=True)
    records: list[Mapping[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                wrapper = _mapping(json.loads(line), f"line[{line_number}]")
            except json.JSONDecodeError as exc:
                raise AdaptiveGateAnalysisError(
                    f"line {line_number}: invalid JSON"
                ) from exc
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
                wrapper["schema_version"] != STREAM_SCHEMA
                or wrapper["event"] != expected_event
                or wrapper["sequence"] != len(records)
            ):
                raise AdaptiveGateAnalysisError(
                    f"line {line_number}: stream identity/order differs from lock"
                )
            records.append(
                _mapping(wrapper["payload"], f"line[{line_number}].payload")
            )
    return records


def _normalize_arm(value: Any, path: str) -> dict[str, Any]:
    arm = _mapping(value, path)
    _exact_keys(arm, ("progress", "endpoint_c_energy", "success"), path)
    return {
        "progress": _finite(arm["progress"], f"{path}.progress"),
        "endpoint_c_energy": _finite(
            arm["endpoint_c_energy"], f"{path}.endpoint_c_energy", nonnegative=True
        ),
        "success": _boolean(arm["success"], f"{path}.success"),
    }


def _normalize_case(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    path = f"records[{index}]"
    _exact_keys(
        record,
        (
            "model_alias",
            "editor_track",
            "case_id",
            "request_id",
            "pass",
            "technical",
            "branch_order",
            "arms",
            "gate",
            "geometry",
            "budgets",
            "compute",
        ),
        path,
    )
    if not isinstance(record["case_id"], str) or not record["case_id"]:
        raise AdaptiveGateAnalysisError(f"{path}.case_id: non-empty string required")
    if not isinstance(record["request_id"], str) or len(record["request_id"]) != 64:
        raise AdaptiveGateAnalysisError(f"{path}.request_id: SHA-256 required")
    if tuple(record["branch_order"]) != AGATE_BRANCH_ORDER:
        raise AdaptiveGateAnalysisError(f"{path}.branch_order differs from lock")

    technical = _mapping(record["technical"], f"{path}.technical")
    _exact_keys(technical, TECHNICAL_KEYS, f"{path}.technical")
    technical_values = {
        key: _boolean(technical[key], f"{path}.technical.{key}")
        for key in TECHNICAL_KEYS
    }
    arms_raw = _mapping(record["arms"], f"{path}.arms")
    _exact_keys(arms_raw, AGATE_BRANCH_ORDER, f"{path}.arms")
    arms = {
        arm: _normalize_arm(arms_raw[arm], f"{path}.arms.{arm}")
        for arm in AGATE_BRANCH_ORDER
    }

    gate = _mapping(record["gate"], f"{path}.gate")
    _exact_keys(
        gate,
        (
            "relative_margin",
            "choices",
            "normalized_predicted_advantages",
            "refreshed_predicted_scores",
            "fixed_predicted_scores",
        ),
        f"{path}.gate",
    )
    choices = tuple(gate["choices"])
    if len(choices) != QSTEP_K - 1 or any(
        choice not in {"refreshed", "fixed"} for choice in choices
    ):
        raise AdaptiveGateAnalysisError(f"{path}.gate.choices differs from lock")
    advantages = tuple(
        _finite(value, f"{path}.gate.advantages[{position}]")
        for position, value in enumerate(gate["normalized_predicted_advantages"])
    )
    refreshed_scores = tuple(
        _finite(value, f"{path}.gate.refreshed_scores[{position}]")
        for position, value in enumerate(gate["refreshed_predicted_scores"])
    )
    fixed_scores = tuple(
        _finite(value, f"{path}.gate.fixed_scores[{position}]")
        for position, value in enumerate(gate["fixed_predicted_scores"])
    )
    if not (
        len(advantages) == len(refreshed_scores) == len(fixed_scores) == QSTEP_K - 1
    ):
        raise AdaptiveGateAnalysisError(f"{path}.gate vectors differ from K=4")
    margin = _finite(gate["relative_margin"], f"{path}.gate.relative_margin")
    if margin != GATE_RELATIVE_MARGIN:
        raise AdaptiveGateAnalysisError(f"{path}.gate margin differs from lock")

    geometry = _mapping(record["geometry"], f"{path}.geometry")
    _exact_keys(
        geometry,
        ("gated_w0_to_refreshed_c_cosines",),
        f"{path}.geometry",
    )
    cosine_steps = tuple(geometry["gated_w0_to_refreshed_c_cosines"])
    if len(cosine_steps) != QSTEP_K:
        raise AdaptiveGateAnalysisError(f"{path}.geometry must contain four steps")
    cosine_values: list[float] = []
    for step, raw in enumerate(cosine_steps, start=1):
        values = _mapping(raw, f"{path}.geometry.step[{step}]")
        if len(values) != 5:
            raise AdaptiveGateAnalysisError(
                f"{path}.geometry.step[{step}] must contain five layers"
            )
        cosine_values.extend(
            _finite(value, f"{path}.geometry.step[{step}].{name}")
            for name, value in values.items()
        )

    budgets = _mapping(record["budgets"], f"{path}.budgets")
    _exact_keys(
        budgets,
        (
            "native_c_energy",
            "native_c_distance",
            "hop_c_distance",
            "per_hop_c_energy",
            "probe_c_distance",
        ),
        f"{path}.budgets",
    )
    budget_values = {
        key: _finite(budgets[key], f"{path}.budgets.{key}", nonnegative=True)
        for key in budgets
    }
    if not all(value > 0.0 for value in budget_values.values()):
        raise AdaptiveGateAnalysisError(f"{path}.budgets must be positive")

    compute = _mapping(record["compute"], f"{path}.compute")
    _exact_keys(
        compute,
        (
            "controlled_nfe",
            "proposal_build_count",
            "probe_panel_count",
            "wall_seconds",
        ),
        f"{path}.compute",
    )
    compute_values = {
        "controlled_nfe": _integer(
            compute["controlled_nfe"], f"{path}.compute.controlled_nfe", positive=True
        ),
        "proposal_build_count": _integer(
            compute["proposal_build_count"],
            f"{path}.compute.proposal_build_count",
            positive=True,
        ),
        "probe_panel_count": _integer(
            compute["probe_panel_count"],
            f"{path}.compute.probe_panel_count",
            positive=True,
        ),
        "wall_seconds": _finite(
            compute["wall_seconds"], f"{path}.compute.wall_seconds", nonnegative=True
        ),
    }
    if (
        compute_values["controlled_nfe"] != AGATE_CONTROLLED_NFE
        or compute_values["proposal_build_count"] != 8
        or compute_values["probe_panel_count"] != AGATE_PROBE_PANEL_COUNT
    ):
        raise AdaptiveGateAnalysisError(f"{path}.compute plan differs from lock")

    return {
        "model_alias": record["model_alias"],
        "editor_track": record["editor_track"],
        "case_id": record["case_id"],
        "request_id": record["request_id"],
        "pass": _boolean(record["pass"], f"{path}.pass"),
        "technical": technical_values,
        "arms": arms,
        "choices": choices,
        "advantages": advantages,
        "cosines": tuple(cosine_values),
        "budgets": budget_values,
        "compute": compute_values,
    }


def _normalize_feature(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    path = f"features[{index}]"
    required = {
        "case_id",
        "request_id",
        "editor_track",
        "k",
        "hop_fraction",
        "native_c_energy",
        "native_c_distance",
        "per_hop_c_energy",
        "hop_c_distance",
        "probe_c_distance",
        "gate_relative_margin",
        "target_identity",
        "w0_panel",
        "w0_layer_weights",
        "w0_transform",
        "paths",
        "compute_plan",
        "feature_hash",
    }
    _exact_keys(record, required, path)
    if (
        record["k"] != QSTEP_K
        or _finite(record["hop_fraction"], f"{path}.hop_fraction")
        != QSTEP_HOP_FRACTION
        or _finite(record["gate_relative_margin"], f"{path}.gate_relative_margin")
        != GATE_RELATIVE_MARGIN
    ):
        raise AdaptiveGateAnalysisError(f"{path}: controller constants differ from lock")
    native_distance = _finite(
        record["native_c_distance"], f"{path}.native_c_distance", nonnegative=True
    )
    probe_distance = _finite(
        record["probe_c_distance"], f"{path}.probe_c_distance", nonnegative=True
    )
    if not math.isclose(
        probe_distance,
        native_distance * QSTEP_PROBE_FRACTION,
        rel_tol=3e-5,
        abs_tol=3e-5,
    ):
        raise AdaptiveGateAnalysisError(f"{path}: probe distance differs from D/64")
    transform = _mapping(record["w0_transform"], f"{path}.w0_transform")
    retention: list[float] = []
    if record["editor_track"] == TRACK_MEMIT:
        if transform:
            raise AdaptiveGateAnalysisError(f"{path}: MEMIT feature has projector metadata")
    elif record["editor_track"] == TRACK_ALPHAEDIT:
        _exact_keys(transform, ("synchronous", "ordered"), f"{path}.w0_transform")
        for family in ("synchronous", "ordered"):
            metadata = _mapping(transform[family], f"{path}.w0_transform.{family}")
            _exact_keys(
                metadata,
                ("projector_manifest_id", "right_norm_retention"),
                f"{path}.w0_transform.{family}",
            )
            values = _mapping(
                metadata["right_norm_retention"],
                f"{path}.w0_transform.{family}.right_norm_retention",
            )
            if len(values) != 5:
                raise AdaptiveGateAnalysisError(
                    f"{path}.w0_transform.{family}: five retention values required"
                )
            retention.extend(
                _finite(value, f"{path}.retention.{family}.{name}", nonnegative=True)
                for name, value in values.items()
            )
    else:
        raise AdaptiveGateAnalysisError(f"{path}: unknown editor track")
    return {
        "case_id": record["case_id"],
        "request_id": record["request_id"],
        "editor_track": record["editor_track"],
        "retention": tuple(retention),
    }


def _bootstrap_ci(values: Sequence[float], stream: int) -> tuple[float, float]:
    rng = random.Random(AGATE_BOOTSTRAP_SEED + 10_007 * stream)
    count = len(values)
    means = sorted(
        math.fsum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(AGATE_BOOTSTRAP_RESAMPLES)
    )
    lower = means[int(0.025 * (len(means) - 1))]
    upper = means[int(0.975 * (len(means) - 1))]
    return lower, upper


def _summary(values: Sequence[float], stream: int) -> dict[str, Any]:
    if len(values) != AGATE_CASE_COUNT:
        raise AdaptiveGateAnalysisError("effect vector differs from eight-case ITD")
    lower, upper = _bootstrap_ci(values, stream)
    return {
        "mean": math.fsum(values) / len(values),
        "median": statistics.median(values),
        "bootstrap_95_ci": [lower, upper],
        "positive_count": sum(value > AGATE_PRACTICAL_FLOOR for value in values),
        "negative_count": sum(value < -AGATE_PRACTICAL_FLOOR for value in values),
        "within_floor_count": sum(
            abs(value) <= AGATE_PRACTICAL_FLOOR for value in values
        ),
    }


def analyze_case_records(
    records: Iterable[Mapping[str, Any]],
    features: Iterable[Mapping[str, Any]],
    *,
    model_alias: str,
    track: str,
) -> dict[str, Any]:
    if model_alias not in MODEL_ALIASES or track not in TRACKS:
        raise AdaptiveGateAnalysisError("model/track is outside the fixed pair")
    materialized = list(records)
    feature_rows = list(features)
    if len(materialized) != AGATE_CASE_COUNT or len(feature_rows) != AGATE_CASE_COUNT:
        raise AdaptiveGateAnalysisError("analysis/features require exact eight-case ITD")
    normalized = [
        _normalize_case(_mapping(record, f"records[{index}]"), index)
        for index, record in enumerate(materialized)
    ]
    normalized_features = [
        _normalize_feature(_mapping(record, f"features[{index}]"), index)
        for index, record in enumerate(feature_rows)
    ]
    identities = [(record["case_id"], record["request_id"]) for record in normalized]
    feature_identities = [
        (record["case_id"], record["request_id"]) for record in normalized_features
    ]
    if (
        len(set(identities)) != AGATE_CASE_COUNT
        or identities != feature_identities
        or {record["model_alias"] for record in normalized} != {model_alias}
        or {record["editor_track"] for record in normalized} != {track}
        or {record["editor_track"] for record in normalized_features} != {track}
    ):
        raise AdaptiveGateAnalysisError("case/model/track identity differs across streams")

    technical_failures: list[str] = []
    for record in normalized:
        if not record["pass"]:
            technical_failures.append(f"{record['case_id']}:pass=false")
        technical_failures.extend(
            f"{record['case_id']}:{key}=false"
            for key, value in record["technical"].items()
            if not value
        )
        technical_failures.extend(
            f"{record['case_id']}:{arm}=failed"
            for arm, value in record["arms"].items()
            if not value["success"]
        )

    effect_vectors: dict[str, list[float]] = {
        "gated_minus_refreshed_G4_minus_A4": [],
        "gated_minus_coefficient_G4_minus_B4": [],
        "gated_minus_fixed_G4_minus_C4": [],
        "native_gap_reduction_abs_A4_native_minus_abs_G4_native": [],
        "gated_minus_casewise_max_A4_B4": [],
        "direction_refresh_A4_minus_B4": [],
        "coefficient_refresh_B4_minus_C4": [],
        "native_split_gap_split4_minus_one_shot": [],
    }
    case_effects: list[dict[str, Any]] = []
    step_refresh_counts = [0, 0, 0]
    all_advantages: list[float] = []
    all_cosines: list[float] = []
    for record in normalized:
        arms = record["arms"]
        g = arms[GATED_FINAL]["progress"]
        a = arms[REFRESHED_FINAL]["progress"]
        b = arms[COEFFICIENT_FINAL]["progress"]
        c = arms[FIXED_FINAL]["progress"]
        native = arms[NATIVE_ORDERED_FULL]["progress"]
        split = arms[NATIVE_ORDERED_SPLIT4]["progress"]
        values = {
            "gated_minus_refreshed_G4_minus_A4": g - a,
            "gated_minus_coefficient_G4_minus_B4": g - b,
            "gated_minus_fixed_G4_minus_C4": g - c,
            "native_gap_reduction_abs_A4_native_minus_abs_G4_native": abs(a - native)
            - abs(g - native),
            "gated_minus_casewise_max_A4_B4": g - max(a, b),
            "direction_refresh_A4_minus_B4": a - b,
            "coefficient_refresh_B4_minus_C4": b - c,
            "native_split_gap_split4_minus_one_shot": split - native,
        }
        for key, value in values.items():
            effect_vectors[key].append(value)
        for step, choice in enumerate(record["choices"]):
            step_refresh_counts[step] += int(choice == "refreshed")
        all_advantages.extend(record["advantages"])
        all_cosines.extend(record["cosines"])
        case_effects.append({"case_id": record["case_id"], **values})

    effects = {
        key: _summary(values, stream=index)
        for index, (key, values) in enumerate(effect_vectors.items())
    }
    broad_benefit = any(
        effects[key]["mean"] > AGATE_PRACTICAL_FLOOR
        for key in (
            "gated_minus_refreshed_G4_minus_A4",
            "gated_minus_coefficient_G4_minus_B4",
            "native_gap_reduction_abs_A4_native_minus_abs_G4_native",
        )
    )
    noncollapse = (
        effects["gated_minus_casewise_max_A4_B4"]["mean"] >= -0.10
    )
    alpha_transfer = any(
        effects[key]["mean"] > AGATE_PRACTICAL_FLOOR
        for key in (
            "gated_minus_coefficient_G4_minus_B4",
            "gated_minus_fixed_G4_minus_C4",
        )
    )
    scientific_gate = (broad_benefit if track == TRACK_MEMIT else alpha_transfer) and noncollapse
    if technical_failures:
        verdict = "BLOCK_TECHNICAL_INVALID"
    elif scientific_gate:
        verdict = "LENIENT_MODEL_PASS"
    elif noncollapse:
        verdict = "LENIENT_MODEL_NO_BENEFIT"
    else:
        verdict = "LENIENT_MODEL_COLLAPSE"

    retention_values = [
        value for feature in normalized_features for value in feature["retention"]
    ]
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "scope": "single_model_single_track",
        "model_alias": model_alias,
        "editor_track": track,
        "fixed_contract": {
            "case_count": AGATE_CASE_COUNT,
            "k": QSTEP_K,
            "hop_fraction": QSTEP_HOP_FRACTION,
            "probe_fraction": QSTEP_PROBE_FRACTION,
            "gate_relative_margin": GATE_RELATIVE_MARGIN,
            "branch_order": list(AGATE_BRANCH_ORDER),
            "bootstrap_seed": AGATE_BOOTSTRAP_SEED,
            "bootstrap_resamples": AGATE_BOOTSTRAP_RESAMPLES,
            "practical_effect_floor": AGATE_PRACTICAL_FLOOR,
        },
        "technical_validity": {
            "pass": not technical_failures,
            "failures": technical_failures,
            "successful_case_count": AGATE_CASE_COUNT if not technical_failures else 0,
            "itd_denominator": AGATE_CASE_COUNT,
        },
        "effects": effects,
        "gate_behavior": {
            "refresh_count": sum(step_refresh_counts),
            "decision_count": AGATE_CASE_COUNT * (QSTEP_K - 1),
            "refresh_rate": sum(step_refresh_counts)
            / (AGATE_CASE_COUNT * (QSTEP_K - 1)),
            "step_refresh_counts": {
                str(step): step_refresh_counts[step - 2]
                for step in range(2, QSTEP_K + 1)
            },
            "mean_normalized_predicted_advantage": math.fsum(all_advantages)
            / len(all_advantages),
        },
        "geometry": {
            "mean_w0_to_refreshed_c_cosine": math.fsum(all_cosines)
            / len(all_cosines),
            "minimum_w0_to_refreshed_c_cosine": min(all_cosines),
        },
        "projector": {
            "right_norm_retention_count": len(retention_values),
            "mean_right_norm_retention": (
                None
                if not retention_values
                else math.fsum(retention_values) / len(retention_values)
            ),
            "minimum_right_norm_retention": (
                None if not retention_values else min(retention_values)
            ),
        },
        "lenient_gate": {
            "broad_benefit": broad_benefit,
            "noncollapse": noncollapse,
            "alpha_transfer": alpha_transfer,
            "model_gate_pass": bool(not technical_failures and scientific_gate),
        },
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
        "claim_boundary": (
            "small CounterFact teacher-forced Motivation signal only; no accuracy, "
            "locality, retention, downstream, lifelong, or method-superiority claim"
        ),
    }


def render_markdown(analysis: Mapping[str, Any]) -> str:
    effects = _mapping(analysis["effects"], "analysis.effects")
    lines = [
        f"# Adaptive-gate 분석 — {analysis['model_alias']} / {analysis['editor_track']}",
        "",
        f"- 판정: `{analysis['verdict']}`",
        f"- technical validity: `{analysis['technical_validity']['pass']}`",
        f"- lenient model gate: `{analysis['lenient_gate']['model_gate_pass']}`",
        f"- refresh rate: `{analysis['gate_behavior']['refresh_rate']:.6f}`",
        "",
        "| 효과 | mean | median | 95% bootstrap CI | + / - / floor |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = (
        ("G4-A4", "gated_minus_refreshed_G4_minus_A4"),
        ("G4-B4", "gated_minus_coefficient_G4_minus_B4"),
        ("G4-C4", "gated_minus_fixed_G4_minus_C4"),
        (
            "native-gap reduction",
            "native_gap_reduction_abs_A4_native_minus_abs_G4_native",
        ),
        ("G4-max(A4,B4)", "gated_minus_casewise_max_A4_B4"),
        ("A4-B4", "direction_refresh_A4_minus_B4"),
        ("B4-C4", "coefficient_refresh_B4_minus_C4"),
        ("split4-native", "native_split_gap_split4_minus_one_shot"),
    )
    for label, key in labels:
        item = effects[key]
        lower, upper = item["bootstrap_95_ci"]
        lines.append(
            f"| {label} | {item['mean']:.8f} | {item['median']:.8f} | "
            f"[{lower:.8f}, {upper:.8f}] | {item['positive_count']} / "
            f"{item['negative_count']} / {item['within_floor_count']} |"
        )
    lines.extend(
        [
            "",
            "## 해석 경계",
            "",
            "- G는 outcome-free adaptive gate, A는 매-hop direction+coefficient refresh, B는 coefficient-only refresh, C는 fixed path다.",
            "- CI는 불확실성 표시이며 이번 lenient Motivation gate의 자동 kill 기준이 아니다.",
            "- 이 결과는 small teacher-forced signal이며 accuracy/locality/retention/downstream/lifelong 우위를 뜻하지 않는다.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ode-edit-adaptive-gate-analysis", allow_abbrev=False
    )
    parser.add_argument("--analysis-cases", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--model", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--track", required=True, choices=TRACKS)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analysis = analyze_case_records(
        load_stream(args.analysis_cases, expected_event=ANALYSIS_EVENT),
        load_stream(args.features, expected_event=FEATURE_EVENT),
        model_alias=args.model,
        track=args.track,
    )
    json_path = Path(args.output_json).expanduser().resolve()
    markdown_path = Path(args.output_markdown).expanduser().resolve()
    if json_path.exists() or markdown_path.exists():
        raise AdaptiveGateAnalysisError("analysis outputs must be exclusive")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(analysis, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(analysis), encoding="utf-8")
    print(
        json.dumps(
            {
                "model": args.model,
                "track": args.track,
                "verdict": analysis["verdict"],
            },
            sort_keys=True,
        )
    )
    return 0 if analysis["technical_validity"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
