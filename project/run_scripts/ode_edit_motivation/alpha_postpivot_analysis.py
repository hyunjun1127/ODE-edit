"""Post-MEMIT Alpha transfer synthesis locked before Alpha outcomes.

The original adaptive-gate analysis remains unchanged and all G/A/B/C arms
remain public.  This module only applies the post-MEMIT common-policy decision:
always-refresh A must beat transported direction B in both fixed models without
a large casewise non-collapse violation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adaptive_gate_analysis import ANALYSIS_SCHEMA
from .adaptive_gate_refresh import AGATE_CASE_COUNT, AGATE_PRACTICAL_FLOOR, TRACK_ALPHAEDIT


POSTPIVOT_SCHEMA = "ode-edit-alpha-postpivot-pair-analysis/v1"
MODEL_ORDER = ("llama3-8b-inst", "qwen2.5-7b-inst")
NONCOLLAPSE_FLOOR = -0.10
EXPECTED_RETENTION_COUNT = AGATE_CASE_COUNT * 2 * 5


class AlphaPostpivotAnalysisError(ValueError):
    """A compact Alpha analysis differs from the post-MEMIT lock."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AlphaPostpivotAnalysisError(f"{path}: mapping required")
    return value


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlphaPostpivotAnalysisError(f"{path}: finite scalar required")
    result = float(value)
    if not math.isfinite(result):
        raise AlphaPostpivotAnalysisError(f"{path}: finite scalar required")
    return result


def _effect_mean(analysis: Mapping[str, Any], key: str, path: str) -> float:
    effects = _mapping(analysis.get("effects"), f"{path}.effects")
    effect = _mapping(effects.get(key), f"{path}.effects.{key}")
    return _finite(effect.get("mean"), f"{path}.effects.{key}.mean")


def _normalize_model(analysis: Mapping[str, Any], model_alias: str) -> dict[str, Any]:
    path = model_alias
    if (
        analysis.get("schema_version") != ANALYSIS_SCHEMA
        or analysis.get("model_alias") != model_alias
        or analysis.get("editor_track") != TRACK_ALPHAEDIT
    ):
        raise AlphaPostpivotAnalysisError(f"{path}: analysis identity differs")
    technical = _mapping(analysis.get("technical_validity"), f"{path}.technical")
    if type(technical.get("pass")) is not bool:
        raise AlphaPostpivotAnalysisError(f"{path}: technical pass must be boolean")
    case_effects = analysis.get("case_effects")
    if not isinstance(case_effects, list) or len(case_effects) != AGATE_CASE_COUNT:
        raise AlphaPostpivotAnalysisError(f"{path}: exact eight case effects required")
    direction_values: list[float] = []
    case_ids: list[str] = []
    for index, raw_case in enumerate(case_effects):
        case = _mapping(raw_case, f"{path}.case_effects[{index}]")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise AlphaPostpivotAnalysisError(f"{path}: case identity is invalid")
        case_ids.append(case_id)
        direction_values.append(
            _finite(
                case.get("direction_refresh_A4_minus_B4"),
                f"{path}.case_effects[{index}].direction",
            )
        )
    if len(set(case_ids)) != AGATE_CASE_COUNT:
        raise AlphaPostpivotAnalysisError(f"{path}: case identities are not unique")
    direction_transfer = _effect_mean(
        analysis, "direction_refresh_A4_minus_B4", path
    )
    recomputed_direction = math.fsum(direction_values) / len(direction_values)
    if not math.isclose(
        direction_transfer, recomputed_direction, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise AlphaPostpivotAnalysisError(f"{path}: direction mean differs from cases")
    coefficient_support = _effect_mean(
        analysis, "coefficient_refresh_B4_minus_C4", path
    )
    a4_noncollapse = math.fsum(min(0.0, value) for value in direction_values) / len(
        direction_values
    )
    projector = _mapping(analysis.get("projector"), f"{path}.projector")
    retention_count = projector.get("right_norm_retention_count")
    if isinstance(retention_count, bool) or retention_count != EXPECTED_RETENTION_COUNT:
        raise AlphaPostpivotAnalysisError(f"{path}: projector retention count differs")
    mean_retention = _finite(
        projector.get("mean_right_norm_retention"), f"{path}.projector.mean"
    )
    minimum_retention = _finite(
        projector.get("minimum_right_norm_retention"), f"{path}.projector.minimum"
    )
    projector_nonzero = bool(mean_retention > 0.0 and minimum_retention >= 0.0)
    direction_pass = direction_transfer > AGATE_PRACTICAL_FLOOR
    noncollapse_pass = a4_noncollapse >= NONCOLLAPSE_FLOOR
    technical_pass = bool(technical["pass"])
    model_pass = technical_pass and projector_nonzero and direction_pass and noncollapse_pass
    return {
        "model_alias": model_alias,
        "fixed_contract": dict(_mapping(analysis.get("fixed_contract"), f"{path}.contract")),
        "case_ids": case_ids,
        "technical_pass": technical_pass,
        "direction_transfer_A4_minus_B4": direction_transfer,
        "coefficient_support_B4_minus_C4": coefficient_support,
        "a4_casewise_noncollapse": a4_noncollapse,
        "projector_retention_count": retention_count,
        "projector_mean_right_norm_retention": mean_retention,
        "projector_minimum_right_norm_retention": minimum_retention,
        "direction_pass": direction_pass,
        "noncollapse_pass": noncollapse_pass,
        "projector_nonzero": projector_nonzero,
        "model_pass": model_pass,
    }


def analyze_postpivot_pair(
    llama: Mapping[str, Any], qwen: Mapping[str, Any]
) -> dict[str, Any]:
    models = {
        MODEL_ORDER[0]: _normalize_model(llama, MODEL_ORDER[0]),
        MODEL_ORDER[1]: _normalize_model(qwen, MODEL_ORDER[1]),
    }
    same_method = (
        models[MODEL_ORDER[0]]["fixed_contract"]
        == models[MODEL_ORDER[1]]["fixed_contract"]
        and models[MODEL_ORDER[0]]["case_ids"] == models[MODEL_ORDER[1]]["case_ids"]
    )
    technical_pass = all(model["technical_pass"] for model in models.values())
    pair_pass = same_method and all(model["model_pass"] for model in models.values())
    if not technical_pass:
        verdict = "BLOCK_TECHNICAL_INVALID"
    elif not same_method:
        verdict = "BLOCK_METHOD_MISMATCH"
    elif pair_pass:
        verdict = "ALPHA_ALWAYS_REFRESH_TRANSFER_SIGNAL"
    else:
        verdict = "ALPHA_COMMON_TRANSFER_NO_SIGNAL"
    return {
        "schema_version": POSTPIVOT_SCHEMA,
        "editor_track": TRACK_ALPHAEDIT,
        "model_order": list(MODEL_ORDER),
        "same_method_gate": same_method,
        "technical_pass": technical_pass,
        "models": models,
        "thresholds": {
            "direction_transfer_strictly_greater_than": AGATE_PRACTICAL_FLOOR,
            "a4_casewise_noncollapse_at_least": NONCOLLAPSE_FLOOR,
            "projector_mean_retention_strictly_greater_than": 0.0,
        },
        "pair_gate_pass": pair_pass,
        "verdict": verdict,
        "historical_selector_status": "G killed by technical-valid MEMIT pair scientific result",
        "claim_boundary": (
            "atomic projected-proposal Motivation transfer only; no native AlphaEdit, "
            "locality, retention, lifelong, or method-superiority claim"
        ),
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Alpha post-MEMIT common-policy 분석",
        "",
        f"- 판정: `{result['verdict']}`",
        f"- same method/case: `{result['same_method_gate']}`",
        f"- technical pass: `{result['technical_pass']}`",
        "- historical selector G: MEMIT 결과로 kill; 전 arm 수치는 별도 원 분석에 공개",
        "",
        "| 모델 | A4-B4 | B4-C4 | A4 casewise noncollapse | projector mean/min | model pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    models = _mapping(result["models"], "models")
    for model_alias in MODEL_ORDER:
        item = _mapping(models[model_alias], model_alias)
        lines.append(
            f"| {model_alias} | {item['direction_transfer_A4_minus_B4']:.8f} | "
            f"{item['coefficient_support_B4_minus_C4']:.8f} | "
            f"{item['a4_casewise_noncollapse']:.8f} | "
            f"{item['projector_mean_right_norm_retention']:.8f} / "
            f"{item['projector_minimum_right_norm_retention']:.8f} | "
            f"{item['model_pass']} |"
        )
    lines.extend(
        [
            "",
            "## 해석 경계",
            "",
            "- 양 모델 중 하나의 실패를 다른 모델의 큰 효과로 상쇄하지 않는다.",
            "- 이는 projected actuator의 small teacher-forced transfer signal일 뿐이다.",
            "",
        ]
    )
    return "\n".join(lines)


def _read(path: str | Path) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    try:
        return _mapping(json.loads(source.read_text(encoding="utf-8")), str(source))
    except json.JSONDecodeError as exc:
        raise AlphaPostpivotAnalysisError(f"{source}: invalid JSON") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ode-edit-alpha-postpivot-analysis", allow_abbrev=False
    )
    parser.add_argument("--llama-analysis", required=True)
    parser.add_argument("--qwen-analysis", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = analyze_postpivot_pair(
        _read(args.llama_analysis), _read(args.qwen_analysis)
    )
    output_json = Path(args.output_json).expanduser().resolve()
    output_markdown = Path(args.output_markdown).expanduser().resolve()
    if output_json.exists() or output_markdown.exists():
        raise AlphaPostpivotAnalysisError("postpivot outputs must be exclusive")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"]}, sort_keys=True))
    return 0 if result["technical_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
