"""Pair-level synthesis for two compact adaptive-gate model analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adaptive_gate_analysis import ANALYSIS_SCHEMA, TRACKS


PAIR_ANALYSIS_SCHEMA = "ode-edit-adaptive-gate-pair-analysis/v1"
LLAMA_ALIAS = "llama3-8b-inst"
QWEN_ALIAS = "qwen2.5-7b-inst"
MODEL_ORDER = (LLAMA_ALIAS, QWEN_ALIAS)


class AdaptiveGatePairAnalysisError(ValueError):
    """Two compact projections cannot be combined under one method gate."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdaptiveGatePairAnalysisError(f"{path}: mapping required")
    return value


def load_analysis(path: str | Path) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    try:
        return _mapping(json.loads(source.read_text(encoding="utf-8")), str(source))
    except json.JSONDecodeError as exc:
        raise AdaptiveGatePairAnalysisError(f"{source}: invalid JSON") from exc


def _effect_mean(analysis: Mapping[str, Any], key: str) -> float:
    effects = _mapping(analysis.get("effects"), "analysis.effects")
    item = _mapping(effects.get(key), f"analysis.effects.{key}")
    value = item.get("mean")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdaptiveGatePairAnalysisError(f"analysis.effects.{key}.mean invalid")
    return float(value)


def analyze_pair(
    llama: Mapping[str, Any],
    qwen: Mapping[str, Any],
    *,
    track: str,
) -> dict[str, Any]:
    if track not in TRACKS:
        raise AdaptiveGatePairAnalysisError("track is outside the fixed pair")
    by_model = {LLAMA_ALIAS: llama, QWEN_ALIAS: qwen}
    for model, analysis in by_model.items():
        if (
            analysis.get("schema_version") != ANALYSIS_SCHEMA
            or analysis.get("model_alias") != model
            or analysis.get("editor_track") != track
        ):
            raise AdaptiveGatePairAnalysisError(
                f"{model}: compact analysis identity differs from request"
            )
        technical = _mapping(
            analysis.get("technical_validity"), f"{model}.technical_validity"
        )
        gate = _mapping(analysis.get("lenient_gate"), f"{model}.lenient_gate")
        if type(technical.get("pass")) is not bool or type(gate.get("model_gate_pass")) is not bool:
            raise AdaptiveGatePairAnalysisError(f"{model}: gate booleans are invalid")

    llama_contract = _mapping(llama.get("fixed_contract"), "llama.fixed_contract")
    qwen_contract = _mapping(qwen.get("fixed_contract"), "qwen.fixed_contract")
    same_method_gate = dict(llama_contract) == dict(qwen_contract)
    technical_by_model = {
        model: bool(_mapping(value["technical_validity"], "technical")["pass"])
        for model, value in by_model.items()
    }
    model_gate_by_model = {
        model: bool(_mapping(value["lenient_gate"], "lenient_gate")["model_gate_pass"])
        for model, value in by_model.items()
    }
    pair_pass = bool(
        same_method_gate
        and all(technical_by_model.values())
        and all(model_gate_by_model.values())
    )
    if not all(technical_by_model.values()):
        verdict = "BLOCK_TECHNICAL_INVALID"
    elif not same_method_gate:
        verdict = "BLOCK_METHOD_MISMATCH"
    elif pair_pass:
        verdict = "LENIENT_PAIR_PASS"
    else:
        verdict = "LENIENT_PAIR_PARTIAL_OR_FAIL"

    effect_keys = (
        "gated_minus_refreshed_G4_minus_A4",
        "gated_minus_coefficient_G4_minus_B4",
        "gated_minus_fixed_G4_minus_C4",
        "native_gap_reduction_abs_A4_native_minus_abs_G4_native",
        "gated_minus_casewise_max_A4_B4",
    )
    model_means = {
        model: {key: _effect_mean(value, key) for key in effect_keys}
        for model, value in by_model.items()
    }
    return {
        "schema_version": PAIR_ANALYSIS_SCHEMA,
        "scope": "two_model_single_track_no_scale_pooling",
        "editor_track": track,
        "model_order": list(MODEL_ORDER),
        "same_method_gate": same_method_gate,
        "technical_by_model": technical_by_model,
        "model_gate_by_model": model_gate_by_model,
        "pair_gate_pass": pair_pass,
        "model_effect_means": model_means,
        "verdict": verdict,
        "combination_policy": (
            "no raw-utility pooling; each fixed model must independently pass the "
            "same precommitted controller/config and its lenient model gate"
        ),
        "claim_boundary": (
            "two-model small teacher-forced Motivation synthesis only; no accuracy, "
            "locality, retention, downstream, lifelong, or method-superiority claim"
        ),
    }


def render_markdown(analysis: Mapping[str, Any]) -> str:
    means = _mapping(analysis["model_effect_means"], "model_effect_means")
    lines = [
        f"# Adaptive-gate pair 분석 — {analysis['editor_track']}",
        "",
        f"- 판정: `{analysis['verdict']}`",
        f"- same method/config: `{analysis['same_method_gate']}`",
        f"- pair lenient gate: `{analysis['pair_gate_pass']}`",
        "- model utility scale은 pooling하지 않음",
        "",
        "| 모델 | G4-A4 | G4-B4 | G4-C4 | native-gap reduction | G4-max(A4,B4) | model gate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        item = _mapping(means[model], f"means.{model}")
        lines.append(
            f"| {model} | {item['gated_minus_refreshed_G4_minus_A4']:.8f} | "
            f"{item['gated_minus_coefficient_G4_minus_B4']:.8f} | "
            f"{item['gated_minus_fixed_G4_minus_C4']:.8f} | "
            f"{item['native_gap_reduction_abs_A4_native_minus_abs_G4_native']:.8f} | "
            f"{item['gated_minus_casewise_max_A4_B4']:.8f} | "
            f"{analysis['model_gate_by_model'][model]} |"
        )
    lines.extend(
        [
            "",
            "## 해석 경계",
            "",
            "- 한 모델의 큰 효과가 다른 모델의 실패를 상쇄하지 않는다.",
            "- 이 pair verdict는 small teacher-forced Motivation gate이며 paper-level 성능 claim이 아니다.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ode-edit-adaptive-gate-pair-analysis", allow_abbrev=False
    )
    parser.add_argument("--llama-analysis", required=True)
    parser.add_argument("--qwen-analysis", required=True)
    parser.add_argument("--track", required=True, choices=TRACKS)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = analyze_pair(
        load_analysis(args.llama_analysis),
        load_analysis(args.qwen_analysis),
        track=args.track,
    )
    json_path = Path(args.output_json).expanduser().resolve()
    markdown_path = Path(args.output_markdown).expanduser().resolve()
    if json_path.exists() or markdown_path.exists():
        raise AdaptiveGatePairAnalysisError("pair analysis outputs must be exclusive")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"track": args.track, "verdict": result["verdict"]}, sort_keys=True))
    return 0 if all(result["technical_by_model"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
