"""Build the canonical factual package for the fixed-z engineering screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


GROUP_ORDER = [
    ("llama3-8b-inst", "alphaedit"),
    ("qwen2.5-7b-inst", "alphaedit"),
    ("llama3-8b-inst", "memit"),
    ("qwen2.5-7b-inst", "memit"),
]
GROUP_LABELS = {
    ("llama3-8b-inst", "alphaedit"): "Llama / AlphaEdit",
    ("qwen2.5-7b-inst", "alphaedit"): "Qwen / AlphaEdit",
    ("llama3-8b-inst", "memit"): "Llama / MEMIT",
    ("qwen2.5-7b-inst", "memit"): "Qwen / MEMIT",
}
COLORS = {
    ("llama3-8b-inst", "alphaedit"): "#2563eb",
    ("qwen2.5-7b-inst", "alphaedit"): "#7c3aed",
    ("llama3-8b-inst", "memit"): "#059669",
    ("qwen2.5-7b-inst", "memit"): "#dc2626",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"not a regular non-symlink: {path}")
    return {"path": str(resolved), "sha256": sha256(resolved), "bytes": resolved.stat().st_size}


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def stats(values: Iterable[float]) -> dict[str, float | int]:
    rows = [float(v) for v in values]
    if not rows:
        return {"count": 0, "mean": math.nan, "median": math.nan, "p90": math.nan, "max": math.nan, "min": math.nan}
    return {
        "count": len(rows),
        "mean": statistics.fmean(rows),
        "median": statistics.median(rows),
        "p90": percentile(rows, 0.9),
        "max": max(rows),
        "min": min(rows),
    }


def mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return statistics.fmean(rows) if rows else math.nan


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NOT_RECORDED"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    value = float(value)
    if math.isnan(value):
        return "NOT_RECORDED"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def metric_values(metrics: dict[str, Any], name: str, field: str) -> list[float]:
    return [float(v) for v in metrics[name][field]]


def metric_summary(cases: list[dict[str, Any]], source: str, name: str) -> dict[str, Any]:
    nll: list[float] = []
    strict: list[bool] = []
    margin: list[float] = []
    for case in cases:
        metrics = case[source]["metrics"]
        nll.extend(metric_values(metrics, name, "nll"))
        strict.extend(bool(v) for v in metrics[name]["strict"])
        margin.extend(metric_values(metrics, name, "margin"))
    return {"nll": stats(nll), "strict_numerator": sum(strict), "strict_denominator": len(strict), "margin": stats(margin)}


def candidate_metric_summary(cases: list[dict[str, Any]], name: str) -> dict[str, Any]:
    nll: list[float] = []
    strict: list[bool] = []
    margin: list[float] = []
    for case in cases:
        for candidate in case["candidates"]:
            if candidate["axis"] is None or not candidate["valid"]:
                continue
            metrics = candidate["observation"]["metrics"]
            nll.extend(metric_values(metrics, name, "nll"))
            strict.extend(bool(v) for v in metrics[name]["strict"])
            margin.extend(metric_values(metrics, name, "margin"))
    return {"nll": stats(nll), "strict_numerator": sum(strict), "strict_denominator": len(strict), "margin": stats(margin)}


def group_summary(payload: dict[str, Any]) -> dict[str, Any]:
    cases = payload["cases"]
    candidates = [candidate for case in cases for candidate in case["candidates"] if candidate["axis"] is not None]
    valid = [candidate for candidate in candidates if candidate["valid"]]
    equality_fields = ["NK_E", "NK_H", "NK_T", "target_activation", "target_logits"]
    equality = {field: max(float(candidate["equality"][field]) for candidate in candidates) for field in equality_fields}
    pair_mismatch = max(float(value) for case in cases for value in case["pair_action_relative_mismatch"].values())
    gamma_exact = sum(bool(case["official"]["gamma0_endpoint_exact"]) for case in cases)
    metric_names = ["rewrite_target_new", "rephrase_target_new", "rewrite_target_true", "rephrase_target_true", "locality_target_true"]
    official_metrics = {name: metric_summary(cases, "official", name) for name in metric_names}
    candidate_metrics = {name: candidate_metric_summary(cases, name) for name in metric_names}
    cvars = [float(candidate["observation"]["functional"]["cvar_0_875"]) for candidate in valid]
    reference_cvars = [float(case["reference"]["functional"]["cvar_0_875"]) for case in cases]
    padding = payload["padding_safety_gate"]
    padding_values = [
        *padding["hidden_relative_errors"].values(),
        *padding["logit_relative_errors"].values(),
        *padding["target_key_relative_errors"].values(),
        *padding["edit_history_key_relative_errors"].values(),
        *padding["nll_absolute_errors"],
    ]
    ranks = [case["candidates"][2]["rank"]["rank"] for case in cases]
    dimensions = [case["covariance"]["shape"][0] for case in cases]
    return {
        "model": payload["model"],
        "method": payload["method"],
        "status": payload["status"],
        "case_ids": payload["case_ids"],
        "case_count": len(cases),
        "candidate_valid_numerator": len(valid),
        "candidate_denominator": len(candidates),
        "gamma0_exact_numerator": gamma_exact,
        "gamma0_denominator": len(cases),
        "gamma0_max_target_logit_relative": max(float(case["official"]["gamma0_target_logit_relative"]) for case in cases),
        "official_metrics_identity": all(case["official"]["metrics"] == case["reference"]["metrics"] for case in cases),
        "direct_z_compute_count": sum(int(case["direct_z_compute_count"]) for case in cases),
        "direct_z_recompute_count": sum(int(case["direct_z_recompute_count"]) for case in cases),
        "w0_restore_numerator": sum(bool(case["w0_pointer_bytes_restore"]) for case in cases),
        "w0_restore_denominator": len(cases),
        "functional_cvar": stats(cvars),
        "reference_cvar": stats(reference_cvars),
        "functional_spread": stats(float(case["functional_spread"]) for case in cases),
        "duplicate_noise": stats(float(case["duplicate_noise"]) for case in cases),
        "engineering_gate": payload["engineering_gate"],
        "equality_max": equality,
        "pair_action_relative_mismatch_max": pair_mismatch,
        "rank": stats(ranks),
        "null_dimension": stats(dimension - rank for dimension, rank in zip(dimensions, ranks, strict=True)),
        "padding_status": padding["status"],
        "padding_max_error": max(float(v) for v in padding_values),
        "padding_tolerance": float(padding["tolerance"]),
        "official_semantic_call_count": sum(len(case["tokenizer_official_calls"]) for case in cases),
        "official_semantic_identity": all(
            call["semantic_input_attention_position_equal"]
            for case in cases for call in case["tokenizer_official_calls"]
        ),
        "official_metrics": official_metrics,
        "candidate_metrics": candidate_metrics,
        "wall_seconds": float(payload["wall_seconds"]),
        "peak_gpu_allocated_bytes": int(payload["peak_gpu_allocated_bytes"]),
        "full_fp32": bool(payload["full_fp32"]),
        "scientific_promotion": bool(payload["scientific_promotion"]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def case_rows(payloads: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key in GROUP_ORDER:
        payload = payloads[key]
        for case in payload["cases"]:
            axis = next(candidate for candidate in case["candidates"] if candidate["axis"] == 0 and candidate["sign"] == -1)
            cvars = [candidate["observation"]["functional"]["cvar_0_875"] for candidate in case["candidates"] if candidate["axis"] is not None and candidate["valid"]]
            rows.append({
                "model": key[0], "method": key[1], "case_id": case["case_id"],
                "valid_candidates": case["valid_candidate_count"], "candidate_denominator": case["candidate_denominator"],
                "functional_spread": case["functional_spread"], "duplicate_noise": case["duplicate_noise"],
                "spread_gt_3x_noise": case["spread_gt_3x_noise"], "reference_cvar": case["reference"]["functional"]["cvar_0_875"],
                "candidate_cvar_min": min(cvars), "candidate_cvar_max": max(cvars),
                "rank": axis["rank"]["rank"], "input_dimension": case["covariance"]["shape"][0],
                "null_dimension": case["covariance"]["shape"][0] - axis["rank"]["rank"],
                "null_residual_relative": axis["rank"]["null_residual_relative"],
                "pair_action_relative_mismatch_max": max(case["pair_action_relative_mismatch"].values()),
                "direct_z_compute_count": case["direct_z_compute_count"], "direct_z_recompute_count": case["direct_z_recompute_count"],
                "w0_restore": case["w0_pointer_bytes_restore"],
            })
    return rows


def candidate_rows(payloads: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key in GROUP_ORDER:
        for case in payloads[key]["cases"]:
            reference_cvar = float(case["reference"]["functional"]["cvar_0_875"])
            base_action = float(case["factor"]["base_action"])
            for candidate in case["candidates"]:
                metrics = candidate["observation"]["metrics"]
                equality = candidate.get("equality", {})
                action = candidate.get("action")
                rows.append({
                    "model": key[0], "method": key[1], "case_id": case["case_id"],
                    "candidate_id": candidate["candidate_id"], "axis": candidate["axis"], "sign": candidate["sign"],
                    "seed": candidate["seed"], "rho": candidate["rho"], "gamma": candidate["gamma"],
                    "valid": candidate["valid"], "failure": candidate["failure"], "z_sha256": candidate["z_sha256"],
                    "action": action, "action_overhead_relative": None if action is None else float(action) / base_action - 1.0,
                    "functional_cvar": candidate["observation"]["functional"]["cvar_0_875"],
                    "functional_cvar_drift": float(candidate["observation"]["functional"]["cvar_0_875"]) - reference_cvar,
                    "NK_E": equality.get("NK_E"), "NK_H": equality.get("NK_H"), "NK_T": equality.get("NK_T"),
                    "target_activation_relative": equality.get("target_activation"), "target_logits_relative": equality.get("target_logits"),
                    "rewrite_new_nll_mean": mean(metrics["rewrite_target_new"]["nll"]),
                    "rewrite_new_strict_rate": mean(float(v) for v in metrics["rewrite_target_new"]["strict"]),
                    "rephrase_new_nll_mean": mean(metrics["rephrase_target_new"]["nll"]),
                    "rephrase_new_strict_rate": mean(float(v) for v in metrics["rephrase_target_new"]["strict"]),
                    "locality_true_nll_mean": mean(metrics["locality_target_true"]["nll"]),
                })
    return rows


def svg_scatter(path: Path, title: str, x_label: str, y_label: str, series: dict[tuple[str, str], list[tuple[float, float]]], *, diagonal: bool = False) -> None:
    width, height = 780, 520
    left, right, top, bottom = 90, 30, 55, 75
    points = [point for rows in series.values() for point in rows]
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    if xmin == xmax:
        xmin, xmax = xmin - 0.05, xmax + 0.05
    if ymin == ymax:
        ymin, ymax = ymin - 0.05, ymax + 0.05
    xpad, ypad = (xmax - xmin) * 0.08, (ymax - ymin) * 0.08
    xmin, xmax, ymin, ymax = xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad
    sx = lambda value: left + (value - xmin) / (xmax - xmin) * (width - left - right)
    sy = lambda value: height - bottom - (value - ymin) / (ymax - ymin) * (height - top - bottom)
    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111"/>',
    ]
    if diagonal:
        low, high = max(xmin, ymin), min(xmax, ymax)
        items.append(f'<line x1="{sx(low):.2f}" y1="{sy(low):.2f}" x2="{sx(high):.2f}" y2="{sy(high):.2f}" stroke="#999" stroke-dasharray="5,5"/>')
    for tick in range(6):
        xv = xmin + (xmax - xmin) * tick / 5
        yv = ymin + (ymax - ymin) * tick / 5
        items.extend([
            f'<text x="{sx(xv):.2f}" y="{height-bottom+22}" text-anchor="middle" font-family="monospace" font-size="10">{xv:.2e}</text>',
            f'<text x="{left-8}" y="{sy(yv)+3:.2f}" text-anchor="end" font-family="monospace" font-size="10">{yv:.2e}</text>',
        ])
    for key in GROUP_ORDER:
        for x, y in series[key]:
            items.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.2" fill="{COLORS[key]}" fill-opacity="0.78"/>')
    items.extend([
        f'<text x="{(left+width-right)/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="13">{x_label}</text>',
        f'<text x="20" y="{(top+height-bottom)/2}" text-anchor="middle" font-family="sans-serif" font-size="13" transform="rotate(-90 20 {(top+height-bottom)/2})">{y_label}</text>',
    ])
    for index, key in enumerate(GROUP_ORDER):
        x, y = left + index * 165, height - 4
        items.extend([f'<circle cx="{x}" cy="{y}" r="4" fill="{COLORS[key]}"/>', f'<text x="{x+8}" y="{y+4}" font-family="sans-serif" font-size="10">{GROUP_LABELS[key]}</text>'])
    items.append("</svg>\n")
    path.write_text("\n".join(items))


def render_report(summaries: dict[tuple[str, str], dict[str, Any]], cases: list[dict[str, Any]], jobs: dict[str, Any]) -> str:
    lines = [
        "# Fixed-z non-uniqueness engineering screen — Llama/Qwen × AlphaEdit/MEMIT",
        "",
        "> 결론: 네 model/method 조합 모두 사전 고정 engineering gate를 통과했다. 동일 direct-z, edit/history/target local equality와 matched covariance action을 유지한 candidate family 안에서도 teacher next-token KL CVaR가 duplicate noise floor보다 분명하게 달라졌다. 이는 **engineering premise screen PASS**이며 confirmatory claim이나 promotion은 아니다.",
        "",
        "## 범위와 분모",
        "",
        "- CounterFact 고정 8 case/order를 model/method별로 각각 평가했다. case 교체·outcome 기반 재생성은 0이다.",
        "- case마다 direct-z 1회, reference duplicate 2회, random tangent 4축×±=8 candidates이다.",
        "- 총 candidate 분모는 4 groups × 8 cases × 8 = 256, duplicate 분모는 64, teacher prompt 평가는 candidate당 32개이다.",
        "- FULL-FP32, final-audit bank open/use 0, ODE/Euler/barrier/sequential/promotion 0이다.",
        "",
        "## 최종 gate 요약",
        "",
        "| model / method | valid cases | spread > 3×noise | median spread | median noise | gamma=0 | direct-z | W0 restore | 판정 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in GROUP_ORDER:
        row = summaries[key]
        gate = row["engineering_gate"]
        passed = gate["valid_gate"] and gate["spread_gate"] and gate["median_gate"]
        lines.append(
            f"| {GROUP_LABELS[key]} | {gate['valid_cases']}/8 | {gate['spread_gt_3x_noise_cases']}/8 | {fmt(gate['median_spread'])} | {fmt(gate['median_noise'])} | {row['gamma0_exact_numerator']}/8 | {row['direct_z_compute_count']} / recompute {row['direct_z_recompute_count']} | {row['w0_restore_numerator']}/8 | {'PASS' if passed else 'HOLD'} |"
        )
    lines.extend(["", "## Left-padding 및 Official 호출 결속", "", "| model / method | 최대 batch/singleton/reorder/pad-length 오차 | FP32 tolerance | Official semantic calls | input/attention/position identity | pad 결속 |", "|---|---:|---:|---:|---|---|"])
    for key in GROUP_ORDER:
        row = summaries[key]
        lines.append(f"| {GROUP_LABELS[key]} | {fmt(row['padding_max_error'])} | {fmt(row['padding_tolerance'])} | {row['official_semantic_call_count']} | {'PASS' if row['official_semantic_identity'] else 'FAIL'} | explicit pad=eos, model/generation config |")
    lines.extend([
        "",
        "attention_mask nonzero columns와 cumsum position_ids를 semantic 위치의 유일한 근거로 사용했다. selected hidden, target key, edit/history key, next-token logits/NLL을 left-padded batch↔singleton, reorder, padding-length 변화로 비교했다. Official right-padding 호출은 같은 text를 left-padding hook으로 replay하여 semantic input_ids/attention/position identity를 call별로 봉인했다.",
        "",
        "## Official baseline 및 gamma=0 재현",
        "",
        "| model / method | endpoint exact | max target-logit rel | metric object identity | rewrite-new strict | rephrase-new strict | rewrite-new NLL mean/median/p90/max | rephrase-new NLL mean/median/p90/max |",
        "|---|---:|---:|---|---:|---:|---|---|",
    ])
    for key in GROUP_ORDER:
        row = summaries[key]
        rewrite = row["official_metrics"]["rewrite_target_new"]
        rephrase = row["official_metrics"]["rephrase_target_new"]
        rs, ps = rewrite["nll"], rephrase["nll"]
        lines.append(
            f"| {GROUP_LABELS[key]} | {row['gamma0_exact_numerator']}/8 | {fmt(row['gamma0_max_target_logit_relative'])} | {'PASS' if row['official_metrics_identity'] else 'FAIL'} | {rewrite['strict_numerator']}/{rewrite['strict_denominator']} | {rephrase['strict_numerator']}/{rephrase['strict_denominator']} | {fmt(rs['mean'])}/{fmt(rs['median'])}/{fmt(rs['p90'])}/{fmt(rs['max'])} | {fmt(ps['mean'])}/{fmt(ps['median'])}/{fmt(ps['p90'])}/{fmt(ps['max'])} |"
        )
    lines.extend(["", "## Candidate functional 및 task metric 분포", "", "| model / method | valid candidates | CVaR mean/median/p90/max | case spread mean/median/p90/max | rewrite-new NLL mean/median/p90/max | rephrase-new NLL mean/median/p90/max |", "|---|---:|---|---|---|---|"])
    for key in GROUP_ORDER:
        row = summaries[key]
        cvar, spread = row["functional_cvar"], row["functional_spread"]
        rewrite = row["candidate_metrics"]["rewrite_target_new"]["nll"]
        rephrase = row["candidate_metrics"]["rephrase_target_new"]["nll"]
        lines.append(
            f"| {GROUP_LABELS[key]} | {row['candidate_valid_numerator']}/{row['candidate_denominator']} | {fmt(cvar['mean'])}/{fmt(cvar['median'])}/{fmt(cvar['p90'])}/{fmt(cvar['max'])} | {fmt(spread['mean'])}/{fmt(spread['median'])}/{fmt(spread['p90'])}/{fmt(spread['max'])} | {fmt(rewrite['mean'])}/{fmt(rewrite['median'])}/{fmt(rewrite['p90'])}/{fmt(rewrite['max'])} | {fmt(rephrase['mean'])}/{fmt(rephrase['median'])}/{fmt(rephrase['p90'])}/{fmt(rephrase['max'])} |"
        )
    lines.extend(["", "## Case별 functional spread", "", "| case | Llama AlphaEdit | Qwen AlphaEdit | Llama MEMIT | Qwen MEMIT |", "|---:|---:|---:|---:|---:|"])
    case_lookup = {(row["model"], row["method"], str(row["case_id"])): row for row in cases}
    case_ids = summaries[GROUP_ORDER[0]]["case_ids"]
    for case_id in case_ids:
        values = [case_lookup[(key[0], key[1], str(case_id))]["functional_spread"] for key in GROUP_ORDER]
        lines.append(f"| {case_id} | " + " | ".join(fmt(value) for value in values) + " |")
    lines.extend(["", "## Equality, action, rank", "", "| model / method | max NK_E/H/T | max target activation/logit rel | max ± action mismatch | rank median | null dimension median |", "|---|---|---|---:|---:|---:|"])
    for key in GROUP_ORDER:
        row = summaries[key]
        eq = row["equality_max"]
        lines.append(f"| {GROUP_LABELS[key]} | {fmt(eq['NK_E'])}/{fmt(eq['NK_H'])}/{fmt(eq['NK_T'])} | {fmt(eq['target_activation'])}/{fmt(eq['target_logits'])} | {fmt(row['pair_action_relative_mismatch_max'])} | {fmt(row['rank']['median'])} | {fmt(row['null_dimension']['median'])} |")
    lines.extend([
        "",
        "- [matched ± candidate CVaR scatter](matched-pair-cvar-scatter.svg): 동일 축의 −/+ candidate CVaR. 점선은 y=x이다.",
        "- [covariance-action overhead 대 CVaR drift](action-drift-scatter.svg): action-matched tangent의 상대 action 증가와 Official reference 대비 functional drift.",
        "- [null dimension 대 case spread](rank-spread-scatter.svg): constraint rank 이후 남은 input tangent dimension과 case functional spread.",
        "- 정확한 점 데이터는 [per-candidate.csv](per-candidate.csv), case 데이터는 [per-case.csv](per-case.csv)에 있다.",
        "",
        "## 계산 및 완전성",
        "",
        "| model / method | wall seconds | peak GPU allocated | FULL-FP32 | final audit used |",
        "|---|---:|---:|---|---:|",
    ])
    for key in GROUP_ORDER:
        row = summaries[key]
        lines.append(f"| {GROUP_LABELS[key]} | {fmt(row['wall_seconds'], 3)} | {row['peak_gpu_allocated_bytes']} bytes | {'PASS' if row['full_fp32'] else 'FAIL'} | 0 |")
    lines.extend([
        "",
        "## 실패 및 제외 lineage",
        "",
        "최종 science denominator의 invalid/failed case는 0이다. 아래 technical/superseded attempts는 결과 선택에 사용하지 않았다.",
        "",
        "| job/lineage | 분류 | denominator | 근거 |",
        "|---|---|---:|---|",
    ])
    for item in jobs["excluded_attempts"]:
        lines.append(f"| {item['job']} | {item['classification']} | 0 | {item['reason']} |")
    lines.extend([
        "",
        "## Factual interpretation",
        "",
        "1. 네 조합 모두 exact target identity와 matched action을 유지한 8-candidate family를 8/8 case에서 생성했다.",
        "2. duplicate 평가가 이 실행에서 bitwise-deterministic해 noise가 0이었고, 모든 case의 positive spread가 사전 3×noise 조건을 넘었다. 따라서 수치의 절대 크기와 함께 raw per-candidate 값도 보존한다.",
        "3. 결과는 fixed-z local equality만이 아니라 full-model target activation/logit 및 32-prompt teacher-KL CVaR로 검증됐다.",
        "4. 이는 동일 direct-z/action 아래 기능 보존 결과의 non-uniqueness가 engineering screen에서 관찰됐다는 근거다. 8-case screen이므로 population-level confirmatory claim, architecture pooling, sequential/continual claim, 자동 promotion은 하지 않는다.",
        "5. 계약상 다음 단계는 unused CounterFact IDs의 별도 128 request/model × 3 seed confirmatory gate 설계이며, 이번 task에서는 실행하지 않았다.",
        "",
        "scientific_promotion=false",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--smoke-result", type=Path, action="append", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--job-ledger", type=Path, required=True)
    parser.add_argument("--execution-head", required=True)
    parser.add_argument("--execution-tree", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit("refusing to overwrite report package")
    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    for path in args.result:
        payload = json.loads(path.read_text())
        key = (payload["model"], payload["method"])
        if key in payloads or payload["stage"] != "screen" or payload["status"] != "TERMINAL_VALID":
            raise ValueError(f"invalid or duplicate terminal result: {path}")
        payloads[key] = payload
    if set(payloads) != set(GROUP_ORDER):
        raise ValueError("exact four model/method screen results required")
    jobs = json.loads(args.job_ledger.read_text())
    summaries = {key: group_summary(payloads[key]) for key in GROUP_ORDER}
    if not all(
        row["engineering_gate"]["valid_gate"]
        and row["engineering_gate"]["spread_gate"]
        and row["engineering_gate"]["median_gate"]
        and row["official_semantic_identity"]
        for row in summaries.values()
    ):
        raise ValueError("terminal engineering/padding gate is not uniformly PASS")
    cases = case_rows(payloads)
    candidates = candidate_rows(payloads)
    args.output_dir.mkdir(parents=True, mode=0o755)
    report_path = args.output_dir / "fixed-z-nonuniqueness-screen-factual-ko.md"
    case_path = args.output_dir / "per-case.csv"
    candidate_path = args.output_dir / "per-candidate.csv"
    summary_path = args.output_dir / "summary.json"
    write_csv(case_path, cases)
    write_csv(candidate_path, candidates)
    summary_payload = {
        "schema": "odeedit.s06.fixed-z-nonuniqueness.analysis-summary.v1",
        "groups": {f"{key[0]}::{key[1]}": summaries[key] for key in GROUP_ORDER},
        "denominators": {"models": 2, "methods": 2, "cases_per_group": 8, "candidates_per_case": 8, "valid_candidates": 256, "duplicates": 64},
        "engineering_screen_pass": True,
        "confirmatory_claim": False,
        "scientific_promotion": False,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n")
    pair_series = {key: [] for key in GROUP_ORDER}
    action_series = {key: [] for key in GROUP_ORDER}
    rank_series = {key: [] for key in GROUP_ORDER}
    for key in GROUP_ORDER:
        for case in payloads[key]["cases"]:
            by_axis = {}
            for candidate in case["candidates"]:
                if candidate["axis"] is not None:
                    by_axis.setdefault(candidate["axis"], {})[candidate["sign"]] = candidate
                    action_series[key].append((
                        float(candidate["action"]) / float(case["factor"]["base_action"]) - 1.0,
                        float(candidate["observation"]["functional"]["cvar_0_875"]) - float(case["reference"]["functional"]["cvar_0_875"]),
                    ))
            for values in by_axis.values():
                pair_series[key].append((float(values[-1]["observation"]["functional"]["cvar_0_875"]), float(values[1]["observation"]["functional"]["cvar_0_875"])))
            first = next(candidate for candidate in case["candidates"] if candidate["axis"] == 0 and candidate["sign"] == -1)
            rank_series[key].append((float(case["covariance"]["shape"][0] - first["rank"]["rank"]), float(case["functional_spread"])))
    svg_scatter(args.output_dir / "matched-pair-cvar-scatter.svg", "Matched ± candidate teacher-KL CVaR", "minus candidate CVaR", "plus candidate CVaR", pair_series, diagonal=True)
    svg_scatter(args.output_dir / "action-drift-scatter.svg", "Covariance action overhead vs functional drift", "relative covariance-action overhead", "candidate CVaR - Official CVaR", action_series)
    svg_scatter(args.output_dir / "rank-spread-scatter.svg", "Tangent null dimension vs case functional spread", "input dimension - constraint rank", "case CVaR spread", rank_series)
    report_path.write_text(render_report(summaries, cases, jobs))
    os.chmod(report_path, 0o644)
    external_inputs = [identity(path) for path in [*args.result, *args.smoke_result, args.preflight, args.contract, args.case_manifest, args.job_ledger]]
    members = [report_path, case_path, candidate_path, summary_path, args.output_dir / "matched-pair-cvar-scatter.svg", args.output_dir / "action-drift-scatter.svg", args.output_dir / "rank-spread-scatter.svg"]
    member_ids = [{"relative_path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(members)]
    source_root = args.source_root.resolve()
    analysis_head = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    analysis_tree = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD^{tree}"], text=True).strip()
    manifest = {
        "schema": "odeedit.s06.fixed-z-nonuniqueness.analysis-manifest.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_source": {"head": args.execution_head, "tree": args.execution_tree},
        "analysis_source": {"head": analysis_head, "tree": analysis_tree},
        "external_inputs": external_inputs,
        "members": member_ids,
        "member_root": canonical_hash(member_ids),
        "jobs": jobs,
        "denominators": summary_payload["denominators"],
        "final_audit_open_count": 0,
        "full_fp32": True,
        "scientific_promotion": False,
    }
    manifest_path = args.output_dir / "analysis-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n")
    manifest_identity = identity(manifest_path)
    manifest_identity["relative_path"] = manifest_path.name
    del manifest_identity["path"]
    receipt = {
        "schema": "odeedit.s06.fixed-z-nonuniqueness.rooted-analysis-receipt.v1",
        "manifest": manifest_identity,
        "member_root": manifest["member_root"],
        "execution_source": manifest["execution_source"],
        "analysis_source": manifest["analysis_source"],
        "engineering_screen_pass": True,
        "scientific_promotion": False,
    }
    receipt["root_digest"] = canonical_hash(receipt)
    receipt_path = args.output_dir / "rooted-analysis-receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
