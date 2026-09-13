"""Endpoint-only factual evaluation; never imported by the controller."""

from __future__ import annotations

import math
import statistics
from typing import Any

from project.run_scripts.fixed_z_nonuniqueness.evaluation import sequence_metrics


def evaluate_rows(model: Any, tokenizer: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for row in rows:
        rewrite = row["requested_rewrite"]
        canonical = [rewrite["prompt"].format(rewrite["subject"])]
        rephrase = list(row.get("paraphrase_prompts", []))
        locality = list(row.get("neighborhood_prompts", []))
        records.append({
            "case_id": int(row["case_id"]),
            "rewrite_target_new": sequence_metrics(model, tokenizer, canonical, rewrite["target_new"]["str"]),
            "rewrite_target_true": sequence_metrics(model, tokenizer, canonical, rewrite["target_true"]["str"]),
            "rephrase_target_new": sequence_metrics(model, tokenizer, rephrase, rewrite["target_new"]["str"]),
            "rephrase_target_true": sequence_metrics(model, tokenizer, rephrase, rewrite["target_true"]["str"]),
            "locality_target_true": sequence_metrics(model, tokenizer, locality, rewrite["target_true"]["str"]),
        })
    return {
        "controller_influence_count": 0,
        "request_count": len(rows),
        "records": records,
        "aggregate": aggregate_records(records),
    }


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    answer: dict[str, Any] = {}
    for metric in (
        "rewrite_target_new", "rewrite_target_true", "rephrase_target_new",
        "rephrase_target_true", "locality_target_true",
    ):
        nll = [float(value) for row in records for value in row[metric]["nll"]]
        strict = [bool(value) for row in records for value in row[metric]["strict"]]
        margin = [float(value) for row in records for value in row[metric]["margin"]]
        answer[metric] = {
            "nll_count": len(nll),
            "nll_mean": statistics.fmean(nll) if nll else math.nan,
            "nll_median": statistics.median(nll) if nll else math.nan,
            "nll_p90": _quantile(nll, 0.9),
            "nll_max": max(nll) if nll else math.nan,
            "strict_numerator": sum(strict),
            "strict_denominator": len(strict),
            "strict_rate": sum(strict) / len(strict) if strict else math.nan,
            "margin_mean": statistics.fmean(margin) if margin else math.nan,
            "margin_median": statistics.median(margin) if margin else math.nan,
            "margin_p10": _quantile(margin, 0.1),
        }
    return answer

