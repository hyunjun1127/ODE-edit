"""Read-only four-arm aggregation and canonical v4 report builder."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from project.run_scripts.ode_bf.contracts import canonical_hash

from .lifelong_finalw_contracts import (
    ABSENT_EXACT_CHECKPOINTS,
    AMENDED_CHECKPOINTS,
    EVALUATOR_IDENTITY,
    FinalWeightBoundary,
    INSTRUCTION_ID,
    ORDER_ROOT,
    ORIGINAL_REQUESTED_CHECKPOINTS,
    SCHEMA,
    STREAM_ROOT,
    STREAM_SEAL,
    STREAM_SEAL_SHA256,
    V3_MANIFEST_SHA256,
    V3_RECEIPT_SHA256,
    V3_REPORT_RELATIVE,
    V3_REPORT_SHA256,
)
from .lifelong_finalw_evaluation import CATEGORIES, age_stratum, sha256_file
from .lifelong_finalw_figures import generate as generate_figures


ARM_ORDER = ("LM", "LA", "QM", "QA")
ARM_KEY = {
    ("llama3-8b-inst", "memit"): "LM",
    ("llama3-8b-inst", "alphaedit"): "LA",
    ("qwen2.5-7b-inst", "memit"): "QM",
    ("qwen2.5-7b-inst", "alphaedit"): "QA",
}
ARM_LABEL = {
    "LM": "Llama-3-8B-Instruct / Official MEMIT",
    "LA": "Llama-3-8B-Instruct / Official AlphaEdit",
    "QM": "Qwen2.5-7B-Instruct / Official MEMIT",
    "QA": "Qwen2.5-7B-Instruct / Official AlphaEdit",
}
REPORT_NAME = "official-layer-debt-lifelong-finalw-full10k-factual-ko.md"
AGE_ORDER = ("EARLY_FIRST_20PCT", "MIDDLE_60PCT", "RECENT_LAST_20PCT")


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalWeightBoundary(f"JSON object expected: {path}")
    return value


def _verify_identity(value: Mapping[str, Any], label: str) -> None:
    expected = value.get("identity_sha256")
    payload = dict(value)
    payload.pop("identity_sha256", None)
    if expected != canonical_hash(payload):
        raise FinalWeightBoundary(f"canonical identity differs: {label}")


def _stream_request_hashes() -> tuple[str, ...]:
    if sha256_file(STREAM_SEAL) != STREAM_SEAL_SHA256:
        raise FinalWeightBoundary("analysis stream-seal SHA differs")
    seal = _load_json(STREAM_SEAL)
    root = seal.pop("root_digest", None)
    if root != STREAM_ROOT or canonical_hash(seal) != STREAM_ROOT:
        raise FinalWeightBoundary("analysis stream root differs")
    training = seal.get("training")
    if not isinstance(training, list) or len(training) != 10_000:
        raise FinalWeightBoundary("analysis stream request denominator differs")
    hashes = tuple(str(row["request_sha256"]) for row in training)
    if (
        [int(row["ordinal"]) for row in training] != list(range(10_000))
        or canonical_hash(list(hashes)) != ORDER_ROOT
        or seal.get("training_order_sha256") != ORDER_ROOT
        or seal.get("sample_duplication_count") != 0
    ):
        raise FinalWeightBoundary("analysis stream/order geometry differs")
    return hashes


def _raw_member(
    *, kind: str, arm: str, path: Path, digest: str, accepted_edit_count: int | str
) -> dict[str, Any]:
    return {
        "kind": kind,
        "arm": arm,
        "accepted_edit_count": accepted_edit_count,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest,
    }


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except ValueError:
        return False
    return True


def _verified_external(path: Path, expected_sha: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha:
        raise FinalWeightBoundary(f"{label} SHA differs")
    return {
        "label": label,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": expected_sha,
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise FinalWeightBoundary("empty distribution")
    if len(ordered) == 1:
        return ordered[0]
    location = (len(ordered) - 1) * probability
    low, high = math.floor(location), math.ceil(location)
    return ordered[low] if low == high else ordered[low] * (high - location) + ordered[high] * (location - low)


def _dist(values: Sequence[float]) -> dict[str, Any]:
    if not values or not all(math.isfinite(value) for value in values):
        raise FinalWeightBoundary("invalid analysis distribution")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": _quantile(values, 0.9),
        "max": max(values),
    }


def _read_records(path: Path, expected_sha: str, expected_rows: int) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha:
        raise FinalWeightBoundary(f"request-metric member SHA differs: {path}")
    records: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            payload = dict(value)
            identity = payload.pop("identity_sha256", None)
            derived_age = payload.pop("age_stratum", None)
            if identity != canonical_hash(payload):
                raise FinalWeightBoundary(
                    f"core evaluation row identity differs: {path}:{len(records) + 1}"
                )
            if derived_age != age_stratum(int(value["ordinal"]), expected_rows):
                raise FinalWeightBoundary(
                    f"derived edit-age stratum differs: {path}:{len(records) + 1}"
                )
            if value.get("raw_prompt_logit_generation_publish_count") != 0:
                raise FinalWeightBoundary("raw prompt/logit/generation publication differs")
            records.append(value)
    if len(records) != expected_rows:
        raise FinalWeightBoundary(f"request-metric row denominator differs: {path}")
    return records


def _recompute(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = {
        category: {"nll": [], "margin": []} for category in CATEGORIES
    }
    request_values: dict[str, dict[str, list[float]]] = {
        category: {"nll": [], "margin": []} for category in CATEGORIES
    }
    prompt_count = {category: 0 for category in CATEGORIES}
    strict_count = {category: 0 for category in CATEGORIES}
    gen_strict = rewrite_preferred = rephrase_preferred = 0
    age_groups: dict[str, list[Mapping[str, Any]]] = {key: [] for key in AGE_ORDER}
    for record in records:
        age_groups[str(record["age_stratum"])].append(record)
        metrics = record["metrics"]
        per_request: dict[str, float] = {}
        for category in CATEGORIES:
            prompts = metrics[category]["prompts"]
            nll = [float(item["nll"]) for item in prompts]
            margin = [float(item["margin"]) for item in prompts]
            values[category]["nll"].extend(nll)
            values[category]["margin"].extend(margin)
            request_values[category]["nll"].append(statistics.fmean(nll))
            request_values[category]["margin"].append(min(margin))
            per_request[category] = statistics.fmean(nll)
            prompt_count[category] += len(prompts)
            strict_count[category] += sum(int(item["strict"]) for item in prompts)
        gen_strict += int(
            metrics["rephrase_target_new"]["strict_count"]
            == metrics["rephrase_target_new"]["prompt_count"]
        )
        rewrite_preferred += int(per_request["rewrite_target_new"] < per_request["rewrite_target_true"])
        rephrase_preferred += int(per_request["rephrase_target_new"] < per_request["rephrase_target_true"])
    count = len(records)
    categories = {}
    for category in CATEGORIES:
        categories[category] = {
            "prompt_denominator": prompt_count[category],
            "strict_numerator": strict_count[category],
            "strict_rate": strict_count[category] / prompt_count[category],
            "prompt_nll": _dist(values[category]["nll"]),
            "request_cluster_nll": _dist(request_values[category]["nll"]),
            "prompt_margin": _dist(values[category]["margin"]),
            "request_cluster_min_margin": _dist(request_values[category]["margin"]),
        }
    age = {}
    for name, group in age_groups.items():
        if not group:
            raise FinalWeightBoundary(f"empty preregistered edit-age stratum: {name}")
        rewrite = [record["metrics"]["rewrite_target_new"] for record in group]
        rephrase = [record["metrics"]["rephrase_target_new"] for record in group]
        locality = [record["metrics"]["locality_target_true"] for record in group]
        req = len(group)
        rp = sum(int(value["prompt_count"]) for value in rephrase)
        lp = sum(int(value["prompt_count"]) for value in locality)
        age_row: dict[str, Any] = {
            "request_denominator": req,
            "eff_numerator": sum(int(value["strict_count"]) for value in rewrite),
            "eff": sum(int(value["strict_count"]) for value in rewrite) / req,
            "rephrase_prompt_denominator": rp,
            "gen_prompt_numerator": sum(int(value["strict_count"]) for value in rephrase),
            "gen_prompt": sum(int(value["strict_count"]) for value in rephrase) / rp,
            "gen_strict_numerator": sum(int(value["strict_count"] == value["prompt_count"]) for value in rephrase),
            "gen_strict": sum(int(value["strict_count"] == value["prompt_count"]) for value in rephrase) / req,
            "locality_prompt_denominator": lp,
            "loc_numerator": sum(int(value["strict_count"]) for value in locality),
            "loc": sum(int(value["strict_count"]) for value in locality) / lp,
        }
        for category in CATEGORIES:
            request_nll = [
                statistics.fmean(float(item["nll"]) for item in record["metrics"][category]["prompts"])
                for record in group
            ]
            request_margin = [
                min(float(item["margin"]) for item in record["metrics"][category]["prompts"])
                for record in group
            ]
            for statistic, value in _dist(request_nll).items():
                age_row[f"{category}_nll_{statistic}"] = value
            for statistic, value in _dist(request_margin).items():
                age_row[f"{category}_margin_{statistic}"] = value
        age[name] = age_row
    return {
        "request_denominator": count,
        "eff_numerator": strict_count["rewrite_target_new"],
        "eff": strict_count["rewrite_target_new"] / count,
        "gen_prompt_numerator": strict_count["rephrase_target_new"],
        "gen_prompt_denominator": prompt_count["rephrase_target_new"],
        "gen_prompt": strict_count["rephrase_target_new"] / prompt_count["rephrase_target_new"],
        "gen_strict_numerator": gen_strict,
        "gen_strict": gen_strict / count,
        "loc_numerator": strict_count["locality_target_true"],
        "loc_denominator": prompt_count["locality_target_true"],
        "loc": strict_count["locality_target_true"] / prompt_count["locality_target_true"],
        "rewrite_preferred_numerator": rewrite_preferred,
        "rewrite_preferred": rewrite_preferred / count,
        "rephrase_preferred_numerator": rephrase_preferred,
        "rephrase_preferred": rephrase_preferred / count,
        "categories": categories,
        "age": age,
    }


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def _validate_against_runtime(recomputed: Mapping[str, Any], runtime: Mapping[str, Any]) -> None:
    pairs = (
        (recomputed["request_denominator"], runtime["request_denominator"]),
        (recomputed["eff_numerator"], runtime["eff_strict_numerator"]),
        (recomputed["eff"], runtime["eff"]),
        (recomputed["gen_prompt_numerator"], runtime["gen_prompt_strict_numerator"]),
        (recomputed["gen_prompt_denominator"], runtime["rephrase_prompt_denominator"]),
        (recomputed["gen_prompt"], runtime["gen_prompt"]),
        (recomputed["gen_strict_numerator"], runtime["gen_strict_numerator"]),
        (recomputed["gen_strict"], runtime["gen_strict"]),
        (recomputed["loc_numerator"], runtime["loc_strict_numerator"]),
        (recomputed["loc_denominator"], runtime["locality_prompt_denominator"]),
        (recomputed["loc"], runtime["loc"]),
        (recomputed["rewrite_preferred_numerator"], runtime["rewrite_new_preferred_numerator"]),
        (recomputed["rewrite_preferred"], runtime["rewrite_new_preferred_rate"]),
        (recomputed["rephrase_preferred_numerator"], runtime["rephrase_new_preferred_numerator"]),
        (recomputed["rephrase_preferred"], runtime["rephrase_new_preferred_rate"]),
    )
    if not all(_close(left, right) for left, right in pairs):
        raise FinalWeightBoundary("independent aggregate/runtime summary differs")
    for category in CATEGORIES:
        expected = recomputed["categories"][category]
        actual = runtime["categories"][category]
        category_pairs: list[tuple[Any, Any]] = [
            (expected["prompt_denominator"], actual["prompt_denominator"]),
            (expected["strict_numerator"], actual["strict_numerator"]),
            (expected["strict_rate"], actual["strict_rate"]),
        ]
        for distribution in ("request_cluster_nll", "request_cluster_min_margin"):
            category_pairs.extend(
                (expected[distribution][statistic], actual[distribution][statistic])
                for statistic in ("n", "mean", "median", "p90", "max")
            )
        if not all(_close(left, right) for left, right in category_pairs):
            raise FinalWeightBoundary(f"independent category aggregate differs: {category}")
    for name in AGE_ORDER:
        expected = recomputed["age"][name]
        actual = runtime["age_strata"][name]
        age_pairs = (
            (expected["request_denominator"], actual["request_denominator"]),
            (expected["eff_numerator"], actual["rewrite_strict_numerator"]),
            (expected["eff"], actual["eff"]),
            (expected["gen_prompt_numerator"], actual["rephrase_strict_numerator"]),
            (expected["gen_prompt"], actual["gen_prompt"]),
            (expected["gen_strict_numerator"], actual["gen_strict_numerator"]),
            (expected["gen_strict"], actual["gen_strict"]),
            (expected["loc_numerator"], actual["locality_strict_numerator"]),
            (expected["loc"], actual["loc"]),
        )
        if not all(_close(left, right) for left, right in age_pairs):
            raise FinalWeightBoundary(f"independent edit-age aggregate differs: {name}")


def _summary_row(arm: str, model: str, method: str, count: int, summary: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "arm": arm,
        "model": model,
        "method": method,
        "accepted_edit_count": count,
        "evaluation_type": "CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS",
        "request_denominator": summary["request_denominator"],
        "eff_numerator": summary["eff_numerator"],
        "eff": summary["eff"],
        "gen_prompt_numerator": summary["gen_prompt_numerator"],
        "gen_prompt_denominator": summary["gen_prompt_denominator"],
        "gen_prompt": summary["gen_prompt"],
        "gen_strict_numerator": summary["gen_strict_numerator"],
        "gen_strict_denominator": summary["request_denominator"],
        "gen_strict": summary["gen_strict"],
        "loc_numerator": summary["loc_numerator"],
        "loc_denominator": summary["loc_denominator"],
        "loc": summary["loc"],
        "rewrite_new_preferred_numerator": summary["rewrite_preferred_numerator"],
        "rewrite_new_preferred_rate": summary["rewrite_preferred"],
        "rephrase_new_preferred_numerator": summary["rephrase_preferred_numerator"],
        "rephrase_new_preferred_rate": summary["rephrase_preferred"],
    }
    for category in CATEGORIES:
        metric = summary["categories"][category]
        for statistic in ("mean", "median", "p90", "max"):
            row[f"{category}_nll_{statistic}"] = metric["request_cluster_nll"][statistic]
            row[f"{category}_margin_{statistic}"] = metric["request_cluster_min_margin"][statistic]
        row[f"{category}_prompt_denominator"] = metric["prompt_denominator"]
        row[f"{category}_strict_numerator"] = metric["strict_numerator"]
        row[f"{category}_strict_rate"] = metric["strict_rate"]
    return row


def _flatten_request(arm: str, model: str, method: str, count: int, record: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "arm": arm,
        "model": model,
        "method": method,
        "accepted_edit_count": count,
        "ordinal": record["ordinal"],
        "batch_index": record["batch_index"],
        "age_stratum": record["age_stratum"],
        "case_identity_sha256": record["case_identity_sha256"],
        "request_sha256": record["request_sha256"],
    }
    for category in CATEGORIES:
        metric = record["metrics"][category]
        row[f"{category}_nll"] = statistics.fmean(float(value["nll"]) for value in metric["prompts"])
        row[f"{category}_margin"] = min(float(value["margin"]) for value in metric["prompts"])
        row[f"{category}_strict_numerator"] = metric["strict_count"]
        row[f"{category}_prompt_denominator"] = metric["prompt_count"]
    row["gen_request_strict"] = int(
        record["metrics"]["rephrase_target_new"]["strict_count"]
        == record["metrics"]["rephrase_target_new"]["prompt_count"]
    )
    row["rewrite_new_preferred"] = int(
        row["rewrite_target_new_nll"] < row["rewrite_target_true_nll"]
    )
    row["rephrase_new_preferred"] = int(
        row["rephrase_target_new_nll"] < row["rephrase_target_true_nll"]
    )
    return row


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, gzip_output: bool = False) -> dict[str, Any]:
    if not rows:
        raise FinalWeightBoundary(f"refusing empty table: {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise FinalWeightBoundary(f"table schema is not stable: {path.name}")
    if gzip_output:
        with path.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)
    else:
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "rows": len(rows)}


def _copy_create_once(source: Path, destination: Path, expected_sha: str) -> dict[str, Any]:
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise FinalWeightBoundary(f"copy source SHA differs: {source}")
    with destination.open("xb") as handle:
        handle.write(raw)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def _write_json_create_once(path: Path, value: Any) -> dict[str, Any]:
    raw = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
    ) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(raw)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    answer = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    answer.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(answer)


def _fmt(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _build_report(
    *, output: Path, source: Mapping[str, Any], final_rows: Sequence[Mapping[str, Any]],
    cumulative_rows: Sequence[Mapping[str, Any]], age_rows: Sequence[Mapping[str, Any]],
    compute_rows: Sequence[Mapping[str, Any]], amendment_sha: str, result_root: Path,
    v3_root: Path,
) -> str:
    final_by_arm = {str(row["arm"]): row for row in final_rows}
    headline = []
    for arm in ARM_ORDER:
        row = final_by_arm[arm]
        headline.append(
            (
                ARM_LABEL[arm],
                f"{row['eff_numerator']}/{row['request_denominator']} ({_fmt(row['eff'], 3)})",
                f"{row['gen_prompt_numerator']}/{row['gen_prompt_denominator']} ({_fmt(row['gen_prompt'], 3)})",
                f"{row['gen_strict_numerator']}/{row['gen_strict_denominator']} ({_fmt(row['gen_strict'], 3)})",
                f"{row['loc_numerator']}/{row['loc_denominator']} ({_fmt(row['loc'], 3)})",
                "/".join(_fmt(row[f"rewrite_target_new_nll_{s}"]) for s in ("mean", "median", "p90", "max")),
                "/".join(_fmt(row[f"rephrase_target_new_nll_{s}"]) for s in ("mean", "median", "p90", "max")),
            )
        )
    category_label = {
        "rewrite_target_new": "Rewrite / target-new",
        "rewrite_target_true": "Rewrite / target-true",
        "rephrase_target_new": "Rephrase / target-new",
        "rephrase_target_true": "Rephrase / target-true",
        "locality_target_true": "Locality / target-true",
    }
    final_detail = []
    for arm in ARM_ORDER:
        row = final_by_arm[arm]
        for category in CATEGORIES:
            final_detail.append(
                (
                    arm,
                    category_label[category],
                    f"{row[f'{category}_strict_numerator']}/{row[f'{category}_prompt_denominator']} ({_fmt(row[f'{category}_strict_rate'], 3)})",
                    "/".join(_fmt(row[f"{category}_nll_{value}"]) for value in ("mean", "median", "p90", "max")),
                    "/".join(_fmt(row[f"{category}_margin_{value}"]) for value in ("mean", "median", "p90", "max")),
                )
            )
    cumulative = []
    for row in cumulative_rows:
        cumulative.append(
            (
                str(row["arm"]), str(row["accepted_edit_count"]),
                f"{row['eff_numerator']}/{row['request_denominator']} ({_fmt(row['eff'], 3)})",
                f"{row['gen_prompt_numerator']}/{row['gen_prompt_denominator']} ({_fmt(row['gen_prompt'], 3)})",
                f"{row['gen_strict_numerator']}/{row['gen_strict_denominator']} ({_fmt(row['gen_strict'], 3)})",
                f"{row['loc_numerator']}/{row['loc_denominator']} ({_fmt(row['loc'], 3)})",
                _fmt(row["rewrite_target_new_nll_median"]),
                _fmt(row["rephrase_target_new_nll_median"]),
            )
        )
    cumulative_by_arm: dict[str, list[Mapping[str, Any]]] = {
        arm: sorted(
            (row for row in cumulative_rows if row["arm"] == arm),
            key=lambda row: int(row["accepted_edit_count"]),
        )
        for arm in ARM_ORDER
    }
    drift = []
    for arm in ARM_ORDER:
        first, final = cumulative_by_arm[arm][0], cumulative_by_arm[arm][-1]
        drift.append(
            (
                arm,
                _fmt(float(final["eff"]) - float(first["eff"]), 4),
                _fmt(float(final["gen_prompt"]) - float(first["gen_prompt"]), 4),
                _fmt(float(final["gen_strict"]) - float(first["gen_strict"]), 4),
                _fmt(float(final["loc"]) - float(first["loc"]), 4),
                _fmt(float(final["rewrite_target_new_nll_median"]) - float(first["rewrite_target_new_nll_median"]), 4),
                _fmt(float(final["rephrase_target_new_nll_median"]) - float(first["rephrase_target_new_nll_median"]), 4),
            )
        )
    terminal_age = [row for row in age_rows if int(row["accepted_edit_count"]) == 10_000]
    age_table = [
        (
            str(row["arm"]), str(row["age_stratum"]), str(row["request_denominator"]),
            _fmt(row["eff"]), _fmt(row["gen_prompt"]), _fmt(row["gen_strict"]), _fmt(row["loc"]),
        )
        for row in terminal_age
    ]
    v3_arms = {ARM_KEY[(row["model"], row["method"])]: row for row in _read_csv(v3_root / "arm-summary.csv")}
    residual = [
        (
            arm,
            _fmt(v3_arms[arm]["terminal_sentinel_q_pre_L8_median"]),
            _fmt(v3_arms[arm]["terminal_sentinel_q_post_L8_median"]),
            _fmt(v3_arms[arm]["terminal_Vbar"]),
            _fmt(v3_arms[arm]["terminal_unreachable_fraction"]),
        )
        for arm in ARM_ORDER
    ]
    compute_table = [
        (
            str(row["arm"]), _fmt(row["wall_hours"], 3), str(row["request_state_denominator"]),
            str(row["model_forward_batch_count"]), str(row["token_example_count"]),
            f"{float(row['peak_gpu_memory_bytes']) / 2**30:.2f}",
        )
        for row in compute_rows
    ]
    lines = [
        "# Official lifelong layer-debt — final W full-10k 및 cumulative seen-prefix v4 사실 보고서",
        "",
        "상태: **FOUR_ARM_FINAL_W_FULL10K_EVALUATION_TERMINAL_VALID**  ",
        "범위: Llama/Qwen × Official MEMIT/AlphaEdit 네 arm, evaluation-only backfill. `scientific_promotion=false`.",
        "",
        "## 1. Final W₁₀₀₀₀에서 전체 10,000 request 재평가 — 대표 결과",
        "",
        "아래가 이 보고서의 유일한 headline 성능이다. 각 arm의 10,000번째 edit 이후 frozen W 하나에서 sealed 10,000 request 전체를 다시 평가했다. 현재 B100, online-at-write, 소형 retention panel 값이 아니다. NLL 열은 request-cluster `mean/median/p90/max`다.",
        "",
        _markdown_table(
            ("모델/방법", "Eff strict", "Gen prompt", "Gen strict", "Loc", "Rewrite new NLL", "Rephrase new NLL"),
            headline,
        ),
        "",
        "![Final full-10k performance](finalw-full10k-performance.png)",
        "",
        "![Final full-10k target NLL](finalw-full10k-target-nll.png)",
        "",
        "### Final W₁₀₀₀₀ 상세 NLL·margin·strict",
        "",
        "NLL과 margin은 request-cluster `mean/median/p90/max` 순서다. Strict 열의 분모는 rewrite 10,000, rephrase 20,000 prompts, locality 100,000 prompts다.",
        "",
        _markdown_table(("arm", "평가축", "strict", "NLL mean/median/p90/max", "margin mean/median/p90/max"), final_detail),
        "",
        "## 2. Metric glossary / 표 읽는 법",
        "",
        "- **Eff (higher is better):** rewrite prompt에서 target-new의 모든 teacher-forced token이 top-1이면 request 성공 1건이다. numerator는 성공 request, denominator는 seen request다.",
        "- **Gen prompt (higher is better):** 각 paraphrase prompt에서 target-new 전체 token이 top-1인 prompt 비율이다. 이 stream은 request당 2개이므로 final denominator는 20,000/arm이다.",
        "- **Gen strict (higher is better):** 한 request의 paraphrase 2개가 모두 strict일 때만 request 성공이다. denominator는 request 수다. Gen prompt와 집계 단위를 섞지 않는다.",
        "- **Loc (higher is better):** neighborhood/locality prompt에서 원래 target-true의 모든 token이 top-1인 prompt 비율이다. request당 10개, final denominator 100,000/arm이다.",
        "- **target-new/target-true NLL (lower is better):** 각각 새 사실/원래 사실 continuation token의 평균 negative log likelihood다. 표의 mean/median/p90/max는 먼저 prompt 내 token 평균, 그 뒤 request 내 prompt 평균을 낸 request-cluster 분포다.",
        "- **margin (higher is better):** 각 continuation token에서 target logit−최고 비-target logit의 prompt 내 최솟값이다. 양수이면 그 prompt의 모든 target token이 top-1이다.",
        "- **preference (higher is better for new):** 동일 prompt에서 target-new NLL < target-true NLL인 request 비율이다. 이는 strict와 다른 비교 지표다.",
        "- **token accuracy:** 기존 canonical evaluator schema가 strict/NLL/margin만 반환하므로 `NOT_RECORDED_EVALUATOR_SCHEMA`; strict를 token accuracy로 대체하지 않았다.",
        "- **mean/median/p90/max:** 산술평균/중앙값/90백분위/최댓값이다. IQR은 p75−p25이며 기존 layer table에서만 사용한다. 누락값 보간·imputation은 0이다.",
        "- **R:** layer write 직전 `z*−activation` residual vector. **A:** Official uniform remaining-layer allocation `R / 남은 layer 수`. **Y:** 실제 write가 줄인 activation residual. **E=A−Y:** allocation-realization gap.",
        "- **q=||R||/||R_entry|| (lower is better for closure):** entry residual로 정규화한 남은 residual. 서로 다른 모델의 절대 norm 대신 무차원 q를 비교한다.",
        "- **rho=<Y,A>/||A||²:** target 방향 realization ratio; 1 exact, 0~1 under-realization, >1 overshoot, <0 opposite progress. **tau=||Y−rho A||/||A||:** 직교 왜곡(낮을수록 작음).",
        "- **d_parallel/d_perp:** L8 진입 residual이 ideal `(1/5)R_entry`에서 벗어난 inherited debt의 entry-target 평행/직교 성분이다. 절대 성능 점수가 아니다.",
        "- **recurrence closure:** `R5=(1/5)R1+(1/4)E1+(1/3)E2+(1/2)E3+E4`의 FP64 상대오차이며 계측 무결성 지표다.",
        "- **D_TV:** layer-wise update-share 분포와 양의 target-progress 분포 사이 total variation 거리(0 동일, 1 최대 분리). Negative progress는 별도 count로 남긴다.",
        "- **Layer-wise Update Magnitude:** 실제 `||ΔW_l||_F`; share는 5개 layer magnitude 합에서의 비중이다. activation progress와 별개 축이며 `weight energy`로 부르지 않는다.",
        "",
        "## 3. Checkpoint final W에서 all-seen-prefix 누적 성능",
        "",
        "평가 타입은 전부 `CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS`다. t=10,000은 §1 결과를 재사용했으며 중복 GPU 평가 0이다.",
        "",
        _markdown_table(("arm", "edits", "Eff", "Gen prompt", "Gen strict", "Loc", "Rewrite new NLL med", "Rephrase new NLL med"), cumulative),
        "",
        "### 1,000→10,000 edits 누적 변화량",
        "",
        "성공률 열은 final−1k(양수 개선), NLL 열도 final−1k이므로 음수일수록 개선이다.",
        "",
        _markdown_table(("arm", "ΔEff", "ΔGen prompt", "ΔGen strict", "ΔLoc", "ΔRewrite new NLL med", "ΔRephrase new NLL med"), drift),
        "",
        "![Cumulative seen-prefix performance](cumulative-seen-prefix-performance.png)",
        "",
        "원래 요청 schedule은 `{100,500,1000,2000,4000,6000,8000,10000}`이었으나 exact frozen state는 `{1000,1500,2000,3000,5000,7500,10000}`에만 존재했다. 이는 outcome을 열기 전 state availability에 따른 amendment다. absent 시점의 replay/reconstruction/interpolation/nearest substitution은 모두 0이며 amendment receipt SHA는 `" + amendment_sha + "`다.",
        "",
        "## 4. Final W₁₀₀₀₀ edit-age retention",
        "",
        "Age bin은 결과 전에 ordinal의 first 20% / middle 60% / last 20%로 고정했다. 각 행의 request denominator와 prompt denominator는 `edit-age-strata-summary.csv`에 있다.",
        "",
        _markdown_table(("arm", "age stratum", "request n", "Eff", "Gen prompt", "Gen strict", "Loc"), age_table),
        "",
        "각 checkpoint×age×target category의 request-cluster NLL와 margin mean/median/p90/max는 `edit-age-strata-summary.csv` 84행에 완전 수록했다.",
        "",
        "![Final edit-age retention](finalw-full10k-edit-age-retention.png)",
        "",
        "Fixed sentinel은 동일 probe가 시간에 따라 어떻게 변하는지, online-at-write는 편집 직후 성능, current B100은 당시 최근 100개 성능, cumulative seen-prefix는 frozen W_t가 지금까지 본 모든 request를 얼마나 유지하는지를 답한다. 네 타입은 서로 대체되지 않는다.",
        "",
        "## 5. 기존 layer residual/update/action-realization 분석",
        "",
        "v3의 layer telemetry bytes는 immutable하게 재사용한다. 아래 q/V̄는 final-W full10k 성능이 아니라 10k checkpoint의 fixed sentinel mechanism probe다.",
        "",
        _markdown_table(("arm", "sentinel pre-L8 q med", "sentinel post-L8 q med", "V_to_go/A0", "unreachable fraction"), residual),
        "",
        "- Llama 두 방법은 terminal sentinel에서 L8 update share가 가장 컸지만 Qwen은 L4가 가장 컸으므로 universal last-layer concentration claim은 유지하지 않는다.",
        "- D_TV drift 방향은 모델별로 달랐고, observation만으로 barrier benefit·ODE necessity·causal effect를 주장하지 않는다.",
        "- 전체 request/layer rho, tau, debt, recurrence, update magnitude/share, completion geometry는 immutable v3의 `layer-q-summary.csv`, `rho-tau-summary.csv`, `inherited-debt-summary.csv`, `recurrence-summary.csv`, `update-magnitude-share.csv`에 있다.",
        "",
        "기존 재현 그림: [residual trajectory](../official-layer-realization-debt-lifelong-b100x100-2026-09-03-v3/primary-residual-trajectory-2x2.png), [rho](../official-layer-realization-debt-lifelong-b100x100-2026-09-03-v3/rho-by-layer-terminal.png), [tau](../official-layer-realization-debt-lifelong-b100x100-2026-09-03-v3/tau-by-layer-terminal.png), [inherited debt](../official-layer-realization-debt-lifelong-b100x100-2026-09-03-v3/inherited-debt-distributions.png), [Layer-wise Update Magnitude](../official-layer-realization-debt-lifelong-b100x100-2026-09-03-v3/layer-wise-update-magnitude.png).",
        "",
        "## 6. Evaluation-only compute와 불변식",
        "",
        _markdown_table(("arm", "wall h", "request-state n", "forward batches", "token examples", "peak GPU GiB"), compute_table),
        "",
        "각 frozen state에서 평가 전후 edited weight pointer/version/byte SHA는 exact했다. `compute_z/writer/key/solve/cache append/cache consume/history append/optimizer/backward/parameter grad/model update during evaluation`은 전부 0이다. Checkpoint load만 7회/arm으로 별도 계수했다. Model parameter는 FULL-FP32, autocast/quantization=false, nonfinite=0, duplicate/imputation=0이다. Alpha cache/MEMIT covariance는 checkpoint container identity에 봉인했지만 forward-only runtime으로 load하지 않았다.",
        "",
        "## 7. Current-B100 diagnostic appendix — headline 아님",
        "",
        "v3의 첫 표와 `endpoint-checkpoint-strict-rates.png`는 checkpoint 당시 current B100만 평가한다. 이 값은 online/local diagnostic으로만 보존하며 final 10k 또는 cumulative retention 결론에 사용하지 않는다. v3 report SHA `" + V3_REPORT_SHA256 + "`는 변경하지 않았다.",
        "",
        "## 8. Provenance, limitations, artifact inventory",
        "",
        f"- Evaluation runtime source HEAD/tree: `{source['runtime_head']}` / `{source['runtime_tree']}`; analysis source HEAD/tree: `{source['analysis_head']}` / `{source['analysis_tree']}`.",
        f"- Evaluation result root: `{result_root}`.",
        f"- Common stream/order/evaluator: `{STREAM_ROOT}` / `{ORDER_ROOT}` / `{EVALUATOR_IDENTITY}`.",
        "- Four-arm checkpoint denominator 28/28; request-state denominator 120,000; final request denominator 40,000; final rephrase prompt denominator 80,000; final locality prompt denominator 400,000.",
        "- Original unavailable checkpoints `{100,500,4000,6000,8000}` remain `NOT_AVAILABLE_EXACT_STATE`; no reconstruction or estimate.",
        "- Raw prompts/logits/generations publish count 0. Per-request output contains request/case hashes and scalar metrics only.",
        "- Per-request `identity_sha256`는 evaluator가 만든 core row(metrics+ordinal+hashes)에 결속되고, outcome-blind 파생 `age_stratum`은 그 뒤 추가된다. 분석기는 core identity와 age(ordinal, seen-count)를 각각 독립 검증했다.",
        "- Technical exclusions are denominator0: two preflight adapter roots and job33300 canonical-batching parity lineage. Their evidence is preserved separately.",
        "- New figures are deterministic headless Python outputs. No Codex visualization/imagegen/manual image editing was used; missing values were not interpolated.",
        "- Full tables and every row/member SHA are in `analysis-manifest.json`; package root is in `rooted-analysis-receipt.json`.",
        "",
        "## 9. Factual conclusion",
        "",
        "Final W₁₀₀₀₀의 실제 10,000-request 성능은 §1만이 대표값이다. Checkpoint 누적곡선은 seen-prefix retention의 진행을, v3 layer probe는 action–realization mechanism을 각각 별도로 보여준다. 이 backfill은 기존 observational 결론의 성능 분모를 교정하지만 barrier 효용이나 인과성을 새로 주장하지 않는다.",
        "",
    ]
    return "\n".join(lines)


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise FinalWeightBoundary(f"refusing to reuse canonical output: {args.output}")
    args.output.mkdir(parents=True, mode=0o755)
    head = _git(args.source_root, "rev-parse", "HEAD")
    tree = _git(args.source_root, "rev-parse", "HEAD^{tree}")
    if head != args.expected_head or _git(args.source_root, "status", "--porcelain", "--untracked-files=no"):
        raise FinalWeightBoundary("analysis source identity differs")
    if sha256_file(args.input_lock) != args.expected_input_lock_sha:
        raise FinalWeightBoundary("analysis availability lock SHA differs")
    input_lock = _load_json(args.input_lock)
    _verify_identity(input_lock, str(args.input_lock))
    if input_lock.get("checkpoint_count") != 28:
        raise FinalWeightBoundary("analysis availability checkpoint denominator differs")
    stream_hashes = _stream_request_hashes()
    locked_checkpoints = {
        (str(value["model"]), str(value["method"]), int(value["accepted_edit_count"])): value
        for value in input_lock["checkpoints"]
    }
    if len(locked_checkpoints) != 28:
        raise FinalWeightBoundary("analysis checkpoint lock uniqueness differs")
    control_receipts = [
        _verified_external(args.source_rebind_receipt, args.expected_source_rebind_sha, "TECH_R3_SOURCE_REBIND"),
        _verified_external(args.submission_receipt, args.expected_submission_sha, "JOB33306_SUBMISSION"),
        _verified_external(args.technical_exclusion, args.expected_technical_exclusion_sha, "JOB33300_TECHNICAL_EXCLUSION"),
    ]
    v3_root = args.source_root / V3_REPORT_RELATIVE
    v3_paths = {
        "report": (v3_root / "official-layer-realization-debt-lifelong-fourarm-exhaustive-factual-ko.md", V3_REPORT_SHA256),
        "manifest": (v3_root / "analysis-manifest.json", V3_MANIFEST_SHA256),
        "receipt": (v3_root / "rooted-analysis-receipt.json", V3_RECEIPT_SHA256),
    }
    for label, (path, digest) in v3_paths.items():
        if sha256_file(path) != digest:
            raise FinalWeightBoundary(f"immutable v3 {label} differs")

    cumulative_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    age_rows: list[dict[str, Any]] = []
    compute_rows: list[dict[str, Any]] = []
    raw_members: list[dict[str, Any]] = [
        _raw_member(
            kind="INPUT_LOCK", arm="GLOBAL", path=args.input_lock,
            digest=args.expected_input_lock_sha, accepted_edit_count="LOCK",
        ),
        _raw_member(
            kind="STREAM_SEAL", arm="GLOBAL", path=STREAM_SEAL,
            digest=STREAM_SEAL_SHA256, accepted_edit_count="LOCK",
        ),
    ]
    raw_members.extend(
        _raw_member(
            kind=value["label"], arm="GLOBAL", path=Path(value["path"]),
            digest=value["sha256"], accepted_edit_count="CONTROL",
        )
        for value in control_receipts
    )
    terminal_identities: list[dict[str, Any]] = []
    seen_arms: set[str] = set()
    for cell in range(4):
        terminal_path = args.result_root / f"cell-{cell}" / "terminal.json"
        if not _within(terminal_path, args.result_root):
            raise FinalWeightBoundary(f"cell terminal escaped result root: {cell}")
        terminal = _load_json(terminal_path)
        _verify_identity(terminal, str(terminal_path))
        if (
            terminal.get("status") != "TERMINAL_PASS"
            or terminal.get("checkpoint_denominator") != 7
            or terminal.get("request_state_denominator") != sum(AMENDED_CHECKPOINTS)
            or terminal.get("nonfinite_count") != 0
            or terminal.get("failure_count") != 0
            or terminal.get("imputation_count") != 0
            or terminal.get("terminal_w0_restore", {}).get("exact") is not True
            or terminal.get("source", {}).get("head") != args.expected_runtime_head
            or terminal.get("source", {}).get("tree") != args.expected_runtime_tree
            or terminal.get("source", {}).get("tracked_clean") is not True
            or terminal.get("model_binding", {}).get("full_fp32") is not True
            or terminal.get("model_binding", {}).get("autocast") is not False
            or terminal.get("model_binding", {}).get("quantized") is not False
        ):
            raise FinalWeightBoundary(f"cell terminal validity differs: {cell}")
        model, method = str(terminal["model"]), str(terminal["method"])
        arm = ARM_KEY[(model, method)]
        if arm in seen_arms:
            raise FinalWeightBoundary(f"duplicate model/method terminal: {arm}")
        seen_arms.add(arm)
        terminal_sha = sha256_file(terminal_path)
        raw_members.append(
            _raw_member(
                kind="CELL_TERMINAL", arm=arm, path=terminal_path,
                digest=terminal_sha, accepted_edit_count="TERMINAL",
            )
        )
        terminal_identities.append({"arm": arm, "sha256": terminal_sha, "identity_sha256": terminal["identity_sha256"]})
        counts_in_terminal = [int(entry["accepted_edit_count"]) for entry in terminal["checkpoint_receipts"]]
        if counts_in_terminal != list(AMENDED_CHECKPOINTS):
            raise FinalWeightBoundary(f"cell checkpoint schedule differs: {arm}")
        for entry in terminal["checkpoint_receipts"]:
            count = int(entry["accepted_edit_count"])
            receipt_path = Path(entry["path"])
            if not _within(receipt_path, terminal_path.parent):
                raise FinalWeightBoundary("checkpoint receipt escaped cell root")
            if sha256_file(receipt_path) != entry["sha256"]:
                raise FinalWeightBoundary("checkpoint evaluation receipt SHA differs")
            receipt = _load_json(receipt_path)
            _verify_identity(receipt, str(receipt_path))
            if (
                receipt.get("accepted_edit_count") != count
                or receipt.get("evaluation_type") != "CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS"
                or receipt.get("weight_pointer_version_bytes_exact") is not True
                or receipt.get("nonfinite_count") != 0
                or receipt.get("state_loaded_once") is not True
                or receipt.get("cache_loaded_into_runtime") is not False
                or receipt.get("before_evaluation_weight_identity") != receipt.get("after_evaluation_weight_identity")
                or receipt.get("source_checkpoint") != locked_checkpoints[(model, method, count)]
                or receipt.get("is_final_w_full10k") != (count == 10_000)
            ):
                raise FinalWeightBoundary("checkpoint evaluation validity differs")
            zero_counts = (
                "compute_z_count", "writer_count", "key_count", "solve_count",
                "cache_append_count", "cache_consume_count", "history_append_count",
                "optimizer_count", "backward_count", "parameter_gradient_count",
                "model_update_during_evaluation_count", "duplicate_evaluation_count",
                "imputation_count",
            )
            if any(int(receipt["compute"][key]) != 0 for key in zero_counts):
                raise FinalWeightBoundary("evaluation-only authority count differs")
            records_spec = receipt["records"]
            records_path = Path(records_spec["path"])
            if not _within(records_path, terminal_path.parent) or int(records_spec["rows"]) != count:
                raise FinalWeightBoundary("request-metric member boundary differs")
            records = _read_records(records_path, records_spec["sha256"], count)
            if [str(row["request_sha256"]) for row in records] != list(stream_hashes[:count]):
                raise FinalWeightBoundary("evaluation request order differs")
            if [int(row["ordinal"]) for row in records] != list(range(count)):
                raise FinalWeightBoundary("evaluation ordinal differs")
            recomputed = _recompute(records)
            _validate_against_runtime(recomputed, receipt["summary"])
            if (
                recomputed["request_denominator"] != count
                or recomputed["gen_prompt_denominator"] != 2 * count
                or recomputed["loc_denominator"] != 10 * count
            ):
                raise FinalWeightBoundary("checkpoint evaluator denominator differs")
            summary_row = _summary_row(arm, model, method, count, recomputed)
            cumulative_rows.append(summary_row)
            if count == 10_000:
                final_rows.append(summary_row)
            for record in records:
                request_rows.append(_flatten_request(arm, model, method, count, record))
            for age in AGE_ORDER:
                value = recomputed["age"][age]
                age_rows.append({"arm": arm, "model": model, "method": method, "accepted_edit_count": count, "age_stratum": age, **value})
            raw_members.extend(
                (
                    _raw_member(
                        kind="CHECKPOINT_EVALUATION_RECEIPT", arm=arm,
                        path=receipt_path, digest=entry["sha256"],
                        accepted_edit_count=count,
                    ),
                    _raw_member(
                        kind="REQUEST_METRICS", arm=arm, path=records_path,
                        digest=records_spec["sha256"], accepted_edit_count=count,
                    ),
                )
            )
        compute = terminal["counts"]
        compute_rows.append(
            {
                "arm": arm,
                "model": model,
                "method": method,
                "job_id": args.job_id,
                "checkpoint_load_count": 7,
                "request_state_denominator": compute["request_count"],
                "model_forward_batch_count": compute["forward_batch_count"],
                "token_example_count": compute["token_example_count"],
                "compute_z_count": 0,
                "writer_count": 0,
                "key_count": 0,
                "solve_count": 0,
                "cache_history_mutation_count": 0,
                "optimizer_count": 0,
                "backward_count": 0,
                "parameter_gradient_count": 0,
                "model_update_during_evaluation_count": 0,
                "duplicate_evaluation_count": 0,
                "imputation_count": 0,
                "wall_seconds": terminal["wall_seconds"],
                "wall_hours": float(terminal["wall_seconds"]) / 3600.0,
                "peak_gpu_memory_bytes": terminal["peak_gpu_memory_bytes"],
                "full_fp32": True,
                "nonfinite_count": 0,
            }
        )
    if seen_arms != set(ARM_ORDER) or {row["arm"] for row in final_rows} != set(ARM_ORDER) or len(cumulative_rows) != 28 or len(request_rows) != 120_000:
        raise FinalWeightBoundary("four-arm aggregate denominator differs")
    cumulative_rows.sort(key=lambda row: (ARM_ORDER.index(str(row["arm"])), int(row["accepted_edit_count"])))
    final_rows.sort(key=lambda row: ARM_ORDER.index(str(row["arm"])))
    age_rows.sort(key=lambda row: (ARM_ORDER.index(str(row["arm"])), int(row["accepted_edit_count"]), AGE_ORDER.index(str(row["age_stratum"]))))
    compute_rows.sort(key=lambda row: ARM_ORDER.index(str(row["arm"])))
    tables = {}
    tables["final"] = _write_csv(args.output / "finalw-full10k-arm-summary.csv", final_rows)
    tables["cumulative"] = _write_csv(args.output / "cumulative-seen-prefix-arm-summary.csv", cumulative_rows)
    tables["requests"] = _write_csv(args.output / "cumulative-seen-prefix-request-complete.csv.gz", request_rows, gzip_output=True)
    tables["age"] = _write_csv(args.output / "edit-age-strata-summary.csv", age_rows)
    tables["compute"] = _write_csv(args.output / "evaluation-compute-accounting.csv", compute_rows)
    amendment_rows = []
    for value in input_lock["checkpoints"]:
        amendment_rows.append(
            {
                "model": value["model"], "method": value["method"],
                "accepted_edit_count": value["accepted_edit_count"],
                "checkpoint_path": value["checkpoint"]["path"], "checkpoint_sha256": value["checkpoint"]["sha256"],
                "state_path": value["state"]["path"], "state_bytes": value["state"]["bytes"], "state_sha256": value["state"]["sha256"],
                "edited_weight_identity_sha256": value["edited_weight_identity_sha256"],
                "cache_or_covariance_kind": value["cache_or_covariance_binding"]["kind"],
            }
        )
    tables["amendment"] = _write_csv(args.output / "checkpoint-state-amendment.csv", amendment_rows)
    tables["raw"] = _write_csv(args.output / "evaluation-raw-member-inventory.csv", raw_members)
    _write_json_create_once(
        args.output / "evaluation-raw-member-inventory.json", raw_members
    )
    technical = [
        {"job_or_stage": "PRECHECK_CAMPAIGN_20260903_V1", "classification": "PURE_TECHNICAL_TERMINAL_STATUS_ADAPTER", "scientific_denominator": 0, "immutable_evidence": "/data/janghj/ODE-edit/local/state/official-layer-realization-debt-lifelong-finalw-full10k-v1/campaign-20260903-v1"},
        {"job_or_stage": "PRECHECK_CAMPAIGN_20260903_TECH_R1", "classification": "PURE_TECHNICAL_V3_CHECKOUT_PATH_BINDING", "scientific_denominator": 0, "immutable_evidence": "/data/janghj/ODE-edit/local/state/official-layer-realization-debt-lifelong-finalw-full10k-v1/campaign-20260903-tech-r1"},
        {"job_or_stage": "33300", "classification": "PURE_TECHNICAL_CROSS_REQUEST_EVALUATOR_BATCHING_PARITY", "scientific_denominator": 0, "immutable_evidence": "/data/janghj/ODE-edit/local/results/official-layer-realization-debt-lifelong-finalw-full10k-v1/campaign-20260903-tech-r2"},
    ]
    tables["technical"] = _write_csv(args.output / "technical-exclusions.csv", technical)
    _copy_create_once(
        args.input_lock,
        args.output / "availability-amendment-receipt.json",
        args.expected_input_lock_sha,
    )
    _copy_create_once(
        args.source_rebind_receipt,
        args.output / "evaluation-source-rebind-receipt.json",
        args.expected_source_rebind_sha,
    )
    _copy_create_once(
        args.submission_receipt,
        args.output / "evaluation-submission-receipt.json",
        args.expected_submission_sha,
    )
    _copy_create_once(
        args.technical_exclusion,
        args.output / "job33300-technical-exclusion.json",
        args.expected_technical_exclusion_sha,
    )

    figure_paths = generate_figures(args.output, args.output)
    plot_code = Path(__file__).with_name("lifelong_finalw_figures.py")
    figure_rows = []
    command = (
        f"MPLCONFIGDIR=/tmp/odeedit-finalw-v4-mpl python -m "
        f"project.run_scripts.official_layer_realization_debt.lifelong_finalw_figures "
        f"--tables {args.output} --output CREATE_ONCE_OUTPUT"
    )
    for path in figure_paths:
        if "performance" in path.name and "cumulative" not in path.name:
            input_name = "finalw-full10k-arm-summary.csv"
        elif "target-nll" in path.name:
            input_name = "finalw-full10k-arm-summary.csv"
        elif "cumulative" in path.name:
            input_name = "cumulative-seen-prefix-arm-summary.csv"
        else:
            input_name = "edit-age-strata-summary.csv"
        figure_rows.append(
            {
                "figure": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "plot_source_path": str(plot_code),
                "plot_source_sha256": sha256_file(plot_code),
                "input_table": input_name,
                "input_table_sha256": sha256_file(args.output / input_name),
                "command": command,
                "backend": "Agg",
                "dpi": 180,
                "missing_policy": "NO_INTERPOLATION_NO_IMPUTATION",
            }
        )
    _write_csv(args.output / "plot-reproduction.csv", figure_rows)
    report = _build_report(
        output=args.output,
        source={
            "analysis_head": head,
            "analysis_tree": tree,
            "runtime_head": args.expected_runtime_head,
            "runtime_tree": args.expected_runtime_tree,
        },
        final_rows=final_rows,
        cumulative_rows=cumulative_rows,
        age_rows=age_rows,
        compute_rows=compute_rows,
        amendment_sha=args.expected_input_lock_sha,
        result_root=args.result_root,
        v3_root=v3_root,
    )
    report_path = args.output / REPORT_NAME
    report_path.write_text(report, encoding="utf-8")
    members = []
    for path in sorted(args.output.iterdir(), key=lambda value: value.name):
        if path.name in {"analysis-manifest.json", "rooted-analysis-receipt.json"}:
            continue
        members.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema": f"{SCHEMA}.analysis-manifest-v4",
        "instruction_id": INSTRUCTION_ID,
        "status": "FOUR_ARM_FINAL_W_FULL10K_EVALUATION_TERMINAL_VALID",
        "source": {
            "analysis": {"root": str(args.source_root), "head": head, "tree": tree, "tracked_clean": True},
            "runtime": {"head": args.expected_runtime_head, "tree": args.expected_runtime_tree},
        },
        "job_id": args.job_id,
        "denominators": {"arms": 4, "checkpoints": 28, "request_states": 120_000, "final_requests": 40_000, "final_rephrase_prompts": 80_000, "final_locality_prompts": 400_000},
        "schedule_amendment": {"original": list(ORIGINAL_REQUESTED_CHECKPOINTS), "exact_stored": list(AMENDED_CHECKPOINTS), "absent": list(ABSENT_EXACT_CHECKPOINTS), "outcome_blind": True, "replay": 0, "interpolation": 0, "substitution": 0},
        "input_lock": {"path": str(args.input_lock), "sha256": args.expected_input_lock_sha, "identity_sha256": input_lock["identity_sha256"]},
        "control_receipts": control_receipts,
        "terminal_identities": terminal_identities,
        "raw_member_root": canonical_hash(raw_members),
        "external_v3": {label: {"path": str(path), "sha256": digest} for label, (path, digest) in v3_paths.items()},
        "members": members,
        "member_root": canonical_hash(members),
        "raw_prompt_logit_generation_publish_count": 0,
        "request_row_identity_scope": "CORE_EVALUATION_ROW_BEFORE_DERIVED_AGE_STRATUM",
        "codex_visualization_or_imagegen_count": 0,
        "imputation_count": 0,
        "scientific_promotion": False,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_path = args.output / "analysis-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schema": f"{SCHEMA}.rooted-analysis-receipt-v4",
        "instruction_id": INSTRUCTION_ID,
        "status": manifest["status"],
        "manifest": {"path": str(manifest_path), "bytes": manifest_path.stat().st_size, "sha256": sha256_file(manifest_path), "identity_sha256": manifest["identity_sha256"]},
        "report": {"path": str(report_path), "bytes": report_path.stat().st_size, "sha256": sha256_file(report_path)},
        "member_root": manifest["member_root"],
        "external_raw_member_root": manifest["raw_member_root"],
        "scientific_promotion": False,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    receipt_path = args.output / "rooted-analysis-receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "status": manifest["status"],
        "report_path": str(report_path), "report_sha256": receipt["report"]["sha256"],
        "manifest_path": str(manifest_path), "manifest_sha256": receipt["manifest"]["sha256"],
        "receipt_path": str(receipt_path), "receipt_sha256": sha256_file(receipt_path),
        "member_root": manifest["member_root"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-runtime-head", required=True)
    parser.add_argument("--expected-runtime-tree", required=True)
    parser.add_argument("--input-lock", type=Path, required=True)
    parser.add_argument("--expected-input-lock-sha", required=True)
    parser.add_argument("--source-rebind-receipt", type=Path, required=True)
    parser.add_argument("--expected-source-rebind-sha", required=True)
    parser.add_argument("--submission-receipt", type=Path, required=True)
    parser.add_argument("--expected-submission-sha", required=True)
    parser.add_argument("--technical-exclusion", type=Path, required=True)
    parser.add_argument("--expected-technical-exclusion-sha", required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
