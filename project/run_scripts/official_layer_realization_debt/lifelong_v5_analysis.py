"""Build the canonical v5 lifelong report from sealed v3 and v4 evidence.

This module is deliberately analysis-only.  It never loads a model, edits a
checkpoint, invokes an evaluator, or mutates the v3/v4 packages.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from project.run_scripts.ode_bf.contracts import canonical_hash

from .lifelong_finalw_contracts import (
    EVALUATOR_IDENTITY,
    ORDER_ROOT,
    STREAM_ROOT,
    V3_MANIFEST_SHA256,
    V3_RECEIPT_SHA256,
    V3_REPORT_SHA256,
)
from .lifelong_v5_figures import generate as generate_figures


INSTRUCTION_ID = "ODEEDIT-S06-OFFICIAL-LAYER-DEBT-LIFELONG-V5-EXHAUSTIVE-CUMULATIVE-INTEGRATION"
REPORT_NAME = "official-layer-realization-debt-lifelong-v5-exhaustive-cumulative-factual-ko.md"
V3_RELATIVE = Path(
    "experiment-reports/servers/server4/"
    "official-layer-realization-debt-lifelong-b100x100-2026-09-03-v3"
)
V4_RELATIVE = Path(
    "experiment-reports/servers/server4/"
    "official-layer-realization-debt-lifelong-b100x100-2026-09-03-v4-r2"
)
V4_REPORT_SHA256 = "225fbe8cc7fea44a2e9e00b01a1e3648686e439cd6618a150165b3f33cc95c68"
V4_MANIFEST_SHA256 = "848dd78ea44d26105eb2f75344aa252413d249fddf2b03dd104e3a78f9cea0a9"
V4_RECEIPT_SHA256 = "bffff6260cae5f4679f0c6ac7b3b5020ac2c286955caebb7e1a4683454561b04"
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
CHECKPOINTS = (1000, 1500, 2000, 3000, 5000, 7500, 10000)
AGE_ORDER = ("EARLY_FIRST_20PCT", "MIDDLE_60PCT", "RECENT_LAST_20PCT")
STATS = ("mean", "median", "p90", "max")
CATEGORIES = (
    "rewrite_target_new",
    "rewrite_target_true",
    "rephrase_target_new",
    "rephrase_target_true",
    "locality_target_true",
)
CATEGORY_LABEL = {
    "rewrite_target_new": "Rewrite / target-new",
    "rewrite_target_true": "Rewrite / target-true",
    "rephrase_target_new": "Rephrase / target-new",
    "rephrase_target_true": "Rephrase / target-true",
    "locality_target_true": "Locality / target-true",
}
CORE_METRICS = (
    ("eff", "eff_numerator", "request_denominator", "Eff"),
    ("gen_prompt", "gen_prompt_numerator", "gen_prompt_denominator", "Gen prompt"),
    ("gen_strict", "gen_strict_numerator", "gen_strict_denominator", "Gen strict"),
    ("loc", "loc_numerator", "loc_denominator", "Loc"),
    (
        "rewrite_new_preferred_rate",
        "rewrite_new_preferred_numerator",
        "request_denominator",
        "Rewrite new preference",
    ),
    (
        "rephrase_new_preferred_rate",
        "rephrase_new_preferred_numerator",
        "request_denominator",
        "Rephrase new preference",
    ),
)
MECHANISM_FIELDS = (
    "q_pre_L8_median",
    "q_post_L8_median",
    "mean_rho_median",
    "mean_tau_median",
    "d_parallel_median",
    "d_perp_median",
    "D_TV_median",
    "update_magnitude_total",
    "layer8_update_share",
    "Vbar",
    "unreachable_fraction",
)
OUTCOME_FIELDS = (
    "eff",
    "gen_prompt",
    "gen_strict",
    "loc",
    "rewrite_new_nll_median",
    "rephrase_new_nll_median",
)


class V5Boundary(RuntimeError):
    """Fail-closed v5 analysis boundary."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise V5Boundary("empty distribution")
    if len(ordered) == 1:
        return ordered[0]
    location = (len(ordered) - 1) * probability
    low, high = math.floor(location), math.ceil(location)
    return (
        ordered[low]
        if low == high
        else ordered[low] * (high - location) + ordered[high] * (location - low)
    )


def _dist(values: Sequence[float]) -> dict[str, Any]:
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise V5Boundary("invalid analysis distribution")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": _quantile(values, 0.9),
        "max": max(values),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    answer = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    answer.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(answer)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise V5Boundary(f"refusing empty v5 table: {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise V5Boundary(f"unstable v5 table schema: {path.name}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": len(rows),
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V5Boundary(f"JSON object expected: {path}")
    return value


def _identity(value: Mapping[str, Any], label: str) -> None:
    payload = dict(value)
    expected = payload.pop("identity_sha256", None)
    if expected != canonical_hash(payload):
        raise V5Boundary(f"canonical identity differs: {label}")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _gzip_rows(path: Path) -> Iterable[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _row_count(path: Path) -> int:
    if path.suffix == ".gz":
        return sum(1 for _ in _gzip_rows(path))
    return len(_rows(path))


def _verify_package(
    root: Path,
    *,
    report_name: str,
    report_sha: str,
    manifest_sha: str,
    receipt_sha: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not root.is_dir() or root.is_symlink():
        raise V5Boundary(f"sealed package root boundary differs: {root}")
    report_path = root / report_name
    manifest_path = root / "analysis-manifest.json"
    receipt_path = root / "rooted-analysis-receipt.json"
    expected = (
        (report_path, report_sha),
        (manifest_path, manifest_sha),
        (receipt_path, receipt_sha),
    )
    for path, digest in expected:
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise V5Boundary(f"sealed package anchor differs: {path}")
    manifest, receipt = _object(manifest_path), _object(receipt_path)
    _identity(manifest, f"{root.name} manifest")
    _identity(receipt, f"{root.name} receipt")
    direct_member_root = canonical_hash(manifest.get("members", []))
    tuple_member_root = canonical_hash(
        [
            [str(row["path"]), int(row["bytes"]), str(row["sha256"])]
            for row in sorted(
                manifest.get("members", []), key=lambda value: str(value["path"])
            )
        ]
    )
    if (
        receipt.get("member_root") != manifest.get("member_root")
        or receipt.get("manifest", {}).get("sha256") != manifest_sha
        or manifest.get("member_root") not in {direct_member_root, tuple_member_root}
    ):
        raise V5Boundary(f"sealed package rooted binding differs: {root}")
    members = list(manifest["members"])
    names: set[str] = set()
    for member in members:
        relative = Path(str(member["path"]))
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name in names:
            raise V5Boundary(f"sealed member path boundary differs: {relative}")
        names.add(relative.name)
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(member["bytes"])
            or sha256_file(path) != member["sha256"]
        ):
            raise V5Boundary(f"sealed member identity differs: {path}")
    actual_names = {
        path.name
        for path in root.iterdir()
        if path.is_file()
        and path.name not in {"analysis-manifest.json", "rooted-analysis-receipt.json"}
    }
    if names != actual_names:
        raise V5Boundary(f"sealed member inventory differs: {root}")
    for name, expected_rows in manifest.get("table_row_counts", {}).items():
        if _row_count(root / name) != int(expected_rows):
            raise V5Boundary(f"sealed derived-table row count differs: {root / name}")
    return manifest, receipt, members


def _verify_v4_external_members(v4: Path, manifest: Mapping[str, Any]) -> int:
    inventory_path = v4 / "evaluation-raw-member-inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if (
        not isinstance(inventory, list)
        or canonical_hash(inventory) != manifest.get("raw_member_root")
    ):
        raise V5Boundary("v4 external raw-member root differs")
    for member in inventory:
        path = Path(member["path"])
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(member["bytes"])
            or sha256_file(path) != member["sha256"]
        ):
            raise V5Boundary(f"v4 external member identity differs: {path}")
    return len(inventory)


def _validate_v4_tables(v4: Path) -> tuple[
    list[dict[str, str]], list[dict[str, str]], dict[str, list[str]]
]:
    cumulative = _rows(v4 / "cumulative-seen-prefix-arm-summary.csv")
    age = _rows(v4 / "edit-age-strata-summary.csv")
    if len(cumulative) != 28 or len(age) != 84:
        raise V5Boundary("v4 core/age row denominator differs")
    expected = {(arm, count) for arm in ARM_ORDER for count in CHECKPOINTS}
    actual = {(row["arm"], int(row["accepted_edit_count"])) for row in cumulative}
    if actual != expected:
        raise V5Boundary("v4 cumulative arm/checkpoint mapping differs")
    age_expected = {(arm, count, age_name) for arm, count in expected for age_name in AGE_ORDER}
    age_actual = {
        (row["arm"], int(row["accepted_edit_count"]), row["age_stratum"])
        for row in age
    }
    if age_actual != age_expected:
        raise V5Boundary("v4 age arm/checkpoint mapping differs")
    counts: dict[tuple[str, int], int] = {}
    ordinals: dict[tuple[str, int], list[int]] = {}
    final_hashes: dict[str, list[str]] = {arm: [] for arm in ARM_ORDER}
    complete = v4 / "cumulative-seen-prefix-request-complete.csv.gz"
    for row in _gzip_rows(complete):
        key = (row["arm"], int(row["accepted_edit_count"]))
        counts[key] = counts.get(key, 0) + 1
        ordinals.setdefault(key, []).append(int(row["ordinal"]))
        if key[1] == 10_000:
            final_hashes[key[0]].append(row["request_sha256"])
    if sum(counts.values()) != 120_000 or set(counts) != expected:
        raise V5Boundary("v4 complete request-state denominator differs")
    for key, count in counts.items():
        if count != key[1] or ordinals[key] != list(range(key[1])):
            raise V5Boundary(f"v4 complete request ordering differs: {key}")
    reference = final_hashes[ARM_ORDER[0]]
    if len(reference) != 10_000 or any(final_hashes[arm] != reference for arm in ARM_ORDER[1:]):
        raise V5Boundary("v4 full-10k request identity differs across arms")
    return cumulative, age, final_hashes


def _category_rows(v4: Path, cumulative: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    indexed = {(row["arm"], int(row["accepted_edit_count"])): row for row in cumulative}
    inventory = _rows(v4 / "evaluation-raw-member-inventory.csv")
    receipts = [row for row in inventory if row["kind"] == "CHECKPOINT_EVALUATION_RECEIPT"]
    if len(receipts) != 28:
        raise V5Boundary("v4 checkpoint receipt inventory differs")
    answer: list[dict[str, Any]] = []
    for specification in receipts:
        arm, count = specification["arm"], int(specification["accepted_edit_count"])
        path = Path(specification["path"])
        if sha256_file(path) != specification["sha256"]:
            raise V5Boundary(f"v4 receipt changed after package seal: {path}")
        receipt = _object(path)
        _identity(receipt, str(path))
        source = indexed[(arm, count)]
        if ARM_KEY[(receipt["model"], receipt["method"])] != arm:
            raise V5Boundary("receipt arm mapping differs")
        for category in CATEGORIES:
            metric = receipt["summary"]["categories"][category]
            row: dict[str, Any] = {
                "arm": arm,
                "model": receipt["model"],
                "method": receipt["method"],
                "accepted_edit_count": count,
                "category": category,
                "prompt_denominator": metric["prompt_denominator"],
                "strict_numerator": metric["strict_numerator"],
                "strict_rate": metric["strict_rate"],
                "token_denominator": metric["token_denominator"],
                "token_accuracy": metric["token_accuracy"],
            }
            for family in (
                "request_cluster_nll",
                "request_cluster_min_margin",
                "prompt_nll",
                "prompt_margin",
            ):
                for statistic in ("n", *STATS):
                    row[f"{family}_{statistic}"] = metric[family][statistic]
            for statistic in STATS:
                if not math.isclose(
                    float(source[f"{category}_nll_{statistic}"]),
                    float(row[f"request_cluster_nll_{statistic}"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise V5Boundary(f"v4 receipt/table NLL differs: {arm}/{count}/{category}")
                if not math.isclose(
                    float(source[f"{category}_margin_{statistic}"]),
                    float(row[f"request_cluster_min_margin_{statistic}"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise V5Boundary(f"v4 receipt/table margin differs: {arm}/{count}/{category}")
            answer.append(row)
    answer.sort(
        key=lambda row: (
            ARM_ORDER.index(str(row["arm"])),
            int(row["accepted_edit_count"]),
            CATEGORIES.index(str(row["category"])),
        )
    )
    if len(answer) != 140:
        raise V5Boundary("cumulative category row denominator differs")
    return answer


def _core_rows(cumulative: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    fields = (
        "arm", "model", "method", "accepted_edit_count", "evaluation_type",
        "request_denominator", "eff_numerator", "eff",
        "gen_prompt_numerator", "gen_prompt_denominator", "gen_prompt",
        "gen_strict_numerator", "gen_strict_denominator", "gen_strict",
        "loc_numerator", "loc_denominator", "loc",
        "rewrite_new_preferred_numerator", "rewrite_new_preferred_rate",
        "rephrase_new_preferred_numerator", "rephrase_new_preferred_rate",
    )
    return [{field: row[field] for field in fields} for row in cumulative]


def _spans(rows: Sequence[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], Mapping[str, Any], str]]:
    ordered = sorted(rows, key=lambda row: int(row["accepted_edit_count"]))
    spans = [(ordered[index - 1], ordered[index], "ADJACENT") for index in range(1, len(ordered))]
    spans.append((ordered[0], ordered[-1], "TOTAL_1K_TO_10K"))
    return spans


def _core_transition_rows(core: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    answer: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        selected = [row for row in core if row["arm"] == arm]
        for before, after, span in _spans(selected):
            for field, numerator, denominator, label in CORE_METRICS:
                answer.append(
                    {
                        "arm": arm,
                        "from_checkpoint": before["accepted_edit_count"],
                        "to_checkpoint": after["accepted_edit_count"],
                        "span": span,
                        "metric": field,
                        "metric_label": label,
                        "from_numerator": before[numerator],
                        "from_denominator": before[denominator],
                        "from_value": before[field],
                        "to_numerator": after[numerator],
                        "to_denominator": after[denominator],
                        "to_value": after[field],
                        "delta": float(after[field]) - float(before[field]),
                    }
                )
    return answer


def _category_transition_rows(categories: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    answer: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        for category in CATEGORIES:
            selected = [row for row in categories if row["arm"] == arm and row["category"] == category]
            for before, after, span in _spans(selected):
                row: dict[str, Any] = {
                    "arm": arm,
                    "category": category,
                    "from_checkpoint": before["accepted_edit_count"],
                    "to_checkpoint": after["accepted_edit_count"],
                    "span": span,
                    "from_prompt_denominator": before["prompt_denominator"],
                    "to_prompt_denominator": after["prompt_denominator"],
                    "from_strict_rate": before["strict_rate"],
                    "to_strict_rate": after["strict_rate"],
                    "delta_strict_rate": float(after["strict_rate"]) - float(before["strict_rate"]),
                }
                for family in (
                    "request_cluster_nll",
                    "request_cluster_min_margin",
                    "prompt_nll",
                    "prompt_margin",
                ):
                    for statistic in STATS:
                        key = f"{family}_{statistic}"
                        row[f"from_{key}"] = before[key]
                        row[f"to_{key}"] = after[key]
                        row[f"delta_{key}"] = float(after[key]) - float(before[key])
                answer.append(row)
    return answer


def _paired_rows(
    core: Sequence[Mapping[str, Any]], categories: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparisons = (
        ("LA", "LM", "ALPHAEDIT_MINUS_MEMIT_LLAMA"),
        ("QA", "QM", "ALPHAEDIT_MINUS_MEMIT_QWEN"),
        ("LM", "QM", "LLAMA_MINUS_QWEN_MEMIT"),
        ("LA", "QA", "LLAMA_MINUS_QWEN_ALPHAEDIT"),
    )
    core_index = {(row["arm"], int(row["accepted_edit_count"])): row for row in core}
    category_index = {
        (row["arm"], int(row["accepted_edit_count"]), row["category"]): row
        for row in categories
    }
    paired_core: list[dict[str, Any]] = []
    paired_category: list[dict[str, Any]] = []
    for left_arm, right_arm, label in comparisons:
        for count in CHECKPOINTS:
            left, right = core_index[(left_arm, count)], core_index[(right_arm, count)]
            for field, numerator, denominator, metric_label in CORE_METRICS:
                paired_core.append(
                    {
                        "comparison": label,
                        "left_arm": left_arm,
                        "right_arm": right_arm,
                        "accepted_edit_count": count,
                        "metric": field,
                        "metric_label": metric_label,
                        "left_numerator": left[numerator],
                        "left_denominator": left[denominator],
                        "left_value": left[field],
                        "right_numerator": right[numerator],
                        "right_denominator": right[denominator],
                        "right_value": right[field],
                        "delta_left_minus_right": float(left[field]) - float(right[field]),
                    }
                )
            for category in CATEGORIES:
                left_cat = category_index[(left_arm, count, category)]
                right_cat = category_index[(right_arm, count, category)]
                row: dict[str, Any] = {
                    "comparison": label,
                    "left_arm": left_arm,
                    "right_arm": right_arm,
                    "accepted_edit_count": count,
                    "category": category,
                    "left_strict_numerator": left_cat["strict_numerator"],
                    "left_prompt_denominator": left_cat["prompt_denominator"],
                    "right_strict_numerator": right_cat["strict_numerator"],
                    "right_prompt_denominator": right_cat["prompt_denominator"],
                    "delta_strict_rate": float(left_cat["strict_rate"]) - float(right_cat["strict_rate"]),
                }
                for family in (
                    "request_cluster_nll",
                    "request_cluster_min_margin",
                    "prompt_nll",
                    "prompt_margin",
                ):
                    for statistic in STATS:
                        key = f"{family}_{statistic}"
                        row[f"delta_{key}"] = float(left_cat[key]) - float(right_cat[key])
                paired_category.append(row)
    return paired_core, paired_category


def _recent_minus_early(age: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    indexed = {
        (row["arm"], int(row["accepted_edit_count"]), row["age_stratum"]): row
        for row in age
    }
    answer: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        arm_counts = sorted(
            {int(row["accepted_edit_count"]) for row in age if row["arm"] == arm}
        )
        for count in arm_counts:
            early = indexed[(arm, count, AGE_ORDER[0])]
            recent = indexed[(arm, count, AGE_ORDER[2])]
            row: dict[str, Any] = {
                "arm": arm,
                "accepted_edit_count": count,
                "early_request_denominator": early["request_denominator"],
                "recent_request_denominator": recent["request_denominator"],
            }
            for metric in ("eff", "gen_prompt", "gen_strict", "loc"):
                row[f"early_{metric}"] = early[metric]
                row[f"recent_{metric}"] = recent[metric]
                row[f"delta_recent_minus_early_{metric}"] = float(recent[metric]) - float(early[metric])
            for category in CATEGORIES:
                for quantity in ("nll", "margin"):
                    for statistic in STATS:
                        key = f"{category}_{quantity}_{statistic}"
                        row[f"delta_recent_minus_early_{key}"] = float(recent[key]) - float(early[key])
            answer.append(row)
    return answer


def _online_forgetting_gap(
    v3: Path,
    cumulative: Sequence[Mapping[str, str]],
    final_hashes: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = {arm: [] for arm in ARM_ORDER}
    for row in _gzip_rows(v3 / "production-request-complete.csv.gz"):
        arm = ARM_KEY[(row["model"], row["method"])]
        grouped[arm].append(row)
    cumulative_index = {(row["arm"], int(row["accepted_edit_count"])): row for row in cumulative}
    answer: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        rows = grouped[arm]
        if len(rows) != 10_000 or [row["request_sha256"] for row in rows] != list(final_hashes[arm]):
            raise V5Boundary(f"v3 online/v4 seen-prefix request identity differs: {arm}")
        for count in CHECKPOINTS:
            prefix = rows[:count]
            if max(int(row["accepted_edit_count"]) for row in prefix) != count:
                raise V5Boundary(f"v3 online prefix clock differs: {arm}/{count}")
            current = cumulative_index[(arm, count)]
            for target in ("new", "true"):
                nll = [float(row[f"target_{target}_nll"]) for row in prefix]
                margin = [float(row[f"target_{target}_margin"]) for row in prefix]
                strict = sum(int(row[f"target_{target}_strict"]) for row in prefix)
                category = f"rewrite_target_{target}"
                row_out: dict[str, Any] = {
                    "arm": arm,
                    "accepted_edit_count": count,
                    "category": category,
                    "request_denominator": count,
                    "online_strict_numerator": strict,
                    "online_strict_rate": strict / count,
                    "cumulative_strict_numerator": current[f"{category}_strict_numerator"],
                    "cumulative_strict_rate": current[f"{category}_strict_rate"],
                    "delta_cumulative_minus_online_strict_rate": float(current[f"{category}_strict_rate"]) - strict / count,
                    "online_gen": "NOT_RECORDED_ONLINE_SCHEMA",
                    "online_locality": "NOT_RECORDED_ONLINE_SCHEMA",
                }
                for quantity, values in (("nll", nll), ("margin", margin)):
                    distribution = _dist(values)
                    for statistic in STATS:
                        cumulative_value = float(current[f"{category}_{quantity}_{statistic}"])
                        row_out[f"online_{quantity}_{statistic}"] = distribution[statistic]
                        row_out[f"cumulative_{quantity}_{statistic}"] = cumulative_value
                        row_out[f"delta_cumulative_minus_online_{quantity}_{statistic}"] = cumulative_value - float(distribution[statistic])
                answer.append(row_out)
    return answer


def _mechanism_rows(v3: Path, cumulative: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    checkpoint = _rows(v3 / "checkpoint-summary.csv")
    updates = _rows(v3 / "checkpoint-update-magnitude-share.csv")
    performance = {(row["arm"], int(row["accepted_edit_count"])): row for row in cumulative}
    request_metrics: dict[tuple[str, int, str], Mapping[str, str]] = {}
    geometry: dict[tuple[str, int], Mapping[str, str]] = {}
    for row in checkpoint:
        if row["fork"] != "ACCUMULATED_OR_STATIC" or not row["accepted_edit_count"]:
            continue
        count = int(row["accepted_edit_count"])
        if count not in CHECKPOINTS:
            continue
        arm = ARM_KEY[(row["model"], row["method"])]
        if row["source_table"] == "checkpoint_request":
            request_metrics[(arm, count, row["metric"])] = row
        elif row["source_table"] == "checkpoint_geometry":
            geometry[(arm, count)] = row
    update_index: dict[tuple[str, int, str, int], Mapping[str, str]] = {}
    for row in updates:
        if row["fork"] != "ACCUMULATED_OR_STATIC" or int(row["accepted_edit_count"]) not in CHECKPOINTS:
            continue
        arm = ARM_KEY[(row["model"], row["method"])]
        update_index[(arm, int(row["accepted_edit_count"]), row["metric"], int(row["layer"]))] = row
    answer: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        for count in CHECKPOINTS:
            row: dict[str, Any] = {
                "arm": arm,
                "accepted_edit_count": count,
            }
            for metric in (
                "q_pre_L8", "q_post_L8", "mean_rho", "mean_tau",
                "d_parallel", "d_perp", "D_TV",
            ):
                source = request_metrics.get((arm, count, metric))
                if source is None or source["availability"] != "RECORDED":
                    raise V5Boundary(f"v3 checkpoint mechanism metric absent: {arm}/{count}/{metric}")
                row[f"{metric}_median"] = source["median"]
                row[f"{metric}_p90"] = source["p90"]
            row["update_magnitude_total"] = sum(
                float(update_index[(arm, count, "frobenius_magnitude", layer)]["mean"])
                for layer in (4, 5, 6, 7, 8)
            )
            row["layer8_update_share"] = update_index[
                (arm, count, "update_magnitude_share", 8)
            ]["mean"]
            geo = geometry[(arm, count)]
            row["Vbar"] = geo["Vbar"]
            row["unreachable_fraction"] = geo["unreachable_fraction"]
            current = performance[(arm, count)]
            row.update(
                {
                    "eff": current["eff"],
                    "gen_prompt": current["gen_prompt"],
                    "gen_strict": current["gen_strict"],
                    "loc": current["loc"],
                    "rewrite_new_nll_median": current["rewrite_target_new_nll_median"],
                    "rephrase_new_nll_median": current["rephrase_target_new_nll_median"],
                }
            )
            answer.append(row)
    return answer


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[indexed[position][0]] = rank
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return 0.0 if denominator == 0.0 else numerator / denominator


def _association_rows(mechanism: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    answer: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        rows = sorted(
            (row for row in mechanism if row["arm"] == arm),
            key=lambda row: int(row["accepted_edit_count"]),
        )
        for mechanism_metric in MECHANISM_FIELDS:
            x = [float(row[mechanism_metric]) for row in rows]
            for outcome_metric in OUTCOME_FIELDS:
                y = [float(row[outcome_metric]) for row in rows]
                spearman = _pearson(_average_ranks(x), _average_ranks(y))
                pearson = _pearson(x, y)
                answer.append(
                    {
                        "arm": arm,
                        "mechanism_metric": mechanism_metric,
                        "outcome_metric": outcome_metric,
                        "checkpoint_n": len(rows),
                        "spearman": spearman,
                        "pearson": pearson,
                        "spearman_direction": "POSITIVE" if spearman > 0 else "NEGATIVE" if spearman < 0 else "ZERO",
                        "causal_claim": 0,
                        "small_n_boundary": "SEVEN_CHECKPOINTS_PER_ARM",
                    }
                )
    return answer


def _v3_section_body(v3: Path, number: int, new_number: int | None = None) -> str:
    report = (v3 / "official-layer-realization-debt-lifelong-fourarm-exhaustive-factual-ko.md").read_text(encoding="utf-8")
    matches = list(re.finditer(r"^## ([0-9]+)\. .*$", report, flags=re.MULTILINE))
    selected = None
    for index, match in enumerate(matches):
        if int(match.group(1)) != number:
            continue
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(report)
        selected = report[match.end() : stop].strip()
        break
    if selected is None:
        raise V5Boundary(f"v3 report section missing: {number}")
    directory = V3_RELATIVE.name
    selected = re.sub(r"\]\(([^/)][^)]*\.png)\)", rf"](../{directory}/\1)", selected)
    if new_number is not None:
        selected = re.sub(
            rf"^### {number}\.(\d+)",
            rf"### {new_number}.\1",
            selected,
            flags=re.MULTILINE,
        )
    selected = selected.replace(
        "Equal-allocation/ideal line과 `bars:` 문구는 없다.",
        "참조 guide line이나 하단 보조 문구를 추가하지 않았다.",
    )
    return selected


def _table_category(rows: Sequence[Mapping[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            (
                str(row["accepted_edit_count"]),
                CATEGORY_LABEL[str(row["category"])],
                f"{row['strict_numerator']}/{row['prompt_denominator']} ({_fmt(row['strict_rate'], 4)})",
                "/".join(_fmt(row[f"request_cluster_nll_{stat}"]) for stat in STATS),
                "/".join(_fmt(row[f"prompt_nll_{stat}"]) for stat in STATS),
                "/".join(_fmt(row[f"request_cluster_min_margin_{stat}"]) for stat in STATS),
                "/".join(_fmt(row[f"prompt_margin_{stat}"]) for stat in STATS),
                str(row["token_accuracy"]),
            )
        )
    return _markdown_table(
        (
            "t", "category", "strict n/d(rate)",
            "request NLL μ/med/p90/max", "prompt NLL μ/med/p90/max",
            "request margin μ/med/p90/max", "prompt margin μ/med/p90/max",
            "token accuracy",
        ),
        body,
    )


def _tail_direction_notes(arm: str, transitions: Sequence[Mapping[str, Any]]) -> str:
    notes = []
    for row in transitions:
        if row["arm"] != arm or row["category"] not in {"rewrite_target_new", "rephrase_target_new"} or row["span"] != "ADJACENT":
            continue
        median = float(row["delta_request_cluster_nll_median"])
        p90 = float(row["delta_request_cluster_nll_p90"])
        maximum = float(row["delta_request_cluster_nll_max"])
        if median * p90 < 0 or median * maximum < 0:
            notes.append(
                f"{row['from_checkpoint']}→{row['to_checkpoint']} {CATEGORY_LABEL[row['category']]}: "
                f"Δmedian={median:.4f}, Δp90={p90:.4f}, Δmax={maximum:.4f}"
            )
    return "; ".join(notes) if notes else "median·p90·max 변화 방향이 갈린 adjacent 구간 없음"


def _report(
    *,
    v3: Path,
    v4: Path,
    source_head: str,
    source_tree: str,
    core: Sequence[Mapping[str, Any]],
    categories: Sequence[Mapping[str, Any]],
    core_transitions: Sequence[Mapping[str, Any]],
    category_transitions: Sequence[Mapping[str, Any]],
    paired_core: Sequence[Mapping[str, Any]],
    paired_category: Sequence[Mapping[str, Any]],
    age: Sequence[Mapping[str, str]],
    recent_early: Sequence[Mapping[str, Any]],
    forgetting: Sequence[Mapping[str, Any]],
    mechanism: Sequence[Mapping[str, Any]],
    associations: Sequence[Mapping[str, Any]],
    package_inventory: Mapping[str, Any],
) -> str:
    core_index = {(row["arm"], int(row["accepted_edit_count"])): row for row in core}
    final = [core_index[(arm, 10_000)] for arm in ARM_ORDER]
    headline = [
        (
            ARM_LABEL[row["arm"]],
            f"{row['eff_numerator']}/{row['request_denominator']} ({_fmt(row['eff'], 5)})",
            f"{row['gen_prompt_numerator']}/{row['gen_prompt_denominator']} ({_fmt(row['gen_prompt'], 5)})",
            f"{row['gen_strict_numerator']}/{row['gen_strict_denominator']} ({_fmt(row['gen_strict'], 5)})",
            f"{row['loc_numerator']}/{row['loc_denominator']} ({_fmt(row['loc'], 5)})",
        )
        for row in final
    ]
    core_table = [
        (
            row["arm"], str(row["accepted_edit_count"]), str(row["request_denominator"]),
            f"{row['eff_numerator']}/{row['request_denominator']} ({_fmt(row['eff'], 4)})",
            f"{row['gen_prompt_numerator']}/{row['gen_prompt_denominator']} ({_fmt(row['gen_prompt'], 4)})",
            f"{row['gen_strict_numerator']}/{row['gen_strict_denominator']} ({_fmt(row['gen_strict'], 4)})",
            f"{row['loc_numerator']}/{row['loc_denominator']} ({_fmt(row['loc'], 4)})",
            f"{row['rewrite_new_preferred_numerator']}/{row['request_denominator']} ({_fmt(row['rewrite_new_preferred_rate'], 4)})",
            f"{row['rephrase_new_preferred_numerator']}/{row['request_denominator']} ({_fmt(row['rephrase_new_preferred_rate'], 4)})",
        )
        for row in core
    ]
    sections: list[str] = [
        "# Official layer-write realization debt — v5 exhaustive cumulative 사실 보고서",
        "",
        "이 문서는 sealed v3 exhaustive observational report를 본문 구조로 직접 확장한 canonical successor다. v3와 v4-r2 bytes는 수정하지 않았다.",
        "",
        "상태: **TASK_COMPLETE_STOP** · 4/4 arms · 28/28 cumulative evaluations · `scientific_promotion=false`.",
        "",
        "## 1. Executive factual findings",
        "",
        "아래 첫 표만 final performance headline이다. 각 arm의 frozen W₁₀₀₀₀에서 sealed 10,000 requests 전체를 다시 평가했다. current B100나 online-at-write가 아니다.",
        "",
        _markdown_table(("model/method", "Eff", "Gen prompt", "Gen strict", "Loc"), headline),
        "",
        "![Final full-10k performance](finalw-full10k-performance.png)",
        "",
        "- FACT: final W₁₀₀₀₀에서 AlphaEdit가 같은 모델의 MEMIT보다 Eff/Gen/Loc를 더 유지했지만, 네 arm 모두 locality 절대값이 0.01413 이하였고 장기 보존은 낮았다.",
        "- FACT: Qwen MEMIT는 final W₁₀₀₀₀에서 Eff/Gen/Loc가 모두 0이었다. 실패 row 삭제나 imputation은 없다.",
        "- INFERENCE: 누적 성능과 mechanism telemetry의 동행은 n=7 checkpoint/arm의 descriptive association일 뿐 causal effect가 아니다.",
        "- DECISION: observational evidence만으로 barrier benefit, ODE necessity, universal last-layer bottleneck을 주장하지 않는다.",
        "",
        "## 2. Metric glossary / 읽는 법",
        "",
        "- **Eff (higher is better):** rewrite request의 target-new teacher-forced continuation 전체가 strict top-1이면 성공 1건이다. numerator=request successes, denominator=seen requests.",
        "- **Gen prompt (higher):** 각 rephrase prompt의 target-new continuation이 strict top-1인 prompt 비율. request당 2 prompts다.",
        "- **Gen strict (higher):** 한 request의 rephrase prompts 2개가 모두 strict일 때만 request 성공. denominator=requests다.",
        "- **Loc (higher):** locality prompt에서 target-true continuation 전체가 strict top-1인 비율. request당 10 prompts다.",
        "- **Preference (higher for new):** 같은 request에서 target-new request-cluster NLL < target-true request-cluster NLL인 비율. strict/token accuracy와 다르다.",
        "- **target-new/true NLL (lower):** 새/원래 continuation의 length-normalized negative log likelihood. prompt-level은 각 prompt, request-cluster는 request 내부 prompts를 먼저 결합한 단위다.",
        "- **margin (higher):** target token logit−최대 비-target logit의 continuation 최소값. 양수이면 strict 조건과 연결되지만 평균 margin과 strict rate는 같은 통계가 아니다.",
        "- **strict:** continuation의 모든 teacher-forced target token이 top-1인 prompt/request 조건. token accuracy는 `NOT_RECORDED_EVALUATOR_SCHEMA`이며 strict로 대체하지 않았다.",
        "- **mean/median/IQR/p90/max:** 산술평균/중앙값/(p75−p25)/90백분위/최댓값. p90/max는 큰 NLL tail 또는 큰 signed margin 쪽이므로 metric 방향과 함께 읽는다.",
        "- **R=z*−Φ(W):** layer 직전 activation residual. **A=R/n_remaining:** Official allocation. **Y:** 실제 write가 줄인 residual. **E=A−Y:** allocation-realization gap.",
        "- **q=||R||/||R_entry|| (lower for closure):** entry-normalized remaining residual. **rho=<Y,A>/||A||²:** target-aligned realization ratio. **tau=||Y−rho A||/||A||:** orthogonal distortion.",
        "- **d_parallel/d_perp:** L8 entry residual과 ideal `(1/5)R_entry` 차이의 target 평행/직교 성분. **recurrence closure:** exact residual recurrence의 FP64 relative error.",
        "- **D_TV:** layer update-share profile과 positive target-progress profile의 total variation distance. 0은 같은 profile, 1은 최대 분리다; negative progress는 별도 보존한다.",
        "- **Layer-wise Update Magnitude:** `||ΔW_l||_F`; share는 다섯 layer magnitude 합에 대한 비중. activation progress와 다른 축이며 squared-norm telemetry와 혼용하지 않는다.",
        "",
        "## 3. Provenance, jobs, denominators, invariants",
        "",
        f"- v3 report SHA `{V3_REPORT_SHA256}`; v4-r2 report SHA `{V4_REPORT_SHA256}`.",
        f"- v5 analysis source HEAD/tree `{source_head}` / `{source_tree}`.",
        f"- Common stream/order/evaluator `{STREAM_ROOT}` / `{ORDER_ROOT}` / `{EVALUATOR_IDENTITY}`.",
        "- New evaluation: 28/28 checkpoint receipts, 120,000 request-state rows, final 40,000 requests/80,000 rephrase prompts/400,000 locality prompts. FULL-FP32; nonfinite/failure/imputation/duplicate=0.",
        "- Evaluation-only compute_z/writer/key/solve/cache-history mutation/backward/gradient/model update=0; before/after weight pointer/version/bytes exact.",
        "- Stored schedule `{1000,1500,2000,3000,5000,7500,10000}`; absent `{100,500,4000,6000,8000}`. This was an outcome-blind state-availability amendment; replay/interpolation/substitution=0.",
        "",
        "Original v3 execution provenance and invariants follow in the same main-body section.",
        "",
        _v3_section_body(v3, 2, 3),
        "",
        "## 4. CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS — 28-row core rates",
        "",
        "각 W_t는 sealed first t requests 전체를 평가한다. current B100나 online-at-write가 아니다.",
        "",
        _markdown_table(
            ("arm", "t", "request n", "Eff", "Gen prompt", "Gen strict", "Loc", "rewrite pref", "rephrase pref"),
            core_table,
        ),
        "",
        "![Cumulative performance](cumulative-performance-trajectories.png)",
        "",
        "## 5. Checkpoint×arm×category full distributions",
        "",
        "각 표는 7 checkpoints×5 categories=35 rows/arm이다. NLL/margin은 mean/median/p90/max; request와 prompt 집계 단위를 분리했다.",
    ]
    for arm in ARM_ORDER:
        sections.extend(
            [
                "",
                f"### 5.{ARM_ORDER.index(arm)+1} {arm} — {ARM_LABEL[arm]}",
                "",
                _table_category([row for row in categories if row["arm"] == arm]),
            ]
        )
    sections.extend(
        [
            "",
            "![Cumulative target NLL](cumulative-target-nll-trajectories.png)",
            "",
            "![Cumulative target margin](cumulative-target-margin-trajectories.png)",
            "",
            "![Cumulative preference](cumulative-preference-trajectories.png)",
            "",
            "## 6. Checkpoint change, collapse/recovery, model/method comparisons",
            "",
        ]
    )
    narrative = {
        "LM": "LM은 Eff가 1k 0.8350→2k 0.1780으로 급락했고 3k 0.2090으로 부분 반등, 5k 0.0162로 재붕괴, 7.5k 0.0933으로 반등한 뒤 10k 0.0357로 다시 낮아졌다.",
        "LA": "LA는 Eff 1k/1.5k/2k=0.9940/0.9920/0.9865로 높게 유지했으나 3k 0.9600 이후 5k 0.6586, 7.5k 0.1957로 급락했고 10k 0.2013은 작은 반등이다.",
        "QM": "QM은 Eff 1k 0.9300→2k 0.2670→3k 0.0153으로 급붕괴했고 5k 이후 0.0002/0.0001/0.0000으로 사실상 0이었다.",
        "QA": "QA는 Eff 1k/1.5k/2k=0.9880/0.9820/0.9680으로 유지했지만 3k 0.8767, 5k 0.4306, 7.5k 0.1261, 10k 0.1084로 단계적으로 붕괴했다.",
    }
    for arm in ARM_ORDER:
        sections.extend(
            [
                f"- **{arm}:** {narrative[arm]} Gen strict와 Loc는 각 checkpoint에서 Eff보다 낮았으며 final 값은 §1의 exact n/d다.",
                f"  Tail-direction audit: {_tail_direction_notes(arm, category_transitions)}.",
            ]
        )
    for arm in ARM_ORDER:
        selected = [row for row in core_transitions if row["arm"] == arm]
        table = [
            (
                f"{row['from_checkpoint']}→{row['to_checkpoint']}",
                row["metric_label"],
                f"{row['from_numerator']}/{row['from_denominator']}→{row['to_numerator']}/{row['to_denominator']}",
                _fmt(row["delta"], 5),
            )
            for row in selected
        ]
        sections.extend(["", f"### 6.{ARM_ORDER.index(arm)+1} {arm} core transition 전량", "", _markdown_table(("span", "metric", "n/d", "delta"), table)])
    sections.extend(["", "### 6.5 Category distribution change 전량", ""])
    for arm in ARM_ORDER:
        rows = [row for row in category_transitions if row["arm"] == arm]
        table = [
            (
                f"{row['from_checkpoint']}→{row['to_checkpoint']}",
                CATEGORY_LABEL[row["category"]],
                _fmt(row["delta_strict_rate"], 4),
                "/".join(_fmt(row[f"delta_request_cluster_nll_{stat}"]) for stat in STATS),
                "/".join(_fmt(row[f"delta_request_cluster_min_margin_{stat}"]) for stat in STATS),
            )
            for row in rows
        ]
        sections.extend(["", f"#### {arm}", "", _markdown_table(("span", "category", "Δstrict", "Δrequest NLL μ/med/p90/max", "Δrequest margin μ/med/p90/max"), table)])
    comparison_labels = tuple(dict.fromkeys(row["comparison"] for row in paired_category))
    sections.extend(["", "### 6.6 Same-checkpoint paired model/method deltas", "", "Delta는 left−right다. n/d는 양쪽을 모두 기록한다."])
    for label in comparison_labels:
        core_rows_for_label = [row for row in paired_core if row["comparison"] == label]
        core_comparison = [
            (
                str(row["accepted_edit_count"]),
                row["metric_label"],
                f"{row['left_numerator']}/{row['left_denominator']} vs {row['right_numerator']}/{row['right_denominator']}",
                _fmt(row["left_value"], 4),
                _fmt(row["right_value"], 4),
                _fmt(row["delta_left_minus_right"], 4),
            )
            for row in core_rows_for_label
        ]
        rows = [row for row in paired_category if row["comparison"] == label]
        table = [
            (
                str(row["accepted_edit_count"]),
                CATEGORY_LABEL[row["category"]],
                f"{row['left_strict_numerator']}/{row['left_prompt_denominator']} vs {row['right_strict_numerator']}/{row['right_prompt_denominator']}",
                _fmt(row["delta_strict_rate"], 4),
                "/".join(_fmt(row[f"delta_request_cluster_nll_{stat}"]) for stat in STATS),
                "/".join(_fmt(row[f"delta_request_cluster_min_margin_{stat}"]) for stat in STATS),
            )
            for row in rows
        ]
        sections.extend(
            [
                "",
                f"#### {label}",
                "",
                _markdown_table(("t", "metric", "left vs right n/d", "left", "right", "delta"), core_comparison),
                "",
                _markdown_table(("t", "category", "strict n/d", "Δstrict", "ΔNLL μ/med/p90/max", "Δmargin μ/med/p90/max"), table),
            ]
        )
    sections.extend(
        [
            "",
            "## 7. Cumulative retention/forgetting — 84 checkpoint×age rows",
            "",
            "Age bins are relative within each seen prefix: first 20%, middle 60%, last 20%. Thus exact request denominators are 0.2t/0.6t/0.2t; prompt denominators follow 1/2/10 prompts per request.",
            "",
            "![Checkpoint age-strata performance](checkpoint-age-strata-performance.png)",
        ]
    )
    for arm in ARM_ORDER:
        selected = [row for row in age if row["arm"] == arm]
        core_age = [
            (
                row["accepted_edit_count"], row["age_stratum"], row["request_denominator"],
                _fmt(row["eff"], 4), _fmt(row["gen_prompt"], 4), _fmt(row["gen_strict"], 4), _fmt(row["loc"], 4),
            )
            for row in selected
        ]
        sections.extend(["", f"### 7.{ARM_ORDER.index(arm)+1} {arm} age-strata core", "", _markdown_table(("t", "age", "request n", "Eff", "Gen prompt", "Gen strict", "Loc"), core_age)])
        detail = []
        for row in selected:
            for category in CATEGORIES:
                prompt_denominator = (
                    row["request_denominator"]
                    if category.startswith("rewrite_")
                    else row["rephrase_prompt_denominator"]
                    if category.startswith("rephrase_")
                    else row["locality_prompt_denominator"]
                )
                detail.append(
                    (
                        row["accepted_edit_count"], row["age_stratum"], CATEGORY_LABEL[category], row["request_denominator"], prompt_denominator,
                        "/".join(_fmt(row[f"{category}_nll_{stat}"]) for stat in STATS),
                        "/".join(_fmt(row[f"{category}_margin_{stat}"]) for stat in STATS),
                    )
                )
        sections.extend(["", _markdown_table(("t", "age", "category", "request n", "prompt n", "NLL μ/med/p90/max", "margin μ/med/p90/max"), detail)])
    recency_table = [
        (
            row["arm"], row["accepted_edit_count"], row["early_request_denominator"], row["recent_request_denominator"],
            _fmt(row["delta_recent_minus_early_eff"], 4),
            _fmt(row["delta_recent_minus_early_gen_prompt"], 4),
            _fmt(row["delta_recent_minus_early_gen_strict"], 4),
            _fmt(row["delta_recent_minus_early_loc"], 4),
        )
        for row in recent_early
    ]
    sections.extend(["", "### 7.5 Recent−early recency effect", "", _markdown_table(("arm", "t", "early n", "recent n", "ΔEff", "ΔGen prompt", "ΔGen strict", "ΔLoc"), recency_table)])
    sections.extend(
        [
            "",
            "## 8. Performance-definition separation and online→cumulative forgetting gap",
            "",
            "- **Final-W full10k:** W₁₀₀₀₀이 모든 10k를 현재 얼마나 기억하는가.",
            "- **Checkpoint cumulative seen-prefix:** W_t가 first t 전체를 현재 얼마나 기억하는가.",
            "- **Online-at-write:** 각 request가 자기 B100 write 직후 보인 성능을 first t까지 모은 것.",
            "- **Current B100:** checkpoint의 최신 100개만 평가한 local diagnostic.",
            "- **Fixed sentinel:** 시간축에서 동일 probe의 mechanism 변화를 본 것. 서로 대체하지 않는다.",
            "",
            "동일 first-t rewrite request/order에서만 cumulative−online forgetting gap을 계산했다. Online raw에는 Gen/Loc가 없어 `NOT_RECORDED_ONLINE_SCHEMA`다.",
        ]
    )
    for arm in ARM_ORDER:
        rows = [row for row in forgetting if row["arm"] == arm]
        table = [
            (
                row["accepted_edit_count"], CATEGORY_LABEL[row["category"]],
                f"{row['online_strict_numerator']}/{row['request_denominator']}→{row['cumulative_strict_numerator']}/{row['request_denominator']}",
                _fmt(row["delta_cumulative_minus_online_strict_rate"], 4),
                "/".join(_fmt(row[f"delta_cumulative_minus_online_nll_{stat}"]) for stat in STATS),
                "/".join(_fmt(row[f"delta_cumulative_minus_online_margin_{stat}"]) for stat in STATS),
            )
            for row in rows
        ]
        sections.extend(["", f"### 8.{ARM_ORDER.index(arm)+1} {arm}", "", _markdown_table(("t", "category", "online→cumulative strict n/d", "Δstrict", "ΔNLL μ/med/p90/max", "Δmargin μ/med/p90/max"), table)])
    sections.extend(
        [
            "",
            "## 9. Mechanism↔cumulative performance descriptive association",
            "",
            "동일 exact 7 checkpoints에서 v3 sentinel mechanism과 cumulative seen-prefix 성능을 결합했다. 각 coefficient n=7/arm이며 chronological confounding이 있으므로 causal claim=0이다.",
            "",
            "![Mechanism association](mechanism-cumulative-associations.png)",
        ]
    )
    for arm in ARM_ORDER:
        rows = [row for row in associations if row["arm"] == arm]
        table = [
            (
                row["mechanism_metric"], row["outcome_metric"], row["checkpoint_n"],
                _fmt(row["spearman"], 4), _fmt(row["pearson"], 4), row["spearman_direction"],
            )
            for row in rows
        ]
        sections.extend(["", f"### 9.{ARM_ORDER.index(arm)+1} {arm}", "", _markdown_table(("mechanism", "outcome", "n", "Spearman", "Pearson", "direction"), table)])
    native_sections = (
        (10, 3, "Residual trajectory와 lifelong drift — cumulative 연결"),
        (11, 4, "Allocation–realization: A, Y, E, rho, tau"),
        (12, 5, "Inherited debt와 recurrence closure"),
        (13, 6, "Layer-wise update magnitude/share"),
        (14, 7, "Endpoint diagnostics — PRE_EDIT, current B100, online, retention"),
        (15, 8, "AlphaEdit accumulated-cache vs reset-cache mechanism fork"),
        (16, 9, "Paired model/method comparisons"),
        (17, 10, "Original v3 descriptive associations"),
        (18, 11, "Completion geometry와 failure-time boundary"),
        (19, 12, "Outlier audit"),
        (20, 13, "Compute/accounting"),
        (21, 14, "Technical exclusions"),
        (22, 15, "NOT_RECORDED와 해석 한계"),
        (23, 16, "Reproducible figures와 artifact inventory"),
    )
    for new_number, old_number, title in native_sections:
        sections.extend(["", f"## {new_number}. {title}", ""])
        if old_number == 3:
            sections.append("이 mechanism trajectory는 §4–§9 cumulative retention과 같은 checkpoint clock에 결합되지만 sentinel 100과 seen-prefix t requests는 denominator가 다르다.")
            sections.append("")
            sections.append(
                "![10k primary residual trajectory](../official-layer-realization-debt-lifelong-b100x100-2026-09-03-v3/primary-residual-trajectory-2x2.png)"
            )
            sections.append("")
        if old_number == 7:
            sections.append("**Diagnostic boundary:** 아래 v3 current-B100/online/retention 수치는 역사적 원문 그대로 보존하지만 final 또는 cumulative headline으로 사용하지 않는다.")
            sections.append("")
        sections.append(_v3_section_body(v3, old_number, new_number))
    sections.extend(
        [
            "",
            "## 24. FACT / INFERENCE / DECISION",
            "",
            "- **FACT:** four-arm final W₁₀₀₀₀ full-10k 성능과 28 checkpoint cumulative rows는 sealed v4-r2 receipt에서 재해시했으며, v3 mechanism tables 48 members도 독립 재해시했다.",
            "- **FACT:** LM은 비단조 collapse/recovery를, LA는 3k 이후 특히 5k→7.5k 급락을, QM은 3k까지 급붕괴 후 5k 이후 거의 0을, QA는 3k 이후 단계적 붕괴를 보였다.",
            "- **FACT:** earliest/middle/recent 격차는 checkpoint와 arm에 따라 달랐고, final W에서 recent retention이 대체로 earliest보다 높았다. 전체 84 rows와 category tails는 §7에 있다.",
            "- **INFERENCE:** AlphaEdit가 같은 모델 MEMIT보다 장기 retention을 더 유지했지만 네 arm 공통 locality collapse와 큰 forgetting gap은 lifelong preservation 한계를 보여준다.",
            "- **INFERENCE:** mechanism correlation은 n=7과 time confounding 때문에 architecture/method-conditioned descriptive association으로만 해석한다.",
            "- **DECISION:** barrier usefulness, ODE necessity, universal last-layer bottleneck, scientific promotion을 주장하지 않는다. `scientific_promotion=false`; 다음 자동 experiment=0.",
            "",
            "## 25. v5 package inventory",
            "",
            f"- New table count: {package_inventory['table_count']}; new figure count: {package_inventory['figure_count']}.",
            f"- v3 member count/root: {package_inventory['v3_member_count']} / `{package_inventory['v3_member_root']}`.",
            f"- v4-r2 member count/root: {package_inventory['v4_member_count']} / `{package_inventory['v4_member_root']}`.",
            "- Complete per-request and raw historical tables remain in sealed v3/v4 packages; v5 does not duplicate model/checkpoint/cache/raw prompt data.",
            "- Every new table row count/SHA, plot input/source/output SHA and command are bound in `analysis-manifest.json` and `rooted-analysis-receipt.json`.",
            "",
        ]
    )
    return "\n".join(sections)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise V5Boundary(f"refusing existing v5 output: {args.output}")
    head = _git(args.source_root, "rev-parse", "HEAD")
    tree = _git(args.source_root, "rev-parse", "HEAD^{tree}")
    if head != args.expected_head or _git(args.source_root, "status", "--porcelain", "--untracked-files=no"):
        raise V5Boundary("v5 analysis source identity differs")
    v3, v4 = args.source_root / V3_RELATIVE, args.source_root / V4_RELATIVE
    v3_manifest, v3_receipt, v3_members = _verify_package(
        v3,
        report_name="official-layer-realization-debt-lifelong-fourarm-exhaustive-factual-ko.md",
        report_sha=V3_REPORT_SHA256,
        manifest_sha=V3_MANIFEST_SHA256,
        receipt_sha=V3_RECEIPT_SHA256,
    )
    v4_manifest, v4_receipt, v4_members = _verify_package(
        v4,
        report_name="official-layer-debt-lifelong-finalw-full10k-factual-ko.md",
        report_sha=V4_REPORT_SHA256,
        manifest_sha=V4_MANIFEST_SHA256,
        receipt_sha=V4_RECEIPT_SHA256,
    )
    v4_external_member_count = _verify_v4_external_members(v4, v4_manifest)
    cumulative, age, final_hashes = _validate_v4_tables(v4)
    core = _core_rows(cumulative)
    categories = _category_rows(v4, cumulative)
    core_transitions = _core_transition_rows(core)
    category_transitions = _category_transition_rows(categories)
    paired_core, paired_category = _paired_rows(core, categories)
    recent_early = _recent_minus_early(age)
    forgetting = _online_forgetting_gap(v3, cumulative, final_hashes)
    mechanism = _mechanism_rows(v3, cumulative)
    associations = _association_rows(mechanism)
    if not (
        len(core) == 28
        and len(categories) == 140
        and len(core_transitions) == 168
        and len(category_transitions) == 140
        and len(paired_core) == 168
        and len(paired_category) == 140
        and len(age) == 84
        and len(recent_early) == 28
        and len(forgetting) == 56
        and len(mechanism) == 28
        and len(associations) == 264
    ):
        raise V5Boundary("v5 derived denominator differs")
    args.output.mkdir(parents=True, mode=0o755)
    tables: dict[str, dict[str, Any]] = {}
    table_values = (
        ("cumulative-core-rates.csv", core),
        ("cumulative-category-distributions.csv", categories),
        ("cumulative-core-transitions.csv", core_transitions),
        ("cumulative-category-transitions.csv", category_transitions),
        ("cumulative-paired-core.csv", paired_core),
        ("cumulative-paired-category.csv", paired_category),
        ("cumulative-age-strata-full.csv", age),
        ("cumulative-age-recent-minus-early.csv", recent_early),
        ("online-cumulative-forgetting-gap.csv", forgetting),
        ("mechanism-cumulative-checkpoints.csv", mechanism),
        ("mechanism-cumulative-associations.csv", associations),
    )
    for name, values in table_values:
        tables[name] = _write_csv(args.output / name, values)
    final_rows = [row for row in cumulative if int(row["accepted_edit_count"]) == 10_000]
    tables["finalw-full10k-arm-summary.csv"] = _write_csv(
        args.output / "finalw-full10k-arm-summary.csv", final_rows
    )
    figure_paths = generate_figures(args.output, args.output)
    plot_source = Path(__file__).with_name("lifelong_v5_figures.py")
    plot_inputs = {
        "finalw-full10k-performance.png": "finalw-full10k-arm-summary.csv",
        "cumulative-performance-trajectories.png": "cumulative-core-rates.csv",
        "cumulative-target-nll-trajectories.png": "cumulative-category-distributions.csv",
        "cumulative-target-margin-trajectories.png": "cumulative-category-distributions.csv",
        "cumulative-preference-trajectories.png": "cumulative-core-rates.csv",
        "checkpoint-age-strata-performance.png": "cumulative-age-strata-full.csv",
        "mechanism-cumulative-associations.png": "mechanism-cumulative-associations.csv",
    }
    figure_rows = []
    for path in figure_paths:
        input_name = plot_inputs[path.name]
        figure_rows.append(
            {
                "figure": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "plot_source": str(plot_source.relative_to(args.source_root)),
                "plot_source_sha256": sha256_file(plot_source),
                "input_table": input_name,
                "input_table_sha256": sha256_file(args.output / input_name),
                "command": (
                    "MPLCONFIGDIR=/tmp/odeedit-lifelong-v5-mpl python3 -m "
                    "project.run_scripts.official_layer_realization_debt.lifelong_v5_figures "
                    "--tables V5_PACKAGE --output CREATE_ONCE_OUTPUT"
                ),
                "backend": "Agg",
                "dpi": 180,
                "missing_policy": "NO_INTERPOLATION_NO_IMPUTATION",
            }
        )
    tables["plot-reproduction.csv"] = _write_csv(args.output / "plot-reproduction.csv", figure_rows)
    inventory = {
        "table_count": len(table_values) + 2,
        "figure_count": len(figure_paths),
        "v3_member_count": len(v3_members),
        "v3_member_root": v3_manifest["member_root"],
        "v4_member_count": len(v4_members),
        "v4_member_root": v4_manifest["member_root"],
        "v4_external_member_count": v4_external_member_count,
    }
    report_text = _report(
        v3=v3,
        v4=v4,
        source_head=head,
        source_tree=tree,
        core=core,
        categories=categories,
        core_transitions=core_transitions,
        category_transitions=category_transitions,
        paired_core=paired_core,
        paired_category=paired_category,
        age=age,
        recent_early=recent_early,
        forgetting=forgetting,
        mechanism=mechanism,
        associations=associations,
        package_inventory=inventory,
    )
    report_path = args.output / REPORT_NAME
    report_path.write_text(report_text, encoding="utf-8")
    members = []
    for path in sorted(args.output.iterdir(), key=lambda value: value.name):
        if path.name in {"analysis-manifest.json", "rooted-analysis-receipt.json"}:
            continue
        members.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    external_packages = [
        {
            "kind": "IMMUTABLE_V3_NATIVE_EXHAUSTIVE",
            "root": str(V3_RELATIVE),
            "report_sha256": V3_REPORT_SHA256,
            "manifest_sha256": V3_MANIFEST_SHA256,
            "receipt_sha256": V3_RECEIPT_SHA256,
            "member_root": v3_manifest["member_root"],
            "member_count": len(v3_members),
        },
        {
            "kind": "IMMUTABLE_V4_R2_CUMULATIVE_EVALUATION",
            "root": str(V4_RELATIVE),
            "report_sha256": V4_REPORT_SHA256,
            "manifest_sha256": V4_MANIFEST_SHA256,
            "receipt_sha256": V4_RECEIPT_SHA256,
            "member_root": v4_manifest["member_root"],
            "member_count": len(v4_members),
            "external_raw_member_root": v4_manifest["raw_member_root"],
        },
    ]
    manifest: dict[str, Any] = {
        "schema": "ode-edit-s06-lifelong-v5-exhaustive-cumulative-analysis/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "TASK_COMPLETE_STOP",
        "source": {"head": head, "tree": tree, "tracked_clean_before_output": True},
        "external_packages": external_packages,
        "external_package_root": canonical_hash(external_packages),
        "denominators": {
            "arms": 4,
            "cumulative_checkpoints": 28,
            "request_states": 120_000,
            "category_rows": 140,
            "age_rows": 84,
            "mechanism_checkpoint_rows": 28,
            "mechanism_association_rows": 264,
        },
        "table_row_counts": {name: int(value["rows"]) for name, value in tables.items()},
        "plot_reproduction": figure_rows,
        "members": members,
        "member_root": canonical_hash(members),
        "raw_model_checkpoint_cache_tensor_log_dataset_credential_git_count": 0,
        "raw_prompt_logit_generation_publish_count": 0,
        "codex_visualization_imagegen_manual_edit_count": 0,
        "imputation_count": 0,
        "scientific_promotion": False,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_path = args.output / "analysis-manifest.json"
    _write_json(manifest_path, manifest)
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s06-lifelong-v5-exhaustive-cumulative-rooted-receipt/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": manifest["status"],
        "manifest": {
            "path": "analysis-manifest.json",
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "identity_sha256": manifest["identity_sha256"],
        },
        "report": {
            "path": REPORT_NAME,
            "bytes": report_path.stat().st_size,
            "sha256": sha256_file(report_path),
        },
        "member_root": manifest["member_root"],
        "external_package_root": manifest["external_package_root"],
        "scientific_promotion": False,
        "next": "STOP",
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    receipt_path = args.output / "rooted-analysis-receipt.json"
    _write_json(receipt_path, receipt)
    return {
        "status": manifest["status"],
        "report": {"path": str(report_path), "sha256": receipt["report"]["sha256"]},
        "manifest": {"path": str(manifest_path), "sha256": receipt["manifest"]["sha256"]},
        "receipt": {"path": str(receipt_path), "sha256": sha256_file(receipt_path)},
        "member_root": manifest["member_root"],
        "tables": len(tables),
        "figures": len(figure_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
