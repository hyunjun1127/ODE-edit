"""Build the baseline-inclusive successor to the immutable round-0 report.

The original v1 package called token-level prediction preservation ``NS``.
This builder leaves v1 byte-for-byte immutable, labels that observation ``PP``,
and joins a W0-only canonical CounterFact NS backfill with denominator 1,000.
It is analysis-only: no model, editor, writer, GPU, or Slurm submission code is
imported or invoked.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd

from .round0_analysis_contracts import (
    AnalysisBoundary,
    canonical_hash,
    member,
    regular_file,
    sha256_file,
    verify_canonical_identity,
    write_json_once,
)
from .round0_package_verify import verify as verify_legacy_package


INSTRUCTION_ID = "ODEEDIT-S06-ORRBODE-ROUND0-BASELINE-INCLUSIVE-PREEDIT-NS-20260904-V2"
STREAM_ROOT = "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
ORDER_ROOT = "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"
MODEL_ORDER = ("llama3-8b-inst", "qwen2.5-7b-inst")
FAMILY_ORDER = ("MEMIT", "AlphaEdit")
ARM_ORDER = ("O", "QCL", "NQFIX", "ORBFH", "JAC")
SOURCE_FILES = (
    "project/run_scripts/ordered_response_barrier_ode/round0_baseline_report.py",
    "project/run_scripts/ordered_response_barrier_ode/round0_baseline_figures.py",
    "project/run_scripts/ordered_response_barrier_ode/tests/test_round0_baseline_report.py",
    "project/run_scripts/ordered_response_barrier_ode/preedit_ns_backfill.py",
    "project/run_scripts/ordered_response_barrier_ode/counterfact_locality_evaluator.py",
    "project/run_scripts/ordered_response_barrier_ode/round0_analysis_contracts.py",
    "project/run_scripts/ordered_response_barrier_ode/round0_package_verify.py",
)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise AnalysisBoundary(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _object(path: Path) -> dict[str, Any]:
    regular_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AnalysisBoundary(f"JSON object expected: {path}")
    return value


def _stat(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise AnalysisBoundary("finite non-empty distribution expected")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def _assert_stats(actual: Mapping[str, Any], expected: Mapping[str, float]) -> None:
    for field in ("mean", "median", "p90", "max"):
        if not math.isclose(float(actual[field]), expected[field], rel_tol=0.0, abs_tol=1e-12):
            raise AnalysisBoundary(f"backfill statistic differs: {field}")


def _read_backfill_task(path: Path, expected_model: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path.is_symlink() or not path.is_dir():
        raise AnalysisBoundary(f"backfill task root differs: {path}")
    for name in ("result.json", "manifest.json", "rooted-receipt.json", "preedit-canonical-ns-prompt-pairs.jsonl.gz"):
        info = regular_file(path / name)
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise AnalysisBoundary(f"backfill member mode differs: {path / name}")
    result, manifest, receipt = (_object(path / name) for name in ("result.json", "manifest.json", "rooted-receipt.json"))
    for payload in (result, manifest, receipt):
        verify_canonical_identity(payload)
    if result.get("status") != "TERMINAL_VALID" or result.get("model_alias") != expected_model:
        raise AnalysisBoundary("backfill terminal/model binding differs")
    if result.get("stream_root") != STREAM_ROOT or result.get("order_root") != ORDER_ROOT:
        raise AnalysisBoundary("backfill stream/order differs")
    if receipt.get("status") != "TERMINAL_VALID" or receipt.get("editing_action_count") != 0:
        raise AnalysisBoundary("backfill action/terminal receipt differs")
    if receipt.get("w0_pointer_version_bytes_unchanged") is not True:
        raise AnalysisBoundary("backfill W0 invariant differs")
    result_path = path / "result.json"
    rows_path = path / "preedit-canonical-ns-prompt-pairs.jsonl.gz"
    if manifest.get("result_sha256") != sha256_file(result_path) or manifest.get("rows_sha256") != sha256_file(rows_path):
        raise AnalysisBoundary("backfill manifest member SHA differs")
    if receipt.get("manifest_sha256") != sha256_file(path / "manifest.json"):
        raise AnalysisBoundary("backfill receipt manifest SHA differs")
    with gzip.open(rows_path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if len(rows) != 1000:
        raise AnalysisBoundary("backfill canonical NS denominator differs")
    keys: list[tuple[int, int]] = []
    for row in rows:
        verify_canonical_identity(row)
        keys.append((int(row["request_ordinal"]), int(row["prompt_index"])))
        new_nll, true_nll = float(row["target_new_nll"]), float(row["target_true_nll"])
        if not math.isfinite(new_nll) or not math.isfinite(true_nll):
            raise AnalysisBoundary("backfill NLL is nonfinite")
        if int(row["ns_success"]) != int(true_nll < new_nll):
            raise AnalysisBoundary("canonical NS strict predicate differs")
        if int(row["nll_tie"]) != int(true_nll == new_nll):
            raise AnalysisBoundary("canonical NS tie policy differs")
    if keys != sorted(keys) or len(set(keys)) != 1000:
        raise AnalysisBoundary("backfill row order/uniqueness differs")
    summary = result["summary"]
    if summary.get("definition") != "target_true_nll < target_new_nll; ties fail":
        raise AnalysisBoundary("canonical NS definition differs")
    if int(summary["ns_denominator"]) != 1000 or int(summary["prompt_pair_count"]) != 1000:
        raise AnalysisBoundary("canonical NS prompt-pair denominator differs")
    if int(summary["ns_numerator"]) != sum(int(row["ns_success"]) for row in rows):
        raise AnalysisBoundary("canonical NS numerator differs")
    if summary.get("row_identity_root") != canonical_hash([row["identity_sha256"] for row in rows]):
        raise AnalysisBoundary("backfill row identity root differs")
    _assert_stats(summary["target_new_nll"], _stat([float(row["target_new_nll"]) for row in rows]))
    _assert_stats(summary["target_true_nll"], _stat([float(row["target_true_nll"]) for row in rows]))
    _assert_stats(
        summary["nll_advantage_new_minus_true"],
        _stat([float(row["nll_advantage_new_minus_true"]) for row in rows]),
    )
    return result, rows


def load_backfill(root: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    results: dict[str, dict[str, Any]] = {}
    combined: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for task, model in enumerate(MODEL_ORDER):
        task_root = root / f"task-{task}"
        result, rows = _read_backfill_task(task_root, model)
        results[model] = result
        for row in rows:
            combined.append({"model": model, **row})
        for path in sorted(task_root.iterdir()):
            inputs.append(member(path, kind="preedit_ns_backfill"))
    if len(combined) != 2000:
        raise AnalysisBoundary("two-model canonical NS denominator differs")
    return results, combined, inputs


def _same_preedit(rows: pd.DataFrame) -> pd.Series:
    ignored = {"cell_id", "writer_family"}
    columns = [column for column in rows.columns if column not in ignored]
    if len(rows) != 2:
        raise AnalysisBoundary("PRE_EDIT writer-family duplicate count differs")
    first, second = rows.iloc[0], rows.iloc[1]
    for column in columns:
        left, right = first[column], second[column]
        if pd.isna(left) and pd.isna(right):
            continue
        if left != right:
            raise AnalysisBoundary(f"PRE_EDIT duplicate differs: {column}")
    return first


def build_performance(core: pd.DataFrame, requests: pd.DataFrame, backfill: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        preedit = _same_preedit(core[(core.model == model) & (core.stage == "PRE_EDIT")].sort_values("writer_family"))
        ns = backfill[model]["summary"]
        rows.append(
            {
                "model": model,
                "method_group": "W0_BASELINE",
                "method_label": "PRE_EDIT (W0 common)",
                "writer_family": "NOT_APPLICABLE",
                "arm": "PRE_EDIT",
                "stage_status": str(preedit.stage_status),
                "rewrite_success_numerator": int(preedit.rewrite_success_count),
                "rewrite_success_denominator": int(preedit.rewrite_success_denominator),
                "rewrite_success_rate": float(preedit.rewrite_success_rate),
                "rephrase_success_numerator": int(preedit.rephrase_success_count),
                "rephrase_success_denominator": int(preedit.rephrase_success_denominator),
                "rephrase_success_rate": float(preedit.rephrase_success_rate),
                "strict_rephrase_success_numerator": int(preedit.strict_rephrase_success_count),
                "strict_rephrase_success_denominator": int(preedit.strict_rephrase_success_denominator),
                "strict_rephrase_success_rate": float(preedit.strict_rephrase_success_rate),
                "canonical_ns_status": "RECORDED_STRICT_COUNTERFACT_NLL_PAIR",
                "canonical_ns_numerator": int(ns["ns_numerator"]),
                "canonical_ns_denominator": int(ns["ns_denominator"]),
                "canonical_ns_rate": float(ns["ns_rate"]),
                "pp_prompt_status": "NOT_APPLICABLE_ENTRY_REFERENCE",
                "pp_prompt_numerator": np.nan,
                "pp_prompt_denominator": np.nan,
                "pp_prompt_rate": np.nan,
                "pp_token_status": "NOT_APPLICABLE_ENTRY_REFERENCE",
                "pp_token_numerator": np.nan,
                "pp_token_denominator": np.nan,
                "pp_token_rate": np.nan,
                "rewrite_target_new_accuracy_rate": float(preedit.rewrite_target_new_accuracy_rate),
                "rewrite_target_true_accuracy_rate": float(preedit.rewrite_target_true_accuracy_rate),
                "rephrase_target_new_accuracy_rate": float(preedit.rephrase_target_new_accuracy_rate),
                "rephrase_target_true_accuracy_rate": float(preedit.rephrase_target_true_accuracy_rate),
                "locality_target_true_accuracy_rate": float(preedit.locality_target_true_accuracy_rate),
            }
        )
        for family in FAMILY_ORDER:
            for arm in ARM_ORDER:
                value = core[(core.model == model) & (core.writer_family == family) & (core.stage == arm)]
                if len(value) != 1:
                    raise AnalysisBoundary("endpoint performance row is not unique")
                row = value.iloc[0]
                per_request = requests[
                    (requests.model == model) & (requests.writer_family == family) & (requests.stage == arm)
                ]
                if len(per_request) != 100:
                    raise AnalysisBoundary("endpoint request denominator differs")
                pp_prompt_num = int(per_request.locality_all_prompt_preserved_count.sum())
                pp_prompt_den = int(per_request.locality_prompt_denominator.sum())
                pp_token_num = int(per_request.locality_prediction_preservation_numerator.sum())
                pp_token_den = int(per_request.locality_prediction_preservation_denominator.sum())
                if pp_prompt_den != 1000 or pp_token_den != 1010:
                    raise AnalysisBoundary("endpoint PP denominator decomposition differs")
                label = f"Official {family}" if arm == "O" else f"{family}+{arm}"
                rows.append(
                    {
                        "model": model,
                        "method_group": "OFFICIAL_BASELINE" if arm == "O" else "OURS",
                        "method_label": label,
                        "writer_family": family,
                        "arm": arm,
                        "stage_status": str(row.stage_status),
                        "rewrite_success_numerator": int(row.rewrite_success_count),
                        "rewrite_success_denominator": int(row.rewrite_success_denominator),
                        "rewrite_success_rate": float(row.rewrite_success_rate),
                        "rephrase_success_numerator": int(row.rephrase_success_count),
                        "rephrase_success_denominator": int(row.rephrase_success_denominator),
                        "rephrase_success_rate": float(row.rephrase_success_rate),
                        "strict_rephrase_success_numerator": int(row.strict_rephrase_success_count),
                        "strict_rephrase_success_denominator": int(row.strict_rephrase_success_denominator),
                        "strict_rephrase_success_rate": float(row.strict_rephrase_success_rate),
                        "canonical_ns_status": "NOT_RECORDED_SCHEMA_GAP_LOCALITY_TARGET_NEW_NLL",
                        "canonical_ns_numerator": np.nan,
                        "canonical_ns_denominator": np.nan,
                        "canonical_ns_rate": np.nan,
                        "pp_prompt_status": "RECORDED_ALL_TARGET_TRUE_TOKENS_PRESERVED_PER_PROMPT",
                        "pp_prompt_numerator": pp_prompt_num,
                        "pp_prompt_denominator": pp_prompt_den,
                        "pp_prompt_rate": pp_prompt_num / pp_prompt_den,
                        "pp_token_status": "RECORDED_TARGET_TRUE_TOKEN_PREDICTION_PRESERVATION",
                        "pp_token_numerator": pp_token_num,
                        "pp_token_denominator": pp_token_den,
                        "pp_token_rate": pp_token_num / pp_token_den,
                        "rewrite_target_new_accuracy_rate": float(row.rewrite_target_new_accuracy_rate),
                        "rewrite_target_true_accuracy_rate": float(row.rewrite_target_true_accuracy_rate),
                        "rephrase_target_new_accuracy_rate": float(row.rephrase_target_new_accuracy_rate),
                        "rephrase_target_true_accuracy_rate": float(row.rephrase_target_true_accuracy_rate),
                        "locality_target_true_accuracy_rate": float(row.locality_target_true_accuracy_rate),
                    }
                )
    frame = pd.DataFrame(rows)
    if len(frame) != 22 or frame.canonical_ns_denominator.notna().sum() != 2:
        raise AnalysisBoundary("baseline-inclusive performance denominator differs")
    return frame


def build_preedit_nll(nll: pd.DataFrame, backfill: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        selected = nll[(nll.model == model) & (nll.stage == "PRE_EDIT")]
        for kind in ("rewrite_target_new", "rewrite_target_true", "rephrase_target_new", "rephrase_target_true"):
            values = selected[selected.kind == kind]
            if len(values) != 2:
                raise AnalysisBoundary("PRE_EDIT NLL duplicate denominator differs")
            fields = ("row_count", "nll_mean", "nll_median", "nll_p90", "nll_max", "all_tokens_correct_count")
            if any(values[field].nunique(dropna=False) != 1 for field in fields):
                raise AnalysisBoundary(f"PRE_EDIT NLL duplicate differs: {model}/{kind}")
            value = values.iloc[0]
            family, target = kind.split("_target_")
            rows.append(
                {
                    "model": model,
                    "prompt_family": family,
                    "quantity": f"target_{target}_nll",
                    "row_count": int(value.row_count),
                    "mean": float(value.nll_mean),
                    "median": float(value.nll_median),
                    "p90": float(value.nll_p90),
                    "max": float(value.nll_max),
                }
            )
        summary = backfill[model]["summary"]
        for quantity, field in (
            ("target_new_nll", "target_new_nll"),
            ("target_true_nll", "target_true_nll"),
            ("new_minus_true_nll_advantage", "nll_advantage_new_minus_true"),
        ):
            value = summary[field]
            rows.append(
                {
                    "model": model,
                    "prompt_family": "locality",
                    "quantity": quantity,
                    "row_count": 1000,
                    "mean": float(value["mean"]),
                    "median": float(value["median"]),
                    "p90": float(value["p90"]),
                    "max": float(value["max"]),
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != 14:
        raise AnalysisBoundary("PRE_EDIT NLL output row count differs")
    return frame


def build_official_deltas(performance: pd.DataFrame, nll: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        pre = performance[(performance.model == model) & (performance.arm == "PRE_EDIT")].iloc[0]
        for family in FAMILY_ORDER:
            endpoint = performance[
                (performance.model == model) & (performance.writer_family == family) & (performance.arm == "O")
            ].iloc[0]
            for metric in ("rewrite_success_rate", "rephrase_success_rate", "strict_rephrase_success_rate"):
                rows.append(
                    {
                        "model": model,
                        "official_baseline": f"Official {family}",
                        "metric": metric,
                        "pre_edit": float(pre[metric]),
                        "official_endpoint": float(endpoint[metric]),
                        "official_minus_pre_edit": float(endpoint[metric] - pre[metric]),
                        "unit": "RATE",
                    }
                )
            for kind in ("rewrite_target_new", "rewrite_target_true", "rephrase_target_new", "rephrase_target_true"):
                entry = nll[(nll.model == model) & (nll.stage == "PRE_EDIT") & (nll.kind == kind)].iloc[0]
                final = nll[
                    (nll.model == model) & (nll.writer_family == family) & (nll.stage == "O") & (nll.kind == kind)
                ].iloc[0]
                for statistic in ("mean", "median", "p90", "max"):
                    field = f"nll_{statistic}"
                    rows.append(
                        {
                            "model": model,
                            "official_baseline": f"Official {family}",
                            "metric": f"{kind}_nll_{statistic}",
                            "pre_edit": float(entry[field]),
                            "official_endpoint": float(final[field]),
                            "official_minus_pre_edit": float(final[field] - entry[field]),
                            "unit": "NLL",
                        }
                    )
    frame = pd.DataFrame(rows)
    if len(frame) != 76:
        raise AnalysisBoundary("Official-vs-PRE_EDIT delta row count differs")
    return frame


def build_metric_correction(prompt: pd.DataFrame) -> pd.DataFrame:
    selected = prompt[
        (prompt.cell_id == 0) & (prompt.stage == "PRE_EDIT") & (prompt.kind == "locality_target_true")
    ]
    distribution = selected.target_token_count.value_counts().sort_index().to_dict()
    if len(selected) != 1000 or distribution != {1: 990, 2: 10} or int(selected.target_token_count.sum()) != 1010:
        raise AnalysisBoundary("PP 1,010-token denominator explanation differs")
    multi = selected[selected.target_token_count == 2]
    if multi.case_id.nunique() != 1 or int(multi.case_id.iloc[0]) != 9936 or multi.prompt_index.nunique() != 10:
        raise AnalysisBoundary("PP multi-token request identity differs")
    return pd.DataFrame(
        [
            {
                "metric": "canonical_ns",
                "unit": "locality_prompt_pair",
                "denominator_per_model_or_endpoint": 1000,
                "definition": "target_true_nll < target_new_nll; ties fail",
                "availability": "PRE_EDIT_RECORDED; ENDPOINT_NOT_RECORDED_SCHEMA_GAP",
            },
            {
                "metric": "pp_prompt",
                "unit": "locality_prompt",
                "denominator_per_model_or_endpoint": 1000,
                "definition": "all target-true token predictions equal PRE_EDIT for a prompt",
                "availability": "ENDPOINT_RECORDED; PRE_EDIT_NOT_APPLICABLE",
            },
            {
                "metric": "pp_token",
                "unit": "target_true_token",
                "denominator_per_model_or_endpoint": 1010,
                "definition": "individual target-true token predictions equal PRE_EDIT",
                "availability": "ENDPOINT_RECORDED; 990x1-token + 10x2-token prompts",
            },
        ]
    )


def _write_csv_once(path: Path, frame: pd.DataFrame, *, compressed: bool = False) -> None:
    if path.exists() or path.is_symlink():
        raise AnalysisBoundary(f"create-once CSV boundary differs: {path}")
    text = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    raw = text
    if compressed:
        buffer = io.BytesIO()
        with gzip.GzipFile(filename="", fileobj=buffer, mode="wb", mtime=0, compresslevel=9) as handle:
            handle.write(text)
        raw = buffer.getvalue()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)


def _rate(numerator: Any, denominator: Any, rate: Any) -> str:
    if pd.isna(numerator) or pd.isna(denominator) or pd.isna(rate):
        return "N/A"
    return f"{int(numerator)}/{int(denominator)} ({100.0 * float(rate):.2f}%)"


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(clean(value) for value in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _legacy_detail(text: str) -> str:
    marker = "## 3. Rewrite NLL 분포"
    if marker not in text:
        raise AnalysisBoundary("legacy detailed report section boundary differs")
    detail = marker + text.split(marker, 1)[1]
    headings = list(re.finditer(r"^## (\d+)\. (.+)$", detail, flags=re.MULTILINE))
    if [int(match.group(1)) for match in headings] != list(range(3, 16)):
        raise AnalysisBoundary("legacy report heading sequence differs")
    detail = re.sub(
        r"^## (\d+)\. (.+)$",
        lambda match: f"### 6.{int(match.group(1)) - 2} {match.group(2)}",
        detail,
        flags=re.MULTILINE,
    )
    # Every uppercase NS in the inherited v1 body referred to prediction
    # preservation, not the canonical CounterFact NLL-pair NS definition.
    detail = detail.replace("NS", "PP")
    detail = detail.replace(
        "`primary-rs-ps-ns.png`: PRE_EDIT/O/QCL/NQFIX/ORBFH/JAC의 primary RS/PS/PP.",
        "`primary-rs-ps-ns.png` (v1 legacy filename): PRE_EDIT/O/QCL/NQFIX/ORBFH/JAC의 RS/PS/token-PP.",
    )
    return detail


def _report(
    performance: pd.DataFrame,
    preedit_nll: pd.DataFrame,
    official_deltas: pd.DataFrame,
    correction: pd.DataFrame,
    backfill: Mapping[str, Mapping[str, Any]],
    legacy_text: str,
    legacy_identity: Mapping[str, Any],
    backfill_inputs: Sequence[Mapping[str, Any]],
) -> str:
    primary_rows = []
    accuracy_rows = []
    for row in performance.itertuples(index=False):
        primary_rows.append(
            (
                row.model,
                row.method_label,
                row.stage_status,
                _rate(row.rewrite_success_numerator, row.rewrite_success_denominator, row.rewrite_success_rate),
                _rate(row.rephrase_success_numerator, row.rephrase_success_denominator, row.rephrase_success_rate),
                _rate(
                    row.strict_rephrase_success_numerator,
                    row.strict_rephrase_success_denominator,
                    row.strict_rephrase_success_rate,
                ),
                _rate(row.canonical_ns_numerator, row.canonical_ns_denominator, row.canonical_ns_rate)
                if row.arm == "PRE_EDIT"
                else "NOT_RECORDED_SCHEMA_GAP",
                _rate(row.pp_prompt_numerator, row.pp_prompt_denominator, row.pp_prompt_rate),
                _rate(row.pp_token_numerator, row.pp_token_denominator, row.pp_token_rate),
            )
        )
        accuracy_rows.append(
            (
                row.model,
                row.method_label,
                f"{100.0 * row.rewrite_target_new_accuracy_rate:.2f}%",
                f"{100.0 * row.rewrite_target_true_accuracy_rate:.2f}%",
                f"{100.0 * row.rephrase_target_new_accuracy_rate:.2f}%",
                f"{100.0 * row.rephrase_target_true_accuracy_rate:.2f}%",
                f"{100.0 * row.locality_target_true_accuracy_rate:.2f}%",
            )
        )
    nll_rows = [
        (
            row.model,
            row.prompt_family,
            row.quantity,
            int(row.row_count),
            f"{row.mean:.6f}",
            f"{row.median:.6f}",
            f"{row.p90:.6f}",
            f"{row.max:.6f}",
        )
        for row in preedit_nll.itertuples(index=False)
    ]
    rate_delta = official_deltas[official_deltas.unit == "RATE"]
    delta_rows = [
        (
            row.model,
            row.official_baseline,
            row.metric,
            f"{100.0 * row.pre_edit:.2f}%",
            f"{100.0 * row.official_endpoint:.2f}%",
            f"{100.0 * row.official_minus_pre_edit:+.2f} pp",
        )
        for row in rate_delta.itertuples(index=False)
    ]
    correction_rows = [tuple(row) for row in correction.itertuples(index=False, name=None)]
    backfill_rows = []
    for task, model in enumerate(MODEL_ORDER):
        summary = backfill[model]["summary"]
        backfill_rows.append(
            (
                model,
                "36275_0→36276" if task == 0 else "36275_1→36275",
                _rate(summary["ns_numerator"], summary["ns_denominator"], summary["ns_rate"]),
                summary["nll_tie_count"],
                summary["row_identity_root"],
                summary["identity_sha256"],
            )
        )
    source_rows = [
        (row["kind"], row["path"], row["bytes"], row["mode"], row["sha256"])
        for row in backfill_inputs
    ]
    legacy_detail = _legacy_detail(legacy_text)
    return "\n".join(
        [
            "# Ordered Response-Barrier ODE-Edit — Server1 round0 B100 baseline-inclusive 상세 사실 보고서",
            "",
            "> 이 문서는 immutable v1 상세 분석의 successor다. v1 raw/표/보고서는 수정하지 않았으며, "
            "Official baseline 이름을 풀어 쓰고 W0-only canonical CounterFact NS backfill을 추가했다.",
            "",
            "## 0. 가장 중요한 정정과 판정",
            "",
            "- canonical round0 과학 분모는 그대로 `4 cells × 5 primary arms × 100 requests = 2,000 endpoints`다.",
            "- `O`는 별도 ours arm이 아니라 각 family의 stock baseline이다: `Official MEMIT` 또는 `Official AlphaEdit`.",
            "- PRE_EDIT/W0 canonical NS는 새 observation-only TECH-R2 job `36275_[0-1]`에서 기록했다: "
            "Llama `893/1000 (89.30%)`, Qwen `849/1000 (84.90%)`.",
            "- 기존 v1 표의 `NS`는 canonical NS가 아니었다. 그것은 PRE_EDIT 대비 target-true token prediction "
            "preservation이며 이 문서에서는 `PP-token`으로 교정한다.",
            "- `1010`은 오류 난 prompt 분모가 아니라 token 분모다. 1,000 locality prompt 중 990개는 target-true "
            "1-token, case `9936`(request ordinal 61)의 10개 prompt는 2-token이라 `990×1 + 10×2 = 1010`이다.",
            "- prompt 단위 all-token prediction preservation은 `PP-prompt`로 따로 산출했으며 분모가 정확히 `1000`이다.",
            "- 원 round0 endpoint에는 `locality_target_new` NLL이 없으므로 endpoint canonical NS는 "
            "`NOT_RECORDED_SCHEMA_GAP`; PP나 target-true accuracy로 대체·추정하지 않았다.",
            "- PRE_EDIT backfill은 model load/evaluator 외 editor, compute_z, key, solve, writer, model update, "
            "cache/history mutation이 모두 0이고 W0 pointer/version/bytes가 동일하다.",
            "- 결론은 서술적 사실이며 `scientific_promotion=false`; remaining-nine 제출은 0이다.",
            "",
            "## 1. 지표 정의와 분모",
            "",
            _markdown_table(
                ("metric", "unit", "denominator", "definition", "availability"), correction_rows
            ),
            "",
            "- `RS`: rewrite 100 pair에서 `target-new NLL < target-true NLL`; tie failure.",
            "- `PS`: rephrase 200 prompt pair에서 같은 strict 비교. `strict PS`는 request별 두 rephrase가 모두 성공한 100-request 분모다.",
            "- `canonical NS`: locality 1,000 prompt pair에서 `target-true NLL < target-new NLL`; tie failure.",
            "- `PP-prompt`: 각 locality prompt의 target-true token prediction이 모두 PRE_EDIT와 동일한지 보는 1,000-prompt 분모다.",
            "- `PP-token`: 각 target-true token prediction 보존 여부를 세는 1,010-token 분모다.",
            "- `locality target-true accuracy`는 target-true를 정확히 예측한 prompt 비율이며 canonical NS와 다르다.",
            "",
            "## 2. PRE_EDIT와 Official baseline을 명시한 핵심 성능표",
            "",
            _markdown_table(
                ("model", "method", "status", "RS", "PS", "strict PS", "canonical NS", "PP-prompt", "PP-token"),
                primary_rows,
            ),
            "",
            "`HORIZON_SEMANTIC_MISS`는 100/100 동시 strict global prefix가 없었다는 과학 관측이다. "
            "endpoint 자체는 terminal-valid하며 낮은 성능이나 Official 차이를 기술 실패로 재분류하지 않았다.",
            "",
            "### 2.1 Secondary target-token accuracy",
            "",
            _markdown_table(
                ("model", "method", "RW new", "RW true", "RP new", "RP true", "LOC true"), accuracy_rows
            ),
            "",
            "## 3. PRE_EDIT/W0 canonical NS backfill",
            "",
            _markdown_table(
                ("model", "scheduler", "canonical NS", "ties", "row identity root", "summary identity"),
                backfill_rows,
            ),
            "",
            "TECH-R1 `36273_[0-1]`은 sealed request row에 존재하지 않는 `round_ordinal` 필드를 읽은 "
            "`KeyError` 기술 실패라 분모 0으로 보존했다. science/threshold/tolerance 변경 없이 ordinal을 sealed list 순서로 "
            "bind한 TECH-R2만 canonical backfill이다.",
            "",
            "### 3.1 PRE_EDIT rewrite/rephrase/locality NLL 전체 분포",
            "",
            _markdown_table(
                ("model", "prompt", "quantity", "n", "mean", "median", "p90", "max"), nll_rows
            ),
            "",
            "locality `new-minus-true advantage`가 양수이면 target-true NLL이 더 낮아 canonical NS에 유리하다. "
            "분포는 prompt 1,000개를 직접 집계했으며 request 평균으로 축소하지 않았다.",
            "",
            "## 4. PRE_EDIT 대비 stock Official baseline 변화",
            "",
            _markdown_table(
                ("model", "baseline", "metric", "PRE_EDIT", "Official endpoint", "delta"), delta_rows
            ),
            "",
            "NLL mean/median/p90/max까지 포함한 76-row 전체 차이는 `official-vs-preedit-deltas.csv`에 있다. "
            "endpoint canonical NS가 없으므로 NS delta는 만들지 않았다. PP는 W0와 endpoint 사이의 정의 자체라 별도 endpoint 사실값으로만 둔다.",
            "",
            "## 5. Arm과 baseline 해석",
            "",
            "- `PRE_EDIT (W0 common)`: 두 writer-family cell에 저장된 entry evaluation이 모델별 byte/수치 동일한 공통 W0 관측이다.",
            "- `Official MEMIT`: stock native MEMIT entrypoint의 round0 B100 endpoint다.",
            "- `Official AlphaEdit`: stock native AlphaEdit entrypoint의 round0 B100 endpoint다.",
            "- `MEMIT/AlphaEdit + QCL/NQFIX/ORBFH/JAC`: 같은 family-native proposal을 각 controller allocation 경계로 실행한 ours endpoints다.",
            "- 모든 ours-vs-Official paired delta는 자기 writer family의 Official baseline과 동일 request/prompt로만 비교했다.",
            "",
            "## 6. Immutable v1의 상세 분석 — 용어 교정본",
            "",
            f"아래 상세 본문은 v1 report SHA `{legacy_identity['report_sha256']}`의 section 3–15를 계승한다. "
            "원 파일명과 표 bytes는 v1 package에 있고, 본문에서 uppercase `NS`였던 prediction-preservation 명칭만 `PP`로 표시했다. "
            "canonical NS로 재해석하지 않는다.",
            "",
            legacy_detail,
            "",
            "## 7. PRE_EDIT NS backfill 입력 identity",
            "",
            _markdown_table(("kind", "absolute path", "bytes", "mode", "sha256"), source_rows),
            "",
            "## 8. 최종 한계",
            "",
            "- post-edit canonical NS는 모든 endpoint에서 `NOT_RECORDED_SCHEMA_GAP`이다. 새 편집 run이나 imputation으로 채우지 않았다.",
            "- PRE_EDIT canonical NS는 동일 sealed round0 첫 B100/W0에서 얻었지만, endpoint PP와 서로 다른 정의이므로 직접 증감으로 빼지 않는다.",
            "- v1의 2,000 endpoint, mechanism, layer, compute 결과는 재실행하지 않았고 입력 SHA를 전후 재확인했다.",
            "- 이 보고서는 baseline 명시와 locality metric schema 교정 목적의 factual successor이며 자동 method 선택이나 promotion을 하지 않는다.",
            "",
        ]
    )


def _scheduler_rows() -> list[dict[str, Any]]:
    expected = {
        "36273_0": ("36274", "FAILED", "1:0"),
        "36273_1": ("36273", "FAILED", "1:0"),
        "36275_0": ("36276", "COMPLETED", "0:0"),
        "36275_1": ("36275", "COMPLETED", "0:0"),
    }
    completed = subprocess.run(
        ["sacct", "-j", "36273,36275", "-X", "--format=JobID,JobIDRaw,State,ExitCode,Elapsed", "-n", "-P"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise AnalysisBoundary(f"sacct backfill query failed: {completed.stderr.strip()}")
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        job_id, raw, state, exit_code, elapsed = line.split("|")[:5]
        if job_id not in expected:
            continue
        expected_raw, expected_state, expected_exit = expected[job_id]
        if (raw, state, exit_code) != (expected_raw, expected_state, expected_exit):
            raise AnalysisBoundary(f"backfill scheduler terminal differs: {job_id}")
        rows.append(
            {
                "job_id": job_id,
                "job_id_raw": raw,
                "state": state,
                "exit_code": exit_code,
                "elapsed": elapsed,
                "inclusion": "INCLUDED_CANONICAL_PREEDIT_NS" if state == "COMPLETED" else "EXCLUDED_TECHNICAL_DENOMINATOR_ZERO",
            }
        )
    if {row["job_id"] for row in rows} != set(expected):
        raise AnalysisBoundary("backfill scheduler row denominator differs")
    return sorted(rows, key=lambda row: row["job_id"])


def build(repo: Path, legacy: Path, backfill_root: Path, failed_root: Path, output: Path, repro_root: Path) -> dict[str, Any]:
    repo, legacy, backfill_root, failed_root, output = (
        repo.resolve(), legacy.resolve(), backfill_root.resolve(), failed_root.resolve(), output.absolute()
    )
    if output.exists() or output.is_symlink():
        raise AnalysisBoundary("successor report output must be create-once")
    legacy_verified = verify_legacy_package(repo, legacy)
    legacy_report_path = legacy / "ordered-response-barrier-ode-round0-b100-exhaustive-factual-ko.md"
    legacy_manifest_path = legacy / "analysis-manifest.json"
    legacy_receipt_path = legacy / "rooted-analysis-receipt.json"
    legacy_identity = {
        "report_sha256": sha256_file(legacy_report_path),
        "manifest_sha256": sha256_file(legacy_manifest_path),
        "receipt_sha256": sha256_file(legacy_receipt_path),
        "member_root": _object(legacy_manifest_path)["member_root"],
        "receipt_identity": _object(legacy_receipt_path)["identity_sha256"],
        "verify_status": legacy_verified["status"],
    }
    legacy_inputs = [member(path, kind="immutable_v1_package") for path in sorted(legacy.iterdir()) if path.is_file()]
    backfill, combined_rows, backfill_inputs = load_backfill(backfill_root)
    failure_inputs: list[dict[str, Any]] = []
    for path in sorted(failed_root.glob("task-*/failure-boundary.json")):
        failure = _object(path)
        verify_canonical_identity(failure)
        if failure.get("status") != "TECHNICAL_INVALID" or failure.get("exception_type") != "KeyError":
            raise AnalysisBoundary("excluded TECH-R1 failure boundary differs")
        failure_inputs.append(member(path, kind="excluded_preedit_ns_technical_attempt"))
    if len(failure_inputs) != 2:
        raise AnalysisBoundary("excluded TECH-R1 failure denominator differs")
    inputs = legacy_inputs + backfill_inputs + failure_inputs
    inputs_before = [dict(row) for row in inputs]
    input_root = canonical_hash(inputs_before)

    core = pd.read_csv(legacy / "core-performance-summary.csv")
    nll = pd.read_csv(legacy / "nll-distribution-summary.csv")
    prompt = pd.read_csv(legacy / "prompt-nll.csv.gz", low_memory=False)
    requests = pd.read_csv(legacy / "request-endpoint-metrics.csv.gz")
    performance = build_performance(core, requests, backfill)
    preedit_nll = build_preedit_nll(nll, backfill)
    official_deltas = build_official_deltas(performance, nll)
    correction = build_metric_correction(prompt)

    output.mkdir(parents=True, mode=0o755)
    _write_csv_once(output / "baseline-inclusive-performance.csv", performance)
    _write_csv_once(output / "preedit-nll-summary.csv", preedit_nll)
    _write_csv_once(output / "official-vs-preedit-deltas.csv", official_deltas)
    _write_csv_once(output / "locality-metric-schema-correction.csv", correction)
    _write_csv_once(output / "preedit-canonical-ns-prompt-pairs.csv.gz", pd.DataFrame(combined_rows), compressed=True)
    ns_summary = pd.DataFrame(
        [
            {
                "model": model,
                "ns_numerator": result["summary"]["ns_numerator"],
                "ns_denominator": result["summary"]["ns_denominator"],
                "ns_rate": result["summary"]["ns_rate"],
                "nll_tie_count": result["summary"]["nll_tie_count"],
                "row_identity_root": result["summary"]["row_identity_root"],
                "summary_identity_sha256": result["summary"]["identity_sha256"],
            }
            for model, result in backfill.items()
        ]
    )
    _write_csv_once(output / "preedit-canonical-ns-summary.csv", ns_summary)
    scheduler = {
        "schema": "odeedit.s06.orbode.round0-baseline-inclusive.scheduler.v2",
        "status": "TECH_R2_2_OF_2_TERMINAL_VALID",
        "rows": _scheduler_rows(),
        "canonical_job": "36275_[0-1]",
        "excluded_job": "36273_[0-1]",
    }
    scheduler["identity_sha256"] = canonical_hash(scheduler)
    write_json_once(output / "preedit-ns-scheduler-terminal.json", scheduler)

    repro_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orbode-baseline-plot-repro-", dir=repro_root) as temporary:
        reproduced = Path(temporary)
        final_command = [sys.executable, "-m", "project.run_scripts.ordered_response_barrier_ode.round0_baseline_figures", "--tables", str(output), "--output", str(output)]
        repro_command = [sys.executable, "-m", "project.run_scripts.ordered_response_barrier_ode.round0_baseline_figures", "--tables", str(output), "--output", str(reproduced)]
        environment = dict(os.environ)
        environment["MPLCONFIGDIR"] = str(repro_root / "mplconfig")
        environment["SOURCE_DATE_EPOCH"] = "0"
        subprocess.run(final_command, cwd=repo, env=environment, check=True, text=True, capture_output=True)
        subprocess.run(repro_command, cwd=repo, env=environment, check=True, text=True, capture_output=True)
        plot_rows: list[dict[str, Any]] = []
        for path in sorted(output.glob("*.png")):
            counterpart = reproduced / path.name
            if not counterpart.is_file() or counterpart.is_symlink():
                raise AnalysisBoundary(f"reproduced plot absent: {path.name}")
            left_sha, right_sha = sha256_file(path), sha256_file(counterpart)
            if left_sha != right_sha or path.stat().st_size != counterpart.stat().st_size:
                raise AnalysisBoundary(f"plot byte reproduction differs: {path.name}")
            plot_rows.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": left_sha,
                    "reproduced_bytes": counterpart.stat().st_size,
                    "reproduced_sha256": right_sha,
                    "byte_stable": True,
                }
            )
    plot_receipt = {
        "schema": "odeedit.s06.orbode.round0-baseline-inclusive.plot-reproduction.v2",
        "status": "BYTE_STABLE_REPRODUCTION_PASS",
        "commands": [shlex.join(final_command), shlex.join(repro_command)],
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "pillow": __import__("PIL").__version__,
            "backend": matplotlib.get_backend(),
            "SOURCE_DATE_EPOCH": "0",
        },
        "inputs": [
            member(output / "baseline-inclusive-performance.csv", relative_to=output, kind="plot_input"),
            member(output / "preedit-canonical-ns-summary.csv", relative_to=output, kind="plot_input"),
        ],
        "outputs": plot_rows,
    }
    plot_receipt["identity_sha256"] = canonical_hash(plot_receipt)
    write_json_once(output / "plot-reproduction.json", plot_receipt)

    legacy_text = legacy_report_path.read_text(encoding="utf-8")
    report_text = _report(
        performance,
        preedit_nll,
        official_deltas,
        correction,
        backfill,
        legacy_text,
        legacy_identity,
        backfill_inputs,
    )
    report_path = output / "ordered-response-barrier-ode-round0-b100-baseline-inclusive-factual-ko.md"
    descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(report_text)

    inputs_after = [member(Path(row["path"]), kind=row["kind"]) for row in inputs_before]
    if inputs_before != inputs_after or input_root != canonical_hash(inputs_after):
        raise AnalysisBoundary("immutable v1/backfill input changed during successor analysis")
    sources = [member(repo / path, relative_to=repo, kind="analysis_source") for path in SOURCE_FILES]
    summary = {
        "schema": "odeedit.s06.orbode.round0-baseline-inclusive.summary.v2",
        "instruction_id": INSTRUCTION_ID,
        "status": "ANALYSIS_ONLY_BASELINE_INCLUSIVE_PASS",
        "canonical_round0_endpoint_count": 2000,
        "canonical_round0_job": "35694",
        "preedit_ns_backfill_job": "36275_[0-1]",
        "preedit_ns": {
            model: {
                "numerator": int(result["summary"]["ns_numerator"]),
                "denominator": int(result["summary"]["ns_denominator"]),
                "rate": float(result["summary"]["ns_rate"]),
            }
            for model, result in backfill.items()
        },
        "endpoint_canonical_ns": "NOT_RECORDED_SCHEMA_GAP_LOCALITY_TARGET_NEW_NLL",
        "pp_prompt_denominator_per_endpoint": 1000,
        "pp_token_denominator_per_endpoint": 1010,
        "pp_token_denominator_formula": "990 locality prompts x 1 token + 10 prompts x 2 tokens",
        "pp_multitoken_case_id": 9936,
        "pp_multitoken_request_ordinal": 61,
        "legacy_identity": legacy_identity,
        "technical_failure_attempt_count": 2,
        "technical_failure_canonical_denominator": 0,
        "backfill_terminal_valid_cell_count": 2,
        "backfill_editor_compute_z_writer_model_update_count": 0,
        "report_generation_model_gpu_slurm_action_count": 0,
        "remaining_nine_submit_count": 0,
        "imputation_count": 0,
        "scientific_promotion": False,
    }
    summary["identity_sha256"] = canonical_hash(summary)
    write_json_once(output / "analysis-summary.json", summary)

    package_members = [
        member(path, relative_to=output, kind="canonical_output")
        for path in sorted(output.iterdir())
        if path.is_file() and path.name not in {"analysis-manifest.json", "rooted-analysis-receipt.json"}
    ]
    manifest = {
        "schema": "odeedit.s06.orbode.round0-baseline-inclusive.manifest.v2",
        "instruction_id": INSTRUCTION_ID,
        "status": summary["status"],
        "analysis_source": {"head": _git(repo, "rev-parse", "HEAD"), "tree": _git(repo, "rev-parse", "HEAD^{tree}")},
        "stream_root": STREAM_ROOT,
        "order_root": ORDER_ROOT,
        "inputs": inputs_before,
        "input_root": input_root,
        "analysis_sources": sources,
        "analysis_source_root": canonical_hash(sources),
        "members": package_members,
        "member_root": canonical_hash(package_members),
        "legacy_identity": legacy_identity,
        "summary_identity_sha256": summary["identity_sha256"],
        "scientific_promotion": False,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_path = output / "analysis-manifest.json"
    write_json_once(manifest_path, manifest)
    receipt = {
        "schema": "odeedit.s06.orbode.round0-baseline-inclusive.rooted-receipt.v2",
        "instruction_id": INSTRUCTION_ID,
        "status": summary["status"],
        "analysis_source": manifest["analysis_source"],
        "stream_root": STREAM_ROOT,
        "order_root": ORDER_ROOT,
        "input_root": input_root,
        "analysis_source_root": manifest["analysis_source_root"],
        "member_root": manifest["member_root"],
        "manifest": {
            "path": manifest_path.name,
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "identity_sha256": manifest["identity_sha256"],
        },
        "report": {
            "path": report_path.name,
            "bytes": report_path.stat().st_size,
            "sha256": sha256_file(report_path),
        },
        "summary_identity_sha256": summary["identity_sha256"],
        "preedit_ns_denominator_per_model": 1000,
        "endpoint_pp_prompt_denominator": 1000,
        "endpoint_pp_token_denominator": 1010,
        "report_generation_model_gpu_slurm_action_count": 0,
        "scientific_promotion": False,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    receipt_path = output / "rooted-analysis-receipt.json"
    write_json_once(receipt_path, receipt)
    return {
        "status": receipt["status"],
        "output": str(output),
        "report_sha256": receipt["report"]["sha256"],
        "manifest_sha256": receipt["manifest"]["sha256"],
        "manifest_identity": receipt["manifest"]["identity_sha256"],
        "receipt_sha256": sha256_file(receipt_path),
        "receipt_identity": receipt["identity_sha256"],
        "member_root": receipt["member_root"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--legacy-package", type=Path, required=True)
    parser.add_argument("--backfill-root", type=Path, required=True)
    parser.add_argument("--failed-backfill-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repro-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.repo,
                args.legacy_package,
                args.backfill_root,
                args.failed_backfill_root,
                args.output,
                args.repro_root,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
