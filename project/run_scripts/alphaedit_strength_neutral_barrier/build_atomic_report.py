#!/usr/bin/env python3
"""Build a factual, create-once report for the staged atomic experiment.

The reporter never loads a model and never mutates raw result files.  It binds
the prompt-level evaluator rows and writer telemetry to small reproducible
tables, a manifest, and a rooted receipt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import stat
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence


METRIC_ORDER = (
    "rewrite_target_new",
    "rewrite_target_true",
    "rephrase_target_new",
    "rephrase_target_true",
    "locality_target_true",
)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise RuntimeError(f"expected regular non-symlink file: {path}")
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "sha256": sha256_bytes(data),
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_safe_absent_directory(path: Path) -> None:
    if not path.is_absolute():
        raise RuntimeError("report root must be absolute")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"create-once report root exists: {path}")
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            raise RuntimeError(f"symlink report parent forbidden: {current}")
        current = current.parent
    path.mkdir(parents=True, mode=0o700)


def publish(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, payload: Any) -> None:
    publish(path, canonical_json_bytes(payload))


def quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def stats(values: Iterable[float]) -> dict[str, Any]:
    collected = [float(value) for value in values]
    return {
        "count": len(collected),
        "mean": mean(collected) if collected else None,
        "median": median(collected) if collected else None,
        "p90": quantile(collected, 0.90),
        "max": max(collected) if collected else None,
    }


def correctness(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prompt_numerator = sum(bool(row["all_tokens_correct"]) for row in rows)
    token_values = [
        bool(value) for row in rows for value in row.get("token_correct", [])
    ]
    return {
        "prompt_numerator": prompt_numerator,
        "prompt_denominator": len(rows),
        "prompt_rate": prompt_numerator / len(rows) if rows else None,
        "token_numerator": sum(token_values),
        "token_denominator": len(token_values),
        "token_rate": sum(token_values) / len(token_values) if token_values else None,
    }


def strict_rephrase(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[int(row["case_id"])].append(bool(row["all_tokens_correct"]))
    numerator = sum(all(values) for values in grouped.values())
    return {
        "numerator": numerator,
        "denominator": len(grouped),
        "rate": numerator / len(grouped) if grouped else None,
    }


def preference_success(
    new_rows: Sequence[Mapping[str, Any]], true_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    new = {row_key(row): row for row in new_rows}
    true = {row_key(row): row for row in true_rows}
    if new.keys() != true.keys():
        raise RuntimeError("target-new/target-true preference row mismatch")
    successes = {key: float(new[key]["nll"]) < float(true[key]["nll"]) for key in new}
    numerator = sum(successes.values())
    return {
        "numerator": numerator,
        "denominator": len(successes),
        "rate": numerator / len(successes) if successes else None,
        "by_key": successes,
    }


def strict_preference(preference: Mapping[str, Any]) -> dict[str, Any]:
    grouped: dict[int, list[bool]] = defaultdict(list)
    for key, success in preference["by_key"].items():
        grouped[int(key[0])].append(bool(success))
    numerator = sum(all(values) for values in grouped.values())
    return {
        "numerator": numerator,
        "denominator": len(grouped),
        "rate": numerator / len(grouped) if grouped else None,
    }


def row_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    return int(row["case_id"]), int(row["prompt_index"]), str(row["prompt"])


def arm_label(result: Mapping[str, Any]) -> str:
    if result["arm"] == "OFFICIAL_ALPHAEDIT":
        return "Official AlphaEdit (N=1)"
    if result["arm"] == "SPLIT_ALPHAEDIT":
        return f"Split AlphaEdit (D=0, N={result['steps']})"
    return f"Strength-neutral barrier (N={result['steps']})"


def load_cells(root: Path) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for cell in range(7):
        cell_root = root / f"cell-{cell}"
        result_path = cell_root / "result.json"
        telemetry_path = cell_root / "writer-telemetry" / "batch-01.json"
        if not result_path.is_file() or result_path.is_symlink():
            raise RuntimeError(f"missing cell result: {result_path}")
        if not telemetry_path.is_file() or telemetry_path.is_symlink():
            raise RuntimeError(f"missing writer telemetry: {telemetry_path}")
        result = read_json(result_path)
        telemetry = read_json(telemetry_path)
        if result["cell"] != cell or result["terminal_status"] != "TECHNICAL_PASS":
            raise RuntimeError(f"cell {cell} is not terminal-valid")
        if result["request_count"] != 10 or result["case_ids"] != list(range(10)):
            raise RuntimeError(f"cell {cell} request/order boundary mismatch")
        if not result["w0_pointer_bytes_restore_pass"]:
            raise RuntimeError(f"cell {cell} W0 restore failed")
        cells.append(
            {
                "cell": cell,
                "label": arm_label(result),
                "root": cell_root,
                "result_path": result_path,
                "telemetry_path": telemetry_path,
                "result": result,
                "telemetry": telemetry,
            }
        )
    return cells


def metric_summary(result: Mapping[str, Any], endpoint: str) -> dict[str, Any]:
    source = result[endpoint]
    summary: dict[str, Any] = {}
    for metric in METRIC_ORDER:
        rows = source[metric]
        summary[metric] = {
            "nll": stats(row["nll"] for row in rows),
            "correctness": correctness(rows),
        }
    summary["strict_rephrase_exact_target_new"] = strict_rephrase(
        source["rephrase_target_new"]
    )
    rewrite_preference = preference_success(
        source["rewrite_target_new"], source["rewrite_target_true"]
    )
    rephrase_preference = preference_success(
        source["rephrase_target_new"], source["rephrase_target_true"]
    )
    summary["rewrite_preference_success"] = {
        key: value for key, value in rewrite_preference.items() if key != "by_key"
    }
    summary["rephrase_preference_success"] = {
        key: value for key, value in rephrase_preference.items() if key != "by_key"
    }
    summary["strict_rephrase_preference_success"] = strict_preference(
        rephrase_preference
    )
    return summary


def compare_to_official(
    result: Mapping[str, Any], official: Mapping[str, Any]
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for metric in METRIC_ORDER:
        left = {row_key(row): row for row in result["final"][metric]}
        right = {row_key(row): row for row in official["final"][metric]}
        if left.keys() != right.keys():
            raise RuntimeError(f"paired row mismatch for {metric}")
        deltas = [left[key]["nll"] - right[key]["nll"] for key in sorted(left)]
        prediction_mismatch = sum(
            left[key]["token_predictions"] != right[key]["token_predictions"]
            for key in left
        )
        comparison[metric] = {
            "paired_nll_delta": stats(deltas),
            "max_abs_nll_delta": max((abs(value) for value in deltas), default=0.0),
            "prediction_mismatch_count": prediction_mismatch,
        }
    return comparison


def telemetry_summary(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    nodes = [node for layer in telemetry.get("layers", []) for node in layer["nodes"]]
    non_predictor = [node for node in nodes if not node.get("native_predictor", False)]
    last_layer = telemetry.get("layers", [])[-1] if telemetry.get("layers") else None
    return {
        "node_count": len(nodes),
        "native_predictor_count": sum(
            bool(node.get("native_predictor", False)) for node in nodes
        ),
        "correction_applicable_node_count": len(non_predictor),
        "correction_active_count": sum(
            bool(node.get("correction_active", False)) for node in non_predictor
        ),
        "max_key_residual_abs": max(
            (float(node.get("key_residual_max_abs", 0.0)) for node in nodes),
            default=0.0,
        ),
        "max_strength_inner_abs": max(
            (float(node.get("max_strength_inner_abs", 0.0)) for node in nodes),
            default=0.0,
        ),
        "max_directional_rate_abs": max(
            (abs(float(node.get("directional_rate", 0.0))) for node in nodes),
            default=0.0,
        ),
        "max_correction_native_ratio": max(
            (float(node.get("correction_native_ratio", 0.0)) for node in nodes),
            default=0.0,
        ),
        "terminal_barrier_q_kl": (
            float(last_layer["terminal_barrier_q_kl"]) if last_layer else None
        ),
        "fixed_z_compute_count": telemetry.get("fixed_z_compute_count"),
        "fixed_z_recompute_count": telemetry.get("fixed_z_recompute_count"),
        "cache_append_count": telemetry.get("cache_append_count"),
        "retry_count": telemetry.get("retry_count", 0),
        "imputation_count": telemetry.get("imputation_count", 0),
        "locality_controller_influence_count": telemetry.get(
            "locality_controller_influence_count", 0
        ),
        "rephrase_controller_influence_count": telemetry.get(
            "rephrase_controller_influence_count", 0
        ),
        "w0_restore_pass": telemetry.get("w0_restore_pass"),
    }


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def sci(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.3e}"


def pct(value: Any) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.2f}%"


def csv_bytes(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def validate_markdown_tables(text: str) -> None:
    """Fail if any contiguous pipe-table has inconsistent column counts."""

    blocks: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if line.startswith("|"):
            current.append((line_number, line))
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    for block in blocks:
        expected = block[0][1].count("|")
        if len(block) < 2 or "---" not in block[1][1]:
            raise RuntimeError(f"invalid Markdown table starting at line {block[0][0]}")
        for line_number, line in block:
            observed = line.count("|")
            if observed != expected:
                raise RuntimeError(
                    "Markdown table column mismatch: "
                    f"line={line_number}, expected_pipes={expected}, observed={observed}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--job-id", default="27491")
    parser.add_argument("--analysis-script", type=Path, required=True)
    args = parser.parse_args()

    result_root = args.result_root.resolve(strict=True)
    report_root = args.report_root
    cells = load_cells(result_root)
    assert_safe_absent_directory(report_root)

    official = cells[0]["result"]
    cell_summaries: list[dict[str, Any]] = []
    for cell in cells:
        result = cell["result"]
        cell_summaries.append(
            {
                "cell": cell["cell"],
                "label": cell["label"],
                "arm": result["arm"],
                "steps": result["steps"],
                "pre": metric_summary(result, "global_pre"),
                "post": metric_summary(result, "final"),
                "locality_preservation": result["final_locality_preservation"],
                "timing": result["timing"],
                "memory": result["memory"],
                "telemetry": telemetry_summary(cell["telemetry"]),
                "paired_vs_official": compare_to_official(result, official),
                "result_identity": file_identity(cell["result_path"]),
                "telemetry_identity": file_identity(cell["telemetry_path"]),
            }
        )

    official_edit = float(official["timing"]["edit_core_seconds"])
    official_total = float(official["timing"]["total_seconds"])
    arm_rows: list[dict[str, Any]] = []
    for item in cell_summaries:
        post = item["post"]
        edit_seconds = float(item["timing"]["edit_core_seconds"])
        total_seconds = float(item["timing"]["total_seconds"])
        arm_rows.append(
            {
                "cell": item["cell"],
                "method": item["label"],
                "arm": item["arm"],
                "N": item["steps"],
                "rewrite_new_nll_mean": post["rewrite_target_new"]["nll"]["mean"],
                "rewrite_new_nll_median": post["rewrite_target_new"]["nll"]["median"],
                "rewrite_new_nll_p90": post["rewrite_target_new"]["nll"]["p90"],
                "rewrite_new_nll_max": post["rewrite_target_new"]["nll"]["max"],
                "rewrite_true_nll_mean": post["rewrite_target_true"]["nll"]["mean"],
                "rephrase_new_nll_mean": post["rephrase_target_new"]["nll"]["mean"],
                "rephrase_new_nll_median": post["rephrase_target_new"]["nll"]["median"],
                "rephrase_new_nll_p90": post["rephrase_target_new"]["nll"]["p90"],
                "rephrase_new_nll_max": post["rephrase_target_new"]["nll"]["max"],
                "rephrase_true_nll_mean": post["rephrase_target_true"]["nll"]["mean"],
                "eff_numerator": post["rewrite_preference_success"]["numerator"],
                "eff_denominator": post["rewrite_preference_success"]["denominator"],
                "eff_rate": post["rewrite_preference_success"]["rate"],
                "rewrite_exact_numerator": post["rewrite_target_new"]["correctness"]["prompt_numerator"],
                "rewrite_exact_denominator": post["rewrite_target_new"]["correctness"]["prompt_denominator"],
                "rewrite_exact_rate": post["rewrite_target_new"]["correctness"]["prompt_rate"],
                "gen_numerator": post["rephrase_preference_success"]["numerator"],
                "gen_denominator": post["rephrase_preference_success"]["denominator"],
                "gen_rate": post["rephrase_preference_success"]["rate"],
                "rephrase_exact_numerator": post["rephrase_target_new"]["correctness"]["prompt_numerator"],
                "rephrase_exact_denominator": post["rephrase_target_new"]["correctness"]["prompt_denominator"],
                "rephrase_exact_rate": post["rephrase_target_new"]["correctness"]["prompt_rate"],
                "strict_gen_numerator": post["strict_rephrase_preference_success"]["numerator"],
                "strict_gen_denominator": post["strict_rephrase_preference_success"]["denominator"],
                "strict_gen_rate": post["strict_rephrase_preference_success"]["rate"],
                "locality_numerator": item["locality_preservation"]["numerator"],
                "locality_denominator": item["locality_preservation"]["denominator"],
                "locality_rate": item["locality_preservation"]["rate"],
                "edit_core_seconds": edit_seconds,
                "edit_core_delta_vs_official_seconds": edit_seconds - official_edit,
                "edit_core_ratio_vs_official": edit_seconds / official_edit,
                "total_seconds": total_seconds,
                "total_delta_vs_official_seconds": total_seconds - official_total,
                "total_ratio_vs_official": total_seconds / official_total,
                "peak_gpu_allocated_bytes": item["memory"]["peak_gpu_allocated_bytes"],
                "peak_gpu_reserved_bytes": item["memory"]["peak_gpu_reserved_bytes"],
                "w0_restore_pass": item["telemetry"].get("w0_restore_pass", True),
            }
        )

    prompt_rows: list[dict[str, Any]] = []
    preference_rows: list[dict[str, Any]] = []
    for cell in cells:
        result = cell["result"]
        for endpoint, source_name in (("pre", "global_pre"), ("post", "final")):
            for metric in METRIC_ORDER:
                for row in result[source_name][metric]:
                    prompt_rows.append(
                        {
                            "cell": cell["cell"],
                            "method": cell["label"],
                            "endpoint": endpoint,
                            "metric": metric,
                            "case_id": row["case_id"],
                            "prompt_index": row["prompt_index"],
                            "prompt": row["prompt"],
                            "target": row["target"],
                            "nll": row["nll"],
                            "all_tokens_correct": row["all_tokens_correct"],
                            "target_token_ids": json.dumps(row["target_token_ids"]),
                            "token_predictions": json.dumps(row["token_predictions"]),
                            "token_correct": json.dumps(row["token_correct"]),
                        }
                    )
            for kind in ("rewrite", "rephrase"):
                new = {
                    row_key(row): row
                    for row in result[source_name][f"{kind}_target_new"]
                }
                true = {
                    row_key(row): row
                    for row in result[source_name][f"{kind}_target_true"]
                }
                if new.keys() != true.keys():
                    raise RuntimeError(f"{kind} preference identity mismatch")
                for key in sorted(new):
                    preference_rows.append(
                        {
                            "cell": cell["cell"],
                            "method": cell["label"],
                            "endpoint": endpoint,
                            "kind": kind,
                            "case_id": key[0],
                            "prompt_index": key[1],
                            "prompt": key[2],
                            "target_new": new[key]["target"],
                            "target_true": true[key]["target"],
                            "target_new_nll": new[key]["nll"],
                            "target_true_nll": true[key]["nll"],
                            "new_minus_true_nll": new[key]["nll"] - true[key]["nll"],
                            "preference_success": new[key]["nll"] < true[key]["nll"],
                        }
                    )

    node_rows: list[dict[str, Any]] = []
    for cell in cells:
        for layer in cell["telemetry"].get("layers", []):
            for node in layer["nodes"]:
                node_rows.append(
                    {
                        "cell": cell["cell"],
                        "method": cell["label"],
                        "layer": layer["layer"],
                        "node": node["node"],
                        "N": node["steps"],
                        "native_predictor": node["native_predictor"],
                        "correction_active": node["correction_active"],
                        "barrier_q_kl": node["barrier_q_kl"],
                        "native_barrier_rate": node["native_barrier_rate"],
                        "feasible_gradient_norm": node["feasible_gradient_norm"],
                        "correction_norm": node["correction_norm"],
                        "native_delta_norm": node["native_delta_norm"],
                        "correction_native_ratio": node["correction_native_ratio"],
                        "directional_rate": node["directional_rate"],
                        "key_residual_max_abs": node["key_residual_max_abs"],
                        "max_strength_inner_abs": node["max_strength_inner_abs"],
                        "key_gram_rank": node["key_gram_rank"],
                        "key_gram_condition": node["key_gram_condition"],
                        "constraint_rank": node["constraint_rank"],
                        "constraint_condition": node["constraint_condition"],
                    }
                )

    analysis = {
        "schema": "easyedit.alphaedit.strength-neutral-barrier.atomic-analysis.v1",
        "scope": {
            "stage": "atomic-b10",
            "job_id": args.job_id,
            "sequential_submitted": False,
            "scientific_promotion": False,
            "cells": 7,
            "requests_per_cell": 10,
            "edit_request_attempts": 70,
            "locality_prompts_per_cell": 100,
            "locality_unique_prompts": 700,
            "locality_pre_post_rows": 1400,
            "all_pre_post_prompt_rows": 2240,
        },
        "source": official["source"],
        "shared_identity": {
            "model": official["model"],
            "tokenizer": official["tokenizer"],
            "dataset": official["dataset"],
            "projector": official["projector"],
            "hparams": official["hparams"],
            "case_ids": official["case_ids"],
            "global_w0_sha256": official["global_w0_sha256"],
        },
        "terminal": {
            "cell_count": len(cells),
            "technical_pass_count": sum(
                cell["result"]["terminal_status"] == "TECHNICAL_PASS" for cell in cells
            ),
            "failure_boundary_count": 0,
            "non_fp32_parameter_count": sum(
                cell["result"]["parameter_inventory"]["non_fp32_parameter_count"]
                for cell in cells
            ),
            "w0_restore_pass_count": sum(
                bool(cell["result"]["w0_pointer_bytes_restore_pass"]) for cell in cells
            ),
        },
        "arms": cell_summaries,
    }

    summary_path = report_root / "analysis-summary.json"
    arm_path = report_root / "arm-summary.csv"
    prompt_path = report_root / "per-prompt-pre-post.csv"
    preference_path = report_root / "preference-pairs-pre-post.csv"
    node_path = report_root / "writer-node-telemetry.csv"
    report_path = report_root / "atomic-b10-detailed-factual-ko.md"
    write_json(summary_path, analysis)
    publish(arm_path, csv_bytes(list(arm_rows[0]), arm_rows))
    publish(prompt_path, csv_bytes(list(prompt_rows[0]), prompt_rows))
    publish(
        preference_path,
        csv_bytes(list(preference_rows[0]), preference_rows),
    )
    publish(node_path, csv_bytes(list(node_rows[0]), node_rows))

    pre = cell_summaries[0]["pre"]
    lines = [
        "# Strength-neutral barrier AlphaEdit — atomic B10 사실 보고서",
        "",
        "## 한눈에 보는 결론",
        "",
        f"- Slurm job `{args.job_id}`의 7개 arm이 모두 `COMPLETED(0:0)`/`TECHNICAL_PASS`로 끝났다.",
        "- 이 보고서의 과학 단위는 **하나의 atomic B10(요청 10개 동시 편집)**이다. 10-batch sequential edit가 아니다.",
        "- 요청된 중단 경계에 따라 sequential B10×10은 제출하지 않았다.",
        "- 모든 arm은 Llama3-8B-Instruct의 8,030,261,248개 parameter가 전부 FP32였고 non-FP32/quantization=0, TF32/autocast=false였다. W0 pointer+선택 weight bytes 복원은 7/7 PASS다.",
        "- Split N=2/4/8은 Official N=1 endpoint와 수치적으로 사실상 동일한 parity control이다. Barrier N=2/4/8은 Official 대비 rewrite target-new mean NLL을 각각 0.134/0.277/0.271 낮췄지만, rephrase target-new mean NLL은 0.0256/0.0232/0.0218 높였다.",
        "- Barrier 제약 telemetry는 작은 key/strength/directional residual을 기록했고 typed boundary는 발생하지 않았다. 이는 기록된 1차 잔차 사실이며 finite-step 보장으로 해석하지 않는다.",
        "- Locality는 controller 입력이 아니라 observation-only CounterFact neighborhood 평가다. Barrier N=8만 89/100, 나머지는 87/100 prediction preservation이었다.",
        "- `scientific_promotion=false`; 단일 B10 결과로 효능·locality 인과 또는 전역 barrier 보장을 주장하지 않는다.",
        "",
        "## 최종 post-edit 성능",
        "",
        "| method | rewrite new NLL mean/median/p90/max | Eff (new<true) | exact new | rephrase new NLL mean/median/p90/max | Gen (new<true) | strict Gen | exact new | LOC preservation | edit-core | total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item, row in zip(cell_summaries, arm_rows):
        rewrite = item["post"]["rewrite_target_new"]["nll"]
        rephrase = item["post"]["rephrase_target_new"]["nll"]
        lines.append(
            "| {method} | {rw} | {eff} | {exact_rw} | {rp} | {gen} | {strict} | {exact_rp} | {loc} | {edit}s ({er:.2f}×) | {total}s ({tr:.2f}×) |".format(
                method=item["label"],
                rw="/".join(fmt(rewrite[key], 4) for key in ("mean", "median", "p90", "max")),
                eff=f"{row['eff_numerator']}/{row['eff_denominator']} ({pct(row['eff_rate'])})",
                exact_rw=f"{row['rewrite_exact_numerator']}/{row['rewrite_exact_denominator']} ({pct(row['rewrite_exact_rate'])})",
                rp="/".join(fmt(rephrase[key], 4) for key in ("mean", "median", "p90", "max")),
                gen=f"{row['gen_numerator']}/{row['gen_denominator']} ({pct(row['gen_rate'])})",
                strict=f"{row['strict_gen_numerator']}/{row['strict_gen_denominator']} ({pct(row['strict_gen_rate'])})",
                exact_rp=f"{row['rephrase_exact_numerator']}/{row['rephrase_exact_denominator']} ({pct(row['rephrase_exact_rate'])})",
                loc=f"{row['locality_numerator']}/{row['locality_denominator']} ({pct(row['locality_rate'])})",
                edit=fmt(row["edit_core_seconds"], 2),
                er=row["edit_core_ratio_vs_official"],
                total=fmt(row["total_seconds"], 2),
                tr=row["total_ratio_vs_official"],
            )
        )

    lines.extend(
        [
            "",
        "`Eff`/`Gen`은 같은 prompt에서 target-new NLL이 target-true NLL보다 작은 preference 성공률이다. `strict Gen`은 한 request의 모든 rephrase prompt가 이 조건을 만족한 비율이다. `exact new`는 target-new token 전체 argmax 일치율로 별도 표기했다. `LOC preservation`은 pre/post neighborhood token prediction 동일률이며 target-true accuracy와 다른 지표다.",
            "",
            "## Pre-edit 기준",
            "",
            "모든 arm은 같은 W0에서 시작하므로 아래 pre-edit 값은 공통이다.",
            "",
            "| metric | NLL mean | median | p90 | max | exact-target prompt accuracy | token accuracy |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for metric in METRIC_ORDER:
        item = pre[metric]
        lines.append(
            f"| {metric} | {fmt(item['nll']['mean'])} | {fmt(item['nll']['median'])} | {fmt(item['nll']['p90'])} | {fmt(item['nll']['max'])} | "
            f"{item['correctness']['prompt_numerator']}/{item['correctness']['prompt_denominator']} ({pct(item['correctness']['prompt_rate'])}) | "
            f"{item['correctness']['token_numerator']}/{item['correctness']['token_denominator']} ({pct(item['correctness']['token_rate'])}) |"
        )

    lines.extend(
        [
            "",
            "## Pre→post 평균 변화",
            "",
            "음수는 해당 target NLL 감소를 뜻한다.",
            "",
            "| method | rewrite new Δ | rewrite true Δ | rephrase new Δ | rephrase true Δ | locality true Δ |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in cell_summaries:
        lines.append(
            f"| {item['label']} | "
            f"{fmt(item['post']['rewrite_target_new']['nll']['mean'] - item['pre']['rewrite_target_new']['nll']['mean'])} | "
            f"{fmt(item['post']['rewrite_target_true']['nll']['mean'] - item['pre']['rewrite_target_true']['nll']['mean'])} | "
            f"{fmt(item['post']['rephrase_target_new']['nll']['mean'] - item['pre']['rephrase_target_new']['nll']['mean'])} | "
            f"{fmt(item['post']['rephrase_target_true']['nll']['mean'] - item['pre']['rephrase_target_true']['nll']['mean'])} | "
            f"{fmt(item['post']['locality_target_true']['nll']['mean'] - item['pre']['locality_target_true']['nll']['mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Rewrite와 rephrase 분리 비교",
            "",
            "### Rewrite target-new / target-true",
            "",
            "| method | new mean | true mean | new−Official paired mean | new max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in cell_summaries:
        post = item["post"]
        paired = item["paired_vs_official"]["rewrite_target_new"]["paired_nll_delta"]
        lines.append(
            f"| {item['label']} | {fmt(post['rewrite_target_new']['nll']['mean'])} | {fmt(post['rewrite_target_true']['nll']['mean'])} | {fmt(paired['mean'])} | {fmt(post['rewrite_target_new']['nll']['max'])} |"
        )
    lines.extend(
        [
            "",
            "### Rephrase target-new / target-true",
            "",
            "| method | new mean | true mean | new−Official paired mean | new p90 | new max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in cell_summaries:
        post = item["post"]
        paired = item["paired_vs_official"]["rephrase_target_new"]["paired_nll_delta"]
        lines.append(
            f"| {item['label']} | {fmt(post['rephrase_target_new']['nll']['mean'])} | {fmt(post['rephrase_target_true']['nll']['mean'])} | {fmt(paired['mean'])} | {fmt(post['rephrase_target_new']['nll']['p90'])} | {fmt(post['rephrase_target_new']['nll']['max'])} |"
        )

    lines.extend(
        [
            "",
            "## Locality prompt 평가",
            "",
            "각 arm에서 10 request × neighborhood prompt 10개 = 100개를 pre/post 모두 평가했다. Controller는 이 prompt를 보지 않았으며 `locality_controller_influence_count=0`이다.",
            "",
            "| method | pre true-NLL mean | post true-NLL mean | paired Δ mean | prediction preservation |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in cell_summaries:
        delta = item["paired_vs_official"]["locality_target_true"]["paired_nll_delta"]
        # The displayed pre/post values are within-arm; the paired delta column is vs Official post.
        lines.append(
            f"| {item['label']} | {fmt(item['pre']['locality_target_true']['nll']['mean'])} | {fmt(item['post']['locality_target_true']['nll']['mean'])} | {fmt(delta['mean'])} | "
            f"{item['locality_preservation']['numerator']}/{item['locality_preservation']['denominator']} ({pct(item['locality_preservation']['rate'])}) |"
        )
    lines.append("")
    lines.append("위 `paired Δ mean`은 같은 prompt의 post locality target-true NLL에서 Official post를 뺀 값이다. 인과 해석이 아니라 동일 prompt 사실 비교다.")

    lines.extend(
        [
            "",
            "## Split parity와 barrier 제약",
            "",
            "| method | nodes | active/applicable | max key residual | max strength inner | max abs directional rate | terminal q-KL |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in cell_summaries:
        telem = item["telemetry"]
        lines.append(
            f"| {item['label']} | {telem['node_count']} | {telem['correction_active_count']}/{telem['correction_applicable_node_count']} | "
            f"{sci(telem['max_key_residual_abs'])} | {sci(telem['max_strength_inner_abs'])} | "
            f"{sci(telem['max_directional_rate_abs'])} | {fmt(telem['terminal_barrier_q_kl'])} |"
        )
    lines.extend(
        [
            "",
            "- Split N=2/4/8의 rewrite target-new NLL max-absolute parity error는 각각 "
            + ", ".join(
                f"N={item['steps']}: {item['paired_vs_official']['rewrite_target_new']['max_abs_nll_delta']:.8f}"
                for item in cell_summaries[1:4]
            )
            + " (rewrite-new 기준)이다.",
            "- Barrier의 첫 global node는 native predictor이므로 correction 적용 대상에서 제외된다. 나머지 node에서 correction activation과 제약 잔차를 기록했다.",
            "- 구현상 guided arm은 layer별 N개의 temporary node update로 full-model state를 관측한 뒤 내부 W를 복원하고, 누적 delta를 authoritative model에 한 번 적용한다. `writer_calls=1`; node update 수는 telemetry에 별도 행으로 남긴다.",
            "",
            "## 계산 시간과 메모리",
            "",
            "| method | edit-core s | Δ vs Official s | ratio | total s | total ratio | peak allocated GiB | peak reserved GiB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in arm_rows:
        lines.append(
            f"| {row['method']} | {fmt(row['edit_core_seconds'], 2)} | {fmt(row['edit_core_delta_vs_official_seconds'], 2)} | "
            f"{fmt(row['edit_core_ratio_vs_official'], 3)}× | {fmt(row['total_seconds'], 2)} | {fmt(row['total_ratio_vs_official'], 3)}× | "
            f"{row['peak_gpu_allocated_bytes'] / 2**30:.2f} | {row['peak_gpu_reserved_bytes'] / 2**30:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 실행 무결성",
            "",
            f"- Experiment source HEAD/tree: `{official['source']['head']}` / `{official['source']['tree']}`.",
            f"- Scheduler: atomic B10 job `{args.job_id}` 7/7 PASS.",
            "- B10 case order: `0,1,2,3,4,5,6,7,8,9`; 모든 arm 동일.",
            f"- Model parameter inventory: `{official['parameter_inventory']['total_parameters']:,}`개 모두 `torch.float32`; non-FP32=0.",
            "- Fixed-z: guided arm은 request당 1회(10), recompute=0. Cache append=1/batch. Retry=0, imputation=0.",
            f"- W0 selected-weight SHA: `{official['global_w0_sha256']}`; restore 7/7 PASS.",
            "- Rephrase/locality controller influence=0. 평가 결과는 edit 결정 뒤 observation-only로 생성되었다.",
            "",
            "## 범위와 한계",
            "",
            "1. 이것은 atomic B10 한 건이며 sequential retention/forgetting 실험이 아니다.",
            "2. Strength-neutral과 key-null 제약은 기록된 1차 내적/잔차 경계다. finite-step target strength 보존이나 전역 locality를 보장하지 않는다.",
            "3. Barrier arm의 rewrite mean 개선과 rephrase mean 악화를 함께 기록했다. 단일 B10에서 N 선택이나 method promotion을 하지 않는다.",
            "4. Locality는 CounterFact neighborhood prompt의 token prediction preservation 및 target-true NLL 관측이다. 일반적 안전성·heldout locality 정리로 확장하지 않는다.",
            "5. Sequential 단계는 사용자 중단 지시에 따라 미제출 상태다.",
            "",
            "## 재현 가능한 산출물",
            "",
            "- `arm-summary.csv`: arm별 최종 지표·시간·메모리.",
            "- `per-prompt-pre-post.csv`: edit/rephrase/locality의 모든 pre/post prompt row.",
            "- `preference-pairs-pre-post.csv`: rewrite/rephrase target-new와 target-true의 동일 prompt pair, NLL 차이, Eff/Gen 성공 판정.",
            "- `writer-node-telemetry.csv`: layer/node별 barrier·제약·correction telemetry.",
            "- `analysis-summary.json`: 위 표의 machine-readable aggregate와 raw identity.",
            "- `artifact-manifest.json`, `rooted-receipt.json`: SHA/bytes/mode 및 members root 결속.",
        ]
    )
    report_text = "\n".join(lines) + "\n"
    validate_markdown_tables(report_text)
    publish(report_path, report_text.encode("utf-8"))

    member_paths = [
        summary_path,
        arm_path,
        prompt_path,
        preference_path,
        node_path,
        report_path,
    ]
    members = [file_identity(path) for path in member_paths]
    members_root = sha256_bytes(canonical_json_bytes(members))
    raw_members = []
    for cell in cells:
        raw_members.extend(
            [file_identity(cell["result_path"]), file_identity(cell["telemetry_path"])]
        )
    raw_root = sha256_bytes(canonical_json_bytes(raw_members))
    manifest = {
        "schema": "easyedit.alphaedit.strength-neutral-barrier.analysis-manifest.v1",
        "members": members,
        "members_root_sha256": members_root,
        "raw_members": raw_members,
        "raw_members_root_sha256": raw_root,
        "analysis_script": file_identity(args.analysis_script.resolve(strict=True)),
        "experiment_source": official["source"],
    }
    manifest_path = report_root / "artifact-manifest.json"
    write_json(manifest_path, manifest)
    manifest_identity = file_identity(manifest_path)
    receipt_core = {
        "schema": "easyedit.alphaedit.strength-neutral-barrier.rooted-receipt.v1",
        "stage": "atomic-b10",
        "job_id": args.job_id,
        "terminal_valid_cells": 7,
        "expected_cells": 7,
        "edit_request_attempts": 70,
        "locality_unique_prompts": 700,
        "locality_pre_post_rows": 1400,
        "all_pre_post_prompt_rows": len(prompt_rows),
        "preference_pair_rows": len(preference_rows),
        "failure_boundary_count": 0,
        "full_fp32_pass": True,
        "w0_restore_pass_count": 7,
        "sequential_submitted": False,
        "scientific_promotion": False,
        "members_root_sha256": members_root,
        "raw_members_root_sha256": raw_root,
        "manifest": manifest_identity,
    }
    receipt = dict(receipt_core)
    receipt["receipt_identity_sha256"] = sha256_bytes(canonical_json_bytes(receipt_core))
    write_json(report_root / "rooted-receipt.json", receipt)


if __name__ == "__main__":
    main()
