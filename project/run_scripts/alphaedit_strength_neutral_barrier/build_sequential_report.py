#!/usr/bin/env python3
"""Build a factual report for a terminal sequential B10x10 SNB run.

This analysis-only entry point never loads a model or changes raw results.  It
reuses the metric and receipt primitives from the atomic reporter while making
the sequential batch/forgetting denominator explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from project.run_scripts.alphaedit_strength_neutral_barrier.build_atomic_report import (
    METRIC_ORDER,
    arm_label,
    assert_safe_absent_directory,
    canonical_json_bytes,
    file_identity,
    fmt,
    metric_summary,
    pct,
    publish,
    read_json,
    row_key,
    sha256_bytes,
    stats,
    telemetry_summary,
    validate_markdown_tables,
    write_json,
)


EXPECTED_STAGE = "sequential-b10x10"
EXPECTED_CELLS = 7
EXPECTED_BATCHES = 10
EXPECTED_REQUESTS = 100


def csv_bytes(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def subset_endpoint(
    endpoint: Mapping[str, list[Mapping[str, Any]]], case_ids: Sequence[int]
) -> dict[str, list[Mapping[str, Any]]]:
    allowed = set(int(case_id) for case_id in case_ids)
    return {
        metric: [row for row in endpoint[metric] if int(row["case_id"]) in allowed]
        for metric in METRIC_ORDER
    }


def load_cells(root: Path) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    shared_identity: dict[str, Any] | None = None
    for cell in range(EXPECTED_CELLS):
        cell_root = root / f"cell-{cell}"
        result_path = cell_root / "result.json"
        if not result_path.is_file() or result_path.is_symlink():
            raise RuntimeError(f"missing regular cell result: {result_path}")
        result = read_json(result_path)
        if result.get("cell") != cell or result.get("terminal_status") != "TECHNICAL_PASS":
            raise RuntimeError(f"cell {cell} is not terminal-valid")
        if result.get("stage") != EXPECTED_STAGE:
            raise RuntimeError(f"cell {cell} stage mismatch")
        if result.get("request_count") != EXPECTED_REQUESTS:
            raise RuntimeError(f"cell {cell} request count mismatch")
        if result.get("batch_count") != EXPECTED_BATCHES:
            raise RuntimeError(f"cell {cell} batch count mismatch")
        if result.get("case_ids") != list(range(EXPECTED_REQUESTS)):
            raise RuntimeError(f"cell {cell} request order mismatch")
        if not result.get("sequential_continuity"):
            raise RuntimeError(f"cell {cell} lacks sequential continuity")
        if not result.get("w0_pointer_bytes_restore_pass"):
            raise RuntimeError(f"cell {cell} W0 restore failed")
        if result["parameter_inventory"]["non_fp32_parameter_count"] != 0:
            raise RuntimeError(f"cell {cell} is not FULL-FP32")
        telemetry_paths = [
            cell_root / "writer-telemetry" / f"batch-{batch:02d}.json"
            for batch in range(1, EXPECTED_BATCHES + 1)
        ]
        if any(not path.is_file() or path.is_symlink() for path in telemetry_paths):
            raise RuntimeError(f"cell {cell} writer telemetry incomplete")
        identity = {
            "source": result["source"],
            "model": result["model"],
            "tokenizer": result["tokenizer"],
            "dataset": result["dataset"],
            "projector": result["projector"],
            "hparams": result["hparams"],
            "global_w0_sha256": result["global_w0_sha256"],
        }
        if shared_identity is None:
            shared_identity = identity
        elif identity != shared_identity:
            raise RuntimeError(f"cell {cell} shared identity mismatch")
        cells.append(
            {
                "cell": cell,
                "label": arm_label(result),
                "root": cell_root,
                "result_path": result_path,
                "result": result,
                "telemetry_paths": telemetry_paths,
                "telemetries": [read_json(path) for path in telemetry_paths],
            }
        )
    return cells


def paired_nll(
    left: Mapping[str, list[Mapping[str, Any]]],
    right: Mapping[str, list[Mapping[str, Any]]],
    metric: str,
) -> dict[str, Any]:
    left_rows = {row_key(row): row for row in left[metric]}
    right_rows = {row_key(row): row for row in right[metric]}
    if left_rows.keys() != right_rows.keys():
        raise RuntimeError(f"paired row mismatch for {metric}")
    return stats(float(left_rows[key]["nll"]) - float(right_rows[key]["nll"]) for key in left_rows)


def report_row(cell: Mapping[str, Any]) -> dict[str, Any]:
    result = cell["result"]
    post = metric_summary(result, "final")
    pre = metric_summary(result, "global_pre")
    locality = result["final_locality_preservation"]
    return {
        "cell": cell["cell"],
        "method": cell["label"],
        "arm": result["arm"],
        "N": result["steps"],
        "pre": pre,
        "post": post,
        "rewrite_new_mean": post["rewrite_target_new"]["nll"]["mean"],
        "rewrite_new_median": post["rewrite_target_new"]["nll"]["median"],
        "rewrite_new_p90": post["rewrite_target_new"]["nll"]["p90"],
        "rewrite_new_max": post["rewrite_target_new"]["nll"]["max"],
        "rewrite_true_mean": post["rewrite_target_true"]["nll"]["mean"],
        "rephrase_new_mean": post["rephrase_target_new"]["nll"]["mean"],
        "rephrase_new_median": post["rephrase_target_new"]["nll"]["median"],
        "rephrase_new_p90": post["rephrase_target_new"]["nll"]["p90"],
        "rephrase_new_max": post["rephrase_target_new"]["nll"]["max"],
        "rephrase_true_mean": post["rephrase_target_true"]["nll"]["mean"],
        "eff_n": post["rewrite_preference_success"]["numerator"],
        "eff_d": post["rewrite_preference_success"]["denominator"],
        "eff_rate": post["rewrite_preference_success"]["rate"],
        "gen_n": post["rephrase_preference_success"]["numerator"],
        "gen_d": post["rephrase_preference_success"]["denominator"],
        "gen_rate": post["rephrase_preference_success"]["rate"],
        "strict_gen_n": post["strict_rephrase_preference_success"]["numerator"],
        "strict_gen_d": post["strict_rephrase_preference_success"]["denominator"],
        "strict_gen_rate": post["strict_rephrase_preference_success"]["rate"],
        "loc_n": locality["numerator"],
        "loc_d": locality["denominator"],
        "loc_rate": locality["rate"],
        "edit_core_seconds": result["timing"]["edit_core_seconds"],
        "evaluation_seconds": result["timing"]["evaluation_seconds"],
        "total_seconds": result["timing"]["total_seconds"],
        "peak_gpu_allocated_bytes": result["memory"]["peak_gpu_allocated_bytes"],
        "peak_gpu_reserved_bytes": result["memory"]["peak_gpu_reserved_bytes"],
    }


def build_batch_rows(cell: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = cell["result"]
    rows: list[dict[str, Any]] = []
    for batch in result["batches"]:
        case_ids = batch["case_ids"]
        immediate = metric_summary(batch, "immediate")
        entry = metric_summary(batch, "entry")
        final_endpoint = subset_endpoint(result["final"], case_ids)
        wrapped = {"endpoint": final_endpoint}
        final = metric_summary(wrapped, "endpoint")
        telemetry_path = cell["telemetry_paths"][int(batch["batch_index"]) - 1]
        telemetry = telemetry_summary(read_json(telemetry_path))
        rows.append(
            {
                "cell": cell["cell"],
                "method": cell["label"],
                "batch": batch["batch_index"],
                "case_first": case_ids[0],
                "case_last": case_ids[-1],
                "rewrite_entry_mean": entry["rewrite_target_new"]["nll"]["mean"],
                "rewrite_immediate_mean": immediate["rewrite_target_new"]["nll"]["mean"],
                "rewrite_final_mean": final["rewrite_target_new"]["nll"]["mean"],
                "rewrite_forgetting_mean": paired_nll(final_endpoint, batch["immediate"], "rewrite_target_new")["mean"],
                "rephrase_entry_mean": entry["rephrase_target_new"]["nll"]["mean"],
                "rephrase_immediate_mean": immediate["rephrase_target_new"]["nll"]["mean"],
                "rephrase_final_mean": final["rephrase_target_new"]["nll"]["mean"],
                "rephrase_forgetting_mean": paired_nll(final_endpoint, batch["immediate"], "rephrase_target_new")["mean"],
                "immediate_eff_rate": immediate["rewrite_preference_success"]["rate"],
                "final_eff_rate": final["rewrite_preference_success"]["rate"],
                "immediate_gen_rate": immediate["rephrase_preference_success"]["rate"],
                "final_gen_rate": final["rephrase_preference_success"]["rate"],
                "immediate_strict_gen_rate": immediate["strict_rephrase_preference_success"]["rate"],
                "final_strict_gen_rate": final["strict_rephrase_preference_success"]["rate"],
                "edit_core_seconds": batch["edit_core_seconds"],
                "evaluation_seconds": batch["evaluation_seconds"],
                "node_count": telemetry["node_count"],
                "correction_active_count": telemetry["correction_active_count"],
                "correction_applicable_node_count": telemetry["correction_applicable_node_count"],
                "max_key_residual_abs": telemetry["max_key_residual_abs"],
                "max_strength_inner_abs": telemetry["max_strength_inner_abs"],
                "max_directional_rate_abs": telemetry["max_directional_rate_abs"],
            }
        )
    return rows


def render_report(
    *, job_id: str, cells: Sequence[Mapping[str, Any]], arm_rows: Sequence[Mapping[str, Any]], batch_rows: Sequence[Mapping[str, Any]]
) -> str:
    official_edit = float(arm_rows[0]["edit_core_seconds"])
    official_total = float(arm_rows[0]["total_seconds"])
    lines = [
        "# Strength-neutral barrier AlphaEdit — sequential B10×10 사실 보고서",
        "",
        "## 한눈에 보는 결론",
        "",
        f"- Slurm job `{job_id}`의 7개 arm이 모두 `TECHNICAL_PASS`로 끝났다.",
        "- 과학 단위는 **B10을 10회 순차 편집한 100-request trajectory**다. 각 arm 안에서는 W/cache가 batch 사이에 이어지고, arm 간에는 동일 W0에서 독립 시작한다.",
        "- 모든 arm은 FULL-FP32, autocast/TF32=false이며 terminal 뒤 W0 pointer/bytes restore 7/7 PASS다.",
        "- 아래 `final`은 10번째 batch가 끝난 W에서 100개 요청 전체를 재평가한 값이다. `immediate`와의 차이는 관측된 forgetting/retention이며 인과로 확대하지 않는다.",
        "- `scientific_promotion=false`; 단일 모델·단일 100-request stream 결과다.",
        "",
        "## 최종 W 종합표",
        "",
        "| method | rewrite new NLL mean/median/p90/max | Eff | rephrase new NLL mean/median/p90/max | Gen | strict Gen | LOC | edit-core s (×Official) | total s (×Official) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in arm_rows:
        lines.append(
            f"| {row['method']} | {fmt(row['rewrite_new_mean'],4)}/{fmt(row['rewrite_new_median'],4)}/{fmt(row['rewrite_new_p90'],4)}/{fmt(row['rewrite_new_max'],4)} | "
            f"{row['eff_n']}/{row['eff_d']} ({pct(row['eff_rate'])}) | "
            f"{fmt(row['rephrase_new_mean'],4)}/{fmt(row['rephrase_new_median'],4)}/{fmt(row['rephrase_new_p90'],4)}/{fmt(row['rephrase_new_max'],4)} | "
            f"{row['gen_n']}/{row['gen_d']} ({pct(row['gen_rate'])}) | "
            f"{row['strict_gen_n']}/{row['strict_gen_d']} ({pct(row['strict_gen_rate'])}) | "
            f"{row['loc_n']}/{row['loc_d']} ({pct(row['loc_rate'])}) | "
            f"{fmt(row['edit_core_seconds'],2)} ({float(row['edit_core_seconds'])/official_edit:.3f}×) | "
            f"{fmt(row['total_seconds'],2)} ({float(row['total_seconds'])/official_total:.3f}×) |"
        )
    lines.extend(
        [
            "",
            "Eff/Gen은 같은 prompt에서 target-new NLL < target-true NLL인 비율이다. strict Gen은 request별 모든 rephrase prompt가 그 조건을 만족한 비율이다. LOC는 pre/final neighborhood token prediction preservation이다.",
            "",
            "## Rewrite / rephrase 분리",
            "",
            "| method | rewrite new mean/median/p90/max | rewrite true mean | rephrase new mean/median/p90/max | rephrase true mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in arm_rows:
        lines.append(
            f"| {row['method']} | {fmt(row['rewrite_new_mean'])}/{fmt(row['rewrite_new_median'])}/{fmt(row['rewrite_new_p90'])}/{fmt(row['rewrite_new_max'])} | {fmt(row['rewrite_true_mean'])} | "
            f"{fmt(row['rephrase_new_mean'])}/{fmt(row['rephrase_new_median'])}/{fmt(row['rephrase_new_p90'])}/{fmt(row['rephrase_new_max'])} | {fmt(row['rephrase_true_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Batch별 immediate→final retention",
            "",
            "`forgetting Δ`는 같은 prompt에서 final-W NLL − 해당 batch immediate-post NLL이다. 양수는 NLL 악화다.",
            "",
            "| method | B | rewrite entry/immediate/final | rewrite forgetting Δ | rephrase entry/immediate/final | rephrase forgetting Δ | Eff immediate→final | Gen immediate→final |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in batch_rows:
        lines.append(
            f"| {row['method']} | {row['batch']} | {fmt(row['rewrite_entry_mean'])}/{fmt(row['rewrite_immediate_mean'])}/{fmt(row['rewrite_final_mean'])} | {fmt(row['rewrite_forgetting_mean'])} | "
            f"{fmt(row['rephrase_entry_mean'])}/{fmt(row['rephrase_immediate_mean'])}/{fmt(row['rephrase_final_mean'])} | {fmt(row['rephrase_forgetting_mean'])} | "
            f"{pct(row['immediate_eff_rate'])}→{pct(row['final_eff_rate'])} | {pct(row['immediate_gen_rate'])}→{pct(row['final_gen_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Barrier/제약 telemetry와 계산",
            "",
            "| method | batches | nodes | correction active/applicable | max key residual | max strength inner | max abs directional rate | edit-core s | eval s | peak alloc/reserved GiB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell, arm in zip(cells, arm_rows):
        telemetry = [telemetry_summary(item) for item in cell["telemetries"]]
        lines.append(
            f"| {arm['method']} | 10 | {sum(item['node_count'] for item in telemetry)} | "
            f"{sum(item['correction_active_count'] for item in telemetry)}/{sum(item['correction_applicable_node_count'] for item in telemetry)} | "
            f"{max(item['max_key_residual_abs'] for item in telemetry):.3e} | {max(item['max_strength_inner_abs'] for item in telemetry):.3e} | "
            f"{max(item['max_directional_rate_abs'] for item in telemetry):.3e} | {fmt(arm['edit_core_seconds'],2)} | {fmt(arm['evaluation_seconds'],2)} | "
            f"{arm['peak_gpu_allocated_bytes']/2**30:.2f}/{arm['peak_gpu_reserved_bytes']/2**30:.2f} |"
        )
    source = cells[0]["result"]["source"]
    lines.extend(
        [
            "",
            "## 실행 경계",
            "",
            f"- Source HEAD/tree: `{source['head']}` / `{source['tree']}`.",
            f"- Stock EasyEdit HEAD/tree: `{source['easyedit_head']}` / `{source['easyedit_tree']}`; tracked clean.",
            "- 7 arms × 10 batches × B10 = 700 edit-request attempts; 각 arm final denominator는 100 requests다.",
            "- Guided fixed-z는 batch/request당 1회, recompute=0. Retry/imputation=0. Rephrase/locality controller influence=0.",
            "- 이 결과는 barrier의 finite-step 보장이나 일반적 locality 보장이 아니다. 기록된 stream/모델/설정에서의 factual outcome만 제시한다.",
            "",
            "## 산출물",
            "",
            "- `arm-summary.csv`: 최종 W arm별 종합 지표.",
            "- `batch-retention.csv`: batch별 entry/immediate/final 및 forgetting.",
            "- `analysis-summary.json`: machine-readable aggregate와 raw identity.",
            "- `artifact-manifest.json`, `rooted-receipt.json`: create-once member/root 결속.",
        ]
    )
    text = "\n".join(lines) + "\n"
    validate_markdown_tables(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--analysis-script", type=Path, required=True)
    args = parser.parse_args()

    result_root = args.result_root.resolve(strict=True)
    report_root = args.report_root
    cells = load_cells(result_root)
    assert_safe_absent_directory(report_root)
    arm_rows = [report_row(cell) for cell in cells]
    batch_rows = [row for cell in cells for row in build_batch_rows(cell)]

    analysis = {
        "schema": "easyedit.alphaedit.strength-neutral-barrier.sequential-analysis.v1",
        "scope": {
            "stage": EXPECTED_STAGE,
            "job_id": args.job_id,
            "cells": EXPECTED_CELLS,
            "batches_per_cell": EXPECTED_BATCHES,
            "requests_per_cell": EXPECTED_REQUESTS,
            "edit_request_attempts": EXPECTED_CELLS * EXPECTED_REQUESTS,
            "scientific_promotion": False,
        },
        "source": cells[0]["result"]["source"],
        "shared_identity": {
            key: cells[0]["result"][key]
            for key in ("model", "tokenizer", "dataset", "projector", "hparams", "global_w0_sha256")
        },
        "terminal": {
            "technical_pass_count": EXPECTED_CELLS,
            "failure_boundary_count": 0,
            "w0_restore_pass_count": EXPECTED_CELLS,
            "non_fp32_parameter_count": 0,
        },
        "arms": arm_rows,
        "batches": batch_rows,
    }

    summary_path = report_root / "analysis-summary.json"
    arm_path = report_root / "arm-summary.csv"
    batch_path = report_root / "batch-retention.csv"
    report_path = report_root / "sequential-b10x10-detailed-factual-ko.md"
    write_json(summary_path, analysis)
    publish(arm_path, csv_bytes([key for key in arm_rows[0] if key not in {"pre", "post"}], arm_rows))
    publish(batch_path, csv_bytes(list(batch_rows[0]), batch_rows))
    publish(report_path, render_report(job_id=args.job_id, cells=cells, arm_rows=arm_rows, batch_rows=batch_rows).encode("utf-8"))

    members = [file_identity(path) for path in (summary_path, arm_path, batch_path, report_path)]
    raw_members = []
    for cell in cells:
        raw_members.append(file_identity(cell["result_path"]))
        raw_members.extend(file_identity(path) for path in cell["telemetry_paths"])
    members_root = sha256_bytes(canonical_json_bytes(members))
    raw_root = sha256_bytes(canonical_json_bytes(raw_members))
    manifest = {
        "schema": "easyedit.alphaedit.strength-neutral-barrier.analysis-manifest.v1",
        "members": members,
        "members_root_sha256": members_root,
        "raw_members": raw_members,
        "raw_members_root_sha256": raw_root,
        "analysis_script": file_identity(args.analysis_script.resolve(strict=True)),
        "experiment_source": cells[0]["result"]["source"],
    }
    manifest_path = report_root / "artifact-manifest.json"
    write_json(manifest_path, manifest)
    receipt_core = {
        "schema": "easyedit.alphaedit.strength-neutral-barrier.rooted-receipt.v1",
        "stage": EXPECTED_STAGE,
        "job_id": args.job_id,
        "terminal_valid_cells": EXPECTED_CELLS,
        "expected_cells": EXPECTED_CELLS,
        "batches_per_cell": EXPECTED_BATCHES,
        "requests_per_cell": EXPECTED_REQUESTS,
        "edit_request_attempts": EXPECTED_CELLS * EXPECTED_REQUESTS,
        "failure_boundary_count": 0,
        "full_fp32_pass": True,
        "w0_restore_pass_count": EXPECTED_CELLS,
        "scientific_promotion": False,
        "members_root_sha256": members_root,
        "raw_members_root_sha256": raw_root,
        "manifest": file_identity(manifest_path),
    }
    receipt = dict(receipt_core)
    receipt["receipt_identity_sha256"] = sha256_bytes(canonical_json_bytes(receipt_core))
    write_json(report_root / "rooted-receipt.json", receipt)


if __name__ == "__main__":
    main()
