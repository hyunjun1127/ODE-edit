#!/usr/bin/env python3
"""Bind atomic and sequential SNB reports into one factual comparison package."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from project.run_scripts.alphaedit_strength_neutral_barrier.build_atomic_report import (
    assert_safe_absent_directory,
    canonical_json_bytes,
    file_identity,
    fmt,
    pct,
    publish,
    read_json,
    sha256_bytes,
    validate_markdown_tables,
    write_json,
)


def locate_report(root: Path, stage: str) -> Path:
    name = (
        "atomic-b10-detailed-factual-ko.md"
        if stage == "atomic-b10"
        else "sequential-b10x10-detailed-factual-ko.md"
    )
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing canonical report: {path}")
    return path


def load_package(
    root: Path, stage: str, expected_head: str, expected_model_path_fragment: str
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    summary_path = root / "analysis-summary.json"
    manifest_path = root / "artifact-manifest.json"
    receipt_path = root / "rooted-receipt.json"
    report_path = locate_report(root, stage)
    for path in (summary_path, manifest_path, receipt_path):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing regular package member: {path}")
    summary = read_json(summary_path)
    receipt = read_json(receipt_path)
    if summary["scope"]["stage"] != stage or receipt["stage"] != stage:
        raise RuntimeError(f"stage mismatch: {root}")
    if summary["source"]["head"] != expected_head:
        raise RuntimeError(f"source HEAD mismatch: {root}")
    if expected_model_path_fragment not in summary["shared_identity"]["model"]["path"]:
        raise RuntimeError(f"model identity mismatch: {root}")
    if summary["terminal"]["technical_pass_count"] != 7:
        raise RuntimeError(f"incomplete terminal package: {root}")
    if summary["terminal"]["w0_restore_pass_count"] != 7:
        raise RuntimeError(f"W0 restore mismatch: {root}")
    return {
        "root": root,
        "summary": summary,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
        "receipt_path": receipt_path,
        "report_path": report_path,
    }


def arm_label(arm: Mapping[str, Any]) -> str:
    return str(arm.get("label", arm.get("method")))


def locality(arm: Mapping[str, Any]) -> dict[str, Any]:
    if "locality_preservation" in arm:
        return dict(arm["locality_preservation"])
    return {"numerator": arm["loc_n"], "denominator": arm["loc_d"], "rate": arm["loc_rate"]}


def timing(arm: Mapping[str, Any]) -> tuple[float, float]:
    if "timing" in arm:
        return float(arm["timing"]["edit_core_seconds"]), float(arm["timing"]["total_seconds"])
    return float(arm["edit_core_seconds"]), float(arm["total_seconds"])


def performance_row(model: str, stage: str, arm: Mapping[str, Any]) -> dict[str, Any]:
    post = arm["post"]
    loc = locality(arm)
    edit, total = timing(arm)
    return {
        "model": model,
        "stage": stage,
        "method": arm_label(arm),
        "rewrite": post["rewrite_target_new"]["nll"],
        "rephrase": post["rephrase_target_new"]["nll"],
        "eff": post["rewrite_preference_success"],
        "gen": post["rephrase_preference_success"],
        "strict_gen": post["strict_rephrase_preference_success"],
        "loc": loc,
        "edit_core_seconds": edit,
        "total_seconds": total,
    }


def paired_arm_delta(rows: Sequence[Mapping[str, Any]], arm_index: int) -> dict[str, float]:
    official = rows[0]
    arm = rows[arm_index]
    return {
        "rewrite_mean_delta": float(arm["rewrite"]["mean"]) - float(official["rewrite"]["mean"]),
        "rephrase_mean_delta": float(arm["rephrase"]["mean"]) - float(official["rephrase"]["mean"]),
        "eff_rate_delta": float(arm["eff"]["rate"]) - float(official["eff"]["rate"]),
        "gen_rate_delta": float(arm["gen"]["rate"]) - float(official["gen"]["rate"]),
        "loc_rate_delta": float(arm["loc"]["rate"]) - float(official["loc"]["rate"]),
    }


def render(packages: Mapping[str, Mapping[str, Any]], rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    lines = [
        "# AlphaEdit strength-neutral barrier — G0/atomic/sequential 통합 사실 보고서",
        "",
        "## 범위",
        "",
        "- 두 모델 모두 repaired G0-A/B/C와 actual barrier gate를 통과한 뒤 atomic B1 → atomic B10 → sequential B10×10 순서로 실행했다.",
        "- Atomic은 하나의 B10, sequential은 B10 10회/100 requests다. 두 denominator를 합치거나 직접 paired causal comparison으로 취급하지 않는다.",
        "- Split N=2/4/8은 Official endpoint parity control이고, strength-neutral barrier N=2/4/8이 실험 arm이다.",
        "- FULL-FP32, pinned stock EasyEdit, sample/order/tolerance/science equation은 고정됐다. scientific_promotion=false.",
        "",
    ]
    for model in ("Llama3-8B-Instruct", "Qwen2.5-7B-Instruct"):
        for stage in ("atomic-b10", "sequential-b10x10"):
            key = f"{model}:{stage}"
            stage_rows = rows[key]
            denominator = 10 if stage == "atomic-b10" else 100
            title = "atomic B10" if stage == "atomic-b10" else "sequential B10×10"
            lines.extend(
                [
                    f"## {model} — {title}",
                    "",
                    f"Final-W denominator: {denominator} requests/arm.",
                    "",
                    "| method | rewrite new NLL mean/median/p90/max | Eff | rephrase new NLL mean/median/p90/max | Gen | strict Gen | LOC | edit-core s | total s |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in stage_rows:
                lines.append(
                    f"| {row['method']} | {fmt(row['rewrite']['mean'],4)}/{fmt(row['rewrite']['median'],4)}/{fmt(row['rewrite']['p90'],4)}/{fmt(row['rewrite']['max'],4)} | "
                    f"{row['eff']['numerator']}/{row['eff']['denominator']} ({pct(row['eff']['rate'])}) | "
                    f"{fmt(row['rephrase']['mean'],4)}/{fmt(row['rephrase']['median'],4)}/{fmt(row['rephrase']['p90'],4)}/{fmt(row['rephrase']['max'],4)} | "
                    f"{row['gen']['numerator']}/{row['gen']['denominator']} ({pct(row['gen']['rate'])}) | "
                    f"{row['strict_gen']['numerator']}/{row['strict_gen']['denominator']} ({pct(row['strict_gen']['rate'])}) | "
                    f"{row['loc']['numerator']}/{row['loc']['denominator']} ({pct(row['loc']['rate'])}) | "
                    f"{fmt(row['edit_core_seconds'],2)} | {fmt(row['total_seconds'],2)} |"
                )
            lines.extend(
                [
                    "",
                    "### Barrier − Official descriptive deltas",
                    "",
                    "| barrier | rewrite mean Δ | rephrase mean Δ | Eff Δpp | Gen Δpp | LOC Δpp |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for index in range(4, 7):
                delta = paired_arm_delta(stage_rows, index)
                lines.append(
                    f"| {stage_rows[index]['method']} | {fmt(delta['rewrite_mean_delta'])} | {fmt(delta['rephrase_mean_delta'])} | "
                    f"{100*delta['eff_rate_delta']:.2f} | {100*delta['gen_rate_delta']:.2f} | {100*delta['loc_rate_delta']:.2f} |"
                )
            lines.append("")
    lines.extend(
        [
            "## 해석 경계",
            "",
            "1. Barrier telemetry의 작은 1차 residual은 finite-step 보장 또는 전역 locality 보장이 아니다.",
            "2. Atomic과 sequential은 request denominator와 state trajectory가 다르므로 cross-stage 차이는 descriptive다.",
            "3. Sequential batch별 immediate→final retention은 각 모델 sequential 상세 보고서에 기록되어 있다.",
            "4. 기술 제외 lineage는 성공 과학 denominator에 포함하지 않았다.",
            "",
            "## 결속된 상세 보고서",
            "",
        ]
    )
    for key, package in packages.items():
        lines.append(f"- `{key}`: `{package['report_path']}`")
    text = "\n".join(lines) + "\n"
    validate_markdown_tables(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-atomic", type=Path, required=True)
    parser.add_argument("--llama-sequential", type=Path, required=True)
    parser.add_argument("--qwen-atomic", type=Path, required=True)
    parser.add_argument("--qwen-sequential", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--analysis-script", type=Path, required=True)
    args = parser.parse_args()

    specs = {
        "Llama3-8B-Instruct:atomic-b10": (args.llama_atomic, "atomic-b10", "Meta-Llama-3-8B-Instruct"),
        "Llama3-8B-Instruct:sequential-b10x10": (args.llama_sequential, "sequential-b10x10", "Meta-Llama-3-8B-Instruct"),
        "Qwen2.5-7B-Instruct:atomic-b10": (args.qwen_atomic, "atomic-b10", "Qwen2.5-7B-Instruct"),
        "Qwen2.5-7B-Instruct:sequential-b10x10": (args.qwen_sequential, "sequential-b10x10", "Qwen2.5-7B-Instruct"),
    }
    packages = {
        key: load_package(path, stage, args.expected_head, model_fragment)
        for key, (path, stage, model_fragment) in specs.items()
    }
    rows = {
        key: [performance_row(key.split(":", 1)[0], specs[key][1], arm) for arm in package["summary"]["arms"]]
        for key, package in packages.items()
    }
    if any(len(value) != 7 for value in rows.values()):
        raise RuntimeError("all integrated stages must have 7 arms")

    report_root = args.report_root
    assert_safe_absent_directory(report_root)
    summary_path = report_root / "integrated-analysis-summary.json"
    report_path = report_root / "integrated-factual-ko.md"
    summary = {
        "schema": "easyedit.alphaedit.strength-neutral-barrier.integrated-analysis.v1",
        "source_head": args.expected_head,
        "scientific_promotion": False,
        "stages": rows,
    }
    write_json(summary_path, summary)
    publish(report_path, render(packages, rows).encode("utf-8"))
    members = [file_identity(summary_path), file_identity(report_path)]
    external_members = []
    for package in packages.values():
        external_members.extend(
            file_identity(package[key])
            for key in ("summary_path", "manifest_path", "receipt_path", "report_path")
        )
    members_root = sha256_bytes(canonical_json_bytes(members))
    external_root = sha256_bytes(canonical_json_bytes(external_members))
    manifest = {
        "schema": "easyedit.alphaedit.strength-neutral-barrier.integrated-manifest.v1",
        "members": members,
        "members_root_sha256": members_root,
        "external_members": external_members,
        "external_members_root_sha256": external_root,
        "analysis_script": file_identity(args.analysis_script.resolve(strict=True)),
    }
    manifest_path = report_root / "artifact-manifest.json"
    write_json(manifest_path, manifest)
    receipt_core = {
        "schema": "easyedit.alphaedit.strength-neutral-barrier.integrated-receipt.v1",
        "source_head": args.expected_head,
        "models": 2,
        "stages": 4,
        "terminal_valid_cells": 28,
        "failure_boundary_count": 0,
        "scientific_promotion": False,
        "members_root_sha256": members_root,
        "external_members_root_sha256": external_root,
        "manifest": file_identity(manifest_path),
    }
    receipt = dict(receipt_core)
    receipt["receipt_identity_sha256"] = sha256_bytes(canonical_json_bytes(receipt_core))
    write_json(report_root / "rooted-receipt.json", receipt)


if __name__ == "__main__":
    main()
