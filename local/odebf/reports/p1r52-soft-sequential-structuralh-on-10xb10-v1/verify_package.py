#!/usr/bin/env python3
"""Fresh-process, report-only rehash and completeness verifier."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    base_manifest = read(OUT / "p1r52-structuralh-on-report-manifest.json")
    v2_manifest = read(OUT / "p1r52-structuralh-on-report-manifest-v2.json")
    receipt = read(OUT / "analysis-receipt-v2.json")
    inputs_ok = all(Path(x["path"]).is_file() and sha(Path(x["path"])) == x["sha256"] and Path(x["path"]).stat().st_size == x["bytes"] for x in base_manifest["input_files"])
    base_outputs_ok = all((OUT / x["name"]).is_file() and sha(OUT / x["name"]) == x["sha256"] and (OUT / x["name"]).stat().st_size == x["bytes"] for x in base_manifest["generated_files"])
    v2_report = Path(v2_manifest["canonical_report"])
    if not v2_report.is_absolute():
        v2_report = OUT / v2_report
    v2_ok = v2_report.is_file() and sha(v2_report) == receipt["report_sha256"] and v2_report.stat().st_size == receipt["report_bytes"] and len(v2_report.read_text().splitlines()) == receipt["report_lines"]
    agg = read(OUT / "p1r52-structuralh-on-aggregate-phase-table.json")
    tables = {
        "checkpoint": read(OUT / "p1r52-structuralh-on-checkpoint-phase-table.json")["row_count"],
        "paired": read(OUT / "p1r52-structuralh-on-paired-native-checkpoint-table.json")["row_count"],
        "per_request": read(OUT / "p1r52-structuralh-on-per-request-lifetime-table.json")["row_count"],
        "routing": read(OUT / "p1r52-structuralh-on-routing-history-table.json")["row_count"],
        "transaction": read(OUT / "p1r52-structuralh-on-transaction-table.json")["row_count"],
        "aggregate": agg["row_count"],
    }
    expected = {"checkpoint": 62, "paired": 31, "per_request": 100, "routing": 80, "transaction": 10, "aggregate": 3}
    rows_ok = tables == expected
    text = v2_report.read_text()
    required_sections = (
        "B1–B10 절대 pre → post → final-W10 표",
        "B100 final-W10 절대값 및 Native 차이",
        "B5 동일 배치 관측",
        "History·anchor·lifetime 및 낮은 성능 cohort",
        "Compute ledger",
        "10개 true entry-pre·immediate post·final B100 집계",
    )
    section_ok = all(x in text for x in required_sections)
    no_runtime_actions = receipt["analysis_actions"] == {"model": 0, "evaluator": 0, "gpu": 0, "slurm": 0, "source_edit": 0, "result_mutation": 0}
    status = "PASS" if all((inputs_ok, base_outputs_ok, v2_ok, rows_ok, section_ok, no_runtime_actions)) else "FAIL"
    out = OUT / "independent-rehash-review.json"
    if out.exists():
        raise RuntimeError(f"create-once exists: {out}")
    payload = {
        "schema": "p1r52-structuralh-on/independent-rehash-review/v1",
        "status": status,
        "fresh_process": True,
        "input_rehash_pass": inputs_ok,
        "base_output_rehash_pass": base_outputs_ok,
        "v2_report_rehash_pass": v2_ok,
        "row_count_pass": rows_ok,
        "row_counts": tables,
        "required_section_pass": section_ok,
        "analysis_runtime_action_zero_pass": no_runtime_actions,
        "canonical_report": str(v2_report),
        "canonical_report_sha256": sha(v2_report),
        "canonical_report_bytes": v2_report.stat().st_size,
        "canonical_report_lines": len(text.splitlines()),
        "source_head": base_manifest["source_head"],
        "native_source_head": base_manifest["native_source_head"],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.chmod(out, 0o600)
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
