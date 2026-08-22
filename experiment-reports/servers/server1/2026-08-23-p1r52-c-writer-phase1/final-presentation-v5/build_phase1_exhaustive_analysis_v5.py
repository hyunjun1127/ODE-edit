#!/usr/bin/env python3
"""Publish v5 with the sequential W10 endpoint as the sole lead table."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
V4 = HERE.parent / "p1r52-c-writer-phase1-full-fp32-sequential-tech-r1-exhaustive-analysis-v4"
REPORT = "p1r52-c-writer-phase1-full-fp32-sequential-tech-r1-exhaustive-factual-ko.md"
V4_REPORT_SHA256 = "0a27d5bb97eca1ee9c65bcf16650cef55b26a6dc2fa2725cfaf6e3a58197d169"
V4_RECEIPT_SHA256 = "49172572d0ef9b49903836bd94d14f8a2e377856fabcce9e90382808635f86e9"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_once(path: Path, payload: bytes) -> Path:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"create-once target exists: {path}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)
    return path


def dump_json(name: str, value: Any) -> Path:
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    return write_once(HERE / name, payload)


def pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        raise RuntimeError("invalid denominator")
    return f"{100.0 * numerator / denominator:.2f}% ({numerator}/{denominator})"


def main() -> None:
    v4_report = V4 / REPORT
    v4_receipt_path = V4 / "rooted-analysis-receipt.json"
    if sha_file(v4_report) != V4_REPORT_SHA256:
        raise RuntimeError("v4 report SHA mismatch")
    if sha_file(v4_receipt_path) != V4_RECEIPT_SHA256:
        raise RuntimeError("v4 receipt SHA mismatch")
    v4_receipt = load(v4_receipt_path)
    if v4_receipt["status"] != "TERMINAL_VALID_EXHAUSTIVE_ANALYSIS_COMPLETE_WITH_COMMON_W0_PREEDIT":
        raise RuntimeError("v4 is not terminal-valid")
    terminal_rows = load(V4 / "terminal-w10-performance-comparison.json")["rows"]
    if len(terminal_rows) != 6 or terminal_rows[0]["arm"] != "pre_edit":
        raise RuntimeError("v4 terminal table mismatch")

    old = v4_report.read_text(encoding="utf-8")
    start = old.index("## 1. 한눈에 보는 최종 edit 성능")
    end = old.index("## 2. arm과 writer 정의")
    lead = [
        "## 1. Sequential 최종 성능 — 전체 B1~B10 @ W10", "",
        "이 표가 본 sequential 실험의 대표 endpoint다. B1→B10 편집을 모두 누적한 최종 모델 `W10`으로 전체 1,000 requests를 재평가했으므로, 후속 batch 편집에 따른 망각과 간섭까지 포함한다. Pre-edit는 동일 공통 W0 참고선이다.", "",
        "|arm / endpoint|EFF|Rewrite accuracy|GEN|Strict GEN|Rephrase accuracy|LOC|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in terminal_rows:
        lead.append(
            f"|{row['label']}|{pct(row['eff_numerator'], row['eff_denominator'])}|"
            f"{pct(row['rewrite_accuracy_numerator'], row['rewrite_accuracy_denominator'])}|"
            f"{pct(row['gen_numerator'], row['gen_denominator'])}|"
            f"{pct(row['gen_strict_numerator'], row['gen_strict_denominator'])}|"
            f"{pct(row['rephrase_accuracy_numerator'], row['rephrase_accuracy_denominator'])}|"
            f"{pct(row['locality_numerator'], row['locality_denominator'])}|"
        )
    lead += [
        "",
        "배치별 `B1@W1 … B10@W10` immediate-post 값은 최종 sequential 성능표로 사용하지 않는다. 다만 z→W realization과 batch별 망각을 확인하는 진단값이므로 아래 Rewrite/Rephrase 및 forgetting 절에만 남겨 둔다.",
        "",
    ]
    report_text = old[:start] + "\n".join(lead) + "\n" + old[end:]
    if "한눈에 보는 최종 edit 성능" in report_text or "### B10 종료 후 final W10 재평가" in report_text:
        raise RuntimeError("superseded lead table survived")
    report_path = write_once(HERE / REPORT, report_text.encode())

    script = Path(__file__).resolve()
    members = [
        {"path": str(script), "bytes": script.stat().st_size, "sha256": sha_file(script)},
        {"path": str(report_path), "bytes": report_path.stat().st_size, "sha256": sha_file(report_path)},
    ]
    manifest = {
        "schema": "phase1/exhaustive-analysis-manifest/v5",
        "input_v4": {
            "report_path": str(v4_report),
            "report_sha256": V4_REPORT_SHA256,
            "receipt_path": str(v4_receipt_path),
            "receipt_sha256": V4_RECEIPT_SHA256,
            "receipt_identity_sha256": v4_receipt["identity_sha256"],
        },
        "change": "PROMOTE_FINAL_W10_TABLE_AND_REMOVE_IMMEDIATE_POST_LEAD_TABLE",
        "terminal_table_row_count": len(terminal_rows),
        "members": members,
        "member_root_sha256": canonical_sha(members),
        "scientific_promotion": False,
    }
    manifest_path = dump_json("analysis-manifest.json", manifest)
    receipt = {
        "schema": "phase1/rooted-analysis-receipt/v5",
        "status": "TERMINAL_VALID_FINAL_W10_LEAD_REPORT_COMPLETE",
        "report": {"path": str(report_path), "bytes": report_path.stat().st_size, "sha256": sha_file(report_path)},
        "manifest": {"path": str(manifest_path), "bytes": manifest_path.stat().st_size, "sha256": sha_file(manifest_path)},
        "analysis_member_root_sha256": manifest["member_root_sha256"],
        "lead_endpoint": "FINAL_W10_REEVALUATION_ALL_B1_TO_B10",
        "removed_lead_endpoint": "POOLED_IMMEDIATE_POST_W",
        "scientific_promotion": False,
    }
    receipt["identity_sha256"] = canonical_sha(receipt)
    dump_json("rooted-analysis-receipt.json", receipt)


if __name__ == "__main__":
    main()
