#!/usr/bin/env python3
"""Append-only raw-free reconciliation for the V1 report-root labels."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


OUT = Path(__file__).resolve().parent
V1 = OUT.parent / "p2r4-phaseb-clamp-on-off-writer-v1"
STEM = "p2r4-phaseb-clamp-on-off-writer-v2-root-reconciliation"


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    return digest(path.read_bytes())


def entry(path: Path) -> dict[str, object]:
    return {"path": path.name, "mode": f"{path.stat().st_mode & 0o777:04o}", "bytes": path.stat().st_size, "sha256": file_digest(path)}


def write_once(path: Path, data: bytes) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return digest(data)


def write_json_once(path: Path, value: object) -> str:
    return write_once(path, canonical(value) + b"\n")


def main() -> None:
    manifest_path = V1 / "p2r4-phaseb-clamp-on-off-writer-v1-analysis-manifest.json"
    receipt_path = V1 / "p2r4-phaseb-clamp-on-off-writer-v1-analysis-receipt.json"
    report_path = V1 / "p2r4-phaseb-clamp-on-off-writer-v1-terminal-report-ko.md"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    report_root_line = next(line for line in report.splitlines() if "analysis root:" in line)
    printed_root = report_root_line.split("`")[1]
    excluded = {manifest_path.name, receipt_path.name, report_path.name, "build_p2r4_phaseb_report.py"}
    core_entries = [entry(path) for path in sorted(V1.iterdir()) if path.is_file() and path.name not in excluded]
    core_root = digest(canonical(core_entries))
    final_entries = manifest["files"]
    final_root = digest(canonical(final_entries))
    if printed_root != core_root:
        raise ValueError("V1 printed core-table root does not rehash")
    if manifest["analysis_root"] != final_root or receipt["analysis_root"] != final_root:
        raise ValueError("V1 final package root does not rehash")
    receipt_without_root = dict(receipt)
    receipt_root = receipt_without_root.pop("receipt_root")
    if receipt_root != digest(canonical(receipt_without_root)):
        raise ValueError("V1 receipt root does not rehash")
    reconciliation = {
        "schema": "ode-edit-s05-p2r4-phaseb-v2-root-reconciliation/v1",
        "status": "PASS",
        "v1_report_path": str(report_path),
        "v1_report_sha256": file_digest(report_path),
        "v1_manifest_path": str(manifest_path),
        "v1_manifest_sha256": file_digest(manifest_path),
        "v1_receipt_path": str(receipt_path),
        "v1_receipt_sha256": file_digest(receipt_path),
        "report_embedded_label": "analysis root",
        "report_embedded_root": printed_root,
        "report_embedded_root_correct_scope": "pre-report core table set only; 12 CSV/JSON table files; excludes V1 generator, Markdown report, manifest and receipt",
        "pre_report_core_table_entry_count": len(core_entries),
        "pre_report_core_table_root_recomputed": core_root,
        "canonical_package_root": final_root,
        "canonical_package_root_correct_scope": "V1 manifest files set; 14 files: V1 generator, 12 CSV/JSON tables, and the Markdown report; excludes V1 manifest and receipt to avoid self-reference",
        "canonical_package_entry_count": len(final_entries),
        "v1_receipt_root": receipt_root,
        "v1_report_or_table_overwrite_delete_mutation": 0,
        "source_model_gpu_slurm_evaluator_action": 0,
        "scientific_promotion": False,
    }
    md = "\n".join([
        "# P2R4 Phase-B V1 root-label reconciliation (factual)",
        "",
        "- status: `PASS`",
        f"- V1 report SHA256: `{reconciliation['v1_report_sha256']}`",
        f"- V1 manifest SHA256: `{reconciliation['v1_manifest_sha256']}`",
        f"- V1 receipt SHA256: `{reconciliation['v1_receipt_sha256']}`",
        f"- V1 report line label `analysis root` = `{printed_root}`. Its verified scope is the 12 pre-report CSV/JSON table files only.",
        f"- canonical V1 package analysis_root = `{final_root}`. Its verified scope is the 14 V1 manifest entries (generator, 12 tables, Markdown report).",
        f"- V1 receipt_root = `{receipt_root}`.",
        "- This append-only reconciliation does not alter V1 report/tables/manifest/receipt or any source/result/job artifact. scientific_promotion=false.",
        "",
    ])
    write_json_once(OUT / f"{STEM}.json", reconciliation)
    write_once(OUT / f"{STEM}.md", md.encode("utf-8"))
    entries = [entry(path) for path in sorted(OUT.iterdir()) if path.is_file()]
    package_root = digest(canonical(entries))
    out_manifest = {
        "schema": "ode-edit-s05-p2r4-phaseb-v2-root-reconciliation-manifest/v1",
        "files": entries,
        "package_root": package_root,
        "source_model_gpu_slurm_evaluator_action": 0,
    }
    manifest_sha = write_json_once(OUT / f"{STEM}-manifest.json", out_manifest)
    out_receipt = {
        "schema": "ode-edit-s05-p2r4-phaseb-v2-root-reconciliation-receipt/v1",
        "reconciliation_sha256": file_digest(OUT / f"{STEM}.json"),
        "manifest_sha256": manifest_sha,
        "package_root": package_root,
        "v1_preserved": True,
        "source_model_gpu_slurm_evaluator_action": 0,
    }
    out_receipt["receipt_root"] = digest(canonical(out_receipt))
    write_json_once(OUT / f"{STEM}-receipt.json", out_receipt)
    for path in OUT.iterdir():
        if path.is_file():
            os.chmod(path, 0o600)


if __name__ == "__main__":
    main()
