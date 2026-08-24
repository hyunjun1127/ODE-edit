#!/usr/bin/env python3
"""Create v2 sequential/integrated report packages with final-W baseline NLL."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.p1r54_native_sequential_w_nll_analysis import (
    METHOD_ALPHA,
    METHOD_FZ,
    METHOD_MEMIT,
    build_manifest,
    detailed_rows,
    sha256_file,
    summarize_fz_final_w10,
    summarize_native_final_w10,
    update_headline_csv,
    update_report_markdown,
    validate_native_transfer,
    write_csv,
    write_json,
)


SEQUENTIAL_V1 = Path(
    "experiment-reports/servers/server4/"
    "p1r54-realization-reset-sequential-10xb100-2026-08-24-v1"
)
INTEGRATED_V1 = Path(
    "experiment-reports/servers/server4/"
    "p1r54-realization-reset-integrated-2026-08-24-v1"
)
SEQUENTIAL_V2 = Path(
    "experiment-reports/servers/server4/"
    "p1r54-realization-reset-sequential-10xb100-2026-08-25-v2"
)
INTEGRATED_V2 = Path(
    "experiment-reports/servers/server4/"
    "p1r54-realization-reset-integrated-2026-08-25-v2"
)


def _copy_members(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise ODEBFContractError(f"report destination is create-once: {destination}")
    destination.mkdir(mode=0o700, parents=True)
    for path in source.iterdir():
        if path.is_file() and path.name not in ("analysis-manifest.json", "rooted-receipt.json"):
            (destination / path.name).write_bytes(path.read_bytes())


def _augment_analysis(path: Path, summaries: dict[str, dict[str, Any]], source_head: str, source_tree: str) -> None:
    value = json.loads(path.read_bytes())
    value.pop("identity_sha256", None)
    value["baseline_final_W10_NLL_backfill"] = summaries
    value["baseline_W_NLL_status"] = "RECORDED_EXACT_SAME_SAMPLE_1000_REQUESTS"
    value["baseline_W_NLL_imputation_count"] = 0
    value["analysis_revision_source"] = source_head
    value["analysis_revision_tree"] = source_tree
    value["identity_sha256"] = canonical_hash(value)
    write_json(path, value)


def _augment_external_inputs(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    value = json.loads(path.read_bytes())
    value.pop("identity_sha256", None)
    value["baseline_final_W10_NLL_backfill"] = [
        {
            "method": method,
            "source_path": summaries[method]["source_path"],
            "source_sha256": summaries[method]["source_sha256"],
            "source_bytes": summaries[method]["source_bytes"],
            "request_denominator": 1000,
            "imputation_count": 0,
        }
        for method in (METHOD_ALPHA, METHOD_MEMIT, METHOD_FZ)
    ]
    value["identity_sha256"] = canonical_hash(value)
    write_json(path, value)


def _write_details(package: Path, summaries: dict[str, dict[str, Any]]) -> None:
    rows = detailed_rows(summaries)
    write_csv(package / "baseline-final-w10-nll.csv", rows)
    payload: dict[str, Any] = {
        "schema": "p1r54-native-sequential-final-w10-nll-backfill/v1",
        "status": "EXACT_SAME_SAMPLE_COMPLETE",
        "methods": summaries,
        "request_denominator_per_method": 1000,
        "imputation_count": 0,
        "scientific_promotion": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    write_json(package / "baseline-final-w10-nll.json", payload)


def _external_inputs(
    summaries: dict[str, dict[str, Any]], transfer: dict[str, Any]
) -> list[dict[str, Any]]:
    inputs = [
        {
            "method": method,
            "path": summaries[method]["source_path"],
            "sha256": summaries[method]["source_sha256"],
            "bytes": summaries[method]["source_bytes"],
            "request_denominator": 1000,
            "raw_mutation": 0,
        }
        for method in (METHOD_ALPHA, METHOD_MEMIT, METHOD_FZ)
    ]
    inputs.append({
        "method": "SH1-NATIVE-FINAL-W10-TRANSFER-PROVENANCE",
        "path": transfer["root"],
        "sha256": transfer["identity_sha256"],
        "stream_root": transfer["stream_root"],
        "order_root": transfer["order_root"],
        "member_sha256": transfer["member_sha256"],
        "request_denominator_per_native_method": 1000,
        "raw_mutation": 0,
    })
    return inputs


def build_packages(
    *,
    source_head: str,
    transfer_root: Path,
    fz_final_w10: Path,
) -> dict[str, Any]:
    observed_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    observed_tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True).strip()
    if observed_head != source_head or subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
    ):
        raise ODEBFContractError("analysis source is not clean/frozen")
    transfer = validate_native_transfer(transfer_root)
    summaries = {
        METHOD_ALPHA: summarize_native_final_w10(transfer_root / "alphaedit", method=METHOD_ALPHA),
        METHOD_MEMIT: summarize_native_final_w10(transfer_root / "memit", method=METHOD_MEMIT),
        METHOD_FZ: summarize_fz_final_w10(fz_final_w10),
    }
    external = _external_inputs(summaries, transfer)
    outputs = []
    for source_relative, destination_relative, headline_name, report_heading in (
        (SEQUENTIAL_V1, SEQUENTIAL_V2, "headline.csv", None),
        (INTEGRATED_V1, INTEGRATED_V2, "sequential-headline.csv", "## Sequential 최종 표"),
    ):
        source = REPO_ROOT / source_relative
        destination = REPO_ROOT / destination_relative
        _copy_members(source, destination)
        headline_path = destination / headline_name
        headline_path.write_text(
            update_headline_csv(headline_path.read_text(encoding="utf-8"), summaries),
            encoding="utf-8",
        )
        report_path = destination / "report-ko.md"
        report_path.write_text(
            update_report_markdown(
                report_path.read_text(encoding="utf-8"),
                summaries,
                start_heading=report_heading,
            ),
            encoding="utf-8",
        )
        _augment_analysis(destination / "analysis.json", summaries, observed_head, observed_tree)
        if (destination / "external-inputs.json").is_file():
            _augment_external_inputs(destination / "external-inputs.json", summaries)
        _write_details(destination, summaries)
        manifest, receipt = build_manifest(
            destination,
            source_head=observed_head,
            source_tree=observed_tree,
            external_inputs=external,
        )
        outputs.append({
            "package": str(destination),
            "report_sha256": receipt["report_sha256"],
            "manifest_sha256": receipt["manifest_sha256"],
            "receipt_sha256": sha256_file(destination / "rooted-receipt.json"),
            "receipt_identity": receipt["identity_sha256"],
            "member_root": manifest["member_root"],
        })
    return {
        "status": "PASS",
        "source_head": observed_head,
        "source_tree": observed_tree,
        "native_transfer": transfer,
        "outputs": outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--transfer-root", required=True, type=Path)
    parser.add_argument("--fz-final-w10", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_packages(
        source_head=args.source_head,
        transfer_root=args.transfer_root,
        fz_final_w10=args.fz_final_w10,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
