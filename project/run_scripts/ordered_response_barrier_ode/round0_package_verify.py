"""Full-member verifier for the ORBODE round-0 analysis package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .round0_analysis_contracts import (
    EXPECTED_DYNAMIC_STEPS,
    EXPECTED_REQUEST_STEPS,
    INSTRUCTION_ID,
    AnalysisBoundary,
    canonical_hash,
    regular_file,
    sha256_file,
    verify_canonical_identity,
)


EXPECTED_TABLE_ROWS = {
    "core-performance-summary.csv": 24,
    "nll-distribution-summary.csv": 120,
    "prompt-nll.csv.gz": 38_400,
    "request-endpoint-metrics.csv.gz": 2_400,
    "paired-official-deltas.csv": 112,
    "method-contrast-deltas.csv": 84,
    "mechanism-arm-summary.csv": 20,
    "mechanism-step-summary.csv": EXPECTED_DYNAMIC_STEPS,
    "mechanism-request-step.csv.gz": EXPECTED_REQUEST_STEPS,
    "mechanism-arm-aggregate.csv": 16,
    "layer-update-summary.csv": 100,
    "compute-summary.csv": 20,
    "runtime-preamble-summary.csv": 4,
    "derived-orbhit-summary.csv": 4,
    "lineage-inventory.csv": 21,
}


def _object(path: Path) -> dict[str, Any]:
    regular_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AnalysisBoundary(f"JSON object expected: {path}")
    return payload


def _identity(payload: Mapping[str, Any]) -> None:
    verify_canonical_identity(payload)


def verify(repo: Path, package: Path) -> dict[str, Any]:
    if package.is_symlink() or not package.is_dir():
        raise AnalysisBoundary("package must be a non-symlink directory")
    manifest_path = package / "analysis-manifest.json"
    receipt_path = package / "rooted-analysis-receipt.json"
    manifest, receipt = _object(manifest_path), _object(receipt_path)
    _identity(manifest)
    _identity(receipt)
    if manifest.get("instruction_id") != INSTRUCTION_ID or receipt.get("instruction_id") != INSTRUCTION_ID:
        raise AnalysisBoundary("instruction identity differs")
    if receipt.get("manifest", {}).get("sha256") != sha256_file(manifest_path):
        raise AnalysisBoundary("manifest SHA binding differs")
    if receipt.get("manifest", {}).get("identity_sha256") != manifest.get("identity_sha256"):
        raise AnalysisBoundary("manifest canonical identity binding differs")
    if receipt.get("member_root") != manifest.get("member_root") or canonical_hash(manifest["members"]) != manifest["member_root"]:
        raise AnalysisBoundary("member root differs")
    expected_names: list[str] = []
    for row in manifest["members"]:
        relative = Path(row["path"])
        if relative.is_absolute() or len(relative.parts) != 1:
            raise AnalysisBoundary(f"invalid package member path: {relative}")
        path = package / relative
        info = regular_file(path)
        if info.st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise AnalysisBoundary(f"member identity differs: {relative}")
        expected_names.append(relative.name)
    actual_names = sorted(
        path.name for path in package.iterdir()
        if path.name not in {"analysis-manifest.json", "rooted-analysis-receipt.json"}
    )
    if sorted(expected_names) != actual_names:
        raise AnalysisBoundary("package member inventory differs")
    for row in manifest["inputs"]:
        path = Path(row["path"])
        info = regular_file(path)
        if info.st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise AnalysisBoundary(f"sealed input changed: {path}")
    for row in manifest["analysis_sources"]:
        path = repo / row["path"]
        info = regular_file(path)
        if info.st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise AnalysisBoundary(f"analysis source changed: {path}")
    if manifest.get("input_root") != canonical_hash(manifest["inputs"]):
        raise AnalysisBoundary("input root differs")
    if manifest.get("analysis_source_root") != canonical_hash(manifest["analysis_sources"]):
        raise AnalysisBoundary("analysis source root differs")
    for name, count in EXPECTED_TABLE_ROWS.items():
        if len(pd.read_csv(package / name)) != count:
            raise AnalysisBoundary(f"table row denominator differs: {name}")
    plots = _object(package / "plot-reproduction.json")
    _identity(plots)
    if plots.get("status") != "BYTE_STABLE_REPRODUCTION_PASS" or len(plots.get("outputs", [])) != 5:
        raise AnalysisBoundary("plot reproduction status/count differs")
    if not all(row.get("byte_stable") for row in plots["outputs"]):
        raise AnalysisBoundary("plot byte-stability differs")
    scheduler = _object(package / "scheduler-terminal.json")
    _identity(scheduler)
    summary = _object(package / "analysis-summary.json")
    _identity(summary)
    audit = _object(package / "analysis-audit.json")
    _identity(audit)
    gates = manifest.get("denominators_and_gates", {})
    required_zero = (
        "technical_failure_count", "scientific_failure_count", "imputation_count",
        "dynamic_z_recompute_count", "forbidden_controller_evaluator_access_count",
        "retry_count", "nonfinite_count", "method_state_content_mismatch_count",
        "inner_history_append_count", "inner_cache_mutation_count",
        "runtime_preamble_failure_count", "official_wrapper_direct_fidelity_failure_count",
        "analysis_only_model_action_count", "analysis_only_gpu_action_count",
        "analysis_only_slurm_submit_count", "remaining_nine_submit_count",
        "plot_byte_reproduction_failure_count", "task_owned_active_job_count",
    )
    for field in required_zero:
        if gates.get(field) != 0:
            raise AnalysisBoundary(f"hard zero gate differs: {field}={gates.get(field)!r}")
    if gates.get("canonical_cell_count") != 4 or gates.get("primary_endpoint_count") != 2000:
        raise AnalysisBoundary("canonical denominator gate differs")
    if gates.get("scientific_promotion") is not False:
        raise AnalysisBoundary("scientific promotion differs")
    return {
        "status": "FULL_PACKAGE_REHASH_PASS",
        "package": str(package),
        "member_count": len(manifest["members"]),
        "manifest_sha256": sha256_file(manifest_path),
        "receipt_sha256": sha256_file(receipt_path),
        "receipt_identity": receipt["identity_sha256"],
        "member_root": manifest["member_root"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.repo.resolve(), args.package.resolve()), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
