"""Full-member rehash verifier for the Phase A canonical package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .contracts import (
    AnalysisBoundary,
    EXPECTED_ACTION_ROWS,
    EXPECTED_LAYER_ROWS,
    EXPECTED_REQUEST_ROWS,
    INSTRUCTION_ID,
    canonical_hash,
    regular_file,
    sha256_file,
)


def _object(path: Path) -> dict[str, Any]:
    regular_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AnalysisBoundary(f"JSON object expected: {path}")
    return payload


def _identity(payload: Mapping[str, Any], label: str) -> None:
    stripped = dict(payload)
    expected = stripped.pop("identity_sha256", None)
    if expected != canonical_hash(stripped):
        raise AnalysisBoundary(f"canonical identity differs: {label}")


def verify(repo: Path, package: Path) -> dict[str, Any]:
    if not package.is_dir() or package.is_symlink():
        raise AnalysisBoundary("package is not a regular directory")
    manifest_path = package / "analysis-manifest.json"
    receipt_path = package / "rooted-analysis-receipt.json"
    manifest = _object(manifest_path)
    receipt = _object(receipt_path)
    _identity(manifest, "manifest")
    _identity(receipt, "receipt")
    if manifest.get("instruction_id") != INSTRUCTION_ID or receipt.get("instruction_id") != INSTRUCTION_ID:
        raise AnalysisBoundary("instruction identity differs")
    if receipt.get("manifest", {}).get("sha256") != sha256_file(manifest_path):
        raise AnalysisBoundary("manifest file SHA binding differs")
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
            raise AnalysisBoundary(f"package member identity differs: {relative}")
        expected_names.append(relative.name)
    actual_names = sorted(
        path.name
        for path in package.iterdir()
        if path.name not in {"analysis-manifest.json", "rooted-analysis-receipt.json"}
    )
    if sorted(expected_names) != actual_names:
        raise AnalysisBoundary("package member inventory differs")
    for row in manifest["inputs"]:
        path = repo / row["path"]
        info = regular_file(path)
        if info.st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise AnalysisBoundary(f"sealed input changed: {row['path']}")
    for row in manifest["analysis_sources"]:
        path = repo / row["path"]
        info = regular_file(path)
        if info.st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise AnalysisBoundary(f"analysis source changed: {row['path']}")
    if manifest.get("input_root") != canonical_hash(manifest["inputs"]):
        raise AnalysisBoundary("input root differs")
    if manifest.get("analysis_source_root") != canonical_hash(manifest["analysis_sources"]):
        raise AnalysisBoundary("analysis source root differs")
    if len(pd.read_csv(package / "request-layer-debt.csv.gz")) != EXPECTED_LAYER_ROWS:
        raise AnalysisBoundary("derived layer row count differs")
    if len(pd.read_csv(package / "request-debt-endpoint.csv.gz")) != EXPECTED_REQUEST_ROWS:
        raise AnalysisBoundary("derived request row count differs")
    if len(pd.read_csv(package / "batch-layer-action-proxy.csv")) != EXPECTED_ACTION_ROWS:
        raise AnalysisBoundary("derived action row count differs")
    plots = _object(package / "plot-reproduction.json")
    _identity(plots, "plot reproduction")
    if plots.get("status") != "BYTE_STABLE_REPRODUCTION_PASS" or not all(row.get("byte_stable") for row in plots["outputs"]):
        raise AnalysisBoundary("plot reproduction gate differs")
    gates = manifest.get("denominators_and_gates", {})
    if (
        gates.get("debt_decomposition_identity_failure_count") != 0
        or gates.get("debt_parallel_native_identity_failure_count") != 0
        or gates.get("debt_native_identity_failure_count") != 0
        or gates.get("core_nonfinite_count") != 0
        or gates.get("layer_to_endpoint_join_missing") != 0
        or gates.get("layer_to_action_join_missing") != 0
        or gates.get("imputation_count") != 0
        or gates.get("interpolation_count") != 0
        or gates.get("scientific_promotion") is not False
    ):
        raise AnalysisBoundary("hard gate differs")
    return {
        "status": "FULL_PACKAGE_REHASH_PASS",
        "package": str(package),
        "member_count": len(manifest["members"]),
        "input_count": len(manifest["inputs"]),
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
