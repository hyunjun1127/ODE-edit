"""Independent full-member verifier for the canonical lifelong v5 package."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from project.run_scripts.ode_bf.contracts import canonical_hash

from .lifelong_v5_analysis import INSTRUCTION_ID, V3_RELATIVE, V4_RELATIVE, V5Boundary


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V5Boundary(f"JSON object expected: {path}")
    return value


def _identity(value: Mapping[str, Any], label: str) -> None:
    payload = dict(value)
    expected = payload.pop("identity_sha256", None)
    if expected != canonical_hash(payload):
        raise V5Boundary(f"canonical identity differs: {label}")


def _row_count(path: Path) -> int:
    if path.suffix == ".gz":
        handle_context = gzip.open(path, "rt", encoding="utf-8", newline="")
    else:
        handle_context = path.open("r", encoding="utf-8", newline="")
    with handle_context as handle:
        return sum(1 for _ in csv.DictReader(handle))


def verify(root: Path, repository: Path) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise V5Boundary("v5 package root is not a regular directory")
    manifest_path = root / "analysis-manifest.json"
    receipt_path = root / "rooted-analysis-receipt.json"
    manifest, receipt = _object(manifest_path), _object(receipt_path)
    _identity(manifest, "v5 manifest")
    _identity(receipt, "v5 receipt")
    if (
        manifest.get("instruction_id") != INSTRUCTION_ID
        or receipt.get("instruction_id") != INSTRUCTION_ID
        or receipt.get("manifest", {}).get("sha256") != sha256_file(manifest_path)
        or receipt.get("manifest", {}).get("identity_sha256") != manifest["identity_sha256"]
        or receipt.get("member_root") != manifest.get("member_root")
        or receipt.get("external_package_root") != manifest.get("external_package_root")
    ):
        raise V5Boundary("v5 manifest/receipt binding differs")
    members = manifest["members"]
    if canonical_hash(members) != manifest["member_root"]:
        raise V5Boundary("v5 package member root differs")
    expected_names: list[str] = []
    for member in members:
        relative = Path(member["path"])
        if relative.is_absolute() or len(relative.parts) != 1:
            raise V5Boundary("v5 member path escaped package")
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(member["bytes"])
            or sha256_file(path) != member["sha256"]
        ):
            raise V5Boundary(f"v5 member identity differs: {relative}")
        expected_names.append(relative.name)
    actual_names = sorted(
        path.name
        for path in root.iterdir()
        if path.name not in {"analysis-manifest.json", "rooted-analysis-receipt.json"}
    )
    if sorted(expected_names) != actual_names:
        raise V5Boundary("v5 member inventory differs")
    for name, rows in manifest["table_row_counts"].items():
        if _row_count(root / name) != int(rows):
            raise V5Boundary(f"v5 table row count differs: {name}")
    report_path = root / receipt["report"]["path"]
    if (
        report_path.stat().st_size != int(receipt["report"]["bytes"])
        or sha256_file(report_path) != receipt["report"]["sha256"]
    ):
        raise V5Boundary("v5 report binding differs")
    external = manifest["external_packages"]
    if canonical_hash(external) != manifest["external_package_root"]:
        raise V5Boundary("v5 external package root differs")
    expected_roots = {str(V3_RELATIVE), str(V4_RELATIVE)}
    if {entry["root"] for entry in external} != expected_roots:
        raise V5Boundary("v5 external package set differs")
    for entry in external:
        package = repository / entry["root"]
        source_manifest = _object(package / "analysis-manifest.json")
        if (
            sha256_file(package / "analysis-manifest.json") != entry["manifest_sha256"]
            or sha256_file(package / "rooted-analysis-receipt.json") != entry["receipt_sha256"]
            or source_manifest["member_root"] != entry["member_root"]
            or len(source_manifest["members"]) != int(entry["member_count"])
        ):
            raise V5Boundary(f"v5 external package binding differs: {package}")
    plot_rows = list(csv.DictReader((root / "plot-reproduction.csv").open(newline="", encoding="utf-8")))
    if len(plot_rows) != 7:
        raise V5Boundary("v5 plot denominator differs")
    for row in plot_rows:
        if (
            sha256_file(root / row["figure"]) != row["sha256"]
            or sha256_file(root / row["input_table"]) != row["input_table_sha256"]
            or sha256_file(repository / row["plot_source"]) != row["plot_source_sha256"]
            or row["missing_policy"] != "NO_INTERPOLATION_NO_IMPUTATION"
        ):
            raise V5Boundary(f"v5 plot reproduction binding differs: {row['figure']}")
    return {
        "status": "FULL_REHASH_PASS",
        "package_root": str(root),
        "package_member_count": len(members),
        "table_count": len(manifest["table_row_counts"]),
        "figure_count": len(plot_rows),
        "manifest_sha256": sha256_file(manifest_path),
        "receipt_sha256": sha256_file(receipt_path),
        "report_sha256": receipt["report"]["sha256"],
        "member_root": manifest["member_root"],
        "external_package_root": manifest["external_package_root"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.package, args.repository), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
