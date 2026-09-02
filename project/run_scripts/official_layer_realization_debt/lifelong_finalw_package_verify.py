"""Independent full-member verifier for the canonical final-W v4 package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from project.run_scripts.ode_bf.contracts import canonical_hash

from .lifelong_finalw_contracts import FinalWeightBoundary, INSTRUCTION_ID
from .lifelong_finalw_evaluation import sha256_file


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalWeightBoundary(f"JSON object expected: {path}")
    return value


def _identity(value: Mapping[str, Any], label: str) -> None:
    payload = dict(value)
    expected = payload.pop("identity_sha256", None)
    if expected != canonical_hash(payload):
        raise FinalWeightBoundary(f"canonical identity differs: {label}")


def verify(root: Path) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise FinalWeightBoundary("canonical package root is not a regular directory")
    manifest_path = root / "analysis-manifest.json"
    receipt_path = root / "rooted-analysis-receipt.json"
    manifest = _object(manifest_path)
    receipt = _object(receipt_path)
    _identity(manifest, "analysis manifest")
    _identity(receipt, "rooted analysis receipt")
    if (
        manifest.get("instruction_id") != INSTRUCTION_ID
        or receipt.get("instruction_id") != INSTRUCTION_ID
        or receipt.get("manifest", {}).get("sha256") != sha256_file(manifest_path)
        or receipt.get("manifest", {}).get("identity_sha256") != manifest["identity_sha256"]
    ):
        raise FinalWeightBoundary("manifest/receipt binding differs")
    report_path = Path(receipt["report"]["path"])
    if report_path.resolve(strict=True) != (root / report_path.name).resolve(strict=True):
        raise FinalWeightBoundary("report escaped canonical package")
    if (
        receipt["report"]["sha256"] != sha256_file(report_path)
        or int(receipt["report"]["bytes"]) != report_path.stat().st_size
    ):
        raise FinalWeightBoundary("report member identity differs")
    members = manifest["members"]
    if canonical_hash(members) != manifest["member_root"] or receipt["member_root"] != manifest["member_root"]:
        raise FinalWeightBoundary("canonical package member root differs")
    expected_names = []
    for member in members:
        relative = Path(member["path"])
        if relative.is_absolute() or len(relative.parts) != 1:
            raise FinalWeightBoundary("nested or absolute package member")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FinalWeightBoundary(f"non-regular canonical member: {relative}")
        if path.stat().st_size != int(member["bytes"]) or sha256_file(path) != member["sha256"]:
            raise FinalWeightBoundary(f"canonical member identity differs: {relative}")
        expected_names.append(relative.name)
    actual_names = sorted(
        path.name for path in root.iterdir()
        if path.name not in {"analysis-manifest.json", "rooted-analysis-receipt.json"}
    )
    if sorted(expected_names) != actual_names:
        raise FinalWeightBoundary("canonical package member inventory differs")
    raw_members_path = root / "evaluation-raw-member-inventory.json"
    raw_members = json.loads(raw_members_path.read_text(encoding="utf-8"))
    if not isinstance(raw_members, list) or canonical_hash(raw_members) != manifest["raw_member_root"]:
        raise FinalWeightBoundary("external raw member root differs")
    for member in raw_members:
        path = Path(member["path"])
        if not path.is_file() or path.is_symlink():
            raise FinalWeightBoundary(f"external raw member type differs: {path}")
        if path.stat().st_size != int(member["bytes"]) or sha256_file(path) != member["sha256"]:
            raise FinalWeightBoundary(f"external raw member identity differs: {path}")
    if receipt["external_raw_member_root"] != manifest["raw_member_root"]:
        raise FinalWeightBoundary("receipt external raw root differs")
    return {
        "status": "FULL_REHASH_PASS",
        "package_root": str(root),
        "package_member_count": len(members),
        "external_raw_member_count": len(raw_members),
        "manifest_sha256": sha256_file(manifest_path),
        "receipt_sha256": sha256_file(receipt_path),
        "member_root": manifest["member_root"],
        "external_raw_member_root": manifest["raw_member_root"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.package), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
