#!/usr/bin/env python3
"""Read-only integrity review for the P1R52 depth-extension analysis package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
INNER_PER_CASE = {"IL8-FULL": 64, "IL10-FULL": 80, "IL15-FULL": 120}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"non-regular input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    manifest = load(HERE / "analysis-manifest.json")
    receipt = load(HERE / "analysis-receipt.json")
    if manifest["identity_sha256"] != canonical_hash(
        {key: value for key, value in manifest.items() if key != "identity_sha256"}
    ):
        raise RuntimeError("manifest identity differs")
    if receipt["identity_sha256"] != canonical_hash(
        {key: value for key, value in receipt.items() if key != "identity_sha256"}
    ):
        raise RuntimeError("receipt identity differs")
    for member in manifest["members"]:
        path = Path(member["path"])
        observed = {
            "bytes": path.stat().st_size,
            "mode": oct(path.stat().st_mode & 0o777),
            "sha256": sha256_file(path),
        }
        for field, expected in observed.items():
            if member[field] != expected:
                raise RuntimeError(f"member {field} differs: {path}")
    if manifest["members_root_sha256"] != canonical_hash(manifest["members"]):
        raise RuntimeError("members root differs")
    if receipt["manifest_sha256"] != sha256_file(HERE / "analysis-manifest.json"):
        raise RuntimeError("receipt manifest SHA differs")
    report = Path(receipt["report_path"])
    if receipt["report_sha256"] != sha256_file(report):
        raise RuntimeError("receipt report SHA differs")
    cases = load(HERE / "p1r52-target-depth-il8-il10-il15-per-case.json")
    failures = load(HERE / "p1r52-target-depth-il8-il10-il15-typed-failures.json")
    case_inner_rows = sum(INNER_PER_CASE[row["depth"]] for row in cases)
    expected_rows = {
        "per_case": len(cases),
        "per_step": len(cases) * 8,
        "per_inner": case_inner_rows,
        "per_inner_request": case_inner_rows * 10,
        "per_request": len(cases) * 10,
        "paired_il3_per_case": len(cases),
        "paired_il3_per_request": len(cases) * 10,
        "global_ordinal_summary": sum(
            int(depth.split("IL", 1)[1].split("-", 1)[0]) * 8
            for depth in {row["depth"] for row in cases}
        ),
        "typed_scientific_failures": sum(
            row.get("classification") != "TECHNICAL_FAIL" for row in failures
        ),
        "technical_failures": sum(
            row.get("classification") == "TECHNICAL_FAIL" for row in failures
        ),
    }
    rows = manifest["row_counts"]
    if rows != expected_rows:
        raise RuntimeError(f"row counts differ: {rows} != {expected_rows}")
    for name in (
        "p1r52-target-depth-il1-il3-il8-il10-il15-depth-summary.json",
        "p1r52-target-depth-il1-il3-il8-il10-il15-comparisons.json",
        "p1r52-target-depth-il8-il10-il15-per-case.json",
        "p1r52-target-depth-il8-il10-il15-per-step.json",
        "p1r52-target-depth-il8-il10-il15-per-inner.json",
        "p1r52-target-depth-il8-il10-il15-per-inner-request.json",
        "p1r52-target-depth-il8-il10-il15-per-request.json",
        "p1r52-target-depth-il8-il10-il15-minus-il3-paired-per-case.json",
        "p1r52-target-depth-il8-il10-il15-minus-il3-paired-per-request.json",
        "p1r52-target-depth-il8-il10-il15-global-ordinal-summary.json",
        "p1r52-target-depth-il8-il10-il15-result-identities.json",
    ):
        load(HERE / name)
    print(
        json.dumps(
            {
                "status": "REPORT_INTEGRITY_PASS",
                "report_sha256": receipt["report_sha256"],
                "manifest_sha256": receipt["manifest_sha256"],
                "manifest_root": receipt["manifest_root"],
                "row_counts": rows,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
