#!/usr/bin/env python3
"""Create the final v5 package receipt after the independent reviewer writes its receipt."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


OUT = Path(__file__).resolve().parent
REVIEW = OUT / "independent-v5-review.json"
TARGET = OUT / "final-package-receipt-v5.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    review = json.loads(REVIEW.read_text())
    if review.get("review_status", review.get("status")) != "PASS":
        raise RuntimeError("independent review is not PASS")
    reports = [
        OUT / "p1r52-soft-sequential-alphacache-on-structuralh-off-10xb10-factual-ko-v5.md",
        OUT / "p1r52-sequential-structuralh-on-off-native-three-arm-ko-v5.md",
    ]
    for path in reports:
        if not path.is_file():
            raise RuntimeError(f"missing canonical report: {path}")
    members = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != TARGET.name:
            members.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)})
    root = hashlib.sha256(json.dumps(members, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    receipt = {
        "schema": "p1r52-sequential/three-arm-final-package/v5",
        "status": "PASS",
        "canonical_control_report": str(reports[0]),
        "canonical_control_report_sha256": digest(reports[0]),
        "canonical_control_report_bytes": reports[0].stat().st_size,
        "canonical_control_report_lines": len(reports[0].read_text().splitlines()),
        "canonical_three_arm_report": str(reports[1]),
        "canonical_three_arm_report_sha256": digest(reports[1]),
        "canonical_three_arm_report_bytes": reports[1].stat().st_size,
        "canonical_three_arm_report_lines": len(reports[1].read_text().splitlines()),
        "readable_v5_manifest_sha256": digest(OUT / "readable-v5-manifest.json"),
        "readable_v5_receipt_sha256": digest(OUT / "readable-v5-receipt.json"),
        "independent_review_sha256": digest(REVIEW),
        "package_members_root_sha256": root,
        "raw_content_included": False,
        "analysis_actions": {"model": 0, "evaluator": 0, "gpu": 0, "slurm": 0, "source_edit": 0, "result_mutation": 0},
    }
    data = (json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(TARGET, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    print(json.dumps({"receipt": str(TARGET), "sha256": digest(TARGET), "root": root}, ensure_ascii=False))


if __name__ == "__main__":
    main()
