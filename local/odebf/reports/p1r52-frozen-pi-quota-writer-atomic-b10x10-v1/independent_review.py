#!/usr/bin/env python3
"""Independent bounded rehash/content review for the raw-free FPiQ report."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(name: str) -> Any:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main() -> None:
    manifest = read("analysis-manifest.json")
    receipt = read("analysis-receipt.json")
    aggregate = read("p1r52-fpiq-aggregate-summary.json")
    checks: dict[str, bool] = {}
    rehash = []
    for member in manifest["members"]:
        path = Path(member["path"])
        actual = {
            "name": member["name"],
            "path": str(path),
            "sha256": sha(path),
            "bytes": path.stat().st_size,
            "mode": oct(path.stat().st_mode & 0o777),
            "rows_or_lines": member["rows_or_lines"],
        }
        rehash.append(actual)
        checks[f"member_{member['name']}_identity"] = actual == member
    checks["members_root"] = (
        hashlib.sha256(canonical_bytes(rehash)).hexdigest()
        == manifest["members_root_sha256"]
    )
    checks["manifest_sha"] = sha(ROOT / "analysis-manifest.json") == receipt["manifest_sha256"]
    checks["receipt_manifest_identity"] = (
        receipt["manifest_identity_sha256"] == manifest["identity_sha256"]
    )
    checks["case_rows_30"] = len(read("p1r52-fpiq-per-case.json")) == 30
    checks["step_rows_229"] = len(read("p1r52-fpiq-per-step.json")) == 229
    checks["layer_rows_1145"] = len(read("p1r52-fpiq-per-layer.json")) == 1145
    checks["failure_rows_23"] = len(read("p1r52-fpiq-typed-failures.json")) == 23
    checks["paired_rows_30"] = len(read("p1r52-fpiq-paired-case-deltas.json")) == 30
    checks["endpoint_denominators"] = [
        aggregate["arms"][arm]["completed_endpoints"] for arm in ("J0", "SV", "FPIQ")
    ] == [10, 8, 9]
    checks["available_prefix_denominators"] = [
        aggregate["arms"][arm]["available_step_prefixes"] for arm in ("J0", "SV", "FPIQ")
    ] == [80, 75, 74]
    checks["typed_incomplete_prefixes"] = sorted(
        (row["arm"], row["case_index"], row["last_valid_prefix_count"])
        for row in read("p1r52-fpiq-typed-failures.json")
        if row["attempt_role"] == "CANONICAL_TECH_R1"
    ) == [("FPIQ", 7, 2), ("SV", 3, 4), ("SV", 6, 7)]
    checks["entry_numeric_identity"] = aggregate["entry_identity"]["max_abs_numeric_residual"] == 0.0
    checks["h_pi_identity"] = (
        aggregate["mechanical"]["pi_application_values"] == [1]
        and aggregate["mechanical"]["physical_h_application_values"] == [1]
        and aggregate["mechanical"]["second_h_application_values"] == [0]
    )
    checks["virtual_commit_identity"] = (
        aggregate["mechanical"]["virtual_commit_identity_pass_count"]
        == aggregate["mechanical"]["virtual_commit_identity_denominator"]
    )
    checks["W0_restore_30_30"] = (
        aggregate["mechanical"]["W0_restore_attempt_pass_count"] == 30
        and aggregate["mechanical"]["W0_restore_attempt_denominator"] == 30
    )
    checks["no_imputation"] = bool(aggregate["no_imputation"] and receipt["no_imputation"])
    report = (ROOT / "p1r52-frozen-pi-quota-writer-atomic-b10x10-analysis-ko.md").read_text(
        encoding="utf-8"
    )
    required_terms = (
        "J0", "SV", "FPIQ", "accepted-z", "physical W", "writer coverage",
        "Structural-P", "negative prefix", "HOLD_NO_PROMOTION_TO_B100_OR_HISTORICAL",
        "NOT_RECORDED", "no imputation",
    )
    checks["required_report_terms"] = all(term in report for term in required_terms)
    checks["scientific_boundary"] = (
        aggregate["promotion_decision"] == "HOLD_NO_PROMOTION_TO_B100_OR_HISTORICAL"
        and "TARGET_SIDE_REGRESSION" in aggregate["scientific_classification"]
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"independent review failed: {failed}")
    review = {
        "schema": "ode-edit-s05-p1r52-fpiq-independent-rehash-review/v1",
        "status": "INDEPENDENT_REHASH_REVIEW_PASS",
        "checks": checks,
        "check_count": len(checks),
        "pass_count": sum(checks.values()),
        "analysis_manifest_path": str(ROOT / "analysis-manifest.json"),
        "analysis_manifest_sha256": sha(ROOT / "analysis-manifest.json"),
        "analysis_manifest_root": manifest["members_root_sha256"],
        "analysis_receipt_path": str(ROOT / "analysis-receipt.json"),
        "analysis_receipt_sha256": sha(ROOT / "analysis-receipt.json"),
        "analysis_receipt_root": receipt["root_digest"],
        "report_path": str(ROOT / "p1r52-frozen-pi-quota-writer-atomic-b10x10-analysis-ko.md"),
        "report_sha256": sha(ROOT / "p1r52-frozen-pi-quota-writer-atomic-b10x10-analysis-ko.md"),
        "rehash_members": rehash,
        "model_action_count": 0,
        "evaluator_action_count": 0,
        "gpu_action_count": 0,
        "slurm_action_count": 0,
        "result_mutation_count": 0,
    }
    review["root_digest"] = hashlib.sha256(canonical_bytes(review)).hexdigest()
    path = ROOT / "independent-review.json"
    path.write_text(
        json.dumps(review, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)
    print(json.dumps({
        "status": review["status"],
        "checks": f"{review['pass_count']}/{review['check_count']}",
        "path": str(path),
        "sha256": sha(path),
        "root": review["root_digest"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
