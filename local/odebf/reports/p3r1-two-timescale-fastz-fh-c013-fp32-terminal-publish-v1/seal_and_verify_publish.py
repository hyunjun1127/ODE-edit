#!/usr/bin/env python3
"""Create-once seal and verify the Git-publishable P3R1 raw-free subset."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
REPORT = HERE / "p3r1-two-timescale-fastz-fh-c013-fp32-terminal-report-ko.md"
MANIFEST = HERE / "analysis-manifest-v2.json"
RECEIPT = HERE / "rooted-analysis-receipt-v2.json"
REVIEW = HERE / "independent-rehash-review.json"
EXCLUDED = {MANIFEST.name, RECEIPT.name, REVIEW.name}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def create_once(path: Path, value: object) -> None:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def entries() -> list[dict]:
    return [{
        "path": path.name,
        "bytes": path.stat().st_size,
        "lines": len(path.read_bytes().splitlines()),
        "mode": oct(path.stat().st_mode & 0o777)[2:].zfill(4),
        "sha256": sha(path),
    } for path in sorted(HERE.iterdir()) if path.is_file() and path.name not in EXCLUDED]


def main() -> None:
    if sys.argv[1:] == ["--rebuild-prepublish"]:
        history = HERE / "preseal-history"
        history.mkdir(mode=0o700, exist_ok=False)
        for path in (MANIFEST, RECEIPT, REVIEW):
            if path.exists():
                path.rename(history / path.name)
    elif sys.argv[1:]:
        raise RuntimeError("unsupported argument")
    if not MANIFEST.exists():
        members = entries()
        manifest = {
            "schema": "ode-edit-s05-p3r1-terminal-analysis-manifest/v2",
            "package_role": "GIT_PUBLISHABLE_RAW_FREE_AGGREGATE_SUBSET",
            "created_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "instruction_id": "ODEEDIT-S05-P3R1-TWO-TIMESCALE-FASTZ-FH-C013-FP32-V1",
            "status": "TERMINAL_ANALYSIS_PASS",
            "source_commits": [
                "b6077c376d1f82c08d464f51f57c9f9c8be04a0c",
                "05a804b0bf554a1ee3af0067dfc41fb1db8b21f8",
                "70946091da1523fb9b26565786b4f3ac1e703b74",
                "770dd02a26efa1805cd4010fef200ba962da711f",
                "f15ed227e62f7187fede2a631c157d2ac31ae824",
            ],
            "endpoint_denominator": {
                "scheduler_attempts": 6,
                "valid_endpoint_jobs": 2,
                "aliases_attempted": 2,
                "aliases_valid": 2,
                "valid_requests_by_model": {"Llama3-8B-Instruct": 100, "Qwen2.5-7B-Instruct": 100},
                "requests_attempted": 200,
                "requests_valid": 200,
                "technical_failed_jobs_excluded_from_endpoints": [22600, 22601, 22765, 22784],
                "typed_scientific_failures": 0,
                "imputation": 0,
            },
            "comparison_binding": {
                "target_only_M25_minus_M5": "MATCHED_WITHIN_MODEL_W0_REQUEST_ORDER_EVALUATOR",
                "C1_minus_C0": "MATCHED_WITHIN_MODEL_REQUEST_ORDER_EVALUATOR_BUT_ARM_LOCAL_DYNAMIC_TARGET_W_TRAJECTORIES",
                "C3_minus_Official_AlphaEdit": "MATCHED_WITHIN_MODEL_W0_REQUEST_ORDER_EVALUATOR; TARGET_AND_FH_POLICY_DIFFER",
            },
            "omitted_from_git": {
                "raw_logs_results_models_checkpoints_generations_tensors_cache": 0,
                "server2_only_large_raw_free_tables": ["per-request.json/csv", "per-inner.json/csv"],
            },
            "scientific_promotion": False,
            "members": members,
            "members_root_sha256": canonical_sha(members),
        }
        create_once(MANIFEST, manifest)
        receipt = {
            "schema": "ode-edit-s05-p3r1-terminal-rooted-analysis-receipt/v2",
            "status": "PASS",
            "canonical_report": REPORT.name,
            "canonical_report_sha256": sha(REPORT),
            "analysis_manifest": MANIFEST.name,
            "analysis_manifest_sha256": sha(MANIFEST),
            "members_root_sha256": manifest["members_root_sha256"],
            "valid_jobs": [22764, 22815],
            "excluded_technical_jobs": [22600, 22601, 22765, 22784],
            "valid_aliases": 2,
            "valid_requests": 200,
            "typed_scientific_failure_count": 0,
            "imputation_count": 0,
            "prohibited_raw_model_data_generation_member_count": 0,
            "scientific_promotion": False,
        }
        receipt["root_digest"] = canonical_sha(receipt)
        create_once(RECEIPT, receipt)

    manifest = json.loads(MANIFEST.read_text())
    receipt = json.loads(RECEIPT.read_text())
    checks = {
        "all_members_exist": all((HERE / row["path"]).is_file() for row in manifest["members"]),
        "all_member_hashes_match": all(sha(HERE / row["path"]) == row["sha256"] for row in manifest["members"]),
        "all_member_sizes_match": all((HERE / row["path"]).stat().st_size == row["bytes"] for row in manifest["members"]),
        "members_root_match": canonical_sha(manifest["members"]) == manifest["members_root_sha256"],
        "manifest_receipt_binding": sha(MANIFEST) == receipt["analysis_manifest_sha256"],
        "report_receipt_binding": sha(REPORT) == receipt["canonical_report_sha256"],
        "valid_requests_200": manifest["endpoint_denominator"]["requests_valid"] == 200,
        "technical_failures_excluded": manifest["endpoint_denominator"]["technical_failed_jobs_excluded_from_endpoints"] == [22600, 22601, 22765, 22784],
        "typed_scientific_failure_zero": manifest["endpoint_denominator"]["typed_scientific_failures"] == 0,
        "imputation_zero": manifest["endpoint_denominator"]["imputation"] == 0,
        "promotion_false": manifest["scientific_promotion"] is False and receipt["scientific_promotion"] is False,
        "prohibited_members_zero": receipt["prohibited_raw_model_data_generation_member_count"] == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps({key: value for key, value in checks.items() if not value}, sort_keys=True))
    if not REVIEW.exists():
        create_once(REVIEW, {
            "schema": "ode-edit-s05-p3r1-git-publish-independent-rehash-review/v1",
            "status": "PASS",
            "reviewed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "checks": checks,
            "canonical_report_sha256": sha(REPORT),
            "analysis_manifest_sha256": sha(MANIFEST),
            "rooted_analysis_receipt_sha256": sha(RECEIPT),
            "review_action_counts": {"model": 0, "GPU": 0, "Slurm": 0, "result": 0},
        })
    print(json.dumps({"status": "PASS", "checks": checks, "manifest_sha256": sha(MANIFEST), "receipt_sha256": sha(RECEIPT)}, sort_keys=True))


if __name__ == "__main__":
    main()
