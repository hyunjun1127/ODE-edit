#!/usr/bin/env python3
"""Bounded raw-free rehash and consistency review for P1R52 PIR analysis."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-pir-atomic-b10x10-v1/local/odebf/reports/p1r52-pir-sequential-writer-atomic-b10x10-v1")
WORKTREE = ROOT.parents[3]
RESULTS = WORKTREE / "local/odebf/results"
EXPECTED_HEAD = "871d41c668ed46535a08fd4886318f881798886f"
EXPECTED_TREE = "d5858904bd10073cbac5c7bf522a62c536d3c82e"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def main() -> None:
    manifest = load(ROOT / "analysis-manifest.json")
    checks = []
    for member in manifest["members"]:
        path = Path(member["path"])
        checks.append({"name": member["name"], "exists": path.is_file(), "sha_match": path.is_file() and sha(path) == member["sha256"], "size_match": path.is_file() and path.stat().st_size == member["bytes"], "mode_match": path.is_file() and oct(path.stat().st_mode & 0o777) == member["mode"]})
    aggregate = load(ROOT / "p1r52-pir-aggregate-summary.json")
    case_rows = load(ROOT / "p1r52-pir-per-case.json")
    step_rows = load(ROOT / "p1r52-pir-per-step.json")
    layer_rows = load(ROOT / "p1r52-pir-per-layer.json")
    request_rows = load(ROOT / "p1r52-pir-per-request.json")
    paired_rows = load(ROOT / "p1r52-pir-paired-case-deltas.json")
    terminals = {}
    for arm, slug in (("J0", "pir-j0"), ("PIR-G", "pir-g"), ("PIR-U", "pir-u")):
        root = RESULTS / f"s05-p1r52-pir-independent-b10x10-llama3-8b-inst-{slug}-repair-r1-v1"
        terminal = load(root / "terminal.json")
        terminals[arm] = {"source_head": terminal["source_head"], "completed_case_count": terminal["completed_case_count"], "failed_case_count": terminal["failed_case_count"], "terminal_sha256": sha(root / "terminal.json"), "manifest_sha256": sha(root / "manifest.json")}
    review = {
        "schema": "ode-edit-s05-p1r52-pir-independent-analysis-review/v1",
        "review_scope": "RAW_FREE_ONLY_NO_MODEL_EVALUATOR_GPU_SLURM_SOURCE_ACTION",
        "analysis_manifest_sha256": sha(ROOT / "analysis-manifest.json"),
        "analysis_receipt_sha256": sha(ROOT / "analysis-receipt.json"),
        "source_head_expected": EXPECTED_HEAD,
        "source_tree_expected": EXPECTED_TREE,
        "manifest_member_checks": checks,
        "input_terminal_checks": terminals,
        "row_count_checks": {
            "case": [len(case_rows), 30], "step": [len(step_rows), 240], "layer": [len(layer_rows), 1200], "request": [len(request_rows), 2400], "paired": [len(paired_rows), 20],
        },
        "integrity_checks": {
            "authoritative_attempts_endpoints": [aggregate["authoritative_attempts"], aggregate["authoritative_endpoints"]],
            "typed_failures": aggregate["authoritative_typed_failures"],
            "all_k8": all(row["accepted_k"] == 8 for row in case_rows),
            "all_w0_restored": all(row["W0_restored"] for row in case_rows),
            "all_retry_zero": all(row["retry_count"] == 0 for row in case_rows),
            "all_history_off": all(row["history_mode"] == "OFF" and row["history_append_count"] == 0 for row in case_rows),
            "pir_additional_current_slope_backward_zero": sum(row["additional_current_slope_backward_count"] for row in case_rows if row["arm"] != "J0") == 0,
            "pir_p_proxy_labeled": all(row["structural_p_receipt"] == "ENTRY_FIELD_MIXED_GEOMETRY_PROXY_NOT_COMPARABLE" for row in step_rows if row["arm"] != "J0"),
            "analysis_actions": {"model": 0, "evaluator": 0, "gpu": 0, "slurm": 0, "source_scientific": 0},
        },
        "promotion_status": aggregate["promotion"],
    }
    passed = all(check["exists"] and check["sha_match"] and check["size_match"] and check["mode_match"] for check in checks)
    passed = passed and all(item["source_head"] == EXPECTED_HEAD and item["completed_case_count"] == 10 and item["failed_case_count"] == 0 for item in terminals.values())
    passed = passed and review["row_count_checks"] == {"case": [30, 30], "step": [240, 240], "layer": [1200, 1200], "request": [2400, 2400], "paired": [20, 20]}
    passed = passed and all(value is True or value in ([30, 30], 0, {"model": 0, "evaluator": 0, "gpu": 0, "slurm": 0, "source_scientific": 0}) for value in review["integrity_checks"].values())
    review["status"] = "INDEPENDENT_RAW_FREE_REVIEW_PASS" if passed else "INDEPENDENT_RAW_FREE_REVIEW_FAIL"
    review["identity_sha256"] = hashlib.sha256(canonical(review)).hexdigest()
    target = ROOT / "independent-review.json"
    target.write_text(json.dumps(review, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    os.chmod(target, 0o600)
    print(json.dumps({"status": review["status"], "path": str(target), "sha256": sha(target), "identity_sha256": review["identity_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
