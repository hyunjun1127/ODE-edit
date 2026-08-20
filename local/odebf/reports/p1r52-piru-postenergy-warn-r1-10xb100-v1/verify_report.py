#!/usr/bin/env python3
"""Independent filesystem rehash and arithmetic review for the final package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPORT_DIR = Path(__file__).resolve().parent
WORKTREE = Path(__file__).resolve().parents[4]
RESULT_ROOT = WORKTREE / "local/odebf/results/s05-p1r52-pir-u-soft-sequential-structuralh-on-10xb100-postenergy-warn-r1-tech-r1-v1"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    manifest_path = REPORT_DIR / "analysis-manifest.json"
    receipt_path = REPORT_DIR / "rooted-analysis-receipt.json"
    manifest = load(manifest_path)
    receipt = load(receipt_path)
    checks: list[dict[str, Any]] = []
    rebuilt = []
    all_pass = True
    for member in manifest["members"]:
        path = REPORT_DIR / member["name"]
        payload = load(path) if path.suffix == ".json" else None
        observed = {
            "name": member["name"],
            "sha256": sha(path),
            "bytes": path.stat().st_size,
            "lines": len(path.read_text(encoding="utf-8").splitlines()),
            "rows": payload.get("row_count") if isinstance(payload, dict) else None,
        }
        ok = observed == {key: member[key] for key in observed}
        checks.append({"gate": f"member:{member['name']}", "status": "PASS" if ok else "FAIL", "expected": member, "observed": observed})
        all_pass &= ok
        rebuilt.append(observed)
    root = digest_text("\n".join(f"{m['name']}|{m['sha256']}|{m['bytes']}|{m['lines']}|{m['rows']}" for m in rebuilt))
    scalar_checks = {
        "member_root": root == manifest["member_root_sha256"] == receipt["member_root_sha256"],
        "manifest_sha": sha(manifest_path) == receipt["analysis_manifest_sha256"],
        "report_sha": sha(REPORT_DIR / "p1r52-piru-postenergy-warn-r1-10xb100-factual-ko.md") == receipt["report_sha256"],
        "terminal_sha": sha(RESULT_ROOT / "terminal.json") == manifest["result_terminal_sha256"],
        "result_manifest_sha": sha(RESULT_ROOT / "manifest.json") == manifest["result_manifest_sha256"],
    }
    for gate, ok in scalar_checks.items():
        checks.append({"gate": gate, "status": "PASS" if ok else "FAIL"})
        all_pass &= ok

    terminal = load(RESULT_ROOT / "terminal.json")
    h = load(RESULT_ROOT / "structural-h-decisions.json")["rows"]
    layers = load(REPORT_DIR / "p1r52-piru-postenergy-per-layer.json")["rows"]
    zw = load(REPORT_DIR / "p1r52-piru-postenergy-per-z-w-request.json")["rows"]
    integrity_checks = {
        "scheduler_terminal_bound": receipt["scheduler"] == {"job_id": 20885, "state": "COMPLETED", "exit_code": "0:0", "elapsed": "03:07:20"},
        "batch_10": terminal["round_count"] == 10 and terminal["action_freeze_checkpoint_count"] == 10,
        "h_80": len(h) == 80 and terminal["structural_h_decision_count"] == 80,
        "layer_400": len(layers) == 400,
        "zw_request_1000": len(zw) == 1000,
        "W0_exact": terminal["terminal_W0_restore"]["pointer_restored_exact"] and terminal["terminal_W0_restore"]["byte_restored_exact"],
        "observation_no_influence": terminal["accepted_z_observation_added_backward_count"] == 0 and terminal["accepted_z_observation_added_generation_call_count"] == 0 and terminal["accepted_z_observation_action_influence_count"] == 0,
        "duplicate_W_forward_0": terminal["duplicate_W_evaluator_model_forward_count"] == 0,
        "warn_count_60": sum(row["post_energy_status"] == "WARN_POSTSOLVE_ENERGY_RESIDUAL" for row in h) == 60,
        "old_energy_false_1": sum(row["energy_violation"] > 1e-12 for row in h) == 1,
        "B4K2_exact": any(row["round"] == 4 and row["step_index"] == 1 and row["energy_violation"] == 1.9768631176475537e-12 and row["certificate_hard_gate_status"] == "PASS" for row in h),
        "all_hard_gates_pass": all(row["certificate_hard_gate_status"] == "PASS" for row in h),
        "norm_share_arithmetic": all(abs(sum(row["actual_update_norm_share"] for row in layers if row["round"] == b and row["outer_k"] == k) - 1.0) <= 1e-12 for b in range(1, 11) for k in range(1, 9)),
    }
    for gate, ok in integrity_checks.items():
        checks.append({"gate": gate, "status": "PASS" if ok else "FAIL"})
        all_pass &= ok

    report = (REPORT_DIR / "p1r52-piru-postenergy-warn-r1-10xb100-factual-ko.md").read_text(encoding="utf-8")
    report_checks = {
        "korean_report_nonempty": len(report.splitlines()) >= 100,
        "five_row_boundary_present": "5행 핵심 표" in report,
        "scientific_amendment_labeled": "명시적 방법 수정" in report,
        "no_imputation_boundary": "batch-entry" in report,
        "promotion_false": "scientific_promotion=false" in report,
    }
    for gate, ok in report_checks.items():
        checks.append({"gate": gate, "status": "PASS" if ok else "FAIL"})
        all_pass &= ok

    review = {
        "schema": "ode-edit-s05-p1r52-piru-postenergy-warn-r1-independent-review/v1",
        "status": "INDEPENDENT_RAWFREE_REHASH_REVIEW_PASS" if all_pass else "INDEPENDENT_RAWFREE_REHASH_REVIEW_FAIL",
        "analysis_manifest_sha256": sha(manifest_path),
        "rooted_receipt_sha256": sha(receipt_path),
        "member_root_recomputed": root,
        "checks": checks,
        "pass_count": sum(check["status"] == "PASS" for check in checks),
        "fail_count": sum(check["status"] != "PASS" for check in checks),
        "model_evaluator_gpu_slurm_actions": {"model": 0, "evaluator": 0, "gpu": 0, "slurm": 0},
    }
    review["identity_sha256"] = digest_text(json.dumps(review, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    out = REPORT_DIR / "independent-rawfree-rehash-review.json"
    out.write_text(json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": review["status"],
        "path": str(out),
        "sha256": sha(out),
        "identity_sha256": review["identity_sha256"],
        "pass_count": review["pass_count"],
        "fail_count": review["fail_count"],
    }, ensure_ascii=False, indent=2))
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
