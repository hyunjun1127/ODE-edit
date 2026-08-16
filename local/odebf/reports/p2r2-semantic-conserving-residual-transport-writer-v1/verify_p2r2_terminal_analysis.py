#!/usr/bin/env python3
"""Independent read-only recomputation and final package sealing for P2R2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


OUT = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-p2r2-semantic-conserving-residual-transport-writer-v1/"
    "local/odebf/reports/p2r2-semantic-conserving-residual-transport-writer-v1"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def record(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "lines": raw.count(b"\n"),
        "mode": oct(path.stat().st_mode & 0o777),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def main() -> None:
    expected_rows = {
        "p2r2-per-case.jsonl": 40,
        "p2r2-per-step.jsonl": 320,
        "p2r2-per-request.jsonl": 400,
        "p2r2-per-step-request.jsonl": 3200,
        "p2r2-response-routing.jsonl": 320,
        "p2r2-comparisons.jsonl": 180,
    }
    row_counts = {name: count_lines(OUT / name) for name in expected_rows}
    assert row_counts == expected_rows

    manifest = read_json(OUT / "analysis-manifest.json")
    assert manifest["root_digest"] == canonical_sha(
        {key: value for key, value in manifest.items() if key != "root_digest"}
    )
    for member in manifest["artifacts"]:
        path = Path(member["path"])
        assert path.is_file() and not path.is_symlink()
        assert sha256(path) == member["sha256"]
        assert path.stat().st_size == member["bytes"]
        assert oct(path.stat().st_mode & 0o777) == member["mode"] == "0o600"
    assert manifest["artifact_root"] == canonical_sha(manifest["artifacts"])

    receipt = read_json(OUT / "analysis-receipt.json")
    assert receipt["root_digest"] == canonical_sha(
        {key: value for key, value in receipt.items() if key != "root_digest"}
    )
    assert receipt["analysis_manifest_sha256"] == sha256(OUT / "analysis-manifest.json")
    assert receipt["analysis_manifest_root"] == manifest["root_digest"]
    assert receipt["case_rows"] == 40
    assert receipt["step_rows"] == 320
    assert receipt["request_rows"] == 400
    assert receipt["step_request_rows"] == 3200
    assert receipt["response_routing_rows"] == 320
    assert receipt["comparison_rows"] == 180
    assert receipt["mechanical_contract_receipt_state"] == "MECHANICAL_CONTRACT_RECEIPT_FAIL"
    assert receipt["outer_h_application_count"] == "NOT_RECORDED"
    assert receipt["physical_h_application_count_sum"] == 0
    assert receipt["one_joint_materialization_count"] == 320

    cases = [json.loads(line) for line in (OUT / "p2r2-per-case.jsonl").read_text().splitlines()]
    steps = [json.loads(line) for line in (OUT / "p2r2-per-step.jsonl").read_text().splitlines()]
    step_requests = [
        json.loads(line) for line in (OUT / "p2r2-per-step-request.jsonl").read_text().splitlines()
    ]
    assert sum(row["endpoints"] for row in cases) == 40
    assert sum(row["failures"] for row in cases) == 0
    assert sum(row["writer_transitions"] for row in cases) == 320
    assert sum(row["writer_materializations"] for row in cases) == 320
    assert all(row["W0_pointer_restored"] and row["W0_bytes_restored"] for row in cases)
    assert all(row["one_joint_materialization_count"] == 1 for row in steps)
    assert all(row["physical_h_application_count"] == 0 for row in steps)
    assert all(row["residual_divisor_values"] == [1] for row in steps)
    assert all(row["self_reachable_ceiling_c_i"] == "NOT_RECORDED" for row in step_requests)
    assert all(
        abs(row["selected_alpha_layer_sum"] - (1.0 if row["active"] else 0.0)) <= 1e-8
        for row in step_requests
    )

    aggregates = read_json(OUT / "p2r2-aggregates.json")["rows"]
    by_cell = {(row["alias"], row["arm"]): row for row in aggregates}
    expected_counts = {
        ("llama3-8b-inst", "NEUTRAL"): (100, 189, 841),
        ("llama3-8b-inst", "SOFTP"): (100, 189, 842),
        ("qwen2.5-7b-inst", "NEUTRAL"): (100, 185, 839),
        ("qwen2.5-7b-inst", "SOFTP"): (100, 185, 839),
    }
    for cell, counts in expected_counts.items():
        row = by_cell[cell]
        assert (row["w_eff_correct"], row["w_gen_correct"], row["w_loc_correct"]) == counts
        assert row["w_eff_correct"] == row["z_eff_correct"]
        assert row["w_gen_correct"] == row["z_gen_correct"]

    gates = read_json(OUT / "p2r2-gates.json")
    assert gates["rows_root"] == canonical_sha(gates["rows"])
    assert any(
        row["gate"] == "MECHANICAL_CONTRACT_RECEIPT"
        and row["state"] == "MECHANICAL_CONTRACT_RECEIPT_FAIL"
        for row in gates["rows"]
    )
    verification = {
        "schema": "ode-edit-s05-p2r2-independent-verification/v1",
        "status": "PASS_WITH_MECHANICAL_CONTRACT_RECEIPT_FAIL_PRESERVED",
        "row_counts": row_counts,
        "manifest_sha256": sha256(OUT / "analysis-manifest.json"),
        "manifest_root": manifest["root_digest"],
        "receipt_sha256": sha256(OUT / "analysis-receipt.json"),
        "receipt_root": receipt["root_digest"],
        "endpoint_counts": {
            f"{alias}:{arm}": list(counts)
            for (alias, arm), counts in sorted(expected_counts.items())
        },
        "mechanical_contract_receipt_state": receipt["mechanical_contract_receipt_state"],
        "scientific_promotion": False,
    }
    verification["root_digest"] = canonical_sha(
        {key: value for key, value in verification.items() if key != "root_digest"}
    )
    verification_path = OUT / "independent-verification.json"
    write_json(verification_path, verification)

    excluded = {"final-package-manifest.json", "final-package-receipt.json"}
    members = [
        record(path)
        for path in sorted(OUT.iterdir())
        if path.is_file() and not path.is_symlink() and path.name not in excluded
    ]
    package = {
        "schema": "ode-edit-s05-p2r2-final-package-manifest/v1",
        "members": members,
        "member_count": len(members),
        "file_tree_root": canonical_sha(members),
    }
    package["root_digest"] = canonical_sha(
        {key: value for key, value in package.items() if key != "root_digest"}
    )
    package_path = OUT / "final-package-manifest.json"
    write_json(package_path, package)
    final_receipt = {
        "schema": "ode-edit-s05-p2r2-final-package-receipt/v1",
        "package_manifest_sha256": sha256(package_path),
        "package_manifest_root": package["root_digest"],
        "file_tree_root": package["file_tree_root"],
        "member_count": package["member_count"],
        "report": record(
            OUT / "p2r2-semantic-conserving-residual-transport-terminal-analysis-ko.md"
        ),
        "row_counts": row_counts,
        "analysis_manifest_sha256": sha256(OUT / "analysis-manifest.json"),
        "analysis_receipt_sha256": sha256(OUT / "analysis-receipt.json"),
        "independent_verification_sha256": sha256(verification_path),
        "scientific_promotion": False,
    }
    final_receipt["root_digest"] = canonical_sha(
        {key: value for key, value in final_receipt.items() if key != "root_digest"}
    )
    write_json(OUT / "final-package-receipt.json", final_receipt)


if __name__ == "__main__":
    main()
