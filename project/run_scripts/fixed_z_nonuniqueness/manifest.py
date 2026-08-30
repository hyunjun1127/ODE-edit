"""Create-once, outcome-independent data role manifest builder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCREEN_CASE_IDS = [21100, 1477, 20838, 18707, 14288, 16426, 19041, 17609]
DIRECT_Z_REPORT = "experiment-reports/global/2026-08-01-direct-z-possibility-alpha-llama-p0-v1.analysis.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _rank(case_id: int, role: str) -> str:
    return hashlib.sha256(f"odeedit-s06-fixed-z-v1|{role}|{case_id}".encode()).hexdigest()


def build(dataset: Path, source_root: Path) -> dict[str, Any]:
    rows = json.loads(dataset.read_text())
    by_id = {int(row["case_id"]): row for row in rows}
    if any(case not in by_id for case in SCREEN_CASE_IDS):
        raise ValueError("screen case missing")
    excluded = set(SCREEN_CASE_IDS)
    pools: dict[str, list[int]] = {}
    sizes = {"history": 32, "calibration": 8, "controller": 8, "screen": 32, "final_audit_sealed": 32}
    used = set(excluded)
    for role, size in sizes.items():
        ordered = sorted((int(row["case_id"]) for row in rows if int(row["case_id"]) not in used), key=lambda cid: _rank(cid, role))
        pools[role] = ordered[:size]
        used.update(pools[role])
    source_report = source_root / DIRECT_Z_REPORT
    report = json.loads(source_report.read_text())
    previous_ids = [int(row["case_id"]) for row in report["case_diagnostics"]]
    if previous_ids != SCREEN_CASE_IDS:
        raise ValueError("preregistered common 8-case order differs")
    screen_rows = []
    for ordinal, case_id in enumerate(SCREEN_CASE_IDS):
        row = by_id[case_id]
        screen_rows.append({
            "ordinal": ordinal,
            "case_id": case_id,
            "request_sha256": canonical_hash(row["requested_rewrite"]),
            "row_sha256": canonical_hash(row),
        })
    payload = {
        "schema": "odeedit.s06.fixed-z-nonuniqueness.case-manifest.v1",
        "selection": "most_recent_main_integrated_preregistered_common_8_case_direct_z_panel",
        "selection_outcome_influence_count": 0,
        "replacement_count": 0,
        "dataset": {"path": str(dataset), "sha256": sha256(dataset), "bytes": dataset.stat().st_size},
        "provenance": {"path": DIRECT_Z_REPORT, "sha256": sha256(source_report)},
        "screen_cases": screen_rows,
        "roles": {role: [{"ordinal": i, "case_id": cid, "row_sha256": canonical_hash(by_id[cid])} for i, cid in enumerate(ids)] for role, ids in pools.items()},
        "final_audit_open_count": 0,
        "sample_duplication_count": 0,
    }
    payload["root_digest"] = canonical_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("refusing to overwrite manifest")
    payload = build(args.dataset.resolve(), args.source_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
