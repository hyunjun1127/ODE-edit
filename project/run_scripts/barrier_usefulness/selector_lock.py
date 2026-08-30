"""Create-once global ctrl selector/candidate-bank lock before Q_gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .firewall import seal_selector_lock
from .hashing import canonical_hash, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctrl-result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells = []
    expected_cases = None
    for path in sorted(args.ctrl_result, key=str):
        row = json.loads(path.read_text())
        if row["status"] != "CTRL_TERMINAL_VALID" or row["edited_gate_access_count"] != 0:
            raise SystemExit(f"ctrl result not gate-lockable: {path}")
        if expected_cases is None:
            expected_cases = row["case_ids"]
        if row["case_ids"] != expected_cases or any(case["valid_candidate_count"] != 32 for case in row["cases"]):
            raise SystemExit("ctrl cell case/candidate completeness mismatch")
        cells.append({
            "cell_id": f"{row['model']}|{row['method']}", "model": row["model"], "method": row["method"],
            "result_path": str(path.resolve()), "result_sha256": file_sha256(path),
            "candidate_bank_hash": row["candidate_bank_hash"],
            "selected_candidate_ids": row["selected_candidate_ids"],
            "selected_candidate_ids_hash": row["selected_candidate_ids_hash"],
        })
    if len(cells) != 4 or len({cell["cell_id"] for cell in cells}) != 4:
        raise SystemExit("selector lock requires exact four model/method cells")
    selected = {cell["cell_id"]: cell["selected_candidate_ids"] for cell in cells}
    banks = {cell["cell_id"]: cell["candidate_bank_hash"] for cell in cells}
    payload = {
        "schema": "odeedit.s06.barrier-usefulness.selector-lock.v1",
        "status": "SELECTOR_LOCKED_BEFORE_GATE", "cells": cells,
        "case_ids": expected_cases, "candidate_bank_hash": canonical_hash(banks),
        "selected_candidate_ids_hash": canonical_hash(selected),
        "edited_gate_access_count": 0, "replacement_count": 0,
        "tie_break": "lexical-candidate-id", "gate_oracle_selection_influence_count": 0,
    }
    locked = seal_selector_lock(args.output, payload)
    print(json.dumps({"path": str(args.output), "sha256": file_sha256(args.output), "identity": locked["selector_lock_identity"]}, sort_keys=True))


if __name__ == "__main__":
    main()
