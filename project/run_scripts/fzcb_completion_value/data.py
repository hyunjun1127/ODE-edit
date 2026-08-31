"""Exact sealed CounterFact request bindings for K0 and K1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import CASE_MANIFEST, DATASET, K0_CASE_IDS, K1_CASE_IDS, ScientificBoundary
from .hashing import canonical_hash


def load_rows(source_root: Path, stage: str) -> list[dict[str, Any]]:
    rows = json.loads(DATASET.read_text())
    indexed = {int(row["case_id"]): row for row in rows}
    manifest = json.loads((source_root / CASE_MANIFEST).read_text())
    expected = K0_CASE_IDS if stage == "k0" else K1_CASE_IDS
    sealed = manifest["roles"]["controller"][:4] if stage == "k0" else manifest["screen_cases"]
    if tuple(int(row["case_id"]) for row in sealed) != expected:
        raise ScientificBoundary(f"{stage} case/order seal mismatch")
    selected = []
    for sealed_row in sealed:
        row = indexed[int(sealed_row["case_id"])]
        if canonical_hash(row) != sealed_row["row_sha256"]:
            raise ScientificBoundary(f"row identity mismatch: {sealed_row['case_id']}")
        selected.append(row)
    return selected


def official_request(row: dict[str, Any]) -> dict[str, Any]:
    value = row["requested_rewrite"]
    return {
        "case_id": str(row["case_id"]), "prompt": value["prompt"],
        "subject": value["subject"], "target_new": value["target_new"]["str"],
        "target_true": value["target_true"]["str"],
    }


def canonical_prompt(row: dict[str, Any]) -> str:
    value = row["requested_rewrite"]
    return value["prompt"].format(value["subject"])
