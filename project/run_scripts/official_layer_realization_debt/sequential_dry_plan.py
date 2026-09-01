"""Create-once four-cell cumulative sequential dry plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from project.run_scripts.ode_bf.contracts import canonical_hash

from .contracts import ObservationBoundary
from .dry_plan import CELL_MAPPING
from .sequential_contracts import INSTRUCTION_ID, NONCE


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    *, source_root: Path, expected_head: str, preflight: Path,
    result_root: Path, campaign_id: str,
) -> dict[str, Any]:
    if _git(source_root, "rev-parse", "HEAD") != expected_head or _git(
        source_root, "status", "--porcelain", "--untracked-files=no"
    ):
        raise ObservationBoundary("sequential source identity differs")
    receipt = json.loads(preflight.read_text())
    if (
        receipt.get("status") != "SEQUENTIAL_PRE_GPU_PASS"
        or receipt.get("instruction_id") != INSTRUCTION_ID
        or receipt.get("source", {}).get("head") != expected_head
    ):
        raise ObservationBoundary("sequential preflight release differs")
    if result_root.exists() or result_root.is_symlink():
        raise ObservationBoundary("sequential result root already exists")
    cells = [
        {
            **mapping,
            "cell_root": str(result_root / f"{mapping['model']}-{mapping['method']}-sequential-b10x10"),
            "execution_semantics": "B1_TO_B10_CUMULATIVE_W_CACHE_SEQUENTIAL",
        }
        for mapping in CELL_MAPPING
    ]
    payload = {
        "schema": "odeedit.s06.official-layer-realization-debt.sequential-dry-plan.v1",
        "instruction_id": INSTRUCTION_ID,
        "nonce": NONCE,
        "status": "SEQUENTIAL_DRY_PLAN_PASS",
        "campaign_id": campaign_id,
        "source": {
            "head": expected_head,
            "tree": _git(source_root, "rev-parse", "HEAD^{tree}"),
            "tracked_clean": True,
        },
        "preflight": {"path": str(preflight), "sha256": _sha256(preflight)},
        "result_root": str(result_root),
        "array": "0-3%4",
        "gpu_per_cell": 1,
        "task_gpu_cap": 4,
        "cells": cells,
        "batch_w0_reset_count": 0,
        "cross_batch_weight_continuity": 1,
        "scientific_promotion": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("refusing to overwrite sequential dry plan")
    payload = build(
        source_root=args.source_root.resolve(),
        expected_head=args.expected_head,
        preflight=args.preflight.resolve(),
        result_root=args.result_root.resolve(),
        campaign_id=args.campaign_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(raw)
    os.chmod(args.output, 0o600)


if __name__ == "__main__":
    main()
