"""Create-once cap2 priority dry plan for four lifelong cells."""

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
from .lifelong_contracts import INSTRUCTION_ID, MODEL_METHOD_PRIORITY


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
        raise ObservationBoundary("lifelong source identity differs")
    receipt = json.loads(preflight.read_text())
    if (
        receipt.get("status") != "LIFELONG_PRE_GPU_PASS"
        or receipt.get("source", {}).get("head") != expected_head
    ):
        raise ObservationBoundary("lifelong preflight release differs")
    if result_root.exists() or result_root.is_symlink():
        raise ObservationBoundary("lifelong result root already exists")
    cells = [
        {
            "priority": index + 1,
            "model": model,
            "method": method,
            "cell_root": str(result_root / f"{model}-{method}-lifelong-b100x100"),
        }
        for index, (model, method) in enumerate(MODEL_METHOD_PRIORITY)
    ]
    payload = {
        "schema": "odeedit.s06.layer-realization-debt.lifelong-dry-plan.v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "LIFELONG_DRY_PLAN_PASS",
        "campaign_id": campaign_id,
        "source": {
            "head": expected_head,
            "tree": _git(source_root, "rev-parse", "HEAD^{tree}"),
            "tracked_clean": True,
        },
        "preflight": {"path": str(preflight), "sha256": _sha256(preflight)},
        "result_root": str(result_root),
        "gpu_cap": 2,
        "first_wave": cells[:2],
        "second_wave": cells[2:],
        "cells": cells,
        "order_seed_count": 1,
        "barrier_submit_count": 0,
        "scientific_promotion": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("refusing to overwrite lifelong dry plan")
    payload = build(
        source_root=args.source_root.resolve(), expected_head=args.expected_head,
        preflight=args.preflight.resolve(), result_root=args.result_root.resolve(),
        campaign_id=args.campaign_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(raw)
    os.chmod(args.output, 0o600)


if __name__ == "__main__":
    main()
