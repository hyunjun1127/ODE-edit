"""Create-once dry plan for the four observational GPU cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from project.run_scripts.ode_bf.contracts import canonical_hash

from .contracts import INSTRUCTION_ID, NONCE, ObservationBoundary


CELL_MAPPING = (
    {"array_index": 0, "model": "llama3-8b-inst", "method": "memit"},
    {"array_index": 1, "model": "llama3-8b-inst", "method": "alphaedit"},
    {"array_index": 2, "model": "qwen2.5-7b-inst", "method": "memit"},
    {"array_index": 3, "model": "qwen2.5-7b-inst", "method": "alphaedit"},
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    *,
    stage: str,
    source_root: Path,
    expected_head: str,
    preflight: Path,
    result_root: Path,
    campaign_id: str,
    b1_root: Path | None,
) -> dict[str, Any]:
    if stage not in {"b1", "b10x10"}:
        raise ObservationBoundary("unknown stage")
    if _git(source_root, "rev-parse", "HEAD") != expected_head:
        raise ObservationBoundary("source HEAD differs")
    if _git(source_root, "status", "--porcelain", "--untracked-files=no"):
        raise ObservationBoundary("source tracked worktree is dirty")
    preflight_payload = json.loads(preflight.read_text())
    if preflight_payload.get("status") != "PRE_GPU_PASS" or preflight_payload.get("source", {}).get("head") != expected_head:
        raise ObservationBoundary("preflight release differs")
    if result_root.exists() or result_root.is_symlink():
        raise ObservationBoundary("result root already exists")
    cells = []
    for mapping in CELL_MAPPING:
        cell = {**mapping, "cell_root": str(result_root / f"{mapping['model']}-{mapping['method']}-{stage}")}
        if stage == "b10x10":
            if b1_root is None:
                raise ObservationBoundary("B1 release root is required")
            b1_result = b1_root / f"{mapping['model']}-{mapping['method']}-b1/result.json"
            payload = json.loads(b1_result.read_text())
            if payload.get("status") != "B1_OBSERVER_ON_OFF_GATE_PASS":
                raise ObservationBoundary(f"B1 cell is not released: {b1_result}")
            if payload.get("model") != mapping["model"] or payload.get("method") != mapping["method"]:
                raise ObservationBoundary(f"B1 cell identity differs: {b1_result}")
            cell["b1_release"] = {"path": str(b1_result), "sha256": _sha256(b1_result)}
        cells.append(cell)
    payload = {
        "schema": "odeedit.s06.official-layer-realization-debt.dry-plan.v1",
        "instruction_id": INSTRUCTION_ID,
        "nonce": NONCE,
        "status": "DRY_PLAN_PASS",
        "stage": stage,
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
        "sequential_submit_count": 0,
        "scientific_promotion": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("b1", "b10x10"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--b1-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("refusing to overwrite dry-plan receipt")
    payload = build(
        stage=args.stage,
        source_root=args.source_root.resolve(),
        expected_head=args.expected_head,
        preflight=args.preflight.resolve(),
        result_root=args.result_root.resolve(),
        campaign_id=args.campaign_id,
        b1_root=None if args.b1_root is None else args.b1_root.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(raw)
    os.chmod(args.output, 0o600)


if __name__ == "__main__":
    main()
