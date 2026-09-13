"""Bounded lifelong source/stream/resource preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from project.run_scripts.fixed_z_nonuniqueness.contracts import MODEL_SPECS
from project.run_scripts.ode_bf.contracts import canonical_hash

from .contracts import ObservationBoundary
from .lifelong_contracts import (
    CONTRACT_BYTES,
    CONTRACT_SHA256,
    CONTRACT_WC_LINES,
    INSTRUCTION_ID,
    LifelongLock,
    MODEL_METHOD_PRIORITY,
)
from .lifelong_stream import verify_existing_1k_prefix, verify_lifelong_stream
from .preflight import build as build_base_preflight


PREVIOUS_100_WALL_SECONDS = {
    "llama3-8b-inst:memit": 637.512100965716,
    "qwen2.5-7b-inst:memit": 659.4088648594916,
    "llama3-8b-inst:alphaedit": 847.9455082919449,
    "qwen2.5-7b-inst:alphaedit": 2116.317554526031,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _storage_estimate() -> dict[str, Any]:
    checkpoints = len(LifelongLock().checkpoint_batches)
    rows = {}
    total = 0
    for alias, spec in MODEL_SPECS.items():
        config = json.loads((spec.model_path / "config.json").read_text())
        hidden = int(config["hidden_size"])
        intermediate = int(config["intermediate_size"])
        weight_bytes = len(LifelongLock().layers) * hidden * intermediate * 4
        alpha_cache_bytes = len(LifelongLock().layers) * intermediate * intermediate * 4
        for method in ("memit", "alphaedit"):
            checkpoint_bytes = weight_bytes + (
                alpha_cache_bytes if method == "alphaedit" else 0
            )
            value = checkpoint_bytes * checkpoints
            rows[f"{alias}:{method}"] = {
                "editable_weight_bytes_per_checkpoint": weight_bytes,
                "dynamic_cache_bytes_per_checkpoint": (
                    alpha_cache_bytes if method == "alphaedit" else 0
                ),
                "checkpoint_count": checkpoints,
                "projected_checkpoint_bytes": value,
            }
            total += value
    projected = int(total * 1.15) + 2 * 1024**3
    usage = shutil.disk_usage("/data/janghj")
    remaining = usage.free - projected
    return {
        "cells": rows,
        "checkpoint_bytes": total,
        "projected_total_bytes_with_15pct_and_journals": projected,
        "available_bytes": usage.free,
        "projected_remaining_bytes": remaining,
        "minimum_remaining_bytes": 100 * 1024**3,
        "storage_safe": remaining >= 100 * 1024**3,
    }


def _gpu_estimate() -> dict[str, Any]:
    # 100x request scale plus 24% for t0/heavy probe forks/checkpoint eval.
    factor = 124.0
    cells = {
        key: value * factor / 3600.0
        for key, value in PREVIOUS_100_WALL_SECONDS.items()
    }
    total = sum(cells.values())
    return {
        "basis": "observed sequential 100-request wall x100 plus 24pct heavy-probe reserve",
        "cells_gpu_hours": cells,
        "total_gpu_hours": total,
        "cap2_ideal_wall_hours": total / 2.0,
        "physical_impossibility": False,
    }


def build(
    *,
    source_root: Path,
    expected_head: str,
    focused_gate: Path,
    stream_seal: Path,
) -> dict[str, Any]:
    inherited = build_base_preflight(source_root, expected_head, focused_gate)
    stream = verify_lifelong_stream(json.loads(stream_seal.read_text()))
    prefix = verify_existing_1k_prefix(
        stream,
        source_root
        / "project/run_scripts/ode_bf/locks/p1r52_sequential_b100x10_stream_seal.json",
    )
    storage = _storage_estimate()
    gpu = _gpu_estimate()
    if not storage["storage_safe"] or gpu["physical_impossibility"]:
        raise ObservationBoundary("lifelong resource estimate is unsafe")
    payload = {
        **inherited,
        "schema": "odeedit.s06.layer-realization-debt.lifelong-pre-gpu.v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "LIFELONG_PRE_GPU_PASS",
        "contract": {
            "sha256": CONTRACT_SHA256,
            "bytes": CONTRACT_BYTES,
            "wc_lines": CONTRACT_WC_LINES,
            "full_read": True,
        },
        "lock": LifelongLock().payload(),
        "stream_10k": {
            "path": str(stream_seal),
            "sha256": _sha256(stream_seal),
            "root": stream["root_digest"],
            "order": stream["training_order_sha256"],
            "sentinel_order": stream["sentinel_order_sha256"],
            "exact_phase123_1k_prefix": prefix,
        },
        "priority": [list(value) for value in MODEL_METHOD_PRIORITY],
        "resource": {"gpu": gpu, "storage": storage, "gpu_cap": 2},
        "barrier_submit_count": 0,
        "scientific_promotion": False,
    }
    payload.pop("identity_sha256", None)
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--focused-gate", required=True, type=Path)
    parser.add_argument("--stream-seal", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("refusing to overwrite lifelong preflight")
    payload = build(
        source_root=args.source_root.resolve(),
        expected_head=args.expected_head,
        focused_gate=args.focused_gate.resolve(),
        stream_seal=args.stream_seal.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(raw)
    os.chmod(args.output, 0o600)


if __name__ == "__main__":
    main()
