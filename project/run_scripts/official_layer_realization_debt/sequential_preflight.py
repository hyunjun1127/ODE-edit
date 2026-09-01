"""Reuse the accepted deployment audit and seal the sequential lock."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from project.run_scripts.ode_bf.contracts import canonical_hash

from .preflight import build as build_base_preflight
from .sequential_contracts import INSTRUCTION_ID, NONCE, SequentialObservationLock


def build(source_root: Path, expected_head: str, focused_gate: Path) -> dict:
    inherited = build_base_preflight(source_root, expected_head, focused_gate)
    payload = {
        **inherited,
        "schema": "odeedit.s06.official-layer-realization-debt.sequential-pre-gpu.v1",
        "instruction_id": INSTRUCTION_ID,
        "nonce": NONCE,
        "status": "SEQUENTIAL_PRE_GPU_PASS",
        "observation_lock": SequentialObservationLock().payload(),
        "inherited_independent_deployment_gate": {
            "schema": inherited["schema"],
            "identity_sha256": inherited["identity_sha256"],
        },
        "cross_batch_weight_continuity": 1,
        "alphaedit_dynamic_cache_continuity": 1,
        "batch_w0_reset_count": 0,
    }
    payload.pop("identity_sha256", None)
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--focused-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("refusing to overwrite sequential preflight")
    payload = build(args.source_root.resolve(), args.expected_head, args.focused_gate.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(raw)
    os.chmod(args.output, 0o600)


if __name__ == "__main__":
    main()
