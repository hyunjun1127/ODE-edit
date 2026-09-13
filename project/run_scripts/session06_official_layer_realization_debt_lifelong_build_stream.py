#!/usr/bin/env python3
"""Create the outcome-free 10k lifelong stream seal once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.official_layer_realization_debt.contracts import DATASET
from project.run_scripts.official_layer_realization_debt.lifelong_stream import (
    build_lifelong_stream,
    verify_existing_1k_prefix,
)


PREFIX = REPO_ROOT / "project/run_scripts/ode_bf/locks/p1r52_sequential_b100x10_stream_seal.json"


def _write_once(path: Path, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build_lifelong_stream(DATASET, REPO_ROOT)
    prefix = verify_existing_1k_prefix(payload, PREFIX)
    payload["existing_phase123_1k_prefix"] = prefix
    payload["root_digest"] = __import__(
        "project.run_scripts.ode_bf.contracts", fromlist=["canonical_hash"]
    ).canonical_hash({key: value for key, value in payload.items() if key != "root_digest"})
    digest = _write_once(args.output, payload)
    print(json.dumps({"path": str(args.output), "sha256": digest, "root": payload["root_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
