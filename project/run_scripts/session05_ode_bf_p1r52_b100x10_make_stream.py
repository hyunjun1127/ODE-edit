#!/usr/bin/env python3
"""Create-once CPU-only stream-seal builder for P1R52 10xB100."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.p1r52_b100x10_stream import (
    PREFIX_SEAL_FILE,
    SEAL_FILE,
    build_p1r52_b100x10_stream,
    verify_b10_prefix,
)


def _write_once(path: Path, payload: dict[str, object]) -> str:
    value = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "project/run_scripts/ode_bf/locks" / SEAL_FILE,
    )
    parser.add_argument(
        "--prefix-seal",
        type=Path,
        default=REPO_ROOT / "project/run_scripts/ode_bf/locks" / PREFIX_SEAL_FILE,
    )
    args = parser.parse_args()
    payload = build_p1r52_b100x10_stream(args.dataset, REPO_ROOT)
    prefix = json.loads(args.prefix_seal.read_text(encoding="utf-8"))
    verify_b10_prefix(prefix, payload)
    digest = _write_once(args.output, payload)
    print(
        json.dumps(
            {"path": str(args.output), "sha256": digest, "root": payload["root_digest"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
