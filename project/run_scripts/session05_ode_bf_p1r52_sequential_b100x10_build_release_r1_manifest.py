#!/usr/bin/env python3
"""Create-once namespace-only release manifest for the TECH-R3 checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.artifacts import sha256_file
from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1r52_sequential_b100x10_panel import (
    LOCK_FILE,
    PARENT,
    PARENT_TREE,
    SOURCE_MANIFEST_TECH_R3_RELEASE_R1,
)
from project.run_scripts.ode_bf.p1r52_sequential_scale import B100X10_INSTRUCTION_ID
from project.run_scripts.session05_ode_bf_p1r52_sequential_b100x10_build_repair_r3_manifest import (
    SOURCE_PATHS as R3_SOURCE_PATHS,
)


LOCK_ROOT = REPO_ROOT / "project/run_scripts/ode_bf/locks"
SOURCE_PATHS = R3_SOURCE_PATHS + (
    "project/run_scripts/session05_ode_bf_p1r52_sequential_b100x10_build_release_r1_manifest.py",
)


def main() -> int:
    entries = []
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        entries.append(
            {"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)}
        )
    value = {
        "schema_version": "ode-edit-s05-p1r52-sequential-b100x10-r52-tech-r3-release-r1-source-manifest/v1",
        "instruction_id": B100X10_INSTRUCTION_ID,
        "parent": PARENT,
        "parent_tree": PARENT_TREE,
        "scientific_source_parent": "cd1e4efbf0b8cfae3a7d07a8054033fa21a2c631",
        "release_revision": "TECH-R3-RELEASE-R1",
        "release_reason": "USER_OPTION_B_AND_CREATE_ONCE_RACE_NAMESPACE_REPLACEMENT",
        "scientific_source_delta_count": 0,
        "batch_entry_evaluator_enabled": False,
        "batch_entry_evaluator_call_count": 0,
        "entries": entries,
        "entry_count": len(entries),
        "numerical_lock_sha256": sha256_file(LOCK_ROOT / LOCK_FILE),
        "stream_seal_sha256": "550600af120070594078b73a5da7d066ad6287d8a3224c55f61babca592e0586",
    }
    payload = {**value, "root_digest": canonical_hash(value)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    destination = LOCK_ROOT / SOURCE_MANIFEST_TECH_R3_RELEASE_R1
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "path": str(destination),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "root": payload["root_digest"],
                "entry_count": len(entries),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
