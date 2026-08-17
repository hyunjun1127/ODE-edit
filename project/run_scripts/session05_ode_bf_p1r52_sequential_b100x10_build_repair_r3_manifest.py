#!/usr/bin/env python3
"""Create-once source manifest for the B100 history-digest TECH-R3 repair."""

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
    SOURCE_MANIFEST_TECH_R3,
)
from project.run_scripts.ode_bf.p1r52_sequential_scale import B100X10_INSTRUCTION_ID
from project.run_scripts.session05_ode_bf_p1r52_sequential_b100x10_build_repair_r2_manifest import (
    SOURCE_PATHS as R2_SOURCE_PATHS,
)


LOCK_ROOT = REPO_ROOT / "project/run_scripts/ode_bf/locks"
SOURCE_PATHS = tuple(
    path
    for path in R2_SOURCE_PATHS
    if not path.endswith("build_repair_r2_manifest.py")
) + (
    "project/run_scripts/ode_bf/tests/test_p1r52_sequential_evaluator.py",
    "project/run_scripts/session05_ode_bf_p1r52_sequential_b100x10_build_repair_r2_manifest.py",
    "project/run_scripts/session05_ode_bf_p1r52_sequential_b100x10_build_repair_r3_manifest.py",
)


def main() -> int:
    entries = []
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        entries.append(
            {"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)}
        )
    value = {
        "schema_version": "ode-edit-s05-p1r52-sequential-b100x10-r52-tech-r3-source-manifest/v1",
        "instruction_id": B100X10_INSTRUCTION_ID,
        "parent": PARENT,
        "parent_tree": PARENT_TREE,
        "supersedes_source_head": "a257e66b816b15d362f88f9f17061074c0b91abc",
        "repair_revision": "TECH-R3",
        "repair_scope": "B100_POST_COMMIT_HISTORY_ORDER_DIGEST_DISPATCH_AND_USER_REMOVED_ENTRY_AGGREGATE",
        "repair_first_false_gate": "POST_COMMIT_HISTORY_CAPTURE_USED_B10_ONLY_ORDER_DIGEST",
        "failed_job": "20457",
        "failed_cell_count": 2,
        "healthy_cell_rerun_count": 0,
        "exception_message_sha256": "389926c0ba116034c83ec8e916aec7b26201731d90723044a200c3c2aaa6883a",
        "failure_sha256_h_on": "23751a8ca4f34dd5e30f3c68108ea751dbb02ab7c81d39d738f1c3c75b1dbc40",
        "batch_entry_evaluator_enabled": False,
        "batch_entry_evaluator_call_count": 0,
        "batch_entry_reporting_influence_count": 0,
        "clamp_radius_delta_count": 0,
        "science_formula_tolerance_config_delta_count": 0,
        "added_model_backward_generation_materialization": [0, 0, 0, 0],
        "entries": entries,
        "entry_count": len(entries),
        "numerical_lock_sha256": sha256_file(LOCK_ROOT / LOCK_FILE),
        "stream_seal_sha256": "550600af120070594078b73a5da7d066ad6287d8a3224c55f61babca592e0586",
    }
    payload = {**value, "root_digest": canonical_hash(value)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    destination = LOCK_ROOT / SOURCE_MANIFEST_TECH_R3
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
