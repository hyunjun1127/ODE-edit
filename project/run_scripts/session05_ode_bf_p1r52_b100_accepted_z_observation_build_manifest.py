#!/usr/bin/env python3
"""Create-once source manifest for accepted-z observation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.artifacts import sha256_file
from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1r52_accepted_z_observation_panel import (
    INSTRUCTION_ID,
    LOCK_FILE,
    MANIFEST_FILE,
    MANIFEST_SCHEMA,
    SOURCE_PARENT,
    SOURCE_PARENT_TREE,
)


SOURCE_PATHS = (
    "project/run_scripts/ode_bf/p1_runtime.py",
    "project/run_scripts/ode_bf/p1r52_sequential_runtime.py",
    "project/run_scripts/ode_bf/p1r52_accepted_z_observation.py",
    "project/run_scripts/ode_bf/p1r52_accepted_z_observation_panel.py",
    "project/run_scripts/ode_bf/tests/test_p1r52_accepted_z_observation.py",
    "project/run_scripts/session05_ode_bf_p1r52_b100_accepted_z_observation.py",
    "project/run_scripts/session05_ode_bf_p1r52_b100_accepted_z_observation_dry_plan.py",
    "project/run_scripts/session05_ode_bf_p1r52_b100_accepted_z_observation.sbatch",
    "project/run_scripts/session05_ode_bf_submit_p1r52_b100_accepted_z_observation.py",
    "project/run_scripts/session05_ode_bf_p1r52_b100_accepted_z_observation_build_lock.py",
    "project/run_scripts/session05_ode_bf_p1r52_b100_accepted_z_observation_build_manifest.py",
    f"project/run_scripts/ode_bf/locks/{LOCK_FILE}",
)


def main() -> int:
    entries = []
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        entries.append(
            {"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)}
        )
    value = {
        "schema_version": MANIFEST_SCHEMA,
        "instruction_id": INSTRUCTION_ID,
        "source_parent": SOURCE_PARENT,
        "source_parent_tree": SOURCE_PARENT_TREE,
        "scientific_method_delta_count": 0,
        "observation_module_count": 1,
        "entries": entries,
        "entry_count": len(entries),
        "numerical_lock_sha256": sha256_file(
            REPO_ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        ),
    }
    payload = {**value, "root_digest": canonical_hash(value)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    destination = REPO_ROOT / "project/run_scripts/ode_bf/locks" / MANIFEST_FILE
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
    print(json.dumps({"path": str(destination), "root": payload["root_digest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
