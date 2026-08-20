#!/usr/bin/env python3
"""Build the rooted P1R52 target-depth execution source manifest."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "project/run_scripts/ode_bf/locks/source_manifest_s05_p1r52_target_depth_il1_il3full.json"
FILES = (
    "project/run_scripts/ode_bf/p1_runtime.py",
    "project/run_scripts/ode_bf/p1r36_independent_b10x10_runtime.py",
    "project/run_scripts/ode_bf/p1_scalable_batched_experiment.py",
    "project/run_scripts/ode_bf/p1r52_r42_safe_kdc.py",
    "project/run_scripts/ode_bf/p1r52_independent_runtime.py",
    "project/run_scripts/ode_bf/p1r52_target_depth.py",
    "project/run_scripts/ode_bf/p1r52_target_depth_panel.py",
    "project/run_scripts/ode_bf/p1r52_target_depth_target_only_runtime.py",
    "project/run_scripts/ode_bf/p2r1_target_only_runtime.py",
    "project/run_scripts/ode_bf/tests/test_p1r52_target_depth.py",
    "project/run_scripts/session05_ode_bf_p1r52_target_depth.py",
    "project/run_scripts/session05_ode_bf_p1r52_target_depth_atomic.py",
    "project/run_scripts/session05_ode_bf_p1r52_target_depth_atomic.sbatch",
    "project/run_scripts/session05_ode_bf_p1r52_target_depth_atomic_dry_plan.py",
    "project/run_scripts/session05_ode_bf_p1r52_target_depth.sbatch",
    "project/run_scripts/session05_ode_bf_p1r52_target_depth_dry_plan.py",
    "project/run_scripts/session05_ode_bf_submit_p1r52_target_depth.py",
    "project/run_scripts/session05_ode_bf_submit_p1r52_target_depth_atomic.py",
)


def main() -> int:
    entries = []
    for relative in sorted(FILES):
        path = REPO_ROOT / relative
        payload = path.read_bytes()
        entries.append(
            {
                "path": relative,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    value = {
        "schema_version": "ode-edit-s05-p1r52-target-depth-source-manifest/v1",
        "instruction_id": "ODEEDIT-S05-P1R52-TARGET-DEPTH-IL1-IL3FULL-V1",
        "source_parent": "d4e3b57d6e5fb0e082d1b51bfe15064458422143",
        "entries": entries,
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["root_digest"] = hashlib.sha256(canonical).hexdigest()
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    OUTPUT.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(hashlib.sha256(payload).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
