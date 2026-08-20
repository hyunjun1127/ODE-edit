#!/usr/bin/env python3
"""Build the rooted source manifest for the target-depth extension."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (
    REPO_ROOT
    / "project/run_scripts/ode_bf/locks/"
    "source_manifest_s05_p1r52_target_depth_il8_il10_il15_inner_telemetry_tech_r1.json"
)
FILES = (
    "project/run_scripts/ode_bf/locks/numerical_lock_s05_p1r52_target_depth_il8_il10_il15.json",
    "project/run_scripts/ode_bf/p1_runtime.py",
    "project/run_scripts/ode_bf/p1_scalable_batched_experiment.py",
    "project/run_scripts/ode_bf/p1r34_w_anchored_finite_demand.py",
    "project/run_scripts/ode_bf/p1r36_independent_b10x10_runtime.py",
    "project/run_scripts/ode_bf/p1r52_independent_runtime.py",
    "project/run_scripts/ode_bf/p1r52_r42_safe_kdc.py",
    "project/run_scripts/ode_bf/p1r52_target_depth.py",
    "project/run_scripts/ode_bf/p1r52_target_depth_inner_telemetry.py",
    "project/run_scripts/ode_bf/p1r52_target_depth_extension.py",
    "project/run_scripts/ode_bf/p1r52_target_depth_extension_panel.py",
    "project/run_scripts/ode_bf/tests/test_p1r52_target_depth.py",
    "project/run_scripts/session05_ode_bf_p1r52_target_depth_extension_atomic.py",
    "project/run_scripts/session05_ode_bf_p1r52_target_depth_extension_atomic.sbatch",
    "project/run_scripts/session05_ode_bf_p1r52_target_depth_extension_atomic_dry_plan.py",
    "project/run_scripts/session05_ode_bf_p1r52_target_depth_extension_build_manifest.py",
    "project/run_scripts/session05_ode_bf_submit_p1r52_target_depth_extension_atomic.py",
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
        "schema_version": (
            "ode-edit-s05-p1r52-target-depth-extension-source-manifest-inner-telemetry-tech-r1/v1"
        ),
        "instruction_id": "ODEEDIT-S05-P1R52-TARGET-DEPTH-IL8-IL10-IL15-V1",
        "source_parent": "0ac0aa2fec7d804b1cb14dfdea28bf6a5f089ec4",
        "technical_parent": "e801ec264d46cf58ea46c042d9dc81f2bd216a89",
        "entries": entries,
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["root_digest"] = hashlib.sha256(canonical).hexdigest()
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    OUTPUT.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(hashlib.sha256(payload).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
