#!/usr/bin/env python3
"""Create-once numerical lock for accepted-z observation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1r52_accepted_z_observation_panel import (
    INSTRUCTION_ID,
    LOCK_FILE,
    LOCK_SCHEMA,
    REFERENCE_NAMES,
    REFERENCE_TERMINAL_SHA256,
    ROLES,
    SOURCE_PARENT,
    SOURCE_PARENT_TREE,
)


def main() -> int:
    value = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": INSTRUCTION_ID,
        "source_parent": SOURCE_PARENT,
        "source_parent_tree": SOURCE_PARENT_TREE,
        "model": "llama3-8b-inst",
        "roles": list(ROLES),
        "round_count": 10,
        "batch_size": 100,
        "request_count_per_role": 1000,
        "stream_root": "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a",
        "stream_order": "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3",
        "native_target_layer": 8,
        "native_lookup": "subject_last",
        "overlay": "METHOD_NATIVE_ABSOLUTE_ACCEPTED_Z_REPLACEMENT",
        "reference_names": REFERENCE_NAMES,
        "reference_terminal_sha256": REFERENCE_TERMINAL_SHA256,
        "batch_entry_W_metrics_count": 0,
        "added_backward_count": 0,
        "added_generation_call_count": 0,
        "action_influence_count": 0,
        "duplicate_W_evaluator_model_forward_count": 0,
        "proxy_z_count": 0,
        "imputation_count": 0,
        "project_gpu_cap": 4,
        "stage_gpu_max": 4,
    }
    payload = {**value, "root_digest": canonical_hash(value)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    destination = REPO_ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
    print(json.dumps({"path": str(destination), "root": payload["root_digest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
