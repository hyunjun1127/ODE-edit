#!/usr/bin/env python3
"""Outcome-free two-cell P1R53 execution plan."""

from __future__ import annotations

import json

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1r53_request_local_speed_b100 import role_for_cell


def build_plan() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "ode-edit-s05-p1r53-request-local-speed-b100-dry-plan/v1",
        "array": "0-1%2",
        "project_gpu_cap": 4,
        "array_concurrency": 2,
        "model": "llama3-8b-inst",
        "selected_batch": "B1",
        "request_count_per_cell": 100,
        "rows": [
            {
                "array_cell": index,
                "role": role_for_cell(index),
                "arm": arm,
                "target_dt": 0.125,
                "target_microsteps_per_outer": 1,
                "target_horizon": 1.0,
                "writer": "C3-OFFICIAL-EASYEDIT-ALPHAEDIT-KSTEP-CACHE",
                "writer_calls": 8,
                "writer_layer_applies": 40,
                "heldout_outer_indices": [8],
                "cache_entry_reuse_count": 8,
                "cache_append_count": 1,
                "native_execution_count": 0,
                "full_fp32": True,
            }
            for index, arm in enumerate(("LP-S", "LFD-E"))
        ],
        "sample_duplication_count": 0,
        "qwen_execution_count": 0,
        "scientific_promotion": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(build_plan(), sort_keys=True))
