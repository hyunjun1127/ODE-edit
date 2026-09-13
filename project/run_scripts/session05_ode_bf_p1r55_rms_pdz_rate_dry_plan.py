#!/usr/bin/env python3
"""Outcome-free P1R55 Phase-1 warm-state 2x2 plan."""

from __future__ import annotations

import json

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1r55_rms_pdz_rate_experiment import (
    PHASE1_ARRAY,
    PHASE1_CELLS,
)


def build_plan() -> dict[str, object]:
    cells = []
    for config in PHASE1_CELLS:
        cells.append(
            {
                **config.raw_free_payload(),
                "canonical_prefix_batches": list(range(1, config.snapshot_batch)),
                "scientific_probe_batch": config.snapshot_batch,
                "scientific_probe_request_count": 100,
                "K": 8,
                "target_h": 0.125,
                "target_microsteps_per_outer": 1,
                "target_horizon": 1.0,
                "heldout_accepted_z_outer_indices": [1, 4, 8],
                "writer_calls_scientific_probe": 8,
                "writer_layer_applies_scientific_probe": 40,
                "entry_weight": (
                    "W0" if config.snapshot_batch == 1 else "CANONICAL_A0_WARM_ENTRY"
                ),
                "entry_cache": (
                    "COLD"
                    if config.snapshot_batch == 1
                    else "CANONICAL_A0_HISTORICAL_ALPHA_CACHE"
                ),
                "exact_restore_after_probe": True,
            }
        )
    payload: dict[str, object] = {
        "schema": "ode-edit-s05-p1r55-rms-pdz-rate-phase1-dry-plan/v1",
        "model": "llama3-8b-inst",
        "full_fp32": True,
        "offline": True,
        "array": PHASE1_ARRAY,
        "project_gpu_cap": 3,
        "gpu_per_cell": 1,
        "cell_count": 12,
        "snapshot_count": 3,
        "arm_count": 4,
        "sample_payload_count": 1,
        "sample_duplication_count": 0,
        "extra_forward_count": 0,
        "extra_backward_count": 0,
        "per_request_backward_loop_count": 0,
        "cross_request_strength_decision_count": 0,
        "cells": cells,
        "scientific_promotion": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(build_plan(), sort_keys=True))
