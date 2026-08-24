#!/usr/bin/env python3
"""Outcome-free four-cell realization-reset execution plan."""

from __future__ import annotations

import json

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1r54_realization_reset import CELLS, source_equivalence_receipt


def build_plan() -> dict[str, object]:
    cells = []
    for config in CELLS:
        sequential = config.cell >= 2
        cells.append(
            {
                **config.raw_free_payload(),
                "canonical_batches": list(range(1, 11)) if sequential else [1],
                "request_count": 1000 if sequential else 100,
                "entry_weight": "W0" if not sequential else "W0_THEN_SEQUENTIAL_COMMIT_CHAIN",
                "entry_cache": "COLD" if not sequential else "COLD_THEN_SEQUENTIAL_ALPHA_CACHE",
                "K_per_batch": 8,
                "target_h": 0.125,
                "target_microsteps_per_outer": 1,
                "writer_calls_per_batch": 8,
                "writer_layer_applies_per_batch": 40,
                "terminal_commanded_z_source": "LAST_COMMANDED_TARGET",
                "next_controller_anchor_source": "POST_WRITE_NEXT_PHYSICAL_TERMINAL_Z",
                "source_equivalence": source_equivalence_receipt(config.cell),
            }
        )
    payload: dict[str, object] = {
        "schema": "ode-edit-s05-p1r54-realization-reset-dry-plan/v1",
        "model": "llama3-8b-inst",
        "full_fp32": True,
        "offline": True,
        "array": "0-3%4",
        "project_gpu_cap": 4,
        "gpu_per_cell": 1,
        "cpu_per_cell": 8,
        "memory_mib_per_cell": 65000,
        "walltime_hours_per_cell": 48,
        "independent_cell_count": 2,
        "sequential_cell_count": 2,
        "new_continuation_run_count": 0,
        "sample_payload_count": 1,
        "sample_duplication_count": 0,
        "reset_added_model_forward_count": 0,
        "reset_added_backward_count": 0,
        "reset_added_materialization_count": 0,
        "reset_added_evaluator_count": 0,
        "cells": cells,
        "scientific_promotion": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(build_plan(), sort_keys=True))
