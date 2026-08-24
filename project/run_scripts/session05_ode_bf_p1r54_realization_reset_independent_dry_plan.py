#!/usr/bin/env python3
"""Outcome-free B2-B10 independent reset expansion plan."""

from __future__ import annotations

from typing import Any, Mapping

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1r54_realization_reset import (
    INDEPENDENT_EXPANSION_CELLS,
    INSTRUCTION_ID,
)


def build_plan() -> Mapping[str, Any]:
    cells = []
    for config in INDEPENDENT_EXPANSION_CELLS:
        cells.append(
            {
                **config.raw_free_payload(),
                "request_count": 100,
                "K": 8,
                "target_h": 0.125,
                "writer_calls": 8,
                "writer_layer_applies": 40,
                "entry_W": "EXACT_W0",
                "entry_alpha_cache": "EXACT_COLD",
                "terminal_W0_restore": True,
                "reset_added_forward_backward_materialization_evaluator": [0, 0, 0, 0],
            }
        )
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-realization-reset-independent-expansion-plan/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "DRY_PLAN_PASS",
        "array": "0-17%2",
        "project_gpu_cap": 4,
        "concurrent_gpu_count": 2,
        "already_frozen_B1_per_arm": 1,
        "new_batch_indices": list(range(2, 11)),
        "new_cell_count": 18,
        "new_request_count": 1800,
        "sample_payload_count": 1,
        "sample_duplication_count": 0,
        "cells": cells,
        "scientific_promotion": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


if __name__ == "__main__":
    import json

    print(json.dumps(build_plan(), sort_keys=True))
