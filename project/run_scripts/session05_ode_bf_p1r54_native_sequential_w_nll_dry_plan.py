#!/usr/bin/env python3
"""Outcome-free plan for two native sequential final-W NLL backfill cells."""

from __future__ import annotations

import json

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1r54_native_sequential_w_nll import (
    CELLS,
    STREAM_ORDER,
    STREAM_ROOT,
    source_equivalence_receipt,
)


def build_plan() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "ode-edit-s05-p1r54-native-sequential-w-nll-dry-plan/v1",
        "array": "0-1%2",
        "project_gpu_cap": 4,
        "gpu_per_cell": 1,
        "cpu_per_cell": 8,
        "memory_mib_per_cell": 65000,
        "walltime_hours": 48,
        "model": "llama3-8b-inst",
        "dtype": "FULL_FP32",
        "stream_root": STREAM_ROOT,
        "order_root": STREAM_ORDER,
        "batches": [f"B{index}" for index in range(1, 11)],
        "request_count_per_batch": 100,
        "request_count_per_cell": 1000,
        "sample_payload_count": 1,
        "sample_duplication_count": 0,
        "sequential_weight_continuity": True,
        "cells": [
            {
                "cell": item.cell,
                "label": item.label,
                "role": item.role,
                "result_name": item.result_name,
                "scientific_call_path": item.scientific_call_path,
                "alpha_cache_semantics": item.alpha_cache_semantics,
                "required_final_w10_request_rows": 1000,
                "required_rewrite_nll": True,
                "required_rephrase_nll": True,
            }
            for item in CELLS
        ],
        "source_equivalence": source_equivalence_receipt(),
        "independent_baseline_rerun_count": 0,
        "fz_sequential_rerun_count": 0,
        "scientific_promotion": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(build_plan(), sort_keys=True))
