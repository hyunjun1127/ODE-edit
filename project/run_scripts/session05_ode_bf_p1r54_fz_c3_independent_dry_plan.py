#!/usr/bin/env python3
"""Outcome-free P1R54 FZ C3 independent array execution plan."""

from __future__ import annotations

import json

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1r54_fz_c3_independent import (
    ROLES,
    STREAM_ORDER,
    STREAM_ROOT,
    atomic_source_equivalence,
    cell_config,
)


def build_plan() -> dict[str, object]:
    cells = []
    for cell in range(10):
        config = cell_config(cell)
        cells.append(
            {
                "cell": cell,
                "canonical_batch_index": config.batch_index,
                "canonical_batch_label": f"B{config.batch_index}",
                "role": config.role,
                "result_name": config.result_name,
                "result_root_policy": "CREATE_ONCE_CELL_LOCAL",
                "log_root_policy": "CREATE_ONCE_CELL_LOCAL",
                "state_root_policy": "CREATE_ONCE_CELL_LOCAL",
                "weight_entry": "EXACT_BASE_W0_VERSION0",
                "alpha_cache_entry": "COLD_ENTRY_VERSION0_WIDTH0",
                "K_writer_calls": 8,
                "writer_layer_applies": 40,
                "cache_entry_reuse_count": 8,
                "cache_append_after_successful_K8_count": 1,
                "endpoint_evaluation_outer_indices": [8],
                "terminal_W0_pointer_bytes_restore": True,
                "cross_batch_state_consumption_count": 0,
                "atomic_source_equivalence": atomic_source_equivalence(
                    config.batch_index
                ),
            }
        )
    payload: dict[str, object] = {
        "schema": "ode-edit-s05-p1r54-fz-c3-independent-dry-plan/v1",
        "model": "llama3-8b-inst",
        "execution_mode": "INDEPENDENT_BATCH",
        "roles": list(ROLES),
        "slurm": {
            "array": "0-9%3",
            "cell_count": 10,
            "max_concurrent_array_cells": 3,
            "gpu_per_cell": 1,
            "cpu_per_cell": 8,
            "memory_mib_per_cell": 65000,
            "walltime_hours_per_cell": 48,
            "project_gpu_cap": 4,
            "preserved_sequential_gpu_count": 1,
            "max_active_plus_new_gpu_count": 4,
        },
        "stream_binding": "SINGLE_CANONICAL_STREAM_REFERENCE_ONLY",
        "stream_root": STREAM_ROOT,
        "order_root": STREAM_ORDER,
        "batches": list(range(1, 11)),
        "request_count_per_cell": 100,
        "total_request_count": 1000,
        "sample_payload_count": 1,
        "sample_payload_copy_count": 0,
        "sample_duplication_count": 0,
        "formula": "F_i,k=m_i,k*||z0_i||*d_i,k",
        "target_dt": 0.125,
        "target_microsteps_per_outer": 1,
        "target_horizon_per_cell": 1.0,
        "target_field_evaluations_per_cell": 8,
        "target_field_evaluations_total": 80,
        "writer": "C3_OFFICIAL_EASYEDIT_ALPHAEDIT_KSTEP",
        "writer_calls_per_cell": 8,
        "writer_calls_total": 80,
        "writer_layer_applies_per_cell": 40,
        "writer_layer_applies_total": 400,
        "weight_continuity": False,
        "alpha_cache_continuity": False,
        "history_factor_materializer_continuity": False,
        "within_cell_cache_semantics": "ATOMIC_BYTE_COMPATIBLE_BATCH_ENTRY_SNAPSHOT",
        "heldout_outer_indices_per_cell": [8],
        "retry_count": 0,
        "imputation_count": 0,
        "heldout_decision_influence_count": 0,
        "reference_execution_count": 0,
        "pdz_execution_count": 0,
        "qwen_execution_count": 0,
        "full_fp32": True,
        "offline": True,
        "sequential_job_mutation_count": 0,
        "scientific_promotion": False,
        "cells": cells,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(build_plan(), sort_keys=True))
