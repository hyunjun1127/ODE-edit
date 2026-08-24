#!/usr/bin/env python3
"""Outcome-free plan for P1R54 FZ final-z one-shot sequential."""

from __future__ import annotations

import json

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1r54_fz_c3_sequential import (
    STREAM_ORDER,
    STREAM_ROOT,
)
from project.run_scripts.ode_bf.p1r54_fz_finalz_oneshot_sequential import (
    RESULT_NAME,
    ROLE,
    control_source_equivalence,
)


def build_plan() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "ode-edit-s05-p1r54-fz-final-z-oneshot-dry-plan/v1",
        "model": "llama3-8b-inst",
        "role": ROLE,
        "result_name": RESULT_NAME,
        "slurm": {
            "job_count": 1,
            "gpu_per_job": 1,
            "cpu_per_job": 8,
            "memory_mib": 65000,
            "walltime_hours": 48,
            "project_gpu_cap": 4,
        },
        "stream_root": STREAM_ROOT,
        "order_root": STREAM_ORDER,
        "batches": list(range(1, 11)),
        "request_count_per_batch": 100,
        "total_request_count": 1000,
        "sample_payload_count": 1,
        "sample_duplication_count": 0,
        "formula": "F_i,k=m_i,k*||z0_i||*d_i,k",
        "target_clock_ordinals": list(range(8)),
        "target_dt": 0.125,
        "target_horizon_per_batch": 1.0,
        "target_field_evaluations_per_batch": 8,
        "writer_cadence": "FINAL_Z_ONESHOT",
        "writer_calls_per_batch": 1,
        "writer_calls_total": 10,
        "writer_layer_applies_per_batch": 5,
        "writer_layer_applies_total": 50,
        "no_writer_outer_indices": list(range(1, 8)),
        "writer_authority_outer_index": 8,
        "finite_demand_builder_calls_per_batch": 1,
        "materialization_count_per_batch": 1,
        "weight_continuity": True,
        "alpha_cache_continuity": True,
        "cache_append_per_successful_batch": 1,
        "heldout_outer_indices_per_batch": [8],
        "control_source_equivalence": control_source_equivalence(),
        "reference_execution_count": 0,
        "full_fp32": True,
        "scientific_promotion": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(build_plan(), sort_keys=True))
