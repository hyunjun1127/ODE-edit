#!/usr/bin/env python3
from __future__ import annotations

import json

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1r52_c_writer_kstep_cache_sequential import CACHE_ARMS, role_for_cell


def build_plan() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "ode-edit-s05-p1r52-c-writer-phase3-dry-plan/v1",
        "array": "0-2%3",
        "project_gpu_cap": 3,
        "rows": [
            {
                "cell": index,
                "arm": arm,
                "role": role_for_cell(index),
                "sequential_batch_count": 10,
                "request_count_per_batch": 100,
                "K_writer_calls_per_batch": 8,
                "alpha_cache_status": "ALPHA_CACHE_CONTINUITY_ON_BATCH_ENTRY_SNAPSHOT",
                "cache_append_once_after_K8": True,
                "same_batch_current_key_history_inclusion_count": 0,
                "terminal_W0_restore": True,
                "full_fp32": True,
            }
            for index, arm in enumerate(CACHE_ARMS)
        ],
        "bf16_fp16_autocast_count": 0,
        "phase2_afterok_dependency_required": True,
        "phase2_terminal_valid_gate_before_model_load": True,
        "phase2_expected_valid_case_count": 30,
        "phase2_expected_valid_request_count": 3000,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(build_plan(), sort_keys=True))
