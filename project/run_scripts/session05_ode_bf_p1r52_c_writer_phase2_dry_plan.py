#!/usr/bin/env python3
from __future__ import annotations
import json
from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1r52_c_writer_kstep_independent import ARMS, role_for_cell


def build_plan() -> dict[str, object]:
    payload: dict[str, object] = {"schema": "ode-edit-s05-p1r52-c-writer-phase2-dry-plan/v1", "array": "0-2%3", "project_gpu_cap": 3, "rows": [{"cell": index, "arm": arm, "role": role_for_cell(index), "independent_case_count": 10, "request_count_per_case": 100, "K_writer_calls_per_case": 8, "alpha_cache_status": "ALPHA_CACHE_OFF_CONTROL", "W0_restore_per_case": True, "full_fp32": True, "paired_final_v6_one_shot_denominator": "10_OF_10_REQUIRED", "same_entry_slice_order_required": True, "same_z_equivalence_claimed": False} for index, arm in enumerate(ARMS)], "bf16_fp16_autocast_count": 0, "phase1_terminal_required_before_submit": True}
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(build_plan(), sort_keys=True))
