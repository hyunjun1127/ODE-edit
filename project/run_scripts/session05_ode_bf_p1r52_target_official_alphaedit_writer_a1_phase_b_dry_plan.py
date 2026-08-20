"""Raw-free released Phase-B dry plan."""

from __future__ import annotations

from project.run_scripts.ode_bf.p1r52_target_official_alphaedit_writer import (
    INSTRUCTION_ID,
    PHASE_B_RESULT_NAME,
    PHASE_B_ROLE,
)


def build_plan(source_head: str) -> dict[str, object]:
    return {
        "schema": "ode-edit-s05-p1r52-target-official-alphaedit-writer-a1-phase-b-dry-plan/v1",
        "instruction_id": INSTRUCTION_ID,
        "source_head": source_head,
        "jobs": [
            {
                "role": PHASE_B_ROLE,
                "model": "llama3-8b-inst",
                "result_name": PHASE_B_RESULT_NAME,
                "batch_count": 10,
                "batch_size": 100,
                "request_count": 1000,
                "gpu_count": 1,
            }
        ],
        "phase_a_status": "RELEASE",
        "native_alphaedit_compute_z_call_count": 0,
        "batch_entry_evaluator_count": 0,
        "structural_h_p_energy_pir_piru_fpiq_influence_count": 0,
        "physical_w_persistence": "W0_TO_W10",
        "official_alphaedit_cache_c_continuity": True,
        "retry_count": 0,
    }


if __name__ == "__main__":
    import json
    import subprocess

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    print(json.dumps(build_plan(head), sort_keys=True, separators=(",", ":")))
