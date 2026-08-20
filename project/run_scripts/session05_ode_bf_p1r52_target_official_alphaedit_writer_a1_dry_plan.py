"""Raw-free dry plan for P1R52 target + Official AlphaEdit writer A1."""

from __future__ import annotations

from project.run_scripts.ode_bf.p1r52_target_official_alphaedit_writer import (
    INSTRUCTION_ID,
    PHASE_A_CASE_COUNT,
    PHASE_A_TECH_R2_RESULT_NAME,
    PHASE_A_ROLE,
)


def build_plan(source_head: str) -> dict[str, object]:
    return {
        "schema": "ode-edit-s05-p1r52-target-official-alphaedit-writer-a1-dry-plan/v1",
        "instruction_id": INSTRUCTION_ID,
        "source_head": source_head,
        "jobs": [
            {
                "role": PHASE_A_ROLE,
                "model": "llama3-8b-inst",
                "result_name": PHASE_A_TECH_R2_RESULT_NAME,
                "attempt_suffix": "tech-r2",
                "phase_a_pilot_case_count": 1,
                "phase_a_conditional_total_case_count": PHASE_A_CASE_COUNT,
                "batch_size": 100,
                "gpu_count": 1,
            }
        ],
        "same_accepted_z_writer_count": 2,
        "native_alphaedit_compute_z_call_count": 0,
        "batch_entry_evaluator_count": 0,
        "structural_h_p_energy_pir_piru_fpiq_influence_count": 0,
        "retry_count": 0,
        "phase_b_status": "CONDITIONAL_ON_COMPLETE_PHASE_A_TRANSFER_IMPROVEMENT",
    }


if __name__ == "__main__":
    import json
    import subprocess

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()
    print(json.dumps(build_plan(head), sort_keys=True, separators=(",", ":")))
