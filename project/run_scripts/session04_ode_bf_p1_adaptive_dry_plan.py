#!/usr/bin/env python3
"""Deterministic no-model plan for the adaptive-tau P1R4 R2 causal pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import MODEL_ALIASES, ODEBFContractError
from project.run_scripts.ode_bf.p1_adaptive import (
    ADAPTIVE_INSTRUCTION_ID,
    ADAPTIVE_VARIANTS,
    adaptive_lock,
)
from project.run_scripts.ode_bf.p1_runtime import (
    expected_p1r4_adaptive_result_name,
)
from project.run_scripts.ode_bf.resource import (
    forecast_p1_adaptive_b10_memory,
    forecast_p1_adaptive_b10_time,
)


JOB_NAMES = {
    "llama3-8b-inst": "odebf_s04_p1r4adaptive_r2_llama",
    "qwen2.5-7b-inst": "odebf_s04_p1r4adaptive_r2_qwen",
}
REPAIR_INSTRUCTION_ID = "ODEEDIT-S04-ODE-BF-P1R4-RISK-RECEIPT-R2-V1"


def build_plan(
    source_head: str,
    *,
    repository_root: Path = REPO_ROOT,
) -> dict[str, object]:
    if len(source_head) != 40 or any(
        character not in "0123456789abcdef" for character in source_head
    ):
        raise ODEBFContractError("adaptive dry-plan source head differs")
    package = repository_root / "project/run_scripts/ode_bf"
    artifact = package / "locks/p0_artifact_lock.json"
    base = repository_root / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json"
    time_forecast = forecast_p1_adaptive_b10_time()
    jobs = []
    for alias in MODEL_ALIASES:
        memory = forecast_p1_adaptive_b10_memory(artifact, base, alias)
        jobs.append(
            {
                "alias": alias,
                "job_name": JOB_NAMES[alias],
                "result_name": expected_p1r4_adaptive_result_name(alias),
                "gpu": 1,
                "cpu": 8,
                "memory_mib": 65_000,
                "time": "24:00:00",
                "node": "server2",
                "memory_forecast": memory.raw_free_payload(),
                "memory_forecast_identity": memory.identity(),
                "time_forecast": time_forecast.raw_free_payload(),
                "time_forecast_identity": time_forecast.identity(),
            }
        )
    return {
        "schema": "ode-edit-s04-ode-bf-p1r4-adaptive-r2-dry-plan/v1",
        "instruction_id": REPAIR_INSTRUCTION_ID,
        "scientific_instruction_id": ADAPTIVE_INSTRUCTION_ID,
        "authorized_attempt": "RISK_RECEIPT_REPAIR_R2_CAUSAL_PAIR",
        "source_head": source_head,
        "benchmark": "CounterFact",
        "scientific_sample_reused_for_causal_diagnostic": True,
        "scientific_promotion_authorized": False,
        "edit_batch_size": 10,
        "sequential_batch_count": 1,
        "common_controller": "R_BF",
        "variants": [item.value for item in ADAPTIVE_VARIANTS],
        "variant_lock_sha256": {
            item.value: adaptive_lock(item).identity()
            for item in ADAPTIVE_VARIANTS
        },
        "full_residual_variants": ["FR-A8", "FR-A16"],
        "legacy_pre_share_variants": ["PS-S8", "PS-A8"],
        "adaptive_same_state_retry_variants": ["PS-A8", "FR-A8", "FR-A16"],
        "accepted_state_field_refresh_only": True,
        "reject_advances_tau": False,
        "reject_advances_accepted_index": False,
        "reject_changes_trust_radius": False,
        "target_weight_shared_delta_tau": True,
        "postfreeze_states": "W0-N32-and-every-unique-accepted-snapshot",
        "rejected_retry_duplicate_evaluation_count": 0,
        "model_h_1_over_32_authorized": False,
        "persistent_endpoint_commit_count": 0,
        "history_append_count": 0,
        "heldout_controller_access_count": 0,
        "generation_call_count": 0,
        "model_load": False,
        "gpu_use": False,
        "slurm_submit": False,
        "retry_or_resubmit": False,
        "jobs": jobs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    print(
        json.dumps(
            build_plan(args.source_head, repository_root=args.repository_root),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
