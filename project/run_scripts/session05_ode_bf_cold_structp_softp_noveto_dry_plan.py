#!/usr/bin/env python3
"""No-model deterministic plan for the fresh S05 cold pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import MODEL_ALIASES, ODEBFContractError
from project.run_scripts.ode_bf.p1_cold_structp_softp_noveto_panel import (
    COLD_INSTRUCTION_ID,
    COLD_PANEL_LABELS,
    expected_cold_result_name,
    forecast_cold_panel,
)


JOB_NAMES = {
    "llama3-8b-inst": "odeedit_s05_p1r6_cold_llama",
    "qwen2.5-7b-inst": "odeedit_s05_p1r6_cold_qwen",
}


def build_plan(
    source_head: str,
    *,
    repository_root: Path = REPO_ROOT,
) -> dict[str, object]:
    if len(source_head) != 40 or any(
        character not in "0123456789abcdef" for character in source_head
    ):
        raise ODEBFContractError("cold dry-plan head differs")
    artifact = (
        repository_root / "project/run_scripts/ode_bf/locks/p0_artifact_lock.json"
    )
    base = repository_root / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json"
    jobs = []
    for alias in MODEL_ALIASES:
        forecast = forecast_cold_panel(artifact, base, alias)
        jobs.append(
            {
                "alias": alias,
                "job_name": JOB_NAMES[alias],
                "result_name": expected_cold_result_name(alias),
                "gpu": 1,
                "cpu": 8,
                "memory_mib": 65_000,
                "time": "24:00:00",
                "node": "server1",
                "forecast": forecast.raw_free_payload(),
            }
        )
    return {
        "schema": "ode-edit-s05-cold-structp-softp-noveto-p1r6-dry-plan/v1",
        "instruction_id": COLD_INSTRUCTION_ID,
        "source_head": source_head,
        "benchmark": "CounterFact",
        "fresh_sealed_b10": True,
        "scientific_promotion_authorized": False,
        "edit_batch_size": 10,
        "sequential_batch_count": 1,
        "server1_project_gpu_cap": 3,
        "new_pair_gpu": 2,
        "panel_labels": list(COLD_PANEL_LABELS),
        "routing_objective": "TARGET_NEW_NLL",
        "target_initialization": "canonical-W0-z-base",
        "native_or_direct_z_cold_access_count": 0,
        "structural_p_hard": True,
        "functional_p_candidate_veto_influence_count": 0,
        "functional_p_field_policy_by_label": {
            COLD_PANEL_LABELS[0]: "PROBE_ONLY",
            COLD_PANEL_LABELS[1]: "SOFT_HARD",
        },
        "first_hit_observation_only": True,
        "full_tau_target": 1.0,
        "no_p_trust_factorial": True,
        "heldout_controller_access_count": 0,
        "generation_call_count": 0,
        "model_alias_specific_controller_branches": 0,
        "pre_submit_review_hold": True,
        "model_load": False,
        "gpu_use": False,
        "slurm_submit": False,
        "result_root_creation": False,
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
