#!/usr/bin/env python3
"""No-model deterministic plan for the S05 NEWNLL H/P soft-hard pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import MODEL_ALIASES, ODEBFContractError
from project.run_scripts.ode_bf.p1_newnll_p_soft_hard_panel import (
    NEWNLL_P_SOFT_HARD_INSTRUCTION_ID,
    NEWNLL_P_SOFT_HARD_PANEL_LABELS,
    expected_newnll_p_soft_hard_result_name,
    forecast_newnll_p_soft_hard_panel,
    newnll_p_soft_hard_panel_specs,
)


JOB_NAMES = {
    "llama3-8b-inst": "odeedit_s05_psoftp1r5r2_llama",
    "qwen2.5-7b-inst": "odeedit_s05_psoftp1r5r2_qwen",
}


def build_plan(
    source_head: str,
    *,
    repository_root: Path = REPO_ROOT,
) -> dict[str, object]:
    if len(source_head) != 40 or any(
        character not in "0123456789abcdef" for character in source_head
    ):
        raise ODEBFContractError("P-soft dry-plan head differs")
    artifact = (
        repository_root / "project/run_scripts/ode_bf/locks/p0_artifact_lock.json"
    )
    base = (
        repository_root / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json"
    )
    jobs = []
    for alias in MODEL_ALIASES:
        forecast = forecast_newnll_p_soft_hard_panel(artifact, base, alias)
        jobs.append({
            "alias": alias,
            "job_name": JOB_NAMES[alias],
            "result_name": expected_newnll_p_soft_hard_result_name(alias),
            "gpu": 1,
            "cpu": 8,
            "memory_mib": 65_000,
            "time": "24:00:00",
            "node": "server1",
            "forecast": forecast.raw_free_payload(),
            "forecast_identity": forecast.identity(),
        })
    specs = newnll_p_soft_hard_panel_specs()
    return {
        "schema": "ode-edit-s05-newnll-p-soft-hard-p1r5-dry-plan/v1",
        "instruction_id": NEWNLL_P_SOFT_HARD_INSTRUCTION_ID,
        "source_head": source_head,
        "benchmark": "CounterFact",
        "scientific_sample_reused_for_causal_diagnostic": True,
        "scientific_promotion_authorized": False,
        "edit_batch_size": 10,
        "sequential_batch_count": 1,
        "server1_project_gpu_cap": 3,
        "new_pair_gpu": 2,
        "panel_labels": list(NEWNLL_P_SOFT_HARD_PANEL_LABELS),
        "clock_variant_by_label": {
            item.label: item.clock_variant.value for item in specs
        },
        "routing_objective_by_label": {
            item.label: item.routing_objective.value for item in specs
        },
        "functional_p_decision_by_label": {
            item.label: item.functional_p_decision.value for item in specs
        },
        "functional_p_field_policy_by_label": {
            item.label: item.functional_p_field_policy.value for item in specs
        },
        "preservation_constraints_by_label": {
            item.label: item.preservation_constraints.value for item in specs
        },
        "generic_hp_soft_capable": True,
        "historical_soft_active": False,
        "historical_soft_reason": "EMPTY_HISTORY",
        "actual_functional_p_hard_gate_pctrl_and_psoft": True,
        "actual_functional_h_hard_gate_all_arms": True,
        "first_hit_observation_only": True,
        "full_tau_target": True,
        "heldout_controller_access_count": 0,
        "generation_call_count": 0,
        "model_alias_specific_controller_branches": 0,
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
    print(json.dumps(
        build_plan(args.source_head, repository_root=args.repository_root),
        sort_keys=True,
        separators=(",", ":"),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
