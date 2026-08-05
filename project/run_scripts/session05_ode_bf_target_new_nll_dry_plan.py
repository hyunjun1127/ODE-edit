#!/usr/bin/env python3
"""No-model deterministic plan for the S05 server1 causal pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import MODEL_ALIASES, ODEBFContractError
from project.run_scripts.ode_bf.p1_target_new_panel import (
    TARGET_NEW_INSTRUCTION_ID,
    TARGET_NEW_PANEL_LABELS,
    expected_target_new_result_name,
    forecast_target_new_panel,
    target_new_panel_specs,
)


JOB_NAMES = {
    "llama3-8b-inst": "odebf_s05_newnll_llama",
    "qwen2.5-7b-inst": "odebf_s05_newnll_qwen",
}


def build_plan(source_head: str, *, repository_root: Path = REPO_ROOT) -> dict[str, object]:
    if len(source_head) != 40 or any(
        character not in "0123456789abcdef" for character in source_head
    ):
        raise ODEBFContractError("target-new dry-plan source head differs")
    artifact = repository_root / "project/run_scripts/ode_bf/locks/p0_artifact_lock.json"
    base = repository_root / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json"
    jobs = []
    for alias in MODEL_ALIASES:
        forecast = forecast_target_new_panel(artifact, base, alias)
        jobs.append(
            {
                "alias": alias,
                "job_name": JOB_NAMES[alias],
                "result_name": expected_target_new_result_name(alias),
                "gpu": 1,
                "cpu": 8,
                "memory_mib": 65_000,
                "time": "24:00:00",
                "node": "server1",
                "forecast": forecast.raw_free_payload(),
                "forecast_identity": forecast.identity(),
            }
        )
    specs = target_new_panel_specs()
    return {
        "schema": "ode-edit-s05-target-new-nll-routing-dry-plan/v1",
        "instruction_id": TARGET_NEW_INSTRUCTION_ID,
        "source_head": source_head,
        "benchmark": "CounterFact",
        "scientific_sample_reused_for_causal_diagnostic": True,
        "scientific_promotion_authorized": False,
        "edit_batch_size": 10,
        "sequential_batch_count": 1,
        "server1_gpu_cap": 3,
        "panel_labels": list(TARGET_NEW_PANEL_LABELS),
        "fr_a16_newnll": {
            "executed": False,
            "forecast_seconds": 161_920,
            "allocation_seconds": 86_400,
            "exclusion_reason": "SIX_CONTEXT_A16_EXCEEDS_LOCKED_24H_ENVELOPE",
            "outcome_metric_used": False,
        },
        "clock_variant_by_label": {
            item.label: item.clock_variant.value for item in specs
        },
        "routing_objective_by_label": {
            item.label: item.routing_objective.value for item in specs
        },
        "routing_context_group_sizes": [1, 5],
        "routing_context_count": 6,
        "routing_context_weighting": "uniform-over-all-six-locked-rendered-contexts",
        "target_state_objective": "MARGIN_LOCKED",
        "old_nll_decision_influence_count_newnll": 0,
        "stepwise_layer_routing_telemetry": {
            "schema": "ode-edit-stepwise-layer-routing-telemetry/v1",
            "trajectory_schema": (
                "ode-edit-stepwise-layer-routing-trajectory/v1"
            ),
            "candidate_layer_ids": [4, 5, 6, 7, 8],
            "defined_for_labels": [
                "FR-A8-MARGIN",
                "FR-A8-NEWNLL",
                "FR-A16-NEWNLL",
            ],
            "fr_a16_executed": False,
            "receipt_categories": [
                "field",
                "trial",
                "transition",
                "accepted",
                "first-hit",
                "terminal",
            ],
            "distribution_normalizations": {
                "coefficient": "absolute-l1",
                "predicted_progress": "positive-only",
                "prequantized_energy": "nonnegative-energy",
                "realized_bf16_energy": "nonnegative-energy",
            },
            "observation_only": True,
            "controller_dependency_count": 0,
            "diagnostic_severe_concentration_only": True,
        },
        "postfreeze_states": "W0-N32-and-every-unique-accepted-snapshot",
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
