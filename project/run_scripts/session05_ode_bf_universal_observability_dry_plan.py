#!/usr/bin/env python3
"""No-model execution-authorized A2 plan for Llama R13 cell pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.ode_bf_observability import (
    R13_LIVE_CELL_IDS,
    paired_arm_ids,
    universal_observability_contract_receipt,
)
from project.run_scripts.ode_bf.p1_universal_observability_panel import (
    COMMON_COLD_CASE_SEAL_FILE,
    UNIVERSAL_OBS_ALLOCATION_SECONDS,
    UNIVERSAL_OBS_AMENDMENT_ID,
    UNIVERSAL_OBS_BASE_RUNTIME,
    UNIVERSAL_OBS_FORECAST_SECONDS,
    UNIVERSAL_OBS_INSTRUCTION_ID,
    UNIVERSAL_OBS_LOCK_FILE,
    UNIVERSAL_OBS_RESULT_TOKEN,
    UNIVERSAL_OBS_SCHEMA_NAMESPACE,
    common_cold_schedule,
    expected_universal_observability_result_name,
    forecast_universal_observability_panel,
    load_and_validate_r12_frozen_reference,
    load_and_validate_universal_observability_lock,
    verify_common_cold_case_seal,
)
from project.run_scripts.ode_bf.sampling import load_p1_sampling_seal


LLAMA_ALIAS = "llama3-8b-inst"
SERVER1_PROJECT_GPU_CAP = 4
CELL_ORDER = R13_LIVE_CELL_IDS
JOB_NAMES = {
    "RS-NEUTRAL": "odeedit_s05_r13a2_sh1_rsn_llama",
    "RS-SOFT": "odeedit_s05_r13a2_sh1_rss_llama",
    "BG-NEUTRAL": "odeedit_s05_r13a2_sh1_bgn_llama",
}
SESSION_ID = "019fe489-c968-75f3-9965-7cfbc26c0a99"
EXECUTION_BRANCH = "codex/odeeditsh1-s05-universal-obs-rs-bg-targethold-p1r13-v1"
EXECUTION_PARENT = UNIVERSAL_OBS_BASE_RUNTIME
EXECUTION_AMENDMENT = "A2"


def _valid_source_head(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def controller_geometry_identity() -> str:
    from project.run_scripts.ode_bf.p1_controller import P1ControllerLock

    return P1ControllerLock().identity()


def _job(
    *,
    cell_id: str,
    forecast: object,
) -> dict[str, object]:
    dynamic_arm, target_hold_arm = paired_arm_ids(cell_id)
    return {
        "alias": LLAMA_ALIAS,
        "cell_id": cell_id,
        "job_name": JOB_NAMES[cell_id],
        "result_name": expected_universal_observability_result_name(
            LLAMA_ALIAS, cell_id
        ),
        "run_token": UNIVERSAL_OBS_RESULT_TOKEN,
        "live_arms": [dynamic_arm, target_hold_arm],
        "dynamic_arm": dynamic_arm,
        "target_hold_arm": target_hold_arm,
        "one_alias_one_cell": True,
        "runtime_kwargs": {"universal_observability_cell": cell_id},
        "gpu": 1,
        "cpu": 8,
        "memory_mib": 65_000,
        "time": "23:59:00",
        "node": "server1",
        "forecast": forecast.raw_free_payload(),
    }


def build_plan(source_head: str, *, repository_root: Path = REPO_ROOT) -> dict[str, object]:
    if not _valid_source_head(source_head):
        raise ODEBFContractError("universal-observability dry-plan head differs")
    locks = repository_root / "project/run_scripts/ode_bf/locks"
    seal = verify_common_cold_case_seal(
        json.loads((locks / COMMON_COLD_CASE_SEAL_FILE).read_text(encoding="utf-8"))
    )
    population = json.loads(
        (locks / "p1r2_p_population_seal.json").read_text(encoding="utf-8")
    )
    base_schedule = load_p1_sampling_seal(
        locks / "p1r2_p_population_seal.json",
        stream_path=locks / "p1r2_seqb10_stream_seal.json",
    )
    schedule = common_cold_schedule(base_schedule)
    controller_identity = controller_geometry_identity()
    numerical_locks: dict[str, str] = {}
    numerical_root_digests: dict[str, str] = {}
    contracts: dict[str, dict[str, object]] = {}
    for cell_id in CELL_ORDER:
        numerical, numerical_sha256 = load_and_validate_universal_observability_lock(
            locks / UNIVERSAL_OBS_LOCK_FILE,
            controller_identity_sha256=controller_identity,
            case_root_digest=seal["root_digest"],
            population_root_digest=population["root_digest"],
            schedule=schedule,
            cell_id=cell_id,
        )
        numerical_locks[cell_id] = numerical_sha256
        numerical_root_digests[cell_id] = str(numerical["root_digest"])
        contracts[cell_id] = universal_observability_contract_receipt(
            instruction_id=UNIVERSAL_OBS_INSTRUCTION_ID,
            amendment_id=UNIVERSAL_OBS_AMENDMENT_ID,
            cell_id=cell_id,
        )
    frozen_r12_reference, r12_reference_sha256 = load_and_validate_r12_frozen_reference(
        locks,
        controller_identity_sha256=controller_identity,
        case_root_digest=seal["root_digest"],
        schedule=schedule,
        alias=LLAMA_ALIAS,
    )
    forecast = forecast_universal_observability_panel(
        locks / "p0_artifact_lock.json",
        repository_root / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json",
        LLAMA_ALIAS,
    )
    jobs = [_job(cell_id=cell_id, forecast=forecast) for cell_id in CELL_ORDER]
    return {
        "schema": f"{UNIVERSAL_OBS_SCHEMA_NAMESPACE}-a2-sh1-dry-plan/v1",
        "instruction_id": UNIVERSAL_OBS_INSTRUCTION_ID,
        "amendment_id": UNIVERSAL_OBS_AMENDMENT_ID,
        "source_head": source_head,
        "base_runtime": UNIVERSAL_OBS_BASE_RUNTIME,
        "session_id": SESSION_ID,
        "execution_branch": EXECUTION_BRANCH,
        "execution_parent": EXECUTION_PARENT,
        "execution_head_policy": "runtime-git-head",
        "execution_amendment": EXECUTION_AMENDMENT,
        "approval_token": f"{EXECUTION_AMENDMENT}:{source_head}",
        "execution_checkpoint_bound": True,
        "execution_authorized": True,
        "execution_hold": False,
        "held_atomic_release_required": True,
        "llama_sh1_only": True,
        "qwen_submission_authorized": False,
        "reused_warm_case_seal_root_digest": seal["root_digest"],
        "request_order_sha256": seal["batch_ordered_request_digest_v1"][0],
        "panel_kind": seal["panel_kind"],
        "numerical_lock_sha256_by_cell": numerical_locks,
        "numerical_lock_root_digest_by_cell": numerical_root_digests,
        "r12_frozen_reference_lock_sha256": r12_reference_sha256,
        "r12_frozen_reference": frozen_r12_reference,
        "factorial_shape": [2, 2, 2],
        "live_cells": list(CELL_ORDER),
        "cell_contracts": contracts,
        "fixed_grid": {"k": 8, "h": 0.125, "tau_final": 1.0},
        "d1_d2_d3_observation_only": True,
        "observability_total_decision_influence_count": 0,
        "scientific_promotion_authorized": False,
        "forecast_seconds_per_cell_pair": UNIVERSAL_OBS_FORECAST_SECONDS,
        "allocation_seconds_per_cell_pair": UNIVERSAL_OBS_ALLOCATION_SECONDS,
        "server1_project_gpu_cap": SERVER1_PROJECT_GPU_CAP,
        "active_plus_new_rule": "active_project_gpu + selected_new_gpu <= 4",
        "fast_launch_cell_order": list(CELL_ORDER),
        "all_three_new_gpu": len(jobs),
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
