#!/usr/bin/env python3
"""No-model deterministic execution plan for the R12 BG-SOFT missing cell."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1_bg_soft_missing_cell_panel import (
    BG_SOFT_AMENDMENT_ID,
    BG_SOFT_EXECUTION_REPAIR_PARENT,
    BG_SOFT_IMPLEMENTATION_PARENT,
    BG_SOFT_INSTRUCTION_ID,
    BG_SOFT_PARENT_HEAD,
    BG_SOFT_REFERENCE_LOCK_FILE,
    BG_SOFT_RESULT_TOKEN,
    BG_SOFT_SCHEMA_NAMESPACE,
    BG_SOFT_SOURCE_MANIFEST_FILE,
    COMMON_COLD_CASE_SEAL_FILE,
    bg_soft_frozen_reference,
    bg_soft_initial_contract,
    common_cold_schedule,
    expected_bg_soft_result_name,
    forecast_common_cold_panel,
    load_and_validate_bg_soft_reference_lock,
    verify_common_cold_case_seal,
)
from project.run_scripts.ode_bf.sampling import load_p1_sampling_seal


LLAMA_ALIAS = "llama3-8b-inst"
JOB_NAME = "odeedit_s05_r12a1_bgsoft_llama"
SERVER1_PROJECT_GPU_CAP = 4


def _valid_source_head(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def controller_geometry_identity() -> str:
    from project.run_scripts.ode_bf.p1_controller import P1ControllerLock

    return P1ControllerLock().identity()


def build_plan(source_head: str, *, repository_root: Path = REPO_ROOT) -> dict[str, object]:
    if not _valid_source_head(source_head):
        raise ODEBFContractError("BG-Soft dry-plan head differs")
    locks = repository_root / "project/run_scripts/ode_bf/locks"
    seal = verify_common_cold_case_seal(
        json.loads((locks / COMMON_COLD_CASE_SEAL_FILE).read_text(encoding="utf-8"))
    )
    base_schedule = load_p1_sampling_seal(
        locks / "p1r2_p_population_seal.json",
        stream_path=locks / "p1r2_seqb10_stream_seal.json",
    )
    schedule = common_cold_schedule(base_schedule)
    reference, reference_sha256 = load_and_validate_bg_soft_reference_lock(
        locks / BG_SOFT_REFERENCE_LOCK_FILE,
        controller_identity_sha256=controller_geometry_identity(),
        case_root_digest=seal["root_digest"],
        schedule=schedule,
    )
    forecast = forecast_common_cold_panel(
        locks / "p0_artifact_lock.json",
        repository_root / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json",
        LLAMA_ALIAS,
    )
    job = {
        "alias": LLAMA_ALIAS,
        "job_name": JOB_NAME,
        "result_name": expected_bg_soft_result_name(LLAMA_ALIAS),
        "run_token": BG_SOFT_RESULT_TOKEN,
        "gpu": 1,
        "cpu": 8,
        "memory_mib": 65_000,
        "time": "23:59:00",
        "node": "server1",
        "forecast": forecast.raw_free_payload(),
    }
    return {
        "schema": f"{BG_SOFT_SCHEMA_NAMESPACE}-dry-plan/v1",
        "instruction_id": BG_SOFT_INSTRUCTION_ID,
        "amendment_id": BG_SOFT_AMENDMENT_ID,
        "source_head": source_head,
        "expected_parent": BG_SOFT_PARENT_HEAD,
        "execution_repair_parent": BG_SOFT_EXECUTION_REPAIR_PARENT,
        "implementation_parent": BG_SOFT_IMPLEMENTATION_PARENT,
        "source_manifest_path": (
            "project/run_scripts/ode_bf/locks/" + BG_SOFT_SOURCE_MANIFEST_FILE
        ),
        "reference_lock_sha256": reference_sha256,
        "reference_lock_root_digest": reference["root_digest"],
        "reused_warm_case_seal_root_digest": seal["root_digest"],
        "request_order_sha256": seal["batch_ordered_request_digest_v1"][0],
        "panel_kind": seal["panel_kind"],
        "live_arms": reference["live_arms"],
        "scale": reference["scale"],
        "routing": reference["routing"],
        "initial_contract": bg_soft_initial_contract(reference, LLAMA_ALIAS),
        "frozen_r10_reference": bg_soft_frozen_reference(reference, LLAMA_ALIAS),
        "fixed_grid": {
            "k": reference["joint_grid_count"],
            "h": reference["joint_h"],
            "tau_final": reference["joint_tau_final"],
        },
        "per_arm_operation_ceiling": {
            "field_count": reference["joint_grid_count"],
            "candidate_count": reference["joint_grid_count"],
            "overlay_target_backward_batch_count": reference[
                "joint_grid_count"
            ],
            "nohook_target_backward_batch_count": reference[
                "joint_grid_count"
            ],
            "functional_basis_endpoint_count": 48,
            "single_write_audit_endpoint_count": 12,
            "nohook_observation_qp_count": 8,
            "postfreeze_w_only_evaluator_forward_count": 80,
            "postfreeze_z_oracle_evaluator_forward_count": 80,
            "postfreeze_terminal_recapture_forward_count": 8,
            "postfreeze_weight_only_six_context_forward_count": 480,
            "scientific_retry_count": 0,
            "scientific_rejection_count": 0,
        },
        "two_arm_operation_ceiling": {
            "field_count": 16,
            "candidate_count": 16,
            "functional_basis_endpoint_count": 96,
            "signed_progress_backward_batch_count": 32,
            "single_write_audit_endpoint_count": 24,
            "nohook_observation_qp_count": 16,
            "postfreeze_primary_evaluator_forward_count": 320,
            "postfreeze_terminal_recapture_forward_count": 16,
            "postfreeze_weight_only_six_context_forward_count": 960,
        },
        "diagnostics": reference["diagnostics"],
        "forecast_risk": (
            "CONSERVATIVE_82800_SECONDS_LEAVES_3540_SECONDS_UNDER_23H59M"
        ),
        "server1_project_gpu_cap": SERVER1_PROJECT_GPU_CAP,
        "new_job_gpu": 1,
        "qwen_submission_authorized": False,
        "scientific_promotion_authorized": False,
        "model_load": False,
        "gpu_use": False,
        "slurm_submit": False,
        "result_root_creation": False,
        "jobs": [job],
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
