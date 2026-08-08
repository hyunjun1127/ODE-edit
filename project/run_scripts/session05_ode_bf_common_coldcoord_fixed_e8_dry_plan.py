#!/usr/bin/env python3
"""No-model deterministic R10 common-cold-coordinate execution plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.common_cold_coordinate import (
    COMMON_COLD_H,
    COMMON_COLD_INSTRUCTION_ID,
    common_cold_source_contract,
)
from project.run_scripts.ode_bf.contracts import MODEL_ALIASES, ODEBFContractError
from project.run_scripts.ode_bf.p1_common_coldcoord_fixed_e8_panel import (
    COMMON_COLD_CASE_SEAL_FILE,
    COMMON_COLD_NUMERICAL_LOCK_FILE,
    COMMON_COLD_PANEL_LABELS,
    common_cold_schedule,
    expected_common_cold_result_name,
    forecast_common_cold_panel,
    load_and_validate_common_cold_lock,
    verify_common_cold_case_seal,
)
from project.run_scripts.ode_bf.sampling import load_p1_sampling_seal


JOB_NAMES = {
    "llama3-8b-inst": "odeedit_s05_r10_llama",
    "qwen2.5-7b-inst": "odeedit_s05_r10_qwen",
}


def build_plan(source_head: str, *, repository_root: Path = REPO_ROOT) -> dict[str, object]:
    if len(source_head) != 40 or any(c not in "0123456789abcdef" for c in source_head):
        raise ODEBFContractError("common cold dry-plan head differs")
    locks = repository_root / "project/run_scripts/ode_bf/locks"
    seal = verify_common_cold_case_seal(
        json.loads((locks / COMMON_COLD_CASE_SEAL_FILE).read_text(encoding="utf-8"))
    )
    base_schedule = load_p1_sampling_seal(
        locks / "p1r2_p_population_seal.json",
        stream_path=locks / "p1r2_seqb10_stream_seal.json",
    )
    schedule = common_cold_schedule(base_schedule)
    population = json.loads(
        (locks / "p1r2_p_population_seal.json").read_text(encoding="utf-8")
    )
    numerical, numerical_sha = load_and_validate_common_cold_lock(
        locks / COMMON_COLD_NUMERICAL_LOCK_FILE,
        controller_identity_sha256=numerical_controller_identity(),
        case_root_digest=seal["root_digest"],
        population_root_digest=population["root_digest"],
        schedule=schedule,
    )
    jobs = []
    for alias in MODEL_ALIASES:
        forecast = forecast_common_cold_panel(
            locks / "p0_artifact_lock.json",
            repository_root / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json",
            alias,
        )
        jobs.append(
            {
                "alias": alias,
                "job_name": JOB_NAMES[alias],
                "result_name": expected_common_cold_result_name(alias),
                "gpu": 1,
                "cpu": 8,
                "memory_mib": 65_000,
                "time": "24:00:00",
                "node": "server1",
                "forecast": forecast.raw_free_payload(),
            }
        )
    return {
        "schema": "ode-edit-s05-common-coldcoord-fixed-e8-p1r10-dry-plan/v1",
        "instruction_id": COMMON_COLD_INSTRUCTION_ID,
        "source_head": source_head,
        "reused_warm_case_seal_root_digest": seal["root_digest"],
        "request_order_sha256": seal["batch_ordered_request_digest_v1"][0],
        "panel_kind": seal["panel_kind"],
        "warm_source_stream_root_digest": seal["warm_source"][
            "source_stream_root_digest"
        ],
        "warm_source_terminal_sha256": {
            alias: value["terminal_sha256"]
            for alias, value in seal["warm_source"]["source_roots"].items()
        },
        "outcome_selection_bias": seal["warm_source"]["outcome_selection_bias"],
        "unseen_or_fresh_sample_claim_authorized": seal[
            "unseen_or_fresh_sample_claim_authorized"
        ],
        "numerical_lock_sha256": numerical_sha,
        "numerical_lock_root_digest": numerical["root_digest"],
        "panel_labels": list(COMMON_COLD_PANEL_LABELS),
        "method": common_cold_source_contract(),
        "bootstrap": {
            "scale_count": 2,
            "transition_count_per_scale": 1,
            "h": COMMON_COLD_H,
            "writer_count": 0,
            "retry_count": 0,
        },
        "per_arm_operation_ceiling": {
            "field_count": 8,
            "candidate_count": 8,
            "functional_basis_endpoint_count": 48,
            "target_backward_batch_count": 8,
            "logical_qp_certificate_count": 32,
            "optimizer_backend_invocation_count": 64,
        },
        "three_arm_operation_ceiling": {
            "field_count": 24,
            "candidate_count": 24,
            "functional_basis_endpoint_count": 144,
            "target_backward_batch_count": 24,
            "logical_qp_certificate_count": 96,
            "optimizer_backend_invocation_count": 192,
        },
        "fixed_grid": {"k": 8, "h": COMMON_COLD_H, "tau_final": 1.0},
        "scientific_retry_count": 0,
        "scientific_rejection_count": 0,
        "cold_native_or_direct_z_access_count": 0,
        "heldout_open_after_all_actions_frozen": True,
        "scientific_promotion_authorized": False,
        "server1_project_gpu_cap": 3,
        "new_pair_gpu": 2,
        "model_load": False,
        "gpu_use": False,
        "slurm_submit": False,
        "result_root_creation": False,
        "jobs": jobs,
    }


def numerical_controller_identity() -> str:
    from project.run_scripts.ode_bf.p1_controller import P1ControllerLock

    return P1ControllerLock().identity()


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
