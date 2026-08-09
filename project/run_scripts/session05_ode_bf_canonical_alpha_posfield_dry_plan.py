#!/usr/bin/env python3
"""No-model deterministic plan for the four P1R11 jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.canonical_alpha_posfield import (
    AlphaActuator,
    CANONICAL_ALPHA_POSFIELD_INSTRUCTION_ID,
    canonical_alpha_posfield_source_contract,
)
from project.run_scripts.ode_bf.contracts import MODEL_ALIASES, ODEBFContractError
from project.run_scripts.ode_bf.p1_canonical_alpha_posfield_panel import (
    CANONICAL_ALPHA_NUMERICAL_LOCK_FILE,
    alpha_actuator_slug,
    expected_canonical_alpha_result_name,
    forecast_canonical_alpha_panel,
    load_and_validate_canonical_alpha_lock,
)
from project.run_scripts.ode_bf.p1_common_coldcoord_fixed_e8_panel import (
    COMMON_COLD_CASE_SEAL_FILE,
    common_cold_schedule,
    verify_common_cold_case_seal,
)
from project.run_scripts.ode_bf.sampling import load_p1_sampling_seal


JOB_NAMES = {
    (alias, actuator.value): (
        "odeedit_s05_r11_"
        f"{'llama' if alias.startswith('llama') else 'qwen'}_"
        f"{'current' if actuator is AlphaActuator.CURRENT_SHARED_AE else 'ordered'}"
    )
    for alias in MODEL_ALIASES
    for actuator in AlphaActuator
}


def numerical_controller_identity() -> str:
    from project.run_scripts.ode_bf.p1_controller import P1ControllerLock

    return P1ControllerLock().identity()


def build_plan(
    source_head: str,
    *,
    repository_root: Path = REPO_ROOT,
) -> dict[str, object]:
    if len(source_head) != 40 or any(
        character not in "0123456789abcdef" for character in source_head
    ):
        raise ODEBFContractError("canonical Alpha dry-plan head differs")
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
    lock_sha256: str | None = None
    lock_root: str | None = None
    jobs: list[dict[str, object]] = []
    for alias in MODEL_ALIASES:
        for actuator in AlphaActuator:
            numerical, observed_sha = load_and_validate_canonical_alpha_lock(
                locks / CANONICAL_ALPHA_NUMERICAL_LOCK_FILE,
                controller_identity_sha256=numerical_controller_identity(),
                case_root_digest=seal["root_digest"],
                population_root_digest=population["root_digest"],
                schedule=schedule,
                actuator=actuator,
            )
            if lock_sha256 is None:
                lock_sha256 = observed_sha
                lock_root = numerical["root_digest"]
            elif (lock_sha256, lock_root) != (
                observed_sha,
                numerical["root_digest"],
            ):
                raise ODEBFContractError("canonical Alpha lock aliases differ")
            forecast = forecast_canonical_alpha_panel(
                locks / "p0_artifact_lock.json",
                repository_root / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json",
                alias,
                actuator,
            )
            jobs.append(
                {
                    "alias": alias,
                    "actuator": actuator.value,
                    "actuator_slug": alpha_actuator_slug(actuator),
                    "job_name": JOB_NAMES[(alias, actuator.value)],
                    "result_name": expected_canonical_alpha_result_name(
                        alias, actuator
                    ),
                    "gpu": 1,
                    "cpu": 8,
                    "memory_mib": 65_000,
                    "time": "24:00:00",
                    "server": "server1",
                    "slurm_node": "devbox",
                    "forecast": forecast.raw_free_payload(),
                }
            )
    return {
        "schema": "ode-edit-s05-canonical-alpha-posfield-p1r11-dry-plan/v1",
        "instruction_id": CANONICAL_ALPHA_POSFIELD_INSTRUCTION_ID,
        "source_head": source_head,
        "reused_warm_case_seal_root_digest": seal["root_digest"],
        "request_order_sha256": seal["batch_ordered_request_digest_v1"][0],
        "panel_kind": "REUSED_WARMUP_SEAL_CAUSAL_REGRESSION",
        "numerical_lock_sha256": lock_sha256,
        "numerical_lock_root_digest": lock_root,
        "method": canonical_alpha_posfield_source_contract(),
        "four_independent_jobs": True,
        "same_bootstrap_contract_across_actuators": True,
        "per_job_operation_ceiling": {
            "bootstrap_transition_count": 1,
            "joint_field_count": 8,
            "joint_candidate_count": 8,
            "functional_basis_endpoint_count": 48,
            "target_backward_batch_count": 8,
            "canonical_full_shadow_layer_count_per_field": 5,
            "canonical_shadow_e8_transition_count": 0,
            "canonical_shadow_endpoint_capacity_count": 0,
        },
        "fixed_grid": {"k": 8, "h": 0.125, "tau_target": 1.0},
        "scientific_retry_count": 0,
        "scientific_rejection_count": 0,
        "cold_native_or_direct_z_access_count": 0,
        "native_and_wb_open_after_action_freeze": True,
        "scientific_promotion_authorized": False,
        "server1_project_gpu_cap": 4,
        "new_job_gpu": 4,
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
