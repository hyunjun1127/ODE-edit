#!/usr/bin/env python3
"""Model-free deterministic P1R16 Stage-A plan."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.artifacts import ODEBFArtifactGuard
from project.run_scripts.ode_bf.contracts import MODEL_ALIASES, ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.coupled_demand_soft_transport import (
    P1R16_AMENDMENT_ID,
    P1R16_INSTRUCTION_ID,
    P1R16_METHOD_ID,
    p1r16_source_contract,
)
from project.run_scripts.ode_bf.p1_coupled_demand_soft_transport_panel import (
    P1R16_NUMERICAL_LOCK_FILE,
    P1R16_SOURCE_MANIFEST_FILE,
    expected_p1r16_stage_a_context_sha256,
    expected_p1r16_stage_a_result_name,
    load_and_validate_p1r16_numerical_lock,
    load_and_validate_p1r16_source_manifest,
    load_and_validate_p1r16_stage_a_seal,
    p1r16_stage_a_forecast,
    validate_p1r16_source_closure,
    validate_p1r16_stage_a_output_root,
)


SESSION_ID = "019fe489-c968-75f3-9965-7cfbc26c0a99"
GH_SESSION_ID = "019fe491-16f4-7bd3-adf5-4b1eb4a57d1f"
EXECUTION_BRANCH = "codex/odeeditsh1-s05-coupled-demand-soft-transport-p1r16-v1"
EXECUTION_PARENT = "ce0d7c0a9ccb91a7d67209bf93cf6b6ac34cff48"
SERVER1_PROJECT_GPU_CAP = 4
LLAMA_ALIAS = "llama3-8b-inst"
QWEN_ALIAS = "qwen2.5-7b-inst"


def _valid_head(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def build_plan(source_head: str, *, repository_root: Path = REPO_ROOT) -> dict[str, object]:
    if not _valid_head(source_head):
        raise ODEBFContractError("P1R16 dry-plan source head differs")
    root = repository_root.resolve(strict=True)
    locks = root / "project" / "run_scripts" / "ode_bf" / "locks"
    seal, seal_sha = load_and_validate_p1r16_stage_a_seal(locks)
    numerical, numerical_sha = load_and_validate_p1r16_numerical_lock(
        locks / P1R16_NUMERICAL_LOCK_FILE
    )
    source, source_sha = load_and_validate_p1r16_source_manifest(
        root, locks / P1R16_SOURCE_MANIFEST_FILE, source_head=source_head
    )
    closure = validate_p1r16_source_closure(root)
    artifacts: dict[str, object] = {}
    forecasts: dict[str, object] = {}
    namespaces: dict[str, object] = {}
    for alias in MODEL_ALIASES:
        guard = ODEBFArtifactGuard(
            root,
            locks / "p0_artifact_lock.json",
            alias,
            require_held_ode_alloc=False,
        )
        receipt = guard.preflight()
        guard.assert_unchanged()
        artifacts[alias] = {
            "artifact_lock_sha256": receipt.lock_sha256,
            "model_revision": receipt.base_model_revision,
            "projector_sha256": receipt.projector_sha256,
        }
        forecasts[alias] = p1r16_stage_a_forecast(alias).raw_free_payload()
        namespace = validate_p1r16_stage_a_output_root(
            repo_root=root,
            alias=alias,
            output_root=(
                root / "local" / "odebf" / "results"
                / expected_p1r16_stage_a_result_name(alias)
            ),
        )
        namespaces[alias] = namespace
    payload: dict[str, object] = {
        "schema": "ode-edit-s05-p1r16-stage-a-dry-plan/v1",
        "instruction_id": P1R16_INSTRUCTION_ID,
        "amendment_id": P1R16_AMENDMENT_ID,
        "source_head": source_head,
        "execution_parent": EXECUTION_PARENT,
        "execution_branch": EXECUTION_BRANCH,
        "session_id": SESSION_ID,
        "gh_session_id": GH_SESSION_ID,
        "method_id": P1R16_METHOD_ID,
        "source_contract": p1r16_source_contract(),
        "science_config_shared_across_aliases": True,
        "stage_a_seal_root": seal["root_digest"],
        "stage_a_seal_sha256": seal_sha,
        "stage_a_request_order_sha256": seal["batch_ordered_request_digest_v1"][0],
        "stage_a_context_sha256_by_alias": {
            alias: expected_p1r16_stage_a_context_sha256(seal, alias)
            for alias in MODEL_ALIASES
        },
        "stage_a_outcome_selected_mechanistic_only": True,
        "stage_b_material_count": 0,
        "stage_b_access_count": 0,
        "source_manifest_sha256": source_sha,
        "source_manifest_root": source["root_digest"],
        "source_closure_sha256": closure["identity_sha256"],
        "numerical_lock_sha256": numerical_sha,
        "numerical_lock_root": numerical["root_digest"],
        "artifacts": artifacts,
        "forecasts": forecasts,
        "namespaces": namespaces,
        "server1_project_gpu_cap": SERVER1_PROJECT_GPU_CAP,
        "server1_pair_owner": [LLAMA_ALIAS, QWEN_ALIAS],
        "sh2_read_only_package_verifier": True,
        "resources_each": {"gpu": 1, "cpu": 8, "memory_mib": 65_000, "time": "23:59:00"},
        "model_load": False,
        "gpu_use": False,
        "slurm_submit": False,
        "result_root_creation": False,
        "scientific_promotion_authorized": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    if any(os.environ.get(name) != "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise ODEBFContractError("P1R16 dry-plan offline environment differs")
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
