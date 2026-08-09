#!/usr/bin/env python3
"""Model-free, deterministic P1R14 execution plan."""

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
from project.run_scripts.ode_bf.contracts import MODEL_ALIASES, ODEBFContractError
from project.run_scripts.ode_bf.integrated_physical_writer import (
    INTEGRATED_PHYSICAL_WRITER_AMENDMENT_IDS,
    INTEGRATED_PHYSICAL_WRITER_INSTRUCTION_ID,
)
from project.run_scripts.ode_bf.p1_integrated_physical_writer_panel import (
    P1R14_NUMERICAL_LOCK_FILE,
    P1R14_RESULT_TOKEN,
    P1R14_SOURCE_MANIFEST_FILE,
    load_and_validate_p1r14_numerical_lock,
    load_and_validate_p1r14_seals,
    load_and_validate_p1r14_source_manifest,
    p1r14_common_science_config,
    p1r14_forecast,
    validate_p1r14_source_closure,
)


SESSION_ID = "019fe489-c968-75f3-9965-7cfbc26c0a99"
GH_SESSION_ID = "019fe491-16f4-7bd3-adf5-4b1eb4a57d1f"
EXECUTION_BRANCH = "codex/odeeditsh1-s05-integrated-physical-writer-p1r14-v1"
EXECUTION_PARENT = "186ded6c2f75cb77c8d777b776a423d8f422dd79"
SERVER1_PROJECT_GPU_CAP = 4
JOB_NAME = "odeedit_s05_p1r14_sh1_integrated_writer_llama_tech_r3"
LLAMA_ALIAS = "llama3-8b-inst"
RESULT_NAME = "s05-integrated-physical-writer-p1r14-llama3-8b-inst-tech-r3-v1"


def _valid_head(value: str) -> bool:
    return len(value) == 40 and all(item in "0123456789abcdef" for item in value)


def build_plan(
    source_head: str,
    *,
    repository_root: Path = REPO_ROOT,
) -> dict[str, object]:
    if not _valid_head(source_head):
        raise ODEBFContractError("integrated dry-plan source head differs")
    root = repository_root.resolve(strict=True)
    locks = root / "project" / "run_scripts" / "ode_bf" / "locks"
    exclusion, seal = load_and_validate_p1r14_seals(locks)
    closure = validate_p1r14_source_closure(root)
    source, source_sha = load_and_validate_p1r14_source_manifest(
        root,
        locks / P1R14_SOURCE_MANIFEST_FILE,
        source_head=source_head,
    )
    artifacts: dict[str, object] = {}
    numerical: dict[str, object] = {}
    for alias in MODEL_ALIASES:
        guard = ODEBFArtifactGuard(
            root,
            locks / "p0_artifact_lock.json",
            alias,
            require_held_ode_alloc=False,
        )
        receipt = guard.preflight()
        guard.assert_unchanged()
        lock, lock_sha = load_and_validate_p1r14_numerical_lock(
            locks / P1R14_NUMERICAL_LOCK_FILE,
            exclusion_root=exclusion["root_digest"],
            fresh_seal_root=seal["root_digest"],
            artifact_lock_root=receipt.lock_sha256,
        )
        artifacts[alias] = {
            "artifact_lock_sha256": receipt.lock_sha256,
            "model_revision": receipt.base_model_revision,
            "projector_sha256": receipt.projector_sha256,
        }
        numerical[alias] = {
            "file_sha256": lock_sha,
            "root_digest": lock["root_digest"],
        }
    forecast = p1r14_forecast()
    if not forecast.fits:
        raise ODEBFContractError("integrated dry-plan forecast differs")
    science = p1r14_common_science_config()
    return {
        "schema": "ode-edit-s05-integrated-physical-writer-p1r14-dry-plan/v1",
        "instruction_id": INTEGRATED_PHYSICAL_WRITER_INSTRUCTION_ID,
        "amendment_ids": list(INTEGRATED_PHYSICAL_WRITER_AMENDMENT_IDS),
        "source_head": source_head,
        "execution_parent": EXECUTION_PARENT,
        "execution_branch": EXECUTION_BRANCH,
        "session_id": SESSION_ID,
        "gh_session_id": GH_SESSION_ID,
        "science_config": science,
        "science_config_shared_across_aliases": True,
        "fresh_seal_root": seal["root_digest"],
        "historical_exclusion_root": exclusion["root_digest"],
        "request_order_sha256": seal["batch_ordered_request_digest_v1"],
        "functional_p_anchor_order_sha256": seal[
            "functional_p_anchor_order_digest"
        ],
        "source_manifest_sha256": source_sha,
        "source_manifest_root": source["root_digest"],
        "source_closure_sha256": closure["identity_sha256"],
        "artifacts": artifacts,
        "numerical_locks": numerical,
        "forecast": forecast.raw_free_payload(),
        "server1_project_gpu_cap": SERVER1_PROJECT_GPU_CAP,
        "server1_submission": {
            "alias": LLAMA_ALIAS,
            "job_name": JOB_NAME,
            "result_name": RESULT_NAME,
            "run_token": P1R14_RESULT_TOKEN,
            "gpu": 1,
            "cpu": 8,
            "memory_mib": 65_000,
            "time": "23:59:00",
            "held_then_atomic_release": True,
        },
        "server2_handoff": {
            "alias": "qwen2.5-7b-inst",
            "package_acceptance_required": True,
            "same_source_head": source_head,
            "same_science_config_sha256": science["identity_sha256"],
        },
        "model_load": False,
        "gpu_use": False,
        "slurm_submit": False,
        "result_root_creation": False,
        "scientific_promotion_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise ODEBFContractError("integrated dry-plan offline environment differs")
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
