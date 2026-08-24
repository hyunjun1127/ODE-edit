#!/usr/bin/env python3
"""No-model/no-CUDA final gate for P1R54 FZ final-z one-shot sequential."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.artifacts import ODEBFArtifactGuard, sha256_file
from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.p1r52_target_timescale_deployment import (
    build_target_timescale_deployment,
)
from project.run_scripts.ode_bf.p1r54_fz_c3_sequential import (
    STREAM_ORDER,
    STREAM_ROOT,
)
from project.run_scripts.ode_bf.p1r54_fz_finalz_oneshot_sequential import (
    CONTROL_MANIFEST_SHA256,
    CONTROL_TERMINAL_SHA256,
    INSTRUCTION_ID,
    RESULT_NAME,
    ROLE,
    TECH_R0_FAILURE_SHA256,
    control_source_equivalence,
)
from project.run_scripts.session05_ode_bf_p1r52_target_timescale_b100_preflight import (
    EXTRACT_ROOT,
    PRIOR_HF_GATE,
    _read_json,
    _rooted,
    _verify_stream,
    _write_or_verify_once,
)
from project.run_scripts.session05_ode_bf_p1r54_fz_finalz_oneshot_sequential_dry_plan import (
    build_plan,
)


PROJECT_GPU_CAP = 4
JOB_GPU_COUNT = 1
NUMERICAL_LOCK = Path(
    "project/run_scripts/ode_bf/locks/"
    "numerical_lock_s05_p1r54_fz_finalz_oneshot_sequential_10xb100_tech_r1_v1.json"
)
TECH_R0_ROOT = Path(
    "/data/janghj/ODE-edit/local/worktrees/"
    "p1r54-fz-finalz-oneshot-seq-v1/local/odebf/results/"
    "s05-p1r54-fz-final-z-oneshot-sequential-10xb100-v1"
)
CONTROL_ROOT = Path(
    "/data/janghj/ODE-edit/local/worktrees/"
    "p1r54-fz-c3-sequential-10xb100-v1/local/odebf/results/"
    "s05-p1r54-fz-c3-sequential-10xb100-v1"
)
SOURCE_FILES = (
    "agents/server4/alphaedit-runtime-path-seal.json",
    "agents/server4/p4-hf-consumed-closure-seal.json",
    "project/run_scripts/alphaedit_runtime_path_seal.py",
    "project/run_scripts/ode_alloc/p0_artifacts.py",
    "project/run_scripts/ode_bf/artifacts.py",
    "project/run_scripts/ode_bf/contracts.py",
    "project/run_scripts/ode_bf/p1_runtime.py",
    "project/run_scripts/ode_bf/p1r52_c_writer_kstep.py",
    "project/run_scripts/ode_bf/p1r52_c_writer_kstep_cache.py",
    "project/run_scripts/ode_bf/p1r52_c_writer_kstep_cache_sequential.py",
    "project/run_scripts/ode_bf/p1r52_joint_pc_fp32_runtime.py",
    "project/run_scripts/ode_bf/p1r52_sequential_runtime.py",
    "project/run_scripts/ode_bf/p1_scalable_batched_experiment.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale_deployment.py",
    "project/run_scripts/ode_bf/p1r54_energyfree_localz.py",
    "project/run_scripts/ode_bf/p1r54_fz_c3_sequential.py",
    "project/run_scripts/ode_bf/p1r54_fz_finalz_oneshot_analysis.py",
    "project/run_scripts/ode_bf/p1r54_fz_finalz_oneshot_sequential.py",
    "project/run_scripts/ode_bf/writer_cadence.py",
    "project/run_scripts/ode_bf/scalable_batched_runtime.py",
    "project/run_scripts/ode_bf/tests/test_p1r54_fz_finalz_oneshot_sequential.py",
    "project/run_scripts/session05_ode_bf_p1r54_fz_finalz_oneshot_sequential.py",
    "project/run_scripts/session05_ode_bf_p1r54_fz_finalz_oneshot_sequential_analyze.py",
    "project/run_scripts/session05_ode_bf_p1r54_fz_finalz_oneshot_sequential_dry_plan.py",
    "project/run_scripts/session05_ode_bf_p1r54_fz_finalz_oneshot_sequential_preflight.py",
    "project/run_scripts/session05_ode_bf_p1r54_fz_finalz_oneshot_sequential_server4.sbatch",
    str(NUMERICAL_LOCK),
    "project/run_scripts/ode_bf/locks/"
    "numerical_lock_s05_p1r54_fz_finalz_oneshot_sequential_10xb100_v1.json",
    "project/run_scripts/ode_bf/locks/p1r52_sequential_b100x10_stream_seal.json",
)


def source_manifest(head: str, tree: str) -> Mapping[str, Any]:
    entries = []
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ODEBFContractError(f"final-z source member differs: {relative}")
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-fz-final-z-source-manifest/v1",
        "source_head": head,
        "source_tree": tree,
        "entries": entries,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def verify_control() -> Mapping[str, Any]:
    terminal_path = CONTROL_ROOT / "terminal.json"
    manifest_path = CONTROL_ROOT / "manifest.json"
    terminal = _read_json(terminal_path)
    if (
        sha256_file(terminal_path) != CONTROL_TERMINAL_SHA256
        or sha256_file(manifest_path) != CONTROL_MANIFEST_SHA256
        or terminal.get("status") != "TERMINAL_VALID"
        or terminal.get("completed_batch_count") != 10
        or terminal.get("valid_request_count") != 1000
        or terminal.get("K_writer_call_count") != 80
        or terminal.get("writer_layer_apply_count") != 400
        or terminal.get("target_field_evaluation_count") != 80
        or terminal.get("stream_root") != STREAM_ROOT
        or terminal.get("stream_order") != STREAM_ORDER
        or terminal.get("W0_restored") is not True
        or terminal.get("dtype_contract", {}).get("status") != "FULL_FP32_PASS"
    ):
        raise ODEBFContractError("P1R54 FZ K-step control differs")
    return {
        "root": str(CONTROL_ROOT),
        "terminal_sha256": CONTROL_TERMINAL_SHA256,
        "manifest_sha256": CONTROL_MANIFEST_SHA256,
        "terminal_identity": terminal.get("identity_sha256"),
        "source_head": terminal.get("source_head"),
        "execution_influence_count": 0,
    }


def verify_excluded_technical_attempt() -> Mapping[str, Any]:
    path = TECH_R0_ROOT / "failure.json"
    failure = _read_json(path)
    if (
        sha256_file(path) != TECH_R0_FAILURE_SHA256
        or failure.get("status") != "FAIL_CLOSED_NO_RETRY"
        or failure.get("exception_class") != "ODEBFStateError"
        or failure.get("last_completed_stage") != "post_model_context_teacher"
    ):
        raise ODEBFContractError("final-z excluded TECH-R0 failure differs")
    return {
        "root": str(TECH_R0_ROOT),
        "failure_sha256": TECH_R0_FAILURE_SHA256,
        "classification": "PURE_TECHNICAL_PRE_FIELD",
        "scientific_denominator_inclusion_count": 0,
        "immutable": True,
    }


def build_receipt(*, source_head: str, final_receipt: Path, session_id: str) -> Mapping[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True
    ).strip()
    role = subprocess.check_output(
        ["git", "config", "--get", "agent.role"], cwd=REPO_ROOT, text=True
    ).strip()
    host = subprocess.check_output(
        ["git", "config", "--get", "agent.hostname"], cwd=REPO_ROOT, text=True
    ).strip()
    if (
        socket.gethostname() != "server4"
        or role != "server-head"
        or host != "server4"
        or head != source_head
        or subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout
        or os.environ.get("PROJECT_GPU_CAP", str(PROJECT_GPU_CAP))
        != str(PROJECT_GPU_CAP)
    ):
        raise ODEBFContractError("final-z host/source/cap differs")

    stream = _verify_stream()
    lock, lock_sha = _rooted(REPO_ROOT / NUMERICAL_LOCK)
    if (
        lock.get("instruction_id") != INSTRUCTION_ID
        or lock.get("model_alias") != "llama3-8b-inst"
        or lock.get("dtype") != "FULL_FP32"
        or lock.get("stream_root") != STREAM_ROOT
        or lock.get("order_root") != STREAM_ORDER
        or lock.get("sequential_batches") != [f"B{i}" for i in range(1, 11)]
        or lock.get("target_clock", {}).get("field_evaluations_total") != 80
        or lock.get("writer", {}).get("cadence") != "FINAL_Z_ONESHOT"
        or lock.get("writer", {}).get("calls_total") != 10
        or lock.get("writer", {}).get("layer_applies_total") != 50
        or lock.get("writer", {}).get("intermediate_target_writer_influence_count") != 0
        or lock.get("technical_attempt") != "TECH-R1"
        or lock.get("parent_numerical_lock_root")
        != "5b60ffc0b9267cc3d3a25263c03b1aebfc50bc4eb2125e7b5e1e44a0d5969018"
        or lock.get("repair", {}).get("accepted_physical_advance_policy")
        != "FINAL_Z_ONESHOT"
        or lock.get("repair", {}).get("stationary_physical_state_count_per_batch")
        != 7
        or lock.get("repair", {}).get("physical_advance_count_per_batch") != 1
        or lock.get("repair", {}).get("scientific_change_count") != 0
        or lock.get("scientific_promotion") is not False
    ):
        raise ODEBFContractError("final-z numerical lock differs")
    control = verify_control()
    excluded_technical_attempt = verify_excluded_technical_attempt()
    deployment = build_target_timescale_deployment(
        repo_root=REPO_ROOT,
        stream_extract_root=EXTRACT_ROOT,
        prior_final_pre_gpu_path=PRIOR_HF_GATE,
        dataset_identity=str(stream["dataset_identity"]),
    )
    guard = ODEBFArtifactGuard(
        REPO_ROOT,
        REPO_ROOT / "project/run_scripts/ode_bf/locks/p0_artifact_lock.json",
        "llama3-8b-inst",
        require_held_ode_alloc=False,
        runtime_path_seal=deployment.runtime_path_seal,
        evaluator_source_paths=deployment.evaluator_source_paths,
        base_guard_override=deployment.base_guard,
    )
    artifact = guard.preflight()
    guard.assert_unchanged()
    final: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-fz-final-z-oneshot-final-pre-gpu/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "FINAL_PRE_GPU_PASS_PENDING_SUBMISSION_AUTHORIZED",
        "model_load_authorized_when_capacity_available": True,
        "resource_hold": False,
        "pending_submission_authorized": True,
        "runtime_release_condition": "SERVER4_JANGHJ_RUNNING_GPU_PLUS_NEW_GPU_LE_4",
        "hostname": "server4",
        "agent_role": role,
        "session_id": session_id,
        "source_head": head,
        "source_tree": tree,
        "source_manifest": source_manifest(head, tree),
        "role": ROLE,
        "result_name": RESULT_NAME,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "job_gpu_count": JOB_GPU_COUNT,
        "stream_binding": stream,
        "numerical_lock_path": str(REPO_ROOT / NUMERICAL_LOCK),
        "numerical_lock_sha256": lock_sha,
        "numerical_lock_root": lock["root_digest"],
        "deployment_identity": deployment.identity_sha256,
        "artifact_receipt": asdict(artifact),
        "control": control,
        "excluded_technical_attempt": excluded_technical_attempt,
        "technical_attempt": "TECH-R1",
        "scientific_change_count": 0,
        "control_source_equivalence": control_source_equivalence(),
        "dry_plan": build_plan(),
        "full_fp32": True,
        "model_gpu_slurm_action_count": 0,
        "scientific_promotion": False,
    }
    final["identity_sha256"] = canonical_hash(final)
    _write_or_verify_once(final_receipt, final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--final-receipt", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_receipt(
                source_head=args.source_head,
                final_receipt=args.final_receipt,
                session_id=args.session_id,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
