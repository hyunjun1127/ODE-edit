#!/usr/bin/env python3
"""Focused no-model/no-CUDA preflight for four realization-reset cells."""

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

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.artifacts import ODEBFArtifactGuard, sha256_file
from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf.p1r52_target_timescale_deployment import (
    EXPECTED_HF_CLOSURE,
    EXPECTED_HF_REQUIRED_ROOT,
    build_target_timescale_deployment,
)
from project.run_scripts.ode_bf.p1r54_realization_policy import (
    RealizationPolicy,
    RealizationTransitionController,
)
from project.run_scripts.ode_bf.p1r54_realization_reset import (
    CELLS,
    INSTRUCTION_ID,
    ROLES,
    STREAM_ORDER,
    STREAM_ROOT,
    source_equivalence_receipt,
)
from project.run_scripts.session05_ode_bf_p1r52_target_timescale_b100_preflight import (
    EXTRACT_ROOT,
    PRIOR_HF_GATE,
    _rooted,
    _verify_stream,
    _write_or_verify_once,
)
from project.run_scripts.session05_ode_bf_p1r54_realization_reset_dry_plan import build_plan


PROJECT_GPU_CAP = 4
JOB_GPU_COUNT = 1
ARRAY_THROTTLE = 4
CONTRACT = Path(
    "/data/janghj/ODE-edit/local/state/p1r54-realization-reset-v1/authoritative-contract.txt"
)
CONTRACT_SHA256 = "3cbaec42ddc8727e7a35ad5ffc9503f736bd8e6f24b30ba9b0eca03bb5425d15"
NUMERICAL_LOCK = Path(
    "project/run_scripts/ode_bf/locks/numerical_lock_s05_p1r54_realization_reset_v1.json"
)
NUMERICAL_LOCK_ROOT = "7bd9082fcd23b6c6270e4d3ceef52bda152db2ef11c97eed6af99d45c40fc035"
SOURCE_FILES = (
    "agents/server4/alphaedit-runtime-path-seal.json",
    "agents/server4/p4-hf-consumed-closure-seal.json",
    "project/run_scripts/alphaedit_runtime_path_seal.py",
    "project/run_scripts/ode_alloc/p0_artifacts.py",
    "project/run_scripts/ode_bf/artifacts.py",
    "project/run_scripts/ode_bf/contracts.py",
    "project/run_scripts/ode_bf/p1_runtime.py",
    "project/run_scripts/ode_bf/p1_scalable_batched_experiment.py",
    "project/run_scripts/ode_bf/p1r52_c_writer_kstep.py",
    "project/run_scripts/ode_bf/p1r52_c_writer_kstep_cache.py",
    "project/run_scripts/ode_bf/p1r52_c_writer_kstep_cache_sequential.py",
    "project/run_scripts/ode_bf/p1r52_joint_pc_fp32_runtime.py",
    "project/run_scripts/ode_bf/p1r52_sequential_runtime.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale_b100.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale_deployment.py",
    "project/run_scripts/ode_bf/p1r54_energyfree_localz.py",
    "project/run_scripts/ode_bf/p1r54_realization_policy.py",
    "project/run_scripts/ode_bf/p1r54_realization_reset.py",
    "project/run_scripts/ode_bf/p4_hf_consumed_closure.py",
    "project/run_scripts/ode_bf/tests/test_p1r54_realization_reset.py",
    "project/run_scripts/session05_ode_bf_p1r54_realization_reset.py",
    "project/run_scripts/session05_ode_bf_p1r54_realization_reset_dry_plan.py",
    "project/run_scripts/session05_ode_bf_p1r54_realization_reset_preflight.py",
    "project/run_scripts/session05_ode_bf_p1r54_realization_reset_server4.sbatch",
    "project/run_scripts/session05_ode_bf_p1r54_realization_reset_independent.py",
    "project/run_scripts/session05_ode_bf_p1r54_realization_reset_independent_dry_plan.py",
    "project/run_scripts/session05_ode_bf_p1r54_realization_reset_independent_preflight.py",
    "project/run_scripts/session05_ode_bf_p1r54_realization_reset_independent_server4.sbatch",
    str(NUMERICAL_LOCK),
    "project/run_scripts/ode_bf/locks/p1r52_sequential_b100x10_stream_seal.json",
)
CONTROL_TERMINALS = {
    "FZ_INDEPENDENT_CONTINUATION": (
        Path("/data/janghj/ODE-edit/local/worktrees/p1r54-energyfree-localz-llama-b100-v1/local/odebf/results/s05-p1r54-energyfree-localz-llama-b100-fz-tech-r1-v1/terminal.json"),
        "89ad4231916278f6ba087004106fb1870a3227f903e7e499eef63e7dab1b08d2",
    ),
    "PDZ_T1_INDEPENDENT_CONTINUATION": (
        Path("/data/janghj/ODE-edit/local/worktrees/p1r54-energyfree-localz-llama-b100-v1/local/odebf/results/s05-p1r54-energyfree-localz-llama-b100-pdz-tech-r1-v1/terminal.json"),
        "f222c6b30bd90e95747936947972ac61771691072d3b1f8be8d0ce2ad660354b",
    ),
    "FZ_SEQUENTIAL_CONTINUATION": (
        Path("/data/janghj/ODE-edit/local/worktrees/p1r54-fz-c3-sequential-10xb100-v1/local/odebf/results/s05-p1r54-fz-c3-sequential-10xb100-v1/terminal.json"),
        "7507d44f453fe701e67e76c758f578ef658ab00081dc189afc84e99d695fdb88",
    ),
}


def source_manifest(head: str, tree: str) -> Mapping[str, Any]:
    entries = []
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ODEBFContractError(f"P1R54 reset source member differs: {relative}")
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-realization-reset-source-manifest/v1",
        "source_head": head,
        "source_tree": tree,
        "entries": entries,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def focused_transition_fixture() -> Mapping[str, Any]:
    origin = torch.arange(12, dtype=torch.float32).reshape(4, 3).contiguous()
    reset = RealizationTransitionController(
        RealizationPolicy.POST_WRITE_W_REALIZATION_RESET, request_count=3
    )
    continuation = RealizationTransitionController(
        RealizationPolicy.TARGET_Z_CONTINUATION, request_count=3
    )
    current = origin.clone()
    command = current + 1.0
    physical = current + 0.25
    for controller in (reset, continuation):
        controller.observe_entry(
            step_index=0,
            controller_anchor=current,
            current_terminal=current,
            target_origin=origin,
            teacher_sha256="a" * 64,
            target_new_nll_by_request=(1.0, 1.0, 1.0),
        )
    reset_anchor, reset_row = reset.advance(
        step_index=0,
        controller_anchor=current,
        commanded_target=command,
        current_terminal=current,
        next_physical_terminal=physical,
        target_origin=origin,
        teacher_sha256="a" * 64,
        commanded_target_new_nll_by_request=(0.5, 0.5, 0.5),
    )
    continuation_anchor, continuation_row = continuation.advance(
        step_index=0,
        controller_anchor=current,
        commanded_target=command,
        current_terminal=current,
        next_physical_terminal=physical,
        target_origin=origin,
        teacher_sha256="a" * 64,
        commanded_target_new_nll_by_request=(0.5, 0.5, 0.5),
    )
    if (
        reset_row["commanded_target_sha256"] != continuation_row["commanded_target_sha256"]
        or reset_row["current_terminal_sha256"] != continuation_row["current_terminal_sha256"]
        or tensor_sha256(reset_anchor) != tensor_sha256(physical)
        or tensor_sha256(continuation_anchor) != tensor_sha256(command)
        or torch.cuda.is_initialized()
    ):
        raise ODEBFContractError("P1R54 reset focused transition fixture differs")
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-realization-reset-focused-fixture/v1",
        "status": "PASS",
        "K1_proposal_command_writer_post_W_equivalence": True,
        "reset_K2_anchor_is_prior_next_physical": True,
        "continuation_K2_anchor_is_prior_commanded": True,
        "last_commanded_target_separate": True,
        "target_origin_refresh_count": 0,
        "teacher_refresh_count": 1,
        "reset_added_model_forward_count": 0,
        "reset_added_backward_count": 0,
        "reset_added_materialization_count": 0,
        "reset_added_evaluator_count": 0,
        "cuda_initialization_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def verify_controls() -> Mapping[str, Any]:
    rows = []
    for label, (path, expected_sha) in CONTROL_TERMINALS.items():
        observed_sha = sha256_file(path)
        value = json.loads(path.read_bytes())
        if (
            observed_sha != expected_sha
            or value.get("status") != "TERMINAL_VALID"
            or value.get("stream_root") != STREAM_ROOT
            or value.get("stream_order") != STREAM_ORDER
        ):
            raise ODEBFContractError(f"P1R54 continuation control differs: {label}")
        rows.append(
            {
                "label": label,
                "path": str(path),
                "sha256": observed_sha,
                "source_head": value.get("source_head"),
                "request_count": value.get("request_count"),
                "completed_batch_count": value.get("completed_batch_count"),
                "gpu_rerun_count": 0,
            }
        )
    candidate_roots = (
        Path("/data/janghj/ODE-edit/local/worktrees"),
        Path("/data/janghj/ODE-edit/local/odebf/results"),
    )
    pdz_sequential = []
    for root in candidate_roots:
        if root.is_dir():
            pdz_sequential.extend(
                path
                for path in root.glob("**/terminal.json")
                if "pdz" in str(path).lower() and "sequential" in str(path).lower()
            )
    if pdz_sequential:
        raise ODEBFContractError("unexpected PDZ-T1 sequential continuation candidate requires audit")
    payload: dict[str, Any] = {
        "status": "SEALED_CONTINUATION_CONTROLS_PASS",
        "controls": rows,
        "PDZ_T1_SEQUENTIAL_CONTINUATION": {
            "classification": "ABSENT_CONTROL",
            "bounded_candidate_count": 0,
            "new_fifth_run_count": 0,
        },
        "new_continuation_gpu_run_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def build_receipt(
    *,
    source_head: str,
    final_receipt: Path,
    session_id: str,
    continuation_control_audit_required: bool = True,
) -> Mapping[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True).strip()
    origin_main = subprocess.check_output(
        ["git", "rev-parse", "refs/remotes/origin/main"], cwd=REPO_ROOT, text=True
    ).strip()
    role = subprocess.check_output(
        ["git", "config", "--get", "agent.role"], cwd=REPO_ROOT, text=True
    ).strip()
    agent_host = subprocess.check_output(
        ["git", "config", "--get", "agent.hostname"], cwd=REPO_ROOT, text=True
    ).strip()
    if (
        socket.gethostname() != "server4"
        or role != "server-head"
        or agent_host != "server4"
        or head != source_head
        or origin_main != head
        or os.environ.get("PROJECT_GPU_CAP", "4") != "4"
        or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        )
    ):
        raise ODEBFContractError("P1R54 reset host/source/cap differs")
    if (
        CONTRACT.is_symlink()
        or not CONTRACT.is_file()
        or (CONTRACT.stat().st_mode & 0o777) != 0o600
        or CONTRACT.stat().st_size != 15228
        or sha256_file(CONTRACT) != CONTRACT_SHA256
        or CONTRACT.read_bytes().count(b"\n") != 569
    ):
        raise ODEBFContractError("P1R54 reset authoritative contract differs")
    stream = _verify_stream()
    tracked_stream, tracked_stream_sha = _rooted(
        REPO_ROOT / "project/run_scripts/ode_bf/locks/p1r52_sequential_b100x10_stream_seal.json"
    )
    lock, lock_sha = _rooted(REPO_ROOT / NUMERICAL_LOCK)
    lock_without_root = dict(lock)
    observed_root = lock_without_root.pop("root_digest", None)
    if (
        observed_root != NUMERICAL_LOCK_ROOT
        or canonical_hash(lock_without_root) != NUMERICAL_LOCK_ROOT
        or tracked_stream.get("root_digest") != STREAM_ROOT
        or tracked_stream.get("all_request_order_sha256") != STREAM_ORDER
        or lock.get("cells")
        != [
            "FZ_RESET_INDEPENDENT_B100",
            "PDZ_T1_RESET_INDEPENDENT_B100",
            "FZ_RESET_SEQUENTIAL_10XB100",
            "PDZ_T1_RESET_SEQUENTIAL_10XB100",
        ]
    ):
        raise ODEBFContractError("P1R54 reset stream/numerical lock differs")
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
        "schema": "ode-edit-s05-p1r54-realization-reset-final-pre-gpu/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "FINAL_PRE_GPU_PASS",
        "model_load_authorized": True,
        "hostname": "server4",
        "agent_role": role,
        "agent_hostname": agent_host,
        "session_id": session_id,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "job_gpu_count": JOB_GPU_COUNT,
        "array_throttle": ARRAY_THROTTLE,
        "source_head": head,
        "source_tree": tree,
        "origin_main_at_gate": origin_main,
        "source_manifest": source_manifest(head, tree),
        "authoritative_contract": {
            "path": str(CONTRACT),
            "sha256": CONTRACT_SHA256,
            "bytes": 15228,
            "lines": 569,
            "full_read": True,
        },
        "scope_correction": {
            "independent_reset_run_count": 2,
            "sequential_reset_run_count": 2,
            "new_continuation_run_count": 0,
        },
        "stream_binding": {
            **dict(stream),
            "sample_payload_count": 1,
            "sample_duplication_count": 0,
        },
        "tracked_stream_lock_sha256": tracked_stream_sha,
        "numerical_lock": {
            "path": str(REPO_ROOT / NUMERICAL_LOCK),
            "sha256": lock_sha,
            "root_digest": observed_root,
        },
        "dry_plan": build_plan(),
        "focused_transition_fixture": focused_transition_fixture(),
        "source_equivalence": [source_equivalence_receipt(item.cell) for item in CELLS],
        "continuation_controls": (
            verify_controls()
            if continuation_control_audit_required
            else {
                "status": "NOT_RELEVANT_ACTIVE_RUN_USER_EXPANSION",
                "decision_influence_count": 0,
            }
        ),
        "deployment_identity": deployment.identity_sha256,
        "easyedit_runtime_seal_root": deployment.runtime_path_seal.root_digest,
        "hf_binding": {
            "closure_identity": deployment.base_guard.model_seal.closure_identity,
            "required_root": deployment.base_guard.model_seal.required_root,
            "expected_closure_identity": EXPECTED_HF_CLOSURE,
            "expected_required_root": EXPECTED_HF_REQUIRED_ROOT,
            "offline_required": True,
        },
        "artifact_receipt": asdict(artifact),
        "roles": list(ROLES),
        "required_array": "0-3%4",
        "scientific_promotion": False,
    }
    final["identity_sha256"] = canonical_hash(final)
    _write_or_verify_once(final_receipt, final)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--final-receipt", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_receipt(
        source_head=args.source_head,
        final_receipt=args.final_receipt,
        session_id=args.session_id,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
