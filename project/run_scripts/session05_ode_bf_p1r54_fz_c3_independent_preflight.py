#!/usr/bin/env python3
"""No-model/no-CUDA final gate for the P1R54 FZ independent array."""

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
from project.run_scripts.ode_bf.p1r54_fz_c3_independent import (
    INSTRUCTION_ID,
    ROLES,
    STREAM_ORDER,
    STREAM_ROOT,
    atomic_source_equivalence,
    cell_config,
)
from project.run_scripts.session05_ode_bf_p1r52_target_timescale_b100_preflight import (
    EXTRACT_ROOT,
    PRIOR_HF_GATE,
    _read_json,
    _rooted,
    _verify_stream,
    _write_or_verify_once,
)
from project.run_scripts.session05_ode_bf_p1r54_fz_c3_independent_dry_plan import (
    build_plan,
)
from project.run_scripts.session05_ode_bf_p1r54_fz_c3_sequential_preflight import (
    verify_references,
)


PROJECT_GPU_CAP = 4
JOB_GPU_COUNT = 1
ARRAY_THROTTLE = 3
IMPLEMENTATION_BASE_HEAD = "6b7364f4caf3b3fbead83a3f5bcb8c39a79b38f5"
IMPLEMENTATION_BASE_TREE = "591d49f60caf3e5311c744478313a63a68c79d22"
NUMERICAL_LOCK = Path(
    "project/run_scripts/ode_bf/locks/"
    "numerical_lock_s05_p1r54_fz_c3_independent_10xb100_v1.json"
)
SEQUENTIAL_NUMERICAL_LOCK = Path(
    "project/run_scripts/ode_bf/locks/"
    "numerical_lock_s05_p1r54_fz_c3_sequential_10xb100_v1.json"
)
SEQUENTIAL_NUMERICAL_LOCK_SHA = (
    "933e70a02777c4fef3304c87136b7478c8170549351a513d31eb78c05896d89a"
)
ATOMIC_NUMERICAL_LOCK = Path(
    "project/run_scripts/ode_bf/locks/"
    "numerical_lock_s05_p1r54_energyfree_localz_llama_b100_v1.json"
)
ATOMIC_NUMERICAL_LOCK_SHA = (
    "2ce9c665ae45c6df0f06199182f3d7770768fec2d491b7b751a11d127d59c675"
)
UNCHANGED_SCIENCE_SOURCES = {
    "project/run_scripts/ode_bf/p1r52_target_timescale_b100.py": (
        "0b19c087df31c4b84dd680366682d91cd6d58e73a38e69fecc0d00eac543f4f3"
    ),
    "project/run_scripts/ode_bf/p1r54_energyfree_localz.py": (
        "9a40912bc07cbd0049e332c3b2bc1a1ff8f105a7cd83cffe9faa5f4f2f83f9e6"
    ),
    "project/run_scripts/ode_bf/p1r54_fz_c3_sequential.py": (
        "00c605d44328a94d22a5a79cf5f16c32a3bab317b7de8968fa85891cb748514d"
    ),
}
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
    "project/run_scripts/ode_bf/p1r52_joint_pc_fp32_runtime.py",
    "project/run_scripts/ode_bf/p1r52_sequential_runtime.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale_b100.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale_deployment.py",
    "project/run_scripts/ode_bf/p1r54_energyfree_localz.py",
    "project/run_scripts/ode_bf/p1r54_fz_c3_sequential.py",
    "project/run_scripts/ode_bf/p1r54_fz_c3_independent.py",
    "project/run_scripts/ode_bf/p4_hf_consumed_closure.py",
    "project/run_scripts/ode_bf/tests/test_p1r54_energyfree_localz.py",
    "project/run_scripts/ode_bf/tests/test_p1r54_fz_c3_sequential.py",
    "project/run_scripts/ode_bf/tests/test_p1r54_fz_c3_independent.py",
    "project/run_scripts/session05_ode_bf_p1r54_fz_c3_independent.py",
    "project/run_scripts/session05_ode_bf_p1r54_fz_c3_independent_analyze.py",
    "project/run_scripts/session05_ode_bf_p1r54_fz_c3_independent_dry_plan.py",
    "project/run_scripts/session05_ode_bf_p1r54_fz_c3_independent_preflight.py",
    "project/run_scripts/session05_ode_bf_p1r54_fz_c3_independent_server4.sbatch",
    str(NUMERICAL_LOCK),
    str(SEQUENTIAL_NUMERICAL_LOCK),
    str(ATOMIC_NUMERICAL_LOCK),
    "project/run_scripts/ode_bf/locks/p1r52_sequential_b100x10_stream_seal.json",
)


def source_manifest(head: str, tree: str) -> Mapping[str, Any]:
    entries = []
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ODEBFContractError(
                f"P1R54 FZ independent source member differs: {relative}"
            )
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-fz-c3-independent-source-manifest/v1",
        "source_head": head,
        "source_tree": tree,
        "entries": entries,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def verify_unchanged_science_sources() -> Mapping[str, Any]:
    rows = []
    for relative, expected_sha in UNCHANGED_SCIENCE_SOURCES.items():
        observed = sha256_file(REPO_ROOT / relative)
        if observed != expected_sha:
            raise ODEBFContractError(
                f"P1R54 independent reused science source differs: {relative}"
            )
        rows.append({"path": relative, "sha256": observed, "changed_count": 0})
    return {
        "status": "UNCHANGED_REUSED_SCIENCE_PASS",
        "sources": rows,
        "field_writer_cache_evaluator_change_count": 0,
    }


def cpu_independent_entry_fixture() -> Mapping[str, Any]:
    base = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    expected_sha = tensor_sha256(base)
    rows = []
    for cell in range(10):
        parameter = torch.nn.Parameter(base.clone())
        pointer = int(parameter.data_ptr())
        with torch.no_grad():
            parameter.add_(float(cell + 1))
            parameter.copy_(base)
        config = cell_config(cell)
        row = {
            "cell": cell,
            "canonical_batch_index": config.batch_index,
            "W0_sha256": tensor_sha256(parameter),
            "W0_pointer_restored_exact": int(parameter.data_ptr()) == pointer,
            "weight_entry_version": config.weight_entry_version,
            "alpha_cache_entry_identity": "COLD_ALPHA_CACHE_ENTRY_V0_WIDTH0",
            "alpha_cache_entry_version": config.alpha_cache_entry_version,
            "alpha_cache_entry_width": config.alpha_cache_entry_width,
            "state_namespace": config.result_name,
            "cross_cell_alias_count": 0,
            "cross_cell_continuity_count": 0,
        }
        rows.append(row)
    if (
        {row["W0_sha256"] for row in rows} != {expected_sha}
        or not all(row["W0_pointer_restored_exact"] for row in rows)
        or {row["weight_entry_version"] for row in rows} != {0}
        or {row["alpha_cache_entry_version"] for row in rows} != {0}
        or {row["alpha_cache_entry_width"] for row in rows} != {0}
        or len({row["state_namespace"] for row in rows}) != 10
        or torch.cuda.is_initialized()
    ):
        raise ODEBFContractError("P1R54 independent CPU entry fixture differs")
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-fz-c3-independent-cpu-entry-fixture/v1",
        "status": "PASS",
        "device": "cpu",
        "model_load_count": 0,
        "cuda_initialization_count": 0,
        "rows": rows,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def build_receipt(
    *, source_head: str, final_receipt: Path, session_id: str
) -> Mapping[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True
    ).strip()
    role = subprocess.check_output(
        ["git", "config", "--get", "agent.role"], cwd=REPO_ROOT, text=True
    ).strip()
    agent_host = subprocess.check_output(
        ["git", "config", "--get", "agent.hostname"], cwd=REPO_ROOT, text=True
    ).strip()
    implementation_base_tree = subprocess.check_output(
        ["git", "rev-parse", f"{IMPLEMENTATION_BASE_HEAD}^{{tree}}"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    origin_main = subprocess.check_output(
        ["git", "rev-parse", "refs/remotes/origin/main"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if (
        socket.gethostname() != "server4"
        or role != "server-head"
        or agent_host != "server4"
        or head != source_head
        or origin_main != head
        or implementation_base_tree != IMPLEMENTATION_BASE_TREE
        or subprocess.run(
            ["git", "merge-base", "--is-ancestor", IMPLEMENTATION_BASE_HEAD, head],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        != 0
        or os.environ.get("PROJECT_GPU_CAP", str(PROJECT_GPU_CAP))
        != str(PROJECT_GPU_CAP)
        or subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout
    ):
        raise ODEBFContractError("P1R54 FZ independent host/source/base/cap differs")

    stream = _verify_stream()
    tracked_stream, tracked_stream_sha = _rooted(
        REPO_ROOT
        / "project/run_scripts/ode_bf/locks/p1r52_sequential_b100x10_stream_seal.json"
    )
    lock, lock_sha = _rooted(REPO_ROOT / NUMERICAL_LOCK)
    sequential_lock, sequential_lock_sha = _rooted(REPO_ROOT / SEQUENTIAL_NUMERICAL_LOCK)
    atomic_lock, atomic_lock_sha = _rooted(REPO_ROOT / ATOMIC_NUMERICAL_LOCK)
    dry_plan = build_plan()
    if (
        tracked_stream.get("root_digest") != STREAM_ROOT
        or tracked_stream.get("all_request_order_sha256") != STREAM_ORDER
        or tracked_stream.get("batch_ordered_request_digest_v1")
        != lock.get("batch_ordered_request_digest_v1")
        or lock.get("instruction_id") != INSTRUCTION_ID
        or lock.get("execution_mode") != "INDEPENDENT_BATCH"
        or lock.get("canonical_batches") != [f"B{index}" for index in range(1, 11)]
        or lock.get("total_request_count") != 1000
        or lock.get("target_clock", {}).get("field_evaluations_total") != 80
        or lock.get("writer", {}).get("calls_total") != 80
        or lock.get("writer", {}).get("layer_applies_total") != 400
        or lock.get("independent_state", {}).get("cross_batch_weight_continuity")
        is not False
        or lock.get("independent_state", {}).get("cross_batch_alpha_cache_continuity")
        is not False
        or any(lock.get("forbidden_decision_access", {}).values())
        or sequential_lock_sha != SEQUENTIAL_NUMERICAL_LOCK_SHA
        or sequential_lock.get("root_digest")
        != "928d326a6e233e1fc42936febb0a0557a6aec773e39c60b73b8c856217c19154"
        or atomic_lock_sha != ATOMIC_NUMERICAL_LOCK_SHA
        or atomic_lock.get("root_digest")
        != "b9f7ca3beac529491b4cf16237f8b7887a3f02913110639ed2c29a63dc4286ad"
        or dry_plan.get("sample_payload_count") != 1
        or dry_plan.get("sample_payload_copy_count") != 0
        or len(dry_plan.get("cells", [])) != 10
    ):
        raise ODEBFContractError("P1R54 FZ independent stream/lock/dry-plan differs")
    science_sources = verify_unchanged_science_sources()
    cpu_fixture = cpu_independent_entry_fixture()
    references = verify_references(
        b1_order=str(tracked_stream["batch_ordered_request_digest_v1"][0])
    )
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
        "schema": "ode-edit-s05-p1r54-fz-c3-independent-final-pre-gpu/v1",
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
        "implementation_base_head": IMPLEMENTATION_BASE_HEAD,
        "implementation_base_tree": IMPLEMENTATION_BASE_TREE,
        "origin_main_at_gate": origin_main,
        "source_manifest": source_manifest(head, tree),
        "authoritative_envelope": {
            "provenance": "GH_DELEGATED_AUTHORITATIVE_TASK_ENVELOPE",
            "instruction_id": INSTRUCTION_ID,
            "full_read": True,
        },
        "stream_binding": {
            **dict(stream),
            "all_batch_order_sha256": tracked_stream[
                "batch_ordered_request_digest_v1"
            ],
            "sample_payload_count": 1,
            "sample_payload_copy_count": 0,
        },
        "tracked_stream_lock_sha256": tracked_stream_sha,
        "numerical_lock": {
            "path": str(REPO_ROOT / NUMERICAL_LOCK),
            "sha256": lock_sha,
            "root_digest": lock["root_digest"],
        },
        "sequential_science_lock": {
            "path": str(REPO_ROOT / SEQUENTIAL_NUMERICAL_LOCK),
            "sha256": sequential_lock_sha,
            "root_digest": sequential_lock["root_digest"],
            "mutation_count": 0,
        },
        "atomic_science_lock": {
            "path": str(REPO_ROOT / ATOMIC_NUMERICAL_LOCK),
            "sha256": atomic_lock_sha,
            "root_digest": atomic_lock["root_digest"],
            "mutation_count": 0,
        },
        "unchanged_science_sources": science_sources,
        "cpu_independent_entry_fixture": cpu_fixture,
        "atomic_source_equivalence": [
            atomic_source_equivalence(batch_index)
            for batch_index in range(1, 11)
        ],
        "external_references": references,
        "reference_execution_count": 0,
        "deployment_identity": deployment.identity_sha256,
        "easyedit_runtime_seal_root": deployment.runtime_path_seal.root_digest,
        "hf_binding": {
            "closure_identity": deployment.base_guard.model_seal.closure_identity,
            "required_root": deployment.base_guard.model_seal.required_root,
            "snapshot": str(deployment.base_guard.snapshot),
            "revision": deployment.base_guard.model_seal.revision,
            "expected_closure_identity": EXPECTED_HF_CLOSURE,
            "expected_required_root": EXPECTED_HF_REQUIRED_ROOT,
            "offline_required": True,
            "extra_influence_count": 0,
        },
        "artifact_receipt": asdict(artifact),
        "dry_plan": dry_plan,
        "roles": list(ROLES),
        "submission_release_boundary": {
            "slurm_submission_performed": False,
            "required_pre_submit_checks": [
                "job23682_state_and_gpu",
                "server4_janghj_active_pending_gpu",
                "active_plus_new_lte_4",
            ],
            "required_array": "0-9%3",
        },
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
