#!/usr/bin/env python3
"""No-model/no-CUDA final gate for the server4 target-timescale array."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
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
from project.run_scripts.ode_bf.p1r52_b100x10_stream import verify_p1r52_b100x10_stream
from project.run_scripts.ode_bf.p1r52_target_timescale import INSTRUCTION_ID, SCHEDULES
from project.run_scripts.ode_bf.p1r52_target_timescale_b100 import role_for_cell
from project.run_scripts.ode_bf.p1r52_target_timescale_deployment import (
    EXPECTED_EVALUATOR_IDENTITY,
    EXPECTED_HF_CLOSURE,
    EXPECTED_HF_REQUIRED_ROOT,
    EXPECTED_STREAM_ORDER,
    EXPECTED_STREAM_ROOT,
    build_target_timescale_deployment,
)


CONTRACT = Path("/data/janghj/ODE-edit/local/state/p1r52-target-timescale-ablation-b100-v1/authoritative-contract.txt")
CONTRACT_SHA256 = "195d80f319b92c616265b775ac798e4379a27c278798f2536391855b9d8b9e14"
CONTRACT_BYTES = 15_041
CONTRACT_LINES = 556
STREAM_RECEIPT = Path("/data/janghj/ODE-edit/local/state/phase123-server1-sample-stream-v1/server4-stream-seal-receipt.json")
STREAM_RECEIPT_SHA256 = "42cab040657025093e7644bb3a9dda550f29bf413f23289c14059ce5d79657a3"
STREAM_RECEIPT_IDENTITY = "9cbcee804df00212ea2d754d25af352c1d111331d4749e08b9afb231f894c5f6"
EXTRACT_ROOT = Path("/data/janghj/ODE-edit/local/state/phase123-server1-sample-stream-v1/extracted-v1/phase123-server1-single-canonical-stream-v1")
PRIOR_HF_GATE = Path("/data/janghj/ODE-edit/local/state/p4-euler-target-side-semantic-barrier-v1/readiness/za-llama-h1-f4fbfb5-r1/final-pre-gpu.json")
NUMERICAL_LOCK = Path("project/run_scripts/ode_bf/locks/numerical_lock_s05_p1r52_target_timescale_b100_v1.json")
TRACKED_STREAM_LOCK = Path("project/run_scripts/ode_bf/locks/p1r52_sequential_b100x10_stream_seal.json")
SOURCE_FILES = (
    "project/run_scripts/alphaedit_runtime_path_seal.py",
    "project/run_scripts/ode_alloc/p0_artifacts.py",
    "project/run_scripts/ode_bf/artifacts.py",
    "project/run_scripts/ode_bf/contracts.py",
    "project/run_scripts/ode_bf/p1_runtime.py",
    "project/run_scripts/ode_bf/p1_scalable_batched_experiment.py",
    "project/run_scripts/ode_bf/p1r52_c_writer_kstep.py",
    "project/run_scripts/ode_bf/p1r52_c_writer_kstep_cache.py",
    "project/run_scripts/ode_bf/p1r52_joint_pc_fp32_runtime.py",
    "project/run_scripts/ode_bf/p1r52_r42_safe_kdc.py",
    "project/run_scripts/ode_bf/p1r52_residual_reserve_phase_a_execution.py",
    "project/run_scripts/ode_bf/p1r52_sequential_runtime.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale_b100.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale_deployment.py",
    "project/run_scripts/ode_bf/p4_hf_consumed_closure.py",
    "project/run_scripts/ode_bf/tests/test_p1r52_target_timescale.py",
    "project/run_scripts/session05_ode_bf_p1r52_target_timescale_b100.py",
    "project/run_scripts/session05_ode_bf_p1r52_target_timescale_b100_dry_plan.py",
    "project/run_scripts/session05_ode_bf_p1r52_target_timescale_b100_preflight.py",
    "project/run_scripts/session05_ode_bf_p1r52_target_timescale_b100_server4.sbatch",
    str(NUMERICAL_LOCK),
    str(TRACKED_STREAM_LOCK),
)


def _read_json(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ODEBFContractError(f"regular file required: {path}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, Mapping):
        raise ODEBFContractError(f"JSON object required: {path}")
    return value


def _rooted(path: Path) -> tuple[Mapping[str, Any], str]:
    value = dict(_read_json(path))
    root = value.pop("root_digest", None)
    if root != canonical_hash(value):
        raise ODEBFContractError(f"rooted JSON differs: {path}")
    value["root_digest"] = root
    return value, sha256_file(path)


def _write_or_verify_once(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != encoded:
            raise ODEBFContractError("create-once pre-GPU receipt differs")
        return digest
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
    return digest


def _source_manifest(head: str, tree: str) -> Mapping[str, Any]:
    entries = []
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ODEBFContractError(f"source manifest member differs: {relative}")
        entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r52-target-timescale-source-manifest/v1",
        "source_head": head,
        "source_tree": tree,
        "entries": entries,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _verify_contract() -> Mapping[str, Any]:
    observed = CONTRACT.lstat()
    if (
        CONTRACT.is_symlink()
        or not CONTRACT.is_file()
        or (observed.st_mode & 0o777) != 0o600
        or observed.st_size != CONTRACT_BYTES
        or sha256_file(CONTRACT) != CONTRACT_SHA256
        or CONTRACT.read_bytes().count(b"\n") != CONTRACT_LINES
    ):
        raise ODEBFContractError("authoritative contract identity differs")
    return {"path": str(CONTRACT), "sha256": CONTRACT_SHA256, "bytes": CONTRACT_BYTES, "lines": CONTRACT_LINES, "mode": "0600", "full_read": True}


def _verify_stream() -> Mapping[str, Any]:
    receipt = _read_json(STREAM_RECEIPT)
    if (
        sha256_file(STREAM_RECEIPT) != STREAM_RECEIPT_SHA256
        or receipt.get("identity_sha256") != STREAM_RECEIPT_IDENTITY
        or receipt.get("status") != "FULL_REHASH_PASS_STREAM_READY_LAUNCHER_HOLD"
        or receipt.get("counts", {}).get("sample_duplication_count") != 0
        or receipt.get("identities", {}).get("stream_root_sha256") != EXPECTED_STREAM_ROOT
        or receipt.get("identities", {}).get("all_request_order_sha256") != EXPECTED_STREAM_ORDER
    ):
        raise ODEBFContractError("server4 stream receipt differs")
    transferred = verify_p1r52_b100x10_stream(_read_json(EXTRACT_ROOT / "canonical/p1r52_sequential_b100x10_stream_seal.json"))
    tracked = verify_p1r52_b100x10_stream(_read_json(REPO_ROOT / TRACKED_STREAM_LOCK))
    index = _read_json(EXTRACT_ROOT / "samples/ordered-record-index.json")
    records = index.get("records")
    if (
        transferred != tracked
        or transferred["root_digest"] != EXPECTED_STREAM_ROOT
        or transferred["all_request_order_sha256"] != EXPECTED_STREAM_ORDER
        or not isinstance(records, list)
        or len(records) != 1000
        or [row["request_sha256"] for row in records[:100]] != [row["request_sha256"] for row in transferred["requests"][:100]]
        or sha256_file(EXTRACT_ROOT / "canonical/context-evaluator-input-identity.json") != EXPECTED_EVALUATOR_IDENTITY
    ):
        raise ODEBFContractError("transferred/tracked B1 stream binding differs")
    return {
        "receipt_path": str(STREAM_RECEIPT),
        "receipt_sha256": STREAM_RECEIPT_SHA256,
        "receipt_identity": STREAM_RECEIPT_IDENTITY,
        "stream_root": EXPECTED_STREAM_ROOT,
        "order_root": EXPECTED_STREAM_ORDER,
        "evaluator_identity": EXPECTED_EVALUATOR_IDENTITY,
        "dataset_identity": transferred["source"]["sha256"],
        "B1_request_count": 100,
        "B1_request_order_sha256": transferred["batch_ordered_request_digest_v1"][0],
        "sample_duplication_count": 0,
        "transferred_equals_tracked_lock": True,
    }


def build_receipt(*, source_head: str, final_receipt: Path, session_id: str) -> Mapping[str, Any]:
    role = subprocess.check_output(["git", "config", "--get", "agent.role"], cwd=REPO_ROOT, text=True).strip()
    agent_host = subprocess.check_output(["git", "config", "--get", "agent.hostname"], cwd=REPO_ROOT, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True).strip()
    if (
        socket.gethostname() != "server4"
        or role != "server-head"
        or agent_host != "server4"
        or os.environ.get("PROJECT_GPU_CAP", "2") != "2"
        or head != source_head
        or subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, check=True).stdout
    ):
        raise ODEBFContractError("server4/source/cap gate differs")
    contract = _verify_contract()
    stream = _verify_stream()
    lock, lock_sha = _rooted(REPO_ROOT / NUMERICAL_LOCK)
    expected_cells = [schedule.raw_free_payload() for schedule in SCHEDULES]
    if (
        lock.get("instruction_id") != INSTRUCTION_ID
        or lock.get("selected_batch") != "B1"
        or lock.get("selected_request_count") != 100
        or lock.get("cells") != [
            {key: row[key] for key in ("cell", "target_horizon", "microsteps_per_outer", "target_dt", "total_field_evaluations")}
            for row in expected_cells
        ]
    ):
        raise ODEBFContractError("target-timescale numerical lock differs")
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
        "schema": "ode-edit-s05-p1r52-target-timescale-final-pre-gpu/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "FINAL_PRE_GPU_PASS",
        "model_load_authorized": True,
        "hostname": "server4",
        "agent_role": role,
        "agent_hostname": agent_host,
        "session_id": session_id,
        "project_gpu_cap": 2,
        "source_head": head,
        "source_tree": tree,
        "source_manifest": _source_manifest(head, tree),
        "authoritative_contract": contract,
        "stream_binding": stream,
        "numerical_lock": {"path": str(REPO_ROOT / NUMERICAL_LOCK), "sha256": lock_sha, "root_digest": lock["root_digest"]},
        "deployment_identity": deployment.identity_sha256,
        "easyedit_runtime_seal_root": deployment.runtime_path_seal.root_digest,
        "hf_binding": {
            "closure_identity": deployment.base_guard.model_seal.closure_identity,
            "required_root": deployment.base_guard.model_seal.required_root,
            "snapshot": str(deployment.base_guard.snapshot),
            "revision": deployment.base_guard.model_seal.revision,
            "expected_closure_identity": EXPECTED_HF_CLOSURE,
            "expected_required_root": EXPECTED_HF_REQUIRED_ROOT,
            "extra_influence_count": 0,
            "prior_full_verification_receipt_sha256": sha256_file(PRIOR_HF_GATE),
        },
        "artifact_receipt": asdict(artifact),
        "dry_plan": {
            "array": "0-4%2",
            "cells": [{"array_cell": index, "role": role_for_cell(index), **schedule.raw_free_payload()} for index, schedule in enumerate(SCHEDULES)],
            "model": "llama3-8b-inst",
            "B1_only": True,
            "writer_calls_per_cell": 8,
            "heldout_K": [1, 4, 8],
            "native_execution_count": 0,
            "full_fp32": True,
        },
        "Z0_control": {
            "existing_sealed_control_reused": False,
            "reason": "PHASE3_C3_B1_RAW_TERMINAL_NOT_PRESENT_ON_SERVER4_MAIN_LINEAGE",
            "execution_required": True,
            "pre_execution_scheduler_exact_identity_gate": "PASS",
        },
        "scientific_promotion": False,
    }
    final["identity_sha256"] = canonical_hash(final)
    final["file_sha256"] = _write_or_verify_once(final_receipt, final)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--final-receipt", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args(argv)
    result = build_receipt(source_head=args.source_head, final_receipt=args.final_receipt, session_id=args.session_id)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
