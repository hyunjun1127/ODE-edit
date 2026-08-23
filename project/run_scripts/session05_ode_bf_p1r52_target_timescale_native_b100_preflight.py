#!/usr/bin/env python3
"""No-model server4 gate for the user-authorized B1 Native references."""

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
from project.run_scripts.ode_bf.p1r52_target_timescale_deployment import (
    EXPECTED_HF_CLOSURE,
    EXPECTED_HF_REQUIRED_ROOT,
    EXPECTED_STREAM_ORDER,
    EXPECTED_STREAM_ROOT,
    build_target_timescale_deployment,
)
from project.run_scripts.ode_bf.p1r52_target_timescale_native_reference import (
    INSTRUCTION_ID,
    METHODS,
    POLICY_PROVENANCE,
    ROLES,
    role_for_cell,
)
from project.run_scripts.session05_ode_bf_p1r52_target_timescale_b100_preflight import (
    EXTRACT_ROOT,
    PRIOR_HF_GATE,
    _verify_contract,
    _verify_stream,
)


NUMERICAL_LOCK = Path(
    "project/run_scripts/ode_bf/locks/"
    "numerical_lock_s05_p1r52_target_timescale_native_b100_user_override_v1.json"
)
SOURCE_FILES = (
    "project/run_scripts/ode_bf/p1_runtime.py",
    "project/run_scripts/ode_bf/p1r52_accepted_z_observation.py",
    "project/run_scripts/ode_bf/p1r52_joint_pc_independent_fp32_runtime.py",
    "project/run_scripts/ode_bf/p1r52_official_sequential_baselines.py",
    "project/run_scripts/ode_bf/p1r52_sequential_runtime.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale_deployment.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale_native_reference.py",
    "project/run_scripts/ode_bf/scalable_batched_native.py",
    "project/run_scripts/ode_bf/tests/test_p1r52_target_timescale_native_reference.py",
    "project/run_scripts/session05_ode_bf_p1r52_target_timescale_native_b100.py",
    "project/run_scripts/session05_ode_bf_p1r52_target_timescale_native_b100_preflight.py",
    "project/run_scripts/session05_ode_bf_p1r52_target_timescale_native_b100_server4.sbatch",
    str(NUMERICAL_LOCK),
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
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != encoded:
            raise ODEBFContractError("create-once Native pre-GPU receipt differs")
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
            raise ODEBFContractError(f"Native source member differs: {relative}")
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r52-target-timescale-native-source-manifest/v1",
        "source_head": head,
        "source_tree": tree,
        "entries": entries,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def build_receipt(
    *, source_head: str, final_receipt: Path, session_id: str
) -> Mapping[str, Any]:
    agent_role = subprocess.check_output(
        ["git", "config", "--get", "agent.role"], cwd=REPO_ROOT, text=True
    ).strip()
    agent_host = subprocess.check_output(
        ["git", "config", "--get", "agent.hostname"], cwd=REPO_ROOT, text=True
    ).strip()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True
    ).strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    if (
        socket.gethostname() != "server4"
        or agent_role != "server-head"
        or agent_host != "server4"
        or os.environ.get("PROJECT_GPU_CAP") != "4"
        or head != source_head
        or dirty
    ):
        raise ODEBFContractError("server4/Native source/cap gate differs")

    contract = _verify_contract()
    stream = _verify_stream()
    lock, lock_sha = _rooted(REPO_ROOT / NUMERICAL_LOCK)
    if (
        lock.get("instruction_id") != INSTRUCTION_ID
        or lock.get("policy_provenance") != POLICY_PROVENANCE
        or lock.get("project_gpu_cap") != 4
        or lock.get("selected_batch") != "B1"
        or lock.get("selected_request_count") != 100
        or lock.get("stream_root") != EXPECTED_STREAM_ROOT
        or lock.get("stream_order") != EXPECTED_STREAM_ORDER
        or [item.get("role") for item in lock.get("methods", ())] != list(ROLES)
        or [item.get("method") for item in lock.get("methods", ())] != list(METHODS)
    ):
        raise ODEBFContractError("target-timescale Native numerical lock differs")

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
        "schema": "ode-edit-s05-p1r52-target-timescale-native-final-pre-gpu/v1",
        "instruction_id": INSTRUCTION_ID,
        "policy_provenance": POLICY_PROVENANCE,
        "status": "NATIVE_REFERENCE_FINAL_PRE_GPU_PASS",
        "model_load_authorized": True,
        "hostname": "server4",
        "agent_role": agent_role,
        "agent_hostname": agent_host,
        "session_id": session_id,
        "project_gpu_cap": 4,
        "source_head": head,
        "source_tree": tree,
        "source_manifest": _source_manifest(head, tree),
        "authoritative_contract": contract,
        "user_override": {
            "supersedes_native_execution_count_zero_for_reference_statistics_only": True,
            "target_timescale_science_changed": False,
            "reference_selection_influence_count": 0,
            "scientific_promotion": False,
        },
        "stream_binding": stream,
        "numerical_lock": {
            "path": str(REPO_ROOT / NUMERICAL_LOCK),
            "sha256": lock_sha,
            "root_digest": lock["root_digest"],
        },
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
            "array": "0-1%2",
            "cells": [
                {"array_cell": index, "role": role_for_cell(index), "method": method}
                for index, method in enumerate(METHODS)
            ],
            "model": "llama3-8b-inst",
            "B1_only": True,
            "request_count": 100,
            "full_fp32": True,
            "offline": True,
            "reference_only": True,
            "target_timescale_field_execution_count": 0,
            "selection_influence_count": 0,
        },
    }
    if (
        deployment.base_guard.model_seal.closure_identity != EXPECTED_HF_CLOSURE
        or deployment.base_guard.model_seal.required_root != EXPECTED_HF_REQUIRED_ROOT
    ):
        raise ODEBFContractError("Native HF binding differs")
    final["identity_sha256"] = canonical_hash(final)
    _write_or_verify_once(final_receipt, final)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--final-receipt", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args(argv)
    receipt = build_receipt(
        source_head=args.source_head,
        final_receipt=args.final_receipt,
        session_id=args.session_id,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
