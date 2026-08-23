#!/usr/bin/env python3
"""No-model/no-CUDA final gate for the server4 P1R53 two-arm array."""

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
    EXPECTED_HF_CLOSURE,
    EXPECTED_HF_REQUIRED_ROOT,
    build_target_timescale_deployment,
)
from project.run_scripts.ode_bf.p1r53_request_local_speed import INSTRUCTION_ID
from project.run_scripts.ode_bf.p1r53_request_local_speed_b100 import (
    ROLES,
    role_for_cell,
)
from project.run_scripts.session05_ode_bf_p1r52_target_timescale_b100_preflight import (
    EXTRACT_ROOT,
    PRIOR_HF_GATE,
    _read_json,
    _rooted,
    _verify_stream,
    _write_or_verify_once,
)


CONTRACT = Path("/data/janghj/ODE-edit/local/state/p1r53-request-local-speed-llama-b100-v1/authoritative-contract.txt")
CONTRACT_SHA256 = "fde535a1f88b10bb82011166255a5d4d08a3c0a3a3d7499933596b2b516da322"
CONTRACT_BYTES = 17_305
CONTRACT_LINES = 808
PROJECT_GPU_CAP = 4
ARRAY_CONCURRENCY = 2
NUMERICAL_LOCK = Path("project/run_scripts/ode_bf/locks/numerical_lock_s05_p1r53_request_local_speed_llama_b100_v1.json")
REFERENCE_ROOT = Path("/data/janghj/ODE-edit/local/worktrees/p1r52-target-timescale-b100-v1/local/odebf/results")
REFERENCES = {
    "global_p1r52_z0_coarse": (
        REFERENCE_ROOT / "s05-p1r52-target-timescale-b100-z0-coarse-tech-r4-v1",
        "683102c81fbc93b4515d885c1a3d78491198cbe04992004101f4e52977273900",
        "6a319aa7f94f3866e23e19f317dad4546e72afd1ceeec8da014644eb1e22dedd",
    ),
    "native_alphaedit": (
        REFERENCE_ROOT / "s05-p1r52-target-timescale-native-b100-official-alphaedit-user-override-v1",
        "4e26d1637202a01a3c9a88207f71b320acc7d02c287a0fa195ed54309f370549",
        "f7a3b8ceec31630e6f27d353b31c21593e2ac03042576f664e60d38f9b9fa252",
    ),
    "native_memit": (
        REFERENCE_ROOT / "s05-p1r52-target-timescale-native-b100-official-memit-user-override-tech-r2-v1",
        "2d4430f5081638984e65aecbb3e3a3e98a1d50bd849ae39e12abd6c0ed781ebf",
        "992a041af73cb860ca1239be44d07f3e8bffbd74cd7fbc5ce10ba7d489a5e8e8",
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
    "project/run_scripts/ode_bf/p1r52_r42_safe_kdc.py",
    "project/run_scripts/ode_bf/p1r52_residual_reserve_phase_a_execution.py",
    "project/run_scripts/ode_bf/p1r52_sequential_runtime.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale_b100.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale_deployment.py",
    "project/run_scripts/ode_bf/p1r53_request_local_speed.py",
    "project/run_scripts/ode_bf/p1r53_request_local_speed_b100.py",
    "project/run_scripts/ode_bf/p4_hf_consumed_closure.py",
    "project/run_scripts/ode_bf/tests/test_p1r52_r42_safe_kdc.py",
    "project/run_scripts/ode_bf/tests/test_p1r52_target_timescale.py",
    "project/run_scripts/ode_bf/tests/test_p1r53_request_local_speed.py",
    "project/run_scripts/session05_ode_bf_p1r53_request_local_speed_b100.py",
    "project/run_scripts/session05_ode_bf_p1r53_request_local_speed_b100_dry_plan.py",
    "project/run_scripts/session05_ode_bf_p1r53_request_local_speed_b100_preflight.py",
    "project/run_scripts/session05_ode_bf_p1r53_request_local_speed_b100_server4.sbatch",
    str(NUMERICAL_LOCK),
    "project/run_scripts/ode_bf/locks/p1r52_sequential_b100x10_stream_seal.json",
)


def _source_manifest(head: str, tree: str) -> Mapping[str, Any]:
    entries = []
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ODEBFContractError(f"P1R53 source member differs: {relative}")
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r53-request-local-speed-source-manifest/v1",
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
        raise ODEBFContractError("P1R53 authoritative contract identity differs")
    return {
        "path": str(CONTRACT),
        "sha256": CONTRACT_SHA256,
        "bytes": CONTRACT_BYTES,
        "lines": CONTRACT_LINES,
        "mode": "0600",
        "full_read": True,
    }


def _verify_references(expected_b1_order: str) -> Mapping[str, Any]:
    observed: dict[str, Any] = {}
    for label, (root, terminal_sha, manifest_sha) in REFERENCES.items():
        terminal_path = root / "terminal.json"
        manifest_path = root / "manifest.json"
        terminal = _read_json(terminal_path)
        if (
            sha256_file(terminal_path) != terminal_sha
            or sha256_file(manifest_path) != manifest_sha
            or terminal.get("status") != "TERMINAL_VALID"
            or terminal.get("request_count") != 100
            or terminal.get("request_order_sha256") != expected_b1_order
            or terminal.get("stream_root") != "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
            or terminal.get("stream_order") != "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"
            or terminal.get("W0_restored") is not True
        ):
            raise ODEBFContractError(f"P1R53 external reference differs: {label}")
        observed[label] = {
            "root": str(root),
            "terminal_sha256": terminal_sha,
            "manifest_sha256": manifest_sha,
            "terminal_identity": terminal.get("identity_sha256"),
            "execution_influence_count": 0,
        }
    return observed


def build_receipt(*, source_head: str, final_receipt: Path, session_id: str) -> Mapping[str, Any]:
    role = subprocess.check_output(
        ["git", "config", "--get", "agent.role"], cwd=REPO_ROOT, text=True
    ).strip()
    agent_host = subprocess.check_output(
        ["git", "config", "--get", "agent.hostname"], cwd=REPO_ROOT, text=True
    ).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True).strip()
    if (
        socket.gethostname() != "server4"
        or role != "server-head"
        or agent_host != "server4"
        or os.environ.get("PROJECT_GPU_CAP", str(PROJECT_GPU_CAP)) != str(PROJECT_GPU_CAP)
        or head != source_head
        or subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True,
            stdout=subprocess.PIPE, check=True,
        ).stdout
    ):
        raise ODEBFContractError("P1R53 server4/source/cap gate differs")
    contract = _verify_contract()
    stream = _verify_stream()
    lock, lock_sha = _rooted(REPO_ROOT / NUMERICAL_LOCK)
    if (
        lock.get("instruction_id") != INSTRUCTION_ID
        or lock.get("model_alias") != "llama3-8b-inst"
        or lock.get("selected_batch") != "B1"
        or lock.get("selected_request_count") != 100
        or lock.get("physical_outer_count") != 8
        or lock.get("target_dt") != 0.125
        or lock.get("target_microsteps_per_outer") != 1
        or lock.get("target_horizon") != 1.0
        or [row.get("arm") for row in lock.get("arms", [])] != ["LP-S", "LFD-E"]
        or lock.get("heldout_outer_indices") != [8]
    ):
        raise ODEBFContractError("P1R53 numerical lock differs")
    references = _verify_references(str(stream["B1_request_order_sha256"]))
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
        "schema": "ode-edit-s05-p1r53-request-local-speed-final-pre-gpu/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "FINAL_PRE_GPU_PASS",
        "model_load_authorized": True,
        "hostname": "server4",
        "agent_role": role,
        "agent_hostname": agent_host,
        "session_id": session_id,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "array_concurrency": ARRAY_CONCURRENCY,
        "source_head": head,
        "source_tree": tree,
        "source_manifest": _source_manifest(head, tree),
        "authoritative_contract": contract,
        "stream_binding": stream,
        "numerical_lock": {
            "path": str(REPO_ROOT / NUMERICAL_LOCK),
            "sha256": lock_sha,
            "root_digest": lock["root_digest"],
        },
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
            "extra_influence_count": 0,
            "prior_full_verification_receipt_sha256": sha256_file(PRIOR_HF_GATE),
        },
        "artifact_receipt": asdict(artifact),
        "dry_plan": {
            "array": "0-1%2",
            "cells": [
                {"array_cell": index, "role": role_for_cell(index), "arm": arm}
                for index, arm in enumerate(("LP-S", "LFD-E"))
            ],
            "model": "llama3-8b-inst",
            "B1_only": True,
            "writer_calls_per_cell": 8,
            "heldout_K": [8],
            "native_execution_count": 0,
            "full_fp32": True,
        },
        "mechanical_gate": {
            "global_p1r52_identity_regression": "PASS",
            "request_local_amplitude_and_energy": "PASS",
            "cross_request_contamination": "PASS",
            "permutation_and_duplicate_invariance_under_frozen_s_B": "PASS",
            "lfd_k0_entry_identity_and_frozen_kappa": "PASS",
            "no_global_decision_access": "PASS",
            "additional_forward_backward_count": 0,
            "per_request_backward_loop_count": 0,
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
    result = build_receipt(
        source_head=args.source_head,
        final_receipt=args.final_receipt,
        session_id=args.session_id,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
