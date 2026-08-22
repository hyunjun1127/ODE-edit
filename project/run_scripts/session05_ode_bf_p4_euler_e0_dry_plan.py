#!/usr/bin/env python3
"""No-model/no-GPU E0 source, session, seal, and calibration dry plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.p4_euler_integrator import P4_EULER_INSTRUCTION_ID
from project.run_scripts.ode_bf.p4_euler_native_reference import (
    build_native_source_contract,
    verify_native_execution_boundary,
)


EXPECTED_SESSION = "codex://threads/01a028a7-9e3c-7541-81ba-efb40555d17d"
EXPECTED_HOST = "server4"
EXPECTED_ROLE = "server-head"
EXPECTED_AGENT = "server4-server-head"
EXPECTED_CONTRACT_SHA256 = "1899bbfa3610ee2edc94044548475dbb39353ebb7aeafeea371c9a879e7d4398"
EXPECTED_STREAM_ROOT = "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6"
EXPECTED_ORDER_SHA256 = "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c"
EXPECTED_EVALUATOR_SHA256 = "72b8ecb737157a42d6a055ffd339dc3f907876a0a98165a00cca49ca9bbed07d"
EXPECTED_EASYEDIT_SEAL_FILE_SHA256 = "176018298d691a1b67ae7ca787c7dcd2bd77d8606044790833d361fd451ddeb3"
EXPECTED_HF_SEAL_FILE_SHA256 = "02db39b636625a513aca0d185d3dab04de2b6ba3ece7d867d241dd3a21c28854"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ODEBFContractError(f"E0 JSON object differs: {path}")
    return value


def _git_config(repo: Path, key: str) -> str:
    process = subprocess.run(
        ["git", "config", "--get", key],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return process.stdout.strip()


def _verify_root_digest(payload: dict[str, Any]) -> None:
    expected = payload.get("root_digest")
    body = dict(payload)
    body.pop("root_digest", None)
    if expected != canonical_hash(body):
        raise ODEBFContractError("E0 numerical/source lock root digest differs")


def _verify_source_manifest(repo: Path, path: Path) -> dict[str, Any]:
    manifest = _json(path)
    _verify_root_digest(manifest)
    for row in manifest.get("entries", []):
        member = repo / row["path"]
        if member.is_symlink() or not member.is_file():
            raise ODEBFContractError("E0 source manifest member differs")
        if member.stat().st_size != row["size"] or _sha(member) != row["sha256"]:
            raise ODEBFContractError("E0 source manifest identity differs")
    return manifest


def build_dry_plan(repo: Path, *, session_id: str) -> dict[str, Any]:
    if session_id != EXPECTED_SESSION:
        raise ODEBFContractError("E0 owner session differs")
    if (
        socket.gethostname() != EXPECTED_HOST
        or _git_config(repo, "agent.role") != EXPECTED_ROLE
        or _git_config(repo, "agent.id") != EXPECTED_AGENT
    ):
        raise ODEBFContractError("E0 host/role/agent identity differs")

    contract = Path(
        "/data/janghj/ODE-edit/local/state/"
        "p4-euler-target-side-semantic-barrier-v1/authoritative-contract.txt"
    )
    contract_stat = contract.stat()
    if (
        contract.is_symlink()
        or not contract.is_file()
        or contract_stat.st_size != 21620
        or _sha(contract) != EXPECTED_CONTRACT_SHA256
        or contract.read_bytes().count(b"\n") != 989
    ):
        raise ODEBFContractError("E0 authoritative contract identity differs")

    numerical_lock_path = (
        repo
        / "project/run_scripts/ode_bf/locks/"
        "numerical_lock_s05_p4_euler_projected_semantic_ode_e0.json"
    )
    numerical_lock = _json(numerical_lock_path)
    _verify_root_digest(numerical_lock)
    if (
        numerical_lock.get("selected_target_horizon") is not None
        or numerical_lock.get("gpu_slurm_authorized") is not False
        or numerical_lock.get("stage_1", {}).get("microsteps") != [1, 3, 5, 10]
        or numerical_lock.get("stage_2", {}).get("microsteps") != [5, 10]
    ):
        raise ODEBFContractError("E0 pending calibration lock differs")

    easyedit_seal_path = repo / "agents/server4/alphaedit-runtime-path-seal.json"
    hf_seal_path = repo / "agents/server4/p4-hf-consumed-closure-seal.json"
    if _sha(easyedit_seal_path) != EXPECTED_EASYEDIT_SEAL_FILE_SHA256:
        raise ODEBFContractError("E0 EasyEdit path-seal tracked identity differs")
    if _sha(hf_seal_path) != EXPECTED_HF_SEAL_FILE_SHA256:
        raise ODEBFContractError("E0 HF closure tracked identity differs")
    easyedit_seal = _json(easyedit_seal_path)
    hf_seal = _json(hf_seal_path)
    if easyedit_seal["mappings"]["easyedit_root"]["runtime_root"] != "/data/janghj/EasyEdit":
        raise ODEBFContractError("E0 EasyEdit runtime root differs")
    if hf_seal["loader_contract"]["model_kwargs"] != {
        "device_map": {"": 0},
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": "torch.float32",
        "trust_remote_code": False,
    }:
        raise ODEBFContractError("E0 HF offline FP32 loader contract differs")

    stream_receipt_path = Path(
        "/data/janghj/ODE-edit/local/state/p4-target-side-semantic-barrier-v1/"
        "sealed-stream/p4-r52-independent-b10x10-sample-order-transfer-v2/"
        "rooted-receipt.json"
    )
    stream_receipt = _json(stream_receipt_path)
    if (
        stream_receipt.get("canonical_stream_root") != EXPECTED_STREAM_ROOT
        or stream_receipt.get("all_request_order_sha256") != EXPECTED_ORDER_SHA256
        or stream_receipt.get("evaluator_configuration_identity_sha256")
        != EXPECTED_EVALUATOR_SHA256
    ):
        raise ODEBFContractError("E0 sealed stream/evaluator identity differs")

    source_manifest_path = (
        repo
        / "project/run_scripts/ode_bf/locks/"
        "source_manifest_s05_p4_euler_projected_semantic_ode_e0.json"
    )
    source_manifest = _verify_source_manifest(repo, source_manifest_path)
    native = {
        method: verify_native_execution_boundary(
            build_native_source_contract(Path("/data/janghj/EasyEdit"), method=method)
        )
        for method in ("native-memit", "native-alphaedit")
    }
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-e0-dry-plan/v1",
        "instruction_id": P4_EULER_INSTRUCTION_ID,
        "status": "E0_PRE_CALIBRATION_PASS_GPU_HOLD",
        "owner_session": session_id,
        "host": EXPECTED_HOST,
        "agent_role": EXPECTED_ROLE,
        "agent_id": EXPECTED_AGENT,
        "authoritative_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "source_manifest_sha256": _sha(source_manifest_path),
        "source_manifest_root": source_manifest["root_digest"],
        "numerical_lock_sha256": _sha(numerical_lock_path),
        "numerical_lock_root": numerical_lock["root_digest"],
        "target_horizon": None,
        "calibration_plan_approval_required": True,
        "easyedit_runtime_root": "/data/janghj/EasyEdit",
        "easyedit_path_seal_file_sha256": EXPECTED_EASYEDIT_SEAL_FILE_SHA256,
        "hf_consumed_closure_file_sha256": EXPECTED_HF_SEAL_FILE_SHA256,
        "hf_offline": True,
        "tensor_state_solver": "FULL_FP32",
        "stream_root": EXPECTED_STREAM_ROOT,
        "order_sha256": EXPECTED_ORDER_SHA256,
        "evaluator_identity_sha256": EXPECTED_EVALUATOR_SHA256,
        "native_reference_source_contracts": native,
        "old_adam_solver_import_count": 0,
        "model_load_count": 0,
        "gpu_action_count": 0,
        "slurm_submit_count": 0,
        "heldout_access_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_dry_plan(args.repo_root.resolve(strict=True), session_id=args.session_id),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
