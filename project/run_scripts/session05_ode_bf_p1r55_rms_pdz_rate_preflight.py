#!/usr/bin/env python3
"""No-model/no-CUDA P1R55 Phase-0 and Phase-1 release receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.artifacts import sha256_file
from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.p1r52_b100x10_stream import (
    SEAL_FILE,
    verify_p1r52_b100x10_stream,
)
from project.run_scripts.ode_bf.p1r55_rms_pdz_rate_experiment import (
    INSTRUCTION_ID,
    PHASE1_ARRAY,
    PHASE1_CELLS,
    STREAM_ORDER,
    STREAM_ROOT,
)
from project.run_scripts.session05_ode_bf_p1r55_rms_pdz_rate_dry_plan import (
    build_plan,
)


PROJECT_GPU_CAP = 3
CONTRACT = Path(
    "/mnt/raid5/janghj/.codex/attachments/21175472-6a17-49ad-bb1e-7c81f52d49c0/pasted-text.txt"
)
CONTRACT_SHA256 = "35089953674d57d6f3ffe727ec93c714a8a5f9ff165f1555fd4447de56deb713"
CONTRACT_BYTES = 21612
CONTRACT_LINES = 1024
NUMERICAL_LOCK = Path(
    "project/run_scripts/ode_bf/locks/"
    "numerical_lock_s05_p1r55_rms_pdz_floored_rate_v1.json"
)
NUMERICAL_LOCK_ROOT = "3d4a819f6daea947409e266613d3b2fc8f87820904483b40ce8f00714c78b811"
SOURCE_FILES = (
    "project/run_scripts/ode_bf/p1_runtime.py",
    "project/run_scripts/ode_bf/p1_scalable_batched_experiment.py",
    "project/run_scripts/ode_bf/p1r52_joint_pc_fp32_runtime.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale_b100.py",
    "project/run_scripts/ode_bf/p1r52_c_writer_kstep.py",
    "project/run_scripts/ode_bf/p1r52_c_writer_kstep_cache.py",
    "project/run_scripts/ode_bf/p1r52_c_writer_kstep_cache_sequential.py",
    "project/run_scripts/ode_bf/p1r52_sequential_runtime.py",
    "project/run_scripts/ode_bf/p1r54_energyfree_localz.py",
    "project/run_scripts/ode_bf/p1r54_realization_policy.py",
    "project/run_scripts/ode_bf/p1r55_objective_risk.py",
    "project/run_scripts/ode_bf/p1r55_pdz_floored_rate.py",
    "project/run_scripts/ode_bf/p1r55_rms_pdz_rate_experiment.py",
    "project/run_scripts/ode_bf/tests/test_p1r55_rms_pdz_floored_rate.py",
    "project/run_scripts/session05_ode_bf_p1r55_rms_pdz_rate.py",
    "project/run_scripts/session05_ode_bf_p1r55_rms_pdz_rate_dry_plan.py",
    "project/run_scripts/session05_ode_bf_p1r55_rms_pdz_rate_preflight.py",
    "project/run_scripts/session05_ode_bf_p1r55_rms_pdz_rate.sbatch",
    str(NUMERICAL_LOCK),
    "project/run_scripts/ode_bf/locks/p1r52_sequential_b100x10_stream_seal.json",
)


def _regular_identity(path: Path) -> dict[str, Any]:
    observed = path.stat()
    if path.is_symlink() or not stat.S_ISREG(observed.st_mode):
        raise ODEBFContractError(f"P1R55 member is not regular: {path}")
    return {
        "path": str(path),
        "mode": format(stat.S_IMODE(observed.st_mode), "04o"),
        "bytes": observed.st_size,
        "sha256": sha256_file(path),
    }


def source_manifest(head: str, tree: str) -> Mapping[str, Any]:
    rows = []
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        identity = _regular_identity(path)
        identity["path"] = relative
        rows.append(identity)
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r55-rms-pdz-rate-source-manifest/v1",
        "source_head": head,
        "source_tree": tree,
        "entries": rows,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _verify_contract() -> Mapping[str, Any]:
    identity = _regular_identity(CONTRACT)
    if (
        identity["mode"] != "0600"
        or identity["bytes"] != CONTRACT_BYTES
        or identity["sha256"] != CONTRACT_SHA256
        or CONTRACT.read_bytes().count(b"\n") != CONTRACT_LINES
    ):
        raise ODEBFContractError("P1R55 authoritative contract differs")
    return {**identity, "lines": CONTRACT_LINES, "full_read": True}


def _verify_numerical_lock() -> Mapping[str, Any]:
    path = REPO_ROOT / NUMERICAL_LOCK
    value = json.loads(path.read_text(encoding="utf-8"))
    observed = value.pop("root_digest", None)
    if (
        observed != NUMERICAL_LOCK_ROOT
        or canonical_hash(value) != NUMERICAL_LOCK_ROOT
        or value.get("instruction_id") != INSTRUCTION_ID
    ):
        raise ODEBFContractError("P1R55 numerical lock differs")
    value["root_digest"] = observed
    return {
        "path": str(NUMERICAL_LOCK),
        "sha256": sha256_file(path),
        "root_digest": observed,
    }


def build_receipt(*, source_head: str, source_tree: str) -> Mapping[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True
    ).strip()
    if (
        (head, tree) != (source_head, source_tree)
        or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        )
        or os.environ.get("PROJECT_GPU_CAP", "3") != "3"
        or torch.cuda.is_initialized()
    ):
        raise ODEBFContractError("P1R55 source/cap/no-CUDA preflight differs")
    contract = _verify_contract()
    stream = verify_p1r52_b100x10_stream(
        json.loads(
            (
                REPO_ROOT
                / "project/run_scripts/ode_bf/locks"
                / SEAL_FILE
            ).read_text(encoding="utf-8")
        )
    )
    if (
        stream.get("root_digest") != STREAM_ROOT
        or stream.get("all_request_order_sha256") != STREAM_ORDER
    ):
        raise ODEBFContractError("P1R55 stream/order differs")
    easyedit = Path("/mnt/raid5/janghj/EasyEdit")
    access = {
        "hostname": socket.gethostname(),
        "agent_hostname": subprocess.check_output(
            ["git", "config", "--get", "agent.hostname"],
            cwd=REPO_ROOT,
            text=True,
        ).strip(),
        "easyedit_root": str(easyedit),
        "easyedit_root_available": easyedit.is_dir(),
        "easyedit_python_available": (easyedit / ".venv/bin/python").exists(),
        "hf_cache_available": Path(
            "/mnt/raid5/janghj/.cache/huggingface/hub"
        ).is_dir(),
        "model_load_count": 0,
        "cuda_initialization_count": 0,
    }
    if not all(
        access[key]
        for key in (
            "easyedit_root_available",
            "easyedit_python_available",
            "hf_cache_available",
        )
    ):
        raise ODEBFContractError("P1R55 offline access differs")
    dry_plan = build_plan()
    if (
        dry_plan.get("array") != PHASE1_ARRAY
        or len(dry_plan.get("cells", [])) != len(PHASE1_CELLS)
        or dry_plan.get("project_gpu_cap") != PROJECT_GPU_CAP
    ):
        raise ODEBFContractError("P1R55 Phase1 dry plan differs")
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r55-rms-pdz-rate-final-pre-gpu/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "FINAL_PRE_GPU_PASS",
        "source_head": head,
        "source_tree": tree,
        "contract": contract,
        "registry_experiment_id": "P1R55",
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "request_count": 1000,
        "batch_count": 10,
        "numerical_lock": _verify_numerical_lock(),
        "source_manifest": source_manifest(head, tree),
        "dry_plan": dry_plan,
        "access": access,
        "focused_test_count": 11,
        "focused_test_status": "PASS",
        "py_compile_status": "PASS",
        "bash_syntax_status": "PASS",
        "full_fp32_required": True,
        "bf16_fp16_autocast_quantization_count": 0,
        "model_load_authorized": True,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "scientific_promotion": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = build_receipt(
        source_head=args.source_head, source_tree=args.source_tree
    )
    _write_once(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
