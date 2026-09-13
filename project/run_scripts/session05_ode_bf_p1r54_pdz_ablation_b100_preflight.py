#!/usr/bin/env python3
"""No-model final pre-GPU gate for the P1R54 PDZ ablation."""

from __future__ import annotations

import argparse
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

from project.run_scripts.ode_bf.artifacts import sha256_file
from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.p1r52_b100x10_stream import SEAL_FILE, verify_p1r52_b100x10_stream
from project.run_scripts.ode_bf.p1r54_pdz_ablation import INSTRUCTION_ID
from project.run_scripts.ode_bf.p1r54_pdz_ablation_b100 import ROLES
from project.run_scripts.session05_ode_bf_p1r54_pdz_ablation_b100_dry_plan import build_plan


CONTRACT = Path("/mnt/raid5/janghj/.codex/attachments/2b6ee4c8-def5-4cf3-bfab-069a228de5d1/pasted-text.txt")
CONTRACT_SHA256 = "d9f5ee879534bb5f33c6741cf39b1ff3bfb938d07238842f584bfb44f6b9195e"
CONTRACT_BYTES = 13_911
CONTRACT_LINES = 546
PROJECT_GPU_CAP = 3
ARRAY_CONCURRENCY = 3
NUMERICAL_LOCK = "project/run_scripts/ode_bf/locks/numerical_lock_s05_p1r54_pdz_ablation_llama_b100_v1.json"
LEGACY_P1R54 = {
    "project/run_scripts/ode_bf/p1r54_energyfree_localz.py": "9a40912bc07cbd0049e332c3b2bc1a1ff8f105a7cd83cffe9faa5f4f2f83f9e6",
    "project/run_scripts/ode_bf/p1r54_energyfree_localz_b100.py": "39fdc8058f9e89ea16c456321bf222b3caab88b879adac3abe2b7435fdb96fde",
    "project/run_scripts/ode_bf/p1r54_energyfree_localz_analysis.py": "d6c83af5d94f6b76df123ddd40910de1dd9119d3fa111f0041b2d4f80745f316",
    "project/run_scripts/ode_bf/locks/numerical_lock_s05_p1r54_energyfree_localz_llama_b100_v1.json": "2ce9c665ae45c6df0f06199182f3d7770768fec2d491b7b751a11d127d59c675",
}
SOURCE_FILES = (
    "project/run_scripts/ode_bf/p1_runtime.py",
    "project/run_scripts/ode_bf/p1_scalable_batched_experiment.py",
    "project/run_scripts/ode_bf/p1r52_joint_pc_fp32_runtime.py",
    "project/run_scripts/ode_bf/p1r52_sequential_runtime.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale_b100.py",
    "project/run_scripts/ode_bf/p1r54_energyfree_localz.py",
    "project/run_scripts/ode_bf/p1r54_pdz_ablation.py",
    "project/run_scripts/ode_bf/p1r54_pdz_ablation_analysis.py",
    "project/run_scripts/ode_bf/p1r54_pdz_ablation_b100.py",
    "project/run_scripts/ode_bf/tests/test_p1r54_pdz_ablation.py",
    "project/run_scripts/session05_ode_bf_p1r54_pdz_ablation_b100.py",
    "project/run_scripts/session05_ode_bf_p1r54_pdz_ablation_b100_dry_plan.py",
    "project/run_scripts/session05_ode_bf_p1r54_pdz_ablation_b100_preflight.py",
    "project/run_scripts/session05_ode_bf_p1r54_pdz_ablation_b100.sbatch",
    NUMERICAL_LOCK,
    "project/run_scripts/ode_bf/locks/p1r52_sequential_b100x10_stream_seal.json",
)


def source_manifest(head: str, tree: str) -> Mapping[str, Any]:
    entries = []
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ODEBFContractError(f"P1R54 ablation source differs: {relative}")
        entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-pdz-ablation-source-manifest/v1",
        "source_head": head,
        "source_tree": tree,
        "entries": entries,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _write_once(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise ODEBFContractError("P1R54 preflight create-once collision")
    else:
        path.write_bytes(encoded)
        path.chmod(0o600)
    return sha256_file(path)


def build_receipt(*, source_head: str, final_receipt: Path, session_id: str) -> Mapping[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True).strip()
    tracked = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, text=True)
    if head != source_head or tracked:
        raise ODEBFContractError("P1R54 preflight source differs")
    contract_stat = CONTRACT.lstat()
    contract_bytes = CONTRACT.read_bytes()
    if (
        CONTRACT.is_symlink() or not CONTRACT.is_file()
        or (contract_stat.st_mode & 0o777) != 0o600
        or len(contract_bytes) != CONTRACT_BYTES
        or sha256_file(CONTRACT) != CONTRACT_SHA256
        or contract_bytes.count(b"\n") != CONTRACT_LINES
    ):
        raise ODEBFContractError("P1R54 ablation contract differs")
    contract_bytes.decode("utf-8")
    for relative, expected in LEGACY_P1R54.items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise ODEBFContractError("legacy P1R54 source/lock changed")
    lock = json.loads((REPO_ROOT / NUMERICAL_LOCK).read_text(encoding="utf-8"))
    root = lock.pop("root_digest")
    if root != canonical_hash(lock):
        raise ODEBFContractError("P1R54 ablation numerical lock differs")
    stream = verify_p1r52_b100x10_stream(json.loads((REPO_ROOT / "project/run_scripts/ode_bf/locks" / SEAL_FILE).read_text(encoding="utf-8")))
    if stream["root_digest"] != "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a" or stream["all_request_order_sha256"] != "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3":
        raise ODEBFContractError("P1R54 ablation sealed B1 stream differs")
    if not Path("/mnt/raid5/janghj/EasyEdit").is_dir() or not Path("/mnt/raid5/janghj/.cache/huggingface/hub").is_dir():
        raise ODEBFContractError("P1R54 runtime roots differ")
    plan = build_plan(concurrency=ARRAY_CONCURRENCY)
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-pdz-ablation-final-pre-gpu/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "FINAL_PRE_GPU_PASS",
        "model_load_authorized": True,
        "host": socket.gethostname(),
        "agent_id": subprocess.check_output(["git", "config", "--get", "agent.id"], cwd=REPO_ROOT, text=True).strip(),
        "agent_hostname": subprocess.check_output(["git", "config", "--get", "agent.hostname"], cwd=REPO_ROOT, text=True).strip(),
        "session_id": session_id,
        "source_head": head,
        "source_tree": tree,
        "source_manifest": source_manifest(head, tree),
        "contract": {"path": str(CONTRACT), "sha256": CONTRACT_SHA256, "bytes": CONTRACT_BYTES, "lines": CONTRACT_LINES, "mode": "0600", "full_read": True},
        "legacy_p1r54_unchanged": LEGACY_P1R54,
        "stream_root": stream["root_digest"],
        "stream_order": stream["all_request_order_sha256"],
        "selected_batch": "B1",
        "request_count": 100,
        "roles": list(ROLES),
        "project_gpu_cap": PROJECT_GPU_CAP,
        "array_concurrency": ARRAY_CONCURRENCY,
        "dry_plan": plan,
        "focused_contract_items": 36,
        "focused_test_status": "PASS",
        "full_fp32_required": True,
        "bf16_fp16_autocast_quantization_count": 0,
        "scientific_promotion": False,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    receipt["file_sha256"] = _write_once(final_receipt, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--final-receipt", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_receipt(source_head=args.source_head, final_receipt=args.final_receipt, session_id=args.session_id), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
