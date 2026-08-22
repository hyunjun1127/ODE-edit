#!/usr/bin/env python3
"""No-model/no-submit P4 server4 readiness and execution dry plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.alphaedit_runtime_path_seal import (
    load_alphaedit_runtime_path_seal,
)
from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p4_semantic_barrier import P4_INSTRUCTION_ID


CONTRACT_RELATIVE = Path(
    "local/state/p4-target-side-semantic-barrier-v1/authoritative-contract.txt"
)
SEAL_RELATIVE = Path("agents/server4/alphaedit-runtime-path-seal.json")
LOCK_RELATIVE = Path(
    "project/run_scripts/ode_bf/locks/numerical_lock_s05_p4_target_side_semantic_barrier.json"
)
SOURCE_MANIFEST_RELATIVE = Path(
    "project/run_scripts/ode_bf/locks/source_manifest_s05_p4_target_side_semantic_barrier.json"
)
STREAM_RELATIVE = Path(
    "local/state/p4-target-side-semantic-barrier-v1/sealed-stream"
)
CONTRACT_SHA256 = "0e6b0ad5110ba8bc758a92ffa126ab094afff57aa18223c4f2d1d16de1170611"
CONTRACT_BYTES = 13744
CONTRACT_LINES = 513


def _contract_gate(path: Path) -> dict[str, object]:
    observed = path.lstat()
    raw = path.read_bytes()
    if (
        not stat.S_ISREG(observed.st_mode)
        or path.is_symlink()
        or stat.S_IMODE(observed.st_mode) != 0o600
        or len(raw) != CONTRACT_BYTES
        or raw.count(b"\n") != CONTRACT_LINES
        or hashlib.sha256(raw).hexdigest() != CONTRACT_SHA256
    ):
        raise ValueError("P4 authoritative contract identity differs")
    return {
        "path": str(path),
        "type": "regular_non_symlink",
        "mode": "0600",
        "bytes": len(raw),
        "lines": raw.count(b"\n"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "full_read": True,
    }


def _source_gate(path: Path, *, repository_root: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    observed_root = value.pop("root_digest", None)
    expected_root = canonical_hash(value)
    if (
        observed_root != expected_root
        or value.get("instruction_id") != P4_INSTRUCTION_ID
        or not isinstance(value.get("entries"), list)
    ):
        raise ValueError("P4 source manifest identity differs")
    for entry in value["entries"]:
        candidate = repository_root / entry["path"]
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.stat().st_size != entry["size"]
            or hashlib.sha256(candidate.read_bytes()).hexdigest() != entry["sha256"]
        ):
            raise ValueError("P4 source manifest member differs")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "root_digest": expected_root,
        "verified_member_count": len(value["entries"]),
    }


def build_plan(*, repository_root: Path = REPO_ROOT) -> dict[str, object]:
    root = repository_root.resolve(strict=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    contract = _contract_gate(root / CONTRACT_RELATIVE)
    lock_path = root / LOCK_RELATIVE
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    observed_lock_root = lock.pop("root_digest", None)
    expected_lock_root = canonical_hash(lock)
    if (
        observed_lock_root != expected_lock_root
        or lock.get("instruction_id") != P4_INSTRUCTION_ID
    ):
        raise ValueError("P4 numerical lock identity differs")
    lock["root_digest"] = observed_lock_root
    source_manifest = _source_gate(
        root / SOURCE_MANIFEST_RELATIVE, repository_root=root
    )
    seal = load_alphaedit_runtime_path_seal(root / SEAL_RELATIVE, repo_root=root)
    stream_root = root / STREAM_RELATIVE
    stream_present = (
        stream_root.exists()
        and stream_root.is_dir()
        and not stream_root.is_symlink()
    )
    blockers: list[str] = []
    if not seal.has_mapping("hf_hub_cache"):
        blockers.append("BLOCKED_HF_MAPPING")
    if not seal.has_mapping("alphaedit_evaluator_root"):
        blockers.append("BLOCKED_EVALUATOR_MAPPING")
    if not stream_present:
        blockers.append("BLOCKED_SEALED_STREAM_TRANSFER")
    if os.environ.get("PROJECT_GPU_CAP", "2") != "2":
        blockers.append("BLOCKED_GPU_CAP_IDENTITY")
    return {
        "schema": "ode-edit-s05-p4-server4-readiness-dry-plan/v1",
        "instruction_id": P4_INSTRUCTION_ID,
        "source_head": head,
        "authoritative_contract": contract,
        "numerical_lock": {
            "path": str(lock_path),
            "sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
            "root_digest": expected_lock_root,
        },
        "source_manifest": source_manifest,
        "server4_deployment": {
            "seal_path": str(root / SEAL_RELATIVE),
            "seal_id": seal.seal_id,
            "seal_sha256": seal.seal_sha256,
            "easyedit_mapping": seal.has_mapping("easyedit_root"),
            "hf_mapping": seal.has_mapping("hf_hub_cache"),
            "evaluator_mapping": seal.has_mapping("alphaedit_evaluator_root"),
            "llama_bundle_sha256": seal.model("llama3-8b-inst").bundle_sha256,
            "qwen_bundle_sha256": seal.model("qwen2.5-7b-inst").bundle_sha256,
        },
        "sealed_stream": {
            "path": str(stream_root),
            "present": stream_present,
            "full_read": False,
        },
        "stages": {
            "ZA": {
                "arms": ["Z+", "Z±", "Native-Z"],
                "pilot_cases_per_model": 1,
                "full_independent_slices_per_model": 10,
                "writer_count": 0,
            },
            "ZB": {
                "arms": ["A+", "A±", "Native"],
                "pilot_cases_per_model": 1,
                "full_independent_slices_per_model": 10,
                "official_alphaedit_writer": True,
                "sequential": False,
            },
        },
        "resources_per_cell": {
            "gpu": 1,
            "cpu": 8,
            "memory_mib": 65000,
            "walltime_hours": 48,
            "project_gpu_cap": 2,
        },
        "blockers": blockers,
        "phase0_pre_gpu": "PASS" if not blockers else "BLOCKED_READINESS",
        "model_load": False,
        "gpu_use": False,
        "slurm_submit": False,
        "scientific_submit": "HOLD",
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    print(json.dumps(build_plan(repository_root=args.repository_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
