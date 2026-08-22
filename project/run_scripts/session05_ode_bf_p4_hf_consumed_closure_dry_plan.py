#!/usr/bin/env python3
"""No-model/no-submit server4 P4 deployment-closure readiness plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.alphaedit_runtime_path_seal import (
    load_alphaedit_runtime_path_seal,
)
from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p4_hf_consumed_closure import (
    EXPECTED_ALIASES,
    load_p4_hf_consumed_closure_seal,
)
from project.run_scripts.ode_bf.p4_sealed_stream import stream_readiness
from project.run_scripts.ode_bf.p4_semantic_barrier import P4_INSTRUCTION_ID
from project.run_scripts.session05_ode_bf_p4_semantic_barrier_dry_plan import (
    CONTRACT_RELATIVE,
    LOCK_RELATIVE,
    SEAL_RELATIVE,
    STREAM_RELATIVE,
    _contract_gate,
)


HF_SEAL_RELATIVE = Path("agents/server4/p4-hf-consumed-closure-seal.json")
SOURCE_MANIFEST_RELATIVE = Path(
    "project/run_scripts/ode_bf/locks/"
    "source_manifest_s05_p4_hf_consumed_closure.json"
)


def _source_gate(path: Path, *, repository_root: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    observed_root = value.pop("root_digest", None)
    expected_root = canonical_hash(value)
    if (
        observed_root != expected_root
        or value.get("instruction_id") != P4_INSTRUCTION_ID
        or not isinstance(value.get("entries"), list)
    ):
        raise ValueError("P4 HF source manifest identity differs")
    for entry in value["entries"]:
        candidate = repository_root / entry["path"]
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.stat().st_size != entry["size"]
            or hashlib.sha256(candidate.read_bytes()).hexdigest() != entry["sha256"]
        ):
            raise ValueError("P4 HF source manifest member differs")
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
    if observed_lock_root != canonical_hash(lock):
        raise ValueError("P4 numerical lock identity differs")
    source_manifest = _source_gate(
        root / SOURCE_MANIFEST_RELATIVE, repository_root=root
    )

    alpha_seal = load_alphaedit_runtime_path_seal(
        root / SEAL_RELATIVE, repo_root=root
    )
    hf_seal = load_p4_hf_consumed_closure_seal(
        root / HF_SEAL_RELATIVE, repo_root=root
    )
    hf_receipts: dict[str, object] = {}
    for alias in EXPECTED_ALIASES:
        model = hf_seal.model(alias)
        receipt = hf_seal.preflight_alias(
            alias,
            requested_paths=[
                member.relative_path for member in model.required_members
            ],
            requested_snapshot=Path(model.snapshot_path),
        )
        hf_receipts[alias] = receipt.to_dict()

    stream = stream_readiness(root / STREAM_RELATIVE)
    blockers: list[str] = []
    if not stream.present:
        blockers.extend(
            ["BLOCKED_SEALED_STREAM_TRANSFER", "BLOCKED_EVALUATOR_PACKAGE"]
        )
    else:
        blockers.append("BLOCKED_STREAM_EXPECTED_IDENTITY_FROM_GH")
    if os.environ.get("PROJECT_GPU_CAP", "2") != "2":
        blockers.append("BLOCKED_GPU_CAP_IDENTITY")
    return {
        "schema": "ode-edit-s05-p4-server4-consumed-closure-dry-plan/v1",
        "instruction_id": P4_INSTRUCTION_ID,
        "source_head": head,
        "authoritative_contract": contract,
        "numerical_lock": {
            "path": str(lock_path),
            "root_digest": observed_lock_root,
        },
        "source_manifest": source_manifest,
        "server4_deployment": {
            "alphaedit_path_seal": {
                "seal_id": alpha_seal.seal_id,
                "seal_sha256": alpha_seal.seal_sha256,
                "easyedit_mapping": alpha_seal.has_mapping("easyedit_root"),
                "generic_hf_mapping_unchanged": alpha_seal.has_mapping(
                    "hf_hub_cache"
                ),
            },
            "p4_hf_consumed_closure": {
                "status": "READY_PINNED_CONSUMED_CLOSURE",
                "seal_id": hf_seal.seal_id,
                "seal_path": hf_seal.source_path,
                "seal_sha256": hf_seal.seal_sha256,
                "seal_root_digest": hf_seal.root_digest,
                "receipts": hf_receipts,
                "offline_local_only": True,
                "exact_snapshot_only": True,
                "extra_influence_count": 0,
                "model_loaded": False,
            },
        },
        "sealed_stream": {
            "path": stream.path,
            "status": stream.status,
            "present": stream.present,
            "full_read": stream.full_read,
            "blocker": stream.blocker,
        },
        "blockers": blockers,
        "phase0_pre_gpu": "PASS" if not blockers else "BLOCKED_READINESS",
        "hf_readiness": "PASS",
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
