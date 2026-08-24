#!/usr/bin/env python3
"""No-model/no-CUDA server4 preflight for native sequential W-NLL backfill."""

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
    build_target_timescale_deployment,
)
from project.run_scripts.ode_bf.p1r54_native_sequential_w_nll import (
    CELLS,
    INSTRUCTION_ID,
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
from project.run_scripts.session05_ode_bf_p1r54_native_sequential_w_nll_dry_plan import (
    build_plan,
)


PROJECT_GPU_CAP = 4
ARRAY_GPU_COUNT = 2
NUMERICAL_LOCK = Path(
    "project/run_scripts/ode_bf/locks/"
    "numerical_lock_s05_p1r54_native_sequential_w_nll_v1.json"
)
PRIOR_PACKAGE = Path(
    "experiment-reports/servers/server1/2026-08-23-p1r52-c-writer-phase1/"
    "final-presentation-v5"
)
PRIOR_PACKAGE_SHA256 = {
    "analysis-manifest.json": "58ee504c70e2a3a5708a937a44f1d6cdb736240e38745e021d4eb1244ba97a70",
    "p1r52-c-writer-phase1-full-fp32-sequential-tech-r1-exhaustive-factual-ko.md": "b93391e5c344bd5635fdd1c2f6532d41e683a88d85dabb3be21e949f05f3891b",
    "rooted-analysis-receipt.json": "6d269759f9cc8b458e9589c1ca64fcd5362681ec5c6cf455fe6376cf4e0a1f66",
}
SOURCE_FILES = (
    "agents/server4/alphaedit-runtime-path-seal.json",
    "agents/server4/p4-hf-consumed-closure-seal.json",
    "project/run_scripts/alphaedit_runtime_path_seal.py",
    "project/run_scripts/ode_alloc/p0_artifacts.py",
    "project/run_scripts/ode_bf/artifacts.py",
    "project/run_scripts/ode_bf/contracts.py",
    "project/run_scripts/ode_bf/p1_runtime.py",
    "project/run_scripts/ode_bf/p1r52_b100x10_stream.py",
    "project/run_scripts/ode_bf/p1r52_c_writer_phase1_sequential.py",
    "project/run_scripts/ode_bf/p1r52_official_sequential_baselines.py",
    "project/run_scripts/ode_bf/p1r52_sequential_runtime.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale_deployment.py",
    "project/run_scripts/ode_bf/p1r54_native_sequential_w_nll.py",
    "project/run_scripts/ode_bf/scalable_batched_native.py",
    "project/run_scripts/ode_bf/tests/test_p1r54_native_sequential_w_nll.py",
    "project/run_scripts/session05_ode_bf_p1r54_native_sequential_w_nll.py",
    "project/run_scripts/session05_ode_bf_p1r54_native_sequential_w_nll_dry_plan.py",
    "project/run_scripts/session05_ode_bf_p1r54_native_sequential_w_nll_preflight.py",
    "project/run_scripts/session05_ode_bf_p1r54_native_sequential_w_nll_server4.sbatch",
    str(NUMERICAL_LOCK),
    "project/run_scripts/ode_bf/locks/p1r52_sequential_b100x10_stream_seal.json",
)


def source_manifest(head: str, tree: str) -> Mapping[str, Any]:
    entries = []
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ODEBFContractError(f"native sequential W-NLL source member differs: {relative}")
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-native-sequential-w-nll-source-manifest/v1",
        "instruction_id": INSTRUCTION_ID,
        "source_head": head,
        "source_tree": tree,
        "entries": entries,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _verify_prior_package() -> Mapping[str, Any]:
    rows = []
    for relative, expected in PRIOR_PACKAGE_SHA256.items():
        path = REPO_ROOT / PRIOR_PACKAGE / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ODEBFContractError(f"prior native sequential package differs: {relative}")
        rows.append({"path": str(path), "sha256": expected})
    return {
        "classification": "EXACT_SAME_SAMPLE_METRIC_PACKAGE_MISSING_FINAL_W_REQUEST_NLL",
        "rows": rows,
        "new_execution_selection_influence_count": 0,
    }


def build_receipt(
    *, source_head: str, final_receipt: Path, result_parent: Path, session_id: str
) -> Mapping[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True).strip()
    role = subprocess.check_output(["git", "config", "--get", "agent.role"], cwd=REPO_ROOT, text=True).strip()
    host = subprocess.check_output(["git", "config", "--get", "agent.hostname"], cwd=REPO_ROOT, text=True).strip()
    if (
        socket.gethostname() != "server4"
        or role != "server-head"
        or host != "server4"
        or head != source_head
        or subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True,
            stdout=subprocess.PIPE, check=True,
        ).stdout
        or os.environ.get("PROJECT_GPU_CAP", str(PROJECT_GPU_CAP)) != str(PROJECT_GPU_CAP)
        or result_parent.is_symlink()
        or not result_parent.is_dir()
        or any(result_parent.iterdir())
    ):
        raise ODEBFContractError("native sequential W-NLL host/source/result/cap differs")

    stream = _verify_stream()
    if (
        stream.get("stream_root") != STREAM_ROOT
        or stream.get("order_root") != STREAM_ORDER
        or stream.get("B1_request_count") != 100
        or stream.get("sample_duplication_count") != 0
    ):
        raise ODEBFContractError("native sequential W-NLL stream differs")
    lock, lock_sha = _rooted(REPO_ROOT / NUMERICAL_LOCK)
    if (
        lock.get("instruction_id") != INSTRUCTION_ID
        or lock.get("model_alias") != "llama3-8b-inst"
        or lock.get("dtype") != "FULL_FP32"
        or lock.get("stream_root") != STREAM_ROOT
        or lock.get("order_root") != STREAM_ORDER
        or lock.get("cell_count") != 2
        or lock.get("native_science_change_count") != 0
        or lock.get("required_final_w10_request_rows_per_cell") != 1000
    ):
        raise ODEBFContractError("native sequential W-NLL numerical lock differs")
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
        "schema": "ode-edit-s05-p1r54-native-sequential-w-nll-final-pre-gpu/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "FINAL_PRE_GPU_PASS_SUBMISSION_AUTHORIZED",
        "hostname": "server4",
        "agent_role": role,
        "session_id": session_id,
        "source_head": head,
        "source_tree": tree,
        "source_manifest": source_manifest(head, tree),
        "result_parent": str(result_parent.resolve()),
        "cells": [asdict(item) for item in CELLS],
        "project_gpu_cap": PROJECT_GPU_CAP,
        "array_gpu_count": ARRAY_GPU_COUNT,
        "stream_binding": stream,
        "numerical_lock_path": str(REPO_ROOT / NUMERICAL_LOCK),
        "numerical_lock_sha256": lock_sha,
        "numerical_lock_root": lock["root_digest"],
        "deployment_identity": deployment.identity_sha256,
        "artifact_receipt": asdict(artifact),
        "prior_package_audit": _verify_prior_package(),
        "source_equivalence": source_equivalence_receipt(),
        "dry_plan": build_plan(),
        "independent_baseline_rerun_count": 0,
        "fz_sequential_rerun_count": 0,
        "native_science_change_count": 0,
        "model_gpu_slurm_action_count": 0,
        "scientific_promotion": False,
    }
    final["identity_sha256"] = canonical_hash(final)
    _write_or_verify_once(final_receipt, final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--final-receipt", required=True, type=Path)
    parser.add_argument("--result-parent", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    print(json.dumps(build_receipt(
        source_head=args.source_head,
        final_receipt=args.final_receipt,
        result_parent=args.result_parent,
        session_id=args.session_id,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
