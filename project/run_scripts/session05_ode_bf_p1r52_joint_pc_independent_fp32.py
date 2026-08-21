#!/usr/bin/env python3
"""Fail-closed launcher for one independent full-FP32 Joint-P/C cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.artifacts import sha256_file
from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1_runtime import (
    P1OutputRootCollision,
    _atomic_write_once,
    run_p1,
    write_p1_failure_once,
)
from project.run_scripts.ode_bf.p1r52_b100x10_stream import (
    SEAL_FILE,
    verify_p1r52_b100x10_stream,
)
from project.run_scripts.ode_bf.p1r52_joint_pc_independent_fp32_runtime import (
    INSTRUCTION_ID,
    expected_result_name,
    role_for_cell,
)


RUN_TOKEN = "p1r52-joint-pc-full-fp32-independent-b100x10-v1"
SOURCE_FILES = (
    "project/run_scripts/ode_bf/p1_runtime.py",
    "project/run_scripts/ode_bf/p1r52_sequential_runtime.py",
    "project/run_scripts/ode_bf/p1r52_accepted_z_observation.py",
    "project/run_scripts/ode_bf/p1r52_joint_pc_fp32_runtime.py",
    "project/run_scripts/ode_bf/p1r52_joint_pc_independent_fp32_runtime.py",
    "project/run_scripts/ode_bf/p1r52_joint_pc_router.py",
    "project/run_scripts/ode_bf/p1r52_joint_pc_runtime.py",
    "project/run_scripts/ode_bf/p1r52_joint_pc_execution.py",
    "project/run_scripts/ode_bf/p1r52_residual_reserve_fp32_transaction.py",
    "project/run_scripts/ode_bf/p1r52_residual_reserve_update_binding.py",
    "project/run_scripts/ode_bf/p1r52_target_official_alphaedit_writer.py",
    "project/run_scripts/ode_bf/scalable_batched_native.py",
    "project/run_scripts/ode_bf/p1r52_official_sequential_baselines.py",
    "project/run_scripts/session05_ode_bf_p1r52_joint_pc_independent_fp32.py",
    "project/run_scripts/session05_ode_bf_p1r52_joint_pc_independent_fp32.sbatch",
    "project/run_scripts/session05_ode_bf_p1r52_joint_pc_independent_fp32_dry_plan.py",
    "project/run_scripts/session05_ode_bf_submit_p1r52_joint_pc_independent_fp32.py",
    "project/run_scripts/ode_bf/tests/test_p1r52_joint_pc_independent_fp32_runtime.py",
    "project/run_scripts/ode_bf/locks/p1r52_sequential_b100x10_stream_seal.json",
)


def source_manifest(source_head: str, source_tree: str) -> dict[str, object]:
    entries = [
        {
            "path": name,
            "bytes": (REPO_ROOT / name).stat().st_size,
            "sha256": sha256_file(REPO_ROOT / name),
        }
        for name in SOURCE_FILES
    ]
    payload: dict[str, object] = {
        "schema": "ode-edit-s05-p1r52-joint-pc-independent-fp32-source-manifest/v1",
        "instruction_id": INSTRUCTION_ID,
        "source_head": source_head,
        "source_tree": source_tree,
        "entries": entries,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--cell", required=True, type=int, choices=range(5))
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    args = parser.parse_args(argv)
    process_started = time.perf_counter()
    role = role_for_cell(args.cell)
    expected_name = expected_result_name(role)
    try:
        observed_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        observed_tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True
        ).strip()
        if observed_head != args.source_head or observed_tree != args.source_tree:
            raise ValueError("independent FP32 source identity differs")
        if args.output_root.name != expected_name:
            raise ValueError("independent FP32 output namespace differs")
        stream_path = REPO_ROOT / "project/run_scripts/ode_bf/locks" / SEAL_FILE
        stream = verify_p1r52_b100x10_stream(
            json.loads(stream_path.read_text(encoding="utf-8"))
        )
        manifest = source_manifest(args.source_head, args.source_tree)
        result = run_p1(
            repo_root=REPO_ROOT,
            alias="llama3-8b-inst",
            output_root=args.output_root,
            source_head=args.source_head,
            p1r52_sequential_role=role,
            p1r52_sequential_scale="b100x10",
            p1r52_batch_entry_evaluator_enabled=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        job_timing = {
            "schema": "ode-edit-s05-p1r52-joint-pc-independent-fp32-job-timing/v1",
            "instruction_id": INSTRUCTION_ID,
            "role": role,
            "cell": args.cell,
            "job_total_seconds": time.perf_counter() - process_started,
            "job_total_boundary": "PROCESS_ENTRY_TO_SCIENTIFIC_TERMINAL_PLUS_FINAL_CUDA_COMPLETION",
            "model_load_preflight_source": "terminal.job_compute.component_wall_seconds",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NOT_RECORDED"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", "NOT_RECORDED"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "NOT_RECORDED"),
            "timing_decision_influence_count": 0,
            "additional_model_forward_backward_count": 0,
            "additional_evaluator_count": 0,
        }
        job_timing["identity_sha256"] = canonical_hash(job_timing)
        source_manifest_sha = _atomic_write_once(
            args.output_root / "source-manifest.json", manifest
        )
        job_timing_sha = _atomic_write_once(
            args.output_root / "job-timing.json", job_timing
        )
        result.update({
            "stream_root": stream["root_digest"],
            "stream_order": stream["all_request_order_sha256"],
            "expected_result_name": expected_name,
            "source_manifest_sha256": source_manifest_sha,
            "job_timing_sha256": job_timing_sha,
        })
    except P1OutputRootCollision as exc:
        print(json.dumps({
            "status": "FAIL_CLOSED_OUTPUT_ROOT_COLLISION",
            "message_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
        }), file=sys.stderr)
        return 1
    except Exception as exc:
        failure_sha, failure = write_p1_failure_once(
            args.output_root, exc, repo_root=REPO_ROOT,
            instruction_id=INSTRUCTION_ID,
            failure_schema="ode-edit-s05-p1r52-joint-pc-independent-fp32-failure/v1",
        )
        print(json.dumps({
            "status": "FAIL_CLOSED",
            "exception_class": failure["exception_class"],
            "failure_sha256": failure_sha,
        }), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
