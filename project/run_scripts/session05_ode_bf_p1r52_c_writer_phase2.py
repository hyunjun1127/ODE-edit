#!/usr/bin/env python3
"""Fail-closed launcher for one Phase-2 C K-step cell."""

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
from project.run_scripts.ode_bf.p1_runtime import P1OutputRootCollision, _atomic_write_once, run_p1, write_p1_failure_once
from project.run_scripts.ode_bf.p1r52_b100x10_stream import SEAL_FILE, verify_p1r52_b100x10_stream
from project.run_scripts.ode_bf.p1r52_c_writer_kstep_independent import INSTRUCTION_ID, expected_result_name, role_for_cell

RUN_TOKEN = "p1r52-c-writer-phase2-independent-kstep-full-fp32-v1"
SOURCE_FILES = (
    "project/run_scripts/ode_bf/p1_runtime.py",
    "project/run_scripts/ode_bf/p1_scalable_batched_experiment.py",
    "project/run_scripts/ode_bf/p1r52_sequential_runtime.py",
    "project/run_scripts/ode_bf/p1r52_c_writer_kstep.py",
    "project/run_scripts/ode_bf/p1r52_c_writer_kstep_independent.py",
    "project/run_scripts/ode_bf/p1r52_joint_pc_fp32_runtime.py",
    "project/run_scripts/ode_bf/p1r52_joint_pc_runtime.py",
    "project/run_scripts/ode_bf/p1r52_target_official_alphaedit_writer.py",
    "project/run_scripts/session05_ode_bf_p1r52_c_writer_phase2.py",
    "project/run_scripts/session05_ode_bf_p1r52_c_writer_phase2.sbatch",
    "project/run_scripts/session05_ode_bf_p1r52_c_writer_phase2_dry_plan.py",
    "project/run_scripts/ode_bf/tests/test_p1r52_c_writer_kstep.py",
    "project/run_scripts/ode_bf/locks/p1r52_sequential_b100x10_stream_seal.json",
    "local/odebf/reports/p1r52-joint-pc-full-fp32-independent-b100x10-final-v6/rooted-analysis-receipt.json",
)


def source_manifest(head: str, tree: str) -> dict[str, object]:
    rows = [{"path": path, "bytes": (REPO_ROOT / path).stat().st_size, "sha256": sha256_file(REPO_ROOT / path)} for path in SOURCE_FILES]
    payload: dict[str, object] = {"schema": "ode-edit-s05-p1r52-c-writer-phase2-source-manifest/v1", "source_head": head, "source_tree": tree, "entries": rows}
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--cell", required=True, type=int, choices=range(3))
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    args = parser.parse_args(argv)
    started = time.perf_counter()
    role = role_for_cell(args.cell)
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True).strip()
        if (head, tree) != (args.source_head, args.source_tree) or args.output_root.name != expected_result_name(role):
            raise ValueError("Phase2 source/output identity differs")
        stream = verify_p1r52_b100x10_stream(json.loads((REPO_ROOT / "project/run_scripts/ode_bf/locks" / SEAL_FILE).read_text(encoding="utf-8")))
        result = run_p1(repo_root=REPO_ROOT, alias="llama3-8b-inst", output_root=args.output_root, source_head=head, p1r52_sequential_role=role, p1r52_sequential_scale="b100x10", p1r52_batch_entry_evaluator_enabled=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timing = {"schema": "ode-edit-s05-p1r52-c-writer-phase2-job-timing/v1", "role": role, "cell": args.cell, "job_total_seconds": time.perf_counter() - started, "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NOT_RECORDED"), "timing_decision_influence_count": 0}
        timing["identity_sha256"] = canonical_hash(timing)
        result.update({"source_manifest_sha256": _atomic_write_once(args.output_root / "source-manifest.json", source_manifest(head, tree)), "job_timing_sha256": _atomic_write_once(args.output_root / "job-timing.json", timing), "stream_root": stream["root_digest"], "stream_order": stream["all_request_order_sha256"]})
    except P1OutputRootCollision as exc:
        print(json.dumps({"status": "FAIL_CLOSED_OUTPUT_ROOT_COLLISION", "message_sha256": hashlib.sha256(str(exc).encode()).hexdigest()}), file=sys.stderr)
        return 1
    except Exception as exc:
        failure_sha, failure = write_p1_failure_once(args.output_root, exc, repo_root=REPO_ROOT, instruction_id=INSTRUCTION_ID, failure_schema="ode-edit-s05-p1r52-c-writer-phase2-failure/v1")
        print(json.dumps({"status": "FAIL_CLOSED", "exception_class": failure["exception_class"], "failure_sha256": failure_sha}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
