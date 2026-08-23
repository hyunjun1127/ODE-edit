#!/usr/bin/env python3
"""Fail-closed launcher for one P1R52 target-timescale B100 cell."""

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
from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.p1_runtime import (
    P1OutputRootCollision,
    _atomic_write_once,
    run_p1,
    write_p1_failure_once,
)
from project.run_scripts.ode_bf.p1r52_target_timescale import INSTRUCTION_ID
from project.run_scripts.ode_bf.p1r52_target_timescale_b100 import (
    expected_result_name,
    role_for_cell,
)
from project.run_scripts.ode_bf.p1r52_target_timescale_deployment import (
    build_target_timescale_deployment,
)
from project.run_scripts.session05_ode_bf_p1r52_target_timescale_b100_preflight import (
    EXTRACT_ROOT,
    PRIOR_HF_GATE,
    _source_manifest,
)


RUN_TOKEN = "p1r52-target-timescale-b100-v1"


def _load_final(path: Path, *, source_head: str, source_tree: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ODEBFContractError("target-timescale final pre-GPU receipt differs")
    value = json.loads(path.read_bytes())
    identity = value.pop("identity_sha256", None)
    if (
        identity != canonical_hash(value)
        or value.get("schema") != "ode-edit-s05-p1r52-target-timescale-final-pre-gpu/v1"
        or value.get("instruction_id") != INSTRUCTION_ID
        or value.get("status") != "FINAL_PRE_GPU_PASS"
        or value.get("model_load_authorized") is not True
        or value.get("source_head") != source_head
        or value.get("source_tree") != source_tree
        or value.get("project_gpu_cap") != 2
        or value.get("source_manifest") != _source_manifest(source_head, source_tree)
    ):
        raise ODEBFContractError("target-timescale final pre-GPU content differs")
    value["identity_sha256"] = identity
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--cell", required=True, type=int, choices=range(5))
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--final-pre-gpu-receipt", required=True, type=Path)
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    args = parser.parse_args(argv)
    started = time.perf_counter()
    role = role_for_cell(args.cell)
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True).strip()
        if (
            (head, tree) != (args.source_head, args.source_tree)
            or args.output_root.resolve(strict=False).parent != (REPO_ROOT / "local/odebf/results").resolve(strict=False)
            or args.output_root.name != expected_result_name(role)
        ):
            raise ODEBFContractError("target-timescale source/output namespace differs")
        final = _load_final(args.final_pre_gpu_receipt, source_head=head, source_tree=tree)
        deployment = build_target_timescale_deployment(
            repo_root=REPO_ROOT,
            stream_extract_root=EXTRACT_ROOT,
            prior_final_pre_gpu_path=PRIOR_HF_GATE,
            dataset_identity=str(final["stream_binding"]["dataset_identity"]),
        )
        if deployment.identity_sha256 != final["deployment_identity"]:
            raise ODEBFContractError("target-timescale deployment identity differs")
        result = run_p1(
            repo_root=REPO_ROOT,
            alias="llama3-8b-inst",
            output_root=args.output_root,
            source_head=head,
            p1r52_sequential_role=role,
            p1r52_sequential_scale="b100x10",
            p1r52_batch_entry_evaluator_enabled=False,
            artifact_runtime_path_seal=deployment.runtime_path_seal,
            artifact_evaluator_source_paths=deployment.evaluator_source_paths,
            artifact_base_guard_override=deployment.base_guard,
            sealed_fp32_model_loader=deployment.base_guard.load_full_fp32,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timing: dict[str, object] = {
            "schema": "ode-edit-s05-p1r52-target-timescale-job-timing/v1",
            "instruction_id": INSTRUCTION_ID,
            "cell": args.cell,
            "role": role,
            "source_head": head,
            "final_pre_gpu_receipt_sha256": sha256_file(args.final_pre_gpu_receipt),
            "job_total_seconds": time.perf_counter() - started,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NOT_RECORDED"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "NOT_RECORDED"),
            "timing_decision_influence_count": 0,
        }
        timing["identity_sha256"] = canonical_hash(timing)
        result.update(
            {
                "source_manifest_sha256": _atomic_write_once(args.output_root / "source-manifest.json", _source_manifest(head, tree)),
                "job_timing_sha256": _atomic_write_once(args.output_root / "job-timing.json", timing),
                "deployment_identity": deployment.identity_sha256,
            }
        )
    except P1OutputRootCollision as exc:
        print(json.dumps({"status": "FAIL_CLOSED_OUTPUT_ROOT_COLLISION", "message_sha256": hashlib.sha256(str(exc).encode()).hexdigest()}), file=sys.stderr)
        return 1
    except Exception as exc:
        failure_sha, failure = write_p1_failure_once(
            args.output_root,
            exc,
            repo_root=REPO_ROOT,
            instruction_id=INSTRUCTION_ID,
            failure_schema="ode-edit-s05-p1r52-target-timescale-failure/v1",
        )
        print(json.dumps({"status": "FAIL_CLOSED", "exception_class": failure["exception_class"], "failure_sha256": failure_sha}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
