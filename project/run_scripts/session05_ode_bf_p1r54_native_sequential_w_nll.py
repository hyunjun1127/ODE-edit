#!/usr/bin/env python3
"""Fail-closed launcher for native sequential final-W NLL backfill."""

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
from project.run_scripts.ode_bf.p1r52_target_timescale_deployment import (
    build_target_timescale_deployment,
    validate_target_timescale_server4_gpu_capacity,
)
from project.run_scripts.ode_bf.p1r54_native_sequential_w_nll import (
    INSTRUCTION_ID,
    cell_for_index,
)
from project.run_scripts.session05_ode_bf_p1r52_target_timescale_b100_preflight import (
    EXTRACT_ROOT,
    PRIOR_HF_GATE,
)
from project.run_scripts.session05_ode_bf_p1r54_native_sequential_w_nll_preflight import (
    ARRAY_GPU_COUNT,
    PROJECT_GPU_CAP,
    source_manifest,
)


RUN_TOKEN = "p1r54-native-sequential-w-nll-backfill-v1"


def _load_final(
    path: Path, *, source_head: str, source_tree: str, result_parent: Path
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ODEBFContractError("native sequential W-NLL pre-GPU receipt differs")
    value = json.loads(path.read_bytes())
    identity = value.pop("identity_sha256", None)
    if (
        identity != canonical_hash(value)
        or value.get("schema") != "ode-edit-s05-p1r54-native-sequential-w-nll-final-pre-gpu/v1"
        or value.get("status") != "FINAL_PRE_GPU_PASS_SUBMISSION_AUTHORIZED"
        or value.get("source_head") != source_head
        or value.get("source_tree") != source_tree
        or value.get("source_manifest") != source_manifest(source_head, source_tree)
        or value.get("result_parent") != str(result_parent.resolve())
        or value.get("project_gpu_cap") != PROJECT_GPU_CAP
        or value.get("array_gpu_count") != ARRAY_GPU_COUNT
        or value.get("native_science_change_count") != 0
    ):
        raise ODEBFContractError("native sequential W-NLL pre-GPU content differs")
    value["identity_sha256"] = identity
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--result-parent", required=True, type=Path)
    parser.add_argument("--final-pre-gpu-receipt", required=True, type=Path)
    parser.add_argument("--cell", required=True, type=int, choices=(0, 1))
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    args = parser.parse_args(argv)
    started = time.perf_counter()
    binding = cell_for_index(args.cell)
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True).strip()
        if (
            (head, tree) != (args.source_head, args.source_tree)
            or args.result_parent.resolve() != args.output_root.resolve(strict=False).parent
            or args.output_root.name != binding.result_name
        ):
            raise ODEBFContractError("native sequential W-NLL source/output differs")
        final = _load_final(
            args.final_pre_gpu_receipt,
            source_head=head,
            source_tree=tree,
            result_parent=args.result_parent,
        )
        deployment = build_target_timescale_deployment(
            repo_root=REPO_ROOT,
            stream_extract_root=EXTRACT_ROOT,
            prior_final_pre_gpu_path=PRIOR_HF_GATE,
            dataset_identity=str(final["stream_binding"]["dataset_identity"]),
        )
        if deployment.identity_sha256 != final["deployment_identity"]:
            raise ODEBFContractError("native sequential W-NLL deployment differs")
        result = run_p1(
            repo_root=REPO_ROOT,
            alias="llama3-8b-inst",
            output_root=args.output_root,
            source_head=head,
            p1r52_sequential_role=binding.role,
            p1r52_sequential_scale="b100x10",
            p1r52_batch_entry_evaluator_enabled=False,
            artifact_runtime_path_seal=deployment.runtime_path_seal,
            artifact_evaluator_source_paths=deployment.evaluator_source_paths,
            artifact_base_guard_override=deployment.base_guard,
            sealed_fp32_model_loader=deployment.base_guard.load_full_fp32,
            runtime_gpu_capacity_validator=validate_target_timescale_server4_gpu_capacity,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        request_table = args.output_root / "immediate-post-final-w10-requests.json"
        if request_table.is_symlink() or not request_table.is_file():
            raise ODEBFContractError("native sequential final-W10 request table is absent")
        request_payload = json.loads(request_table.read_bytes())
        if request_payload.get("row_count") != 1000:
            raise ODEBFContractError("native sequential final-W10 request denominator differs")
        timing: dict[str, object] = {
            "schema": "ode-edit-s05-p1r54-native-sequential-w-nll-job-timing/v1",
            "instruction_id": INSTRUCTION_ID,
            "cell": args.cell,
            "label": binding.label,
            "role": binding.role,
            "source_head": head,
            "final_pre_gpu_receipt_sha256": sha256_file(args.final_pre_gpu_receipt),
            "job_total_seconds": time.perf_counter() - started,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NOT_RECORDED"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "NOT_RECORDED"),
            "timing_decision_influence_count": 0,
        }
        timing["identity_sha256"] = canonical_hash(timing)
        result.update({
            "source_manifest_sha256": _atomic_write_once(
                args.output_root / "source-manifest.json", source_manifest(head, tree)
            ),
            "job_timing_sha256": _atomic_write_once(args.output_root / "job-timing.json", timing),
            "final_w10_request_table_sha256": sha256_file(request_table),
            "deployment_identity": deployment.identity_sha256,
        })
    except P1OutputRootCollision as exc:
        print(json.dumps({
            "status": "FAIL_CLOSED_OUTPUT_ROOT_COLLISION",
            "message_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
        }), file=sys.stderr)
        return 1
    except Exception as exc:
        failure_sha, failure = write_p1_failure_once(
            args.output_root,
            exc,
            repo_root=REPO_ROOT,
            instruction_id=INSTRUCTION_ID,
            failure_schema="ode-edit-s05-p1r54-native-sequential-w-nll-failure/v1",
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
