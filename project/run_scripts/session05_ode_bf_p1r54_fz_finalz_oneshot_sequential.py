#!/usr/bin/env python3
"""Fail-closed launcher for P1R54 FZ final-z one-shot sequential."""

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
from project.run_scripts.ode_bf.p1r54_fz_finalz_oneshot_sequential import (
    INSTRUCTION_ID,
    RESULT_NAME,
    ROLE,
)
from project.run_scripts.session05_ode_bf_p1r54_fz_finalz_oneshot_sequential_preflight import (
    EXTRACT_ROOT,
    JOB_GPU_COUNT,
    PRIOR_HF_GATE,
    PROJECT_GPU_CAP,
    source_manifest,
)


RUN_TOKEN = "p1r54-fz-final-z-oneshot-sequential-10xb100-tech-r1-v1"


def _load_final(path: Path, *, source_head: str, source_tree: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ODEBFContractError("final-z pre-GPU receipt differs")
    value = json.loads(path.read_bytes())
    identity = value.pop("identity_sha256", None)
    if (
        identity != canonical_hash(value)
        or value.get("schema")
        != "ode-edit-s05-p1r54-fz-final-z-oneshot-final-pre-gpu/v1"
        or value.get("instruction_id") != INSTRUCTION_ID
        or value.get("status")
        != "FINAL_PRE_GPU_PASS_PENDING_SUBMISSION_AUTHORIZED"
        or value.get("model_load_authorized_when_capacity_available") is not True
        or value.get("pending_submission_authorized") is not True
        or value.get("runtime_release_condition")
        != "SERVER4_JANGHJ_RUNNING_GPU_PLUS_NEW_GPU_LE_4"
        or value.get("source_head") != source_head
        or value.get("source_tree") != source_tree
        or value.get("source_manifest") != source_manifest(source_head, source_tree)
        or value.get("project_gpu_cap") != PROJECT_GPU_CAP
        or value.get("job_gpu_count") != JOB_GPU_COUNT
        or value.get("role") != ROLE
        or value.get("result_name") != RESULT_NAME
    ):
        raise ODEBFContractError("final-z pre-GPU content differs")
    value["identity_sha256"] = identity
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--final-pre-gpu-receipt", required=True, type=Path)
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True
        ).strip()
        if (
            (head, tree) != (args.source_head, args.source_tree)
            or args.output_root.resolve(strict=False).parent
            != (REPO_ROOT / "local/odebf/results").resolve(strict=False)
            or args.output_root.name != RESULT_NAME
        ):
            raise ODEBFContractError("final-z source/output differs")
        final = _load_final(
            args.final_pre_gpu_receipt,
            source_head=head,
            source_tree=tree,
        )
        deployment = build_target_timescale_deployment(
            repo_root=REPO_ROOT,
            stream_extract_root=EXTRACT_ROOT,
            prior_final_pre_gpu_path=PRIOR_HF_GATE,
            dataset_identity=str(final["stream_binding"]["dataset_identity"]),
        )
        if deployment.identity_sha256 != final["deployment_identity"]:
            raise ODEBFContractError("final-z deployment differs")
        result = run_p1(
            repo_root=REPO_ROOT,
            alias="llama3-8b-inst",
            output_root=args.output_root,
            source_head=head,
            p1r52_sequential_role=ROLE,
            p1r52_sequential_scale="b100x10",
            p1r52_batch_entry_evaluator_enabled=False,
            artifact_runtime_path_seal=deployment.runtime_path_seal,
            artifact_evaluator_source_paths=deployment.evaluator_source_paths,
            artifact_base_guard_override=deployment.base_guard,
            sealed_fp32_model_loader=deployment.base_guard.load_full_fp32,
            runtime_gpu_capacity_validator=(
                validate_target_timescale_server4_gpu_capacity
            ),
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timing: dict[str, object] = {
            "schema": "ode-edit-s05-p1r54-fz-final-z-job-timing/v1",
            "instruction_id": INSTRUCTION_ID,
            "role": ROLE,
            "source_head": head,
            "final_pre_gpu_receipt_sha256": sha256_file(
                args.final_pre_gpu_receipt
            ),
            "job_total_seconds": time.perf_counter() - started,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NOT_RECORDED"),
            "timing_decision_influence_count": 0,
        }
        timing["identity_sha256"] = canonical_hash(timing)
        result.update(
            {
                "source_manifest_sha256": _atomic_write_once(
                    args.output_root / "source-manifest.json",
                    source_manifest(head, tree),
                ),
                "job_timing_sha256": _atomic_write_once(
                    args.output_root / "job-timing.json", timing
                ),
                "deployment_identity": deployment.identity_sha256,
            }
        )
    except P1OutputRootCollision as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED_OUTPUT_ROOT_COLLISION",
                    "message_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
                }
            ),
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        failure_sha, failure = write_p1_failure_once(
            args.output_root,
            exc,
            repo_root=REPO_ROOT,
            instruction_id=INSTRUCTION_ID,
            failure_schema="ode-edit-s05-p1r54-fz-final-z-failure/tech-r1-v1",
        )
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "exception_class": failure["exception_class"],
                    "failure_sha256": failure_sha,
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
