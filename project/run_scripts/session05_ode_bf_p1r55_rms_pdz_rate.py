#!/usr/bin/env python3
"""Fail-closed launcher for one P1R55 Phase-1 warm-state cell."""

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
from project.run_scripts.ode_bf.p1r55_rms_pdz_rate_experiment import (
    INSTRUCTION_ID,
    expected_phase1_result_name,
    phase1_cell,
)
from project.run_scripts.session05_ode_bf_p1r55_rms_pdz_rate_preflight import (
    PROJECT_GPU_CAP,
    source_manifest,
)


RUN_TOKEN = "p1r55-rms-pdz-floored-rate-phase1-warm-t1-v1"


def _owned_state_root(path: Path, *, expected_name: str) -> Path:
    parent = (REPO_ROOT / "local/odebf/state").resolve(strict=False)
    if (
        path.is_symlink()
        or not path.is_dir()
        or path.resolve(strict=False).parent != parent
        or path.name != expected_name
    ):
        raise ODEBFContractError("P1R55 state root differs")
    return path


def _load_preflight(
    path: Path, *, source_head: str, source_tree: str
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ODEBFContractError("P1R55 pre-GPU receipt differs")
    value = json.loads(path.read_bytes())
    identity = value.pop("identity_sha256", None)
    if (
        identity != canonical_hash(value)
        or value.get("schema")
        != "ode-edit-s05-p1r55-rms-pdz-rate-final-pre-gpu/v1"
        or value.get("instruction_id") != INSTRUCTION_ID
        or value.get("status") != "FINAL_PRE_GPU_PASS"
        or value.get("source_head") != source_head
        or value.get("source_tree") != source_tree
        or value.get("project_gpu_cap") != PROJECT_GPU_CAP
        or value.get("model_load_authorized") is not True
        or value.get("source_manifest") != source_manifest(source_head, source_tree)
    ):
        raise ODEBFContractError("P1R55 pre-GPU receipt content differs")
    value["identity_sha256"] = identity
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--cell", required=True, type=int, choices=range(12))
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--final-pre-gpu-receipt", required=True, type=Path)
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    args = parser.parse_args(argv)
    config = phase1_cell(args.cell)
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
            or args.output_root.name != expected_phase1_result_name(config.role)
        ):
            raise ODEBFContractError("P1R55 source/output identity differs")
        preflight = _load_preflight(
            args.final_pre_gpu_receipt, source_head=head, source_tree=tree
        )
        state_root = _owned_state_root(
            args.state_root, expected_name=args.output_root.name
        )
        if preflight["dry_plan"]["cells"][args.cell]["role"] != config.role:
            raise ODEBFContractError("P1R55 released cell differs")
        result = run_p1(
            repo_root=REPO_ROOT,
            alias="llama3-8b-inst",
            output_root=args.output_root,
            source_head=head,
            p1r52_sequential_role=config.role,
            p1r52_sequential_scale="b100x10",
            p1r52_batch_entry_evaluator_enabled=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timing: dict[str, object] = {
            "schema": "ode-edit-s05-p1r55-rms-pdz-rate-phase1-job-timing/v1",
            "instruction_id": INSTRUCTION_ID,
            "cell": config.cell,
            "role": config.role,
            "snapshot_batch": config.snapshot_batch,
            "arm": config.arm.label,
            "job_total_seconds": time.perf_counter() - started,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NOT_RECORDED"),
            "slurm_array_job_id": os.environ.get(
                "SLURM_ARRAY_JOB_ID", "NOT_RECORDED"
            ),
            "slurm_array_task_id": os.environ.get(
                "SLURM_ARRAY_TASK_ID", "NOT_RECORDED"
            ),
            "preflight_receipt_sha256": sha256_file(
                args.final_pre_gpu_receipt
            ),
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
            }
        )
        result["state_terminal_sha256"] = _atomic_write_once(
            state_root / "terminal.json",
            {
                "schema": "ode-edit-s05-p1r55-rms-pdz-rate-phase1-state/v1",
                "status": "TERMINAL_VALID",
                "cell": config.cell,
                "role": config.role,
                "snapshot_batch": config.snapshot_batch,
                "arm": config.arm.label,
                "source_head": head,
                "source_tree": tree,
                "result": dict(result),
            },
        )
    except P1OutputRootCollision as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED_OUTPUT_ROOT_COLLISION",
                    "cell": config.cell,
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
            failure_schema="ode-edit-s05-p1r55-rms-pdz-rate-phase1-failure/v1",
        )
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "cell": config.cell,
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
