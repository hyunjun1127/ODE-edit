#!/usr/bin/env python3
"""Checkpoint/package-bound held submitter for P1R15 Stage-A Llama."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts import session05_ode_bf_perrequest_simplex_transport_dry_plan as dry
from project.run_scripts import session05_ode_bf_perrequest_simplex_transport_package as package
from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.p1_perrequest_target_simplex_transport_panel import (
    P1R15_STAGE_A_RESULT_TOKEN,
    expected_p1r15_stage_a_result_name,
    validate_p1r15_stage_a_output_root,
)
from project.run_scripts.ode_bf.resource import gpu_count_from_tres


SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_perrequest_simplex_transport.sbatch"
STATE_ROOT = REPO_ROOT / "local/odebf/state"
LOG_ROOT = REPO_ROOT / "local/odebf/logs"
SUBMISSION_NAMESPACE = "s05-p1r15-a1-stage-a-llama-v1"
JOB_NAME = "odeedit_s05_p1r15_stage_a_llama"
PROJECT_JOB_PREFIXES = ("odeedit_", "odebf_", "odealloc_")


def _run(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=REPO_ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _write_once(path: Path, value: Mapping[str, Any]) -> str:
    data = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(data).hexdigest()


def _load_ack(
    path: Path, *, source_head: str, local_package: Mapping[str, Any]
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ODEBFContractError("P1R15 SH2 ACK path differs")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    payload = dict(value)
    root = payload.pop("root_digest", None)
    if root != canonical_hash(payload):
        raise ODEBFContractError("P1R15 SH2 ACK root differs")
    expected = {
        "status": "PACKAGE_VERIFICATION_PASS",
        "package_id": package.PACKAGE_ID,
        "source_head": source_head,
        "source_tree": local_package["source_tree"],
        "archive_sha256": local_package["archive_sha256"],
        "manifest_sha256": local_package["manifest_sha256"],
        "receipt_sha256": local_package["receipt_sha256"],
        "manifest_root": local_package["manifest_root"],
        "receipt_root": local_package["receipt_root"],
        "normalized_tree_digest": local_package["normalized_tree_digest"],
        "model_gpu_slurm_result_root_action_count": 0,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ODEBFContractError("P1R15 SH2 ACK identity differs")
    value["root_digest"] = root
    value["file_sha256"] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return value


def _scheduler_gpu_count() -> tuple[int, list[dict[str, Any]]]:
    user = os.environ.get("USER")
    if not user:
        raise ODEBFContractError("P1R15 scheduler user differs")
    output = _run(("squeue", "-h", "-u", user, "-t", "R,PD", "-o", "%i|%j|%T|%b")).stdout
    total = 0
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("|", 3)
        if len(fields) != 4:
            raise ODEBFContractError("P1R15 scheduler row differs")
        job_id, name, state, tres = fields
        gpu = gpu_count_from_tres(tres)
        project = name.startswith(PROJECT_JOB_PREFIXES)
        if project and (gpu is None or gpu <= 0):
            raise ODEBFContractError("P1R15 project GPU TRES differs")
        if project:
            total += int(gpu)
        rows.append({"job_id": job_id, "job_name": name, "state": state, "gpu": gpu or 0})
    return total, rows


def submit(
    *, source_head: str, package_directory: Path, sh2_ack: Path
) -> dict[str, Any]:
    head = _run(("git", "rev-parse", "HEAD")).stdout.strip()
    parent = _run(("git", "rev-parse", "HEAD^")).stdout.strip()
    branch = _run(("git", "branch", "--show-current")).stdout.strip()
    dirty = _run(("git", "status", "--porcelain", "--untracked-files=no")).stdout
    if (
        head != source_head
        or parent != dry.EXECUTION_PARENT
        or branch != dry.EXECUTION_BRANCH
        or dirty
    ):
        raise ODEBFContractError("P1R15 submit source chain differs")
    local_package = package.verify_local_package(package_directory, source_head)
    ack = _load_ack(sh2_ack, source_head=source_head, local_package=local_package)
    plan = dry.build_plan(source_head)
    active, scheduler = _scheduler_gpu_count()
    if active + 1 > dry.SERVER1_PROJECT_GPU_CAP:
        raise ODEBFContractError("P1R15 project GPU cap differs")
    result = REPO_ROOT / "local" / "odebf" / "results" / expected_p1r15_stage_a_result_name(dry.LLAMA_ALIAS)
    namespace = validate_p1r15_stage_a_output_root(
        repo_root=REPO_ROOT, alias=dry.LLAMA_ALIAS, output_root=result
    )
    intent_path = STATE_ROOT / f"{SUBMISSION_NAMESPACE}.intent.json"
    receipt_path = STATE_ROOT / f"{SUBMISSION_NAMESPACE}.submission-receipt.json"
    if (
        result.exists()
        or result.is_symlink()
        or intent_path.exists()
        or receipt_path.exists()
        or list(LOG_ROOT.glob(f"{JOB_NAME}-*"))
    ):
        raise ODEBFContractError("P1R15 submit namespace collides")
    intent = {
        "schema": "ode-edit-s05-p1r15-stage-a-submit-intent/v1",
        "source_head": source_head,
        "source_parent": parent,
        "source_tree": local_package["source_tree"],
        "branch": branch,
        "method_id": plan["method_id"],
        "stage_a_seal_root": plan["stage_a_seal_root"],
        "numerical_lock_root": plan["numerical_lock_root"],
        "package": dict(local_package),
        "sh2_ack_sha256": ack["file_sha256"],
        "alias": dry.LLAMA_ALIAS,
        "job_name": JOB_NAME,
        "result_namespace": namespace,
        "resources": {"gpu": 1, "cpu": 8, "memory_mib": 65_000, "time": "23:59:00"},
        "project_gpu_before": active,
        "project_gpu_after": active + 1,
        "scheduler_snapshot": scheduler,
        "held_then_release": True,
    }
    intent["identity_sha256"] = canonical_hash(intent)
    intent_sha = _write_once(intent_path, intent)
    LOG_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    stdout = LOG_ROOT / f"{JOB_NAME}-%j.out"
    stderr = LOG_ROOT / f"{JOB_NAME}-%j.err"
    environment = (
        f"ALL,ODEBF_P1R15_MODEL_ALIAS={dry.LLAMA_ALIAS},"
        f"ODEBF_P1R15_OUTPUT_ROOT={result},ODEBF_P1R15_SOURCE_HEAD={source_head},"
        "HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1"
    )
    job_id = _run(
        (
            "sbatch",
            "--parsable",
            "--hold",
            f"--job-name={JOB_NAME}",
            f"--chdir={REPO_ROOT}",
            f"--output={stdout}",
            f"--error={stderr}",
            f"--export={environment}",
            str(SBATCH),
        )
    ).stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("P1R15 held job identity differs")
    raw = _run(("scontrol", "show", "job", "--oneliner", job_id)).stdout.strip()
    if (
        f"JobName={JOB_NAME}" not in raw
        or f"WorkDir={REPO_ROOT}" not in raw
        or "NumCPUs=8" not in raw
        or "mem=65000M" not in raw
        or "gres/gpu=1" not in raw
    ):
        _run(("scancel", job_id), check=False)
        raise ODEBFContractError("P1R15 held job inspection differs")
    receipt = {
        "schema": "ode-edit-s05-p1r15-stage-a-submission-receipt/v1",
        "job_id": job_id,
        "intent_sha256": intent_sha,
        "intent_identity_sha256": intent["identity_sha256"],
        "held_raw_scontrol_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "held_inspection_pass": True,
        "atomic_release_count": 1,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    receipt_sha = _write_once(receipt_path, receipt)
    _run(("scontrol", "release", job_id))
    return {
        "status": "P1R15_STAGE_A_LLAMA_SUBMITTED",
        "job_id": job_id,
        "intent_sha256": intent_sha,
        "submission_receipt_sha256": receipt_sha,
        "source_head": source_head,
        "alias": dry.LLAMA_ALIAS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--package-directory", required=True, type=Path)
    parser.add_argument("--sh2-ack", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            submit(
                source_head=args.source_head,
                package_directory=args.package_directory,
                sh2_ack=args.sh2_ack,
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
