#!/usr/bin/env python3
"""Checkpoint/package-bound, create-once Llama submitter for P1R14."""

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

from project.run_scripts import session05_ode_bf_integrated_physical_writer_dry_plan as dry
from project.run_scripts import session05_ode_bf_integrated_physical_writer_package as package
from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.p1_integrated_physical_writer_panel import (
    P1R14_RESULT_TOKEN,
    P1R14_RUN_ATTEMPT_ID,
    validate_p1r14_attempt_output_namespace,
)
from project.run_scripts.ode_bf.resource import gpu_count_from_tres


SUBMISSION_NAMESPACE = "s05-integrated-physical-writer-p1r14-a3-tech-r4-llama-v1"
SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_integrated_physical_writer.sbatch"
DEFAULT_STATE_ROOT = REPO_ROOT / "local/odebf/state"
HOST_MEMORY_REQUEST_MIB = 65_000
GPU_PEAK_FORECAST_MIB = 30_234
PROJECT_JOB_PREFIXES = ("odeedit_", "odebf_", "odealloc_")


def _run(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args), cwd=REPO_ROOT, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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


def _load_sh2_package_ack(
    ack_path: Path,
    *,
    source_head: str,
    local_package: Mapping[str, Any],
) -> dict[str, Any]:
    path = ack_path.expanduser().resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ODEBFContractError("integrated SH2 package ACK path differs")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ODEBFContractError("integrated SH2 package ACK JSON differs") from exc
    if not isinstance(value, dict):
        raise ODEBFContractError("integrated SH2 package ACK mapping differs")
    payload = dict(value)
    root = payload.pop("root_digest", None)
    if root != canonical_hash(payload):
        raise ODEBFContractError("integrated SH2 package ACK root differs")
    expected = {
        "schema": "ode-edit-s05-integrated-physical-writer-p1r14-sh2-package-ack/v1",
        "status": "PACKAGE_VERIFICATION_PASS",
        "package_id": package.PACKAGE_ID,
        "source_head": source_head,
        "source_tree": local_package["source_tree"],
        "receiver_session": package.SH2_RECIPIENT_SESSION,
        "archive_sha256": local_package["archive_sha256"],
        "manifest_sha256": local_package["manifest_sha256"],
        "receipt_sha256": local_package["receipt_sha256"],
        "manifest_root": local_package["manifest_root"],
        "receipt_root": local_package["receipt_root"],
        "normalized_tree_digest": local_package["normalized_tree_digest"],
        "run_attempt_id": local_package["run_attempt_id"],
        "run_attempt_namespace_root": local_package[
            "run_attempt_namespace_root"
        ],
        "model_gpu_slurm_result_root_action_count": 0,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ODEBFContractError("integrated SH2 package ACK identity differs")
    value["root_digest"] = root
    value["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return value


def _assert_execution_identity(
    *,
    source_head: str,
    head: str,
    parent: str,
    branch: str,
    tracked_dirty: str,
) -> None:
    """Fail closed on any launcher checkout or exact-chain mismatch."""

    if (
        head != source_head
        or parent != dry.EXECUTION_PARENT
        or branch != dry.EXECUTION_BRANCH
        or tracked_dirty
    ):
        raise ODEBFContractError("integrated checkpoint/package provenance differs")


def _provenance(
    source_head: str,
    *,
    package_directory: Path,
    sh2_ack_path: Path,
) -> dict[str, Any]:
    local_package = package.verify_local_package(package_directory, source_head)
    package_ack = _load_sh2_package_ack(
        sh2_ack_path,
        source_head=source_head,
        local_package=local_package,
    )
    head = _run(("git", "rev-parse", "HEAD")).stdout.strip()
    parent = _run(("git", "rev-parse", "HEAD^")).stdout.strip()
    branch = _run(("git", "branch", "--show-current")).stdout.strip()
    dirty = _run(("git", "status", "--porcelain", "--untracked-files=no")).stdout
    _assert_execution_identity(
        source_head=source_head,
        head=head,
        parent=parent,
        branch=branch,
        tracked_dirty=dirty,
    )
    return {
        "run_authority": "GH_TECH_R4_CHECKPOINT_BOUND_RUN_APPROVAL_REQUIRED",
        "local_package": dict(local_package),
        "sh2_package_acceptance": package_ack,
        "execution_head": head,
        "exact_parent": parent,
        "branch": branch,
        "tracked_and_index_clean": True,
    }


def _scheduler() -> list[dict[str, Any]]:
    user = os.environ.get("USER")
    if not user:
        raise ODEBFContractError("integrated scheduler user identity unavailable")
    output = _run(("squeue", "-h", "-u", user, "-t", "R,PD", "-o", "%i|%j|%T|%b")).stdout
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("|", 3)
        if len(fields) != 4:
            raise ODEBFContractError("integrated scheduler row differs")
        job_id, name, state, tres = fields
        gpu = gpu_count_from_tres(tres)
        project = name.startswith(PROJECT_JOB_PREFIXES)
        if project and (gpu is None or gpu <= 0):
            raise ODEBFContractError("integrated project GPU TRES is ambiguous")
        rows.append(
            {
                "job_id": job_id,
                "job_name": name,
                "state": state,
                "gpu": 0 if gpu is None else gpu,
                "project_job": project,
            }
        )
    return rows


def _host_and_gpu_inventory() -> dict[str, Any]:
    memory_rows: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == "kB" and parts[1].isdigit():
            memory_rows[parts[0].rstrip(":")] = int(parts[1]) // 1024
    if "MemTotal" not in memory_rows or "MemAvailable" not in memory_rows:
        raise ODEBFContractError("integrated host memory identity unavailable")
    host_mib = memory_rows["MemTotal"]
    host_available_mib = memory_rows["MemAvailable"]
    if host_mib < HOST_MEMORY_REQUEST_MIB or host_available_mib < HOST_MEMORY_REQUEST_MIB:
        raise ODEBFContractError("integrated host memory capacity is insufficient")
    inventory = _run(
        (
            "nvidia-smi", "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        )
    ).stdout.splitlines()
    parsed: list[list[int]] = []
    for row in inventory:
        fields = [item.strip() for item in row.split(",")]
        if len(fields) != 2 or any(not item.isdigit() for item in fields):
            raise ODEBFContractError("integrated physical GPU inventory differs")
        parsed.append([int(fields[0]), int(fields[1])])
    if (
        not parsed
        or max(item[0] for item in parsed) < GPU_PEAK_FORECAST_MIB
        or max(item[1] for item in parsed) < GPU_PEAK_FORECAST_MIB
    ):
        raise ODEBFContractError("integrated physical GPU capacity is insufficient")
    return {
        "host_total_mib": host_mib,
        "host_available_mib": host_available_mib,
        "host_memory_request_mib": HOST_MEMORY_REQUEST_MIB,
        "physical_gpu_count": len(parsed),
        "maximum_gpu_total_mib": max(item[0] for item in parsed),
        "maximum_gpu_free_mib": max(item[1] for item in parsed),
        "gpu_peak_forecast_mib": GPU_PEAK_FORECAST_MIB,
        "host_request_fits": True,
        "gpu_forecast_fits": True,
    }


def _namespace_gate(state_root: Path) -> dict[str, str]:
    expected_state = DEFAULT_STATE_ROOT.resolve()
    observed_state = state_root.expanduser().resolve()
    if observed_state != expected_state:
        raise ODEBFContractError("integrated state root differs")
    for parent in (
        REPO_ROOT / "local/odebf/results",
        REPO_ROOT / "local/odebf/logs",
        observed_state,
    ):
        if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
            raise ODEBFContractError("integrated namespace parent differs")
    result = REPO_ROOT / "local/odebf/results" / dry.RESULT_NAME
    namespace = validate_p1r14_attempt_output_namespace(
        repo_root=REPO_ROOT,
        alias=dry.LLAMA_ALIAS,
        output_root=result,
        run_attempt_id=P1R14_RUN_ATTEMPT_ID,
    )
    intent = observed_state / f"{SUBMISSION_NAMESPACE}.intent.json"
    receipt = observed_state / f"{SUBMISSION_NAMESPACE}.submission-receipt.json"
    logs = REPO_ROOT / "local/odebf/logs"
    if (
        result.exists() or result.is_symlink()
        or intent.exists() or intent.is_symlink()
        or receipt.exists() or receipt.is_symlink()
        or (logs.exists() and list(logs.glob(f"{dry.JOB_NAME}-*")))
    ):
        raise ODEBFContractError("integrated create-once namespace collides")
    return {
        "result_root": namespace["result_root"],
        "run_attempt_id": P1R14_RUN_ATTEMPT_ID,
        "namespace_identity_sha256": namespace["identity_sha256"],
        "state_root": str(observed_state),
        "log_root": str(logs),
    }


def _assert_package_plan_binding(
    local_package: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    """Fail closed unless the accepted portable package is the dry-plan source."""

    if (
        local_package.get("fresh_seal_root") != plan.get("fresh_seal_root")
        or local_package.get("historical_exclusion_root")
        != plan.get("historical_exclusion_root")
        or local_package.get("source_manifest_root")
        != plan.get("source_manifest_root")
        or local_package.get("source_manifest_sha256")
        != plan.get("source_manifest_sha256")
        or local_package.get("source_closure_sha256")
        != plan.get("source_closure_sha256")
        or local_package.get("artifact_identities") != plan.get("artifacts")
        or local_package.get("numerical_lock_identities")
        != plan.get("numerical_locks")
        or local_package.get("run_attempt_id") != plan.get("run_attempt_id")
        or local_package.get("run_attempt_namespaces")
        != plan.get("run_attempt_namespaces")
    ):
        raise ODEBFContractError("integrated package/dry-plan binding differs")


def _preflight(
    source_head: str,
    state_root: Path,
    *,
    package_directory: Path,
    sh2_ack_path: Path,
) -> dict[str, Any]:
    provenance = _provenance(
        source_head,
        package_directory=package_directory,
        sh2_ack_path=sh2_ack_path,
    )
    _run(("scripts/check-session-boundary.sh", dry.SESSION_ID))
    plan = dry.build_plan(source_head)
    local_package = provenance["local_package"]
    _assert_package_plan_binding(local_package, plan)
    server1 = plan.get("server1_submission")
    if (
        not isinstance(server1, Mapping)
        or server1.get("alias") != dry.LLAMA_ALIAS
        or server1.get("gpu") != 1
        or server1.get("cpu") != 8
        or server1.get("memory_mib") != 65_000
        or server1.get("time") != "23:59:00"
    ):
        raise ODEBFContractError("integrated dry-plan allocation differs")
    scheduler = _scheduler()
    active = sum(int(row["gpu"]) for row in scheduler if row["project_job"])
    if active + 1 > dry.SERVER1_PROJECT_GPU_CAP:
        raise ODEBFContractError("integrated server1 project GPU cap would be exceeded")
    if any(row["job_name"] == dry.JOB_NAME for row in scheduler):
        raise ODEBFContractError("integrated scheduler namespace collides")
    return {
        "provenance": provenance,
        "session_boundary": "PASS",
        "plan": plan,
        "scheduler": scheduler,
        "active_project_gpu": active,
        "new_gpu": 1,
        "gpu_cap": dry.SERVER1_PROJECT_GPU_CAP,
        "physical_inventory": _host_and_gpu_inventory(),
        "namespaces": _namespace_gate(state_root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--package-directory", required=True, type=Path)
    parser.add_argument("--sh2-ack", required=True, type=Path)
    args = parser.parse_args()
    preflight = _preflight(
        args.source_head,
        args.state_root,
        package_directory=args.package_directory,
        sh2_ack_path=args.sh2_ack,
    )
    state_root = args.state_root.resolve()
    intent_sha = _write_once(
        state_root / f"{SUBMISSION_NAMESPACE}.intent.json",
        {
            "schema": "ode-edit-s05-integrated-physical-writer-p1r14-submit-intent/v1",
            "source_head": args.source_head,
            "preflight": preflight,
            "llama_only_on_server1": True,
            "qwen_owner": "SH2",
            "retry_or_resubmit_authorized": False,
        },
    )
    logs = REPO_ROOT / "local/odebf/logs"
    logs.mkdir(mode=0o700, parents=True, exist_ok=True)
    result_root = REPO_ROOT / "local/odebf/results" / dry.RESULT_NAME
    job_id: str | None = None
    try:
        result = _run(
            (
                "sbatch", "--parsable", "--hold",
                f"--job-name={dry.JOB_NAME}",
                f"--output={logs / (dry.JOB_NAME + '-%j.out')}",
                f"--error={logs / (dry.JOB_NAME + '-%j.err')}",
                f"--chdir={REPO_ROOT}", str(SBATCH), dry.LLAMA_ALIAS,
                str(result_root), args.source_head, P1R14_RESULT_TOKEN,
                P1R14_RUN_ATTEMPT_ID,
            )
        )
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise ODEBFContractError("integrated Slurm job identity differs")
        receipt_sha = _write_once(
            state_root / f"{SUBMISSION_NAMESPACE}.submission-receipt.json",
            {
                "schema": "ode-edit-s05-integrated-physical-writer-p1r14-submission-receipt/v1",
                "source_head": args.source_head,
                "intent_sha256": intent_sha,
                "job_id": job_id,
                "job_name": dry.JOB_NAME,
                "alias": dry.LLAMA_ALIAS,
                "scheduler_hold": True,
                "atomic_release": True,
                "retry_or_resubmit_authorized": False,
            },
        )
        _run(("scontrol", "release", job_id))
    except Exception:
        if job_id is not None:
            _run(("scancel", job_id), check=False)
        raise
    print(
        json.dumps(
            {
                "status": "LLAMA_SUBMITTED",
                "source_head": args.source_head,
                "job_id": job_id,
                "intent_sha256": intent_sha,
                "submission_receipt_sha256": receipt_sha,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
