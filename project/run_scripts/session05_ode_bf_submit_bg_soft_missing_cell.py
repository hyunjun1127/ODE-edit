#!/usr/bin/env python3
"""Create-once, approval-gated submitter for the Llama R12 BG-SOFT cell."""

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

from project.run_scripts import session05_ode_bf_bg_soft_missing_cell_dry_plan as dry
from project.run_scripts.ode_bf.artifacts import (
    ODEBFArtifactGuard,
    load_rooted_json,
    sha256_file,
)
from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1_bg_soft_missing_cell_panel import (
    BG_SOFT_AMENDMENT_ID,
    BG_SOFT_INSTRUCTION_ID,
    BG_SOFT_PARENT_HEAD,
    BG_SOFT_R10_CASE_ROOT,
    BG_SOFT_RESULT_TOKEN,
    BG_SOFT_SCHEMA_NAMESPACE,
    BG_SOFT_SOURCE_MANIFEST_FILE,
    bg_soft_frozen_reference,
    expected_bg_soft_result_name,
    load_and_validate_bg_soft_reference_lock,
    common_cold_schedule,
)
from project.run_scripts.ode_bf.resource import gpu_count_from_tres
from project.run_scripts.ode_bf.sampling import load_p1_sampling_seal


SESSION_ID = "019fe489-c968-75f3-9965-7cfbc26c0a99"
EXECUTION_BRANCH = (
    "codex/odeeditsh1-s05-bg-soft-missing-cell-p1r12-a1-r1-package"
)
BG_SOFT_RUNTIME_TECH_REPAIR_PARENT = (
    "bfd77ddbf046333c59a8e589429d426fd5eee4c4"
)
BG_SOFT_IMPLEMENTATION_PARENT = "8d0e8c80e3a100fccd4c295c3c3dbab5153602be"
BG_SOFT_EXECUTION_REPAIR_PARENT = "1f58ea7b6732a67cbced132ee56056cd1f79cad3"
BG_SOFT_PACKAGE_REPAIR_PARENT = "4a9c5d8edabf477a709dbe44b4ebdcfee968763b"
BG_SOFT_PACKAGE_REPAIR_INSTRUCTION_ID = (
    "ODEEDIT-S05-ODE-BF-BG-SOFT-MISSING-CELL-P1R12-V1-A1-R1"
)
BG_SOFT_PACKAGE_ID = "BGSOFT_R10_FROZEN_BUNDLE_A1_R1"
LLAMA_ALIAS = "llama3-8b-inst"
SERVER1_PROJECT_GPU_CAP = 4
APPROVAL_ENV = "ODEEDIT_S05_BG_SOFT_MISSING_CELL_P1R12_RUN_APPROVAL"
SUBMISSION_NAMESPACE = "s05-bg-soft-missing-cell-p1r12-a1-v1"
SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_bg_soft_missing_cell.sbatch"
SOURCE_MANIFEST = (
    REPO_ROOT / "project/run_scripts/ode_bf/locks" / BG_SOFT_SOURCE_MANIFEST_FILE
)
DEFAULT_STATE_ROOT = REPO_ROOT / "local/odebf/state"
R10_REFERENCE_REPO = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-fixed-e8-r8-bound-snap-r1/ODE-edit"
)
R10_REFERENCE_RESULTS = R10_REFERENCE_REPO / "local/odebf/results"
R10_REFERENCE_PATHS = {
    "terminal_sha256": "terminal.json",
    "manifest_sha256": "manifest.json",
    "action_freeze_sha256": "raw/common-cold/action-freeze.json",
    "bootstrap_bg_sha256": "raw/common-cold/bootstrap-bg.json",
    "bg_neutral_field0_sha256": "raw/fixed-e8/BG-NEUTRAL/field-0000.json",
    "n32_native_sha256": "raw/common-cold/N32_NATIVE-postfreeze.json",
    "w0_stepwise_sha256": "raw/stepwise/W0_NO_EDIT.json",
    "native_stepwise_sha256": "raw/stepwise/N32_NATIVE.json",
    "panel_sha256": "raw/stepwise/common-cold-panel.json",
    "evaluator_parity_sha256": "raw/stepwise/reused-warm-evaluator-parity.json",
}


def _run(
    args: Sequence[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=REPO_ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _write_once(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _execution_provenance_gate(source_head: str) -> dict[str, Any]:
    expected_approval = f"{BG_SOFT_AMENDMENT_ID}:{source_head}"
    approval = os.environ.get(APPROVAL_ENV)
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    parent = _run(["git", "rev-parse", "HEAD^"]).stdout.strip()
    package_repair_parent = _run(
        ["git", "rev-parse", "HEAD^^"]
    ).stdout.strip()
    execution_repair_parent = _run(
        ["git", "rev-parse", "HEAD^^^"]
    ).stdout.strip()
    implementation_parent = _run(
        ["git", "rev-parse", "HEAD^^^^"]
    ).stdout.strip()
    scientific_parent = _run(
        ["git", "rev-parse", "HEAD^^^^^"]
    ).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(
        ["git", "status", "--porcelain", "--untracked-files=no"]
    ).stdout
    ancestor = _run(
        ["git", "merge-base", "--is-ancestor", BG_SOFT_PARENT_HEAD, head],
        check=False,
    )
    if approval != expected_approval:
        raise ODEBFContractError("BG-Soft checkpoint-bound approval is absent")
    if (
        head != source_head
        or parent != BG_SOFT_RUNTIME_TECH_REPAIR_PARENT
        or package_repair_parent != BG_SOFT_PACKAGE_REPAIR_PARENT
        or execution_repair_parent != BG_SOFT_EXECUTION_REPAIR_PARENT
        or implementation_parent != BG_SOFT_IMPLEMENTATION_PARENT
        or scientific_parent != BG_SOFT_PARENT_HEAD
        or branch != EXECUTION_BRANCH
        or ancestor.returncode != 0
        or dirty
    ):
        raise ODEBFContractError("BG-Soft execution provenance differs")
    return {
        "checkpoint_bound_approval": expected_approval,
        "execution_head": head,
        "exact_execution_parent": parent,
        "exact_package_repair_parent": package_repair_parent,
        "exact_execution_repair_parent": execution_repair_parent,
        "exact_implementation_parent": implementation_parent,
        "exact_scientific_parent": scientific_parent,
        "scientific_parent_is_ancestor": True,
        "branch": branch,
        "tracked_tree_clean": True,
    }


def _source_manifest_gate(source_head: str) -> str:
    value, raw_sha256 = load_rooted_json(
        SOURCE_MANIFEST,
        expected_schema=f"{BG_SOFT_SCHEMA_NAMESPACE}-source-manifest/v1",
    )
    entries = value.get("entries")
    if (
        value.get("instruction_id") != BG_SOFT_INSTRUCTION_ID
        or value.get("amendment_id") != BG_SOFT_AMENDMENT_ID
        or value.get("expected_parent") != BG_SOFT_PARENT_HEAD
        or value.get("runtime_technical_repair_parent")
        != BG_SOFT_RUNTIME_TECH_REPAIR_PARENT
        or value.get("execution_repair_parent")
        != BG_SOFT_EXECUTION_REPAIR_PARENT
        or value.get("implementation_parent") != BG_SOFT_IMPLEMENTATION_PARENT
        or value.get("package_repair_parent") != BG_SOFT_PACKAGE_REPAIR_PARENT
        or value.get("package_repair_instruction_id")
        != BG_SOFT_PACKAGE_REPAIR_INSTRUCTION_ID
        or value.get("frozen_bundle_id") != BG_SOFT_PACKAGE_ID
        or value.get("execution_branch") != EXECUTION_BRANCH
        or value.get("execution_head_policy") != "runtime-git-head"
        or not isinstance(entries, list)
        or not entries
    ):
        raise ODEBFContractError("BG-Soft source manifest header differs")
    observed: list[str] = []
    for entry in entries:
        if (
            not isinstance(entry, Mapping)
            or not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("mode"), int)
            or not isinstance(entry.get("object_id"), str)
            or len(str(entry["object_id"])) != 40
            or not isinstance(entry.get("role"), str)
            or not entry["role"]
        ):
            raise ODEBFContractError("BG-Soft source manifest entry differs")
        relative = str(entry["path"])
        path = REPO_ROOT / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry.get("size")
            or sha256_file(path) != entry.get("sha256")
        ):
            raise ODEBFContractError("BG-Soft source manifest content differs")
        tree_row = _run(
            ["git", "ls-tree", source_head, "--", relative]
        ).stdout.rstrip("\n")
        if "\t" not in tree_row:
            raise ODEBFContractError("BG-Soft source manifest object differs")
        metadata, observed_path = tree_row.split("\t", 1)
        mode, kind, object_id = metadata.split(" ", 2)
        if (
            observed_path != relative
            or kind != "blob"
            or int(mode, 8) != entry["mode"]
            or object_id != entry["object_id"]
        ):
            raise ODEBFContractError("BG-Soft source manifest object differs")
        observed.append(relative)
    if observed != sorted(observed) or len(observed) != len(set(observed)):
        raise ODEBFContractError("BG-Soft source manifest ordering differs")
    manifest_relative = SOURCE_MANIFEST.relative_to(REPO_ROOT).as_posix()
    tracked = set(
        _run(["git", "ls-tree", "-r", "--name-only", source_head]).stdout.splitlines()
    )
    expected = {
        relative
        for relative in tracked
        if (
            relative.startswith("project/run_scripts/ode_bf/")
            or relative.startswith(
                "project/run_scripts/session05_ode_bf_bg_soft_missing_cell"
            )
            or relative
            == "project/run_scripts/session05_ode_bf_submit_bg_soft_missing_cell.py"
        )
        and relative != manifest_relative
    }
    if set(observed) != expected:
        raise ODEBFContractError("BG-Soft source manifest path set differs")
    return raw_sha256


def _scheduler_snapshot() -> list[dict[str, Any]]:
    user = os.environ.get("USER")
    if not user:
        raise ODEBFContractError("BG-Soft scheduler user identity unavailable")
    output = _run(
        ["squeue", "-h", "-u", user, "-t", "R,PD", "-o", "%i|%j|%T|%b"]
    ).stdout
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("|", 3)
        if len(fields) != 4:
            raise ODEBFContractError("BG-Soft scheduler row differs")
        job_id, name, state, tres = fields
        gpu = gpu_count_from_tres(tres)
        project_job = name.startswith("odeedit_")
        if project_job and (gpu is None or gpu <= 0):
            raise ODEBFContractError("BG-Soft project GPU TRES is ambiguous")
        rows.append(
            {
                "job_id": job_id,
                "job_name": name,
                "state": state,
                "gpu": 0 if gpu is None else gpu,
                "project_job": project_job,
            }
        )
    return rows


def _host_memory_gate() -> int:
    total = next(
        (
            line
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
            if line.startswith("MemTotal:")
        ),
        None,
    )
    if total is None:
        raise ODEBFContractError("BG-Soft host memory identity unavailable")
    fields = total.split()
    if len(fields) != 3 or fields[2] != "kB" or not fields[1].isdigit():
        raise ODEBFContractError("BG-Soft host memory units differ")
    total_mib = int(fields[1]) // 1024
    if total_mib < 65_000:
        raise ODEBFContractError("BG-Soft host memory capacity is insufficient")
    return total_mib


def _safe_directory(path: Path, *, label: str) -> None:
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise ODEBFContractError(f"BG-Soft {label} parent differs")


def _state_root_gate(state_root: Path) -> Path:
    expected = DEFAULT_STATE_ROOT.resolve()
    observed = state_root.expanduser().resolve()
    if observed != expected:
        raise ODEBFContractError("BG-Soft state root differs")
    _safe_directory(observed, label="state")
    return observed


def _namespace_gates(state_root: Path) -> dict[str, str]:
    result_parent = REPO_ROOT / "local/odebf/results"
    log_parent = REPO_ROOT / "local/odebf/logs"
    _safe_directory(result_parent, label="result")
    _safe_directory(log_parent, label="log")
    result_root = result_parent / expected_bg_soft_result_name(LLAMA_ALIAS)
    if result_root.exists():
        raise ODEBFContractError("BG-Soft result root collides")
    if log_parent.exists() and list(log_parent.glob(f"{dry.JOB_NAME}-*")):
        raise ODEBFContractError("BG-Soft log namespace collides")
    for suffix in ("intent", "submission-receipt"):
        if (state_root / f"{SUBMISSION_NAMESPACE}.{suffix}.json").exists():
            raise ODEBFContractError("BG-Soft state namespace collides")
    return {
        "result_root": str(result_root),
        "log_parent": str(log_parent),
        "state_root": str(state_root),
    }


def _artifact_gate() -> None:
    guard = ODEBFArtifactGuard(
        REPO_ROOT,
        REPO_ROOT / "project/run_scripts/ode_bf/locks/p0_artifact_lock.json",
        LLAMA_ALIAS,
        require_held_ode_alloc=False,
    )
    guard.preflight()
    guard.assert_unchanged()


def _frozen_r10_reference_gate() -> dict[str, Any]:
    if (
        R10_REFERENCE_REPO.is_symlink()
        or not R10_REFERENCE_REPO.is_dir()
        or _run(
            ["git", "-C", str(R10_REFERENCE_REPO), "rev-parse", "HEAD"]
        ).stdout.strip()
        != BG_SOFT_PARENT_HEAD
    ):
        raise ODEBFContractError("BG-Soft frozen R10 source differs")
    locks = REPO_ROOT / "project/run_scripts/ode_bf/locks"
    schedule = common_cold_schedule(
        load_p1_sampling_seal(
            locks / "p1r2_p_population_seal.json",
            stream_path=locks / "p1r2_seqb10_stream_seal.json",
        )
    )
    from project.run_scripts.ode_bf.p1_controller import P1ControllerLock

    reference_lock, reference_lock_sha256 = (
        load_and_validate_bg_soft_reference_lock(
            locks / "p1r12_bg_soft_reference_lock.json",
            controller_identity_sha256=P1ControllerLock().identity(),
            case_root_digest=BG_SOFT_R10_CASE_ROOT,
            schedule=schedule,
        )
    )
    verified: dict[str, dict[str, str]] = {}
    for alias in ("llama3-8b-inst", "qwen2.5-7b-inst"):
        frozen = bg_soft_frozen_reference(reference_lock, alias)
        root = R10_REFERENCE_RESULTS / (
            "s05-common-coldcoord-fixed-e8-p1r10-r4-" + alias + "-v1"
        )
        if root.is_symlink() or not root.is_dir():
            raise ODEBFContractError("BG-Soft frozen R10 result root differs")
        alias_verified: dict[str, str] = {}
        for identity_key, relative in R10_REFERENCE_PATHS.items():
            path = root / relative
            if (
                path.is_symlink()
                or not path.is_file()
                or sha256_file(path) != frozen[identity_key]
            ):
                raise ODEBFContractError("BG-Soft frozen R10 receipt differs")
            alias_verified[identity_key] = frozen[identity_key]
        verified[alias] = alias_verified
    return {
        "reference_lock_sha256": reference_lock_sha256,
        "reference_lock_root_digest": reference_lock["root_digest"],
        "r10_source_head": BG_SOFT_PARENT_HEAD,
        "verified_receipts": verified,
    }


def _validate_plan(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 1 or not isinstance(jobs[0], Mapping):
        raise ODEBFContractError("BG-Soft dry-plan job inventory differs")
    job = jobs[0]
    if (
        job.get("alias") != LLAMA_ALIAS
        or job.get("job_name") != dry.JOB_NAME
        or job.get("run_token") != BG_SOFT_RESULT_TOKEN
        or job.get("gpu") != 1
        or job.get("cpu") != 8
        or job.get("memory_mib") != 65_000
        or job.get("time") != "23:59:00"
        or job.get("node") != "server1"
        or plan.get("server1_project_gpu_cap") != SERVER1_PROJECT_GPU_CAP
        or plan.get("new_job_gpu") != 1
        or plan.get("qwen_submission_authorized") is not False
    ):
        raise ODEBFContractError("BG-Soft dry-plan allocation differs")
    forecast = job.get("forecast")
    if not isinstance(forecast, Mapping) or forecast.get("fits_envelope") is not True:
        raise ODEBFContractError("BG-Soft resource forecast exceeds allocation")
    return job


def _pre_submit(source_head: str, *, state_root: Path) -> dict[str, Any]:
    provenance = _execution_provenance_gate(source_head)
    _run(["scripts/check-session-boundary.sh", SESSION_ID])
    manifest_sha256 = _source_manifest_gate(source_head)
    plan = dry.build_plan(source_head)
    job = _validate_plan(plan)
    _artifact_gate()
    frozen_reference = _frozen_r10_reference_gate()
    scheduler = _scheduler_snapshot()
    active_project_gpu = sum(row["gpu"] for row in scheduler if row["project_job"])
    if active_project_gpu + 1 > SERVER1_PROJECT_GPU_CAP:
        raise ODEBFContractError("BG-Soft server1 project GPU cap would be exceeded")
    if any(row["job_name"] == dry.JOB_NAME for row in scheduler):
        raise ODEBFContractError("BG-Soft scheduler name collides")
    namespaces = _namespace_gates(state_root)
    return {
        "execution_provenance": provenance,
        "session_gate": SESSION_ID,
        "access_gate": APPROVAL_ENV,
        "source_manifest_sha256": manifest_sha256,
        "artifact_gate": True,
        "frozen_r10_reference_gate": frozen_reference,
        "active_project_gpu": active_project_gpu,
        "server1_project_gpu_cap": SERVER1_PROJECT_GPU_CAP,
        "host_total_mib": _host_memory_gate(),
        "namespace_gates": namespaces,
        "job": dict(job),
        "plan": dict(plan),
    }


def _submit_one(source_head: str) -> str:
    root = REPO_ROOT / "local/odebf/results" / expected_bg_soft_result_name(LLAMA_ALIAS)
    logs = REPO_ROOT / "local/odebf/logs"
    logs.mkdir(mode=0o700, parents=True, exist_ok=True)
    result = _run(
        [
            "sbatch",
            "--parsable",
            "--hold",
            f"--job-name={dry.JOB_NAME}",
            f"--output={logs / (dry.JOB_NAME + '-%j.out')}",
            f"--error={logs / (dry.JOB_NAME + '-%j.err')}",
            f"--chdir={REPO_ROOT}",
            str(SBATCH),
            LLAMA_ALIAS,
            str(root),
            source_head,
            BG_SOFT_RESULT_TOKEN,
        ]
    )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("BG-Soft Slurm job ID differs")
    return job_id


def _cancel_held(job_id: str | None) -> None:
    if job_id is not None:
        _run(["scancel", job_id], check=False)


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    args = parser.parse_args()
    state_root = _state_root_gate(args.state_root)
    preflight = _pre_submit(args.source_head, state_root=state_root)
    intent_sha256 = _write_once(
        state_root / f"{SUBMISSION_NAMESPACE}.intent.json",
        {
            "schema": f"{BG_SOFT_SCHEMA_NAMESPACE}-submit-intent/v1",
            "instruction_id": BG_SOFT_INSTRUCTION_ID,
            "amendment_id": BG_SOFT_AMENDMENT_ID,
            "source_head": args.source_head,
            "preflight": preflight,
            "llama_only": True,
            "qwen_submission_authorized": False,
            "retry_or_resubmit_authorized": False,
        },
    )
    job_id: str | None = None
    try:
        job_id = _submit_one(args.source_head)
        receipt_sha256 = _write_once(
            state_root / f"{SUBMISSION_NAMESPACE}.submission-receipt.json",
            {
                "schema": f"{BG_SOFT_SCHEMA_NAMESPACE}-submission-receipt/v1",
                "instruction_id": BG_SOFT_INSTRUCTION_ID,
                "amendment_id": BG_SOFT_AMENDMENT_ID,
                "source_head": args.source_head,
                "intent_sha256": intent_sha256,
                "jobs": {LLAMA_ALIAS: job_id},
                "llama_only": True,
                "qwen_submission_authorized": False,
                "retry_or_resubmit_authorized": False,
            },
        )
        _run(["scontrol", "release", job_id])
    except Exception:
        _cancel_held(job_id)
        raise
    print(
        json.dumps(
            {
                "status": "LLAMA_SUBMITTED",
                "instruction_id": BG_SOFT_INSTRUCTION_ID,
                "amendment_id": BG_SOFT_AMENDMENT_ID,
                "source_head": args.source_head,
                "jobs": {LLAMA_ALIAS: job_id},
                "intent_sha256": intent_sha256,
                "submission_receipt_sha256": receipt_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
