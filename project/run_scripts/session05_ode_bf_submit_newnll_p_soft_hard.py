#!/usr/bin/env python3
"""Create-once server1 submitter for the S05 NEWNLL H/P soft-hard pair."""

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

from project.run_scripts import session05_ode_bf_newnll_p_soft_hard_dry_plan as dry
from project.run_scripts.ode_bf.artifacts import (
    ODEBFArtifactGuard,
    load_rooted_json,
    sha256_file,
)
from project.run_scripts.ode_bf.contracts import MODEL_ALIASES, ODEBFContractError
from project.run_scripts.ode_bf.p1_newnll_p_soft_hard_panel import (
    NEWNLL_P_SOFT_HARD_INSTRUCTION_ID,
    NEWNLL_P_SOFT_HARD_RESULT_TOKEN,
    expected_newnll_p_soft_hard_result_name,
)
from project.run_scripts.ode_bf.resource import gpu_count_from_tres


SESSION_ID = "019fc63e-5217-7250-9c22-c5b2ec4248f0"
EXPECTED_PARENT = "870d10d6f66fc79626f53e8f4c18cbad9e4c8f07"
EXECUTION_BRANCH = "codex/odeeditsh1-s05-newnll-psoft-hard-p1r5-v1"
SERVER1_PROJECT_GPU_CAP = 3
AUTHORIZATION_TOKEN = "newnll-p-soft-hard-p1r5-r1-v1"
SUBMISSION_NAMESPACE = "s05-newnll-p-soft-hard-p1r5-r1-v1"
SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_newnll_p_soft_hard.sbatch"
SOURCE_MANIFEST = (
    REPO_ROOT
    / "project/run_scripts/ode_bf/locks/"
    "source_manifest_s05_newnll_p_soft_hard.json"
)
PRIOR_IMMUTABLE = {
    "s05-target-new-nll-routing-r2-llama3-8b-inst-v1/terminal.json": (
        "cef80834a0708d3fe0f6545eab3786ae197f2555633cfbafab08f254f22a38e9"
    ),
    "s05-target-new-nll-routing-r2-qwen2.5-7b-inst-v1/terminal.json": (
        "ff0481a5d31ecc5f06bb3666e120b813354e8be7da9217aad25cdfd0581e9da5"
    ),
    "s05-functional-p-off-fulltau-p1r3-llama3-8b-inst-v1/terminal.json": (
        "fa6e471e2e0e1a72a6d33d6d1df7bcfc0ab57d276934f24ae320f5972bfcf724"
    ),
    "s05-functional-p-off-fulltau-p1r3-qwen2.5-7b-inst-v1/failure.json": (
        "b3bb411c478cdf799817a9c78624039476b23b2e695116894e377f3853a0296d"
    ),
    "s05-preservation-alloff-fulltau-p1r4-llama3-8b-inst-v1/terminal.json": (
        "5c8eedcb7b566a41cb70b71ae3b690e193bd102fd5c277265790556838f3b991"
    ),
    "s05-preservation-alloff-fulltau-p1r4-qwen2.5-7b-inst-v1/terminal.json": (
        "7d5f1046999cd280b73e73dd95de3dd77052d9215c88834e4bd858c95d839922"
    ),
}


def _run(
    args: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args), cwd=REPO_ROOT, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def _write_once(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value), ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _source_manifest_gate(source_head: str) -> str:
    value, raw_sha256 = load_rooted_json(
        SOURCE_MANIFEST,
        expected_schema="ode-edit-s05-newnll-p-soft-hard-source-manifest/v1",
    )
    entries = value.get("entries")
    if (
        value.get("instruction_id") != NEWNLL_P_SOFT_HARD_INSTRUCTION_ID
        or value.get("expected_parent") != EXPECTED_PARENT
        or value.get("execution_head_policy") != "runtime-git-head"
        or not isinstance(entries, list)
        or not entries
    ):
        raise ODEBFContractError("P-soft source manifest header differs")
    observed: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ODEBFContractError("P-soft source manifest entry differs")
        relative = entry["path"]
        path = REPO_ROOT / relative
        if (
            not relative.startswith("project/run_scripts/")
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry.get("size")
            or sha256_file(path) != entry.get("sha256")
        ):
            raise ODEBFContractError("P-soft source manifest content differs")
        observed.append(relative)
    if observed != sorted(observed) or len(observed) != len(set(observed)):
        raise ODEBFContractError("P-soft source manifest paths differ")
    manifest_relative = SOURCE_MANIFEST.relative_to(REPO_ROOT).as_posix()
    tracked = set(
        _run(["git", "ls-tree", "-r", "--name-only", source_head])
        .stdout.splitlines()
    )
    expected = {
        relative
        for relative in tracked
        if (
            relative.startswith("project/run_scripts/ode_bf/")
            or relative.startswith("project/run_scripts/session05_ode_bf_")
        )
        and relative != manifest_relative
    }
    if set(observed) != expected:
        raise ODEBFContractError("P-soft source manifest path set differs")
    return raw_sha256


def _prior_immutability_gate() -> None:
    parent = REPO_ROOT / "local/odebf/results"
    for relative, expected in PRIOR_IMMUTABLE.items():
        path = parent / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ODEBFContractError("immutable S05 artifact differs")


def _scheduler_snapshot() -> list[dict[str, Any]]:
    user = os.environ.get("USER")
    if not user:
        raise ODEBFContractError("scheduler user identity unavailable")
    output = _run(
        ["squeue", "-h", "-u", user, "-t", "R,PD", "-o", "%i|%j|%T|%b"]
    ).stdout
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("|", 3)
        if len(fields) != 4:
            raise ODEBFContractError("scheduler row differs")
        job_id, name, state, tres = fields
        gpu = gpu_count_from_tres(tres)
        project_job = name.startswith("odeedit_")
        if project_job and (gpu is None or gpu <= 0):
            raise ODEBFContractError("project GPU TRES is ambiguous")
        rows.append({
            "job_id": job_id,
            "job_name": name,
            "state": state,
            "gpu": 0 if gpu is None else gpu,
            "project_job": project_job,
        })
    return rows


def _host_memory_gate() -> int:
    lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    total_line = next((line for line in lines if line.startswith("MemTotal:")), None)
    if total_line is None:
        raise ODEBFContractError("host memory identity unavailable")
    fields = total_line.split()
    if len(fields) != 3 or fields[2] != "kB" or not fields[1].isdigit():
        raise ODEBFContractError("host memory units differ")
    total_mib = int(fields[1]) // 1024
    if 130_000 > total_mib:
        raise ODEBFContractError("paired host memory exceeds physical capacity")
    return total_mib


def _pre_submit(source_head: str) -> dict[str, Any]:
    if os.environ.get("ODEEDIT_S05_PSOFT_SUBMISSION_AUTHORIZED") != AUTHORIZATION_TOKEN:
        raise ODEBFContractError("P-soft submission token differs")
    if (
        _run(["git", "rev-parse", "HEAD"]).stdout.strip() != source_head
        or _run(["git", "rev-parse", "HEAD^"]).stdout.strip() != EXPECTED_PARENT
        or _run(["git", "branch", "--show-current"]).stdout.strip()
        != EXECUTION_BRANCH
        or _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    ):
        raise ODEBFContractError("P-soft execution boundary differs")
    _run(["scripts/check-session-boundary.sh", SESSION_ID])
    source_manifest_sha256 = _source_manifest_gate(source_head)
    _prior_immutability_gate()
    plan = dry.build_plan(source_head)
    for job in plan["jobs"]:
        forecast = job["forecast"]
        if (
            forecast["conservative_gpu_peak_mib"] > 65_000
            or forecast["conservative_host_peak_mib"] > 65_000
            or forecast["conservative_time_seconds"] > 86_400
        ):
            raise ODEBFContractError("P-soft forecast exceeds allocation")
    for alias in MODEL_ALIASES:
        guard = ODEBFArtifactGuard(
            REPO_ROOT,
            REPO_ROOT / "project/run_scripts/ode_bf/locks/p0_artifact_lock.json",
            alias,
            require_held_ode_alloc=False,
        )
        guard.preflight()
        guard.assert_unchanged()
    scheduler = _scheduler_snapshot()
    active_project_gpu = sum(
        row["gpu"] for row in scheduler if row["project_job"]
    )
    if active_project_gpu + 2 > SERVER1_PROJECT_GPU_CAP:
        raise ODEBFContractError("server1 project GPU cap would be exceeded")
    if any(row["job_name"] in dry.JOB_NAMES.values() for row in scheduler):
        raise ODEBFContractError("P-soft scheduler name collides")
    result_parent = REPO_ROOT / "local/odebf/results"
    log_parent = REPO_ROOT / "local/odebf/logs"
    for alias in MODEL_ALIASES:
        if (result_parent / expected_newnll_p_soft_hard_result_name(alias)).exists():
            raise ODEBFContractError("P-soft result root collides")
        if list(log_parent.glob(f"{dry.JOB_NAMES[alias]}-*")):
            raise ODEBFContractError("P-soft log namespace collides")
    return {
        "source_manifest_sha256": source_manifest_sha256,
        "active_project_gpu": active_project_gpu,
        "server1_project_gpu_cap": SERVER1_PROJECT_GPU_CAP,
        "server_wide_gpu_inventory_not_used_as_project_occupancy": True,
        "host_total_mib": _host_memory_gate(),
        "prior_immutability_gate": True,
        "plan": plan,
    }


def _submit_one(alias: str, source_head: str) -> str:
    root = (
        REPO_ROOT / "local/odebf/results"
        / expected_newnll_p_soft_hard_result_name(alias)
    )
    logs = REPO_ROOT / "local/odebf/logs"
    logs.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = dry.JOB_NAMES[alias]
    result = _run([
        "sbatch", "--parsable", "--hold", f"--job-name={name}",
        f"--output={logs / (name + '-%j.out')}",
        f"--error={logs / (name + '-%j.err')}",
        f"--chdir={REPO_ROOT}", str(SBATCH), alias, str(root), source_head,
        NEWNLL_P_SOFT_HARD_RESULT_TOKEN,
    ])
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("P-soft Slurm job ID differs")
    return job_id


def _cancel_held(jobs: Mapping[str, str]) -> None:
    if jobs:
        _run(["scancel", *jobs.values()], check=False)


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument(
        "--state-root",
        type=Path,
        default=REPO_ROOT / "local/odebf/state",
    )
    args = parser.parse_args()
    preflight = _pre_submit(args.source_head)
    intent = {
        "schema": "ode-edit-s05-newnll-p-soft-hard-submit-intent/v1",
        "instruction_id": NEWNLL_P_SOFT_HARD_INSTRUCTION_ID,
        "source_head": args.source_head,
        "preflight": preflight,
        "pair_back_to_back": True,
        "pair_accepted_while_held": True,
        "retry_or_resubmit_authorized": False,
    }
    intent_sha256 = _write_once(
        args.state_root / f"{SUBMISSION_NAMESPACE}.intent.json", intent
    )
    jobs: dict[str, str] = {}
    try:
        for alias in MODEL_ALIASES:
            jobs[alias] = _submit_one(alias, args.source_head)
        receipt_sha256 = _write_once(
            args.state_root / f"{SUBMISSION_NAMESPACE}.submission-receipt.json",
            {
                "schema": "ode-edit-s05-newnll-p-soft-hard-submission-receipt/v1",
                "instruction_id": NEWNLL_P_SOFT_HARD_INSTRUCTION_ID,
                "source_head": args.source_head,
                "intent_sha256": intent_sha256,
                "jobs": jobs,
                "pair_accepted_while_held": True,
                "retry_or_resubmit_authorized": False,
            },
        )
        _run(["scontrol", "release", *jobs.values()])
    except Exception:
        _cancel_held(jobs)
        raise
    print(json.dumps({
        "status": "PAIR_SUBMITTED",
        "instruction_id": NEWNLL_P_SOFT_HARD_INSTRUCTION_ID,
        "source_head": args.source_head,
        "jobs": jobs,
        "intent_sha256": intent_sha256,
        "submission_receipt_sha256": receipt_sha256,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
