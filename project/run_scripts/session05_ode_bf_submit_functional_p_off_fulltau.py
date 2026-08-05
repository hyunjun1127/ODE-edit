#!/usr/bin/env python3
"""Create-once server1 submitter for the S05 functional-P-off pair."""

from __future__ import annotations

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

from project.run_scripts import (
    session05_ode_bf_functional_p_off_fulltau_dry_plan as dry,
)
from project.run_scripts.ode_bf.artifacts import (
    ODEBFArtifactGuard,
    load_rooted_json,
    sha256_file,
)
from project.run_scripts.ode_bf.contracts import MODEL_ALIASES, ODEBFContractError
from project.run_scripts.ode_bf.p1_functional_p_off_panel import (
    FUNCTIONAL_P_OFF_INSTRUCTION_ID,
    FUNCTIONAL_P_OFF_RESULT_TOKEN,
    expected_functional_p_off_result_name,
)
from project.run_scripts.ode_bf.resource import gpu_count_from_tres


SESSION_ID = "019fc63e-5217-7250-9c22-c5b2ec4248f0"
EXPECTED_PARENT = "46c788df3031ae40fdb7a72f6d1ce70eb4dc5370"
EXECUTION_BRANCH = "codex/odeeditsh1-s05-fpoff-fulltau-p1r3-v1"
SERVER1_GPU_CAP = 3
AUTHORIZATION_TOKEN = "functional-p-off-fulltau-p1r3-v1"
SUBMISSION_NAMESPACE = "s05-functional-p-off-fulltau-p1r3-v1"
SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_functional_p_off_fulltau.sbatch"
SOURCE_MANIFEST = (
    REPO_ROOT
    / "project/run_scripts/ode_bf/locks/source_manifest_s05_functional_p_off_fulltau.json"
)
R2_IMMUTABLE = {
    "llama3-8b-inst": {
        "terminal": "cef80834a0708d3fe0f6545eab3786ae197f2555633cfbafab08f254f22a38e9",
        "manifest": "0e49199a6f205d727f91bff8d20dc552cdce3f6af7590e8dda52996206daa87a",
    },
    "qwen2.5-7b-inst": {
        "terminal": "ff0481a5d31ecc5f06bb3666e120b813354e8be7da9217aad25cdfd0581e9da5",
        "manifest": "1222ecb702853da4f8077157df1944dbaa1f9fec8fe8bdb51863ef4b50c1a803",
    },
}


def _run(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
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
        expected_schema="ode-edit-s05-functional-p-off-source-manifest/v1",
    )
    entries = value.get("entries")
    if (
        value.get("instruction_id") != FUNCTIONAL_P_OFF_INSTRUCTION_ID
        or value.get("expected_parent") != EXPECTED_PARENT
        or value.get("execution_head_policy") != "runtime-git-head"
        or not isinstance(entries, list)
        or not entries
    ):
        raise ODEBFContractError("functional-P source manifest header differs")
    observed: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ODEBFContractError("functional-P source manifest entry differs")
        relative = entry["path"]
        path = REPO_ROOT / relative
        if (
            not relative.startswith("project/run_scripts/")
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry.get("size")
            or sha256_file(path) != entry.get("sha256")
        ):
            raise ODEBFContractError("functional-P source manifest content differs")
        observed.append(relative)
    if observed != sorted(observed) or len(observed) != len(set(observed)):
        raise ODEBFContractError("functional-P source manifest paths differ")
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
        raise ODEBFContractError("functional-P source manifest path set differs")
    return raw_sha256


def _immutable_r2_gate() -> None:
    parent = REPO_ROOT / "local/odebf/results"
    for alias, expected in R2_IMMUTABLE.items():
        root = parent / f"s05-target-new-nll-routing-r2-{alias}-v1"
        for name, digest in expected.items():
            path = root / f"{name}.json"
            if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                raise ODEBFContractError("immutable S05 R2 artifact differs")


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
        task_job = name.startswith("odeedit_s05_fpfullr3_")
        if task_job and (gpu is None or gpu <= 0):
            raise ODEBFContractError("task GPU TRES is ambiguous")
        rows.append(
            {"job_id": job_id, "job_name": name, "state": state,
             "gpu": 0 if gpu is None else gpu, "task_job": task_job}
        )
    return rows


def _pre_submit(source_head: str) -> dict[str, Any]:
    if os.environ.get("ODEEDIT_S05_FPOFF_SUBMISSION_AUTHORIZED") != AUTHORIZATION_TOKEN:
        raise ODEBFContractError("functional-P submission token differs")
    if (
        _run(["git", "rev-parse", "HEAD"]).stdout.strip() != source_head
        or _run(["git", "rev-parse", "HEAD^"]).stdout.strip() != EXPECTED_PARENT
        or _run(["git", "branch", "--show-current"]).stdout.strip()
        != EXECUTION_BRANCH
        or _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    ):
        raise ODEBFContractError("functional-P execution boundary differs")
    _run(["scripts/check-session-boundary.sh", SESSION_ID])
    source_manifest_sha256 = _source_manifest_gate(source_head)
    _immutable_r2_gate()
    plan = dry.build_plan(source_head)
    for job in plan["jobs"]:
        forecast = job["forecast"]
        if (
            forecast["conservative_gpu_peak_mib"] > 65_000
            or forecast["conservative_host_peak_mib"] > 65_000
            or forecast["conservative_time_seconds"] > 86_400
        ):
            raise ODEBFContractError("functional-P forecast exceeds allocation")
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
    active_gpu = sum(row["gpu"] for row in scheduler if row["task_job"])
    if active_gpu + 2 > SERVER1_GPU_CAP:
        raise ODEBFContractError("server1 ODE project GPU cap would be exceeded")
    if any(row["job_name"] in dry.JOB_NAMES.values() for row in scheduler):
        raise ODEBFContractError("functional-P scheduler name collides")
    result_parent = REPO_ROOT / "local/odebf/results"
    log_parent = REPO_ROOT / "local/odebf/logs"
    for alias in MODEL_ALIASES:
        if (result_parent / expected_functional_p_off_result_name(alias)).exists():
            raise ODEBFContractError("functional-P result root collides")
        if list(log_parent.glob(f"{dry.JOB_NAMES[alias]}-*")):
            raise ODEBFContractError("functional-P log namespace collides")
    return {
        "source_manifest_sha256": source_manifest_sha256,
        "active_task_gpu": active_gpu,
        "server1_gpu_cap": SERVER1_GPU_CAP,
        "immutable_r2_gate": True,
        "plan": plan,
    }


def _submit_one(alias: str, source_head: str) -> str:
    root = REPO_ROOT / "local/odebf/results" / expected_functional_p_off_result_name(alias)
    logs = REPO_ROOT / "local/odebf/logs"
    logs.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = dry.JOB_NAMES[alias]
    result = _run(
        [
            "sbatch", "--parsable", "--hold", f"--job-name={name}",
            f"--output={logs / (name + '-%j.out')}",
            f"--error={logs / (name + '-%j.err')}", f"--chdir={REPO_ROOT}",
            str(SBATCH), alias, str(root), source_head, FUNCTIONAL_P_OFF_RESULT_TOKEN,
        ]
    )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("functional-P Slurm job ID differs")
    return job_id


def _submit_pair(source_head: str, state: Path, intent_sha256: str) -> tuple[dict[str, str], str]:
    jobs: dict[str, str] = {}
    try:
        for alias in MODEL_ALIASES:
            jobs[alias] = _submit_one(alias, source_head)
        receipt_sha256 = _write_once(
            state / f"{SUBMISSION_NAMESPACE}.submission-receipt.json",
            {
                "schema": "ode-edit-s05-functional-p-off-submission-receipt/v1",
                "instruction_id": FUNCTIONAL_P_OFF_INSTRUCTION_ID,
                "source_head": source_head,
                "intent_sha256": intent_sha256,
                "jobs": jobs,
                "pair_accepted_while_held": True,
                "retry_or_resubmit_authorized": False,
            },
        )
        _run(["scontrol", "release", *jobs.values()])
        return jobs, receipt_sha256
    except BaseException:
        for job_id in jobs.values():
            _run(["scancel", job_id], check=False)
        raise


def main() -> int:
    source_head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    preflight = _pre_submit(source_head)
    state = REPO_ROOT / "local/odebf/state"
    intent_sha256 = _write_once(
        state / f"{SUBMISSION_NAMESPACE}.submission-intent.json",
        {
            "schema": "ode-edit-s05-functional-p-off-submission-intent/v1",
            "instruction_id": FUNCTIONAL_P_OFF_INSTRUCTION_ID,
            "source_head": source_head,
            "preflight": preflight,
        },
    )
    jobs, receipt_sha256 = _submit_pair(source_head, state, intent_sha256)
    print(json.dumps(
        {"jobs": jobs, "intent_sha256": intent_sha256,
         "receipt_sha256": receipt_sha256},
        sort_keys=True, separators=(",", ":"),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
