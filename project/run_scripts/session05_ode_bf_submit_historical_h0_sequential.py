#!/usr/bin/env python3
"""Held-inspect-release submitter for the ten-task Full-6 Historical array."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts import session05_ode_bf_historical_h0_sequential_dry_plan as dry
from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash


SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_historical_h0_sequential.sbatch"
STATE_ROOT = REPO_ROOT / "local/odebf/state"
LOG_ROOT = REPO_ROOT / "local/odebf/logs/p1r23-full6-structural-historical-tech-r1"
RESULT_PARENT = REPO_ROOT / "local/odebf/results"
SOURCE_MANIFEST = (
    REPO_ROOT
    / "project/run_scripts/ode_bf/locks/source_manifest_s05_historical_h0_sequential.json"
)
NAMESPACE = "s05-p1r23-full6-structural-historical-sh1-array-tech-r1-v1"
BRANCH = "codex/p1r23-compute-a1-historical"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=REPO_ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _write_once(path: Path, value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _verify_source_manifest() -> tuple[str, str]:
    value = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    observed_root = value.pop("root_digest", None)
    if observed_root != canonical_hash(value):
        raise ODEBFContractError("P1R23 Full-6 Historical manifest root differs")
    entries = value.get("entries")
    if not isinstance(entries, list) or value.get("entry_count") != len(entries):
        raise ODEBFContractError("P1R23 Full-6 Historical manifest inventory differs")
    for item in entries:
        path = REPO_ROOT / str(item["path"])
        status = path.lstat()
        payload = path.read_bytes()
        object_id = _run(["git", "hash-object", str(path)]).stdout.strip()
        if (
            not path.is_file()
            or path.is_symlink()
            or status.st_mode != int(item["mode"])
            or len(payload) != int(item["size"])
            or hashlib.sha256(payload).hexdigest() != item["sha256"]
            or object_id != item["object_id"]
        ):
            raise ODEBFContractError(
                "P1R23 Full-6 Historical manifest member differs"
            )
    return hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest(), observed_root


def submit(source_head: str) -> dict[str, object]:
    if _run(["git", "rev-parse", "HEAD"]).stdout.strip() != source_head or _run(["git", "branch", "--show-current"]).stdout.strip() != BRANCH or _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout:
        raise ODEBFContractError("P1R23 Full-6 Historical execution source differs")
    source_manifest_sha, source_manifest_root = _verify_source_manifest()
    alpha_states = {
        item.split("|", 1)[0]: item.split("|", 1)[1]
        for item in _run(
            ["squeue", "-h", "-j", "18768_4,18768_9", "-o", "%i|%T"]
        ).stdout.splitlines()
        if "|" in item
    }
    if alpha_states != {"18768_4": "RUNNING", "18768_9": "RUNNING"}:
        raise ODEBFContractError(
            "P1R23 Full-6 Historical preserved Alpha GPU occupancy differs"
        )
    array_spec = "0-7%2"
    plan = dry.build_plan(
        source_head, attempt_namespace="tech-r1", include_alpha=False
    )
    if any((RESULT_PARENT / str(job["result_name"])).exists() for job in plan["jobs"]):
        raise ODEBFContractError("P1R23 Full-6 Historical result namespace exists")
    intent_path = STATE_ROOT / f"{NAMESPACE}.intent.json"
    receipt_path = STATE_ROOT / f"{NAMESPACE}.submission-receipt.json"
    if any(path.exists() or path.is_symlink() for path in (intent_path, receipt_path)):
        raise ODEBFContractError("P1R23 Full-6 Historical submission namespace exists")
    LOG_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    intent_sha = _write_once(intent_path, {"schema": "ode-edit-s05-p1r23-full6-historical-tech-r1-intent/v1", "source_head": source_head, "source_manifest_sha256": source_manifest_sha, "source_manifest_root": source_manifest_root, "dry_plan": plan, "held_then_atomic_release": True, "array": array_spec, "task_gpu_cap": 4, "preserved_running_alpha_gpu": 2, "initial_replacement_array_throttle": 2, "increase_throttle_to_4_after_alpha_terminal": True, "replaces_failed_or_cancelled_ode_tasks_of_array": "18768", "alphaedit_jobs_preserved": ["18768_4", "18768_9"]})
    job_id = _run(["sbatch", "--hold", "--parsable", "--array", array_spec, "--chdir", str(REPO_ROOT), "--job-name", "odeedit_s05_p1r23_full6_historical_tech_r1", "--output", str(LOG_ROOT / "%A_%a.out"), "--error", str(LOG_ROOT / "%A_%a.err"), str(SBATCH), source_head, str(RESULT_PARENT), "tech-r1"]).stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("P1R23 Full-6 Historical scheduler ID differs")
    observed = _run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
    required = ("JobState=PENDING", "Reason=JobHeldUser", "ArrayTaskThrottle=2", "ReqNodeList=devbox", "TRES=cpu=8,mem=65000M,node=1,billing=8,gres/gpu=1")
    if not all(item in observed for item in required):
        _run(["scancel", job_id])
        raise ODEBFContractError("P1R23 Full-6 Historical held scheduler contract differs")
    receipt = {"schema": "ode-edit-s05-p1r23-full6-historical-tech-r1-submission/v1", "source_head": source_head, "source_manifest_sha256": source_manifest_sha, "source_manifest_root": source_manifest_root, "job_id": job_id, "array": array_spec, "trajectory_count": 8, "ode_trajectory_count": 8, "model_level_alphaedit_count": 0, "task_gpu_cap": 4, "initial_replacement_array_throttle": 2, "preserved_running_alpha_gpu": 2, "intent_sha256": intent_sha, "held_inspection_sha256": hashlib.sha256(observed.encode()).hexdigest(), "held_then_atomic_release": True, "alphaedit_jobs_preserved": ["18768_4", "18768_9"]}
    receipt_sha = _write_once(receipt_path, receipt)
    _run(["scontrol", "release", job_id])
    return {**receipt, "submission_receipt_sha256": receipt_sha}


def submit_tech_r2_missing_qwen(
    source_head: str, *, repair_scope: str = "bulk"
) -> dict[str, object]:
    if (
        _run(["git", "rev-parse", "HEAD"]).stdout.strip() != source_head
        or _run(["git", "branch", "--show-current"]).stdout.strip() != BRANCH
        or _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    ):
        raise ODEBFContractError("P1R23 Full-6 Historical execution source differs")
    source_manifest_sha, source_manifest_root = _verify_source_manifest()
    plan = dry.build_plan(
        source_head, attempt_namespace="tech-r2", include_alpha=False
    )
    if repair_scope == "bulk":
        selected_indices = (4, 6, 7)
        expected_roles = (
            ("qwen2.5-7b-inst", "BG-COMPUTE-FULL6-NOSOFT"),
            ("qwen2.5-7b-inst", "RS-COMPUTE-FULL6-NOSOFT"),
            ("qwen2.5-7b-inst", "RS-COMPUTE-FULL6-SOFT"),
        )
        failed_jobs = ("18782", "18784", "18777_7")
        namespace = "s05-p1r23-full6-structural-historical-sh1-qwen-missing-tech-r2-r1-v1"
    elif repair_scope == "task5":
        selected_indices = (5,)
        expected_roles = (("qwen2.5-7b-inst", "BG-COMPUTE-FULL6-SOFT"),)
        failed_jobs = ("18783",)
        namespace = "s05-p1r23-full6-structural-historical-sh1-qwen-task5-tech-r2-v1"
    else:
        raise ODEBFContractError("P1R23 Full-6 Historical TECH-R2 scope differs")
    selected = [job for job in plan["jobs"] if job["array_index"] in selected_indices]
    if tuple((job["alias"], job["method"]) for job in selected) != expected_roles:
        raise ODEBFContractError("P1R23 Full-6 Historical TECH-R2 role differs")
    result_roots = tuple(RESULT_PARENT / str(job["result_name"]) for job in selected)
    if any(root.exists() or root.is_symlink() for root in result_roots):
        raise ODEBFContractError("P1R23 Full-6 Historical result namespace exists")

    log_root = REPO_ROOT / "local/odebf/logs/p1r23-full6-structural-historical-tech-r2"
    intent_path = STATE_ROOT / f"{namespace}.intent.json"
    receipt_path = STATE_ROOT / f"{namespace}.submission-receipt.json"
    if any(path.exists() or path.is_symlink() for path in (intent_path, receipt_path)):
        raise ODEBFContractError("P1R23 Full-6 Historical submission namespace exists")
    failed_states = {
        job: _run(["sacct", "-n", "-X", "-j", job, "-o", "State", "-P"]).stdout.strip()
        for job in failed_jobs
    }
    if not all(state.startswith("FAILED") for state in failed_states.values()):
        raise ODEBFContractError("P1R23 Full-6 Historical failed ownership differs")

    active_original = len(
        _run(["squeue", "-h", "-j", "18777,18789", "-t", "RUNNING", "-o", "%i"])
        .stdout.splitlines()
    )
    replacement_throttle = min(len(selected_indices), 4 - active_original)
    if replacement_throttle <= 0:
        raise ODEBFContractError("P1R23 Full-6 Historical task GPU cap differs")
    array_members = "4,6-7" if repair_scope == "bulk" else "5"
    array_spec = f"{array_members}%{replacement_throttle}"
    log_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    intent_sha = _write_once(
        intent_path,
        {
            "schema": "ode-edit-s05-p1r23-full6-historical-tech-r2-intent/v1",
            "source_head": source_head,
            "source_manifest_sha256": source_manifest_sha,
            "source_manifest_root": source_manifest_root,
            "failed_jobs": list(failed_jobs),
            "failed_states": failed_states,
            "array_tasks": list(selected_indices),
            "selected_jobs": selected,
            "result_roots": [str(root) for root in result_roots],
            "active_original_gpu": active_original,
            "replacement_array_throttle": replacement_throttle,
            "array": array_spec,
            "task_gpu_cap": 4,
            "held_then_atomic_release": True,
        },
    )
    job_id = _run(
        [
            "sbatch", "--hold", "--parsable", "--array", array_spec,
            "--chdir", str(REPO_ROOT),
            "--job-name", "odeedit_s05_p1r23_full6_historical_tech_r2_task4",
            "--output", str(log_root / "%A_%a.out"),
            "--error", str(log_root / "%A_%a.err"),
            str(SBATCH), source_head, str(RESULT_PARENT), "tech-r2",
        ]
    ).stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("P1R23 Full-6 Historical scheduler ID differs")
    observed = _run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
    required = (
        "JobState=PENDING", "Reason=JobHeldUser", "ReqNodeList=devbox",
        f"ArrayTaskThrottle={replacement_throttle}",
        "TRES=cpu=8,mem=65000M,node=1,billing=8,gres/gpu=1",
    )
    if not all(item in observed for item in required):
        _run(["scancel", job_id])
        raise ODEBFContractError("P1R23 Full-6 Historical held scheduler contract differs")
    receipt = {
        "schema": "ode-edit-s05-p1r23-full6-historical-tech-r2-submission/v1",
        "source_head": source_head,
        "source_manifest_sha256": source_manifest_sha,
        "source_manifest_root": source_manifest_root,
        "job_id": job_id,
        "array_tasks": list(selected_indices),
        "array": array_spec,
        "failed_jobs": list(failed_jobs),
        "result_roots": [str(root) for root in result_roots],
        "active_original_gpu": active_original,
        "replacement_array_throttle": replacement_throttle,
        "task_gpu_cap": 4,
        "intent_sha256": intent_sha,
        "held_inspection_sha256": hashlib.sha256(observed.encode()).hexdigest(),
        "held_then_atomic_release": True,
    }
    receipt_sha = _write_once(receipt_path, receipt)
    _run(["scontrol", "release", job_id])
    return {**receipt, "submission_receipt_sha256": receipt_sha}


def submit_tech_r3_task7(source_head: str) -> dict[str, object]:
    if (
        _run(["git", "rev-parse", "HEAD"]).stdout.strip() != source_head
        or _run(["git", "branch", "--show-current"]).stdout.strip() != BRANCH
        or _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    ):
        raise ODEBFContractError("P1R23 Full-6 Historical execution source differs")
    source_manifest_sha, source_manifest_root = _verify_source_manifest()
    plan = dry.build_plan(
        source_head, attempt_namespace="tech-r3", include_alpha=False
    )
    selected = [job for job in plan["jobs"] if job["array_index"] == 7]
    if len(selected) != 1 or (
        selected[0]["alias"], selected[0]["method"]
    ) != ("qwen2.5-7b-inst", "RS-COMPUTE-FULL6-SOFT"):
        raise ODEBFContractError("P1R23 Full-6 Historical TECH-R3 role differs")
    result_root = RESULT_PARENT / str(selected[0]["result_name"])
    if result_root.exists() or result_root.is_symlink():
        raise ODEBFContractError("P1R23 Full-6 Historical result namespace exists")
    failed_state = _run(
        ["sacct", "-n", "-X", "-j", "18789_7", "-o", "State", "-P"]
    ).stdout.strip()
    if not failed_state.startswith("FAILED"):
        raise ODEBFContractError("P1R23 Full-6 Historical failed ownership differs")
    active_gpu = len(
        _run(["squeue", "-h", "-j", "18789,18792", "-t", "RUNNING", "-o", "%i"])
        .stdout.splitlines()
    )
    if active_gpu > 3:
        raise ODEBFContractError("P1R23 Full-6 Historical task GPU cap differs")
    namespace = "s05-p1r23-full6-structural-historical-sh1-qwen-task7-tech-r3-v1"
    log_root = REPO_ROOT / "local/odebf/logs/p1r23-full6-structural-historical-tech-r3"
    intent_path = STATE_ROOT / f"{namespace}.intent.json"
    receipt_path = STATE_ROOT / f"{namespace}.submission-receipt.json"
    if any(path.exists() or path.is_symlink() for path in (intent_path, receipt_path)):
        raise ODEBFContractError("P1R23 Full-6 Historical submission namespace exists")
    log_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    intent_sha = _write_once(
        intent_path,
        {
            "schema": "ode-edit-s05-p1r23-full6-historical-tech-r3-intent/v1",
            "source_head": source_head,
            "source_manifest_sha256": source_manifest_sha,
            "source_manifest_root": source_manifest_root,
            "failed_job": "18789_7",
            "array_task": 7,
            "selected_job": selected[0],
            "result_root": str(result_root),
            "active_gpu": active_gpu,
            "replacement_gpu": 1,
            "task_gpu_cap": 4,
            "held_then_atomic_release": True,
        },
    )
    job_id = _run(
        [
            "sbatch", "--hold", "--parsable", "--array", "7%1",
            "--chdir", str(REPO_ROOT),
            "--job-name", "odeedit_s05_p1r23_full6_historical_tech_r3_task7",
            "--output", str(log_root / "%A_%a.out"),
            "--error", str(log_root / "%A_%a.err"),
            str(SBATCH), source_head, str(RESULT_PARENT), "tech-r3",
        ]
    ).stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("P1R23 Full-6 Historical scheduler ID differs")
    observed = _run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
    required = (
        "JobState=PENDING", "Reason=JobHeldUser", "ArrayTaskThrottle=1",
        "ReqNodeList=devbox", "TRES=cpu=8,mem=65000M,node=1,billing=8,gres/gpu=1",
    )
    if not all(item in observed for item in required):
        _run(["scancel", job_id])
        raise ODEBFContractError("P1R23 Full-6 Historical held scheduler contract differs")
    receipt = {
        "schema": "ode-edit-s05-p1r23-full6-historical-tech-r3-submission/v1",
        "source_head": source_head,
        "source_manifest_sha256": source_manifest_sha,
        "source_manifest_root": source_manifest_root,
        "job_id": job_id,
        "array_task": 7,
        "failed_job": "18789_7",
        "result_root": str(result_root),
        "active_gpu": active_gpu,
        "replacement_gpu": 1,
        "task_gpu_cap": 4,
        "intent_sha256": intent_sha,
        "held_inspection_sha256": hashlib.sha256(observed.encode()).hexdigest(),
        "held_then_atomic_release": True,
    }
    receipt_sha = _write_once(receipt_path, receipt)
    _run(["scontrol", "release", job_id])
    return {**receipt, "submission_receipt_sha256": receipt_sha}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--repair-missing-qwen", action="store_true")
    parser.add_argument("--repair-task5", action="store_true")
    parser.add_argument("--repair-task7-tech-r3", action="store_true")
    args = parser.parse_args()
    if sum((args.repair_missing_qwen, args.repair_task5, args.repair_task7_tech_r3)) > 1:
        parser.error("repair scopes are mutually exclusive")
    result = (
        submit_tech_r2_missing_qwen(args.source_head, repair_scope="bulk")
        if args.repair_missing_qwen
        else submit_tech_r2_missing_qwen(args.source_head, repair_scope="task5")
        if args.repair_task5
        else submit_tech_r3_task7(args.source_head)
        if args.repair_task7_tech_r3
        else submit(args.source_head)
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
