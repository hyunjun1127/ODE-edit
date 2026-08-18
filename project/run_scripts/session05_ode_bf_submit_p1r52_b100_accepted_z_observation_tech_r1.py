#!/usr/bin/env python3
"""Held-inspect-release invalid-baseline-only TECH-R1 repair submitter."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1r52_accepted_z_observation_panel import (
    MEMIT_ROLE,
    NATIVE_CORRECTED_ROLE,
    expected_result_name,
)


BRANCH = "codex/p1r52-b100-accepted-z-rephrase-obs-tech-r1-v1"
SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_p1r52_b100_accepted_z_observation.sbatch"
RESULT_PARENT = REPO_ROOT / "local/odebf/results"
ORIGINAL_ACTIVE_TASKS = {"20588_2", "20588_3"}
REPAIR_ROLES = (MEMIT_ROLE, NATIVE_CORRECTED_ROLE)


def run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def write_once(path: Path, value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = run(["git", "branch", "--show-current"]).stdout.strip()
    if branch != BRANCH or run(["git", "status", "--porcelain"]).stdout:
        raise ODEBFContractError("accepted-z TECH-R1 source differs")
    result_names = {
        role: expected_result_name(role, repair_revision="TECH-R1")
        for role in REPAIR_ROLES
    }
    if any((RESULT_PARENT / name).exists() for name in result_names.values()):
        raise ODEBFContractError("accepted-z TECH-R1 result exists")
    queue = run(
        [
            "squeue", "-h", "-u", "janghj", "-w", "devbox",
            "-t", "RUNNING,CONFIGURING,PENDING", "-o", "%i|%T|%b",
        ]
    ).stdout.splitlines()
    jobs = []
    allocated = 0
    for line in queue:
        job_id, state, gres = line.split("|", 2)
        count = sum(
            int(value)
            for value in re.findall(r"(?:gres/)?gpu(?::[^,:=|]+)*[:=](\d+)", gres)
        )
        if count:
            jobs.append({"job_id": job_id, "state": state, "gres": gres, "gpu": count})
            if state in {"RUNNING", "CONFIGURING"}:
                allocated += count
    if {str(item["job_id"]) for item in jobs} != ORIGINAL_ACTIVE_TASKS or allocated != 2:
        raise ODEBFContractError("accepted-z TECH-R1 active healthy scope differs")
    memory_mib = next(
        int(line.split()[1]) // 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if allocated + 2 > 4 or memory_mib < 130000:
        raise ODEBFContractError("accepted-z TECH-R1 cap/memory differs")
    state = REPO_ROOT / "local/odebf/state/p1r52-b100-accepted-z-observation-tech-r1-v1"
    logs = REPO_ROOT / "local/odebf/logs/p1r52-b100-accepted-z-observation-tech-r1-v1"
    logs.mkdir(mode=0o700, parents=True, exist_ok=True)
    intent = {
        "schema": "ode-edit-s05-p1r52-b100-accepted-z-tech-r1-intent/v1",
        "source_head": head,
        "repair_parent": "bed5b8da2df01c16a43774bf09e45c15c07a8c9f",
        "roles": list(REPAIR_ROLES),
        "result_names": result_names,
        "active_healthy_jobs": jobs,
        "active_gpu": allocated,
        "new_gpu": 2,
        "project_cap": 4,
        "host_memory_available_mib": memory_mib,
        "held_then_release": True,
    }
    intent_sha = write_once(state / "submission-intent.json", intent)
    command = [
        "sbatch", "--hold", "--parsable", "--array", "0-1%2",
        "--chdir", str(REPO_ROOT), "--nodelist", "devbox",
        "--job-name", "odeedit_p1r52_b100_accepted_z_tech_r1",
        "--output", str(logs / "%A_%a.out"),
        "--error", str(logs / "%A_%a.err"), str(SBATCH), head,
        str(RESULT_PARENT), "accepted-z-rephrase-obs-tech-r1",
    ]
    job_id = run(command).stdout.strip().split(";", 1)[0]
    observed = run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
    if not all(
        item in observed
        for item in ("JobState=PENDING", "Reason=JobHeldUser", "ReqNodeList=devbox")
    ):
        run(["scancel", job_id], check=False)
        raise ODEBFContractError("accepted-z TECH-R1 held inspection differs")
    receipt = {
        **intent,
        "job_id": job_id,
        "array": "0-1%2",
        "intent_sha256": intent_sha,
        "held_inspection_sha256": hashlib.sha256(observed.encode()).hexdigest(),
    }
    receipt_sha = write_once(state / "submission-receipt.json", receipt)
    run(["scontrol", "release", job_id])
    print(json.dumps({**receipt, "submission_receipt_sha256": receipt_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
