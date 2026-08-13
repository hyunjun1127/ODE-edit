#!/usr/bin/env python3
"""Held-inspect-release submitter for ordered P1R30 atomic stages."""

from __future__ import annotations

import argparse
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

from project.run_scripts import session05_ode_bf_p1r30_debt_priority_dry_plan as dry
from project.run_scripts.ode_bf.contracts import ODEBFContractError


SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_p1r30_debt_priority.sbatch"
STATE_ROOT = REPO_ROOT / "local/odebf/state/p1r30-debt-priority-a0-barrier"
RESULT_PARENT = REPO_ROOT / "local/odebf/results"
BRANCH = "codex/odeeditsh2-s05-p1r30-debt-priority-a0-barrier-v1"
SERVER2_GPU_CAP = 4
ORIGINAL_B10_ORDER = (
    "984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b"
)


def _run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _write_once(path: Path, value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ODEBFContractError("P1R30 predecessor artifact differs")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ODEBFContractError("P1R30 predecessor JSON differs") from exc
    if not isinstance(value, dict):
        raise ODEBFContractError("P1R30 predecessor payload differs")
    return value


def _load_decision_gate(
    source_head: str,
    *,
    name: str,
    schema: str,
    required_boolean: str,
) -> dict[str, object]:
    path = STATE_ROOT / f"{name}-{source_head[:12]}.json"
    gate = _load_json(path)
    if (
        gate.get("schema") != schema
        or gate.get("status") != "PASS"
        or gate.get("source_head") != source_head
        or gate.get("aliases") != list(dry.ALIASES)
        or gate.get(required_boolean) is not True
        or gate.get("scientific_invalid_count") != 0
    ):
        raise ODEBFContractError("P1R30 scientific predecessor gate differs")
    terminals = gate.get("terminal_sha256")
    if (
        not isinstance(terminals, dict)
        or set(terminals) != set(dry.ALIASES)
        or any(not isinstance(value, str) or len(value) != 64 for value in terminals.values())
    ):
        raise ODEBFContractError("P1R30 predecessor terminal binding differs")
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": _sha256_file(path),
        "payload": gate,
    }


def _predecessor_gate(source_head: str, phase: str) -> dict[str, object]:
    if phase == "b1":
        return {
            "schema": "ode-edit-s05-p1r30-root-stage/v1",
            "status": "ROOT_STAGE",
        }
    if phase == "b10-neutral":
        return _load_decision_gate(
            source_head,
            name="b1-technical-semantic-gate",
            schema="ode-edit-s05-p1r30-b1-technical-semantic-gate/v1",
            required_boolean="dual_alias_technical_semantic_pass",
        )
    if phase == "full-matrix-completion":
        return _load_decision_gate(
            source_head,
            name="neutral-recovery-gate",
            schema="ode-edit-s05-p1r30-neutral-recovery-gate/v1",
            required_boolean="both_aliases_p1r24_a0_strength_region",
        )
    raise ODEBFContractError("P1R30 predecessor phase differs")


def _active_gpu_allocations() -> tuple[int, list[str]]:
    lines = _run(
        [
            "squeue",
            "-h",
            "-u",
            "janghj",
            "-w",
            "server2",
            "-t",
            "RUNNING",
            "-o",
            "%i|%b",
        ]
    ).stdout.splitlines()
    count = 0
    identities: list[str] = []
    for line in lines:
        match = re.search(
            r"(?:gres/)?gpu(?::[^:,|]+)?:(\d+)", line, re.IGNORECASE
        )
        if match is not None:
            count += int(match.group(1))
            identities.append(line.split("|", 1)[0])
    return count, identities


def submit(
    source_head: str,
    phase: str,
    *,
    attempt_tag: str = "",
) -> dict[str, object]:
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    tree = _run(["git", "rev-parse", "HEAD^{tree}"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(["git", "status", "--porcelain"]).stdout
    if source_head != head or branch != BRANCH or dirty:
        raise ODEBFContractError("P1R30 execution source differs")
    plan = dry.build_plan(source_head, phase, attempt_tag=attempt_tag)
    predecessor_gate = _predecessor_gate(source_head, phase)
    if any(
        (RESULT_PARENT / str(job["result_name"])).exists()
        or (RESULT_PARENT / str(job["result_name"])).is_symlink()
        for job in plan["jobs"]
    ):
        raise ODEBFContractError("P1R30 result namespace exists")
    active, active_job_ids = _active_gpu_allocations()
    available = SERVER2_GPU_CAP - active
    stage_cap = int(plan["stage_max_concurrent_gpu"])
    new_concurrency = min(stage_cap, int(plan["job_count"]), available)
    if new_concurrency <= 0:
        raise ODEBFContractError("P1R30 server2 janghj GPU cap unavailable")
    attempt_component = f"-{attempt_tag}" if attempt_tag else ""
    namespace = (
        f"s05-p1r30-{phase}-{source_head[:12]}{attempt_component}-v1"
    )
    intent_path = STATE_ROOT / f"{namespace}.intent.json"
    receipt_path = STATE_ROOT / f"{namespace}.submission-receipt.json"
    if any(path.exists() or path.is_symlink() for path in (intent_path, receipt_path)):
        raise ODEBFContractError("P1R30 submission namespace exists")
    log_root = REPO_ROOT / (
        f"local/odebf/logs/p1r30-{phase}-{source_head[:12]}"
        f"{attempt_component}"
    )
    log_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    array = f"0-{int(plan['job_count']) - 1}%{new_concurrency}"
    intent = {
        "schema": "ode-edit-s05-p1r30-submission-intent/v1",
        "source_head": source_head,
        "source_tree": tree,
        "phase": phase,
        "technical_attempt_tag": attempt_tag or None,
        "dry_plan": plan,
        "predecessor_gate": predecessor_gate,
        "active_server2_gpu_allocations": active,
        "active_server2_gpu_job_ids": active_job_ids,
        "server2_janghj_gpu_cap": SERVER2_GPU_CAP,
        "released_max_concurrent_gpu": new_concurrency,
        "held_then_atomic_release": True,
        "array": array,
        "callback_job_count": 0,
    }
    intent_sha = _write_once(intent_path, intent)
    submitted = _run(
        [
            "sbatch",
            "--hold",
            "--parsable",
            "--array",
            array,
            "--chdir",
            str(REPO_ROOT),
            "--nodelist",
            "server2",
            "--job-name",
            f"odeedit_s05_p1r30_{phase}",
            "--output",
            str(log_root / "%A_%a.out"),
            "--error",
            str(log_root / "%A_%a.err"),
            str(SBATCH),
            source_head,
            str(RESULT_PARENT),
            phase,
            attempt_tag,
        ]
    )
    job_id = submitted.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("P1R30 scheduler ID differs")
    observed = _run(["scontrol", "show", "job", "-o", job_id]).stdout.strip()
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "ReqNodeList=server2",
        "TRES=cpu=8,mem=65000M,node=1,billing=8,gres/gpu=1",
        f"WorkDir={REPO_ROOT}",
    )
    if not all(item in observed for item in required):
        _run(["scancel", job_id], check=False)
        raise ODEBFContractError("P1R30 held scheduler contract differs")
    receipt = {
        "schema": "ode-edit-s05-p1r30-submission/v1",
        "source_head": source_head,
        "source_tree": tree,
        "phase": phase,
        "technical_attempt_tag": attempt_tag or None,
        "job_id": job_id,
        "array": array,
        "job_count": int(plan["job_count"]),
        "trajectory_count": int(plan["trajectory_count"]),
        "released_max_concurrent_gpu": new_concurrency,
        "server2_janghj_gpu_cap": SERVER2_GPU_CAP,
        "intent_sha256": intent_sha,
        "held_inspection_sha256": hashlib.sha256(observed.encode()).hexdigest(),
        "held_then_atomic_release": True,
        "callback_job_count": 0,
    }
    receipt_sha = _write_once(receipt_path, receipt)
    _run(["scontrol", "release", job_id])
    return {**receipt, "submission_receipt_sha256": receipt_sha}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--phase", required=True, choices=tuple(dry.PHASE_JOBS))
    parser.add_argument("--technical-attempt", default="")
    args = parser.parse_args()
    print(
        json.dumps(
            submit(
                args.source_head,
                args.phase,
                attempt_tag=args.technical_attempt,
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
