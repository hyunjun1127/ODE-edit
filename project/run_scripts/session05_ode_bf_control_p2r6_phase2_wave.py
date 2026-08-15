#!/usr/bin/env python3
"""Fail-close Wave-B release/cancel controller for P2R6 RED-R2 Phase 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_ROOT = REPO_ROOT / "local/odebf/state/p2r6-red-r2-final-v1"
BRANCH = "codex/p2r6-red-r2-final-v1"
WAVE_B_INDICES = (0, 3)
PASS_STATUS = "PHASE2_WAVE_A_PASS"
CANCEL_STATUSES = frozenset(
    {
        "CORE_SCIENTIFIC_FAIL",
        "STRUCTP_VARIANT_KILLED",
        "MECHANISM_NO_SIGNAL",
        "TECHNICAL_INVALID",
        "INCONCLUSIVE",
    }
)


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=True,
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


def _load_gate(path: Path) -> tuple[dict[str, object], str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o777 != 0o600:
        raise ValueError("P2R6 Wave-A gate receipt mode/type differs")
    raw = path.read_bytes()
    value = json.loads(raw)
    if value.get("schema") != "P2R6_PHASE2_WAVE_A_GATE_V1":
        raise ValueError("P2R6 Wave-A gate schema differs")
    return value, hashlib.sha256(raw).hexdigest()


def control(*, job_id: str, action: str, gate_receipt: Path) -> dict[str, object]:
    if not job_id.isdigit():
        raise ValueError("P2R6 Phase2 scheduler ID differs")
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    if branch != BRANCH or dirty:
        raise ValueError("P2R6 Wave-B execution source differs")
    gate, gate_sha = _load_gate(gate_receipt)
    if (
        gate.get("source_head") != head
        or gate.get("phase2_job_id") != job_id
        or gate.get("selected_controller") not in ("AR", "AS")
    ):
        raise ValueError("P2R6 Wave-A gate identity differs")
    status = gate.get("status")
    if action == "release" and status != PASS_STATUS:
        raise ValueError("P2R6 Wave-B release gate differs")
    if action == "cancel" and status not in CANCEL_STATUSES:
        raise ValueError("P2R6 Wave-B cancel gate differs")
    observed: list[dict[str, str]] = []
    for index in WAVE_B_INDICES:
        task_id = f"{job_id}_{index}"
        state = _run(["scontrol", "show", "job", "-o", task_id]).stdout.strip()
        if "JobState=PENDING" not in state or "Reason=JobHeldUser" not in state:
            raise ValueError("P2R6 Wave-B held task state differs")
        observed.append(
            {
                "task_id": task_id,
                "held_inspection_sha256": hashlib.sha256(state.encode()).hexdigest(),
            }
        )
    command = "release" if action == "release" else "cancel"
    executable = "scontrol" if action == "release" else "scancel"
    args = [executable]
    if action == "release":
        args.append(command)
    args.extend(item["task_id"] for item in observed)
    _run(args)
    receipt = {
        "schema": "ode-edit-s05-p2r6-red-r2-phase2-wave-b-control/v1",
        "source_head": head,
        "phase2_job_id": job_id,
        "selected_controller": gate["selected_controller"],
        "wave_a_gate_status": status,
        "wave_a_gate_receipt_sha256": gate_sha,
        "action": action,
        "wave_b_indices": list(WAVE_B_INDICES),
        "held_inspection": observed,
        "method_hparam_change": 0,
    }
    destination = STATE_ROOT / f"phase2-{job_id}-wave-b-{action}.json"
    receipt_sha = _write_once(destination, receipt)
    return {**receipt, "receipt_path": str(destination), "receipt_sha256": receipt_sha}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--action", required=True, choices=("release", "cancel"))
    parser.add_argument("--gate-receipt", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            control(job_id=args.job_id, action=args.action, gate_receipt=args.gate_receipt),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
