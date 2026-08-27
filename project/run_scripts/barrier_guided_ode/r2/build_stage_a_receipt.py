#!/usr/bin/env python3
"""Create the deterministic BGODE-R2 Stage-A source manifest and receipt."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
OUTPUT_DIRECTORY = REPO / "audits/servers/server1"
MANIFEST_PATH = OUTPUT_DIRECTORY / "2026-08-27-bgode-r2-stage-a-source-manifest.json"
RECEIPT_PATH = OUTPUT_DIRECTORY / "2026-08-27-bgode-r2-stage-a-rooted-receipt.json"
MEMBERS = (
    "project/proposals/sections/05-bgode-r2-rho-free-prefix-event-fp64-two-equality.md",
    "audits/servers/server1/2026-08-27-bgode-r2-stage-a-red-team.md",
    "project/run_scripts/barrier_guided_ode/r2/__init__.py",
    "project/run_scripts/barrier_guided_ode/r2/actuator_guard.py",
    "project/run_scripts/barrier_guided_ode/r2/controller.py",
    "project/run_scripts/barrier_guided_ode/r2/dynamics.py",
    "project/run_scripts/barrier_guided_ode/r2/errors.py",
    "project/run_scripts/barrier_guided_ode/r2/events.py",
    "project/run_scripts/barrier_guided_ode/r2/moments.py",
    "project/run_scripts/barrier_guided_ode/r2/telemetry.py",
    "project/run_scripts/barrier_guided_ode/r2/tests/__init__.py",
    "project/run_scripts/barrier_guided_ode/r2/tests/test_dynamics_provenance.py",
    "project/run_scripts/barrier_guided_ode/r2/tests/test_moments_controller.py",
    "project/run_scripts/barrier_guided_ode/r2/tests/test_prefix_events.py",
    "project/run_scripts/barrier_guided_ode/r2/locks/bgode-r2-stage-a-numerical-lock.json",
    "project/run_scripts/barrier_guided_ode/r2/build_stage_a_receipt.py"
)


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def record(relative: str) -> dict[str, object]:
    path = REPO / relative
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode) or path.is_symlink():
        raise SystemExit(f"member is not a regular non-symlink: {relative}")
    data = path.read_bytes()
    return {
        "path": relative,
        "bytes": len(data),
        "mode": format(stat.S_IMODE(status.st_mode), "04o"),
        "sha256": digest(data),
    }


def write_once_or_exact(path: Path, data: bytes) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise SystemExit(f"existing output differs: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def main() -> None:
    records = [record(relative) for relative in sorted(MEMBERS)]
    members_root = digest(canonical(records))
    manifest = {
        "schema": "BGODE_R2_STAGE_A_SOURCE_MANIFEST_V1",
        "registry_id": "BGODE-R2",
        "member_count": len(records),
        "members_root": members_root,
        "members": records,
    }
    manifest_bytes = canonical(manifest)
    receipt = {
        "schema": "BGODE_R2_STAGE_A_ROOTED_RECEIPT_V1",
        "registry_id": "BGODE-R2",
        "authoritative_contract_sha256": "410c42fa90dc6351d2ccb90367a024633d22a4fb57a446dabbdabf65d6f88ab4",
        "authoritative_contract_bytes": 22028,
        "authoritative_contract_lines": 1085,
        "stage": "A_CPU_MATH_CLOSURE",
        "focused_tests": {"passed": 21, "failed": 0},
        "model_action_count": 0,
        "gpu_action_count": 0,
        "slurm_action_count": 0,
        "scientific_promotion": False,
        "manifest_sha256": digest(manifest_bytes),
        "members_root": members_root,
    }
    receipt["receipt_identity"] = digest(canonical(receipt))
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_once_or_exact(MANIFEST_PATH, manifest_bytes)
    write_once_or_exact(RECEIPT_PATH, canonical(receipt))


if __name__ == "__main__":
    main()
