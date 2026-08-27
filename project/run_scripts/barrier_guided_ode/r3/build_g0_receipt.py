#!/usr/bin/env python3
"""Build the deterministic BGODE-R3 G0 source manifest and rooted receipt."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
OUTPUT = REPO / "audits/servers/server1/2026-08-27-bgode-r3-g0-v1"
MANIFEST = OUTPUT / "source-manifest.json"
RECEIPT = OUTPUT / "rooted-receipt.json"
MEMBERS = (
    "audits/servers/server1/2026-08-27-bgode-r3-g0-v1/bgode-r3-method-spec.md",
    "project/run_scripts/barrier_guided_ode/r3/__init__.py",
    "project/run_scripts/barrier_guided_ode/r3/actuators.py",
    "project/run_scripts/barrier_guided_ode/r3/build_g0_receipt.py",
    "project/run_scripts/barrier_guided_ode/r3/dynamics.py",
    "project/run_scripts/barrier_guided_ode/r3/errors.py",
    "project/run_scripts/barrier_guided_ode/r3/events.py",
    "project/run_scripts/barrier_guided_ode/r3/moments.py",
    "project/run_scripts/barrier_guided_ode/r3/reference.py",
    "project/run_scripts/barrier_guided_ode/r3/solver.py",
    "project/run_scripts/barrier_guided_ode/r3/telemetry.py",
    "project/run_scripts/barrier_guided_ode/r3/locks/bgode-r3-g0-numerical-lock.json",
    "project/run_scripts/barrier_guided_ode/r3/tests/__init__.py",
    "project/run_scripts/barrier_guided_ode/r3/tests/test_dynamics_firewall.py",
    "project/run_scripts/barrier_guided_ode/r3/tests/test_events_reference.py",
    "project/run_scripts/barrier_guided_ode/r3/tests/test_solver_actuators.py",
    "audits/servers/server1/2026-08-27-bgode-r3-g0-v1/bgode-r3-g0-factual-ko.md",
    "audits/servers/server1/2026-08-27-bgode-r3-g0-v1/bgode-r3-g0-red-team.md",
    "audits/servers/server1/2026-08-27-bgode-r3-g0-v1/g0-summary.json",
)


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record(relative: str) -> dict[str, object]:
    path = REPO / relative
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode) or path.is_symlink():
        raise SystemExit(f"member is not a regular non-symlink: {relative}")
    data = path.read_bytes()
    return {
        "bytes": len(data),
        "mode": format(stat.S_IMODE(status.st_mode), "04o"),
        "path": relative,
        "sha256": sha256(data),
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
    members_root = sha256(canonical(records))
    manifest = {
        "member_count": len(records),
        "members": records,
        "members_root": members_root,
        "registry_id": "BGODE-R3",
        "schema": "BGODE_R3_G0_SOURCE_MANIFEST_V1",
    }
    manifest_bytes = canonical(manifest)
    receipt = {
        "action_counts": {"gpu": 0, "model": 0, "slurm": 0},
        "authoritative_contract": {
            "bytes": 25746,
            "logical_records": 1089,
            "newline_count": 1088,
            "sha256": "fa0a6923efe944d485dc49802772b4d08f7d02f7c7113dbee302e7d9744642dc",
            "terminal_newline": False,
        },
        "base_head": "8e60f5c2d8de998f128a20e25c09f66aff770597",
        "base_tree": "bc9aae754dec2ea6a55dcf23c2b23d4e644bb299",
        "focused_tests": {
            "legacy_r2_failed": 0,
            "legacy_r2_passed": 21,
            "r3_failed": 0,
            "r3_passed": 28,
        },
        "manifest_sha256": sha256(manifest_bytes),
        "members_root": members_root,
        "registry_id": "BGODE-R3",
        "schema": "BGODE_R3_G0_ROOTED_RECEIPT_V1",
        "scientific_promotion": False,
        "stage": "G0_CPU_MATH_CLOSURE",
        "status": "R3_G0_MATHEMATICAL_PASS_G1_NOT_RUN",
    }
    receipt["receipt_identity"] = sha256(canonical(receipt))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_once_or_exact(MANIFEST, manifest_bytes)
    write_once_or_exact(RECEIPT, canonical(receipt))
    print(
        canonical(
            {
                "manifest_sha256": receipt["manifest_sha256"],
                "members_root": members_root,
                "receipt_identity": receipt["receipt_identity"],
            }
        ).decode("utf-8"),
        end="",
    )


if __name__ == "__main__":
    main()
