#!/usr/bin/env python3
"""Build the create-once BGODE-R3 G1 TECH-R1 serialization repair locks."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
LOCK_PATH = REPO / "project/run_scripts/barrier_guided_ode/r3/locks/bgode-r3-g1-tech-r1-numerical-lock.json"
MANIFEST_PATH = REPO / "project/run_scripts/barrier_guided_ode/r3/locks/bgode-r3-g1-tech-r1-source-manifest.json"
NATURAL_PATH = REPO / "project/run_scripts/barrier_guided_ode/r3/locks/bgode-r3-g1-natural-topology-manifest.json"
MEMBERS = (
    "project/run_scripts/barrier_guided_ode/alphaedit_actuator_interface.py",
    "project/run_scripts/barrier_guided_ode/s1_alphaedit_runtime.py",
    "project/run_scripts/barrier_guided_ode/r2/actuator_guard.py",
    "project/run_scripts/barrier_guided_ode/r3/actuators.py",
    "project/run_scripts/barrier_guided_ode/r3/errors.py",
    "project/run_scripts/barrier_guided_ode/r3/events.py",
    "project/run_scripts/barrier_guided_ode/r3/g1_jvp.py",
    "project/run_scripts/barrier_guided_ode/r3/g1_probe.py",
    "project/run_scripts/barrier_guided_ode/r3/moments.py",
    "project/run_scripts/barrier_guided_ode/r3/natural.py",
    "project/run_scripts/barrier_guided_ode/r3/reference.py",
    "project/run_scripts/barrier_guided_ode/r3/solver.py",
    "project/run_scripts/barrier_guided_ode/r3/telemetry.py",
    "project/run_scripts/barrier_guided_ode/r3/tests/test_g1_pre_gpu.py",
    "project/run_scripts/session05_bgode_r3_g1.py",
    "project/run_scripts/session05_bgode_r3_g1.sbatch",
)


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_once(path: Path, value: object) -> None:
    data = canonical(value) + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise SystemExit(f"existing TECH-R1 lock differs: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def main() -> None:
    natural = json.loads(NATURAL_PATH.read_text())
    body = {
        "schema": "ode-edit-bgode-r3-g1-tech-r1-numerical-lock/v1",
        "instruction_id": "ODEEDIT-S05-BGODE-R3-FINE-EVENT-TARGET-EXCLUDED-FACTOR-SPACE",
        "technical_repair": "OBSERVATION_ONLY_TENSOR_RECEIPT_JSON_SERIALIZATION",
        "superseded_job": 26885,
        "science_change_count": 0,
        "sample_change_count": 0,
        "tolerance_change_count": 0,
        "writer_change_count": 0,
        "array_mapping": {"0": "llama3-8b-inst", "1": "qwen2.5-7b-inst"},
        "array_throttle": 2,
        "project_gpu_cap": 3,
        "natural_topology_manifest_identity": natural["identity"],
        "natural_ordinal": 26,
        "natural_case_id": "17454",
        "model_forward_jvp_dtype": "torch.float32",
        "event_controller_dtype": "torch.float64",
        "physical_write_dtype": "torch.float32",
        "equality_rates": [1.0, 0.0],
        "h_probe": "T_AE/32",
        "fd_epsilon": 0.00390625,
        "fd_absolute_tolerance": 0.02,
        "fd_relative_tolerance": 0.08,
        "solver_rcond": "max(m,n)*eps_FP32",
        "scientific_promotion": False,
    }
    lock = dict(body)
    lock["identity"] = hashlib.sha256(canonical(body)).hexdigest()
    write_once(LOCK_PATH, lock)
    records = []
    for relative in sorted((*MEMBERS, str(NATURAL_PATH.relative_to(REPO)), str(LOCK_PATH.relative_to(REPO)))):
        path = REPO / relative
        status = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(status.st_mode):
            raise SystemExit(f"TECH-R1 member is not regular: {relative}")
        records.append({
            "bytes": path.stat().st_size,
            "mode": format(stat.S_IMODE(status.st_mode), "04o"),
            "path": relative,
            "sha256": digest(path),
        })
    write_once(MANIFEST_PATH, {
        "schema": "ode-edit-bgode-r3-g1-tech-r1-source-manifest/v1",
        "member_count": len(records),
        "members": records,
        "members_root": hashlib.sha256(canonical(records)).hexdigest(),
    })


if __name__ == "__main__":
    main()
