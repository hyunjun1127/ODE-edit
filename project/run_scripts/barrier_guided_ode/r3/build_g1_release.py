#!/usr/bin/env python3
"""Build create-once BGODE-R3 G1 source and numerical locks."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
LOCK_PATH = REPO / "project/run_scripts/barrier_guided_ode/r3/locks/bgode-r3-g1-numerical-lock.json"
MANIFEST_PATH = REPO / "project/run_scripts/barrier_guided_ode/r3/locks/bgode-r3-g1-source-manifest.json"
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
    "project/run_scripts/barrier_guided_ode/r3/build_g1_natural_manifest.py",
    "project/run_scripts/barrier_guided_ode/r3/build_g1_release.py",
    "project/run_scripts/barrier_guided_ode/r3/tests/test_g1_pre_gpu.py",
    "project/run_scripts/session05_bgode_r3_g1.py",
    "project/run_scripts/session05_bgode_r3_g1.sbatch",
)


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once_or_exact(path: Path, payload: dict[str, object]) -> None:
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise SystemExit(f"existing lock differs: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)


def main() -> None:
    natural = json.loads(NATURAL_PATH.read_text(encoding="utf-8"))
    lock_body: dict[str, object] = {
        "schema": "ode-edit-bgode-r3-g1-numerical-lock/v1",
        "instruction_id": "ODEEDIT-S05-BGODE-R3-FINE-EVENT-TARGET-EXCLUDED-FACTOR-SPACE",
        "registry_id": "BGODE-R3",
        "models": ["llama3-8b-inst", "qwen2.5-7b-inst"],
        "array_mapping": {"0": "llama3-8b-inst", "1": "qwen2.5-7b-inst"},
        "array_throttle": 2,
        "project_gpu_cap": 3,
        "batch_size": 1,
        "canonical_stream_sha256": "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a",
        "canonical_order_sha256": "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3",
        "natural_topology_manifest_identity": natural["identity"],
        "natural_selection": "EARLIEST_UNEQUAL_NONPREFIX_WITH_AT_LEAST_ONE_MULTITOKEN_CONTINUATION",
        "natural_ordinal": 26,
        "natural_case_id": "17454",
        "unavailable_topology_policy": "NATURAL_TOPOLOGY_UNAVAILABLE_NO_SYNTHETIC_REPLACEMENT",
        "model_forward_jvp_dtype": "torch.float32",
        "event_controller_dtype": "torch.float64",
        "physical_write_dtype": "torch.float32",
        "actuator_normalization": "FACTOR_GRAM_FROBENIUS_FP64",
        "equality_rates": [1.0, 0.0],
        "controller": "T0_FISHER_FULL_IDENTITY_THEN_FISHER_PROBE",
        "h_probe": "T_AE/32",
        "fd_scope": "ALL_INTERNAL_PREFIXES_ALL_FIVE_NORMALIZED_ACTUATORS_AND_RETAINED_MODES",
        "fd_epsilon": 0.00390625,
        "fd_absolute_tolerance": 0.02,
        "fd_relative_tolerance": 0.08,
        "solver_rcond": "max(m,n)*eps_FP32",
        "fixed_target_compute_count": 1,
        "fixed_target_recompute_count": 0,
        "node_history_append_count": 0,
        "terminal_history_append_count": 0,
        "probability_floor_count": 0,
        "ridge_count": 0,
        "damping_count": 0,
        "fallback_count": 0,
        "scientific_promotion": False,
    }
    lock = dict(lock_body)
    lock["identity"] = sha256_bytes(canonical_json(lock_body).encode("utf-8"))
    _write_once_or_exact(LOCK_PATH, lock)

    records: list[dict[str, object]] = []
    for relative in sorted((*MEMBERS, str(LOCK_PATH.relative_to(REPO)), str(NATURAL_PATH.relative_to(REPO)))):
        path = REPO / relative
        status = path.lstat()
        if not stat.S_ISREG(status.st_mode) or path.is_symlink():
            raise SystemExit(f"source member is not regular: {relative}")
        records.append(
            {
                "bytes": path.stat().st_size,
                "mode": format(stat.S_IMODE(status.st_mode), "04o"),
                "path": relative,
                "sha256": _digest(path),
            }
        )
    manifest = {
        "schema": "ode-edit-bgode-r3-g1-source-manifest/v1",
        "instruction_id": lock_body["instruction_id"],
        "member_count": len(records),
        "members": records,
        "members_root": sha256_bytes(canonical_json(records).encode("utf-8")),
    }
    _write_once_or_exact(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
