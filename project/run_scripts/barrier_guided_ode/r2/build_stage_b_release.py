#!/usr/bin/env python3
"""Build create-once BGODE-R2 Stage-B numerical and source locks."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
LOCK_PATH = REPO / "project/run_scripts/barrier_guided_ode/r2/locks/bgode-r2-stage-b-numerical-lock.json"
MANIFEST_PATH = REPO / "project/run_scripts/barrier_guided_ode/r2/locks/bgode-r2-stage-b-source-manifest.json"
MEMBERS = (
    "project/run_scripts/barrier_guided_ode/alphaedit_actuator_interface.py",
    "project/run_scripts/barrier_guided_ode/s1_alphaedit_runtime.py",
    "project/run_scripts/barrier_guided_ode/s1_contract.py",
    "project/run_scripts/barrier_guided_ode/s1_experiment.py",
    "project/run_scripts/barrier_guided_ode/r2/__init__.py",
    "project/run_scripts/barrier_guided_ode/r2/actuator_guard.py",
    "project/run_scripts/barrier_guided_ode/r2/controller.py",
    "project/run_scripts/barrier_guided_ode/r2/errors.py",
    "project/run_scripts/barrier_guided_ode/r2/events.py",
    "project/run_scripts/barrier_guided_ode/r2/moments.py",
    "project/run_scripts/barrier_guided_ode/r2/stage_b_contract.py",
    "project/run_scripts/barrier_guided_ode/r2/stage_b_probe.py",
    "project/run_scripts/barrier_guided_ode/r2/telemetry.py",
    "project/run_scripts/barrier_guided_ode/r2/build_stage_b_release.py",
    "project/run_scripts/barrier_guided_ode/r2/tests/test_stage_b_pre_gpu.py",
    "project/run_scripts/session05_bgode_r2_stage_b.py",
    "project/run_scripts/session05_bgode_r2_stage_b.sbatch",
)


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


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
    lock_body: dict[str, object] = {
        "schema": "ode-edit-bgode-r2-stage-b-numerical-lock/v1",
        "instruction_id": "ODEEDIT-S05-BGODE-R2-RHO-FREE-PREFIX-EVENT-FP64-TWO-EQUALITY",
        "registry_id": "BGODE-R2",
        "models": ["llama3-8b-inst", "qwen2.5-7b-inst"],
        "array_mapping": {"0": "llama3-8b-inst", "1": "qwen2.5-7b-inst"},
        "array_throttle": 2,
        "project_gpu_cap": 3,
        "canonical_stream_sha256": "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a",
        "canonical_order_sha256": "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3",
        "sample_selection": "B1_FIRST_REQUEST_ORDINAL_ZERO_OUTCOME_INDEPENDENT",
        "batch_size": 1,
        "termination_token_count": 0,
        "model_forward_jvp_dtype": "torch.float32",
        "event_controller_dtype": "torch.float64",
        "physical_write_dtype": "torch.float32",
        "controller": "FISHER_ONLY_TWO_EQUALITY_NODE0",
        "equality_rates": [1.0, 0.0],
        "tolerance_policy": "dtype-dimension-norm-backward-error-only",
        "fd_epsilon": 0.00390625,
        "fd_absolute_tolerance": 0.02,
        "fd_relative_tolerance": 0.08,
        "dynamic_writer_action_count": 0,
        "fixed_target_compute_count": 1,
        "fixed_target_recompute_count": 0,
        "node_history_append_count": 0,
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
    for relative in sorted((*MEMBERS, str(LOCK_PATH.relative_to(REPO)))):
        path = REPO / relative
        status = path.lstat()
        if not stat.S_ISREG(status.st_mode) or path.is_symlink():
            raise SystemExit(f"source member is not regular: {relative}")
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "mode": format(stat.S_IMODE(status.st_mode), "04o"),
                "sha256": _digest(path),
            }
        )
    manifest = {
        "schema": "ode-edit-bgode-r2-stage-b-source-manifest/v1",
        "instruction_id": lock_body["instruction_id"],
        "member_count": len(records),
        "members_root": sha256_bytes(canonical_json(records).encode("utf-8")),
        "members": records,
    }
    _write_once_or_exact(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
