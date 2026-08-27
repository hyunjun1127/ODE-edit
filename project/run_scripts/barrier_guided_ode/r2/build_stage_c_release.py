#!/usr/bin/env python3
"""Build create-once BGODE-R2 Stage-C numerical and source locks."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
LOCK_PATH = REPO / "project/run_scripts/barrier_guided_ode/r2/locks/bgode-r2-stage-c-numerical-lock.json"
MANIFEST_PATH = REPO / "project/run_scripts/barrier_guided_ode/r2/locks/bgode-r2-stage-c-source-manifest.json"
TOPOLOGY_PATH = REPO / "project/run_scripts/barrier_guided_ode/r2/locks/bgode-r2-stage-c-topology-manifest.json"
MEMBERS = (
    "project/run_scripts/barrier_guided_ode/alphaedit_actuator_interface.py",
    "project/run_scripts/barrier_guided_ode/s1_alphaedit_runtime.py",
    "project/run_scripts/barrier_guided_ode/s1_contract.py",
    "project/run_scripts/barrier_guided_ode/s1_experiment.py",
    "project/run_scripts/barrier_guided_ode/r2/actuator_guard.py",
    "project/run_scripts/barrier_guided_ode/r2/controller.py",
    "project/run_scripts/barrier_guided_ode/r2/errors.py",
    "project/run_scripts/barrier_guided_ode/r2/events.py",
    "project/run_scripts/barrier_guided_ode/r2/moments.py",
    "project/run_scripts/barrier_guided_ode/r2/stage_b_contract.py",
    "project/run_scripts/barrier_guided_ode/r2/stage_b_probe.py",
    "project/run_scripts/barrier_guided_ode/r2/stage_c_contract.py",
    "project/run_scripts/barrier_guided_ode/r2/build_stage_c_topology_manifest.py",
    "project/run_scripts/barrier_guided_ode/r2/build_stage_c_release.py",
    "project/run_scripts/barrier_guided_ode/r2/tests/test_stage_c_topology.py",
    "project/run_scripts/session05_bgode_r2_stage_b.py",
    "project/run_scripts/session05_bgode_r2_stage_c.py",
    "project/run_scripts/session05_bgode_r2_stage_c.sbatch",
)


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_once(path: Path, payload: dict[str, object]) -> None:
    encoded = (canonical(payload) + "\n").encode()
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise SystemExit(f"existing Stage-C lock differs: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)


def main() -> None:
    topology = json.loads(TOPOLOGY_PATH.read_text())
    topology_identity = topology["identity"]
    topology_sha = digest(TOPOLOGY_PATH)
    mapping = {
        "0": "llama3-8b-inst/both-multi-unequal-nonprefix",
        "1": "llama3-8b-inst/source-prefix-target",
        "2": "llama3-8b-inst/target-prefix-source",
        "3": "qwen2.5-7b-inst/both-multi-unequal-nonprefix",
        "4": "qwen2.5-7b-inst/source-prefix-target",
        "5": "qwen2.5-7b-inst/target-prefix-source",
    }
    body: dict[str, object] = {
        "schema": "ode-edit-bgode-r2-stage-c-numerical-lock/v1",
        "instruction_id": "ODEEDIT-S05-BGODE-R2-RHO-FREE-PREFIX-EVENT-FP64-TWO-EQUALITY",
        "registry_id": "BGODE-R2",
        "array_mapping": mapping,
        "array_throttle": 3,
        "project_gpu_cap": 3,
        "topology_manifest_sha256": topology_sha,
        "topology_manifest_identity": topology_identity,
        "selection_influence": topology["selection_influence"],
        "sample_identity": topology["sample_identity"],
        "termination_token_count": 0,
        "model_forward_jvp_dtype": "torch.float32",
        "event_controller_dtype": "torch.float64",
        "physical_write_dtype": "torch.float32",
        "controller": "FISHER_ONLY_TWO_EQUALITY_TOPOLOGY_GATE",
        "equality_rates": [1.0, 0.0],
        "tolerance_policy": "dtype-dimension-norm-backward-error-only",
        "dynamic_writer_action_count": 0,
        "fixed_target_compute_count": 1,
        "fixed_target_recompute_count": 0,
        "node_history_append_count": 0,
        "scalar_localizer_count": 0,
        "probability_floor_count": 0,
        "ridge_count": 0,
        "damping_count": 0,
        "fallback_count": 0,
        "scientific_promotion": False,
    }
    lock = dict(body)
    lock["identity"] = hashlib.sha256(canonical(body).encode()).hexdigest()
    write_once(LOCK_PATH, lock)
    records = []
    for relative in sorted((*MEMBERS, str(LOCK_PATH.relative_to(REPO)), str(TOPOLOGY_PATH.relative_to(REPO)))):
        path = REPO / relative
        status = path.lstat()
        if not stat.S_ISREG(status.st_mode) or path.is_symlink():
            raise SystemExit(f"Stage-C source member is not regular: {relative}")
        records.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "mode": format(stat.S_IMODE(status.st_mode), "04o"),
            "sha256": digest(path),
        })
    manifest = {
        "schema": "ode-edit-bgode-r2-stage-c-source-manifest/v1",
        "instruction_id": body["instruction_id"],
        "member_count": len(records),
        "members_root": hashlib.sha256(canonical(records).encode()).hexdigest(),
        "members": records,
    }
    write_once(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
