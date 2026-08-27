#!/usr/bin/env python3
"""Build create-once BGODE-R3 G2 Cauchy and source locks."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
LOCK_PATH = REPO / "project/run_scripts/barrier_guided_ode/r3/locks/bgode-r3-g2-cauchy-numerical-lock.json"
MANIFEST_PATH = REPO / "project/run_scripts/barrier_guided_ode/r3/locks/bgode-r3-g2-source-manifest.json"
NATURAL_PATH = REPO / "project/run_scripts/barrier_guided_ode/r3/locks/bgode-r3-g1-natural-topology-manifest.json"
G1_RECEIPT_PATH = REPO / "audits/servers/server1/2026-08-27-bgode-r3-g1-terminal-v1/rooted-receipt.json"
MEMBERS = (
    "project/run_scripts/barrier_guided_ode/alphaedit_actuator_interface.py",
    "project/run_scripts/barrier_guided_ode/s1_alphaedit_runtime.py",
    "project/run_scripts/barrier_guided_ode/r2/actuator_guard.py",
    "project/run_scripts/barrier_guided_ode/r3/actuators.py",
    "project/run_scripts/barrier_guided_ode/r3/dynamics.py",
    "project/run_scripts/barrier_guided_ode/r3/errors.py",
    "project/run_scripts/barrier_guided_ode/r3/events.py",
    "project/run_scripts/barrier_guided_ode/r3/g1_jvp.py",
    "project/run_scripts/barrier_guided_ode/r3/g1_probe.py",
    "project/run_scripts/barrier_guided_ode/r3/g2_convergence.py",
    "project/run_scripts/barrier_guided_ode/r3/g2_experiment.py",
    "project/run_scripts/barrier_guided_ode/r3/g2_trajectory.py",
    "project/run_scripts/barrier_guided_ode/r3/moments.py",
    "project/run_scripts/barrier_guided_ode/r3/natural.py",
    "project/run_scripts/barrier_guided_ode/r3/reference.py",
    "project/run_scripts/barrier_guided_ode/r3/solver.py",
    "project/run_scripts/barrier_guided_ode/r3/telemetry.py",
    "project/run_scripts/barrier_guided_ode/r3/tests/test_dynamics_firewall.py",
    "project/run_scripts/barrier_guided_ode/r3/tests/test_events_reference.py",
    "project/run_scripts/barrier_guided_ode/r3/tests/test_g2_convergence.py",
    "project/run_scripts/barrier_guided_ode/r3/tests/test_g2_pre_gpu.py",
    "project/run_scripts/barrier_guided_ode/r3/tests/test_solver_actuators.py",
    "project/run_scripts/session05_bgode_r3_g2.py",
    "project/run_scripts/session05_bgode_r3_g2.sbatch",
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
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise SystemExit(f"existing G2 lock differs: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def main() -> None:
    natural = json.loads(NATURAL_PATH.read_text())
    g1 = json.loads(G1_RECEIPT_PATH.read_text())
    body = {
        "schema": "ode-edit-bgode-r3-g2-cauchy-numerical-lock/v1",
        "instruction_id": "ODEEDIT-S05-BGODE-R3-FINE-EVENT-TARGET-EXCLUDED-FACTOR-SPACE",
        "g1_terminal_receipt_identity": g1["receipt_identity"],
        "g1_terminal_status": g1["status"],
        "g1_terminal_identities": g1["terminal_identities"],
        "natural_topology_manifest_identity": natural["identity"],
        "natural_ordinal": 26,
        "natural_case_id": "17454",
        "natural_topology": "unequal-non-prefix",
        "synthetic_replacement_count": 0,
        "array_mapping": {"0": "llama3-8b-inst", "1": "qwen2.5-7b-inst"},
        "array_throttle": 2,
        "gpu_per_task": 1,
        "project_gpu_cap": 3,
        "batch_size": 1,
        "arms": [
            "PLAIN_NORMALIZED_EUCLIDEAN",
            "FISHER_FACTOR_SPACE",
            "FULL_TARGET_EXCLUDED_BARRIER",
        ],
        "n_grid": [4, 8, 16, 32],
        "panel_count_per_model": 12,
        "horizon": "ENTRY_MATCHED_OFFICIAL_ALPHAEDIT_TARGET_LOGIT_T_AE",
        "step_size": "T_AE/N",
        "cauchy_pairs": [[4, 8], [8, 16], [16, 32]],
        "physical_endpoint_distance": "ACTUAL_DENSE_BLOCK_W_2N_MINUS_W_N_FROBENIUS",
        "physical_contraction_required": True,
        "target_logit_gap_contraction_required": True,
        "q0_conditional_kl_gap_contraction_required": True,
        "first_order_physical_ratio_floor": 1.25,
        "final_relative_physical_distance_ceiling": 0.05,
        "final_normalized_target_logit_gap_ceiling": 0.05,
        "final_normalized_q_kl_gap_ceiling": 0.05,
        "rounding_tolerance": "256*eps_FP32*(1+observed_scalar_scale)",
        "solver_rcond": "max(m,n)*eps_FP32",
        "fd_epsilon": 0.00390625,
        "fd_absolute_tolerance": 0.02,
        "fd_relative_tolerance": 0.08,
        "all_internal_prefix_fd_each_node": True,
        "model_forward_jvp_dtype": "torch.float32",
        "event_controller_dtype": "torch.float64",
        "physical_write_dtype": "torch.float32",
        "future_fixed_T_curve_predeclared_not_run": {"T": [0.5, 1.0, 2.0, 3.0, 5.0], "N": 32},
        "history_append_count": 0,
        "localizer_root_ridge_damping_floor_fallback_count": 0,
        "endpoint_selection_influence_count": 0,
        "scientific_promotion": False,
    }
    lock = dict(body)
    lock["identity"] = hashlib.sha256(canonical(body)).hexdigest()
    write_once(LOCK_PATH, lock)
    records = []
    for relative in sorted(
        (*MEMBERS, str(NATURAL_PATH.relative_to(REPO)), str(LOCK_PATH.relative_to(REPO)))
    ):
        path = REPO / relative
        status = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(status.st_mode):
            raise SystemExit(f"G2 source member is not regular: {relative}")
        records.append(
            {
                "bytes": path.stat().st_size,
                "mode": format(stat.S_IMODE(status.st_mode), "04o"),
                "path": relative,
                "sha256": digest(path),
            }
        )
    write_once(
        MANIFEST_PATH,
        {
            "schema": "ode-edit-bgode-r3-g2-source-manifest/v1",
            "member_count": len(records),
            "members": records,
            "members_root": hashlib.sha256(canonical(records)).hexdigest(),
        },
    )


if __name__ == "__main__":
    main()
