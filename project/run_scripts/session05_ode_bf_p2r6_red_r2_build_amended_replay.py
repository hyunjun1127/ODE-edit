#!/usr/bin/env python3
"""Create-once B2-amended CPU replay from the sealed legacy P2R6 capsule."""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p2r6_semantic_region_controller import (
    P2R6_NUMERICAL_EPSILON,
    _semantic_region_optimum,
)


FIXED_MEMBERS = ("S", "d", "semantic_scale", "Q_C", "c_C", "mass_matrix")
REGENERATED_MEMBERS = ("alpha_start", "b")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes_once(path: Path, raw: bytes) -> str:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(raw).hexdigest()


def _write_npy_once(path: Path, value: np.ndarray) -> dict[str, object]:
    canonical = np.ascontiguousarray(value, dtype=np.dtype("<f8"))
    buffer = BytesIO()
    np.save(buffer, canonical, allow_pickle=False)
    raw = buffer.getvalue()
    return {
        "filename": path.name,
        "shape": list(canonical.shape),
        "dtype": canonical.dtype.str,
        "finite": bool(np.all(np.isfinite(canonical))),
        "npy_bytes": len(raw),
        "npy_sha256": _write_bytes_once(path, raw),
        "raw_c_order_sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
    }


def build(legacy_root: Path, destination: Path, source_head: str) -> dict[str, object]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("P2R6 amended replay destination exists")
    legacy_raw = (legacy_root / "capsule.json").read_bytes()
    legacy = json.loads(legacy_raw)
    if (
        legacy.get("identity_sha256")
        != "c03ed8d79bb79b31cadf117d95a2ae40e66f8ce8f60e6304e82bf106bf6e3b17"
        or legacy.get("capsule_complete") is not True
    ):
        raise ValueError("P2R6 legacy capsule identity differs")
    arrays = {
        name: np.load(legacy_root / f"{name}.npy", allow_pickle=False)
        for name in (*FIXED_MEMBERS, *REGENERATED_MEMBERS)
    }
    for name in FIXED_MEMBERS:
        expected = legacy["members"][name]
        source = legacy_root / f"{name}.npy"
        if source.is_symlink() or _sha256(source) != expected["npy_sha256"]:
            raise ValueError(f"P2R6 fixed legacy member differs: {name}")
    alpha_start, semantic_scale, b_new, xi_new, e1_receipt = (
        _semantic_region_optimum(
            arrays["S"],
            arrays["d"],
            arrays["d"],
            scale_policy="CURRENT_DEFICIT",
        )
    )
    semantic_response = arrays["S"] @ alpha_start
    mass = arrays["mass_matrix"] @ alpha_start
    exact_feasibility = {
        "max_b_minus_semantic_response": max(
            0.0, float(np.max(b_new - semantic_response))
        ),
        "mass_excess": max(0.0, float(np.max(mass - 1.0))),
        "negative_alpha_violation": max(0.0, -float(np.min(alpha_start))),
        "certificate_tolerance": P2R6_NUMERICAL_EPSILON,
    }
    if max(exact_feasibility.values()) > P2R6_NUMERICAL_EPSILON:
        raise ValueError("P2R6 amended E1 exact-feasibility differs")
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.chmod(destination, 0o700)
    members: dict[str, dict[str, object]] = {}
    for name in FIXED_MEMBERS:
        source = legacy_root / f"{name}.npy"
        target = destination / source.name
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        observed = dict(legacy["members"][name])
        observed["fixed_source_npy_sha256"] = observed["npy_sha256"]
        observed["destination_npy_sha256"] = _sha256(target)
        observed["byte_identity_pass"] = (
            observed["fixed_source_npy_sha256"]
            == observed["destination_npy_sha256"]
        )
        members[name] = observed
    members["alpha_start"] = _write_npy_once(
        destination / "alpha_start.npy", alpha_start
    )
    members["b"] = _write_npy_once(destination / "b.npy", b_new)
    payload: dict[str, object] = {
        "schema": "ode-edit-s05-p2r6-red-r2-b2-amended-private-replay/v1",
        "instruction_id": "ODEEDIT-S05-P2R6-RED-R2-FINAL-SCIENTIFIC-RUN-V1-B1-AMENDED-REPLAY-R1",
        "status": "B2_AMENDED_REPLAY_EXACT_FEASIBLE",
        "source_head": source_head,
        "legacy_capsule_json_sha256": hashlib.sha256(legacy_raw).hexdigest(),
        "legacy_capsule_identity_sha256": legacy["identity_sha256"],
        "legacy_equation_status": "LEGACY_EQUATION_EXACT_INFEASIBLE",
        "fixed_member_names": list(FIXED_MEMBERS),
        "regenerated_member_names": list(REGENERATED_MEMBERS),
        "fixed_member_byte_identity_all": all(
            bool(members[name]["byte_identity_pass"]) for name in FIXED_MEMBERS
        ),
        "members": members,
        "e1_old_xi": float(legacy["e1_xi"]),
        "e1_new_xi": float(xi_new),
        "b_old_new_max_abs_difference": float(
            np.max(np.abs(arrays["b"] - b_new))
        ),
        "e1_receipt": e1_receipt,
        "exact_feasibility": exact_feasibility,
        "feasible_set_rhs_envelope": 0.0,
        "capture_invocation_count": 0,
        "model_gpu_slurm_action_count": 0,
        "private_tensor_artifact": True,
        "raw_prompt_target_generation_weight_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    capsule_sha = _write_bytes_once(
        destination / "capsule.json",
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n",
    )
    return {
        "status": payload["status"],
        "destination": str(destination),
        "capsule_sha256": capsule_sha,
        "capsule_identity_sha256": payload["identity_sha256"],
        "e1_new_xi": xi_new,
        "b_old_new_max_abs_difference": payload["b_old_new_max_abs_difference"],
        "exact_feasibility": exact_feasibility,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--legacy-root", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.legacy_root, args.destination, args.source_head),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
