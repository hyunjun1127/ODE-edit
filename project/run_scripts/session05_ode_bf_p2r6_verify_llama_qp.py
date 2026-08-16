#!/usr/bin/env python3
"""Fresh-process verifier for the sealed private Llama outer1 QP capsule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.optimize import linprog

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p2r6_certified_convex_qp import (
    P2R6CertifiedQPError,
    solve_certified_semantic_region_qp,
)


EXPECTED_CAPSULE_SHA256 = "18481bd9aa237009108e5638582f809034888e72b0d152a7e2fa83e61bfac0af"
EXPECTED_CAPSULE_IDENTITY = "6264ac74ff804c06a121d234a551016e6897146e8770fb03337686825284d707"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once(path: Path, value: dict[str, object]) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode() + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _load_capsule(root: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    capsule_path = root / "capsule.json"
    raw = capsule_path.read_bytes()
    capsule = json.loads(raw)
    if (
        hashlib.sha256(raw).hexdigest() != EXPECTED_CAPSULE_SHA256
        or capsule.get("identity_sha256") != EXPECTED_CAPSULE_IDENTITY
        or capsule.get("model_alias") != "llama3-8b-inst"
        or capsule.get("case_index") != 5
        or capsule.get("arm") != "AR-CAP"
        or capsule.get("outer_step") != 1
    ):
        raise ValueError("P2R6 Llama capsule identity differs")
    arrays: dict[str, np.ndarray] = {}
    for name in ("S", "b", "Q_C", "c_C", "M", "alpha_start"):
        member = capsule["members"][name]
        path = root / f"{name}.npy"
        if path.is_symlink() or _sha256(path) != member["sha256"]:
            raise ValueError(f"P2R6 Llama capsule member differs: {name}")
        arrays[name] = np.load(path, allow_pickle=False)
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"P2R6 Llama capsule member is nonfinite: {name}")
    return capsule, arrays


def _phase_one(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    solved = linprog(
        np.zeros(50, dtype=np.float64),
        A_ub=np.concatenate((arrays["M"], -arrays["S"]), axis=0),
        b_ub=np.concatenate((np.ones(10), -arrays["b"])),
        bounds=[(0.0, None)] * 50,
        method="highs",
        options={
            "primal_feasibility_tolerance": 1.0e-10,
            "dual_feasibility_tolerance": 1.0e-10,
            "ipm_optimality_tolerance": 1.0e-12,
        },
    )
    if not solved.success or solved.x is None:
        raise ValueError("P2R6 Llama Phase-I exact feasibility failed")
    violations = {
        "negative_alpha": max(0.0, -float(np.min(solved.x))),
        "mass_upper": max(0.0, float(np.max(arrays["M"] @ solved.x - 1.0))),
        "semantic_lower": max(
            0.0, float(np.max(arrays["b"] - arrays["S"] @ solved.x))
        ),
    }
    return {
        "success": True,
        "solver_status": int(solved.status),
        "iterations": int(solved.nit),
        "raw_violations": violations,
        "raw_max_violation": max(violations.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--capsule-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--mode", required=True, choices=("legacy", "repaired"))
    args = parser.parse_args()
    capsule, arrays = _load_capsule(args.capsule_root)
    phase_one = _phase_one(arrays)
    if phase_one["raw_max_violation"] > 1.0e-8:
        raise ValueError("P2R6 Llama exact feasibility certificate differs")
    solver_receipt: dict[str, object]
    if args.mode == "legacy":
        try:
            solve_certified_semantic_region_qp(
                arrays["alpha_start"],
                arrays["Q_C"],
                arrays["c_C"],
                arrays["S"],
                arrays["b"],
                arrays["M"],
                stage="LLAMA_OUTER1_LEGACY_FAILURE_REPRODUCTION",
                legacy_failure_reproduction=True,
            )
        except P2R6CertifiedQPError as exc:
            solver_receipt = dict(exc.raw_free_receipt)
        else:
            raise ValueError("P2R6 Llama legacy solver unexpectedly passed")
        if (
            solver_receipt.get("solver_status") != "MAX_ITERATIONS"
            or solver_receipt.get("iterations") != 300
            or solver_receipt.get("linear_solve_count") != 600
        ):
            raise ValueError("P2R6 Llama legacy failure identity differs")
        status = "LEGACY_LLAMA_OUTER1_FAILURE_REPRODUCED"
        final_alpha_sha = hashlib.sha256(
            np.ascontiguousarray(
                solver_receipt["final_alpha"], dtype=np.dtype("<f8")
            ).tobytes()
        ).hexdigest()
    else:
        result = solve_certified_semantic_region_qp(
            arrays["alpha_start"],
            arrays["Q_C"],
            arrays["c_C"],
            arrays["S"],
            arrays["b"],
            arrays["M"],
            stage="LLAMA_OUTER1_REPAIRED",
        )
        solver_receipt = dict(result.receipt)
        if solver_receipt.get("certificate_pass") is not True:
            raise ValueError("P2R6 Llama repaired certificate failed")
        status = "REPAIRED_LLAMA_OUTER1_ORIGINAL_KKT_STRICT_PASS"
        final_alpha_sha = hashlib.sha256(
            np.ascontiguousarray(result.value, dtype=np.dtype("<f8")).tobytes()
        ).hexdigest()
    payload: dict[str, object] = {
        "schema": "ode-edit-s05-p2r6-llama-outer1-fresh-process-verification/v1",
        "instruction_id": (
            "ODEEDIT-S05-P2R6-LLAMA-OUTER1-QP-CAPTURE-REPAIR-PHASE1-V1"
        ),
        "source_head": args.source_head,
        "mode": args.mode,
        "status": status,
        "capsule_sha256": EXPECTED_CAPSULE_SHA256,
        "capsule_identity_sha256": capsule["identity_sha256"],
        "phase_one": phase_one,
        "common_seed": 41,
        "pythonhashseed": int(os.environ.get("PYTHONHASHSEED", "-1")),
        "python_executable": os.path.realpath(os.sys.executable),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "solver_receipt": solver_receipt,
        "final_alpha_raw_sha256": final_alpha_sha,
        "model_gpu_slurm_materialization_count": 0,
        "private_technical_receipt": True,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    receipt_sha = _write_once(args.receipt, payload)
    print(
        json.dumps(
            {
                "status": status,
                "receipt_sha256": receipt_sha,
                "identity_sha256": payload["identity_sha256"],
                "solver_identity_sha256": solver_receipt["identity_sha256"],
                "final_alpha_raw_sha256": final_alpha_sha,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
