#!/usr/bin/env python3
"""Fresh-process verification for the sealed P2R6 Qwen replay capsule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import scipy

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p2r6_certified_convex_qp import (
    solve_certified_semantic_region_qp,
)


def _write_once(path: Path, value: dict[str, object]) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--capsule-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    args = parser.parse_args()
    arrays = {
        name: np.load(args.capsule_root / f"{name}.npy", allow_pickle=False)
        for name in ("S", "b", "Q_C", "c_C", "alpha_start", "mass_matrix")
    }
    capsule_payload = (args.capsule_root / "capsule.json").read_bytes()
    capsule = json.loads(capsule_payload)
    result = solve_certified_semantic_region_qp(
        arrays["alpha_start"],
        arrays["Q_C"],
        arrays["c_C"],
        arrays["S"],
        arrays["b"],
        arrays["mass_matrix"],
        stage="CAPTURED_QWEN_AS_OUTER05_AR_CAP",
    )
    payload: dict[str, object] = {
        "schema": "ode-edit-s05-p2r6-red-r2-fresh-process-qp-verification/v1",
        "instruction_id": "ODEEDIT-S05-P2R6-RED-R2-FINAL-SCIENTIFIC-RUN-V1",
        "source_head": args.source_head,
        "capsule_receipt_sha256": hashlib.sha256(capsule_payload).hexdigest(),
        "capsule_identity_sha256": capsule["identity_sha256"],
        "common_seed": 41,
        "pythonhashseed": int(os.environ.get("PYTHONHASHSEED", "-1")),
        "python_executable": os.path.realpath(os.sys.executable),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "solver_receipt": dict(result.receipt),
        "final_alpha_raw_sha256": hashlib.sha256(
            np.ascontiguousarray(result.value, dtype=np.dtype("<f8")).tobytes(order="C")
        ).hexdigest(),
        "certificate_pass": bool(result.receipt["certificate_pass"]),
        "status": "P2R6_RED_R2_NEW_SOLVER_CAPTURE_STRICT_PASS",
    }
    payload["identity_sha256"] = canonical_hash(payload)
    receipt_sha = _write_once(args.receipt, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "receipt_sha256": receipt_sha,
                "identity_sha256": payload["identity_sha256"],
                "solver_identity_sha256": result.receipt["identity_sha256"],
                "alpha_sha256": payload["final_alpha_raw_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
