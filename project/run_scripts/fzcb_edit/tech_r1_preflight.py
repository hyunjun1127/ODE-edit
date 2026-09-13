"""TECH-R1 joint-B1 source, tolerance, stream, and launcher preflight."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .hashing import canonical_hash, file_sha256, write_json_once
from .preflight import build as build_base
from .tech_r1_joint_launcher import joint_matrix
from .tolerances import UnitTolerancePolicy, synthetic_calibration_receipt


def build(
    source_root: Path,
    expected_source_head: str | None = None,
    *,
    focused_gate_path: Path,
    expected_focused_gate_sha: str,
) -> dict[str, Any]:
    base = build_base(source_root, expected_source_head)
    if (
        not focused_gate_path.is_file()
        or focused_gate_path.is_symlink()
        or file_sha256(focused_gate_path) != expected_focused_gate_sha
    ):
        raise RuntimeError("focused gate identity mismatch")
    import json

    focused = json.loads(focused_gate_path.read_text(encoding="utf-8"))
    if (
        focused.get("status") != "FOCUSED_GATE_PASS"
        or focused.get("source", {}).get("head") != base["source"]["head"]
    ):
        raise RuntimeError("focused gate source/status mismatch")
    policy = UnitTolerancePolicy.load()
    calibration = synthetic_calibration_receipt(policy)
    matrix = joint_matrix()
    payload = {
        "schema": "odeedit.s06.fzcb-tech-r1.joint-b1-pre-gpu.v1",
        "status": "PRE_GPU_PASS",
        "source": base["source"],
        "authority": base["authority"],
        "easyedit": base["easyedit"],
        "stream": {"B1": base["stream"]["B1"]},
        "artifact_preflight": base["artifact_preflight"],
        "numerical_lock": base["numerical_lock"],
        "tolerance_policy": policy.receipt(),
        "tolerance_calibration": calibration,
        "focused_gate": {
            "path": str(focused_gate_path.resolve()),
            "sha256": expected_focused_gate_sha,
            "test_count": focused["tests"]["run"],
            "member_root": focused["member_root"],
        },
        "joint_matrix": matrix,
        "joint_matrix_identity": canonical_hash(matrix),
        "full_fp32": True,
        "dense_inverse_count": 0,
        "explicit_kronecker_count": 0,
        "controller_output_metric_access_count": 0,
        "b10_submit_count": 0,
        "main_push_count": 0,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-head")
    parser.add_argument("--focused-gate", type=Path, required=True)
    parser.add_argument("--expected-focused-gate-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json_once(
        args.output,
        build(
            args.source_root,
            args.expected_source_head,
            focused_gate_path=args.focused_gate,
            expected_focused_gate_sha=args.expected_focused_gate_sha,
        ),
    )


if __name__ == "__main__":
    main()
