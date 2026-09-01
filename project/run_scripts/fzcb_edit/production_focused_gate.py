"""Run only the twelve contract-mandated atomic-B10 focused gates."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .hashing import file_sha256, write_json_once


MODULES = (
    "production_contracts.py",
    "production_sensitivity.py",
    "production_target.py",
    "production_geometry.py",
    "production_controller.py",
    "production_journal.py",
    "production_evaluation.py",
    "production_runtime.py",
    "production_preflight.py",
    "production_launcher.py",
    "controller_validity.py",
)


def run(source_root: Path) -> dict[str, object]:
    package = source_root / "project/run_scripts/fzcb_edit"
    compile_command = [sys.executable, "-m", "py_compile", *[str(package / name) for name in MODULES]]
    compile_run = subprocess.run(compile_command, text=True, capture_output=True, check=False)
    if compile_run.returncode != 0:
        raise RuntimeError(f"py_compile failed: {compile_run.stderr}")
    test_command = [
        sys.executable,
        "-m",
        "unittest",
        "-v",
        "project.run_scripts.fzcb_edit.tests.test_production_b10",
    ]
    tests = subprocess.run(test_command, cwd=source_root, text=True, capture_output=True, check=False)
    combined = tests.stdout + tests.stderr
    if tests.returncode != 0 or "Ran 12 tests" not in combined or "OK" not in combined:
        raise RuntimeError(f"focused production tests failed:\n{combined}")
    sbatch = source_root / "project/run_scripts/session06_fzcb_atomic_b10_production_server4.sbatch"
    bash = subprocess.run(["bash", "-n", str(sbatch)], text=True, capture_output=True, check=False)
    if bash.returncode != 0:
        raise RuntimeError(f"sbatch syntax failed: {bash.stderr}")
    members = [
        {"path": str((package / name).relative_to(source_root)), "sha256": file_sha256(package / name)}
        for name in MODULES
    ]
    return {
        "schema": "odeedit.s06.fzcb.atomic-b10-production-pilot.focused-gates.v1",
        "status": "PASS",
        "tests_passed": 12,
        "tests_failed": 0,
        "gate_names": [
            "normalized_cbf_action_rescaling_invariance",
            "equality_direction_normalization_and_derivative_rescaling",
            "fp64_scalar_reduction",
            "quadratic_analytic_derivative_identity",
            "uncertainty_interval_numerical_inconclusive",
            "actual_hcc_violation_rollback_backtracking_boundary",
            "weight_cache_pointer_bytes_restore",
            "append_only_arm_journal",
            "raw_fd_axis_report_row_identity",
            "cohort_target_hash_identity",
            "fixed_z_waypoint_recompute_zero",
            "corrector_action_in_total_E",
        ],
        "py_compile_members": members,
        "bash_syntax": {"path": str(sbatch.relative_to(source_root)), "sha256": file_sha256(sbatch)},
        "test_output": combined,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json_once(args.output, run(args.source_root.resolve()))


if __name__ == "__main__":
    main()
