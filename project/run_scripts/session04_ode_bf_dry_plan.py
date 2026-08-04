#!/usr/bin/env python3
"""Deterministic, no-model ODE-BF technical P0 plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import MODEL_ALIASES, ODEBFContractError
from project.run_scripts.ode_bf.p0_runtime import expected_result_name
from project.run_scripts.ode_bf.resource import MODEL_GEOMETRY, forecast_p0_b10_memory


JOB_NAMES = {
    "llama3-8b-inst": "odebf_s04_p0r3_llama",
    "qwen2.5-7b-inst": "odebf_s04_p0r3_qwen",
}
PACKAGE_ROOT = Path(__file__).resolve().parent / "ode_bf"


def build_plan(
    source_head: str,
    *,
    repository_root: Path = REPO_ROOT,
) -> dict[str, object]:
    artifact = PACKAGE_ROOT / "locks/p0_artifact_lock.json"
    base = repository_root / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json"
    jobs = []
    for alias in MODEL_ALIASES:
        forecast = forecast_p0_b10_memory(artifact, base, alias)
        out_features, in_features, layer_count = MODEL_GEOMETRY[alias]
        # The earlier forecast already includes Native and one WB candidate.
        # R3 retains the same four diagnostic endpoints as R2 while promoting
        # only W64 to the prospective technical path. D32 and W64 therefore
        # remain the two additional host BF16 candidates, with the same fixed
        # 512 MiB raw-free evaluator/receipt reserve.
        additional_host_mib = (
            2 * layer_count * out_features * in_features * 2
            + (1024 * 1024 - 1)
        ) // (1024 * 1024) + 512
        r3_host_peak_mib = forecast.forecast_host_peak_mib + additional_host_mib
        if r3_host_peak_mib > 65_000:
            raise ODEBFContractError("R3 mixed64 four-path host memory forecast exceeds request")
        jobs.append(
            {
                "alias": alias,
                "job_name": JOB_NAMES[alias],
                "result_name": expected_result_name(alias),
                "gpu": 1,
                "cpu": 8,
                "memory_mib": 65_000,
                "time": "04:00:00",
                "node": "server2",
                "memory_forecast": forecast.raw_free_payload(),
                "r3_additional_host_mib": additional_host_mib,
                "r3_forecast_host_peak_mib": r3_host_peak_mib,
            }
        )
    return {
        "schema": "ode-edit-s04-ode-bf-p0-dry-plan/v1",
        "instruction_id": "ODEEDIT-S04-ODE-BF-W64-CANONICAL-RECEIPT-P0-R3-V1",
        "source_head": source_head,
        "edit_batch_size": 10,
        "joint_editor_invocations_per_job": 1,
        "k_resolution": 8,
        "correction_cycles": 1,
        "model_load": False,
        "gpu_use": False,
        "slurm_submit": False,
        "retry_or_resubmit": False,
        "diagnostic_paths": ["N32", "D32", "W32", "W64"],
        "cross_solver_byte_gate": "diagnostic-only",
        "prospective_technical_path": "W64",
        "w32_fallback": False,
        "strict_byte_gates": ["W64-virtual-vs-commit", "rollback-vs-W0", "final-W0-restore"],
        "jobs": jobs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    print(
        json.dumps(
            build_plan(args.source_head, repository_root=args.repository_root),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
