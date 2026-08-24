#!/usr/bin/env python3
"""Focused preflight for user-directed B2-B10 independent reset expansion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from project.run_scripts.ode_bf.artifacts import sha256_file
from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.p1r54_realization_reset import (
    INDEPENDENT_EXPANSION_CELLS,
    INDEPENDENT_EXPANSION_ROLES,
    INSTRUCTION_ID,
)
from project.run_scripts.session05_ode_bf_p1r52_target_timescale_b100_preflight import (
    _write_or_verify_once,
)
from project.run_scripts.session05_ode_bf_p1r54_realization_reset_independent_dry_plan import (
    build_plan,
)
from project.run_scripts.session05_ode_bf_p1r54_realization_reset_preflight import (
    EXTRACT_ROOT,
    JOB_GPU_COUNT,
    PRIOR_HF_GATE,
    PROJECT_GPU_CAP,
    build_receipt as build_base_receipt,
    source_manifest,
)


ARRAY_THROTTLE = 2
REQUIRED_ARRAY = "0-17%2"


def build_receipt(
    *, source_head: str, final_receipt: Path, session_id: str
) -> Mapping[str, Any]:
    base_path = final_receipt.parent / "base-fourcell-final-pre-gpu.json"
    base = build_base_receipt(
        source_head=source_head,
        final_receipt=base_path,
        session_id=session_id,
        continuation_control_audit_required=False,
    )
    plan = build_plan()
    if (
        plan["array"] != REQUIRED_ARRAY
        or len(plan["cells"]) != 18
        or tuple(item["role"] for item in plan["cells"])
        != INDEPENDENT_EXPANSION_ROLES
        or any(item["batch_index"] == 1 for item in plan["cells"])
    ):
        raise ODEBFContractError("P1R54 reset independent expansion plan differs")
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-realization-reset-independent-final-pre-gpu/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "FINAL_PRE_GPU_PASS",
        "model_load_authorized": True,
        "source_head": base["source_head"],
        "source_tree": base["source_tree"],
        "source_manifest": source_manifest(base["source_head"], base["source_tree"]),
        "project_gpu_cap": PROJECT_GPU_CAP,
        "job_gpu_count": JOB_GPU_COUNT,
        "array_throttle": ARRAY_THROTTLE,
        "required_array": REQUIRED_ARRAY,
        "dry_plan": plan,
        "stream_binding": base["stream_binding"],
        "deployment_identity": base["deployment_identity"],
        "easyedit_runtime_seal_root": base["easyedit_runtime_seal_root"],
        "hf_binding": base["hf_binding"],
        "artifact_receipt": base["artifact_receipt"],
        "numerical_lock": base["numerical_lock"],
        "base_preflight_path": str(base_path),
        "base_preflight_sha256": sha256_file(base_path),
        "base_preflight_identity": base["identity_sha256"],
        "roles": list(INDEPENDENT_EXPANSION_ROLES),
        "result_names": [item.result_name for item in INDEPENDENT_EXPANSION_CELLS],
        "B1_reexecution_count": 0,
        "new_continuation_gpu_run_count": 0,
        "cross_batch_state_consumption_count": 0,
        "scientific_promotion": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    _write_or_verify_once(final_receipt, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--final-receipt", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    print(json.dumps(build_receipt(
        source_head=args.source_head,
        final_receipt=args.final_receipt,
        session_id=args.session_id,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
