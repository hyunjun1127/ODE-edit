#!/usr/bin/env python3
"""Model-free dry plan for the one-shot P2R6 Llama QP capture."""

from __future__ import annotations

import argparse
import json


FAILED_SCIENTIFIC_SOURCE_HEAD = "1246e5047bdac2e59d5cc0642ebd8d7104c6d066"
STREAM_ROOT = "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6"
STREAM_ORDER = "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c"


def build_plan(source_head: str) -> dict[str, object]:
    return {
        "schema": "ode-edit-s05-p2r6-llama-outer1-qp-capture-dry-plan/v1",
        "source_head": source_head,
        "failed_scientific_source_head": FAILED_SCIENTIFIC_SOURCE_HEAD,
        "model": "llama3-8b-inst",
        "case_index": 5,
        "arm": "AR-CAP",
        "outer0_physical_prefix_count": 1,
        "capture_outer_step": 1,
        "selected_outer1_qp_solve_count": 0,
        "capture_invocation_budget": 1,
        "scientific_endpoint_count": 0,
        "terminal_evaluator_count": 0,
        "added_model_forward_count": 0,
        "added_model_backward_count": 0,
        "added_materialization_count": 0,
        "w0_restore_required": True,
        "gpu_count": 1,
        "cpu_count": 8,
        "memory_mib": 65000,
        "node": "devbox",
        "server1_project_gpu_cap": 4,
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "phase1_status": "NOT_EXECUTED_CAPTURE_ONLY",
        "phase2_status": "CLOSED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    args = parser.parse_args()
    print(json.dumps(build_plan(args.source_head), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
