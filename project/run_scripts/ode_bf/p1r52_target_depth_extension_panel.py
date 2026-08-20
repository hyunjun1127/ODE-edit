"""Numerical and immutable-reference gates for the target-depth extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json, sha256_file
from .contracts import ODEBFContractError
from .p1r52_target_depth_extension import EXTENSION_DEPTHS, INSTRUCTION_ID


LOCK_SCHEMA = "ode-edit-s05-p1r52-target-depth-il8-il10-il15-lock/v1"
LOCK_FILE = "numerical_lock_s05_p1r52_target_depth_il8_il10_il15.json"
SOURCE_PARENT = "0ac0aa2fec7d804b1cb14dfdea28bf6a5f089ec4"
IL1_REPORT = "experiment-reports/global/2026-08-17-p1r52-rsa-r42safekdc-m1-repair-r1-final-ko.md"
IL1_INDEX = "experiment-reports/global/2026-08-17-p1r52-rsa-r42safekdc-m1-repair-r1-artifacts.json"
IL3_REPORT_ROOT = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh2-s05-p1r52-target-depth-il1-il3full-v1/"
    "local/odebf/reports/p1r52-target-depth-phase1b-atomic-v3"
)
IL3_FILES = {
    "p1r52-target-depth-phase1b-atomic-terminal-report-ko.md": "4925b7e0219f5cc17788cf674ff0aa81c23f94a111176bf3e25ccc59fa64b3ec",
    "manifest.json": "9e0958855505fec4b33362551e7710a800bf7f5df02330924fd8618a5a16cbdb",
    "group-summary.json": "c7024e39d143af9b9700614b4c54fb5e9e9e04c2045ae5a6cac1d9b45451c5b6",
    "per-case.json": "dde1c6e7015d706fb18cce41c15a94a319409396f18b89345cb0a56e6f7061d7",
    "per-request.json": "46daeca58465a974a96dfa43b6a539cc151e07207248e653d215abb99373b3bf",
    "per-step.json": "e3e1d8f7c7459595274154326e5aa5672bd8a0d54eb0ee91109e425dce352ff0",
}


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": INSTRUCTION_ID,
        "source_parent": SOURCE_PARENT,
        "model": "llama3-8b-inst",
        "arm": "soft",
        "depth_policies": {item.value: item.inner_count for item in EXTENSION_DEPTHS},
        "comparison_depths": [1, 3, 8, 10, 15],
        "inner_h": 0.125,
        "outer_step_count": 8,
        "case_count_per_depth": 10,
        "request_attempt_count_per_depth": 100,
        "project_gpu_cap": 3,
        "stream_root": "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6",
        "all_request_order_sha256": "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c",
        "target_operator": "P1R52_RSA_R42SAFEKDC_M1_REPAIR_R1",
        "writer": "P1R52_ORIGINAL_J0_FROZEN",
        "entry_norm_calibration_count_per_case": 1,
        "inner_writer_materialization_count": 0,
        "outer_writer_materialization_count": 1,
        "inner_heldout_access_policy": "OBSERVATION_ONLY_AFTER_ACCEPT",
        "inner_accepted_z_observation_pass_count_per_inner": 1,
        "outer_w_z_observation_pass_count_per_outer": 2,
        "inner_added_backward_count": 0,
        "inner_added_generation_count": 0,
        "inner_observation_action_influence_count": 0,
        "duplicate_evaluation_count": 0,
        "expected_inner_rows_per_case": {"IL8-FULL": 64, "IL10-FULL": 80, "IL15-FULL": 120},
        "expected_outer_rows_per_case": 8,
        "early_stop": "ENTIRE_SELECTED_FP32_TARGET_BYTE_IDENTICAL_ONLY",
        "il5_action_count": 0,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("P1R52 target-depth extension lock differs")
    zero_fields = (
        "h_over_depth_count",
        "inner_covariance_recompute_count",
        "inner_factor_build_count",
        "inner_finite_demand_count",
        "inner_history_append_count",
        "inner_weight_mutation_count",
        "inner_writer_count",
        "retry_count",
        "backtracking_count",
        "writer_router_change_count",
        "main_push_count",
    )
    if any(value.get(key) != 0 for key in zero_fields):
        raise ODEBFContractError("P1R52 target-depth extension forbidden counter differs")


def load_and_validate_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_lock(value)
    return value, file_sha


def verify_immutable_references(repo_root: Path) -> dict[str, str]:
    expected = {
        repo_root / IL1_REPORT: "9434cb5ab4ae650b5e72226d6833ea46960d2303a4634a249a63018e9509a5d6",
        repo_root / IL1_INDEX: "6cedac885cea1e35f3e2ad2f5b827b96a25175deb303745a949fa1bff7a0d129",
        **{IL3_REPORT_ROOT / name: digest for name, digest in IL3_FILES.items()},
    }
    observed: dict[str, str] = {}
    for path, digest in expected.items():
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise ODEBFContractError("P1R52 immutable IL1/IL3 reference identity differs")
        observed[str(path)] = digest
    return observed


__all__ = [
    "IL3_FILES",
    "IL3_REPORT_ROOT",
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "SOURCE_PARENT",
    "load_and_validate_lock",
    "validate_lock",
    "verify_immutable_references",
]
