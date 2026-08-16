"""Closed source/numerical identities for P2R1 target-only Gate A."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import ODEBFContractError
from .p2r1_target_only_runtime import CASE_COUNTS, METHOD


LOCK_SCHEMA = "ode-edit-s05-p2r1-rms-tangent-target-only-lock/v1"
LOCK_FILE = "numerical_lock_s05_p2r1_rms_tangent_target_only.json"
PARENT = "11508b6da11d606521b703037034e1814b70d8a8"
PARENT_TREE = "a0e71bbb27cf36b3b51bf6cde4c92cdc9a47879b"


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": "ODEEDIT-S05-P2R1-BASELINE-CALIBRATED-RMS-TANGENT-TARGET-FLOW-V1",
        "contract_sha256": "4bfb6c99e8a71ae822afe08e32e2263e8d6c69485d6b54a5cbd12b985884c414",
        "exact_p1r43_parent": PARENT,
        "exact_p1r43_parent_tree": PARENT_TREE,
        "method": METHOD,
        "models": ["llama3-8b-inst", "qwen2.5-7b-inst"],
        "stream_root": "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6",
        "all_request_order_sha256": "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c",
        "case_counts": list(CASE_COUNTS),
        "request_count_per_case": 10,
        "context_ordinals": [0, 1, 2, 3, 4, 5],
        "outer_state_count": 8,
        "target_microsteps_per_outer_state": 3,
        "target_microstep_count": 24,
        "target_schedule": "FIXED_MULTIRATE_EULER_LIE_TROTTER_STYLE_ALTERNATING_FLOW",
        "semantic_objective": "REQUESTWISE_FULL_SIX_TARGET_NEW_SUFFIX_NLL",
        "target_field": "CUMULATIVE_RMS_SEMANTIC_FIRST_KL_DECAY_TANGENT",
        "teacher_refresh_count_per_case": 1,
        "decay_origin": "OUTER_ENTRY_CANONICAL_ACTIVATION",
        "llama_eta_kl_decay_clamp": [0.1, 0.0625, 0.5, 0.75],
        "qwen_eta_kl_decay_clamp": [0.5, 0.0625, 0.001, 4.0],
        "llama_alphaedit_hparams_sha256": "d403e1875e62096b089be5343d33896510e454b0cdd2b256618ab53de6609ef8",
        "qwen_alphaedit_hparams_sha256": "82d04976c4ab65e67c537ac3bd1b04d42c8f7527e2a749bdefcce63e43b995c3",
        "alphaedit_compute_z_sha256": "e12140c66b467759f3fb0061ec844f79147221e0649f0d0a5f9657225ff12c95",
        "stream_seal_file_sha256": "01612c28f700d7281e7f4887189577305396a47715760c98997f97d2acfe596b",
        "frozen_evaluator_sha256": "25c3f49039e7bacb58de6d9ab8139f6f24f21532434a08690e9ec71ea939e145",
        "frozen_aggregator_sha256": "64f009b2fb648627a956b95abd801838d86edf04c8e1888d90ade0ec07b745c0",
        "project_gpu_cap": 4,
        "stage_gpu_max": 2,
        "scientific_promotion_authorized": False,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("P2R1 numerical/source lock differs")
    zero_fields = (
        "writer_update_count",
        "writer_materialization_count",
        "native_endpoint_runtime_access_count",
        "heldout_controller_access_count",
        "retry_count",
        "hold_count",
        "semantic_debt_input_count",
        "hard_p_budget_influence_count",
        "early_stop_count",
        "adam_momentum_access_count",
        "beta_state_access_count",
        "bias_correction_access_count",
    )
    if any(value.get(key) != 0 for key in zero_fields):
        raise ODEBFContractError("P2R1 forbidden influence differs")


def load_and_validate_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_lock(value)
    return value, file_sha


__all__ = [
    "LOCK_FILE",
    "LOCK_SCHEMA",
    "PARENT",
    "PARENT_TREE",
    "load_and_validate_lock",
    "validate_lock",
]
