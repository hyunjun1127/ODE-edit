"""Closed source/numerical identities for the P1R43-T3 target-only ablation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import ODEBFContractError
from .p1r43_t3_target_only_runtime import CASE_COUNTS, METHOD


LOCK_SCHEMA = "ode-edit-s05-p1r43-t3-fixed-w0-target-horizon-lock/v1"
LOCK_FILE = "numerical_lock_s05_p1r43_t3_fixed_w0_target_horizon.json"
PARENT = "11508b6da11d606521b703037034e1814b70d8a8"
PARENT_TREE = "a0e71bbb27cf36b3b51bf6cde4c92cdc9a47879b"


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": "ODEEDIT-S05-P1R43-T3-FIXED-W0-TARGET-HORIZON-ABLATION-V1",
        "contract_sha256": "d9245782bf3339bfabe48296a9e673ba33763d1bca43be60982a182084ea67e7",
        "exact_p1r43_parent": PARENT,
        "exact_p1r43_parent_tree": PARENT_TREE,
        "method": METHOD,
        "models": ["llama3-8b-inst", "qwen2.5-7b-inst"],
        "stream_root": "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6",
        "all_request_order_sha256": "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c",
        "case_counts": list(CASE_COUNTS),
        "request_count_per_case": 10,
        "context_ordinals": [0, 1, 2, 3, 4, 5],
        "target_update_count": 24,
        "target_update_indices": list(range(24)),
        "target_snapshot_indices": [0, 8, 24],
        "target_endpoint_labels": ["P1R43_T8", "P1R43_T24"],
        "h": 0.125,
        "target_horizon_tau": 3.0,
        "target_schedule": "ONE_CONTINUOUS_P1R43_FIXED_W0_TRAJECTORY",
        "semantic_objective": "REQUESTWISE_FULL_SIX_TARGET_NEW_SUFFIX_NLL",
        "target_field": "P1R43_ENTRY_CALIBRATED_PURE_SEMANTIC_GRADIENT",
        "selection": "REQUESTWISE_PRIMARY_RESCUE_CURRENT_ACTUAL_NLL",
        "entry_norm_calibration_count_per_case": 1,
        "controller_reset_at_8_count": 0,
        "controller_reset_at_16_count": 0,
        "field_update_dtype": "torch.float64",
        "selected_target_storage_dtype": "torch.float32",
        "evaluator_snapshot_index": 8,
        "evaluator_accepted_snapshot_count": 9,
        "evaluator_fixed_budget_slots_completed": 8,
        "stream_seal_file_sha256": "01612c28f700d7281e7f4887189577305396a47715760c98997f97d2acfe596b",
        "frozen_evaluator_sha256": "25c3f49039e7bacb58de6d9ab8139f6f24f21532434a08690e9ec71ea939e145",
        "frozen_aggregator_sha256": "64f009b2fb648627a956b95abd801838d86edf04c8e1888d90ade0ec07b745c0",
        "project_gpu_cap": 4,
        "stage_gpu_max": 2,
        "scientific_promotion_authorized": False,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("P1R43-T3 numerical/source lock differs")
    zero_fields = (
        "writer_update_count",
        "writer_materialization_count",
        "native_endpoint_runtime_access_count",
        "heldout_controller_access_count",
        "retry_count",
        "persistent_freeze_count",
        "semantic_debt_input_count",
        "hard_p_budget_influence_count",
        "early_stop_count",
        "adam_momentum_access_count",
        "beta_state_access_count",
        "bias_correction_access_count",
        "p2r1_rms_influence_count",
        "p2r1_tangent_influence_count",
        "target_kl_decision_influence_count",
        "target_decay_decision_influence_count",
        "target_gamma_decision_influence_count",
        "target_clamp_decision_influence_count",
        "writer_key_factor_covariance_router_influence_count",
        "virtual_w_update_count",
    )
    if any(value.get(key) != 0 for key in zero_fields):
        raise ODEBFContractError("P1R43-T3 forbidden influence differs")


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
