"""Closed numerical/source identities for P1R52 execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_rooted_json
from .contracts import ODEBFContractError
from .p1r52_independent_runtime import ARMS


LOCK_SCHEMA = "ode-edit-s05-p1r52-rsa-r42safekdc-lock/v1"
LOCK_FILE = "numerical_lock_s05_p1r52_rsa_r42safekdc_m1.json"
PARENT = "e9e31260f452ec30ef42393d33eaf37f32f5e1dd"
PARENT_TREE = "29ebec8aa5da4b884863e1f0d3021209b0ecbebb"


def validate_lock(value: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": LOCK_SCHEMA,
        "instruction_id": "ODEEDIT-S05-P1R52-RSA-R42SAFEKDC-M1-V1",
        "contract_sha256": "c7b15a25262a1a8e17e0b9a6b60fced73d49dc4828cc6eea74a6aa447a2212f6",
        "exact_parent": PARENT,
        "exact_parent_tree": PARENT_TREE,
        "models": ["llama3-8b-inst", "qwen2.5-7b-inst"],
        "arms": list(ARMS),
        "stream_root": "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6",
        "all_request_order_sha256": "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c",
        "case_count_per_cell": 10,
        "request_count_per_case": 10,
        "grid_count": 8,
        "h": 0.125,
        "tau_final": 1.0,
        "context_ordinals": list(range(6)),
        "target_direction_policy": "P1R51_RSA_R42_ONE_SIDED_SAFE_KL_DECAY_ORIGIN_CLAMP",
        "amplitude_policy": "P1R51_REQUESTWISE_NLL_PROPORTIONAL_ENERGY_ALLOCATION",
        "corrector_policy": "P1R43_ONE_SCALAR_ACTUAL_NLL_CORRECTOR",
        "corrector_delta_source": "POST_CLAMP_POST_FP32_CAST_ACTUAL_DELTA",
        "kl_direction": "KL_CURRENT_TO_TEACHER_W0",
        "kl_teacher_capture_count_per_case": 1,
        "kl_teacher_refresh_count": 0,
        "writer_coordinate": "FULL_CURRENT_RESIDUAL",
        "routing_arms": ["NEUTRAL", "SOFT"],
        "history_mode": "OFF",
        "numerical_epsilon": 1e-12,
        "numerical_epsilon_source": "P1R39_NORMALIZATION_EPSILON",
        "semantic_certificate_tolerance": 1e-8,
        "alias_hparams": {
            "llama3-8b-inst": {"kl_factor": 0.0625, "decay_factor": 0.5, "clamp_factor": 0.75},
            "qwen2.5-7b-inst": {"kl_factor": 0.0625, "decay_factor": 0.001, "clamp_factor": 4.0},
        },
        "project_gpu_cap": 4,
        "stage_gpu_max": 4,
        "scientific_promotion_authorized": False,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ODEBFContractError("P1R52 numerical/source lock differs")
    zero = (
        "additional_kl_forward_count",
        "additional_kl_backward_count",
        "adaptive_target_inner_loop_count",
        "p2_operator_access_count",
        "target_hit_threshold_access_count",
        "persistent_freeze_input_count",
        "semantic_debt_input_count",
        "remaining_horizon_decision_influence_count",
        "writer_router_change_count",
        "hard_p_budget_influence_count",
        "functional_p_veto_influence_count",
        "historical_sequential_state_count",
        "inner_step_heldout_evaluation_count",
        "retry_count",
        "backtracking_count",
        "clamp_energy_redistribution_count",
    )
    if any(value.get(key) != 0 for key in zero):
        raise ODEBFContractError("P1R52 forbidden influence differs")


def load_and_validate_lock(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_rooted_json(path, expected_schema=LOCK_SCHEMA)
    validate_lock(value)
    return value, file_sha


__all__ = ["LOCK_FILE", "LOCK_SCHEMA", "PARENT", "PARENT_TREE", "load_and_validate_lock", "validate_lock"]
