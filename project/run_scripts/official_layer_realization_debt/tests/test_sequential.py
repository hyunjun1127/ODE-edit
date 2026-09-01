from __future__ import annotations

from unittest import mock

from project.run_scripts.official_layer_realization_debt.continuity import (
    verify_cache_continuity,
    verify_cell_totals,
    verify_observed_batch,
    verify_weight_continuity,
)
from project.run_scripts.official_layer_realization_debt.contracts import Method


def _observed(*, compute_ks: int = 5) -> dict:
    return {
        "target_backward_count": 10,
        "official_call_audit": {"compute_ks_call_count": compute_ks, "torch_linalg_solve_call_count": 5},
        "layer_realization_observer": {
            "direct_z_compute_count": 10,
            "direct_z_recompute_count": 0,
            "layer_loop_observation_copy_count": 5,
            "terminal_post_L8_forward_count": 1,
            "residual_debt": {"nonfinite_count": 0, "maximum_recurrence_closure_relative_error": 1e-8},
            "overhead_wall_seconds": {
                "additional_z_optimization": 0.0,
                "additional_key_compute": 0.0,
                "additional_closed_form_solve": 0.0,
            },
        },
    }


def test_sequential_observer_batch_contract() -> None:
    assert all(verify_observed_batch(_observed(), method=Method.MEMIT, request_count=10).values())
    assert all(verify_observed_batch(_observed(compute_ks=10), method=Method.ALPHAEDIT, request_count=10).values())


def test_weight_commit_to_next_entry() -> None:
    payload = {
        "entry_sha256": {"w": "a"},
        "original_copy_sha256": {"w": "a"},
        "edited_sha256": {"w": "b"},
    }
    assert verify_weight_continuity(payload, expected_entry={"w": "a"})["physical_transition_nonzero"]


def test_alpha_reset_once_and_dynamic_cache_link() -> None:
    first = {
        "alphaedit_dynamic_cache_contract": {
            "reset_cache_requested": True,
            "logical_history_width_at_entry": 0,
            "logical_history_width_after_append": 10,
            "append_request_count": 10,
            "entry": {"sha256": None},
            "exit": {"sha256": "x"},
            "solver_consumed_entry_cache": False,
            "static_projection_distinct_from_dynamic_cache": True,
        }
    }
    second = {
        "alphaedit_dynamic_cache_contract": {
            "reset_cache_requested": False,
            "logical_history_width_at_entry": 10,
            "logical_history_width_after_append": 20,
            "append_request_count": 10,
            "entry": {"sha256": "x"},
            "exit": {"sha256": "y"},
            "solver_consumed_entry_cache": True,
            "static_projection_distinct_from_dynamic_cache": True,
        }
    }
    assert verify_cache_continuity(Method.ALPHAEDIT, first, batch_index=1, request_count=10, prior_exit_sha256=None)["exit_width"] == 10
    assert verify_cache_continuity(Method.ALPHAEDIT, second, batch_index=2, request_count=10, prior_exit_sha256="x")["entry_width"] == 10


def test_memit_static_covariance_has_no_history_state() -> None:
    state = {"identity_sha256": "cov", "request_history_width": 0, "historical_decision_state": False}
    payload = {"covariance_cache": {"entry": state, "exit": state, "entry_reused_count": 5, "silent_reset_count": 0}}
    receipt = verify_cache_continuity(Method.MEMIT, payload, batch_index=2, request_count=10, prior_exit_sha256="cov")
    assert receipt["kind"] == "STATIC_MEMIT_COVARIANCE_COMPUTATION_CACHE"


def test_cell_totals_and_nine_weight_links() -> None:
    rows = [
        {
            "request_count": 10,
            "direct_z_compute_count": 10,
            "direct_z_recompute_count": 0,
            "layer_observation_count": 5,
            "terminal_forward_count": 1,
            "weight_link_from_previous": int(index > 0),
        }
        for index in range(10)
    ]
    assert verify_cell_totals(rows)["weight_links_9_of_9"]
