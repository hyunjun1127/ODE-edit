"""Pure continuity verifiers for the sequential observer binding."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import Method, ObservationBoundary, ObservationLock


def verify_observed_batch(payload: Mapping[str, Any], *, request_count: int) -> dict[str, Any]:
    observer = payload["layer_realization_observer"]
    debt = observer["residual_debt"]
    audit = payload["official_call_audit"]
    checks = {
        "direct_z_once_per_request": int(observer["direct_z_compute_count"]) == request_count,
        "direct_z_recompute_zero": int(observer["direct_z_recompute_count"]) == 0,
        "layer_observation_5_of_5": int(observer["layer_loop_observation_copy_count"]) == 5,
        "terminal_forward_1": int(observer["terminal_post_L8_forward_count"]) == 1,
        "nonfinite_zero": int(debt["nonfinite_count"]) == 0,
        "recurrence_closure": float(debt["maximum_recurrence_closure_relative_error"])
        < ObservationLock().recurrence_relative_tolerance,
        "additional_z_zero": float(observer["overhead_wall_seconds"]["additional_z_optimization"]) == 0.0,
        "additional_key_zero": float(observer["overhead_wall_seconds"]["additional_key_compute"]) == 0.0,
        "additional_solve_zero": float(observer["overhead_wall_seconds"]["additional_closed_form_solve"]) == 0.0,
        "compute_ks_5": int(audit["compute_ks_call_count"]) == 5,
        "solve_5": int(audit["torch_linalg_solve_call_count"]) == 5,
    }
    if not all(checks.values()):
        raise ObservationBoundary(f"sequential batch observer invariant differs: {checks}")
    return checks


def verify_weight_continuity(
    payload: Mapping[str, Any],
    *,
    expected_entry: Mapping[str, str],
) -> dict[str, Any]:
    entry = dict(payload["entry_sha256"])
    original = dict(payload["original_copy_sha256"])
    edited = dict(payload["edited_sha256"])
    checks = {
        "entry_matches_previous_commit": entry == dict(expected_entry),
        "official_original_copy_matches_entry": original == entry,
        "physical_transition_nonzero": edited != entry,
    }
    if not all(checks.values()):
        raise ObservationBoundary(f"sequential W continuity differs: {checks}")
    return {**checks, "entry_sha256": entry, "commit_sha256": edited}


def verify_cache_continuity(
    method: Method,
    payload: Mapping[str, Any],
    *,
    batch_index: int,
    request_count: int,
    prior_exit_sha256: str | None,
) -> dict[str, Any]:
    if method is Method.ALPHAEDIT:
        cache = payload["alphaedit_dynamic_cache_contract"]
        expected_entry_width = (batch_index - 1) * request_count
        expected_exit_width = batch_index * request_count
        checks = {
            "reset_only_at_first_entry": bool(cache["reset_cache_requested"]) == (batch_index == 1),
            "entry_width_exact": int(cache["logical_history_width_at_entry"]) == expected_entry_width,
            "consume_width_exact": int(cache["logical_history_width_at_entry"]) == expected_entry_width,
            "append_width_exact": int(cache["append_request_count"]) == request_count,
            "exit_width_exact": int(cache["logical_history_width_after_append"]) == expected_exit_width,
            "entry_exit_sha_link": batch_index == 1 or str(cache["entry"]["sha256"]) == prior_exit_sha256,
            "prior_cache_consumed": batch_index == 1 or bool(cache["solver_consumed_entry_cache"]),
            "static_projection_separate": bool(cache["static_projection_distinct_from_dynamic_cache"]),
        }
        if not all(checks.values()):
            raise ObservationBoundary(f"AlphaEdit sequential cache continuity differs: {checks}")
        return {
            "kind": "OFFICIAL_ALPHAEDIT_DYNAMIC_CACHE_C",
            "entry_width": expected_entry_width,
            "consume_width": expected_entry_width,
            "append_width": request_count,
            "exit_width": expected_exit_width,
            "entry_sha256": cache["entry"]["sha256"],
            "exit_sha256": cache["exit"]["sha256"],
            "checks": checks,
        }

    cache = payload["covariance_cache"]
    entry = cache["entry"]
    exit_state = cache["exit"]
    checks = {
        "static_covariance_identity_link": batch_index == 1
        or str(entry["identity_sha256"]) == prior_exit_sha256,
        "request_history_width_zero": int(entry["request_history_width"]) == 0
        and int(exit_state["request_history_width"]) == 0,
        "historical_decision_state_false": not bool(entry["historical_decision_state"])
        and not bool(exit_state["historical_decision_state"]),
        "entry_reuse_after_first": batch_index == 1 or int(cache["entry_reused_count"]) == 5,
        "silent_reset_zero": int(cache["silent_reset_count"]) == 0,
    }
    if not all(checks.values()):
        raise ObservationBoundary(f"MEMIT static covariance continuity differs: {checks}")
    return {
        "kind": "STATIC_MEMIT_COVARIANCE_COMPUTATION_CACHE",
        "entry_width": 0,
        "consume_width": 0,
        "append_width": 0,
        "exit_width": 0,
        "entry_sha256": entry["identity_sha256"],
        "exit_sha256": exit_state["identity_sha256"],
        "checks": checks,
    }


def verify_cell_totals(journals: list[Mapping[str, Any]]) -> dict[str, Any]:
    if len(journals) != 10:
        raise ObservationBoundary("sequential cell batch denominator differs")
    direct = sum(int(row["direct_z_compute_count"]) for row in journals)
    recompute = sum(int(row["direct_z_recompute_count"]) for row in journals)
    layers = sum(int(row["layer_observation_count"]) for row in journals)
    terminal = sum(int(row["terminal_forward_count"]) for row in journals)
    checks = {
        "batch_count_10": len(journals) == 10,
        "request_count_100": sum(int(row["request_count"]) for row in journals) == 100,
        "direct_z_100": direct == 100,
        "recompute_0": recompute == 0,
        "layer_observation_50": layers == 50,
        "terminal_forward_10": terminal == 10,
        "weight_links_9_of_9": sum(int(row["weight_link_from_previous"]) for row in journals[1:]) == 9,
    }
    if not all(checks.values()):
        raise ObservationBoundary(f"sequential cell totals differ: {checks}")
    return {**checks, "direct_z": direct, "recompute": recompute, "layer_observation": layers, "terminal_forward": terminal}
