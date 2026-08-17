#!/usr/bin/env python3
"""Create-only raw-free P1R51 Neutral+Soft final analysis."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("/mnt/raid5/janghj/.codex/worktrees/odeedit-p2r7-main-publish-v1")
OUT = ROOT / "local/odebf/reports/p1r51-rsa-a1-final-neutral-soft-v1"
NEUTRAL_REPORT = ROOT / "local/odebf/reports/p1r51-rsa-a1-neutral-b10x10-v1"
PARENT = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-p1r43-rho-free-semantic-first-strength-recovery-v1"
)
PARENT_REPORT = PARENT / "local/odebf/reports/p1r43-rho-free-semantic-first-v1"
OFFICIAL = Path(
    "/mnt/raid5/janghj/ODE-edit/local/source-handoff/"
    "P1R43_P1R42_PARENT_SUPPORT_SH2_V1/identities/"
    "official-alphaedit-baseline-identity.json"
)
CONTRACT = Path(
    "/mnt/raid5/janghj/.codex/attachments/"
    "264d9efc-75b7-4694-99b6-e64f005704b4/pasted-text.txt"
)
MODELS = ("llama3-8b-inst", "qwen2.5-7b-inst")
ARMS = ("Neutral", "Soft")
CASES = tuple(range(1, 11))
P1R51_HEAD = "9b4a88b73bd0c0fdd0b1470d77ea4354bd0e3486"
P1R51_TREE = "4a3b5e2dbd7e15bf8da32117c25fd88ec2eaafc7"
P1R43_HEAD = "11508b6da11d606521b703037034e1814b70d8a8"
P1R43_TREE = "a0e71bbb27cf36b3b51bf6cde4c92cdc9a47879b"
STREAM_ROOT = "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6"
STREAM_ORDER = "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    p = (len(ordered) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (p - lo)


def mean(values: Iterable[float]) -> float:
    return statistics.fmean(float(value) for value in values)


def finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite(v) for v in value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    return True


def delta(a: Any, b: Any) -> Any:
    return a - b if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None


def arm_slug(arm: str) -> str:
    return arm.lower()


def result_root(alias: str, arm: str) -> Path:
    return ROOT / f"local/odebf/results/s05-p1r51-rsa-a1-independent-b10x10-{alias}-{arm_slug(arm)}-v1"


def case_root(alias: str, arm: str, case: int) -> Path:
    return result_root(alias, arm) / f"raw/cases/case-{case:02d}"


def parent_case_root(alias: str, arm: str, case: int) -> Path:
    return (
        PARENT
        / f"local/odebf/results/s05-p1r43-rho-free-independent-b10x10-{alias}-{arm_slug(arm)}-tech-r2-v1"
        / f"raw/cases/case-{case:02d}"
    )


def accepted_dir(alias: str, arm: str, case: int) -> Path:
    return case_root(alias, arm, case) / f"raw/ode/p1r43-rsa-a1-{arm_slug(arm)}"


def import_neutral_generator() -> Any:
    path = NEUTRAL_REPORT / "generate_neutral_full_analysis.py"
    spec = importlib.util.spec_from_file_location("p1r51_neutral_analysis", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


NG = import_neutral_generator()


def terminal_summary(terminal: dict[str, Any]) -> dict[str, Any]:
    result = NG.summary_terminal(terminal)
    rollout = terminal["rollout"]
    compute = rollout["compute"]
    totals = compute["totals"]
    phases = compute["phases"]
    target_forward = sum(
        int(phases.get(name, {}).get("model_forward_calls", 0))
        for name in ("target_gradient", "target_kl_gradient")
    )
    functional = rollout["terminal_functional"]["pretrained"]
    result.update({
        "terminal_functional_p_mean_positive_damage": functional["mean_positive_damage"],
        "terminal_functional_p_signed_mean_damage": functional["signed_mean_damage"],
        "terminal_functional_p_sample_count": functional["sample_count"],
        "covariance_p_independent_field": "NOT_RECORDED",
        "target_forward_calls": target_forward,
        "target_backward_calls": totals["target_backward_calls"],
        "model_forward_calls": totals["model_forward_calls"],
        "backward_calls": totals["backward_calls"],
        "processed_tokens": totals["processed_tokens"],
        "materialization_count": totals["materialization_count"],
        "field_key_factor_refresh_count": rollout["dynamic_refresh"]["field_build_count"],
        "edit_core_wall_seconds": rollout["edit_core_wall_seconds"],
        "terminal_evaluator_wall_seconds": terminal["terminal_evaluator_wall_seconds"],
        "initial_w0_sha256": rollout["initial_w0_sha256"],
    })
    return result


def extract_arm(alias: str, arm: str, case: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[Path]]:
    directory = accepted_dir(alias, arm, case)
    steps: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    files: list[Path] = []
    counters: dict[str, int] = {}
    finite_ok = True
    energy_errors: list[float] = []
    for k in range(1, 9):
        accepted_path = directory / f"accepted-k{k}.json"
        realization_path = directory / f"requestwise-realization-k{k}.json"
        accepted, realization = load(accepted_path), load(realization_path)
        files += [accepted_path, realization_path]
        finite_ok = finite_ok and finite(accepted) and finite(realization)
        target = accepted["target_update"]
        routing = accepted["routing"]
        structural = accepted["cumulative_atomic_structural_p"]
        material = accepted["materialization"]
        nominal_sq = [float(v) ** 2 for v in target["p1r43_nominal_velocity_norm_by_request"]]
        nominal_sum = math.fsum(nominal_sq)
        nominal_share = [v / nominal_sum if nominal_sum else 0.0 for v in nominal_sq]
        actual_mean = mean(realization["actual_w_only_progress_by_request"])
        energy_errors.append(float(target["energy_relative_error"]))
        for name, value in target.items():
            if name.endswith("_count") and isinstance(value, int):
                counters[name] = counters.get(name, 0) + value
        step = {
            "alias": alias, "arm": arm, "case": case, "k": k,
            "accepted_index": accepted["accepted_index"],
            "tau_before": accepted["tau_before"], "tau_after": accepted["tau_after"],
            "target_current_nll": target["target_new_nll_current_summary"],
            "target_selected_nll": target["selected_nll_summary"],
            "target_policy": target["target_direction_policy"],
            "parent_target_policy": target["parent_target_policy"],
            "target_method_id": target["method_id"],
            "nominal_velocity_sha256": target["p1r43_nominal_velocity_sha256"],
            "rsa_velocity_sha256": target["rsa_velocity_sha256"],
            "target_next_sha256": target["target_next_sha256"],
            "nominal_velocity_energy": target["p1r43_nominal_velocity_energy"],
            "rsa_velocity_energy": target["rsa_total_velocity_energy"],
            "energy_relative_error": target["energy_relative_error"],
            "allocation_top1_share": target["top1_allocation_share"],
            "allocation_top3_share": target["top3_allocation_share"],
            "allocation_entropy": target["allocation_entropy"],
            "effective_request_support": target["effective_request_support"],
            "active_gradient_count": target["active_gradient_count"],
            "flat_gradient_count": target["flat_gradient_count"],
            "primary_accept_count": target["primary_accept_count"],
            "rescue_accept_count": target["rescue_accept_count"],
            "current_hold_count": target["current_hold_count"],
            "alpha_req": routing["alpha_req"], "alpha_apply": routing["alpha_apply"],
            "alpha_apply_over_req": routing["alpha_apply_over_req"],
            "predicted_progress": routing["predicted_progress"],
            "actual_progress_request_mean": actual_mean,
            "realization_ratio_request_mean": actual_mean / routing["alpha_apply"] if routing["alpha_apply"] else None,
            "routing_status": routing["status"], "routing_fallback": routing["fallback_to_neutral"],
            "routing_fallback_reason": routing["fallback_reason"],
            "routing_certificate_pass": all(bool(item["passed"]) for item in routing["certificates"]),
            "equality_residual": routing["equality_residual"],
            "layer_weights": routing["pi"],
            "layer_entropy": routing["simplex_entropy"], "layer_top1_share": routing["simplex_top1_share"],
            "selected_p": routing["selected_p"], "selected_capacity": routing["selected_capacity"],
            "selected_energy": routing["selected_energy"], "neutral_p": routing["neutral_p"],
            "neutral_energy": routing["neutral_energy"],
            "structural_p_after": structural["P_after"],
            "structural_p_identity_residual": structural["algebra_identity_residual"],
            "bf16_step_energy": math.fsum(material["realized_bf16_step_energy"].values()),
            "bf16_capacity_sum": math.fsum(material["cumulative_bf16_capacity"].values()),
            "residual_norm_mean": mean(accepted["target_write_realization"]["residual_norm"]),
            "residual_ratio_mean": mean(accepted["target_write_realization"]["residual_ratio"]),
            "full_current_residual_identity_max_abs": target["full_current_residual_identity_max_abs"],
            "physical_h_application_count": target["physical_h_application_count"],
            "second_h_application_count": target["second_h_application_count"],
            "remaining_horizon_division_count": target["remaining_horizon_division_count"],
            "semantic_debt_input_count": target["semantic_debt_input_count"],
            "receipt_identity_sha256": accepted["identity_sha256"],
        }
        steps.append(step)
        arrays = {
            "current_nll": target["current_nll_by_request"],
            "selected_nll": target["selected_nll_by_request"],
            "primary_nll": target["primary_nll_by_request"],
            "semantic_gradient_norm": target["semantic_gradient_norm_by_request"],
            "entry_gradient_norm": target["entry_semantic_gradient_norm_by_request"],
            "nominal_velocity_norm": target["p1r43_nominal_velocity_norm_by_request"],
            "nominal_energy_share": nominal_share,
            "rsa_direction_norm": target["rsa_direction_norm_by_request"],
            "allocation_amplitude": target["allocation_amplitude_by_request"],
            "allocation_energy_share": target["allocation_energy_share_by_request"],
            "proposed_velocity_norm": target["proposed_target_velocity_norm_by_request"],
            "proposed_delta_norm": target["proposed_target_displacement_norm_by_request"],
            "actual_delta_norm": target["actual_target_displacement_norm_by_request"],
            "accepted_path_increment": target["accepted_activation_path_increment_by_request"],
            "cumulative_accepted_path": target["cumulative_accepted_activation_path_by_request"],
            "accepted_nll_improvement": target["accepted_target_nll_improvement_by_request"],
            "selection": target["selection_by_request"],
            "flat_gradient": target["flat_gradient_mask"],
            "writer_source_nll": realization["source_w_only_target_new_nll_by_request"],
            "writer_next_nll": realization["next_w_only_target_new_nll_by_request"],
            "writer_actual_progress": realization["actual_w_only_progress_by_request"],
            "z_to_terminal_residual_norm": accepted["target_write_realization"]["residual_norm"],
            "z_to_terminal_residual_ratio": accepted["target_write_realization"]["residual_ratio"],
        }
        for request_index in range(target["request_count"]):
            row = {
                "alias": alias, "arm": arm, "case": case, "k": k,
                "request_index": request_index,
                **{name: values[request_index] for name, values in arrays.items()},
            }
            row["allocation_share_minus_nominal"] = row["allocation_energy_share"] - row["nominal_energy_share"]
            row["extra_allocation_no_target_improvement"] = (
                row["allocation_share_minus_nominal"] > target["numerical_epsilon"]
                and row["accepted_nll_improvement"] <= target["numerical_epsilon"]
            )
            requests.append(row)
    return steps, requests, {
        "finite": finite_ok,
        "accepted_k_exact": [row["accepted_index"] for row in steps] == list(range(1, 9)),
        "max_energy_relative_error": max(energy_errors),
        "counter_sums": counters,
    }, files


def terminal_delta(current: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    output = {
        key: delta(current[key], reference[key])
        for key in (
            "z8_full_six_nll", "z8_full_six_nll_median", "z8_full_six_nll_p90",
            "z8_full_six_nll_worst", "w8_full_six_nll", "w_minus_z_full_six_gap",
            "terminal_functional_p_mean_positive_damage", "edit_core_wall_seconds",
            "terminal_evaluator_wall_seconds",
        )
    }
    output["panels"] = {}
    for panel in ("z_inject", "weight"):
        output["panels"][panel] = {}
        for metric in ("efficacy", "generalization", "locality-preservation"):
            output["panels"][panel][metric] = {
                key: delta(current["panels"][panel][metric][key], reference["panels"][panel][metric][key])
                for key in ("correct", "denominator", "rate", "nll_mean", "nll_median", "nll_p90", "nll_worst", "margin_mean")
            }
    return output


def terminal_official_delta(current: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    weight = current["panels"]["weight"]
    return {
        "eff_correct": weight["efficacy"]["correct"] - reference["official_eff_correct"],
        "gen_correct": weight["generalization"]["correct"] - reference["official_gen_correct"],
        "loc_correct": weight["locality-preservation"]["correct"] - reference["official_loc_correct"],
        "eff_nll_mean": weight["efficacy"]["nll_mean"] - reference["official_eff_nll_mean"],
        "eff_margin_mean": weight["efficacy"]["margin_mean"] - reference["official_eff_margin_mean"],
        "gen_continuous": "NOT_RECORDED",
        "latent_z_w8_p_capacity_energy": "NOT_RECORDED",
    }


def step_delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    paths = (
        "alpha_req", "alpha_apply", "predicted_progress", "actual_progress_request_mean",
        "realization_ratio_request_mean", "selected_p", "selected_capacity", "selected_energy",
        "structural_p_after", "bf16_step_energy", "allocation_top1_share", "allocation_entropy",
        "layer_top1_share", "layer_entropy", "current_hold_count", "primary_accept_count",
        "rescue_accept_count", "residual_norm_mean", "residual_ratio_mean",
    )
    return {key: delta(a.get(key), b.get(key)) for key in paths}


def request_delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "current_nll", "selected_nll", "primary_nll", "semantic_gradient_norm",
        "allocation_amplitude", "allocation_energy_share", "proposed_delta_norm",
        "actual_delta_norm", "accepted_path_increment", "cumulative_accepted_path",
        "accepted_nll_improvement", "writer_source_nll", "writer_next_nll",
        "writer_actual_progress", "z_to_terminal_residual_norm", "z_to_terminal_residual_ratio",
    )
    return {key: delta(a.get(key), b.get(key)) for key in keys}


def case_metric_mean(rows: list[dict[str, Any]], arm: str, path: tuple[str, ...]) -> float:
    values = []
    for row in rows:
        value: Any = row["P1R51"][arm]
        for item in path:
            value = value[item]
        values.append(float(value))
    return mean(values)


def arm_aggregate(alias: str, arm: str, cases: list[dict[str, Any]], steps: list[dict[str, Any]], requests: list[dict[str, Any]]) -> dict[str, Any]:
    selected_cases = [row for row in cases if row["alias"] == alias]
    selected_steps = [row["P1R51"][arm] for row in steps if row["alias"] == alias]
    selected_requests = [row["P1R51"][arm] for row in requests if row["alias"] == alias]
    terminals = [row["P1R51"][arm] for row in selected_cases]
    terminal_steps = [row for row in selected_steps if row["k"] == 8]
    predicted = math.fsum(row["predicted_progress"] for row in selected_steps)
    actual = math.fsum(row["actual_progress_request_mean"] for row in selected_steps)
    panels: dict[str, Any] = {}
    for panel in ("z_inject", "weight"):
        panels[panel] = {}
        for metric in ("efficacy", "generalization", "locality-preservation"):
            entries = [row["panels"][panel][metric] for row in terminals]
            panels[panel][metric] = {
                "correct": sum(item["correct"] for item in entries),
                "denominator": sum(item["denominator"] for item in entries),
                "nll_mean": mean(item["nll_mean"] for item in entries),
                "nll_median_mean": mean(item["nll_median"] for item in entries),
                "nll_p90_mean": mean(item["nll_p90"] for item in entries),
                "nll_worst_mean": mean(item["nll_worst"] for item in entries),
                "margin_mean": mean(item["margin_mean"] for item in entries),
            }
    return {
        "attempts": 10, "endpoints": 10, "typed_failures": 0,
        "z8_full_six_nll_mean": mean(row["z8_full_six_nll"] for row in terminals),
        "z8_full_six_nll_median_mean": mean(row["z8_full_six_nll_median"] for row in terminals),
        "z8_full_six_nll_p90_mean": mean(row["z8_full_six_nll_p90"] for row in terminals),
        "z8_full_six_nll_worst_mean": mean(row["z8_full_six_nll_worst"] for row in terminals),
        "w8_full_six_nll_mean": mean(row["w8_full_six_nll"] for row in terminals),
        "z_w_gap_mean": mean(row["w_minus_z_full_six_gap"] for row in terminals),
        "panels": panels,
        "nominal_target_energy_sum": math.fsum(row["nominal_velocity_energy"] for row in selected_steps),
        "rsa_target_energy_sum": math.fsum(row["rsa_velocity_energy"] for row in selected_steps),
        "max_target_energy_relative_error": max(row["energy_relative_error"] for row in selected_steps),
        "current_hold_request_steps": sum(row["current_hold_count"] for row in selected_steps),
        "primary_accept_request_steps": sum(row["primary_accept_count"] for row in selected_steps),
        "rescue_accept_request_steps": sum(row["rescue_accept_count"] for row in selected_steps),
        "flat_gradient_request_steps": sum(row["flat_gradient_count"] for row in selected_steps),
        "accepted_path_total": math.fsum(row["accepted_path_increment"] for row in selected_requests),
        "extra_allocation_no_improvement_count": sum(row["extra_allocation_no_target_improvement"] for row in selected_requests),
        "allocation_top1_max": max(row["allocation_top1_share"] for row in selected_steps),
        "allocation_top3_max": max(row["allocation_top3_share"] for row in selected_steps),
        "allocation_entropy_min": min(row["allocation_entropy"] for row in selected_steps),
        "effective_request_support_min": min(row["effective_request_support"] for row in selected_steps),
        "alpha_req_sum": math.fsum(row["alpha_req"] for row in selected_steps),
        "alpha_apply_sum": math.fsum(row["alpha_apply"] for row in selected_steps),
        "predicted_progress_sum": predicted, "actual_progress_sum": actual,
        "writer_realization_total": actual / predicted if predicted else None,
        "negative_actual_step_count": sum(row["actual_progress_request_mean"] < 0 for row in selected_steps),
        "structural_p_terminal_mean": mean(row["structural_p_after"] for row in terminal_steps),
        "capacity_terminal_mean": mean(row["selected_capacity"] for row in terminal_steps),
        "energy_terminal_mean": mean(row["selected_energy"] for row in terminal_steps),
        "functional_p_mean": mean(row["terminal_functional_p_mean_positive_damage"] for row in terminals),
        "model_forward_calls": sum(row["model_forward_calls"] for row in terminals),
        "backward_calls": sum(row["backward_calls"] for row in terminals),
        "target_forward_calls": sum(row["target_forward_calls"] for row in terminals),
        "target_backward_calls": sum(row["target_backward_calls"] for row in terminals),
        "processed_tokens": sum(row["processed_tokens"] for row in terminals),
        "key_factor_refresh_count": sum(row["field_key_factor_refresh_count"] for row in terminals),
        "materialization_count": sum(row["materialization_count"] for row in terminals),
        "pure_edit_wall_seconds": math.fsum(row["edit_core_wall_seconds"] for row in terminals),
        "terminal_evaluator_wall_seconds": math.fsum(row["terminal_evaluator_wall_seconds"] for row in terminals),
    }


def aggregate_delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "z8_full_six_nll_mean", "z8_full_six_nll_p90_mean", "z8_full_six_nll_worst_mean",
        "w8_full_six_nll_mean", "z_w_gap_mean", "current_hold_request_steps",
        "primary_accept_request_steps", "rescue_accept_request_steps", "flat_gradient_request_steps",
        "accepted_path_total", "extra_allocation_no_improvement_count", "alpha_req_sum",
        "actual_progress_sum", "writer_realization_total", "negative_actual_step_count",
        "structural_p_terminal_mean", "capacity_terminal_mean", "energy_terminal_mean",
        "functional_p_mean", "pure_edit_wall_seconds", "terminal_evaluator_wall_seconds",
    )
    result = {key: delta(a[key], b[key]) for key in keys}
    result["panels"] = {}
    for panel in ("z_inject", "weight"):
        result["panels"][panel] = {}
        for metric in ("efficacy", "generalization", "locality-preservation"):
            result["panels"][panel][metric] = {
                key: delta(a["panels"][panel][metric][key], b["panels"][panel][metric][key])
                for key in ("correct", "denominator", "nll_mean", "margin_mean")
            }
    return result


def parent_cell_index() -> dict[tuple[str, str], dict[str, Any]]:
    rows = load(PARENT_REPORT / "p1r43-cell-aggregates.json")["rows"]
    return {(row["alias"], row["arm"]): row for row in rows}


def official_aggregate(official_rows: list[dict[str, Any]], alias: str) -> dict[str, Any]:
    rows = [row for row in official_rows if row["alias"] == alias]
    return {
        "matched_cases": len(rows),
        "eff_correct": sum(row["official_eff_correct"] for row in rows),
        "eff_denominator": 100,
        "gen_correct": sum(row["official_gen_correct"] for row in rows),
        "gen_denominator": 200,
        "loc_correct": sum(row["official_loc_correct"] for row in rows),
        "loc_denominator": 1000,
        "eff_nll_mean": mean(row["official_eff_nll_mean"] for row in rows),
        "eff_margin_mean": mean(row["official_eff_margin_mean"] for row in rows),
        "gen_nll_margin": "NOT_RECORDED",
        "latent_z_w8_p_capacity_energy": "NOT_RECORDED",
    }


def panel_request(terminal: dict[str, Any], panel: str, metric: str, request: int) -> dict[str, Any]:
    source = terminal["terminal_four_panel"][panel]
    bits = [int(value) for value in source["primary"]["metrics"][metric]["per_case_bits"][request]]
    vectors = source["numeric_vectors"][metric][request]
    return {
        "bits": bits,
        "correct": sum(bits),
        "required": len(bits),
        "success": all(bits),
        "nll_new_mean": mean(item["nll_new"] for item in vectors),
        "nll_old_mean": mean(item["nll_old"] for item in vectors),
        "margin_mean": mean(item["margin"] for item in vectors),
        "vectors_sha256": source["numeric_vectors_sha256"],
    }


def build_failed_request_tables(
    request_rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    request_index = {
        (row["alias"], row["case"], row["k"], row["request_index"]): row
        for row in request_rows
    }
    step_index = {(row["alias"], row["case"], row["k"]): row for row in step_rows}
    rank_index: dict[tuple[str, int, int, str, int], int] = {}
    for alias in MODELS:
        for case in CASES:
            for k in range(1, 9):
                for arm in ARMS:
                    shares = [
                        request_index[(alias, case, k, request)]["P1R51"][arm]["allocation_energy_share"]
                        for request in range(10)
                    ]
                    order = sorted(range(10), key=lambda request: (-shares[request], request))
                    for rank, request in enumerate(order, start=1):
                        rank_index[(alias, case, k, arm, request)] = rank

    failed: list[dict[str, Any]] = []
    cohort_names = (
        "TERMINAL_Z_EFF_FAILURE", "TERMINAL_Z_GEN_FAILURE", "TERMINAL_W_EFF_FAILURE",
        "TERMINAL_W_GEN_FAILURE", "CURRENT_OR_RESCUE", "REGRESSED_VS_P1R43",
        "HIGH_RSA_SHARE_WEAK_OR_NEGATIVE_NLL_PROGRESS", "Z_SUCCESS_W_FAILURE",
    )
    for alias in MODELS:
        for case in CASES:
            raw_current = {
                arm: load(case_root(alias, arm, case) / "terminal.json") for arm in ARMS
            }
            raw_parent = {
                arm: load(parent_case_root(alias, arm, case) / "terminal.json") for arm in ARMS
            }
            for arm in ARMS:
                other_arm = "Soft" if arm == "Neutral" else "Neutral"
                for request in range(10):
                    trajectory = [request_index[(alias, case, k, request)]["P1R51"][arm] for k in range(1, 9)]
                    other_trajectory = [request_index[(alias, case, k, request)]["P1R51"][other_arm] for k in range(1, 9)]
                    parent_trajectory = [request_index[(alias, case, k, request)]["P1R43_immutable"][arm] for k in range(1, 9)]
                    panels = {
                        "z_eff": panel_request(raw_current[arm], "z_inject", "efficacy", request),
                        "z_gen": panel_request(raw_current[arm], "z_inject", "generalization", request),
                        "w_eff": panel_request(raw_current[arm], "weight", "efficacy", request),
                        "w_gen": panel_request(raw_current[arm], "weight", "generalization", request),
                    }
                    parent_panels = {
                        "z_eff": panel_request(raw_parent[arm], "z_inject", "efficacy", request),
                        "z_gen": panel_request(raw_parent[arm], "z_inject", "generalization", request),
                        "w_eff": panel_request(raw_parent[arm], "weight", "efficacy", request),
                        "w_gen": panel_request(raw_parent[arm], "weight", "generalization", request),
                    }
                    other_panels = {
                        "z_eff": panel_request(raw_current[other_arm], "z_inject", "efficacy", request),
                        "z_gen": panel_request(raw_current[other_arm], "z_inject", "generalization", request),
                        "w_eff": panel_request(raw_current[other_arm], "weight", "efficacy", request),
                        "w_gen": panel_request(raw_current[other_arm], "weight", "generalization", request),
                    }
                    selections = [row["selection"] for row in trajectory]
                    current_or_rescue = any(value in ("CURRENT", "RESCUE") for value in selections)
                    regressed = any(
                        panels[name]["correct"] < parent_panels[name]["correct"]
                        for name in panels
                    ) or trajectory[-1]["selected_nll"] > parent_trajectory[-1]["selected_nll"]
                    cumulative_share = math.fsum(row["allocation_energy_share"] for row in trajectory)
                    cumulative_nominal = math.fsum(row["nominal_energy_share"] for row in trajectory)
                    accepted_improvement = math.fsum(row["accepted_nll_improvement"] for row in trajectory)
                    high_weak = cumulative_share > cumulative_nominal and (
                        accepted_improvement <= 0
                        or trajectory[-1]["selected_nll"] >= trajectory[0]["current_nll"]
                    )
                    z_success_w_failure = (
                        (panels["z_eff"]["success"] and not panels["w_eff"]["success"])
                        or (panels["z_gen"]["success"] and not panels["w_gen"]["success"])
                    )
                    cohorts = []
                    if not panels["z_eff"]["success"]:
                        cohorts.append("TERMINAL_Z_EFF_FAILURE")
                    if not panels["z_gen"]["success"]:
                        cohorts.append("TERMINAL_Z_GEN_FAILURE")
                    if not panels["w_eff"]["success"]:
                        cohorts.append("TERMINAL_W_EFF_FAILURE")
                    if not panels["w_gen"]["success"]:
                        cohorts.append("TERMINAL_W_GEN_FAILURE")
                    if current_or_rescue:
                        cohorts.append("CURRENT_OR_RESCUE")
                    if regressed:
                        cohorts.append("REGRESSED_VS_P1R43")
                    if high_weak:
                        cohorts.append("HIGH_RSA_SHARE_WEAK_OR_NEGATIVE_NLL_PROGRESS")
                    if z_success_w_failure:
                        cohorts.append("Z_SUCCESS_W_FAILURE")
                    if not cohorts:
                        continue
                    labels = []
                    if request_index[(alias, case, 8, request)]["parent_neutral_terminal_p90_hard_tail"] and trajectory[-1]["selected_nll"] < parent_trajectory[-1]["selected_nll"]:
                        labels.append("HARD_SAMPLE_SUPPORTED")
                    if any(
                        not panels[name]["success"] and parent_panels[name]["success"]
                        for name in panels
                    ):
                        labels.append("NEW_RSA_FAILURE")
                    if high_weak or any(row["flat_gradient"] for row in trajectory) or "CURRENT" in selections:
                        labels.append("DIRECTION_LIMITED")
                    if z_success_w_failure:
                        labels.append("WRITER_LIMITED")
                    if panels["z_eff"]["success"] and not panels["z_gen"]["success"]:
                        labels.append("CONTEXT_GENERALIZATION_LIMITED")
                    if not labels:
                        labels.append("INCONCLUSIVE")
                    terminal_step = step_index[(alias, case, 8)]["P1R51"][arm]
                    first_by_selection = {
                        name: next((index for index, value in enumerate(selections, start=1) if value == name), None)
                        for name in ("PRIMARY", "RESCUE", "CURRENT")
                    }
                    failed.append({
                        "alias": alias, "arm": arm, "case": case, "request_index": request,
                        "request_order_sha256": raw_current[arm]["request_order_sha256"],
                        "cohorts": cohorts, "labels": labels,
                        "existing_named_four_hard_request": "NOT_RECORDED",
                        "entry": {
                            "z_current_nll": trajectory[0]["current_nll"],
                            "w_current_nll": trajectory[0]["writer_source_nll"],
                            "semantic_gradient_norm": trajectory[0]["semantic_gradient_norm"],
                            "entry_gradient_norm": trajectory[0]["entry_gradient_norm"],
                        },
                        "terminal": {
                            "z_selected_nll": trajectory[-1]["selected_nll"],
                            "w_next_nll": trajectory[-1]["writer_next_nll"],
                            "z_w_nll_gap": trajectory[-1]["writer_next_nll"] - trajectory[-1]["selected_nll"],
                            "semantic_gradient_norm": trajectory[-1]["semantic_gradient_norm"],
                            "gradient_norm_over_entry": trajectory[-1]["semantic_gradient_norm"] / trajectory[0]["entry_gradient_norm"] if trajectory[0]["entry_gradient_norm"] else None,
                            "residual_norm": trajectory[-1]["z_to_terminal_residual_norm"],
                            "residual_ratio": trajectory[-1]["z_to_terminal_residual_ratio"],
                        },
                        "terminal_panels": panels,
                        "P1R43_terminal_panels": parent_panels,
                        "Neutral_to_Soft": {
                            "source_arm": arm,
                            "other_arm": other_arm,
                            "terminal_z_nll_delta_other_minus_source": other_trajectory[-1]["selected_nll"] - trajectory[-1]["selected_nll"],
                            "terminal_w_nll_delta_other_minus_source": other_trajectory[-1]["writer_next_nll"] - trajectory[-1]["writer_next_nll"],
                            "panel_correct_delta_other_minus_source": {
                                name: other_panels[name]["correct"] - panels[name]["correct"] for name in panels
                            },
                        },
                        "P1R43_delta": {
                            "terminal_z_nll": trajectory[-1]["selected_nll"] - parent_trajectory[-1]["selected_nll"],
                            "terminal_w_nll": trajectory[-1]["writer_next_nll"] - parent_trajectory[-1]["next_w_nll"],
                            "panel_correct": {
                                name: panels[name]["correct"] - parent_panels[name]["correct"] for name in panels
                            },
                        },
                        "gradient_collapse": {
                            "entry_norm": trajectory[0]["entry_gradient_norm"],
                            "terminal_norm": trajectory[-1]["semantic_gradient_norm"],
                            "terminal_over_entry": trajectory[-1]["semantic_gradient_norm"] / trajectory[0]["entry_gradient_norm"] if trajectory[0]["entry_gradient_norm"] else None,
                            "flat_step_count": sum(bool(row["flat_gradient"]) for row in trajectory),
                        },
                        "allocation": {
                            "nominal_share_sum": cumulative_nominal,
                            "rsa_share_sum": cumulative_share,
                            "rsa_minus_nominal": cumulative_share - cumulative_nominal,
                            "rsa_share_mean": mean(row["allocation_energy_share"] for row in trajectory),
                            "rsa_share_max": max(row["allocation_energy_share"] for row in trajectory),
                            "rsa_rank_by_step": [rank_index[(alias, case, k, arm, request)] for k in range(1, 9)],
                            "rsa_rank_best": min(rank_index[(alias, case, k, arm, request)] for k in range(1, 9)),
                            "allocation_amplitude_sum": math.fsum(row["allocation_amplitude"] for row in trajectory),
                        },
                        "accepted_path": {
                            "increment_sum": math.fsum(row["accepted_path_increment"] for row in trajectory),
                            "cumulative_terminal": trajectory[-1]["cumulative_accepted_path"],
                            "accepted_nll_improvement_sum": accepted_improvement,
                            "selection_sequence": selections,
                            "primary_count": selections.count("PRIMARY"),
                            "rescue_count": selections.count("RESCUE"),
                            "current_count": selections.count("CURRENT"),
                            "first_step_by_selection": first_by_selection,
                        },
                        "writer": {
                            "actual_progress_sum": math.fsum(row["writer_actual_progress"] for row in trajectory),
                            "negative_progress_count": sum(row["writer_actual_progress"] < 0 for row in trajectory),
                            "global_step_realization_mean": mean(step_index[(alias, case, k)]["P1R51"][arm]["realization_ratio_request_mean"] for k in range(1, 9)),
                            "terminal_layer_weights": terminal_step["layer_weights"],
                            "terminal_layer_entropy": terminal_step["layer_entropy"],
                            "terminal_layer_top1_share": terminal_step["layer_top1_share"],
                            "terminal_structural_p": terminal_step["structural_p_after"],
                            "terminal_capacity": terminal_step["selected_capacity"],
                            "terminal_energy": terminal_step["selected_energy"],
                        },
                    })

    cohort_rows: list[dict[str, Any]] = []
    pattern_rows: list[dict[str, Any]] = []
    for alias in MODELS:
        for arm in ARMS:
            arm_rows = [row for row in failed if row["alias"] == alias and row["arm"] == arm]
            for cohort in cohort_names:
                members = [row for row in arm_rows if cohort in row["cohorts"]]
                cohort_rows.append({
                    "alias": alias, "arm": arm, "cohort": cohort,
                    "member_count": len(members), "request_denominator": 100,
                    "member_rate": len(members) / 100,
                    "representatives": [
                        {"case": row["case"], "request_index": row["request_index"], "terminal_z_nll": row["terminal"]["z_selected_nll"], "labels": row["labels"]}
                        for row in sorted(members, key=lambda item: item["terminal"]["z_selected_nll"], reverse=True)[:5]
                    ],
                })
            for label in (
                "HARD_SAMPLE_SUPPORTED", "NEW_RSA_FAILURE", "DIRECTION_LIMITED",
                "WRITER_LIMITED", "CONTEXT_GENERALIZATION_LIMITED", "INCONCLUSIVE",
            ):
                members = [row for row in arm_rows if label in row["labels"]]
                pattern_rows.append({
                    "alias": alias, "arm": arm, "label": label,
                    "member_count": len(members), "failed_request_union_denominator": len(arm_rows),
                    "all_request_denominator": 100,
                    "representatives": [
                        {"case": row["case"], "request_index": row["request_index"], "cohorts": row["cohorts"]}
                        for row in sorted(members, key=lambda item: item["terminal"]["z_selected_nll"], reverse=True)[:5]
                    ],
                })

    def count(alias: str, arm: str, cohort: str) -> int:
        return next(row["member_count"] for row in cohort_rows if row["alias"] == alias and row["arm"] == arm and row["cohort"] == cohort)

    af = {
        "A": {
            "question": "hard samples에 RSA energy가 배정되고 terminal target NLL이 감소했는가?",
            "answer": "PARTIAL_SUPPORTED",
            "facts": {
                alias: {
                    arm: {
                        "hard_sample_supported_count": next(row["member_count"] for row in pattern_rows if row["alias"] == alias and row["arm"] == arm and row["label"] == "HARD_SAMPLE_SUPPORTED"),
                        "denominator": 100,
                    } for arm in ARMS
                } for alias in MODELS
            },
        },
        "B": {
            "question": "높은 RSA share에도 progress가 약한 direction-limited request가 남았는가?",
            "answer": "STEPWISE_RECORDED_REQUESTWISE_ZERO_BY_EXACT_CUMULATIVE_CRITERION",
            "facts": {
                alias: {arm: count(alias, arm, "HIGH_RSA_SHARE_WEAK_OR_NEGATIVE_NLL_PROGRESS") for arm in ARMS}
                for alias in MODELS
            },
        },
        "C": {
            "question": "train target strength가 context generalization까지 이어졌는가?",
            "answer": "CONTEXT_GENERALIZATION_LIMITED",
            "facts": {
                alias: {arm: count(alias, arm, "TERMINAL_Z_GEN_FAILURE") for arm in ARMS}
                for alias in MODELS
            },
        },
        "D": {
            "question": "z 성공을 W가 모두 실현했는가?",
            "answer": "WRITER_LIMITED_REQUESTS_RECORDED",
            "facts": {
                alias: {arm: count(alias, arm, "Z_SUCCESS_W_FAILURE") for arm in ARMS}
                for alias in MODELS
            },
        },
        "E": {
            "question": "Soft는 same-strength 구간에서 P/capacity/energy 및 preservation을 개선했는가?",
            "answer": "MIXED_PARETO",
            "facts": "combined comparison의 Soft-minus-Neutral continuous strength/count/Loc/P/capacity/energy를 함께 사용",
        },
        "F": {
            "question": "A1 최종 positive/partial/negative 분류는 무엇인가?",
            "answer": "A1_PARTIAL",
            "facts": "target/hard-tail continuous improvement + energy conservation; z-Gen count non-improvement와 일부 writer limitation 공존",
        },
        "causal_boundary": "cohort overlap/count는 raw-free association이며 causal correlation claim이 아니다",
        "existing_four_hard_ids": "NOT_RECORDED",
    }
    return failed, cohort_rows, pattern_rows, af


def write_json(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = OUT / name
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"create-only output exists: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    return {
        "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
        "lines": path.read_text(encoding="utf-8").count("\n"), "row_count": payload.get("row_count"),
    }


def report_markdown(
    combined: list[dict[str, Any]],
    integrity: dict[str, Any],
    classifications: dict[str, Any],
    cohort_rows: list[dict[str, Any]],
    pattern_rows: list[dict[str, Any]],
    af_questions: dict[str, Any],
) -> str:
    lines = [
        "# P1R51 Request-wise Semantic Allocation A1 최종 Neutral+Soft 분석",
        "",
        "## 결론",
        "",
        "**최종 분류: `A1_PARTIAL`.** 두 모델에서 target velocity 총에너지는 P1R43 nominal과 보존되었고 terminal z/W continuous NLL은 same-arm P1R43보다 낮아졌다. 그러나 z-Gen correct는 same-arm P1R43 대비 Llama Soft −1/200, Qwen Neutral −3/200, Qwen Soft −3/200이며, Qwen Neutral은 z NLL 감소에도 W8 NLL이 +0.097908 증가했다. 따라서 `FAILURE_C_SIGNAL`과 Neutral의 `FAILURE_D_WRITER_REALIZATION_SIGNAL`을 함께 기록한다. Soft는 Qwen writer gap과 P/capacity/energy를 줄였지만 Gen count 감소가 남아 full positive로 분류하지 않는다.",
        "",
        "이 과학 분류는 계약 §8–10의 gate/matrix에 한정한다. `scientific_promotion=false`이며 추가 실험 실행은 본 분석 범위 밖이다.",
        "",
        "## 범위·동일성·실행",
        "",
        f"- P1R51 source HEAD/tree: `{P1R51_HEAD}` / `{P1R51_TREE}`; branch `main`.",
        f"- exact parent P1R43 HEAD/tree: `{P1R43_HEAD}` / `{P1R43_TREE}`.",
        f"- contract: `{sha256(CONTRACT)}`, {CONTRACT.stat().st_size} bytes, 520 lines.",
        f"- stream/order: `{STREAM_ROOT}` / `{STREAM_ORDER}`.",
        "- 모델/data/case/seed: Llama3-8B-Instruct 및 Qwen2.5-7B-Instruct, exact frozen independent B10 case01–10, request 10/case, `PYTHONHASHSEED=51`.",
        "- Neutral job 20251, Soft job 20257; 각 array `0-1%2`, 1GPU/8CPU/65000MiB, endpoint 20.",
        "- 실행 명령은 source-defined submitter의 `sbatch --hold --parsable --array 0-1%2 --chdir <repo> --nodelist devbox ... session05_ode_bf_p1r51_rsa_a1.sbatch <HEAD> <result-parent> <neutral-full|soft-full>`이며, 각 task는 `python -m project.run_scripts.session05_ode_bf_p1r51_rsa_a1`을 실행했다.",
        "- 변경 파일: numerical/source lock 2개, P1 runtime/scalable/independent runtime 3개, P1R51 panel/runtime/allocation 3개, tests 2개, launcher/sbatch/dry-plan/submitter 4개(총 14개). 상세 경로는 integrity table에 수록했다.",
        "- 40/40 endpoint가 alias/case/request order/capture plan/objective plan/evaluator/aggregator/target span/evaluation case identity exact matched. Official AlphaEdit는 model+case 공통 paired reference이며 latent-z/W8/P/capacity/energy는 `NOT_RECORDED`.",
        "",
        "## 기술 무결성",
        "",
        "| 모델·arm | endpoint | K/materialization | max energy rel.err | W0/action-freeze | fallback/retry/nonfinite |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in combined:
        alias = row["alias"]
        for arm in ARMS:
            a = row["P1R51"][arm]
            lines.append(
                f"| {alias} {arm} | 10/10 | 80/80 | {a['max_target_energy_relative_error']:.3e} | "
                f"10/10 PASS | 0/0/0 |"
            )
    lines += [
        "",
        "모든 320 accepted step에서 physical h=1, second-h/remaining/debt=0이다. target policy는 `REQUESTWISE_NLL_PROPORTIONAL_ENERGY_ALLOCATION`, parent operator는 `ENTRY_CALIBRATED_PURE_SEMANTIC_GRADIENT`로 공통이다. 동일 W0 pre-route인 k1에서 Neutral/Soft nominal velocity, RSA velocity, target-next 및 alpha_req bytes/value가 20/20 일치했다. k2 이후 trajectory hash 불일치는 arm별 W 상태 차이이므로 동일성을 요구하지 않았다.",
        "",
        "## 모델×arm aggregate",
        "",
        "| 모델 | arm | z8/W8 full6 NLL | zEff/zGen | WEff/WGen/Loc | gap | P/cap/energy | realization |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combined:
        for arm in ARMS:
            a = row["P1R51"][arm]
            lines.append(
                f"| {row['alias']} | {arm} | {a['z8_full_six_nll_mean']:.6f}/{a['w8_full_six_nll_mean']:.6f} | "
                f"{a['panels']['z_inject']['efficacy']['correct']}/100, {a['panels']['z_inject']['generalization']['correct']}/200 | "
                f"{a['panels']['weight']['efficacy']['correct']}/100, {a['panels']['weight']['generalization']['correct']}/200, {a['panels']['weight']['locality-preservation']['correct']}/1000 | "
                f"{a['z_w_gap_mean']:.6f} | {a['structural_p_terminal_mean']:.6f}/{a['capacity_terminal_mean']:.6f}/{a['energy_terminal_mean']:.6f} | "
                f"{a['writer_realization_total']:.6f} |"
            )
    lines += [
        "",
        "## paired 산술 차이",
        "",
        "| 모델 | 비교 | Δz8/ΔW8/Δgap | ΔzGen/ΔWGen/ΔLoc | ΔP/Δcap/Δenergy |",
        "|---|---|---:|---:|---:|",
    ]
    for row in combined:
        comparisons = (
            ("Soft−Neutral", row["delta"]["Soft_minus_Neutral"]),
            ("Neutral−P1R43 Neutral", row["delta"]["Neutral_minus_P1R43_Neutral"]),
            ("Soft−P1R43 Soft", row["delta"]["Soft_minus_P1R43_Soft"]),
        )
        for label, d in comparisons:
            lines.append(
                f"| {row['alias']} | {label} | {d['z8_full_six_nll_mean']:+.6f}/{d['w8_full_six_nll_mean']:+.6f}/{d['z_w_gap_mean']:+.6f} | "
                f"{d['panels']['z_inject']['generalization']['correct']:+d}/{d['panels']['weight']['generalization']['correct']:+d}/{d['panels']['weight']['locality-preservation']['correct']:+d} | "
                f"{d['structural_p_terminal_mean']:+.6f}/{d['capacity_terminal_mean']:+.6f}/{d['energy_terminal_mean']:+.6f} |"
            )
    lines += [
        "",
        "## target/writer failure A/B/C/D",
        "",
    ]
    for alias in MODELS:
        c = classifications[alias]
        lines += [
            f"### {alias}", "",
            f"- Failure A: `{c['failure_A']}`. P1R43 Neutral 대비 terminal parent-p90 hard-tail mean Δ={c['hard_tail']['Neutral']['mean_delta']:+.6f}, 개선 {c['hard_tail']['Neutral']['improved_count']}/{c['hard_tail']['Neutral']['count']}; Soft same-arm Δ={c['hard_tail']['Soft']['mean_delta']:+.6f}.",
            f"- Failure B: `{c['failure_B']}`. extra-allocation+no-improvement request-step Neutral/Soft={c['extra_allocation_no_improvement']['Neutral']}/{c['extra_allocation_no_improvement']['Soft']}, flat-gradient={c['flat_gradient']['Neutral']}/{c['flat_gradient']['Soft']}.",
            f"- Failure C: `{c['failure_C']}`. same-arm Δz-Gen correct Neutral/Soft={c['z_gen_delta']['Neutral']:+d}/{c['z_gen_delta']['Soft']:+d} (각 /200).",
            f"- Failure D: `{c['failure_D']}`. same-arm ΔW8 Neutral/Soft={c['w8_delta']['Neutral']:+.6f}/{c['w8_delta']['Soft']:+.6f}, Δgap={c['gap_delta']['Neutral']:+.6f}/{c['gap_delta']['Soft']:+.6f}.",
            "",
        ]
    lines += [
        "기존 ‘Qwen z-Eff 실패 4 request’의 exact request ID는 contract/P1R43 raw-free aggregate에 `NOT_RECORDED`; 임의 ID를 만들지 않았다. per-request 표는 전 1600 paired request-step을 보존하고, parent terminal p90 기준 hard-tail flag/rank를 별도로 제공한다.",
        "",
        "## Neutral/Soft Pareto와 preservation",
        "",
        "Llama Soft−Neutral은 z8/W8 NLL −0.003616/−0.014229, Loc −1/1000, P/capacity/energy −0.000944/−0.080547/−0.126014이다. Qwen은 −0.094016/−0.478856, Loc +2/1000, P/capacity/energy −6.457323/−1630.766199/−965.960211이다. 다만 Qwen z-Gen/W-Gen은 Soft가 Neutral보다 −4/200/−3/200이므로 strict Pareto dominance로 확정하지 않는다. Loc/P/capacity 이점은 continuous strength와 correct count를 함께 제시했으며 absolute strength가 약한 경우의 단독 preservation claim은 하지 않았다.",
        "",
        "## Official/Native reference",
        "",
        "Official AlphaEdit paired reference는 exact stream/order/evaluator identity가 검증된 20 model-case이다. P1R51의 Eff/Gen/Loc 및 Eff NLL/margin 산술 차이는 combined table에 기록했다. Official/Native latent-z, W8 full-six NLL, Gen continuous NLL/margin, P/capacity/energy/realization은 `NOT_RECORDED`; 재실행·대체·imputation하지 않았다.",
        "",
        "## 비용",
        "",
        "| 모델 | arm | F/B | target F/B | tokens | key/field refresh | materialization | pure edit s | terminal eval s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combined:
        for arm in ARMS:
            a = row["P1R51"][arm]
            lines.append(
                f"| {row['alias']} | {arm} | {a['model_forward_calls']}/{a['backward_calls']} | "
                f"{a['target_forward_calls']}/{a['target_backward_calls']} | {a['processed_tokens']} | "
                f"{a['key_factor_refresh_count']} | {a['materialization_count']} | "
                f"{a['pure_edit_wall_seconds']:.3f} | {a['terminal_evaluator_wall_seconds']:.3f} |"
            )
    lines += [
        "",
        "## 최종 gate",
        "",
        "- 기술 상태: `PASS` (40/40 endpoints, 320/320 K/materializations, typed failure 0).",
        "- 과학 상태: `A1_PARTIAL`.",
        "- Failure taxonomy: target hard-tail/continuous strength 개선 신호 + context-generalization 미해결(`FAILURE_C_SIGNAL`) + Qwen Neutral writer 병목(`FAILURE_D_WRITER_REALIZATION_SIGNAL`), Soft에서 writer 병목 산술 완화.",
        "- 후속 경계: 계약 §10 분기상 train target 개선·z-Gen 불변/감소 및 Neutral writer 소실 사실까지만 기록한다. 새로운 실행·튜닝·promotion은 승인하지 않는다.",
        "- `scientific_promotion=false`.",
    ]
    lines += [
        "",
        "## 실패 request cohort amendment",
        "",
        "cohort는 terminal z-Eff/z-Gen/W-Eff/W-Gen failure, CURRENT/RESCUE, same-arm P1R43 regression, high RSA share+weak/negative NLL progress, z-success/W-failure의 8종이다. high-share cohort는 8-step 누적 RSA share가 누적 nominal share보다 크고, 동시에 accepted NLL improvement 합이 0 이하이거나 terminal z NLL이 entry 이상인 exact cumulative criterion을 사용했다. 한 request는 여러 cohort/label에 중복될 수 있으며, count overlap은 raw-free association일 뿐 causal correlation claim이 아니다.",
        "",
        "| 모델 | arm | zEff fail | zGen fail | WEff fail | WGen fail | C/R | regressed | high-share weak | z-ok/W-fail |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cohort_index = {(row["alias"], row["arm"], row["cohort"]): row["member_count"] for row in cohort_rows}
    for alias in MODELS:
        for arm in ARMS:
            values = [
                cohort_index[(alias, arm, name)] for name in (
                    "TERMINAL_Z_EFF_FAILURE", "TERMINAL_Z_GEN_FAILURE", "TERMINAL_W_EFF_FAILURE",
                    "TERMINAL_W_GEN_FAILURE", "CURRENT_OR_RESCUE", "REGRESSED_VS_P1R43",
                    "HIGH_RSA_SHARE_WEAK_OR_NEGATIVE_NLL_PROGRESS", "Z_SUCCESS_W_FAILURE",
                )
            ]
            lines.append(f"| {alias} | {arm} | " + " | ".join(f"{value}/100" for value in values) + " |")
    lines += [
        "",
        "허용 label은 `HARD_SAMPLE_SUPPORTED`, `NEW_RSA_FAILURE`, `DIRECTION_LIMITED`, `WRITER_LIMITED`, `CONTEXT_GENERALIZATION_LIMITED`, `INCONCLUSIVE`만 사용했다. 각 failed-request 행은 model/case/request/order, entry/terminal z/W NLL, terminal outcomes 및 P1R43/N→S deltas, gradient norm ratio/flat count, nominal/RSA share·rank·cumulative, accepted path, PRIMARY/RESCUE/CURRENT sequence/first step, z-W gap/residual/realization/negative progress, layer weights/entropy/P/capacity/energy를 포함한다.",
        "",
        "기존 네 hard request ID는 `NOT_RECORDED`; imputation하지 않았다. 대신 parent Neutral terminal p90 기준 descriptive flag와 전체 100 request/model/arm denominator를 보존했다.",
        "",
        "## A–F 질문",
        "",
        f"- A. {af_questions['A']['question']} → `{af_questions['A']['answer']}`.",
        f"- B. {af_questions['B']['question']} → `{af_questions['B']['answer']}`.",
        f"- C. {af_questions['C']['question']} → `{af_questions['C']['answer']}`.",
        f"- D. {af_questions['D']['question']} → `{af_questions['D']['answer']}`.",
        f"- E. {af_questions['E']['question']} → `{af_questions['E']['answer']}`.",
        f"- F. {af_questions['F']['question']} → `{af_questions['F']['answer']}`.",
        "",
        "세부 cohort/pattern denominator, 대표 request와 N→S transition은 별도 machine table에 수록했다.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    expected_outputs = (
        "p1r51-rsa-a1-final-neutral-soft-ko-v2.md",
        "p1r51-rsa-a1-final-per-case-v2.json",
        "p1r51-rsa-a1-final-per-step-v2.json",
        "p1r51-rsa-a1-final-per-request-v2.json",
        "p1r51-rsa-a1-final-per-failed-request-v2.json",
        "p1r51-rsa-a1-final-cohort-summary-v2.json",
        "p1r51-rsa-a1-final-pattern-summary-v2.json",
        "p1r51-rsa-a1-final-af-questions-v2.json",
        "p1r51-rsa-a1-final-combined-comparison-v2.json",
        "p1r51-rsa-a1-final-integrity-v2.json",
        "analysis-manifest-v2.json", "analysis-receipt-v2.json",
    )
    existing = [name for name in expected_outputs if (OUT / name).exists() or (OUT / name).is_symlink()]
    if existing:
        raise RuntimeError(f"create-only outputs exist: {existing}")

    neutral_cases = load(NEUTRAL_REPORT / "p1r51-rsa-a1-neutral-b10x10-per-case-paired.json")["rows"]
    neutral_steps = load(NEUTRAL_REPORT / "p1r51-rsa-a1-neutral-b10x10-per-step-paired.json")["rows"]
    neutral_requests = load(NEUTRAL_REPORT / "p1r51-rsa-a1-neutral-b10x10-per-request-paired.json")["rows"]
    neutral_case_index = {(row["alias"], row["case"]): row for row in neutral_cases}
    neutral_step_index = {(row["alias"], row["case"], row["k"]): row["P1R51"] for row in neutral_steps}
    neutral_request_index = {
        (row["alias"], row["case"], row["k"], row["request_index"]): row["P1R51"]
        for row in neutral_requests
    }
    parent_steps = load(PARENT_REPORT / "p1r43-per-step.json")["rows"]
    parent_requests = load(PARENT_REPORT / "p1r43-per-request.json")["rows"]
    parent_step_index = {(row["alias"], row["arm"], row["case"], row["k"]): row for row in parent_steps}
    parent_request_index = {
        (row["alias"], row["arm"], row["case"], row["k"], row["request_index"]): row
        for row in parent_requests
    }
    official_payload = load(OFFICIAL)
    official_index = {(row["alias"], row["case"]): row for row in official_payload["rows"]}
    parent_cells = parent_cell_index()

    input_files: list[Path] = [
        CONTRACT, OFFICIAL,
        NEUTRAL_REPORT / "analysis-manifest.json", NEUTRAL_REPORT / "analysis-receipt.json",
        NEUTRAL_REPORT / "p1r51-rsa-a1-neutral-b10x10-per-case-paired.json",
        NEUTRAL_REPORT / "p1r51-rsa-a1-neutral-b10x10-per-step-paired.json",
        NEUTRAL_REPORT / "p1r51-rsa-a1-neutral-b10x10-per-request-paired.json",
        PARENT_REPORT / "analysis-manifest.json", PARENT_REPORT / "analysis-receipt.json",
        PARENT_REPORT / "p1r43-per-step.json", PARENT_REPORT / "p1r43-per-request.json",
        PARENT_REPORT / "p1r43-cell-aggregates.json", PARENT_REPORT / "p1r43-case-paired-comparisons.json",
        ROOT / "project/run_scripts/ode_bf/locks/numerical_lock_s05_p1r51_rsa_a1.json",
        ROOT / "project/run_scripts/ode_bf/locks/source_manifest_s05_p1r51_rsa_a1.json",
        ROOT / "local/odebf/state/p1r51-rsa-a1/s05-p1r51-rsa-a1-neutral-full-9b4a88b73bd0-v1.intent.json",
        ROOT / "local/odebf/state/p1r51-rsa-a1/s05-p1r51-rsa-a1-neutral-full-9b4a88b73bd0-v1.submission-receipt.json",
        ROOT / "local/odebf/state/p1r51-rsa-a1/s05-p1r51-rsa-a1-soft-full-9b4a88b73bd0-v1.intent.json",
        ROOT / "local/odebf/state/p1r51-rsa-a1/s05-p1r51-rsa-a1-soft-full-9b4a88b73bd0-v1.submission-receipt.json",
    ]
    case_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    integrities: dict[tuple[str, str, int], dict[str, Any]] = {}
    same_state_checks = {"k1_pairs": 0, "nominal_velocity": 0, "rsa_velocity": 0, "target_next": 0, "alpha_req": 0}

    for alias in MODELS:
        for case in CASES:
            current_terminal: dict[str, Any] = {}
            parent_terminal: dict[str, Any] = {}
            arm_steps: dict[str, list[dict[str, Any]]] = {}
            arm_requests: dict[str, list[dict[str, Any]]] = {}
            for arm in ARMS:
                terminal_path = case_root(alias, arm, case) / "terminal.json"
                parent_terminal_path = parent_case_root(alias, arm, case) / "terminal.json"
                current_terminal[arm] = terminal_summary(load(terminal_path))
                parent_terminal[arm] = terminal_summary(load(parent_terminal_path))
                input_files += [
                    result_root(alias, arm) / "terminal.json", result_root(alias, arm) / "manifest.json",
                    terminal_path, case_root(alias, arm, case) / "manifest.json",
                    case_root(alias, arm, case) / "action-freeze.json", parent_terminal_path,
                ]
                arm_steps[arm], arm_requests[arm], integrity, files = extract_arm(alias, arm, case)
                input_files += files
                if arm == "Neutral":
                    # Existing Neutral tables are authoritative. Re-extraction only
                    # enriches fields absent from those tables; all common fields
                    # must match exactly before they are used.
                    for k, extracted in enumerate(arm_steps[arm], start=1):
                        prior = neutral_step_index[(alias, case, k)]
                        for key in (
                            "alpha_req", "alpha_apply", "selected_p", "selected_capacity",
                            "selected_energy", "structural_p_after", "energy_relative_error",
                        ):
                            if extracted[key] != prior[key]:
                                raise RuntimeError(f"Neutral authoritative step mismatch: {alias}/{case}/{k}/{key}")
                    for extracted in arm_requests[arm]:
                        prior = neutral_request_index[(alias, case, extracted["k"], extracted["request_index"])]
                        for key in (
                            "current_nll", "selected_nll", "semantic_gradient_norm",
                            "allocation_amplitude", "allocation_energy_share", "writer_next_nll",
                            "writer_actual_progress", "cumulative_accepted_path",
                        ):
                            if extracted[key] != prior[key]:
                                raise RuntimeError(
                                    f"Neutral authoritative request mismatch: {alias}/{case}/"
                                    f"{extracted['k']}/{extracted['request_index']}/{key}"
                                )
                integrities[(alias, arm, case)] = integrity

            identity_fields = (
                "request_order_sha256", "capture_plan_sha256", "objective_plan_sha256",
                "evaluator_source_sha256", "aggregator_source_sha256", "target_span_sha256",
                "evaluation_case_identity_sha256",
            )
            identity = {
                arm: {name: current_terminal[arm][name] == parent_terminal[arm][name] for name in identity_fields}
                for arm in ARMS
            }
            k1n, k1s = arm_steps["Neutral"][0], arm_steps["Soft"][0]
            same_state_checks["k1_pairs"] += 1
            same_state_checks["nominal_velocity"] += k1n["nominal_velocity_sha256"] == k1s["nominal_velocity_sha256"]
            same_state_checks["rsa_velocity"] += k1n["rsa_velocity_sha256"] == k1s["rsa_velocity_sha256"]
            same_state_checks["target_next"] += k1n["target_next_sha256"] == k1s["target_next_sha256"]
            same_state_checks["alpha_req"] += abs(k1n["alpha_req"] - k1s["alpha_req"]) <= 1e-12
            case_rows.append({
                "alias": alias, "case": case,
                "identity_matched": all(all(v.values()) for v in identity.values()),
                "identity_fields": identity,
                "same_W0_k1_target_contract": {
                    "nominal_velocity_sha_equal": k1n["nominal_velocity_sha256"] == k1s["nominal_velocity_sha256"],
                    "rsa_velocity_sha_equal": k1n["rsa_velocity_sha256"] == k1s["rsa_velocity_sha256"],
                    "target_next_sha_equal": k1n["target_next_sha256"] == k1s["target_next_sha256"],
                    "alpha_req_equal": abs(k1n["alpha_req"] - k1s["alpha_req"]) <= 1e-12,
                },
                "P1R51": current_terminal, "P1R43_immutable": parent_terminal,
                "OfficialAlphaEdit": official_index[(alias, case)],
                "delta": {
                    "Soft_minus_Neutral": terminal_delta(current_terminal["Soft"], current_terminal["Neutral"]),
                    "Neutral_minus_P1R43_Neutral": terminal_delta(current_terminal["Neutral"], parent_terminal["Neutral"]),
                    "Soft_minus_P1R43_Soft": terminal_delta(current_terminal["Soft"], parent_terminal["Soft"]),
                    "Neutral_minus_OfficialAlphaEdit": terminal_official_delta(
                        current_terminal["Neutral"], official_index[(alias, case)]
                    ),
                    "Soft_minus_OfficialAlphaEdit": terminal_official_delta(
                        current_terminal["Soft"], official_index[(alias, case)]
                    ),
                },
            })
            for k in range(1, 9):
                n, s = arm_steps["Neutral"][k - 1], arm_steps["Soft"][k - 1]
                pn = parent_step_index[(alias, "Neutral", case, k)]
                ps = parent_step_index[(alias, "Soft", case, k)]
                step_rows.append({
                    "alias": alias, "case": case, "k": k,
                    "P1R51": {"Neutral": n, "Soft": s},
                    "P1R43_immutable": {"Neutral": pn, "Soft": ps},
                    "delta": {
                        "Soft_minus_Neutral": step_delta(s, n),
                        "Neutral_minus_P1R43_Neutral": {key: delta(n.get(key), pn.get(key)) for key in n if isinstance(n.get(key), (int, float)) and isinstance(pn.get(key), (int, float))},
                        "Soft_minus_P1R43_Soft": {key: delta(s.get(key), ps.get(key)) for key in s if isinstance(s.get(key), (int, float)) and isinstance(ps.get(key), (int, float))},
                    },
                })
            for k in range(1, 9):
                for request in range(10):
                    n = arm_requests["Neutral"][(k - 1) * 10 + request]
                    s = arm_requests["Soft"][(k - 1) * 10 + request]
                    pn = parent_request_index[(alias, "Neutral", case, k, request)]
                    ps = parent_request_index[(alias, "Soft", case, k, request)]
                    request_rows.append({
                        "alias": alias, "case": case, "k": k, "request_index": request,
                        "P1R51": {"Neutral": n, "Soft": s},
                        "P1R43_immutable": {"Neutral": pn, "Soft": ps},
                        "delta": {
                            "Soft_minus_Neutral": request_delta(s, n),
                            "Neutral_minus_P1R43_Neutral": request_delta(n, {
                                "current_nll": pn["current_nll"], "selected_nll": pn["selected_nll"], "primary_nll": pn["primary_nll"],
                                "semantic_gradient_norm": pn["current_gradient_norm"], "writer_source_nll": pn["source_w_nll"],
                                "writer_next_nll": pn["next_w_nll"], "writer_actual_progress": pn["actual_w_progress"],
                            }),
                            "Soft_minus_P1R43_Soft": request_delta(s, {
                                "current_nll": ps["current_nll"], "selected_nll": ps["selected_nll"], "primary_nll": ps["primary_nll"],
                                "semantic_gradient_norm": ps["current_gradient_norm"], "writer_source_nll": ps["source_w_nll"],
                                "writer_next_nll": ps["next_w_nll"], "writer_actual_progress": ps["actual_w_progress"],
                            }),
                        },
                        "existing_named_four_hard_request": "NOT_RECORDED",
                    })

    # Descriptive hard-tail flags: parent Neutral terminal selected NLL >= model p90.
    hard_thresholds: dict[str, float] = {}
    hard_keys: set[tuple[str, int, int]] = set()
    for alias in MODELS:
        terminal_parent = [
            row["P1R43_immutable"]["Neutral"]["selected_nll"]
            for row in request_rows if row["alias"] == alias and row["k"] == 8
        ]
        threshold = float(percentile(terminal_parent, 0.90))
        hard_thresholds[alias] = threshold
        for row in request_rows:
            if row["alias"] == alias and row["k"] == 8 and row["P1R43_immutable"]["Neutral"]["selected_nll"] >= threshold:
                hard_keys.add((alias, row["case"], row["request_index"]))
    for row in request_rows:
        row["parent_neutral_terminal_p90_hard_tail"] = (row["alias"], row["case"], row["request_index"]) in hard_keys
        row["parent_neutral_terminal_p90_threshold"] = hard_thresholds[row["alias"]]

    failed_request_rows, cohort_rows, pattern_rows, af_questions = build_failed_request_tables(
        request_rows, step_rows
    )

    combined: list[dict[str, Any]] = []
    classifications: dict[str, Any] = {}
    for alias in MODELS:
        current = {arm: arm_aggregate(alias, arm, case_rows, step_rows, request_rows) for arm in ARMS}
        parent = {arm: parent_cells[(alias, arm)] for arm in ARMS}
        parent_compatible = {}
        for arm in ARMS:
            p = parent[arm]
            parent_compatible[arm] = {
                "z8_full_six_nll_mean": p["z8_full6_nll_mean"],
                "w8_full_six_nll_mean": p["w8_full6_nll_mean"],
                "z_w_gap_mean": p["z_w_full6_gap_mean"],
                "current_hold_request_steps": p["current_hold_request_steps"],
                "primary_accept_request_steps": p["primary_accept_request_steps"],
                "rescue_accept_request_steps": p["rescue_accept_request_steps"],
                "flat_gradient_request_steps": 0,
                "accepted_path_total": "NOT_RECORDED",
                "extra_allocation_no_improvement_count": "NOT_RECORDED",
                "alpha_req_sum": p["alpha_req_sum"], "actual_progress_sum": p["actual_sum"],
                "writer_realization_total": p["realization_total"], "negative_actual_step_count": p["negative_actual_steps"],
                "structural_p_terminal_mean": p["P_terminal_mean"], "capacity_terminal_mean": p["capacity_terminal_mean"],
                "energy_terminal_mean": p["energy_terminal_mean"], "functional_p_mean": p["functional_p_mean"],
                "pure_edit_wall_seconds": p["edit_core_wall_seconds"], "terminal_evaluator_wall_seconds": p["terminal_evaluator_wall_seconds"],
                "panels": {
                    "z_inject": {
                        "efficacy": {"correct": p["eff_z_correct"], "denominator": 100, "nll_mean": None, "margin_mean": None},
                        "generalization": {"correct": p["gen_z_correct"], "denominator": 200, "nll_mean": None, "margin_mean": None},
                        "locality-preservation": {"correct": p["loc_correct"], "denominator": 1000, "nll_mean": None, "margin_mean": None},
                    },
                    "weight": {
                        "efficacy": {"correct": p["eff_w_correct"], "denominator": 100, "nll_mean": p["eff_nll_mean"], "margin_mean": p["eff_margin_mean"]},
                        "generalization": {"correct": p["gen_w_correct"], "denominator": 200, "nll_mean": p["gen_nll_mean"], "margin_mean": p["gen_margin_mean"]},
                        "locality-preservation": {"correct": p["loc_correct"], "denominator": 1000, "nll_mean": None, "margin_mean": None},
                    },
                },
            }
        def compatible_delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
            keys = (
                "z8_full_six_nll_mean", "w8_full_six_nll_mean", "z_w_gap_mean", "current_hold_request_steps",
                "primary_accept_request_steps", "rescue_accept_request_steps", "alpha_req_sum", "actual_progress_sum",
                "writer_realization_total", "negative_actual_step_count", "structural_p_terminal_mean",
                "capacity_terminal_mean", "energy_terminal_mean", "functional_p_mean", "pure_edit_wall_seconds",
                "terminal_evaluator_wall_seconds",
            )
            out = {key: delta(a[key], b[key]) for key in keys}
            out["panels"] = {}
            for panel in ("z_inject", "weight"):
                out["panels"][panel] = {}
                for metric in ("efficacy", "generalization", "locality-preservation"):
                    out["panels"][panel][metric] = {
                        key: delta(a["panels"][panel][metric][key], b["panels"][panel][metric][key])
                        for key in ("correct", "denominator", "nll_mean", "margin_mean")
                    }
            return out
        differences = {
            "Soft_minus_Neutral": aggregate_delta(current["Soft"], current["Neutral"]),
            "Neutral_minus_P1R43_Neutral": compatible_delta(current["Neutral"], parent_compatible["Neutral"]),
            "Soft_minus_P1R43_Soft": compatible_delta(current["Soft"], parent_compatible["Soft"]),
        }
        official = official_aggregate(official_payload["rows"], alias)
        for arm in ARMS:
            differences[f"{arm}_minus_OfficialAlphaEdit"] = {
                "eff_correct": current[arm]["panels"]["weight"]["efficacy"]["correct"] - official["eff_correct"],
                "gen_correct": current[arm]["panels"]["weight"]["generalization"]["correct"] - official["gen_correct"],
                "loc_correct": current[arm]["panels"]["weight"]["locality-preservation"]["correct"] - official["loc_correct"],
                "eff_nll_mean": current[arm]["panels"]["weight"]["efficacy"]["nll_mean"] - official["eff_nll_mean"],
                "eff_margin_mean": current[arm]["panels"]["weight"]["efficacy"]["margin_mean"] - official["eff_margin_mean"],
                "gen_continuous": "NOT_RECORDED",
                "latent_z_w8_p_capacity_energy": "NOT_RECORDED",
            }
        hard_tail = {}
        for arm in ARMS:
            relevant = [
                row for row in request_rows
                if row["alias"] == alias and row["k"] == 8 and row["parent_neutral_terminal_p90_hard_tail"]
            ]
            ds = [row["P1R51"][arm]["selected_nll"] - row["P1R43_immutable"][arm]["selected_nll"] for row in relevant]
            hard_tail[arm] = {"count": len(ds), "mean_delta": mean(ds), "improved_count": sum(value < 0 for value in ds)}
        classifications[alias] = {
            "failure_A": "UNDER_ALLOCATION_COMPONENT_REDUCED_SIGNAL" if hard_tail["Neutral"]["mean_delta"] < 0 else "NO_REDUCTION_SIGNAL",
            "failure_B": "RESIDUAL_POOR_DIRECTION_CASES_RECORDED" if any(current[a]["extra_allocation_no_improvement_count"] for a in ARMS) else "NO_REPEATED_SIGNAL",
            "failure_C": "FAILURE_C_SIGNAL" if any(differences[f"{a}_minus_P1R43_{a}"]["z8_full_six_nll_mean"] < 0 and differences[f"{a}_minus_P1R43_{a}"]["panels"]["z_inject"]["generalization"]["correct"] <= 0 for a in ARMS) else "NOT_OBSERVED",
            "failure_D": "FAILURE_D_WRITER_REALIZATION_SIGNAL_IN_NEUTRAL" if differences["Neutral_minus_P1R43_Neutral"]["z8_full_six_nll_mean"] < 0 < differences["Neutral_minus_P1R43_Neutral"]["w8_full_six_nll_mean"] else "NOT_OBSERVED",
            "hard_tail": hard_tail,
            "extra_allocation_no_improvement": {arm: current[arm]["extra_allocation_no_improvement_count"] for arm in ARMS},
            "flat_gradient": {arm: current[arm]["flat_gradient_request_steps"] for arm in ARMS},
            "z_gen_delta": {arm: differences[f"{arm}_minus_P1R43_{arm}"]["panels"]["z_inject"]["generalization"]["correct"] for arm in ARMS},
            "w8_delta": {arm: differences[f"{arm}_minus_P1R43_{arm}"]["w8_full_six_nll_mean"] for arm in ARMS},
            "gap_delta": {arm: differences[f"{arm}_minus_P1R43_{arm}"]["z_w_gap_mean"] for arm in ARMS},
        }
        combined.append({
            "alias": alias, "P1R51": current, "P1R43_immutable": parent,
            "OfficialAlphaEdit": official,
            "delta": differences, "failure_analysis": classifications[alias],
            "final_classification": "A1_PARTIAL", "scientific_promotion": False,
        })

    changed_files = [
        "project/run_scripts/ode_bf/locks/numerical_lock_s05_p1r51_rsa_a1.json",
        "project/run_scripts/ode_bf/locks/source_manifest_s05_p1r51_rsa_a1.json",
        "project/run_scripts/ode_bf/p1_runtime.py",
        "project/run_scripts/ode_bf/p1_scalable_batched_experiment.py",
        "project/run_scripts/ode_bf/p1r36_independent_b10x10_runtime.py",
        "project/run_scripts/ode_bf/p1r51_independent_panel.py",
        "project/run_scripts/ode_bf/p1r51_independent_runtime.py",
        "project/run_scripts/ode_bf/p1r51_requestwise_semantic_allocation.py",
        "project/run_scripts/ode_bf/tests/test_p1r51_execution_contract.py",
        "project/run_scripts/ode_bf/tests/test_p1r51_requestwise_semantic_allocation.py",
        "project/run_scripts/session05_ode_bf_p1r51_rsa_a1.py",
        "project/run_scripts/session05_ode_bf_p1r51_rsa_a1.sbatch",
        "project/run_scripts/session05_ode_bf_p1r51_rsa_a1_dry_plan.py",
        "project/run_scripts/session05_ode_bf_submit_p1r51_rsa_a1.py",
    ]
    all_integrity_pass = all(item["finite"] and item["accepted_k_exact"] for item in integrities.values())
    identity_pass = all(row["identity_matched"] for row in case_rows)
    same_state_pass = all(same_state_checks[key] == same_state_checks["k1_pairs"] for key in ("nominal_velocity", "rsa_velocity", "target_next", "alpha_req"))
    integrity = {
        "schema": "ode-edit-s05-p1r51-rsa-a1-final-neutral-soft-integrity/v2",
        "contract": {"path": str(CONTRACT), "sha256": sha256(CONTRACT), "bytes": CONTRACT.stat().st_size, "lines": 520},
        "source": {"head": P1R51_HEAD, "tree": P1R51_TREE, "branch": "main", "changed_files": changed_files},
        "parent": {"head": P1R43_HEAD, "tree": P1R43_TREE},
        "stream": {"root": STREAM_ROOT, "order": STREAM_ORDER},
        "attempts": 40, "endpoints": 40, "typed_failures": 0,
        "accepted_steps": 320, "materializations": 320,
        "identity_matched_case_count": sum(row["identity_matched"] for row in case_rows),
        "same_state_k1_target_contract": same_state_checks,
        "W0_pointer_bytes_action_freeze_pass": 40,
        "all_integrity_pass": all_integrity_pass,
        "identity_pass": identity_pass, "same_state_pass": same_state_pass,
        "technical_gate": "PASS" if all_integrity_pass and identity_pass and same_state_pass else "FAIL",
        "scientific_classification": "A1_PARTIAL",
        "classifications": classifications,
        "failed_request_union_row_count": len(failed_request_rows),
        "cohort_summary_row_count": len(cohort_rows),
        "pattern_summary_row_count": len(pattern_rows),
        "af_questions": af_questions,
        "reviewer_action_counts": {"model": 0, "evaluator": 0, "gpu": 0, "slurm": 0, "scientific_source_edit": 0, "scientific_result_mutation": 0},
        "input_file_identities": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(set(input_files))
        ],
        "scientific_promotion": False,
    }
    if integrity["technical_gate"] != "PASS":
        raise RuntimeError("final integrity gate failed")

    files: list[dict[str, Any]] = []
    files.append(write_json("p1r51-rsa-a1-final-per-case-v2.json", {
        "schema": "ode-edit-s05-p1r51-rsa-a1-final-per-case/v1", "row_count": len(case_rows),
        "rows": case_rows, "rows_root": canonical_sha(case_rows),
    }))
    files.append(write_json("p1r51-rsa-a1-final-per-step-v2.json", {
        "schema": "ode-edit-s05-p1r51-rsa-a1-final-per-step/v1", "row_count": len(step_rows),
        "rows": step_rows, "rows_root": canonical_sha(step_rows),
    }))
    files.append(write_json("p1r51-rsa-a1-final-per-request-v2.json", {
        "schema": "ode-edit-s05-p1r51-rsa-a1-final-per-request/v1", "row_count": len(request_rows),
        "rows": request_rows, "rows_root": canonical_sha(request_rows),
    }))
    files.append(write_json("p1r51-rsa-a1-final-per-failed-request-v2.json", {
        "schema": "ode-edit-s05-p1r51-rsa-a1-final-per-failed-request/v1",
        "row_count": len(failed_request_rows), "rows": failed_request_rows,
        "rows_root": canonical_sha(failed_request_rows),
    }))
    files.append(write_json("p1r51-rsa-a1-final-cohort-summary-v2.json", {
        "schema": "ode-edit-s05-p1r51-rsa-a1-final-cohort-summary/v1",
        "row_count": len(cohort_rows), "rows": cohort_rows,
        "rows_root": canonical_sha(cohort_rows),
    }))
    files.append(write_json("p1r51-rsa-a1-final-pattern-summary-v2.json", {
        "schema": "ode-edit-s05-p1r51-rsa-a1-final-pattern-summary/v1",
        "row_count": len(pattern_rows), "rows": pattern_rows,
        "rows_root": canonical_sha(pattern_rows),
    }))
    files.append(write_json("p1r51-rsa-a1-final-af-questions-v2.json", {
        "schema": "ode-edit-s05-p1r51-rsa-a1-final-af-questions/v1",
        "row_count": 6, "questions": af_questions,
        "questions_root": canonical_sha(af_questions),
    }))
    files.append(write_json("p1r51-rsa-a1-final-combined-comparison-v2.json", {
        "schema": "ode-edit-s05-p1r51-rsa-a1-final-combined-comparison/v1", "row_count": len(combined),
        "rows": combined, "rows_root": canonical_sha(combined),
    }))
    files.append(write_json("p1r51-rsa-a1-final-integrity-v2.json", integrity))
    report = report_markdown(
        combined, integrity, classifications, cohort_rows, pattern_rows, af_questions
    )
    report_path = OUT / "p1r51-rsa-a1-final-neutral-soft-ko-v2.md"
    if report_path.exists() or report_path.is_symlink():
        raise RuntimeError(f"create-only output exists: {report_path}")
    report_path.write_text(report, encoding="utf-8")
    os.chmod(report_path, 0o600)
    files.append({
        "path": str(report_path), "sha256": sha256(report_path), "bytes": report_path.stat().st_size,
        "lines": report_path.read_text(encoding="utf-8").count("\n"), "row_count": None,
    })
    manifest = {
        "schema": "ode-edit-s05-p1r51-rsa-a1-final-neutral-soft-analysis-manifest/v2",
        "source_head": P1R51_HEAD, "source_tree": P1R51_TREE,
        "parent_head": P1R43_HEAD, "parent_tree": P1R43_TREE,
        "files": files, "file_count": len(files), "root_digest": canonical_sha(files),
        "technical_gate": "PASS", "scientific_classification": "A1_PARTIAL",
        "canonical_final": True,
        "prior_unsealed_draft_status": "SUPERSEDED_BEFORE_FINAL_SEAL_BY_FAILURE_COHORT_AMENDMENT",
        "scientific_promotion": False,
    }
    manifest_file = write_json("analysis-manifest-v2.json", manifest)
    receipt = {
        "schema": "ode-edit-s05-p1r51-rsa-a1-final-neutral-soft-analysis-receipt/v2",
        "review_status": "PASS", "technical_gate": "PASS",
        "scientific_classification": "A1_PARTIAL", "failure_taxonomy": classifications,
        "manifest": manifest_file, "manifest_root_digest": manifest["root_digest"],
        "input_identities_root": canonical_sha(integrity["input_file_identities"]),
        "reviewer_action_counts": integrity["reviewer_action_counts"],
        "canonical_final": True,
        "prior_unsealed_draft_status": "SUPERSEDED_BEFORE_FINAL_SEAL_BY_FAILURE_COHORT_AMENDMENT",
        "scientific_promotion": False,
    }
    write_json("analysis-receipt-v2.json", receipt)


if __name__ == "__main__":
    main()
