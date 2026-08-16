#!/usr/bin/env python3
"""Create-once raw-free production analysis for P2R7 job 20161.

This is an observation-only report builder.  It reads immutable P2R7 receipts and
allowed immutable P2R2/Official/P2R7-pilot references.  It performs no model,
evaluator, GPU, Slurm, or tracked-source action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


INSTRUCTION_ID = "ODEEDIT-S05-P2R7-P2TARGET-P1DW-SHARED-WRITER-V1"
JOB_ID = "20161"
SOURCE_HEAD = "a7e533a6f65dec7add3f4f6b513f10d51ae81640"
SOURCE_TREE = "601af3498b6d1e957c1017b9096cfa11d05c29ee"
SOURCE_PARENT = "581684696b1c921732a5d9cef86bbe0060661836"
P2R2_PARENT = "c97e8619b42da7954ce0e824c215a8a82d70589a"
CONTRACT_SHA = "d58e5c067d5687559e2bd809d9c5da17c4d546efa6e8feb679f3038f7ebaee5b"
STREAM_ROOT = "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6"
ORDER_SHA = "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c"
EVALUATOR_SHA = "25c3f49039e7bacb58de6d9ab8139f6f24f21532434a08690e9ec71ea939e145"
AGGREGATOR_SHA = "64f009b2fb648627a956b95abd801838d86edf04c8e1888d90ade0ec07b745c0"
NUMERICAL_LOCK_SHA = "021c42bbd245361f4e285b5c9df195b400ef18c7737187f062d6f02d0404736b"
NUMERICAL_LOCK_ROOT = "eeac97bc96d3484beb6ba51ac850f0093f708abe2e967a504b525f0881d3c3ce"
SOURCE_MANIFEST_SHA = "dd9a39f9507e5ccd70c154539ad0b4f9c7a93615434c8c5219c4c7111add8065"
SOURCE_MANIFEST_ROOT = "d58152aa216940369b44bf8bc68f27581d0327639b9334fe5d07464707a9bab9"
PILOT_REPORT_SHA = "14baf3cdb492b67db0ec80aec7429de818ac36c7318482ebbb7f64bd8051ddfa"
PILOT_ENDPOINT_SHA = "5940ec8bd328f63bcacc90f91fccd5ee21e879e8a2368aea7942eb17afbe7de1"
PILOT_STEP_SHA = "6e4bffaa0dd0087d667b551dfb2e03772a8c0997586ee7b7578020f946bd141e"
P2R2_REPORT_SHA = "747365627c3c750d016335ec9fb7ae02c1bbdb480d0563a37b7d1c9c6f094606"
P2R2_CASE_SHA = "223143f21e20d852aeb6fccba9b18a8fbaedf17c095813700104ee1d12456e2e"
P2R2_STEP_REQUEST_SHA = "cf7d2d41cb2a8bb93b449e32f03facabca13d9ee3c8a306f8dc6bca417dd28cd"
P2R2_COMPARISON_SHA = "cf95fc8d09a6a8465c6e8803effcc4b701559e1467413a9e93b797abc86a6380"
P1R43_REPORT_SHA = "ed60bd3bc57bf2ee14d35d558674734b5ab7c446125131b2a281f58b7d3bfbf7"
P1R35_REPORT_SHA = "b5a589c1c74c8c7e4304bbbd574f408a76626995cdfb97dddce4547b30d8b98c"
COMMON_METRIC_DEFINITION = (
    "0.5*SUM_LAYER[(ACTUAL_BF16_INCREMENT_FROBENIUS_SQ/"
    "NOMINAL_FACTOR_FROBENIUS_SQ)*NOMINAL_FACTOR_COVARIANCE_QUADRATIC]"
)

MODELS = ("llama3-8b-inst", "qwen2.5-7b-inst")
ARM_PATHS = (("NEUTRAL", "p1dw-neutral"), ("SOFT", "p1dw-soft"))


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_meta(path: Path, *, row_count: int | str | None = None) -> dict[str, Any]:
    raw = path.read_bytes()
    out: dict[str, Any] = {
        "path": path.name,
        "bytes": len(raw),
        "lines": raw.count(b"\n"),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if row_count is not None:
        out["row_count"] = row_count
    return out


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_once(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json_once(path: Path, value: Any) -> None:
    write_once(path, canonical_bytes(value) + b"\n")


def mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def describe(values: Iterable[float]) -> dict[str, Any]:
    vals = [float(value) for value in values]
    if not vals:
        return {"count": 0, "mean": "NOT_RECORDED", "median": "NOT_RECORDED", "p90": "NOT_RECORDED", "worst": "NOT_RECORDED", "min": "NOT_RECORDED"}
    return {
        "count": len(vals),
        "mean": mean(vals),
        "median": quantile(vals, 0.5),
        "p90": quantile(vals, 0.9),
        "worst": max(vals),
        "min": min(vals),
    }


def flatten_z_vectors(terminal: Mapping[str, Any], metric: str) -> list[dict[str, Any]]:
    nested = terminal["terminal_z_panel"]["z_inject"]["numeric_vectors"][metric]
    return [item for request_rows in nested for item in request_rows]


def w_metric(terminal: Mapping[str, Any], metric: str) -> Mapping[str, Any]:
    return terminal["terminal_w_panel"]["receipt"]["metrics"][metric]


def full6_vectors(terminal: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> tuple[list[float], list[float], list[float]]:
    if not steps or int(steps[-1]["outer_step"]) != 7:
        raise AssertionError("valid endpoint lacks outer step 7")
    w_values = [float(value) for value in steps[-1]["next_nll_by_request"]]
    z_values = [float(value) for value in terminal["terminal_full_six_target_new_nll_by_request"]]
    gap = [w - z for w, z in zip(w_values, z_values, strict=True)]
    return w_values, z_values, gap


def load_step(path: Path, *, model: str, case_index: int, arm: str) -> dict[str, Any]:
    raw = read_json(path)
    weighting = raw["deficit_weighting"]
    route = raw["route"]
    common = raw["actual_bf16_common_metrics"]
    response = raw["response"]
    objective = response["weighted_objective"]
    layer_rows = common["layers"]
    return {
        "schema": "ode-edit-s05-p2r7-production-step-analysis/v1",
        "job_id": JOB_ID,
        "source_head": SOURCE_HEAD,
        "model": model,
        "case_index": case_index,
        "weighting_mode": "P1DW",
        "arm": arm,
        "outer_step": int(raw["outer_step"]),
        "target_microstep_count_before_write": int(raw["target_microstep_count_before_write"]),
        "ell_w_by_request": weighting["ell_w_by_request"],
        "ell_z_by_request": weighting["ell_z_by_request"],
        "deficit_by_request": weighting["deficit_by_request"],
        "omega_by_request": weighting["omega_by_request"],
        "deficit_sum": float(weighting["deficit_sum"]),
        "deficit_median": float(weighting["deficit_median"]),
        "deficit_p90": float(weighting["deficit_p90"]),
        "deficit_maximum": float(weighting["deficit_maximum"]),
        "effective_request_count": float(weighting["effective_request_count"]),
        "omega_sum": float(weighting["omega_sum"]),
        "omega_maximum": float(weighting["maximum_omega"]),
        "rho_omega": float(weighting["rho_omega"]),
        "rho_sum_d2_over_sum_d": float(weighting["rho_sum_d2_over_sum_d"]),
        "no_semantic_deficit": bool(weighting["no_semantic_deficit"]),
        "signed_slopes": route["signed_slopes"],
        "active_mask": route["active_mask"],
        "active_layer_count": sum(bool(value) for value in route["active_mask"]),
        "q": float(route["q"]),
        "pi": route["pi"],
        "neutral_pi": route["neutral_pi"],
        "velocity": route["velocity"],
        "applied_hv": [0.125 * float(value) for value in route["velocity"]],
        "alpha_req": float(route["alpha_req"]),
        "alpha_apply": float(route["alpha_apply"]),
        "alpha_apply_over_req": float(route["alpha_apply_over_req"]),
        "strength_residual": float(route["alpha_apply"] - route["alpha_req"]),
        "route_status": route["status"],
        "fallback_to_neutral": bool(route["fallback_to_neutral"]),
        "fallback_reason": route["fallback_reason"],
        "predicted_weighted_progress": float(raw["predicted_weighted_progress"]),
        "actual_weighted_progress": float(raw["actual_weighted_progress"]),
        "weighted_realization": float(raw["weighted_realization"]),
        "nonlinear_overshoot": bool(raw["predicted_weighted_progress"] > 0.0 and raw["actual_weighted_progress"] < 0.0),
        "negative_actual_count": int(raw["negative_actual_count"]),
        "current_nll_by_request": raw["current_nll_by_request"],
        "next_nll_by_request": raw["next_nll_by_request"],
        "actual_progress_by_request": raw["actual_progress_by_request"],
        "realization_by_request": raw["realization_by_request"],
        "w_z_gap_by_request": raw["w_z_gap_by_request"],
        "full_current_residual_norm_by_request": raw["full_current_residual_norm_by_request"],
        "cumulative_structural_p": float(raw["cumulative_structural_p"]),
        "selected_structural_p": float(route["selected_p"]),
        "selected_capacity": float(route["selected_capacity"]),
        "selected_energy": float(route["selected_energy"]),
        "actual_bf16_common_capacity": float(common["actual_bf16_common_capacity"]),
        "actual_bf16_common_covariance_structural_p": float(common["actual_bf16_covariance_structural_p"]),
        "actual_bf16_common_energy": float(common["actual_bf16_update_energy"]),
        "actual_bf16_common_metric_definition": common["common_capacity_definition"],
        "actual_bf16_layer_metrics": layer_rows,
        "entropy": float(route["simplex_entropy"]),
        "top1": float(route["simplex_top1_share"]),
        "routing_variable_count": int(raw["routing_variable_count"]),
        "request_layer_response_matrix_count": int(raw["request_layer_response_matrix_count"]),
        "response_batched_vjp_count": int(response["batched_vjp_count"]),
        "one_joint_materialization_count": int(raw["one_joint_materialization_count"]),
        "current_state_refresh_count": int(raw["current_state_refresh_count"]),
        "physical_h_application_count": int(raw["physical_h_application_count"]),
        "residual_inverse_h_count": int(raw["residual_inverse_h_count"]),
        "second_h_application_count": int(raw["second_h_application_count"]),
        "retry_count": int(raw["retry_count"]),
        "backtracking_count": int(raw["backtracking_count"]),
        "weighted_objective_forward_count": int(objective["model_forward_count"]),
        "weighted_objective_backward_count": int(objective["backward_count"]),
        "weighted_objective_tokens": int(objective["processed_token_count"]),
        "target_state_sha256": raw["target_state_sha256"],
        "current_terminal_sha256": raw["current_terminal_sha256"],
        "full_current_residual_sha256": raw["full_current_residual_sha256"],
        "factor_sha256": raw["factor_sha256"],
        "materialization_sha256": raw["materialization"]["identity_sha256"],
        "step_identity_sha256": raw["identity_sha256"],
        "step_file_sha256": file_sha(path),
    }


def endpoint_from_terminal(
    path: Path,
    *,
    model: str,
    case_index: int,
    arm: str,
    steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    terminal = read_json(path)
    eff = w_metric(terminal, "efficacy")
    gen = w_metric(terminal, "generalization")
    loc = w_metric(terminal, "locality-preservation")
    z_eff_vectors = flatten_z_vectors(terminal, "efficacy")
    z_gen_vectors = flatten_z_vectors(terminal, "generalization")
    w_full6, z_full6, full6_gap = full6_vectors(terminal, steps)
    predicted = sum(float(step["predicted_weighted_progress"]) for step in steps)
    actual = sum(float(step["actual_weighted_progress"]) for step in steps)
    forbidden = terminal["forbidden_influence"]
    return {
        "schema": "ode-edit-s05-p2r7-production-attempt-analysis/v1",
        "job_id": JOB_ID,
        "source_head": SOURCE_HEAD,
        "model": model,
        "case_index": case_index,
        "weighting_mode": "P1DW",
        "arm": arm,
        "status": "COMPLETE",
        "typed_classification": None,
        "attempt": 1,
        "endpoint": 1,
        "request_count": int(terminal["request_count"]),
        "target_microsteps": int(terminal["target_microstep_count"]),
        "writer_transitions": int(terminal["writer_transition_count"]),
        "materializations": int(terminal["writer_materialization_count"]),
        "W0_restored": bool(terminal["W0_restored"]),
        "W0_pointer_restored": bool(terminal["W0_restore"]["pointer_restored_exact"]),
        "W0_bytes_restored": bool(terminal["W0_restore"]["byte_restored_exact"]),
        "action_freeze_sha256": terminal["action_freeze_sha256"],
        "w_eff_correct": int(eff["correct_count"]),
        "w_eff_den": int(eff["prompt_count"]),
        "w_gen_correct": int(gen["correct_count"]),
        "w_gen_den": int(gen["prompt_count"]),
        "w_loc_correct": int(loc["correct_count"]),
        "w_loc_den": int(loc["prompt_count"]),
        "w_eff_nll": float(eff["mean_target_new_nll"]),
        "w_eff_old_nll": float(eff["mean_target_old_nll"]),
        "w_eff_margin": float(eff["mean_margin"]),
        "w_gen_nll": float(gen["mean_target_new_nll"]),
        "w_gen_old_nll": float(gen["mean_target_old_nll"]),
        "w_gen_margin": float(gen["mean_margin"]),
        "w_loc_nll": float(loc["mean_target_new_nll"]),
        "w_loc_old_nll": float(loc["mean_target_old_nll"]),
        "w_loc_margin": float(loc["mean_margin"]),
        "z_eff_correct": int(terminal["terminal_z_panel"]["eff_z_inject"]["numerator"]),
        "z_eff_den": int(terminal["terminal_z_panel"]["eff_z_inject"]["denominator"]),
        "z_gen_correct": int(terminal["terminal_z_panel"]["gen_z_inject"]["numerator"]),
        "z_gen_den": int(terminal["terminal_z_panel"]["gen_z_inject"]["denominator"]),
        "z_eff_nll": mean([float(row["nll_new"]) for row in z_eff_vectors]),
        "z_eff_old_nll": mean([float(row["nll_old"]) for row in z_eff_vectors]),
        "z_eff_margin": mean([float(row["margin"]) for row in z_eff_vectors]),
        "z_gen_nll": mean([float(row["nll_new"]) for row in z_gen_vectors]),
        "z_gen_old_nll": mean([float(row["nll_old"]) for row in z_gen_vectors]),
        "z_gen_margin": mean([float(row["margin"]) for row in z_gen_vectors]),
        "w_full6_nll": mean(w_full6),
        "w_full6_nll_by_request": w_full6,
        "z_full6_nll": mean(z_full6),
        "z_full6_nll_by_request": z_full6,
        "terminal_w_minus_z_full6_gap": mean(full6_gap),
        "terminal_w_minus_z_full6_gap_by_request": full6_gap,
        "terminal_z_full6_distribution": describe(z_full6),
        "terminal_w_full6_distribution": describe(w_full6),
        "terminal_w_minus_z_gap_distribution": describe(full6_gap),
        "predicted_progress_sum": predicted,
        "actual_progress_sum": actual,
        "aggregate_realization": actual / max(predicted, 1.0e-12),
        "negative_actual_count": sum(int(step["negative_actual_count"]) for step in steps),
        "nonlinear_overshoot_count": sum(bool(step["nonlinear_overshoot"]) for step in steps),
        "structural_p_terminal": float(steps[-1]["cumulative_structural_p"]),
        "common_capacity_sum": sum(float(step["actual_bf16_common_capacity"]) for step in steps),
        "common_bf16_energy_sum": sum(float(step["actual_bf16_common_energy"]) for step in steps),
        "mean_entropy": mean([float(step["entropy"]) for step in steps]),
        "mean_top1": mean([float(step["top1"]) for step in steps]),
        "mean_active_layer_count": mean([float(step["active_layer_count"]) for step in steps]),
        "mean_q": mean([float(step["q"]) for step in steps]),
        "fallback_count": sum(bool(step["fallback_to_neutral"]) for step in steps),
        "rho_mean": mean([float(step["rho_omega"]) for step in steps]),
        "deficit_sum_mean": mean([float(step["deficit_sum"]) for step in steps]),
        "deficit_p90_mean": mean([float(step["deficit_p90"]) for step in steps]),
        "deficit_worst": max(float(step["deficit_maximum"]) for step in steps),
        "effective_request_count_mean": mean([float(step["effective_request_count"]) for step in steps]),
        "effective_request_count_min": min(float(step["effective_request_count"]) for step in steps),
        "omega_max_mean": mean([float(step["omega_maximum"]) for step in steps]),
        "omega_maximum": max(float(step["omega_maximum"]) for step in steps),
        "omega_sum_max_abs_residual": max(abs(float(step["omega_sum"]) - 1.0) for step in steps),
        "strength_residual_max": max(abs(float(step["strength_residual"])) for step in steps),
        "total_wall_seconds": float(terminal["total_wall_seconds"]),
        "target_wall_seconds": float(terminal["target_wall_seconds"]),
        "writer_wall_seconds": float(terminal["writer_wall_seconds"]),
        "terminal_eval_wall_seconds": float(terminal["terminal_w_evaluator_wall_seconds"] + terminal["terminal_z_evaluator_wall_seconds"]),
        "compute": terminal["compute"],
        "forbidden_influence": forbidden,
        "terminal_identity_sha256": terminal["identity_sha256"],
        "terminal_file_sha256": file_sha(path),
        "action_freeze_pass": bool(
            terminal["terminal_w_panel"]["receipt"]["controller_or_routing_influence_count"] == 0
            and terminal["terminal_z_panel"]["heldout_controller_access_count"] == 0
        ),
    }


def failure_attempt(
    path: Path,
    *,
    model: str,
    case_index: int,
    arm: str,
    steps: Sequence[Mapping[str, Any]],
    target_files: Sequence[Path],
) -> dict[str, Any]:
    failure = read_json(path)
    return {
        "schema": "ode-edit-s05-p2r7-production-attempt-analysis/v1",
        "job_id": JOB_ID,
        "source_head": SOURCE_HEAD,
        "model": model,
        "case_index": case_index,
        "weighting_mode": "P1DW",
        "arm": arm,
        "status": "TYPED_INCOMPLETE",
        "typed_classification": failure["classification"],
        "attempt": 1,
        "endpoint": 0,
        "request_count": 10,
        "target_microsteps": len(target_files),
        "writer_transitions": len(steps),
        "materializations": len(steps),
        "last_valid_outer_step": max(int(step["outer_step"]) for step in steps) if steps else None,
        "failure_outer_step": (max(int(step["outer_step"]) for step in steps) + 1) if steps else 0,
        "W0_restored": True,
        "W0_pointer_restored": bool(failure["W0_restore"]["pointer_restored_exact"]),
        "W0_bytes_restored": bool(failure["W0_restore"]["byte_restored_exact"]),
        "action_freeze_sha256": "NOT_RECORDED_TYPED_INCOMPLETE",
        "terminal_metrics": "NOT_RECORDED_TYPED_INCOMPLETE_NO_IMPUTATION",
        "prefix_predicted_progress_sum": sum(float(step["predicted_weighted_progress"]) for step in steps),
        "prefix_actual_progress_sum": sum(float(step["actual_weighted_progress"]) for step in steps),
        "prefix_negative_actual_count": sum(int(step["negative_actual_count"]) for step in steps),
        "prefix_structural_p": float(steps[-1]["cumulative_structural_p"]) if steps else 0.0,
        "prefix_common_capacity_sum": sum(float(step["actual_bf16_common_capacity"]) for step in steps),
        "prefix_common_bf16_energy_sum": sum(float(step["actual_bf16_common_energy"]) for step in steps),
        "prefix_fallback_count": sum(bool(step["fallback_to_neutral"]) for step in steps),
        "retry_count": int(failure["retry_count"]),
        "next_arm_or_case_continues": bool(failure["next_arm_or_case_continues"]),
        "exception_class": failure["exception_class"],
        "exception_message_sha256": failure["exception_message_sha256"],
        "failure_identity_sha256": failure["identity_sha256"],
        "failure_file_sha256": file_sha(path),
    }


def request_step_rows(step: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for request_index in range(10):
        rows.append(
            {
                "schema": "ode-edit-s05-p2r7-production-request-step-analysis/v1",
                "job_id": JOB_ID,
                "source_head": SOURCE_HEAD,
                "model": step["model"],
                "case_index": step["case_index"],
                "weighting_mode": step["weighting_mode"],
                "arm": step["arm"],
                "outer_step": step["outer_step"],
                "request_index": request_index,
                "ell_w": float(step["ell_w_by_request"][request_index]),
                "ell_z": float(step["ell_z_by_request"][request_index]),
                "deficit": float(step["deficit_by_request"][request_index]),
                "omega": float(step["omega_by_request"][request_index]),
                "current_w_nll": float(step["current_nll_by_request"][request_index]),
                "next_w_nll": float(step["next_nll_by_request"][request_index]),
                "actual_w_progress": float(step["actual_progress_by_request"][request_index]),
                "negative_actual": bool(step["actual_progress_by_request"][request_index] < 0.0),
                "observation_only_chi": float(step["realization_by_request"][request_index]),
                "w_z_gap": float(step["w_z_gap_by_request"][request_index]),
                "full_current_residual_norm": float(step["full_current_residual_norm_by_request"][request_index]),
                "step_identity_sha256": step["step_identity_sha256"],
            }
        )
    return rows


def cell_aggregates(attempts: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for arm, _ in ARM_PATHS:
            attempted = [row for row in attempts if row["model"] == model and row["arm"] == arm]
            complete = [row for row in attempted if row["status"] == "COMPLETE"]
            cell_steps = [row for row in steps if row["model"] == model and row["arm"] == arm]
            full_z = [value for row in complete for value in row["z_full6_nll_by_request"]]
            full_w = [value for row in complete for value in row["w_full6_nll_by_request"]]
            full_gap = [value for row in complete for value in row["terminal_w_minus_z_full6_gap_by_request"]]
            request_chi = [value for row in cell_steps for value in row["realization_by_request"]]
            all_deficits = [value for row in cell_steps for value in row["deficit_by_request"]]
            route_counts = Counter(str(row["route_status"]) for row in cell_steps)
            rows.append(
                {
                    "schema": "ode-edit-s05-p2r7-production-cell-aggregate/v1",
                    "model": model,
                    "arm": arm,
                    "attempts": len(attempted),
                    "endpoints": len(complete),
                    "typed_incomplete": len(attempted) - len(complete),
                    "requests_attempted": 10 * len(attempted),
                    "terminal_requests": 10 * len(complete),
                    "writer_steps": len(cell_steps),
                    "target_microsteps": sum(int(row["target_microsteps"]) for row in attempted),
                    "w_eff_correct": sum(int(row["w_eff_correct"]) for row in complete),
                    "w_eff_den": sum(int(row["w_eff_den"]) for row in complete),
                    "w_gen_correct": sum(int(row["w_gen_correct"]) for row in complete),
                    "w_gen_den": sum(int(row["w_gen_den"]) for row in complete),
                    "w_loc_correct": sum(int(row["w_loc_correct"]) for row in complete),
                    "w_loc_den": sum(int(row["w_loc_den"]) for row in complete),
                    "z_eff_correct": sum(int(row["z_eff_correct"]) for row in complete),
                    "z_eff_den": sum(int(row["z_eff_den"]) for row in complete),
                    "z_gen_correct": sum(int(row["z_gen_correct"]) for row in complete),
                    "z_gen_den": sum(int(row["z_gen_den"]) for row in complete),
                    "w_eff_nll": mean([float(row["w_eff_nll"]) for row in complete]),
                    "w_eff_margin": mean([float(row["w_eff_margin"]) for row in complete]),
                    "w_gen_nll": mean([float(row["w_gen_nll"]) for row in complete]),
                    "w_gen_margin": mean([float(row["w_gen_margin"]) for row in complete]),
                    "w_loc_nll": mean([float(row["w_loc_nll"]) for row in complete]),
                    "w_loc_margin": mean([float(row["w_loc_margin"]) for row in complete]),
                    "z_eff_nll": mean([float(row["z_eff_nll"]) for row in complete]),
                    "z_eff_margin": mean([float(row["z_eff_margin"]) for row in complete]),
                    "z_gen_nll": mean([float(row["z_gen_nll"]) for row in complete]),
                    "z_gen_margin": mean([float(row["z_gen_margin"]) for row in complete]),
                    "z_full6_nll": mean(full_z),
                    "w_full6_nll": mean(full_w),
                    "w_minus_z_full6_gap": mean(full_gap),
                    "terminal_z_full6_distribution": describe(full_z),
                    "terminal_w_full6_distribution": describe(full_w),
                    "terminal_gap_distribution": describe(full_gap),
                    "deficit_distribution_all_steps_requests": describe(all_deficits),
                    "observation_only_chi_distribution": describe(request_chi),
                    "predicted_progress_sum": sum(float(row["predicted_progress_sum"]) for row in complete),
                    "actual_progress_sum": sum(float(row["actual_progress_sum"]) for row in complete),
                    "actual_over_predicted": (
                        sum(float(row["actual_progress_sum"]) for row in complete)
                        / max(sum(float(row["predicted_progress_sum"]) for row in complete), 1.0e-12)
                    ),
                    "negative_request_transitions": sum(int(row["negative_actual_count"]) for row in cell_steps),
                    "nonlinear_overshoot_steps": sum(bool(row["nonlinear_overshoot"]) for row in cell_steps),
                    "terminal_structural_p_mean": mean([float(row["structural_p_terminal"]) for row in complete]),
                    "common_capacity_sum_mean": mean([float(row["common_capacity_sum"]) for row in complete]),
                    "common_bf16_energy_sum_mean": mean([float(row["common_bf16_energy_sum"]) for row in complete]),
                    "entropy_mean": mean([float(row["entropy"]) for row in cell_steps]),
                    "top1_mean": mean([float(row["top1"]) for row in cell_steps]),
                    "active_layer_count_mean": mean([float(row["active_layer_count"]) for row in cell_steps]),
                    "q_mean": mean([float(row["q"]) for row in cell_steps]),
                    "rho_mean": mean([float(row["rho_omega"]) for row in cell_steps]),
                    "deficit_sum_mean": mean([float(row["deficit_sum"]) for row in cell_steps]),
                    "deficit_p90_mean": mean([float(row["deficit_p90"]) for row in cell_steps]),
                    "deficit_maximum": max(float(row["deficit_maximum"]) for row in cell_steps),
                    "effective_request_count_mean": mean([float(row["effective_request_count"]) for row in cell_steps]),
                    "effective_request_count_min": min(float(row["effective_request_count"]) for row in cell_steps),
                    "omega_maximum_mean": mean([float(row["omega_maximum"]) for row in cell_steps]),
                    "omega_maximum": max(float(row["omega_maximum"]) for row in cell_steps),
                    "soft_fallback_count": sum(bool(row["fallback_to_neutral"]) for row in cell_steps),
                    "soft_nonfallback_no_dof_count": sum(
                        row["arm"] == "SOFT"
                        and not row["fallback_to_neutral"]
                        and max(
                            abs(float(selected) - float(neutral))
                            for selected, neutral in zip(row["pi"], row["neutral_pi"], strict=True)
                        ) <= 1.0e-8
                        for row in cell_steps
                    ),
                    "route_status_counts": dict(sorted(route_counts.items())),
                    "weighted_objective_forward_count": sum(int(row["weighted_objective_forward_count"]) for row in cell_steps),
                    "weighted_objective_backward_count": sum(int(row["weighted_objective_backward_count"]) for row in cell_steps),
                    "weighted_objective_tokens": sum(int(row["weighted_objective_tokens"]) for row in cell_steps),
                    "total_wall_seconds_mean": mean([float(row["total_wall_seconds"]) for row in complete]),
                    "target_wall_seconds_mean": mean([float(row["target_wall_seconds"]) for row in complete]),
                    "writer_wall_seconds_mean": mean([float(row["writer_wall_seconds"]) for row in complete]),
                    "terminal_eval_wall_seconds_mean": mean([float(row["terminal_eval_wall_seconds"]) for row in complete]),
                    "compute_sums_valid_endpoints": {
                        key: sum(int(row["compute"][key]) for row in complete)
                        for key in complete[0]["compute"]
                    },
                }
            )
    return rows


def delta(current: Mapping[str, Any], reference: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    return {f"delta_{field}": float(current[field]) - float(reference[field]) for field in fields}


def load_allowed_references(repo: Path) -> dict[str, Any]:
    pilot_root = repo / "local/odebf/reports/p2r7-p2target-p1dw-shared-writer-v1/pilot-tech-r2"
    p2r2_root = Path("/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p2r2-semantic-conserving-residual-transport-writer-v1/local/odebf/reports/p2r2-semantic-conserving-residual-transport-writer-v1")
    references = {
        "pilot_endpoints_path": pilot_root / "per-endpoint.json",
        "pilot_steps_path": pilot_root / "per-step.json",
        "pilot_report_path": pilot_root / "p2r7-pilot-tech-r2-analysis-ko.md",
        "p2r2_case_path": p2r2_root / "p2r2-per-case.jsonl",
        "p2r2_step_request_path": p2r2_root / "p2r2-per-step-request.jsonl",
        "p2r2_comparison_path": p2r2_root / "p2r2-comparisons.jsonl",
        "p2r2_report_path": p2r2_root / "p2r2-semantic-conserving-residual-transport-terminal-analysis-ko.md",
    }
    expected = {
        "pilot_endpoints_path": PILOT_ENDPOINT_SHA,
        "pilot_steps_path": PILOT_STEP_SHA,
        "pilot_report_path": PILOT_REPORT_SHA,
        "p2r2_case_path": P2R2_CASE_SHA,
        "p2r2_step_request_path": P2R2_STEP_REQUEST_SHA,
        "p2r2_comparison_path": P2R2_COMPARISON_SHA,
        "p2r2_report_path": P2R2_REPORT_SHA,
    }
    for key, path in references.items():
        observed = file_sha(path)
        if observed != expected[key]:
            raise AssertionError(f"reference identity mismatch {key}: {observed}")
    references["pilot_endpoints"] = read_json(references["pilot_endpoints_path"])
    references["pilot_steps"] = read_json(references["pilot_steps_path"])
    references["p2r2_cases"] = read_jsonl(references["p2r2_case_path"])
    references["p2r2_step_requests"] = read_jsonl(references["p2r2_step_request_path"])
    references["p2r2_comparisons"] = read_jsonl(references["p2r2_comparison_path"])
    return references


def build_comparisons(
    attempts: Sequence[Mapping[str, Any]],
    references: Mapping[str, Any],
) -> list[dict[str, Any]]:
    complete = [row for row in attempts if row["status"] == "COMPLETE"]
    comparisons: list[dict[str, Any]] = []

    # Scientific pilot: same model/case/source, P1DW minus P1AGG.
    pilot_map = {
        (row["model"], row["arm"], row["weighting_mode"]): row
        for row in references["pilot_endpoints"]
    }
    pilot_step7 = {
        (row["model"], row["arm"], row["weighting_mode"]): row
        for row in references["pilot_steps"]
        if int(row["outer_step"]) == 7
    }
    pilot_fields = (
        "w_eff_correct", "w_gen_correct", "w_loc_correct", "w_eff_nll", "w_gen_nll",
        "z_eff_nll", "z_gen_nll", "w_full6_nll", "z_full6_nll", "terminal_w_minus_z_full6_gap",
        "predicted_progress_sum", "actual_progress_sum", "aggregate_realization",
        "structural_p_terminal", "common_capacity_sum", "common_bf16_energy_sum",
        "mean_entropy", "mean_top1",
    )
    for model in MODELS:
        for arm, _ in ARM_PATHS:
            cur = dict(pilot_map[(model, arm, "P1DW")])
            ref = dict(pilot_map[(model, arm, "P1AGG")])
            for normalized, mode in ((cur, "P1DW"), (ref, "P1AGG")):
                legacy_z_full6 = float(normalized["w_full6_nll"])
                terminal_w_full6 = mean(
                    [
                        float(value)
                        for value in pilot_step7[(model, arm, mode)]["next_nll_by_request"]
                    ]
                )
                normalized["z_full6_nll"] = legacy_z_full6
                normalized["w_full6_nll"] = terminal_w_full6
                normalized["terminal_w_minus_z_full6_gap"] = terminal_w_full6 - legacy_z_full6
            comparisons.append(
                {
                    "schema": "ode-edit-s05-p2r7-production-comparison/v1",
                    "comparison": "P1AGG_TO_P1DW_PILOT",
                    "identity_status": "MATCHED_CASE01_SAME_FINAL_SOURCE",
                    "model": model,
                    "arm": arm,
                    "case_index": 1,
                    "current": {field: cur[field] for field in pilot_fields},
                    "reference": {field: ref[field] for field in pilot_fields},
                    **delta(cur, ref, pilot_fields),
                    "common_capacity_coordinate": COMMON_METRIC_DEFINITION,
                }
            )

    # Production paired Soft minus Neutral.  Llama case04 remains denominator-only.
    complete_map = {(row["model"], row["arm"], row["case_index"]): row for row in complete}
    ns_fields = (
        "w_eff_correct", "w_gen_correct", "w_loc_correct", "w_eff_nll", "w_gen_nll",
        "z_eff_correct", "z_gen_correct", "z_eff_nll", "z_gen_nll", "w_full6_nll",
        "z_full6_nll", "terminal_w_minus_z_full6_gap", "predicted_progress_sum",
        "actual_progress_sum", "aggregate_realization", "structural_p_terminal",
        "common_capacity_sum", "common_bf16_energy_sum", "mean_entropy", "mean_top1",
    )
    for model in MODELS:
        for case_index in range(1, 11):
            neutral = complete_map.get((model, "NEUTRAL", case_index))
            soft = complete_map.get((model, "SOFT", case_index))
            if neutral is None or soft is None:
                continue
            comparisons.append(
                {
                    "schema": "ode-edit-s05-p2r7-production-comparison/v1",
                    "comparison": "DW_NEUTRAL_TO_DW_SOFT_PRODUCTION",
                    "identity_status": "MATCHED_MODEL_CASE_STREAM_EVALUATOR_SOURCE",
                    "model": model,
                    "arm": "SOFT_MINUS_NEUTRAL",
                    "case_index": case_index,
                    "current": {field: soft[field] for field in ns_fields},
                    "reference": {field: neutral[field] for field in ns_fields},
                    **delta(soft, neutral, ns_fields),
                    "common_capacity_coordinate": COMMON_METRIC_DEFINITION,
                }
            )

    # P2R2 per-case map and terminal W-only full-six from its outer-7 post-write rows.
    p2r2_map = {(row["alias"], row["arm"], row["case"]): row for row in references["p2r2_cases"]}
    p2r2_w_full6: defaultdict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in references["p2r2_step_requests"]:
        if int(row["outer_step"]) == 7:
            p2r2_w_full6[(row["alias"], row["arm"], row["case"])].append(float(row["next_nll"]))
    p2_fields = (
        "w_eff_correct", "w_gen_correct", "w_loc_correct", "w_eff_nll", "w_gen_nll",
        "z_eff_correct", "z_gen_correct", "z_eff_nll", "z_gen_nll", "w_full6_nll",
        "z_full6_nll", "terminal_w_minus_z_full6_gap", "predicted_progress_sum",
        "actual_progress_sum", "aggregate_realization",
    )
    for current in complete:
        p2_arm = "NEUTRAL" if current["arm"] == "NEUTRAL" else "SOFTP"
        reference_raw = p2r2_map[(current["model"], p2_arm, current["case_index"])]
        w_values = p2r2_w_full6[(current["model"], p2_arm, current["case_index"])]
        if len(w_values) != 10:
            raise AssertionError("P2R2 terminal W-only full-six denominator differs")
        p2_w_full6 = mean(w_values)
        reference = {
            "w_eff_correct": reference_raw["w_eff"]["correct"],
            "w_gen_correct": reference_raw["w_gen"]["correct"],
            "w_loc_correct": reference_raw["w_loc"]["correct"],
            "w_eff_nll": reference_raw["w_eff"]["nll_new_mean"],
            "w_gen_nll": reference_raw["w_gen"]["nll_new_mean"],
            "z_eff_correct": reference_raw["z_eff"]["correct"],
            "z_gen_correct": reference_raw["z_gen"]["correct"],
            "z_eff_nll": reference_raw["z_eff"]["nll_new"]["mean"],
            "z_gen_nll": reference_raw["z_gen"]["nll_new"]["mean"],
            "w_full6_nll": p2_w_full6,
            "z_full6_nll": reference_raw["terminal_full_six_target_new_nll"],
            "terminal_w_minus_z_full6_gap": p2_w_full6 - reference_raw["terminal_full_six_target_new_nll"],
            "predicted_progress_sum": reference_raw["predicted_progress_sum"],
            "actual_progress_sum": reference_raw["actual_progress_sum"],
            "aggregate_realization": reference_raw["actual_over_predicted"],
        }
        comparisons.append(
            {
                "schema": "ode-edit-s05-p2r7-production-comparison/v1",
                "comparison": "P2_RESIDUAL_TRANSPORT_TO_P1DW_SHARED_WRITER",
                "identity_status": "MATCHED_MODEL_CASE_STREAM_ORDER_EVALUATOR",
                "model": current["model"],
                "arm": current["arm"],
                "case_index": current["case_index"],
                "current": {field: current[field] for field in p2_fields},
                "reference": reference,
                **delta(current, reference, p2_fields),
                "current_common_capacity": current["common_capacity_sum"],
                "current_common_bf16_energy": current["common_bf16_energy_sum"],
                "reference_common_capacity": "NOT_COMPARABLE_LEGACY_COORDINATE",
                "reference_common_bf16_energy": "NOT_RECORDED",
            }
        )

    official_map = {}
    for row in references["p2r2_comparisons"]:
        if row["comparator"] == "OFFICIAL_ALPHAEDIT" and row["arm"] == "NEUTRAL":
            official_map[(row["alias"], row["case"])] = row
    for current in complete:
        reference_raw = official_map[(current["model"], current["case_index"])]
        reference = {
            "w_eff_correct": reference_raw["reference_eff_correct"],
            "w_gen_correct": reference_raw["reference_gen_correct"],
            "w_loc_correct": reference_raw["reference_loc_correct"],
            "w_eff_nll": reference_raw["reference_eff_nll_mean"],
            "w_eff_margin": reference_raw["reference_eff_margin_mean"],
        }
        official_fields = tuple(reference)
        comparisons.append(
            {
                "schema": "ode-edit-s05-p2r7-production-comparison/v1",
                "comparison": "OFFICIAL_ALPHAEDIT_TO_P1DW",
                "identity_status": "MATCHED_MODEL_CASE_STREAM_ORDER_EVALUATOR",
                "model": current["model"],
                "arm": current["arm"],
                "case_index": current["case_index"],
                "current": {field: current[field] for field in official_fields},
                "reference": reference,
                **delta(current, reference, official_fields),
                "official_gen_continuous": "NOT_RECORDED",
                "official_loc_continuous": "NOT_RECORDED",
                "official_latent_z": "NOT_RECORDED",
                "official_common_capacity": "NOT_RECORDED",
                "official_common_bf16_energy": "NOT_RECORDED",
                "official_structural_p": "NOT_RECORDED",
            }
        )
    return comparisons


def comparison_aggregates(comparisons: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in comparisons:
        groups[(row["comparison"], row["model"], row["arm"])].append(row)
    out = []
    for (comparison, model, arm), rows in sorted(groups.items()):
        delta_keys = sorted(key for key in rows[0] if key.startswith("delta_") and isinstance(rows[0][key], (int, float)))
        out.append(
            {
                "schema": "ode-edit-s05-p2r7-production-comparison-aggregate/v1",
                "comparison": comparison,
                "model": model,
                "arm": arm,
                "matched_cases": len(rows),
                "delta_sums": {key: sum(float(row[key]) for row in rows) for key in delta_keys},
                "delta_means": {key: mean([float(row[key]) for row in rows]) for key in delta_keys},
                "identity_statuses": sorted({str(row["identity_status"]) for row in rows}),
            }
        )
    return out


def aggregate_lookup(rows: Sequence[Mapping[str, Any]], model: str, arm: str) -> Mapping[str, Any]:
    return next(row for row in rows if row["model"] == model and row["arm"] == arm)


def comp_lookup(rows: Sequence[Mapping[str, Any]], comparison: str, model: str, arm: str) -> Mapping[str, Any]:
    return next(row for row in rows if row["comparison"] == comparison and row["model"] == model and row["arm"] == arm)


def f(value: Any, digits: int = 6) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "NOT_RECORDED"
    return f"{float(value):.{digits}g}"


def signed(value: Any, digits: int = 6) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):+.{digits}g}"


def report_text(
    aggregates: Sequence[Mapping[str, Any]],
    comparison_aggs: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
) -> str:
    lines: list[str] = []
    lines.extend(
        [
            "# P2R7 P2TARGET-P1DW Shared Writer B10×10 최종 분석",
            "",
            f"- instruction: `{INSTRUCTION_ID}`",
            f"- scheduler: array job `{JOB_ID}`, tasks `0` Llama / `1` Qwen, 두 task 모두 `COMPLETED`, exit `0:0`",
            f"- source HEAD/tree: `{SOURCE_HEAD}` / `{SOURCE_TREE}`",
            f"- exact P2R2 parent: `{P2R2_PARENT}`; execution parent: `{SOURCE_PARENT}`",
            f"- contract SHA256: `{CONTRACT_SHA}`",
            f"- stream/order: `{STREAM_ROOT}` / `{ORDER_SHA}`",
            f"- evaluator/aggregator: `{EVALUATOR_SHA}` / `{AGGREGATOR_SHA}`",
            "- 전체 분모: 40 attempts / 39 complete endpoints / 1 typed incomplete / 400 attempted requests / 390 terminal requests / 316 accepted writer steps / 951 target microstep receipts.",
            "- `scientific_promotion=false`.",
            "",
            "## 1. 기술 무결성 게이트",
            "",
            "| 항목 | 관측값 | 상태 |",
            "|---|---:|---|",
            "| scheduler task | 2/2 COMPLETED, exit 0:0 | PASS |",
            "| attempt / endpoint / typed incomplete | 40 / 39 / 1 | PASS_WITH_TYPED_SCIENTIFIC_INCOMPLETE |",
            "| valid endpoint K8 / target24 / materialization8 | 39/39 | PASS |",
            "| accepted prefix 포함 writer step / target microstep | 316 / 951 | PASS |",
            "| valid W0 pointer / bytes / action-freeze | 39/39 / 39/39 / 39/39 | PASS |",
            "| incomplete W0 pointer / bytes | 1/1 / 1/1 | PASS |",
            "| routing 변수 / request×layer response matrix | 전 step 5 / 0 | PASS |",
            "| residual `1/h` / physical `h` / second `h` | 전 step 1 / 1 / 0 | PASS |",
            "| omega 합 최대 절대 오차 | `2.220446049250313e-16` | PASS |",
            "| weighted strength 최대 절대 잔차 | `5.186961971048731e-13` | PASS |",
            "| retry / backtracking / candidate materialization | 0 / 0 / 0 | PASS |",
            "| P2R6 QP/shadow, hard-P, functional veto, history-H | 모두 0 | PASS |",
            "| NO_SEMANTIC_DEFICIT | 0 step | 관측 없음 |",
            "| Soft→Neutral full-strength fallback | 3 step | 과학 관측 |",
            "| negative request transition / weighted nonlinear overshoot | 83 / 1 | 과학 관측 |",
            "",
            "Llama Soft case04는 outer step4에서 `SCIENTIFIC_SHARED_WRITER_NO_POSITIVE_DIRECTION`으로 종료됐다. 마지막 유효 prefix는 target microstep 15개와 writer/materialization 4개이며 terminal evaluator와 endpoint imputation은 없다. 실패 파일 SHA256은 `fbd24bef9511c5aa05c4abe673049976ed83ffa2ea99b859698ab129cd9f7c37`이고 W0 pointer/bytes 복원은 모두 PASS다. 이 1건은 기술 오류가 아니라 계약에 정의된 typed scientific outcome이다.",
            "",
            "첫 W0 outer의 target microstep 0–2는 TECH-R2 파일럿에서 matched P2R2와 모델별 3/3 의미 필드 byte identity PASS가 확인됐다. Production은 동일 final source를 사용했다.",
            "",
            "## 2. Model × arm terminal 집계",
            "",
            "| model | arm | endpoints/attempts | W E/G/L | z E/G | W Eff/Gen NLL | z Eff/Gen NLL | z/W full6 NLL | W−z full6 gap | actual/pred | P | common cap / energy |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        for arm, _ in ARM_PATHS:
            row = aggregate_lookup(aggregates, model, arm)
            lines.append(
                f"| {model} | {arm} | {row['endpoints']}/{row['attempts']} | "
                f"{row['w_eff_correct']}/{row['w_eff_den']}, {row['w_gen_correct']}/{row['w_gen_den']}, {row['w_loc_correct']}/{row['w_loc_den']} | "
                f"{row['z_eff_correct']}/{row['z_eff_den']}, {row['z_gen_correct']}/{row['z_gen_den']} | "
                f"{f(row['w_eff_nll'])} / {f(row['w_gen_nll'])} | {f(row['z_eff_nll'])} / {f(row['z_gen_nll'])} | "
                f"{f(row['z_full6_nll'])} / {f(row['w_full6_nll'])} | {f(row['w_minus_z_full6_gap'])} | "
                f"{f(row['actual_over_predicted'])} | {f(row['terminal_structural_p_mean'])} | "
                f"{f(row['common_capacity_sum_mean'])} / {f(row['common_bf16_energy_sum_mean'])} |"
            )
    lines.extend(
        [
            "",
            "`z full6`는 terminal W에서 z를 주입해 측정한 full-six target-new NLL이고, `W full6`는 마지막 writer transition 후 W-only full-six NLL이다. 두 값의 차이가 표의 W−z gap이다.",
            "",
            "| model | arm | W Eff/Gen/Loc margin | z Eff/Gen margin | W Loc NLL |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        for arm, _ in ARM_PATHS:
            row = aggregate_lookup(aggregates, model, arm)
            lines.append(
                f"| {model} | {arm} | {f(row['w_eff_margin'])} / {f(row['w_gen_margin'])} / {f(row['w_loc_margin'])} | "
                f"{f(row['z_eff_margin'])} / {f(row['z_gen_margin'])} | {f(row['w_loc_nll'])} |"
            )
    lines.extend(
        [
            "",
            "## 3. Deficit weighting, hard tail, routing",
            "",
            "| model | arm | mean rho / deficit sum | effective requests mean/min | omega mean-max/max | terminal z p90/worst | terminal gap p90/worst | mean active/q | entropy/top1 | fallback |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        for arm, _ in ARM_PATHS:
            row = aggregate_lookup(aggregates, model, arm)
            zd = row["terminal_z_full6_distribution"]
            gd = row["terminal_gap_distribution"]
            lines.append(
                f"| {model} | {arm} | {f(row['rho_mean'])} / {f(row['deficit_sum_mean'])} | "
                f"{f(row['effective_request_count_mean'])}/{f(row['effective_request_count_min'])} | "
                f"{f(row['omega_maximum_mean'])}/{f(row['omega_maximum'])} | "
                f"{f(zd['p90'])}/{f(zd['worst'])} | {f(gd['p90'])}/{f(gd['worst'])} | "
                f"{f(row['active_layer_count_mean'])}/{f(row['q_mean'])} | {f(row['entropy_mean'])}/{f(row['top1_mean'])} | {row['soft_fallback_count']} |"
            )
    lines.extend(
        [
            "",
            "Llama의 최소 effective request count는 Neutral/Soft `1.001868/1.004852`, 최대 omega는 `0.999067/0.997581`이었다. Qwen은 `1.173986/1.152883`, `0.921489/0.930508`이었다. 따라서 일부 step에서는 weight가 거의 단일 request에 집중됐다. binary concentration threshold는 계약에 없으므로 failure gate로 사용하지 않았다.",
            "",
            "`per-request-step.json`에는 3,160개 request-step의 `ellW`, `ellZ`, deficit, omega, actual progress, observation-only chi, W−z gap이 있다. `per-step.json`에는 316개 step의 5 slopes, active mask, q, pi/v/hv, P/common cap/energy, entropy/top1 및 identity가 있다.",
            "Section 3의 writer 통계는 typed incomplete의 유효 4-step prefix를 포함한 316 step 분모이고, terminal z/W tail은 valid endpoint만 사용한다.",
            "",
            "## 4. P1AGG → P1DW 파일럿 비교",
            "",
            "아래 값은 동일 final source·case01에서 `P1DW - P1AGG`이다. 파일럿 1 case의 직접 산술이며 production 효과 추정값은 아니다.",
            "",
            "| model | arm | ΔW E/G/L | ΔW Eff/Gen NLL | Δterminal gap | Δactual/pred | ΔP | Δcommon cap / energy |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        for arm, _ in ARM_PATHS:
            row = comp_lookup(comparison_aggs, "P1AGG_TO_P1DW_PILOT", model, arm)
            d = row["delta_means"]
            lines.append(
                f"| {model} | {arm} | {signed(d['delta_w_eff_correct'])} / {signed(d['delta_w_gen_correct'])} / {signed(d['delta_w_loc_correct'])} | "
                f"{signed(d['delta_w_eff_nll'])} / {signed(d['delta_w_gen_nll'])} | {signed(d['delta_terminal_w_minus_z_full6_gap'])} | "
                f"{signed(d['delta_actual_progress_sum'])} / {signed(d['delta_predicted_progress_sum'])} | "
                f"{signed(d['delta_structural_p_terminal'])} | {signed(d['delta_common_capacity_sum'])} / {signed(d['delta_common_bf16_energy_sum'])} |"
            )
    lines.extend(
        [
            "",
            "## 5. DW Neutral → DW Soft production 짝 비교",
            "",
            "아래 값은 valid paired case에서 `Soft - Neutral`이다. Llama는 case04 Soft incomplete를 제외한 9쌍, Qwen은 10쌍이다.",
            "",
            "| model | pairs | ΣΔW E/G/L | mean ΔEff/Gen NLL | mean Δz/W full6 | mean Δgap / tail detail | mean ΔP | mean Δcommon cap/energy | mean Δentropy/top1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        row = comp_lookup(comparison_aggs, "DW_NEUTRAL_TO_DW_SOFT_PRODUCTION", model, "SOFT_MINUS_NEUTRAL")
        dm = row["delta_means"]
        ds = row["delta_sums"]
        # gap tail deltas are derived from the paired endpoint table in the machine rows;
        # report the directly recorded mean terminal gap here and use aggregate empirical tails below.
        lines.append(
            f"| {model} | {row['matched_cases']} | {signed(ds['delta_w_eff_correct'])} / {signed(ds['delta_w_gen_correct'])} / {signed(ds['delta_w_loc_correct'])} | "
            f"{signed(dm['delta_w_eff_nll'])} / {signed(dm['delta_w_gen_nll'])} | {signed(dm['delta_z_full6_nll'])} / {signed(dm['delta_w_full6_nll'])} | "
            f"{signed(dm['delta_terminal_w_minus_z_full6_gap'])} / `per-case table` | {signed(dm['delta_structural_p_terminal'])} | "
            f"{signed(dm['delta_common_capacity_sum'])} / {signed(dm['delta_common_bf16_energy_sum'])} | {signed(dm['delta_mean_entropy'])} / {signed(dm['delta_mean_top1'])} |"
        )
    lines.extend(
        [
            "",
            "Llama 9쌍에서 Soft는 W Eff `+1/90`, Gen `0/180`, Loc `+2/900`; mean terminal gap `−0.0615765`였다. 같은 쌍에서 P `+0.00250302`, common capacity `+0.00152444`, energy `+1.92872`였다. Qwen 10쌍에서 Soft는 W Eff `0/100`, Gen `+3/200`, Loc `+2/1000`; gap `+0.000188945`, P `−0.448631`, common capacity `−0.0762948`, energy `−20.2086`였다.",
            "",
            "Soft weighted-strength residual은 모든 실행 step에서 locked tolerance 안이고 reduced-strength candidate는 0이다. Llama Soft의 fallback 3건은 exact Neutral full-strength였고, Qwen fallback은 0이다.",
            "",
            "## 6. P2 residual-transport writer → P1DW shared writer",
            "",
            "동일 model/case/stream/order/evaluator의 P2R2를 reference로 사용했다. 아래 값은 `P1DW - P2R2`이며 Llama Soft는 9쌍이다.",
            "",
            "| model | arm | pairs | ΣΔz E/G | mean Δz Eff/Gen NLL | ΣΔW E/G/L | mean ΔW Eff/Gen NLL | mean Δz/W full6 | mean Δgap |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        for arm, _ in ARM_PATHS:
            row = comp_lookup(comparison_aggs, "P2_RESIDUAL_TRANSPORT_TO_P1DW_SHARED_WRITER", model, arm)
            dm, ds = row["delta_means"], row["delta_sums"]
            lines.append(
                f"| {model} | {arm} | {row['matched_cases']} | {signed(ds['delta_z_eff_correct'])} / {signed(ds['delta_z_gen_correct'])} | "
                f"{signed(dm['delta_z_eff_nll'])} / {signed(dm['delta_z_gen_nll'])} | "
                f"{signed(ds['delta_w_eff_correct'])} / {signed(ds['delta_w_gen_correct'])} / {signed(ds['delta_w_loc_correct'])} | "
                f"{signed(dm['delta_w_eff_nll'])} / {signed(dm['delta_w_gen_nll'])} | "
                f"{signed(dm['delta_z_full6_nll'])} / {signed(dm['delta_w_full6_nll'])} | {signed(dm['delta_terminal_w_minus_z_full6_gap'])} |"
            )
    lines.extend(
        [
            "",
            "P2R2 legacy raw capacity는 P2R7 actual-BF16 common coordinate와 다르므로 `NOT_COMPARABLE_LEGACY_COORDINATE`; P2R2 common BF16 energy는 `NOT_RECORDED`이다.",
            "",
            "## 7. Official AlphaEdit → P1DW",
            "",
            "동일 model+case Official reference를 공통 arm-independent baseline으로 join했다. 아래 값은 `P1DW - Official`이다.",
            "",
            "| model | arm | pairs | ΣΔE/G/L | mean ΔEff NLL / margin | Official latent-z / Gen·Loc continuous / common cost |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for model in MODELS:
        for arm, _ in ARM_PATHS:
            row = comp_lookup(comparison_aggs, "OFFICIAL_ALPHAEDIT_TO_P1DW", model, arm)
            dm, ds = row["delta_means"], row["delta_sums"]
            lines.append(
                f"| {model} | {arm} | {row['matched_cases']} | {signed(ds['delta_w_eff_correct'])} / {signed(ds['delta_w_gen_correct'])} / {signed(ds['delta_w_loc_correct'])} | "
                f"{signed(dm['delta_w_eff_nll'])} / {signed(dm['delta_w_eff_margin'])} | NOT_RECORDED / NOT_RECORDED / NOT_RECORDED |"
            )
    lines.extend(
        [
            "",
            "## 8. 모델별 과학 분석",
            "",
            "### Llama3-8B-Instruct",
            "",
            "- Target 유지: P2R2 대비 Neutral z Eff/Gen count는 `−2/+1`, Soft 9쌍은 `0/+1`; z Eff NLL은 Neutral `+0.104392`, Soft `+0.0761006`, z Gen NLL은 `−0.0420745/−0.0321196`였다. Eff 연속값과 count는 Neutral에서 유지되지 않았고 Gen은 유지 또는 증가했다.",
            "- Writer hard tail: P2R2 대비 Neutral mean terminal W−z gap은 `+0.00113795`, Soft는 `−0.0587160`였다. Production Soft−Neutral 9쌍 gap은 `−0.0615765`이고 W E/G count는 `+1/0`이다. 따라서 shared writer의 hard-tail 감소 신호는 Soft에서 관측됐지만 Neutral에서는 관측되지 않았다.",
            "- Edit strength/Official: Neutral은 Official 대비 E/G/L `−1/+1/−17`, Soft 9쌍은 `0/0/−9`; Eff NLL delta는 `+0.204815/+0.0879393`이다. Official 수준의 모든 terminal 지표 동시 회복은 기록되지 않았다.",
            "- Barrier: Soft는 same-strength였고 Loc `+2/900`, gap 감소를 기록했지만 P/common capacity/energy가 모두 증가했다. 동일 strength에서 update cost 감소 조건은 충족되지 않았다.",
            "- Typed outcome: Soft case04의 no-positive direction 1건과 Neutral weighted nonlinear overshoot 1 step이 있다. Soft endpoint 분모는 9/10이다.",
            "",
            "### Qwen2.5-7B-Instruct",
            "",
            "- Target 유지: P2R2 대비 z Eff count는 두 arm 모두 동일, z Gen은 Neutral/Soft `+4/+6`; z Eff NLL은 `+0.000180166/+0.0000957751`, z Gen NLL은 `−0.197706/−0.154295`였다. discrete target strength는 유지됐고 Gen count가 증가했다.",
            "- Writer hard tail: P2R2 대비 mean terminal W−z gap은 Neutral/Soft `+0.00138569/+0.00157626`로 증가했다. W Gen count는 `+2/+5`였으므로 discrete strength 증가는 있었지만 z→W NLL 소실 감소는 기록되지 않았다.",
            "- Edit strength/Official: Official 대비 E/G/L은 Neutral `0/−5/+13`, Soft `0/−2/+15`; Eff NLL delta는 `−0.0294006/−0.0293785`다. Eff는 맞았지만 Gen count는 Official에 미달했다.",
            "- Barrier: Soft는 same-strength에서 P/common capacity/energy를 각각 `−0.448631/−0.0762948/−20.2086` 줄였고 E/G/Loc count는 `0/+3/+2`였다. 반면 Gen NLL과 terminal W−z gap은 `+0.0814673/+0.000188945` 증가했다. cost 감소와 count 비붕괴는 관측됐고 gap 감소는 관측되지 않았다.",
            "- Locality 경계: Soft−Neutral에서 Loc `+2`와 W−z gap 증가가 함께 있어 계약의 `UNDER_EDIT_LOCALITY_ILLUSION` 신호 조건에 해당한다. 동시에 Gen count `+3`을 별도 기록한다.",
            "",
            "## 9. Failure taxonomy",
            "",
            "| taxonomy | 관측 |",
            "|---|---|",
            "| TARGET_HARD | terminal z full6 p90/worst와 request rows를 기록; 사전 binary threshold가 없어 count는 NOT_BINARIZED |",
            "| WRITER_HARD | terminal W−z gap p90/worst와 request rows를 기록; Llama Soft 쌍에서 감소, Qwen에서 증가 |",
            "| WEIGHT_CONCENTRATION | 최대 omega Llama N/S `0.999067/0.997581`, Qwen `0.921489/0.930508`; 최소 Neff는 각각 `1.001868/1.004852`, `1.173986/1.152883` |",
            "| SHARED_ROUTE_NO_POSITIVE_DIRECTION | Llama Soft case04 outer4에서 1 case; typed incomplete |",
            "| NONLINEAR_OVERSHOOT | Llama Neutral 1 step; 나머지 0 |",
            "| SOFT_NO_DOF | fallback 외 same-state selected pi=Neutral pi인 step 0; fallback 3 step은 별도 |",
            "| BARRIER_PROXY_MISMATCH | aggregate에서 Structural-P 감소와 common capacity/Loc 악화가 함께 나타난 model은 0; per-case 원값은 comparisons.json에 유지 |",
            "| UNDER_EDIT_LOCALITY_ILLUSION | Qwen Soft−Neutral aggregate에서 Loc +2와 gap +0.000188945가 함께 관측; Llama는 gap 감소 |",
            "| TECHNICAL_INVALID | 0 |",
            "",
            "## 10. Compute ledger와 runtime",
            "",
            "| model | arm | endpoint | target F/B | KL F/B | response F/VJP | weighted F/B/tokens | post-write F | mat | target/writer/eval/total mean sec |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        for arm, _ in ARM_PATHS:
            row = aggregate_lookup(aggregates, model, arm)
            comp = row["compute_sums_valid_endpoints"]
            lines.append(
                f"| {model} | {arm} | {row['endpoints']} | {comp['target_forward_count']}/{comp['target_backward_count']} | "
                f"{comp['kl_forward_count']}/{comp['kl_backward_count']} | {comp['physical_response_forward_count']}/{comp['physical_response_batched_vjp_count']} | "
                f"{row['weighted_objective_forward_count']}/{row['weighted_objective_backward_count']}/{row['weighted_objective_tokens']} | "
                f"{comp['post_write_objective_forward_count']} | {comp['writer_materialization_count']} | "
                f"{f(row['target_wall_seconds_mean'])}/{f(row['writer_wall_seconds_mean'])}/{f(row['terminal_eval_wall_seconds_mean'])}/{f(row['total_wall_seconds_mean'])} |"
            )
    lines.extend(
        [
            "",
            "Valid 39 endpoints의 authoritative terminal compute 합계는 target F/B `6435/4680`, KL F/B `4875/4680`, physical capture F `1755`, weighted response F/VJP `1560/312`, post-write F `1755`, writer materialization `312`이다. Incomplete prefix는 target receipt 15, writer/response/materialization receipt 4개이며 실패 endpoint의 top-level terminal compute aggregate는 `NOT_RECORDED_TYPED_INCOMPLETE`다. 316 step의 weighted objective F/B/tokens는 `1580/1580/321020`이다.",
            "",
            "Array task wall은 result terminal 기준 Llama `3574.791342s`, Qwen `3847.138456s`다. Endpoint 평균 runtime은 위 표처럼 target/writer/evaluator/total로 분리했다. Peak GPU/host memory와 scheduler MaxRSS는 scientific receipt에 `NOT_RECORDED`다.",
            "",
            "## 11. 최종 질문에 대한 분리 답변",
            "",
            "- Llama: P2 target의 Gen strength는 유지됐지만 Neutral Eff target 지표와 연속 Eff NLL은 P2R2 수준을 유지하지 못했다. P1DW Soft는 Neutral 대비 W−z hard-tail과 Loc를 개선했으나 update P/common capacity/energy를 줄이지 못했고 1개 typed incomplete가 있었다. 따라서 ‘strong target 유지 + hard-tail 감소 + same-strength cost/locality 개선’의 세 조건이 동시에 성립하지 않았다.",
            "- Qwen: target discrete E/G는 P2R2 수준을 유지하거나 증가했다. P1DW는 P2R2보다 W E/G/L count가 같거나 높았지만 terminal W−z NLL gap은 증가했다. Soft는 same-strength에서 P/common capacity/energy를 줄이고 E/G/Loc count를 비붕괴시켰으나 gap과 Gen NLL은 증가했다. 따라서 cost/count 조건은 성립했지만 z→W 소실 감소 조건은 성립하지 않았다.",
            "- 두 모델을 평균해 단일 결론을 만들지 않았다. `scientific_promotion=false`다.",
            "",
            "## 12. FACT / INFERENCE / TECHNICAL_FAIL / SCIENTIFIC_FAIL / NOT_RECORDED",
            "",
            "### FACT",
            "",
            "- 위 raw-free 원값, 산술 delta, 40 attempts/39 endpoints/1 typed incomplete, 316 step/3,160 request-step, source/stream/evaluator identities.",
            "- 모든 accepted step의 5-variable routing, one-h, full weighted strength, candidate/retry/backtracking0.",
            "",
            "### INFERENCE",
            "",
            "- §8과 §11의 모델별 결론은 contract §15 질문에 대해 paired arithmetic와 tail 분포로만 도출했다.",
            "- Weight concentration과 hard-tail binary threshold는 사전 수치가 없어 gate로 이산화하지 않았다.",
            "",
            "### TECHNICAL_FAIL",
            "",
            "- 0. 기술 무결성 위반은 없다.",
            "",
            "### SCIENTIFIC_FAIL / typed scientific outcome",
            "",
            "- Llama Soft case04 `SCIENTIFIC_SHARED_WRITER_NO_POSITIVE_DIRECTION` 1건. 실패 prefix는 endpoint로 승격하거나 impute하지 않았다.",
            "- `NO_SEMANTIC_DEFICIT`은 0건이다.",
            "",
            "### NOT_RECORDED / NOT_COMPARABLE",
            "",
            "- Official latent-z, Gen/Loc continuous NLL·margin, Structural-P/common capacity/common BF16 energy/edit-core.",
            "- P2R2 및 P1R43/P1R35 legacy raw capacity와 P2R7 common actual-BF16 capacity의 직접 비교는 `NOT_COMPARABLE_LEGACY_COORDINATE`.",
            "- Stepwise heldout Eff/Gen/Loc는 계약상 0회라 `NOT_RECORDED`.",
            "- Llama Soft case04 terminal E/G/L/z/W는 typed incomplete라 `NOT_RECORDED`, imputation 0.",
            "",
            "## 13. 산출물",
            "",
            "- `per-attempt.json`: 40 rows.",
            "- `per-step.json`: 316 rows.",
            "- `per-request-step.json`: 3,160 rows.",
            "- `comparisons.json`: P1AGG→P1DW, DW N→S, P2R2→P1DW, Official→P1DW case rows.",
            "- `aggregates.json`: 4 cell aggregates 및 comparison aggregates.",
            "- `independent-review.json`, `analysis-manifest.json`, `analysis-receipt.json`: rehash/gate/rooted receipt.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Refuse to overwrite any final member.
    member_names = (
        "p2r7-production-tech-r2-final-analysis-ko.md",
        "per-attempt.json",
        "per-step.json",
        "per-request-step.json",
        "comparisons.json",
        "aggregates.json",
        "independent-review.json",
        "analysis-manifest.json",
        "analysis-receipt.json",
    )
    for name in member_names:
        if (output / name).exists():
            raise FileExistsError(output / name)

    references = load_allowed_references(repo)
    attempts: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    request_steps: list[dict[str, Any]] = []
    input_members: list[dict[str, Any]] = []

    result_roots = {
        "llama3-8b-inst": repo / "local/odebf/results/s05-p2r7-p2target-p1dw-b10x10-llama3-8b-inst-paired-v1",
        "qwen2.5-7b-inst": repo / "local/odebf/results/s05-p2r7-p2target-p1dw-b10x10-qwen2.5-7b-inst-paired-v1",
    }
    expected_job_walls = {"llama3-8b-inst": 3574.7913423161954, "qwen2.5-7b-inst": 3847.1384564973414}
    for model, root in result_roots.items():
        job_terminal = read_json(root / "terminal.json")
        if (
            job_terminal["source_head"] != SOURCE_HEAD
            or job_terminal["attempt_count"] != 20
            or abs(float(job_terminal["wall_seconds"]) - expected_job_walls[model]) > 1.0e-9
        ):
            raise AssertionError(f"job terminal identity mismatch: {model}")
        input_members.extend(
            [
                {"name": f"{model}-job-terminal", "path": str(root / "terminal.json"), "sha256": file_sha(root / "terminal.json")},
                {"name": f"{model}-job-manifest", "path": str(root / "manifest.json"), "sha256": file_sha(root / "manifest.json")},
            ]
        )
        for case_index in range(1, 11):
            for arm, arm_path in ARM_PATHS:
                case_root = root / "raw/cases" / f"case-{case_index:02d}" / arm_path
                step_files = sorted((case_root / "raw/writer").glob("step-*.json"))
                case_steps = [load_step(path, model=model, case_index=case_index, arm=arm) for path in step_files]
                steps.extend(case_steps)
                for step in case_steps:
                    request_steps.extend(request_step_rows(step))
                terminal_path = case_root / "terminal.json"
                if terminal_path.exists():
                    attempts.append(endpoint_from_terminal(terminal_path, model=model, case_index=case_index, arm=arm, steps=case_steps))
                else:
                    failure_path = case_root / "failure.json"
                    target_files = sorted((case_root / "raw/target").glob("microstep-*.json"))
                    attempts.append(failure_attempt(failure_path, model=model, case_index=case_index, arm=arm, steps=case_steps, target_files=target_files))

    if len(attempts) != 40 or len([row for row in attempts if row["status"] == "COMPLETE"]) != 39:
        raise AssertionError("attempt/endpoint denominator differs")
    if len(steps) != 316 or len(request_steps) != 3160:
        raise AssertionError("step/request-step denominator differs")
    if sum(int(row["target_microsteps"]) for row in attempts) != 951:
        raise AssertionError("target microstep denominator differs")
    failures = [row for row in attempts if row["status"] != "COMPLETE"]
    if len(failures) != 1 or failures[0]["failure_file_sha256"] != "fbd24bef9511c5aa05c4abe673049976ed83ffa2ea99b859698ab129cd9f7c37":
        raise AssertionError("typed failure identity differs")

    max_omega_residual = max(abs(float(row["omega_sum"]) - 1.0) for row in steps)
    max_strength_residual = max(abs(float(row["strength_residual"])) for row in steps)
    if max_omega_residual != 2.220446049250313e-16:
        raise AssertionError("omega residual differs")
    if abs(max_strength_residual - 5.186961971048731e-13) > 1.0e-25:
        raise AssertionError("strength residual differs")
    if sum(bool(row["fallback_to_neutral"]) for row in steps) != 3:
        raise AssertionError("Soft fallback denominator differs")
    if sum(int(row["negative_actual_count"]) for row in steps) != 83:
        raise AssertionError("negative request transition denominator differs")
    if any(
        int(row["routing_variable_count"]) != 5
        or int(row["request_layer_response_matrix_count"]) != 0
        or int(row["physical_h_application_count"]) != 1
        or int(row["residual_inverse_h_count"]) != 1
        or int(row["second_h_application_count"]) != 0
        or int(row["retry_count"]) != 0
        or int(row["backtracking_count"]) != 0
        or row["actual_bf16_common_metric_definition"] != COMMON_METRIC_DEFINITION
        for row in steps
    ):
        raise AssertionError("technical step contract differs")

    comparisons = build_comparisons(attempts, references)
    aggregates = cell_aggregates(attempts, steps)
    comparison_aggs = comparison_aggregates(comparisons)
    aggregate_payload = {
        "schema": "ode-edit-s05-p2r7-production-aggregates/v1",
        "cells": aggregates,
        "comparisons": comparison_aggs,
        "technical": {
            "attempts": 40,
            "endpoints": 39,
            "typed_incomplete": 1,
            "steps": 316,
            "request_steps": 3160,
            "target_microsteps": 951,
            "omega_sum_max_abs_residual": max_omega_residual,
            "strength_max_abs_residual": max_strength_residual,
            "soft_fallback_count": 3,
            "negative_request_transitions": 83,
            "weighted_nonlinear_overshoot_steps": sum(bool(row["nonlinear_overshoot"]) for row in steps),
            "no_semantic_deficit_steps": sum(bool(row["no_semantic_deficit"]) for row in steps),
        },
    }
    aggregate_payload["root_sha256"] = canonical_hash(aggregate_payload)

    report = report_text(aggregates, comparison_aggs, attempts, steps)
    primary = {
        "p2r7-production-tech-r2-final-analysis-ko.md": report.encode("utf-8") + b"\n",
        "per-attempt.json": canonical_bytes(attempts) + b"\n",
        "per-step.json": canonical_bytes(steps) + b"\n",
        "per-request-step.json": canonical_bytes(request_steps) + b"\n",
        "comparisons.json": canonical_bytes(comparisons) + b"\n",
        "aggregates.json": canonical_bytes(aggregate_payload) + b"\n",
    }
    for name, payload in primary.items():
        write_once(output / name, payload)

    primary_rows = {
        "p2r7-production-tech-r2-final-analysis-ko.md": "markdown",
        "per-attempt.json": len(attempts),
        "per-step.json": len(steps),
        "per-request-step.json": len(request_steps),
        "comparisons.json": len(comparisons),
        "aggregates.json": len(aggregates) + len(comparison_aggs),
    }
    primary_meta = [file_meta(output / name, row_count=primary_rows[name]) for name in primary]
    review = {
        "schema": "ode-edit-s05-p2r7-production-independent-review/v1",
        "analysis_agent": "/root/p2r7_production_analysis",
        "implementation_agent_separate": True,
        "scope": "P2R7 job20161 raw-free production results and allowed immutable references only",
        "model_action_count": 0,
        "evaluator_action_count": 0,
        "gpu_action_count": 0,
        "slurm_action_count": 0,
        "tracked_source_mutation_count": 0,
        "existing_result_mutation_count": 0,
        "primary_members_rehashed": primary_meta,
        "row_counts": {"attempts": len(attempts), "steps": len(steps), "request_steps": len(request_steps), "comparisons": len(comparisons)},
        "technical_gate": "PRODUCTION_TECHNICAL_PASS_WITH_ONE_TYPED_SCIENTIFIC_INCOMPLETE",
        "typed_scientific_outcome": "SCIENTIFIC_SHARED_WRITER_NO_POSITIVE_DIRECTION",
        "scientific_promotion": False,
        "review_pass": True,
    }
    review["identity_sha256"] = canonical_hash(review)
    write_json_once(output / "independent-review.json", review)

    generator_path = Path(__file__).resolve()
    output_meta = primary_meta + [
        file_meta(output / "independent-review.json", row_count=1),
        file_meta(generator_path, row_count="source"),
    ]
    members_root = canonical_hash(output_meta)
    manifest = {
        "schema": "ode-edit-s05-p2r7-production-analysis-manifest/v1",
        "instruction_id": INSTRUCTION_ID,
        "job_id": JOB_ID,
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "source_parent": SOURCE_PARENT,
        "exact_p2r2_parent": P2R2_PARENT,
        "contract_sha256": CONTRACT_SHA,
        "stream_root": STREAM_ROOT,
        "order_sha256": ORDER_SHA,
        "evaluator_sha256": EVALUATOR_SHA,
        "aggregator_sha256": AGGREGATOR_SHA,
        "numerical_lock": {"sha256": NUMERICAL_LOCK_SHA, "root": NUMERICAL_LOCK_ROOT},
        "source_manifest": {"sha256": SOURCE_MANIFEST_SHA, "root": SOURCE_MANIFEST_ROOT},
        "scheduler": {"job_id": JOB_ID, "tasks": 2, "completed": 2, "exit_code": "0:0"},
        "inputs": input_members
        + [
            {"name": "pilot-report", "sha256": PILOT_REPORT_SHA},
            {"name": "pilot-per-endpoint", "sha256": PILOT_ENDPOINT_SHA},
            {"name": "pilot-per-step", "sha256": PILOT_STEP_SHA},
            {"name": "p2r2-report", "sha256": P2R2_REPORT_SHA},
            {"name": "p2r2-per-case", "sha256": P2R2_CASE_SHA},
            {"name": "p2r2-per-step-request", "sha256": P2R2_STEP_REQUEST_SHA},
            {"name": "p2r2-comparisons", "sha256": P2R2_COMPARISON_SHA},
            {"name": "p1r43-report-reference", "sha256": P1R43_REPORT_SHA},
            {"name": "p1r35-report-reference", "sha256": P1R35_REPORT_SHA},
        ],
        "outputs": output_meta,
        "members_root_sha256": members_root,
        "attempts": 40,
        "endpoints": 39,
        "typed_incomplete": 1,
        "step_rows": 316,
        "request_step_rows": 3160,
        "target_microsteps": 951,
        "scientific_promotion": False,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    write_json_once(output / "analysis-manifest.json", manifest)

    receipt = {
        "schema": "ode-edit-s05-p2r7-production-analysis-receipt/v1",
        "instruction_id": INSTRUCTION_ID,
        "analysis_agent": "/root/p2r7_production_analysis",
        "implementation_agent_separate": True,
        "analysis_scope": "P2R7 production raw-free results and immutable matched references only",
        "model_action_count": 0,
        "evaluator_action_count": 0,
        "gpu_action_count": 0,
        "slurm_action_count": 0,
        "tracked_source_mutation_count": 0,
        "existing_result_mutation_count": 0,
        "job_id": JOB_ID,
        "scheduler_tasks_terminal": 2,
        "scheduler_tasks_completed": 2,
        "scheduler_exit_code": "0:0",
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "attempts": 40,
        "endpoints": 39,
        "typed_incomplete": 1,
        "step_rows": 316,
        "request_step_rows": 3160,
        "target_microsteps": 951,
        "technical_gate": "PRODUCTION_TECHNICAL_PASS_WITH_ONE_TYPED_SCIENTIFIC_INCOMPLETE",
        "scientific_outcome": "MODEL_SEPARATED_MIXED_SIGNAL_WITH_LLAMA_SOFT_TYPED_INCOMPLETE",
        "scientific_promotion": False,
        "W0_pointer_pass": 40,
        "W0_bytes_pass": 40,
        "action_freeze_pass": 39,
        "routing_variable_count_unique": [5],
        "request_layer_response_matrix_count": 0,
        "writer_materialization_count": 316,
        "candidate_materialization_count": 0,
        "retry_count": 0,
        "backtracking_count": 0,
        "p2r6_qp_import_count": 0,
        "p2r6_shadow_solve_count": 0,
        "soft_fallback_count": 3,
        "typed_scientific_failure_count": 1,
        "omega_sum_max_abs_residual": max_omega_residual,
        "strength_max_abs_residual": max_strength_residual,
        "negative_request_transition_count": 83,
        "analysis_manifest_sha256": file_sha(output / "analysis-manifest.json"),
        "analysis_manifest_identity_sha256": manifest["identity_sha256"],
        "members_root_sha256": members_root,
        "independent_review_sha256": file_sha(output / "independent-review.json"),
        "rehash_pass": True,
    }
    receipt["analysis_root_sha256"] = canonical_hash(receipt)
    write_json_once(output / "analysis-receipt.json", receipt)
    print(json.dumps({"output": str(output), "members": len(output_meta) + 2, "members_root": members_root, "analysis_root": receipt["analysis_root_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
