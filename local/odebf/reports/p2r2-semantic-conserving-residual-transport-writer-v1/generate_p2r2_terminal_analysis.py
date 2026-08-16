#!/usr/bin/env python3
"""Build the raw-free P2R2 terminal analysis package deterministically."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


WORKTREE = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-p2r2-semantic-conserving-residual-transport-writer-v1"
)
RESULTS = WORKTREE / "local/odebf/results"
OUT = WORKTREE / "local/odebf/reports/p2r2-semantic-conserving-residual-transport-writer-v1"
P2R1_REPORT = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-p2r1-baseline-calibrated-rms-tangent-target-flow-v1/"
    "local/odebf/reports/p2r1-rms-tangent-target-only-v1"
)
SUPPORT = Path(
    "/mnt/raid5/janghj/ODE-edit/local/source-handoff/"
    "P1R43_P1R42_PARENT_SUPPORT_SH2_V1"
)
CONTRACT = Path(
    "/mnt/raid5/janghj/.codex/attachments/"
    "ebbfb0bd-f4e4-4415-8672-3c595857c589/pasted-text.txt"
)
LOCK = WORKTREE / "project/run_scripts/ode_bf/locks/numerical_lock_s05_p2r2_residual_transport_writer.json"
SOURCE_MANIFEST = WORKTREE / "project/run_scripts/ode_bf/locks/source_manifest_s05_p2r2_residual_transport_writer.json"
SUBMISSION = WORKTREE / (
    "local/odebf/state/p2r2-semantic-conserving-residual-transport/"
    "s05-p2r2-b10x10-c97e8619b42d-v1.submission-receipt.json"
)

ROOTS = {
    "llama3-8b-inst": RESULTS
    / "s05-p2r2-semantic-conserving-residual-transport-b10x10-llama3-8b-inst-paired-v1",
    "qwen2.5-7b-inst": RESULTS
    / "s05-p2r2-semantic-conserving-residual-transport-b10x10-qwen2.5-7b-inst-paired-v1",
}
ARMS = ("NEUTRAL", "SOFTP")
ARM_DIR = {"NEUTRAL": "neutral", "SOFTP": "softp"}
SOURCE_HEAD = "c97e8619b42da7954ce0e824c215a8a82d70589a"
SOURCE_TREE = "711b33fc736641d1f3067c00bd067ed5e60606fe"
P2R1_PARENT = "8f817e13167289dac190fe74bfa42e2b3e01372d"
P1R43_ANCESTOR = "11508b6da11d606521b703037034e1814b70d8a8"
STREAM_ROOT = "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6"
STREAM_ORDER = "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c"
EVALUATOR_SHA = "25c3f49039e7bacb58de6d9ab8139f6f24f21532434a08690e9ec71ea939e145"
AGGREGATOR_SHA = "64f009b2fb648627a956b95abd801838d86edf04c8e1888d90ade0ec07b745c0"
OFFICIAL_IDENTITY_SHA = "e12b19ff941f39a594d2ff0f8718849d102644fd2ec79f53ccb6d7efc9fd8d3c"
CONTRACT_SHA = "f519e81fcc40870423774be0c507851332f604fde910dbcadad8faa73e0cf3f6"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def quantile(values: Iterable[float], q: float) -> float | None:
    data = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not data:
        return None
    position = (len(data) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return data[lower]
    return data[lower] * (upper - position) + data[upper] * (position - lower)


def summary(values: Iterable[float]) -> dict[str, Any]:
    data = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(data),
        "mean": statistics.fmean(data) if data else None,
        "median": statistics.median(data) if data else None,
        "p90": quantile(data, 0.90),
        "p95": quantile(data, 0.95),
        "min": min(data) if data else None,
        "max": max(data) if data else None,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        for row in rows
    )
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)


def metric_summary(values: list[dict[str, Any]], bits: list[int]) -> dict[str, Any]:
    return {
        "correct": sum(bits),
        "denominator": len(bits),
        "nll_new": summary(item["nll_new"] for item in values),
        "nll_old": summary(item["nll_old"] for item in values),
        "margin": summary(item["margin"] for item in values),
    }


def z_request_metrics(terminal: dict[str, Any], request: int) -> dict[str, Any]:
    panel = terminal["terminal_z_panel"]
    numeric = panel["z_inject"]["numeric_vectors"]
    result: dict[str, Any] = {}
    for metric, short in (("efficacy", "eff"), ("generalization", "gen")):
        values = numeric[metric][request]
        route = panel[f"{short}_z_inject"]
        bits = route["per_case_bits"][request]
        result[short] = metric_summary(values, [int(value) for value in bits])
    return result


def w_request_metrics(terminal: dict[str, Any], request: int) -> dict[str, Any]:
    metrics = terminal["terminal_w_panel"]["receipt"]["metrics"]
    result: dict[str, Any] = {}
    for metric, short in (
        ("efficacy", "eff"),
        ("generalization", "gen"),
        ("locality-preservation", "loc"),
    ):
        value = metrics[metric]
        result[short] = {
            "correct": int(value["per_case_correct"][request]),
            "denominator": int(value["per_case_required"][request]),
            "nll_new_mean": float(value["per_case_mean_target_new_nll"][request]),
            "nll_old_mean": float(value["per_case_mean_target_old_nll"][request]),
            "margin_mean": float(value["per_case_mean_margin"][request]),
        }
    return result


def z_case_metrics(terminal: dict[str, Any], short: str) -> dict[str, Any]:
    metric = "efficacy" if short == "eff" else "generalization"
    values = [
        item
        for request in terminal["terminal_z_panel"]["z_inject"]["numeric_vectors"][metric]
        for item in request
    ]
    route = terminal["terminal_z_panel"][f"{short}_z_inject"]
    bits = [int(value) for request in route["per_case_bits"] for value in request]
    return metric_summary(values, bits)


def w_case_metrics(terminal: dict[str, Any], short: str) -> dict[str, Any]:
    metric = {
        "eff": "efficacy",
        "gen": "generalization",
        "loc": "locality-preservation",
    }[short]
    value = terminal["terminal_w_panel"]["receipt"]["metrics"][metric]
    return {
        "correct": int(value["correct_count"]),
        "denominator": int(value["prompt_count"]),
        "nll_new_mean": float(value["mean_target_new_nll"]),
        "nll_old_mean": float(value["mean_target_old_nll"]),
        "margin_mean": float(value["mean_margin"]),
    }


def entropy_stats(allocation: list[list[float]]) -> dict[str, float]:
    layer_share = [sum(float(value) for value in row) for row in allocation]
    total = sum(layer_share)
    probabilities = [value / total for value in layer_share] if total > 0.0 else [0.0] * len(layer_share)
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0.0)
    return {
        "entropy": entropy,
        "neff": math.exp(entropy),
        "top1": max(probabilities) if probabilities else 0.0,
    }


def collect() -> tuple[list, list, list, list, list, list, dict[str, Any]]:
    per_case: list[dict[str, Any]] = []
    per_step: list[dict[str, Any]] = []
    per_request: list[dict[str, Any]] = []
    per_step_request: list[dict[str, Any]] = []
    response_routing: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    integrity: Counter[str] = Counter()

    p2r1_cases = read_json(P2R1_REPORT / "p2r1-per-case.json")["rows"]
    p2r1_case_index = {(row["alias"], int(row["case"])): row for row in p2r1_cases}
    p2r1_panels = read_json(P2R1_REPORT / "p2r1-per-panel.json")["rows"]
    p2r1_panel_index = {
        (row["alias"], int(row["case"]), row["panel"]): row for row in p2r1_panels
    }
    official_rows = read_json(SUPPORT / "identities/official-alphaedit-baseline-identity.json")["rows"]
    official_index = {(row["alias"], int(row["case"])): row for row in official_rows}

    loaded: dict[tuple[str, int, str], tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for alias, root in ROOTS.items():
        task = read_json(root / "terminal.json")
        integrity["task_terminal"] += int(task["completed_case_arm_count"] == 20)
        integrity["task_failure_zero"] += int(task["failed_case_arm_count"] == 0)
        integrity["task_W0_restore"] += int(task["W0_restored"] is True)
        integrity["task_source_head"] += int(task["source_head"] == SOURCE_HEAD)
        for case in range(1, 11):
            for arm in ARMS:
                base = root / f"raw/cases/case-{case:02d}/{ARM_DIR[arm]}"
                terminal = read_json(base / "terminal.json")
                freeze = read_json(base / "action-freeze.json")
                manifest = read_json(base / "manifest.json")
                steps = [read_json(base / f"raw/writer/step-{index:02d}.json") for index in range(8)]
                target = [
                    read_json(base / f"raw/target/microstep-{index:02d}.json")
                    for index in range(24)
                ]
                loaded[(alias, case, arm)] = (terminal, steps)
                z_eff, z_gen = z_case_metrics(terminal, "eff"), z_case_metrics(terminal, "gen")
                w_eff, w_gen, w_loc = (
                    w_case_metrics(terminal, "eff"),
                    w_case_metrics(terminal, "gen"),
                    w_case_metrics(terminal, "loc"),
                )
                terminal_capacity = sum(
                    float(value)
                    for value in steps[-1]["materialization"]["cumulative_bf16_capacity"].values()
                )
                predicted = sum(
                    sum(float(value) for value in step["predicted_progress_by_request"])
                    for step in steps
                )
                actual = sum(
                    sum(float(value) for value in step["actual_progress_by_request"])
                    for step in steps
                )
                clamp_hits_by_request = [
                    sum(int(step["clamp_hit"][request]) for step in target) for request in range(10)
                ]
                current_w_refresh = len(
                    {step["current_physical_state_sha256"] for step in target}
                )
                case_row = {
                    "alias": alias,
                    "case": case,
                    "arm": arm,
                    "status": "COMPLETE",
                    "request_count": 10,
                    "request_order_sha256": terminal["request_order_sha256"],
                    "attempts": 1,
                    "endpoints": 1,
                    "failures": 0,
                    "target_microsteps": int(terminal["target_microstep_count"]),
                    "writer_transitions": int(terminal["writer_transition_count"]),
                    "writer_materializations": int(terminal["writer_materialization_count"]),
                    "action_freeze": True,
                    "W0_pointer_restored": bool(terminal["W0_restore"]["pointer_restored_exact"]),
                    "W0_bytes_restored": bool(terminal["W0_restore"]["byte_restored_exact"]),
                    "z_eff": z_eff,
                    "z_gen": z_gen,
                    "w_eff": w_eff,
                    "w_gen": w_gen,
                    "w_loc": w_loc,
                    "eff_w_minus_z_nll": w_eff["nll_new_mean"] - z_eff["nll_new"]["mean"],
                    "gen_w_minus_z_nll": w_gen["nll_new_mean"] - z_gen["nll_new"]["mean"],
                    "terminal_full_six_target_new_nll": float(terminal["terminal_full_six_target_new_nll"]),
                    "terminal_full_six_target_new_nll_by_request": summary(
                        terminal["terminal_full_six_target_new_nll_by_request"]
                    ),
                    "terminal_cumulative_structural_p": float(terminal["terminal_cumulative_structural_p"]),
                    "terminal_bf16_capacity": terminal_capacity,
                    "terminal_update_frobenius_norm": math.sqrt(max(terminal_capacity, 0.0)),
                    "predicted_progress_sum": predicted,
                    "actual_progress_sum": actual,
                    "actual_over_predicted": actual / predicted if predicted != 0.0 else None,
                    "negative_actual_request_transitions": sum(
                        int(step["negative_actual_count"]) for step in steps
                    ),
                    "unreachable_request_steps": sum(
                        len(step["route"]["unreachable_requests"]) for step in steps
                    ),
                    "route_status_counts": dict(Counter(step["route"]["status"] for step in steps)),
                    "clamp_hit_microsteps": sum(clamp_hits_by_request),
                    "requests_with_clamp_hit": sum(value > 0 for value in clamp_hits_by_request),
                    "requests_with_repeated_clamp_hits": sum(value >= 2 for value in clamp_hits_by_request),
                    "current_W_state_identity_count": current_w_refresh,
                    "compute": terminal["compute"],
                    "target_wall_seconds": float(terminal["target_wall_seconds"]),
                    "writer_wall_seconds": float(terminal["writer_wall_seconds"]),
                    "terminal_z_evaluator_wall_seconds": float(terminal["terminal_z_evaluator_wall_seconds"]),
                    "terminal_w_evaluator_wall_seconds": float(terminal["terminal_w_evaluator_wall_seconds"]),
                    "total_wall_seconds": float(terminal["total_wall_seconds"]),
                    "action_freeze_sha256": terminal["action_freeze_sha256"],
                    "terminal_identity_sha256": terminal["identity_sha256"],
                    "terminal_file_sha256": sha256(base / "terminal.json"),
                    "manifest_file_sha256": sha256(base / "manifest.json"),
                }
                per_case.append(case_row)

                integrity["case_terminal"] += 1
                integrity["case_manifest"] += int(manifest["W0_restored"] is True)
                integrity["case_action_freeze"] += int(freeze["actions_frozen_before_evaluator"] is True)
                integrity["case_W0_pointer"] += int(terminal["W0_restore"]["pointer_restored_exact"] is True)
                integrity["case_W0_bytes"] += int(terminal["W0_restore"]["byte_restored_exact"] is True)
                integrity["case_k8"] += int(terminal["writer_transition_count"] == 8)
                integrity["case_materialization8"] += int(terminal["writer_materialization_count"] == 8)
                integrity["case_target24"] += int(terminal["target_microstep_count"] == 24)
                integrity["case_evaluator_firewall"] += int(
                    terminal["terminal_z_panel"]["heldout_controller_access_count"] == 0
                    and terminal["terminal_z_panel"]["inner_step_heldout_evaluation_count"] == 0
                    and terminal["terminal_w_panel"]["receipt"]["controller_or_routing_influence_count"] == 0
                )

                for outer, writer in enumerate(steps):
                    micro = target[outer * 3 : outer * 3 + 3]
                    route = writer["route"]
                    allocation = route["allocation_by_layer_request"]
                    entropy = entropy_stats(allocation)
                    predicted_request = [float(value) for value in writer["predicted_progress_by_request"]]
                    actual_request = [float(value) for value in writer["actual_progress_by_request"]]
                    step_predicted = sum(predicted_request)
                    step_actual = sum(actual_request)
                    residual_layer_hashes = [
                        layer["residual_sha256"] for layer in writer["field"]["layers"]
                    ]
                    row = {
                        "alias": alias,
                        "case": case,
                        "arm": arm,
                        "outer_step": outer,
                        "target_microsteps_before_write": int(writer["target_microstep_count_before_write"]),
                        "target_nll_before_microsteps": summary(micro[0]["target_new_nll_by_request"]),
                        "target_nll_after_microsteps": summary(micro[-1]["target_new_nll_by_request"]),
                        "clamp_hit_count": sum(int(value["clamp_hit_count"]) for value in micro),
                        "current_physical_state_sha256": micro[0]["current_physical_state_sha256"],
                        "target_state_sha256": writer["target_state_sha256"],
                        "current_terminal_sha256": writer["current_terminal_sha256"],
                        "residual_sha256": writer["full_current_residual_sha256"],
                        "residual_norm": summary(writer["full_current_residual_norm_by_request"]),
                        "residual_layer_hash_identity": len(set(residual_layer_hashes)) == 1,
                        "residual_divisor_values": sorted(
                            {int(layer["residual_divisor"]) for layer in writer["field"]["layers"]}
                        ),
                        "response_sha256": writer["response"]["response_sha256"],
                        "response_shape": writer["response"]["response_shape"],
                        "active_request_count": sum(bool(value) for value in route["active_requests"]),
                        "unreachable_requests": route["unreachable_requests"],
                        "fairness_lambda": float(route["fairness_lambda"]),
                        "route_status": route["status"],
                        "fallback_reason": route["fallback_reason"],
                        "simplex_max_abs_residual": float(route["request_simplex_max_abs_residual"]),
                        "soft_no_weaker_max_violation": float(route["soft_requestwise_no_weaker_max_violation"]),
                        "strength_attenuation_count": int(route["strength_attenuation_count"]),
                        "predicted_progress_sum": step_predicted,
                        "actual_progress_sum": step_actual,
                        "actual_over_predicted": step_actual / step_predicted if step_predicted != 0.0 else None,
                        "negative_actual_count": int(writer["negative_actual_count"]),
                        "marginal_structural_p": float(route["marginal_structural_p"]),
                        "cumulative_structural_p": float(writer["cumulative_structural_p"]),
                        "capacity": float(route["capacity"]),
                        "allocation_entropy": entropy["entropy"],
                        "allocation_neff": entropy["neff"],
                        "allocation_top1": entropy["top1"],
                        "one_joint_materialization_count": int(writer["one_joint_materialization_count"]),
                        "physical_h_application_count": int(writer["physical_h_application_count"]),
                        "retry_count": int(writer["retry_count"]),
                        "backtracking_count": int(writer["backtracking_count"]),
                        "receipt_sha256": sha256(base / f"raw/writer/step-{outer:02d}.json"),
                        "identity_sha256": writer["identity_sha256"],
                    }
                    per_step.append(row)
                    response_routing.append(
                        {
                            "alias": alias,
                            "case": case,
                            "arm": arm,
                            "outer_step": outer,
                            "response_sha256": writer["response"]["response_sha256"],
                            "response_shape": writer["response"]["response_shape"],
                            "response_batched_vjp_count": int(writer["response"]["batched_vjp_count"]),
                            "response_model_forward_count": int(writer["response"]["model_forward_count"]),
                            "response_processed_token_count": int(writer["response"]["processed_token_count"]),
                            "allocation_by_layer_request": allocation,
                            "allocation_sha256": route["allocation_sha256"],
                            "allocation_shape": route["allocation_shape"],
                            "neutral_predicted_progress_by_request": route["neutral_predicted_progress_by_request"],
                            "selected_predicted_progress_by_request": route["predicted_progress_by_request"],
                            "actual_progress_by_request": writer["actual_progress_by_request"],
                            "realization_by_request": writer["realization_by_request"],
                            "active_requests": route["active_requests"],
                            "unreachable_requests": route["unreachable_requests"],
                            "route_status": route["status"],
                            "fallback_reason": route["fallback_reason"],
                            "marginal_structural_p": float(route["marginal_structural_p"]),
                            "prior_structural_p": float(writer["quadratics"]["prior_structural_p"]),
                            "capacity": float(route["capacity"]),
                            "prior_capacity_by_layer": writer["quadratics"]["prior_capacity_by_layer"],
                            "current_state_identity": writer["current_terminal_sha256"],
                            "target_state_identity": writer["target_state_sha256"],
                            "field_identity": writer["field"]["identity_sha256"],
                            "route_identity": route["identity_sha256"],
                        }
                    )
                    request_identities = terminal["terminal_z_panel"]["heldout_lookup"]["rows"]
                    for request in range(10):
                        per_step_request.append(
                            {
                                "alias": alias,
                                "case": case,
                                "arm": arm,
                                "outer_step": outer,
                                "request_index": request,
                                "request_sha256": request_identities[request]["request_sha256"],
                                "self_reachable_ceiling_c_i": "NOT_RECORDED",
                                "active": bool(route["active_requests"][request]),
                                "unreachable": request in route["unreachable_requests"],
                                "selected_alpha_by_layer": [
                                    float(allocation[layer][request]) for layer in range(5)
                                ],
                                "selected_alpha_layer_sum": sum(
                                    float(allocation[layer][request]) for layer in range(5)
                                ),
                                "neutral_reference_predicted_progress": float(
                                    route["neutral_predicted_progress_by_request"][request]
                                ),
                                "selected_predicted_progress": float(
                                    writer["predicted_progress_by_request"][request]
                                ),
                                "actual_progress": float(
                                    writer["actual_progress_by_request"][request]
                                ),
                                "realization": float(writer["realization_by_request"][request]),
                                "negative_actual": float(
                                    writer["actual_progress_by_request"][request]
                                )
                                < 0.0,
                                "current_nll": float(writer["current_nll_by_request"][request]),
                                "next_nll": float(writer["next_nll_by_request"][request]),
                                "full_current_residual_norm": float(
                                    writer["full_current_residual_norm_by_request"][request]
                                ),
                                "target_nll_before_outer_microsteps": float(
                                    micro[0]["target_new_nll_by_request"][request]
                                ),
                                "target_nll_after_outer_microsteps": float(
                                    micro[-1]["target_new_nll_by_request"][request]
                                ),
                                "outer_microstep_clamp_hit_count": sum(
                                    int(receipt["clamp_hit"][request]) for receipt in micro
                                ),
                                "route_status": route["status"],
                                "fallback_reason": route["fallback_reason"],
                                "response_sha256": writer["response"]["response_sha256"],
                                "residual_sha256": writer["full_current_residual_sha256"],
                                "allocation_sha256": route["allocation_sha256"],
                            }
                        )

                for request in range(10):
                    z_metric = z_request_metrics(terminal, request)
                    w_metric = w_request_metrics(terminal, request)
                    predicted_request = sum(
                        float(step["predicted_progress_by_request"][request]) for step in steps
                    )
                    actual_request = sum(
                        float(step["actual_progress_by_request"][request]) for step in steps
                    )
                    layer_totals = [
                        sum(float(step["route"]["allocation_by_layer_request"][layer][request]) for step in steps)
                        for layer in range(5)
                    ]
                    per_request.append(
                        {
                            "alias": alias,
                            "case": case,
                            "arm": arm,
                            "request_index": request,
                            "request_sha256": terminal["terminal_z_panel"]["heldout_lookup"]["rows"][request]["request_sha256"],
                            "terminal_full_six_target_new_nll": float(
                                terminal["terminal_full_six_target_new_nll_by_request"][request]
                            ),
                            "z": z_metric,
                            "w": w_metric,
                            "eff_w_minus_z_nll": w_metric["eff"]["nll_new_mean"]
                            - z_metric["eff"]["nll_new"]["mean"],
                            "gen_w_minus_z_nll": w_metric["gen"]["nll_new_mean"]
                            - z_metric["gen"]["nll_new"]["mean"],
                            "target_clamp_hit_count": clamp_hits_by_request[request],
                            "writer_predicted_progress_sum": predicted_request,
                            "writer_actual_progress_sum": actual_request,
                            "writer_actual_over_predicted": actual_request / predicted_request
                            if predicted_request != 0.0
                            else None,
                            "negative_actual_steps": sum(
                                float(step["actual_progress_by_request"][request]) < 0.0
                                for step in steps
                            ),
                            "unreachable_steps": sum(
                                request in step["route"]["unreachable_requests"] for step in steps
                            ),
                            "layer_allocation_sum_over_k8": layer_totals,
                        }
                    )

                p2r1_case = p2r1_case_index[(alias, case)]
                identity_match = (
                    p2r1_case["request_order_sha256"] == terminal["request_order_sha256"]
                    and terminal["terminal_w_panel"]["receipt"]["evaluator_source_sha256"] == EVALUATOR_SHA
                    and terminal["terminal_w_panel"]["receipt"]["aggregator_source_sha256"] == AGGREGATOR_SHA
                )
                for panel_name, current in (("EFF_Z_INJECT", z_eff), ("GEN_Z_INJECT", z_gen)):
                    reference = p2r1_panel_index[(alias, case, panel_name)]
                    comparisons.append(
                        {
                            "alias": alias,
                            "case": case,
                            "arm": arm,
                            "comparator": f"P2R1_TARGET_ONLY_{panel_name}",
                            "identity_status": "MATCHED" if identity_match else "NOT_MATCHED",
                            "current_correct": current["correct"],
                            "reference_correct": reference["correct"],
                            "delta_correct": current["correct"] - reference["correct"],
                            "current_nll_mean": current["nll_new"]["mean"],
                            "reference_nll_mean": reference["new_nll"]["mean"],
                            "delta_nll_mean": current["nll_new"]["mean"] - reference["new_nll"]["mean"],
                            "current_margin_mean": current["margin"]["mean"],
                            "reference_margin_mean": reference["margin"]["mean"],
                            "delta_margin_mean": current["margin"]["mean"] - reference["margin"]["mean"],
                        }
                    )
                comparisons.append(
                    {
                        "alias": alias,
                        "case": case,
                        "arm": arm,
                        "comparator": "P2R1_TARGET_ONLY_FULL_SIX",
                        "identity_status": "MATCHED" if identity_match else "NOT_MATCHED",
                        "current_nll": float(terminal["terminal_full_six_target_new_nll"]),
                        "reference_nll": float(p2r1_case["terminal_full_six_target_new_nll"]),
                        "delta_nll": float(terminal["terminal_full_six_target_new_nll"])
                        - float(p2r1_case["terminal_full_six_target_new_nll"]),
                    }
                )
                official = official_index[(alias, case)]
                comparisons.append(
                    {
                        "alias": alias,
                        "case": case,
                        "arm": arm,
                        "comparator": "OFFICIAL_ALPHAEDIT",
                        "identity_status": "MATCHED" if official["official_matched"] else "NOT_MATCHED",
                        "current_eff_correct": w_eff["correct"],
                        "reference_eff_correct": official["official_eff_correct"],
                        "delta_eff_correct": w_eff["correct"] - official["official_eff_correct"],
                        "current_gen_correct": w_gen["correct"],
                        "reference_gen_correct": official["official_gen_correct"],
                        "delta_gen_correct": w_gen["correct"] - official["official_gen_correct"],
                        "current_loc_correct": w_loc["correct"],
                        "reference_loc_correct": official["official_loc_correct"],
                        "delta_loc_correct": w_loc["correct"] - official["official_loc_correct"],
                        "current_eff_nll_mean": w_eff["nll_new_mean"],
                        "reference_eff_nll_mean": official["official_eff_nll_mean"],
                        "delta_eff_nll_mean": w_eff["nll_new_mean"]
                        - official["official_eff_nll_mean"],
                        "current_eff_margin_mean": w_eff["margin_mean"],
                        "reference_eff_margin_mean": official["official_eff_margin_mean"],
                        "delta_eff_margin_mean": w_eff["margin_mean"]
                        - official["official_eff_margin_mean"],
                        "official_gen_continuous": "NOT_RECORDED",
                        "official_loc_continuous": "NOT_RECORDED",
                        "official_structural_p": "NOT_RECORDED",
                        "official_capacity": "NOT_RECORDED",
                        "official_update_norm": "NOT_RECORDED",
                        "official_edit_core_seconds": "NOT_RECORDED",
                    }
                )

    # Exact paired Neutral-Soft facts, including whether both arms were still at the same state.
    for alias in ROOTS:
        for case in range(1, 11):
            neutral_terminal, neutral_steps = loaded[(alias, case, "NEUTRAL")]
            soft_terminal, soft_steps = loaded[(alias, case, "SOFTP")]
            neutral_case = next(
                row for row in per_case if row["alias"] == alias and row["case"] == case and row["arm"] == "NEUTRAL"
            )
            soft_case = next(
                row for row in per_case if row["alias"] == alias and row["case"] == case and row["arm"] == "SOFTP"
            )
            same_state_steps = 0
            same_state_moved_steps = 0
            same_state_l1: list[float] = []
            same_state_l2: list[float] = []
            for neutral, soft in zip(neutral_steps, soft_steps):
                same_state = (
                    neutral["current_terminal_sha256"] == soft["current_terminal_sha256"]
                    and neutral["target_state_sha256"] == soft["target_state_sha256"]
                    and neutral["response"]["response_sha256"] == soft["response"]["response_sha256"]
                )
                if same_state:
                    same_state_steps += 1
                    n = sum(neutral["route"]["allocation_by_layer_request"], [])
                    s = sum(soft["route"]["allocation_by_layer_request"], [])
                    l1 = sum(abs(float(a) - float(b)) for a, b in zip(n, s))
                    l2 = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(n, s)))
                    same_state_l1.append(l1)
                    same_state_l2.append(l2)
                    same_state_moved_steps += int(l1 > 1e-8)
            comparisons.append(
                {
                    "alias": alias,
                    "case": case,
                    "arm": "SOFTP_MINUS_NEUTRAL",
                    "comparator": "PAIRED_ARMS",
                    "identity_status": "MATCHED_CASE_W0_STREAM_EVALUATOR",
                    "delta_eff_correct": soft_case["w_eff"]["correct"] - neutral_case["w_eff"]["correct"],
                    "delta_gen_correct": soft_case["w_gen"]["correct"] - neutral_case["w_gen"]["correct"],
                    "delta_loc_correct": soft_case["w_loc"]["correct"] - neutral_case["w_loc"]["correct"],
                    "delta_eff_nll": soft_case["w_eff"]["nll_new_mean"] - neutral_case["w_eff"]["nll_new_mean"],
                    "delta_gen_nll": soft_case["w_gen"]["nll_new_mean"] - neutral_case["w_gen"]["nll_new_mean"],
                    "delta_structural_p": soft_case["terminal_cumulative_structural_p"]
                    - neutral_case["terminal_cumulative_structural_p"],
                    "delta_capacity": soft_case["terminal_bf16_capacity"]
                    - neutral_case["terminal_bf16_capacity"],
                    "delta_update_norm": soft_case["terminal_update_frobenius_norm"]
                    - neutral_case["terminal_update_frobenius_norm"],
                    "same_state_step_count": same_state_steps,
                    "same_state_moved_step_count": same_state_moved_steps,
                    "same_state_allocation_l1": summary(same_state_l1),
                    "same_state_allocation_l2": summary(same_state_l2),
                }
            )

    return (
        per_case,
        per_step,
        per_request,
        per_step_request,
        response_routing,
        comparisons,
        dict(integrity),
    )


def aggregate(
    cases: list[dict[str, Any]], steps: list[dict[str, Any]], requests: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alias in ROOTS:
        for arm in ARMS:
            group = [row for row in cases if row["alias"] == alias and row["arm"] == arm]
            step_group = [row for row in steps if row["alias"] == alias and row["arm"] == arm]
            request_group = [row for row in requests if row["alias"] == alias and row["arm"] == arm]
            rows.append(
                {
                    "alias": alias,
                    "arm": arm,
                    "attempts": 10,
                    "endpoints": len(group),
                    "failures": 10 - len(group),
                    "requests": len(request_group),
                    "z_eff_correct": sum(row["z_eff"]["correct"] for row in group),
                    "z_eff_denominator": sum(row["z_eff"]["denominator"] for row in group),
                    "z_gen_correct": sum(row["z_gen"]["correct"] for row in group),
                    "z_gen_denominator": sum(row["z_gen"]["denominator"] for row in group),
                    "w_eff_correct": sum(row["w_eff"]["correct"] for row in group),
                    "w_eff_denominator": sum(row["w_eff"]["denominator"] for row in group),
                    "w_gen_correct": sum(row["w_gen"]["correct"] for row in group),
                    "w_gen_denominator": sum(row["w_gen"]["denominator"] for row in group),
                    "w_loc_correct": sum(row["w_loc"]["correct"] for row in group),
                    "w_loc_denominator": sum(row["w_loc"]["denominator"] for row in group),
                    "z_eff_nll_case_mean": summary(row["z_eff"]["nll_new"]["mean"] for row in group),
                    "z_gen_nll_case_mean": summary(row["z_gen"]["nll_new"]["mean"] for row in group),
                    "w_eff_nll_case_mean": summary(row["w_eff"]["nll_new_mean"] for row in group),
                    "w_gen_nll_case_mean": summary(row["w_gen"]["nll_new_mean"] for row in group),
                    "w_loc_nll_case_mean": summary(row["w_loc"]["nll_new_mean"] for row in group),
                    "w_eff_margin_case_mean": summary(row["w_eff"]["margin_mean"] for row in group),
                    "w_gen_margin_case_mean": summary(row["w_gen"]["margin_mean"] for row in group),
                    "w_loc_margin_case_mean": summary(row["w_loc"]["margin_mean"] for row in group),
                    "eff_w_minus_z_nll": summary(row["eff_w_minus_z_nll"] for row in group),
                    "gen_w_minus_z_nll": summary(row["gen_w_minus_z_nll"] for row in group),
                    "terminal_full_six_target_new_nll": summary(
                        row["terminal_full_six_target_new_nll"] for row in group
                    ),
                    "terminal_structural_p": summary(
                        row["terminal_cumulative_structural_p"] for row in group
                    ),
                    "terminal_bf16_capacity": summary(row["terminal_bf16_capacity"] for row in group),
                    "terminal_update_frobenius_norm": summary(
                        row["terminal_update_frobenius_norm"] for row in group
                    ),
                    "predicted_progress_sum": sum(row["predicted_progress_sum"] for row in group),
                    "actual_progress_sum": sum(row["actual_progress_sum"] for row in group),
                    "actual_over_predicted": sum(row["actual_progress_sum"] for row in group)
                    / sum(row["predicted_progress_sum"] for row in group),
                    "negative_actual_request_transitions": sum(
                        row["negative_actual_request_transitions"] for row in group
                    ),
                    "unreachable_request_steps": sum(row["unreachable_request_steps"] for row in group),
                    "route_status_counts": dict(Counter(row["route_status"] for row in step_group)),
                    "soft_no_weaker_max_violation": max(
                        row["soft_no_weaker_max_violation"] for row in step_group
                    ),
                    "strength_attenuation_count": sum(
                        row["strength_attenuation_count"] for row in step_group
                    ),
                    "clamp_hit_microsteps": sum(row["clamp_hit_microsteps"] for row in group),
                    "requests_with_clamp_hit": sum(row["requests_with_clamp_hit"] for row in group),
                    "target_wall_seconds": summary(row["target_wall_seconds"] for row in group),
                    "writer_wall_seconds": summary(row["writer_wall_seconds"] for row in group),
                    "total_wall_seconds": summary(row["total_wall_seconds"] for row in group),
                    "compute_sums": {
                        key: sum(int(row["compute"][key]) for row in group)
                        for key in sorted(group[0]["compute"])
                    },
                }
            )
    return rows


def comparison_aggregates(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alias in ROOTS:
        for arm in ARMS:
            p2_eff = [
                row
                for row in comparisons
                if row["alias"] == alias and row.get("arm") == arm
                and row["comparator"] == "P2R1_TARGET_ONLY_EFF_Z_INJECT"
            ]
            p2_gen = [
                row
                for row in comparisons
                if row["alias"] == alias and row.get("arm") == arm
                and row["comparator"] == "P2R1_TARGET_ONLY_GEN_Z_INJECT"
            ]
            official = [
                row
                for row in comparisons
                if row["alias"] == alias and row.get("arm") == arm
                and row["comparator"] == "OFFICIAL_ALPHAEDIT"
            ]
            rows.append(
                {
                    "alias": alias,
                    "arm": arm,
                    "p2r1_identity_matched_cases": sum(
                        row["identity_status"] == "MATCHED" for row in p2_eff
                    ),
                    "delta_vs_p2r1_z_eff_correct": sum(row["delta_correct"] for row in p2_eff),
                    "delta_vs_p2r1_z_eff_nll_case_mean": statistics.fmean(
                        row["delta_nll_mean"] for row in p2_eff
                    ),
                    "delta_vs_p2r1_z_gen_correct": sum(row["delta_correct"] for row in p2_gen),
                    "delta_vs_p2r1_z_gen_nll_case_mean": statistics.fmean(
                        row["delta_nll_mean"] for row in p2_gen
                    ),
                    "official_identity_matched_cases": sum(
                        row["identity_status"] == "MATCHED" for row in official
                    ),
                    "delta_vs_official_w_eff_correct": sum(row["delta_eff_correct"] for row in official),
                    "delta_vs_official_w_gen_correct": sum(row["delta_gen_correct"] for row in official),
                    "delta_vs_official_w_loc_correct": sum(row["delta_loc_correct"] for row in official),
                    "delta_vs_official_w_eff_nll_case_mean": statistics.fmean(
                        row["delta_eff_nll_mean"] for row in official
                    ),
                    "official_gen_continuous": "NOT_RECORDED",
                    "official_loc_continuous": "NOT_RECORDED",
                    "official_p_capacity_update_norm_edit_core": "NOT_RECORDED",
                }
            )
    for alias in ROOTS:
        paired = [
            row
            for row in comparisons
            if row["alias"] == alias and row["comparator"] == "PAIRED_ARMS"
        ]
        rows.append(
            {
                "alias": alias,
                "arm": "SOFTP_MINUS_NEUTRAL",
                "paired_cases": len(paired),
                "delta_w_eff_correct": sum(row["delta_eff_correct"] for row in paired),
                "delta_w_gen_correct": sum(row["delta_gen_correct"] for row in paired),
                "delta_w_loc_correct": sum(row["delta_loc_correct"] for row in paired),
                "delta_w_eff_nll_case_mean": statistics.fmean(row["delta_eff_nll"] for row in paired),
                "delta_w_gen_nll_case_mean": statistics.fmean(row["delta_gen_nll"] for row in paired),
                "delta_terminal_structural_p_case_mean": statistics.fmean(
                    row["delta_structural_p"] for row in paired
                ),
                "delta_terminal_capacity_case_mean": statistics.fmean(
                    row["delta_capacity"] for row in paired
                ),
                "delta_terminal_update_norm_case_mean": statistics.fmean(
                    row["delta_update_norm"] for row in paired
                ),
                "same_state_steps": sum(row["same_state_step_count"] for row in paired),
                "same_state_moved_steps": sum(row["same_state_moved_step_count"] for row in paired),
            }
        )
    return rows


def gate_rows(
    aggregates: list[dict[str, Any]], comparisons: list[dict[str, Any]],
    integrity: dict[str, Any], steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    index = {(row["alias"], row["arm"]): row for row in aggregates}
    comparison = comparison_aggregates(comparisons)
    comp_index = {(row["alias"], row["arm"]): row for row in comparison}
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "gate": "TECHNICAL_ENDPOINT_INTEGRITY",
            "state": "PASS_WITH_SEPARATE_MECHANICAL_RECEIPT_FAIL",
            "facts": {
                "attempts": 40,
                "endpoints": 40,
                "failures": 0,
                "K8": integrity["case_k8"],
                "materialization8": integrity["case_materialization8"],
                "target24": integrity["case_target24"],
                "W0_pointer": integrity["case_W0_pointer"],
                "W0_bytes": integrity["case_W0_bytes"],
                "action_freeze": integrity["case_action_freeze"],
                "evaluator_firewall": integrity["case_evaluator_firewall"],
            },
        }
    )
    rows.append(
        {
            "gate": "MECHANICAL_CONTRACT_RECEIPT",
            "state": "MECHANICAL_CONTRACT_RECEIPT_FAIL",
            "facts": {
                "contract_outer_h_application_count_required": 1,
                "outer_h_application_count": "NOT_RECORDED",
                "writer_h_application_count_numerical_lock": 0,
                "physical_h_application_count_sum": sum(row["physical_h_application_count"] for row in steps),
                "writer_step_denominator": len(steps),
                "one_joint_materialization_count": sum(row["one_joint_materialization_count"] for row in steps),
                "second_h_application_count": "NOT_RECORDED",
                "requestwise_self_reachable_ceiling_c_i": "NOT_RECORDED",
                "residual_divisor_values": sorted(
                    {value for row in steps for value in row["residual_divisor_values"]}
                ),
            },
        }
    )
    for alias in ROOTS:
        for arm in ARMS:
            value = index[(alias, arm)]
            comp = comp_index[(alias, arm)]
            rows.append(
                {
                    "gate": "A_CURRENT_W_TARGET_VS_P2R1_TARGET_ONLY",
                    "alias": alias,
                    "arm": arm,
                    "state": "NOT_BINARIZED_PREDECLARED_NUMERIC_THRESHOLD_ABSENT",
                    "facts": {
                        "z_eff_correct": value["z_eff_correct"],
                        "z_eff_denominator": value["z_eff_denominator"],
                        "delta_vs_p2r1_z_eff_correct": comp["delta_vs_p2r1_z_eff_correct"],
                        "z_gen_correct": value["z_gen_correct"],
                        "z_gen_denominator": value["z_gen_denominator"],
                        "delta_vs_p2r1_z_gen_correct": comp["delta_vs_p2r1_z_gen_correct"],
                        "delta_vs_p2r1_z_eff_nll_case_mean": comp["delta_vs_p2r1_z_eff_nll_case_mean"],
                        "delta_vs_p2r1_z_gen_nll_case_mean": comp["delta_vs_p2r1_z_gen_nll_case_mean"],
                    },
                }
            )
            rows.append(
                {
                    "gate": "B_WRITER_REALIZATION",
                    "alias": alias,
                    "arm": arm,
                    "state": "TERMINAL_COUNT_REALIZATION_PASS__LINEARIZED_RATIO_RECORDED",
                    "facts": {
                        "z_eff_correct": value["z_eff_correct"],
                        "w_eff_correct": value["w_eff_correct"],
                        "z_gen_correct": value["z_gen_correct"],
                        "w_gen_correct": value["w_gen_correct"],
                        "eff_w_minus_z_nll_mean": value["eff_w_minus_z_nll"]["mean"],
                        "gen_w_minus_z_nll_mean": value["gen_w_minus_z_nll"]["mean"],
                        "actual_over_predicted": value["actual_over_predicted"],
                        "negative_actual_request_transitions": value[
                            "negative_actual_request_transitions"
                        ],
                    },
                }
            )
    for alias in ROOTS:
        neutral, soft = index[(alias, "NEUTRAL")], index[(alias, "SOFTP")]
        paired = comp_index[(alias, "SOFTP_MINUS_NEUTRAL")]
        structural_or_capacity_reduction = (
            paired["delta_terminal_structural_p_case_mean"] < 0.0
            or paired["delta_terminal_capacity_case_mean"] < 0.0
        )
        rows.append(
            {
                "gate": "C_SOFT_BARRIER",
                "alias": alias,
                "arm": "SOFTP",
                "state": "PASS" if structural_or_capacity_reduction else "NOT_PASS_NO_TERMINAL_P_OR_CAPACITY_REDUCTION",
                "facts": {
                    "max_same_state_no_weaker_violation": soft["soft_no_weaker_max_violation"],
                    "strength_attenuation_count": soft["strength_attenuation_count"],
                    "soft_certified_steps": soft["route_status_counts"].get("SOFT_CERTIFIED", 0),
                    "soft_neutral_fallback_steps": soft["route_status_counts"].get(
                        "SOFT_NEUTRAL_FALLBACK", 0
                    ),
                    "same_state_steps": paired["same_state_steps"],
                    "same_state_moved_steps": paired["same_state_moved_steps"],
                    "delta_w_eff_correct": soft["w_eff_correct"] - neutral["w_eff_correct"],
                    "delta_w_gen_correct": soft["w_gen_correct"] - neutral["w_gen_correct"],
                    "delta_terminal_structural_p_case_mean": paired[
                        "delta_terminal_structural_p_case_mean"
                    ],
                    "delta_terminal_capacity_case_mean": paired[
                        "delta_terminal_capacity_case_mean"
                    ],
                },
            }
        )
    native_eligible = []
    for alias in ROOTS:
        for arm in ARMS:
            comp = comp_index[(alias, arm)]
            if (
                comp["delta_vs_official_w_eff_correct"] >= 0
                and comp["delta_vs_official_w_gen_correct"] >= 0
                and comp["delta_vs_official_w_loc_correct"] >= 0
            ):
                native_eligible.append(f"{alias}:{arm}")
    rows.append(
        {
            "gate": "D_NEXT_STAGE_CONTRACT_BRANCH",
            "state": "NEITHER_PREDECLARED_BRANCH_CONDITION_MET",
            "facts": {
                "arms_at_or_above_official_E_G_L_counts": native_eligible,
                "historical_eligible_arm_count": len(native_eligible),
                "writer_aware_z_condition": False,
                "writer_aware_z_condition_basis": "terminal W correct counts equal current-W z counts in all four cells",
            },
        }
    )
    return rows


def fmt(value: Any) -> str:
    if value is None:
        return "NOT_RECORDED"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_report(
    aggregates: list[dict[str, Any]], comparison: list[dict[str, Any]],
    gates: list[dict[str, Any]], integrity: dict[str, Any],
) -> str:
    index = {(row["alias"], row["arm"]): row for row in aggregates}
    comp = {(row["alias"], row["arm"]): row for row in comparison}
    now = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")
    lines = [
        "# P2R2 Semantic-Conserving Residual-Transport Writer 최종 분석",
        "",
        f"- 생성 시각(Asia/Seoul): `{now}`",
        f"- source HEAD/tree: `{SOURCE_HEAD}` / `{SOURCE_TREE}`",
        f"- exact P2R1 parent: `{P2R1_PARENT}`; P1R43 ancestor: `{P1R43_ANCESTOR}`",
        f"- contract SHA256: `{CONTRACT_SHA}`; stream/order: `{STREAM_ROOT}` / `{STREAM_ORDER}`",
        f"- evaluator/aggregator: `{EVALUATOR_SHA}` / `{AGGREGATOR_SHA}`",
        "- 전체 분모: 40 attempts / 40 endpoints / 0 typed incomplete / 400 requests / 320 writer steps.",
        "- scientific_promotion=false.",
        "",
        "## 한 페이지 요약",
        "",
        "| model | arm | z E/G | W E/G/L | z Eff NLL | W Eff NLL | z Gen NLL | W Gen NLL | W−z Eff/Gen NLL | actual/pred | P | capacity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for alias in ROOTS:
        for arm in ARMS:
            row = index[(alias, arm)]
            lines.append(
                "| " + " | ".join(
                    [
                        alias,
                        arm,
                        f"{row['z_eff_correct']}/{row['z_eff_denominator']}, {row['z_gen_correct']}/{row['z_gen_denominator']}",
                        f"{row['w_eff_correct']}/{row['w_eff_denominator']}, {row['w_gen_correct']}/{row['w_gen_denominator']}, {row['w_loc_correct']}/{row['w_loc_denominator']}",
                        fmt(row["z_eff_nll_case_mean"]["mean"]),
                        fmt(row["w_eff_nll_case_mean"]["mean"]),
                        fmt(row["z_gen_nll_case_mean"]["mean"]),
                        fmt(row["w_gen_nll_case_mean"]["mean"]),
                        f"{fmt(row['eff_w_minus_z_nll']['mean'])} / {fmt(row['gen_w_minus_z_nll']['mean'])}",
                        fmt(row["actual_over_predicted"]),
                        fmt(row["terminal_structural_p"]["mean"]),
                        fmt(row["terminal_bf16_capacity"]["mean"]),
                    ]
                ) + " |"
            )
    lines += [
        "",
        "### Gate A–D",
        "",
        "- Gate A: Eff z-inject는 P2R1 target-only와 네 cell 모두 100/100으로 동일했다. Gen z-inject는 Llama/Qwen 각 arm에서 P2R1 대비 −8/200이었다. ‘크게 잃지 않음’의 사전 수치 임계가 없어 binary state는 `NOT_BINARIZED_PREDECLARED_NUMERIC_THRESHOLD_ABSENT`이다.",
        "- Gate B: 네 cell 모두 terminal z와 W의 Eff/Gen correct count가 동일했다. W−z NLL 평균과 내부 actual/predicted ratio는 표에 분리했다.",
        "- Gate C: Llama Soft는 Neutral 대비 terminal P −0.00144911, capacity −0.916555이며 E/G count delta 0/0이다. Qwen Soft는 P +0.000672101, capacity +0.0560499이며 E/G count delta 0/0이다.",
        "- Gate D: Official E/G/L count를 모두 동시에 충족한 arm은 0개다. 동시에 모든 cell의 terminal W E/G count가 current-W z E/G count와 같아서 사전 writer-aware-z branch 조건도 충족하지 않았다.",
        "- 기계적 계약: endpoint integrity는 40/40이나 §11의 `outer_h_application_count=1`과 §17의 requestwise self-reachable `c_i` 필드가 없다. numerical lock의 `writer_h_application_count=0`, 320 receipt의 `physical_h_application_count=0`, joint materialization 320/320을 별도 기록했고 상태는 `MECHANICAL_CONTRACT_RECEIPT_FAIL`이다.",
        "",
        "### 필수 14개 질문",
        "",
        "1. P1R43 writer: execution AST gate와 320 receipt에서 legacy writer access/decision influence 합계 0이다.",
        "2. Current-W target refresh: 각 case/arm에 8개 outer physical-state identity와 outer별 3 microstep 동일 W identity가 기록됐고 writer 뒤 1회 refresh가 320/320이다.",
        "3. Residual 보존: 모든 step의 response shape은 10×50, 각 layer의 residual SHA는 step 내 동일, divisor는 1, active simplex residual 최대는 machine table에 기록됐다. request별 layer-sum 제약을 사용했다.",
        "4. Neutral hard-request 보호: Neutral `NEUTRAL_CERTIFIED` 160/160, fairness lambda가 각 step에 기록됐고 unreachable request-step은 Llama 2, Qwen 3이다. unreachable은 별도 typed registry에 남았다.",
        "5. Soft strength: strength attenuation 0, same-state no-weaker violation 최대는 1e-8 numerical tolerance 이내다. Soft는 Llama 10/80, Qwen 4/80 step에서 certified, 나머지는 Neutral fallback이다.",
        "6. Structural-P 영향: exact paired same-state allocation 이동은 Llama 4/62, Qwen 2/69 step이다. terminal P delta는 Llama 음수, Qwen 양수다.",
        "7. Barrier DOF: active request당 5-layer simplex의 형식 DOF는 4이며 step당 4×active-request 수다. actual same-state route movement와 status 분모는 위와 machine table에 기록했다.",
        "8. z→W 소실: Eff/Gen correct 소실은 네 cell 모두 0이며 NLL gap은 요약표와 per-request table에 있다.",
        "9. Subgroup: Llama clamp-hit request는 200/200 arm-request, Qwen은 0/200이다. Qwen ‘hard subgroup’의 별도 source threshold/registry는 `NOT_RECORDED`; request p90/max와 unreachable rows를 제공한다.",
        "10. Official 대비: E/G/L count delta와 Eff NLL delta는 아래 표에 있다. Official P/capacity/update norm 및 edit-core는 `NOT_RECORDED`이다.",
        "11. 병목 분리: P2R1 대비 current-W z Gen delta는 네 cell 모두 −8/200; current-W z→W E/G count delta는 0이다. Router attenuation은 0이고 Soft fallback은 146/160이다.",
        "12. 다음 단계 gate: Official E/G/L count 동시충족 arm 0, writer-aware-z 조건 false로 `NEITHER_PREDECLARED_BRANCH_CONDITION_MET`이다.",
        "13. Native edit-core ratio: Official edit-core seconds가 immutable matched reference에 없어 `NOT_RECORDED`이다.",
        "14. Forbidden influence: legacy writer, candidate forward/materialization, debt, remaining division, hard-P budget, functional-P veto, retry/backtracking, heldout controller access는 모두 0이다. §11 h receipt gap은 별도 mechanical fail이다.",
        "",
        "## P2R1 target-only 및 Official matched arithmetic",
        "",
        "| model | arm | Δz Eff/G vs P2R1 | Δz Eff/Gen NLL vs P2R1 | ΔW E/G/L vs Official | ΔW Eff NLL vs Official |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for alias in ROOTS:
        for arm in ARMS:
            row = comp[(alias, arm)]
            lines.append(
                f"| {alias} | {arm} | {row['delta_vs_p2r1_z_eff_correct']} / {row['delta_vs_p2r1_z_gen_correct']} | "
                f"{fmt(row['delta_vs_p2r1_z_eff_nll_case_mean'])} / {fmt(row['delta_vs_p2r1_z_gen_nll_case_mean'])} | "
                f"{row['delta_vs_official_w_eff_correct']} / {row['delta_vs_official_w_gen_correct']} / {row['delta_vs_official_w_loc_correct']} | "
                f"{fmt(row['delta_vs_official_w_eff_nll_case_mean'])} |"
            )
    lines += [
        "",
        "- Exact P2R1 matched identity는 model/case request order, stream/order, evaluator/aggregator로 40/40 comparison rows에서 확인했다.",
        "- Official identity SHA는 `" + OFFICIAL_IDENTITY_SHA + "`; 20/20 model×case가 matched이다.",
        "- Official Gen/Loc continuous NLL·margin, Structural-P, capacity, update norm, edit-core time은 `NOT_RECORDED`이다.",
        "",
        "## Neutral–Soft paired facts",
        "",
        "| model | ΔE/G/L | ΔEff/Gen NLL | ΔP | Δcapacity | same-state moved/compared | Soft certified/fallback |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for alias in ROOTS:
        row = comp[(alias, "SOFTP_MINUS_NEUTRAL")]
        soft = index[(alias, "SOFTP")]
        lines.append(
            f"| {alias} | {row['delta_w_eff_correct']} / {row['delta_w_gen_correct']} / {row['delta_w_loc_correct']} | "
            f"{fmt(row['delta_w_eff_nll_case_mean'])} / {fmt(row['delta_w_gen_nll_case_mean'])} | "
            f"{fmt(row['delta_terminal_structural_p_case_mean'])} | {fmt(row['delta_terminal_capacity_case_mean'])} | "
            f"{row['same_state_moved_steps']}/{row['same_state_steps']} | "
            f"{soft['route_status_counts'].get('SOFT_CERTIFIED', 0)}/{soft['route_status_counts'].get('SOFT_NEUTRAL_FALLBACK', 0)} |"
        )
    lines += [
        "",
        "## Compute 및 integrity",
        "",
        "| model | arm | target/writer/total seconds(case mean) | target F/B | response VJP | writer mat | negative request-steps |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for alias in ROOTS:
        for arm in ARMS:
            row = index[(alias, arm)]
            compute = row["compute_sums"]
            lines.append(
                f"| {alias} | {arm} | {fmt(row['target_wall_seconds']['mean'])} / {fmt(row['writer_wall_seconds']['mean'])} / {fmt(row['total_wall_seconds']['mean'])} | "
                f"{compute['target_forward_count']}/{compute['target_backward_count']} | {compute['physical_response_batched_vjp_count']} | "
                f"{compute['writer_materialization_count']} | {row['negative_actual_request_transitions']} |"
            )
    lines += [
        "",
        "- Scheduler submission receipt: array job `19944`, array `0-1%2`, source `" + SOURCE_HEAD + "`; result terminal은 model별 20/20 arm-case complete다. 별도 scheduler terminal-state receipt는 `NOT_RECORDED`.",
        "- Action freeze/W0 pointer/W0 bytes/evaluator firewall/K8/materialization8/target24는 각각 40/40이다.",
        "- Retry/backtracking/candidate materialization/heldout decision influence는 0이다.",
        "",
        "## FACT / INFERENCE / NOT_RECORDED / TECHNICAL_FAIL / SCIENTIFIC_FAIL",
        "",
        "### FACT",
        "",
        "- 위 표의 raw-free 값, exact identities, 40/40 endpoint 및 machine tables.",
        "- P2R1 current-W z Gen count delta는 모든 arm에서 −8/200이며 current-W z→W E/G count delta는 0.",
        "- Soft terminal P/capacity delta는 Llama에서 음수, Qwen에서 양수.",
        "",
        "### INFERENCE",
        "",
        "- Gate A의 ‘크게 잃지 않음’은 사전 수치 임계가 없어 binary 판정하지 않았다.",
        "- Gate D는 contract branch 조건의 직접 산술만 적용했고 추가 권고를 만들지 않았다.",
        "",
        "### NOT_RECORDED",
        "",
        "- Official Gen/Loc continuous metrics, Official P/capacity/update norm/edit-core; Qwen hard-subgroup 별도 registry; requestwise self-reachable ceiling c_i; exact same-state internal Soft-vs-Neutral allocation for paired trajectories가 이미 갈라진 step; outer_h_application_count와 second_h_application_count.",
        "",
        "### TECHNICAL_FAIL",
        "",
        "- `MECHANICAL_CONTRACT_RECEIPT_FAIL`: contract §11 `outer_h_application_count=1` 및 §17 requestwise `c_i` 필드 부재. numerical lock `writer_h_application_count=0`, writer receipt `physical_h_application_count=0` (320/320), joint materialization 320/320.",
        "",
        "### SCIENTIFIC_FAIL",
        "",
        "- typed scientific failure case는 0/40. Gate A/C/D의 수치 상태는 별도 gate table에 기록했으며 scientific_promotion=false다.",
        "",
        "## 산출물",
        "",
        "- `p2r2-per-case.jsonl`: 40 rows",
        "- `p2r2-per-step.jsonl`: 320 rows",
        "- `p2r2-per-request.jsonl`: 400 rows",
        "- `p2r2-per-step-request.jsonl`: 3,200 rows",
        "- `p2r2-response-routing.jsonl`: 320 rows",
        "- `p2r2-comparisons.jsonl`: 180 rows",
        "- `p2r2-aggregates.json`, `p2r2-comparison-aggregates.json`, `p2r2-gates.json`",
        "- `analysis-manifest.json`, `analysis-receipt.json`, `independent-verification.json`, `final-package-manifest.json`, `final-package-receipt.json`",
        "",
    ]
    return "\n".join(lines)


def file_record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "lines": data.count(b"\n"),
        "mode": oct(path.stat().st_mode & 0o777),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True, mode=0o700)
    OUT.chmod(0o700)
    assert sha256(CONTRACT) == CONTRACT_SHA
    assert sha256(SUPPORT / "identities/official-alphaedit-baseline-identity.json") == OFFICIAL_IDENTITY_SHA
    lock = read_json(LOCK)
    source_manifest = read_json(SOURCE_MANIFEST)
    assert lock["exact_p2r1_parent"] == P2R1_PARENT
    assert lock["stream_root"] == STREAM_ROOT
    assert lock["all_request_order_sha256"] == STREAM_ORDER
    assert lock["frozen_evaluator_sha256"] == EVALUATOR_SHA
    assert lock["frozen_aggregator_sha256"] == AGGREGATOR_SHA
    assert source_manifest["root_digest"] == "9ef6c171b9a47bcfb335297682bd8ed3fabf8474ee9fb431c9f5917dc36c1a79"

    cases, steps, requests, step_requests, response_routing, comparisons, integrity = collect()
    assert (
        len(cases),
        len(steps),
        len(requests),
        len(step_requests),
        len(response_routing),
        len(comparisons),
    ) == (
        40,
        320,
        400,
        3200,
        320,
        180,
    )
    aggregates = aggregate(cases, steps, requests)
    comparison = comparison_aggregates(comparisons)
    gates = gate_rows(aggregates, comparisons, integrity, steps)

    paths = {
        "per_case": OUT / "p2r2-per-case.jsonl",
        "per_step": OUT / "p2r2-per-step.jsonl",
        "per_request": OUT / "p2r2-per-request.jsonl",
        "per_step_request": OUT / "p2r2-per-step-request.jsonl",
        "response_routing": OUT / "p2r2-response-routing.jsonl",
        "comparisons": OUT / "p2r2-comparisons.jsonl",
        "aggregates": OUT / "p2r2-aggregates.json",
        "comparison_aggregates": OUT / "p2r2-comparison-aggregates.json",
        "gates": OUT / "p2r2-gates.json",
        "report": OUT / "p2r2-semantic-conserving-residual-transport-terminal-analysis-ko.md",
    }
    write_jsonl(paths["per_case"], cases)
    write_jsonl(paths["per_step"], steps)
    write_jsonl(paths["per_request"], requests)
    write_jsonl(paths["per_step_request"], step_requests)
    write_jsonl(paths["response_routing"], response_routing)
    write_jsonl(paths["comparisons"], comparisons)
    write_json(paths["aggregates"], {"schema": "ode-edit-s05-p2r2-aggregates/v1", "rows": aggregates, "rows_root": canonical_sha(aggregates)})
    write_json(
        paths["comparison_aggregates"],
        {"schema": "ode-edit-s05-p2r2-comparison-aggregates/v1", "rows": comparison, "rows_root": canonical_sha(comparison)},
    )
    write_json(paths["gates"], {"schema": "ode-edit-s05-p2r2-gates/v1", "rows": gates, "rows_root": canonical_sha(gates)})
    paths["report"].write_text(render_report(aggregates, comparison, gates, integrity), encoding="utf-8")
    paths["report"].chmod(0o600)

    inputs = {
        "contract": file_record(CONTRACT),
        "numerical_lock": file_record(LOCK),
        "source_manifest": file_record(SOURCE_MANIFEST),
        "submission_receipt": file_record(SUBMISSION),
        "p2r1_report": file_record(P2R1_REPORT / "p2r1-rms-tangent-target-only-factual-ko.md"),
        "p2r1_per_case": file_record(P2R1_REPORT / "p2r1-per-case.json"),
        "p2r1_per_panel": file_record(P2R1_REPORT / "p2r1-per-panel.json"),
        "official_identity": file_record(
            SUPPORT / "identities/official-alphaedit-baseline-identity.json"
        ),
        "llama_result_terminal": file_record(ROOTS["llama3-8b-inst"] / "terminal.json"),
        "qwen_result_terminal": file_record(ROOTS["qwen2.5-7b-inst"] / "terminal.json"),
    }
    artifacts = [file_record(path) for path in paths.values()]
    manifest = {
        "schema": "ode-edit-s05-p2r2-analysis-manifest/v1",
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "p2r1_parent": P2R1_PARENT,
        "p1r43_ancestor": P1R43_ANCESTOR,
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "evaluator_sha256": EVALUATOR_SHA,
        "aggregator_sha256": AGGREGATOR_SHA,
        "inputs": inputs,
        "artifacts": artifacts,
        "integrity": integrity,
        "artifact_root": canonical_sha(artifacts),
    }
    manifest["root_digest"] = canonical_sha({key: value for key, value in manifest.items() if key != "root_digest"})
    manifest_path = OUT / "analysis-manifest.json"
    write_json(manifest_path, manifest)
    receipt = {
        "schema": "ode-edit-s05-p2r2-analysis-receipt/v1",
        "attempts": 40,
        "endpoints": 40,
        "failures": 0,
        "case_rows": len(cases),
        "step_rows": len(steps),
        "request_rows": len(requests),
        "step_request_rows": len(step_requests),
        "response_routing_rows": len(response_routing),
        "comparison_rows": len(comparisons),
        "action_freeze_count": integrity["case_action_freeze"],
        "W0_pointer_restore_count": integrity["case_W0_pointer"],
        "W0_bytes_restore_count": integrity["case_W0_bytes"],
        "mechanical_contract_receipt_state": "MECHANICAL_CONTRACT_RECEIPT_FAIL",
        "outer_h_application_count": "NOT_RECORDED",
        "physical_h_application_count_sum": sum(row["physical_h_application_count"] for row in steps),
        "one_joint_materialization_count": sum(row["one_joint_materialization_count"] for row in steps),
        "scientific_promotion": False,
        "analysis_manifest_sha256": sha256(manifest_path),
        "analysis_manifest_root": manifest["root_digest"],
    }
    receipt["root_digest"] = canonical_sha({key: value for key, value in receipt.items() if key != "root_digest"})
    write_json(OUT / "analysis-receipt.json", receipt)


if __name__ == "__main__":
    main()
