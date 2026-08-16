#!/usr/bin/env python3
"""Build raw-free factual P2R4 Phase-B terminal tables and report.

The input allowlist is restricted to JSON receipts under the four completed
P2R4 Phase-B result roots.  It deliberately does not open tensors, prompts,
targets, generations, weights, or scheduler/runtime logs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "local/odebf/results"
STATE = REPO / "local/odebf/state/p2r4-phaseb-clamp-on-off-writer"
OUT = Path(__file__).resolve().parent
REPORT_STEM = "p2r4-phaseb-clamp-on-off-writer-v1"
ARMS = (("NEUTRAL", "neutral"), ("SOFTP", "softp"))
METRIC_LABELS = (("efficacy", "eff"), ("generalization", "gen"), ("locality-preservation", "loc"))
RESULT_ROOTS = (
    RESULTS / "s05-p2r4-phaseb-b10x10-llama3-8b-inst-clamp-on-p2r2-v2-paired-server2-node-r1-v1",
    RESULTS / "s05-p2r4-phaseb-b10x10-llama3-8b-inst-clamp-off-p2r2-v2-paired-server2-node-r1-v1",
    RESULTS / "s05-p2r4-phaseb-b10x10-qwen2.5-7b-inst-clamp-on-p2r2-v2-paired-server2-node-r1-v1",
    RESULTS / "s05-p2r4-phaseb-b10x10-qwen2.5-7b-inst-clamp-off-p2r2-v2-paired-server2-node-r1-v1",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"nonfinite {label}: {number}")
    return number


def floats(values: Any, expected: int, label: str) -> list[float]:
    if not isinstance(values, list) or len(values) != expected:
        raise ValueError(f"{label} length differs")
    return [finite(value, label) for value in values]


def bools(values: Any, expected: int, label: str) -> list[bool]:
    if not isinstance(values, list) or len(values) != expected:
        raise ValueError(f"{label} length differs")
    if not all(isinstance(value, bool) for value in values):
        raise ValueError(f"{label} element type differs")
    return list(values)


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return None if not values else float(sum(values) / len(values))


def quantile(values: Iterable[float], q: float) -> float | None:
    values = sorted(float(value) for value in values)
    if not values:
        return None
    position = (len(values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def weighted_mean(rows: list[dict[str, Any]], value_key: str, weight_key: str) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = row.get(value_key)
        weight = row.get(weight_key)
        if value is None or weight is None:
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
    return None if denominator == 0.0 else numerator / denominator


def clean_number(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"nonfinite output: {value}")
        return value
    return value


def write_once(path: Path, payload: bytes) -> str:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def write_json_once(path: Path, value: Any) -> str:
    return write_once(path, canonical_bytes(value) + b"\n")


def write_csv_once(path: Path, rows: list[dict[str, Any]]) -> str:
    fields = sorted({key for row in rows for key in row})
    text_rows: list[dict[str, str]] = []
    for row in rows:
        text_rows.append({
            key: "" if row.get(key) is None else (
                json.dumps(row[key], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if isinstance(row[key], (list, dict)) else str(clean_number(row[key]))
            )
            for key in fields
        })
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(text_rows)
    return write_once(path, buffer.getvalue().encode("utf-8"))


def metric_payload(metrics: dict[str, Any], metric: str, prefix: str, out: dict[str, Any]) -> None:
    data = metrics[metric]
    out[f"{prefix}_{metric}_correct"] = int(data.get("correct_count", data.get("numerator")))
    out[f"{prefix}_{metric}_denominator"] = int(data.get("prompt_count", data.get("denominator")))
    out[f"{prefix}_{metric}_rate"] = finite(data.get("aggregate", out[f"{prefix}_{metric}_correct"] / out[f"{prefix}_{metric}_denominator"]), f"{prefix}_{metric}_rate")
    out[f"{prefix}_{metric}_mean_target_new_nll"] = finite(data.get("mean_target_new_nll"), f"{prefix}_{metric}_new_nll") if "mean_target_new_nll" in data else None
    out[f"{prefix}_{metric}_mean_margin"] = finite(data.get("mean_margin"), f"{prefix}_{metric}_margin") if "mean_margin" in data else None
    out[f"{prefix}_{metric}_per_request_correct"] = list(data.get("per_case_correct", []))
    out[f"{prefix}_{metric}_per_request_required"] = list(data.get("per_case_required", []))
    out[f"{prefix}_{metric}_per_request_new_nll"] = list(data.get("per_case_mean_target_new_nll", []))
    out[f"{prefix}_{metric}_per_request_margin"] = list(data.get("per_case_mean_margin", []))


def get_w_metrics(terminal: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    metrics = terminal["terminal_w_panel"]["receipt"]["metrics"]
    for metric, _ in METRIC_LABELS:
        metric_payload(metrics, metric, "w", payload)
    return payload


def get_z_metrics(terminal: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    z_panel = terminal["terminal_z_panel"]["z_inject"]
    metrics = z_panel["primary"]["metrics"]
    numeric_vectors = z_panel["numeric_vectors"]
    for metric, _ in METRIC_LABELS:
        metric_payload(metrics, metric, "z", payload)
        request_vectors = numeric_vectors[metric]
        if not isinstance(request_vectors, list) or len(request_vectors) != 10:
            raise ValueError(f"z numeric vector cardinality differs: {metric}")
        per_request_new_nll: list[float] = []
        per_request_margin: list[float] = []
        all_new_nll: list[float] = []
        all_margin: list[float] = []
        for request_index, prompts in enumerate(request_vectors):
            if not isinstance(prompts, list) or not prompts:
                raise ValueError(f"z numeric vector prompts differ: {metric}/{request_index}")
            new_nll = [finite(value["nll_new"], f"z {metric} nll") for value in prompts]
            margin = [finite(value["margin"], f"z {metric} margin") for value in prompts]
            per_request_new_nll.append(float(sum(new_nll) / len(new_nll)))
            per_request_margin.append(float(sum(margin) / len(margin)))
            all_new_nll.extend(new_nll)
            all_margin.extend(margin)
        payload[f"z_{metric}_per_request_new_nll"] = per_request_new_nll
        payload[f"z_{metric}_per_request_margin"] = per_request_margin
        payload[f"z_{metric}_mean_target_new_nll"] = float(sum(all_new_nll) / len(all_new_nll))
        payload[f"z_{metric}_mean_margin"] = float(sum(all_margin) / len(all_margin))
    return payload


def collect_root(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = root / "manifest.json"
    job_terminal_path = root / "terminal.json"
    manifest = load_json(manifest_path)
    job_terminal = load_json(job_terminal_path)
    if sha256_file(job_terminal_path) != manifest["terminal_sha256"]:
        raise ValueError(f"top terminal SHA differs: {root}")
    if job_terminal["completed_case_arm_count"] != 20 or job_terminal["failed_case_arm_count"] != 0:
        raise ValueError(f"job terminal completion differs: {root}")
    if not job_terminal["W0_restored"] or not manifest["W0_restored"]:
        raise ValueError(f"top W0 receipt differs: {root}")
    alias = str(job_terminal["alias"])
    policy = str(job_terminal["clamp_policy"])
    case_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for arm, folder in ARMS:
        for case_index in range(1, 11):
            case_root = root / "raw/cases" / f"case-{case_index:02d}" / folder
            case_manifest_path = case_root / "manifest.json"
            case_terminal_path = case_root / "terminal.json"
            case_manifest = load_json(case_manifest_path)
            terminal = load_json(case_terminal_path)
            if sha256_file(case_terminal_path) != case_manifest["terminal_sha256"]:
                raise ValueError(f"case terminal SHA differs: {case_terminal_path}")
            if terminal["alias"] != alias or terminal["clamp_policy"] != policy or terminal["arm"] != arm:
                raise ValueError(f"case identity differs: {case_terminal_path}")
            if not terminal["W0_restored"] or not terminal["W0_restore"]["byte_restored_exact"] or not terminal["W0_restore"]["pointer_restored_exact"]:
                raise ValueError(f"W0 restoration differs: {case_terminal_path}")
            if terminal["target_microstep_count"] != 24 or terminal["writer_transition_count"] != 8 or terminal["writer_materialization_count"] != 8:
                raise ValueError(f"step/materialization cardinality differs: {case_terminal_path}")
            if terminal["request_count"] != 10:
                raise ValueError(f"request count differs: {case_terminal_path}")
            action = load_json(case_root / "action-freeze.json")
            if not action["actions_frozen_before_evaluator"] or action["heldout_controller_access_count"] != 0:
                raise ValueError(f"action freeze differs: {case_root}")
            if action["writer_materialization_count"] != 8 or action["writer_transition_count"] != 8:
                raise ValueError(f"action writer count differs: {case_root}")
            w = get_w_metrics(terminal)
            z = get_z_metrics(terminal)
            target_receipts: list[dict[str, Any]] = []
            target_path_by_request = [0.0] * 10
            target_event_by_request = [0] * 10
            target_event_label = "clamp_hit" if policy == "ON" else "clamp_would_hit"
            final_off_net: list[float] | None = None
            for microstep in range(24):
                receipt = load_json(case_root / "raw/target" / f"microstep-{microstep:02d}.json")
                if receipt["request_count"] != 10 or receipt["phase_b_clamp_policy"] != policy:
                    raise ValueError(f"target receipt identity differs: {case_root} {microstep}")
                if receipt["retry_count"] != 0 or receipt["early_stop_count"] != 0 or receipt["heldout_controller_access_count"] != 0:
                    raise ValueError(f"target forbidden count differs: {case_root} {microstep}")
                before = floats(receipt["pre_clamp_displacement_norm"], 10, "pre displacement")
                after = floats(receipt["post_clamp_displacement_norm"], 10, "post displacement")
                events = bools(receipt[target_event_label], 10, target_event_label)
                if policy == "ON":
                    if int(receipt["clamp_hit_count"]) != sum(events):
                        raise ValueError(f"ON hit count differs: {case_root} {microstep}")
                else:
                    if int(receipt["clamp_would_hit_count"]) != sum(events):
                        raise ValueError(f"OFF would-hit count differs: {case_root} {microstep}")
                    if int(receipt["identity_projection_application_count"]) != 1 or int(receipt["clamp_application_count"]) != 0 or int(receipt["clamp_decision_influence_count"]) != 0:
                        raise ValueError(f"OFF identity clamp policy differs: {case_root} {microstep}")
                target_next_finite = receipt.get("target_next_finite")
                if target_next_finite is not None and not bool(target_next_finite):
                    raise ValueError(f"nonfinite target state: {case_root} {microstep}")
                for ordinal in range(10):
                    target_path_by_request[ordinal] += after[ordinal]
                    target_event_by_request[ordinal] += int(events[ordinal])
                    target_rows.append({
                        "alias": alias,
                        "clamp_policy": policy,
                        "arm": arm,
                        "case_index": case_index,
                        "request_ordinal": ordinal + 1,
                        "microstep_index": microstep,
                        "outer_step": int(receipt["outer_step"]),
                        "within_outer_microstep": int(receipt["within_outer_microstep"]),
                        "event_kind": target_event_label,
                        "clamp_event": events[ordinal],
                        "pre_clamp_displacement_norm": before[ordinal],
                        "post_clamp_displacement_norm": after[ordinal],
                        "target_new_nll": floats(receipt["target_new_nll_by_request"], 10, "target nll")[ordinal],
                        "semantic_gradient_norm": floats(receipt["semantic_gradient_norm"], 10, "semantic gradient norm")[ordinal],
                        "preservation_gradient_norm": floats(receipt["preservation_gradient_norm"], 10, "preservation gradient norm")[ordinal],
                        "gamma": floats(receipt["gamma_by_request"], 10, "gamma")[ordinal],
                        "q": floats(receipt["q_by_request"], 10, "q")[ordinal],
                        "clamp_ratio": floats(receipt["clamp_ratio"], 10, "clamp ratio")[ordinal],
                        "target_next_finite": None if target_next_finite is None else bool(target_next_finite),
                        "target_next_max_abs": None if "target_next_max_abs" not in receipt else finite(receipt["target_next_max_abs"], "target next max abs"),
                        "target_receipt_identity_sha256": str(receipt["identity_sha256"]),
                    })
                if microstep == 23 and "target_next_origin_displacement_norm" in receipt:
                    final_off_net = floats(receipt["target_next_origin_displacement_norm"], 10, "target net norm")
                target_receipts.append(receipt)
            writer_payloads: list[dict[str, Any]] = []
            layer_sum = [0.0] * 5
            layer_total = 0.0
            route_status = Counter()
            fallback = Counter()
            unreachable_set: set[int] = set()
            writer_energy_path = 0.0
            final_capacity = None
            negative_total = 0
            adapter_counts = Counter()
            realization_values: list[float] = []
            for outer in range(8):
                receipt = load_json(case_root / "raw/writer" / f"step-{outer:02d}.json")
                if receipt["outer_step"] != outer or receipt["clamp_policy"] != policy or receipt["arm"] != arm:
                    raise ValueError(f"writer receipt identity differs: {case_root} {outer}")
                expected_adapter = {
                    "outer_h_application_count": 1,
                    "applied_coordinate_update_count": 1,
                    "writer_h_numeric_multiplication_count": 0,
                    "physical_h_application_count": 0,
                    "second_h_application_count": 0,
                    "residual_presplit_count": 0,
                    "remaining_division_count": 0,
                    "semantic_debt_input_count": 0,
                }
                for key, expected in expected_adapter.items():
                    actual = int(receipt[key])
                    if actual != expected:
                        raise ValueError(f"adapter count differs {key}: {case_root} {outer}")
                    adapter_counts[key] += actual
                if int(receipt["one_joint_materialization_count"]) != 1 or int(receipt["retry_count"]) != 0 or int(receipt["backtracking_count"]) != 0:
                    raise ValueError(f"writer transition count differs: {case_root} {outer}")
                route = receipt["route"]
                if int(route["candidate_forward_count"]) != 0 or int(route["candidate_materialization_count"]) != 0 or int(route["functional_p_veto_count"]) != 0 or int(route["p_budget_influence_count"]) != 0 or int(route["strength_attenuation_count"]) != 0:
                    raise ValueError(f"route forbidden count differs: {case_root} {outer}")
                route_status[str(route["status"])] += 1
                fallback[str(route["fallback_reason"])] += 1
                active = bools(route["active_requests"], 10, "active requests")
                unreachable = {int(value) for value in route["unreachable_requests"]}
                if any(value < 0 or value >= 10 for value in unreachable):
                    raise ValueError(f"unreachable ordinal differs: {case_root} {outer}")
                unreachable_set |= unreachable
                allocation = route["allocation_by_layer_request"]
                if not isinstance(allocation, list) or len(allocation) != 5 or any(not isinstance(row, list) or len(row) != 10 for row in allocation):
                    raise ValueError(f"allocation shape differs: {case_root} {outer}")
                allocation = [[finite(value, "allocation") for value in row] for row in allocation]
                for layer in range(5):
                    layer_sum[layer] += sum(allocation[layer])
                layer_total += sum(sum(row) for row in allocation)
                current_nll = floats(receipt["current_nll_by_request"], 10, "writer current nll")
                next_nll = floats(receipt["next_nll_by_request"], 10, "writer next nll")
                predicted = floats(receipt["predicted_progress_by_request"], 10, "predicted progress")
                actual = floats(receipt["actual_progress_by_request"], 10, "actual progress")
                realization = floats(receipt["realization_by_request"], 10, "realization")
                residual_norm = floats(receipt["full_current_residual_norm_by_request"], 10, "residual norm")
                if int(receipt["negative_actual_count"]) != sum(value < 0.0 for value in actual):
                    raise ValueError(f"negative actual count differs: {case_root} {outer}")
                negative_total += int(receipt["negative_actual_count"])
                realization_values.extend(realization)
                materialization = receipt["materialization"]
                writer_energy_path += sum(finite(value, "BF16 step energy") for value in materialization["realized_bf16_step_energy"].values())
                final_capacity = sum(finite(value, "BF16 capacity") for value in materialization["cumulative_bf16_capacity"].values())
                pre_event = [0] * 10
                for target_receipt in target_receipts[outer * 3:(outer + 1) * 3]:
                    events = bools(target_receipt[target_event_label], 10, target_event_label)
                    for ordinal in range(10):
                        pre_event[ordinal] += int(events[ordinal])
                terminal_per_request = {}
                for metric, short in METRIC_LABELS:
                    for prefix in ("w", "z"):
                        correct = w if prefix == "w" else z
                        terminal_per_request[f"{prefix}_{short}_correct"] = correct[f"{prefix}_{metric}_per_request_correct"]
                        terminal_per_request[f"{prefix}_{short}_required"] = correct[f"{prefix}_{metric}_per_request_required"]
                        terminal_per_request[f"{prefix}_{short}_new_nll"] = correct[f"{prefix}_{metric}_per_request_new_nll"]
                        terminal_per_request[f"{prefix}_{short}_margin"] = correct[f"{prefix}_{metric}_per_request_margin"]
                for ordinal in range(10):
                    row = {
                        "alias": alias,
                        "clamp_policy": policy,
                        "arm": arm,
                        "case_index": case_index,
                        "outer_step": outer,
                        "request_ordinal": ordinal + 1,
                        "active": active[ordinal],
                        "unreachable": ordinal in unreachable,
                        "preceding_target_event_kind": target_event_label,
                        "preceding_target_event_count": pre_event[ordinal],
                        "full_current_residual_norm": residual_norm[ordinal],
                        "current_nll": current_nll[ordinal],
                        "next_nll": next_nll[ordinal],
                        "predicted_progress": predicted[ordinal],
                        "actual_progress": actual[ordinal],
                        "realization": realization[ordinal],
                        "allocation_layer_4": allocation[0][ordinal],
                        "allocation_layer_5": allocation[1][ordinal],
                        "allocation_layer_6": allocation[2][ordinal],
                        "allocation_layer_7": allocation[3][ordinal],
                        "allocation_layer_8": allocation[4][ordinal],
                        "allocation_sum": sum(layer[ordinal] for layer in allocation),
                        "terminal_full_six_target_new_nll": floats(terminal["terminal_full_six_target_new_nll_by_request"], 10, "terminal full six")[ordinal],
                    }
                    for key, values in terminal_per_request.items():
                        row[key] = values[ordinal] if values else None
                    request_rows.append(row)
                step_rows.append({
                    "alias": alias,
                    "clamp_policy": policy,
                    "arm": arm,
                    "case_index": case_index,
                    "outer_step": outer,
                    "route_status": str(route["status"]),
                    "fallback_reason": route["fallback_reason"],
                    "active_request_count": sum(active),
                    "unreachable_request_count": len(unreachable),
                    "predicted_progress_sum": sum(predicted),
                    "actual_progress_sum": sum(actual),
                    "realization_mean": mean(realization),
                    "negative_actual_count": int(receipt["negative_actual_count"]),
                    "cumulative_structural_p": finite(receipt["cumulative_structural_p"], "cumulative P"),
                    "marginal_structural_p": finite(route["marginal_structural_p"], "marginal P"),
                    "capacity": finite(route["capacity"], "route capacity"),
                    "bf16_step_energy_sum": sum(finite(value, "BF16 energy") for value in materialization["realized_bf16_step_energy"].values()),
                    "bf16_cumulative_capacity_sum": final_capacity,
                    "allocation_coefficient_share_layer_4": None if layer_total == 0.0 else sum(allocation[0]) / sum(sum(row) for row in allocation),
                    "allocation_coefficient_share_layer_5": None if layer_total == 0.0 else sum(allocation[1]) / sum(sum(row) for row in allocation),
                    "allocation_coefficient_share_layer_6": None if layer_total == 0.0 else sum(allocation[2]) / sum(sum(row) for row in allocation),
                    "allocation_coefficient_share_layer_7": None if layer_total == 0.0 else sum(allocation[3]) / sum(sum(row) for row in allocation),
                    "allocation_coefficient_share_layer_8": None if layer_total == 0.0 else sum(allocation[4]) / sum(sum(row) for row in allocation),
                    "soft_requestwise_no_weaker_max_violation": finite(route["soft_requestwise_no_weaker_max_violation"], "soft certificate"),
                    "request_simplex_max_abs_residual": finite(route["request_simplex_max_abs_residual"], "simplex residual"),
                    "outer_h_application_count": int(receipt["outer_h_application_count"]),
                    "applied_coordinate_update_count": int(receipt["applied_coordinate_update_count"]),
                    "writer_h_numeric_multiplication_count": int(receipt["writer_h_numeric_multiplication_count"]),
                    "physical_h_application_count": int(receipt["physical_h_application_count"]),
                    "second_h_application_count": int(receipt["second_h_application_count"]),
                    "residual_presplit_count": int(receipt["residual_presplit_count"]),
                    "remaining_division_count": int(receipt["remaining_division_count"]),
                    "semantic_debt_input_count": int(receipt["semantic_debt_input_count"]),
                    "writer_receipt_identity_sha256": str(receipt["identity_sha256"]),
                })
                writer_payloads.append(receipt)
            case_row: dict[str, Any] = {
                "alias": alias,
                "clamp_policy": policy,
                "arm": arm,
                "case_index": case_index,
                "request_count": 10,
                "request_order_sha256": str(terminal["request_order_sha256"]),
                "w0_restored": bool(terminal["W0_restored"]),
                "w0_byte_restored_exact": bool(terminal["W0_restore"]["byte_restored_exact"]),
                "w0_pointer_restored_exact": bool(terminal["W0_restore"]["pointer_restored_exact"]),
                "action_frozen_before_evaluator": bool(action["actions_frozen_before_evaluator"]),
                "heldout_controller_access_count": int(action["heldout_controller_access_count"]),
                "target_microstep_count": int(terminal["target_microstep_count"]),
                "writer_transition_count": int(terminal["writer_transition_count"]),
                "writer_materialization_count": int(terminal["writer_materialization_count"]),
                "terminal_full_six_target_new_nll": finite(terminal["terminal_full_six_target_new_nll"], "terminal full six"),
                "terminal_cumulative_structural_p": finite(terminal["terminal_cumulative_structural_p"], "terminal P"),
                "bf16_capacity_final_sum": final_capacity,
                "bf16_update_energy_path_sum": writer_energy_path,
                "writer_update_norm": None,
                "writer_path_norm": None,
                "writer_net_norm": None,
                "target_path_displacement_norm_mean": mean(target_path_by_request),
                "target_net_origin_displacement_norm_mean": mean(final_off_net) if final_off_net is not None else None,
                "target_event_kind": target_event_label,
                "target_event_occurrence_count": sum(target_event_by_request),
                "target_event_request_count": sum(value > 0 for value in target_event_by_request),
                "negative_actual_count": negative_total,
                "unreachable_event_count": sum(len(payload["route"]["unreachable_requests"]) for payload in writer_payloads),
                "unreachable_unique_request_count": len(unreachable_set),
                "writer_realization_mean": mean(realization_values),
                "writer_realization_median": quantile(realization_values, 0.5),
                "writer_realization_p90": quantile(realization_values, 0.9),
                "route_status_counts": dict(sorted(route_status.items())),
                "route_fallback_counts": dict(sorted(fallback.items())),
                "routing_dof": "NOT_RECORDED",
                "soft_requestwise_no_weaker_max_violation": max(finite(payload["route"]["soft_requestwise_no_weaker_max_violation"], "soft certificate") for payload in writer_payloads),
                "request_simplex_max_abs_residual": max(finite(payload["route"]["request_simplex_max_abs_residual"], "simplex residual") for payload in writer_payloads),
                "allocation_coefficient_share_layer_4": None if layer_total == 0.0 else layer_sum[0] / layer_total,
                "allocation_coefficient_share_layer_5": None if layer_total == 0.0 else layer_sum[1] / layer_total,
                "allocation_coefficient_share_layer_6": None if layer_total == 0.0 else layer_sum[2] / layer_total,
                "allocation_coefficient_share_layer_7": None if layer_total == 0.0 else layer_sum[3] / layer_total,
                "allocation_coefficient_share_layer_8": None if layer_total == 0.0 else layer_sum[4] / layer_total,
                "target_wall_seconds": finite(terminal["target_wall_seconds"], "target wall"),
                "writer_wall_seconds": finite(terminal["writer_wall_seconds"], "writer wall"),
                "terminal_w_evaluator_wall_seconds": finite(terminal["terminal_w_evaluator_wall_seconds"], "W evaluator wall"),
                "terminal_z_evaluator_wall_seconds": finite(terminal["terminal_z_evaluator_wall_seconds"], "Z evaluator wall"),
                "total_wall_seconds": finite(terminal["total_wall_seconds"], "total wall"),
                "response_batched_vjp_count": int(terminal["response_batched_vjp_count"]),
                "case_terminal_sha256": sha256_file(case_terminal_path),
                "case_terminal_identity_sha256": str(terminal["identity_sha256"]),
                "action_freeze_sha256": str(terminal["action_freeze_sha256"]),
                "target_trajectory_file_sha256": str(terminal["target_trajectory_file_sha256"]),
            }
            for prefix, values in (("w", w), ("z", z)):
                for metric, _ in METRIC_LABELS:
                    for suffix in ("correct", "denominator", "rate", "mean_target_new_nll", "mean_margin"):
                        case_row[f"{prefix}_{metric}_{suffix}"] = values[f"{prefix}_{metric}_{suffix}"]
            for metric, _ in METRIC_LABELS:
                case_row[f"z_to_w_{metric}_new_nll_gap"] = case_row[f"w_{metric}_mean_target_new_nll"] - case_row[f"z_{metric}_mean_target_new_nll"]
                case_row[f"z_to_w_{metric}_margin_gap"] = case_row[f"w_{metric}_mean_margin"] - case_row[f"z_{metric}_mean_margin"]
            for key in (
                "outer_h_application_count", "applied_coordinate_update_count", "writer_h_numeric_multiplication_count",
                "physical_h_application_count", "second_h_application_count", "residual_presplit_count",
                "remaining_division_count", "semantic_debt_input_count",
            ):
                case_row[f"{key}_total"] = int(adapter_counts[key])
            for key, value in terminal["compute"].items():
                case_row[f"compute_{key}"] = int(value)
            case_rows.append(case_row)
    root_record = {
        "path": str(root.relative_to(REPO)),
        "alias": alias,
        "clamp_policy": policy,
        "manifest_sha256": sha256_file(manifest_path),
        "terminal_sha256": sha256_file(job_terminal_path),
        "job_terminal_identity_sha256": job_terminal["identity_sha256"],
        "source_head": job_terminal["source_head"],
        "sealed_p2r2_v2_source_head": job_terminal["sealed_p2r2_v2_source_head"],
        "attempt_count": job_terminal["attempt_count"],
        "completed_case_arm_count": job_terminal["completed_case_arm_count"],
        "failed_case_arm_count": job_terminal["failed_case_arm_count"],
        "request_attempt_count": job_terminal["request_attempt_count"],
        "W0_restored": job_terminal["W0_restored"],
        "total_wall_seconds": job_terminal["total_wall_seconds"],
        "job_compute": job_terminal["job_compute"],
    }
    return case_rows, step_rows, request_rows, target_rows, root_record


def aggregate_summary(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[(row["alias"], row["clamp_policy"], row["arm"])].append(row)
    output: list[dict[str, Any]] = []
    for (alias, policy, arm), rows in sorted(grouped.items()):
        result: dict[str, Any] = {
            "alias": alias,
            "clamp_policy": policy,
            "arm": arm,
            "case_arm_denominator": len(rows),
            "request_denominator": sum(int(row["request_count"]) for row in rows),
            "W0_restored_count": sum(bool(row["w0_restored"]) for row in rows),
            "action_frozen_count": sum(bool(row["action_frozen_before_evaluator"]) for row in rows),
            "target_microsteps": sum(int(row["target_microstep_count"]) for row in rows),
            "writer_transitions": sum(int(row["writer_transition_count"]) for row in rows),
            "writer_materializations": sum(int(row["writer_materialization_count"]) for row in rows),
            "target_event_kind": rows[0]["target_event_kind"],
            "target_event_occurrence_count": sum(int(row["target_event_occurrence_count"]) for row in rows),
            "target_event_request_count": sum(int(row["target_event_request_count"]) for row in rows),
            "negative_actual_count": sum(int(row["negative_actual_count"]) for row in rows),
            "unreachable_event_count": sum(int(row["unreachable_event_count"]) for row in rows),
            "unreachable_unique_request_count_sum": sum(int(row["unreachable_unique_request_count"]) for row in rows),
            "terminal_cumulative_structural_p_mean": mean(float(row["terminal_cumulative_structural_p"]) for row in rows),
            "bf16_capacity_final_sum_mean": mean(float(row["bf16_capacity_final_sum"]) for row in rows),
            "bf16_update_energy_path_sum_mean": mean(float(row["bf16_update_energy_path_sum"]) for row in rows),
            "target_path_displacement_norm_mean": mean(float(row["target_path_displacement_norm_mean"]) for row in rows),
            "target_net_origin_displacement_norm_mean": mean(float(row["target_net_origin_displacement_norm_mean"]) for row in rows if row["target_net_origin_displacement_norm_mean"] is not None),
            "terminal_full_six_target_new_nll_mean": weighted_mean(rows, "terminal_full_six_target_new_nll", "request_count"),
            "writer_realization_mean": mean(float(row["writer_realization_mean"]) for row in rows),
            "writer_realization_median_mean_of_case_medians": mean(float(row["writer_realization_median"]) for row in rows),
            "writer_realization_p90_mean_of_case_p90": mean(float(row["writer_realization_p90"]) for row in rows),
            "soft_requestwise_no_weaker_max_violation_max": max(float(row["soft_requestwise_no_weaker_max_violation"]) for row in rows),
            "request_simplex_max_abs_residual_max": max(float(row["request_simplex_max_abs_residual"]) for row in rows),
            "routing_dof": "NOT_RECORDED",
            "target_wall_seconds_sum": sum(float(row["target_wall_seconds"]) for row in rows),
            "writer_wall_seconds_sum": sum(float(row["writer_wall_seconds"]) for row in rows),
            "terminal_w_evaluator_wall_seconds_sum": sum(float(row["terminal_w_evaluator_wall_seconds"]) for row in rows),
            "terminal_z_evaluator_wall_seconds_sum": sum(float(row["terminal_z_evaluator_wall_seconds"]) for row in rows),
            "total_wall_seconds_sum": sum(float(row["total_wall_seconds"]) for row in rows),
            "writer_update_norm": "NOT_RECORDED",
            "writer_path_norm": "NOT_RECORDED",
            "writer_net_norm": "NOT_RECORDED",
        }
        for metric, _ in METRIC_LABELS:
            for prefix in ("w", "z"):
                result[f"{prefix}_{metric}_correct"] = sum(int(row[f"{prefix}_{metric}_correct"]) for row in rows)
                result[f"{prefix}_{metric}_denominator"] = sum(int(row[f"{prefix}_{metric}_denominator"]) for row in rows)
                result[f"{prefix}_{metric}_rate"] = result[f"{prefix}_{metric}_correct"] / result[f"{prefix}_{metric}_denominator"]
                result[f"{prefix}_{metric}_mean_target_new_nll"] = weighted_mean(rows, f"{prefix}_{metric}_mean_target_new_nll", f"{prefix}_{metric}_denominator")
                result[f"{prefix}_{metric}_mean_margin"] = weighted_mean(rows, f"{prefix}_{metric}_mean_margin", f"{prefix}_{metric}_denominator")
            result[f"z_to_w_{metric}_new_nll_gap"] = result[f"w_{metric}_mean_target_new_nll"] - result[f"z_{metric}_mean_target_new_nll"]
            result[f"z_to_w_{metric}_margin_gap"] = result[f"w_{metric}_mean_margin"] - result[f"z_{metric}_mean_margin"]
        for layer in range(4, 9):
            result[f"allocation_coefficient_share_layer_{layer}_mean"] = mean(float(row[f"allocation_coefficient_share_layer_{layer}"]) for row in rows)
        for key in (
            "outer_h_application_count", "applied_coordinate_update_count", "writer_h_numeric_multiplication_count",
            "physical_h_application_count", "second_h_application_count", "residual_presplit_count",
            "remaining_division_count", "semantic_debt_input_count",
        ):
            result[f"{key}_total"] = sum(int(row[f"{key}_total"]) for row in rows)
        statuses = Counter()
        fallbacks = Counter()
        for row in rows:
            statuses.update(row["route_status_counts"])
            fallbacks.update(row["route_fallback_counts"])
        result["route_status_counts"] = dict(sorted(statuses.items()))
        result["route_fallback_counts"] = dict(sorted(fallbacks.items()))
        for key in sorted(key for key in rows[0] if key.startswith("compute_")):
            result[f"{key}_sum"] = sum(int(row[key]) for row in rows)
        output.append(result)
    return output


def paired_rows(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["alias"], row["arm"], row["case_index"], row["clamp_policy"]): row for row in case_rows}
    rows: list[dict[str, Any]] = []
    for alias, arm, case_index in sorted({(row["alias"], row["arm"], row["case_index"]) for row in case_rows}):
        on = lookup[(alias, arm, case_index, "ON")]
        off = lookup[(alias, arm, case_index, "OFF")]
        identity_pass = all(on[key] == off[key] for key in ("alias", "arm", "case_index", "request_order_sha256", "request_count"))
        row: dict[str, Any] = {
            "alias": alias,
            "arm": arm,
            "case_index": case_index,
            "pair_identity_pass": identity_pass,
            "request_order_sha256": on["request_order_sha256"],
            "case_arm_completed_on": True,
            "case_arm_completed_off": True,
        }
        numeric = (
            "terminal_full_six_target_new_nll", "terminal_cumulative_structural_p", "bf16_capacity_final_sum",
            "bf16_update_energy_path_sum", "target_path_displacement_norm_mean", "target_net_origin_displacement_norm_mean",
            "negative_actual_count", "unreachable_event_count", "unreachable_unique_request_count", "writer_realization_mean",
            "target_event_occurrence_count", "target_event_request_count",
        )
        for key in numeric:
            row[f"on_{key}"] = on[key]
            row[f"off_{key}"] = off[key]
            row[f"off_minus_on_{key}"] = None if on[key] is None or off[key] is None else float(off[key]) - float(on[key])
        for metric, _ in METRIC_LABELS:
            for prefix in ("w", "z"):
                for suffix in ("correct", "denominator", "mean_target_new_nll", "mean_margin"):
                    key = f"{prefix}_{metric}_{suffix}"
                    row[f"on_{key}"] = on[key]
                    row[f"off_{key}"] = off[key]
                    row[f"off_minus_on_{key}"] = float(off[key]) - float(on[key])
            for suffix in ("new_nll_gap", "margin_gap"):
                key = f"z_to_w_{metric}_{suffix}"
                row[f"on_{key}"] = on[key]
                row[f"off_{key}"] = off[key]
                row[f"off_minus_on_{key}"] = float(off[key]) - float(on[key])
        rows.append(row)
    if len(rows) != 40 or not all(row["pair_identity_pass"] for row in rows):
        raise ValueError("paired ON/OFF identity differs")
    return rows


def paired_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["alias"], row["arm"])].append(row)
    output: list[dict[str, Any]] = []
    for (alias, arm), group in sorted(grouped.items()):
        result = {"alias": alias, "arm": arm, "paired_case_denominator": len(group), "pair_identity_pass_count": sum(bool(row["pair_identity_pass"]) for row in group)}
        for key in sorted(key for key in group[0] if key.startswith("off_minus_on_")):
            values = [row[key] for row in group if row[key] is not None]
            result[f"{key}_mean"] = mean(float(value) for value in values) if values else None
            if key.endswith("_correct") or key.endswith("_denominator") or key.endswith("_count"):
                result[f"{key}_sum"] = sum(float(value) for value in values) if values else None
        output.append(result)
    return output


def markdown(summary: list[dict[str, Any]], paired: list[dict[str, Any]], roots: list[dict[str, Any]], report_root: str) -> str:
    def f(value: Any, digits: int = 4) -> str:
        if value is None:
            return "NOT_RECORDED"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines = [
        "# P2R4 Phase-B Clamp ON/OFF × P2R2 V2 Writer — 터미널 사실 보고",
        "",
        "- instruction_id: `ODEEDIT-S05-P2R4-P2R1-CLAMP-ON-OFF-CAUSAL-ABLATION-V1-PHASE-B`",
        "- method_id: `P2R4-P2R1-CLAMP-ON-OFF-P2R2-V2-WRITER-V1`",
        "- source head: `96f23e804be66569ad3fcf98bfcdd8a02ff3f7fb`; sealed P2R2 V2: `c97e8619b42da7954ce0e824c215a8a82d70589a`.",
        "- 4 job roots × 2 arms × 10 independent B10 = 80 completed case-arm attempts; 800 request attempts; no imputation.",
        "- `scientific_promotion=false`. 본 문서는 수치·식별자·기계적 수신증만 기록한다.",
        "",
        "## 터미널 무결성",
        "",
        "| model / clamp | completed / failed | request attempts | W0 restored | job wall s | job model F | job tokens | terminal SHA |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for root in sorted(roots, key=lambda item: (item["alias"], item["clamp_policy"])):
        counters = root["job_compute"]["counters"]
        lines.append(
            f"| {root['alias']} / {root['clamp_policy']} | {root['completed_case_arm_count']} / {root['failed_case_arm_count']} | {root['request_attempt_count']} | {root['W0_restored']} | {f(root['total_wall_seconds'],3)} | {counters['model_forward']} | {counters['processed_tokens']} | `{root['terminal_sha256']}` |"
        )
    lines += [
        "",
        "모든 case manifest→terminal SHA 및 최상위 manifest→terminal SHA를 재해시해 일치시켰다. 모든 case에서 byte/pointer W0 restore 및 action freeze=true, heldout controller access=0이다.",
        "",
        "## W-only 및 z-inject endpoint",
        "",
        "| model / clamp / arm | W Eff | W Gen | W Loc | z Eff | z Gen | z Loc | W Eff new NLL / margin | z Eff new NLL / margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {alias} / {clamp_policy} / {arm} | {we}/{wed} | {wg}/{wgd} | {wl}/{wld} | {ze}/{zed} | {zg}/{zgd} | {zl}/{zld} | {wen} / {wem} | {zen} / {zem} |".format(
                alias=row["alias"], clamp_policy=row["clamp_policy"], arm=row["arm"],
                we=row["w_efficacy_correct"], wed=row["w_efficacy_denominator"],
                wg=row["w_generalization_correct"], wgd=row["w_generalization_denominator"],
                wl=row["w_locality-preservation_correct"], wld=row["w_locality-preservation_denominator"],
                ze=row["z_efficacy_correct"], zed=row["z_efficacy_denominator"],
                zg=row["z_generalization_correct"], zgd=row["z_generalization_denominator"],
                zl=row["z_locality-preservation_correct"], zld=row["z_locality-preservation_denominator"],
                wen=f(row["w_efficacy_mean_target_new_nll"], 6), wem=f(row["w_efficacy_mean_margin"], 6),
                zen=f(row["z_efficacy_mean_target_new_nll"], 6), zem=f(row["z_efficacy_mean_margin"], 6),
            )
        )
    lines += [
        "",
        "## z→W / writer / clamp 수신증",
        "",
        "| model / clamp / arm | z→W Eff NLL gap | z→W Gen NLL gap | terminal full-six target-new NLL | realization mean / p90 | cumulative Structural-P mean | final BF16 capacity sum mean | BF16 step-energy path sum mean | negative actual | unreachable events | target event occurrences / requests |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {alias} / {clamp_policy} / {arm} | {e} | {g} | {full} | {real} / {p90} | {p} | {cap} | {energy} | {negative} | {unreach} | {events} / {event_requests} ({event_kind}) |".format(
                alias=row["alias"], clamp_policy=row["clamp_policy"], arm=row["arm"],
                e=f(row["z_to_w_efficacy_new_nll_gap"], 6), g=f(row["z_to_w_generalization_new_nll_gap"], 6),
                full=f(row["terminal_full_six_target_new_nll_mean"], 6),
                real=f(row["writer_realization_mean"], 6), p90=f(row["writer_realization_p90_mean_of_case_p90"], 6),
                p=f(row["terminal_cumulative_structural_p_mean"], 6), cap=f(row["bf16_capacity_final_sum_mean"], 6),
                energy=f(row["bf16_update_energy_path_sum_mean"], 6), negative=row["negative_actual_count"],
                unreach=row["unreachable_event_count"], events=row["target_event_occurrence_count"],
                event_requests=row["target_event_request_count"], event_kind=row["target_event_kind"],
            )
        )
    lines += [
        "",
        "writer update/path/net norm은 수신증에 독립 scalar norm으로 기록되지 않아 `NOT_RECORDED`이다. target path displacement norm은 표 파일에 있으며, ON target origin-net norm은 수신증에 없다. BF16 energy/capacity는 위 표의 명시된 원시 receipt 필드 합산값이다.",
        "",
        "## 라우팅 및 receipt-adapter",
        "",
        "| model / clamp / arm | route status counts | routing DOF | layer coefficient shares L4/L5/L6/L7/L8 | soft no-weaker max violation | simplex residual max | outer-h | applied-coordinate | numeric-h / physical-h / second-h | pre-split / remaining / debt |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        shares = "/".join(f(row[f"allocation_coefficient_share_layer_{layer}_mean"], 6) for layer in range(4, 9))
        lines.append(
            "| {alias} / {clamp_policy} / {arm} | `{status}` | {dof} | {shares} | {soft} | {simplex} | {outer} | {coord} | {numeric} / {physical} / {second} | {presplit} / {remaining} / {debt} |".format(
                alias=row["alias"], clamp_policy=row["clamp_policy"], arm=row["arm"], status=json.dumps(row["route_status_counts"], sort_keys=True),
                dof=row["routing_dof"], shares=shares,
                soft=f(row["soft_requestwise_no_weaker_max_violation_max"], 12), simplex=f(row["request_simplex_max_abs_residual_max"], 12),
                outer=row["outer_h_application_count_total"], coord=row["applied_coordinate_update_count_total"],
                numeric=row["writer_h_numeric_multiplication_count_total"], physical=row["physical_h_application_count_total"], second=row["second_h_application_count_total"],
                presplit=row["residual_presplit_count_total"], remaining=row["remaining_division_count_total"], debt=row["semantic_debt_input_count_total"],
            )
        )
    lines += [
        "",
        "`outer_h_application_count=1` 및 `applied_coordinate_update_count=1`은 committed joint writer transition의 논리적 ODE-transition receipt이다. `writer_h_numeric_multiplication_count=0`, `physical_h_application_count=0`, `second_h_application_count=0`은 추가 수치 h 곱셈이 없음을 기록한다.",
        "",
        "## 정확한 case-paired OFF−ON 산술",
        "",
        "| model / arm | matched cases | Δ W Eff correct | Δ W Gen correct | Δ W Loc correct | mean Δ W Eff new NLL | mean Δ z→W Eff NLL gap | mean Δ Structural-P | mean Δ BF16 capacity | mean Δ BF16 energy path |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            "| {alias} / {arm} | {n} | {eff} | {gen} | {loc} | {nll} | {gap} | {p} | {cap} | {energy} |".format(
                alias=row["alias"], arm=row["arm"], n=row["paired_case_denominator"],
                eff=f(row.get("off_minus_on_w_efficacy_correct_sum"), 0), gen=f(row.get("off_minus_on_w_generalization_correct_sum"), 0), loc=f(row.get("off_minus_on_w_locality-preservation_correct_sum"), 0),
                nll=f(row.get("off_minus_on_w_efficacy_mean_target_new_nll_mean"), 6),
                gap=f(row.get("off_minus_on_z_to_w_efficacy_new_nll_gap_mean"), 6),
                p=f(row.get("off_minus_on_terminal_cumulative_structural_p_mean"), 6),
                cap=f(row.get("off_minus_on_bf16_capacity_final_sum_mean"), 6),
                energy=f(row.get("off_minus_on_bf16_update_energy_path_sum_mean"), 6),
            )
        )
    lines += [
        "",
        "OFF−ON은 동일 alias/arm/case_index, request order 및 request cardinality가 일치한 40개 case pair의 산술값이다. denominator 또는 대응 데이터가 없는 값은 표·JSON에서 `NOT_RECORDED`로 표시한다.",
        "",
        "## Compute ledger",
        "",
        "| model / clamp / arm | target F/B | KL F/B | response F/VJP | capture F | post-write F | writer materializations | target/writer/evaluator wall s (sum) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {alias} / {clamp_policy} / {arm} | {tf}/{tb} | {kf}/{kb} | {rf}/{rv} | {cf} | {pf} | {mat} | {tw}/{ww}/{ew} |".format(
                alias=row["alias"], clamp_policy=row["clamp_policy"], arm=row["arm"],
                tf=row["compute_target_forward_count_sum"], tb=row["compute_target_backward_count_sum"],
                kf=row["compute_kl_forward_count_sum"], kb=row["compute_kl_backward_count_sum"],
                rf=row["compute_physical_response_forward_count_sum"], rv=row["compute_physical_response_batched_vjp_count_sum"],
                cf=row["compute_physical_capture_forward_count_sum"], pf=row["compute_post_write_objective_forward_count_sum"],
                mat=row["writer_materializations"], tw=f(row["target_wall_seconds_sum"], 3), ww=f(row["writer_wall_seconds_sum"], 3),
                ew=f(row["terminal_w_evaluator_wall_seconds_sum"] + row["terminal_z_evaluator_wall_seconds_sum"], 3),
            )
        )
    lines += [
        "",
        "## 19965 technical-repair boundary",
        "",
        "- initial B1 submission receipt: job `19965`, source `49490d23c6a156c6e9e1a1ce93ce2d85145f1659`, array `0-3%2`.",
        "- replacement B1 receipt: job `19969`, source `96f23e804be66569ad3fcf98bfcdd8a02ff3f7fb`, attempt suffix `server2-node-r1`.",
        "- production receipt: job `19973`, source `96f23e804be66569ad3fcf98bfcdd8a02ff3f7fb`, 80 case-arm attempts.",
        "- source commit `96f23e804be66569ad3fcf98bfcdd8a02ff3f7fb` changes scheduler targeting in the P2R4 Phase-B sbatch/submitter from `devbox` to `server2` and updates its focused launcher test plus source manifest. No P2R2 writer equation file is modified by that commit.",
        "- scheduler State/ExitCode for job `19965` is `NOT_RECORDED` in the permitted raw-free receipt set; no log was read.",
        "",
        "## Artifact scope and root",
        "",
        f"- analysis root: `{report_root}`",
        "- tables contain scalar receipt fields, endpoint counts/continuous metrics, hashes and typed status only; no prompts, targets, generations, tensors, weights, model/data/cache, or runtime logs are copied.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(mode=0o700, parents=True, exist_ok=True)
    if any(path.exists() for path in OUT.iterdir() if path.name != Path(__file__).name):
        raise FileExistsError(f"report namespace is not empty: {OUT}")
    all_case_rows: list[dict[str, Any]] = []
    all_step_rows: list[dict[str, Any]] = []
    all_request_rows: list[dict[str, Any]] = []
    all_target_rows: list[dict[str, Any]] = []
    roots: list[dict[str, Any]] = []
    for root in RESULT_ROOTS:
        case_rows, step_rows, request_rows, target_rows, root_record = collect_root(root)
        all_case_rows.extend(case_rows)
        all_step_rows.extend(step_rows)
        all_request_rows.extend(request_rows)
        all_target_rows.extend(target_rows)
        roots.append(root_record)
    if (len(all_case_rows), len(all_step_rows), len(all_request_rows), len(all_target_rows)) != (80, 640, 6400, 19200):
        raise ValueError("derived table cardinality differs")
    summary_rows = aggregate_summary(all_case_rows)
    if len(summary_rows) != 8:
        raise ValueError("summary cell cardinality differs")
    paired_case_rows = paired_rows(all_case_rows)
    paired_summary_rows = paired_summary(paired_case_rows)
    if len(paired_summary_rows) != 4:
        raise ValueError("paired summary cardinality differs")
    outputs: dict[str, Any] = {
        f"{REPORT_STEM}-summary.json": {
            "schema": "ode-edit-s05-p2r4-phaseb-terminal-summary/v1",
            "source_head": "96f23e804be66569ad3fcf98bfcdd8a02ff3f7fb",
            "sealed_p2r2_v2_source_head": "c97e8619b42da7954ce0e824c215a8a82d70589a",
            "cell_count": len(summary_rows),
            "case_arm_count": len(all_case_rows),
            "writer_step_count": len(all_step_rows),
            "writer_request_step_count": len(all_request_rows),
            "target_request_microstep_count": len(all_target_rows),
            "paired_case_count": len(paired_case_rows),
            "roots": roots,
            "summary_cells": summary_rows,
            "paired_summary": paired_summary_rows,
            "scientific_promotion": False,
        },
        f"{REPORT_STEM}-per-case.json": all_case_rows,
        f"{REPORT_STEM}-per-step.json": all_step_rows,
        f"{REPORT_STEM}-per-request-writer-step.json": all_request_rows,
        f"{REPORT_STEM}-per-request-target-microstep.json": all_target_rows,
        f"{REPORT_STEM}-paired-off-minus-on.json": paired_case_rows,
    }
    for filename, value in outputs.items():
        write_json_once(OUT / filename, value)
    for filename, rows in (
        (f"{REPORT_STEM}-per-case.csv", all_case_rows),
        (f"{REPORT_STEM}-per-step.csv", all_step_rows),
        (f"{REPORT_STEM}-per-request-writer-step.csv", all_request_rows),
        (f"{REPORT_STEM}-per-request-target-microstep.csv", all_target_rows),
        (f"{REPORT_STEM}-paired-off-minus-on.csv", paired_case_rows),
        (f"{REPORT_STEM}-summary.csv", summary_rows),
    ):
        write_csv_once(OUT / filename, rows)
    core_files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != Path(__file__).name)
    core_entries = [{
        "path": path.name,
        "mode": f"{path.stat().st_mode & 0o777:04o}",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    } for path in core_files]
    analysis_root = canonical_hash(core_entries)
    report = markdown(summary_rows, paired_summary_rows, roots, analysis_root)
    report_path = OUT / f"{REPORT_STEM}-terminal-report-ko.md"
    write_once(report_path, report.encode("utf-8"))
    all_entries = [{
        "path": path.name,
        "mode": f"{path.stat().st_mode & 0o777:04o}",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    } for path in sorted(OUT.iterdir()) if path.is_file()]
    analysis_root = canonical_hash(all_entries)
    manifest = {
        "schema": "ode-edit-s05-p2r4-phaseb-analysis-manifest/v1",
        "analysis_namespace": REPORT_STEM,
        "source_head": "96f23e804be66569ad3fcf98bfcdd8a02ff3f7fb",
        "sealed_p2r2_v2_source_head": "c97e8619b42da7954ce0e824c215a8a82d70589a",
        "input_root_count": len(roots),
        "input_roots": roots,
        "counts": {
            "cells": 8,
            "case_arm_rows": len(all_case_rows),
            "writer_step_rows": len(all_step_rows),
            "writer_request_step_rows": len(all_request_rows),
            "target_request_microstep_rows": len(all_target_rows),
            "paired_case_rows": len(paired_case_rows),
        },
        "analysis_generator_sha256": sha256_file(Path(__file__)),
        "files": all_entries,
        "analysis_root": analysis_root,
        "scientific_promotion": False,
    }
    manifest_path = OUT / f"{REPORT_STEM}-analysis-manifest.json"
    manifest_sha = write_json_once(manifest_path, manifest)
    receipt = {
        "schema": "ode-edit-s05-p2r4-phaseb-analysis-receipt/v1",
        "analysis_manifest": manifest_path.name,
        "analysis_manifest_sha256": manifest_sha,
        "analysis_root": analysis_root,
        "input_terminal_rehash_pass": True,
        "case_terminal_rehash_pass": True,
        "W0_restore_pass": True,
        "action_freeze_pass": True,
        "receipt_adapter_pass": True,
        "full_denominator_no_imputation": True,
        "raw_prompt_target_generation_tensor_weight_model_data_cache_runtime_log_copied": False,
        "scientific_promotion": False,
    }
    receipt["receipt_root"] = canonical_hash(receipt)
    write_json_once(OUT / f"{REPORT_STEM}-analysis-receipt.json", receipt)
    for path in OUT.iterdir():
        if path.is_file():
            os.chmod(path, 0o600)


if __name__ == "__main__":
    main()
