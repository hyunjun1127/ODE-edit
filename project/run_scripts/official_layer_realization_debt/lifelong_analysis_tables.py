"""Typed extraction and descriptive summaries for lifelong observer artifacts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .lifelong_analysis_io import (
    AnalysisBoundary,
    CELL_ORDER,
    IDEAL_Q,
    LAYERS,
    Q_LABELS,
    ValidatedArm,
)


NOT_RECORDED = "NOT_RECORDED_SCHEMA"


def summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise AnalysisBoundary("summary input is empty")
    if not np.isfinite(array).all():
        raise AnalysisBoundary("summary input contains nonfinite values")
    p25, median, p75, p90 = np.percentile(array, (25, 50, 75, 90))
    return {
        "n": int(array.size),
        "sum": float(array.sum()),
        "mean": float(array.mean()),
        "median": float(median),
        "p25": float(p25),
        "p75": float(p75),
        "iqr": float(p75 - p25),
        "p90": float(p90),
        "max": float(array.max()),
        "min": float(array.min()),
    }


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | str:
    if len(x) != len(y) or len(x) < 3:
        return "NOT_IDENTIFIABLE_N_LT_3"
    rx, ry = _rankdata(x), _rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return "NOT_IDENTIFIABLE_CONSTANT"
    return float(np.corrcoef(rx, ry)[0, 1])


def _observer_rows(
    *,
    model: str,
    method: str,
    stage: str,
    batch_index: int,
    checkpoint: int | str,
    fork: str,
    observer: Mapping[str, Any],
    endpoint: Mapping[str, Mapping[str, Any]] | None = None,
    action: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []
    layers: list[dict[str, Any]] = []
    weights: list[dict[str, Any]] = []
    for record in observer["residual_debt"]["records"]:
        request_hash = str(record["request_sha256"])
        q = [float(value) for value in record["q"]]
        if len(q) != len(Q_LABELS):
            raise AnalysisBoundary(f"q trajectory length differs: {model}/{method}/{stage}")
        layer_records = list(record["layers"])
        if [int(value["layer"]) for value in layer_records] != list(LAYERS):
            raise AnalysisBoundary(f"layer trajectory order differs: {model}/{method}/{stage}")
        endpoint_row = {} if endpoint is None else dict(endpoint.get(request_hash, {}))
        action_row = {} if action is None else dict(action.get(request_hash, {}))
        request = {
            "model": model,
            "method": method,
            "stage": stage,
            "batch_index": batch_index,
            "accepted_edit_count": checkpoint,
            "fork": fork,
            "request_sha256": request_hash,
            "case_identity_sha256": str(record["case_identity_sha256"]),
            **{f"q_{label.replace('-', '_')}": q[index] for index, label in enumerate(Q_LABELS)},
            **{
                f"ideal_abs_deviation_{label.replace('-', '_')}": abs(q[index] - IDEAL_Q[index])
                for index, label in enumerate(Q_LABELS)
            },
            "q_pre_L8_gt_0_2": int(bool(record["q_L8_gt_0_2"])),
            "d_parallel": float(record["d_parallel"]),
            "d_perp": float(record["d_perp"]),
            "mean_rho": float(np.mean([float(value["rho"]) for value in layer_records])),
            "mean_tau": float(np.mean([float(value["tau"]) for value in layer_records])),
            "recurrence_closure_relative_error": float(record["recurrence_closure_relative_error"]),
            "target_new_nll": endpoint_row.get("target_new_nll", NOT_RECORDED),
            "target_true_nll": endpoint_row.get("target_true_nll", NOT_RECORDED),
            "target_new_margin": endpoint_row.get("target_new_margin", NOT_RECORDED),
            "target_true_margin": endpoint_row.get("target_true_margin", NOT_RECORDED),
            "target_new_strict": (
                int(bool(endpoint_row["target_new_strict"])) if "target_new_strict" in endpoint_row else NOT_RECORDED
            ),
            "target_true_strict": (
                int(bool(endpoint_row["target_true_strict"])) if "target_true_strict" in endpoint_row else NOT_RECORDED
            ),
            "D_TV": action_row.get("D_TV", NOT_RECORDED),
            "delta_center": action_row.get("delta_center", NOT_RECORDED),
            "negative_progress_count": action_row.get("negative_progress_count", NOT_RECORDED),
            "negative_progress_total": action_row.get("negative_progress_total", NOT_RECORDED),
            "positive_progress_total": action_row.get("positive_progress_total", NOT_RECORDED),
        }
        requests.append(request)
        for layer_record in layer_records:
            allocation = float(layer_record["allocation_norm_over_R1"])
            rho = float(layer_record["rho"])
            tau = float(layer_record["tau"])
            layer = int(layer_record["layer"])
            layers.append(
                {
                    "model": model,
                    "method": method,
                    "stage": stage,
                    "batch_index": batch_index,
                    "accepted_edit_count": checkpoint,
                    "fork": fork,
                    "request_sha256": request_hash,
                    "case_identity_sha256": str(record["case_identity_sha256"]),
                    "layer": layer,
                    "q_pre": float(layer_record["q_pre"]),
                    "q_post": float(layer_record["q_post"]),
                    "allocation_norm_over_R1": allocation,
                    "realized_reduction_norm_over_R1": float(layer_record["realized_reduction_norm_over_R1"]),
                    "gap_norm_over_R1": float(layer_record["gap_norm_over_R1"]),
                    "rho": rho,
                    "tau": tau,
                    "under_realization_coefficient": float(layer_record["under_realization_coefficient"]),
                    "Y_parallel_over_R1": rho * allocation,
                    "Y_perp_over_R1": tau * allocation,
                    "E_parallel_over_R1": (1.0 - rho) * allocation,
                    "E_perp_magnitude_over_R1": tau * allocation,
                    "realization_class": "OPPOSITE" if rho < 0 else ("UNDER" if rho < 1 else ("OVERSHOOT" if rho > 1 else "EXACT")),
                    "next_pre_link_relative_difference": (
                        "NOT_APPLICABLE"
                        if layer_record.get("next_pre_link_relative_difference") is None
                        else float(layer_record["next_pre_link_relative_difference"])
                    ),
                    "recurrence_closure_relative_error": float(record["recurrence_closure_relative_error"]),
                }
            )
    weight_action = observer["weight_action"]
    for record in weight_action["layers"]:
        weights.append(
            {
                "model": model,
                "method": method,
                "stage": stage,
                "batch_index": batch_index,
                "accepted_edit_count": checkpoint,
                "fork": fork,
                "weight_observation_unit": "ONE_B100_BATCH_OR_SENTINEL_PROBE_LAYER_UPDATE",
                "layer": int(record["layer"]),
                "frobenius_magnitude": float(record["frobenius_magnitude"]),
                "frobenius_squared_telemetry": float(record["frobenius_energy"]),
                "relative_update_magnitude": float(record["relative_update_magnitude"]),
                "update_magnitude_share": float(record["update_magnitude_share"]),
                "entry_weight_frobenius_magnitude": float(record["entry_weight_frobenius_magnitude"]),
                "entry_weight_sha256": str(record["entry_weight_sha256"]),
                "edited_weight_sha256": str(record["edited_weight_sha256"]),
            }
        )
    return requests, layers, weights


def _functional_rows(model: str, method: str, checkpoint: Mapping[str, Any]) -> list[dict[str, Any]]:
    accepted = int(checkpoint["accepted_edit_count"])
    functional = checkpoint["functional"]
    panels: list[tuple[str, Mapping[str, Any]]] = []
    if accepted == 0:
        panels.append(("sentinel_pre_edit", functional["sentinel_pre_edit"]))
    else:
        panels.append(("current", functional["current"]))
        panels.extend((f"retention_{name}", panel) for name, panel in functional["retention"].items())
    rows: list[dict[str, Any]] = []
    for cohort, panel in panels:
        for record in panel["records"]:
            for metric in ("rewrite_target_new", "rewrite_target_true", "rephrase_target_new", "locality_target_true"):
                value = record.get(metric, {"status": NOT_RECORDED})
                available = "count" in value
                rows.append(
                    {
                        "model": model,
                        "method": method,
                        "accepted_edit_count": accepted,
                        "batch_index": int(checkpoint["batch_index"]),
                        "cohort": cohort,
                        "panel_role": str(panel["panel_role"]),
                        "sampling_identity": (
                            functional.get("sampling_identity", NOT_RECORDED)
                            if accepted > 0
                            else NOT_RECORDED
                        ),
                        "request_sha256": str(record["request_sha256"]),
                        "case_identity_sha256": str(record["case_identity_sha256"]),
                        "metric": metric,
                        "availability": "RECORDED" if available else str(value.get("status", NOT_RECORDED)),
                        "prompt_count": int(value["count"]) if available else NOT_RECORDED,
                        "nll_mean": float(value["nll_mean"]) if available else NOT_RECORDED,
                        "margin_mean": float(value["margin_mean"]) if available else NOT_RECORDED,
                        "strict_count": int(value["strict_count"]) if available else NOT_RECORDED,
                        "strict_rate": float(value["strict_count"] / value["count"]) if available else NOT_RECORDED,
                        "raw_prompt_logit_generation_publish_count": int(record["raw_prompt_logit_generation_publish_count"]),
                    }
                )
    return rows


def _attach_checkpoint_action_realization(
    requests: Sequence[dict[str, Any]],
    layers: Sequence[Mapping[str, Any]],
    weights: Sequence[Mapping[str, Any]],
) -> None:
    """Derive the preregistered action/progress mismatch for checkpoint probes.

    Checkpoint observer payloads publish request-level allocation/rho and one
    B100-level layer update profile, but not the redundant action-realization
    summary.  This deterministic join reconstructs the same definitions used
    by production without treating the batch-level weight profile as 100
    independent weight observations.
    """
    update_share = {int(row["layer"]): float(row["update_magnitude_share"]) for row in weights}
    if set(update_share) != set(LAYERS) or not math.isclose(
        sum(update_share.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise AnalysisBoundary("checkpoint update-magnitude profile differs")
    by_request: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in layers:
        by_request[str(row["request_sha256"])].append(row)
    for request in requests:
        selected = sorted(by_request[str(request["request_sha256"])], key=lambda row: int(row["layer"]))
        if [int(row["layer"]) for row in selected] != list(LAYERS):
            raise AnalysisBoundary("checkpoint action-realization layer membership differs")
        signed_progress = [
            float(row["rho"]) * float(row["allocation_norm_over_R1"])
            for row in selected
        ]
        positive = [max(value, 0.0) for value in signed_progress]
        positive_total = sum(positive)
        if not math.isfinite(positive_total) or positive_total <= 0.0:
            raise AnalysisBoundary("checkpoint positive target-progress profile is undefined")
        progress_share = [value / positive_total for value in positive]
        weight_share = [update_share[layer] for layer in LAYERS]
        request["D_TV"] = 0.5 * sum(
            abs(left - right) for left, right in zip(weight_share, progress_share, strict=True)
        )
        request["delta_center"] = sum(
            layer * (right - left)
            for layer, left, right in zip(LAYERS, weight_share, progress_share, strict=True)
        )
        request["negative_progress_count"] = sum(value < 0.0 for value in signed_progress)
        request["negative_progress_total"] = sum(min(value, 0.0) for value in signed_progress)
        request["positive_progress_total"] = positive_total


@dataclass(slots=True)
class AnalysisTables:
    production_request: list[dict[str, Any]]
    production_layer: list[dict[str, Any]]
    production_weight: list[dict[str, Any]]
    production_batch: list[dict[str, Any]]
    checkpoint_request: list[dict[str, Any]]
    checkpoint_layer: list[dict[str, Any]]
    checkpoint_weight: list[dict[str, Any]]
    checkpoint_geometry: list[dict[str, Any]]
    checkpoint_waypoint: list[dict[str, Any]]
    checkpoint_same_entry: list[dict[str, Any]]
    functional: list[dict[str, Any]]
    compute: list[dict[str, Any]]
    summary: list[dict[str, Any]]
    paired_delta: list[dict[str, Any]]
    association: list[dict[str, Any]]
    outlier: list[dict[str, Any]]

    def named(self) -> dict[str, list[dict[str, Any]]]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def extract_tables(arms: Sequence[ValidatedArm]) -> AnalysisTables:
    production_request: list[dict[str, Any]] = []
    production_layer: list[dict[str, Any]] = []
    production_weight: list[dict[str, Any]] = []
    production_batch: list[dict[str, Any]] = []
    checkpoint_request: list[dict[str, Any]] = []
    checkpoint_layer: list[dict[str, Any]] = []
    checkpoint_weight: list[dict[str, Any]] = []
    checkpoint_geometry: list[dict[str, Any]] = []
    checkpoint_waypoint: list[dict[str, Any]] = []
    checkpoint_same_entry: list[dict[str, Any]] = []
    functional: list[dict[str, Any]] = []
    compute: list[dict[str, Any]] = []

    for arm in arms:
        model, method = arm.spec.model, arm.spec.method
        for journal in arm.journals:
            batch_index = int(journal["batch_index"])
            observer = journal["apply"]["layer_realization_observer"]
            endpoints = {row["request_sha256"]: row for row in journal["endpoint"]["records"]}
            actions = {row["request_sha256"]: row for row in journal["action_realization"]["requests"]}
            req, layers, weights = _observer_rows(
                model=model,
                method=method,
                stage="PRODUCTION_B100",
                batch_index=batch_index,
                checkpoint=batch_index * 100,
                fork="OFFICIAL_ORDERED",
                observer=observer,
                endpoint=endpoints,
                action=actions,
            )
            production_request.extend(req)
            production_layer.extend(layers)
            production_weight.extend(weights)
            cache = journal["cache_continuity"]
            action = journal["action_realization"]
            production_batch.append(
                {
                    "model": model,
                    "method": method,
                    "batch_index": batch_index,
                    "accepted_edit_start": int(journal["accepted_edit_start"]),
                    "accepted_edit_end": int(journal["accepted_edit_end"]),
                    "request_denominator": int(journal["request_count"]),
                    "D_TV_mean": summary(row["D_TV"] for row in action["requests"])["mean"],
                    "D_TV_median": summary(row["D_TV"] for row in action["requests"])["median"],
                    "delta_center_mean": summary(row["delta_center"] for row in action["requests"])["mean"],
                    "negative_progress_count": int(action["negative_progress_count"]),
                    "negative_progress_total": float(action["negative_progress_total"]),
                    "cache_kind": str(cache["kind"]),
                    "cache_entry_width": int(cache["entry_width"]),
                    "cache_exit_width": int(cache["exit_width"]),
                    "cache_append_width": int(cache["append_width"]),
                    "cache_consume_width": int(cache["consume_width"]),
                }
            )
            overhead = observer["overhead_wall_seconds"]
            audit = journal["apply"]["official_call_audit"]
            compute.append(
                {
                    "model": model,
                    "method": method,
                    "stage": "PRODUCTION_B100",
                    "batch_index": batch_index,
                    "accepted_edit_count": batch_index * 100,
                    "request_denominator": 100,
                    "direct_z_compute_count": int(observer["direct_z_compute_count"]),
                    "direct_z_recompute_count": int(observer["direct_z_recompute_count"]),
                    "layer_observation_count": int(observer["layer_loop_observation_copy_count"]),
                    "terminal_forward_count": int(observer["terminal_post_L8_forward_count"]),
                    "compute_ks_call_count": int(audit["compute_ks_call_count"]),
                    "solve_call_count": int(audit["torch_linalg_solve_call_count"]),
                    "target_backward_count": int(journal["apply"]["target_backward_count"]),
                    "jvp_count": NOT_RECORDED,
                    "vjp_count": NOT_RECORDED,
                    "hvp_count": NOT_RECORDED,
                    "total_forward_count": NOT_RECORDED,
                    "edit_core_wall_seconds": float(journal["apply"]["edit_core_wall_seconds"]),
                    "observer_wall_seconds": float(sum(float(value) for value in overhead.values())),
                    "gpu_peak_memory_bytes": NOT_RECORDED,
                }
            )

        for checkpoint in arm.checkpoints:
            accepted = int(checkpoint["accepted_edit_count"])
            functional.extend(_functional_rows(model, method, checkpoint))
            probe = checkpoint["probe"]
            compute.append(
                {
                    "model": model,
                    "method": method,
                    "stage": "CHECKPOINT_SHARED_TARGET_LEDGER",
                    "batch_index": int(checkpoint["batch_index"]),
                    "accepted_edit_count": accepted,
                    "request_denominator": int(probe["sentinel_request_count"]),
                    "direct_z_compute_count": int(probe["direct_z_optimizer_count"]),
                    "direct_z_recompute_count": int(probe["direct_z_recompute_count"]),
                    "layer_observation_count": 0,
                    "terminal_forward_count": 0,
                    "compute_ks_call_count": 0,
                    "solve_call_count": 0,
                    "target_backward_count": 0,
                    "shared_z_fork_replay_count": int(probe["shared_z_fork_replay_count"]),
                    "same_entry_activation_forward_count": 0,
                    "jvp_count": NOT_RECORDED,
                    "vjp_count": NOT_RECORDED,
                    "hvp_count": NOT_RECORDED,
                    "total_forward_count": NOT_RECORDED,
                    "edit_core_wall_seconds": 0.0,
                    "observer_wall_seconds": 0.0,
                    "gpu_peak_memory_bytes": NOT_RECORDED,
                }
            )
            for fork_name, fork in (
                ("ACCUMULATED_OR_STATIC", checkpoint["probe"]["primary"]),
                ("RESET_CACHE", checkpoint["probe"].get("reset_cache_fork")),
            ):
                if fork is None:
                    continue
                ordered = fork["ordered"]
                ordered_observer = ordered["layer_realization_observer"]
                ordered_audit = ordered["official_call_audit"]
                ordered_overhead = ordered_observer["overhead_wall_seconds"]
                compute.append(
                    {
                        "model": model,
                        "method": method,
                        "stage": f"CHECKPOINT_ORDERED_{fork_name}",
                        "batch_index": int(checkpoint["batch_index"]),
                        "accepted_edit_count": accepted,
                        "request_denominator": int(ordered_observer["request_count"]),
                        "direct_z_compute_count": int(ordered_observer["direct_z_compute_count"]),
                        "direct_z_recompute_count": int(ordered_observer["direct_z_recompute_count"]),
                        "layer_observation_count": int(ordered_observer["layer_loop_observation_copy_count"]),
                        "terminal_forward_count": int(ordered_observer["terminal_post_L8_forward_count"]),
                        "compute_ks_call_count": int(ordered_audit["compute_ks_call_count"]),
                        "solve_call_count": int(ordered_audit["torch_linalg_solve_call_count"]),
                        "target_backward_count": int(ordered["target_backward_count"]),
                        "shared_z_fork_replay_count": int(ordered_observer["direct_z_shared_replay_count"]),
                        "same_entry_activation_forward_count": int(fork["same_entry"]["activation_forward_count"]),
                        "jvp_count": NOT_RECORDED,
                        "vjp_count": NOT_RECORDED,
                        "hvp_count": NOT_RECORDED,
                        "total_forward_count": NOT_RECORDED,
                        "edit_core_wall_seconds": float(ordered["edit_core_wall_seconds"]),
                        "observer_wall_seconds": float(sum(float(value) for value in ordered_overhead.values())),
                        "gpu_peak_memory_bytes": NOT_RECORDED,
                    }
                )
                req, layers, weights = _observer_rows(
                    model=model,
                    method=method,
                    stage="CHECKPOINT_SENTINEL",
                    batch_index=int(checkpoint["batch_index"]),
                    checkpoint=accepted,
                    fork=fork_name,
                    observer=ordered["layer_realization_observer"],
                )
                _attach_checkpoint_action_realization(req, layers, weights)
                checkpoint_request.extend(req)
                checkpoint_layer.extend(layers)
                checkpoint_weight.extend(weights)
                geometry = fork["completion_geometry"]
                checkpoint_geometry.append(
                    {
                        "model": model,
                        "method": method,
                        "accepted_edit_count": accepted,
                        "batch_index": int(checkpoint["batch_index"]),
                        "fork": fork_name,
                        "history_state": str(fork["history_state"]),
                        "A0": float(geometry["A0"]),
                        "V_to_go": float(geometry["V_to_go"]),
                        "Vbar": float(geometry["Vbar"]),
                        "minimum_completion_budget_h_normalized": float(geometry["minimum_completion_budget_h_normalized"]),
                        "unreachable_fraction": float(geometry["unreachable_fraction"]),
                        "authority": str(geometry["authority"]),
                        "full_parameter_space_claim": int(bool(geometry["full_parameter_space_claim"])),
                    }
                )
                for waypoint in geometry["waypoints"]:
                    checkpoint_waypoint.append(
                        {
                            "model": model,
                            "method": method,
                            "accepted_edit_count": accepted,
                            "batch_index": int(checkpoint["batch_index"]),
                            "fork": fork_name,
                            **{
                                key: value
                                for key, value in waypoint.items()
                                if key
                                in {
                                    "waypoint",
                                    "remaining_layer_count",
                                    "A0",
                                    "A_committed",
                                    "Abar_committed",
                                    "V_to_go",
                                    "Vbar",
                                    "actual_official_tail_action",
                                    "completion_budget_h",
                                    "completion_budget_h_normalized",
                                    "effective_rank",
                                    "minimum_effective_singular_value",
                                    "unreachable_fraction",
                                }
                            },
                        }
                    )
                same = fork["same_entry"]
                for row in same["layers"]:
                    checkpoint_same_entry.append(
                        {
                            "model": model,
                            "method": method,
                            "accepted_edit_count": accepted,
                            "batch_index": int(checkpoint["batch_index"]),
                            "fork": fork_name,
                            "layer": int(row["layer"]),
                            "delta_frobenius_magnitude": float(row["delta_frobenius_magnitude"]),
                            "delta_frobenius_squared": float(row["delta_frobenius_squared"]),
                            "response_frobenius_magnitude": float(row["response_frobenius_magnitude"]),
                            "additional_compute_z": int(same["additional_compute_z"]),
                            "additional_key_compute": int(same["additional_key_compute"]),
                            "additional_solve": int(same["additional_solve"]),
                            "weight_restore_exact": int(bool(same["restore"]["weights"]["exact"])),
                            "cache_restore_exact": int(bool(same["restore"]["cache"]["exact"])),
                        }
                    )

    provisional = AnalysisTables(
        production_request,
        production_layer,
        production_weight,
        production_batch,
        checkpoint_request,
        checkpoint_layer,
        checkpoint_weight,
        checkpoint_geometry,
        checkpoint_waypoint,
        checkpoint_same_entry,
        functional,
        compute,
        [],
        [],
        [],
        [],
    )
    provisional.summary = build_summaries(provisional)
    provisional.paired_delta = build_paired_deltas(provisional)
    provisional.association = build_associations(provisional)
    provisional.outlier = build_outliers(provisional)
    return provisional


def _summarize_groups(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    metrics: Sequence[str],
    *,
    table: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for group, selected in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        for metric in metrics:
            values = [float(row[metric]) for row in selected if isinstance(row.get(metric), (int, float))]
            if not values:
                output.append({"source_table": table, **dict(zip(keys, group)), "metric": metric, "availability": NOT_RECORDED})
            else:
                output.append({"source_table": table, **dict(zip(keys, group)), "metric": metric, "availability": "RECORDED", **summary(values)})
    return output


def build_summaries(tables: AnalysisTables) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    q_metrics = [f"q_{label.replace('-', '_')}" for label in Q_LABELS]
    ideal_metrics = [f"ideal_abs_deviation_{label.replace('-', '_')}" for label in Q_LABELS]
    request_metrics = [*q_metrics, *ideal_metrics, "q_pre_L8_gt_0_2", "d_parallel", "d_perp", "mean_rho", "mean_tau", "recurrence_closure_relative_error", "D_TV", "delta_center", "target_new_nll", "target_true_nll", "target_new_margin", "target_true_margin", "target_new_strict", "target_true_strict"]
    output.extend(_summarize_groups(tables.production_request, ("model", "method"), request_metrics, table="production_request"))
    output.extend(_summarize_groups(tables.production_request, ("model", "method", "batch_index"), request_metrics, table="production_request_by_batch"))
    layer_metrics = ["allocation_norm_over_R1", "realized_reduction_norm_over_R1", "gap_norm_over_R1", "rho", "tau", "Y_parallel_over_R1", "Y_perp_over_R1", "E_parallel_over_R1", "E_perp_magnitude_over_R1"]
    output.extend(_summarize_groups(tables.production_layer, ("model", "method", "layer"), layer_metrics, table="production_layer"))
    output.extend(_summarize_groups(tables.production_weight, ("model", "method", "layer"), ("frobenius_magnitude", "frobenius_squared_telemetry", "relative_update_magnitude", "update_magnitude_share"), table="production_weight_batch_unit"))
    output.extend(
        _summarize_groups(
            tables.checkpoint_request,
            ("model", "method", "accepted_edit_count", "fork"),
            [
                *q_metrics,
                "q_pre_L8_gt_0_2",
                "d_parallel",
                "d_perp",
                "mean_rho",
                "mean_tau",
                "recurrence_closure_relative_error",
                "D_TV",
                "delta_center",
                "negative_progress_count",
                "negative_progress_total",
                "positive_progress_total",
            ],
            table="checkpoint_request",
        )
    )
    output.extend(_summarize_groups(tables.checkpoint_layer, ("model", "method", "accepted_edit_count", "fork", "layer"), layer_metrics, table="checkpoint_layer"))
    output.extend(_summarize_groups(tables.checkpoint_weight, ("model", "method", "accepted_edit_count", "fork", "layer"), ("frobenius_magnitude", "frobenius_squared_telemetry", "relative_update_magnitude", "update_magnitude_share"), table="checkpoint_weight"))
    output.extend(_summarize_groups(tables.checkpoint_same_entry, ("model", "method", "accepted_edit_count", "fork", "layer"), ("delta_frobenius_magnitude", "delta_frobenius_squared", "response_frobenius_magnitude"), table="checkpoint_same_entry"))
    functional_recorded = [row for row in tables.functional if row["availability"] == "RECORDED"]
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in functional_recorded:
        groups[(row["model"], row["method"], row["accepted_edit_count"], row["cohort"], row["metric"])].append(row)
    for (model, method, accepted, cohort, panel_metric), selected in sorted(groups.items()):
        for measured in ("prompt_count", "strict_count", "nll_mean", "margin_mean", "strict_rate"):
            output.append(
                {
                    "source_table": "functional",
                    "model": model,
                    "method": method,
                    "accepted_edit_count": accepted,
                    "cohort": cohort,
                    "panel_metric": panel_metric,
                    "metric": measured,
                    "availability": "RECORDED",
                    **summary(float(row[measured]) for row in selected),
                }
            )
    return output


def _paired(
    rows: Sequence[Mapping[str, Any]],
    left_filter: Callable[[Mapping[str, Any]], bool],
    right_filter: Callable[[Mapping[str, Any]], bool],
    key_fields: Sequence[str],
    metrics: Sequence[str],
    comparison: str,
    cluster_field: str | None = None,
) -> list[dict[str, Any]]:
    left = {tuple(row[key] for key in key_fields): row for row in rows if left_filter(row)}
    right = {tuple(row[key] for key in key_fields): row for row in rows if right_filter(row)}
    common = sorted(set(left) & set(right), key=lambda value: tuple(str(item) for item in value))
    output: list[dict[str, Any]] = []
    for metric in metrics:
        keyed_deltas = [
            (key, float(left[key][metric]) - float(right[key][metric]))
            for key in common
            if isinstance(left[key].get(metric), (int, float))
            and isinstance(right[key].get(metric), (int, float))
        ]
        deltas = [value for _, value in keyed_deltas]
        row: dict[str, Any] = {"comparison": comparison, "metric": metric, "paired_n": len(deltas)}
        row.update(summary(deltas) if deltas else {"availability": NOT_RECORDED})
        if deltas:
            seed = int.from_bytes(hashlib.sha256(f"{comparison}:{metric}".encode()).digest()[:8], "big")
            generator = np.random.default_rng(seed)
            if cluster_field is None:
                values = np.asarray(deltas, dtype=np.float64)
                bootstrap = np.asarray(
                    [np.mean(generator.choice(values, size=values.size, replace=True)) for _ in range(2000)],
                    dtype=np.float64,
                )
                uncertainty_unit = "PAIRED_REQUEST_HASH"
                cluster_n = len(values)
            else:
                if cluster_field not in key_fields:
                    raise AnalysisBoundary(f"cluster field {cluster_field} absent from paired key")
                offset = key_fields.index(cluster_field)
                clusters: dict[Any, list[float]] = defaultdict(list)
                for key, value in keyed_deltas:
                    clusters[key[offset]].append(value)
                labels = sorted(clusters, key=str)
                bootstrap_values = []
                for _ in range(2000):
                    selected_labels = generator.choice(labels, size=len(labels), replace=True)
                    sample = [value for label in selected_labels for value in clusters[label]]
                    bootstrap_values.append(float(np.mean(sample)))
                bootstrap = np.asarray(bootstrap_values, dtype=np.float64)
                uncertainty_unit = f"PAIRED_{cluster_field.upper()}_CLUSTER"
                cluster_n = len(labels)
            row["paired_mean_bootstrap95_low"] = float(np.percentile(bootstrap, 2.5))
            row["paired_mean_bootstrap95_high"] = float(np.percentile(bootstrap, 97.5))
            row["uncertainty_unit"] = uncertainty_unit
            row["uncertainty_cluster_n"] = cluster_n
            row["bootstrap_seed"] = seed
            row["bootstrap_resamples"] = 2000
        output.append(row)
    return output


def build_paired_deltas(tables: AnalysisTables) -> list[dict[str, Any]]:
    metrics = ["q_pre_L8", "q_post_L8", "d_parallel", "d_perp", "mean_rho", "mean_tau", "target_new_nll", "target_true_nll"]
    output: list[dict[str, Any]] = []
    for model in ("llama3-8b-inst", "qwen2.5-7b-inst"):
        output.extend(
            _paired(
                tables.production_request,
                lambda row, model=model: row["model"] == model and row["method"] == "alphaedit",
                lambda row, model=model: row["model"] == model and row["method"] == "memit",
                ("batch_index", "request_sha256"),
                metrics,
                f"{model}:alphaedit-minus-memit:production",
                cluster_field="batch_index",
            )
        )
    for method in ("alphaedit", "memit"):
        output.extend(
            _paired(
                tables.production_request,
                lambda row, method=method: row["method"] == method and row["model"] == "llama3-8b-inst",
                lambda row, method=method: row["method"] == method and row["model"] == "qwen2.5-7b-inst",
                ("batch_index", "request_sha256"),
                metrics,
                f"{method}:llama-minus-qwen:production",
                cluster_field="batch_index",
            )
        )
    checkpoint_metrics = [
        "q_pre_L8",
        "q_post_L8",
        "d_parallel",
        "d_perp",
        "mean_rho",
        "mean_tau",
        "D_TV",
        "delta_center",
    ]
    for accepted in (0, 1000, 1500, 2000, 3000, 5000, 7500, 10000):
        for model in ("llama3-8b-inst", "qwen2.5-7b-inst"):
            output.extend(
                _paired(
                    tables.checkpoint_request,
                    lambda row, model=model, accepted=accepted: row["model"] == model and row["method"] == "alphaedit" and row["accepted_edit_count"] == accepted and row["fork"] == "ACCUMULATED_OR_STATIC",
                    lambda row, model=model, accepted=accepted: row["model"] == model and row["method"] == "memit" and row["accepted_edit_count"] == accepted and row["fork"] == "ACCUMULATED_OR_STATIC",
                    ("request_sha256",),
                    checkpoint_metrics,
                    f"{model}:alphaedit-minus-memit:sentinel-{accepted}",
                )
            )
        for method in ("alphaedit", "memit"):
            output.extend(
                _paired(
                    tables.checkpoint_request,
                    lambda row, method=method, accepted=accepted: row["method"] == method and row["model"] == "llama3-8b-inst" and row["accepted_edit_count"] == accepted and row["fork"] == "ACCUMULATED_OR_STATIC",
                    lambda row, method=method, accepted=accepted: row["method"] == method and row["model"] == "qwen2.5-7b-inst" and row["accepted_edit_count"] == accepted and row["fork"] == "ACCUMULATED_OR_STATIC",
                    ("request_sha256",),
                    checkpoint_metrics,
                    f"{method}:llama-minus-qwen:sentinel-{accepted}",
                )
            )
    for model, method in CELL_ORDER:
        output.extend(
            _paired(
                tables.checkpoint_request,
                lambda row, model=model, method=method: row["model"] == model and row["method"] == method and row["accepted_edit_count"] == 10000 and row["fork"] == "ACCUMULATED_OR_STATIC",
                lambda row, model=model, method=method: row["model"] == model and row["method"] == method and row["accepted_edit_count"] == 0 and row["fork"] == "ACCUMULATED_OR_STATIC",
                ("request_sha256",),
                checkpoint_metrics,
                f"{model}:{method}:sentinel-10k-minus-t0",
            )
        )
    for model in ("llama3-8b-inst", "qwen2.5-7b-inst"):
        for accepted in (0, 1000, 1500, 2000, 3000, 5000, 7500, 10000):
            output.extend(
                _paired(
                    tables.checkpoint_request,
                    lambda row, model=model, accepted=accepted: row["model"] == model and row["method"] == "alphaedit" and row["accepted_edit_count"] == accepted and row["fork"] == "RESET_CACHE",
                    lambda row, model=model, accepted=accepted: row["model"] == model and row["method"] == "alphaedit" and row["accepted_edit_count"] == accepted and row["fork"] == "ACCUMULATED_OR_STATIC",
                    ("request_sha256",),
                    checkpoint_metrics,
                    f"{model}:alphaedit-reset-minus-accumulated:sentinel-{accepted}",
                )
            )
    # Checkpoint functional panels are paired only within the same checkpoint,
    # cohort, panel metric and request hash.  Current cohorts at different
    # times are deliberately never paired.
    recorded_functional = [row for row in tables.functional if row["availability"] == "RECORDED"]
    accepted_values = sorted({int(row["accepted_edit_count"]) for row in recorded_functional})
    for accepted in accepted_values:
        cohorts = sorted({str(row["cohort"]) for row in recorded_functional if int(row["accepted_edit_count"]) == accepted})
        for cohort in cohorts:
            panel_metrics = sorted({str(row["metric"]) for row in recorded_functional if int(row["accepted_edit_count"]) == accepted and row["cohort"] == cohort})
            for panel_metric in panel_metrics:
                def matches(row: Mapping[str, Any]) -> bool:
                    return (
                        int(row["accepted_edit_count"]) == accepted
                        and row["cohort"] == cohort
                        and row["metric"] == panel_metric
                        and row["availability"] == "RECORDED"
                    )

                for model in ("llama3-8b-inst", "qwen2.5-7b-inst"):
                    output.extend(
                        _paired(
                            recorded_functional,
                            lambda row, model=model: matches(row) and row["model"] == model and row["method"] == "alphaedit",
                            lambda row, model=model: matches(row) and row["model"] == model and row["method"] == "memit",
                            ("request_sha256",),
                            ("nll_mean", "margin_mean", "strict_rate"),
                            f"{model}:alphaedit-minus-memit:functional-{accepted}:{cohort}:{panel_metric}",
                        )
                    )
                for method in ("alphaedit", "memit"):
                    output.extend(
                        _paired(
                            recorded_functional,
                            lambda row, method=method: matches(row) and row["model"] == "llama3-8b-inst" and row["method"] == method,
                            lambda row, method=method: matches(row) and row["model"] == "qwen2.5-7b-inst" and row["method"] == method,
                            ("request_sha256",),
                            ("nll_mean", "margin_mean", "strict_rate"),
                            f"{method}:llama-minus-qwen:functional-{accepted}:{cohort}:{panel_metric}",
                        )
                    )
    return output


def build_associations(tables: AnalysisTables) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    pairs = (
        ("q_post_L8", "target_new_nll"),
        ("q_pre_L8", "target_new_nll"),
        ("d_parallel", "target_new_nll"),
        ("d_perp", "target_new_nll"),
        ("mean_rho", "target_new_nll"),
        ("mean_tau", "target_new_nll"),
        ("D_TV", "target_new_nll"),
        ("delta_center", "target_new_nll"),
        ("q_post_L8", "target_new_margin"),
    )
    for model, method in CELL_ORDER:
        selected = [row for row in tables.production_request if row["model"] == model and row["method"] == method]
        for x_name, y_name in pairs:
            by_batch: dict[int, list[tuple[float, float]]] = defaultdict(list)
            for row in selected:
                if isinstance(row.get(x_name), (int, float)) and isinstance(row.get(y_name), (int, float)):
                    by_batch[int(row["batch_index"])].append((float(row[x_name]), float(row[y_name])))
            cluster_values = [
                value
                for batch in sorted(by_batch)
                if isinstance(
                    (value := spearman([row[0] for row in by_batch[batch]], [row[1] for row in by_batch[batch]])),
                    float,
                )
            ]
            cluster_stats = summary(cluster_values)
            seed = int.from_bytes(hashlib.sha256(f"assoc:{model}:{method}:{x_name}:{y_name}".encode()).digest()[:8], "big")
            generator = np.random.default_rng(seed)
            array = np.asarray(cluster_values, dtype=np.float64)
            bootstrap = np.asarray([np.mean(generator.choice(array, size=array.size, replace=True)) for _ in range(2000)])
            output.append(
                {
                    "model": model,
                    "method": method,
                    "unit": "B100_CLUSTERED_REQUEST_DESCRIPTIVE",
                    "x": x_name,
                    "y": y_name,
                    "request_n": sum(len(value) for value in by_batch.values()),
                    "batch_cluster_n": len(cluster_values),
                    "batch_spearman_mean": cluster_stats["mean"],
                    "batch_spearman_median": cluster_stats["median"],
                    "batch_spearman_iqr": cluster_stats["iqr"],
                    "batch_spearman_p90": cluster_stats["p90"],
                    "cluster_bootstrap95_low": float(np.percentile(bootstrap, 2.5)),
                    "cluster_bootstrap95_high": float(np.percentile(bootstrap, 97.5)),
                    "bootstrap_seed": seed,
                    "bootstrap_resamples": 2000,
                    "causal_claim": 0,
                }
            )
        # Across-batch drift is kept separate from within-B100 request
        # association.  This descriptive statistic is chronologically
        # confounded and therefore never labelled causal.
        request_by_batch: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        weight_by_batch: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in selected:
            request_by_batch[int(row["batch_index"])].append(row)
        for row in tables.production_weight:
            if row["model"] == model and row["method"] == method:
                weight_by_batch[int(row["batch_index"])].append(row)
        batch_rows = []
        for batch in sorted(request_by_batch):
            requests = request_by_batch[batch]
            weights = weight_by_batch[batch]
            batch_rows.append(
                {
                    "q_post_L8": float(np.median([row["q_post_L8"] for row in requests])),
                    "D_TV": float(np.median([row["D_TV"] for row in requests])),
                    "target_new_nll": float(np.median([row["target_new_nll"] for row in requests])),
                    "total_update_magnitude": float(sum(row["frobenius_magnitude"] for row in weights)),
                }
            )
        for x_name in ("q_post_L8", "D_TV", "total_update_magnitude"):
            estimate = spearman(
                [row[x_name] for row in batch_rows],
                [row["target_new_nll"] for row in batch_rows],
            )
            seed = int.from_bytes(hashlib.sha256(f"batch-assoc:{model}:{method}:{x_name}".encode()).digest()[:8], "big")
            generator = np.random.default_rng(seed)
            bootstrap_values: list[float] = []
            indices = np.arange(len(batch_rows))
            for _ in range(2000):
                chosen = generator.choice(indices, size=len(indices), replace=True)
                value = spearman(
                    [batch_rows[index][x_name] for index in chosen],
                    [batch_rows[index]["target_new_nll"] for index in chosen],
                )
                if isinstance(value, float):
                    bootstrap_values.append(value)
            output.append(
                {
                    "model": model,
                    "method": method,
                    "unit": "B100_BATCH_TIME_SERIES_DESCRIPTIVE",
                    "x": x_name,
                    "y": "batch_median_target_new_nll",
                    "request_n": 10_000,
                    "batch_cluster_n": len(batch_rows),
                    "batch_spearman_mean": estimate,
                    "batch_spearman_median": estimate,
                    "batch_spearman_iqr": 0.0 if isinstance(estimate, float) else NOT_RECORDED,
                    "batch_spearman_p90": estimate,
                    "cluster_bootstrap95_low": float(np.percentile(bootstrap_values, 2.5)),
                    "cluster_bootstrap95_high": float(np.percentile(bootstrap_values, 97.5)),
                    "bootstrap_seed": seed,
                    "bootstrap_resamples": 2000,
                    "causal_claim": 0,
                    "chronological_edit_count_confounding": 1,
                }
            )
    return output


def build_outliers(tables: AnalysisTables) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metrics = ("q_pre_L8", "q_post_L8", "d_parallel", "d_perp", "recurrence_closure_relative_error", "target_new_nll", "target_true_nll")
    for model, method in CELL_ORDER:
        selected = [row for row in tables.production_request if row["model"] == model and row["method"] == method]
        for metric in metrics:
            values = [row for row in selected if isinstance(row.get(metric), (int, float))]
            for rank, row in enumerate(sorted(values, key=lambda value: float(value[metric]), reverse=True)[:10], start=1):
                output.append(
                    {
                        "model": model,
                        "method": method,
                        "metric": metric,
                        "rank_descending": rank,
                        "value": float(row[metric]),
                        "batch_index": int(row["batch_index"]),
                        "request_sha256": row["request_sha256"],
                        "case_identity_sha256": row["case_identity_sha256"],
                        "recurrence_closure_relative_error": float(row["recurrence_closure_relative_error"]),
                        "identity_valid": 1,
                        "nonfinite": 0,
                    }
                )
        terminal = [row for row in tables.checkpoint_request if row["model"] == model and row["method"] == method and row["accepted_edit_count"] == 10000 and row["fork"] == "ACCUMULATED_OR_STATIC"]
        for metric in ("q_pre_L8", "q_post_L8", "d_parallel", "d_perp", "recurrence_closure_relative_error"):
            for rank, row in enumerate(sorted(terminal, key=lambda value: float(value[metric]), reverse=True)[:10], start=1):
                output.append(
                    {
                        "model": model,
                        "method": method,
                        "metric": f"terminal_sentinel_{metric}",
                        "rank_descending": rank,
                        "value": float(row[metric]),
                        "batch_index": 100,
                        "request_sha256": row["request_sha256"],
                        "case_identity_sha256": row["case_identity_sha256"],
                        "recurrence_closure_relative_error": float(row["recurrence_closure_relative_error"]),
                        "identity_valid": 1,
                        "nonfinite": 0,
                    }
                )
    return output


__all__ = [
    "AnalysisTables",
    "NOT_RECORDED",
    "build_associations",
    "build_outliers",
    "build_paired_deltas",
    "build_summaries",
    "extract_tables",
    "spearman",
    "summary",
]
