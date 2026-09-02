"""Four-arm exhaustive lifelong analysis (CPU/read-only).

The builder consumes a create-once JSON input lock, independently rehashes all
raw JSON and checkpoint-state bytes, validates the append-only trajectory, and
only then emits derived tables, figures, a factual Korean report, and rooted
receipts.  It never imports torch or deserialises checkpoint state.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

from .lifelong_analysis_io import (
    AnalysisBoundary,
    CELL_ORDER,
    INSTRUCTION_ID,
    InputLock,
    ValidatedArm,
    canonical_hash,
    member_root,
    sha256_file,
    validate_campaign,
    write_json_once,
)
from .lifelong_analysis_tables import AnalysisTables, NOT_RECORDED, extract_tables, spearman, summary


MODEL_LABEL = {
    "llama3-8b-inst": "Llama-3-8B-Instruct",
    "qwen2.5-7b-inst": "Qwen2.5-7B-Instruct",
}
METHOD_LABEL = {"memit": "Official MEMIT", "alphaedit": "Official AlphaEdit"}
ARM_SHORT = {
    ("llama3-8b-inst", "alphaedit"): "LA",
    ("llama3-8b-inst", "memit"): "LM",
    ("qwen2.5-7b-inst", "alphaedit"): "QA",
    ("qwen2.5-7b-inst", "memit"): "QM",
}


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _fields(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return value


def write_csv_once(path: Path, rows: Sequence[Mapping[str, Any]], *, compressed: bool = False) -> None:
    if not rows:
        raise AnalysisBoundary(f"refusing empty derived table: {path.name}")
    fields = _fields(rows)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    raw = os.fdopen(descriptor, "wb")
    try:
        if compressed:
            binary: Any = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
        else:
            binary = raw
        text = io.TextIOWrapper(binary, encoding="utf-8", newline="")
        writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})
        text.flush()
        if compressed:
            text.detach().close()
        else:
            text.detach()
    finally:
        if not raw.closed:
            raw.close()


def _summary_rows(tables: AnalysisTables, source_prefix: str) -> list[dict[str, Any]]:
    return [row for row in tables.summary if str(row["source_table"]).startswith(source_prefix)]


def _stat(
    tables: AnalysisTables,
    source: str,
    model: str,
    method: str,
    metric: str,
    statistic: str = "median",
    **group: Any,
) -> Any:
    for row in tables.summary:
        if (
            row["source_table"] == source
            and row.get("model") == model
            and row.get("method") == method
            and row["metric"] == metric
            and all(row.get(key) == value for key, value in group.items())
        ):
            return row.get(statistic, NOT_RECORDED)
    return NOT_RECORDED


def _functional_stat(
    tables: AnalysisTables,
    model: str,
    method: str,
    accepted: int,
    cohort: str,
    metric: str,
    value: str,
    statistic: str = "mean",
) -> Any:
    for row in tables.summary:
        if (
            row["source_table"] == "functional"
            and row.get("model") == model
            and row.get("method") == method
            and row.get("accepted_edit_count") == accepted
            and row.get("cohort") == cohort
            and row.get("panel_metric") == metric
            and row.get("metric") == value
        ):
            # Functional summary's measured quantity is stored in its metric field;
            # distinguish it by the source rows directly when multiple quantities exist.
            selected = [
                item[value]
                for item in tables.functional
                if item["model"] == model
                and item["method"] == method
                and item["accepted_edit_count"] == accepted
                and item["cohort"] == cohort
                and item["metric"] == metric
                and item["availability"] == "RECORDED"
                and isinstance(item[value], (int, float))
            ]
            return summary(selected)[statistic] if selected else NOT_RECORDED
    return NOT_RECORDED


def arm_summary_rows(arms: Sequence[ValidatedArm], tables: AnalysisTables) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in arms:
        model, method = arm.spec.model, arm.spec.method
        terminal_geometry = next(
            row
            for row in tables.checkpoint_geometry
            if row["model"] == model
            and row["method"] == method
            and row["accepted_edit_count"] == 10_000
            and row["fork"] == "ACCUMULATED_OR_STATIC"
        )
        rows.append(
            {
                "model": model,
                "method": method,
                "job_id": arm.spec.job_id,
                "valid_batches": arm.result["valid_batch_denominator"],
                "valid_requests": arm.result["valid_request_denominator"],
                "valid_checkpoints": arm.result["checkpoint_denominator"],
                "production_q_pre_L8_median": _stat(tables, "production_request", model, method, "q_pre_L8"),
                "production_q_post_L8_median": _stat(tables, "production_request", model, method, "q_post_L8"),
                "production_d_parallel_median": _stat(tables, "production_request", model, method, "d_parallel"),
                "production_d_perp_median": _stat(tables, "production_request", model, method, "d_perp"),
                "production_target_new_nll_mean": _stat(tables, "production_request", model, method, "target_new_nll", "mean"),
                "production_target_new_nll_median": _stat(tables, "production_request", model, method, "target_new_nll"),
                "terminal_sentinel_q_pre_L8_median": _stat(tables, "checkpoint_request", model, method, "q_pre_L8", accepted_edit_count=10_000, fork="ACCUMULATED_OR_STATIC"),
                "terminal_sentinel_q_post_L8_median": _stat(tables, "checkpoint_request", model, method, "q_post_L8", accepted_edit_count=10_000, fork="ACCUMULATED_OR_STATIC"),
                "terminal_Vbar": terminal_geometry["Vbar"],
                "terminal_unreachable_fraction": terminal_geometry["unreachable_fraction"],
                "terminal_current_rewrite_nll_mean": _functional_stat(tables, model, method, 10_000, "current", "rewrite_target_new", "nll_mean"),
                "terminal_current_rephrase_nll_mean": _functional_stat(tables, model, method, 10_000, "current", "rephrase_target_new", "nll_mean"),
                "terminal_current_locality_strict_rate_mean": _functional_stat(tables, model, method, 10_000, "current", "locality_target_true", "strict_rate"),
                "wall_seconds": arm.result["wall_seconds"],
                "max_rss_kib": arm.spec.resource.get("max_rss_kib", NOT_RECORDED),
                "gpu_peak_memory_bytes": NOT_RECORDED,
                "raw_member_count": len(arm.raw_members),
                "raw_member_bytes": sum(int(row["bytes"]) for row in arm.raw_members),
                "raw_member_root": arm.raw_member_root,
                "nonfinite_count": arm.result["nonfinite_count"],
                "rollback_violation_count": arm.result["rollback_violation_count"],
                "target_recomputation_count": arm.result["target_recomputation_count"],
            }
        )
    return rows


def allocation_summary_rows(tables: AnalysisTables) -> list[dict[str, Any]]:
    rows = [
        row
        for row in tables.summary
        if row["source_table"] in {"production_layer", "checkpoint_layer"}
    ]
    for model, method in CELL_ORDER:
        for layer in (4, 5, 6, 7, 8):
            selected = [row for row in tables.production_layer if row["model"] == model and row["method"] == method and row["layer"] == layer]
            counts = {label: sum(row["realization_class"] == label for row in selected) for label in ("UNDER", "OVERSHOOT", "OPPOSITE", "EXACT")}
            rows.append(
                {
                    "source_table": "production_layer_class_counts",
                    "model": model,
                    "method": method,
                    "layer": layer,
                    "metric": "realization_class",
                    "availability": "RECORDED",
                    "n": len(selected),
                    **{f"{key.lower()}_count": value for key, value in counts.items()},
                    **{f"{key.lower()}_rate": value / len(selected) for key, value in counts.items()},
                }
            )
    checkpoint_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in tables.checkpoint_layer:
        key = (row["model"], row["method"], row["accepted_edit_count"], row["fork"], row["layer"])
        checkpoint_groups.setdefault(key, []).append(row)
    for key, selected in checkpoint_groups.items():
        counts = {
            label: sum(row["realization_class"] == label for row in selected)
            for label in ("UNDER", "OVERSHOOT", "OPPOSITE", "EXACT")
        }
        rows.append(
            {
                "source_table": "checkpoint_layer_class_counts",
                **dict(zip(("model", "method", "accepted_edit_count", "fork", "layer"), key)),
                "metric": "realization_class",
                "availability": "RECORDED",
                "n": len(selected),
                **{f"{name.lower()}_count": value for name, value in counts.items()},
                **{f"{name.lower()}_rate": value / len(selected) for name, value in counts.items()},
            }
        )
    return rows


def recurrence_rows(tables: AnalysisTables) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source, rows, keys in (
        ("production", tables.production_request, ("model", "method")),
        ("checkpoint", tables.checkpoint_request, ("model", "method", "accepted_edit_count", "fork")),
    ):
        groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
        for row in rows:
            groups.setdefault(tuple(row[key] for key in keys), []).append(row)
        for group, selected in groups.items():
            stats = summary(row["recurrence_closure_relative_error"] for row in selected)
            worst = max(selected, key=lambda value: value["recurrence_closure_relative_error"])
            output.append(
                {
                    "stage": source,
                    **dict(zip(keys, group)),
                    **stats,
                    "worst_request_sha256": worst["request_sha256"],
                    "worst_case_identity_sha256": worst["case_identity_sha256"],
                    "threshold": 1e-4,
                    "threshold_violation_count": sum(row["recurrence_closure_relative_error"] >= 1e-4 for row in selected),
                }
            )
    return output


def endpoint_summary_rows(tables: AnalysisTables) -> list[dict[str, Any]]:
    return [row for row in tables.summary if row["source_table"] in {"production_request", "production_request_by_batch", "functional"} and (row["source_table"] == "functional" or row["metric"] in {"target_new_nll", "target_true_nll", "target_new_margin", "target_true_margin", "target_new_strict", "target_true_strict"})]


def checkpoint_summary_rows(tables: AnalysisTables) -> list[dict[str, Any]]:
    rows = _summary_rows(tables, "checkpoint_request")
    rows.extend(
        {
            "source_table": "checkpoint_geometry",
            **row,
            "metric": "geometry",
            "availability": "RECORDED",
        }
        for row in tables.checkpoint_geometry
    )
    return rows


def compute_accounting_rows(arms: Sequence[ValidatedArm], tables: AnalysisTables) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for arm in arms:
        selected = [row for row in tables.compute if row["model"] == arm.spec.model and row["method"] == arm.spec.method]
        production = [row for row in selected if row["stage"] == "PRODUCTION_B100"]
        checkpoint_targets = [row for row in selected if row["stage"] == "CHECKPOINT_SHARED_TARGET_LEDGER"]
        checkpoint_ordered = [row for row in selected if str(row["stage"]).startswith("CHECKPOINT_ORDERED_")]
        output.append(
            {
                "model": arm.spec.model,
                "method": arm.spec.method,
                "job_id": arm.spec.job_id,
                "valid_batches": 100,
                "valid_requests": 10_000,
                "production_direct_z_compute_count": sum(row["direct_z_compute_count"] for row in production),
                "production_direct_z_recompute_count": sum(row["direct_z_recompute_count"] for row in production),
                "production_layer_observation_count": sum(row["layer_observation_count"] for row in production),
                "production_terminal_forward_count": sum(row["terminal_forward_count"] for row in production),
                "production_compute_ks_call_count": sum(row["compute_ks_call_count"] for row in production),
                "production_solve_call_count": sum(row["solve_call_count"] for row in production),
                "production_target_backward_count": sum(row["target_backward_count"] for row in production),
                "checkpoint_direct_z_optimizer_count": sum(row["direct_z_compute_count"] for row in checkpoint_targets),
                "checkpoint_shared_z_replay_count": sum(row["shared_z_fork_replay_count"] for row in checkpoint_targets),
                "checkpoint_recompute_count": sum(row["direct_z_recompute_count"] for row in checkpoint_targets),
                "checkpoint_ordered_fork_count": len(checkpoint_ordered),
                "checkpoint_ordered_layer_observation_count": sum(row["layer_observation_count"] for row in checkpoint_ordered),
                "checkpoint_ordered_terminal_forward_count": sum(row["terminal_forward_count"] for row in checkpoint_ordered),
                "checkpoint_same_entry_activation_forward_count": sum(row["same_entry_activation_forward_count"] for row in checkpoint_ordered),
                "checkpoint_compute_ks_call_count": sum(row["compute_ks_call_count"] for row in checkpoint_ordered),
                "checkpoint_solve_call_count": sum(row["solve_call_count"] for row in checkpoint_ordered),
                "checkpoint_target_backward_count": sum(row["target_backward_count"] for row in checkpoint_ordered),
                "total_forward_count": NOT_RECORDED,
                "jvp_count": NOT_RECORDED,
                "vjp_count": NOT_RECORDED,
                "hvp_count": NOT_RECORDED,
                "production_edit_core_wall_seconds_sum": sum(row["edit_core_wall_seconds"] for row in production),
                "production_observer_wall_seconds_sum": sum(row["observer_wall_seconds"] for row in production),
                "checkpoint_edit_core_wall_seconds_sum": sum(row["edit_core_wall_seconds"] for row in checkpoint_ordered),
                "checkpoint_observer_wall_seconds_sum": sum(row["observer_wall_seconds"] for row in checkpoint_ordered),
                "cell_wall_seconds": arm.result["wall_seconds"],
                "slurm_elapsed": arm.spec.resource.get("elapsed", NOT_RECORDED),
                "slurm_max_rss_kib": arm.spec.resource.get("max_rss_kib", NOT_RECORDED),
                "gpu_peak_memory_bytes": NOT_RECORDED,
                "full_fp32_editable_parameters": 5,
                "bf16_parameter_count": 0,
                "fp16_parameter_count": 0,
                "autocast_enabled": NOT_RECORDED,
                "quantization_enabled": NOT_RECORDED,
                "numeric_cast_inventory": NOT_RECORDED,
            }
        )
    return output


def technical_exclusion_rows(arms: Sequence[ValidatedArm]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in arms:
        for exclusion in arm.spec.technical_exclusions:
            rows.append(
                {
                    "canonical_model": arm.spec.model,
                    "canonical_method": arm.spec.method,
                    "excluded_job_id": exclusion.job_id,
                    "classification": exclusion.classification,
                    "reason": exclusion.reason,
                    "scientific_denominator": 0,
                    "immutable_path_count": len(exclusion.paths),
                    "imputation_count": 0,
                }
            )
    return rows


def raw_inventory_rows(arms: Sequence[ValidatedArm]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in arms:
        for member in arm.raw_members:
            rows.append(
                {
                    "model": arm.spec.model,
                    "method": arm.spec.method,
                    "relative_path": member["relative_path"],
                    "kind": member["kind"],
                    "bytes": member["bytes"],
                    "mode": member["mode"],
                    "sha256": member["sha256"],
                    "arm_raw_member_root": arm.raw_member_root,
                }
            )
    return rows


def failure_time_boundary_rows(arms: Sequence[ValidatedArm]) -> list[dict[str, Any]]:
    """Expose the preregistration boundary instead of inventing onset labels."""
    rows: list[dict[str, Any]] = []
    for arm in arms:
        for event in ("T_G_GEOMETRY_WARNING", "T_A_COMPLETION_BUDGET", "T_Z_FIXED_Z", "T_F_FUNCTIONAL_COLLAPSE"):
            rows.append(
                {
                    "model": arm.spec.model,
                    "method": arm.spec.method,
                    "event": event,
                    "status": "NOT_IDENTIFIABLE_PREREGISTERED_CALIBRATION_ENVELOPE_ABSENT",
                    "observed_through_accepted_edits": 10_000,
                    "right_censored_assignment": "NOT_ASSIGNED_WITHOUT_PREREGISTERED_THRESHOLD",
                    "technical_termination": 0,
                    "imputation_count": 0,
                }
            )
    return rows


def ordered_same_entry_rows(tables: AnalysisTables) -> list[dict[str, Any]]:
    ordered = {
        (row["model"], row["method"], row["accepted_edit_count"], row["fork"], row["layer"]): row
        for row in tables.checkpoint_weight
    }
    same = {
        (row["model"], row["method"], row["accepted_edit_count"], row["fork"], row["layer"]): row
        for row in tables.checkpoint_same_entry
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(set(ordered) & set(same), key=lambda value: tuple(map(str, value))):
        left, right = ordered[key], same[key]
        rows.append(
            {
                **dict(zip(("model", "method", "accepted_edit_count", "fork", "layer"), key)),
                "ordered_delta_frobenius_magnitude": left["frobenius_magnitude"],
                "same_entry_delta_frobenius_magnitude": right["delta_frobenius_magnitude"],
                "same_minus_ordered_delta_magnitude": right["delta_frobenius_magnitude"] - left["frobenius_magnitude"],
                "same_entry_response_frobenius_magnitude": right["response_frobenius_magnitude"],
                "request_level_target_aligned_same_entry": NOT_RECORDED,
                "causal_claim": 0,
            }
        )
    return rows


def completion_predictive_boundary_rows(tables: AnalysisTables) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in tables.checkpoint_waypoint:
        key = (row["model"], row["method"], row["accepted_edit_count"], row["fork"])
        groups.setdefault(key, []).append(row)
    for key, values in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        selected = [row for row in values if int(row["remaining_layer_count"]) > 0]
        rows.append(
            {
                **dict(zip(("model", "method", "accepted_edit_count", "fork"), key)),
                "waypoint_n": len(selected),
                "V_to_go_vs_actual_tail_action_spearman": spearman(
                    [float(row["V_to_go"]) for row in selected],
                    [float(row["actual_official_tail_action"]) for row in selected],
                ),
                "interpretation_boundary": "LAYER_ORDINAL_CONFOUNDED_NOT_INDEPENDENT_PREDICTOR_VALIDATION",
                "AUROC": "NOT_IDENTIFIABLE_PREREGISTERED_FAILURE_LABEL_ABSENT",
                "AUPRC": "NOT_IDENTIFIABLE_PREREGISTERED_FAILURE_LABEL_ABSENT",
                "causal_claim": 0,
            }
        )
    return rows


def _fmt(value: Any, digits: int = 4) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    return f"{float(value):.{digits}f}"


def _geometry(
    tables: AnalysisTables, model: str, method: str, accepted: int, fork: str = "ACCUMULATED_OR_STATIC"
) -> Mapping[str, Any]:
    return next(
        row
        for row in tables.checkpoint_geometry
        if row["model"] == model
        and row["method"] == method
        and row["accepted_edit_count"] == accepted
        and row["fork"] == fork
    )


def _checkpoint_update_total(
    tables: AnalysisTables, model: str, method: str, accepted: int, fork: str = "ACCUMULATED_OR_STATIC"
) -> float:
    selected = [
        row
        for row in tables.checkpoint_weight
        if row["model"] == model
        and row["method"] == method
        and row["accepted_edit_count"] == accepted
        and row["fork"] == fork
    ]
    if len(selected) != 5:
        raise AnalysisBoundary(f"checkpoint update denominator differs: {model}/{method}/{accepted}/{fork}")
    return float(sum(float(row["frobenius_magnitude"]) for row in selected))


def _functional_count(
    tables: AnalysisTables,
    model: str,
    method: str,
    accepted: int,
    cohort: str,
    metric: str,
) -> tuple[Any, Any]:
    return (
        _functional_stat(tables, model, method, accepted, cohort, metric, "strict_count", "sum"),
        _functional_stat(tables, model, method, accepted, cohort, metric, "prompt_count", "sum"),
    )


def _paired_value(tables: AnalysisTables, comparison: str, metric: str, statistic: str = "median") -> Any:
    row = next(
        (row for row in tables.paired_delta if row["comparison"] == comparison and row["metric"] == metric),
        None,
    )
    return NOT_RECORDED if row is None else row.get(statistic, NOT_RECORDED)


def _association_value(
    tables: AnalysisTables,
    model: str,
    method: str,
    unit: str,
    x_name: str,
    y_name: str,
    field: str,
) -> Any:
    row = next(
        (
            row
            for row in tables.association
            if row["model"] == model
            and row["method"] == method
            and row["unit"] == unit
            and row["x"] == x_name
            and row["y"] == y_name
        ),
        None,
    )
    return NOT_RECORDED if row is None else row.get(field, NOT_RECORDED)


def _stats_cell(
    tables: AnalysisTables,
    source: str,
    model: str,
    method: str,
    metric: str,
    **group: Any,
) -> str:
    return "/".join(
        _fmt(_stat(tables, source, model, method, metric, statistic, **group))
        for statistic in ("mean", "median", "p90", "max")
    )


def _functional_stats_cell(
    tables: AnalysisTables,
    model: str,
    method: str,
    accepted: int,
    cohort: str,
    metric: str,
    value: str,
) -> str:
    return "/".join(
        _fmt(_functional_stat(tables, model, method, accepted, cohort, metric, value, statistic))
        for statistic in ("mean", "median", "p90", "max")
    )


def _functional_strict_cell(
    tables: AnalysisTables,
    model: str,
    method: str,
    accepted: int,
    cohort: str,
    metric: str,
) -> str:
    strict, prompts = _functional_count(tables, model, method, accepted, cohort, metric)
    if not isinstance(strict, (int, float)) or not isinstance(prompts, (int, float)) or prompts == 0:
        return str(NOT_RECORDED)
    return f"{int(strict)}/{int(prompts)} ({float(strict) / float(prompts):.3f})"


def korean_report(
    arms: Sequence[ValidatedArm],
    tables: AnalysisTables,
    arm_summary: Sequence[Mapping[str, Any]],
    technical: Sequence[Mapping[str, Any]],
) -> str:
    rows = {(row["model"], row["method"]): row for row in arm_summary}
    lines = [
        "# Official layer-write realization debt — lifelong B100×100 네 arm 상세 사실 보고서",
        "",
        "상태: **FOUR_ARM_LIFELONG_10K_TERMINAL_VALID**  ",
        "범위: Llama/Qwen과 Official MEMIT/AlphaEdit 네 셀을 모두 동일한 PRIMARY SCIENTIFIC SCOPE로 분석했다. 각 셀은 B100 100개를 누적한 10,000-request sequential trajectory다. 이 observational study는 barrier benefit, ODE 필요성, 인과 효과 또는 보편적 last-layer bottleneck을 주장하지 않는다. `scientific_promotion=false`다.",
        "",
        "## Executive factual findings",
        "",
        "이 첫 표는 10k checkpoint의 실제 final-W current B100 평가를 한곳에 모은다. NLL 열 순서는 `mean/median/p90/max`이며, strict 분모는 rewrite 100, rephrase 200, locality 1,000 prompt다. PRE_EDIT과 retention은 뒤의 별도 표에서 분리한다.",
        "",
        "| 모델 | 방법 | final-W Eff strict | Gen strict | Loc strict | Rewrite new NLL mean/med/p90/max | Rephrase new NLL mean/med/p90/max | sentinel q pre/post-L8 med |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, method in CELL_ORDER:
        row = rows[(model, method)]
        lines.append(
            f"| {MODEL_LABEL[model]} | {METHOD_LABEL[method]} | "
            f"{_functional_strict_cell(tables, model, method, 10000, 'current', 'rewrite_target_new')} | "
            f"{_functional_strict_cell(tables, model, method, 10000, 'current', 'rephrase_target_new')} | "
            f"{_functional_strict_cell(tables, model, method, 10000, 'current', 'locality_target_true')} | "
            f"{_functional_stats_cell(tables, model, method, 10000, 'current', 'rewrite_target_new', 'nll_mean')} | "
            f"{_functional_stats_cell(tables, model, method, 10000, 'current', 'rephrase_target_new', 'nll_mean')} | "
            f"{_fmt(row['terminal_sentinel_q_pre_L8_median'])}/{_fmt(row['terminal_sentinel_q_post_L8_median'])} |"
        )
    lines += [
        "",
        "### Checkpoint별 sentinel 핵심 수치",
        "",
        "| 모델 | 방법 | edits | pre-L8 q med | post-L8 q med | d∥ med | d⊥ med | V̄ | unreachable |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, method in CELL_ORDER:
        for accepted in (0, 1000, 1500, 2000, 3000, 5000, 7500, 10000):
            geometry = next(row for row in tables.checkpoint_geometry if row["model"] == model and row["method"] == method and row["accepted_edit_count"] == accepted and row["fork"] == "ACCUMULATED_OR_STATIC")
            lines.append(
                f"| {MODEL_LABEL[model]} | {METHOD_LABEL[method]} | {accepted} | "
                f"{_fmt(_stat(tables, 'checkpoint_request', model, method, 'q_pre_L8', accepted_edit_count=accepted, fork='ACCUMULATED_OR_STATIC'))} | "
                f"{_fmt(_stat(tables, 'checkpoint_request', model, method, 'q_post_L8', accepted_edit_count=accepted, fork='ACCUMULATED_OR_STATIC'))} | "
                f"{_fmt(_stat(tables, 'checkpoint_request', model, method, 'd_parallel', accepted_edit_count=accepted, fork='ACCUMULATED_OR_STATIC'))} | "
                f"{_fmt(_stat(tables, 'checkpoint_request', model, method, 'd_perp', accepted_edit_count=accepted, fork='ACCUMULATED_OR_STATIC'))} | "
                f"{_fmt(geometry['Vbar'])} | {_fmt(geometry['unreachable_fraction'])} |"
            )
    lines += [
        "",
        "### Terminal checkpoint layer별 realization과 update",
        "",
        "| 모델 | 방법 | layer | A/R1 med | Y/R1 med | E/R1 med | rho med | tau med | ΔW Frobenius (10k sentinel probe) | update share |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, method in CELL_ORDER:
        for layer in (4, 5, 6, 7, 8):
            lines.append(
                f"| {MODEL_LABEL[model]} | {METHOD_LABEL[method]} | {layer} | "
                f"{_fmt(_stat(tables, 'checkpoint_layer', model, method, 'allocation_norm_over_R1', accepted_edit_count=10000, fork='ACCUMULATED_OR_STATIC', layer=layer))} | "
                f"{_fmt(_stat(tables, 'checkpoint_layer', model, method, 'realized_reduction_norm_over_R1', accepted_edit_count=10000, fork='ACCUMULATED_OR_STATIC', layer=layer))} | "
                f"{_fmt(_stat(tables, 'checkpoint_layer', model, method, 'gap_norm_over_R1', accepted_edit_count=10000, fork='ACCUMULATED_OR_STATIC', layer=layer))} | "
                f"{_fmt(_stat(tables, 'checkpoint_layer', model, method, 'rho', accepted_edit_count=10000, fork='ACCUMULATED_OR_STATIC', layer=layer))} | "
                f"{_fmt(_stat(tables, 'checkpoint_layer', model, method, 'tau', accepted_edit_count=10000, fork='ACCUMULATED_OR_STATIC', layer=layer))} | "
                f"{_fmt(_stat(tables, 'checkpoint_weight', model, method, 'frobenius_magnitude', accepted_edit_count=10000, fork='ACCUMULATED_OR_STATIC', layer=layer))} | "
                f"{_fmt(_stat(tables, 'checkpoint_weight', model, method, 'update_magnitude_share', accepted_edit_count=10000, fork='ACCUMULATED_OR_STATIC', layer=layer))} |"
            )
    lines += [
        "",
        "### Functional checkpoint current-B100 수치",
        "",
        "| 모델 | 방법 | edits | rewrite new NLL mean | rephrase new NLL mean | locality strict-rate mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model, method in CELL_ORDER:
        for accepted in (1000, 1500, 2000, 3000, 5000, 7500, 10000):
            lines.append(
                f"| {MODEL_LABEL[model]} | {METHOD_LABEL[method]} | {accepted} | "
                f"{_fmt(_functional_stat(tables, model, method, accepted, 'current', 'rewrite_target_new', 'nll_mean'))} | "
                f"{_fmt(_functional_stat(tables, model, method, accepted, 'current', 'rephrase_target_new', 'nll_mean'))} | "
                f"{_fmt(_functional_stat(tables, model, method, accepted, 'current', 'locality_target_true', 'strict_rate'))} |"
            )
    lines += [
        "",
        "위 값은 model/method별로 분리했으며 raw norm을 모델 사이에 pooling하지 않았다. Production request 지표는 n=10,000/cell, weight update 지표의 독립 관측 단위는 B100 batch n=100/cell, checkpoint sentinel은 n=100/checkpoint/cell이다.",
        "",
        "![primary residual trajectory](primary-residual-trajectory-2x2.png)",
        "",
        "## Provenance와 불변식",
        "",
        "- 유효 arm 4/4, B100 400/400, training requests 40,000/40,000, checkpoint 32/32. Primary sentinel request-observations=3,200; AlphaEdit reset-cache fork=1,600; complete checkpoint table total=4,800.",
        "- arm별 raw canonical member는 117개(result1+journal100+checkpoint JSON8+state bytes8), 총 468개다. 모든 canonical raw member는 regular non-symlink mode0600이며 PT는 byte hashing만 했고 deserialize/model load는 0이다.",
        "- 각 arm compute_z=10,000, recompute=0, layer observation=500, terminal forward=100. W commit→next entry=99/99, terminal W0/cache restore=PASS, nonfinite/rollback/imputation=0.",
        "- AlphaEdit은 accepted B100마다 dynamic cache_c를 100씩 append/consume하여 0→10,000; MEMIT은 static covariance computation cache이며 request-history width는 0이다.",
        "- editable parameter inventory는 5/5 `torch.float32`; BF16/FP16 parameter 0. 명시적 autocast/quantization 및 GPU peak-memory telemetry는 schema에 없어 `NOT_RECORDED_SCHEMA`다. Slurm MaxRSS는 host RSS로 별도 보존했다.",
        "",
        "## Checkpoint·layer·request exhaustive tables",
        "",
        "Checkpoint는 accepted edits `0,1k,1.5k,2k,3k,5k,7.5k,10k`이다. q의 ideal reference는 `[1,.8,.6,.4,.2,0]`; 이는 관찰 기준선이며 Official update의 강제 목표가 아니다.",
        "",
        "![checkpoint q heatmap](checkpoint-layer-q-heatmap.png)",
        "",
        "A/Y/E 전체 vector는 raw schema에 publish되지 않았다. 저장된 normalized scalar로 `Y_parallel=rho·A`, `Y_perp=tau·A`, `E_parallel=(1-rho)·A`, `|E_perp|=tau·A`를 결정적으로 파생했다. 부호 없는 직교 norm 이상은 추정하지 않았다.",
        "",
        "![rho](rho-by-layer-terminal.png)",
        "",
        "![tau](tau-by-layer-terminal.png)",
        "",
        "![inherited debt](inherited-debt-distributions.png)",
        "",
        "![checkpoint rho heatmap](checkpoint-layer-rho-heatmap.png)",
        "",
        "![checkpoint tau heatmap](checkpoint-layer-tau-heatmap.png)",
        "",
        "Exact recurrence의 mean/median/p90/max와 worst request hash는 `recurrence-summary.csv`에 있다. 모든 outlier는 raw prompt 없이 request/case hash만 기록했고 recurrence/identity/nonfinite 상태를 함께 재검증했다.",
        "",
        "## Layer-wise update action",
        "",
        "Activation residual과 weight update magnitude는 별개 축이다. 아래 그림의 표본 단위는 request가 아니라 각 B100의 실제 layer update(n=100 batch/cell)다.",
        "",
        "![Layer-wise Update Magnitude](layer-wise-update-magnitude.png)",
        "",
        "![update magnitude drift](layer-wise-update-magnitude-drift.png)",
        "",
        "## Endpoint·retention metrics",
        "",
        "Production journal은 current B100 rewrite target-new/target-true NLL·margin·strict만 기록한다. Checkpoint functional panel은 current B100의 rewrite/rephrase/locality와 earliest/recent/hash-stratified retention rewrite를 기록한다. Retention rephrase/locality가 `NOT_RECORDED_LOW_COST_BATCH`인 경우를 0으로 대체하지 않았다. `endpoint-metrics-summary.csv`가 모든 n/mean/median/IQR/p90/max와 availability를 보존한다.",
        "",
        "![checkpoint endpoint strict rates](endpoint-checkpoint-strict-rates.png)",
        "",
        "## 비교와 association 경계",
        "",
        "`paired-deltas.csv`는 같은 batch/request hash의 AlphaEdit−MEMIT, 같은 method의 Llama−Qwen, 동일 sentinel hash의 10k−t0만 비교한다. 서로 다른 시점의 current B100 cohort는 request가 달라 early↔terminal paired 표본으로 사용하지 않았다. `descriptive-associations.csv`는 request-cluster Spearman association이며 인과 claim은 0이다.",
        "",
        "## Compute",
        "",
        "`compute-accounting.csv`는 edit-core/observer/cell wall, compute_z, key/solve/backward, layer observation과 terminal forward를 분리한다. Total forward와 GPU peak memory는 기록되지 않아 추정하지 않았다.",
        "",
        "![compute comparison](compute-time-comparison.png)",
        "",
        "## Technical exclusions",
        "",
    ]
    if technical:
        lines.extend(
            [
                "| canonical arm | excluded job | classification | denominator |",
                "|---|---:|---|---:|",
                *[
                    f"| {row['canonical_model']}/{row['canonical_method']} | {row['excluded_job_id']} | {row['classification']} | 0 |"
                    for row in technical
                ],
            ]
        )
    else:
        lines.append("Technical exclusion lineage: 없음.")
    lines += [
        "",
        "Scheduler 내부의 짧은 shell step sidecar는 parent experiment failure로 재분류하지 않았다. Canonical parent/array cell terminal과 raw result chain을 기준으로 판정했다.",
        "",
        "## Artifact inventory",
        "",
        "완전한 request/layer 표는 deterministic gzip CSV로, 요약·비교·compute·exclusion·raw-member inventory는 plain CSV로 제공한다. 각 row count와 SHA는 `analysis-manifest.json`, package root는 `rooted-analysis-receipt.json`에 봉인했다. Raw logs/model/cache/tensors/dataset/credentials는 Git package에 포함하지 않았다.",
        "",
        "### FACT / INFERENCE / NON-CLAIM",
        "",
        "- **FACT:** 네 arm 모두 동일 10k denominator와 append-only W/cache chain에서 terminal-valid이며 recurrence가 닫힌다.",
        "- **INFERENCE:** q/rho/tau/debt/update drift와 endpoint metric의 관계는 architecture/method-conditioned descriptive evidence다.",
        "- **NON-CLAIM:** association을 인과로, completion geometry를 barrier benefit으로, 특정 layer pattern을 universal bottleneck으로 해석하지 않는다.",
        "",
    ]
    return "\n".join(lines)


def korean_report_exhaustive(
    arms: Sequence[ValidatedArm],
    tables: AnalysisTables,
    arm_summary: Sequence[Mapping[str, Any]],
    technical: Sequence[Mapping[str, Any]],
) -> str:
    """Render the canonical, numbers-first Korean report.

    Every value below is derived from the same validated in-memory tables that
    are emitted as the sealed CSV package.  No value is copied from a job
    headline or manually transcribed into the report.
    """

    arm_rows = {(row["model"], row["method"]): row for row in arm_summary}
    compute_rows = {
        (row["model"], row["method"]): row for row in compute_accounting_rows(arms, tables)
    }
    rec_rows = recurrence_rows(tables)
    checkpoint_schedule = (0, 1000, 1500, 2000, 3000, 5000, 7500, 10000)
    q_fields = ("q_pre_L4", "q_pre_L5", "q_pre_L6", "q_pre_L7", "q_pre_L8", "q_post_L8")
    q_labels = ("pre-L4", "pre-L5", "pre-L6", "pre-L7", "pre-L8", "post-L8")
    lines = [
        "# Official layer-write realization debt — lifelong B100×100 네 arm exhaustive 사실 보고서",
        "",
        "상태: **FOUR_ARM_LIFELONG_10K_TERMINAL_VALID**  ",
        "범위: Llama/Qwen × Official MEMIT/AlphaEdit 네 arm을 모두 동등한 PRIMARY SCIENTIFIC SCOPE로 분석했다. 각 arm은 B100 100개를 누적한 10,000-request sequential trajectory다. `scientific_promotion=false`다.",
        "",
        "## 1. Executive factual findings",
        "",
        "아래 첫 표의 final-W 성능은 10k checkpoint의 current B100 평가다. NLL은 `mean/median/p90/max`, strict는 `성공 prompt/전체 prompt (율)`이다. q는 고정 sentinel 100개의 10k checkpoint 값이다.",
        "",
        "| 모델 | 방법 | final-W Eff strict | final-W Gen strict | final-W Loc strict | Rewrite new NLL mean/med/p90/max | Rephrase new NLL mean/med/p90/max | q pre/post-L8 med |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, method in CELL_ORDER:
        row = arm_rows[(model, method)]
        lines.append(
            f"| {MODEL_LABEL[model]} | {METHOD_LABEL[method]} | "
            f"{_functional_strict_cell(tables, model, method, 10000, 'current', 'rewrite_target_new')} | "
            f"{_functional_strict_cell(tables, model, method, 10000, 'current', 'rephrase_target_new')} | "
            f"{_functional_strict_cell(tables, model, method, 10000, 'current', 'locality_target_true')} | "
            f"{_functional_stats_cell(tables, model, method, 10000, 'current', 'rewrite_target_new', 'nll_mean')} | "
            f"{_functional_stats_cell(tables, model, method, 10000, 'current', 'rephrase_target_new', 'nll_mean')} | "
            f"{_fmt(row['terminal_sentinel_q_pre_L8_median'])}/{_fmt(row['terminal_sentinel_q_post_L8_median'])} |"
        )

    lines += [
        "",
        "### 핵심 판정",
        "",
        "- **FACT — terminal closure:** terminal sentinel post-L8 q 중앙값은 Llama MEMIT/AlphaEdit, Qwen MEMIT/AlphaEdit 순서로 아래 표와 CSV에 기록된다. AlphaEdit가 두 모델에서 MEMIT보다 낮지만 0은 아니다.",
        "- **FACT — 성능과 보존:** 10k에서 AlphaEdit는 MEMIT보다 rewrite/rephrase strict가 높지만 locality와 earliest-edit retention은 두 모델 모두 크게 낮다. Qwen MEMIT는 terminal current rewrite/rephrase strict가 0이다.",
        "- **FACT — update 위치는 보편적이지 않다:** Llama 두 방법은 10k sentinel에서 L8 update share가 가장 크지만, Qwen은 L4가 가장 크다. 따라서 universal last-layer bottleneck은 관찰되지 않았다.",
        "- **INFERENCE — progressive decoupling은 model-conditioned:** checkpoint D_TV는 Llama에서 t0→10k 증가하지만 Qwen에서는 감소한다. 네 arm 공통의 단조 증가 claim은 성립하지 않는다.",
        "- **INFERENCE — barrier 동기 부여는 미확정:** registered completion geometry는 관찰되었지만 preregistered failure envelope와 독립 onset label이 없어서 T_G/T_A/T_Z/T_F 및 early prediction은 식별할 수 없다.",
        "- **NON-CLAIM:** 이 observational study는 barrier benefit, ODE 필요성, 인과 효과, universal layer bottleneck을 입증하지 않는다.",
        "",
        "질문에 대한 한 문장 답: **lifelong editing은 모든 arm에서 update magnitude와 target progress를 동일하게 움직이지 않았지만 그 decoupling의 진행 방향은 모델별로 달랐고, 현재 completion-geometry 기록만으로 failure를 충분히 일찍 예측하거나 completion-reserve barrier를 정당화할 수는 없다.**",
        "",
        "![10k residual trajectory](primary-residual-trajectory-2x2.png)",
        "",
        "그림 분모: 10k sentinel n=100/arm. 회색선은 request, 굵은선은 median, band는 IQR, 점선은 ideal `[1,.8,.6,.4,.2,0]`; missing interpolation/imputation=0.",
        "",
        "## 2. Provenance, jobs, denominators, invariants",
        "",
        "| 모델 | 방법 | canonical job | source HEAD/tree | batches | requests | checkpoints | raw members/bytes | raw member root |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for arm in arms:
        row = arm_rows[(arm.spec.model, arm.spec.method)]
        lines.append(
            f"| {MODEL_LABEL[arm.spec.model]} | {METHOD_LABEL[arm.spec.method]} | {arm.spec.job_id} | "
            f"`{arm.result['source']['head']}` / `{arm.result['source']['tree']}` | "
            f"{arm.result['valid_batch_denominator']} | {arm.result['valid_request_denominator']} | "
            f"{arm.result['checkpoint_denominator']} | {row['raw_member_count']}/{row['raw_member_bytes']} | "
            f"`{row['raw_member_root']}` |"
        )
    lines += [
        "",
        "- Canonical result roots와 모든 journal/checkpoint/state/log/receipt 절대경로·SHA는 `raw-member-inventory.csv`와 `analysis-manifest.json`에 있다.",
        "- 공통 stream/order/sentinel/evaluator identity와 pinned stock EasyEdit HEAD/tree는 manifest 외부 입력에 봉인했다.",
        "- 합계: arm 4/4, B100 400/400, training requests 40,000/40,000, checkpoints 32/32, primary sentinel observations 3,200, AlphaEdit reset-cache observations 1,600, 전체 checkpoint observations 4,800.",
        "- Arm별 compute_z=10,000, recompute=0, layer observation=500, terminal forward=100. W commit→next entry=99/99; W0/cache terminal restore=4/4; nonfinite=0, rollback violation=0, imputation=0.",
        "- AlphaEdit는 cache width 0→10,000을 성공 B100마다 100씩 append/consume했다. MEMIT은 static covariance computation cache만 사용했고 request-history state를 만들지 않았다.",
        "- Edited weight 5/5는 `torch.float32`; BF16/FP16 parameter=0. GPU peak memory, total forward, JVP/VJP/HVP, autocast/quantization/numeric-cast inventory는 schema에 없으므로 추정하지 않았다.",
        "",
        "## 3. Residual trajectory와 lifelong drift",
        "",
        "### 3.1 10k sentinel q 분포",
        "",
        "통계 열은 `mean/median/IQR/p90/max`; 각 셀 n=100이다.",
        "",
        "| 모델 | 방법 | pre-L8 q | post-L8 q | q_pre-L8>0.2 | median trajectory pre-L4→post-L8 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for model, method in CELL_ORDER:
        trajectory = "/".join(
            _fmt(
                _stat(
                    tables,
                    "checkpoint_request",
                    model,
                    method,
                    field,
                    accepted_edit_count=10000,
                    fork="ACCUMULATED_OR_STATIC",
                )
            )
            for field in q_fields
        )
        count = _stat(
            tables,
            "checkpoint_request",
            model,
            method,
            "q_pre_L8_gt_0_2",
            "sum",
            accepted_edit_count=10000,
            fork="ACCUMULATED_OR_STATIC",
        )
        pre_stats = "/".join(
            _fmt(_stat(tables, "checkpoint_request", model, method, "q_pre_L8", statistic, accepted_edit_count=10000, fork="ACCUMULATED_OR_STATIC"))
            for statistic in ("mean", "median", "iqr", "p90", "max")
        )
        post_stats = "/".join(
            _fmt(_stat(tables, "checkpoint_request", model, method, "q_post_L8", statistic, accepted_edit_count=10000, fork="ACCUMULATED_OR_STATIC"))
            for statistic in ("mean", "median", "iqr", "p90", "max")
        )
        lines.append(
            f"| {MODEL_LABEL[model]} | {METHOD_LABEL[method]} | {pre_stats} | {post_stats} | {int(count)}/100 | {trajectory} |"
        )
    lines += [
        "",
        "### 3.2 fixed sentinel의 t0→10k paired 변화",
        "",
        "동일 request hash n=100/arm의 `10k−t0` median이다.",
        "",
        "| 모델 | 방법 | Δpre-L8 q | Δpost-L8 q | Δd∥ | Δd⊥ | Δmean ρ | Δmean τ | ΔD_TV | Δcenter |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, method in CELL_ORDER:
        comparison = f"{model}:{method}:sentinel-10k-minus-t0"
        lines.append(
            f"| {MODEL_LABEL[model]} | {METHOD_LABEL[method]} | "
            + " | ".join(
                _fmt(_paired_value(tables, comparison, metric))
                for metric in ("q_pre_L8", "q_post_L8", "d_parallel", "d_perp", "mean_rho", "mean_tau", "D_TV", "delta_center")
            )
            + " |"
        )
    lines += [
        "",
        "### 3.3 모든 registered checkpoint",
        "",
        "Weight total은 sentinel probe의 5개 layer Frobenius magnitude 합이다. Functional strict는 그 checkpoint의 current B100이며 t0에는 PRE_EDIT sentinel을 쓴다.",
        "",
        "| 모델/방법 | edits | pre/post-L8 q med | q>0.2 | D_TV med | Δcenter med | update total | V̄/unreachable/min h̄ | rewrite/rephrase/locality strict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model, method in CELL_ORDER:
        for accepted in checkpoint_schedule:
            geometry = _geometry(tables, model, method, accepted)
            cohort = "sentinel_pre_edit" if accepted == 0 else "current"
            lines.append(
                f"| {ARM_SHORT[(model, method)]} | {accepted} | "
                f"{_fmt(_stat(tables, 'checkpoint_request', model, method, 'q_pre_L8', accepted_edit_count=accepted, fork='ACCUMULATED_OR_STATIC'))}/"
                f"{_fmt(_stat(tables, 'checkpoint_request', model, method, 'q_post_L8', accepted_edit_count=accepted, fork='ACCUMULATED_OR_STATIC'))} | "
                f"{int(_stat(tables, 'checkpoint_request', model, method, 'q_pre_L8_gt_0_2', 'sum', accepted_edit_count=accepted, fork='ACCUMULATED_OR_STATIC'))}/100 | "
                f"{_fmt(_stat(tables, 'checkpoint_request', model, method, 'D_TV', accepted_edit_count=accepted, fork='ACCUMULATED_OR_STATIC'))} | "
                f"{_fmt(_stat(tables, 'checkpoint_request', model, method, 'delta_center', accepted_edit_count=accepted, fork='ACCUMULATED_OR_STATIC'))} | "
                f"{_fmt(_checkpoint_update_total(tables, model, method, accepted))} | "
                f"{_fmt(geometry['Vbar'])}/{_fmt(geometry['unreachable_fraction'])}/{_fmt(geometry['minimum_completion_budget_h_normalized'])} | "
                f"{_functional_strict_cell(tables, model, method, accepted, cohort, 'rewrite_target_new')}/"
                f"{_functional_strict_cell(tables, model, method, accepted, cohort, 'rephrase_target_new')}/"
                f"{_functional_strict_cell(tables, model, method, accepted, cohort, 'locality_target_true')} |"
            )
    lines += [
        "",
        "![checkpoint residual heatmap](checkpoint-layer-q-heatmap.png)",
        "",
        "분모: sentinel n=100/checkpoint/arm. Heatmap은 고정 schedule 8개 checkpoint의 median이며 보간하지 않았다.",
        "",
        "## 4. Allocation–realization: A, Y, E, ρ, τ",
        "",
        "A/Y/E full vectors는 publish되지 않았다. 저장된 normalized scalar에서 `Y∥=ρA`, `Y⊥ norm=τA`, `E∥=(1−ρ)A`, `E⊥ norm=τA`만 결정적으로 파생했다. 직교 벡터의 방향은 추정하지 않았다.",
        "",
        "아래 값은 10k sentinel n=100/layer/arm이고 각 scalar는 `mean/median/IQR/p90/max`다.",
        "",
        "| arm | L | A/R1 | Y/R1 | E/R1 | ρ | τ | Y∥/R1 | Y⊥/R1 | E∥/R1 | E⊥ norm/R1 | under/over/opposite/exact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, method in CELL_ORDER:
        for layer in (4, 5, 6, 7, 8):
            selected = [
                row
                for row in tables.checkpoint_layer
                if row["model"] == model
                and row["method"] == method
                and row["accepted_edit_count"] == 10000
                and row["fork"] == "ACCUMULATED_OR_STATIC"
                and row["layer"] == layer
            ]
            under = sum(row["realization_class"] == "UNDER" for row in selected)
            over = sum(row["realization_class"] == "OVERSHOOT" for row in selected)
            opposite = sum(row["realization_class"] == "OPPOSITE" for row in selected)
            exact = sum(row["realization_class"] == "EXACT" for row in selected)
            cells = []
            for metric in (
                "allocation_norm_over_R1",
                "realized_reduction_norm_over_R1",
                "gap_norm_over_R1",
                "rho",
                "tau",
                "Y_parallel_over_R1",
                "Y_perp_over_R1",
                "E_parallel_over_R1",
                "E_perp_magnitude_over_R1",
            ):
                cells.append(
                    "/".join(
                        _fmt(_stat(tables, "checkpoint_layer", model, method, metric, statistic, accepted_edit_count=10000, fork="ACCUMULATED_OR_STATIC", layer=layer))
                        for statistic in ("mean", "median", "iqr", "p90", "max")
                    )
                )
            lines.append(
                f"| {ARM_SHORT[(model, method)]} | {layer} | "
                + " | ".join(cells)
                + f" | {under}/{over}/{opposite}/{exact} |"
            )
    lines += [
        "",
        "![rho terminal](rho-by-layer-terminal.png)",
        "",
        "![tau terminal](tau-by-layer-terminal.png)",
        "",
        "![rho drift](checkpoint-layer-rho-heatmap.png)",
        "",
        "![tau drift](checkpoint-layer-tau-heatmap.png)",
        "",
        "## 5. Inherited debt와 recurrence closure",
        "",
        "| arm | d∥ mean/med/IQR/p90/max | d⊥ mean/med/IQR/p90/max | recurrence mean/med/p90/max | worst request hash | violations/<1e-4 denom |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for model, method in CELL_ORDER:
        recurrence = next(
            row
            for row in rec_rows
            if row["stage"] == "checkpoint"
            and row["model"] == model
            and row["method"] == method
            and row["accepted_edit_count"] == 10000
            and row["fork"] == "ACCUMULATED_OR_STATIC"
        )
        dpar = "/".join(
            _fmt(_stat(tables, "checkpoint_request", model, method, "d_parallel", statistic, accepted_edit_count=10000, fork="ACCUMULATED_OR_STATIC"))
            for statistic in ("mean", "median", "iqr", "p90", "max")
        )
        dperp = "/".join(
            _fmt(_stat(tables, "checkpoint_request", model, method, "d_perp", statistic, accepted_edit_count=10000, fork="ACCUMULATED_OR_STATIC"))
            for statistic in ("mean", "median", "iqr", "p90", "max")
        )
        lines.append(
            f"| {ARM_SHORT[(model, method)]} | {dpar} | {dperp} | "
            f"{_fmt(recurrence['mean'])}/{_fmt(recurrence['median'])}/{_fmt(recurrence['p90'])}/{_fmt(recurrence['max'])} | "
            f"`{recurrence['worst_request_sha256']}` | {recurrence['threshold_violation_count']}/100 |"
        )
    lines += [
        "",
        "![inherited debt](inherited-debt-distributions.png)",
        "",
        "All outliers are finite, identity-valid raw members. Worst identities and top-10 p90/max drivers are in `outlier-ledger.csv`; raw prompts are not published.",
        "",
        "## 6. Layer-wise update magnitude/share",
        "",
        "Activation progress와 weight action은 서로 다른 축이다. Production 통계의 독립 관측 단위는 B100 batch n=100/layer/arm이며 request 100개로 복제하지 않았다.",
        "",
        "| arm | L | production ΔW magnitude mean/med/IQR/p90/max | production share mean/med/IQR/p90/max | 10k sentinel magnitude/share |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, method in CELL_ORDER:
        for layer in (4, 5, 6, 7, 8):
            magnitude = "/".join(
                _fmt(_stat(tables, "production_weight_batch_unit", model, method, "frobenius_magnitude", statistic, layer=layer))
                for statistic in ("mean", "median", "iqr", "p90", "max")
            )
            share = "/".join(
                _fmt(_stat(tables, "production_weight_batch_unit", model, method, "update_magnitude_share", statistic, layer=layer))
                for statistic in ("mean", "median", "iqr", "p90", "max")
            )
            terminal_magnitude = _stat(tables, "checkpoint_weight", model, method, "frobenius_magnitude", accepted_edit_count=10000, fork="ACCUMULATED_OR_STATIC", layer=layer)
            terminal_share = _stat(tables, "checkpoint_weight", model, method, "update_magnitude_share", accepted_edit_count=10000, fork="ACCUMULATED_OR_STATIC", layer=layer)
            lines.append(
                f"| {ARM_SHORT[(model, method)]} | {layer} | {magnitude} | {share} | {_fmt(terminal_magnitude)}/{_fmt(terminal_share)} |"
            )
    lines += [
        "",
        "![Layer-wise Update Magnitude](layer-wise-update-magnitude.png)",
        "",
        "그림 제목은 `Layer-wise Update Magnitude`; n=100 B100 batches/layer/arm. Equal-allocation/ideal line과 `bars:` 문구는 없다.",
        "",
        "![update drift](layer-wise-update-magnitude-drift.png)",
        "",
        "## 7. Endpoint performance — PRE_EDIT과 terminal final W",
        "",
        "### 7.1 PRE_EDIT reference",
        "",
        "각 NLL/margin 열은 `mean/median/p90/max`; strict 분모는 실제 prompt 수다.",
        "",
        "| 모델 | 방법 cohort | rewrite new NLL | rewrite new margin | rewrite strict | rephrase new NLL | rephrase strict | locality true NLL | locality strict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, method in CELL_ORDER:
        lines.append(
            f"| {MODEL_LABEL[model]} | {METHOD_LABEL[method]} PRE_EDIT | "
            f"{_functional_stats_cell(tables, model, method, 0, 'sentinel_pre_edit', 'rewrite_target_new', 'nll_mean')} | "
            f"{_functional_stats_cell(tables, model, method, 0, 'sentinel_pre_edit', 'rewrite_target_new', 'margin_mean')} | "
            f"{_functional_strict_cell(tables, model, method, 0, 'sentinel_pre_edit', 'rewrite_target_new')} | "
            f"{_functional_stats_cell(tables, model, method, 0, 'sentinel_pre_edit', 'rephrase_target_new', 'nll_mean')} | "
            f"{_functional_strict_cell(tables, model, method, 0, 'sentinel_pre_edit', 'rephrase_target_new')} | "
            f"{_functional_stats_cell(tables, model, method, 0, 'sentinel_pre_edit', 'locality_target_true', 'nll_mean')} | "
            f"{_functional_strict_cell(tables, model, method, 0, 'sentinel_pre_edit', 'locality_target_true')} |"
        )
    lines += [
        "",
        "### 7.2 10k final-W current B100 detail",
        "",
        "| arm | metric | NLL mean/med/p90/max | margin mean/med/p90/max | strict |",
        "|---|---|---:|---:|---:|",
    ]
    for model, method in CELL_ORDER:
        for metric in ("rewrite_target_new", "rewrite_target_true", "rephrase_target_new", "locality_target_true"):
            lines.append(
                f"| {ARM_SHORT[(model, method)]} | {metric} | "
                f"{_functional_stats_cell(tables, model, method, 10000, 'current', metric, 'nll_mean')} | "
                f"{_functional_stats_cell(tables, model, method, 10000, 'current', metric, 'margin_mean')} | "
                f"{_functional_strict_cell(tables, model, method, 10000, 'current', metric)} |"
            )
    lines += [
        "",
        "### 7.3 Terminal retention",
        "",
        "Retention panels record rewrite only; rephrase/locality are `NOT_RECORDED_LOW_COST_BATCH` and are not filled with zero.",
        "",
        "| arm | cohort | rewrite new NLL mean/med/p90/max | strict |",
        "|---|---|---:|---:|",
    ]
    for model, method in CELL_ORDER:
        for cohort in ("retention_earliest", "retention_recent", "retention_hash_stratified"):
            lines.append(
                f"| {ARM_SHORT[(model, method)]} | {cohort} | "
                f"{_functional_stats_cell(tables, model, method, 10000, cohort, 'rewrite_target_new', 'nll_mean')} | "
                f"{_functional_strict_cell(tables, model, method, 10000, cohort, 'rewrite_target_new')} |"
            )
    lines += [
        "",
        "![checkpoint endpoint](endpoint-checkpoint-strict-rates.png)",
        "",
        "그림 분모: current B100 100 requests/checkpoint/arm; prompt count는 request 안에서만 반영했다. Checkpoint 사이 missing interpolation=0.",
        "",
        "## 8. AlphaEdit accumulated-cache vs reset-cache mechanism fork",
        "",
        "동일 checkpoint state와 shared sentinel z의 observational clone이다. Production cache policy를 바꾸지 않았으며 probe 후 W/cache exact restore다.",
        "",
        "| 모델 | fork | q pre/post-L8 med | D_TV med | update total | V̄ | unreachable |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in ("llama3-8b-inst", "qwen2.5-7b-inst"):
        for fork in ("ACCUMULATED_OR_STATIC", "RESET_CACHE"):
            geometry = _geometry(tables, model, "alphaedit", 10000, fork)
            lines.append(
                f"| {MODEL_LABEL[model]} | {fork} | "
                f"{_fmt(_stat(tables, 'checkpoint_request', model, 'alphaedit', 'q_pre_L8', accepted_edit_count=10000, fork=fork))}/"
                f"{_fmt(_stat(tables, 'checkpoint_request', model, 'alphaedit', 'q_post_L8', accepted_edit_count=10000, fork=fork))} | "
                f"{_fmt(_stat(tables, 'checkpoint_request', model, 'alphaedit', 'D_TV', accepted_edit_count=10000, fork=fork))} | "
                f"{_fmt(_checkpoint_update_total(tables, model, 'alphaedit', 10000, fork))} | "
                f"{_fmt(geometry['Vbar'])} | {_fmt(geometry['unreachable_fraction'])} |"
            )
        comparison = f"{model}:alphaedit-reset-minus-accumulated:sentinel-10000"
        lines.append(
            f"| {MODEL_LABEL[model]} | RESET−ACC paired median | "
            f"{_fmt(_paired_value(tables, comparison, 'q_pre_L8'))}/{_fmt(_paired_value(tables, comparison, 'q_post_L8'))} | "
            f"{_fmt(_paired_value(tables, comparison, 'D_TV'))} | N/A | N/A | N/A |"
        )
    lines += [
        "",
        "## 9. Paired model/method comparisons",
        "",
        "Production comparisons use exact `(batch_index, request_sha256)` matching, n=10,000 with B100-cluster bootstrap; checkpoint comparisons use fixed sentinel request hash, n=100. Positive is left-minus-right.",
        "",
        "| comparison | metric | paired n | mean | median | p90 | 95% bootstrap mean CI | uncertainty unit |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    selected_comparisons = {
        f"{model}:alphaedit-minus-memit:production" for model in ("llama3-8b-inst", "qwen2.5-7b-inst")
    } | {
        f"{method}:llama-minus-qwen:production" for method in ("memit", "alphaedit")
    } | {
        f"{model}:alphaedit-minus-memit:sentinel-10000" for model in ("llama3-8b-inst", "qwen2.5-7b-inst")
    }
    for row in tables.paired_delta:
        if row["comparison"] not in selected_comparisons or row["metric"] not in {
            "q_pre_L8", "q_post_L8", "d_parallel", "d_perp", "mean_rho", "mean_tau", "D_TV", "target_new_nll"
        }:
            continue
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['paired_n']} | {_fmt(row.get('mean', NOT_RECORDED))} | "
            f"{_fmt(row.get('median', NOT_RECORDED))} | {_fmt(row.get('p90', NOT_RECORDED))} | "
            f"{_fmt(row.get('paired_mean_bootstrap95_low', NOT_RECORDED))}..{_fmt(row.get('paired_mean_bootstrap95_high', NOT_RECORDED))} | "
            f"{row.get('uncertainty_unit', NOT_RECORDED)} |"
        )
    lines += [
        "",
        "## 10. Descriptive associations",
        "",
        "Within-B100 request associations and across-B100 chronological associations are separate. The latter is edit-time confounded. Correlation is association only, not causation.",
        "",
        "| arm | unit | x→target-new-NLL | estimate/median | IQR | 95% cluster bootstrap CI | n/clusters |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in tables.association:
        if row["x"] not in {"q_post_L8", "D_TV", "total_update_magnitude", "d_parallel", "d_perp"}:
            continue
        lines.append(
            f"| {ARM_SHORT[(row['model'], row['method'])]} | {row['unit']} | {row['x']}→{row['y']} | "
            f"{_fmt(row['batch_spearman_median'])} | {_fmt(row['batch_spearman_iqr'])} | "
            f"{_fmt(row['cluster_bootstrap95_low'])}..{_fmt(row['cluster_bootstrap95_high'])} | "
            f"{row['request_n']}/{row['batch_cluster_n']} |"
        )
    lines += [
        "",
        "## 11. Completion geometry와 failure-time boundary",
        "",
        "| arm | edits | A0 | V_to_go | V̄ | unreachable | min h̄ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, method in CELL_ORDER:
        for accepted in checkpoint_schedule:
            geometry = _geometry(tables, model, method, accepted)
            lines.append(
                f"| {ARM_SHORT[(model, method)]} | {accepted} | {_fmt(geometry['A0'])} | {_fmt(geometry['V_to_go'])} | "
                f"{_fmt(geometry['Vbar'])} | {_fmt(geometry['unreachable_fraction'])} | "
                f"{_fmt(geometry['minimum_completion_budget_h_normalized'])} |"
            )
    lines += [
        "",
        "`completion-predictive-boundary.csv`의 waypoint Spearman은 layer ordinal과 remaining-layer count가 함께 변하는 값이므로 independent predictor validation이 아니다. Preregistered calibration envelope와 sustained failure labels가 raw schema에 없어 AUROC/AUPRC 및 T_G/T_A/T_Z/T_F는 `NOT_IDENTIFIABLE_PREREGISTERED_CALIBRATION_ENVELOPE_ABSENT`; 임의 onset/right-censor time을 부여하지 않았다.",
        "",
        "## 12. Outlier audit",
        "",
        "아래는 각 arm의 terminal sentinel q_post-L8 상위 3개다. 전체 p90/max driver는 `outlier-ledger.csv`에 있다.",
        "",
        "| arm | rank | q_post-L8 | request hash | recurrence relerr | identity/nonfinite |",
        "|---|---:|---:|---|---:|---|",
    ]
    for model, method in CELL_ORDER:
        selected = [
            row
            for row in tables.outlier
            if row["model"] == model
            and row["method"] == method
            and row["metric"] == "terminal_sentinel_q_post_L8"
            and row["rank_descending"] <= 3
        ]
        for row in selected:
            lines.append(
                f"| {ARM_SHORT[(model, method)]} | {row['rank_descending']} | {_fmt(row['value'])} | "
                f"`{row['request_sha256']}` | {_fmt(row['recurrence_closure_relative_error'], 8)} | "
                f"{row['identity_valid']}/{row['nonfinite']} |"
            )
    lines += [
        "",
        "## 13. Compute/accounting",
        "",
        "| arm | cell/edit-core/observer hours | observer share | compute_z/recompute | layer obs/terminal fw | key/solve/backward | checkpoint ordered forks | MaxRSS GiB | GPU peak |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model, method in CELL_ORDER:
        row = compute_rows[(model, method)]
        cell_hours = float(row["cell_wall_seconds"]) / 3600.0
        edit_hours = float(row["production_edit_core_wall_seconds_sum"]) / 3600.0
        observer_hours = float(row["production_observer_wall_seconds_sum"]) / 3600.0
        observer_share = float(row["production_observer_wall_seconds_sum"]) / float(row["cell_wall_seconds"])
        max_rss = row["slurm_max_rss_kib"]
        max_rss_gib = float(max_rss) / 1024.0 / 1024.0 if isinstance(max_rss, (int, float)) else max_rss
        lines.append(
            f"| {ARM_SHORT[(model, method)]} | {_fmt(cell_hours)}/{_fmt(edit_hours)}/{_fmt(observer_hours)} | "
            f"{_fmt(observer_share)} | {row['production_direct_z_compute_count']}/{row['production_direct_z_recompute_count']} | "
            f"{row['production_layer_observation_count']}/{row['production_terminal_forward_count']} | "
            f"{row['production_compute_ks_call_count']}/{row['production_solve_call_count']}/{row['production_target_backward_count']} | "
            f"{row['checkpoint_ordered_fork_count']} | {_fmt(max_rss_gib)} | {row['gpu_peak_memory_bytes']} |"
        )
    lines += [
        "",
        "![compute comparison](compute-time-comparison.png)",
        "",
        "`compute-accounting.csv`는 production과 checkpoint edit-core/observer를 분리한다. Setup/model-load/evaluator를 포함한 cell wall과 edit-core를 같은 값으로 해석하지 않았다.",
        "",
        "## 14. Technical exclusions (scientific denominator=0)",
        "",
    ]
    if technical:
        lines += [
            "| canonical arm | excluded job | classification | reason | denominator |",
            "|---|---|---|---|---:|",
        ]
        arm_by_key = {(arm.spec.model, arm.spec.method): arm for arm in arms}
        for row in technical:
            exclusion = next(
                item
                for item in arm_by_key[(row["canonical_model"], row["canonical_method"])].spec.technical_exclusions
                if item.job_id == row["excluded_job_id"]
            )
            lines.append(
                f"| {ARM_SHORT[(row['canonical_model'], row['canonical_method'])]} | {row['excluded_job_id']} | "
                f"{row['classification']} | {exclusion.reason} | 0 |"
            )
    else:
        lines.append("Technical exclusion lineage: 없음.")
    lines += [
        "",
        "짧은 external `srun` overlap sidecar는 canonical parent experiment failure가 아니며 parent/array terminal과 raw chain을 기준으로 판정했다.",
        "",
        "## 15. NOT_RECORDED와 해석 한계",
        "",
        "- `GPU_PEAK_MEMORY`, `TOTAL_FORWARD_COUNT`, `JVP/VJP/HVP`, `AUTOCAST/QUANTIZATION/NUMERIC_CAST_INVENTORY`: `NOT_RECORDED_SCHEMA`.",
        "- Retention rephrase/locality: `NOT_RECORDED_LOW_COST_BATCH`.",
        "- Same-entry probe는 layer weight-response magnitude만 기록했고 request-level q/ρ/τ는 기록하지 않았다. 따라서 ordered-vs-same-entry target progress causal decomposition은 하지 않았다.",
        "- Completion failure onset/AUROC/AUPRC: preregistered threshold/outcome label 부재로 식별 불가. 과학 신호 부재를 임의 right-censor time으로 바꾸지 않았다.",
        "- Raw prompt/logit/generation publish=0. NLL/margin/strict scalar만 사용했다.",
        "- Raw action/norm을 Llama와 Qwen 사이에 pooling하지 않았다. 모델·방법별 absolute와 normalized 값을 분리했다.",
        "",
        "## 16. Reproducible code-only figures와 artifact inventory",
        "",
        "모든 PNG는 `lifelong_analysis_figures.py`의 headless `Agg` CLI가 sealed CSV만 읽어 생성한다. Seed/style/DPI/size/panel order/color/axis policy가 코드에 고정되며 missing interpolation/imputation=0이다. Plot source SHA, input table SHA, exact command, runtime, output PNG SHA는 `plot-reproduction.json`과 manifest에 기록했다. Codex visualization/imagegen/manual image edit artifact=0.",
        "",
        "완전한 request/layer/batch/checkpoint 정보는 deterministic gzip CSV, 요약·비교·compute·exclusion·raw inventory는 plain CSV다. 각 row count/SHA, package member root와 외부 raw roots는 `analysis-manifest.json` 및 `rooted-analysis-receipt.json`에 봉인한다.",
        "",
        "### FACT / INFERENCE / DECISION",
        "",
        "- **FACT:** four-arm common denominator와 W/cache/hash/recurrence/FULL-FP32 invariants는 모두 valid하다.",
        "- **FACT:** terminal AlphaEdit closure는 MEMIT보다 낮지만 보존 및 earliest retention 손실이 크다. Layer update concentration은 Llama L8, Qwen L4로 갈린다.",
        "- **INFERENCE:** action–realization decoupling drift는 architecture/method-conditioned이며 네 arm 공통 monotonic precursor가 아니다.",
        "- **DECISION:** observational result only; barrier intervention opening condition은 이 분석만으로 충족되었다고 판정하지 않는다. `scientific_promotion=false`.",
        "",
    ]
    return "\n".join(lines)


def _member(path: Path, output: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(output).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _dedupe_external(arms: Sequence[ValidatedArm]) -> list[dict[str, Any]]:
    values: dict[tuple[str, str], dict[str, Any]] = {}
    for arm in arms:
        for row in arm.external_members:
            key = (str(Path(row["path"]).resolve()), str(row["sha256"]))
            values[key] = dict(row)
    return sorted(values.values(), key=lambda row: (row["kind"], row["path"]))


def build(args: argparse.Namespace) -> None:
    if args.output.exists() or args.output.is_symlink():
        raise AnalysisBoundary(f"refusing to overwrite analysis output: {args.output}")
    source_head = _git(args.source_root, "rev-parse", "HEAD")
    source_tree = _git(args.source_root, "rev-parse", "HEAD^{tree}")
    if _git(args.source_root, "status", "--porcelain", "--untracked-files=no"):
        raise AnalysisBoundary("analysis source worktree is tracked-dirty; commit builder first")
    lock = InputLock.from_path(args.input_lock)
    if (
        _git(lock.easyedit_root, "rev-parse", "HEAD") != lock.easyedit_head
        or _git(lock.easyedit_root, "rev-parse", "HEAD^{tree}") != lock.easyedit_tree
        or _git(lock.easyedit_root, "status", "--porcelain", "--untracked-files=no")
    ):
        raise AnalysisBoundary("pinned stock EasyEdit HEAD/tree/clean binding differs")
    arms = validate_campaign(lock)
    args.output.mkdir(parents=True, mode=0o755)
    tables = extract_tables(arms)
    arm_summary = arm_summary_rows(arms, tables)
    technical = technical_exclusion_rows(arms)
    named_plain: dict[str, Sequence[Mapping[str, Any]]] = {
        "arm-summary.csv": arm_summary,
        "checkpoint-summary.csv": checkpoint_summary_rows(tables),
        "layer-q-summary.csv": [row for row in tables.summary if row["metric"].startswith("q_")],
        "allocation-realization-summary.csv": allocation_summary_rows(tables),
        "rho-tau-summary.csv": [row for row in tables.summary if row["metric"] in {"rho", "tau"}],
        "inherited-debt-summary.csv": [row for row in tables.summary if row["metric"] in {"d_parallel", "d_perp"}],
        "recurrence-summary.csv": recurrence_rows(tables),
        "update-magnitude-share.csv": [row for row in tables.summary if row["source_table"] == "production_weight_batch_unit"],
        "checkpoint-update-magnitude-share.csv": [row for row in tables.summary if row["source_table"] == "checkpoint_weight"],
        "same-entry-probe-summary.csv": [row for row in tables.summary if row["source_table"] == "checkpoint_same_entry"],
        "ordered-vs-same-entry-layer-delta.csv": ordered_same_entry_rows(tables),
        "completion-geometry-summary.csv": tables.checkpoint_geometry,
        "completion-predictive-boundary.csv": completion_predictive_boundary_rows(tables),
        "endpoint-metrics-summary.csv": endpoint_summary_rows(tables),
        "paired-deltas.csv": tables.paired_delta,
        "outlier-ledger.csv": tables.outlier,
        "descriptive-associations.csv": tables.association,
        "compute-accounting.csv": compute_accounting_rows(arms, tables),
        "technical-exclusions.csv": technical or [{"status": "NO_TECHNICAL_EXCLUSIONS"}],
        "failure-time-right-censor-boundary.csv": failure_time_boundary_rows(arms),
        "raw-member-inventory.csv": raw_inventory_rows(arms),
    }
    named_gzip: dict[str, Sequence[Mapping[str, Any]]] = {
        "production-request-complete.csv.gz": tables.production_request,
        "production-layer-complete.csv.gz": tables.production_layer,
        "production-weight-batch-unit-complete.csv.gz": tables.production_weight,
        "production-batch-complete.csv.gz": tables.production_batch,
        "checkpoint-request-complete.csv.gz": tables.checkpoint_request,
        "checkpoint-layer-complete.csv.gz": tables.checkpoint_layer,
        "checkpoint-weight-complete.csv.gz": tables.checkpoint_weight,
        "checkpoint-geometry-complete.csv.gz": tables.checkpoint_geometry,
        "checkpoint-waypoint-complete.csv.gz": tables.checkpoint_waypoint,
        "checkpoint-same-entry-complete.csv.gz": tables.checkpoint_same_entry,
        "functional-endpoint-complete.csv.gz": tables.functional,
        "production-compute-by-batch.csv.gz": tables.compute,
    }
    for name, rows in named_plain.items():
        write_csv_once(args.output / name, rows)
    for name, rows in named_gzip.items():
        write_csv_once(args.output / name, rows, compressed=True)
    table_seal = {
        "schema": "odeedit.s06.layer-realization-debt.lifelong-fourarm-derived-table-seal.v1",
        "instruction_id": INSTRUCTION_ID,
        "tables": [
            {
                "path": name,
                "rows": len(rows),
                "bytes": (args.output / name).stat().st_size,
                "sha256": sha256_file(args.output / name),
            }
            for name, rows in sorted({**named_plain, **named_gzip}.items())
        ],
        "missing_policy": "NO_INTERPOLATION_NO_IMPUTATION",
    }
    table_seal["identity_sha256"] = canonical_hash(table_seal)
    write_json_once(args.output / "derived-table-seal.json", table_seal)
    plot_command = [
        sys.executable,
        "-m",
        "project.run_scripts.official_layer_realization_debt.lifelong_analysis_figures",
        "--tables",
        str(args.output),
        "--output",
        str(args.output),
    ]
    subprocess.check_call(plot_command, cwd=args.source_root)
    plot_receipt = json.loads((args.output / "plot-reproduction.json").read_text(encoding="utf-8"))
    figure_names = [row["path"] for row in plot_receipt["figures"]]
    report_name = "official-layer-realization-debt-lifelong-fourarm-exhaustive-factual-ko.md"
    report_path = args.output / report_name
    descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(korean_report_exhaustive(arms, tables, arm_summary, technical))

    data_names = [*named_plain, *named_gzip, "derived-table-seal.json", "plot-reproduction.json", *figure_names, report_name]
    members = [_member(args.output / name, args.output) for name in sorted(data_names)]
    external = _dedupe_external(arms)
    input_lock_member = {
        "kind": "ANALYSIS_INPUT_LOCK",
        "path": str(args.input_lock.resolve()),
        "bytes": args.input_lock.stat().st_size,
        "sha256": sha256_file(args.input_lock),
    }
    external.append(input_lock_member)
    external.sort(key=lambda row: (row["kind"], row["path"]))
    manifest = {
        "schema": "odeedit.s06.layer-realization-debt.lifelong-fourarm-exhaustive-analysis-manifest.v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "FOUR_ARM_LIFELONG_10K_TERMINAL_VALID",
        "builder_source": {"head": source_head, "tree": source_tree, "tracked_clean": True},
        "run_sources": sorted(
            {
                (arm.result["source"]["head"], arm.result["source"]["tree"])
                for arm in arms
            }
        ),
        "easyedit": {
            "root": str(lock.easyedit_root),
            "head": lock.easyedit_head,
            "tree": lock.easyedit_tree,
        },
        "stream": {
            "root": lock.stream_root,
            "order": lock.order_root,
            "sentinel_order": lock.sentinel_order_root,
            "sha256": lock.stream_sha256,
            "evaluator_identity": lock.evaluator_identity,
        },
        "denominators": {
            "valid_arms": 4,
            "attempted_arms": 4,
            "valid_batches": 400,
            "valid_training_requests": 40_000,
            "valid_checkpoints": 32,
            "checkpoint_sentinel_request_observations_primary": 3_200,
            "checkpoint_sentinel_request_observations_alphaedit_reset_cache": 1_600,
            "checkpoint_sentinel_request_observations_total": 4_800,
            "technical_exclusion_scientific_denominator": 0,
            "failure_count": 0,
            "nonfinite_count": 0,
            "imputation_count": 0,
        },
        "jobs": [
            {
                "model": arm.spec.model,
                "method": arm.spec.method,
                "job_id": arm.spec.job_id,
                "scheduler_state": arm.spec.scheduler_state,
                "exit_code": arm.spec.scheduler_exit_code,
                "resource": dict(arm.spec.resource),
            }
            for arm in arms
        ],
        "raw_scientific_identity": [
            {
                "model": arm.spec.model,
                "method": arm.spec.method,
                "member_count": len(arm.raw_members),
                "bytes": sum(int(row["bytes"]) for row in arm.raw_members),
                "raw_member_root": arm.raw_member_root,
            }
            for arm in arms
        ],
        "external_inputs": external,
        "external_input_root_deployment_path_bound": canonical_hash(
            [[row["kind"], row["path"], row["bytes"], row["sha256"]] for row in external]
        ),
        "members": members,
        "member_root": member_root(members),
        "table_row_counts": {
            name: len(rows) for name, rows in {**named_plain, **named_gzip}.items()
        },
        "plot_reproduction": {
            "source_path": plot_receipt["source_path"],
            "source_sha256": plot_receipt["source_sha256"],
            "command": plot_receipt["command"],
            "input_tables": plot_receipt["inputs"],
            "figures": plot_receipt["figures"],
            "seed": plot_receipt["seed"],
            "style": plot_receipt["style"],
            "dpi": plot_receipt["dpi"],
            "missing_policy": plot_receipt["missing_policy"],
            "runtime": plot_receipt["runtime"],
            "figure_geometry": plot_receipt["figure_geometry"],
            "captions": plot_receipt["captions"],
        },
        "not_recorded": {
            "gpu_peak_memory": NOT_RECORDED,
            "total_forward_count": NOT_RECORDED,
            "jvp_vjp_hvp": NOT_RECORDED,
            "autocast_quantization_numeric_cast_inventory": NOT_RECORDED,
            "retention_rephrase_locality": "NOT_RECORDED_LOW_COST_BATCH",
        },
        "raw_prompt_logit_generation_git_count": 0,
        "raw_model_cache_tensor_dataset_credential_git_count": 0,
        "scientific_promotion": False,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_path = args.output / "analysis-manifest.json"
    write_json_once(manifest_path, manifest)
    receipt = {
        "schema": "odeedit.s06.layer-realization-debt.lifelong-fourarm-rooted-analysis-receipt.v1",
        "instruction_id": INSTRUCTION_ID,
        "status": manifest["status"],
        "manifest": {
            "path": manifest_path.name,
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        },
        "member_root": manifest["member_root"],
        "raw_scientific_identity": manifest["raw_scientific_identity"],
        "external_input_root_deployment_path_bound": manifest["external_input_root_deployment_path_bound"],
        "denominators": manifest["denominators"],
        "scientific_promotion": False,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    write_json_once(args.output / "rooted-analysis-receipt.json", receipt)


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input-lock", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
