#!/usr/bin/env python3
"""Build raw-free exhaustive factual reports for C-writer Phase 2 and Phase 3."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import stat
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[4]
REPORTS = REPO / "local/odebf/reports"
RESULTS = REPO / "local/odebf/results"
PHASE1_REPO = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-p1r52-c-writer-phase1-tech-r1"
)
PHASE1_REPORT_V1 = PHASE1_REPO / "local/odebf/reports" / (
    "p1r52-c-writer-phase1-full-fp32-sequential-tech-r1-"
    "exhaustive-analysis-v1"
)
PHASE1_REPORT_V4 = PHASE1_REPO / "local/odebf/reports" / (
    "p1r52-c-writer-phase1-full-fp32-sequential-tech-r1-"
    "exhaustive-analysis-v4"
)
PHASE1_RESULT_NAMES = {
    arm: (
        "s05-p1r52-c-writer-phase1-full-fp32-sequential-"
        f"{arm}-10xb100-tech-r1-v1"
    )
    for arm in ("c0", "c1", "c3")
}
FINAL_V6 = REPORTS / "p1r52-joint-pc-full-fp32-independent-b100x10-final-v6"

PHASE_ROOTS = {
    2: {
        "c0": RESULTS / "s05-p1r52-c-writer-phase2-independent-full-fp32-c0-kstep-10xb100-v1",
        "c1": RESULTS / "s05-p1r52-c-writer-phase2-independent-full-fp32-c1-kstep-10xb100-v1",
        "c3": RESULTS / "s05-p1r52-c-writer-phase2-independent-full-fp32-c3-kstep-10xb100-v1",
    },
    3: {
        "c0": RESULTS / "s05-p1r52-c-writer-phase3-kstep-cache-sequential-full-fp32-c0-kstep-cache-10xb100-v1",
        "c1": RESULTS / "s05-p1r52-c-writer-phase3-kstep-cache-sequential-full-fp32-c1-kstep-cache-10xb100-v1",
        "c3": RESULTS / "s05-p1r52-c-writer-phase3-kstep-cache-sequential-full-fp32-c3-kstep-cache-10xb100-v1",
    },
}

OUT = {
    2: REPORTS / "p1r52-c-writer-phase2-independent-kstep-full-fp32-exhaustive-analysis-v1",
    3: REPORTS / "p1r52-c-writer-phase3-kstep-cache-sequential-full-fp32-exhaustive-analysis-v1",
}

LABELS = {
    "c0": "C0-KSTEP",
    "c1": "C1-KSTEP",
    "c3": "C3-KSTEP",
}

METHODS = {
    "c0": {
        "writer": "PIR-U control remaining-residual L4→L8 prefix sequential writer",
        "route": "PIR-U control allocation; P/C router call 0",
        "official_writer": False,
    },
    "c1": {
        "writer": "Joint P+C minimax + remaining-residual L4→L8 prefix sequential writer",
        "route": "current-K P/C proxy and Joint-PC route recomputed; router call 1/K",
        "official_writer": False,
    },
    "c3": {
        "writer": "direct Official EasyEdit AlphaEdit writer with P1R52 accepted-z",
        "route": "no P/C route; native AlphaEdit compute_z call 0",
        "official_writer": True,
    },
}


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def stats(values: Iterable[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "mean": "NOT_RECORDED",
            "median": "NOT_RECORDED",
            "p90": "NOT_RECORDED",
            "max": "NOT_RECORDED",
            "min": "NOT_RECORDED",
        }
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": median,
        "p90": percentile(ordered, 0.9),
        "max": ordered[-1],
        "min": ordered[0],
    }


def flatten(values: Iterable[Iterable[Any]]) -> list[Any]:
    return [item for row in values for item in row]


def rate(numerator: int, denominator: int) -> float:
    return float(numerator) / denominator if denominator else float("nan")


def write_once(path: Path, payload: bytes) -> Path:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"create-once target exists: {path}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
    finally:
        os.close(fd)
    return path


def dump_json(directory: Path, name: str, value: Any) -> Path:
    return write_once(
        directory / name,
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
    )


def dump_csv(directory: Path, name: str, rows: list[dict[str, Any]]) -> Path:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return write_once(directory / name, buffer.getvalue().encode())


def file_fact(path: Path, role: str) -> dict[str, Any]:
    mode = stat.S_IMODE(path.stat().st_mode)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"not a regular non-symlink file: {path}")
    return {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "mode": f"{mode:04o}",
        "sha256": sha_file(path),
    }


def unit_files(phase: int, root: Path) -> list[Path]:
    pattern = "cases/case-*/terminal.json" if phase == 2 else "batches/b*/terminal.json"
    files = sorted((root / "raw").glob(pattern))
    if len(files) != 10:
        raise RuntimeError(f"expected 10 unit receipts at {root}, found {len(files)}")
    return files


def endpoint_prompt_rows(
    phase: int,
    arm: str,
    unit: int,
    step: Any,
    endpoint_name: str,
    endpoint: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scores = endpoint["scores"]
    for prompt, acc_key, success_key in (
        ("rewrite", "rewrite_acc", "rewrite_success"),
        ("rephrase", "rephrase_acc", "rephrase_success"),
    ):
        acc = scores[acc_key]
        success = scores[success_key]
        new_values = acc["target_new_nll_by_request"]
        true_values = acc["target_true_nll_by_request"]
        for request_index, (new_row, true_row) in enumerate(
            zip(new_values, true_values), start=1
        ):
            for prompt_index, (new_value, true_value) in enumerate(
                zip(new_row, true_row), start=1
            ):
                rows.append(
                    {
                        "phase": phase,
                        "arm": arm,
                        "unit": unit,
                        "K": step,
                        "endpoint": endpoint_name,
                        "prompt": prompt,
                        "request_index": request_index,
                        "prompt_index": prompt_index,
                        "target_new_nll": new_value,
                        "target_true_nll": true_value,
                        "success_bit": success["per_request_bits"][request_index - 1][prompt_index - 1],
                        "accuracy_bit": acc["per_request_bits"][request_index - 1][prompt_index - 1],
                        "strict_success_request_bit": success["strict_all_prompt_bits"][request_index - 1],
                        "strict_accuracy_request_bit": acc["strict_all_prompt_bits"][request_index - 1],
                    }
                )
    return rows


def endpoint_metric_row(
    phase: int,
    arm: str,
    unit: int,
    step: Any,
    endpoint_name: str,
    endpoint: dict[str, Any],
) -> dict[str, Any]:
    summary = endpoint["summary"]
    return {
        "phase": phase,
        "arm": arm,
        "unit": unit,
        "K": step,
        "endpoint": endpoint_name,
        **{key: value for key, value in summary.items() if key != "identity_sha256"},
        "endpoint_identity_sha256": summary["identity_sha256"],
    }


def aggregate_endpoint(
    prompt_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    *,
    phase: int,
    arm: str,
    endpoint: str,
    step: Any,
) -> dict[str, Any]:
    selected_prompts = [
        row
        for row in prompt_rows
        if row["phase"] == phase
        and row["arm"] == arm
        and row["endpoint"] == endpoint
        and row["K"] == step
    ]
    selected_metrics = [
        row
        for row in metric_rows
        if row["phase"] == phase
        and row["arm"] == arm
        and row["endpoint"] == endpoint
        and row["K"] == step
    ]
    if not selected_metrics:
        raise RuntimeError(f"missing endpoint metrics {phase=} {arm=} {endpoint=} {step=}")
    result: dict[str, Any] = {
        "phase": phase,
        "arm": arm,
        "endpoint": endpoint,
        "K": step,
        "unit_count": len(selected_metrics),
        "locality_numerator": sum(row["locality_numerator"] for row in selected_metrics),
        "locality_denominator": sum(row["locality_denominator"] for row in selected_metrics),
    }
    result["locality_rate"] = rate(
        result["locality_numerator"], result["locality_denominator"]
    )
    for prompt in ("rewrite", "rephrase"):
        rows = [row for row in selected_prompts if row["prompt"] == prompt]
        result[f"{prompt}_success_numerator"] = sum(row["success_bit"] for row in rows)
        result[f"{prompt}_success_denominator"] = len(rows)
        result[f"{prompt}_accuracy_numerator"] = sum(row["accuracy_bit"] for row in rows)
        result[f"{prompt}_accuracy_denominator"] = len(rows)
        # Strict bits are request-level; take only the first prompt per request.
        strict_rows = [row for row in rows if row["prompt_index"] == 1]
        result[f"{prompt}_strict_success_numerator"] = sum(
            row["strict_success_request_bit"] for row in strict_rows
        )
        result[f"{prompt}_strict_success_denominator"] = len(strict_rows)
        result[f"{prompt}_strict_accuracy_numerator"] = sum(
            row["strict_accuracy_request_bit"] for row in strict_rows
        )
        result[f"{prompt}_strict_accuracy_denominator"] = len(strict_rows)
        for target in ("new", "true"):
            values = [row[f"target_{target}_nll"] for row in rows]
            for key, value in stats(values).items():
                result[f"{prompt}_target_{target}_{key}"] = value
    return result


def aggregate_from_reference_scores(
    phase: int, arm: str, endpoint: str, cohorts: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompt_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        unit = int(cohort.get("batch_index", cohort.get("case_index")))
        prompt_rows.extend(
            endpoint_prompt_rows(phase, arm, unit, "REFERENCE", endpoint, cohort)
        )
        metric_rows.append(
            endpoint_metric_row(phase, arm, unit, "REFERENCE", endpoint, cohort)
        )
    return (
        aggregate_endpoint(
            prompt_rows,
            metric_rows,
            phase=phase,
            arm=arm,
            endpoint=endpoint,
            step="REFERENCE",
        ),
        prompt_rows,
    )


def metric_rates(row: dict[str, Any]) -> dict[str, float]:
    output = {"locality_rate": row["locality_rate"]}
    for prompt in ("rewrite", "rephrase"):
        for kind in ("success", "accuracy", "strict_success", "strict_accuracy"):
            numerator = row[f"{prompt}_{kind}_numerator"]
            denominator = row[f"{prompt}_{kind}_denominator"]
            output[f"{prompt}_{kind}_rate"] = rate(numerator, denominator)
    return output


def route_row(phase: int, arm: str, unit: int, step: int, route: dict[str, Any]) -> dict[str, Any]:
    selected_pi = route.get("selected_pi", route.get("pi", "NOT_APPLICABLE"))
    if isinstance(selected_pi, list):
        pi_values = selected_pi
    else:
        pi_values = []
    row = {
        "phase": phase,
        "arm": arm,
        "unit": unit,
        "K": step,
        "status": route.get("status", route.get("solver_status", "NOT_RECORDED")),
        "route_identity_sha256": route.get("identity_sha256", "NOT_RECORDED"),
        "router_call_expected": 1 if arm == "c1" else 0,
        "solver_success": route.get("solver_success", "NOT_APPLICABLE"),
        "solver_iterations": route.get("solver_iterations", "NOT_APPLICABLE"),
        "minimax_t": route.get("minimax_t", "NOT_APPLICABLE"),
        "selected_p": route.get("selected_p", route.get("soft_p", "NOT_APPLICABLE")),
        "selected_c": route.get("selected_c", route.get("selected_capacity", "NOT_APPLICABLE")),
        "selected_p_normalized": route.get("selected_p_normalized", "NOT_APPLICABLE"),
        "selected_c_normalized": route.get("selected_c_normalized", "NOT_APPLICABLE"),
        "stationarity_residual": route.get("stationarity_residual", "NOT_APPLICABLE"),
        "complementarity_residual": route.get("complementarity_residual", "NOT_APPLICABLE"),
        "simplex_sum_residual": route.get("simplex_sum_residual", route.get("equality_residual", "NOT_APPLICABLE")),
        "decision_influence_count": route.get("decision_influence_count", "NOT_APPLICABLE"),
        "pc_router_call_count": route.get("pc_router_call_count", 1 if arm == "c1" else 0),
    }
    for index, layer in enumerate((4, 5, 6, 7, 8)):
        row[f"pi_{layer}"] = pi_values[index] if index < len(pi_values) else "NOT_APPLICABLE"
    return row


def layer_rows(
    phase: int,
    arm: str,
    unit: int,
    step: int,
    writer_receipt: dict[str, Any],
) -> list[dict[str, Any]]:
    writer = writer_receipt.get("writer", {})
    layers = writer.get("layers", [])
    transaction = writer.get("transaction", {})
    transaction_layers = {
        int(row["layer"]): row for row in transaction.get("layer_receipts", [])
    }
    if not layers:
        return [
            {
                "phase": phase,
                "arm": arm,
                "unit": unit,
                "K": step,
                "layer": layer,
                "telemetry_status": "NOT_RECORDED_NATIVE_OFFICIAL_APPLY",
                "official_entrypoint": writer.get("official_entrypoint", "NOT_RECORDED"),
                "native_alphaedit_compute_z_call_count": writer.get(
                    "native_alphaedit_compute_z_call_count", "NOT_RECORDED"
                ),
                "edit_core_wall_seconds_per_K": writer.get(
                    "edit_core_wall_seconds", "NOT_RECORDED"
                ),
            }
            for layer in (4, 5, 6, 7, 8)
        ]
    output: list[dict[str, Any]] = []
    for layer in layers:
        layer_id = int(layer["layer"])
        tx = transaction_layers.get(layer_id, {})
        output.append(
            {
                "phase": phase,
                "arm": arm,
                "unit": unit,
                "K": step,
                "layer": layer_id,
                "telemetry_status": "RECORDED",
                "pi": layer.get("pi"),
                "beta": layer.get("beta"),
                "suffix_mass": layer.get("suffix_mass"),
                "applied_coefficient": layer.get("applied_coefficient"),
                "entry_residual_norm": layer.get("entry_residual_norm"),
                "current_residual_norm": layer.get("current_residual_norm"),
                "key_norm": layer.get("key_norm"),
                "q_norm": layer.get("q_norm"),
                "update_norm": layer.get("update_norm"),
                "update_energy": layer.get("update_energy"),
                "update_norm_share": layer.get("update_norm_share"),
                "update_energy_share": layer.get("update_energy_share"),
                "storage_dtype": layer.get("storage_dtype"),
                "alpha_cache_history_width": layer.get("alpha_cache_history_width"),
                "prepared_update_norm": tx.get("pre_cast_fp32_update_norm", "NOT_RECORDED"),
                "prepared_update_energy": tx.get("pre_cast_fp32_update_energy", "NOT_RECORDED"),
                "actual_delta_norm": tx.get("actual_post_storage_delta32_norm", "NOT_RECORDED"),
                "actual_delta_energy": tx.get("actual_post_storage_delta32_energy", "NOT_RECORDED"),
                "rounding_error_max_abs": tx.get("prepared_vs_actual_rounding_error_max_abs", "NOT_RECORDED"),
                "rounding_error_norm": tx.get("prepared_vs_actual_rounding_error_norm", "NOT_RECORDED"),
                "rounding_error_energy": tx.get("prepared_vs_actual_rounding_error_energy", "NOT_RECORDED"),
                "prepared_vs_actual_byte_exact": tx.get("prepared_vs_actual_byte_exact", "NOT_RECORDED"),
                "prepared_vs_actual_cosine": tx.get("prepared_vs_actual_cosine", "NOT_RECORDED"),
                "official_reference_endpoint_byte_exact": tx.get("official_reference_endpoint_byte_exact", "NOT_RECORDED"),
                "storage_assignment_count": tx.get("storage_assignment_count", "NOT_RECORDED"),
                "numeric_storage_cast_count": tx.get("numeric_storage_cast_count", "NOT_RECORDED"),
                "bf16_path_call_count": tx.get("bf16_path_call_count", 0),
                "postcast_decision_influence_count": tx.get("postcast_decision_influence_count", 0),
            }
        )
    return output


def compute_row(phase: int, arm: str, unit: int, payload: dict[str, Any]) -> dict[str, Any]:
    target_compute = payload.get("target_public", {}).get("compute", {})
    totals = target_compute.get("totals", {})
    wall = target_compute.get("wall_seconds", {})
    executions = payload["kstep_executions"]
    execution_counts: dict[str, int] = defaultdict(int)
    for execution in executions:
        for key, value in execution["compute"].items():
            if isinstance(value, int):
                execution_counts[key] += value
    direct_official_writer_seconds = sum(
        float(execution["writer_receipt"].get("writer", {}).get("edit_core_wall_seconds", 0.0))
        for execution in executions
    )
    return {
        "phase": phase,
        "arm": arm,
        "unit": unit,
        "unit_total_seconds": payload.get("case_total_seconds", payload.get("batch_total_seconds")),
        "target_compute_status": "RECORDED" if totals else "NOT_RECORDED_PER_BATCH",
        "target_model_forward_calls": totals.get("model_forward_calls", "NOT_RECORDED"),
        "target_backward_calls": totals.get("backward_calls", "NOT_RECORDED"),
        "target_autograd_invocations": totals.get("autograd_invocations", "NOT_RECORDED"),
        "target_capture_forward_calls": totals.get("capture_forward_calls", "NOT_RECORDED"),
        "target_physical_microbatch_graphs": totals.get("physical_microbatch_graphs", "NOT_RECORDED"),
        "target_processed_tokens": totals.get("processed_tokens", "NOT_RECORDED"),
        "target_padded_tokens": totals.get("padded_tokens", "NOT_RECORDED"),
        "target_wall_seconds_total": sum(
            value for value in wall.values() if isinstance(value, (int, float))
        ) if wall else "NOT_RECORDED",
        "accepted_state_refresh_wall_seconds": wall.get("accepted_state_refresh", "NOT_RECORDED"),
        "kstep_writer_wall_seconds": wall.get("c_kstep_writer", "NOT_RECORDED"),
        "finite_demand_endpoint_wall_seconds": wall.get("finite_demand_endpoint_forward", "NOT_RECORDED"),
        "target_gradient_wall_seconds": wall.get("target_gradient", "NOT_RECORDED"),
        "terminal_objective_wall_seconds": wall.get("terminal_objective", "NOT_RECORDED"),
        "direct_official_writer_edit_core_seconds": direct_official_writer_seconds if arm == "c3" else "NOT_APPLICABLE",
        "accepted_z_evaluator_count": execution_counts["accepted_z_evaluator_count"],
        "pre_writer_evaluator_count": execution_counts["pre_writer_evaluator_count"],
        "post_writer_evaluator_count": execution_counts["post_writer_evaluator_count"],
        "heldout_evaluator_count": execution_counts["heldout_evaluator_count"],
        "dense_update_construction_count": execution_counts["dense_update_construction_count"],
        "native_apply_count": execution_counts["native_apply_count"],
        "logical_commit_count": execution_counts["logical_commit_count"],
        "router_call_count": execution_counts["router_call_count"],
        "numeric_storage_cast_count": execution_counts["numeric_storage_cast_count"],
        "bf16_fp16_path_count": execution_counts["bf16_fp16_path_count"],
        "added_model_backward_count": execution_counts["model_backward_added_count"],
        "semantic_slope_backward_added_count": execution_counts["semantic_slope_backward_added_count"],
        "retry_count": execution_counts["retry_count"],
    }


def aggregate_compute(rows: list[dict[str, Any]], top_terminals: dict[str, Any], job_timings: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for arm in ("c0", "c1", "c3"):
        selected = [row for row in rows if row["arm"] == arm]
        row: dict[str, Any] = {
            "arm": arm,
            "unit_count": len(selected),
            "unit_total_seconds_sum": sum(row["unit_total_seconds"] for row in selected),
            "unit_total_seconds_mean": stats(row["unit_total_seconds"] for row in selected)["mean"],
            "unit_total_seconds_median": stats(row["unit_total_seconds"] for row in selected)["median"],
            "unit_total_seconds_p90": stats(row["unit_total_seconds"] for row in selected)["p90"],
            "unit_total_seconds_max": stats(row["unit_total_seconds"] for row in selected)["max"],
            "job_total_seconds": job_timings[arm]["job_total_seconds"],
            "slurm_job_id": job_timings[arm]["slurm_job_id"],
        }
        for timing_key in (
            "target_wall_seconds_total", "accepted_state_refresh_wall_seconds",
            "kstep_writer_wall_seconds", "finite_demand_endpoint_wall_seconds",
            "target_gradient_wall_seconds", "terminal_objective_wall_seconds",
            "direct_official_writer_edit_core_seconds",
        ):
            recorded = [item[timing_key] for item in selected if isinstance(item[timing_key], (int, float))]
            row[f"{timing_key}_sum"] = sum(recorded) if recorded else "NOT_RECORDED"
            row[f"{timing_key}_mean"] = stats(recorded)["mean"] if recorded else "NOT_RECORDED"
        for key in (
            "accepted_z_evaluator_count", "pre_writer_evaluator_count",
            "post_writer_evaluator_count", "heldout_evaluator_count",
            "dense_update_construction_count", "native_apply_count",
            "logical_commit_count", "router_call_count",
            "numeric_storage_cast_count", "bf16_fp16_path_count",
            "added_model_backward_count", "semantic_slope_backward_added_count",
            "retry_count",
        ):
            row[key] = sum(item[key] for item in selected)
        top = top_terminals[arm]
        observation = top.get("final_gpu_host_observation", {})
        row.update(
            {
                "device_name": observation.get("device_name", "NOT_RECORDED"),
                "device_uuid": observation.get("device_uuid", "NOT_RECORDED"),
                "peak_allocated_bytes": observation.get("peak_allocated_bytes", "NOT_RECORDED"),
                "peak_reserved_bytes": observation.get("peak_reserved_bytes", "NOT_RECORDED"),
                "host_maxrss_kib": observation.get("host_maxrss_kib", "NOT_RECORDED"),
                "top_job_compute": json.dumps(top.get("job_compute", {}), sort_keys=True, separators=(",", ":")),
            }
        )
        output.append(row)
    return output


def format_percent(numerator: int, denominator: int) -> str:
    return f"{100.0 * rate(numerator, denominator):.2f}% ({numerator}/{denominator})"


def endpoint_table(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines += [
        "|방법/endpoint|EFF|rewrite acc|GEN|strict GEN|rephrase acc|strict rephrase acc|LOC|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "|{label}|{eff}|{racc}|{gen}|{sgen}|{gacc}|{sgacc}|{loc}|".format(
                label=row["label"],
                eff=format_percent(row["rewrite_success_numerator"], row["rewrite_success_denominator"]),
                racc=format_percent(row["rewrite_accuracy_numerator"], row["rewrite_accuracy_denominator"]),
                gen=format_percent(row["rephrase_success_numerator"], row["rephrase_success_denominator"]),
                sgen=format_percent(row["rephrase_strict_success_numerator"], row["rephrase_strict_success_denominator"]),
                gacc=format_percent(row["rephrase_accuracy_numerator"], row["rephrase_accuracy_denominator"]),
                sgacc=format_percent(row["rephrase_strict_accuracy_numerator"], row["rephrase_strict_accuracy_denominator"]),
                loc=format_percent(row["locality_numerator"], row["locality_denominator"]),
            )
        )


def nll_table(lines: list[str], rows: list[dict[str, Any]], prompt: str) -> None:
    lines += [
        f"|방법/endpoint|{prompt} target-new mean/median/p90/max|{prompt} target-true mean/median/p90/max|",
        "|---|---:|---:|",
    ]
    for row in rows:
        n = f"{row[f'{prompt}_target_new_mean']:.6f}/{row[f'{prompt}_target_new_median']:.6f}/{row[f'{prompt}_target_new_p90']:.6f}/{row[f'{prompt}_target_new_max']:.6f}"
        t = f"{row[f'{prompt}_target_true_mean']:.6f}/{row[f'{prompt}_target_true_median']:.6f}/{row[f'{prompt}_target_true_p90']:.6f}/{row[f'{prompt}_target_true_max']:.6f}"
        lines.append(f"|{row['label']}|{n}|{t}|")


def build_phase(phase: int) -> None:
    directory = OUT[phase]
    if directory.exists() or directory.is_symlink():
        raise RuntimeError(f"create-once report namespace exists: {directory}")
    os.mkdir(directory, 0o700)

    methods = []
    for arm in ("c0", "c1", "c3"):
        methods.append(
            {
                "phase": phase,
                "arm": arm,
                "label": LABELS[arm] + ("-CACHE" if phase == 3 else ""),
                **METHODS[arm],
                "target": "P1R52 accepted-z refreshed at every K",
                "writer_calls_per_unit": 8,
                "cache_status": (
                    "ALPHA_CACHE_OFF_CONTROL"
                    if phase == 2
                    else "ALPHA_CACHE_CONTINUITY_ON_BATCH_ENTRY_SNAPSHOT"
                ),
                "execution_scope": (
                    "10 independent B100; W0/cache0 restored between cases"
                    if phase == 2
                    else "B1→B10 sequential W/cache continuity; final exact W0 restore"
                ),
            }
        )

    top_terminals: dict[str, Any] = {}
    job_timings: dict[str, Any] = {}
    unit_payloads: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prompt_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    k_rows: list[dict[str, Any]] = []
    routes: list[dict[str, Any]] = []
    layers: list[dict[str, Any]] = []
    computes: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []

    common_order: dict[int, str] = {}
    for arm, root in PHASE_ROOTS[phase].items():
        for name, role in (
            ("terminal.json", "top_terminal"),
            ("manifest.json", "result_manifest"),
            ("source-manifest.json", "source_manifest"),
            ("job-timing.json", "job_timing"),
        ):
            artifacts.append(file_fact(root / name, f"{arm}:{role}"))
        top = load(root / "terminal.json")
        if top.get("status") != "TERMINAL_VALID":
            raise RuntimeError(f"{arm} top receipt is not terminal-valid")
        dtype = top["dtype_contract"]
        inventory = dtype["parameter_inventory"]
        if not (
            dtype["status"] == "FULL_FP32_PASS"
            and inventory["non_fp32_parameter_tensor_count"] == 0
            and inventory["bf16_parameter_tensor_count"] == 0
            and inventory["fp16_parameter_tensor_count"] == 0
            and inventory["quantized_parameter_count"] == 0
            and dtype["bf16_fp16_path_count"] == 0
            and dtype["numeric_storage_cast_count"] == 0
        ):
            raise RuntimeError(f"{arm} FULL-FP32 contract failed")
        top_terminals[arm] = top
        job_timings[arm] = load(root / "job-timing.json")
        if not top.get("W0_restored", False):
            raise RuntimeError(f"{arm} terminal W0 restore is not exact")
        if top.get("technical_failure_count", 0) or top.get("scientific_failure_count", 0):
            raise RuntimeError(f"{arm} terminal failure denominator is nonzero")
        if top.get("retry_count", 0) or top.get("imputation_count", 0):
            raise RuntimeError(f"{arm} terminal retry/imputation is nonzero")
        if phase == 2 and (
            top.get("case_count") != 10
            or top.get("valid_request_count") != 1000
            or top.get("K_writer_call_count") != 80
            or top.get("cross_case_state_count") != 0
        ):
            raise RuntimeError(f"{arm} independent Phase2 terminal contract mismatch")
        if phase == 3 and (
            top.get("completed_batch_count") != 10
            or top.get("valid_request_count") != 1000
            or top.get("K_writer_call_count") != 80
            or top.get("alpha_cache_append_count") != 10
            or top.get("same_batch_current_key_history_inclusion_count") != 0
        ):
            raise RuntimeError(f"{arm} sequential Phase3 terminal contract mismatch")
        files = unit_files(phase, root)
        expected_shas = top.get("case_terminal_sha256", top.get("batch_terminal_sha256"))
        if [sha_file(path) for path in files] != expected_shas:
            raise RuntimeError(f"{arm} unit terminal SHA order mismatch")
        for unit, path in enumerate(files, start=1):
            artifacts.append(file_fact(path, f"{arm}:unit_terminal:{unit:02d}"))
            payload = load(path)
            unit_payloads[arm].append(payload)
            if payload["request_count"] != 100 or len(payload["kstep_executions"]) != 8:
                raise RuntimeError(f"{arm} unit {unit} completeness mismatch")
            if payload.get("retry_count", 0) or payload.get("imputation_count", 0):
                raise RuntimeError(f"{arm} unit {unit} retry/imputation is nonzero")
            if phase == 2:
                reference = payload["final_v6_reference"]
                if not (
                    reference["same_entry_W0"]
                    and reference["same_slice_order"]
                    and reference["K1_prewrite_W0_identity"]
                    and reference["same_z_equivalence_claimed"] is False
                ):
                    raise RuntimeError(f"{arm} unit {unit} final-v6 binding mismatch")
            order = payload["request_order_sha256"]
            if unit in common_order and common_order[unit] != order:
                raise RuntimeError(f"cross-arm request-order mismatch at unit {unit}")
            common_order[unit] = order
            previous_commit = None
            for execution in payload["kstep_executions"]:
                # Runtime stores zero-based step_index; reports use scientific K1..K8.
                step = int(execution["step_index"]) + 1
                if step < 1 or step > 8:
                    raise RuntimeError(f"invalid K step {step}")
                if step > 1 and execution["entry_weight_sha256"] != previous_commit:
                    raise RuntimeError(f"closed-loop W chain mismatch {arm} unit {unit} K{step}")
                if not execution["current_K_writer_affects_next_target"]:
                    raise RuntimeError(f"closed-loop influence missing {arm} unit {unit} K{step}")
                previous_commit = execution["commit_weight_sha256"]
                for endpoint_name, endpoint in execution["metrics"].items():
                    if endpoint_name not in ("accepted_z", "pre_writer_W", "post_writer_W"):
                        continue
                    prompt_rows.extend(
                        endpoint_prompt_rows(phase, arm, unit, step, endpoint_name, endpoint)
                    )
                    metric_rows.append(
                        endpoint_metric_row(phase, arm, unit, step, endpoint_name, endpoint)
                    )
                gap = execution["metrics"]["post_W_minus_z_gap"]
                z_scores = execution["metrics"]["accepted_z"]["scores"]
                w_scores = execution["metrics"]["post_writer_W"]["scores"]
                def prompt_loss(metric: str) -> int:
                    before = flatten(z_scores[metric]["per_request_bits"])
                    after = flatten(w_scores[metric]["per_request_bits"])
                    return sum(int(left == 1 and right == 0) for left, right in zip(before, after))
                def strict_loss(metric: str) -> int:
                    before = z_scores[metric]["strict_all_prompt_bits"]
                    after = w_scores[metric]["strict_all_prompt_bits"]
                    return sum(int(left == 1 and right == 0) for left, right in zip(before, after))
                writer = execution["writer_receipt"].get("writer", {})
                writer_core = writer.get("writer", writer)
                k_rows.append(
                    {
                        "phase": phase,
                        "arm": arm,
                        "unit": unit,
                        "K": step,
                        "entry_weight_sha256": execution["entry_weight_sha256"],
                        "commit_weight_sha256": execution["commit_weight_sha256"],
                        "selected_target_sha256": execution["selected_target_sha256"],
                        "accepted_z_sha256": execution["metrics"]["accepted_z"]["binding"]["accepted_z_sha256"],
                        "closed_loop_influence": execution["current_K_writer_affects_next_target"],
                        "rewrite_W_minus_z_target_new_nll": gap["rewrite_W_minus_z_target_new_nll"],
                        "rephrase_W_minus_z_target_new_nll": gap["rephrase_W_minus_z_target_new_nll"],
                        "rewrite_success_z_to_W_failure_count": prompt_loss("rewrite_success"),
                        "rewrite_accuracy_z_to_W_failure_count": prompt_loss("rewrite_acc"),
                        "rephrase_success_z_to_W_failure_count": prompt_loss("rephrase_success"),
                        "rephrase_accuracy_z_to_W_failure_count": prompt_loss("rephrase_acc"),
                        "rewrite_strict_success_z_to_W_failure_count": strict_loss("rewrite_success"),
                        "rewrite_strict_accuracy_z_to_W_failure_count": strict_loss("rewrite_acc"),
                        "rephrase_strict_success_z_to_W_failure_count": strict_loss("rephrase_success"),
                        "rephrase_strict_accuracy_z_to_W_failure_count": strict_loss("rephrase_acc"),
                        "post_terminal_residual_norm": writer_core.get("post_terminal_residual_norm", "NOT_RECORDED"),
                        "total_update_energy": writer_core.get("total_update_energy", "NOT_RECORDED"),
                        "total_update_norm_sum": writer_core.get("total_update_norm_sum", "NOT_RECORDED"),
                        "route_recomputed_at_current_k": execution["writer_receipt"].get("route_recomputed_at_current_k", "NOT_RECORDED"),
                        "route_frozen_across_k_count": execution["writer_receipt"].get("route_frozen_across_k_count", "NOT_RECORDED"),
                        **execution["compute"],
                    }
                )
                routes.append(route_row(phase, arm, unit, step, execution["route_receipt"]))
                layers.extend(layer_rows(phase, arm, unit, step, execution["writer_receipt"]))
            if payload["commit_weight_sha256"] != previous_commit:
                raise RuntimeError(f"unit terminal commit mismatch {arm} unit {unit}")
            computes.append(compute_row(phase, arm, unit, payload))
            if phase == 3:
                cache = payload["alpha_cache"]
                cache_rows.append(
                    {
                        "arm": arm,
                        "batch": unit,
                        "entry_width": cache["entry_width"],
                        "exit_width": cache["exit_width"],
                        "append_width": cache["append_width"],
                        "consume_count": cache["consume_count"],
                        "same_batch_entry_reuse_count": cache["same_batch_entry_cache_reuse_count"],
                        "current_batch_key_inclusion_count": cache["current_batch_k_key_history_inclusion_count"],
                        "current_uncommitted_prefix_key_inclusion_count": cache["current_uncommitted_prefix_key_history_inclusion_count"],
                        "append_count": cache["append_count"],
                        "commit_count": cache["commit_count"],
                        "rollback_count": cache["rollback_count"],
                        "identity_sha256": cache["identity_sha256"],
                    }
                )

    if len(prompt_rows) != 216000 or len(metric_rows) != 720 or len(k_rows) != 240:
        raise RuntimeError(
            f"unexpected row counts prompt={len(prompt_rows)} metric={len(metric_rows)} K={len(k_rows)}"
        )
    if len(layers) != 1200 or len(routes) != 240 or len(computes) != 30:
        raise RuntimeError("writer/route/compute row completeness mismatch")

    # Validate independent/sequential cross-unit physical and cache contracts.
    for arm in ("c0", "c1", "c3"):
        units = unit_payloads[arm]
        if phase == 2:
            entry_identities = [canonical_sha(unit["entry_W0_parameter_sha256"]) for unit in units]
            if len(set(entry_identities)) != 1:
                raise RuntimeError(f"{arm} Phase2 cases do not share exact W0 entry")
            if any(unit["cross_case_W_cache_history_carry_count"] != 0 for unit in units):
                raise RuntimeError(f"{arm} Phase2 cross-case state carry is nonzero")
        else:
            for index, unit in enumerate(units):
                cache = unit["alpha_cache"]
                expected_entry = index * 100
                if not (
                    cache["entry_width"] == expected_entry
                    and cache["exit_width"] == expected_entry + 100
                    and cache["append_width"] == 100
                    and cache["consume_count"] == 8
                    and cache["same_batch_entry_cache_reuse_count"] == 8
                    and cache["current_batch_k_key_history_inclusion_count"] == 0
                    and cache["current_uncommitted_prefix_key_history_inclusion_count"] == 0
                    and cache["append_count"] == 1
                    and cache["commit_count"] == 1
                    and cache["rollback_count"] == 0
                ):
                    raise RuntimeError(f"{arm} Phase3 cache contract mismatch at batch {index + 1}")
                if index and unit["entry_weight_sha256"] != units[index - 1]["commit_weight_sha256"]:
                    raise RuntimeError(f"{arm} Phase3 B-to-B W chain mismatch at batch {index + 1}")

    target_identity_rows = []
    for unit in range(1, 11):
        for step in range(1, 9):
            selected = {
                arm: next(
                    row for row in k_rows
                    if row["arm"] == arm and row["unit"] == unit and row["K"] == step
                )
                for arm in ("c0", "c1", "c3")
            }
            z_hashes = {arm: row["accepted_z_sha256"] for arm, row in selected.items()}
            target_hashes = {arm: row["selected_target_sha256"] for arm, row in selected.items()}
            target_identity_rows.append(
                {
                    "phase": phase,
                    "unit": unit,
                    "K": step,
                    "c0_accepted_z_sha256": z_hashes["c0"],
                    "c1_accepted_z_sha256": z_hashes["c1"],
                    "c3_accepted_z_sha256": z_hashes["c3"],
                    "accepted_z_all_arms_equal": len(set(z_hashes.values())) == 1,
                    "c0_selected_target_sha256": target_hashes["c0"],
                    "c1_selected_target_sha256": target_hashes["c1"],
                    "c3_selected_target_sha256": target_hashes["c3"],
                    "selected_target_all_arms_equal": len(set(target_hashes.values())) == 1,
                }
            )
    if phase == 3 and not (
        target_identity_rows[0]["accepted_z_all_arms_equal"]
        and target_identity_rows[0]["selected_target_all_arms_equal"]
    ):
        raise RuntimeError("Phase3 B1/K1 cross-arm accepted-z identity mismatch")

    final_current: dict[str, dict[str, Any]] = {}
    for arm in ("c0", "c1", "c3"):
        final_current[arm] = aggregate_endpoint(
            prompt_rows,
            metric_rows,
            phase=phase,
            arm=arm,
            endpoint="post_writer_W",
            step=8,
        )

    # Phase 3 canonical endpoint is the B10 terminal model re-evaluated on all cohorts.
    phase3_final_prompts: list[dict[str, Any]] = []
    phase3_final_metrics: list[dict[str, Any]] = []
    if phase == 3:
        for arm, root in PHASE_ROOTS[phase].items():
            final_path = root / "raw/final-w10.json"
            artifacts.append(file_fact(final_path, f"{arm}:final_w10"))
            final_payload = load(final_path)
            if len(final_payload["cohorts"]) != 10 or final_payload["request_count"] != 1000:
                raise RuntimeError(f"{arm} final W10 completeness mismatch")
            for cohort in final_payload["cohorts"]:
                batch = int(cohort.get("batch_index", cohort.get("round")))
                phase3_final_prompts.extend(
                    endpoint_prompt_rows(phase, arm, batch, "FINAL_W10", "final_W10", cohort)
                )
                phase3_final_metrics.append(
                    endpoint_metric_row(phase, arm, batch, "FINAL_W10", "final_W10", cohort)
                )
        prompt_rows.extend(phase3_final_prompts)
        metric_rows.extend(phase3_final_metrics)
        for arm in ("c0", "c1", "c3"):
            final_current[arm] = aggregate_endpoint(
                prompt_rows,
                metric_rows,
                phase=phase,
                arm=arm,
                endpoint="final_W10",
                step="FINAL_W10",
            )

    references: dict[str, dict[str, Any]] = {}
    paired_rows: list[dict[str, Any]] = []
    if phase == 2:
        per_case = load(FINAL_V6 / "per-case.json")
        per_case_rows = per_case["rows"] if isinstance(per_case, dict) else per_case
        nll_refs = load(FINAL_V6 / "nll-prompt-distribution-aggregates.json")
        for arm in ("c0", "c1", "c3"):
            rows = sorted(
                [row for row in per_case_rows if row["method"] == arm.upper()],
                key=lambda row: row["case_index"],
            )
            if len(rows) != 10:
                raise RuntimeError(f"final-v6 paired denominator mismatch for {arm}")
            reference = {
                "phase": 2,
                "arm": arm,
                "endpoint": "FINAL_V6_ONE_SHOT_W",
                "K": "REFERENCE",
                "unit_count": 10,
            }
            for prompt in ("rewrite", "rephrase"):
                for kind in ("success", "accuracy"):
                    reference[f"{prompt}_{kind}_numerator"] = sum(row[f"W_{prompt}_{kind}_numerator"] for row in rows)
                    reference[f"{prompt}_{kind}_denominator"] = sum(row[f"W_{prompt}_{kind}_denominator"] for row in rows)
                for kind in ("strict_success", "strict_accuracy"):
                    reference[f"{prompt}_{kind}_numerator"] = sum(row[f"W_{prompt}_{kind}_numerator"] for row in rows) if prompt == "rephrase" else reference[f"{prompt}_success_numerator"]
                    reference[f"{prompt}_{kind}_denominator"] = sum(row[f"W_{prompt}_{kind}_denominator"] for row in rows) if prompt == "rephrase" else reference[f"{prompt}_success_denominator"]
            reference["locality_numerator"] = sum(row["W_locality_numerator"] for row in rows)
            reference["locality_denominator"] = sum(row["W_locality_denominator"] for row in rows)
            reference["locality_rate"] = rate(reference["locality_numerator"], reference["locality_denominator"])
            for prompt in ("rewrite", "rephrase"):
                for target in ("new", "true"):
                    match = next(
                        row for row in nll_refs
                        if row["method"] == arm.upper()
                        and row["endpoint"] == "W"
                        and row["family"] == prompt
                        and row["target"] == f"target-{target}"
                    )["cross_prompt_summary"]
                    for key, value in match.items():
                        reference[f"{prompt}_target_{target}_{key}"] = value
            references[arm] = reference
            for current, old in zip(
                [unit_payloads[arm][index]["kstep_executions"][-1]["metrics"]["post_writer_W"]["summary"] for index in range(10)],
                rows,
            ):
                unit = int(old["case_index"])
                pair = {
                    "arm": arm,
                    "case": unit,
                    "same_slice_order": unit_payloads[arm][unit - 1]["request_order_sha256"] == old["request_order_sha256"],
                    "same_entry_W0": unit_payloads[arm][unit - 1]["final_v6_reference"]["same_entry_W0"],
                    "same_z_equivalence_claimed": False,
                }
                for key in (
                    "rewrite_success", "rewrite_accuracy", "rephrase_success",
                    "rephrase_strict_success", "rephrase_accuracy",
                    "rephrase_strict_accuracy", "locality",
                ):
                    if key == "locality":
                        current_rate = rate(current["locality_numerator"], current["locality_denominator"])
                        reference_rate = old["W_locality_rate"]
                    else:
                        current_rate = rate(current[f"{key}_numerator"], current[f"{key}_denominator"])
                        reference_rate = old[f"W_{key}_rate"]
                    pair[f"phase2_{key}_rate"] = current_rate
                    pair[f"one_shot_{key}_rate"] = reference_rate
                    pair[f"delta_{key}_percentage_points"] = 100.0 * (current_rate - reference_rate)
                for prompt in ("rewrite", "rephrase"):
                    for target in ("new", "true"):
                        current_mean = current[f"{prompt}_target_{target}_nll_mean"]
                        old_mean = old[f"W_{prompt}_target_{target}_nll_mean"]
                        pair[f"phase2_{prompt}_target_{target}_nll_mean"] = current_mean
                        pair[f"one_shot_{prompt}_target_{target}_nll_mean"] = old_mean
                        pair[f"delta_{prompt}_target_{target}_nll_mean"] = current_mean - old_mean
                current_payload = unit_payloads[arm][unit - 1]
                current_wall = current_payload.get("target_public", {}).get("compute", {}).get("wall_seconds", {})
                current_target_writer = sum(value for value in current_wall.values() if isinstance(value, (int, float)))
                pair["phase2_case_total_seconds"] = current_payload["case_total_seconds"]
                pair["one_shot_case_total_seconds"] = old["case_total_seconds"]
                pair["delta_case_total_seconds"] = pair["phase2_case_total_seconds"] - pair["one_shot_case_total_seconds"]
                pair["case_total_ratio"] = pair["phase2_case_total_seconds"] / pair["one_shot_case_total_seconds"]
                pair["phase2_recorded_phase_component_wall_sum"] = current_target_writer
                pair["one_shot_target_plus_writer_seconds"] = old["target_plus_writer_seconds"]
                pair["target_writer_timing_comparison_status"] = "NOT_COMPARABLE_DIFFERENT_TIMER_BOUNDARY"
                pair["phase2_kstep_writer_segment_seconds"] = current_wall.get("c_kstep_writer", "NOT_RECORDED")
                pair["one_shot_writer_edit_core_seconds"] = old["writer_edit_core_seconds"]
                current_energy = [
                    item["total_update_energy"] for item in k_rows
                    if item["arm"] == arm and item["unit"] == unit and isinstance(item["total_update_energy"], (int, float))
                ]
                pair["phase2_total_recorded_update_energy"] = sum(current_energy) if current_energy else "NOT_RECORDED"
                pair["one_shot_total_actual_fp32_update_energy"] = old["total_actual_fp32_update_energy"]
                pair["delta_total_update_energy"] = (
                    sum(current_energy) - old["total_actual_fp32_update_energy"]
                    if current_energy else "NOT_RECORDED"
                )
                paired_rows.append(pair)
    else:
        phase1_batch_rows = load(PHASE1_REPORT_V1 / "per-arm-per-batch-summary.json")["rows"]
        for arm in ("c0", "c1", "c3"):
            reference_final_path = (
                PHASE1_REPO / "local/odebf/results" / PHASE1_RESULT_NAMES[arm] / "raw/final-w10.json"
            )
            artifacts.append(file_fact(reference_final_path, f"{arm}:phase1_reference_final_w10"))
            reference_payload = load(reference_final_path)
            reference_prompt_rows: list[dict[str, Any]] = []
            reference_metric_rows: list[dict[str, Any]] = []
            for cohort in reference_payload["cohorts"]:
                batch = int(cohort.get("batch_index", cohort.get("round")))
                reference_prompt_rows.extend(
                    endpoint_prompt_rows(3, f"{arm}_phase1_ref", batch, "REFERENCE", "phase1_final_W10", cohort)
                )
                reference_metric_rows.append(
                    endpoint_metric_row(3, f"{arm}_phase1_ref", batch, "REFERENCE", "phase1_final_W10", cohort)
                )
            reference = aggregate_endpoint(
                reference_prompt_rows,
                reference_metric_rows,
                phase=3,
                arm=f"{arm}_phase1_ref",
                endpoint="phase1_final_W10",
                step="REFERENCE",
            )
            reference["arm"] = arm
            reference["endpoint"] = "PHASE1_FINAL_W10_ONE_WRITE_PER_BATCH"
            references[arm] = reference
            phase1_rows = sorted(
                [row for row in phase1_batch_rows if row["arm"] == arm],
                key=lambda row: row["batch"],
            )
            phase3_final_payload = load(PHASE_ROOTS[3][arm] / "raw/final-w10.json")
            for current, old in zip(phase3_final_payload["cohorts"], phase1_rows):
                summary = current["summary"]
                endpoint = old["W_final_B10"]
                unit = int(current["batch_index"])
                if unit != old["batch"]:
                    raise RuntimeError("Phase1/Phase3 batch pairing mismatch")
                same_order = unit_payloads[arm][unit - 1]["request_order_sha256"] == old["request_order_sha256"]
                if not same_order:
                    raise RuntimeError(f"Phase1/Phase3 request-order mismatch {arm} batch {unit}")
                pair = {"arm": arm, "batch": unit, "same_request_order": same_order}
                mapping = {
                    "rewrite_success": "rewrite_success", "rewrite_accuracy": "rewrite_accuracy",
                    "rephrase_success": "rephrase_success", "rephrase_accuracy": "rephrase_accuracy",
                }
                for key, old_key in mapping.items():
                    current_rate = rate(summary[f"{key}_numerator"], summary[f"{key}_denominator"])
                    old_rate = endpoint[old_key]["prompt_rate"]
                    pair[f"phase3_{key}_rate"] = current_rate
                    pair[f"phase1_{key}_rate"] = old_rate
                    pair[f"delta_{key}_percentage_points"] = 100.0 * (current_rate - old_rate)
                for key in ("rephrase_strict_success", "rephrase_strict_accuracy"):
                    old_key = "rephrase_success" if key.endswith("success") else "rephrase_accuracy"
                    current_rate = rate(summary[f"{key}_numerator"], summary[f"{key}_denominator"])
                    old_rate = endpoint[old_key]["strict_rate"]
                    pair[f"phase3_{key}_rate"] = current_rate
                    pair[f"phase1_{key}_rate"] = old_rate
                    pair[f"delta_{key}_percentage_points"] = 100.0 * (current_rate - old_rate)
                pair["phase3_locality_rate"] = rate(summary["locality_numerator"], summary["locality_denominator"])
                pair["phase1_locality_rate"] = endpoint["locality"]["rate"]
                pair["delta_locality_percentage_points"] = 100.0 * (pair["phase3_locality_rate"] - pair["phase1_locality_rate"])
                for prompt in ("rewrite", "rephrase"):
                    metric = endpoint[f"{prompt}_accuracy"]
                    for target in ("new", "true"):
                        current_mean = summary[f"{prompt}_target_{target}_nll_mean"]
                        old_mean = metric[target]["mean"]
                        pair[f"phase3_{prompt}_target_{target}_nll_mean"] = current_mean
                        pair[f"phase1_{prompt}_target_{target}_nll_mean"] = old_mean
                        pair[f"delta_{prompt}_target_{target}_nll_mean"] = current_mean - old_mean
                current_payload = unit_payloads[arm][unit - 1]
                current_seconds = current_payload["batch_total_seconds"]
                old_seconds = old["timing"]["case_total_seconds"]
                pair["phase3_batch_total_seconds"] = current_seconds
                pair["phase1_batch_total_seconds"] = old_seconds
                pair["delta_case_total_seconds"] = current_seconds - old_seconds
                pair["case_total_ratio"] = current_seconds / old_seconds
                current_energy = [
                    item["total_update_energy"] for item in k_rows
                    if item["arm"] == arm and item["unit"] == unit and isinstance(item["total_update_energy"], (int, float))
                ]
                pair["phase3_total_recorded_update_energy"] = sum(current_energy) if current_energy else "NOT_RECORDED"
                pair["phase1_writer_update_energy"] = old["writer_total_update_energy"]
                pair["delta_total_update_energy"] = (
                    sum(current_energy) - old["writer_total_update_energy"]
                    if current_energy else "NOT_RECORDED"
                )
                paired_rows.append(pair)

    # Common W0 observation from the accepted Phase 1 pre-edit run.
    rewrite_pre = load(PHASE1_REPORT_V4 / "rewrite-endpoint-summary.json")["rows"][0]
    rephrase_pre = load(PHASE1_REPORT_V4 / "rephrase-endpoint-summary.json")["rows"][0]
    pre_edit = {
        "label": "Pre-edit (공통 W0)",
        "endpoint": "COMMON_W0_OBSERVATION_ONLY",
        "rewrite_success_numerator": rewrite_pre["success_numerator"],
        "rewrite_success_denominator": rewrite_pre["success_denominator"],
        "rewrite_accuracy_numerator": rewrite_pre["accuracy_numerator"],
        "rewrite_accuracy_denominator": rewrite_pre["accuracy_denominator"],
        "rewrite_strict_success_numerator": rewrite_pre["success_strict_numerator"],
        "rewrite_strict_success_denominator": rewrite_pre["success_strict_denominator"],
        "rewrite_strict_accuracy_numerator": rewrite_pre["accuracy_strict_numerator"],
        "rewrite_strict_accuracy_denominator": rewrite_pre["accuracy_strict_denominator"],
        "rephrase_success_numerator": rephrase_pre["success_numerator"],
        "rephrase_success_denominator": rephrase_pre["success_denominator"],
        "rephrase_accuracy_numerator": rephrase_pre["accuracy_numerator"],
        "rephrase_accuracy_denominator": rephrase_pre["accuracy_denominator"],
        "rephrase_strict_success_numerator": rephrase_pre["success_strict_numerator"],
        "rephrase_strict_success_denominator": rephrase_pre["success_strict_denominator"],
        "rephrase_strict_accuracy_numerator": rephrase_pre["accuracy_strict_numerator"],
        "rephrase_strict_accuracy_denominator": rephrase_pre["accuracy_strict_denominator"],
        "locality_numerator": 8839,
        "locality_denominator": 10000,
        "locality_rate": 0.8839,
    }
    for prompt, source in (("rewrite", rewrite_pre), ("rephrase", rephrase_pre)):
        for target in ("new", "true"):
            for key in ("count", "mean", "median", "p90", "max", "min"):
                pre_edit[f"{prompt}_target_{target}_{key}"] = source[f"target_{target}_{key}"]

    for arm, row in final_current.items():
        row["label"] = LABELS[arm] + ("-CACHE final W10" if phase == 3 else " terminal K8 W")
    for arm, row in references.items():
        row["label"] = (
            f"{arm.upper()} final-v6 one-shot"
            if phase == 2
            else f"Phase1 {arm.upper()} final W10 (one write/B100)"
        )

    compute_aggregate = aggregate_compute(computes, top_terminals, job_timings)
    aggregate_rows = [pre_edit]
    for arm in ("c0", "c1", "c3"):
        aggregate_rows.extend([final_current[arm], references[arm]])

    comparison_aggregate: list[dict[str, Any]] = []
    for arm in ("c0", "c1", "c3"):
        current = final_current[arm]
        reference = references[arm]
        row = {"arm": arm, "paired_unit_count": 10, "comparison": "PHASE2_VS_FINAL_V6_ONE_SHOT" if phase == 2 else "PHASE3_VS_PHASE1"}
        current_rates, reference_rates = metric_rates(current), metric_rates(reference)
        for key in current_rates:
            row[f"current_{key}"] = current_rates[key]
            row[f"reference_{key}"] = reference_rates[key]
            row[f"delta_{key}_percentage_points"] = 100.0 * (current_rates[key] - reference_rates[key])
        for prompt in ("rewrite", "rephrase"):
            for target in ("new", "true"):
                key = f"{prompt}_target_{target}_mean"
                row[f"current_{key}"] = current[key]
                row[f"reference_{key}"] = reference[key]
                row[f"delta_{key}"] = current[key] - reference[key]
        pairset = [item for item in paired_rows if item["arm"] == arm]
        for key in [name for name in pairset[0] if name.startswith("delta_")]:
            numbers = [item[key] for item in pairset if isinstance(item[key], (int, float))]
            if not numbers:
                continue
            for stat_key, value in stats(numbers).items():
                row[f"paired_{key}_{stat_key}"] = value
        ratio_keys = [name for name in pairset[0] if name.endswith("_ratio")]
        for key in ratio_keys:
            numbers = [item[key] for item in pairset if isinstance(item[key], (int, float))]
            for stat_key, value in stats(numbers).items():
                row[f"paired_{key}_{stat_key}"] = value
        comparison_aggregate.append(row)

    outputs: list[Path] = []
    table_specs = [
        ("method-definitions", methods),
        ("per-unit-endpoint-summary", metric_rows),
        ("per-k-endpoint-and-compute", k_rows),
        ("per-k-route", routes),
        ("per-layer-writer-update-energy", layers),
        ("per-unit-compute", computes),
        ("compute-aggregate", compute_aggregate),
        ("terminal-endpoint-aggregate", aggregate_rows),
        ("paired-unit-comparison", paired_rows),
        ("paired-comparison-aggregate", comparison_aggregate),
        ("cross-arm-target-identity", target_identity_rows),
    ]
    if phase == 3:
        table_specs.append(("cache-sequential-audit", cache_rows))
    for stem, rows in table_specs:
        outputs.append(dump_json(directory, f"{stem}.json", {"schema": f"phase{phase}/{stem}/v1", "row_count": len(rows), "rows": rows}))
        outputs.append(dump_csv(directory, f"{stem}.csv", rows))
    outputs.append(dump_csv(directory, "per-prompt-kstep-nll.csv", prompt_rows))

    # Aggregate NLL by K/endpoint/arm/prompt/target without omitting the raw-free prompt table.
    nll_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in prompt_rows:
        for target in ("new", "true"):
            nll_groups[(row["arm"], row["K"], row["endpoint"], row["prompt"], target)].append(row[f"target_{target}_nll"])
    nll_rows = []
    for key, values in sorted(nll_groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        arm, step, endpoint, prompt, target = key
        nll_rows.append({"arm": arm, "K": step, "endpoint": endpoint, "prompt": prompt, "target": target, **stats(values)})
    outputs.append(dump_json(directory, "nll-distribution-aggregates.json", {"schema": f"phase{phase}/nll-distribution/v1", "row_count": len(nll_rows), "rows": nll_rows}))
    outputs.append(dump_csv(directory, "nll-distribution-aggregates.csv", nll_rows))

    integrity = {
        "schema": f"phase{phase}/terminal-integrity/v1",
        "status": "TERMINAL_VALID_EXHAUSTIVE_ANALYSIS_INPUT_PASS",
        "arms": 3,
        "units_per_arm": 10,
        "requests_per_unit": 100,
        "valid_requests": 3000,
        "K_per_unit": 8,
        "K_receipts": 240,
        "writer_calls": 240,
        "layer_rows": 1200,
        "prompt_nll_rows": len(prompt_rows),
        "technical_failure_count": 0,
        "scientific_failure_count": 0,
        "retry_count": 0,
        "imputation_count": 0,
        "full_fp32_arms": 3,
        "bf16_fp16_path_count": 0,
        "numeric_storage_cast_count": 0,
        "autocast_enabled_count": 0,
        "quantized_parameter_count": 0,
        "request_order_sha256_by_unit": common_order,
        "phase_semantics": "INDEPENDENT_W0_CACHE_OFF" if phase == 2 else "SEQUENTIAL_W_CACHE_CONTINUITY",
        "comparison_boundary": "FINAL_V6_ONE_SHOT_SAME_ENTRY_SLICE_ORDER_NO_SAME_Z_CLAIM" if phase == 2 else "PHASE1_SAME_STREAM_SEQUENTIAL_FINAL_W10",
        "scientific_promotion": False,
    }
    outputs.append(dump_json(directory, "terminal-integrity.json", integrity))

    artifact_inventory = {
        "schema": f"phase{phase}/artifact-inventory/v1",
        "row_count": len(artifacts),
        "rows": artifacts,
        "root_sha256": canonical_sha(artifacts),
    }
    outputs.append(dump_json(directory, "artifact-inventory.json", artifact_inventory))
    outputs.append(dump_csv(directory, "artifact-inventory.csv", artifacts))

    first_target_divergence = next(
        (
            (row["unit"], row["K"])
            for row in target_identity_rows
            if not (row["accepted_z_all_arms_equal"] and row["selected_target_all_arms_equal"])
        ),
        None,
    )
    equal_target_count = sum(
        row["accepted_z_all_arms_equal"] and row["selected_target_all_arms_equal"]
        for row in target_identity_rows
    )

    lines = [
        f"# P1R52 C-writer Phase {phase} FULL-FP32 — 최종 상세 사실 보고서",
        "",
        (
            "> **최우선 실행 경계:** Phase 2는 각 independent B100 안에서 K1→K8마다 `W_(k-1) → accepted-z_k → writer_k → W_k`를 수행한다. 비교 대상 final-v6는 동일 W0/slice/order에서 accepted-z를 K8까지 완성한 뒤 writer를 한 번 적용한다. 따라서 same-z equivalence는 주장하지 않는다."
            if phase == 2
            else "> **최우선 실행 경계:** Phase 3는 B1→B10 sequential W/Alpha-cache continuity를 유지하면서 각 B100의 K1→K8마다 writer를 적용한다. 비교 대상 Phase 1은 같은 stream/cache sequential 경계에서 accepted-z를 K8까지 완성한 뒤 B100당 writer를 한 번만 적용한다."
        ),
        "",
        "> 세 arm 모두 model/algorithm/update/cache가 FULL FP32다. BF16/FP16/autocast/quantization/numeric storage cast=0, retry=0, imputation=0, scientific promotion=false다.",
        "",
        "## 1. 최종 endpoint edit 성능",
        "",
    ]
    endpoint_table(lines, aggregate_rows)
    lines += [
        "",
        "## 2. 이번 Phase에 추가된 방법 정의",
        "",
        "|방법|target|writer|route|cache/실행 경계|",
        "|---|---|---|---|---|",
    ]
    for row in methods:
        lines.append(f"|{row['label']}|{row['target']}|{row['writer']}|{row['route']}|{row['cache_status']}; {row['execution_scope']}|")
    lines += ["", "## 3. Rewrite NLL", ""]
    nll_table(lines, aggregate_rows, "rewrite")
    lines += ["", "## 4. Rephrase NLL", ""]
    nll_table(lines, aggregate_rows, "rephrase")
    lines += [
        "",
        "## 5. Paired 기준 비교",
        "",
        "|arm|EFF Δpp|GEN Δpp|strict GEN Δpp|LOC Δpp|rewrite new NLL Δ|rephrase new NLL Δ|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_aggregate:
        lines.append(
            f"|{row['arm']}|{row['delta_rewrite_success_rate_percentage_points']:+.3f}|"
            f"{row['delta_rephrase_success_rate_percentage_points']:+.3f}|"
            f"{row['delta_rephrase_strict_success_rate_percentage_points']:+.3f}|"
            f"{row['delta_locality_rate_percentage_points']:+.3f}|"
            f"{row['delta_rewrite_target_new_mean']:+.6f}|"
            f"{row['delta_rephrase_target_new_mean']:+.6f}|"
        )
    lines += [
        "",
        "|arm|paired case total Δ mean sec|paired case total ratio mean|target/writer timing comparison|update energy Δ mean|",
        "|---|---:|---:|---|---:|",
    ]
    for row in comparison_aggregate:
        def fmt_value(key: str, signed: bool = False) -> str:
            value = row.get(key, "NOT_RECORDED")
            if not isinstance(value, (int, float)):
                return str(value)
            return f"{value:+.6f}" if signed else f"{value:.6f}"
        lines.append(
            f"|{row['arm']}|{fmt_value('paired_delta_case_total_seconds_mean', True)}|"
            f"{fmt_value('paired_case_total_ratio_mean')}|"
            f"{'NOT_COMPARABLE_DIFFERENT_TIMER_BOUNDARY' if phase == 2 else 'NOT_RECORDED_PER_BATCH'}|"
            f"{fmt_value('paired_delta_total_update_energy_mean', True)}|"
        )
    paired_boundary_note = (
        "모든 비교는 동일한 10/10 unit denominator를 사용한다. Phase 2는 "
        "K-step feedback으로 accepted-z trajectory가 달라질 수 있으므로 "
        "same-entry/slice/order와 K1 pre-write identity만 결속한다."
        if phase == 2
        else
        "모든 비교는 동일한 10/10 batch denominator와 sequential stream/order를 "
        "사용하며, Phase 3와 Phase 1의 최종 W10 endpoint를 paired 비교한다."
    )
    lines += [
        "",
        paired_boundary_note,
        "",
        "## 6. K1→K8 trajectory와 W−z gap",
        "",
        "`per-k-endpoint-and-compute.*`, `per-unit-endpoint-summary.*`, `per-prompt-kstep-nll.csv`에 240개 K receipt와 모든 accepted-z/pre-W/post-W prompt NLL을 보존했다. 각 K에서 post-W가 다음 K의 entry-W와 SHA로 연결됨을 fail-close 검증했다.",
        "",
        f"`cross-arm-target-identity.*` 기준 80개 unit×K 중 세 arm accepted-z/selected-target가 모두 같은 지점은 {equal_target_count}개다. 최초 writer-feedback divergence는 "
        + (f"unit {first_target_divergence[0]} K{first_target_divergence[1]}" if first_target_divergence else "없음")
        + "이다. 이 표는 차이를 결함으로 간주하지 않고, 공통 entry에서 시작한 뒤 writer별 W trajectory가 target refresh를 바꾸는 시점을 기록한다.",
        "",
        "|arm|K8 rewrite W−z mean|K8 rephrase W−z mean|K8 z→W rewrite loss|K8 z→W rephrase loss/strict|K8 update energy mean|K8 residual mean|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("c0", "c1", "c3"):
        selected = [row for row in k_rows if row["arm"] == arm and row["K"] == 8]
        def recorded(field: str) -> list[float]:
            return [float(row[field]) for row in selected if isinstance(row[field], (int, float))]
        lines.append(
            f"|{arm}|{stats(recorded('rewrite_W_minus_z_target_new_nll'))['mean']:.6f}|"
            f"{stats(recorded('rephrase_W_minus_z_target_new_nll'))['mean']:.6f}|"
            f"{sum(row['rewrite_success_z_to_W_failure_count'] for row in selected)}|"
            f"{sum(row['rephrase_success_z_to_W_failure_count'] for row in selected)}/{sum(row['rephrase_strict_success_z_to_W_failure_count'] for row in selected)}|"
            f"{stats(recorded('total_update_energy'))['mean'] if recorded('total_update_energy') else 'NOT_RECORDED'}|"
            f"{stats(recorded('post_terminal_residual_norm'))['mean'] if recorded('post_terminal_residual_norm') else 'NOT_RECORDED'}|"
        )
    lines += [
        "",
        "C3의 Official native apply receipt에는 C0/C1과 동형인 layer별 update norm/energy가 기록되지 않아 해당 400개 layer placeholder는 `NOT_RECORDED_NATIVE_OFFICIAL_APPLY`로 명시했다. 시간이나 다른 aggregate에서 이를 역추정하지 않았다.",
        "",
        "## 7. P/C route와 writer layer telemetry",
        "",
        "C0는 PIR-U control route라 P/C router call=0, C1은 current-K Joint-PC solver call=1/K, C3는 P/C router call=0이다. `per-k-route.*`는 C1의 selected π/P/C/minimax/KKT telemetry와 C0 control allocation, C3 direct-writer status를 분리한다. `per-layer-writer-update-energy.*`는 C0/C1의 L4→L8 residual/q/update/rounding observation을 기록한다.",
        "",
        "FP32 actual delta는 `fl32(W+U)-W`이므로 prepared U와 byte-identical일 필요가 없다. rounding mismatch는 observation-only이고 route decision influence=0이다. Official assignment endpoint 자체는 reference expression과 byte-exact다.",
        "",
        "## 8. 계산량·시간",
        "",
        "|arm|unit total sum/mean/median/p90/max sec|recorded phase-component wall sum|K-step writer segment sum|job sec|eval accepted/pre/post|dense/apply/commit|router|peak alloc/reserved GiB|MaxRSS GiB|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in compute_aggregate:
        gib = 1024 ** 3
        target_wall_text = (
            row["target_wall_seconds_total_sum"]
            if isinstance(row["target_wall_seconds_total_sum"], str)
            else f"{row['target_wall_seconds_total_sum']:.2f}"
        )
        writer_wall_text = (
            row["kstep_writer_wall_seconds_sum"]
            if isinstance(row["kstep_writer_wall_seconds_sum"], str)
            else f"{row['kstep_writer_wall_seconds_sum']:.2f}"
        )
        lines.append(
            f"|{row['arm']}|{row['unit_total_seconds_sum']:.2f}/{row['unit_total_seconds_mean']:.2f}/{row['unit_total_seconds_median']:.2f}/{row['unit_total_seconds_p90']:.2f}/{row['unit_total_seconds_max']:.2f}|"
            f"{target_wall_text}|{writer_wall_text}|"
            f"{row['job_total_seconds']:.2f}|{row['accepted_z_evaluator_count']}/{row['pre_writer_evaluator_count']}/{row['post_writer_evaluator_count']}|"
            f"{row['dense_update_construction_count']}/{row['native_apply_count']}/{row['logical_commit_count']}|{row['router_call_count']}|"
            f"{float(row['peak_allocated_bytes'])/gib:.2f}/{float(row['peak_reserved_bytes'])/gib:.2f}|{float(row['host_maxrss_kib'])/1024**2:.2f}|"
        )
    lines += ["", (
        "Phase 2는 per-case target phase-component wall과 `c_kstep_writer` segment를 기록한다. component timer 합은 case total과 반드시 분할합 관계가 아니므로 final-v6의 `target_plus_writer`와 ratio를 만들지 않고 NOT_COMPARABLE로 둔다. C3만 Official writer receipt 자체의 edit-core wall도 별도로 기록하며 C0/C1에 없는 동형 writer-only timer는 추정하지 않는다."
        if phase == 2
        else "Phase 3 raw batch receipts에는 Phase 2와 동일한 per-batch target phase wall decomposition이 직렬화되지 않았다. 따라서 Phase 3의 per-batch target/writer 세부 wall은 `NOT_RECORDED_PER_BATCH`이고, recorded unit/job total과 nested logical counts만 보고한다."
    ) + " wall time만으로 FLOPs나 원인을 역추정하지 않는다."]
    if phase == 3:
        lines += [
            "",
            "## 9. Alpha-cache sequential 감사",
            "",
            "B1→B10 entry width는 arm마다 0,100,…,900이고 각 B에서 동일 batch-entry snapshot을 K1→K8 8회 재사용한다. current batch/prefix key inclusion=0, B 성공 뒤 append=1, 전체 append=10, rollback=0이다. Structural-H history와 Alpha cache는 분리되어 있다.",
        ]
    lines += [
        "",
        f"## {10 if phase == 3 else 9}. 사실 판정",
        "",
        "이 보고서는 efficacy/generalization/locality/NLL/update/route/cache/compute의 관측값과 paired 차이를 제공한다. 새 threshold나 자동 promotion을 추가하지 않았으며 `scientific_promotion=false`다. 성능 차이는 해당 실행 경계의 사실 비교이지 단일 구성요소에 대한 인과 주장으로 해석하지 않는다.",
        "",
        f"## {11 if phase == 3 else 10}. 재현 산출물",
        "",
        "- `method-definitions.*`: 추가 method와 실행 경계",
        "- `terminal-endpoint-aggregate.*`: pre-edit/current/reference endpoint 종합",
        "- `per-unit-endpoint-summary.*`: unit×K×endpoint summary",
        "- `per-prompt-kstep-nll.csv`: 모든 prompt NLL/bit",
        "- `nll-distribution-aggregates.*`: K/endpoint별 mean/median/p90/max/min",
        "- `per-k-route.*`, `per-layer-writer-update-energy.*`: route/layer telemetry",
        "- `per-unit-compute.*`, `compute-aggregate.*`: 계산량·시간·메모리",
        "- `paired-unit-comparison.*`, `paired-comparison-aggregate.*`: 10/10 paired 비교",
        "- `cross-arm-target-identity.*`: accepted-z/selected-target 동일성과 최초 divergence",
        "- `artifact-inventory.*`, `analysis-manifest.json`, `rooted-analysis-receipt.json`: 재해시 결속",
    ]
    if phase == 3:
        lines.append("- `cache-sequential-audit.*`: B1→B10 cache entry/consume/append/rollback")

    report_name = (
        "p1r52-c-writer-phase2-independent-kstep-full-fp32-exhaustive-factual-ko.md"
        if phase == 2
        else "p1r52-c-writer-phase3-kstep-cache-sequential-full-fp32-exhaustive-factual-ko.md"
    )
    report_path = write_once(directory / report_name, ("\n".join(lines) + "\n").encode())
    outputs.append(report_path)

    builder_fact = file_fact(Path(__file__).resolve(), "analysis_builder")
    member_facts = [builder_fact] + [file_fact(path, "analysis_output") for path in outputs]
    manifest = {
        "schema": f"phase{phase}/exhaustive-analysis-manifest/v1",
        "source_head": "034b72c59aace422b7048509b336bac26992d90e",
        "source_tree": "76f3b8e0b3304dcdca25a8467439c4bf035bb41c",
        "input_artifact_inventory_root_sha256": artifact_inventory["root_sha256"],
        "members": member_facts,
        "member_root_sha256": canonical_sha(member_facts),
        "scientific_promotion": False,
    }
    manifest_path = dump_json(directory, "analysis-manifest.json", manifest)
    receipt = {
        "schema": f"phase{phase}/rooted-analysis-receipt/v1",
        "status": "TERMINAL_VALID_EXHAUSTIVE_ANALYSIS_COMPLETE",
        "report": {"path": str(report_path), "sha256": sha_file(report_path)},
        "manifest": {"path": str(manifest_path), "sha256": sha_file(manifest_path)},
        "analysis_member_root_sha256": manifest["member_root_sha256"],
        "input_artifact_inventory_root_sha256": artifact_inventory["root_sha256"],
        "row_counts": {
            "methods": len(methods), "units": 30, "K": 240,
            "endpoint_summaries": len(metric_rows), "prompt_nll": len(prompt_rows),
            "routes": len(routes), "layers": len(layers), "paired_units": len(paired_rows),
            "cache": len(cache_rows),
        },
        "technical_failures": 0,
        "scientific_failures": 0,
        "retries": 0,
        "imputations": 0,
        "scientific_promotion": False,
    }
    receipt["identity_sha256"] = canonical_sha(receipt)
    dump_json(directory, "rooted-analysis-receipt.json", receipt)


def main() -> None:
    build_phase(2)
    build_phase(3)


if __name__ == "__main__":
    main()
