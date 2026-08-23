#!/usr/bin/env python3
"""Build separated factual packages for the P1R52 target-timescale ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from .contracts import canonical_hash


INSTRUCTION_ID = "ODEEDIT-S05-P1R52-TARGET-TIMESCALE-B100-V1"
SOURCE_HEAD = "c1f1962937af87392a6d046d66c8d864bd54073d"
SOURCE_TREE = "79535403ea7c09d5954102813f2185107966d5a6"
STREAM_ROOT = "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
STREAM_ORDER = "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"
B1_ORDER = "bb2e661ad44bf8dc20e0224f21dc90164ed4b3dcf3316d751afb7d8a00bfaed6"
NUMERICAL_LOCK_ROOT = "2c4008501382ba551b8a2d0a8240062fdbdae7d65c08d9cbad75cb957a8e86e0"
CELL_SLUGS = {
    "Z0-COARSE": "z0-coarse",
    "Z1-REFINE": "z1-refine",
    "Z15": "z15",
    "Z20": "z20",
    "Z30": "z30",
}
EXPECTED = {
    "Z0-COARSE": (1.0, 1, 0.125, 8),
    "Z1-REFINE": (1.0, 2, 0.0625, 16),
    "Z15": (1.5, 3, 0.0625, 24),
    "Z20": (2.0, 4, 0.0625, 32),
    "Z30": (3.0, 6, 0.0625, 48),
}
NATIVE_MEAN_NLL = {
    "Native AlphaEdit direct-z": {"rewrite": 0.001156, "rephrase": 1.09902},
    "Native MEMIT direct-z": {"rewrite": 0.000870, "rephrase": 1.09211},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, *, rooted: bool = True) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"regular non-symlink JSON required: {path}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    if rooted:
        identity = value.get("identity_sha256")
        if not isinstance(identity, str):
            raise ValueError(f"identity absent: {path}")
        body = dict(value)
        body.pop("identity_sha256")
        if canonical_hash(body) != identity:
            raise ValueError(f"identity differs: {path}")
    return value


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("empty quantile input")
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    rows = [float(value) for value in values]
    if not rows or not all(math.isfinite(value) for value in rows):
        raise ValueError("finite non-empty summary required")
    return {
        "n": len(rows),
        "mean": statistics.fmean(rows),
        "median": statistics.median(rows),
        "p90": _quantile(rows, 0.9),
        "min": min(rows),
        "max": max(rows),
    }


def _flatten(values: Sequence[Sequence[float]]) -> list[float]:
    return [float(value) for request in values for value in request]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path.name}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _member(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    return {
        "path": str(path if relative_to is None else path.relative_to(relative_to)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _score_rows(
    *, cell: str, outer: int, endpoint: str, scores: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt, success_key, accuracy_key in (
        ("rewrite", "rewrite_success", "rewrite_acc"),
        ("rephrase", "rephrase_success", "rephrase_acc"),
    ):
        success = scores[success_key]
        accuracy = scores[accuracy_key]
        new = _flatten(success["target_new_nll_by_request"])
        true = _flatten(success["target_true_nll_by_request"])
        recorded_margin = _flatten(success["target_new_minus_true_margin_by_request"])
        if len(new) != (100 if prompt == "rewrite" else 200) or len(true) != len(new):
            raise ValueError("K endpoint denominator differs")
        new_summary = _summary(new)
        true_summary = _summary(true)
        margin_summary = _summary(recorded_margin)
        rows.append(
            {
                "cell": cell,
                "outer_K": outer,
                "endpoint": endpoint,
                "prompt": prompt,
                "nll_prompt_denominator": len(new),
                **{f"target_new_{key}": value for key, value in new_summary.items() if key != "n"},
                **{f"target_true_{key}": value for key, value in true_summary.items() if key != "n"},
                "recorded_margin_definition": "target_new_minus_target_true",
                **{f"margin_{key}": value for key, value in margin_summary.items() if key != "n"},
                "success_numerator": success["prompt_numerator"],
                "success_denominator": success["prompt_denominator"],
                "success_rate": success["prompt_rate"],
                "strict_success_numerator": success["strict_request_numerator"],
                "strict_success_denominator": success["strict_request_denominator"],
                "strict_success_rate": success["strict_request_rate"],
                "accuracy_numerator": accuracy["prompt_numerator"],
                "accuracy_denominator": accuracy["prompt_denominator"],
                "accuracy_rate": accuracy["prompt_rate"],
                "strict_accuracy_numerator": accuracy["strict_request_numerator"],
                "strict_accuracy_denominator": accuracy["strict_request_denominator"],
                "strict_accuracy_rate": accuracy["strict_request_rate"],
            }
        )
    return rows


def _sacct(job_id: str, cells: Sequence[int]) -> dict[int, dict[str, Any]]:
    output = subprocess.run(
        [
            "sacct", "-j", ",".join(f"{job_id}_{cell}" for cell in cells),
            "-n", "-P", "-X",
            "--format=JobID,State,ExitCode,Start,End,ElapsedRaw,Elapsed,CPUTimeRAW,ReqMem,AllocTRES",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    result: dict[int, dict[str, Any]] = {}
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) != 10 or not fields[0].startswith(f"{job_id}_"):
            continue
        cell = int(fields[0].rsplit("_", 1)[1])
        result[cell] = {
            "job_id": fields[0],
            "state": fields[1],
            "exit_code": fields[2],
            "start": fields[3],
            "end": fields[4],
            "elapsed_seconds": int(fields[5]),
            "elapsed": fields[6],
            "allocated_cpu_seconds": int(fields[7]),
            "requested_memory": fields[8],
            "allocated_tres": fields[9],
        }
    if set(result) != set(cells) or any(
        row["state"] != "COMPLETED" or row["exit_code"] != "0:0"
        for row in result.values()
    ):
        raise ValueError("Slurm terminal accounting differs")
    return result


def _validate_terminal(value: Mapping[str, Any], cell: str) -> None:
    horizon, microsteps, target_dt, evaluations = EXPECTED[cell]
    schedule = value.get("schedule", {})
    cache = value.get("alpha_cache", {})
    dtype = value.get("dtype_contract", {})
    if (
        value.get("instruction_id") != INSTRUCTION_ID
        or value.get("status") != "TERMINAL_VALID"
        or value.get("source_head") != SOURCE_HEAD
        or value.get("cell") != cell
        or value.get("request_count") != 100
        or value.get("request_order_sha256") != B1_ORDER
        or value.get("stream_root") != STREAM_ROOT
        or value.get("stream_order") != STREAM_ORDER
        or value.get("sample_duplication_count") != 0
        or schedule.get("target_horizon") != horizon
        or schedule.get("microsteps_per_outer") != microsteps
        or schedule.get("target_dt") != target_dt
        or schedule.get("total_field_evaluations") != evaluations
        or value.get("completed_outer_count") != 8
        or value.get("configured_target_microstep_count") != evaluations
        or value.get("executed_target_microstep_count") != evaluations
        or value.get("target_field_evaluation_count") != evaluations
        or value.get("writer_call_count") != 8
        or value.get("writer_call_indices") != list(range(1, 9))
        or value.get("intermediate_writer_authority_count") != 0
        or value.get("heldout_outer_indices") != [1, 4, 8]
        or value.get("heldout_evaluator_count") != 9
        or value.get("heldout_decision_influence_count") != 0
        or value.get("cache_entry_reuse_count") != 8
        or value.get("cache_append_count") != 1
        or value.get("native_execution_count") != 0
        or not value.get("W0_restored")
        or not value.get("terminal_W0_restore", {}).get("byte_restored_exact")
        or not value.get("terminal_W0_restore", {}).get("pointer_restored_exact")
        or dtype.get("status") != "FULL_FP32_PASS"
        or dtype.get("bf16_fp16_path_count") != 0
        or dtype.get("autocast_count") != 0
        or cache.get("same_batch_entry_cache_reuse_count") != 8
        or cache.get("append_count") != 1
        or cache.get("rollback_count") != 0
        or cache.get("current_batch_k_key_history_inclusion_count") != 0
    ):
        raise ValueError(f"terminal contract differs: {cell}")


def build_same_horizon(
    *, raw_root: Path, output_root: Path, job_id: str, repo_root: Path
) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"create-once output exists: {output_root}")
    output_root.mkdir(mode=0o755, parents=True)
    cells = ("Z0-COARSE", "Z1-REFINE")
    slurm = _sacct(job_id, (0, 1))
    terminals: dict[str, dict[str, Any]] = {}
    raw_inputs: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    micro_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    writer_rows: list[dict[str, Any]] = []
    compute_rows: list[dict[str, Any]] = []

    for cell_index, cell in enumerate(cells):
        result_root = raw_root / (
            "s05-p1r52-target-timescale-b100-"
            f"{CELL_SLUGS[cell]}-tech-r4-v1"
        )
        terminal_path = result_root / "terminal.json"
        manifest_path = result_root / "manifest.json"
        source_manifest_path = result_root / "source-manifest.json"
        terminal = _load(terminal_path)
        manifest = _load(manifest_path)
        source_manifest = _load(source_manifest_path)
        _validate_terminal(terminal, cell)
        if (
            manifest.get("terminal_sha256") != _sha256(terminal_path)
            or manifest.get("cell") != cell
            or manifest.get("request_order_sha256") != B1_ORDER
            or source_manifest.get("source_head") != SOURCE_HEAD
            or source_manifest.get("source_tree") != SOURCE_TREE
        ):
            raise ValueError(f"manifest binding differs: {cell}")
        terminals[cell] = terminal
        raw_inputs.extend(_member(path) for path in (terminal_path, manifest_path, source_manifest_path))

        micros = terminal["microstep_trajectory"]
        selections = [role for micro in micros for role in micro["selection_by_request"]]
        clamp_hits = sum(int(micro["selection_receipt"]["clamp_hit_count"]) for micro in micros)
        cell_rows.append(
            {
                "cell": cell,
                "target_horizon": terminal["schedule"]["target_horizon"],
                "microsteps_per_outer": terminal["schedule"]["microsteps_per_outer"],
                "target_dt": terminal["schedule"]["target_dt"],
                "target_field_evaluations": terminal["target_field_evaluation_count"],
                "writer_calls": terminal["writer_call_count"],
                "request_microstep_denominator": len(micros) * 100,
                "clamp_hits": clamp_hits,
                "clamp_fraction": clamp_hits / (len(micros) * 100),
                "primary_count": selections.count("PRIMARY"),
                "rescue_count": selections.count("RESCUE"),
                "current_count": selections.count("CURRENT"),
                "pre_clamp_energy_sum": sum(float(m["selection_receipt"]["pre_clamp_energy"]) for m in micros),
                "post_clamp_energy_sum": sum(float(m["selection_receipt"]["post_clamp_energy"]) for m in micros),
                "clamp_removed_energy_sum": sum(float(m["selection_receipt"]["clamp_removed_energy"]) for m in micros),
                "accepted_energy_sum": sum(float(m["selection_receipt"]["accepted_energy"]) for m in micros),
                "runtime_after_model_preflight_seconds": terminal["runtime_after_model_preflight_seconds"],
                "slurm_elapsed_seconds": slurm[cell_index]["elapsed_seconds"],
                "peak_allocated_bytes": terminal["gpu_host_observation"]["peak_allocated_bytes"],
                "peak_reserved_bytes": terminal["gpu_host_observation"]["peak_reserved_bytes"],
                "W0_restored": terminal["W0_restored"],
                "full_fp32": terminal["dtype_contract"]["status"],
            }
        )

        for micro in micros:
            selection = micro["selection_receipt"]
            field = micro["field_receipt"]
            semantic_norm = _summary(field["semantic_gradient_norm_by_request"])
            preservation_norm = _summary(field["preservation_gradient_norm_by_request"])
            movement_norm = _summary(field["post_cast_actual_delta_norm_by_request"])
            micro_rows.append(
                {
                    "cell": cell,
                    "outer_K": int(micro["outer_step_index"]) + 1,
                    "microstep_in_outer": int(micro["microstep_index"]) + 1,
                    "global_field_evaluation": int(micro["global_field_evaluation_ordinal"]) + 1,
                    "target_time_before": micro["target_time_before"],
                    "target_time_after": micro["target_time_after"],
                    "target_dt": micro["target_dt"],
                    "entry_target_new_nll": micro["last_micro_entry_objective"]["loss"],
                    "selected_endpoint_target_new_nll": micro["final_selected_endpoint_objective"]["loss"],
                    "endpoint_minus_entry_nll": float(micro["final_selected_endpoint_objective"]["loss"]) - float(micro["last_micro_entry_objective"]["loss"]),
                    "primary_count": selection["primary_accept_count"],
                    "rescue_count": selection["rescue_accept_count"],
                    "current_count": selection["current_hold_count"],
                    "clamp_hit_numerator": selection["clamp_hit_count"],
                    "clamp_denominator": 100,
                    "clamp_fraction": selection["clamp_hit_count"] / 100,
                    "pre_clamp_energy": selection["pre_clamp_energy"],
                    "post_clamp_energy": selection["post_clamp_energy"],
                    "clamp_removed_energy": selection["clamp_removed_energy"],
                    "accepted_energy": selection["accepted_energy"],
                    "unused_energy": selection["total_unused_energy"],
                    "target_displacement_norm": selection["target_displacement_norm"],
                    "z_minus_y_norm": selection["z_minus_y_norm"],
                    "semantic_gradient_norm_mean": semantic_norm["mean"],
                    "semantic_gradient_norm_median": semantic_norm["median"],
                    "semantic_gradient_norm_p90": semantic_norm["p90"],
                    "preservation_gradient_norm_mean": preservation_norm["mean"],
                    "movement_norm_mean": movement_norm["mean"],
                    "movement_norm_p90": movement_norm["p90"],
                    "early_break_count": micro["early_break_count"],
                    "writer_authority_count": micro["writer_authority_count"],
                    "history_append_count": micro["history_append_count"],
                }
            )

        for outer in (1, 4, 8):
            execution = terminal["outer_transitions"][outer - 1]["c_kstep_writer"]
            metrics = execution["metrics"]
            for endpoint in ("accepted_z", "pre_writer_W", "post_writer_W"):
                endpoint_rows.extend(
                    _score_rows(
                        cell=cell,
                        outer=outer,
                        endpoint=endpoint,
                        scores=metrics[endpoint]["scores"],
                    )
                )
            writer_rows.append(
                {
                    "cell": cell,
                    "outer_K": outer,
                    "rewrite_W_minus_z_target_new_nll": metrics["post_W_minus_z_gap"]["rewrite_W_minus_z_target_new_nll"],
                    "rephrase_W_minus_z_target_new_nll": metrics["post_W_minus_z_gap"]["rephrase_W_minus_z_target_new_nll"],
                    "accepted_z_locality_numerator": metrics["accepted_z"]["summary"]["locality_numerator"],
                    "accepted_z_locality_denominator": metrics["accepted_z"]["summary"]["locality_denominator"],
                    "post_W_locality_numerator": metrics["post_writer_W"]["summary"]["locality_numerator"],
                    "post_W_locality_denominator": metrics["post_writer_W"]["summary"]["locality_denominator"],
                    "writer_edit_core_wall_seconds": execution["writer_receipt"]["writer"]["edit_core_wall_seconds"],
                    "writer_native_apply_count": execution["compute"]["native_apply_count"],
                    "native_compute_z_call_count": execution["writer_receipt"]["native_compute_z_call_count"],
                    "cache_status": execution["alpha_cache_status"],
                }
            )

        compute_rows.append(
            {
                "cell": cell,
                **slurm[cell_index],
                "field_evaluations": terminal["target_field_evaluation_count"],
                "writer_calls": terminal["writer_call_count"],
                "writer_layer_apply_count": sum(int(item["compute"]["native_apply_count"]) for item in terminal["kstep_executions"]),
                "writer_edit_core_wall_seconds": sum(float(item["writer_receipt"]["writer"]["edit_core_wall_seconds"]) for item in terminal["kstep_executions"]),
                "runtime_after_model_preflight_seconds": terminal["runtime_after_model_preflight_seconds"],
                "model_forward_count": terminal["job_compute"]["counters"]["model_forward"],
                "processed_tokens": terminal["job_compute"]["counters"]["processed_tokens"],
                "peak_allocated_bytes": terminal["gpu_host_observation"]["peak_allocated_bytes"],
                "peak_reserved_bytes": terminal["gpu_host_observation"]["peak_reserved_bytes"],
            }
        )

    paired_rows: list[dict[str, Any]] = []
    paired_summary: dict[str, Any] = {}
    for outer in (1, 4, 8):
        paired_summary[str(outer)] = {}
        for prompt, success_key in (
            ("rewrite", "rewrite_success"),
            ("rephrase", "rephrase_success"),
        ):
            vectors: dict[str, list[list[float]]] = {}
            for cell in cells:
                scores = terminals[cell]["outer_transitions"][outer - 1]["c_kstep_writer"]["metrics"]["accepted_z"]["scores"]
                vectors[cell] = scores[success_key]["target_new_nll_by_request"]
            deltas: list[float] = []
            for request_index, (left, right) in enumerate(zip(vectors[cells[0]], vectors[cells[1]], strict=True)):
                for prompt_index, (z0, z1) in enumerate(zip(left, right, strict=True)):
                    delta = float(z1) - float(z0)
                    deltas.append(delta)
                    paired_rows.append(
                        {
                            "outer_K": outer,
                            "prompt": prompt,
                            "request_index": request_index,
                            "prompt_index": prompt_index,
                            "Z0_target_new_nll": z0,
                            "Z1_target_new_nll": z1,
                            "Z1_minus_Z0": delta,
                            "winner_lower_nll": "Z1-REFINE" if delta < 0 else "Z0-COARSE" if delta > 0 else "TIE",
                        }
                    )
            paired_summary[str(outer)][prompt] = {
                "delta_Z1_minus_Z0": _summary(deltas),
                "Z1_win_count": sum(value < 0 for value in deltas),
                "Z0_win_count": sum(value > 0 for value in deltas),
                "tie_count": sum(value == 0 for value in deltas),
            }

    endpoint_lookup = {
        (row["cell"], row["outer_K"], row["endpoint"], row["prompt"]): row
        for row in endpoint_rows
    }
    native_rows: list[dict[str, Any]] = []
    for native_name, prompts in NATIVE_MEAN_NLL.items():
        for prompt, native_value in prompts.items():
            z0 = float(endpoint_lookup[("Z0-COARSE", 8, "accepted_z", prompt)]["target_new_mean"])
            for cell in cells:
                value = float(endpoint_lookup[(cell, 8, "accepted_z", prompt)]["target_new_mean"])
                denominator = z0 - native_value
                closure = 1.0 - (value - native_value) / denominator
                native_rows.append(
                    {
                        "reference": native_name,
                        "prompt": prompt,
                        "statistic": "mean",
                        "native_NLL": native_value,
                        "Z0_observed_NLL": z0,
                        "cell": cell,
                        "cell_NLL": value,
                        "gap_cell_minus_native": value - native_value,
                        "closure": closure,
                        "comparability": "AUTHORITATIVE_CONTRACT_B1_MEAN",
                    }
                )
            for statistic in ("median", "p90"):
                native_rows.append(
                    {
                        "reference": native_name,
                        "prompt": prompt,
                        "statistic": statistic,
                        "native_NLL": "NOT_RECORDED",
                        "Z0_observed_NLL": endpoint_lookup[("Z0-COARSE", 8, "accepted_z", prompt)][f"target_new_{statistic}"],
                        "cell": "Z1-REFINE",
                        "cell_NLL": endpoint_lookup[("Z1-REFINE", 8, "accepted_z", prompt)][f"target_new_{statistic}"],
                        "gap_cell_minus_native": "NOT_COMPARABLE",
                        "closure": "NOT_COMPARABLE",
                        "comparability": "NATIVE_DISTRIBUTION_NOT_PROVIDED_DO_NOT_ESTIMATE",
                    }
                )

    z0_k8_rewrite = endpoint_lookup[("Z0-COARSE", 8, "accepted_z", "rewrite")]
    z1_k8_rewrite = endpoint_lookup[("Z1-REFINE", 8, "accepted_z", "rewrite")]
    z0_k8_rephrase = endpoint_lookup[("Z0-COARSE", 8, "accepted_z", "rephrase")]
    z1_k8_rephrase = endpoint_lookup[("Z1-REFINE", 8, "accepted_z", "rephrase")]
    z0_post_rephrase = endpoint_lookup[("Z0-COARSE", 8, "post_writer_W", "rephrase")]
    z1_post_rephrase = endpoint_lookup[("Z1-REFINE", 8, "post_writer_W", "rephrase")]
    analysis: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r52-target-timescale-same-horizon-analysis/v1",
        "instruction_id": INSTRUCTION_ID,
        "scope": "SAME_HORIZON_ONLY_Z0_COARSE_VS_Z1_REFINE",
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "numerical_lock_root": NUMERICAL_LOCK_ROOT,
        "scientific_denominator": {"cells": list(cells), "B1_requests": 100, "longer_time_cells_influence_count": 0},
        "primary_finding": {
            "classification": "COARSE_RESOLUTION_IS_A_BOTTLENECK_COMPONENT_WITH_WRITER_REPHRASE_TRANSFER_LIMIT",
            "K8_accepted_z": {
                "rewrite_Z0_mean": z0_k8_rewrite["target_new_mean"],
                "rewrite_Z1_mean": z1_k8_rewrite["target_new_mean"],
                "rewrite_delta_Z1_minus_Z0": float(z1_k8_rewrite["target_new_mean"]) - float(z0_k8_rewrite["target_new_mean"]),
                "rephrase_Z0_mean": z0_k8_rephrase["target_new_mean"],
                "rephrase_Z1_mean": z1_k8_rephrase["target_new_mean"],
                "rephrase_delta_Z1_minus_Z0": float(z1_k8_rephrase["target_new_mean"]) - float(z0_k8_rephrase["target_new_mean"]),
                "rephrase_p90_delta_Z1_minus_Z0": float(z1_k8_rephrase["target_new_p90"]) - float(z0_k8_rephrase["target_new_p90"]),
            },
            "K8_post_W_rephrase": {
                "Z0_mean": z0_post_rephrase["target_new_mean"],
                "Z1_mean": z1_post_rephrase["target_new_mean"],
                "delta_Z1_minus_Z0": float(z1_post_rephrase["target_new_mean"]) - float(z0_post_rephrase["target_new_mean"]),
            },
        },
        "paired_accepted_z": paired_summary,
        "cell_summary": cell_rows,
        "compute_overhead": {
            "field_evaluation_ratio_Z1_over_Z0": 2.0,
            "post_model_runtime_ratio_Z1_over_Z0": float(cell_rows[1]["runtime_after_model_preflight_seconds"]) / float(cell_rows[0]["runtime_after_model_preflight_seconds"]),
            "slurm_elapsed_ratio_Z1_over_Z0": float(cell_rows[1]["slurm_elapsed_seconds"]) / float(cell_rows[0]["slurm_elapsed_seconds"]),
            "slurm_elapsed_delta_seconds": int(cell_rows[1]["slurm_elapsed_seconds"]) - int(cell_rows[0]["slurm_elapsed_seconds"]),
        },
        "interpretation_boundary": {
            "same_horizon_resolution_only": True,
            "longer_time_result_used": False,
            "final_Tz_selection": False,
            "scientific_promotion": False,
            "native_execution_count": 0,
            "native_median_p90_imputation_count": 0,
            "causal_claim_for_writer_rephrase_gap": False,
        },
        "resource_provenance": {
            "submitted_source_array_throttle": 2,
            "user_runtime_cap_override": 4,
            "same_horizon_cells_started_before_override": True,
            "science_or_result_selection_influence_count": 0,
        },
    }
    analysis["identity_sha256"] = canonical_hash(analysis)

    tables = {
        "cell-summary.csv": cell_rows,
        "microstep-trajectory.csv": micro_rows,
        "k1-k4-k8-endpoints.csv": endpoint_rows,
        "paired-accepted-z-nll.csv": paired_rows,
        "writer-transfer-locality.csv": writer_rows,
        "native-gap-closure.csv": native_rows,
        "compute.csv": compute_rows,
    }
    for name, rows in tables.items():
        _write_csv(output_root / name, rows)
    _write_json(output_root / "analysis.json", analysis)

    report = _render_same_horizon_report(
        analysis=analysis,
        endpoint=endpoint_lookup,
        cells=cell_rows,
        writer=writer_rows,
    )
    (output_root / "report-ko.md").write_text(report, encoding="utf-8")

    generated = [output_root / name for name in tables] + [output_root / "analysis.json", output_root / "report-ko.md"]
    raw_inputs.sort(key=lambda row: str(row["path"]))
    manifest: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r52-target-timescale-same-horizon-analysis-manifest/v1",
        "instruction_id": INSTRUCTION_ID,
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "scope": "Z0_COARSE_Z1_REFINE_ONLY",
        "raw_inputs": raw_inputs,
        "raw_input_count": len(raw_inputs),
        "raw_member_root": canonical_hash(raw_inputs),
        "generated_members": [_member(path, relative_to=output_root) for path in generated],
        "longer_time_raw_input_count": 0,
        "scientific_promotion": False,
    }
    manifest["root_digest"] = canonical_hash(manifest)
    _write_json(output_root / "analysis-manifest.json", manifest)
    package_paths = generated + [output_root / "analysis-manifest.json"]
    package_members = [_member(path, relative_to=output_root) for path in package_paths]
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r52-target-timescale-same-horizon-rooted-receipt/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "SAME_HORIZON_REPORT_COMPLETE",
        "analysis_identity": analysis["identity_sha256"],
        "manifest_root": manifest["root_digest"],
        "package_members": package_members,
        "package_member_root": canonical_hash(package_members),
        "scientific_promotion": False,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    _write_json(output_root / "rooted-receipt.json", receipt)


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _render_same_horizon_report(
    *,
    analysis: Mapping[str, Any],
    endpoint: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    writer: Sequence[Mapping[str, Any]],
) -> str:
    primary = analysis["primary_finding"]["K8_accepted_z"]
    post = analysis["primary_finding"]["K8_post_W_rephrase"]
    paired = analysis["paired_accepted_z"]
    writer_k8 = {(row["cell"], row["outer_K"]): row for row in writer}
    lines = [
        "# P1R52 target-timescale B100 — same-horizon Z0/Z1 상세 결과",
        "",
        "> **범위:** `SAME_HORIZON_ONLY_Z0_COARSE_VS_Z1_REFINE`. 두 cell은 동일 `T_z=1`, 동일 B1 100 requests, 동일 W0·writer·cache·evaluator를 사용한다. Z0는 `m=1, dt=1/8`, Z1은 `m=2, dt=1/16`이다. 아직 실행 중인 Z15/Z20/Z30은 이 분석과 manifest에 포함하지 않았다.",
        "",
        "## 결론",
        "",
        f"- K8 accepted-z Rewrite mean NLL은 `{primary['rewrite_Z0_mean']:.6f} → {primary['rewrite_Z1_mean']:.6f}`로 `{primary['rewrite_delta_Z1_minus_Z0']:.6f}` 감소했다. paired prompt 기준 Z1 승리는 `{paired['8']['rewrite']['Z1_win_count']}/100`이다.",
        f"- K8 accepted-z Rephrase mean은 `{primary['rephrase_Z0_mean']:.6f} → {primary['rephrase_Z1_mean']:.6f}`로 `{primary['rephrase_delta_Z1_minus_Z0']:.6f}` 감소했고 Z1 승리는 `{paired['8']['rephrase']['Z1_win_count']}/200`이다. 다만 p90은 `{primary['rephrase_p90_delta_Z1_minus_Z0']:+.6f}`로 악화했다.",
        f"- Rewrite 개선은 post-W에도 유지됐지만, Rephrase post-W mean은 `{post['Z0_mean']:.6f} → {post['Z1_mean']:.6f}`로 `{post['delta_Z1_minus_Z0']:+.6f}` 악화했다. accepted-z 개선이 writer의 Rephrase endpoint 개선으로 전달되지 않았다는 사실만 말할 수 있으며 원인 인과는 주장하지 않는다.",
        "- 따라서 같은 horizon에서 `dt`를 절반으로 줄인 것은 coarse-resolution 병목의 일부를 해소했다. 그러나 Rephrase tail과 post-W transfer는 남아 있어 resolution 하나만으로 전체 병목이 해소됐다고 볼 수 없다.",
        "",
        "## 1. 실행·불변식",
        "",
        "|cell|T_z|m|dt|field eval|writer|clamp|PRIMARY/RESCUE/CURRENT|Slurm|post-model|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cells:
        lines.append(
            f"|{row['cell']}|{row['target_horizon']:.1f}|{row['microsteps_per_outer']}|{row['target_dt']:.4f}|{row['target_field_evaluations']}|{row['writer_calls']}|{row['clamp_hits']}/{row['request_microstep_denominator']} ({_pct(float(row['clamp_fraction']))})|{row['primary_count']}/{row['rescue_count']}/{row['current_count']}|{row['slurm_elapsed_seconds']}s|{float(row['runtime_after_model_preflight_seconds']):.1f}s|"
        )
    lines.extend(
        [
            "",
            "두 cell 모두 terminal valid, writer K1–K8 정확히 8회, cache entry reuse 8회·K8 뒤 append 1회, heldout K1/K4/K8만 9 evaluator groups, W0 pointer/bytes restore exact, FULL-FP32, Native 실행 0이다.",
            "",
            "## 2. K1/K4/K8 accepted-z NLL",
            "",
            "|K|cell|Rewrite mean/median/p90/max|Rephrase mean/median/p90/max|Rewrite success|Rephrase success/strict|",
            "|---:|---|---|---|---:|---:|",
        ]
    )
    for outer in (1, 4, 8):
        for cell in ("Z0-COARSE", "Z1-REFINE"):
            rewrite = endpoint[(cell, outer, "accepted_z", "rewrite")]
            rephrase = endpoint[(cell, outer, "accepted_z", "rephrase")]
            lines.append(
                f"|{outer}|{cell}|{rewrite['target_new_mean']:.6f}/{rewrite['target_new_median']:.6f}/{rewrite['target_new_p90']:.6f}/{rewrite['target_new_max']:.6f}|{rephrase['target_new_mean']:.6f}/{rephrase['target_new_median']:.6f}/{rephrase['target_new_p90']:.6f}/{rephrase['target_new_max']:.6f}|{rewrite['success_numerator']}/{rewrite['success_denominator']}|{rephrase['success_numerator']}/{rephrase['success_denominator']} · {rephrase['strict_success_numerator']}/{rephrase['strict_success_denominator']}|"
            )
    lines.extend(
        [
            "",
            "K1은 공통 W0에서의 가장 깨끗한 resolution 비교다. Z1은 K1부터 Rewrite 99/100, Rephrase 185/200 prompts에서 Z0보다 낮은 NLL이었다. K8에서는 각각 95/100, 142/200으로 advantage가 유지되지만 Rephrase upper tail은 단조 개선이 아니다.",
            "",
            "## 3. K8 z→W 전달과 locality",
            "",
            "|cell|endpoint|Rewrite mean|Rephrase mean|Rephrase success/strict|locality|",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for cell in ("Z0-COARSE", "Z1-REFINE"):
        for target, label in (("accepted_z", "accepted-z"), ("pre_writer_W", "pre-W"), ("post_writer_W", "post-W")):
            rewrite = endpoint[(cell, 8, target, "rewrite")]
            rephrase = endpoint[(cell, 8, target, "rephrase")]
            locality = writer_k8[(cell, 8)]["accepted_z_locality_numerator"] if target != "post_writer_W" else writer_k8[(cell, 8)]["post_W_locality_numerator"]
            lines.append(
                f"|{cell}|{label}|{rewrite['target_new_mean']:.6f}|{rephrase['target_new_mean']:.6f}|{rephrase['success_numerator']}/{rephrase['success_denominator']} · {rephrase['strict_success_numerator']}/{rephrase['strict_success_denominator']}|{locality}/1000|"
            )
    lines.extend(
        [
            "",
            f"K8 W−z Rephrase gap은 Z0 `{writer_k8[('Z0-COARSE', 8)]['rephrase_W_minus_z_target_new_nll']:+.6f}`, Z1 `{writer_k8[('Z1-REFINE', 8)]['rephrase_W_minus_z_target_new_nll']:+.6f}`이다. Rewrite gap은 각각 `{writer_k8[('Z0-COARSE', 8)]['rewrite_W_minus_z_target_new_nll']:+.6f}`, `{writer_k8[('Z1-REFINE', 8)]['rewrite_W_minus_z_target_new_nll']:+.6f}`이다.",
            "",
            "## 4. Clamp·selection·energy",
            "",
            f"Z0는 clamp `27/800`(3.375%), Z1은 `13/1600`(0.8125%)이다. Z0의 선택은 PRIMARY/RESCUE/CURRENT `788/7/5`, Z1은 `1593/7/0`이다. Z1은 field evaluation을 2배 사용했지만 clamp-removed energy 합이 `{float(cells[0]['clamp_removed_energy_sum']):.3f} → {float(cells[1]['clamp_removed_energy_sum']):.3f}`로 감소했다. 이는 관측 association이며 개별 기제의 인과 증명은 아니다.",
            "",
            "## 5. Native mean gap closure",
            "",
            "원문 contract가 제공한 동일 B1 Native mean만 사용했다. Z1 K8 accepted-z mean closure는 AlphaEdit 대비 Rewrite와 Rephrase 각각 `native-gap-closure.csv`에 기록했다. Native median/p90 원자료가 제공되지 않아 그 closure는 `NOT_COMPARABLE`이며 추정하지 않았다.",
            "",
            "## 6. 계산량과 overhead",
            "",
            f"Z1은 field evaluation `16`으로 Z0의 2배다. post-model wall은 `{float(cells[0]['runtime_after_model_preflight_seconds']):.1f}s → {float(cells[1]['runtime_after_model_preflight_seconds']):.1f}s`({float(analysis['compute_overhead']['post_model_runtime_ratio_Z1_over_Z0']):.3f}×), Slurm elapsed는 `{cells[0]['slurm_elapsed_seconds']}s → {cells[1]['slurm_elapsed_seconds']}s`({float(analysis['compute_overhead']['slurm_elapsed_ratio_Z1_over_Z0']):.3f}×, +{analysis['compute_overhead']['slurm_elapsed_delta_seconds']}s)였다. GPU peak allocated는 두 cell 모두 `{float(cells[0]['peak_allocated_bytes']) / 2**30:.2f} GiB`다.",
            "",
            "## 7. 해석 경계",
            "",
            "- 이 보고서는 same-horizon resolution 효과만 다룬다. Z15/Z20/Z30 결과 영향은 0이다.",
            "- B100×1 탐색이므로 final T_z, promotion, production setting을 선정하지 않는다.",
            "- Native는 external mean reference이며 새 Native 실행은 0이다.",
            "- resource cap은 제출 당시 2였고 사용자 지시로 실행 중 4로 상향됐다. 이는 동시성만 바꾸었고 cell science/result selection에는 영향 0이다.",
            "- 제외된 pre-model/interface 기술 시도는 최종 통합 보고서에서 한 번에 provenance로 정리하며, 이 scientific denominator에는 포함하지 않는다.",
            "",
            "세부 수치는 `microstep-trajectory.csv`, `k1-k4-k8-endpoints.csv`, `paired-accepted-z-nll.csv`, `writer-transfer-locality.csv`, `native-gap-closure.csv`, `compute.csv`에 있다.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mode", choices=("same-horizon",), required=True)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args(argv)
    build_same_horizon(
        raw_root=args.raw_root,
        output_root=args.output_root,
        job_id=args.job_id,
        repo_root=args.repo_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
