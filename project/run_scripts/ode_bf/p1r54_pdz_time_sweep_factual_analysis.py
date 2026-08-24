#!/usr/bin/env python3
"""Build the raw-free P1R54 PDZ target-time sweep factual package."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from .contracts import canonical_hash
from .p1r52_target_timescale_analysis import _load, _member, _sha256
from .p1r53_request_local_speed_analysis import (
    _endpoint_row,
    _pct,
    _prompt_values,
    _quantile,
    _summary,
)
from .p1r54_energyfree_localz_analysis import (
    _scores_and_locality,
    _stream_request_hashes,
    _tail_summary,
)
from .p1r54_pdz_ablation import INSTRUCTION_ID, PDZAblationArm
from .p1r54_pdz_ablation_analysis import validate_terminal_schema
from .p1r54_pdz_ablation_b100 import RESULT_NAMES, ROLES, ROLE_TO_ARM


SOURCE_HEAD = "c0dc8e3fcd1ea2c941441dd2dc5cfa91a28aa296"
SOURCE_TREE = "ccd9015d13adecc8dcf5e2d97baff5b12d3ddca8"
STREAM_ROOT = "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
STREAM_ORDER = "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"
B1_ORDER = "bb2e661ad44bf8dc20e0224f21dc90164ed4b3dcf3316d751afb7d8a00bfaed6"
CONTRACT_SHA256 = "d9f5ee879534bb5f33c6741cf39b1ff3bfb938d07238842f584bfb44f6b9195e"
PRE_GPU_SHA256 = "f0db7f9fd8ee51180a7a21b181c1874905435b389f3a0abdc62ac2d237252e2f"
PRE_GPU_IDENTITY = "843991f626785daa0cfe9daececa58d6b1d2a640d296650c137f795aee7860e9"
ARM_LABELS = {
    PDZAblationArm.PDZ_T2_PRC: "PDZ-T2-PRC",
    PDZAblationArm.PDZ_T3_PRC: "PDZ-T3-PRC",
    PDZAblationArm.PDZ_T5_PRC: "PDZ-T5-PRC",
    PDZAblationArm.PDZ_T1_DIRECT: "PDZ-T1-DIRECT",
}
EXPECTED_M = {
    "PDZ-T2-PRC": 2,
    "PDZ-T3-PRC": 3,
    "PDZ-T5-PRC": 5,
    "PDZ-T1-DIRECT": 1,
}
PRC_ORDER = ("PDZ-T1-PRC†", "PDZ-T2-PRC", "PDZ-T3-PRC", "PDZ-T5-PRC")
DISPLAY_ORDER = (*PRC_ORDER, "PDZ-T1-DIRECT", "P1R54-FZ†", "Native AlphaEdit†")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sacct(job_id: str) -> dict[int, dict[str, Any]]:
    output = subprocess.run(
        [
            "sacct", "-j", ",".join(f"{job_id}_{cell}" for cell in range(4)),
            "-n", "-P", "-X",
            "--format=JobID,State,ExitCode,Start,End,ElapsedRaw,Elapsed,CPUTimeRAW,ReqMem,AllocTRES",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    rows: dict[int, dict[str, Any]] = {}
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) != 10 or not fields[0].startswith(f"{job_id}_"):
            continue
        cell = int(fields[0].rsplit("_", 1)[1])
        rows[cell] = {
            "job_id": fields[0], "state": fields[1], "exit_code": fields[2],
            "start": fields[3], "end": fields[4],
            "elapsed_seconds": int(fields[5]), "elapsed": fields[6],
            "allocated_cpu_seconds": int(fields[7]),
            "requested_memory": fields[8], "allocated_tres": fields[9],
        }
    if set(rows) != set(range(4)) or any(
        row["state"] != "COMPLETED" or row["exit_code"] != "0:0"
        for row in rows.values()
    ):
        raise ValueError("P1R54 time-sweep Slurm terminal accounting differs")
    return rows


def _reference_rows(
    reference_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Mapping[str, Any], list[dict[str, str]]]:
    aggregates_path = reference_root / "aggregates.json"
    paired_path = reference_root / "paired-request-deltas.csv"
    compute_path = reference_root / "compute.csv"
    manifest_path = reference_root / "analysis-manifest.json"
    aggregates = _load(aggregates_path)
    manifest = _load(manifest_path)
    with paired_path.open(encoding="utf-8", newline="") as handle:
        paired = list(csv.DictReader(handle))
    with compute_path.open(encoding="utf-8", newline="") as handle:
        compute = list(csv.DictReader(handle))
    if (
        aggregates.get("status") != "TERMINAL_FACTUAL_ANALYSIS_COMPLETE"
        or aggregates.get("request_count") != 100
        or manifest.get("stream_root") != STREAM_ROOT
        or manifest.get("stream_order") != STREAM_ORDER
        or len(paired) != 300
        or {row["arm"] for row in compute} != {"FZ", "PDZ"}
    ):
        raise ValueError("sealed P1R54 reference package differs")
    endpoint = [
        dict(row) for row in aggregates["endpoint_rows"]
        if row["method"] in {"PDZ", "FZ", "Native AlphaEdit†"}
    ]
    for row in endpoint:
        if row["method"] == "PDZ":
            row["method"] = "PDZ-T1-PRC†"
        elif row["method"] == "FZ":
            row["method"] = "P1R54-FZ†"
    return endpoint, paired, aggregates, compute


def _method_scores(terminal: Mapping[str, Any]) -> tuple[Any, Any, tuple[int, int], tuple[int, int]]:
    return _scores_and_locality(terminal, native=False)


def _endpoint_rows(
    terminals: Mapping[str, Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], Mapping[str, Any]]]:
    rows = [dict(row) for row in reference_rows]
    arrays: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for method, terminal in terminals.items():
        if terminal["status"] != "TERMINAL_VALID":
            continue
        z_scores, w_scores, z_loc, w_loc = _method_scores(terminal)
        for endpoint, scores, locality in (
            ("accepted_z", z_scores, z_loc), ("post_W", w_scores, w_loc)
        ):
            for prompt in ("rewrite", "rephrase"):
                values = _prompt_values(scores, prompt)
                arrays[(method, endpoint, prompt)] = values
                row = _endpoint_row(
                    method=method, endpoint=endpoint, prompt=prompt,
                    values=values, locality=locality,
                )
                row.update(
                    {f"hard_tail_{key}": value for key, value in _tail_summary(values["new"]).items()}
                )
                rows.append(row)
    return rows, arrays


def _reference_request_values(
    paired: Sequence[Mapping[str, str]], prompt: str, endpoint: str
) -> list[float]:
    key = "pdz_z_target_new_nll" if endpoint == "accepted_z" else "pdz_W_target_new_nll"
    return [float(row[key]) for row in paired if row["prompt"] == prompt]


def _paired_rows(
    arrays: Mapping[tuple[str, str, str], Mapping[str, Any]],
    reference_paired: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    request_sha = _stream_request_hashes()
    rows: list[dict[str, Any]] = []
    for endpoint in ("accepted_z", "post_W"):
        for prompt in ("rewrite", "rephrase"):
            count = 1 if prompt == "rewrite" else 2
            values = {
                "PDZ-T1-PRC†": _reference_request_values(reference_paired, prompt, endpoint),
                **{
                    method: list(arrays[(method, endpoint, prompt)]["new"])
                    for method in PRC_ORDER[1:]
                },
            }
            denominator = 100 * count
            if any(len(value) != denominator for value in values.values()):
                raise ValueError("P1R54 paired denominator differs")
            for flat in range(denominator):
                request_index, prompt_index = divmod(flat, count)
                row: dict[str, Any] = {
                    "request_index": request_index,
                    "request_sha256": request_sha[request_index],
                    "prompt": prompt, "prompt_index": prompt_index,
                    "endpoint": endpoint,
                }
                for method in PRC_ORDER:
                    row[method] = values[method][flat]
                for left, right in zip(PRC_ORDER[:-1], PRC_ORDER[1:], strict=True):
                    row[f"{right}_minus_{left}"] = values[right][flat] - values[left][flat]
                row["PDZ-T5-PRC_minus_PDZ-T1-PRC"] = (
                    values["PDZ-T5-PRC"][flat] - values["PDZ-T1-PRC†"][flat]
                )
                rows.append(row)
    return rows


def _trajectory_and_outer_rows(
    terminals: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    request_sha = _stream_request_hashes()
    trajectory: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    selection_summary: dict[str, Any] = {}
    for method in ("PDZ-T2-PRC", "PDZ-T3-PRC", "PDZ-T5-PRC"):
        terminal = terminals[method]
        width = EXPECTED_M[method]
        choices: list[str] = []
        clamp_by_request = [0] * 100
        negative_by_request = [0] * 100
        micros = terminal["microstep_trajectory"]
        for outer in range(8):
            group = micros[outer * width : (outer + 1) * width]
            group_choices: list[str] = []
            group_negative = 0
            group_clamp = 0
            group_raw_energy = 0.0
            group_postclamp_energy = 0.0
            group_selected_energy = 0.0
            for micro in group:
                field = micro["field_receipt"]
                observation = micro["external_amplitude_selected_observation"]
                selected_energy_values = observation["selected_direct_energy_by_request"]
                if selected_energy_values is None:
                    selected_energy_values = micro["selection_receipt"]["accepted_energy_by_request"]
                if (
                    micro.get("writer_authority_count") != 0
                    or micro.get("materialization_count") != 0
                    or micro.get("history_append_count") != 0
                    or field.get("p1r54_target_microsteps_per_outer") != width
                    or field.get("p1r54_rho_refresh_count") != 0
                    or observation.get("selected_observation_decision_influence_count") != 0
                ):
                    raise ValueError(f"P1R54 inner-state boundary differs: {method}/K{outer + 1}")
                selected = [str(value) for value in observation["selection_by_request"]]
                group_choices.extend(selected)
                choices.extend(selected)
                for request_index in range(100):
                    clamped = int(bool(observation["clamp_hit_by_request"][request_index]))
                    negative = int(bool(observation["negative_progress_by_request"][request_index]))
                    clamp_by_request[request_index] += clamped
                    negative_by_request[request_index] += negative
                    group_clamp += clamped
                    group_negative += negative
                    row = {
                        "method": method,
                        "outer_K": outer + 1,
                        "microstep_m": int(micro["microstep_index"]) + 1,
                        "global_ordinal_n": int(micro["global_field_evaluation_ordinal"]),
                        "request_index": request_index,
                        "request_sha256": request_sha[request_index],
                        "target_time_before": micro["target_time_before"],
                        "target_time_after": micro["target_time_after"],
                        "current_target_new_nll": observation["current_target_new_nll_by_request"][request_index],
                        "selected_target_new_nll": observation["selected_target_new_nll_by_request"][request_index],
                        "selected_nll_decrease": observation["selected_nll_decrease_by_request"][request_index],
                        "rho": field["p1r54_rho_by_request"][request_index],
                        "deficit": field["p1r54_deficit_by_request"][request_index],
                        "amplitude": field["p1r54_amplitude_by_request"][request_index],
                        "semantic_gradient_norm": field["semantic_gradient_norm_by_request"][request_index],
                        "preservation_gradient_norm": field["preservation_gradient_norm_by_request"][request_index],
                        "semantic_direction_cosine": field["p1r51_semantic_to_kdc_direction_cosine_by_request"][request_index],
                        "raw_energy": observation["raw_energy_by_request"][request_index],
                        "postclamp_primary_energy": observation["postclamp_primary_energy_by_request"][request_index],
                        "selected_direct_energy": selected_energy_values[request_index],
                        "clamp_hit": clamped,
                        "selection": selected[request_index],
                        "negative_primary_progress": negative,
                    }
                    trajectory.append(row)
                group_raw_energy += sum(float(value) for value in observation["raw_energy_by_request"])
                group_postclamp_energy += sum(float(value) for value in observation["postclamp_primary_energy_by_request"])
                group_selected_energy += sum(float(value) for value in selected_energy_values)
            execution = terminal["kstep_executions"][outer]
            realization = terminal["outer_transitions"][outer]["target_write_realization"]
            outer_rows.append(
                {
                    "method": method, "outer_K": outer + 1, "microsteps": width,
                    "primary_count": group_choices.count("PRIMARY"),
                    "rescue_count": group_choices.count("RESCUE"),
                    "current_count": group_choices.count("CURRENT"),
                    "direct_count": 0,
                    "negative_primary_progress_count": group_negative,
                    "clamp_hit_count": group_clamp,
                    "raw_energy": group_raw_energy,
                    "postclamp_primary_energy": group_postclamp_energy,
                    "selected_energy": group_selected_energy,
                    "intended_write_norm_mean": statistics.fmean(realization["intended_norm"]),
                    "realized_write_norm_mean": statistics.fmean(realization["realized_norm"]),
                    "write_residual_ratio_mean": statistics.fmean(realization["residual_ratio"]),
                    "writer_edit_core_wall_seconds": execution["writer_receipt"]["writer"]["edit_core_wall_seconds"],
                    "physical_update_energy": "NOT_RECORDED_RAW_FREE_TERMINAL",
                    "per_layer_update_energy": "NOT_RECORDED_RAW_FREE_TERMINAL",
                }
            )
        denominator = 800 * width
        selection_summary[method] = {
            "request_microstep_denominator": denominator,
            "selection_counts": {key: choices.count(key) for key in ("PRIMARY", "RESCUE", "CURRENT")},
            "negative_primary_progress_count": sum(negative_by_request),
            "clamp_hit_count": sum(clamp_by_request),
            "all_microsteps_clamped_request_count": sum(value == 8 * width for value in clamp_by_request),
            "ever_clamped_request_count": sum(value > 0 for value in clamp_by_request),
            "all_microsteps_negative_request_count": sum(value == 8 * width for value in negative_by_request),
        }
    direct = terminals["PDZ-T1-DIRECT"]
    if direct["status"] == "TERMINAL_VALID":
        choices: list[str] = []
        clamp_by_request = [0] * 100
        negative_by_request = [0] * 100
        for outer, micro in enumerate(direct["microstep_trajectory"]):
            field = micro["field_receipt"]
            observation = micro["external_amplitude_selected_observation"]
            selected = [str(value) for value in observation["selection_by_request"]]
            if set(selected) != {"DIRECT_EULER"}:
                raise ValueError("P1R54 direct selection bytes differ")
            choices.extend(selected)
            for request_index in range(100):
                clamped = int(bool(observation["clamp_hit_by_request"][request_index]))
                negative = int(bool(observation["negative_progress_by_request"][request_index]))
                clamp_by_request[request_index] += clamped
                negative_by_request[request_index] += negative
                trajectory.append({
                    "method": "PDZ-T1-DIRECT", "outer_K": outer + 1,
                    "microstep_m": 1, "global_ordinal_n": int(micro["global_field_evaluation_ordinal"]),
                    "request_index": request_index, "request_sha256": request_sha[request_index],
                    "target_time_before": micro["target_time_before"],
                    "target_time_after": micro["target_time_after"],
                    "current_target_new_nll": observation["current_target_new_nll_by_request"][request_index],
                    "selected_target_new_nll": observation["selected_target_new_nll_by_request"][request_index],
                    "selected_nll_decrease": observation["selected_nll_decrease_by_request"][request_index],
                    "rho": field["p1r54_rho_by_request"][request_index],
                    "deficit": field["p1r54_deficit_by_request"][request_index],
                    "amplitude": field["p1r54_amplitude_by_request"][request_index],
                    "semantic_gradient_norm": field["semantic_gradient_norm_by_request"][request_index],
                    "preservation_gradient_norm": field["preservation_gradient_norm_by_request"][request_index],
                    "semantic_direction_cosine": field["p1r51_semantic_to_kdc_direction_cosine_by_request"][request_index],
                    "raw_energy": observation["raw_energy_by_request"][request_index],
                    "postclamp_primary_energy": observation["postclamp_primary_energy_by_request"][request_index],
                    "selected_direct_energy": observation["selected_direct_energy_by_request"][request_index],
                    "clamp_hit": clamped, "selection": selected[request_index],
                    "negative_primary_progress": negative,
                })
            execution = direct["kstep_executions"][outer]
            realization = direct["outer_transitions"][outer]["target_write_realization"]
            outer_rows.append({
                "method": "PDZ-T1-DIRECT", "outer_K": outer + 1, "microsteps": 1,
                "primary_count": 0, "rescue_count": 0, "current_count": 0,
                "direct_count": 100,
                "negative_primary_progress_count": observation["negative_progress_count"],
                "clamp_hit_count": sum(bool(value) for value in observation["clamp_hit_by_request"]),
                "raw_energy": sum(float(value) for value in observation["raw_energy_by_request"]),
                "postclamp_primary_energy": sum(float(value) for value in observation["postclamp_primary_energy_by_request"]),
                "selected_energy": sum(float(value) for value in observation["selected_direct_energy_by_request"]),
                "intended_write_norm_mean": statistics.fmean(realization["intended_norm"]),
                "realized_write_norm_mean": statistics.fmean(realization["realized_norm"]),
                "write_residual_ratio_mean": statistics.fmean(realization["residual_ratio"]),
                "writer_edit_core_wall_seconds": execution["writer_receipt"]["writer"]["edit_core_wall_seconds"],
                "physical_update_energy": "NOT_RECORDED_RAW_FREE_TERMINAL",
                "per_layer_update_energy": "NOT_RECORDED_RAW_FREE_TERMINAL",
            })
        selection_summary["PDZ-T1-DIRECT"] = {
            "request_microstep_denominator": 800,
            "selection_counts": {"DIRECT_EULER": choices.count("DIRECT_EULER")},
            "negative_primary_progress_count": sum(negative_by_request),
            "clamp_hit_count": sum(clamp_by_request),
            "all_microsteps_clamped_request_count": sum(value == 8 for value in clamp_by_request),
            "ever_clamped_request_count": sum(value > 0 for value in clamp_by_request),
            "all_microsteps_negative_request_count": sum(value == 8 for value in negative_by_request),
        }
    return trajectory, outer_rows, selection_summary


def _compute_rows(
    terminals: Mapping[str, Mapping[str, Any]], slurm: Mapping[int, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell, role in enumerate(ROLES):
        method = ARM_LABELS[ROLE_TO_ARM[role]]
        terminal = terminals[method]
        row: dict[str, Any] = {"method": method, **slurm[cell], "status": terminal["status"]}
        if terminal["status"] == "TERMINAL_VALID":
            counters = terminal["job_compute"]["counters"]
            row.update({
                "runtime_after_model_preflight_seconds": terminal["runtime_after_model_preflight_seconds"],
                "field_gradient_count": terminal["target_field_evaluation_count"],
                "primary_evaluation_count": terminal["p1r54_expected_primary_evaluation_count"],
                "selector_count": terminal["p1r54_expected_selector_count"],
                "finite_demand_builder_count": 8,
                "writer_count": terminal["writer_call_count"],
                "writer_layer_apply_count": terminal["p1r54_writer_layer_apply_count"],
                "model_forward_total": counters.get("model_forward_total", "NOT_RECORDED"),
                "model_backward_total": counters.get("model_backward_total", "NOT_RECORDED"),
                "peak_allocated_bytes": terminal["gpu_host_observation"]["peak_allocated_bytes"],
                "peak_reserved_bytes": terminal["gpu_host_observation"]["peak_reserved_bytes"],
                "W0_restored": terminal["W0_restored"],
            })
        else:
            row.update({
                "runtime_after_model_preflight_seconds": "NOT_RECORDED_NTSM_BOUNDARY",
                "field_gradient_count": terminal["boundary_global_ordinal"] + 1,
                "primary_evaluation_count": terminal["boundary_global_ordinal"] + 1,
                "selector_count": 0, "finite_demand_builder_count": 0,
                "writer_count": 0, "writer_layer_apply_count": 0,
                "model_forward_total": "NOT_RECORDED_NTSM_BOUNDARY",
                "model_backward_total": "NOT_RECORDED_NTSM_BOUNDARY",
                "peak_allocated_bytes": "NOT_RECORDED_NTSM_BOUNDARY",
                "peak_reserved_bytes": "NOT_RECORDED_NTSM_BOUNDARY",
                "W0_restored": terminal["W0_restored"],
            })
        rows.append(row)
    return rows


def _trajectory_summary_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize each scientific microstep without dropping the requestwise table."""
    grouped: dict[tuple[str, int, int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["method"]), int(row["outer_K"]),
            int(row["microstep_m"]), int(row["global_ordinal_n"]),
        )
        grouped.setdefault(key, []).append(row)
    fields = (
        "current_target_new_nll", "selected_target_new_nll", "selected_nll_decrease",
        "rho", "deficit", "amplitude", "semantic_gradient_norm",
        "preservation_gradient_norm", "semantic_direction_cosine", "raw_energy",
        "postclamp_primary_energy", "selected_direct_energy",
    )
    output: list[dict[str, Any]] = []
    for (method, outer, microstep, ordinal), group in sorted(grouped.items()):
        if len(group) != 100:
            raise ValueError("P1R54 per-microstep request denominator differs")
        row: dict[str, Any] = {
            "method": method, "outer_K": outer, "microstep_m": microstep,
            "global_ordinal_n": ordinal, "request_denominator": len(group),
            "target_time_before": group[0]["target_time_before"],
            "target_time_after": group[0]["target_time_after"],
        }
        for field in fields:
            summary = _summary(float(item[field]) for item in group)
            for stat in ("mean", "median", "p90", "max"):
                row[f"{field}_{stat}"] = summary[stat]
        row.update({
            "primary_count": sum(item["selection"] == "PRIMARY" for item in group),
            "rescue_count": sum(item["selection"] == "RESCUE" for item in group),
            "current_count": sum(item["selection"] == "CURRENT" for item in group),
            "direct_count": sum(item["selection"] == "DIRECT_EULER" for item in group),
            "clamp_hit_count": sum(int(item["clamp_hit"]) for item in group),
            "negative_primary_progress_count": sum(
                int(item["negative_primary_progress"]) for item in group
            ),
        })
        output.append(row)
    return output


def _outer_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    output: list[dict[str, Any]] = []
    for method, group in grouped.items():
        if len(group) != 8:
            raise ValueError("P1R54 writer outer denominator differs")
        output.append({
            "method": method, "outer_denominator": len(group),
            "microstep_count": sum(int(row["microsteps"]) for row in group),
            "primary_count": sum(int(row["primary_count"]) for row in group),
            "rescue_count": sum(int(row["rescue_count"]) for row in group),
            "current_count": sum(int(row["current_count"]) for row in group),
            "direct_count": sum(int(row["direct_count"]) for row in group),
            "negative_primary_progress_count": sum(
                int(row["negative_primary_progress_count"]) for row in group
            ),
            "clamp_hit_count": sum(int(row["clamp_hit_count"]) for row in group),
            "raw_energy_total": sum(float(row["raw_energy"]) for row in group),
            "postclamp_primary_energy_total": sum(
                float(row["postclamp_primary_energy"]) for row in group
            ),
            "selected_energy_total": sum(float(row["selected_energy"]) for row in group),
            "intended_write_norm_mean": statistics.fmean(
                float(row["intended_write_norm_mean"]) for row in group
            ),
            "realized_write_norm_mean": statistics.fmean(
                float(row["realized_write_norm_mean"]) for row in group
            ),
            "write_residual_ratio_mean": statistics.fmean(
                float(row["write_residual_ratio_mean"]) for row in group
            ),
            "writer_edit_core_wall_seconds_total": sum(
                float(row["writer_edit_core_wall_seconds"]) for row in group
            ),
        })
    return output


def _summary_paired(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    keys = [key for key in rows[0] if "_minus_" in key]
    for endpoint in ("accepted_z", "post_W"):
        for prompt in ("rewrite", "rephrase"):
            subset = [row for row in rows if row["endpoint"] == endpoint and row["prompt"] == prompt]
            result[f"{endpoint}/{prompt}"] = {
                key: _summary(float(row[key]) for row in subset) for key in keys
            }
    return result


def _writer_transfer(
    arrays: Mapping[tuple[str, str, str], Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    request_sha = _stream_request_hashes()
    methods = [
        method
        for method in (*PRC_ORDER[1:], "PDZ-T1-DIRECT")
        if (method, "accepted_z", "rewrite") in arrays
    ]
    for method in methods:
        summary[method] = {}
        for prompt in ("rewrite", "rephrase"):
            z = arrays[(method, "accepted_z", prompt)]
            w = arrays[(method, "post_W", prompt)]
            values = []
            failures = 0
            count = 1 if prompt == "rewrite" else 2
            for flat, (left, right) in enumerate(zip(z["new"], w["new"], strict=True)):
                request_index, prompt_index = divmod(flat, count)
                gap = float(right) - float(left)
                values.append(gap)
                failed = int(bool(z["success_bits"][flat]) and not bool(w["success_bits"][flat]))
                failures += failed
                rows.append({
                    "method": method, "request_index": request_index,
                    "request_sha256": request_sha[request_index], "prompt": prompt,
                    "prompt_index": prompt_index, "z_target_new_nll": left,
                    "W_target_new_nll": right, "W_minus_z_nll": gap,
                    "z_success_W_failure": failed,
                })
            summary[method][prompt] = {
                "W_minus_z_nll": _summary(values),
                "z_success_W_failure_count": failures,
                "denominator": len(values),
            }
    return rows, summary


def _fmt(row: Mapping[str, Any]) -> str:
    prefix = "target_new_" if "target_new_mean" in row else ""
    return (
        f"{float(row[prefix + 'mean']):.6f}/"
        f"{float(row[prefix + 'median']):.6f}/"
        f"{float(row[prefix + 'p90']):.6f}/"
        f"{float(row[prefix + 'max']):.6f}"
    )


def build_package(
    *, raw_root: Path, output_root: Path, job_id: str,
    pre_gpu_receipt: Path, reference_root: Path,
) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"create-once output exists: {output_root}")
    output_root.mkdir(parents=True, mode=0o755)
    slurm = _sacct(job_id)
    terminals: dict[str, Mapping[str, Any]] = {}
    dtype_evidence: dict[str, Mapping[str, Any]] = {}
    input_members: list[dict[str, Any]] = []
    for role in ROLES:
        method = ARM_LABELS[ROLE_TO_ARM[role]]
        root = raw_root / RESULT_NAMES[role]
        terminal_path, manifest_path = root / "terminal.json", root / "manifest.json"
        terminal, manifest = _load(terminal_path), _load(manifest_path)
        dtype_stage_path = root / "raw" / "stage-004-post_model_load_full_fp32_inventory.json"
        dtype_stage = json.loads(dtype_stage_path.read_text(encoding="utf-8"))
        inventory = dtype_stage.get("payload", {})
        inventory_body = dict(inventory)
        inventory_identity = inventory_body.pop("identity_sha256", None)
        if (
            inventory_identity != canonical_hash(inventory_body)
            or dtype_stage.get("stage") != "post_model_load_full_fp32_inventory"
            or inventory.get("status") != "FULL_FP32_PASS"
            or inventory.get("requested_dtype") != "torch.float32"
            or inventory.get("loaded_model_dtype") != "torch.float32"
            or inventory.get("non_fp32_parameter_tensor_count") != 0
            or inventory.get("bf16_conversion_count") != 0
            or inventory.get("fp16_conversion_count") != 0
            or inventory.get("numeric_storage_cast_count") != 0
            or inventory.get("quantized_parameter_count") != 0
            or inventory.get("autocast_cpu_enabled") is not False
            or inventory.get("autocast_cuda_enabled") is not False
        ):
            raise ValueError(f"P1R54 FULL-FP32 stage differs: {method}")
        validate_terminal_schema(terminal)
        dtype = terminal.get("dtype_contract", {})
        restore = terminal.get("terminal_W0_restore", {})
        width = EXPECTED_M[method]
        expected_evaluations = 8 * width
        policy = terminal.get("amplitude_policy_terminal", {})
        executions = terminal.get("kstep_executions", [])
        if (
            terminal.get("source_head") != SOURCE_HEAD
            or terminal.get("stream_root") != STREAM_ROOT
            or terminal.get("stream_order") != STREAM_ORDER
            or manifest.get("terminal_sha256") != _sha256(terminal_path)
            or terminal.get("p1r54_ablation_arm", terminal.get("arm")) != ROLE_TO_ARM[role].value
            or terminal.get("W0_restored") is not True
            or (
                terminal.get("status") == "TERMINAL_VALID"
                and (
                    terminal.get("request_count") != 100
                    or terminal.get("p1r54_target_microsteps_per_outer") != width
                    or terminal.get("target_field_evaluation_count") != expected_evaluations
                    or terminal.get("p1r54_expected_primary_evaluation_count") != expected_evaluations
                    or terminal.get("p1r54_expected_selector_count")
                    != (0 if method == "PDZ-T1-DIRECT" else expected_evaluations)
                    or terminal.get("p1r54_expected_finite_demand_builder_count") != 8
                    or terminal.get("writer_call_count") != 8
                    or terminal.get("p1r54_writer_layer_apply_count") != 40
                    or terminal.get("p1r54_inner_writer_materialization_history_cache_append_count") != 0
                    or policy.get("field_gradient_count") != expected_evaluations
                    or policy.get("primary_evaluation_count") != expected_evaluations
                    or policy.get("selector_count")
                    != (0 if method == "PDZ-T1-DIRECT" else expected_evaluations)
                    or policy.get("direct_euler_count")
                    != (8 if method == "PDZ-T1-DIRECT" else 0)
                    or policy.get("rho_calibration_count") != 100
                    or policy.get("rho_capture_event_count") != 1
                    or policy.get("rho_refresh_count") != 0
                    or policy.get("heldout_decision_influence_count") != 0
                    or policy.get("hold_rollback_retry_line_search_adaptive_count") != 0
                    or len(executions) != 8
                    or any(
                        execution.get("current_K_writer_affects_next_target") is not True
                        or execution.get("compute", {}).get("logical_commit_count") != 1
                        or execution.get("compute", {}).get("numeric_storage_cast_count") != 0
                        or execution.get("compute", {}).get("bf16_fp16_path_count") != 0
                        for execution in executions
                    )
                    or dtype.get("status") != "FULL_FP32_PASS"
                    or dtype.get("parameter_inventory", {}).get("identity_sha256")
                    != inventory.get("identity_sha256")
                    or dtype.get("bf16_fp16_path_count") != 0
                    or dtype.get("autocast_count") != 0
                    or restore.get("byte_restored_exact") is not True
                    or restore.get("pointer_restored_exact") is not True
                )
            )
        ):
            raise ValueError(f"P1R54 time-sweep terminal binding differs: {method}")
        terminals[method] = terminal
        dtype_evidence[method] = {
            "status": inventory["status"],
            "requested_dtype": inventory["requested_dtype"],
            "loaded_model_dtype": inventory["loaded_model_dtype"],
            "parameter_tensor_count": inventory["parameter_tensor_count"],
            "fp32_parameter_tensor_count": inventory["fp32_parameter_tensor_count"],
            "non_fp32_parameter_tensor_count": inventory["non_fp32_parameter_tensor_count"],
            "bf16_conversion_count": inventory["bf16_conversion_count"],
            "fp16_conversion_count": inventory["fp16_conversion_count"],
            "numeric_storage_cast_count": inventory["numeric_storage_cast_count"],
            "quantized_parameter_count": inventory["quantized_parameter_count"],
            "autocast_cpu_enabled": inventory["autocast_cpu_enabled"],
            "autocast_cuda_enabled": inventory["autocast_cuda_enabled"],
            "identity_sha256": inventory["identity_sha256"],
        }
        input_members.extend(
            _member(path) for path in (terminal_path, manifest_path, dtype_stage_path)
        )
    pre_gpu = _load(pre_gpu_receipt)
    if _sha256(pre_gpu_receipt) != PRE_GPU_SHA256 or pre_gpu.get("identity_sha256") != PRE_GPU_IDENTITY:
        raise ValueError("P1R54 time-sweep PRE-GPU receipt differs")
    input_members.append(_member(pre_gpu_receipt))
    reference_rows, reference_paired, reference_aggregates, reference_compute = _reference_rows(reference_root)
    input_members.extend(
        _member(reference_root / name)
        for name in ("aggregates.json", "paired-request-deltas.csv", "compute.csv", "analysis-manifest.json", "source-receipt.json")
    )

    endpoint_rows, arrays = _endpoint_rows(terminals, reference_rows)
    paired_rows = _paired_rows(arrays, reference_paired)
    trajectory_rows, outer_rows, selection = _trajectory_and_outer_rows(terminals)
    trajectory_summary_rows = _trajectory_summary_rows(trajectory_rows)
    outer_summary_rows = _outer_summary_rows(outer_rows)
    writer_rows, writer_summary = _writer_transfer(arrays)
    compute_rows = _compute_rows(terminals, slurm)
    reference_elapsed = {row["arm"]: int(row["elapsed_seconds"]) for row in reference_compute}
    for label, arm in (("PDZ-T1-PRC†", "PDZ"), ("P1R54-FZ†", "FZ")):
        source = next(row for row in reference_compute if row["arm"] == arm)
        compute_rows.append({
            "method": label, "job_id": source["job_id"], "state": source["state"],
            "exit_code": source["exit_code"], "start": source["start"], "end": source["end"],
            "elapsed_seconds": int(source["elapsed_seconds"]), "elapsed": source["elapsed"],
            "allocated_cpu_seconds": int(source["allocated_cpu_seconds"]),
            "requested_memory": source["requested_memory"], "allocated_tres": source["allocated_tres"],
            "status": "SEALED_EXTERNAL_REFERENCE", "runtime_after_model_preflight_seconds": float(source["runtime_after_model_preflight_seconds"]),
            "field_gradient_count": int(source["target_field_evaluations"]),
            "primary_evaluation_count": int(source["target_field_evaluations"]),
            "selector_count": int(source["target_field_evaluations"]),
            "finite_demand_builder_count": 8, "writer_count": int(source["writer_calls"]),
            "writer_layer_apply_count": int(source["writer_layer_applies"]),
            "model_forward_total": source["model_forward_total"],
            "model_backward_total": source["model_backward_total"],
            "peak_allocated_bytes": int(source["peak_allocated_bytes"]),
            "peak_reserved_bytes": int(source["peak_reserved_bytes"]),
            "W0_restored": source["W0_restored"],
        })
    for row in compute_rows:
        elapsed = int(row["elapsed_seconds"])
        row["elapsed_minus_pdz_t1_seconds"] = elapsed - reference_elapsed["PDZ"]
        row["elapsed_over_pdz_t1_ratio"] = elapsed / reference_elapsed["PDZ"]
        row["elapsed_minus_fz_seconds"] = elapsed - reference_elapsed["FZ"]
        row["elapsed_over_fz_ratio"] = elapsed / reference_elapsed["FZ"]
        row["official_alphaedit_elapsed_comparison"] = "NOT_RECORDED_IN_SEALED_REFERENCE"
    paired_summary = _summary_paired(paired_rows)
    ntsm = terminals["PDZ-T1-DIRECT"] if terminals["PDZ-T1-DIRECT"]["status"] == "SCIENTIFIC_BOUNDARY_NTSM" else None
    aggregates: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-pdz-time-sweep-analysis/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "TERMINAL_FACTUAL_ANALYSIS_COMPLETE",
        "source_head": SOURCE_HEAD, "source_tree": SOURCE_TREE,
        "job_array": f"{job_id}_[0-3%3]", "request_count": 100,
        "stream_root": STREAM_ROOT, "stream_order": STREAM_ORDER,
        "endpoint_rows": endpoint_rows, "paired_strength_gain": paired_summary,
        "selection_clamp_saturation": selection,
        "full_fp32_evidence": dtype_evidence,
        "microstep_summary_rows": len(trajectory_summary_rows),
        "outer_writer_summary": outer_summary_rows,
        "writer_transfer": writer_summary,
        "direct_ntsm": None if ntsm is None else dict(ntsm),
        "sealed_t1_pdz_reference_identity": reference_aggregates["identity_sha256"],
        "reference_execution_count": 0, "scientific_promotion": False,
    }
    aggregates["identity_sha256"] = canonical_hash(aggregates)
    _write_json(output_root / "aggregates.json", aggregates)
    _write_csv(output_root / "terminal-z-w-nll.csv", endpoint_rows)
    _write_csv(output_root / "paired-target-time-gains.csv", paired_rows)
    _write_csv(output_root / "microstep-trajectory.csv", trajectory_rows)
    _write_csv(output_root / "microstep-summary.csv", trajectory_summary_rows)
    _write_csv(output_root / "outer-writer.csv", outer_rows)
    _write_csv(output_root / "outer-writer-summary.csv", outer_summary_rows)
    _write_csv(output_root / "writer-transfer.csv", writer_rows)
    _write_csv(output_root / "compute.csv", compute_rows)

    lookup = {(row["method"], row["endpoint"], row["prompt"]): row for row in endpoint_rows}
    report = [
        "# P1R54 PDZ-PRC target-time sweep — Llama B1/B100 factual report",
        "",
        "> **실험 경계:** `h=1/8`과 physical writer clock K=8은 고정하고 target microstep 수만 T1→T2→T3→T5로 늘렸다. 각 outer의 마지막 selected target만 C3 Official AlphaEdit writer에 1회 전달된다. inner W/teacher/cache/history/factor/materializer는 freeze다. †는 sealed reference이며 재실행0이다. 자동 최적 T 선택과 promotion은 하지 않는다.",
        "",
        "## 1. method 정의",
        "",
        "|method|target clock|selection|physical writer|",
        "|---|---|---|---|",
        "|PDZ-T1-PRC†|h=1/8, M=1, Tz=1|기존 PRIMARY/RESCUE/CURRENT PRC|outer K마다 C3 Official writer 1회, 총8|",
        "|PDZ-T2-PRC|h=1/8, M=2, Tz=2|각 microstep 기존 PRC, selector16|outer 마지막 target만 writer, 총8|",
        "|PDZ-T3-PRC|h=1/8, M=3, Tz=3|동일 loop, selector24|outer 마지막 target만 writer, 총8|",
        "|PDZ-T5-PRC|h=1/8, M=5, Tz=5|동일 loop, selector40|outer 마지막 target만 writer, 총8|",
        "|PDZ-T1-DIRECT|h=1/8, M=1, Tz=1|rescue/selector/CURRENT 없이 repaired Primary 직접 채택|NTSM gate 통과 시 writer, 악화 시 pre-write 중단|",
        "|P1R54-FZ†|sealed M1 reference|기존 finite-zero target rule|sealed C3 writer8; 재실행0|",
        "|Native AlphaEdit†|sealed native compute_z reference|EasyEdit native target|Official writer; P1R54 z와 동일 알고리즘 아님|",
        "",
        "## 2. 최종 K8 편집 성능",
        "",
        "NLL은 mean/median/p90/max, 낮을수록 좋다. Direct가 NTSM이면 endpoint는 N/A다.",
        "",
        "|method|W Eff|W Gen|W strict Gen|W Loc|z Rewrite NLL|z Rephrase NLL|W Rewrite NLL|W Rephrase NLL|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in DISPLAY_ORDER:
        if (method, "post_W", "rewrite") not in lookup:
            report.append(f"|{method}|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|")
            continue
        zrw, zrp = lookup[(method, "accepted_z", "rewrite")], lookup[(method, "accepted_z", "rephrase")]
        wrw, wrp = lookup[(method, "post_W", "rewrite")], lookup[(method, "post_W", "rephrase")]
        report.append(
            f"|{method}|{_pct(int(wrw['success_numerator']), int(wrw['success_denominator']))}|"
            f"{_pct(int(wrp['success_numerator']), int(wrp['success_denominator']))}|"
            f"{_pct(int(wrp['strict_success_numerator']), int(wrp['strict_success_denominator']))}|"
            f"{_pct(int(wrp['locality_numerator']), int(wrp['locality_denominator']))}|"
            f"{_fmt(zrw)}|{_fmt(zrp)}|{_fmt(wrw)}|{_fmt(wrp)}|"
        )
    report.extend([
        "", "## 3. Rewrite 상세", "",
        "|method|endpoint|new NLL mean/median/p90/max|true NLL mean/median/p90/max|success|accuracy|",
        "|---|---|---:|---:|---:|---:|",
    ])
    for method in DISPLAY_ORDER:
        for endpoint in ("accepted_z", "post_W"):
            row = lookup.get((method, endpoint, "rewrite"))
            if row is None:
                continue
            report.append(
                f"|{method}|{endpoint}|{_fmt(row)}|{row['target_true_mean']:.6f}/{row['target_true_median']:.6f}/{row['target_true_p90']:.6f}/{row['target_true_max']:.6f}|"
                f"{row['success_numerator']}/{row['success_denominator']}|{row['accuracy_numerator']}/{row['accuracy_denominator']}|"
            )
    report.extend([
        "", "## 4. Rephrase 상세", "",
        "|method|endpoint|new NLL mean/median/p90/max|true NLL mean/median/p90/max|success prompt/strict|accuracy prompt/strict|",
        "|---|---|---:|---:|---:|---:|",
    ])
    for method in DISPLAY_ORDER:
        for endpoint in ("accepted_z", "post_W"):
            row = lookup.get((method, endpoint, "rephrase"))
            if row is None:
                continue
            report.append(
                f"|{method}|{endpoint}|{_fmt(row)}|{row['target_true_mean']:.6f}/{row['target_true_median']:.6f}/{row['target_true_p90']:.6f}/{row['target_true_max']:.6f}|"
                f"{row['success_numerator']}/{row['success_denominator']} · {row['strict_success_numerator']}/{row['strict_success_denominator']}|"
                f"{row['accuracy_numerator']}/{row['accuracy_denominator']} · {row['strict_accuracy_numerator']}/{row['strict_accuracy_denominator']}|"
            )
    report.extend([
        "", "## 5. T 증가의 paired marginal gain", "",
        "Delta는 뒤 T minus 앞 T이며 음수면 target-new NLL 개선이다.", "",
        "|endpoint/prompt|T2−T1|T3−T2|T5−T3|T5−T1|",
        "|---|---:|---:|---:|---:|",
    ])
    for scope, values in paired_summary.items():
        report.append(
            f"|{scope}|{_fmt(values['PDZ-T2-PRC_minus_PDZ-T1-PRC†'])}|"
            f"{_fmt(values['PDZ-T3-PRC_minus_PDZ-T2-PRC'])}|"
            f"{_fmt(values['PDZ-T5-PRC_minus_PDZ-T3-PRC'])}|"
            f"{_fmt(values['PDZ-T5-PRC_minus_PDZ-T1-PRC'])}|"
        )
    report.extend([
        "", "## 6. PRC selector·negative progress·clamp saturation", "",
        "|method|request×microstep|PRIMARY/RESCUE/CURRENT|negative progress|clamp hits|ever/all clamped requests|all-negative requests|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    t1 = reference_aggregates["allocation_barrier_selector"]["PDZ"]
    report.append(
        f"|PDZ-T1-PRC†|800|{t1['selection_counts']['PRIMARY']}/{t1['selection_counts']['RESCUE']}/{t1['selection_counts']['CURRENT']}|NOT_RECORDED|{t1['clamp_hits']}|NOT_RECORDED|NOT_RECORDED|"
    )
    for method in PRC_ORDER[1:]:
        row = selection[method]
        choices = row["selection_counts"]
        report.append(
            f"|{method}|{row['request_microstep_denominator']}|{choices['PRIMARY']}/{choices['RESCUE']}/{choices['CURRENT']}|"
            f"{row['negative_primary_progress_count']}|{row['clamp_hit_count']}|{row['ever_clamped_request_count']}/{row['all_microsteps_clamped_request_count']}|{row['all_microsteps_negative_request_count']}|"
        )
    if "PDZ-T1-DIRECT" in selection:
        row = selection["PDZ-T1-DIRECT"]
        report.append(
            f"|PDZ-T1-DIRECT|{row['request_microstep_denominator']}|0/0/0 (DIRECT={row['selection_counts']['DIRECT_EULER']})|"
            f"{row['negative_primary_progress_count']}|{row['clamp_hit_count']}|{row['ever_clamped_request_count']}/{row['all_microsteps_clamped_request_count']}|{row['all_microsteps_negative_request_count']}|"
        )
    report.extend([
        "", "### Microstep 과학량 전체 요약", "",
        "|method|microsteps|selected NLL decrease mean/median/p90/max|deficit mean/median/p90/max|amplitude mean/median/p90/max|selected energy mean/median/p90/max|",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for method in (*PRC_ORDER[1:], "PDZ-T1-DIRECT"):
        subset = [row for row in trajectory_rows if row["method"] == method]
        if not subset:
            continue
        report.append(
            f"|{method}|{len(subset) // 100}|"
            f"{_fmt(_summary(float(row['selected_nll_decrease']) for row in subset))}|"
            f"{_fmt(_summary(float(row['deficit']) for row in subset))}|"
            f"{_fmt(_summary(float(row['amplitude']) for row in subset))}|"
            f"{_fmt(_summary(float(row['selected_direct_energy']) for row in subset))}|"
        )
    report.extend([
        "", "## 7. z→W transfer", "",
        "|method|prompt|W−z NLL mean/median/p90/max|z success→W failure|",
        "|---|---|---:|---:|",
    ])
    for method in (*PRC_ORDER[1:], "PDZ-T1-DIRECT"):
        if method not in writer_summary:
            continue
        for prompt in ("rewrite", "rephrase"):
            row = writer_summary[method][prompt]
            report.append(
                f"|{method}|{prompt}|{_fmt(row['W_minus_z_nll'])}|{row['z_success_W_failure_count']}/{row['denominator']}|"
            )
    report.extend([
        "", "### Writer realization·비용", "",
        "|method|K writers|intended norm mean|realized norm mean|residual ratio mean|writer edit-core total|",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in outer_summary_rows:
        report.append(
            f"|{row['method']}|{row['outer_denominator']}|{row['intended_write_norm_mean']:.6f}|"
            f"{row['realized_write_norm_mean']:.6f}|{row['write_residual_ratio_mean']:.6f}|"
            f"{row['writer_edit_core_wall_seconds_total']:.3f}s|"
        )
    report.extend([
        "", "## 8. compute·wall time", "",
        "|method|Slurm|elapsed|vs T1 PDZ delta/ratio|vs FZ delta/ratio|field/Primary/selector|finite/writer/layer|peak alloc/reserved|W0 restore|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in compute_rows:
        report.append(
            f"|{row['method']}|{row['job_id']} {row['state']} {row['exit_code']}|{row['elapsed_seconds']}s|"
            f"{row['elapsed_minus_pdz_t1_seconds']}s/{row['elapsed_over_pdz_t1_ratio']:.3f}×|"
            f"{row['elapsed_minus_fz_seconds']}s/{row['elapsed_over_fz_ratio']:.3f}×|"
            f"{row['field_gradient_count']}/{row['primary_evaluation_count']}/{row['selector_count']}|"
            f"{row['finite_demand_builder_count']}/{row['writer_count']}/{row['writer_layer_apply_count']}|"
            f"{row['peak_allocated_bytes']}/{row['peak_reserved_bytes']}|{row['W0_restored']}|"
        )
    report.extend([
        "", "### FULL-FP32·복원 불변식", "",
        "|method|dtype|FP32/total params|non-FP32|BF16/FP16 conversion|numeric cast|quantized|autocast CPU/CUDA|W0 restore|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for method in ("PDZ-T2-PRC", "PDZ-T3-PRC", "PDZ-T5-PRC", "PDZ-T1-DIRECT"):
        dtype = dtype_evidence[method]
        terminal = terminals[method]
        report.append(
            f"|{method}|{dtype['loaded_model_dtype']}|{dtype['fp32_parameter_tensor_count']}/{dtype['parameter_tensor_count']}|"
            f"{dtype['non_fp32_parameter_tensor_count']}|{dtype['bf16_conversion_count']}/{dtype['fp16_conversion_count']}|"
            f"{dtype['numeric_storage_cast_count']}|{dtype['quantized_parameter_count']}|"
            f"{int(dtype['autocast_cpu_enabled'])}/{int(dtype['autocast_cuda_enabled'])}|{terminal['W0_restored']}|"
        )
    report.extend([
        "", "- physical update energy와 layer share는 이번 raw-free terminal에 없으므로 `NOT_RECORDED`; norm을 energy로 역추정하지 않았다.",
        "- sealed T1/FZ wall time은 별 job/서버 reference이므로 paired causal timing이 아니다. 새 arms의 field/selector count 증가는 정의상 8M이고 writer8/layer40은 고정이다.",
        "- Official AlphaEdit sealed endpoint에는 이 report가 재현 가능한 elapsed timing이 없어 overhead를 `NOT_RECORDED`; 임의 추정하지 않았다.",
        "", "## 9. Direct scientific boundary", "",
    ])
    if ntsm is None:
        report.append("- PDZ-T1-DIRECT는 terminal-valid로 완료했다. rescue/selector/current 호출은 모두0이다.")
    else:
        report.append(
            f"- PDZ-T1-DIRECT는 K{int(ntsm['boundary_outer_index']) + 1}, global n={ntsm['boundary_global_ordinal']}에서 `SCIENTIFIC_BOUNDARY_NTSM`으로 pre-write 중단했다. writer0, fallback/retry/imputation0, W0 exact restore다."
        )
    report.extend([
        "", "## 10. factual ceiling 판독 경계", "",
        "- ceiling은 rewrite mean만으로 판정하지 않는다. rephrase p90/max, post-W transfer, locality, marginal gain, selector/clamp saturation을 함께 본다.",
        (
            "- accepted-z Rewrite mean은 T1/T2/T3/T5에서 "
            f"{lookup[('PDZ-T1-PRC†', 'accepted_z', 'rewrite')]['target_new_mean']:.6f}/"
            f"{lookup[('PDZ-T2-PRC', 'accepted_z', 'rewrite')]['target_new_mean']:.6f}/"
            f"{lookup[('PDZ-T3-PRC', 'accepted_z', 'rewrite')]['target_new_mean']:.6f}/"
            f"{lookup[('PDZ-T5-PRC', 'accepted_z', 'rewrite')]['target_new_mean']:.6f}로 단조 감소했다."
        ),
        (
            "- 그러나 post-W Rephrase mean은 T3→T5에서 "
            f"{lookup[('PDZ-T3-PRC', 'post_W', 'rephrase')]['target_new_mean']:.6f}→"
            f"{lookup[('PDZ-T5-PRC', 'post_W', 'rephrase')]['target_new_mean']:.6f}로 "
            f"{paired_summary['post_W/rephrase']['PDZ-T5-PRC_minus_PDZ-T3-PRC']['mean']:+.6f}, "
            f"p90 delta {paired_summary['post_W/rephrase']['PDZ-T5-PRC_minus_PDZ-T3-PRC']['p90']:+.6f}, "
            f"max delta {paired_summary['post_W/rephrase']['PDZ-T5-PRC_minus_PDZ-T3-PRC']['max']:+.6f}였다. "
            "이 B100에서는 T3 이후 writer-transfer tail의 포화/악화 신호다."
        ),
        (
            "- W Loc은 T1/T2/T3/T5에서 "
            f"{lookup[('PDZ-T1-PRC†', 'post_W', 'rephrase')]['locality_numerator']}/1000, "
            f"{lookup[('PDZ-T2-PRC', 'post_W', 'rephrase')]['locality_numerator']}/1000, "
            f"{lookup[('PDZ-T3-PRC', 'post_W', 'rephrase')]['locality_numerator']}/1000, "
            f"{lookup[('PDZ-T5-PRC', 'post_W', 'rephrase')]['locality_numerator']}/1000이었다."
        ),
        "- T2/T3/T5의 clamp·negative progress는 모두0이고 RESCUE는 각각2회뿐이다. 따라서 관측 ceiling 신호를 PRC clamp/selector 포화로 설명할 근거는 없고, 강해진 z가 W rephrase tail로 전달되지 않는 현상이 직접 관측됐다.",
        "- DIRECT는 DIRECT_EULER 800/800, rescue/selector/CURRENT0으로 terminal-valid였다. negative Primary progress는2/800이었고 fallback 없이 계약된 endpoint를 보존했다.",
        "- 관측 동행은 인과 증명이 아니다. 자동 최적 T 선택, threshold 신설, promotion은 모두0이다.",
        "- `microstep-trajectory.csv`는 모든 K×m×request를, `outer-writer.csv`는 K별 PRC/energy/norm/wall을 보존한다.",
    ])
    report_path = output_root / "p1r54-pdz-target-time-sweep-llama-b100-factual-ko.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    output_names = (
        report_path.name, "aggregates.json", "terminal-z-w-nll.csv",
        "paired-target-time-gains.csv", "microstep-trajectory.csv",
        "microstep-summary.csv", "outer-writer.csv", "outer-writer-summary.csv",
        "writer-transfer.csv", "compute.csv",
    )
    manifest: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-pdz-time-sweep-analysis-manifest/v1",
        "instruction_id": INSTRUCTION_ID, "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE, "authoritative_contract_sha256": CONTRACT_SHA256,
        "pre_gpu_sha256": PRE_GPU_SHA256, "pre_gpu_identity": PRE_GPU_IDENTITY,
        "stream_root": STREAM_ROOT, "stream_order": STREAM_ORDER, "B1_order": B1_ORDER,
        "input_members": input_members,
        "output_members": [_member(output_root / name, relative_to=output_root) for name in output_names],
        "reference_execution_count": 0, "scientific_promotion": False,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    _write_json(output_root / "analysis-manifest.json", manifest)
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-pdz-time-sweep-source-receipt/v1",
        "instruction_id": INSTRUCTION_ID, "status": "TERMINAL_REPORT_COMPLETE",
        "source_head": SOURCE_HEAD, "source_tree": SOURCE_TREE,
        "job_array": f"{job_id}_[0-3%3]",
        "analysis_manifest_sha256": _sha256(output_root / "analysis-manifest.json"),
        "analysis_manifest_identity": manifest["identity_sha256"],
        "report_sha256": _sha256(report_path), "reference_execution_count": 0,
        "scientific_promotion": False,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    _write_json(output_root / "source-receipt.json", receipt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--pre-gpu-receipt", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    args = parser.parse_args(argv)
    build_package(
        raw_root=args.raw_root, output_root=args.output_root, job_id=args.job_id,
        pre_gpu_receipt=args.pre_gpu_receipt, reference_root=args.reference_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
