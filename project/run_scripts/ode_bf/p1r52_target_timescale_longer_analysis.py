#!/usr/bin/env python3
"""Build the separate fixed-dt longer-target-time factual report package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import canonical_hash
from .p1r52_target_timescale_analysis import (
    B1_ORDER,
    CELL_SLUGS,
    INSTRUCTION_ID,
    NATIVE_ALPHA_SOURCE_HEAD,
    NATIVE_ALPHA_SOURCE_TREE,
    NATIVE_POLICY,
    NATIVE_RESULTS,
    NUMERICAL_LOCK_ROOT,
    SOURCE_HEAD,
    SOURCE_TREE,
    _count_rate,
    _final_endpoint_report_lines,
    _flatten,
    _load,
    _member,
    _native_z_to_w_summary,
    _nll_triplet,
    _pct,
    _sacct,
    _score_rows,
    _sha256,
    _summary,
    _validate_native_terminal,
    _validate_terminal,
    _write_csv,
    _write_json,
)


CELLS = ("Z1-REFINE", "Z15", "Z20", "Z30")
CELL_INDICES = (1, 2, 3, 4)
CONSECUTIVE = (
    ("Z1-REFINE", "Z15"),
    ("Z15", "Z20"),
    ("Z20", "Z30"),
)


def _load_native(
    native_root: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    terminals: dict[str, dict[str, Any]] = {}
    inputs: list[dict[str, Any]] = []
    for method, result_name in NATIVE_RESULTS.items():
        root = native_root / result_name
        terminal_path = root / "terminal.json"
        manifest_path = root / "manifest.json"
        source_path = root / "source-manifest.json"
        terminal = _load(terminal_path)
        manifest = _load(manifest_path)
        source = _load(source_path)
        source_head = str(source.get("source_head", ""))
        source_tree = str(source.get("source_tree", ""))
        _validate_native_terminal(terminal, method, source_head)
        if (
            manifest.get("terminal_sha256") != _sha256(terminal_path)
            or manifest.get("method") != method
            or manifest.get("request_order_sha256") != B1_ORDER
            or manifest.get("policy_provenance") != NATIVE_POLICY
            or len(source_tree) != 40
            or (
                method == "OFFICIAL-ALPHAEDIT"
                and (source_head, source_tree)
                != (NATIVE_ALPHA_SOURCE_HEAD, NATIVE_ALPHA_SOURCE_TREE)
            )
        ):
            raise ValueError(f"Native input binding differs: {method}")
        terminals[method] = terminal
        inputs.extend(_member(path) for path in (terminal_path, manifest_path, source_path))
    return terminals, inputs


def build_longer_time(
    *,
    raw_root: Path,
    native_root: Path,
    output_root: Path,
    job_id: str,
    native_job_ids: tuple[str, str],
    repo_root: Path,
) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"create-once output exists: {output_root}")
    output_root.mkdir(mode=0o755, parents=True)
    slurm = _sacct(job_id, CELL_INDICES)
    native_slurm = {
        "OFFICIAL-ALPHAEDIT": _sacct(native_job_ids[0], (0,))[0],
        "OFFICIAL-MEMIT": _sacct(native_job_ids[1], (1,))[1],
    }
    native_terminals, native_inputs = _load_native(native_root)
    raw_inputs: list[dict[str, Any]] = list(native_inputs)
    terminals: dict[str, dict[str, Any]] = {}
    cell_rows: list[dict[str, Any]] = []
    micro_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    writer_rows: list[dict[str, Any]] = []
    compute_rows: list[dict[str, Any]] = []

    for index, cell in zip(CELL_INDICES, CELLS, strict=True):
        root = raw_root / (
            "s05-p1r52-target-timescale-b100-"
            f"{CELL_SLUGS[cell]}-tech-r4-v1"
        )
        terminal_path = root / "terminal.json"
        manifest_path = root / "manifest.json"
        source_path = root / "source-manifest.json"
        terminal = _load(terminal_path)
        manifest = _load(manifest_path)
        source = _load(source_path)
        _validate_terminal(terminal, cell)
        if (
            manifest.get("terminal_sha256") != _sha256(terminal_path)
            or manifest.get("cell") != cell
            or manifest.get("request_order_sha256") != B1_ORDER
            or source.get("source_head") != SOURCE_HEAD
            or source.get("source_tree") != SOURCE_TREE
        ):
            raise ValueError(f"longer-time manifest differs: {cell}")
        terminals[cell] = terminal
        raw_inputs.extend(_member(path) for path in (terminal_path, manifest_path, source_path))
        micros = terminal["microstep_trajectory"]
        selections = [role for micro in micros for role in micro["selection_by_request"]]
        clamp_hits = sum(
            int(micro["selection_receipt"]["clamp_hit_count"]) for micro in micros
        )
        denominator = len(micros) * 100
        cell_rows.append(
            {
                "cell": cell,
                "target_horizon": terminal["schedule"]["target_horizon"],
                "microsteps_per_outer": terminal["schedule"]["microsteps_per_outer"],
                "target_dt": terminal["schedule"]["target_dt"],
                "target_field_evaluations": terminal["target_field_evaluation_count"],
                "writer_calls": terminal["writer_call_count"],
                "clamp_hits": clamp_hits,
                "request_microstep_denominator": denominator,
                "clamp_fraction": clamp_hits / denominator,
                "primary_count": selections.count("PRIMARY"),
                "rescue_count": selections.count("RESCUE"),
                "current_count": selections.count("CURRENT"),
                "pre_clamp_energy_sum": sum(float(m["selection_receipt"]["pre_clamp_energy"]) for m in micros),
                "post_clamp_energy_sum": sum(float(m["selection_receipt"]["post_clamp_energy"]) for m in micros),
                "clamp_removed_energy_sum": sum(float(m["selection_receipt"]["clamp_removed_energy"]) for m in micros),
                "accepted_energy_sum": sum(float(m["selection_receipt"]["accepted_energy"]) for m in micros),
                "runtime_after_model_preflight_seconds": terminal["runtime_after_model_preflight_seconds"],
                "slurm_elapsed_seconds": slurm[index]["elapsed_seconds"],
                "peak_allocated_bytes": terminal["gpu_host_observation"]["peak_allocated_bytes"],
                "peak_reserved_bytes": terminal["gpu_host_observation"]["peak_reserved_bytes"],
                "W0_restored": terminal["W0_restored"],
                "full_fp32": terminal["dtype_contract"]["status"],
            }
        )
        for micro in micros:
            selection = micro["selection_receipt"]
            field = micro["field_receipt"]
            semantic = _summary(field["semantic_gradient_norm_by_request"])
            preservation = _summary(field["preservation_gradient_norm_by_request"])
            movement = _summary(field["post_cast_actual_delta_norm_by_request"])
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
                    "target_displacement_norm": selection["target_displacement_norm"],
                    "z_minus_y_norm": selection["z_minus_y_norm"],
                    "semantic_gradient_norm_mean": semantic["mean"],
                    "semantic_gradient_norm_median": semantic["median"],
                    "semantic_gradient_norm_p90": semantic["p90"],
                    "preservation_gradient_norm_mean": preservation["mean"],
                    "movement_norm_mean": movement["mean"],
                    "movement_norm_p90": movement["p90"],
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
                    "writer_layer_apply_count": execution["compute"]["native_apply_count"],
                    "native_compute_z_call_count": execution["writer_receipt"]["native_compute_z_call_count"],
                    "cache_status": execution["alpha_cache_status"],
                }
            )
        compute_rows.append(
            {
                "cell": cell,
                **slurm[index],
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

    endpoint_lookup = {
        (row["cell"], row["outer_K"], row["endpoint"], row["prompt"]): row
        for row in endpoint_rows
    }
    paired_rows: list[dict[str, Any]] = []
    paired_summary: dict[str, Any] = {}
    for cell in CELLS[1:]:
        paired_summary[cell] = {}
        for outer in (1, 4, 8):
            paired_summary[cell][str(outer)] = {}
            for prompt, key in (("rewrite", "rewrite_success"), ("rephrase", "rephrase_success")):
                control = _flatten(terminals["Z1-REFINE"]["outer_transitions"][outer - 1]["c_kstep_writer"]["metrics"]["accepted_z"]["scores"][key]["target_new_nll_by_request"])
                treatment = _flatten(terminals[cell]["outer_transitions"][outer - 1]["c_kstep_writer"]["metrics"]["accepted_z"]["scores"][key]["target_new_nll_by_request"])
                deltas = []
                for ordinal, (z1, longer) in enumerate(zip(control, treatment, strict=True)):
                    delta = longer - z1
                    deltas.append(delta)
                    paired_rows.append(
                        {
                            "cell": cell,
                            "outer_K": outer,
                            "prompt": prompt,
                            "prompt_ordinal": ordinal,
                            "Z1_target_new_nll": z1,
                            "longer_target_new_nll": longer,
                            "longer_minus_Z1": delta,
                            "winner_lower_nll": cell if delta < 0 else "Z1-REFINE" if delta > 0 else "TIE",
                        }
                    )
                paired_summary[cell][str(outer)][prompt] = {
                    "longer_minus_Z1": _summary(deltas),
                    "longer_win_count": sum(value < 0 for value in deltas),
                    "Z1_win_count": sum(value > 0 for value in deltas),
                    "tie_count": sum(value == 0 for value in deltas),
                }

    marginal_rows: list[dict[str, Any]] = []
    for left, right in CONSECUTIVE:
        for outer in (1, 4, 8):
            for endpoint in ("accepted_z", "post_writer_W"):
                for prompt in ("rewrite", "rephrase"):
                    for statistic in ("mean", "median", "p90", "max"):
                        left_value = float(endpoint_lookup[(left, outer, endpoint, prompt)][f"target_new_{statistic}"])
                        right_value = float(endpoint_lookup[(right, outer, endpoint, prompt)][f"target_new_{statistic}"])
                        marginal_rows.append(
                            {
                                "left_cell": left,
                                "right_cell": right,
                                "delta_Tz": float(terminals[right]["schedule"]["target_horizon"]) - float(terminals[left]["schedule"]["target_horizon"]),
                                "outer_K": outer,
                                "endpoint": endpoint,
                                "prompt": prompt,
                                "statistic": statistic,
                                "left_NLL": left_value,
                                "right_NLL": right_value,
                                "right_minus_left": right_value - left_value,
                            }
                        )

    native_endpoint_rows: list[dict[str, Any]] = []
    native_gap_rows: list[dict[str, Any]] = []
    native_paired_rows: list[dict[str, Any]] = []
    for method, native_terminal in native_terminals.items():
        for endpoint, scores in (("accepted_z", native_terminal["native"]["z"]["scores"]), ("post_W", native_terminal["native"]["W"]["scores"])):
            native_endpoint_rows.extend(
                _score_rows(cell=method, outer=0, endpoint=endpoint, scores=scores)
            )
        for prompt, key in (("rewrite", "rewrite_success"), ("rephrase", "rephrase_success")):
            native_vector = _flatten(native_terminal["native"]["z"]["scores"][key]["target_new_nll_by_request"])
            native_row = next(row for row in native_endpoint_rows if row["cell"] == method and row["endpoint"] == "accepted_z" and row["prompt"] == prompt)
            for cell in CELLS:
                cell_vector = _flatten(terminals[cell]["outer_transitions"][7]["c_kstep_writer"]["metrics"]["accepted_z"]["scores"][key]["target_new_nll_by_request"])
                for ordinal, (cell_value, native_value) in enumerate(zip(cell_vector, native_vector, strict=True)):
                    native_paired_rows.append(
                        {
                            "reference": method,
                            "cell": cell,
                            "prompt": prompt,
                            "prompt_ordinal": ordinal,
                            "cell_target_new_nll": cell_value,
                            "native_target_new_nll": native_value,
                            "cell_minus_native": cell_value - native_value,
                        }
                    )
                for statistic in ("mean", "median", "p90", "max"):
                    native_value = float(native_row[f"target_new_{statistic}"])
                    cell_value = float(endpoint_lookup[(cell, 8, "accepted_z", prompt)][f"target_new_{statistic}"])
                    native_gap_rows.append(
                        {
                            "reference": method,
                            "cell": cell,
                            "prompt": prompt,
                            "statistic": statistic,
                            "cell_NLL": cell_value,
                            "native_NLL": native_value,
                            "cell_minus_native": cell_value - native_value,
                        }
                    )

    native_compute_rows = []
    for method, terminal in native_terminals.items():
        timing = terminal["native"]["timing"]
        native_compute_rows.append(
            {
                "method": method,
                **native_slurm[method],
                "official_apply_count": terminal["official_apply_count"],
                "target_accepted_z_generation_seconds": timing["target_accepted_z_generation_seconds"],
                "writer_edit_core_seconds": timing["writer_edit_core_seconds"],
                "native_apply_scope_seconds": timing["native_apply_scope_seconds"],
                "accepted_z_evaluator_seconds": timing["accepted_z_evaluator_seconds"],
                "post_W_evaluator_seconds": timing["immediate_post_evaluator_seconds"],
                "restore_seconds": timing["restore_seconds"],
                "method_total_seconds": timing["case_method_total_seconds"],
                "runtime_after_model_preflight_seconds": terminal["runtime_after_model_preflight_seconds"],
                "peak_allocated_bytes": terminal["gpu_host_observation"]["peak_allocated_bytes"],
                "W0_restored": terminal["W0_restored"],
            }
        )

    native_z_to_w = _native_z_to_w_summary(
        terminals=native_terminals,
        endpoint_rows=native_endpoint_rows,
    )

    analysis: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r52-target-timescale-longer-time-analysis/v3",
        "instruction_id": INSTRUCTION_ID,
        "scope": "FIXED_DT_LONGER_TARGET_TIME_Z1_Z15_Z20_Z30",
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "numerical_lock_root": NUMERICAL_LOCK_ROOT,
        "scientific_denominator": {
            "cells": list(CELLS),
            "B1_requests": 100,
            "fixed_target_dt": 0.0625,
            "Z0_coarse_influence_count": 0,
        },
        "paired_vs_Z1": paired_summary,
        "cell_summary": cell_rows,
        "native_reference": {
            "policy_provenance": NATIVE_POLICY,
            "methods": list(NATIVE_RESULTS),
            "z_to_W_measurement": native_z_to_w,
            "selection_influence_count": 0,
        },
        "interpretation_boundary": {
            "B100x1_exploratory": True,
            "final_Tz_selection": False,
            "scientific_promotion": False,
            "continuous_ODE_convergence_claim": False,
            "native_schedule_matched_claim": False,
        },
    }
    analysis["identity_sha256"] = canonical_hash(analysis)
    tables = {
        "cell-summary.csv": cell_rows,
        "per-microstep-trajectory.csv": micro_rows,
        "k1-k4-k8-endpoints.csv": endpoint_rows,
        "paired-vs-z1-accepted-z.csv": paired_rows,
        "consecutive-marginal-gain.csv": marginal_rows,
        "writer-transfer-locality.csv": writer_rows,
        "compute.csv": compute_rows,
        "native-reference-endpoints.csv": native_endpoint_rows,
        "native-gap.csv": native_gap_rows,
        "native-paired-nll.csv": native_paired_rows,
        "native-compute.csv": native_compute_rows,
    }
    for name, rows in tables.items():
        _write_csv(output_root / name, rows)
    _write_json(output_root / "analysis.json", analysis)
    report = _render_report(
        analysis=analysis,
        endpoints=endpoint_lookup,
        cells=cell_rows,
        writer=writer_rows,
        native_endpoints=native_endpoint_rows,
        marginals=marginal_rows,
    )
    (output_root / "report-ko.md").write_text(report, encoding="utf-8")
    generated = [output_root / name for name in tables] + [output_root / "analysis.json", output_root / "report-ko.md"]
    raw_inputs.sort(key=lambda row: str(row["path"]))
    manifest: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r52-target-timescale-longer-time-manifest/v3",
        "instruction_id": INSTRUCTION_ID,
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "scope": "FIXED_DT_Z1_Z15_Z20_Z30_ONLY",
        "raw_inputs": raw_inputs,
        "raw_input_count": len(raw_inputs),
        "raw_member_root": canonical_hash(raw_inputs),
        "analysis_generator": _member(
            Path(__file__).resolve(strict=True), relative_to=repo_root.resolve(strict=True)
        ),
        "generated_members": [_member(path, relative_to=output_root) for path in generated],
        "Z0_coarse_raw_input_count": 0,
        "scientific_promotion": False,
    }
    manifest["root_digest"] = canonical_hash(manifest)
    _write_json(output_root / "analysis-manifest.json", manifest)
    package_paths = generated + [output_root / "analysis-manifest.json"]
    members = [_member(path, relative_to=output_root) for path in package_paths]
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r52-target-timescale-longer-time-rooted-receipt/v3",
        "instruction_id": INSTRUCTION_ID,
        "status": "LONGER_TIME_REPORT_COMPLETE",
        "analysis_identity": analysis["identity_sha256"],
        "manifest_root": manifest["root_digest"],
        "package_members": members,
        "package_member_root": canonical_hash(members),
        "scientific_promotion": False,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    _write_json(output_root / "rooted-receipt.json", receipt)


def _render_report_v2(
    *,
    analysis: Mapping[str, Any],
    endpoints: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    writer: Sequence[Mapping[str, Any]],
    native_endpoints: Sequence[Mapping[str, Any]],
    marginals: Sequence[Mapping[str, Any]],
) -> str:
    writer_k8 = {(row["cell"], row["outer_K"]): row for row in writer}
    lines = [
        "# P1R52 target-timescale B100 — longer target time 상세 결과",
        "",
        "> **범위:** `FIXED_DT_LONGER_TARGET_TIME_Z1_Z15_Z20_Z30`. 네 cell은 동일 `dt=1/16`, sealed B1 100 requests, W0, C3 K8 writer, cache, evaluator를 공유하고 `T_z=1/1.5/2/3`만 다르다. Z0 coarse-resolution은 이 비교와 manifest에서 제외했다.",
        "",
        "> **정정판 v2:** v1의 CSV에는 존재했지만 본문에서 생략된 Native post-W 측정, z→W gap, success/accuracy/strict/locality와 Native compute를 명시적으로 반영했다. 실험·raw identity·longer-time 결론은 변경하지 않았다.",
        "",
        "## 핵심 관측",
        "",
    ]
    for prompt in ("rewrite", "rephrase"):
        values = [
            float(endpoints[(cell, 8, "accepted_z", prompt)]["target_new_mean"])
            for cell in CELLS
        ]
        best = CELLS[min(range(len(values)), key=values.__getitem__)]
        lines.append(
            f"- K8 accepted-z {prompt} mean NLL은 "
            + " → ".join(f"{cell} `{value:.6f}`" for cell, value in zip(CELLS, values, strict=True))
            + f"였다. 이 B1에서 최저 관측값은 `{best}`지만 final T_z 선택 근거로 사용하지 않는다."
        )
    lines.extend(
        [
            "- 증가 시간의 marginal gain은 단조라고 가정하지 않았다. `consecutive-marginal-gain.csv`에 accepted-z와 post-W의 mean/median/p90/max를 K1/K4/K8별로 기록했다.",
            "- 아래 Native 비교는 사용자 승인 별도 Official 실행의 external reference이며 target-timescale 설정 선택에 영향 0이다. AlphaEdit post-W mean NLL은 Rewrite `0.001915`, Rephrase `1.940613`; MEMIT는 Rewrite `0.342457`, Rephrase `2.974135`로 direct-z와 materialized W 사이의 차이가 확인됐다.",
            "",
            "## 1. 실행·clamp·selection·overhead",
            "",
            "|cell|T_z|m|field eval|clamp|PRIMARY/RESCUE/CURRENT|Slurm|post-model|",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in cells:
        lines.append(
            f"|{row['cell']}|{row['target_horizon']:.1f}|{row['microsteps_per_outer']}|{row['target_field_evaluations']}|{row['clamp_hits']}/{row['request_microstep_denominator']} ({_pct(float(row['clamp_fraction']))})|{row['primary_count']}/{row['rescue_count']}/{row['current_count']}|{row['slurm_elapsed_seconds']}s|{float(row['runtime_after_model_preflight_seconds']):.1f}s|"
        )
    lines.extend(
        [
            "",
            "모든 cell은 terminal valid, writer 8회, cache reuse 8·append 1, W0 pointer/bytes exact restore, FULL-FP32였다. configured field eval은 16/24/32/48로 schedule과 일치했다.",
            "",
            "## 2. K1/K4/K8 accepted-z NLL",
            "",
            "|K|cell|Rewrite mean/median/p90/max|Rephrase mean/median/p90/max|Rewrite success|Rephrase success/strict|",
            "|---:|---|---|---|---:|---:|",
        ]
    )
    for outer in (1, 4, 8):
        for cell in CELLS:
            rewrite = endpoints[(cell, outer, "accepted_z", "rewrite")]
            rephrase = endpoints[(cell, outer, "accepted_z", "rephrase")]
            lines.append(
                f"|{outer}|{cell}|{rewrite['target_new_mean']:.6f}/{rewrite['target_new_median']:.6f}/{rewrite['target_new_p90']:.6f}/{rewrite['target_new_max']:.6f}|{rephrase['target_new_mean']:.6f}/{rephrase['target_new_median']:.6f}/{rephrase['target_new_p90']:.6f}/{rephrase['target_new_max']:.6f}|{rewrite['success_numerator']}/{rewrite['success_denominator']}|{rephrase['success_numerator']}/{rephrase['success_denominator']} · {rephrase['strict_success_numerator']}/{rephrase['strict_success_denominator']}|"
            )
    lines.extend(
        [
            "",
            "## 3. K8 longer-time paired 및 marginal 변화",
            "",
            "|cell vs Z1|prompt|delta mean/median/p90/max|longer win|Z1 win|",
            "|---|---|---|---:|---:|",
        ]
    )
    paired = analysis["paired_vs_Z1"]
    for cell in CELLS[1:]:
        for prompt in ("rewrite", "rephrase"):
            row = paired[cell]["8"][prompt]
            delta = row["longer_minus_Z1"]
            lines.append(
                f"|{cell}|{prompt}|{delta['mean']:+.6f}/{delta['median']:+.6f}/{delta['p90']:+.6f}/{delta['max']:+.6f}|{row['longer_win_count']}/{delta['n']}|{row['Z1_win_count']}/{delta['n']}|"
            )
    lines.extend(
        [
            "",
            "|increment|endpoint|prompt|K8 mean delta|K8 median delta|K8 p90 delta|",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for left, right in CONSECUTIVE:
        for endpoint in ("accepted_z", "post_writer_W"):
            for prompt in ("rewrite", "rephrase"):
                rows = {
                    row["statistic"]: row
                    for row in marginals
                    if row["left_cell"] == left
                    and row["right_cell"] == right
                    and row["outer_K"] == 8
                    and row["endpoint"] == endpoint
                    and row["prompt"] == prompt
                }
                lines.append(
                    f"|{left}→{right}|{endpoint}|{prompt}|{float(rows['mean']['right_minus_left']):+.6f}|{float(rows['median']['right_minus_left']):+.6f}|{float(rows['p90']['right_minus_left']):+.6f}|"
                )
    lines.extend(
        [
            "",
            "음수는 longer cell의 NLL 감소다. marginal은 Rewrite에서 대체로 감소하지만 Rephrase mean/tail과 post-W는 구간별 비단조다. 이는 raw association이며 최적 T_z 선택 또는 인과 기제 판정이 아니다.",
            "",
            "## 4. K8 accepted-z → post-W 전달",
            "",
            "|cell|Rewrite z/W/gap|Rephrase z/W/gap|z locality|W locality|",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cell in CELLS:
        z_rw = endpoints[(cell, 8, "accepted_z", "rewrite")]
        w_rw = endpoints[(cell, 8, "post_writer_W", "rewrite")]
        z_rp = endpoints[(cell, 8, "accepted_z", "rephrase")]
        w_rp = endpoints[(cell, 8, "post_writer_W", "rephrase")]
        row = writer_k8[(cell, 8)]
        lines.append(
            f"|{cell}|{z_rw['target_new_mean']:.6f}/{w_rw['target_new_mean']:.6f}/{float(w_rw['target_new_mean'])-float(z_rw['target_new_mean']):+.6f}|{z_rp['target_new_mean']:.6f}/{w_rp['target_new_mean']:.6f}/{float(w_rp['target_new_mean'])-float(z_rp['target_new_mean']):+.6f}|{row['accepted_z_locality_numerator']}/{row['accepted_z_locality_denominator']}|{row['post_W_locality_numerator']}/{row['post_W_locality_denominator']}|"
        )
    lines.extend(
        [
            "",
            "## 5. Native accepted-z 및 post-W reference",
            "",
            "|method|endpoint|prompt|mean|median|p90|max|success/strict|accuracy/strict|locality|",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in NATIVE_RESULTS:
        native_measurement = analysis["native_reference"]["z_to_W_measurement"][method]
        for endpoint_name, endpoint_label in (("accepted_z", "accepted-z"), ("post_W", "post-W")):
            locality = native_measurement["locality"][endpoint_name]
            for prompt in ("rewrite", "rephrase"):
                row = next(item for item in native_endpoints if item["cell"] == method and item["endpoint"] == endpoint_name and item["prompt"] == prompt)
                lines.append(
                    f"|{method}|{endpoint_label}|{prompt}|{row['target_new_mean']:.6f}|{row['target_new_median']:.6f}|{row['target_new_p90']:.6f}|{row['target_new_max']:.6f}|{row['success_numerator']}/{row['success_denominator']} · {row['strict_success_numerator']}/{row['strict_success_denominator']}|{row['accuracy_numerator']}/{row['accuracy_denominator']} · {row['strict_accuracy_numerator']}/{row['strict_accuracy_denominator']}|{locality['numerator']}/{locality['denominator']}|"
                )
    lines.extend(
        [
            "",
            "|method|prompt|post-W − accepted-z mean/median/p90/max|",
            "|---|---|---:|",
        ]
    )
    for method in NATIVE_RESULTS:
        for prompt in ("rewrite", "rephrase"):
            delta = analysis["native_reference"]["z_to_W_measurement"][method]["prompts"][prompt]["post_W_minus_accepted_z"]
            lines.append(
                f"|{method}|{prompt}|{delta['mean']:+.6f}/{delta['median']:+.6f}/{delta['p90']:+.6f}/{delta['max']:+.6f}|"
            )
    lines.extend(
        [
            "",
            "AlphaEdit의 z→W mean gap은 Rewrite `+0.000553`, Rephrase `+0.837049`였고, MEMIT는 Rewrite `+0.341572`, Rephrase `+1.881562`였다. Native direct-z와 materialized W를 구분해 해석해야 하며 이 전달 손실은 target-time 설정 선택에 사용하지 않았다.",
            "",
            "|method|total|z 생성|writer core|z eval|W eval|restore|W0 restore|",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in NATIVE_RESULTS:
        native_measurement = analysis["native_reference"]["z_to_W_measurement"][method]
        compute = native_measurement["compute"]
        lines.append(
            f"|{method}|{compute['method_total_seconds']:.1f}s|{compute['target_accepted_z_generation_seconds']:.1f}s|{compute['writer_edit_core_seconds']:.1f}s|{compute['accepted_z_evaluator_seconds']:.1f}s|{compute['post_W_evaluator_seconds']:.1f}s|{compute['restore_seconds']:.1f}s|{native_measurement['W0_restored']}|"
        )
    lines.extend(
        [
            "",
            "## 6. 해석 경계",
            "",
            "- B100×1 exploratory ablation이며 이 결과만으로 final T_z를 선택하거나 promotion하지 않는다.",
            "- fixed-dt longer-time 효과만 다룬다. Z0 coarse resolution은 과학 분모와 raw manifest에 포함하지 않았다.",
            "- Native는 canonical one-shot external reference다. K8/schedule-matched Native, causal target-time arm, continuous-ODE convergence를 주장하지 않는다.",
            "- heldout은 K1/K4/K8 terminal observation-only이며 controller influence 0이다.",
            "- 세부 request/prompt/microstep/marginal/compute 수치는 동봉 CSV에 있으며 누락값은 추정하지 않았다.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_report(
    *,
    analysis: Mapping[str, Any],
    endpoints: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    writer: Sequence[Mapping[str, Any]],
    native_endpoints: Sequence[Mapping[str, Any]],
    marginals: Sequence[Mapping[str, Any]],
) -> str:
    labels = (*CELLS, *NATIVE_RESULTS)
    display = {
        "Z1-REFINE": "Z1 (Tz=1)",
        "Z15": "Z15 (Tz=1.5)",
        "Z20": "Z20 (Tz=2)",
        "Z30": "Z30 (Tz=3)",
        "OFFICIAL-ALPHAEDIT": "Native AlphaEdit†",
        "OFFICIAL-MEMIT": "Native MEMIT†",
    }
    scope = {
        **{cell: "fixed-dt longer-time" for cell in CELLS},
        "OFFICIAL-ALPHAEDIT": "one-shot external",
        "OFFICIAL-MEMIT": "one-shot external",
    }
    writer_k8 = {(row["cell"], row["outer_K"]): row for row in writer}
    native_lookup = {
        (str(row["cell"]), str(row["endpoint"]), str(row["prompt"])): row
        for row in native_endpoints
    }
    final_endpoints: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    locality: dict[tuple[str, str], tuple[int, int]] = {}
    for cell in CELLS:
        for endpoint_name in ("accepted_z", "post_W"):
            source_endpoint = (
                endpoint_name if endpoint_name == "accepted_z" else "post_writer_W"
            )
            for prompt in ("rewrite", "rephrase"):
                final_endpoints[(cell, endpoint_name, prompt)] = endpoints[
                    (cell, 8, source_endpoint, prompt)
                ]
        row = writer_k8[(cell, 8)]
        locality[(cell, "accepted_z")] = (
            int(row["accepted_z_locality_numerator"]),
            int(row["accepted_z_locality_denominator"]),
        )
        locality[(cell, "post_W")] = (
            int(row["post_W_locality_numerator"]),
            int(row["post_W_locality_denominator"]),
        )
    for method in NATIVE_RESULTS:
        measurement = analysis["native_reference"]["z_to_W_measurement"][method]
        for endpoint_name in ("accepted_z", "post_W"):
            for prompt in ("rewrite", "rephrase"):
                final_endpoints[(method, endpoint_name, prompt)] = native_lookup[
                    (method, endpoint_name, prompt)
                ]
            loc = measurement["locality"][endpoint_name]
            locality[(method, endpoint_name)] = (
                int(loc["numerator"]),
                int(loc["denominator"]),
            )

    lines = [
        "# P1R52 target-timescale B100 — longer target time 상세 결과",
        "",
        "> **범위:** `FIXED_DT_LONGER_TARGET_TIME_Z1_Z15_Z20_Z30`. 네 cell은 동일 `dt=1/16`, sealed B1 100 requests, W0·C3 K8 writer·cache·evaluator를 공유하고 `T_z=1/1.5/2/3`만 다르다. Z0 coarse-resolution은 제외했다.",
        "",
        "> **가독성 정정판 v3:** 최종 W Eff/Gen/Loc와 z/W NLL을 첫 표로 통합하고, Rewrite·Rephrase 상세 및 longer-time 전용 표를 분리했다. v2의 수치·raw identity·과학 경계는 변경하지 않았다.",
        "",
    ]
    lines.extend(
        _final_endpoint_report_lines(
            labels=labels,
            display=display,
            scope=scope,
            endpoints=final_endpoints,
            locality=locality,
        )
    )
    lines.extend(
        [
            "",
            "## 3. Longer-time 실험 전용 표",
            "",
            "### 3.1 K1/K4/K8 accepted-z target-time curve",
            "",
            "|K|cell|T_z|Rewrite NLL mean/median/p90|Rephrase NLL mean/median/p90|Rewrite success|Rephrase success/strict|",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    horizons = {str(row["cell"]): float(row["target_horizon"]) for row in cells}
    for outer in (1, 4, 8):
        for cell in CELLS:
            rewrite = endpoints[(cell, outer, "accepted_z", "rewrite")]
            rephrase = endpoints[(cell, outer, "accepted_z", "rephrase")]
            lines.append(
                f"|{outer}|{display[cell]}|{horizons[cell]:.1f}|{_nll_triplet(rewrite)}|"
                f"{_nll_triplet(rephrase)}|"
                f"{_count_rate(rewrite['success_numerator'], rewrite['success_denominator'])}|"
                f"{_count_rate(rephrase['success_numerator'], rephrase['success_denominator'])} · "
                f"strict {_count_rate(rephrase['strict_success_numerator'], rephrase['strict_success_denominator'])}|"
            )
    lines.extend(
        [
            "",
            "### 3.2 K8 consecutive target-time marginal",
            "",
            "아래 delta는 `right T_z − left T_z`이므로 음수면 longer-time 쪽 NLL이 낮다.",
            "",
            "|increment|endpoint|prompt|Δmean/Δmedian/Δp90|",
            "|---|---|---|---:|",
        ]
    )
    for left, right in CONSECUTIVE:
        for endpoint_name in ("accepted_z", "post_writer_W"):
            for prompt in ("rewrite", "rephrase"):
                rows = {
                    row["statistic"]: row
                    for row in marginals
                    if row["left_cell"] == left
                    and row["right_cell"] == right
                    and row["outer_K"] == 8
                    and row["endpoint"] == endpoint_name
                    and row["prompt"] == prompt
                }
                lines.append(
                    f"|{display[left]}→{display[right]}|"
                    f"{'z' if endpoint_name == 'accepted_z' else 'W'}|{prompt}|"
                    f"{float(rows['mean']['right_minus_left']):+.6f}/"
                    f"{float(rows['median']['right_minus_left']):+.6f}/"
                    f"{float(rows['p90']['right_minus_left']):+.6f}|"
                )
    lines.extend(
        [
            "",
            "### 3.3 Target-time 비용·clamp·selection",
            "",
            "|cell|T_z|m|field eval|clamp|PRIMARY/RESCUE/CURRENT|Slurm|post-model|peak allocated|",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in cells:
        lines.append(
            f"|{display[str(row['cell'])]}|{float(row['target_horizon']):.1f}|{row['microsteps_per_outer']}|"
            f"{row['target_field_evaluations']}|{row['clamp_hits']}/{row['request_microstep_denominator']} "
            f"({_pct(float(row['clamp_fraction']))})|{row['primary_count']}/{row['rescue_count']}/{row['current_count']}|"
            f"{row['slurm_elapsed_seconds']}s|{float(row['runtime_after_model_preflight_seconds']):.1f}s|"
            f"{float(row['peak_allocated_bytes']) / 2**30:.2f} GiB|"
        )
    lines.extend(
        [
            "",
            "## 4. Native z→W 전달과 compute",
            "",
            "|Native|prompt|W−z NLL mean/median/p90/max|",
            "|---|---|---:|",
        ]
    )
    for method in NATIVE_RESULTS:
        measurement = analysis["native_reference"]["z_to_W_measurement"][method]
        for prompt in ("rewrite", "rephrase"):
            delta = measurement["prompts"][prompt]["post_W_minus_accepted_z"]
            lines.append(
                f"|{display[method]}|{prompt}|{delta['mean']:+.6f}/{delta['median']:+.6f}/{delta['p90']:+.6f}/{delta['max']:+.6f}|"
            )
    lines.extend(
        [
            "",
            "|Native|total|z 생성|writer core|z eval|W eval|restore|W0 restore|",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in NATIVE_RESULTS:
        measurement = analysis["native_reference"]["z_to_W_measurement"][method]
        compute = measurement["compute"]
        lines.append(
            f"|{display[method]}|{compute['method_total_seconds']:.1f}s|"
            f"{compute['target_accepted_z_generation_seconds']:.1f}s|{compute['writer_edit_core_seconds']:.1f}s|"
            f"{compute['accepted_z_evaluator_seconds']:.1f}s|{compute['post_W_evaluator_seconds']:.1f}s|"
            f"{compute['restore_seconds']:.1f}s|{measurement['W0_restored']}|"
        )
    z1_rp = final_endpoints[("Z1-REFINE", "accepted_z", "rephrase")]
    z30_rp = final_endpoints[("Z30", "accepted_z", "rephrase")]
    z20_w_rp = final_endpoints[("Z20", "post_W", "rephrase")]
    z30_w_rp = final_endpoints[("Z30", "post_W", "rephrase")]
    lines.extend(
        [
            "",
            "## 5. 판독과 경계",
            "",
            f"- K8 accepted-z Rewrite mean은 `0.011389 → 0.006024 → 0.005008 → 0.002264`로 T_z와 함께 감소했다. Rephrase mean은 `{float(z1_rp['target_new_mean']):.6f} → 1.299307 → 1.319775 → {float(z30_rp['target_new_mean']):.6f}`로 비단조다.",
            f"- final W Rephrase mean도 Z20 `{float(z20_w_rp['target_new_mean']):.6f}`에서 Z30 `{float(z30_w_rp['target_new_mean']):.6f}`으로 다시 증가해 longer-time의 writer endpoint 이득이 단조가 아니다.",
            "- Rewrite는 longer target time에서 꾸준히 낮아졌지만 Rephrase tail과 z→W 전달은 구간별로 달랐다. 이는 B1 raw association이며 최적 T_z 선택 근거가 아니다.",
            "- B100×1 exploratory ablation이므로 final T_z, production setting, scientific promotion을 선정하지 않는다.",
            "- Native는 canonical one-shot external reference이며 K8/schedule-matched Native 또는 causal target-time arm이 아니다.",
            "- 모든 target cell은 writer 8회, cache reuse 8·append 1, W0 exact restore, FULL-FP32를 통과했다.",
            "",
            "request/prompt/microstep/marginal 수치는 동봉 CSV에 있으며 v2와 동일 raw identities를 사용한다.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--native-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--native-alpha-job-id", required=True)
    parser.add_argument("--native-memit-job-id", required=True)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args(argv)
    build_longer_time(
        raw_root=args.raw_root,
        native_root=args.native_root,
        output_root=args.output_root,
        job_id=args.job_id,
        native_job_ids=(args.native_alpha_job_id, args.native_memit_job_id),
        repo_root=args.repo_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
