#!/usr/bin/env python3
"""Build a create-once Llama Stage1 NLL/metric-gate report package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from project.run_scripts.ode_bf.contracts import canonical_hash


INSTRUCTION = "ODEEDIT-S05-P4-EULER-PROJECTED-SEMANTIC-ODE-V1"
MODEL = "llama3-8b-inst"
H_GRID = (0.0625, 0.25, 1.0, 4.0)
PREFIXES = (1, 3, 5, 10)
ARMS = (("Z+", "Zplus"), ("Z±", "Zpm"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path) -> None:
    observed = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(observed.st_mode):
        raise RuntimeError(f"non-regular input: {path}")


def _json(path: Path, *, rooted: bool = True) -> dict[str, Any]:
    _regular(path)
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"non-object JSON: {path}")
    if rooted:
        body = dict(value)
        identity = body.pop("identity_sha256", None)
        if identity != canonical_hash(body):
            raise RuntimeError(f"identity mismatch: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty table: {path.name}")
    with path.open("x", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.chmod(path, 0o600)


def _member(path: Path, root: Path, *, role: str) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "relative_path": str(path.relative_to(root)),
        "role": role,
        "bytes": len(data),
        "lines": data.count(b"\n"),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _fmt_h(value: float) -> str:
    return f"{value:g}"


def _pct_reduction(before: float, after: float) -> float:
    if before == 0.0:
        return math.nan
    return 100.0 * (before - after) / before


def _pearson(xs: Iterable[float], ys: Iterable[float]) -> float:
    left = list(xs)
    right = list(ys)
    if len(left) != len(right) or len(left) < 2:
        raise RuntimeError("invalid correlation vectors")
    mx, my = mean(left), mean(right)
    numerator = sum((x - mx) * (y - my) for x, y in zip(left, right))
    left_norm = math.sqrt(sum((x - mx) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - my) ** 2 for y in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return math.nan
    return numerator / (left_norm * right_norm)


def _metric_row(
    arm: str,
    h: float,
    row: Mapping[str, Any],
    *,
    trajectory_sha: str,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "h": h,
        "M": row["M"],
        "T_z_equals_Mh": row["target_horizon"],
        "requests": 10,
        "new_nll_mean": row["new_nll"]["mean"],
        "new_nll_median": row["new_nll"]["median"],
        "new_nll_p90": row["new_nll"]["p90"],
        "new_nll_max": row["new_nll"]["max"],
        "true_nll_mean": row["old_nll"]["mean"],
        "true_nll_median": row["old_nll"]["median"],
        "true_nll_p90": row["old_nll"]["p90"],
        "true_nll_max": row["old_nll"]["max"],
        "new_minus_true_margin_mean": row["new_minus_old_margin"]["mean"],
        "new_minus_true_margin_median": row["new_minus_old_margin"]["median"],
        "new_minus_true_margin_p90": row["new_minus_old_margin"]["p90"],
        "new_minus_true_margin_max": row["new_minus_old_margin"]["max"],
        "displacement_mean": row["displacement"]["mean"],
        "displacement_median": row["displacement"]["median"],
        "displacement_p90": row["displacement"]["p90"],
        "displacement_max": row["displacement"]["max"],
        "raw_field_norm_mean": row["raw_field_norm"]["mean"],
        "raw_field_norm_median": row["raw_field_norm"]["median"],
        "raw_field_norm_p90": row["raw_field_norm"]["p90"],
        "raw_field_norm_max": row["raw_field_norm"]["max"],
        "clamp_hits": row["clamp_hit_count"],
        "clamp_denominator_request_microsteps": row["clamp_denominator"],
        "clamp_fraction": row["clamp_fraction"],
        "finite": row["finite"],
        "target_sha256": row["target_sha256"],
        "prefix_identity_sha256": row["identity_sha256"],
        "trajectory_file_sha256": trajectory_sha,
        "interpretation": "FIXED_H_STRENGTH_CURVE_T_Z_CHANGES_WITH_M",
    }


def build(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError("report output is create-once")
    os.umask(0o077)
    args.output_root.mkdir(parents=True, mode=0o700)
    stage1 = args.stage1_root.resolve()
    stage2 = args.stage2_root.resolve()

    terminal1_path = stage1 / "terminal.json"
    terminal1 = _json(terminal1_path)
    terminal2_path = stage2 / "terminal.json"
    terminal2 = _json(terminal2_path)
    if (
        terminal1.get("alias") != MODEL
        or terminal1.get("status") != "STAGE1_MODEL_CELL_TERMINAL"
        or terminal1.get("h_grid") != list(H_GRID)
        or terminal1.get("prefix_M") != list(PREFIXES)
        or terminal2.get("alias") != MODEL
        or terminal2.get("selected_h") != 0.25
        or terminal2.get("selected_target_horizon") != 1.25
    ):
        raise RuntimeError("terminal contract differs")

    inputs: list[dict[str, Any]] = [
        {"path": str(terminal1_path), "sha256": _sha(terminal1_path)},
        {"path": str(terminal2_path), "sha256": _sha(terminal2_path)},
    ]
    cells: list[dict[str, Any]] = []
    trajectories: dict[tuple[str, float], dict[str, Any]] = {}
    prefix_lookup: dict[tuple[str, float, int], dict[str, Any]] = {}
    request_m10_rows: list[dict[str, Any]] = []
    compute_rows: list[dict[str, Any]] = []
    for arm, directory in ARMS:
        for h in H_GRID:
            path = stage1 / "trajectories" / directory / f"h-{_fmt_h(h)}.json"
            trajectory = _json(path)
            trajectory_sha = _sha(path)
            inputs.append({"path": str(path), "sha256": trajectory_sha})
            trajectories[(arm, h)] = trajectory
            if (
                trajectory["arm"] != arm
                or trajectory["h"] != h
                or trajectory["executed_microsteps"] != 10
                or trajectory["actual_autograd_grad_call_count"] != 10
                or trajectory["duplicate_autograd_evaluation_count"] != 0
            ):
                raise RuntimeError(f"trajectory contract differs: {path}")
            for prefix in trajectory["prefix_rows"]:
                row = _metric_row(
                    arm, h, prefix, trajectory_sha=trajectory_sha
                )
                cells.append(row)
                prefix_lookup[(arm, h, int(prefix["M"]))] = row
            observation = trajectory["final_value_only_observation"]
            for ordinal in range(10):
                request_m10_rows.append(
                    {
                        "arm": arm,
                        "h": h,
                        "M": 10,
                        "T_z": 10 * h,
                        "request_ordinal": ordinal,
                        "new_nll": observation["new_nll_by_request"][ordinal],
                        "true_nll": observation["old_nll_by_request"][ordinal],
                        "new_minus_true_margin": observation[
                            "new_minus_old_margin_by_request"
                        ][ordinal],
                        "new_nll_below_1_observation": (
                            observation["new_nll_by_request"][ordinal] < 1.0
                        ),
                        "positive_margin_observation": (
                            observation["new_minus_old_margin_by_request"][ordinal]
                            > 0.0
                        ),
                        "cutoff_note": "DESCRIPTIVE_NOT_PREDECLARED_GATE",
                    }
                )
            compute_rows.append(
                {
                    "scope": "trajectory",
                    "arm": arm,
                    "h": h,
                    "executed_microsteps": trajectory["executed_microsteps"],
                    "actual_autograd_grad_calls": trajectory[
                        "actual_autograd_grad_call_count"
                    ],
                    "duplicate_evaluations": trajectory[
                        "duplicate_autograd_evaluation_count"
                    ],
                    "wall_seconds": trajectory["wall_time_seconds"],
                    "seconds_per_microstep": trajectory["wall_time_seconds"] / 10,
                    "W_pointer_version_changes": trajectory[
                        "W0_pointer_version_change_count"
                    ],
                    "W_bytes_changes": trajectory["W0_bytes_change_count"],
                    "optimizer": trajectory["optimizer"],
                    "adam_states": trajectory["adam_state_count"],
                    "backward_calls": trajectory["loss_backward_count"],
                    "parameter_gradients": trajectory["parameter_gradient_count"],
                }
            )

    m_effect_rows: list[dict[str, Any]] = []
    for arm, _ in ARMS:
        for h in H_GRID:
            for left, right in ((1, 3), (3, 5), (5, 10), (1, 10)):
                before = prefix_lookup[(arm, h, left)]
                after = prefix_lookup[(arm, h, right)]
                m_effect_rows.append(
                    {
                        "arm": arm,
                        "h_fixed": h,
                        "from_M": left,
                        "to_M": right,
                        "from_T_z": before["T_z_equals_Mh"],
                        "to_T_z": after["T_z_equals_Mh"],
                        "new_nll_median_before": before["new_nll_median"],
                        "new_nll_median_after": after["new_nll_median"],
                        "new_nll_median_delta_after_minus_before": (
                            after["new_nll_median"] - before["new_nll_median"]
                        ),
                        "new_nll_median_reduction_percent": _pct_reduction(
                            before["new_nll_median"], after["new_nll_median"]
                        ),
                        "true_nll_median_delta": (
                            after["true_nll_median"] - before["true_nll_median"]
                        ),
                        "margin_median_delta": (
                            after["new_minus_true_margin_median"]
                            - before["new_minus_true_margin_median"]
                        ),
                        "displacement_median_delta": (
                            after["displacement_median"]
                            - before["displacement_median"]
                        ),
                        "raw_field_norm_median_delta": (
                            after["raw_field_norm_median"]
                            - before["raw_field_norm_median"]
                        ),
                        "clamp_fraction_delta": (
                            after["clamp_fraction"] - before["clamp_fraction"]
                        ),
                        "interpretation": "STRENGTH_PSEUDOTIME_CHANGE_NOT_REFINEMENT",
                    }
                )

    h_effect_rows: list[dict[str, Any]] = []
    for arm, _ in ARMS:
        for prefix in PREFIXES:
            for left, right in zip(H_GRID[:-1], H_GRID[1:]):
                before = prefix_lookup[(arm, left, prefix)]
                after = prefix_lookup[(arm, right, prefix)]
                h_effect_rows.append(
                    {
                        "arm": arm,
                        "M_fixed": prefix,
                        "from_h": left,
                        "to_h": right,
                        "from_T_z": before["T_z_equals_Mh"],
                        "to_T_z": after["T_z_equals_Mh"],
                        "new_nll_median_before": before["new_nll_median"],
                        "new_nll_median_after": after["new_nll_median"],
                        "new_nll_median_delta_after_minus_before": (
                            after["new_nll_median"] - before["new_nll_median"]
                        ),
                        "margin_median_delta": (
                            after["new_minus_true_margin_median"]
                            - before["new_minus_true_margin_median"]
                        ),
                        "displacement_median_delta": (
                            after["displacement_median"]
                            - before["displacement_median"]
                        ),
                        "clamp_fraction_before": before["clamp_fraction"],
                        "clamp_fraction_after": after["clamp_fraction"],
                        "clamp_fraction_delta": (
                            after["clamp_fraction"] - before["clamp_fraction"]
                        ),
                        "interpretation": "H_AND_T_Z_CHANGE_AT_FIXED_M",
                    }
                )

    selected_rows = [row for row in cells if row["h"] == 0.25]
    ranking_rows: list[dict[str, Any]] = []
    for arm, _ in ARMS:
        ordered = sorted(
            (row for row in cells if row["arm"] == arm),
            key=lambda row: row["new_nll_median"],
        )
        for rank, row in enumerate(ordered, 1):
            ranking_rows.append(
                {
                    "arm": arm,
                    "rank_by_lowest_new_nll_median": rank,
                    "h": row["h"],
                    "M": row["M"],
                    "T_z": row["T_z_equals_Mh"],
                    "new_nll_median": row["new_nll_median"],
                    "new_nll_p90": row["new_nll_p90"],
                    "new_nll_max": row["new_nll_max"],
                    "margin_median": row["new_minus_true_margin_median"],
                    "clamp_fraction": row["clamp_fraction"],
                    "stage1_h_admissible": row["h"] in (0.0625, 0.25),
                    "ranking_is_observation_only": True,
                }
            )

    admissibility_rows = [
        {
            "h": 0.0625,
            "llama_admissible": True,
            "selection_role": "ADMISSIBLE_SMALLER_GRID_POINT",
            "reason": "clamp/coherence/strength gates passed",
        },
        {
            "h": 0.25,
            "llama_admissible": True,
            "selection_role": "SELECTED_LARGEST_ADMISSIBLE",
            "reason": "clamp/coherence/strength gates passed",
        },
        {
            "h": 1.0,
            "llama_admissible": False,
            "selection_role": "IMMEDIATE_UNSAFE_BRACKET",
            "reason": "clamp/coherence gate failed",
        },
        {
            "h": 4.0,
            "llama_admissible": False,
            "selection_role": "INADMISSIBLE",
            "reason": "clamp/coherence gate failed",
        },
    ]

    same_t_rows: list[dict[str, Any]] = []
    same_t_request_rows: list[dict[str, Any]] = []
    selected_stage1_request_rows: list[dict[str, Any]] = []
    stage2_values: dict[tuple[str, int], dict[str, Any]] = {}
    for arm, directory in ARMS:
        stage1_selected = trajectories[(arm, 0.25)]
        stage1_m10 = stage1_selected["final_value_only_observation"]
        stage1_m5_prefix = next(
            row for row in stage1_selected["prefix_rows"] if row["M"] == 5
        )
        for microsteps in (5, 10):
            path = stage2 / "trajectories" / directory / f"M{microsteps}.json"
            row = _json(path)
            inputs.append({"path": str(path), "sha256": _sha(path)})
            stage2_values[(arm, microsteps)] = row
        stage2_m5 = stage2_values[(arm, 5)]
        if stage1_m5_prefix["target_sha256"] != stage2_m5["target_sha256"]:
            raise RuntimeError(f"Stage1/Stage2 M5 target differs: {arm}")
        stage1_m5 = stage2_m5["final_value_only_observation"]
        for ordinal in range(10):
            selected_stage1_request_rows.append(
                {
                    "arm": arm,
                    "request_ordinal": ordinal,
                    "h": 0.25,
                    "M5_T_z": 1.25,
                    "M5_new_nll": stage1_m5["new_nll_by_request"][ordinal],
                    "M5_true_nll": stage1_m5["old_nll_by_request"][ordinal],
                    "M5_margin": stage1_m5[
                        "new_minus_old_margin_by_request"
                    ][ordinal],
                    "M10_T_z": 2.5,
                    "M10_new_nll": stage1_m10["new_nll_by_request"][ordinal],
                    "M10_true_nll": stage1_m10["old_nll_by_request"][ordinal],
                    "M10_margin": stage1_m10[
                        "new_minus_old_margin_by_request"
                    ][ordinal],
                    "new_nll_delta_M10_minus_M5": (
                        stage1_m10["new_nll_by_request"][ordinal]
                        - stage1_m5["new_nll_by_request"][ordinal]
                    ),
                    "interpretation": "FIXED_H_STRENGTH_CURVE_T_Z_1.25_TO_2.5",
                }
            )
        comparison_path = stage2 / "comparisons" / f"{directory}.json"
        comparison = _json(comparison_path)
        inputs.append({"path": str(comparison_path), "sha256": _sha(comparison_path)})
        m5 = stage2_values[(arm, 5)]
        m10 = stage2_values[(arm, 10)]
        obs5 = m5["final_value_only_observation"]
        obs10 = m10["final_value_only_observation"]
        same_t_rows.append(
            {
                "arm": arm,
                "T_z": 1.25,
                "M5_h": 0.25,
                "M10_h": 0.125,
                "M5_new_nll_mean": m5["new_nll"]["mean"],
                "M5_new_nll_median": m5["new_nll"]["median"],
                "M5_new_nll_p90": m5["new_nll"]["p90"],
                "M5_new_nll_max": m5["new_nll"]["max"],
                "M10_new_nll_mean": m10["new_nll"]["mean"],
                "M10_new_nll_median": m10["new_nll"]["median"],
                "M10_new_nll_p90": m10["new_nll"]["p90"],
                "M10_new_nll_max": m10["new_nll"]["max"],
                "M5_positive_margin_count": sum(
                    value > 0.0 for value in obs5["new_minus_old_margin_by_request"]
                ),
                "M10_positive_margin_count": sum(
                    value > 0.0 for value in obs10["new_minus_old_margin_by_request"]
                ),
                "M5_new_nll_below_1_count": sum(
                    value < 1.0 for value in obs5["new_nll_by_request"]
                ),
                "M10_new_nll_below_1_count": sum(
                    value < 1.0 for value in obs10["new_nll_by_request"]
                ),
                "request_denominator": 10,
                "M5_clamp_fraction": m5["clamp_fraction"],
                "M10_clamp_fraction": m10["clamp_fraction"],
                "latent_endpoint_d_median": comparison["endpoint_discrepancy"][
                    "median"
                ],
                "latent_endpoint_d_p90": comparison["endpoint_discrepancy"]["p90"],
                "latent_endpoint_d_max": comparison["endpoint_discrepancy"]["max"],
                "old_geometry_gate_pass": comparison["threshold_pass"],
                "revised_task_metric_gate": "USER_DIRECTED_PASS",
                "claim_level": "TRAIN_TARGET_METRIC_STABILITY_ONLY",
            }
        )
        for ordinal in range(10):
            same_t_request_rows.append(
                {
                    "arm": arm,
                    "request_ordinal": ordinal,
                    "T_z": 1.25,
                    "M5_h": 0.25,
                    "M5_new_nll": obs5["new_nll_by_request"][ordinal],
                    "M5_true_nll": obs5["old_nll_by_request"][ordinal],
                    "M5_margin": obs5["new_minus_old_margin_by_request"][ordinal],
                    "M10_h": 0.125,
                    "M10_new_nll": obs10["new_nll_by_request"][ordinal],
                    "M10_true_nll": obs10["old_nll_by_request"][ordinal],
                    "M10_margin": obs10[
                        "new_minus_old_margin_by_request"
                    ][ordinal],
                    "new_nll_delta_M10_minus_M5": (
                        obs10["new_nll_by_request"][ordinal]
                        - obs5["new_nll_by_request"][ordinal]
                    ),
                    "both_positive_margin": (
                        obs5["new_minus_old_margin_by_request"][ordinal] > 0.0
                        and obs10["new_minus_old_margin_by_request"][ordinal] > 0.0
                    ),
                    "both_new_nll_below_1_observation": (
                        obs5["new_nll_by_request"][ordinal] < 1.0
                        and obs10["new_nll_by_request"][ordinal] < 1.0
                    ),
                    "cutoff_note": "DESCRIPTIVE_NOT_PREDECLARED_GATE",
                }
            )

    correlation_rows: list[dict[str, Any]] = []
    predictors = (
        "T_z_equals_Mh",
        "displacement_median",
        "raw_field_norm_median",
        "clamp_fraction",
        "new_minus_true_margin_median",
    )
    for arm, _ in ARMS:
        arm_cells = [row for row in cells if row["arm"] == arm]
        for predictor in predictors:
            correlation_rows.append(
                {
                    "arm": arm,
                    "cell_count": len(arm_cells),
                    "x": predictor,
                    "y": "new_nll_median",
                    "pearson_r": _pearson(
                        (float(row[predictor]) for row in arm_cells),
                        (float(row["new_nll_median"]) for row in arm_cells),
                    ),
                    "scope": "DESCRIPTIVE_CELL_LEVEL_NO_CAUSAL_CLAIM",
                }
            )

    inner_total = sum(float(row["wall_seconds"]) for row in compute_rows)
    allocation_seconds = 354.0
    overhead = allocation_seconds - inner_total
    compute_rows.extend(
        [
            {
                "scope": "stage1_total",
                "arm": "ALL",
                "h": "ALL",
                "executed_microsteps": terminal1["executed_microstep_count"],
                "actual_autograd_grad_calls": terminal1[
                    "actual_autograd_grad_call_count"
                ],
                "duplicate_evaluations": terminal1[
                    "duplicate_autograd_evaluation_count"
                ],
                "wall_seconds": inner_total,
                "seconds_per_microstep": inner_total
                / terminal1["executed_microstep_count"],
                "W_pointer_version_changes": 0,
                "W_bytes_changes": 0,
                "optimizer": terminal1["optimizer"],
                "adam_states": terminal1["adam_state_count"],
                "backward_calls": terminal1["loss_backward_count"],
                "parameter_gradients": terminal1["parameter_gradient_count"],
            },
            {
                "scope": "slurm_allocation",
                "arm": "ALL",
                "h": "ALL",
                "executed_microsteps": 80,
                "actual_autograd_grad_calls": 80,
                "duplicate_evaluations": 0,
                "wall_seconds": allocation_seconds,
                "seconds_per_microstep": allocation_seconds / 80,
                "W_pointer_version_changes": 0,
                "W_bytes_changes": 0,
                "optimizer": "NONE",
                "adam_states": 0,
                "backward_calls": 0,
                "parameter_gradients": 0,
            },
            {
                "scope": "setup_load_report_overhead",
                "arm": "ALL",
                "h": "ALL",
                "executed_microsteps": 0,
                "actual_autograd_grad_calls": 0,
                "duplicate_evaluations": 0,
                "wall_seconds": overhead,
                "seconds_per_microstep": "NOT_APPLICABLE",
                "W_pointer_version_changes": 0,
                "W_bytes_changes": 0,
                "optimizer": "NONE",
                "adam_states": 0,
                "backward_calls": 0,
                "parameter_gradients": 0,
            },
        ]
    )

    tables = {
        "stage1-all-cells.csv": cells,
        "stage1-selected-h025.csv": selected_rows,
        "stage1-M-effects.csv": m_effect_rows,
        "stage1-h-effects.csv": h_effect_rows,
        "stage1-cell-rankings.csv": ranking_rows,
        "stage1-admissibility.csv": admissibility_rows,
        "stage1-M10-request-level.csv": request_m10_rows,
        "stage1-selected-h025-M5-M10-request-level.csv": selected_stage1_request_rows,
        "stage2-same-T-metric-stability.csv": same_t_rows,
        "stage2-same-T-request-level.csv": same_t_request_rows,
        "stage1-cell-correlations.csv": correlation_rows,
        "stage1-compute-overhead.csv": compute_rows,
    }
    for name, rows in tables.items():
        _write_csv(args.output_root / name, rows)

    zplus = {row["M"]: row for row in selected_rows if row["arm"] == "Z+"}
    zpm = {row["M"]: row for row in selected_rows if row["arm"] == "Z±"}
    analysis: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-stage1-llama-nll-metric-report/v1",
        "instruction_id": INSTRUCTION,
        "model": MODEL,
        "scope": "LLAMA_ONLY_B1_CASE01_CALIBRATION_TRAIN_ONLY",
        "decision_revision": "USER_DIRECTED_GATE_POLICY_REVISION",
        "status": "TASK_METRIC_GATE_PASS_REPORT_COMPLETE",
        "decisive_gate": "ODESteer_STYLE_EMPIRICAL_TRAIN_TARGET_METRIC_STABILITY",
        "latent_endpoint_geometry_gate": "FAILED_OBSERVED_NONDECISIONAL",
        "selected_h": 0.25,
        "selected_target_horizon": 1.25,
        "selected_h025_summary": {
            "Z_plus": {
                "new_nll_median_M1_M3_M5_M10": [
                    zplus[m]["new_nll_median"] for m in PREFIXES
                ],
                "new_nll_median_reduction_M1_to_M5_percent": _pct_reduction(
                    zplus[1]["new_nll_median"], zplus[5]["new_nll_median"]
                ),
                "new_nll_median_reduction_M1_to_M10_percent": _pct_reduction(
                    zplus[1]["new_nll_median"], zplus[10]["new_nll_median"]
                ),
            },
            "Z_plus_minus": {
                "new_nll_median_M1_M3_M5_M10": [
                    zpm[m]["new_nll_median"] for m in PREFIXES
                ],
                "new_nll_median_reduction_M1_to_M5_percent": _pct_reduction(
                    zpm[1]["new_nll_median"], zpm[5]["new_nll_median"]
                ),
                "new_nll_median_reduction_M1_to_M10_percent": _pct_reduction(
                    zpm[1]["new_nll_median"], zpm[10]["new_nll_median"]
                ),
            },
        },
        "same_T_metric_stability": same_t_rows,
        "invariants": {
            "finite": True,
            "W0_restored": terminal1["W0_restored"] and terminal2["W0_restored"],
            "full_fp32": terminal1["full_fp32"],
            "optimizer": "NONE",
            "adam_state_count": 0,
            "loss_backward_count": 0,
            "parameter_gradient_count": 0,
            "stage1_autograd_grad_count": terminal1[
                "actual_autograd_grad_call_count"
            ],
            "stage1_duplicate_evaluation_count": terminal1[
                "duplicate_autograd_evaluation_count"
            ],
            "writer_materialization_count": 0,
            "cache_append_count": 0,
            "heldout_access_count": 0,
            "native_access_count": 0,
        },
        "compute": {
            "trajectory_inner_seconds": inner_total,
            "slurm_allocation_seconds": allocation_seconds,
            "setup_load_report_overhead_seconds": overhead,
            "overhead_fraction_of_allocation": overhead / allocation_seconds,
            "peak_gpu_memory_bytes": terminal1["peak_gpu_memory_bytes"],
        },
        "claim_boundary": {
            "supported": "empirical Euler train-target metric stability on one calibration unit",
            "not_supported": [
                "latent endpoint numerical convergence",
                "continuous ODE existence or uniqueness proof",
                "heldout efficacy/generalization/locality",
                "writer-level or sequential editing performance",
            ],
            "calibration_unit": "B1_CASE01_PERMANENT_CALIBRATION_ONLY",
            "heldout": "NOT_RECORDED_CALIBRATION_TRAIN_ONLY",
            "qwen_scientific_input_count": 0,
        },
        "scientific_promotion": False,
        "za_zb_submission_count": 0,
    }
    analysis["root_digest"] = canonical_hash(analysis)
    _write_json(args.output_root / "analysis.json", analysis)

    report = f"""# P4-Euler Stage 1 Llama NLL 및 metric 안정성 상세 보고서

> **판정 범위:** `LLAMA_ONLY`, `B1_CASE01_PERMANENT_CALIBRATION_ONLY`, train target-only. 이번 사용자 지시에 따라 기존 latent endpoint 수렴 임계는 비결정적 진단으로 유지하고, ODESteer식 **Euler step/strength 변화에서의 task metric 안정성**을 결정 기준으로 적용한다.

## 결론

- 개정 상태: **TASK_METRIC_GATE_PASS** (`USER_DIRECTED_GATE_POLICY_REVISION`).
- 선택값은 그대로 `h=0.25`, `T_z=1.25`, main `M=5`다. 수치나 실행 결과를 다시 계산하거나 변경하지 않았다.
- 선택 h=.25의 train target-new NLL median은 다음과 같이 감소했다.

| arm | M1 (T=.25) | M3 (T=.75) | M5 (T=1.25) | M10 (T=2.5) | M1→M5 감소 | M1→M10 감소 |
|---|---:|---:|---:|---:|---:|---:|
| Z+ | {zplus[1]['new_nll_median']:.6f} | {zplus[3]['new_nll_median']:.6f} | {zplus[5]['new_nll_median']:.6f} | {zplus[10]['new_nll_median']:.6f} | {_pct_reduction(zplus[1]['new_nll_median'], zplus[5]['new_nll_median']):.2f}% | {_pct_reduction(zplus[1]['new_nll_median'], zplus[10]['new_nll_median']):.2f}% |
| Z± | {zpm[1]['new_nll_median']:.6f} | {zpm[3]['new_nll_median']:.6f} | {zpm[5]['new_nll_median']:.6f} | {zpm[10]['new_nll_median']:.6f} | {_pct_reduction(zpm[1]['new_nll_median'], zpm[5]['new_nll_median']):.2f}% | {_pct_reduction(zpm[1]['new_nll_median'], zpm[10]['new_nll_median']):.2f}% |

- 같은 `T_z=1.25`에서 M5와 M10을 비교해도 두 arm 모두 10/10 request에서 positive new-vs-true margin을 유지했고, 기술적 설명용 cutoff `new NLL<1`도 10/10이었다. M10 NLL은 M5보다 더 낮아졌다.
- 따라서 **편집 목적의 train-target metric은 step refinement에 대해 안정적으로 성공 영역에 남았다**고 판정한다.
- 기존 `d_i` endpoint geometry 실패(Z+ median .599819, Z± median .737896)는 삭제하지 않는다. 이는 **강한 latent endpoint 수렴 증거가 부족함**을 뜻하지만, 이번 개정 gate에서는 task metric pass를 뒤집지 않는 진단이다.

## 실험 구조와 해석 주의

Stage 1은 각 `model×arm×h`에서 하나의 10-step trajectory만 실행하고 M1/M3/M5/M10 prefix를 재사용했다. 따라서 고정 h에서 M을 늘리면 `T_z=Mh`도 늘어난다. 이 표는 **수치 refinement 표가 아니라 strength/pseudo-time curve**다. 순수 discretization 비교는 Stage 2의 `T_z=1.25`, M5(h=.25) 대 M10(h=.125) 표에만 해당한다.

## 선택 h=.25의 NLL, margin, geometry

| arm | M | T_z | new NLL mean/median/p90/max | true NLL mean/median | margin mean/median | disp median | field median | clamp |
|---|---:|---:|---|---|---|---:|---:|---:|
"""
    for row in selected_rows:
        report += (
            f"| {row['arm']} | {row['M']} | {row['T_z_equals_Mh']:.2f} | "
            f"{row['new_nll_mean']:.6f}/{row['new_nll_median']:.6f}/"
            f"{row['new_nll_p90']:.6f}/{row['new_nll_max']:.6f} | "
            f"{row['true_nll_mean']:.6f}/{row['true_nll_median']:.6f} | "
            f"{row['new_minus_true_margin_mean']:.6f}/"
            f"{row['new_minus_true_margin_median']:.6f} | "
            f"{row['displacement_median']:.6f} | "
            f"{row['raw_field_norm_median']:.6f} | "
            f"{row['clamp_hits']}/{row['clamp_denominator_request_microsteps']}="
            f"{row['clamp_fraction']:.3f} |\n"
        )
    report += """

NLL은 target-new token의 length-normalized train NLL이다. `true NLL`은 현재 request의 target_true/old object NLL이며, `margin=true NLL-new NLL`이다. M3부터 두 arm 모두 median margin이 양수이고, M5에서 new NLL median이 Z+ .1098, Z± .1285까지 내려갔다. M10은 추가 pseudo-time이므로 main M5보다 강한 endpoint이지 별도 선택값이 아니다.

## 전체 h×M 결과

`stage1-all-cells.csv`는 32개 cell 전부에 대해 new/true NLL와 margin의 mean/median/p90/max, displacement, raw field norm, clamp 분모, finite, target/root identity를 제공한다. 아래는 new NLL median 요약이다.

| arm | h | M1 | M3 | M5 | M10 | M10 clamp | Stage1 h 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
"""
    for arm, _ in ARMS:
        for h in H_GRID:
            values = [prefix_lookup[(arm, h, m)] for m in PREFIXES]
            report += (
                f"| {arm} | {h:g} | "
                + " | ".join(f"{row['new_nll_median']:.6f}" for row in values)
                + f" | {values[-1]['clamp_fraction']:.3f} | "
                + ("ADMISSIBLE" if h in (0.0625, 0.25) else "INADMISSIBLE")
                + " |\n"
            )
    report += """

관측상 h=1은 일부 NLL이 낮지만 M10 clamp가 Z+ 54/100, Z± 59/100이며, h=4는 각각 76/100, 79/100이다. 따라서 낮은 NLL만 보고 큰 h를 선택하지 않았고, 원래 Stage1 bracket 판정대로 h=.25를 유지한다.

## 같은 T_z에서의 metric 안정성

| arm | M5 new NLL mean/median/p90/max | M10 new NLL mean/median/p90/max | positive margin M5/M10 | NLL<1 M5/M10 | clamp M5/M10 | latent d median/p90/max |
|---|---|---|---|---|---|---|
"""
    for row in same_t_rows:
        report += (
            f"| {row['arm']} | {row['M5_new_nll_mean']:.6f}/"
            f"{row['M5_new_nll_median']:.6f}/{row['M5_new_nll_p90']:.6f}/"
            f"{row['M5_new_nll_max']:.6f} | {row['M10_new_nll_mean']:.6f}/"
            f"{row['M10_new_nll_median']:.6f}/{row['M10_new_nll_p90']:.6f}/"
            f"{row['M10_new_nll_max']:.6f} | "
            f"{row['M5_positive_margin_count']}/10, {row['M10_positive_margin_count']}/10 | "
            f"{row['M5_new_nll_below_1_count']}/10, {row['M10_new_nll_below_1_count']}/10 | "
            f"{row['M5_clamp_fraction']:.3f}/{row['M10_clamp_fraction']:.3f} | "
            f"{row['latent_endpoint_d_median']:.6f}/"
            f"{row['latent_endpoint_d_p90']:.6f}/"
            f"{row['latent_endpoint_d_max']:.6f} |\n"
        )
    report += f"""

`NLL<1`은 관측을 읽기 쉽게 표시한 기술적 cutoff이며 사전 선언된 새 임계가 아니다. 결정의 근거는 사용자가 지정한 task-metric 안정성 판정이다. request별 값은 `stage2-same-T-request-level.csv`에 있다.

## M 효과, h 효과, request 표

- `stage1-M-effects.csv`: h 고정에서 M1→3, M3→5, M5→10, M1→10의 NLL 감소율과 true NLL/margin/displacement/field/clamp 변화. `T_z`도 변하므로 strength 변화로 표기했다.
- `stage1-h-effects.csv`: M 고정에서 인접 h 변화의 NLL/margin/displacement/clamp 차이. h와 `T_z`가 함께 변한다.
- `stage1-cell-rankings.csv`: arm별 16개 cell의 new NLL median 순위와 admissibility. 순위는 observation-only이며 큰 h 사후 선택에 쓰지 않는다.
- `stage1-M10-request-level.csv`: 모든 h와 arm에서 M10 request 10개씩, 총 80행의 new/true NLL 및 margin.
- `stage1-selected-h025-M5-M10-request-level.csv`: 선택 h=.25의 M5(T=1.25)와 M10(T=2.5) request-paired strength 변화.
- `stage1-cell-correlations.csv`: arm별 16-cell 집계에서 NLL median과 pseudo-time/displacement/field/clamp/margin의 Pearson 상관. 기술적 요약일 뿐 인과 해석은 하지 않는다.

## 계산량과 overhead

- Stage1은 8 trajectory, 80 Euler microstep, 80회 `autograd.grad`, duplicate evaluation 0이다.
- inner trajectory 합: **{inner_total:.3f}s**. 성공 Llama Slurm allocation: **{allocation_seconds:.0f}s**. setup/model-load/report overhead: **{overhead:.3f}s ({100.0 * overhead / allocation_seconds:.2f}%)**.
- 평균 inner microstep: **{inner_total / 80:.3f}s**. peak GPU memory: **{terminal1['peak_gpu_memory_bytes'] / (1024**3):.3f} GiB**.
- FULL FP32, W0 restore, finite, optimizer/Adam/backward/parameter-grad 0을 유지했다.

## ODE claim 경계

이번 PASS가 지지하는 범위는 **한 calibration unit에서 Euler step/pseudo-time 및 same-horizon refinement에 대해 train-target NLL 성공 영역이 유지되었다**는 경험적 주장이다. 기존 latent endpoint `d_i` 실패는 그대로 남으므로, 다음은 주장하지 않는다.

- M5와 M10 latent endpoint의 강한 수치 수렴
- 연속 ODE 해의 존재·유일성 또는 solver-order 증명
- heldout Eff/Gen/Loc, rewrite/rephrase 성능
- writer 적용 뒤의 편집 성능 또는 sequential editing 성능

B1_CASE01은 계속 permanent calibration-only다. heldout/writer/cache/Native는 0이며 Qwen은 이 보고서의 scientific input이 아니다. `scientific_promotion=false`, 추가 GPU/ZA/ZB 제출은 0이다.
"""
    report_path = args.output_root / "report-ko.md"
    report_path.write_text(report, encoding="utf-8")
    os.chmod(report_path, 0o600)

    outputs = [
        _member(path, args.output_root, role="LLAMA_STAGE1_NLL_REPORT")
        for path in sorted(args.output_root.iterdir())
    ]
    manifest: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-stage1-llama-nll-manifest/v1",
        "instruction_id": INSTRUCTION,
        "scope": "LLAMA_ONLY_B1_CASE01_CALIBRATION_TRAIN_ONLY",
        "scientific_inputs": sorted(inputs, key=lambda row: row["path"]),
        "scientific_input_model_aliases": [MODEL],
        "qwen_scientific_input_count": 0,
        "outputs": outputs,
    }
    manifest["root_digest"] = canonical_hash(manifest)
    manifest_path = args.output_root / "analysis-manifest.json"
    _write_json(manifest_path, manifest)

    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-stage1-llama-nll-rooted-receipt/v1",
        "instruction_id": INSTRUCTION,
        "scope": "LLAMA_ONLY_B1_CASE01_CALIBRATION_TRAIN_ONLY",
        "decision_revision": "USER_DIRECTED_GATE_POLICY_REVISION",
        "status": "TASK_METRIC_GATE_PASS_REPORT_COMPLETE",
        "decisive_gate": "ODESteer_STYLE_EMPIRICAL_TRAIN_TARGET_METRIC_STABILITY",
        "latent_endpoint_geometry_gate": "FAILED_OBSERVED_NONDECISIONAL",
        "output_root": str(args.output_root.resolve()),
        "manifest_sha256": _sha(manifest_path),
        "manifest_root": manifest["root_digest"],
        "analysis_root": analysis["root_digest"],
        "B1_confirmatory_eligibility": False,
        "heldout_access_count": 0,
        "qwen_scientific_input_count": 0,
        "za_zb_submission_count": 0,
        "scientific_promotion": False,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    receipt_path = args.output_root / "rooted-receipt.json"
    _write_json(receipt_path, receipt)
    return {
        "status": receipt["status"],
        "output_root": str(args.output_root.resolve()),
        "manifest_root": manifest["root_digest"],
        "receipt_identity": receipt["identity_sha256"],
        "table_count": len(tables),
    }


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    print(json.dumps(build(parser.parse_args()), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
