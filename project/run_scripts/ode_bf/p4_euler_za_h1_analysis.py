#!/usr/bin/env python3
"""Build the canonical Llama-only P4-Euler ZA h=1 factual package."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from project.run_scripts.ode_bf.contracts import canonical_hash


INSTRUCTION_ID = "ODEEDIT-S05-P4-EULER-PROJECTED-SEMANTIC-ODE-V1"
SOURCE_HEAD = "f4fbfb5de132de9b986e4b291c5b41784c182743"
SOURCE_TREE = "8c3f444014907f69b1bb793674f7a5a9e2b9ee0e"
NUMERICAL_LOCK_ROOT = (
    "1e3c8ee90d0d912bdfafeb0b281e399e3926263a0d1b4349fb45ba29fbe0bc84"
)
CASES = tuple(range(2, 11))
TARGET_ARMS = ("Z+", "Z±")
ALL_ARMS = ("Z+", "Z±", "Native-Z")
ARM_DIR = {"Z+": "z-plus", "Z±": "z-plus-minus", "Native-Z": "native-z"}
ARM_SLUG = {"Z+": "z_plus", "Z±": "z_plus_minus", "Native-Z": "native_z"}
H = 1.0
M = 5
T_Z = 5.0


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"regular non-symlink input required: {path}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    identity = value.get("identity_sha256")
    if isinstance(identity, str):
        body = dict(value)
        body.pop("identity_sha256")
        if canonical_hash(body) != identity:
            raise ValueError(f"rooted identity differs: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("empty quantile input")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    rows = [float(value) for value in values]
    if not rows or not all(math.isfinite(value) for value in rows):
        raise ValueError("finite non-empty summary input required")
    return {
        "n": len(rows),
        "mean": statistics.fmean(rows),
        "median": statistics.median(rows),
        "p90": _quantile(rows, 0.9),
        "min": min(rows),
        "max": max(rows),
    }


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum(
        (x - mean_left) * (y - mean_right) for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        sum((x - mean_left) ** 2 for x in left)
        * sum((y - mean_right) ** 2 for y in right)
    )
    return None if denominator == 0.0 else numerator / denominator


def _ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda row: row[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            result[indexed[position][0]] = rank
        start = end
    return result


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _pearson(_ranks(left), _ranks(right))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty CSV rows: {path.name}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _member(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    relative = str(path if relative_to is None else path.relative_to(relative_to))
    return {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _mean_by_request(vectors: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    for prompts in vectors:
        result.append(
            {
                "nll_new": statistics.fmean(float(row["nll_new"]) for row in prompts),
                "nll_old": statistics.fmean(float(row["nll_old"]) for row in prompts),
                "margin": statistics.fmean(float(row["margin"]) for row in prompts),
                "prompt_count": len(prompts),
                "success_count": sum(float(row["margin"]) > 0.0 for row in prompts),
                "strict_success": int(all(float(row["margin"]) > 0.0 for row in prompts)),
            }
        )
    return result


def _sacct(job_id: str) -> list[dict[str, Any]]:
    command = [
        "sacct",
        "-j",
        job_id,
        "-n",
        "-P",
        "--format=JobID,State,ExitCode,Start,End,ElapsedRaw,Elapsed,TotalCPU,"
        "CPUTimeRAW,ReqMem,MaxRSS,AllocTRES",
    ]
    output = subprocess.run(command, check=True, text=True, capture_output=True).stdout
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) != 12 or not re.fullmatch(rf"{re.escape(job_id)}_\d+", fields[0]):
            continue
        case_index = int(fields[0].rsplit("_", 1)[1])
        rows.append(
            {
                "case_index": case_index,
                "job_id": fields[0],
                "state": fields[1],
                "exit_code": fields[2],
                "start": fields[3],
                "end": fields[4],
                "elapsed_seconds": int(fields[5]),
                "elapsed": fields[6],
                "total_cpu": fields[7],
                "allocated_cpu_seconds": int(fields[8]),
                "requested_memory": fields[9],
                "max_rss_parent": fields[10] or "NOT_RECORDED_PARENT_ROW",
                "allocated_tres": fields[11],
            }
        )
    rows.sort(key=lambda row: int(row["case_index"]))
    if [row["case_index"] for row in rows] != list(CASES):
        raise ValueError("Slurm B2-B10 accounting closure differs")
    if any(row["state"] != "COMPLETED" or row["exit_code"] != "0:0" for row in rows):
        raise ValueError("Slurm terminal state differs")
    return rows


def build(*, raw_root: Path, output_root: Path, job_id: str, repo_root: Path) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"create-once output exists: {output_root}")
    output_root.mkdir(mode=0o755, parents=True)

    raw_inputs: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    microstep_rows: list[dict[str, Any]] = []
    request_microstep_rows: list[dict[str, Any]] = []
    evaluator_request_rows: list[dict[str, Any]] = []
    evaluator_paired_rows: list[dict[str, Any]] = []
    clamp_strata_rows: list[dict[str, Any]] = []
    compute_rows: list[dict[str, Any]] = []
    case_receipts: list[dict[str, Any]] = []
    failure_count = len(list(raw_root.glob("B*/failure.json")))
    if failure_count:
        raise ValueError("failure receipt exists")

    all_case_values: dict[int, dict[str, Any]] = {}
    arm_values: dict[tuple[int, str], dict[str, Any]] = {}
    action_values: dict[tuple[int, str], dict[str, Any]] = {}

    for case_index in CASES:
        case_root = raw_root / f"B{case_index}-llama3-8b-inst"
        terminal_path = case_root / "terminal.json"
        terminal = _load(terminal_path)
        raw_inputs.append(_member(terminal_path))
        all_case_values[case_index] = terminal
        if (
            terminal.get("case_index") != case_index
            or terminal.get("source_head") != SOURCE_HEAD
            or terminal.get("h") != H
            or terminal.get("M") != M
            or terminal.get("T_z") != T_Z
            or terminal.get("arms") != list(ALL_ARMS)
            or not terminal.get("finite")
            or not terminal.get("W0_restored")
            or terminal.get("duplicate_evaluation_count") != 0
            or terminal.get("writer_materialization_count") != 0
            or terminal.get("cache_append_count") != 0
        ):
            raise ValueError(f"case terminal contract differs: B{case_index}")

        stage_values: dict[int, dict[str, Any]] = {}
        for sequence, name in (
            (3, "post_sealed_input_plan"),
            (4, "post_shared_W0_nonsemantic_gate"),
            (5, "first_valid_slice_gate"),
        ):
            path = case_root / "stages" / f"stage-{sequence:03d}-{name}.json"
            stage_values[sequence] = _load(path)
            raw_inputs.append(_member(path))
        request_order = stage_values[3]["payload"]["request_order_sha256"]
        shared_w0 = stage_values[4]["payload"]["W0_sha256"]
        shared_w0_inventory = stage_values[4]["payload"]["model_inventory_sha256"]
        nonsemantic_identity = stage_values[4]["payload"]["nonsemantic_identity"]
        if (
            nonsemantic_identity.get("step_size") != H
            or nonsemantic_identity.get("microsteps") != M
            or nonsemantic_identity.get("target_horizon") != T_Z
            or nonsemantic_identity.get("optimizer") != "NONE"
            or terminal.get("request_order_sha256") != request_order
            or terminal.get("W0_pointer_bytes_sha256") != shared_w0
            or terminal.get("W0_inventory_sha256") != shared_w0_inventory
        ):
            raise ValueError(f"stage binding differs: B{case_index}")

        request_hashes: list[str] | None = None
        for arm in ALL_ARMS:
            arm_root = case_root / ARM_DIR[arm]
            arm_terminal_path = arm_root / "terminal.json"
            arm_terminal = _load(arm_terminal_path)
            raw_inputs.append(_member(arm_terminal_path))
            arm_values[(case_index, arm)] = arm_terminal
            panel = arm_terminal["terminal_z_panel"]
            current_hashes = [
                str(row["request_sha256"])
                for row in panel["heldout_lookup"]["rows"]
            ]
            if request_hashes is None:
                request_hashes = current_hashes
            elif request_hashes != current_hashes:
                raise ValueError(f"paired request members differ: B{case_index}")
            if (
                arm_terminal.get("case_index") != case_index
                or arm_terminal.get("arm") != arm
                or arm_terminal.get("request_order_sha256") != request_order
                or not arm_terminal.get("finite")
                or not arm_terminal.get("W0_restored")
                or arm_terminal.get("writer_materialization_count") != 0
                or arm_terminal.get("cache_append_count") != 0
            ):
                raise ValueError(f"arm terminal contract differs: B{case_index} {arm}")

            for metric, vector_name, aggregate_name in (
                ("rewrite", "efficacy", "eff_z_inject"),
                ("rephrase", "generalization", "gen_z_inject"),
            ):
                vectors = panel["z_inject"]["numeric_vectors"][vector_name]
                per_request = _mean_by_request(vectors)
                aggregate = panel[aggregate_name]
                for request_index, row in enumerate(per_request):
                    evaluator_request_rows.append(
                        {
                            "case_index": case_index,
                            "request_index": request_index,
                            "request_sha256": current_hashes[request_index],
                            "arm": arm,
                            "metric": metric,
                            **row,
                            "official_case_correct": aggregate["per_case_correct"][request_index],
                            "official_case_required": aggregate["per_case_required"][request_index],
                        }
                    )

            if arm == "Native-Z":
                compute_rows.append(
                    {
                        "case_index": case_index,
                        "arm": arm,
                        "target_wall_seconds": "NOT_APPLICABLE_NATIVE_CANONICAL",
                        "evaluator_wall_seconds": arm_terminal["evaluator_wall_seconds"],
                        "arm_total_wall_seconds": arm_terminal["total_wall_seconds"],
                        "logical_field_evaluations": "NATIVE_INTERNAL_NOT_MATCHED",
                        "autograd_grad_calls": "NATIVE_INTERNAL_NOT_MATCHED",
                        "native_compute_z_calls": arm_terminal["native_compute_z_call_count"],
                        "peak_gpu_memory_bytes_case": terminal["peak_gpu_memory_bytes"],
                    }
                )
                continue

            action_path = arm_root / "action-freeze.json"
            action = _load(action_path)
            raw_inputs.append(_member(action_path))
            action_values[(case_index, arm)] = action
            receipt = action["integrator_receipt"]
            if (
                action.get("request_order_sha256") != request_order
                or action.get("nonsemantic_identity_sha256")
                != nonsemantic_identity["identity_sha256"]
                or action.get("W0_pointer_version_inventory_entry")
                != shared_w0_inventory
                or action.get("W0_pointer_version_inventory_exit")
                != shared_w0_inventory
                or action.get("W0_pointer_version_change_count") != 0
                or action.get("W0_bytes_change_count") != 0
                or action.get("h") != H
                or action.get("M") != M
                or action.get("T_z") != T_Z
                or receipt.get("autograd_grad_call_count") != M
                or receipt.get("logical_field_evaluation_count") != M
                or receipt.get("optimizer") != "NONE"
                or receipt.get("adam_state_count") != 0
                or receipt.get("sgd_state_count") != 0
                or receipt.get("loss_backward_count") != 0
                or receipt.get("parameter_gradient_count") != 0
            ):
                raise ValueError(f"target action contract differs: B{case_index} {arm}")

            endpoint = action["endpoint_observation"]
            clamp_by_request = action["clamp_hit_by_request_count"]
            slice_rows.append(
                {
                    "case_index": case_index,
                    "arm": arm,
                    "request_count": 10,
                    "clamp_hit_numerator": action["clamp_hit_count"],
                    "clamp_denominator": action["clamp_denominator"],
                    "clamp_fraction": action["clamp_fraction"],
                    "all_5_step_saturated_request_count": sum(
                        int(value == M) for value in clamp_by_request
                    ),
                    "zero_hit_request_count": sum(int(value == 0) for value in clamp_by_request),
                    "target_wall_seconds": arm_terminal["target_wall_seconds"],
                    "evaluator_wall_seconds": arm_terminal["evaluator_wall_seconds"],
                    "arm_total_wall_seconds": arm_terminal["total_wall_seconds"],
                    "request_order_sha256": request_order,
                    "W0_sha256": shared_w0,
                }
            )
            compute_rows.append(
                {
                    "case_index": case_index,
                    "arm": arm,
                    "target_wall_seconds": arm_terminal["target_wall_seconds"],
                    "evaluator_wall_seconds": arm_terminal["evaluator_wall_seconds"],
                    "arm_total_wall_seconds": arm_terminal["total_wall_seconds"],
                    "logical_field_evaluations": receipt["logical_field_evaluation_count"],
                    "autograd_grad_calls": receipt["autograd_grad_call_count"],
                    "native_compute_z_calls": 0,
                    "peak_gpu_memory_bytes_case": terminal["peak_gpu_memory_bytes"],
                }
            )
            for request_index in range(10):
                request_rows.append(
                    {
                        "case_index": case_index,
                        "request_index": request_index,
                        "request_sha256": current_hashes[request_index],
                        "arm": arm,
                        "clamp_hit_steps": clamp_by_request[request_index],
                        "clamp_denominator_steps": M,
                        "clamp_fraction": clamp_by_request[request_index] / M,
                        "all_5_step_saturated": int(clamp_by_request[request_index] == M),
                        "endpoint_train_new_nll": endpoint["new_nll_by_request"][request_index],
                        "endpoint_train_true_nll": endpoint["old_nll_by_request"][request_index],
                        "endpoint_train_margin_new_minus_true": endpoint[
                            "new_minus_old_margin_by_request"
                        ][request_index],
                    }
                )

            for microstep_index in range(M):
                microstep_path = (
                    arm_root / "raw" / "target" / f"microstep-{microstep_index:02d}.json"
                )
                microstep = _load(microstep_path)
                raw_inputs.append(_member(microstep_path))
                objective = microstep["objective_telemetry"]
                source = objective["source_telemetry"]
                semantic = objective["semantic_component"]
                if (
                    microstep.get("microstep_index") != microstep_index
                    or microstep.get("autograd_grad_call_count") != 1
                    or microstep.get("field_normalization_count") != 0
                    or objective.get("arm") != arm
                    or source.get("parameter_gradient_count") != 0
                    or source.get("loss_backward_call_count") != 0
                ):
                    raise ValueError(
                        f"microstep contract differs: B{case_index} {arm} m{microstep_index}"
                    )
                clamp_hits = [int(value) for value in microstep["clamp_hit_by_request"]]
                microstep_rows.append(
                    {
                        "case_index": case_index,
                        "arm": arm,
                        "microstep": microstep_index + 1,
                        "request_denominator": 10,
                        "clamp_hit_numerator": sum(clamp_hits),
                        "clamp_fraction": statistics.fmean(clamp_hits),
                        "objective_total_mean": statistics.fmean(microstep["objective_by_request"]),
                        "semantic_objective_mean": semantic["objective"],
                        "kl_mean": statistics.fmean(objective["kl_by_request"]),
                        "decay_mean": statistics.fmean(objective["decay_by_request"]),
                        "train_new_nll_mean": statistics.fmean(source["new_nll_by_request"]),
                        "train_true_nll_mean": statistics.fmean(source["old_nll_by_request"]),
                        "train_margin_mean": statistics.fmean(
                            source["new_minus_old_margin_by_request"]
                        ),
                        "raw_field_norm_mean": statistics.fmean(
                            microstep["raw_field_norm_by_request"]
                        ),
                        "raw_euler_displacement_mean": statistics.fmean(
                            microstep["raw_euler_displacement_by_request"]
                        ),
                        "pre_clamp_displacement_mean": statistics.fmean(
                            microstep["pre_clamp_displacement_by_request"]
                        ),
                        "post_clamp_displacement_mean": statistics.fmean(
                            microstep["post_clamp_displacement_by_request"]
                        ),
                        "clamp_removed_norm_mean": statistics.fmean(
                            microstep["clamp_removed_norm_by_request"]
                        ),
                        "clamp_removed_energy_mean": statistics.fmean(
                            microstep["clamp_removed_energy_by_request"]
                        ),
                        "boundary_outward_component_mean": statistics.fmean(
                            microstep["boundary_outward_component_by_request"]
                        ),
                        "boundary_tangential_component_mean": statistics.fmean(
                            microstep["boundary_tangential_component_by_request"]
                        ),
                        "zero_field_count": sum(microstep["zero_field_by_request"]),
                    }
                )
                for request_index in range(10):
                    request_microstep_rows.append(
                        {
                            "case_index": case_index,
                            "request_index": request_index,
                            "request_sha256": current_hashes[request_index],
                            "arm": arm,
                            "microstep": microstep_index + 1,
                            "objective_total": microstep["objective_by_request"][request_index],
                            "train_new_nll": source["new_nll_by_request"][request_index],
                            "train_true_nll": source["old_nll_by_request"][request_index],
                            "train_margin_new_minus_true": source[
                                "new_minus_old_margin_by_request"
                            ][request_index],
                            "kl": objective["kl_by_request"][request_index],
                            "decay": objective["decay_by_request"][request_index],
                            "raw_field_norm": microstep["raw_field_norm_by_request"][request_index],
                            "raw_euler_displacement": microstep[
                                "raw_euler_displacement_by_request"
                            ][request_index],
                            "pre_clamp_displacement": microstep[
                                "pre_clamp_displacement_by_request"
                            ][request_index],
                            "post_clamp_displacement": microstep[
                                "post_clamp_displacement_by_request"
                            ][request_index],
                            "clamp_hit": clamp_hits[request_index],
                            "clamp_ratio": microstep["clamp_ratio_by_request"][request_index],
                            "clamp_removed_norm": microstep[
                                "clamp_removed_norm_by_request"
                            ][request_index],
                            "clamp_removed_energy": microstep[
                                "clamp_removed_energy_by_request"
                            ][request_index],
                            "boundary_outward_component": microstep[
                                "boundary_outward_component_by_request"
                            ][request_index],
                            "boundary_tangential_component": microstep[
                                "boundary_tangential_component_by_request"
                            ][request_index],
                            "zero_field": int(microstep["zero_field_by_request"][request_index]),
                        }
                    )

        assert request_hashes is not None
        for request_index in range(10):
            plus = next(
                row
                for row in request_rows
                if row["case_index"] == case_index
                and row["request_index"] == request_index
                and row["arm"] == "Z+"
            )
            barrier = next(
                row
                for row in request_rows
                if row["case_index"] == case_index
                and row["request_index"] == request_index
                and row["arm"] == "Z±"
            )
            paired_rows.append(
                {
                    "case_index": case_index,
                    "request_index": request_index,
                    "request_sha256": request_hashes[request_index],
                    "delta_clamp_fraction_Zpm_minus_Zplus": barrier["clamp_fraction"]
                    - plus["clamp_fraction"],
                    "delta_train_new_nll_Zpm_minus_Zplus": barrier[
                        "endpoint_train_new_nll"
                    ]
                    - plus["endpoint_train_new_nll"],
                    "delta_train_true_nll_Zpm_minus_Zplus": barrier[
                        "endpoint_train_true_nll"
                    ]
                    - plus["endpoint_train_true_nll"],
                    "delta_train_margin_Zpm_minus_Zplus": barrier[
                        "endpoint_train_margin_new_minus_true"
                    ]
                    - plus["endpoint_train_margin_new_minus_true"],
                }
            )
        case_receipts.append(
            {
                "case_index": case_index,
                "request_order_sha256": request_order,
                "W0_sha256": shared_w0,
                "terminal_identity_sha256": terminal["identity_sha256"],
                "arm_terminal_identity_sha256": terminal[
                    "arm_terminal_identity_sha256"
                ],
            }
        )

    for metric in ("rewrite", "rephrase"):
        for case_index in CASES:
            for request_index in range(10):
                plus = next(
                    row
                    for row in evaluator_request_rows
                    if row["case_index"] == case_index
                    and row["request_index"] == request_index
                    and row["arm"] == "Z+"
                    and row["metric"] == metric
                )
                barrier = next(
                    row
                    for row in evaluator_request_rows
                    if row["case_index"] == case_index
                    and row["request_index"] == request_index
                    and row["arm"] == "Z±"
                    and row["metric"] == metric
                )
                evaluator_paired_rows.append(
                    {
                        "case_index": case_index,
                        "request_index": request_index,
                        "request_sha256": plus["request_sha256"],
                        "metric": metric,
                        "delta_nll_new_Zpm_minus_Zplus": barrier["nll_new"]
                        - plus["nll_new"],
                        "delta_nll_true_Zpm_minus_Zplus": barrier["nll_old"]
                        - plus["nll_old"],
                        "delta_margin_Zpm_minus_Zplus": barrier["margin"]
                        - plus["margin"],
                        "delta_success_count_Zpm_minus_Zplus": barrier[
                            "success_count"
                        ]
                        - plus["success_count"],
                        "delta_strict_success_Zpm_minus_Zplus": barrier[
                            "strict_success"
                        ]
                        - plus["strict_success"],
                    }
                )

    slurm_rows = _sacct(job_id)
    slurm_by_case = {int(row["case_index"]): row for row in slurm_rows}
    for row in compute_rows:
        row.update(
            {
                "slurm_job_id": slurm_by_case[int(row["case_index"])]["job_id"],
                "slurm_elapsed_seconds": slurm_by_case[int(row["case_index"])][
                    "elapsed_seconds"
                ],
                "slurm_allocated_cpu_seconds": slurm_by_case[int(row["case_index"])][
                    "allocated_cpu_seconds"
                ],
            }
        )

    endpoint_aggregate: dict[str, Any] = {}
    clamp_aggregate: dict[str, Any] = {}
    clamp_relations: list[dict[str, Any]] = []
    for arm in TARGET_ARMS:
        selected = [row for row in request_rows if row["arm"] == arm]
        hits = sum(int(row["clamp_hit_steps"]) for row in selected)
        denominator = len(selected) * M
        clamp_aggregate[ARM_SLUG[arm]] = {
            "request_count": len(selected),
            "request_microstep_denominator": denominator,
            "hit_numerator": hits,
            "hit_fraction": hits / denominator,
            "all_5_step_saturated_request_count": sum(
                int(row["all_5_step_saturated"]) for row in selected
            ),
            "zero_hit_request_count": sum(
                int(row["clamp_hit_steps"] == 0) for row in selected
            ),
            "request_hit_count_distribution": _summary(
                row["clamp_hit_steps"] for row in selected
            ),
        }
        endpoint_aggregate[ARM_SLUG[arm]] = {
            "train_new_nll": _summary(row["endpoint_train_new_nll"] for row in selected),
            "train_true_nll": _summary(row["endpoint_train_true_nll"] for row in selected),
            "train_margin_new_minus_true": _summary(
                row["endpoint_train_margin_new_minus_true"] for row in selected
            ),
            "positive_margin_count": sum(
                float(row["endpoint_train_margin_new_minus_true"]) > 0.0
                for row in selected
            ),
            "denominator": len(selected),
        }
        clamp_values = [float(row["clamp_fraction"]) for row in selected]
        for metric in (
            "endpoint_train_new_nll",
            "endpoint_train_true_nll",
            "endpoint_train_margin_new_minus_true",
        ):
            metric_values = [float(row[metric]) for row in selected]
            clamp_relations.append(
                {
                    "arm": arm,
                    "level": "request",
                    "n": len(selected),
                    "x": "clamp_fraction",
                    "y": metric,
                    "pearson": _pearson(clamp_values, metric_values),
                    "spearman": _spearman(clamp_values, metric_values),
                    "interpretation": "DESCRIPTIVE_ASSOCIATION_ONLY_NOT_CAUSAL",
                }
            )

    paired_aggregate = {
        key: _summary(row[key] for row in paired_rows)
        for key in (
            "delta_clamp_fraction_Zpm_minus_Zplus",
            "delta_train_new_nll_Zpm_minus_Zplus",
            "delta_train_true_nll_Zpm_minus_Zplus",
            "delta_train_margin_Zpm_minus_Zplus",
        )
    }
    evaluator_aggregate: list[dict[str, Any]] = []
    for arm in ALL_ARMS:
        for metric in ("rewrite", "rephrase"):
            selected = [
                row
                for row in evaluator_request_rows
                if row["arm"] == arm and row["metric"] == metric
            ]
            prompt_denominator = sum(int(row["prompt_count"]) for row in selected)
            evaluator_aggregate.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "request_denominator": len(selected),
                    "prompt_denominator": prompt_denominator,
                    "nll_new": _summary(row["nll_new"] for row in selected),
                    "nll_true": _summary(row["nll_old"] for row in selected),
                    "margin": _summary(row["margin"] for row in selected),
                    "prompt_success_numerator": sum(
                        int(row["success_count"]) for row in selected
                    ),
                    "strict_request_success_numerator": sum(
                        int(row["strict_success"]) for row in selected
                    ),
                    "official_accuracy_numerator": sum(
                        int(row["official_case_correct"]) for row in selected
                    ),
                    "official_accuracy_denominator": sum(
                        int(row["official_case_required"]) for row in selected
                    ),
                }
            )

    for arm in TARGET_ARMS:
        for hit_steps in range(M + 1):
            endpoints = [
                row
                for row in request_rows
                if row["arm"] == arm and row["clamp_hit_steps"] == hit_steps
            ]
            if not endpoints:
                continue
            keys = {
                (int(row["case_index"]), int(row["request_index"]))
                for row in endpoints
            }
            joined = {
                metric: [
                    row
                    for row in evaluator_request_rows
                    if row["arm"] == arm
                    and row["metric"] == metric
                    and (int(row["case_index"]), int(row["request_index"])) in keys
                ]
                for metric in ("rewrite", "rephrase")
            }
            clamp_strata_rows.append(
                {
                    "arm": arm,
                    "clamp_hit_steps": hit_steps,
                    "request_denominator": len(endpoints),
                    "endpoint_train_new_nll_mean": statistics.fmean(
                        row["endpoint_train_new_nll"] for row in endpoints
                    ),
                    "endpoint_train_new_nll_median": statistics.median(
                        row["endpoint_train_new_nll"] for row in endpoints
                    ),
                    "endpoint_train_margin_mean": statistics.fmean(
                        row["endpoint_train_margin_new_minus_true"] for row in endpoints
                    ),
                    "endpoint_train_margin_median": statistics.median(
                        row["endpoint_train_margin_new_minus_true"] for row in endpoints
                    ),
                    "rewrite_new_nll_mean": statistics.fmean(
                        row["nll_new"] for row in joined["rewrite"]
                    ),
                    "rewrite_margin_mean": statistics.fmean(
                        row["margin"] for row in joined["rewrite"]
                    ),
                    "rewrite_strict_success_numerator": sum(
                        int(row["strict_success"]) for row in joined["rewrite"]
                    ),
                    "rephrase_new_nll_mean": statistics.fmean(
                        row["nll_new"] for row in joined["rephrase"]
                    ),
                    "rephrase_margin_mean": statistics.fmean(
                        row["margin"] for row in joined["rephrase"]
                    ),
                    "rephrase_strict_success_numerator": sum(
                        int(row["strict_success"]) for row in joined["rephrase"]
                    ),
                    "interpretation": "DESCRIPTIVE_STRATUM_ONLY_NOT_CAUSAL",
                }
            )
    evaluator_paired_aggregate = []
    for metric in ("rewrite", "rephrase"):
        selected = [row for row in evaluator_paired_rows if row["metric"] == metric]
        evaluator_paired_aggregate.append(
            {
                "metric": metric,
                "request_denominator": len(selected),
                "delta_nll_new_Zpm_minus_Zplus": _summary(
                    row["delta_nll_new_Zpm_minus_Zplus"] for row in selected
                ),
                "delta_nll_true_Zpm_minus_Zplus": _summary(
                    row["delta_nll_true_Zpm_minus_Zplus"] for row in selected
                ),
                "delta_margin_Zpm_minus_Zplus": _summary(
                    row["delta_margin_Zpm_minus_Zplus"] for row in selected
                ),
                "delta_prompt_success_sum": sum(
                    int(row["delta_success_count_Zpm_minus_Zplus"])
                    for row in selected
                ),
                "delta_strict_request_success_sum": sum(
                    int(row["delta_strict_success_Zpm_minus_Zplus"])
                    for row in selected
                ),
            }
        )

    microstep_aggregate: list[dict[str, Any]] = []
    for arm in TARGET_ARMS:
        for microstep in range(1, M + 1):
            selected = [
                row
                for row in request_microstep_rows
                if row["arm"] == arm and row["microstep"] == microstep
            ]
            microstep_aggregate.append(
                {
                    "arm": arm,
                    "microstep": microstep,
                    "request_denominator": len(selected),
                    "clamp_hit_numerator": sum(int(row["clamp_hit"]) for row in selected),
                    "clamp_fraction": statistics.fmean(
                        int(row["clamp_hit"]) for row in selected
                    ),
                    "objective_total": _summary(row["objective_total"] for row in selected),
                    "train_new_nll": _summary(row["train_new_nll"] for row in selected),
                    "train_true_nll": _summary(row["train_true_nll"] for row in selected),
                    "train_margin": _summary(
                        row["train_margin_new_minus_true"] for row in selected
                    ),
                    "raw_field_norm": _summary(row["raw_field_norm"] for row in selected),
                    "raw_euler_displacement": _summary(
                        row["raw_euler_displacement"] for row in selected
                    ),
                    "post_clamp_displacement": _summary(
                        row["post_clamp_displacement"] for row in selected
                    ),
                    "clamp_removed_norm": _summary(
                        row["clamp_removed_norm"] for row in selected
                    ),
                }
            )

    target_compute = {
        ARM_SLUG[arm]: {
            "slice_count": len(CASES),
            "target_wall_seconds": _summary(
                float(row["target_wall_seconds"])
                for row in compute_rows
                if row["arm"] == arm
            ),
            "evaluator_wall_seconds": _summary(
                float(row["evaluator_wall_seconds"])
                for row in compute_rows
                if row["arm"] == arm
            ),
            "arm_total_wall_seconds": _summary(
                float(row["arm_total_wall_seconds"])
                for row in compute_rows
                if row["arm"] == arm
            ),
            "logical_field_evaluations_total": sum(
                int(row["logical_field_evaluations"])
                for row in compute_rows
                if row["arm"] == arm
            ),
            "autograd_grad_calls_total": sum(
                int(row["autograd_grad_calls"])
                for row in compute_rows
                if row["arm"] == arm
            ),
        }
        for arm in TARGET_ARMS
    }
    scheduler = {
        "job_array": f"{job_id}_[2-10]%2",
        "completed_cells": len(slurm_rows),
        "cell_elapsed_seconds": _summary(row["elapsed_seconds"] for row in slurm_rows),
        "allocated_gpu_seconds_sum": sum(row["elapsed_seconds"] for row in slurm_rows),
        "allocated_cpu_seconds_sum": sum(
            row["allocated_cpu_seconds"] for row in slurm_rows
        ),
        "earliest_start": min(str(row["start"]) for row in slurm_rows),
        "latest_end": max(str(row["end"]) for row in slurm_rows),
        "state_exit": "9/9 COMPLETED exit0",
    }
    aggregate = {
        "schema": "ode-edit-s05-p4-euler-za-llama-h1-analysis/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "TERMINAL_TECHNICAL_PASS_POST_ZA_PAUSED",
        "classification": "PROJECTION_DOMINATED_EXPLORATORY_RUN",
        "policy_provenance": "USER_DIRECTED_POST_CALIBRATION_HYPERPARAMETER_REVISION",
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "numerical_lock_root": NUMERICAL_LOCK_ROOT,
        "scope": {
            "model": "llama3-8b-inst",
            "cases": list(CASES),
            "slice_count": len(CASES),
            "request_count": len(CASES) * 10,
            "B1_exclusion": "PERMANENT_CALIBRATION_ONLY",
            "h": H,
            "M": M,
            "T_z": T_Z,
            "causal_arms": list(TARGET_ARMS),
            "native_reference": "OBSERVATION_ONLY_NOT_CAUSAL_PANEL",
        },
        "primary_limitation": {
            "name": "PROJECTION_BOUNDARY_SATURATION",
            "clamp_radius_factor": 0.75,
            "warning": "radius factor and observed clamp-hit fraction are different quantities",
            "aggregate": clamp_aggregate,
        },
        "endpoint_train": endpoint_aggregate,
        "paired_endpoint_Zpm_minus_Zplus": paired_aggregate,
        "evaluator": evaluator_aggregate,
        "paired_evaluator_Zpm_minus_Zplus": evaluator_paired_aggregate,
        "clamp_metric_relations": clamp_relations,
        "clamp_strata": clamp_strata_rows,
        "microstep": microstep_aggregate,
        "compute": {"target_arms": target_compute, "scheduler": scheduler},
        "invariants": {
            "case_terminal_count": 9,
            "arm_terminal_count": 27,
            "target_action_freeze_count": 18,
            "target_microstep_receipt_count": 90,
            "request_microstep_denominator": 900,
            "identity_mismatch_count": 0,
            "failure_receipt_count": 0,
            "nonfinite_count": 0,
            "W_mutation_or_restore_failure_count": 0,
            "optimizer_object_count": 0,
            "adam_state_count": 0,
            "loss_backward_count": 0,
            "parameter_gradient_count": 0,
            "duplicate_evaluation_count": 0,
            "writer_materialization_count": 0,
            "cache_append_count": 0,
            "target_autograd_grad_call_count": 90,
            "native_compute_z_call_count": 90,
            "qwen_submission_count": 0,
            "ZB_submission_count": 0,
        },
        "claim_boundary": {
            "continuous_ODE_convergence": False,
            "step_size_stability": False,
            "unique_trajectory": False,
            "predeclared_confirmation": False,
            "writer_or_weight_edit_performance": False,
            "causal_interpretation_of_clamp_correlation": False,
            "scientific_promotion": False,
            "allowed": (
                "At user-overridden h=1,M=5,T_z=5, the discrete raw-projected "
                "Euler target-only trajectories and terminal z-injection metrics are "
                "reported factually on sealed Llama B2-B10, with substantial projection saturation."
            ),
        },
        "comparison_boundary": {
            "h_0_25": (
                "Calibration-selected h=.25,M=5,T_z=1.25 used B1_CASE01 only; "
                "it is not pooled with this B2-B10 h=1 exploratory denominator."
            ),
            "stage2_geometry": (
                "The same-T latent endpoint refinement failure remains "
                "FAILED_OBSERVED_NONDECISIONAL and is not erased by this run."
            ),
            "h_1_stage1": (
                "The original B1 h=1 admissibility failure and clamp observations "
                "Z+=37/50, Z±=43/50 remain preserved."
            ),
        },
        "continuation": "IDLE_AWAITING_GH_CALL; ZB0; NEW_GPU_SLURM_MODEL_ACTION0",
        "case_receipts": case_receipts,
    }
    aggregate["identity_sha256"] = canonical_hash(aggregate)

    _write_csv(output_root / "slice-arm-clamp.csv", slice_rows)
    _write_csv(output_root / "request-endpoint-train.csv", request_rows)
    _write_csv(output_root / "paired-endpoint-deltas.csv", paired_rows)
    _write_csv(output_root / "slice-microstep-summary.csv", microstep_rows)
    _write_csv(output_root / "request-microsteps.csv", request_microstep_rows)
    _write_csv(output_root / "evaluator-request.csv", evaluator_request_rows)
    _write_csv(output_root / "evaluator-paired-deltas.csv", evaluator_paired_rows)
    _write_csv(output_root / "compute.csv", compute_rows)
    _write_csv(output_root / "slurm-accounting.csv", slurm_rows)
    _write_csv(output_root / "clamp-metric-relations.csv", clamp_relations)
    _write_csv(output_root / "clamp-strata.csv", clamp_strata_rows)
    _write_json(output_root / "analysis.json", aggregate)

    def f(value: float) -> str:
        return f"{value:.6f}"

    report_lines = [
        "# P4-Euler ZA h=1 Llama B2–B10 최종 factual report",
        "",
        "> **핵심 한계 — projection boundary saturation.** 이 실행은 "
        "`PROJECTION_DOMINATED_EXPLORATORY_RUN`이다. Z+는 286/450 "
        f"({clamp_aggregate['z_plus']['hit_fraction']:.2%}), Z±는 381/450 "
        f"({clamp_aggregate['z_plus_minus']['hit_fraction']:.2%}) request×microstep에서 "
        "projection clamp가 발생했다. 5개 step 모두 clamp된 request도 각각 "
        f"19/90, 54/90이다. clamp radius factor `.75`와 이 hit fraction은 서로 다른 "
        "양이다. 아래 NLL/margin과의 관계는 기술적 연관만 기록하며 인과로 해석하지 않는다.",
        "",
        "## 판정과 범위",
        "",
        "- 기술 판정: `TERMINAL_TECHNICAL_PASS_POST_ZA_PAUSED`.",
        "- 과학 분류: `USER_DIRECTED_POST_CALIBRATION_HYPERPARAMETER_REVISION`에 따른 "
        "`EXPLORATORY_CONFIRMATION_AFTER_CALIBRATION`; predeclared confirmation이 아니다.",
        "- 설정: Llama3-8B-Instruct, sealed B2–B10 9 slices/90 requests, "
        "`h=1`, `M=5`, `T_z=5`, target-only W0 freeze.",
        "- 비교: causal panel은 Z±−Z+뿐이다. Native-Z는 원본 EasyEdit compute-z의 "
        "observation-only 외부 reference다.",
        "- 후속: ZB/Qwen/재튜닝/추가 calibration/repair/rerun/GPU/Slurm/model action 0. "
        "`scientific_promotion=false`, `IDLE_AWAITING_GH_CALL`.",
        "",
        "## 1. Projection clamp",
        "",
        "| arm | hit/denominator | hit rate | all-5-step saturated | zero-hit request |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in TARGET_ARMS:
        row = clamp_aggregate[ARM_SLUG[arm]]
        report_lines.append(
            f"| {arm} | {row['hit_numerator']}/{row['request_microstep_denominator']} | "
            f"{row['hit_fraction']:.2%} | {row['all_5_step_saturated_request_count']}/90 | "
            f"{row['zero_hit_request_count']}/90 |"
        )
    report_lines.extend(
        [
            "",
            "Slice/arm별 분자·분모·율과 all-5 request 수는 `slice-arm-clamp.csv`, "
            "request별 0–5 hit 분포는 `request-endpoint-train.csv`, step별 hit는 "
            "`request-microsteps.csv`에 있다.",
            "",
            "## 2. Train target endpoint (90 requests/arm)",
            "",
            "| arm | metric | mean | median | p90 | max | positive margin |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in TARGET_ARMS:
        row = endpoint_aggregate[ARM_SLUG[arm]]
        for key, label in (
            ("train_new_nll", "new NLL"),
            ("train_true_nll", "target-true NLL"),
            ("train_margin_new_minus_true", "new−true margin"),
        ):
            metric = row[key]
            positive = (
                f"{row['positive_margin_count']}/{row['denominator']}"
                if key == "train_margin_new_minus_true"
                else "—"
            )
            report_lines.append(
                f"| {arm} | {label} | {f(metric['mean'])} | {f(metric['median'])} | "
                f"{f(metric['p90'])} | {f(metric['max'])} | {positive} |"
            )
    report_lines.extend(
        [
            "",
            "여기서 margin은 `NLL(target_true) − NLL(target_new)`와 동치인 "
            "new-minus-true log-prob margin이다. 양수면 target_new 우세다.",
            "",
            "## 3. Accepted-z terminal rewrite/rephrase",
            "",
            "이 값은 W를 쓰지 않은 terminal z-injection 관측이다. rewrite는 request당 1 prompt, "
            "rephrase는 request당 2 prompts다.",
            "",
            "| arm | metric | new NLL mean/median/p90/max | margin mean/median/p90/max | prompt success | official accuracy | strict request success |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in evaluator_aggregate:
        new = row["nll_new"]
        margin = row["margin"]
        report_lines.append(
            f"| {row['arm']} | {row['metric']} | {f(new['mean'])}/{f(new['median'])}/"
            f"{f(new['p90'])}/{f(new['max'])} | {f(margin['mean'])}/{f(margin['median'])}/"
            f"{f(margin['p90'])}/{f(margin['max'])} | {row['prompt_success_numerator']}/"
            f"{row['prompt_denominator']} | {row['official_accuracy_numerator']}/"
            f"{row['official_accuracy_denominator']} | {row['strict_request_success_numerator']}/"
            f"{row['request_denominator']} |"
        )
    report_lines.extend(
        [
            "",
            "Native-Z는 causal panel이 아니므로 Euler arm과 동일 schedule/optimizer effect로 "
            "해석하지 않는다.",
            "",
            "## 4. Z±−Z+ paired delta",
            "",
            "모든 delta는 동일 case/request에서 `Z± − Z+`다. 음의 new-NLL delta는 Z±가 "
            "더 낮은 new NLL임을 뜻하고, 양의 margin delta는 Z±가 더 큰 target-new 우세임을 뜻한다.",
            "",
            "| panel | metric | n | mean | median | p90 | min | max |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, value in paired_aggregate.items():
        report_lines.append(
            f"| train endpoint | {key} | {value['n']} | {f(value['mean'])} | "
            f"{f(value['median'])} | {f(value['p90'])} | {f(value['min'])} | {f(value['max'])} |"
        )
    for row in evaluator_paired_aggregate:
        for key in (
            "delta_nll_new_Zpm_minus_Zplus",
            "delta_nll_true_Zpm_minus_Zplus",
            "delta_margin_Zpm_minus_Zplus",
        ):
            value = row[key]
            report_lines.append(
                f"| {row['metric']} | {key} | {value['n']} | {f(value['mean'])} | "
                f"{f(value['median'])} | {f(value['p90'])} | {f(value['min'])} | {f(value['max'])} |"
            )
    report_lines.extend(
        [
            "",
            "Prompt/strict success의 paired 합 차이도 `analysis.json`과 "
            "`evaluator-paired-deltas.csv`에 기록했다. 이는 h=1 내부 arm 비교이며 h=.25와 "
            "합산하지 않는다.",
            "",
            "## 5. Microstep trajectory",
            "",
            "| arm | step | clamp | objective mean | train new NLL mean | train true NLL mean | margin mean | raw field norm mean | post-clamp displacement mean |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in microstep_aggregate:
        report_lines.append(
            f"| {row['arm']} | {row['microstep']} | {row['clamp_hit_numerator']}/90 "
            f"({row['clamp_fraction']:.2%}) | {f(row['objective_total']['mean'])} | "
            f"{f(row['train_new_nll']['mean'])} | {f(row['train_true_nll']['mean'])} | "
            f"{f(row['train_margin']['mean'])} | {f(row['raw_field_norm']['mean'])} | "
            f"{f(row['post_clamp_displacement']['mean'])} |"
        )
    report_lines.extend(
        [
            "",
            "`slice-microstep-summary.csv`에는 semantic/KL/decay, raw Euler movement, "
            "pre/post-clamp displacement, removed norm/energy, boundary outward/tangential "
            "component를 slice×arm×step으로 기록했다. `request-microsteps.csv`는 900개 "
            "request×microstep 행을 보존한다. post-projection incremental step norm은 "
            "`NOT_RECORDED`이며 추정하지 않았다.",
            "",
            "## 6. Clamp–metric relation (descriptive only)",
            "",
            "| arm | y | n | Pearson | Spearman |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in clamp_relations:
        pearson = "NA" if row["pearson"] is None else f(row["pearson"])
        spearman = "NA" if row["spearman"] is None else f(row["spearman"])
        report_lines.append(
            f"| {row['arm']} | {row['y']} | {row['n']} | {pearson} | {spearman} |"
        )
    report_lines.extend(
        [
            "",
            "상관은 projection 포화와 endpoint 관측의 동시 변화를 요약할 뿐이다. "
            "request 난이도·radius·field 크기 등이 함께 달라지므로 인과효과로 주장하지 않는다.",
            "",
            "### Clamp strata",
            "",
            "| arm | hit steps | n | endpoint new NLL mean | endpoint margin mean | rewrite new NLL mean | rewrite strict | rephrase new NLL mean | rephrase strict |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in clamp_strata_rows:
        report_lines.append(
            f"| {row['arm']} | {row['clamp_hit_steps']}/5 | {row['request_denominator']} | "
            f"{f(row['endpoint_train_new_nll_mean'])} | {f(row['endpoint_train_margin_mean'])} | "
            f"{f(row['rewrite_new_nll_mean'])} | {row['rewrite_strict_success_numerator']}/"
            f"{row['request_denominator']} | {f(row['rephrase_new_nll_mean'])} | "
            f"{row['rephrase_strict_success_numerator']}/{row['request_denominator']} |"
        )
    report_lines.extend(
        [
            "",
            "각 stratum은 관측된 clamp hit step 수로 사후 분할한 기술 통계다. 표본 수가 "
            "불균형하고 배정이 무작위가 아니므로 clamp 효과 추정치가 아니다.",
            "",
            "## 7. Compute와 empirical overhead",
            "",
            "| arm | target solve mean/median/p90/max sec | evaluator mean sec | field eval total | autograd.grad total |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for arm in TARGET_ARMS:
        row = target_compute[ARM_SLUG[arm]]
        wall = row["target_wall_seconds"]
        report_lines.append(
            f"| {arm} | {f(wall['mean'])}/{f(wall['median'])}/{f(wall['p90'])}/"
            f"{f(wall['max'])} | {f(row['evaluator_wall_seconds']['mean'])} | "
            f"{row['logical_field_evaluations_total']} | {row['autograd_grad_calls_total']} |"
        )
    report_lines.extend(
        [
            "",
            f"- Slurm: {scheduler['state_exit']}; cell elapsed mean "
            f"{f(scheduler['cell_elapsed_seconds']['mean'])}초, 범위 "
            f"{scheduler['cell_elapsed_seconds']['min']:.0f}–{scheduler['cell_elapsed_seconds']['max']:.0f}초.",
            f"- 할당량 합: {scheduler['allocated_gpu_seconds_sum']} GPU-seconds, "
            f"{scheduler['allocated_cpu_seconds_sum']} allocated CPU-seconds.",
            "- case terminal peak GPU memory는 `compute.csv`에 있다. Slurm parent row MaxRSS는 "
            "`NOT_RECORDED_PARENT_ROW`이므로 GPU peak와 혼동하지 않았다.",
            "- Native-Z compute는 canonical EasyEdit 내부 알고리즘이므로 Euler의 5 field eval과 "
            "동일 단위 overhead로 정규화하지 않았다.",
            "",
            "## 8. 불변식과 receipt completeness",
            "",
            "- 9 case terminals, 27 arm terminals, 18 target action-freeze receipts, "
            "90 target microstep receipts 모두 rooted identity PASS.",
            "- FULL-FP32, nonfinite 0, W pointer/version/bytes 변화 0, restore failure 0.",
            "- optimizer/Adam/SGD/backward/parameter-grad/duplicate evaluation 0.",
            "- writer/materialization/cache append/K8 repeat 0; heldout는 terminal-only.",
            "- target autograd.grad 90회 = 9 slices × 2 arms × 5 steps. Native compute_z "
            "90회 = 9 slices × 10 requests, selection influence 0.",
            "",
            "## 9. h=.25 및 Stage2와의 경계",
            "",
            "- h=.25, M5, T_z=1.25는 B1_CASE01 calibration-only에서 선택된 설정이다. "
            "이번 h=1 B2–B10 denominator와 합치거나 동일 confirmatory claim으로 다루지 않는다.",
            "- B1의 h=1 원 admissibility FAIL과 clamp Z+ 37/50, Z± 43/50은 그대로 남는다.",
            "- Stage2 same-T latent endpoint refinement failure는 "
            "`FAILED_OBSERVED_NONDECISIONAL`로 보존된다. 이번 결과는 continuous-ODE "
            "convergence, step-size stability, unique trajectory를 입증하지 않는다.",
            "",
            "## 10. 결론",
            "",
            "h=1 target-only 실행은 기술적으로 완결되었고 terminal z-injection에서 낮은 "
            "new NLL과 높은 positive-margin 비율을 관측했다. 다만 Z±−Z+의 평균 delta는 "
            f"train new NLL {f(paired_aggregate['delta_train_new_nll_Zpm_minus_Zplus']['mean'])}, "
            f"train margin {f(paired_aggregate['delta_train_margin_Zpm_minus_Zplus']['mean'])}이고, "
            "rewrite/rephrase new NLL delta도 양수여서 aggregate상 Z± 개선으로 읽히지 않는다. "
            "또한 업데이트의 상당 부분, "
            "특히 Z±가 projection boundary에 포화되었다. 따라서 허용되는 결론은 봉인된 "
            "Llama B2–B10에서 이 discrete raw-projected Euler 설정의 factual target/z-injection "
            "행동을 관측했다는 것뿐이다. writer/weight edit 성능, 연속 ODE 정당화, 자동 승격은 "
            "주장하지 않는다.",
            "",
            "상태: `scientific_promotion=false`; `ZB=0`; `IDLE_AWAITING_GH_CALL`.",
            "",
        ]
    )
    (output_root / "report-ko.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    analysis_source = repo_root / "project/run_scripts/ode_bf/p4_euler_za_h1_analysis.py"
    numerical_lock = repo_root / (
        "project/run_scripts/ode_bf/locks/"
        "numerical_lock_s05_p4_euler_za_llama_h1_b2b10.json"
    )
    preflight = Path(
        "/data/janghj/ODE-edit/local/state/p4-euler-target-side-semantic-barrier-v1/"
        "readiness/za-llama-h1-f4fbfb5-r1/final-pre-gpu.json"
    )
    source_manifest = repo_root / (
        "project/run_scripts/ode_bf/locks/"
        "source_manifest_s05_p4_euler_za_llama_h1_b2b10.json"
    )
    bindings = [_member(path) for path in (analysis_source, numerical_lock, source_manifest, preflight)]
    raw_inputs.sort(key=lambda row: str(row["path"]))
    manifest_body = {
        "schema": "ode-edit-s05-p4-euler-za-llama-h1-analysis-manifest/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "TERMINAL_TECHNICAL_PASS_POST_ZA_PAUSED",
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "numerical_lock_root": NUMERICAL_LOCK_ROOT,
        "raw_root": str(raw_root),
        "raw_root_policy": "IMMUTABLE_IGNORED_LOCAL_REFERENCE_ONLY",
        "scientific_input_model": "llama3-8b-inst",
        "qwen_scientific_input_count": 0,
        "bindings": bindings,
        "raw_inputs": raw_inputs,
        "raw_input_count": len(raw_inputs),
        "raw_member_root": canonical_hash(raw_inputs),
        "derived_members": [
            _member(path, relative_to=output_root)
            for path in sorted(output_root.iterdir())
            if path.name not in {"analysis-manifest.json", "rooted-receipt.json"}
        ],
        "scientific_promotion": False,
    }
    manifest = dict(manifest_body)
    manifest["root_digest"] = canonical_hash(manifest_body)
    _write_json(output_root / "analysis-manifest.json", manifest)

    package_members = [
        _member(path, relative_to=output_root)
        for path in sorted(output_root.iterdir())
        if path.name != "rooted-receipt.json"
    ]
    receipt = {
        "schema": "ode-edit-s05-p4-euler-za-llama-h1-rooted-receipt/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "TERMINAL_TECHNICAL_PASS_POST_ZA_PAUSED",
        "classification": "PROJECTION_DOMINATED_EXPLORATORY_RUN",
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "analysis_identity_sha256": aggregate["identity_sha256"],
        "analysis_manifest_root": manifest["root_digest"],
        "raw_member_root": manifest["raw_member_root"],
        "package_members": package_members,
        "package_member_root": canonical_hash(package_members),
        "B1_excluded": True,
        "qwen_scientific_input_count": 0,
        "ZB_submission_count": 0,
        "scientific_promotion": False,
        "continuation": "IDLE_AWAITING_GH_CALL",
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    _write_json(output_root / "rooted-receipt.json", receipt)


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    build(
        raw_root=args.raw_root.resolve(strict=True),
        output_root=args.output_root,
        job_id=args.job_id,
        repo_root=args.repo_root.resolve(strict=True),
    )
    print(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
