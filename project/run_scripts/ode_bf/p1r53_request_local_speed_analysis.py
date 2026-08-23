#!/usr/bin/env python3
"""Build the P1R53 Llama B1 factual package from raw-free terminal receipts."""

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
from .p1r53_request_local_speed import INSTRUCTION_ID
from .p1r53_request_local_speed_b100 import RESULT_NAMES, ROLES
from project.run_scripts.session05_ode_bf_p1r53_request_local_speed_b100_preflight import (
    CONTRACT_SHA256,
    REFERENCES,
)


SOURCE_HEAD = "715df5845a312d10f4825560829ea271e9ef854b"
SOURCE_TREE = "cbf6a961aaed6df09c80126028de7121add107b9"
STREAM_ROOT = "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
STREAM_ORDER = "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"
B1_ORDER = "bb2e661ad44bf8dc20e0224f21dc90164ed4b3dcf3316d751afb7d8a00bfaed6"
NUMERICAL_LOCK_ROOT = "e577df9dfe3dd9a292cae10232d8d776d459f7617cefd5c2ec021526919509bb"
PRE_GPU_SHA256 = "d0618eea9f93705ffa9b5a2426e232d853b2223704c8c19371f3220c65cc5dce"
PRE_GPU_IDENTITY = "bda68a64e4d37cedda94b281558ac82b13d02cdc17d08a073f224fae463a903b"
ARM_LABELS = {ROLES[0]: "LP-S", ROLES[1]: "LFD-E"}
REFERENCE_LABELS = {
    "global_p1r52_z0_coarse": "Global P1R52 Z0-COARSE†",
    "native_alphaedit": "Native AlphaEdit†",
    "native_memit": "Native MEMIT†",
}


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(item) for item in values)
    if not ordered:
        raise ValueError("empty quantile input")
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    rows = [float(item) for item in values]
    if not rows or not all(math.isfinite(item) for item in rows):
        raise ValueError("finite non-empty summary required")
    return {
        "n": len(rows),
        "mean": statistics.fmean(rows),
        "median": statistics.median(rows),
        "p10": _quantile(rows, 0.1),
        "p90": _quantile(rows, 0.9),
        "min": min(rows),
        "max": max(rows),
    }


def _flatten(values: Sequence[Sequence[float]]) -> list[float]:
    return [float(item) for request in values for item in request]


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


def _rooted_identity(value: Mapping[str, Any]) -> bool:
    identity = value.get("identity_sha256")
    body = dict(value)
    body.pop("identity_sha256", None)
    return isinstance(identity, str) and canonical_hash(body) == identity


def _sacct(job_id: str) -> dict[int, dict[str, Any]]:
    output = subprocess.run(
        [
            "sacct", "-j", f"{job_id}_0,{job_id}_1", "-n", "-P", "-X",
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
    if set(rows) != {0, 1} or any(
        row["state"] != "COMPLETED" or row["exit_code"] != "0:0"
        for row in rows.values()
    ):
        raise ValueError("P1R53 terminal Slurm accounting differs")
    return rows


def _validate_terminal(value: Mapping[str, Any], *, role: str, arm: str) -> None:
    schedule = value.get("schedule", {})
    cache = value.get("alpha_cache", {})
    dtype = value.get("dtype_contract", {})
    policy = value.get("amplitude_policy_terminal", {})
    expected_policy = {
        "LP-S": "REQUEST_LOCAL_PARENT_AMPLITUDE_WITH_SHARED_BATCH_SCALE",
        "LFD-E": "ENTRY_MATCHED_LOCAL_FINITE_DEMAND",
    }[arm]
    if (
        not _rooted_identity(value)
        or value.get("schema") != "ode-edit-s05-p1r53-request-local-speed-cell-terminal/v1"
        or value.get("instruction_id") != INSTRUCTION_ID
        or value.get("status") != "TERMINAL_VALID"
        or value.get("source_head") != SOURCE_HEAD
        or value.get("role") != role
        or value.get("p1r53_arm") != arm
        or value.get("p1r53_amplitude_policy") != expected_policy
        or value.get("request_count") != 100
        or value.get("request_order_sha256") != B1_ORDER
        or value.get("stream_root") != STREAM_ROOT
        or value.get("stream_order") != STREAM_ORDER
        or value.get("sample_duplication_count") != 0
        or schedule.get("cell") != "Z0-COARSE"
        or schedule.get("target_horizon") != 1.0
        or schedule.get("microsteps_per_outer") != 1
        or schedule.get("target_dt") != 0.125
        or schedule.get("total_field_evaluations") != 8
        or value.get("completed_outer_count") != 8
        or value.get("target_field_evaluation_count") != 8
        or value.get("writer_call_count") != 8
        or value.get("writer_call_indices") != list(range(1, 9))
        or value.get("p1r53_writer_layer_apply_count") != 40
        or value.get("heldout_outer_indices") != [8]
        or value.get("heldout_evaluator_count") != 3
        or value.get("heldout_decision_influence_count") != 0
        or value.get("cache_entry_reuse_count") != 8
        or value.get("cache_append_count") != 1
        or value.get("native_execution_count") != 0
        or value.get("p1r53_global_p1r52_execution_count") != 0
        or value.get("p1r53_native_execution_count") != 0
        or not value.get("W0_restored")
        or not value.get("terminal_W0_restore", {}).get("byte_restored_exact")
        or not value.get("terminal_W0_restore", {}).get("pointer_restored_exact")
        or dtype.get("status") != "FULL_FP32_PASS"
        or dtype.get("bf16_fp16_path_count") != 0
        or dtype.get("autocast_count") != 0
        or cache.get("same_batch_entry_cache_reuse_count") != 8
        or cache.get("append_count") != 1
        or cache.get("rollback_count") != 0
        or policy.get("arm") != arm
        or policy.get("amplitude_call_count") != 8
        or policy.get("selected_observation_count") != 8
        or policy.get("kappa_refresh_count") != 0
        or (arm == "LP-S" and policy.get("kappa_calibration_count") != 0)
        or (arm == "LFD-E" and policy.get("kappa_calibration_count") != 100)
        or (arm == "LFD-E" and policy.get("k0_lp_lfd_amplitude_identity") is not True)
    ):
        raise ValueError(f"P1R53 terminal contract differs: {arm}")
    micros = value.get("microstep_trajectory")
    if not isinstance(micros, list) or len(micros) != 8:
        raise ValueError(f"P1R53 microstep trajectory differs: {arm}")
    for index, micro in enumerate(micros):
        field = micro.get("field_receipt", {})
        observation = micro.get("external_amplitude_selected_observation", {})
        if (
            micro.get("outer_step_index") != index
            or micro.get("microstep_index") != 0
            or micro.get("global_field_evaluation_ordinal") != index
            or field.get("p1r53_arm") != arm
            or field.get("p1r53_outer_index") != index
            or field.get("p1r53_target_dt") != 0.125
            or field.get("p1r53_per_request_backward_loop_count") != 0
            or field.get("p1r53_added_model_forward_count") != 0
            or field.get("p1r53_added_backward_count") != 0
            or observation.get("outer_index") != index
            or observation.get("selected_observation_decision_influence_count") != 0
        ):
            raise ValueError(f"P1R53 per-K receipt differs: {arm}/K{index + 1}")


def _scores_for_terminal(value: Mapping[str, Any]) -> Mapping[str, Any]:
    execution = value["kstep_executions"][7]
    if execution["step_index"] != 7 or execution["metrics"]["status"] != "HELDOUT_TERMINAL_ONLY_OBSERVATION":
        raise ValueError("P1R53 K8 evaluator binding differs")
    return execution["metrics"]


def _reference_scores(value: Mapping[str, Any], label: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if label == "global_p1r52_z0_coarse":
        metrics = value["kstep_executions"][7]["metrics"]
        return metrics["accepted_z"]["scores"], metrics["post_writer_W"]["scores"]
    return value["native"]["z"]["scores"], value["native"]["W"]["scores"]


def _prompt_values(scores: Mapping[str, Any], prompt: str) -> dict[str, Any]:
    success = scores[f"{prompt}_success"]
    accuracy = scores[f"{prompt}_acc"]
    return {
        "new": _flatten(success["target_new_nll_by_request"]),
        "true": _flatten(success["target_true_nll_by_request"]),
        "margin": _flatten(success["target_new_minus_true_margin_by_request"]),
        "success_bits": _flatten(success["per_request_bits"]),
        "accuracy_bits": _flatten(accuracy["per_request_bits"]),
        "success_numerator": int(success["prompt_numerator"]),
        "success_denominator": int(success["prompt_denominator"]),
        "strict_success_numerator": int(success["strict_request_numerator"]),
        "strict_success_denominator": int(success["strict_request_denominator"]),
        "accuracy_numerator": int(accuracy["prompt_numerator"]),
        "accuracy_denominator": int(accuracy["prompt_denominator"]),
        "strict_accuracy_numerator": int(accuracy["strict_request_numerator"]),
        "strict_accuracy_denominator": int(accuracy["strict_request_denominator"]),
    }


def _endpoint_row(
    *, method: str, endpoint: str, prompt: str, values: Mapping[str, Any], locality: tuple[int, int]
) -> dict[str, Any]:
    new = _summary(values["new"])
    true = _summary(values["true"])
    margin = _summary(values["margin"])
    return {
        "method": method,
        "endpoint": endpoint,
        "prompt": prompt,
        "prompt_denominator": new["n"],
        **{f"target_new_{key}": value for key, value in new.items() if key != "n"},
        **{f"target_true_{key}": value for key, value in true.items() if key != "n"},
        "margin_definition": "target_new_minus_target_true",
        **{f"margin_{key}": value for key, value in margin.items() if key != "n"},
        "success_numerator": values["success_numerator"],
        "success_denominator": values["success_denominator"],
        "strict_success_numerator": values["strict_success_numerator"],
        "strict_success_denominator": values["strict_success_denominator"],
        "accuracy_numerator": values["accuracy_numerator"],
        "accuracy_denominator": values["accuracy_denominator"],
        "strict_accuracy_numerator": values["strict_accuracy_numerator"],
        "strict_accuracy_denominator": values["strict_accuracy_denominator"],
        "locality_numerator": locality[0],
        "locality_denominator": locality[1],
    }


def _format_triplet(row: Mapping[str, Any]) -> str:
    return (
        f"{float(row['target_new_mean']):.6f}/"
        f"{float(row['target_new_median']):.6f}/"
        f"{float(row['target_new_p90']):.6f}"
    )


def _pct(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({numerator / denominator:.3f})"


def build_package(
    *,
    raw_root: Path,
    output_root: Path,
    job_id: str,
    pre_gpu_receipt: Path,
) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"create-once output exists: {output_root}")
    output_root.mkdir(mode=0o755, parents=True)
    slurm = _sacct(job_id)
    terminals: dict[str, dict[str, Any]] = {}
    input_members: list[dict[str, Any]] = []
    for cell, role in enumerate(ROLES):
        arm = ARM_LABELS[role]
        root = raw_root / RESULT_NAMES[role]
        terminal_path = root / "terminal.json"
        manifest_path = root / "manifest.json"
        source_path = root / "source-manifest.json"
        terminal = _load(terminal_path)
        manifest = _load(manifest_path)
        source = _load(source_path)
        _validate_terminal(terminal, role=role, arm=arm)
        if (
            manifest.get("terminal_sha256") != _sha256(terminal_path)
            or manifest.get("source_head") != SOURCE_HEAD
            or source.get("source_head") != SOURCE_HEAD
            or source.get("source_tree") != SOURCE_TREE
        ):
            raise ValueError(f"P1R53 source/manifest binding differs: {arm}")
        terminals[arm] = terminal
        input_members.extend(_member(path) for path in (terminal_path, manifest_path, source_path))

    pre_gpu = _load(pre_gpu_receipt)
    if (
        _sha256(pre_gpu_receipt) != PRE_GPU_SHA256
        or pre_gpu.get("identity_sha256") != PRE_GPU_IDENTITY
        or pre_gpu.get("source_head") != SOURCE_HEAD
        or pre_gpu.get("source_tree") != SOURCE_TREE
    ):
        raise ValueError("P1R53 PRE-GPU receipt differs")
    input_members.append(_member(pre_gpu_receipt))

    references: dict[str, dict[str, Any]] = {}
    for label, (root, terminal_sha, manifest_sha) in REFERENCES.items():
        terminal_path = root / "terminal.json"
        manifest_path = root / "manifest.json"
        terminal = _load(terminal_path)
        if _sha256(terminal_path) != terminal_sha or _sha256(manifest_path) != manifest_sha:
            raise ValueError(f"P1R53 reference identity differs: {label}")
        references[label] = terminal
        input_members.extend(_member(path) for path in (terminal_path, manifest_path))

    methods: dict[str, tuple[Mapping[str, Any], Mapping[str, Any], tuple[int, int], tuple[int, int]]] = {}
    for arm, terminal in terminals.items():
        metrics = _scores_for_terminal(terminal)
        methods[arm] = (
            metrics["accepted_z"]["scores"],
            metrics["post_writer_W"]["scores"],
            (
                int(metrics["accepted_z"]["summary"]["locality_numerator"]),
                int(metrics["accepted_z"]["summary"]["locality_denominator"]),
            ),
            (
                int(metrics["post_writer_W"]["summary"]["locality_numerator"]),
                int(metrics["post_writer_W"]["summary"]["locality_denominator"]),
            ),
        )
    for label, terminal in references.items():
        z_scores, w_scores = _reference_scores(terminal, label)
        if label == "global_p1r52_z0_coarse":
            metrics = terminal["kstep_executions"][7]["metrics"]
            z_loc = (
                int(metrics["accepted_z"]["summary"]["locality_numerator"]),
                int(metrics["accepted_z"]["summary"]["locality_denominator"]),
            )
            w_loc = (
                int(metrics["post_writer_W"]["summary"]["locality_numerator"]),
                int(metrics["post_writer_W"]["summary"]["locality_denominator"]),
            )
        else:
            z_loc = (
                int(terminal["native"]["z"]["summary"]["locality_numerator"]),
                int(terminal["native"]["z"]["summary"]["locality_denominator"]),
            )
            w_loc = (
                int(terminal["native"]["W"]["summary"]["locality_numerator"]),
                int(terminal["native"]["W"]["summary"]["locality_denominator"]),
            )
        methods[REFERENCE_LABELS[label]] = (z_scores, w_scores, z_loc, w_loc)

    endpoint_rows: list[dict[str, Any]] = []
    arrays: dict[tuple[str, str, str], dict[str, Any]] = {}
    for method, (z_scores, w_scores, z_loc, w_loc) in methods.items():
        for endpoint, scores, locality in (
            ("accepted_z", z_scores, z_loc),
            ("post_W", w_scores, w_loc),
        ):
            for prompt in ("rewrite", "rephrase"):
                values = _prompt_values(scores, prompt)
                arrays[(method, endpoint, prompt)] = values
                endpoint_rows.append(
                    _endpoint_row(
                        method=method,
                        endpoint=endpoint,
                        prompt=prompt,
                        values=values,
                        locality=locality,
                    )
                )

    stream_lock_path = (
        Path(__file__).resolve().parent
        / "locks/p1r52_sequential_b100x10_stream_seal.json"
    )
    stream_lock = json.loads(stream_lock_path.read_text(encoding="utf-8"))
    stream_lock_body = dict(stream_lock)
    stream_lock_identity = stream_lock_body.pop("root_digest", None)
    if (
        stream_lock_identity != STREAM_ROOT
        or canonical_hash(stream_lock_body) != stream_lock_identity
    ):
        raise ValueError("tracked Phase1/2/3 stream lock identity differs")
    request_hashes = stream_lock["requests"][:100]
    request_sha = [str(item["request_sha256"]) for item in request_hashes]
    paired_rows: list[dict[str, Any]] = []
    writer_rows: list[dict[str, Any]] = []
    for prompt in ("rewrite", "rephrase"):
        per_request = 1 if prompt == "rewrite" else 2
        for flat_index in range(100 * per_request):
            request_index, prompt_index = divmod(flat_index, per_request)
            lp_z = arrays[("LP-S", "accepted_z", prompt)]["new"][flat_index]
            fd_z = arrays[("LFD-E", "accepted_z", prompt)]["new"][flat_index]
            lp_w = arrays[("LP-S", "post_W", prompt)]["new"][flat_index]
            fd_w = arrays[("LFD-E", "post_W", prompt)]["new"][flat_index]
            global_z = arrays[("Global P1R52 Z0-COARSE†", "accepted_z", prompt)]["new"][flat_index]
            global_w = arrays[("Global P1R52 Z0-COARSE†", "post_W", prompt)]["new"][flat_index]
            alpha_z = arrays[("Native AlphaEdit†", "accepted_z", prompt)]["new"][flat_index]
            alpha_w = arrays[("Native AlphaEdit†", "post_W", prompt)]["new"][flat_index]
            memit_z = arrays[("Native MEMIT†", "accepted_z", prompt)]["new"][flat_index]
            memit_w = arrays[("Native MEMIT†", "post_W", prompt)]["new"][flat_index]
            paired_rows.append(
                {
                    "request_index": request_index,
                    "request_sha256": request_sha[request_index],
                    "prompt": prompt,
                    "prompt_index": prompt_index,
                    "lp_s_z_target_new_nll": lp_z,
                    "lfd_e_z_target_new_nll": fd_z,
                    "lfd_minus_lp_z_nll": fd_z - lp_z,
                    "global_z_target_new_nll": global_z,
                    "lp_minus_global_z_nll": lp_z - global_z,
                    "lfd_minus_global_z_nll": fd_z - global_z,
                    "lp_s_W_target_new_nll": lp_w,
                    "lfd_e_W_target_new_nll": fd_w,
                    "lfd_minus_lp_W_nll": fd_w - lp_w,
                    "global_W_target_new_nll": global_w,
                    "lp_minus_global_W_nll": lp_w - global_w,
                    "lfd_minus_global_W_nll": fd_w - global_w,
                    "native_alpha_z_target_new_nll": alpha_z,
                    "lp_minus_native_alpha_z_nll": lp_z - alpha_z,
                    "lfd_minus_native_alpha_z_nll": fd_z - alpha_z,
                    "native_alpha_W_target_new_nll": alpha_w,
                    "lp_minus_native_alpha_W_nll": lp_w - alpha_w,
                    "lfd_minus_native_alpha_W_nll": fd_w - alpha_w,
                    "native_memit_z_target_new_nll": memit_z,
                    "lp_minus_native_memit_z_nll": lp_z - memit_z,
                    "lfd_minus_native_memit_z_nll": fd_z - memit_z,
                    "native_memit_W_target_new_nll": memit_w,
                    "lp_minus_native_memit_W_nll": lp_w - memit_w,
                    "lfd_minus_native_memit_W_nll": fd_w - memit_w,
                }
            )
            for arm, z_value, w_value in (
                ("LP-S", lp_z, lp_w),
                ("LFD-E", fd_z, fd_w),
            ):
                z_success = int(
                    arrays[(arm, "accepted_z", prompt)]["success_bits"][flat_index]
                )
                w_success = int(
                    arrays[(arm, "post_W", prompt)]["success_bits"][flat_index]
                )
                writer_rows.append(
                    {
                        "arm": arm,
                        "request_index": request_index,
                        "request_sha256": request_sha[request_index],
                        "prompt": prompt,
                        "prompt_index": prompt_index,
                        "z_target_new_nll": z_value,
                        "W_target_new_nll": w_value,
                        "pointwise_W_minus_z_nll": w_value - z_value,
                        "z_success": z_success,
                        "W_success": w_success,
                        "z_success_W_failure": int(z_success == 1 and w_success == 0),
                    }
                )

    trajectory_rows: list[dict[str, Any]] = []
    for arm, terminal in terminals.items():
        for micro in terminal["microstep_trajectory"]:
            field = micro["field_receipt"]
            observation = micro["external_amplitude_selected_observation"]
            selection = observation["selection_by_request"]
            for index in range(100):
                nominal = observation["nominal_nll_decrease_by_request"]
                realized = observation["realized_demand_ratio_by_request"]
                unmet = observation["unmet_demand_by_request"]
                trajectory_rows.append(
                    {
                        "arm": arm,
                        "outer_K": int(micro["outer_step_index"]) + 1,
                        "request_index": index,
                        "request_sha256": request_sha[index],
                        "current_target_new_nll": observation["current_target_new_nll_by_request"][index],
                        "selected_target_new_nll": observation["selected_target_new_nll_by_request"][index],
                        "actual_selected_nll_decrease": observation["actual_selected_nll_decrease_by_request"][index],
                        "local_parent_speed": field["p1r53_local_parent_speed_by_request"][index],
                        "selected_speed": field["allocation_amplitude_by_request"][index],
                        "fd_to_lp_speed_ratio": field["p1r53_fd_to_lp_speed_ratio_by_request"][index],
                        "semantic_slope_sigma": field["p1r53_local_semantic_slope_by_request"][index],
                        "local_kappa": field["p1r53_local_kappa_by_request"][index],
                        "current_to_entry_gradient_norm_ratio": (
                            field["semantic_gradient_norm_by_request"][index]
                            / field["entry_semantic_gradient_norm_by_request"][index]
                        ),
                        "semantic_to_kdc_direction_cosine": field["p1r51_semantic_to_kdc_direction_cosine_by_request"][index],
                        "preservation_to_semantic_norm_ratio": field["preservation_to_semantic_norm_ratio_by_request"][index],
                        "pre_clamp_velocity_norm": observation["pre_clamp_velocity_norm_by_request"][index],
                        "post_clamp_actual_velocity_norm": observation["post_clamp_actual_velocity_norm_by_request"][index],
                        "clamp_hit": int(observation["clamp_hit_by_request"][index]),
                        "selection": selection[index],
                        "nominal_nll_decrease": None if nominal is None else nominal[index],
                        "realized_demand_ratio": None if realized is None else realized[index],
                        "unmet_demand": None if unmet is None else unmet[index],
                    }
                )

    compute_rows: list[dict[str, Any]] = []
    for cell, arm in enumerate(("LP-S", "LFD-E")):
        terminal = terminals[arm]
        counters = terminal["job_compute"]["counters"]
        compute_rows.append(
            {
                "arm": arm,
                "job_id": slurm[cell]["job_id"],
                "state": slurm[cell]["state"],
                "exit_code": slurm[cell]["exit_code"],
                "elapsed_seconds": slurm[cell]["elapsed_seconds"],
                "allocated_cpu_seconds": slurm[cell]["allocated_cpu_seconds"],
                "runtime_after_model_preflight_seconds": terminal["runtime_after_model_preflight_seconds"],
                "target_field_evaluations": terminal["target_field_evaluation_count"],
                "writer_calls": terminal["writer_call_count"],
                "writer_layer_applies": terminal["p1r53_writer_layer_apply_count"],
                "added_model_forward_count": 0,
                "added_backward_count": 0,
                "per_request_backward_loop_count": 0,
                "retry_count": terminal["retry_count"],
                "model_forward_total": counters.get("model_forward_total", "NOT_RECORDED"),
                "model_backward_total": counters.get("model_backward_total", "NOT_RECORDED"),
                "peak_allocated_bytes": terminal["gpu_host_observation"]["peak_allocated_bytes"],
                "peak_reserved_bytes": terminal["gpu_host_observation"]["peak_reserved_bytes"],
                "W0_restored": terminal["W0_restored"],
            }
        )

    endpoint_lookup = {
        (row["method"], row["endpoint"], row["prompt"]): row
        for row in endpoint_rows
    }
    comparison_summaries: dict[str, Any] = {}
    for prompt in ("rewrite", "rephrase"):
        prompt_rows = [row for row in paired_rows if row["prompt"] == prompt]
        comparison_summaries[prompt] = {
            key: _summary(float(row[key]) for row in prompt_rows)
            for key in (
                "lfd_minus_lp_z_nll",
                "lp_minus_global_z_nll",
                "lfd_minus_global_z_nll",
                "lfd_minus_lp_W_nll",
                "lp_minus_global_W_nll",
                "lfd_minus_global_W_nll",
                "lp_minus_native_alpha_z_nll",
                "lfd_minus_native_alpha_z_nll",
                "lp_minus_native_alpha_W_nll",
                "lfd_minus_native_alpha_W_nll",
                "lp_minus_native_memit_z_nll",
                "lfd_minus_native_memit_z_nll",
                "lp_minus_native_memit_W_nll",
                "lfd_minus_native_memit_W_nll",
            )
        }
    writer_summaries: dict[str, Any] = {}
    for arm in ("LP-S", "LFD-E"):
        writer_summaries[arm] = {}
        for prompt in ("rewrite", "rephrase"):
            rows = [row for row in writer_rows if row["arm"] == arm and row["prompt"] == prompt]
            writer_summaries[arm][prompt] = {
                "pointwise_W_minus_z_nll": _summary(
                    float(row["pointwise_W_minus_z_nll"]) for row in rows
                ),
                "z_success_W_failure_count": sum(
                    int(row["z_success_W_failure"]) for row in rows
                ),
                "denominator": len(rows),
                "ours_z_better_than_native_z_but_ours_W_worse_than_native_W": {
                    native: sum(
                        int(
                            float(row["z_target_new_nll"])
                            < arrays[(native, "accepted_z", prompt)]["new"][idx]
                            and float(row["W_target_new_nll"])
                            > arrays[(native, "post_W", prompt)]["new"][idx]
                        )
                        for idx, row in enumerate(rows)
                    )
                    for native in ("Native AlphaEdit†", "Native MEMIT†")
                },
            }

    allocation: dict[str, Any] = {}
    for arm in ("LP-S", "LFD-E"):
        rows = [row for row in trajectory_rows if row["arm"] == arm]
        energy_by_k = []
        for outer in range(1, 9):
            speeds = [
                float(row["selected_speed"])
                for row in rows
                if row["outer_K"] == outer
            ]
            energies = [value * value for value in speeds]
            total = sum(energies)
            shares = sorted((value / total for value in energies), reverse=True) if total else [0.0] * 100
            energy_by_k.append(
                {
                    "outer_K": outer,
                    "speed": _summary(speeds),
                    "top1_energy_share": shares[0],
                    "top10_energy_share": sum(shares[:10]),
                    "bottom90_energy_share": sum(shares[10:]),
                }
            )
        selection = [str(row["selection"]) for row in rows]
        allocation[arm] = {
            "request_outer_denominator": len(rows),
            "speed": _summary(float(row["selected_speed"]) for row in rows),
            "local_parent_speed": _summary(float(row["local_parent_speed"]) for row in rows),
            "semantic_slope_sigma": _summary(float(row["semantic_slope_sigma"]) for row in rows),
            "current_to_entry_gradient_norm_ratio": _summary(
                float(row["current_to_entry_gradient_norm_ratio"]) for row in rows
            ),
            "clamp_hits": sum(int(row["clamp_hit"]) for row in rows),
            "clamp_denominator": len(rows),
            "selection_counts": {
                choice: selection.count(choice) for choice in ("PRIMARY", "RESCUE", "CURRENT")
            },
            "energy_share_by_K": energy_by_k,
        }
        if arm == "LFD-E":
            allocation[arm]["realized_demand_ratio"] = _summary(
                float(row["realized_demand_ratio"]) for row in rows
            )
            allocation[arm]["unmet_demand"] = _summary(
                float(row["unmet_demand"]) for row in rows
            )

    aggregates: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r53-request-local-speed-analysis/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "TERMINAL_FACTUAL_ANALYSIS_COMPLETE",
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "B1_order": B1_ORDER,
        "request_count": 100,
        "arms": ["LP-S", "LFD-E"],
        "endpoint_rows": endpoint_rows,
        "paired_comparisons": comparison_summaries,
        "writer_transfer": writer_summaries,
        "allocation_and_pacing": allocation,
        "technical_attempts": {
            "23314_[0-1]": {
                "classification": "PURE_TECHNICAL_PRE_MODEL_SESSION_BOUNDARY_MISSING",
                "exit_code": "4:0",
                "scientific_denominator_influence_count": 0,
            },
            f"{job_id}_[0-1]": {
                "classification": "VALID_TECH_R1",
                "cells": slurm,
            },
        },
        "reference_policy": "SEALED_EXTERNAL_REFERENCE_REUSE_ONLY",
        "reference_execution_count": 0,
        "scientific_promotion": False,
    }
    aggregates["identity_sha256"] = canonical_hash(aggregates)

    _write_json(output_root / "aggregates.json", aggregates)
    _write_csv(output_root / "paired-request-deltas.csv", paired_rows)
    _write_csv(output_root / "target-speed-trajectory.csv", trajectory_rows)
    _write_csv(output_root / "terminal-z-w-nll.csv", endpoint_rows)
    _write_csv(output_root / "writer-gap.csv", writer_rows)
    _write_csv(output_root / "compute.csv", compute_rows)

    report_lines = [
        "# P1R53 request-local target-speed Llama B100 — factual report",
        "",
        "> **범위:** sealed B1 100 requests의 possibility experiment다. LP-S와 LFD-E만 새로 실행했으며, Global P1R52 Z0-COARSE 및 Native AlphaEdit/MEMIT는 동일 B1/W0/evaluator의 sealed external reference다(†). 단일 B100 결과로 promotion하거나 최종 정책을 선택하지 않는다.",
        "",
        "## 방법·불변식",
        "",
        "- LP-S: `a_i = m_i · s_B · ||g_i(t)|| / (||g_i(0)|| + eps_n)`. `s_B`는 K0에서 봉인한 batch median shared scale이다. 다른 request의 NLL/gradient/unused energy를 decision에 사용하지 않았다.",
        "- LFD-E: K0에서 `kappa_i = a_LP,i(0) · sigma_i(0) / ell_i(0)`를 request별 1회 계산·고정하고, K1–K8에서 `a_i = m_i · kappa_i · ell_i / sigma_i`를 사용했다. `sigma_i = -g_i·d_i > 0`, `d_i = c_i/||c_i||`이다.",
        "- 두 arm 모두 K=8, h=1/8, microstep=1, T=1이며 매 K의 final selected z만 Official AlphaEdit writer에 1회 전달했다. 총 writer 8회/layer apply 40회다.",
        "- target 내부 W/teacher/origin/cache/factor는 고정했고, 추가 model F/B와 request별 backward loop는 0이다. selector/clamp/preservation/direction/evaluator/stream은 Global control과 동일하다.",
        "",
        "## 1. 최종 K8 핵심 표",
        "",
        "NLL 표기는 `mean/median/p90`이고 낮을수록 좋다. Eff는 W Rewrite success, Gen은 W Rephrase prompt/strict success, Loc는 W locality다.",
        "",
        "|method|W Eff|W Gen prompt|W Gen strict|W Loc|z Rewrite NLL|z Rephrase NLL|W Rewrite NLL|W Rephrase NLL|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        zrw = endpoint_lookup[(method, "accepted_z", "rewrite")]
        zrp = endpoint_lookup[(method, "accepted_z", "rephrase")]
        wrw = endpoint_lookup[(method, "post_W", "rewrite")]
        wrp = endpoint_lookup[(method, "post_W", "rephrase")]
        report_lines.append(
            f"|{method}|{_pct(int(wrw['success_numerator']), int(wrw['success_denominator']))}|"
            f"{_pct(int(wrp['success_numerator']), int(wrp['success_denominator']))}|"
            f"{_pct(int(wrp['strict_success_numerator']), int(wrp['strict_success_denominator']))}|"
            f"{_pct(int(wrp['locality_numerator']), int(wrp['locality_denominator']))}|"
            f"{_format_triplet(zrw)}|{_format_triplet(zrp)}|{_format_triplet(wrw)}|{_format_triplet(wrp)}|"
        )

    report_lines.extend([
        "",
        "## 2. Rewrite 상세",
        "",
        "|method|endpoint|new NLL mean|median|p90|max|true NLL mean|margin mean (new−true)|success|accuracy|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for method in methods:
        for endpoint in ("accepted_z", "post_W"):
            row = endpoint_lookup[(method, endpoint, "rewrite")]
            report_lines.append(
                f"|{method}|{endpoint}|{row['target_new_mean']:.6f}|{row['target_new_median']:.6f}|{row['target_new_p90']:.6f}|{row['target_new_max']:.6f}|{row['target_true_mean']:.6f}|{row['margin_mean']:.6f}|"
                f"{row['success_numerator']}/{row['success_denominator']}|{row['accuracy_numerator']}/{row['accuracy_denominator']}|"
            )
    report_lines.extend([
        "",
        "## 3. Rephrase 상세",
        "",
        "|method|endpoint|new NLL mean|median|p90|max|true NLL mean|margin mean (new−true)|success prompt/strict|accuracy prompt/strict|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for method in methods:
        for endpoint in ("accepted_z", "post_W"):
            row = endpoint_lookup[(method, endpoint, "rephrase")]
            report_lines.append(
                f"|{method}|{endpoint}|{row['target_new_mean']:.6f}|{row['target_new_median']:.6f}|{row['target_new_p90']:.6f}|{row['target_new_max']:.6f}|{row['target_true_mean']:.6f}|{row['margin_mean']:.6f}|"
                f"{row['success_numerator']}/{row['success_denominator']} · {row['strict_success_numerator']}/{row['strict_success_denominator']}|"
                f"{row['accuracy_numerator']}/{row['accuracy_denominator']} · {row['strict_accuracy_numerator']}/{row['strict_accuracy_denominator']}|"
            )

    report_lines.extend([
        "",
        "## 4. 이 실험에 특수한 request-local speed 표",
        "",
        "|arm|request×K|speed mean/median/p90/max|σ mean/median/p10/min|clamp|PRIMARY/RESCUE/CURRENT|",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for arm in ("LP-S", "LFD-E"):
        row = allocation[arm]
        speed = row["speed"]
        sigma = row["semantic_slope_sigma"]
        choices = row["selection_counts"]
        report_lines.append(
            f"|{arm}|{row['request_outer_denominator']}|{speed['mean']:.6g}/{speed['median']:.6g}/{speed['p90']:.6g}/{speed['max']:.6g}|"
            f"{sigma['mean']:.6g}/{sigma['median']:.6g}/{sigma['p10']:.6g}/{sigma['min']:.6g}|"
            f"{row['clamp_hits']}/{row['clamp_denominator']} ({row['clamp_hits']/row['clamp_denominator']:.3f})|"
            f"{choices['PRIMARY']}/{choices['RESCUE']}/{choices['CURRENT']}|"
        )
    report_lines.extend([
        "",
        "### K별 speed energy 집중도",
        "",
        "|arm|K|speed mean|median|p90|max|top-1 energy|top-10|bottom-90|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for arm in ("LP-S", "LFD-E"):
        for row in allocation[arm]["energy_share_by_K"]:
            speed = row["speed"]
            report_lines.append(
                f"|{arm}|{row['outer_K']}|{speed['mean']:.6g}|{speed['median']:.6g}|{speed['p90']:.6g}|{speed['max']:.6g}|"
                f"{row['top1_energy_share']:.6f}|{row['top10_energy_share']:.6f}|{row['bottom90_energy_share']:.6f}|"
            )

    report_lines.extend([
        "",
        "## 5. Paired delta와 writer 전달",
        "",
        "Delta는 앞 method minus 뒤 method이며 음수가 NLL 개선이다. 아래 mean/median/p90은 먼저 pointwise delta를 만든 뒤 집계했다.",
        "",
        "|prompt|comparison|z delta mean/median/p90|W delta mean/median/p90|",
        "|---|---|---:|---:|",
    ])
    for prompt in ("rewrite", "rephrase"):
        rows = comparison_summaries[prompt]
        for label, zkey, wkey in (
            ("LFD-E − LP-S", "lfd_minus_lp_z_nll", "lfd_minus_lp_W_nll"),
            ("LP-S − Global", "lp_minus_global_z_nll", "lp_minus_global_W_nll"),
            ("LFD-E − Global", "lfd_minus_global_z_nll", "lfd_minus_global_W_nll"),
            ("LP-S − Native AlphaEdit", "lp_minus_native_alpha_z_nll", "lp_minus_native_alpha_W_nll"),
            ("LFD-E − Native AlphaEdit", "lfd_minus_native_alpha_z_nll", "lfd_minus_native_alpha_W_nll"),
            ("LP-S − Native MEMIT", "lp_minus_native_memit_z_nll", "lp_minus_native_memit_W_nll"),
            ("LFD-E − Native MEMIT", "lfd_minus_native_memit_z_nll", "lfd_minus_native_memit_W_nll"),
        ):
            zrow, wrow = rows[zkey], rows[wkey]
            report_lines.append(
                f"|{prompt}|{label}|{zrow['mean']:.6f}/{zrow['median']:.6f}/{zrow['p90']:.6f}|{wrow['mean']:.6f}/{wrow['median']:.6f}/{wrow['p90']:.6f}|"
            )
    report_lines.extend([
        "",
        "|arm|prompt|pointwise W−z mean/median/p90|max|z success→W failure|",
        "|---|---|---:|---:|---:|",
    ])
    for arm in ("LP-S", "LFD-E"):
        for prompt in ("rewrite", "rephrase"):
            row = writer_summaries[arm][prompt]
            gap = row["pointwise_W_minus_z_nll"]
            report_lines.append(
                f"|{arm}|{prompt}|{gap['mean']:.6f}/{gap['median']:.6f}/{gap['p90']:.6f}/{gap['max']:.6f}|{row['z_success_W_failure_count']}/{row['denominator']}|"
            )

    lp_vs_global = comparison_summaries["rewrite"]["lp_minus_global_z_nll"]
    fd_vs_lp = comparison_summaries["rewrite"]["lfd_minus_lp_z_nll"]
    lp_signal = "지원 방향" if lp_vs_global["mean"] < 0 and lp_vs_global["median"] < 0 else "일관된 지원 아님"
    fd_signal = "지원 방향" if fd_vs_lp["mean"] < 0 and fd_vs_lp["median"] < 0 else "일관된 지원 아님"
    report_lines.extend([
        "",
        "## 6. 계산량·완전성",
        "",
        "|arm|Slurm|elapsed|field eval|writer/layer apply|추가 F/B|peak allocated/reserved|W0 restore|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in compute_rows:
        report_lines.append(
            f"|{row['arm']}|{row['job_id']} {row['state']} {row['exit_code']}|{row['elapsed_seconds']}s|{row['target_field_evaluations']}|{row['writer_calls']}/{row['writer_layer_applies']}|0/0|{row['peak_allocated_bytes']}/{row['peak_reserved_bytes']}|{row['W0_restored']}|"
        )
    report_lines.extend([
        "",
        "- 최초 `23314_[0-1]`은 모델 로드 전 local session-boundary 파일 부재로 exit `4:0`; PURE_TECHNICAL이며 scientific denominator 영향 0이다.",
        f"- 유효 실행 `{job_id}_[0-1]`: 두 arm 모두 terminal, FULL-FP32, field eval8, writer8/layer40, cache reuse8/append1, W0 byte/pointer restore exact.",
        "- K1–K7 heldout full evaluation은 0이고 K8 accepted-z/pre-W/post-W/locality만 관측했다. controller influence는 0이다.",
        "- LP-S/LFD-E 추가 semantic/ KL F/B 및 per-request backward loop는 0이다.",
        "",
        "## 7. 가설 판정과 경계",
        "",
        f"- `LP-S > Global`의 Rewrite accepted-z mean+median 방향: **{lp_signal}**. 이는 단일 B100 descriptive signal이며 hard promotion 판정이 아니다.",
        f"- `LFD-E > LP-S`의 Rewrite accepted-z mean+median 방향: **{fd_signal}**. Rephrase와 materialized W, tail 및 clamp/selector 결과를 함께 봐야 한다.",
        "- † reference는 동일 sealed B1/W0/evaluator의 기존 실행을 재사용했지만 새 두 arm과 같은 job의 동시 causal panel은 아니다. 설정 선택 영향과 재실행은 0이다.",
        "- request-local은 shared batch scale `s_B`를 유지하므로 완전한 batch-invariant method라고 주장하지 않는다.",
        "- 단일 B100×1 결과로 sequential 성능, Qwen 전이, 최종 promotion을 주장하지 않는다. `scientific_promotion=false`다.",
        "",
        "## 8. 세부 산출물",
        "",
        "- `paired-request-deltas.csv`: arm/reference request-paired z/W delta",
        "- `target-speed-trajectory.csv`: K×request speed, σ, κ, clamp, selection, finite-demand telemetry",
        "- `terminal-z-w-nll.csv`: z/W rewrite/rephrase mean·median·p90·tail 및 success/accuracy/locality",
        "- `writer-gap.csv`: pointwise W−z와 z-success/W-failure",
        "- `compute.csv`: Slurm/runtime/field/writer/메모리 ledger",
        "- `aggregates.json`, `analysis-manifest.json`, `source-receipt.json`: rooted factual package",
    ])
    report_path = output_root / "p1r53-request-local-speed-llama-b100-factual-ko.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    output_names = (
        "p1r53-request-local-speed-llama-b100-factual-ko.md",
        "aggregates.json",
        "paired-request-deltas.csv",
        "target-speed-trajectory.csv",
        "terminal-z-w-nll.csv",
        "writer-gap.csv",
        "compute.csv",
    )
    manifest: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r53-request-local-speed-analysis-manifest/v1",
        "instruction_id": INSTRUCTION_ID,
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "authoritative_contract_sha256": CONTRACT_SHA256,
        "numerical_lock_root": NUMERICAL_LOCK_ROOT,
        "pre_gpu_sha256": PRE_GPU_SHA256,
        "pre_gpu_identity": PRE_GPU_IDENTITY,
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "B1_order": B1_ORDER,
        "input_members": input_members,
        "output_members": [_member(output_root / name, relative_to=output_root) for name in output_names],
        "scientific_promotion": False,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    _write_json(output_root / "analysis-manifest.json", manifest)
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r53-request-local-speed-source-receipt/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "TERMINAL_REPORT_COMPLETE",
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "job_array": f"{job_id}_[0-1]",
        "analysis_manifest_sha256": _sha256(output_root / "analysis-manifest.json"),
        "analysis_manifest_identity": manifest["identity_sha256"],
        "report_sha256": _sha256(report_path),
        "technical_exclusion": "23314_[0-1]_PURE_TECHNICAL_PRE_MODEL",
        "reference_execution_count": 0,
        "qwen_execution_count": 0,
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
    args = parser.parse_args(argv)
    build_package(
        raw_root=args.raw_root,
        output_root=args.output_root,
        job_id=args.job_id,
        pre_gpu_receipt=args.pre_gpu_receipt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
