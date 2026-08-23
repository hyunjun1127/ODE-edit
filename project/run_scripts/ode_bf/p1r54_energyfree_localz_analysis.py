#!/usr/bin/env python3
"""Build the P1R54 Llama B1 factual package from raw-free receipts."""

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
    _format_triplet,
    _pct,
    _prompt_values,
    _quantile,
    _summary,
)
from .p1r54_energyfree_localz import INSTRUCTION_ID
from .p1r54_energyfree_localz_b100 import RESULT_NAMES, ROLES
from project.run_scripts.session05_ode_bf_p1r54_energyfree_localz_b100_preflight import (
    CONTRACT_SHA256,
    REFERENCES,
)


SOURCE_HEAD = "27339f987c4f4147a6bf6347061444f53af095a4"
SOURCE_TREE = "3683ae0bdb7004148516514ffbf567c566dfab94"
STREAM_ROOT = "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
STREAM_ORDER = "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"
B1_ORDER = "bb2e661ad44bf8dc20e0224f21dc90164ed4b3dcf3316d751afb7d8a00bfaed6"
NUMERICAL_LOCK_ROOT = "b9f7ca3beac529491b4cf16237f8b7887a3f02913110639ed2c29a63dc4286ad"
PRE_GPU_SHA256 = "d8600915156a1428608a1b5603311a39d091d3e70645180913c936e71e1a36bd"
PRE_GPU_IDENTITY = "0e0463bc6af79622f8d0e08dd953fa43e3acb3cff95e8b9f2067a9436d71b0a6"
ARM_LABELS = {ROLES[0]: "FZ", ROLES[1]: "PDZ"}
REFERENCE_LABELS = {
    "global_p1r52_z0_coarse": "Global P1R52 Z0-COARSE†",
    "p1r53_lp_s": "P1R53 LP-S†",
    "p1r53_lfd_e": "P1R53 LFD-E†",
    "native_alphaedit": "Native AlphaEdit†",
    "native_memit": "Native MEMIT†",
}
METHOD_ORDER = (
    "FZ",
    "PDZ",
    "Global P1R52 Z0-COARSE†",
    "P1R53 LP-S†",
    "P1R53 LFD-E†",
    "Native AlphaEdit†",
    "Native MEMIT†",
)


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


def _rooted_identity(value: Mapping[str, Any]) -> bool:
    identity = value.get("identity_sha256")
    body = dict(value)
    body.pop("identity_sha256", None)
    return isinstance(identity, str) and canonical_hash(body) == identity


def _tail_summary(values: Iterable[float]) -> dict[str, float | int]:
    rows = sorted(float(value) for value in values)
    if not rows or not all(math.isfinite(value) for value in rows):
        raise ValueError("finite hard-tail values required")
    tail_count = max(1, math.ceil(len(rows) * 0.1))
    tail = rows[-tail_count:]
    return {
        "top10_count": tail_count,
        "top10_mean": statistics.fmean(tail),
        "gt1_count": sum(value > 1.0 for value in rows),
        "gt5_count": sum(value > 5.0 for value in rows),
        "gt10_count": sum(value > 10.0 for value in rows),
    }


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
        raise ValueError("P1R54 Slurm terminal accounting differs")
    return rows


def _validate_terminal(value: Mapping[str, Any], *, role: str, arm: str) -> None:
    schedule = value.get("schedule", {})
    cache = value.get("alpha_cache", {})
    dtype = value.get("dtype_contract", {})
    policy = value.get("amplitude_policy_terminal", {})
    expected_policy = {
        "FZ": "FIXED_ORIGIN_NORM_LOCAL_Z",
        "PDZ": "NLL_PACED_ORIGIN_NORM_LOCAL_Z",
    }[arm]
    if (
        not _rooted_identity(value)
        or value.get("schema") != "ode-edit-s05-p1r54-energyfree-localz-cell-terminal/v1"
        or value.get("instruction_id") != INSTRUCTION_ID
        or value.get("status") != "TERMINAL_VALID"
        or value.get("source_head") != SOURCE_HEAD
        or value.get("role") != role
        or value.get("p1r54_arm") != arm
        or value.get("p1r54_amplitude_policy") != expected_policy
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
        or value.get("p1r54_writer_layer_apply_count") != 40
        or value.get("heldout_outer_indices") != [8]
        or value.get("heldout_evaluator_count") != 3
        or value.get("heldout_decision_influence_count") != 0
        or value.get("cache_entry_reuse_count") != 8
        or value.get("cache_append_count") != 1
        or value.get("native_execution_count") != 0
        or value.get("p1r54_global_p1r52_execution_count") != 0
        or value.get("p1r54_p1r53_execution_count") != 0
        or value.get("p1r54_native_execution_count") != 0
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
        or policy.get("policy_name") != expected_policy
        or policy.get("amplitude_call_count") != 8
        or policy.get("rho_calibration_count") != 100
        or policy.get("rho_refresh_count") != 0
        or policy.get("forbidden_decision_access_count") != 0
        or policy.get("additional_model_forward_count") != 0
        or policy.get("additional_backward_count") != 0
        or policy.get("per_request_backward_loop_count") != 0
        or policy.get("retry_fallback_adaptive_count") != 0
    ):
        raise ValueError(f"P1R54 terminal contract differs: {arm}")
    micros = value.get("microstep_trajectory")
    if not isinstance(micros, list) or len(micros) != 8:
        raise ValueError(f"P1R54 trajectory differs: {arm}")
    forbidden_fragments = (
        "shared_scale_decision_access_count",
        "global_reference_energy_decision_access_count",
        "batch_nll_denominator_decision_access_count",
        "cross_request_nll_decision_access_count",
        "cross_request_gradient_reduction_decision_access_count",
        "entry_gradient_denominator_decision_access_count",
        "gradient_ratio_decision_access_count",
        "kappa_decision_access_count",
        "inverse_slope_decision_access_count",
        "unused_energy_redistribution_count",
        "added_model_forward_count",
        "added_backward_count",
        "per_request_backward_loop_count",
    )
    rho_sha = policy["rho_sha256"]
    for index, micro in enumerate(micros):
        field = micro.get("field_receipt", {})
        observation = micro.get("external_amplitude_selected_observation", {})
        if (
            micro.get("outer_step_index") != index
            or micro.get("microstep_index") != 0
            or micro.get("global_field_evaluation_ordinal") != index
            or field.get("p1r54_arm") != arm
            or field.get("p1r54_outer_index") != index
            or field.get("p1r54_target_dt") != 0.125
            or field.get("p1r54_rho_sha256") != rho_sha
            or field.get("p1r54_rho_refresh_count") != 0
            or any(field.get(f"p1r54_{fragment}") != 0 for fragment in forbidden_fragments)
            or observation.get("outer_index") != index
            or observation.get("selected_observation_decision_influence_count") != 0
        ):
            raise ValueError(f"P1R54 per-K receipt differs: {arm}/K{index + 1}")


def _scores_and_locality(value: Mapping[str, Any], *, native: bool) -> tuple[Any, Any, tuple[int, int], tuple[int, int]]:
    if native:
        z = value["native"]["z"]
        w = value["native"]["W"]
    else:
        metrics = value["kstep_executions"][7]["metrics"]
        z = metrics["accepted_z"]
        w = metrics["post_writer_W"]
    return (
        z["scores"],
        w["scores"],
        (int(z["summary"]["locality_numerator"]), int(z["summary"]["locality_denominator"])),
        (int(w["summary"]["locality_numerator"]), int(w["summary"]["locality_denominator"])),
    )


def _stream_request_hashes() -> list[str]:
    path = Path(__file__).resolve().parent / "locks/p1r52_sequential_b100x10_stream_seal.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    body = dict(value)
    root = body.pop("root_digest")
    if root != STREAM_ROOT or canonical_hash(body) != root:
        raise ValueError("tracked stream lock identity differs")
    return [str(item["request_sha256"]) for item in value["requests"][:100]]


def _fmt_summary(row: Mapping[str, Any]) -> str:
    return f"{row['mean']:.6f}/{row['median']:.6f}/{row['p90']:.6f}"


def build_package(*, raw_root: Path, output_root: Path, job_id: str, pre_gpu_receipt: Path) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"create-once output exists: {output_root}")
    output_root.mkdir(mode=0o755, parents=True)
    slurm = _sacct(job_id)
    terminals: dict[str, dict[str, Any]] = {}
    input_members: list[dict[str, Any]] = []
    for role in ROLES:
        arm = ARM_LABELS[role]
        root = raw_root / RESULT_NAMES[role]
        paths = [root / name for name in ("terminal.json", "manifest.json", "source-manifest.json", "job-timing.json")]
        terminal, manifest, source, timing = (_load(path) for path in paths)
        _validate_terminal(terminal, role=role, arm=arm)
        if (
            manifest.get("terminal_sha256") != _sha256(paths[0])
            or manifest.get("source_head") != SOURCE_HEAD
            or source.get("source_head") != SOURCE_HEAD
            or source.get("source_tree") != SOURCE_TREE
            or timing.get("source_head") != SOURCE_HEAD
        ):
            raise ValueError(f"P1R54 source/manifest binding differs: {arm}")
        terminals[arm] = terminal
        input_members.extend(_member(path) for path in paths)
    if terminals["FZ"]["amplitude_policy_terminal"]["rho_sha256"] != terminals["PDZ"]["amplitude_policy_terminal"]["rho_sha256"]:
        raise ValueError("P1R54 cross-arm rho differs")

    pre_gpu = _load(pre_gpu_receipt)
    if (
        _sha256(pre_gpu_receipt) != PRE_GPU_SHA256
        or pre_gpu.get("identity_sha256") != PRE_GPU_IDENTITY
        or pre_gpu.get("source_head") != SOURCE_HEAD
        or pre_gpu.get("source_tree") != SOURCE_TREE
    ):
        raise ValueError("P1R54 PRE-GPU receipt differs")
    input_members.append(_member(pre_gpu_receipt))

    references: dict[str, dict[str, Any]] = {}
    for label, (root, terminal_sha, manifest_sha) in REFERENCES.items():
        terminal_path, manifest_path = root / "terminal.json", root / "manifest.json"
        terminal = _load(terminal_path)
        if _sha256(terminal_path) != terminal_sha or _sha256(manifest_path) != manifest_sha:
            raise ValueError(f"P1R54 reference identity differs: {label}")
        references[label] = terminal
        input_members.extend(_member(path) for path in (terminal_path, manifest_path))

    methods: dict[str, tuple[Any, Any, tuple[int, int], tuple[int, int]]] = {}
    for arm, terminal in terminals.items():
        methods[arm] = _scores_and_locality(terminal, native=False)
    for label, terminal in references.items():
        methods[REFERENCE_LABELS[label]] = _scores_and_locality(
            terminal, native=label.startswith("native_")
        )

    endpoint_rows: list[dict[str, Any]] = []
    arrays: dict[tuple[str, str, str], dict[str, Any]] = {}
    for method in METHOD_ORDER:
        z_scores, w_scores, z_loc, w_loc = methods[method]
        for endpoint, scores, locality in (("accepted_z", z_scores, z_loc), ("post_W", w_scores, w_loc)):
            for prompt in ("rewrite", "rephrase"):
                values = _prompt_values(scores, prompt)
                arrays[(method, endpoint, prompt)] = values
                row = _endpoint_row(method=method, endpoint=endpoint, prompt=prompt, values=values, locality=locality)
                row.update({f"hard_tail_{key}": value for key, value in _tail_summary(values["new"]).items()})
                endpoint_rows.append(row)

    request_sha = _stream_request_hashes()
    paired_rows: list[dict[str, Any]] = []
    writer_rows: list[dict[str, Any]] = []
    comparison_methods = METHOD_ORDER[2:]
    for prompt in ("rewrite", "rephrase"):
        count = 1 if prompt == "rewrite" else 2
        for flat_index in range(100 * count):
            request_index, prompt_index = divmod(flat_index, count)
            fz_z = arrays[("FZ", "accepted_z", prompt)]["new"][flat_index]
            fz_w = arrays[("FZ", "post_W", prompt)]["new"][flat_index]
            pdz_z = arrays[("PDZ", "accepted_z", prompt)]["new"][flat_index]
            pdz_w = arrays[("PDZ", "post_W", prompt)]["new"][flat_index]
            row: dict[str, Any] = {
                "request_index": request_index,
                "request_sha256": request_sha[request_index],
                "prompt": prompt,
                "prompt_index": prompt_index,
                "fz_z_target_new_nll": fz_z,
                "pdz_z_target_new_nll": pdz_z,
                "pdz_minus_fz_z_nll": pdz_z - fz_z,
                "fz_W_target_new_nll": fz_w,
                "pdz_W_target_new_nll": pdz_w,
                "pdz_minus_fz_W_nll": pdz_w - fz_w,
            }
            for method in comparison_methods:
                slug = {
                    "Global P1R52 Z0-COARSE†": "global",
                    "P1R53 LP-S†": "lp_s",
                    "P1R53 LFD-E†": "lfd_e",
                    "Native AlphaEdit†": "native_alpha",
                    "Native MEMIT†": "native_memit",
                }[method]
                ref_z = arrays[(method, "accepted_z", prompt)]["new"][flat_index]
                ref_w = arrays[(method, "post_W", prompt)]["new"][flat_index]
                row.update(
                    {
                        f"{slug}_z_target_new_nll": ref_z,
                        f"fz_minus_{slug}_z_nll": fz_z - ref_z,
                        f"pdz_minus_{slug}_z_nll": pdz_z - ref_z,
                        f"{slug}_W_target_new_nll": ref_w,
                        f"fz_minus_{slug}_W_nll": fz_w - ref_w,
                        f"pdz_minus_{slug}_W_nll": pdz_w - ref_w,
                    }
                )
            paired_rows.append(row)
            for arm, z_value, w_value in (("FZ", fz_z, fz_w), ("PDZ", pdz_z, pdz_w)):
                z_success = int(arrays[(arm, "accepted_z", prompt)]["success_bits"][flat_index])
                w_success = int(arrays[(arm, "post_W", prompt)]["success_bits"][flat_index])
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
    capacity_rows: list[dict[str, Any]] = []
    allocation: dict[str, Any] = {}
    for arm, terminal in terminals.items():
        arm_rows = []
        for micro, execution, transition in zip(
            terminal["microstep_trajectory"], terminal["kstep_executions"], terminal["outer_transitions"], strict=True
        ):
            field = micro["field_receipt"]
            observation = micro["external_amplitude_selected_observation"]
            outer = int(micro["outer_step_index"]) + 1
            amplitudes = [float(value) for value in field["p1r54_amplitude_by_request"]]
            energy = sorted((value * value for value in amplitudes), reverse=True)
            total_energy = sum(energy)
            choices = observation["selection_by_request"]
            realization = transition["target_write_realization"]
            capacity_rows.append(
                {
                    "arm": arm,
                    "outer_K": outer,
                    "amplitude_mean": statistics.fmean(amplitudes),
                    "amplitude_median": statistics.median(amplitudes),
                    "amplitude_p90": _quantile(amplitudes, 0.9),
                    "amplitude_max": max(amplitudes),
                    "pre_clamp_energy": field["pre_clamp_energy"],
                    "post_cast_energy": field["post_cast_energy"],
                    "top1_energy_share": energy[0] / total_energy,
                    "top10_energy_share": sum(energy[:10]) / total_energy,
                    "bottom90_energy_share": sum(energy[10:]) / total_energy,
                    "clamp_hits": field["clamp_hit_count"],
                    "primary_count": choices.count("PRIMARY"),
                    "rescue_count": choices.count("RESCUE"),
                    "current_count": choices.count("CURRENT"),
                    "semantic_direction_cosine_mean": statistics.fmean(field["p1r51_semantic_to_kdc_direction_cosine_by_request"]),
                    "preservation_to_semantic_norm_ratio_mean": statistics.fmean(field["preservation_to_semantic_norm_ratio_by_request"]),
                    "intended_write_norm_mean": statistics.fmean(realization["intended_norm"]),
                    "realized_write_norm_mean": statistics.fmean(realization["realized_norm"]),
                    "write_norm_gain_mean": statistics.fmean(realization["norm_gain"]),
                    "write_residual_ratio_mean": statistics.fmean(realization["residual_ratio"]),
                    "writer_edit_core_wall_seconds": execution["writer_receipt"]["writer"]["edit_core_wall_seconds"],
                    "weight_update_energy": "NOT_RECORDED_RAW_FREE_TERMINAL",
                }
            )
            for index in range(100):
                row = {
                    "arm": arm,
                    "outer_K": outer,
                    "request_index": index,
                    "request_sha256": request_sha[index],
                    "rho": field["p1r54_rho_by_request"][index],
                    "deficit": None if field["p1r54_deficit_by_request"] is None else field["p1r54_deficit_by_request"][index],
                    "pacing_factor": field["p1r54_pacing_factor_by_request"][index],
                    "amplitude": amplitudes[index],
                    "semantic_inner_product": field["p1r54_semantic_inner_product_by_request"][index],
                    "semantic_direction_cosine": field["p1r51_semantic_to_kdc_direction_cosine_by_request"][index],
                    "preservation_to_semantic_norm_ratio": field["preservation_to_semantic_norm_ratio_by_request"][index],
                    "current_target_new_nll": observation["current_target_new_nll_by_request"][index],
                    "selected_target_new_nll": observation["selected_target_new_nll_by_request"][index],
                    "actual_selected_nll_decrease": observation["actual_selected_nll_decrease_by_request"][index],
                    "pre_clamp_velocity_norm": observation["pre_clamp_velocity_norm_by_request"][index],
                    "post_clamp_actual_velocity_norm": observation["post_clamp_actual_velocity_norm_by_request"][index],
                    "clamp_hit": int(observation["clamp_hit_by_request"][index]),
                    "selection": choices[index],
                }
                trajectory_rows.append(row)
                arm_rows.append(row)
        selection = [str(row["selection"]) for row in arm_rows]
        allocation[arm] = {
            "request_outer_denominator": len(arm_rows),
            "amplitude": _summary(float(row["amplitude"]) for row in arm_rows),
            "rho": _summary(float(row["rho"]) for row in arm_rows),
            "semantic_direction_cosine": _summary(float(row["semantic_direction_cosine"]) for row in arm_rows),
            "preservation_to_semantic_norm_ratio": _summary(float(row["preservation_to_semantic_norm_ratio"]) for row in arm_rows),
            "clamp_hits": sum(int(row["clamp_hit"]) for row in arm_rows),
            "selection_counts": {choice: selection.count(choice) for choice in ("PRIMARY", "RESCUE", "CURRENT")},
        }
        if arm == "PDZ":
            allocation[arm]["deficit"] = _summary(float(row["deficit"]) for row in arm_rows)

    comparison_summaries: dict[str, Any] = {}
    delta_keys = [key for key in paired_rows[0] if key.endswith("_nll") and "minus" in key]
    for prompt in ("rewrite", "rephrase"):
        rows = [row for row in paired_rows if row["prompt"] == prompt]
        comparison_summaries[prompt] = {
            key: _summary(float(row[key]) for row in rows) for key in delta_keys
        }
    writer_summaries: dict[str, Any] = {}
    for arm in ("FZ", "PDZ"):
        writer_summaries[arm] = {}
        for prompt in ("rewrite", "rephrase"):
            rows = [row for row in writer_rows if row["arm"] == arm and row["prompt"] == prompt]
            writer_summaries[arm][prompt] = {
                "pointwise_W_minus_z_nll": _summary(float(row["pointwise_W_minus_z_nll"]) for row in rows),
                "z_success_W_failure_count": sum(int(row["z_success_W_failure"]) for row in rows),
                "denominator": len(rows),
            }

    compute_rows = []
    for cell, arm in enumerate(("FZ", "PDZ")):
        terminal = terminals[arm]
        counters = terminal["job_compute"]["counters"]
        compute_rows.append(
            {
                "arm": arm,
                **slurm[cell],
                "runtime_after_model_preflight_seconds": terminal["runtime_after_model_preflight_seconds"],
                "target_field_evaluations": terminal["target_field_evaluation_count"],
                "writer_calls": terminal["writer_call_count"],
                "writer_layer_applies": terminal["p1r54_writer_layer_apply_count"],
                "added_model_forward_count": 0,
                "added_backward_count": 0,
                "per_request_backward_loop_count": 0,
                "model_forward_total": counters.get("model_forward_total", "NOT_RECORDED"),
                "model_backward_total": counters.get("model_backward_total", "NOT_RECORDED"),
                "peak_allocated_bytes": terminal["gpu_host_observation"]["peak_allocated_bytes"],
                "peak_reserved_bytes": terminal["gpu_host_observation"]["peak_reserved_bytes"],
                "W0_restored": terminal["W0_restored"],
            }
        )

    aggregates: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-energyfree-localz-analysis/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "TERMINAL_FACTUAL_ANALYSIS_COMPLETE",
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "request_count": 100,
        "endpoint_rows": endpoint_rows,
        "paired_comparisons": comparison_summaries,
        "writer_transfer": writer_summaries,
        "allocation_barrier_selector": allocation,
        "reference_policy": "SEALED_EXTERNAL_REFERENCE_REUSE_ONLY",
        "reference_execution_count": 0,
        "scientific_promotion": False,
    }
    aggregates["identity_sha256"] = canonical_hash(aggregates)

    _write_json(output_root / "aggregates.json", aggregates)
    _write_csv(output_root / "terminal-z-w-nll.csv", endpoint_rows)
    _write_csv(output_root / "paired-request-deltas.csv", paired_rows)
    _write_csv(output_root / "target-amplitude-trajectory.csv", trajectory_rows)
    _write_csv(output_root / "capacity-barrier-selector.csv", capacity_rows)
    _write_csv(output_root / "writer-gap.csv", writer_rows)
    _write_csv(output_root / "compute.csv", compute_rows)

    lookup = {(row["method"], row["endpoint"], row["prompt"]): row for row in endpoint_rows}
    report = [
        "# P1R54 Energy-Free Local-Z Llama B100 — factual report",
        "",
        "> **결론 요약:** FZ는 Global/LP-S보다 accepted-z와 post-W NLL의 mean/median을 낮추고 concentration/clamp를 완화했지만, 일부 hard tail과 K 후반 preservation-direction cosine 저하, z→W Rephrase 전달 손실이 남았다. PDZ는 clamp 없이 안정적이지만 pacing이 빠르게 약해져 FZ보다 under-edit 방향이다. 단일 B100 가능성 실험이며 `scientific_promotion=false`다.",
        "",
        "## 1. 최종 K8 핵심 표",
        "",
        "NLL은 `mean/median/p90`이며 낮을수록 좋다. Eff/Gen/Loc는 post-W 기준이다. †는 sealed external reference다.",
        "",
        "|method|W Eff|W Gen prompt|W Gen strict|W Loc|z Rewrite NLL|z Rephrase NLL|W Rewrite NLL|W Rephrase NLL|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        zrw, zrp = lookup[(method, "accepted_z", "rewrite")], lookup[(method, "accepted_z", "rephrase")]
        wrw, wrp = lookup[(method, "post_W", "rewrite")], lookup[(method, "post_W", "rephrase")]
        report.append(
            f"|{method}|{_pct(int(wrw['success_numerator']), int(wrw['success_denominator']))}|"
            f"{_pct(int(wrp['success_numerator']), int(wrp['success_denominator']))}|"
            f"{_pct(int(wrp['strict_success_numerator']), int(wrp['strict_success_denominator']))}|"
            f"{_pct(int(wrp['locality_numerator']), int(wrp['locality_denominator']))}|"
            f"{_format_triplet(zrw)}|{_format_triplet(zrp)}|{_format_triplet(wrw)}|{_format_triplet(wrp)}|"
        )

    report.extend([
        "",
        "## 2. 수식과 역할 분리",
        "",
        "- FZ: `rho_i=||z0_i||`, `F_i=m_i rho_i d_i`.",
        "- PDZ: `delta_i=-expm1(-ell_i)`, `F_i=m_i rho_i delta_i d_i`.",
        "- `rho_i`는 K0 case-entry request-local scale이며 refresh0. `ell_i`는 current six-context target-new train NLL이다.",
        "- 기존 one-sided preservation barrier는 `d_i=c_i/||c_i||`의 방향만 회전한다. P1R54 amplitude를 감쇠·reject하지 않는다. sigma는 telemetry-only다.",
        "- shared scale/global energy/batch denominator/gradient ratio/kappa/inverse sigma/redistribution decision access는 모두 0이다.",
        "",
        "## 3. Rewrite 상세",
        "",
        "|method|endpoint|new NLL mean|median|p90|max|top10% mean|>1/>5/>10|true mean|margin mean|success|accuracy|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for method in METHOD_ORDER:
        for endpoint in ("accepted_z", "post_W"):
            row = lookup[(method, endpoint, "rewrite")]
            report.append(
                f"|{method}|{endpoint}|{row['target_new_mean']:.6f}|{row['target_new_median']:.6f}|{row['target_new_p90']:.6f}|{row['target_new_max']:.6f}|{row['hard_tail_top10_mean']:.6f}|"
                f"{row['hard_tail_gt1_count']}/{row['hard_tail_gt5_count']}/{row['hard_tail_gt10_count']}|{row['target_true_mean']:.6f}|{row['margin_mean']:.6f}|"
                f"{row['success_numerator']}/{row['success_denominator']}|{row['accuracy_numerator']}/{row['accuracy_denominator']}|"
            )
    report.extend([
        "",
        "## 4. Rephrase 상세",
        "",
        "|method|endpoint|new NLL mean|median|p90|max|top10% mean|>1/>5/>10|success prompt/strict|accuracy prompt/strict|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for method in METHOD_ORDER:
        for endpoint in ("accepted_z", "post_W"):
            row = lookup[(method, endpoint, "rephrase")]
            report.append(
                f"|{method}|{endpoint}|{row['target_new_mean']:.6f}|{row['target_new_median']:.6f}|{row['target_new_p90']:.6f}|{row['target_new_max']:.6f}|{row['hard_tail_top10_mean']:.6f}|"
                f"{row['hard_tail_gt1_count']}/{row['hard_tail_gt5_count']}/{row['hard_tail_gt10_count']}|"
                f"{row['success_numerator']}/{row['success_denominator']} · {row['strict_success_numerator']}/{row['strict_success_denominator']}|"
                f"{row['accuracy_numerator']}/{row['accuracy_denominator']} · {row['strict_accuracy_numerator']}/{row['strict_accuracy_denominator']}|"
            )

    report.extend([
        "",
        "## 5. P1R54 특수 K별 amplitude·barrier·capacity",
        "",
        "|arm|K|amp mean/median/p90/max|top1/top10/bottom90 energy|clamp|P/R/C|semantic↔d cosine|pres/semantic|write intended/realized/gain|writer seconds|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in capacity_rows:
        report.append(
            f"|{row['arm']}|{row['outer_K']}|{row['amplitude_mean']:.6f}/{row['amplitude_median']:.6f}/{row['amplitude_p90']:.6f}/{row['amplitude_max']:.6f}|"
            f"{row['top1_energy_share']:.6f}/{row['top10_energy_share']:.6f}/{row['bottom90_energy_share']:.6f}|{row['clamp_hits']}/100|"
            f"{row['primary_count']}/{row['rescue_count']}/{row['current_count']}|{row['semantic_direction_cosine_mean']:.6f}|{row['preservation_to_semantic_norm_ratio_mean']:.6f}|"
            f"{row['intended_write_norm_mean']:.6f}/{row['realized_write_norm_mean']:.6f}/{row['write_norm_gain_mean']:.6f}|{row['writer_edit_core_wall_seconds']:.3f}|"
        )
    report.extend([
        "",
        "- FZ/PDZ 모두 clamp 0/800이다. FZ amplitude energy top10 share는 모든 K에서 request-local rho만 반영한다; PDZ는 NLL deficit이 줄며 후기 hard-request 쪽으로 상대 집중된다.",
        "- weight-update tensor energy 자체는 raw-free terminal에 `NOT_RECORDED`; 위 capacity 표는 target-field energy와 intended/realized z-write norm이다. 값을 추정하지 않았다.",
        "",
        "## 6. Request-paired NLL delta",
        "",
        "Delta는 앞 method minus 뒤 method이며 음수가 개선이다. pointwise delta 후 mean/median/p90을 집계했다.",
        "",
        "|prompt|comparison|z delta mean/median/p90|W delta mean/median/p90|",
        "|---|---|---:|---:|",
    ])
    comparison_rows = (
        ("PDZ−FZ", "pdz_minus_fz_z_nll", "pdz_minus_fz_W_nll"),
        ("FZ−Global", "fz_minus_global_z_nll", "fz_minus_global_W_nll"),
        ("PDZ−Global", "pdz_minus_global_z_nll", "pdz_minus_global_W_nll"),
        ("FZ−LP-S", "fz_minus_lp_s_z_nll", "fz_minus_lp_s_W_nll"),
        ("PDZ−LP-S", "pdz_minus_lp_s_z_nll", "pdz_minus_lp_s_W_nll"),
        ("FZ−LFD-E", "fz_minus_lfd_e_z_nll", "fz_minus_lfd_e_W_nll"),
        ("PDZ−LFD-E", "pdz_minus_lfd_e_z_nll", "pdz_minus_lfd_e_W_nll"),
        ("FZ−Native AlphaEdit", "fz_minus_native_alpha_z_nll", "fz_minus_native_alpha_W_nll"),
        ("PDZ−Native AlphaEdit", "pdz_minus_native_alpha_z_nll", "pdz_minus_native_alpha_W_nll"),
        ("FZ−Native MEMIT", "fz_minus_native_memit_z_nll", "fz_minus_native_memit_W_nll"),
        ("PDZ−Native MEMIT", "pdz_minus_native_memit_z_nll", "pdz_minus_native_memit_W_nll"),
    )
    for prompt in ("rewrite", "rephrase"):
        for label, zkey, wkey in comparison_rows:
            report.append(
                f"|{prompt}|{label}|{_fmt_summary(comparison_summaries[prompt][zkey])}|{_fmt_summary(comparison_summaries[prompt][wkey])}|"
            )

    report.extend([
        "",
        "## 7. z→W 전달과 compute",
        "",
        "|arm|prompt|W−z NLL mean/median/p90|max|z success→W failure|",
        "|---|---|---:|---:|",
    ])
    for arm in ("FZ", "PDZ"):
        for prompt in ("rewrite", "rephrase"):
            row = writer_summaries[arm][prompt]
            gap = row["pointwise_W_minus_z_nll"]
            report.append(
                f"|{arm}|{prompt}|{gap['mean']:.6f}/{gap['median']:.6f}/{gap['p90']:.6f}/{gap['max']:.6f}|{row['z_success_W_failure_count']}/{row['denominator']}|"
            )
    report.extend([
        "",
        "|arm|Slurm|elapsed|field eval|writer/layer|extra F/B|peak allocated/reserved|W0 restore|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in compute_rows:
        report.append(
            f"|{row['arm']}|{row['job_id']} {row['state']} {row['exit_code']}|{row['elapsed_seconds']}s|{row['target_field_evaluations']}|{row['writer_calls']}/{row['writer_layer_applies']}|0/0|{row['peak_allocated_bytes']}/{row['peak_reserved_bytes']}|{row['W0_restored']}|"
        )

    report.extend([
        "",
        "## 8. 병목 판정과 경계",
        "",
        "- **FZ:** amplitude concentration/clamp 병목은 이 B100에서 완화됐다(top10 energy 약 0.13, clamp0). accepted-z도 Global/LP-S보다 낮은 NLL을 보였다. 다만 K8에서 semantic↔direction cosine 저하와 RESCUE 증가가 관측되어 barrier/preservation interaction이 남고, Rephrase W−z gap은 writer reachability/generalization 병목을 가리키는 factual association이다.",
        "- **PDZ:** `delta=-expm1(-NLL)`가 K 후반 amplitude를 빠르게 줄여 FZ보다 accepted-z/W NLL이 높았다. 이 arm의 우선 병목 분류는 amplitude pacing/under-edit이다. inverse-sigma 폭주는 제거됐고 clamp0이지만 성능 개선으로 이어지지 않았다.",
        "- barrier cosine·selector·NLL의 동행은 인과 증명이 아니다. 별도 barrier ablation 없이 barrier 단독 원인으로 결론내리지 않는다.",
        "- † reference는 동일 sealed B1/W0/evaluator지만 새 두 arm과 같은 동시 job의 causal panel은 아니다. 실행/선택 영향0이다.",
        "- 단일 B100 결과로 Qwen/sequential/lifelong 우위나 hard promotion을 주장하지 않는다. 추가 adaptive follow-up 없이 중단한다.",
        "",
        "## 9. 산출물",
        "",
        "- `terminal-z-w-nll.csv`: method×z/W×rewrite/rephrase 전체 분포·hard tail",
        "- `paired-request-deltas.csv`: FZ/PDZ 대 Global/P1R53/Native pointwise delta",
        "- `target-amplitude-trajectory.csv`: K×request rho/deficit/amplitude/barrier/clamp/selector",
        "- `capacity-barrier-selector.csv`: K별 energy concentration·write norm·wall time",
        "- `writer-gap.csv`, `compute.csv`, `aggregates.json`, rooted manifest/receipt",
    ])
    report_path = output_root / "p1r54-energyfree-localz-llama-b100-factual-ko.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    output_names = (
        report_path.name,
        "aggregates.json",
        "terminal-z-w-nll.csv",
        "paired-request-deltas.csv",
        "target-amplitude-trajectory.csv",
        "capacity-barrier-selector.csv",
        "writer-gap.csv",
        "compute.csv",
    )
    manifest: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-energyfree-localz-analysis-manifest/v1",
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
        "schema": "ode-edit-s05-p1r54-energyfree-localz-source-receipt/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "TERMINAL_REPORT_COMPLETE",
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "job_array": f"{job_id}_[0-1]",
        "analysis_manifest_sha256": _sha256(output_root / "analysis-manifest.json"),
        "analysis_manifest_identity": manifest["identity_sha256"],
        "report_sha256": _sha256(report_path),
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
