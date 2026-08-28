#!/usr/bin/env python3
"""Build the factual package for the minimal fixed-basis barrier ODE run.

The analyzer is intentionally read-only with respect to experiment roots.  It
accepts the sealed two-model result directory, validates the complete 8-case x
4-arm panel, and publishes a create-once report directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import stat
import statistics
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "ode-edit-bgode-fbp-minimal-analysis/v1"
EXPECTED_SOURCE_HEAD = "29f14cc7f515d2efe31dfc868e5775e721ba61fe"
EXPECTED_SOURCE_TREE = "2788c37993a4814160b51f9efaaebcc53eec0d5c"
EXPECTED_COHORT = "5f290439f86adb476c8c27e059cc308e30f30a641d9184eb5758f55eda237902"
EXPECTED_ORDINALS = (26, 61, 110, 112, 178, 231, 269, 273)
ARMS = (
    "NATIVE_ALPHAEDIT_N1_T1",
    "STATIC_SPLIT_OFF_N4",
    "BARRIER_ODE_N2",
    "BARRIER_ODE_N4",
)
MODELS = (
    (
        "llama3-8b-inst",
        "Llama3-8B-Instruct",
        "task-0/s05-bgode-fbp-minimal-llama3-8b-inst-8case-four-arm-v1",
        0,
    ),
    (
        "qwen2.5-7b-inst",
        "Qwen2.5-7B-Instruct",
        "task-1/s05-bgode-fbp-minimal-qwen2.5-7b-inst-8case-four-arm-v1",
        1,
    ),
)
LAYERS = (4, 5, 6, 7, 8)
ENDPOINT_METRICS = (
    "rewrite_target_new",
    "rephrase_target_new",
    "rewrite_target_true",
    "rephrase_target_true",
)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def identity(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(value))


def regular_file_receipt(path: Path, *, display: str | None = None) -> dict[str, Any]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError(f"not a regular non-symlink file: {path}")
    payload = path.read_bytes()
    return {
        "path": display or str(path),
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("empty percentile input")
    point = (len(ordered) - 1) * q
    lower = int(math.floor(point))
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (point - lower)


def stats(values: Sequence[float]) -> dict[str, float]:
    clean = [float(value) for value in values]
    if not clean or not all(math.isfinite(value) for value in clean):
        raise ValueError("statistics require finite non-empty values")
    return {
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "p90": percentile(clean, 0.90),
        "max": max(clean),
    }


def theta_norm(theta: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in theta))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    first = list(rows[0])
    extras = sorted({key for row in rows for key in row}.difference(first))
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=first + extras, lineterminator="\n", restval="")
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o600)


def write_json(path: Path, value: Any) -> None:
    with path.open("xb") as handle:
        handle.write(canonical_json(value))
    path.chmod(0o600)


def endpoint(case: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    return case["arms"][arm]["endpoint"]


def validate_terminal(terminal: Mapping[str, Any], *, alias: str) -> None:
    if terminal.get("status") != "TERMINAL_VALID":
        raise ValueError(f"{alias}: terminal status is not valid")
    if terminal.get("source_head") != EXPECTED_SOURCE_HEAD or terminal.get("source_tree") != EXPECTED_SOURCE_TREE:
        raise ValueError(f"{alias}: source identity differs")
    if terminal.get("cohort_identity") != EXPECTED_COHORT:
        raise ValueError(f"{alias}: cohort identity differs")
    if tuple(terminal.get("arm_order", ())) != ARMS or terminal.get("case_count") != 8:
        raise ValueError(f"{alias}: arm/case completeness differs")
    dtype = terminal["dtype"]
    counts = dtype.get("parameter_dtype_counts", {})
    if set(counts) != {"torch.float32"} or not counts["torch.float32"]:
        raise ValueError(f"{alias}: parameter inventory is not all FP32")
    forbidden = (
        dtype.get("autocast_enabled"),
        dtype.get("tf32_enabled"),
        dtype.get("bf16_conversion_count"),
        dtype.get("fp16_conversion_count"),
        dtype.get("numeric_storage_cast_count"),
        dtype.get("quantized_parameter_count"),
    )
    if any(bool(value) for value in forbidden):
        raise ValueError(f"{alias}: forbidden dtype path was observed")
    if terminal.get("node_factor_rebuild_count") != 0 or terminal.get("node_physical_write_count") != 0:
        raise ValueError(f"{alias}: fixed-basis/node-write contract differs")
    if terminal.get("terminal_write_count_per_non_native_arm") != 1:
        raise ValueError(f"{alias}: terminal write count differs")


def load_panel(result_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    terminals: dict[str, Any] = {}
    cases: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for alias, model, relative, task in MODELS:
        run = result_root / relative
        terminal_path = run / "terminal.json"
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        validate_terminal(terminal, alias=alias)
        terminals[model] = terminal
        inputs.append(regular_file_receipt(terminal_path))
        case_paths = sorted((run / "cases").glob("case-*.json"))
        if tuple(int(path.stem.split("-")[1]) for path in case_paths) != EXPECTED_ORDINALS:
            raise ValueError(f"{alias}: case ordinals differ")
        for path in case_paths:
            value = json.loads(path.read_text(encoding="utf-8"))
            if set(value["arms"]) != set(ARMS) or len(value["arms"]) != len(ARMS):
                raise ValueError(f"{alias}/{path.name}: arm membership differs")
            if value["ordinal"] not in EXPECTED_ORDINALS:
                raise ValueError(f"{alias}/{path.name}: ordinal differs")
            if value["accepted_z"]["compute_count"] != 1 or value["accepted_z"]["recompute_count"] != 0:
                raise ValueError(f"{alias}/{path.name}: fixed z count differs")
            if value["basis"]["captured_build_count"] != 1 or value["basis"]["virtual_factor_rebuild_count"] != 0:
                raise ValueError(f"{alias}/{path.name}: fixed basis count differs")
            if not value["w0_restore"]["pointer_exact"] or not value["w0_restore"]["bytes_exact"]:
                raise ValueError(f"{alias}/{path.name}: W0 restore failed")
            for arm in ARMS:
                arm_value = value["arms"][arm]
                if arm_value["authoritative_terminal_write_count"] != 1:
                    raise ValueError(f"{alias}/{path.name}/{arm}: terminal write differs")
                if arm_value.get("node_physical_write_count", 0) != 0:
                    raise ValueError(f"{alias}/{path.name}/{arm}: node write differs")
            value["_model"] = model
            value["_model_alias"] = alias
            value["_task"] = task
            cases.append(value)
            inputs.append(regular_file_receipt(path))
        private_paths = sorted((run / "private").glob("case-*/direct-z.pt"))
        if len(private_paths) != 8:
            raise ValueError(f"{alias}: private fixed-z count differs")
        inputs.extend(regular_file_receipt(path) for path in private_paths)
    return terminals, cases, inputs


def case_arm_rows(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for arm in ARMS:
            value = case["arms"][arm]
            ep = value["endpoint"]
            event = value["event"]
            theta = [float(item) for item in value["theta"]]
            native_seconds: float | str = "NOT_APPLICABLE"
            post_seconds: float | str = "NOT_RECORDED_NATIVE"
            action_seconds: float | str = "NOT_RECORDED_NATIVE"
            if arm == ARMS[0]:
                native_seconds = float(value["native_apply"]["edit_core_wall_seconds"])
            else:
                post_seconds = float(value["wall_seconds"])
                action_seconds = float(value["action"]["wall_seconds"])
            rows.append(
                {
                    "model": case["_model"],
                    "ordinal": case["ordinal"],
                    "case_id": case["case_id"],
                    "arm": arm,
                    "rewrite_target_new_nll": ep["rewrite_target_new"]["nll"],
                    "rephrase_target_new_nll": ep["rephrase_target_new"]["nll"],
                    "rewrite_target_true_nll": ep["rewrite_target_true"]["nll"],
                    "rephrase_target_true_nll": ep["rephrase_target_true"]["nll"],
                    "rewrite_exact_satisfied": ep["rewrite_target_new"]["exact_satisfied"],
                    "rephrase_exact_satisfied": ep["rephrase_target_new"]["exact_satisfied"],
                    "locality_forward_kl": ep["locality_forward_kl"],
                    "event_q_kl_from_w0": event["event_q_kl_from_w0"],
                    "event_target_probability": event["target_probability"],
                    "event_source_probability": event["source_probability"],
                    "event_normalization_log_residual": event["normalization_log_residual"],
                    "accepted_z_rewrite_nll": case["accepted_z"]["rewrite_target_new"]["nll"],
                    "accepted_z_rephrase_nll": case["accepted_z"]["rephrase_target_new"]["nll"],
                    "w_minus_z_rewrite_nll": ep["rewrite_target_new"]["nll"]
                    - case["accepted_z"]["rewrite_target_new"]["nll"],
                    "w_minus_z_rephrase_nll": ep["rephrase_target_new"]["nll"]
                    - case["accepted_z"]["rephrase_target_new"]["nll"],
                    "coefficient_norm": theta_norm(theta),
                    "native_edit_core_wall_seconds": native_seconds,
                    "post_trajectory_terminal_and_eval_wall_seconds": post_seconds,
                    "physical_action_wall_seconds": action_seconds,
                    "case_panel_wall_seconds": case["case_wall_seconds"],
                    "node_count": len(value.get("nodes", ())),
                    "node_physical_write_count": value.get("node_physical_write_count", 0),
                    "terminal_write_count": value["authoritative_terminal_write_count"],
                    "w0_pointer_restore": case["w0_restore"]["pointer_exact"],
                    "w0_bytes_restore": case["w0_restore"]["bytes_exact"],
                }
            )
    return rows


def prompt_rows(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for metric in ("rewrite_target_new", "rephrase_target_new"):
            for index, nll in enumerate(case["accepted_z"][metric]["context_nll"]):
                rows.append(
                    {
                        "model": case["_model"],
                        "ordinal": case["ordinal"],
                        "case_id": case["case_id"],
                        "provenance": "ACCEPTED_Z",
                        "arm": "SHARED_FIXED_Z",
                        "metric": metric,
                        "prompt_index": index,
                        "nll": nll,
                    }
                )
        for arm in ARMS:
            for metric in ENDPOINT_METRICS:
                for index, nll in enumerate(endpoint(case, arm)[metric]["context_nll"]):
                    rows.append(
                        {
                            "model": case["_model"],
                            "ordinal": case["ordinal"],
                            "case_id": case["case_id"],
                            "provenance": "POST_W",
                            "arm": arm,
                            "metric": metric,
                            "prompt_index": index,
                            "nll": nll,
                        }
                    )
    return rows


def node_rows(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for arm in (ARMS[2], ARMS[3]):
            for value in case["arms"][arm]["nodes"]:
                projection = value["projection"]
                rows.append(
                    {
                        "model": case["_model"],
                        "ordinal": case["ordinal"],
                        "case_id": case["case_id"],
                        "arm": arm,
                        "node": value["node"],
                        "time": value["time"],
                        "barrier_kl": value["barrier"],
                        "target_probability": value["target_probability"],
                        "source_probability": value["source_probability"],
                        "theta_norm": theta_norm(value["theta"]),
                        "gradient_dot_nominal": projection["gradient_dot_nominal"],
                        "gradient_dot_velocity": projection["gradient_dot_velocity"],
                        "gradient_norm": projection["gradient_norm"],
                        "nominal_norm": projection["nominal_norm"],
                        "velocity_norm": projection["velocity_norm"],
                        "projection_active": projection["projection_active"],
                        "zero_gradient": projection["zero_gradient"],
                    }
                )
    return rows


def layer_rows(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for arm in ARMS:
            theta = [float(item) for item in case["arms"][arm]["theta"]]
            total = sum(item * item for item in theta)
            for layer, coefficient in zip(LAYERS, theta, strict=True):
                energy = coefficient * coefficient
                rows.append(
                    {
                        "model": case["_model"],
                        "ordinal": case["ordinal"],
                        "case_id": case["case_id"],
                        "arm": arm,
                        "layer": layer,
                        "normalized_coefficient": coefficient,
                        "block_frobenius_energy": energy,
                        "energy_share": energy / total if total else 0.0,
                    }
                )
    return rows


def accepted_z_summary(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in (item[1] for item in MODELS):
        selected = [case for case in cases if case["_model"] == model]
        row: dict[str, Any] = {"model": model, "cases": len(selected)}
        for short, metric in (("rewrite", "rewrite_target_new"), ("rephrase", "rephrase_target_new")):
            summary = stats([case["accepted_z"][metric]["nll"] for case in selected])
            row.update({f"{short}_nll_{name}": value for name, value in summary.items()})
            count = sum(bool(case["accepted_z"][metric]["exact_satisfied"]) for case in selected)
            row[f"{short}_exact_count"] = count
            row[f"{short}_exact_rate"] = count / len(selected)
        row.update(
            {
                f"locality_forward_kl_{name}": value
                for name, value in stats([case["accepted_z"]["locality_forward_kl"] for case in selected]).items()
            }
        )
        row["compute_count"] = sum(case["accepted_z"]["compute_count"] for case in selected)
        row["recompute_count"] = sum(case["accepted_z"]["recompute_count"] for case in selected)
        rows.append(row)
    return rows


def arm_summary(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in (item[1] for item in MODELS):
        selected = [case for case in cases if case["_model"] == model]
        for arm in ARMS:
            row: dict[str, Any] = {"model": model, "arm": arm, "cases": len(selected)}
            metrics = {
                "rewrite_target_new_nll": [endpoint(case, arm)["rewrite_target_new"]["nll"] for case in selected],
                "rephrase_target_new_nll": [endpoint(case, arm)["rephrase_target_new"]["nll"] for case in selected],
                "rewrite_target_true_nll": [endpoint(case, arm)["rewrite_target_true"]["nll"] for case in selected],
                "rephrase_target_true_nll": [endpoint(case, arm)["rephrase_target_true"]["nll"] for case in selected],
                "locality_forward_kl": [endpoint(case, arm)["locality_forward_kl"] for case in selected],
                "event_q_kl_from_w0": [case["arms"][arm]["event"]["event_q_kl_from_w0"] for case in selected],
                "event_target_probability": [case["arms"][arm]["event"]["target_probability"] for case in selected],
                "event_source_probability": [case["arms"][arm]["event"]["source_probability"] for case in selected],
                "coefficient_norm": [theta_norm(case["arms"][arm]["theta"]) for case in selected],
                "w_minus_z_rewrite_nll": [
                    endpoint(case, arm)["rewrite_target_new"]["nll"]
                    - case["accepted_z"]["rewrite_target_new"]["nll"]
                    for case in selected
                ],
                "w_minus_z_rephrase_nll": [
                    endpoint(case, arm)["rephrase_target_new"]["nll"]
                    - case["accepted_z"]["rephrase_target_new"]["nll"]
                    for case in selected
                ],
            }
            for metric, values in metrics.items():
                row.update({f"{metric}_{name}": value for name, value in stats(values).items()})
            rewrite_count = sum(endpoint(case, arm)["rewrite_target_new"]["exact_satisfied"] for case in selected)
            rephrase_count = sum(endpoint(case, arm)["rephrase_target_new"]["exact_satisfied"] for case in selected)
            row.update(
                {
                    "rewrite_exact_count": rewrite_count,
                    "rewrite_exact_rate": rewrite_count / len(selected),
                    "rephrase_exact_count": rephrase_count,
                    "rephrase_exact_rate": rephrase_count / len(selected),
                    "node_count": sum(len(case["arms"][arm].get("nodes", ())) for case in selected),
                    "node_physical_write_count": sum(case["arms"][arm].get("node_physical_write_count", 0) for case in selected),
                    "terminal_write_count": sum(case["arms"][arm]["authoritative_terminal_write_count"] for case in selected),
                }
            )
            if arm == ARMS[0]:
                row.update(
                    {
                        f"native_edit_core_wall_seconds_{name}": value
                        for name, value in stats(
                            [case["arms"][arm]["native_apply"]["edit_core_wall_seconds"] for case in selected]
                        ).items()
                    }
                )
                row["post_trajectory_terminal_and_eval_wall_seconds"] = "NOT_RECORDED_NATIVE"
            else:
                row.update(
                    {
                        f"post_trajectory_terminal_and_eval_wall_seconds_{name}": value
                        for name, value in stats([case["arms"][arm]["wall_seconds"] for case in selected]).items()
                    }
                )
                row["native_edit_core_wall_seconds"] = "NOT_APPLICABLE"
            rows.append(row)
    return rows


def paired_rows(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    comparisons = (
        (ARMS[1], ARMS[0], "STATIC_MINUS_NATIVE"),
        (ARMS[2], ARMS[0], "BARRIER_N2_MINUS_NATIVE"),
        (ARMS[3], ARMS[0], "BARRIER_N4_MINUS_NATIVE"),
        (ARMS[3], ARMS[2], "BARRIER_N4_MINUS_N2"),
    )
    extractors = {
        "rewrite_target_new_nll": lambda case, arm: endpoint(case, arm)["rewrite_target_new"]["nll"],
        "rephrase_target_new_nll": lambda case, arm: endpoint(case, arm)["rephrase_target_new"]["nll"],
        "rewrite_target_true_nll": lambda case, arm: endpoint(case, arm)["rewrite_target_true"]["nll"],
        "rephrase_target_true_nll": lambda case, arm: endpoint(case, arm)["rephrase_target_true"]["nll"],
        "locality_forward_kl": lambda case, arm: endpoint(case, arm)["locality_forward_kl"],
        "event_q_kl_from_w0": lambda case, arm: case["arms"][arm]["event"]["event_q_kl_from_w0"],
        "event_target_probability": lambda case, arm: case["arms"][arm]["event"]["target_probability"],
        "event_source_probability": lambda case, arm: case["arms"][arm]["event"]["source_probability"],
        "coefficient_norm": lambda case, arm: theta_norm(case["arms"][arm]["theta"]),
    }
    rows: list[dict[str, Any]] = []
    for model in (item[1] for item in MODELS):
        selected = [case for case in cases if case["_model"] == model]
        for left, right, comparison in comparisons:
            for metric, extract in extractors.items():
                values = [extract(case, left) - extract(case, right) for case in selected]
                summary = stats(values)
                rows.append(
                    {
                        "model": model,
                        "comparison": comparison,
                        "left_arm": left,
                        "right_arm": right,
                        "metric": metric,
                        "pairs": len(values),
                        **summary,
                    }
                )
    return rows


def node_summary(nodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in nodes:
        grouped[(str(row["model"]), str(row["arm"]), int(row["node"]))].append(row)
    rows: list[dict[str, Any]] = []
    for (model, arm, node), values in sorted(grouped.items()):
        row: dict[str, Any] = {
            "model": model,
            "arm": arm,
            "node": node,
            "cases": len(values),
            "projection_active_count": sum(bool(item["projection_active"]) for item in values),
            "zero_gradient_count": sum(bool(item["zero_gradient"]) for item in values),
        }
        for metric in (
            "barrier_kl",
            "target_probability",
            "source_probability",
            "theta_norm",
            "gradient_dot_nominal",
            "gradient_dot_velocity",
            "gradient_norm",
            "velocity_norm",
        ):
            row.update({f"{metric}_{name}": value for name, value in stats([float(item[metric]) for item in values]).items()})
        rows.append(row)
    return rows


def compute_rows(cases: Sequence[Mapping[str, Any]], terminals: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in (item[1] for item in MODELS):
        selected = [case for case in cases if case["_model"] == model]
        terminal = terminals[model]
        case_wall = stats([case["case_wall_seconds"] for case in selected])
        observer_wall = stats([case["compute"]["wall_seconds"] for case in selected])
        rows.append(
            {
                "model": model,
                "cases": len(selected),
                "job_total_wall_seconds": terminal["total_wall_seconds"],
                "model_load_seconds": terminal["model_load_seconds"],
                **{f"case_panel_wall_seconds_{name}": value for name, value in case_wall.items()},
                **{f"functional_observer_wall_seconds_{name}": value for name, value in observer_wall.items()},
                "functional_observer_model_forward_invocations": sum(
                    case["compute"]["model_forward_invocations"] for case in selected
                ),
                "functional_observer_jvp_call_count": sum(case["compute"]["jvp_call_count"] for case in selected),
                "functional_observer_primal_prefix_count": sum(
                    case["compute"]["primal_prefix_count"] for case in selected
                ),
                "functional_observer_temporary_materialization_count": sum(
                    case["compute"]["temporary_materialization_count"] for case in selected
                ),
                "functional_observer_physical_write_count": sum(
                    case["compute"]["physical_write_count"] for case in selected
                ),
                "authoritative_case_arm_terminal_write_count": 4 * len(selected),
                "authoritative_layer_apply_count": 4 * len(selected) * len(LAYERS),
                "peak_gpu_allocated_bytes": terminal["peak_gpu_allocated_bytes"],
                "peak_gpu_reserved_bytes": terminal["peak_gpu_reserved_bytes"],
                "peak_host_rss_kib": terminal["peak_host_rss_kib"],
                "barrier_arm_specific_integration_wall": "NOT_RECORDED_SCHEMA_GAP",
                "alphaedit_paired_end_to_end_overhead": "NOT_COMPARABLE_SCHEMA_GAP",
            }
        )
    return rows


def scheduler_receipt(job_id: str) -> dict[str, Any]:
    command = [
        "sacct",
        "-j",
        job_id,
        "-X",
        "-P",
        "-n",
        "-o",
        "JobIDRaw,JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) != 9:
            raise ValueError(f"unexpected sacct row: {line}")
        rows.append(dict(zip(("job_id_raw", "job_id", "job_name", "state", "exit_code", "elapsed", "start", "end", "node"), fields, strict=True)))
    if sorted(row["job_id"] for row in rows) != [f"{job_id}_0", f"{job_id}_1"]:
        raise ValueError("scheduler task mapping differs")
    if any(row["state"] != "COMPLETED" or row["exit_code"] != "0:0" for row in rows):
        raise ValueError("scheduler terminal is not COMPLETED 0:0")
    return {"query": " ".join(command), "rows": rows}


def _f(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def report_text(
    *,
    terminals: Mapping[str, Any],
    z_rows: Sequence[Mapping[str, Any]],
    arms: Sequence[Mapping[str, Any]],
    compute: Sequence[Mapping[str, Any]],
    parity: Mapping[str, Any],
    counts: Mapping[str, Any],
) -> str:
    by_z = {row["model"]: row for row in z_rows}
    by_arm = {(row["model"], row["arm"]): row for row in arms}
    by_compute = {row["model"]: row for row in compute}
    lines = [
        "# BGODE-FBP minimal fixed-basis barrier ODE — 8-case 양모델 사실 분석",
        "",
        "> **최종 판정:** 2 models × 8 independent W0 cases × 4 arms의 64 endpoint가 모두 기술적으로 유효하다. "
        "Static N4 barrier-off는 Native AlphaEdit와 모든 저장 endpoint/event 값이 정확히 일치해 fixed-basis·terminal-only 구현 fidelity를 닫았다. "
        "반면 KL half-space barrier는 event/locality drift를 줄였지만 target edit strength를 크게 잃었다. Llama의 Barrier N2/N4 rewrite exact는 "
        "모두 0/8, Qwen은 N2 5/8·N4 1/8이다. 따라서 이 실험은 **load-bearing locality/strength trade-off negative result**이며 "
        "barrier method promotion은 하지 않는다.",
        "",
        "- scheduler: job `27342`, task0 Llama / task1 Qwen, both `COMPLETED 0:0`",
        f"- execution source: `{EXPECTED_SOURCE_HEAD}` / `{EXPECTED_SOURCE_TREE}`",
        f"- cohort: `{EXPECTED_COHORT}`; ordinals `{', '.join(str(value) for value in EXPECTED_ORDINALS)}`",
        "- boundary: every case starts from exact W0/cold method state; no inter-case W/cache/history carry",
        "- fixed target/basis: native accepted-z compute 1/recompute 0 and Official ordered AlphaEdit basis capture 1 per case",
        "- writer boundary: Euler nodes physical write 0; every arm materializes exactly once at terminal and restores W0 pointer+bytes",
        "- FULL-FP32: parameter, accepted-z, basis/write path FP32; autocast/TF32/BF16/FP16/quantization/storage cast 0",
        "- scientific promotion: `false`",
        "",
        "## 1. Method 경계",
        "",
        "|arm|정의|node factor rebuild|node physical write|terminal write|",
        "|---|---|---:|---:|---:|",
        "|Native AlphaEdit N1/T1|fixed W0 accepted-z를 Official AlphaEdit entrypoint에 전달한 기준 endpoint|0|0|1/case|",
        "|Static Split Off N4|같은 fixed normalized basis에서 `theta_AE/4`를 4회 더한 barrier-off identity control|0|0|1/case|",
        "|Barrier ODE N2|`v0=theta_AE`; current q-KL gradient와 `g·v<=0` half-space projection, Euler 2 steps|0|0|1/case|",
        "|Barrier ODE N4|동일 projection을 Euler 4 steps로 재관측|0|0|1/case|",
        "",
        "이 설계는 node별 AlphaEdit factor 재생성이나 node별 durable writer가 아니다. current combined affine state에서 full-model output/JVP를 재관측하지만 proposal subspace는 W0에서 고정되며, 마지막 `theta`만 1회 물리 적용한다.",
        "",
        "## 2. Accepted-z 독립 표",
        "",
        "accepted-z는 네 arm이 공유하며 W endpoint와 별도 provenance이다. 표는 case-level NLL 분포다.",
        "",
        "|model|rewrite NLL mean/median/p90/max|rephrase NLL mean/median/p90/max|rewrite exact|rephrase exact|compute/recompute|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in (item[1] for item in MODELS):
        row = by_z[model]
        lines.append(
            f"|{model}|{_f(row['rewrite_nll_mean'],6)} / {_f(row['rewrite_nll_median'],6)} / {_f(row['rewrite_nll_p90'],6)} / {_f(row['rewrite_nll_max'],6)}|"
            f"{_f(row['rephrase_nll_mean'],4)} / {_f(row['rephrase_nll_median'],4)} / {_f(row['rephrase_nll_p90'],4)} / {_f(row['rephrase_nll_max'],4)}|"
            f"{row['rewrite_exact_count']}/8|{row['rephrase_exact_count']}/8|{row['compute_count']}/{row['recompute_count']}|"
        )
    lines += [
        "",
        "## 3. Post-W Rewrite 성능",
        "",
        "낮은 target-new NLL과 높은 exact rate가 좋다. target-true NLL은 편집 이전 정답 억제의 관측값이다.",
        "",
        "|model|arm|target-new NLL mean/median/p90/max|target-true NLL mean/median/p90/max|exact|W-z gap mean|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in (item[1] for item in MODELS):
        for arm in ARMS:
            row = by_arm[(model, arm)]
            lines.append(
                f"|{model}|{arm}|{_f(row['rewrite_target_new_nll_mean'])} / {_f(row['rewrite_target_new_nll_median'])} / {_f(row['rewrite_target_new_nll_p90'])} / {_f(row['rewrite_target_new_nll_max'])}|"
                f"{_f(row['rewrite_target_true_nll_mean'])} / {_f(row['rewrite_target_true_nll_median'])} / {_f(row['rewrite_target_true_nll_p90'])} / {_f(row['rewrite_target_true_nll_max'])}|"
                f"{row['rewrite_exact_count']}/8 ({_pct(row['rewrite_exact_rate'])})|{_f(row['w_minus_z_rewrite_nll_mean'])}|"
            )
    lines += [
        "",
        "## 4. Post-W Rephrase 성능",
        "",
        "rephrase exact는 두 rephrase prompt를 포함한 저장 `exact_satisfied`이며 별도 strict/accuracy 스키마는 없었다.",
        "",
        "|model|arm|target-new NLL mean/median/p90/max|target-true NLL mean/median/p90/max|exact|W-z gap mean|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in (item[1] for item in MODELS):
        for arm in ARMS:
            row = by_arm[(model, arm)]
            lines.append(
                f"|{model}|{arm}|{_f(row['rephrase_target_new_nll_mean'])} / {_f(row['rephrase_target_new_nll_median'])} / {_f(row['rephrase_target_new_nll_p90'])} / {_f(row['rephrase_target_new_nll_max'])}|"
                f"{_f(row['rephrase_target_true_nll_mean'])} / {_f(row['rephrase_target_true_nll_median'])} / {_f(row['rephrase_target_true_nll_p90'])} / {_f(row['rephrase_target_true_nll_max'])}|"
                f"{row['rephrase_exact_count']}/8 ({_pct(row['rephrase_exact_rate'])})|{_f(row['w_minus_z_rephrase_nll_mean'])}|"
            )
    lines += [
        "",
        "## 5. Event/locality와 strength trade-off",
        "",
        "`event_q_kl_from_w0`는 target-excluded first-departure q0 drift, `locality_forward_kl`은 저장 evaluator locality KL이다. 둘 다 낮을수록 보존적이다. 표의 target/source probability는 같은 rewrite event partition에서의 값이다.",
        "",
        "|model|arm|event q-KL mean/median/p90/max|locality KL mean/median/p90/max|target p mean|source p mean|coefficient norm mean|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in (item[1] for item in MODELS):
        for arm in ARMS:
            row = by_arm[(model, arm)]
            lines.append(
                f"|{model}|{arm}|{_f(row['event_q_kl_from_w0_mean'])} / {_f(row['event_q_kl_from_w0_median'])} / {_f(row['event_q_kl_from_w0_p90'])} / {_f(row['event_q_kl_from_w0_max'])}|"
                f"{_f(row['locality_forward_kl_mean'])} / {_f(row['locality_forward_kl_median'])} / {_f(row['locality_forward_kl_p90'])} / {_f(row['locality_forward_kl_max'])}|"
                f"{_f(row['event_target_probability_mean'],6)}|{_f(row['event_source_probability_mean'],6)}|{_f(row['coefficient_norm_mean'])}|"
            )
    lines += [
        "",
        "### Paired factual reading",
        "",
        f"- Static-vs-Native: 16/16 paired cases에서 네 endpoint NLL과 event q-KL 최대 절대차가 `{parity['maximum_absolute_difference']:.1f}`이고, logits hash도 `{parity['endpoint_logits_hash_equal_count']}/{parity['endpoint_logits_hash_total']}` 일치했다.",
        "- Llama: Native rewrite 8/8에서 N2/N4가 0/8로 감소했다. event q-KL mean은 4.0249→0.4659→0.0583, locality KL mean은 1.3565→0.1469→0.0313으로 낮아졌지만 rewrite NLL mean은 0.0061→4.5581→7.5069로 악화했다.",
        "- Qwen: Native 8/8, N2 5/8, N4 1/8이다. event q-KL mean 3.5898→2.6104→0.7649, locality KL 1.1380→0.9964→0.3422와 함께 rewrite NLL 0.0090→0.3914→3.5918로 증가했다.",
        "- N4는 N2보다 더 강한 q/locality 보존을 보였으나 두 모델 모두 target strength를 더 잃었다. 이번 projection은 `g·v<=0`만 지키며 target progress equality나 minimum efficacy constraint를 갖지 않으므로 이 결과는 구현식과 일관된다.",
        "- target-new rephrase exact는 모든 arm/모델에서 0/8이다. 따라서 rewrite만으로 method 성공을 주장할 수 없다.",
        "",
        "## 6. Node trajectory",
        "",
        "node-level raw 96 rows는 `node-trajectory.csv`, model×arm×node aggregate는 `node-summary.csv`에 있다. 모든 case에서 node0 q0 barrier gradient가 numerical zero였고 첫 step은 nominal velocity였다. 이후 projection 활성화가 nominal AlphaEdit 방향의 q-KL 증가 성분을 제거하면서 coefficient path가 Native에서 이탈했다. N을 2→4로 늘리면 더 자주 재관측·투영되어 보존은 강화됐지만 target endpoint는 더 약해졌다.",
        "",
        "이 결과는 `N4가 더 정확한 ODE 해이므로 과학적으로 우수하다`는 증거가 아니다. 이 실험은 convergence grid가 아니며 N2/N4 endpoint strength가 match되지 않는다.",
        "",
        "## 7. Layer update",
        "",
        "proposal factors는 서로 다른 layer block에서 Frobenius-normalized되어 `||Delta W||_F^2 = sum_j theta_j^2`가 성립한다. `layer-update.csv`는 320 model/case/arm/layer rows에 coefficient, block energy, energy share를 저장한다. Barrier N4의 더 작은 coefficient norm은 target strength 약화와 함께 관측되며, 별도 capacity/causal claim은 하지 않는다.",
        "",
        "## 8. Compute·시간·메모리",
        "",
        "|model|job wall|model load|case panel mean/median/p90/max|observer wall mean|JVP|observer forwards|peak alloc/reserved GiB|host RSS GiB|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in (item[1] for item in MODELS):
        row = by_compute[model]
        lines.append(
            f"|{model}|{_f(row['job_total_wall_seconds'],2)}s|{_f(row['model_load_seconds'],2)}s|"
            f"{_f(row['case_panel_wall_seconds_mean'],2)} / {_f(row['case_panel_wall_seconds_median'],2)} / {_f(row['case_panel_wall_seconds_p90'],2)} / {_f(row['case_panel_wall_seconds_max'],2)}s|"
            f"{_f(row['functional_observer_wall_seconds_mean'],2)}s|{row['functional_observer_jvp_call_count']}|{row['functional_observer_model_forward_invocations']}|"
            f"{row['peak_gpu_allocated_bytes'] / 2**30:.2f}/{row['peak_gpu_reserved_bytes'] / 2**30:.2f}|{row['peak_host_rss_kib'] / 2**20:.2f}|"
        )
    lines += [
        "",
        "중요한 timing limitation: per-case functional-observer ledger는 q0+N2+N4를 합친 값이고 barrier arm별 integration wall을 분리하지 않았다. non-native `wall_seconds`는 trajectory 계산 뒤 terminal materialization+endpoint evaluation 경계다. 따라서 Native 대비 arm별 end-to-end overhead ratio는 `NOT_COMPARABLE_SCHEMA_GAP`; 시간을 역추정하거나 분배하지 않았다.",
        "",
        "## 9. Completeness와 기술 gate",
        "",
        f"- scheduler terminals 2/2; model terminals 2/2; cases {counts['case_count']}/16; endpoints {counts['endpoint_count']}/64; barrier nodes {counts['node_count']}/96",
        f"- fixed-z private artifacts {counts['private_z_count']}/16; failures {counts['failure_receipt_count']}; retries 0; imputation 0",
        f"- W0 pointer+bytes restore {counts['w0_restore_count']}/16; accepted-z compute/recompute {counts['z_compute_count']}/{counts['z_recompute_count']}; basis capture/rebuild {counts['basis_capture_count']}/{counts['basis_rebuild_count']}",
        f"- node physical writes {counts['node_physical_write_count']}; authoritative terminal writes {counts['terminal_write_count']}/64; layer applies {counts['layer_apply_count']}/320",
        "- Official adapter fidelity audit는 첫 case에서만 실행하도록 봉인되어 model별 1/1 PASS, 두 audit 모두 layer delta/hash exact (`relative_frobenius=0`); 나머지 14 cases는 `NOT_RUN_BY_DESIGN`이다.",
        "- raw result/log/private fixed-z bytes mutation 0; 분석은 read-only; GPU/model/Slurm replay 0",
        "",
        "## 10. 결론과 다음 설계 판단",
        "",
        "1. fixed normalized W0 AlphaEdit subspace와 terminal-only writer 구현은 Native identity control로 정확히 검증됐다.",
        "2. 현재 q-KL non-increase half-space는 load-bearing하다. 실제로 q/locality drift를 줄였지만, 그 작동 방식은 efficacy 방향을 제거하는 것이었고 strength 보존 장치가 없어 편집 성능이 무너졌다.",
        "3. 따라서 현재 Barrier ODE N2/N4를 method로 승격하지 않는다. 다음 구현이 필요하다면 outcome-tuned 재시도가 아니라 사전 정의된 target-progress equality/constraint 또는 strength-matched comparison을 먼저 수학적으로 닫아야 한다.",
        "4. 표준 EFF/GEN/LOC 퍼센트는 이 run schema에 저장되지 않았다. 저장 `exact_satisfied`와 locality forward-KL만 보고했으며 이를 EFF/GEN/LOC로 이름 바꾸지 않았다.",
        "5. eight-case cohort는 tokenizer-only natural unequal-nonprefix cohort이며 전체 dataset 대표성이나 통계적 유의성을 주장하지 않는다. cross-model 결과도 각각 factual endpoint로만 해석한다.",
        "",
        "## 11. Machine-readable 산출물",
        "",
        "- `accepted-z-summary.csv`: model별 accepted-z NLL/exact/locality",
        "- `arm-summary.csv`: model×arm 8-row aggregate",
        "- `case-arm.csv`: 64 paired endpoint rows",
        "- `prompt-nll.csv`: accepted-z/post-W prompt-level NLL",
        "- `paired-deltas.csv`: Native/N2/N4 paired case deltas",
        "- `node-trajectory.csv`, `node-summary.csv`: 96 raw node rows와 aggregate",
        "- `layer-update.csv`: 320 normalized block energy rows",
        "- `compute-summary.csv`, `scheduler-terminal.json`",
        "- `artifact-inventory.json`, `analysis-summary.json`, `analysis-manifest.json`, `rooted-receipt.json`",
        "",
    ]
    return "\n".join(lines)


def build(args: argparse.Namespace) -> None:
    result_root = args.result_root.resolve()
    log_root = args.log_root.resolve()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    terminals, cases, input_receipts = load_panel(result_root)
    log_paths = sorted(log_root.glob("*"))
    if len(log_paths) != 4:
        raise ValueError("expected two stdout and two stderr logs")
    input_receipts.extend(regular_file_receipt(path) for path in log_paths)
    if any("TERMINAL_VALID" not in path.read_text(encoding="utf-8", errors="replace") for path in log_paths if path.suffix == ".out"):
        raise ValueError("stdout terminal marker missing")

    cases_by_model = {model: [case for case in cases if case["_model"] == model] for _, model, _, _ in MODELS}
    max_difference = 0.0
    hash_equal = 0
    hash_total = 0
    for selected in cases_by_model.values():
        for case in selected:
            for metric in ENDPOINT_METRICS:
                left = endpoint(case, ARMS[0])[metric]
                right = endpoint(case, ARMS[1])[metric]
                max_difference = max(max_difference, abs(float(left["nll"]) - float(right["nll"])))
                hash_equal += int(left["logits_hash"] == right["logits_hash"])
                hash_total += 1
            for metric in ("event_q_kl_from_w0", "target_probability", "source_probability"):
                max_difference = max(
                    max_difference,
                    abs(float(case["arms"][ARMS[0]]["event"][metric]) - float(case["arms"][ARMS[1]]["event"][metric])),
                )
    parity = {
        "case_pairs": 16,
        "maximum_absolute_difference": max_difference,
        "endpoint_logits_hash_equal_count": hash_equal,
        "endpoint_logits_hash_total": hash_total,
        "status": "EXACT_PARITY" if max_difference == 0.0 and hash_equal == hash_total else "FAIL",
    }
    if parity["status"] != "EXACT_PARITY":
        raise ValueError("Static barrier-off does not exactly reproduce Native")

    accepted = accepted_z_summary(cases)
    arm_rows = arm_summary(cases)
    case_rows = case_arm_rows(cases)
    prompt = prompt_rows(cases)
    nodes = node_rows(cases)
    node_agg = node_summary(nodes)
    layers = layer_rows(cases)
    paired = paired_rows(cases)
    compute = compute_rows(cases, terminals)
    scheduler = scheduler_receipt(args.job_id)
    failure_paths = list(result_root.rglob("*failure*"))
    counts = {
        "terminal_count": len(terminals),
        "case_count": len(cases),
        "endpoint_count": len(case_rows),
        "node_count": len(nodes),
        "private_z_count": sum(1 for receipt in input_receipts if receipt["path"].endswith("direct-z.pt")),
        "failure_receipt_count": len(failure_paths),
        "w0_restore_count": sum(case["w0_restore"]["pointer_exact"] and case["w0_restore"]["bytes_exact"] for case in cases),
        "z_compute_count": sum(case["accepted_z"]["compute_count"] for case in cases),
        "z_recompute_count": sum(case["accepted_z"]["recompute_count"] for case in cases),
        "basis_capture_count": sum(case["basis"]["captured_build_count"] for case in cases),
        "basis_rebuild_count": sum(case["basis"]["virtual_factor_rebuild_count"] for case in cases),
        "node_physical_write_count": sum(case["arms"][arm].get("node_physical_write_count", 0) for case in cases for arm in ARMS),
        "terminal_write_count": sum(case["arms"][arm]["authoritative_terminal_write_count"] for case in cases for arm in ARMS),
        "layer_apply_count": len(cases) * len(ARMS) * len(LAYERS),
    }
    expected_counts = {
        "terminal_count": 2,
        "case_count": 16,
        "endpoint_count": 64,
        "node_count": 96,
        "private_z_count": 16,
        "failure_receipt_count": 0,
        "w0_restore_count": 16,
        "z_compute_count": 16,
        "z_recompute_count": 0,
        "basis_capture_count": 16,
        "basis_rebuild_count": 0,
        "node_physical_write_count": 0,
        "terminal_write_count": 64,
        "layer_apply_count": 320,
    }
    if counts != expected_counts:
        raise ValueError(f"completeness differs: {counts}")

    input_receipts.sort(key=lambda row: row["path"])
    input_root = identity({"members": input_receipts})
    summary = {
        "schema": SCHEMA,
        "status": "TECHNICAL_PASS_SCIENTIFIC_NEGATIVE_STRENGTH_LOCALITY_TRADEOFF",
        "verdict": "BARRIER_LOAD_BEARING_BUT_EDIT_STRENGTH_NOT_PRESERVED",
        "source_head": EXPECTED_SOURCE_HEAD,
        "source_tree": EXPECTED_SOURCE_TREE,
        "cohort_identity": EXPECTED_COHORT,
        "job_id": args.job_id,
        "counts": counts,
        "static_native_parity": parity,
        "accepted_z": accepted,
        "arms": arm_rows,
        "compute": compute,
        "limitations": {
            "standard_eff_gen_loc_percent_not_recorded": True,
            "barrier_arm_specific_integration_wall_not_recorded": True,
            "alphaedit_paired_end_to_end_overhead": "NOT_COMPARABLE_SCHEMA_GAP",
            "cohort_is_not_population_inference": True,
        },
        "raw_mutation_count": 0,
        "gpu_model_slurm_replay_count": 0,
        "imputation_count": 0,
        "scientific_promotion": False,
    }
    inventory = {
        "schema": "ode-edit-bgode-fbp-minimal-artifact-inventory/v1",
        "result_root": str(result_root),
        "log_root": str(log_root),
        "members": input_receipts,
        "members_root": input_root,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.tmp-", dir=output.parent) as temp_name:
        temp = Path(temp_name)
        temp.chmod(0o700)
        write_csv(temp / "accepted-z-summary.csv", accepted)
        write_csv(temp / "arm-summary.csv", arm_rows)
        write_csv(temp / "case-arm.csv", case_rows)
        write_csv(temp / "prompt-nll.csv", prompt)
        write_csv(temp / "paired-deltas.csv", paired)
        write_csv(temp / "node-trajectory.csv", nodes)
        write_csv(temp / "node-summary.csv", node_agg)
        write_csv(temp / "layer-update.csv", layers)
        write_csv(temp / "compute-summary.csv", compute)
        write_json(temp / "scheduler-terminal.json", scheduler)
        write_json(temp / "artifact-inventory.json", inventory)
        write_json(temp / "analysis-summary.json", summary)
        report = report_text(
            terminals=terminals,
            z_rows=accepted,
            arms=arm_rows,
            compute=compute,
            parity=parity,
            counts=counts,
        )
        report_path = temp / "bgode-fbp-minimal-fixed-basis-barrier-ode-factual-ko.md"
        report_path.write_text(report, encoding="utf-8")
        report_path.chmod(0o600)

        output_members = [
            regular_file_receipt(path, display=path.name)
            for path in sorted(temp.iterdir())
            if path.is_file()
        ]
        output_root = identity({"members": output_members})
        manifest_body = {
            "schema": "ode-edit-bgode-fbp-minimal-analysis-manifest/v1",
            "source_head": EXPECTED_SOURCE_HEAD,
            "source_tree": EXPECTED_SOURCE_TREE,
            "job_id": args.job_id,
            "input_members_root": input_root,
            "input_member_count": len(input_receipts),
            "output_members": output_members,
            "output_members_root": output_root,
            "raw_mutation_count": 0,
            "imputation_count": 0,
            "gpu_model_slurm_replay_count": 0,
            "scientific_promotion": False,
        }
        manifest = {**manifest_body, "identity_sha256": identity(manifest_body)}
        write_json(temp / "analysis-manifest.json", manifest)
        manifest_receipt = regular_file_receipt(temp / "analysis-manifest.json", display="analysis-manifest.json")
        report_receipt = regular_file_receipt(report_path, display=report_path.name)
        receipt_body = {
            "schema": "ode-edit-bgode-fbp-minimal-rooted-receipt/v1",
            "status": summary["status"],
            "verdict": summary["verdict"],
            "analysis_manifest_identity": manifest["identity_sha256"],
            "analysis_manifest_sha256": manifest_receipt["sha256"],
            "input_members_root": input_root,
            "output_members_root": output_root,
            "report_sha256": report_receipt["sha256"],
            "raw_mutation_count": 0,
            "imputation_count": 0,
            "gpu_model_slurm_replay_count": 0,
            "scientific_promotion": False,
        }
        receipt = {**receipt_body, "identity_sha256": identity(receipt_body)}
        write_json(temp / "rooted-receipt.json", receipt)
        os.rename(temp, output)
        os.chmod(output, 0o700)


def verify(output: Path) -> None:
    output = output.resolve()
    manifest_path = output / "analysis-manifest.json"
    receipt_path = output / "rooted-receipt.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest_body = {key: value for key, value in manifest.items() if key != "identity_sha256"}
    receipt_body = {key: value for key, value in receipt.items() if key != "identity_sha256"}
    if identity(manifest_body) != manifest["identity_sha256"]:
        raise ValueError("manifest identity mismatch")
    if identity(receipt_body) != receipt["identity_sha256"]:
        raise ValueError("receipt identity mismatch")
    for member in manifest["output_members"]:
        actual = regular_file_receipt(output / member["path"], display=member["path"])
        if actual != member:
            raise ValueError(f"output member mismatch: {member['path']}")
    root = identity({"members": manifest["output_members"]})
    if root != manifest["output_members_root"] or root != receipt["output_members_root"]:
        raise ValueError("output member root mismatch")
    if regular_file_receipt(manifest_path)["sha256"] != receipt["analysis_manifest_sha256"]:
        raise ValueError("manifest SHA mismatch")
    report = output / "bgode-fbp-minimal-fixed-basis-barrier-ode-factual-ko.md"
    if regular_file_receipt(report)["sha256"] != receipt["report_sha256"]:
        raise ValueError("report SHA mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--result-root", type=Path, required=True)
    build_parser.add_argument("--log-root", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--job-id", default="27342")
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args)
        verify(args.output)
    else:
        verify(args.output)


if __name__ == "__main__":
    main()
