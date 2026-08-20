#!/usr/bin/env python3
"""Build the raw-free P1R52 IL1/3/8/10/15 depth-extension report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PHASE1B = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh2-s05-p1r52-target-depth-il1-il3full-v1/local/odebf/reports/"
    "p1r52-target-depth-phase1b-atomic-v3"
)
RESULT_PARENT = REPO / "local/odebf/results"
SOURCE_HEAD = "8d94a8c96db99c5ffdadbcd13f6c0347e9f2ca8d"
SOURCE_TREE = "d974ee426a033285d5efaa7dcfa7f1e3a8effaf6"
CONTRACT_SHA256 = "06e65d4a2df4a4610ef09818c2afd96b8dbe86cc3041afdbc5b099ca2358ec63"
STREAM_ROOT = "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6"
ORDER_SHA256 = "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c"
NUMERICAL_LOCK_SHA256 = "82b9c996287b8d42d27193ac14ad765d158d32367f1c00a027aadc98987bcbdb"
REFERENCE_SHA256 = {
    "il1_report": "9434cb5ab4ae650b5e72226d6833ea46960d2303a4634a249a63018e9509a5d6",
    "il1_artifact_index": "6cedac885cea1e35f3e2ad2f5b827b96a25175deb303745a949fa1bff7a0d129",
    "il3_report": "4925b7e0219f5cc17788cf674ff0aa81c23f94a111176bf3e25ccc59fa64b3ec",
    "il3_manifest": "9e0958855505fec4b33362551e7710a800bf7f5df02330924fd8618a5a16cbdb",
    "il3_group_summary": "c7024e39d143af9b9700614b4c54fb5e9e9e04c2045ae5a6cac1d9b45451c5b6",
    "il3_per_case": "dde1c6e7015d706fb18cce41c15a94a319409396f18b89345cb0a56e6f7061d7",
    "il3_per_request": "46daeca58465a974a96dfa43b6a539cc151e07207248e653d215abb99373b3bf",
    "il3_per_step": "e3e1d8f7c7459595274154326e5aa5672bd8a0d54eb0ee91109e425dce352ff0",
}
DEPTHS = (8, 10, 15)
ROOTS = {
    depth: RESULT_PARENT
    / (
        "s05-p1r52-target-depth-atomic-b10x10-llama3-8b-inst-soft-"
        f"il{depth}full-extension-inner-telemetry-tech-r1-v1"
    )
    for depth in DEPTHS
}
EXPECTED_INNER_ROWS = {8: 64, 10: 80, 15: 120}
REPORT = HERE / "p1r52-target-depth-il1-il3-il8-il10-il15-terminal-report-ko.md"


def verify_references() -> None:
    paths = {
        "il1_report": REPO
        / "experiment-reports/global/2026-08-17-p1r52-rsa-r42safekdc-m1-repair-r1-final-ko.md",
        "il1_artifact_index": REPO
        / "experiment-reports/global/2026-08-17-p1r52-rsa-r42safekdc-m1-repair-r1-artifacts.json",
        "il3_report": PHASE1B
        / "p1r52-target-depth-phase1b-atomic-terminal-report-ko.md",
        "il3_manifest": PHASE1B / "manifest.json",
        "il3_group_summary": PHASE1B / "group-summary.json",
        "il3_per_case": PHASE1B / "per-case.json",
        "il3_per_request": PHASE1B / "per-request.json",
        "il3_per_step": PHASE1B / "per-step.json",
    }
    for label, path in paths.items():
        observed = sha256_file(path)
        if observed != REFERENCE_SHA256[label]:
            raise RuntimeError(f"immutable reference SHA differs: {label}")
    lock_path = (
        REPO
        / "project/run_scripts/ode_bf/locks/numerical_lock_s05_p1r52_target_depth_il8_il10_il15.json"
    )
    if sha256_file(lock_path) != NUMERICAL_LOCK_SHA256:
        raise RuntimeError("numerical lock SHA differs")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required input absent: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> str:
    return write_once(
        path,
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    fields = sorted({key for row in rows for key in row})
    lines: list[str] = []
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
        )
    return write_once(path, buffer.getvalue().encode())


def percentile(values: Sequence[float], q: float) -> float | str:
    if not values:
        return "NOT_RECORDED"
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def stats(values: Iterable[float]) -> dict[str, float | str | int]:
    observed = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(observed),
        "mean": statistics.fmean(observed) if observed else "NOT_RECORDED",
        "median": statistics.median(observed) if observed else "NOT_RECORDED",
        "p10": percentile(observed, 0.10),
        "p90": percentile(observed, 0.90),
        "max": max(observed) if observed else "NOT_RECORDED",
    }


def flatten(groups: Sequence[Sequence[float]]) -> list[float]:
    return [float(value) for group in groups for value in group]


def layer_sum(value: Mapping[str, float]) -> float:
    return sum(float(item) for item in value.values())


def raw_metric(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return raw["primary"]["metrics"][name]


def numeric_groups(raw: Mapping[str, Any], name: str) -> Sequence[Sequence[Mapping[str, Any]]]:
    return raw["numeric_vectors"][name]


@lru_cache(maxsize=1)
def il3_request_id_map() -> dict[tuple[int, int], str]:
    rows = {
        (int(row["case_index"]), int(row["request_index"])): row["request_sha256"]
        for row in load(PHASE1B / "per-request.json")
        if row["alias"] == "llama3-8b-inst" and row["arm"] == "soft"
    }
    if len(rows) != 100:
        raise RuntimeError("immutable IL3 request identity denominator differs")
    return rows


def terminal_request_rows(
    depth: int, case_index: int, terminal: Mapping[str, Any]
) -> list[dict[str, Any]]:
    panel = terminal["terminal_four_panel"]
    z_raw = panel["z_inject"]
    w_raw = panel["weight"]
    rows = []
    for request_index in range(10):
        z_eff = numeric_groups(z_raw, "efficacy")[request_index]
        z_gen = numeric_groups(z_raw, "generalization")[request_index]
        w_eff = numeric_groups(w_raw, "efficacy")[request_index]
        w_gen = numeric_groups(w_raw, "generalization")[request_index]
        w_loc = numeric_groups(w_raw, "locality-preservation")[request_index]
        z_eff_nll = float(z_eff[0]["nll_new"])
        w_eff_nll = float(w_eff[0]["nll_new"])
        z_gen_nll = [float(item["nll_new"]) for item in z_gen]
        w_gen_nll = [float(item["nll_new"]) for item in w_gen]
        z_eff_metric = raw_metric(z_raw, "efficacy")
        z_gen_metric = raw_metric(z_raw, "generalization")
        w_eff_metric = raw_metric(w_raw, "efficacy")
        w_gen_metric = raw_metric(w_raw, "generalization")
        w_loc_metric = raw_metric(w_raw, "locality-preservation")
        rows.append(
            {
                "depth": f"IL{depth}-FULL",
                "case_index": case_index,
                "request_index": request_index,
                "request_sha256": il3_request_id_map()[(case_index, request_index)],
                "z_rewrite_nll": z_eff_nll,
                "z_rephrase_nll": z_gen_nll,
                "w_rewrite_nll": w_eff_nll,
                "w_rephrase_nll": w_gen_nll,
                "z_rewrite_margin": float(z_eff[0]["margin"]),
                "z_rephrase_margin": [float(item["margin"]) for item in z_gen],
                "w_rewrite_margin": float(w_eff[0]["margin"]),
                "w_rephrase_margin": [float(item["margin"]) for item in w_gen],
                "rewrite_w_minus_z_nll_gap": w_eff_nll - z_eff_nll,
                "rephrase_w_minus_z_nll_gap": statistics.fmean(w_gen_nll)
                - statistics.fmean(z_gen_nll),
                "z_rewrite_success": int(
                    z_eff_metric["per_case_correct"][request_index]
                    == z_eff_metric["per_case_required"][request_index]
                ),
                "z_rephrase_strict_success": int(
                    z_gen_metric["per_case_correct"][request_index]
                    == z_gen_metric["per_case_required"][request_index]
                ),
                "w_rewrite_success": int(
                    w_eff_metric["per_case_correct"][request_index]
                    == w_eff_metric["per_case_required"][request_index]
                ),
                "w_rephrase_strict_success": int(
                    w_gen_metric["per_case_correct"][request_index]
                    == w_gen_metric["per_case_required"][request_index]
                ),
                "w_locality_success": int(
                    w_loc_metric["per_case_correct"][request_index]
                    == w_loc_metric["per_case_required"][request_index]
                ),
                "w_locality_nll": [float(item["nll_new"]) for item in w_loc],
                "accuracy": "NOT_RECORDED",
            }
        )
    return rows


def extract_new() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    inners: list[dict[str, Any]] = []
    inner_requests: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for depth, root in ROOTS.items():
        group = load(root / "terminal.json")
        if group["source_head"] != SOURCE_HEAD or group["configured_inner_count"] != depth:
            raise RuntimeError(f"group source/depth differs: IL{depth}")
        if group["completed_case_count"] + group["failed_case_count"] != 10:
            raise RuntimeError(f"group denominator differs: IL{depth}")
        for case_index in range(1, 11):
            case_root = root / "raw/cases" / f"case-{case_index:02d}"
            terminal_path = case_root / "terminal.json"
            if not terminal_path.is_file():
                failure = load(case_root / "failure.json")
                failures.append({"depth": f"IL{depth}-FULL", **failure})
                continue
            terminal = load(terminal_path)
            if not terminal["W0_restored"]:
                raise RuntimeError("W0 restore differs")
            accepted_root = case_root / "raw/ode/p1r52-rsa-r42safekdc-m1-soft"
            accepted = [load(accepted_root / f"accepted-k{k}.json") for k in range(1, 9)]
            executed_inner_rows = sum(
                int(item["target_update"]["executed_inner_count"])
                for item in accepted
            )
            if (
                terminal["rollout"]["target_depth_inner_telemetry_row_count"]
                != executed_inner_rows
            ):
                raise RuntimeError("inner telemetry/executed row count differs")
            if executed_inner_rows > EXPECTED_INNER_ROWS[depth]:
                raise RuntimeError("inner row count exceeds configured maximum")
            for item in accepted:
                target_update = item["target_update"]
                trajectory = target_update["inner_trajectory"]
                if target_update["configured_inner_count"] != depth:
                    raise RuntimeError("configured inner count differs")
                if len(trajectory) != target_update["executed_inner_count"]:
                    raise RuntimeError("inner trajectory/executed count differs")
                if len(trajectory) < depth:
                    if not target_update["early_stop_byte_identical"]:
                        raise RuntimeError("short inner trajectory without typed early stop")
                    if len(trajectory) < 2 or (
                        trajectory[-1]["target_next_sha256"]
                        != trajectory[-2]["target_next_sha256"]
                    ):
                        raise RuntimeError("early stop byte-identity certificate differs")
            if terminal["rollout"]["target_depth_outer_telemetry_row_count"] != 8:
                raise RuntimeError("outer row count differs")
            if terminal["rollout"]["target_depth_duplicate_evaluation_count"] != 0:
                raise RuntimeError("duplicate evaluator count differs")
            requests.extend(terminal_request_rows(depth, case_index, terminal))
            predicted = sum(float(item["progress"]["predicted"]) for item in accepted)
            delayed = {
                int(row["transition_index"]): row
                for row in terminal["rollout"]["delayed_progress"]
            }
            step_actuals = []
            for transition, item in enumerate(accepted, start=1):
                if transition in delayed:
                    observed_actual = float(delayed[transition]["actual"])
                elif transition == 8:
                    observed_actual = float(item["physical_slope"]["mean_objective_value"]) - float(
                        terminal["terminal_w8_full_six_target_new_nll"]
                    )
                else:
                    raise RuntimeError("delayed actual progress coverage differs")
                step_actuals.append(observed_actual)
            actual = sum(step_actuals)
            path_energy = sum(
                layer_sum(item["materialization"]["realized_bf16_step_energy"])
                for item in accepted
            )
            endpoint_capacity = layer_sum(
                accepted[-1]["materialization"]["cumulative_bf16_capacity"]
            )
            outer_last = accepted[-1]["target_depth_outer_terminal_telemetry"]
            compute = terminal["rollout"]["compute"]
            final_inner_marginals = []
            final_inner_movements = []
            for item in accepted:
                trajectory = item["target_update"]["inner_trajectory"]
                final_inner_movements.append(
                    float(trajectory[-1]["accepted_z_observation"]["inner_step_movement_norm"])
                )
                if len(trajectory) > 1:
                    final_inner_marginals.append(
                        float(trajectory[-1]["accepted_z_observation"]["target_objective_nll_mean"])
                        - float(trajectory[-2]["accepted_z_observation"]["target_objective_nll_mean"])
                    )
            cases.append(
                {
                    "depth": f"IL{depth}-FULL",
                    "case_index": case_index,
                    "classification": "VALID_ENDPOINT",
                    "W0_restored": True,
                    "action_freeze_sha256": terminal["action_freeze_sha256"],
                    "terminal_sha256": sha256_file(terminal_path),
                    "terminal_identity_sha256": terminal["identity_sha256"],
                    "inner_row_count": terminal["rollout"]["target_depth_inner_telemetry_row_count"],
                    "configured_inner_row_maximum": EXPECTED_INNER_ROWS[depth],
                    "byte_identical_early_stop_outer_count": sum(
                        int(item["target_update"]["early_stop_byte_identical"])
                        for item in accepted
                    ),
                    "outer_row_count": terminal["rollout"]["target_depth_outer_telemetry_row_count"],
                    "z_full_six_nll": float(terminal["terminal_z8_oracle"]["target_new_nll"]),
                    "w_full_six_nll": float(terminal["terminal_w8_full_six_target_new_nll"]),
                    "full_six_w_minus_z_gap": float(terminal["terminal_w8_full_six_target_new_nll"])
                    - float(terminal["terminal_z8_oracle"]["target_new_nll"]),
                    "z_eff_numerator": outer_last["accepted_z_efficacy"]["numerator"],
                    "z_eff_denominator": outer_last["accepted_z_efficacy"]["denominator"],
                    "z_gen_numerator": outer_last["accepted_z_generalization"]["numerator"],
                    "z_gen_denominator": outer_last["accepted_z_generalization"]["denominator"],
                    "z_gen_strict_numerator": outer_last["accepted_z_generalization"]["strict_request_count"],
                    "w_eff_numerator": outer_last["writer_w_efficacy"]["numerator"],
                    "w_eff_denominator": outer_last["writer_w_efficacy"]["denominator"],
                    "w_gen_numerator": outer_last["writer_w_generalization"]["numerator"],
                    "w_gen_denominator": outer_last["writer_w_generalization"]["denominator"],
                    "w_gen_strict_numerator": outer_last["writer_w_generalization"]["strict_request_count"],
                    "loc_numerator": outer_last["writer_w_locality"]["numerator"],
                    "loc_denominator": outer_last["writer_w_locality"]["denominator"],
                    "z_rewrite_nll_mean": statistics.fmean(outer_last["accepted_z_rewrite_nll_by_request"]),
                    "z_rephrase_nll_mean": statistics.fmean(flatten(outer_last["accepted_z_rephrase_nll_by_request"])),
                    "w_rewrite_nll_mean": statistics.fmean(outer_last["writer_w_rewrite_nll_by_request"]),
                    "w_rephrase_nll_mean": statistics.fmean(flatten(outer_last["writer_w_rephrase_nll_by_request"])),
                    "rewrite_w_minus_z_gap_mean": outer_last["rewrite_w_minus_z_nll_gap_mean"],
                    "rephrase_w_minus_z_gap_mean": outer_last["rephrase_w_minus_z_nll_gap_mean"],
                    "predicted_progress_sum": predicted,
                    "actual_progress_sum": actual,
                    "realization_ratio": actual / predicted if predicted else "NOT_RECORDED",
                    "negative_actual_count": sum(int(value < 0) for value in step_actuals),
                    "terminal_cumulative_structural_p": accepted[-1]["cumulative_atomic_structural_p"]["P_after"],
                    "selected_capacity_mean": statistics.fmean(float(item["routing"]["selected_capacity"]) for item in accepted),
                    "selected_energy_mean": statistics.fmean(float(item["routing"]["selected_energy"]) for item in accepted),
                    "bf16_path_energy": path_energy,
                    "cumulative_bf16_capacity_endpoint": endpoint_capacity,
                    "simplex_entropy_mean": statistics.fmean(float(item["routing"]["simplex_entropy"]) for item in accepted),
                    "simplex_top1_share_mean": statistics.fmean(float(item["routing"]["simplex_top1_share"]) for item in accepted),
                    "velocity_abs_top1_share_mean": statistics.fmean(
                        max(abs(float(value)) for value in item["routing"]["velocity"])
                        / max(
                            sum(abs(float(value)) for value in item["routing"]["velocity"]),
                            1e-30,
                        )
                        for item in accepted
                    ),
                    "layer_energy_top1_share_mean": statistics.fmean(
                        max(item["materialization"]["realized_bf16_step_energy"].values())
                        / max(
                            layer_sum(item["materialization"]["realized_bf16_step_energy"]),
                            1e-30,
                        )
                        for item in accepted
                    ),
                    "outer_target_path_norm_sum": sum(float(item["target_update"]["outer_net_target_displacement_norm"]) for item in accepted),
                    "inner_target_path_norm_sum": sum(
                        float(inner["accepted_z_observation"]["inner_step_movement_norm"])
                        for item in accepted
                        for inner in item["target_update"]["inner_trajectory"]
                    ),
                    "final_inner_target_objective_nll_delta_mean": statistics.fmean(
                        final_inner_marginals
                    ),
                    "final_inner_movement_norm_mean": statistics.fmean(
                        final_inner_movements
                    ),
                    "model_forward_calls": compute["totals"]["model_forward_calls"],
                    "model_backward_calls": compute["totals"]["backward_calls"],
                    "processed_tokens": compute["totals"]["processed_tokens"],
                    "materializations": compute["totals"]["materialization_count"],
                    "edit_core_wall_seconds": terminal["rollout"]["edit_core_wall_seconds"],
                    "terminal_evaluator_wall_seconds": terminal["terminal_evaluator_wall_seconds"],
                    "target_gradient_wall_seconds": compute["wall_seconds"].get("target_gradient", 0.0),
                    "inner_observation_wall_seconds": compute["wall_seconds"].get("target_depth_inner_z_telemetry", 0.0),
                    "outer_observation_wall_seconds": compute["wall_seconds"].get("target_depth_outer_w_z_telemetry", 0.0),
                    "inner_observation_forward_calls": compute["phases"]["target_depth_inner_z_telemetry"]["model_forward_calls"],
                    "outer_observation_forward_calls": compute["phases"]["target_depth_outer_w_z_telemetry"]["model_forward_calls"],
                    "accuracy": "NOT_RECORDED",
                }
            )
            for transition, (item, observed_actual) in enumerate(
                zip(accepted, step_actuals), start=1
            ):
                outer = item["target_depth_outer_terminal_telemetry"]
                material = item["materialization"]
                step_row = {
                    "depth": f"IL{depth}-FULL",
                    "case_index": case_index,
                    "transition_index": transition,
                    "inner_count": len(item["target_update"]["inner_trajectory"]),
                    "configured_inner_count": depth,
                    "early_stop_byte_identical": item["target_update"]["early_stop_byte_identical"],
                    "predicted_progress": item["progress"]["predicted"],
                    "actual_progress": observed_actual,
                    "realization_ratio": observed_actual
                    / float(item["progress"]["predicted"])
                    if float(item["progress"]["predicted"])
                    else "NOT_RECORDED",
                    "negative_actual": observed_actual < 0,
                    "outer_net_target_displacement_norm": item["target_update"]["outer_net_target_displacement_norm"],
                    "final_full_current_residual_norm": item["target_update"]["final_full_current_residual_norm"],
                    "structural_p": item["structural_p"],
                    "cumulative_structural_p": item["cumulative_atomic_structural_p"]["P_after"],
                    "selected_capacity": item["routing"]["selected_capacity"],
                    "selected_energy": item["routing"]["selected_energy"],
                    "simplex_entropy": item["routing"]["simplex_entropy"],
                    "simplex_top1_share": item["routing"]["simplex_top1_share"],
                    "velocity_abs_top1_share": max(
                        abs(float(value)) for value in item["routing"]["velocity"]
                    )
                    / max(
                        sum(abs(float(value)) for value in item["routing"]["velocity"]),
                        1e-30,
                    ),
                    "realized_bf16_step_energy": layer_sum(material["realized_bf16_step_energy"]),
                    "layer_energy_top1_share": max(material["realized_bf16_step_energy"].values())
                    / max(layer_sum(material["realized_bf16_step_energy"]), 1e-30),
                    "z_rewrite_nll_mean": statistics.fmean(outer["accepted_z_rewrite_nll_by_request"]),
                    "z_rephrase_nll_mean": statistics.fmean(flatten(outer["accepted_z_rephrase_nll_by_request"])),
                    "w_rewrite_nll_mean": statistics.fmean(outer["writer_w_rewrite_nll_by_request"]),
                    "w_rephrase_nll_mean": statistics.fmean(flatten(outer["writer_w_rephrase_nll_by_request"])),
                    "rewrite_w_minus_z_gap_mean": outer["rewrite_w_minus_z_nll_gap_mean"],
                    "rephrase_w_minus_z_gap_mean": outer["rephrase_w_minus_z_nll_gap_mean"],
                    "identity_sha256": item["identity_sha256"],
                }
                steps.append(step_row)
                for inner in item["target_update"]["inner_trajectory"]:
                    observation = inner["accepted_z_observation"]
                    selection = list(inner["selection_by_request"])
                    inner_row = {
                        "depth": f"IL{depth}-FULL",
                        "case_index": case_index,
                        "outer_step_index": observation["outer_step_index"],
                        "inner_index": observation["inner_index"],
                        "global_target_update_ordinal": observation["global_target_update_ordinal"],
                        "target_objective_nll_mean": observation["target_objective_nll_mean"],
                        "selected_endpoint_nll": inner["selected_endpoint_nll"],
                        "primary_selection_count": selection.count("PRIMARY"),
                        "rescue_selection_count": selection.count("RESCUE"),
                        "current_selection_count": selection.count("CURRENT"),
                        "accepted_z_rewrite_nll": stats(observation["accepted_z_rewrite_nll_by_request"]),
                        "accepted_z_rephrase_nll": stats(flatten(observation["accepted_z_rephrase_nll_by_request"])),
                        "inner_step_movement_norm": observation["inner_step_movement_norm"],
                        "cumulative_outer_entry_movement_norm": observation["cumulative_outer_entry_movement_norm"],
                        "z_eff_numerator": observation["accepted_z_efficacy"]["numerator"],
                        "z_gen_numerator": observation["accepted_z_generalization"]["numerator"],
                        "z_gen_strict_numerator": observation["accepted_z_generalization"]["strict_request_count"],
                        "added_model_forward_count": observation["added_model_forward_count"],
                        "added_backward_count": observation["added_backward_count"],
                        "added_generation_count": observation["added_generation_count"],
                        "controller_action_influence_count": observation["controller_action_influence_count"],
                        "duplicate_evaluation_count": observation["duplicate_evaluation_count"],
                        "identity_sha256": observation["identity_sha256"],
                    }
                    inners.append(inner_row)
                    for request_index in range(10):
                        inner_requests.append(
                            {
                                "depth": f"IL{depth}-FULL",
                                "case_index": case_index,
                                "outer_step_index": observation["outer_step_index"],
                                "inner_index": observation["inner_index"],
                                "global_target_update_ordinal": observation["global_target_update_ordinal"],
                                "request_index": request_index,
                                "target_objective_nll": observation["target_objective_nll_by_request"][request_index],
                                "accepted_z_rewrite_nll": observation["accepted_z_rewrite_nll_by_request"][request_index],
                                "accepted_z_rephrase_nll": observation["accepted_z_rephrase_nll_by_request"][request_index],
                                "inner_step_movement_norm": observation["inner_step_movement_norm_by_request"][request_index],
                                "cumulative_outer_entry_movement_norm": observation["cumulative_outer_entry_movement_norm_by_request"][request_index],
                            }
                        )
    return cases, steps, inners, inner_requests, requests, failures


def new_summary(
    depth: int,
    cases: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    label = f"IL{depth}-FULL"
    own_cases = [row for row in cases if row["depth"] == label]
    own_requests = [row for row in requests if row["depth"] == label]
    own_failures = [row for row in failures if row["depth"] == label]
    technical_failures = sum(
        row.get("classification") == "TECHNICAL_FAIL" for row in own_failures
    )
    scientific_failures = len(own_failures) - technical_failures
    predicted_sum = sum(row["predicted_progress_sum"] for row in own_cases)
    actual_sum = sum(row["actual_progress_sum"] for row in own_cases)
    return {
        "depth": label,
        "attempt_count": 10,
        "endpoint_count": len(own_cases),
        "typed_scientific_failure_count": scientific_failures,
        "technical_failure_count": technical_failures,
        "request_endpoint_count": len(own_requests),
        "executed_inner_telemetry_rows": sum(row["inner_row_count"] for row in own_cases),
        "configured_inner_telemetry_row_maximum": sum(
            row["configured_inner_row_maximum"] for row in own_cases
        ),
        "byte_identical_early_stop_outer_count": sum(
            row["byte_identical_early_stop_outer_count"] for row in own_cases
        ),
        "z_eff_numerator": sum(row["z_eff_numerator"] for row in own_cases),
        "z_eff_denominator": sum(row["z_eff_denominator"] for row in own_cases),
        "z_gen_numerator": sum(row["z_gen_numerator"] for row in own_cases),
        "z_gen_denominator": sum(row["z_gen_denominator"] for row in own_cases),
        "z_gen_strict_numerator": sum(row["z_gen_strict_numerator"] for row in own_cases),
        "w_eff_numerator": sum(row["w_eff_numerator"] for row in own_cases),
        "w_eff_denominator": sum(row["w_eff_denominator"] for row in own_cases),
        "w_gen_numerator": sum(row["w_gen_numerator"] for row in own_cases),
        "w_gen_denominator": sum(row["w_gen_denominator"] for row in own_cases),
        "w_gen_strict_numerator": sum(row["w_gen_strict_numerator"] for row in own_cases),
        "loc_numerator": sum(row["loc_numerator"] for row in own_cases),
        "loc_denominator": sum(row["loc_denominator"] for row in own_cases),
        "z_full_six_nll": stats(row["z_full_six_nll"] for row in own_cases),
        "w_full_six_nll": stats(row["w_full_six_nll"] for row in own_cases),
        "full_six_w_minus_z_gap": stats(row["full_six_w_minus_z_gap"] for row in own_cases),
        "z_rewrite_nll": stats(row["z_rewrite_nll"] for row in own_requests),
        "z_rephrase_nll": stats(
            statistics.fmean(row["z_rephrase_nll"]) for row in own_requests
        ),
        "w_rewrite_nll": stats(row["w_rewrite_nll"] for row in own_requests),
        "w_rephrase_nll": stats(
            statistics.fmean(row["w_rephrase_nll"]) for row in own_requests
        ),
        "z_rewrite_margin": stats(row["z_rewrite_margin"] for row in own_requests),
        "z_rephrase_margin": stats(
            statistics.fmean(row["z_rephrase_margin"]) for row in own_requests
        ),
        "w_rewrite_margin": stats(row["w_rewrite_margin"] for row in own_requests),
        "w_rephrase_margin": stats(
            statistics.fmean(row["w_rephrase_margin"]) for row in own_requests
        ),
        "rewrite_w_minus_z_gap": stats(row["rewrite_w_minus_z_nll_gap"] for row in own_requests),
        "rephrase_w_minus_z_gap": stats(row["rephrase_w_minus_z_nll_gap"] for row in own_requests),
        "predicted_progress_sum": predicted_sum,
        "actual_progress_sum": actual_sum,
        "realization_ratio": actual_sum / predicted_sum
        if predicted_sum
        else "NOT_RECORDED",
        "negative_actual_count": sum(row["negative_actual_count"] for row in own_cases),
        "terminal_cumulative_structural_p": stats(row["terminal_cumulative_structural_p"] for row in own_cases),
        "cumulative_bf16_capacity_endpoint": stats(row["cumulative_bf16_capacity_endpoint"] for row in own_cases),
        "bf16_path_energy": stats(row["bf16_path_energy"] for row in own_cases),
        "simplex_top1_share": stats(row["simplex_top1_share_mean"] for row in own_cases),
        "velocity_abs_top1_share": stats(
            row["velocity_abs_top1_share_mean"] for row in own_cases
        ),
        "layer_energy_top1_share": stats(
            row["layer_energy_top1_share_mean"] for row in own_cases
        ),
        "outer_target_path_norm_sum": stats(row["outer_target_path_norm_sum"] for row in own_cases),
        "inner_target_path_norm_sum": stats(row["inner_target_path_norm_sum"] for row in own_cases),
        "final_inner_target_objective_nll_delta": stats(
            row["final_inner_target_objective_nll_delta_mean"] for row in own_cases
        ),
        "final_inner_movement_norm": stats(
            row["final_inner_movement_norm_mean"] for row in own_cases
        ),
        "model_forward_calls": sum(row["model_forward_calls"] for row in own_cases),
        "model_backward_calls": sum(row["model_backward_calls"] for row in own_cases),
        "processed_tokens": sum(row["processed_tokens"] for row in own_cases),
        "materializations": sum(row["materializations"] for row in own_cases),
        "edit_core_wall_seconds": stats(row["edit_core_wall_seconds"] for row in own_cases),
        "terminal_evaluator_wall_seconds": stats(row["terminal_evaluator_wall_seconds"] for row in own_cases),
        "target_gradient_wall_seconds": stats(row["target_gradient_wall_seconds"] for row in own_cases),
        "inner_observation_wall_seconds": stats(row["inner_observation_wall_seconds"] for row in own_cases),
        "outer_observation_wall_seconds": stats(row["outer_observation_wall_seconds"] for row in own_cases),
        "accuracy": "NOT_RECORDED",
    }


def il1_il3_summaries() -> list[dict[str, Any]]:
    group = load(PHASE1B / "group-summary.json")
    il3 = next(
        row for row in group if row["alias"] == "llama3-8b-inst" and row["arm"] == "soft"
    )
    comparison = next(
        row
        for row in load(PHASE1B / "il3-minus-il1-comparison.json")
        if row["alias"] == "llama3-8b-inst" and row["arm"] == "soft"
    )
    il3_requests = [
        row
        for row in load(PHASE1B / "per-request.json")
        if row["alias"] == "llama3-8b-inst" and row["arm"] == "soft"
    ]
    if len(il3_requests) != 100:
        raise RuntimeError("immutable IL3 request denominator differs")
    il3_cases = [
        row
        for row in load(PHASE1B / "per-case.json")
        if row["alias"] == "llama3-8b-inst" and row["arm"] == "soft"
    ]
    if len(il3_cases) != 10:
        raise RuntimeError("immutable IL3 case denominator differs")
    il3_summary = {
        "depth": "IL3-FULL",
        "attempt_count": 10,
        "endpoint_count": il3["endpoint_count"],
        "typed_scientific_failure_count": 0,
        "technical_failure_count": 0,
        "request_endpoint_count": 100,
        "executed_inner_telemetry_rows": "NOT_RECORDED_IL3_PRE_TELEMETRY",
        "configured_inner_telemetry_row_maximum": 240,
        "byte_identical_early_stop_outer_count": "NOT_RECORDED_IL3_PRE_TELEMETRY",
        "z_eff_numerator": il3["z_eff_numerator"],
        "z_eff_denominator": il3["z_eff_denominator"],
        "z_gen_numerator": il3["z_gen_numerator"],
        "z_gen_denominator": il3["z_gen_denominator"],
        "z_gen_strict_numerator": il3["z_rephrase_strict_request_numerator"],
        "w_eff_numerator": il3["w_eff_numerator"],
        "w_eff_denominator": il3["w_eff_denominator"],
        "w_gen_numerator": il3["w_gen_numerator"],
        "w_gen_denominator": il3["w_gen_denominator"],
        "w_gen_strict_numerator": il3["w_rephrase_strict_request_numerator"],
        "loc_numerator": il3["loc_numerator"],
        "loc_denominator": il3["loc_denominator"],
        "z_full_six_nll": {"mean": il3["z_full_six_nll_mean"]},
        "w_full_six_nll": {"mean": il3["w_full_six_nll_mean"]},
        "full_six_w_minus_z_gap": {"mean": il3["w_minus_z_nll_gap_mean"]},
        "z_rewrite_nll": {"mean": il3["z_eff_nll_mean"], "median": il3["z_rewrite_nll_median"], "p90": il3["z_rewrite_nll_p90"], "max": max(row["z_eff_nll_new"] for row in il3_requests)},
        "z_rephrase_nll": {"mean": il3["z_gen_nll_mean"], "median": il3["z_rephrase_nll_median"], "p90": il3["z_rephrase_nll_p90"], "max": max(row["z_gen_nll_new"] for row in il3_requests)},
        "w_rewrite_nll": {"mean": il3["w_eff_nll_mean"], "median": il3["w_rewrite_nll_median"], "p90": il3["w_rewrite_nll_p90"], "max": max(row["w_eff_nll_new"] for row in il3_requests)},
        "w_rephrase_nll": {"mean": il3["w_gen_nll_mean"], "median": il3["w_rephrase_nll_median"], "p90": il3["w_rephrase_nll_p90"], "max": max(row["w_gen_nll_new"] for row in il3_requests)},
        "z_rewrite_margin": stats(row["z_eff_margin"] for row in il3_requests),
        "z_rephrase_margin": stats(row["z_gen_margin"] for row in il3_requests),
        "w_rewrite_margin": stats(row["w_eff_margin"] for row in il3_requests),
        "w_rephrase_margin": stats(row["w_gen_margin"] for row in il3_requests),
        "rewrite_w_minus_z_gap": {"mean": il3["rewrite_nll_gap_w_minus_z_mean"]},
        "rephrase_w_minus_z_gap": {"mean": il3["rephrase_nll_gap_w_minus_z_mean"]},
        "predicted_progress_sum": il3["predicted_progress_sum"],
        "actual_progress_sum": il3["actual_progress_sum"],
        "realization_ratio": il3["realization_ratio"],
        "negative_actual_count": il3["negative_actual_count"],
        "terminal_cumulative_structural_p": {"mean": il3["terminal_cumulative_structural_p_mean"]},
        "cumulative_bf16_capacity_endpoint": {"mean": il3["cumulative_bf16_capacity_endpoint_mean"]},
        "bf16_path_energy": {"mean": il3["realized_bf16_path_energy_mean"]},
        "simplex_top1_share": {"mean": il3["simplex_top1_share_mean"]},
        "velocity_abs_top1_share": "NOT_RECORDED_IL3_AGGREGATE",
        "layer_energy_top1_share": "NOT_RECORDED_IL3_AGGREGATE",
        "model_forward_calls": il3["model_forward_calls_completed_endpoints"],
        "model_backward_calls": il3["model_backward_calls_completed_endpoints"],
        "processed_tokens": il3["processed_tokens_completed_endpoints"],
        "materializations": il3["accepted_materializations_all_prefixes"],
        "edit_core_wall_seconds": {"mean": il3["edit_core_wall_seconds_mean"]},
        "terminal_evaluator_wall_seconds": stats(
            row["terminal_evaluator_wall_seconds"] for row in il3_cases
        ),
        "target_gradient_wall_seconds": "NOT_RECORDED_IL3_AGGREGATE",
        "inner_observation_wall_seconds": "NOT_APPLICABLE_IL3_PRE_TELEMETRY",
        "outer_observation_wall_seconds": "NOT_APPLICABLE_IL3_PRE_TELEMETRY",
        "final_inner_target_objective_nll_delta": "NOT_RECORDED_IL3_PRE_TELEMETRY",
        "final_inner_movement_norm": "NOT_RECORDED_IL3_PRE_TELEMETRY",
        "accuracy": "NOT_RECORDED",
    }
    il1 = {
        "depth": "IL1",
        "attempt_count": 10,
        "endpoint_count": comparison["il1_endpoint_count"],
        "typed_scientific_failure_count": 0,
        "technical_failure_count": 0,
        "request_endpoint_count": 100,
        "executed_inner_telemetry_rows": "NOT_RECORDED_IL1_PRE_TELEMETRY",
        "configured_inner_telemetry_row_maximum": 80,
        "byte_identical_early_stop_outer_count": "NOT_RECORDED_IL1_PRE_TELEMETRY",
        "z_eff_numerator": 100,
        "z_eff_denominator": 100,
        "z_gen_numerator": 183,
        "z_gen_denominator": 200,
        "z_gen_strict_numerator": "NOT_RECORDED",
        "w_eff_numerator": 100,
        "w_eff_denominator": 100,
        "w_gen_numerator": 181,
        "w_gen_denominator": 200,
        "w_gen_strict_numerator": "NOT_RECORDED",
        "loc_numerator": 871,
        "loc_denominator": 1000,
        "z_full_six_nll": {"mean": il3["z_full_six_nll_mean"] + comparison["full_six_target_nll_gain_il1_minus_il3"]},
        "w_full_six_nll": {"mean": il3["w_full_six_nll_mean"] + comparison["full_six_writer_nll_gain_il1_minus_il3"]},
        "full_six_w_minus_z_gap": {"mean": il3["w_minus_z_nll_gap_mean"] - comparison["gap_delta_il3_minus_il1"]},
        "z_rewrite_nll": "NOT_RECORDED_IL1_CONTINUOUS_TABLE_NOT_LOCAL",
        "z_rephrase_nll": "NOT_RECORDED_IL1_CONTINUOUS_TABLE_NOT_LOCAL",
        "w_rewrite_nll": "NOT_RECORDED_IL1_CONTINUOUS_TABLE_NOT_LOCAL",
        "w_rephrase_nll": "NOT_RECORDED_IL1_CONTINUOUS_TABLE_NOT_LOCAL",
        "rewrite_w_minus_z_gap": "NOT_RECORDED_IL1_CONTINUOUS_TABLE_NOT_LOCAL",
        "rephrase_w_minus_z_gap": "NOT_RECORDED_IL1_CONTINUOUS_TABLE_NOT_LOCAL",
        "z_rewrite_margin": "NOT_RECORDED_IL1_CONTINUOUS_TABLE_NOT_LOCAL",
        "z_rephrase_margin": "NOT_RECORDED_IL1_CONTINUOUS_TABLE_NOT_LOCAL",
        "w_rewrite_margin": "NOT_RECORDED_IL1_CONTINUOUS_TABLE_NOT_LOCAL",
        "w_rephrase_margin": "NOT_RECORDED_IL1_CONTINUOUS_TABLE_NOT_LOCAL",
        "predicted_progress_sum": "NOT_RECORDED_IL1_AGGREGATE",
        "actual_progress_sum": "NOT_RECORDED_IL1_AGGREGATE",
        "realization_ratio": "NOT_RECORDED_IL1_AGGREGATE",
        "negative_actual_count": il3["negative_actual_count"] - comparison["negative_actual_delta"],
        "terminal_cumulative_structural_p": {"mean": il3["terminal_cumulative_structural_p_mean"] - comparison["structural_p_delta"]},
        "cumulative_bf16_capacity_endpoint": {"mean": il3["cumulative_bf16_capacity_endpoint_mean"] - comparison["capacity_delta"]},
        "bf16_path_energy": {"mean": il3["realized_bf16_path_energy_mean"] - comparison["selected_energy_delta"]},
        "simplex_top1_share": "NOT_RECORDED_IL1_AGGREGATE",
        "velocity_abs_top1_share": "NOT_RECORDED_IL1_AGGREGATE",
        "layer_energy_top1_share": "NOT_RECORDED_IL1_AGGREGATE",
        "model_forward_calls": "NOT_RECORDED_IL1_LLAMA_SOFT_CELL",
        "model_backward_calls": "NOT_RECORDED_IL1_LLAMA_SOFT_CELL",
        "processed_tokens": "NOT_RECORDED_IL1_LLAMA_SOFT_CELL",
        "materializations": 80,
        "edit_core_wall_seconds": "NOT_RECORDED_IL1_LLAMA_SOFT_CELL",
        "terminal_evaluator_wall_seconds": "NOT_RECORDED_IL1_LLAMA_SOFT_CELL",
        "target_gradient_wall_seconds": "NOT_RECORDED_IL1_LLAMA_SOFT_CELL",
        "inner_observation_wall_seconds": "NOT_APPLICABLE_IL1_PRE_TELEMETRY",
        "outer_observation_wall_seconds": "NOT_APPLICABLE_IL1_PRE_TELEMETRY",
        "final_inner_target_objective_nll_delta": "NOT_APPLICABLE_IL1_SINGLE_INNER",
        "final_inner_movement_norm": "NOT_RECORDED_IL1_PRE_TELEMETRY",
        "accuracy": "NOT_RECORDED",
    }
    return [il1, il3_summary]


def paired_il3_case_rows(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    il3_rows = {
        int(row["case_index"]): row
        for row in load(PHASE1B / "per-case.json")
        if row["alias"] == "llama3-8b-inst" and row["arm"] == "soft"
    }
    if len(il3_rows) != 10:
        raise RuntimeError("immutable IL3 Llama Soft case denominator differs")
    output = []
    for new in cases:
        old = il3_rows[int(new["case_index"])]
        output.append(
            {
                "comparison": f"{new['depth']}-IL3-FULL",
                "case_index": new["case_index"],
                "pair_binding": "same_stream_order_case_index",
                "delta_z_full_six_nll": new["z_full_six_nll"] - old["z_full_six_nll"],
                "delta_w_full_six_nll": new["w_full_six_nll"] - old["w_full_six_nll"],
                "delta_full_six_w_minus_z_gap": new["full_six_w_minus_z_gap"] - old["w_minus_z_nll_gap"],
                "delta_z_rewrite_nll_mean": new["z_rewrite_nll_mean"] - old["z_eff_nll_mean"],
                "delta_z_rephrase_nll_mean": new["z_rephrase_nll_mean"] - old["z_gen_nll_mean"],
                "delta_w_rewrite_nll_mean": new["w_rewrite_nll_mean"] - old["w_eff_nll_mean"],
                "delta_w_rephrase_nll_mean": new["w_rephrase_nll_mean"] - old["w_gen_nll_mean"],
                "delta_rewrite_w_minus_z_gap": new["rewrite_w_minus_z_gap_mean"] - old["rewrite_nll_gap_w_minus_z"],
                "delta_rephrase_w_minus_z_gap": new["rephrase_w_minus_z_gap_mean"] - old["rephrase_nll_gap_w_minus_z"],
                "delta_z_eff": new["z_eff_numerator"] - old["z_eff_numerator"],
                "delta_z_gen": new["z_gen_numerator"] - old["z_gen_numerator"],
                "delta_w_eff": new["w_eff_numerator"] - old["w_eff_numerator"],
                "delta_w_gen": new["w_gen_numerator"] - old["w_gen_numerator"],
                "delta_loc": new["loc_numerator"] - old["loc_numerator"],
                "delta_predicted_progress": new["predicted_progress_sum"] - old["predicted_progress_sum"],
                "delta_actual_progress": new["actual_progress_sum"] - old["actual_progress_sum"],
                "delta_realization_ratio": new["realization_ratio"] - old["realization_ratio"],
                "delta_negative_actual_count": new["negative_actual_count"] - old["negative_actual_count"],
                "delta_structural_p": new["terminal_cumulative_structural_p"] - old["terminal_cumulative_structural_p"],
                "delta_capacity": new["cumulative_bf16_capacity_endpoint"] - old["cumulative_bf16_capacity_endpoint"],
                "delta_bf16_path_energy": new["bf16_path_energy"] - old["realized_bf16_path_energy"],
                "new_terminal_sha256": new["terminal_sha256"],
                "il3_terminal_sha256": old["terminal_sha256"],
            }
        )
    return output


def paired_il3_request_rows(
    requests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    il3_rows = {
        (int(row["case_index"]), int(row["request_index"])): row
        for row in load(PHASE1B / "per-request.json")
        if row["alias"] == "llama3-8b-inst" and row["arm"] == "soft"
    }
    if len(il3_rows) != 100:
        raise RuntimeError("immutable IL3 Llama Soft request denominator differs")
    output = []
    for new in requests:
        old = il3_rows[(int(new["case_index"]), int(new["request_index"]))]
        output.append(
            {
                "comparison": f"{new['depth']}-IL3-FULL",
                "case_index": new["case_index"],
                "request_index": new["request_index"],
                "request_sha256": old["request_sha256"],
                "pair_binding": "same_stream_order_case_request",
                "delta_z_rewrite_nll": new["z_rewrite_nll"] - old["z_eff_nll_new"],
                "delta_z_rephrase_nll": statistics.fmean(new["z_rephrase_nll"]) - old["z_gen_nll_new"],
                "delta_w_rewrite_nll": new["w_rewrite_nll"] - old["w_eff_nll_new"],
                "delta_w_rephrase_nll": statistics.fmean(new["w_rephrase_nll"]) - old["w_gen_nll_new"],
                "delta_rewrite_w_minus_z_gap": new["rewrite_w_minus_z_nll_gap"] - old["rewrite_nll_gap_w_minus_z"],
                "delta_rephrase_w_minus_z_gap": new["rephrase_w_minus_z_nll_gap"] - old["rephrase_nll_gap_w_minus_z"],
                "delta_z_rewrite_success": new["z_rewrite_success"] - old["z_rewrite_success"],
                "delta_z_rephrase_strict_success": new["z_rephrase_strict_success"] - old["z_rephrase_strict_success"],
                "delta_w_rewrite_success": new["w_rewrite_success"] - old["w_rewrite_success"],
                "delta_w_rephrase_strict_success": new["w_rephrase_strict_success"] - old["w_rephrase_strict_success"],
                "accuracy": "NOT_RECORDED",
            }
        )
    return output


def global_ordinal_rows(inners: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in inners:
        groups.setdefault(
            (str(row["depth"]), int(row["global_target_update_ordinal"])), []
        ).append(row)
    output = []
    for (depth, ordinal), rows in sorted(
        groups.items(),
        key=lambda item: (
            int(item[0][0].split("IL", 1)[1].split("-", 1)[0]),
            item[0][1],
        ),
    ):
        output.append(
            {
                "depth": depth,
                "global_target_update_ordinal": ordinal,
                "case_denominator": len(rows),
                "target_objective_nll_mean": statistics.fmean(
                    float(row["target_objective_nll_mean"]) for row in rows
                ),
                "accepted_z_rewrite_nll_mean": statistics.fmean(
                    float(row["accepted_z_rewrite_nll"]["mean"]) for row in rows
                ),
                "accepted_z_rephrase_nll_mean": statistics.fmean(
                    float(row["accepted_z_rephrase_nll"]["mean"]) for row in rows
                ),
                "inner_step_movement_norm_mean": statistics.fmean(
                    float(row["inner_step_movement_norm"]) for row in rows
                ),
                "cumulative_outer_entry_movement_norm_mean": statistics.fmean(
                    float(row["cumulative_outer_entry_movement_norm"])
                    for row in rows
                ),
                "primary_selection_count": sum(
                    int(row["primary_selection_count"]) for row in rows
                ),
                "rescue_selection_count": sum(
                    int(row["rescue_selection_count"]) for row in rows
                ),
                "current_selection_count": sum(
                    int(row["current_selection_count"]) for row in rows
                ),
            }
        )
    return output


def hard_cohort_rows(
    requests: Sequence[Mapping[str, Any]], top_k: int = 10
) -> list[dict[str, Any]]:
    """Bind the observed hardest accepted-z rephrase requests per new depth."""
    output: list[dict[str, Any]] = []
    for depth in DEPTHS:
        label = f"IL{depth}-FULL"
        own = [row for row in requests if row["depth"] == label]
        ranked = sorted(
            own,
            key=lambda row: statistics.fmean(row["z_rephrase_nll"]),
            reverse=True,
        )[:top_k]
        for rank, row in enumerate(ranked, start=1):
            output.append(
                {
                    "depth": label,
                    "hard_cohort_definition": "top10_accepted_z_rephrase_nll_mean",
                    "rank": rank,
                    "case_index": row["case_index"],
                    "request_index": row["request_index"],
                    "request_sha256": row["request_sha256"],
                    "z_rewrite_nll": row["z_rewrite_nll"],
                    "z_rephrase_nll_mean": statistics.fmean(row["z_rephrase_nll"]),
                    "w_rewrite_nll": row["w_rewrite_nll"],
                    "w_rephrase_nll_mean": statistics.fmean(row["w_rephrase_nll"]),
                    "rewrite_w_minus_z_nll_gap": row["rewrite_w_minus_z_nll_gap"],
                    "rephrase_w_minus_z_nll_gap": row["rephrase_w_minus_z_nll_gap"],
                    "z_rewrite_success": row["z_rewrite_success"],
                    "z_rephrase_strict_success": row["z_rephrase_strict_success"],
                    "w_rewrite_success": row["w_rewrite_success"],
                    "w_rephrase_strict_success": row["w_rephrase_strict_success"],
                    "w_locality_success": row["w_locality_success"],
                }
            )
    return output


def focused_summary_rows(
    summaries: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> list[dict[str, Any]]:
    return [
        {"depth": row["depth"], **{field: row[field] for field in fields}}
        for row in summaries
    ]


def comparisons(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(summaries, key=lambda row: int(str(row["depth"]).split("IL", 1)[1].split("-", 1)[0]))
    output = []
    for left, right in zip(ordered, ordered[1:]):
        def mean(row: Mapping[str, Any], key: str) -> float | str:
            value = row[key]
            return value.get("mean", "NOT_RECORDED") if isinstance(value, Mapping) else "NOT_RECORDED"

        row: dict[str, Any] = {
            "comparison": f"{right['depth']}-{left['depth']}",
            "left_endpoints": left["endpoint_count"],
            "right_endpoints": right["endpoint_count"],
            "delta_z_eff": right["z_eff_numerator"] - left["z_eff_numerator"],
            "delta_z_gen": right["z_gen_numerator"] - left["z_gen_numerator"],
            "delta_w_eff": right["w_eff_numerator"] - left["w_eff_numerator"],
            "delta_w_gen": right["w_gen_numerator"] - left["w_gen_numerator"],
            "delta_loc": right["loc_numerator"] - left["loc_numerator"],
        }
        for field in (
            "z_full_six_nll",
            "w_full_six_nll",
            "full_six_w_minus_z_gap",
            "z_rewrite_nll",
            "z_rephrase_nll",
            "w_rewrite_nll",
            "w_rephrase_nll",
            "rewrite_w_minus_z_gap",
            "rephrase_w_minus_z_gap",
            "terminal_cumulative_structural_p",
            "cumulative_bf16_capacity_endpoint",
            "bf16_path_energy",
            "edit_core_wall_seconds",
        ):
            a, b = mean(left, field), mean(right, field)
            row[f"delta_{field}"] = b - a if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "NOT_RECORDED"
        output.append(row)
    return output


def fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def report_text(summaries: Sequence[Mapping[str, Any]], delta_rows: Sequence[Mapping[str, Any]], failures: Sequence[Mapping[str, Any]], row_counts: Mapping[str, int]) -> str:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    numeric = [row for row in summaries if isinstance(row["w_full_six_nll"], Mapping)]
    lowest_z = min(numeric, key=lambda row: float(row["z_full_six_nll"]["mean"]))
    lowest_w = min(numeric, key=lambda row: float(row["w_full_six_nll"]["mean"]))
    max_w_gen = max(int(row["w_gen_numerator"]) for row in summaries)
    max_w_gen_depths = ", ".join(
        str(row["depth"]) for row in summaries if row["w_gen_numerator"] == max_w_gen
    )
    max_loc = max(int(row["loc_numerator"]) for row in summaries)
    max_loc_depths = ", ".join(
        str(row["depth"]) for row in summaries if row["loc_numerator"] == max_loc
    )
    lines = [
        "# P1R52 Target Depth IL1/IL3/IL8/IL10/IL15 Atomic 사실 보고서",
        "",
        f"- 생성 시각(KST): `{now}`",
        f"- 실행 source HEAD/tree: `{SOURCE_HEAD}` / `{SOURCE_TREE}`",
        f"- contract SHA256: `{CONTRACT_SHA256}`",
        f"- numerical lock SHA256: `{NUMERICAL_LOCK_SHA256}`",
        f"- stream/order: `{STREAM_ROOT}` / `{ORDER_SHA256}`",
        "- immutable IL1/IL3 report/table references: independent SHA rehash PASS.",
        "- 범위: Llama3-8B-Instruct, Soft J0, independent B10×10, K8; 신규 IL8/IL10/IL15. IL1/IL3는 immutable reference 재사용.",
        "- IL5/Qwen/Neutral/Native/sequential/Historical 실행: `0`.",
        "",
        "## 한 문단 결론",
        "",
        f"30/30 신규 endpoint가 유효했다. 관측 범위에서 z full-six NLL 최저는 {lowest_z['depth']} "
        f"({fmt(lowest_z['z_full_six_nll']['mean'])}), W full-six NLL 최저는 {lowest_w['depth']} "
        f"({fmt(lowest_w['w_full_six_nll']['mean'])})였다. W Gen 최대값 {max_w_gen}/200은 "
        f"{max_w_gen_depths}, Loc 최대값 {max_loc}/1000은 {max_loc_depths}에서 기록됐다. "
        "depth별 z/W·전달·보존·에너지·compute 절대값은 아래 표에 병렬 제시하며, "
        "단일 NLL만으로 scientific promotion을 선택하지 않았다 (`scientific_promotion=false`).",
        "",
        "## 절대 분모와 z/W 지표",
        "",
        "성공은 pinned evaluator의 margin-positive prompt bit이며 strict Gen은 request의 모든 rephrase prompt 성공이다. accuracy는 별도 기록이 없어 `NOT_RECORDED`다.",
        "",
        "| depth | valid/attempt | z Eff | z Gen/strict | W Eff | W Gen/strict | Loc | z/W/gap full6 NLL | z rewrite NLL mean/med/p90/max | z rephrase NLL mean/med/p90/max | W rewrite NLL mean/med/p90/max | W rephrase NLL mean/med/p90/max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        def triple(field: str) -> str:
            value = row[field]
            if not isinstance(value, Mapping):
                return str(value)
            return "/".join(
                fmt(value.get(key, "NOT_RECORDED"))
                for key in ("mean", "median", "p90", "max")
            )

        lines.append(
            "| {depth} | {valid}/{attempt} | {ze}/{zed} | {zg}/{zgd}; {zgs} | {we}/{wed} | {wg}/{wgd}; {wgs} | {loc}/{locd} | {zf}/{wf}/{gap} | {zrw} | {zrp} | {wrw} | {wrp} |".format(
                depth=row["depth"], valid=row["endpoint_count"], attempt=row["attempt_count"],
                ze=row["z_eff_numerator"], zed=row["z_eff_denominator"], zg=row["z_gen_numerator"], zgd=row["z_gen_denominator"], zgs=row["z_gen_strict_numerator"],
                we=row["w_eff_numerator"], wed=row["w_eff_denominator"], wg=row["w_gen_numerator"], wgd=row["w_gen_denominator"], wgs=row["w_gen_strict_numerator"],
                loc=row["loc_numerator"], locd=row["loc_denominator"], zf=fmt(row["z_full_six_nll"]["mean"]), wf=fmt(row["w_full_six_nll"]["mean"]), gap=fmt(row["full_six_w_minus_z_gap"]["mean"]),
                zrw=triple("z_rewrite_nll"), zrp=triple("z_rephrase_nll"), wrw=triple("w_rewrite_nll"), wrp=triple("w_rephrase_nll"),
            )
        )
    lines += [
        "",
        "## Rewrite/rephrase margin",
        "",
        "| depth | z rewrite margin mean/p10 | z rephrase margin mean/p10 | W rewrite margin mean/p10 | W rephrase margin mean/p10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        def margin_pair(field: str) -> str:
            value = row[field]
            if not isinstance(value, Mapping):
                return str(value)
            return f"{fmt(value.get('mean', 'NOT_RECORDED'))}/{fmt(value.get('p10', 'NOT_RECORDED'))}"

        lines.append(
            f"| {row['depth']} | {margin_pair('z_rewrite_margin')} | {margin_pair('z_rephrase_margin')} | "
            f"{margin_pair('w_rewrite_margin')} | {margin_pair('w_rephrase_margin')} |"
        )
    lines += [
        "",
        "## Writer 전달·보존·계산",
        "",
        "| depth | predicted/actual/realization | neg | P endpoint mean | capacity endpoint mean | BF16 path energy mean | simplex/velocity/layer-energy top1 | target path outer/inner mean | final-inner ΔNLL/move | F/B/tokens/mat | target/inner-obs/outer-obs/edit/eval mean(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['depth']} | {fmt(row['predicted_progress_sum'])}/{fmt(row['actual_progress_sum'])}/{fmt(row['realization_ratio'])} | {row['negative_actual_count']} | "
            f"{fmt(row['terminal_cumulative_structural_p'].get('mean') if isinstance(row['terminal_cumulative_structural_p'], Mapping) else row['terminal_cumulative_structural_p'])} | "
            f"{fmt(row['cumulative_bf16_capacity_endpoint'].get('mean') if isinstance(row['cumulative_bf16_capacity_endpoint'], Mapping) else row['cumulative_bf16_capacity_endpoint'])} | "
            f"{fmt(row['bf16_path_energy'].get('mean') if isinstance(row['bf16_path_energy'], Mapping) else row['bf16_path_energy'])} | "
            f"{fmt(row['simplex_top1_share'].get('mean') if isinstance(row['simplex_top1_share'], Mapping) else row['simplex_top1_share'])}/"
            f"{fmt(row['velocity_abs_top1_share'].get('mean') if isinstance(row['velocity_abs_top1_share'], Mapping) else row['velocity_abs_top1_share'])}/"
            f"{fmt(row['layer_energy_top1_share'].get('mean') if isinstance(row['layer_energy_top1_share'], Mapping) else row['layer_energy_top1_share'])} | "
            f"{fmt(row.get('outer_target_path_norm_sum', {}).get('mean', 'NOT_RECORDED') if isinstance(row.get('outer_target_path_norm_sum'), Mapping) else 'NOT_RECORDED')}/"
            f"{fmt(row.get('inner_target_path_norm_sum', {}).get('mean', 'NOT_RECORDED') if isinstance(row.get('inner_target_path_norm_sum'), Mapping) else 'NOT_RECORDED')} | "
            f"{fmt(row['final_inner_target_objective_nll_delta'].get('mean') if isinstance(row['final_inner_target_objective_nll_delta'], Mapping) else row['final_inner_target_objective_nll_delta'])}/"
            f"{fmt(row['final_inner_movement_norm'].get('mean') if isinstance(row['final_inner_movement_norm'], Mapping) else row['final_inner_movement_norm'])} | "
            f"{row['model_forward_calls']}/{row['model_backward_calls']}/{row['processed_tokens']}/{row['materializations']} | "
            f"{fmt(row['target_gradient_wall_seconds'].get('mean') if isinstance(row['target_gradient_wall_seconds'], Mapping) else row['target_gradient_wall_seconds'])}/"
            f"{fmt(row['inner_observation_wall_seconds'].get('mean') if isinstance(row['inner_observation_wall_seconds'], Mapping) else row['inner_observation_wall_seconds'])}/"
            f"{fmt(row['outer_observation_wall_seconds'].get('mean') if isinstance(row['outer_observation_wall_seconds'], Mapping) else row['outer_observation_wall_seconds'])}/"
            f"{fmt(row['edit_core_wall_seconds'].get('mean') if isinstance(row['edit_core_wall_seconds'], Mapping) else row['edit_core_wall_seconds'])}/"
            f"{fmt(row['terminal_evaluator_wall_seconds'].get('mean') if isinstance(row['terminal_evaluator_wall_seconds'], Mapping) else row['terminal_evaluator_wall_seconds'])} |"
        )
    lines += [
        "",
        "## 인접 depth 산술 차이",
        "",
        "| comparison | Δz full6 | ΔW full6 | ΔW-z gap | Δz E/G | ΔW E/G/Loc | ΔP/cap/energy | Δedit-core(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in delta_rows:
        lines.append(
            f"| {row['comparison']} | {fmt(row['delta_z_full_six_nll'])} | {fmt(row['delta_w_full_six_nll'])} | {fmt(row['delta_full_six_w_minus_z_gap'])} | "
            f"{row['delta_z_eff']}/{row['delta_z_gen']} | {row['delta_w_eff']}/{row['delta_w_gen']}/{row['delta_loc']} | "
            f"{fmt(row['delta_terminal_cumulative_structural_p'])}/{fmt(row['delta_cumulative_bf16_capacity_endpoint'])}/{fmt(row['delta_bf16_path_energy'])} | {fmt(row['delta_edit_core_wall_seconds'])} |"
        )
    lines += [
        "",
        "- IL1 continuous rewrite/rephrase NLL, strict Gen, per-case/per-request compute는 local immutable input에 없어 `NOT_RECORDED`; proxy/imputation 0.",
        f"- 신규 typed scientific failures: `{row_counts['typed_scientific_failures']}`; technical failure/retry/imputation: `{row_counts['technical_failures']}/0/0`.",
        f"- table rows: per-case `{row_counts['per_case']}`, per-step `{row_counts['per_step']}`, per-inner `{row_counts['per_inner']}`, per-inner-request `{row_counts['per_inner_request']}`, terminal per-request `{row_counts['per_request']}`, global-ordinal summary `{row_counts['global_ordinal_summary']}`.",
        "",
        "## Per-inner telemetry 완전성",
        "",
        "| depth | executed/configured max rows (10 cases) | byte-identical early-stop outers | outer rows | request-bound fields | duplicate eval | added B/gen/action influence |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in summaries:
        if str(row["depth"]) in {"IL1", "IL3-FULL"}:
            lines.append(
                f"| {row['depth']} | {row['executed_inner_telemetry_rows']}/{row['configured_inner_telemetry_row_maximum']} | "
                f"{row['byte_identical_early_stop_outer_count']} | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |"
            )
        else:
            lines.append(
                f"| {row['depth']} | {row['executed_inner_telemetry_rows']}/{row['configured_inner_telemetry_row_maximum']} | "
                f"{row['byte_identical_early_stop_outer_count']} | 80 | target objective/rewrite/rephrase/movement | 0 | 0/0/0 |"
            )
    lines += [
        "",
        "- IL15 일부 outer는 계약상 허용된 전체 selected FP32 tensor byte-identical early stop으로 15회 이전 종료됐다. 각 short trajectory는 마지막 두 `target_next_sha256` 동일성으로 검증했다; 누락/추정/imputation으로 채우지 않았다.",
        "- 신규 각 accepted inner observation은 target objective/rewrite/rephrase/movement를 request/depth/case/outer/inner/ordinal에 결합하며 added backward/generation/action influence/duplicate evaluation은 모두 0이다.",
        "- per-inner selection PRIMARY/RESCUE/CURRENT와 selected endpoint NLL은 기록됨. 원 target-step의 상세 clamp/KL/decay/reference-energy receipt 본문은 accepted result에 저장되지 않아 `NOT_RECORDED`; hash만 기록됨.",
        "",
        "## 계약 질문 5개",
        "",
        "1. target-depth 자체의 z 변화: 위 depth별 z full-six/rewrite/rephrase/Eff/Gen 및 인접 산술 차이에 기록했다.",
        "2. z 변화의 J0 writer 전달: W 지표, W-z gap, predicted/actual/realization에 기록했다.",
        "3. sequential 누적 유지: 이 extension에서 sequential 실행 0이므로 `NOT_EVALUATED`.",
        "4. KDC/barrier 대 depth: KDC와 writer는 고정되고 depth만 바뀌었으나 barrier 단독 기여 분리는 수행하지 않았다. `TARGET_DEPTH_ONLY_ABLATION`.",
        "5. Qwen/Llama 차이: 신규 실행은 Llama Soft만이므로 Qwen 신규 비교는 `NOT_EVALUATED`.",
        "",
        "scientific_promotion=false. 추가 model/GPU/Slurm job=0.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate terminal inputs and derived row counts without writing outputs.",
    )
    args = parser.parse_args()
    verify_references()
    cases, steps, inners, inner_requests, requests, failures = extract_new()
    summaries = il1_il3_summaries() + [
        new_summary(depth, cases, requests, failures) for depth in DEPTHS
    ]
    delta_rows = comparisons(summaries)
    paired_case_rows = paired_il3_case_rows(cases)
    paired_request_rows = paired_il3_request_rows(requests)
    ordinal_rows = global_ordinal_rows(inners)
    hard_rows = hard_cohort_rows(requests)
    wz_rows = focused_summary_rows(
        summaries,
        (
            "endpoint_count",
            "z_full_six_nll",
            "w_full_six_nll",
            "full_six_w_minus_z_gap",
            "rewrite_w_minus_z_gap",
            "rephrase_w_minus_z_gap",
            "predicted_progress_sum",
            "actual_progress_sum",
            "realization_ratio",
            "negative_actual_count",
        ),
    )
    compute_layer_rows = focused_summary_rows(
        summaries,
        (
            "terminal_cumulative_structural_p",
            "cumulative_bf16_capacity_endpoint",
            "bf16_path_energy",
            "simplex_top1_share",
            "velocity_abs_top1_share",
            "layer_energy_top1_share",
            "model_forward_calls",
            "model_backward_calls",
            "processed_tokens",
            "materializations",
            "edit_core_wall_seconds",
            "terminal_evaluator_wall_seconds",
            "target_gradient_wall_seconds",
            "inner_observation_wall_seconds",
            "outer_observation_wall_seconds",
        ),
    )
    row_counts = {
        "per_case": len(cases),
        "per_step": len(steps),
        "per_inner": len(inners),
        "per_inner_request": len(inner_requests),
        "per_request": len(requests),
        "paired_il3_per_case": len(paired_case_rows),
        "paired_il3_per_request": len(paired_request_rows),
        "global_ordinal_summary": len(ordinal_rows),
        "hard_cohort": len(hard_rows),
        "w_z_gap_summary": len(wz_rows),
        "compute_layer_summary": len(compute_layer_rows),
        "typed_scientific_failures": sum(
            row.get("classification") != "TECHNICAL_FAIL" for row in failures
        ),
        "technical_failures": sum(
            row.get("classification") == "TECHNICAL_FAIL" for row in failures
        ),
    }
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALIDATION_PASS",
                    "depths": list(DEPTHS),
                    "row_counts": row_counts,
                    "summary_rows": len(summaries),
                    "comparison_rows": len(delta_rows),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if any(path.exists() for path in HERE.glob("p1r52-target-depth-il1-il3-il8-il10-il15-*.json")) or REPORT.exists():
        raise RuntimeError("analysis output namespace is not empty")
    outputs: dict[str, str] = {}
    tables = {
        "p1r52-target-depth-il1-il3-il8-il10-il15-depth-summary.json": summaries,
        "p1r52-target-depth-il1-il3-il8-il10-il15-comparisons.json": delta_rows,
        "p1r52-target-depth-il8-il10-il15-per-case.json": cases,
        "p1r52-target-depth-il8-il10-il15-per-step.json": steps,
        "p1r52-target-depth-il8-il10-il15-per-inner.json": inners,
        "p1r52-target-depth-il8-il10-il15-per-inner-request.json": inner_requests,
        "p1r52-target-depth-il8-il10-il15-per-request.json": requests,
        "p1r52-target-depth-il8-il10-il15-minus-il3-paired-per-case.json": paired_case_rows,
        "p1r52-target-depth-il8-il10-il15-minus-il3-paired-per-request.json": paired_request_rows,
        "p1r52-target-depth-il8-il10-il15-global-ordinal-summary.json": ordinal_rows,
        "p1r52-target-depth-il8-il10-il15-hard-cohort.json": hard_rows,
        "p1r52-target-depth-il1-il3-il8-il10-il15-w-z-gap-summary.json": wz_rows,
        "p1r52-target-depth-il1-il3-il8-il10-il15-compute-layer-summary.json": compute_layer_rows,
        "p1r52-target-depth-il8-il10-il15-typed-failures.json": failures,
    }
    for name, value in tables.items():
        outputs[name] = write_json(HERE / name, value)
        if isinstance(value, list) and value:
            outputs[name.replace(".json", ".csv")] = write_csv(
                HERE / name.replace(".json", ".csv"), value
            )
    outputs[REPORT.name] = write_once(
        REPORT,
        report_text(summaries, delta_rows, failures, row_counts).encode(),
    )
    identities = {
        "schema": "ode-edit-s05-p1r52-target-depth-extension-result-identities/v1",
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "contract_sha256": CONTRACT_SHA256,
        "numerical_lock_sha256": NUMERICAL_LOCK_SHA256,
        "stream_root": STREAM_ROOT,
        "request_order_sha256": ORDER_SHA256,
        "new_roots": {
            f"IL{depth}-FULL": {
                "path": str(root),
                "terminal_sha256": sha256_file(root / "terminal.json"),
                "manifest_sha256": sha256_file(root / "manifest.json"),
            }
            for depth, root in ROOTS.items()
        },
        "immutable_il1_report_sha256": sha256_file(
            REPO / "experiment-reports/global/2026-08-17-p1r52-rsa-r42safekdc-m1-repair-r1-final-ko.md"
        ),
        "immutable_il3_manifest_sha256": sha256_file(PHASE1B / "manifest.json"),
        "immutable_reference_sha256": REFERENCE_SHA256,
        "row_counts": row_counts,
    }
    identities["identity_sha256"] = canonical_hash(identities)
    outputs["p1r52-target-depth-il8-il10-il15-result-identities.json"] = write_json(
        HERE / "p1r52-target-depth-il8-il10-il15-result-identities.json", identities
    )
    members = []
    for name in sorted(["build_report.py", "verify_report.py", *outputs]):
        path = HERE / name
        members.append(
            {
                "name": name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "mode": oct(path.stat().st_mode & 0o777),
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema": "ode-edit-s05-p1r52-target-depth-extension-analysis-manifest/v1",
        "instruction_id": "ODEEDIT-S05-P1R52-TARGET-DEPTH-IL8-IL10-IL15-V1",
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "row_counts": row_counts,
        "members": members,
        "members_root_sha256": canonical_hash(members),
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = write_json(HERE / "analysis-manifest.json", manifest)
    receipt = {
        "schema": "ode-edit-s05-p1r52-target-depth-extension-analysis-receipt/v1",
        "status": "PASS",
        "report_path": str(REPORT),
        "report_sha256": outputs[REPORT.name],
        "manifest_path": str(HERE / "analysis-manifest.json"),
        "manifest_sha256": manifest_sha,
        "manifest_root": manifest["members_root_sha256"],
        "attempts": 30,
        "valid_endpoints": len(cases),
        "typed_scientific_failures": row_counts["typed_scientific_failures"],
        "technical_failures": row_counts["technical_failures"],
        "imputation_count": 0,
        "scientific_promotion": False,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    write_json(HERE / "analysis-receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
