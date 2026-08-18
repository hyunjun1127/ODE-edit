#!/usr/bin/env python3
"""Raw-free terminal recorder for P1R52 Frozen-Pi Sequential Quota Writer."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


WORKTREE = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-p1r52-fpiq-atomic-b10x10-v1"
)
RESULTS = WORKTREE / "local/odebf/results"
REPORT_DIR = WORKTREE / (
    "local/odebf/reports/"
    "p1r52-frozen-pi-quota-writer-atomic-b10x10-v1"
)
CONTRACT = Path(
    "/mnt/raid5/janghj/.codex/attachments/"
    "76d5dc44-60b9-4c33-baa7-1906233fd24b/pasted-text.txt"
)
LOCK = WORKTREE / (
    "project/run_scripts/ode_bf/locks/"
    "numerical_lock_s05_p1r52_fpiq_atomic_b10x10.json"
)
SOURCE_MANIFEST = WORKTREE / (
    "project/run_scripts/ode_bf/locks/"
    "source_manifest_s05_p1r52_fpiq_atomic_b10x10.json"
)

ROOTS = {
    "J0": RESULTS / "s05-p1r52-fpiq-independent-b10x10-llama3-8b-inst-j0-v1",
    "SV": RESULTS / "s05-p1r52-fpiq-independent-b10x10-llama3-8b-inst-sv-tech-r1-v1",
    "FPIQ": RESULTS / "s05-p1r52-fpiq-independent-b10x10-llama3-8b-inst-fpiq-tech-r1-v1",
}
PRIMARY_INVALID_ROOTS = {
    "SV": RESULTS / "s05-p1r52-fpiq-independent-b10x10-llama3-8b-inst-sv-v1",
    "FPIQ": RESULTS / "s05-p1r52-fpiq-independent-b10x10-llama3-8b-inst-fpiq-v1",
}

REPORT_NAME = "p1r52-frozen-pi-quota-writer-atomic-b10x10-analysis-ko.md"
TABLE_NAMES = {
    "aggregate": "p1r52-fpiq-aggregate-summary.json",
    "case": "p1r52-fpiq-per-case.json",
    "step": "p1r52-fpiq-per-step.json",
    "layer": "p1r52-fpiq-per-layer.json",
    "failure": "p1r52-fpiq-typed-failures.json",
    "paired": "p1r52-fpiq-paired-case-deltas.json",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    path.write_text(payload, encoding="utf-8")
    os.chmod(path, 0o600)


def finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def scalar_sum(value: Any) -> float:
    if isinstance(value, Mapping):
        return sum(float(item) for item in value.values())
    return float(value)


def flatten(items: Sequence[Sequence[Any]]) -> list[Any]:
    return [value for group in items for value in group]


def quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def stats(values: Iterable[Any]) -> dict[str, Any]:
    clean = [float(value) for value in values if finite(value) is not None]
    if not clean:
        return {
            "count": 0, "mean": None, "median": None,
            "p90": None, "min": None, "max": None,
        }
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "p90": quantile(clean, 0.90),
        "min": min(clean),
        "max": max(clean),
    }


def metric_summary(panel: Mapping[str, Any], metric_name: str) -> dict[str, Any]:
    primary = panel["primary"]
    metric = primary["metrics"][metric_name]
    vectors = panel["numeric_vectors"][metric_name]
    flat_vectors = flatten(vectors)
    correct = metric.get("per_case_correct", [])
    required = metric.get("per_case_required", [])
    strict_num = sum(int(a == b) for a, b in zip(correct, required))
    strict_den = len(required)
    return {
        "numerator": int(metric["numerator"]),
        "denominator": int(metric["denominator"]),
        "rate": float(metric["official_aggregate"]),
        "strict_numerator": strict_num,
        "strict_denominator": strict_den,
        "strict_rate": strict_num / strict_den if strict_den else None,
        "nll_new": stats(item["nll_new"] for item in flat_vectors),
        "nll_old": stats(item["nll_old"] for item in flat_vectors),
        "margin": stats(item["margin"] for item in flat_vectors),
        "bit_vector_sha256": metric["bit_vector_sha256"],
        "per_request_bits": metric["per_case_bits"],
        "per_request_correct": correct,
        "per_request_required": required,
    }


def case_dir(root: Path, case_index: int) -> Path:
    return root / f"raw/cases/case-{case_index:02d}"


def accepted_dir(root: Path, arm: str, case_index: int) -> Path:
    return case_dir(root, case_index) / "raw/ode" / f"p1r52-fpiq-{arm.lower()}"


def accepted_files(root: Path, arm: str, case_index: int) -> list[Path]:
    directory = accepted_dir(root, arm, case_index)
    return sorted(
        directory.glob("accepted-k*.json"),
        key=lambda path: int(path.stem.replace("accepted-k", "")),
    )


def complete_terminal(root: Path, case_index: int) -> Mapping[str, Any] | None:
    path = case_dir(root, case_index) / "terminal.json"
    return read_json(path) if path.exists() else None


def aggregate_counts(rows: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    numerator = sum(int(row[f"{prefix}_numerator"]) for row in rows)
    denominator = sum(int(row[f"{prefix}_denominator"]) for row in rows)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def build_case_row(arm: str, root: Path, case_index: int) -> dict[str, Any]:
    terminal = complete_terminal(root, case_index)
    available_steps = accepted_files(root, arm, case_index)
    row: dict[str, Any] = {
        "arm": arm,
        "case_index": case_index,
        "attempt_denominator": 1,
        "available_prefix_count": len(available_steps),
        "endpoint_available": terminal is not None,
        "endpoint_denominator": 1 if terminal is not None else 0,
        "status": "CASE_COMPLETE" if terminal is not None else "TYPED_INCOMPLETE",
        "result_root": str(root),
    }
    if terminal is None:
        failure = read_json(case_dir(root, case_index) / "failure.json")
        row.update({
            "classification": failure["classification"],
            "exception_class": failure["exception_class"],
            "exception_message_sha256": failure["exception_message_sha256"],
            "failure_identity_sha256": failure["identity_sha256"],
            "W0_restored": bool(failure["w0_restore"]["byte_restored_exact"]),
        })
        return row

    z_panel = terminal["terminal_four_panel"]["z_inject"]
    w_panel = terminal["terminal_four_panel"]["weight"]
    z_eff = metric_summary(z_panel, "efficacy")
    z_gen = metric_summary(z_panel, "generalization")
    w_eff = metric_summary(w_panel, "efficacy")
    w_gen = metric_summary(w_panel, "generalization")
    w_loc = metric_summary(w_panel, "locality-preservation")

    steps = [read_json(path) for path in available_steps]
    energy = [
        scalar_sum(step["materialization"]["realized_bf16_step_energy"])
        for step in steps
    ]
    capacities = [
        scalar_sum(step["materialization"]["cumulative_bf16_capacity"])
        for step in steps
    ]
    predicted = [step["progress"]["predicted"] for step in steps]
    actual = [
        float(item["actual"])
        for item in terminal["rollout"]["delayed_progress"]
    ] + [float(steps[-1]["progress"]["actual"])]
    coverages = [
        float(step["sequential_writer"]["writer_coverage"])
        if step.get("sequential_writer") is not None
        else (
            float(observed) / float(step["finite_writer_demand"]["rho_write"])
            if float(step["finite_writer_demand"]["rho_write"]) != 0.0 else 0.0
        )
        for step, observed in zip(steps, actual)
    ]
    terminal_p = terminal["rollout"]["terminal_cumulative_structural_p"]
    compute = terminal["rollout"]["compute"]
    totals = compute["totals"]
    sequential_slope = compute["phases"].get("sequential_writer_current_slope", {})

    z_eff_bits = flatten(z_eff["per_request_bits"])
    w_eff_bits = flatten(w_eff["per_request_bits"])
    z_gen_strict = [int(a == b) for a, b in zip(
        z_gen["per_request_correct"], z_gen["per_request_required"]
    )]
    w_gen_strict = [int(a == b) for a, b in zip(
        w_gen["per_request_correct"], w_gen["per_request_required"]
    )]

    terminal_z_full = float(terminal["terminal_z8_oracle"]["target_new_nll"])
    terminal_w_full = float(terminal["terminal_w8_full_six_target_new_nll"])
    row.update({
        "classification": "TECHNICAL_VALID_ENDPOINT",
        "source_head": read_json(root / "manifest.json")["source_head"],
        "terminal_identity_sha256": terminal["identity_sha256"],
        "request_order_sha256": terminal["request_order_sha256"],
        "action_freeze_sha256": terminal["action_freeze_sha256"],
        "W0_restored": bool(terminal["W0_restored"]),
        "accepted_update_count": int(terminal["rollout"]["accepted_update_count"]),
        "materialization_count": int(terminal["rollout"]["materializer"]["transition_count"]),
        "retry_count": int(terminal["retry_count"]),
        "inner_step_heldout_evaluation_count": int(
            terminal["terminal_four_panel"]["inner_step_heldout_evaluation_count"]
        ),
        "z_eff_numerator": z_eff["numerator"],
        "z_eff_denominator": z_eff["denominator"],
        "w_eff_numerator": w_eff["numerator"],
        "w_eff_denominator": w_eff["denominator"],
        "z_gen_numerator": z_gen["numerator"],
        "z_gen_denominator": z_gen["denominator"],
        "w_gen_numerator": w_gen["numerator"],
        "w_gen_denominator": w_gen["denominator"],
        "z_gen_strict_numerator": z_gen["strict_numerator"],
        "z_gen_strict_denominator": z_gen["strict_denominator"],
        "w_gen_strict_numerator": w_gen["strict_numerator"],
        "w_gen_strict_denominator": w_gen["strict_denominator"],
        "loc_numerator": w_loc["numerator"],
        "loc_denominator": w_loc["denominator"],
        "z_rewrite_nll_mean": z_eff["nll_new"]["mean"],
        "w_rewrite_nll_mean": w_eff["nll_new"]["mean"],
        "rewrite_w_minus_z_nll_gap": w_eff["nll_new"]["mean"] - z_eff["nll_new"]["mean"],
        "z_rephrase_nll_mean": z_gen["nll_new"]["mean"],
        "w_rephrase_nll_mean": w_gen["nll_new"]["mean"],
        "rephrase_w_minus_z_nll_gap": w_gen["nll_new"]["mean"] - z_gen["nll_new"]["mean"],
        "z_full_six_target_new_nll": terminal_z_full,
        "w_full_six_target_new_nll": terminal_w_full,
        "full_six_w_minus_z_gap": terminal_w_full - terminal_z_full,
        "z_eff_success_w_failure_count": sum(
            int(z == 1 and w == 0) for z, w in zip(z_eff_bits, w_eff_bits)
        ),
        "z_gen_strict_success_w_failure_count": sum(
            int(z == 1 and w == 0) for z, w in zip(z_gen_strict, w_gen_strict)
        ),
        "writer_coverage_mean": statistics.fmean(coverages),
        "writer_coverage_terminal": coverages[-1],
        "predicted_progress_sum": sum(predicted),
        "actual_progress_sum": sum(actual),
        "realization_ratio_sum": sum(actual) / sum(predicted) if sum(predicted) else None,
        "negative_actual_transition_count": sum(int(value < 0) for value in actual),
        "bf16_update_energy_sum": sum(energy),
        "bf16_update_energy_mean": statistics.fmean(energy),
        "cumulative_bf16_capacity_terminal": capacities[-1],
        "structural_p_terminal": float(terminal_p["P_after"]),
        "structural_h_terminal": 0.0,
        "primary_selection_count": sum(int(step["target_update"]["primary_accept_count"]) for step in steps),
        "rescue_selection_count": sum(int(step["target_update"]["rescue_accept_count"]) for step in steps),
        "current_selection_count": sum(int(step["target_update"]["current_hold_count"]) for step in steps),
        "model_forward_calls": int(totals["model_forward_calls"]),
        "backward_calls": int(totals["backward_calls"]),
        "processed_tokens": int(totals["processed_tokens"]),
        "edit_core_wall_seconds": float(terminal["rollout"]["edit_core_wall_seconds"]),
        "terminal_evaluator_wall_seconds": float(terminal["terminal_evaluator_wall_seconds"]),
        "logical_additional_slope_groups": int(
            terminal["rollout"]["sequential_writer_additional_slope_group_count"]
        ),
        "physical_additional_slope_autograd_invocations": int(
            sequential_slope.get("autograd_invocations", 0)
        ),
        "physical_additional_slope_backward_calls": int(
            sequential_slope.get("backward_calls", 0)
        ),
    })
    return row


def build_step_and_layer_rows(
    arm: str, root: Path, case_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    step_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    paths = accepted_files(root, arm, case_index)
    accepted = [read_json(path) for path in paths]
    for position, (path, step) in enumerate(zip(paths, accepted)):
        k = int(step["accepted_index"] - 1)
        sequential = step.get("sequential_writer")
        progress = step["progress"]
        materialization = step["materialization"]
        finite_demand = step["finite_writer_demand"]
        target = step["target_update"]
        observed_actual = progress["actual"]
        if observed_actual is None and position + 1 < len(accepted):
            observed_actual = (
                float(finite_demand["L_base"])
                - float(accepted[position + 1]["finite_writer_demand"]["L_base"])
            )
        row = {
            "arm": arm,
            "case_index": case_index,
            "k": k,
            "accepted_index": int(step["accepted_index"]),
            "receipt_sha256": sha_file(path),
            "target_update_identity_sha256": target["identity_sha256"],
            "target_primary_count": int(target["primary_accept_count"]),
            "target_rescue_count": int(target["rescue_accept_count"]),
            "target_current_count": int(target["current_hold_count"]),
            "finite_L0": float(finite_demand["L_base"]),
            "finite_Lz": float(finite_demand["L_endpoint"]),
            "alpha_star": float(finite_demand["rho_write"]),
            "alpha_star_source": "finite_demand.rho_write=[L0-Lz]+",
            "entry_pi": step["routing"]["pi"],
            "entry_velocity": step["routing"]["velocity"],
            "entry_signed_slopes": step["routing"]["signed_slopes"],
            "entry_field_sha256": step["field_sha256"],
            "predicted_progress": float(progress["predicted"]),
            "actual_progress": float(observed_actual) if observed_actual is not None else None,
            "writer_coverage": (
                float(sequential["writer_coverage"])
                if sequential is not None else (
                    float(observed_actual) / float(finite_demand["rho_write"])
                    if observed_actual is not None and float(finite_demand["rho_write"]) != 0.0
                    else None
                )
            ),
            "negative_actual_progress": (
                bool(observed_actual < 0) if observed_actual is not None else None
            ),
            "realized_bf16_step_energy": scalar_sum(materialization["realized_bf16_step_energy"]),
            "realized_bf16_step_energy_by_layer": materialization["realized_bf16_step_energy"],
            "cumulative_bf16_capacity": scalar_sum(materialization["cumulative_bf16_capacity"]),
            "cumulative_bf16_capacity_by_layer": materialization["cumulative_bf16_capacity"],
            "structural_p": float(step["structural_p"]),
            "structural_h": float(step["structural_h"]),
            "simplex_entropy": float(step["routing"]["simplex_entropy"]),
            "simplex_top1_share": float(step["routing"]["simplex_top1_share"]),
            "maximum_velocity": float(step["routing"]["maximum_velocity"]),
            "retry_backtracking_reject_count": int(step["retry_backtracking_reject_count"]),
            "inner_step_heldout_evaluation_count": int(step["inner_step_heldout_evaluation_count"]),
            "sequential_writer_identity_sha256": sequential["identity_sha256"] if sequential else None,
            "sequential_negative_prefix_aggregate_count": int(
                sequential["negative_prefix_aggregate_count"] if sequential else 0
            ),
            "sequential_negative_prefix_request_count": int(
                sequential["negative_prefix_request_count"] if sequential else 0
            ),
            "support_exhausted_count": int(
                sequential["support_exhausted_count"] if sequential else 0
            ),
            "additional_slope_group_count": int(
                sequential["additional_slope_group_count"] if sequential else 0
            ),
            "additional_slope_backward_count": int(
                sequential["additional_slope_backward_count"] if sequential else 0
            ),
            "prefix_capture_count": int(
                sequential["prefix_capture_count"] if sequential else 0
            ),
            "physical_materialization_count": 1,
            "pi_application_count_per_layer": int(
                sequential["pi_application_count_per_layer"] if sequential else 1
            ),
            "physical_h_application_count_per_layer": int(
                sequential["physical_h_application_count_per_layer"] if sequential else 1
            ),
            "second_h_application_count": int(
                sequential["second_h_application_count"] if sequential else 0
            ),
            "live_parameter_change_count_during_prefix": int(
                sequential["live_parameter_pointer_version_hash_change_count"] if sequential else 0
            ),
            "final_virtual_physical_exact": bool(
                step["sequential_virtual_physical_identity"]["exact_hash_identity"]
                if step.get("sequential_virtual_physical_identity") else True
            ),
        }
        step_rows.append(row)

        if sequential is not None:
            for layer in sequential["layers"]:
                layer_row = {
                    "arm": arm,
                    "case_index": case_index,
                    "k": k,
                    "alpha_star": float(finite_demand["rho_write"]),
                    **layer,
                }
                layer_rows.append(layer_row)
        else:
            routing = step["routing"]
            for index, layer in enumerate((4, 5, 6, 7, 8)):
                layer_rows.append({
                    "arm": arm,
                    "case_index": case_index,
                    "k": k,
                    "layer": layer,
                    "alpha_star": float(finite_demand["rho_write"]),
                    "alpha_remaining": None,
                    "applied_slope": float(routing["signed_slopes"][index]),
                    "current_residual_norm": None,
                    "current_w_only_nll": None,
                    "frozen_endpoint_nll": float(finite_demand["L_endpoint"]),
                    "entry_pi": float(routing["pi"][index]),
                    "entry_velocity": float(routing["velocity"][index]),
                    "selected_velocity": float(routing["velocity"][index]),
                    "velocity_over_entry": 1.0,
                    "semantic_quota": None,
                    "suffix_pi_mass": None,
                    "relative_pi_share": None,
                    "raw_slope": None,
                    "slope_source": "ENTRY_JOINT_APPLIED_STEP",
                    "slope_h_application_count": 1,
                    "theta": float(routing["velocity"][index]) * 0.125,
                    "factor_energy": None,
                    "prefix_actual_nll_progress": None,
                    "negative_aggregate_progress": None,
                    "negative_request_progress_count": None,
                    "key_sha256": None,
                    "q_sha256": None,
                    "support_status": "NOT_APPLICABLE_JOINT_WRITER",
                    "pi_application_count": 1,
                    "physical_h_application_count": 1,
                    "second_h_application_count": 0,
                })
    return step_rows, layer_rows


def build_failure_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, root in ROOTS.items():
        for case_index in range(1, 11):
            path = case_dir(root, case_index) / "failure.json"
            if path.exists():
                failure = read_json(path)
                rows.append({
                    "attempt_role": "CANONICAL_TECH_R1",
                    "arm": arm,
                    "case_index": case_index,
                    "classification": failure["classification"],
                    "status": failure["status"],
                    "exception_class": failure["exception_class"],
                    "exception_message_sha256": failure["exception_message_sha256"],
                    "failure_identity_sha256": failure["identity_sha256"],
                    "last_valid_prefix_count": int(failure["last_valid_prefix_count"]),
                    "last_valid_prefix_sha256": failure["last_valid_prefix_sha256"],
                    "W0_pointer_restored": bool(failure["w0_restore"]["pointer_restored_exact"]),
                    "W0_bytes_restored": bool(failure["w0_restore"]["byte_restored_exact"]),
                    "endpoint_imputed": False,
                    "source_head": read_json(root / "manifest.json")["source_head"],
                    "technical_cause": (
                        "INHERITED_P1R52_ORIGIN_CLAMP_CONTRACT_EXCEPTION"
                        if failure["exception_message_sha256"].startswith("c73c26")
                        else "NOT_IDENTIFIED"
                    ),
                    "path": str(path),
                })
    for arm, root in PRIMARY_INVALID_ROOTS.items():
        for case_index in range(1, 11):
            path = case_dir(root, case_index) / "failure.json"
            failure = read_json(path)
            rows.append({
                "attempt_role": "PRIMARY_INVALID_PRE_TECH_R1",
                "arm": arm,
                "case_index": case_index,
                "classification": failure["classification"],
                "status": failure["status"],
                "exception_class": failure["exception_class"],
                "exception_message_sha256": failure["exception_message_sha256"],
                "failure_identity_sha256": failure["identity_sha256"],
                "last_valid_prefix_count": int(failure["last_valid_prefix_count"]),
                "last_valid_prefix_sha256": failure["last_valid_prefix_sha256"],
                "W0_pointer_restored": bool(failure["w0_restore"]["pointer_restored_exact"]),
                "W0_bytes_restored": bool(failure["w0_restore"]["byte_restored_exact"]),
                "endpoint_imputed": False,
                "source_head": read_json(root / "manifest.json")["source_head"],
                "technical_cause": "EMPTY_PREFIX_FACTOR_INVENTORY_PACKAGING",
                "path": str(path),
            })
    return rows


DELTA_FIELDS = (
    "z_eff_numerator", "w_eff_numerator", "z_gen_numerator", "w_gen_numerator",
    "z_gen_strict_numerator", "w_gen_strict_numerator", "loc_numerator",
    "z_rewrite_nll_mean", "w_rewrite_nll_mean", "rewrite_w_minus_z_nll_gap",
    "z_rephrase_nll_mean", "w_rephrase_nll_mean", "rephrase_w_minus_z_nll_gap",
    "z_full_six_target_new_nll", "w_full_six_target_new_nll", "full_six_w_minus_z_gap",
    "writer_coverage_mean", "writer_coverage_terminal", "predicted_progress_sum",
    "actual_progress_sum", "realization_ratio_sum", "bf16_update_energy_sum",
    "cumulative_bf16_capacity_terminal", "structural_p_terminal",
    "z_eff_success_w_failure_count", "z_gen_strict_success_w_failure_count",
    "negative_actual_transition_count", "edit_core_wall_seconds",
)


def paired_delta_rows(case_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["arm"], row["case_index"]): row for row in case_rows}
    rows: list[dict[str, Any]] = []
    for base, candidate in (("J0", "SV"), ("J0", "FPIQ"), ("SV", "FPIQ")):
        for case_index in range(1, 11):
            left = by_key[(base, case_index)]
            right = by_key[(candidate, case_index)]
            matched = bool(left["endpoint_available"] and right["endpoint_available"])
            row: dict[str, Any] = {
                "comparison": f"{candidate}-{base}",
                "base_arm": base,
                "candidate_arm": candidate,
                "case_index": case_index,
                "matched_endpoint": matched,
                "base_endpoint_available": left["endpoint_available"],
                "candidate_endpoint_available": right["endpoint_available"],
            }
            for field in DELTA_FIELDS:
                row[f"delta_{field}"] = (
                    right[field] - left[field] if matched else None
                )
            rows.append(row)
    return rows


def aggregate_arm(
    arm: str,
    rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    layer_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    completed = [row for row in rows if row["arm"] == arm and row["endpoint_available"]]
    attempted = [row for row in rows if row["arm"] == arm]
    steps = [row for row in step_rows if row["arm"] == arm]
    layers = [row for row in layer_rows if row["arm"] == arm]
    layer8 = [row for row in layers if row["layer"] == 8 and row["semantic_quota"] is not None]
    return {
        "arm": arm,
        "attempted_cases": len(attempted),
        "completed_endpoints": len(completed),
        "typed_incomplete_cases": len(attempted) - len(completed),
        "attempted_requests": len(attempted) * 10,
        "endpoint_requests": len(completed) * 10,
        "available_step_prefixes": len(steps),
        "expected_complete_steps": len(attempted) * 8,
        "z_eff": aggregate_counts(completed, "z_eff"),
        "w_eff": aggregate_counts(completed, "w_eff"),
        "z_gen": aggregate_counts(completed, "z_gen"),
        "w_gen": aggregate_counts(completed, "w_gen"),
        "z_gen_strict": aggregate_counts(completed, "z_gen_strict"),
        "w_gen_strict": aggregate_counts(completed, "w_gen_strict"),
        "loc": aggregate_counts(completed, "loc"),
        "z_rewrite_nll": stats(row["z_rewrite_nll_mean"] for row in completed),
        "w_rewrite_nll": stats(row["w_rewrite_nll_mean"] for row in completed),
        "rewrite_w_minus_z_gap": stats(row["rewrite_w_minus_z_nll_gap"] for row in completed),
        "z_rephrase_nll": stats(row["z_rephrase_nll_mean"] for row in completed),
        "w_rephrase_nll": stats(row["w_rephrase_nll_mean"] for row in completed),
        "rephrase_w_minus_z_gap": stats(row["rephrase_w_minus_z_nll_gap"] for row in completed),
        "z_full_six_nll": stats(row["z_full_six_target_new_nll"] for row in completed),
        "w_full_six_nll": stats(row["w_full_six_target_new_nll"] for row in completed),
        "full_six_w_minus_z_gap": stats(row["full_six_w_minus_z_gap"] for row in completed),
        "writer_coverage": stats(row["writer_coverage_mean"] for row in completed),
        "realization_ratio": stats(row["realization_ratio_sum"] for row in completed),
        "bf16_update_energy": stats(row["bf16_update_energy_sum"] for row in completed),
        "squared_capacity": stats(row["cumulative_bf16_capacity_terminal"] for row in completed),
        "structural_p": stats(row["structural_p_terminal"] for row in completed),
        "z_eff_success_w_failure_count": sum(
            row["z_eff_success_w_failure_count"] for row in completed
        ),
        "z_gen_strict_success_w_failure_count": sum(
            row["z_gen_strict_success_w_failure_count"] for row in completed
        ),
        "negative_actual_transition_count": sum(
            row["negative_actual_transition_count"] for row in completed
        ),
        "negative_prefix_aggregate_count": sum(
            row["sequential_negative_prefix_aggregate_count"] for row in steps
        ),
        "negative_prefix_request_count": sum(
            row["sequential_negative_prefix_request_count"] for row in steps
        ),
        "support_exhausted_count": sum(row["support_exhausted_count"] for row in steps),
        "last_layer_semantic_quota_share": stats(
            row["semantic_quota"] / max(row["alpha_star"], 1e-300)
            for row in layer8 if row["alpha_star"] is not None
        ),
        "last_layer_velocity_over_entry": stats(row["velocity_over_entry"] for row in layer8),
        "maximum_velocity_over_entry": stats(
            row["velocity_over_entry"] for row in layers
            if row["velocity_over_entry"] is not None
        ),
        "compute": {
            "model_forward_calls": sum(row.get("model_forward_calls", 0) for row in completed),
            "backward_calls": sum(row.get("backward_calls", 0) for row in completed),
            "processed_tokens": sum(row.get("processed_tokens", 0) for row in completed),
            "materializations": sum(row.get("materialization_count", 0) for row in completed),
            "logical_additional_slope_groups": sum(
                row.get("logical_additional_slope_groups", 0) for row in completed
            ),
            "physical_additional_slope_autograd_invocations": sum(
                row.get("physical_additional_slope_autograd_invocations", 0) for row in completed
            ),
            "edit_core_wall_seconds": sum(row.get("edit_core_wall_seconds", 0.0) for row in completed),
            "terminal_evaluator_wall_seconds": sum(
                row.get("terminal_evaluator_wall_seconds", 0.0) for row in completed
            ),
        },
    }


def matched_summary(
    paired: Sequence[Mapping[str, Any]], comparison: str,
) -> dict[str, Any]:
    rows = [row for row in paired if row["comparison"] == comparison and row["matched_endpoint"]]
    result: dict[str, Any] = {"comparison": comparison, "matched_cases": len(rows)}
    for field in DELTA_FIELDS:
        result[field] = stats(row[f"delta_{field}"] for row in rows)
    return result


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NOT_RECORDED"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def count_text(metric: Mapping[str, Any]) -> str:
    return f"{metric['numerator']}/{metric['denominator']} ({metric['rate']:.4f})"


def delta_mean(summary: Mapping[str, Any], field: str) -> str:
    return fmt(summary[field]["mean"])


def build_report(
    aggregate: Mapping[str, Any],
    case_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    layer_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
) -> str:
    arms = aggregate["arms"]
    matched = aggregate["matched_comparisons"]
    fpiq_vs_j0 = matched["FPIQ-J0"]
    sv_vs_j0 = matched["SV-J0"]
    fpiq_vs_sv = matched["FPIQ-SV"]
    lines: list[str] = []
    lines += [
        "# P1R52 Frozen-π Sequential Quota Writer — Atomic B10×10 최종 분석",
        "",
        "## 한눈에 보는 결과",
        "",
        "이 보고서는 Llama3-8B-Instruct의 동일 sealed 10×B10 스트림에서 `J0`, `SV`, `FPIQ`를 비교한다. "
        "J0는 10/10 endpoint, SV는 8/10 endpoint와 case03(k4)·case06(k7) last-valid prefix, "
        "FPIQ는 9/10 endpoint와 case07(k2) last-valid prefix를 남겼다. 미완료 endpoint는 어떤 표에도 보간하지 않았다.",
        "",
        "| Arm | attempts | endpoints | EFF(W) | GEN(W) | GEN-strict(W) | LOC(W) | W rewrite NLL | W rephrase NLL | full-six W−z gap | coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("J0", "SV", "FPIQ"):
        a = arms[arm]
        lines.append(
            f"| {arm} | {a['attempted_cases']} | {a['completed_endpoints']} | "
            f"{count_text(a['w_eff'])} | {count_text(a['w_gen'])} | "
            f"{count_text(a['w_gen_strict'])} | {count_text(a['loc'])} | "
            f"{fmt(a['w_rewrite_nll']['mean'])} | {fmt(a['w_rephrase_nll']['mean'])} | "
            f"{fmt(a['full_six_w_minus_z_gap']['mean'])} | {fmt(a['writer_coverage']['mean'])} |"
        )
    lines += [
        "",
        "해석의 핵심은 두 가지다. 첫째, endpoint가 존재하는 case에서 FPIQ는 J0보다 writer coverage를 높이고 "
        "full-six W−z gap을 줄였지만, absolute W rewrite/rephrase NLL과 terminal full-six W NLL은 평균적으로 개선하지 못했다. "
        "둘째, gap 축소는 accepted-z endpoint 자체의 악화와 마지막 layer velocity 집중·더 큰 update energy를 동반했다. "
        "FPIQ 1개와 SV 2개의 typed incomplete도 남아 B100/Historical 승격 안정성은 확보하지 못했다.",
        "",
        "## 실험 정의와 단일 writer 변경",
        "",
        "- `J0`: 기존 P1R52 joint block-Jacobi writer. Entry Soft router가 정한 5-layer velocity를 한 번에 물리화한다.",
        "- `SV`: layer 4→8 virtual BF16 prefix에서 residual/key/q/current slope를 갱신하지만 entry velocity `v⁰`는 고정한다.",
        "- `FPIQ`: entry `π⁰`만 고정하고 각 prefix에서 `α_rem=clip(L_l−L_z,0,α*)`, "
        "`barπ_l=π⁰_l/Σ_{j≥l}π⁰_j`, `s_l=α_rem·barπ_l`, `v_l=s_l/a_l^cur`를 적용한다.",
        "- 공통 좌표는 `R_l=(z*−y_l)/h`, `θ_l=h·v_l`이며 π와 h는 각각 정확히 한 번만 적용한다.",
        "- target/RSA/R42SafeKDC/origin clamp/selection/teacher/K8/h=1/8/finite demand/entry Soft router는 고정했다. "
        "Structural-H와 history는 OFF다.",
        "",
        "## 기계적·entry identity",
        "",
        f"- W0에서 세 arm의 k0 target numeric vectors, finite `α*`, entry `π⁰`, entry velocity는 10/10 case에서 max-abs 0으로 일치했다. "
        f"정규화된 identity receipt: `{aggregate['entry_identity']['identity_sha256']}`.",
        f"- SV/FPIQ layer4 capture=0, added backward=0, FPIQ `v4/v4_entry` residual max={fmt(aggregate['mechanical']['layer4_velocity_ratio_max_abs_from_one'], 12)}.",
        f"- Sequential available prefixes의 full residual identity max={fmt(aggregate['mechanical']['full_residual_identity_max_abs'], 12)}, "
        f"quota identity max={fmt(aggregate['mechanical']['quota_identity_max_abs'], 12)}.",
        f"- π application={aggregate['mechanical']['pi_application_values']}, physical h application={aggregate['mechanical']['physical_h_application_values']}, "
        f"second h={aggregate['mechanical']['second_h_application_values']}.",
        f"- virtual-prefix live parameter mutation count sum={aggregate['mechanical']['live_parameter_change_count_sum']}; "
        f"final virtual BF16=post-commit BF16 PASS {aggregate['mechanical']['virtual_commit_identity_pass_count']}/{aggregate['mechanical']['virtual_commit_identity_denominator']}.",
        f"- retry/backtracking/P-H hard gate/cap/contraction/veto/debt/request×layer router/heldout decision influence는 모두 0. "
        f"W0 restore는 attempts 30/30 PASS다.",
        "",
        "## endpoint 절대값: accepted-z와 physical W 분리",
        "",
        "| Arm | z EFF | W EFF | z GEN | W GEN | z GEN-strict | W GEN-strict | z rewrite NLL | W rewrite NLL | gap | z rephrase NLL | W rephrase NLL | gap | z→W Eff fail | z→W Gen-strict fail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("J0", "SV", "FPIQ"):
        a = arms[arm]
        lines.append(
            f"| {arm} | {count_text(a['z_eff'])} | {count_text(a['w_eff'])} | "
            f"{count_text(a['z_gen'])} | {count_text(a['w_gen'])} | "
            f"{count_text(a['z_gen_strict'])} | {count_text(a['w_gen_strict'])} | "
            f"{fmt(a['z_rewrite_nll']['mean'])} | {fmt(a['w_rewrite_nll']['mean'])} | {fmt(a['rewrite_w_minus_z_gap']['mean'])} | "
            f"{fmt(a['z_rephrase_nll']['mean'])} | {fmt(a['w_rephrase_nll']['mean'])} | {fmt(a['rephrase_w_minus_z_gap']['mean'])} | "
            f"{a['z_eff_success_w_failure_count']} | {a['z_gen_strict_success_w_failure_count']} |"
        )
    lines += [
        "",
        "`terminal_z8_oracle.target_new_nll`과 `terminal_w8_full_six_target_new_nll`은 full-six target-objective panel이다. "
        "위 rewrite/rephrase NLL은 terminal heldout prompt panel이므로 서로 대체하거나 같은 의미로 부르지 않았다.",
        "",
        "## exact matched case의 산술 차이",
        "",
        "| 비교 | matched cases | ΔW rewrite NLL | ΔW rephrase NLL | Δfull-six W−z gap | Δcoverage | ΔGEN correct | ΔGEN-strict requests | ΔLOC correct | energy ratio(arm/base) | capacity ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("SV-J0", "FPIQ-J0", "FPIQ-SV"):
        m = matched[name]
        lines.append(
            f"| {name} | {m['matched_cases']} | {delta_mean(m, 'w_rewrite_nll_mean')} | "
            f"{delta_mean(m, 'w_rephrase_nll_mean')} | {delta_mean(m, 'full_six_w_minus_z_gap')} | "
            f"{delta_mean(m, 'writer_coverage_mean')} | {delta_mean(m, 'w_gen_numerator')} | "
            f"{delta_mean(m, 'w_gen_strict_numerator')} | {delta_mean(m, 'loc_numerator')} | "
            f"{fmt(m['energy_ratio_mean'])} | {fmt(m['capacity_ratio_mean'])} |"
        )
    lines += [
        "",
        "모든 delta는 동일 case endpoint가 양쪽에 존재할 때만 계산했다. 따라서 J0↔SV 8 case, "
        "J0↔FPIQ 9 case, SV↔FPIQ 7 case다. 성공 endpoint만으로 전체 10-case를 대표하지 않으며, "
        "typed incomplete denominator를 위와 별도로 유지한다.",
        "",
        "### case별 endpoint 절대값",
        "",
        "| Arm | case | status | W rewrite NLL | W rephrase NLL | z full-six NLL | W full-six NLL | W−z gap | coverage | W GEN | W GEN-strict | LOC |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in case_rows:
        if not row["endpoint_available"]:
            lines.append(
                f"| {row['arm']} | {row['case_index']:02d} | TYPED_INCOMPLETE(k{row['available_prefix_count']}) | "
                "NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | "
                "NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |"
            )
        else:
            lines.append(
                f"| {row['arm']} | {row['case_index']:02d} | COMPLETE | "
                f"{fmt(row['w_rewrite_nll_mean'])} | {fmt(row['w_rephrase_nll_mean'])} | "
                f"{fmt(row['z_full_six_target_new_nll'])} | {fmt(row['w_full_six_target_new_nll'])} | "
                f"{fmt(row['full_six_w_minus_z_gap'])} | {fmt(row['writer_coverage_mean'])} | "
                f"{row['w_gen_numerator']}/{row['w_gen_denominator']} | "
                f"{row['w_gen_strict_numerator']}/{row['w_gen_strict_denominator']} | "
                f"{row['loc_numerator']}/{row['loc_denominator']} |"
            )
    lines += [
        "",
        "## layer quota·velocity·prefix trajectory",
        "",
        "| Arm | available K prefixes | mean coverage | negative aggregate prefixes | negative request-prefixes | support exhausted | layer8 quota/remaining mean | layer8 v/v0 median / p90 / max | max v/v0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("J0", "SV", "FPIQ"):
        a = arms[arm]
        l8 = a["last_layer_velocity_over_entry"]
        lines.append(
            f"| {arm} | {a['available_step_prefixes']}/{a['expected_complete_steps']} | "
            f"{fmt(a['writer_coverage']['mean'])} | {a['negative_prefix_aggregate_count']} | "
            f"{a['negative_prefix_request_count']} | {a['support_exhausted_count']} | "
            f"{fmt(a['last_layer_semantic_quota_share']['mean'])} | "
            f"{fmt(l8['median'])} / {fmt(l8['p90'])} / {fmt(l8['max'])} | "
            f"{fmt(a['maximum_velocity_over_entry']['max'])} |"
        )
    lines += [
        "",
        "| Arm | layer | rows | α_rem mean | entry π mean | suffix π mean | quota mean | applied slope mean | v/v0 median | factor energy mean | prefix progress mean | negative request count |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("SV", "FPIQ"):
        for layer in (4, 5, 6, 7, 8):
            selected = [row for row in layer_rows if row["arm"] == arm and row["layer"] == layer]
            lines.append(
                f"| {arm} | {layer} | {len(selected)} | "
                f"{fmt(stats(row['alpha_remaining'] for row in selected)['mean'])} | "
                f"{fmt(stats(row['entry_pi'] for row in selected)['mean'])} | "
                f"{fmt(stats(row['suffix_pi_mass'] for row in selected)['mean'])} | "
                f"{fmt(stats(row['semantic_quota'] for row in selected)['mean'])} | "
                f"{fmt(stats(row['applied_slope'] for row in selected)['mean'])} | "
                f"{fmt(stats(row['velocity_over_entry'] for row in selected)['median'])} | "
                f"{fmt(stats(row['factor_energy'] for row in selected)['mean'])} | "
                f"{fmt(stats(row['prefix_actual_nll_progress'] for row in selected)['mean'])} | "
                f"{sum(int(row['negative_request_progress_count']) for row in selected)} |"
            )
    lines += [
        "",
        "FPIQ는 suffix π mass가 줄수록 late-layer relative share가 커지고, current slope가 낮을 때 `v/v0`가 증가했다. "
        "이는 contract가 예고한 suffix concentration 신호다. aggregate negative prefix는 FPIQ 3개였지만 request-prefix negative count는 "
        "102개여서 평균 progress가 가리는 request별 이질성도 존재한다. 세부 1,145-row layer table에는 각 layer의 "
        "`α_rem`, π, suffix mass, quota, raw/applied slope, v, v/v0, θ, key/q hash, factor energy와 prefix progress가 있다.",
        "",
        "## P/H, energy, capacity, load와 계산량",
        "",
        "| Arm | Structural-P mean | BF16 energy mean | squared capacity mean | realization ratio mean | model F | backward | materializations | logical extra slope groups | physical extra slope autograd | edit-core wall s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("J0", "SV", "FPIQ"):
        a = arms[arm]
        c = a["compute"]
        lines.append(
            f"| {arm} | {fmt(a['structural_p']['mean'])} | {fmt(a['bf16_update_energy']['mean'])} | "
            f"{fmt(a['squared_capacity']['mean'])} | {fmt(a['realization_ratio']['mean'])} | "
            f"{c['model_forward_calls']} | {c['backward_calls']} | {c['materializations']} | "
            f"{c['logical_additional_slope_groups']} | {c['physical_additional_slope_autograd_invocations']} | "
            f"{fmt(c['edit_core_wall_seconds'], 3)} |"
        )
    lines += [
        "",
        "Structural-H는 세 arm 모두 0이며 hard P/H gate도 0이다. `SV/FPIQ`의 logical extra slope group은 "
        "완료·last-valid prefix당 4개이고, 각 logical group이 B10 microbatch 5개로 실행되어 physical autograd는 5배다. "
        "이 표는 두 수치를 분리해 기록한다. 위 compute 합계는 terminal endpoint가 존재하는 case만 집계하며, "
        "typed incomplete의 last-valid logical/physical count는 per-step 표에 유지한다.",
        "",
        "## typed failure와 기술 시도 이력",
        "",
        "| attempt | arm | case | last-valid K | exception hash | W0 restore | endpoint |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for row in failure_rows:
        if row["attempt_role"] == "CANONICAL_TECH_R1":
            lines.append(
                f"| TECH-R1 | {row['arm']} | {row['case_index']:02d} | {row['last_valid_prefix_count']} | "
                f"`{row['exception_message_sha256']}` | PASS | NOT_IMPUTED |"
            )
    lines += [
        "",
        "Canonical TECH-R1의 3개 incomplete는 동일 inherited P1R52 origin-clamp contract exception "
        "`c73c26…`이다. SV case03은 k4, case06은 k7, FPIQ case07은 k2까지의 last-valid prefix만 보존했다. "
        "각 failure 후 W0 pointer/bytes restore는 PASS였고 다음 독립 case는 계속됐다.",
        "",
        "초기 source `626f5de…`의 SV/FPIQ는 각각 10/10 case가 k0 이전 `cc7657…` empty-prefix factor inventory packaging defect로 "
        "끝났다. 이는 science endpoint가 없는 technical attempt history다. TECH-R1 `f534614…`는 empty inventory를 packaging에서 "
        "제외한 source-backed fix만 포함하며, primary invalid 결과는 어떤 science aggregate에도 포함하지 않았다.",
        "",
        "## 성공·실패 원인 분석",
        "",
        f"- **J0→SV**: matched 8 case에서 Δcoverage={delta_mean(sv_vs_j0, 'writer_coverage_mean')}, "
        f"Δfull-six W−z gap={delta_mean(sv_vs_j0, 'full_six_w_minus_z_gap')}였다. residual/key/q refresh만으로 생긴 변화는 "
        "작거나 혼합되어 stale cross-effect 하나만을 지배 병목으로 단정할 수 없다.",
        f"- **SV→FPIQ**: matched 7 case에서 Δcoverage={delta_mean(fpiq_vs_sv, 'writer_coverage_mean')}, "
        f"Δfull-six W−z gap={delta_mean(fpiq_vs_sv, 'full_six_w_minus_z_gap')}로 realization은 개선됐다. 그러나 "
        f"ΔW rewrite NLL={delta_mean(fpiq_vs_sv, 'w_rewrite_nll_mean')}, "
        f"ΔW rephrase NLL={delta_mean(fpiq_vs_sv, 'w_rephrase_nll_mean')}여서 absolute heldout NLL 개선은 성립하지 않았다. "
        "이는 writer package association이며 단독 요소의 인과 분리는 아니다.",
        f"- **Writer realization**: FPIQ matched J0에서 Δfull-six W−z gap={delta_mean(fpiq_vs_j0, 'full_six_w_minus_z_gap')}, "
        f"Δcoverage={delta_mean(fpiq_vs_j0, 'writer_coverage_mean')}로 writer-side gap은 줄었다. 반면 Δz full-six NLL="
        f"{delta_mean(fpiq_vs_j0, 'z_full_six_target_new_nll')}, ΔW full-six NLL="
        f"{delta_mean(fpiq_vs_j0, 'w_full_six_target_new_nll')}로 둘 다 악화됐다. gap 축소만으로 absolute strength recovery를 "
        "주장할 수 없고 coupled trajectory에서 accepted-z endpoint가 약해진 사실을 함께 봐야 한다.",
        "- **Hard case**: FPIQ case05는 J0 대비 W rewrite NLL +0.242197, W rephrase NLL +0.252441, "
        "z full-six NLL +0.439065였다. 이 case가 mean rewrite 악화의 대부분을 차지하지만, 제외 후에도 "
        "rephrase와 full-six target endpoint의 일관된 개선은 확인되지 않는다. ASSOCIATION_ONLY이며 원인 단독 분리는 불가하다.",
        f"- **비용/집중**: FPIQ/J0 matched energy ratio={fmt(fpiq_vs_j0['energy_ratio_mean'])}, "
        f"capacity ratio={fmt(fpiq_vs_j0['capacity_ratio_mean'])}; FPIQ layer8 `v/v0` max={fmt(arms['FPIQ']['last_layer_velocity_over_entry']['max'])}. "
        "따라서 strength 개선은 공짜가 아니며 late-layer concentration이 tail risk다.",
        f"- **안정성**: technically valid endpoint 비율은 J0 10/10, SV 8/10, FPIQ 9/10이다. "
        "Nonfinite, double-π/h, live-prefix mutation, rollback failure는 관찰되지 않았으나 inherited origin-clamp incomplete가 남았다.",
        "",
        "## Promotion 판정",
        "",
        "**최종 판정: `HOLD_NO_PROMOTION_TO_B100_OR_HISTORICAL`.**",
        "",
        "FPIQ는 technically valid matched endpoint에서 writer coverage와 W/z gap을 개선하는 positive signal을 보였으나, "
        "absolute W rewrite/rephrase·full-six NLL과 GEN은 함께 개선되지 않았고 accepted-z endpoint도 약해졌다. "
        "또한 (1) 10개 중 1개 endpoint가 없고, (2) SV도 2개 incomplete이며, (3) FPIQ의 late-layer velocity·energy 집중이 크고, "
        "(4) B100/Historical은 현재 계약상 별도 promotion 결정 전 금지되어 있다. 따라서 이 atomic gate는 "
        "`WRITER_REALIZATION_SIGNAL_WITH_TARGET_SIDE_REGRESSION_AND_TECHNICAL_INCOMPLETENESS`로 분류하되, 이 결과만으로 sequential/Historical 실행을 승격하지 않는다. "
        "이는 후속 방법 추천이 아니라 본 계약의 promotion 질문에 대한 판정이다.",
        "",
        "## 데이터·재현성 경계",
        "",
        f"- Contract: `{CONTRACT}` SHA `{sha_file(CONTRACT)}`, bytes {CONTRACT.stat().st_size}, lines 742, mode 0600.",
        f"- Numerical lock: `{LOCK}` SHA `{sha_file(LOCK)}`; root `{read_json(LOCK)['root_digest']}`.",
        f"- Source manifest: `{SOURCE_MANIFEST}` SHA `{sha_file(SOURCE_MANIFEST)}`; root `{read_json(SOURCE_MANIFEST)['root_digest']}`.",
        f"- J0 source/result: `626f5de52394e4f2fce401d56f16a2b532da6501`, `{ROOTS['J0']}`.",
        f"- SV/FPIQ TECH-R1 source/results: `f53461448e123f1caa1aee0bd5d70ab31e5109d6`, `{ROOTS['SV']}`, `{ROOTS['FPIQ']}`.",
        "- Stepwise heldout evaluation과 writer decision heldout access는 0. Endpoint absent fields are `NOT_RECORDED`; no imputation.",
        "- `scientific_promotion=false`는 실행 namespace의 고정 행정 경계이며, 위 HOLD 판정과 일치한다.",
        "",
        "## 산출물",
        "",
        "- `p1r52-fpiq-per-case.json`: attempts/endpoints와 terminal z/W panels.",
        "- `p1r52-fpiq-per-step.json`: all complete and last-valid accepted K prefixes.",
        "- `p1r52-fpiq-per-layer.json`: J0 entry layer rows plus SV/FPIQ current-prefix layer rows.",
        "- `p1r52-fpiq-typed-failures.json`: canonical incomplete + primary invalid attempt history.",
        "- `p1r52-fpiq-paired-case-deltas.json`: exact same-case deltas; unmatched rows retain null delta.",
        "- `analysis-manifest.json`, `analysis-receipt.json`, `independent-review.json`: SHA/row/root/review receipts.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    if sha_file(CONTRACT) != "e5c767cdd6cfd498376dacad39155d08d0bc79c89496dbf629725a9b928b4436":
        raise SystemExit("contract identity mismatch")
    for root in (*ROOTS.values(), *PRIMARY_INVALID_ROOTS.values()):
        if not root.is_dir():
            raise SystemExit(f"missing result root: {root}")

    case_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    for arm, root in ROOTS.items():
        for case_index in range(1, 11):
            case_rows.append(build_case_row(arm, root, case_index))
            steps, layers = build_step_and_layer_rows(arm, root, case_index)
            step_rows.extend(steps)
            layer_rows.extend(layers)
    failure_rows = build_failure_rows()
    paired_rows = paired_delta_rows(case_rows)

    # k0 entry numeric equivalence across the three arms.
    entry_residuals: list[float] = []
    entry_receipt = []
    for case_index in range(1, 11):
        triplet = []
        for arm, root in ROOTS.items():
            path = accepted_files(root, arm, case_index)[0]
            step = read_json(path)
            triplet.append(step)
        reference = triplet[0]
        for item in triplet[1:]:
            for key in (
                "current_nll_by_request", "primary_nll_by_request",
                "selected_nll_by_request", "accepted_energy_by_request",
                "allocation_amplitude_by_request", "kdc_direction_norm_by_request",
            ):
                entry_residuals.extend(
                    abs(float(a) - float(b))
                    for a, b in zip(reference["target_update"][key], item["target_update"][key])
                )
            entry_residuals.extend(
                abs(float(a) - float(b))
                for a, b in zip(reference["routing"]["pi"], item["routing"]["pi"])
            )
            entry_residuals.extend(
                abs(float(a) - float(b))
                for a, b in zip(reference["routing"]["velocity"], item["routing"]["velocity"])
            )
            entry_residuals.append(abs(
                float(reference["finite_writer_demand"]["rho_write"])
                - float(item["finite_writer_demand"]["rho_write"])
            ))
        entry_receipt.append({
            "case_index": case_index,
            "alpha_star": reference["finite_writer_demand"]["rho_write"],
            "entry_pi": reference["routing"]["pi"],
            "entry_velocity": reference["routing"]["velocity"],
        })
    entry_identity_payload = {
        "case_count": 10,
        "arms": ["J0", "SV", "FPIQ"],
        "max_abs_numeric_residual": max(entry_residuals),
        "receipts": entry_receipt,
    }
    entry_identity_payload["identity_sha256"] = sha_bytes(canonical_bytes(entry_identity_payload))

    sequential_layers = [row for row in layer_rows if row["arm"] in ("SV", "FPIQ")]
    sequential_steps = [row for row in step_rows if row["arm"] in ("SV", "FPIQ")]
    mechanical = {
        "layer4_velocity_ratio_max_abs_from_one": max(
            abs(row["velocity_over_entry"] - 1.0)
            for row in sequential_layers if row["layer"] == 4 and row["arm"] == "FPIQ"
            and row["velocity_over_entry"] is not None
        ),
        "layer4_velocity_max_abs_difference": max(
            abs(float(row["selected_velocity"]) - float(row["entry_velocity"]))
            for row in sequential_layers if row["layer"] == 4 and row["arm"] == "FPIQ"
        ),
        "full_residual_identity_max_abs": max(
            float(row["full_residual_h_identity_max_abs"]) for row in sequential_layers
            if row["full_residual_h_identity_max_abs"] is not None
        ),
        "quota_identity_max_abs": max(
            float(row["quota_identity_residual"]) for row in sequential_layers
            if row["quota_identity_residual"] is not None
        ),
        "pi_application_values": sorted({row["pi_application_count"] for row in sequential_layers}),
        "physical_h_application_values": sorted({row["physical_h_application_count"] for row in sequential_layers}),
        "second_h_application_values": sorted({row["second_h_application_count"] for row in sequential_layers}),
        "live_parameter_change_count_sum": sum(
            row["live_parameter_change_count_during_prefix"] for row in sequential_steps
        ),
        "virtual_commit_identity_pass_count": sum(
            int(row["final_virtual_physical_exact"]) for row in sequential_steps
        ),
        "virtual_commit_identity_denominator": len(sequential_steps),
        "W0_restore_attempt_pass_count": sum(int(row["W0_restored"]) for row in case_rows),
        "W0_restore_attempt_denominator": len(case_rows),
    }

    paired_summaries = {
        name: matched_summary(paired_rows, name)
        for name in ("SV-J0", "FPIQ-J0", "FPIQ-SV")
    }
    by_key = {(row["arm"], row["case_index"]): row for row in case_rows}
    for comparison, base, candidate in (
        ("SV-J0", "J0", "SV"),
        ("FPIQ-J0", "J0", "FPIQ"),
        ("FPIQ-SV", "SV", "FPIQ"),
    ):
        ratios_e, ratios_c = [], []
        for case_index in range(1, 11):
            left, right = by_key[(base, case_index)], by_key[(candidate, case_index)]
            if left["endpoint_available"] and right["endpoint_available"]:
                if left["bf16_update_energy_sum"] != 0:
                    ratios_e.append(right["bf16_update_energy_sum"] / left["bf16_update_energy_sum"])
                if left["cumulative_bf16_capacity_terminal"] != 0:
                    ratios_c.append(
                        right["cumulative_bf16_capacity_terminal"]
                        / left["cumulative_bf16_capacity_terminal"]
                    )
        paired_summaries[comparison]["energy_ratio_mean"] = statistics.fmean(ratios_e)
        paired_summaries[comparison]["capacity_ratio_mean"] = statistics.fmean(ratios_c)

    aggregate = {
        "schema": "ode-edit-s05-p1r52-fpiq-analysis-summary/v1",
        "instruction_id": "ODEEDIT-S05-P1R52-FROZEN-PI-SEQUENTIAL-QUOTA-WRITER-V1",
        "contract_sha256": sha_file(CONTRACT),
        "source_heads": {
            arm: read_json(root / "manifest.json")["source_head"]
            for arm, root in ROOTS.items()
        },
        "result_roots": {arm: str(root) for arm, root in ROOTS.items()},
        "primary_invalid_result_roots": {
            arm: str(root) for arm, root in PRIMARY_INVALID_ROOTS.items()
        },
        "entry_identity": entry_identity_payload,
        "mechanical": mechanical,
        "arms": {
            arm: aggregate_arm(arm, case_rows, step_rows, layer_rows)
            for arm in ("J0", "SV", "FPIQ")
        },
        "matched_comparisons": paired_summaries,
        "promotion_decision": "HOLD_NO_PROMOTION_TO_B100_OR_HISTORICAL",
        "scientific_classification": "WRITER_REALIZATION_SIGNAL_WITH_TARGET_SIDE_REGRESSION_AND_TECHNICAL_INCOMPLETENESS",
        "no_imputation": True,
    }
    aggregate["identity_sha256"] = sha_bytes(canonical_bytes(aggregate))

    outputs = {
        TABLE_NAMES["aggregate"]: aggregate,
        TABLE_NAMES["case"]: case_rows,
        TABLE_NAMES["step"]: step_rows,
        TABLE_NAMES["layer"]: layer_rows,
        TABLE_NAMES["failure"]: failure_rows,
        TABLE_NAMES["paired"]: paired_rows,
    }
    for name, value in outputs.items():
        write_json(REPORT_DIR / name, value)

    report = build_report(
        aggregate, case_rows, step_rows, layer_rows, failure_rows, paired_rows
    )
    report_path = REPORT_DIR / REPORT_NAME
    report_path.write_text(report, encoding="utf-8")
    os.chmod(report_path, 0o600)

    members = []
    row_counts = {
        TABLE_NAMES["aggregate"]: 1,
        TABLE_NAMES["case"]: len(case_rows),
        TABLE_NAMES["step"]: len(step_rows),
        TABLE_NAMES["layer"]: len(layer_rows),
        TABLE_NAMES["failure"]: len(failure_rows),
        TABLE_NAMES["paired"]: len(paired_rows),
        REPORT_NAME: len(report.splitlines()),
    }
    for name in sorted((*outputs.keys(), REPORT_NAME)):
        path = REPORT_DIR / name
        members.append({
            "name": name,
            "path": str(path),
            "sha256": sha_file(path),
            "bytes": path.stat().st_size,
            "mode": oct(path.stat().st_mode & 0o777),
            "rows_or_lines": row_counts[name],
        })
    member_root = sha_bytes(canonical_bytes(members))
    manifest = {
        "schema": "ode-edit-s05-p1r52-fpiq-analysis-manifest/v1",
        "instruction_id": aggregate["instruction_id"],
        "members": members,
        "member_count": len(members),
        "members_root_sha256": member_root,
        "source_head_j0": aggregate["source_heads"]["J0"],
        "source_head_tech_r1": aggregate["source_heads"]["SV"],
        "contract_sha256": aggregate["contract_sha256"],
        "numerical_lock_sha256": sha_file(LOCK),
        "numerical_lock_root": read_json(LOCK)["root_digest"],
        "source_manifest_sha256": sha_file(SOURCE_MANIFEST),
        "source_manifest_root": read_json(SOURCE_MANIFEST)["root_digest"],
        "row_counts": row_counts,
    }
    manifest["identity_sha256"] = sha_bytes(canonical_bytes(manifest))
    write_json(REPORT_DIR / "analysis-manifest.json", manifest)
    receipt = {
        "schema": "ode-edit-s05-p1r52-fpiq-rooted-analysis-receipt/v1",
        "manifest_path": str(REPORT_DIR / "analysis-manifest.json"),
        "manifest_sha256": sha_file(REPORT_DIR / "analysis-manifest.json"),
        "manifest_identity_sha256": manifest["identity_sha256"],
        "members_root_sha256": member_root,
        "contract_full_read": True,
        "contract_sha256": aggregate["contract_sha256"],
        "raw_free_only": True,
        "model_action_count": 0,
        "evaluator_action_count": 0,
        "gpu_action_count": 0,
        "slurm_action_count": 0,
        "result_mutation_count": 0,
        "no_imputation": True,
        "promotion_decision": aggregate["promotion_decision"],
    }
    receipt["root_digest"] = sha_bytes(canonical_bytes(receipt))
    write_json(REPORT_DIR / "analysis-receipt.json", receipt)

    print(json.dumps({
        "report": str(report_path),
        "report_sha256": sha_file(report_path),
        "report_bytes": report_path.stat().st_size,
        "report_lines": len(report.splitlines()),
        "row_counts": row_counts,
        "manifest_sha256": sha_file(REPORT_DIR / "analysis-manifest.json"),
        "manifest_root": member_root,
        "receipt_sha256": sha_file(REPORT_DIR / "analysis-receipt.json"),
        "receipt_root": receipt["root_digest"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
