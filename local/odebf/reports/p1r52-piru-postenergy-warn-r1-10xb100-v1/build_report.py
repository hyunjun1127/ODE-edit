#!/usr/bin/env python3
"""Build the raw-free terminal package for the PIR-U Post-Energy-Warn R1 run."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


WORKTREE = Path(__file__).resolve().parents[4]
REPORT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = WORKTREE / "local/odebf/results/s05-p1r52-pir-u-soft-sequential-structuralh-on-10xb100-postenergy-warn-r1-tech-r1-v1"
FAILED_ROOT = WORKTREE / "local/odebf/results/s05-p1r52-pir-u-soft-sequential-structuralh-on-10xb100-postenergy-warn-r1-v1"
FOUR_ARM_DIR = WORKTREE / "local/odebf/reports/p1r52-llama-sequential-10xb100-fourarm-v1"
ACCEPTED_Z_DIR = WORKTREE / "local/odebf/reports/p1r52-b100-accepted-z-rephrase-observation-v1"

REPORT_NAME = "p1r52-piru-postenergy-warn-r1-10xb100-factual-ko.md"
CORE_NAME = "p1r52-piru-postenergy-core-comparison.json"
BATCH_NAME = "p1r52-piru-postenergy-per-batch.json"
H_NAME = "p1r52-piru-postenergy-per-h-decision.json"
LAYER_NAME = "p1r52-piru-postenergy-per-layer.json"
ZW_NAME = "p1r52-piru-postenergy-per-z-w-request.json"
HARD_NAME = "p1r52-piru-postenergy-hard-cohorts.json"
COMPUTE_NAME = "p1r52-piru-postenergy-compute-integrity.json"
MANIFEST_NAME = "analysis-manifest.json"
RECEIPT_NAME = "rooted-analysis-receipt.json"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def identity(payload: Any) -> str:
    return sha256_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def flatten(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if isinstance(value, list):
            out.extend(flatten(value))
        elif value is not None:
            out.append(float(value))
    return out


def quantile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def dist(values: Iterable[Any]) -> dict[str, Any]:
    xs = flatten(values)
    if not xs:
        return {"count": 0, "mean": None, "median": None, "p90": None, "raw_max": None, "raw_min": None}
    return {
        "count": len(xs),
        "mean": sum(xs) / len(xs),
        "median": statistics.median(xs),
        "p90": quantile(xs, 0.90),
        "raw_max": max(xs),
        "raw_min": min(xs),
    }


def pct(num: int | float, den: int | float) -> str:
    return "NOT_RECORDED" if not den else f"{num}/{den} ({100.0 * num / den:.2f}%)"


def f6(value: Any) -> str:
    return "NOT_RECORDED" if value is None else f"{float(value):.6f}"


def metric_ref(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_numerator": metric["numerator"],
        "prompt_denominator": metric["denominator"],
        "prompt_rate": metric["rate"],
        "strict_request_numerator": metric.get("strict_numerator", metric["numerator"]),
        "strict_request_denominator": metric.get("strict_denominator", metric["denominator"]),
        "strict_request_rate": metric.get("strict_rate", metric["rate"]),
        "target_new_nll": metric.get("target_new_nll"),
        "target_true_nll": metric.get("target_old_nll"),
        "target_true_minus_new_margin": metric.get("margin"),
    }


def metric_new(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_numerator": metric["prompt_numerator"],
        "prompt_denominator": metric["prompt_denominator"],
        "prompt_rate": metric["prompt_rate"],
        "strict_request_numerator": metric["strict_request_numerator"],
        "strict_request_denominator": metric["strict_request_denominator"],
        "strict_request_rate": metric["strict_request_rate"],
        "target_new_nll": metric.get("target_new_nll"),
        "target_true_nll": metric.get("target_old_nll"),
        "target_true_minus_new_margin": metric.get("target_old_minus_new_margin"),
    }


def panel_ref(panel: dict[str, Any]) -> dict[str, Any]:
    return {
        "rewrite_success": metric_ref(panel["rewrite_success"]),
        "rewrite_acc": metric_ref(panel["rewrite_acc"]),
        "paraphrase_success": metric_ref(panel["paraphrase_success"]),
        "paraphrase_acc": metric_ref(panel["paraphrase_acc"]),
        "locality": panel["locality"],
    }


def panel_new(panel: dict[str, Any]) -> dict[str, Any]:
    return {
        "rewrite_success": metric_new(panel["metrics"]["rewrite_success"]),
        "rewrite_acc": metric_new(panel["metrics"]["rewrite_acc"]),
        "paraphrase_success": metric_new(panel["metrics"]["paraphrase_success"]),
        "paraphrase_acc": metric_new(panel["metrics"]["paraphrase_acc"]),
        "locality": panel["locality"],
    }


def summarize_z_metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    metrics = [row[key] for row in rows]
    new_nll = flatten(m["target_new_nll_by_request"] for m in metrics)
    true_nll = flatten(m["target_true_nll_by_request"] for m in metrics)
    margins = [old - new for old, new in zip(true_nll, new_nll)]
    return {
        "prompt_numerator": sum(m["prompt_numerator"] for m in metrics),
        "prompt_denominator": sum(m["prompt_denominator"] for m in metrics),
        "prompt_rate": sum(m["prompt_numerator"] for m in metrics) / sum(m["prompt_denominator"] for m in metrics),
        "strict_request_numerator": sum(m["strict_request_numerator"] for m in metrics),
        "strict_request_denominator": sum(m["strict_request_denominator"] for m in metrics),
        "strict_request_rate": sum(m["strict_request_numerator"] for m in metrics) / sum(m["strict_request_denominator"] for m in metrics),
        "target_new_nll": dist(new_nll),
        "target_true_nll": dist(true_nll),
        "target_true_minus_new_margin": dist(margins),
    }


def main() -> None:
    terminal = load(RESULT_ROOT / "terminal.json")
    result_manifest = load(RESULT_ROOT / "manifest.json")
    h_payload = load(RESULT_ROOT / "structural-h-decisions.json")
    h_rows = h_payload["rows"]
    norm_payload = load(RESULT_ROOT / "actual-update-norm-share.json")
    norm_rows = norm_payload["rows"]
    z_payload = load(RESULT_ROOT / "z-w-writer-realization.json")
    z_rows = z_payload["rows"]
    checkpoint_rows = load(RESULT_ROOT / "b1-b10-post-final-checkpoints.json")["rows"]
    cohort_rows = load(RESULT_ROOT / "batch-age-post-final-cohorts.json")["rows"]
    retention_rows = load(RESULT_ROOT / "immediate-post-final-w10-requests.json")["rows"]
    four_arm = load(FOUR_ARM_DIR / "p1r52-b100-four-arm-aggregate.json")
    accepted_ref_rows = load(ACCEPTED_Z_DIR / "p1r52-accepted-z-aggregates.json")["rows"]
    accepted_ref = {row["arm"]: row for row in accepted_ref_rows}

    labels = {
        "memit": "Official EasyEdit MEMIT Sequential",
        "alphaedit": "Official EasyEdit AlphaEdit Sequential (cache_c ON)",
        "r52_h_on": "P1R52 J0 Structural-H ON",
        "piru": "P1R52 PIR-U Structural-H ON / Post-Energy-Warn R1",
        "w0": "공통 original W0 pre-edit",
    }

    common_w0 = panel_ref(four_arm["common_W0_B1_panel"])
    reference_panels: dict[str, dict[str, Any]] = {}
    for arm in ("memit", "alphaedit", "r52_h_on"):
        reference_panels[arm] = {
            "label": labels[arm],
            "immediate_post": panel_ref(four_arm["arms"][arm]["immediate_post_B100_panels"]),
            "final_W10": panel_ref(four_arm["arms"][arm]["final_W10_B1000"]),
            "source_scope": "SEALED_DIFFERENT_SOURCE_REFERENCE",
        }
    piru_immediate = panel_new(terminal["immediate_post_all_B100"])
    piru_final = panel_new(terminal["final_B1000"])

    piru_accepted = {
        "rewrite_success": summarize_z_metric(z_rows, "direct_z_rewrite_success"),
        "rewrite_acc": summarize_z_metric(z_rows, "direct_z_rewrite_accuracy"),
        "paraphrase_success": summarize_z_metric(z_rows, "direct_z_rephrase_success"),
        "paraphrase_acc": summarize_z_metric(z_rows, "direct_z_rephrase_accuracy"),
    }
    accepted_comparison: dict[str, Any] = {}
    for arm in ("memit", "alphaedit", "r52_h_on"):
        accepted_comparison[arm] = {
            "label": labels[arm],
            "accepted_z": accepted_ref[arm]["accepted_z"],
            "source_scope": "SEALED_DIFFERENT_SOURCE_REFERENCE",
        }
    accepted_comparison["piru"] = {
        "label": labels["piru"],
        "accepted_z": piru_accepted,
        "source_scope": "INLINE_SAME_RUN_OBSERVATION",
    }

    j0_final = reference_panels["r52_h_on"]["final_W10"]
    delta_vs_j0: dict[str, Any] = {}
    for metric in ("rewrite_success", "rewrite_acc", "paraphrase_success", "paraphrase_acc"):
        delta_vs_j0[metric] = {
            "prompt_numerator": piru_final[metric]["prompt_numerator"] - j0_final[metric]["prompt_numerator"],
            "prompt_denominator": piru_final[metric]["prompt_denominator"],
            "strict_request_numerator": piru_final[metric]["strict_request_numerator"] - j0_final[metric]["strict_request_numerator"],
            "strict_request_denominator": piru_final[metric]["strict_request_denominator"],
            "target_new_nll_mean": piru_final[metric]["target_new_nll"]["mean"] - j0_final[metric]["target_new_nll"]["mean"],
        }
    delta_vs_j0["locality"] = {
        "numerator": piru_final["locality"]["numerator"] - j0_final["locality"]["numerator"],
        "denominator": piru_final["locality"]["denominator"],
        "rate": piru_final["locality"]["rate"] - j0_final["locality"]["rate"],
    }

    core = {
        "schema": "ode-edit-s05-p1r52-piru-postenergy-warn-r1-core-comparison/v1",
        "comparison_boundary": "SEALED_REFERENCE_DIFFERENT_SOURCE_NOT_EXACT_SAME_HEAD_CAUSAL_ISOLATION",
        "common_original_W0": common_w0,
        "sealed_references": reference_panels,
        "piru_postenergy_warn_r1": {
            "label": labels["piru"],
            "immediate_post": piru_immediate,
            "final_W10": piru_final,
            "source_scope": "AUTHORITATIVE_CURRENT_RUN",
        },
        "accepted_z_comparison": accepted_comparison,
        "piru_minus_j0_final_W10": delta_vs_j0,
        "scientific_summary": {
            "status": "TECHNICALLY_VALID_NON_PARETO_PACKAGE_RESULT",
            "isolated_postenergy_causal_claim": False,
            "scientific_promotion": False,
        },
    }
    core["identity_sha256"] = identity(core)
    dump(REPORT_DIR / CORE_NAME, core)

    retention_by_key = {(row["round"], row["request_index"]): row for row in retention_rows}
    zw_request_rows: list[dict[str, Any]] = []
    for batch in z_rows:
        round_index = int(batch["round"])
        for request_index in range(100):
            retention = retention_by_key[(round_index, request_index)]
            rw_z = batch["rewrite"]["per_request_z_target_new_nll"][request_index]
            rw_w = batch["rewrite"]["per_request_physical_w_target_new_nll"][request_index]
            rp_z = batch["rephrase"]["per_request_z_target_new_nll"][request_index]
            rp_w = batch["rephrase"]["per_request_physical_w_target_new_nll"][request_index]
            row = {
                "round": round_index,
                "request_index": request_index,
                "case_id": retention["case_id"],
                "request_sha256": retention["request_sha256"],
                "history_width_at_entry": retention["history_width_at_entry"],
                "rewrite_z_target_new_nll": rw_z,
                "rewrite_W_immediate_target_new_nll": rw_w,
                "rewrite_writer_realization_gap_W_minus_z": [w - z for w, z in zip(rw_w, rw_z)],
                "rephrase_z_target_new_nll": rp_z,
                "rephrase_W_immediate_target_new_nll": rp_w,
                "rephrase_writer_realization_gap_W_minus_z": [w - z for w, z in zip(rp_w, rp_z)],
                "z_rewrite_success": batch["direct_z_rewrite_success"]["per_request_correct"][request_index],
                "W_immediate_rewrite_success": batch["physical_w_rewrite_success"]["per_request_correct"][request_index],
                "z_rephrase_success_correct": batch["direct_z_rephrase_success"]["per_request_correct"][request_index],
                "z_rephrase_success_required": batch["direct_z_rephrase_success"]["per_request_required"][request_index],
                "W_immediate_rephrase_success_correct": batch["physical_w_rephrase_success"]["per_request_correct"][request_index],
                "W_immediate_rephrase_success_required": batch["physical_w_rephrase_success"]["per_request_required"][request_index],
                "z_rephrase_strict_success": batch["direct_z_rephrase_success"]["strict_all_prompt_bits"][request_index],
                "W_immediate_rephrase_strict_success": batch["physical_w_rephrase_success"]["strict_all_prompt_bits"][request_index],
                "z_rewrite_acc": batch["direct_z_rewrite_accuracy"]["per_request_correct"][request_index],
                "W_immediate_rewrite_acc": batch["physical_w_rewrite_accuracy"]["per_request_correct"][request_index],
                "z_rephrase_acc_correct": batch["direct_z_rephrase_accuracy"]["per_request_correct"][request_index],
                "W_immediate_rephrase_acc_correct": batch["physical_w_rephrase_accuracy"]["per_request_correct"][request_index],
                "W_final_W10": retention["final_W10"],
                "W_immediate_post": retention["immediate_post"],
                "W_final_minus_immediate": retention["deltas"],
            }
            row["identity_sha256"] = identity(row)
            zw_request_rows.append(row)
    zw_table = {
        "schema": "ode-edit-s05-p1r52-piru-postenergy-warn-r1-per-z-w-request/v1",
        "canonical_gap_term": "WRITER_REALIZATION_GAP_W_MINUS_Z",
        "row_count": len(zw_request_rows),
        "rows": zw_request_rows,
    }
    zw_table["identity_sha256"] = identity(zw_table)
    dump(REPORT_DIR / ZW_NAME, zw_table)

    layer_rows: list[dict[str, Any]] = []
    for transition in norm_rows:
        for layer in transition["layers"]:
            row = {
                "round": transition["round"],
                "outer_k": transition["outer_k"],
                "history_width": transition["history_width"],
                **layer,
                "primary_concentration_metric": transition["primary_concentration_metric"],
                "secondary_concentration_metric": transition["secondary_concentration_metric"],
            }
            row["identity_sha256"] = identity(row)
            layer_rows.append(row)
    layer_table = {
        "schema": "ode-edit-s05-p1r52-piru-postenergy-warn-r1-per-layer/v1",
        "primary_concentration_metric": "SQRT_REALIZED_BF16_STEP_ENERGY_SHARE",
        "row_count": len(layer_rows),
        "rows": layer_rows,
    }
    layer_table["identity_sha256"] = identity(layer_table)
    dump(REPORT_DIR / LAYER_NAME, layer_table)

    h_table = {
        "schema": "ode-edit-s05-p1r52-piru-postenergy-warn-r1-per-h-decision/v1",
        "postsolve_energy_warn_enabled": True,
        "decision_influence_count": h_payload["postsolve_energy_decision_influence_count"],
        "row_count": len(h_rows),
        "rows": h_rows,
    }
    h_table["identity_sha256"] = identity(h_table)
    dump(REPORT_DIR / H_NAME, h_table)

    norms_by_batch_layer: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in layer_rows:
        norms_by_batch_layer[(row["round"], row["layer"])].append(row["actual_update_norm_share"])
    h_by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in h_rows:
        h_by_batch[int(row["round"])].append(row)
    z_by_batch = {int(row["round"]): row for row in z_rows}
    cohort_by_batch = {int(row["round"]): row for row in cohort_rows}
    batch_rows: list[dict[str, Any]] = []
    for checkpoint in checkpoint_rows:
        round_index = int(checkpoint["round"])
        zrow = z_by_batch[round_index]
        hset = h_by_batch[round_index]
        row = {
            "round": round_index,
            "history_width_at_entry": checkpoint["history_width_at_entry"],
            "immediate_post": checkpoint["immediate_post_summary"],
            "final_W10": cohort_by_batch[round_index]["final_W10"],
            "final_minus_immediate": cohort_by_batch[round_index]["final_minus_post"],
            "direct_z": {
                "rewrite_success": zrow["direct_z_rewrite_success"],
                "rewrite_acc": zrow["direct_z_rewrite_accuracy"],
                "paraphrase_success": zrow["direct_z_rephrase_success"],
                "paraphrase_acc": zrow["direct_z_rephrase_accuracy"],
            },
            "writer_realization": {"rewrite": zrow["rewrite"], "rephrase": zrow["rephrase"]},
            "postsolve_energy_warning_count": sum(x["post_energy_status"] == "WARN_POSTSOLVE_ENERGY_RESIDUAL" for x in hset),
            "postsolve_energy_over_1e_12_count": sum(x["energy_violation"] > 1e-12 for x in hset),
            "postsolve_energy_max_residual": max(x["energy_violation"] for x in hset),
            "layer_norm_share": {str(layer): dist(norms_by_batch_layer[(round_index, layer)]) for layer in range(4, 9)},
        }
        row["identity_sha256"] = identity(row)
        batch_rows.append(row)
    batch_table = {
        "schema": "ode-edit-s05-p1r52-piru-postenergy-warn-r1-per-batch/v1",
        "batch_entry_metrics_status": "REMOVED_BY_USER_AMENDMENT",
        "row_count": len(batch_rows),
        "rows": batch_rows,
    }
    batch_table["identity_sha256"] = identity(batch_table)
    dump(REPORT_DIR / BATCH_NAME, batch_table)

    z_rephrase_fail = [r for r in zw_request_rows if not r["z_rephrase_strict_success"]]
    z_to_w_rephrase_fail = [r for r in zw_request_rows if r["z_rephrase_strict_success"] and not r["W_immediate_rephrase_strict_success"]]
    final_rephrase_fail = [r for r in zw_request_rows if not r["W_final_W10"]["paraphrase_success"]["strict_all_prompts_pass"]]
    new_final_fail = [r for r in zw_request_rows if r["W_immediate_rephrase_strict_success"] and not r["W_final_W10"]["paraphrase_success"]["strict_all_prompts_pass"]]
    hard = {
        "schema": "ode-edit-s05-p1r52-piru-postenergy-warn-r1-hard-cohorts/v1",
        "request_denominator": 1000,
        "z_rephrase_strict_failure_count": len(z_rephrase_fail),
        "z_success_to_W_immediate_rephrase_strict_failure_count": len(z_to_w_rephrase_fail),
        "W_final_rephrase_strict_failure_count": len(final_rephrase_fail),
        "W_immediate_success_to_final_failure_count": len(new_final_fail),
        "z_rephrase_strict_failure_request_ids": [{"round": r["round"], "request_index": r["request_index"], "case_id": r["case_id"]} for r in z_rephrase_fail],
        "z_success_to_W_failure_request_ids": [{"round": r["round"], "request_index": r["request_index"], "case_id": r["case_id"]} for r in z_to_w_rephrase_fail],
        "final_rephrase_strict_failure_request_ids": [{"round": r["round"], "request_index": r["request_index"], "case_id": r["case_id"]} for r in final_rephrase_fail],
        "interpretation_boundary": "ASSOCIATION_ONLY_NO_ISOLATED_CAUSAL_CLAIM",
    }
    hard["identity_sha256"] = identity(hard)
    dump(REPORT_DIR / HARD_NAME, hard)

    accepted_files = sorted(RESULT_ROOT.glob("raw/batches/b*/raw/ode/p1r52-pir-pir-u/accepted-k*.json"), key=lambda p: (p.parts[-5], int(p.stem.split("k")[-1])))
    accepted_receipts = [load(path) for path in accepted_files]
    target_counts = Counter()
    route_counts = Counter()
    progress_predicted = []
    progress_actual = []
    for row in accepted_receipts:
        target = row["target_update"]
        target_counts["PRIMARY"] += target["primary_accept_count"]
        target_counts["RESCUE"] += target["rescue_accept_count"]
        target_counts["CURRENT"] += target["current_hold_count"]
        target_counts["ACTIVE_GRADIENT"] += target["active_gradient_count"]
        target_counts["CLAMP_HIT"] += target["clamp_hit_count"]
        route_counts["FALLBACK"] += int(row["routing"]["fallback_to_neutral"])
        progress_predicted.append(row["progress"]["predicted"])
        progress_actual.append(row["progress"]["actual"])

    compute_totals = Counter()
    compute_wall = Counter()
    for round_index in range(1, 11):
        bterm = load(RESULT_ROOT / f"raw/batches/b{round_index:02d}/terminal.json")
        for key, value in bterm["atomic_or_native"]["compute"]["totals"].items():
            compute_totals[key] += value
        for key, value in bterm["atomic_or_native"]["compute"]["wall_seconds"].items():
            compute_wall[key] += value

    warning_rows = [row for row in h_rows if row["post_energy_status"] == "WARN_POSTSOLVE_ENERGY_RESIDUAL"]
    old_strict_energy_false = [row for row in h_rows if row["energy_violation"] > row["external_tolerances"]["postsolve_energy_residual"]]
    layer_stats = {
        str(layer): {
            "actual_update_norm_share": dist(row["actual_update_norm_share"] for row in layer_rows if row["layer"] == layer),
            "squared_energy_share": dist(row["squared_energy_share"] for row in layer_rows if row["layer"] == layer),
        }
        for layer in range(4, 9)
    }
    rewrite_gaps = flatten(row["rewrite_writer_realization_gap_W_minus_z"] for row in zw_request_rows)
    rephrase_gaps = flatten(row["rephrase_writer_realization_gap_W_minus_z"] for row in zw_request_rows)
    compute_integrity = {
        "schema": "ode-edit-s05-p1r52-piru-postenergy-warn-r1-compute-integrity/v1",
        "scheduler": {"job_id": 20885, "state": "COMPLETED", "exit_code": "0:0", "elapsed": "03:07:20"},
        "source": {
            "head": terminal["source_head"],
            "tree": "b2c599d5082f78a11980abd1f56adf011a1f2ce7",
            "parent": "e062694e5fc815eefe88ce1dff607dff95b7aae2",
            "initial_failed_job": 20868,
            "initial_failed_attempt_preserved": (FAILED_ROOT / "failure.json").exists(),
            "tech_r1_scope": "INLINE_SAME_RUN_ACCEPTED_Z_REFERENCE_ASSERTION_ONLY",
        },
        "integrity": {
            "batch_terminal_count": 10,
            "accepted_transition_count": len(accepted_receipts),
            "structural_h_decision_count": len(h_rows),
            "action_freeze_checkpoint_count": terminal["action_freeze_checkpoint_count"],
            "W0_pointer_restored_exact": terminal["terminal_W0_restore"]["pointer_restored_exact"],
            "W0_bytes_restored_exact": terminal["terminal_W0_restore"]["byte_restored_exact"],
            "retry_count": 0,
            "backtracking_count": 0,
            "failure_receipt_present": (RESULT_ROOT / "failure.json").exists(),
            "batch_entry_evaluator_count": terminal["batch_entry_pre_evaluator_count"],
            "accepted_z_added_backward": terminal["accepted_z_observation_added_backward_count"],
            "accepted_z_added_generation": terminal["accepted_z_observation_added_generation_call_count"],
            "accepted_z_action_influence": terminal["accepted_z_observation_action_influence_count"],
            "duplicate_W_evaluator_forward": terminal["duplicate_W_evaluator_model_forward_count"],
        },
        "postenergy_warn": {
            "warning_count": len(warning_rows),
            "warning_magnitude_total": sum(row["post_energy_warning_magnitude"] for row in warning_rows),
            "over_old_1e_12_count": len(old_strict_energy_false),
            "maximum_energy_violation": max(row["energy_violation"] for row in h_rows),
            "maximum_location": [{"round": row["round"], "step_index": row["step_index"]} for row in h_rows if row["energy_violation"] == max(x["energy_violation"] for x in h_rows)],
            "maximum_strength_residual": max(abs(row["strength_residual"]) for row in h_rows),
            "maximum_p_violation": max(row["p_violation"] for row in h_rows),
            "minimum_selected": min(row["selected_action_summary"]["minimum"] for row in h_rows),
            "hard_gate_failures": sum(row["certificate_hard_gate_status"] != "PASS" for row in h_rows),
            "optimizer_energy_constraint_changed": False,
            "postsolve_energy_acceptance_changed_to_warn": True,
        },
        "routing": {
            "target_selection_counts": dict(target_counts),
            "fallback_count": route_counts["FALLBACK"],
            "predicted_progress": dist(progress_predicted),
            "actual_progress": dist(progress_actual),
            "structural_h_status_counts": dict(Counter(row["status"] for row in h_rows)),
            "history_widths": terminal["history_widths"],
        },
        "writer_realization": {
            "rewrite_gap": dist(rewrite_gaps),
            "rewrite_W_worse_than_z_count": sum(value > 0 for value in rewrite_gaps),
            "rewrite_denominator": len(rewrite_gaps),
            "rephrase_gap": dist(rephrase_gaps),
            "rephrase_W_worse_than_z_count": sum(value > 0 for value in rephrase_gaps),
            "rephrase_denominator": len(rephrase_gaps),
        },
        "layer_concentration": layer_stats,
        "compute": {
            "edit_totals": dict(compute_totals),
            "edit_wall_seconds_by_phase": dict(compute_wall),
            "terminal_job_compute": terminal["job_compute"],
            "total_wall_seconds": terminal["total_wall_seconds"],
        },
        "result_manifest": result_manifest,
    }
    compute_integrity["identity_sha256"] = identity(compute_integrity)
    dump(REPORT_DIR / COMPUTE_NAME, compute_integrity)

    def panel_cells(panel: dict[str, Any]) -> list[str]:
        return [
            pct(panel["rewrite_success"]["prompt_numerator"], panel["rewrite_success"]["prompt_denominator"]),
            pct(panel["rewrite_acc"]["prompt_numerator"], panel["rewrite_acc"]["prompt_denominator"]),
            pct(panel["paraphrase_success"]["prompt_numerator"], panel["paraphrase_success"]["prompt_denominator"]),
            pct(panel["paraphrase_success"]["strict_request_numerator"], panel["paraphrase_success"]["strict_request_denominator"]),
            pct(panel["paraphrase_acc"]["prompt_numerator"], panel["paraphrase_acc"]["prompt_denominator"]),
            pct(panel["paraphrase_acc"]["strict_request_numerator"], panel["paraphrase_acc"]["strict_request_denominator"]),
            pct(panel["locality"]["numerator"], panel["locality"]["denominator"]),
        ]

    report: list[str] = []
    report += [
        "# P1R52 PIR-U Sequential 10×B100 Structural-H Post-Energy-Warn R1 — 최종 사실 보고서",
        "",
        "> Llama3-8B-Instruct, 동일 봉인 1,000-request stream, B100×10, K8. 이 실행은 Structural-H 최적화의 energy 제약은 유지하되 post-solve energy 재검산만 WARN으로 바꾼 명시적 과학 방법 수정입니다.",
        "",
        "## 완료 상태와 핵심 사실",
        "",
        "- job `20885`는 `COMPLETED/0:0`, 배치 `10/10`, 요청 `1,000/1,000`, 물리 전환 `80/80`, H 결정 `80/80`로 끝났습니다.",
        "- W0 pointer/parameter bytes가 모두 정확히 복원됐고, retry/backtracking/failure endpoint는 `0/0/0`입니다.",
        f"- post-energy WARN은 `{len(warning_rows)}/80`회였습니다. 기존 1e-12 hard 경계를 넘은 결정은 `{len(old_strict_energy_false)}/80`회이며 최대값은 `{max(row['energy_violation'] for row in h_rows):.16e}`(B4-K2)입니다.",
        f"- J0 Structural-H ON 대비 final W10에서 GEN은 `{delta_vs_j0['paraphrase_success']['prompt_numerator']:+d}/2000`, strict GEN은 `{delta_vs_j0['paraphrase_success']['strict_request_numerator']:+d}/1000`이지만, EFF는 `{delta_vs_j0['rewrite_success']['prompt_numerator']:+d}/1000`, LOC는 `{delta_vs_j0['locality']['numerator']:+d}/10000`입니다.",
        "- 따라서 이 봉인 패키지는 `TECHNICALLY_VALID_NON_PARETO_PACKAGE_RESULT`입니다. PIR-U가 rephrase 쪽 일부 지표를 높였지만 rewrite/locality 저하가 함께 있어 우월성 또는 승격을 주장하지 않습니다.",
        "",
        "## 공통 original W0와 final W10 — 5행 핵심 표",
        "",
        "|방법|EFF|Rewrite Acc|GEN|GEN strict|Rephrase Acc|Acc strict|LOC|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, panel in [
        (labels["w0"], common_w0),
        (labels["memit"], reference_panels["memit"]["final_W10"]),
        (labels["alphaedit"], reference_panels["alphaedit"]["final_W10"]),
        (labels["r52_h_on"], reference_panels["r52_h_on"]["final_W10"]),
        (labels["piru"], piru_final),
    ]:
        report.append("|" + "|".join([label, *panel_cells(panel)]) + "|")
    report += [
        "",
        "`EFF ≡ rewrite_success`, `GEN ≡ paraphrase_success(rephrase_success)`. Accuracy는 teacher-forced strict suffix-token 정확도이며 success와 별도입니다. method-specific batch-entry W_(b-1) aggregate는 사용자 지시에 따라 측정·보고하지 않습니다.",
        "",
        "## Immediate-post와 final W10 절대값",
        "",
        "|방법|패널|EFF|Rewrite Acc|GEN|GEN strict|Rephrase Acc|Acc strict|LOC|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("memit", "alphaedit", "r52_h_on"):
        for panel_name, panel in (("10× immediate-post", reference_panels[arm]["immediate_post"]), ("final W10/B1000", reference_panels[arm]["final_W10"])):
            report.append("|" + "|".join([labels[arm], panel_name, *panel_cells(panel)]) + "|")
    for panel_name, panel in (("10× immediate-post", piru_immediate), ("final W10/B1000", piru_final)):
        report.append("|" + "|".join([labels["piru"], panel_name, *panel_cells(panel)]) + "|")

    report += [
        "",
        "## Final W10 target-new/true NLL와 margin",
        "",
        "|방법|Rewrite new / true / margin|Rephrase new / true / margin|",
        "|---|---:|---:|",
    ]
    for arm, panel in (("memit", reference_panels["memit"]["final_W10"]), ("alphaedit", reference_panels["alphaedit"]["final_W10"]), ("r52_h_on", reference_panels["r52_h_on"]["final_W10"]), ("piru", piru_final)):
        rw = panel["rewrite_success"]
        rp = panel["paraphrase_success"]
        report.append(f"|{labels[arm]}|{f6(rw['target_new_nll']['mean'])} / {f6(rw['target_true_nll']['mean'])} / {f6(rw['target_true_minus_new_margin']['mean'])}|{f6(rp['target_new_nll']['mean'])} / {f6(rp['target_true_nll']['mean'])} / {f6(rp['target_true_minus_new_margin']['mean'])}|")

    report += [
        "",
        "## Accepted-z 직접 주입과 물리 W writer realization",
        "",
        "|방법|z rewrite NLL|z rephrase NLL|W immediate rewrite NLL|W immediate rephrase NLL|W−z rewrite gap|W−z rephrase gap|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("memit", "alphaedit", "r52_h_on"):
        row = accepted_ref[arm]
        report.append(f"|{labels[arm]}|{row['accepted_z']['rewrite_success']['target_new_nll']['mean']:.6f}|{row['accepted_z']['paraphrase_success']['target_new_nll']['mean']:.6f}|{row['W_immediate_post']['rewrite_success']['target_new_nll']['mean']:.6f}|{row['W_immediate_post']['paraphrase_success']['target_new_nll']['mean']:.6f}|{row['Wpost_minus_z_rewrite_new_nll']['mean']:+.6f}|{row['Wpost_minus_z_rephrase_new_nll']['mean']:+.6f}|")
    report.append(f"|{labels['piru']}|{piru_accepted['rewrite_success']['target_new_nll']['mean']:.6f}|{piru_accepted['paraphrase_success']['target_new_nll']['mean']:.6f}|{piru_immediate['rewrite_success']['target_new_nll']['mean']:.6f}|{piru_immediate['paraphrase_success']['target_new_nll']['mean']:.6f}|{compute_integrity['writer_realization']['rewrite_gap']['mean']:+.6f}|{compute_integrity['writer_realization']['rephrase_gap']['mean']:+.6f}|")
    report += [
        "",
        f"PIR-U writer realization gap(W−z)은 rewrite 평균/중앙값/p90/max `{compute_integrity['writer_realization']['rewrite_gap']['mean']:.6f}/{compute_integrity['writer_realization']['rewrite_gap']['median']:.6f}/{compute_integrity['writer_realization']['rewrite_gap']['p90']:.6f}/{compute_integrity['writer_realization']['rewrite_gap']['raw_max']:.6f}`, rephrase `{compute_integrity['writer_realization']['rephrase_gap']['mean']:.6f}/{compute_integrity['writer_realization']['rephrase_gap']['median']:.6f}/{compute_integrity['writer_realization']['rephrase_gap']['p90']:.6f}/{compute_integrity['writer_realization']['rephrase_gap']['raw_max']:.6f}`입니다.",
        f"물리 W가 z보다 NLL이 높은 prompt는 rewrite `{compute_integrity['writer_realization']['rewrite_W_worse_than_z_count']}/{compute_integrity['writer_realization']['rewrite_denominator']}`, rephrase `{compute_integrity['writer_realization']['rephrase_W_worse_than_z_count']}/{compute_integrity['writer_realization']['rephrase_denominator']}`입니다. gap이 음수면 W가 더 낮은 NLL이므로 손실로 부르지 않습니다.",
        "",
        "## PIR-U B1–B10 절대 지표와 z/W gap",
        "",
        "|B|history|W post EFF|W post GEN|GEN strict|LOC|z rewrite NLL|z rephrase NLL|W−z rewrite|W−z rephrase|energy WARN|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in batch_rows:
        post = row["immediate_post"]
        zrw = row["direct_z"]["rewrite_success"]
        zrp = row["direct_z"]["paraphrase_success"]
        report.append(
            f"|B{row['round']}|{row['history_width_at_entry']}|"
            f"{pct(post['metrics']['rewrite_success']['prompt_numerator'], post['metrics']['rewrite_success']['prompt_denominator'])}|"
            f"{pct(post['metrics']['paraphrase_success']['prompt_numerator'], post['metrics']['paraphrase_success']['prompt_denominator'])}|"
            f"{pct(post['metrics']['paraphrase_success']['strict_request_numerator'], post['metrics']['paraphrase_success']['strict_request_denominator'])}|"
            f"{pct(post['locality']['numerator'], post['locality']['denominator'])}|"
            f"{statistics.mean(flatten(zrw['target_new_nll_by_request'])):.6f}|{statistics.mean(flatten(zrp['target_new_nll_by_request'])):.6f}|"
            f"{row['writer_realization']['rewrite']['writer_realization_gap_w_minus_z']['mean']:+.6f}|{row['writer_realization']['rephrase']['writer_realization_gap_w_minus_z']['mean']:+.6f}|"
            f"{row['postsolve_energy_warning_count']}/8|"
        )

    report += [
        "",
        "## Structural-H hard gate와 Post-Energy WARN",
        "",
        f"- H 상태: `{dict(Counter(row['status'] for row in h_rows))}`; hard gate 실패 `0/80`.",
        f"- strength residual 최대 `{max(abs(row['strength_residual']) for row in h_rows):.3e}` (limit 1e-8), P residual 최대 `{max(row['p_violation'] for row in h_rows):.3e}` (limit 1e-8), min(selected) 최소 `{min(row['selected_action_summary']['minimum'] for row in h_rows):.3e}` (limit −1e-8).",
        f"- energy residual은 WARN `{len(warning_rows)}/80`; 총 magnitude `{sum(row['post_energy_warning_magnitude'] for row in warning_rows):.3e}`. 기존 1e-12 경계 초과는 B4-K2의 `{max(row['energy_violation'] for row in h_rows):.16e}` 한 건입니다.",
        "- optimizer 내부 `E(v) <= E_limit`은 그대로이며, solver/strength/P/nonnegative 실패는 끝까지 hard fail입니다. 새 임계값, polish, shrink, retry, tolerance 변경은 없습니다.",
        "- 이 실행은 old strict H certificate의 기술 수리가 아니라, post-solve energy 재검산을 WARN으로 바꾼 명시적 방법 수정입니다.",
        "",
        "## 실제 post-BF16 update NormShare",
        "",
        "주지표는 `sqrt(realized_bf16_step_energy) / Σ sqrt(E_l)`입니다. factor energy, beta, coefficient는 보조 설명값입니다.",
        "",
        "|layer|NormShare mean|median|p90|max|squared EnergyShare mean|",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in range(4, 9):
        s = layer_stats[str(layer)]["actual_update_norm_share"]
        e = layer_stats[str(layer)]["squared_energy_share"]
        report.append(f"|L{layer}|{s['mean']:.6f}|{s['median']:.6f}|{s['p90']:.6f}|{s['raw_max']:.6f}|{e['mean']:.6f}|")
    report += [
        "",
        f"L8 NormShare는 평균 `{layer_stats['8']['actual_update_norm_share']['mean']:.4f}`, 중앙값 `{layer_stats['8']['actual_update_norm_share']['median']:.4f}`, p90 `{layer_stats['8']['actual_update_norm_share']['p90']:.4f}`, 최대 `{layer_stats['8']['actual_update_norm_share']['raw_max']:.4f}`입니다. 후기 layer 집중은 유지됐으며, 이는 실제 BF16 update norm 기준입니다.",
        "",
        "## Target/controller/writer와 순차 상태",
        "",
        f"- PRIMARY/RESCUE/CURRENT: `{target_counts['PRIMARY']}/{target_counts['RESCUE']}/{target_counts['CURRENT']}` over `{target_counts['ACTIVE_GRADIENT']}` active request-steps; clamp-hit `{target_counts['CLAMP_HIT']}`, Soft→Neutral fallback `{route_counts['FALLBACK']}`.",
        f"- history widths `{terminal['history_widths']}`; terminal active history/lifetime anchors `{terminal['terminal_active_history_count']}/{terminal['terminal_lifetime_anchor_count']}`.",
        "- batch-entry evaluator `0`, inner heldout controller access `0`, accepted-z observation action influence/backward/generation `0/0/0`, duplicate W evaluator forward `0`.",
        "- K당 virtual prefix는 q-only current residual/key/q refresh를 사용하며 current-slope backward `0`; outer당 물리 materialization `1`; physical W는 B1→B10 지속 후 terminal W0로 정확 복원됐습니다.",
        "",
        "## Compute와 무결성",
        "",
        f"- scheduler wall `{terminal['total_wall_seconds']:.1f}s`; accepted-z observation F/B/generation `{terminal['accepted_z_observation_model_forward_count']}/{terminal['accepted_z_observation_added_backward_count']}/{terminal['accepted_z_observation_added_generation_call_count']}`, tokens `{terminal['accepted_z_observation_processed_token_count']}`.",
        f"- edit ledger totals: model F `{compute_totals['model_forward_calls']}`, backward `{compute_totals['backward_calls']}`, target backward `{compute_totals['target_backward_calls']}`, slope backward `{compute_totals['slope_backward_calls']}`, prefix captures `{compute_totals['capture_forward_calls']}`, materializations `{compute_totals['materialization_count']}`.",
        f"- terminal evaluator F/tokens `{terminal['job_compute']['counters']['evaluator_forward']}/{terminal['job_compute']['counters']['evaluator_tokens']}`; W0 pointer/bytes restore `True/True`.",
        f"- 실행 소스 HEAD/tree `{terminal['source_head']}` / `b2c599d5082f78a11980abd1f56adf011a1f2ce7`. 첫 시도 job20868은 inline accepted-z sealed-reference assertion의 순수 기술 실패로 보존됐고, TECH-R1은 그 assertion만 same-run path에 맞게 수정했습니다.",
        "",
        "## Hard cohort와 해석 경계",
        "",
        f"- z rephrase strict 실패 `{hard['z_rephrase_strict_failure_count']}/1000`; z strict 성공→W immediate strict 실패 `{hard['z_success_to_W_immediate_rephrase_strict_failure_count']}/1000`; final W10 strict 실패 `{hard['W_final_rephrase_strict_failure_count']}/1000`.",
        f"- immediate-post strict 성공→final 실패 `{hard['W_immediate_success_to_final_failure_count']}/1000`. 요청별 ID와 수치는 `{HARD_NAME}`에 있습니다.",
        "- sealed MEMIT/AlphaEdit/J0는 동일 1,000-request/order이지만 다른 source revision에서 생성됐습니다. 따라서 비교는 matched sealed reference이며 exact same-head 단일변수 인과 격리가 아닙니다.",
        "- post-energy WARN 수정과 결과 변화의 고립된 인과를 주장하지 않습니다. `scientific_promotion=false`입니다.",
        "",
        "## Machine-readable artifacts",
        "",
        f"- core comparison: `{REPORT_DIR / CORE_NAME}`",
        f"- per batch: `{REPORT_DIR / BATCH_NAME}` (10 rows)",
        f"- per H decision: `{REPORT_DIR / H_NAME}` (80 rows)",
        f"- per layer: `{REPORT_DIR / LAYER_NAME}` (400 rows)",
        f"- per z/W request: `{REPORT_DIR / ZW_NAME}` (1,000 rows)",
        f"- hard cohorts: `{REPORT_DIR / HARD_NAME}`",
        f"- compute/integrity: `{REPORT_DIR / COMPUTE_NAME}`",
        "",
        "scientific_promotion=false",
    ]
    (REPORT_DIR / REPORT_NAME).write_text("\n".join(report) + "\n", encoding="utf-8")

    artifact_names = [REPORT_NAME, CORE_NAME, BATCH_NAME, H_NAME, LAYER_NAME, ZW_NAME, HARD_NAME, COMPUTE_NAME]
    members = []
    for name in artifact_names:
        path = REPORT_DIR / name
        payload = load(path) if path.suffix == ".json" else None
        members.append({
            "name": name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "lines": len(path.read_text(encoding="utf-8").splitlines()),
            "rows": payload.get("row_count") if isinstance(payload, dict) else None,
        })
    member_root = sha256_bytes("\n".join(f"{m['name']}|{m['sha256']}|{m['bytes']}|{m['lines']}|{m['rows']}" for m in members).encode())
    manifest = {
        "schema": "ode-edit-s05-p1r52-piru-postenergy-warn-r1-analysis-manifest/v1",
        "instruction_id": terminal["instruction_id"],
        "source_head": terminal["source_head"],
        "source_tree": "b2c599d5082f78a11980abd1f56adf011a1f2ce7",
        "result_terminal_sha256": sha256_file(RESULT_ROOT / "terminal.json"),
        "result_manifest_sha256": sha256_file(RESULT_ROOT / "manifest.json"),
        "members": members,
        "member_root_sha256": member_root,
    }
    manifest["identity_sha256"] = identity(manifest)
    dump(REPORT_DIR / MANIFEST_NAME, manifest)
    receipt = {
        "schema": "ode-edit-s05-p1r52-piru-postenergy-warn-r1-rooted-analysis-receipt/v1",
        "status": "SEALED_TERMINAL_RAWFREE_PACKAGE",
        "scheduler": {"job_id": 20885, "state": "COMPLETED", "exit_code": "0:0", "elapsed": "03:07:20"},
        "report_path": str(REPORT_DIR / REPORT_NAME),
        "report_sha256": sha256_file(REPORT_DIR / REPORT_NAME),
        "analysis_manifest_path": str(REPORT_DIR / MANIFEST_NAME),
        "analysis_manifest_sha256": sha256_file(REPORT_DIR / MANIFEST_NAME),
        "member_root_sha256": member_root,
        "terminal_integrity": compute_integrity["integrity"],
        "scientific_status": "TECHNICALLY_VALID_NON_PARETO_PACKAGE_RESULT",
        "scientific_promotion": False,
    }
    receipt["root_digest"] = identity(receipt)
    dump(REPORT_DIR / RECEIPT_NAME, receipt)

    print(json.dumps({
        "status": receipt["status"],
        "report": str(REPORT_DIR / REPORT_NAME),
        "report_sha256": receipt["report_sha256"],
        "manifest": str(REPORT_DIR / MANIFEST_NAME),
        "manifest_sha256": receipt["analysis_manifest_sha256"],
        "receipt": str(REPORT_DIR / RECEIPT_NAME),
        "receipt_root": receipt["root_digest"],
        "member_root": member_root,
        "rows": {"batch": len(batch_rows), "h": len(h_rows), "layer": len(layer_rows), "z_w_request": len(zw_request_rows)},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
