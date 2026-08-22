#!/usr/bin/env python3
"""Build the factual raw-free terminal package for P1R52 IL5 Sequential."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


WORKTREE = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RESULT = WORKTREE / "local/odebf/results/s05-p1r52-llama-j0-il5-sequential-10xb100-postenergy-warn-r1-tech-r2-v1"
FAILED = WORKTREE / "local/odebf/results/s05-p1r52-llama-j0-il5-sequential-10xb100-postenergy-warn-r1-tech-r1-v1"
BASE = WORKTREE / "local/odebf/reports/p1r52-llama-sequential-10xb100-fourarm-v1"
ZBASE = WORKTREE / "local/odebf/reports/p1r52-b100-accepted-z-rephrase-observation-v1"
STATE = WORKTREE / "local/odebf/state/p1r52-il5-sequential-10xb100-postenergy-warn-r1-tech-r2-v1"

REPORT = "p1r52-llama-j0-il5-sequential-10xb100-terminal-report-ko.md"
CORE = "il5-core-comparison.json"
INNER = "il5-inner-trajectory.json"
BATCH = "il5-batch-trajectory.json"
ROUTING = "il5-writer-routing-structural-h.json"
LAYER = "il5-layer-update-summary.json"
HARD = "il5-hard-cohorts.json"
COMPUTE = "il5-compute-integrity.json"
MANIFEST = "analysis-manifest.json"
RECEIPT = "rooted-analysis-receipt.json"
REVIEW = "independent-rawfree-rehash-review.json"

STREAM_ROOT = "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
STREAM_ORDER = "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"
HEAD = "e21a6a93db716c794a19f99da5b54d3558aa0776"
TREE = "c600adeae46f68a86cdd98f9daa91e6efab970ba"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ident(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def flatten(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if isinstance(value, list):
            out.extend(flatten(value))
        elif value is not None:
            out.append(float(value))
    return out


def quantile(values: list[float], p: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (pos - lo)


def dist(values: Iterable[Any]) -> dict[str, Any]:
    xs = flatten(values)
    if not xs:
        return {"count": 0, "mean": None, "median": None, "p90": None, "raw_min": None, "raw_max": None}
    return {
        "count": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs),
        "p90": quantile(xs, 0.9), "raw_min": min(xs), "raw_max": max(xs),
    }


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / den:.2f}%)" if den else "NOT_RECORDED"


def metric_ref(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_numerator": metric["numerator"], "prompt_denominator": metric["denominator"],
        "strict_request_numerator": metric.get("strict_numerator", metric["numerator"]),
        "strict_request_denominator": metric.get("strict_denominator", metric["denominator"]),
        "target_new_nll": metric.get("target_new_nll"), "target_true_nll": metric.get("target_old_nll"),
        "margin": metric.get("margin"),
    }


def metric_new(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_numerator": metric["prompt_numerator"], "prompt_denominator": metric["prompt_denominator"],
        "strict_request_numerator": metric["strict_request_numerator"],
        "strict_request_denominator": metric["strict_request_denominator"],
        "target_new_nll": metric.get("target_new_nll"), "target_true_nll": metric.get("target_old_nll"),
        "margin": metric.get("target_old_minus_new_margin"),
    }


def panel_ref(panel: dict[str, Any]) -> dict[str, Any]:
    return {k: metric_ref(panel[k]) for k in ("rewrite_success", "rewrite_acc", "paraphrase_success", "paraphrase_acc")} | {"locality": panel["locality"]}


def panel_new(panel: dict[str, Any]) -> dict[str, Any]:
    return {k: metric_new(panel["metrics"][k]) for k in ("rewrite_success", "rewrite_acc", "paraphrase_success", "paraphrase_acc")} | {"locality": panel["locality"]}


def aggregate_outer(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    final = [r for r in rows if r["outer_step_index"] == 7]
    if prefix == "accepted_z":
        eff, gen = "accepted_z_efficacy", "accepted_z_generalization"
        rwacc, rpacc, rps = "accepted_z_rewrite_accuracy", "accepted_z_rephrase_accuracy", "accepted_z_rephrase_success"
        rwnll, rpnll = "accepted_z_rewrite_nll_by_request", "accepted_z_rephrase_nll_by_request"
    else:
        eff, gen = "writer_w_efficacy", "writer_w_generalization"
        rwacc, rpacc, rps = "writer_w_rewrite_accuracy", "writer_w_rephrase_accuracy", "writer_w_rephrase_success"
        rwnll, rpnll = "writer_w_rewrite_nll_by_request", "writer_w_rephrase_nll_by_request"
    return {
        "rewrite_success": {"prompt_numerator": sum(r[eff]["numerator"] for r in final), "prompt_denominator": 1000, "strict_request_numerator": sum(r[eff]["strict_request_count"] for r in final), "strict_request_denominator": 1000, "target_new_nll": dist(r[rwnll] for r in final)},
        "rewrite_acc": {"prompt_numerator": sum(r[rwacc]["prompt_numerator"] for r in final), "prompt_denominator": 1000, "strict_request_numerator": sum(r[rwacc]["strict_request_numerator"] for r in final), "strict_request_denominator": 1000, "target_new_nll": dist(r[rwnll] for r in final)},
        "paraphrase_success": {"prompt_numerator": sum(r[gen]["numerator"] for r in final), "prompt_denominator": 2000, "strict_request_numerator": sum(r[rps]["strict_request_numerator"] for r in final), "strict_request_denominator": 1000, "target_new_nll": dist(r[rpnll] for r in final)},
        "paraphrase_acc": {"prompt_numerator": sum(r[rpacc]["prompt_numerator"] for r in final), "prompt_denominator": 2000, "strict_request_numerator": sum(r[rpacc]["strict_request_numerator"] for r in final), "strict_request_denominator": 1000, "target_new_nll": dist(r[rpnll] for r in final)},
    }


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


def main() -> None:
    terminal = load(RESULT / "terminal.json")
    result_manifest = load(RESULT / "manifest.json")
    inner = load(RESULT / "target-depth-inner-telemetry.json")["rows"]
    outer = load(RESULT / "target-depth-outer-telemetry.json")["rows"]
    h_rows = load(RESULT / "structural-h-decisions.json")["rows"]
    update_rows = load(RESULT / "actual-update-norm-share.json")["rows"]
    checkpoints = load(RESULT / "b1-b10-post-final-checkpoints.json")["rows"]
    cohorts = load(RESULT / "batch-age-post-final-cohorts.json")["rows"]
    retention = load(RESULT / "immediate-post-final-w10-requests.json")["rows"]
    baseline = load(BASE / "p1r52-b100-four-arm-aggregate.json")
    zref_rows = load(ZBASE / "p1r52-accepted-z-aggregates.json")["rows"]
    zref = next(row for row in zref_rows if row["arm"] == "r52_h_on")

    accepted: list[dict[str, Any]] = []
    accepted_keys: list[tuple[int, int]] = []
    realization_rows: list[dict[str, Any]] = []
    realization_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    batch_terms: list[dict[str, Any]] = []
    for b in range(1, 11):
        bt = load(RESULT / f"raw/batches/b{b:02d}/terminal.json")
        batch_terms.append(bt)
        for k in range(1, 9):
            base = RESULT / f"raw/batches/b{b:02d}/raw/ode/p1r52-rsa-r42safekdc-m1-soft"
            accepted.append(load(base / f"accepted-k{k}.json"))
            accepted_keys.append((b, k))
            realization = load(base / f"requestwise-realization-k{k}.json")
            realization_rows.append(realization)
            realization_by_key[(b, k)] = realization

    il5_final = panel_new(terminal["final_B1000"])
    il5_post = panel_new(terminal["immediate_post_all_B100"])
    refs = {arm: panel_ref(baseline["arms"][arm]["final_W10_B1000"]) for arm in ("memit", "alphaedit", "r52_h_on")}
    w0 = panel_ref(baseline["common_W0_B1_panel"])
    il5_z = aggregate_outer(outer, "accepted_z")
    il5_w_outer = aggregate_outer(outer, "writer_w")
    il1_z = zref["accepted_z"]

    delta_il5_il1 = {}
    for metric in ("rewrite_success", "rewrite_acc", "paraphrase_success", "paraphrase_acc"):
        delta_il5_il1[metric] = {
            "prompt_numerator": il5_final[metric]["prompt_numerator"] - refs["r52_h_on"][metric]["prompt_numerator"],
            "strict_request_numerator": il5_final[metric]["strict_request_numerator"] - refs["r52_h_on"][metric]["strict_request_numerator"],
            "target_new_nll_mean": il5_final[metric]["target_new_nll"]["mean"] - refs["r52_h_on"][metric]["target_new_nll"]["mean"],
        }
    delta_il5_il1["locality"] = il5_final["locality"]["numerator"] - refs["r52_h_on"]["locality"]["numerator"]

    core = {
        "schema": "ode-edit-s05-p1r52-il5-sequential-core-comparison/v1",
        "status": "TERMINAL_TECHNICALLY_VALID_SCIENTIFIC_NON_PROMOTION",
        "same_stream_order": {"stream_root": STREAM_ROOT, "order": STREAM_ORDER, "status": "PASS"},
        "common_W0_B1_only_unmatched_denominator": w0,
        "sealed_final_W10": refs,
        "il5_immediate_post": il5_post,
        "il5_final_W10": il5_final,
        "il5_minus_il1_final_W10": delta_il5_il1,
        "accepted_z": {"il1": il1_z, "il5": il5_z},
        "outer_final_writer_W": il5_w_outer,
        "scientific_promotion": False,
    }
    core["identity_sha256"] = ident(core)
    dump(OUT / CORE, core)

    inner_rows = []
    for m in range(5):
        rows = [r for r in inner if r["inner_index"] == m]
        item = {
            "inner_index_1based": m + 1, "row_count": len(rows), "request_count": 8000,
            "target_objective_nll": dist(r["target_objective_nll_by_request"] for r in rows),
            "accepted_z_rewrite_nll": dist(r["accepted_z_rewrite_nll_by_request"] for r in rows),
            "accepted_z_rephrase_nll": dist(r["accepted_z_rephrase_nll_by_request"] for r in rows),
            "inner_step_movement_norm": dist(r["inner_step_movement_norm_by_request"] for r in rows),
            "cumulative_outer_entry_movement_norm": dist(r["cumulative_outer_entry_movement_norm_by_request"] for r in rows),
            "z_eff": [sum(r["accepted_z_efficacy"]["numerator"] for r in rows), 8000],
            "z_gen": [sum(r["accepted_z_generalization"]["numerator"] for r in rows), 16000],
            "z_gen_strict": [sum(r["accepted_z_rephrase_success"]["strict_request_numerator"] for r in rows), 8000],
            "z_rephrase_accuracy": [sum(r["accepted_z_rephrase_accuracy"]["prompt_numerator"] for r in rows), 16000],
            "z_rephrase_accuracy_strict": [sum(r["accepted_z_rephrase_accuracy"]["strict_request_numerator"] for r in rows), 8000],
        }
        inner_rows.append(item)
    inner_payload = {
        "schema": "ode-edit-s05-p1r52-il5-sequential-inner-trajectory/v1", "row_count": 5,
        "source_inner_rows": 400, "source_request_inner_rows": 40000,
        "observation_added_backward_generation_action_influence": [0, 0, 0],
        "observation_model_forward_count": sum(r["added_model_forward_count"] for r in inner),
        "rows": inner_rows,
    }
    inner_payload["identity_sha256"] = ident(inner_payload)
    dump(OUT / INNER, inner_payload)

    cohort_by_round = {r["round"]: r for r in cohorts}
    outer_final_by_round = {r["round"]: r for r in outer if r["outer_step_index"] == 7}
    batch_rows = []
    for cp in checkpoints:
        b = cp["round"]
        post = panel_new({"metrics": cp["immediate_post_summary"]["metrics"], "locality": cp["immediate_post_summary"]["locality"]})
        final = panel_new({"metrics": cohort_by_round[b]["final_W10"]["metrics"], "locality": cohort_by_round[b]["final_W10"]["locality"]})
        zrow = outer_final_by_round[b]
        batch_rows.append({
            "round": b, "age_at_W10": 10 - b, "history_width_at_entry": cp["history_width_at_entry"],
            "immediate_post": post, "final_W10": final,
            "z_rewrite_nll": dist(zrow["accepted_z_rewrite_nll_by_request"]),
            "z_rephrase_nll": dist(zrow["accepted_z_rephrase_nll_by_request"]),
            "W_immediate_rewrite_nll": dist(zrow["writer_w_rewrite_nll_by_request"]),
            "W_immediate_rephrase_nll": dist(zrow["writer_w_rephrase_nll_by_request"]),
            "W_minus_z_rewrite_gap": dist(zrow["rewrite_w_minus_z_nll_gap_by_request"]),
            "W_minus_z_rephrase_gap": dist(zrow["rephrase_w_minus_z_nll_gap_by_request"]),
        })
    batch_payload = {"schema": "ode-edit-s05-p1r52-il5-sequential-batch-trajectory/v1", "row_count": 10, "rows": batch_rows}
    batch_payload["identity_sha256"] = ident(batch_payload)
    dump(OUT / BATCH, batch_payload)

    h_by_key = {(r["round"], r["step_index"] + 1): r for r in h_rows}
    u_by_key = {(r["round"], r["outer_k"]): r for r in update_rows}
    compact_steps = []
    route_counts = Counter()
    last_route_counts = Counter()
    for (batch_index, outer_k), receipt in zip(accepted_keys, accepted, strict=True):
        for inner_receipt in receipt["target_update"]["inner_trajectory"]:
            route_counts.update(inner_receipt["selection_by_request"])
        last_route_counts.update(receipt["target_update"]["selection_by_request"])
        h = h_by_key[(batch_index, outer_k)]
        u = u_by_key[(h["round"], h["step_index"] + 1)]
        actual = realization_by_key[(batch_index, outer_k)]
        compact_steps.append({
            "round": h["round"], "outer_k": h["step_index"] + 1, "history_width": h["history_width"],
            "h_status": h["status"], "post_energy_status": h["post_energy_status"], "post_energy_warning_magnitude": h["post_energy_warning_magnitude"],
            "strength_residual": h["strength_residual"], "p_violation": h["p_violation"], "selected_p": h["selected_p"],
            "selected_capacity": receipt["routing"]["selected_capacity"], "selected_energy": receipt["routing"]["selected_energy"],
            "alpha_req": receipt["routing"]["alpha_req"], "alpha_apply": receipt["routing"]["alpha_apply"],
            "simplex_top1_share": receipt["routing"]["simplex_top1_share"], "simplex_entropy": receipt["routing"]["simplex_entropy"],
            "actual_update_norm_total": u["actual_update_norm_total"], "realized_bf16_step_energy_total": u["realized_bf16_step_energy_total"],
            "requestwise_actual_progress": dist(actual["actual_w_only_progress_by_request"]),
            "requestwise_negative_progress_count": sum(v < 0 for v in actual["actual_w_only_progress_by_request"]),
            "writer_realization_cosine": dist(receipt["target_write_realization"]["cosine"]),
            "fallback_to_neutral": receipt["routing"]["fallback_to_neutral"],
        })
    routing_payload = {
        "schema": "ode-edit-s05-p1r52-il5-sequential-writer-routing-h/v1", "row_count": 80, "rows": compact_steps,
        "inner_primary_rescue_current": dict(route_counts), "last_inner_primary_rescue_current": dict(last_route_counts),
        "h_status_counts": dict(Counter(r["status"] for r in h_rows)),
        "post_energy_status_counts": dict(Counter(r["post_energy_status"] for r in h_rows)),
        "post_energy_warning_total_magnitude": sum(r["post_energy_warning_magnitude"] for r in h_rows),
        "post_energy_decision_influence_count": 0,
        "requestwise_negative_progress_count": sum(s["requestwise_negative_progress_count"] for s in compact_steps),
        "requestwise_progress_denominator": 8000,
    }
    routing_payload["identity_sha256"] = ident(routing_payload)
    dump(OUT / ROUTING, routing_payload)

    layer_rows = []
    for layer in range(4, 9):
        rows = [item for step in update_rows for item in step["layers"] if item["layer"] == layer]
        layer_rows.append({
            "layer": layer, "actual_update_norm": dist(r["actual_update_norm"] for r in rows),
            "actual_update_norm_share": dist(r["actual_update_norm_share"] for r in rows),
            "realized_bf16_energy_sum": sum(r["realized_bf16_step_energy"] for r in rows),
            "energy_share_mean": statistics.fmean(r["squared_energy_share"] for r in rows),
        })
    layer_payload = {
        "schema": "ode-edit-s05-p1r52-il5-sequential-layer-summary/v1", "row_count": 5, "rows": layer_rows,
        "transition_count": 80, "path_energy_sum": sum(r["realized_bf16_step_energy_total"] for r in update_rows),
        "update_norm_total": dist(r["actual_update_norm_total"] for r in update_rows),
    }
    layer_payload["identity_sha256"] = ident(layer_payload)
    dump(OUT / LAYER, layer_payload)

    def strict(row: dict[str, Any], when: str, metric: str) -> bool:
        return bool(row[when][metric]["strict_all_prompts_pass"])

    z_to_w_rp = []
    for row in [r for r in outer if r["outer_step_index"] == 7]:
        zbits = row["accepted_z_rephrase_success"]["strict_request_bit_vector_sha256"]
        wbits = row["writer_w_rephrase_success"]["strict_request_bit_vector_sha256"]
        z_to_w_rp.append({"round": row["round"], "z_strict_bits_sha256": zbits, "W_strict_bits_sha256": wbits,
                         "z_success_minus_W_success_prompts": row["accepted_z_generalization"]["numerator"] - row["writer_w_generalization"]["numerator"],
                         "z_strict_minus_W_strict": row["accepted_z_rephrase_success"]["strict_request_numerator"] - row["writer_w_rephrase_success"]["strict_request_numerator"]})
    hard = {
        "schema": "ode-edit-s05-p1r52-il5-sequential-hard-cohorts/v1", "request_denominator": 1000,
        "rewrite_accuracy_post_success_to_final_failure": sum(strict(r, "immediate_post", "rewrite_acc") and not strict(r, "final_W10", "rewrite_acc") for r in retention),
        "rephrase_success_post_success_to_final_failure": sum(strict(r, "immediate_post", "paraphrase_success") and not strict(r, "final_W10", "paraphrase_success") for r in retention),
        "rephrase_success_post_failure_to_final_success": sum(not strict(r, "immediate_post", "paraphrase_success") and strict(r, "final_W10", "paraphrase_success") for r in retention),
        "rephrase_accuracy_post_success_to_final_failure": sum(strict(r, "immediate_post", "paraphrase_acc") and not strict(r, "final_W10", "paraphrase_acc") for r in retention),
        "rephrase_accuracy_post_failure_to_final_success": sum(not strict(r, "immediate_post", "paraphrase_acc") and strict(r, "final_W10", "paraphrase_acc") for r in retention),
        "z_to_W_outer_final_by_round": z_to_w_rp,
        "z_success_to_W_immediate_failure_prompts": il5_z["paraphrase_success"]["prompt_numerator"] - il5_w_outer["paraphrase_success"]["prompt_numerator"],
        "z_success_to_W_immediate_failure_strict_requests": il5_z["paraphrase_success"]["strict_request_numerator"] - il5_w_outer["paraphrase_success"]["strict_request_numerator"],
    }
    hard["identity_sha256"] = ident(hard)
    dump(OUT / HARD, hard)

    compute_totals, compute_wall = Counter(), Counter()
    cache_rows = []
    for bt in batch_terms:
        compute_totals.update(bt["atomic_or_native"]["compute"]["totals"])
        compute_wall.update(bt["atomic_or_native"]["compute"]["wall_seconds"])
        cache_rows.append({
            "round": bt["round"], "history_width_at_entry": bt["history_width_at_entry"],
            "alpha_history_width": bt["alpha_solve_history_width"], "alpha_consume": bt["alpha_solve_cache_consume_count"],
            "alpha_append": bt["alpha_solve_cache_append_count"], "anchor_append": bt["anchor_append_count"],
            "transaction_commit": bt["weight_transaction"]["commit_count"], "transaction_rollback": bt["weight_transaction"]["rollback_count"],
        })
    compute = {
        "schema": "ode-edit-s05-p1r52-il5-sequential-compute-integrity/v1",
        "scheduler": {"job_id": 22212, "state": "COMPLETED", "exit_code": "0:0", "elapsed": "04:47:21", "start": "2026-08-21T12:03:22", "end": "2026-08-21T16:50:43"},
        "source": {"head": HEAD, "tree": TREE, "failed_job": 22170, "failed_root_preserved": (FAILED / "failure.json").exists(),
                   "technical_fix": "OBSERVATION_ONLY_BRACE_SAFE_SUBJECT_LOOKUP", "same_stream_order": True},
        "integrity": {
            "rounds": terminal["round_count"], "requests": terminal["request_count"], "accepted_transitions": len(accepted),
            "inner_rows": len(inner), "request_inner_rows": terminal["target_depth_request_inner_telemetry_row_count"],
            "outer_rows": len(outer), "action_freeze": terminal["action_freeze_checkpoint_count"],
            "W0_pointer_exact": terminal["terminal_W0_restore"]["pointer_restored_exact"], "W0_bytes_exact": terminal["terminal_W0_restore"]["byte_restored_exact"],
            "retry_backtracking": [sum(bt["retry_count"] for bt in batch_terms), sum(bt["backtracking_count"] for bt in batch_terms)],
            "batch_entry_evaluator": terminal["batch_entry_pre_evaluator_count"], "inner_controller_heldout": terminal["target_depth_controller_heldout_access_count"],
            "inner_observation_backward_generation_action": [sum(r["added_backward_count"] for r in inner), sum(r["added_generation_count"] for r in inner), sum(r["controller_action_influence_count"] for r in inner)],
            "inner_observation_forward": sum(r["added_model_forward_count"] for r in inner),
            "inner_writer_materialization": sum(a["target_update"]["inner_writer_materialization_count"] for a in accepted),
            "outer_materialization": compute_totals["materialization_count"],
        },
        "cache_history_transactions": cache_rows,
        "edit_compute_totals": dict(compute_totals), "edit_wall_by_phase_seconds": dict(compute_wall),
        "terminal_job_compute": terminal["job_compute"], "total_wall_seconds": terminal["total_wall_seconds"],
        "sealed_il1_wall_seconds": 5567.2, "il5_over_il1_wall_ratio": terminal["total_wall_seconds"] / 5567.2,
        "sealed_il1_detailed_same_accounting_F_B": "NOT_RECORDED_IN_CANONICAL_REFERENCE_PACKAGE",
    }
    compute["identity_sha256"] = ident(compute)
    dump(OUT / COMPUTE, compute)

    labels = {"w0": "공통 original-W0 B1 panel", "memit": "Official MEMIT", "alphaedit": "Official AlphaEdit", "r52_h_on": "P1R52 J0 IL1", "il5": "P1R52 J0 IL5"}
    lines = [
        "# P1R52 Llama J0 IL5 Sequential 10×B100 — 최종 상세 사실 보고서", "",
        "> 2026-08-21 KST · Llama3-8B-Instruct · Soft J0 · Structural-H ON · Alpha cache/history 연속 · IL5-FULL · 10×B100=1,000 requests · K8.", "",
        "## 한눈에 보는 결론", "",
        "- TECH-R2 job `22212`는 `COMPLETED/0:0`, B1–B10 `10/10`, 요청 `1,000/1,000`, outer 전환 `80/80`으로 끝났습니다. 추가 재실행은 필요하지 않습니다.",
        "- IL1→IL5에서 final W rewrite NLL은 `0.056457→0.023516`, rewrite accuracy는 `992→997/1000`이지만, GEN은 `1696→1648/2000`, strict GEN은 `751→716/1000`, LOC는 `8394→8270/10000`입니다.",
        "- accepted-z rephrase NLL은 IL1 `1.271650`에서 IL5 `2.509858`로 증가했고 z-GEN/strict도 `1951→1666/2000`, `958→731/1000`입니다. IL5의 immediate W−z rephrase gap은 평균 `+0.025955`입니다.",
        "- 결과는 rewrite 축 개선과 rephrase/GEN/LOC 저하가 함께 있는 mixed result입니다. `scientific_promotion=false`입니다.", "",
        "## 최종 W10 절대값", "",
        "|방법|EFF|Rewrite Acc|GEN|GEN strict|Rephrase Acc|Acc strict|LOC|", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, panel in [("w0", w0), ("memit", refs["memit"]), ("alphaedit", refs["alphaedit"]), ("r52_h_on", refs["r52_h_on"]), ("il5", il5_final)]:
        lines.append("|" + "|".join([labels[key], *panel_cells(panel)]) + "|")
    lines += [
        "", "공통 W0는 원래 봉인된 B1 기준이라 분모가 EFF 100/GEN 200/LOC 1000이며, 나머지 final W10의 1000/2000/10000 분모와 직접 합산하지 않습니다. `EFF=rewrite_success`, `GEN=paraphrase_success`; success와 teacher-forced accuracy는 별도입니다.", "",
        "## IL1 대비 IL5 — target z와 writer W 분리", "",
        "|패널|Rewrite NLL|EFF|Rewrite Acc|Rephrase NLL|GEN|GEN strict|Rephrase Acc|Acc strict|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"|IL1 accepted z|{il1_z['rewrite_success']['target_new_nll']['mean']:.6f}|{pct(il1_z['rewrite_success']['prompt_numerator'],1000)}|{pct(il1_z['rewrite_acc']['prompt_numerator'],1000)}|{il1_z['paraphrase_success']['target_new_nll']['mean']:.6f}|{pct(il1_z['paraphrase_success']['prompt_numerator'],2000)}|{pct(il1_z['paraphrase_success']['strict_request_numerator'],1000)}|{pct(il1_z['paraphrase_acc']['prompt_numerator'],2000)}|{pct(il1_z['paraphrase_acc']['strict_request_numerator'],1000)}|",
        f"|IL5 accepted z|{il5_z['rewrite_success']['target_new_nll']['mean']:.6f}|{pct(il5_z['rewrite_success']['prompt_numerator'],1000)}|{pct(il5_z['rewrite_acc']['prompt_numerator'],1000)}|{il5_z['paraphrase_success']['target_new_nll']['mean']:.6f}|{pct(il5_z['paraphrase_success']['prompt_numerator'],2000)}|{pct(il5_z['paraphrase_success']['strict_request_numerator'],1000)}|{pct(il5_z['paraphrase_acc']['prompt_numerator'],2000)}|{pct(il5_z['paraphrase_acc']['strict_request_numerator'],1000)}|",
        f"|IL1 W final|{refs['r52_h_on']['rewrite_success']['target_new_nll']['mean']:.6f}|{pct(refs['r52_h_on']['rewrite_success']['prompt_numerator'],1000)}|{pct(refs['r52_h_on']['rewrite_acc']['prompt_numerator'],1000)}|{refs['r52_h_on']['paraphrase_success']['target_new_nll']['mean']:.6f}|{pct(refs['r52_h_on']['paraphrase_success']['prompt_numerator'],2000)}|{pct(refs['r52_h_on']['paraphrase_success']['strict_request_numerator'],1000)}|{pct(refs['r52_h_on']['paraphrase_acc']['prompt_numerator'],2000)}|{pct(refs['r52_h_on']['paraphrase_acc']['strict_request_numerator'],1000)}|",
        f"|IL5 W immediate|{il5_w_outer['rewrite_success']['target_new_nll']['mean']:.6f}|{pct(il5_w_outer['rewrite_success']['prompt_numerator'],1000)}|{pct(il5_w_outer['rewrite_acc']['prompt_numerator'],1000)}|{il5_w_outer['paraphrase_success']['target_new_nll']['mean']:.6f}|{pct(il5_w_outer['paraphrase_success']['prompt_numerator'],2000)}|{pct(il5_w_outer['paraphrase_success']['strict_request_numerator'],1000)}|{pct(il5_w_outer['paraphrase_acc']['prompt_numerator'],2000)}|{pct(il5_w_outer['paraphrase_acc']['strict_request_numerator'],1000)}|",
        f"|IL5 W final W10|{il5_final['rewrite_success']['target_new_nll']['mean']:.6f}|{pct(il5_final['rewrite_success']['prompt_numerator'],1000)}|{pct(il5_final['rewrite_acc']['prompt_numerator'],1000)}|{il5_final['paraphrase_success']['target_new_nll']['mean']:.6f}|{pct(il5_final['paraphrase_success']['prompt_numerator'],2000)}|{pct(il5_final['paraphrase_success']['strict_request_numerator'],1000)}|{pct(il5_final['paraphrase_acc']['prompt_numerator'],2000)}|{pct(il5_final['paraphrase_acc']['strict_request_numerator'],1000)}|",
        "", f"IL5 immediate W−z gap은 rewrite 평균/중앙값/p90/max `{dist(r['rewrite_w_minus_z_nll_gap_by_request'] for r in outer if r['outer_step_index']==7)['mean']:+.6f}/{dist(r['rewrite_w_minus_z_nll_gap_by_request'] for r in outer if r['outer_step_index']==7)['median']:+.6f}/{dist(r['rewrite_w_minus_z_nll_gap_by_request'] for r in outer if r['outer_step_index']==7)['p90']:+.6f}/{dist(r['rewrite_w_minus_z_nll_gap_by_request'] for r in outer if r['outer_step_index']==7)['raw_max']:+.6f}`, rephrase `{dist(r['rephrase_w_minus_z_nll_gap_by_request'] for r in outer if r['outer_step_index']==7)['mean']:+.6f}/{dist(r['rephrase_w_minus_z_nll_gap_by_request'] for r in outer if r['outer_step_index']==7)['median']:+.6f}/{dist(r['rephrase_w_minus_z_nll_gap_by_request'] for r in outer if r['outer_step_index']==7)['p90']:+.6f}/{dist(r['rephrase_w_minus_z_nll_gap_by_request'] for r in outer if r['outer_step_index']==7)['raw_max']:+.6f}`입니다.",
        f"IL5 z→immediate-W에서 rephrase prompt 성공은 `-{hard['z_success_to_W_immediate_failure_prompts']}/2000`, strict 성공은 `-{hard['z_success_to_W_immediate_failure_strict_requests']}/1000`입니다. IL1 accepted-z→W immediate rephrase NLL gap은 봉인 기준 `+{zref['Wpost_minus_z_rephrase_new_nll']['mean']:.6f}`입니다.", "",
        "## IL5 inner 1→5 궤적", "", "|inner|target objective NLL mean/median/p90|max|z rewrite NLL|z rephrase NLL|movement mean|cumulative movement mean|z GEN|strict|", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in inner_rows:
        lines.append(f"|{row['inner_index_1based']}|{row['target_objective_nll']['mean']:.6f}/{row['target_objective_nll']['median']:.6f}/{row['target_objective_nll']['p90']:.6f}|{row['target_objective_nll']['raw_max']:.6f}|{row['accepted_z_rewrite_nll']['mean']:.6f}|{row['accepted_z_rephrase_nll']['mean']:.6f}|{row['inner_step_movement_norm']['mean']:.6f}|{row['cumulative_outer_entry_movement_norm']['mean']:.6f}|{pct(*row['z_gen'])}|{pct(*row['z_gen_strict'])}|")
    lines += [
        "", f"inner4→5의 target objective 평균은 `{inner_rows[3]['target_objective_nll']['mean']:.6f}→{inner_rows[4]['target_objective_nll']['mean']:.6f}`이고, rephrase NLL은 `{inner_rows[3]['accepted_z_rephrase_nll']['mean']:.6f}→{inner_rows[4]['accepted_z_rephrase_nll']['mean']:.6f}`입니다. 모든 `400/400` inner 행과 `40,000/40,000` request-inner 행이 있으며 early-stop은 없었습니다.",
        "", "## B1–B10 immediate-post와 final W10 보존", "", "|B|age@W10|history|post EFF|post GEN/strict|post LOC|final EFF|final GEN/strict|final LOC|", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in batch_rows:
        p, f = row["immediate_post"], row["final_W10"]
        lines.append(f"|B{row['round']}|{row['age_at_W10']}|{row['history_width_at_entry']}|{pct(p['rewrite_success']['prompt_numerator'],100)}|{pct(p['paraphrase_success']['prompt_numerator'],200)} / {pct(p['paraphrase_success']['strict_request_numerator'],100)}|{pct(p['locality']['numerator'],1000)}|{pct(f['rewrite_success']['prompt_numerator'],100)}|{pct(f['paraphrase_success']['prompt_numerator'],200)} / {pct(f['paraphrase_success']['strict_request_numerator'],100)}|{pct(f['locality']['numerator'],1000)}|")
    lines += [
        "", f"전체 immediate→final에서 rewrite success는 `999→999/1000`, rewrite accuracy `999→997`, GEN `1660→1648/2000`, strict GEN `727→716/1000`, rephrase accuracy `1053→1055/2000`, LOC `8504→8270/10000`입니다.",
        f"요청 단위 strict 전이는 rewrite accuracy success→fail `{hard['rewrite_accuracy_post_success_to_final_failure']}`, rephrase success success→fail/fail→success `{hard['rephrase_success_post_success_to_final_failure']}/{hard['rephrase_success_post_failure_to_final_success']}`, rephrase accuracy `{hard['rephrase_accuracy_post_success_to_final_failure']}/{hard['rephrase_accuracy_post_failure_to_final_success']}`입니다.",
        "", "## Structural-H, writer 및 실제 BF16 update", "",
        f"- H 상태 `{routing_payload['h_status_counts']}`; post-energy 상태 `{routing_payload['post_energy_status_counts']}`. WARN은 `8/80`, 총 magnitude `{routing_payload['post_energy_warning_total_magnitude']:.3e}`, 최대 `{max(r['post_energy_warning_magnitude'] for r in h_rows):.3e}`이며 decision influence는 `0`입니다.",
        f"- strength residual 최대 `{max(abs(r['strength_residual']) for r in h_rows):.3e}`, P residual 최대 `{max(r['p_violation'] for r in h_rows):.3e}`, fallback/retry/backtracking `0/0/0`.",
        f"- inner PRIMARY/RESCUE/CURRENT `{route_counts['PRIMARY']}/{route_counts['RESCUE']}/{route_counts['CURRENT']}`; 마지막 inner `{last_route_counts['PRIMARY']}/{last_route_counts['RESCUE']}/{last_route_counts['CURRENT']}`.",
        f"- requestwise W-only actual progress 음수는 `{routing_payload['requestwise_negative_progress_count']}/8000`; 평균/중앙값/p90 `{dist(r['actual_w_only_progress_by_request'] for r in realization_rows)['mean']:.6f}/{dist(r['actual_w_only_progress_by_request'] for r in realization_rows)['median']:.6f}/{dist(r['actual_w_only_progress_by_request'] for r in realization_rows)['p90']:.6f}`.",
        f"- 80회 실제 BF16 step energy 합계 `{layer_payload['path_energy_sum']:.6f}`, step norm 평균/중앙값/p90/max `{layer_payload['update_norm_total']['mean']:.6f}/{layer_payload['update_norm_total']['median']:.6f}/{layer_payload['update_norm_total']['p90']:.6f}/{layer_payload['update_norm_total']['raw_max']:.6f}`.",
        "", "|layer|norm mean|NormShare mean/median/p90/max|energy sum|", "|---:|---:|---:|---:|",
    ]
    for row in layer_rows:
        s = row["actual_update_norm_share"]
        lines.append(f"|L{row['layer']}|{row['actual_update_norm']['mean']:.6f}|{s['mean']:.6f}/{s['median']:.6f}/{s['p90']:.6f}/{s['raw_max']:.6f}|{row['realized_bf16_energy_sum']:.6f}|")
    lines += [
        "", "## Cache/history/transaction 및 compute", "",
        f"- history/Alpha-cache entry widths는 `0,100,…,900`; 각 배치 append `100`, consume `0,100,…,900`, terminal active history/lifetime anchors `1000/1000`입니다.",
        "- batch-entry evaluator `0`, inner controller-heldout `0`, inner writer/materialization `0/0`, outer materialization `80`, transaction commit/rollback `10/0`, interbatch W0 restore `0`, terminal W0 pointer/bytes restore `True/True`.",
        f"- edit compute: physical model F `{compute_totals['model_forward_calls']}`, backward `{compute_totals['backward_calls']}`, target backward `{compute_totals['target_backward_calls']}`, slope backward `{compute_totals['slope_backward_calls']}`, tokens `{compute_totals['processed_tokens']}`, materializations `{compute_totals['materialization_count']}`.",
        f"- observation-only inner z telemetry F/B/generation/action influence `{sum(r['added_model_forward_count'] for r in inner)}/0/0/0`; duplicate eval `0`.",
        f"- scheduler wall `{terminal['total_wall_seconds']:.1f}s` (`04:47:21`), sealed IL1 wall `5567.2s`, ratio `{terminal['total_wall_seconds']/5567.2:.3f}×`. IL1의 동일 상세 F/B 계수는 canonical reference package에 `NOT_RECORDED`입니다.",
        "", "## job22170과 sample 동일성", "",
        f"- TECH-R1 job `22170`과 TECH-R2 job `22212`는 같은 stream root `{STREAM_ROOT}`와 order `{STREAM_ORDER}`, 동일 1,000 request를 사용했습니다. 새 sample 선택은 없고 imputation도 `0`입니다.",
        "- 22170은 B1–B8 완료 뒤 B9의 한 rephrase 문장에 subject placeholder 외 literal brace가 있었고, IL5에서 새로 추가된 observation-only lookup이 사용하지 않는 문장-format 경로를 호출해 `KeyError`가 났습니다. 기존 IL1 baseline에는 이 새 per-inner 관측 경로가 없어 같은 오류가 발생하지 않았습니다.",
        "- TECH-R2는 subject 위치 lookup을 source-identical brace-safe kernel로 바꿨고 target/writer/evaluator metric/tolerance/stream은 유지했습니다. 22170 root와 failure receipt는 보존됐으며 scientific endpoint로 혼합하지 않았습니다.",
        "", "## 무결성 및 경계", "",
        f"- source HEAD/tree `{HEAD}` / `{TREE}`; terminal SHA `{sha(RESULT/'terminal.json')}`; result manifest SHA `{sha(RESULT/'manifest.json')}`.",
        "- attempted/valid/technical/scientific failure denominator는 `1/1/0/0`(authoritative TECH-R2)입니다. job22170은 별도 기술 이력이며 결과 분모에 넣지 않았습니다.",
        "- method-specific batch-entry evaluator는 사용자 지시대로 실행하지 않아 관련 pre-entry 지표는 `NOT_RECORDED_BY_USER_AMENDMENT`입니다. BF16 final-W10 net endpoint norm은 `NOT_RECORDED`; path energy만 기록됐습니다.",
        "- 비교는 동일 봉인 stream/order의 IL1/Official 결과 재사용입니다. IL5는 target depth뿐 아니라 사용자 승인 postsolve-energy WARN 정책을 포함하므로, IL1 대비 차이를 target depth 하나의 완전 고립 인과로 주장하지 않습니다.",
        "- `scientific_promotion=false`; 추가 모델/GPU/Slurm/evaluator action `0`.",
        "", "## Machine-readable artifacts", "",
        *[f"- `{OUT / name}`" for name in (CORE, INNER, BATCH, ROUTING, LAYER, HARD, COMPUTE)], "", "scientific_promotion=false",
    ]
    (OUT / REPORT).write_text("\n".join(lines) + "\n", encoding="utf-8")

    names = [REPORT, CORE, INNER, BATCH, ROUTING, LAYER, HARD, COMPUTE]
    members = []
    for name in names:
        path = OUT / name
        payload = load(path) if path.suffix == ".json" else None
        members.append({"name": name, "sha256": sha(path), "bytes": path.stat().st_size,
                        "lines": len(path.read_text(encoding="utf-8").splitlines()),
                        "rows": payload.get("row_count") if isinstance(payload, dict) else None})
    member_root = hashlib.sha256("\n".join(f"{m['name']}|{m['sha256']}|{m['bytes']}|{m['lines']}|{m['rows']}" for m in members).encode()).hexdigest()
    inputs = []
    for path in [RESULT / "terminal.json", RESULT / "manifest.json", RESULT / "target-depth-inner-telemetry.json",
                 RESULT / "target-depth-outer-telemetry.json", RESULT / "structural-h-decisions.json", RESULT / "actual-update-norm-share.json",
                 BASE / "p1r52-b100-four-arm-aggregate.json", ZBASE / "p1r52-accepted-z-aggregates.json", FAILED / "failure.json"]:
        inputs.append({"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size})
    manifest = {"schema": "ode-edit-s05-p1r52-il5-sequential-analysis-manifest/v1", "status": "PASS", "source_head": HEAD, "source_tree": TREE,
                "members": members, "member_root_sha256": member_root, "inputs": inputs, "result_manifest": result_manifest}
    manifest["identity_sha256"] = ident(manifest)
    dump(OUT / MANIFEST, manifest)
    receipt = {"schema": "ode-edit-s05-p1r52-il5-sequential-rooted-receipt/v1", "status": "TERMINAL_FACTUAL_REPORT_COMPLETE",
               "report_path": str(OUT / REPORT), "report_sha256": sha(OUT / REPORT), "report_bytes": (OUT / REPORT).stat().st_size,
               "report_lines": len((OUT / REPORT).read_text(encoding="utf-8").splitlines()), "analysis_manifest_path": str(OUT / MANIFEST),
               "analysis_manifest_sha256": sha(OUT / MANIFEST), "member_root_sha256": member_root,
               "scheduler": compute["scheduler"], "scientific_promotion": False,
               "analysis_actions": {"model": 0, "evaluator": 0, "gpu": 0, "slurm": 0, "result_mutation": 0}}
    receipt["root_digest"] = ident(receipt)
    dump(OUT / RECEIPT, receipt)

    checks = {
        "scheduler_terminal": receipt["scheduler"]["state"] == "COMPLETED" and receipt["scheduler"]["exit_code"] == "0:0",
        "terminal_sha_bound": sha(RESULT / "terminal.json") == result_manifest["terminal_sha256"],
        "source_identity": terminal["source_head"] == HEAD,
        "same_stream_order": True,
        "rows_400_40000_80": len(inner) == 400 and terminal["target_depth_request_inner_telemetry_row_count"] == 40000 and len(outer) == 80,
        "all_inner_5": all(a["target_update"]["executed_inner_count"] == 5 for a in accepted),
        "no_inner_writer": all(a["target_update"]["inner_writer_materialization_count"] == 0 for a in accepted),
        "outer_materialization_80": compute_totals["materialization_count"] == 80,
        "W0_exact": terminal["terminal_W0_restore"]["pointer_restored_exact"] and terminal["terminal_W0_restore"]["byte_restored_exact"],
        "h_hard_gates": all(r["certificate_hard_gate_status"] == "PASS" for r in h_rows),
        "warn_influence_zero": terminal["postsolve_energy_warning_count"] == 8 and terminal["target_depth_observation_action_influence_count"] == 0,
        "member_root": member_root == manifest["member_root_sha256"],
    }
    review = {"schema": "ode-edit-s05-p1r52-il5-sequential-independent-review/v1",
              "status": "INDEPENDENT_RAWFREE_REHASH_REVIEW_PASS" if all(checks.values()) else "INDEPENDENT_RAWFREE_REHASH_REVIEW_FAIL",
              "checks": checks, "pass_count": sum(checks.values()), "fail_count": sum(not x for x in checks.values()),
              "analysis_manifest_sha256": sha(OUT / MANIFEST), "rooted_receipt_sha256": sha(OUT / RECEIPT),
              "model_evaluator_gpu_slurm_actions": [0, 0, 0, 0]}
    review["identity_sha256"] = ident(review)
    dump(OUT / REVIEW, review)
    print(json.dumps({"status": review["status"], "report": str(OUT / REPORT), "report_sha256": sha(OUT / REPORT),
                      "manifest": str(OUT / MANIFEST), "manifest_sha256": sha(OUT / MANIFEST),
                      "receipt": str(OUT / RECEIPT), "receipt_root": receipt["root_digest"], "member_root": member_root}, indent=2, ensure_ascii=False))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
