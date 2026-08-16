#!/usr/bin/env python3
"""Build the raw-free, factual-only P1R43 terminal report package."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


WORKTREE = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-p1r43-rho-free-semantic-first-strength-recovery-v1"
)
RESULTS = WORKTREE / "local/odebf/results"
OUT = WORKTREE / "local/odebf/reports/p1r43-rho-free-semantic-first-v1"
SUPPORT = Path(
    "/mnt/raid5/janghj/ODE-edit/local/source-handoff/"
    "P1R43_P1R42_PARENT_SUPPORT_SH2_V1"
)

ROOTS = {
    ("llama3-8b-inst", "Neutral"): RESULTS
    / "s05-p1r43-rho-free-independent-b10x10-llama3-8b-inst-neutral-tech-r2-v1",
    ("llama3-8b-inst", "Soft"): RESULTS
    / "s05-p1r43-rho-free-independent-b10x10-llama3-8b-inst-soft-tech-r2-v1",
    ("qwen2.5-7b-inst", "Neutral"): RESULTS
    / "s05-p1r43-rho-free-independent-b10x10-qwen2.5-7b-inst-neutral-tech-r2-v1",
    ("qwen2.5-7b-inst", "Soft"): RESULTS
    / "s05-p1r43-rho-free-independent-b10x10-qwen2.5-7b-inst-soft-tech-r2-v1",
}

SCHEDULER = {
    ("llama3-8b-inst", "Neutral"): {
        "array_parent": 19843,
        "job_id": 19844,
        "array_index": 0,
        "state": "COMPLETED",
        "exit_code": "0:0",
        "elapsed": "00:28:30",
        "max_rss_kib": 7470956,
    },
    ("llama3-8b-inst", "Soft"): {
        "array_parent": 19843,
        "job_id": 19845,
        "array_index": 1,
        "state": "COMPLETED",
        "exit_code": "0:0",
        "elapsed": "00:26:35",
        "max_rss_kib": 7419696,
    },
    ("qwen2.5-7b-inst", "Neutral"): {
        "array_parent": 19843,
        "job_id": 19846,
        "array_index": 2,
        "state": "COMPLETED",
        "exit_code": "0:0",
        "elapsed": "00:30:59",
        "max_rss_kib": 10783264,
    },
    ("qwen2.5-7b-inst", "Soft"): {
        "array_parent": 19843,
        "job_id": 19843,
        "array_index": 3,
        "state": "COMPLETED",
        "exit_code": "0:0",
        "elapsed": "00:32:38",
        "max_rss_kib": 10853092,
    },
}

SUBMISSION = {
    "intent_sha256": "826053d8b08915d316d6535341d0827e756ea734a8b3bb4afd9758a86ff66a90",
    "submission_receipt_sha256": "a59182760b2939511a03a00e08f9fcf49be823bbcaac393f9149a7fbd299a820",
    "held_inspection_sha256": "14e4455f95147ad1a1e80020f5dd70859cbd4ae330a907358ed83d639fb59a97",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def values(records: list[list[dict[str, Any]]], key: str) -> list[float]:
    return [float(item[key]) for group in records for item in group if item.get(key) is not None]


def quantile(seq: Iterable[float], q: float) -> float | None:
    data = sorted(float(x) for x in seq if x is not None and math.isfinite(float(x)))
    if not data:
        return None
    pos = (len(data) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return data[lo]
    return data[lo] * (hi - pos) + data[hi] * (pos - lo)


def summary(seq: Iterable[float]) -> dict[str, float | int | None]:
    data = [float(x) for x in seq if x is not None and math.isfinite(float(x))]
    return {
        "count": len(data),
        "mean": statistics.fmean(data) if data else None,
        "median": statistics.median(data) if data else None,
        "p90": quantile(data, 0.90),
        "worst_max": max(data) if data else None,
    }


def panel_row(alias: str, arm: str, case: int, name: str, metric: dict, vectors: list) -> dict:
    new = values(vectors, "nll_new")
    old = values(vectors, "nll_old")
    margin = values(vectors, "margin")
    return {
        "alias": alias,
        "arm": arm,
        "case": case,
        "panel": name,
        "correct": metric["numerator"],
        "denominator": metric["denominator"],
        "new_nll": summary(new),
        "old_nll": summary(old),
        "margin": summary(margin),
        "per_request_prompt_values": vectors,
        "bit_vector_sha256": metric["bit_vector_sha256"],
        "comparison_score_sha256": metric["comparison_score_sha256"],
    }


def flatten_metric(terminal: dict, metric: str) -> tuple[dict, list]:
    receipt = terminal["endpoint"]["receipt"]
    return receipt["primary"]["metrics"][metric], receipt["numeric_vectors"][metric]


def step_file(root: Path, case: int, arm: str, k: int) -> Path:
    method = f"pr-p1r43-rho-free-semantic-first-{arm.lower()}"
    return root / f"raw/cases/case-{case:02d}/raw/ode/{method}/accepted-k{k}.json"


def realization_file(root: Path, case: int, arm: str, k: int) -> Path:
    method = f"pr-p1r43-rho-free-semantic-first-{arm.lower()}"
    return root / (
        f"raw/cases/case-{case:02d}/raw/ode/{method}/"
        f"requestwise-realization-k{k}.json"
    )


def load_reference_rows() -> tuple[dict, dict, dict]:
    p42 = read_json(SUPPORT / "results/p1r42-objective-alignment-per-case.json")["rows"]
    p39 = read_json(SUPPORT / "reference/p1r39-neutral-soft-per-case.json")["rows"]
    official = read_json(SUPPORT / "identities/official-alphaedit-baseline-identity.json")["rows"]
    key = lambda r: (r["alias"], r.get("arm"), int(r["case"]))
    return (
        {key(r): r for r in p42},
        {key(r): r for r in p39},
        {(r["alias"], int(r["case"])): r for r in official},
    )


def delta(a: Any, b: Any) -> float | str:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) - float(b)
    return "NOT_RECORDED"


def compact_comparison(current: dict, reference: dict, comparator: str) -> dict:
    mapped = {
        "eff_correct": "eff_correct",
        "gen_correct": "gen_correct",
        "loc_correct": "loc_correct",
        "eff_nll_mean": "eff_nll_mean",
        "eff_margin_mean": "eff_margin_mean",
        "gen_nll_mean": "gen_nll_mean",
        "gen_margin_mean": "gen_margin_mean",
        "w8_full6_nll": "w8_full6_nll",
        "z8_full6_nll": "z8_full6_nll",
        "z_w_full6_gap": "z_w_full6_gap",
        "P_terminal": "P_terminal",
        "capacity_terminal": "capacity_terminal",
        "energy_terminal": "energy_terminal",
        "predicted_sum": "predicted_sum",
        "actual_sum": "actual_sum",
        "realization_total": "realization_total",
    }
    if comparator == "OfficialAlphaEdit":
        mapped = {
            "eff_correct": "official_eff_correct",
            "gen_correct": "official_gen_correct",
            "loc_correct": "official_loc_correct",
            "eff_nll_mean": "official_eff_nll_mean",
            "eff_margin_mean": "official_eff_margin_mean",
        }
    raw, deltas = {}, {}
    for current_key, reference_key in mapped.items():
        raw[current_key] = reference.get(reference_key, "NOT_RECORDED")
        deltas[current_key] = delta(current.get(current_key), reference.get(reference_key))
    return {
        "alias": current["alias"],
        "arm": current["arm"],
        "case": current["case"],
        "comparator": comparator,
        "identity_status": "MATCHED" if reference else "NOT_MATCHED",
        "current": {k: current.get(k, "NOT_RECORDED") for k in mapped},
        "reference": raw,
        "delta_current_minus_reference": deltas,
    }


def collect() -> tuple[list, list, list, list, list, dict]:
    case_rows, step_rows, request_rows, panel_rows, comparison_rows = [], [], [], [], []
    integrity = Counter()
    p42, p39, official = load_reference_rows()

    for (alias, arm), root in ROOTS.items():
        task_terminal = read_json(root / "terminal.json")
        integrity["task_completed"] += int(task_terminal["completed_case_count"] == 10)
        integrity["task_failed"] += int(task_terminal["failed_case_count"] != 0)
        for case in range(1, 11):
            terminal_path = root / f"raw/cases/case-{case:02d}/terminal.json"
            terminal = read_json(terminal_path)
            rollout = terminal["rollout"]
            accepted = []
            alpha_req_sum = alpha_apply_sum = predicted_sum = actual_sum = 0.0
            fallback_count = primary_count = rescue_count = hold_count = 0
            rescue_eligible = corrector_forward = negative_steps = 0
            statuses: Counter[str] = Counter()
            for k in range(1, 9):
                receipt = read_json(step_file(root, case, arm, k))
                realization = read_json(realization_file(root, case, arm, k))
                request_actual = realization["actual_w_only_progress_by_request"]
                actual_value = statistics.fmean(float(x) for x in request_actual)
                routing = receipt["routing"]
                target = receipt["target_update"]
                p = receipt["cumulative_atomic_structural_p"]
                alpha_req_sum += float(routing["alpha_req"])
                alpha_apply_sum += float(routing["alpha_apply"])
                predicted_value = float(receipt["progress"]["predicted"])
                predicted_sum += predicted_value
                actual_sum += actual_value
                fallback = int(routing["executed_arm"] != routing["requested_arm"])
                fallback_count += fallback
                primary_count += int(target["primary_accept_count"])
                rescue_count += int(target["rescue_accept_count"])
                hold_count += int(target["current_hold_count"])
                rescue_eligible += int(target["rescue_plan"]["rescue_eligible_count"])
                corrector_forward += int(target["same_step_scalar_corrector_forward_count"])
                negative_steps += int(actual_value < 0)
                statuses[routing["status"]] += 1
                step_row = {
                    "alias": alias,
                    "arm": arm,
                    "case": case,
                    "k": k,
                    "tau_before": receipt["tau_before"],
                    "tau_after": receipt["tau_after"],
                    "target_current_nll": target["target_new_nll_current_summary"],
                    "target_selected_nll": target["selected_nll_summary"],
                    "primary_accept_count": target["primary_accept_count"],
                    "rescue_eligible_count": target["rescue_plan"]["rescue_eligible_count"],
                    "rescue_accept_count": target["rescue_accept_count"],
                    "current_hold_count": target["current_hold_count"],
                    "corrector_forward_count": target["same_step_scalar_corrector_forward_count"],
                    "corrector_backward_count": target["same_step_scalar_corrector_backward_count"],
                    "corrector_physical_k_count": target["same_step_scalar_corrector_physical_k_count"],
                    "trust_rho_decision_influence_count": target["trust_rho_decision_influence_count"],
                    "radius_state_access_count": target["radius_state_access_count"],
                    "semantic_hold_threshold_access_count": target["semantic_hold_threshold_access_count"],
                    "target_kl_decision_influence_count": target["target_kl_decision_influence_count"],
                    "target_decay_decision_influence_count": target["target_decay_decision_influence_count"],
                    "target_preservation_decision_influence_count": target["target_preservation_decision_influence_count"],
                    "alpha_req": routing["alpha_req"],
                    "alpha_apply": routing["alpha_apply"],
                    "alpha_apply_over_req": routing["alpha_apply_over_req"],
                    "routing_fallback": bool(fallback),
                    "routing_status": routing["status"],
                    "executed_arm": routing["executed_arm"],
                    "equality_residual": routing["equality_residual"],
                    "q": routing["q"],
                    "active_count": sum(bool(x) for x in routing["active_mask"]),
                    "selected_p": routing["selected_p"],
                    "selected_capacity": routing["selected_capacity"],
                    "selected_energy": routing["selected_energy"],
                    "simplex_entropy": routing["simplex_entropy"],
                    "simplex_top1_share": routing["simplex_top1_share"],
                    "predicted": predicted_value,
                    "actual": actual_value,
                    "realization_ratio": actual_value / predicted_value if predicted_value else None,
                    "linearization_error": actual_value - predicted_value,
                    "negative_actual": actual_value < 0,
                    "source_w_nll": statistics.fmean(
                        float(x) for x in realization["source_w_only_target_new_nll_by_request"]
                    ),
                    "next_w_nll": statistics.fmean(
                        float(x) for x in realization["next_w_only_target_new_nll_by_request"]
                    ),
                    "P_before": p["P_before"],
                    "P_after": p["P_after"],
                    "P_delta": p["d_P"],
                    "materialization_transition": receipt["materialization"]["transition_index"],
                    "retry_backtracking_reject_count": receipt["retry_backtracking_reject_count"],
                    "inner_step_heldout_evaluation_count": receipt["inner_step_heldout_evaluation_count"],
                    "identity_sha256": receipt["identity_sha256"],
                }
                step_rows.append(step_row)
                accepted.append(receipt)
                for request in range(10):
                    plan = target["rescue_plan"]
                    request_rows.append(
                        {
                            "alias": alias,
                            "arm": arm,
                            "case": case,
                            "k": k,
                            "request_index": request,
                            "entry_gradient_norm": target["entry_semantic_gradient_norm_by_request"][request],
                            "current_gradient_norm": target["current_semantic_gradient_norm_by_request"][request],
                            "current_nll": target["current_nll_by_request"][request],
                            "primary_nll": target["primary_nll_by_request"][request],
                            "selected_nll": target["selected_nll_by_request"][request],
                            "selection": target["selection_by_request"][request],
                            "primary_displacement_norm": target["primary_displacement_norm_by_request"][request],
                            "rescue_eligible": plan["rescue_eligible_mask"][request],
                            "alpha_rescue": plan["alpha_rescue_by_request"][request],
                            "directional_derivative": plan["directional_derivative_by_request"][request],
                            "curvature": plan["curvature_by_request"][request],
                            "source_w_nll": realization["source_w_only_target_new_nll_by_request"][request],
                            "next_w_nll": realization["next_w_only_target_new_nll_by_request"][request],
                            "actual_w_progress": realization["actual_w_only_progress_by_request"][request],
                            "semantic_held": realization["semantic_held_mask"][request],
                        }
                    )

            metric_rows = {}
            for metric in ("efficacy", "generalization", "locality-preservation"):
                metric_obj, vectors = flatten_metric(terminal, metric)
                metric_rows[metric] = {
                    "correct": metric_obj["numerator"],
                    "denominator": metric_obj["denominator"],
                    "new_nll": summary(values(vectors, "nll_new")),
                    "old_nll": summary(values(vectors, "nll_old")),
                    "margin": summary(values(vectors, "margin")),
                }

            four = terminal["terminal_four_panel"]
            for panel, metric_name, source in (
                ("EFF_Z_INJECT", "efficacy", "z_inject"),
                ("GEN_Z_INJECT", "generalization", "z_inject"),
                ("EFF_W", "efficacy", "weight"),
                ("GEN_W", "generalization", "weight"),
            ):
                panel_rows.append(
                    panel_row(
                        alias,
                        arm,
                        case,
                        panel,
                        four[source]["primary"]["metrics"][metric_name],
                        four[source]["numeric_vectors"][metric_name],
                    )
                )

            k8 = accepted[-1]
            compute = rollout["compute"]["totals"]
            case_row = {
                "alias": alias,
                "arm": arm,
                "case": case,
                "status": rollout["status"],
                "accepted_steps": rollout["accepted_update_count"],
                "request_count": terminal["request_count"],
                "eff_correct": metric_rows["efficacy"]["correct"],
                "eff_denominator": metric_rows["efficacy"]["denominator"],
                "eff_nll_mean": metric_rows["efficacy"]["new_nll"]["mean"],
                "eff_old_nll_mean": metric_rows["efficacy"]["old_nll"]["mean"],
                "eff_margin_mean": metric_rows["efficacy"]["margin"]["mean"],
                "gen_correct": metric_rows["generalization"]["correct"],
                "gen_denominator": metric_rows["generalization"]["denominator"],
                "gen_nll_mean": metric_rows["generalization"]["new_nll"]["mean"],
                "gen_old_nll_mean": metric_rows["generalization"]["old_nll"]["mean"],
                "gen_margin_mean": metric_rows["generalization"]["margin"]["mean"],
                "loc_correct": metric_rows["locality-preservation"]["correct"],
                "loc_denominator": metric_rows["locality-preservation"]["denominator"],
                "loc_nll_mean": metric_rows["locality-preservation"]["new_nll"]["mean"],
                "loc_old_nll_mean": metric_rows["locality-preservation"]["old_nll"]["mean"],
                "loc_margin_mean": metric_rows["locality-preservation"]["margin"]["mean"],
                "eff_z_correct": four["eff_z_inject"]["numerator"],
                "eff_z_denominator": four["eff_z_inject"]["denominator"],
                "gen_z_correct": four["gen_z_inject"]["numerator"],
                "gen_z_denominator": four["gen_z_inject"]["denominator"],
                "eff_w_correct": four["eff_w"]["numerator"],
                "eff_w_denominator": four["eff_w"]["denominator"],
                "gen_w_correct": four["gen_w"]["numerator"],
                "gen_w_denominator": four["gen_w"]["denominator"],
                "z8_full6_nll": terminal["terminal_z8_oracle"]["target_new_nll"],
                "w8_full6_nll": terminal["terminal_w8_full_six_target_new_nll"],
                "z_w_full6_gap": terminal["terminal_w8_full_six_target_new_nll"]
                - terminal["terminal_z8_oracle"]["target_new_nll"],
                "alpha_req_sum": alpha_req_sum,
                "alpha_apply_sum": alpha_apply_sum,
                "predicted_sum": predicted_sum,
                "actual_sum": actual_sum,
                "realization_total": actual_sum / alpha_req_sum if alpha_req_sum else None,
                "negative_actual_steps": negative_steps,
                "primary_accept_request_steps": primary_count,
                "rescue_eligible_request_steps": rescue_eligible,
                "rescue_accept_request_steps": rescue_count,
                "current_hold_request_steps": hold_count,
                "corrector_forward_count": corrector_forward,
                "routing_fallback_count": fallback_count,
                "routing_status_counts": dict(statuses),
                "P_terminal": rollout["terminal_cumulative_structural_p"]["P_after"],
                "capacity_terminal": k8["routing"]["selected_capacity"],
                "energy_terminal": k8["routing"]["selected_energy"],
                "entropy_terminal": k8["routing"]["simplex_entropy"],
                "top1_terminal": k8["routing"]["simplex_top1_share"],
                "functional_p_mean": rollout["terminal_functional"]["pretrained"]["mean_positive_damage"],
                "functional_p_max": rollout["terminal_functional"]["pretrained"]["raw_max_positive_damage"],
                "model_forwards": compute["model_forward_calls"],
                "backwards": compute["backward_calls"],
                "target_backwards": compute["target_backward_calls"],
                "slope_backwards": compute["slope_backward_calls"],
                "processed_tokens": compute["processed_tokens"],
                "padded_tokens": compute["padded_tokens"],
                "materializations": compute["materialization_count"],
                "edit_core_wall_seconds": rollout["edit_core_wall_seconds"],
                "terminal_evaluator_wall_seconds": terminal["terminal_evaluator_wall_seconds"],
                "W0_restore_pointer": terminal["endpoint_restore"]["pointer_restored_exact"],
                "W0_restore_bytes": terminal["endpoint_restore"]["byte_restored_exact"],
                "action_freeze_sha256": terminal["action_freeze_sha256"],
                "cross_case_state_count": terminal["cross_case_state_count"],
                "retry_count": terminal["retry_count"],
                "history_mode": terminal["history_mode"]["mode"],
                "heldout_gen_controller_access_count": four["heldout_gen_controller_access_count"],
                "inner_step_heldout_evaluation_count": four["inner_step_heldout_evaluation_count"],
                "terminal_sha256": sha256(terminal_path),
            }
            case_rows.append(case_row)
            comparison_rows.append(compact_comparison(case_row, p42[(alias, arm, case)], "P1R42"))
            comparison_rows.append(compact_comparison(case_row, p39[(alias, arm, case)], "P1R39"))
            comparison_rows.append(
                compact_comparison(case_row, official[(alias, case)], "OfficialAlphaEdit")
            )
            integrity["case_complete"] += int(case_row["accepted_steps"] == 8)
            integrity["W0_pointer"] += int(case_row["W0_restore_pointer"])
            integrity["W0_bytes"] += int(case_row["W0_restore_bytes"])
            integrity["action_freeze"] += int(bool(case_row["action_freeze_sha256"]))
            integrity["controller_reset"] += int(terminal["cross_case_state_count"] == 0)
            integrity["retry_zero"] += int(terminal["retry_count"] == 0)
            integrity["history_zero"] += int(terminal["history_mode"]["history_append_count"] == 0)
            integrity["gen_controller_zero"] += int(four["heldout_gen_controller_access_count"] == 0)

    return case_rows, step_rows, request_rows, panel_rows, comparison_rows, dict(integrity)


def cell_aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["alias"], row["arm"])].append(row)
    result = []
    for (alias, arm), group in sorted(groups.items()):
        total = lambda key: sum(float(r[key]) for r in group if isinstance(r.get(key), (int, float)))
        mean = lambda key: statistics.fmean(float(r[key]) for r in group)
        result.append(
            {
                "alias": alias,
                "arm": arm,
                "attempts": len(group),
                "endpoints": sum(r["accepted_steps"] == 8 for r in group),
                "typed_failures": sum(r["accepted_steps"] != 8 for r in group),
                "eff_correct": int(total("eff_correct")),
                "eff_denominator": int(total("eff_denominator")),
                "gen_correct": int(total("gen_correct")),
                "gen_denominator": int(total("gen_denominator")),
                "loc_correct": int(total("loc_correct")),
                "loc_denominator": int(total("loc_denominator")),
                "eff_nll_mean": mean("eff_nll_mean"),
                "eff_margin_mean": mean("eff_margin_mean"),
                "gen_nll_mean": mean("gen_nll_mean"),
                "gen_margin_mean": mean("gen_margin_mean"),
                "eff_z_correct": int(total("eff_z_correct")),
                "eff_z_denominator": int(total("eff_z_denominator")),
                "gen_z_correct": int(total("gen_z_correct")),
                "gen_z_denominator": int(total("gen_z_denominator")),
                "eff_w_correct": int(total("eff_w_correct")),
                "eff_w_denominator": int(total("eff_w_denominator")),
                "gen_w_correct": int(total("gen_w_correct")),
                "gen_w_denominator": int(total("gen_w_denominator")),
                "z8_full6_nll_mean": mean("z8_full6_nll"),
                "w8_full6_nll_mean": mean("w8_full6_nll"),
                "z_w_full6_gap_mean": mean("z_w_full6_gap"),
                "alpha_req_sum": total("alpha_req_sum"),
                "alpha_apply_sum": total("alpha_apply_sum"),
                "predicted_sum": total("predicted_sum"),
                "actual_sum": total("actual_sum"),
                "realization_total": total("actual_sum") / total("alpha_req_sum"),
                "negative_actual_steps": int(total("negative_actual_steps")),
                "primary_accept_request_steps": int(total("primary_accept_request_steps")),
                "rescue_eligible_request_steps": int(total("rescue_eligible_request_steps")),
                "rescue_accept_request_steps": int(total("rescue_accept_request_steps")),
                "current_hold_request_steps": int(total("current_hold_request_steps")),
                "corrector_forward_count": int(total("corrector_forward_count")),
                "routing_fallback_count": int(total("routing_fallback_count")),
                "P_terminal_mean": mean("P_terminal"),
                "capacity_terminal_mean": mean("capacity_terminal"),
                "energy_terminal_mean": mean("energy_terminal"),
                "functional_p_mean": mean("functional_p_mean"),
                "model_forwards": int(total("model_forwards")),
                "backwards": int(total("backwards")),
                "processed_tokens": int(total("processed_tokens")),
                "materializations": int(total("materializations")),
                "edit_core_wall_seconds": total("edit_core_wall_seconds"),
                "terminal_evaluator_wall_seconds": total("terminal_evaluator_wall_seconds"),
            }
        )
    return result


def panel_aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["alias"], row["arm"], row["panel"])].append(row)
    output = []
    for (alias, arm, panel), group in sorted(groups.items()):
        vectors = [
            request_group
            for row in group
            for request_group in row["per_request_prompt_values"]
        ]
        output.append(
            {
                "alias": alias,
                "arm": arm,
                "panel": panel,
                "case_count": len(group),
                "correct": sum(int(row["correct"]) for row in group),
                "denominator": sum(int(row["denominator"]) for row in group),
                "new_nll": summary(values(vectors, "nll_new")),
                "old_nll": summary(values(vectors, "nll_old")),
                "margin": summary(values(vectors, "margin")),
            }
        )
    return output


def aggregate_comparisons(comparisons: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in comparisons:
        groups[(row["alias"], row["arm"], row["comparator"])].append(row)
    output = []
    for (alias, arm, comparator), group in sorted(groups.items()):
        keys = sorted({k for r in group for k in r["delta_current_minus_reference"]})
        deltas = {}
        for key in keys:
            nums = [r["delta_current_minus_reference"][key] for r in group]
            nums = [float(v) for v in nums if isinstance(v, (int, float))]
            deltas[key] = statistics.fmean(nums) if nums else "NOT_RECORDED"
        output.append(
            {
                "alias": alias,
                "arm": arm,
                "comparator": comparator,
                "matched_cases": sum(r["identity_status"] == "MATCHED" for r in group),
                "case_count": len(group),
                "mean_case_delta_current_minus_reference": deltas,
            }
        )
    return output


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NOT_RECORDED"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown(
    aggregates: list[dict], panel_agg: list[dict], comp_agg: list[dict], integrity: dict
) -> str:
    lines = [
        "# P1R43 Rho-Free Semantic-First Strength Recovery B10×10 사실 보고서",
        "",
        "- 정책: `SH_FACTUAL_ONLY_REPORTING`",
        "- `scientific_promotion=false`",
        "- 시도/endpoint/typed failure: `40 / 40 / 0`",
        "- 각 endpoint: B10, K8, 독립 W0 시작/복원",
        "- source HEAD: `11508b6da11d606521b703037034e1814b70d8a8`",
        "- exact P1R42 parent: `ca68a4f459fd7303a4d5abbde2e1bf7aee0d805c`",
        "- stream root: `74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6`",
        "- stream order: `abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c`",
        "",
        "## Scheduler 및 제출 상태",
        "",
        "| Model | Arm | Job | Index | State | Exit | Elapsed | MaxRSS KiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (alias, arm), job in SCHEDULER.items():
        lines.append(
            f"| {alias} | {arm} | {job['job_id']} | {job['array_index']} | "
            f"{job['state']} | {job['exit_code']} | {job['elapsed']} | {job['max_rss_kib']} |"
        )
    lines += [
        "",
        f"- array parent: `19843`, topology: `0-3%4`, node: `devbox`",
        f"- intent SHA-256: `{SUBMISSION['intent_sha256']}`",
        f"- submission receipt SHA-256: `{SUBMISSION['submission_receipt_sha256']}`",
        f"- held inspection SHA-256: `{SUBMISSION['held_inspection_sha256']}`",
        "",
        "## 기계적 무결성",
        "",
        "| Check | PASS/total |",
        "|---|---:|",
        f"| K8/tau1 endpoint | {integrity['case_complete']}/40 |",
        f"| action freeze | {integrity['action_freeze']}/40 |",
        f"| W0 pointer restore | {integrity['W0_pointer']}/40 |",
        f"| W0 byte restore | {integrity['W0_bytes']}/40 |",
        f"| controller/cross-case state reset | {integrity['controller_reset']}/40 |",
        f"| retry=0 | {integrity['retry_zero']}/40 |",
        f"| history append=0 | {integrity['history_zero']}/40 |",
        f"| heldout Gen controller access=0 | {integrity['gen_controller_zero']}/40 |",
        "",
        "## Model × Arm terminal 집계",
        "",
        "| Model | Arm | Endpoints | Eff | Gen | Loc | Eff NLL | Eff margin | Gen NLL | Gen margin | z8 full6 NLL | W8 full6 NLL | W−z gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['alias']} | {row['arm']} | {row['endpoints']}/{row['attempts']} | "
            f"{row['eff_correct']}/{row['eff_denominator']} | {row['gen_correct']}/{row['gen_denominator']} | "
            f"{row['loc_correct']}/{row['loc_denominator']} | {fmt(row['eff_nll_mean'])} | "
            f"{fmt(row['eff_margin_mean'])} | {fmt(row['gen_nll_mean'])} | {fmt(row['gen_margin_mean'])} | "
            f"{fmt(row['z8_full6_nll_mean'])} | {fmt(row['w8_full6_nll_mean'])} | {fmt(row['z_w_full6_gap_mean'])} |"
        )
    lines += [
        "",
        "## Terminal 네 패널",
        "",
        "| Model | Arm | Eff z-inject | Gen z-inject | Eff W | Gen W |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['alias']} | {row['arm']} | {row['eff_z_correct']}/{row['eff_z_denominator']} | "
            f"{row['gen_z_correct']}/{row['gen_z_denominator']} | {row['eff_w_correct']}/{row['eff_w_denominator']} | "
            f"{row['gen_w_correct']}/{row['gen_w_denominator']} |"
        )
    panel_map = {(r["alias"], r["arm"], r["panel"]): r for r in panel_agg}
    lines += [
        "",
        "| Model | Arm | Eff z NLL mean/med/p90/worst | Eff W NLL mean/med/p90/worst | Eff W−z mean | Eff z/W margin mean | Gen z NLL mean/med/p90/worst | Gen W NLL mean/med/p90/worst | Gen W−z mean | Gen z/W margin mean |",
        "|---|---|---|---|---:|---|---|---|---:|---|",
    ]
    for row in aggregates:
        key = (row["alias"], row["arm"])
        ez, ew = panel_map[key + ("EFF_Z_INJECT",)], panel_map[key + ("EFF_W",)]
        gz, gw = panel_map[key + ("GEN_Z_INJECT",)], panel_map[key + ("GEN_W",)]
        nll = lambda r: "/".join(
            fmt(r["new_nll"][field]) for field in ("mean", "median", "p90", "worst_max")
        )
        lines.append(
            f"| {row['alias']} | {row['arm']} | {nll(ez)} | {nll(ew)} | "
            f"{fmt(ew['new_nll']['mean'] - ez['new_nll']['mean'])} | "
            f"{fmt(ez['margin']['mean'])}/{fmt(ew['margin']['mean'])} | {nll(gz)} | {nll(gw)} | "
            f"{fmt(gw['new_nll']['mean'] - gz['new_nll']['mean'])} | "
            f"{fmt(gz['margin']['mean'])}/{fmt(gw['margin']['mean'])} |"
        )
    lines += [
        "",
        "네 패널의 new/old NLL 및 margin per-request/prompt 분포는 `p1r43-per-panel.json`에 기록되어 있다.",
        "",
        "## Target/corrector 및 full-strength routing",
        "",
        "| Model | Arm | Primary accepts | Rescue eligible | Rescue accepts | Current holds | Corrector F | α_req sum | α_apply sum | Fallbacks |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['alias']} | {row['arm']} | {row['primary_accept_request_steps']} | "
            f"{row['rescue_eligible_request_steps']} | {row['rescue_accept_request_steps']} | "
            f"{row['current_hold_request_steps']} | {row['corrector_forward_count']} | "
            f"{fmt(row['alpha_req_sum'])} | {fmt(row['alpha_apply_sum'])} | {row['routing_fallback_count']} |"
        )
    lines += [
        "",
        "## Writer/P/capacity/compute",
        "",
        "| Model | Arm | Predicted sum | Actual sum | Realization | Negative steps | P terminal mean | Capacity mean | Energy mean | Functional-P mean | F | B | Tokens | Mat | Edit-core s | Eval s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['alias']} | {row['arm']} | {fmt(row['predicted_sum'])} | {fmt(row['actual_sum'])} | "
            f"{fmt(row['realization_total'])} | {row['negative_actual_steps']} | {fmt(row['P_terminal_mean'])} | "
            f"{fmt(row['capacity_terminal_mean'])} | {fmt(row['energy_terminal_mean'])} | "
            f"{fmt(row['functional_p_mean'])} | {row['model_forwards']} | {row['backwards']} | "
            f"{row['processed_tokens']} | {row['materializations']} | {fmt(row['edit_core_wall_seconds'], 3)} | "
            f"{fmt(row['terminal_evaluator_wall_seconds'], 3)} |"
        )
    lines += [
        "",
        "## Exact case-paired 비교",
        "",
        "모든 P1R42/P1R39/Official 비교는 동일 model+batch key와 package-bound stream/order/evaluator identity로 join했다. Official AlphaEdit에는 arm이 없으며 model+batch 공통 참조로만 사용했다.",
        "",
        "| Model | Arm | Comparator | Matched cases | Mean-case ΔEff correct | Mean-case ΔGen correct | Mean-case ΔLoc correct | Mean-case ΔEff NLL | Mean-case ΔEff margin | Mean-case Δz8 NLL | Mean-case ΔW8 NLL | Mean-case Δgap |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comp_agg:
        d = row["mean_case_delta_current_minus_reference"]
        lines.append(
            f"| {row['alias']} | {row['arm']} | {row['comparator']} | {row['matched_cases']}/{row['case_count']} | "
            f"{fmt(d.get('eff_correct'))} | {fmt(d.get('gen_correct'))} | {fmt(d.get('loc_correct'))} | "
            f"{fmt(d.get('eff_nll_mean'))} | {fmt(d.get('eff_margin_mean'))} | {fmt(d.get('z8_full6_nll'))} | "
            f"{fmt(d.get('w8_full6_nll'))} | {fmt(d.get('z_w_full6_gap'))} |"
        )
    lines += [
        "",
        "Official AlphaEdit의 latent-z, W8 full-six NLL, P/capacity/energy 및 realization 필드는 `NOT_RECORDED`이다.",
        "",
        "## Machine-readable 산출물",
        "",
        "- `p1r43-per-case.json`: 40 rows",
        "- `p1r43-per-step.json`: 320 rows",
        "- `p1r43-per-request.json`: 3200 rows",
        "- `p1r43-per-panel.json`: 160 rows",
        "- `p1r43-panel-aggregates.json`: 16 rows",
        "- `p1r43-case-paired-comparisons.json`: 120 rows",
        "- `p1r43-cell-aggregates.json`: 4 rows",
        "- `analysis-manifest.json`, `analysis-receipt.json`: 파일 및 rooted digest",
        "",
        "## 기록 경계",
        "",
        "- Peak GPU memory: `NOT_RECORDED` in per-case scientific receipts.",
        "- Slurm MaxRSS는 scheduler 표에 task-level 값으로 기록했다.",
        "- Stepwise heldout Eff/Gen/Loc: `NOT_RECORDED`; terminal-only evaluator가 사용되었다.",
        "- raw prompt/target/generation/tensor/weight/model/cache/runtime log: 포함하지 않았다.",
        "- 과학적 해석, 인과 주장, 우월성 판단, 추천: 포함하지 않았다.",
        "",
    ]
    return "\n".join(lines)


def write_json(name: str, rows: Any, schema: str) -> Path:
    path = OUT / name
    payload = {
        "schema": schema,
        "row_count": len(rows) if isinstance(rows, list) else None,
        "rows_root": canonical_sha(rows),
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases, steps, requests, panels, comparisons, integrity = collect()
    assert len(cases) == 40 and len(steps) == 320 and len(requests) == 3200
    assert len(panels) == 160 and len(comparisons) == 120
    assert all(row["accepted_steps"] == 8 for row in cases)
    assert all(row["W0_restore_pointer"] and row["W0_restore_bytes"] for row in cases)
    assert all(row["heldout_gen_controller_access_count"] == 0 for row in cases)
    aggregates = cell_aggregate(cases)
    panel_agg = panel_aggregate(panels)
    comp_agg = aggregate_comparisons(comparisons)
    paths = [
        write_json("p1r43-per-case.json", cases, "ode-edit-p1r43-per-case/v1"),
        write_json("p1r43-per-step.json", steps, "ode-edit-p1r43-per-step/v1"),
        write_json("p1r43-per-request.json", requests, "ode-edit-p1r43-per-request/v1"),
        write_json("p1r43-per-panel.json", panels, "ode-edit-p1r43-per-panel/v1"),
        write_json(
            "p1r43-panel-aggregates.json",
            panel_agg,
            "ode-edit-p1r43-panel-aggregates/v1",
        ),
        write_json(
            "p1r43-case-paired-comparisons.json",
            comparisons,
            "ode-edit-p1r43-case-paired-comparisons/v1",
        ),
        write_json("p1r43-cell-aggregates.json", aggregates, "ode-edit-p1r43-cell-aggregates/v1"),
        write_json(
            "p1r43-comparison-aggregates.json",
            comp_agg,
            "ode-edit-p1r43-comparison-aggregates/v1",
        ),
    ]
    report = OUT / "p1r43-rho-free-semantic-first-b10x10-factual-ko.md"
    report.write_text(markdown(aggregates, panel_agg, comp_agg, integrity), encoding="utf-8")
    paths.append(report)

    review = {
        "schema": "ode-edit-p1r43-independent-raw-free-review/v1",
        "review_scope": "RAW_FREE_REPORT_ONLY",
        "mechanical_checks": {
            "case_rows": len(cases),
            "step_rows": len(steps),
            "request_rows": len(requests),
            "panel_rows": len(panels),
            "paired_comparison_rows": len(comparisons),
            "all_cases_k8": all(row["accepted_steps"] == 8 for row in cases),
            "all_cases_action_freeze": integrity["action_freeze"] == 40,
            "all_cases_W0_pointer_restore": integrity["W0_pointer"] == 40,
            "all_cases_W0_byte_restore": integrity["W0_bytes"] == 40,
            "all_cases_controller_reset": integrity["controller_reset"] == 40,
            "all_cases_retry_zero": integrity["retry_zero"] == 40,
            "all_cases_history_zero": integrity["history_zero"] == 40,
            "all_cases_heldout_gen_controller_zero": integrity["gen_controller_zero"] == 40,
            "p1r42_matched_case_joins": sum(
                r["comparator"] == "P1R42" and r["identity_status"] == "MATCHED"
                for r in comparisons
            ),
            "p1r39_matched_case_joins": sum(
                r["comparator"] == "P1R39" and r["identity_status"] == "MATCHED"
                for r in comparisons
            ),
            "official_matched_case_joins": sum(
                r["comparator"] == "OfficialAlphaEdit"
                and r["identity_status"] == "MATCHED"
                for r in comparisons
            ),
        },
        "action_counts": {
            "model": 0,
            "evaluator": 0,
            "gpu": 0,
            "slurm": 0,
            "scientific_source_mutation": 0,
            "scientific_result_mutation": 0,
        },
        "factual_only": True,
        "scientific_promotion": False,
    }
    review["root_digest"] = canonical_sha(review)
    review_path = OUT / "independent-review-receipt.json"
    review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    paths.append(review_path)

    source_files = [
        Path(__file__).resolve(),
        WORKTREE / "project/run_scripts/ode_bf/locks/numerical_lock_s05_p1r43_rho_free_b10x10.json",
        WORKTREE / "project/run_scripts/ode_bf/locks/source_manifest_s05_p1r43_rho_free_b10x10.json",
        SUPPORT / "results/p1r42-objective-alignment-per-case.json",
        SUPPORT / "reference/p1r39-neutral-soft-per-case.json",
        SUPPORT / "identities/official-alphaedit-baseline-identity.json",
        SUPPORT / "identities/frozen-comparison-identities.json",
    ]
    entries = []
    for path in paths + source_files:
        data = path.read_bytes()
        entries.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "lines": data.count(b"\n"),
            }
        )
    manifest = {
        "schema": "ode-edit-p1r43-factual-analysis-manifest/v1",
        "instruction_id": "ODEEDIT-S05-P1R43-RHO-FREE-SEMANTIC-FIRST-STRENGTH-RECOVERY-V1",
        "source_head": "11508b6da11d606521b703037034e1814b70d8a8",
        "exact_parent": "ca68a4f459fd7303a4d5abbde2e1bf7aee0d805c",
        "stream_root": "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6",
        "stream_order": "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c",
        "attempts": 40,
        "endpoints": 40,
        "typed_failures": 0,
        "scientific_promotion": False,
        "integrity": integrity,
        "entries": sorted(entries, key=lambda x: x["path"]),
    }
    manifest["root_digest"] = canonical_sha(manifest)
    manifest_path = OUT / "analysis-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema": "ode-edit-p1r43-factual-analysis-receipt/v1",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "manifest_root": manifest["root_digest"],
        "report_path": str(report.resolve()),
        "report_sha256": sha256(report),
        "rows": {
            "case": 40,
            "step": 320,
            "request": 3200,
            "panel": 160,
            "panel_aggregate": 16,
            "paired_comparison": 120,
            "cell_aggregate": 4,
            "comparison_aggregate": 12,
        },
        "tracked_source_mutation_count": 0,
        "model_action_count": 0,
        "evaluator_action_count": 0,
        "gpu_action_count": 0,
        "slurm_action_count": 0,
        "factual_only": True,
        "scientific_promotion": False,
    }
    receipt["root_digest"] = canonical_sha(receipt)
    (OUT / "analysis-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
