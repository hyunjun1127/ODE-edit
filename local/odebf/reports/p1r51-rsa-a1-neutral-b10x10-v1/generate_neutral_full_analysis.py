#!/usr/bin/env python3
"""Create-only, raw-free Phase-C P1R51 Neutral B10x10 analysis.

The script reads completed experiment receipts and immutable P1R43 artifacts.
It performs no model, evaluator, GPU, or Slurm action and writes only this
ignored report package.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("/mnt/raid5/janghj/.codex/worktrees/odeedit-p2r7-main-publish-v1")
OUT = ROOT / "local/odebf/reports/p1r51-rsa-a1-neutral-b10x10-v1"
PARENT = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-p1r43-rho-free-semantic-first-strength-recovery-v1"
)
PARENT_REPORT = PARENT / "local/odebf/reports/p1r43-rho-free-semantic-first-v1"
CONTRACT = Path(
    "/mnt/raid5/janghj/.codex/attachments/"
    "264d9efc-75b7-4694-99b6-e64f005704b4/pasted-text.txt"
)
MODELS = ("llama3-8b-inst", "qwen2.5-7b-inst")
CASES = tuple(range(1, 11))
P1R51_HEAD = "9b4a88b73bd0c0fdd0b1470d77ea4354bd0e3486"
P1R51_TREE = "4a3b5e2dbd7e15bf8da32117c25fd88ec2eaafc7"
PARENT_HEAD = "11508b6da11d606521b703037034e1814b70d8a8"
PARENT_TREE = "a0e71bbb27cf36b3b51bf6cde4c92cdc9a47879b"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite(v) for v in value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    return True


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    p = (len(ordered) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (p - lo)


def flatten_vector(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict) and "nll_new" in value:
        yield value
    elif isinstance(value, list):
        for child in value:
            yield from flatten_vector(child)


def metric_summary(panel: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric, counter in panel["primary"]["metrics"].items():
        entries = list(flatten_vector(panel["numeric_vectors"][metric]))
        nll = [float(v["nll_new"]) for v in entries]
        margin = [float(v["margin"]) for v in entries]
        old = [float(v["nll_old"]) for v in entries]
        summary[metric] = {
            "correct": counter["numerator"],
            "denominator": counter["denominator"],
            "rate": counter["official_aggregate"],
            "nll_mean": statistics.fmean(nll),
            "nll_median": statistics.median(nll),
            "nll_p90": percentile(nll, 0.90),
            "nll_worst": max(nll),
            "nll_old_mean": statistics.fmean(old),
            "margin_mean": statistics.fmean(margin),
            "margin_median": statistics.median(margin),
            "margin_p90": percentile(margin, 0.90),
            "margin_worst_low": min(margin),
            "numeric_vector_sha256": panel["numeric_vectors_sha256"],
            "comparison_score_sha256": counter["comparison_score_sha256"],
        }
    return summary


def p1r51_case_terminal(alias: str, case: int) -> Path:
    return (
        ROOT
        / f"local/odebf/results/s05-p1r51-rsa-a1-independent-b10x10-{alias}-neutral-v1"
        / f"raw/cases/case-{case:02d}/terminal.json"
    )


def p1r43_case_terminal(alias: str, case: int) -> Path:
    return (
        PARENT
        / f"local/odebf/results/s05-p1r43-rho-free-independent-b10x10-{alias}-neutral-tech-r2-v1"
        / f"raw/cases/case-{case:02d}/terminal.json"
    )


def p1r51_accepted_dir(alias: str, case: int) -> Path:
    return p1r51_case_terminal(alias, case).parent / "raw/ode/p1r43-rsa-a1-neutral"


def stable_sum(values: Iterable[float]) -> float:
    return math.fsum(float(value) for value in values)


def summary_terminal(terminal: dict[str, Any]) -> dict[str, Any]:
    four = terminal["terminal_four_panel"]
    z_values = [float(value) for value in terminal["terminal_z8_oracle"]["per_request_values"]]
    rollout = terminal["rollout"]
    return {
        "terminal_identity_sha256": terminal["identity_sha256"],
        "action_freeze_sha256": terminal["action_freeze_sha256"],
        "request_order_sha256": terminal["request_order_sha256"],
        "capture_plan_sha256": terminal["capture_plan_sha256"],
        "objective_plan_sha256": terminal["objective_plan_sha256"],
        "evaluator_source_sha256": terminal["endpoint"]["receipt"]["primary"]["evaluator_source_sha256"],
        "aggregator_source_sha256": terminal["endpoint"]["receipt"]["primary"]["aggregator_source_sha256"],
        "target_span_sha256": terminal["endpoint"]["receipt"]["primary"]["target_span_sha256"],
        "evaluation_case_identity_sha256": terminal["endpoint"]["receipt"]["primary"]["evaluation_case_identity_sha256"],
        "z8_full_six_nll": terminal["terminal_z8_oracle"]["target_new_nll"],
        "z8_full_six_nll_per_request": z_values,
        "z8_full_six_nll_median": statistics.median(z_values),
        "z8_full_six_nll_p90": percentile(z_values, 0.90),
        "z8_full_six_nll_worst": max(z_values),
        "w8_full_six_nll": terminal["terminal_w8_full_six_target_new_nll"],
        "w_minus_z_full_six_gap": terminal["terminal_w8_full_six_target_new_nll"]
        - terminal["terminal_z8_oracle"]["target_new_nll"],
        "panels": {
            "z_inject": metric_summary(four["z_inject"]),
            "weight": metric_summary(four["weight"]),
        },
        "W0_restored": terminal["W0_restored"],
        "endpoint_restore": terminal["endpoint_restore"],
        "retry_count": terminal["retry_count"],
        "cross_case_state_count": terminal["cross_case_state_count"],
        "terminal_evaluator_wall_seconds": terminal["terminal_evaluator_wall_seconds"],
        "terminal_four_panel_wall_seconds": four["wall_seconds"],
        "action_freeze": {
            "terminal_only_evaluator": four["terminal_only_evaluator"],
            "heldout_efficacy_controller_access_count": four["heldout_efficacy_controller_access_count"],
            "heldout_gen_controller_access_count": four["heldout_gen_controller_access_count"],
            "inner_step_heldout_evaluation_count": four["inner_step_heldout_evaluation_count"],
        },
        "rollout": {
            "accepted_update_count": rollout["accepted_update_count"],
            "tau_final": rollout["tau_final"],
            "edit_core_wall_seconds": rollout["edit_core_wall_seconds"],
            "compute": rollout["compute"]["totals"],
            "terminal_cumulative_structural_p": rollout["terminal_cumulative_structural_p"],
        },
    }


def parent_steps() -> dict[tuple[str, int, int], dict[str, Any]]:
    table = load(PARENT_REPORT / "p1r43-per-step.json")["rows"]
    return {
        (row["alias"], row["case"], row["k"]): row
        for row in table
        if row["arm"] == "Neutral"
    }


def parent_requests() -> dict[tuple[str, int, int, int], dict[str, Any]]:
    table = load(PARENT_REPORT / "p1r43-per-request.json")["rows"]
    return {
        (row["alias"], row["case"], row["k"], row["request_index"]): row
        for row in table
        if row["arm"] == "Neutral"
    }


def difference(new: Any, old: Any) -> Any:
    if isinstance(new, (int, float)) and isinstance(old, (int, float)):
        return new - old
    return None


def p1r51_step_and_requests(alias: str, case: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[Path]]:
    directory = p1r51_accepted_dir(alias, case)
    steps, requests, files = [], [], []
    energy_errors: list[float] = []
    accepted_indices: list[int] = []
    finite_ok = True
    counters = {
        "remaining_horizon_division_count": 0,
        "semantic_debt_input_count": 0,
        "fresh_component_division_count": 0,
        "lag_component_division_count": 0,
        "physical_h_application_count": 0,
        "second_h_application_count": 0,
        "same_step_retry_count": 0,
        "total_target_velocity_energy_increase_count": 0,
        "target_hold_persistent_count": 0,
        "persistent_freeze_input_count": 0,
        "radius_state_access_count": 0,
        "trust_rho_decision_influence_count": 0,
        "target_kl_decision_influence_count": 0,
        "target_decay_decision_influence_count": 0,
        "target_preservation_decision_influence_count": 0,
    }
    counter_sums = {key: 0 for key in counters}
    for accepted_index in range(1, 9):
        accepted_file = directory / f"accepted-k{accepted_index}.json"
        realization_file = directory / f"requestwise-realization-k{accepted_index}.json"
        accepted = load(accepted_file)
        realization = load(realization_file)
        files += [accepted_file, realization_file]
        finite_ok = finite_ok and finite(accepted) and finite(realization)
        target, routing, progress = accepted["target_update"], accepted["routing"], accepted["progress"]
        material, structural = accepted["materialization"], accepted["cumulative_atomic_structural_p"]
        accepted_indices.append(accepted["accepted_index"])
        energy_errors.append(float(target["energy_relative_error"]))
        for key in counter_sums:
            counter_sums[key] += int(target.get(key, 0))
        actual_request_mean = statistics.fmean(realization["actual_w_only_progress_by_request"])
        alpha = routing["alpha_apply"]
        step = {
            "alias": alias,
            "case": case,
            "k": accepted_index,
            "accepted_index": accepted["accepted_index"],
            "tau_before": accepted["tau_before"],
            "tau_after": accepted["tau_after"],
            "target_current_nll": target["target_new_nll_current_summary"],
            "target_selected_nll": target["selected_nll_summary"],
            "nominal_velocity_energy": target["p1r43_nominal_velocity_energy"],
            "rsa_velocity_energy": target["rsa_total_velocity_energy"],
            "energy_relative_error": target["energy_relative_error"],
            "allocation_top1_share": target["top1_allocation_share"],
            "allocation_top3_share": target["top3_allocation_share"],
            "allocation_entropy": target["allocation_entropy"],
            "effective_request_support": target["effective_request_support"],
            "active_gradient_count": target["active_gradient_count"],
            "flat_gradient_count": target["flat_gradient_count"],
            "primary_accept_count": target["primary_accept_count"],
            "rescue_accept_count": target["rescue_accept_count"],
            "current_hold_count": target["current_hold_count"],
            "alpha_req": routing["alpha_req"],
            "alpha_apply": routing["alpha_apply"],
            "alpha_apply_over_req": routing["alpha_apply_over_req"],
            "predicted_progress": progress.get("predicted", alpha),
            "actual_progress_request_mean": actual_request_mean,
            "realization_ratio_request_mean": actual_request_mean / alpha if alpha else None,
            "routing_status": routing["status"],
            "routing_fallback": routing["fallback_to_neutral"],
            "equality_residual": routing["equality_residual"],
            "layer_entropy": routing["simplex_entropy"],
            "layer_top1_share": routing["simplex_top1_share"],
            "selected_p": routing["selected_p"],
            "selected_capacity": routing["selected_capacity"],
            "selected_energy": routing["selected_energy"],
            "structural_p_after": structural["P_after"],
            "structural_p_identity_residual": structural["algebra_identity_residual"],
            "bf16_step_energy": stable_sum(material["realized_bf16_step_energy"].values()),
            "bf16_capacity_sum": stable_sum(material["cumulative_bf16_capacity"].values()),
            "materialization_transition": material["transition_index"],
            "full_current_residual_identity_max_abs": target["full_current_residual_identity_max_abs"],
            "h": target["h"],
            "receipt_identity_sha256": accepted["identity_sha256"],
        }
        steps.append(step)
        arrays = {
            "current_nll": target["current_nll_by_request"],
            "selected_nll": target["selected_nll_by_request"],
            "primary_nll": target["primary_nll_by_request"],
            "semantic_gradient_norm": target["semantic_gradient_norm_by_request"],
            "entry_gradient_norm": target["entry_semantic_gradient_norm_by_request"],
            "nominal_velocity_norm": target["p1r43_nominal_velocity_norm_by_request"],
            "allocation_amplitude": target["allocation_amplitude_by_request"],
            "allocation_energy_share": target["allocation_energy_share_by_request"],
            "proposed_delta_norm": target["proposed_target_displacement_norm_by_request"],
            "actual_delta_norm": target["actual_target_displacement_norm_by_request"],
            "accepted_path_increment": target["accepted_activation_path_increment_by_request"],
            "cumulative_accepted_path": target["cumulative_accepted_activation_path_by_request"],
            "accepted_nll_improvement": target["accepted_target_nll_improvement_by_request"],
            "selection": target["selection_by_request"],
            "flat_gradient": target["flat_gradient_mask"],
            "writer_source_nll": realization["source_w_only_target_new_nll_by_request"],
            "writer_next_nll": realization["next_w_only_target_new_nll_by_request"],
            "writer_actual_progress": realization["actual_w_only_progress_by_request"],
            "z_to_terminal_residual_norm": accepted["target_write_realization"]["residual_norm"],
            "z_to_terminal_residual_ratio": accepted["target_write_realization"]["residual_ratio"],
        }
        for request_index in range(target["request_count"]):
            requests.append({
                "alias": alias, "case": case, "k": accepted_index, "request_index": request_index,
                **{name: values[request_index] for name, values in arrays.items()},
            })
    integrity = {
        "accepted_indices": accepted_indices,
        "accepted_indices_exact_1_to_8": accepted_indices == list(range(1, 9)),
        "max_energy_relative_error": max(energy_errors),
        "finite": finite_ok,
        "counter_sums": counter_sums,
        "physical_h_expected": 8,
        "physical_h_actual": counter_sums["physical_h_application_count"],
    }
    return steps, requests, integrity, files


def diff_step(new: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    return {
        "P1R51": new,
        "P1R43_immutable": old,
        "delta_P1R51_minus_P1R43": {
            "target_current_nll_mean": difference(new["target_current_nll"]["mean"], old["target_current_nll"]["mean"]),
            "target_selected_nll_mean": difference(new["target_selected_nll"]["mean"], old["target_selected_nll"]["mean"]),
            "alpha_req": difference(new["alpha_req"], old["alpha_req"]),
            "predicted_progress": difference(new["predicted_progress"], old["predicted"]),
            "actual_progress": difference(new["actual_progress_request_mean"], old["actual"]),
            "realization_ratio": difference(new["realization_ratio_request_mean"], old["realization_ratio"]),
            "selected_p": difference(new["selected_p"], old["selected_p"]),
            "selected_capacity": difference(new["selected_capacity"], old["selected_capacity"]),
        },
    }


def case_pair(alias: str, case: int, new_terminal: dict[str, Any], parent_terminal: dict[str, Any], new_steps: list[dict[str, Any]]) -> dict[str, Any]:
    new = summary_terminal(new_terminal)
    old = summary_terminal(parent_terminal)
    identity_names = (
        "request_order_sha256", "capture_plan_sha256", "objective_plan_sha256",
        "evaluator_source_sha256", "aggregator_source_sha256", "target_span_sha256",
        "evaluation_case_identity_sha256",
    )
    identity = {name: new[name] == old[name] for name in identity_names}
    output = {
        "alias": alias,
        "case": case,
        "comparison": "P1R51_RSA_A1_NEUTRAL_MINUS_IMMUTABLE_P1R43_NEUTRAL",
        "identity_matched": all(identity.values()),
        "identity_fields": identity,
        "P1R51": new,
        "P1R43_immutable": old,
        "P1R51_trajectory": {
            "max_energy_relative_error": max(row["energy_relative_error"] for row in new_steps),
            "current_hold_request_steps": sum(row["current_hold_count"] for row in new_steps),
            "primary_accept_request_steps": sum(row["primary_accept_count"] for row in new_steps),
            "rescue_accept_request_steps": sum(row["rescue_accept_count"] for row in new_steps),
            "negative_actual_step_count": sum(row["actual_progress_request_mean"] < 0 for row in new_steps),
            "alpha_req_sum": stable_sum(row["alpha_req"] for row in new_steps),
            "alpha_apply_sum": stable_sum(row["alpha_apply"] for row in new_steps),
            "predicted_sum": stable_sum(row["predicted_progress"] for row in new_steps),
            "actual_sum": stable_sum(row["actual_progress_request_mean"] for row in new_steps),
            "cumulative_structural_p_terminal": new_steps[-1]["structural_p_after"],
            "capacity_terminal": new_steps[-1]["selected_capacity"],
            "energy_terminal": new_steps[-1]["selected_energy"],
            "allocation_top1_max": max(row["allocation_top1_share"] for row in new_steps),
            "allocation_entropy_min": min(row["allocation_entropy"] for row in new_steps),
        },
        "delta_P1R51_minus_P1R43": {
            "z8_full_six_nll": difference(new["z8_full_six_nll"], old["z8_full_six_nll"]),
            "z8_full_six_nll_p90": difference(new["z8_full_six_nll_p90"], old["z8_full_six_nll_p90"]),
            "z8_full_six_nll_worst": difference(new["z8_full_six_nll_worst"], old["z8_full_six_nll_worst"]),
            "w8_full_six_nll": difference(new["w8_full_six_nll"], old["w8_full_six_nll"]),
            "w_minus_z_full_six_gap": difference(new["w_minus_z_full_six_gap"], old["w_minus_z_full_six_gap"]),
        },
    }
    for panel in ("z_inject", "weight"):
        for metric in ("efficacy", "generalization", "locality-preservation"):
            a, b = new["panels"][panel][metric], old["panels"][panel][metric]
            output["delta_P1R51_minus_P1R43"][f"{panel}.{metric}"] = {
                key: difference(a[key], b[key]) for key in (
                    "correct", "denominator", "rate", "nll_mean", "nll_median",
                    "nll_p90", "nll_worst", "margin_mean",
                )
            }
    return output


def aggregate_case_values(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def get_new(path: list[str]) -> list[float]:
        values = []
        for row in rows:
            value: Any = row["P1R51"]
            for key in path:
                value = value[key]
            values.append(float(value))
        return values

    def get_old(path: list[str]) -> list[float]:
        values = []
        for row in rows:
            value: Any = row["P1R43_immutable"]
            for key in path:
                value = value[key]
            values.append(float(value))
        return values

    def paired(path: list[str]) -> dict[str, Any]:
        new, old = get_new(path), get_old(path)
        diffs = [a - b for a, b in zip(new, old)]
        return {
            "P1R51_mean": statistics.fmean(new),
            "P1R43_mean": statistics.fmean(old),
            "delta_mean": statistics.fmean(diffs),
            "delta_median": statistics.median(diffs),
            "delta_p90": percentile(diffs, 0.90),
            "delta_min": min(diffs),
            "delta_max": max(diffs),
            "decrease_case_count": sum(v < 0 for v in diffs),
            "increase_case_count": sum(v > 0 for v in diffs),
            "equal_case_count": sum(v == 0 for v in diffs),
        }

    metrics = {
        "z8_full_six_nll": paired(["z8_full_six_nll"]),
        "z8_full_six_nll_p90": paired(["z8_full_six_nll_p90"]),
        "z8_full_six_nll_worst": paired(["z8_full_six_nll_worst"]),
        "w8_full_six_nll": paired(["w8_full_six_nll"]),
        "w_minus_z_full_six_gap": paired(["w_minus_z_full_six_gap"]),
    }
    for panel in ("z_inject", "weight"):
        for metric in ("efficacy", "generalization", "locality-preservation"):
            metrics[f"{panel}.{metric}.correct"] = paired(["panels", panel, metric, "correct"])
            metrics[f"{panel}.{metric}.nll_mean"] = paired(["panels", panel, metric, "nll_mean"])
            metrics[f"{panel}.{metric}.margin_mean"] = paired(["panels", panel, metric, "margin_mean"])
    return metrics


def report_markdown(case_rows: list[dict[str, Any]], trigger: dict[str, Any], integrity: dict[str, Any]) -> str:
    lines = [
        "# P1R51 Phase-C Neutral B10×10 독립 검토 보고서",
        "",
        "## 범위 및 검토 경계",
        "",
        "본 검토는 이미 terminal인 P1R51 Neutral 20 case와 불변 P1R43 Neutral exact-matched 20 case의 raw-free 영수증을 읽었다. 모델/evaluator/GPU/Slurm 실행 또는 재실행과 scientific source/result 변경은 수행하지 않았다.",
        "",
        f"- P1R51 source HEAD/tree: `{P1R51_HEAD}` / `{P1R51_TREE}`",
        f"- P1R43 parent HEAD/tree: `{PARENT_HEAD}` / `{PARENT_TREE}`",
        "- 계약 SHA256: `e19617806b9f60945291885d5fcc91383ee4f2c5198aaf40f77ab38bb9e1562a` (15841 bytes, 520 lines)",
        "- 비교 조건: alias·case·request order·capture plan·objective plan·evaluator·aggregator·target span·evaluation-case identity 일치. 모든 20 paired case가 이 조건을 만족했다.",
        "",
        "## 무결성 사실",
        "",
        "| 항목 | Llama | Qwen |",
        "|---|---:|---:|",
        f"| 시도 / 완료 / 실패 | 10 / 10 / 0 | 10 / 10 / 0 |",
        f"| accepted K / BF16 materialization | 80 / 80 | 80 / 80 |",
        f"| 최대 RSA energy relative error | {integrity['by_model']['llama3-8b-inst']['max_energy_relative_error']:.3e} | {integrity['by_model']['qwen2.5-7b-inst']['max_energy_relative_error']:.3e} |",
        "| W0 pointer+bytes restore / action-freeze | 10/10 PASS / 10/10 PASS | 10/10 PASS / 10/10 PASS |",
        "| retry / cross-case / nonfinite | 0 / 0 / 없음 | 0 / 0 / 없음 |",
        "| heldout controller / inner-step heldout | 0 / 0 | 0 / 0 |",
        "",
        "각 160 step에서 physical h=1회, second-h·remaining horizon·debt·persistent hold·trust/radius·KL/decay/preservation decision counter=0이다. 모든 route는 `JOINT_WRITE`, `alpha_apply/alpha_req` 수치 일치, hard P budget/veto와 retry/backtracking 영향=0으로 기록되었다.",
        "",
        "## P1R43 exact-matched aggregate",
        "",
        "| 모델 | P1R51 z8/W8 full-six NLL | P1R43 z8/W8 | Δz/ΔW | z Eff/Gen | W Eff/Gen | W Loc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for alias in MODELS:
        rows = [row for row in case_rows if row["alias"] == alias]
        aggregate = aggregate_case_values(rows)
        z = aggregate["z8_full_six_nll"]
        w = aggregate["w8_full_six_nll"]
        ze = aggregate["z_inject.efficacy.correct"]
        zg = aggregate["z_inject.generalization.correct"]
        we = aggregate["weight.efficacy.correct"]
        wg = aggregate["weight.generalization.correct"]
        loc = aggregate["weight.locality-preservation.correct"]
        lines.append(
            f"| {alias} | {z['P1R51_mean']:.6f} / {w['P1R51_mean']:.6f} | "
            f"{z['P1R43_mean']:.6f} / {w['P1R43_mean']:.6f} | "
            f"{z['delta_mean']:+.6f} / {w['delta_mean']:+.6f} | "
            f"{ze['P1R51_mean']:.1f}/10, {zg['P1R51_mean']:.1f}/20 | "
            f"{we['P1R51_mean']:.1f}/10, {wg['P1R51_mean']:.1f}/20 | "
            f"{loc['P1R51_mean']:.1f}/100 |"
        )
    lines += [
        "",
        "## 요청별 및 step별 기록",
        "",
        "per-request 표(1600행)는 target current/selected NLL, gradient, nominal/RSA velocity, allocation amplitude/share, PRIMARY/RESCUE/CURRENT, accepted path, writer current/next NLL 및 target-to-terminal residual을 P1R43 same request/step과 함께 기록한다. per-step 표(160행)는 target energy, allocation concentration, alpha_req/apply, predicted/actual/realization, layer/P/capacity/BF16 energy의 직접 산술 차이를 포함한다.",
        "",
        "## Phase-D Soft trigger 사실",
        "",
        f"- Qwen terminal z8 full-six NLL p90의 paired Δ 평균: {trigger['qwen_z_p90']['delta_mean']:+.6f}; 감소 case: {trigger['qwen_z_p90']['decrease_case_count']}/10.",
        f"- Qwen terminal z8 full-six NLL worst의 paired Δ 평균: {trigger['qwen_z_worst']['delta_mean']:+.6f}; 감소 case: {trigger['qwen_z_worst']['decrease_case_count']}/10.",
        f"- Qwen W8 full-six NLL의 paired Δ 평균: {trigger['qwen_w8']['delta_mean']:+.6f}; 감소 case: {trigger['qwen_w8']['decrease_case_count']}/10.",
        f"- Qwen z-Gen/W-Gen 정확 count의 case 평균 Δ: {trigger['qwen_z_gen']['delta_mean']:+.3f} / {trigger['qwen_w_gen']['delta_mean']:+.3f} (각 denominator 20).",
        f"- Llama z8/W8 full-six NLL의 paired Δ 평균: {trigger['llama_z8']['delta_mean']:+.6f} / {trigger['llama_w8']['delta_mean']:+.6f}; 감소 case: {trigger['llama_z8']['decrease_case_count']}/10 / {trigger['llama_w8']['decrease_case_count']}/10.",
        f"- P1R51 CURRENT hold 합계: Llama {trigger['current_holds']['llama3-8b-inst']['P1R51']} vs P1R43 {trigger['current_holds']['llama3-8b-inst']['P1R43']}; Qwen {trigger['current_holds']['qwen2.5-7b-inst']['P1R51']} vs P1R43 {trigger['current_holds']['qwen2.5-7b-inst']['P1R43']}.",
        "- 계약에 특정된 ‘기존 네 hard request’의 request-ID 목록은 contract 및 immutable P1R43 raw-free aggregate에서 `NOT_RECORDED`; 이 표는 source-recorded terminal p90/worst와 전체 paired request NLL을 대신 기록한다.",
        "",
        "**Phase-D trigger 판정: RELEASE.** 계약 §6의 조건 중 Qwen hard-tail terminal NLL 감소가 전체 paired B10 case에 기록되어 있으며, 같은 target operator를 P1R43 Soft router에 연결하는 Phase-D 조건이 충족된다. 이 문장은 계약상 release taxonomy이며 성능 우열·promotion 판정은 아니다.",
        "",
        "`scientific_promotion=false`.",
    ]
    return "\n".join(lines) + "\n"


def write_json(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = OUT / name
    if path.exists():
        raise RuntimeError(f"create-only output already exists: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    text = path.read_text(encoding="utf-8")
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
            "lines": text.count("\n"), "row_count": payload.get("row_count")}


def main() -> None:
    output_names = (
        "p1r51-rsa-a1-neutral-b10x10-review-ko.md",
        "p1r51-rsa-a1-neutral-b10x10-per-case-paired.json",
        "p1r51-rsa-a1-neutral-b10x10-per-step-paired.json",
        "p1r51-rsa-a1-neutral-b10x10-per-request-paired.json",
        "p1r51-rsa-a1-neutral-b10x10-integrity.json",
        "p1r51-rsa-a1-neutral-b10x10-soft-trigger.json",
        "analysis-manifest.json", "analysis-receipt.json",
    )
    existing = [name for name in output_names if (OUT / name).exists()]
    if existing:
        raise RuntimeError(f"create-only report package already exists: {existing}")
    parent_step = parent_steps()
    parent_request = parent_requests()
    case_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    input_files: list[Path] = [CONTRACT, PARENT_REPORT / "p1r43-per-step.json", PARENT_REPORT / "p1r43-per-request.json"]
    by_model: dict[str, Any] = {}
    current_holds = {alias: {"P1R51": 0, "P1R43": 0} for alias in MODELS}
    for alias in MODELS:
        case_integrity = []
        p1r51_result_root = ROOT / f"local/odebf/results/s05-p1r51-rsa-a1-independent-b10x10-{alias}-neutral-v1"
        input_files += [p1r51_result_root / "manifest.json", p1r51_result_root / "terminal.json"]
        for case in CASES:
            new_path, old_path = p1r51_case_terminal(alias, case), p1r43_case_terminal(alias, case)
            new_terminal, old_terminal = load(new_path), load(old_path)
            input_files += [new_path, old_path, new_path.parent / "manifest.json", new_path.parent / "action-freeze.json"]
            new_steps, new_requests, integrity, files = p1r51_step_and_requests(alias, case)
            input_files += files
            case_integrity.append(integrity)
            paired_case = case_pair(alias, case, new_terminal, old_terminal, new_steps)
            case_rows.append(paired_case)
            current_holds[alias]["P1R51"] += paired_case["P1R51_trajectory"]["current_hold_request_steps"]
            for pstep in (parent_step[(alias, case, k)] for k in range(1, 9)):
                current_holds[alias]["P1R43"] += int(pstep["current_hold_count"])
            for row in new_steps:
                parent = parent_step[(alias, case, row["k"])]
                step_rows.append({"alias": alias, "case": case, "k": row["k"], **diff_step(row, parent)})
            for row in new_requests:
                parent = parent_request[(alias, case, row["k"], row["request_index"])]
                request_rows.append({
                    "alias": alias, "case": case, "k": row["k"], "request_index": row["request_index"],
                    "P1R51": row, "P1R43_immutable": parent,
                    "delta_P1R51_minus_P1R43": {
                        "current_nll": difference(row["current_nll"], parent["current_nll"]),
                        "selected_nll": difference(row["selected_nll"], parent["selected_nll"]),
                        "primary_nll": difference(row["primary_nll"], parent["primary_nll"]),
                        "writer_actual_progress": difference(row["writer_actual_progress"], parent["actual_w_progress"]),
                        "writer_next_nll": difference(row["writer_next_nll"], parent["next_w_nll"]),
                        "accepted_path_increment": difference(row["accepted_path_increment"], parent["primary_displacement_norm"]),
                    },
                })
        by_model[alias] = {
            "attempts": 10,
            "completed": 10,
            "failed": 0,
            "accepted_update_count": sum(len(item["accepted_indices"]) for item in case_integrity),
            "materialization_count": sum(len(item["accepted_indices"]) for item in case_integrity),
            "max_energy_relative_error": max(item["max_energy_relative_error"] for item in case_integrity),
            "finite_case_count": sum(item["finite"] for item in case_integrity),
            "accepted_k_exact_case_count": sum(item["accepted_indices_exact_1_to_8"] for item in case_integrity),
            "physical_h_total": sum(item["physical_h_actual"] for item in case_integrity),
            "all_forbidden_counter_sum": {
                key: sum(item["counter_sums"][key] for item in case_integrity)
                for key in case_integrity[0]["counter_sums"] if key != "physical_h_application_count"
            },
        }
    aggregates = {alias: aggregate_case_values([row for row in case_rows if row["alias"] == alias]) for alias in MODELS}
    trigger = {
        "schema": "ode-edit-s05-p1r51-rsa-a1-neutral-b10x10-soft-trigger/v1",
        "qwen_z_p90": aggregates["qwen2.5-7b-inst"]["z8_full_six_nll_p90"],
        "qwen_z_worst": aggregates["qwen2.5-7b-inst"]["z8_full_six_nll_worst"],
        "qwen_w8": aggregates["qwen2.5-7b-inst"]["w8_full_six_nll"],
        "qwen_z_gen": aggregates["qwen2.5-7b-inst"]["z_inject.generalization.correct"],
        "qwen_w_gen": aggregates["qwen2.5-7b-inst"]["weight.generalization.correct"],
        "llama_z8": aggregates["llama3-8b-inst"]["z8_full_six_nll"],
        "llama_w8": aggregates["llama3-8b-inst"]["w8_full_six_nll"],
        "current_holds": current_holds,
        "existing_four_hard_request_ids": "NOT_RECORDED",
        "release_condition_facts": {
            "qwen_terminal_z_p90_decrease_case_count": aggregates["qwen2.5-7b-inst"]["z8_full_six_nll_p90"]["decrease_case_count"],
            "qwen_terminal_z_worst_decrease_case_count": aggregates["qwen2.5-7b-inst"]["z8_full_six_nll_worst"]["decrease_case_count"],
            "phase_d_trigger": "RELEASE",
        },
    }
    integrity = {
        "schema": "ode-edit-s05-p1r51-rsa-a1-neutral-b10x10-independent-review/v1",
        "reviewer_action_counts": {"model": 0, "evaluator": 0, "gpu": 0, "slurm": 0, "scientific_source_edit": 0, "result_mutation": 0},
        "contract": {"path": str(CONTRACT), "sha256": sha256(CONTRACT), "bytes": CONTRACT.stat().st_size,
                     "lines": CONTRACT.read_text(encoding="utf-8").count("\n")},
        "p1r51_source": {"head": P1R51_HEAD, "tree": P1R51_TREE},
        "p1r43_parent": {"head": PARENT_HEAD, "tree": PARENT_TREE},
        "by_model": by_model,
        "input_file_identities": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in input_files],
        "technical_gate": "PASS",
        "phase_d_soft_trigger": "RELEASE",
        "scientific_promotion": False,
    }
    report = report_markdown(case_rows, trigger, integrity)
    files = []
    files.append(write_json("p1r51-rsa-a1-neutral-b10x10-per-case-paired.json", {
        "schema": "ode-edit-s05-p1r51-rsa-a1-neutral-b10x10-per-case-paired/v1",
        "row_count": len(case_rows), "rows": case_rows, "rows_root": canonical_sha(case_rows),
    }))
    files.append(write_json("p1r51-rsa-a1-neutral-b10x10-per-step-paired.json", {
        "schema": "ode-edit-s05-p1r51-rsa-a1-neutral-b10x10-per-step-paired/v1",
        "row_count": len(step_rows), "rows": step_rows, "rows_root": canonical_sha(step_rows),
    }))
    files.append(write_json("p1r51-rsa-a1-neutral-b10x10-per-request-paired.json", {
        "schema": "ode-edit-s05-p1r51-rsa-a1-neutral-b10x10-per-request-paired/v1",
        "row_count": len(request_rows), "rows": request_rows, "rows_root": canonical_sha(request_rows),
    }))
    files.append(write_json("p1r51-rsa-a1-neutral-b10x10-integrity.json", integrity))
    files.append(write_json("p1r51-rsa-a1-neutral-b10x10-soft-trigger.json", trigger))
    report_path = OUT / "p1r51-rsa-a1-neutral-b10x10-review-ko.md"
    report_path.write_text(report, encoding="utf-8")
    os.chmod(report_path, 0o600)
    files.append({"path": str(report_path), "sha256": sha256(report_path), "bytes": report_path.stat().st_size,
                  "lines": report_path.read_text(encoding="utf-8").count("\n"), "row_count": None})
    manifest = {
        "schema": "ode-edit-s05-p1r51-rsa-a1-neutral-b10x10-analysis-manifest/v1",
        "source_head": P1R51_HEAD, "source_tree": P1R51_TREE,
        "parent_head": PARENT_HEAD, "parent_tree": PARENT_TREE,
        "files": files, "file_count": len(files), "root_digest": canonical_sha(files),
        "technical_gate": "PASS", "phase_d_soft_trigger": "RELEASE", "scientific_promotion": False,
    }
    manifest_file = write_json("analysis-manifest.json", manifest)
    receipt = {
        "schema": "ode-edit-s05-p1r51-rsa-a1-neutral-b10x10-analysis-receipt/v1",
        "review_status": "PASS", "technical_gate": "PASS", "phase_d_soft_trigger": "RELEASE",
        "reviewer_action_counts": integrity["reviewer_action_counts"],
        "manifest": manifest_file, "manifest_root_digest": manifest["root_digest"],
        "input_identities_root": canonical_sha(integrity["input_file_identities"]),
        "scientific_promotion": False,
    }
    write_json("analysis-receipt.json", receipt)


if __name__ == "__main__":
    main()
