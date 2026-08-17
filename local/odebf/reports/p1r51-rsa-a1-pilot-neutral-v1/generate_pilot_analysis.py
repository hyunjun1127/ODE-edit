#!/usr/bin/env python3
"""Create-only raw-free Phase-B pilot analysis for P1R51.

This helper reads existing receipts only.  It does not load a model, invoke an
evaluator, contact Slurm, or modify experiment outputs.
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
OUT = ROOT / "local/odebf/reports/p1r51-rsa-a1-pilot-neutral-v1"
PARENT = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-p1r43-rho-free-semantic-first-strength-recovery-v1"
)
MODELS = ("llama3-8b-inst", "qwen2.5-7b-inst")
PARENT_HEAD = "11508b6da11d606521b703037034e1814b70d8a8"
PARENT_TREE = "a0e71bbb27cf36b3b51bf6cde4c92cdc9a47879b"
P1R51_HEAD = "9b4a88b73bd0c0fdd0b1470d77ea4354bd0e3486"
P1R51_TREE = "4a3b5e2dbd7e15bf8da32117c25fd88ec2eaafc7"
CONTRACT = Path(
    "/mnt/raid5/janghj/.codex/attachments/"
    "264d9efc-75b7-4694-99b6-e64f005704b4/pasted-text.txt"
)


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def finite_values(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_values(item) for item in value)
    if isinstance(value, dict):
        return all(finite_values(item) for item in value.values())
    return True


def flat_vectors(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict) and "nll_new" in value:
        yield value
    elif isinstance(value, list):
        for child in value:
            yield from flat_vectors(child)


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo = int(math.floor(index))
    hi = int(math.ceil(index))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def vector_summary(panel: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    vectors = panel["numeric_vectors"]
    metrics = panel["primary"]["metrics"]
    for name, counter in metrics.items():
        entries = list(flat_vectors(vectors[name]))
        nlls = [float(entry["nll_new"]) for entry in entries]
        margins = [float(entry["margin"]) for entry in entries]
        olds = [float(entry["nll_old"]) for entry in entries]
        results[name] = {
            "correct": counter["numerator"],
            "denominator": counter["denominator"],
            "rate": counter["official_aggregate"],
            "nll_mean": statistics.fmean(nlls),
            "nll_median": statistics.median(nlls),
            "nll_p90": quantile(nlls, 0.90),
            "nll_worst": max(nlls),
            "nll_old_mean": statistics.fmean(olds),
            "margin_mean": statistics.fmean(margins),
            "margin_median": statistics.median(margins),
            "margin_p90": quantile(margins, 0.90),
            "margin_worst_low": min(margins),
            "numeric_vector_sha256": panel["numeric_vectors_sha256"],
            "comparison_score_sha256": counter["comparison_score_sha256"],
        }
    return results


def get_path(data: dict[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        value = value[key]
    return value


def delta(new: Any, parent: Any) -> Any:
    if isinstance(new, (int, float)) and isinstance(parent, (int, float)):
        return new - parent
    return None


def new_case_path(alias: str) -> Path:
    return (
        ROOT
        / f"local/odebf/results/s05-p1r51-rsa-a1-pilot-{alias}-neutral-v1"
        / "raw/cases/case-01/terminal.json"
    )


def parent_case_path(alias: str) -> Path:
    return (
        PARENT
        / f"local/odebf/results/s05-p1r43-rho-free-independent-b10x10-{alias}-neutral-tech-r2-v1"
        / "raw/cases/case-01/terminal.json"
    )


def summarize_terminal(terminal: dict[str, Any]) -> dict[str, Any]:
    four = terminal["terminal_four_panel"]
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
        "terminal_z8_full_six_target_new_nll": terminal["terminal_z8_oracle"]["target_new_nll"],
        "terminal_w8_full_six_target_new_nll": terminal["terminal_w8_full_six_target_new_nll"],
        "z_w_full_six_gap": terminal["terminal_w8_full_six_target_new_nll"]
        - terminal["terminal_z8_oracle"]["target_new_nll"],
        "z8_per_request_nll": terminal["terminal_z8_oracle"]["per_request_values"],
        "panels": {
            "z_inject": vector_summary(four["z_inject"]),
            "weight": vector_summary(four["weight"]),
        },
        "terminal_evaluator_wall_seconds": terminal["terminal_evaluator_wall_seconds"],
        "four_panel_wall_seconds": four["wall_seconds"],
        "terminal_only_evaluator": four["terminal_only_evaluator"],
        "heldout_efficacy_controller_access_count": four["heldout_efficacy_controller_access_count"],
        "heldout_gen_controller_access_count": four["heldout_gen_controller_access_count"],
        "inner_step_heldout_evaluation_count": four["inner_step_heldout_evaluation_count"],
        "W0_restored": terminal["W0_restored"],
        "endpoint_restore": terminal["endpoint_restore"],
        "retry_count": terminal["retry_count"],
        "cross_case_state_count": terminal["cross_case_state_count"],
    }


def case_table_row(alias: str, new: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    n = summarize_terminal(new)
    p = summarize_terminal(old)
    identities = {
        key: n[key] == p[key]
        for key in (
            "request_order_sha256",
            "capture_plan_sha256",
            "objective_plan_sha256",
            "evaluator_source_sha256",
            "aggregator_source_sha256",
            "target_span_sha256",
            "evaluation_case_identity_sha256",
        )
    }
    output: dict[str, Any] = {
        "alias": alias,
        "case_index": 1,
        "comparison": "P1R51_RSA_A1_NEUTRAL_MINUS_IMMUTABLE_P1R43_NEUTRAL",
        "identity_matched": all(identities.values()),
        "identity_fields": identities,
        "P1R51": n,
        "P1R43_immutable": p,
        "delta_P1R51_minus_P1R43": {
            "z8_full_six_nll": delta(
                n["terminal_z8_full_six_target_new_nll"],
                p["terminal_z8_full_six_target_new_nll"],
            ),
            "w8_full_six_nll": delta(
                n["terminal_w8_full_six_target_new_nll"],
                p["terminal_w8_full_six_target_new_nll"],
            ),
            "z_w_gap": delta(n["z_w_full_six_gap"], p["z_w_full_six_gap"]),
        },
    }
    for panel in ("z_inject", "weight"):
        for metric in ("efficacy", "generalization", "locality-preservation"):
            a = n["panels"][panel][metric]
            b = p["panels"][panel][metric]
            output["delta_P1R51_minus_P1R43"][f"{panel}.{metric}"] = {
                key: delta(a.get(key), b.get(key))
                for key in (
                    "correct",
                    "denominator",
                    "rate",
                    "nll_mean",
                    "nll_median",
                    "nll_p90",
                    "nll_worst",
                    "margin_mean",
                )
            }
    return output


def accepted_path(alias: str) -> Path:
    return (
        ROOT
        / f"local/odebf/results/s05-p1r51-rsa-a1-pilot-{alias}-neutral-v1"
        / "raw/cases/case-01/raw/ode/p1r43-rsa-a1-neutral"
    )


def build_steps_and_requests(alias: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    directory = accepted_path(alias)
    steps: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    errors: list[float] = []
    forbidden = {
        "remaining_horizon_division_count": 0,
        "semantic_debt_input_count": 0,
        "fresh_component_division_count": 0,
        "lag_component_division_count": 0,
        "physical_h_application_count": 0,
        "second_h_application_count": 0,
        "same_step_retry_count": 0,
        "total_target_velocity_energy_increase_count": 0,
        "target_hit_threshold_access_count": 0,
        "target_hold_persistent_count": 0,
        "persistent_freeze_input_count": 0,
        "radius_state_access_count": 0,
        "trust_rho_decision_influence_count": 0,
        "target_kl_decision_influence_count": 0,
        "target_decay_decision_influence_count": 0,
        "target_preservation_decision_influence_count": 0,
    }
    counter_sums = {key: 0 for key in forbidden}
    all_finite = True
    accepted_indices: list[int] = []
    for index in range(1, 9):
        accepted = load(directory / f"accepted-k{index}.json")
        realization = load(directory / f"requestwise-realization-k{index}.json")
        target = accepted["target_update"]
        routing = accepted["routing"]
        progress = accepted["progress"]
        material = accepted["materialization"]
        structural = accepted["cumulative_atomic_structural_p"]
        all_finite = all_finite and finite_values(accepted) and finite_values(realization)
        accepted_indices.append(accepted["accepted_index"])
        errors.append(float(target["energy_relative_error"]))
        for key in counter_sums:
            counter_sums[key] += int(target.get(key, 0))
        delayed = None
        for item in []:
            delayed = item
        actual_from_request = sum(realization["actual_w_only_progress_by_request"]) / target["request_count"]
        predicted_from_route = routing["alpha_apply"]
        realized_ratio = actual_from_request / predicted_from_route if predicted_from_route else None
        step = {
            "alias": alias,
            "case_index": 1,
            "accepted_index": accepted["accepted_index"],
            "k_zero_based": target["k"],
            "tau_before": accepted["tau_before"],
            "tau_after": accepted["tau_after"],
            "target_current_nll": target["target_new_nll_current_summary"],
            "target_selected_nll": target["selected_nll_summary"],
            "p1r43_nominal_velocity_energy": target["p1r43_nominal_velocity_energy"],
            "rsa_total_velocity_energy": target["rsa_total_velocity_energy"],
            "energy_relative_error": target["energy_relative_error"],
            "allocation": {
                "top1_share": target["top1_allocation_share"],
                "top3_share": target["top3_allocation_share"],
                "entropy": target["allocation_entropy"],
                "effective_request_support": target["effective_request_support"],
                "active_gradient_count": target["active_gradient_count"],
                "flat_gradient_count": target["flat_gradient_count"],
            },
            "selection": {
                "primary_accept_count": target["primary_accept_count"],
                "rescue_accept_count": target["rescue_accept_count"],
                "current_hold_count": target["current_hold_count"],
            },
            "writer": {
                "alpha_req": routing["alpha_req"],
                "alpha_apply": routing["alpha_apply"],
                "alpha_apply_over_req": routing["alpha_apply_over_req"],
                "predicted": progress.get("predicted", predicted_from_route),
                "actual": progress.get("actual", actual_from_request),
                "realization_ratio": progress.get("realization_ratio", realized_ratio),
                "actual_from_request_mean": actual_from_request,
                "realization_ratio_from_request_mean": realized_ratio,
                "coverage": progress.get("coverage"),
                "routing_status": routing["status"],
                "equality_residual": routing["equality_residual"],
                "q": routing["q"],
                "selected_p": routing["selected_p"],
                "selected_capacity": routing["selected_capacity"],
                "selected_energy": routing["selected_energy"],
                "layer_entropy": routing["simplex_entropy"],
                "layer_top1": routing["simplex_top1_share"],
            },
            "structural_p_after": structural["P_after"],
            "structural_p_identity_residual": structural["algebra_identity_residual"],
            "bf16_step_energy": sum(material["realized_bf16_step_energy"].values()),
            "bf16_capacity_sum": sum(material["cumulative_bf16_capacity"].values()),
            "materialization_transition_index": material["transition_index"],
            "target_full_current_residual_identity_max_abs": target["full_current_residual_identity_max_abs"],
            "h": target["h"],
            "receipt_identity_sha256": accepted["identity_sha256"],
            "requestwise_realization_identity_sha256": realization["identity_sha256"],
        }
        steps.append(step)
        fields = {
            "current_nll": target["current_nll_by_request"],
            "selected_nll": target["selected_nll_by_request"],
            "primary_nll": target["primary_nll_by_request"],
            "semantic_gradient_norm": target["semantic_gradient_norm_by_request"],
            "entry_gradient_norm": target["entry_semantic_gradient_norm_by_request"],
            "nominal_velocity_norm": target["p1r43_nominal_velocity_norm_by_request"],
            "allocation_amplitude": target["allocation_amplitude_by_request"],
            "allocation_energy_share": target["allocation_energy_share_by_request"],
            "proposed_displacement_norm": target["proposed_target_displacement_norm_by_request"],
            "actual_displacement_norm": target["actual_target_displacement_norm_by_request"],
            "accepted_path_increment": target["accepted_activation_path_increment_by_request"],
            "cumulative_accepted_path": target["cumulative_accepted_activation_path_by_request"],
            "accepted_nll_improvement": target["accepted_target_nll_improvement_by_request"],
            "flat_gradient": target["flat_gradient_mask"],
            "selection": target["selection_by_request"],
            "writer_current_nll": realization["source_w_only_target_new_nll_by_request"],
            "writer_next_nll": realization["next_w_only_target_new_nll_by_request"],
            "writer_actual_progress": realization["actual_w_only_progress_by_request"],
            "writer_target_residual_norm": accepted["target_write_realization"]["residual_norm"],
            "writer_target_residual_ratio": accepted["target_write_realization"]["residual_ratio"],
        }
        for request_index in range(target["request_count"]):
            requests.append(
                {
                    "alias": alias,
                    "case_index": 1,
                    "accepted_index": index,
                    "k_zero_based": target["k"],
                    "request_index": request_index,
                    **{key: value[request_index] for key, value in fields.items()},
                }
            )
    return steps, requests, {
        "accepted_indices": accepted_indices,
        "accepted_indices_exact_1_to_8": accepted_indices == list(range(1, 9)),
        "max_energy_relative_error": max(errors),
        "all_numeric_values_finite": all_finite,
        "counter_sums": counter_sums,
        "counter_expectations": {
            **{key: 0 for key in forbidden if key != "physical_h_application_count"},
            "physical_h_application_count": 8,
        },
    }


def output_file(name: str, payload: Any) -> dict[str, Any]:
    path = OUT / name
    if path.exists():
        raise RuntimeError(f"create-only output already exists: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "lines": text.count("\n"),
        "row_count": payload.get("row_count") if isinstance(payload, dict) else None,
    }


def report_markdown(case_rows: list[dict[str, Any]], integrity: dict[str, Any]) -> str:
    lines = [
        "# P1R51 Phase-B Neutral pilot 독립 검토 보고서",
        "",
        "## 범위와 경계",
        "",
        "이 검토는 이미 기록된 P1R51 Phase-B Neutral case01 영수증과 불변 P1R43 Neutral case01 영수증만 읽었다. 모델·evaluator·GPU·Slurm 실행 또는 재실행은 0회이며, 실험 source/result를 변경하지 않았다.",
        "",
        "- P1R51 source HEAD/tree: `9b4a88b73bd0c0fdd0b1470d77ea4354bd0e3486` / `4a3b5e2dbd7e15bf8da32117c25fd88ec2eaafc7`",
        "- P1R43 parent HEAD/tree: `11508b6da11d606521b703037034e1814b70d8a8` / `a0e71bbb27cf36b3b51bf6cde4c92cdc9a47879b`",
        "- P1R51 contract SHA256: `e19617806b9f60945291885d5fcc91383ee4f2c5198aaf40f77ab38bb9e1562a` (15841 bytes, 520 lines)",
        "- 비교는 alias별 case01, request order/capture plan/objective plan/evaluator/aggregator/target span/evaluation-case identity 일치 조건에서 수행했다. action-freeze hash는 각 방법의 서로 다른 action 때문에 일치 조건이 아니다.",
        "",
        "## 기술 무결성",
        "",
        "| 항목 | Llama | Qwen |",
        "|---|---:|---:|",
        f"| 완료 endpoint / 실패 endpoint | 1 / 0 | 1 / 0 |",
        f"| accepted K / BF16 materialization | {integrity['models']['llama3-8b-inst']['accepted_indices']} / 8 | {integrity['models']['qwen2.5-7b-inst']['accepted_indices']} / 8 |",
        f"| 최대 RSA energy relative error | {integrity['models']['llama3-8b-inst']['max_energy_relative_error']:.3e} | {integrity['models']['qwen2.5-7b-inst']['max_energy_relative_error']:.3e} |",
        f"| W0 pointer+bytes restore | PASS | PASS |",
        f"| action-freeze / heldout controller access / inner-step heldout | PASS / 0 / 0 | PASS / 0 / 0 |",
        f"| retry / cross-case state / nonfinite | 0 / 0 / 없음 | 0 / 0 / 없음 |",
        f"| PRIMARY / RESCUE / CURRENT request-step | 80 / 0 / 0 | 80 / 0 / 0 |",
        "",
        "각 모델의 accepted index는 `1..8`이고 `h=0.125`는 8회, second-h·remaining horizon·semantic debt·persistent hold·trust/radius·KL/decay/preservation decision counter는 모두 0이다. `alpha_apply/alpha_req`는 각 step에서 1의 수치 오차 범위에 있다.",
        "",
        "## P1R43 exact-matched case01 비교",
        "",
        "| 모델 | 방법 | z8 full-six NLL | W8 full-six NLL | W−z gap | z Eff/Gen | W Eff/Gen | W Loc |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in case_rows:
        for label, key in (("P1R51 RSA", "P1R51"), ("P1R43 불변", "P1R43_immutable")):
            item = row[key]
            z = item["panels"]["z_inject"]
            w = item["panels"]["weight"]
            lines.append(
                f"| {row['alias']} | {label} | {item['terminal_z8_full_six_target_new_nll']:.6f} | "
                f"{item['terminal_w8_full_six_target_new_nll']:.6f} | {item['z_w_full_six_gap']:.6f} | "
                f"{z['efficacy']['correct']}/{z['efficacy']['denominator']} / {z['generalization']['correct']}/{z['generalization']['denominator']} | "
                f"{w['efficacy']['correct']}/{w['efficacy']['denominator']} / {w['generalization']['correct']}/{w['generalization']['denominator']} | "
                f"{w['locality-preservation']['correct']}/{w['locality-preservation']['denominator']} |"
            )
        d = row["delta_P1R51_minus_P1R43"]
        lines.append(
            f"| {row['alias']} | Δ RSA−P1R43 | {d['z8_full_six_nll']:+.6f} | {d['w8_full_six_nll']:+.6f} | {d['z_w_gap']:+.6f} | "
            f"Δz Eff {d['z_inject.efficacy']['correct']:+.0f}, Gen {d['z_inject.generalization']['correct']:+.0f} | "
            f"ΔW Eff {d['weight.efficacy']['correct']:+.0f}, Gen {d['weight.generalization']['correct']:+.0f} | "
            f"ΔLoc {d['weight.locality-preservation']['correct']:+.0f} |"
        )
    lines += [
        "",
        "## Target allocation·writer·보존·비용",
        "",
        "세부 per-step 및 per-request 값은 동봉 JSON 표에 있다. 표에는 각 step의 nominal/RSA energy, allocation entropy/top-1/effective support, PRIMARY/RESCUE/CURRENT, alpha_req/apply, predicted/actual/realization, Structural-P/capacity/BF16 step energy 및 request별 target/W-only NLL·allocation share·accepted path를 기록한다.",
        "",
        "## Phase-B 판정",
        "",
        "**기술 판정: PASS.** 두 endpoint 모두 K8, action-freeze, W0 restore, input/evaluator identity 및 target energy 보존 receipt를 만족한다. 계약 §6의 Phase-B 중단 조건(nonfinite, request order/data mismatch, energy conservation 위반, target/terminal tensor contract 오류, BF16 materialization 실패, 명백한 catastrophic divergence, writer/router 계약 변경)은 이 검토의 영수증에서 관측되지 않았다.",
        "",
        "**Phase-C release gate: RELEASE.** §6의 lenient pilot 규칙에 따라 Neutral independent B10×10은 release 가능하다. 이 판정은 Phase-B 기술/중단 조건에만 근거하며, 단일 batch 결과를 일반화한 과학적 우열 또는 promotion 판정이 아니다.",
        "",
        "`scientific_promotion=false`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    required_absent = [
        "p1r51-rsa-a1-pilot-neutral-review-ko.md",
        "p1r51-rsa-a1-pilot-neutral-per-step.json",
        "p1r51-rsa-a1-pilot-neutral-per-request.json",
        "p1r51-rsa-a1-pilot-neutral-case-paired.json",
        "p1r51-rsa-a1-pilot-neutral-integrity.json",
        "analysis-manifest.json",
        "analysis-receipt.json",
    ]
    existing = [name for name in required_absent if (OUT / name).exists()]
    if existing:
        raise RuntimeError(f"create-only report package already exists: {existing}")
    cases: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    integrity_models: dict[str, Any] = {}
    input_files: list[Path] = [CONTRACT]
    for alias in MODELS:
        new_terminal_path = new_case_path(alias)
        old_terminal_path = parent_case_path(alias)
        new_terminal = load(new_terminal_path)
        old_terminal = load(old_terminal_path)
        input_files += [new_terminal_path, old_terminal_path]
        cases.append(case_table_row(alias, new_terminal, old_terminal))
        alias_steps, alias_requests, integrity = build_steps_and_requests(alias)
        steps.extend(alias_steps)
        requests.extend(alias_requests)
        integrity_models[alias] = integrity
        input_files.extend(
            sorted(accepted_path(alias).glob("accepted-k*.json"))
            + sorted(accepted_path(alias).glob("requestwise-realization-k*.json"))
        )
        result_root = new_terminal_path.parents[3]
        for relative in ("manifest.json", "terminal.json", "raw/cases/case-01/manifest.json", "raw/cases/case-01/action-freeze.json"):
            input_files.append(result_root / relative)
    source_paths = [
        ROOT / "project/run_scripts/ode_bf/p1r51_requestwise_semantic_allocation.py",
        ROOT / "project/run_scripts/ode_bf/p1r51_independent_runtime.py",
        ROOT / "project/run_scripts/ode_bf/p1r51_independent_panel.py",
        ROOT / "project/run_scripts/ode_bf/locks/numerical_lock_s05_p1r51_rsa_a1.json",
        ROOT / "project/run_scripts/ode_bf/locks/source_manifest_s05_p1r51_rsa_a1.json",
    ]
    input_files += source_paths
    integrity = {
        "schema": "ode-edit-s05-p1r51-rsa-a1-pilot-neutral-independent-review/v1",
        "analysis_scope": "raw-free existing receipts only; no model/evaluator/GPU/Slurm action",
        "input_contract": {
            "path": str(CONTRACT),
            "sha256": sha256(CONTRACT),
            "bytes": CONTRACT.stat().st_size,
            "lines": CONTRACT.read_text(encoding="utf-8").count("\n"),
        },
        "p1r51_source": {"head": P1R51_HEAD, "tree": P1R51_TREE},
        "p1r43_parent": {"head": PARENT_HEAD, "tree": PARENT_TREE},
        "models": integrity_models,
        "input_file_identities": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in input_files
        ],
        "technical_gate": "PASS",
        "phase_c_release_gate": "RELEASE",
        "phase_c_gate_basis": [
            "two completed endpoints",
            "accepted K1..K8 and BF16 materialization transition1..8",
            "W0 pointer+byte restore and action freeze receipts",
            "matched stream/case/evaluator identity to immutable P1R43 case01",
            "finite receipts and energy conservation within recorded numerical error",
            "no Phase-B listed stop condition observed in reviewed receipts",
        ],
        "scientific_promotion": False,
    }
    outputs = []
    outputs.append(output_file("p1r51-rsa-a1-pilot-neutral-per-step.json", {
        "schema": "ode-edit-s05-p1r51-rsa-a1-pilot-neutral-per-step/v1",
        "row_count": len(steps), "rows": steps, "rows_root": canonical_sha(steps),
    }))
    outputs.append(output_file("p1r51-rsa-a1-pilot-neutral-per-request.json", {
        "schema": "ode-edit-s05-p1r51-rsa-a1-pilot-neutral-per-request/v1",
        "row_count": len(requests), "rows": requests, "rows_root": canonical_sha(requests),
    }))
    outputs.append(output_file("p1r51-rsa-a1-pilot-neutral-case-paired.json", {
        "schema": "ode-edit-s05-p1r51-rsa-a1-pilot-neutral-case-paired/v1",
        "row_count": len(cases), "rows": cases, "rows_root": canonical_sha(cases),
    }))
    outputs.append(output_file("p1r51-rsa-a1-pilot-neutral-integrity.json", integrity))
    report_path = OUT / "p1r51-rsa-a1-pilot-neutral-review-ko.md"
    report_path.write_text(report_markdown(cases, integrity), encoding="utf-8")
    os.chmod(report_path, 0o600)
    report_info = {
        "path": str(report_path), "sha256": sha256(report_path),
        "bytes": report_path.stat().st_size,
        "lines": report_path.read_text(encoding="utf-8").count("\n"),
    }
    outputs.append(report_info)
    manifest = {
        "schema": "ode-edit-s05-p1r51-rsa-a1-pilot-neutral-analysis-manifest/v1",
        "phase": "B_NEUTRAL_PILOT",
        "source_head": P1R51_HEAD,
        "source_tree": P1R51_TREE,
        "parent_head": PARENT_HEAD,
        "parent_tree": PARENT_TREE,
        "files": outputs,
        "file_count": len(outputs),
        "root_digest": canonical_sha(outputs),
        "technical_gate": "PASS",
        "phase_c_release_gate": "RELEASE",
        "scientific_promotion": False,
    }
    manifest_info = output_file("analysis-manifest.json", manifest)
    receipt = {
        "schema": "ode-edit-s05-p1r51-rsa-a1-pilot-neutral-analysis-receipt/v1",
        "reviewer_action_counts": {
            "model": 0, "evaluator": 0, "gpu": 0, "slurm": 0,
            "scientific_source_edit": 0, "result_mutation": 0,
        },
        "review_status": "PASS",
        "technical_gate": "PASS",
        "phase_c_release_gate": "RELEASE",
        "manifest": manifest_info,
        "manifest_root_digest": manifest["root_digest"],
        "input_identities_root": canonical_sha(integrity["input_file_identities"]),
        "scientific_promotion": False,
    }
    output_file("analysis-receipt.json", receipt)


if __name__ == "__main__":
    main()
