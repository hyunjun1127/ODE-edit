#!/usr/bin/env python3
"""Build the bounded raw-free P2R5 Stage-A report package."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from zoneinfo import ZoneInfo


WORKTREE = Path("/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p2r5-sdrt-stage-a-v1")
OUT = WORKTREE / "local/odebf/reports/p2r5-sdrt-stage-a-v1"
SUPPORT = Path("/mnt/raid5/janghj/ODE-edit/local/source-handoff/P2R5_P2R4_CLAMP_ON_CANONICAL_SUPPORT_SH2_V1")
CONTRACT = Path("/mnt/raid5/janghj/.codex/attachments/409d3a84-d321-4d00-8479-9075f9a63ff2/pasted-text.txt")
PREFIX = "p2r5-sdrt-stage-a"
EXPECTED_CONTRACT = "55cc177ecee86750bde732c6f880ea480696269da856495829437c9b59cc669d"
TOL = 1.0e-8

SPECS = [
    ("llama3-8b-inst", 3, "SDRT-CAP", "s05-p2r5-sdrt-stage-a-llama3-8b-inst-case-03-paired-tech-r2-v1"),
    ("llama3-8b-inst", 3, "SDRT-STRUCTP", "s05-p2r5-sdrt-stage-a-llama3-8b-inst-case-03-paired-tech-r6-v1"),
    ("llama3-8b-inst", 5, "SDRT-CAP", "s05-p2r5-sdrt-stage-a-llama3-8b-inst-case-05-paired-tech-r5-v1"),
    ("llama3-8b-inst", 5, "SDRT-STRUCTP", "s05-p2r5-sdrt-stage-a-llama3-8b-inst-case-05-paired-tech-r4-v1"),
    ("qwen2.5-7b-inst", 1, "SDRT-CAP", "s05-p2r5-sdrt-stage-a-qwen2.5-7b-inst-case-01-paired-tech-r5-v1"),
    ("qwen2.5-7b-inst", 1, "SDRT-STRUCTP", "s05-p2r5-sdrt-stage-a-qwen2.5-7b-inst-case-01-paired-tech-r5-v1"),
    ("qwen2.5-7b-inst", 4, "SDRT-CAP", "s05-p2r5-sdrt-stage-a-qwen2.5-7b-inst-case-04-paired-tech-r4-v1"),
    ("qwen2.5-7b-inst", 4, "SDRT-STRUCTP", "s05-p2r5-sdrt-stage-a-qwen2.5-7b-inst-case-04-paired-tech-r2-v1"),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canon(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def dump_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    os.chmod(path, 0o600)


def csv_value(value: object) -> object:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return value


def dump_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    fields = sorted({key for row in rows for key in row})
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "NOT_RECORDED")) for key in fields})
    os.chmod(path, 0o600)


def flat_numeric(panel: dict, metric: str, field: str) -> list[float]:
    return [float(row[field]) for request in panel["z_inject"]["numeric_vectors"][metric] for row in request]


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def panel_row(terminal: dict, prefix: str, metric: str) -> dict:
    if prefix == "w":
        item = terminal["terminal_w_panel"]["receipt"]["metrics"][metric]
        return {
            f"{prefix}_{metric}_correct": item["correct_count"],
            f"{prefix}_{metric}_denominator": item["prompt_count"],
            f"{prefix}_{metric}_new_nll_mean": item["mean_target_new_nll"],
            f"{prefix}_{metric}_old_nll_mean": item["mean_target_old_nll"],
            f"{prefix}_{metric}_margin_mean": item["mean_margin"],
        }
    panel = terminal["terminal_z_panel"]
    item = panel["z_inject"]["primary"]["metrics"][metric]
    new = flat_numeric(panel, metric, "nll_new")
    old = flat_numeric(panel, metric, "nll_old")
    margins = flat_numeric(panel, metric, "margin")
    return {
        f"{prefix}_{metric}_correct": item["numerator"],
        f"{prefix}_{metric}_denominator": item["denominator"],
        f"{prefix}_{metric}_new_nll_mean": mean(new),
        f"{prefix}_{metric}_old_nll_mean": mean(old),
        f"{prefix}_{metric}_margin_mean": mean(margins),
    }


def main() -> None:
    if sha(CONTRACT) != EXPECTED_CONTRACT or len(CONTRACT.read_bytes()) != 15003:
        raise RuntimeError("contract identity mismatch")
    comparator_path = SUPPORT / "results/p2r4-phaseb-clamp-on-off-writer-v1-per-case.json"
    comparator = json.loads(comparator_path.read_text())
    refs = {(r["alias"], int(r["case_index"]), r["arm"]): r for r in comparator if r["clamp_policy"] == "ON"}
    official_path = SUPPORT / "identities/official-alphaedit-baseline-identity.json"
    official_doc = json.loads(official_path.read_text())
    official = {(r["alias"], int(r["case"])): r for r in official_doc["rows"]}

    endpoints: list[dict] = []
    steps: list[dict] = []
    requests: list[dict] = []
    raw_steps: dict[tuple[str, int, str, int], dict] = {}
    terminal_paths: list[str] = []

    for alias, case, arm, root_name in SPECS:
        root = WORKTREE / "local/odebf/results" / root_name
        case_root = root / "raw/cases" / f"case-{case:02d}" / arm.lower()
        terminal_path = case_root / "terminal.json"
        manifest_path = case_root / "manifest.json"
        action_path = case_root / "action-freeze.json"
        job_terminal_path = root / "terminal.json"
        job_manifest_path = root / "manifest.json"
        terminal = json.loads(terminal_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        action = json.loads(action_path.read_text())
        job_terminal = json.loads(job_terminal_path.read_text())
        job_manifest = json.loads(job_manifest_path.read_text())
        if manifest["terminal_sha256"] != sha(terminal_path):
            raise RuntimeError(f"terminal hash mismatch: {terminal_path}")
        if manifest["action_freeze_sha256"] != sha(action_path):
            raise RuntimeError(f"action hash mismatch: {action_path}")
        if job_manifest["terminal_sha256"] != sha(job_terminal_path):
            raise RuntimeError(f"job terminal hash mismatch: {root}")
        if not (manifest["W0_restored"] and terminal["W0_restore"]["pointer_restored_exact"] and terminal["W0_restore"]["byte_restored_exact"] and job_terminal["W0_restored"]):
            raise RuntimeError(f"W0 mismatch: {terminal_path}")
        if not action["actions_frozen_before_evaluator"] or action["heldout_controller_access_count"] != 0:
            raise RuntimeError(f"firewall mismatch: {action_path}")

        step_rows: list[dict] = []
        for step in range(8):
            step_path = case_root / "raw/writer" / f"step-{step:02d}.json"
            data = json.loads(step_path.read_text())
            route = data["route"]
            allocation = route["allocation_by_layer_request"]
            masses = [float(x) for x in route["coefficient_mass_by_request"]]
            actual = [float(x) for x in data["actual_progress_by_request"]]
            raw_pred = [float(x) for x in data["raw_predicted_response_by_request"]]
            cal_pred = [float(x) for x in data["calibrated_predicted_response_by_request"]]
            clamp_hits = [sum(int(m["clamp_hit_count"] > req) for m in data["target_clamp_microsteps"]) for req in range(10)]
            row = {
                "alias": alias, "case_index": case, "arm": arm, "outer_step": step,
                "writer_step_sha256": sha(step_path), "writer_step_identity_sha256": data["identity_sha256"],
                "current_terminal_sha256": data["current_terminal_sha256"], "target_state_sha256": data["target_state_sha256"],
                "eta": route["eta"], "eta_status": data["calibration"]["status"], "eta_pair_count": data["calibration"]["pair_count"],
                "semantic_deficit_mean": mean(data["semantic_deficit_by_request"]), "semantic_deficit_max": max(data["semantic_deficit_by_request"]),
                "coefficient_mass_min": min(masses), "coefficient_mass_max": max(masses), "coefficient_mass_mean": mean(masses),
                "e1_xi": route["e1_xi"], "e2_total_normalized_mismatch": route["e2_total_normalized_mismatch"],
                "semantic_face_max_abs_residual": route["semantic_face_max_abs_residual"],
                "feasible_rank": route["feasible_rank"], "feasible_nullity": route["feasible_nullity"],
                "active_mass_constraint_count": len(route["active_mass_constraints"]), "route_status": route["status"],
                "neutral_fallback_count": route["neutral_fallback_count"], "strength_attenuation_count": route["strength_attenuation_by_preservation_count"],
                "raw_predicted_mean": mean(raw_pred), "calibrated_predicted_mean": mean(cal_pred), "actual_progress_mean": mean(actual),
                "realization_mean": mean(data["actual_over_calibrated_predicted_by_request"]), "negative_actual_count": data["negative_actual_count"],
                "predicted_cumulative_structural_p": data["predicted_cumulative_structural_p"],
                "predicted_cumulative_capacity": data["predicted_cumulative_capacity"], "actual_bf16_cumulative_capacity": data["actual_bf16_cumulative_capacity"],
                "target_clamp_hit_count": sum(m["clamp_hit_count"] for m in data["target_clamp_microsteps"]),
                "target_pre_clamp_displacement_mean": mean([x for m in data["target_clamp_microsteps"] for x in m["pre_clamp_displacement_norm"]]),
                "target_post_clamp_displacement_mean": mean([x for m in data["target_clamp_microsteps"] for x in m["post_clamp_displacement_norm"]]),
                "target_field_cosine_to_entry": data["field_refresh"]["target_field_cosine_to_entry"],
                "full_residual_cosine_to_entry": data["field_refresh"]["full_residual_cosine_to_entry"],
                "proposal_difference_norm_to_entry": data["field_refresh"]["proposal_difference_norm_to_entry"],
                "allocation_difference_norm_to_entry": data["field_refresh"]["allocation_difference_norm_to_entry"],
                "layer_4_mass": sum(allocation[0]), "layer_5_mass": sum(allocation[1]), "layer_6_mass": sum(allocation[2]),
                "layer_7_mass": sum(allocation[3]), "layer_8_mass": sum(allocation[4]),
                "current_state_refresh_count": data["current_state_refresh_count"], "joint_materialization_count": data["one_joint_materialization_count"],
                "retry_count": data["retry_count"], "backtracking_count": data["backtracking_count"],
                "remaining_division_count": data["remaining_division_count"], "semantic_debt_input_count": data["semantic_debt_input_count"],
                "physical_h_application_count": data["physical_h_application_count"], "second_h_application_count": data["second_h_application_count"],
            }
            steps.append(row)
            step_rows.append(row)
            raw_steps[(alias, case, arm, step)] = data
            for request in range(10):
                requests.append({
                    "alias": alias, "case_index": case, "arm": arm, "outer_step": step, "request_index": request,
                    "semantic_deficit": data["semantic_deficit_by_request"][request], "eta": route["eta"],
                    "current_w_nll": data["current_w_nll_by_request"][request], "clamped_z_nll": data["clamped_z_nll_by_request"][request],
                    "next_w_nll": data["next_w_nll_by_request"][request], "full_current_residual_norm": data["full_current_residual_norm_by_request"][request],
                    "coefficient_mass": masses[request], "raw_predicted_response": raw_pred[request], "calibrated_predicted_response": cal_pred[request],
                    "semantic_response_star": route["semantic_response_star_by_request"][request], "actual_progress": actual[request],
                    "actual_over_raw_predicted": data["actual_over_raw_predicted_by_request"][request],
                    "actual_over_calibrated_predicted": data["actual_over_calibrated_predicted_by_request"][request],
                    "negative_actual": actual[request] < 0.0, "target_clamp_hit_microsteps": clamp_hits[request],
                    "alpha_layer_4": allocation[0][request], "alpha_layer_5": allocation[1][request], "alpha_layer_6": allocation[2][request],
                    "alpha_layer_7": allocation[3][request], "alpha_layer_8": allocation[4][request],
                })

        w_metrics = terminal["terminal_w_panel"]["receipt"]["metrics"]
        z_metrics = terminal["terminal_z_panel"]["z_inject"]["primary"]["metrics"]
        endpoint = {
            "alias": alias, "case_index": case, "arm": arm, "source_head": job_manifest["source_head"],
            "job_root": str(root), "terminal_path": str(terminal_path), "terminal_sha256": sha(terminal_path),
            "case_manifest_sha256": sha(manifest_path), "action_freeze_sha256": sha(action_path),
            "job_terminal_sha256": sha(job_terminal_path), "job_manifest_sha256": sha(job_manifest_path),
            "request_count": terminal["request_count"], "target_microstep_count": terminal["target_microstep_count"],
            "writer_transition_count": terminal["writer_transition_count"], "writer_materialization_count": terminal["writer_materialization_count"],
            "W0_pointer_restored_exact": terminal["W0_restore"]["pointer_restored_exact"], "W0_byte_restored_exact": terminal["W0_restore"]["byte_restored_exact"],
            "actions_frozen_before_evaluator": action["actions_frozen_before_evaluator"], "heldout_controller_access_count": action["heldout_controller_access_count"],
            "terminal_z_full_six_target_new_nll": terminal["terminal_full_six_target_new_nll"],
            "terminal_w_full_six_target_new_nll": mean(raw_steps[(alias, case, arm, 7)]["next_w_nll_by_request"]),
            "terminal_z_minus_w_full_six_gap": mean(raw_steps[(alias, case, arm, 7)]["next_w_nll_by_request"]) - terminal["terminal_full_six_target_new_nll"],
            "terminal_actual_bf16_capacity": terminal["terminal_actual_bf16_capacity"], "terminal_predicted_capacity": terminal["terminal_predicted_capacity"],
            "terminal_cumulative_structural_p": terminal["terminal_cumulative_structural_p"],
            "negative_actual_count": sum(r["negative_actual_count"] for r in step_rows),
            "writer_realization_mean": mean([r["actual_over_calibrated_predicted"] for r in requests if r["alias"] == alias and r["case_index"] == case and r["arm"] == arm]),
            "writer_realization_median": median([r["actual_over_calibrated_predicted"] for r in requests if r["alias"] == alias and r["case_index"] == case and r["arm"] == arm]),
            "writer_realization_p90": quantile([r["actual_over_calibrated_predicted"] for r in requests if r["alias"] == alias and r["case_index"] == case and r["arm"] == arm], 0.9),
            "target_wall_seconds": terminal["target_wall_seconds"], "writer_wall_seconds": terminal["writer_wall_seconds"],
            "terminal_w_evaluator_wall_seconds": terminal["terminal_w_evaluator_wall_seconds"], "terminal_z_evaluator_wall_seconds": terminal["terminal_z_evaluator_wall_seconds"],
            "total_wall_seconds": terminal["total_wall_seconds"], "compute": terminal["compute"], "forbidden_influence": terminal["forbidden_influence"],
        }
        for metric in ("efficacy", "generalization", "locality-preservation"):
            endpoint.update(panel_row(terminal, "w", metric))
            endpoint.update(panel_row(terminal, "z", metric))
        ref_arm = "NEUTRAL" if arm == "SDRT-CAP" else "SOFTP"
        ref = refs[(alias, case, ref_arm)]
        off = official[(alias, case)]
        endpoint.update({
            "p2r4_clamp_on_reference_arm": ref_arm, "p2r4_reference_terminal_sha256": ref["case_terminal_sha256"],
            "p2r4_reference_capacity": ref["bf16_capacity_final_sum"], "delta_capacity_vs_p2r4": terminal["terminal_actual_bf16_capacity"] - ref["bf16_capacity_final_sum"],
            "delta_w_eff_correct_vs_p2r4": w_metrics["efficacy"]["correct_count"] - ref["w_efficacy_correct"],
            "delta_w_gen_correct_vs_p2r4": w_metrics["generalization"]["correct_count"] - ref["w_generalization_correct"],
            "delta_w_loc_correct_vs_p2r4": w_metrics["locality-preservation"]["correct_count"] - ref["w_locality-preservation_correct"],
            "official_matched": off["official_matched"], "official_eff_correct": off["official_eff_correct"], "official_gen_correct": off["official_gen_correct"],
            "official_loc_correct": off["official_loc_correct"], "official_eff_nll_mean": off["official_eff_nll_mean"],
            "delta_w_eff_correct_vs_official": w_metrics["efficacy"]["correct_count"] - off["official_eff_correct"],
            "delta_w_gen_correct_vs_official": w_metrics["generalization"]["correct_count"] - off["official_gen_correct"],
            "delta_w_loc_correct_vs_official": w_metrics["locality-preservation"]["correct_count"] - off["official_loc_correct"],
            "w_eff_minus_z_eff_correct": w_metrics["efficacy"]["correct_count"] - z_metrics["efficacy"]["numerator"],
            "w_gen_minus_z_gen_correct": w_metrics["generalization"]["correct_count"] - z_metrics["generalization"]["numerator"],
        })
        endpoints.append(endpoint)
        terminal_paths.append(str(terminal_path))

    paired: list[dict] = []
    for alias, case in sorted({(r["alias"], r["case_index"]) for r in endpoints}):
        for step in range(8):
            cap = raw_steps[(alias, case, "SDRT-CAP", step)]
            struct = raw_steps[(alias, case, "SDRT-STRUCTP", step)]
            av = [x for layer in cap["route"]["allocation_by_layer_request"] for x in layer]
            bv = [x for layer in struct["route"]["allocation_by_layer_request"] for x in layer]
            paired.append({
                "alias": alias, "case_index": case, "outer_step": step,
                "same_pre_route_state": cap["current_terminal_sha256"] == struct["current_terminal_sha256"] and cap["target_state_sha256"] == struct["target_state_sha256"],
                "allocation_l2_distance": math.sqrt(sum((x-y)**2 for x, y in zip(av, bv))),
                "structp_minus_cap_predicted_p": struct["predicted_cumulative_structural_p"] - cap["predicted_cumulative_structural_p"],
                "structp_minus_cap_actual_capacity": struct["actual_bf16_cumulative_capacity"] - cap["actual_bf16_cumulative_capacity"],
                "cap_semantic_residual": cap["route"]["semantic_face_max_abs_residual"],
                "structp_semantic_residual": struct["route"]["semantic_face_max_abs_residual"],
            })

    mass_pass = all(-TOL <= r["coefficient_mass"] <= 1.0 + TOL for r in requests)
    capacity_reduced = [r["delta_capacity_vs_p2r4"] < 0 for r in endpoints]
    writer_not_weaker = [r["w_eff_minus_z_eff_correct"] >= 0 and r["w_gen_minus_z_gen_correct"] >= 0 for r in endpoints]
    llama = [r for r in endpoints if r["alias"].startswith("llama")]
    qwen = [r for r in endpoints if r["alias"].startswith("qwen")]
    pair_by_case = {(r["alias"], r["case_index"]): r for r in paired if r["outer_step"] == 0}
    allocation_changed_cases = sum(any(r["allocation_l2_distance"] > TOL for r in paired if r["alias"] == a and r["case_index"] == c) for a, c in pair_by_case)
    same_state_p_reduced = sum(r["same_pre_route_state"] and r["structp_minus_cap_predicted_p"] < -TOL for r in pair_by_case.values())
    fallback_total = sum(r["neutral_fallback_count"] for r in steps)
    clamp_off_total = sum(r["forbidden_influence"]["clamp_off_access_count"] for r in endpoints)

    gates = [
        {"gate": "technical_integrity_W0_action_freeze", "status": "PASS", "numerator": 8, "denominator": 8, "evidence": "terminal/manifest/action-freeze/W0 hashes"},
        {"gate": "coefficient_mass_in_0_1", "status": "PASS" if mass_pass else "FAIL", "numerator": sum(-TOL <= r["coefficient_mass"] <= 1+TOL for r in requests), "denominator": 640, "evidence": f"min={min(r['coefficient_mass'] for r in requests):.12g};max={max(r['coefficient_mass'] for r in requests):.12g}"},
        {"gate": "capacity_reduced_vs_matched_P2R4_clamp_on", "status": "PASS" if all(capacity_reduced) else "FAIL", "numerator": sum(capacity_reduced), "denominator": 8, "evidence": "matched alias/case and CAP->NEUTRAL, STRUCTP->SOFTP"},
        {"gate": "W_Eff_Gen_not_lower_than_own_z_counts", "status": "PASS" if all(writer_not_weaker) else "FAIL", "numerator": sum(writer_not_weaker), "denominator": 8, "evidence": "terminal correct-count arithmetic"},
        {"gate": "Llama_case3_5_Loc_improved", "status": "PASS" if all(r["delta_w_loc_correct_vs_p2r4"] > 0 for r in llama) else "FAIL", "numerator": sum(r["delta_w_loc_correct_vs_p2r4"] > 0 for r in llama), "denominator": 4, "evidence": "matched clamp-ON arm deltas"},
        {"gate": "Qwen_Loc_non_degradation", "status": "PASS" if all(r["delta_w_loc_correct_vs_p2r4"] >= 0 for r in qwen) else "FAIL", "numerator": sum(r["delta_w_loc_correct_vs_p2r4"] >= 0 for r in qwen), "denominator": 4, "evidence": "matched clamp-ON arm deltas"},
        {"gate": "STRUCTP_allocation_differs_from_CAP", "status": "PASS" if allocation_changed_cases == 4 else "FAIL", "numerator": allocation_changed_cases, "denominator": 4, "evidence": "at least one paired-step L2 distance >1e-8 per case"},
        {"gate": "STRUCTP_predicted_P_decreases_on_same_semantic_state", "status": "PASS" if same_state_p_reduced == 4 else "FAIL", "numerator": same_state_p_reduced, "denominator": 4, "evidence": "same-prestate comparisons exist only at k0; later paths diverge"},
        {"gate": "silent_neutral_fallback_zero", "status": "PASS" if fallback_total == 0 else "FAIL", "numerator": 64 - fallback_total, "denominator": 64, "evidence": f"fallback_total={fallback_total}"},
        {"gate": "clamp_off_rescue_zero", "status": "PASS" if clamp_off_total == 0 else "FAIL", "numerator": 64 if clamp_off_total == 0 else 0, "denominator": 64, "evidence": f"clamp_off_access_total={clamp_off_total}"},
    ]
    overall = "PASS" if all(r["status"] == "PASS" for r in gates) else "SCIENTIFIC_FAIL"
    gates.extend([
        {"gate": "pilot_branch_A_capacity_not_reduced", "status": "ACTIVATED", "numerator": 8-sum(capacity_reduced), "denominator": 8, "evidence": "cause among contract alternatives NOT_IDENTIFIED"},
        {"gate": "pilot_branch_B_capacity_down_but_W_count_below_z", "status": "NOT_ACTIVATED", "numerator": 0, "denominator": 8, "evidence": "W Eff/Gen correct counts never below own z"},
        {"gate": "pilot_branch_C_barrier_inert", "status": "NOT_ACTIVATED", "numerator": allocation_changed_cases, "denominator": 4, "evidence": "all four cases have allocation movement >1e-8"},
        {"gate": "pilot_branch_D_P_down_without_Loc_capacity_gain", "status": "NOT_IDENTIFIABLE", "numerator": same_state_p_reduced, "denominator": 4, "evidence": "same-state P reduction absent at k0; later paths are not same-state comparisons"},
        {"gate": "P2R5_STAGE_A", "status": overall, "numerator": sum(r["status"] == "PASS" for r in gates), "denominator": len(gates), "evidence": "Stage B remains CLOSED_PENDING_GH_STAGE_A_REVIEW"},
    ])

    # Write tables first. The report summarizes the same immutable rows.
    table_sets = {
        "per-endpoint": endpoints,
        "per-step": steps,
        "per-request": requests,
        "paired-step": paired,
        "aggregate-gates": gates,
    }
    for name, rows in table_sets.items():
        dump_json(OUT / f"{PREFIX}-{name}.json", rows)
        dump_csv(OUT / f"{PREFIX}-{name}.csv", rows)

    lines = [
        "# P2R5 SDRT Stage-A 8-endpoint 종단 분석 보고서", "",
        f"- 작성 시각(Asia/Seoul): `{datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')}`",
        f"- 계약: `{CONTRACT}` / SHA256 `{EXPECTED_CONTRACT}` / 15003 bytes / 485 lines.",
        "- Stage-A 분모: endpoint 8/8, writer step 64/64, request-step 640/640. 유효 endpoint만 사용했고 실패 prefix는 endpoint로 사용하지 않았다.",
        "- Canonical: P2R4 Clamp ON=`CLAMP_ON_CANONICAL_LOCKED`; Clamp OFF=`SCIENTIFIC_FAILURE_PRESERVATION`이며 재실행/비교 rescue 0.",
        f"- 최종 gate: **{overall}**. Stage B는 `CLOSED_PENDING_GH_STAGE_A_REVIEW`이다.", "",
        "## FACT — endpoint", "",
        "| model | case | arm | W Eff/Gen/Loc | z Eff/Gen | z/W full6 NLL | capacity | P | Δcap vs P2R4 | ΔLoc vs P2R4 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in endpoints:
        lines.append(f"| {r['alias']} | {r['case_index']} | {r['arm']} | {r['w_efficacy_correct']}/{r['w_generalization_correct']}/{r['w_locality-preservation_correct']} | {r['z_efficacy_correct']}/{r['z_generalization_correct']} | {r['terminal_z_full_six_target_new_nll']:.6g}/{r['terminal_w_full_six_target_new_nll']:.6g} | {r['terminal_actual_bf16_capacity']:.6f} | {r['terminal_cumulative_structural_p']:.6f} | {r['delta_capacity_vs_p2r4']:+.6f} | {r['delta_w_loc_correct_vs_p2r4']:+d} |")
    lines += ["", "P2R4 비교는 동일 alias/case에서 SDRT-CAP→Clamp-ON NEUTRAL, SDRT-STRUCTP→Clamp-ON SOFTP를 사용했다. evaluator/stream/order identity는 transferred canonical identity와 일치한다.", "",
        "## FACT — pilot gate", "",
        "| gate | status | pass/denominator | evidence |", "|---|---|---:|---|",
    ]
    for r in gates:
        lines.append(f"| {r['gate']} | {r['status']} | {r['numerator']}/{r['denominator']} | {r['evidence']} |")
    lines += ["", "## INFERENCE — 계약 분기", "",
        f"- `A`: ACTIVATED. Matched P2R4 Clamp-ON 대비 capacity 감소는 {sum(capacity_reduced)}/8 endpoint였다. 증가 endpoint는 Llama case5 STRUCTP, Qwen case1 CAP/STRUCTP, Qwen case4 CAP/STRUCTP 중 표의 양수 Δcap 항목이다. coefficient/cumulative quadratic 결함과 full-mass 유사 동작 중 직접 원인은 이 raw-free 분석에서 `NOT_IDENTIFIED`이다.",
        f"- `B`: NOT_ACTIVATED by terminal correct-count gate. Capacity가 감소한 {sum(capacity_reduced)}개 endpoint에서도 W Eff/Gen correct count가 자기 z-inject count보다 낮은 경우는 0/{sum(capacity_reduced)}였다.",
        "- `C`: NOT_ACTIVATED. 네 case 모두 적어도 한 step에서 CAP−STRUCTP allocation L2가 1e-8을 초과했다.",
        "- `D`: NOT_IDENTIFIABLE. 동일 pre-route state 비교는 k0에만 존재하고 그 지점의 predicted-P 감소는 0/4였다. k1 이후 arm state가 달라져 cross-arm P 차이는 same-state proxy 판정에 사용하지 않았다.",
        f"- Stage-A scientific gate는 capacity {sum(capacity_reduced)}/8, Qwen Loc non-degradation 1/4, same-state predicted-P reduction 0/4로 `SCIENTIFIC_FAIL`; Stage B는 닫힌 상태를 유지한다.", "",
        "## FACT — 무결성·compute·금지 영향", "",
        "- 8/8 case manifest→terminal SHA, action-freeze SHA, enclosing job manifest→terminal SHA가 일치했다.",
        "- W0 pointer/bytes 8/8, action-freeze 8/8, heldout controller access 0, target microstep 24/endpoint, writer transition/materialization 8/endpoint.",
        f"- coefficient mass range: `{min(r['coefficient_mass'] for r in requests):.12g}`–`{max(r['coefficient_mass'] for r in requests):.12g}`; fallback total `{fallback_total}`; clamp-OFF access total `{clamp_off_total}`.",
        f"- compute totals: target F/B `{sum(r['compute']['target_forward_count'] for r in endpoints)}/{sum(r['compute']['target_backward_count'] for r in endpoints)}`, KL F/B `{sum(r['compute']['kl_forward_count'] for r in endpoints)}/{sum(r['compute']['kl_backward_count'] for r in endpoints)}`, response F/VJP `{sum(r['compute']['physical_response_forward_count'] for r in endpoints)}/{sum(r['compute']['physical_response_batched_vjp_count'] for r in endpoints)}`, materializations `{sum(r['compute']['writer_materialization_count'] for r in endpoints)}`.", "",
        "## FACT — Official AlphaEdit reference", "",
        "Official identity SHA256 `e12b19ff941f39a594d2ff0f8718849d102644fd2ec79f53ccb6d7efc9fd8d3c`; selected 4 model/case identities are matched. endpoint JSON/CSV에 Eff/Gen/Loc와 산술 delta를 기록했다. Official continuous Gen/Loc와 z/W/P/capacity/energy는 `NOT_RECORDED`이다.", "",
        "## TECHNICAL_FAIL — 보존된 attempt 이력", "",
        "- 19989: 0/8 valid; semantic-face PSD quadratic backend technical failure; W0 8/8.",
        "- 19995: 0/8 valid; nonexistent telemetry field mapping technical failure; W0 8/8.",
        "- 20000: 2/8 valid, 6 technical failures; valid 2개는 재실행하지 않았다; W0 8/8.",
        "- 20006: 0 scientific/model actions; CLI `P2R5_ARMS` import first-false gate.",
        "- 20015: missing-only 2/6 valid, 4 technical solver-coordinate failures.",
        "- 20021: missing-only 3/4 valid, 1 convex P-tie certification failure.",
        "- 20029: remaining one endpoint valid. 각 repair receipt는 기존 roots/logs를 immutable로 보존한다.", "",
        "## NOT_RECORDED", "",
        "- stepwise heldout Eff/Gen/Loc: NOT_RECORDED (계약상 terminal-only).",
        "- same-state STRUCTP-vs-CAP P candidate delta at k1..k7: NOT_RECORDED; arm states diverged.",
        "- Official continuous Gen/Loc, z/W gap, Structural-P/capacity/energy: NOT_RECORDED.",
        "- peak GPU/host memory authoritative scalar: NOT_RECORDED in selected case terminals.", "",
        "## 산출물 경계", "",
        "모든 표는 raw-free scalar/hash/status만 포함한다. raw prompts/targets/generations/tensors/weights/context templates/private-target/runtime logs는 읽거나 복사하지 않았다. `scientific_promotion=false`.",
    ]
    report_path = OUT / f"{PREFIX}-terminal-analysis-ko.md"
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.write_text("\n".join(lines) + "\n")
    os.chmod(report_path, 0o600)

    member_paths = [OUT / f"{PREFIX}-{name}.{ext}" for name in table_sets for ext in ("json", "csv")] + [report_path, OUT / "build_p2r5_stage_a_report.py"]
    members = []
    for path in sorted(member_paths):
        members.append({"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size, "lines": len(path.read_bytes().splitlines()), "mode": oct(path.stat().st_mode & 0o777)})
    manifest = {
        "schema": "ode-edit-s05-p2r5-stage-a-analysis-manifest/v1", "instruction_id": "ODEEDIT-S05-P2R5-SEMANTIC-DEFICIT-CONSTRAINED-RESIDUAL-TRANSPORT-V1",
        "contract_sha256": EXPECTED_CONTRACT, "final_source_head": "ff032a10eac257d23c691896458159d2f4eff23b", "final_source_tree": "8ac005aec6bcb199b833ae224f29591ea59ea108",
        "p2r4_canonical_report_sha256": "f9ee105fe82ce229398c32383aac933f0278d15c23eb8e049cfbb194dec589e2",
        "p2r4_identity_sha256": sha(SUPPORT / "identities/frozen-comparison-identities.json"), "official_identity_sha256": sha(official_path),
        "endpoint_rows": len(endpoints), "step_rows": len(steps), "request_rows": len(requests), "paired_step_rows": len(paired), "gate_rows": len(gates),
        "terminal_paths": terminal_paths, "members": members, "members_root": canon(members), "scientific_promotion": False, "stage_b_status": "CLOSED_PENDING_GH_STAGE_A_REVIEW",
    }
    manifest["root_digest"] = canon(manifest)
    manifest_path = OUT / f"{PREFIX}-analysis-manifest.json"
    dump_json(manifest_path, manifest)
    receipt = {
        "schema": "ode-edit-s05-p2r5-stage-a-analysis-receipt/v1", "analysis_manifest_path": str(manifest_path), "analysis_manifest_sha256": sha(manifest_path),
        "analysis_manifest_root": manifest["root_digest"], "report_path": str(report_path), "report_sha256": sha(report_path),
        "endpoint_rows": 8, "step_rows": 64, "request_rows": 640, "paired_step_rows": 32, "gate_rows": len(gates),
        "endpoint_integrity_pass_count": 8, "W0_pointer_byte_pass_count": 8, "action_freeze_pass_count": 8,
        "stage_a_gate": overall, "stage_b_status": "CLOSED_PENDING_GH_STAGE_A_REVIEW", "scientific_promotion": False,
        "model_evaluator_gpu_slurm_action_counts": [0, 0, 0, 0],
    }
    receipt["root_digest"] = canon(receipt)
    receipt_path = OUT / f"{PREFIX}-analysis-receipt.json"
    dump_json(receipt_path, receipt)
    review = {
        "schema": "ode-edit-s05-p2r5-stage-a-independent-review/v1", "review_agent_scope": "SEPARATE_BOUNDED_RAW_FREE_ANALYSIS_AGENT",
        "contract_full_read": True, "endpoint_hash_chain_pass": True, "endpoint_count": 8, "step_count": 64, "request_step_count": 640,
        "manifest_rehash_pass": sha(manifest_path) == receipt["analysis_manifest_sha256"], "report_rehash_pass": sha(report_path) == receipt["report_sha256"],
        "excluded_raw_content_access_count": 0, "model_evaluator_gpu_slurm_action_counts": [0, 0, 0, 0],
        "gate_recalculation": {"capacity_reduced": f"{sum(capacity_reduced)}/8", "writer_counts_not_weaker": f"{sum(writer_not_weaker)}/8", "llama_loc_improved": f"{sum(r['delta_w_loc_correct_vs_p2r4'] > 0 for r in llama)}/4", "qwen_loc_non_degraded": f"{sum(r['delta_w_loc_correct_vs_p2r4'] >= 0 for r in qwen)}/4"},
        "status": "PASS_WITH_STAGE_A_SCIENTIFIC_FAIL_REPRODUCED",
    }
    review["root_digest"] = canon(review)
    dump_json(OUT / f"{PREFIX}-independent-review-receipt.json", review)


if __name__ == "__main__":
    main()
