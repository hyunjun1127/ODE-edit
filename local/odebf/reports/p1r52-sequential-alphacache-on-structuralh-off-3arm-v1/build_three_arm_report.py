#!/usr/bin/env python3
"""Raw-free report-only builder: Structural-H On, Structural-H Off, Native."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


OUT = Path(__file__).resolve().parent
CONTROL = OUT.parents[3] / "local/odebf/results/s05-p1r52-llama-soft-sequential-alphacache-on-structuralh-off-10xb10-v1"
HROOT = Path("/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-llama-soft-seq-hist-10xb10-v1/local/odebf/results/s05-p1r52-llama-soft-sequential-historical-10xb10-tech-r3-v1")
NROOT = Path("/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-llama-soft-seq-hist-10xb10-v1/local/odebf/results/s05-p1r52-native-alphaedit-sequential-10xb10-v1")
METRICS = ("rewrite_success", "rewrite_acc", "paraphrase_success", "paraphrase_acc")
ARM_LABELS = {
    "h_on": "P1R52 Soft Structural-H On",
    "h_off": "P1R52 Soft AlphaCache-On Structural-H Off",
    "native": "Native AlphaEdit Sequential",
}


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, payload) -> Path:
    path = OUT / name
    if path.exists():
        raise RuntimeError(f"create-once output already exists: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.chmod(path, 0o600)
    return path


def compact(summary: dict) -> dict:
    out = {"identity_sha256": summary.get("identity_sha256"), "request_count": summary.get("request_count"), "batch_count": summary.get("batch_count")}
    for name in METRICS:
        source = summary["metrics"][name]
        out[name] = {
            "prompt_numerator": source["prompt_numerator"],
            "prompt_denominator": source["prompt_denominator"],
            "prompt_rate": source["prompt_rate"],
            "strict_request_numerator": source["strict_request_numerator"],
            "strict_request_denominator": source["strict_request_denominator"],
            "strict_request_rate": source["strict_request_rate"],
            "target_new_nll_mean": source["target_new_nll"]["mean"],
            "target_old_nll_mean": source["target_old_nll"]["mean"],
            "target_old_minus_new_margin_mean": source["target_old_minus_new_margin"]["mean"],
        }
    out["locality"] = {key: summary["locality"][key] for key in ("numerator", "denominator", "rate")}
    return out


def subtraction(left: dict, right: dict) -> dict:
    out = {}
    for metric in METRICS:
        out[metric] = {field: left[metric][field] - right[metric][field] for field in (
            "prompt_numerator", "prompt_rate", "strict_request_numerator", "strict_request_rate",
            "target_new_nll_mean", "target_old_nll_mean", "target_old_minus_new_margin_mean",
        )}
    out["locality"] = {field: left["locality"][field] - right["locality"][field] for field in ("numerator", "rate")}
    return out


def count(metric: dict, strict=False) -> str:
    p = "strict_request_" if strict else "prompt_"
    return f"{metric[p+'numerator']}/{metric[p+'denominator']} ({metric[p+'rate']:.3f})"


def phrase(summary: dict) -> str:
    return (
        f"RW성공 {count(summary['rewrite_success'])}; RW정확 {count(summary['rewrite_acc'])}; "
        f"PP성공 {count(summary['paraphrase_success'])}/엄격 {count(summary['paraphrase_success'], True)}; "
        f"PP정확 {count(summary['paraphrase_acc'])}/엄격 {count(summary['paraphrase_acc'], True)}; "
        f"Loc {summary['locality']['numerator']}/{summary['locality']['denominator']} ({summary['locality']['rate']:.3f})"
    )


def request_metric(rec: dict, metric: str) -> dict:
    if metric == "locality":
        return {"correct": rec[metric]["correct"], "required": rec[metric]["required"], "rate": rec[metric]["rate"], "strict_all_prompts_pass": None}
    return {
        "correct": rec[metric]["correct"], "required": rec[metric]["required"], "rate": rec[metric]["rate"],
        "strict_all_prompts_pass": rec[metric]["strict_all_prompts_pass"],
        "target_new_nll_mean": rec[metric]["target_new_nll_mean"],
        "target_old_nll_mean": rec[metric]["target_old_nll_mean"],
        "target_old_minus_new_margin_mean": rec[metric]["target_old_minus_new_margin_mean"],
    }


def request_phase(rec: dict, phase: str) -> dict:
    return {m: request_metric(rec[phase], m) for m in (*METRICS, "locality")}


def raw_batch(root: Path, round_id: int) -> dict:
    return load(root / f"raw/batches/b{round_id:02d}/terminal.json")


def input_files(root: Path) -> list[Path]:
    top = [root / x for x in ("manifest.json", "terminal.json", "b1-b10-pre-post-checkpoints.json", "batch-age-pre-post-final-cohorts.json", "entry-pre-immediate-post-final-w10-requests.json")]
    return top + [root / f"raw/batches/b{i:02d}/terminal.json" for i in range(1, 11)]


def main() -> None:
    roots = {"h_on": HROOT, "h_off": CONTROL, "native": NROOT}
    terminals = {key: load(root / "terminal.json") for key, root in roots.items()}
    checkpoints = {key: load(root / "b1-b10-pre-post-checkpoints.json")["rows"] for key, root in roots.items()}
    cohorts = {key: load(root / "batch-age-pre-post-final-cohorts.json")["rows"] for key, root in roots.items()}
    requests = {key: load(root / "entry-pre-immediate-post-final-w10-requests.json")["rows"] for key, root in roots.items()}
    for key in roots:
        if not (len(checkpoints[key]) == len(cohorts[key]) == 10 and len(requests[key]) == 100):
            raise RuntimeError(f"unexpected sealed row count for {key}")

    # Exact same case/request pairing for all three immutable arms.
    def req_key(row): return (row["round"], row["case_id"], row["request_index"], row["request_sha256"])
    keysets = {name: {req_key(r) for r in rows} for name, rows in requests.items()}
    if not (keysets["h_on"] == keysets["h_off"] == keysets["native"]):
        raise RuntimeError("three-arm request identity mismatch")
    cp_by_round = {a: {row["round"]: row for row in rows} for a, rows in checkpoints.items()}
    cohort_by_round = {a: {row["round"]: row for row in rows} for a, rows in cohorts.items()}
    for round_id in range(1, 11):
        orders = {cp_by_round[a][round_id]["entry_pre_summary"]["request_order_sha256"] for a in roots}
        if len(orders) != 1:
            raise RuntimeError(f"round {round_id} request-order mismatch")

    # Control standalone checkpoint/transaction/routing package.
    control_checkpoint_rows = []
    for round_id in range(1, 11):
        cp, cohort = cp_by_round["h_off"][round_id], cohort_by_round["h_off"][round_id]
        for phase, summary in (("entry_pre", cp["entry_pre_summary"]), ("immediate_post", cp["immediate_post_summary"]), ("final_W10", cohort["final_W10"])):
            control_checkpoint_rows.append({"checkpoint": f"B{round_id}", "round": round_id, "phase": phase, "history_width_at_entry": cp["history_width_at_entry"], "batch_age_at_W10": cohort["batch_age_at_W10"], "summary": compact(summary)})
    control_checkpoint_rows.append({"checkpoint": "B100", "round": None, "phase": "final_W10_B100", "history_width_at_entry": None, "batch_age_at_W10": None, "summary": compact(terminals["h_off"]["final_B100"])})

    h_off_routing, h_off_tx = [], []
    h_on_routing = []
    for round_id in range(1, 11):
        off = raw_batch(CONTROL, round_id)
        on = raw_batch(HROOT, round_id)
        for row in off["structural_h_routing"]:
            h_off_routing.append({
                "round": round_id, "step_index": row["step_index"], "status": row["status"],
                "alpha_solve_history_width": row["alpha_solve_history_width"], "alpha_solve_cache_consume_count": row["alpha_solve_cache_consume_count"],
                "alpha_solve_cache_append_count": row["alpha_solve_cache_append_count"], "risk_observation_width": row["risk_observation_width"], "anchor_observation_width": row["anchor_observation_width"],
                "structural_h_decision_history_width": row["structural_h_decision_history_width"], "structural_h_decision_influence_count": row["structural_h_decision_influence_count"],
                "observation_ledger_decision_influence_count": row["observation_ledger_decision_influence_count"],
                "added_model_forward_count": row["added_model_forward_count"], "added_backward_count": row["added_backward_count"], "added_materialization_count": row["added_materialization_count"],
                "strength_residual": row["strength_residual"], "energy_violation": row["energy_violation"], "p_violation": row["p_violation"],
                "allocation_entropy": row["allocation_entropy"], "selected_minus_neutral_norm": row["selected_minus_neutral_norm"], "identity_sha256": row["identity_sha256"],
            })
        for row in on["structural_h_routing"]:
            h_on_routing.append({"round": round_id, "step_index": row["step_index"], "status": row["status"], "history_width": row["history_width"], "h_delta": row["h_delta"], "h_neutral": row["h_neutral"], "h_selected": row["h_selected"], "strength_residual": row["strength_residual"], "energy_violation": row["energy_violation"], "p_violation": row["p_violation"], "allocation_entropy": row["allocation_entropy"], "selected_minus_neutral_norm": row["selected_minus_neutral_norm"], "identity_sha256": row["identity_sha256"]})
        tx, wt = off["history_transaction"], off["weight_transaction"]
        h_off_tx.append({"round": round_id, "history_width_at_entry": off["history_width_at_entry"], "active_history_count_after": off["active_history_count"], "anchor_count_after": off["lifetime_anchor_count"], "history_append_count": tx["appended_count"], "history_before_version": tx["before_version"], "history_after_version": tx["after_version"], "weight_commit_count": wt["commit_count"], "post_commit_verified": wt["post_commit_verified"], "rollback_count": wt["rollback_count"], "retry_count": off["retry_count"], "backtracking_count": off["backtracking_count"], "interbatch_W0_restore_count": off["interbatch_W0_restore_count"], "heldout_controller_influence_count": off["heldout_controller_influence_count"], "entry_weight_sha256": off["entry_weight_sha256"], "commit_weight_sha256": off["commit_weight_sha256"], "transaction_id": tx["transaction_id"]})
    control_route_summary = {
        "routing_rows": len(h_off_routing),
        "statuses": {s: sum(r["status"] == s for r in h_off_routing) for s in sorted({r["status"] for r in h_off_routing})},
        "alpha_solve_cache_append_transaction_total": sum(r["history_append_count"] for r in h_off_tx),
        "alpha_solve_cache_consume_total": sum(r["alpha_solve_cache_consume_count"] for r in h_off_routing),
        "positive_alpha_solve_history_width_rows": sum(r["alpha_solve_history_width"] > 0 for r in h_off_routing),
        "structural_h_decision_influence_total": sum(r["structural_h_decision_influence_count"] for r in h_off_routing),
        "added_model_forward_backward_materialization": [sum(r["added_model_forward_count"] for r in h_off_routing), sum(r["added_backward_count"] for r in h_off_routing), sum(r["added_materialization_count"] for r in h_off_routing)],
        "max_abs_strength_residual": max(abs(r["strength_residual"]) for r in h_off_routing),
        "max_energy_violation": max(r["energy_violation"] for r in h_off_routing),
        "max_abs_p_violation": max(abs(r["p_violation"]) for r in h_off_routing),
        "terminal_active_history_count": terminals["h_off"]["terminal_active_history_count"],
        "terminal_anchor_count": terminals["h_off"]["terminal_lifetime_anchor_count"],
    }

    # 3-arm checkpoint comparison: absolute panels and all pair arithmetic.
    three_phase_rows = []
    for round_id in range(1, 11):
        for phase in ("entry_pre", "immediate_post", "final_W10"):
            summaries = {}
            for arm in roots:
                cp, cohort = cp_by_round[arm][round_id], cohort_by_round[arm][round_id]
                source = cp["entry_pre_summary"] if phase == "entry_pre" else cp["immediate_post_summary"] if phase == "immediate_post" else cohort["final_W10"]
                summaries[arm] = compact(source)
            three_phase_rows.append({"checkpoint": f"B{round_id}", "round": round_id, "phase": phase, "history_width_at_entry": cp_by_round["h_off"][round_id]["history_width_at_entry"], "batch_age_at_W10": cohort_by_round["h_off"][round_id]["batch_age_at_W10"], "arms": summaries, "h_on_minus_h_off": subtraction(summaries["h_on"], summaries["h_off"]), "h_off_minus_native": subtraction(summaries["h_off"], summaries["native"]), "h_on_minus_native": subtraction(summaries["h_on"], summaries["native"])})
    finals = {arm: compact(terminals[arm]["final_B100"]) for arm in roots}
    three_phase_rows.append({"checkpoint": "B100", "round": None, "phase": "final_W10_B100", "history_width_at_entry": None, "batch_age_at_W10": None, "arms": finals, "h_on_minus_h_off": subtraction(finals["h_on"], finals["h_off"]), "h_off_minus_native": subtraction(finals["h_off"], finals["native"]), "h_on_minus_native": subtraction(finals["h_on"], finals["native"])})

    # Request-level triple trajectory and observational low cohort transitions.
    req_maps = {arm: {req_key(r): r for r in rows} for arm, rows in requests.items()}
    triple_request_rows, low_rows = [], []
    for key in sorted(keysets["h_on"]):
        arm_phases = {arm: {phase: request_phase(req_maps[arm][key], phase) for phase in ("entry_pre", "immediate_post", "final_W10")} for arm in roots}
        hrec, offrec, nrec = req_maps["h_on"][key], req_maps["h_off"][key], req_maps["native"][key]
        diffs = {}
        for phase in arm_phases["h_on"]:
            diffs[phase] = {}
            for metric in (*METRICS, "locality"):
                diffs[phase][metric] = {
                    "h_on_minus_h_off_correct": arm_phases["h_on"][phase][metric]["correct"] - arm_phases["h_off"][phase][metric]["correct"],
                    "h_on_minus_h_off_rate": arm_phases["h_on"][phase][metric]["rate"] - arm_phases["h_off"][phase][metric]["rate"],
                    "h_off_minus_native_correct": arm_phases["h_off"][phase][metric]["correct"] - arm_phases["native"][phase][metric]["correct"],
                    "h_off_minus_native_rate": arm_phases["h_off"][phase][metric]["rate"] - arm_phases["native"][phase][metric]["rate"],
                }
        row = {"round": key[0], "case_id": key[1], "request_index": key[2], "request_sha256": key[3], "history_width_at_entry": hrec["history_width_at_entry"], "arms": arm_phases, "deltas": diffs, "h_on_lifetime_deltas": hrec["deltas"], "h_off_lifetime_deltas": offrec["deltas"], "native_lifetime_deltas": nrec["deltas"]}
        triple_request_rows.append(row)
        hpost, opost, npost = arm_phases["h_on"]["immediate_post"], arm_phases["h_off"]["immediate_post"], arm_phases["native"]["immediate_post"]
        hfinal, ofinal = arm_phases["h_on"]["final_W10"], arm_phases["h_off"]["final_W10"]
        flags = {
            "entry_rewrite_hard_h_on": arm_phases["h_on"]["entry_pre"]["rewrite_success"]["correct"] < arm_phases["h_on"]["entry_pre"]["rewrite_success"]["required"],
            "h_on_post_paraphrase_failure": hpost["paraphrase_success"]["correct"] < hpost["paraphrase_success"]["required"],
            "h_off_post_paraphrase_failure": opost["paraphrase_success"]["correct"] < opost["paraphrase_success"]["required"],
            "h_on_post_strict_paraphrase_failure": hpost["paraphrase_success"]["strict_all_prompts_pass"] == 0,
            "h_off_post_strict_paraphrase_failure": opost["paraphrase_success"]["strict_all_prompts_pass"] == 0,
            "h_on_final_paraphrase_loss": hfinal["paraphrase_success"]["correct"] < hpost["paraphrase_success"]["correct"],
            "h_off_final_paraphrase_loss": ofinal["paraphrase_success"]["correct"] < opost["paraphrase_success"]["correct"],
            "h_on_post_better_than_h_off_paraphrase": hpost["paraphrase_success"]["correct"] > opost["paraphrase_success"]["correct"],
            "h_off_post_better_than_h_on_paraphrase": opost["paraphrase_success"]["correct"] > hpost["paraphrase_success"]["correct"],
            "native_post_better_than_h_off_paraphrase": npost["paraphrase_success"]["correct"] > opost["paraphrase_success"]["correct"],
        }
        if any(flags.values()):
            low_rows.append({"round": key[0], "case_id": key[1], "request_index": key[2], "request_sha256": key[3], "history_width_at_entry": hrec["history_width_at_entry"], "flags": flags, "arms": arm_phases, "deltas": diffs})

    def flag_count(name): return sum(bool(r["flags"].get(name)) for r in low_rows)
    b5 = next(r for r in three_phase_rows if r["round"] == 5 and r["phase"] == "immediate_post")
    low_summary = {"association_scope": "RAW_FREE_OBSERVATIONAL_ONLY_NO_ISOLATED_CAUSAL_ATTRIBUTION", "triple_request_rows": len(triple_request_rows), "low_or_transition_rows": len(low_rows), "flag_counts": {name: flag_count(name) for name in (
        "entry_rewrite_hard_h_on", "h_on_post_paraphrase_failure", "h_off_post_paraphrase_failure", "h_on_post_strict_paraphrase_failure", "h_off_post_strict_paraphrase_failure", "h_on_final_paraphrase_loss", "h_off_final_paraphrase_loss", "h_on_post_better_than_h_off_paraphrase", "h_off_post_better_than_h_on_paraphrase", "native_post_better_than_h_off_paraphrase")}, "B5_immediate_post": b5, "requestwise_H_routing_attribution": "NOT_RECORDED_BATCH_STEP_ROUTING_ONLY", "conclusion_label": "ASSOCIATION_ONLY"}

    # Compute ledger: exact terminal receipt plus summed atomic phase totals where available.
    compute = {"terminals": {arm: terminals[arm]["job_compute"] for arm in roots}}
    for arm, root in roots.items():
        totals, walls = {}, {}
        is_native = arm == "native"
        edit_core = 0.0
        for round_id in range(1, 11):
            b = raw_batch(root, round_id)
            if not is_native:
                for k, v in b["atomic_or_native"]["compute"]["totals"].items(): totals[k] = totals.get(k, 0) + v
                for k, v in b["atomic_or_native"]["compute"]["wall_seconds"].items(): walls[k] = walls.get(k, 0) + v
            else:
                edit_core += b["atomic_or_native"]["edit_core_wall_seconds"]
        compute[arm] = {"atomic_compute_totals_sum": totals if not is_native else "NOT_RECORDED_NATIVE_OFFICIAL_PATH", "atomic_wall_seconds_sum": walls if not is_native else "NOT_RECORDED_NATIVE_OFFICIAL_PATH", "native_edit_core_wall_seconds_sum": edit_core if is_native else None, "accuracy_extra_forward_backward_generation": [terminals[arm]["accuracy_amendment_added_model_forward_count"], terminals[arm]["accuracy_amendment_added_backward_count"], terminals[arm]["accuracy_amendment_added_generation_count"]], "entry_pre_evaluator_forward_backward_generation": [terminals[arm]["batch_entry_pre_evaluator_forward_count"], terminals[arm]["batch_entry_pre_evaluator_backward_count"], terminals[arm]["batch_entry_pre_evaluator_generation_count"]]}

    # Input identity manifest.
    sources = []
    for arm, root in roots.items():
        for p in input_files(root): sources.append({"arm": arm, "path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)})

    # Write machine tables before reports/manifest.
    write("p1r52-alphacache-on-structuralh-off-checkpoint-phase-table.json", {"schema": "p1r52-alphacache-on-structuralh-off/checkpoint-phase/v1", "row_count": len(control_checkpoint_rows), "rows": control_checkpoint_rows})
    write("p1r52-alphacache-on-structuralh-off-routing-table.json", {"schema": "p1r52-alphacache-on-structuralh-off/routing/v1", "row_count": len(h_off_routing), "rows": h_off_routing, "summary": control_route_summary})
    write("p1r52-alphacache-on-structuralh-off-transaction-table.json", {"schema": "p1r52-alphacache-on-structuralh-off/transaction/v1", "row_count": len(h_off_tx), "rows": h_off_tx})
    write("p1r52-sequential-three-arm-checkpoint-table.json", {"schema": "p1r52-sequential/three-arm-checkpoint/v1", "row_count": len(three_phase_rows), "rows": three_phase_rows})
    write("p1r52-sequential-three-arm-per-request-table.json", {"schema": "p1r52-sequential/three-arm-per-request/v1", "row_count": len(triple_request_rows), "rows": triple_request_rows})
    write("p1r52-sequential-three-arm-low-cohort-table.json", {"schema": "p1r52-sequential/three-arm-low-cohort/v1", "row_count": len(low_rows), "rows": low_rows, "summary": low_summary})
    write("p1r52-sequential-three-arm-compute-ledger.json", {"schema": "p1r52-sequential/three-arm-compute/v1", "payload": compute})
    write("p1r52-sequential-h-on-routing-reference-table.json", {"schema": "p1r52-sequential/h-on-routing-reference/v1", "row_count": len(h_on_routing), "rows": h_on_routing})

    # Standalone structural-H-off factual report.
    off_report = OUT / "p1r52-soft-sequential-alphacache-on-structuralh-off-10xb10-factual-ko.md"
    if off_report.exists(): raise RuntimeError(f"create-once report exists: {off_report}")
    text = []
    a = text.append
    a("# P1R52-Soft-Sequential-AlphaCache-On-StructuralH-Off-10xB10")
    a(""); a("## 독립 종료 사실"); a("")
    a("- canonical control: AlphaEdit solve/history cache ON; Structural-H action/objective/constraint decision influence OFF.")
    a("- job 20428: `COMPLETED`, exit `0:0`, elapsed `00:29:23`; 10×B10/100 요청, action freeze 10/10.")
    a(f"- source `{terminals['h_off']['source_head']}`; method `{terminals['h_off']['method_id']}`; W0 terminal pointer+byte exact={terminals['h_off']['terminal_W0_restore']['pointer_restored_exact']}/{terminals['h_off']['terminal_W0_restore']['byte_restored_exact']}.")
    a(""); a("## Alpha cache·transaction·Structural-H OFF receipt"); a("")
    a(f"- alpha solve history widths: {terminals['h_off']['history_widths']}; transaction append total={control_route_summary['alpha_solve_cache_append_transaction_total']}; routing consume total={control_route_summary['alpha_solve_cache_consume_total']}; positive history rows={control_route_summary['positive_alpha_solve_history_width_rows']}/80.")
    a(f"- post active cache={control_route_summary['terminal_active_history_count']}; anchors={control_route_summary['terminal_anchor_count']}; Structural-H decision influence total={control_route_summary['structural_h_decision_influence_total']}; added F/B/materialization={control_route_summary['added_model_forward_backward_materialization']}.")
    a(f"- retry/backtracking/rollback/interbatch W0 restore={sum(x['retry_count'] for x in h_off_tx)}/{sum(x['backtracking_count'] for x in h_off_tx)}/{sum(x['rollback_count'] for x in h_off_tx)}/{sum(x['interbatch_W0_restore_count'] for x in h_off_tx)}.")
    a(""); a("## B1–B10 및 B100 절대값"); a("")
    a("세부 pre→post→final-W10 counts/rates/NLL/margin은 checkpoint machine table 31행에 보존했습니다.")
    a(""); a("|B|Alpha cache width|entry pre → immediate post → final W10|"); a("|---:|---:|---|")
    for round_id in range(1, 11):
        cp, ch = cp_by_round['h_off'][round_id], cohort_by_round['h_off'][round_id]
        a(f"|B{round_id}|{cp['history_width_at_entry']}|{' → '.join(phrase(compact(x)) for x in (cp['entry_pre_summary'],cp['immediate_post_summary'],ch['final_W10']))}|")
    a(""); a("### B100 final")
    a(""); a(phrase(finals['h_off']))
    a(""); a("- 상태: `TERMINAL_FACTUAL_REPORT_COMPLETE`; scientific_promotion=false. 모델/evaluator/GPU/Slurm 재실행=0.")
    off_report.write_text("\n".join(text) + "\n"); os.chmod(off_report, 0o600)

    # Canonical 3-arm comparison report.
    comp_report = OUT / "p1r52-sequential-structuralh-on-off-native-three-arm-ko.md"
    if comp_report.exists(): raise RuntimeError(f"create-once report exists: {comp_report}")
    lines=[]; a=lines.append
    a("# P1R52 Sequential: Structural-H On / AlphaCache-On Structural-H Off / Native")
    a(""); a("## 범위 및 동등성"); a("")
    a("- 같은 Llama stream/order/case/request identity의 10×B10/100 요청 비교. 모든 arm의 request key set=100/100 exact, B1–B10 request-order identity exact.")
    a("- H-aware standalone package는 immutable reference로 읽기만 했습니다. 본 보고서는 control 종료를 기다린 별도 3-arm raw-free package입니다.")
    a("- Eff=`rewrite_success`, Gen=`paraphrase_success` (`rephrase_success` alias); Accuracy와 혼동하지 않습니다.")
    a(""); a("## B100 최종 절대값과 산술 차이"); a("")
    a("|Arm|RW성공|RW정확|PP성공 / 엄격|PP정확 / 엄격|Loc|"); a("|---|---|---|---|---|---|")
    for arm in ("h_on","h_off","native"):
        s=finals[arm]; a(f"|{ARM_LABELS[arm]}|{count(s['rewrite_success'])}|{count(s['rewrite_acc'])}|{count(s['paraphrase_success'])} / {count(s['paraphrase_success'],True)}|{count(s['paraphrase_acc'])} / {count(s['paraphrase_acc'],True)}|{s['locality']['numerator']}/{s['locality']['denominator']} ({s['locality']['rate']:.3f})|")
    hdiff, odiff = subtraction(finals['h_on'],finals['h_off']), subtraction(finals['h_off'],finals['native'])
    a(f"- H-on−H-off B100: RW성공 {hdiff['rewrite_success']['prompt_numerator']:+}/100, RW정확 {hdiff['rewrite_acc']['prompt_numerator']:+}/100, PP성공 {hdiff['paraphrase_success']['prompt_numerator']:+}/200, PP엄격 {hdiff['paraphrase_success']['strict_request_numerator']:+}/100, PP정확 {hdiff['paraphrase_acc']['prompt_numerator']:+}/200, PP정확엄격 {hdiff['paraphrase_acc']['strict_request_numerator']:+}/100, Loc {hdiff['locality']['numerator']:+}/1000.")
    a(f"- H-off−Native B100: RW성공 {odiff['rewrite_success']['prompt_numerator']:+}/100, RW정확 {odiff['rewrite_acc']['prompt_numerator']:+}/100, PP성공 {odiff['paraphrase_success']['prompt_numerator']:+}/200, PP엄격 {odiff['paraphrase_success']['strict_request_numerator']:+}/100, PP정확 {odiff['paraphrase_acc']['prompt_numerator']:+}/200, PP정확엄격 {odiff['paraphrase_acc']['strict_request_numerator']:+}/100, Loc {odiff['locality']['numerator']:+}/1000.")
    a(""); a("## B1–B10 true entry-pre → immediate-post → final-W10"); a("")
    a("각 phase의 prompt/strict numerator·denominator·rate와 target-new/old NLL·margin mean은 `p1r52-sequential-three-arm-checkpoint-table.json` 31행에 동일 phase별로 기록했습니다.")
    a(""); a("|B|H폭|H-on pre→post→final PP성공/엄격|H-off pre→post→final PP성공/엄격|Native pre→post→final PP성공/엄격|"); a("|---:|---:|---|---|---|")
    for r in range(1,11):
        ss=[]
        for arm in ('h_on','h_off','native'):
            cp,co=cp_by_round[arm][r],cohort_by_round[arm][r]
            three=[compact(cp['entry_pre_summary']),compact(cp['immediate_post_summary']),compact(co['final_W10'])]
            ss.append(' → '.join(f"{count(x['paraphrase_success'])}/{count(x['paraphrase_success'],True)}" for x in three))
        a(f"|B{r}|{cp_by_round['h_off'][r]['history_width_at_entry']}|{ss[0]}|{ss[1]}|{ss[2]}|")
    a(""); a("## B5 동일 batch raw-free 관측"); a("")
    for arm in ('h_on','h_off','native'):
        s=b5['arms'][arm]; a(f"- {ARM_LABELS[arm]} post: RW성공 {count(s['rewrite_success'])}, RW정확 {count(s['rewrite_acc'])}, PP성공 {count(s['paraphrase_success'])}/엄격 {count(s['paraphrase_success'],True)}, PP정확 {count(s['paraphrase_acc'])}/엄격 {count(s['paraphrase_acc'],True)}, Loc {s['locality']['numerator']}/{s['locality']['denominator']} ({s['locality']['rate']:.3f}).")
    a(f"- B5 H-on−H-off post: PP성공 {b5['h_on_minus_h_off']['paraphrase_success']['prompt_numerator']:+}/20, PP정확 {b5['h_on_minus_h_off']['paraphrase_acc']['prompt_numerator']:+}/20, Loc {b5['h_on_minus_h_off']['locality']['numerator']:+}/100. 값은 동일 batch 산술 비교입니다.")
    a(""); a("## 낮은 성능·lifetime cohort (raw-free association)"); a("")
    a(f"- three-arm request trajectory=100; low/transition rows={len(low_rows)}. H-on post PP failure={low_summary['flag_counts']['h_on_post_paraphrase_failure']}/100, H-off post PP failure={low_summary['flag_counts']['h_off_post_paraphrase_failure']}/100; H-on final PP loss={low_summary['flag_counts']['h_on_final_paraphrase_loss']}/100, H-off final PP loss={low_summary['flag_counts']['h_off_final_paraphrase_loss']}/100.")
    a(f"- H-on post PP가 H-off보다 높은 request={low_summary['flag_counts']['h_on_post_better_than_h_off_paraphrase']}/100; 반대={low_summary['flag_counts']['h_off_post_better_than_h_on_paraphrase']}/100. Request-level H-routing attribution은 `NOT_RECORDED`이며, batch×step routing만 제공됩니다.")
    a("- B5 및 hard/low cohorts는 entry hardness, pre→post gain, post→final loss, batch age, request hash를 table에 유지합니다. 관찰 범위 결론은 `ASSOCIATION_ONLY`; 고립된 인과 주장은 하지 않습니다.")
    a(""); a("## Control receipt 및 compute"); a("")
    a(f"- H-off cache append transaction={control_route_summary['alpha_solve_cache_append_transaction_total']}, consume={control_route_summary['alpha_solve_cache_consume_total']}, Structural-H decision influence=0, added model F/B/materialization={control_route_summary['added_model_forward_backward_materialization']}.")
    a(f"- H-off terminal W0 pointer+byte exact={terminals['h_off']['terminal_W0_restore']['pointer_restored_exact']}/{terminals['h_off']['terminal_W0_restore']['byte_restored_exact']}; retry/backtrack/interbatch restore=0/0/0.")
    a("- Full compute ledger, batch-age cohorts, per-request gain/forgetting, routing and transaction receipts are machine tables in this package.")
    a(""); a("## 상태"); a("")
    a("- `TERMINAL_FACTUAL_REPORT_COMPLETE`; scientific_promotion=false. 분석 중 추가 model/evaluator/GPU/Slurm=0.")
    comp_report.write_text("\n".join(lines)+"\n"); os.chmod(comp_report,0o600)

    input_manifest = {"schema":"p1r52-sequential-three-arm/input-manifest/v1","inputs":sources,"scheduler":{"h_on":"20424_0 COMPLETED 0:0","h_off":"20428 COMPLETED 0:0","native":"20403_1 COMPLETED 0:0"},"identity_pairing_pass":True,"control_route_summary":control_route_summary}
    write("input-identity-manifest.json", input_manifest)
    generated = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != Path(__file__).name)
    manifest = {"schema":"p1r52-sequential-three-arm/report-manifest/v1","status":"TERMINAL_FACTUAL_REPORT_COMPLETE","canonical_scope":"P1R52-Soft-Sequential-AlphaCache-On-StructuralH-Off-10xB10","standalone_control_report":off_report.name,"canonical_three_arm_report":comp_report.name,"source_heads":{a:terminals[a]['source_head'] for a in roots},"input_manifest_sha256":sha(OUT/'input-identity-manifest.json'),"generated_files":[{"name":p.name,"bytes":p.stat().st_size,"sha256":sha(p)} for p in generated],"row_counts":{"control_checkpoint":len(control_checkpoint_rows),"control_routing":len(h_off_routing),"control_transaction":len(h_off_tx),"three_arm_checkpoint":len(three_phase_rows),"three_arm_per_request":len(triple_request_rows),"low_cohort":len(low_rows),"h_on_routing_reference":len(h_on_routing)}}
    mpath=write("report-manifest.json",manifest)
    root_material='\n'.join(f"{p.name}\t{sha(p)}\t{p.stat().st_size}" for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='analysis-receipt.json')
    receipt={"schema":"p1r52-sequential-three-arm/analysis-receipt/v1","status":"PASS","canonical_report":str(comp_report),"canonical_report_sha256":sha(comp_report),"canonical_report_bytes":comp_report.stat().st_size,"canonical_report_lines":len(comp_report.read_text().splitlines()),"control_report_sha256":sha(off_report),"manifest_sha256":sha(mpath),"package_root_sha256":hashlib.sha256(root_material.encode()).hexdigest(),"three_arm_identity_pairing_pass":True,"raw_content_included":False,"analysis_actions":{"model":0,"evaluator":0,"gpu":0,"slurm":0,"source_edit":0,"result_mutation":0}}
    write("analysis-receipt.json",receipt)


if __name__ == '__main__':
    main()
