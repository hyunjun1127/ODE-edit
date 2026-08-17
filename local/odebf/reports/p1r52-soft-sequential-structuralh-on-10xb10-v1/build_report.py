#!/usr/bin/env python3
"""Create a raw-free standalone report for the completed Structural-H-on arm.

This is a report-only transformer.  It reads sealed result receipts and writes
only derived raw-free tables into its own create-once report directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[4]
H_ROOT = ROOT / "local/odebf/results/s05-p1r52-llama-soft-sequential-historical-10xb10-tech-r3-v1"
N_ROOT = ROOT / "local/odebf/results/s05-p1r52-native-alphaedit-sequential-10xb10-v1"
OUT = Path(__file__).resolve().parent
METRICS = ("rewrite_success", "rewrite_acc", "paraphrase_success", "paraphrase_acc")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(name: str, payload) -> Path:
    path = OUT / name
    if path.exists():
        raise RuntimeError(f"create-once output already exists: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.chmod(path, 0o600)
    return path


def compact(summary: dict) -> dict:
    """Return only report-safe absolute counters and continuous NLL/margin."""
    result = {"identity_sha256": summary.get("identity_sha256"), "request_count": summary.get("request_count")}
    for name in METRICS:
        item = summary["metrics"][name]
        result[name] = {
            "prompt_numerator": item["prompt_numerator"],
            "prompt_denominator": item["prompt_denominator"],
            "prompt_rate": item["prompt_rate"],
            "strict_request_numerator": item["strict_request_numerator"],
            "strict_request_denominator": item["strict_request_denominator"],
            "strict_request_rate": item["strict_request_rate"],
            "target_new_nll_mean": item["target_new_nll"]["mean"],
            "target_old_nll_mean": item["target_old_nll"]["mean"],
            "target_old_minus_new_margin_mean": item["target_old_minus_new_margin"]["mean"],
        }
    loc = summary.get("locality", {})
    result["locality"] = {
        "numerator": loc.get("numerator"),
        "denominator": loc.get("denominator"),
        "rate": loc.get("rate"),
    }
    return result


def delta(a: dict, b: dict) -> dict:
    """Arithmetic H-minus-Native delta for compact summaries."""
    result = {}
    for name in METRICS:
        result[name] = {}
        for field in (
            "prompt_numerator", "prompt_rate", "strict_request_numerator",
            "strict_request_rate", "target_new_nll_mean",
            "target_old_nll_mean", "target_old_minus_new_margin_mean",
        ):
            av, bv = a[name].get(field), b[name].get(field)
            result[name][field] = None if av is None or bv is None else av - bv
    for field in ("numerator", "rate"):
        av, bv = a["locality"].get(field), b["locality"].get(field)
        result.setdefault("locality", {})[field] = None if av is None or bv is None else av - bv
    return result


def fmt_count(item: dict, strict: bool = False) -> str:
    prefix = "strict_request_" if strict else "prompt_"
    n, d, r = item[f"{prefix}numerator"], item[f"{prefix}denominator"], item[f"{prefix}rate"]
    return f"{n}/{d} ({r:.3f})"


def fmt_phase(c: dict) -> str:
    return (
        f"RW성공 {fmt_count(c['rewrite_success'])}; RW정확 {fmt_count(c['rewrite_acc'])}; "
        f"PP성공 {fmt_count(c['paraphrase_success'])}/엄격 {fmt_count(c['paraphrase_success'], True)}; "
        f"PP정확 {fmt_count(c['paraphrase_acc'])}/엄격 {fmt_count(c['paraphrase_acc'], True)}; "
        f"Loc {c['locality']['numerator']}/{c['locality']['denominator']} ({c['locality']['rate']:.3f})"
    )


def success_bit(item: dict, metric: str) -> int:
    if metric == "locality":
        return int(item[metric]["correct"] == item[metric]["required"])
    return int(item[metric]["correct"] == item[metric]["required"])


def metric_record(item: dict, metric: str) -> dict:
    if metric == "locality":
        return {
            "correct": item[metric]["correct"], "required": item[metric]["required"],
            "rate": item[metric]["rate"], "strict_all_prompts_pass": None,
        }
    return {
        "correct": item[metric]["correct"], "required": item[metric]["required"],
        "rate": item[metric]["rate"],
        "strict_all_prompts_pass": item[metric]["strict_all_prompts_pass"],
        "target_new_nll_mean": item[metric]["target_new_nll_mean"],
        "target_old_nll_mean": item[metric]["target_old_nll_mean"],
        "target_old_minus_new_margin_mean": item[metric]["target_old_minus_new_margin_mean"],
    }


def main() -> None:
    h_manifest = read(H_ROOT / "manifest.json")
    n_manifest = read(N_ROOT / "manifest.json")
    h_terminal = read(H_ROOT / "terminal.json")
    n_terminal = read(N_ROOT / "terminal.json")
    h_cp = read(H_ROOT / "b1-b10-pre-post-checkpoints.json")["rows"]
    n_cp = read(N_ROOT / "b1-b10-pre-post-checkpoints.json")["rows"]
    h_cohort = read(H_ROOT / "batch-age-pre-post-final-cohorts.json")["rows"]
    n_cohort = read(N_ROOT / "batch-age-pre-post-final-cohorts.json")["rows"]
    h_req = read(H_ROOT / "entry-pre-immediate-post-final-w10-requests.json")["rows"]
    n_req = read(N_ROOT / "entry-pre-immediate-post-final-w10-requests.json")["rows"]

    if not (len(h_cp) == len(n_cp) == len(h_cohort) == len(n_cohort) == 10):
        raise RuntimeError("checkpoint/cohort rows are not ten-per-arm")
    if not (len(h_req) == len(n_req) == 100):
        raise RuntimeError("request rows are not 100-per-arm")

    # Exact case/request-order pairing for the immutable same-stream comparison.
    h_keys = {(r["round"], r["case_id"], r["request_index"], r["request_sha256"]) for r in h_req}
    n_keys = {(r["round"], r["case_id"], r["request_index"], r["request_sha256"]) for r in n_req}
    if h_keys != n_keys:
        raise RuntimeError("Native request identities do not exactly match Structural-H-on")
    for hr, nr in zip(sorted(h_cp, key=lambda x: x["round"]), sorted(n_cp, key=lambda x: x["round"])):
        if hr["round"] != nr["round"]:
            raise RuntimeError("checkpoint round mismatch")
        if hr["entry_pre_summary"]["request_order_sha256"] != nr["entry_pre_summary"]["request_order_sha256"]:
            raise RuntimeError("checkpoint request-order mismatch")

    # Normalized checkpoint table: three true phases per B10 and B100 final only.
    checkpoint_rows = []
    by_round_h = {r["round"]: r for r in h_cp}
    by_round_n = {r["round"]: r for r in n_cp}
    cohort_h = {r["round"]: r for r in h_cohort}
    cohort_n = {r["round"]: r for r in n_cohort}
    for arm, cps, cohorts in (("P1R52_StructuralH_On", by_round_h, cohort_h), ("Native_AlphaEdit", by_round_n, cohort_n)):
        for round_id in range(1, 11):
            cp, cohort = cps[round_id], cohorts[round_id]
            for phase, summary in (
                ("entry_pre", cp["entry_pre_summary"]),
                ("immediate_post", cp["immediate_post_summary"]),
                ("final_W10", cohort["final_W10"]),
            ):
                checkpoint_rows.append({
                    "arm": arm,
                    "checkpoint": f"B{round_id}",
                    "round": round_id,
                    "phase": phase,
                    "history_width_at_entry": cp["history_width_at_entry"],
                    "batch_age_at_W10": cohort["batch_age_at_W10"],
                    "summary": compact(summary),
                })
        checkpoint_rows.append({
            "arm": arm, "checkpoint": "B100", "round": None,
            "phase": "final_W10_B100", "history_width_at_entry": None,
            "batch_age_at_W10": None, "summary": compact((h_terminal if arm.startswith("P1R52") else n_terminal)["final_B100"]),
        })

    # Same-batch comparison, keeping all absolute values beside deltas.
    paired_batch_rows = []
    for round_id in range(1, 11):
        hc, nc, hh, nh = by_round_h[round_id], by_round_n[round_id], cohort_h[round_id], cohort_n[round_id]
        phases = (
            ("entry_pre", hc["entry_pre_summary"], nc["entry_pre_summary"]),
            ("immediate_post", hc["immediate_post_summary"], nc["immediate_post_summary"]),
            ("final_W10", hh["final_W10"], nh["final_W10"]),
        )
        for phase, hs, ns in phases:
            ch, cn = compact(hs), compact(ns)
            paired_batch_rows.append({
                "round": round_id,
                "checkpoint": f"B{round_id}",
                "phase": phase,
                "history_width_at_entry": hc["history_width_at_entry"],
                "batch_age_at_W10": hh["batch_age_at_W10"],
                "structural_h_on": ch,
                "native_alphaedit": cn,
                "h_minus_native": delta(ch, cn),
            })
    h100, n100 = compact(h_terminal["final_B100"]), compact(n_terminal["final_B100"])
    paired_batch_rows.append({"round": None, "checkpoint": "B100", "phase": "final_W10_B100", "history_width_at_entry": None, "batch_age_at_W10": None, "structural_h_on": h100, "native_alphaedit": n100, "h_minus_native": delta(h100, n100)})

    # Per-request true entry→post→final trajectory plus immutable Native pairing.
    n_req_map = {(r["round"], r["case_id"], r["request_index"], r["request_sha256"]): r for r in n_req}
    per_request_rows = []
    failure_rows = []
    for hr in sorted(h_req, key=lambda r: (r["round"], r["request_index"])):
        key = (hr["round"], hr["case_id"], hr["request_index"], hr["request_sha256"])
        nr = n_req_map[key]
        phases = {phase: {m: metric_record(hr[phase], m) for m in (*METRICS, "locality")} for phase in ("entry_pre", "immediate_post", "final_W10")}
        native_phases = {phase: {m: metric_record(nr[phase], m) for m in (*METRICS, "locality")} for phase in ("entry_pre", "immediate_post", "final_W10")}
        d = {}
        for phase in phases:
            d[phase] = {}
            for m in (*METRICS, "locality"):
                d[phase][m] = {
                    "correct": phases[phase][m]["correct"] - native_phases[phase][m]["correct"],
                    "rate": phases[phase][m]["rate"] - native_phases[phase][m]["rate"],
                }
        row = {
            "round": hr["round"], "case_id": hr["case_id"], "request_index": hr["request_index"],
            "request_sha256": hr["request_sha256"], "history_width_at_entry": hr["history_width_at_entry"],
            "structural_h_on": phases, "native_alphaedit": native_phases, "h_minus_native": d,
            "lifetime_deltas_structural_h_on": hr["deltas"],
        }
        per_request_rows.append(row)
        # Raw-free low-performance cohort membership (not a causal attribution).
        post = phases["immediate_post"]
        final = phases["final_W10"]
        pre = phases["entry_pre"]
        flags = {
            "post_rewrite_success_failure": post["rewrite_success"]["correct"] < post["rewrite_success"]["required"],
            "post_rewrite_accuracy_failure": post["rewrite_acc"]["correct"] < post["rewrite_acc"]["required"],
            "post_paraphrase_success_failure": post["paraphrase_success"]["correct"] < post["paraphrase_success"]["required"],
            "post_paraphrase_accuracy_failure": post["paraphrase_acc"]["correct"] < post["paraphrase_acc"]["required"],
            "post_strict_paraphrase_success_failure": post["paraphrase_success"]["strict_all_prompts_pass"] == 0,
            "post_strict_paraphrase_accuracy_failure": post["paraphrase_acc"]["strict_all_prompts_pass"] == 0,
            "final_paraphrase_success_loss": final["paraphrase_success"]["correct"] < post["paraphrase_success"]["correct"],
            "final_paraphrase_accuracy_loss": final["paraphrase_acc"]["correct"] < post["paraphrase_acc"]["correct"],
            "native_post_paraphrase_success_higher": native_phases["immediate_post"]["paraphrase_success"]["correct"] > post["paraphrase_success"]["correct"],
            "entry_hard_by_rewrite_success": pre["rewrite_success"]["correct"] < pre["rewrite_success"]["required"],
        }
        if any(flags.values()):
            failure_rows.append({
                "round": hr["round"], "case_id": hr["case_id"], "request_index": hr["request_index"],
                "request_sha256": hr["request_sha256"], "history_width_at_entry": hr["history_width_at_entry"],
                "flags": flags, "structural_h_on": phases, "native_alphaedit": native_phases,
                "h_minus_native": d,
            })

    # Routing / transaction ledger as independent raw-free evidence.
    routing_rows, transaction_rows = [], []
    for round_id in range(1, 11):
        hbt = read(H_ROOT / f"raw/batches/b{round_id:02d}/terminal.json")
        for item in hbt["structural_h_routing"]:
            routing_rows.append({
                "round": round_id,
                "step_index": item["step_index"],
                "history_width": item["history_width"], "status": item["status"],
                "h_neutral": item["h_neutral"], "h_selected": item["h_selected"], "h_delta": item["h_delta"],
                "strength_residual": item["strength_residual"], "energy_violation": item["energy_violation"],
                "p_violation": item["p_violation"], "allocation_entropy": item["allocation_entropy"],
                "selected_minus_neutral_norm": item["selected_minus_neutral_norm"],
                "maximum_layer_velocity": item["maximum_layer_velocity"], "identity_sha256": item["identity_sha256"],
            })
        tx = hbt["history_transaction"]
        wt = hbt["weight_transaction"]
        transaction_rows.append({
            "round": round_id,
            "history_width_at_entry": hbt["history_width_at_entry"],
            "active_history_count_after": hbt["active_history_count"],
            "anchor_count_after": hbt["lifetime_anchor_count"],
            "history_append_count": tx["appended_count"],
            "history_before_version": tx["before_version"], "history_after_version": tx["after_version"],
            "history_idempotent_replay": tx["idempotent_replay"], "history_obsolete_count": tx["obsolete_count"],
            "weight_commit_count": wt["commit_count"], "weight_post_commit_verified": wt["post_commit_verified"],
            "weight_rollback_count": wt["rollback_count"], "retry_count": hbt["retry_count"],
            "backtracking_count": hbt["backtracking_count"], "interbatch_W0_restore_count": hbt["interbatch_W0_restore_count"],
            "heldout_controller_influence_count": hbt["heldout_controller_influence_count"],
            "entry_weight_sha256": hbt["entry_weight_sha256"], "commit_weight_sha256": hbt["commit_weight_sha256"],
            "history_transaction_id": tx["transaction_id"],
        })

    h_active = [r for r in routing_rows if r["status"] == "H_ACTIVE_CERTIFIED"]
    h_empty = [r for r in routing_rows if r["status"] != "H_ACTIVE_CERTIFIED"]
    routing_summary = {
        "routing_receipt_count": len(routing_rows),
        "structural_h_active_certified_count": len(h_active),
        "structural_h_nonactive_count": len(h_empty),
        "selected_receipt_count": len(routing_rows),
        "max_abs_strength_residual": max(abs(r["strength_residual"]) for r in routing_rows),
        "max_energy_violation": max(r["energy_violation"] for r in routing_rows),
        "max_abs_p_violation": max(abs(r["p_violation"]) for r in routing_rows),
        "h_delta_sum": sum(r["h_delta"] for r in routing_rows),
        "h_delta_min": min(r["h_delta"] for r in routing_rows),
        "h_delta_max": max(r["h_delta"] for r in routing_rows),
        "nonzero_selected_minus_neutral_count": sum(r["selected_minus_neutral_norm"] > 0 for r in routing_rows),
        "history_width_positive_routing_receipt_count": sum(r["history_width"] > 0 for r in routing_rows),
        "history_width_sum_over_routing_receipts": sum(r["history_width"] for r in routing_rows),
        "alpha_solve_cache_append_count": sum(r["history_append_count"] for r in transaction_rows),
        "alpha_solve_cache_consume_explicit_counter": "NOT_RECORDED",
        "alpha_solve_cache_consume_receipt_proxy": "history_width_positive_routing_receipt_count",
        "terminal_active_history_count": h_terminal["terminal_active_history_count"],
        "terminal_lifetime_anchor_count": h_terminal["terminal_lifetime_anchor_count"],
    }

    # Compute only from existing receipts.  No evaluator/model work is performed here.
    def add_numbers(dst, src):
        for key, value in src.items():
            if isinstance(value, (int, float)):
                dst[key] = dst.get(key, 0) + value
    h_batch_compute = {}
    h_batch_walls = {}
    for round_id in range(1, 11):
        terminal = read(H_ROOT / f"raw/batches/b{round_id:02d}/terminal.json")
        add_numbers(h_batch_compute, terminal["atomic_or_native"]["compute"]["totals"])
        add_numbers(h_batch_walls, terminal["atomic_or_native"]["compute"]["wall_seconds"])
    n_edit_core_seconds = sum(read(N_ROOT / f"raw/batches/b{round_id:02d}/terminal.json")["atomic_or_native"]["edit_core_wall_seconds"] for round_id in range(1, 11))
    compute_payload = {
        "structural_h_on_batch_atomic_compute_totals_sum": h_batch_compute,
        "structural_h_on_batch_atomic_wall_seconds_sum": h_batch_walls,
        "structural_h_on_terminal_job_compute": h_terminal["job_compute"],
        "native_terminal_job_compute": n_terminal["job_compute"],
        "native_edit_core_wall_seconds_sum": n_edit_core_seconds,
        "one_pass_accuracy_added_model_forward_backward_generation": [
            h_terminal["accuracy_amendment_added_model_forward_count"],
            h_terminal["accuracy_amendment_added_backward_count"],
            h_terminal["accuracy_amendment_added_generation_count"],
        ],
        "entry_pre_evaluator_model_forward_backward_generation": [
            h_terminal["batch_entry_pre_evaluator_forward_count"],
            h_terminal["batch_entry_pre_evaluator_backward_count"],
            h_terminal["batch_entry_pre_evaluator_generation_count"],
        ],
        "inner_K_step_heldout_evaluation_count": h_terminal["inner_K_step_heldout_evaluation_count"],
        "evaluator_controller_influence_count": h_terminal["evaluator_controller_influence_count"],
    }

    # Observational B5/hard-cohort summary only, no causal attribution.
    b5 = next(r for r in paired_batch_rows if r["round"] == 5 and r["phase"] == "immediate_post")
    def count_flag(name: str) -> int:
        return sum(bool(r["flags"].get(name)) for r in failure_rows)
    low_summary = {
        "association_scope": "RAW_FREE_OBSERVATIONAL_ONLY_NO_ISOLATED_CAUSAL_ATTRIBUTION",
        "per_request_rows": 100,
        "low_or_failed_request_rows": len(failure_rows),
        "flag_counts": {name: count_flag(name) for name in (
            "post_rewrite_success_failure", "post_rewrite_accuracy_failure", "post_paraphrase_success_failure",
            "post_paraphrase_accuracy_failure", "post_strict_paraphrase_success_failure",
            "post_strict_paraphrase_accuracy_failure", "final_paraphrase_success_loss",
            "final_paraphrase_accuracy_loss", "native_post_paraphrase_success_higher", "entry_hard_by_rewrite_success",
        )},
        "B5_immediate_post_absolute": {
            "structural_h_on": b5["structural_h_on"], "native_alphaedit": b5["native_alphaedit"], "h_minus_native": b5["h_minus_native"],
        },
        "routing_data_granularity": "batch_step_only; requestwise H-routing attribution=NOT_RECORDED",
        "conclusion_label": "ASSOCIATION_ONLY",
    }

    input_files = [
        H_ROOT / "manifest.json", H_ROOT / "terminal.json", H_ROOT / "b1-b10-pre-post-checkpoints.json",
        H_ROOT / "batch-age-pre-post-final-cohorts.json", H_ROOT / "entry-pre-immediate-post-final-w10-requests.json",
        N_ROOT / "manifest.json", N_ROOT / "terminal.json", N_ROOT / "b1-b10-pre-post-checkpoints.json",
        N_ROOT / "batch-age-pre-post-final-cohorts.json", N_ROOT / "entry-pre-immediate-post-final-w10-requests.json",
    ] + [H_ROOT / f"raw/batches/b{i:02d}/terminal.json" for i in range(1, 11)] + [N_ROOT / f"raw/batches/b{i:02d}/terminal.json" for i in range(1, 11)]
    inputs = [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)} for p in input_files]

    # Machine-readable outputs.
    write("p1r52-structuralh-on-checkpoint-phase-table.json", {"schema": "p1r52-structuralh-on/checkpoint-phase/v1", "row_count": len(checkpoint_rows), "rows": checkpoint_rows})
    write("p1r52-structuralh-on-paired-native-checkpoint-table.json", {"schema": "p1r52-structuralh-on/paired-native-checkpoint/v1", "row_count": len(paired_batch_rows), "rows": paired_batch_rows})
    write("p1r52-structuralh-on-per-request-lifetime-table.json", {"schema": "p1r52-structuralh-on/per-request-lifetime/v1", "row_count": len(per_request_rows), "rows": per_request_rows})
    write("p1r52-structuralh-on-low-cohort-table.json", {"schema": "p1r52-structuralh-on/low-cohort/v1", "row_count": len(failure_rows), "rows": failure_rows, "summary": low_summary})
    write("p1r52-structuralh-on-routing-history-table.json", {"schema": "p1r52-structuralh-on/routing-history/v1", "row_count": len(routing_rows), "rows": routing_rows, "summary": routing_summary})
    write("p1r52-structuralh-on-transaction-table.json", {"schema": "p1r52-structuralh-on/transaction/v1", "row_count": len(transaction_rows), "rows": transaction_rows})
    write("p1r52-structuralh-on-compute-ledger.json", {"schema": "p1r52-structuralh-on/compute/v1", "payload": compute_payload})

    # Korean standalone factual report.  Deltas are always adjacent to absolute values.
    lines = []
    a = lines.append
    a("# P1R52-Soft-Sequential-StructuralH-On-10xB10")
    a("")
    a("## 독립 종료 사실")
    a("")
    a("- 범위: Llama3-8B-Instruct, P1R52 Repair-R1 Soft, Structural-H On, 순차 10×B10/100 요청.")
    a("- 스케줄러: job 20424_0 `COMPLETED`, exit `0:0`, elapsed `00:36:26`.")
    a(f"- 소스: `{h_terminal['source_head']}`; 방법: `{h_terminal['method_id']}`; 결과 단계: 10/10, 요청: 100/100.")
    a("- Native 비교는 immutable job 20403_1 (`COMPLETED`, exit `0:0`)의 같은 stream/order/request identity만 사용했고 재실행은 0회입니다.")
    a("- Eff는 `rewrite_success`, Gen은 `paraphrase_success`(사용자 표기 `rephrase_success`)입니다. Accuracy와 동의어가 아닙니다.")
    a("- H-off control은 후속 별도 비교 대상이며, 이 독립 보고서의 입력 또는 완료 조건이 아닙니다.")
    a("")
    a("## 무결성·순차 상태")
    a("")
    a(f"- W0: inter-batch restore={h_terminal['interbatch_W0_restore_count']}, terminal restore={h_terminal['terminal_W0_restore_count']}, pointer+byte exact={h_terminal['terminal_W0_restore']['pointer_restored_exact']}/{h_terminal['terminal_W0_restore']['byte_restored_exact']}.")
    a(f"- history entry widths: {h_terminal['history_widths']}; terminal active records={h_terminal['terminal_active_history_count']}; lifetime anchors={h_terminal['terminal_lifetime_anchor_count']}.")
    a(f"- history transaction append 합계={routing_summary['alpha_solve_cache_append_count']}; positive-history routing receipt={routing_summary['history_width_positive_routing_receipt_count']}/80; history-width 합={routing_summary['history_width_sum_over_routing_receipts']}.")
    a("- 명시적인 `alpha_solve_cache_consume_count` 필드는 sealed 입력에 없음(`NOT_RECORDED`); 대신 각 routing receipt의 nonzero history-width를 별도 보존했습니다.")
    a(f"- rollback={sum(r['weight_rollback_count'] for r in transaction_rows)}, retry={sum(r['retry_count'] for r in transaction_rows)}, backtracking={sum(r['backtracking_count'] for r in transaction_rows)}, heldout controller influence={sum(r['heldout_controller_influence_count'] for r in transaction_rows)}.")
    a(f"- Structural-H routing: active/certified={routing_summary['structural_h_active_certified_count']}, non-active(B1 empty)={routing_summary['structural_h_nonactive_count']}, selected receipts={routing_summary['selected_receipt_count']}, max |strength residual|={routing_summary['max_abs_strength_residual']:.3e}, max energy violation={routing_summary['max_energy_violation']:.3e}, max |P violation|={routing_summary['max_abs_p_violation']:.3e}.")
    a("")
    a("## B1–B10 절대 pre → post → final-W10 표")
    a("")
    a("표기: RW=rewite, PP=paraphrase/rephrase. `엄격`은 request 단위 모든 paraphrase 통과. 각 칸은 `분자/분모 (율)`입니다.")
    a("")
    a("|B|H폭|H-aware entry-pre → post → final-W10|Native entry-pre → post → final-W10|")
    a("|---:|---:|---|---|")
    for round_id in range(1, 11):
        hc, nc, hh, nh = by_round_h[round_id], by_round_n[round_id], cohort_h[round_id], cohort_n[round_id]
        htxt = " → ".join(fmt_phase(compact(x)) for x in (hc["entry_pre_summary"], hc["immediate_post_summary"], hh["final_W10"]))
        ntxt = " → ".join(fmt_phase(compact(x)) for x in (nc["entry_pre_summary"], nc["immediate_post_summary"], nh["final_W10"]))
        a(f"|B{round_id}|{hc['history_width_at_entry']}|{htxt}|{ntxt}|")
    a("")
    a("## B100 final-W10 절대값 및 Native 차이")
    a("")
    a("|Arm|RW성공|RW정확|PP성공 / 엄격|PP정확 / 엄격|Loc|")
    a("|---|---|---|---|---|---|")
    for label, s in (("Structural-H On", h100), ("Native AlphaEdit", n100)):
        a(f"|{label}|{fmt_count(s['rewrite_success'])}|{fmt_count(s['rewrite_acc'])}|{fmt_count(s['paraphrase_success'])} / {fmt_count(s['paraphrase_success'], True)}|{fmt_count(s['paraphrase_acc'])} / {fmt_count(s['paraphrase_acc'], True)}|{s['locality']['numerator']}/{s['locality']['denominator']} ({s['locality']['rate']:.3f})|")
    b100d = delta(h100, n100)
    a(f"- H-aware−Native (B100): rewrite success {b100d['rewrite_success']['prompt_numerator']:+}/100, rewrite acc {b100d['rewrite_acc']['prompt_numerator']:+}/100, paraphrase success {b100d['paraphrase_success']['prompt_numerator']:+}/200, strict success {b100d['paraphrase_success']['strict_request_numerator']:+}/100, paraphrase acc {b100d['paraphrase_acc']['prompt_numerator']:+}/200, strict acc {b100d['paraphrase_acc']['strict_request_numerator']:+}/100, locality {b100d['locality']['numerator']:+}/1000.")
    a("")
    a("## B5 동일 배치 관측")
    a("")
    a("- Structural-H On B5 pre→post: rewrite success 1/10→10/10, rewrite acc 0/10→10/10, paraphrase success 2/20→14/20 (엄격 1/10→5/10), paraphrase acc 0/20→11/20 (엄격 0/10→3/10), locality 93/100→92/100.")
    a("- Native B5 pre→post: rewrite success 1/10→10/10, rewrite acc 0/10→10/10, paraphrase success 2/20→18/20 (엄격 1/10→8/10), paraphrase acc 1/20→14/20 (엄격 0/10→5/10), locality 92/100→82/100.")
    a("- 같은 post 시점 H-aware−Native: paraphrase success −4/20, paraphrase acc −3/20, locality +10/100. 이 값은 sealed 동일 배치 산술 비교이며 단일 원인의 인과 판정은 하지 않습니다.")
    a("")
    a("## History·anchor·lifetime 및 낮은 성능 cohort")
    a("")
    a(f"- final W10 batch-age cohorts=10, request lifetime rows=100, low/failed-observation rows={len(failure_rows)}.")
    a(f"- final post→W10 paraphrase success loss flag={low_summary['flag_counts']['final_paraphrase_success_loss']}/100, paraphrase accuracy loss flag={low_summary['flag_counts']['final_paraphrase_accuracy_loss']}/100, Native post paraphrase-success higher flag={low_summary['flag_counts']['native_post_paraphrase_success_higher']}/100.")
    a(f"- post strict paraphrase success fail={low_summary['flag_counts']['post_strict_paraphrase_success_failure']}/100, post strict paraphrase accuracy fail={low_summary['flag_counts']['post_strict_paraphrase_accuracy_failure']}/100, entry rewrite-success hard flag={low_summary['flag_counts']['entry_hard_by_rewrite_success']}/100.")
    a("- history age, entry hardness, immediate gain, final loss 및 request hash는 per-request/batch-age machine table에 보존했습니다. Routing은 batch×step receipt만 있으므로 request별 H-routing 귀속은 `NOT_RECORDED`입니다.")
    a("- 관찰 범위 결론: `ASSOCIATION_ONLY` (원시 raw-free receipt 기반 cohort 동시 발생). 독립된 Structural-H 인과 주장은 하지 않습니다.")
    a("")
    a("## Compute ledger")
    a("")
    a(f"- H-aware 10 batch atomic 합: model forwards={h_batch_compute.get('model_forward_calls')}, backward={h_batch_compute.get('backward_calls')}, materializations={h_batch_compute.get('materialization_count')}, processed tokens={h_batch_compute.get('processed_tokens')}.")
    a(f"- evaluator: entry-pre forward={h_terminal['batch_entry_pre_evaluator_forward_count']}, backward={h_terminal['batch_entry_pre_evaluator_backward_count']}, generation={h_terminal['batch_entry_pre_evaluator_generation_count']}; accuracy amendment 추가 forward/backward/generation={h_terminal['accuracy_amendment_added_model_forward_count']}/{h_terminal['accuracy_amendment_added_backward_count']}/{h_terminal['accuracy_amendment_added_generation_count']}; inner-K heldout eval={h_terminal['inner_K_step_heldout_evaluation_count']}.")
    a(f"- Native edit-core wall 합={n_edit_core_seconds:.6f}s. 전체 세부 ledger는 `p1r52-structuralh-on-compute-ledger.json`에 보존했습니다.")
    a("")
    a("## 범위·상태")
    a("")
    a("- 상태: `TERMINAL_FACTUAL_REPORT_COMPLETE`; scientific_promotion=false.")
    a("- 이 보고서는 H-aware Structural-H On 단독 결과입니다. AlphaCache-On/StructuralH-Off control의 결과는 포함하지 않았습니다.")
    a("- 별도 모델·evaluator·GPU·Slurm 실행은 이 분석에서 0회입니다.")
    report = OUT / "p1r52-soft-sequential-structuralh-on-10xb10-factual-ko.md"
    if report.exists():
        raise RuntimeError(f"create-once report exists: {report}")
    report.write_text("\n".join(lines) + "\n")
    os.chmod(report, 0o600)

    # Manifest follows generated content (not itself), then a receipt binds its root.
    generated = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != Path(__file__).name)
    manifest = {
        "schema": "p1r52-structuralh-on/report-manifest/v1",
        "canonical_scope": "P1R52-Soft-Sequential-StructuralH-On-10xB10",
        "status": "TERMINAL_FACTUAL_REPORT_COMPLETE",
        "scientific_promotion": False,
        "source_head": h_terminal["source_head"],
        "native_source_head": n_terminal["source_head"],
        "scheduler": {"structural_h_job": "20424_0", "native_job": "20403_1", "structural_h_exit": "0:0", "native_exit": "0:0"},
        "input_files": inputs,
        "generated_files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in generated],
        "row_counts": {"checkpoint_phase": len(checkpoint_rows), "paired_checkpoint": len(paired_batch_rows), "per_request_lifetime": len(per_request_rows), "low_cohort": len(failure_rows), "routing_history": len(routing_rows), "transaction": len(transaction_rows)},
        "routing_summary": routing_summary,
    }
    manifest_path = write("p1r52-structuralh-on-report-manifest.json", manifest)
    root_material = "\n".join(f"{p.name}\t{sha(p)}\t{p.stat().st_size}" for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "analysis-receipt.json")
    receipt = {
        "schema": "p1r52-structuralh-on/analysis-receipt/v1",
        "status": "PASS",
        "analysis_actions": {"model": 0, "evaluator": 0, "gpu": 0, "slurm": 0, "source_edit": 0, "result_mutation": 0},
        "manifest_sha256": sha(manifest_path),
        "package_root_sha256": hashlib.sha256(root_material.encode()).hexdigest(),
        "report_sha256": sha(report),
        "report_bytes": report.stat().st_size,
        "report_lines": len(report.read_text().splitlines()),
        "input_identity_pass": True,
        "request_pairing_pass": True,
        "raw_content_included": False,
        "h_off_control_status": "PENDING_FOLLOWUP_NOT_A_REPORT_BLOCKER",
    }
    write("analysis-receipt.json", receipt)


if __name__ == "__main__":
    main()
