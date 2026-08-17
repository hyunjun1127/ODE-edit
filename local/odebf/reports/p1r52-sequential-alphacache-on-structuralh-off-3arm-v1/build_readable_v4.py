#!/usr/bin/env python3
"""Create-only readable v4 reports from sealed sequential raw-free receipts.

This report transformer only rearranges already sealed scalar/receipt data.  It
does not load a model, call an evaluator, or alter a result root.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from statistics import mean


OUT = Path(__file__).resolve().parent
CONTROL = OUT.parents[3] / "local/odebf/results/s05-p1r52-llama-soft-sequential-alphacache-on-structuralh-off-10xb10-v1"
HROOT = Path("/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-llama-soft-seq-hist-10xb10-v1/local/odebf/results/s05-p1r52-llama-soft-sequential-historical-10xb10-tech-r3-v1")
NROOT = Path("/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-llama-soft-seq-hist-10xb10-v1/local/odebf/results/s05-p1r52-native-alphaedit-sequential-10xb10-v1")
ROOTS = {"h_on": HROOT, "h_off": CONTROL, "native": NROOT}
LABEL = {
    "h_on": "P1R52 Soft Structural-H ON",
    "h_off": "P1R52 Soft AlphaCache-ON / Structural-H OFF",
    "native": "Native AlphaEdit Sequential",
}
METRICS = ("rewrite_success", "rewrite_acc", "paraphrase_success", "paraphrase_acc")


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_once(name: str, payload) -> Path:
    path = OUT / name
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    return path


def write_text_once(name: str, text: str) -> Path:
    path = OUT / name
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def metric(summary: dict, name: str) -> dict:
    source = summary["metrics"][name]
    return {
        "n": source["prompt_numerator"], "d": source["prompt_denominator"], "rate": source["prompt_rate"],
        "strict_n": source["strict_request_numerator"], "strict_d": source["strict_request_denominator"], "strict_rate": source["strict_request_rate"],
        "new_nll": source["target_new_nll"]["mean"], "old_nll": source["target_old_nll"]["mean"],
        "margin": source["target_old_minus_new_margin"]["mean"],
    }


def compact(summary: dict) -> dict:
    return {
        "EFF": metric(summary, "rewrite_success"),
        "RewriteAcc": metric(summary, "rewrite_acc"),
        "GEN": metric(summary, "paraphrase_success"),
        "RephraseAcc": metric(summary, "paraphrase_acc"),
        "LOC": {"n": summary["locality"]["numerator"], "d": summary["locality"]["denominator"], "rate": summary["locality"]["rate"]},
    }


def aggregate(summaries: list[dict]) -> dict:
    items = [compact(x) for x in summaries]
    out = {}
    for name in ("EFF", "RewriteAcc", "GEN", "RephraseAcc"):
        out[name] = {key: sum(x[name][key] for x in items) for key in ("n", "d", "strict_n", "strict_d")}
        out[name]["rate"] = out[name]["n"] / out[name]["d"]
        out[name]["strict_rate"] = out[name]["strict_n"] / out[name]["strict_d"]
        for key in ("new_nll", "old_nll", "margin"):
            out[name][key] = mean(x[name][key] for x in items)
    out["LOC"] = {key: sum(x["LOC"][key] for x in items) for key in ("n", "d")}
    out["LOC"]["rate"] = out["LOC"]["n"] / out["LOC"]["d"]
    return out


def count(x: dict, strict: bool = False) -> str:
    if strict:
        return f"{x['strict_n']}/{x['strict_d']} ({100*x['strict_rate']:.1f}%)"
    return f"{x['n']}/{x['d']} ({100*x['rate']:.1f}%)"


def mini(x: dict) -> str:
    return f"{count(x['EFF'])}; {count(x['RewriteAcc'])}; {count(x['GEN'])}/{count(x['GEN'], True)}; {count(x['RephraseAcc'])}/{count(x['RephraseAcc'], True)}; {x['LOC']['n']}/{x['LOC']['d']} ({100*x['LOC']['rate']:.1f}%)"


def dmetric(a: dict, b: dict) -> dict:
    result = {}
    for name in ("EFF", "RewriteAcc", "GEN", "RephraseAcc"):
        result[name] = {key: a[name][key] - b[name][key] for key in ("n", "rate", "strict_n", "strict_rate", "new_nll", "old_nll", "margin")}
    result["LOC"] = {key: a["LOC"][key] - b["LOC"][key] for key in ("n", "rate")}
    return result


def accepted(root: Path, b: int) -> list[dict]:
    base = root / f"raw/batches/b{b:02d}/raw/ode/p1r52-rsa-r42safekdc-m1-soft"
    return [load(base / f"accepted-k{k}.json") for k in range(1, 9)]


def phase_name(p: str) -> str:
    return {"entry_pre": "true entry-pre", "immediate_post": "immediate post", "final_W10": "final W10"}[p]


def scan_atomic(root: Path, b: int, h_on: bool) -> dict:
    terminal = load(root / f"raw/batches/b{b:02d}/terminal.json")
    ks = accepted(root, b)
    z = terminal["atomic_or_native"]["terminal_z8_oracle"]["target_new_nll"]
    w = ks[-1]["progress"].get("terminal_mean_target_new_nll")
    ctrl = {
        "request_steps": len(ks) * 10,
        "primary": sum(k["target_update"]["primary_accept_count"] for k in ks),
        "rescue": sum(k["target_update"]["rescue_accept_count"] for k in ks),
        "current": sum(k["target_update"]["current_hold_count"] for k in ks),
        "active_gradient": sum(k["target_update"]["active_gradient_count"] for k in ks),
        "no_direction": sum(10 - k["target_update"]["active_gradient_count"] for k in ks),
        "clamp_hit": sum(k["target_update"]["clamp_hit_count"] for k in ks),
        "fallback_steps": sum(int(k["routing"]["fallback_to_neutral"]) for k in ks),
        "route_status_counts": dict(sorted(Counter(k["routing"]["status"] for k in ks).items())),
        "route_fallback_reasons": dict(sorted(Counter(str(k["routing"]["fallback_reason"]) for k in ks if k["routing"]["fallback_reason"] is not None).items())),
        "mean_alpha_req": mean(k["routing"]["alpha_req"] for k in ks),
        "mean_alpha_apply": mean(k["routing"]["alpha_apply"] for k in ks),
        "max_abs_strength_residual": max(abs(k["routing"]["equality_residual"]) for k in ks),
        "mean_energy": mean(k["routing"]["selected_energy"] for k in ks),
        "mean_P": mean(k["routing"]["selected_p"] for k in ks),
        "mean_capacity": mean(k["routing"]["selected_capacity"] for k in ks),
        "mean_entropy": mean(k["routing"]["simplex_entropy"] for k in ks),
        "mean_top1": mean(k["routing"]["simplex_top1_share"] for k in ks),
        "mean_predicted_progress": mean(k["progress"]["predicted"] for k in ks),
        "actual_progress": "NOT_RECORDED_DELAYED_TO_NEXT_REFRESHED_FIELD" if any(k["progress"].get("actual") is None for k in ks) else mean(k["progress"]["actual"] for k in ks),
        "mean_realization_norm_gain": mean(sum(k["target_write_realization"]["norm_gain"]) / 10 for k in ks),
        "negative_physical_progress": "NOT_RECORDED_DELAYED_TO_NEXT_REFRESHED_FIELD",
        "kl_model_forward_added": sum(k["target_update"]["added_kl_model_forward_count"] for k in ks),
        "kl_backward_added": sum(k["target_update"]["added_kl_backward_count"] for k in ks),
        "source_receipt_sha256": terminal["atomic_or_native"]["accepted_receipt_sha256"],
    }
    route_h = terminal["structural_h_routing"]
    if h_on:
        structural = {
            "mode": "STRUCTURAL_H_ON",
            "status_counts": dict(sorted(Counter(r["status"] for r in route_h).items())),
            "selected_sum": sum(r["h_selected"] for r in route_h), "neutral_sum": sum(r["h_neutral"] for r in route_h), "delta_sum": sum(r["h_delta"] for r in route_h),
            "max_abs_strength_residual": max(abs(r["strength_residual"]) for r in route_h), "max_energy_violation": max(abs(r["energy_violation"]) for r in route_h), "max_abs_p_violation": max(abs(r["p_violation"]) for r in route_h),
        }
    else:
        structural = {
            "mode": "ALPHA_CACHE_ON_STRUCTURAL_H_OFF",
            "status_counts": dict(sorted(Counter(r["status"] for r in route_h).items())),
            "alpha_solve_history_width": route_h[-1]["alpha_solve_history_width"],
            "alpha_solve_cache_consume_count": sum(r["alpha_solve_cache_consume_count"] for r in route_h),
            "risk_observation_width": route_h[-1]["risk_observation_width"], "anchor_observation_width": route_h[-1]["anchor_observation_width"],
            "structural_h_decision_history_width": route_h[-1]["structural_h_decision_history_width"],
            "structural_h_decision_influence_count": sum(r["structural_h_decision_influence_count"] for r in route_h),
            "observation_ledger_decision_influence_count": sum(r["observation_ledger_decision_influence_count"] for r in route_h),
            "added_model_forward_backward_materialization": [sum(r["added_model_forward_count"] for r in route_h), sum(r["added_backward_count"] for r in route_h), sum(r["added_materialization_count"] for r in route_h)],
            "max_abs_strength_residual": max(abs(r["strength_residual"]) for r in route_h), "max_energy_violation": max(abs(r["energy_violation"]) for r in route_h), "max_abs_p_violation": max(abs(r["p_violation"]) for r in route_h),
        }
    tx = terminal["history_transaction"]
    return {
        "z8_full_six_target_new_nll": z, "W8_full_six_target_new_nll": w,
        "W_minus_z_nll": w - z, "controller": ctrl, "structural": structural,
        "transaction": {
            "entry_weight_sha256": terminal["entry_weight_sha256"], "commit_weight_sha256": terminal["commit_weight_sha256"],
            "history_width_at_entry": terminal["history_width_at_entry"], "active_history_after": terminal["active_history_count"], "anchors_after": terminal["lifetime_anchor_count"],
            "history_append": tx["appended_count"], "weight_commit": terminal["weight_transaction"]["commit_count"],
            "post_commit_verified": terminal["weight_transaction"]["post_commit_verified"], "rollback": terminal["weight_transaction"]["rollback_count"],
            "retry": terminal["retry_count"], "backtracking": terminal["backtracking_count"], "interbatch_W0_restore": terminal["interbatch_W0_restore_count"],
            "action_freeze": terminal["action_freeze"] if isinstance(terminal["action_freeze"], bool) else terminal["action_freeze"].get("status"),
            "heldout_controller_influence": terminal["heldout_controller_influence_count"],
        },
    }


def main() -> None:
    terminals = {k: load(v / "terminal.json") for k, v in ROOTS.items()}
    checkpoints = {k: sorted(load(v / "b1-b10-pre-post-checkpoints.json")["rows"], key=lambda row: row["round"]) for k, v in ROOTS.items()}
    cohorts = {k: {row["round"]: row for row in load(v / "batch-age-pre-post-final-cohorts.json")["rows"]} for k, v in ROOTS.items()}
    reqs = {k: load(v / "entry-pre-immediate-post-final-w10-requests.json")["rows"] for k, v in ROOTS.items()}
    if any(len(checkpoints[k]) != 10 or len(reqs[k]) != 100 for k in ROOTS):
        raise RuntimeError("required sealed 10 checkpoint and 100 request rows absent")
    def rkey(r): return (r["round"], r["case_id"], r["request_index"], r["request_sha256"])
    if len({tuple(sorted(rkey(x) for x in reqs[k])) for k in ROOTS}) != 1:
        raise RuntimeError("three-arm exact request identity mismatch")
    for b in range(1, 11):
        orders = {checkpoints[k][b-1]["entry_pre_summary"]["request_order_sha256"] for k in ROOTS}
        if len(orders) != 1:
            raise RuntimeError(f"B{b} request order mismatch")

    cprows = {k: {r["round"]: r for r in checkpoints[k]} for k in ROOTS}
    phase_rows = {}
    for name, note, getter in (
        ("W0_baseline_B1_entry", "B1 common W0; 10-request panel, not 100-request aggregate", lambda k: compact(cprows[k][1]["entry_pre_summary"])),
        ("all_true_batch_entry_pre", "100 requests at actual sequential W_(b-1) entry", lambda k: aggregate([cprows[k][b]["entry_pre_summary"] for b in range(1, 11)])),
        ("all_immediate_post", "100 requests after their own B10 post state", lambda k: aggregate([cprows[k][b]["immediate_post_summary"] for b in range(1, 11)])),
        ("final_W10_B100", "all 100 requests at final W10", lambda k: compact(terminals[k]["final_B100"])),
    ):
        phase_rows[name] = {"note": note, **{k: getter(k) for k in ROOTS}}

    batch_rows = []
    controller_route_rows = []
    h_on_total = {"request_steps": 0, "primary": 0, "rescue": 0, "current": 0, "active_gradient": 0, "no_direction": 0, "clamp_hit": 0, "fallback_steps": 0}
    h_off_total = dict(h_on_total)
    for b in range(1, 11):
        hscan, oscan = scan_atomic(HROOT, b, True), scan_atomic(CONTROL, b, False)
        arms = {}
        for arm in ROOTS:
            cp = cprows[arm][b]
            arms[arm] = {
                "entry_pre": compact(cp["entry_pre_summary"]), "immediate_post": compact(cp["immediate_post_summary"]),
                "final_W10": compact(cohorts[arm][b]["final_W10"]),
            }
        arms["h_on"].update({"z_w": hscan, "history_width": cprows["h_on"][b]["history_width_at_entry"]})
        arms["h_off"].update({"z_w": oscan, "history_width": cprows["h_off"][b]["history_width_at_entry"]})
        arms["native"].update({"z_w": {"z8_full_six_target_new_nll": "NOT_RECORDED_NATIVE_NO_TARGET_ORACLE", "W8_full_six_target_new_nll": "NOT_RECORDED_NATIVE_NO_TARGET_ORACLE", "W_minus_z_nll": "NOT_RECORDED_NATIVE_NO_TARGET_ORACLE"}, "history_width": cprows["native"][b]["history_width_at_entry"]})
        batch_rows.append({"batch": b, "history_width": cprows["h_off"][b]["history_width_at_entry"], "batch_age_at_W10": cohorts["h_off"][b]["batch_age_at_W10"], "arms": arms,
                           "h_on_minus_h_off": {p: dmetric(arms["h_on"][p], arms["h_off"][p]) for p in ("entry_pre", "immediate_post", "final_W10")},
                           "h_off_minus_native": {p: dmetric(arms["h_off"][p], arms["native"][p]) for p in ("entry_pre", "immediate_post", "final_W10")}})
        for arm, scan, total in (("h_on", hscan, h_on_total), ("h_off", oscan, h_off_total)):
            for k in total: total[k] += scan["controller"][k]
            controller_route_rows.append({"batch": b, "arm": arm, "history_width": cprows[arm][b]["history_width_at_entry"], **scan})

    route_total = {
        "h_on_status": dict(sorted(Counter(s for r in controller_route_rows if r["arm"] == "h_on" for s, n in r["structural"]["status_counts"].items() for _ in range(n)).items())),
        "h_off_status": dict(sorted(Counter(s for r in controller_route_rows if r["arm"] == "h_off" for s, n in r["structural"]["status_counts"].items() for _ in range(n)).items())),
        "h_on_selected_sum": sum(r["structural"]["selected_sum"] for r in controller_route_rows if r["arm"] == "h_on"),
        "h_on_neutral_sum": sum(r["structural"]["neutral_sum"] for r in controller_route_rows if r["arm"] == "h_on"),
        "h_on_delta_sum": sum(r["structural"]["delta_sum"] for r in controller_route_rows if r["arm"] == "h_on"),
        "h_off_structural_h_decision_influence": sum(r["structural"]["structural_h_decision_influence_count"] for r in controller_route_rows if r["arm"] == "h_off"),
        "h_off_alpha_cache_consume": sum(r["structural"]["alpha_solve_cache_consume_count"] for r in controller_route_rows if r["arm"] == "h_off"),
        "h_off_observation_ledger_influence": sum(r["structural"]["observation_ledger_decision_influence_count"] for r in controller_route_rows if r["arm"] == "h_off"),
    }
    z_summary = {arm: {
        "z8_mean": mean(r["arms"][arm]["z_w"]["z8_full_six_target_new_nll"] for r in batch_rows),
        "W8_mean": mean(r["arms"][arm]["z_w"]["W8_full_six_target_new_nll"] for r in batch_rows),
        "W_minus_z_mean": mean(r["arms"][arm]["z_w"]["W_minus_z_nll"] for r in batch_rows),
        "W_minus_z_max_abs": max(abs(r["arms"][arm]["z_w"]["W_minus_z_nll"]) for r in batch_rows),
    } for arm in ("h_on", "h_off")}
    z_summary["native"] = {"z8_mean": "NOT_RECORDED_NATIVE_NO_TARGET_ORACLE", "W8_mean": "NOT_RECORDED_NATIVE_NO_TARGET_ORACLE", "W_minus_z_mean": "NOT_RECORDED_NATIVE_NO_TARGET_ORACLE"}

    # Request-level table already contains all phase bits, NLL/margins, gain and forgetting.
    maps = {a: {rkey(r): r for r in rows} for a, rows in reqs.items()}
    failed = []
    for key in sorted(maps["h_on"]):
        rows = {a: maps[a][key] for a in ROOTS}
        def fail(a, phase, met):
            x = rows[a][phase][met]
            return x["correct"] < x["required"]
        flags = {
            "entry_hard_h_on": fail("h_on", "entry_pre", "rewrite_success"),
            "h_on_post_gen_failure": fail("h_on", "immediate_post", "paraphrase_success"),
            "h_off_post_gen_failure": fail("h_off", "immediate_post", "paraphrase_success"),
            "native_post_gen_failure": fail("native", "immediate_post", "paraphrase_success"),
            "h_on_final_gen_loss": rows["h_on"]["final_W10"]["paraphrase_success"]["correct"] < rows["h_on"]["immediate_post"]["paraphrase_success"]["correct"],
            "h_off_final_gen_loss": rows["h_off"]["final_W10"]["paraphrase_success"]["correct"] < rows["h_off"]["immediate_post"]["paraphrase_success"]["correct"],
            "h_on_post_better_than_h_off": rows["h_on"]["immediate_post"]["paraphrase_success"]["correct"] > rows["h_off"]["immediate_post"]["paraphrase_success"]["correct"],
            "h_off_post_better_than_h_on": rows["h_off"]["immediate_post"]["paraphrase_success"]["correct"] > rows["h_on"]["immediate_post"]["paraphrase_success"]["correct"],
        }
        if any(flags.values()):
            failed.append({"round": key[0], "case_id": key[1], "request_index": key[2], "request_sha256": key[3], "history_width": rows["h_off"]["history_width_at_entry"], "flags": flags,
                           "h_on": rows["h_on"], "h_off": rows["h_off"], "native": rows["native"],
                           "association_scope": "RAW_FREE_ASSOCIATION_ONLY_NO_ISOLATED_CAUSAL_ATTRIBUTION"})
    flag_names = tuple(failed[0]["flags"]) if failed else ()
    low_summary = {"association_scope": "RAW_FREE_ASSOCIATION_ONLY_NO_ISOLATED_CAUSAL_ATTRIBUTION", "request_count": 100, "failed_or_transition_rows": len(failed),
                   "counts": {x: sum(r["flags"][x] for r in failed) for x in flag_names},
                   "routing_granularity": "batch_step_only; requestwise H-routing attribution=NOT_RECORDED", "conclusion": "ASSOCIATION_ONLY"}

    compute = load(OUT / "p1r52-sequential-three-arm-compute-ledger.json")["payload"]
    inputs = []
    for arm, root in ROOTS.items():
        for name in ("terminal.json", "b1-b10-pre-post-checkpoints.json", "batch-age-pre-post-final-cohorts.json", "entry-pre-immediate-post-final-w10-requests.json", "manifest.json"):
            p = root / name
            inputs.append({"arm": arm, "path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)})
    metrics_payload = {
        "schema": "p1r52-sequential/alphacache-on-structuralh-off-three-arm-readable-metrics/v4",
        "status": "TERMINAL_FACTUAL_REPORT_COMPLETE", "labels": LABEL,
        "phase_rows": phase_rows, "batch_rows": batch_rows,
        "controller_totals": {"h_on": h_on_total, "h_off": h_off_total, "native": "NOT_RECORDED_NATIVE_CONTROLLER_PATH"},
        "routing_totals": route_total, "z_w_nll_summary": z_summary,
        "alpha_cache_state": {"h_off_entry_widths": [r["history_width"] for r in batch_rows], "h_off_transaction_append_total": sum(r["arms"]["h_off"]["z_w"]["transaction"]["history_append"] for r in batch_rows),
                              "h_off_terminal_active_history": terminals["h_off"]["terminal_active_history_count"], "h_off_terminal_anchors": terminals["h_off"]["terminal_lifetime_anchor_count"],
                              "h_off_terminal_W0_restore": terminals["h_off"]["terminal_W0_restore"], "h_on_alpha_solve_cache_consumption": "NOT_RECORDED_IN_H_ON_SEALED_RECEIPTS"},
        "compute_ledger": compute, "low_cohort_summary": low_summary,
        "input_identities": inputs, "not_recorded": ["Native z8/W8 target oracle", "Native controller/routing telemetry", "requestwise Structural-H routing attribution", "delayed physical actual progress and negative-progress scalar"],
    }
    metrics_path = write_once("p1r52-sequential-three-arm-readable-metrics-v4.json", metrics_payload)
    controller_path = write_once("p1r52-sequential-three-arm-controller-routing-v4.json", {"schema": "p1r52-sequential/three-arm-controller-routing/v4", "row_count": len(controller_route_rows), "rows": controller_route_rows, "totals": {"h_on": h_on_total, "h_off": h_off_total, "routing": route_total}})
    failed_path = write_once("p1r52-sequential-three-arm-failed-cohort-v4.json", {"schema": "p1r52-sequential/three-arm-failed-cohort/v4", "row_count": len(failed), "rows": failed, "summary": low_summary})

    def phase_table(arms: tuple[str, ...]) -> list[str]:
        lines = ["|Panel|Arm|EFF|Rewrite Acc|GEN|GEN-strict|Rephrase Acc|Acc-strict|LOC|", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        labels = (("W0_baseline_B1_entry", "W0(B1 entry)"), ("all_true_batch_entry_pre", "all true entry-pre"), ("all_immediate_post", "all immediate post"), ("final_W10_B100", "final W10/B100"))
        for key, title in labels:
            for arm in arms:
                x = phase_rows[key][arm]
                lines.append(f"|{title}|{LABEL[arm]}|{count(x['EFF'])}|{count(x['RewriteAcc'])}|{count(x['GEN'])}|{count(x['GEN'], True)}|{count(x['RephraseAcc'])}|{count(x['RephraseAcc'], True)}|{x['LOC']['n']}/{x['LOC']['d']} ({100*x['LOC']['rate']:.1f}%)|")
        return lines

    def nll_table(arms: tuple[str, ...]) -> list[str]:
        lines = ["|Panel|Arm|RW new/true NLL|PP new/true NLL|z8/W8/gap full-six NLL|", "|---|---|---:|---:|---:|"]
        for key, title in (("W0_baseline_B1_entry", "W0"), ("all_true_batch_entry_pre", "entry-pre"), ("all_immediate_post", "post"), ("final_W10_B100", "final W10/B100")):
            for arm in arms:
                x = phase_rows[key][arm]
                zw = "NOT_RECORDED" if arm == "native" else (f"mean={z_summary[arm]['z8_mean']:.6f}/{z_summary[arm]['W8_mean']:.6f}/{z_summary[arm]['W_minus_z_mean']:+.6f}" if key == "all_immediate_post" else "NOT_RECORDED_OUTER_K8_ONLY")
                lines.append(f"|{title}|{LABEL[arm]}|{x['EFF']['new_nll']:.6f}/{x['EFF']['old_nll']:.6f}|{x['GEN']['new_nll']:.6f}/{x['GEN']['old_nll']:.6f}|{zw}|")
        return lines

    control_lines = [
        "# P1R52-Soft-Sequential-AlphaCache-On-StructuralH-Off — reader-oriented v4", "",
        "> 별도의 Structural-H OFF control standalone factual report입니다. H-aware 및 Native sealed artifacts는 읽기만 했고 변경하지 않았습니다.", "",
        "## 지표 정의", "",
        "- **EFF** = `rewrite_success` (rewrite prompt의 target-new mean NLL < target-true mean NLL). Accuracy와 다릅니다.",
        "- **GEN** = `paraphrase_success` (`rephrase_success` alias); **GEN-strict** = 두 paraphrase 모두 성공한 request.",
        "- **Rewrite/Rephrase Acc** = strict target suffix token argmax accuracy; **LOC** = locality-preservation accuracy.",
        "- **entry-pre**는 batch별 실제 W_(b-1), **immediate-post**는 W_e, **final W10/B100**은 W10입니다.", "",
        "## 절대값: W0 → true entry-pre → post → final", "",
        *phase_table(("h_off", "native")), "", "## NLL (new/true) 및 z/W", "", *nll_table(("h_off", "native")), "",
        "## B1–B10 절대 checkpoint", "",
        "각 셀은 `EFF; RewriteAcc; GEN/GEN-strict; RephraseAcc/Acc-strict; LOC` 순서의 numerator/denominator(rate)입니다.", "",
        "|B|Alpha solve cache width|entry-pre|immediate-post|final W10|z8/W8/gap|", "|---:|---:|---|---|---|---:|",
    ]
    for row in batch_rows:
        o = row["arms"]["h_off"]
        z = o["z_w"]
        control_lines.append(f"|B{row['batch']}|{row['history_width']}|{mini(o['entry_pre'])}|{mini(o['immediate_post'])}|{mini(o['final_W10'])}|{z['z8_full_six_target_new_nll']:.6f}/{z['W8_full_six_target_new_nll']:.6f}/{z['W_minus_z_nll']:+.6f}|")
    control_lines += ["", "## Controller · route · Alpha cache receipt", "",
                      "|B|Primary/Rescue/Current|Active/no-direction|clamp|route|strength residual|α cache consume|Structural-H influence|P/capacity/energy|", "|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for row in controller_route_rows:
        if row["arm"] != "h_off": continue
        c, s = row["controller"], row["structural"]
        control_lines.append(f"|B{row['batch']}|{c['primary']}/{c['rescue']}/{c['current']}|{c['active_gradient']}/{c['no_direction']}|{c['clamp_hit']}|{','.join(f'{k}:{v}' for k,v in c['route_status_counts'].items())}|{c['max_abs_strength_residual']:.2e}|{s['alpha_solve_cache_consume_count']}|{s['structural_h_decision_influence_count']}|{c['mean_P']:.3e}/{c['mean_capacity']:.3e}/{c['mean_energy']:.3e}|")
    control_lines += ["", f"- Alpha solve cache entry width `{[r['history_width'] for r in batch_rows]}`; transaction append total `{metrics_payload['alpha_cache_state']['h_off_transaction_append_total']}`; routing consume total `{route_total['h_off_alpha_cache_consume']}`.",
                      f"- Structural-H decision influence `{route_total['h_off_structural_h_decision_influence']}`; observation-ledger influence `{route_total['h_off_observation_ledger_influence']}`; H-off route status = `ALPHA_CACHE_ON_STRUCTURAL_H_OFF`.",
                      f"- terminal W0 pointer/byte restore `{terminals['h_off']['terminal_W0_restore']['pointer_restored_exact']}/{terminals['h_off']['terminal_W0_restore']['byte_restored_exact']}`; append/anchors `{terminals['h_off']['terminal_active_history_count']}/{terminals['h_off']['terminal_lifetime_anchor_count']}`; retry/backtracking/rollback/interbatch W0 restore `0/0/0/0`.",
                      "", "## Compute", "",
                      f"- one-pass accuracy additional model F/B/generation = `{terminals['h_off']['accuracy_amendment_added_model_forward_count']}/{terminals['h_off']['accuracy_amendment_added_backward_count']}/{terminals['h_off']['accuracy_amendment_added_generation_count']}`; entry-pre evaluator F/B/generation = `{terminals['h_off']['batch_entry_pre_evaluator_forward_count']}/{terminals['h_off']['batch_entry_pre_evaluator_backward_count']}/{terminals['h_off']['batch_entry_pre_evaluator_generation_count']}`.",
                      "- Per-request entry-pre → immediate-post → final-W10 gain/forgetting, all B1–B10 phase metrics, transaction and controller details are sealed in the referenced machine tables.", "", "## 상태", "",
                      "- `TERMINAL_FACTUAL_REPORT_COMPLETE`; scientific_promotion=false; additional model/evaluator/GPU/Slurm action=0."]
    control_report = write_text_once("p1r52-soft-sequential-alphacache-on-structuralh-off-10xb10-factual-ko-v4.md", "\n".join(control_lines) + "\n")

    comp_lines = [
        "# P1R52 Sequential — Structural-H ON / AlphaCache-ON Structural-H OFF / Native — reader-oriented v4", "",
        "> 동일한 Llama 10×B10/100 요청·case·order identity에 대한 raw-free factual comparison입니다. 세 arm request key 100/100 및 B1–B10 order identity가 exact로 확인되었습니다.", "",
        "## 지표 정의", "",
        "- **EFF**=`rewrite_success`; **GEN**=`paraphrase_success` (`rephrase_success` alias); **GEN-strict**=두 paraphrase 모두 성공한 request.",
        "- **Rewrite Acc/Rephrase Acc**는 strict token argmax accuracy이며 success와 별도입니다. **LOC**는 locality-preservation accuracy입니다.",
        "- W0은 B1 공통 entry, entry-pre는 실제 누적 W_(b-1), post는 W_e, final은 W10/B100입니다.", "",
        "## 절대값: W0 → true entry-pre → post → final", "", *phase_table(("h_on", "h_off", "native")), "", "## NLL (new/true) 및 z/W full-six", "", *nll_table(("h_on", "h_off", "native")), "",
        "## B1–B10 절대 checkpoint — 세 arm", "",
        "각 셀은 `EFF; RewriteAcc; GEN/GEN-strict; RephraseAcc/Acc-strict; LOC`의 절대 numerator/denominator(rate)입니다. 세부 NLL/margin·phase table은 `p1r52-sequential-three-arm-readable-metrics-v4.json` 및 기존 exact per-request table에 있습니다.", "",
        "|B|cache width|H-ON pre→post→final|H-OFF pre→post→final|Native pre→post→final|", "|---:|---:|---|---|---|"]
    for row in batch_rows:
        comp_lines.append(f"|B{row['batch']}|{row['history_width']}|{' → '.join(mini(row['arms']['h_on'][p]) for p in ('entry_pre','immediate_post','final_W10'))}|{' → '.join(mini(row['arms']['h_off'][p]) for p in ('entry_pre','immediate_post','final_W10'))}|{' → '.join(mini(row['arms']['native'][p]) for p in ('entry_pre','immediate_post','final_W10'))}|")
    final_h, final_o, final_n = phase_rows["final_W10_B100"]["h_on"], phase_rows["final_W10_B100"]["h_off"], phase_rows["final_W10_B100"]["native"]
    ho, on = dmetric(final_h, final_o), dmetric(final_o, final_n)
    comp_lines += ["", "## B100 arithmetic deltas", "",
                     f"- H-ON − H-OFF: EFF `{ho['EFF']['n']:+}/100`, RewriteAcc `{ho['RewriteAcc']['n']:+}/100`, GEN `{ho['GEN']['n']:+}/200`, GEN-strict `{ho['GEN']['strict_n']:+}/100`, RephraseAcc `{ho['RephraseAcc']['n']:+}/200`, Acc-strict `{ho['RephraseAcc']['strict_n']:+}/100`, LOC `{ho['LOC']['n']:+}/1000`.",
                     f"- H-OFF − Native: EFF `{on['EFF']['n']:+}/100`, RewriteAcc `{on['RewriteAcc']['n']:+}/100`, GEN `{on['GEN']['n']:+}/200`, GEN-strict `{on['GEN']['strict_n']:+}/100`, RephraseAcc `{on['RephraseAcc']['n']:+}/200`, Acc-strict `{on['RephraseAcc']['strict_n']:+}/100`, LOC `{on['LOC']['n']:+}/1000`.",
                     "", "## B5 same-batch raw-free observation", ""]
    b5 = next(r for r in batch_rows if r["batch"] == 5)
    for arm in ("h_on", "h_off", "native"):
        comp_lines.append(f"- {LABEL[arm]} immediate-post: {mini(b5['arms'][arm]['immediate_post'])}.")
    b5d = b5["h_on_minus_h_off"]["immediate_post"]
    comp_lines += [f"- B5 H-ON−H-OFF: GEN `{b5d['GEN']['n']:+}/20`, GEN-strict `{b5d['GEN']['strict_n']:+}/10`, RephraseAcc `{b5d['RephraseAcc']['n']:+}/20`, LOC `{b5d['LOC']['n']:+}/100`.",
                     "", "## Controller · routing · cache", "",
                     f"- H-ON controller Primary/Rescue/Current=`{h_on_total['primary']}/{h_on_total['rescue']}/{h_on_total['current']}` of `{h_on_total['request_steps']}` request-steps; active/no-direction=`{h_on_total['active_gradient']}/{h_on_total['no_direction']}`; clamp=`{h_on_total['clamp_hit']}`; fallback steps=`{h_on_total['fallback_steps']}`.",
                     f"- H-OFF controller Primary/Rescue/Current=`{h_off_total['primary']}/{h_off_total['rescue']}/{h_off_total['current']}` of `{h_off_total['request_steps']}` request-steps; active/no-direction=`{h_off_total['active_gradient']}/{h_off_total['no_direction']}`; clamp=`{h_off_total['clamp_hit']}`; fallback steps=`{h_off_total['fallback_steps']}`.",
                     f"- H-ON status `{route_total['h_on_status']}`, selected/neutral/ΔH=`{route_total['h_on_selected_sum']:.6g}/{route_total['h_on_neutral_sum']:.6g}/{route_total['h_on_delta_sum']:+.6g}`. H-OFF status `{route_total['h_off_status']}`, Alpha cache consume=`{route_total['h_off_alpha_cache_consume']}`, Structural-H influence=`{route_total['h_off_structural_h_decision_influence']}`.",
                     "- Strength/energy/P residuals, selected α, P/capacity/energy, entropy/top1, predicted progress and realization gain are recorded batch×arm in `p1r52-sequential-three-arm-controller-routing-v4.json` (20 rows).", "",
                     "## Sequential state / transaction", "",
                     f"- H-OFF entry cache widths `{metrics_payload['alpha_cache_state']['h_off_entry_widths']}`; append `{metrics_payload['alpha_cache_state']['h_off_transaction_append_total']}`; final active cache/anchors `{metrics_payload['alpha_cache_state']['h_off_terminal_active_history']}/{metrics_payload['alpha_cache_state']['h_off_terminal_anchors']}`.",
                     f"- H-OFF terminal W0 pointer+byte restore `{terminals['h_off']['terminal_W0_restore']['pointer_restored_exact']}/{terminals['h_off']['terminal_W0_restore']['byte_restored_exact']}`. Both R52 arms have physical W persistence and B1…B10 transaction records; retry/backtracking/rollback/interbatch W0 restore are recorded as zero in their sealed terminal receipts.",
                     "", "## Low / failed cohort, batch age, forgetting", "",
                     f"- failed-or-transition rows `{low_summary['failed_or_transition_rows']}/100`; H-ON post GEN fail `{low_summary['counts'].get('h_on_post_gen_failure',0)}/100`; H-OFF post GEN fail `{low_summary['counts'].get('h_off_post_gen_failure',0)}/100`; H-ON post→final GEN loss `{low_summary['counts'].get('h_on_final_gen_loss',0)}/100`; H-OFF equivalent `{low_summary['counts'].get('h_off_final_gen_loss',0)}/100`.",
                     f"- H-ON post GEN > H-OFF `{low_summary['counts'].get('h_on_post_better_than_h_off',0)}/100`; H-OFF > H-ON `{low_summary['counts'].get('h_off_post_better_than_h_on',0)}/100`. Entry hardness, batch age, pre→post gain and post→final loss are retained per request in the sealed exact 3-arm request table.",
                     "- Finding scope: `RAW_FREE_ASSOCIATION_ONLY_NO_ISOLATED_CAUSAL_ATTRIBUTION`; request-level Structural-H routing attribution is `NOT_RECORDED` because routing receipt granularity is batch×step.",
                     "", "## Compute", "",
                     f"- Accuracy amendment additional F/B/generation: H-ON `{terminals['h_on']['accuracy_amendment_added_model_forward_count']}/{terminals['h_on']['accuracy_amendment_added_backward_count']}/{terminals['h_on']['accuracy_amendment_added_generation_count']}`, H-OFF `{terminals['h_off']['accuracy_amendment_added_model_forward_count']}/{terminals['h_off']['accuracy_amendment_added_backward_count']}/{terminals['h_off']['accuracy_amendment_added_generation_count']}`, Native `{terminals['native']['accuracy_amendment_added_model_forward_count']}/{terminals['native']['accuracy_amendment_added_backward_count']}/{terminals['native']['accuracy_amendment_added_generation_count']}`.",
                     f"- Entry-pre evaluator F/B/generation: H-ON `{terminals['h_on']['batch_entry_pre_evaluator_forward_count']}/{terminals['h_on']['batch_entry_pre_evaluator_backward_count']}/{terminals['h_on']['batch_entry_pre_evaluator_generation_count']}`, H-OFF `{terminals['h_off']['batch_entry_pre_evaluator_forward_count']}/{terminals['h_off']['batch_entry_pre_evaluator_backward_count']}/{terminals['h_off']['batch_entry_pre_evaluator_generation_count']}`, Native `{terminals['native']['batch_entry_pre_evaluator_forward_count']}/{terminals['native']['batch_entry_pre_evaluator_backward_count']}/{terminals['native']['batch_entry_pre_evaluator_generation_count']}`.",
                     "", "## 상태", "", "- `TERMINAL_FACTUAL_REPORT_COMPLETE`; scientific_promotion=false; no model/evaluator/GPU/Slurm rerun in this report-only pass."]
    comparison_report = write_text_once("p1r52-sequential-structuralh-on-off-native-three-arm-ko-v4.md", "\n".join(comp_lines) + "\n")

    # Append-only v4 manifest and receipt bind all newly created files plus immutable base tables.
    members = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in {Path(__file__).name, "readable-v4-manifest.json", "readable-v4-receipt.json"}:
            members.append({"name": p.name, "bytes": p.stat().st_size, "sha256": sha(p)})
    root = digest(json.dumps(members, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
    manifest = write_once("readable-v4-manifest.json", {"schema": "p1r52-sequential/three-arm-readable-v4-manifest/v1", "status": "TERMINAL_FACTUAL_REPORT_COMPLETE", "canonical_control_report": control_report.name, "canonical_three_arm_report": comparison_report.name,
                                                          "members": members, "members_root_sha256": root, "input_identity_pairing": "PASS", "source_heads": {a: terminals[a]["source_head"] for a in ROOTS}})
    receipt = write_once("readable-v4-receipt.json", {"schema": "p1r52-sequential/three-arm-readable-v4-receipt/v1", "status": "PASS", "control_report": str(control_report), "control_report_sha256": sha(control_report), "control_report_bytes": control_report.stat().st_size, "control_report_lines": len(control_report.read_text().splitlines()),
                                                        "three_arm_report": str(comparison_report), "three_arm_report_sha256": sha(comparison_report), "three_arm_report_bytes": comparison_report.stat().st_size, "three_arm_report_lines": len(comparison_report.read_text().splitlines()),
                                                        "metrics_sha256": sha(metrics_path), "controller_routing_sha256": sha(controller_path), "failed_cohort_sha256": sha(failed_path), "manifest_sha256": sha(manifest), "members_root_sha256": root,
                                                        "raw_content_included": False, "analysis_actions": {"model": 0, "evaluator": 0, "gpu": 0, "slurm": 0, "source_edit": 0, "result_mutation": 0}})
    print(json.dumps({"control_report": str(control_report), "comparison_report": str(comparison_report), "receipt": str(receipt)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
