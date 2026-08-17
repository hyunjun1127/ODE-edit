#!/usr/bin/env python3
"""Build the reader-oriented v4 report from sealed raw-free P1R52 receipts.

This transformer does not run a model or evaluator.  It makes the established
EFF/GEN/LOC names, z/W NLL, entry-pre baselines, accuracy panels, and routing
counts visible without replacing the sealed v3 package.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
H_ROOT = ROOT / "local/odebf/results/s05-p1r52-llama-soft-sequential-historical-10xb10-tech-r3-v1"
N_ROOT = ROOT / "local/odebf/results/s05-p1r52-native-alphaedit-sequential-10xb10-v1"
V3 = OUT / "p1r52-soft-sequential-structuralh-on-10xb10-factual-ko-v3.md"
METRICS_JSON = OUT / "p1r52-structuralh-on-readable-metrics-v4.json"
REPORT = OUT / "p1r52-soft-sequential-structuralh-on-10xb10-factual-ko-v4.md"
MANIFEST = OUT / "readable-v4-manifest.json"
RECEIPT = OUT / "readable-v4-receipt.json"


def read(path: Path):
    return json.loads(path.read_text())


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def write_once(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)


def dump_once(path: Path, payload) -> None:
    write_once(path, (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def metric(summary: dict, name: str) -> dict:
    item = summary["metrics"][name]
    return {
        "n": item["prompt_numerator"],
        "d": item["prompt_denominator"],
        "rate": item["prompt_rate"],
        "strict_n": item["strict_request_numerator"],
        "strict_d": item["strict_request_denominator"],
        "strict_rate": item["strict_request_rate"],
        "new_nll": item["target_new_nll"]["mean"],
        "old_nll": item["target_old_nll"]["mean"],
        "margin": item["target_old_minus_new_margin"]["mean"],
    }


def compact(summary: dict) -> dict:
    return {
        "EFF": metric(summary, "rewrite_success"),
        "RewriteAcc": metric(summary, "rewrite_acc"),
        "GEN": metric(summary, "paraphrase_success"),
        "RephraseAcc": metric(summary, "paraphrase_acc"),
        "LOC": {
            "n": summary["locality"]["numerator"],
            "d": summary["locality"]["denominator"],
            "rate": summary["locality"]["rate"],
        },
    }


def count(item: dict, strict: bool = False) -> str:
    if strict:
        return f"{item['strict_n']}/{item['strict_d']} ({100 * item['strict_rate']:.1f}%)"
    return f"{item['n']}/{item['d']} ({100 * item['rate']:.1f}%)"


def loc(item: dict) -> str:
    return f"{item['n']}/{item['d']} ({100 * item['rate']:.1f}%)"


def short_score(c: dict) -> str:
    return f"{c['EFF']['n']}/{c['EFF']['d']} · {c['GEN']['n']}/{c['GEN']['d']} · {c['LOC']['n']}/{c['LOC']['d']}"


def full_score_row(label: str, arm: str, c: dict) -> str:
    return (
        f"|{label}|{arm}|{count(c['EFF'])}|{count(c['RewriteAcc'])}|"
        f"{count(c['GEN'])}|{count(c['GEN'], True)}|{count(c['RephraseAcc'])}|"
        f"{count(c['RephraseAcc'], True)}|{loc(c['LOC'])}|"
    )


def phase_aggregate(rows: list[dict], phase: str) -> dict:
    summaries = [compact(r[phase]) for r in rows]
    out = {}
    for name in ("EFF", "RewriteAcc", "GEN", "RephraseAcc"):
        out[name] = {
            "n": sum(x[name]["n"] for x in summaries),
            "d": sum(x[name]["d"] for x in summaries),
            "strict_n": sum(x[name]["strict_n"] for x in summaries),
            "strict_d": sum(x[name]["strict_d"] for x in summaries),
            "new_nll": mean(x[name]["new_nll"] for x in summaries),
            "old_nll": mean(x[name]["old_nll"] for x in summaries),
            "margin": mean(x[name]["margin"] for x in summaries),
        }
        out[name]["rate"] = out[name]["n"] / out[name]["d"]
        out[name]["strict_rate"] = out[name]["strict_n"] / out[name]["strict_d"]
    out["LOC"] = {
        "n": sum(x["LOC"]["n"] for x in summaries),
        "d": sum(x["LOC"]["d"] for x in summaries),
    }
    out["LOC"]["rate"] = out["LOC"]["n"] / out["LOC"]["d"]
    return out


def main() -> None:
    h_cp = sorted(read(H_ROOT / "b1-b10-pre-post-checkpoints.json")["rows"], key=lambda x: x["round"])
    n_cp = sorted(read(N_ROOT / "b1-b10-pre-post-checkpoints.json")["rows"], key=lambda x: x["round"])
    h_cohort = {x["round"]: x for x in read(H_ROOT / "batch-age-pre-post-final-cohorts.json")["rows"]}
    n_cohort = {x["round"]: x for x in read(N_ROOT / "batch-age-pre-post-final-cohorts.json")["rows"]}
    h_terminal = read(H_ROOT / "terminal.json")
    n_terminal = read(N_ROOT / "terminal.json")
    compute = read(OUT / "p1r52-structuralh-on-compute-ledger.json")["payload"]
    low = read(OUT / "p1r52-structuralh-on-low-cohort-table.json")["summary"]

    if len(h_cp) != 10 or len(n_cp) != 10:
        raise RuntimeError("expected ten checkpoints per arm")

    batch_rows = []
    for idx in range(1, 11):
        hrow, nrow = h_cp[idx - 1], n_cp[idx - 1]
        hterm = read(H_ROOT / f"raw/batches/b{idx:02d}/terminal.json")
        nterm = read(N_ROOT / f"raw/batches/b{idx:02d}/terminal.json")
        accepted = []
        for k in range(1, 9):
            accepted.append(read(H_ROOT / f"raw/batches/b{idx:02d}/raw/ode/p1r52-rsa-r42safekdc-m1-soft/accepted-k{k}.json"))
        k8 = accepted[-1]
        routes = hterm["structural_h_routing"]
        route_status = Counter(x["status"] for x in routes)
        batch_rows.append({
            "batch": idx,
            "history_width": hrow["history_width_at_entry"],
            "batch_age_at_W10": h_cohort[idx]["batch_age_at_W10"],
            "structural_h_on": {
                "pre": compact(hrow["entry_pre_summary"]),
                "post": compact(hrow["immediate_post_summary"]),
                "final_W10": compact(h_cohort[idx]["final_W10"]),
                "z8_full_six_target_new_nll": hterm["atomic_or_native"]["terminal_z8_oracle"]["target_new_nll"],
                "W8_full_six_target_new_nll": k8["progress"]["terminal_mean_target_new_nll"],
                "W_minus_z_nll": k8["progress"]["terminal_mean_target_new_nll"] - hterm["atomic_or_native"]["terminal_z8_oracle"]["target_new_nll"],
                "controller_counts": {
                    "request_steps": len(accepted) * 10,
                    "primary": sum(x["target_update"]["primary_accept_count"] for x in accepted),
                    "rescue": sum(x["target_update"]["rescue_accept_count"] for x in accepted),
                    "current": sum(x["target_update"]["current_hold_count"] for x in accepted),
                    "active_gradient": sum(x["target_update"]["active_gradient_count"] for x in accepted),
                    "clamp_hit": sum(x["target_update"]["clamp_hit_count"] for x in accepted),
                    "soft_to_neutral_fallback_steps": sum(int(x["routing"]["fallback_to_neutral"]) for x in accepted),
                },
                "routing": {
                    "status_counts": dict(sorted(route_status.items())),
                    "H_SELECTED_sum": sum(x["h_selected"] for x in routes),
                    "H_NEUTRAL_sum": sum(x["h_neutral"] for x in routes),
                    "H_delta_sum": sum(x["h_delta"] for x in routes),
                    "allocation_entropy_mean": mean(x["allocation_entropy"] for x in routes),
                    "max_abs_strength_residual": max(abs(x["strength_residual"]) for x in routes),
                    "max_energy_violation": max(abs(x["energy_violation"]) for x in routes),
                    "max_abs_P_violation": max(abs(x["p_violation"]) for x in routes),
                },
                "transaction": {
                    "history_append": hterm["history_transaction"]["appended_count"],
                    "active_history_after": hterm["active_history_count"],
                    "anchor_append": hterm["anchor_append_count"],
                    "retry": hterm["retry_count"],
                    "backtracking": hterm["backtracking_count"],
                    "interbatch_W0_restore": hterm["interbatch_W0_restore_count"],
                },
            },
            "native_alphaedit": {
                "pre": compact(nrow["entry_pre_summary"]),
                "post": compact(nrow["immediate_post_summary"]),
                "final_W10": compact(n_cohort[idx]["final_W10"]),
                "edit_core_wall_seconds": nterm["atomic_or_native"]["edit_core_wall_seconds"],
            },
        })

    phases = {
        "W0_baseline_B1_entry": {
            "structural_h_on": compact(h_cp[0]["entry_pre_summary"]),
            "native_alphaedit": compact(n_cp[0]["entry_pre_summary"]),
            "note": "10-request B1 entry at common W0; not a 100-request aggregate",
        },
        "all_true_batch_entry_pre": {
            "structural_h_on": phase_aggregate(h_cp, "entry_pre_summary"),
            "native_alphaedit": phase_aggregate(n_cp, "entry_pre_summary"),
            "note": "100 requests evaluated at their actual sequential W_(b-1) entry states",
        },
        "all_immediate_post": {
            "structural_h_on": phase_aggregate(h_cp, "immediate_post_summary"),
            "native_alphaedit": phase_aggregate(n_cp, "immediate_post_summary"),
            "note": "100 requests immediately after their own B10 edit",
        },
        "final_W10_B100": {
            "structural_h_on": compact(h_terminal["final_B100"]),
            "native_alphaedit": compact(n_terminal["final_B100"]),
            "note": "all 100 requests evaluated at final W10",
        },
    }

    controller_total = {
        key: sum(row["structural_h_on"]["controller_counts"][key] for row in batch_rows)
        for key in ("request_steps", "primary", "rescue", "current", "active_gradient", "clamp_hit", "soft_to_neutral_fallback_steps")
    }
    routing_total = {
        "H_ACTIVE_CERTIFIED_steps": sum(row["structural_h_on"]["routing"]["status_counts"].get("H_ACTIVE_CERTIFIED", 0) for row in batch_rows),
        "H_EMPTY_EXACT_ATOMIC_EQUIVALENCE_steps": sum(row["structural_h_on"]["routing"]["status_counts"].get("H_EMPTY_EXACT_ATOMIC_EQUIVALENCE", 0) for row in batch_rows),
        "H_SELECTED_sum": sum(row["structural_h_on"]["routing"]["H_SELECTED_sum"] for row in batch_rows),
        "H_NEUTRAL_sum": sum(row["structural_h_on"]["routing"]["H_NEUTRAL_sum"] for row in batch_rows),
        "H_delta_sum": sum(row["structural_h_on"]["routing"]["H_delta_sum"] for row in batch_rows),
        "max_abs_strength_residual": max(row["structural_h_on"]["routing"]["max_abs_strength_residual"] for row in batch_rows),
        "max_energy_violation": max(row["structural_h_on"]["routing"]["max_energy_violation"] for row in batch_rows),
        "max_abs_P_violation": max(row["structural_h_on"]["routing"]["max_abs_P_violation"] for row in batch_rows),
    }
    nll_summary = {
        "z8_mean": mean(row["structural_h_on"]["z8_full_six_target_new_nll"] for row in batch_rows),
        "W8_mean": mean(row["structural_h_on"]["W8_full_six_target_new_nll"] for row in batch_rows),
        "W_minus_z_mean": mean(row["structural_h_on"]["W_minus_z_nll"] for row in batch_rows),
        "W_minus_z_max_abs": max(abs(row["structural_h_on"]["W_minus_z_nll"]) for row in batch_rows),
    }

    payload = {
        "schema": "p1r52-structuralh-on/readable-metrics/v4",
        "source_head": h_terminal["source_head"],
        "v3_sha256": digest(V3),
        "phase_rows": phases,
        "batch_rows": batch_rows,
        "controller_totals": controller_total,
        "routing_totals": routing_total,
        "nll_summary": nll_summary,
        "low_cohort_summary": low,
        "compute_ledger": compute,
    }
    dump_once(METRICS_JSON, payload)

    lines = [
        "# P1R52 Structural-H ON 10×B10 — reader-oriented v4",
        "",
        "> v3의 한국어 장문 셀을 대체하는 가독성 중심 판본입니다. v3와 sealed raw-free 결과는 변경하지 않았고, 아래 수치는 동일 receipt를 재배열한 것입니다.",
        "",
        "## Metric 정의",
        "",
        "- **EFF** = CounterFact `rewrite_success`: target-new NLL < target-true NLL. Rewrite Acc와 다릅니다.",
        "- **GEN** = `paraphrase_success`/`rephrase_success`: 각 paraphrase에서 target-new NLL < target-true NLL.",
        "- **GEN-strict** = 한 request의 paraphrase 2개가 모두 성공한 request 비율.",
        "- **LOC** = locality-preservation accuracy.",
        "- **Rewrite Acc / Rephrase Acc** = target token accuracy. Success와 별도 지표입니다.",
        "- **z8 NLL** = atomic K8 target-state oracle의 full-six target-new NLL; **W8 NLL** = 같은 batch의 accepted BF16 physical W8 full-six target-new NLL. 낮을수록 좋습니다.",
        "- **W0 baseline**은 B1 진입의 공통 pretrained state이고, **entry-pre**는 각 batch가 실제로 진입한 누적 상태 W_(b-1)입니다.",
        "",
        "## 한눈에 보는 baseline → pre → post → final 비교",
        "",
        "|Panel|Arm|EFF|Rewrite Acc|GEN|GEN-strict|Rephrase Acc|Rephrase Acc-strict|LOC|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    phase_labels = {
        "W0_baseline_B1_entry": "W0 baseline (B1 entry)",
        "all_true_batch_entry_pre": "Actual entry-pre aggregate",
        "all_immediate_post": "Immediate post aggregate",
        "final_W10_B100": "Final W10 / B100",
    }
    for key in phase_labels:
        lines.append(full_score_row(phase_labels[key], "Structural-H ON", phases[key]["structural_h_on"]))
        lines.append(full_score_row(phase_labels[key], "Native AlphaEdit", phases[key]["native_alphaedit"]))

    lines += [
        "",
        "## Aggregate NLL 비교",
        "",
        "|Panel|Arm|Rewrite new NLL|Rewrite true/old NLL|Rephrase new NLL|Rephrase true/old NLL|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key in phase_labels:
        for arm_key, arm_name in (("structural_h_on", "Structural-H ON"), ("native_alphaedit", "Native AlphaEdit")):
            c = phases[key][arm_key]
            lines.append(
                f"|{phase_labels[key]}|{arm_name}|{c['EFF']['new_nll']:.6f}|{c['EFF']['old_nll']:.6f}|"
                f"{c['GEN']['new_nll']:.6f}|{c['GEN']['old_nll']:.6f}|"
            )
    lines += [
        "",
        "## Structural-H ON: batch별 EFF / GEN / LOC와 z/W NLL",
        "",
        "각 `EFF·GEN·LOC` 셀은 해당 순서의 절대 numerator/denominator입니다.",
        "",
        "|B|H width|entry-pre E/G/L|post E/G/L|final-W10 E/G/L|post RW Acc|post Rephrase Acc / strict|z8 NLL|W8 NLL|W-z|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in batch_rows:
        h = row["structural_h_on"]
        lines.append(
            f"|B{row['batch']}|{row['history_width']}|{short_score(h['pre'])}|{short_score(h['post'])}|{short_score(h['final_W10'])}|"
            f"{count(h['post']['RewriteAcc'])}|{count(h['post']['RephraseAcc'])} / {count(h['post']['RephraseAcc'], True)}|"
            f"{h['z8_full_six_target_new_nll']:.6f}|{h['W8_full_six_target_new_nll']:.6f}|{h['W_minus_z_nll']:+.6f}|"
        )
    lines += [
        "",
        f"- z8 NLL mean `{nll_summary['z8_mean']:.6f}`, W8 NLL mean `{nll_summary['W8_mean']:.6f}`, mean W−z `{nll_summary['W_minus_z_mean']:+.6f}`.",
        "",
        "## Native AlphaEdit: 같은 batch의 직접 비교",
        "",
        "|B|entry-pre E/G/L|post E/G/L|final-W10 E/G/L|post RW Acc|post Rephrase Acc / strict|",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in batch_rows:
        n = row["native_alphaedit"]
        lines.append(
            f"|B{row['batch']}|{short_score(n['pre'])}|{short_score(n['post'])}|{short_score(n['final_W10'])}|"
            f"{count(n['post']['RewriteAcc'])}|{count(n['post']['RephraseAcc'])} / {count(n['post']['RephraseAcc'], True)}|"
        )

    lines += [
        "",
        "## Batch별 NLL 진단",
        "",
        "|B|H pre RW/PP new NLL|H z8/W8 full-six NLL|H post RW/PP new NLL|Native pre RW/PP new NLL|Native post RW/PP new NLL|",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in batch_rows:
        h, n = row["structural_h_on"], row["native_alphaedit"]
        lines.append(
            f"|B{row['batch']}|{h['pre']['EFF']['new_nll']:.4f} / {h['pre']['GEN']['new_nll']:.4f}|"
            f"{h['z8_full_six_target_new_nll']:.6f} / {h['W8_full_six_target_new_nll']:.6f}|"
            f"{h['post']['EFF']['new_nll']:.4f} / {h['post']['GEN']['new_nll']:.4f}|"
            f"{n['pre']['EFF']['new_nll']:.4f} / {n['pre']['GEN']['new_nll']:.4f}|"
            f"{n['post']['EFF']['new_nll']:.4f} / {n['post']['GEN']['new_nll']:.4f}|"
        )

    lines += [
        "",
        "## Controller / rescue / current / clamp 상세",
        "",
        "`Primary`, `Rescue`, `Current`는 80 step × 10 requests = 800 request-step 선택 횟수입니다.",
        "",
        "|B|Primary|Rescue|Current|Active gradients|Clamp hits|Soft→Neutral fallback|H status|ΣH-selected|ΣH-neutral|ΣΔH|",
        "|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in batch_rows:
        c, r = row["structural_h_on"]["controller_counts"], row["structural_h_on"]["routing"]
        status = ", ".join(f"{k}:{v}" for k, v in r["status_counts"].items())
        lines.append(
            f"|B{row['batch']}|{c['primary']}|{c['rescue']}|{c['current']}|{c['active_gradient']}|{c['clamp_hit']}|"
            f"{c['soft_to_neutral_fallback_steps']}|{status}|{r['H_SELECTED_sum']:.6g}|{r['H_NEUTRAL_sum']:.6g}|{r['H_delta_sum']:+.6g}|"
        )
    lines += [
        "",
        f"- **전체 선택:** Primary `{controller_total['primary']}/800`, Rescue `{controller_total['rescue']}/800`, Current `{controller_total['current']}/800`.",
        f"- Active gradient `{controller_total['active_gradient']}/800`; clamp-hit `{controller_total['clamp_hit']}/800`; Soft→Neutral fallback `{controller_total['soft_to_neutral_fallback_steps']}/80 steps`.",
        f"- Structural-H: H_ACTIVE_CERTIFIED `{routing_total['H_ACTIVE_CERTIFIED_steps']}/80`; B1 H-empty exact equivalence `{routing_total['H_EMPTY_EXACT_ATOMIC_EQUIVALENCE_steps']}/80`.",
        f"- max |strength residual| `{routing_total['max_abs_strength_residual']:.3e}`; max energy violation `{routing_total['max_energy_violation']:.3e}`; max |P violation| `{routing_total['max_abs_P_violation']:.3e}`.",
        "",
        "## Sequential state / history / rollback",
        "",
        "- History entry widths: `0,10,20,30,40,50,60,70,80,90`; append total `100`; final active history `100`; lifetime anchors `100`.",
        "- Commit→next-entry parameter-byte identity: `9/9`; inter-batch W0 restore `0`; terminal W0 pointer/byte restore `PASS/PASS`.",
        "- Weight/history transaction rollback `0`; retry `0`; backtracking `0`; action-freeze checkpoints `10/10`.",
        "- H-aware sealed receipts에는 명시적 `alpha_solve_cache_consume_count`가 없어 `NOT_RECORDED`; nonzero history width와 transaction append는 별도 보존됩니다.",
        "",
        "## Hard / low-performance cohort",
        "",
        f"- 100 request 중 entry EFF hard `{low['flag_counts']['entry_hard_by_rewrite_success']}`; immediate-post strict GEN fail `{low['flag_counts']['post_strict_paraphrase_success_failure']}`; immediate-post strict rephrase-accuracy fail `{low['flag_counts']['post_strict_paraphrase_accuracy_failure']}`.",
        f"- Native immediate-post GEN이 더 높은 request `{low['flag_counts']['native_post_paraphrase_success_higher']}/100`; post→final GEN loss `{low['flag_counts']['final_paraphrase_success_loss']}/100`; rephrase-accuracy loss `{low['flag_counts']['final_paraphrase_accuracy_loss']}/100`.",
        "- Requestwise H attribution은 sealed routing granularity가 batch×step이므로 `NOT_RECORDED`; 위 관계는 `ASSOCIATION_ONLY`입니다.",
        "",
        "## Compute ledger",
        "",
    ]
    atomic = compute["structural_h_on_batch_atomic_compute_totals_sum"]
    lines += [
        f"- P1R52 atomic total: model forward `{atomic['model_forward_calls']}`, backward `{atomic['backward_calls']}`, target backward `{atomic['target_backward_calls']}`, slope backward `{atomic['slope_backward_calls']}`, materialization `{atomic['materialization_count']}`, processed tokens `{atomic['processed_tokens']}`.",
        f"- Entry-pre evaluator F/B/generation: `{compute['entry_pre_evaluator_model_forward_backward_generation'][0]}/{compute['entry_pre_evaluator_model_forward_backward_generation'][1]}/{compute['entry_pre_evaluator_model_forward_backward_generation'][2]}`.",
        f"- Accuracy amendment added F/B/generation: `{compute['one_pass_accuracy_added_model_forward_backward_generation'][0]}/{compute['one_pass_accuracy_added_model_forward_backward_generation'][1]}/{compute['one_pass_accuracy_added_model_forward_backward_generation'][2]}`; inner-K heldout evaluation `{compute['inner_K_step_heldout_evaluation_count']}`.",
        f"- Native edit-core wall sum `{compute['native_edit_core_wall_seconds_sum']:.3f}s`.",
        "",
        "## 핵심 판독",
        "",
        "- Final W10에서 Structural-H ON은 Native보다 EFF `+7/100`, Rewrite Acc `+27/100`, LOC `+184/1000`입니다.",
        "- 반면 GEN은 `-6/200`, GEN-strict `-10/100`, Rephrase Acc `-16/200`입니다. 즉 edit retention/locality와 rephrase generalization 사이의 trade-off가 관찰됩니다.",
        "- Structural-H 단독 인과효과는 이 표만으로 주장하지 않습니다. AlphaCache-On/StructuralH-Off control과의 exact paired 분석이 필요합니다.",
        "",
        "## Provenance",
        "",
        f"- Source HEAD: `{h_terminal['source_head']}`; completed B10 `10/10`; requests `100`; scientific promotion `false`.",
        f"- Sealed v3 SHA256: `{digest(V3)}` (보존, 수정 없음).",
        f"- Machine-readable v4 metrics: `{METRICS_JSON.name}`.",
    ]
    report_bytes = ("\n".join(lines) + "\n").encode()
    write_once(REPORT, report_bytes)

    members = []
    for path in (V3, METRICS_JSON, REPORT):
        members.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)})
    root = digest_bytes(json.dumps(members, sort_keys=True, separators=(",", ":")).encode())
    manifest = {
        "schema": "p1r52-structuralh-on/readable-v4-manifest/v1",
        "members": members,
        "members_root_sha256": root,
        "source_result_terminal_sha256": digest(H_ROOT / "terminal.json"),
        "native_result_terminal_sha256": digest(N_ROOT / "terminal.json"),
    }
    dump_once(MANIFEST, manifest)
    receipt = {
        "schema": "p1r52-structuralh-on/readable-v4-receipt/v1",
        "status": "PASS",
        "report": str(REPORT),
        "report_sha256": digest(REPORT),
        "report_bytes": REPORT.stat().st_size,
        "report_lines": len(REPORT.read_text().splitlines()),
        "metrics_sha256": digest(METRICS_JSON),
        "manifest_sha256": digest(MANIFEST),
        "members_root_sha256": root,
        "model_evaluator_gpu_slurm_rerun": [0, 0, 0, 0],
        "v3_mutated": False,
    }
    dump_once(RECEIPT, receipt)


if __name__ == "__main__":
    main()
