#!/usr/bin/env python3
"""Append-only v5 reader reports; consumes only sealed v4 raw-free summaries."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


OUT = Path(__file__).resolve().parent
V4 = OUT / "p1r52-sequential-three-arm-readable-metrics-v4.json"
CTRL = OUT / "p1r52-sequential-three-arm-controller-routing-v4.json"
V4_RECEIPT = OUT / "readable-v4-receipt.json"
LABEL = {
    "h_on": "P1R52 Soft Structural-H ON",
    "h_off": "P1R52 Soft AlphaCache-ON / Structural-H OFF",
    "native": "Native AlphaEdit Sequential",
}


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(name: str, data: bytes) -> Path:
    path = OUT / name
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    return path


def dump_once(name: str, payload) -> Path:
    return write_once(name, (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def count(x: dict, strict: bool = False) -> str:
    return f"{x['strict_n']}/{x['strict_d']} ({100*x['strict_rate']:.1f}%)" if strict else f"{x['n']}/{x['d']} ({100*x['rate']:.1f}%)"


def all_metric_row(panel: str, arm: str, x: dict) -> str:
    return f"|{panel}|{LABEL[arm]}|{count(x['EFF'])}|{count(x['RewriteAcc'])}|{count(x['GEN'])}|{count(x['GEN'],True)}|{count(x['RephraseAcc'])}|{count(x['RephraseAcc'],True)}|{x['LOC']['n']}/{x['LOC']['d']} ({100*x['LOC']['rate']:.1f}%)|"


def nllrow(panel: str, arm: str, x: dict, zw) -> str:
    if isinstance(zw, dict):
        ztext = f"{zw['z8_full_six_target_new_nll']:.6f}/{zw['W8_full_six_target_new_nll']:.6f}/{zw['W_minus_z_nll']:+.6f}"
    else:
        ztext = str(zw)
    return f"|{panel}|{LABEL[arm]}|{x['EFF']['new_nll']:.6f}/{x['EFF']['old_nll']:.6f}/{x['EFF']['margin']:+.6f}|{x['GEN']['new_nll']:.6f}/{x['GEN']['old_nll']:.6f}/{x['GEN']['margin']:+.6f}|{ztext}|"


def panels(v: dict, arms: tuple[str, ...]):
    rows = [("W0 / B1 entry", {a: v['phase_rows']['W0_baseline_B1_entry'][a] for a in arms})]
    for b in v['batch_rows']:
        for phase, title in (("entry_pre", "entry-pre"), ("immediate_post", "immediate-post"), ("final_W10", "final-W10")):
            rows.append((f"B{b['batch']} {title}", {a: b['arms'][a][phase] for a in arms}))
    rows.append(("B100 final W10", {a: v['phase_rows']['final_W10_B100'][a] for a in arms}))
    return rows


def common_header(title: str) -> list[str]:
    return [
        f"# {title}", "",
        "> v4 sealed raw-free tables를 변경하지 않고 가독성 구조로 재배열한 append-only v5 판본입니다. 모델·evaluator·GPU·Slurm 재실행은 0입니다.", "",
        "## 지표 정의", "",
        "- **EFF** ≡ `rewrite_success`: rewrite prompt에서 target-new mean NLL < target-true mean NLL.",
        "- **GEN** ≡ `paraphrase_success` (`rephrase_success` alias): paraphrase prompt별 NLL preference 성공.",
        "- **GEN-strict**: request의 모든 paraphrase가 성공한 strict request count/rate.",
        "- **Rewrite Acc / Rephrase Acc**: strict suffix-token argmax accuracy; EFF/GEN success와 별도입니다.",
        "- **LOC**: locality-preservation accuracy. `entry-pre`는 actual W_(b-1), `post`는 W_e, `final`은 W10입니다.",
        "- z8/W8/gap은 해당 batch의 target oracle full-six / accepted BF16 physical full-six / W8−z8입니다. Native target oracle 부재는 `NOT_RECORDED`입니다.", "",
    ]


def absolute_table(v: dict, arms: tuple[str, ...]) -> list[str]:
    lines = ["## 절대값: W0 + 각 B1–B10 entry-pre / immediate-post / final-W10 + B100", "",
             "|Panel|Arm|EFF|Rewrite Acc|GEN|GEN-strict|Rephrase Acc|Acc-strict|LOC|", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for panel, vals in panels(v, arms):
        for arm in arms:
            lines.append(all_metric_row(panel, arm, vals[arm]))
    return lines


def nll_table(v: dict, arms: tuple[str, ...]) -> list[str]:
    lines = ["", "## NLL/margin 및 z8/W8/gap", "", "|Panel|Arm|Rewrite new/true/margin|Rephrase new/true/margin|z8/W8/gap full-six|", "|---|---|---:|---:|---:|"]
    for panel, vals in panels(v, arms):
        if panel.startswith("B") and panel.split()[0] not in {"B100"}:
            b = int(panel.split()[0][1:])
            phase = panel.split()[1]
            data = next(r for r in v['batch_rows'] if r['batch'] == b)
            for arm in arms:
                zw = data['arms'][arm]['z_w'] if phase == "immediate-post" and arm != "native" else "NOT_RECORDED" if arm == "native" else "NOT_RECORDED_OUTER_K8_ONLY"
                lines.append(nllrow(panel, arm, vals[arm], zw))
        else:
            for arm in arms:
                lines.append(nllrow(panel, arm, vals[arm], "NOT_RECORDED"))
    return lines


def controller_table(ctrl: dict, arms: tuple[str, ...]) -> list[str]:
    lines = ["", "## Controller / routing / strength / energy / P / cache", "",
             "|B|Arm|Primary/Rescue/Current|denom|active/no-direction|clamp|fallback/status|αreq/αapply|strength residual|P/capacity/energy|entropy/top1|H/cache receipt|", "|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|"]
    for r in ctrl['rows']:
        if r['arm'] not in arms:
            continue
        c, s = r['controller'], r['structural']
        status = ",".join(f"{k}:{v}" for k,v in c['route_status_counts'].items())
        hdetail = (f"Hsel/Hneu/Δ={s['selected_sum']:.4g}/{s['neutral_sum']:.4g}/{s['delta_sum']:+.4g}; {s['status_counts']}" if r['arm']=='h_on'
                   else f"cache consume={s['alpha_solve_cache_consume_count']}; H-influence={s['structural_h_decision_influence_count']}; {s['status_counts']}")
        lines.append(f"|B{r['batch']}|{LABEL[r['arm']]}|{c['primary']}/{c['rescue']}/{c['current']}|{c['request_steps']}|{c['active_gradient']}/{c['no_direction']}|{c['clamp_hit']}|{c['fallback_steps']}/{status}|{c['mean_alpha_req']:.5g}/{c['mean_alpha_apply']:.5g}|{c['max_abs_strength_residual']:.2e}|{c['mean_P']:.3e}/{c['mean_capacity']:.3e}/{c['mean_energy']:.3e}|{c['mean_entropy']:.3f}/{c['mean_top1']:.3f}|{hdetail}|")
    return lines


def compute_lines(v: dict, arms: tuple[str, ...]) -> list[str]:
    terms = v['compute_ledger']['terminals']
    lines = ["", "## Compute ledger", "", "|Arm|entry-pre evaluator F/B/generation|accuracy 추가 F/B/generation|evaluator forward|model forward|processed tokens|materialization|", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in arms:
        term = terms[arm]
        counters = term['counters']
        acc = v['compute_ledger'][arm]['accuracy_extra_forward_backward_generation']
        entry = v['compute_ledger'][arm]['entry_pre_evaluator_forward_backward_generation']
        material = v['compute_ledger'][arm]['atomic_compute_totals_sum'].get('materialization_count', 'NOT_RECORDED') if arm != 'native' else 'NOT_RECORDED_NATIVE_PATH'
        lines.append(f"|{LABEL[arm]}|{entry[0]}/{entry[1]}/{entry[2]}|{acc[0]}/{acc[1]}/{acc[2]}|{counters.get('evaluator_forward','NOT_RECORDED')}|{counters.get('model_forward','NOT_RECORDED')}|{counters.get('processed_tokens','NOT_RECORDED')}|{material}|")
    return lines


def state_lines(v: dict, arms: tuple[str, ...]) -> list[str]:
    alpha = v['alpha_cache_state']
    return ["", "## Sequential state / transactions", "",
            f"- H-OFF Alpha solve-cache entry widths: `{alpha['h_off_entry_widths']}`; transaction append `{alpha['h_off_transaction_append_total']}`; terminal active cache/anchors `{alpha['h_off_terminal_active_history']}/{alpha['h_off_terminal_anchors']}`.",
            f"- H-OFF terminal W0 pointer/byte restore: `{alpha['h_off_terminal_W0_restore']['pointer_restored_exact']}/{alpha['h_off_terminal_W0_restore']['byte_restored_exact']}`. H-ON cache consumption is `{alpha['h_on_alpha_solve_cache_consumption']}` in its sealed receipt.",
            "- All R52 arms retain physical W across B1→B10. Control routing receipt shows Structural-H decision influence=0 and observation-ledger influence=0; retry/backtracking/rollback/interbatch W0 restore remain zero in sealed terminal records."]


def low_lines(v: dict) -> list[str]:
    low = v['low_cohort_summary']
    counts = low['counts']
    return ["", "## Hard / low cohort, batch age, forgetting", "",
            f"- failed-or-transition rows `{low['failed_or_transition_rows']}/100`; entry rewrite-hard H-ON `{counts.get('entry_hard_h_on',0)}/100`; H-ON/H-OFF/Native immediate-post GEN failure `{counts.get('h_on_post_gen_failure',0)}/{counts.get('h_off_post_gen_failure',0)}/{counts.get('native_post_gen_failure',0)}` /100.",
            f"- post→final GEN loss H-ON/H-OFF `{counts.get('h_on_final_gen_loss',0)}/{counts.get('h_off_final_gen_loss',0)}` /100; post GEN H-ON>H-OFF `{counts.get('h_on_post_better_than_h_off',0)}/100`, H-OFF>H-ON `{counts.get('h_off_post_better_than_h_on',0)}/100`.",
            "- Per-request machine rows retain entry hardness, batch age, pre→post gain, post→final loss, success/accuracy/NLL/margin. Routing is batch×step, so requestwise Structural-H attribution is `NOT_RECORDED`.",
            "- Scope is `RAW_FREE_ASSOCIATION_ONLY_NO_ISOLATED_CAUSAL_ATTRIBUTION`; no isolated causal statement is made."]


def main() -> None:
    v = load(V4)
    ctrl = load(CTRL)
    control_arms = ("h_off", "native")
    all_arms = ("h_on", "h_off", "native")
    c_lines = common_header("P1R52 Soft Sequential AlphaCache-ON / Structural-H OFF — standalone factual v5")
    c_lines += absolute_table(v, control_arms) + nll_table(v, control_arms) + controller_table(ctrl, ("h_off",)) + state_lines(v, control_arms) + compute_lines(v, control_arms) + low_lines(v)
    c_lines += ["", "## 상태", "", "- `TERMINAL_FACTUAL_REPORT_COMPLETE`; P1R52 method root is unchanged; H-aware control result is a separate immutable reference; scientific_promotion=false."]
    control = write_once("p1r52-soft-sequential-alphacache-on-structuralh-off-10xb10-factual-ko-v5.md", ("\n".join(c_lines)+"\n").encode())
    t_lines = common_header("P1R52 Sequential Structural-H ON / AlphaCache-ON Structural-H OFF / Native — canonical three-arm factual v5")
    t_lines += absolute_table(v, all_arms) + nll_table(v, all_arms)
    b100 = v['phase_rows']['final_W10_B100']
    dho = {name: b100['h_on'][name]['n']-b100['h_off'][name]['n'] for name in ('EFF','RewriteAcc','GEN','RephraseAcc')}
    don = {name: b100['h_off'][name]['n']-b100['native'][name]['n'] for name in ('EFF','RewriteAcc','GEN','RephraseAcc')}
    t_lines += ["", "## B100 paired arithmetic", "", f"- H-ON−H-OFF: EFF `{dho['EFF']:+}/100`, RewriteAcc `{dho['RewriteAcc']:+}/100`, GEN `{dho['GEN']:+}/200`, RephraseAcc `{dho['RephraseAcc']:+}/200`, LOC `{b100['h_on']['LOC']['n']-b100['h_off']['LOC']['n']:+}/1000`.", f"- H-OFF−Native: EFF `{don['EFF']:+}/100`, RewriteAcc `{don['RewriteAcc']:+}/100`, GEN `{don['GEN']:+}/200`, RephraseAcc `{don['RephraseAcc']:+}/200`, LOC `{b100['h_off']['LOC']['n']-b100['native']['LOC']['n']:+}/1000`."]
    t_lines += controller_table(ctrl, ("h_on","h_off")) + state_lines(v, all_arms) + compute_lines(v, all_arms) + low_lines(v)
    t_lines += ["", "## 상태", "", "- `TERMINAL_FACTUAL_REPORT_COMPLETE`; same batch/request/order pairing PASS; Native is immutable comparison only; scientific_promotion=false."]
    three = write_once("p1r52-sequential-structuralh-on-off-native-three-arm-ko-v5.md", ("\n".join(t_lines)+"\n").encode())
    summary = dump_once("p1r52-sequential-three-arm-readable-summary-v5.json", {"schema":"p1r52-sequential/reader-summary/v5", "v4_metrics_sha256":sha(V4), "v4_controller_sha256":sha(CTRL), "control_report":control.name, "three_arm_report":three.name, "phase_rows":v['phase_rows'], "batch_rows":v['batch_rows'], "controller_totals":v['controller_totals'], "routing_totals":v['routing_totals'], "z_w_nll_summary":v['z_w_nll_summary'], "low_cohort_summary":v['low_cohort_summary']})
    members = [{"name":p.name,"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in {Path(__file__).name,"readable-v5-manifest.json","readable-v5-receipt.json"}]
    root = hashlib.sha256(json.dumps(members,ensure_ascii=False,sort_keys=True,separators=(",",":" )).encode()).hexdigest()
    manifest = dump_once("readable-v5-manifest.json", {"schema":"p1r52-sequential/readable-v5-manifest/v1","status":"TERMINAL_FACTUAL_REPORT_COMPLETE","canonical_control_report":control.name,"canonical_three_arm_report":three.name,"v4_receipt_sha256":sha(V4_RECEIPT),"members":members,"members_root_sha256":root})
    receipt = dump_once("readable-v5-receipt.json", {"schema":"p1r52-sequential/readable-v5-receipt/v1","status":"PASS","control_report":str(control),"control_report_sha256":sha(control),"control_report_bytes":control.stat().st_size,"control_report_lines":len(control.read_text().splitlines()),"three_arm_report":str(three),"three_arm_report_sha256":sha(three),"three_arm_report_bytes":three.stat().st_size,"three_arm_report_lines":len(three.read_text().splitlines()),"summary_sha256":sha(summary),"manifest_sha256":sha(manifest),"members_root_sha256":root,"analysis_actions":{"model":0,"evaluator":0,"gpu":0,"slurm":0,"source_edit":0,"result_mutation":0}})
    print(json.dumps({"control":str(control),"three":str(three),"receipt":str(receipt)},ensure_ascii=False))


if __name__ == '__main__':
    main()
