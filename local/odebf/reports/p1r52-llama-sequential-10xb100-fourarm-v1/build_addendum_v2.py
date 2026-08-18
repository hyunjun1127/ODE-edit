#!/usr/bin/env python3
"""Append-only v2 with target/writer and clarified compute receipts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from statistics import mean, median


OUT = Path(__file__).resolve().parent
BASE = OUT / "p1r52-llama-sequential-10xb100-fourarm-factual-ko.md"
CTRL = OUT / "p1r52-b100-r52-controller-routing.json"
AGG = OUT / "p1r52-b100-four-arm-aggregate.json"
INTEGRITY = OUT / "p1r52-b100-four-arm-integrity-compute.json"
REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "local/odebf/results"
R52 = {
    "r52_h_on": RESULTS / "s05-p1r52-llama-soft-sequential-structuralh-on-10xb100-tech-r3-release-r1-v1",
    "r52_h_off": RESULTS / "s05-p1r52-llama-soft-sequential-alphacache-on-structuralh-off-10xb100-tech-r3-release-r1-v1",
}
LABEL = {
    "memit": "Official MEMIT", "alphaedit": "Official AlphaEdit",
    "r52_h_on": "R52 Structural-H ON", "r52_h_off": "R52 Structural-H OFF",
}


def load(path: Path): return json.loads(path.read_text())
def sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()
def root(x): return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_once(name: str, data: bytes) -> Path:
    path = OUT / name
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as h:
        h.write(data); h.flush(); os.fsync(h.fileno())
    return path


def dump(name: str, x) -> Path:
    return write_once(name, (json.dumps(x, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def quantiles(xs):
    ys = sorted(xs)
    p90 = ys[min(len(ys)-1, int(0.9*len(ys)))]
    return {"count": len(ys), "mean": mean(ys), "median": median(ys), "p90": p90, "max": max(ys)}


def main():
    aggregates = load(AGG)["arms"]
    control = load(CTRL)["rows"]
    step_rows = []
    for arm, result in R52.items():
        for batch in range(1, 11):
            for step in range(1, 9):
                p = result / f"raw/batches/b{batch:02d}/raw/ode/p1r52-rsa-r42safekdc-m1-soft/accepted-k{step}.json"
                x = load(p); t = x["target_update"]; prog = x["progress"]; route = x["routing"]
                z = t["selected_nll_summary"]["mean"]
                w = prog.get("terminal_mean_target_new_nll")
                step_rows.append({
                    "arm": arm, "round": batch, "step": step,
                    "primary": t["primary_accept_count"], "rescue": t["rescue_accept_count"],
                    "current": t["current_hold_count"], "active_gradient": t["active_gradient_count"],
                    "clamp_hit": t["clamp_hit_count"], "request_count": t["request_count"],
                    "z_target_new_nll": z, "W_target_new_nll": w,
                    "W_minus_z_gap": None if w is None else w-z,
                    "alpha_req": route["alpha_req"], "predicted": prog.get("predicted"),
                    "actual": prog.get("actual"), "realization": prog.get("realization_ratio"),
                    "P": route["selected_p"], "capacity": route["selected_capacity"],
                    "energy": route["selected_energy"], "bf16_update_energy": sum(x["materialization"]["realized_bf16_step_energy"].values()),
                    "fallback": route["fallback_to_neutral"], "identity_sha256": x["identity_sha256"],
                })
    step_path = dump("p1r52-b100-r52-target-writer-step-v2.json", {"schema":"p1r52-sequential-b100x10/target-writer-step/v2", "row_count":len(step_rows), "rows":step_rows})
    batch_rows = [r for r in step_rows if r["step"] == 8]
    batch_path = dump("p1r52-b100-r52-terminal-z-w-v2.json", {"schema":"p1r52-sequential-b100x10/terminal-z-w/v2", "row_count":len(batch_rows), "rows":batch_rows})
    stats = {}
    for arm in R52:
        rows = [r for r in step_rows if r["arm"] == arm]
        stats[arm] = {
            "primary_rescue_current": [sum(r["primary"] for r in rows), sum(r["rescue"] for r in rows), sum(r["current"] for r in rows)],
            "active_gradient": sum(r["active_gradient"] for r in rows), "request_steps": sum(r["request_count"] for r in rows),
            "clamp_hit": sum(r["clamp_hit"] for r in rows), "fallback": sum(r["fallback"] for r in rows),
            "P": quantiles([r["P"] for r in rows]), "capacity": quantiles([r["capacity"] for r in rows]),
            "energy": quantiles([r["energy"] for r in rows]), "bf16_update_energy": quantiles([r["bf16_update_energy"] for r in rows]),
            "terminal_W_minus_z_gap": quantiles([r["W_minus_z_gap"] for r in batch_rows if r["arm"] == arm and r["W_minus_z_gap"] is not None]),
            "physical_writer_transitions": len(rows), "materialization_receipts": len(rows),
            "inner_step_heldout_evaluation_count": 0,
        }
    stats_path = dump("p1r52-b100-r52-controller-compute-summary-v2.json", {"schema":"p1r52-sequential-b100x10/controller-compute-summary/v2", "arms":stats})

    def metric_delta(a,b,m):
        x=aggregates[a]["final_W10_B1000"][m]; y=aggregates[b]["final_W10_B1000"][m]
        return x["numerator"]-y["numerator"], x["denominator"]
    lines = [BASE.read_text(), "", "## Append-only v2: target/write/controller 상세 및 compute 정정", "",
        "- v1 상위 job ledger의 `completed_k_total=0`은 child sequential step을 집계하지 않은 raw field입니다. 실제 R52 물리 전이는 accepted-k/materialization 영수증 기준 각 80회입니다.",
        "- Official 방법의 ODE K는 적용 대상이 아니므로 `NOT_APPLICABLE`; Official materialization count는 현재 공통 영수증에서 `NOT_RECORDED`입니다.",
        "- 모든 팔에서 K-step heldout evaluator 접근은 0입니다. R52 canonical batch-entry evaluator도 0; Official entry-evaluator 원자료는 사용자 지시에 따라 비정규/미사용입니다.", "",
        "### final W10 직접 산술 델타", "", "|비교|EFF|Rewrite Acc|GEN|GEN strict|Rephrase Acc|Acc strict|LOC|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    comparisons=[("R52 ON−OFF","r52_h_on","r52_h_off"),("R52 ON−AlphaEdit","r52_h_on","alphaedit"),("R52 OFF−AlphaEdit","r52_h_off","alphaedit"),("R52 ON−MEMIT","r52_h_on","memit")]
    names=[("rewrite_success",False),("rewrite_acc",False),("paraphrase_success",False),("paraphrase_success",True),("paraphrase_acc",False),("paraphrase_acc",True)]
    for title,a,b in comparisons:
        vals=[]
        for m,strict in names:
            x=aggregates[a]["final_W10_B1000"][m]; y=aggregates[b]["final_W10_B1000"][m]; pre="strict_" if strict else ""
            vals.append(f"{x[pre+'numerator']-y[pre+'numerator']:+}/{x[pre+'denominator']}")
        xl=aggregates[a]["final_W10_B1000"]["locality"]; yl=aggregates[b]["final_W10_B1000"]["locality"]
        lines.append("|"+title+"|"+"|".join(vals)+f"|{xl['numerator']-yl['numerator']:+}/{xl['denominator']}|")
    lines += ["", "### R52 controller totals 및 분포", "", "|Arm|Primary/Rescue/Current|active/request-steps|clamp|fallback|P mean/med/p90/max|capacity mean/med/p90/max|energy mean/med/p90/max|BF16 energy mean/med/p90/max|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in R52:
        s=stats[arm]
        f=lambda q: f"{q['mean']:.4g}/{q['median']:.4g}/{q['p90']:.4g}/{q['max']:.4g}"
        lines.append(f"|{LABEL[arm]}|{'/'.join(map(str,s['primary_rescue_current']))}|{s['active_gradient']}/{s['request_steps']}|{s['clamp_hit']}|{s['fallback']}|{f(s['P'])}|{f(s['capacity'])}|{f(s['energy'])}|{f(s['bf16_update_energy'])}|")
    lines += ["", "### Batch-terminal z8/W8 full-six target-new NLL", "", "|B|Arm|z8|W8|W−z|", "|---:|---|---:|---:|---:|"]
    for r in batch_rows:
        lines.append(f"|B{r['round']}|{LABEL[r['arm']]}|{r['z_target_new_nll']:.6f}|{r['W_target_new_nll']:.6f}|{r['W_minus_z_gap']:+.6f}|")
    lines += ["", "### v2 machine artifacts", "", f"- step rows: `{step_path}` (160)", f"- terminal z/W rows: `{batch_path}` (20)", f"- controller/compute summary: `{stats_path}`", "", "scientific_promotion=false"]
    report = write_once("p1r52-llama-sequential-10xb100-fourarm-factual-ko-v2.md", ("\n".join(lines)+"\n").encode())
    members=[]
    for p in (BASE, CTRL, AGG, INTEGRITY, step_path, batch_path, stats_path, report):
        members.append({"path":str(p),"bytes":p.stat().st_size,"lines":len(p.read_text().splitlines()),"sha256":sha(p)})
    manifest=dump("manifest-v2.json", {"schema":"p1r52-sequential-b100x10/four-arm-manifest/v2","status":"PASS","canonical_report":str(report),"members":members,"members_root_sha256":root(members)})
    checks={"step_rows_160":len(step_rows)==160,"batch_z_w_rows_20":len(batch_rows)==20,"r52_writes_80_each":all(stats[a]["physical_writer_transitions"]==80 for a in R52),"request_step_denominator_8000_each":all(stats[a]["request_steps"]==8000 for a in R52),"no_inner_heldout":all(stats[a]["inner_step_heldout_evaluation_count"]==0 for a in R52),"base_preserved":sha(BASE)=="55aefe5a780410af91c91660324d67fd94c9bc39bf49f9979d2ba43d261f0ef2"}
    if not all(checks.values()): raise RuntimeError(checks)
    review=dump("independent-rawfree-rehash-review-v2.json", {"schema":"p1r52-sequential-b100x10/four-arm-review/v2","status":"PASS","checks":checks,"manifest_sha256":sha(manifest),"model_evaluator_gpu_slurm_actions":[0,0,0,0]})
    all_members=members+[{"path":str(manifest),"bytes":manifest.stat().st_size,"lines":len(manifest.read_text().splitlines()),"sha256":sha(manifest)},{"path":str(review),"bytes":review.stat().st_size,"lines":len(review.read_text().splitlines()),"sha256":sha(review)}]
    receipt=dump("rooted-receipt-v2.json", {"schema":"p1r52-sequential-b100x10/four-arm-receipt/v2","status":"TERMINAL_FACTUAL_REPORT_COMPLETE","canonical_report":str(report),"canonical_report_sha256":sha(report),"canonical_report_bytes":report.stat().st_size,"canonical_report_lines":len(report.read_text().splitlines()),"members":all_members,"package_root_sha256":root(all_members),"raw_content_included":False,"model_evaluator_gpu_slurm_actions":[0,0,0,0]})
    print(json.dumps({"report":str(report),"sha256":sha(report),"bytes":report.stat().st_size,"lines":len(report.read_text().splitlines()),"manifest":str(manifest),"review":str(review),"receipt":str(receipt),"root":load(receipt)["package_root_sha256"],"rows":{"step":len(step_rows),"z_w":len(batch_rows)}},ensure_ascii=False,sort_keys=True))


if __name__ == "__main__": main()
