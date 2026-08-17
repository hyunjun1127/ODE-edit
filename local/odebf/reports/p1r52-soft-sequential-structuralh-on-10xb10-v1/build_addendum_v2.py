#!/usr/bin/env python3
"""Append-only aggregate completion for the Structural-H-on standalone package."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
H = ROOT / "local/odebf/results/s05-p1r52-llama-soft-sequential-historical-10xb10-tech-r3-v1/terminal.json"
N = ROOT / "local/odebf/results/s05-p1r52-native-alphaedit-sequential-10xb10-v1/terminal.json"
METRICS = ("rewrite_success", "rewrite_acc", "paraphrase_success", "paraphrase_acc")


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_new(name, value):
    p = OUT / name
    if p.exists():
        raise RuntimeError(f"create-once exists: {p}")
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.chmod(p, 0o600)
    return p


def compact(s):
    o = {"identity_sha256": s["identity_sha256"], "request_count": s["request_count"], "batch_count": s["batch_count"]}
    for m in METRICS:
        x = s["metrics"][m]
        o[m] = {k: x[k] for k in ("prompt_numerator", "prompt_denominator", "prompt_rate", "strict_request_numerator", "strict_request_denominator", "strict_request_rate")}
        o[m]["target_new_nll_mean"] = x["target_new_nll"]["mean"]
        o[m]["target_old_nll_mean"] = x["target_old_nll"]["mean"]
        o[m]["target_old_minus_new_margin_mean"] = x["target_old_minus_new_margin"]["mean"]
    o["locality"] = {k: s["locality"][k] for k in ("numerator", "denominator", "rate")}
    return o


def delta(a, b):
    r = {}
    for m in METRICS:
        r[m] = {k: a[m][k] - b[m][k] for k in ("prompt_numerator", "prompt_rate", "strict_request_numerator", "strict_request_rate", "target_new_nll_mean", "target_old_nll_mean", "target_old_minus_new_margin_mean")}
    r["locality"] = {k: a["locality"][k] - b["locality"][k] for k in ("numerator", "rate")}
    return r


def count(m, strict=False):
    pre = "strict_request_" if strict else "prompt_"
    return f"{m[pre+'numerator']}/{m[pre+'denominator']} ({m[pre+'rate']:.3f})"


def line(c):
    return f"RW성공 {count(c['rewrite_success'])}; RW정확 {count(c['rewrite_acc'])}; PP성공 {count(c['paraphrase_success'])}/엄격 {count(c['paraphrase_success'], True)}; PP정확 {count(c['paraphrase_acc'])}/엄격 {count(c['paraphrase_acc'], True)}; Loc {c['locality']['numerator']}/{c['locality']['denominator']} ({c['locality']['rate']:.3f})"


def main():
    h, n = load(H), load(N)
    phases = (("true_entry_pre_all_B10", "10개 실제 batch-entry pre"), ("immediate_post_all_B10", "10개 immediate post"), ("final_B100", "최종 W10/B100"))
    rows = []
    for key, display in phases:
        hc, nc = compact(h[key]), compact(n[key])
        rows.append({"phase": key, "display": display, "structural_h_on": hc, "native_alphaedit": nc, "h_minus_native": delta(hc, nc)})
    table = dump_new("p1r52-structuralh-on-aggregate-phase-table.json", {"schema": "p1r52-structuralh-on/aggregate-phase/v1", "row_count": len(rows), "rows": rows})

    base = OUT / "p1r52-soft-sequential-structuralh-on-10xb10-factual-ko.md"
    v2 = OUT / "p1r52-soft-sequential-structuralh-on-10xb10-factual-ko-v2.md"
    if v2.exists():
        raise RuntimeError(f"create-once exists: {v2}")
    extra = [
        "", "## 10개 true entry-pre·immediate post·final B100 집계", "",
        "아래 집계는 각 B10의 실제 batch-entry W 상태를 그대로 측정한 10개 pre panel, 각 batch 직후 10개 post panel, W10에서 100 요청을 다시 평가한 B100 panel입니다. 분자/분모/율은 절대값입니다.",
        "", "|Panel|Structural-H On|Native AlphaEdit|H-aware−Native 산술 차이|", "|---|---|---|---|",
    ]
    for r in rows:
        d = r["h_minus_native"]
        extra.append(f"|{r['display']}|{line(r['structural_h_on'])}|{line(r['native_alphaedit'])}|RW성공 {d['rewrite_success']['prompt_numerator']:+}; RW정확 {d['rewrite_acc']['prompt_numerator']:+}; PP성공 {d['paraphrase_success']['prompt_numerator']:+}; PP엄격 {d['paraphrase_success']['strict_request_numerator']:+}; PP정확 {d['paraphrase_acc']['prompt_numerator']:+}; PP정확엄격 {d['paraphrase_acc']['strict_request_numerator']:+}; Loc {d['locality']['numerator']:+}|" )
    extra += [
        "", "## v2 봉인 범위", "",
        "- v2는 원본 standalone 보고서에 누락되지 않도록 10개 entry-pre/10개 immediate-post/B100 집계를 append-only로 완성한 canonical report입니다.",
        "- 개별 B1–B10 pre→post→final-W10 절대값은 v2 앞부분, batch-age 및 request-level true trajectory는 machine table에 보존됩니다.",
    ]
    v2.write_text(base.read_text() + "\n".join(extra) + "\n")
    os.chmod(v2, 0o600)

    source_manifest = load(OUT / "p1r52-structuralh-on-report-manifest.json")
    v2manifest = {
        "schema": "p1r52-structuralh-on/report-manifest-v2/v1",
        "canonical_report": v2.name,
        "canonical_scope": "P1R52-Soft-Sequential-StructuralH-On-10xB10",
        "base_report": {"name": base.name, "sha256": sha(base), "bytes": base.stat().st_size, "lines": len(base.read_text().splitlines())},
        "aggregate_phase_table": {"name": table.name, "sha256": sha(table), "bytes": table.stat().st_size, "rows": len(rows)},
        "input_terminal_sha256": {"structural_h_on": sha(H), "native": sha(N)},
        "base_manifest_sha256": sha(OUT / "p1r52-structuralh-on-report-manifest.json"),
        "base_package_root": load(OUT / "analysis-receipt.json")["package_root_sha256"],
    }
    manifest = dump_new("p1r52-structuralh-on-report-manifest-v2.json", v2manifest)
    root_items = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in {"analysis-receipt-v2.json", "independent-rehash-review.json"}:
            root_items.append(f"{p.name}\t{sha(p)}\t{p.stat().st_size}")
    receipt = {
        "schema": "p1r52-structuralh-on/analysis-receipt-v2/v1",
        "status": "PASS",
        "canonical_report": str(v2),
        "report_sha256": sha(v2), "report_bytes": v2.stat().st_size, "report_lines": len(v2.read_text().splitlines()),
        "manifest_sha256": sha(manifest),
        "package_root_sha256": hashlib.sha256("\n".join(root_items).encode()).hexdigest(),
        "aggregate_phase_table_rows": len(rows),
        "analysis_actions": {"model": 0, "evaluator": 0, "gpu": 0, "slurm": 0, "source_edit": 0, "result_mutation": 0},
    }
    dump_new("analysis-receipt-v2.json", receipt)


if __name__ == '__main__':
    main()
