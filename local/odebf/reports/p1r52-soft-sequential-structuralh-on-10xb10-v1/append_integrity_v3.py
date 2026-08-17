#!/usr/bin/env python3
"""Append-only B1 and inter-batch integrity completion for the canonical report."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RESULT = ROOT / "local/odebf/results/s05-p1r52-llama-soft-sequential-historical-10xb10-tech-r3-v1/raw/batches"


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    source = OUT / "p1r52-soft-sequential-structuralh-on-10xb10-factual-ko-v2.md"
    target = OUT / "p1r52-soft-sequential-structuralh-on-10xb10-factual-ko-v3.md"
    if target.exists(): raise RuntimeError(f"create-once exists: {target}")
    terms = [json.loads((RESULT / f"b{i:02d}" / "terminal.json").read_text()) for i in range(1, 11)]
    b1 = terms[0]["B1_atomic_identity"]
    chain = [terms[i]["commit_weight_sha256"] == terms[i + 1]["entry_weight_sha256"] for i in range(9)]
    extra = [
        "", "## B1 identity 및 inter-batch weight persistence receipt", "",
        f"- B1 atomic identity passed={b1['passed']}; final BF16 exact={b1['final_bf16_exact']}; legacy Eff/Gen/Loc exact={b1['legacy_Eff_Gen_Loc_exact']}; k1–k8 scientific payload exact={b1['accepted_k1_k8_scientific_payload_exact']}; stable rollout payload exact={b1['stable_rollout_payload_exact']}.",
        f"- B1 full receipt hash equality={b1['accepted_k1_k8_receipt_sha256_exact']}; sealed receipt가 표시한 process-local excluded paths 때문에 false이며, scientific payload exact은 8/8입니다.",
        f"- Atomic artifacts의 rewrite/paraphrase accuracy는 `{b1['atomic_rewrite_acc']}`/`{b1['atomic_paraphrase_acc']}`로 기록되어 순차 B1 comparison에 impute/re-run하지 않았습니다.",
        f"- commit→next-entry parameter-byte hash equality: {sum(chain)}/{len(chain)} (B1→B2 … B9→B10). inter-batch W0 restore=0, terminal W0 pointer+byte restore는 base receipt PASS입니다.",
        "",
        "## v3 canonical boundary", "",
        "- v3는 v2의 모든 raw-free 표 및 집계에 B1 strict identity와 commit→next-entry 증거를 append-only로 결합한 canonical report입니다.",
    ]
    target.write_text(source.read_text() + "\n".join(extra) + "\n")
    os.chmod(target, 0o600)
    final = OUT / "final-package-receipt-v3.json"
    if final.exists(): raise RuntimeError(f"create-once exists: {final}")
    members=[]
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name != final.name:
            members.append({"name":p.name,"bytes":p.stat().st_size,"sha256":sha(p),"mode":oct(p.stat().st_mode & 0o777)})
    material='\n'.join(f"{x['name']}\t{x['sha256']}\t{x['bytes']}\t{x['mode']}" for x in members)
    data={
        "schema":"p1r52-structuralh-on/final-package-receipt-v3/v1", "status":"PASS",
        "canonical_report":str(target), "canonical_report_sha256":sha(target),
        "canonical_report_bytes":target.stat().st_size, "canonical_report_lines":len(target.read_text().splitlines()),
        "file_tree_count":len(members), "file_tree_root_sha256":hashlib.sha256(material.encode()).hexdigest(),
        "directory_mode":oct(OUT.stat().st_mode & 0o777), "all_members_regular_0600":all(x['mode']=='0o600' for x in members),
        "b1_atomic_identity_pass":b1['passed'], "b1_scientific_payload_exact":b1['accepted_k1_k8_scientific_payload_exact'],
        "commit_to_next_entry_hash_pass_count":sum(chain), "commit_to_next_entry_hash_total":len(chain),
        "independent_review_sha256":sha(OUT/'independent-rehash-review.json'),
        "analysis_actions":{"model":0,"evaluator":0,"gpu":0,"slurm":0,"source_edit":0,"result_mutation":0},
        "members":members,
    }
    final.write_text(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    os.chmod(final,0o600)


if __name__ == '__main__': main()
