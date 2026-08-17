#!/usr/bin/env python3
"""Fresh-process review of v3 and final rooted receipt."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

OUT=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text())

def main():
    report=OUT/'p1r52-soft-sequential-structuralh-on-10xb10-factual-ko-v3.md'
    final3=load(OUT/'final-package-receipt-v3.json')
    oldreview=load(OUT/'independent-rehash-review.json')
    review=OUT/'independent-rehash-review-v3.json'
    if review.exists(): raise RuntimeError(f'create-once exists: {review}')
    text=report.read_text()
    pass_report = sha(report)==final3['canonical_report_sha256'] and report.stat().st_size==final3['canonical_report_bytes'] and len(text.splitlines())==final3['canonical_report_lines']
    pass_b1=all(x in text for x in ('B1 atomic identity passed=True','commit→next-entry parameter-byte hash equality: 9/9','B1–B10 절대 pre → post → final-W10 표','10개 true entry-pre·immediate post·final B100 집계'))
    status='PASS' if pass_report and pass_b1 and oldreview['status']=='PASS' else 'FAIL'
    data={'schema':'p1r52-structuralh-on/independent-rehash-review-v3/v1','status':status,'fresh_process':True,'v3_report_rehash_pass':pass_report,'v3_required_integrity_section_pass':pass_b1,'base_independent_review_status':oldreview['status'],'canonical_report':str(report),'canonical_report_sha256':sha(report),'canonical_report_bytes':report.stat().st_size,'canonical_report_lines':len(text.splitlines()),'analysis_actions':{'model':0,'evaluator':0,'gpu':0,'slurm':0,'source_edit':0,'result_mutation':0}}
    review.write_text(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True)+'\n'); os.chmod(review,0o600)
    if status!='PASS': raise SystemExit(1)
    final=OUT/'final-package-receipt-v4.json'
    if final.exists(): raise RuntimeError(f'create-once exists: {final}')
    members=[]
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name!=final.name:
            members.append({'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p),'mode':oct(p.stat().st_mode&0o777)})
    root=hashlib.sha256('\n'.join(f"{x['name']}\t{x['sha256']}\t{x['bytes']}\t{x['mode']}" for x in members).encode()).hexdigest()
    seal={'schema':'p1r52-structuralh-on/final-package-receipt-v4/v1','status':'PASS','canonical_scope':'P1R52-Soft-Sequential-StructuralH-On-10xB10','canonical_report':str(report),'canonical_report_sha256':sha(report),'canonical_report_bytes':report.stat().st_size,'canonical_report_lines':len(text.splitlines()),'file_tree_count':len(members),'file_tree_root_sha256':root,'directory_mode':oct(OUT.stat().st_mode&0o777),'all_members_regular_0600':all(x['mode']=='0o600' for x in members),'independent_review_v3_sha256':sha(review),'independent_review_v3_status':'PASS','analysis_actions':data['analysis_actions'],'members':members}
    final.write_text(json.dumps(seal,ensure_ascii=False,indent=2,sort_keys=True)+'\n'); os.chmod(final,0o600)

if __name__=='__main__': main()
