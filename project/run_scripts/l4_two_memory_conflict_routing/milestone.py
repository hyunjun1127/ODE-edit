"""Compact completed-primary milestone; smaller-B/extra observations stay pending."""
import argparse
import subprocess
from pathlib import Path
from .analysis import read,summarize,csv_save
from .identity import save,member,digest,sha
from .report import table,num

def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);rows=[];harms=[];refs=[];closed=0
    for run in map(Path,args.runs):
        t=read(run/'terminal.json')
        if t['status']!='TERMINAL_VALID' or t['batch_raw']!=100:raise ValueError('PRIMARY_NOT_TERMINAL')
        closed+=t['new_path_count'];ref=read(run/'N/full.json')['rows'];refs.append(member(run/'terminal.json'))
        for arm in t['arms']:
            rows.extend(dict(entry=t['entry'],arm=arm,**r) for r in summarize(read(run/arm/'full.json')['rows'],ref))
            h=read(run/arm/'harms.json')
            harms.append(dict(entry=t['entry'],arm=arm,**{k:h[k]['value'] for k in ('Past','Base','BaseAudit')}))
            refs.extend(member(run/arm/n) for n in ('terminal.json','full.json','harms.json'))
    if closed!=10:raise ValueError('PRIMARY_COUNT')
    csv_save(out/'primary-endpoints.csv',rows);csv_save(out/'primary-harms.csv',harms)
    text=['# B100 세 entry 주 비교 완료 — v2 중간 보고','',
        '새 B100 경로10개(OS/BF1/BF8×3 + Middle Frozen)와 N3 reference가 완료됐다. '
        'B1/B7 및 추가 Base NS/context 관측은 별도 진행하며 전체16개 완료 보고로 표시하지 않는다. '
        'NLL-pair RS/PS는 new<true, NS는 true<new; tie fail. scientific_promotion=false.','',
        table(['entry','arm','panel','metric','n/d','N 대비 Δpp','new NLL mean','true NLL mean'],
            [[r['entry'],r['arm'],r['panel'],r['metric'],f"{r['numerator']}/{r['denominator']}",num(r['delta_pp']),num(r['new_nll_mean']),num(r['true_nll_mean'])] for r in rows]),
        table(['entry','arm','Past F','Base We KL','Base audit We KL'],
            [[r['entry'],r['arm']]+[num(r[k]) for k in ('Past','Base','BaseAudit')] for r in harms]),
        '유리/불리한 모든 값을 보존한다. Final은 별도 frozen-raw 전체 집계와 관측 coverage를 포함한다.','']
    with (out/'primary-comparison-ko.md').open('x') as f:f.write('\n'.join(text))
    save(out/'receipt.json',dict(status='B100_PRIMARY_COMPLETE',new_paths=10,native_reused=3,full_campaign_complete=False,
        source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),members=refs,members_root=digest(refs),
        report_sha256=sha(out/'primary-comparison-ko.md'),scientific_promotion=False))
    print(str(out/'primary-comparison-ko.md'),sha(out/'primary-comparison-ko.md'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True);main(p.parse_args())
