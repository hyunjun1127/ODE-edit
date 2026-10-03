"""CPU-only extraction of existing published W5 baselines, never a new fit."""
import argparse
import csv
import json
from pathlib import Path
from .common import member,write,require


def main():
    p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    old=a.repository/'experiment-reports/servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1'
    own=a.repository/'experiment-reports/servers/server3/memit-history-fixed10k-20260928-v1/completion-review-r1'
    def read(path):return list(csv.DictReader(path.open()))
    chosen={'MEMIT_ORIGINAL':'MEMIT-BLUE (L4+L8)','BASE_ALPHAEDIT':'AlphaEdit','AlphaEdit_ORIGINAL':'AlphaEdit-BLUE (L4+L8)','BASE_MEMIT':'MEMIT'}
    rows=[];members=[member(old/'cumulative-metrics.csv'),member(old/'compatibility.csv'),
        member(old/'configured-policy-comparison.csv'),member(old/'source-config-compatibility.csv'),
        member(own/'all-seen-metrics.csv'),member(own/'four-method-comparison-manifest.json')]
    for r in read(old/'cumulative-metrics.csv'):
        if r['batch']!='5' or r['arm'] not in chosen:continue
        label='true' if r['metric']=='NS' else 'new'
        rows.append(dict(method=chosen[r['arm']],source_arm=r['arm'],endpoint='W5_FIRST500',metric=r['metric'],
            numerator=int(r['numerator']),denominator=int(r['denominator']),rate=float(r['rate']),
            desired=label,TF_correct=int(r[label+'_token_correct']),TF_tokens=int(r[label+'_token_den']),
            TF_strict_numerator=int(r[label+'_strict_num']),TF_strict_denominator=int(r[label+'_strict_den']),
            TF_prompt_macro='NOT_RECORDED_IN_SELECTED_AGGREGATE',new_nll=float(r['new_nll_prompt_mean']),
            true_nll=float(r['true_nll_prompt_mean']),classification='HISTORICAL_REFERENCE'))
    for r in read(own/'all-seen-metrics.csv'):
        if r['batch']!='5':continue
        rows.append(dict(method='MEMIT-H',source_arm='MEMIT_HISTORY_NATIVE',endpoint='W5_FIRST500',metric=r['metric'],
            numerator=int(r['numerator']),denominator=int(r['denominator']),rate=float(r['rate']),
            desired='true' if r['metric']=='NS' else 'new',TF_correct=int(r['tf_token_correct']),TF_tokens=int(r['tf_token_count']),
            TF_strict_numerator=int(r['tf_strict_numerator']),TF_strict_denominator=int(r['denominator']),
            TF_prompt_macro=float(r['tf_prompt_macro']),new_nll=float(r['new_nll']),true_nll=float(r['true_nll']),
            classification='HISTORICAL_REFERENCE'))
    require(len(rows)==15,'BASELINE_ROWS')
    for r in rows:
        require(r['denominator']==dict(RS=500,PS=1000,NS=5000)[r['metric']],'W5_BASELINE_DENOMINATOR')
        require(abs(r['rate']-r['numerator']/r['denominator'])<1e-12,'BASELINE_AGGREGATE')
        r['TF_token_micro']=r['TF_correct']/r['TF_tokens']
        r['TF_strict']=r['TF_strict_numerator']/r['TF_strict_denominator']
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'historical-W5.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    compatibility=[r for r in read(old/'compatibility.csv') if r['arm'] in chosen]
    policy=[r for r in read(old/'configured-policy-comparison.csv') if r['arm'] in chosen]
    runtime=[r for r in read(old/'source-config-compatibility.csv') if r['arm'] in chosen]
    H=json.loads((own/'four-method-comparison-manifest.json').read_text())
    write(a.out/'historical-manifest.json',dict(inputs=members,output=member(a.out/'historical-W5.csv'),
        source=member(Path(__file__)),compatibility=compatibility,policy=policy,runtime=runtime,
        MEMIT_H=[r for r in H['methods'] if r['source_arm']=='MEMIT_HISTORY_NATIVE'],
        classification='HISTORICAL_REFERENCE_NOT_NEW_MATCHED_CONTROL',
        limitations=['Different method/layers/hparams/seed; old seed20260907 vs v10 seed20261002.',
        'H200 MEMIT-H vs Blackwell older baselines; old cuDNN TF32 true vs v10 false.',
        'Published same fixed10k first500 aggregate reused; no remote raw pull or new model evaluation.',
        'Token prompt macro absent from some selected tables stays NOT_RECORDED.',
        'Paired baseline-v10 IDs require existing bound per-case raw; not inferred from totals.'],
        fits=0,new_GPU_evaluation=0,scientific_promotion=False))


if __name__=='__main__':main()
