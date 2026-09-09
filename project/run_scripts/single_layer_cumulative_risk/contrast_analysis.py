"""Predeclared method contrasts from measured prompt rows, request bootstrap2000."""
import argparse
import csv
import gzip
import io
import json
from collections import defaultdict
from pathlib import Path
from .analysis import bootstrap,summary,write_csv
from .metadata_analysis import decode
from .import_assets import sha
from .records import save,digest

def contrast(post,base):
    reference={(r['case_id'],r['prompt_index']):r for r in base}
    assert len(reference)==len(base)==len(post)
    rows=[]
    for r in post:
        b=reference[(r['case_id'],r['prompt_index'])]
        assert r['identity']==b['identity']
        rows.append(dict(case_id=r['case_id'],success=r['success'],entry_success=b['success'],
             new_nll_delta=r['new_nll']-b['new_nll'],true_nll_delta=r['true_nll']-b['true_nll'],
             margin_delta=r['margin']-b['margin']))
    result=dict(numerator=sum(r['success'] for r in rows),comparator_numerator=sum(r['entry_success'] for r in rows),
         denominator=len(rows),success_delta=sum(int(r['success'])-int(r['entry_success']) for r in rows)/len(rows),
         loss=sum(r['entry_success'] and not r['success'] for r in rows),
         recovery=sum(not r['entry_success'] and r['success'] for r in rows),
         comparator_success_denominator=sum(r['entry_success'] for r in rows),
         comparator_failure_denominator=sum(not r['entry_success'] for r in rows),**bootstrap(rows))
    for field in ['new_nll_delta','true_nll_delta','margin_delta']:
        result.update({field+'_'+k:v for k,v in summary([r[field] for r in rows]).items()})
    return result

def pairs(stage,entry,completion):
    if stage=='A':
        b=f"B-alpha-{completion['selections']['B']['alpha']}/eval-032"
        c=f"C-alpha-{completion['selections']['C']['alpha']}/eval-032"
        return [(b,'N-full'),(c,'N-full'),(c,b)]
    if stage=='B':
        return [(f'GFminus-amplitude-{amp}/eval',f'{other}-amplitude-{amp}/eval')
                for amp in [.03,.1,.3] for other in ['GFplus','Random1','Random2','LFminus','OPminus','COVminus']]
    comparisons=[(arm,'Continue') for arm in ['FrozenGlobal','RefreshedGlobal','RefreshedLocal','SoftGlobal']]
    comparisons += [('RefreshedGlobal','FrozenGlobal'),('RefreshedGlobal','RefreshedLocal'),('RefreshedGlobal','SoftGlobal')]
    return [(f'{a}/eval-{step:03d}',f'{b}/eval-{step:03d}') for step in [2,4,8] for a,b in comparisons]

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['A','B','C'],required=True)
    p.add_argument('--request-table',type=Path,required=True);p.add_argument('--completion',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    completion=json.loads(a.completion.read_text());assert completion['status']=='COMPLETE'
    groups=defaultdict(list)
    for row in csv.DictReader(io.StringIO(gzip.decompress(a.request_table.read_bytes()).decode())):
        if row.get('reference','') not in ['', 'N']:continue
        row=decode(row);row['prompt_index']=int(row['prompt_index'])
        groups[tuple(row[k] for k in ['entry','endpoint','resolution','panel','metric'])].append(row)
    output=[]
    for entry in sorted({k[0] for k in groups}):
        for post,base in pairs(a.stage,entry,completion):
            matches=[key for key in groups if key[0]==entry and key[1]==post]
            if not matches:raise ValueError(f'MISSING_PREDECLARED_CONTRAST {entry}/{post}')
            for key in sorted(matches):
                other=(entry,base,*key[2:])
                if other not in groups:raise ValueError(f'UNMATCHED_CONTRAST_DENOMINATOR {other}')
                output.append(dict(stage=a.stage,entry=entry,endpoint=post,comparator=base,
                     resolution=key[2],panel=key[3],metric=key[4],**contrast(groups[key],groups[other])))
    a.output.mkdir(parents=True,exist_ok=False)
    member=write_csv(a.output/'paired-method-contrasts.csv',output)
    save(a.output/'contrast-receipt.json',dict(stage=a.stage,input_path=str(a.request_table),input_sha=sha(a.request_table),
         completion_sha=sha(a.completion),output=member,output_root=digest(member),
         paired_request_bootstrap_repetitions=2000,model_action=0,performance_selection=0,
         scope='Method-minus-comparator measured paired differences; no relabeling as inherited NS. Exploratory intervals, not family-wise hypothesis rejection.'))

if __name__=='__main__':main()
