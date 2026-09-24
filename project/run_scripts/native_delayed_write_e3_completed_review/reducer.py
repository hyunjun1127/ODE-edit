"""Independent raw NLL/token reducer. No model, torch, or original reducer import."""
import collections
import csv
import math
import numpy as np
from .common import *

def write_csv(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)

def load_scores(path,panel):
    rows=[r for p in sorted(Path(path).glob('*.json')) for r in read(p)]
    assert [r['row_id'] for r in rows]==[r['row_id'] for r in panel], ('ORDER_CARDINALITY',str(path),len(rows))
    assert len({r['row_id'] for r in rows})==len(rows)
    pairs=collections.defaultdict(dict);general=[];maxdiff=0.
    for r,p in zip(rows,panel,strict=True):
        assert all(r[k]==p[k] for k in ('row_id','pair_id','panel','kind','label','case_id','subject','prompt_cluster'))
        assert r['input_sha']==digest([p['input_ids'],p['positions'],p['target_ids']])
        assert r['target_count']==len(p['target_ids'])==len(r['token_nll'])==len(r['token_predictions'])
        assert all(math.isfinite(float(x)) for x in [r['nll'],*r['token_nll']])
        correct=[x==y for x,y in zip(r['token_predictions'],p['target_ids'],strict=True)]
        assert sum(correct)==r['token_correct'] and all(correct)==r['strict']
        maxdiff=max(maxdiff,abs(float(np.mean(r['token_nll']))-r['nll']))
        if r['label']=='natural':general.append(r)
        else:
            assert r['label'] not in pairs[r['pair_id']]
            pairs[r['pair_id']][r['label']]=(r,p)
    result=[]
    for pair,d in pairs.items():
        assert set(d)=={'true','new'}
        t,p=d['true'];n,_=d['new'];prefer_true=t['kind'] in ('N','BASE');desired=t if prefer_true else n
        g=n['nll']-t['nll']
        result.append(dict(pair_id=pair,case_id=t['case_id'],subject=t['subject'],prompt_cluster=t['prompt_cluster'],
            panel=t['panel'],kind=t['kind'],prompt_index=p['prompt_index'],active_target_hash=p.get('active_target_hash'),
            true_nll=t['nll'],new_nll=n['nll'],g=g,m=-g,desired_margin=g if prefer_true else -g,
            success=g>0 if prefer_true else g<0,tie=g==0,strict=desired['strict'],desired_nll=desired['nll'],
            token_correct=desired['token_correct'],token_count=desired['target_count'],
            prompt_accuracy=desired['token_correct']/desired['target_count']))
    assert len(general)==128
    validation=dict(path=str(path),completion_rows=len(rows),paired_prompts=len(result),general_documents=len(general),
        token_positions=sum(r['target_count'] for r in rows),CPU64_vs_GPU32_mean_max_abs=maxdiff,order_token_strict_recount=True)
    return result,general,validation

def summarize(label,pairs,general):
    result=[]
    for panel,kind in sorted({(r['panel'],r['kind']) for r in pairs}):
        rr=[r for r in pairs if (r['panel'],r['kind'])==(panel,kind)]
        row=dict(endpoint=label,panel=panel,kind=kind,count=len(rr),unique_cases=len({r['case_id'] for r in rr}),
            success=sum(r['success'] for r in rr),strict=sum(r['strict'] for r in rr),ties=sum(r['tie'] for r in rr),
            token_correct=sum(r['token_correct'] for r in rr),token_total=sum(r['token_count'] for r in rr),
            TF_prompt_macro=float(np.mean([r['prompt_accuracy'] for r in rr])))
        row['TF_token_micro']=row['token_correct']/row['token_total']
        for metric in ('true_nll','new_nll','desired_nll','g','desired_margin'):
            x=[r[metric] for r in rr];row[metric+'_mean']=float(np.mean(x))
            for q in (.01,.05,.5,.95,.99):row[f'{metric}_q{int(q*100):02d}']=float(np.quantile(x,q))
        result.append(row)
    bycase=collections.defaultdict(list)
    for r in pairs:
        if r['panel']=='H_diag_B1_R100_P200':bycase[r['case_id']].append(r)
    assert len(bycase)==100 and all(len(v)==3 for v in bycase.values())
    result.append(dict(endpoint=label,panel='H_diag_B1_R100_P200',kind='R+twoP_joint',count=100,
        success=sum(all(r['success'] for r in rr) for rr in bycase.values()),
        strict=sum(all(r['strict'] for r in rr) for rr in bycase.values())))
    result.append(dict(endpoint=label,panel='GeneralEval128',kind='GENERAL',count=len(general),
        unique_cases=len(general),strict=sum(r['strict'] for r in general),
        token_correct=sum(r['token_correct'] for r in general),token_total=sum(r['target_count'] for r in general),
        TF_token_micro=sum(r['token_correct'] for r in general)/sum(r['target_count'] for r in general),
        TF_prompt_macro=float(np.mean([r['token_correct']/r['target_count'] for r in general])),
        natural_nll_mean=float(np.mean([r['nll'] for r in general])),
        W0_forward_KL=float(np.mean([r['w0_forward_kl'] for r in general])),
        W0_top1_matches=sum(r['w0_top1_agree'] for r in general)))
    return result

def ci(values,identities):
    parent=list(range(len(values)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i,r in enumerate(identities):
        for k in [('case',r['case_id']),('subject',r['subject']),('prompt',r['prompt_cluster'])]:
            if k in seen:parent[find(i)]=find(seen[k])
            else:seen[k]=i
    groups=collections.defaultdict(list)
    for i,v in enumerate(values):groups[find(i)].append(v)
    sums=np.array([sum(v) for v in groups.values()]);counts=np.array([len(v) for v in groups.values()])
    rng=np.random.default_rng(20260924);boot=[]
    for _ in range(2000):
        ix=rng.integers(0,len(sums),len(sums));boot.append(float(sums[ix].sum()/counts[ix].sum()))
    return dict(cluster_count=len(sums),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),seed=20260924,bootstrap_repetitions=2000)

def compare(label,left,right,ids_out):
    a={r['pair_id']:r for r in left};b={r['pair_id']:r for r in right};assert a.keys()==b.keys()
    result=[]
    for panel,kind in sorted({(r['panel'],r['kind']) for r in left}):
        ids=[i for i,r in a.items() if (r['panel'],r['kind'])==(panel,kind)];x=[a[i] for i in ids];y=[b[i] for i in ids]
        gained=[i for i in ids if not a[i]['success'] and b[i]['success']];lost=[i for i in ids if a[i]['success'] and not b[i]['success']]
        sg=[i for i in ids if not a[i]['strict'] and b[i]['strict']];sl=[i for i in ids if a[i]['strict'] and not b[i]['strict']]
        row=dict(contrast=label,panel=panel,kind=kind,count=len(ids),before_success=sum(r['success'] for r in x),
            after_success=sum(r['success'] for r in y),gained=len(gained),lost=len(lost),strict_gained=len(sg),strict_lost=len(sl),
            before_strict=sum(r['strict'] for r in x),after_strict=sum(r['strict'] for r in y))
        for metric in ('true_nll','new_nll','desired_nll','g','desired_margin'):
            d=[z[metric]-w[metric] for w,z in zip(x,y,strict=True)];row[metric+'_delta']=float(np.mean(d))
            for q in (.01,.05,.5,.95,.99):row[f'{metric}_delta_q{int(q*100):02d}']=float(np.quantile(d,q))
        row.update(ci([z['desired_nll']-w['desired_nll'] for w,z in zip(x,y,strict=True)],x));result.append(row)
        ids_out.append(dict(contrast=label,panel=panel,kind=kind,success_gained=gained,success_lost=lost,strict_gained=sg,strict_lost=sl,
            worst_NLL10=[r['pair_id'] for r in sorted(y,key=lambda r:r['desired_nll']-a[r['pair_id']]['desired_nll'],reverse=True)[:10]]))
    return result

def first():
    panel=read(ROOT/'inputs/panels-v2/rows.json');prior=None;binding=None;stage=[]
    for s in read(ROOT/'inputs/design/review/e3-dependency-plan.json')['stages']:
        name=s['stage_id'];p=OUTPUT/name/'gate-result.json';g=read(p)
        assert g['status']=='PASS' and g['stage']==name
        binding=binding or g['binding'];assert binding==g['binding'] and g['predecessor_sha256']==prior
        prior=sha(p);stage.append(dict(stage=name,status='COMPLETED',receipt=str(p),sha256=prior,elapsed_seconds=g.get('elapsed_seconds')))
    assert not (OUTPUT/'failure.json').exists()
    assert sha(ROOT/'inputs/panels-v2/panel-manifest.json')==binding['panel_sha256']
    allrows=[];validation=[]
    names=['W0']+[f'{f}_W{b:03d}' for f in ('BASE_ALPHAEDIT','BASE_MEMIT') for b in (1,5,10,20,30,40,50,60,70,80,90,100)]
    for name in names:
        p,g,v=load_scores(OUTPUT/'G20'/name,panel);allrows+=summarize(name,p,g);validation.append(v)
    write_csv(REPORT/'first-endpoint-table.csv',allrows);write_csv(REPORT/'stage-coverage.csv',stage)
    save(AUDIT/'first-table-check.json',dict(first_table=record(REPORT/'first-endpoint-table.csv'),stage_chain_valid=True,
        validation=validation,level='Independent E1 row/token/preference recount; detailed E3 and full inventory not yet audited',GPU_calls=0))
    for name in names:
        r=[x for x in allrows if x['endpoint']==name]
        print(name,[(x['kind'],x.get('success'),x['count']) for x in r if x['panel']!='H_active_variants' and x['kind']!='GENERAL'])

if __name__=='__main__':first()
