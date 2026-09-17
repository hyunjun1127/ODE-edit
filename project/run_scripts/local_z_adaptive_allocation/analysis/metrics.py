"""Independent saved-NLL reducer, canonical identity guards, request clusters."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .bootstrap import ROOT,REVIEW,identity,save

ARMS=('N4','REFIT4','L75','T75','L4D','LD','TD')
MULT={'RS':1,'PS':2,'NS':10}

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()

def table(path,rows,fields=None):
    rows=list(rows);fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def read(path):return json.loads(Path(path).read_text())

def validate(doc,records):
    assert doc['requests']==len(records)
    assert doc['request_order']==digest([r['case_id'] for r in records])
    for tag,mult in MULT.items():
        metric=doc['metrics'][tag];rows=metric['rows'];expected=[]
        for record in records:
            rw=record['requested_rewrite'];prompts=([rw['prompt'].format(rw['subject'])] if tag=='RS' else record['paraphrase_prompts' if tag=='PS' else 'neighborhood_prompts'])
            assert len(prompts)==mult
            expected.extend((record['case_id'],i,digest([record['case_id'],i,p,rw['target_new']['str'],rw['target_true']['str']])) for i,p in enumerate(prompts))
        assert [(r['case_id'],r['prompt_index'],r['identity']) for r in rows]==expected
        assert len(set(expected))==len(expected)==len(records)*mult
        success=[]
        for r in rows:
            assert math.isfinite(r['new_nll']) and math.isfinite(r['true_nll'])
            margin=r['new_nll']-r['true_nll'] if tag=='NS' else r['true_nll']-r['new_nll']
            success.append(margin>0);assert bool(r['success'])==(margin>0)
            assert r.get('desired_margin',margin)==margin
            for side in ('new','true'):
                assert 0<=r[side+'_token_correct']<=r[side+'_token_count']
                assert bool(r[side+'_strict'])==(r[side+'_token_correct']==r[side+'_token_count'])
        assert sum(success)==metric['numerator'] and metric['denominator']==len(expected)
        assert metric['rate']==sum(success)/len(expected)
        assert metric['bit_order_sha256']==digest([(r['identity'],r['success']) for r in rows])
    return doc

def select(doc,ids):
    ids=set(ids)
    return {tag:[r for r in doc['metrics'][tag]['rows'] if r['case_id'] in ids] for tag in MULT}

def stats(values):
    a=np.array(values,dtype=float)
    if not len(a):return dict(mean=None,median=None,p90=None,p95=None,p99=None,min=None,max=None)
    return dict(mean=float(a.mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),
        p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)),min=float(a.min()),max=float(a.max()))

def summary(rows,tag):
    n=len(rows);success=sum((r['new_nll']<r['true_nll']) if tag!='NS' else (r['true_nll']<r['new_nll']) for r in rows)
    desired='true' if tag=='NS' else 'new'
    out=dict(metric=tag,numerator=success,denominator=n,percent=100*success/n if n else None,
        ties=sum(r['new_nll']==r['true_nll'] for r in rows),desired_TF_strict=sum(r[desired+'_strict'] for r in rows),
        desired_token_correct=sum(r[desired+'_token_correct'] for r in rows),desired_token_count=sum(r[desired+'_token_count'] for r in rows))
    for field in ('new_nll','true_nll','desired_margin'):
        out.update({field+'_'+k:v for k,v in stats([r[field] for r in rows]).items()})
    if tag=='PS':
        by={}
        for r in rows:by.setdefault(r['case_id'],[]).append(bool(r['new_strict']))
        assert all(len(v)==2 for v in by.values())
        out['two_P_strict']=sum(all(v) for v in by.values());out['two_P_denominator']=len(by)
    return out

def pair(a,b,tag,*,ci=False):
    def key(r):return r['case_id'],r['prompt_index'],r['identity']
    aa={key(r):r for r in a};bb={key(r):r for r in b};assert len(aa)==len(a) and aa.keys()==bb.keys()
    ordered=list(aa);n=len(ordered)
    x=np.array([aa[k]['success'] for k in ordered],bool);y=np.array([bb[k]['success'] for k in ordered],bool)
    out=dict(metric=tag,denominator=n,before=int(x.sum()),after=int(y.sum()),delta_pp=100*float((y.astype(float)-x).mean()),
        lost=int((x&~y).sum()),gained=int((~x&y).sum()),both_success=int((x&y).sum()),both_failure=int((~x&~y).sum()),
        conditional_loss_denominator=int(x.sum()),conditional_recovery_denominator=int((~x).sum()))
    for f in ('new_nll','true_nll','desired_margin'):
        delta=[bb[k][f]-aa[k][f] for k in ordered]
        out.update({f+'_delta_'+s:v for s,v in stats(delta).items()})
    desired='true_nll' if tag=='NS' else 'new_nll'
    harms=np.array([bb[k][desired]-aa[k][desired] for k in ordered])
    out['desired_NLL_harmed']=int((harms>0).sum())
    if ci:
        cases={}
        for i,k in enumerate(ordered):cases.setdefault(k[0],[]).append(int(y[i])-int(x[i]))
        clustered=np.array([np.mean(v) for v in cases.values()]);rng=np.random.default_rng(20260917)
        boot=clustered[rng.integers(0,len(clustered),size=(2000,len(clustered)))].mean(1)*100
        out.update(request_clusters=len(clustered),CI95_low=float(np.quantile(boot,.025)),CI95_high=float(np.quantile(boot,.975)),
                   CI='DESCRIPTIVE_REQUEST_CLUSTER_2000_FIXED_ORDER_NOT_BATCH_REPLICATES')
    return out

def first():
    from scripts.fixed_counterfact import load_prefix
    records=load_prefix('/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1',1000)
    out=REVIEW/'first-tables';out.mkdir(exist_ok=False)
    rows=[];inputs=[];baseline=None
    for arm in ARMS:
        root=ROOT/'arms'/arm/'attempt-v1/output';terminal=read(root/'terminal.json')
        assert terminal['status']=='COMPLETED_1000_REQUESTS' and terminal['requests']==1000 and terminal['batches']==10
        doc=validate(read(root/'B010/seen-full.json'),records)
        s={tag:summary(doc['metrics'][tag]['rows'],tag) for tag in MULT}
        if arm=='N4':baseline=s
        row=dict(arm=arm,status='W10_METRIC_IDENTITY_NLL_REDUCED; STATE_AUDIT_PENDING')
        for tag in ('RS','PS','NS'):
            row.update({tag+'_count':s[tag]['numerator'],tag+'_denominator':s[tag]['denominator'],
                tag+'_percent':s[tag]['percent'],tag+'_vs_N4_pp':s[tag]['percent']-baseline[tag]['percent']})
        rows.append(row);inputs.extend([identity(root/'terminal.json'),identity(root/'B010/seen-full.json')])
    table(out/'first-final-table.csv',rows);save(out/'metric-receipt.json',dict(inputs=inputs,model_calls=0,
        status='INDEPENDENT_NLL_REDUCER_IDENTITY_DENOMINATORS_PASS_NOT_FULL_STATE_VALIDATION',rows=rows))
    print(json.dumps(rows,ensure_ascii=False,indent=2))

if __name__=='__main__':first()
