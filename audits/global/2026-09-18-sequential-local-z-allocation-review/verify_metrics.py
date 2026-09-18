#!/usr/bin/env python3
"""Read remote raw experiment JSON, return aggregates only; no model/GPU imports.

Run locally: python3 verify_metrics.py
Creates metrics-checks.json beside this script. Remote source and raw are read-only.
"""
import json
from pathlib import Path
import subprocess

REMOTE = r'''
import collections, hashlib, json, math
from pathlib import Path
import numpy as np

ROOT=Path('/data/janghj/ODE-edit/local/sequential-local-z-allocation/20260917-v2')
ARMS=['N4','F48','G48','C4','C48','C45678']
MULT={'RS':1,'PS':2,'NS':10}
load=lambda p:json.loads(Path(p).read_text())
mean=lambda xs:math.fsum(xs)/len(xs) if xs else None
records=load('/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json')[:1000]
digest=lambda v:hashlib.sha256(json.dumps(v,ensure_ascii=True,sort_keys=True,separators=(',',':')).encode()).hexdigest()
bit=lambda r,t:r['true_nll']<r['new_nll'] if t=='NS' else r['new_nll']<r['true_nll']
def validate(doc,recs):
    assert doc['requests']==len(recs)
    assert doc['request_order']==digest([r['case_id'] for r in recs])
    for tag,m in MULT.items():
        expected=[]
        for rec in recs:
            rw=rec['requested_rewrite']
            prompts=[rw['prompt'].format(rw['subject'])] if tag=='RS' else rec['paraphrase_prompts' if tag=='PS' else 'neighborhood_prompts']
            assert len(prompts)==m
            expected += [(rec['case_id'],i,digest([rec['case_id'],i,p,rw['target_new']['str'],rw['target_true']['str']])) for i,p in enumerate(prompts)]
        metric=doc['metrics'][tag]; rows=metric['rows']
        assert [(r['case_id'],r['prompt_index'],r['identity']) for r in rows]==expected
        assert len(set(expected))==len(rows)==m*len(recs)==metric['denominator']
        assert all(math.isfinite(r[k]) for r in rows for k in ['new_nll','true_nll'])
        assert [bit(r,tag) for r in rows]==[r['success'] for r in rows]
        assert sum(bit(r,tag) for r in rows)==metric['numerator']
def pair(after,before,tag):
    assert [r['identity'] for r in after]==[r['identity'] for r in before]
    field='true' if tag=='NS' else 'new'
    ds=[a[field+'_nll']-b[field+'_nll'] for a,b in zip(after,before)]
    return dict(n=len(after),before=sum(bit(r,tag) for r in before),after=sum(bit(r,tag) for r in after),
        lost=sum(bit(b,tag) and not bit(a,tag) for a,b in zip(after,before)),
        gained=sum(bit(a,tag) and not bit(b,tag) for a,b in zip(after,before)),
        strict_lost=sum(b[field+'_strict'] and not a[field+'_strict'] for a,b in zip(after,before)),
        strict_gained=sum(a[field+'_strict'] and not b[field+'_strict'] for a,b in zip(after,before)),
        desired_nll_delta_mean=mean(ds),desired_nll_delta_p99=float(np.quantile(ds,.99)),desired_nll_delta_max=max(ds))
result=dict(method='INDEPENDENT_RAW_JSON_REDUCER_NO_RUNTIME_IMPORT_NO_GPU',root=str(ROOT),arms=ARMS,
    checked_final_rows=0,checked_score_files=0,checked_candidate_guards=0,
    max_fp32_context_mean_difference=0.0,final={},paired={},selected_tails=[],own_n4_to_selected=[],
    retention={},traces={},past_coverage=[],cost={})
final={}
for arm in ARMS:
    out=ROOT/'arms'/arm/'attempt-v1/output'
    full=load(out/'B010/seen-full.json');validate(full,records);final[arm]=full
    row={}
    for tag in MULT:
        rows=full['metrics'][tag]['rows'];field='true' if tag=='NS' else 'new'
        row[tag]=dict(count=sum(bit(r,tag) for r in rows),denominator=len(rows),strict=sum(r[field+'_strict'] for r in rows),
                      ties=sum(r['new_nll']==r['true_nll'] for r in rows))
        result['checked_final_rows']+=len(rows)
    groups=collections.defaultdict(list)
    for r in full['metrics']['PS']['rows']:groups[r['case_id']].append(r)
    row['two_P_strict']=sum(len(g)==2 and all(r['new_strict'] for r in g) for g in groups.values())
    result['final'][arm]=row
    w5=load(out/'B005/seen-full.json');validate(w5,records[:500]);atwrite={t:[] for t in MULT};trace=[]
    for batch in range(1,11):
        b=out/f'B{batch:03d}';s=load(b/'selection.json');ref=next(c for c in s['candidates'] if c['is_n4'])['scores'];selected=s['selected']['scores']
        current=load(b/'selected-current.json');validate(current,records[(batch-1)*100:batch*100])
        for t in MULT:atwrite[t]+=current['metrics'][t]['rows']
        trace.append(dict(batch=batch,S64=selected['base_kl'],own_N4_S64=ref['base_kl'],gates=s['selected']['gates'],
            current_counts={t:current['metrics'][t]['numerator'] for t in MULT},
            Dev128=load(b/'Dev128.json')['D'] if batch in [5,10] else None))
        scoremap={}
        key=lambda x:(x['training_e'],x['past_h'],x['base_kl'],x['canonical_e'])
        for p in sorted((b/'episode/scores').glob('*.json')):
            m=load(p)['metrics'];d=m['details'];tr=d['training']['rows'];assert len(tr)==100
            err=max(abs(mean(r['context_nll'])-r['nll']) for r in tr)
            result['max_fp32_context_mean_difference']=max(result['max_fp32_context_mean_difference'],err)
            assert err<1e-6
            assert abs(mean([r['nll'] for r in tr])-m['E'])<1e-12
            assert abs(mean([r['nll'] for r in d['current']['rows']])-m['controller']['canonical_e'])<1e-12
            assert len(d['generic']['rows'])==64
            assert abs(mean([r['kl'] for r in d['generic']['rows']])-m['B'])<1e-12
            if batch>1:
                assert len(d['past']['rows'])==64
                assert abs(mean([r['nll'] for r in d['past']['rows']])-m['H'])<1e-12
            for label in ['current','past']:
                rows=d[label]['rows'] if d[label] else []
                assert set(r['case_id'] for r in rows if all(r['token_correct']))==set(m['controller'][label+'_strict'] or [])
                assert set(r['case_id'] for r in rows if r['old_nll'] is not None and r['new_nll']<r['old_nll'])==set(m['controller'][label+'_pair'] or [])
            scoremap[key(m['controller'])]=m
            result['checked_score_files']+=1
        for c in s['candidates']:
            q=c['scores'];ok=q['training_e']<=ref['training_e']+1e-4 and (batch==1 or q['past_h']<=ref['past_h']+1e-4)
            ok=ok and all(set(ref[k] or [])<=set(q[k] or []) for k in ['current_strict','current_pair','past_strict','past_pair'])
            assert ok==c['feasible'];result['checked_candidate_guards']+=1
        n=scoremap[key(ref)]['details'];a=scoremap[key(selected)]['details']
        tail=dict(arm=arm,batch=batch,gates=s['selected']['gates'],feasible=s['selected']['feasible'],
            delta_E=selected['training_e']-ref['training_e'],delta_B=selected['base_kl']-ref['base_kl'],
            delta_H=None if batch==1 else selected['past_h']-ref['past_h'])
        for label in ['current','past']:
            ar=a[label]['rows'] if a[label] else [];nr=n[label]['rows'] if n[label] else []
            assert [r['case_id'] for r in ar]==[r['case_id'] for r in nr]
            ds=[x['nll']-y['nll'] for x,y in zip(ar,nr)]
            tail[label]=dict(n=len(ds),worse_gt_1e4=sum(v>1e-4 for v in ds),delta_mean=mean(ds),delta_max=max(ds) if ds else None)
        result['selected_tails'].append(tail)
        if batch in [1,5,10]:
            obs={}
            for p in (b/'candidate-observer').glob('*-binding.json'):
                binding=load(p);doc=load(binding['rows']['path']);validate(doc,records[(batch-1)*100:batch*100]);obs[binding['state_token']]=doc
            own=next(c for c in s['candidates'] if c['is_n4']);a=obs[s['selected']['state_token']];n=obs[own['state_token']]
            result['own_n4_to_selected'].append(dict(arm=arm,batch=batch,**{t:pair(a['metrics'][t]['rows'],n['metrics'][t]['rows'],t) for t in MULT}))
    result['traces'][arm]=trace
    result['retention'][arm]={t:dict(atwrite_to_W10=pair(full['metrics'][t]['rows'],atwrite[t],t),
        W5_to_W10_first500=pair(full['metrics'][t]['rows'][:500*MULT[t]],w5['metrics'][t]['rows'],t)) for t in MULT}
    fits=[load(p) for p in out.glob('B*/episode/fits/*/receipt.json')];terminal=load(out/'terminal.json')
    result['cost'][arm]=dict(**{k:sum(f[k] for f in fits) for k in ['compute_z','adam_updates','loss_evaluations','solve','compute_z_seconds','compute_ks_seconds','solve_seconds']},
        online_scores=sum(len(list(p.glob('*.json'))) for p in out.glob('B*/episode/scores')),
        wall=terminal['seconds'],controller_inclusive=terminal['timers']['components']['controller_inclusive'])
for arm in ARMS[1:]:
    result['paired'][arm+'-N4']={t:pair(final[arm]['metrics'][t]['rows'],final['N4']['metrics'][t]['rows'],t) for t in MULT}
result['cluster_bootstrap']={}
for a,b in [('G48','N4'),('C45678','N4'),('C45678','G48'),('C48','G48')]:
    metrics={}
    for t in MULT:
        by=collections.defaultdict(list);strict=collections.defaultdict(list);sk=('true' if t=='NS' else 'new')+'_strict'
        for x,y in zip(final[a]['metrics'][t]['rows'],final[b]['metrics'][t]['rows']):
            assert x['identity']==y['identity'];by[x['case_id']].append(int(bit(x,t))-int(bit(y,t)));strict[x['case_id']].append(int(x[sk])-int(y[sk]))
        x=np.array([mean(by[c]) for c in sorted(by)]);st=np.array([mean(strict[c]) for c in sorted(strict)])
        ix=np.random.default_rng(20260918).integers(0,len(x),size=(2000,len(x)))
        metrics[t]=dict(delta_pp=float(x.mean()*100),ci95_pp=(np.quantile(x[ix].mean(1),[.025,.975])*100).tolist(),
            strict_delta_pp=float(st.mean()*100),strict_ci95_pp=(np.quantile(st[ix].mean(1),[.025,.975])*100).tolist())
    result['cluster_bootstrap'][a+'-'+b]=metrics
w0=load(load(ROOT/'execution.lock.json')['W0_observation']['path']);validate(w0,records)
good={r['identity'] for r in w0['metrics']['NS']['rows'] if bit(r,'NS')}
result['W0_success_conditioned_N']=dict(denominator=len(good),counts={a:sum(r['identity'] in good and bit(r,'NS') for r in final[a]['metrics']['NS']['rows']) for a in ARMS})
seen=collections.Counter();fact=lambda r:(r['requested_rewrite']['subject'],r['requested_rewrite']['relation_id'])
for batch in range(2,11):
    p=load(ROOT/'arms/N4/attempt-v1/output'/f'B{batch:03d}'/'past64.json')
    for a in ARMS[1:]:assert p['case_ids']==load(ROOT/'arms'/a/'attempt-v1/output'/f'B{batch:03d}'/'past64.json')['case_ids']
    seen.update(p['case_ids']);prior={fact(r):r for r in records[:(batch-1)*100]};current={fact(r) for r in records[(batch-1)*100:batch*100]}
    eligible=sum(k not in current for k in prior)
    result['past_coverage'].append(dict(batch=batch,eligible=eligible,sample=64,fraction=64/eligible,cumulative_unique=len(seen),cohort_counts=dict(collections.Counter(o//100+1 for o in p['ordinals']))))
result['past_unique']=len(seen);result['past_repetition_distribution']=dict(collections.Counter(seen.values()))
assert result['checked_final_rows']==78000 and result['checked_score_files']==446 and result['checked_candidate_guards']==446
result['status']='PASS'
print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
'''

if __name__ == '__main__':
    response = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', 'codex-server4',
         'CUDA_VISIBLE_DEVICES= /data/janghj/EasyEdit/.venv/bin/python -'],
        input=REMOTE, text=True, capture_output=True, check=True,
    )
    result = json.loads(response.stdout)
    target = Path(__file__).with_name('metrics-checks.json')
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['status','checked_final_rows','checked_score_files','checked_candidate_guards','past_unique']}))
