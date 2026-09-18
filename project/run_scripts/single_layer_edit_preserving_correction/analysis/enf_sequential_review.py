"""CPU-only EN-F 50071_1 review. Never imports a model/runtime/evaluator.

Raw scientific inputs are read-only; outputs contain aggregate tables and hashes.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/data/janghj/ODE-edit/local/single-layer-edit-preserving-correction/20260918-v1/S/attempt-v1')
ARM = ROOT / 'arms/EN-F/attempt-v1'
REPORT = Path('experiment-reports/servers/server4/single-layer-edit-preserving-correction-2026-09-18-v1/sequential-four-r1/en-f-completed-review-r1')

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def member(p):
    p = Path(p)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def csvout(p, rows):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        w.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()} for r in rows)

def pairs(obs):
    """Independent canonical comparison; stored aggregates never determine success."""
    out = []
    for metric, prefix, n in [('RS','rewrite',1),('PS','rephrase',2),('NS','locality',10)]:
        new = obs['raw'][prefix+'_target_new']; old = obs['raw'][prefix+'_target_true']
        assert len(new) == len(old) == obs['requests']*n, ('cardinality',metric)
        seen = set()
        for a,b in zip(new,old):
            key = (int(a['case_id']),int(a['prompt_index']))
            assert key == (int(b['case_id']),int(b['prompt_index']))
            assert a['prompt'] == b['prompt'] and key not in seen
            seen.add(key)
            x,y = float(a['nll']),float(b['nll'])
            assert math.isfinite(x) and math.isfinite(y)
            identity = digest([metric,key,a['prompt'],a['target'],b['target'],a['target_token_ids'],b['target_token_ids']])
            historical_identity=hashlib.sha256(json.dumps([key[0],key[1],a['prompt'],a['target'],b['target']],sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
            out.append(dict(metric=metric,case_id=key[0],prompt_index=key[1],identity=identity,historical_identity=historical_identity,
                new_nll=x,true_nll=y,margin=(x-y if metric=='NS' else y-x),
                success=(y<x if metric=='NS' else x<y),tie=(x==y),
                new_strict=all(a['token_correct']),true_strict=all(b['token_correct']),
                new_correct=sum(a['token_correct']),new_tokens=len(a['token_correct']),
                true_correct=sum(b['token_correct']),true_tokens=len(b['token_correct'])))
        assert sum(r['success'] for r in out if r['metric']==metric)==obs['metrics'][metric]['numerator']
    return out

def summary(rows, scope):
    out=[]
    for m in ('RS','PS','NS'):
        a=[r for r in rows if r['metric']==m]; num=sum(r['success'] for r in a)
        out.append(dict(scope=scope,metric=m,numerator=num,denominator=len(a),percent=100*num/len(a),ties=sum(r['tie'] for r in a)))
    return out

def strict(rows):
    cases=sorted({r['case_id'] for r in rows}); d={}
    for case in cases:
        a=[r for r in rows if r['case_id']==case]
        R=[r for r in a if r['metric']=='RS']; P=[r for r in a if r['metric']=='PS']
        assert len(R)==1 and len(P)==2
        d[case]=dict(rewrite_strict=R[0]['new_strict'],two_P_strict=all(r['new_strict'] for r in P),
            R_two_P_strict=all(r['new_strict'] for r in R+P),R_two_P_NLL_joint=all(r['success'] for r in R+P))
    return {k:sum(v[k] for v in d.values()) for k in next(iter(d.values()))}|dict(denominator=len(cases))

def first(repo, scratch):
    repo=Path(repo); out=repo/REPORT; scratch=Path(scratch)
    lock=read(ROOT/'execution.lock.json')
    assert lock['S_arms'][1]=='EN-F' and Path(lock['S_root'])/'EN-F/attempt-v1'==ARM
    assert sha(ROOT/'execution.lock.json')=='4d600d4b57df58203fb21f447116c0362b8a731b9b6d71c4485353d913224d56'
    term=read(ARM/'TERMINAL.json'); assert term['status']=='COMPLETED' and term['commits']==10 and term['requests']==1000
    prior=read(ROOT.parent.parent/'receipts/full-read-m0.json'); bindings=[]
    for v in prior['files']:
        rel=v['path'].split('/worktree/',1)[1]; p=repo/rel
        assert sha(p)==v['sha256'], ('FULL_READ_REUSE_DRIFT',rel)
        bindings.append(dict(relative=rel,sha256=v['sha256'],mode='EXACT_PRIOR_FULL_READ_REUSE'))
    write(scratch/'full-read-reuse.json',dict(prior=member(ROOT.parent.parent/'receipts/full-read-m0.json'),bindings=bindings,
        new_envelope=member(repo/'messages/head/2026-09-18-sh4-enfc-sequential-enf-completed-review.md')))
    p=ARM/'B010/observers/selected-fullseen.json'; obs=read(p); rows=pairs(obs)
    assert [r['case_id'] for r in rows if r['metric']=='RS']==lock['sample_order']
    assert len(set(lock['sample_order']))==1000
    seal=read(ARM/'B010/SELECTION_SEALED.json')
    assert obs['selection_seal']['endpoint_weight_sha256']==seal['next_selected_weight_sha']
    st=strict(rows)
    for k,v in st.items(): assert obs['strict'][k]==v,(k,v,obs['strict'][k])
    final=summary(rows,'W10_full1000'); csvout(out/'first-final-table.csv', final)
    fallback=sum(read(ARM/f'B{b:03d}/selection-ledger.json')['native_fallback'] for b in range(1,11))
    s64=read(ARM/'B010/S64-selected.json'); dev=read(ARM/'B010/observers/Dev128.json')
    aux=dict(strict=st,W0_correct_NS={k:v for k,v in obs['W0_correct_NS'].items() if k!='rows'},
        S64=s64['loss'],Dev128=dev['loss'],fallback_batches=fallback,nonzero_batches=10-fallback,
        allocated_GPU_seconds=17814,allocated_GPU_hours=17814/3600,
        generation=dict(requests=len(obs['generation']),prefix_match=sum(r['target_prefix_match'] for r in obs['generation']),
            reached_max=sum(r['reached_max_new_tokens'] for r in obs['generation']),censored=sum(r['target_over_32_censored'] for r in obs['generation'])))
    write(out/'first-table-receipt.json',dict(status='RAW_NLL_REDUCED_NOT_FULL_STATE_AUDIT',input=member(p),table=member(out/'first-final-table.csv'),
        terminal=member(ARM/'TERMINAL.json'),first=final,auxiliary=aux,
        T='SKIPPED_USER_DIRECTED',full_numerical_validation='NOT_ESTABLISHED',new_GPU=0))
    print(json.dumps(dict(table=final,auxiliary=aux,table_sha=sha(out/'first-final-table.csv')),ensure_ascii=False))

def quantiles(x, prefix=''):
    import numpy as np
    a=np.asarray(x,dtype=float)
    if not len(a): return {}
    return {prefix+k:float(v) for k,v in zip(['mean','min','p50','p95','p99','max'],
        [a.mean(),a.min(),*np.quantile(a,[.5,.95,.99]),a.max()])}

def paired(left, right, label, local_rows):
    a={r['identity']:r for r in left}; b={r['identity']:r for r in right}
    assert len(a)==len(left) and len(b)==len(right) and a.keys()==b.keys(),label
    out=[]
    for m in ('RS','PS','NS'):
        aa=[(r,b[k]) for k,r in a.items() if r['metric']==m]
        lost=sum(x['success'] and not y['success'] for x,y in aa)
        gain=sum(not x['success'] and y['success'] for x,y in aa)
        out.append(dict(comparison=label,metric=m,denominator=len(aa),before=sum(x['success'] for x,y in aa),
            after=sum(y['success'] for x,y in aa),lost=lost,gained=gain,net_pp=100*(gain-lost)/len(aa),
            **quantiles([y['new_nll']-x['new_nll'] for x,y in aa],'new_nll_delta_'),
            **quantiles([y['true_nll']-x['true_nll'] for x,y in aa],'true_nll_delta_'),
            **quantiles([y['margin']-x['margin'] for x,y in aa],'desired_margin_delta_')))
        for x,y in aa:
            local_rows.append(dict(comparison=label,metric=m,case_id=x['case_id'],prompt_index=x['prompt_index'],identity=x['identity'],
                before=x['success'],after=y['success'],lost=x['success'] and not y['success'],gained=not x['success'] and y['success'],
                new_nll_delta=y['new_nll']-x['new_nll'],true_nll_delta=y['true_nll']-x['true_nll'],margin_delta=y['margin']-x['margin']))
    return out

def quality(candidate, anchor):
    assert candidate.keys()==anchor.keys()
    reasons=[]; diffs=[]
    for sid,a in anchor.items():
        c=candidate[sid]
        assert all(math.isfinite(float(v['nll'])) for v in (a,c))
        if a['branch']=='new':
            diffs.append(c['nll']-a['nll'])
            if c['nll']>a['nll']+1e-4: reasons.append([sid,'PER_SEQUENCE_NLL'])
            if a['strict'] and not c['strict']: reasons.append([sid,'STRICT_ID_LOST'])
            old=sid.removesuffix('new')+'old'
            if a['kind']=='canonical' and old in anchor and a['nll']<anchor[old]['nll'] and not c['nll']<candidate[old]['nll']:
                reasons.append([sid,'PAIR_ID_LOST'])
    return reasons,diffs

def analyze(repo,scratch):
    from collections import Counter
    repo=Path(repo); scratch=Path(scratch); out=repo/REPORT
    lock=read(ROOT/'execution.lock.json'); order=lock['sample_order']; prefix=set(order[:500]); ordinal={c:i for i,c in enumerate(order)}
    finals=[]; batches=[]; pairsumm=[]; localrows=[]; stricts=[]; tails=[]; generic=[]; trials=[]; guards=[]; invariants=[]; mechanism=[]; compute=[]; work=[]
    allselected=[]; allnative=[]; allentry=[]; selected={}; observations={}; completeness=[]
    for n in range(1,11):
        b=ARM/f'B{n:03d}'; entry=read(b/'ENTRY.json'); commit=read(b/'COMMIT.json'); seal=read(b/'SELECTION_SEALED.json'); done=read(b/'BATCH_COMPLETE.json')
        assert entry['case_ids']==order[(n-1)*100:n*100] and commit['history_append']==1 and commit['inner_history_append']==0
        assert commit['history'][0]['weight_sha256']==seal['next_selected_weight_sha']==commit['state']['W']
        assert commit['history'][0]['before_sha256']==entry['identity']['M']
        assert commit['history'][0]['after_sha256']==commit['state']['M']
        if n>1: assert entry['identity']==read(ARM/f'B{n-1:03d}/COMMIT.json')['state']
        completeness.append(dict(batch=n,commit=True,history_append=1,inner=0,previous_link=n>1,
            W=commit['state']['W'],M=commit['state']['M'],checkpoint_bytes=commit['checkpoint']['bytes'],checkpoint_path=commit['checkpoint']['path']))
        for role in ('entry','own-native','selected'):
            ob=read(b/f'observers/{role}-current.json'); rr=pairs(ob)
            assert [r['case_id'] for r in rr if r['metric']=='RS']==entry['case_ids']
            assert ob['selection_seal']['endpoint_weight_sha256']==(entry['identity']['W'] if role=='entry' else seal['next_selected_weight_sha'] if role=='selected' else read(b/'native/native-binding.json')['endpoint'])
            assert ob['selection_seal_verified_before_P_N_access'] and ob['RNG_unchanged_and_restored'] and ob['entry_selected_weight_restored_exact']
            observations[n,role]=rr
            batches.extend(dict(batch=n,role=role,**v) for v in summary(rr,'Current100'))
            stricts.append(dict(batch=n,role=role,**strict(rr),generation_prefix_match=sum(g['target_prefix_match'] for g in ob['generation']),
                generation_censored=ob['generation_censored'],generation_max32=sum(g['reached_max_new_tokens'] for g in ob['generation'])))
            work.extend(dict(batch=n,component='official_'+role,unit=k,value=v) for k,v in ob['work'].items())
        allentry.extend(observations[n,'entry']);allnative.extend(observations[n,'own-native']);allselected.extend(observations[n,'selected'])
        pairsumm.extend(paired(observations[n,'entry'],observations[n,'own-native'],f'B{n:02d}_ENTRY_to_NATIVE',localrows))
        pairsumm.extend(paired(observations[n,'own-native'],observations[n,'selected'],f'B{n:02d}_NATIVE_to_SELECTED',localrows))
        s=read(b/'selection-ledger.json'); selected[n]=s; space=read(b/'geometry/space.json'); rank=space['reduced_rank']
        assert s['counters']['accepted_rounds']==sum(t['accepted'] for t in s['trials'])
        assert s['counters']['attempted_trial_slots']==len(s['trials'])<=8
        ev=[read(p)['record'] for p in sorted((b/'events').glob('*.json'))]
        direction=next(e for e in ev if e['event']=='direction'); gradient=next(e for e in ev if e['event']=='gradient_observed')
        anchor=read(b/'native-quality.json')
        for e in ev:
            if e['event']=='quality_guard_checked':
                d=e['details']; cr,cd=quality(d['current_rows'],anchor['current']);pr,pd=quality(d['past_rows'],anchor['past'])
                assert sorted(cr)==sorted(d['current_reasons']) and sorted(pr)==sorted(d['past_reasons'])
                assert e['passed']==(not cr and not pr)
                guards.append(dict(batch=n,trial=e['trial'],passed=e['passed'],current_failed=len(cr),past_failed=len(pr),
                    current_reasons=dict(Counter(v[1] for v in cr)),past_reasons=dict(Counter(v[1] for v in pr)),
                    current_max_delta=max(cd,default=0),past_max_delta=max(pd,default=0),current_sequences=len(cd),past_sequences=len(pd)))
            if e['event']=='actual_invariant_checked':
                d=e['details']; good=(d['ideal_response_relative']<=1e-10 and d['actual_max_token_normalized_response']<=1e-5 and
                    d['actual_projection_leakage_relative']<=1e-5 and d['max_NLL_difference']<=1e-4 and d['logit_max']<=1e-3 and
                    d['logit_rms']<=1e-4 and not d['pair_symmetric_difference'] and not d['strict_symmetric_difference'])
                assert good==e['passed']
                invariants.append(dict(batch=n,trial=e['trial'],passed=e['passed'],**d))
        for t in s['trials']:
            assert math.isclose(t['eta0'],gradient['loss']/direction['chi'],rel_tol=1e-12)
            assert t['eta']==t['eta0']*.5**t['trial']
            if 'armijo_bound' in t: assert math.isclose(t['armijo_bound'],t['current_loss']+1e-4*t['p_actual'],rel_tol=1e-13)
            if t['accepted']:
                assert t['loss']<=t['armijo_bound'] and t['decrease']>1e-6
                assert any(e['event']=='quality_guard_checked' and e['trial']==t['trial'] and e['passed'] for e in ev)
                assert any(e['event']=='actual_invariant_checked' and e['trial']==t['trial'] and e['passed'] for e in ev)
                assert t is s['trials'][-1]
            trials.append(dict(batch=n,**t))
        sv=rank['singular_values']; tau=max(rank['shape'])*2.220446049250313e-16*sv[0]
        assert math.isclose(tau,rank['threshold'],rel_tol=1e-14)
        assert sum(v>tau for v in sv)==space['blocked_dimension']
        assert [i for i,v in enumerate(sv) if tau/10<=v<=tau*10]==rank['ambiguous_indices']
        assert space['dimension']==space['allowed_dimension']-space['blocked_dimension']
        mechanism.append(dict(batch=n,key_columns=space['key_shape'][1],P_star_rank=space['allowed_dimension'],rank_J=space['blocked_dimension'],q=space['dimension'],
            ambiguity=len(rank['ambiguous_indices']),tau=tau,sigma_min=sv[-1],sigma_max=sv[0],G_norm=gradient['gradient_norm'],GQ_norm=direction['chi']**.5,
            chi=direction['chi'],GP_norm='NOT_RECORDED',actual_norm=s['actual_delta_norm'],ideal_norm=s['ideal_delta_norm'],fallback=s['native_fallback'],
            trials=len(s['trials']),accepted_trial=next((t['trial'] for t in s['trials'] if t['accepted']),None),stop=s['stop_reason']))
        for label in ('native','selected'):
            z=read(b/f'S64-{label}.json');assert len(z['rows'])==64 and math.isclose(sum(r['loss'] for r in z['rows'])/64,z['loss'],abs_tol=1e-14)
            generic.append(dict(batch=n,panel='S64',role=label,loss=z['loss'],documents=64,controller=True))
        if n in (5,10):
            ob=read(b/'observers/selected-fullseen.json'); rr=pairs(ob);observations[n,'fullseen']=rr
            assert [r['case_id'] for r in rr if r['metric']=='RS']==order[:n*100]
            finals.extend(summary(rr,f'W{n}_full{n*100}'))
            stricts.append(dict(batch=n,role='fullseen',**strict(rr),generation_prefix_match=sum(g['target_prefix_match'] for g in ob['generation']),
                generation_censored=ob['generation_censored'],generation_max32=sum(g['reached_max_new_tokens'] for g in ob['generation'])))
            dev=read(b/'observers/Dev128.json');assert len(dev['rows'])==128 and math.isclose(sum(r['loss'] for r in dev['rows'])/128,dev['loss'],abs_tol=1e-14)
            generic.append(dict(batch=n,panel='Dev128',role='selected',loss=dev['loss'],documents=128,controller=False))
            for panel,obj in [('fullseen',ob),('Dev128',dev)]:
                work.extend(dict(batch=n,component=panel,unit=k,value=v) for k,v in obj['work'].items())
        compute.append(dict(batch=n,**s['counters'],wall_cumulative=done['wall_seconds'],peak_GPU_bytes=done['peak_GPU_bytes'],host_peak_KiB=done['host_peak_KiB']))
        for panel,w in read(b/'oracle-work.json').items():
            work.extend(dict(batch=n,component='oracle_'+panel,unit=k,value=v) for k,v in w.items())
    final=observations[10,'fullseen'];f500=[r for r in final if r['case_id'] in prefix];last=[r for r in final if r['case_id'] not in prefix]
    finals+=summary(f500,'W10_first500')+summary(last,'W10_last500')+summary(allselected,'pooled_atwrite1000_DIFFERENT_W')
    pairsumm+=paired(observations[5,'fullseen'],f500,'W5_to_W10_first500',localrows)
    pairsumm+=paired(allselected,final,'ATWRITE_to_W10_all1000',localrows)
    pairsumm+=paired(allnative,allselected,'pooled_NATIVE_to_SELECTED',localrows)
    pairsumm+=paired(allentry,allnative,'pooled_ENTRY_to_NATIVE',localrows)
    cohorts=[]
    for n in range(1,11):
        rr=[r for r in final if ordinal[r['case_id']]//100==n-1]
        cohorts.extend(dict(cohort=n,age_batches=10-n,**v) for v in summary(rr,'W10_cohort'))
    for label,rows in [('W10_all1000',final),('W5_first500',observations[5,'fullseen']),('pooled_atwrite',allselected)]:
        for m in ('RS','PS','NS'):
            a=[r for r in rows if r['metric']==m]
            tails.append(dict(scope=label,metric=m,**quantiles([r['new_nll'] for r in a],'new_nll_'),
                **quantiles([r['true_nll'] for r in a],'true_nll_'),**quantiles([r['margin'] for r in a],'desired_margin_')))
    w0spec=lock['observer_reuse_binding']['W0_first1000'];assert sha(w0spec['path'])==w0spec['sha256']
    w0=read(w0spec['path']);wr=w0.get('metrics',w0)
    # Prior W0 holds reduced canonical pairs, not retained token logits.
    w0n={(int(r['case_id']),int(r['prompt_index'])):r for r in wr['NS']['rows']}
    finaln=[r for r in final if r['metric']=='NS']
    assert len(w0n)==len(finaln)==10000
    for r in finaln:
        old=w0n[r['case_id'],r['prompt_index']]
        assert old['identity']==r['historical_identity'], 'W0_PROMPT_TARGET_IDENTITY'
        assert math.isfinite(old['new_nll']) and math.isfinite(old['true_nll'])
        assert old['success']==(old['true_nll']<old['new_nll'])
    ret=sum(w0n[r['case_id'],r['prompt_index']]['success'] and r['success'] for r in finaln)
    den=sum(w0n[r['case_id'],r['prompt_index']]['success'] for r in finaln)
    assert (ret,den)==(7857,8820)
    tables={'final-table':finals,'batch-current':batches,'paired':pairsumm,'strict-generation':stricts,'nll-tails':tails,'generic':generic,
            'trials':trials,'guards':guards,'invariants':invariants,'mechanism':mechanism,'compute':compute,'work-counters':work,'cohorts':cohorts,'commit-links':completeness}
    for name,rr in tables.items():csvout(out/(name+'.csv'),rr)
    # Request/prompt IDs and exact NLL transitions remain local, not Git payload.
    csvout(scratch/'paired-local-rows.csv',localrows)
    write(scratch/'metrics-summary.json',dict(tables=tables,trial_reasons=dict(Counter(t['reason'] for t in trials)),
        W0_N=dict(retained=ret,denominator=den,lost=den-ret),fullseen_input=member(ARM/'B010/observers/selected-fullseen.json')))
    print(json.dumps(dict(paired=[x for x in pairsumm if x['comparison'].startswith(('W5_','ATWRITE_','pooled_NATIVE'))],
        trial_reasons=dict(Counter(t['reason'] for t in trials)),tables=len(tables)),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--scratch',type=Path,required=True);p.add_argument('--phase',choices=['first','metrics'],default='first')
    a=p.parse_args();(first if a.phase=='first' else analyze)(a.repo,a.scratch)
