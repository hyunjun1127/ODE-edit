"""CPU-only independent reduction of sealed EP47962 JSON; no model imports."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

TASK='ODEEDIT-S06-EP-TW1-47962-COMPLETED-DETAILED-REVIEW-SH4-V1'
ROOT=Path('/data/janghj/ODE-edit/local/ep-tw1-c4/20260915-v1/gate-skip-r1')
REPORT='experiment-reports/servers/server4/ep-tw1-c4-2026-09-15-v1/gate-skip-r1/completed-review-v1'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    return h.hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),bytes=Path(p).stat().st_size,sha256=sha(p))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    return ref(p)
def table(p,rows):
    rows=list(rows);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    return ref(p)
def quantile(x,q):
    x=sorted(x)
    if not x:return None
    z=(len(x)-1)*q;i=int(z);return x[i]+(x[min(i+1,len(x)-1)]-x[i])*(z-i)
def dist(x,prefix):
    x=list(x)
    return {prefix+k:v for k,v in dict(mean=statistics.fmean(x) if x else None,
        median=quantile(x,.5),p90=quantile(x,.9),p95=quantile(x,.95),p99=quantile(x,.99),
        min=min(x) if x else None,max=max(x) if x else None).items()}
def key(r):return r['case_id'],r['prompt_index'],r['identity']
def success(r,m):return r['true_nll']<r['new_nll'] if m=='NS' else r['new_nll']<r['true_nll']
def summary(rows,m,**labels):
    assert len({key(r) for r in rows})==len(rows),'DUPLICATE_PROMPT_IDENTITY'
    assert all(math.isfinite(r[k]) for r in rows for k in ('new_nll','true_nll'))
    n=sum(success(r,m) for r in rows);d=len(rows);bycase=defaultdict(list)
    for r in rows:bycase[r['case_id']].append(r)
    desired='true' if m=='NS' else 'new'
    return dict(labels,metric=m,numerator=n,denominator=d,percent=100*n/d if d else None,
        request_denominator=len(bycase),ties=sum(r['new_nll']==r['true_nll'] for r in rows),
        desired_strict_n=sum(r[desired+'_strict'] for r in rows),desired_strict_d=d,
        new_strict_n=sum(r['new_strict'] for r in rows),true_strict_n=sum(r['true_strict'] for r in rows),
        desired_token_correct=sum(r[desired+'_token_correct'] for r in rows),
        desired_token_d=sum(r[desired+'_token_count'] for r in rows),
        all_prompt_strict_requests=sum(all(r[desired+'_strict'] for r in rr) for rr in bycase.values()),
        **dist((r['new_nll'] for r in rows),'new_nll_'),**dist((r['true_nll'] for r in rows),'true_nll_'),
        **dist(((r['new_nll']-r['true_nll']) if m=='NS' else (r['true_nll']-r['new_nll']) for r in rows),'desired_margin_'))
def panel_rows(panel,m):
    v=panel['metrics'][m];rows=v['rows'];s=summary(rows,m)
    assert s['numerator']==v['numerator'] and s['denominator']==v['denominator']
    assert all(success(r,m)==r['success'] for r in rows),'RECORDED_SUCCESS_MISMATCH'
    return rows
def transition(before,after,m,**labels):
    b={key(r):r for r in before};a={key(r):r for r in after};assert b.keys()==a.keys(),'PAIR_IDENTITY_MISMATCH'
    pairs=[(b[k],a[k]) for k in b];states=Counter((success(x,m),success(y,m)) for x,y in pairs)
    strict=Counter((x['true_strict' if m=='NS' else 'new_strict'],y['true_strict' if m=='NS' else 'new_strict']) for x,y in pairs)
    d=len(pairs);bn=sum(success(x,m) for x,y in pairs);an=sum(success(y,m) for x,y in pairs)
    target='true_nll' if m=='NS' else 'new_nll';comp='new_nll' if m=='NS' else 'true_nll'
    return dict(labels,metric=m,denominator=d,before_num=bn,after_num=an,delta_pp=(an-bn)*100/d if d else None,
        retained=states[True,True],lost=states[True,False],gained=states[False,True],both_failed=states[False,False],
        conditional_loss_den=bn,conditional_recovery_den=d-bn,strict_lost=strict[True,False],strict_gained=strict[False,True],
        strict_before=sum(x['true_strict' if m=='NS' else 'new_strict'] for x,y in pairs),
        strict_after=sum(y['true_strict' if m=='NS' else 'new_strict'] for x,y in pairs),
        **dist((y[target]-x[target] for x,y in pairs),'desired_NLL_harm_'),
        **dist((y[comp]-x[comp] for x,y in pairs),'competing_NLL_change_'))

def reduce(worktree,output,*,root=None,expected_lock_sha='5a19c2be919362d08b2ea80e69a406d7b6de569aade8de639036f69db5d5d8f9',allocated_gpu_seconds=7694,instruction_id=TASK):
    w,out=Path(worktree),Path(output);out.mkdir(parents=True,exist_ok=False)
    source_root=Path(root) if root is not None else ROOT
    raw=source_root/'scientific-v1';lock=read(source_root/'execution.lock.json')
    assert sha(source_root/'execution.lock.json')==expected_lock_sha
    terminal=read(raw/'terminal.json');assert len(terminal['commits'])==10 and terminal['completed_requests']==1000
    assert terminal['numerical_validation']=='NOT_ESTABLISHED'
    metrics=[];whole=[];pairs=[];candidates=[];policyrows=[];warnings=[];ledgerrows=[];cost=[];general=[];atwrite={m:[] for m in ('RS','PS','NS')}
    final=read(raw/'B010/selected-evaluation.json');finalrows={m:panel_rows(final['seen_full'],m) for m in atwrite}
    first=[summary(finalrows[m],m,scope='W10_FULL_FIRST1000',policy='EP-TW-1') for m in atwrite]
    assert [r['denominator'] for r in first]==[1000,2000,10000]
    table(out/'first-final-table.csv',first)
    for b in range(1,11):
        p=raw/f'B{b:03d}';entry=read(p/'entry-evaluation.json');rawpanel=read(p/'raw-evaluation.json');sel=read(p/'selected-evaluation.json')
        c=read(p/'commit.json');pol=read(p/'policy.json');ce=read(p/'candidate-evaluation.json')
        assert c['next_ordinal']==b*100 and c['batch']==b and c['numerical_validation']=='NOT_ESTABLISHED'
        for state,pan in [('ENTRY',entry),('RAW',rawpanel),('SELECTED',sel['selected'])]:
            assert pan['endpoint_state']==(c['entry'] if state=='ENTRY' else c['endpoint']) if state!='RAW' else True
            for population in ('current','accepted_old'):
                if not pan.get(population):continue
                for m in atwrite:
                    rr=panel_rows(pan[population],m)
                    metrics.append(summary(rr,m,batch=b,state=state,population=population.upper()))
                    if state=='SELECTED' and population=='current':atwrite[m]+=rr
        for m in atwrite:
            for name,left,right in [('ENTRY_TO_RAW',entry['current'],rawpanel['current']),('RAW_TO_SELECTED',rawpanel['current'],sel['selected']['current'])]:
                pairs.append(transition(panel_rows(left,m),panel_rows(right,m),m,comparison=name,batch=b,population='CURRENT'))
            if entry.get('accepted_old'):
                pairs.append(transition(panel_rows(rawpanel['accepted_old'],m),panel_rows(sel['selected']['accepted_old'],m),m,
                    comparison='RAW_TO_SELECTED',batch=b,population='ACCEPTED_OLD'))
        for st,pan,obs in [('RAW',rawpanel,ce['RAW']),('SELECTED',sel['selected'],ce[c['selected']])]:
            rr=pan['current']['metrics']['RS']['rows'];mr=obs['current']['rows'];assert [x['case_id'] for x in rr]==[x['case_id'] for x in mr]
            diffs=[abs(x['new_nll']-y['nll']) for x,y in zip(rr,mr)]
            cs={x['case_id'] for x in rr if x['new_strict']};ms=set(obs['current']['strict_ids'])
            warnings.append(dict(batch=b,state=st,recorded_status=pan['method_observer_parity']['status'],
                unequal_NLL_rows=sum(x!=0 for x in diffs),max_abs_NLL_difference=max(diffs),strict_lost=len(ms-cs),strict_gained=len(cs-ms),
                additional_forwards=0,warning_not_numerical_PASS=True))
        rawobs=ce['RAW'];rp=rawobs['current']['E'];rd=rawobs['generic']['D'];rids=set(rawobs['current']['strict_ids'])
        feasible=[]
        for r in pol['selection']['candidate_receipts']:
            cid=r['id'];src=r.get('evaluation_source_id',r.get('duplicate_of') or cid);obs=ce.get(src)
            if obs and 'current' in obs:
                assert len(obs['current']['rows'])==100 and obs['current']['denominator']==100
                assert math.isclose(statistics.fmean(x['nll'] for x in obs['current']['rows']),obs['current']['E'],rel_tol=1e-12)
                assert len(obs['generic']['rows'])==64
                ids=set(obs['current']['strict_ids']);e=obs['current']['E'];d=obs['generic']['D']
                ok=r['trust_valid'] and math.isfinite(e) and math.isfinite(d) and e<=rp and rids<=ids
                # Runtime serializes IDs sorted by str; semantics are an exact
                # set, not numeric ordering. Keep cardinality and set checks.
                lost_ids=r.get('raw_strict_lost_ids',[])
                assert ok==r['feasible']
                assert len(lost_ids)==len(set(lost_ids))==len(rids-ids)
                assert set(lost_ids)==rids-ids
                if ok:feasible.append((cid,d,r['actual_correction_norm']))
                row=dict(batch=b,candidate=cid,called=cid in ce,duplicate_of=r['duplicate_of'],finite=True,
                    trust_valid=r['trust_valid'],E=e,E_minus_RAW=e-rp,D64=d,D64_minus_RAW=d-rd,strict_count=len(ids),
                    RAW_strict_lost=len(rids-ids),feasible=ok,selected=cid==c['selected'],actual_correction_norm=r['actual_correction_norm'],
                    reason='|'.join(r['reason']) if isinstance(r['reason'],list) else r['reason'])
            else:row=dict(batch=b,candidate=cid,called=cid in ce,selected=cid==c['selected'],reason=str(r))
            candidates.append(row)
        best=min(x[1] for x in feasible);np=lock['numerical_policy'];order={'RAW':0,'C1':1,'C05':2,'C025':3}
        shortlist=[x for x in feasible if x[1]-best<=np['d_tie_atol']+np['d_tie_rtol']*max(abs(x[1]),abs(best))]
        computed=min(shortlist,key=lambda x:(x[0]!='RAW',x[2],order[x[0]]))[0]
        assert computed==c['selected']==pol['selection']['selected_id']
        diag=pol['correction'];chosen=ce[c['selected']]
        policyrows.append(dict(batch=b,selected=c['selected'],selection_reason=pol['selection']['reason'],raw_E=rp,selected_E=chosen['current']['E'],
            raw_D64=rd,selected_D64=chosen['generic']['D'],D64_change=chosen['generic']['D']-rd,
            raw_strict=len(rids),selected_strict=len(chosen['current']['strict_ids']),
            native_delta_norm=diag['actual_native_delta_norm'],correction_norm=next(r['actual_correction_norm'] for r in pol['selection']['candidate_receipts'] if r['id']==c['selected']),
            alpha=diag.get('alpha'),ball_projected_requests=diag.get('ball_projected_requests'),trust_retraction=diag.get('trust_retraction'),
            q=diag.get('projection',{}).get('q_ge_gd'),ge_dot_d=diag.get('projection',{}).get('ge_dot_d'),
            ge_dot_correction=diag.get('ge_dot_correction'),gd_dot_correction=diag.get('gd_dot_correction'),selector_arithmetic='MATCH'))
        ann=sel['requested_annotations'];accepted=[x for x in ann if x['selected_strict_success']]
        latest={};overwrites=0;reissues=0
        for x in accepted:
            k=(x['subject'],x['relation_id'])
            if k in latest:
                if latest[k]!=x['target']:overwrites+=1
                else:reissues+=1
            latest[k]=x['target']
        expected=[('UNKNOWN' if (x['subject'],x['relation_id']) not in latest else 'ACTIVE_TARGET' if latest[x['subject'],x['relation_id']]==x['target'] else 'SUPERSEDED') for x in ann]
        assert expected==[x['status'] for x in ann]
        lc=Counter(expected);ledgerrows.append(dict(batch=b,requested=len(ann),accepted=len(accepted),distinct_accepted_caseids=len({x['case_id'] for x in accepted}),
            distinct_accepted_subject_relation=len(latest),accepted_target_overwrites=overwrites,same_target_reissues=reissues,
            active=lc['ACTIVE_TARGET'],superseded=lc['SUPERSEDED'],unknown=lc['UNKNOWN'],accepted_current=c['ledger']['accepted_count']))
        counts=c['native'];anchor=counts['anchor_capture']
        cost.append(dict(batch=b,native_targets=counts['compute_z'],native_solves=counts['solve'],native_key_calls=counts['compute_ks'],
            actual_Adam=anchor['adam_updates'],loss_evaluations=anchor['loss_evaluations'],inner_history=counts['history_append'],final_history=c['history'][0]['history_append'],
            native_seconds=c['cost']['native_instrumented'],z_seconds=counts['compute_z_seconds'],native_keys_seconds=counts['compute_ks_seconds'],readout_seconds=counts['get_module_input_output_at_words_seconds'],
            solve_seconds=counts['solve_seconds'],map_seconds=c['cost']['map_setup_no_comparison'],screen_seconds=c['cost']['screen'],history_seconds=c['cost']['history'],
            selected_evaluation_seconds=c['cost']['selected_evaluation'],
            current_gradient_total=c['cost']['current_gradient'],current_gradient_FB='NOT_SEPARATELY_TIMED',
            S64_gradient_total=c['cost']['generic_gradient']['total'],S64_gradient_FB=c['cost']['generic_gradient']['forward_backward'],
            S64_teacher_read=c['cost']['generic_gradient']['teacher_read'],
            current_gradient_forwards=pol['gradient_current']['forwards'],current_gradient_backwards=pol['gradient_current']['backwards'],
            current_gradient_input_tokens=pol['gradient_current']['input_tokens'],current_gradient_scored_tokens=pol['gradient_current']['scored_tokens'],
            S64_gradient_forwards=pol['gradient_S64']['forwards'],S64_gradient_backwards=pol['gradient_S64']['backwards'],
            S64_gradient_scored_tokens=pol['gradient_S64']['scored_tokens'],
            corrected_candidate_forward_current=sum(v['current']['counts']['forwards'] for k,v in ce.items() if k!='RAW' and 'current' in v),
            corrected_candidate_forward_S64=sum(v['generic']['counts']['forwards'] for k,v in ce.items() if k!='RAW' and 'generic' in v),
            pure_writer='NOT_SEPARATED',checkpoint_IO='NOT_SEPARATELY_TIMED',peak_GPU_allocated=c['peak_gpu_allocated_bytes']))
        if b in (5,10):
            for scope,panel in [('FULL_SEEN',sel['seen_full'])]+([('FIRST500',sel['first500'])] if b==10 else []):
                for m in atwrite:whole.append(summary(panel_rows(panel,m),m,batch=b,scope=scope))
            dev=sel['Dev128'];general.append(dict(batch=b,population='Dev128_OBSERVER',D=dev['D'],denominator=dev['denominator'],**dev['counts']))
    # First table is actual W10; online own-batch pooling is explicitly separate.
    cohorts=[];annotations={x['case_id']:x for x in final['requested_annotations']}
    for m in atwrite:
        pairs.append(transition(atwrite[m],finalrows[m],m,comparison='ATWRITE_TO_W10',batch=10,population='ALL_REQUESTED'))
        whole.append(summary(atwrite[m],m,batch='mixed1-10',scope='ONLINE_OWN_BATCH_NOT_RETENTION'))
        w5=read(raw/'B005/selected-evaluation.json')['seen_full']
        pairs.append(transition(panel_rows(w5,m),panel_rows(final['first500'],m),m,comparison='W5_TO_W10_FIRST500',batch=10,population='FIRST500'))
        for cohort in range(1,11):
            ids=set(lock['batches'][cohort-1]['case_ids']);f=[x for x in finalrows[m] if x['case_id'] in ids];a=[x for x in atwrite[m] if x['case_id'] in ids]
            cohorts.append(dict(transition(a,f,m,comparison='ATWRITE_TO_W10',cohort=cohort,future_batches=10-cohort),**{'final_percent':100*sum(success(x,m) for x in f)/len(f)}))
        for status in ('ACTIVE_TARGET','SUPERSEDED','UNKNOWN','ACCEPTED_ONLY','UNACCEPTED_ONLY'):
            ids={i for i,x in annotations.items() if (x['selected_strict_success'] if status=='ACCEPTED_ONLY' else not x['selected_strict_success'] if status=='UNACCEPTED_ONLY' else x['status']==status)}
            rr=[x for x in finalrows[m] if x['case_id'] in ids]
            if rr:whole.append(summary(rr,m,batch=10,scope=status))
    for name,data in [('batch-current-metrics',metrics),('whole-prefix-metrics',whole),('paired-transitions',pairs),('cohort-retention',cohorts),
        ('candidate-details',candidates),('batch-policy',policyrows),('parity-warnings',warnings),('ledger-summary',ledgerrows),('cost-by-batch',cost),('general-observer',general)]:table(out/(name+'.csv'),data)
    result=dict(instruction_id=instruction_id,status='INDEPENDENT_NLL_AND_SELECTION_REDUCED',terminal=ref(raw/'terminal.json'),
        final=first,selection_counts=dict(Counter(x['selected'] for x in policyrows)),
        candidate_feasible_counts=dict(Counter(x['candidate'] for x in candidates if x.get('feasible'))),
        warnings=dict(panels=len(warnings),warn_panels=sum(x['recorded_status']=='WARNING' for x in warnings),
            max_abs_NLL_difference=max(x['max_abs_NLL_difference'] for x in warnings),strict_disagreements=sum(x['strict_lost']+x['strict_gained'] for x in warnings)),
        ledger_final=ledgerrows[-1],cost_totals={k:sum(x[k] for x in cost) for k in cost[0] if k not in ('batch','peak_GPU_allocated') and isinstance(cost[0][k],(int,float))},
        peak_GPU_allocated=max(x['peak_GPU_allocated'] for x in cost),
        terminal_seconds=terminal['seconds'],allocated_GPU_seconds=allocated_gpu_seconds,numerical_validation='NOT_ESTABLISHED',
        checksum_scope='THIS_REDUCER_JSON_INPUTS; tensor/raw full inventory separate')
    save(out/'metrics-summary.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worktree',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(reduce(a.worktree,a.output),ensure_ascii=False))
