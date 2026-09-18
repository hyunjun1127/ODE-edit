"""Independent stored-evidence audit; never imports the scientific runtime."""
import collections, datetime, hashlib, json, math, subprocess
from pathlib import Path
from reducer import *

CHECKS=[]
def check(name,condition,**ctx):
    CHECKS.append(dict(check=name,pass_=bool(condition),**ctx))
def reference(ref):
    p=Path(ref['path']);ok=p.is_file() and p.stat().st_size==ref['bytes'] and sha(p)==ref['sha256']
    check('REFERENCED_MEMBER_SHA',ok,path=str(p));return load(p)
def reasons(c,n):
    s,r=c['scores'],n['scores'];out=[]
    if s['training_e']>r['training_e']+1e-4:out.append('CURRENT_TRAINING_MEAN')
    for prefix in ['current','past']:
        if prefix=='past':
            if r['past_h'] is None:continue
            if s['past_h'] is None or s['past_h']>r['past_h']+1e-4:out.append('PAST_MEAN')
        for kind in ['strict','pair']:
            key=prefix+'_'+kind
            if r[key] is not None and not set(r[key]).issubset(s[key] or []):out.append(prefix.upper()+'_'+kind.upper()+'_IDS')
    return out
def choose(cs,arm):
    if arm=='F48':return next(c for c in cs if c['gates']==[.75,.5])
    n=next(c for c in cs if c['is_n4']);fs=[c for c in cs if not reasons(c,n)]
    best=min(c['scores']['base_kl'] for c in fs)
    return min((c for c in fs if c['scores']['base_kl']<=best+1e-6),key=lambda c:(not c['is_n4'],len(c['active_layers']),c['action_norm'],c['gates'],c['state_token']))
def token(state):return digest({k:v for k,v in state.items() if k!='M'})
def score_equal(a,b):
    # Runtime serial() sorts IDs numerically, controller json_safe() sorts repr.
    # These four fields are sets by contract; margins/scalars remain exact.
    def normalized(s):return {k:sorted(v) if k in ['current_strict','current_pair','past_strict','past_pair'] and v is not None else v for k,v in s.items()}
    return normalized(a)==normalized(b)
def fact(r):return (r['requested_rewrite']['subject'],r['requested_rewrite']['relation_id'])
def past_expected(records,start,stop):
    latest={fact(r):i for i,r in enumerate(records[:start])};excluded={fact(r) for r in records[start:stop]}
    def eid(i):
        r=records[i];return digest(dict(ordinal=i,case_id=r['case_id'],subject=fact(r)[0],relation=fact(r)[1],target=r['requested_rewrite']['target_new']['str']))
    eligible=[i for f,i in latest.items() if f not in excluded]
    return sorted(eligible,key=lambda i:(hashlib.sha256(('LZ-ALLOC-v1|20260916|past|'+eid(i)).encode()).hexdigest(),eid(i)))[:64],eid
def finite(x):
    if isinstance(x,float):return math.isfinite(x)
    if isinstance(x,dict):return all(finite(v) for v in x.values())
    if isinstance(x,list):return all(finite(v) for v in x)
    return True
def main():
    records=load('/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json')[:1000]
    lock=load(ROOT/'execution.lock.json');source=[]
    for name in ['common','controller','runner','runtime','native','metrics','technical','control','operations']:
        rel='project/run_scripts/sequential_local_z_allocation/'+name+'.py';p=ROOT/'source-v1'/rel
        gitbytes=subprocess.check_output(['git','show',lock['source_head']+':'+rel],cwd=WT)
        check('FROZEN_SOURCE_COMMIT',p.read_bytes()==gitbytes and (WT/rel).read_bytes()==gitbytes,path=rel)
        source.append(dict(path=rel,sha256=sha(p),bytes=p.stat().st_size,execution=lock['source_head']))
    for key in ['source_archive','cold_capsule','teacher_manifest','W0_observation']:
        ref=lock[key];check('LOCKED_SMALL_INPUT',sha(ref['path'])==ref['sha256'],input=key)
    check('EXECUTION_LOCK',sha(ROOT/'execution.lock.json')=='a41cb76a25ab98b02c043397cd9cf4db0768e9ae312931b349008c3e9b303d18')
    ready=load(ROOT/'technical/attempt-v1/READY.json');reference(ready['capsule'])
    selections=[];candidates=[];costs=[];fits=[];scorecost=[];prunes=[];history=[];states=[];events_rows=[];common_start=None
    for arm in ARMS:
        out=ROOT/'arms'/arm/'attempt-v1/output';start=load(out/'start.json');term=load(out/'terminal.json')
        check('NO_FAILURE_MARKER',not (out/'failure.json').exists(),arm=arm)
        check('TERMINAL',term['batches']==10 and term['requests']==1000 and term['status']=='COMPLETED_1000_REQUESTS',arm=arm)
        check('COLD_AND_LOCK',start['W0_cold'] and start['M0']=='EXACT_ZERO_ALL5' and start['execution_lock']['sha256']==sha(ROOT/'execution.lock.json'),arm=arm)
        if common_start is None:common_start=start
        check('COMMON_COLD_STATE',start['entry']==common_start['entry'] and start['teacher']==common_start['teacher'],arm=arm)
        previous=start['entry'];count=collections.Counter()
        for b in range(1,11):
            bd=out/f'B{b:03d}';ctx=dict(arm=arm,batch=b);entry=load(bd/'entry.json');sel=load(bd/'selection.json');commit=load(bd/'commit.json');hi=load(bd/'episode/history.json')
            reference(term['commits'][b-1]);reference(commit['selection']);reference(commit['history'])
            check('ENTRY_LINK',entry['state']==previous and commit['entry']==entry['state'],**ctx)
            check('ENTRY_REQUEST_ORDER',entry['ordinals']==[(b-1)*100,b*100] and entry['record_digest']==digest(records[(b-1)*100:b*100]),**ctx)
            check('COMMIT_SELECTED',commit['selected']==sel['selected'] and commit['source']==lock['source_head'],**ctx)
            check('SELECTED_STATE_HASH',token(commit['state'])==sel['selected']['state_token'],**ctx)
            check('RECEIVED_ALL_REQUESTS',commit['received']==list(range(b*100)) and commit['next_ordinal']==b*100 and commit['next_batch']==b+1,**ctx)
            check('P_CONTEXT_RNG_FIXED',all(commit['state'][k]==start['entry'][k] for k in ['P','contexts','rng']),**ctx)
            check('HISTORY_ALL5_ONCE',commit['history_appends']==5 and commit['inner_history_appends']==0 and hi['appends']==5 and hi['candidate_appends']==0 and hi['all_current_requests']==100 and not hi['gate_weighting'] and [r['layer'] for r in hi['rows']]==[4,5,6,7,8],**ctx)
            check('HISTORY_STATE_LINK',hi['selected']==commit['state'] and hi['entry']['M']==entry['state']['M'] and token(hi['entry'])==sel['selected']['state_token'],**ctx)
            for h in hi['rows']:
                l=str(h['layer']);check('HISTORY_LAYER_HASH',h['history_append']==1 and h['before_sha256']==entry['state']['M'][l] and h['after_sha256']==commit['state']['M'][l] and h['weight_sha256']==commit['state']['W'][l],**ctx,layer=l)
                history.append(dict(**ctx,**h,selected_layer_changed=l in [str(x) for x in sel['selected']['active_layers']]))
            ps=load(bd/'past64.json');po,eid=past_expected(records,(b-1)*100,b*100)
            check('PAST64_RECEIVED_HASH_PRIORITY',ps['ordinals']==po and ps['case_ids']==[records[i]['case_id'] for i in po] and ps['stable_event_ids']==[eid(i) for i in po],**ctx)
            check('NO_DISK_CHECKPOINT',commit['checkpoint'] is None and not commit['tensors_independently_reconstructable'],**ctx)
            cs=sel['candidates'];n=next(c for c in cs if c['is_n4']);chosen=choose(cs,arm)
            check('INDEPENDENT_SELECTOR',chosen==sel['selected'],**ctx)
            ev=[load(p) for p in sorted((bd/'controller-ledger').glob('*.json'))]
            ec=collections.Counter(e['event'] for e in ev)
            check('LEDGER_SEQUENCE',all(e['sequence']==i for i,e in enumerate(ev)),**ctx)
            check('SEALED_AND_FINALIZED_ONCE',ec['SELECTION_SEALED']==ec['FINALIZATION_BEGIN']==ec['FINALIZATION_COMPLETE']==1 and ec['TECHNICAL_FAILURE']==0,**ctx)
            check('POSTSEAL_CONTROLLER_QUIET',[e['event'] for e in ev if e['sequence']>next(e['sequence'] for e in ev if e['event']=='SELECTION_SEALED')]==['FINALIZATION_BEGIN','FINALIZATION_COMPLETE'],**ctx)
            for name,val in ec.items():events_rows.append(dict(**ctx,event=name,count=val))
            cache=set();phasecount=collections.Counter();adam=0
            for e in ev:
                if e['event']=='FIT_RESERVED':
                    key=tuple(e['cache_key']);check('FIT_FRESH_KEY',key not in cache,**ctx);cache.add(key)
                    if e['layer']!=4:
                        check('WHOLE_FIT_RESERVATION',e['extra_reservation']==2400 and e['counts']['extra_adam']<=7200,**ctx)
                if e['event']=='FIT_CACHE_HIT':check('CACHE_KEY_WAS_FITTED',tuple(e['cache_key']) in cache,**ctx)
                if e['event']=='FIT_COMPLETE' and e['layer']!=4:
                    phasecount[e['phase']+'_fits']+=1;adam+=e['actual_adam'];check('ADAM_REFUND',e['unused_adam_reservation_released']==2400-e['actual_adam'],**ctx)
                if e['event']=='SCORE_COMPLETE' and not e['baseline']:phasecount[e['phase']+'_scores']+=1
                if e['event']=='RAW_BOUNDS_REJECTED':check('RAW_BOUNDS_GPU0',e['model_called'] is False and e['clipped'] is False and e['scored_candidate'] is False,**ctx)
            ct=commit['controller_counts'];check('BUDGET_LIMITS',phasecount['search_fits']<=32 and sum(v for k,v in phasecount.items() if k.endswith('_fits'))<=40 and phasecount['search_scores']<=24 and ct['endpoints']<=28 and adam==ct['extra_adam']<=9600 and ct['extra_adam_reserved']==0,**ctx)
            check('CALL_COUNTS',ct['l4_fits']==1 and ct['commits']==1 and ct['technical_failures']==0 and len(commit['fit_receipts'])==1+ct['suffix_fits'] and len(commit['score_receipts'])==1+ct['endpoints'],**ctx)
            for i,c in enumerate(cs):
                rs=reasons(c,n);check('FEASIBILITY_EXACT_IDS',c['feasible']==(not rs) and c['reasons']==rs and finite(c['scores']),**ctx,candidate=i)
                s=c['scores'];r=n['scores'];candidates.append(dict(**ctx,candidate=i,gates=json.dumps(c['gates']),state=c['state_token'],phase=c['phase'],own_n4=c['is_n4'],selected=c==sel['selected'],E=s['training_e'],canonical_E=s['canonical_e'],H=s['past_h'],B=s['base_kl'],delta_E=s['training_e']-r['training_e'],delta_B=s['base_kl']-r['base_kl'],delta_H=None if r['past_h'] is None else s['past_h']-r['past_h'],strict=len(s['current_strict']),past_strict=len(s['past_strict']),strict_lost=len(set(r['current_strict'])-set(s['current_strict'])),pair_lost=len(set(r['current_pair'] or [])-set(s['current_pair'] or [])),past_strict_lost=len(set(r['past_strict'])-set(s['past_strict'])),feasible=not rs,reasons=';'.join(rs),support=len(c['active_layers']),gate_support=sum(g!=0 for g in c['gates']),active_layers=json.dumps(c['active_layers']),action_norm=c['action_norm']))
            check('INCOMPLETE_NOT_SCORED',all(not i['scored'] and not i['completed'] and not i['feasible'] for i in sel['incomplete']),**ctx)
            search=[c for c in cs if c['phase']=='search'];coverage=dict(completed_search_gate_vectors=len(search),completed_unique_state_tokens=len({c['state_token'] for c in search}),distinct_a4=len({c['gates'][0] for c in search}))
            cv=sel['stop'].get('coverage');check('COVERAGE_RECOUNT',cv is None or all(cv[k]==v for k,v in coverage.items()),**ctx)
            c=sel['selected'];sc=c['scores'];row=dict(**ctx,gates=json.dumps(c['gates']),support=len(c['active_layers']),gate_support=sum(g!=0 for g in c['gates']),action_norm=c['action_norm'],own_n4=c['is_n4'],feasible=c['feasible'],E=sc['training_e'],H=sc['past_h'],B=sc['base_kl'],raw_E=n['scores']['training_e'],raw_B=n['scores']['base_kl'],delta_E=sc['training_e']-n['scores']['training_e'],delta_B=sc['base_kl']-n['scores']['base_kl'],stop=sel['stop']['reason'],candidates=len(cs),unique_endpoints=len({x['state_token'] for x in cs}),incomplete=len(sel['incomplete']),adequate=None if cv is None else cv['adequate_by_count_proxy'],**coverage,**ct,**phasecount)
            selections.append(row)
            for e in ev:
                if e['event'].startswith('PRUNING_'):prunes.append(dict(**ctx,event=e['event'],sequence=e['sequence'],layer=e.get('layer'),details=json.dumps({k:v for k,v in e.items() if k not in ['counts','sequence','event']},sort_keys=True)))
            for ref in commit['fit_receipts']:
                f=reference(ref);l=str(f['layer']);check('FIT_STATE_CONFINEMENT',f['history_append']==0 and f['compute_z']==100 and f['solve']==1 and f['compute_ks']==1 and f['input_state']['M']==entry['state']['M'] and all(f['input_state'][k]==f['output_state'][k] for k in ['M','P','contexts','rng']) and all(f['input_state']['W'][j]==f['output_state']['W'][j] for j in ['4','5','6','7','8'] if j!=l),**ctx,fit=ref['path'])
                fits.append(dict(**ctx,path=ref['path'],layer=f['layer'],targets=f['compute_z'],Adam=f['adam_updates'],loss=f['loss_evaluations'],clamp=f['clamp_hits'],solve=f['solve'],key=f['compute_ks'],seconds=f['seconds'],target_seconds=f['compute_z_seconds'],key_seconds=f['compute_ks_seconds'],solve_seconds=f['solve_seconds'],input_state=digest(f['input_state']),output_state=digest(f['output_state']),evidence=f['evidence']['path'],evidence_sha=f['evidence']['sha256'],evidence_bytes=f['evidence']['bytes']))
            fit_events=[e for e in ev if e['event']=='FIT_COMPLETE']
            check('NATIVE_PREFIX_EVENT_BINDING',all(e['cache_key'][2]==token(load(r['path'])['input_state']) and e['cache_key'][3]==digest(load(r['path'])['input_state']['M']) and e['native_state']==token(load(r['path'])['output_state']) for e,r in zip(fit_events,commit['fit_receipts'])),**ctx)
            bytoken={};states_by_token={}
            for ref in commit['score_receipts']:
                score=reference(ref);m=score['metrics'];dt=m['details'];bytoken[token(score['state'])]=m['controller'];states_by_token[token(score['state'])]=score['state']
                check('ONLINE_INPUT_SEPARATION',m['official_P_N_access']==0 and m['Dev_access']==0 and m['gradients']==0 and m['canonical_Current_mean_is_guard'] is False,**ctx)
                check('ONLINE_REDUCTION',abs(math.fsum(x['nll'] for x in dt['training']['rows'])/100-m['E'])<1e-12 and len(dt['training']['rows'])==100 and dt['generic']['role']=='S64' and dt['generic']['denominator']==64,**ctx)
                # Canonical records carry the original argmax tie convention; only re-aggregate stored booleans.
                for panel in ['current','past']:
                    d=dt[panel]
                    if d is None:continue
                    check('STRICT_PAIR_AGGREGATION',set(d['strict_ids'])=={r['case_id'] for r in d['rows'] if r['all_tokens_correct']} and set(d['pair_ids'])=={r['case_id'] for r in d['rows'] if r['old_nll'] is not None and r['new_nll']<r['old_nll']},**ctx,panel=panel)
                for panel,d in dt.items():
                    if d is None:continue
                    counts=d.get('counts',{});secs=d.get('seconds',0)
                    scorecost.append(dict(**ctx,path=ref['path'],panel=panel,seconds=secs if isinstance(secs,(float,int)) else secs.get('total'),forwards=d.get('model_forwards',counts.get('forwards')),backwards=counts.get('backwards',0),input_tokens=d.get('input_tokens',counts.get('input_tokens')),scored_tokens=d.get('scored_tokens',counts.get('scored_tokens')),teacher_reads=counts.get('teacher_reads',0),teacher_read_seconds=secs.get('teacher_read',0) if isinstance(secs,dict) else 0))
            check('CANDIDATE_SCORE_BINDING',all(score_equal(bytoken[c['state_token']],c['scores']) for c in cs),**ctx)
            check('ACTUAL_SUPPORT_FROM_WEIGHT_HASH',all(c['active_layers']==[l for l in range(4,9) if states_by_token[c['state_token']]['W'][str(l)]!=entry['state']['W'][str(l)]] for c in cs),**ctx)
            for binding in (bd/'candidate-observer').glob('*-binding.json'):
                d=load(binding);reference(d['rows']);check('CANDIDATE_OBSERVER_SEAL',d['selection_seal']==commit['selection'] and d['controller_feedback']==0 and d['incomplete_suffix_completed'] is False and d['state_token'] in bytoken,**ctx)
            states.append(dict(**ctx,entry=digest(entry['state']),selected=digest(commit['state']),selection_sha=sha(bd/'selection.json'),commit_sha=sha(bd/'commit.json'),past_ids_sha=digest(ps['case_ids']),W=digest(commit['state']['W']),M=digest(commit['state']['M']),nextordinal=commit['next_ordinal'],checkpoint='NOT_SAVED_USER_DIRECTED'))
            previous=commit['state'];count.update(ct)
        check('TERMINAL_STATE',previous==term['terminal_state'] and term['history_counts']=={str(l):10 for l in range(4,9)},arm=arm)
        gate=load(out/'initial-gate.json');check('STORED_INITIAL_LINK',gate['B2_entry']==load(out/'B002/entry.json')['state'] and gate['history_appends_B1']==5,arm=arm)
        costs.append(dict(arm=arm,program_seconds=term['seconds'],peak_GPU_allocated=term['peak_GPU_allocated'],peak_GPU_reserved=term['peak_GPU_reserved'],peak_host_RSS_KiB=term['peak_host_RSS_KiB'],**count))
        print('AUDITED',arm,flush=True)
    for name,rows in [('selection.csv',selections),('candidate.csv',candidates),('fit-cost.csv',fits),('score-cost.csv',scorecost),('pruning.csv',prunes),('history.csv',history),('state-links.csv',states),('event-counts.csv',events_rows),('compute.csv',costs),('source-inventory.csv',source)]:csvout(name,rows)
    writejson(LOCAL/'audit-checks.json',CHECKS)
    writejson(REPORT/'audit-summary.json',dict(checks=len(CHECKS),failed=[x for x in CHECKS if not x['pass_']],selected=len(selections),candidates=len(candidates),fits=len(fits),history_rows=len(history),sources=source))
    print('CHECKS',len(CHECKS),'FAIL',sum(not x['pass_'] for x in CHECKS))
if __name__=='__main__':main()
