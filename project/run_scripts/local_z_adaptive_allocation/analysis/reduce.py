"""Completed local-z factual tables. Independent arithmetic, no runtime imports."""
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import numpy as np
from .bootstrap import ROOT,REVIEW,identity,save
from .metrics import ARMS,MULT,read,table,digest,validate,select,summary,pair,stats

def fact(r):
    q=r['requested_rewrite'];return q['subject'],q['relation_id']

def past(records,stop,current):
    latest={fact(r):i for i,r in enumerate(records[:stop])};excluded={fact(r) for r in current}
    def priority(i):
        r=records[i];rw=r['requested_rewrite']
        event=digest(dict(ordinal=i,case_id=r['case_id'],subject=rw['subject'],relation=rw['relation_id'],target=rw['target_new']['str']))
        import hashlib
        return hashlib.sha256(('LZ-ALLOC-v1|20260916|past|'+event).encode()).hexdigest(),event
    return sorted([i for k,i in latest.items() if k not in excluded],key=priority)[:64]

def statuses(records):
    last={fact(r):r['requested_rewrite']['target_new']['str'] for r in records}
    return {r['case_id']:('ACTIVE' if r['requested_rewrite']['target_new']['str']==last[fact(r)] else 'SUPERSEDED') for r in records}

def selector(rows,mode='standard'):
    for row in rows:
        if not all(math.isfinite(row[k]) for k in ('E','D','action_norm')) or (row['H'] is not None and not math.isfinite(row['H'])):
            raise ValueError('NONFINITE_STORED_SCORE_NOT_QUALITY_FALLBACK')
    ref=next(r for r in rows if r['candidate_id']=='N4');observed=[];feasible=[]
    for r in rows:
        reasons=[]
        if mode!='strict_only' and r['E']>((max(ref['E'],.05) if mode=='standard' else ref['E'])+1e-4):reasons.append('CURRENT_MEAN')
        if not set(ref['S_cur'])<=set(r['S_cur']):reasons.append('CURRENT_STRICT_IDS')
        if ref['H'] is not None:
            if mode!='strict_only' and (r['H'] is None or r['H']>ref['H']+1e-4):reasons.append('PAST_MEAN')
            if not set(ref['S_past'])<=set(r['S_past']):reasons.append('PAST_STRICT_IDS')
        observed.append(dict(candidate_id=r['candidate_id'],reasons=reasons,feasible=not reasons))
        if not reasons:feasible.append(r)
    assert feasible
    lowest=min(r['D'] for r in feasible);tied=[r for r in feasible if r['D']<=lowest+1e-6]
    chosen=sorted(tied,key=lambda r:(r['candidate_id']!='N4',not r['L8_zero'],r['action_norm'],r['candidate_id']))[0]
    return chosen['candidate_id'],observed,[r['candidate_id'] for r in tied]

def scheduler_tables():
    audit=read(REVIEW/'scheduler-once.json');rows=[]
    mapping={'48679':'PREPARATION',**{f'48680_{i}':arm for i,arm in enumerate(('LD','N4','REFIT4','L75','T75','L4D','TD'))}}
    lines=[s.split('|') for s in audit['accounting']['stdout'].splitlines()]
    peaks={r[1].removesuffix('.batch'):r[13] for r in lines if r[1].endswith('.batch')}
    for r in lines:
        if '.' in r[1]:continue
        assert r[1] in mapping and r[3]=='janghj' and r[4]=='COMPLETED' and r[5]=='0:0'
        assert r[12]=='server4' and 'gres/gpu=1' in r[10]
        rows.append(dict(arm=mapping[r[1]],job=r[1],physical_job=r[0],state=r[4],exit=r[5],start_KST=r[7],end_KST=r[8],
            GPU_seconds=int(r[9]),GPU_hours=int(r[9])/3600,MaxRSS=peaks[r[1]],allocation=r[10],submit_KST=r[14],eligible_KST=r[15],
            queue_seconds=(datetime.fromisoformat(r[7])-datetime.fromisoformat(r[14])).total_seconds()))
    events=[]
    for r in rows:
        events.extend([(datetime.fromisoformat(r['start_KST']),1,r['arm']),(datetime.fromisoformat(r['end_KST']),-1,r['arm'])])
    events.sort(key=lambda x:(x[0],x[1]));active=set();segments=[];previous=None
    for time,change,arm in events:
        if previous is not None and time>previous:
            segments.append(dict(start_KST=previous.isoformat(),end_KST=time.isoformat(),GPUs=len(active),arms=','.join(sorted(active)),
                seconds=(time-previous).total_seconds(),above_cap1=len(active)>1))
        if change==1:active.add(arm)
        else:active.remove(arm)
        previous=time
    return rows,segments

def run(attempt='analysis-v1'):
    from scripts.fixed_counterfact import load_prefix
    records=load_prefix('/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1',1000)
    out=REVIEW/attempt;out.mkdir(exist_ok=False)
    batch_rows=[];population=[];transitions=[];cohorts=[];candidates=[];selection_rows=[];constraint=[];links=[];past_rows=[];costs=[];generic=[]
    finals={};atwrites={};entries={};selected_current={};selection_counts={};violations=[];inputs=[]
    w0=validate(read(ROOT/'technical/attempt-v1/W0-first1000.json'),records)
    population.extend(dict(arm='COMMON_W0',population='ALL1000',**summary(w0['metrics'][tag]['rows'],tag)) for tag in MULT)
    cold=read(ROOT/'technical/attempt-v1/cold-capsule.json');ready=read(ROOT/'technical/attempt-v1/READY.json')
    for check in ready['checks']:assert identity(check['path'])==check
    expected_counts={'N4':(100,1,1),'REFIT4':(200,2,1),'L75':(200,2,1),'T75':(100,2,1),'L4D':(100,1,2),'LD':(300,3,6),'TD':(200,4,7)}
    for arm in ARMS:
        root=ROOT/'arms'/arm/'attempt-v1/output';term=read(root/'terminal.json');start=read(root/'start.json')
        assert start['cold_zero'] and start['entry']['W']==cold['W0'] and start['entry']['rng']==digest(cold['rng'])
        assert start['teacher']['manifest_sha256']==cold['teacher_manifest']['sha256']
        previous=start['entry'];aw={tag:[] for tag in MULT};entry_all={tag:[] for tag in MULT};selection_counts[arm]=Counter()
        counters=Counter();components=Counter()
        final=validate(read(root/'B010/seen-full.json'),records);finals[arm]=final
        full5=validate(read(root/'B005/seen-full.json'),records[:500])
        for label,doc,subset_records in [('W10_ALL1000',final,records),('W5_FIRST500',full5,records[:500])]:
            state_status=statuses(subset_records)
            groups={'ALL':[r['case_id'] for r in subset_records]}
            groups.update({s:[i for i,v in state_status.items() if v==s] for s in ('ACTIVE','SUPERSEDED')})
            for group,ids in groups.items():
                if not ids:continue
                for tag,rr in select(doc,ids).items():population.append(dict(arm=arm,population=label,status=group,**summary(rr,tag)))
        for label,rrs in [('W10_FIRST500',records[:500]),('W10_LAST500',records[500:])]:
            for tag,rr in select(final,[r['case_id'] for r in rrs]).items():population.append(dict(arm=arm,population=label,status='ALL',**summary(rr,tag)))
        for tag,rr in select(final,[r['case_id'] for r in records[:500]]).items():
            transitions.append(dict(arm=arm,contrast='W5_FIRST500_TO_W10_SAME500',**pair(full5['metrics'][tag]['rows'],rr,tag)))
        for b in range(1,11):
            path=root/f'B{b:03d}';cur=records[(b-1)*100:b*100];curids=[r['case_id'] for r in cur]
            entry=read(path/'entry.json');commit=read(path/'commit.json');generation=read(path/'proposals/generation.json');sel=read(path/'selection.json')
            assert entry['state']==previous==commit['entry']==generation['entry']
            assert entry['received']==list(range((b-1)*100)) and entry['record_digest']==digest(cur)
            assert commit['next_ordinal']==b*100 and commit['received_count']==b*100
            assert identity(path/'commit.json')==term['commits'][b-1]
            assert generation['inner_history_append']==0
            assert (generation['targets'],generation['solves'],len(generation['declared_candidates']))==expected_counts[arm]
            eligible=[4,8] if arm in ('LD','TD','L75','T75') else [4]
            assert [h['layer'] for h in commit['history']]==eligible
            for h in commit['history']:
                assert h['history_append']==1 and h['before_sha256']==entry['state']['M'][str(h['layer'])]
                assert h['after_sha256']==commit['state']['M'][str(h['layer'])] and h['weight_sha256']==commit['state']['W'][str(h['layer'])]
                counters['history']+=1
            for l in (4,8):
                if l not in eligible:assert entry['state']['M'][str(l)]==commit['state']['M'][str(l)]
            assert commit['state']['P']==start['entry']['P'] and commit['state']['contexts']==start['entry']['contexts']
            assert commit['state']['rng']==start['entry']['rng']
            if b>1:
                assert entry['previous_commit']==term['commits'][b-2]
                links.append(dict(arm=arm,from_batch=b-1,to_batch=b,W_M_P_context_RNG_exact=True,received_exact=True))
            previous=commit['state'];selection_counts[arm][sel['selected']]+=1
            assert commit['selected']==sel['selected'] and commit['actual_candidate']['weights']==commit['state']['W']
            observed_past=read(path/'past64.json');computed=past(records,(b-1)*100,cur)
            assert observed_past['ordinals']==computed and observed_past['case_ids']==[records[i]['case_id'] for i in computed]
            past_rows.append(dict(arm=arm,batch=b,count=len(computed),ordinals_sha=digest(computed),excludes_current_facts=True,received_only=True))
            docs={name:validate(read(path/(name+'-current.json')),cur) for name in ('entry','selected')}
            for state,doc in docs.items():
                for tag,m in doc['metrics'].items():batch_rows.append(dict(arm=arm,batch=b,state=state,**summary(m['rows'],tag)))
            for tag in MULT:
                before=docs['entry']['metrics'][tag]['rows'];after=docs['selected']['metrics'][tag]['rows']
                transitions.append(dict(arm=arm,batch=b,contrast='ENTRY_TO_ATWRITE',**pair(before,after,tag)))
                later=select(final,curids)[tag]
                tr=dict(arm=arm,batch=b,contrast='ATWRITE_TO_W10',future_batches=10-b,**pair(after,later,tag))
                transitions.append(tr);cohorts.append(tr)
                aw[tag].extend(after);entry_all[tag].extend(before)
            cr=[read(p) for p in sorted(path.glob('candidate-*.json'))]
            assert len(cr)==len(generation['declared_candidates'])==sel['declared']
            by={r['candidate_id']:r for r in cr};meta={r['candidate_id']:r for r in generation['declared_candidates']}
            assert by.keys()==meta.keys() and len({digest(r['weights']) for r in cr})==sel['distinct_materializations']
            dynamic=arm in ('L4D','LD','TD')
            if dynamic:
                chosen,conditions,ties=selector(cr);assert chosen==sel['selected']
                assert set(ties)==set(sel['ties'])
                assert selector(cr,'no_plateau')[0]==sel['shadow_no_plateau'] and selector(cr,'strict_only')[0]==sel['shadow_strict_only']
                old_conditions={r['candidate_id']:r for r in sel['observations']}
                for c in conditions:assert c['reasons']==old_conditions[c['candidate_id']]['reasons'] and c['feasible']==old_conditions[c['candidate_id']]['feasible']
            else:conditions=[dict(candidate_id=cr[0]['candidate_id'],feasible=True,reasons=[])];assert sel['quality_screen']=='NOT_APPLIED_FIXED_POLICY'
            ref=by.get('N4');chosen=by[sel['selected']]
            selection_rows.append(dict(arm=arm,batch=b,selected=sel['selected'],a4=chosen['a4'],a8=chosen['a8'],D=chosen['D'],E=chosen['E'],H=chosen['H'],
                current_strict=len(chosen['S_cur']),past_strict=len(chosen['S_past']),action_norm=chosen['action_norm'],own_N4=sel['selected']=='N4',
                shadow_no_plateau=sel.get('shadow_no_plateau','NOT_APPLIED'),shadow_strict_only=sel.get('shadow_strict_only','NOT_APPLIED'),
                declared=sel['declared'],distinct=sel['distinct_materializations'],D_vs_own_N4=None if ref is None else chosen['D']-ref['D'],
                E_vs_own_N4=None if ref is None else chosen['E']-ref['E']))
            for r in cr:
                detail=r['details'];training=detail['training'];canonical=detail['current'];old=detail['past'];g=detail['generic']
                assert all(math.isfinite(r[k]) for k in ('E','D','action_norm')) and (r['H'] is None or math.isfinite(r['H']))
                assert r['weights']==meta[r['candidate_id']]['weights']
                assert [x['case_id'] for x in canonical['rows']]==curids
                assert r['S_cur']==[x['case_id'] for x in canonical['rows'] if x['all_tokens_correct']]
                assert abs(math.fsum(x['nll'] for x in training['rows'])/100-r['E'])<1e-12
                assert abs(math.fsum(x['kl'] for x in g['rows'])/64-r['D'])<1e-12
                assert g['denominator']==64 and all(x['scored_positions']==128 for x in g['rows'])
                if old:
                    assert [x['case_id'] for x in old['rows']]==observed_past['case_ids']
                    assert r['S_past']==[x['case_id'] for x in old['rows'] if x['all_tokens_correct']]
                    assert abs(math.fsum(x['nll'] for x in old['rows'])/len(computed)-r['H'])<1e-12
                else:assert b==1 and r['H'] is None and not r['S_past']
                cond=next(c for c in conditions if c['candidate_id']==r['candidate_id'])
                constraint.append(dict(cond,arm=arm,batch=b,reasons=';'.join(cond['reasons']),screen='DYNAMIC' if dynamic else 'FIXED_NO_SCREEN'))
                row=dict(arm=arm,batch=b,candidate=r['candidate_id'],a4=r['a4'],a8=r['a8'],E=r['E'],H=r['H'],D=r['D'],
                    current_strict=len(r['S_cur']),past_strict=len(r['S_past']),action_norm=r['action_norm'],L8_zero=r['L8_zero'],
                    selected=r['candidate_id']==sel['selected'],score_reused_from=r['score_reused_from'],
                    E_ceiling=None if not dynamic else max(ref['E'],.05)+1e-4,
                    H_ceiling=None if not dynamic or ref['H'] is None else ref['H']+1e-4,
                    strict_lost_vs_N4=None if ref is None else len(set(ref['S_cur'])-set(r['S_cur'])),
                    past_strict_lost_vs_N4=None if ref is None else len(set(ref['S_past'])-set(r['S_past'])))
                row.update({'current_new_NLL_'+k:v for k,v in stats([x['nll'] for x in canonical['rows']]).items()})
                if ref:
                    delta=[x['nll']-y['nll'] for x,y in zip(canonical['rows'],ref['details']['current']['rows'])]
                    row.update({'new_NLL_vs_ownN4_'+k:v for k,v in stats(delta).items()});row['requests_harmed_vs_ownN4']=sum(v>0 for v in delta)
                candidates.append(row)
                if r['score_reused_from'] is None:
                    counters['E_forward_groups']+=training['model_forwards'];counters['S64_forward_docs']+=g['counts']['forwards']
                    counters['S64_scored_tokens']+=g['counts']['scored_tokens'];counters['canonical_current_forward_groups']+=canonical['counts']['forwards']
                    counters['Past_forward_groups']+=old['counts']['forwards'] if old else 0
                    components['training_E_seconds']+=training['seconds'];components['S64_seconds']+=g['seconds']['total']
                    components['teacher_stream_seconds']+=g['seconds']['teacher_read'];components['canonical_current_seconds']+=canonical['seconds']
                    components['Past_seconds']+=old['seconds'] if old else 0
            counters['target_calls']+=generation['targets'];counters['solve']+=generation['solves'];counters['candidate_declared']+=sel['declared'];counters['candidate_distinct']+=sel['distinct_materializations']
            for fit in (path/'proposals').glob('*.json'):
                if fit.name=='generation.json':continue
                f=read(fit);assert f['input_state']['M']==entry['state']['M'] and f['input_state']['P']==entry['state']['P']
                assert f['input_state']['contexts']==entry['state']['contexts'] and f['input_state']['rng']==entry['state']['rng']
                for k in ('compute_z_seconds','compute_ks_seconds','get_module_input_output_at_words_seconds','solve_seconds'):
                    components[k]+=f.get(k,0)
                counters['local_Adam']+=f.get('anchor_capture',{}).get('adam_updates',0);counters['local_loss']+=f.get('anchor_capture',{}).get('loss_evaluations',0)
            if b in (5,10):
                dev=read(path/'Dev128.json');generic.append(dict(arm=arm,batch=b,role='Dev128',D=dev['D'],documents=128,observer_only=True,
                    natural_NLL_mean=float(np.mean([r['natural_nll'] for r in dev['rows']]))))
        assert previous==term['terminal_state'] and len(term['commits'])==10
        atwrites[arm]=aw;entries[arm]=entry_all
        for tag in MULT:
            population.append(dict(arm=arm,population='POOLED_ATWRITE_NOT_SINGLE_MODEL',status='ALL',**summary(aw[tag],tag)))
            transitions.append(dict(arm=arm,contrast='POOLED_ATWRITE_TO_W10',**pair(aw[tag],final['metrics'][tag]['rows'],tag)))
            transitions.append(dict(arm=arm,contrast='COMMON_W0_TO_W10',**pair(w0['metrics'][tag]['rows'],final['metrics'][tag]['rows'],tag)))
        costs.append(dict(arm=arm,**counters,**components,program_seconds=term['seconds'],
            peak_GPU_allocated_bytes=term['peak_GPU_allocated'],peak_GPU_reserved_bytes=term['peak_GPU_reserved'],
            peak_host_RSS_KiB=term['peak_host_RSS_KiB'],**{k+'_inclusive':v for k,v in term['timers']['components'].items()},
            **{k+'_runtime':v for k,v in term['timers']['runtime'].items()},nested_time_sum_forbidden=True))
    contrasts=[(a,'N4') for a in ARMS if a!='N4']+[('LD','L4D'),('LD','L75'),('LD','REFIT4'),('LD','TD'),('T75','L75')]
    comparisons=[]
    for treatment,reference in contrasts:
        for tag in MULT:comparisons.append(dict(treatment=treatment,reference=reference,population='W10_ALL1000',**pair(finals[reference]['metrics'][tag]['rows'],finals[treatment]['metrics'][tag]['rows'],tag,ci=True)))
    for arm in ARMS:
        current=statuses(records)
        for group in ('ACTIVE','SUPERSEDED'):
            ids={k for k,v in current.items() if v==group}
            if not ids:continue
            for tag in MULT:transitions.append(dict(arm=arm,contrast='ATWRITE_TO_W10_'+group,**pair([r for r in atwrites[arm][tag] if r['case_id'] in ids],select(finals[arm],ids)[tag],tag)))
    sched,overlap=scheduler_tables()
    for name,rows in [('batch-metrics',batch_rows),('population-metrics',population),('paired-transitions',transitions),('cohorts',cohorts),
        ('paired-policy-comparisons',comparisons),('candidate-metrics',candidates),('selection',selection_rows),('constraints',constraint),
        ('state-links',links),('past64',past_rows),('compute-components',costs),('generic-observer',generic),('scheduler-cost',sched),('allocation-intervals',overlap)]:table(out/(name+'.csv'),rows)
    summary_data=dict(status='CPU_METRICS_STATE_JSON_SELECTOR_PAST_COUNTS_PASS',arms=7,batches=70,links=len(links),
        candidates=len(candidates),history=sum(r['history'] for r in costs),target_calls=sum(r['target_calls'] for r in costs),solves=sum(r['solve'] for r in costs),
        selection_counts={a:dict(c) for a,c in selection_counts.items()},teacher_regenerated=cold['teacher_regenerated'],
        teacher_W0_D=read(ROOT/'technical/attempt-v1/teacher-effective-check.json')['D'],
        maximum_allocated_GPUs=max(r['GPUs'] for r in overlap),cap1_exceed_seconds=sum(r['seconds'] for r in overlap if r['above_cap1']),
        science_GPU_seconds=sum(r['GPU_seconds'] for r in sched if r['arm']!='PREPARATION'),preparation_GPU_seconds=sched[0]['GPU_seconds'],
        unchanged_current_N4_TD_NLL_pairs=all(finals['N4']['metrics'][t]['rows']==finals['TD']['metrics'][t]['rows'] for t in MULT),
        active_counts=dict(Counter(statuses(records).values())),violations=violations,GPU_calls=0,
        missing=['candidate_official_R_P_N','saved_finalizer_keys','native_iteration_clamp_hits','GPU_off_on_continuation'])
    save(out/'summary.json',summary_data);print(json.dumps(summary_data,ensure_ascii=False,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--attempt',default='analysis-v1');a=p.parse_args();run(a.attempt)
