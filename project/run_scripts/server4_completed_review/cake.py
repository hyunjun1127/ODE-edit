"""CAKE saved NLL/state telemetry review; no model, CUDA, evaluator or replay."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

from .begin import LOCAL, ROOT, WT, TASK
from project.run_scripts.bg_tw_reference.ep_tw import review_nogate as r

ATTEMPT = ROOT/'local/cake-native-lifelong/20260915-v1/attempt-v1'
RAW = ATTEMPT/'output/main'
OLDREPORT = WT/'experiment-reports/servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1'
MS = ('RS','PS','NS')
CP = (1,5,10,20,30,40,50,60,70,80,90,100)

def csvread(p):
    with Path(p).open() as f: return list(csv.DictReader(f))

def label(x):
    return {'BASE_ALPHAEDIT':'BASE_ALPHAEDIT_NATIVE','BASE_MEMIT':'BASE_MEMIT_NATIVE',
            'AlphaEdit_ORIGINAL':'AlphaEdit_BLUE(L4+L8)','MEMIT_ORIGINAL':'MEMIT_BLUE(L4+L8)'}.get(x,x)

def first(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    terminal=r.read(RAW/'terminal.json'); assert terminal['status']=='TERMINAL_VALID' and terminal['requests']==10000
    final=r.read(RAW/'B100/seen-full.json'); assert final['state']==terminal['actual_final_state']
    rows=[r.summary(r.panel_rows(final,m),m,method='CAKE_NATIVE',state='ACTUAL_W100_FULL10000') for m in MS]
    assert [x['denominator'] for x in rows]==[10000,20000,100000]
    r.table(out/'first-final-table.csv',rows)
    return r.save(out/'receipt.json',dict(task=TASK,inputs=[r.ref(RAW/'terminal.json'),r.ref(RAW/'B100/seen-full.json')],
        rows=rows,level='NLL_CARDINALITY_FINITE_TIES_IDENTITY_TERMINAL_BINDING_NOT_FULL_AUDIT',model_forwards=0))

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    lock=r.read(ATTEMPT/'execution.lock.json');rt=r.read(RAW/'runtime.json');term=r.read(RAW/'terminal.json')
    assert r.sha(ATTEMPT/'execution.lock.json')=='f2ade3ee9dbdabdc6a1ac00a9d36b0e902710a44cf5e2a6a49c6d861e18eb401'
    assert rt['lock_sha256']==r.sha(ATTEMPT/'execution.lock.json') and rt['cold_history']
    assert rt['source']==lock['source'] and rt['hparams']==lock['hparams']
    assert rt['model_revision']==lock['revision'] and rt['projector_mapping']==lock['projector_mapping']
    assert rt['seed']==lock['seed']==20260907 and rt['context_digest']==lock['context_digest']
    assert lock['evaluation_batches']==list(CP) and lock['checkpoint_storage']=='DISABLED_USER_DIRECTED_NO_W_M_TENSORS'
    assert term['batches']==100 and term['requests']==10000 and term['nonfinite']==0
    assert len(list(RAW.glob('B*/commit.json')))==100 and not (RAW/'failure.json').exists()
    assert not list(RAW.rglob('*.pt')), 'UNEXPECTED_PERSISTENT_TENSOR'
    inventory=[]
    for p in sorted(RAW.rglob('*')):
        if not p.is_file():continue
        assert not p.is_symlink();s=p.stat();v=r.ref(p);t=p.stat()
        assert (s.st_ino,s.st_size,s.st_mtime_ns)==(t.st_ino,t.st_size,t.st_mtime_ns)
        inventory.append(dict(v,relative_path=str(p.relative_to(RAW)),mtime_ns=s.st_mtime_ns))
    bypath={x['relative_path']:x for x in inventory}
    for x in term['manifest_members']:
        a=bypath[x['path']];assert a['bytes']==x['bytes'] and a['sha256']==x['sha256']
    r.table(out/'raw-inventory.csv',inventory)
    final=r.read(RAW/'B100/seen-full.json');frows={m:r.panel_rows(final,m) for m in MS}
    assert final['state']==term['actual_final_state']
    dataset=Path(lock['dataset_root'])/'counterfact.json'
    assert r.sha(dataset)=='3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1'
    records=r.read(dataset);assert len(records)==10000
    ids=[x['case_id'] for x in records];assert len(set(ids))==10000
    latest={}
    for rec in records:
        q=rec['requested_rewrite'];latest[q['subject'],q['relation_id']]=q['target_new']['str']
    active={rec['case_id']:('ACTIVE_TARGET' if latest[rec['requested_rewrite']['subject'],rec['requested_rewrite']['relation_id']]==rec['requested_rewrite']['target_new']['str'] else 'SUPERSEDED') for rec in records}
    ordinal={x:i for i,x in enumerate(ids)}
    current=[];seen_metrics=[];cost=[];links=[];actions=[];pairs=[];cohorts=[]
    atwrite={m:[] for m in MS};previous=rt['W0'];prev_rng=None;histories=defaultdict(list)
    for b in range(1,101):
        p=RAW/f'B{b:03d}';c=r.read(p/'commit.json');e=r.read(p/'entry.json');obs=r.read(p/'native-observation.json')
        assert c['batch']==b and c['next_ordinal']==100*b and c['next_batch']==b+1
        assert c['entry']==e['entry']==previous and c['context_digest']==e['context_digest']==lock['context_digest']
        assert e['case_ids']==ids[(b-1)*100:b*100]
        assert c['history_entries_in']==(b-1)*100 and c['history_entries_out']==b*100
        assert c['compute_z']==obs['compute_z']==100 and c['solve_calls']==obs['solve_calls']==5
        assert c['history_append_passes']==obs['history_append_passes']==1 and c['layer_history_updates']==5
        assert [(x['layer'],x['case_id']) for x in obs['z']]==[(8,x) for x in e['case_ids']]
        assert [x['layer'] for x in obs['keys']]==[4,5,6,7,8]*2 and len(obs['solves'])==5
        assert obs['extra_forward_count']==0 and obs['nonselected_pointer_version_exact']
        assert c['saved_weight_tensor_count']==c['saved_history_tensor_count']==c['evaluator_mutation']==c['nonfinite']==0
        if b>1:
            assert prev_rng==e['rng']
            links.append(dict(from_batch=b-1,to_batch=b,recorded_W_M_hash_match=True,RNG_match=True,context_match=True,
                              verification='STORED_METADATA_LINK_ONLY_NO_TENSOR_RELOAD'))
        cur=r.read(p/'current.json')
        assert cur['weight_state']==c['endpoint']['weights'] and cur['cache_sha256']==c['endpoint']['cache']
        assert cur['before_after_exact'] and cur['evaluator_controller_influence']==0
        for m,n in zip(MS,(100,200,1000)):
            rows=r.panel_rows(cur,m);assert len(rows)==n
            assert list(dict.fromkeys(x['case_id'] for x in rows))==e['case_ids']
            current.append(r.summary(rows,m,batch=b,population='CURRENT100'));atwrite[m]+=rows
        rw=r.read(p/'seen-rewrite.json');rr=r.panel_rows(rw,'RS');assert rw['state']==c['endpoint'] and len(rr)==100*b
        seen_metrics.append(r.summary(rr,'RS',batch=b,population='ACTUAL_ALL_SEEN_REWRITE'))
        for row in rr:histories[row['case_id']].append((b,r.success(row,'RS')))
        if b in CP:
            full=r.read(p/'seen-full.json');assert full['state']==c['endpoint'] and full['current_rows_reused']
            for m,n in zip(MS,(b*100,b*200,b*1000)):
                rows=r.panel_rows(full,m);assert len(rows)==n
                assert rows[-len(cur['metrics'][m]['rows']):]==cur['metrics'][m]['rows']
                assert list(dict.fromkeys(x['case_id'] for x in rows))==ids[:b*100]
                seen_metrics.append(r.summary(rows,m,batch=b,population='ACTUAL_FULL_SEEN'))
        cost.append(dict(batch=b,**{k:c[k] for k in ('compute_z','solve_calls','history_append_passes','layer_history_updates','edit_seconds','target_seconds','key_seconds','solve_seconds','evaluation_seconds','peak_gpu_allocated_bytes')},
                         key_calls=len(obs['keys']),actual_Adam='NOT_RECORDED',loss_evaluations='NOT_RECORDED',purewriter='NOT_SEPARATED',checkpoint_IO='NO_TENSOR_CP_USER_DIRECTED'))
        actions.extend(dict(batch=b,layer=k,**v) for k,v in c['layer_updates'].items())
        previous=c['endpoint'];prev_rng=c['rng']
    assert previous==term['actual_final_state'] and len(links)==99
    for m in MS:
        pairs.append(r.transition(atwrite[m],frows[m],m,comparison='ATWRITE_TO_W100',population='ALL_REQUESTED'))
        for cohort in range(1,101):
            a=[x for x in atwrite[m] if ordinal[x['case_id']]//100+1==cohort]
            f=[x for x in frows[m] if ordinal[x['case_id']]//100+1==cohort]
            cohorts.append(r.transition(a,f,m,comparison='ATWRITE_TO_W100',cohort=cohort,future_batches=100-cohort))
        for status in ('ACTIVE_TARGET','SUPERSEDED'):
            a=[x for x in atwrite[m] if active[x['case_id']]==status];f=[x for x in frows[m] if active[x['case_id']]==status]
            pairs.append(r.transition(a,f,m,comparison='ATWRITE_TO_W100',population=status))
        for earlier,cut in ((5,500),(10,1000),(50,5000)):
            a=r.panel_rows(r.read(RAW/f'B{earlier:03d}/seen-full.json'),m)
            f=[x for x in frows[m] if ordinal[x['case_id']]<cut]
            pairs.append(r.transition(a,f,m,comparison=f'W{earlier}_TO_W100',population=f'SAME_FIRST{cut}'))
    trajectories=[]
    for cohort in range(1,101):
        rr=[histories[x] for x in ids[(cohort-1)*100:cohort*100]]
        firstfailure=[next((b for b,ok in h if not ok),None) for h in rr]
        trajectories.append(dict(cohort=cohort,requests=100,atwrite_success=sum(h[0][1] for h in rr),final_success=sum(h[-1][1] for h in rr),
             ever_failed=sum(any(not ok for _,ok in h) for h in rr),
             loss_events=sum(sum(x[1] and not y[1] for x,y in zip(h,h[1:])) for h in rr),
             recovery_events=sum(sum(not x[1] and y[1] for x,y in zip(h,h[1:])) for h in rr),
             first_observed_failure_min=min((v for v in firstfailure if v is not None),default=None),
             first_observed_failure_max=max((v for v in firstfailure if v is not None),default=None),
             observation='POST_COMMIT_INTEGER_BATCH_ENDPOINTS_ONLY_RS'))
    allsummary=[r.summary(frows[m],m,arm='CAKE_NATIVE',state='ACTUAL_W100_FULL10000') for m in MS]
    r.table(out/'first-final-table.csv',allsummary)
    for name,rows in [('batch-current',current),('seen-prefix',seen_metrics),('cost-by-batch',cost),('state-links',links),('layer-actions',actions),('paired-transitions',pairs),('cohort-retention',cohorts),('rewrite-trajectories',trajectories)]:r.table(out/(name+'.csv'),rows)
    # Explicit label repair for legacy ORIGINAL which means BLUE, never native.
    family=[];baseline_pairs=[];inputs=[]
    sources={x['arm']:x for x in csvread(OLDREPORT/'source-config-compatibility.csv')}
    for method in ('AlphaEdit','MEMIT'):
        for row in csvread(OLDREPORT/(method+'-final-summary.csv')):
            x=dict(row);x['arm']=label(x['arm']);x['evidence']='SEALED_PRIOR_AGGREGATE_REUSE';family.append(x)
            src=sources.get(row['arm']);p=Path(src['raw_root'])/'B100/seen-full.json' if src else None
            if p is not None and p.exists():
                old=r.read(p);inputs.append(r.ref(p))
                for m in MS:baseline_pairs.append(r.transition(r.panel_rows(old,m),frows[m],m,comparison=label(row['arm'])+'_TO_CAKE',family=method))
        cr=dict(arm='CAKE_NATIVE',method=method,comparison_role='CAKE_REFERENCE',evidence='NEW_NLL_REDUCTION',scheduler_seconds=32194)
        for m in MS:
            s=next(x for x in allsummary if x['metric']==m)
            cr.update({m+'_numerator':s['numerator'],m+'_denominator':s['denominator'],m+'_rate':s['percent']/100})
        family.append(cr)
    r.table(out/'family-final-comparison.csv',family);r.table(out/'baseline-paired.csv',baseline_pairs)
    # Matched B10 reference only, not W100/10k mixed with EP W10/1k.
    one=r.read(RAW/'B010/seen-full.json')
    r.table(out/'CAKE-B10-first1000.csv',[r.summary(r.panel_rows(one,m),m,arm='CAKE_NATIVE',state='ACTUAL_W10_FIRST1000') for m in MS])
    totals={k:sum(x[k] for x in cost) for k in ('compute_z','solve_calls','history_append_passes','layer_history_updates','key_calls','edit_seconds','target_seconds','key_seconds','solve_seconds','evaluation_seconds')}
    return r.save(out/'summary.json',dict(task=TASK,status='SAVED_NLL_AND_METADATA_VERIFIED_NO_TENSOR_CP',rows=allsummary,
        batches=100,requests=10000,links=99,fullseen_points=list(CP),raw_files=len(inventory),raw_bytes=sum(x['bytes'] for x in inventory),
        totals=totals,allocated_GPU_seconds=32194,program_seconds=term['seconds'],load_setup_seconds=rt['load_setup_seconds'],
        peak_GPU_allocated_bytes=max(x['peak_gpu_allocated_bytes'] for x in cost),
        layer_path_lengths={k:sum(x['norm'] for x in actions if x['layer']==k) for k in sorted({x['layer'] for x in actions})},
        layer_net_update_norm='NOT_RECORDED_NO_WEIGHTS',actual_Adam='NOT_RECORDED',loss_evaluations='NOT_RECORDED',
        saved_tensor_CP=0,tensor_reload='NOT_AVAILABLE_USER_DIRECTED_NO_SAVE',GPU_continuation='NOT_TESTED',
        pure_writer='NOT_SEPARATED',baseline_inputs=inputs,lock=r.ref(ATTEMPT/'execution.lock.json'),runtime=r.ref(RAW/'runtime.json'),model_forwards=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['first','full']);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(first(a.output) if a.command=='first' else run(a.output)))
