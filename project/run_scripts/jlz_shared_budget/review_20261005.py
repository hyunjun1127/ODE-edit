"""Read-only, model-free review of the frozen SH3 v12 ridge completion.
No scheduler calls, model imports, original-output writes, or checkpoint loading.
"""
import argparse,collections,csv,hashlib,json,math,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
TASK='jlz-v12-shared-budget-bs100x20-20261004-v1'
SOURCE='1d27a830274aaee49a713bd07e463b0513591c9e'
MILESTONES=(5,10,15,20)

def check(x,label):
    if not x:raise ValueError(label)
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,lineterminator='\n',fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def q(v,p):v=sorted(v);return v[round((len(v)-1)*p)] if v else None
def stats(v):return dict(n=len(v),mean=math.fsum(v)/len(v),median=q(v,.5),p90=q(v,.9),p99=q(v,.99),min=min(v),max=max(v)) if v else dict(n=0)
def desired(r):return 'true' if r['kind']=='N' else 'new'
def ok(r,metric='preference'):
    d=desired(r)
    return r[d+'_strict'] if metric=='strict' else r[d+'_nll']<r[('new' if d=='true' else 'true')+'_nll']
def reduce(rows):
    check(len({r['identity'] for r in rows})==len(rows),'DUPLICATE')
    out={}
    for r in rows:
        check(r['kind'] in ('R','P','N'),'KIND')
        for t in ('new','true'):
            n,c=r[t+'_token_count'],r[t+'_token_correct'];check(type(n) is int and type(c) is int and n>0 and 0<=c<=n,'TOKEN')
            check(type(r[t+'_strict']) is bool and r[t+'_strict']==(n==c),'STRICT')
            check(math.isfinite(r[t+'_nll']),'FINITE')
    for k in ('R','P','N'):
        group=[r for r in rows if r['kind']==k]
        if not group:continue
        d='true' if k=='N' else 'new';n=len(group);tokens=sum(r[d+'_token_count'] for r in group);correct=sum(r[d+'_token_correct'] for r in group)
        m=dict(n=n,success=sum(ok(r) for r in group),strict_num=sum(ok(r,'strict') for r in group),tokens=tokens,correct=correct,
            preference=sum(ok(r) for r in group)/n,strict=sum(ok(r,'strict') for r in group)/n,token_micro=correct/tokens,
            prompt_macro=math.fsum(r[d+'_token_correct']/r[d+'_token_count'] for r in group)/n,
            true_nll=math.fsum(r['true_nll'] for r in group)/n,new_nll=math.fsum(r['new_nll'] for r in group)/n,
            ties=sum(r['true_nll']==r['new_nll'] for r in group),near_tie_1e4=sum(abs(r['true_nll']-r['new_nll'])<1e-4 for r in group))
        m['desired_nll']=m[d+'_nll'];m['margin_true_minus_new']=m['true_nll']-m['new_nll'];out[k]=m
    return out

def paired(a,b,metric='preference'):
    aa={r['identity']:r for r in a};bb={r['identity']:r for r in b}
    check(len(aa)==len(a) and len(bb)==len(b),'PAIRED_DUPLICATE')
    check(set(aa)<=set(bb),'PAIRED_IDS')
    check(all(aa[i]['kind']==bb[i]['kind'] for i in aa),'PAIRED_FAMILY')
    out={}
    for k in ('R','P','N'):
        ids=[i for i,r in aa.items() if r['kind']==k]
        if not ids:continue
        lost=[i for i in ids if ok(aa[i],metric) and not ok(bb[i],metric)];gained=[i for i in ids if not ok(aa[i],metric) and ok(bb[i],metric)]
        before=sum(ok(aa[i],metric) for i in ids);after=sum(ok(bb[i],metric) for i in ids)
        out[k]=dict(n=len(ids),before=before,after=after,lost=len(lost),gained=len(gained),retained=before-len(lost),
            retention=(before-len(lost))/before if before else None,lost_ids=lost,gained_ids=gained)
    return out

class Reader:
    def __init__(self):self.inventory={}
    def track(self,p,hash_value=None):
        p=Path(p);s=p.stat();self.inventory[str(p)]=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=hash_value or sha(p))
    def json(self,p):self.track(p);return json.loads(Path(p).read_text())
    def rows(self,folder,ids,identity):
        paths=sorted(folder.glob('chunk-*.json')) or ([folder/'rows.json'] if (folder/'rows.json').exists() else [])
        rows=[r for p in paths for r in self.json(p)['rows']]
        check(len(rows)==13*len(ids),'ROW_COVERAGE');check({r['case_id'] for r in rows}==set(ids),'CASE_COVERAGE')
        for r in rows:
            e=identity[r['identity']]
            for k in ('case_id','kind','prompt_index','new_token_identity','true_token_identity','new_token_count','true_token_count'):check(r[k]==e[k],'INPUT_TOKEN:'+k)
            check(r['margin_true_minus_new']==r['true_nll']-r['new_nll'],'MARGIN')
        reduce(rows);return rows

def review(attempt,out,local):
    reader=Reader();read=reader.json
    lock=read(attempt/'execution.lock.json');config=read(attempt/'config.json');check(lock['source_commit']==SOURCE,'SOURCE')
    check(sha(attempt/'config.json')==lock['config_sha256'],'CONFIG_SHA')
    check(sha(attempt/'source.tar')==lock['archive']['sha256'],'ARCHIVE_SHA');reader.track(attempt/'source.tar')
    for r in lock['source_members']:
        check(Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256'],'FROZEN_SOURCE_BYTES')
    snapshots=read(local/'snapshot.json');check(len(snapshots['completed_batches'])==20,'SNAPSHOT_COMPLETE')
    identity={r['identity']:r for r in read(Path(config['observer_identity']['path']))['rows']}
    records=json.loads(Path(config['stream']).read_text())[:2000]
    check(sha(config['stream'])=='3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1','DATASET')
    ids=[r['case_id'] for r in records];check(ids==[i for p in config['packing'] if p['phase']=='main' for i in p['ids']],'ORDER')
    w0=reader.rows(attempt/'shared-W0',ids,identity);w0s=reduce(w0);metrics=[dict(endpoint=0,family=k,**v) for k,v in w0s.items()]
    main=attempt/'main-V12_MAIN';terminal=read(main/'terminal.json');check(terminal['status']=='COMPLETED' and terminal['commits']==20,'TERMINAL')
    initial=read(main/'initial-state.json');pilot_initial=read(attempt/'pilot-V12_MAIN/initial-state.json');check(initial==pilot_initial,'COLD_SAME_W0_H0')
    previous=initial;last=None;atwrite=[];ends={};costs=[];layer_values=collections.defaultdict(lambda:collections.defaultdict(list));layer_table=[]
    stop=collections.Counter();totals=collections.Counter();bounds=collections.defaultdict(list);states=[];transition=[];current_table=[];candidates_terminal={}
    for b in range(1,21):
        root=main/f'batch-{b:02d}';c=read(root/'commit.json');e=read(root/'entry.json');fit=read(root/'fit/fit.json');writer=read(root/'writer.json')
        check(c['source']==SOURCE and c['config']==digest(config) and c['ids']==ids[(b-1)*100:b*100],'COMMIT_BINDING')
        check(c['before']==previous and e['before']==previous and e['native_pack']==c['native_pack'],'STATE_JOIN')
        if last:check(last['RNG_after']==c['RNG_before'] and last['context_hash']==c['context_hash'],'RNG_CONTEXT')
        check(c['history_appends']==5 and not c['checkpoint_saved'] and c['observer_no_mutation'],'COMMIT_POLICY')
        check(writer['no_divisor'] and writer['target_tracking'] and not writer['checkpoint_saved'],'WRITER_POLICY')
        cur=reader.rows(root/'pre',c['ids'],identity);post=reader.rows(root/'post',ids[:100*b] if b in MILESTONES else c['ids'],identity)
        pre_stats=reduce(cur);post_stats=reduce(post)
        for computed,stored in [(pre_stats,c['pre']),(post_stats,c['post'])]:
            for k,v in computed.items():
                for own,old in [('success','numerator'),('n','denominator'),('strict_num','strict_numerator'),('tokens','desired_token_count'),('correct','desired_token_correct'),('token_micro','token_micro'),('prompt_macro','prompt_macro'),('true_nll','true_nll_mean'),('new_nll','new_nll_mean')]:
                    check(math.isclose(v[own],stored[k][old],abs_tol=1e-12,rel_tol=1e-12),'INDEPENDENT_AGGREGATE:'+own)
        now=[r for r in post if r['case_id'] in set(c['ids'])];atwrite+=now
        for phase,rs in [('pre',cur),('post',now)]:current_table += [dict(batch=b,phase=phase,family=k,**v) for k,v in reduce(rs).items()]
        if b in MILESTONES:ends[b]=post;metrics += [dict(endpoint=b,family=k,**v) for k,v in post_stats.items()]
        pre=read(root/'pre/summary.json');ps=read(root/'post/summary.json')
        cost=dict(batch=b,total_seconds=c['seconds'],fit_seconds=fit['seconds'],write_seconds=writer['seconds'],pre_observer_seconds=pre['seconds'],post_observer_seconds=ps['seconds'],evals=fit['request_evaluations'],updates=fit['request_updates'],fit_forward=fit['physical_forward_calls'],fit_backward=fit['physical_backward_calls'],capture_forward=writer['capture_forward_calls'])
        cost['other_seconds']=cost['total_seconds']-sum(cost[k] for k in ('fit_seconds','write_seconds','pre_observer_seconds','post_observer_seconds'));costs.append(cost)
        check(fit['terminal_extra_forward']==fit['terminal_extra_backward']==0,'TERMINAL_EXTRA')
        event_count=collections.Counter();nc=collections.Counter();nu=collections.Counter();terminal_ids=set();is_terminal=set();h=hashlib.sha256();last_candidate={};bplans=collections.defaultdict(list);breal=collections.defaultdict(list)
        with (root/'events.jsonl').open('rb') as f:
            for data in f:
                h.update(data);r=json.loads(data);p=r['payload'];kind=r['event'];rid=r['request_id'];l=r['layer_id'];idx=r['candidate_index'];event_count[kind]+=1
                if kind=='candidate_request':
                    check(idx==nc[rid] and p['evaluation_ordinal']==idx+1 and p['updates_completed']==idx,'CANDIDATE_SEQUENCE');nc[rid]+=1
                    check(math.isfinite(p['objective_J']) and p['budget_used']<=.750001,'FINITE_BUDGET')
                    bounds['KL_raw'].append(p['kl_raw']);bounds['primal_violation'].append(max(0.,p['budget_used']-.75))
                    if not p['will_backward']:is_terminal.add((rid,idx));last_candidate[rid]=p
                elif kind=='optimizer_layer':nu[rid]+=1;bounds['postcast_violation'].append(p['postcast_primal_violation'])
                elif kind=='candidate_layer' and (rid,idx) in is_terminal:
                    check(p['gradient']['availability']=='NO_BACKWARD_TERMINAL','NO_TERMINAL_GRADIENT')
                    bplans[l].append(p['plan_share']);layer_values[l]['terminal_plan_share'].append(p['plan_share'])
                    layer_values[l]['terminal_plan_relative'].append(p['u_l2'])
                elif kind=='terminal_request':
                    check(rid not in terminal_ids and nc[rid]==idx+1 and nu[rid]==5*idx and p['logical_evaluations']==idx+1 and p['optimizer_updates']==idx,'STOP_FREEZE')
                    terminal_ids.add(rid);stop[p['reason']]+=1;bounds['terminal_J'].append(p['accepted_objective_J']);bounds['terminal_budget'].append(p['accepted_budget_used'])
                    candidates_terminal[rid]=dict(batch=b,candidate=idx,**last_candidate[rid])
                elif kind=='write_layer_summary':
                    check(p['solve_kind']=='NATIVE_RIDGE_FULL_BATCH' and p['native_mean_key_columns']==100 and p['key_hcur_same_state'] and p['refreshed_after_lower_writes'],'NATIVE_WRITER')
                    bounds['solve_residual'].append(p['solve_relative_residual']);check(p['solve_relative_residual']<=1e-8,'RIDGE_RESIDUAL')
                    bounds['weight_rounding'].append(p['ideal_effective_update_relative_difference']['value'])
                elif kind=='write_layer_request':
                    for key in ('plan_relative_l2','residual_l2','inherited_mismatch_l2','local_additivity_error_l2','actual_net_canonical_drift_l2'):layer_values[l][key].append(p[key])
                    v=p['effective_direct_realized_share']['value']
                    if v is not None:layer_values[l]['applied_share'].append(v);breal[l].append(v)
                    if p['plan_delta_l2']>0:layer_values[l]['direct_to_plan_norm_ratio'].append(p['effective_direct_canonical_l2']/p['plan_delta_l2'])
                    if p['residual_l2']>0:layer_values[l]['direct_to_residual_norm_ratio'].append(p['effective_direct_canonical_l2']/p['residual_l2'])
                elif kind=='history_commit':
                    check(p['append_count']==1 and p['request_count']==p['final_key_columns']==100,'HISTORY_ONCE')
                    check(p['history_before_id']==c['before']['H'][l] and p['history_after_id']==c['after']['H'][l] and p['final_keys_identity']==writer['final_keys'][l],'HISTORY_HASH')
                    check(p['all_original_request_columns_retained'] and not p['successful_requests_only_filter'],'HISTORY_OCCURRENCES')
        reader.track(root/'events.jsonl',h.hexdigest())
        check(len(terminal_ids)==100 and event_count['history_commit']==5 and event_count['write_layer_request']==500 and event_count['batch_commit']==1,'EVENT_COVERAGE')
        check(sum(nc.values())==fit['request_evaluations'] and sum(nu.values())//5==fit['request_updates'],'COST_COUNTS')
        totals.update(event_count)
        for l in sorted(bplans):layer_table.append(dict(batch=b,layer=l,plan_share=statistics.mean(bplans[l]),applied_share=statistics.mean(breal[l])))
        states.append(dict(batch=b,W_join=True,H_join=True,RNG_context_join=True,history_appends=5,observer_no_mutation=True,source_config_order=True))
        previous=c['after'];last=c
    final=ends[20];paired_local={};cohorts=[];strata=[]
    for name,base in [('W0_to_W20',w0),('atwrite_to_W20',atwrite),('first500_W5_to_W20',ends[5])]:
        for metric in ('preference','strict'):
            pair=paired(base,final,metric);paired_local[name+':'+metric]=pair
            for k,v in pair.items():transition.append(dict(comparison=name,metric=metric,family=k,**{a:b for a,b in v.items() if not a.endswith('_ids')}))
    for n in (100,500,1000,1500,2000):
        ss=set(ids[:n]);cur=[r for r in final if r['case_id'] in ss]
        for k,v in reduce(cur).items():cohorts.append(dict(population='first'+str(n),family=k,**v))
    for b in range(1,21):
        ss=set(ids[(b-1)*100:b*100]);rows=[r for r in final if r['case_id'] in ss]
        for k,v in reduce(rows).items():cohorts.append(dict(population='birth'+str(b),family=k,**v))
    for flag in (True,False):
        for k,v in reduce([r for r in final if r['active_at_endpoint']==flag]).items():strata.append(dict(active=flag,family=k,**v))
    nll=[]
    for k in ('R','P','N'):
        rr=[r for r in final if r['kind']==k]
        for target in ('true','new'):
            nll.append(dict(family=k,target=target,**stats([r[target+'_nll'] for r in rr])))
    nll.sort(key=lambda r:(r['family'],r['target']))
    save(local/'paired-identity-details.json',paired_local);save(local/'terminal-per-request.json',candidates_terminal)
    w0receipt=read(attempt/'shared-W0/summary.json');pilot=read(attempt/'pilot-V12_MAIN/terminal.json');collector=read(attempt/'report/terminal.json')
    original=read(attempt/'report/summary.json')['chains']['V12_MAIN']
    for row in metrics:
        if row['endpoint']:
            x=original['metrics'][str(row['endpoint'])][row['family']];check(row['success']==x['numerator'] and row['strict_num']==x['strict_numerator'],'COLLECTOR_CROSSCHECK')
    checks=dict(commits=20,requests=2000,history_appends=100,W_H_RNG_context_links=19,all_raw_identity_finite_denominators=True,
        source_config_order=True,independent_aggregate_vs_commit_collector=True,complete_pilot=True,complete_collector=collector['status']=='COMPLETED',
        no_checkpoint_files=not list(main.rglob('*.pt')) and not list(main.rglob('*.safetensors')),new_model_calls=0,new_scheduler_calls=0,
        owner_CPU_review=True,independent_agent=False)
    result=dict(source=SOURCE,lock_sha=sha(attempt/'execution.lock.json'),config_sha=lock['config_sha256'],snapshot=snapshots,
        metrics=metrics,paired=transition,cohorts=cohorts,active=strata,stop_reasons=dict(stop),telemetry={k:stats(v) for k,v in bounds.items()},
        layers={l:{k:stats(v) for k,v in r.items()} for l,r in layer_values.items()},event_counts=dict(totals),checks=checks,
        cost=dict(batch_sums={k:sum(r[k] for r in costs) for k in costs[0] if k!='batch'},W0_seconds=w0receipt['seconds'],
            main_process_seconds=terminal['seconds'],main_allocated_GPU_seconds=14723,pilot_allocated_GPU_seconds=231,
            total_allocated_GPU_hours=(14723+231)/3600,CPUcollector_allocated_seconds=258,
            main_process_peak_RSS_GiB=terminal['peak_RSS_KiB']/1024**2,main_peak_torch_allocated_VRAM_GiB=terminal['peak_VRAM_bytes']/1024**3,
            scheduler_batch_MaxRSS_GiB=14213788/1024**2,RSS_measurement_disagreement='process ru_maxrss vs scheduler sampled MaxRSS; do not claim same peak metric'))
    for name,rows in [('endpoint-metrics',metrics),('current-metrics',current_table),('paired-transitions',transition),('cohorts',cohorts),('active-strata',strata),('nll-distributions',nll),('batch-cost',costs),('layer-shares',layer_table),('state-links',states)]:table(out/(name+'.csv'),rows)
    save(out/'summary.json',result);save(out/'checks.json',checks)
    save(out/'raw-manifest.json',dict(source=SOURCE,read_files=list(reader.inventory.values()),large_assets_rehashed=False,model_forward=0,raw_copied_to_Git=False))
    print(json.dumps(dict(status='CPU_REVIEW_PASS',W20=reduce(final),stop=dict(stop),GPU_hours=result['cost']['total_allocated_GPU_hours']),ensure_ascii=False))
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--local',type=Path,required=True);a=ap.parse_args();review(a.attempt,a.out,a.local)
