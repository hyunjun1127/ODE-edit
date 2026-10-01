"""Independent CPU review of an immutable recall snapshot. No production imports.

Never queries a scheduler, runs a model, modifies raw files, or opens checkpoints.
The caller supplies fixed snapshot manifests and output directories. Prompt-level
transition IDs stay local; Git receives aggregates, compact telemetry and figures.
"""
import argparse, collections, csv, hashlib, itertools, json, math, os, resource, time
from pathlib import Path
import numpy as np

TAGS = ('RS', 'PS', 'NS')
MULT = dict(RS=1, PS=2, NS=10)
ROOT = '5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729'
DATA_SHA = '3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1'

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def digest(x, ascii=True):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=ascii).encode()).hexdigest()

def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)+'\n')

def csvout(p, rows):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n"); w.writeheader(); w.writerows(rows)

def summary(rows, tag):
    """Recompute both targets, tie-failure preference, token/macro/strict, tails."""
    assert rows and len({r['identity'] for r in rows}) == len(rows), 'DUPLICATE_OR_EMPTY'
    d = 'true' if tag == 'NS' else 'new'; n = len(rows)
    for r in rows:
        assert all(math.isfinite(r[k]) for k in ['new_nll', 'true_nll', 'margin']), 'NONFINITE'
        assert r['margin'] == r['true_nll']-r['new_nll'], 'MARGIN'
        good = r['true_nll'] < r['new_nll'] if tag == 'NS' else r['new_nll'] < r['true_nll']
        assert r['success'] == good, 'PREFERENCE'
        for t in ['new', 'true']:
            c, den = r[t+'_token_correct'], r[t+'_token_count']
            assert isinstance(c, int) and isinstance(den, int) and 0 <= c <= den and den > 0, 'TOKENS'
            assert r[t+'_strict'] == (c == den), 'STRICT'
    correct = sum(r[d+'_token_correct'] for r in rows); count = sum(r[d+'_token_count'] for r in rows)
    strict = sum(r[d+'_strict'] for r in rows); success = sum(r['success'] for r in rows)
    out = dict(numerator=success, denominator=n, rate=success/n, desired=d,
               tf_token_correct=correct, tf_token_count=count, tf_token_micro=correct/count,
               tf_prompt_macro=math.fsum(r[d+'_token_correct']/r[d+'_token_count'] for r in rows)/n,
               tf_strict_numerator=strict, tf_strict_denominator=n, tf_strict=strict/n,
               ties=sum(r['new_nll'] == r['true_nll'] for r in rows),
               bit_order_sha256=digest([(r['identity'], r['success']) for r in rows]))
    for t in ['new', 'true', 'desired']:
        vals = np.array([r[(d if t == 'desired' else t)+'_nll'] for r in rows])
        out[t+'_nll'] = float(vals.mean())
        for q in [.05, .5, .9, .95, .99, 1.]: out[f'{t}_nll_q{q:g}'] = float(np.quantile(vals, q))
        cutoff = float(np.quantile(vals, .95)); out[t+'_nll_tail_ge_q95_mean'] = float(vals[vals >= cutoff].mean())
    out['margin_true_minus_new'] = math.fsum(r['margin'] for r in rows)/n
    out['desired_advantage_nll'] = out['margin_true_minus_new'] * (-1 if tag == 'NS' else 1)
    return out

def compare_stored(s, m):
    for k, v in s.items():
        if k not in m: continue
        if isinstance(v, float): assert math.isclose(v, m[k], abs_tol=1e-11, rel_tol=1e-12), ('STORED_AGGREGATE', k, v, m[k])
        else: assert v == m[k], ('STORED_AGGREGATE', k, v, m[k])

def transitions(before, after, tag, metric='preference'):
    assert len(before) == len(after) and len({r['identity'] for r in before}) == len(before)
    b = {r['identity']: r for r in before}; a = {r['identity']: r for r in after}
    assert b.keys() == a.keys(), 'PAIRED_IDENTITY'
    d = 'true' if tag == 'NS' else 'new'
    good = lambda r: r['success'] if metric == 'preference' else r[d+'_strict']
    ids = {k: [] for k in ['lost', 'gained', 'retained', 'both_failed']}
    for k in b:
        x, y = good(b[k]), good(a[k]); label = 'retained' if x and y else 'lost' if x else 'gained' if y else 'both_failed'
        ids[label].append(k)
    out = {k: len(v) for k, v in ids.items()}; out.update(denominator=len(b), metric=metric)
    out['delta_pp'] = 100*(out['gained']-out['lost'])/len(b) if b else None
    out['retention_given_before_correct'] = out['retained']/(out['retained']+out['lost']) if out['retained']+out['lost'] else None
    return out, ids

def panel(records, n):
    if n == 0: return []
    groups = [[] for _ in range(4)]
    for i, r in enumerate(records[:n]):
        h = digest(dict(namespace='memit-hj-v2-past400', anchor_requests=n, ordered_root=ROOT,
                        occurrence_ordinal=i, request_hash=digest(r['requested_rewrite'], False)), False)
        groups[4*i//n].append((h, i))
    return sorted(i for g in groups for _, i in sorted(g)[:100])

def expected(records):
    ids = {t: {} for t in TAGS}
    for r in records:
        q = r['requested_rewrite']; case = int(r['case_id'])
        prompts = dict(RS=[q['prompt'].format(q['subject'])], PS=r['paraphrase_prompts'], NS=r['neighborhood_prompts'])
        for tag, ps in prompts.items():
            for j, p in enumerate(ps): ids[tag][case, j] = digest([case, j, p, q['target_new']['str'], q['target_true']['str']])
    return ids

def audit_calibration(cal):
    probes, rows = cal['probes'], cal['results']
    assert len(probes) == 96 and len(rows) == 32
    assert len({(p['anchor'], p['case_id'], p['point']) for p in probes}) == 96
    assert all(p['actual_FP64_forward_backward'] for p in probes)
    q95 = {str(a):float(np.quantile([p['normalized_error'] for p in probes if p['anchor'] == a], .95)) for a in [0, 1000]}
    tol = max(10*np.finfo(np.float32).eps, 5*max(q95.values()))
    assert tol == cal['tol'] and all(r['calls'] <= 400 and r['final_recomputed'] for r in rows)
    assert sum(r['calls'] for r in rows) == cal['SPG_calibration_calls'] <= 12800
    reasons=[]
    if any(r['status'] not in ['CONVERGED','POLICY_ZERO_STEP','STALLED_AT_PRECISION'] for r in rows):reasons.append('CENSORED_OR_FAILED')
    medians={}; stats=[]
    for a in [0,1000,'pooled']:
        rr=[r for r in rows if a=='pooled' or r['anchor']==a]
        for zero in [False,True]:
            vals=[r['calls'] for r in rr if zero or r['status']!='POLICY_ZERO_STEP'];m=float(np.median(vals)) if vals else None
            medians[f'{a}_include_zero_{zero}']=m
            if m is not None and m>200:reasons.append('MEDIAN_GT_200')
        stats.append(dict(anchor=a, requests=len(rr), statuses=dict(collections.Counter(r['status'] for r in rr)), calls=sum(r['calls'] for r in rr),
          accepted_steps=sum(r['accepted_steps'] for r in rr), rejected_trials=sum(r['backtracks'] for r in rr),
          oracle_seconds=sum(r['oracle_seconds'] for r in rr), token_forward=sum(r['token_forward'] for r in rr), token_backward=sum(r['token_backward'] for r in rr),
          native_final_loss_mean=float(np.mean([r['native_final_loss'] for r in rr])), SPG_loss_mean=float(np.mean([r['value'] for r in rr])),
          matched_adam_loss_mean=float(np.mean([r['matched_adam']['value'] for r in rr])), matched_adam_calls=sum(r['matched_adam']['calls'] for r in rr)))
    cap=max(25,math.ceil(1.25*max(float(np.quantile([r['calls'] for r in rows if r['anchor']==a],.95)) for a in [0,1000])))
    if cap>400:reasons.append('PRODUCTION_CAP_GT_400')
    if tol>1e-3:reasons.append('PRECISION_FLOOR_GT_1E-3')
    assert sorted(set(reasons))==cal['reasons'] and medians==cal['medians'] and cap==cal['diagnostic_proposed_cap']
    for r in rows:
        if r['status']=='CONVERGED':assert r['normalized_PG']<=tol
        if r['status']=='STALLED_AT_PRECISION':
            v=r['loss_window'];floor=10*np.finfo(np.float32).eps*max(max(map(abs,v)),np.finfo(np.float32).tiny)
            assert r['accepted_steps']>=6 and len(v)==6 and max(abs(v[i+1]-v[i]) for i in range(5))<=floor and v[0]-min(v)<=floor
    return dict(status=cal['status'], tol=tol, production_cap=cal['cap'], diagnostic_proposed_cap=cap, reasons=cal['reasons'], normalized_PG_error_Q95=q95, medians=medians, groups=stats,
                probe_FP64_calls=96, directional_extra_FP64_calls=8, SPG_calls=sum(r['calls'] for r in rows), native_fit_requests=32,
                native_fit_loop_count='NOT_SEPARATED', calibration_standalone_seconds='NOT_SEPARATED', prior_T0_origin_precision='NOT_ESTABLISHED_AT_ORIGIN')

def bootstrap(before, after, tag, records_by_case):
    """Optional design CI: fixed-trajectory fact clusters, no independent-order claim."""
    assert [r['identity'] for r in before]==[r['identity'] for r in after]
    groups=collections.defaultdict(lambda:[0.,0])
    for b,a in zip(before,after):
        q=records_by_case[b['case_id']]['requested_rewrite'];g=(q['subject'],q['relation_id'])
        groups[g][0]+=int(a['success'])-int(b['success']);groups[g][1]+=1
    vals=np.array(list(groups.values()));rng=np.random.default_rng(20260930);deltas=[]
    for _ in range(100):
        ii=rng.integers(0,len(vals),size=(100,len(vals)));s=vals[ii].sum(1);deltas.extend(100*s[:,0]/s[:,1])
    return dict(clusters=len(vals), replicates=10000, analysis_seed=20260930,
                CI95_lo_pp=float(np.quantile(deltas,.025)),CI95_hi_pp=float(np.quantile(deltas,.975)),
                conditioning='fixed trajectory; fact(subject,relation_id) clusters; no independent training/order inference')

def run(args):
    started=time.monotonic();snap=Path(args.snapshot);local=Path(args.local);out=Path(args.out);audit=Path(args.audit)
    out.mkdir(parents=True,exist_ok=True);audit.mkdir(parents=True,exist_ok=True);local.mkdir(parents=True,exist_ok=True)
    manifests=[read(x) for x in args.manifest];members=[m for x in manifests for m in x['files']]
    assert len({m['path'] for m in members})==len(members)
    for m in members:
        p=snap/m['path'];assert p.stat().st_size==m['bytes'] and sha(p)==m['sha256'],('SNAPSHOT_HASH',m['path'])
    assert sha(args.dataset)==DATA_SHA
    records=read(args.dataset);bycase={int(r['case_id']):r for r in records};ordinal={int(r['case_id']):i for i,r in enumerate(records)}
    identity=expected(records);cells=list(csv.DictReader(Path(args.cells).open()));assert len(cells)==28
    checks=collections.Counter();token_counts={};aggregates=[];coverage=[];costs=[];layers=[];chains=[];trajectories=[];cohorts=[];pairs=[];pair_ids={};endpoints={};atwrites={}
    def verify_result(p, ords=None):
        result=read(p)
        for tag,m in result['metrics'].items():
            rows=m['rows'];stats=summary(rows,tag);compare_stored(stats,m)
            for r in rows:
                assert r['identity']==identity[tag][r['case_id'],r['prompt_index']],('INPUT_IDENTITY',str(p))
                v=(r['new_token_count'],r['true_token_count']);key=(tag,r['identity'])
                if key in token_counts:assert token_counts[key]==v,'TOKEN_COUNT_CONTINUITY'
                token_counts[key]=v
            if ords is not None:
                want=[(int(records[i]['case_id']),j) for i in ords for j in range(MULT[tag])]
                assert [(r['case_id'],r['prompt_index']) for r in rows]==want,('ORDER_CARDINALITY',str(p),tag)
            checks['metric_groups']+=1;checks['row_occurrences']+=len(rows)
        if 'before_after_exact' in result:
            assert result['before_after_exact'] and result['evaluator_controller_influence']==0
            assert result.get('observer_context_rng_ledger_unchanged',True);checks['observer_files']+=1
        return result
    # Verify every immutable metric file, not only favorable endpoints.
    for m in members:
        p=snap/m['path']
        if p.name in ['current.json','all-seen.json','continuation.json','continuation-past.json','past-full.json','past-rewrite.json','past400.json','entry-past400.json','entry-next1000.json','W0-all10k.json','adapter-eval.json','native-eval.json']:
            verify_result(p)
    w0=verify_result(snap/'W0-all10k.json',list(range(10000)));w0maps={t:{r['identity']:r for r in w0['metrics'][t]['rows']} for t in TAGS}
    def addmetrics(name,pop,result,anchor,step):
        for t,m in result['metrics'].items():aggregates.append(dict(cell=name,population=pop,anchor=anchor,continuation=step,tag=t,**summary(m['rows'],t)))
    def addpair(label,b,a,tag,meta,ci=False):
        for metric in ['preference','TF_strict']:
            stats,ids=transitions(b,a,tag,metric);pairs.append(dict(comparison=label,tag=tag,**meta,**stats));pair_ids[f'{label}/{tag}/{metric}']=ids
            if ci and metric=='preference':pairs[-1].update(bootstrap(b,a,tag,bycase))
    for cell in cells:
        name=cell['cell_id'];cr=snap/'cells'/name;bs=int(cell['batch_size']);anchor=int(cell['anchor_requests']);cps=sorted(cr.glob('C*/commit.json'))
        term=read(cr/'terminal.json') if (cr/'terminal.json').exists() else {};lin=read(cr/'lineage.json') if (cr/'lineage.json').exists() else {}
        status=term.get('status','PARTIAL_COMMITTED_JOB_CANCELLED' if cps else 'NOT_REACHED_JOB_CANCELLED')
        if cell['z_solver']=='spg' and not cps:status='NOT_EXECUTED_JOB_CANCELLED;CALIBRATION_BLOCKED'
        if name=='main_001' and not cps:status='NO_1K_TRIGGER;FINAL_ALIAS_NOT_PUBLISHED'
        coverage.append(dict(cell=name,family=cell['family'],BS=bs,anchor=anchor,planned_requests=cell['total_requests'],commits=len(cps),last_cursor=read(cps[-1])['stop'] if cps else None,
                             completed_1k=bool(cps and read(cps[-1])['stop']-anchor>=1000),status=status,terminal_present=bool(term)))
        if not cps:continue
        prior=None;at={t:[] for t in TAGS};cost=collections.Counter();wr_calls=collections.Counter();timers=collections.Counter();nested_calls=collections.Counter();nested_timers=collections.Counter()
        for n,cp in enumerate(cps):
            c=read(cp);e=read(cp.parent/'entry.json');wr=read(cp.parent/'writer.json');start=anchor+n*bs;stop=start+bs
            assert (c['start'],c['stop'],c['requests'])==(start,stop,bs) and c['status']=='COMMITTED'
            assert e['identity']==c['entry_identity'] and e['request_ids']==[int(r['case_id']) for r in records[start:stop]]
            req=[dict(r['requested_rewrite'],case_id=int(r['case_id'])) for r in records[start:stop]]
            assert e['request_hashes']==[digest(r) for r in req]
            if prior is not None:assert c['entry_identity']==prior,'CHAIN_CONTINUITY'
            prior=c['endpoint_identity'];assert prior['ledger']==digest(list(range(stop)))
            assert c['history_append_layers']==5 and c['history_phase']=='POST_ALL_FIVE_WRITES'
            assert all(c[k] for k in ['prior_history_used','cache_c_returned_same_object','finite_W_H','observer_nonmutating'])
            assert wr['history_appends']==5 and [x['layer'] for x in wr['append']]==list(range(4,9))
            assert all(x['phase']=='POST_ALL_FIVE_WRITES' for x in wr['append'])
            if wr.get('keys'):assert [(x['layer'],x['phase']) for x in wr['keys']]==[(l,phase) for phase in ['pre_layer','post_all_layers'] for l in range(4,9)]
            current=verify_result(cp.parent/'current.json',list(range(start,stop)))
            assert current['weight_state']==prior['state']['weights'] and current['cache_sha256']==prior['state']['cache']
            for t in TAGS:at[t].extend(current['metrics'][t]['rows']);compare_stored(summary(current['metrics'][t]['rows'],t),c['current'][t])
            chains.append(dict(cell=name,start=start,stop=stop,state_link=True,request_order=True,history_appends=5,observer_nonmutation=True,
              H0_zero=all(v==0 for v in e['prior_history_norms']) if start==0 else None,prior_H_nonzero=all(v>0 for v in e['prior_history_norms']) if start else None))
            checks['commits']+=1;cost['requests']+=bs;cost['writes']+=1
            for k in ['write_seconds','observer_seconds','total_step_seconds','rollback_RAM_snapshot_seconds']:cost[k]+=c[k]
            cost['max_GPU_allocated_bytes']=max(cost['max_GPU_allocated_bytes'],c['peak_GPU_bytes'])
            wr_calls.update(wr.get('calls',{}));timers.update(wr.get('timers',{}))
            for shadow in ['shadow_divisor','shadow_joint']:
                if shadow in wr:nested_calls.update(wr[shadow]['calls']);nested_timers.update(wr[shadow]['timers'])
            if 'energy_scale' in wr:
                assert math.isclose(wr['energy_scale'],math.sqrt(wr['entry_Ejoint']/wr['entry_Ediv']),rel_tol=1e-13)
                assert wr['entry_A_common'] and wr['shadow_z_reuse_same_entry']
            for layer in wr['layers']:
                lr=dict(cell=name,start=start,layer=layer['layer'])
                for k,v in layer.items():
                    if isinstance(v,(int,float,bool,str)) or v is None:lr[k]=v
                if 'energy_scale' in wr:lr.update(energy_scale=wr['energy_scale'],entry_Ediv=wr['entry_Ediv'],entry_Ejoint=wr['entry_Ejoint'])
                layers.append(lr)
            for f,pop in [('all-seen.json','all_seen'),('continuation.json','continuation1000'),('past400.json','past400')]:
                p=cp.parent/f
                if p.exists():
                    ords=list(range(stop)) if f=='all-seen.json' else panel(records,anchor) if f=='past400.json' else list(range(anchor,stop))
                    rr=verify_result(p,ords)
                    for tag,m in rr['metrics'].items():trajectories.append(dict(cell=name,population=pop,step=stop-anchor,tag=tag,**summary(m['rows'],tag)))
        costs.append(dict(cell=name,**cost,**{'calls_'+k:v for k,v in wr_calls.items()},**{'timer_'+k:v for k,v in timers.items()},**{'shadow_calls_'+k:v for k,v in nested_calls.items()},**{'shadow_timer_'+k:v for k,v in nested_timers.items()}))
        end=cps[-1].parent;ep=verify_result(end/('all-seen.json' if bs==100 else 'continuation.json'),list(range(anchor,anchor+1000)))
        assert read(cps[-1])['stop']==anchor+1000,'REVIEW_SCOPE_NOT_1K'
        endpoints[name]=ep;atwrites[name]=at;addmetrics(name,'main_first1000' if bs==100 else 'continuation1000',ep,anchor,1000)
        if bs==10:
            assert lin['independent_CPU_clone'] and not lin['CP'] and lin['past_panel_ordinals']==panel(records,anchor)
            if term:assert term['endpoint']==read(cps[-1])['endpoint_identity']
        latest={}
        for r in records[:anchor+1000]:
            q=r['requested_rewrite'];latest[q['subject'],q['relation_id']]=q['target_new']['str']
        for tag,m in ep['metrics'].items():
            rows=m['rows'];base=[w0maps[tag][r['identity']] for r in rows]
            addpair(name+':atwrite_to_1k',at[tag],rows,tag,dict(kind='within_trajectory'))
            addpair(name+':W0_to_1k',base,rows,tag,dict(kind='W0_reference'))
            for label,subset in [('first100',lambda i:i<anchor+100),('first500',lambda i:i<anchor+500),('active',lambda i:records[i]['requested_rewrite']['target_new']['str']==latest[records[i]['requested_rewrite']['subject'],records[i]['requested_rewrite']['relation_id']]),('superseded',lambda i:records[i]['requested_rewrite']['target_new']['str']!=latest[records[i]['requested_rewrite']['subject'],records[i]['requested_rewrite']['relation_id']])]+[(f'birth{b+1}',lambda i,b=b:anchor+b*bs<=i<anchor+(b+1)*bs) for b in range(1000//bs)]:
                rr=[r for r in rows if subset(ordinal[r['case_id']])]
                if rr:cohorts.append(dict(cell=name,cohort=label,tag=tag,**summary(rr,tag)))
                else:cohorts.append(dict(cell=name,cohort=label,tag=tag,denominator=0))
            ids={r['identity'] for r in base if r['success']};aa=[r for r in rows if r['identity'] in ids];bb=[r for r in base if r['identity'] in ids]
            if bb:addpair(name+':W0_correct_retention',bb,aa,tag,dict(kind='conditional_W0_correct'))
        early=cr/f'C{anchor+500:05d}'/('all-seen.json' if bs==100 else 'continuation.json')
        for tag,m in read(early)['metrics'].items():
            ids={r['identity'] for r in m['rows']};after=[r for r in ep['metrics'][tag]['rows'] if r['identity'] in ids]
            addpair(name+':first500_at500_to1000',m['rows'],after,tag,dict(kind='first500_retention'))
        if anchor:
            past=verify_result(end/'past400.json',panel(records,anchor));addmetrics(name,'past400',past,anchor,1000)
            for tag,m in past['metrics'].items():addpair(name+':past400_entry_to_end',read(cr/'entry-past400.json')['metrics'][tag]['rows'],m['rows'],tag,dict(kind='past400'))
    # Matched writer methods share entry W/H/data/BS; history interventions share pre-refresh entry.
    groups=[[c for c in endpoints if c.startswith('writer_'+str(a)+'_')] for a in [0,1000]]+[['history_1000_stale','history_1000_forced_refresh']]
    for group in groups:
        for a,b in itertools.combinations(group,2):
            la=read(snap/'cells'/a/'lineage.json');lb=read(snap/'cells'/b/'lineage.json');assert la['fork_identity']==lb['fork_identity']
            for tag in TAGS:
                addpair(b+' minus '+a,endpoints[a]['metrics'][tag]['rows'],endpoints[b]['metrics'][tag]['rows'],tag,dict(kind='matched',same_entry=True,same_BS=10),ci=True)
                common={r['identity'] for r,s in zip(atwrites[a][tag],atwrites[b][tag]) if r['identity']==s['identity'] and r['success'] and s['success']}
                aa=[r for r in endpoints[a]['metrics'][tag]['rows'] if r['identity'] in common];bb=[r for r in endpoints[b]['metrics'][tag]['rows'] if r['identity'] in common]
                if aa:addpair(b+' minus '+a+':common_atwrite_success',aa,bb,tag,dict(kind='matched_common_atwrite'))
                if la['anchor']:
                    pa=read(snap/'cells'/a/'C02000/past400.json')['metrics'][tag]['rows'];pb=read(snap/'cells'/b/'C02000/past400.json')['metrics'][tag]['rows']
                    addpair(b+' minus '+a+':past400',pa,pb,tag,dict(kind='matched_past400'),ci=True)
    calibration=audit_calibration(read(snap/'calibration/lock.json'))
    cr=read(snap/'calibration/lock.json')['results']
    csvout(out/'calibration-requests.csv',[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in cr])
    save(out/'calibration-summary.json',calibration)
    for name,rs in [('coverage',coverage),('endpoint-metrics',aggregates),('trajectory',trajectories),('cohorts',cohorts),('paired',pairs),('cost',costs),('layer-telemetry',layers),('chain-checks',chains)]:csvout(out/(name+'.csv'),rs)
    save(local/'paired-ids.json',pair_ids)
    save(local/'identity-to-request.json',{r['identity']:dict(case_id=r['case_id'],prompt_index=r['prompt_index'],ordinal=ordinal[r['case_id']],family=t) for t in TAGS for r in w0['metrics'][t]['rows']})
    csvout(audit/'raw-inventory.csv',members)
    check=dict(status='PASS',owner_audit=True,independent_reviewer=False,checks=dict(checks),snapshot_files=len(members),snapshot_bytes=sum(x['bytes'] for x in members),
       distinct_prompt_family_identities=len(token_counts),stored_token_counts_consistent=True,actual_per_prompt_token_ids='NOT_RECORDED',
       scope='scalar/input identity/commit/source-backed assertions; no new model/forward/CP load',seconds=time.monotonic()-started,host_peak_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save(audit/'cpu-review-checks.json',check)
    save(local/'analysis-result.json',dict(checks=check,coverage=coverage,calibration=calibration))
    print(json.dumps(check))


def supplement(args):
    """Source-bound static audit, stored telemetry, figures and historical tables."""
    snap=Path(args.snapshot);local=Path(args.local);out=Path(args.out);audit=Path(args.audit)
    task=snap.parent.parent;workspace=Path(args.cells).resolve().parents[3]
    def rows(name):return list(csv.DictReader((out/name).open()))
    endpoints=rows('endpoint-metrics.csv');cost=rows('cost.csv');layers=rows('layer-telemetry.csv')
    # Keep per-layer-per-write telemetry local, publish 55 bounded aggregates.
    grouped=collections.defaultdict(list)
    for r in layers:grouped[r['cell'],r['layer']].append(r)
    if len(layers)>100:
        csvout(local/'layer-telemetry-per-write.csv',layers)
        compact=[]
        for (cell,layer),rr in grouped.items():
            agg=dict(cell=cell,layer=layer,writes=len(rr))
            for k in ['trace','rank','energy','ideal_energy','anchor_A_energy','D_norm','observed_norm','alpha','cosine','error','q','residual_norm','solve_residual','adj_identity_error','direct_comparison_error','ideal_to_FP32_update_relative','ideal_to_FP32_relative','gap','energy_scale']:
                v=[float(r[k]) for r in rr if r.get(k) not in ['',None]]
                if v:agg.update({k+'_mean':float(np.mean(v)),k+'_min':min(v),k+'_max':max(v),k+'_n':len(v)})
            agg['direct_fallbacks']=sum(r.get('direct_fallback')=='True' for r in rr);compact.append(agg)
        csvout(out/'layer-telemetry.csv',compact)
    else:compact=layers
    cohort_rows=rows('cohorts.csv')
    if any('desired_nll_q0.95' in x for x in cohort_rows):
        csvout(local/'cohorts-full.csv',cohort_rows)
        fields=['cell','cohort','tag','denominator','numerator','rate','tf_token_correct','tf_token_count','tf_token_micro','tf_prompt_macro','tf_strict_numerator','tf_strict','new_nll','true_nll','desired_nll','margin_true_minus_new','ties']
        csvout(out/'cohorts.csv',[{k:r.get(k,'') for k in fields} for r in cohort_rows])
    artifact_checks=[]
    for manifest in sorted((snap/'cells').glob('*/artifact-manifest.json')):
        checked=0
        for m in read(manifest)['members']:
            p=manifest.parent/m['path']
            if p.exists():assert p.stat().st_size==m['bytes'] and sha(p)==m['sha256'];checked+=1
        artifact_checks.append(dict(cell=manifest.parent.name,checked_members=checked,total_members=len(read(manifest)['members']),status='EXACT_SNAPSHOT_MEMBERS'))
    save(audit/'cell-artifact-checks.json',artifact_checks)
    bindings=[];source_members=[]
    for label,rel,want in [('P','attempt-v1/execution-full-sha.lock.json','1aa28cb09de1cae5c00824bdafb86eb2a7f83dda24422cf5cf11906a055d43c5'),('A_B_C_D','attempt-r2/execution-prerequisite-bound.lock.json','023d92c47d33e7d57b7eb4eee98d4f37ffe561c46b06e3ae5b6657d7102ed9f1')]:
        p=task/rel;assert sha(p)==want;lock=read(p)
        for m in lock['members']:
            path=Path(m['path'])
            if path.suffix in ['.py','.sh','.sbatch']:
                assert sha(path)==m['sha256'];source_members.append(dict(group=label,**{k:v for k,v in m.items() if k!='stat_identity'}))
        bindings.append(dict(group=label,lock_path=str(p),lock_sha256=want,source=lock['source_commit'],bindings=lock['bindings'],packages=lock['packages'],entrypoint=lock['entrypoint'],entrypoint_sha256=lock['entrypoint_sha256'],resource=lock['resource']))
    csvout(audit/'source-files.csv',source_members);save(audit/'execution-provenance.json',bindings)
    imports=read(snap/'groups/P/actual-import-closure.json')
    for m in imports['files']:assert sha(m['path'])==m['sha256']
    save(audit/'import-closure-check.json',dict(P_import_files=len(imports['files']),P_import_manifest_sha256=sha(snap/'groups/P/actual-import-closure.json'),P_status='EXACT',A_terminal_import_manifest='NOT_PUBLISHED',A_runtime_entrypoint=read(snap/'groups/A/runtime.json')['entrypoint_sha256'],A_source_lock='EXACT_SCOPED_SOURCE_HASH'))
    # Original 9 authorities: byte match authorizes reusing previous FULL_READ, not new model PASS.
    authority=workspace/'audits/global/2026-09-30-memit-hj-sh3-dispatch/input-manifest.json';am=read(authority)
    for m in am['members']:assert sha(workspace/m['path'])==m['sha256'] and (workspace/m['path']).stat().st_size==m['bytes']
    save(audit/'authority-read.json',dict(nonce='ODEEDIT-GH-SH3-MEMIT-HJ-1K-COMPLETED-REVIEW-20261002-R1',authority_commit='1482a5a667225b95e0b0af5f5654efe8cec774bc',envelope_sha256='7d6d3c1a79859223952534079698f8ea4c5e5cd2380d699e9a76eded22ba2d22',exact_members=am['members'],prior_full_read='audits/servers/server3/memit-hj-20260930-v2/full-read.json',read_status='FULL_READ_ENVELOPE; EXACT_PRIOR_FULL_READ_REUSED_9',analysis_source_sha256=sha(__file__)))
    # Telemetry checks are scalar assertions, not reconstruction of discarded matrices.
    for r in layers if len(layers)>100 else []:
        if r.get('solve_residual'):assert float(r['solve_residual'])<=1e-8
        if r.get('adj_identity_error'):assert float(r['adj_identity_error'])<=1e-8
        if r.get('min_eigenvalue'):assert float(r['min_eigenvalue'])>=-1e-10*max(1.,abs(float(r['max_eigenvalue'])))
    trigger=read(snap/'cells/main_000/trigger-01000.json');refresh=read(snap/'cells/history_1000_forced_refresh/forced-refresh.json')
    assert trigger['layers']==[] and trigger['nonintervening']
    for l,r in trigger['observations'].items():assert r['fired']==(int(l)>4 and r['drift_median']>r['dispersion_median'])
    assert refresh['occurrences']==1000 and refresh['membership']==digest(list(range(1000)))
    assert [r['layer'] for r in refresh['layers']]==[5,6,7,8] and all(r['probe_relative_error']<=1e-5 for r in refresh['layers'])
    assert refresh['W_output_unchanged'] and refresh['reference_reset'] and refresh['write_origin_preserved']
    save(out/'refresh-evidence.json',dict(automatic_trigger_at1000=trigger,forced_diagnostic=refresh,automatic_child_terminal_alias='NOT_PUBLISHED',later_triggers='NOT_OBSERVED'))
    cps=[dict(metadata_file=p.name,**read(p)) for p in sorted((snap/'temporary-checkpoints').glob('*.json'))]
    save(out/'checkpoint-metadata.json',dict(records=cps,original_snapshot_terminal=read(snap/'terminal.json'),review_CP_open_write_delete=0,physical_file_rehash='NOT_PERFORMED; metadata/tombstone/terminal only'))
    # Single saved scheduler observation, no subprocess here.
    scheduler=read(snap.parent/'scheduler-snapshot.json');lines=scheduler[0]['stdout'].strip().splitlines();account=list(csv.DictReader(lines,delimiter='|'))
    parents=[r for r in account if '.' not in r['JobIDRaw']];csvout(out/'job-snapshot.csv',parents)
    for r in parents:
        assert r['User']=='janghj' and r['JobName'].startswith('odeedit_memit_hj_')
    consumed=sum(int(r['ElapsedRaw']) for r in parents if 'gres/gpu=1' in r['AllocTRES'])
    totals={k:sum(float(r.get(k) or 0) for r in cost) for k in ['requests','writes','write_seconds','observer_seconds','total_step_seconds','rollback_RAM_snapshot_seconds']}
    save(out/'cost-summary.json',dict(parent_allocated_GPU_seconds=consumed,parent_allocated_GPU_hours=consumed/3600,P_GPU_seconds=int(next(r['ElapsedRaw'] for r in parents if r['JobIDRaw']=='56007')),A_GPU_seconds=int(next(r['ElapsedRaw'] for r in parents if r['JobIDRaw']=='56033')),
        committed_science=totals,calibration=read(out/'calibration-summary.json'),forced_refresh_seconds=refresh['seconds'],
        nested_timers='write includes native/key/solve/shadow; Gram includes direct solve; do not add phase timers to wall',
        unseparated_A_seconds=int(next(r['ElapsedRaw'] for r in parents if r['JobIDRaw']=='56033'))-totals['total_step_seconds']-totals['rollback_RAM_snapshot_seconds'],
        unseparated_contains='entry and W0/model loads, calibration native/FP64/SPG/Adam, sample/trigger, entry observations, CP, refresh, RAM fork/restore, IO and interruption; not pure overhead',
        energy_materialization_counter='final manual materialization key/forward calls NOT_SEPARATELY_COUNTED; two shadow phase counters available',
        observer_tokens='P stored full-model counters; A group terminal absent, observer full token total NOT_RECORDED',
        peak_committed_GPU_bytes=max(float(r['max_GPU_allocated_bytes']) for r in cost),review_GPU_seconds=0))
    # Historical aggregates, separate from same-run matched BS10 comparisons.
    historical=[];histfiles=[]
    base=workspace/'experiment-reports/servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1'
    for file,arms in [('MEMIT-cumulative-metrics.csv',['BASE_MEMIT']),('AlphaEdit-cumulative-metrics.csv',['BASE_ALPHAEDIT','AlphaEdit_ORIGINAL'])]:
        p=base/file;histfiles.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
        for r in csv.DictReader(p.open()):
            if r['batch']!='10' or r['arm'] not in arms or r['scope']!='CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS':continue
            tag=r['metric'];d='true' if tag=='NS' else 'new';assert int(r['denominator'])==1000*MULT[tag]
            historical.append(dict(method=r['arm'],tag=tag,numerator=int(r['numerator']),denominator=int(r['denominator']),rate=float(r['rate']),
               tf_token_micro=int(r[d+'_token_correct'])/int(r[d+'_token_den']),tf_strict=int(r[d+'_strict_num'])/int(r[d+'_strict_den']),
               tf_prompt_macro='NOT_RECORDED_IN_REUSED_TABLE',new_nll=float(r['new_nll_prompt_mean']),true_nll=float(r['true_nll_prompt_mean']),
               evidence='prior local reviewed aggregate; per-ID paired unavailable',hardware='S4 RTX PRO 6000 Blackwell',layers='4,8' if r['arm']=='AlphaEdit_ORIGINAL' else '4,5,6,7,8',
               blue=r['arm']=='AlphaEdit_ORIGINAL'))
    hp=snap.parent/'supplemental-inputs/historical54007/seen-full.json';old=read(hp);new=read(snap/'cells/main_000/C01000/all-seen.json');histpairs=[];ids={}
    for tag in TAGS:
        a=old['metrics'][tag]['rows'];b=new['metrics'][tag]['rows'];s=summary(a,tag);compare_stored(s,old['metrics'][tag]);assert [r['identity'] for r in a]==[r['identity'] for r in b]
        historical.append(dict(method='MEMIT_H_54007',tag=tag,**s,evidence='local completed B010 raw; fresh prior trajectory',hardware='S3 H200 NVL',layers='4,5,6,7,8',blue=False))
        for metric in ['preference','TF_strict']:
            t,i=transitions(a,b,tag,metric);histpairs.append(dict(comparison='HJ_main000_minus_historical54007',tag=tag,**t,max_abs_new_nll_difference=max(abs(x['new_nll']-y['new_nll']) for x,y in zip(a,b)),max_abs_true_nll_difference=max(abs(x['true_nll']-y['true_nll']) for x,y in zip(a,b))));ids[tag+'/'+metric]=i
    csvout(out/'historical-1k.csv',historical);csvout(out/'historical-paired.csv',histpairs);save(local/'historical-paired-ids.json',ids)
    save(audit/'historical-inputs.json',dict(existing_local_aggregate_files=histfiles,local_selected_raw_manifest=read(snap.parent/'supplemental-inputs/manifest.json'),model_evaluation=0,remote_raw_transfer=0,
        compatibility_reference='experiment-reports/servers/server3/memit-history-fixed10k-20260928-v1/completion-review-r1/four-method-comparison-manifest.json'))
    # Figures: standard CPU plotting, not AI-created images.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'svg.hashsalt':'memit-hj-1k-review-20261002'})
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    groups=[('W0 writer BS10',[r for r in endpoints if r['cell'].startswith('writer_0_')]),('Anchor1k + next1k BS10',[r for r in endpoints if r['cell'].startswith('writer_1000_') and r['population']=='continuation1000']),('Anchor1k history BS10',[r for r in endpoints if r['cell'].startswith('history_1000_') and r['population']=='continuation1000'])]
    for ax,(title,rr) in zip(axes,groups):
        names=list(dict.fromkeys(r['cell'] for r in rr));xx=np.arange(len(names))
        for j,tag in enumerate(TAGS):
            v=[100*float(next(r['rate'] for r in rr if r['cell']==n and r['tag']==tag)) for n in names];ax.bar(xx+(j-1)*.23,v,.23,label=tag)
        ax.set_xticks(xx,[n.split('_',2)[2].replace('_','\n') for n in names],fontsize=8);ax.set_ylim(0,115);ax.set_title(title);ax.set_ylabel('NLL preference (%)');ax.legend(ncol=3,loc='upper center',fontsize=8)
    fig.tight_layout();fig.savefig(out/'matched-endpoints.png',dpi=180);fig.savefig(out/'matched-endpoints.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5));tr=rows('trajectory.csv')
    for ax,tag in zip(axes,TAGS):
        for name in [r['cell'] for r in endpoints if r['cell'].startswith('writer_0_') and r['tag']=='RS']:
            rr=[r for r in tr if r['cell']==name and r['tag']==tag and r['population']=='continuation1000'];ax.plot([int(r['step']) for r in rr],[100*float(r['rate']) for r in rr],marker='o',label=name[9:])
        ax.set_title('W0 BS10 '+tag);ax.set_xlabel('Committed continuation requests');ax.set_ylabel('NLL preference (%)');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'writer-W0-trajectory.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.4));cal=read(snap/'calibration/lock.json')
    for a,color in [(0,'#3977b5'),(1000,'#e38b35')]:
        rr=[r for r in cal['results'] if r['anchor']==a];axes[0].scatter(range(len(rr)),[r['calls'] for r in rr],label=f'anchor {a}',alpha=.7,color=color);axes[1].scatter([r['calls'] for r in rr],[r['normalized_PG'] for r in rr],label=f'anchor {a}',alpha=.7,color=color)
    axes[0].axhline(400,color='black',linestyle='--');axes[0].set(xlabel='Calibration request index',ylabel='Total SPG oracle calls');axes[1].axhline(cal['tol'],color='black',linestyle='--',label='calibrated tolerance');axes[1].set(xlabel='Total SPG calls',ylabel='Return-point normalized PG',yscale='log')
    for ax in axes:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'calibration-blocked.png',dpi=180);plt.close(fig)
    for svg in out.glob('*.svg'):
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print('Supplemental source/telemetry/history/figures completed; no scheduler/model operations.')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['snapshot','local','out','audit','dataset','cells']:p.add_argument('--'+key,required=True)
    p.add_argument('--manifest',action='append',required=True);p.add_argument('--supplement-only',action='store_true')
    args=p.parse_args()
    if not args.supplement_only:run(args)
    supplement(args)
