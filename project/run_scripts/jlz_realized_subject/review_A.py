"""Independent, CPU-only A500 review of a sealed snapshot; no job/model access.

This module deliberately does not import the production observer/reducer or torch.
Per-case transition IDs stay in --local-out. Published products contain aggregates.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import unicodedata
import numpy as np

FAMILIES = ('R', 'P', 'N')
METRIC = dict(R='RS', P='PS', N='NS')
LAYERS = tuple(map(str, range(4, 9)))
DATA_SHA = '3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1'
SOURCE = 'c2d5fb107a0435491d8b4705b43b75f6177bbb5c'
LOCK_SHA = 'cf03a94ccf873d93f36bdfe3c0a378998bc2340ffbb4cda4d5b18914c31cbe23'
TASK = 'jlz-realized-subject-v10-tprime-bs100x5-sh3-20261003-v1'

def require(ok, msg):
    if not ok: raise ValueError(msg)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def digest(v):
    return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()

def read(p): return json.loads(Path(p).read_text())

def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def member(p):
    p=Path(p);return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))

def table(p, rows):
    if not rows: return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)

def finite_tree(v):
    if isinstance(v,float): require(math.isfinite(v),'NONFINITE')
    elif isinstance(v,dict):
        for x in v.values(): finite_tree(x)
    elif isinstance(v,list):
        for x in v: finite_tree(x)

def success(r, strict=False):
    if strict: return r[('true' if r['kind']=='N' else 'new')+'_strict']
    return r['true_nll']<r['new_nll'] if r['kind']=='N' else r['new_nll']<r['true_nll']

def distribution(values):
    v=np.asarray(values,dtype=np.float64);require(v.size>0 and np.isfinite(v).all(),'DISTRIBUTION')
    return dict(mean=float(v.mean()),median=float(np.quantile(v,.5)),p90=float(np.quantile(v,.9)),
                p95=float(np.quantile(v,.95)),p99=float(np.quantile(v,.99)),min=float(v.min()),max=float(v.max()))

def reduce_rows(rows):
    require(len(set(r['identity'] for r in rows))==len(rows),'DUPLICATE_ID')
    for r in rows:
        require(r['kind'] in FAMILIES,'FAMILY')
        for target in ('true','new'):
            require(type(r[target+'_nll']) in (int,float) and math.isfinite(r[target+'_nll']),'NLL_FINITE')
            n,c=r[target+'_token_count'],r[target+'_token_correct']
            require(type(n) is int and type(c) is int and n>0 and 0<=c<=n,'TOKEN_COUNTS')
            require(type(r[target+'_strict']) is bool and r[target+'_strict']==(n==c),'STRICT_COUNTS')
        require(math.isclose(r['margin_true_minus_new'],r['true_nll']-r['new_nll'],abs_tol=1e-10),'MARGIN')
    out={}
    for k in FAMILIES:
        group=[r for r in rows if r['kind']==k]
        if not group: continue
        d='true' if k=='N' else 'new'; n=len(group); c=sum(r[d+'_token_correct'] for r in group);t=sum(r[d+'_token_count'] for r in group)
        out[k]=dict(denominator=n,numerator=sum(success(r) for r in group),rate=sum(success(r) for r in group)/n,
            ties=sum(r['new_nll']==r['true_nll'] for r in group),desired_token_correct=c,desired_token_count=t,token_micro=c/t,
            prompt_macro=statistics.mean(r[d+'_token_correct']/r[d+'_token_count'] for r in group),
            strict_numerator=sum(success(r,True) for r in group),strict_denominator=n,
            true_nll_mean=statistics.mean(r['true_nll'] for r in group),new_nll_mean=statistics.mean(r['new_nll'] for r in group),
            desired_nll_mean=statistics.mean(r[d+'_nll'] for r in group),
            desired_margin_mean=statistics.mean((r['new_nll']-r['true_nll']) if k=='N' else (r['true_nll']-r['new_nll']) for r in group))
    return out

def validate_identity(rows, reference, ids):
    order={cid:i for i,cid in enumerate(ids)}
    expected=sorted([r for r in reference if r['case_id'] in order],key=lambda r:(order[r['case_id']],FAMILIES.index(r['kind']),r['prompt_index']))
    require([r['identity'] for r in rows]==[r['identity'] for r in expected],'ORDER_OR_IDENTITY')
    fields=('case_id','kind','prompt_index','true_token_identity','new_token_identity','true_token_count','new_token_count')
    require(all(all(a[k]==b[k] for k in fields) for a,b in zip(rows,expected)),'TARGET_TOKEN_IDENTITY')
    require(len(rows)==13*len(ids),'CARDINALITY')
    for cid in ids:
        counts=collections.Counter(r['kind'] for r in rows if r['case_id']==cid)
        require(counts==dict(R=1,P=2,N=10),'PER_REQUEST_CARDINALITY')

def pair(before, after, tag):
    b={r['identity']:r for r in before};a={r['identity']:r for r in after}
    require(len(b)==len(before) and len(a)==len(after) and a.keys()==b.keys(),'PAIR_IDENTITY')
    rows=[];detail={}
    for family in FAMILIES:
        ids=[i for i in b if b[i]['kind']==family]
        if not ids: continue
        for strict in (False,True):
            kind='TF_STRICT' if strict else 'PREFERENCE'
            groups={name:[] for name in ('lost','gained','retained','both_failed')}
            for i in ids:
                x,y=success(b[i],strict),success(a[i],strict)
                name='retained' if x and y else 'lost' if x else 'gained' if y else 'both_failed'
                groups[name].append(dict(identity=i,case_id=b[i]['case_id'],prompt_index=b[i]['prompt_index']))
            counts={k:len(v) for k,v in groups.items()};bs=counts['lost']+counts['retained'];afs=counts['gained']+counts['retained']
            rows.append(dict(comparison=tag,family=family,definition=kind,denominator=len(ids),before=bs,after=afs,
                **counts,retention_given_before_success=counts['retained']/bs if bs else None))
            detail[tag+'/'+family+'/'+kind]=groups
    return rows,detail

def active_ids(records):
    latest={}
    def claim(r):
        q=r['requested_rewrite'];return (unicodedata.normalize('NFC',' '.join(q['subject'].split())),q['relation_id'])
    def target(r):
        x=r['requested_rewrite']['target_new'];return x.get('id',x['str'])
    for r in records:latest[claim(r)]=target(r)
    return {r['case_id'] for r in records if target(r)==latest[claim(r)]}

def baselines(repo):
    hist=repo/'experiment-reports/servers/server3'/TASK/'historical'
    cake=repo/'experiment-reports/servers/server4/cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v2'
    paths=[hist/'historical-W5.csv',hist/'historical-manifest.json',cake/'seen-prefix.csv',cake/'compatibility.json',cake/'diagnostic-report-ko.md']
    rows=[]
    for r in csv.DictReader(paths[0].open()):
        if r['method'] not in ('MEMIT-H','AlphaEdit','AlphaEdit-BLUE (L4+L8)'):continue
        rows.append(dict(method=r['method'],family={'RS':'R','PS':'P','NS':'N'}[r['metric']],endpoint='W5_FIRST500',
            numerator=int(r['numerator']),denominator=int(r['denominator']),rate=float(r['rate']),
            desired_token_correct=int(r['TF_correct']),desired_token_count=int(r['TF_tokens']),token_micro=float(r['TF_token_micro']),
            strict_numerator=int(r['TF_strict_numerator']),strict_denominator=int(r['TF_strict_denominator']),
            prompt_macro=r['TF_prompt_macro'],new_nll_mean=float(r['new_nll']),true_nll_mean=float(r['true_nll']),
            source=str(paths[0].relative_to(repo)),comparison_class='HISTORICAL_AGGREGATE'))
    for r in csv.DictReader(paths[2].open()):
        if r['batch']!='5' or r['population']!='ACTUAL_FULL_SEEN':continue
        n=int(r['denominator']);c=int(r['desired_token_correct']);t=int(r['desired_token_d'])
        rows.append(dict(method='CAKE',family={'RS':'R','PS':'P','NS':'N'}[r['metric']],endpoint='W5_FIRST500',
            numerator=int(r['numerator']),denominator=n,rate=int(r['numerator'])/n,
            desired_token_correct=c,desired_token_count=t,token_micro=c/t,strict_numerator=int(r['desired_strict_n']),
            strict_denominator=n,prompt_macro='NOT_RECORDED_IN_SELECTED_AGGREGATE',new_nll_mean=float(r['new_nll_mean']),
            true_nll_mean=float(r['true_nll_mean']),source=str(paths[2].relative_to(repo)),comparison_class='HISTORICAL_AGGREGATE'))
    require(len(rows)==12 and len({(r['method'],r['family']) for r in rows})==12,'BASELINE_COVERAGE')
    for r in rows:
        require(r['denominator']==dict(R=500,P=1000,N=5000)[r['family']],'BASELINE_ENDPOINT')
        require(math.isclose(r['rate'],r['numerator']/r['denominator'],abs_tol=1e-12),'BASELINE_ARITHMETIC')
        r['strict_rate']=r['strict_numerator']/r['strict_denominator']
        r['desired_nll_mean']=r['true_nll_mean'] if r['family']=='N' else r['new_nll_mean']
    return rows,[member(p) for p in paths]

def review(snapshot,repo,out,local):
    out.mkdir(parents=True,exist_ok=True);local.mkdir(parents=True,exist_ok=True)
    inventory=read(snapshot/'snapshot-manifest.json')
    for row in inventory['members']:
        p=Path(row['snapshot']);require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'SNAPSHOT_SHA')
    lock=read(snapshot/'execution.lock.json');config=read(snapshot/'config.json');root=snapshot/'main-A'
    require(sha(snapshot/'execution.lock.json')==LOCK_SHA and lock['source_commit']==SOURCE,'EXECUTION_SOURCE')
    require(sha(snapshot/'config.json')==lock['config_sha256'],'CONFIG_SHA')
    imports=read(root/'actual-imports.json');source_check=[]
    for row in lock['source_members']+lock['runtime_sources']+lock['native_reference']+list(imports.values()):
        p=Path(row['path']);require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'FROZEN_SOURCE_SHA')
        source_check.append(dict(**member(p),check='PASS'))
    table(out/'source-verification.csv',source_check)
    require(sha(config['stream'])==DATA_SHA and sha(config['W0_reuse']['path'])==config['W0_reuse']['sha256'],'INPUT_SHA')
    data=read(config['stream'])[:500];ids=[r['case_id'] for r in data]
    require(digest(ids)=='0be7d88c759e7f65a690514035f40a18c5c19d8591ddd33f97b2fa6187b64a95','FIRST500_ORDER')
    reference=read(config['W0_reuse']['path'])['rows'];require(len(reference)==6500,'W0_COVERAGE')
    final=None;previous=None;atwrite=[];metrics=[];cost=[];states=[];losses=[];layers=[];components=[];decomp=[];budget=[];allrows={};nlldist=[]
    for b in range(1,6):
        bp=root/f'batch-{b:02d}';cp=read(bp/'commit.json');e=read(bp/'entry.json');inp=read(bp/'input.json');term=read(bp/'fit/terminal-actual.json')
        expected=ids[(b-1)*100:b*100]
        require(cp['ids']==inp['ids']==expected and inp['identity']==config['packed_native'][b-1]['identity']==e['input'],'NATIVE_INPUT')
        require(cp['source']==SOURCE and cp['config']==digest(config) and cp['candidates']==25 and cp['updates']==24,'COMMIT_SOURCE_BUDGET')
        require(e['before']==cp['before'] and e['aux']==cp['before_aux'],'ENTRY_COMMIT')
        if previous:require(cp['before']==previous['after'] and cp['before_aux']==previous['after_aux'],'STATE_CONTINUITY')
        else:
            ini=read(root/'initial-state.json');require(ini['state']==cp['before'] and ini['aux']==cp['before_aux'],'COLD_ENTRY')
        h=cp['history'];require(h['accepted_weight_copy_exact'] and h['history_appends']==5 and h['after']==cp['after'],'HISTORY')
        require(not cp['checkpoint_saved'] and not h['checkpoint_saved'] and cp['exact_resume']=='NOT_AVAILABLE','NOCP')
        require(not term['weights_rebuilt'] and not term['fit_feedback'],'TERMINAL_NO_REBUILD')
        require(h['history_key_hash']=={l:term['key_comparison'][l]['terminal_key_sha'] for l in LAYERS},'HISTORY_TERMINAL_KEYS')
        candidates=[read(x) for x in sorted((bp/'fit').glob('candidate-*.json'))]
        require([r['candidate'] for r in candidates]==list(range(1,26)),'CANDIDATE_COVERAGE')
        finite_tree(term);finite_tree(candidates)
        for c in candidates:
            i=c['candidate'];require(c['actual_B']==100 and c['Adam_updates_after']==min(i,24) and c['optimizer_builder_bridges']==1,'OPTIMIZER')
            require(c['terminal_backward']==(i==25) and c['terminal_update'] is False and c['component_builder_backwards']==(4 if i in (2,9,25) else 0),'BACKWARD_BUDGET')
            require(c['coefficients']==dict(norm=.5,allocation=.1,kl=.0625),'COEFFICIENT')
            require(not any(c[k] for k in ('clamp','pulse','E_penalty','replay','R_or_P_saved','checkpoint_saved')),'EXCLUDED_FEATURE')
            require(c['upper_solve_count']==4 and c['first_solve_this_candidate']==(i==1),'CACHE_BUDGET')
            require(all(c['solve'][l]['grad_K_enabled'] and c['solve'][l]['grad_P_enabled'] for l in LAYERS[1:]),'UPPER_TOTAL_GRADIENT_FLAGS')
            require(max(x['relative_residual'] for x in c['solve'].values())<=1e-8,'SOLVE_RESIDUAL')
            require(math.isclose(c['total_mean'],sum(c['losses'][k]*dict(nll=1,kl=.0625,norm=.5,allocation=.1)[k] for k in c['losses']),rel_tol=1e-6,abs_tol=1e-8),'LOSS_SUM')
            require(math.isclose(c['losses']['allocation'],sum(x['c'] for x in c['energy'].values()),rel_tol=1e-8,abs_tol=1e-10),'A_ALLOCATION_L1')
            losses.append(dict(batch=b,candidate=i,**c['losses'],total=c['total_mean'],seconds=c['seconds'],native_seconds_nested=c['native_seconds'],builder_seconds_nested=c['builder_seconds']))
            if c['gradient']['components']:
                require(c['gradient']['weighted_sum_RMS']<=c['gradient']['weighted_sum_limit'],'GRADIENT_SUM')
                for component,g in c['gradient']['components'].items():
                    for coordinate in ('R','q'):
                        raw=g[coordinate+'_raw'];weighted=g[coordinate+'_weighted']
                        require(math.isclose(weighted['norm'],g['coefficient']*raw['norm'],rel_tol=2e-6,abs_tol=1e-9),'GRADIENT_COEFFICIENT')
                        components.append(dict(batch=b,candidate=i,component=component,coordinate=coordinate,coefficient=g['coefficient'],raw_norm=raw['norm'],weighted_norm=weighted['norm'],cosine_total=weighted['cosine_total'],radial=weighted['radial'],weighted_sum_RMS=c['gradient']['weighted_sum_RMS'],weighted_sum_limit=c['gradient']['weighted_sum_limit']))
            for l in LAYERS:
                z=c['layer'][l]
                for family in ('rewrite','kl'):
                    radius=z['radius'][family]
                    layers.append(dict(batch=b,candidate=i,layer=l,family=family,**{k:radius[k] for k in ('count','exceeds_075','max','mean')},Q=c['energy'][l]['Q'],c=c['energy'][l]['c'],mean_share=z['mean_share'],context_share=z['context_share']))
        obs=root/f'observe-W{b:02d}';saved=read(obs/'summary.json');rows=[]
        for p in sorted(obs.glob('chunk-*.json')):
            d=read(p);require(d['state']==cp['after'] and d['optimizer_feedback'] is False,'CHUNK_STATE');rows+=d['rows']
        validate_identity(rows,reference,ids if b==5 else expected)
        require(all(r['endpoint']==b for r in rows),'ENDPOINT')
        active=active_ids(data[:b*100]);require(all(r['active_at_endpoint']==(r['case_id'] in active) for r in rows),'ACTIVE_IDENTITY')
        reduced=reduce_rows(rows)
        require(saved['state']==cp['after'] and saved['no_mutation'] and not saved['replay'] and not saved['optimizer_feedback'],'OBSERVER_STATE')
        require(digest([r['identity'] for r in rows])==saved['row_order'],'ROW_ORDER_SHA')
        for k,v in reduced.items():
            for field in ('numerator','denominator','desired_token_correct','desired_token_count','strict_numerator'):
                require(v[field]==saved['summary'][k][field],'RECORDED_COUNTS')
            for field in ('rate','token_micro','prompt_macro','true_nll_mean','new_nll_mean'):
                require(math.isclose(v[field],saved['summary'][k][field],abs_tol=1e-10),'RECORDED_MEANS')
            metrics.append(dict(endpoint=b,population='FIRST500' if b==5 else 'CURRENT100',family=k,**v))
        current=[r for r in rows if r['case_id'] in set(expected)];atwrite+=current;allrows[b]=rows
        if b==5:final=rows
        states.append(dict(batch=b,source=cp['source'],W_before=digest(cp['before']['W']),W_after=digest(cp['after']['W']),H_before=digest(cp['before']['H']),H_after=digest(cp['after']['H']),aux_before=digest(cp['before_aux']),aux_after=digest(cp['after_aux']),history_appends=5,terminal_keys_bound=True,observer_no_mutation=True,entry_continuity=True))
        cost.append(dict(batch=b,batch_seconds=cp['seconds'],candidate_seconds_nested=sum(c['seconds'] for c in candidates),native_seconds_nested=sum(c['native_seconds'] for c in candidates),builder_seconds_nested=sum(c['builder_seconds'] for c in candidates),terminal_seconds_nested=term['seconds'],observer_seconds=saved['seconds'],candidates=25,updates=24,logical_ridge_builds=101,masked_backward_calls=sum(c['backward_calls'] for c in candidates),component_builder_backwards=12,optimizer_builder_bridges=25,prediction_positions_one_evaluation_per_candidate=sum(c['prediction_tokens'] for c in candidates),terminal_prediction_positions=term['prediction_tokens'],solve_residual_max=max(v['relative_residual'] for c in candidates for v in c['solve'].values()),LU_fallback_unique=sum(c['solve'][l]['same_A_LU_fallback'] for c in candidates for l in LAYERS if l!='4' or c['candidate']==1),gradient_sum_RMS_max=max(c['gradient'].get('weighted_sum_RMS',0) for c in candidates)))
        last=candidates[-1]
        budget.append(dict(batch=b,fit_nll=last['losses']['nll'],actual_nll_mean=float(np.mean(term['actual_context_nll'])),fit_kl=last['losses']['kl'],actual_kl_mean=float(np.mean(term['actual_native_kl'])),fit_to_actual_target_KL_mean=float(np.mean(term['fit_to_actual_target_kl'])),ideal_Q=sum(v['Q'] for v in last['energy'].values()),effective_Q=sum(term['effective_energy']['Q_effective'].values()),terminal_key_error_max=max(v['error_max'] for v in term['key_comparison'].values())))
        for l in LAYERS:
            for family in ('rewrite','kl'):
                group=[d['layer'][l] for d in term['decomposition'] if d['kind']==family]
                for field in ('key_relative','v_actual_minus_v_fit','prebase_gap','additive_reconstruction_max','builder_key_max'):
                    decomp.append(dict(batch=b,layer=l,family=family,field=field,count=len(group),**distribution([d[field] for d in group])))
        previous=cp
    terminal=read(root/'terminal.json');chain=read(root/'chain-complete.json')
    require(terminal['status']=='COMPLETED' and terminal['source']==SOURCE and terminal['config_sha256']==lock['config_sha256'] and terminal['no_B6'] and terminal['commits']==[['A',b] for b in range(1,6)],'TERMINAL')
    require(chain['state']==previous['after'] and chain['candidates']==125 and chain['updates']==120,'CHAIN_TERMINAL')
    finalmetrics=reduce_rows(final);w0metrics=reduce_rows(reference);pairs=[];pair_detail={};cohorts=[]
    for name,bef in [('W0_TO_W5',reference),('ATWRITE_TO_W5',atwrite)]:
        rows,detail=pair(bef,final,name);pairs+=rows;pair_detail.update(detail)
    for cohort in range(1,6):
        chosen=set(ids[(cohort-1)*100:cohort*100]);a=[r for r in final if r['case_id'] in chosen];before=[r for r in atwrite if r['case_id'] in chosen]
        for stage,rr in [('ATWRITE',before),('W5',a)]:
            for k,v in reduce_rows(rr).items():cohorts.append(dict(birth_batch=cohort,stage=stage,family=k,**v))
        rows,detail=pair(before,a,f'BIRTH_B{cohort}_TO_W5');pairs+=rows;pair_detail.update(detail)
    active=active_ids(data);strata=[]
    for name,chosen in [('ACTIVE',active),('SUPERSEDED',set(ids)-active)]:
        rr=[r for r in final if r['case_id'] in chosen]
        if not rr:strata.append(dict(stratum=name,requests=0,status='NA_EMPTY'));continue
        for k,v in reduce_rows(rr).items():strata.append(dict(stratum=name,requests=len(chosen),status='OBSERVED',family=k,**v))
    for k in FAMILIES:
        group=[r for r in final if r['kind']==k]
        for field in ('true_nll','new_nll','desired_nll','desired_margin'):
            vals=[r[('true' if k=='N' else 'new')+'_nll'] if field=='desired_nll' else (r['new_nll']-r['true_nll'] if k=='N' else r['true_nll']-r['new_nll']) if field=='desired_margin' else r[field] for r in group]
            nlldist.append(dict(family=k,field=field,count=len(vals),**distribution(vals)))
    hist,hist_inputs=baselines(repo);comparison=[]
    for k,v in finalmetrics.items():comparison.append(dict(method='JLZ-v10-A',family=k,endpoint='W5_FIRST500',**v,strict_rate=v['strict_numerator']/v['denominator'],comparison_class='CURRENT_REDUCED_RAW'))
    comparison+=hist
    deltas=[]
    for r in hist:
        a=finalmetrics[r['family']]
        deltas.append(dict(reference=r['method'],family=r['family'],delta_preference_pp=100*(a['rate']-r['rate']),delta_TF_micro_pp=100*(a['token_micro']-r['token_micro']),delta_TF_strict_pp=100*(a['strict_numerator']/a['denominator']-r['strict_rate']),delta_desired_nll=a['desired_nll_mean']-r['desired_nll_mean'],paired_baseline_IDs='NOT_AVAILABLE_AGGREGATE_ONLY'))
    joint={}
    for name,rr in [('ATWRITE',atwrite),('W5',final)]:
        joint[name]=dict(R_plus_twoP_TF_strict=sum(all(r['new_strict'] for r in rr if r['case_id']==cid and r['kind'] in ('R','P')) for cid in ids),denominator=500)
    outputs={'endpoint-metrics.csv':metrics,'cohorts.csv':cohorts,'paired-transitions.csv':pairs,'active-strata.csv':strata,'W5-NLL-distributions.csv':nlldist,'W5-baseline-comparison.csv':comparison,'W5-deltas.csv':deltas,'state-links.csv':states,'cost-by-batch.csv':cost,'candidate-losses.csv':losses,'radius-energy.csv':layers,'component-gradients.csv':components,'terminal-fit-actual.csv':budget,'terminal-decomposition.csv':decomp}
    for filename,rows in outputs.items():table(out/filename,rows)
    write(local/'paired-IDs-local.json',pair_detail)
    accounting=read(snapshot/'accounting.json')
    summary=dict(analysis='독립 CPU raw reducer; owner audit, 별도 reviewer 미사용',A_status='A500_COMPLETE_REVIEWED',B_status='NOT_OBSERVED_THIS_REVIEW',source=SOURCE,lock_sha256=LOCK_SHA,config_sha256=lock['config_sha256'],snapshot_at=inventory['observed_at'],snapshot_files=len(inventory['members']),snapshot_bytes=sum(r['bytes'] for r in inventory['members']),W5=finalmetrics,W0=w0metrics,pairs=pairs[:12],joint=joint,active_requests=len(active),superseded_requests=500-len(active),accounting=accounting,terminal=terminal,cost=dict(candidates=125,updates=120,history_appends=25,logical_ridge_builds=505,masked_backward_calls=sum(r['masked_backward_calls'] for r in cost),component_builder_backwards=60,optimizer_bridges=125,batch_seconds=sum(r['batch_seconds'] for r in cost),observer_seconds=sum(r['observer_seconds'] for r in cost),candidate_seconds_nested=sum(r['candidate_seconds_nested'] for r in cost),terminal_seconds_nested=sum(r['terminal_seconds_nested'] for r in cost)),validation='PASS',no_new_GPU=True,no_new_slurm_write=True,noCP=True,exact_resume='NOT_AVAILABLE',scientific_promotion=False,post_review='REVIEW_COMPLETE_STOP')
    write(out/'summary.json',summary)
    plots(out,comparison,cohorts,losses)
    render_report(out,summary,comparison,cohorts,cost,budget,layers,decomp,nlldist,deltas)
    inputs=[member(snapshot/'snapshot-manifest.json'),member(snapshot/'accounting.json'),member(config['stream']),member(config['W0_reuse']['path'])]+hist_inputs
    write(out/'manifest.json',dict(inputs=inputs,snapshot_members=inventory['members'],analysis_source=member(Path(__file__)),outputs=[member(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'],local_only_transition_IDs=member(local/'paired-IDs-local.json'),source_check_count=len(source_check)))
    print(json.dumps(dict(W5=finalmetrics,validation='PASS',cost=summary['cost'],pairs=pairs[:12]),ensure_ascii=False))

def plots(out,comparison,cohorts,losses):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
    methods=['JLZ-v10-A','MEMIT-H','AlphaEdit','AlphaEdit-BLUE (L4+L8)','CAKE']
    for ax,k in zip(axes,FAMILIES):
        vals=[100*next(r['rate'] for r in comparison if r['method']==m and r['family']==k) for m in methods]
        bars=ax.barh(range(len(methods)),vals,color=['#276FBF']+['#9FA9B7']*4)
        ax.set_yticks(range(len(methods)),methods);ax.invert_yaxis();ax.set_xlim(0,112);ax.set_title(METRIC[k]);ax.set_xlabel('Preference accuracy (%)')
        ax.bar_label(bars,fmt='%.2f',padding=3);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('W5 / same fixed first500 — historical baselines; configurations differ',fontsize=12)
    fig.savefig(out/'W5-comparison.png',dpi=160);fig.savefig(out/'W5-comparison.svg');plt.close(fig)
    svg=out/'W5-comparison.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,k in zip(axes,FAMILIES):
        for stage,marker in [('ATWRITE','o'),('W5','s')]:
            rr=[r for r in cohorts if r['family']==k and r['stage']==stage];ax.plot([r['birth_batch'] for r in rr],[100*r['rate'] for r in rr],marker=marker,label=stage)
        ax.set_title(METRIC[k]);ax.set_xticks(range(1,6));ax.set_xlabel('Birth batch (100 requests)');ax.set_ylabel('Preference (%)');ax.legend();ax.grid(alpha=.2)
    axes[0].set_ylim(99,100.5);axes[0].set_yticks([99,99.5,100])
    fig.suptitle('A: same request cohorts at write and at W5');fig.savefig(out/'cohort-retention.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for b in range(1,6):
        rr=[r for r in losses if r['batch']==b]
        for ax,field in zip(axes,('nll','total')):ax.plot([r['candidate'] for r in rr],[r[field] for r in rr],label=f'B{b}');ax.set_yscale('log');ax.set_xlabel('Evaluated candidate');ax.set_title(field+' (subject-only fit)');ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'fit-losses.png',dpi=160);plt.close(fig)

def render_report(out, summary, comparison, cohorts, cost, terminal_rows, radius, decomp, nlldist, deltas):
    s=summary
    def pct(x):return f'{100*float(x):.2f}'
    def md(headers, rows):
        return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows]
    def lookup(method,k):return next(r for r in comparison if r['method']==method and r['family']==k)
    methods=['JLZ-v10-A','MEMIT-H','AlphaEdit','AlphaEdit-BLUE (L4+L8)','CAKE']
    lines=['# JLZ v10 T′ A500 완료 CPU 리뷰 및 기존 baseline 비교','',
      '**A500_COMPLETE_REVIEWED** — job57699의 B100×5가 COMPLETED/0:0이고, 5 commit·125 candidate·120 Adam update·층별 history 25 append·W5 R500/P1000/N5000을 확인했다. 이 문서는 A arm 완료본이다. B arm과 후속 collector는 이번 리뷰에서 조회하지 않았다.','',
      '사용자 recall: “A 부분 결과 나왔으니 리뷰하고 산출물과 코드 main에 push해. baseline들 (memit-h, alphaedit, alphaedit-blue, CAKE)도 비교에 넣어라.” 부모 instruction은 `ODEEDIT-USER-GH-SH3-JLZ-V10-TPRIME-500-20261003-R1`이다.','',
      '## 1. 동일 W5 / fixed first500 비교','',
      'R/P는 new NLL<true NLL, N은 true NLL<new NLL이며 tie는 실패다. 모든 행의 분모는 R500/P1000/N5000이다. 기존 네 방법은 저장된 **역사 baseline 집계**를 재사용했으며 새 matched 대조 실행이 아니다.','']
    lines+=md(['방법','RS n/500 (%)','PS n/1000 (%)','NS n/5000 (%)'],[[m]+[f"{lookup(m,k)['numerator']}/{lookup(m,k)['denominator']} ({pct(lookup(m,k)['rate'])})" for k in FAMILIES] for m in methods])
    lines+=['','![동일 W5 비교](W5-comparison.png)','','A−baseline의 산술 차이(pp):','']
    lines+=md(['기준','ΔRS','ΔPS','ΔNS'],[[m]+[f"{next(r['delta_preference_pp'] for r in deltas if r['reference']==m and r['family']==k):+.2f}" for k in FAMILIES] for m in methods[1:]])
    lines+=['','A와 AlphaEdit-BLUE는 W5 RS가 모두500/500이다. PS/NS의 baseline 대비 개별 lost/gained는 aggregate만으로 계산할 수 없어 `NOT_AVAILABLE_AGGREGATE_ONLY`로 기록했다. A 내부의 at-write/W0 pairing은 원 ID로 검산했다.','',
      '### 비교 조건과 출처','',
      '| 방법 / job | 층·writer | 설정·실행 차이 |','|---|---|---|',
      '| JLZ v10 A / 57699 | L4–L8, ridge 15000C0+H+KKᵀ; T′, allocation=sum c | seed20261002, norm.5/allocation.1, 25후보24update, active clamp 없음, H200 |',
      '| MEMIT-H / 54007 | BLUE311b076 MEMIT_seq, blue=false, L4–L8; residual5/4/3/2/1 | native L8 z, decay.5/clamp.75, seed20260907, H200 |',
      '| AlphaEdit / 42657 | BASE_ALPHAEDIT, blue=false, L4–L8, projected writer L2=10 | native decay.5/clamp.75, seed20260907, RTX PRO6000 Blackwell |',
      '| AlphaEdit-BLUE / 39283_1 | AlphaEdit_ORIGINAL, blue=true, **L4+L8**, L2=1 | selected-layer native z, decay.5/clamp.75, seed20260907, RTX PRO6000 Blackwell |',
      '| CAKE / 48101 | CAKE_NATIVE L4–L8, projected writer L2=10, native causal weights | upstream c8243e1 / wrapper7884aeb; decay.4/clamp.5/temperature.1, seed20260907, server4 |','',
      '모두 Llama3-8B-Instruct revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, 같은 fixed10k 앞500과 canonical R/P/N 정의를 사용했다는 기존 source/config 증거에 결속했다. 실제 torch2.9.1+cu128 / transformers4.44.2 / FP32 eager / matmulTF32=false다. A는 cuDNNTF32=false, 역사 baseline은 true다. Fit MB2·observer MB16인 A와 native fitting 방법을 동일 알고리즘으로 취급하지 않는다.','',
      '과거 BLUE 데이터 파일 자체 SHA는 원 전체파일 식별자일 수 있어 현재 fixed10k JSON SHA와 다르다. 과거 비교의 ordered root 및 batch request-order 검산을 재사용했다. Canonical context digest `cf14b857…`와 context 파일 bytes SHA `33cec0ee…`는 서로 다른 해시 정의다. 이 리뷰는 cross-host full-model bitwise parity를 새로 측정하지 않았다.','',
      '- [MEMIT-H W5 원표](../../memit-history-fixed10k-20260928-v1/completion-review-r1/all-seen-metrics.csv) / [조건](../historical/historical-manifest.json)',
      '- [AlphaEdit / AlphaEdit-BLUE 원표](../../../server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1/cumulative-metrics.csv)',
      '- [CAKE W5 원표](../../../server4/cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v2/seen-prefix.csv) / [기존 검산](../../../server4/cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v2/diagnostic-report-ko.md)',
      '', '## 2. TF 정확도와 NLL','',
      'TF는 teacher-forced 정확도이며 자유생성 정확도가 아니다. R/P desired=new, N desired=true. token-micro는 정답 target token 합/valid token 합, prompt-macro는 각 prompt token 비율의 평균, strict는 target 전체 token 일치다. Preference와 별개 지표다.','']
    lines+=md(['방법','R TF micro / strict %','P TF micro / strict %','N TF micro / strict %'],[[m]+[f"{pct(lookup(m,k)['token_micro'])} / {pct(lookup(m,k)['strict_rate'])}" for k in FAMILIES] for m in methods])
    lines+=['','A의 정확 분모 및 prompt-macro:','']
    lines+=md(['Family','token correct/valid','prompt-macro %','strict n/d','true NLL','new NLL','desired margin'],[[k,f"{v['desired_token_correct']}/{v['desired_token_count']}",pct(v['prompt_macro']),f"{v['strict_numerator']}/{v['strict_denominator']}",f"{v['true_nll_mean']:.6f}",f"{v['new_nll_mean']:.6f}",f"{v['desired_margin_mean']:.6f}"] for k,v in s['W5'].items()])
    lines+=['','Margin은 R/P=true−new, N=new−true로 양수가 preference 성공 방향이다. A W5 ties는 R/P/N 모두0. Baseline TF prompt-macro는 원 집계에 있는 MEMIT-H만 표에 보존하고, 다른 세 방법은 `NOT_RECORDED_IN_SELECTED_AGGREGATE`로 둔다.','',
      '평균 true/new/desired NLL 전체 비교는 [W5-baseline-comparison.csv](W5-baseline-comparison.csv), A의 median/p90/p95/p99/min/max는 [W5-NLL-distributions.csv](W5-NLL-distributions.csv)다. 원하는 target의 tail:','']
    lines+=md(['Family','평균','median','p95','p99','max'],[[r['family']]+[f"{r[x]:.6f}" for x in ('mean','median','p95','p99','max')] for r in nlldist if r['field']=='desired_nll'])
    lines+=['','## 3. 작성 직후→W5 및 W0 retention','',
      'At-write는 각 요청이 속한 batch 직후의 서로 다른 endpoint다. 하나의 W0 또는 W5 상태로 부르지 않는다. B5 current100은 W5 전체500 raw의 부분집합이며 추가 평가로 중복 계산하지 않았다.','']
    lines+=md(['비교','family / 정의','before→after','lost','gained','retained','분모'],[[r['comparison'],r['family']+'/'+r['definition'],f"{r['before']}→{r['after']}",r['lost'],r['gained'],r['retained'],r['denominator']] for r in s['pairs']])
    n=next(r for r in s['pairs'] if r['comparison']=='W0_TO_W5' and r['family']=='N' and r['definition']=='PREFERENCE')
    lines += ['',f"W0-correct neighborhood 보존은 {n['retained']}/{n['before']}={pct(n['retention_given_before_success'])}%다. W0 대비 NS 총점 변화와 조건부 retention 분모는 구분한다. R+twoP joint TF strict는 at-write {s['joint']['ATWRITE']['R_plus_twoP_TF_strict']}/500 → W5 {s['joint']['W5']['R_plus_twoP_TF_strict']}/500이다.",
      '', '각 birth cohort의 W5 수치(모든 cohort는 R100/P200/N1000):','']
    lines+=md(['Birth batch','W5 RS%','W5 PS%','W5 NS%'],[[b]+[pct(next(r['rate'] for r in cohorts if r['birth_batch']==b and r['stage']=='W5' and r['family']==k)) for k in FAMILIES] for b in range(1,6)])
    lines+=['','![cohort](cohort-retention.png)','',
      f"First100은 birth B1, first500은 전체다. W5까지 재계산한 active 요청은 {s['active_requests']}, superseded는 {s['superseded_requests']}이다. 미래 W20 등의 active mask를 상속하지 않았다. 빈 superseded 층은 NA_EMPTY다. Paired 변화 ID와 case/prompt-index는 local `analysis/paired-IDs-local.json`에 보존했고 Git에는 집계만 게시한다. 사전 계약에 없는 유의성 기준·bootstrap CI를 새로 추가하지 않았다.",
      '', '## 4. 실제 실행·입력·상태 검산','',
      f"고정 snapshot UTC {s['snapshot_at']}: A 및 source/config/제출 receipt {s['snapshot_files']}개, {s['snapshot_bytes']:,}B. 완료 JSON의 stat 전후 동일성과 size/SHA를 봉인했다. Scheduler는 job57699만 한 번 조회했고 이후 반복 polling을 하지 않았다.",
      '',f"Execution source `{SOURCE}`, execution lock `{LOCK_SHA}`, config bytes SHA `{s['config_sha256']}`. 실제 import와 frozen source/참조 파일은 [source-verification.csv](source-verification.csv)로 검산했다. 원 production source와 기존 CPU collector는 변경하지 않았다. 새 리뷰 코드는 `project/run_scripts/jlz_realized_subject/review_A.py` 및 `test_review_A.py`다.",
      '',f"Dataset SHA `{DATA_SHA}`, ordered root `5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729`, first500 case-ID SHA `0be7d88c759e7f65a690514035f40a18c5c19d8591ddd33f97b2fa6187b64a95`. W0 reuse SHA `6b2d25f39f751937e1b6f797e98e8c4117d539d86fdddc21e4ebabc385a97836`. 원 model/C0를 재해시·전송하지 않았다.",
      '', '| 검산 | 결과 / 증거 경계 |','|---|---|',
      '| case/order/prompt-target hash/token ID | 5 native packed identity, W1–W4 각1300행/W5 6500행, 요청별1R/2P/10N 및 W0 token identity PASS |',
      '| 저장 집계와 독립 CPU reducer | integer counts 동일; rate/micro/macro/NLL 평균 차 ≤1e−10 |',
      '| 실제 동일 terminal weight commit | 5 accepted_weight_copy_exact, evaluated payload 그대로 복사 source 및 receipt |',
      '| history keys와 append | terminal actual key hash와 commit key hash 25개 일치, 층별 batch당1회, CPUFP32 Gram |',
      '| W/H/context/RNG/ledger 연속 | 초기 state→B1 및 인접 B1→B2→B3→B4→B5 hash 일치 |',
      '| Observer 비변이 | 저장 no_mutation 및 W/H hash; frozen source의 guard/hooks/RNG/context 검사와 next-entry aux 연결 |',
      '| 25후보/24update / 계수 | 5batch 전125후보·120update, terminal25 component backward 수행/update0, .5/.1/.0625 불변 |',
      '| allocation / upper gradient | A sum(c)와 total loss 산술 일치; upper L5–8 K/P gradient-enabled receipt; Q1 actual gradient 증거는 별도 |',
      '| excluded feature / noCP | clamp/pulse/E/replay와 복원 payload 저장 false; B6 없음 |','',
      'W/H 텐서는 noCP로 저장되지 않았다. 저장 hash·source·receipt 검산을 사후 tensor 재실행 또는 exact resume로 확대하지 않는다. `exact_resume=NOT_AVAILABLE`. W0 selected weights는 실행 setup에서 기존 bound W0와 비교했다.','',
      '독립 reducer는 production observer/collector 및 torch를 import하지 않는다. CPU synthetic 7검사는 tie/NS 방향, micro≠macro, strict/finite/중복 rejection, family-major W0와 case-major actual 순서, token mismatch, 총점이 같은 lost/gained ID, active mask와 quantile을 검사했다. Owner audit이며 별도 독립 agent review는 수행하지 않았다.','',
      '## 5. T′ 수치·fit/actual·계산량 관측','',
      'Frozen causal_builder의 whole-B ridge, RW-only K, lower actual all-token write, 두 linear input VJP, terminal materialized weight 재사용에 대해 이전 Q1 실제 qualification 및 이번 source/telemetry 결속을 구분한다. Q1 job57698의 dense gradient RMS 최대차8.103e−9, reversed MB1 최대2.949e−7, 고정허용오차 내였다는 [기존 기록](../Q1-ko.md)을 재사용한다. 이 값은 다른 host·모든 입력의 bitwise certification이 아니다.','',
      f"A 저장 solve residual 최대 {max(r['solve_residual_max'] for r in cost):.6g} ≤1e−8. Same-A LU fallback {sum(r['LU_fallback_unique'] for r in cost)}회. 후보2/9/25의 coefficient-weighted gradient 합 RMS 최대 {max(r['gradient_sum_RMS_max'] for r in cost):.6g}; 각 기록의 기존 RMS limit 이내다. Upper K/P gradient flag는 실제 graph 수학의 새 독립 GPU 증명과 구분한다.",
      '', 'Terminal25의 subject-only fit loss와 actual all-token forward는 다음과 같다. Actual NLL은 native RW6 context 평균이며 canonical R/P/N 평가 NLL과 다른 입력이다.','']
    lines+=md(['Batch','fit NLL','actual RW NLL','fit KL','actual KL','ideal Q','effective Q'],[[r['batch']]+[f"{r[k]:.6f}" for k in ('fit_nll','actual_nll_mean','fit_kl','actual_kl_mean','ideal_Q','effective_Q')] for r in terminal_rows])
    lines+=['', 'fit/actual gap은 record-only이며 0을 요구하는 gate가 아니다. Ideal Q와 actual effective Q, layer norm share와 인과 기여를 같은 값으로 부르지 않는다. 세부값은 [terminal-fit-actual.csv](terminal-fit-actual.csv), [terminal-decomposition.csv](terminal-decomposition.csv), [component-gradients.csv](component-gradients.csv)에 보존했다.','']
    for family in ('rewrite','kl'):
        rr=[r for r in radius if r['family']==family]
        lines.append(f"- {family}: 전125후보×5층의 관측 {sum(r['count'] for r in rr):,} request/context/layer/candidate 값 중 v/a>.75는 {sum(r['exceeds_075'] for r in rr)}, 최대 {max(r['max'] for r in rr):.6f}. 이는 서로 독립인 요청 수가 아니다.")
    lines+=['','![fit losses](fit-losses.png)','',
      '## 6. 비용과 자원','']
    a=s['accounting'];t=s['terminal'];c=s['cost']
    lines+=md(['항목','실측 / 구분'],[
      ['Slurm A parent allocation',f"{a['allocated_GPU_seconds']} GPU-sec = {a['allocated_GPU_seconds']/3600:.6f} GPUh; 1GPU/8CPU/59GiB"],
      ['scheduler 시간',a['start_scheduler']+' → '+a['end_scheduler']+' (scheduler 표기)'],
      ['program wall',f"{t['seconds']:.3f}s; allocation 안에 포함"],
      ['5batch fitting/commit',f"{c['batch_seconds']:.3f}s"],
      ['canonical observer',f"{c['observer_seconds']:.3f}s; W1–4 current100 + W5 all500"],
      ['candidate timer 합',f"{c['candidate_seconds_nested']:.3f}s; batch 내부, terminal observation도 내부"],
      ['terminal actual observation',f"{c['terminal_seconds_nested']:.3f}s; candidate25 내부라 별도 가산0"],
      ['Torch GPU peak',f"{t['peak_gpu_bytes']:,}B = {t['peak_gpu_bytes']/2**30:.3f}GiB (allocated peak, reserved/device전체 아님)"],
      ['process ru_maxrss',f"{t['peak_host_kib']:,}KiB = {t['peak_host_kib']/2**20:.3f}GiB"],
      ['sacct batch MaxRSS',f"{a['batch_MaxRSS_KiB']:,}KiB = {a['batch_MaxRSS_KiB']/2**20:.3f}GiB; sampling/정의가 달라 ru_maxrss와 구분"],
      ['logical ridge builds','505 = (1+25×4)×5; kernel/linear solve 총호출 수와 다름'],
      ['masked backward calls',str(c['masked_backward_calls'])],
      ['component builder backwards / optimizer bridge','60 /125; component 진단은 추가비용'],
      ['Q1 공유 준비비용','425 GPU-sec, A allocation6155에 미포함; B에 다시 독립 준비비용으로 중복 청구하지 않음']])
    lines+=['', 'Prediction-position count는 [cost-by-batch.csv](cost-by-batch.csv)에 원 계측 의미대로 저장했다. 모든 repeated forward/backward의 token FLOPs·kernel profiler·history/IO/metadata별 wall 분해는 NOT_SEPARATED다. 기존 baseline의 100batch 전체 allocation과 A의5batch allocation을 동일500 throughput으로 비교하지 않았다.','',
      '## 7. 재현·산출물·종료','',
      '실행은 아래 CPU 명령으로 같은 고정 snapshot에서 재현한다. 원본/실행 작업을 쓰거나 scheduler를 조회하는 코드가 없다.','',
      '```bash',
      'PY=/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/venv/bin/python',
      'REVIEW=/data/janghj/ODE-edit/local/jlz-realized-subject-v10/20261003-v1/review-A-20261003-v1',
      'OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 "$PY" -m unittest project.run_scripts.jlz_realized_subject.test_review_A -v',
      'OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 "$PY" -m project.run_scripts.jlz_realized_subject.review_A \\',
      '  --snapshot "$REVIEW/snapshot" --repository . \\',
      '  --out "$REVIEW/reproduction-report" --local-out "$REVIEW/reproduction-local"',
      '```','',
      'Source, raw snapshot, comparison input의 size/SHA 및 산출물은 [manifest.json](manifest.json)에 있다. Raw·prompt·tensor·fullstdout·paired ID 목록은 Git에 넣지 않았다. 표/그림/source와 작은 manifest만 main 게시한다. 원 실행 코드 c2d5fb10은 이미 main ancestry에 포함되어 있고 이번 commit은 CPU 리뷰 코드와 결과물을 추가한다.','',
      '이 리뷰의 확인 범위는 A500이다. B 진행/종료, 원 collector/수리 collector의 결과는 NOT_OBSERVED_THIS_REVIEW이며 변경하지 않았다. 새 GPU forward/fit, baseline rerun, job 제출·취소·수리·resume, checkpoint 저장·삭제 모두0. 게시 후 REVIEW_COMPLETE_STOP; 자동 polling/heartbeat/recall0. 과학적 원인·우열·promotion 판정은 하지 않았다.','']
    (out/'report-ko.md').write_text('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--repository',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--local-out',type=Path,required=True);a=p.parse_args();review(a.snapshot,a.repository,a.out,a.local_out)
