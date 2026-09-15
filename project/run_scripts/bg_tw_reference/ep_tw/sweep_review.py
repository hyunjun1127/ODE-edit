"""CPU-only cap-sweep aggregation; no model/native/evaluator imports.

Run `first` as endpoints arrive, `arm` only for a completed chain, then
`combine` after all three independent reductions. Artifact outputs create once.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

from . import review_nogate as r
from .sweep_control import ROOT,OLD,NEW_ARMS,TASK,verify_arm_lock

REPORT='experiment-reports/servers/server4/ep-tw1-alpha-cap-sweep-2026-09-15-v1'
OLD_REPORT='experiment-reports/servers/server4/ep-tw1-c4-2026-09-15-v1/gate-skip-r1/completed-review-v1'
LABELS=('CAP1',*NEW_ARMS)
METRICS=('RS','PS','NS')


def status(output):
    """One explicit own-job observation. No polling/sleep/callback."""
    release=r.read(ROOT/'submission-v1/release-receipt.json')
    jobs=[x['job_id'] for x in release['jobs']];assert jobs==['48148','48149','48150']
    queries={}
    for label,cmd in (
        ('squeue',['squeue','-h','-j',','.join(jobs),'-o','%i|%j|%T|%M|%R']),
        ('sacct',['sacct','-n','-P','-j',','.join(jobs),'--format=JobIDRaw,JobName%40,User,State,ExitCode,Submit,Start,End,ElapsedRaw,AllocTRES%150,MaxRSS,MaxVMSize,ReqMem,NodeList'])):
        p=subprocess.run(cmd,text=True,capture_output=True)
        queries[label]=dict(args=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    states=[]
    for job in release['jobs']:
        raw=Path(job['output']);commits=sorted(raw.glob('B*/commit.json'))
        states.append(dict(arm=job['arm'],job_id=job['job_id'],output_exists=raw.exists(),
            committed_batches=len(commits),last_commit=r.ref(commits[-1]) if commits else None,
            terminal=r.read(raw/'terminal.json') if (raw/'terminal.json').exists() else None,
            failure=r.read(raw/'failure.json') if (raw/'failure.json').exists() else None,
            initial_marker_exists=(raw/'INITIAL_EXECUTION_OBSERVED_WITH_VALIDATION_SKIPPED.json').exists()))
    return r.save(output,dict(time=datetime.now(timezone.utc).isoformat(),queries=queries,states=states,
        scope='EXACT_THREE_SWEEP_JOBS_ONLY_NO_CAKE_QUERY',automatic_callback=False))


def csvrows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def attempt(arm):return OLD if arm=='CAP1' else ROOT/arm/'attempt-v1'


def finalrows(arm):
    data=r.read(attempt(arm)/'scientific-v1/B010/selected-evaluation.json')
    return {m:r.panel_rows(data['seen_full'],m) for m in METRICS}


def first(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    rows=[];counts=[];inputs=[]
    for arm in LABELS:
        p=attempt(arm);t=p/'scientific-v1/terminal.json'
        if not t.exists():
            rows.append(dict(arm=arm,status='NOT_TERMINAL',numerator='NOT_AVAILABLE',denominator='NOT_AVAILABLE'))
            continue
        term=r.read(t);assert term['completed_requests']==1000 and len(term['commits'])==10
        rr=finalrows(arm);inputs.extend([r.ref(t),r.ref(p/'scientific-v1/B010/selected-evaluation.json')])
        row=dict(arm=arm,status='STORED_TERMINAL_NLL_REDUCED_NOT_FULL_AUDIT',reuse=arm=='CAP1')
        for m,expected in zip(METRICS,(1000,2000,10000)):
            summary=r.summary(rr[m],m);assert summary['denominator']==expected
            row.update({m+'_n':summary['numerator'],m+'_d':expected,m+'_percent':summary['percent']})
        rows.append(row)
        c=Counter(r.read(p/f'scientific-v1/B{b:03d}/commit.json')['selected'] for b in range(1,11))
        counts.append(dict(arm=arm,**{k:c[k] for k in ('RAW','C1','C05','C025')}))
    r.table(out/'first-final-table.csv',rows);r.table(out/'selection-frequency.csv',counts)
    return r.save(out/'receipt.json',dict(instruction_id=TASK,inputs=inputs,rows=rows,
        numerical_validation='NOT_ESTABLISHED',not_final_audit=True,new_model_forwards=0))


def load_state_module():
    path=Path(__file__).parent/'review_nogate/state_audit.py'
    spec=importlib.util.spec_from_file_location('ep_sweep_cpu_state',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def arm_review(worktree,arm,output,allocated):
    assert arm in NEW_ARMS
    assert isinstance(allocated,int) and allocated>0,'ACTUAL_ALLOCATION_REQUIRED_NOT_PRIOR_ESTIMATE'
    w,out=Path(worktree),Path(output);out.mkdir(parents=True,exist_ok=False)
    a=attempt(arm);lock=r.read(a/'execution.lock.json');verify_arm_lock(lock)
    term=r.read(a/'scientific-v1/terminal.json');assert term['completed_requests']==1000
    result=r.reduce(w,out/'metrics',root=a,expected_lock_sha=r.sha(a/'execution.lock.json'),
        allocated_gpu_seconds=allocated,instruction_id=TASK)
    load_state_module().run(out/'state',root=a)
    # Mechanism scalars from actual new tensors only. No CAP1 geometry rerun.
    import torch
    torch.set_num_threads(4)
    action=[];generic=[];probes=[];previous=None;s64_ids=None;dev_ids=None
    def norm(t):return float(t.detach().double().norm())
    def dot(x,y):return float((x.detach().double()*y.detach().double()).sum())
    def check_generic(obs,b,cid):
        nonlocal s64_ids,dev_ids
        rr=obs['rows'];role=obs['role'];n=64 if role=='S64' else 128
        ids=[x['source_row_id'] for x in rr]
        assert len(ids)==len(set(ids))==n==obs['denominator']
        assert all(x['scored_positions']==128 and x['input_tokens']==257 and x['role']==role for x in rr)
        assert all(math.isfinite(x['kl']) and math.isfinite(x['natural_nll']) for x in rr)
        assert abs(statistics.fmean(x['kl'] for x in rr)-obs['D'])<1e-14
        if role=='S64':
            if s64_ids is None:s64_ids=ids
            assert ids==s64_ids
        else:
            if dev_ids is None:dev_ids=ids
            assert ids==dev_ids
        generic.append(dict(arm=arm,batch=b,candidate=cid,role=role,documents=n,D64_or_D128=obs['D'],
            natural_NLL=statistics.fmean(x['natural_nll'] for x in rr),input_tokens=n*257,scored_positions=n*128,
            forwards=obs['counts']['forwards'],backwards=obs['counts']['backwards'],
            row_identity_sha=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()))
    for b in range(1,11):
        p=a/f'scientific-v1/B{b:03d}';pol=r.read(p/'policy.json');diag=pol['correction']
        route=torch.load(p/'route.pt',map_location='cpu',weights_only=True,mmap=True)
        cp=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=True,mmap=True)
        native=torch.load(p/'native-targets-map.pt',map_location='cpu',weights_only=True,mmap=True)
        Vp=native['native_proposal'];W=next(iter(cp['weights'].values()));C=route['C'];A=route['map_A']
        ge,gd=route['gE'],route['gD'];q=dot(ge,gd);e2=dot(ge,ge);coef=min(q,0)/e2 if e2 else 0
        direction=-gd if coef==0 else -gd+coef*ge
        arnorm=.25*diag['actual_native_delta_norm']/(diag['mapped_direction_norm']+1e-12)
        mode=lock['numerical_policy']['alpha_cap_mode'];cap=lock['numerical_policy']['alpha_cap']
        alpha=min(cap,arnorm) if mode=='bounded' else arnorm
        assert alpha==diag['alpha'] and arnorm==diag['alpha_norm']
        assert diag['alpha_cap_mode']==mode and diag['alpha_cap']==cap
        assert diag['alpha_cap_active']==(mode=='bounded' and cap<arnorm)
        candidates=pol['selection']['candidate_receipts'];selected=next(x for x in candidates if x['id']==pol['selection']['selected_id'])
        c1=next(x for x in candidates if x['id']=='C1');actual=W-Vp
        actual_norm=norm(actual);native_norm=diag['actual_native_delta_norm']
        assert abs(actual_norm-selected['actual_correction_norm'])<1e-10*max(1,actual_norm)
        reconstructed=Vp.clone() if selected['id']=='RAW' else Vp+(C@A)*selected['beta']
        row=dict(arm=arm,batch=b,selected=selected['id'],cap_mode=mode,cap=cap,
            alpha_norm=arnorm,alpha_used=alpha,cap_active=diag['alpha_cap_active'],
            native_action_norm=native_norm,C1_norm=c1['actual_correction_norm'],C1_native_percent=100*c1['actual_correction_norm']/native_norm,
            selected_norm=actual_norm,selected_native_percent=100*actual_norm/native_norm,
            pre_ball_residual_norm=diag.get('pre_ball_residual_norm','NOT_RECORDED'),
            post_ball_mapped_norm=diag.get('post_ball_mapped_norm','NOT_RECORDED'),
            ball_hit=diag['ball_projected_requests'],trust_retraction=diag['trust_retraction'],
            q_CPU=q,q_recorded=diag['projection']['q_ge_gd'],ge_norm=norm(ge),gd_norm=norm(gd),
            gradient_cos=q/(norm(ge)*norm(gd)) if norm(ge)*norm(gd) else None,
            projection_active=q<0,coefficient_CPU=coef,ge_dot_d_CPU=dot(ge,direction),gd_dot_d_CPU=dot(gd,direction),
            ge_dot_C_CPU=dot(ge,C),gd_dot_C_CPU=dot(gd,C),ge_dot_C_recorded=diag['ge_dot_correction'],gd_dot_C_recorded=diag['gd_dot_correction'],
            C_norm=norm(C),M4_norm=norm(cp['M4']),W4_norm=norm(W),
            selected_reconstruction_CPU_maxabs=float((W-reconstructed).abs().max()),
            selected_reconstruction_CPU_unequal=int((W!=reconstructed).sum()),
            reconstruction_is_GPU_parity=False,direction_tensor='CPU_DERIVED_FROM_STORED_GRADIENTS_NOT_ORIGINAL_SAVED_D')
        if previous is not None:
            nd=Vp-previous;nn=norm(nd);dp=dot(nd,actual);parallel=dp/nn if nn else 0
            row.update(native_norm_CPU=nn,selected_increment_norm=norm(W-previous),
                correction_native_cos=dp/(nn*actual_norm) if nn*actual_norm else None,
                parallel_signed_norm=parallel,orthogonal_norm=math.sqrt(max(0,actual_norm**2-parallel**2)))
        else:row['B1_native_geometry']='RECORDED_ACTION_NORM_AND_W0_HASH; no new base-model tensor reload'
        action.append(row);previous=W.clone()
        ce=r.read(p/'candidate-evaluation.json');raw=ce['RAW']
        for cr in candidates:
            cid=cr['id'];src=cr.get('evaluation_source_id') or cr.get('duplicate_of') or cid
            obs=ce.get(src)
            if not obs or 'current' not in obs:continue
            if cid in ce:check_generic(obs['generic'],b,cid)
            de=obs['current']['E']-raw['current']['E'];dd=obs['generic']['D']-raw['generic']['D']
            raw_byid={x['case_id']:x for x in raw['current']['rows']}
            harms=[x['nll']-raw_byid[x['case_id']]['nll'] for x in obs['current']['rows']]
            probe=dict(arm=arm,batch=b,candidate=cid,selected=cid==selected['id'],beta=cr['beta'],
                E=obs['current']['E'],D64=obs['generic']['D'],actual_delta_E=de,actual_delta_D=dd,
                firstorder_delta_E=cr['beta']*diag['ge_dot_correction'],firstorder_delta_D=cr['beta']*diag['gd_dot_correction'],
                current_NLL_harmed=sum(v>0 for v in harms),current_NLL_improved=sum(v<0 for v in harms),
                **r.dist(harms,'current_NLL_harm_'),feasible=cr['feasible'],strict_lost=cr['raw_strict_lost_count'],
                called=cid in ce,duplicate_of=cr['duplicate_of'],correction_native_percent=100*cr['actual_correction_norm']/native_norm)
            probes.append(probe)
        if b in (5,10):check_generic(r.read(p/'selected-evaluation.json')['Dev128'],b,'SELECTED')
        del route,cp,native,Vp,W,C,A,ge,gd,direction,actual,reconstructed
    assert set(s64_ids).isdisjoint(dev_ids)
    r.table(out/'per-batch-actions.csv',action);r.table(out/'candidate-firstorder-observed.csv',probes)
    r.table(out/'generic-panels.csv',generic)
    member_refs=[r.ref(a/'execution.lock.json'),r.ref(a/'input-verification.json'),r.ref(a/'scientific-v1/terminal.json')]
    state=r.read(out/'state/state-summary.json')
    return r.save(out/'arm-summary.json',dict(arm=arm,execution_source=lock['source_head'],execution_tree=lock['source_tree'],
        inputs=member_refs,metrics=result,state=state,cap_mode=mode,cap=cap,
        action_summary=dict(C1_percent=r.dist((x['C1_native_percent'] for x in action),''),
            selected_including_RAW_percent=r.dist((x['selected_native_percent'] for x in action),''),
            nonzero_selected_percent=r.dist((x['selected_native_percent'] for x in action if x['selected_norm']>0),''),
            selected_nonzero=sum(x['selected_norm']>0 for x in action),
            cap_active=sum(x['cap_active'] for x in action),ball_hits=sum(x['ball_hit'] for x in action),
            trust_retractions=sum(x['trust_retraction']<1 for x in action)),
        S64_Dev128_disjoint=True,numerical_validation='NOT_ESTABLISHED',new_model_forwards=0,
        final_W10_weight_retained=r.ref(a/'scientific-v1/B010/checkpoint.pt')))


def cluster_interval(before,after,metric,*,seed=20260915,draws=2000):
    """Resample requests, never P or N prompts or ten batches as replicates."""
    import numpy as np
    pairs={r.key(x):x for x in before};bycase={}
    assert set(pairs)=={r.key(y) for y in after}
    for y in after:
        x=pairs[r.key(y)];v=bycase.setdefault(y['case_id'],[0,0])
        v[0]+=int(r.success(y,metric))-int(r.success(x,metric));v[1]+=1
    vals=np.array(list(bycase.values()),dtype=np.float64);rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(draws):
        take=vals[rng.integers(0,len(vals),size=len(vals))]
        samples.append(100*take[:,0].sum()/take[:,1].sum())
    return dict(CI_low=float(np.quantile(samples,.025)),CI_high=float(np.quantile(samples,.975)),
        bootstrap_unit='REQUEST_CLUSTER_FIXED_SINGLE_ORDER_NOT_SEQUENTIAL_REPLICATION',bootstrap_draws=draws,bootstrap_seed=seed)


def combine(worktree,analysis_root,output):
    w,ar,out=Path(worktree),Path(analysis_root),Path(output);out.mkdir(parents=True,exist_ok=False)
    oldpub=w/OLD_REPORT;inputs=[]
    tables=('batch-current-metrics','whole-prefix-metrics','paired-transitions','cohort-retention','candidate-details',
        'batch-policy','parity-warnings','ledger-summary','cost-by-batch','general-observer')
    for name in tables:
        rows=[]
        for arm in LABELS:
            p=oldpub/(name+'.csv') if arm=='CAP1' else ar/arm/'metrics'/(name+'.csv')
            inputs.append(r.ref(p));rows.extend(dict(arm=arm,provenance='SEALED_CAP1_REUSE' if arm=='CAP1' else 'NEW_CHAIN_CPU_REDUCTION',**x) for x in csvrows(p))
        r.table(out/(name+'.csv'),rows)
    final={arm:finalrows(arm) for arm in LABELS};first=[];paired=[]
    for arm in LABELS:
        for metric in METRICS:
            s=r.summary(final[arm][metric],metric,arm=arm,state='W10_FULL_FIRST1000',reuse=arm=='CAP1')
            b=r.summary(final['CAP1'][metric],metric)
            s.update(delta_CAP1_count=s['numerator']-b['numerator'],delta_CAP1_pp=s['percent']-b['percent']);first.append(s)
        if arm!='CAP1':
            for metric in METRICS:
                paired.append(dict(r.transition(final['CAP1'][metric],final[arm][metric],metric,comparison='CAP1_TO_'+arm),
                    **cluster_interval(final['CAP1'][metric],final[arm][metric],metric)))
    for left,right in (('CAP10','CAP100'),('CAP100','NORM_ONLY')):
        for metric in METRICS:
            paired.append(dict(r.transition(final[left][metric],final[right][metric],metric,comparison=left+'_TO_'+right),
                **cluster_interval(final[left][metric],final[right][metric],metric)))
    r.table(out/'first-final-table.csv',first);r.table(out/'cross-arm-paired.csv',paired)
    actions=[];probes=[];generic=[]
    for arm in NEW_ARMS:
        for name,dst in [('per-batch-actions',actions),('candidate-firstorder-observed',probes),('generic-panels',generic)]:
            p=ar/arm/(name+'.csv');inputs.append(r.ref(p));dst.extend(csvrows(p))
    # CAP1 geometry is already reviewed; read only the published scalar table.
    oldmech=csvrows(oldpub/'per-batch-mechanism.csv');oldcand=csvrows(oldpub/'candidate-details.csv')
    for x in oldmech:
        b=x['batch'];c1=next(c for c in oldcand if c['batch']==b and c['candidate']=='C1')
        actions.append(dict(arm='CAP1',batch=b,selected=x['selected'],cap_mode='bounded',cap=1,
            alpha_norm=.25*float(x['native_action_norm'])/(float(x['mapped_direction_norm_recorded'])+1e-12),
            alpha_used=x['alpha'],cap_active=True,native_action_norm=x['native_action_norm'],
            C1_norm=c1['actual_correction_norm'],C1_native_percent=100*float(c1['actual_correction_norm'])/float(x['native_action_norm']),
            selected_norm=x['actual_correction_norm'],selected_native_percent=100*float(x['correction_native_ratio']),
            ball_hit=x['ball_clamped_requests'],trust_retraction=x['trust_retraction'],
            source='SEALED_CAP1_SCALARS_REUSED_CP_CURRENT_ABSENT'))
    actions.sort(key=lambda x:(LABELS.index(x['arm']),int(x['batch'])))
    r.table(out/'per-batch-actions.csv',actions);r.table(out/'candidate-firstorder-observed.csv',probes);r.table(out/'generic-panels.csv',generic)
    action_summary=[]
    for arm in LABELS:
        aa=[x for x in actions if x['arm']==arm];freq=Counter(x['selected'] for x in aa)
        action_summary.append(dict(arm=arm,**r.dist((float(x['C1_native_percent']) for x in aa),'C1_percent_'),
            **r.dist((float(x['selected_native_percent']) for x in aa),'selected_with_RAW_percent_'),
            **r.dist((float(x['selected_native_percent']) for x in aa if float(x['selected_norm'])>0),'nonzero_selected_percent_'),
            **{k:freq[k] for k in ('RAW','C1','C05','C025')}))
    r.table(out/'action-summary.csv',action_summary)
    # Published aggregates are not reverse-engineered into per-request rows.
    for name in ('baseline-first1000.csv','compatibility.csv'):
        r.table(out/name,csvrows(oldpub/name));inputs.append(r.ref(oldpub/name))
    base_sources=csvrows(w/'experiment-reports/servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1/source-config-compatibility.csv')
    baseline_pairs=[]
    for label in ('AlphaEdit_L4_ONLY','AlphaEdit_ORIGINAL','MEMIT_ORIGINAL','BASE_ALPHAEDIT','BASE_MEMIT'):
        source=next(x for x in base_sources if x['arm']==label);path=Path(source['raw_root'])/'B010/seen-full.json'
        if not path.exists():continue
        base=r.read(path);inputs.append(r.ref(path))
        for arm in LABELS:
            for metric in METRICS:
                baseline_pairs.append(r.transition(r.panel_rows(base,metric),final[arm][metric],metric,
                    comparison=label+'_TO_'+arm,baseline=label,arm=arm,state='MATCHED_PROMPT_IDENTITIES_DIFFERENT_TRAJECTORY_HPARAMS_SEED'))
    r.table(out/'baseline-paired.csv',baseline_pairs)
    costs=[];states=[]
    for arm in NEW_ARMS:
        s=r.read(ar/arm/'arm-summary.json');inputs.append(r.ref(ar/arm/'arm-summary.json'))
        costs.append(dict(arm=arm,cost_origin='NEW_ALLOCATION',allocated_GPU_seconds=s['metrics']['allocated_GPU_seconds'],
            program_seconds=s['metrics']['terminal_seconds'],**s['metrics']['cost_totals'],peak_GPU_allocated=s['metrics']['peak_GPU_allocated']))
        states.append(dict(arm=arm,**s['state']))
    prior=r.read(oldpub/'metrics-summary.json')
    costs.insert(0,dict(arm='CAP1',cost_origin='REUSED_NOT_NEW_EXPENSE',allocated_GPU_seconds=7694,
        program_seconds=prior['terminal_seconds'],**prior['cost_totals'],peak_GPU_allocated=prior['peak_GPU_allocated']))
    r.table(out/'cost-summary.csv',costs);r.table(out/'new-state-summary.csv',states)
    inventory=[]
    for arm in NEW_ARMS:
        for name in ('checkpoint-inventory','state-links','raw-member-inventory','target-counters'):
            p=ar/arm/'state'/(name+'.csv')
            r.table(out/(arm+'-'+name+'.csv'),csvrows(p));inputs.append(r.ref(p))
    summary=dict(instruction_id=TASK,arms=LABELS,new_chains=3,reused_chains=1,new_batches=30,reused_batches=10,
        unique_requests=1000,new_arm_request_observations=3000,final=first,actions=action_summary,
        new_allocated_GPU_seconds=sum(x['allocated_GPU_seconds'] for x in costs if x['cost_origin']=='NEW_ALLOCATION'),
        reused_CAP1_GPU_seconds=7694,reused_teacher_GPU_seconds=98,old_failed_GPU_seconds=473+69,
        old_native286_seconds_nested_in473=True,inputs=inputs,numerical_validation='NOT_ESTABLISHED',
        validation_mode='SKIPPED_USER_DIRECTED',CAP1_CP_current='ABSENT_PER_SEALED_REVIEW_NOT_FULL_RESUME',
        GH_geometry='OLD_CAP1_SAVED_EPISODES_CPU_ONLY_NOT_THESE_NEW_TRAJECTORIES',
        no_model_forwards_in_analysis=True,source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip())
    return r.save(out/'sweep-summary.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['status','first','arm','combine']);p.add_argument('--output',required=True)
    p.add_argument('--worktree',default='.');p.add_argument('--arm',choices=NEW_ARMS);p.add_argument('--allocated',type=int)
    p.add_argument('--analysis-root');x=p.parse_args()
    result=status(x.output) if x.command=='status' else first(x.output) if x.command=='first' else arm_review(x.worktree,x.arm,x.output,x.allocated) if x.command=='arm' else combine(x.worktree,x.analysis_root,x.output)
    print(json.dumps(result,ensure_ascii=False))
