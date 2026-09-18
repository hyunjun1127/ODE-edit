"""EN-F scoped publication, source audit, figures. No scientific execution."""
import argparse
import ast
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
from .enf_sequential_review import ROOT,ARM,REPORT,read,sha,member,write,csvout,pairs,summary,quantiles,strict

def provenance(repo,scratch):
    out=repo/REPORT;lock=read(ROOT/'execution.lock.json');checks=[]
    for m in lock['execution']['members']:
        current=member(m['path']);assert current['sha256']==m['sha256'] and current['bytes']==m['bytes']
        checks.append(current|dict(level='CURRENT_FULL_SHA_FROZEN_SOURCE'))
    a=member(lock['execution']['archive']['path']);assert a==lock['execution']['archive'];checks.append(a|dict(level='CURRENT_FULL_SHA_ARCHIVE'))
    for key in ('authority','teacher_manifest','cold_capsule','technical_evidence'):
        m=lock[key];assert member(m['path'])==m;checks.append(m|dict(level='CURRENT_SMALL_MANIFEST_FULL_SHA',role=key))
    for key in ('config4',):checks.append(member(lock[key])|dict(role=key,level='CURRENT_FULL_SHA'))
    # Avoid rehashing static full models / teacher payloads. Prior immutable seals
    # are reused explicitly, never described as a new payload audit.
    for key in ('P_star_basis',):
        m=lock[key];assert Path(m['path']).stat().st_size==m['bytes'];checks.append(m|dict(level='PRIOR_FULL_SHA_REUSED_CURRENT_STAT',role=key))
    family=REPORT.parent
    for rel in (family/'submission/diagnostic-report-ko.md',family/'initial/diagnostic-report-ko.md',
                Path('messages/head/2026-09-18-sh4-enfc-skip-t-all-m.md')):
        checks.append(member(repo/rel)|dict(level='FULL_READ_CURRENT_SHA_HISTORICAL_AUTHORITY_OR_HANDOFF'))
    checks.append(lock['authority']|dict(level='FULL_READ_CURRENT_SHA_S_OVERRIDE'))
    core=ROOT/'source/project/run_scripts/single_layer_edit_preserving_correction'
    spec=[
      ('cold/common/input','runtime.py','Runtime.__init__','W0/M0/model revision/FP32/TF32/seed/prefix/context/P4','runtime-load + B001 ENTRY + lock','SOURCE_CONFIRMED_AND_STORED_CONSISTENT','Full model reload not performed'),
      ('own native B2+','sequential_runtime.py','SequentialRuntime.native_batch','own entry W/M + current 100, one native z per request','native-binding + target tensors + prior commit','CPU_STORED_EVIDENCE_CONSISTENT','Native solver not rerun'),
      ('K_E union','binding.py','protected_sequences','native/canonical old+new, all valid token prefixes, no pad/EOS/P-N','protected-provenance 1400 rows per batch and key aliases','SOURCE_CONFIRMED_AND_STORED_CONSISTENT','Full captured K_E is not retained'),
      ('actual key dedup','runtime.py','Runtime.protected_oracle','only same prefix AND exact FP32 key bytes','key hash/shape/actual alias mapping','SOURCE_CONFIRMED_AND_STORED_CONSISTENT','Cannot re-capture keys CPU-only'),
      ('allowed null space','geometry.py','edit_null_space','Q=V(I-UU^T)V^T; fixed rank cutoff and ambiguity band','space/factors + singular spectrum arithmetic','CPU_STORED_EVIDENCE_CONSISTENT','No new SVD, P-star original technical certification incomplete'),
      ('all-token candidate','alltoken.py','FullWeightLlamaOracle.suffix_hidden','full weight F.linear(cache.keys,W)+residual, all downstream layers','oracle-work cached suffix and post-selection physical official observer','SOURCE_CONFIRMED','No physical-versus-cached parity rerun'),
      ('signed full-vocab KL','alltoken.py','signed_forward_kl','FP32 logsoftmax then signed FP64 p0*(logp0-logp), no clamp','S64 doc means and Dev128','CPU_SCALAR_REDUCTION_CONSISTENT','Teacher full payload prior identity reused'),
      ('gradient sweep','alltoken.py','FullWeightLlamaOracle.kl','one S64 sweep at own WN; separate from observer','9 new sweeps + B1 saved gradient reuse','SOURCE_AND_COUNTERS_CONFIRMED','FD/direct-gradient NOT_ESTABLISHED'),
      ('Polyak/Armijo','optimizer.py','optimize','eta=L/chi, .5 halving, 1 gradient/8 trials, first accepted','20 trials, 10 accepted; actual p recomputed from CPU stored tensors','CPU_STORED_ARITHMETIC_CONSISTENT','No optimizer rerun or new trial'),
      ('individual Current/Past guard','binding.py','quality_ok','each new NLL+1e-4; exact strict/pair success-ID subset; no plateau','14 saved guard row panels independently checked','CPU_RESELECTION_CONSISTENT','Past64 is not all past requests'),
      ('response invariant','runtime.py','invariant','DK/leak/logit/NLL and exact strict/pair bounds','10 accepted invariant receipts checked against fixed ceilings','STORED_SCALAR_THRESHOLDS_CONSISTENT','DK/full logits cannot be independently recomputed without missing inputs/forward'),
      ('Past received ledger','sequential_state.py','past64','latest raw(subject,relation), exclude current overwrite, stable SHA, no success filter','all 10 CP ledgers -> exact Past IDs','CPU_INDEPENDENT_ID_REPLAY_PASS','No earlier unobserved state inferred'),
      ('selected history once','sequential_runner.py','run','selected/native fallback -> finalizer once, candidate inner zero','10 history receipts + 10 M tensors + 9 entry links','CPU_STORED_STATE_CONSISTENT','Selected finalizer keys not stored; M addition not independently rerun'),
      ('selection before P/N/Dev','sequential_runner.py','observe_batch','seal+durable commit then official observers; restore W/RNG','selection hash + compatibility + observer restoration receipt','SOURCE_AND_STORED_GUARDS_CONFIRMED','No repeated physical evaluation'),
      ('full numerical validation','sequential_runner.py','run','user T skip and user M-to-S advance','validation-status and terminal','SKIPPED_USER_DIRECTED','FD/direct-gradient/teacher/noop parity NOT_ESTABLISHED'),
    ]
    rows=[]
    for requirement,file,func,rule,evidence,level,limit in spec:
        p=core/file;tree=ast.parse(p.read_text());parts=func.split('.');nodes=tree.body
        for part in parts:
            node=next(n for n in nodes if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==part);nodes=node.body
        rows.append(dict(requirement=requirement,rule=rule,file=str(p),function=func,line=node.lineno,source_sha256=sha(p),evidence=evidence,status=level,limit=limit))
    csvout(out/'source-conformance.csv',rows);csvout(out/'source-input-inventory.csv',checks)
    write(out/'input-manifest.json',dict(execution=lock['execution']['head'],tree=lock['execution']['tree'],lock=member(ROOT/'execution.lock.json'),
        source_archive=a,review_envelope=member(repo/'messages/head/2026-09-18-sh4-enfc-sequential-enf-completed-review.md'),
        full_read=read(scratch/'full-read-reuse.json'),model_payload='PRIOR_SEAL_REUSE_NOT_REHASHED_OR_LOADED',
        teacher_payload='PRIOR_SEAL_REUSE_NO_REGENERATION',source_inventory='source-input-inventory.csv',sibling_raw_access=False))
    scheduler=dict(job='50071_1',raw_job='50073',owner='janghj',name='odeedit_enfc_S4_s4',state='COMPLETED',exit='0:0',
        start_KST='2026-09-18T15:01:07+09:00',end_KST='2026-09-18T19:58:01+09:00',allocated_GPU_seconds=17814,
        GPU_count=1,CPUs=8,host_memory_MiB=60416,MaxRSS_batch_KiB=31869140,query_count_this_recall=1,
        evidence='Single exact child accounting; batch/extern not added',other_jobs_queried=False)
    write(out/'scheduler-receipt.json',scheduler)
    print('PROVENANCE_OK',len(checks),len(rows))

def supplement(repo,scratch):
    out=repo/REPORT;s=read(scratch/'metrics-summary.json');ts=read(scratch/'tensor-summary.json');tab=s['tables'];
    registry={r['case_id']:r['status'] for r in csv.DictReader((scratch/'active-registry.csv').open())}
    registry={int(k):v for k,v in registry.items()}
    obs=read(ARM/'B010/observers/selected-fullseen.json');rr=pairs(obs);pop=[];token=[]
    for scope in ['ALL','ACTIVE','SUPERSEDED']:
        a=[r for r in rr if scope=='ALL' or registry[r['case_id']]==scope]
        if a:pop.extend(summary(a,scope))
    for m in ('RS','PS','NS'):
        a=[r for r in rr if r['metric']==m];label='true' if m=='NS' else 'new'
        token.append(dict(metric=m,desired_label=label,correct=sum(r[label+'_correct'] for r in a),tokens=sum(r[label+'_tokens'] for r in a),
            strict=sum(r[label+'_strict'] for r in a),sequences=len(a)))
    csvout(out/'population.csv',pop);csvout(out/'token-accuracy.csv',token)
    def casebits(o):
        rr=pairs(o);bits={};gens={g['case_id']:g for g in o['generation']}
        for case in sorted({r['case_id'] for r in rr}):
            a=[r for r in rr if r['case_id']==case];R=[r for r in a if r['metric']=='RS'];P=[r for r in a if r['metric']=='PS']
            bits[case]=dict(R_strict=R[0]['new_strict'],twoP_strict=all(r['new_strict'] for r in P),
                R_twoP_strict=all(r['new_strict'] for r in R+P),R_twoP_NLL_joint=all(r['success'] for r in R+P),
                greedy32_prefix=gens[case]['target_prefix_match'])
        return bits
    finalbits=casebits(obs);writebits={}
    for b in range(1,11):writebits.update(casebits(read(ARM/f'B{b:03d}/observers/selected-current.json')))
    w5bits=casebits(read(ARM/'B005/observers/selected-fullseen.json'));stricttrans=[]
    for label,left in [('ATWRITE_to_W10',writebits),('W5_to_W10_first500',w5bits)]:
        for key in next(iter(left.values())):
            pairs_=[(v[key],finalbits[c][key]) for c,v in left.items()]
            stricttrans.append(dict(comparison=label,metric=key,denominator=len(pairs_),before=sum(a for a,b in pairs_),after=sum(b for a,b in pairs_),
                lost=sum(a and not b for a,b in pairs_),gained=sum(not a and b for a,b in pairs_)))
    csvout(out/'strict-retention.csv',stricttrans)
    # Output small loss/gain ID inventories, no prompt/target/text/generation.
    rows=list(csv.DictReader((scratch/'paired-local-rows.csv').open()));idrows=[]
    for label in ('ATWRITE_to_W10_all1000','W5_to_W10_first500','pooled_NATIVE_to_SELECTED'):
        for m in ('RS','PS','NS'):
            a=[r for r in rows if r['comparison']==label and r['metric']==m]
            for direction in ('lost','gained'):
                ids=[f"{r['case_id']}:{r['prompt_index']}" for r in a if r[direction]=='True']
                idrows.append(dict(comparison=label,metric=m,direction=direction,count=len(ids),case_prompt_ids=ids))
    csvout(out/'loss-gain-identities.csv',idrows)
    totals=defaultdict(float)
    for r in tab['work-counters']:totals[r['component'],r['unit']]+=r['value']
    cost=[dict(component=k,unit=u,value=v,nesting='WITHIN_PROGRAM_WALL_NOT_ADDITIVE_WITH_PARENT') for (k,u),v in sorted(totals.items())]
    native=[]
    for n in range(1,11):
        b=read(ARM/f'B{n:03d}/native/native-binding.json');native.append(b)
        for k in ['compute_z','compute_z_seconds','compute_ks','compute_ks_seconds','solve','solve_seconds','seconds']:
            cost.append(dict(component=f'B{n:02d}_native',unit=k,value=b['receipt'][k],nesting='REUSE_PRIOR_NOT_NEW' if n==1 else 'NATIVE_INCLUSIVE_OVERLAP'))
    term=read(ARM/'TERMINAL.json')
    for k,v in term['timing'].items():cost.append(dict(component='program_timing',unit=k,value=v,nesting='INCLUSIVE_DO_NOT_SUM_NESTED'))
    cost.append(dict(component='scheduler',unit='allocated_GPU_seconds',value=17814,nesting='PARENT_ONLY'))
    cost.append(dict(component='program',unit='wall_seconds',value=term['wall_seconds'],nesting='NOT_ALLOCATION_UTILIZATION'))
    csvout(out/'cost-components.csv',cost)
    # Sealed historical N4 table only; no old scheduler/raw expansion.
    base=repo/'experiment-reports/servers/server4/local-z-adaptive-allocation-seq1000-2026-09-16-v1/completed-review-20260917-v1'
    br=list(csv.DictReader((base/'final-seven-arm.csv').open()));n4=[r for r in br if r.get('arm',r.get('Arm'))=='N4']
    write(out/'baseline-reuse.json',dict(status='REFERENCE_ONLY_NOT_NEW_MATCHED_S_ARM',source=member(base/'final-seven-arm.csv'),rows=n4,
        report=member(base/'diagnostic-report-ko.md'),exact_aggregate=dict(RS=999,RS_den=1000,PS=1934,PS_den=2000,NS=8026,NS_den=10000),
        shared=['model revision','cold W0/zeroM4','method seed20260916','fixed first1000/order','cold context token capsule','FP32/eager/TF32off','native L4 BLUE hparams','canonical inequalities'],
        differences=['execution/controller wrapper source','new EN-F correction and Past guard','validation/observer schedule','cost protocol'],
        paired_baseline_raw_this_review='NOT_REAUDITED_NO_SYNTHETIC_PAIRED_ROWS',own_native='same-state EN-F shadow not independent N4 chain'))
    summaryextra=dict(population=pop,token=token,cost=cost,historical_N4=n4,
        native_new_fit=sum(b['native_fit_new_calls'] for b in native),native_new_target=sum(b['native_target_new_calls'] for b in native),
        correction_counters={k:sum(r[k] for r in tab['compute']) for k in tab['compute'][0] if k not in ['batch','wall_cumulative','peak_GPU_bytes','host_peak_KiB']},
        max_GPU_bytes=max(r['peak_GPU_bytes'] for r in tab['compute']),max_host_KiB=max(r['host_peak_KiB'] for r in tab['compute']))
    write(scratch/'supplement-summary.json',summaryextra)
    print('SUPPLEMENT_OK',summaryextra['native_new_fit'],summaryextra['native_new_target'])

def plots(repo,scratch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=read(scratch/'metrics-summary.json')['tables'];t=read(scratch/'tensor-summary.json')['batches']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for m,c in [('RS','#2563eb'),('PS','#c2410c'),('NS','#047857')]:
        a=[r for r in s['batch-current'] if r['role']=='selected' and r['metric']==m]
        ax[0,0].plot([r['batch'] for r in a],[r['percent'] for r in a],marker='o',label=m,color=c)
    ax[0,0].set(title='Actual selected Current100 (different W at each batch)',xlabel='Batch',ylabel='Percent');ax[0,0].legend()
    for role,c in [('native','#64748b'),('selected','#7c3aed')]:
        a=[r for r in s['generic'] if r['panel']=='S64' and r['role']==role]
        ax[0,1].plot([r['batch'] for r in a],[r['loss'] for r in a],marker='o',label='S64 '+role,color=c)
    ax[0,1].set(title='Controller S64 KL (not official NS)',xlabel='Batch',ylabel='KL');ax[0,1].legend()
    ax[1,0].bar([r['batch'] for r in t],[100*r['correction_native_ratio'] for r in t],color='#7c3aed')
    ax[1,0].set(title='Actual correction / native Frobenius norm',xlabel='Batch',ylabel='Percent')
    a=[r for r in s['paired'] if r['comparison']=='W5_to_W10_first500'];x=list(range(3))
    ax[1,1].bar([v-.18 for v in x],[r['lost'] for r in a],width=.36,label='Lost',color='#b91c1c')
    ax[1,1].bar([v+.18 for v in x],[r['gained'] for r in a],width=.36,label='Gained',color='#15803d')
    ax[1,1].set(xticks=x,xticklabels=[f"{r['metric']} / {r['denominator']}" for r in a],title='W5 to W10: identical first500 prompts',ylabel='Prompt count');ax[1,1].legend()
    dest=repo/REPORT/'figures';dest.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest/'en-f-observations.png',dpi=160,metadata={'Software':'EN-F CPU review'});plt.close(fig)
    write(dest/'figure-provenance.json',dict(code=member(Path(__file__)),inputs=[member(scratch/'metrics-summary.json'),member(scratch/'tensor-summary.json')],
        output=member(dest/'en-f-observations.png'),new_scientific_evaluation=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--scratch',required=True,type=Path);p.add_argument('--phase',choices=['provenance','supplement','plots'],required=True)
    a=p.parse_args();globals()[a.phase](a.repo,a.scratch)
