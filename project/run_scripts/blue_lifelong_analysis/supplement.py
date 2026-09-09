"""Read-only historical provenance and numerical comparison, not experiment replay."""
from collections import defaultdict
import itertools,json,re
import numpy as np
from .common import *
from project.run_scripts.blue_fivearm_analysis.common import BLUE_ROOTS

def main(out):
    repo=Path(__file__).resolve().parents[3]
    old=repo/'experiment-reports/servers/server4/blue-alphaedit-fivearm-sequential1000-review-2026-09-07-v1'
    assert sha(old/'factual-report-ko.md')=='7c10f5c3797cca8b90d5db4de79583a56f3020041fe67a156ac24b8caeb67da9'
    manifest=read(old/'analysis-manifest.json');oldinv=[]
    for m in manifest['members']:
        p=old/m['path'];assert sha(p)==m['sha256'] and p.stat().st_size==m['bytes']
        oldinv.append(m)
    comparisons=[]
    for cell,key in [(1,'BLUE'),(3,'BLUE_L4_ONLY'),(5,'BLUE_L8_ONLY')]:
        oldroot=BLUE_ROOTS[key];ot=read(oldroot/'terminal.json');om={m['path']:m for m in ot['manifest_members']}
        oe=read(oldroot/'B10/seen-full.json');ne=read(root(cell)/'B010/seen-full.json')
        checked_member(oldroot,'B10/seen-full.json',om)
        ort=read(oldroot/'runtime.json');nrt=read(root(cell)/'runtime.json')
        for tag in MULT:
            a=oe['metrics'][tag];b=ne['metrics'][tag]
            assert [(x['case_id'],x['prompt_index'],x['identity']) for x in a['rows']]==[(x['case_id'],x['prompt_index'],x['identity']) for x in b['rows']]
            comparisons.append(dict(arm=ARMS[cell],metric=tag,prior_1k_num=a['numerator'],lifelong_1k_num=b['numerator'],denominator=b['denominator'],rate_delta=b['rate']-a['rate'],
                 NLL_pair_rows_exact=a['rows']==b['rows'],old_evaluation_sha256=sha(oldroot/'B10/seen-full.json'),new_evaluation_sha256=sha(root(cell)/'B010/seen-full.json'),
                 model_revision_equal=ort.get('model_revision')==nrt.get('model_revision'),old_model_revision=ort.get('model_revision','NOT_RECORDED'),new_model_revision=nrt['model_revision'],
                 old_runtime_sha256=sha(oldroot/'runtime.json'),new_runtime_sha256=sha(root(cell)/'runtime.json'),comparison='same first1000/order; independent run; exact row equality assessed rather than assumed'))
    csvwrite(out/'prior-1k-comparison.csv',comparisons);save(out/'prior-report-input.json',dict(path=str(old),report_sha256=sha(old/'factual-report-ko.md'),members=len(oldinv),member_root=digest(oldinv)))
    b=BASE/'local/blue-lifelong-b100x100';ex=[];prov=[]
    paths=[b/'attempt-v1/main-submission.json',b/'attempt-v1/smoke-submission.json',b/'attempt-checkpoint-r2/amendment.json',b/'attempt-cell0-checkpoint-r3/old-lineage-exclusion.json',b/'attempt-cell0-checkpoint-r3/submission.json',b/'attempt-checkpoint-r2/submission.json',b/'attempt-cell0-checkpoint-r3/uniform-schedule-receipt.json']
    for p in paths:
        d=read(p);prov.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),receipt=d))
    d=read(paths[3]);assert (d['completed_prefix_requests'],d['allocated_elapsed_seconds'],d['new_canonical_denominator'])==(600,2766,0)
    for m in d['commits']:assert sha(m['identity']['path'])==m['identity']['sha256']
    ex.append(dict(job='39183_0/39218',status=d['status'],canonical_denominator=0,committed_prefix_requests=600,uncommitted_batch=7,allocated_seconds=2766,allocated_GPU_hours=2766/3600,receipt_path=str(paths[3]),receipt_sha256=sha(paths[3])))
    for j in range(1,6):ex.append(dict(job=f'39183_{j}',status='PENDING_CANCEL_REPLACED_BY_CHECKPOINT_AMENDMENT',canonical_denominator=0,committed_prefix_requests=0,allocated_seconds=0,evidence=str(paths[2])))
    ex.append(dict(job='39172',status='SMOKE_SKIPPED_USER_DIRECTED_PENDING_CANCELLED',canonical_denominator=0,committed_prefix_requests=0,allocated_seconds=0,evidence=str(paths[0])))
    csvwrite(out/'technical-exclusions.csv',ex);save(out/'operational-receipt-inputs.json',prov)
    findings=[];lock=read(attempt(0)/'execution.lock.json');blue=Path(lock['blue_root']);helper=Path(lock['source_root'])
    paths={blue/'AlphaEdit/AlphaEdit_main.py':['if hparams.blue:','z_layer = layer','resid = targets','cache_c[i,:,:] +=','return model,'],
           blue/'memit/memit_main.py':['execute_memit_blue','resid = targets  #','mom2_update_weight * cov.double()','w[...] +=','weights[k][...] = weights_copy[k]'],
           attempt(0)/'lifelong/runtime.py':['model=AutoModel','tok.padding_side','heavy=','past=evaluate','assert signature','prior=content(endpoint)','FINAL_W0_RESTORE'],
           attempt(0)/'lifelong/method.py':['indices=','projector=','state=','model_name','apply_memit'],
           helper/'project/run_scripts/blue_alphaedit_sequential_comparison/evaluation.py':['success=','before = signature','if before !=','microbatch_size=16'],
           helper/'project/run_scripts/alphaedit_strength_neutral_barrier/evaluator.py':['def evaluate_pairs','def sequence_metrics','attention_mask','position_ids','cross_entropy']}
    for p,terms in paths.items():
        body=p.read_text().splitlines()
        for i,line in enumerate(body,1):
            if any(t in line for t in terms):findings.append(dict(path=str(p),sha256=sha(p),line=i,expression=line.strip()))
    csvwrite(out/'source-findings.csv',findings)
    # Exact arithmetic differences, no p-value or causal attribution.
    cum=csvread(out/'cumulative-metrics.csv');pair=[]
    for method in ['MEMIT','AlphaEdit']:
        for a,z in itertools.combinations([x for x in ARMS if x.startswith(method+'_')],2):
            for batch in SCHEDULE:
                for tag in MULT:
                    x=next(r for r in cum if r['arm']==a and int(r['batch'])==batch and r['metric']==tag);y=next(r for r in cum if r['arm']==z and int(r['batch'])==batch and r['metric']==tag)
                    q=dict(method=method,arm_before=a,arm_after=z,batch=batch,metric=tag,before_num=x['numerator'],after_num=y['numerator'],denominator=x['denominator'])
                    for k in ['rate']+[f'{side}_nll_prompt_{stat}' for side in ['new','true'] for stat in ['mean','median','p90','max']]+[f'margin_prompt_{v}' for v in ['mean','median','p90','max']]:q[k+'_delta']=float(y[k])-float(x[k])
                    pair.append(q)
    csvwrite(out/'paired-checkpoint-deltas.csv',pair)
    layers=csvread(out/'layer-updates.csv');current=csvread(out/'current-metrics.csv');assoc=[]
    def rank(v):
        order=np.argsort(v,kind='stable');r=np.empty(len(v),float)
        for u in np.unique(v):ix=np.flatnonzero(np.asarray(v)==u);r[ix]=np.mean(np.flatnonzero(np.isin(order,ix)))
        return r
    for arm in ARMS:
        for scope,metrics,batches in [('CURRENT',current,range(1,101)),('CUMULATIVE',cum,SCHEDULE)]:
            for tag in MULT:
                for feature in ['update_norm','sum_batch_net_lengths','native_prewrite_z_error_printed']:
                    x=[];y=[]
                    for b0 in batches:
                        ll=[q for q in layers if q['arm']==arm and int(q['batch'])==b0]
                        x.append(sum(float(q[feature]) for q in ll));y.append(float(next(q['rate'] for q in metrics if q['arm']==arm and int(q['batch'])==b0 and q['metric']==tag)))
                    rho=float(np.corrcoef(rank(x),rank(y))[0,1]) if np.std(x)>0 and np.std(y)>0 else 'UNDEFINED_CONSTANT'
                    assoc.append(dict(arm=arm,scope=scope,metric=tag,feature=feature,n_checkpoints_or_batches=len(x),spearman=rho,claim='DESCRIPTIVE_TEMPORALLY_DEPENDENT_NO_CAUSALITY_NO_PVALUE'))
    csvwrite(out/'mechanism-performance-associations.csv',assoc)

if __name__=='__main__':main(cli().out)
