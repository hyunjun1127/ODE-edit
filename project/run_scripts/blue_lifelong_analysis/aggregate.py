"""All recorded checkpoints, current cohorts, transitions, tails and native costs."""
from collections import defaultdict
import itertools,math,re
import numpy as np
from .common import *
from .metrics import reduce_eval,reduce_rows,preferred

def transition(a,b):
    assert len(a)==len(b)
    assert [(r['case_id'],r['prompt_index'],r['identity']) for r in a]==[(r['case_id'],r['prompt_index'],r['identity']) for r in b]
    aa=np.array([r['success'] for r in a]);bb=np.array([r['success'] for r in b]);n=len(a)
    return dict(denominator=n,before_num=int(aa.sum()),after_num=int(bb.sum()),retained=int((aa&bb).sum()),lost=int((aa&~bb).sum()),gained=int((~aa&bb).sum()),both_failed=int((~aa&~bb).sum()),conditional_loss_den=int(aa.sum()),conditional_loss_rate=float((aa&~bb).sum()/aa.sum()) if aa.sum() else 'UNDEFINED_NO_PRIOR_SUCCESS',paired_rate_delta=float(bb.mean()-aa.mean()))

def main(out):
    s=sample();ids=[r['case_id'] for r in s['records']];ordinal={i:j for j,i in enumerate(ids)}
    records={r['case_id']:r for r in s['records']}
    current=[];cumulative=[];age=[];cohorts=[];changes=[];layers=[];costs=[];trans=[];outliers=[];overwrite=[];online=[];final_data={};states={};logmembers=[]
    for cell,arm in enumerate(ARMS):
        r=root(cell);atwrite={tag:[] for tag in MULT};prev=None;prev_b=0;pathwork=defaultdict(float)
        # Only extract scalar lines, never publish prompt/target-bearing optimizer text.
        logs=sorted((attempt(cell)/'logs').glob('*.out'))
        log=next(p for p in logs if (p.name=='main-39307.out' if cell==0 else p.name==f'main-39283_{cell}.out'))
        logmembers.append(dict(arm=arm,path=str(log),bytes=log.stat().st_size,sha256=sha(log)))
        errors=[]
        with log.open() as f:
            for line in f:
                m=re.match(r'^z error tensor\(([-+0-9.eE]+)',line)
                if m:errors.append(float(m.group(1)))
        rt=read(r/'runtime.json');ls=rt['spec']['layers'];assert len(errors)==100*len(ls),(arm,len(errors))
        for b in range(1,101):
            br=r/f'B{b:03d}';c=read(br/'commit.json');ev=read(br/'current.json');ob=read(br/'native-observation.json')
            assert ev['weight_state']==c['endpoint']['weights'] and ev['cache_sha256']==c['endpoint']['cache']
            assert ev['before_after_exact'] and not ev['evaluator_controller_influence']
            assert ev['request_order']==digest(ids[(b-1)*100:b*100])
            red=reduce_eval(ev,ids[(b-1)*100:b*100],arm,b,'CURRENT_B100');current+=red
            for tag in MULT:atwrite[tag]+=ev['metrics'][tag]['rows']
            for li,(key,u) in enumerate(c['layer_updates'].items()):
                l=int(key.split('.')[2]);pathwork[l]+=u['norm']
                z=[x['norm'] for x in ob['z'] if x['layer']==l]
                layers.append(dict(arm=arm,batch=b,layer=l,update_norm=u['norm'],squared_norm=u['squared_norm'],relative_norm=u['relative_norm'],magnitude_share=u['magnitude_share'],sum_batch_net_lengths=pathwork[l],
                                   native_prewrite_z_error_printed=errors[(b-1)*len(ls)+ls.index(l)],native_prewrite_z_error_precision='stdout rounded scalar; not post-write realization',**{'target_norm_'+k:v for k,v in stats(z).items()}))
            costs.append(dict(arm=arm,batch=b,**{k:c[k] for k in ['edit_seconds','target_seconds','key_seconds','solve_seconds','evaluation_seconds','peak_gpu_bytes','compute_z','solve_calls','history_append_passes']},
                              edit_other_inclusive_seconds=c['edit_seconds']-c['target_seconds']-c['key_seconds']-c['solve_seconds'],checkpoint_bytes=c['checkpoint']['bytes'] if c['checkpoint'] else 0,
                              snapshot_history_seconds='NOT_RECORDED_SEPARATELY',forward_backward_JVP_counts='NOT_RECORDED'))
            if b not in SCHEDULE:continue
            full=read(br/'seen-full.json');assert full['state']==c['endpoint'] and full['current_rows_reused'] and full['evaluation_type']=='CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS'
            reds=reduce_eval(full,ids[:b*100],arm,b,'CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS');cumulative+=reds
            for tag,mult in MULT.items():
                rr=full['metrics'][tag]['rows'];assert rr[-100*mult:]==ev['metrics'][tag]['rows']
                online_r=reduce_rows(atwrite[tag],tag);online.append(dict(arm=arm,batch=b,scope='ONLINE_AT_WRITE_PREFIX_DISTINCT_STATES',**online_r))
                trans.append(dict(arm=arm,batch=b,metric=tag,comparison='AT_WRITE_TO_CHECKPOINT',**transition(atwrite[tag],rr)))
                if prev:
                    old=prev['metrics'][tag]['rows'];common=rr[:len(old)]
                    trans.append(dict(arm=arm,batch=b,previous_batch=prev_b,metric=tag,comparison='PREVIOUS_CHECKPOINT_SAME_OLD_PREFIX',**transition(old,common)))
                n=b*100
                for label,lo,hi in [('early',0,n//5),('middle',n//5,4*n//5),('recent',4*n//5,n)]:
                    v=rr[lo*mult:hi*mult]
                    age.append(dict(arm=arm,batch=b,stratum=label,ordinal_start=lo,ordinal_end_exclusive=hi,definition='relative seen-prefix first20/middle60/last20 percent',**reduce_rows(v,tag)))
                for cohort in range(1,b+1):
                    v=rr[(cohort-1)*100*mult:cohort*100*mult];a=atwrite[tag][(cohort-1)*100*mult:cohort*100*mult]
                    cohorts.append(dict(arm=arm,batch=b,cohort=cohort,age_batches=b-cohort,metric=tag,numerator=sum(x['success'] for x in v),denominator=len(v),rate=sum(x['success'] for x in v)/len(v),
                                        new_nll_mean=float(np.mean([x['new_nll'] for x in v])),true_nll_mean=float(np.mean([x['true_nll'] for x in v])),margin_mean=float(np.mean([x['margin'] for x in v])),**{'atwrite_'+k:x for k,x in transition(a,v).items()}))
                # Stored prompt-pair outliers only; no raw prompt text.
                for side in ('new','true'):
                    worst=sorted(rr,key=lambda x:(-x[side+'_nll'],x['identity']))[:5]
                    for rank,x in enumerate(worst,1):outliers.append(dict(arm=arm,batch=b,metric=tag,target=side,rank=rank,case_hash=digest(x['case_id']),prompt_hash=x['identity'],cohort=ordinal[x['case_id']]//100+1,prompt_index=x['prompt_index'],nll=x[side+'_nll'],margin=x['margin'],success=x['success'],finite=True))
            if b==100:
                final_data[arm]=full;states[arm]=full['state']
                # Request-level partition; possible conflicts are retained, not attributed as causes.
                last_by_group=defaultdict(list)
                for rec in s['records']:last_by_group[rec['subject_relation_group']].append(rec)
                rr=full['metrics']['RS']['rows'];a=atwrite['RS']
                partitions=defaultdict(list)
                for x in rr:
                    rec=records[x['case_id']];later=[q for q in last_by_group[rec['subject_relation_group']] if q['ordinal']>rec['ordinal']]
                    label='later_different_target_same_subject_relation' if any(q['target_new_sha256']!=rec['target_new_sha256'] for q in later) else 'later_same_target_only' if later else 'no_later_same_subject_relation'
                    partitions[label].append(ordinal[x['case_id']])
                for label,ix in partitions.items():overwrite.append(dict(arm=arm,category=label,**transition([a[i] for i in ix],[rr[i] for i in ix])))
            prev=full;prev_b=b
            print('CHECKPOINT_REDUCED',arm,b,flush=True)
        # Numeric chronological changes, including tails. No outcome-dependent thresholds.
        for tag in MULT:
            rr=[x for x in cumulative if x['arm']==arm and x['metric']==tag]
            pairs=list(zip(rr,rr[1:]))+[(next(x for x in rr if x['batch']==10),rr[-1])]
            for a,z in pairs:
                row=dict(arm=arm,metric=tag,from_batch=a['batch'],to_batch=z['batch'],scope='CHANGING_SEEN_PREFIX_DESCRIPTIVE_DELTA')
                for key,val in z.items():
                    if key not in ['batch'] and isinstance(val,(int,float)) and key in a:row[key+'_delta']=val-a[key]
                changes.append(row)
        print('AGGREGATE_ARM_DONE',arm,flush=True)
    paired=[]
    for method in ['MEMIT','AlphaEdit']:
        names=[x for x in ARMS if x.startswith(method+'_')]
        for a,b in itertools.combinations(names,2):
            for tag in MULT:paired.append(dict(method=method,arm_before=a,arm_after=b,metric=tag,scope='FINAL_W100_DIFFERENT_STATES_SAME_PROMPTS',**transition(final_data[a]['metrics'][tag]['rows'],final_data[b]['metrics'][tag]['rows'])))
    for name,rows in [('current-metrics',current),('cumulative-metrics',cumulative),('age-strata-metrics',age),('cohort-retention',cohorts),('checkpoint-deltas',changes),('layer-updates',layers),('batch-cost',costs),('prompt-transitions',trans),('paired-final-transitions',paired),('outlier-ledger',outliers),('overwrite-strata',overwrite),('online-at-write',online)]:csvwrite(out/(name+'.csv'),rows)
    save(out/'log-input-inventory.json',logmembers)
    save(out/'aggregation-receipt.json',dict(checkpoints=72,cumulative_request_state_rows=sum(SCHEDULE)*100*6,current_request_state_rows=60000,final_request_rows=60000,unique_requests=10000,
        final_included_in_cumulative=True,current_checkpoint_rows_reused=True,independently_checked_prompt_rows=(sum(SCHEDULE)*100+10000)*6*13,
        no_imputation=True,no_replay=True,rows={k:len(v) for k,v in [('cumulative',cumulative),('current',current),('age',age),('cohort',cohorts)]},
        missing=['W0 performance','all-seen noncheckpoint batches','post-write realization residual/vector','native action in covariance/history metric','full weight byte preservation outside selected layers','separate history/snapshot timing']))

if __name__=='__main__':main(cli().out)
