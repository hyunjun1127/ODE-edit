"""Reuse old six aggregate evidence, reduce new eight and pair final states."""
from collections import defaultdict
import itertools,gc
from .common import *

NAMES=['current-metrics','cumulative-metrics','age-strata-metrics','cohort-retention','checkpoint-deltas','layer-updates','batch-cost','prompt-transitions','outlier-ledger','overwrite-strata','online-at-write']

def main():
    s=sample();ids=[r['case_id'] for r in s['records']];ordinal={i:j for j,i in enumerate(ids)}
    tables={n:csvread(INHERITED/(n+'.csv')) for n in NAMES}
    for rows in tables.values():
        for r in rows:r['validation_origin']='REUSED_V3_AGGREGATE'
    bygroup=defaultdict(list)
    for rec in s['records']:bygroup[rec['subject_relation_group']].append(rec)
    for arm in NEWARMS:
        rp=root(arm);rt=read(rp/'runtime.json');ls=rt['hparams']['layers'];atwrite={tag:[] for tag in MULT};prev=None;prev_b=None;pathwork=defaultdict(float)
        for b in range(1,101):
            br=rp/f'B{b:03d}';c=read(br/'commit.json');ev=read(br/'current.json');ob=read(br/'native-observation.json')
            assert ev['weight_state']==c['endpoint']['weights'] and ev['cache_sha256']==c['endpoint']['cache']
            assert ev['before_after_exact'] and not ev['evaluator_controller_influence'] and ev['request_order']==digest(ids[(b-1)*100:b*100])
            tables['current-metrics']+=reductions(ev,ids[(b-1)*100:b*100],arm,b,'CURRENT_B100')
            for tag in MULT:atwrite[tag]+=ev['metrics'][tag]['rows']
            for key,u in c['layer_updates'].items():
                l=int(key.split('.')[2]);pathwork[l]+=u['norm'];z=[x['norm'] for x in ob['z'] if x['layer']==l]
                row=dict(arm=arm,batch=b,layer=l,update_norm=u['norm'],squared_norm=u['squared_norm'],relative_norm=u['relative_norm'],magnitude_share=u['magnitude_share'],sum_batch_net_lengths=pathwork[l],native_prewrite_z_error_printed='NOT_REDUCED_FROM_LOG')
                if z:row.update({'target_norm_'+k:v for k,v in stats(z).items()})
                else:row['target_norm_status']='NATIVE_SHARED_L8_TARGET; NOT_LAYER_LOCAL'
                tables['layer-updates'].append(row)
            tables['batch-cost'].append(dict(arm=arm,batch=b,**{k:c[k] for k in ['edit_seconds','target_seconds','key_seconds','solve_seconds','evaluation_seconds','peak_gpu_bytes','compute_z','solve_calls','history_append_passes']},edit_other_inclusive_seconds=c['edit_seconds']-c['target_seconds']-c['key_seconds']-c['solve_seconds'],checkpoint_bytes=c['checkpoint']['bytes'] if c['checkpoint'] else 0,snapshot_history_seconds='NOT_RECORDED_SEPARATELY',forward_backward_JVP_counts='NOT_RECORDED'))
            if b not in SCHEDULE:continue
            full=read(br/'seen-full.json');assert full['state']==c['endpoint'] and full['current_rows_reused']
            rr=reductions(full,ids[:b*100],arm,b,'CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS');tables['cumulative-metrics']+=rr
            for tag,mult in MULT.items():
                rows=full['metrics'][tag]['rows'];assert rows[-100*mult:]==ev['metrics'][tag]['rows']
                tables['online-at-write'].append(dict(arm=arm,batch=b,scope='ONLINE_AT_WRITE_PREFIX_DISTINCT_STATES',**reduce_rows(atwrite[tag],tag)))
                tables['prompt-transitions'].append(dict(arm=arm,batch=b,metric=tag,comparison='AT_WRITE_TO_CHECKPOINT',**transition(atwrite[tag],rows)))
                if prev:
                    old=prev['metrics'][tag]['rows'];tables['prompt-transitions'].append(dict(arm=arm,batch=b,previous_batch=prev_b,metric=tag,comparison='PREVIOUS_CHECKPOINT_SAME_OLD_PREFIX',**transition(old,rows[:len(old)])))
                n=b*100
                for name,lo,hi in [('early',0,n//5),('middle',n//5,4*n//5),('recent',4*n//5,n)]:
                    part=rows[lo*mult:hi*mult]
                    tables['age-strata-metrics'].append(dict(arm=arm,batch=b,stratum=name,ordinal_start=lo,ordinal_end_exclusive=hi,definition='relative seen-prefix first20/middle60/last20 percent',**reduce_rows(part,tag),**p99(part)))
                for cohort in range(1,b+1):
                    part=rows[(cohort-1)*100*mult:cohort*100*mult];a=atwrite[tag][(cohort-1)*100*mult:cohort*100*mult]
                    tables['cohort-retention'].append(dict(arm=arm,batch=b,cohort=cohort,age_batches=b-cohort,metric=tag,numerator=sum(x['success'] for x in part),denominator=len(part),rate=np.mean([x['success'] for x in part]),new_nll_mean=np.mean([x['new_nll'] for x in part]),true_nll_mean=np.mean([x['true_nll'] for x in part]),margin_mean=np.mean([x['margin'] for x in part]),**{'atwrite_'+k:v for k,v in transition(a,part).items()}))
                for side in ['new','true']:
                    for rank,x in enumerate(sorted(rows,key=lambda r:(-r[side+'_nll'],r['identity']))[:5],1):
                        tables['outlier-ledger'].append(dict(arm=arm,batch=b,metric=tag,target=side,rank=rank,case_hash=digest(x['case_id']),prompt_hash=x['identity'],cohort=ordinal[x['case_id']]//100+1,prompt_index=x['prompt_index'],nll=x[side+'_nll'],margin=x['margin'],success=x['success'],finite=True))
            prev=full;prev_b=b
            print('AGGREGATE_CP',arm,b,flush=True)
        partitions=defaultdict(list)
        for rec in s['records']:
            later=[q for q in bygroup[rec['subject_relation_group']] if q['ordinal']>rec['ordinal']]
            name='later_different_target_same_subject_relation' if any(q['target_new_sha256']!=rec['target_new_sha256'] for q in later) else 'later_same_target_only' if later else 'no_later_same_subject_relation'
            partitions[name].append(rec['ordinal'])
        for name,ix in partitions.items():tables['overwrite-strata'].append(dict(arm=arm,category=name,**transition([atwrite['RS'][i] for i in ix],[full['metrics']['RS']['rows'][i] for i in ix])))
        for tag in MULT:
            r=[x for x in tables['cumulative-metrics'] if x['arm']==arm and x['metric']==tag]
            for a,z in list(zip(r,r[1:]))+[(next(x for x in r if x['batch']==10),r[-1])]:
                d=dict(arm=arm,metric=tag,from_batch=a['batch'],to_batch=z['batch'],scope='CHANGING_PREFIX_DESCRIPTIVE')
                for key,v in z.items():
                    if key!='batch' and isinstance(v,(int,float)) and key in a:d[key+'_delta']=v-a[key]
                tables['checkpoint-deltas'].append(d)
        del atwrite,full,prev;gc.collect()
    for name,rows in tables.items():csvwrite(OUT/(name+'.csv'),rows)
    # Only final files of old chains reread for new paired comparisons/p99; no old CP rehash.
    finals={};dist=[]
    for arm in ARMS:
        full=read(root(arm)/'B100/seen-full.json');finals[arm]=full
        dist+=reductions(full,ids,arm,100,'FINAL_W100_FULL10000')
    paired=[];locality=[];fixedcohorts=[]
    for method in ['MEMIT','AlphaEdit']:
        names=[a for a in ARMS if family(a)==method]
        for a,b in itertools.combinations(names,2):
            for tag in MULT:
                aa=finals[a]['metrics'][tag]['rows'];bb=finals[b]['metrics'][tag]['rows']
                paired.append(dict(method=method,arm_before=a,arm_after=b,metric=tag,scope='FINAL_DIFFERENT_STATES_EXACT_SAME_PROMPTS',**transition(aa,bb)))
                if tag=='NS':
                    for state,selector in [('all',lambda x,y:True),('loss',lambda x,y:x['success'] and not y['success']),('recovery',lambda x,y:not x['success'] and y['success'])]:
                        selected=[(x,y) for x,y in zip(aa,bb) if selector(x,y)]
                        row=dict(method=method,arm_before=a,arm_after=b,transition=state,denominator=len(selected),unit='same neighborhood prompts; correlated within request')
                        for side in ['new','true']:
                            vals=[y[side+'_nll']-x[side+'_nll'] for x,y in selected]
                            if vals:row.update({side+'_nll_delta_'+k:v for k,v in stats(vals).items()});row[side+'_nll_increased']=sum(v>0 for v in vals)
                        locality.append(row)
    for a,full in finals.items():
        for tag,mult in MULT.items():
            for name,lo,hi in [('first100',0,100),('first500',0,500),('first1000',0,1000),('early',0,2000),('middle',2000,8000),('recent',8000,10000),('last1000',9000,10000)]:
                rr=full['metrics'][tag]['rows'][lo*mult:hi*mult]
                fixedcohorts.append(dict(arm=a,stratum=name,ordinal_start=lo,ordinal_end_exclusive=hi,**reduce_rows(rr,tag),**p99(rr)))
    csvwrite(OUT/'final-distributions.csv',dist);csvwrite(OUT/'paired-final-transitions.csv',paired);csvwrite(OUT/'paired-locality-nll-changes.csv',locality);csvwrite(OUT/'final-fixed-cohorts.csv',fixedcohorts)
    save(OUT/'aggregation-receipt.json',dict(chains=14,current_request_state_rows=140000,cumulative_request_state_rows=sum(SCHEDULE)*100*14,checkpoint_evaluations=12*14,unique_requests=10000,final_request_state_rows=140000,final_included_in_cumulative=True,current_rows_reused=True,imputation=0,new_evaluator=0,w0_new_arm_pairing='NOT_AVAILABLE_LOCAL_PROMPT_W0_PUBLICATION; existing six W0 transitions reused only',table_rows={k:len(v) for k,v in tables.items()}))

if __name__=='__main__':main()
