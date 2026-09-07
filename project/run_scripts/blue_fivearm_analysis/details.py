"""Pure CPU scalar aggregations; missing fields remain unavailable."""
import argparse
from collections import defaultdict
from itertools import combinations
from .common import *
from .metrics import reduce_eval, normalize_j, TAGS, MULT

def published(repo,name):
 out=[]
 for r in csvread(repo/REVIEW_REL/name):
  if r.get('alias')=='llama3-8b-inst' and r.get('arm') in ['O_NATIVE','JV_NATIVE','L8_ONLY_NATIVE','PRE_EDIT_ORIGINAL_W0']:
   q=dict(r);q['arm']={'JV_NATIVE':'JVP','L8_ONLY_NATIVE':'JVP_L8'}.get(r['arm'],r['arm']);out.append(q)
 return out

def retention(rows,atwrite,arm,b,cohort_by_id,overwrite):
 out=[]
 for cohort in range(1,b+1):
  rs=[r for r in rows if cohort_by_id[r['case_id']]==cohort];assert len(rs)==100
  a=[atwrite[r['case_id']] for r in rs];old=[bool(r['success']) for r in a];now=[bool(r['success']) for r in rs]
  out.append(dict(arm=arm,batch=b,cohort=cohort,age=b-cohort,canonical_denominator=100,current_success=sum(now),
   at_write_success=sum(old),initially_failed=100-sum(old),at_write_success_now_failure=sum(o and not n for o,n in zip(old,now)),
   prior_failure_now_recovery=sum(not o and n for o,n in zip(old,now)),forgetting_conditional_denominator=sum(old),
   prior_failure_denominator=100-sum(old),margin_mean=float(np.mean([r['true_nll']-r['new_nll'] for r in rs])),
   overwrite_candidate_count=sum(r['case_id'] in overwrite.get(b,set()) for r in rs),
   nonoverwrite_forgetting_num=sum(o and not n and r['case_id'] not in overwrite.get(b,set()) for o,n,r in zip(old,now,rs))))
 return out

def transitions(a,b,arm,comparison,tag,batch):
 assert [(r['identity'],r['case_id'],r['prompt_index']) for r in a]==[(r['identity'],r['case_id'],r['prompt_index']) for r in b]
 old=[r['success'] for r in a];new=[r['success'] for r in b]
 return dict(arm=arm,reference=comparison,metric=tag,batch=batch,denominator=len(a),reference_success=sum(old),arm_success=sum(new),
  success_to_loss=sum(x and not y for x,y in zip(old,new)),failure_to_recovery=sum(not x and y for x,y in zip(old,new)),
  stable_success=sum(x and y for x,y in zip(old,new)),stable_failure=sum(not x and not y for x,y in zip(old,new)),
  paired_identity_hash=digest([r['identity'] for r in a]),unit='prompt; descriptive, no independence-based significance')

def run(repo):
 out=repo/OUT_REL;sample=read(repo/REF_REL/'sample.lock.json');ids=[r['case_id'] for r in sample['records']]
 cohorts={r['case_id']:r['batch_index'] for r in sample['records']}
 overwrites={};collision=[]
 for b in range(1,11):
  seen=sample['records'][:100*b];groups=defaultdict(list)
  for r in seen:groups[r['subject_relation_group']].append(r)
  overwrites[b]=set()
  for group,rr in groups.items():
   for x,y in combinations(rr,2):
    if x['target_new_sha256']!=y['target_new_sha256']:
     overwrites[b].add(x['case_id']);collision.append(dict(batch=b,group_hash=group,older_case_hash=digest(x['case_id']),newer_case_hash=digest(y['case_id']),older_cohort=x['batch_index'],newer_cohort=y['batch_index'],classification='METADATA_OVERWRITE_CANDIDATE_NOT_EXCLUSION'))
 current=published(repo,'current_batch_metrics.csv');seen=published(repo,'seen_prefix_metrics.csv')
 cohort_rows=published(repo,'retention_cohort_metrics.csv');cost=published(repo,'cost_by_arm.csv');costbatch=published(repo,'cost_by_batch.csv')
 actions=published(repo,'batch_layer_actions.csv');partitions=published(repo,'endpoint_partitions.csv')
 pre=[r for r in published(repo,'final_metrics.csv') if r['arm']=='PRE_EDIT_ORIGINAL_W0']
 allrewrite=[];trans=[];finalraw={};pool=[];availability=[];source_records=[]
 for arm,root in BLUE_ROOTS.items():
  atwrite={};allcurrent={t:[] for t in TAGS};rt=read(root/'runtime.json');terminal=read(root/'terminal.json')
  pre_raw=read(root/'pre-edit.json');pre.append(reduce_eval(pre_raw,arm,0,'W0_FULL1000',ids))
  finalraw[arm]=read(root/'B10/seen-full.json')
  elapsed={'BLUE':3980,'BLUE_L4_ONLY':3661,'BLUE_L8_ONLY':3510}[arm]
  cr=dict(arm=arm,process_seconds=terminal['seconds'],scheduler_elapsed_seconds=elapsed,allocated_gpu_hours=elapsed/3600,
   model_load_seconds=rt['model_load_seconds'],preedit_seconds=read(root/'pre-edit-timing.json')['seconds'],
   edit_seconds=0,evaluation_seconds=0,target_seconds=0,key_seconds=0,compute_z=terminal['compute_z'],
   solve_instrumented=0 if arm!='BLUE' else NA,source_expected_solves=10*len(rt['hparams']['layers']),history_append_passes=10,
   forward=NA,backward=NA,main_JVP=0,solve_seconds=NA,history_seconds=NA,materialization_seconds=NA,
   local_storage_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),gpu=rt['gpu'],server='server4',
   occupancy='scheduler dedicated one-GPU allocation; utilization NOT_RECORDED')
  for b in range(1,11):
   d=root/f'B{b:02d}';cur=read(d/'current.json');co=read(d/'commit.json');ob=read(d/'native-observation.json');rw=read(d/'seen-rewrite.json')
   current.append(reduce_eval(cur,arm,b,'CURRENT_B100',ids[(b-1)*100:b*100]))
   atwrite.update({r['case_id']:r for r in cur['metrics']['RS']['rows']})
   for tag in TAGS:allcurrent[tag]+=cur['metrics'][tag]['rows']
   allrewrite.append(reduce_eval(dict(metrics={'RS':dict(rows=rw['rows'])}),arm,b,'ALL_SEEN_REWRITE',ids[:b*100]))
   cohort_rows+=retention(rw['rows'],atwrite,arm,b,cohorts,overwrites)
   if b in [1,5,10]:
    full=read(d/'seen-full.json');seen.append(reduce_eval(full,arm,b,'CHECKPOINT_ALL_SEEN',ids[:b*100]))
    for tag in TAGS:
     old=pre_raw['metrics'][tag]['rows'][:b*100*MULT[tag]]
     trans.append(transitions(old,full['metrics'][tag]['rows'],arm,'OWN_W0',tag,b))
   for k,v in co['layer_updates'].items():
    actions.append(dict(arm=arm,batch=b,layer=int(k.split('.')[2]),batch_net_norm=v['norm'],batch_net_squared=v['squared_norm'],batch_net_magnitude_share=v['magnitude_share'],relative_norm=v['relative_norm'],
     raw_native_work=NA,normalized_native_work=NA,W0_net_squared=NA,actual_step_norm_sum=NA,provenance='ACTUAL_FP32_ENTRY_ENDPOINT_DIFFERENCE'))
   cb={k:co[k] for k in ['edit_seconds','evaluation_seconds','target_seconds','key_seconds','compute_z','peak_gpu_bytes']}
   cb.update(arm=arm,batch=b,solve_calls=ob.get('solve_calls',NA),key_calls=len(ob['keys']),history_append_passes=1)
   costbatch.append(cb)
   for k in ['edit_seconds','evaluation_seconds','target_seconds','key_seconds']:cr[k]+=co[k]
   if arm!='BLUE':cr['solve_instrumented']+=ob['solve_calls']
  cr['peak_gpu_bytes']=max(read(root/f'B{b:02d}/commit.json')['peak_gpu_bytes'] for b in range(1,11))
  cr['key_calls']=sum(len(read(root/f'B{b:02d}/native-observation.json')['keys']) for b in range(1,11))
  cr['unpartitioned_edit_seconds']=cr['edit_seconds']-cr['target_seconds']-cr['key_seconds']
  cost.append(cr)
  pool.append(reduce_eval(dict(metrics={t:dict(rows=rr) for t,rr in allcurrent.items()}),arm,10,'ONLINE_OWN_BATCH_POOL_NOT_FINAL',ids))
  last=[r for r in cohort_rows if r['arm']==arm and int(r['batch'])==10]
  partitions.append(dict(arm=arm,all_denominator=1000,at_write_success=sum(r['at_write_success'] for r in last),initially_failed=sum(r['initially_failed'] for r in last),
   at_write_success_to_final_failure=sum(r['at_write_success_now_failure'] for r in last),initially_failed_to_final_recovery=sum(r['prior_failure_now_recovery'] for r in last),
   final_RS=sum(r['current_success'] for r in last),overwrite_candidates=len(overwrites[10])))
 # All ten JVP-L8 rewrite rows are local and already covered by the raw member rehash.
 for b in range(1,11):
  rw=read(JCHAIN/f'batch-{b:02d}/rewrite-retention.json')
  assert [r['case_id'] for r in rw]==ids[:100*b]
  s=stats([r['new_nll'] for r in rw]);m=stats([r['margin'] for r in rw])
  allrewrite.append(dict(arm='JVP_L8',batch=b,scope='ALL_SEEN_REWRITE',request_count=len(rw),RS_num=sum(r['success'] for r in rw),RS_den=len(rw),RS_rate=sum(r['success'] for r in rw)/len(rw),
    **{'rewrite_target_new_nll_'+k:v for k,v in s.items()},**{'rewrite_margin_'+k:v for k,v in m.items()}))
 for arm in ['O_NATIVE','JVP']:
  for b in range(1,11):
   rr=[r for r in cohort_rows if r['arm']==arm and int(r['batch'])==b]
   n=sum(int(r['current_success']) for r in rr);den=sum(int(r['canonical_denominator']) for r in rr)
   allrewrite.append(dict(arm=arm,batch=b,scope='ALL_SEEN_REWRITE_PUBLISHED_COHORT_SUM',request_count=den,RS_num=n,RS_den=den,RS_rate=n/den,
     rewrite_margin_mean=sum(float(r['margin_mean'])*int(r['canonical_denominator']) for r in rr)/den))
  # Published online summary reused, not recomputed quantiles from batch quantiles.
  for r in csvread(repo/REF_REL/'online_own_batch_metrics.csv'):
   if r.get('alias')=='llama3-8b-inst' and r['arm']==('JV_NATIVE' if arm=='JVP' else arm):pool.append(dict(r,arm=arm))
 for a,b in combinations(BLUE_ROOTS,2):
  for tag in TAGS:trans.append(transitions(finalraw[a]['metrics'][tag]['rows'],finalraw[b]['metrics'][tag]['rows'],b,a,tag,10))
 # Every final NLL/margin distribution, no quantile subtraction.
 dist=[]
 for table_name,rs in [('final',csvread(out/'final_metrics.csv')),('current',current),('seen',seen),('preedit',pre)]:
  for r in rs:
   for tag,cat in TAGS.items():
    for side in ['new','true','margin']:
     prefix=cat+'_margin_' if side=='margin' else cat+'_target_'+side+'_nll_'
     dist.append(dict(arm=r['arm'],scope=table_name,batch=r.get('batch',0),category=cat,quantity=side,
      n=r.get(tag+'_den',NA),**{k:r.get(prefix+k,NA) for k in ['mean','median','q25','q75','p90','max']}))
 age=[]
 for r in cohort_rows:
  if int(r['batch'])==10:age.append(r)
 delta=[];final=csvread(out/'final_metrics.csv')
 for a,b in combinations(final,2):
  z=dict(arm=a['arm'],reference=b['arm'],paired_request_count=1000,comparison='END_TO_END_NOT_SAME_STATE_CAUSAL')
  for t in TAGS:z[t+'_delta_pp']=100*(float(a[t+'_rate'])-float(b[t+'_rate']))
  delta.append(z)
 availability=[dict(item='Qwen BLUE',status='ORIGINAL_QWEN_CONFIG_UNAVAILABLE_NOT_RUN'),dict(item='L4 pre-run tests/smoke',status='SKIPPED_USER_DIRECTED'),
  dict(item='JVP/O server2 raw',status='NOT_AVAILABLE_GIT_PUBLICATION_ONLY'),dict(item='cross BLUE vs JVP prompt-pair transitions',status='NOT_AVAILABLE_COMMON_PAIR_HASH; request/cohort comparison only'),
  dict(item='PS/NS B2-4 B6-9 seen-prefix',status='NOT_RECORDED_NO_INTERPOLATION'),dict(item='common terminal activation / residual across BLUE and JVP',status='NOT_RECORDED; target layers/policy differ'),
  dict(item='BLUE native metric action/path/JVP counters',status='NOT_RECORDED; Frobenius not relabeled native action'),dict(item='GPU utilization / solver-only/history-only BLUE time',status='NOT_RECORDED'),
  dict(item='L4 observer projector_asset_index',status='METADATA_MISMATCH_4; actual runtime selects0; see CPU hash audit')]
 for name,rs in [('current_batch',current),('seen_prefix',seen),('allseen_rewrite',allrewrite),('retention_cohort',cohort_rows),('final_age_cohort',age),('preedit',pre),
  ('layer_action',actions),('compute',cost),('compute_by_batch',costbatch),('endpoint_partitions',partitions),('online_pool',pool),('prompt_transitions',trans),
  ('nll_margin_distributions',dist),('final_deltas',delta),('overwrite_candidates',collision),('availability',availability)]:csvwrite(out/(name+'.csv'),rs)
 # Preserve full relevant source-publication scalar tables, without raw prompts or tensors.
 for name in ['reference_failure_registry.csv','reference_neighborhood_transition_summary.csv','reference_run_registry.csv','node_mechanism.csv','batch_mechanism.csv','normalization.csv']:
  rs=published(repo,name)
  csvwrite(out/name,rs)
 print('DETAIL_TABLES_COMPLETE',len(current),len(seen),len(cohort_rows),len(dist),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);run(p.parse_args().repo)
