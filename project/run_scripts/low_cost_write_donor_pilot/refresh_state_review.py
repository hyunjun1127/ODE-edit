"""Bounded CPU verification of terminal chunk/optimizer/history evidence."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from .runtime import file_sha,save
from .refresh_review import (policy_roots,checked_ref,load_policy_chain,
                             load_committed_evaluation)
from .review_metrics import write_csv,require,panel_digest,DATA_SHA

def tensor_hash_pair(value):
 """Two independent conventions; neither digest can stand in for the other."""
 from .fitting import tensor_sha as raw_sha
 from project.run_scripts.baseline_mechanism_first.fixtures import tensor_sha as state_sha
 return dict(raw_sha256=raw_sha(value),state_sha256=state_sha(value),
             dtype=str(value.dtype),shape=list(value.shape))

def dual_hash_bridge(value,raw_expected,state_expected):
 result=tensor_hash_pair(value)
 require(result['raw_sha256']==raw_expected,'RAW_TENSOR_BRIDGE')
 require(result['state_sha256']==state_expected,'HEADER_TENSOR_BRIDGE')
 return result

def reference_raw_endpoint(commit,raw_entry_w,raw_entry_m,raw_projector):
 """Reused native/REFIT receipts form their own raw-byte hash ledger."""
 first=commit['first_fit'];second=commit['second_fit'];mat=commit['materialization']
 require(first['entry_weight_sha256']==raw_entry_w and first['history_sha256']==raw_entry_m and first['projector_sha256']==raw_projector,'REFERENCE_RAW_FIRST_FIT')
 if second is not None:
  require(second['entry_weight_sha256']==mat['weight_sha256'] and second['history_sha256']==raw_entry_m and second['projector_sha256']==raw_projector,'REFERENCE_RAW_SECOND_FIT')
  return second['endpoint_weight_sha256']
 require(first['endpoint_weight_sha256']==mat['weight_sha256'],'REFERENCE_NATIVE_FULL_MATERIALIZATION')
 return mat['weight_sha256']

def chunk_counters(e,summary,cap,old=None,frozen_reuse=False):
 """Crosscheck actual counters against saved loss history, never infer them."""
 updates=summary['actual_adam_updates'];losses=summary['target_loss_evaluations']
 require(type(updates) is int and type(losses) is int and 0<=updates<=cap and losses>=0,'TARGET_COUNTER_SCHEMA')
 previous_updates=0 if old is None else old['t']
 previous_losses=0 if old is None else old['target_loss_evaluations']
 require(e['t']==e['actual_adam_updates']==previous_updates+updates,'CHUNK_COUNTER_INCREMENT')
 require(e['target_loss_evaluations']==previous_losses+losses,'CUMULATIVE_LOSS_COUNTER')
 if frozen_reuse:
  require(cap==0 and updates==losses==0,'FROZEN_COUNTERS')
 else:
  require(losses==updates+1==len(e['losses']),'ACTUAL_LOSS_OBSERVATIONS')
  require([r['iteration'] for r in e['losses']]==list(range(losses)),'LOSS_ITERATION_ORDER')
  require(all(math.isfinite(r[k]) for r in e['losses'] for k in ['total','nll','kl','regularizer']),'FINITE_LOSS_COMPONENTS')
  if summary['stop_reason']=='LOSS_BELOW_0_05':require(e['losses'][-1]['total']<.05,'EARLY_STOP_SOURCE_RULE')
  else:require(summary['stop_reason']=='CHUNK_QUOTA_EXHAUSTED' and updates==cap and e['losses'][-1]['total']>=.05,'QUOTA_STOP_SOURCE_RULE')
  require(summary['target_forwards']==losses and summary['target_backwards']==updates,'TARGET_FB_COUNTS')
  require(summary['quota_transferred']==0 and summary['unused_quota']==cap-updates,'NO_QUOTA_TRANSFER')
 return updates,losses

def verify(attempt,out):
 import torch
 from .fitting import tensor_sha
 torch.set_num_threads(8)
 a=Path(attempt);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 lock,roots=policy_roots(a)
 dataset=Path(lock['dataset_root'])/'counterfact.json';require(file_sha(dataset)==DATA_SHA,'FIXED_DATASET_SHA')
 records=json.loads(dataset.read_text())
 files={};links=[];chunks=[];checkpoints=[];cost=[];target_counts=[];bridges=[];raw_links=[]
 def checked(ref):
  return checked_ref(ref,files)
 prepared=torch.load(checked(lock['prepared']),map_location='cpu',weights_only=True,mmap=True)
 # Projector source is shared immutable input: reuse its locked whole-file
 # audit, and inspect/hash only the actual physical L4 slice for this bridge.
 projector=torch.load(lock['projector'],map_location='cpu',weights_only=True,mmap=True)[:1]
 common={}
 for name,value,expected in [('W4',prepared['weights'][4],lock['common_state']['weights']['4']),('M4',prepared['M4'],lock['common_state']['M4']),('P4',projector,lock['common_state']['P4'])]:
  pair=tensor_hash_pair(value);require(pair['state_sha256']==expected,'COMMON_HEADER_BRIDGE')
  common[name]=pair
  bridges.append(dict(policy='ALL_SIX_COMMON_ENTRY',batch=50,tensor=name,**pair,source='CPU_PREPARED_OR_PHYSICAL_P4_SLICE'))
 del prepared,projector
 for policy,root in roots.items():
  terminal,commits=load_policy_chain(lock,root,policy,files)
  previous=terminal['entry_state']
  previous_raw_w=common['W4']['raw_sha256'];previous_raw_m=common['M4']['raw_sha256']
  policy_adam=policy_losses=policy_solves=policy_optimized=0
  for batch,c in zip(range(51,61),commits):
   require(c['batch']==batch and c['entry']==previous,'W_M_RNG_NEXT_ENTRY')
   evaluation=load_committed_evaluation(root,batch,c,files);del evaluation
   links.append(dict(policy=policy,batch=batch,entry_previous_exact=True,first_common_entry=batch==51,continued_link=batch>51))
   previous=c['endpoint'];require(c['evaluation_nonmutation'] is True,'EVAL_NONMUTATION')
   require(len(c['history'])==1 and c['history'][0]['layer']==4 and c['history'][0]['history_append']==1,'HISTORY_APPEND_ONCE')
   require(c['history'][0]['compute_ks']==1 and c['history'][0].get('compute_z',0)==0 and c['history'][0].get('solve',0)==0,'FINALIZATION_ONLY_KEYS')
   history=c['history'][0]
   require(history['before_sha256']==previous_raw_m,'RAW_HISTORY_PREVIOUS_ENDPOINT')
   require(c['history_counts']['4']==batch-50,'CHRONOLOGICAL_HISTORY_COUNT')
   if policy in lock['reference_reuse']:
    raw_w=reference_raw_endpoint(c,previous_raw_w,previous_raw_m,common['P4']['raw_sha256'])
    require(history['weight_sha256']==raw_w,'REFERENCE_RAW_FINALIZER_WEIGHT')
    raw_links.append(dict(policy=policy,batch=batch,raw_entry_W=previous_raw_w,raw_endpoint_W=raw_w,raw_entry_M=previous_raw_m,raw_endpoint_M=history['after_sha256'],bridge='COMMON_ENTRY_NEW_CPU; REFERENCE_CP_AUDIT_REUSED'))
    previous_raw_w=raw_w;previous_raw_m=history['after_sha256']
    cost.append(dict(policy=policy,batch=batch,reused=True,online_seconds=c['policy_instrumented_online_seconds'],evaluation_seconds=c['evaluation_seconds'],new_spending=0))
    continue
   cfg=next(p for p in lock['policies'] if p['id']==policy)
   require(len(c['subwrites'])==len(cfg['write_gammas']),'SUBWRITE_COUNT')
   snapshots={};total_adam=total_losses=0;previous_subwrite=c['entry'];raw_subwrite_w=previous_raw_w
   current=records[(batch-1)*100:batch*100]
   require(len(current)==100 and len({r['case_id'] for r in current})==100,'B100_SOURCE_CARDINALITY')
   normalized=[]
   for rec in current:
    request=deepcopy(rec['requested_rewrite'])
    if not request['target_new']['str'].startswith(' '):request['target_new']['str']=' '+request['target_new']['str']
    normalized.append(panel_digest(request))
   for chunk,r in enumerate(c['subwrites']):
    sub=json.loads(checked(r).read_text());fit=sub['fit']
    require(sub['policy']==policy and sub['batch']==batch and sub['chunk']==chunk,'SUBWRITE_ASSOCIATION')
    require(sub['entry']==previous_subwrite,'SUBWRITE_CONTINUITY')
    previous_subwrite=sub['endpoint']
    require(sub['gamma']==cfg['write_gammas'][chunk] and sub['cap']==cfg['target_update_caps'][chunk],'POLICY_CAP_GAMMA')
    require(fit['compute_z']==0 and fit['target_supply']==100 and fit['solve']==1 and sub['inner_history_appends']==0,'NATIVE_WRITER_COUNTERS')
    require(sub['entry']['M4']==c['entry']['M4']==sub['endpoint']['M4'],'INNER_HISTORY_IMMUTABLE')
    require(sub['entry']['P4']==sub['endpoint']['P4']==c['entry']['P4'],'PROJECTOR_IMMUTABLE')
    require(sub['entry']['contexts']==sub['endpoint']['contexts']==c['entry']['contexts'],'CONTEXT_IMMUTABLE')
    require(fit['entry_weight_sha256']==raw_subwrite_w and fit['history_sha256']==previous_raw_m and fit['projector_sha256']==common['P4']['raw_sha256'],'RAW_FIT_INPUT_BINDING')
    raw_subwrite_w=sub['materialization']['weight_sha256']
    require(fit['request_sha256']==normalized and len(fit['target_sha256'])==100,'NATIVE_NORMALIZED_REQUEST_ORDER')
    if sub['gamma']==1.:require(fit['endpoint_weight_sha256']==raw_subwrite_w,'GAMMA_ONE_RAW_EXACT_ENDPOINT')
    tensor_ref=sub['tensors'];loaded=torch.load(checked(tensor_ref),map_location='cpu',weights_only=True,mmap=True)
    require(loaded['entry']==sub['entry'] and loaded['endpoint']==sub['endpoint'] and loaded['policy']==policy and loaded['batch']==batch and loaded['chunk']==chunk,'SUBWRITE_TENSOR_ASSOCIATION')
    require(loaded['current_y'].shape==loaded['residual'].shape==(4096,100),'Y_R_SCHEMA')
    require(loaded['targets'].shape==(100,4096) and loaded['actual_delta'].shape==loaded['native_candidate_delta'].shape==(4096,14336),'WEIGHT_TARGET_SCHEMA')
    require(torch.equal(loaded['targets'].T-loaded['current_y'],loaded['residual']),'CURRENT_Y_RESIDUAL')
    require(tensor_sha(loaded['current_y'])==fit['current_y_sha256'] and tensor_sha(loaded['residual'])==fit['residual_sha256'],'READOUT_RESIDUAL_BINDING')
    for name in ['actual_delta','native_candidate_delta','current_y','residual','targets']:
     require(loaded[name].dtype==torch.float32 and bool(torch.isfinite(loaded[name]).all()),'FINITE_FP32_SUBWRITE')
    require(len(sub['request_chunk_states'])==100,'B100_REQUEST_CHUNK_COUNT')
    for i,rr in enumerate(sub['request_chunk_states']):
     value=torch.load(checked(rr),map_location='cpu',weights_only=True)
     require(value['request_index']==i and value['batch']==batch and value['chunk']==chunk and value['policy']==policy,'REQUEST_CHUNK_ASSOCIATION')
     require(value['case_id']==current[i]['case_id'] and value['request_sha256']==panel_digest(current[i]['requested_rewrite']),'FIXED_REQUEST_IDENTITY')
     e=value['evidence'];summary=value['summary']
     require(value['input_state']==sub['entry'],'TARGET_BARRIER_INPUT_STATE')
     require(torch.equal(e['a0']+e['u'],e['Z']),'ABSOLUTE_TARGET_FORMULA')
     require(torch.equal(loaded['targets'][i],e['Z']),'TARGET_SOLVER_ASSOCIATION')
     require(fit['target_sha256'][i]==tensor_sha(e['Z']),'SUPPLIED_TARGET_HASH')
     require(e['teacher_sha256']==tensor_sha(e['teacher']),'TEACHER_IDENTITY')
     require(e['t']==e['actual_adam_updates'],'ACTUAL_ADAM_COUNTER')
     for name in ['u','Z','a0','aj','m','v','teacher']:
      require(e[name].dtype==torch.float32 and bool(torch.isfinite(e[name]).all()),'FINITE_TARGET_ADAM')
     if chunk==0:
      require(torch.count_nonzero(e['initial_u']).item()==0 and not e['initial_adam'],'REQUEST_COLD_OPTIMIZER')
      require(torch.equal(e['a0'],e['aj']),'CHUNK_ZERO_ANCHOR_CONNECTION')
     else:
      old=snapshots[i]
      require(torch.equal(e['a0'],old['a0']) and torch.equal(e['teacher'],old['teacher']),'FIXED_ENTRY_ANCHOR_TEACHER')
      if policy.startswith('FROZEN'):
       require(all(torch.equal(e[k],old[k]) for k in ['Z','u','m','v']) and e['t']==old['t'],'FROZEN_OWN_TARGET_AND_OPTIMIZER')
      else:
       require(torch.equal(e['initial_u'],old['u']),'U_CARRY')
       opt=e['initial_adam']
       if old['t']:
        require(torch.equal(opt['exp_avg'],old['m']) and torch.equal(opt['exp_avg_sq'],old['v']) and int(opt['step'].item())==old['t'],'ADAM_M_V_T_CARRY')
       else:require(not opt,'ZERO_STEP_ADAM_UNINITIALIZED')
       require(e['t']==old['t']+summary['actual_adam_updates'],'CHUNK_COUNTER_INCREMENT')
     frozen_reuse=chunk>0 and policy.startswith('FROZEN')
     updates,losses=chunk_counters(e,summary,sub['cap'],snapshots.get(i),frozen_reuse)
     total_adam+=updates;total_losses+=losses;policy_optimized+=int(not frozen_reuse)
     target_counts.append(dict(policy=policy,batch=batch,chunk=chunk,request_index=i,adam=updates,loss=losses,optimized_target_chunk=not frozen_reuse,frozen_reuse=frozen_reuse,stop=summary.get('stop_reason',summary.get('status')),clamp_hits=summary.get('clamp_hits',0)))
     snapshots[i]={k:e[k] for k in ['u','m','v','t','a0','teacher','Z','target_loss_evaluations']}
     del value,e
    chunks.append(dict(policy=policy,batch=batch,chunk=chunk,requests=100,gamma=sub['gamma'],cap=sub['cap'],history_appends=0,solve=1,current_residual='CPU_EXACT_PASS',actual_delta_norm=float(loaded['actual_delta'].double().norm()),native_candidate_delta_norm=float(loaded['native_candidate_delta'].double().norm()),write_seconds=sub['write_seconds']))
    del loaded
   require(total_adam==c['actual_adam_updates'] and total_losses==c['target_loss_evaluations'],'BATCH_TARGET_TOTALS')
   require(previous_subwrite['weights']==c['endpoint']['weights'] and previous_subwrite['P4']==c['endpoint']['P4'] and previous_subwrite['contexts']==c['endpoint']['contexts'],'FINAL_SUBWRITE_COMMIT_BINDING')
   require(history['weight_sha256']==raw_subwrite_w,'RAW_FINALIZER_LAST_MATERIALIZATION')
   raw_links.append(dict(policy=policy,batch=batch,raw_entry_W=previous_raw_w,raw_endpoint_W=raw_subwrite_w,raw_entry_M=previous_raw_m,raw_endpoint_M=history['after_sha256'],bridge='CPU_CHECKPOINT' if c['checkpoint'] else 'DUAL_LEDGERS_ONLY_NO_NEW_TENSOR_BRIDGE'))
   previous_raw_w=raw_subwrite_w;previous_raw_m=history['after_sha256']
   policy_adam+=total_adam;policy_losses+=total_losses;policy_solves+=len(c['subwrites'])
   require(bool(c['checkpoint'])==(batch in [51,55,60]),'CHECKPOINT_SCHEDULE_COMPLETENESS')
   if c['checkpoint']:
    ref=c['checkpoint'];cp=torch.load(checked(ref),map_location='cpu',weights_only=True,mmap=True)
    require(set(cp['weights'])=={4},'CP_SELECTED_WEIGHT_KEYS')
    for name,value,raw_expected,state_expected in [('W4',cp['weights'][4],previous_raw_w,c['endpoint']['weights']['4']),('M4',cp['M4'],previous_raw_m,c['endpoint']['M4'])]:
     pair=dual_hash_bridge(value,raw_expected,state_expected)
     bridges.append(dict(policy=policy,batch=batch,tensor=name,**pair,source='CPU_CHECKPOINT_SELECTED_TENSOR'))
    require(all(t.dtype==torch.float32 and bool(torch.isfinite(t).all()) for t in [cp['weights'][4],cp['M4']]),'CP_FINITE_FP32')
    require(panel_digest(cp['contexts'])==c['endpoint']['contexts'] and panel_digest(cp['rng'])==c['endpoint']['rng'],'CP_CONTEXT_RNG')
    require(cp['metadata']['next_batch']==batch+1 and cp['metadata']['batch']==batch and cp['metadata']['policy']==policy,'CP_NEXT_INDEX')
    require(cp['metadata']['state']==c['endpoint'] and cp['metadata']['seen_ids']==[r['case_id'] for r in records[:batch*100]],'CP_STATE_SEEN_IDS')
    checkpoints.append(dict(policy=policy,batch=batch,**files[str(Path(ref['path']))],CPU_reload='PASS',GPU_continuation='NOT_TESTED'))
    del cp
   cost.append(dict(policy=policy,batch=batch,reused=False,online_seconds=c['instrumented_online_seconds'],target_including_nested_IO_seconds=c['target_seconds_including_request_state_IO'],native_writer_seconds=c['native_writer_seconds'],history_seconds=c['history_seconds'],materialization_seconds=c['materialization_seconds'],evaluation_seconds=c['evaluation_seconds'],snapshot_seconds=c.get('checkpoint_save_reload_seconds','NOT_RECORDED'),nested_IO_seconds=c['nested_request_subwrite_IO_seconds'],new_spending='YES'))
  require(previous==terminal['terminal_state'],'TERMINAL_STATE')
  if policy not in lock['reference_reuse']:
   require(terminal['actual_adam_updates']==policy_adam and terminal['target_loss_evaluations']==policy_losses and terminal['fit_solve_total']==policy_solves and terminal['request_target_chunks']==policy_optimized and terminal['history_counts']['4']==10,'TERMINAL_ACTUAL_COUNTERS')
 optimized=sum(r['optimized_target_chunk'] for r in target_counts)
 require(len(links)==60 and sum(r['continued_link'] for r in links)==54 and len(chunks)==120 and len(checkpoints)==12 and len(target_counts)==12000 and optimized==8000,'COMPLETE_LOGICAL_AND_NEW_SCOPE')
 inventory=[{k:v for k,v in row.items() if k!='_stat'} for row in files.values()]
 for name,rows in [('state-links',links),('raw-writer-history-links',raw_links),('tensor-hash-bridges',bridges),('subwrites',chunks),('checkpoints',checkpoints),('compute-ledger',cost),('target-counters',target_counts),('new-raw-rehash-inventory',inventory)]:write_csv(out/(name+'.csv'),rows)
 receipt=dict(status='CPU_STATE_AND_COUNTER_REDUCTION_PASS',logical_batches=len(links),logical_links=sum(r['continued_link'] for r in links),new_subwrites=len(chunks),new_checkpoints=len(checkpoints),stored_request_chunk_states=len(target_counts),optimized_target_chunks=optimized,frozen_target_reuse_states=len(target_counts)-optimized,target_supplies=len(target_counts),rehash_files=len(files),rehash_bytes=sum(r['bytes'] for r in files.values()),rehash_scope='newly_rehashed_new_payload_plus_common_prepared_and_reused_reference_commit_evaluation_metadata',tensor_hash_conventions=dict(writer_targets_teacher='SHA256_RAW_CONTIGUOUS_BYTES',state_ledger='SHA256_DTYPE_SHAPE_HEADER_THEN_RAW_BYTES',cross_convention_bridges='COMMON_PREPARED_W4_M4_P4_AND_NEW_CP51_55_60_ONLY',missing_intermediate_tensor_bridge='NOT_TESTED_DUAL_LEDGER_CHAIN_CHECKS_ONLY'),reference_prior_CP_audit='EXACT_EXECUTION_PUBLICATION_REUSED_NOT_NEW_FULL_REHASH',GPU_continuation='NOT_TESTED',native_model_off_on_parity='TECHNICAL_I1_GATE_SEPARATE',delta_replay='NOT_TESTED',scientific_promotion=False)
 print(json.dumps(save(out/'state-review-receipt.json',receipt)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--out',required=True);args=p.parse_args();verify(args.attempt,args.out)
