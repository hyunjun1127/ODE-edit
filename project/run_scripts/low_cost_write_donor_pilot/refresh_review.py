"""Independent CPU NLL reduction for six logical policies and four new chains.

The reader never imports a model/evaluator or treats reused subsets as new
observations. Missing/active endpoints are not imputed. Run only on terminals.
"""
import argparse
import json
import math
import stat
from pathlib import Path
from statistics import fmean

from .review_metrics import (aggregate,desired,distribution,pairs,key,require,
                            write_csv,file_sha,MULT,DATA_SHA,panel_digest,mmlu_prediction)
from .seq_review_metrics import load_eval,validate,annotation,subset,dump

ORDER=['N4','REFIT4','FROZEN2','I2','FROZEN4','I4']
CONTRASTS=[('FROZEN2','REFIT4'),('FROZEN2','I2'),('FROZEN4','I4'),('I2','I4')]+[('N4',p) for p in ORDER[1:]]
EVALUATOR_LAYOUT='HISTORICAL_MICROBATCH16_MANUAL_LEFT_PADDING_NO_POSITION_OVERRIDE'

def checked_ref(ref, cache):
 """Full bytes/hash once, conflicting duplicate refs and changed stat rejected."""
 path=Path(ref['path']);key=str(path)
 require(path.is_absolute(),'REFERENCE_ABSOLUTE_PATH')
 st=path.lstat();identity=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
 require(stat.S_ISREG(st.st_mode) and not path.is_symlink(),'RAW_REGULAR_FILE')
 require(type(ref['bytes']) is int and ref['bytes']==st.st_size,'RAW_FILE_SIZE')
 if key in cache:
  old=cache[key]
  require((old['bytes'],old['sha256'])==(ref['bytes'],ref['sha256']),'CONFLICTING_FILE_REFERENCE')
  require(old['_stat']==identity,'RAW_CHANGED_AFTER_HASH')
 else:
  require(file_sha(path)==ref['sha256'],'RAW_FILE_SHA')
  after=path.lstat()
  require(identity==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'RAW_CHANGED_DURING_HASH')
  cache[key]=dict(path=key,bytes=st.st_size,sha256=ref['sha256'],_stat=identity)
 return path

def terminal_commit_refs(term):
 require(term['status']=='TEN_SEQUENTIAL_BATCHES_COMPLETE','NOT_TERMINAL')
 require(term['batches']==list(range(51,61)),'TERMINAL_BATCH_SCHEDULE')
 refs=term['commits']
 require(len(refs)==10 and len({r['path'] for r in refs})==10,'TERMINAL_COMMIT_CARDINALITY')
 return refs

def l4_state(state):
 return dict(weights={'4':state['weights']['4']},M4=state['M4'],P4=state['P4'],contexts=state['contexts'],rng=state['rng'])

def load_policy_chain(lock,root,policy,cache):
 path=root/'terminal.json'
 if policy in lock['reference_reuse']:
  ref=lock['reference_reuse'][policy]['terminal']
  require(Path(ref['path'])==path,'REFERENCE_TERMINAL_PATH')
 else:ref=dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha(path))
 term=json.loads(checked_ref(ref,cache).read_text())
 require(term.get('policy',term.get('arm'))==policy,'TERMINAL_POLICY')
 expected_lock=(lock['reference_lock']['sha256'] if policy in lock['reference_reuse']
                else lock['_analysis_execution_lock_sha256'])
 require(term['source_lock_sha256']==expected_lock,'TERMINAL_SOURCE_LOCK')
 require(l4_state(term['entry_state'])==l4_state(lock['common_state']),'COMMON_ENTRY_CAPSULE')
 commits=[];previous=term['entry_state']
 for batch,ref in zip(range(51,61),terminal_commit_refs(term)):
  require(Path(ref['path'])==root/f'B{batch:03d}/commit.json','COMMIT_CANONICAL_PATH')
  commit=json.loads(checked_ref(ref,cache).read_text())
  require(commit.get('policy',commit.get('arm'))==policy and commit['batch']==batch,'COMMIT_POLICY_BATCH')
  require(commit['entry']==previous,'COMMIT_CONTINUITY')
  previous=commit['endpoint'];commits.append(commit)
 require(previous==term['terminal_state'],'TERMINAL_COMMITTED_ENDPOINT')
 return term,commits

def load_committed_evaluation(root,batch,commit,cache):
 path=root/f'B{batch:03d}/evaluation.json'
 require(Path(commit['evaluation']['path'])==path,'EVALUATION_CANONICAL_PATH')
 checked_ref(commit['evaluation'],cache)
 d=load_eval(path,batch)
 require(commit['evaluation_nonmutation'] is True and d['endpoint_state']==commit['endpoint'],'EVALUATION_COMMITTED_ENDPOINT')
 return d

def validate_panel(doc, records, endpoint):
 require(doc['evaluator_layout']==EVALUATOR_LAYOUT,'CANONICAL_MB16_LAYOUT')
 return validate(doc,records,endpoint)

def policy_roots(attempt):
 a=Path(attempt);lock=json.loads((a/'execution.lock.json').read_text())
 lock['_analysis_execution_lock_sha256']=file_sha(a/'execution.lock.json')
 require(lock['array_mapping']==['FROZEN2','I2','FROZEN4','I4'] and set(lock['reference_reuse'])=={'N4','REFIT4'},'SIX_POLICY_MAPPING')
 roots={p:Path(r['root']) for p,r in lock['reference_reuse'].items()}
 roots.update({p:a/'output'/f'cell-{i}' for i,p in enumerate(lock['array_mapping'])})
 return lock,roots

def strict_two_p(rows):
 cases={}
 for r in rows:
  group=cases.setdefault(r['case_id'],{})
  require(r['prompt_index'] not in group,'DUPLICATE_P_PROMPT')
  require(type(r['new_strict']) is bool,'STRICT_BOOLEAN_SCHEMA')
  group[r['prompt_index']]=r['new_strict']
 require(all(set(v)=={0,1} for v in cases.values()),'P_REQUEST_CARDINALITY')
 return dict(two_P_request_strict_n=sum(all(v.values()) for v in cases.values()),two_P_request_strict_d=len(cases))

def first(attempt,out,records):
 lock,roots=policy_roots(attempt);rows=[];files={}
 for policy in ORDER:
  root=roots[policy];terminal=root/'terminal.json'
  require(terminal.exists(),f'NOT_TERMINAL_{policy}')
  term,commits=load_policy_chain(lock,root,policy,files)
  p=root/'B060/evaluation.json';doc=load_committed_evaluation(root,60,commits[-1],files)
  metrics=validate_panel(doc['fullseen'],records[:6000],doc['endpoint_state_sha256'])
  row=dict(policy=policy,status='TERMINAL_METRICS_VERIFIED_STATE_AUDIT_SEPARATE',requests=6000,
           result_reuse=policy in lock['reference_reuse'],evaluation_sha256=file_sha(p))
  for m,v in metrics.items():row.update({f'{m}_n':v['numerator'],f'{m}_d':v['prompt_denominator'],f'{m}_percent':100*v['rate']})
  rows.append(row)
 for row in rows:
  for m in MULT:
   row[f'{m}_delta_n_N4']=row[f'{m}_n']-rows[0][f'{m}_n']
   row[f'{m}_delta_pp_N4']=row[f'{m}_percent']-rows[0][f'{m}_percent']
 write_csv(out/'first-final-table.csv',rows);dump(out/'first-final-table.json',rows)
 return rows

def reduce(attempt,out,records):
 from .refresh_uncertainty import request_cluster_bootstrap
 lock,roots=policy_roots(attempt)
 hist=[records[i] for i in lock['historical_ordinals']]
 wiki_rows=json.loads(Path(lock['wiki_panel']).read_text())['rows']
 wiki_expected=[(r['ordinal'],panel_digest(r['input_ids']),len(r['input_ids'])-1) for r in wiki_rows]
 mmlu_rows=json.loads(Path(lock['mmlu100']).read_text())
 dev=[mmlu_rows[i] for i in lock['mmlu_development_indices']]
 rates=[];tails=[];checks=[];inputs=[];transitions=[];cohorts=[];general=[];docs={};files={}
 def summarize(rr,p,b,pop,m,group='ALL'):
  ident=dict(policy=p,batch=b,population=pop,metric=m,group=group)
  agg=aggregate(rr,m)
  if m=='PS':agg.update(strict_two_p(rr))
  rates.append(dict(ident,**agg))
  clusters={}
  for row in rr:clusters.setdefault(row['case_id'],[]).append(row)
  for field in ['new_nll','true_nll','desired_margin']:
   fn=lambda r:desired(r,m) if field=='desired_margin' else r[field]
   for unit,values in [('PROMPT',[fn(r) for r in rr]),('REQUEST_CLUSTER_MEAN',[fmean(fn(r) for r in group) for group in clusters.values()])]:
    tails.append(dict(ident,field=field,unit=unit,**distribution(values)))
 for policy in ORDER:
  docs[policy]={}
  term,commits=load_policy_chain(lock,roots[policy],policy,files)
  for batch in range(51,61):
   path=roots[policy]/f'B{batch:03d}/evaluation.json';d=load_committed_evaluation(roots[policy],batch,commits[batch-51],files);docs[policy][batch]=d
   inputs.append(dict(policy=policy,batch=batch,path=str(path),bytes=path.stat().st_size,sha256=file_sha(path),reused=policy in lock['reference_reuse']))
   pops={'current':records[(batch-1)*100:batch*100],'historical':hist}
   if batch in (55,60):pops['suffix']=records[5000:batch*100]
   if batch==60:pops.update(fullseen=records[:6000],entry_old=records[:5000])
   ann=annotation(records,batch*100)
   for pop,selected in pops.items():
    validate_panel(d[pop],selected,d['endpoint_state_sha256'])
    for m in MULT:
     rr=d[pop]['metrics'][m]['rows'];summarize(rr,policy,batch,pop,m)
     if pop=='historical':require(all(r['historical_status']['status']==ann[r['case_id']] for r in rr),'HISTORICAL_ACTIVE_DEFINITION')
     if pop!='current':
      for status in ['ACTIVE_TARGET','SUPERSEDED','UNKNOWN_RELATION']:
       summarize([r for r in rr if ann[r['case_id']]==status],policy,batch,pop,m,status)
     checks.append(dict(policy=policy,batch=batch,population=pop,metric=m,n=len(rr),duplicate=0,nonfinite=0,imputation=0,identity_nll='PASS'))
   source='fullseen' if batch==60 else ('suffix' if batch==55 else None)
   for reuse in d['reuse']:
    require(reuse['added_forwards']==0 and reuse['source_endpoint_state_sha256']==d['endpoint_state_sha256'],'REUSE_FORWARD_AND_ENDPOINT')
   if source:
    for pop in (['current','historical','entry_old','suffix'] if batch==60 else ['current']):
     for m in MULT:
      lookup={key(r):r for r in d[source]['metrics'][m]['rows']}
      for r in d[pop]['metrics'][m]['rows']:
       require(all(r[f]==lookup[key(r)][f] for f in ['new_nll','true_nll','success','new_strict','true_strict','new_token_correct','true_token_correct','new_token_count','true_token_count']),'SUBSET_EXACT_REUSE')
   if batch in (55,60):
    for m in MULT:
     rr=subset(d['suffix']['metrics'][m]['rows'],[r['case_id'] for r in records[5000:5500]])
     summarize(rr,policy,batch,'first_suffix500',m)
     if 'first_suffix500' in d:
      require([key(r) for r in rr]==[key(r) for r in d['first_suffix500']['metrics'][m]['rows']],'FIRST500_IDENTITY')
      require(all(a['new_nll']==z['new_nll'] and a['true_nll']==z['true_nll'] for a,z in zip(rr,d['first_suffix500']['metrics'][m]['rows'])),'FIRST500_REUSE')
   if batch==60:
    for m in MULT:
     full=d['fullseen']['metrics'][m]['rows']
     for cb in range(1,61):
      rr=subset(full,[r['case_id'] for r in records[(cb-1)*100:cb*100]])
      cohorts.append(dict(policy=policy,cohort_batch=cb,metric=m,**aggregate(rr,m)))
    wr=d['wiki']['rows'];mr=d['mmlu']['rows']
    require(len(wr)==128 and len(mr)==32,'GENERAL_PANEL_CARDINALITY')
    require([(r['ordinal'],r['input_sha256'],r['predicted_tokens']) for r in wr]==wiki_expected,'WIKI_INPUT_MASK')
    require(all(math.isfinite(r['nll']) for r in wr),'WIKI_NONFINITE')
    require(math.isclose(fmean(r['nll'] for r in wr),d['wiki']['mean_nll'],abs_tol=1e-12),'WIKI_MEAN')
    require([r['row_sha256'] for r in mr]==[panel_digest(r) for r in dev],'MMLU_DEV32_IDENTITY')
    for row,source in zip(mr,dev):
     require(len(row['alternative_nll'])==4,'MMLU_ALTERNATIVE_NLL_CARDINALITY')
     pred=mmlu_prediction(row['alternative_probability'])
     require(row['gold']==source['answer'] and row['prediction']==pred and row['correct']==(pred==source['answer']),'MMLU_ALTERNATIVE_INTEGER')
     require(row['tie_or_underflow_invalid']==(pred==-1),'MMLU_INVALID')
     require(all(math.isfinite(n) and math.isclose(p,math.exp(-n),rel_tol=1e-6,abs_tol=1e-40) for p,n in zip(row['alternative_probability'],row['alternative_nll'])),'MMLU_NLL_PAIR')
    correct=sum(r['correct'] for r in mr);invalid=sum(r['prediction']==-1 for r in mr)
    require(d['mmlu']['denominator']==32 and d['mmlu']['correct']==correct and d['mmlu']['invalid']==invalid and d['mmlu']['accuracy']==correct/32,'MMLU_STORED_AGGREGATE')
    general.append(dict(policy=policy,batch=batch,wiki_nll=fmean(r['nll'] for r in wr),wiki_tokens=sum(r['predicted_tokens'] for r in wr),wiki_sequences=128,mmlu_correct=sum(r['correct'] for r in mr),mmlu_invalid=sum(r['prediction']==-1 for r in mr),mmlu_d=32))
  for m in MULT:
   atwrite=[r for b in range(51,61) for r in docs[policy][b]['current']['metrics'][m]['rows']]
   end=docs[policy][60]['suffix']['metrics'][m]['rows'];ann=annotation(records,6000)
   summarize(atwrite,policy,60,'ONLINE_AT_WRITE_1000_DIFFERENT_STATES',m)
   for status in ['ALL','ACTIVE_TARGET','SUPERSEDED','UNKNOWN_RELATION']:
    left=atwrite if status=='ALL' else [r for r in atwrite if ann[r['case_id']]==status]
    right=end if status=='ALL' else [r for r in end if ann[r['case_id']]==status]
    transitions.append(dict(contrast='AT_WRITE_TO_W60',before=policy,after=policy,population='suffix1000',metric=m,group=status,**pairs(left,right,m)))
   for cb in range(51,61):
    before=docs[policy][cb]['current']['metrics'][m]['rows'];after=subset(end,[r['case_id'] for r in records[(cb-1)*100:cb*100]])
    transitions.append(dict(contrast='AT_WRITE_TO_W60',before=policy,after=policy,population=f'cohort{cb}',metric=m,group='ALL',future_batch_exposures=60-cb,**pairs(before,after,m)))
   before=docs[policy][55]['suffix']['metrics'][m]['rows'];after=subset(end,[r['case_id'] for r in records[5000:5500]])
   transitions.append(dict(contrast='W55_TO_W60_FIRST500',before=policy,after=policy,population='first_suffix500',metric=m,group='ALL',**pairs(before,after,m)))
 for before,after in CONTRASTS:
  for b in range(51,61):
   for pop in ['current','historical']+(['suffix'] if b==55 else [])+(['suffix','entry_old','fullseen'] if b==60 else []):
    for m in MULT:
     a=docs[before][b][pop]['metrics'][m]['rows'];z=docs[after][b][pop]['metrics'][m]['rows']
     transitions.append(dict(contrast='CROSS_POLICY_SAME_ITEMS_DIFFERENT_TRAJECTORIES',before=before,after=after,batch=b,population=pop,metric=m,group='ALL',**pairs(a,z,m)))
  # Common success/failure strata at W55; post-treatment strata, not causal.
  for m in MULT:
   a55=docs[before][55]['suffix']['metrics'][m]['rows'];z55=docs[after][55]['suffix']['metrics'][m]['rows']
   require([key(r) for r in a55]==[key(r) for r in z55],'PAIR_W55_IDENTITY')
   for tag,pred in [('COMMON_SUCCESS',lambda a,z:a['success'] and z['success']),('COMMON_FAILURE',lambda a,z:not a['success'] and not z['success'])]:
    keys={key(a) for a,z in zip(a55,z55) if pred(a,z)}
    for p in [before,after]:
     a=[r for r in docs[p][55]['suffix']['metrics'][m]['rows'] if key(r) in keys]
     z=[r for r in docs[p][60]['suffix']['metrics'][m]['rows'] if key(r) in keys]
     transitions.append(dict(contrast=f'{before}_{after}_W55_CONDITIONAL_NOT_CAUSAL',before=p,after=p,population='first_suffix500',metric=m,group=tag,**pairs(a,z,m)))
 uncertainty=[]
 for before,after in CONTRASTS:
  for pop in ['suffix','entry_old','fullseen']:
   for m in MULT:
    uncertainty.append(dict(contrast='TERMINAL_CROSS_POLICY_DIFFERENT_TRAJECTORIES',before=before,after=after,population=pop,
     **request_cluster_bootstrap(docs[before][60][pop]['metrics'][m]['rows'],docs[after][60][pop]['metrics'][m]['rows'],m)))
 for p in ORDER:
  for m in MULT:
   after=subset(docs[p][60]['suffix']['metrics'][m]['rows'],[r['case_id'] for r in records[5000:5500]])
   uncertainty.append(dict(contrast='W55_TO_W60_FIRST500',before=p,after=p,population='first_suffix500',
    **request_cluster_bootstrap(docs[p][55]['suffix']['metrics'][m]['rows'],after,m)))
 dump(out/'request-cluster-uncertainty.json',uncertainty)
 flat=[]
 for r in uncertainty:
  row={k:v for k,v in r.items() if k not in ['method','preference_delta_pp_ci95','desired_target_nll_mean_delta_ci95']}
  for field in ['preference_delta_pp','desired_target_nll_mean_delta']:
   row.update({field+'_ci95_'+k:v for k,v in r[field+'_ci95'].items()})
  row['bootstrap_seed']=r['method']['seed'];row['CI_zero_is_gate']=False
  flat.append(row)
 tables={'batchmetrics':rates,'nll-distributions':tails,'cohort-final':cohorts,'paired-transitions':transitions,'terminal-general':general,'metric-validation':checks,'metric-input-inventory':inputs,'request-cluster-uncertainty':flat}
 for name,rows in tables.items():write_csv(out/(name+'.csv'),rows)
 dump(out/'metric-reduction-receipt.json',dict(status='INDEPENDENT_METRICS_PASS_STATE_AUDIT_SEPARATE',logical_policies=6,new_policies=4,logical_batches=60,new_batches=40,new_forward=0,reference_reused=['N4','REFIT4'],row_identity='case/prompt/hash/target not positionalpairing',NLL_definition='RSPS new<true; NS true<new; tie failure',reference_internal_optimizer='NOT_RECORDED',table_rows={n:len(r) for n,r in tables.items()},table_sha256={n:file_sha(out/(n+'.csv')) for n in tables},audit_evaluated=False,claim_decision='PENDING_GH_REVIEW',scientific_promotion=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--first-only',action='store_true');args=p.parse_args()
 lock,_=policy_roots(args.attempt);dataset=Path(lock['dataset_root'])/'counterfact.json'
 require(file_sha(dataset)==DATA_SHA,'FIXED_DATASET_SHA');records=json.loads(dataset.read_text())
 args.out.mkdir(parents=True,exist_ok=True);first(args.attempt,args.out,records)
 if not args.first_only:reduce(args.attempt,args.out,records)
