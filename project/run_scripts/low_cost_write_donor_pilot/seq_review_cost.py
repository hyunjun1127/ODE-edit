"""CPU source identities, recorded cost, and source-bound stdout iteration reducer."""
import argparse, datetime, hashlib, json
from pathlib import Path
from .review_provenance import sha,write,csvwrite

ARMS=['N4','RES8','S875','S75','FULL8','REFIT4']
CONTROL_93C={'sequential_submit.py':'46e7e9bebbfc65637c53450e0cc545f5237e26cb750123d4a2430b09749798dc','test_sequential_submission.py':'8345efa076b51074deeb19341a38bec8046f27dfb6be40d34c11afcbfba8ef2a'}
# Single bounded sacct observation on 2026-09-14; no scheduler queries in reproduction.
SCHED=[
 (5237,'2026-09-13T21:06:30','2026-09-13T22:33:47',38041400),
 (5953,'2026-09-13T21:06:30','2026-09-13T22:45:43',37262076),
 (5327,'2026-09-13T22:33:51','2026-09-14T00:02:38',37373360),
 (5275,'2026-09-13T22:45:58','2026-09-14T00:13:53',35121204),
 (5656,'2026-09-14T00:02:39','2026-09-14T01:36:55',35169140),
 (6027,'2026-09-14T00:13:58','2026-09-14T01:54:25',36370320)]
def load(p):return json.loads(Path(p).read_text())
def segments(lines):
 result=[]; n=None
 for line in lines:
  if line.startswith('Computing right vector'):
   if n is not None:result.append(n)
   n=0
  elif line.startswith('loss ') and n is not None:n+=1
 if n is not None:result.append(n)
 if not result or not all(1<=x<=25 for x in result):raise ValueError('INVALID_Z_SEGMENT_COVERAGE')
 return result
def run(attempt,out,repo):
 a=Path(attempt);o=Path(out);repo=Path(repo);o.mkdir(parents=True,exist_ok=True)
 lock=load(a/'execution.lock.json')
 assert sha(a/'execution.lock.json')=='695a2d985d5abb8fae1cc6f0b1933022895a47a71142aa032a9da3b6be1bbb28'
 assert sha(a/'execution-source.tar')=='3dadca9463ab131846bff5227fe1bbbb540dd5b8f238bfad53b15b121ed0eb42'
 members=[]
 for r in lock['members']:
  p=Path(r['path']);new=p.is_relative_to(a/'source') or (p.is_relative_to(a) and p.suffix!='.pt')
  if new:assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)
  members.append(dict(r,verification='NEW_FULL_SHA_SIZE' if new else 'REUSED_LOCK_AND_EXECUTION_VERIFY'))
 csvwrite(o/'execution-member-identities.csv',members)
 source=[]
 for p in sorted((a/'source/project/run_scripts/low_cost_write_donor_pilot').glob('*')):
  if p.suffix not in ('.py','.sbatch'):continue
  q=repo/'project/run_scripts/low_cost_write_donor_pilot'/p.name
  exact=q.exists() and sha(p)==sha(q)
  if q.exists() and not exact:assert CONTROL_93C.get(p.name)==sha(q),('EXECUTED_PUBLICATION_BYTES',p.name)
  source.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),publication_path=str(q),publication_exact=exact,publication_sha256=sha(q) if q.exists() else None,publication_lineage='EXECUTION_EXACT' if exact else '93c3e4f_CONTROL_ONLY_NO_RUNTIME_ARCHIVE_MUTATION'))
 csvwrite(o/'executed-source-identities.csv',source)
 sums=[];batchrows=[];fitrows=[];zr=[];scheduler=[];logrefs=[]
 for i,arm in enumerate(ARMS):
  root=a/'output'/f'cell-{i}';t=load(root/'terminal.json')
  assert t['status']=='TEN_SEQUENTIAL_BATCHES_COMPLETE' and t['arm']==arm
  log=a/'slurm'/f'46475_{i}.out';ss=segments(log.read_text().splitlines())
  assert len(ss)==t['request_z_total']==(2000 if arm in ['RES8','FULL8','REFIT4'] else 1000)
  logrefs.append(dict(path=str(log),bytes=log.stat().st_size,sha256=sha(log),coverage='FULL_STDOUT_SEGMENT_SCAN_NO_PROMPTS_PUBLISHED'))
  offset=0; own=[]
  for b in range(51,61):
   c=load(root/f'B{b:03d}'/'commit.json');e=load(root/f'B{b:03d}'/'evaluation.json')
   costs={k:c[k] for k in ['first_fit_seconds','second_fit_seconds','materialization_seconds','finalization_seconds','evaluation_seconds','policy_instrumented_online_seconds']}
   row=dict(arm=arm,batch=b,**costs);own.append(row);batchrows.append(row)
   for stage in ['first_fit','second_fit']:
    f=c[stage]
    if f is None:continue
    z=ss[offset:offset+100];offset+=100
    assert len(z)==f['compute_z']==100
    fitrows.append(dict(arm=arm,batch=b,stage=stage,layer=f['layer'],compute_z=f['compute_z'],compute_z_seconds=f['compute_z_seconds'],compute_ks=f['compute_ks'],compute_ks_seconds=f['compute_ks_seconds'],readout_seconds=f['get_module_input_output_at_words_seconds'],solve=f['solve'],solve_seconds=f['solve_seconds'],fit_receipt_seconds=f['seconds'],actual_delta_norm=f['actual_delta_norm'],history_append=f['history_append'],loss_evaluations=sum(z),adam_updates=sum(x-1 for x in z),early_stop_requests=sum(x<25 for x in z)))
    for j,n in enumerate(z):zr.append(dict(arm=arm,batch=b,fit_stage=stage,request_ordinal=(b-1)*100+j,loss_evaluations=n,adam_updates=n-1,source='COMPLETED_STDOUT_ORDER_PLUS_PINNED_NATIVE_LOOP'))
  assert offset==len(ss)
  elapsed,start,end,rss=SCHED[i];dt=datetime.datetime.fromisoformat
  assert (dt(end)-dt(start)).total_seconds()==elapsed
  scheduler.append(dict(job=f'46475_{i}',arm=arm,job_name='odeedit_lowcost_seq10_s4',owner='janghj',node='server4',state='COMPLETED',exit_code='0:0',submit='2026-09-13T21:05:19',start=start,end=end,elapsed_seconds=elapsed,allocated_gpu_seconds=elapsed,allocated_GPUh=elapsed/3600,queue_seconds=(dt(start)-dt('2026-09-13T21:05:19')).total_seconds(),allocated_gpus=1,cpus=8,mem_MiB=60416,max_rss_KiB=rss))
  sums.append(dict(arm=arm,**{k:sum(r[k] for r in own) for k in costs},model_load_seconds=t['model_seconds'],program_seconds=t['seconds'],allocated_gpu_seconds=elapsed,allocated_GPUh=elapsed/3600,loss_evaluations=sum(ss),adam_updates=sum(x-1 for x in ss),early_stop_requests=sum(x<25 for x in ss),request_z=t['request_z_total'],fit_solve=t['fit_solve_total'],peak_allocated_bytes=t['peak_allocated_bytes'],peak_reserved_bytes=t['peak_reserved_bytes'],max_host_rss_KiB=rss,pure_writer_seconds='NOT_SEPARATED',snapshot_IO_seconds='NOT_SEPARATELY_RECORDED',current_M8_reconstruction=0))
 n4=sums[0]
 for r in sums:
  r['online_ratio_N4']=r['policy_instrumented_online_seconds']/n4['policy_instrumented_online_seconds']
  r['allocated_ratio_N4']=r['allocated_gpu_seconds']/n4['allocated_gpu_seconds']
  r['reference_1_5x']='WITHIN_REFERENCE' if r['online_ratio_N4']<=1.5 else 'EXCEEDS_REFERENCE'
  r['reference_2x']='WITHIN_REFERENCE' if r['online_ratio_N4']<=2 else 'EXCEEDS_REFERENCE'
 for name,rows in [('compute-summary',sums),('batch-compute',batchrows),('fit-compute',fitrows),('z-iterations',zr),('scheduler',scheduler),('stdout-identities',logrefs)]:csvwrite(o/(name+'.csv'),rows)
 events=sorted([(r['start'],1) for r in scheduler]+[(r['end'],-1) for r in scheduler]);active=peak=0;last=None;dur={0:0,1:0,2:0}
 for time,delta in events:
  if last is not None:dur[active]+=(dt(time)-dt(last)).total_seconds()
  active+=delta;peak=max(peak,active);last=time
 assert peak==2 and active==0
 write(o/'compute-accounting.json',dict(scheduler_observation='ONE_BOUNDED_SACCT_2026_09_14',allocated_gpu_seconds=sum(x[0] for x in SCHED),allocated_GPUh=sum(x[0] for x in SCHED)/3600,concurrency_seconds=dur,peak_project_campaign_concurrency=peak,elapsed_campaign_seconds=sum(dur.values()),reused_M8_setup_seconds=276.130524,prior_M8_setup_not_added_to_new_allocation=True,prior_estimate_perchain_GPUh=[2,8],prior_estimate_campaign_GPUh=[12,48],prior_storage_estimate_GiB=80,scheduler_walltime_limit_hours=24,GPU_hour_budget=None,actual_request_z=sum(r['request_z'] for r in sums),actual_loss_evaluations=sum(r['loss_evaluations'] for r in sums),actual_adam_updates=sum(r['adam_updates'] for r in sums),IO_restore_P_diagnostic_breakdown='NOT_SEPARATELY_RECORDED',same_host_component_ratios_only=True))
 write(o/'evidence-reuse-manifest.json',dict(execution_commit='5e96dcb3745977b1f273e3f5afbee61167248d49',execution_tree='6e9f8bdb432392fd5f0d7d0937ff7058666944c0',analysis_start_main='83ca22bd1bd318d5e6bdfd32c37794ffbbcd68ed',analysis_start_tree='38a6318d6b87941421255ae886ab66a97f5adc47',execution_lock_sha256=sha(a/'execution.lock.json'),archive_sha256=sha(a/'execution-source.tar'),sample_lock_sha256=sha(a/'sample-sequential.lock.json'),source_members=len(members),new_rehash=sum(x['verification']=='NEW_FULL_SHA_SIZE' for x in members),reused_verified_inputs=sum(x['verification']!='NEW_FULL_SHA_SIZE' for x in members),prepared=lock['prepared'],prior_initial_agent_observation='N4_B51_TO_B52_ONLY',current_scope='CPU_COMPLETED_SEQ10_REVIEW',GPU_model_evaluator_Slurm_mutation=0,broadcast='NO_BROADCAST_NOT_REQUIRED',claim_decision='PENDING_GH_REVIEW',scientific_promotion=False))
 print(json.dumps(sums))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--out',required=True);p.add_argument('--repo',default='.');a=p.parse_args();run(a.attempt,a.out,a.repo)
