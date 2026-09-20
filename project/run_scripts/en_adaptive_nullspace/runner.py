"""Single persistent GPU lane: bounded T0, shared B1, three own B300 chains."""
from __future__ import annotations
import argparse,copy,gc,hashlib,json,os,resource,time,traceback
from pathlib import Path
import numpy as np
import torch
from .runtime import Runtime
from .objective import Objective,active_history,receive_all,registry_status
from .native import requests_from_records
from .current import capture_current
from .geometry import build_geometry,request_alias_weights
from .selector import select_arms
from .controller import run_controller
from . import metrics

ROOT=Path('/data/janghj/ODE-edit/local/en-adaptive-nullspace/20260920-v1')
ARMS=['N4','EN_EXACT','EN_NUM','EN_ADAPT'];CHAINS=['N4','EN_EXACT','EN_ADAPT']

def save(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def compact(obj):return {k:v for k,v in obj.items() if k not in ['rows','weight','actual_delta','history']}
def identity(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def main_run(output):
 start=time.monotonic();rt=Runtime(output);obj=Objective(rt.model,ROOT/'inputs/generated-v1',ROOT/'inputs/reference-inputs.json');rt.objective=obj
 V=np.load(ROOT/'inputs/pstar-derived-v1/basis.npy',mmap_mode='r',allow_pickle=False)
 from .technical import run_t0
 technical=run_t0(rt,obj,V,output/'T0');save(output/'T0-result.json',technical)
 rt.install(rt.W0);rt.guard()
 states={};observations={a:[] for a in ARMS};atwrite={a:[] for a in ARMS};base_parts=[];all_summary=[];native_requests=0;native_batches=0;geometries=0
 for batch in range(1,4):
  records=rt.records[(batch-1)*100:batch*100];arms=ARMS if batch==1 else CHAINS;pending={};bdir=output/f'B{batch}';bdir.mkdir(parents=True)
  native_groups=['SHARED'] if batch==1 else CHAINS
  for group in native_groups:
   entry=states.get(group,dict(weight=rt.W0,M=torch.zeros_like(rt.P),ledger=[]))
   rt.install(entry['weight']);past=active_history(entry['ledger'],records);active_ids=[r['case_id'] for r in past];requests=requests_from_records(records)
   t=time.monotonic();fit=rt.native_runner.fit(requests,entry['M']);native_batches+=1;native_requests+=len(records);WN=fit['weight'];delta=fit['actual_delta'];obj.rebind(WN)
   save(bdir/f'{group}-native.json',dict(receipt=fit['receipt'],seconds=time.monotonic()-t,actual_delta_norm=float(delta.norm()),entry_ledger=registry_status(entry['ledger'],records),checkpoint=False))
   # Native remains immutable RAM; no native target/factor or equivalent update dump.
   if group=='N4':
    pending['N4']=dict(weight=WN,entry=entry,selection={'status':'NATIVE_BASELINE'},active_ids=active_ids);del fit,delta;continue
   geometry_start=time.monotonic();capture_start=time.monotonic();captured=capture_current(rt.model,rt.tok,rt.etok,requests,rt.context)
   captured['manifest']['K_sha256']=hashlib.sha256(captured['K'].contiguous().numpy().tobytes()).hexdigest()
   save(bdir/f'{group}-current-inputs.json',captured['manifest'])
   capture_seconds=time.monotonic()-capture_start;svd_start=time.monotonic()
   geo=build_geometry(captured['K'],captured['weights'],captured['representative_indices'],V);geometries+=1
   svd_seconds=time.monotonic()-svd_start;gradient_start=time.monotonic()
   J,G,native_objective=obj.evaluate(WN,active_ids=active_ids,arm=None if group=='SHARED' else group,gradient=True,kind='native')
   save(bdir/f'{group}-native-objective.json',native_objective)
   gradient_seconds=time.monotonic()-gradient_start;spectrum_start=time.monotonic()
   spectrum=geo.gradient_spectrum(G,delta,J);selection=select_arms(spectrum)
   spectrum['G_sha256']=hashlib.sha256(G.contiguous().numpy().tobytes()).hexdigest()
   save(bdir/f'{group}-spectrum.json',dict(geometry=geo.diagnostic,spectrum=spectrum,selection=selection,seconds=time.monotonic()-geometry_start,timing=dict(current_capture=capture_seconds,weighted_TSQR_SVD=svd_seconds,objective_gradient=gradient_seconds,spectrum_projection=time.monotonic()-spectrum_start)))
   shared=[];input_id=identity(dict(batch=batch,group=group,current=captured['manifest']['identity_sha256'],K=captured['manifest']['K_sha256'],active_ids=active_ids,teacher=obj.store.receipt['manifest_sha256']))
   selected_arms=['EN_EXACT','EN_NUM','EN_ADAPT'] if group=='SHARED' else [group]
   if group=='SHARED':pending['N4']=dict(weight=WN,entry=entry,selection={'status':'NATIVE_BASELINE','objective':native_objective},active_ids=[])
   def evaluate(weight):
    _,_,r=obj.evaluate(weight,active_ids=active_ids,arm=None if group=='SHARED' else group,kind='candidate');return r
   for arm in selected_arms:
    row=selection['selected'][arm];D=-row['eta']*geo.direction(G,row)
    result=run_controller(WN,G,D,native_objective,evaluate,geo,native_norm=spectrum['native_norm'],native_action=spectrum['native_action'],input_identity=input_id,arm=arm,objective_cache=shared,request_columns=request_alias_weights(captured['manifest']))
    save(bdir/f'{arm}-controller.json',{k:v for k,v in result.items() if k!='weight'})
    pending[arm]=dict(weight=result['weight'],entry=entry,selection={k:v for k,v in result.items() if k not in ['weight','ledger']},active_ids=active_ids)
    del D,result
   # Captured current tensors are reusable inputs, not retained as checkpoint.
   del captured,geo,G,delta,fit,shared;gc.collect()
  assert set(pending)==set(arms)
  save(bdir/'SELECTIONS_SEALED.json',dict(batch=batch,arms=arms,selections={a:compact(pending[a]['selection']) for a in arms},official_metrics_used=False,selection_complete=True))
  # W0 observer sees current batch only after all corresponding endpoints seal.
  rt.install(rt.W0);baseline=metrics.evaluate(rt.model,rt.etok,records);base_parts.append(baseline);base_all=metrics.merge_at_write(base_parts)
  save(bdir/'W0-current.json',baseline)
  observer_cache=[];dev_cache=[]
  for arm in arms:
   p=pending[arm];rt.install(p['weight']);committed=rt.native_runner.finalize(requests_from_records(records),p['entry']['M']);rt.guard()
   if batch>1 and arm=='N4':
    _,_,native_obs=obj.evaluate(p['weight'],active_ids=p['active_ids'],arm=arm,kind='observer');save(bdir/f'{arm}-reference-history-observer.json',native_obs)
   t=time.monotonic();alias=next((x for x in observer_cache if torch.equal(x[0].view(torch.int32),p['weight'].view(torch.int32))),None)
   if alias is None:
    obs=metrics.evaluate(rt.model,rt.etok,rt.records[:batch*100]);observer_cache.append((p['weight'],obs,arm))
   else:obs=alias[1]
   observations[arm].append(obs);cur=metrics.subset(obs,[r['case_id'] for r in records]);atwrite[arm].append(cur)
   save(bdir/f'{arm}-metrics.json',obs)
   analyses=dict(current=cur['aggregates'],all_seen=obs['aggregates'],active_past=metrics.subset(obs,p['active_ids'])['aggregates'] if p['active_ids'] else None,
      atwrite_to_now=metrics.paired(metrics.merge_at_write(atwrite[arm]),obs),W0_correct_neighborhood=metrics.retention(base_all,metrics.merge_at_write(atwrite[arm]),obs))
   if batch>1:analyses['first100_from_B1']=metrics.paired(observations[arm][0],metrics.subset(obs,[r['case_id'] for r in rt.records[:100]]))
   save(bdir/f'{arm}-paired-analysis.json',analyses)
   if batch==1 and arm in ['N4','EN_ADAPT']:
    dev_alias=next((x for x in dev_cache if torch.equal(x[0].view(torch.int32),p['weight'].view(torch.int32))),None)
    if dev_alias is None:
     _,_,dev=obj.evaluate(p['weight'],role='Dev128',kind='observer');dev_cache.append((p['weight'],dev,arm))
    else:dev=dict(dev_alias[1],alias=dev_alias[2])
    save(bdir/f'{arm}-Dev128.json',dev)
   if arm in CHAINS and batch<3:
    h=obj.capture_history(arm,records,rt.etok,p['weight']);save(bdir/f'{arm}-atwrite-teacher.json',h)
   states[arm]=dict(weight=p['weight'],M=committed['history'],ledger=receive_all(p['entry']['ledger'],records))
   summary=dict(batch=batch,arm=arm,metrics={k:{x:v for x,v in m.items() if x!='rows'} for k,m in obs['metrics'].items()},aggregates=obs['aggregates'],selection_status=p['selection']['status'],observer_seconds=time.monotonic()-t,observer_alias=None if alias is None else alias[2])
   all_summary.append(summary);save(bdir/f'{arm}-summary.json',summary)
   print('ENDPOINT',batch,arm,json.dumps(summary,ensure_ascii=False),flush=True)
  if batch==1:
   for arm in ['EN_EXACT','EN_NUM','EN_ADAPT']:
    save(bdir/f'{arm}-versus-N4.json',metrics.paired(observations['N4'][0],observations[arm][0]))
   states.pop('EN_NUM',None)
  else:
   for arm in ['EN_EXACT','EN_ADAPT']:
    save(bdir/f'{arm}-versus-N4.json',metrics.paired(observations['N4'][-1],observations[arm][-1]))
  assert native_batches<=7 and native_requests<=700 and obj.counts['reference_gradient']<=5 and obj.counts['reference_candidate']<=14
  save(bdir/'cost.json',dict(native_batches=native_batches,native_requests=native_requests,weighted_SVD=geometries,objective=obj.counts,sweeps=obj.sweeps,elapsed_seconds=time.monotonic()-start,peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
  print('BATCH_COMPLETE',batch,flush=True)
 save(output/'complete.json',dict(status='B300_COMPLETE',technical=technical,summary=all_summary,native_batches=native_batches,native_requests=native_requests,weighted_SVD=geometries,counts=obj.counts,seconds=time.monotonic()-start,job_id=os.environ.get('SLURM_JOB_ID'),gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),host_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,save_checkpoints=False,exact_crash_resume='NOT_AVAILABLE'))
 rt.install(rt.W0);rt.guard();print('TASK_EXECUTION_COMPLETE',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 try:main_run(out)
 except BaseException as e:
  save(out/'technical-failure.json',dict(status='TECHNICAL_FAILURE',error=repr(e),traceback=traceback.format_exc(),job_id=os.environ.get('SLURM_JOB_ID'),save_checkpoints=False,exact_crash_resume='NOT_AVAILABLE'));raise
if __name__=='__main__':main()
