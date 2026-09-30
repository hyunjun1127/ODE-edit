"""One live model. Explicit RAM branches; immutable scalar observation files."""
import copy,json,time,traceback
from pathlib import Path
import torch
from project.run_scripts.memit_history_lifelong.io import save,signature,content,tensor_sha,digest,file_sha
from project.run_scripts.memit_history_lifelong.metrics import evaluate,merge,strata
from .writer import Adapter
from . import algebra
from .state import States,Checkpoints
from .history import History
from .plan import past_panel

FULL=[100,500]+list(range(1000,10001,1000))
class Engine:
 def __init__(self,model,tok,evaltok,module,hp,weights,H,rows,lock,output):
  self.model,self.tok,self.evaltok,self.module,self.hp=model,tok,evaltok,module,hp
  self.weights,self.H,self.rows,self.lock,self.output=weights,H,rows,lock,Path(output)
  self.meta=dict(cursor=0,ledger=[],samples=[],triggers=[],commits=[],cell='W0')
  self.states=States(weights,H,module,model,tok,rows)
  self.adapter=Adapter(module,model,tok,hp,H);self.history=History(module,model,tok,hp,H,rows)
  self.cp=Checkpoints(self.output/'temporary-checkpoints',self.states,lock['bindings'])
  self.total=dict(physical_requests=0,physical_writes=0,technical_requests=0,shadow_requests=0)
  self.stage='INITIALIZED';self.completed={};self.w0=None;self.forward_cost={}
  def forward_event(module,args,kwargs):
   tokens=kwargs.get('input_ids',args[0] if args else None)
   if tokens is not None:
    key=getattr(module,'_hj_phase',None) or self.stage
    cost=self.forward_cost.setdefault(key,dict(calls=0,padded_tokens=0,valid_tokens=0));cost['calls']+=1;cost['padded_tokens']+=tokens.numel()
    mask=kwargs.get('attention_mask');cost['valid_tokens']+=int(mask.sum()) if mask is not None else tokens.numel()
  self.forward_handle=model.register_forward_pre_hook(forward_event,with_kwargs=True)
 def requests(self,start,stop):return [dict(r['requested_rewrite'],case_id=int(r['case_id'])) for r in self.rows[start:stop]]
 def snapshot(self):
  t=time.monotonic();s=self.states.snapshot(self.meta);s['copy_seconds']=time.monotonic()-t;return s
 def restore(self,s,output_parity=None):
  t=time.monotonic();self.meta=self.states.restore(s)
  if output_parity is not None:assert self.states.probe()==output_parity,'FORK_OUTPUT_PARITY'
  return time.monotonic()-t
 def observe(self,ids,path,full=True):
  if not ids:return None
  before=self.states.identity(self.meta);t=time.monotonic()
  result=evaluate(self.model,self.evaltok,[self.rows[i] for i in ids],self.weights,self.H,full=full)
  assert self.states.identity(self.meta)==before,'OBSERVER_AUX_MUTATION'
  result.update(ordinal_sha256=digest(ids),seconds=time.monotonic()-t,observer_context_rng_ledger_unchanged=True)
  save(path,result);return result
 def refresh(self,layers,path):
  before={k:tensor_sha(w) for k,w in self.weights.items()};probe=self.states.probe();aux=self.states.identity(self.meta)
  rec=self.history.rebuild(self.meta,layers)
  assert before=={k:tensor_sha(w) for k,w in self.weights.items()} and probe==self.states.probe(),'REFRESH_CHANGED_WEIGHTS_OUTPUT'
  after=self.states.identity(self.meta)
  assert all(aux[k]==after[k] for k in ['rng','contexts','ledger'])
  rec.update(W_output_unchanged=True,reference_reset=True,write_origin_preserved=True)
  save(path,rec);return rec
 def energy_write(self,requests,entry,count):
  # Shadow proposals share ONLY this entry's native z, not another trajectory.
  a=self.adapter;t=time.monotonic()
  divisor=a.run(requests,count=count,capture_updates=True);updates=a.updates;zs=a.zvectors
  Ediv=sum(x['ideal_energy'] for x in divisor['layers']);self.restore(entry)
  joint=a.run(requests,mode='joint',count=count,preloaded=zs)
  Ejoint=sum(x['ideal_energy'] for x in joint['layers']);self.restore(entry)
  scale=(Ejoint/Ediv)**.5 if Ediv>0 else None
  # Ediv==0: the zero divisor stays zero; scale is explicitly undefined.
  factor=scale if scale is not None else 1.
  record=dict(mode='energy_matched_divisor',zmode='native',requests=len(requests),
   shadow_divisor=divisor,shadow_joint=joint,shadow_seconds=time.monotonic()-t,
   energy_scale=scale,energy_scale_status='DEFINED' if scale is not None else 'NA_ZERO_DIVISOR_KEEP_ZERO',
   entry_Ediv=Ediv,entry_Ejoint=Ejoint,entry_A_common=True,shadow_z_reuse_same_entry=True,layers=[],append=[],calls={})
  # Materialize the scaled divisor shadow endpoint once. This is the
  # pre-registered energy diagnostic, not a re-fit under scaled lower writes.
  z=torch.stack([v for _,v in zs],1).cuda()
  def residual():
   cur=self.module.get_module_input_output_at_words(self.model,self.tok,8,context_templates=[r['prompt'] for r in requests],words=[r['subject'] for r in requests],module_template=self.hp.layer_module_tmp,fact_token_strategy=self.hp.fact_token)[1].T
   return (z-cur).double()
  with torch.no_grad():
   r0=residual()
   for i,(name,w) in enumerate(self.weights.items()):
    k=self.module.compute_ks(self.model,self.tok,requests,self.hp,i+4,self.module.CONTEXT_TEMPLATES_CACHE).T.double();before=residual()
    delta=updates[i].cuda()*factor;w.copy_(entry['weights'][name].cuda()+delta.float())
    actual=w.double()-entry['weights'][name].cuda().double();after=residual()
    cov=self.module.get_cov(self.model,self.tok,self.hp.rewrite_module_tmp.format(i+4),self.hp.mom2_dataset,self.hp.mom2_n_samples,self.hp.mom2_dtype)
    ai=self.hp.mom2_update_weight*cov.double()+entry['H'][i].cuda().double()
    layer=algebra.geometry(r0,before,after,actual,k,ai)
    layer.update(layer=i+4,ideal_to_FP32_relative=algebra.relative(actual,delta) if float(delta.norm()) else None,divisor_shadow_capacity=divisor['layers'][i]['trace'])
    if a.anchor_H is not None:
     anchor_a=ai+(a.anchor_H[i]-entry['H'][i]).cuda().double();layer['anchor_A_energy']=float(((actual@anchor_a)*actual).sum());del anchor_a
    record['layers'].append(layer);del actual,delta,ai,k
   for i,l in enumerate(self.hp.layers):
    k=self.module.compute_ks(self.model,self.tok,requests,self.hp,l,self.module.CONTEXT_TEMPLATES_CACHE).T
    self.H[i].add_(k.cpu()@k.cpu().T)
    record['append'].append(dict(layer=l,key_sha=tensor_sha(k),H_sha=tensor_sha(self.H[i]),phase='POST_ALL_FIVE_WRITES'))
  record.update(history_appends=5,returned_history_identity=True,seconds=time.monotonic()-t)
  self.total['shadow_requests']+=2*len(requests)
  del updates,zs;a.updates=[];a.zvectors=[];torch.cuda.empty_cache();return record
 def step(self,cell,bs,anchor=0,panel=None):
  name=cell['cell_id'];start=self.meta['cursor'];stop=start+bs
  assert self.meta['ledger']==list(range(start)) and stop<=10000
  root=self.output/'cells'/name/f'C{stop:05d}';self.stage=name+':ENTRY:'+str(start)
  before=self.states.identity(self.meta)
  previous=self.output/'cells'/name/f'C{start:05d}'/'commit.json'
  if previous.exists():
   committed=json.loads(previous.read_text())['endpoint_identity']
   # Boundary refresh may replace H and sample metadata, but never W, RNG,
   # native context or the semantic ledger. Ordinary steps bind all state.
   boundary=bs==100 and start%1000==0
   keys=['rng','contexts','ledger'] if boundary else list(before)
   assert all(before[k]==committed[k] for k in keys),'CHAIN_ENTRY_CONTINUITY'
   assert before['state']['weights']==committed['state']['weights'],'CHAIN_WEIGHT_CONTINUITY'
  entry=self.snapshot();requests=self.requests(start,stop)
  save(root/'entry.json',dict(cell=name,start=start,stop=stop,identity=before,request_ids=[r['case_id'] for r in requests],request_hashes=[digest(r) for r in requests],prior_history_norms=[float(h.norm()) for h in self.H]))
  if name=='writer_0_divisor' and start==10:
   previous=json.loads((self.output/'cells'/name/'C00010/commit.json').read_text())
   assert before==previous['endpoint_identity'],'INITIAL_NEXT_ENTRY_CONTINUITY'
   save(self.output/'actual-initial-gate.json',dict(status='PASS',cell=name,committed_requests=10,next_entry=10,
    commit_path=str(self.output/'cells'/name/'C00010/commit.json'),next_entry_path=str(root/'entry.json'),
    autonomous_plan=self.lock['plan'],T0_actual=True,actual_terminal='NOT_OBSERVED',science_quality_gate=False))
  try:
   self.stage=name+':WRITE:'+str(start);t=time.monotonic()
   if cell['allocation']=='energy_matched_divisor':rec=self.energy_write(requests,entry,start)
   else:rec=self.adapter.run(requests,mode=cell['allocation'],zmode=cell['z_solver'],count=start)
   assert all(torch.isfinite(w).all() for w in self.weights.values()) and torch.isfinite(self.H).all(),'NONFINITE_W_H'
   assert len(rec['append'])==5 and [x['layer'] for x in rec['append']]==self.hp.layers
   self.meta['cursor']=stop;self.meta['ledger'].extend(range(start,stop));self.meta['commits'].append(dict(cell=name,start=start,stop=stop))
   if bs==100:
    immutable=self.states.identity(self.meta);self.history.record(self.meta,start,stop)
    assert all(self.states.identity(self.meta)[k]==immutable[k] for k in ['state','rng','contexts','ledger']),'SAMPLE_OBSERVER_MUTATION'
   save(root/'writer.json',rec)
   self.stage=name+':OBSERVER:'+str(stop);obs_start=time.monotonic()
   current=self.observe(list(range(start,stop)),root/'current.json')
   if bs==100:
    full=stop in FULL
    past=self.observe(list(range(start)),root/'past-full.json' if full else root/'past-rewrite.json',full=full) if start else None
    seen=merge(past,current,full=full);seen['strata']=strata(seen,self.rows[:stop]);save(root/'all-seen.json',seen)
   elif stop-anchor in [10,100,500,1000]:
    past=self.observe(list(range(anchor,start)),root/'continuation-past.json') if start>anchor else None
    save(root/'continuation.json',merge(past,current))
    if panel:self.observe(panel,root/'past400.json')
   end=self.states.identity(self.meta)
   receipt=dict(status='COMMITTED',cell=name,start=start,stop=stop,requests=bs,entry_identity=before,endpoint_identity=end,
    prior_history_used=True,history_append_layers=5,history_phase='POST_ALL_FIVE_WRITES',
    cache_c_returned_same_object=rec['returned_history_identity'],finite_W_H=True,
    history_norms=[float(h.norm()) for h in self.H],observer_nonmutating=True,
    current={k:{a:b for a,b in m.items() if a!='rows'} for k,m in current['metrics'].items()},
    write_seconds=rec['seconds'],observer_seconds=time.monotonic()-obs_start,total_step_seconds=time.monotonic()-t,
    rollback_RAM_snapshot_seconds=entry['copy_seconds'],peak_GPU_bytes=torch.cuda.max_memory_allocated())
   assert [current['metrics'][k]['denominator'] for k in ['RS','PS','NS']]==[bs,bs*2,bs*10]
   save(root/'commit.json',receipt);self.total['physical_requests']+=bs;self.total['physical_writes']+=1
   print('HJ_COMMITTED',name,start,stop,receipt['total_step_seconds'],flush=True)
   return receipt
  except BaseException:
   original=traceback.format_exc();self.restore(entry)
   save(root/'rollback.json',dict(status='TECHNICAL_FAILURE',traceback=original,restored=self.states.identity(self.meta)==before,stage=self.stage));raise
  finally:del entry;torch.cuda.empty_cache()
 def trigger(self,path):
  before=self.states.identity(self.meta);layers,rec=self.history.trigger(self.meta)
  assert self.states.identity(self.meta)==before,'TRIGGER_OBSERVER_MUTATION'
  self.meta['triggers'].append(dict(cursor=self.meta['cursor'],layers=layers,observations=rec))
  save(path,dict(cursor=self.meta['cursor'],layers=layers,observations=rec,nonintervening=True));return layers
 def complete(self,cell,extra=None):
  name=cell['cell_id'];root=self.output/'cells'/name
  rec=dict(status='COMPLETED',cell=name,cursor=self.meta['cursor'],endpoint=self.states.identity(self.meta),extra=extra or {})
  from .reducer import verify_cell
  rec['CPU_scalar_verification']=verify_cell(self.output,cell,rec)
  save(root/'artifact-manifest.json',dict(members=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=file_sha(p)) for p in sorted(root.rglob('*')) if p.is_file()]))
  save(root/'terminal.json',rec);self.completed[name]=rec
  return rec
