"""Frozen depth-first cell schedule. Every scientific continuation is registered."""
import copy,json,time,traceback
from project.run_scripts.memit_history_lifelong.io import save
from .plan import past_panel
from .calibration import Calibration

class Orchestrator:
 def __init__(self,engine,cells):
  self.e=engine;self.cells={c['cell_id']:c for c in cells};self.cal=Calibration(engine);self.zlock=None;self.zerror=None
 def probe(self,n):
  if self.zerror:return
  try:self.cal.probe(n)
  except (RuntimeError,NotImplementedError,FloatingPointError) as ex:
   # Unsupported true FP64 is a Z-only blocker; state/identity corruption is
   # an AssertionError and must abort. Never replace it with native targets.
   self.zerror=dict(status='BLOCKED',reasons=['FP64_ORACLE_TECHNICAL_UNAVAILABLE'],error=repr(ex),traceback=traceback.format_exc())
   save(self.e.output/f'calibration/blocked-at-{n}.json',self.zerror)
 def diagnostic(self,cell,anchor):
  e=self.e;name=cell['cell_id'];n=int(cell['anchor_requests'])
  if name in e.completed:return
  e.restore(anchor)
  probe=e.states.probe();e.meta['cell']=name;e.adapter.anchor_H=anchor['H']
  panel=past_panel(e.rows,n)
  if n:assert len(panel)==400
  save(e.output/'cells'/name/'lineage.json',dict(parent='main_000',anchor=n,past_panel_ordinals=panel,
    fork_identity=anchor['identity'],independent_CPU_clone=True,CP=False,continuation_requests=1000,batch_size=10))
  if cell['history']=='forced_refresh':e.refresh([5,6,7,8],e.output/'cells'/name/'forced-refresh.json')
  if panel:e.observe(panel,e.output/'cells'/name/'entry-past400.json')
  e.observe(list(range(n,n+1000)),e.output/'cells'/name/'entry-next1000.json')
  for _ in range(100):e.step(cell,10,n,panel)
  e.complete(cell);e.adapter.anchor_H=None
  e.restore(anchor,output_parity=probe)
 def diagnostics(self,n,anchor):
  for family in ['writer','history']:
   for c in self.cells.values():
    if c['family']==family and int(c['anchor_requests'])==n:self.diagnostic(c,anchor)
 def boundary(self,arm,cell,limit,child,fired,layers,cp):
  e=self.e;n=e.meta['cursor']
  if arm=='000' and n in [1000,3000,5000,7000]:
   anchor=e.snapshot()
   if n==1000:
    if (e.output/'calibration/lock.json').exists():
     self.zlock=json.loads((e.output/'calibration/lock.json').read_text());assert self.zlock['bindings']==e.lock['bindings']
    else:
     if 0 not in self.cal.banks and not self.zerror:
      e.restore(e.w0);self.probe(0);e.restore(anchor)
     self.probe(n)
     if self.zerror:self.zlock=self.zerror;save(e.output/'calibration/lock.json',dict(**self.zlock,bindings=e.lock['bindings']))
     else:self.zlock=self.cal.finish(e.w0,anchor)
   self.diagnostics(n,anchor);e.restore(anchor);del anchor
   if n==1000 and not (e.output/'calibration/cost-lock.json').exists():self.freeze_cost()
  if child and 'main_'+child in e.completed:fired=True
  if child and not fired and layers and n<int(self.cells['main_'+child]['total_requests']):
   snapshot=e.snapshot();probe=e.states.probe();fired=True
   if cp:e.cp.pin(cp['path'],+1,'pending_parent_during_child')
   self.run_main(child,snapshot,layers,parent=cell['cell_id'],pin=cp['path'] if cp else None)
   e.restore(snapshot,output_parity=probe);e.adapter.zlock=self.zlock
   if cp:e.cp.pin(cp['path'],-1,'child_finished_parent_restored')
   del snapshot
  e.meta['child_fired']=fired;e.meta['boundary_pending']=False
  return fired
 def run_main(self,arm,from_snapshot=None,first_layers=None,parent=None,pin=None,resume=False):
  e=self.e;cell=self.cells['main_'+arm];limit=int(cell['total_requests']);refresh=cell['history']=='refresh'
  e.restore(from_snapshot or e.w0);e.meta['cell']=cell['cell_id'];e.meta['resume_cell']=cell['cell_id']
  start=e.meta['cursor'];e.adapter.zlock=self.zlock
  lineage=e.output/'cells'/cell['cell_id']/'lineage.json'
  if not lineage.exists():
   save(lineage,dict(parent=parent,prefix_requests=start,entry_identity=e.states.identity(e.meta),
    prefix_evaluation_reference=parent,source_locked=True,independent_CPU_clone=True,pre_refresh=True,
    resume_parent_checkpoint=pin,shared_prefix_is_independent_replication=False))
  else:assert resume,'DUPLICATE_CELL_START'
  if first_layers:
   e.refresh(first_layers,e.output/'cells'/cell['cell_id']/f'refresh-{start:05d}.json')
   e.meta['boundary_pending']=False;e.meta['child_fired']=False
   if arm in ['000','100','110','111']:e.cp.write(arm,start,e.meta)
  child={'000':'001','100':'101','010':'011','110':'111'}.get(arm)
  fired=e.meta.get('child_fired',False) if resume else False
  if resume and e.meta.get('boundary_pending'):
   cp=e.cp.write(arm,start,e.meta)
   fired=self.boundary(arm,cell,limit,child,fired,e.meta['boundary_layers'],cp)
  for cursor in range(start,limit,100):
   e.step(cell,100);n=e.meta['cursor'];layers=[]
   if n%1000==0:layers=e.trigger(e.output/'cells'/cell['cell_id']/f'trigger-{n:05d}.json')
   if refresh and layers and n<limit:e.refresh(layers,e.output/'cells'/cell['cell_id']/f'refresh-{n:05d}.json')
   if n%1000==0:
    e.meta.update(boundary_pending=True,boundary_layers=layers,child_fired=fired)
    cp=e.cp.write(arm,n,e.meta) if arm in ['000','100','110','111'] else None
    fired=self.boundary(arm,cell,limit,child,fired,layers,cp)
    save(e.output/'cells'/cell['cell_id']/f'boundary-{n:05d}.json',dict(cursor=n,child_fired=fired,completed_cells=sorted(e.completed),checkpoint=cp,restored_identity=e.states.identity(e.meta)))
  e.complete(cell)
  if child and not fired:
   c=self.cells['main_'+child];alias=dict(status='NOT_FIRED',cell=c['cell_id'],parent=cell['cell_id'],
    endpoint=int(c['total_requests']),evaluation_reference=str(e.output/'cells'/cell['cell_id']/f"C{int(c['total_requests']):05d}"),
    independent_run=False,physical_requests=0)
   save(e.output/'cells'/c['cell_id']/'terminal.json',alias);e.completed[c['cell_id']]=alias
  if arm in ['000','100','110','111']:e.cp.complete(arm)
 def freeze_cost(self):
  e=self.e
  rows=[]
  for p in (e.output/'cells').glob('*/C*/commit.json'):rows.append(json.loads(p.read_text()))
  by={}
  for r in rows:by.setdefault(r['cell'],[]).append(r)
  means={k:dict(writes=len(v),write_mean_seconds=sum(x['write_seconds'] for x in v)/len(v),observer_mean_seconds=sum(x['observer_seconds'] for x in v)/len(v)) for k,v in by.items()}
  import numpy as np
  from .engine import FULL
  t0=json.loads((e.output/'T0/adapter.json').read_text())
  writers=[json.loads(p.read_text()) for p in (e.output/'cells/main_000').glob('C*/writer.json')]
  div_core=float(np.mean([r['seconds']-r['timers'].get('native_z',0.) for r in writers]))
  native_per_request=float(np.mean([r['timers']['native_z']/r['requests'] for r in writers]))
  joint_extra=t0['timers'].get('factorization',0.)+2*t0['timers'].get('A_solve',0.)+t0['timers']['native_key']
  full_per_case=json.loads((e.output/'W0-all10k.json').read_text())['seconds']/10000
  rw=[json.loads(p.read_text()) for p in (e.output/'cells/main_000').glob('C*/past-rewrite.json')]
  rewrite_per_case=sum(r['seconds'] for r in rw)/sum(r['requests'] for r in rw)
  calibration_file=json.loads((e.output/'calibration/lock.json').read_text())
  spg_times=[r['oracle_seconds'] for r in calibration_file.get('results',self.cal.results)]
  spg_mean=float(np.mean(spg_times)) if spg_times else None
  spg_p95=float(np.quantile(spg_times,.95)) if spg_times else None
  def observation_cases(end,start):
   full=rew=0
   for n in range(start+100,end+1,100):
    full+=100
    if n in FULL:full+=n-100
    else:rew+=n-100
   return full,rew
  diag_write=0.
  for kind,count in [('divisor',1100),('joint',300),('frozen_upper_joint',300),('energy_matched_divisor',300)]:
   diag_write+=means['writer_0_'+kind]['write_mean_seconds']*count
  forecasts=[]
  for maximum in [False,True]:
   paths=[('native','divisor',10000,0),('native','joint',10000,0)]
   if maximum:paths += [('native','divisor',2000,1000),('native','joint',2000,1000)]
   if self.zlock['status']=='PASS':
    paths += [('spg','divisor',2000,0),('spg','joint',10000,0)]
    if maximum:paths += [('spg','divisor',2000,1000),('spg','joint',10000,1000)]
   edit=observer=0.
   for solver,writer,end,start in paths:
    n=end-start;z=native_per_request if solver=='native' else (spg_p95 if maximum else spg_mean)
    edit+=n*z+n/100*(div_core+(joint_extra if writer=='joint' else 0.))
    f,r=observation_cases(end,start);observer+=f*full_per_case+r*rewrite_per_case
   # 20 diagnostic paths, all current + entry/continuation observations;
   # 16 nonzero-anchor paths also have five past400 observations each.
   observer+=(20*3570+16*5*400)*full_per_case
   refresh_paths=list((e.output/'cells').glob('*/forced-refresh.json'))
   unit=json.loads(refresh_paths[0].read_text())['seconds']/1000 if refresh_paths else 0.
   # Four-layer forced reconstruction at 1/3/5/7k plus worst automatic
   # child reconstructions at 1k..9k; zero in the no-trigger estimate.
   refresh=(1000+3000+5000+7000)*unit+(45000+3*1000)*unit*maximum
   forecasts.append(dict(case='EARLY_TRIGGERS_AND_P95_Z' if maximum else 'NO_TRIGGERS_AND_MEAN_Z',
    main_write_seconds=edit,diagnostic_write_seconds=diag_write,observer_seconds=observer,refresh_seconds=refresh,
    subtotal_GPU_hours=(edit+diag_write+observer+refresh)/3600,
    exclusions='T0/probes/model loading/CP/CPU serialization; measured separately, not counted twice'))
  save(e.output/'calibration/cost-lock.json',dict(status='MEASURED_COMPONENT_EXTRAPOLATION_NOT_ACTUAL_TOTAL',components=means,Z=self.zlock,
   forecasts=forecasts,native_seconds_per_request=native_per_request,SPG_mean_seconds=spg_mean,SPG_P95_seconds=spg_p95,
   main_joint_extra_basis='measured BS100 LU/direct-solve and key phases; not request-only scaling',
   allocation_wall_hours=e.lock['resource']['wall_hours'],GPUh_hardcap=None,runtime_phase_counters=True))
 def run_group(self,group):
  e=self.e
  if group=='A':
   self.diagnostics(0,e.w0);self.probe(0);self.run_main('000')
  elif group=='B':self.run_main('100')
  elif group in ['C','D']:
   lock_path=e.output/'calibration/lock.json'
   if lock_path.exists():
    self.zlock=json.loads(lock_path.read_text());assert self.zlock['bindings']==e.lock['bindings']
   else:self.zlock=dict(status='BLOCKED',reasons=['UPSTREAM_CALIBRATION_NOT_AVAILABLE'],upstream_technical=True)
   arms=['010','011'] if group=='C' else ['110','111']
   if self.zlock['status']=='PASS':self.run_main(arms[0])
   else:
    for arm in arms:
     rec=dict(status='BLOCKED_Z_CALIBRATION',cell='main_'+arm,calibration=self.zlock)
     save(e.output/'cells'/('main_'+arm)/'terminal.json',rec);e.completed['main_'+arm]=rec
  else:raise ValueError('UNKNOWN_SCIENCE_GROUP')
  expected={c['cell_id'] for c in self.cells.values() if (group=='A' and (c['family']!='main' or c['arm'] in ['000','001'])) or (group=='B' and c['family']=='main' and c['arm'] in ['100','101']) or (group=='C' and c['family']=='main' and c['arm'] in ['010','011']) or (group=='D' and c['family']=='main' and c['arm'] in ['110','111'])}
  assert set(e.completed)==expected,(set(e.completed),expected)
