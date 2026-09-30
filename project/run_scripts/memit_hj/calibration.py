"""Two-anchor precision calibration, with native loss-only and matched Adam."""
import copy,time,math
import numpy as np
import torch
from .oracle import Oracle,castcache
from .precision import fp64_reference
from .spg import project,solve,calibrate
from project.run_scripts.single_layer_mechanism_first.z_hook import normalize_requests
from project.run_scripts.memit_history_lifelong.io import save,tensor_sha

class Calibration:
 def __init__(self,engine):self.e=engine;self.probes=[];self.banks={};self.results=[]
 def probe(self,anchor):
  e=self.e;before=e.states.identity(e.meta);before_output=e.states.probe();reqs=e.requests(anchor,anchor+16);bank=[]
  try:
   for req in normalize_requests(reqs):
    oracle=Oracle(e.model,e.tok,req,e.hp,e.module.CONTEXT_TEMPLATES_CACHE,e.module.find_fact_lookup_idx)
    target,delta=e.module.compute_z(e.model,e.tok,req,e.hp,8,e.module.CONTEXT_TEMPLATES_CACHE,return_delta=True,verbose=False)
    h=oracle.initial[0];points=[torch.zeros_like(delta),delta*.5,delta]
    vals=[]
    for d in points:
     f,g,parts=oracle(d);vals.append((f,g.cpu(),parts))
    # Actual suffix forward/backward, including RMSNorm and attention
    # scratch arithmetic, uses FP64 on promoted immutable FP32 inputs.
    with fp64_reference(e.model,oracle) as patches:
     vals64=[]
     for d in points:
      f,g,parts=oracle(d.double());vals64.append((f,g.cpu(),parts))
     if not bank:
      direction=torch.zeros_like(delta,dtype=torch.float64);direction[0]=1.
      step=1e-4*max(1.,oracle.radius)
      middle=points[1].double()
      if not torch.count_nonzero(middle):middle=.01*oracle.radius*direction
      fm,gm,_=oracle(middle);fp,_,_=oracle(middle+step*direction);fn,_,_=oracle(middle-step*direction)
      central=(fp-fn)/(2*step);ad=float(gm@direction)
      fo,_,_=oracle(step*direction);one_sided=(fo-vals64[0][0])/step
      expected_origin=float(vals64[0][1][0])+e.hp.v_weight_decay/float(oracle.initial.norm())**2
      save(e.output/f'calibration/directional-{anchor}.json',dict(case_id=req['case_id'],step=step,central_FD=central,AD=ad,absolute_difference=abs(central-ad),origin_one_sided=one_sided,origin_norm_subgradient_directional=expected_origin,origin_difference=abs(one_sided-expected_origin),extra_FP64_oracle_calls=4,in_96_point_tolerance_population=False,finite=True,numerical_tolerance_PASS_claim=False))
    if len(self.probes)==0:save(e.output/'calibration/FP64-adapter.json',dict(patches=patches,cache_regeneration=False,model_values='FP32_PROMOTED_EXACTLY',restored_FP32=True))
    scale=max(1.,float(vals[0][1].norm()))
    for j,d in enumerate(points):
     x=d.detach().cpu().double();g32=vals[j][1].double();g64=vals64[j][1]
     reference_pg=project(x-g64,oracle.radius)-x
     production_pg=(project(x.float()-g32.float(),oracle.radius)-x.float()).double()
     error=float((production_pg-reference_pg).norm())/scale
     self.probes.append(dict(anchor=anchor,case_id=req['case_id'],point=j,normalized_error=error,gradient_relative_error=float((g32-g64).norm())/max(float(g64.norm()),1e-30),f32=vals[j][0],f64=vals64[j][0],input_identity=oracle.batch['identity'],delta_sha=tensor_sha(d),actual_FP64_forward_backward=True))
    # Only small FP32 prefix caches and deltas, no model/edit checkpoint.
    bank.append(dict(request=req,delta=delta.detach().cpu(),native_target_sha=tensor_sha(target),native_final=vals[2][0],prefix=castcache(oracle.prefix,torch.float32),initial=oracle.initial.clone(),kl=oracle.kl.clone(),batch=oracle.batch,radius=oracle.radius))
   self.banks[anchor]=bank
   save(e.output/f'calibration/probes-{anchor}.json',dict(probes=[p for p in self.probes if p['anchor']==anchor],status='PROBES_RECORDED',independent_per_anchor=16))
  finally:
   assert e.states.identity(e.meta)==before,'PROBE_CHANGED_EDITOR_STATE'
   assert e.states.probe()==before_output,'FP64_PROBE_CHANGED_OUTPUT'
 def finish(self,snapshot0,snapshot1):
  e=self.e;tol=max(10*np.finfo(np.float32).eps,5*max(float(np.quantile([p['normalized_error'] for p in self.probes if p['anchor']==a],.95)) for a in [0,1000]))
  if tol>1e-3:
   lock=dict(status='BLOCKED',tol=tol,cap=None,reasons=['PRECISION_FLOOR_GT_1E-3'],production=False)
  else:
   for anchor,snap in [(0,snapshot0),(1000,snapshot1)]:
    e.meta=e.states.restore(snap)
    for b in self.banks[anchor]:
     oracle=Oracle.__new__(Oracle);oracle.model=e.model;oracle.hp=e.hp;oracle.prefix=b['prefix'];oracle.initial=b['initial'];oracle.kl=b['kl'];oracle.batch=b['batch'];oracle.radius=b['radius'];oracle.calls=0;oracle.seconds=0.;oracle.tokens=0;oracle.origin_subgradient=None
     target,rec=oracle.spg(tol,400);rec.update(anchor=anchor,case_id=b['request']['case_id'],native_final_loss=b['native_final'])
     _,adam=oracle.adam(rec['calls']);rec['matched_adam']=adam
     self.results.append(rec)
   lock=calibrate(self.probes,self.results)
  e.meta=e.states.restore(snapshot1)
  save(e.output/'calibration/lock.json',dict(**lock,probes=self.probes,results=self.results,bindings=e.lock['bindings']));self.banks={};return lock
