"""AST scoped adapter of pinned BLUE. Native z/solve/materialization/append retained."""
import ast,copy,inspect,time,types
import torch
from project.run_scripts.memit_history_lifelong.io import tensor_sha
from . import algebra
from .oracle import Oracle
from project.run_scripts.single_layer_mechanism_first.z_hook import instrument_native_compute_z
import sys
from project.run_scripts.memit_history_lifelong.method import Proxy

class Adapter:
 def __init__(self,module,model,tok,hp,history):
  self.module,self.model,self.tok,self.hp,self.H=module,model,tok,hp,history
  self.mode='divisor';self.zmode='native';self.zlock=None;self.entry_count=0;self.anchor_H=None
  self.native_z=instrument_native_compute_z(module.compute_z,self.native_event)
  self.function=self.compile();ns=dict(module.__dict__);ns['execute_memit']=self.function
  f=module.apply_memit_seq_to_model;self.apply=types.FunctionType(f.__code__,ns,f.__name__,f.__defaults__,f.__closure__)
 def compile(self):
  tree=ast.parse(inspect.getsource(self.module.execute_memit));owner=self;changes=[0,0,0]
  class Rewrite(ast.NodeTransformer):
   def visit_Assign(self,n):
    if any(isinstance(x,ast.Name) and x.id=='resid' for x in n.targets):
     assert ast.unparse(n.value)=='targets / (len(hparams.layers) - i)'
     n.value=ast.parse('_adapter.residual(i, layer, targets, adj_k, layer_ks, model, tok, requests, hparams, context_templates, cache_c)',mode='eval').body;changes[0]+=1
    if any(isinstance(x,ast.Name) and x.id=='zs' for x in n.targets):return [n,ast.parse('_adapter.zs = zs.detach().clone()').body[0]]
    if any(isinstance(x,ast.Subscript) and ast.unparse(x).startswith('weights[weight_name]') for x in n.targets):
     changes[1]+=1;return [n,ast.parse('_adapter.after_layer(i, layer, targets, upd_matrix, layer_ks, model, tok, requests, hparams, weights_copy[weight_name], weights[weight_name])').body[0]]
    return self.generic_visit(n)
   def visit_AugAssign(self,n):
    if ast.unparse(n.target).startswith('cache_c['):changes[2]+=1;return [ast.parse('_adapter.append_start = time.monotonic()').body[0],n,ast.parse('_adapter.appended(i, layer_ks)').body[0]]
    return self.generic_visit(n)
  tree=Rewrite().visit(tree);assert changes==[1,1,1],changes
  ns=dict(self.module.__dict__);ns.update(_adapter=self,compute_z=self.z,compute_ks=self.key,time=time,torch=Proxy(self.module.torch,linalg=Proxy(self.module.torch.linalg,solve=self.native_solve)))
  exec(compile(ast.fix_missing_locations(tree),__file__+':pinned_execute_adapter','exec'),ns);return ns['execute_memit']
 def timed(self,key,fn,*args,**kw):
  t=time.monotonic();prior=getattr(self.model,'_hj_phase',None)
  try:
   setattr(self.model,'_hj_phase',key);v=fn(*args,**kw)
  finally:setattr(self.model,'_hj_phase',prior)
  torch.cuda.synchronize();self.record['timers'][key]=self.record['timers'].get(key,0.)+time.monotonic()-t
  self.record['calls'][key]=self.record['calls'].get(key,0)+1;return v
 def native_solve(self,a,k,*args,**kw):
  assert len(self.record['layers'])==self.record['calls'].get('native_solve',0)
  return self.timed('native_solve',self.module.torch.linalg.solve,a,k,*args,**kw)
 def native_event(self,kind,it,delta,loss,nll,kl,decay):
  if kind=='loss':self._ztrace.append(dict(iteration=it,total=float(loss),nll=float(nll),kl=float(kl),decay=float(decay),delta_norm=float(delta.norm())))
  frame=sys._getframe(1);tokens=frame.f_locals['input_tok']['input_ids'].numel()
  key='native_loss_calls' if kind=='loss' else 'native_backward_calls'
  self.record['calls'][key]=self.record['calls'].get(key,0)+1
  key='token_forward' if kind=='loss' else 'token_backward'
  self.record['calls'][key]=self.record['calls'].get(key,0)+tokens
 def z(self,model,tok,req,hp,layer,ctx):
  if self.preloaded is not None:
   identity,z=next(self.preloaded);assert identity==req['case_id']
   self.record['z'].append(dict(case_id=identity,sha256=tensor_sha(z),reused_same_entry_shadow=True));self.zvectors.append((identity,z.clone()));return z.cuda()
  if self.zmode=='native':
   self._ztrace=[]
   z=self.timed('native_z',self.native_z,model,tok,req,hp,layer,ctx)
   rec=dict(case_id=req['case_id'],sha256=tensor_sha(z),loss_trace=self._ztrace,native_loss_calls=len(self._ztrace))
  else:
   assert self.zlock and self.zlock['status']=='PASS','Z_UNCALIBRATED'
   oracle=Oracle(model,tok,req,hp,ctx,self.module.find_fact_lookup_idx)
   z,rec=self.timed('SPG_z',oracle.spg,self.zlock['tol'],self.zlock['cap']);rec['case_id']=req['case_id']
   if rec['status']=='NONFINITE':raise FloatingPointError('Z_NONFINITE')
  self.record['z'].append(rec);self.zvectors.append((req['case_id'],z.detach().cpu().clone()));return z
 def key(self,*args,**kw):
  index=self.record['calls'].get('native_key',0);assert args[4]==4+index%5 and index<10
  assert len(self.record['layers'])==(index if index<5 else 5)
  k=self.timed('native_key',self.module.compute_ks,*args,**kw)
  self.record['keys'].append(dict(layer=args[4],phase='pre_layer' if index<5 else 'post_all_layers',sha256=tensor_sha(k)))
  return k
 def A(self,i):
  if i not in self.matrices:
   h=self.hp;cov=self.module.get_cov(self.model,self.tok,h.rewrite_module_tmp.format(h.layers[i]),h.mom2_dataset,h.mom2_n_samples,h.mom2_dtype)
   self.matrices[i]=h.mom2_update_weight*cov.double()+self.H[i].cuda().double()
  return self.matrices[i]
 def direct(self,i,k):
  if i not in self.factors:self.factors[i]=self.timed('factorization',torch.linalg.lu_factor,self.A(i))
  return self.timed('A_solve',torch.linalg.lu_solve,*self.factors[i],k)
 def upper(self,j,requests,ctx):
  k=self.timed('lookahead_key',self.module.compute_ks,self.model,self.tok,requests,self.hp,self.hp.layers[j],ctx).T.double()
  g=self.timed('Gram',lambda:algebra.sym(k.T@self.direct(j,k)));algebra.psd(g);return g
 def residual(self,i,layer,r,adj,k,model,tok,requests,hp,ctx,history):
  assert history is self.H and hp.blue is False
  if i==0:self.r0=r.detach().clone()
  if self.mode=='frozen_upper_joint' and not self.frozen:
   self.frozen={j:self.upper(j,requests,ctx) for j in range(1,5)}
  g,rec=algebra.capacity(k,adj,lambda kk:self.direct(i,kk),check=self.entry_count in [0,1000,5000,9000] or self.technical)
  # Residual check of the actual native solve, no extra direct solve in ordinary batches.
  a=self.A(i);solve_error=algebra.relative(a@adj+k@(k.T@adj),k)
  if solve_error>1e-8:raise ArithmeticError('NATIVE_SOLVE_RESIDUAL')
  rec.update(layer=layer,solve_residual=solve_error,prior_H_sha=tensor_sha(history[i]))
  gs=[g]
  if self.mode in ['joint','frozen_upper_joint']:
   for j in range(i+1,5):gs.append(self.frozen[j] if self.mode=='frozen_upper_joint' else self.upper(j,requests,ctx))
   resid=algebra.joint_residual(r,gs)
  else:resid=r/(5-i)
  rec['remaining_trace_capacity']=[float(x.trace()) for x in gs];rec['mode']=self.mode
  self.record['layers'].append(rec);return resid
 def after_layer(self,i,layer,r,delta,k,model,tok,requests,hp,entry_weight,materialized_weight):
  cur=self.timed('realization_forward',self.module.get_module_input_output_at_words,model,tok,8,
   context_templates=[x['prompt'] for x in requests],words=[x['subject'] for x in requests],module_template=hp.layer_module_tmp,fact_token_strategy=hp.fact_token)[1].T
  after=(self.zs-cur).double();actual=materialized_weight.double()-entry_weight.double()
  g=algebra.geometry(self.r0,r,after,actual,k,self.A(i));g['ideal_to_FP32_update_relative']=algebra.relative(actual,delta);g['ideal_energy']=float(((delta@self.A(i))*delta).sum());self.record['layers'][i].update(g)
  if self.anchor_H is not None:
   ai=self.A(i)+(self.anchor_H[i]-self.H[i]).cuda().double()
   self.record['layers'][i]['anchor_A_energy']=float(((actual@ai)*actual).sum());del ai
  if self.capture_updates:self.updates.append(delta.detach().cpu().clone())
 def appended(self,i,k):
  self.record['timers']['history_append']=self.record['timers'].get('history_append',0.)+time.monotonic()-self.append_start
  assert len(self.record['layers'])==5 and len(self.record['append'])==i
  self.record['append'].append(dict(layer=i+4,key_sha=tensor_sha(k),H_sha=tensor_sha(self.H[i]),phase='POST_ALL_FIVE_WRITES'))
 def run(self,requests,mode='divisor',zmode='native',count=0,technical=False,preloaded=None,capture_updates=False):
  self.mode,self.zmode,self.entry_count,self.technical=mode,zmode,count,technical
  self.record=dict(layers=[],z=[],keys=[],append=[],timers={},calls={});self.matrices={};self.factors={};self.frozen={};self.zvectors=[];self.updates=[];self.capture_updates=capture_updates
  self.preloaded=iter(preloaded) if preloaded is not None else None
  t=time.monotonic()
  try:
   model,h=self.apply(self.model,self.tok,requests,self.hp,copy=False,return_orig_weights=False,cache_template=None,cache_c=self.H)
   assert model is self.model and h is self.H
   assert len(self.record['layers'])==len(self.record['append'])==5
   if self.preloaded is not None:assert next(self.preloaded,None) is None
   self.record.update(seconds=time.monotonic()-t,requests=len(requests),history_appends=5,returned_history_identity=True,mode=mode,zmode=zmode)
   return self.record
  finally:self.matrices={};self.factors={};self.frozen={};self.preloaded=None;torch.cuda.empty_cache()
