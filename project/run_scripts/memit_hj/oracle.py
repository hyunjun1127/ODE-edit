"""Task-local L8 oracle using pinned z_hook helpers and fixed FP32 caches."""
import ast,inspect,time
import torch
from project.run_scripts.single_layer_mechanism_first import z_hook as hook
from .spg import solve,project

# The old hook hardcodes FP32 scratch. Keep original bytes read-only and replace
# only scratch dtype for actual FP64 probes; all objective arithmetic preserved.
def loss_function():
 source=inspect.getsource(hook.native_losses);tree=ast.parse(source)
 class Dtype(ast.NodeTransformer):
  def visit_Attribute(self,n):
   if isinstance(n.value,ast.Name) and n.value.id=='torch' and n.attr=='float32':return ast.parse('delta.dtype',mode='eval').body
   return self.generic_visit(n)
 ns=dict(hook.__dict__);exec(compile(ast.fix_missing_locations(Dtype().visit(tree)),__file__+':dtype_adapter','exec'),ns)
 return ns['native_losses']
LOSSES=loss_function()
def castcache(x,dtype):
 if isinstance(x,torch.Tensor):return x.to(dtype=dtype) if x.is_floating_point() else x
 if isinstance(x,dict):return {k:castcache(v,dtype) for k,v in x.items()}
 if isinstance(x,list):return [castcache(v,dtype) for v in x]
 if isinstance(x,tuple):return tuple(castcache(v,dtype) for v in x)
 return x
class Oracle:
 def __init__(self,model,tok,request,hp,contexts,lookup):
  started=time.monotonic();self.model,self.hp=model,hp;self.batch=hook.prepare_batch(tok,[request],contexts,hp,lookup,'cuda')
  self.prefix=hook.capture_prefix(model,hp,8,self.batch['tokens']);s=self.batch['specs'][0]
  self.initial=self.prefix['hidden'][s['offset'],s['lookup'][0]].detach().clone()[None]
  self.radius=float(hp.clamp_norm_factor*self.initial.norm())
  if not self.radius>0 or not torch.isfinite(self.initial).all():raise FloatingPointError('INVALID_NATIVE_ANCHOR')
  # Native full-model zero-delta KL teacher, rather than regenerating it
  # with a different selected-row GEMM. Only this request's cache survives.
  with torch.no_grad():
   logits=model(**self.batch['tokens'],use_cache=False).logits
   self.kl=logits[self.batch['kl_rows'],self.batch['kl_cols']].log_softmax(-1).detach().clone()
   del logits
  self.prefix_seconds=time.monotonic()-started;self.calls=0;self.seconds=0.;self.tokens=0;self.precision='float32'
  self.origin_subgradient=None
 def __call__(self,delta):
  start=time.monotonic();x=delta.detach().clone().requires_grad_(True)
  lh,fh=hook.suffix_hidden(self.model,self.hp,8,self.prefix,x[None],self.batch)
  assert lh.dtype==fh.dtype==x.dtype,'ORACLE_HIDDEN_DTYPE'
  total,nll,kl,decay,self.kl=LOSSES(self.model,self.hp,self.batch,lh,fh,x[None],self.initial,self.kl)
  assert total.dtype==x.dtype,'ORACLE_OBJECTIVE_DTYPE'
  g=torch.autograd.grad(total[0],x)[0]
  if not torch.isfinite(g).all():raise FloatingPointError('NONFINITE_ORACLE_GRADIENT')
  torch.cuda.synchronize()
  self.calls+=1;self.seconds+=time.monotonic()-start;self.tokens+=self.batch['tokens']['input_ids'].numel()
  parts=dict(nll=float(nll[0]),kl=float(kl[0]),decay=float(decay[0]))
  if not torch.count_nonzero(delta):self.origin_subgradient=max(float(g.norm())-self.hp.v_weight_decay/float(self.initial.norm())**2,0.)
  return float(total[0].detach()),g.detach(),parts
 def dtype(self,dtype):
  self.prefix=castcache(self.prefix,dtype);self.initial=self.initial.to(dtype);self.kl=castcache(self.kl,dtype);self.precision=str(dtype)
 def spg(self,tol,cap):
  delta,receipt=solve(self,self.initial.numel(),self.radius,tol,cap)
  receipt.update(input_identity=self.batch['identity'],oracle_seconds=self.seconds,oracle_calls=self.calls,token_forward=self.tokens,token_backward=self.tokens,origin_norm_subgradient=self.origin_subgradient,prefix_calls=1,full_model_KL_teacher_calls=1,prefix_seconds=getattr(self,'prefix_seconds',0.),prefix_token_forward=self.batch['tokens']['input_ids'].numel(),suffix_layers=23,prefix_layers=9,head_target_positions_per_call=int((self.batch['targets']!=-100).sum()),head_KL_positions_per_call=1)
  return (self.initial[0]+delta).detach(),receipt
 def adam(self,calls):
  x=torch.zeros_like(self.initial[0],requires_grad=True);opt=torch.optim.Adam([x],lr=self.hp.v_lr);count=0
  for i in range(max(1,calls-1)):
   f,g,parts=self(x);count+=1
   if f<.05:break
   x.grad=g;opt.step();opt.zero_grad()
   with torch.no_grad():x.copy_(project(x,self.radius))
  f,g,parts=self(x);count+=1
  return x.detach(),dict(value=f,parts=parts,calls=count,norm_over_radius=float(x.detach().norm())/self.radius)
