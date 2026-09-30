"""FP64 joint allocation and numerical checks; no regularizer/jitter."""
import torch

def sym(a):return (a+a.T)*.5

def relative(a,b):return float((a-b).norm()/b.norm().clamp_min(torch.finfo(b.dtype).tiny))

def psd(g):
 e=torch.linalg.eigvalsh(sym(g));scale=max(1.,float(e.abs().max()))
 if float(e.min()) < -1e-10*scale:raise ArithmeticError('CAPACITY_NOT_PSD')
 return dict(eigenvalues=e.tolist(),min_eigenvalue=float(e.min()),max_eigenvalue=float(e.max()),rank=int((e>1e-10*scale).sum()),diagonal=g.diag().tolist(),trace=float(g.trace()),offdiagonal_norm=float((g-torch.diag(g.diag())).norm()))

def capacity(k,adj,direct,check=False):
 f=sym(k.T@adj);eye=torch.eye(f.shape[0],dtype=f.dtype,device=f.device)
 gap=1-float(torch.linalg.eigvalsh(f).max());fallback=gap<1e-6;reason='CONDITION' if fallback else None
 try:
  g=sym(torch.linalg.solve(eye-f,f));psd(g)
 except (ArithmeticError,RuntimeError):fallback=True;reason='PSD_OR_SOLVE';g=None
 rec=dict(gap=gap,direct_fallback=False,reason=reason)
 if check or fallback:
  ak=direct(k);gd=sym(k.T@ak);psd(gd)
  err=relative(g,gd) if g is not None else None
  identity=relative(adj,torch.linalg.solve(eye+gd,ak.T).T)
  rec.update(direct_comparison_error=err,adj_identity_error=identity)
  if identity>1e-8:raise ArithmeticError('ADJ_IDENTITY_ERROR')
  if fallback or err>1e-8:g=gd;rec.update(direct_fallback=True,reason=reason or 'RECOVERY_ERROR')
 assert g is not None
 rec.update(psd(g));return g,rec

def joint_residual(r,gs):
 if len(gs)==1:return r # contractual bit-exact L8
 eye=torch.eye(gs[0].shape[0],device=r.device,dtype=r.dtype);b=eye+sum(gs)
 torch.linalg.cholesky(b)
 x=torch.linalg.solve(b,eye+gs[0]);res=relative(b@x,eye+gs[0])
 if res>1e-8:raise ArithmeticError('JOINT_SOLVE_RESIDUAL')
 return r@x

def geometry(r0,before,after,delta,k,a):
 d=delta@k;obs=before-after;dn=float(d.norm());on=float(obs.norm());rnorm=float(r0.norm())
 return dict(residual_norm=float(after.norm()),q=float(after.norm())/rnorm if rnorm else None,
 D_norm=dn,observed_norm=on,alpha=float((obs*d).sum())/dn**2 if dn else None,
 cosine=float((obs*d).sum())/(dn*on) if dn and on else None,error=float((obs-d).norm())/dn if dn else None,
 update_norm=float(delta.norm()),energy=float(((delta@a)*delta).sum()))
