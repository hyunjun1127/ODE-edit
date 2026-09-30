"""V2 total-call bounded SPG; every trial and return evaluation charged."""
import math
import torch

def project(x,r):
 n=x.norm();return x if float(n)<=r else x*(r/n)

def pg(x,g,r,scale):return float((project(x-g,r)-x).norm())/scale

def solve(oracle,dim,radius,tol,cap,device='cuda',dtype=torch.float32):
 if cap<2:raise ValueError('RETURN_EVALUATION_RESERVE')
 x=torch.zeros(dim,device=device,dtype=dtype);calls=0;backs=0;accepted=[];parts={}
 def evaluate(x):
  nonlocal calls,parts
  f,g,parts=oracle(x);calls+=1
  if not math.isfinite(f) or not torch.isfinite(g).all():raise FloatingPointError('NONFINITE_SPG')
  return f,g
 f,g=evaluate(x);scale=max(1.,float(g.norm()));accepted=[f];status='NOT_CONVERGED'
 alpha=1/max(pg(x,g,radius,1.),1e-12);step_norm=0.
 if f<.05:status='POLICY_ZERO_STEP'
 else:
  while calls<cap-1:
   if pg(x,g,radius,scale)<=tol:status='CONVERGED';break
   direction=project(x-alpha*g,radius)-x;gd=float(g@direction);fmax=max(accepted[-10:]);lam=1.;ok=False
   for trial in range(50):
    if calls>=cap-1:break
    xn=x+lam*direction;fn,gn=evaluate(xn)
    if fn<=fmax+1e-4*lam*gd:ok=True;break
    backs+=1;lam*=.5
   if not ok:
    status='NOT_CONVERGED' if calls>=cap-1 else 'LINESEARCH_FAILED';break
   step=xn-x;y=gn-g;sy=float(step@y);alpha=min(1e12,max(1e-12,float(step@step)/sy)) if sy>0 else 1e12
   x,f,g=xn,fn,gn;accepted.append(f);step_norm=float(step.norm())
   if len(accepted)>=6:
    window=accepted[-6:];floor=10*torch.finfo(dtype).eps*max(max(map(abs,window)),torch.finfo(dtype).tiny)
    best=[min(window[:i+1]) for i in range(6)]
    if max(abs(window[i+1]-window[i]) for i in range(5))<=floor and best[0]-best[-1]<=floor:
     status='STALLED_AT_PRECISION';break
 f,g=evaluate(x);finalpg=pg(x,g,radius,scale);norm=float(x.norm());boundary=norm>0 and abs(norm/radius-1)<=1e-6
 if status=='CONVERGED' and finalpg>tol:status='NOT_CONVERGED'
 if status=='NOT_CONVERGED' and finalpg<=tol:status='CONVERGED'
 assert calls<=cap and norm<=radius*(1+1e-6)
 return x,dict(status=status,calls=calls,value=f,parts=parts,normalized_PG=finalpg,gradient_scale=scale,norm_over_radius=norm/radius,
 clamp_multiplier=max(0.,-float(g@(x/radius))) if boundary else 0.,boundary=boundary,
 accepted_steps=len(accepted)-1,backtracks=backs,last_step_norm=step_norm,loss_window=accepted[-6:],final_recomputed=True)

def calibrate(probes,results):
 import numpy as np
 errors={a:[p['normalized_error'] for p in probes if p['anchor']==a] for a in [0,1000]}
 if any(not v for v in errors.values()):return dict(status='BLOCKED',tol=None,cap=None,reasons=['INCOMPLETE_PRECISION_PROBES'])
 tol=max(10*np.finfo(np.float32).eps,5*max(float(np.quantile(v,.95)) for v in errors.values()))
 groups={a:[r for r in results if r['anchor']==a] for a in [0,1000]};groups['pooled']=results
 reasons=[]
 if len(probes)!=96 or len(results)!=32:reasons.append('INCOMPLETE_CALIBRATION')
 if tol>1e-3:reasons.append('PRECISION_FLOOR_GT_1E-3')
 if any(r['status'] not in ['CONVERGED','POLICY_ZERO_STEP','STALLED_AT_PRECISION'] for r in results):reasons.append('CENSORED_OR_FAILED')
 medians={}
 for a,rs in groups.items():
  for zero in [True,False]:
   cs=[r['calls'] for r in rs if zero or r['status']!='POLICY_ZERO_STEP'];m=float(np.median(cs)) if cs else None
   medians[str(a)+'_include_zero_'+str(zero)]=m
   if m is not None and m>200:reasons.append('MEDIAN_GT_200')
 cap=max(25,math.ceil(1.25*max(float(np.quantile([r['calls'] for r in rs],.95)) for a,rs in groups.items() if a!='pooled'))) if all(groups[a] for a in [0,1000]) else None
 if cap is not None and cap>400:reasons.append('PRODUCTION_CAP_GT_400')
 assert sum(r['calls'] for r in results)<=12800 and all(r['calls']<=400 for r in results),'CALIBRATION_CALL_CAP'
 return dict(status='BLOCKED' if reasons else 'PASS',tol=tol,cap=cap if not reasons else None,diagnostic_proposed_cap=cap,reasons=sorted(set(reasons)),medians=medians,SPG_calibration_calls=sum(r['calls'] for r in results))
