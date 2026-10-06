"""Absolute-R Euclidean capped weighted group-L1, FP64 sorted breakpoints."""
import math
import time
import torch


class ProjectionFailure(RuntimeError):
    def __init__(self,code,receipt):
        super().__init__(code);self.receipt=dict(status=code,**receipt)


def _lengths(norms,weights,caps,beta):
    local=[min(n,d) for n,d in zip(norms,caps)]
    spend=sum(w*t for w,t in zip(weights,local))
    if spend<=beta:return local,0.,False,'SHARED_INACTIVE'
    knots=sorted(set([0.]+[v for n,w,d in zip(norms,weights,caps)
                         for v in ((n-d)/w,n/w) if v>=0]))
    for ix,lo in enumerate(knots):
        at_lo=[min(d,max(n-lo*w,0.)) for n,w,d in zip(norms,weights,caps)]
        spend_lo=sum(w*t for w,t in zip(weights,at_lo))
        if spend_lo==beta:return at_lo,lo,True,'EXACT_BREAKPOINT_MINIMUM_TAU'
        if ix+1==len(knots):
            if beta==0:return at_lo,lo,True,'ZERO_BUDGET_FINAL_BREAKPOINT'
            continue
        hi=knots[ix+1];mid=lo+(hi-lo)/2
        free=[i for i,(n,w,d) in enumerate(zip(norms,weights,caps)) if 0<n-mid*w<d]
        capped=[i for i,(n,w,d) in enumerate(zip(norms,weights,caps)) if n-mid*w>=d]
        denominator=sum(weights[i]**2 for i in free)
        if denominator==0:
            # A plateau never divides by zero; its deterministic dual is the
            # left endpoint only when the plateau's exact spend is beta.
            if spend_lo==beta:return at_lo,lo,True,'FLAT_INTERVAL_MINIMUM_TAU'
            continue
        numerator=sum(weights[i]*norms[i] for i in free)+sum(weights[i]*caps[i] for i in capped)-beta
        tau=numerator/denominator
        if math.isfinite(tau) and lo<=tau<=hi:
            return [min(d,max(n-tau*w,0.)) for n,w,d in zip(norms,weights,caps)],tau,True,'SORTED_BREAKPOINT_ROOT'
    raise ProjectionFailure('PROJECTION_SORTED_BREAKPOINT_ROOT_UNAVAILABLE',
                            dict(proposal_norm=norms,weights=weights,caps=caps,beta=beta,breakpoints=knots))


@torch.no_grad()
def project_capped_weighted_l1(R,caps,weights,beta):
    started=time.monotonic();layers=tuple(R);first=R[layers[0]];B=first.shape[1]
    cap=torch.as_tensor(caps,device='cpu',dtype=torch.float64)
    weight=torch.as_tensor(weights,device='cpu',dtype=torch.float64)
    budget=torch.as_tensor(beta,device='cpu',dtype=torch.float64)
    if cap.shape!=(len(layers),B) or weight.shape!=cap.shape or budget.shape!=(B,):raise RuntimeError('PROJECTION_SCHEMA')
    if not bool(torch.isfinite(cap).all() and torch.isfinite(weight).all() and torch.isfinite(budget).all()
                and (cap>=0).all() and (weight>0).all() and (budget>=0).all()):raise RuntimeError('PROJECTION_DOMAIN')
    for l in layers:
        if R[l].ndim!=2 or R[l].shape[1]!=B or R[l].dtype!=torch.float32 or not bool(torch.isfinite(R[l]).all()):raise RuntimeError('PROJECTION_R_FP32')
    norm=torch.stack([R[l].double().norm(dim=0).cpu() for l in layers]);length=torch.zeros_like(norm)
    taus=[];active=[];conventions=[];kkt=[]
    for r in range(B):
        ns=norm[:,r].tolist();ws=weight[:,r].tolist();ds=cap[:,r].tolist();br=float(budget[r])
        ts,tau,on,convention=_lengths(ns,ws,ds,br)
        operands=dict(owner=r,proposal_norm=ns,weights=ws,caps=ds,beta=br,tau=tau,projected_norm=ts)
        if not math.isfinite(tau) or tau<0:raise ProjectionFailure('PROJECTION_DUAL_NONNEGATIVE',operands)
        expression=[min(d,max(n-tau*w,0.)) for n,w,d in zip(ns,ws,ds)]
        errors=[abs(t-e) for t,e in zip(ts,expression)]
        if any(error>1e-10*max(1,n,d) for error,n,d in zip(errors,ns,ds)):
            raise ProjectionFailure('PROJECTION_LENGTH_KKT',dict(**operands,length_errors=errors))
        spend=sum(w*t for w,t in zip(ws,ts));excess=max(spend-br,0.)
        comp=abs(tau*(spend-br));comp_limit=1e-10*max(1,tau*br)
        if excess>1e-10*max(1,br) or comp>comp_limit:
            raise ProjectionFailure('PROJECTION_SHARED_KKT',dict(**operands,spend=spend,shared_excess=excess,
                shared_limit=1e-10*max(1,br),complementarity=comp,complementarity_limit=comp_limit))
        length[:,r]=torch.tensor(ts,dtype=torch.float64);taus.append(tau);active.append(on);conventions.append(convention)
        kkt.append(dict(owner=r,length_error_max=max(errors),shared_excess=excess,
                        complementarity=comp,complementarity_limit=comp_limit))
    stored={}
    for i,l in enumerate(layers):
        n=norm[i].to(R[l].device);t=length[i].to(R[l].device)
        scale=torch.zeros_like(n);nonzero=n>0;scale[nonzero]=t[nonzero]/n[nonzero]
        stored[l]=(R[l].double()*scale[None,:]).float()
    storednorm=torch.stack([stored[l].double().norm(dim=0).cpu() for l in layers])
    local_excess=torch.clamp(storednorm-cap,min=0);spend=(weight*storednorm).sum(0)
    shared_excess=torch.clamp(spend-budget,min=0);shared_limit=1e-6*torch.maximum(torch.ones_like(budget),budget)
    if not bool(torch.isfinite(storednorm).all() and (local_excess<=1e-6).all()
                and (shared_excess<=shared_limit).all()):
        raise ProjectionFailure('POSTCAST_PRICED_FEASIBILITY',dict(local_excess=local_excess.tolist(),
            shared_excess=shared_excess.tolist(),shared_limit=shared_limit.tolist(),local_limit=1e-6,
            stored_norm=storednorm.tolist(),weighted_spend=spend.tolist(),beta=budget.tolist()))
    zero=length==0;capped=(length==cap)&~zero;free=~(zero|capped)
    return stored,dict(tau=taus,shared_active=active,dual_convention=conventions,
        pre_norm=norm.tolist(),post_norm=storednorm.tolist(),fp64_projected_norm=length.tolist(),
        weighted_spend=spend.tolist(),beta=budget.tolist(),zero_mask=zero.tolist(),capped_mask=capped.tolist(),free_mask=free.tolist(),
        local_excess=local_excess.tolist(),shared_excess=shared_excess.tolist(),shared_limit=shared_limit.tolist(),
        local_limit=1e-6,fp64_kkt=kkt,postcast_repair=False,coordinate='absolute_R_Euclidean',
        seconds=time.monotonic()-started)
