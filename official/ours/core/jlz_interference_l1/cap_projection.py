"""Tagged FP64 group-L1 endpoints; true nullable uncapped branch.

Only exact tagged equality canonicalizes a selected endpoint. The inherited
1e-10 KKT knot acceptance remains fixed; no near-endpoint ULP rule is added.
Positive interior lengths and Adam moments are never threshold-pruned.
"""
import math
import time
import torch
from .projection import ProjectionFailure


def lengths(norms,weights,caps,beta):
    def expression(tau):
        positive=[max(n-tau*w,0.) for n,w in zip(norms,weights)]
        return positive if caps is None else [min(d,t) for d,t in zip(caps,positive)]
    tagged=[(0.,-1,'ORIGIN')]
    for i,(n,w) in enumerate(zip(norms,weights)):
        tagged.append((n/w,i,'ZERO'))
        if caps is not None and (n-caps[i])/w>=0:tagged.append(((n-caps[i])/w,i,'CAP'))
    tagged.sort();knots=sorted({x[0] for x in tagged})
    def at(tau):
        raw=expression(tau);values=list(raw);owners=[]
        for value,i,kind in tagged:
            if value==tau and i>=0:
                values[i]=0. if kind=='ZERO' else caps[i]
                owners.append(dict(block=i,kind=kind,value=value))
        return values,dict(selected_endpoints=owners,raw_expression=raw,
            correction=[v-r for v,r in zip(values,raw)],near_endpoint_rule='NONE_EXACT_TAG_EQUALITY_ONLY')
    local,meta=at(0.)
    if sum(w*t for w,t in zip(weights,local))<=beta:return local,0.,False,'SHARED_INACTIVE',meta
    for ix,lo in enumerate(knots):
        vals,meta=at(lo);spend=sum(w*t for w,t in zip(weights,vals))
        if (abs(spend-beta)<=1e-10*max(1,beta)
                and abs(lo*(spend-beta))<=1e-10*max(1,lo*beta)):
            return vals,lo,True,'TAGGED_BREAKPOINT_KKT_MINIMUM_TAU',meta
        if ix+1==len(knots):continue
        hi=knots[ix+1];mid=lo+(hi-lo)/2
        free=[i for i,(n,w) in enumerate(zip(norms,weights))
              if n-mid*w>0 and (caps is None or n-mid*w<caps[i])]
        capped=[] if caps is None else [i for i,(n,w,d) in enumerate(zip(norms,weights,caps)) if n-mid*w>=d]
        denom=sum(weights[i]**2 for i in free)
        if not denom:continue
        numerator=sum(weights[i]*norms[i] for i in free)+sum(weights[i]*caps[i] for i in capped)-beta
        tau=numerator/denom
        if math.isfinite(tau) and lo<=tau<=hi:
            vals,meta=at(tau)
            return vals,tau,True,'TAGGED_OR_INTERIOR_AFFINE_ROOT',meta
    raise ProjectionFailure('PROJECTION_SORTED_BREAKPOINT_ROOT_UNAVAILABLE',
        dict(proposal_norm=norms,weights=weights,caps=caps,beta=beta,tagged_knots=tagged))


def first_store(ideal, cap_columns, mode):
    """Predeclared first conversion, never selected from a feasibility result.

    The default is the historical nearest FP32 conversion. The opt-in repair
    uses toward-zero component rounding ONLY for ideal cap endpoint columns.
    Interior columns, the ideal Euclidean solution and Adam state are unchanged.
    """
    if mode not in ('nearest', 'cap_endpoint_toward_zero_v1'):
        raise RuntimeError('ENDPOINT_CAST_MODE')
    nearest = ideal.float()
    if mode == 'nearest':
        return nearest, None
    outward = (nearest.double().abs() > ideal.abs()) & cap_columns[None, :]
    stored = torch.where(outward, torch.nextafter(nearest, torch.zeros_like(nearest)), nearest)
    if bool((stored[:, cap_columns].double().abs() > ideal[:, cap_columns].abs()).any()):
        raise RuntimeError('CAP_ENDPOINT_DIRECTED_CAST')
    correction = (stored.double() - nearest.double()).abs()
    evidence = dict(rounding_rule=mode, selected_cap_columns=cap_columns.cpu().tolist(),
        changed_component_count=outward.sum(0).cpu().tolist(),
        max_component_correction=correction.amax(0).cpu().tolist(),
        nearest_norm=nearest.double().norm(dim=0).cpu().tolist(),
        stored_norm=stored.double().norm(dim=0).cpu().tolist(),
        decision_depends_on_postcast_feasibility=False, scalar_rescale=False,
        positive_interior_nearest_unchanged=True)
    return stored, evidence


@torch.no_grad()
def project_capped_weighted_l1(R,caps,weights,beta,endpoint_cast_mode='nearest'):
    started=time.monotonic();layers=tuple(R);B=R[layers[0]].shape[1];L=len(layers)
    cap=None if caps is None else torch.as_tensor(caps,device='cpu',dtype=torch.float64)
    weight=torch.as_tensor(weights,device='cpu',dtype=torch.float64)
    budget=torch.as_tensor(beta,device='cpu',dtype=torch.float64)
    if weight.shape!=(L,B) or budget.shape!=(B,) or (cap is not None and cap.shape!=(L,B)):raise RuntimeError('PROJECTION_SCHEMA')
    if not bool(torch.isfinite(weight).all() and (weight>0).all() and torch.isfinite(budget).all() and (budget>=0).all()):raise RuntimeError('PROJECTION_DOMAIN')
    if cap is not None and not bool(torch.isfinite(cap).all() and (cap>=0).all()):raise RuntimeError('PROJECTION_CAP_DOMAIN')
    if any(v.ndim!=2 or v.shape[1]!=B or v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()) for v in R.values()):raise RuntimeError('PROJECTION_R_FP32')
    norm=torch.stack([R[l].double().norm(dim=0).cpu() for l in layers]);length=torch.zeros_like(norm)
    taus=[];active=[];conventions=[];kkt=[];endpoints=[]
    for r in range(B):
        ns=norm[:,r].tolist();ws=weight[:,r].tolist();ds=None if cap is None else cap[:,r].tolist();br=float(budget[r])
        ts,tau,on,convention,meta=lengths(ns,ws,ds,br)
        operands=dict(owner=r,proposal_norm=ns,weights=ws,caps=ds,beta=br,tau=tau,projected_norm=ts,endpoint=meta)
        if not math.isfinite(tau) or tau<0 or any(not math.isfinite(t) or t<0 for t in ts):raise ProjectionFailure('PROJECTION_DUAL_LENGTH_DOMAIN',operands)
        errors=[abs(t-e) for t,e in zip(ts,meta['raw_expression'])]
        if any(error>1e-10*max(1,ns[i],0 if ds is None else ds[i]) for i,error in enumerate(errors)):
            raise ProjectionFailure('PROJECTION_LENGTH_KKT',operands)
        spend=sum(w*t for w,t in zip(ws,ts));excess=max(spend-br,0.);comp=abs(tau*(spend-br));limit=1e-10*max(1,tau*br)
        if excess>1e-10*max(1,br) or comp>limit:raise ProjectionFailure('PROJECTION_SHARED_KKT',dict(**operands,spend=spend,complementarity=comp))
        length[:,r]=torch.tensor(ts,dtype=torch.float64);taus.append(tau);active.append(on);conventions.append(convention)
        endpoints.append(dict(owner=r,**meta));kkt.append(dict(owner=r,length_error_max=max(errors),shared_excess=excess,complementarity=comp,complementarity_limit=limit))
    stored={};cast_evidence=[]
    for i,l in enumerate(layers):
        n=norm[i].to(R[l].device);t=length[i].to(R[l].device);scale=torch.zeros_like(n);nz=n>0
        scale[nz]=t[nz]/n[nz]
        capped_columns=torch.zeros_like(nz) if cap is None else (t==cap[i].to(R[l].device))&(t>0)
        stored[l],evidence=first_store(R[l].double()*scale[None,:],capped_columns,endpoint_cast_mode)
        if evidence is not None:cast_evidence.append(dict(layer=l,**evidence))
        stored[l][:,t==0]=0.
    storednorm=torch.stack([stored[l].double().norm(dim=0).cpu() for l in layers])
    local_excess=None if cap is None else torch.clamp(storednorm-cap,min=0)
    spend=(weight*storednorm).sum(0);excess=torch.clamp(spend-budget,min=0);limit=1e-6*torch.maximum(torch.ones_like(budget),budget)
    if not bool(torch.isfinite(storednorm).all() and (excess<=limit).all()) or (cap is not None and not bool((local_excess<=1e-6).all())):
        raise ProjectionFailure('POSTCAST_PRICED_FEASIBILITY',dict(stored_norm=storednorm.tolist(),beta=budget.tolist(),weighted_spend=spend.tolist(),local_excess=None if cap is None else local_excess.tolist(),shared_excess=excess.tolist(),
            proposal_norm=norm.tolist(),weights=weight.tolist(),caps=None if cap is None else cap.tolist(),
            tau=taus,fp64_projected_norm=length.tolist(),endpoint_corrections=endpoints,
            endpoint_cast_mode=endpoint_cast_mode,cast_realization=cast_evidence))
    zero=length==0;capped=torch.zeros_like(zero) if cap is None else (length==cap)&~zero;free=~(zero|capped)
    if not torch.equal(zero,storednorm==0):raise RuntimeError('EXACT_ENDPOINT_ZERO_STORAGE')
    return stored,dict(schema='PRICE_CAP_BASE_PROJECTION_V1',cap_mode='none' if cap is None else 'native',
        tau=taus,shared_active=active,dual_convention=conventions,endpoint_corrections=endpoints,
        pre_norm=norm.tolist(),post_norm=storednorm.tolist(),fp64_projected_norm=length.tolist(),
        weighted_spend=spend.tolist(),beta=budget.tolist(),zero_mask=zero.tolist(),capped_mask=None if cap is None else capped.tolist(),free_mask=free.tolist(),
        local_excess=None if cap is None else local_excess.tolist(),shared_excess=excess.tolist(),shared_limit=limit.tolist(),
        local_limit=None if cap is None else 1e-6,fp64_kkt=kkt,postcast_repair=False,
        endpoint_cast_mode=endpoint_cast_mode,cast_realization=cast_evidence,
        coordinate='absolute_R_Euclidean',seconds=time.monotonic()-started)
