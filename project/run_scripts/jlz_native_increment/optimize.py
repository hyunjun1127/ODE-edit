"""25 evaluated candidates, <=24 q-Adam updates, total-MEAN early stop."""
import torch
from .subject import native
from .geometry import allocation
from .common import require, write

def coordinates(anchors, dimensions):
    m = len(anchors)
    return {l:(a.double()/(dimensions[l][0]*m)**.5).float() for l,a in anchors.items()}

@torch.no_grad()
def clamp(q, scales, anchors):
    receipts = {}
    for l in q:
        norm = torch.linalg.vector_norm(scales[l]*q[l],dim=0)
        ratio = norm/(.75*anchors[l])
        q[l].mul_(torch.where(ratio > 1,ratio.reciprocal(),torch.ones_like(ratio)))
        after = torch.linalg.vector_norm(scales[l]*q[l],dim=0)
        require(bool((after <= .75*anchors[l]+5e-6).all()), 'PHYSICAL_CLAMP')
        receipts[l] = dict(clipped=int((ratio>1).sum()),pre_ratio=ratio.tolist(),post_norm=after.tolist())
    return receipts

def stop(mean, candidate):
    require(torch.isfinite(torch.tensor(mean)), 'NONFINITE_TOTAL_MEAN')
    return mean < .05 or candidate == 25

def fit(adapter,entry,frozen,chain,out):
    require(chain in ('MAIN','NOALLOC'), 'FIT_CHAIN')
    coefficient = .1 if chain=='MAIN' else 0.
    B=entry['pack']['n_requests'];scales=coordinates(entry['anchors'],adapter.dims)
    q={l:torch.zeros((adapter.dims[l][0],B),device=adapter.device,requires_grad=True) for l in adapter.sites}
    optimizer=torch.optim.Adam(list(q.values()),lr=.1,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
    updates=0;visited=[]
    for candidate in range(1,26):
        D={l:(scales[l]*q[l]).detach().requires_grad_(True) for l in q}
        # First evaluation decides stopping; no optimizer backward at a stopped candidate.
        value=native(adapter,entry,D)
        policy,costs=allocation(D,frozen,coefficient)
        mean=value['mean']+float(policy.detach())
        boundary_reference=None
        if abs(mean-.05)<=2e-5+2e-4*.05:
            prior_route=adapter.native_route
            try:
                adapter.native_route='full';reference=native(adapter,entry,D)
            finally:adapter.native_route=prior_route
            boundary_reference=dict(optimized_mean=mean,reference_mean=reference['mean']+float(policy.detach()),
                                    extra_forward_groups=reference['forward_groups'])
            mean=boundary_reference['reference_mean']
        terminal=stop(mean,candidate)
        diagnostic=candidate in (2,9) or terminal
        row=dict(candidate=candidate,updates_before=updates,total_mean=mean,native=value['components'],
                 allocation_mean=float(policy.detach()),layer_cost={l:float(v.detach()) for l,v in costs.items()},
                 lr=.1,early_stop=mean<.05,terminal=terminal,forward_evaluation={k:v for k,v in value.items() if k!='gradients'},
                 gradient_measured=False,optimizer_backward=False,diagnostic_backward=False,boundary_reference=boundary_reference)
        row['delta_over_anchor']={l:(d.detach().norm(dim=0)/entry['anchors'][l]).tolist() for l,d in D.items()}
        row['planned_norm']={l:d.detach().norm(dim=0).tolist() for l,d in D.items()}
        with torch.no_grad():
            row['proxy_GE']={l:{name:float(((d.double().T@d.double())*frozen[l][name].T).sum())
                for name in ('G','E')} for l,d in D.items() if 'G' in frozen[l]}
        if not terminal or diagnostic:
            gradient=native(adapter,entry,D,True,diagnostic)
            pg=torch.autograd.grad(B*policy,tuple(D.values()))
            for l,g in zip(D,pg):D[l].grad.add_(g)
            require(all(torch.isfinite(d.grad).all() for d in D.values()),'NONFINITE_D_GRADIENT')
            row.update(gradient_measured=True,optimizer_backward=not terminal,diagnostic_backward=diagnostic,
                       recompute_forward_groups=gradient['forward_groups'],backward_calls=gradient['backward_calls']+1,
                       recompute_seconds=gradient['seconds'],D_gradient_norm={l:float(d.grad.norm()) for l,d in D.items()})
            if diagnostic:
                row['component_gradient_norm']={name:{l:float(g.norm()/B) for l,g in gs.items()} for name,gs in gradient['gradients'].items()}
                row['component_gradient_norm']['allocation']={l:float(g.norm()/B) for l,g in zip(D,pg)}
                components=dict(gradient['gradients'],allocation={l:g for l,g in zip(D,pg)})
                row['component_direction']={}
                for name,gs in components.items():
                    values={}
                    for l,g in gs.items():
                        total=D[l].grad;d=D[l].detach();gn=float(g.norm());tn=float(total.norm());dn=float(d.norm())
                        values[l]=dict(D_grad_mean_norm=gn/B,q_grad_SUM_norm=float((scales[l]*g).norm()),
                            radial_cosine=float((g*d).sum())/(gn*dn) if gn and dn else None,
                            total_cosine=float((g*total).sum())/(gn*tn) if gn and tn else None)
                    row['component_direction'][name]=values
            if not terminal:
                optimizer.zero_grad(set_to_none=True)
                for l in q:q[l].grad=scales[l]*D[l].grad # native SUM already contains B once
                optimizer.step();updates+=1
                row['post_update_clamp']=clamp(q,scales,entry['anchors'])
        row['updates_after']=updates
        write(out/f'candidate-{candidate:02d}.json',row);visited.append(row)
        if terminal:
            return {l:d.detach() for l,d in D.items()},dict(candidates=candidate,updates=updates,
                early_stop=mean<.05,terminal_total_mean=mean,diagnostic_candidates=[r['candidate'] for r in visited if r['diagnostic_backward']],
                native_forward_groups=sum(r['forward_evaluation']['forward_groups']+r.get('recompute_forward_groups',0) for r in visited),
                backward_calls=sum(r.get('backward_calls',0) for r in visited),no_pulse=True,no_replay=True)
    raise AssertionError('CANDIDATE_BUDGET')
