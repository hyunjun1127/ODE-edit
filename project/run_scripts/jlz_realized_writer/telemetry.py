"""No tensors persisted; signed directional quantities and zero semantics."""
import torch

def vector_metrics(action,target,anchor):
    x,y=action.double(),target.double();nx=x.norm(dim=0);ny=y.norm(dim=0);dot=(x*y).sum(0)
    error=(x-y).norm(dim=0);rows=[]
    for i in range(x.shape[1]):
        n=float(ny[i]);a=float(nx[i]);d=float(dot[i]);e=float(error[i])
        rows.append(dict(action_norm=a,target_norm=n,error_norm=e,
            norm_ratio=a/n if n else None,directional_ratio=d/(n*n) if n else None,
            cosine=d/(a*n) if a and n else None,relative_error=e/n if n else None,
            ratio_status='DEFINED' if n else 'ZERO_TARGET',
            zero_target_leakage_over_anchor=a/float(anchor[i]) if not n else None))
    return rows

def shares(plan,actions,anchors,layers):
    p=torch.stack([plan[l].double().norm(dim=0) for l in layers])
    x=torch.stack([actions[l].double().norm(dim=0)/anchors[l].double().cpu() for l in layers])
    rows=[]
    for r in range(p.shape[1]):
        pt=float(p[:,r].sum());xt=float(x[:,r].sum())
        ps=p[:,r]/pt if pt else None;xs=x[:,r]/xt if xt else None
        rows.append(dict(owner=r,planned_share=None if ps is None else ps.tolist(),
            direct_share=None if xs is None or ps is None else xs.tolist(),
            status='NO_PLANNED_EDIT' if not pt else 'ZERO_REALIZED_TOTAL' if not xt else 'DEFINED',
            share_L1=float((ps-xs).abs().sum()) if ps is not None and xs is not None else None,
            realized_relative_total=xt))
    return rows
