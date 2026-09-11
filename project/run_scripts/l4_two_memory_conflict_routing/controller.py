"""Two-channel elastic joint step, including 1/h cost and cumulative anchor."""
import itertools
import math
import torch
from .geometry import dot,finite

def joint_dual(matrix,rhs):
    """Deterministic exact active-set enumeration (<=2), FP64; no scalar clips."""
    n=len(rhs)
    if n>2:raise ValueError('ONLY_TWO_MEMORY_CHANNELS')
    matrix=finite(matrix.double());rhs=finite(rhs.double())
    scale=max(1.,float(matrix.abs().max()) if n else 0.,float(rhs.abs().max()) if n else 0.)
    tol=64*max(n,1)*torch.finfo(torch.float64).eps*scale
    candidates=[]
    for active in itertools.product((False,True),repeat=n):
        mask=torch.tensor(active,device=rhs.device,dtype=torch.bool);lam=torch.zeros_like(rhs)
        if mask.any():lam[mask]=torch.linalg.solve(matrix[mask][:,mask],rhs[mask])
        grad=matrix@lam-rhs
        if (lam>=-tol).all() and (grad[~mask]>=-tol).all() and (grad[mask].abs()<=tol*(1+lam.norm())).all():
            objective=.5*dot(lam,matrix@lam)-dot(rhs,lam)
            candidates.append((float(objective),active,lam,grad))
    if not candidates:raise FloatingPointError('JOINT_KKT_NO_FEASIBLE_ACTIVE_SET')
    _,active,lam,grad=min(candidates,key=lambda x:(x[0],x[1]))
    return lam,dict(active=list(active),dual=lam.tolist(),dual_gradient=grad.tolist(),
                    complementarity=float((lam*grad).abs().max()) if n else 0.,tolerance=tol)

def calibration(metric,anchor,gradients,harms):
    aa=metric.action(anchor)
    if aa==0:return dict(stationary=True,aa=0.,sigma=harms+.01,epsilon=torch.zeros_like(harms))
    sigma=harms+.01
    normalized=[g/s for g,s in zip(gradients,sigma)]
    qref=torch.stack([aa*metric.allowed_white(g).square().sum() for g in normalized]) if normalized else harms.clone()
    return dict(stationary=False,aa=float(aa),sigma=sigma,epsilon=.1*(qref+1e-4),qref=qref,
                terminal_budget=.9*harms/sigma,gradients=normalized)

def step(metric,z,gradients,current,proposal,budget,s,h,epsilon,aa):
    """Risks and gradients are normalized by common OS sigma by the caller."""
    anchor=-h/(1+h)*z
    c=(s+h)*budget-math.exp(-2*h)*(s*budget-current)
    e=proposal-c
    ep=e+torch.stack([dot(g,anchor) for g in gradients]) if gradients else e
    # H=A/aa: whiten_H=sqrt(aa)*whiten_A, inverse_H=aa*inverse_A.
    white=[metric.allowed_white(g)*math.sqrt(aa) for g in gradients]
    a=h/(1+h)
    q=torch.stack([torch.stack([dot(x,y) for y in white]) for x in white]) if white else torch.empty((0,0),device=z.device,dtype=torch.float64)
    lam,kkt=joint_dual(a*q+h*torch.diag(epsilon),ep)
    correction=anchor.clone()
    for l,w in zip(lam,white):correction-=a*l*math.sqrt(aa)*metric.unwhiten(w)
    xi=h*epsilon*lam
    linear=e+torch.stack([dot(g,correction) for g in gradients]) if gradients else e
    terms=dict(h=h,s=s,s_next=s+h,e=e.tolist(),eprime=ep.tolist(),xi=xi.tolist(),xi_per_h=(xi/h).tolist(),
        gram=q.tolist(),linear_harm_after=linear.tolist(),kkt=kkt,
        constraint_residual=(linear-xi).tolist(),
        action_over_h=float(metric.action(correction)/aa/h),
        cumulative_anchor_action=float(metric.action(z+correction)/aa),
        step_action=float(metric.action(correction)/aa),
        progress_leakage=float(dot(metric.t,correction)))
    return finite(z+correction),correction,terms
