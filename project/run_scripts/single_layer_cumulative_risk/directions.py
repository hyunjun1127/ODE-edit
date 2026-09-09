"""B/C coefficient-space risk probes. CPU math; no execution or stage admission."""
import hashlib
import math
import torch

def group_indices(case_ids):
    assert len(case_ids)==100 and len(set(case_ids))==100
    order=sorted(range(100),key=lambda i:(hashlib.sha256(('CUMRISK-CURRENT-GROUP|'+str(case_ids[i])).encode()).hexdigest(),str(case_ids[i])))
    return [order[i:i+10] for i in range(0,100,10)]

def normalized_rows(gradients):
    """FP32 arithmetic-resolution policy, not a performance/eligibility gate.

    A row below eps32 times the largest row norm cannot be resolved relative to
    that collection at working precision; it is retained as an explicit zero
    row, not amplified. No requests/groups are removed from the denominator.
    """
    flat=gradients.flatten(1)
    norms=flat.double().norm(dim=1)
    if not torch.isfinite(norms).all():raise FloatingPointError('NONFINITE_GROUP_GRADIENT')
    threshold=torch.finfo(gradients.dtype).eps*norms.max()
    active=norms>threshold
    rows=torch.zeros_like(flat)
    rows[active]=flat[active]/norms[active,None].to(flat.dtype)/math.sqrt(len(norms))
    return rows,dict(raw_norms=norms.cpu().tolist(),zero_rows=int((~active).sum()),
                     working_resolution_threshold=float(threshold),request_exclusions=0,group_count=len(norms))

def soft_filter(g,j,gamma=.1):
    shape=g.shape;flat=g.flatten()
    # Only the 10x10 system is FP64; coefficient/model storage is FP32.
    gram=(j@j.T).double();rhs=(j@flat).double()
    small=torch.linalg.solve(gram+gamma*torch.eye(len(j),dtype=torch.float64,device=j.device),rhs)
    result=(flat-j.T@small.to(j.dtype)).reshape(shape)
    return result,dict(gamma=gamma,raw_norm=float(g.double().norm()),filtered_norm=float(result.double().norm()),
             leakage_before=(j@flat).cpu().tolist(),leakage_after=(j@result.flatten()).cpu().tolist(),
             small_solve_residual=float((gram@small+gamma*small-rhs).norm()))

def physical_unit(d,u):
    physical=d@u.T;norm=physical.double().norm()
    if not torch.isfinite(norm):raise FloatingPointError('NONFINITE_DIRECTION')
    # No epsilon floor or forced amplification of unresolved/zero values.
    coefficient_norm=d.double().norm()
    resolution=torch.finfo(d.dtype).eps*coefficient_norm
    if norm==0 or norm<=resolution:
        return torch.zeros_like(d),dict(status='UNDEFINED_ZERO_DIRECTION',physical_norm=float(norm),applied_norm=0.)
    result=d/norm.to(d.dtype)
    return result,dict(status='FINITE_DIRECTION',physical_norm=float(norm),applied_norm=float((result@u.T).double().norm()))

def frobenius_finite_change(w,reference,delta):
    linear=((w-reference).double()*delta.double()).sum()
    quadratic=.5*delta.double().square().sum()
    actual=.5*((w+delta-reference).double().square().sum()-(w-reference).double().square().sum())
    return dict(first_order=float(linear),second_order=float(quadratic),actual=float(actual),
                rounding_residual=float(actual-linear-quadratic))

def operator_gradient(d,seeds=(20260910,20260911),steps=30):
    runs=[]
    for seed in seeds:
        rng=torch.Generator(device=d.device);rng.manual_seed(seed)
        v=torch.randn(d.shape[1],device=d.device,dtype=d.dtype,generator=rng);v/=v.norm()
        for _ in range(steps):
            y=d.T@(d@v)
            if y.norm()==0:break
            v=y/y.norm()
        value=(d@v).square().sum();residual=(d.T@(d@v)-value*v).norm()
        runs.append((float(value),v,dict(seed=seed,rayleigh=float(value),residual=float(residual))))
    chosen=max(runs,key=lambda x:x[0]);v=chosen[1]
    return (d@v)[:,None]*v[None,:],dict(iterations=steps,runs=[x[2] for x in runs],upper_bound=False)
