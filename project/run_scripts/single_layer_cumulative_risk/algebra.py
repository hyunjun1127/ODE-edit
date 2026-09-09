"""Physical orthobasis SGD and native action; no success-based constraints."""
import torch

def orthobasis(matrix):
    u,s,_=torch.linalg.svd(matrix,full_matrices=False)
    tol=max(matrix.shape)*torch.finfo(matrix.dtype).eps*s.max()
    keep=s>tol
    return u[:,keep].contiguous(),dict(rank=int(keep.sum()),tolerance=float(tol),singular_values=s.detach().cpu().tolist())

def projector_basis(p):
    # Full stored range, not a random or fixed-rank approximation.
    vals,q=torch.linalg.eigh((p+p.T)*.5)
    tol=p.shape[0]*torch.finfo(p.dtype).eps*vals.abs().max()
    keep=vals>tol
    basis=q[:,keep].contiguous()
    residual=(basis@basis.T-p).norm()/p.norm()
    return basis,dict(rank=int(keep.sum()),tolerance=float(tol),projector_relative_residual=float(residual),eigen_min=float(vals.min()),eigen_max=float(vals.max()))

def action(delta,m,l2=1.):return ((delta@m)*delta).sum()+l2*delta.square().sum()

def reduced_metric(u,m,l2=1.):return u.T@m@u+l2*torch.eye(u.shape[1],device=u.device,dtype=u.dtype)

def calibrated_eta(alpha,native_norm,g):
    norm=g.norm()
    if not torch.isfinite(norm):raise FloatingPointError('NONFINITE_INITIAL_GRADIENT')
    if norm==0:return 0.,'EXACT_ZERO_GRADIENT_FINITE_NO_ACTION'
    return float(alpha*native_norm/norm),'NATIVE_ASSISTED_ONCE'

def momentum_update(a,g,velocity,eta):
    v=g.clone() if velocity is None else .9*velocity+g
    return a-eta*v,v
