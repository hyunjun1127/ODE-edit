"""Full P* geometry and the We-preserving static affine solution (FP64).

Only the excluded eigenvectors are stored when this is smaller. This represents
the complete >.5 eigenspace, not a rank-truncated approximation to that space.
"""
import torch

def finite(x):
    if not torch.isfinite(x).all():raise FloatingPointError('NONFINITE_GEOMETRY')
    return x

def dot(x,y):return torch.sum(x*y)

class Projector:
    def __init__(self, vectors, complement, dimension, spectrum=None):
        self.vectors=vectors; self.complement=complement; self.dimension=dimension
        self.spectrum=spectrum

    @classmethod
    def from_raw(cls, raw):
        raw=finite(raw.double()); sym=(raw+raw.T)*.5
        vals,vec=torch.linalg.eigh(sym); keep=vals>.5
        complement=int(keep.sum())>raw.shape[0]//2
        chosen=vec[:,~keep if complement else keep].contiguous()
        obj=cls(chosen,complement,raw.shape[0],vals)
        # Frobenius residual computed using the eigenbasis; no second dense P*.
        obj.receipt=dict(rank=int(keep.sum()),dimension=raw.shape[0],threshold=.5,
            eigen_min=float(vals.min()),eigen_max=float(vals.max()),
            below_max=float(vals[~keep].max()) if (~keep).any() else None,
            above_min=float(vals[keep].min()) if keep.any() else None,
            stored_excluded_space=complement,stored_columns=chosen.shape[1],
            raw_symmetry_relative=float((raw-raw.T).norm()/raw.norm()),
            symmetric_raw_difference_relative=float((vals-keep.double()).norm()/vals.norm()),
            stored_basis_orthogonality=float((chosen.T@chosen-torch.eye(chosen.shape[1],device=raw.device,dtype=torch.float64)).norm()))
        return obj

    def right(self,x):
        v=self.vectors.to(x.device)
        component=(x@v)@v.T
        return x-component if self.complement else component

    def to(self,device):
        self.vectors=self.vectors.to(device);return self

class Metric:
    """Exact P*(lambda I + P*UU'P*)^-1P* via a full-dimensional Cholesky.

    CPU/GPU chunking changes storage, not space or rank. No native history enters
    this routing metric. Native history remains in the native proposal only.
    """
    def __init__(self,projector,factors,ridge):
        self.p=projector;self.factors=factors;self.ridge=float(ridge)
        d=projector.dimension;dev=factors.device
        projected=self.p.right(factors.T).T
        matrix=projected@projected.T
        matrix.diagonal().add_(self.ridge)
        self.chol=torch.linalg.cholesky(finite((matrix+matrix.T)*.5))
        self.t_unit=None;self.t_white=None

    def whiten(self,x):
        return torch.linalg.solve_triangular(self.chol,self.p.right(x).T,upper=False).T

    def unwhiten(self,x):
        solved=torch.linalg.solve_triangular(self.chol.T,x.T,upper=True).T
        return self.p.right(solved)

    def inverse(self,x):return self.unwhiten(self.whiten(x))

    def action(self,x):return (x@self.factors).square().sum()+self.ridge*x.square().sum()

    def bind_progress(self,t):
        self.t=t
        # Normalize the observer before metric projection; exact zero is distinct.
        n=t.norm()
        self.t_unit=t/n if n>0 else torch.zeros_like(t)
        self.t_white=self.whiten(self.t_unit)
        self.q=self.t_white.square().sum()

    def allowed_white(self,g):
        x=self.whiten(g)
        if self.q>0:x=x-dot(x,self.t_white)/self.q*self.t_white
        return x

    def allowed_inverse(self,g):return self.unwhiten(self.allowed_white(g))

def static_solution(delta,k,r,context_factor,past_factor,base_factor,projector):
    """B is logical effective inventory; no W0 argument or recovery cross term."""
    b=k.shape[1]
    if b==0:return torch.zeros_like(delta),None,dict(status='B0_NOOP')
    ek=k/(b**.5);t=(r@k.T)/b
    u=torch.cat([ek,context_factor,past_factor,base_factor],dim=1)
    ridge=.001*float(u.square().sum())/k.shape[0]+1e-8
    metric=Metric(projector,u,ridge);metric.bind_progress(t)
    # CE, Cp, Cb and lambda_r; intentionally NOT gamma*Cresp in L.
    ll=ridge*delta-t
    for factor in (ek,past_factor,base_factor):ll=ll+(delta@factor)@factor.T
    zw=-metric.allowed_white(ll);z=metric.unwhiten(zw)
    eq=float(dot(t,z));zn=float(z.norm())
    receipt=dict(status='STATIC_SOLUTION',batch=b,ridge=ridge,
        progress_reference=float(dot(t,delta)),algebraic_equality_residual=eq,
        algebraic_equality_relative=abs(eq)/(float(t.norm())*zn) if zn and t.norm()>0 else 0.,
        span_relative=float((z-projector.right(z)).norm())/zn if zn else 0.,
        stationarity_allowed_norm=float(metric.allowed_white(z*ridge+(z@u)@u.T+ll).norm()),
        zero_allowed_progress=bool(metric.q==0),w0_controller_access=0)
    finite(z)
    return z,metric,receipt
