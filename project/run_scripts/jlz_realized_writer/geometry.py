"""FP64 metric/SVD writer; rank deficiency is distinct from numerical failure."""
import time
import torch
from .common import require

def prepare_metric(raw):
    require(raw.dtype==torch.float64 and raw.ndim==2 and raw.shape[0]==raw.shape[1],'METRIC_SHAPE_DTYPE')
    require(bool(torch.isfinite(raw).all()),'METRIC_NONFINITE')
    skew=float((raw-raw.T).norm());A=(raw+raw.T)*.5
    L=torch.linalg.cholesky(A)
    return A,L,dict(skew_frobenius=skew,skew_relative=skew/float(A.norm()),SPD=True,jitter=0)

def solve_realized(A,L,K,T,weights,kind):
    started=time.monotonic()
    require(kind in ('ridge','equality'),'SOLVER_KIND')
    require(K.dtype==T.dtype==weights.dtype==torch.float64,'SOLVER_PRECISION')
    require(K.ndim==T.ndim==2 and K.shape[1]==T.shape[1] and K.shape[0]==A.shape[0],'SOLVER_SHAPE')
    require(weights.shape==(K.shape[1],) and bool((weights>0).all()) and abs(float(weights.sum())-1)<1e-12,'WEIGHTS')
    require(all(bool(torch.isfinite(v).all()) for v in (K,T,weights)),'SOLVER_NONFINITE_INPUT')
    root=weights.sqrt();Z=T*root;zero=not bool(torch.count_nonzero(T))
    rank=None;singular=[];cutoff=None;discarded=None;condition=None
    if kind=='ridge':
        # Native soft constraint has unweighted KK^T, not sum-one KK^T/B.
        system=A+K@K.T;P=torch.linalg.solve(system,K)
        error=(system@P-K).norm();den=K.norm()
        residual=float(error/den) if bool(den>0) else float(error)
        require(bool(torch.isfinite(P).all()) and residual<=1e-8,'RIDGE_NUMERICAL_RESIDUAL')
        U=T@P.T;status='NATIVE_RIDGE';projected=None
    else:
        X=torch.linalg.solve_triangular(L,K*root,upper=False)
        # QR-SVD is a direct factorization of X, never normal equations.
        Q,R=torch.linalg.qr(X,mode='reduced')
        left,s,right=torch.linalg.svd(R,full_matrices=False)
        cutoff=float(torch.finfo(torch.float64).eps*max(X.shape)*s[0])
        keep=s>cutoff;rank=int(keep.sum());singular=s.tolist()
        discarded=int(((s>0)&~keep).sum())
        if rank:
            basis=Q@left[:,keep];vr=right[keep]
            coefficients=(Z@vr.T)/s[keep]
            white=coefficients@basis.T
            U=torch.linalg.solve_triangular(L.T,white.T,upper=True).T
            projected=(Z@vr.T)@vr
            condition=float(s[keep][0]/s[keep][-1])
        else:
            U=torch.zeros((T.shape[0],K.shape[0]),dtype=T.dtype,device=T.device)
            projected=torch.zeros_like(Z)
        residual=float((U@K*root-projected).norm())
        status='COMPATIBLE_EXACT_WITHIN_TOLERANCE' if float((Z-projected).norm())<=1e-10+1e-8*float(Z.norm()) else 'RANGE_PROJECTED_WEIGHTED_LEAST_SQUARES'
    require(bool(torch.isfinite(U).all()),'UPDATE_NONFINITE')
    tolerance=1e-10+1e-8*float(Z.norm())
    verified=residual<=tolerance if kind=='equality' else residual<=1e-8
    diag=dict(kind=kind,status=status if verified else 'NUMERICAL_REALIZATION_FAILURE',
        nominal_geometry_status=status,numerical_projection_verified=verified,
        shape=dict(n_in=K.shape[0],n_out=T.shape[0],constraints=K.shape[1]),
        rank=rank,singular_values=singular,cutoff=cutoff,cutoff_rule='eps64*max(X.shape)*sigma_max; strict>',
        positive_discarded_count=discarded,retained_condition=condition,
        projected_target_residual=residual,numerical_tolerance=tolerance if kind=='equality' else 1e-8,
        weighted_target_norm=float(Z.norm()),weighted_target_mismatch=float(((U@K-T)*root).norm()),
        incompatible_target_norm=None if projected is None else float((Z-projected).norm()),
        incompatible_by_column=None if projected is None else (Z-projected).norm(dim=0).tolist(),
        zero_target=zero,zero_target_fastpath=False,ideal_Q=float((U@L).square().sum()),
        ideal_update_norm=float(U.norm()),seconds=time.monotonic()-started)
    # Caller persists diagnostics before raising on a failed numeric identity.
    return U,diag

def action_parity(candidate,reference):
    error=(candidate.double()-reference.double()).abs();limit=2e-5+2e-4*reference.double().abs()
    return dict(pass_=bool((error<=limit).all()),max_absolute=float(error.max()),
        max_excess=float((error-limit).max()),failed_elements=int((error>limit).sum()),
        elements=error.numel(),atol=2e-5,rtol=2e-4,reduction='elementwise')

def targets(branch,D,z,h,owners):
    require(branch in ('RT','RD','MT','MD','CD'),'BRANCH')
    return z-h if branch in ('RT','MT') else D[:,owners] if branch=='CD' else D

def owner_weights(owners,B):
    counts=torch.bincount(owners,minlength=B)
    require(bool((counts>0).all()),'OWNER_COVERAGE')
    return 1./(B*counts[owners].double())
