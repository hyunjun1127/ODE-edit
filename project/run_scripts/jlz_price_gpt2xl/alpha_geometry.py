"""Frozen .02 projector, nonsymmetric Alpha LU + thin Woodbury; no reverse."""
import functools
import math
import time
import numpy as np
import torch
from project.run_scripts.jlz_native_writer_aware.common import require

@functools.lru_cache(maxsize=1)
def projector(path):
    value=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    require(isinstance(value,torch.Tensor) and value.dtype==torch.float32 and value.ndim==3
        and value.shape[0]==5 and not value.requires_grad,'ALPHAEDIT_PROJECTOR_SCHEMA')
    return value

def product(matrix,thin):
    """FP64 product from readonly CPU FP32 matrix with bounded row casting."""
    x=thin.detach().cpu().double();vector=x.ndim==1
    if vector:x=x[:,None]
    out=torch.empty((matrix.shape[0],x.shape[1]),dtype=torch.float64)
    for i in range(0,matrix.shape[0],128):out[i:i+128]=matrix[i:i+128].double()@x
    require(bool(torch.isfinite(out).all()),'ALPHAEDIT_PRODUCT_NONFINITE')
    return out[:,0] if vector else out

@torch.no_grad()
def prior(path,history,device,binding,layer):
    started=time.monotonic();N=projector(binding['path'])[binding['physical_layers'].index(layer)]
    require(tuple(N.shape)==tuple(history.shape) and history.dtype==torch.float32
        and history.device.type=='cpu','ALPHAEDIT_HISTORY_PROJECTOR_DIMENSION')
    with np.load(path,allow_pickle=False) as archive:
        raw=archive['mom2.mom2'];count=float(archive['mom2.count'].item())
    require(raw.dtype==np.float32 and raw.shape==tuple(N.shape) and math.isfinite(count)
        and count>0 and count.is_integer(),'ALPHAEDIT_C0_IDENTITY')
    C0=torch.from_numpy(raw);C0.div_(int(count))
    require(bool(torch.isfinite(C0).all()),'ALPHAEDIT_C0_NONFINITE')
    Ng=N.to(device=device,dtype=torch.float64);Hg=history.to(device=device,dtype=torch.float64)
    A0=Ng@Hg;A0.diagonal().add_(10.)
    require(bool(torch.isfinite(A0).all()),'ALPHAEDIT_A0_NONFINITE')
    asym=0.
    for i in range(0,A0.shape[0],128):
        asym=max(asym,float((A0[i:i+128]-A0[:,i:i+128].T).abs().max()))
    del Ng,Hg
    LU,pivots,info=torch.linalg.lu_factor_ex(A0)
    require(int(info)==0 and bool(torch.isfinite(LU).all()),'ALPHAEDIT_A0_LU_FAILURE')
    del A0
    metadata=dict(writer='alphaedit',lambda_alpha=10.,projector_sha256=binding['sha256'],
        physical_layer=layer,projector_index=binding['physical_layers'].index(layer),
        projector_cutoff=.02,backend='nonsymmetric_A0_LU_once_per_layer_batch',
        asymmetry_max=asym,symmetrized=False,jitter=0,C0_diagnostic_scale=1.,
        C0_normalization='native_FP32_mom2/count_then_FP64',C0_count=int(count),
        seconds=time.monotonic()-started,history='CPU_FP32_rewrite_only')
    return dict(N=N,H=history,C0=C0,LU=LU,pivots=pivots,device=device,
        asymmetry_max=asym,metadata=metadata),metadata

@torch.no_grad()
def ridge(K,prior):
    started=time.monotonic();K=K.double();B=K.shape[1]
    require(bool(torch.isfinite(K).all()),'ALPHAEDIT_KEY_NONFINITE')
    NK=product(prior['N'],K).to(K.device)
    Y=torch.linalg.lu_solve(prior['LU'],prior['pivots'],NK)
    S=torch.eye(B,dtype=K.dtype,device=K.device)+K.T@Y
    Q=torch.linalg.solve(S.T,Y.T).T
    # Actual original operator, not a symmetric surrogate or energy metric.
    qcpu=Q.cpu();kcpu=K.cpu();nkcpu=NK.cpu()
    hq=product(prior['H'],qcpu)
    aq=10.*qcpu+product(prior['N'],hq+kcpu@(kcpu.T@qcpu))
    absolute=float((aq-nkcpu).norm());rhs=float(nkcpu.norm())
    relative=None if rhs==0 else absolute/rhs
    require(bool(torch.isfinite(Q).all()) and (absolute<=1e-8 if rhs==0 else relative<=1e-8),
        'ALPHAEDIT_OPERATOR_RESIDUAL:'+str(relative))
    knorm=kcpu.norm(dim=0);nknorm=nkcpu.norm(dim=0)
    M=Q.T@K
    metadata=dict(prior['metadata'],actual_B=B,relative_residual=relative,residual_absolute=absolute,
        residual_rhs_NK_norm=rhs,residual_limit=1e-8,operator='10Q+N(HQ+K(K.TQ))=NK',
        factor_reused=True,thin_solve_shape=[B,B],M_diagonal=M.diagonal().cpu().tolist(),
        projected_key_norm=nknorm.tolist(),key_norm=knorm.tolist(),
        projected_key_normratio=[float(nknorm[i]/knorm[i]) if knorm[i]>0 else None for i in range(B)],
        key_reduction='FP32_groupmean_stackmean_then_FP64',grad_K_enabled=False,grad_P_enabled=False,
        seconds=time.monotonic()-started)
    return dict(P=Q,K=K,M=M,metadata=metadata)
