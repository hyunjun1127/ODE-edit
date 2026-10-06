"""Native FP32 sum/count and strict S<.02; diagnostics never substitute centered covariance."""
import ast
from types import SimpleNamespace
import numpy as np
import torch
from .common import require

def load_c0(path,width=6400):
    with np.load(path,allow_pickle=False) as x:
        require(set(x.files)=={'mom2.constructor','mom2.count','mom2.mom2','sample_size'},'NATIVE_SCHEMA')
        count=int(x['mom2.count']);require(count>0 and int(x['sample_size'])==100000,'SAMPLE_COUNT')
        raw=x['mom2.mom2'];require(raw.shape==(width,width) and raw.dtype==np.float32,'RAW_SHAPE_DTYPE')
    a=torch.from_numpy(raw);require(bool(torch.isfinite(a).all()),'RAW_NONFINITE')
    return a/count,count # Native FP32 division, not double then cast.

def native_project(cov,alpha_source):
    """Execute exact frozen get_project AST; get_cov adapter returns already normalized CPU C0."""
    tree=ast.parse(alpha_source)
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='get_project')
    captured={}
    def svd(a,**kw):
        result=torch.linalg.svd(a,**kw);captured['S']=result[1];return result
    proxy=SimpleNamespace(linalg=SimpleNamespace(svd=svd))
    env=dict(torch=proxy,get_cov=lambda *a,**k:cov)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'frozen_native_get_project','exec'),env)
    hp=SimpleNamespace(rewrite_module_tmp='transformer.h.{}.mlp.c_proj',mom2_dataset='wikipedia',mom2_n_samples=100000,mom2_dtype='float32',nullspace_threshold=.02)
    p=env['get_project'](None,None,13,hp)
    return p,captured['S']

def validate(cov,p,nullity,tol):
    require(cov.dtype==p.dtype==torch.float32 and cov.shape==p.shape,'C0_P_SHAPE_DTYPE')
    require(bool(torch.isfinite(cov).all()) and bool(torch.isfinite(p).all()),'NONFINITE')
    scale=max(1.,float(torch.linalg.vector_norm(cov)))
    symmetry=float(torch.max(torch.abs(cov-cov.T)))/scale
    psym=float(torch.max(torch.abs(p-p.T)))
    pnorm=float(torch.linalg.vector_norm(p));require(pnorm>0 or nullity==0,'P_RANK_MISMATCH')
    idem=float(torch.linalg.vector_norm(p@p-p))/max(1.,pnorm)
    trace=float(torch.trace(p))
    cp=cov@p
    # Full Frobenius diagnostic: RMS retained C0 response, NOT spectral norm or exact null.
    residual=float(torch.linalg.vector_norm(cp))/max(1.,nullity**.5)
    # Symmetrization/FP64 is diagnostic only; never feeds native C0/P or a writer.
    eig=torch.linalg.eigvalsh((cov.double()+cov.double().T)*.5)
    eigen_min=float(eig.min());eigen_max=float(eig.max())
    out=dict(c0_symmetry_relative=symmetry,P_symmetry_maxabs=psym,
        P_idempotence_full_Fro_relative=idem,P_trace=trace,recorded_nullity=nullity,
        C0P_retained_RMS_Fro=residual,C0P_metric='Fro(C0@P)/sqrt(recorded_nullity); not operator norm',
        C0_eigenvalue_min_FP64_symmetric_diagnostic=eigen_min,C0_eigenvalue_max_FP64_symmetric_diagnostic=eigen_max,
        C0_FP64_eigenvalues_lt_threshold=int((eig<.02).sum()),
        C0_threshold_distance_min_FP64=float(torch.min(torch.abs(eig-.02))),
        C0_PSD='full FP64 eigvalsh of diagnostic (C0+C0.T)/2; native C0/P remain unmodified',
        threshold=.02,strict_comparison='<',tolerances=tol)
    require(symmetry<=tol['c0_symmetry_relative'],'C0_SYMMETRY')
    require(psym<=tol['P_symmetry_maxabs'],'P_SYMMETRY')
    require(idem<=tol['P_idempotence_relative'],'P_IDEMPOTENCE')
    require(abs(trace-nullity)<=tol['P_trace_abs'],'P_TRACE')
    require(residual<=.02+tol['retained_RMS_slack'],'RETAINED_RESPONSE')
    require(eigen_min>=-tol['c0_psd_relative']*max(1.,eigen_max),'C0_PSD')
    return out
