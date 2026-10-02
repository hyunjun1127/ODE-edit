"""Native FP32 nested mean, same-A ridge and exact diagnostic operators."""
import time
import torch
from project.run_scripts.jlz_writer_coupled.geometry import build_prior_from_npz
from .common import require

def mean_keys(raw, rows, pack):
    """Preserve native request/context order, independent of forward chunk order."""
    require(raw.dtype == torch.float32 and raw.ndim == 2, 'NATIVE_KEY_FP32')
    index = {r['global_row']: j for j,r in enumerate(rows)}
    require(len(index) == len(rows) == pack['n_requests']*pack['n_rw'], 'WHOLE_B_KEY_COVERAGE')
    result=[]
    for request in range(pack['n_requests']):
        groups=[]
        for start,stop in pack['context_group_slices']:
            cols=[index[request*(pack['n_rw']+1)+c] for c in range(start,stop)]
            # [context, feature] matches BLUE compute_ks.mean(0) exactly.
            groups.append(raw[:,cols].T.contiguous().mean(0))
        result.append(torch.stack(groups,0).mean(0))
    return torch.stack(result,0).T.contiguous()

class PriorProduct(torch.autograd.Function):
    """Same stored CPU FP64 A; bounded GPU memory, full P adjoint."""
    @staticmethod
    def forward(ctx,A,P):
        ctx.A=A;ctx.device=P.device
        return (A @ P.to('cpu')).to(P.device)
    @staticmethod
    def backward(ctx,gradient):
        return None,(ctx.A.T @ gradient.to('cpu')).to(ctx.device)

@torch.no_grad()
def prior(path,history,device,coefficient=15000.):
    start=time.monotonic()
    A=build_prior_from_npz(path,history,lambda_c=coefficient,device='cpu')
    gpu=A.to(device);L,info=torch.linalg.cholesky_ex(gpu)
    asym=0.
    for start_row in range(0,A.shape[0],256):
        asym=max(asym,float((A[start_row:start_row+256]-A[:,start_row:start_row+256].T).abs().max()))
    result=dict(A=A,device=device,L=None,LU=None,pivots=None,SPD=int(info)==0,asymmetry_max=asym)
    if int(info)==0: result['L']=L
    else: result['LU'],result['pivots']=torch.linalg.lu_factor(gpu)
    del gpu
    return result,dict(seconds=time.monotonic()-start,normalization='native_FP32_mom2/count_then_FP64',
        history='native_CPU_FP32_no_symmetrization',asymmetry_max=asym,
        initial_backend='cholesky' if int(info)==0 else 'same_A_LU',jitter=0)

def inverse_apply(prior,K):
    if prior['LU'] is not None:
        return torch.linalg.lu_solve(prior['LU'],prior['pivots'],K)
    return torch.cholesky_solve(K,prior['L'])

def ridge(K,prior):
    start=time.monotonic();K=K.double();B=K.shape[1]
    require(bool(torch.isfinite(K).all()),'NONFINITE_KEYS')
    Y=inverse_apply(prior,K);S=torch.eye(B,device=K.device,dtype=K.dtype)+K.T@Y
    P=torch.linalg.solve(S.T,Y.T).T
    AP=PriorProduct.apply(prior['A'],P)
    residual=(AP+K@(K.T@P)-K).norm()/K.norm().clamp_min(1e-30)
    initial=float(residual.detach());fallback=False
    if initial>1e-8 and prior['LU'] is None:
        # One same-A reference, without symmetrizing history or adding jitter.
        prior['LU'],prior['pivots']=torch.linalg.lu_factor(prior['A'].to(K.device))
        prior['L']=None;fallback=True
        Y=inverse_apply(prior,K);S=torch.eye(B,device=K.device,dtype=K.dtype)+K.T@Y
        P=torch.linalg.solve(S.T,Y.T).T;AP=PriorProduct.apply(prior['A'],P)
        residual=(AP+K@(K.T@P)-K).norm()/K.norm().clamp_min(1e-30)
    measured=float(residual.detach())
    require(bool(torch.isfinite(P).all()) and measured<=1e-8,'SAME_A_SOLVE_RESIDUAL:'+str(measured))
    M=P.T@K;R=M-torch.eye(B,device=K.device,dtype=K.dtype)
    G=P.T@AP;E=R@R.T
    return dict(P=P,K=K,M=M,G=G,E=E,metadata=dict(actual_B=B,relative_residual=measured,
        initial_residual=initial,same_A_LU_fallback=fallback,seconds=time.monotonic()-start,
        key_reduction='FP32_groupmean_stackmean_then_FP64',grad_K_enabled=K.requires_grad,
        grad_P_enabled=P.requires_grad,geometry_identity_max=float((G+E-(torch.eye(B,device=K.device)-M)).detach().abs().max())))

@torch.no_grad()
def exact(K,prior,D,entry):
    """No optimization, no pinv, no jitter; unsupported is an explicit result."""
    K=K.double();D=D.double();B=K.shape[1]
    result=dict(status='numerically_unqualified',writer_kind='exact',actual_B=B)
    try:
        if not prior['SPD']:
            return None,dict(result,reason='A_not_SPD')
        Y=inverse_apply(prior,K);X=K.T@Y
        # Qualification measures raw asymmetry; never alters X for the solve.
        eig=torch.linalg.eigvalsh(X);rank_tol=max(K.shape)*torch.finfo(torch.float64).eps*float(eig.max())
        rank=int((eig>rank_tol).sum());rcond=float(torch.linalg.svdvals(X)[-1]/torch.linalg.svdvals(X)[0])
        result.update(eigenvalues=eig.cpu().tolist(),rank=rank,rank_tol=rank_tol,rcond=rcond,
                      X_asymmetry_max=float((X-X.T).abs().max()))
        if rank<B:return None,dict(result,status='rank_unsupported',reason='full_column_rank_required')
        if rcond<1e-10:return None,dict(result,reason='rcond_below_1e-10')
        P=torch.linalg.solve(X,Y.T).T
        U=D@P.T;error=U@K-D
        rel=float(error.norm()/D.norm().clamp_min(1e-30))
        W=entry+U.to(entry.dtype)
        actual=torch.nn.functional.linear(K.T.float(),W)-torch.nn.functional.linear(K.T.float(),entry)
        rms=float((actual.T.double()-D).square().mean().sqrt());limit=1e-5+1e-4*float(D.square().mean().sqrt())
        result.update(FP64_relative=rel,FP64_absolute=float(error.norm()),D_zero=not bool(D.any()),FP32_RMS=rms,FP32_limit=limit)
        if not bool(torch.isfinite(W).all()) or rel>1e-8:return None,dict(result,reason='FP64_equality_or_nonfinite')
        if rms>limit:return None,dict(result,status='algebra_exact_but_FP32_unqualified',reason='actual_forward_equality')
        return dict(P=P,K=K,W=W),dict(result,status='QUALIFIED',reason=None)
    except torch.linalg.LinAlgError as exc:
        return None,dict(result,reason='exact_solve:'+str(exc))
