"""One Current equality plus simultaneous Base/Past elastic constraints.

Small dual algebra FP64, exact nonnegative dual domain, no posthoc clipping.
Large K inverses are independent matrix-free PCG solves. A balance, when used,
is incorporated by caller in K and u, not approximated here.
"""
from dataclasses import dataclass
import itertools
import torch
from .linear_solve import WeightTree, Operator, pcg, dot, norm, add, scale, zeros


def joint_dual(matrix: torch.Tensor, rhs: torch.Tensor):
    matrix=matrix.detach().double();rhs=rhs.detach().double()
    if matrix.shape!=(2,2) or rhs.shape!=(2,):raise ValueError('TWO_CHANNELS_REQUIRED')
    if not bool(torch.isfinite(matrix).all() and torch.isfinite(rhs).all()):
        raise FloatingPointError('NONFINITE_DUAL_INPUT')
    if not bool((torch.linalg.eigvalsh(matrix)>0).all()):raise FloatingPointError('DUAL_NOT_SPD')
    candidates=[]
    for active in itertools.product((False,True),repeat=2):
        mask=torch.tensor(active,device=rhs.device,dtype=torch.bool);nu=torch.zeros_like(rhs)
        if mask.any():nu[mask]=torch.linalg.solve(matrix[mask][:,mask],rhs[mask])
        if not bool((nu>=0).all()):continue
        grad=matrix@nu-rhs
        tol=128*torch.finfo(torch.float64).eps*torch.maximum(
            matrix.abs()@nu.abs()+rhs.abs(),torch.full_like(rhs,torch.finfo(torch.float64).tiny))
        if bool((grad[~mask]>=-tol[~mask]).all() and (grad[mask].abs()<=tol[mask]).all()):
            candidates.append((float(.5*nu@(matrix@nu)-rhs@nu),active,nu,grad,tol))
    if not candidates:raise FloatingPointError('JOINT_KKT_NO_FEASIBLE_ACTIVE_SET')
    _,active,nu,grad,tol=min(candidates,key=lambda x:(x[0],x[1]))
    return nu,dict(active=list(active),dual=nu.tolist(),dual_gradient=grad.tolist(),
        gradient_tolerance=tol.tolist(),complementarity=float((nu*grad).abs().max()),
        dual_nonnegativity='EXACT_DOMAIN_NO_CLIP',matrix_condition=float(torch.linalg.cond(matrix)))


@dataclass
class ElasticResult:
    correction: WeightTree
    equality_correction: WeightTree
    dual: torch.Tensor
    slack: torch.Tensor
    diagnostics: dict


def solve_constrained(operator: Operator, u: WeightTree, a: WeightTree, t_e: float,
                      *, precondition: Operator | None = None,
                      gradients: tuple[WeightTree,WeightTree] | None = None,
                      risks=None, budgets=None, h=1.0, rho=(2.0,1.0),
                      rtol=1e-4, maxiter=20) -> ElasticResult:
    if h<=0 or min(rho)<=0:raise ValueError('INVALID_ELASTIC_SCALING')
    solved={}
    def inverse(label, rhs):
        result=pcg(operator,rhs,precondition,rtol=rtol,maxiter=maxiter)
        solved[label]=result.receipt();return result.solution
    iu=inverse('u',u);cu=scale(iu,-1)
    zero_a=dot(a,a)==0
    ia=zeros(a) if zero_a else inverse('a',a)
    aa=0.0 if zero_a else dot(a,ia)
    if not zero_a and aa<=0:raise FloatingPointError('NONPOSITIVE_EQUALITY_INVERSE_NORM')
    ceq=cu if zero_a else add(cu,ia,(float(t_e)-dot(a,cu))/aa)
    nu=torch.zeros(2,dtype=torch.float64);slack=torch.zeros_like(nu);dual_diag=None
    gram_raw=None;gram_asym=0.0;v=None;pkgs=[]
    if gradients is not None:
        if len(gradients)!=2:raise ValueError('TWO_GRADIENT_CHANNELS_REQUIRED')
        risks=torch.as_tensor(risks,dtype=torch.float64);budgets=torch.as_tensor(budgets,dtype=torch.float64)
        if risks.shape!=(2,) or budgets.shape!=(2,):raise ValueError('TWO_RISK_VALUES_REQUIRED')
        for j,g in enumerate(gradients):
            ig=inverse(('g_base','g_past')[j],g)
            pkgs.append(ig if zero_a else add(ig,ia,-dot(a,ig)/aa))
        gram_raw=torch.tensor([[dot(g,p) for p in pkgs] for g in gradients],dtype=torch.float64)
        gram_asym=float(torch.linalg.norm(gram_raw-gram_raw.T))
        # Scalar quadratic uses its symmetric part; preserve raw asymmetry from
        # independently approximated PCG actions in diagnostics, never hide it.
        matrix=.5*(gram_raw+gram_raw.T)+h*torch.diag(1/torch.tensor(rho,dtype=torch.float64))
        v=risks+torch.tensor([dot(g,ceq) for g in gradients],dtype=torch.float64)-budgets
        nu,dual_diag=joint_dual(matrix,v)
        slack=h*nu/torch.tensor(rho,dtype=torch.float64)
    c=ceq
    for p,n in zip(pkgs,nu):c=add(c,p,-float(n))
    # Actual K action, not the recursive PCG residual, gives stationarity.
    station=add(operator(c),u)
    if gradients is not None:
        for g,n in zip(gradients,nu):station=add(station,g,float(n))
    equality_dual=0.0 if zero_a else -dot(a,station)/dot(a,a)
    station=add(station,a,equality_dual)
    eq_res=dot(a,c)-float(t_e)
    primal=[] if gradients is None else (risks+torch.tensor([dot(g,c) for g in gradients],dtype=torch.float64)-budgets-slack).tolist()
    approximate=any(x['status']=='APPROXIMATE_PCG_NONCONVERGENCE' for x in solved.values())
    status='APPROXIMATE_PCG_NONCONVERGENCE' if approximate else 'SOLVED_LINEARIZED_QUADRATIC'
    if zero_a and t_e!=0:status='CURRENT_DEFICIT_FIRST_ORDER_UNRECOVERABLE'
    diag=dict(status=status,pcg=solved,h=h,rho=list(rho),zero_current_gradient=zero_a,
         t_e=float(t_e),a_kinv_a=aa,equality_whitened_row_scale=aa**.5,
         equality_residual=eq_res,equality_dual=equality_dual,stationarity_norm=norm(station),
         stationarity_reference_norm=norm(u),primal_residual=primal,
         slack=slack.tolist(),dual=nu.tolist(),dual_diagnostics=dual_diag,
         slack_stationarity=(torch.tensor(rho,dtype=torch.float64)*slack/h-nu).tolist(),
         complementarity=[] if gradients is None else (nu*torch.tensor(primal,dtype=torch.float64)).tolist(),
         risk_gram_raw=None if gram_raw is None else gram_raw.tolist(),risk_gram_antisymmetry_norm=gram_asym,
         explicit_inverse=0,rank_truncation=0,functional_cross_block_dropped=0,
         extra_stationarity_matvecs=1,independent_rhs_count=len(solved))
    return ElasticResult(c,ceq,nu,slack,diag)
