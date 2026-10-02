#!/usr/bin/env python3
"""CPU-only algebra audit for v9 writer discussion, not a writer selection.

No production imports, model loading, CUDA work, optimizer changes, or jobs.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import torch

torch.set_num_threads(1)
torch.manual_seed(20261003)
F64=torch.float64
HERE=Path(__file__).resolve().parent
CHECKS=[]
DETAILS={}


class SelectedRoot(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        out=x.clamp_min(0).sqrt();ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx,upstream):
        out,=ctx.saved_tensors
        scale=torch.zeros_like(out);positive=out>0
        scale[positive]=.5/out[positive]
        return upstream*scale


def scalar(x):
    return float(x.detach()) if isinstance(x,torch.Tensor) else float(x)


def check(name,condition,**details):
    passed=bool(condition)
    CHECKS.append(dict(name=name,passed=passed,**details))
    if not passed:raise AssertionError(f'{name}: {details}')


def close(name,got,want,atol=1e-10,rtol=1e-9):
    got=torch.as_tensor(got).detach().double()
    want=torch.as_tensor(want).detach().double()
    error=(got-want).abs()
    limit=atol+rtol*want.abs()
    check(name,(error<=limit).all(),max_absolute=scalar(error.max()),
          max_excess=scalar((error-limit).max()),atol=atol,rtol=rtol)


def spd(n):
    z=torch.randn(n,n,dtype=F64)
    return z@z.T/n+1.5*torch.eye(n,dtype=F64)


def gram(A,K):
    AK=torch.linalg.solve(A,K)
    return K.T@AK,AK


def ridge(A,K,D):
    X,AK=gram(A,K)
    P=torch.linalg.solve(A+K@K.T,K)
    U=D@P.T
    I=torch.eye(K.shape[1],dtype=F64)
    M=X@torch.linalg.solve(I+X,I)
    G=P.T@A@P
    E=(P.T@K-I)@(P.T@K-I).T
    return dict(X=X,P=P,U=U,M=M,G=G,E=E)


def exact_writer(A,K,D):
    X,AK=gram(A,K)
    # P= A^-1 K X^-1; transposed solve avoids an explicit inverse.
    P=torch.linalg.solve(X.T,AK.T).T
    U=D@P.T
    energy=torch.trace(D@torch.linalg.solve(X,D.T))
    return dict(X=X,P=P,U=U,energy=energy)


def exact_P_vjp(A,K,P,Q):
    X,AK=gram(A,K)
    C=torch.linalg.solve(X,torch.eye(X.shape[0],dtype=F64))
    barX=-C.T@AK.T@Q@C.T
    return torch.linalg.solve(A.T,Q@C.T)+AK@(barX+barX.T)


def ridge_and_exact_tests():
    observations=[]
    for n,B,out in ((3,1,2),(5,2,3),(6,3,2)):
        A=spd(n)
        K=torch.randn(n,B,dtype=F64)
        D=torch.randn(out,B,dtype=F64)
        r=ridge(A,K,D);e=exact_writer(A,K,D)
        tag=f'n{n}_B{B}_out{out}'
        close(f'ridge_UK_equals_DM_{tag}',r['U']@K,D@r['M'])
        close(f'ridge_PTK_equals_M_{tag}',r['P'].T@K,r['M'])
        close(f'ridge_GE_identity_{tag}',r['G']+r['E'],torch.eye(B,dtype=F64)-r['M'])
        eig=torch.linalg.eigvalsh(r['M'])
        check(f'full_rank_ridge_eigenvalues_strictly_between_zero_one_{tag}',((eig>0)&(eig<1)).all(),eigenvalues=eig.tolist())
        close(f'exact_UK_equals_D_{tag}',e['U']@K,D)
        close(f'exact_energy_identity_{tag}',torch.trace(e['U']@A@e['U'].T),e['energy'])
        # Nullspace perturbations cover feasible competitors with larger energy.
        N=torch.eye(n,dtype=F64)-K@torch.linalg.pinv(K)
        Z=torch.randn(out,n,dtype=F64)@N
        close(f'feasible_nullspace_perturbation_{tag}',Z@K,torch.zeros_like(D))
        competitor=e['U']+Z
        close(f'competitor_same_targets_{tag}',competitor@K,D)
        increase=torch.trace(competitor@A@competitor.T)-e['energy']
        close(f'exact_minimum_energy_gap_{tag}',increase,torch.trace(Z@A@Z.T))
        check(f'exact_minimum_energy_competitor_positive_gap_{tag}',scalar(increase)>1e-8)
        observations.append(dict(n=n,B=B,out=out,condition_X=scalar(torch.linalg.cond(e['X'])),
            ridge_realization_eigenvalues=eig.tolist(),ridge_target_error=scalar((r['U']@K-D).norm()),
            exact_target_error=scalar((e['U']@K-D).norm()),
            ridge_energy=scalar(torch.trace(r['U']@A@r['U'].T)),exact_energy=scalar(e['energy'])))
        # Closed-form exact-P adjoint remains fully dependent on current K.
        Kvar=K.detach().requires_grad_(True)
        ev=exact_writer(A,Kvar,D)
        Q=torch.randn_like(ev['P'])
        automatic=torch.autograd.grad((ev['P']*Q).sum(),Kvar,retain_graph=True)[0]
        close(f'exact_P_VJP_{tag}',exact_P_vjp(A,Kvar,ev['P'],Q),automatic)
        energy_grad=torch.autograd.grad(ev['energy'],Kvar)[0]
        C=torch.linalg.solve(ev['X'],torch.eye(B,dtype=F64))
        close(f'exact_energy_K_gradient_{tag}',-2*ev['P']@(D.T@D)@C,energy_grad)
    DETAILS['full_rank']=observations


def rank_deficient_tests():
    A=torch.eye(2,dtype=F64)
    K=torch.tensor([[1.,1.,0.],[0.,0.,1.]],dtype=F64)
    X,AK=gram(A,K)
    Xi=torch.linalg.pinv(X,rtol=1e-12,atol=0.,hermitian=True)
    projector=Xi@X
    compatible=torch.tensor([[2.,2.,-1.],[.5,.5,3.]],dtype=F64)
    conflicting=torch.tensor([[2.,-2.,-1.],[.5,1.5,3.]],dtype=F64)
    close('rankdef_projector_idempotent',projector@projector,projector)
    close('rankdef_compatible_D_equals_DXpinvX',compatible,compatible@projector)
    U=compatible@Xi@AK.T
    close('rankdef_compatible_pseudoinverse_exact_realization',U@K,compatible)
    close('rankdef_compatible_minimum_energy',torch.trace(U@A@U.T),torch.trace(compatible@Xi@compatible.T))
    check('rankdef_conflicting_D_is_incompatible',not torch.allclose(conflicting,conflicting@projector))
    Ub=conflicting@Xi@AK.T
    realized=Ub@K
    close('rankdef_pseudoinverse_realizes_projected_targets',realized,conflicting@projector)
    check('rankdef_pseudoinverse_cannot_fix_conflicting_duplicate_keys',not torch.allclose(realized,conflicting),
          target_error=scalar((realized-conflicting).norm()))
    arbitrary=torch.randn(4,2,dtype=F64)
    close('duplicate_keys_force_identical_outputs_for_any_writer',(arbitrary@K)[:,0],(arbitrary@K)[:,1])
    DETAILS['rank_deficient']=dict(K=K.tolist(),X=X.tolist(),rank=int(torch.linalg.matrix_rank(X)),
        pseudoinverse_rtol=1e-12,projector=projector.tolist(),compatible_D=compatible.tolist(),
        conflicting_D=conflicting.tolist(),conflicting_realization=realized.tolist(),
        compatibility='D = D X_dagger X',
        interpretation='Failure is infeasibility of exact target constraints, not a numerical defect of pinv.')


def context_mean_tests():
    # Native groups: canonical singleton and five augmented contexts.
    groups=[torch.tensor([[[1.,0.]]],dtype=torch.float32),
            torch.tensor([[[3.,2.],[-1.,-2.],[1.,0.],[1.,0.],[1.,0.]]],dtype=torch.float32)]
    K=torch.stack([g.mean(1) for g in groups]).mean(0).T.double()
    allK=torch.cat(groups,1)[0].T.double()
    A=torch.eye(2,dtype=F64);D=torch.ones(1,1,dtype=F64)
    e=exact_writer(A,K,D)
    realized=e['U']@allK
    close('mean_exact_native_group_mean_constraint',e['U']@K,D)
    realized_group_mean=.5*realized[:,0:1]+.5*realized[:,1:].mean(1,keepdim=True)
    close('mean_exact_group_weighted_output_mean',realized_group_mean,D)
    error=realized-D.expand_as(realized)
    check('mean_exact_does_not_imply_per_context_exact',scalar(error.abs().max())>1.,per_context_error=error.tolist())
    # A stronger context-exact solution happens to exist in this example;
    # mean-only minimum energy does not select it.
    context_exact=torch.tensor([[1.,-1.]],dtype=F64)
    close('context_exact_alternative_exists',context_exact@allK,D.expand_as(realized))
    check('context_exact_alternative_costs_more_than_mean_exact',
          scalar(torch.trace(context_exact@A@context_exact.T))>scalar(e['energy']))
    DETAILS['mean_vs_context']=dict(native_group_sizes=[1,5],K=K.tolist(),mean_exact_U=e['U'].tolist(),
        context_outputs=realized.tolist(),target_per_context=D.expand_as(realized).tolist(),
        context_errors=error.tolist(),native_weighted_context_squared_error=scalar(.5*error[:,0].square()+.5*error[:,1:].square().mean(1)),
        mean_exact_energy=scalar(e['energy']),context_exact_U=context_exact.tolist(),context_exact_energy=2.)


def K_from_M(M):
    I=torch.eye(M.shape[0],dtype=F64)
    X=torch.linalg.solve((I-M).T,M.T).T
    eig,Q=torch.linalg.eigh(X)
    check(f'counterexample_X_positive_{len(CHECKS)}',(eig>0).all())
    K=Q@torch.diag(eig.sqrt())@Q.T
    return X,K


def identical_ge_counterexamples():
    records=[]
    for theta in (math.pi/12,math.pi/4):
        R=torch.tensor([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]],dtype=F64)
        M=R@torch.diag(torch.tensor([.2,.8],dtype=F64))@R.T
        X,K=K_from_M(M)
        D=torch.eye(2,dtype=F64)
        r=ridge(torch.eye(2,dtype=F64),K,D)
        close(f'fullrank_D_counterexample_M_theta{theta}',r['M'],M)
        g2=torch.trace(D@r['G']@D.T);e2=torch.trace(D@r['E']@D.T)
        close(f'fullrank_D_counterexample_g2_theta{theta}',g2,torch.tensor(.32,dtype=F64))
        close(f'fullrank_D_counterexample_e2_theta{theta}',e2,torch.tensor(.68,dtype=F64))
        check(f'fullrank_D_counterexample_has_cross_coupling_theta{theta}',abs(scalar(M[0,1]))>.1)
        records.append(dict(theta=theta,D=D.tolist(),M=M.tolist(),diag_M=M.diag().tolist(),X=X.tolist(),
            g_squared=scalar(g2),e_squared=scalar(e2),g_with_B2_sigma1=math.sqrt(scalar(g2)/2),
            e_with_B2_sigma1=math.sqrt(scalar(e2)/2),realized_D=(D@M).tolist()))
    check('same_g_e_but_different_diagM_full_rank_D',
          max(abs(x-y) for x,y in zip(records[0]['diag_M'],records[1]['diag_M']))>.2)
    check('same_g_e_but_different_actual_request_outputs_full_rank_D',
          not torch.allclose(torch.tensor(records[0]['realized_D']),torch.tensor(records[1]['realized_D'])))
    # Complementary fixed-task-direction example with unequal target columns.
    D=torch.tensor([[1.,2.]],dtype=F64)
    u=D.reshape(2)/math.sqrt(5);v=torch.tensor([-2.,1.],dtype=F64)/math.sqrt(5)
    directional=[]
    for perpendicular in (.2,.8):
        M=.6*torch.outer(u,u)+perpendicular*torch.outer(v,v)
        _,K=K_from_M(M)
        r=ridge(torch.eye(2,dtype=F64),K,D)
        g2=torch.trace(D@r['G']@D.T);e2=torch.trace(D@r['E']@D.T)
        close(f'fixed_direction_g2_perp{perpendicular}',g2,torch.tensor(1.2,dtype=F64))
        close(f'fixed_direction_e2_perp{perpendicular}',e2,torch.tensor(.8,dtype=F64))
        close(f'fixed_direction_DM_perp{perpendicular}',D@M,.6*D)
        directional.append(dict(perpendicular_eigenvalue=perpendicular,M=M.tolist(),diag_M=M.diag().tolist(),
                                g_squared=scalar(g2),e_squared=scalar(e2),realized_D=(D@M).tolist()))
    DETAILS['same_ge_different_realization']=dict(full_rank_D=records,fixed_direction_D=directional,
        implication='Two scalar aggregate costs do not identify the per-request realization matrix, even with full-rank D and cross-request coupling.',
        caveat='diag(M) is a same-column coefficient; actual request output is (D M)_r and also includes off-diagonal contributions.')


def fixture(B):
    n=4
    centers=torch.eye(n,dtype=F64)[:B]*.7
    groups=[centers[:,None,:]+.04*torch.randn(B,c,n,dtype=F64) for c in (1,3)]
    return dict(B=B,n=n,groups=groups,A=[spd(n) for _ in range(3)],
        W=[torch.eye(n,dtype=F64)+.04*torch.randn(n,n,dtype=F64) for _ in range(3)],
        target=torch.randn(B,4,n,dtype=F64)*.2)


def causal_loss(D,f,detach_P=False,detach_upper_K=False):
    h=[x.clone() for x in f['groups']]
    energies=[];cache=[]
    for l,d in enumerate(D):
        K=torch.stack([x.mean(1) for x in h]).mean(0).T
        if detach_upper_K and l>0:K=K.detach()
        e=exact_writer(f['A'][l],K,d)
        P=e['P'].detach() if detach_P else e['P']
        U=d@P.T
        energies.append(torch.trace(U@f['A'][l]@U.T))
        h=[torch.tanh(x@(f['W'][l]+U).T) for x in h]
        cache.append(dict(K=K,P=P,condition_X=torch.linalg.cond(e['X']),realized=U@K))
    # Fixed mathematical surrogate; does not prescribe v9's method objective.
    return .5*(torch.cat(h,1)-f['target']).square().mean()+.01*sum(energies),cache


def full_causal_tests():
    receipts=[]
    for B in (1,2):
        f=fixture(B)
        D=[(.04*torch.randn(4,B,dtype=F64)).requires_grad_(True) for _ in range(3)]
        loss,cache=causal_loss(D,f)
        grad=torch.autograd.grad(loss,D)
        eps=1e-6;fd=[]
        for l,d in enumerate(D):
            close(f'causal_exact_realization_B{B}_L{l}',cache[l]['realized'],d)
            numerical=torch.empty_like(d)
            for j in range(d.numel()):
                plus=[x.detach().clone() for x in D];minus=[x.detach().clone() for x in D]
                plus[l].reshape(-1)[j]+=eps;minus[l].reshape(-1)[j]-=eps
                numerical.reshape(-1)[j]=(causal_loss(plus,f)[0]-causal_loss(minus,f)[0])/(2*eps)
            close(f'full_causal_exact_writer_FD_B{B}_L{l}',grad[l],numerical,atol=2e-7,rtol=5e-5)
            fd.append(numerical)
        gaps={}
        for stop in ('P','upper_K'):
            stopped,_=causal_loss(D,f,detach_P=stop=='P',detach_upper_K=stop=='upper_K')
            close(f'detach_{stop}_same_forward_B{B}',stopped,loss)
            sg=torch.autograd.grad(stopped,D)
            gap=max(scalar((g-s).abs().max()) for g,s in zip(grad,sg))
            check(f'detach_{stop}_loses_causal_gradient_B{B}',gap>1e-7,max_absolute_gradient_gap=gap)
            gaps[stop]=gap
        changed=[x.detach().clone() for x in D];changed[0]+=.015
        _,newcache=causal_loss(changed,f)
        close(f'first_key_remains_constant_B{B}',newcache[0]['K'],cache[0]['K'])
        check(f'upper_key_responds_to_lower_writer_B{B}',scalar((newcache[1]['K']-cache[1]['K']).abs().max())>1e-5)
        check(f'upper_exact_P_responds_to_lower_writer_B{B}',scalar((newcache[1]['P']-cache[1]['P']).abs().max())>1e-5)
        receipts.append(dict(B=B,layers=3,max_FD_absolute=max(scalar((g-n).abs().max()) for g,n in zip(grad,fd)),
            epsilon=eps,condition_X=[scalar(c['condition_X']) for c in cache],detached_gradient_gaps=gaps))
    DETAILS['causal_exact_writer']=dict(scope='FP64 well-conditioned nonlinear toy; not production FP32, LM task loss, or optimizer qualification',runs=receipts)


def near_collinear_tests():
    A=torch.eye(2,dtype=F64)
    D=torch.tensor([[1.,-1.]],dtype=F64)
    observations=[]
    for epsilon in (.1,.01,.001,.0001):
        K=torch.tensor([[1.,1.],[0.,epsilon]],dtype=F64)
        e=exact_writer(A,K,D);r=ridge(A,K,D)
        expected=1+4/epsilon**2
        close(f'near_collinear_exact_energy_eps{epsilon}',e['energy'],torch.tensor(expected,dtype=F64),atol=1e-7,rtol=2e-8)
        close(f'near_collinear_exact_target_eps{epsilon}',e['U']@K,D,atol=2e-8,rtol=2e-8)
        observations.append(dict(epsilon=epsilon,condition_X=scalar(torch.linalg.cond(e['X'])),
            exact_energy=scalar(e['energy']),analytic_exact_energy=expected,
            exact_target_error=scalar((e['U']@K-D).norm()),ridge_energy=scalar(torch.trace(r['U']@A@r['U'].T)),
            ridge_target_error=scalar((r['U']@K-D).norm()),ridge_realization=(r['U']@K).tolist()))
    for i in range(1,len(observations)):
        ratio=observations[i]['exact_energy']/observations[i-1]['exact_energy']
        check(f'near_collinear_energy_approximately_100x_for_10x_smaller_angle_{i}',99<ratio<101,ratio=ratio)
    DETAILS['near_collinear']=dict(A=A.tolist(),D=D.tolist(),K_definition='[[1,1],[0,epsilon]]',
        exact_U_definition='[1,-2/epsilon]',analytic_energy='1+4/epsilon^2',runs=observations,
        interpretation='Exact realization can make A-weighted write energy diverge near incompatible duplicate keys. A-energy is a preservation proxy, not a measured locality score.')


def merged_cost_tests():
    xs=torch.tensor([0.,1e-6,.01,.1,.171572875,1.,10.,100.],dtype=F64,requires_grad=True)
    new=(1+xs).rsqrt()
    old=(xs.sqrt()+1)/(1+xs)
    check('merged_scalar_capacity_cost_strictly_decreases',bool((new.detach()[1:]<new.detach()[:-1]).all()))
    derivative=torch.autograd.grad(new.sum(),xs)[0]
    close('merged_scalar_capacity_derivative',derivative,-.5*(1+xs).pow(-1.5))
    check('old_split_cost_can_increase_with_small_positive_capacity',scalar(old[3])>scalar(old[0]))
    # Same lambda; the change is a new aggregation, not a rescaling to old cost.
    pairs=[(torch.tensor([0.,.3,2.,7.],dtype=F64),torch.tensor([1.,3.,.2,0.],dtype=F64)),
           (torch.tensor([1.,2.,3.],dtype=F64),torch.tensor([1.,2.,3.],dtype=F64)),
           (torch.zeros(3,dtype=F64),torch.tensor([.1,1.,10.],dtype=F64))]
    inequalities=[]
    for i,(g,e) in enumerate(pairs):
        c=(g.square()+e.square()).sqrt()
        for arm in ('A','B'):
            newcost=.1*(c.sum() if arm=='A' else c.norm())
            oldcost=.1*(g.sum()+e.sum() if arm=='A' else g.norm()+e.norm())
            check(f'merged_old_new_norm_inequality_pair{i}_{arm}',
                  scalar(newcost)-1e-12<=scalar(oldcost)<=math.sqrt(2)*scalar(newcost)+1e-12,
                  merged=scalar(newcost),split=scalar(oldcost),upper_bound=math.sqrt(2)*scalar(newcost))
            inequalities.append(dict(pair=i,arm=arm,merged=scalar(newcost),split=scalar(oldcost)))
    # Values and all D/K adjoints: G+E implementation vs an independent
    # inverse-system identity, including the common sqrt across layers in B.
    value_records=[]
    for B in (1,3):
        n=5;L=3
        As=[spd(n) for _ in range(L)]
        Ks=[torch.randn(n,B,dtype=F64,requires_grad=True) for _ in range(L)]
        Ds=[torch.randn(2,B,dtype=F64,requires_grad=True) for _ in range(L)]
        sigmas=[.7,1.,1.3]
        for arm in ('A','B'):
            q=[];reference=[]
            for l,(A,K,D,sigma) in enumerate(zip(As,Ks,Ds,sigmas)):
                r=ridge(A,K,D)
                energy=torch.trace(D@(r['G']+r['E'])@D.T)/(B*sigma*sigma)
                identity=torch.trace(D@(torch.eye(B,dtype=F64)-r['M'])@D.T)/(B*sigma*sigma)
                direct=torch.trace(D@torch.linalg.solve(torch.eye(B,dtype=F64)+r['X'],D.T))/(B*sigma*sigma)
                close(f'merged_GplusE_IminusM_identity_B{B}_{arm}_L{l}',energy,identity)
                close(f'merged_GplusE_inverse_system_identity_B{B}_{arm}_L{l}',energy,direct)
                q.append(energy);reference.append(direct)
            merged=.1*(SelectedRoot.apply(torch.stack(q)).sum() if arm=='A' else SelectedRoot.apply(torch.stack(q).sum()))
            alternate=.1*(SelectedRoot.apply(torch.stack(reference)).sum() if arm=='A' else SelectedRoot.apply(torch.stack(reference).sum()))
            close(f'merged_policy_value_B{B}_{arm}',merged,alternate)
            grad=torch.autograd.grad(merged,Ds+Ks)
            refgrad=torch.autograd.grad(alternate,Ds+Ks)
            for i,(g,expected) in enumerate(zip(grad,refgrad)):
                close(f'merged_policy_full_DK_gradient_B{B}_{arm}_index{i}',g,expected)
            value_records.append(dict(B=B,arm=arm,merged_cost=scalar(merged),max_gradient_error=max(scalar((g-r).abs().max()) for g,r in zip(grad,refgrad))))
    for arm in ('A','B'):
        zero=[torch.zeros(2,3,dtype=F64,requires_grad=True) for _ in range(5)]
        capacities=[spd(3) for _ in zero]
        costs=[torch.trace(d@torch.linalg.solve(torch.eye(3,dtype=F64)+x,d.T))/3 for d,x in zip(zero,capacities)]
        policy=.1*(SelectedRoot.apply(torch.stack(costs)).sum() if arm=='A' else SelectedRoot.apply(torch.stack(costs).sum()))
        grad=torch.autograd.grad(policy,zero)
        check(f'merged_zero_selected_subgradient_{arm}',scalar(policy)==0 and all(bool(torch.isfinite(g).all() and (g==0).all()) for g in grad))
    # Conditional Loewner monotonicity at fixed D also follows the inverse.
    X1=spd(3);z=torch.randn(3,2,dtype=F64);X2=X1+z@z.T
    D=torch.randn(2,3,dtype=F64);I=torch.eye(3,dtype=F64)
    c1=torch.trace(D@torch.linalg.solve(I+X1,D.T))
    c2=torch.trace(D@torch.linalg.solve(I+X2,D.T))
    check('merged_fixed_D_PSD_capacity_increase_nonincreasing_cost',scalar(c2)<=scalar(c1)+1e-12)
    DETAILS['merged_policy']=dict(lambda_policy=.1,
        layer_cost='c_l = sqrt(g_l^2+e_l^2) = sqrt(tr(D_l (I-M_l) D_l^T)/(B sigma_l^2))',
        arm_A='sum_l c_l',arm_B='sqrt(sum_l c_l^2)',
        scalar_capacity=dict(x=xs.detach().tolist(),merged=new.detach().tolist(),split=old.detach().tolist()),
        inequalities=inequalities,value_gradient_records=value_records,
        monotonicity_scope='Fixed D and fixed normalization; does not assert monotonicity when D, keys, all layers, and model losses jointly change.',
        numerical_note='I-M is an algebraic identity; solve(I+X, D^T) or G+E avoids cancellation in explicitly subtracting nearly identity M.')


def inverse_compensation_tests():
    n=5;B=2
    C=spd(n);K=torch.randn(n,B,dtype=F64);D=torch.randn(3,B,dtype=F64)
    reference=None;ridge_writes=[];runs=[]
    for alpha in (.1,1.,10.,15000.):
        A=alpha*C;r=ridge(A,K,D);e=exact_writer(A,K,D)
        corrected_target=D@torch.linalg.solve(r['M'],torch.eye(B,dtype=F64))
        compensated=corrected_target@r['P'].T
        close(f'inverse_M_compensated_ridge_is_exact_alpha{alpha}',compensated,e['U'],atol=1e-9,rtol=1e-8)
        if reference is None:reference=e['U'].detach().clone()
        close(f'exact_writer_uniform_A_scale_cancels_alpha{alpha}',e['U'],reference,atol=1e-9,rtol=1e-8)
        ridge_writes.append(r['U'].detach())
        runs.append(dict(alpha=alpha,exact_writer_norm=scalar(e['U'].norm()),ridge_writer_norm=scalar(r['U'].norm()),
                         inverse_compensated_target_norm=scalar(corrected_target.norm()),exact_energy=scalar(e['energy'])))
    check('ridge_writer_keeps_uniform_A_scale_effect',scalar((ridge_writes[0]-ridge_writes[-1]).norm())>1e-3)
    H=torch.diag(torch.tensor([.1,.4,3.,.2,1.],dtype=F64))
    low=exact_writer(.1*C+H,K,D)['U'];high=exact_writer(10*C+H,K,D)['U']
    check('fixed_nonproportional_H_prevents_general_alphaC_invariance',scalar((low-high).norm())>1e-5,
          writer_gap=scalar((low-high).norm()))
    DETAILS['inverse_M_compensation']=dict(identity='D M^-1 P_ridge^T = D X^-1 K^T A^-1',
        runs=runs,scope='Uniform A=alpha*C scaling at fixed D,K. With fixed nonproportional H in A=alpha*C+H, exact writer need not be invariant.',
        interpretation='Inverse-realization compensation removes ridge attenuation and inherits exact-writer energy/conditioning issues; it is not a free correction.')


def main():
    started=datetime.datetime.now(datetime.timezone.utc).isoformat();start=time.monotonic()
    status='PASS';error=None
    try:
        check('CPU_only',os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available())
        ridge_and_exact_tests()
        rank_deficient_tests()
        context_mean_tests()
        identical_ge_counterexamples()
        full_causal_tests()
        near_collinear_tests()
        merged_cost_tests()
        inverse_compensation_tests()
    except Exception:
        status='FAIL';error=traceback.format_exc()
    results=dict(status=status,check_count=len(CHECKS),passed=sum(c['passed'] for c in CHECKS),
        failed=sum(not c['passed'] for c in CHECKS),error=error,checks=CHECKS,details=DETAILS,
        writer_selection='ridge main; exact realization pilot only',pseudoinverse_fallback=False,v8_q_scale_Adam_changed=False,
        limitations=['No production imports, language model, benchmark, CUDA, experiment, or runtime qualification.',
            'Exact-writer and finite-difference algebra is FP64; no FP32 materialization or commit parity claim.',
            'Context group means use the specified FP32 native aggregation in their separate example.',
            'Exact mean-key realization does not imply exact per-context behavior, paraphrase success, or locality preservation.',
            'Pseudoinverse thresholds affect numerical rank; the duplicate-key incompatibility example is exact algebra.'])
    path=HERE/'results.json';path.write_text(json.dumps(results,indent=2)+'\n')
    receipt=dict(status=status,started_utc=started,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-start,python=sys.version,torch=torch.__version__,device='cpu',
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],cuda_available=torch.cuda.is_available(),
        script=str(Path(__file__).resolve()),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        results=str(path),results_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),check_count=len(CHECKS),
        writer_selection='ridge main; exact realization pilot only',pseudoinverse_fallback=False,
        production_imports=False,production_code_changed=False,gpu_jobs_launched=False,
        v8_q_scale_Adam_changed=False,exit_code=0 if status=='PASS' else 1)
    (HERE/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status=status,checks=len(CHECKS),passed=results['passed'],elapsed_seconds=receipt['elapsed_seconds'],results=str(path))))
    if error:print(error,file=sys.stderr)
    return receipt['exit_code']


if __name__=='__main__':raise SystemExit(main())
