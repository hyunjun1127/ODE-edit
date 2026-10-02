#!/usr/bin/env python3
"""CPU-only v8 mathematical qualification; no production imports or model loads.

FP32 native aggregation is tested as executed. Finite differences use a smooth
FP64 algebra surrogate, not a claim about differentiating FP32 rounding.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
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
DTYPE = torch.float64
HERE = Path(__file__).resolve().parent
CHECKS = []
DETAILS = {}


class SelectedRoot(torch.autograd.Function):
    """Unsmoothed square root with the specified zero selected subgradient."""
    @staticmethod
    def forward(ctx, x):
        out=x.clamp_min(0).sqrt()
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad):
        out,=ctx.saved_tensors
        scale=torch.zeros_like(out)
        nonzero=out>0
        scale[nonzero]=.5/out[nonzero]
        return grad*scale


def number(value):
    return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)


def require(name, condition, **detail):
    passed = bool(condition)
    CHECKS.append(dict(name=name, passed=passed, **detail))
    if not passed:
        raise AssertionError(f'{name}: {detail}')


def close(name, actual, expected, atol=1e-10, rtol=1e-9):
    actual = torch.as_tensor(actual).detach().double()
    expected = torch.as_tensor(expected).detach().double()
    err = (actual - expected).abs()
    limit = atol + rtol * expected.abs()
    require(name, (err <= limit).all(), max_absolute=number(err.max()),
            max_excess=number((err-limit).max()), atol=atol, rtol=rtol)


def exact(name, actual, expected):
    require(name, actual.dtype == expected.dtype and torch.equal(actual, expected),
            dtype=str(actual.dtype), shape=list(actual.shape))


def spd(n, shift=1.0):
    x = torch.randn(n, n, dtype=DTYPE)
    return x @ x.T / n + shift * torch.eye(n, dtype=DTYPE)


def aggregate(groups, fp32=True):
    """Each group is [request, context, input feature]."""
    dtype = torch.float32 if fp32 else DTYPE
    means = [g.to(dtype).mean(dim=1) for g in groups]
    return torch.stack(means, dim=0).mean(dim=0).T.to(DTYPE)


def geometry(A, K, route='dense'):
    if route == 'dense':
        P = torch.linalg.solve(A + K @ K.T, K)
    else:
        AK = torch.linalg.solve(A, K)
        P = torch.linalg.solve(torch.eye(K.shape[1], dtype=DTYPE) + K.T @ AK, AK.T).T
    F = P.T @ K - torch.eye(K.shape[1], dtype=DTYPE)
    return P, P.T @ A @ P, F @ F.T


def solve_vjp(A, K, P, Q):
    Y = torch.linalg.solve((A + K @ K.T).T, Q)
    return Y @ (torch.eye(K.shape[1], dtype=DTYPE) - P.T @ K) - P @ (Y.T @ K)


def aggregation_tests():
    groups = [torch.randn(3, n, 5, dtype=torch.float32, requires_grad=True) for n in (1,5)]
    K = aggregate(groups)
    reference32 = torch.stack([g.mean(1) for g in groups]).mean(0).T
    exact('native_kappa_FP32_nested_mean_then_FP64', K, reference32.double())
    flat = torch.cat(groups,1).mean(1).T.double()
    require('native_group_mean_differs_from_flat_uniform_context_mean', not torch.allclose(K,flat),
            max_absolute=number((K-flat).abs().max()))
    upstream = torch.randn_like(K)
    grads = torch.autograd.grad((K*upstream).sum(), groups)
    for i,(g,grad) in enumerate(zip(groups,grads)):
        expected = ((upstream.float().T/len(groups)).unsqueeze(1).expand_as(g)/g.shape[1])
        exact(f'FP32_context_adjoint_group_{i}', grad, expected)
    # Composition check: the mean operation carries the implicit writer and
    # direct-E adjoints back to every original context with native FP32 casts.
    K=aggregate(groups)
    A=spd(K.shape[0]);D=torch.randn(2,K.shape[1],dtype=DTYPE)
    P,G,E=geometry(A,K)
    automatic=torch.autograd.grad(.5*torch.trace(D@(G+E)@D.T),groups)
    F=P.T@K-torch.eye(K.shape[1],dtype=DTYPE);Qd=D.T@D
    adjoint=solve_vjp(A,K,P,A@P@Qd+K@F.T@Qd)+P@Qd@F
    for i,(g,grad) in enumerate(zip(groups,automatic)):
        expected=((adjoint.float().T/len(groups)).unsqueeze(1).expand_as(g)/g.shape[1])
        close(f'complete_writer_to_native_context_adjoint_{i}',grad,expected,atol=1e-7,rtol=1e-6)
    # A precision witness: conversion before averaging changes the operation.
    witness = [torch.tensor([[[1e8],[1.],[-1e8]]],dtype=torch.float32),
               torch.tensor([[[3.]]],dtype=torch.float32)]
    native = aggregate(witness,True)
    early64 = aggregate(witness,False)
    require('mean_before_FP64_cast_is_numerically_material', not torch.equal(native,early64),
            native=number(native),cast_before_means=number(early64))
    DETAILS['aggregation'] = dict(native_K_shape=list(K.shape),context_group_sizes=[1,5],
        native_dtype_before_cast='torch.float32',solve_dtype='torch.float64',
        context_adjoint='cast grad_kappa to FP32, transpose, / number_of_groups, / group_size')


def geometry_tests():
    for n,B in ((3,1),(5,2),(4,3)):
        A = spd(n)
        K = torch.randn(n,B,dtype=DTYPE,requires_grad=True)
        D = torch.randn(3,B,dtype=DTYPE)
        P,G,E = geometry(A,K)
        Pd,Gd,Ed = geometry(A,K,'dual')
        tag=f'n{n}_B{B}'
        close(f'dense_dual_P_{tag}',P,Pd)
        close(f'dense_dual_G_{tag}',G,Gd)
        close(f'dense_dual_E_{tag}',E,Ed)
        close(f'GE_identity_{tag}',G+E,torch.eye(B,dtype=DTYPE)-K.T@P)
        residual=(A+K@K.T)@P-K
        require(f'solve_residual_{tag}',number(residual.norm()/K.norm())<1e-12,
                relative_residual=number(residual.norm()/K.norm()))
        Q=torch.randn_like(P)
        automatic=torch.autograd.grad((P*Q).sum(),K,retain_graph=True)[0]
        close(f'implicit_solve_VJP_{tag}',solve_vjp(A,K,P,Q),automatic)
        F=P.T@K-torch.eye(B,dtype=DTYPE)
        Qd=D.T@D
        gradP=A@P@Qd+K@F.T@Qd
        directE=P@Qd@F
        energy=.5*torch.trace(D@(G+E)@D.T)
        total=torch.autograd.grad(energy,K)[0]
        close(f'direct_E_plus_implicit_geometry_VJP_{tag}',solve_vjp(A,K,P,gradP)+directE,total)
        Kleaf=K.detach().requires_grad_(True)
        Pleaf=P.detach()
        Ehalf=.5*(D@(Pleaf.T@Kleaf-torch.eye(B,dtype=DTYPE))).square().sum()
        close(f'direct_E_K_adjoint_{tag}',torch.autograd.grad(Ehalf,Kleaf)[0],directE)

    # Preserving context moments is a different writer, even at the same mean.
    A=2*torch.eye(2,dtype=DTYPE)
    K=torch.tensor([[1.],[0.]],dtype=DTYPE)
    contexts=torch.tensor([[2.,0.],[2.,-2.]],dtype=DTYPE)
    meanP=geometry(A,K)[0]
    fullP=torch.linalg.solve(A+.5*contexts@contexts.T,K)
    require('mean_key_writer_is_not_full_context_writer',not torch.allclose(meanP,fullP),
            mean_key_P=meanP.tolist(),full_context_P=fullP.tolist())


def direct_linear_tests():
    X=torch.randn(7,4,dtype=DTYPE,requires_grad=True)
    D=torch.randn(3,2,dtype=DTYPE,requires_grad=True)
    P=torch.randn(4,2,dtype=DTYPE,requires_grad=True)
    W=torch.randn(3,4,dtype=DTYPE)
    upstream=torch.randn(7,3,dtype=DTYPE)
    y=X@(W+D@P.T).T
    gx,gd,gp=torch.autograd.grad((y*upstream).sum(),(X,D,P))
    close('physical_linear_D_adjoint',upstream.T@(X@P),gd)
    close('physical_linear_P_adjoint',X.T@(upstream@D),gp)
    close('physical_linear_input_adjoint',upstream@(W+D@P.T),gx)


def make_causal_fixture(B):
    n=3
    return dict(B=B,n=n,groups=[torch.randn(B,c,n,dtype=DTYPE)*.3 for c in (1,2)],
        weights=[torch.eye(n,dtype=DTYPE)+torch.randn(n,n,dtype=DTYPE)*.12 for _ in range(3)],
        priors=[spd(n,1.5) for _ in range(3)],
        anchors=[.8+torch.rand(B,dtype=DTYPE) for _ in range(3)],
        target=torch.randn(B,3,n,dtype=DTYPE)*.2)


def causal_components(D,fixture,arm,detach_upper_K=False,detach_P=False,route='dense'):
    h=[g.clone() for g in fixture['groups']]
    v=[g.clone() for g in fixture['groups']]
    gs=[];es=[];cache=[]
    for l,d in enumerate(D):
        K=aggregate(h,False)
        if detach_upper_K and l>0:K=K.detach()
        P,G,E=geometry(fixture['priors'][l],K,route)
        if detach_P:
            P=P.detach()
            F=P.T@K-torch.eye(fixture['B'],dtype=DTYPE)
            G=P.T@fixture['priors'][l]@P;E=F@F.T
        U=d@P.T
        W=fixture['weights'][l]
        h=[torch.tanh(x@(W+U).T) for x in h]
        v=[torch.tanh(x@W.T)+d.T[:,None,:] for x in v]
        scale=fixture['B']*fixture['anchors'][l].square().mean()
        gs.append(torch.trace(d@G@d.T)/scale)
        es.append(torch.trace(d@E@d.T)/scale)
        cache.append(dict(K=K,P=P))
    policy=.1*sum(SelectedRoot.apply(torch.stack(q)).sum() if arm=='A' else SelectedRoot.apply(torch.stack(q).sum()) for q in (gs,es))
    native=.3*(torch.cat(v,1)-fixture['target']).square().mean()
    native=native+.07*sum((d.norm(dim=0)/a.square()).sum()/fixture['B'] for d,a in zip(D,fixture['anchors']))
    actual=.2*(torch.cat(h,1)-fixture['target']).square().mean()
    return (native,policy,actual),cache


def full_causal_tests():
    fd_receipts=[]
    for B in (1,3):
        fixture=make_causal_fixture(B)
        for arm in ('A','B'):
            D=[(.08*torch.randn(3,B,dtype=DTYPE)).requires_grad_(True) for _ in range(3)]
            parts,cache=causal_components(D,fixture,arm)
            loss=sum(parts)
            grad=torch.autograd.grad(loss,D)
            finite=[]
            eps=1e-6
            for l,d in enumerate(D):
                fd=torch.empty_like(d)
                for j in range(d.numel()):
                    plus=[x.detach().clone() for x in D]
                    minus=[x.detach().clone() for x in D]
                    plus[l].view(-1)[j]+=eps;minus[l].view(-1)[j]-=eps
                    lp=sum(causal_components(plus,fixture,arm)[0])
                    lm=sum(causal_components(minus,fixture,arm)[0])
                    fd.view(-1)[j]=(lp-lm)/(2*eps)
                finite.append(fd)
                close(f'full_causal_FD_B{B}_{arm}_L{l}',grad[l],fd,atol=2e-7,rtol=5e-5)
            fd_receipts.append(dict(B=B,arm=arm,max_absolute=max(number((g-f).abs().max()) for g,f in zip(grad,finite)),
                                    epsilon=eps,aggregation='FP64 smooth algebra surrogate'))
            for stop in ('K','P'):
                parts_stopped,_=causal_components(D,fixture,arm,detach_upper_K=stop=='K',detach_P=stop=='P')
                close(f'stopped_{stop}_same_forward_B{B}_{arm}',sum(parts_stopped),loss)
                stopped=torch.autograd.grad(sum(parts_stopped),D)
                gap=max(number((g-s).abs().max()) for g,s in zip(grad,stopped))
                require(f'stopped_{stop}_loses_causal_gradient_B{B}_{arm}',gap>1e-7,max_gradient_absolute=gap)
            dualparts,dualcache=causal_components(D,fixture,arm,route='dual')
            close(f'full_causal_dense_dual_loss_B{B}_{arm}',sum(dualparts),loss)
            dualgrad=torch.autograd.grad(sum(dualparts),D)
            for l,(g,dg) in enumerate(zip(grad,dualgrad)):
                close(f'full_causal_dense_dual_gradient_B{B}_{arm}_L{l}',g,dg)
            perturbed=[x.detach().clone() for x in D]
            perturbed[0]+=.03
            _,changed=causal_components(perturbed,fixture,arm)
            exact(f'first_key_invariant_B{B}_{arm}',cache[0]['K'],changed[0]['K'])
            require(f'upper_key_changes_with_lower_writer_B{B}_{arm}',
                    number((cache[1]['K']-changed[1]['K']).abs().max())>1e-6)
    DETAILS['finite_difference']=fd_receipts
    fixture=make_causal_fixture(2)
    for arm in ('A','B'):
        zero=[torch.zeros(3,2,dtype=DTYPE,requires_grad=True) for _ in range(3)]
        parts,_=causal_components(zero,fixture,arm)
        total=torch.autograd.grad(sum(parts),zero,retain_graph=True)
        policy_grad=torch.autograd.grad(parts[1],zero)
        require(f'zero_D_finite_total_gradient_{arm}',all(bool(torch.isfinite(g).all()) for g in total))
        require(f'zero_D_policy_selected_zero_subgradient_{arm}',number(parts[1])==0 and all(bool((g==0).all()) for g in policy_grad))


def bridge_tests():
    for B in (1,3):
        fixture=make_causal_fixture(B)
        m=len(fixture['weights'])
        scale=[a[None,:]/math.sqrt(3*m) for a in fixture['anchors']]
        for arm in ('A','B'):
            q=[(.2*torch.randn(3,B,dtype=DTYPE)).requires_grad_(True) for _ in range(m)]
            physical=[s*x for s,x in zip(scale,q)]
            parts,_=causal_components(physical,fixture,arm)
            gq=torch.autograd.grad(sum(parts),q)
            leaves=[x.detach().requires_grad_(True) for x in physical]
            leafparts,_=causal_components(leaves,fixture,arm)
            close(f'bridge_same_physical_loss_B{B}_{arm}',sum(leafparts),sum(parts))
            total=[torch.zeros_like(d) for d in leaves]
            for component in leafparts:
                partial=torch.autograd.grad(component,leaves,retain_graph=True)
                for acc,g in zip(total,partial):acc.add_(g)
            for l,(s,gd,expected) in enumerate(zip(scale,total,gq)):
                close(f'Dleaf_to_q_adjoint_once_B{B}_{arm}_L{l}',s*gd,expected)
            # Negative control: forgetting the scaling is not equivalent.
            require(f'bridge_missing_scale_detectable_B{B}_{arm}',
                    max(number((gd-g).abs().max()) for gd,g in zip(total,gq))>1e-5)
    # Generic dimensions and anchors: pointwise coordinate changes preserve
    # native norm and the physical feasible ball without an extra batch factor.
    for B in (1,4):
        dims=[1,3,7,11,19];m=len(dims)
        for l,d in enumerate(dims):
            a=torch.linspace(.2,2.,B,dtype=DTYPE)
            s=a[None,:]/math.sqrt(d*m)
            physical=torch.randn(d,B,dtype=DTYPE)*a[None,:]*.1
            q=physical/s
            close(f'coordinate_roundtrip_B{B}_L{l}',s*q,physical)
            close(f'native_norm_coordinate_identity_B{B}_L{l}',
                  .5*(physical.norm(dim=0)/a.square()).sum(),
                  .5*(q.norm(dim=0)/(a*math.sqrt(d*m))).sum())


def first_step_tests():
    receipts=[]
    for dims in ([1],[4096],[1,3,7,11,4096]):
        m=len(dims)
        for B in (1,3):
            q=[torch.zeros(d,B,dtype=DTYPE,requires_grad=True) for d in dims]
            anchors=[torch.linspace(.2,4.,B,dtype=DTYPE)*(1+.1*l) for l in range(m)]
            scales=[a[None,:]/math.sqrt(d*m) for a,d in zip(anchors,dims)]
            opt=torch.optim.Adam(q,lr=.1,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
            for l,(v,s) in enumerate(zip(q,scales)):
                gd=torch.linspace(-1.,2.,v.numel(),dtype=DTYPE).reshape_as(v)
                gd.view(-1)[0]=0.
                if gd.numel()>1:gd.view(-1)[1]=1e-14
                v.grad=s*gd
            opt.step()
            ratios=[]
            for l,(v,s,a,d) in enumerate(zip(q,scales,anchors,dims)):
                ratio=(s*v).norm(dim=0)/a
                ratios.append(ratio)
                require(f'first_step_per_layer_bound_dims{dims}_B{B}_L{l}',
                        bool((ratio<=.1/math.sqrt(m)+1e-13).all()),maximum=number(ratio.max()))
                # A radial scale on q must preserve the physical cap definition.
                big=(v.detach()+1)*1000
                D=s*big
                factor=torch.minimum(torch.ones_like(a),.75*a/D.norm(dim=0))
                qc=big*factor
                close(f'q_and_D_clamp_equivalence_dims{dims}_B{B}_L{l}',s*qc,D*factor)
                require(f'physical_clamp_bound_dims{dims}_B{B}_L{l}',
                        bool(((s*qc).norm(dim=0)<=.75*a+1e-12).all()))
                close(f'q_clamp_radius_dims{dims}_B{B}_L{l}',qc.norm(dim=0),torch.full_like(a,.75*math.sqrt(d*m)))
            joint=torch.stack(ratios).square().sum(0)
            require(f'first_step_joint_bound_dims{dims}_B{B}',bool((joint<=.01+1e-13).all()),maximum=number(joint.max()))
            receipts.append(dict(dimensions=dims,B=B,m=m,joint_normalized_squared_step_max=number(joint.max())))
    DETAILS['first_step_bounds']=receipts


def counterexamples():
    d=4096;m=5
    anchors=[torch.tensor([2.+l],dtype=DTYPE) for l in range(m)]
    D=[torch.zeros(d,1,dtype=DTYPE,requires_grad=True) for _ in range(m)]
    raw=torch.optim.Adam(D,lr=.1,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
    for x in D:x.grad=-torch.ones_like(x)
    raw.step()
    pre=[number(x.norm()/a) for x,a in zip(D,anchors)]
    require('unscaled_Adam_first_step_all_exceed_native_radius',all(v>.75 for v in pre),relative_norms=pre)
    q=[torch.zeros(d,1,dtype=DTYPE,requires_grad=True) for _ in range(m)]
    scales=[a[None,:]/math.sqrt(d*m) for a in anchors]
    opt=torch.optim.Adam(q,lr=.1,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
    history=[]
    for t in range(1,25):
        for v,s in zip(q,scales):v.grad=-s*torch.ones_like(v)
        opt.step()
        ratios=[number((s*v).norm()/a) for s,v,a in zip(scales,q,anchors)]
        clips=sum(v>.75 for v in ratios)
        with torch.no_grad():
            for v,s,a in zip(q,scales,anchors):
                factor=torch.minimum(torch.ones_like(a),.75*a/(s*v).norm(dim=0))
                v.mul_(factor)
        history.append(dict(update=t,pre_clamp_relative_norms=ratios,clipped_layers=clips))
    require('normalized_first_step_no_clipping',history[0]['clipped_layers']==0)
    require('aligned_gradient_first_clips_at_update17',next(h['update'] for h in history if h['clipped_layers'])==17)
    require('normalized_coordinates_do_not_forbid_late_all_layer_cap',history[-1]['clipped_layers']==m)
    DETAILS['constant_direction_counterexample']=history
    # Independent boundary/momentum witness. Objective is convex with its
    # minimum at .65, inside the .75 cap: f(x)=1.1 max(x-.65,0)-x.
    x=0.;mm=0.;vv=0.;out=[]
    for t in range(1,25):
        g=-1. if x<.65 else .1
        mm=.9*mm+.1*g;vv=.999*vv+.001*g*g
        step=-.1*(mm/(1-.9**t))/(math.sqrt(vv/(1-.999**t))+1e-8)
        x=max(0.,min(.75,x+step))
        out.append(dict(update=t,gradient=g,value=x,proposed_step=step))
    require('post_clamp_momentum_can_remain_outward_with_inward_gradient',
            all(r['gradient']>0 and r['proposed_step']>0 and r['value']==.75 for r in out[7:]))
    DETAILS['momentum_counterexample']=dict(objective='1.1*relu(x-.65)-x',interior_minimum=.65,
        cap=.75,scope='Scalar projected-Adam limitation; not a simulation of the method or evidence of its actual gradient directions',trajectory=out)


def asymmetric_toy():
    results=[]
    for arm in ('A','B'):
        m=2;d=1
        q=[torch.zeros(1,1,dtype=DTYPE,requires_grad=True) for _ in range(m)]
        s=1/math.sqrt(d*m)
        optimizer=torch.optim.Adam(q,lr=.1,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
        target=torch.tensor([.6,.2],dtype=DTYPE)
        first=None
        for t in range(24):
            optimizer.zero_grad(set_to_none=True)
            D=torch.cat([s*v.reshape(1) for v in q])
            policy=.1*(D.abs().sum() if arm=='A' else D.norm())
            loss=5*(D-target).square().sum()+.5*D.abs().sum()+policy
            loss.backward();optimizer.step()
            with torch.no_grad():
                for v in q:v.mul_(min(1.,.75/number((s*v).norm())))
            if first is None:first=[number(s*v.detach()) for v in q]
        final=[number(s*v.detach()) for v in q]
        require(f'asymmetric_toy_all_layers_eligible_{arm}',all(v>0 for v in first))
        require(f'asymmetric_toy_unequal_amplitudes_possible_{arm}',abs(final[0]-final[1])>.15 and all(0<v<.75 for v in final),final=final)
        results.append(dict(arm=arm,first=first,after_24_updates=final))
    DETAILS['asymmetric_toy']=dict(scope='Capability witness only: scalar quadratic task surrogate + norm + aggregation penalty; no language-model quality claim',runs=results)


def history_tests():
    groups=[torch.randn(3,n,5,dtype=torch.float32) for n in (1,5)]
    k32=aggregate(groups).float()
    H0=torch.zeros(5,5,dtype=torch.float32,device='cpu')
    H1=H0+k32@k32.T
    require('native_history_shape_dtype_device',H1.shape==(5,5) and H1.dtype==torch.float32 and H1.device.type=='cpu')
    exact('native_history_one_mean_key_Gram_append',H1-H0,k32@k32.T)
    context_moment=sum(torch.einsum('bci,bcj->ij',g,g)/g.shape[1]/len(groups) for g in groups)
    require('native_history_mean_key_Gram_not_full_context_Gram',not torch.allclose(H1,context_moment),
            max_absolute=number((H1-context_moment).abs().max()))
    require('duplicate_append_is_distinguishable',not torch.equal(H1,H1+k32@k32.T))
    DETAILS['history']=dict(update='H_next = H_entry + kappa_FP32 @ kappa_FP32.T',
        append_count_in_this_test=1,shape=list(H1.shape),dtype=str(H1.dtype),device=str(H1.device),
        scope='Algebra and FP32 CPU Gram semantics only; production transaction/once-only append behavior is not tested')


def fp32_optimizer_bridge_tests():
    """Production-coordinate dtype contract; still no production/model imports."""
    eta=.1
    joint_limit=eta**2
    joint_tolerance=1e-8+1e-5*joint_limit
    receipts=[]
    for dims in ([4096],[4096]*5,[1,7,19,128,4096]):
        m=len(dims)
        for B in (1,3):
            tag=f'dims{dims}_B{B}'
            anchors=[torch.linspace(.2,4.,B,dtype=torch.float32)*(1+.1*l) for l in range(m)]
            scales=[(a.double()/math.sqrt(d*m)).float()[None,:] for a,d in zip(anchors,dims)]
            q=[torch.zeros(d,B,dtype=torch.float32,requires_grad=True) for d in dims]
            optimizer=torch.optim.Adam(q,lr=eta,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
            for l,(v,s,a,d) in enumerate(zip(q,scales,anchors,dims)):
                exact(f'FP32_scale_after_FP64_division_{tag}_L{l}',s,(a.double()/math.sqrt(d*m)).float()[None,:])
                gd=torch.linspace(-1.,2.,v.numel(),dtype=torch.float32).reshape_as(v)
                gd.view(-1)[0]=0.
                if gd.numel()>1:gd.view(-1)[1]=1e-14
                v.grad=s*gd
                require(f'FP32_q_scale_D_gradient_dtype_{tag}_L{l}',
                        all(t.dtype==torch.float32 for t in (v,s,s*v,gd,v.grad)))
            optimizer.step()
            physical=[s*v for s,v in zip(scales,q)]
            ratios=[]
            for l,(v,d,a) in enumerate(zip(q,physical,anchors)):
                state=optimizer.state[v]
                require(f'FP32_Adam_moment_dtype_{tag}_L{l}',state['exp_avg'].dtype==torch.float32 and state['exp_avg_sq'].dtype==torch.float32)
                ratio=d.detach().double().norm(dim=0)/a.double()
                ratios.append(ratio)
                bound=eta/math.sqrt(m)
                require(f'FP32_first_step_layer_bound_{tag}_L{l}',bool((ratio<=bound+1e-8+1e-5*bound).all()),
                        maximum=number(ratio.max()),theoretical_bound=bound)
            joint=torch.stack(ratios).square().sum(0)
            require(f'FP32_first_step_joint_bound_{tag}',bool((joint<=joint_limit+joint_tolerance).all()),
                    measured_maximum=number(joint.max()),theoretical_bound=joint_limit,measurement_tolerance=joint_tolerance)
            clipped=sum(int((r>.75).sum()) for r in ratios)
            require(f'FP32_first_step_no_clamp_{tag}',clipped==0,clipped_layer_requests=clipped)

            # Fixed physical D, mixed FP32 task/norm + FP64 geometry surrogate.
            # The objective is constructed as B * mean_loss, exactly once.
            targets=[torch.randn(d,B,dtype=torch.float32)*.2 for d in dims]
            def mean_loss(ds):
                native=sum(.5*(x-t).square().sum(0).mean()+.5*(x.norm(dim=0)/a.square()).mean()
                           for x,t,a in zip(ds,targets,anchors))
                Q=torch.eye(B,dtype=DTYPE)+.2*torch.ones(B,B,dtype=DTYPE)
                geometry_cost=.03*sum(((x.double().T@x.double())*Q).sum()/B for x in ds)
                return native.double()+geometry_cost
            fullq=[(.04*torch.randn(d,B,dtype=torch.float32)).requires_grad_(True) for d in dims]
            fullD=[s*v for s,v in zip(scales,fullq)]
            full_mean=mean_loss(fullD)
            full_sum=B*full_mean
            fullgrad=torch.autograd.grad(full_sum,fullq)
            leafD=[x.detach().requires_grad_(True) for x in fullD]
            leaf_mean=mean_loss(leafD)
            leaf_sum=B*leaf_mean
            gd=torch.autograd.grad(leaf_sum,leafD,retain_graph=True)
            gd_mean=torch.autograd.grad(leaf_mean,leafD)
            close(f'FP32_bridge_SUM_loss_same_physical_D_{tag}',leaf_sum,full_sum,atol=1e-8,rtol=1e-8)
            bridge=[s*g for s,g in zip(scales,gd)]
            bridge_errors=[]
            for l,(g,expected,gmean,gD) in enumerate(zip(bridge,fullgrad,gd_mean,gd)):
                require(f'FP32_bridge_gradient_dtype_{tag}_L{l}',g.dtype==expected.dtype==gD.dtype==torch.float32)
                close(f'FP32_bridge_chain_once_{tag}_L{l}',g,expected,atol=1e-7,rtol=1e-6)
                close(f'FP32_bridge_B_SUM_not_MEAN_{tag}_L{l}',gD,B*gmean,atol=2e-6,rtol=2e-5)
                bridge_errors.append(number((g-expected).abs().max()))
            if B>1:
                require(f'FP32_bridge_missing_B_detectable_{tag}',
                        max(number((s*gmean-g).abs().max()) for s,gmean,g in zip(scales,gd_mean,fullgrad))>1e-5)
            # One actual FP32 Adam update from full autograd vs explicit bridge.
            bridgedq=[v.detach().clone().requires_grad_(True) for v in fullq]
            fullopt=torch.optim.Adam(fullq,lr=eta,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
            bridgeopt=torch.optim.Adam(bridgedq,lr=eta,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
            for vf,vb,gf,gb in zip(fullq,bridgedq,fullgrad,bridge):vf.grad=gf;vb.grad=gb
            fullopt.step();bridgeopt.step()
            for l,(vf,vb) in enumerate(zip(fullq,bridgedq)):
                close(f'FP32_bridge_Adam_update_parity_{tag}_L{l}',vb,vf,atol=1e-7,rtol=1e-6)
            receipts.append(dict(dimensions=dims,m=m,B=B,joint_normalized_squared_step_max=number(joint.max()),
                joint_bound=joint_limit,measurement_tolerance=joint_tolerance,
                first_step_clipped=clipped,bridge_max_absolute=max(bridge_errors),
                q_scale_D_grad_moments_dtype='torch.float32'))
    DETAILS['FP32_optimizer_bridge']=dict(scale='(anchor.double()/sqrt(d*m)).float(), fixed within batch',
        first_joint_bound_tolerance='1e-8 + 1e-5 * eta^2',
        measurement='Physical D is FP32; its normalized squared norm is reduced in FP64 for reporting',
        bridge_scope='Fixed-D FP32 coordinate and B-SUM chain rule with FP64 geometry surrogate; full causal FD remains separately FP64',
        runs=receipts)


def main():
    start=time.monotonic()
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    status='PASS';error=None
    try:
        require('CPU_only',os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available())
        aggregation_tests()
        geometry_tests()
        direct_linear_tests()
        full_causal_tests()
        bridge_tests()
        first_step_tests()
        counterexamples()
        asymmetric_toy()
        history_tests()
        fp32_optimizer_bridge_tests()
    except Exception:
        status='FAIL';error=traceback.format_exc()
    result=dict(status=status,check_count=len(CHECKS),passed=sum(c['passed'] for c in CHECKS),
        failed=sum(not c['passed'] for c in CHECKS),error=error,checks=CHECKS,details=DETAILS,
        limitations=['No model, benchmark, GPU, timing, or production runner qualification.',
            'Finite differences use a smooth FP64 algebra surrogate; FP32 aggregation is separately checked exactly.',
            'Unscaled/normalized/momentum/asymmetry examples are mathematical capability and limitation witnesses, not experimental efficacy evidence.',
            'First-step bound does not imply later non-saturation, sparse allocation, better locality, or convergence in 24 updates.'])
    result_path=HERE/'results.json'
    result_path.write_text(json.dumps(result,indent=2)+'\n')
    receipt=dict(status=status,started_utc=started,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-start,python=sys.version,torch=torch.__version__,device='cpu',
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],cuda_available=torch.cuda.is_available(),
        script=str(Path(__file__).resolve()),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        output=str(result_path),output_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        check_count=len(CHECKS),exit_code=0 if status=='PASS' else 1,
        production_code_changed=False,gpu_jobs_launched=False)
    (HERE/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status=status,checks=len(CHECKS),passed=result['passed'],elapsed_seconds=receipt['elapsed_seconds'],results=str(result_path))))
    if error:print(error,file=sys.stderr)
    return receipt['exit_code']


if __name__=='__main__':
    raise SystemExit(main())
