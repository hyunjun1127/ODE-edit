"""Bounded same-mask fixed-candidate checks; full builder reverse is diagnostic."""
import time
import torch
from project.run_scripts.jlz_native_writer_aware.builder import reverse as diagnostic_reverse
from project.run_scripts.jlz_native_writer_aware.physical import linear
from project.run_scripts.jlz_realized_subject.causal_builder import build as explicit_build
from project.run_scripts.jlz_realized_subject.qualification import full_masked
from project.run_scripts.jlz_realized_subject.writer import rng_snapshot,rng_equal
from project.run_scripts.jlz_native_writer_aware.common import require,write
from .engine import CandidateObjective
from .subject import evaluate,pullback
from .projection import project_capped_energy


def comparison(x,y,atol=2e-5,rtol=2e-4):
    x,y=x.detach().double().cpu(),y.detach().double().cpu();require(x.shape==y.shape,'QUALIFICATION_SHAPE')
    require(bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all()),'QUALIFICATION_NONFINITE')
    error=(x-y).abs();limit=atol+rtol*y.abs()
    return dict(passed=bool((error<=limit).all()),maximum_error=float(error.max()),
        failed_elements=int((error>limit).sum()),atol=atol,rtol=rtol)


def gradient_comparison(x,y):
    x,y=x.detach().cpu(),y.detach().cpu()
    require(bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all()),'QUALIFICATION_NONFINITE_GRADIENT')
    error=float((x.double()-y.double()).square().mean().sqrt());ref=float(y.double().square().mean().sqrt())
    dot=float((x.double()*y.double()).sum());norm=float(x.double().norm()*y.double().norm())
    return dict(passed=error<=1e-6+1e-3*ref,error_RMS=error,reference_RMS=ref,
        limit=1e-6+1e-3*ref,relative_norm_error=float((x.double()-y.double()).norm()/y.double().norm()) if y.norm()>0 else None,
        cosine=dot/norm if norm else None)


def same_layer_reference(a,entry,R,built,adjoint):
    """V14 direct physical Linear VJP, not the production matmul helper."""
    result={}
    for l in a.sites:
        r=R[l].detach().requires_grad_(True);P=built['P'][l].detach();key=built['raw'][l].to(a.device)
        with torch.enable_grad():
            action=linear(key,r,P,built['weights'][l],route='direct')
            result[l]=torch.autograd.grad(action,r,adjoint[l].to(a.device))[0].double()
    return result


def MB2_entry(entry):
    """V14 row-stream shape reference; immutable original owner-prefix cache.

    This is a BUILD-value diagnostic only.  Production subject groups remain
    whole native owners, and no partial-owner stopping/reduction is asserted.
    """
    groups=[]
    for group in entry['groups']:
        count=len(group['rows'])
        for begin in range(0,count,2):
            end=min(begin+2,count)
            def select(value):
                if isinstance(value,torch.Tensor):
                    # A 1D cache_position is a TIME axis, even when T happens
                    # to equal the native owner's row count.
                    return value[begin:end] if value.ndim>=2 and value.shape[0]==count else value
                if isinstance(value,tuple):return tuple(select(v) for v in value)
                if isinstance(value,dict):return {k:select(v) for k,v in value.items()}
                return value
            groups.append(dict(rows=group['rows'][begin:end],tokens={k:v[begin:end] for k,v in group['tokens'].items()},
                cache=dict(key=group['cache']['key'][begin:end],residual=group['cache']['residual'][begin:end],
                           kwargs=select(group['cache']['kwargs']))))
    return dict(entry,groups=groups,first_geometry={})


def diagnose(engine,R,built,mask,fixedowners=None):
    """Same terminal, whole-B writer, fixed owners' SAME loss mask only."""
    start=time.monotonic();a,entry=engine.a,engine.entry;B=entry['pack']['n_requests']
    selected=None if fixedowners is None else set(map(int,fixedowners))
    require(selected is None or selected=={0,1},'DIAGNOSTIC_FIXED_OWNER_POLICY')
    fixed=torch.as_tensor(mask,dtype=torch.bool).cpu().clone()
    if selected is not None:fixed &= torch.tensor([r in selected for r in range(B)])
    before=rng_snapshot();guard=a.guard();physical_before=dict(getattr(a,'physical_calls',{}))
    observed=evaluate(a,entry,built,backward=True,fixed_mask=fixed,owner_subset=selected)
    g,receipt=pullback(built,observed['adjoint'],R)
    same=same_layer_reference(a,entry,R,built,observed['adjoint'])
    full,reverse=diagnostic_reverse(a,entry,R,built,observed['adjoint'],cached=True,prune_first=False)
    samecmp={str(l):gradient_comparison(g[l],same[l]) for l in a.sites}
    fullcmp={str(l):gradient_comparison(g[l],full[l]) for l in a.sites}
    require(all(v['passed'] for v in samecmp.values()),'SAME_LAYER_REFERENCE_PARITY')
    require(fullcmp[str(a.sites[-1])]['passed'],'LAST_LAYER_SAME_MASK_FULL_GRADIENT_PARITY')
    require(rng_equal(before) and a.guard()==guard,'DIAGNOSTIC_NONMUTATION')
    return dict(status='MEASURED',candidate=built['candidate'],fixed_owners=None if selected is None else sorted(selected),
        full_writer_B=B,same_loss_mask=fixed.tolist(),same_layer=samecmp,full_gradient=fullcmp,
        lower_full_gradient_gate=False,subject_forward_groups=observed['forward_groups'],
        subject_backward_groups=observed['backward_groups'],diagnostic_reverse=reverse,
        production_reverse=False,new_fits=0,optimizer_updates=0,extra_candidate_BUILD=0,
        physical_recompute_calls={k:v-physical_before.get(k,0) for k,v in getattr(a,'physical_calls',{}).items()},
        timer_policy='diagnostic total inclusive; reverse/subject timers nested, not additive',
        seconds=time.monotonic()-start)


laterdiagnostic=diagnose


def qualify(a,entry,out=None,max_candidates=3):
    B=entry['pack']['n_requests'];require(2<=B<=4 and 1<=max_candidates<=3,'FIXED_QUALIFICATION_BUDGET')
    start=time.monotonic();engine=CandidateObjective(a,entry);before=rng_snapshot();guard=a.guard();records=[]
    for candidate in range(max_candidates):
        R={};projection=None
        for l in a.sites:
            x=torch.sin(torch.arange(a.dims[l][0]*B,device=a.device,dtype=torch.float32).reshape(a.dims[l][0],B)+l+.3)
            x=x/x.double().norm(dim=0).float()
            amount=0. if candidate==0 else .025
            R[l]=(x*entry['anchors'][l][None,:]*amount).detach()
            if candidate==2:R[l][:,0]=0
        if candidate==2:
            # The third fixed candidate is mixed-zero AND on the feasible
            # native energy boundary.  This is a projection check, no fit.
            caps=torch.stack([.75*entry['anchors'][l] for l in a.sites])
            rho=torch.minimum(caps[0],.75*entry['anchors'][a.profile['anchor_layer']])
            proposal={l:v*60 for l,v in R.items()}
            R,projection=project_capped_energy(proposal,caps,rho)
        built=engine.build(R,candidate)
        with torch.no_grad():reference=explicit_build(a,dict(entry,first_geometry={}),R,candidate,route='dense')
        mb2=CandidateObjective(a,MB2_entry(entry)).build(R,candidate);checks={};mb2checks={}
        for l in a.sites:
            checks[str(l)]={name:comparison(built[name][l],reference['geometry'][l][name] if name in ('K','P') else reference[name][l])
                            for name in ('K','P','weights','v')}
            mb2checks[str(l)]={name:comparison(built[name][l],mb2[name][l]) for name in ('K','P','weights','v')}
        require(all(v['passed'] for parts in checks.values() for v in parts.values()),'NATIVE_BUILD_EXPLICIT_PARITY')
        require(all(v['passed'] for parts in mb2checks.values() for v in parts.values()),'V14_MB2_BUILD_VALUE_PARITY')
        mask=torch.ones(B,dtype=torch.bool)
        if candidate==2:mask[0]=False
        observed=evaluate(a,entry,built,backward=True,fixed_mask=mask)
        with full_masked(a):full=evaluate(a,entry,built,backward=True,fixed_mask=mask)
        loss=comparison(observed['F'],full['F'],1e-5,1e-4)
        require(loss['passed'],'NATIVE_OWNER_FULL_HOOK_LOSS_PARITY')
        g,_=pullback(built,observed['adjoint'],R);nativeg,_=pullback(built,full['adjoint'],R)
        nativegrad={str(l):gradient_comparison(g[l],nativeg[l]) for l in a.sites}
        require(all(v['passed'] for v in nativegrad.values()),'NATIVE_OWNER_FULL_HOOK_GRADIENT_PARITY')
        diagnostic=diagnose(engine,R,built,mask)
        records.append(dict(candidate=candidate,build=checks,V14_MB2_build_values=mb2checks,
            MB2_prefix='same immutable native-owner entry boundary; BUILD physical row groups <=2',
            FP32_feasible_boundary_projection=projection,loss=loss,native_hook_gradient=nativegrad,diagnostic=diagnostic))
        del built,reference,mb2,observed,full
    require(rng_equal(before) and a.guard()==guard,'QUALIFICATION_STATE_NONMUTATION')
    record=dict(status='TECHNICAL_READY',B=B,fixed_candidates=max_candidates,extra_fit=0,optimizer_updates=0,
        permanent_commit=0,production_builder_reverse=0,diagnostic_only_reverse=True,
        candidate_checks=records,source_profile_generic=True,CPU_tests_not_actual_model=True,
        call_policy='max3 MAIN fixed candidates total; full owner groups; same fixed active masks; no fit',
        seconds=time.monotonic()-start)
    if out is not None:write(out/'qualification.json',record)
    return record
