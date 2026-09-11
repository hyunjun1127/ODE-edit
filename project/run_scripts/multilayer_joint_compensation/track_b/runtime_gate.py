"""Minimum actual first-request correctness, not a performance/endpoint gate.

Finite differences validate mean Current NLL gradient, not the full Hessian.
GGN is independently checked for symmetry/PSD. Perturbations are engineering
probes discarded before the sealed B trajectory; no controller feedback.
"""
import math
import torch
from ..evaluation import panel,materialized
from ..observations import pack
from ..linear_solve import norm,dot,scale,add,finite
from ..contracts import digest,save,tensor_sha

FD_RELATIVE_TOLERANCE=.02
GGN_RELATIVE_TOLERANCE=.002
FD_DISPLACEMENT_RELATIVE_WEIGHT_NORM=.001
FD_STEPS=(1.,.5,.25)
FD_REFINEMENT_STEPS=(.125,.0625,.03125,.015625)


def fd_observation(plus,minus,ad,eps,perturbation):
    """Original allowance unchanged; resolution cannot enlarge that allowance."""
    fd=(plus-minus)/(2*eps)
    roundoff=64*torch.finfo(torch.float32).eps*max(abs(plus),abs(minus),1.)/eps
    bound=FD_RELATIVE_TOLERANCE*max(abs(ad),abs(fd))+roundoff
    error=abs(fd-ad)
    sign_resolved=abs(ad)>roundoff and abs(fd)>roundoff
    sufficient=sign_resolved and all(r['nonzero_fraction']>0 and r['actual_norm']>r['rounding_norm'] for r in perturbation)
    return dict(eps=eps,plus_loss=plus,minus_loss=minus,ad=ad,fd=fd,error=error,
        relative_error=error/max(abs(ad),abs(fd)) if max(abs(ad),abs(fd)) else None,
        numerical_bound=bound,roundoff=roundoff,loss_difference=abs(plus-minus),
        loss_resolution=2*eps*roundoff,sign_resolved=sign_resolved,sufficient_signal=sufficient,
        passed=math.isfinite(fd) and error<=bound,perturbation=perturbation)


def adjacent_fd_pass(rows):
    pairs=[dict(indices=[i,i+1],passed=all(r['passed'] and r['sufficient_signal'] for r in rows[i:i+2]))
           for i in range(len(rows)-1)]
    return any(p['passed'] for p in pairs),pairs


def initial_gate(problem,view,packed,wn_teacher,tok,model,ledger,out):
    if model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError('INITIAL_MODEL_NOT_FROZEN_EVAL')
    if not packed['Current']:
        return dict(status='B0_NOOP_NO_MODEL_DIRECTION',no_forward_for_empty=True)
    first_id=packed['Current'][0]['ordinal']
    rows=[dict(r) for r in packed['Current'] if r['ordinal']==first_id]
    total=sum(r['context_weight'] for r in rows)
    for row in rows:row['context_weight']/=total
    p=panel(view,rows,wn_teacher,tok.pad_token_id,'current',2)
    w=tuple(t.detach().cpu().clone() for t in problem.wn)
    linear=p.linearize(w,need_nll_gradient=True)
    a=problem.project(linear.nll_gradient);anorm=norm(a)
    raw_direction=a if anorm else problem.project(tuple(torch.ones_like(t) for t in w))
    denominator=norm(raw_direction)
    if not denominator:
        raise RuntimeError('INITIAL_NO_PERMITTED_PROBE_DIRECTION')
    displacement=FD_DISPLACEMENT_RELATIVE_WEIGHT_NORM*norm(w)
    d=scale(raw_direction,displacement/denominator)
    ad=dot(linear.nll_gradient,d);fdrows=[]
    identity=dict(WN=[tensor_sha(t) for t in w],direction=[tensor_sha(t) for t in d],
        direction_formula='unchanged projected Current NLL gradient scaled to .001*WN Frobenius norm',
        first_ordinal=first_id,rows=digest(rows),AD=ad,
        historical_direction_byte_hash='NOT_RECORDED_44991; same derivation/input/teacher reused, no retrospective byte-equality claim')
    with ledger.time('initial_actual_directional_fd'):
        for eps in FD_STEPS+FD_REFINEMENT_STEPS:
            losses=[];perturbations=[]
            for sign in (1,-1):
                state=add(w,d,sign*eps)
                actual=tuple(x-r for x,r in zip(state,w));nominal=scale(d,sign*eps)
                rounding=add(actual,nominal,-1)
                perturbations.append(dict(sign=sign,actual_norm=norm(actual),nominal_norm=norm(nominal),
                    rounding_norm=norm(rounding),nonzero_fraction=sum(int(torch.count_nonzero(x)) for x in actual)/sum(x.numel() for x in actual)))
                losses.append(p.observe(state)['mean_nll'])
            fdrows.append(fd_observation(*losses,ad,eps,perturbations))
        passed,pairs=adjacent_fd_pass(fdrows)
        fd_receipt=dict(status='RAW_ADJACENT_FD_PASS' if passed else 'JVP_FD_NUMERICALLY_UNRESOLVED',
            identity=identity,rows=fdrows,adjacent_pairs=pairs,old_eps=list(FD_STEPS),new_eps=list(FD_REFINEMENT_STEPS),
            original_allowance_unchanged=True,extrapolation_gate_influence=0,
            signal_rule='abs(AD),abs(rawFD)>original roundoff; both actual FP32 displacements nonzero and norm>rounding norm')
        if out is not None:save(out/'fd-refinement.json',fd_receipt)
        if not passed:raise RuntimeError('JVP_FD_NUMERICALLY_UNRESOLVED:'+repr(fdrows))
    y=problem.project(tuple(t.roll(1,0) for t in d))
    with ledger.time('initial_actual_ggn_symmetry_psd'):
        fd_action=finite(p.ggn(w,d));fy=finite(p.ggn(w,y))
        quadratic=dot(d,fd_action);symmetry=abs(dot(d,fy)-dot(fd_action,y))
        scale_q=norm(d)*norm(fd_action);scale_sym=norm(d)*norm(fy)+norm(fd_action)*norm(y)
        qbound=GGN_RELATIVE_TOLERANCE*scale_q;sbound=GGN_RELATIVE_TOLERANCE*scale_sym
        if quadratic < -qbound or symmetry>sbound:raise RuntimeError('INITIAL_GGN_PSD_OR_SYMMETRY')
    first=next(iter(pack(view,rows,tok.pad_token_id,2)))
    candidate=add(w,d)
    with torch.no_grad(),ledger.time('initial_actual_materialization_parity'):
        expected=first.logits(candidate)
        full_candidate=view.full_weights(tuple(t.to(first.input_ids.device) for t in candidate))
        with materialized(model,view.full.names,full_candidate,ledger,purpose='initial_gate_probe'):
            actual_all=model(input_ids=first.input_ids,attention_mask=first.attention_mask,use_cache=False).logits
            actual=actual_all[first.positions[0],first.positions[1]]
            difference=float((actual-expected).abs().max())
            if not torch.equal(actual,expected):raise RuntimeError(f'INITIAL_FUNCTIONAL_PHYSICAL_BYTES_FORWARD_MISMATCH:{difference}')
        view.full.assert_live(bytes_check=True)
    ledger.add('initial_actual_request_count',1)
    return dict(status='ACTUAL_FIRST_REQUEST_KERNEL_AND_APPLICATION_VALID',first_ordinal=first_id,
        first_case_id=rows[0]['case_id'],row_identity=digest([r['identity'] for r in rows]),
        logical_requests=1,physical_microbatch=2,global_context_weight_sum=sum(r['context_weight'] for r in rows),
        reduction='same token/context formula; nested first effective request',
        FD=fdrows,FD_adjacent_pairs=pairs,FD_identity=identity,FD_relative_tolerance=FD_RELATIVE_TOLERANCE,FD_steps=list(FD_STEPS+FD_REFINEMENT_STEPS),
        FD_displacement_weight_norm_ratio=FD_DISPLACEMENT_RELATIVE_WEIGHT_NORM,
        current_gradient_norm=anorm,GGN_quadratic=quadratic,GGN_PSD_bound=qbound,
        GGN_symmetry_absolute=symmetry,GGN_symmetry_bound=sbound,GGN_relative_tolerance=GGN_RELATIVE_TOLERANCE,
        GGN_operator='FULL_SEQUENCE_CURRENT_KL_PLUS_NLL_PROFILE_OUTER',
        cross_layer_actual_A_gate='SH1_SEPARATE_REQUIRED_NOT_CLAIMED_HERE',
        functional_physical_forward_max_abs=difference,physical_equality='BYTE_EXACT',
        original_live_pointer_version_bytes_restored=True,engineering_probe_carried_to_trajectory=False,
        observer_current_only=True,semantic_performance_gate=False,source_weight_dtype='float32',
        panel_compute_counts=p.counts,additional_direct_materialized_forwards=1,
        full_B100_fidelity_and_endpoint='NOT_YET_COMPLETE')
