"""One native anchor, fixed response, first acceptable L8-only endpoint."""
import math
import numpy as np
import torch
from .common import save,tensor_save,identity,Timer
from .numerical import POLICY
from .runtime import tensor_sha,capture_rng,restore_rng

def jsonable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,torch.Tensor):return x.detach().cpu().tolist()
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    return x

def materialize(anchor,Q,c):
    out=anchor.detach().cpu().clone()
    assert out.dtype==torch.float32 and len(Q)==len(c)
    for q,coefficient in zip(Q,c):
        assert q.dtype==torch.float32 and q.shape==out.shape and q.device.type=='cpu'
        assert math.isfinite(float(coefficient))
        out.add_(q*float(coefficient))
    assert torch.isfinite(out).all(),'NONFINITE_L8_WRITE'
    return out

def quality(anchor,candidate):
    reasons=[]
    for name in ('current','past'):
        ref=anchor[name];row=candidate[name]
        if ref is None:
            assert row is None;continue
        assert math.isfinite(row['E']),'NONFINITE_REWRITE'
        if row['E']>ref['E']+POLICY['epsilon_L']:reasons.append(name+'_MEAN')
        for key in ('strict_ids','preference_ids'):
            if not set(ref[key]).issubset(row[key]):reasons.append(name+'_'+key)
    return reasons

def score(rt,current,past):
    return dict(current=rt.observe(lambda:rt.response.panel(current)[0]),
        past=rt.observe(lambda:rt.response.panel(past)[0]) if past else None,
        base=rt.observe(lambda:rt.response.base()[0]))

def build(rt,current,past,arm,out,*,pilot=False):
    from .geometry import orthonormal_basis
    rng=capture_rng();state=rt.state();anchor8=rt.W[8].detach().cpu().clone()
    gradients={};counts={}
    base,gB=rt.observe(lambda:rt.response.base(gradient=True));gradients['B']=gB
    save(out/'anchor-base.json',base)
    if pilot:tensor_save(out/'gradient-B.pt',dict(gradient=gB,weight_hash=state['W']['8']))
    cur,gR=rt.observe(lambda:rt.response.panel(current,gradient=arm=='R-QP'))
    save(out/'anchor-current.json',cur)
    if gR is not None:
        gradients['R']=gR
        if pilot:tensor_save(out/'gradient-R.pt',dict(gradient=gR,weight_hash=state['W']['8']))
    old,gH=rt.observe(lambda:rt.response.panel(past,gradient=arm=='R-QP')) if past else (None,None)
    if old is not None:save(out/'anchor-past.json',old)
    if gH is not None:gradients['H']=gH
    # Inputs are fixed before proposal. No evaluator P/N or Dev access here.
    anchor=dict(current=cur,past=old,base=base)
    arrays,basis=orthonormal_basis(gradients)
    Q=[torch.from_numpy(q) for q in arrays]
    save(out/'basis.json',jsonable(basis))
    qref=tensor_save(out/'directions.pt',dict(Q=Q,anchor_W8_sha=state['W']['8'],basis=jsonable(basis)))
    dots=np.asarray([sum(float((g.flatten()[i:i+1000000].double()*q.flatten()[i:i+1000000].double()).sum())
        for i in range(0,g.numel(),1000000)) for q in Q for g in [gB]],dtype=np.float64)
    response=None
    if Q and (pilot or base['B']>POLICY['epsilon_B']):
        response=rt.observe(lambda:rt.response.response(Q,current,past,cur,old,include_guards=arm=='R-QP'))
        response['b_from_full_gradient']=dots
        save(out/'response.json',jsonable(response))
    save(out/'model-seal.json',dict(arm=arm,anchor_state=state,base_B=base['B'],m=len(Q),
        direction_payload=qref,b_actual_QP=dots.tolist(),gradient_norms={k:float(g.double().norm()) for k,g in gradients.items()},
        current_gradient_sweeps=int(gR is not None),past_gradient_sweeps=int(gH is not None),base_gradient_sweeps=1,
        official_P_N_controller_access=0,operational_P8_M8=False,L8_target_calls=0))
    assert rt.state()==state,'BUILD_STATE_CHANGED'
    return dict(anchor=anchor,anchor8=anchor8,Q=Q,basis=basis,response=response,gradients=gradients if pilot else None,
        rng=rng,state=state,arm=arm,dots=dots)

def select(rt,current,past,model,out):
    from .qp import solve_repair_qp
    anchor=model['anchor'];B=anchor['base']['B'];Q=model['Q'];m=len(Q)
    assert math.isfinite(B),'NONFINITE_ANCHOR_BASE'
    probes=[];selected='WN';reason='BASE_SMALL' if B<=POLICY['epsilon_B'] else 'BASIS_ZERO'
    chosen=model['anchor8']; selected_score=anchor
    if m and B>POLICY['epsilon_B']:
        response=model['response'];radius=math.sqrt(2*B/m)
        for index in range(POLICY['max_endpoints']):
            folder=out/f'probe-{index:02d}';folder.mkdir()
            solution=solve_repair_qp(model['dots'],response['H'],response['A'],response['s'],B,
                arm=model['arm'],radius=radius)
            save(folder/'QP.json',jsonable(solution))
            c=np.asarray(solution['c'],dtype=np.float64)
            predicted=float(solution['predicted_gain'])
            assert math.isfinite(predicted),'NONFINITE_PREDICTED_GAIN'
            candidate=materialize(model['anchor8'],Q,c)
            delta_norm=float((candidate.double()-model['anchor8'].double()).norm())
            probe=dict(index=index,radius=radius,c=c.tolist(),predicted=predicted,
                actual_weight_sha=tensor_sha(candidate),actual_delta_norm=delta_norm,
                QP=identity(folder/'QP.json'),actual=None,agreement=None,reasons=[],accepted=False)
            if predicted<=POLICY['epsilon_B']:probe['reasons'].append('PREDICTED_SMALL')
            if delta_norm==0:probe['reasons'].append('FP32_ZERO')
            if not probe['reasons']:
                rt.apply8(candidate,model['rng'])
                assert tensor_sha(rt.W[4])==model['state']['W']['4'] and tensor_sha(rt.M4)==model['state']['M4']
                observed=score(rt,current,past);save(folder/'scores.json',observed)
                assert math.isfinite(observed['base']['B']),'NONFINITE_CANDIDATE_BASE'
                actual=B-observed['base']['B'];agreement=actual/predicted
                assert math.isfinite(actual) and math.isfinite(agreement),'NONFINITE_GAIN_AGREEMENT'
                probe.update(actual=actual,agreement=agreement)
                probe['reasons']=quality(anchor,observed)
                if actual<=POLICY['epsilon_B']:probe['reasons'].append('ACTUAL_SMALL_OR_NEGATIVE')
                if agreement<POLICY['minimum_agreement']:probe['reasons'].append('AGREEMENT')
                if not probe['reasons']:
                    probe['accepted']=True;selected=f'probe-{index:02d}';chosen=candidate;selected_score=observed
            save(folder/'decision.json',probe);probes.append(probe)
            rt.apply8(model['anchor8'],model['rng'])
            assert rt.state()==model['state'],'PROBE_ROLLBACK_FAILED'
            if probe['accepted']:reason='FIRST_ACCEPTABLE';break
            reason=','.join(probe['reasons']);radius*=POLICY['shrink']
    selection=dict(arm=model['arm'],selected=selected,reason=reason,probes=probes,
        anchor_state=model['state'],selected_W8_sha=tensor_sha(chosen),current=selected_score['current'],
        past=selected_score['past'],base=selected_score['base'],m=m,selection_sealed=True,
        official_P_N_access_before_seal=0,checkpoint='SKIPPED_USER_DIRECTED')
    save(out/'selection.json',selection)
    rt.apply8(chosen,model['rng'])
    assert tensor_sha(rt.W[4])==model['state']['W']['4']
    return selection,chosen
