"""BPCW current guard and bounded integrated actual checks, no optimizer."""
from pathlib import Path
import math
import time
import numpy as np
import torch
from project.run_scripts.single_layer_edit_preserving_correction.binding import score_rows,quality_ok
from project.run_scripts.single_layer_edit_preserving_correction import geometry
from project.run_scripts.single_layer_edit_preserving_correction.common import tensor_sha
from project.run_scripts.single_layer_edit_preserving_correction.sequential_state import atomic_tensor
from .provenance import create_json
from .config import NUMERIC

class TechnicalHold(RuntimeError):pass

def require(path,condition,evidence):
    create_json(path,dict(evidence,check_status='PASS' if condition else 'FAIL'))
    if not condition:raise TechnicalHold(Path(path).stem)

def current_guard(oracle,rows,anchor,weight,WN,ideal,K,allowed):
    result=geometry.invariant_diagnostics(ideal,weight.double()-WN.double(),WN,K,allowed)
    if not result['ideal_pass']:raise TechnicalHold('NOMINAL_DKE')
    current=score_rows(oracle,weight,rows)
    quality,reasons=quality_ok(current,anchor,1e-4)
    maxnll=max(abs(current[k]['nll']-anchor[k]['nll']) for k in anchor)
    stats=[oracle.compare_logits(i,WN,weight,left_route='cached',right_route='cached') for i in range(len(oracle.caches))]
    mx=max(s['max_abs'] for s in stats);rms=math.sqrt(sum(s['squared_error'] for s in stats)/sum(s['logit_elements'] for s in stats))
    result.update(quality=quality,reasons=reasons,max_NLL_difference=maxnll,logit_max=mx,logit_rms=rms,
        current_sequence_scores=current,anchor_sequence_scores=anchor,
        native_success_ID_rule='SUBSET; gains allowed',past='NOT_APPLICABLE_EMPTY_B1')
    result['pass']=bool(quality and maxnll<=1e-4 and mx<=1e-3 and rms<=1e-4 and
                        result['actual_response_pass'] and result['actual_leakage_pass'])
    return result

def base_binding(oracle,W0,out):
    """Every protected answer token: raw greedy ↔ TF argmax, then fixed4 repeats."""
    out=Path(out);scan=oracle.scan(W0,0.);diff=[]
    for doc,cap in zip(scan['documents'],oracle.capsules,strict=True):
        for row,base in zip(doc['positions'],cap['steps'],strict=True):
            diff.append(dict(source_row_id=cap['source_row_id'],position=row['position'],
                margin_abs=abs(row['margin']-base['margin']),logp_abs=abs(row['logp']-base['logp']),id_same=row['preserved']))
    require(out/'generation-TF.json',all(r['id_same'] for r in diff) and max(r['logp_abs'] for r in diff)<=1e-4,
        dict(all_position_comparisons=diff,EOS_included=True,censored_fake_EOS=False,raw_greedy_vs_TF=True))
    variation=0.;repeats=[]
    for i in range(min(4,len(oracle.capsules))):
        cap=oracle.capsules[i];values=[]
        for _ in range(3):
            logits=oracle.logits_at(i,W0,cap['positions'])
            y=torch.tensor(cap['y0'],device=oracle.device);idx=torch.arange(len(y),device=oracle.device)
            chosen=logits[idx,y].double();other=logits.clone();other[idx,y]=-torch.inf
            values.append((chosen-other.max(-1).values.double()).cpu())
        change=float((torch.stack(values).max(0).values-torch.stack(values).min(0).values).max())
        variation=max(variation,change);repeats.append(dict(index=i,max_gap_variation=change))
    tau=max(1e-4,10*variation)
    require(out/'gap-reserve.json',tau<=1e-3,dict(tau_gap=tau,max_repeat_gap_variation=variation,repeats=repeats,
        frozen_before_native_choice_scan=True,calibration_weight='W0',never_performance_tuned=True))
    return tau,scan

def fd_pair(oracle,pair,WN,G,direction,native_norm,out):
    out=Path(out);norm=float(direction.norm())
    if norm==0:
        result=dict(status='NO_DIRECTION',derivative_pass=False,all_grid_not_applicable=True)
        create_json(out/'result.json',result);return result
    v=direction.double()/norm;AD=float((G*v).sum())
    args=(pair['index'],WN,pair['position'],pair['target'],pair['competitor'])
    repeats=[oracle.pair_value(*args) for _ in range(3)]
    noise=max(max(repeats)-min(repeats),np.finfo(np.float64).eps*max(1,abs(repeats[0])))
    h0=.01*native_norm;points=[]
    atomic_tensor(out/'direction.pt',dict(direction=v,AD=AD,pair=pair,WN=tensor_sha(WN)))
    spacing=(torch.nextafter(WN,torch.full_like(WN,float('inf')))-WN).double().abs()
    for k in range(12):
        h=h0/2**k;values={};actual={}
        for name,sign in (('plus',1),('minus',-1)):
            weight=(WN.double()+sign*h*v).float();d=weight.double()-WN.double()
            nz=d!=0;ulp=d.abs()/spacing.clamp_min(torch.finfo(torch.float32).tiny)
            probe=dict(h=h,sign=sign,nonzero=int(nz.sum()),actual_norm=float(d.norm()),
                max_ULP=float(ulp.max()),median_nonzero_ULP=float(ulp[nz].median()) if nz.any() else 0.,
                weight_sha256=tensor_sha(weight),actual_linear_delta=float((G*d).sum()))
            create_json(out/f'{k:02d}-{name}-before.json',probe)
            values[name]=oracle.pair_value(pair['index'],weight,pair['position'],pair['target'],pair['competitor'])
            actual[name]=probe
        fd=(values['plus']-values['minus'])/(2*h)
        relative=abs(fd-AD)/abs(AD) if AD else None;signal=abs(values['plus']-values['minus'])/2
        match=bool(relative is not None and relative<=.01 and signal>10*noise and all(r['nonzero'] for r in actual.values()))
        point=dict(k=k,h=h,AD=AD,FD=fd,relative=relative,signal=signal,noise=noise,match=match,values=values,actual=actual,
            actual_linear_FD=(actual['plus']['actual_linear_delta']-actual['minus']['actual_linear_delta'])/(2*h))
        create_json(out/f'{k:02d}.json',point);points.append(point)
    windows=[i for i in range(11) if points[i]['match'] and points[i+1]['match']]
    status='PASS' if windows else ('SMALL_AD_UNRESOLVED' if h0*abs(AD)<=10*noise else 'NO_RESOLVED_ADJACENT_WINDOW')
    result=dict(status=status,AD=AD,h0=h0,repeats=repeats,noise=noise,points=points,
        selected_adjacent=None if not windows else [windows[0],windows[0]+1],competitor_frozen=True)
    create_json(out/'result.json',result)
    if not windows:raise TechnicalHold('PAIR_FD:'+status)
    return result

def integrated(rt,ref,cur,rows,K,WN,space,allowed,out):
    out=Path(out);started=time.monotonic();checks=[]
    # Actual4 reference plus first native/canonical old/new inputs only.
    for label,oracle,indices in (('reference',ref,range(4)),('current',cur,range(min(4,len(cur.caches))))):
        for i in indices:
            parity=oracle.compare_logits(i,WN,WN);rt.sync_oracles()
            stationary=oracle.key_stationarity(i,WN);rt.sync_oracles()
            require(out/f'{label}-{i}-physical.json',parity['max_abs']<=1e-4 and stationary['byte_equal'],
                dict(parity=parity,stationarity=stationary));checks.append(f'{label}-{i}-physical')
    pj=geometry.projector_diagnostics(space)
    if space.status!='RANK_UNRESOLVED':require(out/'projector.json',pj['status']=='PASS',pj)
    else:create_json(out/'projector.json',dict(status='RANK_UNRESOLVED',new_nonzero_not_tested=True))
    gradient_directions=[];factor_checks=[]
    for i in range(4):
        cap=ref.capsules[i];step=cap['steps'][0]
        pair=dict(index=i,position=cap['positions'][0],target=cap['y0'][0],competitor=step['competitor'])
        A,k,value=ref.pair_factor(i,WN,**{x:pair[x] for x in ('position','target','competitor')})
        direct,g=ref.direct_pair(i,WN,**{x:pair[x] for x in ('position','target','competitor')});rt.sync_oracles()
        factor=A.double()@k.double().T
        relative=float((factor-g).norm()/g.norm().clamp_min(1e-300))
        atomic_tensor(out/f'pair-{i}-evidence.pt',dict(A=A,K=k,pair=pair,direct_gradient=g,WN_sha256=tensor_sha(WN)))
        require(out/f'pair-{i}-AD.json',relative<=1e-4 and abs(value-direct)<=1e-4,
            dict(pair=pair,cached_margin=value,direct_margin=direct,relative_gradient_error=relative,
                factor_norm=float(factor.norm()),direct_norm=float(g.norm()),physical_selected_leaf=True))
        if i<2 and space.status=='RESOLVED':
            direction=space.project(factor)
            gradient_directions.append((pair,g,direction))
            # Independent dense contraction confirms the factor orientation.
            U=space.project(k.T).T
            projected=space.project(factor)
            reconstructed=A.double()@U.T
            error=float((projected-reconstructed).norm()/projected.norm().clamp_min(1e-300))
            factor_checks.append(dict(index=i,projected_factor_relative=error,
                dense_gram_diagonal=float(projected.square().sum()),factor_gram_diagonal=float(((A.double().T@A.double())*(U.T@U)).sum())))
            require(out/f'pair-{i}-projected-factor.json',error<=1e-10,factor_checks[-1])
        del factor,g,A,k
    for i,(pair,g,direction) in enumerate(gradient_directions):
        fd_pair(ref,pair,WN,g,direction,float((WN.double()-rt.W0.double()).norm()),out/f'FD-pair{i}')
    # Repeat current sequence observation on bounded prefix; full final guard remains one.
    subset=[r for r in rows if r['cache']<4]
    q0=score_rows(cur,WN,subset);q1=score_rows(cur,WN,subset)
    nll=max(abs(q0[k]['nll']-q1[k]['nll']) for k in q0)
    require(out/'current-noop.json',nll<=1e-5 and all(q0[k]['strict']==q1[k]['strict'] for k in q0),dict(max_NLL_difference=nll,sequence_count=len(subset)))
    rt.guard()
    result=dict(status=('RANK_UNRESOLVED_NOT_FULL_TECHNICAL_PASS' if space.status=='RANK_UNRESOLVED'
                        else 'INTEGRATED_MODEL_CHECKS_PASS'),scope='reference4/current-prefix, fresh B100 native shared',
        seconds=time.monotonic()-started,space_status=space.status,nonzero_FD_directions=len(gradient_directions),
        histories_during_checks=0,continuation='B1 checkpoint reload/resume check still pending',
        sequential_authorized=False,checks=checks,factor_checks=factor_checks)
    create_json(out/'result.json',result);return result
