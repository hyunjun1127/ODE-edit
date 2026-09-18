"""Bounded cold8 actual T. Every stage/probe is persisted before judgement.

This entrypoint cannot submit jobs. M is fail-closed on its READY receipt.
"""
import argparse
import json
import math
import resource
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from .common import write,save_tensor,tensor_sha,member,digest,Timer
from .runtime import Runtime,invariant
from .binding import score_rows
from . import geometry

NUMERIC=dict(version='ENFC-T-v1',FD_scales=12,FD_initial_native_relative=.01,FD_relative=.01,
    FD_signal_noise=10,FD_adjacent=2,FD_noise='maximum three repeated WN KL range and fp64 epsilon*max(1,abs(L)); observed noise, not full model bound',
    FD_selection='first adjacent pair in locked descending-h grid; full grid retained',
    FD_small_AD='all nominal h*abs(AD) <=10*noise -> SMALL_AD_UNRESOLVED, never derivative PASS',
    FD_zero_direction='exact zero -> NO_DIRECTION; remaining nonzero directions still required',
    gradient_relative=1e-4,projector=1e-10,ideal_DK=1e-10,actual_DK=1e-5,actual_leak=1e-5,
    protected_logit_max=1e-3,protected_logit_RMS=1e-4,protected_NLL=1e-4,ID_changes=0,
    noop_NLL=1e-5,noop_logit=1e-4,KL_floor=1e-6,COV_roundoff_multiplier=10,
    random_seed=20260918,physical_and_cached_microbatch=1,max_extra_gradient_directions=3,
    invariant_probe='first EN-F method Polyak proposal L/chi at WN, not diagnostic FD h; floor/zero classified separately',
    native_fits=2,native_requests=8,diagnostic_gradient_objective='S64 fixed W0 signed full-vocab KL')


class TechnicalHold(RuntimeError):pass


def require(directory,name,condition,evidence):
    payload=dict(evidence);payload['check_status']='PASS' if condition else 'FAIL'
    payload.setdefault('status',payload['check_status'])
    receipt=write(directory/(name+'.json'),payload)
    if not condition:raise TechnicalHold(name)
    return receipt


def fd(oracle,WN,gradient,direction,native_norm,noise,directory,name):
    directory=Path(directory)/name;directory.mkdir(parents=True,exist_ok=False)
    norm=float(direction.norm())
    if not math.isfinite(norm):raise FloatingPointError('NONFINITE_FD_DIRECTION')
    if norm==0:
        return write(directory/'result.json',dict(status='NO_DIRECTION',direction_norm=0,derivative_pass=False))
    v=direction.double()/norm;AD=float((gradient*v).sum());h0=.01*native_norm
    save_tensor(directory/'direction.pt',dict(direction=v,AD=AD,WN_sha=tensor_sha(WN)))
    points=[]
    for k in range(12):
        h=h0/(2**k);values={};weights={};actual={}
        for sign,s in (('plus',1),('minus',-1)):
            weight=(WN.double()+s*h*v).float();delta=weight.double()-WN.double()
            spacing=(torch.nextafter(WN,torch.full_like(WN,float('inf')))-WN).double().abs()
            normalized=delta.abs()/spacing.clamp_min(torch.finfo(torch.float32).tiny)
            perturb=dict(k=k,h=h,sign=sign,weight_sha=tensor_sha(weight),actual_norm=float(delta.norm()),
                nonzero=int(torch.count_nonzero(delta)),nominal_actual_error_norm=float((delta-s*h*v).norm()),
                max_ULP=float(normalized.max()),median_nonzero_ULP=float(normalized[delta!=0].median()) if bool((delta!=0).any()) else 0,
                nominal_AD=AD,actual_linear_delta=float((gradient*delta).sum()))
            # Actual perturbation identity saved BEFORE its forward.
            write(directory/f'k{k:02d}-{sign}-perturbation.json',perturb)
            loss,_,rows=oracle.kl(weight)
            write(directory/f'k{k:02d}-{sign}-loss.json',dict(loss=loss,rows=rows))
            values[sign]=loss;weights[sign]=perturb['weight_sha'];actual[sign]=perturb
        FD=(values['plus']-values['minus'])/(2*h)
        rel=abs(FD-AD)/abs(AD) if AD!=0 else None
        signal=abs(values['plus']-values['minus'])/2
        resolved=signal>10*noise and weights['plus']!=weights['minus'] and all(x['nonzero']>0 for x in actual.values())
        point=dict(k=k,h=h,AD=AD,FD=FD,relative_error=rel,absolute_error=abs(FD-AD),noise=noise,signal=signal,
            resolved=resolved,match=bool(resolved and rel is not None and rel<=.01),loss=values,actual=actual,
            rounded_linear_FD=(actual['plus']['actual_linear_delta']-actual['minus']['actual_linear_delta'])/(2*h))
        write(directory/f'k{k:02d}-point.json',point);points.append(point)
    windows=[(i,i+1) for i in range(11) if points[i]['match'] and points[i+1]['match']]
    small=max(p['h']*abs(AD) for p in points)<=10*noise
    status='PASS' if windows else ('SMALL_AD_UNRESOLVED' if small else 'NO_RESOLVED_ADJACENT_WINDOW')
    result=write(directory/'result.json',dict(status=status,AD=AD,h0=h0,noise=noise,selected_window=windows[0] if windows else None,
        full_grid=points,old_failures_reclassified=False))
    if not windows:raise TechnicalHold(name+':'+status)
    return result


def run(lock):
    directory=Path(lock['output']);directory.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();stage='load';rt=None
    write(directory/'technical-contract.json',NUMERIC)
    if lock['numeric_contract']!=NUMERIC:raise ValueError('NUMERIC_CONTRACT_NOT_LOCKED')
    try:
        rt=Runtime(lock,directory);records=rt.records[:8]
        stage='nonselected_initial_hash';before=rt.byte_hash_nonselected()
        write(directory/'nonselected-before.json',before)
        stage='native_repeat'
        first=rt.native(records,directory/'native1');WN=first['weight'].clone()
        second=rt.native(records,directory/'native2')
        require(directory,'native-repeat',tensor_sha(WN)==tensor_sha(second['weight']) and tensor_sha(first['target'])==tensor_sha(second['target']),
            dict(endpoint1=tensor_sha(WN),endpoint2=tensor_sha(second['weight']),targets1=tensor_sha(first['target']),targets2=tensor_sha(second['target']),M0=tensor_sha(rt.M),history_appends=0))
        del second;rt.copy_weight(WN)
        stage='alltoken_cache';ref=rt.reference_oracle();cur,rows,K,meta=rt.protected_oracle(records)
        write(directory/'protected-provenance.json',meta);save_tensor(directory/'protected-keys.pt',dict(K=K))
        stage='teacher_noop'
        L0,_,teacher_rows=ref.kl(rt.W0)
        write(directory/'teacher-W0.json',dict(loss=L0,rows=teacher_rows,teacher=rt.teacher.receipt))
        require(directory,'teacher-fixed-binding',abs(L0)<=1e-6,dict(W0_signed_KL=L0,ceiling=1e-6))
        repeats=[ref.kl(WN)[0] for _ in range(3)]
        q0=score_rows(cur,WN,rows);q1=score_rows(cur,WN,rows)
        repeat_logits=[cur.compare_logits(i,WN,WN,left_route='cached',right_route='cached') for i in range(len(cur.caches))]
        nllgap=max(abs(q0[k]['nll']-q1[k]['nll']) for k in q0)
        noise=max(max(repeats)-min(repeats),np.finfo(np.float64).eps*max(1,abs(repeats[0])))
        require(directory,'noop-repeat',nllgap<=1e-5 and max(r['max_abs'] for r in repeat_logits)<=1e-4,
            dict(KL=repeats,observed_KL_noise=noise,max_NLL_difference=nllgap,logits=repeat_logits,quality0=q0,quality1=q1))
        cov_repeats=[rt.covariance(ref,WN)[0] for _ in range(3)]
        cov_noise=max(cov_repeats)-min(cov_repeats)
        write(directory/'COV-resolution.json',dict(repeat_values=cov_repeats,independent_roundoff_band=cov_noise,
            resolution=10*cov_noise,units='mean_document_mean_validtoken_half_squared_W0_activation_drift',pinned_before_M=True))
        stage='physical_gradient'
        Lphysical,Gphysical,pr=ref.kl(WN,gradient=True,route='physical');rt.sync_oracles()
        save_tensor(directory/'physical-gradient.pt',dict(gradient=Gphysical,loss=Lphysical,weight_sha=tensor_sha(WN)))
        Lcached,G,cr=ref.kl(WN,gradient=True,route='cached')
        save_tensor(directory/'cached-gradient.pt',dict(gradient=G,loss=Lcached,weight_sha=tensor_sha(WN)))
        relative=float((G-Gphysical).norm()/Gphysical.norm().clamp_min(1e-300))
        require(directory,'direct-cached-gradient',relative<=1e-4 and abs(Lphysical-Lcached)<=5e-7,
            dict(relative=relative,physical_loss=Lphysical,cached_loss=Lcached,physical_rows=pr,cached_rows=cr,
                 physical_gradient=tensor_sha(Gphysical),cached_gradient=tensor_sha(G),independent_physical_parameter_leaf=True))
        del Gphysical
        stage='physical_forward_stationarity'
        parity=[];stationarity=[]
        for i in range(len(cur.caches)):
            parity.append(cur.compare_logits(i,WN,WN));stationarity.append(cur.key_stationarity(i,WN))
        rt.sync_oracles()
        require(directory,'all-token-forward-stationarity',max(r['max_abs'] for r in parity)<=1e-4 and all(r['byte_equal'] for r in stationarity),
            dict(parity=parity,stationarity=stationarity,scope='all cold8 old/new native+canonical valid positions'))
        stage='P_range'
        with Timer(rt.timing,'P_star_recovery'):allowed=geometry.allowed_range(rt.P[0])
        save_tensor(directory/'P-star-basis.pt',dict(basis=torch.from_numpy(allowed.basis),diagnostic=allowed.diagnostic,
            P_raw_sha=geometry.matrix_sha256(rt.P[0])))
        write(directory/'P-star.json',allowed.receipt())
        stage='full_token_nullspace'
        with Timer(rt.timing,'nullspace'):space=geometry.edit_null_space(allowed,K)
        write(directory/'EN-F-space.json',space.receipt())
        pj=geometry.projector_diagnostics(space)
        if space.status!='RANK_UNRESOLVED':require(directory,'projector',pj['status']=='PASS',pj)
        else:write(directory/'projector.json',dict(status='RANK_UNRESOLVED',derivative_NOT_ESTABLISHED=True))
        stage='A_map';A=rt.factor_A(first);ca=geometry.row_space(A)
        save_tensor(directory/'A.pt',A);write(directory/'CA-space.json',ca.receipt())
        stage='FD'
        generator=torch.Generator().manual_seed(20260918)
        random=torch.randn(WN.shape,generator=generator,dtype=torch.float64)
        directions={'CA':ca.project(G),'random':random}
        if space.status=='RANK_UNRESOLVED':write(directory/'EN-F-FD-not-run.json',dict(status='RANK_UNRESOLVED',derivative_pass=False))
        else:directions['EN-F']=space.project(G)
        fd_results={}
        for name,v in directions.items():
            fd_results[name]=fd(ref,WN,G,v,float((WN.double()-rt.W0.double()).norm()),noise,directory/'FD',name)
        stage='actual_nullspace_response'
        if space.status not in ('RANK_UNRESOLVED','REPAIR_SPACE_EMPTY'):
            v=space.project(G);norm=float(v.norm())
            if norm and Lcached>1e-6:
                eta=Lcached/(norm*norm)
                if not math.isfinite(eta):raise TechnicalHold('POLYAK_STEP_SCALE_UNRESOLVED')
                ideal=-eta*v
                weight=(WN.double()+ideal).float()
                save_tensor(directory/'invariant-probe.pt',dict(ideal=ideal,actual_weight=weight,WN_sha=tensor_sha(WN)))
                inv=invariant(cur,rows,q0,weight,WN,ideal,K,allowed)
                require(directory,'actual-invariant',inv['pass'],inv)
                station=[cur.key_stationarity(i,weight) for i in range(len(cur.caches))];rt.sync_oracles()
                require(directory,'perturbed-key-stationarity',all(s['byte_equal'] for s in station),dict(rows=station))
                physical_parity=[cur.compare_logits(i,weight,weight) for i in range(len(cur.caches))];rt.sync_oracles()
                require(directory,'nonzero-physical-cached-parity',max(s['max_abs'] for s in physical_parity)<=1e-4,
                    dict(rows=physical_parity,scope='same nonzero absoluteFP32 Polyak candidate, cached versus independent physical model'))
            else:write(directory/'actual-invariant.json',dict(status='NO_DIRECTION_OR_LOSS_FLOOR',actual_nonzero_path_NOT_TESTED=True))
        else:write(directory/'actual-invariant.json',dict(status=space.status,actual_nonzero_path_NOT_TESTED=True))
        stage='restore'
        rt.guard();after=rt.byte_hash_nonselected()
        require(directory,'nonselected-exact-restore',before==after and torch.equal(rt.W.detach().cpu(),WN),
            dict(nonselected_before=digest(before),nonselected_after=digest(after),selected_expected=tensor_sha(WN),selected_actual=tensor_sha(rt.W),M0=tensor_sha(rt.M),history_appends=0))
        write(directory/'READY.json',dict(status='T_READY',scope='cold8 ENFC technical; not M science',space_status=space.status,
            FD=fd_results,EN_COV_resolution=10*cov_noise,technical_contract=digest(NUMERIC),source=lock['execution'],
            lock_identity=lock['lock_identity'],teacher=rt.teacher.receipt,identity=rt.identity,
            timing=rt.timing,oracle_work=dict(reference=ref.work,protected=cur.work),wall_seconds=time.monotonic()-start,
            peak_gpu_allocated=torch.cuda.max_memory_allocated(),peak_gpu_reserved=torch.cuda.max_memory_reserved(),
            peak_host_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,M_new_submissions=0))
    except BaseException as exc:
        write(directory/'failure.json',dict(status='TECHNICAL_HOLD' if isinstance(exc,TechnicalHold) else 'TECHNICAL_FAILURE',
            stage=stage,error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-start,
            timing={} if rt is None else rt.timing,thresholds_unchanged=True,main_started=False))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);a=p.parse_args()
    lock=json.loads(Path(a.lock).read_text())
    if lock.get('resume_prior'):
        from .technical_resume import run as continue_validated_stages
        continue_validated_stages(lock)
    else:run(lock)

if __name__=='__main__':main()
