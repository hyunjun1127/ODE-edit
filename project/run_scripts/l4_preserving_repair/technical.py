"""Bounded actual first100 pilot, fresh main admission only after READY.

No saved W/M/RNG checkpoint. All anchor/rollback tensors are in process memory;
early gradient/probe evidence remains local, not a continuation checkpoint.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import resource
import time
import traceback
import numpy as np
import torch
from .common import ROOT,save,tensor_save,identity,verify,Timer
from .numerical import POLICY

def repeat(a,b):
    for key in ('current','past'):
        x,y=a[key],b[key]
        if x is None or y is None:assert x is y;continue
        assert abs(x['E']-y['E'])<=POLICY['repeat_E_H'],('REPEAT_NLL',key)
        assert x['strict_ids']==y['strict_ids'] and x['preference_ids']==y['preference_ids'],('REPEAT_IDS',key)
    assert abs(a['base']['B']-b['base']['B'])<=POLICY['repeat_B'],'REPEAT_BASE'

def fd_checks(rt,cur,model,out):
    from .runtime import tensor_sha
    from .engine import jsonable
    rows=[]
    for objective,key in [('B','B'),('R','E')]:
        gradient=model['gradients'][objective];norm=float(gradient.double().norm())
        assert norm>1e-12,('FD_UNRESOLVED_ZERO_GRADIENT',objective)
        direction=(gradient.double()/norm).float()
        ad=float((gradient.double()*direction.double()).sum())
        eval_fn=(lambda:rt.response.base()[0]) if objective=='B' else (lambda:rt.response.panel(cur)[0])
        original=model['anchor']['base' if objective=='B' else 'current'][key]
        zero1=rt.observe(eval_fn);zero2=rt.observe(eval_fn)
        jitter=abs(zero1[key]-zero2[key])
        save(out/f'{objective}-zero-repeat.json',dict(first=zero1,second=zero2,jitter=jitter,original=original))
        directions=tensor_save(out/f'{objective}-FD-direction.pt',dict(direction=direction,
            source_gradient_sha=tensor_sha(gradient),norm=norm,AD=ad))
        points=[]
        for index,relative in enumerate(POLICY['FD_relative_weight_steps']):
            h=relative*max(float(model['anchor8'].double().norm()),1.)
            values=[]
            for sign in (1,-1):
                candidate=model['anchor8']+float(sign*h)*direction
                rt.apply8(candidate,model['rng'])
                result=rt.observe(eval_fn)
                actual_delta=candidate.double()-model['anchor8'].double()
                record=dict(objective=objective,index=index,sign=sign,h=h,result=result,
                    nominal_action_norm=h*float(direction.double().norm()),actual_action_norm=float(actual_delta.norm()),
                    rounding_error_norm=float((actual_delta-sign*h*direction.double()).norm()),
                    nonzero_elements=int(torch.count_nonzero(actual_delta)),weight_sha=tensor_sha(candidate),
                    direction=directions,W4_sha=tensor_sha(rt.W[4]))
                save(out/f'{objective}-scale{index}-sign{sign}.json',record)
                values.append(result[key]);rt.apply8(model['anchor8'],model['rng'])
                assert rt.state()==model['state'],'FD_RESTORE'
            derivative=(values[0]-values[1])/(2*h)
            signal=abs(values[0]-values[1]);noise=max(jitter,abs(zero1[key]-original),abs(zero2[key]-original))
            error=abs(derivative-ad);limit=POLICY['FD_absolute_tolerance']+POLICY['FD_relative_tolerance']*abs(ad)
            row=dict(objective=objective,h=h,AD=ad,FD=derivative,absolute_error=error,tolerance=limit,
                signal=signal,jitter=jitter,noise_observed=noise,resolved=signal>POLICY['FD_noise_multiplier']*noise,
                plus=values[0],minus=values[1],E0=original,
                taylor_plus=values[0]-original-h*ad,taylor_minus=values[1]-original+h*ad)
            save(out/f'{objective}-scale{index}-result.json',row);points.append(row);rows.append(row)
        stable=abs(points[0]['FD']-points[1]['FD'])<=POLICY['FD_absolute_tolerance']+POLICY['FD_stability_relative']*abs(ad)
        save(out/f'{objective}-verdict.json',dict(points=points,stable=stable,
            status='PASS' if stable and all(x['resolved'] and x['absolute_error']<=x['tolerance'] for x in points) else 'UNRESOLVED'))
        assert stable and all(x['resolved'] and x['absolute_error']<=x['tolerance'] for x in points),('SIGNED_FD_UNRESOLVED',objective)
    return rows

def run(lock_path):
    from .runtime import Runtime,restore_rng,tensor_sha
    from .engine import build,select,score,materialize,jsonable
    from .runner import native_evidence
    from .geometry import whiten_gn
    lock=json.loads(Path(lock_path).read_text());verify(lock)
    out=Path(lock['technical_output']);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();stage='LOAD';rt=None;entry=None;timing={}
    try:
        common=json.loads(Path(lock['cold_capsule']['path']).read_text());rt=Runtime(lock,common)
        entry=rt.snapshot();cur=rt.records[:100]
        save(out/'start.json',dict(lock=identity(lock_path),W0=rt.state(),cold_capsule=lock['cold_capsule'],
            checkpoint='SKIPPED_USER_DIRECTED',teacher=rt.teacher.receipt,scientific_commits=0))
        stage='FIXED_W0_TEACHER'
        teacher=rt.observe(lambda:rt.response.base()[0]);save(out/'teacher-effective-check.json',teacher)
        assert abs(teacher['B'])<=POLICY['repeat_B'],'TEACHER_W0_REPRODUCTION_REQUIRES_SCOPED_REPAIR'
        stage='NATIVE_L4_ONCE'
        fit=rt.fit(cur);tensor_save(out/'native-targets.pt',native_evidence(fit));del fit
        restore_rng(entry['rng']);native_state=rt.state()
        assert native_state['M4']==tensor_sha(entry['M4']) and native_state['W']['8']==tensor_sha(entry['W'][8])
        stage='EARLY_GRADIENT_AND_RESPONSE'
        model=build(rt,cur,[],'R-QP',out,pilot=True)
        assert len(model['Q'])<=2
        if model['response'] is None:
            save(out/'coverage-hold.json',dict(status='TECHNICAL_UNRESOLVED',reason='NO_RESOLVED_DIRECTION_FOR_REQUIRED_MODEL_CHECKS',
                normal_repair_off=True,derivative_GN_materialization='NOT_RUN',scientific_quality_failure=False))
            raise RuntimeError('TECHNICAL_COVERAGE_UNRESOLVED_NOT_READY')
        stage='REPEATED_ANCHOR'
        repeated=score(rt,cur,[]);save(out/'anchor-repeat.json',repeated);repeat(model['anchor'],repeated)
        # Nonempty H repeat uses only arrived first100, not a scientific B1 Past.
        h1=rt.observe(lambda:rt.response.panel(cur[:4])[0]);h2=rt.observe(lambda:rt.response.panel(cur[:4])[0])
        save(out/'technical-H-repeat.json',dict(first=h1,second=h2,scientific_B1_Past='EMPTY'))
        assert abs(h1['E']-h2['E'])<=POLICY['repeat_E_H'] and h1['strict_ids']==h2['strict_ids'] and h1['preference_ids']==h2['preference_ids']
        if model['response'] is not None:
            stage='JVP_GN_CONNECTION'
            response=model['response'];ad=model['dots'];forward=response['b'].numpy()
            differences=np.abs(forward-ad);limits=POLICY['JVP_projection_absolute']+POLICY['JVP_projection_relative']*np.abs(ad)
            gR=model['gradients']['R']
            directR=np.array([float((gR.double()*q.double()).sum()) for q in model['Q']])
            jvpR=response['A'][0].numpy()
            rc=dict(Base_direct=ad.tolist(),Base_JVP=forward.tolist(),Base_error=differences.tolist(),
                Current_direct=directR.tolist(),Current_JVP=jvpR.tolist(),Current_error=np.abs(directR-jvpR).tolist())
            save(out/'JVP-gradient-connection.json',rc)
            assert np.all(differences<=limits),'BASE_JVP_REVERSE_CONNECTION'
            assert np.all(np.abs(directR-jvpR)<=POLICY['JVP_projection_absolute']+POLICY['JVP_projection_relative']*np.abs(directR)),'CURRENT_JVP_REVERSE_CONNECTION'
            white=whiten_gn(response['H'],model['dots']);save(out/'GN-whitening.json',white)
            stage='SIGNED_FD_TWO_SCALES';fd_checks(rt,cur,model,out)
            stage='ALL_TOKEN_FUNCTIONAL_MATERIALIZATION'
            c=[POLICY['FD_relative_weight_steps'][1]*max(float(model['anchor8'].double().norm()),1.)]+[0.]*(len(model['Q'])-1)
            candidate=materialize(model['anchor8'],model['Q'],c)
            functional=dict(current=rt.observe(lambda:rt.response.panel(cur,weight=candidate)[0]),past=None,
                base=rt.observe(lambda:rt.response.base(weight=candidate)[0]))
            save(out/'functional-candidate.json',functional)
            rt.apply8(candidate,model['rng']);physical=score(rt,cur,[]);save(out/'physical-candidate.json',physical)
            repeat(functional,physical)
            assert tensor_sha(rt.W[4])==model['state']['W']['4'],'PILOT_W4_MUTATION'
            rt.apply8(model['anchor8'],model['rng']);assert rt.state()==model['state']
        stage='QP_FIRST_ACCEPTABLE'
        selection,selected8=select(rt,cur,[],model,out)
        # Reuse this same WN / gB / Q0 / H00 for the scalar R-GD path. It has no
        # proposal guards but still executes the actual rewrite acceptance.
        if model['response'] is not None and model['Q']:
            stage='R_GD_SHARED_ANCHOR_FIRST_ACCEPTABLE'
            rt.apply8(model['anchor8'],model['rng'])
            gd_model=dict(model,arm='R-GD',Q=model['Q'][:1],dots=model['dots'][:1],
                response=dict(H=model['response']['H'][:1,:1],A=[],s=[]))
            gd_out=out/'R-GD-shared-anchor';gd_out.mkdir()
            gd_selection,_=select(rt,cur,[],gd_model,gd_out)
            save(out/'R-GD-pilot-link.json',dict(native_fit_reused=True,
                anchor_state=model['state'],new_native_target_calls=0,scientific_commits=0,
                selection=identity(gd_out/'selection.json'),shared_technical_gradients=True,
                actual_scientific_R_GD_current_past_gradient_sweeps=0))
            rt.apply8(selected8,model['rng'])
        stage='HISTORY_ONCE_ROLLBACK'
        assert rt.state()['M4']==native_state['M4'],'INNER_HISTORY_MUTATION'
        history=rt.finalize(cur);committed=rt.snapshot();after=rt.state()
        assert tensor_sha(rt.W[4])==native_state['W']['4']
        rt.restore(entry);rt.restore(committed);assert rt.state()==after
        save(out/'history-restore.json',dict(history=history,M4_append=1,M8_append=0,inner_append=0,
            selected_state=after,same_process_restore='PASS',GPU_off_on='NOT_TESTED',disk_checkpoint='SKIPPED_USER_DIRECTED',
            scientific_commits=0))
        rt.restore(entry);assert all(torch.equal(rt.W[l].detach().cpu(),entry['W'][l]) for l in (4,8))
        elapsed=time.monotonic()-start
        save(Path(lock['common_ready']),dict(status='TECHNICAL_READY',execution_lock_sha256=identity(lock_path)['sha256'],
            native_target_calls=100,native_solve=1,L8_target=0,scientific_commits=0,
            capsule=lock['cold_capsule'],checks=[identity(p) for p in sorted(out.glob('*.json'))],
            seconds=elapsed,timers=rt.timing,peak_GPU_allocated=torch.cuda.max_memory_allocated(),
            peak_GPU_reserved=torch.cuda.max_memory_reserved(),peak_host_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            measured_storage_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
            checkpoint='SKIPPED_USER_DIRECTED',main_fresh_W0_required=True,monitoring_callback=False))
    except BaseException as e:
        restored=False
        if rt is not None and entry is not None:
            try:rt.restore(entry);restored=True
            except BaseException:pass
        save(out/'failure.json',dict(status='TECHNICAL_FAILURE',stage=stage,error=repr(e),traceback=traceback.format_exc(),
            exception_receipt=jsonable(getattr(e,'receipt',None)),seconds=time.monotonic()-start,
            rollback_to_W0=restored,scientific_commits=0,science_ready=False))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);a=p.parse_args();run(a.lock)
