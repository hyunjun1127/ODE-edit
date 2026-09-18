"""One GPU, fresh cold B1 only. No submission, continuation or B2 loop."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time
import traceback
import torch
import numpy as np
from .config import require_scope
from .provenance import create_json,sha
from .model import Runtime
from .choice import build_capsules,ChoiceOracle,raw_greedy
from .checks import base_binding,integrated,TechnicalHold
from .transaction import commit
from .controller import run as control
from . import qp
from project.run_scripts.single_layer_edit_preserving_correction import geometry
from project.run_scripts.single_layer_edit_preserving_correction.binding import score_rows
from project.run_scripts.single_layer_edit_preserving_correction.common import digest,tensor_sha,member
from project.run_scripts.single_layer_edit_preserving_correction.sequential_state import atomic_tensor
from project.run_scripts.single_layer_edit_preserving_correction.observer import CanonicalObserver

def seal(rt,weight,arm,selection):
    return dict(status='SELECTION_SEALED',episode_id='BPCW512-cold-B001',endpoint_id=arm,
        selection_ledger_sha256=digest(selection),endpoint_weight_sha256=tensor_sha(weight),
        request_order_sha256=digest(rt.lock['sample_order']))

def run(lock):
    require_scope(lock);out=Path(lock['output']);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();stage='preflight';rt=None;timers={}
    create_json(out/'execution-entry.json',dict(max_batches=1,sequential_authorized=False,lock=lock))
    try:
        for item in lock['execution']['members']:
            p=Path(item['path'])
            if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('FROZEN_SOURCE:'+str(p))
        if lock['qp_policy']!=asdict(qp.DEFAULT_QP_POLICY):raise ValueError('QP_POLICY_DRIFT')
        stage='load';rt=Runtime(lock,out)
        nonselected_before=rt.byte_hash_nonselected();create_json(out/'nonselected-before.json',nonselected_before)
        inputs_path=Path(lock['reference_inputs']['path'])
        if sha(inputs_path)!=lock['reference_inputs']['sha256']:raise ValueError('REFERENCE_INPUT_HASH')
        inputs=json.loads(inputs_path.read_text())
        stage='W0-answer-capsules';t=time.monotonic();capsules=build_capsules(rt,inputs,out/'capsules')
        timers['answer_capsule_setup']=time.monotonic()-t
        capsule_manifest=dict(W0=rt.identity['W0'],inputs=lock['reference_inputs'],count=len(capsules),
            max_new_tokens=16,raw_argmax=True,corpus_revision=lock['corpus_revision'],model_revision=lock['model_revision'],
            train=[member(p) for p in sorted((out/'capsules/R512').glob('*.json'))],
            dev=[member(p) for p in sorted((out/'capsules/Dev128').glob('*.json'))])
        create_json(out/'capsule-manifest.json',capsule_manifest)
        stage='fresh-matched-native';t=time.monotonic();native=rt.native(rt.records,out/'native',reuse=False)
        WN=native['weight'];timers['native_shared_once']=time.monotonic()-t
        stage='reference-caches';t=time.monotonic()
        ref=ChoiceOracle(rt,[r for r in capsules if r['role']=='R512'])
        dev=ChoiceOracle(rt,[r for r in capsules if r['role']=='Dev128'])
        timers['fixed_reference_key_cache_setup']=time.monotonic()-t
        stage='base-capsule-binding';t=time.monotonic()
        tau,w0reference=base_binding(ref,rt.W0,out/'technical/base-R512')
        _,w0dev=base_binding(dev,rt.W0,out/'technical/base-Dev128')
        timers['base_capsule_validation']=time.monotonic()-t
        stage='current-K-Q';t=time.monotonic()
        cur,rows,K,provenance=rt.protected_oracle(rt.records)
        create_json(out/'protected-provenance.json',provenance)
        atomic_tensor(out/'protected-keys.pt',dict(K=K))
        ps=lock['P_star_basis']
        if sha(ps['path'])!=ps['sha256']:raise ValueError('P_STAR_BASIS_SHA')
        saved=torch.load(ps['path'],map_location='cpu',weights_only=True,mmap=True)
        basis=saved['basis'].numpy()
        allowed=geometry.allowed_range(rt.P[0],provenance_basis=basis,provenance=dict(
            raw_sha256=saved['P_raw_sha'],basis_sha256=geometry.matrix_sha256(basis),allowed_rank=basis.shape[1]))
        space=geometry.edit_null_space(allowed,K)
        atomic_tensor(out/'Q-factors.pt',dict(basis=torch.from_numpy(space.basis),blocked=torch.from_numpy(space.blocked)))
        create_json(out/'allowed-range.json',allowed.receipt());create_json(out/'edit-null-space.json',space.receipt())
        timers['current_keys_geometry']=time.monotonic()-t
        stage='integrated-actual-checks'
        technical=integrated(rt,ref,cur,rows,K,WN,space,allowed,out/'technical')
        timers['integrated_technical']=technical['seconds']
        stage='current-native-anchor';t=time.monotonic();anchor=score_rows(cur,WN,rows)
        create_json(out/'current-native-anchor.json',anchor);timers['current_anchor']=time.monotonic()-t
        stage='BPCW-controller';selected,ideal,selection,selected_scan=control(rt,WN,ref,space,allowed,cur,rows,K,anchor,tau,out/'controller')
        timers['controller']=selection['seconds']
        atomic_tensor(out/'selected-ideal.pt',dict(ideal=ideal,native=tensor_sha(WN),selected=tensor_sha(selected)))
        # Both choices are sealed before any official P/N or Dev endpoint result.
        create_json(out/'endpoint-seals.json',dict(N4=seal(rt,WN,'N4',dict(status='NATIVE_ENDPOINT')),BPCW512=seal(rt,selected,'BPCW512',selection)))
        stage='history-checkpoint';commits={}
        for arm,w,choice in (('N4',WN,dict(status='NATIVE_ENDPOINT')),('BPCW512',selected,selection)):
            commits[arm]=commit(rt,arm,w,choice,member(out/'capsule-manifest.json'),out/'arms'/arm)
            # Check restored GPU endpoint response against original saved-weight route.
            parity=ref.compare_logits(0,w,rt.W.detach(),left_route='cached',right_route='physical');rt.sync_oracles()
            if parity['max_abs']>1e-4:raise TechnicalHold('CHECKPOINT_PHYSICAL_RELOAD_PARITY')
            create_json(out/'arms'/arm/'reload-forward.json',parity)
        stage='postseal-observers';t=time.monotonic()
        observer=CanonicalObserver(rt.model,rt.etok,runtime_identity=digest(rt.identity))
        w0=observer.observe(rt.records,rt.W0,selection_seal=seal(rt,rt.W0,'W0',dict(status='PRETRAINED_FIXED')),greedy=False);rt.sync_oracles()
        create_json(out/'W0-current.json',w0);observed={}
        for arm,w,choice in (('N4',WN,dict(status='NATIVE_ENDPOINT')),('BPCW512',selected,selection)):
            value=observer.observe(rt.records,w,selection_seal=seal(rt,w,arm,choice),w0_result=w0,greedy=True);rt.sync_oracles()
            create_json(out/'arms'/arm/'current.json',value);observed[arm]=value
            dv=dev.scan(w,tau);create_json(out/'arms'/arm/'Dev128-choice.json',dv)
            # Fixed8 fresh greedy is observer only; TF bank is the scientific constraint.
            fixed=set(lock['fixed8']);spots=[];rt.copy_weight(w)
            for cap in ref.capsules:
                if cap['source_row_id'] in fixed:
                    spots.append(dict(source_row_id=cap['source_row_id'],base_y0=cap['y0'],
                        current=raw_greedy(rt.model,cap['input_ids'])))
            create_json(out/'arms'/arm/'fixed8-greedy.json',spots)
        timers['official_dev_generation_observers']=time.monotonic()-t
        stage='QP-independent-order-audit';t=time.monotonic()
        audit=[]
        for p in sorted((out/'controller').glob('round*/qp-problem.pt')):
            problem=torch.load(p,weights_only=True,mmap=True,map_location='cpu')
            try:
                result=qp.audit_same_problem(problem['G'].numpy(),problem['b'].numpy(),problem['ids'])
            except qp.QPInfeasible:
                outcomes={}
                for ordering in ('full','gss','most_violation'):
                    try:qp.solve(problem['G'].numpy(),problem['b'].numpy(),problem['ids'],order=ordering)
                    except qp.QPInfeasible as exc:outcomes[ordering]=exc.receipt
                if len(outcomes)!=3:raise TechnicalHold('QP_INFEASIBLE_ORDERING_DISAGREEMENT')
                result=dict(status='SAME_LOCAL_INFEASIBILITY_NUMERICAL_CERTIFICATES',pass_all=True,outcomes=outcomes)
            if result.get('pass') is False:raise TechnicalHold('QP_ORDER_AUDIT_MISMATCH')
            create_json(p.parent/'ordering-audit.json',result);audit.append(dict(path=str(p),audit=result))
        timers['CPU_QP_order_audit']=time.monotonic()-t
        if not audit:create_json(out/'QP-ordering-audit.json',dict(status='NOT_APPLICABLE_NO_LOCAL_QP'))
        stage='final-state';rt.guard();after=rt.byte_hash_nonselected()
        if after!=nonselected_before:raise TechnicalHold('NONSELECTED_PARAMETER_BYTES_CHANGED')
        create_json(out/'nonselected-after.json',after)
        result=dict(status='B1_ENDPOINTS_COMPLETE_REVIEW_REQUIRED',batch=1,requests=100,arms=['N4','BPCW512'],
            source=lock['execution'],lock_identity=lock['lock_identity'],identity=rt.identity,
            selection=selection,commits=commits,technical=technical,space_status=space.status,
            timings=timers,native_counts=native['receipt'],native_actual_executions=1,
            reference_work=ref.work,pair_work=ref.pair_work,dev_work=dev.work,current_work=cur.work,observer_work=observer.work,
            wall_seconds=time.monotonic()-started,peak_gpu_allocated=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved=torch.cuda.max_memory_reserved(),peak_host_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            max_batches=1,sequential_authorized=False,next='WAITING_USER_APPROVAL_FOR_SEQUENTIAL',
            scientific_gate='CPU_REDUCER_PENDING',automatic_resume=False,nonselected_bytes_equal=True)
        create_json(out/'terminal.json',result);print('B1_COMPLETE_NO_SEQUENTIAL',flush=True)
    except BaseException as exc:
        create_json(out/'failure.json',dict(status='TECHNICAL_HOLD' if isinstance(exc,TechnicalHold) else 'TECHNICAL_FAILURE',
            stage=stage,error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-started,
            timers=timers,native_fit_count=None if rt is None else rt.native_calls,
            sequential_authorized=False,partial_artifacts_preserved=True))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--max-batches',type=int,required=True)
    a=p.parse_args();lock=json.loads(Path(a.lock).read_text())
    if a.max_batches!=1:raise ValueError('B1_ONLY_NO_SEQUENTIAL_AUTHORITY')
    run(lock)
