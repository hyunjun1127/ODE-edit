"""Task-owned complete batch computation. Stage authority lives in program.py.

All candidate policies finish before this module exposes official observers.
One native fit per actual entry; no correction-specific target fitting.
USER no-checkpoint: full W/M, correction vectors, dense gradients and basis
directions stay in RAM. Persist only scientific input/factor evidence, small
coefficient Jacobians and tensor identity/norm receipts, not restart state.
"""
import copy
import gc
from pathlib import Path
import time
import numpy as np
import torch
from .basis import build_functional_basis, append_covariance_direction, BasisResult
from .current import CurrentGuard
from .decision import DecisionOracle, EndpointBinding, FactorArchive
from .history import active_history, StreamingHistoryOracle
from .transaction import commit
from project.run_scripts.single_layer_edit_preserving_correction.common import write,save_tensor,tensor_sha,digest,member
from project.run_scripts.single_layer_edit_preserving_correction.geometry import RightSpace,edit_null_space
from project.run_scripts.single_layer_edit_preserving_correction.runner import ideal_check
from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng


def bound(rt,reference,weight,name):
    return EndpointBinding(weight,reference.device,endpoint_id=name,source_identity=rt.identity)


def scan(rt,reference,decision,weight,name,*,gradient=False,factors=None):
    endpoint=bound(rt,reference,weight,name)
    try:return decision.scan(endpoint,gradient=gradient,factor_sink=factors)
    finally:endpoint.close()


def allowed_space(rt):
    from project.run_scripts.single_layer_edit_preserving_correction.common import sha
    item=rt.lock['P_star_basis'];p=Path(item['path'])
    s=p.stat();fingerprint=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,item['sha256'])
    cached=getattr(rt,'_slmf_allowed_space',None)
    if cached is not None:
        if cached[0]!=fingerprint:raise ValueError('P_STAR_FIXED_INPUT_CHANGED')
        return cached[1]
    if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('P_STAR_INPUT_IDENTITY')
    data=torch.load(p,weights_only=True,mmap=True,map_location='cpu')
    v=data['basis'].numpy()
    value=RightSpace(v,np.empty((v.shape[1],0),dtype=np.float64),'RESOLVED' if v.shape[1] else 'REPAIR_SPACE_EMPTY',data['diagnostic'])
    rt._slmf_allowed_space=(fingerprint,value)
    return value


def history_evaluate(rt,reference,history,weight,name,*,anchor=None,gradient=False,factors=None):
    endpoint=bound(rt,reference,weight,name)
    try:return history.evaluate(endpoint,entry_anchor=anchor,gradient=gradient,factor_sink=factors)
    finally:endpoint.close()


def _tensor_summary(value):
    """Compact byte identity only; never serialize the dense input tensor."""
    if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.ndim!=2:
        raise ValueError('CPU_DIAGNOSTIC_MATRIX_REQUIRED')
    if not torch.isfinite(value).all():raise FloatingPointError('NONFINITE_DIAGNOSTIC_MATRIX')
    return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=tensor_sha(value),
                norm=float(value.double().norm()),bytes=value.numel()*value.element_size(),
                persisted=False)


def batch(rt,reference,records,*,stage,arm_names,batch_number,ledger,directory,gate=None,shadow=False,
          history_cache_directory=None):
    """B1 shared four policies, otherwise one chain (B2 N4 adds shadow pair)."""
    from .controller import native_result,optimize_kl,optimize_decision
    from .history_runtime import HistoryFactory
    from .config import B1_ARMS,CHAIN_ARMS,require_stage
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    if len(records)!=100:raise ValueError('EXACT_B100_REQUIRED')
    if stage=='B1' and tuple(arm_names)!=B1_ARMS:raise ValueError('B1_FOUR_UNIQUE_ARMS')
    if stage!='B1' and (len(arm_names)!=1 or arm_names[0] not in CHAIN_ARMS):raise ValueError('ONE_OWN_ENTRY_CHAIN')
    for arm in arm_names:require_stage(stage,arm,batch_number,gate)
    started=time.monotonic();times={};entry=rt.W.detach().cpu().clone();entryM=rt.M.clone()
    entry_rng=digest(capture_rng());entry_identity=dict(W=tensor_sha(entry),M=tensor_sha(entryM),ledger=digest(ledger),rng=entry_rng)
    write(directory/'batch-entry.json',dict(stage=stage,arms=arm_names,batch=batch_number,identity=entry_identity,
        case_ids=[r['case_id'] for r in records],prior_gate=gate,fresh_own_entry=True))
    # Only L4 down_proj is edited: its input/pre-MLP residual is fixed. The
    # create-once store is source/input bound, not batch/selected-weight bound.
    # Keep it stable across batches; an explicit chain-local path is supported.
    history_cache_directory=Path(history_cache_directory) if history_cache_directory is not None else Path(rt.output)/'history-prefix'
    active=active_history(ledger,records);factory=HistoryFactory(rt,active,history_cache_directory)
    history=StreamingHistoryOracle(active,factory)
    t=time.monotonic();entryhist=history_evaluate(rt,reference,history,entry,'entry');times['entry_history']=time.monotonic()-t
    write(directory/'history-entry.json',entryhist.compact())
    t=time.monotonic();native=rt.native(records,directory/'native');WN=native['weight'];times['native']=time.monotonic()-t
    decision=DecisionOracle(reference)
    need_decision=any(a!='N4' for a in arm_names) or shadow
    nativehist=None;reference_native=None;results={};selections={};weights={};commits={}
    if need_decision:
        t=time.monotonic();current,rows,K,meta=rt.protected_oracle(records);allowed=allowed_space(rt)
        space=edit_null_space(allowed,K);times['geometry_current_prefix']=time.monotonic()-t
        write(directory/'geometry/space.json',space.receipt());write(directory/'geometry/keys-provenance.json',meta)
        save_tensor(directory/'geometry/keys.pt',K)
        save_tensor(directory/'geometry/blocked.pt',dict(blocked=torch.from_numpy(space.blocked),allowed=rt.lock['P_star_basis']))
        current_guard=CurrentGuard(rt,WN,current,rows,K,allowed,meta)
        native_binding=bound(rt,reference,WN,'native-center')
        try:
            sink=FactorArchive(directory/'reference-factors',dict(endpoint=native_binding.identity,input=decision.input_identity))
            t=time.monotonic();reference_native=decision.scan(native_binding,gradient=True,factor_sink=sink)
            times['decision_reference_derivative']=time.monotonic()-t
        finally:native_binding.close()
        write(directory/'reference-native.json',reference_native.compact())
        hf=FactorArchive(directory/'history-factors',dict(endpoint=tensor_sha(WN),history=history.history_identity)) if active else None
        nativehist=history_evaluate(rt,reference,history,WN,'history-native',anchor=entryhist.anchor(),gradient=True,factors=hf)
        write(directory/'history-native.json',nativehist.compact())
        totalG=reference_native.gradient.clone()
        if nativehist.gradient is not None:totalG+=nativehist.gradient
        t=time.monotonic();GQ=space.project(totalG) if space.status!='RANK_UNRESOLVED' else torch.zeros_like(totalG)
        functional=build_functional_basis(-GQ,project=space.project) if space.status!='RANK_UNRESOLVED' else BasisResult((),dict(shape=list(WN.shape),status='RANK_UNRESOLVED',rank=0))
        times['projection_functional_basis']=time.monotonic()-t
        write(directory/'gradient-identities.json',dict(decision_gradient=_tensor_summary(totalG),
            projected_gradient=_tensor_summary(GQ),dense_gradient_storage='SKIPPED_USER_DIRECTED',
            exact_state_restart='NOT_AVAILABLE'))
        write(directory/'basis-functional.json',functional.receipt)

        def proposal(d):return ideal_check(d,K)
        def quality(w,d,trial):return current_guard.candidate(w,d,trial)
        def past(w,trial):
            observation=history_evaluate(rt,reference,history,w,f'history-trial-{trial}',anchor=entryhist.anchor())
            return observation
        def evaluate(w,trial):
            observation=scan(rt,reference,decision,w,f'candidate-{trial}')
            return observation
        def objective(w,gradient=False):
            value=reference.kl(w,gradient=gradient,role='R512')
            if not np.isfinite(value[0]) or (gradient and not torch.isfinite(value[1]).all()):raise FloatingPointError('KL_NONFINITE')
            return value
        def event_for(arm):
            count=0
            def event(row,*payload):
                nonlocal count
                write(directory/'arms'/arm/'events'/f'{count:04d}.json',row);count+=1
            return event
        run_names=list(arm_names)
        if shadow:run_names+=['DEC_MODES_STEP','DEC_MODES_CUM']
        try:
            for arm in run_names:
                t=time.monotonic()
                if arm=='N4':result=native_result(WN,native_history=nativehist)
                elif arm=='EN_KL_Q':
                    result=optimize_kl(WN,objective=objective,project=space.project,current_check=quality,
                        proposal_check=proposal,history_check=past,native_history=nativehist,
                        objective_id=decision.input_identity,event=event_for(arm),space_status=space.status)
                else:
                    if arm=='DEC_LINE':
                        n=float(GQ.norm());basis=BasisResult(((-GQ/n).numpy(),) if n else (),dict(shape=list(WN.shape),rank=int(n>0)))
                    elif functional.rank:
                        displacement=WN.double()-(entry if arm=='DEC_MODES_STEP' else rt.W0).double()
                        basis=append_covariance_direction(functional,displacement,
                            (reference.caches[i].valid_keys() for i in range(512)),project=space.project)
                    else:basis=functional
                    directions=[torch.from_numpy(d) for d in basis.directions]
                    J=decision.jacobian(reference_native,directions) if directions else torch.empty((512,0),dtype=torch.float64)
                    JH=history.jacobian(nativehist,directions) if directions else torch.empty((len(active),0),dtype=torch.float64)
                    write(directory/'arms'/arm/'basis.json',dict(**basis.receipt,
                        direction_identities=[_tensor_summary(d) for d in directions],
                        direction_storage='RAM_ONLY_USER_NO_CHECKPOINT',exact_state_restart='NOT_AVAILABLE'))
                    save_tensor(directory/'arms'/arm/'jacobians.pt',dict(J=J,JH=JH))
                    result=optimize_decision(arm,WN,native_reference=reference_native,raw_gradient=totalG,
                        projected_gradient=GQ,basis=basis,reference_J=J,evaluate_reference=evaluate,
                        current_check=quality,proposal_check=proposal,native_history=nativehist,
                        history_J=JH,history_check=past,event=event_for(arm),space_status=space.status)
                times[arm+'_controller_inclusive']=time.monotonic()-t
                results[arm]=result;weights[arm]=result.weight
                selections[arm]=write(directory/'arms'/arm/'selection-ledger.json',result.receipt())
        finally:
            current_guard.close();rt.oracles.remove(current);del current,rows,K,allowed,space,current_guard
            gc.collect()
    else:
        reference_native=scan(rt,reference,decision,WN,'native-N4')
        write(directory/'reference-native.json',reference_native.compact())
        nativehist=history_evaluate(rt,reference,history,WN,'history-native',anchor=entryhist.anchor())
        write(directory/'history-native.json',nativehist.compact())
        results['N4']=native_result(WN,native_history=nativehist);weights['N4']=WN
        selections['N4']=write(directory/'arms/N4/selection-ledger.json',results['N4'].receipt())
    # No candidate may access the observers below. Every selection is immutable.
    seal_all=write(directory/'SELECTIONS_SEALED.json',dict(selections=selections,shadow=shadow,
        official_P_N_access_so_far=0,next_state_chosen=True,controller_immutable=True))
    for arm in arm_names:
        commits[arm]=commit(rt,arm,batch_number,records,weights[arm],entryM,ledger,
            selections[arm],directory/'commits'/arm,stage=stage,gate=gate)
    write(directory/'history-prefix-receipt.json',factory.receipt)
    write(directory/'batch-compute.json',dict(timing=times,total_inclusive=time.monotonic()-started,
        nested_timers_do_not_sum=True,native_fit_calls=1,native_target_calls=100,
        histories_appended=len(arm_names),shadow_has_history_append=False if shadow else None,
        shared_derivative_for_B1_or_same_entry_only=True,
        dense_gradient_and_basis_storage='RAM_ONLY_USER_NO_CHECKPOINT',
        disk_W_M_or_full_delta_checkpoint=False,exact_crash_resume='NOT_AVAILABLE'))
    return dict(results=results,weights=weights,selections=selections,commits=commits,
        native=native,entry=entry,entryM=entryM,reference_native=reference_native,
        native_history=nativehist,decision=decision,records=records,selection_seal=seal_all,
        directory=directory,timing=times)
