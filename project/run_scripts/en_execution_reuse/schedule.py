"""Same immutable EN-F optimizer, independent gradient/trials per schedule."""
from copy import deepcopy
import time
import numpy as np
import torch
from .config import ARMS, DATA_ID
from .endpoint_session import EndpointSession, RuntimePolicy
from .current_observation import CurrentObservationController
from .preparation import create_json
from .generated_oracle import FiniteTrialModelOverflow
from project.run_scripts.single_layer_edit_preserving_correction.alltoken import model_guard, tensor_sha256
from project.run_scripts.single_layer_edit_preserving_correction.binding import score_rows, quality_ok
from project.run_scripts.single_layer_edit_preserving_correction.runtime import invariant
from project.run_scripts.single_layer_edit_preserving_correction.optimizer import optimize, Check, TrialNumericalOverflow
from project.run_scripts.single_layer_edit_preserving_correction.runner import EventStore, ideal_check
from project.run_scripts.single_layer_edit_preserving_correction.common import digest
from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng, restore_rng


def _difference(before, after):
    return {k:v-before.get(k,0) for k,v in after.items() if isinstance(v,(int,float))}


def recheck_fixed_teacher_bank(store):
    """Only a finite model trial may backtrack; corrupt fixed input never may."""
    started=time.monotonic();indices=store.indices('R512');positions=0
    if len(indices)!=512 or len(set(indices))!=512:raise ValueError('OVERFLOW_FIXED_BANK_CARDINALITY')
    if not getattr(store,'verify_payloads',True):
        return dict(documents=512,positions=store.receipt['position_counts']['R512'],
                    complete=False,status='SKIPPED_USER_DIRECTED',model_calls=0,
                    seconds=time.monotonic()-started)
    for index in indices:
        # document() independently checks immutable bytes and schema on entry
        # and exit. No model/gradient work or shortened bank is introduced.
        with store.document(index) as document:
            if not bool(np.isfinite(document.logp).all()):raise FloatingPointError('CORRUPT_FIXED_TEACHER')
            positions+=document.logp.shape[0]
    return dict(documents=512,positions=positions,complete=True,seconds=time.monotonic()-started,
                model_calls=0,teacher_manifest_sha256=store.receipt['manifest_sha256'])


def exact_receipt(rt, result, sweeps, geometry_id, input_id, teacher, epoch):
    gradients=[e for e in result.events if e['event']=='gradient_observed']
    direction=next((e for e in result.events if e['event']=='direction'),{})
    g=gradients[0] if gradients else {}
    gradient_sweeps=[s for s in sweeps if s['gradient']]
    coverage=gradient_sweeps[0]['coverage'] if gradient_sweeps else None
    rows=[]
    for trial in result.trials:
        key=(trial['round'],trial['trial'])
        def event(name):
            value=next((e for e in result.events if e['event']==name and (e.get('round'),e.get('trial'))==key),None)
            return None if value is None else {k:v for k,v in value.items() if k not in ('event_index','arm','event','round','trial')}
        rows.append(dict(trial=trial['trial'],weight_sha256=trial['weight_sha256'],loss=trial.get('loss'),
            actual_p=trial.get('p_actual'),armijo=None if 'loss' not in trial else trial['loss']<=trial['armijo_bound'],
            guard=event('quality_guard_checked'),invariant=event('actual_invariant_checked'),
            decision=dict(accepted=trial['accepted'],reason=trial.get('reason')),
            full_trial=deepcopy(trial)))
    docs=rt.generated_store.indices('R512')
    return dict(data_id=DATA_ID,input_sha256=rt.lock['reference_inputs']['sha256'],
        native_sha256=result.events[0]['native_weight_sha256'],geometry_sha256=geometry_id,
        current_binding_sha256=input_id,teacher_sha256=teacher,
        model_epoch=digest(dict(runtime=rt.identity,native=result.events[0]['native_weight_sha256'])),
        physical_guard_epoch=repr(epoch),
        loss=g.get('loss'),gradient_sha256=g.get('gradient_sha256'),
        projected_gradient_sha256=direction.get('projected_sha256'),chi=direction.get('chi'),
        eta0=result.trials[0]['eta0'] if result.trials else None,
        reference_document_ids=[rt.generated_store.capsule(i)['source_row_id'] for i in docs],
        reference_position_counts=[rt.generated_store.capsule(i)['actual_length'] for i in docs],
        gradient_coverage=coverage,gradient_sweeps=result.counters['gradient_sweeps'],
        full_sweep_rows=[dict(gradient=s['gradient'],weight_sha256=s['weight_sha256'],loss=s['value'],
            rows_sha256=s['rows_sha256'],coverage=s['coverage']) for s in sweeps],
        method_gradient_shared=False,trials=rows,selected_sha256=tensor_sha256(result.weight),
        stop_reason=result.stop_reason,history_appends=None,
        zero_gradient_scope='SPACE_EMPTY_OR_RANK_UNRESOLVED' if not gradients else None)


def run_schedule(rt, arm, WN, reference, current, rows, K, allowed, space, meta, directory):
    if arm not in ARMS:raise ValueError('EXACT_SCHEDULE_ALLOWLIST')
    entry_started=time.monotonic()
    reuse=arm==ARMS[1]
    rt.copy_weight(WN);restore_rng(rt.rng);rt.guard()
    entry_rng=digest(capture_rng());epoch=model_guard(rt.model)
    reference_reset=reference.reset_counters()
    current.collect_weight_timing()
    current_before=deepcopy(current.work);weight_before=dict(current.weight_work)
    input_id=digest(meta);teacher=rt.generated_store.receipt['manifest_sha256']
    geometry_id=digest(space.receipt());session=controller=None
    weights={};sweeps=[];events=EventStore(directory/'events',directory/'gradients')
    torch.cuda.synchronize();started=time.monotonic()
    entry_seconds=started-entry_started
    try:
        if reuse:
            identities=dict(current=input_id,teacher=teacher)
            policy=RuntimePolicy(tuple(WN.shape),current.device,epoch,identities,rt.lock['execution']['commit'],
                epoch_getter=lambda:model_guard(rt.model),input_identity_getter=lambda:dict(identities),
                source_identity_getter=lambda:rt.lock['execution']['commit'],verify_bytes=False)
            session=EndpointSession(rt.identity,dict(case_ids=rt.lock['sample_order'],batch=1),policy)
            native_handle=session.bind_native(WN)
            controller=CurrentObservationController(WN,current,rows,session,native_handle,
                teacher_identity=dict(manifest=teacher),input_identity=dict(sha256=input_id))
            anchor=controller.anchor_rows
        else:
            native_handle=None
            anchor=score_rows(current,WN,rows)
        create_json(directory/'native-current-anchor.json',anchor)
        objective_id=digest(dict(data=DATA_ID,teacher=teacher,reduction='document_actual_T_mean_then512',fullvocab=True))

        def handle_for(weight):
            h=tensor_sha256(weight)
            if h==tensor_sha256(WN):return native_handle
            if weights.get('sha')!=h:
                weights.update(sha=h,handle=session.bind_candidate(weight,dict(objective=objective_id,ordinal=len(sweeps),weight=h)))
            return weights['handle']

        def objective(weight,gradient=False):
            if gradient and any(s['gradient'] for s in sweeps):raise ValueError('MORE_THAN_ONE_METHOD_GRADIENT')
            try:
                value=reference.kl(weight,gradient=gradient,session=session,
                    handle=handle_for(weight) if reuse else None,
                    allow_trial_overflow=not gradient and tensor_sha256(weight)!=tensor_sha256(WN))
            except FiniteTrialModelOverflow as exc:
                receipt=exc.receipt
                # First preserve the interrupted scope even if independent
                # full-bank revalidation subsequently finds a technical error.
                create_json(directory/'reference-overflow'/f'{len(sweeps):02d}-partial.json',receipt)
                checked=recheck_fixed_teacher_bank(rt.generated_store)
                receipt.update(weight_sha256=tensor_sha256(weight),value=None,ordinal=len(sweeps),
                    rows_sha256=digest(receipt['partial_rows']),entire_fixed_teacher_bank_rechecked=checked['complete'],
                    fixed_bank_recheck=checked)
                create_json(directory/'reference-sweeps'/f'{len(sweeps):02d}.json',receipt)
                sweeps.append(receipt)
                raise TrialNumericalOverflow('FINITE_TRIAL_MODEL_OVERFLOW_FIXED_BANK_'+
                    ('RECHECKED' if checked['complete'] else 'VALIDATION_SKIPPED_USER_DIRECTED')) from exc
            receipt=reference.last_sweep
            receipt.update(weight_sha256=tensor_sha256(weight),value=value[0],ordinal=len(sweeps),rows_sha256=digest(value[2]))
            create_json(directory/'reference-sweeps'/f'{len(sweeps):02d}.json',receipt)
            sweeps.append(receipt)
            return value

        def guard(weight):
            if reuse:
                passed,reasons=controller.guard(handle_for(weight));values=controller.candidate_rows
            else:
                values=score_rows(current,weight,rows);passed,reasons=quality_ok(values,anchor)
            return Check(passed,'PER_SEQUENCE_AND_EXACT_ID_GUARD',dict(reasons=reasons,rows=values))

        def actual_invariant(weight,ideal,actual):
            value=(controller.invariant(handle_for(weight),ideal,K,allowed) if reuse else
                   invariant(current,rows,anchor,weight,WN,ideal,K,allowed))
            return Check(value['pass'],'FULL_TOKEN_ACTUAL_INVARIANT',value)

        result=optimize('EN-F',WN,objective=objective,space=space,guard=guard,
            invariant=actual_invariant,proposal_check=lambda d:ideal_check(d,K),
            event=events,objective_id=objective_id,initial_observation=None)
        rt.guard()
        if digest(capture_rng())!=entry_rng:raise ValueError('SCHEDULE_RNG_MUTATION')
        if controller is not None:controller.close()
        if session is not None:session.close()
        current.collect_weight_timing();torch.cuda.synchronize()
        seconds=time.monotonic()-started
        exact=exact_receipt(rt,result,sweeps,geometry_id,input_id,teacher,epoch)
        work=dict(wall_seconds=seconds,reference=deepcopy(reference.work),
            current=_difference(current_before,current.work),session=None if session is None else dict(session.work),
            external_current_weight=_difference(weight_before,current.weight_work),
            observation=None if controller is None else dict(controller.work),
            optimizer=result.counters,schedule_entry_pre_timing_seconds=entry_seconds,
            previous_reference_counters_before_reset_DO_NOT_ADD=reference_reset,
            timing_boundary='native anchor/session creation through controller and session close; CUDA synchronized boundaries',
            includes_controller_evidence_IO=True,excludes_shared_native_geometry_teacher_commit_observer=True,
            no_nested_timer_sum=True,method_gradient_shared=False,Past_actual='EMPTY_NA')
        create_json(directory/'selection-ledger.json',dict(**result.receipt(),work=work,event_refs=events.refs))
        create_json(directory/'execution-exactness-before-history.json',exact)
        return result,exact,work
    finally:
        if controller is not None:controller.close()
        if session is not None:session.close()
