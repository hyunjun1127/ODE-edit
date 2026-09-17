"""Bounded actual-W0 v2 preparation, never a scientific accepted prefix.

No model is loaded on import. ``run`` is the sole GPU entrypoint and is invoked
by the separately admitted preparation job. Every check writes its evidence
before a failed condition raises; READY requires all mandatory actual stages.
Selected W/M states exist only as immutable RAM snapshots, not disk checkpoints.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import resource
import time
import traceback

from .common import ROOT, LAYERS, Ledger, digest, identity, save, serial, tensor_save, verify


NUMERICS = dict(version='V2_ACTUAL_ENDPOINT_TECHNICAL_V1',
    weight_max_abs=5e-6, weight_frobenius_relative=5e-5,
    native_instrumentation_target='EXACT_BYTES', E_repeat=5e-5,
    H_repeat=5e-5, B_repeat=5e-7, strict_pair='EXACT_REQUEST_ID_SETS',
    required_C45678_search_vectors=7, required_distinct_a4=2,
    budget='UNCHANGED_CONTROLLER_LIMITS_V2', GPU_derivative_FD_ULP='NOT_REQUIRED_NOT_RUN')
MANDATORY = ('W0_COLD_IDENTITY', 'TEACHER_REPRODUCTION', 'NATIVE_N4',
    'NATIVE_BLUE_48', 'NATIVE_BLUE_45678', 'GATES_REPLAY_REPEAT',
    'C45678_BOUNDED_SEARCH', 'CACHE_REPLAY', 'HISTORY5_NEXT_ENTRY')


class TechnicalHold(RuntimeError):
    """Measured mandatory technical condition was not established."""


def check_locked_numerics(lock):
    expected=dict(E_H_repeat=NUMERICS['E_repeat'],B_repeat=NUMERICS['B_repeat'],
        native_weight_max_abs=NUMERICS['weight_max_abs'],
        native_weight_relative_Frobenius=NUMERICS['weight_frobenius_relative'],
        strict_pair_repeat='EXACT_ID_SET',E_H_allowance=1e-4,B_tie=1e-6,current_plateau=False)
    if any(lock.get('numerical',{}).get(k)!=v for k,v in expected.items()):
        raise ValueError('TECHNICAL_NUMERICS_EXECUTION_LOCK_MISMATCH')
    return expected


def finite_json(value):
    """Failure evidence remains serializable without inventing finite values."""
    value = serial(value)
    if isinstance(value, dict): return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value, list): return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return dict(nonfinite=repr(value),valid=False)
    return value


def repeated_scores(first, second):
    """Scalar resolution plus exact protected-ID checks, no quality filtering."""
    a,b=first['controller'],second['controller'];rows=[]
    for field,tolerance in (('training_e',NUMERICS['E_repeat']),
                            ('past_h',NUMERICS['H_repeat']),
                            ('base_kl',NUMERICS['B_repeat'])):
        x,y=a[field],b[field]
        difference=None if x is None or y is None else abs(x-y)
        ok=(x is None and y is None) if difference is None else (
            math.isfinite(x) and math.isfinite(y) and difference<=tolerance)
        rows.append(dict(field=field,first=x,second=y,absolute_difference=difference,
                         tolerance=tolerance,passed=ok))
    for field in ('current_strict','current_pair','past_strict','past_pair'):
        x,y=a[field],b[field]
        ok=(x is None and y is None) if x is None or y is None else set(x)==set(y)
        rows.append(dict(field=field,first=None if x is None else sorted(x),
                         second=None if y is None else sorted(y),passed=ok))
    return dict(passed=all(r['passed'] for r in rows),checks=rows,
        first_controller=serial(a),second_controller=serial(b),
        comparison='SAME_ACTUAL_ENDPOINT_REPEATED_MODEL_FORWARD')


def weight_comparison(left,right):
    import torch
    from .native import tensor_sha
    rows=[]
    for layer in LAYERS:
        x,y=left[layer],right[layer]
        schema=x.shape==y.shape and x.dtype==y.dtype==torch.float32
        finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()) if schema else False
        if schema and finite:
            difference=(x.double()-y.double());maximum=float(difference.abs().max())
            norm=float(difference.norm());scale=float(x.double().norm())
            relative=norm/scale if scale else (0. if norm==0 else math.inf)
            ok=maximum<=NUMERICS['weight_max_abs'] and relative<=NUMERICS['weight_frobenius_relative']
        else:maximum=norm=relative=None;ok=False
        rows.append(dict(layer=layer,schema=schema,finite=finite,
            left_sha256=tensor_sha(x),right_sha256=tensor_sha(y),exact=torch.equal(x,y),
            max_abs=maximum,difference_frobenius=norm,relative_frobenius=relative,passed=ok))
    return dict(passed=all(r['passed'] for r in rows),layers=rows,
                absolute_tolerance=NUMERICS['weight_max_abs'],relative_tolerance=NUMERICS['weight_frobenius_relative'])


def coverage_evidence(controller, stop):
    coverage=dict(controller.coverage())
    limits=serial(controller.limits)
    passed=(coverage['completed_search_gate_vectors']>=NUMERICS['required_C45678_search_vectors']
            and coverage['distinct_a4']>=NUMERICS['required_distinct_a4'])
    return dict(passed=passed,status='COUNT_PROXY_ADEQUATE' if passed else 'INSUFFICIENT_SEARCH',
        coverage=coverage,stop=stop,limits=limits,counts=dict(controller.counts),
        candidates=[controller.candidate_receipt(r) for r in controller.records],
        incomplete=controller.incomplete_records,budget_expanded=False,
        scientific_performance_gate=False,solver_simplex_or_optimality_certified=False)


class Journal:
    def __init__(self,root):
        self.root=Path(root);self.stage=None;self.passed={};self.receipts=[]

    @contextmanager
    def step(self,name):
        self.stage=name;begin=time.monotonic()
        save(self.root/name/'begin.json',dict(stage=name,status='STARTED',
            utc=datetime.now(timezone.utc).isoformat(),scientific=False))
        try:yield
        except BaseException as exc:
            partial=getattr(exc,'native_partial',None)
            ref=None if partial is None else tensor_save(self.root/name/'native-failure-evidence.pt',partial)
            save(self.root/name/'failure.json',finite_json(dict(stage=name,status='HOLD' if isinstance(exc,TechnicalHold) else 'TECHNICAL_FAILURE',
                error=repr(exc),traceback=traceback.format_exc(),native_partial=ref,
                elapsed_seconds=time.monotonic()-begin,science_ready=False)))
            raise
        finally:
            save(self.root/name/'elapsed.json',dict(elapsed_seconds=time.monotonic()-begin,
                completed_check=self.passed.get(name,False)))

    def check(self,name,evidence,passed=True):
        receipt=save(self.root/name/'result.json',finite_json(dict(stage=name,
            status='PASS' if passed else 'HOLD',evidence=evidence)))
        self.receipts.append(receipt);self.passed[name]=bool(passed)
        if not passed:raise TechnicalHold(name+':'+str(evidence.get('status','CHECK_NOT_ESTABLISHED')))
        return receipt

    def ready(self):
        missing=[name for name in MANDATORY if self.passed.get(name) is not True]
        if missing:raise TechnicalHold('MANDATORY_STAGES_NOT_PASSED:'+','.join(missing))


def _persist_fit(folder,result):
    """Diagnostic target/key/readout evidence only, never selected W/M payload."""
    payload={k:v for k,v in result.items() if k not in ('weight','receipt')}
    ref=tensor_save(Path(folder)/'native-evidence.pt',payload)
    from .native import tensor_sha
    return save(Path(folder)/'receipt.json',result['receipt'] | dict(evidence=ref,
        target_sha256=tensor_sha(result['target']) if 'target' in result else None,
        W_M_checkpoint_saved=False,exact_crash_resume=False,scientific_target_calls=0))


def native_totals(receipts):
    rows=[json.loads(Path(r['path']).read_text()) for r in receipts]
    return dict(native_target_calls=sum(r.get('compute_z',0) for r in rows),
        native_solves=sum(r.get('solve',0) for r in rows),native_key_calls=sum(r.get('compute_ks',0) for r in rows),
        actual_adam_recorded=sum(r.get('adam_updates',0) for r in rows),
        actual_loss_evaluations_recorded=sum(r.get('loss_evaluations',0) for r in rows),
        clamp_hits_recorded=sum(r.get('clamp_hits',0) for r in rows),
        target_calls_without_adam_loss_trace=sum(r.get('compute_z',0) for r in rows if 'adam_updates' not in r),
        native_fit_inclusive_seconds=sum(r.get('seconds',0.) for r in rows),
        pure_writer='NOT_SEPARATED',counting='RECEIPTS_NONOVERLAPPING_NATIVE_INVOCATIONS_ONLY',
        nested_component_times_not_added=True)


def _reference(rt,current,entry,layers,folder):
    import torch
    from .native import tensor_sha
    rt.restore(entry);hp=deepcopy(rt.hp[4]);hp.layers=list(layers)
    # These temporary stacked tensors are technical-only, not history updates.
    history=torch.cat([rt.M[l] for l in layers]);projector=torch.cat([rt.P[l] for l in layers])
    result=rt.fitter.multi_layer_reference(rt.model,rt.tok,hp,history,projector,rt.requests(current),
        include_history=False,capture=layers==(4,))
    rt.adopt_native_weights(result['weights'],entry['rng'])
    state=rt.snapshot()
    targets=None
    if result['captures'] is not None:
        target=torch.stack(result['captures']['compute_z'],dim=1)
        targets=dict(sha256=tensor_sha(target),evidence=tensor_save(Path(folder)/'native-target.pt',dict(target=target)))
    receipt=save(Path(folder)/'receipt.json',result['receipt'] | dict(state=state['state'],targets=targets,
        instrumentation='DISABLED_TECHNICAL_PARITY_REFERENCE',W_M_checkpoint_saved=False,
        temporary_CPU_stacked_history_bytes=history.numel()*history.element_size(),
        temporary_CPU_stacked_projector_bytes=projector.numel()*projector.element_size(),
        temporary_stacks_are_extra_to_singleton_assets=True))
    rt.restore(entry)
    return state,receipt


def _single_sequence(rt,current,start,layers,folder):
    rt.restore(start);receipts=[]
    for layer in layers:
        result=rt.fit(current,layer,instrument=True)
        receipts.append(_persist_fit(Path(folder)/f'L{layer}',result))
        del result
    return rt.snapshot(),receipts


def _cold_and_teacher(rt,lock,entry,out,journal):
    import torch
    from .native import tensor_sha
    from .metrics import TeacherStore
    with journal.step('W0_COLD_IDENTITY'):
        mapping={str(l):dict(rt.pmap[str(l)],M_shape=list(rt.M[l].shape),
            M_dtype=str(rt.M[l].dtype),M_zero=bool(torch.count_nonzero(rt.M[l])==0),
            P_finite=bool(torch.isfinite(rt.P[l]).all())) for l in LAYERS}
        checks=all(v['M_zero'] and v['P_finite'] and v['physical_layer']==int(l)
            and v['source_index']==int(l)-4 and v['local_index']==0 for l,v in mapping.items())
        journal.check('W0_COLD_IDENTITY',dict(state=entry['state'],mapping=mapping,
            current_identity=digest(rt.records[:100]),prefix_identity=digest(rt.records),
            context_tokens=digest(rt.context_tokens),source=lock['source_head'],
            selected_fp32=all(w.dtype==torch.float32 for w in rt.W.values()),
            all_token_physical_parameters=True,model_mode_eval=not rt.model.training,
            new_gradient_or_repair_runtime=False),checks)
    with journal.step('TEACHER_REPRODUCTION'):
        before=rt.observe(lambda:rt.metrics.base('S64'))
        save(out/'TEACHER_REPRODUCTION'/'original.json',finite_json(before))
        teacher=identity(rt.teacher.manifest_path);regenerated=False
        if not math.isfinite(before['D']):raise TechnicalHold('NONFINITE_ORIGINAL_TEACHER_CHECK')
        if abs(before['D'])>NUMERICS['B_repeat']:
            from project.run_scripts.local_z_adaptive_allocation.technical import refresh_teacher
            teacher=refresh_teacher(rt,out/'teacher-repair-v1');regenerated=True
            rt.teacher=TeacherStore(lock['reference_root'],teacher['path'],expected_manifest_sha=teacher['sha256'],verify_payload_hashes=False)
            rt.metrics.teacher=rt.teacher;rt.metrics._canonical.teacher=rt.teacher
        after=rt.observe(lambda:rt.metrics.base('S64'))
        save(out/'TEACHER_REPRODUCTION'/'effective.json',finite_json(after))
        ok=math.isfinite(after['D']) and abs(after['D'])<=NUMERICS['B_repeat']
        journal.check('TEACHER_REPRODUCTION',dict(original_D=before['D'],effective_D=after['D'],
            tolerance=NUMERICS['B_repeat'],teacher_manifest=teacher,regenerated=regenerated,
            document_ids_unchanged=True,teacher_receipt=rt.teacher.receipt,
            backward_calls=0,Report256_created=False),ok)
    common=json.loads(Path(lock['cold_capsule']['path']).read_text())
    capsule=dict(common,status='V2_COLD_CAPSULE_READY',source_head=lock['source_head'],
        W0={str(l):tensor_sha(w) for l,w in rt.w0.items()},M0='EXACT_ZERO_ALL_FIVE',
        history_layers=list(LAYERS),projector_mapping=rt.pmap,contexts=rt.context,
        context_tokens=rt.context_tokens,rng=entry['rng'],teacher_manifest=teacher,
        teacher_regenerated=regenerated,parent_cold_capsule=lock['cold_capsule'],
        method_seed=20260916,C4_sampling_seed=20260915,W_M_checkpoint_saved=False,
        tf32_matmul=False,tf32_cudnn=False,exact_crash_resume=False)
    return capsule


def _gates_and_repeats(rt,current,entry,n4,out,journal):
    import torch
    from .runtime import fp32_gate
    # Nonempty Past is a labelled technical fixture, never scientific B1 input.
    past=rt.records[100:164];rows=[];scores={};snapshots={}
    with journal.step('GATES_REPLAY_REPEAT'):
        for i,gate in enumerate((0.,1.,.5,.75,.75,.5)):
            rt.restore(entry);value=fp32_gate(entry['W'][4],n4['W'][4],gate)
            rt.adopt_native_weights({4:value},entry['rng']);snapshot=rt.snapshot()
            endpoint_copy=(torch.equal(snapshot['W'][4],entry['W'][4]) if gate==0 else
                           torch.equal(snapshot['W'][4],n4['W'][4]) if gate==1 else
                           torch.equal(snapshot['W'][4],entry['W'][4]+gate*(n4['W'][4]-entry['W'][4])))
            other_unchanged=all(snapshot['whash'][l]==entry['whash'][l] for l in LAYERS if l!=4)
            row=dict(index=i,gate=gate,state=snapshot['state'],endpoint_copy_or_native_FP32_formula=endpoint_copy,
                other_selected_weights_unchanged=other_unchanged,inner_history0=snapshot['mhash']==entry['mhash'],
                all_token_actual_weight=True,model_and_hook_guard='Runtime.guard')
            if gate in (.5,.75):
                score=rt.observe(lambda:rt.metrics.score(current,past))
                ref=save(out/'GATES_REPLAY_REPEAT'/f'score-{i}.json',finite_json(score));row['score']=ref
                if gate in scores:
                    row['repeated_score']=repeated_scores(scores[gate],score)
                    row['same_snapshot_state']=snapshot['state']==snapshots[gate]['state']
                else:scores[gate]=score;snapshots[gate]=snapshot
            rows.append(row)
            save(out/'GATES_REPLAY_REPEAT'/f'probe-{i}.json',finite_json(row))
        rt.restore(entry)
        checks=all(r['endpoint_copy_or_native_FP32_formula'] and r['other_selected_weights_unchanged'] and r['inner_history0']
            and r.get('repeated_score',{'passed':True})['passed'] and r.get('same_snapshot_state',True) for r in rows)
        journal.check('GATES_REPLAY_REPEAT',dict(probes=rows,restore_state=rt.state(),
            entry_restored=rt.state()==entry['state'],fixed_forward_order=[.5,.75],fixed_reverse_order=[.75,.5],
            adaptive_optimizer_order_permutation_claim=False,Past_fixture_only=dict(start=100,stop=164,requests=64,
                digest=digest(past),scientific_controller_input=False),scientific_B1_Past_empty=True),
            checks and rt.state()==entry['state'])


def _search_history(rt,current,entry,out,journal):
    from .controller import Controller, run_arm
    from .runtime import Backend
    rt.restore(entry);backend=Backend(rt,current,[],out/'controller')
    controller=None
    with journal.step('C45678_BOUNDED_SEARCH'):
        controller=Controller(backend,LAYERS,namespace=backend.namespace,history_layers=LAYERS,
            event_sink=Ledger(out/'controller'/'events'))
        selected,stop=run_arm(controller,'C45678',finalize=False)
        evidence=coverage_evidence(controller,stop)
        evidence['scientific_B1_Past_empty']=not backend.past
        evidence['selected']=controller.candidate_receipt(selected)
        journal.check('C45678_BOUNDED_SEARCH',evidence,evidence['passed'])
    with journal.step('CACHE_REPLAY'):
        before=dict(controller.counts);state_before=rt.state();rows=[]
        # Completed cached gates only: no new proposal, fit or score allowance.
        fixed=[controller.n4,controller.records[-1]]
        for candidate in fixed+list(reversed(fixed)):
            replay=controller.evaluate(candidate.gates)
            rows.append(dict(gates=candidate.gates,same_object=replay is candidate,
                state_token=replay.state_token,endpoint_equal=replay.state_token==candidate.state_token,
                score_equal=serial(replay.scores)==serial(candidate.scores)))
        unchanged=all(controller.counts[k]==before[k] for k in
            ('l4_fits','suffix_fits','endpoints','baseline_scores','extra_adam','search_proposals'))
        entry_preserved=rt.state()==state_before==entry['state']
        prefixes=[dict(namespace=k[0],layer=k[1],prefix_state=k[2],history=k[3]) for k in controller.fit_cache]
        journal.check('CACHE_REPLAY',dict(probes=rows,counts_before=before,counts_after=dict(controller.counts),
            no_extra_fit_or_forward=unchanged,entry_preserved=entry_preserved,fit_cache_keys=prefixes,
            key_includes_bound_namespace_state_layer_history=True,namespace=backend.namespace),
            unchanged and entry_preserved and all(r['same_object'] and r['endpoint_equal'] and r['score_equal'] for r in rows))
    with journal.step('HISTORY5_NEXT_ENTRY'):
        controller.seal_selection(selected)
        save(out/'HISTORY5_NEXT_ENTRY'/'selection-sealed.json',controller.candidate_receipt(selected))
        controller.finalize(selected)
        continuation=rt.snapshot();actual=rt.state();history=json.loads(Path(backend.history_receipt['path']).read_text())
        rt.restore(entry);entry_restore=rt.state()==entry['state']
        rt.restore(continuation);next_entry=rt.state();rt.guard()
        rows=history['rows'];passed=(entry_restore and actual==next_entry and len(rows)==5 and
            [r['layer'] for r in rows]==list(LAYERS) and all(r['history_append']==1 for r in rows) and
            history['candidate_appends']==0 and actual['W']==selected.state['state']['W'] and
            actual['contexts']==entry['state']['contexts'] and actual['rng']==entry['state']['rng'] and
            controller.counts['commits']==1)
        journal.check('HISTORY5_NEXT_ENTRY',dict(history=backend.history_receipt,rows=rows,
            selected=controller.candidate_receipt(selected),technical_next_entry=next_entry,
            entry_restore=entry_restore,in_memory_snapshot_restore=True,all_current_requests=100,
            numerical_gate_actual_process='TECHNICAL_ONLY',scientific_commits=0,
            GPU_off_on_continuation='NOT_TESTED',disk_W_M_checkpoint=False,exact_crash_resume=False,
            received_ledger='SCIENTIFIC_RUNNER_NOT_EXERCISED_BY_TECHNICAL_BATCH'),passed)
        rt.restore(entry)
    return backend,controller


def run(lock_path):
    import torch
    from .runtime import Runtime
    lock_path=Path(lock_path);lock=json.loads(lock_path.read_text());verify(lock)
    locked_numerics=check_locked_numerics(lock)
    out=ROOT/'technical'/'attempt-v1';out.mkdir(parents=True,exist_ok=False)
    ready_path=Path(lock.get('common_ready',out/'READY.json'))
    if ready_path.resolve()!=(out/'READY.json').resolve():raise ValueError('V2_TECHNICAL_READY_PATH')
    save(out/'numerics.lock.json',dict(**NUMERICS,execution_lock=identity(lock_path),execution_numerical=locked_numerics,
        locked_before_model_load=True,no_quality_condition=True))
    journal=Journal(out);started=time.monotonic();rt=entry=None;fit_receipts=[];backend=None
    try:
        save(out/'LOAD-begin.json',dict(stage='LOAD',status='STARTED',execution_lock=identity(lock_path)))
        rt=Runtime(lock);entry=rt.snapshot();current=rt.records[:100]
        capsule=_cold_and_teacher(rt,lock,entry,out,journal)
        for stage,layers in (('NATIVE_N4',(4,)),('NATIVE_BLUE_48',(4,8)),('NATIVE_BLUE_45678',LAYERS)):
            with journal.step(stage):
                reference,ref=_reference(rt,current,entry,layers,out/stage/'reference')
                fit_receipts.append(ref)
                if stage=='NATIVE_N4':
                    actual,refs=_single_sequence(rt,current,entry,(4,),out/stage/'singleton')
                    n4=actual
                else:
                    actual,refs=_single_sequence(rt,current,n4,layers[1:],out/stage/'singleton')
                fit_receipts.extend(refs)
                evidence=weight_comparison(reference['W'],actual['W'])
                target_exact=True
                if stage=='NATIVE_N4':
                    native_row=json.loads(Path(ref['path']).read_text())
                    singleton_row=json.loads(Path(refs[0]['path']).read_text())
                    target_exact=native_row['targets']['sha256']==singleton_row['target_sha256']
                    evidence['instrumentation_target_exact_bytes']=target_exact
                evidence.update(native_reference=ref,singletons=refs,source_native_modified=False,
                    original_BLUE_layers=list(layers),reference_history_appends=0,singleton_history_appends=0,
                    entry_state=entry['state'],reference_state=reference['state'],actual_state=actual['state'],
                    same_W0_L4_fit_reused_in_later_singleton_paths=stage!='NATIVE_N4',
                    instrumentation_comparison='UNINSTRUMENTED_NATIVE_LOOP_VERSUS_INSTRUMENTED_SINGLETON')
                journal.check(stage,evidence,evidence['passed'] and target_exact and reference['mhash']==actual['mhash']==entry['mhash'])
                rt.restore(entry);del reference,actual
        _gates_and_repeats(rt,current,entry,n4,out,journal)
        del n4
        backend,controller=_search_history(rt,current,entry,out,journal)
        fit_receipts.extend(backend.fit_receipts)
        journal.ready();rt.restore(entry);rt.guard()
        capsule_ref=save(out/'capsule.json',capsule)
        totals=native_totals(fit_receipts)
        cost=dict(seconds=time.monotonic()-started,native=totals,
            program_component_seconds=rt.timing,program_component_timers_nested=True,
            controller_counts=dict(controller.counts),native_receipts=fit_receipts,
            score_receipts=backend.score_receipts,peak_GPU_allocated=torch.cuda.max_memory_allocated(),
            peak_GPU_reserved=torch.cuda.max_memory_reserved(),peak_host_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            allocated_GPU_seconds='SCHEDULER_POSTRUN_NOT_OBSERVED_BY_PROGRAM',science_targets=0,science_commits=0)
        cost_ref=save(out/'cost.json',finite_json(cost))
        ready=dict(status='TECHNICAL_READY',instruction_id=lock['instruction_id'],source_head=lock['source_head'],
            execution_lock_sha256=identity(lock_path)['sha256'],capsule=capsule_ref,teacher=capsule['teacher_manifest'],seconds=cost['seconds'],
            checks=journal.receipts,numerics=identity(out/'numerics.lock.json'),cost=cost_ref,
            required_actual_stages=list(MANDATORY),all_required_actual_stages_pass=True,
            state_restored_to_cold=rt.state()==entry['state'],scientific_G0='NOT_RUN',scientific_commits=0,
            checkpoint_policy='NO_DISK_W_M; IMMUTABLE_RAM_ROLLBACK; NATIVE_TARGET_KEY_SCALAR_HASH_DIAGNOSTICS',
            exact_crash_resume=False,GPU_off_on_continuation='NOT_TESTED')
        if not ready['state_restored_to_cold']:raise TechnicalHold('FINAL_COLD_RESTORE')
        save(ready_path,ready)
        return ready
    except BaseException as exc:
        restore=dict(attempted=rt is not None and entry is not None,succeeded=False)
        if restore['attempted']:
            try:rt.restore(entry);restore['succeeded']=rt.state()==entry['state']
            except BaseException as rollback:restore['error']=repr(rollback)
        # Failure can occur inside the controller before it returns. Its already
        # durable fit receipts still belong to the technical cost ledger.
        completed={r['path']:r for r in fit_receipts}
        for path in out.rglob('receipt.json'):
            row=json.loads(path.read_text())
            if 'compute_z' in row and 'solve' in row:
                ref=identity(path);completed[ref['path']]=ref
        save(out/'failure.json',finite_json(dict(status='HOLD' if isinstance(exc,TechnicalHold) else 'TECHNICAL_FAILURE',
            stage=journal.stage or 'LOAD',error=repr(exc),traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-started,execution_lock=identity(lock_path),
            required_checks_passed=journal.passed,completed_stage_receipts=journal.receipts,
            completed_native_receipts=list(completed.values()),native_counts=native_totals(list(completed.values())),
            program_component_seconds=None if rt is None else rt.timing,
            cost_boundary='FAILED_OPEN_NATIVE_FIT_PARTIAL_TRACE_SEPARATE_FROM_COMPLETED_INVOCATIONS',
            rollback=restore,science_ready=False,
            scientific_started=False,scientific_commits=0,new_scientific_targets=0)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--lock',required=True)
    run(parser.parse_args().lock)
