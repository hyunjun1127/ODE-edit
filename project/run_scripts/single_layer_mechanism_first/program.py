"""One immutable dependency job: finish T0, B1, then gated B3/B10.

No scheduler API, automatic submission, foreign task inspection or retry loop.
The user may stop the agent after registration; the submitted program has all
permitted computation and fail-closed scientific/technical boundaries inside.
"""
import copy
import gc
import json
import os
from pathlib import Path
import resource
import time
import torch
from .config import B1_ARMS,CHAIN_ARMS
from .current import fixed_panel
from .science import batch,allowed_space
from .observers import evaluate,seal
from .gates import b1_gate,s3_gate,reduce_observation
from .transaction import restore,clone_b1_cum_as_step
from .decision import DecisionOracle
from project.run_scripts.single_layer_edit_preserving_correction.common import write,member,sha,tensor_sha,digest
from project.run_scripts.single_layer_edit_preserving_correction.observer import CanonicalObserver,strict_summary
from project.run_scripts.baseline_mechanism_first.fixtures import restore_rng


class StageResourceHold(RuntimeError):
    pass


B1_ONLY_AUTHORITY = 'USER_B1_ONLY_MONITOR_TO_COMPLETION_20260920'


def require_program(lock):
    if lock.get('phase')!='GATED_PROGRAM':
        raise ValueError('EXACT_GATED_PROGRAM')
    if lock.get('maximum_batch')==1:
        if (lock.get('b1_only_authority')!=B1_ONLY_AUTHORITY or
                lock.get('sequential_authorized') is not False or
                lock.get('auto_continue') is not False or
                lock.get('agent_monitoring_after_release') is not True):
            raise ValueError('B1_ONLY_EXPLICIT_AUTHORITY_REQUIRED')
    elif lock.get('maximum_batch')!=10 or lock.get('b1_only_authority'):
        raise ValueError('EXACT_GATED_TEN_BATCH_PROGRAM')
    if lock.get('scientific_gates_required') is not True or lock.get('scheduler_writes_in_program') is not False:
        raise ValueError('PROGRAM_MUST_PRESERVE_GATES_NO_SUBMISSIONS')
    if lock.get('maximum_batch')==10 and lock.get('agent_monitoring_after_release') is not False:
        raise ValueError('USER_NO_MONITORING')
    if lock.get('disk_state_checkpoints') is not False or lock.get('save_checkpoints') is not False:
        raise ValueError('USER_NO_CHECKPOINT')


def space_check(root,required,phase):
    stat=os.statvfs(root);actual=stat.f_bavail*stat.f_frsize
    receipt=dict(phase=phase,free_bytes=actual,needed_estimate_bytes=required,
                 free_inodes=stat.f_favail,waiver=False,reserved=False,exclusive=False)
    write(Path(root)/f'storage-{phase}.json',receipt)
    if actual<required:raise StageResourceHold('STORAGE_ESTIMATE_UNAVAILABLE:'+phase)


def inherit_hook(rt,root):
    """Read only the exact dependency's completed evidence, inside the job.
    No agent poll and no repeated z fitting when existing evidence is valid.
    """
    parent=rt.lock['hook_dependency'];lock=json.loads(Path(parent['lock']['path']).read_text())
    if sha(parent['lock']['path'])!=parent['lock']['sha256']:raise ValueError('HOOK_LOCK_BYTES')
    if lock['execution']['commit']!=parent['source']:raise ValueError('HOOK_SOURCE')
    out=Path(lock['output']);terminal=json.loads((out/'terminal.json').read_text())
    if terminal['status']!='T0_HOOK_PART_COMPLETE_NOT_FULL_T0_READY' or terminal['result']['pass_'] is not True:
        raise ValueError('DEPENDENCY_HOOK_NOT_VALID')
    # New code versions may add runners but cannot change the measured hook.
    for name in ('z_hook.py','z_hook_parity.py'):
        old=Path(lock['execution']['source'])/'project/run_scripts/single_layer_mechanism_first'/name
        current=Path(__file__).parent/name
        if sha(old)!=sha(current):raise ValueError('HOOK_IMPLEMENTATION_NOT_COVERED:'+name)
    local=out/'technical-hook'
    p=torch.load(local/'actual-write-native.pt',weights_only=True,mmap=True,map_location='cpu')
    indices=fixed_panel(rt.records[:100],lambda r:r['case_id']);records=[rt.records[i] for i in indices]
    panel=json.loads((local/'panel.json').read_text())
    if panel['case_ids']!=[r['case_id'] for r in records]:raise ValueError('HOOK_PANEL_IDENTITY')
    write(root/'hook-reuse.json',dict(job_id=parent['job_id'],terminal=member(out/'terminal.json'),
        native_payload=member(local/'actual-write-native.pt'),new_z_fits=0,
        hook_only_not_full_T0=True,measured_configured_batch1=True,full_batch16='NOT_TESTED'))
    return p,records


def subset_w0(rt,full,records,endpoint_seal):
    """Exact observed W0 rows, subset reduction; no invented token predictions."""
    ids=[r['case_id'] for r in records];wanted=set(ids);out=copy.deepcopy(full)
    for tag,multiple in [('RS',1),('PS',2),('NS',10)]:
        group=out['metrics'][tag];rows=[r for r in group['rows'] if r['case_id'] in wanted]
        if [(r['case_id'],r['prompt_index']) for r in rows]!=[(c,i) for c in ids for i in range(multiple)]:
            raise ValueError('W0_SUBSET_ORDER')
        count=sum(r['success'] for r in rows)
        group.update(rows=rows,numerator=count,denominator=len(rows),rate=count/len(rows))
    out['raw']={k:[r for r in v if r['case_id'] in wanted] for k,v in out['raw'].items()}
    obs=CanonicalObserver(rt.model,rt.etok,runtime_identity=digest(rt.identity))
    out.update(requests=len(ids),request_order=digest(ids),strict=strict_summary(out['metrics'],ids),
        compatibility=obs.compatibility_for(records,rt.W0,selection_seal=endpoint_seal),
        selection_seal=endpoint_seal,work={k:0 for k in out['work']},
        W0_observer_reuse=dict(source_population=full['requests'],selected_population=len(ids),
            values='EXACT_SOURCE_ROWS',microbatch_padding_bit_parity='NOT_CLAIMED',new_forwards=0))
    reduce_observation(out)
    return out


def observe_batch(rt,reference,result,full_w0,*,batch_number,stage,extra_reference=True):
    root=result['directory'];records=result['records'];ledgers=dict(result['selections']);endpoints=dict(result['weights'])
    # Entry and own-native are same-state diagnostics, never an independent N4.
    default=next(iter(ledgers.values()))
    endpoints.update(ENTRY=result['entry'],OWN_NATIVE=result['native']['weight'])
    ledgers.update(ENTRY=default,OWN_NATIVE=default)
    wseal=seal(f'B{batch_number}','W0',rt.W0,records,result['selection_seal']['sha256'])
    w0=subset_w0(rt,full_w0,records,wseal)
    values,refs,_=evaluate(rt,reference,result['decision'],records,endpoints,ledgers,root/'observers/current',
        episode=f'{stage}:B{batch_number}:current',reference_observers=extra_reference,
        w0_result=w0)
    # Fixed first1000N repeated panel, not every previously seen N every step.
    if batch_number!=1:
        first=rt.records[:100];selected={a:result['weights'][a] for a in result['commits']}
        firstseal=seal(f'B{batch_number}','W0',rt.W0,first,result['selection_seal']['sha256'])
        evaluate(rt,reference,result['decision'],first,selected,{a:ledgers[a] for a in selected},
            root/'observers/first100',episode=f'{stage}:B{batch_number}:first100',reference_observers=False,
            greedy=False,w0_result=subset_w0(rt,full_w0,first,firstseal))
    seenvalues=values
    if batch_number in (3,5,10):
        seen=rt.records[:100*batch_number];selected={a:result['weights'][a] for a in result['commits']}
        seen_seal=seal(f'B{batch_number}','W0',rt.W0,seen,result['selection_seal']['sha256'])
        seenvalues,_,_=evaluate(rt,reference,result['decision'],seen,selected,{a:ledgers[a] for a in selected},
            root/'observers/allseen',episode=f'{stage}:B{batch_number}:allseen',reference_observers=False,
            w0_result=subset_w0(rt,full_w0,seen,seen_seal))
    return values,seenvalues


def run(rt,out):
    require_program(rt.lock);out=Path(out);start=time.monotonic()
    b1_only=rt.lock['maximum_batch']==1
    state={'stage':'T0','completed':{},'maximum_batch':rt.lock['maximum_batch'],
           'monitor_to_completion':b1_only}
    def mark(stage,**extra):
        state['stage']=stage
        print(json.dumps(dict(event='PROGRAM_PHASE',stage=stage,**extra)),flush=True)
    try:
        space_check(out,24*2**30,'T0_B1')
        if rt.lock.get('completed_hook_reuse'):
            from .reuse_completed_hook import reuse_completed_hook
            mark('T0_COMPLETED_HOOK_REUSE')
            native4,panel=reuse_completed_hook(rt,out/'completed-hook-reuse')
        elif rt.lock.get('hook_repair'):
            from .technical_repair import run_hook_repair
            mark('T0_HOOK_REPAIR')
            native4,panel=run_hook_repair(rt,out/'technical-hook-repair')
        else:
            native4,panel=inherit_hook(rt,out)
        reference=rt.reference()
        from .technical_decision import run_checks
        mark('T0_ACTUAL_DECISION')
        validation=run_checks(rt,reference,panel,native4,out/'technical-decision')
        # The technical function must return explicit evidence, not no-exception PASS.
        if not isinstance(validation,dict) or validation.get('pass_') is not True:
            raise RuntimeError('T0_FULL_CHECKS_NOT_ESTABLISHED')
        write(out/'T0_READY.json',validation)
        rt.copy_weight(rt.W0);rt.M.zero_();restore_rng(rt.rng);rt.oracles=[reference];rt.sync_oracles()
        del native4;gc.collect()
        from .z_hook import ZHookConfig
        from .native import CapturedHookedFitter
        rt.fitter=CapturedHookedFitter(rt.module,expected_source_sha256=rt.lock['editor_sha256'],
            contexts=rt.context,config=ZHookConfig(batch_size=1,record_vectors=False))
        mark('B1')
        result=batch(rt,reference,rt.records[:100],stage='B1',arm_names=B1_ARMS,batch_number=1,ledger=[],directory=out/'B1')
        # A single common W0 1000-request observer is kept out of all controllers.
        observer=CanonicalObserver(rt.model,rt.etok,runtime_identity=digest(rt.identity))
        w0records=rt.records[:100] if b1_only else rt.records
        w0name='W0-common100' if b1_only else 'W0-common1000'
        w0seal=seal(w0name,'W0',rt.W0,w0records,result['selection_seal']['sha256'])
        w0=observer.observe(w0records,rt.W0,selection_seal=w0seal,greedy=False);rt.sync_oracles()
        write(out/(w0name+'.json'),w0)
        values,seen=observe_batch(rt,reference,result,w0,batch_number=1,stage='B1')
        from .postselection import mechanism_report
        mechanism_report(rt,reference,result,values,w0,out/'B1/mechanism',technical=validation)
        gate=b1_gate(values['N4'],values['DEC_MODES_CUM'],result['results']['DEC_MODES_CUM'].receipt())
        write(out/'B1-to-S3.json',gate)
        state['completed']['B1']=list(B1_ARMS)
        if b1_only:
            # A PASS is a recorded scientific result, never B2 authorization.
            mark('B1_COMPLETE_USER_LIMIT')
            return finish(out,state,start,gate)
        last={a:result['commits'][a]['_state'] for a in ('N4','DEC_MODES_CUM')}
        last['DEC_MODES_STEP']=clone_b1_cum_as_step(last['DEC_MODES_CUM'])
        write(out/'B1/STEP-CUM-alias.json',dict(identity=last['DEC_MODES_STEP']['identity'],
            independent_in_memory_clone=True,new_fit_calls=0,disk_checkpoint=False))
        state['completed']['B1']=list(B1_ARMS)
        if not gate['pass']:
            mark('STOPPED_B1_GATE');return finish(out,state,start,gate)
        del result,values,seen;gc.collect()
        for stage,indices in [('S3',range(2,4)),('S10',range(4,11))]:
            # Capacity is rechecked, not waived or represented as exclusive reservation.
            space_check(out,(32 if stage=='S3' else 112)*2**30,stage)
            stage_seen={};stage_selected={}
            for b in indices:
                for arm in CHAIN_ARMS:
                    mark(stage,batch=b,arm=arm)
                    ledger,identity=restore(rt,last[arm],arm=arm,next_batch=b)
                    result=batch(rt,reference,rt.records[(b-1)*100:b*100],stage=stage,arm_names=(arm,),
                        batch_number=b,ledger=ledger,directory=out/stage/arm/f'B{b}',gate=gate,
                        shadow=(arm=='N4' and b==2),history_cache_directory=out/'history-prefix'/arm)
                    if result['entryM'].shape!=rt.M.shape:raise ValueError('CHAIN_ENTRY_MEMORY_SHAPE')
                    write(result['directory']/'parent-link.json',dict(parent_state_storage='IN_MEMORY_ONLY',
                        parent_identity=identity,entry_W=tensor_sha(result['entry']),entry_M=tensor_sha(result['entryM']),
                        disk_checkpoint=False,exact_crash_resume='NOT_AVAILABLE'))
                    values,allseen=observe_batch(rt,reference,result,w0,batch_number=b,stage=stage,extra_reference=b in (3,5,10))
                    mechanism_report(rt,reference,result,values,w0,result['directory']/'mechanism',technical=validation,
                                     interventions=False)
                    last[arm]=result['commits'][arm]['_state']
                    stage_seen[arm]=allseen[arm];stage_selected[arm]=result['results'][arm].receipt()
                    state['completed'].setdefault(arm,[]).append(b)
                    del result,values,allseen;gc.collect()
            if stage=='S3':
                gate=s3_gate(stage_seen['N4'],stage_seen['DEC_MODES_CUM'],stage_selected['DEC_MODES_CUM'],
                    stage_seen['N4']['W0_correct_NS'],stage_seen['DEC_MODES_CUM']['W0_correct_NS'])
                write(out/'S3-to-S10.json',gate)
                if not gate['pass']:
                    mark('STOPPED_S3_GATE');return finish(out,state,start,gate)
        mark('S10_COMPLETE');return finish(out,state,start,gate)
    except StageResourceHold as exc:
        state['stage']='RESOURCE_HOLD';state['reason']=str(exc)
        return finish(out,state,start,None)


def finish(out,state,start,gate):
    return write(out/'terminal.json',dict(status=state['stage'],completed=state['completed'],
        reason=state.get('reason'),last_gate=gate,source='SEE_EXECUTION_ENTRY_LOCK',
        program_seconds=time.monotonic()-start,peak_host_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_GPU_allocated=torch.cuda.max_memory_allocated(),peak_GPU_reserved=torch.cuda.max_memory_reserved(),
        maximum_batch=state.get('maximum_batch',10),scientific_gates_preserved=True,automatic_slurm_submissions=0,
        checkpoint_saved=False,exact_resume='NOT_AVAILABLE',
        agent_monitoring_required=state.get('monitor_to_completion',False),
        completed_review_requires_USER_recall=not state.get('monitor_to_completion',False),
        sequential_authorized=False if state.get('maximum_batch')==1 else 'HISTORICAL_GATED_PROGRAM'))
