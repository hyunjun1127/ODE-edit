"""A complete cold ten-batch v2 program; no agent wakeup, no disk W/M state."""
import argparse
import json
import os
from pathlib import Path
import resource
import time
import traceback
from .common import ROOT,ARMS,ARM_LAYERS,LAYERS,Ledger,save,identity,verify,digest,serial,Timer


def run(lock_path,arm):
    import torch
    from .runtime import Runtime,Backend
    from .controller import Controller,run_arm
    from project.run_scripts.local_z_adaptive_allocation.policy import past_indices,event_id
    from project.run_scripts.low_cost_write_donor_pilot.evaluation import counterfact
    from project.run_scripts.baseline_mechanism_first.performance_schema import subset
    lock=json.loads(Path(lock_path).read_text());verify(lock);assert arm in ARMS
    arm_lock_path=ROOT/'arms'/arm/'attempt-v1'/'arm.lock.json'
    arm_lock=json.loads(arm_lock_path.read_text())
    assert arm_lock['arm']==arm and arm_lock['execution_lock']==identity(lock_path) and arm_lock['W0_cold']
    if 'SLURM_ARRAY_TASK_ID' in os.environ:assert ARMS[int(os.environ['SLURM_ARRAY_TASK_ID'])]==arm
    ready_path=Path(lock['common_ready']);ready=json.loads(ready_path.read_text())
    assert ready['status']=='TECHNICAL_READY' and ready['execution_lock_sha256']==identity(lock_path)['sha256']
    assert identity(ready['capsule']['path'])==ready['capsule']
    common=json.loads(Path(ready['capsule']['path']).read_text())
    out=ROOT/'arms'/arm/'attempt-v1'/'output';out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();stage='LOAD';rt=None;rollback=None;commits=[];received=[];timing={}
    try:
        rt=Runtime(lock,common);records=rt.records;previous=rt.state()
        assert all(torch.count_nonzero(m)==0 for m in rt.M.values())
        save(out/'start.json',dict(arm=arm,execution_lock=identity(lock_path),ready=identity(ready_path),entry=previous,
            W0_cold=True,M0='EXACT_ZERO_ALL5',teacher=rt.teacher.receipt,prior_W0_observer=lock['W0_observation'],
            disk_weight_history_checkpoint=False,crash_resume='NOT_AVAILABLE',scientific_scope=1000))
        for batch in range(1,11):
            stage=f'B{batch:03d}';bdir=out/stage;bdir.mkdir();cur=records[(batch-1)*100:batch*100]
            assert rt.state()==previous,'COMMIT_NEXT_ENTRY_LINK'
            rollback=rt.snapshot();entry=rt.state()
            save(bdir/'entry.json',dict(batch=batch,state=entry,received=received,record_digest=digest(cur),
                ordinals=[(batch-1)*100,batch*100],previous_commit=commits[-1] if commits else None))
            if batch==2:
                save(out/'initial-gate.json',dict(status='INITIAL_VALID',arm=arm,actual_committed_B1=True,
                    B2_entry=entry,history_appends_B1=5,previous_commit=commits[-1],next_ordinal=100,
                    technical_ready=identity(ready_path),full1000_complete=False,all_six_actual_PASS=False))
            past_ord=past_indices(received,cur,records);past=[records[i] for i in past_ord]
            save(bdir/'past64.json',dict(ordinals=past_ord,case_ids=[r['case_id'] for r in past],
                stable_event_ids=[event_id(i,records[i]) for i in past_ord],received_only=True,
                current_overwrite_excluded=True,success_filter=False))
            backend=Backend(rt,cur,past,bdir/'episode')
            with Timer(timing,'controller_inclusive'):
                controller=Controller(backend,ARM_LAYERS[arm],namespace=backend.namespace,
                    history_layers=LAYERS,event_sink=Ledger(bdir/'controller-ledger'))
                selected,stop=run_arm(controller,arm,finalize=False)
                controller.seal_selection(selected)
            # Next state is frozen BEFORE any official P/N or Dev observation.
            selected_ref=save(bdir/'selection.json',dict(arm=arm,selected=controller.candidate_receipt(selected),
                stop=stop,counts=controller.counts,candidates=[controller.candidate_receipt(r) for r in controller.records],
                incomplete=controller.incomplete_records,official_observer_access_before_seal=0,
                final_next_state_decided=True,F48_quality_guard_observational_only=arm=='F48'))
            with Timer(timing,'postseal_entry_observer'):
                rt.restore(rollback)
                save(bdir/'entry-current.json',rt.observe(lambda:counterfact(rt.model,rt.etok,cur,panel='ENTRY_CURRENT_AFTER_SEAL')))
            measured={}
            with Timer(timing,'postseal_selected_observer'):
                rt.restore(selected.state)
                if batch in (5,10):
                    seen=records[:batch*100]
                    full=rt.observe(lambda:counterfact(rt.model,rt.etok,seen,panel=f'W{batch}_ALL'))
                    fullref=save(bdir/'seen-full.json',full)
                    current=subset(full,(batch-1)*100,batch*100,seen)
                    if batch==10:
                        save(bdir/'first500.json',subset(full,0,500,seen));save(bdir/'last500.json',subset(full,500,1000,seen))
                    save(bdir/'Dev128.json',rt.observe(lambda:rt.metrics.base('Dev128')))
                else:
                    current=rt.observe(lambda:counterfact(rt.model,rt.etok,cur,panel='SELECTED_CURRENT'))
                currentref=save(bdir/'selected-current.json',current)
                measured[selected.state_token]=currentref
            if batch in (1,5,10):
                with Timer(timing,'postseal_all_completed_candidate_observer'):
                    for i,candidate in enumerate(controller.records):
                        if candidate.scores is None:continue
                        if candidate.state_token in measured:
                            ref=measured[candidate.state_token];reuse=True
                        else:
                            rt.restore(candidate.state)
                            result=rt.observe(lambda:counterfact(rt.model,rt.etok,cur,panel='COMPLETED_CANDIDATE_POSTSEAL'))
                            ref=save(bdir/'candidate-observer'/f'{i:03d}-current.json',result)
                            measured[candidate.state_token]=ref;reuse=False
                        save(bdir/'candidate-observer'/f'{i:03d}-binding.json',dict(gates=candidate.gates,
                            state_token=candidate.state_token,selection_seal=selected_ref,rows=ref,
                            exact_endpoint_reuse=reuse,controller_feedback=0,incomplete_suffix_completed=False))
            with Timer(timing,'finalize_inclusive'):
                controller.finalize(selected)
            assert backend.history_receipt is not None and controller.counts['commits']==1
            received.extend(range((batch-1)*100,batch*100));state=rt.state()
            commit=save(bdir/'commit.json',dict(batch=batch,selected=controller.candidate_receipt(selected),state=state,
                entry=entry,history=backend.history_receipt,history_appends=5,inner_history_appends=0,
                next_ordinal=batch*100,next_batch=batch+1,received=received,received_count=len(received),
                selection=selected_ref,controller_counts=controller.counts,fit_receipts=backend.fit_receipts,
                score_receipts=backend.score_receipts,checkpoint=None,
                memory_continuation='ACTUAL_W_M_CONTEXT_RNG_RECEIVED',disk_crash_resume='NOT_AVAILABLE',
                tensors_independently_reconstructable=False,teacher=ready['capsule'],source=lock['source_head']))
            commits.append(commit);previous=state;rollback=None
            del controller,backend,selected,measured
            print('COMMITTED',arm,batch,flush=True)
        save(out/'terminal.json',dict(status='COMPLETED_1000_REQUESTS',arm=arm,batches=10,requests=1000,
            commits=commits,terminal_state=rt.state(),history_counts={str(l):10 for l in LAYERS},
            seconds=time.monotonic()-started,timers=dict(runtime=rt.timing,components=timing,nested_sum_forbidden=True),
            peak_GPU_allocated=torch.cuda.max_memory_allocated(),peak_GPU_reserved=torch.cuda.max_memory_reserved(),
            peak_host_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            disk_W_M_checkpoints=0,exact_crash_resume='NOT_AVAILABLE',additional_experiments=0))
    except BaseException as exc:
        error=traceback.format_exc();restored=False;restore_error=None
        if rt is not None and rollback is not None:
            try:rt.restore(rollback);restored=True
            except BaseException as re:restore_error=repr(re)
        save(out/'failure.json',dict(status='TECHNICAL_FAILURE',stage=stage,error=repr(exc),traceback=error,
            completed_batches=len(commits),commits=commits,open_batch_rollback=restored,restore_error=restore_error,
            seconds=time.monotonic()-started,no_automatic_retry=True))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--arm',choices=ARMS,required=True)
    a=p.parse_args();run(a.lock,a.arm)
