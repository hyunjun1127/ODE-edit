"""One autonomous cold chain. Finalize once, preserve failure and rollback."""
import argparse
import json
import os
from pathlib import Path
import resource
import time
import traceback
from .common import ARMS,ROOT,save,tensor_save,identity,verify,digest,layers,Timer

def run(lock_path,arm):
    import torch
    from .model import Runtime,tensor_sha,capture_rng,restore_rng
    from .engine import generate,score_and_select
    from .policy import past_indices,fact
    from project.run_scripts.low_cost_write_donor_pilot.evaluation import counterfact
    from project.run_scripts.baseline_mechanism_first.performance_schema import subset
    lock=json.loads(Path(lock_path).read_text());verify(lock);assert arm in ARMS
    arm_lock_path=ROOT/'arms'/arm/'attempt-v1'/'arm.lock.json'
    arm_lock=json.loads(arm_lock_path.read_text())
    assert arm_lock['arm']==arm and arm_lock['execution_lock']==identity(lock_path) and arm_lock['W0_cold'] is True
    if 'SLURM_ARRAY_TASK_ID' in os.environ:assert ARMS[int(os.environ['SLURM_ARRAY_TASK_ID'])]==arm
    ready_path=Path(lock['common_ready']);ready=json.loads(ready_path.read_text())
    assert ready['status']=='TECHNICAL_READY' and ready['execution_lock_sha256']==identity(lock_path)['sha256']
    common=json.loads(Path(ready['capsule']['path']).read_text());assert identity(ready['capsule']['path'])==ready['capsule']
    out=ROOT/'arms'/arm/'attempt-v1'/'output';out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();stage='LOAD';rt=None;rollback=None;commits=[];received=[];previous=None
    try:
        rt=Runtime(lock,common); records=rt.records;previous=rt.state();timing={}
        save(out/'start.json',dict(arm=arm,lock=identity(lock_path),common_ready=identity(ready_path),entry=previous,
            cold_zero=all(torch.count_nonzero(m)==0 for m in rt.M.values()),teacher=rt.teacher.receipt))
        for batch in range(1,11):
            stage=f'B{batch:03d}';bdir=out/stage;bdir.mkdir();cur=records[(batch-1)*100:batch*100]
            assert rt.state()==previous,'COMMIT_NEXT_ENTRY_LINK'
            rollback=rt.snapshot();entry=rt.state();save(bdir/'entry.json',dict(batch=batch,state=entry,received=received,
                record_digest=digest(cur),ordinals=[(batch-1)*100,batch*100],previous_commit=commits[-1] if commits else None))
            if batch==2:
                save(out/'initial-gate.json',dict(status='INITIAL_DYNAMIC_VALID' if arm in ('LD','TD') else 'INITIAL_FIXED_VALID',
                    actual_committed_B1=True,B2_entry=entry,previous_commit=commits[-1],next_ordinal=100,
                    technical_ready=identity(ready_path),full1000_complete=False))
            with Timer(timing,'entry_observer'):
                save(bdir/'entry-current.json',rt.observe(lambda:counterfact(rt.model,rt.etok,cur,panel='ENTRY_CURRENT')))
            past_ord=past_indices(received,cur,records);past=[records[i] for i in past_ord]
            save(bdir/'past64.json',dict(ordinals=past_ord,case_ids=[r['case_id'] for r in past],received_only=True,
                current_overwrite_excluded=True,selection_score_access=0))
            with Timer(timing,'proposal_inclusive'):
                candidates,meta,entry_snap,fits=generate(rt,cur,arm,bdir/'proposals')
            with Timer(timing,'candidate_scoring'):
                selection,rows=score_and_select(rt,cur,past,arm,candidates,meta,entry_snap,bdir)
            selected=selection['selected'];rt.apply(candidates[selected],entry_snap['rng'])
            with Timer(timing,'selected_observer'):
                if batch in (5,10):
                    seen=records[:batch*100];full=rt.observe(lambda:counterfact(rt.model,rt.etok,seen,panel=f'W{batch}_ALL'))
                    save(bdir/'seen-full.json',full)
                    save(bdir/'selected-current.json',subset(full,(batch-1)*100,batch*100,seen))
                    if batch==10:
                        save(bdir/'first500.json',subset(full,0,500,seen));save(bdir/'last500.json',subset(full,500,1000,seen))
                    save(bdir/'Dev128.json',rt.observe(lambda:rt.observer.generic('Dev128')))
                else:save(bdir/'selected-current.json',rt.observe(lambda:counterfact(rt.model,rt.etok,cur,panel='SELECTED_CURRENT')))
            with Timer(timing,'history_finalization'):
                history=rt.fitter.finalize(rt.model,rt.tok,rt.requests(cur),[(l,rt.hp[l],rt.M[l],rt.P[l]) for l in layers(arm)])
            assert [x['layer'] for x in history]==list(layers(arm)) and all(x['history_append']==1 for x in history)
            restore_rng(entry_snap['rng']);rt.guard();received.extend(range((batch-1)*100,batch*100))
            state=rt.state()
            with Timer(timing,'state_IO'):
                delta=tensor_save(bdir/'selected-delta.pt',dict(delta={l:candidates[selected][l]-entry_snap['W'][l] for l in (4,8)},
                    actual_weight_hash=state['W'],entry=entry,selected=selected,FP32_delta_not_exact_replay_claim=True))
                checkpoint=None
                if batch in (1,5,10):
                    checkpoint=tensor_save(bdir/'checkpoint.pt',dict(**rt.snapshot(),received=received,next_batch=batch+1,
                        next_ordinal=batch*100,source=lock['source_head'],model_revision=lock['model_revision'],
                        sample_digest=lock['records_digest'],projector=rt.pmap,common=ready['capsule'],state=state))
                    loaded=torch.load(checkpoint['path'],map_location='cpu',weights_only=True,mmap=True)
                    assert {str(l):tensor_sha(w) for l,w in loaded['W'].items()}==state['W']
                    assert {str(l):tensor_sha(m) for l,m in loaded['M'].items()}==state['M'];del loaded
            commit=save(bdir/'commit.json',dict(batch=batch,selected=selected,state=state,entry=entry,history=history,
                next_ordinal=batch*100,received_count=len(received),delta=delta,checkpoint=checkpoint,
                actual_candidate=meta[selected],checkpoint_CPU_reload=checkpoint is not None,GPU_off_on_replay='NOT_TESTED'))
            commits.append(commit);previous=state;rollback=None
            del fits,candidates,entry_snap
            print('COMMITTED',arm,batch,selected,flush=True)
        save(out/'terminal.json',dict(status='COMPLETED_1000_REQUESTS',arm=arm,batches=10,requests=1000,commits=commits,
            terminal_state=rt.state(),history_counts={str(l):10 if l in layers(arm) else 0 for l in (4,8)},
            seconds=time.monotonic()-started,timers=dict(runtime=rt.timing,components=timing,nested_sum_forbidden=True),
            peak_GPU_allocated=torch.cuda.max_memory_allocated(),peak_GPU_reserved=torch.cuda.max_memory_reserved(),
            peak_host_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,additional_experiments=0))
    except BaseException as e:
        restored=False
        if rt is not None and rollback is not None:
            try:rt.restore(rollback);restored=True
            except BaseException:pass
        save(out/'failure.json',dict(status='TECHNICAL_FAILURE',stage=stage,error=repr(e),traceback=traceback.format_exc(),
            completed_batches=len(commits),commits=commits,open_batch_rollback=restored,seconds=time.monotonic()-started))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--arm',choices=ARMS,required=True)
    a=p.parse_args();run(a.lock,a.arm)
