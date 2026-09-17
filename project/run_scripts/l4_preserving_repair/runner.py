"""Two approved persistent cold chains. No checkpoints per user override."""
import argparse
import json
from pathlib import Path
import resource
import time
import traceback
from .common import ROOT,ARMS,verify,save,tensor_save,identity,digest,Timer

def native_evidence(fit):
    # No endpoint weight payload / M / RNG snapshot. The actual tensor stays in RAM.
    return {k:v for k,v in fit.items() if k!='weight'}

def run(lock_path,arm,resource_path):
    import torch
    from .runtime import Runtime,capture_rng,restore_rng,tensor_sha
    from .engine import build,select,materialize
    from project.run_scripts.local_z_adaptive_allocation.policy import past_indices
    from project.run_scripts.low_cost_write_donor_pilot.evaluation import counterfact
    from project.run_scripts.baseline_mechanism_first.performance_schema import subset
    lock=json.loads(Path(lock_path).read_text());verify(lock);assert arm in ARMS
    ready=json.loads(Path(lock['common_ready']).read_text())
    assert ready['status']=='TECHNICAL_READY' and ready['execution_lock_sha256']==identity(lock_path)['sha256']
    resource_lock=json.loads(Path(resource_path).read_text())
    assert resource_lock['technical_ready']==identity(lock['common_ready']) and resource_lock['arm']==arm
    common=json.loads(Path(lock['cold_capsule']['path']).read_text())
    out=ROOT/'twoarms'/arm/'attempt-v1'/'output';out.mkdir(parents=True,exist_ok=False)
    begin=time.monotonic();stage='LOAD';rt=None;rollback=None;commits=[];received=[];timing={}
    try:
        rt=Runtime(lock,common);records=rt.records;previous=rt.state()
        save(out/'start.json',dict(arm=arm,lock=identity(lock_path),resource=identity(resource_path),entry=previous,
            M4_exactzero=bool(torch.count_nonzero(rt.M4)==0),checkpoint='SKIPPED_USER_DIRECTED',
            exact_restart='NOT_AVAILABLE',W0_reference=lock['W0_reference']))
        for batch in range(1,11):
            stage=f'B{batch:03d}';folder=out/stage;folder.mkdir()
            assert rt.state()==previous,'COMMIT_NEXT_ENTRY_LINK'
            cur=records[(batch-1)*100:batch*100];rollback=rt.snapshot();entry=rt.state()
            save(folder/'entry.json',dict(batch=batch,state=entry,received=received,next_ordinal=(batch-1)*100,
                records_digest=digest(cur),previous_commit=commits[-1] if commits else None))
            if batch==2:save(out/'initial-gate.json',dict(status='INITIAL_REPAIR_EXECUTION_OBSERVED',
                B1_commit=commits[-1],B2_entry=entry,next_ordinal=100,full1000_complete=False,
                checkpoint='SKIPPED_USER_DIRECTED',GPU_off_on='NOT_TESTED'))
            past_ord=past_indices(received,cur,records);past=[records[i] for i in past_ord]
            save(folder/'past64.json',dict(ordinals=past_ord,case_ids=[r['case_id'] for r in past],received_only=True))
            with Timer(timing,'entry_online_observation'):
                save(folder/'entry-rewrite.json',rt.observe(lambda:rt.response.panel(cur)[0]))
                if past:save(folder/'entry-past.json',rt.observe(lambda:rt.response.panel(past)[0]))
                save(folder/'entry-base.json',rt.observe(lambda:rt.response.base()[0]))
            fit=rt.fit(cur);tensor_save(folder/'native-targets.pt',native_evidence(fit))
            WN={l:w.detach().cpu().clone() for l,w in rt.W.items()}
            restore_rng(rollback['rng']);native_state=rt.state()
            with Timer(timing,'gradients_response_inclusive'):
                model=build(rt,cur,past,arm,folder)
            with Timer(timing,'QP_candidates_inclusive'):
                selection,selected8=select(rt,cur,past,model,folder)
            assert (folder/'selection.json').is_file(),'OBSERVER_BEFORE_SELECTION_SEAL'
            selectedW={4:WN[4],8:selected8};observations={}
            def observe_cf(weights,population,label):
                rt.apply_weights(weights,rollback['rng'])
                key=(tuple(tensor_sha(weights[l]) for l in (4,8)),digest(population))
                if key in observations:return observations[key]
                value=rt.observe(lambda:counterfact(rt.model,rt.etok,population,panel=label))
                observations[key]=value;return value
            with Timer(timing,'sealed_observers'):
                save(folder/'entry-current.json',observe_cf(rollback['W'],cur,'ENTRY_CURRENT'))
                save(folder/'native-current.json',observe_cf(WN,cur,'OWN_NATIVE_CURRENT'))
                if batch in (1,5,10):
                    early=records[:100]
                    for probe in selection['probes']:
                        # Only endpoints actually evaluated during acceptance.
                        if probe['actual'] is None:continue
                        pw={4:WN[4],8:materialize(model['anchor8'],model['Q'],probe['c'])}
                        assert tensor_sha(pw[8])==probe['actual_weight_sha']
                        target=folder/f"probe-{probe['index']:02d}"
                        save(target/'observer-current.json',observe_cf(pw,cur,'POST_SEAL_CANDIDATE_CURRENT'))
                        save(target/'observer-early.json',observe_cf(pw,early,'POST_SEAL_FIRST100'))
                if batch in (5,10):
                    seen=records[:batch*100];full=observe_cf(selectedW,seen,f'W{batch}_ALL')
                    save(folder/'seen-full.json',full)
                    save(folder/'selected-current.json',subset(full,(batch-1)*100,batch*100,seen))
                    if batch==10:
                        save(folder/'first500.json',subset(full,0,500,seen));save(folder/'last500.json',subset(full,500,1000,seen))
                    rt.apply_weights(selectedW,rollback['rng'])
                    save(folder/'Dev128.json',rt.observe(lambda:rt.response.base('Dev128')[0]))
                else:save(folder/'selected-current.json',observe_cf(selectedW,cur,'SELECTED_CURRENT'))
            rt.apply_weights(selectedW,rollback['rng'])
            assert tensor_sha(rt.W[4])==native_state['W']['4']
            with Timer(timing,'history_once'):
                history=rt.finalize(cur)
            restore_rng(rollback['rng']);rt.guard();received.extend(range((batch-1)*100,batch*100))
            state=rt.state()
            commit=save(folder/'commit.json',dict(batch=batch,entry=entry,native=native_state,state=state,
                selection=identity(folder/'selection.json'),history=history,M8_append=0,inner_append=0,
                received_count=len(received),received=received,next_ordinal=batch*100,next_batch=batch+1,
                checkpoint='SKIPPED_USER_DIRECTED',tensor_payload_W_M_RNG=False,exact_restart='NOT_AVAILABLE',
                source=lock['source_head'],model_revision=lock['model_revision'],common=lock['cold_capsule'],
                actual_selected_W8_sha=selection['selected_W8_sha'],native_receipt=fit['receipt']))
            commits.append(commit);previous=state;rollback=None
            del fit,WN,model,selectedW,selected8,observations
            print('COMMITTED',arm,batch,selection['selected'],flush=True)
        save(out/'terminal.json',dict(status='COMPLETED_1000_REQUESTS',arm=arm,batches=10,requests=1000,commits=commits,
            terminal_state=rt.state(),history_counts={'4':10,'8':0},seconds=time.monotonic()-begin,
            timers=dict(runtime=rt.timing,components=timing,nested_sum_forbidden=True),
            peak_GPU_allocated=torch.cuda.max_memory_allocated(),peak_GPU_reserved=torch.cuda.max_memory_reserved(),
            peak_host_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            checkpoint='SKIPPED_USER_DIRECTED',exact_restart='NOT_AVAILABLE',automatic_followup=False))
    except BaseException as e:
        restored=False
        if rt is not None and rollback is not None:
            try:rt.restore(rollback);restored=True
            except BaseException:pass
        save(out/'failure.json',dict(status='TECHNICAL_FAILURE',stage=stage,error=repr(e),traceback=traceback.format_exc(),
            exception_receipt=getattr(e,'receipt',None),commits=commits,completed_batches=len(commits),
            open_batch_rollback=restored,seconds=time.monotonic()-begin,exact_restart='NOT_AVAILABLE'))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--resource-lock',required=True);a=p.parse_args();run(a.lock,a.arm,a.resource_lock)
