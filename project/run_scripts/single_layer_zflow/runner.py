"""Single approved W0/cold-M0 B100x10 SL-ZFlow chain, with durable resumes."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import uuid
import torch
from .technical import load_inputs
from .runtime import (load_model,prepare_batch,run_flow,save,capture_rng,restore_rng,
                      file_sha,digest,check_parity,state_fingerprint)
from .native_binding import select_l4_projector,build_training_sequences
from .llama_adapter import WEIGHT,model_guard
from .durable import CheckpointStore,prepare_state,tensor_sha256
from .transaction import materialized_cost


def atomic_json(path,value):
    """Same-filesystem atomic create-once JSON; partial temps remain evidence."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.partial-',dir=path.parent)
    with os.fdopen(fd,'w') as handle:
        json.dump(value,handle,sort_keys=True,indent=2,ensure_ascii=True,allow_nan=False)
        handle.flush();os.fsync(handle.fileno())
    os.link(name,path)  # Atomic no-replace. Never overwrite an existing result.
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(directory);Path(name).unlink();os.fsync(directory)
    finally:os.close(directory)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha(path))


def next_private_attempt(root,prefix):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    index=1
    while True:
        path=root/f'{prefix}-r{index:04d}'
        try:path.mkdir(mode=0o700);return path
        except FileExistsError:index+=1


def check_resume_latest(store,requested):
    ids=sorted(p.name for p in store.root.iterdir() if re.fullmatch(r'B\d{3}',p.name))
    if requested=='W0':
        if ids:raise RuntimeError('OLDER_RESUME_WITH_NEWER_COMMITTED_CHECKPOINT')
        return
    if not ids or requested!=ids[-1]:raise RuntimeError('OLDER_RESUME_WITH_NEWER_COMMITTED_CHECKPOINT')
    if ids!=[f'B{i:03d}' for i in range(1,int(requested[1:])+1)]:
        raise RuntimeError('COMMITTED_CHECKPOINT_PREFIX_GAP')


def observe(model,tok,records,history,path,*,state_binding):
    from project.run_scripts.blue_alphaedit_sequential_comparison.evaluation import evaluate
    begin=time.perf_counter();before=model_guard(model)
    weights={WEIGHT:dict(model.named_parameters())[WEIGHT]}
    output=evaluate(model,tok,records,weights,history,full=True)
    if model_guard(model)!=before:raise RuntimeError('EVALUATION_MODEL_MUTATION')
    for name,n in [('RS',len(records)),('PS',2*len(records)),('NS',10*len(records))]:
        if output['metrics'][name]['denominator']!=n:raise RuntimeError('CANONICAL_EVAL_DENOMINATOR')
    output.update(evaluation_seconds=time.perf_counter()-begin,evaluator_microbatch=16,
        label='CURRENT_B100' if len(records)==100 else 'FIXED_SEEN_PREFIX_FINAL_STATE',
        new_forward=True,state_binding=state_binding)
    save(path,output)
    return {k:{f:v[f] for f in ('numerator','denominator','rate')} for k,v in output['metrics'].items()},output['evaluation_seconds']


def observation_binding(receipt,source,kind,records):
    return dict(checkpoint_payload_sha256=receipt['payload_sha256'],
        checkpoint_receipt_sha256=receipt['receipt_sha256'],batch_id=receipt['batch_id'],
        weight_sha256=receipt['weight_sha256'],history_sha256=receipt['history_sha256'],
        source=source,kind=kind,request_count=len(records),
        request_order=digest([r['case_id'] for r in records]),
        request_ids=[r['case_id'] for r in records],evaluator_microbatch=16)


def validate_observation(path,binding):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise RuntimeError('OBSERVATION_NOT_REGULAR')
    raw=json.loads(path.read_text())
    if raw.get('state_binding')!=binding or raw.get('request_order')!=binding['request_order']:
        raise RuntimeError('OBSERVATION_STATE_REQUEST_BINDING')
    if raw.get('before_after_exact') is not True or raw.get('evaluator_controller_influence')!=0:
        raise RuntimeError('OBSERVATION_NONMUTATION_EVIDENCE')
    summary={}
    for kind,multiplier in [('RS',1),('PS',2),('NS',10)]:
        metric=raw['metrics'][kind];rows=metric['rows'];den=binding['request_count']*multiplier
        if len(rows)!=den or metric['denominator']!=den or len({r['identity'] for r in rows})!=den:
            raise RuntimeError('OBSERVATION_DENOMINATOR_OR_DUPLICATE')
        if [(r['case_id'],r['prompt_index']) for r in rows] != [(case,index) for case in binding['request_ids'] for index in range(multiplier)]:
            raise RuntimeError('OBSERVATION_CASE_PROMPT_ORDER')
        for row in rows:
            if not math.isfinite(row['new_nll']) or not math.isfinite(row['true_nll']):
                raise RuntimeError('OBSERVATION_NONFINITE')
            success=row['true_nll']<row['new_nll'] if kind=='NS' else row['new_nll']<row['true_nll']
            if row['success']!=success:raise RuntimeError('OBSERVATION_NLL_SUCCESS')
        numerator=sum(r['success'] for r in rows)
        if metric['numerator']!=numerator or metric['rate']!=numerator/den:
            raise RuntimeError('OBSERVATION_NUMERATOR_RATE')
        summary[kind]={key:metric[key] for key in ('numerator','denominator','rate')}
    return raw,summary


def ensure_observation(batch_root,kind,records,receipt,source,model,tok,history,
                       *,observer=observe,derive=None,crash_hook=None):
    """Reuse only a full-hash/binding verified result; preserve partial attempts.

    Raw written before a crash but not yet registered can be independently
    validated and registered without another forward. A corrupt sealed registry
    is a blocker, never a reason to silently replace a published result.
    """
    batch_root=Path(batch_root);binding=observation_binding(receipt,source,kind,records)
    registry=batch_root/'observation-registry'/kind/'000001.json'
    if registry.exists():
        record=json.loads(registry.read_text());path=batch_root/record['relative_path']
        if not path.resolve().is_relative_to(batch_root.resolve()):raise RuntimeError('OBSERVATION_PATH_ESCAPE')
        if (record['binding']!=binding or path.stat().st_size!=record['bytes'] or file_sha(path)!=record['sha256']):
            raise RuntimeError('SEALED_OBSERVATION_CORRUPT')
        raw,metrics=validate_observation(path,binding)
        return dict(record,raw=raw,metrics=metrics,reused=True)
    path=None;performed=False
    for attempt in sorted((batch_root/'observation-attempts').glob(kind+'-r*')):
        try:validate_observation(attempt/'result.json',binding)
        except (OSError,ValueError,KeyError,TypeError,RuntimeError):continue
        path=attempt/'result.json';break
    if path is None:
        attempt=next_private_attempt(batch_root/'observation-attempts',kind);path=attempt/'result.json'
        if derive is None:
            observer(model,tok,records,history,path,state_binding=binding);performed=True
        else:save(path,derive(binding))
        if crash_hook:crash_hook('after_observation_result')
    raw,metrics=validate_observation(path,binding)
    record=dict(kind=kind,binding=binding,relative_path=str(path.relative_to(batch_root)),
        bytes=path.stat().st_size,sha256=file_sha(path),evaluation_seconds=raw.get('evaluation_seconds'),
        new_forward=raw.get('new_forward',True),metrics=metrics)
    atomic_json(registry,record)
    if crash_hook:crash_hook('after_observation_registry')
    return dict(record,raw=raw,reused=not performed)


def reconcile_observations(root,batch_index,receipt,source,records,model,tok,history,
                           *,observer=observe,timing=None,crash_hook=None):
    """Finish required observations of THIS committed state before advancing."""
    root=Path(root);batch_root=root/f'B{batch_index:03d}';batch_root.mkdir(parents=True,exist_ok=True)
    current=records[(batch_index-1)*100:batch_index*100];artifacts={}
    artifacts['current']=ensure_observation(batch_root,'current',current,receipt,source,model,tok,history,
                                            observer=observer,crash_hook=crash_hook)
    if batch_index in (5,10):
        artifacts['seen-full']=ensure_observation(batch_root,'seen-full',records[:batch_index*100],receipt,
            source,model,tok,history,observer=observer,crash_hook=crash_hook)
    if batch_index==10:
        full=artifacts['seen-full'];ids={r['case_id'] for r in records[:500]}
        def derive(binding):
            metrics={}
            for kind,data in full['raw']['metrics'].items():
                rows=[r for r in data['rows'] if r['case_id'] in ids];n=sum(r['success'] for r in rows)
                metrics[kind]=dict(rows=rows,numerator=n,denominator=len(rows),rate=n/len(rows))
            return dict(metrics=metrics,source_full_sha256=full['sha256'],new_forward=False,evaluation_seconds=0.,
                state_binding=binding,request_order=binding['request_order'],before_after_exact=True,
                evaluator_controller_influence=0,weight_state=full['raw']['weight_state'],
                cache_sha256=full['raw']['cache_sha256'],label='W10_SAME_FIRST500_SUBSET')
        artifacts['first500']=ensure_observation(batch_root,'first500',records[:500],receipt,source,
            model,tok,history,observer=observer,derive=derive,crash_hook=crash_hook)
    path=batch_root/'COMPLETE.json'
    if path.exists():
        prior=json.loads(path.read_text())
        if prior['checkpoint']['payload_sha256']!=receipt['payload_sha256']:
            raise RuntimeError('BATCH_COMPLETE_CHECKPOINT_CONFLICT')
        if any(prior['artifacts'][kind]['sha256']!=artifact['sha256'] for kind,artifact in artifacts.items()):
            raise RuntimeError('BATCH_COMPLETE_OBSERVATION_CONFLICT')
        return prior
    loaded=CheckpointStore(root/'checkpoints').load(receipt['batch_id'])
    ledger=loaded['metadata']['ledger'];flow=ledger['flow'];prep=ledger['preparation']
    timing={} if timing is None else timing
    if 'peak_reader' in timing:
        timing=dict(timing,**timing['peak_reader']())
    summary=dict(batch=receipt['batch_id'],batch_index=batch_index,request_count=100,
        metrics=artifacts['current']['metrics'],solver_status=flow['status'],accepted=flow['accepted'],
        rejected=flow['rejected'],oracle_calls=flow['oracle_calls'],history_append=receipt['history_append'],
        flow_seconds=flow['flow_seconds'],preparation_seconds=prep['total_preparation_seconds'],
        commit_seconds=timing.get('commit_seconds'),evaluation_seconds=sum(a['evaluation_seconds'] or 0 for a in artifacts.values()),
        total_seconds=(time.perf_counter()-timing['batch_begin']) if 'batch_begin' in timing else None,
        checkpoint=receipt,source=source,recovery=timing.get('recovery',True),
        peak_gpu_allocated=timing.get('peak_gpu_allocated'),peak_gpu_reserved=timing.get('peak_gpu_reserved'),
        missing_interrupted_timing='NOT_RECORDED' if timing.get('recovery',True) else None,
        artifacts={kind:{k:v for k,v in artifact.items() if k not in ('raw','reused')}
                   for kind,artifact in artifacts.items()})
    atomic_json(path,summary)
    return summary


def validate_completed_prefix(root,end,source,records):
    """Read-only verification of already observed states; never fill using Wlater."""
    root=Path(root)
    for index in range(1,end+1):
        batch_root=root/f'B{index:03d}';path=batch_root/'COMPLETE.json'
        if not path.is_file():raise RuntimeError('EARLIER_COMMITTED_OBSERVATION_INCOMPLETE')
        complete=json.loads(path.read_text());receipt=complete['checkpoint']
        if complete['source']!=source or receipt['batch_id']!=f'B{index:03d}':
            raise RuntimeError('COMPLETED_PREFIX_SOURCE_STATE_MISMATCH')
        if file_sha(root/'checkpoints'/receipt['batch_id']/'manifest.json')!=receipt['manifest_sha256']:
            raise RuntimeError('COMPLETED_PREFIX_CHECKPOINT_MANIFEST_MISMATCH')
        inventories={'current':records[(index-1)*100:index*100]}
        if index in (5,10):inventories['seen-full']=records[:index*100]
        if index==10:inventories['first500']=records[:500]
        for kind,selected in inventories.items():
            if not (batch_root/'observation-registry'/kind/'000001.json').is_file():
                raise RuntimeError('EARLIER_COMMITTED_OBSERVATION_INCOMPLETE')
            def no_forward(*args,**kwargs):raise RuntimeError('HISTORICAL_OBSERVATION_REQUIRES_EXACT_OLD_STATE')
            artifact=ensure_observation(batch_root,kind,selected,receipt,source,None,None,None,observer=no_forward)
            if complete['artifacts'][kind]['sha256']!=artifact['sha256']:
                raise RuntimeError('COMPLETED_PREFIX_OBSERVATION_MISMATCH')


def run(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    lock,records,contexts,contract=load_inputs(args.lock)
    technical=Path(args.technical)
    gate=json.loads((technical/'TECHNICAL_VALID.json').read_text())
    if gate['status']!='ACTUAL_LLAMA_TECHNICAL_VALID' or gate['source_input_lock_sha256']!=file_sha(args.lock):
        raise RuntimeError('ACTUAL_TECHNICAL_GATE_SOURCE_MISMATCH')
    tolerance=json.loads((technical/'parity-tolerance.json').read_text())
    if file_sha(technical/'parity-tolerance.json')!=gate['tolerance_sha256']:
        raise RuntimeError('NUMERICAL_TOLERANCE_LOCK_MISMATCH')
    if not args.resume and (root/'runtime.json').exists():raise FileExistsError('MAIN_ATTEMPT_CREATE_ONCE')
    store=CheckpointStore(root/'checkpoints')
    if args.resume:check_resume_latest(store,args.resume)
    begin=time.perf_counter();model,tok,evaltok,runtime=load_model(lock)
    stack=torch.load(lock['projector_path'],map_location='cpu',weights_only=True,mmap=True)
    projector,pbinding=select_l4_projector(stack);projector=projector.clone();del stack
    weight=dict(model.named_parameters())[WEIGHT];history=torch.zeros_like(projector)
    initial_other=tuple(v for v in model_guard(model) if v[0]!=WEIGHT)
    parent=None;start=1
    source=dict(input_lock_sha256=file_sha(args.lock),source_head=lock['source_head'],source_tree=lock['source_tree'])
    if args.resume and args.resume!='W0':
        loaded=store.load(args.resume)
        if loaded['metadata']['config']!=contract or loaded['metadata']['source']!=source or loaded['metadata']['context']!=contexts:
            raise RuntimeError('RESUME_CONFIG_SOURCE_CONTEXT_CONFLICT')
        committed_index=loaded['metadata']['batch_index']
        old_records=records[(committed_index-1)*100:committed_index*100]
        restored_binding=build_training_sequences(tok,[dict(r['requested_rewrite'],case_id=r['case_id']) for r in old_records],contexts)
        if digest(restored_binding.metadata)!=loaded['metadata']['cache_resume_fingerprint']:
            raise RuntimeError('RESUME_CACHE_BINDING_FINGERPRINT')
        with torch.no_grad():weight.copy_(loaded['tensors']['W'].cuda())
        history=loaded['tensors']['M'];restore_rng(loaded['rng'])
        if state_fingerprint(capture_rng())!=state_fingerprint(loaded['rng']):
            raise RuntimeError('RESUME_RNG_STATE_MISMATCH')
        parent=loaded['receipt'];start=loaded['metadata']['next_batch_index']
        if tensor_sha256(weight)!=parent['weight_sha256'] or tensor_sha256(history)!=parent['history_sha256']:
            raise RuntimeError('RESUME_PHYSICAL_W_M_MISMATCH')
        validate_completed_prefix(root,committed_index-1,source,records)
        reconcile_observations(root,committed_index,parent,source,records,model,evaltok,history)
        restore_rng(loaded['rng'])  # Observer is outside the edit RNG trajectory.
        atomic_json(root/'resume-events'/f'{args.resume}-{uuid.uuid4().hex}.json',dict(parent=parent,next_batch_index=start,
            W_M_RNG_context_complete=True,latest_observations_reconciled=True,runtime=runtime))
        del loaded,restored_binding
    elif args.resume=='W0':
        atomic_json(root/'resume-events'/f'W0-{uuid.uuid4().hex}.json',dict(parent=None,next_batch_index=1,
            fresh_pretrained_W0_cold_M0=True,runtime=runtime))
    else:
        atomic_json(root/'runtime.json',dict(**runtime,fresh_pretrained_W0_cold_M0=True,
            technical_state_carried=False,seed=20260907,P=pbinding,source=source,
            sample_unique_requests=1000,scientific_batches=10,conditional_N4_run=False))
    completed=[]
    for batch_index in range(start,11):
        batch=f'B{batch_index:03d}';out=next_private_attempt(root/batch/'edit-attempts','attempt')
        batch_begin=time.perf_counter();current=records[(batch_index-1)*100:batch_index*100]
        entry=weight.detach().cpu().clone();h_identity=(history.data_ptr(),history._version)
        save(out/'entry.json',dict(batch_index=batch_index,request_ids=[r['case_id'] for r in current],
            weight_sha256=tensor_sha256(entry),history_sha256=tensor_sha256(history),
            parent=parent,teacher='own batch-entry essence',native_z_calls=0))
        binding,keys,b,s,geom,oracle,prep=prepare_batch(model,tok,current,contexts,projector,history)
        save(out/'preparation.json',prep)
        result,flow=run_flow(oracle,geom,contract)
        save(out/'flow.json',flow)
        if h_identity!=(history.data_ptr(),history._version) or not torch.equal(weight.cpu(),entry):
            raise RuntimeError('INNER_WEIGHT_HISTORY_MUTATION')
        commit_begin=time.perf_counter()
        candidate=(entry.cuda()+result.x@b).cpu() if result.accepted_steps else entry.clone()
        parity=check_parity(oracle.physical_terminal_parity(result.x,candidate),tolerance)
        save(out/'physical-parity.json',parity)
        if not parity['passed']:raise RuntimeError('COMMIT_PARITY_FAIL')
        actual_cost_begin=time.perf_counter()
        actual=materialized_cost(candidate.double()-entry.double(),history,request_count=100)
        actual_cost_seconds=time.perf_counter()-actual_cost_begin
        predicted=geom.cost(result.x);relative=abs(actual-predicted)/max(actual,predicted,1e-30)
        cost=dict(actual_cost=actual,predicted_cost=predicted,relative_error=relative,
             tolerance=tolerance['cost_relative'],passed=relative<=tolerance['cost_relative'],
             actual_delta_norm=float((candidate.double()-entry.double()).norm()),explicit_cost_seconds=actual_cost_seconds)
        save(out/'physical-cost.json',cost)
        if not cost['passed']:raise RuntimeError('COMMIT_COST_FAIL')
        prepare_commit_begin=time.perf_counter()
        state=prepare_state(entry,history,candidate,result.x.cpu(),b.cpu(),s.cpu(),keys,
              accepted=result.accepted_steps,request_count=100,parity_evidence=parity,cost_evidence=cost)
        prepare_commit_seconds=time.perf_counter()-prepare_commit_begin
        checkpoint_rng=capture_rng()
        parent=store.publish(batch,parent,batch_index+1,state,config=contract,source=source,
            context=contexts,rng=checkpoint_rng,ledger=dict(preparation=prep,flow=flow,
                edit_attempt=str(out.relative_to(root)),explicit_cost_seconds=actual_cost_seconds,
                prepare_commit_seconds=prepare_commit_seconds,physical_cost_evaluations=2 if result.accepted_steps else 1),
            cache_resume_fingerprint=digest(binding.metadata))
        # Published complete W/M is authority; physical model follows it only
        # after publication. A crash here resumes the same committed payload.
        with torch.no_grad():weight.copy_(state.tensors['W'].cuda())
        history=state.tensors['M']
        if tensor_sha256(weight)!=parent['weight_sha256'] or tensor_sha256(history)!=parent['history_sha256']:
            raise RuntimeError('PUBLISHED_PHYSICAL_COMMIT_MISMATCH')
        if initial_other!=tuple(v for v in model_guard(model) if v[0]!=WEIGHT):
            raise RuntimeError('NONSELECTED_PRETRAINED_PARAMETER_MUTATION')
        commit_seconds=time.perf_counter()-commit_begin
        atomic_json(out/'commit.json',dict(receipt=parent,seconds=commit_seconds,
              no_update=result.accepted_steps==0,history_append=parent['history_append']))
        del oracle,geom,b,s,keys,binding,result,state,entry,candidate
        gc.collect();torch.cuda.empty_cache()
        summary=reconcile_observations(root,batch_index,parent,source,records,model,evaltok,history,
            timing=dict(commit_seconds=commit_seconds,batch_begin=batch_begin,recovery=False,
                peak_reader=lambda:dict(peak_gpu_allocated=torch.cuda.max_memory_allocated(),
                                        peak_gpu_reserved=torch.cuda.max_memory_reserved())))
        restore_rng(checkpoint_rng)
        completed.append(summary)
        print(json.dumps(dict(event='BATCH_COMPLETE',batch=batch,metrics=summary['metrics'],
              oracle_calls=flow['oracle_calls'],accepted=flow['accepted'],seconds=summary['total_seconds'])),flush=True)
    # Reconstruct the entire observed chain, including pre-resume completed nodes.
    validate_completed_prefix(root,10,source,records)
    all_batches=[json.loads((root/f'B{i:03d}/COMPLETE.json').read_text()) for i in range(1,11)]
    if (root/'TERMINAL.json').exists():
        prior=json.loads((root/'TERMINAL.json').read_text())
        if prior['source']!=source or prior['final_checkpoint']['payload_sha256']!=parent['payload_sha256']:
            raise RuntimeError('TERMINAL_IDENTITY_CONFLICT')
        return
    atomic_json(root/'TERMINAL.json',dict(status='SEQ1000_COMPLETED',source=source,
        batches=10,unique_requests=1000,attempted_requests=1000,
        history_appends=sum(r['history_append'] for r in all_batches),
        no_update_batches=sum(r['accepted']==0 for r in all_batches),
        accepted=sum(r['accepted'] for r in all_batches),rejected=sum(r['rejected'] for r in all_batches),
        oracle_calls=sum(r['oracle_calls'] for r in all_batches),
        latest_process_seconds=time.perf_counter()-begin,completed=all_batches,
        final_checkpoint=parent,scientific_promotion=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--technical',required=True)
    p.add_argument('--output',required=True);p.add_argument('--resume',help='Latest Bnnn, or W0 only if no checkpoint exists');args=p.parse_args()
    try:run(args)
    except Exception as error:
        atomic_json(Path(args.output)/'failure-events'/f'{uuid.uuid4().hex}.json',
             dict(status='TECHNICAL_FAILURE',exception=type(error).__name__,message=str(error),
                  no_native_fallback=True,failed_requests_not_excluded=True))
        raise

if __name__=='__main__':main()
