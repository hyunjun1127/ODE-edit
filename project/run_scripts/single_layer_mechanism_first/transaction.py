"""Whole-batch history once and gated in-memory continuation (USER noCP)."""
import copy
from pathlib import Path
import time
import torch
from .config import require_stage
from project.run_scripts.single_layer_edit_preserving_correction.common import tensor_sha, digest, write
from project.run_scripts.single_layer_edit_preserving_correction.sequential_state import state_identity, require_finalizer
from .history import receive_all, registry_status
from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng, restore_rng


def commit(rt, arm, batch, records, weight, entry_memory, ledger, selection, directory, *, stage=None, gate=None):
    require_stage(stage or rt.lock['stage'],arm,batch,gate or rt.lock.get('prior_stage_gate'))
    root=Path(directory)
    if (root/'commit.json').exists():raise ValueError('DUPLICATE_COMMIT')
    start=time.monotonic();saved=rt.W.detach().cpu().clone();old_M=rt.M.clone();rng=capture_rng()
    try:
        rt.copy_weight(weight);memory=entry_memory.clone()
        before=tensor_sha(memory)
        t=time.monotonic()
        history=rt.fitter.finalize(rt.model,rt.tok,rt.requests(records),[(4,rt.hp,memory,rt.P)])
        history_seconds=time.monotonic()-t
        require_finalizer(history,before,tensor_sha(weight),tensor_sha(memory))
        if digest(rng)!=digest(capture_rng()):raise ValueError('HISTORY_RNG_MUTATION')
        # The durable online registry contains received edits, not official
        # paraphrase/neighborhood observer payloads.
        events=[dict(case_id=r['case_id'],requested_rewrite=copy.deepcopy(r['requested_rewrite'])) for r in records]
        received=receive_all(ledger,events)
        identity=state_identity(weight,memory,rng,received,rt.context)
        payload=dict(schema='SL-MECHANISM-FIRST-STATE-v1',arm=arm,batch=batch,next_batch=batch+1,
            weight=weight.cpu().clone(),M4=memory.cpu().clone(),rng=copy.deepcopy(rng),context=copy.deepcopy(rt.context),
            received_ledger=received,registry=registry_status(received),identity=identity,
            sample_order=rt.lock['sample_order'],history=history,selection=selection,
            model_revision=rt.lock['model_revision'],source=rt.lock['execution'],
            teacher_manifest=rt.generated_store.receipt['manifest_sha256'],projector=rt.pmap,
            next_stage_requires_contract_gate=True)
        if state_identity(payload['weight'],payload['M4'],payload['rng'],payload['received_ledger'],payload['context'])!=identity:
            raise ValueError('IN_MEMORY_STATE_IDENTITY')
        rt.copy_weight(payload['weight']);restore_rng(payload['rng']);rt.guard()
        result=dict(status='COMMITTED_IN_MEMORY',arm=arm,batch=batch,checkpoint=None,identity=identity,
            history=history,history_appends=1,candidate_observer_appends=0,
            history_seconds=history_seconds,checkpoint_IO_seconds=0,seconds=time.monotonic()-start,
            disk_checkpoint='SKIPPED_USER_DIRECTED',exact_crash_resume='NOT_AVAILABLE',
            CPU_reload='NOT_TESTED_NO_DISK_CP',physical_reload='EXACT_IN_MEMORY_WEIGHT_COPY',
            GPU_continuation='NOT_YET_TESTED',selection=selection)
        write(root/'commit.json',result)
        result['_state']=payload
        return result
    finally:
        rt.M.copy_(old_M);rt.copy_weight(saved);restore_rng(rng)


def restore(rt, cp, *, arm, next_batch):
    """Same-process state identity and actual weight/memory install, not resume."""
    if cp['schema']!='SL-MECHANISM-FIRST-STATE-v1' or cp['arm']!=arm or cp['next_batch']!=next_batch:
        raise ValueError('CHECKPOINT_ARM_NEXT_BATCH')
    if cp['model_revision']!=rt.lock['model_revision'] or cp['sample_order']!=rt.lock['sample_order']:
        raise ValueError('CHECKPOINT_MODEL_ORDER')
    identity=state_identity(cp['weight'],cp['M4'],cp['rng'],cp['received_ledger'],cp['context'])
    if identity!=cp['identity'] or cp['context']!=rt.context or cp['projector']!=rt.pmap:
        raise ValueError('CHECKPOINT_STATE_BINDING')
    if cp['teacher_manifest']!=rt.generated_store.receipt['manifest_sha256']:
        raise ValueError('CHECKPOINT_TEACHER')
    rt.M.copy_(cp['M4']);rt.copy_weight(cp['weight']);restore_rng(cp['rng']);rt.guard()
    return copy.deepcopy(cp['received_ledger']),identity


def clone_b1_cum_as_step(cp):
    """B1 STEP=CUM identity alias, independent later state; no extra fitting."""
    if cp['arm']!='DEC_MODES_CUM' or cp['batch']!=1 or cp['next_batch']!=2:
        raise ValueError('ONLY_B1_STEP_CUM_ALIAS')
    clone=copy.deepcopy(cp)
    clone.update(arm='DEC_MODES_STEP',state_clone_of=cp['identity'],new_fit_calls=0)
    return clone
