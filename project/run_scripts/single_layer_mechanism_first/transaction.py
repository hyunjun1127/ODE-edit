"""Durable whole-batch L4 history once, all-event ledger, gated continuation."""
import copy
from pathlib import Path
import time
import torch
from .config import require_stage
from project.run_scripts.single_layer_edit_preserving_correction.common import tensor_sha, digest, write
from project.run_scripts.single_layer_edit_preserving_correction.sequential_state import (
    atomic_tensor, receive, registry, state_identity, require_finalizer)
from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng, restore_rng


def commit(rt, arm, batch, records, weight, entry_memory, ledger, selection, directory):
    require_stage(rt.lock['stage'],arm,batch,rt.lock.get('prior_stage_gate'))
    root=Path(directory)
    if (root/'commit.json').exists() or (root/'checkpoint.pt').exists():raise ValueError('DUPLICATE_COMMIT')
    start=time.monotonic();saved=rt.W.detach().cpu().clone();old_M=rt.M.clone();rng=capture_rng()
    try:
        rt.copy_weight(weight);memory=entry_memory.clone()
        before=tensor_sha(memory)
        t=time.monotonic()
        history=rt.fitter.finalize(rt.model,rt.tok,rt.requests(records),[(4,rt.hp,memory,rt.P)])
        history_seconds=time.monotonic()-t
        require_finalizer(history,before,tensor_sha(weight),tensor_sha(memory))
        if digest(rng)!=digest(capture_rng()):raise ValueError('HISTORY_RNG_MUTATION')
        received=receive(ledger,records)
        identity=state_identity(weight,memory,rng,received,rt.context)
        payload=dict(schema='SL-MECHANISM-FIRST-STATE-v1',arm=arm,batch=batch,next_batch=batch+1,
            weight=weight.cpu(),M4=memory.cpu(),rng=rng,context=copy.deepcopy(rt.context),
            received_ledger=received,registry=registry(received),identity=identity,
            sample_order=rt.lock['sample_order'],history=history,selection=selection,
            model_revision=rt.lock['model_revision'],source=rt.lock['execution'],
            teacher_manifest=rt.generated_store.receipt['manifest_sha256'],projector=rt.pmap,
            next_stage_requires_contract_gate=True)
        t=time.monotonic();cp=atomic_tensor(root/'checkpoint.pt',payload);io=time.monotonic()-t
        check=torch.load(cp['path'],weights_only=True,mmap=True,map_location='cpu')
        if state_identity(check['weight'],check['M4'],check['rng'],check['received_ledger'],check['context'])!=identity:
            raise ValueError('CPU_CP_IDENTITY')
        rt.copy_weight(check['weight']);restore_rng(check['rng']);rt.guard()
        result=dict(status='COMMITTED',arm=arm,batch=batch,checkpoint=cp,identity=identity,
            history=history,history_appends=1,candidate_observer_appends=0,
            history_seconds=history_seconds,checkpoint_IO_seconds=io,seconds=time.monotonic()-start,
            CPU_reload='HASH_SCHEMA_VERIFIED',physical_reload='EXACT_WEIGHT_COPY',
            GPU_continuation='NOT_YET_TESTED',selection=selection)
        write(root/'commit.json',result)
        return result
    finally:
        rt.M.copy_(old_M);rt.copy_weight(saved);restore_rng(rng)
