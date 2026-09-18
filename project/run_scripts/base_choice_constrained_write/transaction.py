"""One native history append and durable arm-owned B1 state; never runs B2."""
from pathlib import Path
import copy
import time
import torch
from .config import require_scope
from .provenance import create_json
from project.run_scripts.single_layer_edit_preserving_correction.common import tensor_sha,digest
from project.run_scripts.single_layer_edit_preserving_correction.sequential_state import atomic_tensor,receive,registry,require_finalizer
from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng,restore_rng

def commit(rt,arm,weight,selection,reference_manifest,directory):
    require_scope(rt.lock);directory=Path(directory);cp=directory/'checkpoint.pt'
    if arm not in ('N4','BPCW512') or cp.exists() or (directory/'commit.json').exists():
        raise ValueError('DUPLICATE_OR_UNAUTHORIZED_COMMIT')
    start=time.monotonic();rt.copy_weight(weight);memory=torch.zeros_like(rt.M)
    before=tensor_sha(memory);rng=capture_rng()
    ht=time.monotonic()
    receipts=rt.fitter.finalize(rt.model,rt.tok,rt.requests(rt.records),[(4,rt.hp,memory,rt.P)])
    history_seconds=time.monotonic()-ht
    require_finalizer(receipts,before,tensor_sha(weight),tensor_sha(memory))
    if digest(capture_rng())!=digest(rng):raise ValueError('HISTORY_RNG_MUTATION')
    ledger=receive([],rt.records);identity=dict(W=tensor_sha(weight),M=tensor_sha(memory),rng=digest(rng),
        context=digest(rt.context),ledger=digest(ledger),order=digest(rt.lock['sample_order']))
    payload=dict(schema='BPCW-B1-resume-v1',arm=arm,batch=1,next_batch=2,max_batches=1,
        sequential_authorized=False,weight=weight.detach().cpu(),M4=memory.cpu(),rng=rng,
        context=copy.deepcopy(rt.context),received_ledger=ledger,registry=registry(ledger),identity=identity,
        history=receipts,selection=selection,reference_manifest=reference_manifest,
        source=rt.lock['execution'],model_revision=rt.lock['model_revision'],projector=rt.pmap)
    io=time.monotonic();member=atomic_tensor(cp,payload);checkpoint_io_seconds=time.monotonic()-io
    # Actual physical install after independent CPU checkpoint reload. No next-batch fit.
    loaded=torch.load(cp,weights_only=True,mmap=True,map_location='cpu')
    if (loaded['identity']!=identity or tensor_sha(loaded['M4'])!=identity['M'] or
        tensor_sha(loaded['weight'])!=identity['W'] or loaded['next_batch']!=2):raise ValueError('RESUME_SCHEMA_HASH')
    rt.copy_weight(loaded['weight']);restore_rng(loaded['rng']);rt.guard()
    if digest(capture_rng())!=identity['rng']:raise ValueError('RESUME_RNG_IDENTITY')
    result=dict(status='B1_COMMITTED',arm=arm,batch=1,history=receipts,identity=identity,checkpoint=member,
        next_batch=2,next_batch_execution_authorized=False,history_appends=1,duplicate_guard='checkpoint existence BEFORE finalizer',
        CPU_reload='PASS',physical_weight_reload='PASS',GPU_next_batch_continuation='NOT_RUN_NOT_AUTHORIZED',
        memory_reload='CPU_SCHEMA_BYTES_PASS_NO_MODEL_HISTORY_USE_AFTER_RELOAD',seconds=time.monotonic()-start,
        history_seconds=history_seconds,checkpoint_io_seconds=checkpoint_io_seconds)
    create_json(directory/'commit.json',result);return result
