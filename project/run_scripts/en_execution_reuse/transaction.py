"""B1-only final history1 and create-once checkpoints; no continuation API."""
import copy
from pathlib import Path
import time
import torch
from .config import ARMS
from .model import require_lock
from .preparation import create_json
from project.run_scripts.single_layer_edit_preserving_correction.common import tensor_sha, digest
from project.run_scripts.single_layer_edit_preserving_correction.sequential_state import (
    atomic_tensor, receive, registry, require_finalizer)
from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng, restore_rng


def _preflight(rt,arm,directory):
    require_lock(rt.lock)
    directory=Path(directory); checkpoint=directory/'checkpoint.pt'
    if (rt.lock['stage']!='MATCHED_B1' or arm not in ('N4',*ARMS) or
        checkpoint.exists() or (directory/'commit.json').exists()):
        raise ValueError('DUPLICATE_OR_UNAUTHORIZED_B1_COMMIT')
    if not directory.resolve().is_relative_to(Path(rt.lock['output']).resolve()/'checkpoints'):
        raise ValueError('CHECKPOINT_OUTPUT_SCOPE')


def _commit(rt, arm, weight, selection, teacher_manifest, directory):
    directory=Path(directory);checkpoint=directory/'checkpoint.pt'
    started=time.monotonic()
    rt.copy_weight(weight)
    memory=torch.zeros_like(rt.M)
    before=tensor_sha(memory);rng=capture_rng()
    timer=time.monotonic()
    history=rt.fitter.finalize(rt.model,rt.tok,rt.requests(rt.records),[(4,rt.hp,memory,rt.P)])
    history_seconds=time.monotonic()-timer
    require_finalizer(history,before,tensor_sha(weight),tensor_sha(memory))
    if digest(capture_rng())!=digest(rng):raise ValueError('HISTORY_RNG_MUTATED')
    ledger=receive([],rt.records)
    identity=dict(W=tensor_sha(weight),M=tensor_sha(memory),rng=digest(rng),context=digest(rt.context),
                  ledger=digest(ledger),order=digest(rt.lock['sample_order']))
    payload=dict(schema='EN-EXECUTION-REUSE-B1-v1',arm=arm,batch=1,max_batches=1,
        next_batch=2,sequential_authorized=False,auto_continue=False,weight=weight.detach().cpu(),M4=memory.cpu(),
        rng=rng,context=copy.deepcopy(rt.context),received_ledger=ledger,registry=registry(ledger),
        sample_order=list(rt.lock['sample_order']),
        identity=identity,history=history,selection=selection,teacher_manifest=teacher_manifest,
        source=rt.lock['execution'],model_revision=rt.lock['model_revision'],projector=rt.pmap)
    timer=time.monotonic();saved=atomic_tensor(checkpoint,payload);io_seconds=time.monotonic()-timer
    restored=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
    scalars=lambda p:{k:v for k,v in p.items() if k not in ('weight','M4')}
    if (digest(scalars(restored))!=digest(scalars(payload)) or
        any(restored[k].shape!=payload[k].shape or restored[k].dtype!=torch.float32 or
            not bool(torch.isfinite(restored[k]).all()) for k in ('weight','M4')) or
        tensor_sha(restored['weight'])!=identity['W'] or tensor_sha(restored['M4'])!=identity['M'] or
        digest(restored['context'])!=identity['context'] or digest(restored['received_ledger'])!=identity['ledger'] or
        digest(restored['sample_order'])!=identity['order'] or
        restored['registry']!=registry(restored['received_ledger'])):
        raise ValueError('CHECKPOINT_RELOAD_IDENTITY')
    rt.copy_weight(restored['weight']);restore_rng(restored['rng']);rt.guard()
    if digest(capture_rng())!=identity['rng']:raise ValueError('CHECKPOINT_RNG_RESTORE')
    result=dict(status='B1_COMMITTED',arm=arm,batch=1,identity=identity,history=history,
        history_appends=1,candidate_history_appends=0,checkpoint=saved,
        CPU_reload='SHAPE_FINITE_BYTES_VERIFIED',physical_weight_reload='EXACT_COPY_CHECKED',
        M_resume='CPU_ONLY_NO_NEXT_BATCH_AUTHORITY',next_batch_authorized=False,
        seconds=time.monotonic()-started,history_seconds=history_seconds,checkpoint_IO_seconds=io_seconds)
    create_json(directory/'commit.json',result)
    return result


def commit(rt, arm, weight, selection, teacher_manifest, directory):
    """Failure rollback is runtime-only; partial files/costs remain preserved."""
    _preflight(rt,arm,directory)
    rt.guard()
    entry=rt.W.detach().cpu().clone();memory=rt.M.detach().cpu().clone();rng=capture_rng()
    try:
        return _commit(rt,arm,weight,selection,teacher_manifest,directory)
    except BaseException:
        rt.copy_weight(entry)
        rt.M.copy_(memory);restore_rng(rng);rt.guard()
        raise
