"""FE/CAKE saved-W20 restore adapter for the single shared CF Flu/Con evaluator.

User 2026-10-11: "FE Flu/Con, CAKE Flu/Con 모두 각각 server1,2에 나눠서 제출".
The evaluator is official.runners.server1.flucon_eval.run, unchanged: FP32 eager
model with TF32 off, NativeGenerationObserver (PROFILE, eval seed 20261007), the
shared reference identity, the ordered CF first 2,000 and the x100 display. As in
official.runners.server2.flucon_eval, only the restore and the host labels are
bound process-locally. Nothing is edited, fitted or generated here.

Checkpoint schemas
* official-baseline-checkpoint-v1 (MEMIT_FE_HISTORY 63382, CAKE 63296/63298/63324):
  the FP32 selected weights are copied exactly.
* native-fe-sequential-diagnostic-v1 (author FE main.py B20.pt, 63385/63389):
  the BF16 selected weights are cast to FP32. Every BF16 value is an FP32 value,
  so the cast is exact; an exact BF16 round trip is checked.
In both cases each payload tensor must hash to its preparation binding, the
restored FP32 tensors must hash to the run's own record when one exists (63382
commit B20), and every other parameter keeps its storage and version. History,
contexts and RNG stored in the checkpoint are not consumed.
"""
import argparse
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path

import torch

from official.experiments.prepare import digest
from official.runners.server1 import flucon_eval as shared
from official.runners.server1.common import read, verify
from official.tracking.schema import config as validate_tracking

AUTHORITY = 'USER-FE-CAKE-FLUCON-S1-S2-20261011-R1'
TASK = 'fe-cake-flucon-20261011'
LOCAL = Path('/mnt/raid5/janghj/ODE-edit/local') / TASK
OFFICIAL = 'official-baseline-checkpoint-v1'
NATIVE = 'native-fe-sequential-diagnostic-v1'
NODES = dict(server1='devbox', server2='server2')
NATIVE_Z = dict(batch_entry='CURRENT_BATCH_ENTRY_NO_FUTURE_TARGETS',
                initial_model_on_arrival='CURRENT_BATCH_ONLY_INITIAL_MODEL')
_tracking = shared.tracking_values


def require(ok, code):
    if not ok:
        raise ValueError(code)


def raw_sha(tensor):
    """SHA-256 of the tensor's own bytes; equals server1 native.tensor_sha for FP32."""
    value = tensor.detach().to('cpu').contiguous()
    return hashlib.sha256(memoryview(value.view(torch.uint8).numpy().reshape(-1))).hexdigest()


def selected_names(hparams):
    names = {hparams['rewrite_module_tmp'].format(layer) + '.weight' for layer in hparams['layers']}
    require(len(names) == len(hparams['layers']) > 0, 'SELECTED_WEIGHT_NAMES')
    return names


def bind_weights(model, tensors, original, stored_dtype):
    """Validate every member before any write; then exact copy, hashes, untouched others."""
    parameters = dict(model.named_parameters())
    names = selected_names(original['hparams'])
    expected = original['payload_tensor_sha256']
    require(set(tensors) == names == set(expected) and names <= parameters.keys(), 'RESTORE_SELECTED_WEIGHT_SET')
    for name in sorted(names):
        value, target = tensors[name], parameters[name]
        require(torch.is_tensor(value) and value.dtype == stored_dtype and target.dtype == torch.float32
                and tuple(value.shape) == tuple(target.shape) and bool(torch.isfinite(value).all()),
                'RESTORE_SHAPE_DTYPE_FINITE')
        require(raw_sha(value) == expected[name], 'RESTORE_PAYLOAD_TENSOR_SHA')
    untouched = {k: (id(v), v.data_ptr(), v._version) for k, v in parameters.items() if k not in names}
    restored = {}
    with torch.no_grad():
        for name in sorted(names):
            value, target = tensors[name], parameters[name]
            exact = value.to(device=target.device, dtype=torch.float32)
            target.copy_(exact)
            require(torch.equal(target, exact), 'RESTORE_EXACT_FP32')
            if stored_dtype != torch.float32:
                require(torch.equal(target.to(stored_dtype), value.to(target.device)), 'RESTORE_EXACT_CAST_ROUNDTRIP')
            del exact
            restored[name] = raw_sha(target)
    require(untouched == {k: (id(v), v.data_ptr(), v._version) for k, v in parameters.items() if k not in names},
            'RESTORE_NONSELECTED_MUTATION')
    recorded = original.get('recorded_selected_weights_sha256')
    if recorded is not None:
        require(restored == recorded, 'RESTORED_NOT_RECORDED_W20')
    return restored


def receipt(original, restored, stored_dtype):
    return dict(checkpoint_schema=original['checkpoint_schema'], method=original['method'],
        source_job_id=original['job_id'], checkpoint=original['checkpoint'], batch=20, edits_in_state=2000,
        restored_selected_weights_fp32_sha256=restored,
        payload_tensor_sha256=original['payload_tensor_sha256'],
        recorded_W20_hash_match=('EXACT' if original.get('recorded_selected_weights_sha256') else 'NO_TENSOR_RECORD_IN_RUN'),
        stored_dtype=str(stored_dtype).replace('torch.', ''),
        cast=('none: FP32 copied exactly' if stored_dtype == torch.float32
              else 'exact BF16->FP32 (round trip verified)'),
        evaluation_precision='FP32 eager, TF32 off (shared evaluator)',
        nonselected_unchanged=True, history_contexts_rng_consumed=False, actual_edit_calls=0)


def restore_official(model, original):
    checkpoint = original['checkpoint']
    pointer = read(verify(original['pointer']))
    require(pointer.get('batch') == 20 and pointer.get('final_W20') is True
            and pointer.get('sha256') == checkpoint['sha256']
            and pointer.get('identity_sha256') == digest(original['identity'])
            and pointer.get('file') == Path(checkpoint['path']).name, 'OFFICIAL_W20_POINTER')
    payload = torch.load(verify(checkpoint), map_location='cpu', mmap=True, weights_only=False)
    require(payload.get('schema') == OFFICIAL and payload.get('batch') == 20
            and payload.get('identity') == original['identity'] and payload.get('method') == original['method'],
            'OFFICIAL_CHECKPOINT_IDENTITY')
    cursor = payload.get('evaluation_cursor') or {}
    if original['method'] == 'CAKE':
        require(cursor.get('batch') == 20 and cursor.get('edits') == 2000, 'CAKE_FINAL_CURSOR')
    else:
        require(cursor.get('completed_batch') == 20
                and (payload.get('contexts') or {}).get('successful_calls') == 20, 'FINAL_NATIVE_CURSOR')
    layers = {str(layer) for layer in original['hparams']['layers']}
    require(type(payload.get('cache_c')) is dict and set(payload['cache_c']) == layers, 'HISTORY_LAYER_IDENTITY')
    restored = bind_weights(model, payload['weights'], original, torch.float32)
    return receipt(original, restored, torch.float32)


def restore_native(model, original):
    checkpoint, identity = original['checkpoint'], original['identity']
    record = read(verify(original['pointer']))
    require(record.get('sha256') == checkpoint['sha256'] and record.get('bytes') == checkpoint['bytes']
            and record.get('completed_batches') == 20 and record.get('completed_edits') == 2000
            and record.get('identity_sha256') == identity['native_identity_sha256'], 'NATIVE_B20_RECEIPT')
    payload = torch.load(verify(checkpoint), map_location='cpu', mmap=True, weights_only=False)
    require(payload.get('schema') == NATIVE and payload.get('identity_sha256') == identity['native_identity_sha256']
            and payload.get('completed_batches') == 20 and payload.get('completed_edits') == 2000
            and payload.get('z_policy') == identity['z_policy'], 'NATIVE_CHECKPOINT_IDENTITY')
    require(digest(payload.get('seen_case_ids')) == original['ordered_case_ids_sha256'], 'NATIVE_STREAM_ORDER')
    native = payload.get('config') or {}
    require(native.get('model_dtype') == 'bfloat16' and native.get('bs') == 100 and native.get('num_edits') == 2000
            and native.get('algs', {}).get('add_old_keys') is True
            and native.get('diagnostic_identity_sha256') == identity['native_identity_sha256']
            and NATIVE_Z.get(native.get('diagnostic_z_reference', 'batch_entry')) == identity['z_policy']
            and {k: native.get('llms', {}).get(k) for k in ('rewrite_module_tmp', 'layers')}
            == {k: original['hparams'][k] for k in ('rewrite_module_tmp', 'layers')}, 'NATIVE_CONFIG')
    restored = bind_weights(model, payload['weights'], original, torch.bfloat16)
    return receipt(original, restored, torch.bfloat16)


def restore(model, original):
    schema = original.get('checkpoint_schema')
    if schema == OFFICIAL:
        return restore_official(model, original)
    if schema == NATIVE:
        return restore_native(model, original)
    raise ValueError('CHECKPOINT_SCHEMA_NOT_ALLOWLISTED')


def tracking_values(config, source):
    value = _tracking(config, source)
    value.update(server=config['server'], task_id=TASK)
    return validate_tracking(value)


@contextmanager
def host_bindings():
    # Process-local dependency injection only; the shared evaluator source and
    # other jobs are never modified. Reset even on exception.
    previous = shared.tracking_values, shared.restore
    shared.tracking_values, shared.restore = tracking_values, restore
    try:
        yield
    finally:
        shared.tracking_values, shared.restore = previous


def run(config, lock):
    c = read(config)
    require(c.get('registration_authority') == AUTHORITY and c.get('asset_schema') == 'history'
            and c.get('server') in NODES, 'FE_CAKE_FLUCON_SCOPE')
    node = os.environ.get('SLURMD_NODENAME')
    require(node is None or node == NODES[c['server']], 'FE_CAKE_FLUCON_NODE')
    with host_bindings():
        shared.run(config, lock)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['run', 'collect'])
    p.add_argument('--config'); p.add_argument('--lock', required=True)
    p.add_argument('--preparation'); p.add_argument('--output'); a = p.parse_args()
    if a.mode == 'run':
        run(a.config, a.lock)
    else:
        shared.collect(a.preparation, a.output, a.lock)
