"""Exact retained independent-cold native reuse, not process/checkpoint resume."""
import json
from pathlib import Path
import torch
from .common import member, tensor_sha, write


def endpoint_identity(identity):
    # TorchVersion is a str subclass but pickle records its unsupported class.
    # Normalize metadata only; the identity value and every tensor stay unchanged.
    return dict(identity, torch=str(identity['torch']), transformers=str(identity['transformers']))


def validate_native(result, binding, identity, ids):
    receipt=result['receipt'];weight=result['weight']
    if binding['case_ids']!=ids or [x['case_id'] for x in result['target_observations']]!=ids:
        raise ValueError('RETAINED_NATIVE_ORDER')
    if (binding['entry']['W']!=identity['W0'] or binding['entry']['M']!=identity['M0'] or
        binding['entry']['rng']!=identity['rng'] or not binding['entry']['independent_cold'] or
        receipt['entry_weight_sha256']!=identity['W0'] or receipt['history_sha256']!=identity['M0']):
        raise ValueError('RETAINED_NATIVE_NOT_COLD')
    if receipt!=binding['receipt'] or receipt['projector_sha256']!=identity['P4']['selected_tensor_sha256']:
        raise ValueError('RETAINED_NATIVE_RECEIPT_OR_P')
    if receipt['compute_z']!=len(ids) or receipt['history_append']!=0 or receipt['layer']!=4 or receipt['solve']!=1:
        raise ValueError('RETAINED_NATIVE_COUNTS')
    if weight.dtype!=torch.float32 or not torch.isfinite(weight).all():raise ValueError('RETAINED_NATIVE_WEIGHT')
    h=tensor_sha(weight)
    if h!=binding['endpoint'] or h!=receipt['endpoint_weight_sha256']:raise ValueError('RETAINED_NATIVE_WEIGHT_HASH')


def load_native(rt, records, directory, episode):
    spec=rt.lock.get('retained_cold_native',{}).get(str(episode))
    if spec is None:return rt.native(records,directory,reuse=episode==0)
    for field in ('native','binding','runtime'):
        if member(spec[field]['path'])!=spec[field]:raise ValueError('RETAINED_NATIVE_FILE:'+field)
    identity=json.loads(Path(spec['runtime']['path']).read_text())['identity']
    if rt.identity!=identity:raise ValueError('RETAINED_NATIVE_RUNTIME_IDENTITY')
    before=rt.reset()
    binding=json.loads(Path(spec['binding']['path']).read_text())
    result=torch.load(spec['native']['path'],weights_only=True,mmap=True,map_location='cpu')
    validate_native(result,binding,identity,[r['case_id'] for r in records])
    if tensor_sha(rt.P)!=result['receipt']['projector_sha256']:raise ValueError('RETAINED_NATIVE_ACTUAL_P')
    rt.copy_weight(result['weight'])
    write(Path(directory)/'native-binding.json',dict(source=spec['native'],mode='REUSE',entry=before,
        endpoint=binding['endpoint'],case_ids=binding['case_ids'],receipt=result['receipt'],
        native_fit_new_calls=0,native_target_new_calls=0,prior_attempt=spec['prior_lock'],
        prior_binding=spec['binding'],full_process_resume=False,prior_cancel_rollback='NOT_VERIFIED'))
    from project.run_scripts.baseline_mechanism_first.fixtures import restore_rng
    restore_rng(rt.rng)
    return result
