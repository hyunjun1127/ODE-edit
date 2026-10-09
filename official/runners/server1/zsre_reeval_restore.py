"""Read-only checkpoint restoration for inference; no native apply/cache binding."""
import torch
from official.experiments.prepare import digest
from .common import verify,read

def restore_payload(model,payload,original):
    if payload.get('schema')!='official-baseline-checkpoint-v1' or payload.get('batch')!=20:
        raise ValueError('FINAL_W20_REQUIRED')
    if payload.get('identity')!=original['identity'] or payload.get('method')!=original['method']:
        raise ValueError('ORIGINAL_CHECKPOINT_IDENTITY')
    if payload.get('evaluation_cursor',{}).get('completed_batch')!=20:
        raise ValueError('FINAL_CURSOR_REQUIRED')
    if payload.get('contexts',{}).get('successful_calls')!=20:
        raise ValueError('NATIVE_CONTEXT_CURSOR_REQUIRED')
    hp=original['hparams'];parameters=dict(model.named_parameters())
    if original['method']=='FT':
        selected={k for k in parameters if any(hp['rewrite_module_tmp'].format(l) in k for l in hp['layers'])}
    else:selected={hp['rewrite_module_tmp'].format(l)+'.weight' for l in hp['layers']}
    if not selected or set(payload.get('weights',{}))!=selected or not selected<=parameters.keys():
        raise ValueError('EXACT_SELECTED_WEIGHTS_REQUIRED')
    history=payload.get('cache_c')
    if type(history) is not dict or bool(history)!=original['expected_history']:
        raise ValueError('NATIVE_HISTORY_SCHEMA')
    if history and set(history)!={str(l) for l in hp['layers']}:
        raise ValueError('NATIVE_HISTORY_LAYER_IDENTITY')
    # Validate all members before changing any selected parameter.
    for key in selected:
        p,w=parameters[key],payload['weights'][key]
        if not torch.is_tensor(w) or w.dtype!=torch.float32 or w.shape!=p.shape or not torch.isfinite(w).all():
            raise ValueError('SELECTED_SHAPE_DTYPE_FINITE')
    for value in history.values():
        if not torch.is_tensor(value) or value.dtype!=torch.float32 or value.ndim!=2 or value.shape[0]!=value.shape[1] or not torch.isfinite(value).all():
            raise ValueError('HISTORY_SHAPE_DTYPE_FINITE')
    untouched={k:(id(v),v.data_ptr(),v._version) for k,v in parameters.items() if k not in selected}
    with torch.no_grad():
        for key in selected:parameters[key].copy_(payload['weights'][key])
    for key,expected in untouched.items():
        p=parameters[key]
        if (id(p),p.data_ptr(),p._version)!=expected:raise ValueError('RESTORE_NONSELECTED_MUTATION')
    # Native cache/context/RNG are provenance only: eval uses no editor and no target fit.
    # Evaluation observer must preserve the actual evaluation-entry RNG, not resample it.
    return dict(restored_selected_weights=len(selected),original_identity_sha256=digest(original['identity']),
                restored_batch=20,history_preserved_in_original_CP=True,no_native_apply=True)

def restore(model,original):
    path=verify(original['checkpoint']);pointer=read(verify(original['pointer']))
    if pointer['sha256']!=original['checkpoint']['sha256'] or pointer['batch']!=20 or not pointer['final_W20']:
        raise ValueError('POINTER_CHANGED')
    payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    return restore_payload(model,payload,original)
