"""USER additive-history FE variant; stock FE fit/context/residual are reused.

This is not upstream MEMIT-FE. No extra fitting or B1 qualification is performed.
H stays CPU FP32 across batches, enters the native FP64 solve, and uses keys
recaptured on the complete final model. Updates are staged before H commit.
"""
from copy import deepcopy
import torch
from .easyedit.models.memit_FE import memit_FE_main as native
from .easyedit.util import nethook

MEMITFEHyperParams = native.MEMITFEHyperParams

def history_solve(cov, keys, coefficient, history):
    if history.shape != cov.shape or history.dtype != torch.float32 or not torch.isfinite(history).all():
        raise ValueError('FE_HISTORY_SHAPE_DTYPE_FINITE')
    base = coefficient * cov.double()
    entry = history.to(device=base.device, dtype=torch.float64)
    system = base + entry + keys @ keys.T
    if torch.count_nonzero(history) == 0:
        # Compare the actual B1 system, not an additional solve/fit/forward.
        if not torch.equal(system, base + keys @ keys.T):
            raise ValueError('FE_HISTORY_H0_NATIVE_SYSTEM_MISMATCH')
    return torch.linalg.solve(system, keys)

def apply_memit_fe_history_to_model(model, tok, requests, hparams, *, history,
                                    copy=False, return_orig_weights=False, cache_template=None):
    if copy or return_orig_weights or cache_template is not None:
        raise ValueError('FE_HISTORY_NATIVE_INPLACE_NO_ZCACHE_REQUIRED')
    layers = [str(layer) for layer in hparams.layers]
    if set(history) != set(layers):
        raise ValueError('FE_HISTORY_LAYER_MAPPING')
    weights = {f'{hparams.rewrite_module_tmp.format(layer)}.weight':
               nethook.get_parameter(model, f'{hparams.rewrite_module_tmp.format(layer)}.weight')
               for layer in hparams.layers}
    before = {name: value.detach().clone() for name,value in weights.items()}
    metadata = {name:(id(p),p.data_ptr(),p._version) for name,p in model.named_parameters()}
    try:
        result = native.apply_memit_FE_to_model(model, tok, requests, hparams,
                    copy=False, return_orig_weights=False, cache_template=None, history_entry=history)
        if result[0] is not model:
            raise ValueError('FE_HISTORY_CUMULATIVE_MODEL')
        normalized = deepcopy(requests)
        for request in normalized:
            if '{}' not in request['prompt']:
                if request['subject'] not in request['prompt']:
                    raise ValueError('FE_HISTORY_SUBJECT_MISSING')
                request['prompt'] = request['prompt'].replace(request['subject'], '{}')
        pending = {}
        with torch.no_grad():
            for layer in hparams.layers:
                key = native.compute_ks(model, tok, normalized, hparams, layer,
                                        native.get_context_templates(model, tok)).T
                old = history[str(layer)]
                if key.shape != (old.shape[0],len(requests)) or not torch.isfinite(key).all():
                    raise ValueError('FE_HISTORY_FINAL_NATIVE_MEAN_KEY')
                delta = (key @ key.T).to(device='cpu',dtype=torch.float32)
                pending[str(layer)] = old + delta
                if not torch.isfinite(pending[str(layer)]).all():
                    raise ValueError('FE_HISTORY_APPEND_NONFINITE')
        for name,p in model.named_parameters():
            prior = metadata[name]
            if id(p)!=prior[0] or p.data_ptr()!=prior[1] or (name not in weights and p._version!=prior[2]):
                raise ValueError('FE_HISTORY_NONSELECTED_OR_IDENTITY_MUTATION')
        if not all(torch.isfinite(p).all() for p in weights.values()):
            raise ValueError('FE_HISTORY_COMMIT_NONFINITE')
        # Atomic mapping replacement; failures before this point do not advance H.
        history.update(pending)
    except BaseException:
        with torch.no_grad():
            for name,p in weights.items(): p.copy_(before[name])
        raise
    return model, {}
