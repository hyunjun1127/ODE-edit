"""Read-only original BLUE binding; arbitrary effective B, no copied solver."""
import contextlib
import importlib
import copy
import torch
from .identity import ABC,WEIGHT,Ledger
from project.run_scripts.single_layer_cumulative_risk import binding as legacy
from project.run_scripts.single_layer_cumulative_risk.microbatch import bounded_reader

@contextlib.contextmanager
def reader(ledger):
    legacy.kernel()
    rt=importlib.import_module('rome.repr_tools');old=rt.get_reprs_at_idxs
    rt.get_reprs_at_idxs=bounded_reader(old,ledger,2)
    try:yield
    finally:rt.get_reprs_at_idxs=old

def hp():
    legacy.kernel()
    return importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams.from_json(ABC/'imports/config.json')

def representations(model,tok,rows,contexts,ledger,*,all_contexts):
    native=legacy.kernel();h=hp()
    patterns=[];words=[]
    flat=[c for kind in contexts for c in kind]
    weights=[1/(len(contexts)*len(kind)) for kind in contexts for c in kind]
    for r in rows:
        rw=r['requested_rewrite']
        for c in flat if all_contexts else ['{}']:
            patterns.append(c.format(rw['prompt']));words.append(rw['subject'])
    with reader(ledger),ledger.time('native_representation_capture'):
        kin,_=native.get_module_input_output_at_words(model,tok,4,patterns,words,h.rewrite_module_tmp,h.fact_token)
        if all_contexts:
            _,outputs=native.get_module_input_output_at_words(model,tok,4,
                [r['requested_rewrite']['prompt'] for r in rows],
                [r['requested_rewrite']['subject'] for r in rows],h.layer_module_tmp,h.fact_token)
        else:outputs=None
    return kin,outputs,weights

def write(model,tok,cp,targets,rows,praw,ledger):
    """Caller passes request-labeled fixed targets; original BLUE solve untouched."""
    if tok.padding_side!='right':raise ValueError('NATIVE_RIGHT_PADDING')
    native=legacy.kernel();h=hp();w=dict(model.named_parameters())[WEIGHT]
    old_weight=w.detach().clone();oldcache=native.CONTEXT_TEMPLATES_CACHE
    native.CONTEXT_TEMPLATES_CACHE=cp['metadata']['contexts']
    state=cp['cache_c'].clone();oldz=native.compute_z
    def cached(model_,tok_,request,hp_,layer,contexts):
        if layer!=4 or contexts!=cp['metadata']['contexts']:raise ValueError('FIXED_Z_CONTEXT')
        ledger.add('native_z_cache_hits')
        return targets[request['case_id']].to(w.device)
    native.compute_z=cached
    try:
        with reader(ledger),ledger.time('native_joint_write'):
            native.apply_AlphaEdit_to_model(model,tok,[dict(r['requested_rewrite'],case_id=r['case_id']) for r in rows],
                h,cache_c=state,P=praw[None])
        result=w.detach().cpu().clone();committed=state.cpu().clone()
        if not torch.isfinite(result).all():raise FloatingPointError('NATIVE_NONFINITE')
    finally:
        native.compute_z=oldz;native.CONTEXT_TEMPLATES_CACHE=oldcache
        with torch.no_grad():w.copy_(old_weight)
        if not torch.equal(w,old_weight):raise RuntimeError('NATIVE_RESTORE')
    ledger.add('native_proposal_build');ledger.add('native_temporary_restore')
    return result,committed
