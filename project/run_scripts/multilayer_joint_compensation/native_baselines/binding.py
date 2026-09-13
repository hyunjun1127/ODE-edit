"""Direct pinned BLUE native entrypoints, no B controller inside baselines."""
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types
import torch

SOURCE=Path('/mnt/raid5/janghj/ODE-edit/local/multilayer-joint-compensation/20260911-v1/imports/native-source-v1')
REV='311b076a92e4ed0f14f5c8b4909732da781bc5f7'
LAYERS=(4,5,6,7,8)


def verify_source(root=SOURCE):
    manifest=json.loads((root/'source-manifest.json').read_text())
    if manifest['revision']!=REV:raise ValueError('NATIVE_SOURCE_REVISION')
    for row in manifest['members']:
        path=root/row['path']
        if path.is_symlink() or not path.is_file():raise ValueError('NATIVE_NONREGULAR_SOURCE')
        data=path.read_bytes()
        if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise ValueError('NATIVE_SOURCE_CHANGED')
    return manifest


def load_native(method,root=SOURCE):
    """One fresh baseline process; existing foreign imports fail closed."""
    if method not in ('ALPHAEDIT_NATIVE','MEMIT_NATIVE'):raise ValueError('NATIVE_METHOD')
    manifest=verify_source(root)
    for name in ('AlphaEdit','memit','rome','util'):
        existing=sys.modules.get(name)
        if existing is not None and list(getattr(existing,'__path__',[]))!=[str(root/name)]:
            raise RuntimeError('FOREIGN_NATIVE_MODULE_NAMESPACE:'+name)
        if existing is None:
            module=types.ModuleType(name);module.__path__=[str(root/name)];module.__package__=name;sys.modules[name]=module
    previous=os.getcwd()
    try:
        os.chdir(root)
        module=importlib.import_module('AlphaEdit.AlphaEdit_main' if method=='ALPHAEDIT_NATIVE' else 'memit.memit_main')
    finally:os.chdir(previous)
    hpclass=module.AlphaEditHyperParams if method=='ALPHAEDIT_NATIVE' else module.MEMITHyperParams
    hp=hpclass.from_json(root/('hparams/AlphaEdit/Llama3-8B.json' if method=='ALPHAEDIT_NATIVE' else 'hparams/MEMIT/Llama3-8B.json'))
    validate_hparams(method,hp)
    return module,hp,dict(revision=REV,tree=manifest['tree'],source_manifest_sha256=hashlib.sha256((root/'source-manifest.json').read_bytes()).hexdigest(),
        function='apply_AlphaEdit_to_model' if method=='ALPHAEDIT_NATIVE' else 'apply_memit_to_model',
        initializer_bypass=['AlphaEdit.__init__','memit.__init__','rome.__init__','util.__init__'],
        source_bytes_modified=0,new_controller_influence=0)


def validate_hparams(method,hp):
    if tuple(hp.layers)!=LAYERS or hp.blue or hp.v_num_grad_steps!=25 or hp.v_lr!=.1 or hp.kl_factor!=.0625:
        raise ValueError('NATIVE_HPARAM_CONTRACT')
    if method=='ALPHAEDIT_NATIVE' and hp.L2!=10:raise ValueError('NATIVE_ALPHA_L2')
    if method=='MEMIT_NATIVE' and (hp.edit_layer!=-1 or hp.mom2_update_weight!=15000):raise ValueError('NATIVE_MEMIT_CONTRACT')


def bind_covariance_cache(module,model,covariances):
    """Bind exact existing static moments, never collect Wikipedia statistics."""
    if set(covariances)!=set(LAYERS):raise ValueError('MEMIT_FIVE_COVARIANCES_REQUIRED')
    name=model.config._name_or_path.replace('/','_')
    module.COV_CACHE={}
    for layer,tensor in covariances.items():
        if tensor.shape!=(14336,14336) or tensor.dtype!=torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError('MEMIT_COVARIANCE_SHAPE_DTYPE_FINITE')
        module.COV_CACHE[(name,f'model.layers.{layer}.mlp.down_proj')]=tensor.detach().cpu()


@contextmanager
def native_telemetry(module,model,ledger):
    """Observe native calls; always restore functions, no target/writer formula copy."""
    originals={name:getattr(module,name) for name in ('compute_z','compute_ks')}
    target_rows=[];key_rows=[];target_tensors=[];solve_rows=[]
    for name in originals:
        def wrapped(*args,_name=name,**kwargs):
            with ledger.time('native_'+_name):value=originals[_name](*args,**kwargs)
            ledger.add('native_'+_name)
            if _name=='compute_z':
                target_rows.append(dict(case_id=args[2].get('case_id'),layer=args[4],shape=list(value.shape),dtype=str(value.dtype)))
                target_tensors.append(value.detach().cpu().clone())
            else:key_rows.append(dict(layer=args[4],shape=list(value.shape),dtype=str(value.dtype)))
            return value
        setattr(module,name,wrapped)
    previous_stats=getattr(module,'layer_stats',None)
    def forbidden_stats(*args,**kwargs):raise RuntimeError('UNSEALED_COVARIANCE_REGENERATION_FORBIDDEN')
    if previous_stats is not None:module.layer_stats=forbidden_stats
    original_solve=torch.linalg.solve;original_backward=torch.autograd.backward
    def solve(*args,**kwargs):
        with ledger.time('native_linear_solve'):value=original_solve(*args,**kwargs)
        ledger.add('native_linear_solve')
        solve_rows.append(dict(matrix_shape=list(args[0].shape),rhs_shape=list(args[1].shape),matrix_dtype=str(args[0].dtype),rhs_dtype=str(args[1].dtype)))
        return value
    def backward(*args,**kwargs):
        with ledger.time('native_backward'):value=original_backward(*args,**kwargs)
        ledger.add('native_backward');return value
    def forward_count(_model,args,kwargs):
        ids=kwargs.get('input_ids',args[0] if args else None)
        ledger.add('native_model_forward')
        if ids is not None:ledger.add('native_forward_padded_tokens',ids.numel())
        mask=kwargs.get('attention_mask')
        if mask is not None:ledger.add('native_forward_nonpadding_tokens',int(mask.sum()))
    hook=model.register_forward_pre_hook(forward_count,with_kwargs=True)
    torch.linalg.solve=solve;torch.autograd.backward=backward
    try:yield dict(targets=target_rows,target_tensors_local_only=target_tensors,key_captures=key_rows,solves=solve_rows)
    finally:
        hook.remove();torch.linalg.solve=original_solve;torch.autograd.backward=original_backward
        for name,value in originals.items():setattr(module,name,value)
        if previous_stats is not None:module.layer_stats=previous_stats


def execute(module,hp,method,model,tok,requests,contexts,ledger,*,history=None,projectors=None):
    validate_hparams(method,hp)
    if not requests:return dict(method=method,status='B0_NOOP',native_entrypoint_calls=0,history_append=0,targets=[])
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise TypeError('NATIVE_MODEL_NOT_FP32')
    module.CONTEXT_TEMPLATES_CACHE=contexts
    if method=='ALPHAEDIT_NATIVE':
        if history is None or history.shape!=(5,14336,14336) or history.dtype!=torch.float32:
            raise ValueError('ALPHA_RECONSTRUCTED_ENTRY_HISTORY_REQUIRED')
        if projectors is None or projectors.shape!=(5,14336,14336) or projectors.dtype!=torch.float32:raise ValueError('ALPHA_PROJECTOR_INVENTORY')
    elif history is not None or projectors is not None:raise ValueError('MEMIT_ALPHA_HISTORY_INFLUENCE')
    with native_telemetry(module,model,ledger) as receipt:
        with ledger.time('native_full_entrypoint'):
            if method=='ALPHAEDIT_NATIVE':
                module.apply_AlphaEdit_to_model(model,tok,requests,hp,cache_c=history,P=projectors,cache_template=None)
            else:module.apply_memit_to_model(model,tok,requests,hp,cache_template=None)
    if len(receipt['targets'])!=len(requests):raise RuntimeError('NATIVE_Z_CALL_COUNT')
    if any(p.dtype!=torch.float32 or not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('NATIVE_ENDPOINT_DTYPE_NONFINITE')
    receipt.update(method=method,entrypoint_direct=True,cache_template=None,new_objective_influence=0,
        target_reuse_from_B=0,history_policy='NATIVE_ALPHA_ONCE_FULL_INVENTORY' if method=='ALPHAEDIT_NATIVE' else 'STATIC_COV_NO_ALPHA_HISTORY',
        solve_dtype='float32' if method=='ALPHAEDIT_NATIVE' else 'float64_NATIVE_EXCEPTION',model_write_dtype='float32')
    return receipt
