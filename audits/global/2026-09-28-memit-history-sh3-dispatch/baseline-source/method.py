"""Method-specific state and path portability, never writer equations."""
import importlib
from pathlib import Path
from project.run_scripts.blue_alphaedit_sequential_comparison.integrity import tensor_sha

def bind(name, config, lock, model, weights):
    import torch
    if name=='AlphaEdit':
        module=importlib.import_module('AlphaEdit.AlphaEdit_main')
        hp=importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams.from_json(config)
        allp=torch.load(lock['projector'],map_location='cpu',weights_only=True)
        indices=[lock['projector_source_layers'].index(l) for l in hp.layers]
        projector=allp[indices].clone(); del allp
        state=torch.zeros_like(projector)
        extra=dict(projector_physical_layers=hp.layers,projector_indices=indices,projector_sha256=tensor_sha(projector),history_policy='native final selected-layer key append once/batch')
        def apply(tok,requests):
            returned,cache=module.apply_AlphaEdit_to_model(model,tok,requests,hp,cache_template=None,cache_c=state,P=projector)
            assert returned is model and cache is state
    else:
        module=importlib.import_module('memit.memit_main')
        hp=importlib.import_module('memit.memit_hparams').MEMITHyperParams.from_json(config)
        assert not module.COV_CACHE
        # Native layer_stats supports model_name for an asset directory binding.
        # Preserve native loader, float conversion and get_cov solve semantics.
        original_stats=module.layer_stats
        module.STATS_DIR=Path(lock['stats_root'])
        def stats(*args,**kw):
            assert not kw.get('force_recompute',False)
            kw['model_name']='Meta-Llama-3-8B-Instruct'
            expected=Path(args[3])/kw['model_name']/f"{args[4]}_stats"/f"{args[2]}_{kw['precision']}_mom2_{kw['sample_size']}.npz"
            assert expected.is_file(),f'IMMUTABLE_STATS_UNAVAILABLE:{expected}'
            return original_stats(*args,**kw)
        module.layer_stats=stats
        state=torch.empty(0,dtype=torch.float32) # Signature sentinel, NOT history.
        extra=dict(history_policy='NONE; native static covariance computation cache',stats_root=lock['stats_root'],stats_model_name='Meta-Llama-3-8B-Instruct',covariance_snapshots='immutable NPZ + native float32 load; cached tensor identity recorded')
        def apply(tok,requests):
            returned,originals=module.apply_memit_to_model(model,tok,requests,hp,copy=False,return_orig_weights=False,cache_template=None)
            assert returned is model and originals=={}
    assert not hp.blue, 'NON_BLUE_NATIVE_CONFIG_REQUIRED'
    return module,hp,state,extra,apply

def covariance_identity(module):
    return {repr(k):dict(sha256=tensor_sha(v),pointer=v.data_ptr(),version=v._version,dtype=str(v.dtype),shape=list(v.shape)) for k,v in getattr(module,'COV_CACHE',{}).items()}

