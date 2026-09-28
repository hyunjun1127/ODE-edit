"""Pinned BLUE MEMIT_seq binding and observations, no writer/hparam changes."""
import importlib,time
from contextlib import contextmanager
from pathlib import Path
from .io import tensor_sha,digest

class Proxy:
    def __init__(self,base,**changes):self.base,self.changes=base,changes
    def __getattr__(self,key):return self.changes.get(key,getattr(self.base,key))

def bind(config,lock,model):
    import torch
    module=importlib.import_module('memit.memit_seq_main')
    hp=importlib.import_module('memit.memit_hparams').MEMITHyperParams.from_json(config)
    assert not hp.blue and hp.layers==[4,5,6,7,8]
    assert hp.mom2_update_weight==15000
    weights={hp.rewrite_module_tmp.format(l)+'.weight':module.nethook.get_parameter(model,hp.rewrite_module_tmp.format(l)+'.weight') for l in hp.layers}
    assert all(tuple(w.shape)==(4096,14336) and w.dtype==torch.float32 for w in weights.values())
    state=torch.zeros((5,14336,14336),dtype=torch.float32,device='cpu')
    assert not module.COV_CACHE
    module.CONTEXT_TEMPLATES_CACHE=__import__('json').loads(Path(lock['context']).read_text())
    original_stats=module.layer_stats;module.STATS_DIR=Path(lock['stats_root'])
    def stats(*a,**kw):
        assert not kw.get('force_recompute',False)
        kw['model_name']='Meta-Llama-3-8B-Instruct'
        expected=Path(a[3])/kw['model_name']/f'{a[4]}_stats'/f"{a[2]}_{kw['precision']}_mom2_{kw['sample_size']}.npz"
        assert str(expected) in lock['stats_paths'] and expected.is_file()
        return original_stats(*a,**kw)
    module.layer_stats=stats
    return module,hp,weights,state

def nonselected(model,weights):
    return {k:(v.data_ptr(),v._version) for k,v in model.named_parameters() if k not in weights}

def cov_guard(module):
    return {str(k):(v.data_ptr(),v._version) for k,v in module.COV_CACHE.items()}

@contextmanager
def observe(module,hp,weights,state,requests,model):
    import torch
    originals={k:getattr(module,k) for k in ('torch','compute_z','compute_ks','execute_memit')}
    prior={k:tensor_sha(v) for k,v in weights.items()};ns=nonselected(model,weights)
    receipt=dict(z=[],keys=[],solves=[],target_seconds=0.,key_seconds=0.,solve_seconds=0.,history_append_layers=0)
    phase='z'
    def z(*a,**kw):
        assert len(receipt['keys'])==0
        start=time.monotonic();v=originals['compute_z'](*a,**kw);torch.cuda.synchronize()
        assert torch.isfinite(v).all()
        receipt['target_seconds']+=time.monotonic()-start
        receipt['z'].append(dict(case_id=a[2]['case_id'],layer=a[4],sha256=tensor_sha(v),norm=float(v.detach().norm())))
        return v
    def keys(*a,**kw):
        index=len(receipt['keys']);assert len(receipt['z'])==len(requests)
        assert a[4]==hp.layers[index%5] and index<10
        assert len(receipt['solves'])==min(index,5),'POST_HISTORY_BEFORE_ALL_WRITES'
        start=time.monotonic();v=originals['compute_ks'](*a,**kw);torch.cuda.synchronize()
        assert torch.isfinite(v).all()
        receipt['key_seconds']+=time.monotonic()-start
        receipt['keys'].append(dict(layer=a[4],phase='pre_layer' if index<5 else 'post_all_layers',shape=list(v.shape),sha256=tensor_sha(v)))
        return v
    def solve(a,b,*args,**kw):
        index=len(receipt['solves']);assert len(receipt['keys'])==index+1 and index<5
        assert a.dtype==b.dtype==torch.float64
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        start=time.monotonic();v=originals['torch'].linalg.solve(a,b,*args,**kw);torch.cuda.synchronize()
        assert torch.isfinite(v).all()
        receipt['solve_seconds']+=time.monotonic()-start
        receipt['solves'].append(dict(layer=hp.layers[index],dtype=str(a.dtype),shape=list(a.shape),rhs_shape=list(b.shape),prior_history_sha256=tensor_sha(state[index])))
        return v
    def execute(*a,**kw):
        assert kw['cache_c'] is state and kw['cache_template'] is None
        result=originals['execute_memit'](*a,**kw)
        assert result[1] is state
        assert {k:tensor_sha(v) for k,v in weights.items()}==prior,'NATIVE_TEMPORARY_RESTORE'
        receipt['temporary_restore_exact']=True
        return result
    module.compute_z=z;module.compute_ks=keys;module.execute_memit=execute
    module.torch=Proxy(originals['torch'],linalg=Proxy(originals['torch'].linalg,solve=solve))
    try:
        yield receipt
        assert [(r['case_id'],r['layer']) for r in receipt['z']]==[(r['case_id'],8) for r in requests]
        assert [r['layer'] for r in receipt['keys']]==hp.layers*2
        assert len(receipt['solves'])==5 and receipt['temporary_restore_exact']
        assert nonselected(model,weights)==ns
        receipt.update(compute_z=len(requests),solve_calls=5,history_append_layers=5,nonselected_pointer_version_exact=True,history_key_phase='post_all_five_temporary_writes',cache_template=None)
    finally:
        for k,v in originals.items():setattr(module,k,v)
