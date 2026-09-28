"""Nonintervening method-scoped call counts; no global torch patch."""
import time
from contextlib import contextmanager
from project.run_scripts.blue_alphaedit_sequential_comparison.integrity import tensor_sha,digest

class Proxy:
    def __init__(self,base,**changes):self.base,self.changes=base,changes
    def __getattr__(self,key):return self.changes[key] if key in self.changes else getattr(self.base,key)

def nonselected(model,selected,full=False):
    return {n:dict(pointer=p.data_ptr(),version=p._version,shape=list(p.shape),dtype=str(p.dtype),**({'sha256':tensor_sha(p)} if full else {})) for n,p in model.named_parameters() if n not in selected}

@contextmanager
def observe(module,hp,weights,state,requests,model,method,smoke=False):
    original={k:getattr(module,k) for k in ('torch','compute_z','compute_ks')}
    before=nonselected(model,weights,False)
    receipt=dict(z=[],keys=[],solves=[],target_seconds=0.,key_seconds=0.,solve_seconds=0.,_target_tensors=[])
    def z(*a,**kw):
        start=time.monotonic();v=original['compute_z'](*a,**kw)
        receipt['target_seconds']+=time.monotonic()-start
        receipt['z'].append(dict(case_id=a[2]['case_id'],layer=a[4],sha256=tensor_sha(v),norm=float(v.detach().double().norm())))
        receipt['_target_tensors'].append(v.detach().cpu().clone());return v
    def keys(*a,**kw):
        start=time.monotonic();v=original['compute_ks'](*a,**kw)
        receipt['key_seconds']+=time.monotonic()-start
        receipt['keys'].append(dict(layer=a[4],sha256=tensor_sha(v),shape=list(v.shape)));return v
    def solve(a,b,*args,**kw):
        start=time.monotonic();v=original['torch'].linalg.solve(a,b,*args,**kw)
        receipt['solve_seconds']+=time.monotonic()-start
        receipt['solves'].append(dict(shape=list(a.shape),rhs_shape=list(b.shape),dtype=str(a.dtype)));return v
    module.compute_z=z;module.compute_ks=keys
    module.torch=Proxy(original['torch'],linalg=Proxy(original['torch'].linalg,solve=solve))
    try:
        yield receipt
        assert not hp.blue, 'NATIVE_ONLY'
        expected=[(hp.layers[-1],r['case_id']) for r in requests]
        assert [(x['layer'],x['case_id']) for x in receipt['z']]==expected,'Z_LAYER_REQUEST_ORDER'
        assert [x['layer'] for x in receipt['keys']]==hp.layers*(2 if method=='AlphaEdit' else 1),'KEY_HISTORY_POLICY'
        assert len(receipt['solves'])==len(hp.layers),'NATIVE_SOLVE_COUNT'
        assert before==nonselected(model,weights,False),'NONSELECTED_WEIGHT_MUTATION'
        receipt.update(compute_z=len(expected),solve_calls=len(hp.layers),history_append_passes=int(method=='AlphaEdit'),cold_reset_inside_batch=0,z_hash_order=digest(receipt['z']),nonselected_pointer_version_exact=True,nonselected_full_bytes_checked=False,projector_physical_layers=hp.layers if method=='AlphaEdit' else [],projector_asset_indices=[l-4 for l in hp.layers] if method=='AlphaEdit' else [])
    finally:
        for k,v in original.items():setattr(module,k,v)
        assert all(getattr(module,k) is v for k,v in original.items())

