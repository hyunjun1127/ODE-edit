"""File-backed import provenance and scalar RAM-state identities only."""
from pathlib import Path
from inspect import getattr_static
from .io import file_sha,digest,tensor_sha

def import_closure(modules,roots):
    roots=[Path(p).resolve() for p in roots]
    files={};virtual=[]
    for name,module in list(modules.items()):
        if module is None:continue
        attrs=vars(module);raw=getattr_static(module,'__file__',None)
        if not isinstance(raw,str) or not raw.endswith('.py'):continue
        spec=attrs.get('__spec__');origin=getattr(spec,'origin',None)
        path=Path(raw)
        if not path.is_absolute():
            if isinstance(origin,str) and Path(origin).is_absolute():path=Path(origin)
            else:
                virtual.append(dict(module=name,file=raw,reason='relative_virtual_file_without_absolute_spec_origin'))
                continue
        path=path.resolve()
        if not any(path.is_relative_to(root) for root in roots):continue
        if not path.is_file():raise FileNotFoundError('REAL_SCOPED_IMPORT_MISSING:'+str(path))
        files[str(path)]=dict(path=str(path),sha256=file_sha(path))
    return dict(files=[files[k] for k in sorted(files)],virtual_modules=virtual)

def rng_fingerprint():
    import random,numpy as np,torch
    n=np.random.get_state()
    return dict(python=digest(random.getstate()),numpy=digest([n[0],n[1].tolist(),n[2],n[3],n[4]]),
        torch_cpu=tensor_sha(torch.get_rng_state()),torch_cuda=[tensor_sha(x) for x in torch.cuda.get_rng_state_all()])

def auxiliary_state(contexts,records):
    return dict(context_hash=digest(contexts),rng=rng_fingerprint(),ledger=dict(requests=len(records),case_order_hash=digest([int(r['case_id']) for r in records])))
