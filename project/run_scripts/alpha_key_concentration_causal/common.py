"""Task-owned strict create-once evidence, clocks and RAM-only transactions."""
import hashlib
import json
import os
from pathlib import Path
import random
import time
from contextlib import contextmanager

def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()

def clean(value):
    import numpy as np
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(x) for x in value]
    return value

def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(clean(value),ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    tmp=p.with_name(p.name+'.atomic-partial')
    with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.link(tmp,p);tmp.unlink()
    return dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())

def tensor_sha(t):
    import torch
    t=t.detach().contiguous().cpu()
    h=hashlib.sha256(str((str(t.dtype),list(t.shape))).encode())
    a=t.reshape(-1).view(torch.uint8).numpy()
    for i in range(0,a.size,8<<20):h.update(memoryview(a[i:i+(8<<20)]))
    return h.hexdigest()

def tensor_file(path,value):
    """Only explicitly authorized diagnostic tensors; not W/M/resume state."""
    import torch
    forbidden={'weights','cache_c','optimizer','rng','resume','model_state_dict'}
    if isinstance(value,dict):assert not forbidden.intersection(value),'NEW_RESUME_CHECKPOINT_FORBIDDEN'
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.atomic-partial')
    with tmp.open('xb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.link(tmp,p);tmp.unlink()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha(p))

def rng_get():
    import torch,numpy as np
    return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())

def rng_set(x):
    import torch,numpy as np
    random.setstate(x[0]);np.random.set_state(x[1]);torch.set_rng_state(x[2]);torch.cuda.set_rng_state_all(x[3])

@contextmanager
def rng_preserved():
    x=rng_get()
    try:yield
    finally:rng_set(x)

class Timer:
    def __init__(self):self.events=[]
    @contextmanager
    def measure(self,kind,**meta):
        import torch
        torch.cuda.synchronize();t=time.monotonic()
        try:yield
        finally:
            torch.cuda.synchronize();self.events.append(dict(kind=kind,seconds=time.monotonic()-t,**meta))
