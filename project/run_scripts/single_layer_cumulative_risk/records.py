"""Create-once receipts, immutable identities, and measured compute ledgers."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import time

def canonical(x):
    return json.dumps(x,sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False).encode()

def digest(x):return hashlib.sha256(canonical(x)).hexdigest()

def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(canonical(value)+b'\n');f.flush();os.fsync(f.fileno())

def tensor_save(path,value):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    with temporary.open('xb') as f:torch.save(value,f)
    os.link(temporary,path);temporary.unlink()

class Ledger:
    def __init__(self):self.counts={};self.seconds={}
    def add(self,key,n=1):self.counts[key]=self.counts.get(key,0)+n
    @contextlib.contextmanager
    def time(self,key):
        import torch
        if torch.cuda.is_initialized():torch.cuda.synchronize()
        start=time.monotonic()
        try:yield
        finally:
            if torch.cuda.is_initialized():torch.cuda.synchronize()
            self.seconds[key]=self.seconds.get(key,0)+time.monotonic()-start
    def receipt(self):return dict(counts=self.counts.copy(),seconds=self.seconds.copy(),timing='start sync excluded; end sync included')
