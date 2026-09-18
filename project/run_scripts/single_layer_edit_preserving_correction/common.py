"""Create-once task evidence; no model or scheduler imports."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path('/data/janghj/ODE-edit/local/single-layer-edit-preserving-correction/20260918-v1')
INSTRUCTION = 'ODEEDIT-S06-SINGLE-LAYER-EDIT-PRESERVING-CORRECTION-M-SH4-V1'
STEM = '2026-09-18-single-layer-edit-preserving-correction'
ARMS = ('N4','SCALE','CA','KL-P','EN-S','EN-F','EN-COV','EN-F4')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def member(path):
    p=Path(path)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))

def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return member(p)

def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def save_tensor(path,obj):
    import torch
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:torch.save(obj,f)
    return member(p)

class Timer:
    def __init__(self,counts,key):self.counts,self.key=counts,key
    def __enter__(self):self.start=time.monotonic();return self
    def __exit__(self,*a):self.counts[self.key]=self.counts.get(self.key,0)+time.monotonic()-self.start
