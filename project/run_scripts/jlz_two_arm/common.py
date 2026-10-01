import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOCAL = Path('/data/janghj/ODE-edit/local/jlz-twoarm/20261002-bs100x20-v1')
INSTRUCTION = 'ODEEDIT-USER-GH-SH4-JLZ-TWOARM-BS100X20-20261002-R1'
CONTRACT = ROOT/'plans/global/2026-10-02-jlz-twoarm-bs100x20-execution-v1/contract.json'
PYTHON = '/data/janghj/EasyEdit/.venv/bin/python'
LAYERS = (4, 5, 6, 7, 8)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8*1024*1024), b''): h.update(data)
    return h.hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def member(path):
    p=Path(path); s=p.stat()
    return dict(path=str(p.resolve()), bytes=s.st_size, sha256=sha(p), inode=s.st_ino,mtime_ns=s.st_mtime_ns)

def jsonable(obj):
    if hasattr(obj,'detach'): return jsonable(obj.detach().cpu().tolist())
    if isinstance(obj,dict): return {str(k):jsonable(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [jsonable(v) for v in obj]
    if isinstance(obj,Path): return str(obj)
    if hasattr(obj,'item'): return jsonable(obj.item())
    if isinstance(obj,float) and not math.isfinite(obj): return str(obj)
    return obj

def write(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('x') as f:
        json.dump(jsonable(obj),f,ensure_ascii=False,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

def require(ok,message):
    if not ok: raise RuntimeError(message)

def numerical(values,thresholds):
    """Original numerical verdict preserved; never relabel finite exceedance PASS."""
    require(all(math.isfinite(float(v)) for v in values.values()),'NONFINITE_DIAGNOSTIC')
    passed=all(abs(float(v))<=thresholds[k] for k,v in values.items())
    return dict(values=values,thresholds=thresholds,original_verdict='PASS' if passed else 'FAIL',
                action='CONTINUE' if passed else 'CONTINUE_WITH_WARNING',certification='NOT_ESTABLISHED')
