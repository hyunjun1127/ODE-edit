"""Task-local immutable evidence; no shared environment or input mutations."""
import hashlib
import json
import os
import ctypes
from pathlib import Path

ROOT = Path('/data/janghj/ODE-edit/local/temporal-routing-diagnostic/20260929-v1')
NONCE = 'ODEEDIT-GH-SH4-TEMPORAL-ROUTING-DIAGNOSTIC-20260929-R1'
REVISION = '8afb486c1db24fe5011ec46dfbe5b5dccdb575c2'
DEPS = Path('/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/deps-transformers-4.44.2')
NATIVE = Path('/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/blue-source')
MODEL = Path('/data/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots') / REVISION
DATA = Path('/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1')
PYTHON = '/data/janghj/EasyEdit/.venv/bin/python'
LAYERS = tuple(range(4, 9))
MILESTONES = (0, 1, 10, 25, 50, 100)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for x in iter(lambda: f.read(8 << 20), b''): h.update(x)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def record(path):
    p = Path(path); s = p.stat()
    return dict(path=str(p), bytes=s.st_size, sha256=sha(p), inode=s.st_ino, mtime_ns=s.st_mtime_ns)


def save(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    b = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    tmp = p.with_name(p.name+'.partial')
    with tmp.open('xb') as f: f.write(b); f.flush(); os.fsync(f.fileno())
    publish(tmp,p)
    return dict(path=str(p), bytes=len(b), sha256=hashlib.sha256(b).hexdigest())


def tensor_sha(t):
    import torch
    a = t.detach().cpu().contiguous()
    h = hashlib.sha256(str((str(a.dtype), list(a.shape))).encode())
    v = a.reshape(-1).view(torch.uint8).numpy()
    for i in range(0, len(v), 8 << 20): h.update(memoryview(v[i:i+(8 << 20)]))
    return h.hexdigest()


def tensor_save(path, value):
    import torch
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.partial')
    with tmp.open('xb') as f: torch.save(value, f); f.flush(); os.fsync(f.fileno())
    publish(tmp,p)
    return record(p)


def publish(tmp,path):
    """Linux atomic rename with NOREPLACE: never overwrite existing evidence."""
    libc=ctypes.CDLL(None,use_errno=True)
    result=libc.renameat2(-100,os.fsencode(tmp),-100,os.fsencode(path),1)
    if result:
        e=ctypes.get_errno();raise OSError(e,os.strerror(e),str(path))
    fd=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def csv_rows(path):
    import csv
    with open(path, newline='') as f: return list(csv.DictReader(f))


def require(condition, label):
    if not condition: raise RuntimeError(label)
