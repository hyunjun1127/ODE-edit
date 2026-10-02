"""V5 scalar evidence, immutable receipts, and RAM-only tensor identities."""
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOCAL = Path('/data/janghj/ODE-edit/local/jlz-writer-coupled-v5/20261002-v1')
INSTRUCTION = 'ODEEDIT-USER-GH-SH4-JLZ-V5-CANCEL-V4-IMPLEMENT-2K-20261002-R1'
TASK = 'jlz-writer-coupled-v5-bs100x20-20261002-v1'

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def member(path):
    p = Path(path); s = p.stat()
    return dict(path=str(p.resolve()), bytes=s.st_size, sha256=sha(p),
                inode=s.st_ino, mtime_ns=s.st_mtime_ns)

def serial(value):
    if hasattr(value, 'detach'):
        require(value.numel() <= 10000, 'NO_TENSOR_PAYLOAD_IN_EVIDENCE')
        return serial(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'item'):
        return serial(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value

def write(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(serial(value), ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n').encode()
    if p.exists():
        require(p.read_bytes() == data, 'IMMUTABLE_RECEIPT_CONFLICT:'+str(p)); return
    tmp = p.with_name(p.name+'.tmp')
    with tmp.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.link(tmp, p); tmp.unlink()

def tensor_sha(value):
    x = value.detach().cpu().contiguous()
    h = hashlib.sha256(str((tuple(x.shape), str(x.dtype))).encode())
    h.update(memoryview(x.numpy()).cast('B'))
    return h.hexdigest()

def state(adapter, history):
    return dict(W={str(l):tensor_sha(w) for l,w in adapter.weights.items()},
                H={str(l):tensor_sha(h) for l,h in history.items()})
