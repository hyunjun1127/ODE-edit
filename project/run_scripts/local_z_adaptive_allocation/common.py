"""Create-once evidence and explicit state identities (local raw only)."""
import hashlib
import json
import os
from pathlib import Path
import time

TASK = 'ODEEDIT-S06-LOCAL-Z-ADAPTIVE-ALLOCATION-SEQ1000-SH4-V1'
ROOT = Path('/data/janghj/ODE-edit/local/local-z-adaptive-allocation/20260916-v1')
ARMS = ('LD', 'N4', 'REFIT4', 'L75', 'T75', 'L4D', 'TD')
EXPECTED = {'N4': (100, 1, 1), 'REFIT4': (200, 2, 1), 'L75': (200, 2, 1),
            'T75': (100, 2, 1), 'L4D': (100, 1, 2), 'LD': (300, 3, 6), 'TD': (200, 4, 7)}

def encoded(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()

def digest(x):
    return hashlib.sha256(encoded(x)).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def identity(path):
    p = Path(path)
    return dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=sha(p))

def save(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        f.write(encoded(value)+b'\n'); f.flush(); os.fsync(f.fileno())
    return identity(p)

def tensor_save(path, value):
    import torch
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    return identity(p)

def layers(arm):
    if arm not in ARMS: raise ValueError('UNREGISTERED_ARM')
    return (4, 8) if arm in ('LD', 'TD', 'L75', 'T75') else (4,)

def verify(lock):
    assert lock['instruction_id'] == TASK and lock['arms'] == list(ARMS)
    assert lock['seed'] == 20260916 and lock['gpu_cap'] == 1
    assert lock['tf32_matmul'] is False and lock['tf32_cudnn'] is False
    assert lock['batches'] == 10 and lock['batch_size'] == 100
    for m in lock['members']:
        p = Path(m['path']); s = p.stat()
        assert s.st_size == m['bytes'], ('INPUT_SIZE', str(p))
        if m.get('verification') == 'PRIOR_FULL_SHA_STABLE_STAT':
            assert [s.st_dev, s.st_ino, s.st_mtime_ns] == m['stat'], ('IMMUTABLE_ASSET_DRIFT', str(p))
        else:
            assert sha(p) == m['sha256'], ('SOURCE_INPUT_SHA', str(p))

class Timer:
    def __init__(self, book, key): self.book, self.key = book, key
    def __enter__(self): self.begin = time.monotonic()
    def __exit__(self, *args): self.book[self.key] = self.book.get(self.key, 0.) + time.monotonic()-self.begin
