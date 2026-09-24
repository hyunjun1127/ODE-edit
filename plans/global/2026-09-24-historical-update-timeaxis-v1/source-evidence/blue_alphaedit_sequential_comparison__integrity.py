"""Create-once artifacts and byte-level state identities (no editing equations)."""
import hashlib
import json
import os
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(value, f, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    return dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha(path))


def tensor_sha(t):
    import torch
    x = t.detach().contiguous().cpu()
    h = hashlib.sha256(str((str(x.dtype), list(x.shape))).encode())
    a = x.view(torch.uint8).numpy().reshape(-1)
    for i in range(0, a.size, 8 << 20):
        h.update(memoryview(a[i:i + (8 << 20)]))
    return h.hexdigest()


def signature(weights, cache, versions=True):
    r = {k: dict(sha256=tensor_sha(v), shape=list(v.shape), dtype=str(v.dtype),
                 pointer=v.data_ptr(), object_id=id(v)) for k, v in weights.items()}
    if versions:
        for k, v in weights.items():
            r[k]['version'] = v._version
    return dict(weights=r, cache_sha256=tensor_sha(cache), cache_pointer=cache.data_ptr())


def content(sig):
    return dict(weights={k: v['sha256'] for k, v in sig['weights'].items()}, cache=sig['cache_sha256'])


def restore(weights, snapshot, cache, cache_snapshot):
    import torch
    with torch.no_grad():
        for k, v in weights.items():
            v.copy_(snapshot[k].to(v.device))
        cache.copy_(cache_snapshot)
    if any(tensor_sha(weights[k]) != tensor_sha(v) for k, v in snapshot.items()) or tensor_sha(cache) != tensor_sha(cache_snapshot):
        raise RuntimeError('EXACT_ROLLBACK_BYTES')


def tensor_artifact(path, obj):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    return dict(path=str(path), sha256=file_sha(path), bytes=path.stat().st_size)


def checkpoint(path, weights, cache, metadata):
    import torch
    obj = dict(weights={k: v.detach().cpu().clone() for k, v in weights.items()},
               cache_c=cache.clone(), metadata=metadata)
    receipt = tensor_artifact(path, obj)
    loaded = torch.load(path, map_location='cpu', weights_only=True)
    assert {k: tensor_sha(v) for k, v in loaded['weights'].items()} == {k: tensor_sha(v) for k, v in weights.items()}
    assert tensor_sha(loaded['cache_c']) == tensor_sha(cache)
    return dict(**receipt,
                reload_weights_cache_exact=True, full_model_saved=False)
