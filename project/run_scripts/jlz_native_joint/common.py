"""Small immutable evidence only. Never serialize learned tensors."""
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOCAL = Path('/data/janghj/ODE-edit/local/jlz-native-joint-v4/20261002-compute-r1')
INSTRUCTION = 'ODEEDIT-USER-GH-SH4-JLZ-V4-COMPUTE-R1-2K-20261002-R1'
DESIGN = ROOT / 'plans/global/2026-10-02-jlz-native-joint-v4'

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(data)
    return h.hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def member(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path.resolve()), bytes=stat.st_size, sha256=sha(path),
                inode=stat.st_ino, mtime_ns=stat.st_mtime_ns)

def serial(obj):
    if hasattr(obj, 'detach'):
        require(obj.numel() <= 10000, 'NO_TENSOR_PAYLOAD_IN_EVIDENCE')
        return serial(obj.detach().cpu().tolist())
    if isinstance(obj, dict):
        return {str(k): serial(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serial(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, 'item'):
        return serial(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    return obj

def write(path, obj):
    """Create-once final receipt, atomic rename; existing evidence immutable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(serial(obj), ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode()
    if path.exists():
        require(path.read_bytes() == data, 'IMMUTABLE_RECEIPT_CONFLICT:' + str(path))
        return
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(tmp, path)
    tmp.unlink()

def tensor_sha(tensor):
    x = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(str((tuple(x.shape), str(x.dtype))).encode())
    h.update(memoryview(x.numpy()).cast('B'))
    return h.hexdigest()
