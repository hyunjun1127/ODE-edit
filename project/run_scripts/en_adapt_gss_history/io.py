"""Lossless compact observation storage; strict scalar JSON, no tensor serializer."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from project.run_scripts.en_adaptive_nullspace.json_io import scalar, save


def save_gzip(path, value):
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                     allow_nan=False, default=scalar).encode('utf8')
    payload = gzip.compress(raw, compresslevel=1, mtime=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return dict(path=str(path), bytes=len(payload), decoded_bytes=len(raw),
                sha256=hashlib.sha256(payload).hexdigest(),
                decoded_sha256=hashlib.sha256(raw).hexdigest(), compression='gzip-lossless-level1')


def read(path):
    path = Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes())
