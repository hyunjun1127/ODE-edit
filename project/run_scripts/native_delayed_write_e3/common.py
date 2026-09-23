"""Task-local evidence utilities; never mutate an input or replace an artifact."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/data/janghj/ODE-edit/local/native-delayed-write-e3/20260924-v1')
INSTRUCTION = 'GH-SH4-NATIVE-DELAYED-WRITE-E3-20260924-V1'
BATCHES = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
BASE = Path('/data/janghj/ODE-edit/local/fixed10k-native-baselines/attempt-v1')
DEPS = Path('/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/deps-transformers-4.44.2')
MODEL = Path('/data/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2')
DATA = Path('/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json')
PANELS = ROOT / 'inputs/panels-v2'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    temp = path.with_name(path.name + '.partial')
    with temp.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.link(temp, path)  # atomic create-once, never replace
    temp.unlink()
    return dict(path=str(path), bytes=len(data), sha256=sha(path))


def file_record(path):
    path = Path(path)
    s = path.stat()
    return dict(path=str(path), bytes=s.st_size, sha256=sha(path), mtime_ns=s.st_mtime_ns)
