"""Create-once artifact I/O and explicit immutable input bindings."""
import csv
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = Path('/mnt/raid5/janghj/ODE-edit')
TASK = ROOT / 'local/checkpoint-mechanism-audit/20260920-v1'
ATTEMPT = TASK / 'attempt-v1'
CONTRACT_PATH = REPO / 'plans/global/2026-09-20-server2-checkpoint-mechanism-audit-contract-v1.json'
CONTRACT = json.loads(CONTRACT_PATH.read_text())
S4ROOT = '/data/janghj/ODE-edit'
S4RUN = S4ROOT + '/local/blue-lifelong-b100x100/attempt-checkpoint-r2'
S4CELL = S4RUN + '/output/main-cell-3'
S4OLD = S4ROOT + '/local/blue-alphaedit-sequential-comparison/attempt-v1'
SAMPLE = S4ROOT + '/local/blue-lifelong-b100x100/attempt-v1/sample.lock.json'
CPROOT = Path(CONTRACT['paths']['checkpoint_root'])
INITIAL = ROOT/'local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1'
WEIGHT = CONTRACT['identity']['weight_name']

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')

def write_csv(path, rows, fields=None):
    rows = list(rows); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def stat_identity(path):
    p=Path(path); s=p.stat()
    return dict(path=str(p), bytes=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino, device=s.st_dev, realpath=str(p.resolve()))

def tensor_sha(t):
    t=t.detach().cpu().contiguous()
    # Pinned original helper hashes dtype/shape followed by contiguous bytes.
    h=hashlib.sha256(str((str(t.dtype),list(t.shape))).encode())
    h.update(memoryview(t.numpy()).cast('B'))
    return h.hexdigest()

def source_map(path=None):
    return read(path or ATTEMPT/'inputs/source-map.json')

def mapped(source, mapping=None):
    return Path((mapping or source_map())[source])
