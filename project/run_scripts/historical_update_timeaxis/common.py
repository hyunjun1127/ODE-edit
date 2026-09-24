"""Task-local identity and atomic JSON utilities; no old runtime mutation."""
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT=Path('/data/janghj/ODE-edit/local/historical-update-timeaxis/20260924-v1')
DESIGN=ROOT/'inputs/design-package/members'
REPO=Path(__file__).resolve().parents[3]
INSTRUCTION='GH-SH4-HISTORICAL-UPDATE-TIMEAXIS-20260924-V1'
SESSION='01a04939-b5c7-7a03-ba2d-ef3343d62cfd'
REPORT=REPO/'experiment-reports/servers/server4/historical-update-timeaxis-20260924-v1'
AUDIT=REPO/'audits/servers/server4/historical-update-timeaxis-20260924-v1'
MODEL=Path('/data/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2')
DEPS=Path('/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/deps-transformers-4.44.2')
DATA=Path('/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1')
BASE=Path('/data/janghj/ODE-edit/local/fixed10k-native-baselines/attempt-v1/output')
FAMILIES=('BASE_ALPHAEDIT','BASE_MEMIT')
TIMES=(0,1,5,10,20,30,40,50,60,70,80,90,100)
KEYS=tuple(f'model.layers.{l}.mlp.down_proj.weight' for l in range(4,9))

def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def historical_digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
def tensor_sha(t):
    x=t.detach().cpu().contiguous()
    h=hashlib.sha256(str((str(x.dtype),list(x.shape))).encode())
    h.update(memoryview(x.numpy()).cast('B'));return h.hexdigest()
def boolstr(x):return x=='True' if isinstance(x,str) else bool(x)
def active(f,t):return not boolstr(f['same_batch_conflict']) and (not f['first_later_conflict_batch'] or int(f['first_later_conflict_batch'])>t)
def record(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial')
    with tmp.open('x') as f:json.dump(v,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.link(tmp,p);tmp.unlink();return record(p)
