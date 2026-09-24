"""CPU-only completed review. Original artifacts are read-only."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path('/data/janghj/ODE-edit/local/native-delayed-write-e3/20260924-v1')
LOCAL=Path(os.environ.get('DELAYED_REVIEW_LOCAL',str(ROOT/'completed-review-r1')))
ATTEMPT=ROOT/'attempt-v1'
OUTPUT=ATTEMPT/'output'
REPO=Path(__file__).resolve().parents[3]
REPORT=Path(os.environ.get('DELAYED_REVIEW_REPORT',str(REPO/'experiment-reports/servers/server4/native-delayed-write-e3-20260924-v1/completed-review-r1')))
AUDIT=Path(os.environ.get('DELAYED_REVIEW_AUDIT',str(REPO/'audits/servers/server4/native-delayed-write-e3-20260924-v1/completed-review-r1')))
INSTRUCTION='ODEEDIT-GH-SH4-DELAYED-E3-COMPLETED-REVIEW-20260924-R1'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def record(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
