import csv
import hashlib
import json
from pathlib import Path
import numpy as np

BASE=Path('/data/janghj/ODE-edit')
SAMPLE_ROOT='40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd'
ARMS=['BLUE','BLUE_L4_ONLY','BLUE_L8_ONLY','JVP','JVP_L8','O_NATIVE']
BLUE_ROOTS={
 'BLUE':BASE/'local/blue-alphaedit-sequential-comparison/attempt-v1/execution-tech-r2/main-llama',
 'BLUE_L4_ONLY':BASE/'local/blue-alphaedit-l4-oneshot-sequential/attempt-v1/main-llama',
 'BLUE_L8_ONLY':BASE/'local/blue-alphaedit-l8-oneshot-sequential/attempt-v1/main-llama'}
JROOT=BASE/'local/state/alpha-jv-migration-server4-20260907/tech-r1'
JCHAIN=JROOT/'chain-4-llama3-8b-inst-L8_ONLY_NATIVE'
REF_REL='experiment-reports/servers/server2/alpha-native-response-v31-sequential-routing-2026-09-06-v1/main-four-terminal-v1'
REVIEW_REL='experiment-reports/servers/server4/alpha-jv-l8-only-sequential1000-review-2026-09-07-v1'
OUT_REL='experiment-reports/servers/server4/blue-alphaedit-fivearm-sequential1000-review-2026-09-07-v1'
NA='NOT_AVAILABLE'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvread(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def stats(v):
 a=np.asarray(v,dtype=np.float64);assert a.size and np.isfinite(a).all()
 return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),q25=float(np.quantile(a,.25)),q75=float(np.quantile(a,.75)),p90=float(np.quantile(a,.9)),max=float(a.max()))
def fmt(x):
 if x is None or x=='' or str(x).startswith(('NOT_','NA')):return 'NA'
 try:return f'{float(x):.6g}'
 except (ValueError,TypeError):return str(x)
def table(rows,cols):
 return '\n'.join(['|'+'|'.join(cols)+'|','|'+'|'.join(['---']*len(cols))+'|']+['|'+'|'.join(fmt(r.get(k,NA)).replace('|','/') for k in cols)+'|' for r in rows])+'\n'

def verify_members(root,members,group):
 out=[]
 for m in members:
  p=Path(root)/m['path'];assert p.is_file() and not p.is_symlink(),str(p)
  s=sha(p);assert s==m['sha256'] and p.stat().st_size==m['bytes'],str(p)
  out.append(dict(group=group,path=str(p),bytes=p.stat().st_size,sha256=s,status='FULL_REHASH_PASS'))
 return out
