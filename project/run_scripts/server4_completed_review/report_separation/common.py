"""Constants and compact publication I/O; scientific raw paths are never inputs."""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

WT=Path(__file__).resolve().parents[4]
BASE=WT/'experiment-reports/servers/server4'
LOCAL=Path('/data/janghj/ODE-edit/local/report-separation/20260916-v1')
TASK='ODEEDIT-S06-CAKE-BASELINE-CAP-REPORT-SEPARATION-SH4-V1'
CAKE_PARENT=BASE/'cake-native-lifelong-b100x100-2026-09-15-v1'
CAP_PARENT=BASE/'ep-tw1-alpha-cap-sweep-2026-09-15-v1'
CAKE_OLD=CAKE_PARENT/'completed-review-v1'
CAP_OLD=CAP_PARENT/'completed-review-v1'
CAKE=CAKE_PARENT/'completed-review-v2'
CAP=CAP_PARENT/'completed-review-v2'
INDEX=BASE/'completed-experiments-review-2026-09-16-v2'
BASELINE=BASE/'blue-native-lifelong-comprehensive-review-2026-09-11-v1'
AUDIT=WT/'audits/servers/server4/2026-09-16-cake-baseline-cap-report-separation'
EVIDENCE_COMMIT='307ba7ae9426324404cfa83df3b4b766846bca74'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):
    p=Path(p);return dict(path=str(p.relative_to(WT)),bytes=p.stat().st_size,sha256=sha(p))
def readcsv(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def write(p,text):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:f.write(text.rstrip()+'\n')
def save(p,data):write(p,json.dumps(data,ensure_ascii=False,indent=2))
def table(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r));p=Path(p)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)
def mdtable(headers,rows):
    cell=lambda x:str(x).replace('|','\\|').replace('\n',' ')
    return '\n'.join('| '+' | '.join(map(cell,row))+' |' for row in [headers,['---']*len(headers),*rows])
def label(raw):
    if raw in ('BASE_ALPHAEDIT','BASE_ALPHAEDIT_NATIVE'):return 'BASE_ALPHAEDIT_NATIVE'
    if raw in ('BASE_MEMIT','BASE_MEMIT_NATIVE'):return 'BASE_MEMIT_NATIVE'
    for family in ('AlphaEdit','MEMIT'):
        if raw in (family+'_ORIGINAL',family+'_BLUE(L4+L8)',family+'_BLUE (L4+L8)'):
            return family+'_BLUE(L4+L8)'
        m=re.fullmatch(family+r'(?:_BLUE)?_L([4-8])_ONLY',raw)
        if m:return family+f'_BLUE(L{m[1]}-only)'
    if raw in ('CAKE_NATIVE','W0'):return raw
    raise ValueError(f'Non-allowlisted baseline arm: {raw}')
def label_text(text):
    for family in ('AlphaEdit','MEMIT'):
        for layer in range(4,9):
            for old in (f'{family}_BLUE_L{layer}_ONLY',f'{family}_L{layer}_ONLY'):
                text=text.replace(old,f'{family}_BLUE(L{layer}-only)')
    return text
def git(*args):return subprocess.check_output(['git',*args],cwd=WT,text=True).strip()
def protected_trees():
    paths=[BASE/'completed-experiments-review-2026-09-16-v1',CAKE_OLD,CAP_OLD,BASELINE,
           BASE/'blue-l4-l8-lifelong-b100x100-review-2026-09-09-v3']
    return [dict(path=str(p.relative_to(WT)),git_tree=git('rev-parse',f'HEAD:{p.relative_to(WT)}'),
                 evidence_tree=git('rev-parse',f'{EVIDENCE_COMMIT}:{p.relative_to(WT)}')) for p in paths]
def with_toc(text):
    lines=text.splitlines();toc=[];out=[];n=0
    for line in lines:
        if line.startswith('## '):
            n+=1;toc.append(f'- [{line[3:]}](#section-{n})');out.extend([f'<a id="section-{n}"></a>',''])
        out.append(line)
    i=next(i for i,l in enumerate(out) if l.startswith('<a id='))
    out[i:i]=['## 목차','',*toc,'']
    return '\n'.join(out)
