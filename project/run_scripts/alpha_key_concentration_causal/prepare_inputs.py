"""CPU-only immutable source/panel binding and exact sidecar inventory."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

from .exact_pull import digest

def create(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
    if path.exists():
        assert path.read_bytes()==data, ('CREATE_ONCE_COLLISION',str(path))
    else:
        with path.open('xb') as f: f.write(data)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--repo',required=True)
    a=p.parse_args();root=Path(a.root).resolve();repo=Path(a.repo).resolve();design=root/'inputs/design'
    contract=json.loads((design/'contract.json').read_text())
    old=json.loads((design/'evidence/audits/global/2026-09-22-alphaedit-native-criticality-audit/target-native-execution.lock.json').read_text())
    approval=json.loads((repo/'transfers/approvals/2026-09-23-alpha-key-causal-sh4-inputs.json').read_text())
    for r in approval['members']:
        f=design/r['destination_relative_path'];assert f.stat().st_size==r['bytes'] and digest(f)==r['sha256']
    # Complete parse of the sealed metadata, not regeneration by GH scripts.
    g=json.loads((design/'geometry-panels.json').read_text())['cohorts']
    assert sum(len(x['case_ids']) for x in g.values())==4000
    assert sum(len(x['calibration_case_ids']) for x in g.values())==512
    for x in g.values():
        assert len(x['case_ids'])==1000 and len(x['assessment_case_ids'])==872
        assert set(x['calibration_case_ids']).isdisjoint(x['assessment_case_ids'])
        assert set(x['case_ids'])==set(x['calibration_case_ids'])|set(x['assessment_case_ids'])
    for n in ('history512','neighborhood512'):
        panel=json.loads((design/(n+'.json')).read_text());assert panel['n']==len(panel['records'])==512
    orders=json.loads((design/'order-controls.json').read_text())
    for block in orders.values():
        if not isinstance(block,dict):continue
        for o in block['orders']:
            assert sorted(o['batch_order'])==list(range(block['batch_start'],block['batch_end']+1))
            assert all(o['batch_order'].index(u)<o['batch_order'].index(v) for u,v in block['constraint_edges'])
    cells=list(csv.DictReader((design/'cells.csv').open()))
    assert len(cells)==101
    allowed={'E1','E2','E3','E4-H','E4-W','E4-KR'}
    assert sum(r['phase'] in allowed for r in cells)==94
    # Exact original dependency members are reused. Missing source can only be
    # recovered from a matching Git blob; no source substitutions by filename.
    mapping=[];unresolved=[]
    for m in old['members']:
        f=Path(m['path']);candidate=f
        if not f.is_file():
            marker=old['source_root']+'/'
            if str(f).startswith(marker):
                rel=str(f)[len(marker):]
            elif '/policy-source/' in str(f):
                rel=str(f).split('/policy-source/',1)[1]
            else:
                unresolved.append(m);continue
            candidate=repo/rel
            if not candidate.is_file() or digest(candidate)!=m['sha256']:
                blob=subprocess.run(['git','-C',str(repo),'show',old['source_head']+':'+rel],capture_output=True)
                if blob.returncode or hashlib.sha256(blob.stdout).hexdigest()!=m['sha256']:
                    unresolved.append(m);continue
                candidate=root/'inputs/frozen-helper'/rel
                candidate.parent.mkdir(parents=True,exist_ok=True)
                if candidate.exists():assert digest(candidate)==m['sha256']
                else:
                    with candidate.open('xb') as out:out.write(blob.stdout)
        assert candidate.stat().st_size==m['bytes'] and digest(candidate)==m['sha256'], ('NATIVE_ASSET_DRIFT',str(candidate))
        mapping.append(dict(m,resolved_path=str(candidate),verification='CURRENT_FULL_SHA'))
    create(root/'receipts/native-input-binding-r1.json',dict(status='PASS' if not unresolved else 'INCOMPLETE',members=mapping,unresolved=unresolved,original_lock_sha256=contract['source_lock']['sha256']))
    # Allowed sidecars only, exact12 checkpoint epochs; inventory precedes pull.
    cp=json.loads((design/'checkpoint-transfer-manifest.json').read_text())
    paths=[cp['source_root']+r['path'].split('/')[0]+'/'+name for r in cp['files'] for name in ('contexts.json','commit.json','native-observation.json')]
    code='''import sys,json,os,stat,hashlib
for p in json.loads(sys.stdin.read()):
 if not os.path.exists(p): print(json.dumps(dict(path=p,status='MISSING')));continue
 s=os.lstat(p);assert stat.S_ISREG(s.st_mode),p
 with open(p,'rb') as f: h=hashlib.sha256(f.read()).hexdigest()
 print(json.dumps(dict(path=p,bytes=s.st_size,sha256=h,uid=s.st_uid,inode=s.st_ino,status='PRESENT')))
'''
    raw=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','codex-server2','python3 -c '+shlex.quote(code)],input=json.dumps(paths).encode())
    stats=[json.loads(x) for x in raw.splitlines()]
    members=[dict(source_path=x['path'],destination_relative_path=x['path'][len(cp['source_root']):],bytes=x['bytes'],sha256=x['sha256']) for x in stats if x['status']=='PRESENT']
    create(root/'receipts/sidecar-source-inventory-r1.json',dict(members=stats,source_keep=True))
    create(root/'inputs/sidecar-pull-manifest.json',dict(members=members,source_keep=True))
    print(json.dumps(dict(native_members=len(mapping),unresolved=unresolved,sidecars=len(members),approved_cells=94,followups_not_submitted=7)))

if __name__=='__main__':main()
