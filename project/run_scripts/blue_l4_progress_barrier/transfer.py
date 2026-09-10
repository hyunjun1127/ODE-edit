"""Approved, single-owner, exact allowlist import; no source mutation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

SOURCE = '/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1'
DEST = Path('/mnt/raid5/janghj/ODE-edit/local/blue-l4-progress-barrier/attempt-v1')

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def save(p, x):
    with p.open('x') as f: json.dump(x, f, indent=2, sort_keys=True)

REMOTE = '''
import pathlib,json,hashlib
r=pathlib.Path(SOURCE)
paths=[r/'input.lock.json',r/'imports/config.json']
for entry,rev in [('Early','r4'),('Middle','r1'),('Late','r4')]:
 d=r/f'A/{entry}/native-{rev}'
 names=['prepared.pt','prepared-receipt.json','runtime.json','terminal.json','N-generation.json']
 names += [s+'-'+k+'.json' for s in ['W0','ENTRY','N'] for k in ['full','structure','train']]
 paths += [d/n for n in names if (d/n).is_file()]
for b in [10,50,90]:
 paths += [r/f'imports/entries/B{b:03d}/W-method-state.pt']
 paths += [r/f'imports/entries/B{b+1:03d}/'+pathlib.Path(n) for n in []]
 for n in ['native-targets.pt','entry.json','contexts.json']:
  p=r/f'imports/entries/B{b+1:03d}'/n
  if p.is_file(): paths.append(p)
for folder in ['blue-source','historical']:
 paths += [p for p in (r/'imports'/folder).rglob('*') if p.is_file() and (p.suffix=='.py' or p.name=='globals.yml')]
members=[]
for p in sorted(set(paths)):
 assert p.is_file() and not p.is_symlink(),str(p)
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 members.append(dict(relative=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=h.hexdigest()))
print(json.dumps(members))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory-only',action='store_true');a=p.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    inv=DEST/'transfer-source-manifest.json'
    if inv.exists(): members=json.loads(inv.read_text())['members']
    else:
        script='SOURCE='+repr(SOURCE)+'\n'+REMOTE
        members=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','rke-server1','python3','-'],input=script.encode()))
        save(inv,dict(source=SOURCE,members=members,owner='SH2',instruction_id='ODEEDIT-S06-BLUE-L4-PROGRESS-PRESERVING-BARRIER-SH2-V1'))
    total=sum(m['bytes'] for m in members)
    print(json.dumps(dict(files=len(members),bytes=total,free=shutil.disk_usage(DEST).free)),flush=True)
    if a.inventory_only:return
    stage=DEST/'imports.partial';target=DEST/'imports'
    if target.exists(): raise FileExistsError('SEALED_IMPORT_EXISTS_VERIFY_ONLY: '+str(target))
    stage.mkdir(mode=0o700,exist_ok=False)
    assert shutil.disk_usage(DEST).free > total+100*(1<<30),'INSUFFICIENT_DISK_RESERVE'
    allow=DEST/'transfer-allowlist.txt'
    with allow.open('x') as f:f.write(''.join(m['relative']+'\n' for m in members))
    subprocess.run(['rsync','--archive','--relative','--partial','--protect-args','--files-from='+str(allow),'rke-server1:'+SOURCE+'/',str(stage)+'/'],check=True)
    for m in members:
        path=stage/m['relative']
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size==m['bytes'] and sha(path)==m['sha256'],m['relative']
    os.rename(stage,target)
    save(DEST/'transfer-receipt.json',dict(status='FULL_SHA_PASS',files=len(members),bytes=total,root=str(target),manifest_sha=sha(inv),overwrite=0,source_mutation=0))
    print('TRANSFER_SEALED',target,flush=True)

if __name__=='__main__':main()
