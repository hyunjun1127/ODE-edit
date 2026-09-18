"""Exact stopped-B1 allowlist, remote read-only hash, create-once pull/seal."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

REMOTE_ROOT = '/data/janghj/ODE-edit/local/'
PREFIXES = ('single-layer-edit-preserving-correction/', 'local-z-adaptive-allocation/',
            'bg1-c4-ours-first/', 'ep-tw1-c4/')
REMOTE = r'''
import hashlib,json,pathlib,stat
R=pathlib.Path('/data/janghj/ODE-edit/local')
E=R/'single-layer-edit-preserving-correction/20260918-v1'
B=E/'M/attempt-metadata-r1/episodes/b001/attempt-v1'
paths=set(p for p in B.rglob('*') if p.is_file())
lockpath=E/'M/attempt-metadata-r1/execution.lock.json'
lock=json.loads(lockpath.read_text())
paths.update([lockpath,E/'inputs/base-binding.json',E/'reuse/observer-binding-r1.json',E/'reuse/b001-native-binding.json'])
for k in ('P_star_basis','cold_capsule','teacher_manifest'):
 paths.add(pathlib.Path(lock[k]['path']))
for k in ('config4','reused_native_b1'): paths.add(pathlib.Path(lock[k]))
for k in ('technical_evidence',):
 v=lock.get(k)
 if isinstance(v,dict) and 'path' in v:paths.add(pathlib.Path(v['path']))
 elif isinstance(v,str):paths.add(pathlib.Path(v))
reuse=json.loads((E/'reuse/observer-binding-r1.json').read_text())
for k in ('N4_B1','W0_first1000','prior_lock','commit'):paths.add(pathlib.Path(reuse[k]['path']))
tm=json.loads(pathlib.Path(lock['teacher_manifest']['path']).read_text())
for k in ('reference_tokens','splits','lock'):paths.add(pathlib.Path(tm[k]['path']))
paths.update(pathlib.Path(m['path']) for m in tm['cache_shards'])
reference=pathlib.Path(lock['reference_root'])/'manifest.json'
if reference.is_file():paths.add(reference)
members=[]
for p in sorted(paths):
 rel=p.relative_to(R).as_posix()
 assert rel.startswith(('single-layer-edit-preserving-correction/','local-z-adaptive-allocation/','bg1-c4-ours-first/','ep-tw1-c4/'))
 assert '/S/' not in '/'+rel and '/sequential' not in rel
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and p.resolve()==p,'NONREGULAR_OR_SYMLINK'
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 t=p.stat();assert (s.st_size,s.st_mtime_ns,s.st_ino)==(t.st_size,t.st_mtime_ns,t.st_ino),'SOURCE_CHANGED'
 members.append(dict(path=str(p),relative=rel,bytes=s.st_size,sha256=h.hexdigest(),mode=oct(stat.S_IMODE(s.st_mode)),mtime_ns=s.st_mtime_ns))
print(json.dumps(dict(source_host='rke-server4',source_root=str(R),scope='STOPPED_M_B001_AND_EXACT_COMPANIONS_ONLY',members=members,total_bytes=sum(m['bytes'] for m in members),S_live_access=0)))
'''

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def once(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')

def inventory(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    data=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',
        'rke-server4','python3 -c '+shlex.quote(REMOTE)],text=True))
    if data['total_bytes']>32*(1<<30):raise ValueError('IMPORT_SIZE_EXCEEDS_PREDECLARED_32G')
    for m in data['members']:
        if not m['relative'].startswith(PREFIXES) or '..' in Path(m['relative']).parts:
            raise ValueError('ALLOWLIST_PREFIX')
    once(root/'allowlist.json',data)
    with (root/'files.nul').open('xb') as f:
        f.write(b''.join(m['relative'].encode()+b'\0' for m in data['members']))
    return data

def pull(root):
    root=Path(root);plan=json.loads((root/'allowlist.json').read_text())
    stage=root/'staging';destination=root/'sealed'
    if destination.exists():raise ValueError('CREATE_ONCE_DESTINATION_EXISTS')
    if shutil.disk_usage(root).free<plan['total_bytes']+32*(1<<30):raise ValueError('DISK_RESERVE')
    stage.mkdir(exist_ok=False)
    subprocess.run(['rsync','-rt','--ignore-existing','--from0','--files-from='+str(root/'files.nul'),
        '-e','ssh -o BatchMode=yes -o ConnectTimeout=10','rke-server4:'+REMOTE_ROOT,str(stage)+'/'],check=True)
    for m in plan['members']:
        p=stage/m['relative']
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=m['bytes'] or sha(p)!=m['sha256']:
            raise ValueError('RECEIVED_IDENTITY:'+m['relative'])
    # Same-host atomic publication only after complete size/SHA agreement.
    stage.rename(destination)
    once(root/'receiver-seal.json',dict(status='FULL_SIZE_SHA_VERIFIED',allowlist_sha256=sha(root/'allowlist.json'),
        member_count=len(plan['members']),total_bytes=plan['total_bytes'],destination=str(destination),
        sole_destination_writer='SH1',nonoverwrite=True,source_mutation=0,live_S_access=0,
        custom_import_exception='USER_EXACT_ALLOWLIST_AUTHORIZED',model_load=0,GPU=0))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['inventory','pull']);p.add_argument('--root',required=True)
    a=p.parse_args();(inventory if a.mode=='inventory' else pull)(a.root)
    print(json.dumps(dict(status=a.mode.upper()+'_DONE',root=a.root)))

if __name__=='__main__':main()
