"""Exact small-file migration; no checkpoint/model transfer or source mutation."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from .common import save, record, require, sha

ROOT = Path('/mnt/raid5/janghj/ODE-edit/local/joint-multilayer-bs1/20260929-v1')
REMOTE = '/data/janghj/ODE-edit/local/joint-multilayer-bs10/20260929-v1/migration-to-s2-r1/'

def remote_bytes(path):
    require(path.startswith(REMOTE) and '..' not in Path(path).parts, 'SOURCE_SCOPE')
    code = ('import os,stat,sys; p=sys.argv[1]; s=os.lstat(p); '
            'assert stat.S_ISREG(s.st_mode) and s.st_uid==1025 and s.st_nlink==1; '
            'sys.stdout.buffer.write(open(p,"rb").read())')
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','rke-server4',
        'python3 -c '+shlex.quote(code)+' '+shlex.quote(path)])

def receive():
    dest=ROOT/'inputs/server4-handoff';dest.mkdir(parents=True,exist_ok=True)
    name='receiver-exact-allowlist-s2.json';p=dest/name
    if not p.exists():
        b=remote_bytes(REMOTE+name)
        with p.open('xb') as f:f.write(b)
    manifest=json.loads(p.read_bytes());rows=[]
    require(manifest['destination_root']==str(dest), 'LANDING')
    for m in manifest['files']:
        rel=Path(m['destination_relative_path'])
        require(not rel.is_absolute() and '..' not in rel.parts,'TRAVERSAL')
        target=dest/rel
        require(str(target)==m['destination_path'],'DESTINATION')
        require(m['bytes']<8*2**20,'SMALL_FILE_ONLY')
        if target.exists():
            require(target.is_file() and not target.is_symlink() and target.stat().st_size==m['bytes'] and sha(target)==m['sha256'],'EXISTING_CONFLICT')
            disposition='REUSE'
        else:
            b=remote_bytes(m['source_path'])
            require(len(b)==m['bytes'] and hashlib.sha256(b).hexdigest()==m['sha256'],'TRANSFER_HASH')
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(b)
            disposition='RECEIVED'
        rows.append(dict(source_path=m['source_path'],destination=record(target),disposition=disposition))
    cancellation=json.loads((dest/'cancellation-receipt.json').read_bytes())
    expected={f'55090_{i}' for i in range(9)}|{'55091'}
    require({r['job_id'] for r in cancellation['jobs']}==expected,'CANCEL_JOB_SET')
    require(all(r['state'].startswith('CANCELLED') and r['elapsed_seconds']==0 and not r['allocated_tres'] for r in cancellation['jobs']),'CANCEL_NONACTIVE')
    require(cancellation['active_pending_empty'] and cancellation['parent_allocation_gpu_seconds']==0,'CANCEL_RESOURCE')
    save(ROOT/'inputs/transfer-receipt.json',dict(manifest=record(p),members=rows,cancellation=record(dest/'cancellation-receipt.json'),
        source_keep=True,heavy_transfer_bytes=0,bytes=sum(m['destination']['bytes'] for m in rows),files=len(rows)))
    print('VERIFIED',len(rows),sum(m['destination']['bytes'] for m in rows))

if __name__=='__main__':receive()
