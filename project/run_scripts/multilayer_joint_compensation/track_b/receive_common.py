"""SH2 sole allowlist rsync receiver, invoked only after explicit SH1 READY."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import tempfile

SOURCE_PREFIX='/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-multilayer-joint-edit-a-v1/local/multilayer-joint-compensation/20260911-v1/common'
LANDING=Path('/mnt/raid5/janghj/ODE-edit/local/multilayer-joint-compensation/20260911-v1/imports')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8<<20),b''):h.update(data)
    return h.hexdigest()


def publish_noreplace(source,destination):
    libc=ctypes.CDLL(None,use_errno=True)
    code=libc.renameat2(-100,os.fsencode(source),-100,os.fsencode(destination),1)
    if code:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()),str(destination))


def main():
    p=argparse.ArgumentParser();p.add_argument('--ready-path',required=True);p.add_argument('--ready-sha',required=True)
    p.add_argument('--bundle',required=True);args=p.parse_args()
    source_ready=Path(args.ready_path)
    source_ready.relative_to(SOURCE_PREFIX)
    if source_ready.name!='READY.json' or '/' in args.bundle or args.bundle.startswith('.'):
        raise ValueError('EXACT_READY_AND_BUNDLE_REQUIRED')
    destination=LANDING/args.bundle
    if destination.exists():raise FileExistsError('EXISTING_SEALED_BUNDLE_NO_OVERWRITE')
    script='import pathlib,stat,sys; p=pathlib.Path('+repr(str(source_ready))+'); s=p.lstat(); assert stat.S_ISREG(s.st_mode); p.resolve().relative_to('+repr(SOURCE_PREFIX)+'); sys.stdout.buffer.write(p.read_bytes())'
    raw=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','rke-server1','python3','-c',shlex.quote(script)],timeout=60)
    if hashlib.sha256(raw).hexdigest()!=args.ready_sha:raise ValueError('SOURCE_READY_SHA_MISMATCH')
    ready=json.loads(raw)
    if ready['status']!='COMMON_GPU_PREPARED_READY_FOR_SH2_VERIFY':raise ValueError('SOURCE_NOT_READY')
    source_root=source_ready.parent
    members=ready['members'];relative=[]
    for m in members:
        path=Path(m['path']);path.relative_to(SOURCE_PREFIX)
        rel=path.relative_to(source_root)
        if '..' in rel.parts or '\n' in str(rel):raise ValueError('ALLOWLIST_PATH_ESCAPE')
        relative.append(str(rel))
    if len(set(relative))!=len(relative):raise ValueError('DUPLICATE_ALLOWLIST')
    total=sum(m['bytes'] for m in members)+len(raw)
    LANDING.mkdir(parents=True,exist_ok=True,mode=0o700)
    free=shutil.disk_usage(LANDING).free;inodes=os.statvfs(LANDING).f_favail
    reserve=8<<30
    if free<total+reserve or inodes<len(relative)+100:raise RuntimeError(f'CAPACITY_HOLD required={total} reserve={reserve} free={free} inodes={inodes}')
    staging=Path(tempfile.mkdtemp(prefix='.'+args.bundle+'-',dir=LANDING))
    payload=staging/'payload';payload.mkdir(mode=0o700)
    allowlist=staging/'allowlist.txt'
    with allowlist.open('x') as f:f.write('\n'.join(relative+['READY.json'])+'\n')
    command=['rsync','-a','--protect-args','--relative','--files-from='+str(allowlist),
             'rke-server1:'+str(source_root)+'/',str(payload)+'/']
    subprocess.run(command,check=True)
    if sha(payload/'READY.json')!=args.ready_sha:raise ValueError('READY_CHANGED_DURING_TRANSFER')
    mapping=[]
    for m,rel in zip(members,relative):
        path=payload/rel;s=path.lstat()
        if not stat.S_ISREG(s.st_mode) or s.st_size!=m['bytes'] or sha(path)!=m['sha256']:
            raise ValueError('DESTINATION_MEMBER_MISMATCH:'+rel)
        mapping.append(dict(source_path=m['path'],destination_relative='payload/'+rel,bytes=s.st_size,sha256=m['sha256']))
    # Reuse source CPU tensor/schema evidence only after exact full-member SHA.
    receipt=dict(status='COMMON_DESTINATION_FULL_SHA_VERIFIED',source_ready=args.ready_path,
        source_ready_sha256=args.ready_sha,source_head=ready['source_head'],entry=ready['entry'],
        destination=str(destination),count=len(mapping),bytes=sum(m['bytes'] for m in mapping),
        capacity=dict(free_before=free,inodes_before=inodes,reserve_bytes=reserve),members=mapping,
        source_members_root=ready['members_root'],external_history=ready.get('external_history'),
        CPU_schema='SOURCE_EXACT_SHA_BOUND; independent runtime load/shape required before GPU',
        actual_cross_server_model_parity='NOT_YET_RUN',rsync_sole_owner='SH2',overwrite=0,
        GPU=0,Slurm=0,source_mutation=0,scientific_promotion=False)
    with (staging/'receiver-ready.json').open('x') as f:json.dump(receipt,f,indent=2)
    publish_noreplace(staging,destination)
    print(json.dumps(dict(path=str(destination/'receiver-ready.json'),sha256=sha(destination/'receiver-ready.json'),status=receipt['status'])))


if __name__=='__main__':main()
