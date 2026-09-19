"""Exact-allowlist receiver. Never writes the remote source or copies W/M."""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from .common import *

def plan():
    out=ATTEMPT/'inputs'; out.mkdir(parents=True, exist_ok=True)
    authority=read(REPO/'messages/head/2026-09-20-sh2-checkpoint-mechanism-authority.json')
    for m in authority['members']:
        p=REPO/m['path']; assert p.stat().st_size==m['bytes'] and sha256(p)==m['sha256'], str(p)
    write_json(out/'authority-full-read.json',dict(status='FULL_READ_EXACT',members=authority['members'],
      authority_commit='af87b3fbab250e88a34c6882252a8d484b450f56',source_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
      resource_override='4440e111251cc6a0a924e9e5b4589d61333f14dc',project_gpu_cap=2,task_lanes=2,save_checkpoints=False,
      helper_boundary='canonical root check PASS; new worktree helper lacks local config; explicit envelope permits worktree and task-local identity receipt without global helper changes'))
    for m in authority['members']:
        dest=ATTEMPT/'authority'/m['path']; dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f: f.write((REPO/m['path']).read_bytes())
    cpmanifest=read(INITIAL/'source/checkpoint-manifest.json')
    closure={m['source']:INITIAL/'source'/m['relative'] for m in cpmanifest['source_closure']}
    lock_path=closure[S4RUN+'/execution.lock.json']; lock=read(lock_path)
    terminal_path=closure[S4CELL+'/terminal.json']; terminal=read(terminal_path)
    expected={}
    def add(m,path=None): expected[path or m['path']]=dict(bytes=m['bytes'],sha256=m['sha256'])
    for m in cpmanifest['source_closure']:
        if m['source'] in [S4RUN+'/execution.lock.json',S4RUN+'/local-source.tar',S4CELL+'/runtime.json',S4CELL+'/terminal.json']:
            add(m,m['source'])
    for m in terminal['manifest_members']:
        rel=m['path']; parts=rel.split('/')
        if len(parts)!=2 or not parts[0].startswith('B'): continue
        b=int(parts[0][1:]); name=parts[1]
        permitted=name in ['current.json','commit.json','entry.json','native-observation.json','contexts.json']
        permitted |= name=='seen-full.json' and b in CONTRACT['inputs']['checkpoint_batches']
        permitted |= name=='native-targets.pt' and b in CONTRACT['inputs']['target_batches']
        if permitted: add(m,S4CELL+'/'+rel)
    for m in lock['members']:
        p=m['path']
        permitted=p==SAMPLE or p==CONTRACT['paths']['dataset'].replace('/mnt/raid5','/data')
        permitted |= p==S4ROOT+'/local/blue-lifelong-b100x100/attempt-v1/configs/AlphaEdit-L4_ONLY.json'
        permitted |= p.startswith(S4OLD+'/source-tech-r2/project/')
        permitted |= p.startswith(S4OLD+'/deps-transformers-4.44.2/')
        permitted |= p.startswith(S4OLD+'/blue-source/') and '/memit/' not in p and '/hparams/MEMIT/' not in p
        permitted |= p.startswith(S4RUN+'/lifelong/')
        if permitted: add(m)
    mapping={}; members=[]; missing=[]
    for source,m in sorted(expected.items()):
        candidates=[]
        if source in closure: candidates.append(closure[source])
        if source.startswith(S4ROOT+'/'):
            rel=source[len(S4ROOT)+1:]
            candidates += [ROOT/'local/checkpoint-archives/server4-migration-20260911-v1/new177-v1/closure'/rel,
                           ROOT/'local/checkpoint-archives/server4-migration-20260911-v1/new177-v1/supplement'/rel]
        if source==SAMPLE: candidates.append(ROOT/'local/datasets/counterfact-fixed-10k-v1/source-sample.lock.json')
        if source==CONTRACT['paths']['dataset'].replace('/mnt/raid5','/data'): candidates.append(Path(CONTRACT['paths']['dataset']))
        for remote,local in [
          (S4OLD+'/blue-source/',ROOT/'local/blue-l4-progress-barrier/attempt-v1/imports/imports/blue-source'),
          (S4OLD+'/source-tech-r2/',ROOT/'local/fixed10k-preedit-eval/attempt-v1/helper-source'),
          (S4OLD+'/deps-transformers-4.44.2/',ROOT/'local/fixed10k-preedit-eval/attempt-v1/deps-transformers-4.44.2')]:
            if source.startswith(remote): candidates.append(local/source[len(remote):])
        if source.startswith(S4OLD+'/source-tech-r2/'):
            candidates.append(REPO/source.split('/source-tech-r2/',1)[1])
        good=None
        for p in candidates:
            if p.is_file() and p.stat().st_size==m['bytes'] and sha256(p)==m['sha256']:
                good=p;break
        member=dict(source=source,**m)
        if good:
            mapping[source]=str(good); member.update(mode='LOCAL_EXACT_REUSE',destination=str(good),stat=stat_identity(good))
        else:
            dest=out/'received'/source.lstrip('/')
            member.update(mode='MISSING_TRANSFER',destination=str(dest));missing.append(member)
        members.append(member)
    # Full original files are never overwritten and no checkpoint is a transfer candidate.
    assert not any('W-method-state.pt' in m['source'] for m in missing)
    free=shutil.disk_usage(TASK).free; amount=sum(m['bytes'] for m in missing)
    assert free>amount*2+20*1024**3,(free,amount)
    write_json(out/'staging-plan.json',dict(members=members,missing=missing,local_mapping=mapping,new_bytes=amount,free_bytes=free,source_keep=True))
    print(json.dumps(dict(local_reuse=len(mapping),missing=len(missing),new_bytes=amount,free_bytes=free)))

def receive():
    out=ATTEMPT/'inputs'; plan=read(out/'staging-plan.json'); missing=plan['missing']
    # Remote operation is exact named-file stat/hash only, never traversal or execution of source scripts.
    code='''import json,hashlib,pathlib,stat,sys\nrows=json.loads(sys.stdin.readline()); out=[]\nfor m in rows:\n p=pathlib.Path(m['source']); s=p.lstat(); assert stat.S_ISREG(s.st_mode) and not p.is_symlink(),str(p)\n h=hashlib.sha256()\n with p.open('rb') as f:\n  for b in iter(lambda:f.read(8<<20),b''):h.update(b)\n assert s.st_size==m['bytes'] and h.hexdigest()==m['sha256'],str(p)\n after=p.stat();assert(s.st_size,s.st_mtime_ns,s.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)\n out.append(dict(path=str(p),bytes=s.st_size,sha256=h.hexdigest(),mtime_ns=s.st_mtime_ns,inode=s.st_ino))\nprint(json.dumps(out))'''
    import shlex
    t=time.perf_counter()
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','rke-server4','python3 -c '+shlex.quote(code)],input=json.dumps(missing)+'\n',text=True,capture_output=True,timeout=300)
    if r.returncode: raise RuntimeError('REMOTE_HASH_FAILED: '+r.stderr[-2000:])
    remote=json.loads(r.stdout);write_json(out/'remote-exact-verification.json',dict(members=remote,seconds=time.perf_counter()-t))
    dest=out/'received.partial';dest.mkdir(exist_ok=False)
    allow=out/'rsync-files.txt'
    with allow.open('x') as f: f.write(''.join(m['source'].lstrip('/')+'\n' for m in missing))
    t=time.perf_counter()
    r=subprocess.run(['rsync','-rt','--relative','--ignore-existing','--files-from='+str(allow),'-e','ssh -o BatchMode=yes -o ConnectTimeout=10','rke-server4:/',str(dest)+'/'],capture_output=True,text=True,timeout=900)
    write_json(out/'transfer-command.json',dict(args=r.args,returncode=r.returncode,seconds=time.perf_counter()-t,stdout=r.stdout,stderr=r.stderr))
    if r.returncode: raise RuntimeError('RSYNC_FAILED')
    for m in missing:
        p=dest/m['source'].lstrip('/');assert p.is_file() and p.stat().st_size==m['bytes'] and sha256(p)==m['sha256'],str(p)
    final=out/'received';assert not final.exists();dest.rename(final)
    mapping=plan['local_mapping']|{m['source']:str(final/m['source'].lstrip('/')) for m in missing}
    write_json(out/'source-map.json',mapping)
    write_json(out/'READY.json',dict(status='VERIFIED_SOURCE_KEEP',source_map_sha256=sha256(out/'source-map.json'),members=len(mapping),transferred_files=len(missing),transferred_bytes=sum(m['bytes'] for m in missing),W_M_transfers=0))
    print(json.dumps(read(out/'READY.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','receive']);a=p.parse_args()
    (plan if a.action=='plan' else receive)()
