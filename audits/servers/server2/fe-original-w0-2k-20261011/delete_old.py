"""Explicit user-authorized FE checkpoint deletion; never recursive removal."""
import json,os,stat,subprocess
from pathlib import Path
from official.experiments.prepare import file_sha,write_new,digest
B=Path('/mnt/raid5/janghj/ODE-edit/local');O=Path(__file__).parent
specs=[
 ('official-baselines-server2/20261008-r1/registration-no-gpu-qual-r1/ZSRE_MEMIT_FE','checkpoints','result.json','61734'),
 ('official-baselines-server2/20261008-r1/registration-cf-display-r1/CF_MEMIT_FE','checkpoints','result.json','61780'),
 ('qwen-baseline-mask-cold-rerun-20261010/registration-r1/runs/qwen25-cf-memit_fe','checkpoint','terminal.json','62077'),
 ('qwen-baseline-mask-cold-rerun-20261010/registration-r1/runs/qwen25-zsre-memit_fe','checkpoint','terminal.json','62085'),
 ('fe-author-hparams-2k-20261010/configs-r1/runs/cf','checkpoint','COMPLETE.json','62531'),
 ('fe-author-hparams-2k-20261010/configs-r1/runs/zsre','checkpoint',None,'62532')]
def read(p):return json.loads(p.read_text())
def info(p):
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_nlink==1
 assert p.resolve()==p and p.is_relative_to(B)
 return dict(path=str(p),bytes=s.st_size,dev=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def accounting():
 ids=[s[3] for s in specs]+['62868','62872','61946']
 text=subprocess.check_output(['sacct','-X','-nP','-j',','.join(ids),'--format=JobIDRaw,User,State,NodeList'],text=True)
 rows={v[0]:v for v in (s.split('|') for s in text.splitlines())}
 for j in ids:
  r=rows[j];assert r[1]=='janghj' and r[2].split()[0] in ('COMPLETED','CANCELLED','FAILED'),r
 return text
before_account=accounting();items=[]
for rel,cpdir,receipt,job in specs:
 folder=B/rel;pointer=read(folder/cpdir/'latest.json');p=folder/cpdir/pointer['file']
 if 'fe-author' in rel:
  reg=read(B/'fe-author-hparams-2k-20261010/registration-r1/submission.json');r=reg['jobs'][folder.name]
  cfg=read(Path(r['config']['path']));assert str(r['job_id'])==job and cfg['method']=='MEMIT_FE_HISTORY'
  identity=read(folder/'commits'/f"batch-{pointer['batch']:02d}.json")['identity']
  evidence=r['config'];assert file_sha(evidence['path'])==evidence['sha256']
 else:
  t=read(folder/receipt);identity=t['checkpoint_identity']
  if receipt=='result.json':assert t['method']=='MEMIT_FE'
  else:
   cfgpath=B/'qwen-baseline-mask-cold-rerun-20261010/registration-r1/configs'/f'{folder.name}.json'
   cfg=read(cfgpath);assert cfg['method']=='MEMIT_FE' and t['actual_job_id']==job
  evidence=dict(path=str(folder/receipt),sha256=file_sha(folder/receipt))
 assert digest(identity)==pointer['identity_sha256']
 item=info(p);assert file_sha(p)==pointer['sha256'];assert info(p)==item
 item.update(sha256=pointer['sha256'],job=job,identity=identity,receipt=evidence,pointer=dict(path=str(folder/cpdir/'latest.json'),sha256=file_sha(folder/cpdir/'latest.json')))
 items.append(item)
 # No silently retained checkpoint-only temporary/backup files.
 assert {q.name for q in (folder/cpdir).iterdir()}=={'latest.json',pointer['file']}
# All local checkpoint payload paths, for replica discovery (no destructive globs).
same_size=[]
sizes={i['bytes'] for i in items};allowed={i['path'] for i in items};scanned=0
for d,ds,fs in os.walk(B):
 ds[:]=[x for x in ds if x not in ('.git','.venv','source','wandb','node_modules','datasets','models','snapshots','blobs','stats','author-FE')]
 for f in fs:
  p=Path(d)/f
  if p.suffix in ('.pt','.pth','.safetensors','.bin'):
   scanned+=1
   if str(p) not in allowed and p.stat().st_size in sizes:
    h=file_sha(p);same_size.append(dict(path=str(p),sha256=h,is_replica=h in {i['sha256'] for i in items}))
assert not any(x['is_replica'] for x in same_size),'EXTRA_REPLICA_REQUIRES_IDENTITY'
# No open writer or consumer descriptors on these exact objects.
keys={(i['dev'],i['inode']) for i in items};open_matches=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if proc.stat().st_uid!=os.getuid():continue
  for fd in (proc/'fd').iterdir():
   try:
    s=fd.stat()
    if (s.st_dev,s.st_ino) in keys:open_matches.append(str(fd))
   except (FileNotFoundError,PermissionError):pass
 except (FileNotFoundError,PermissionError):pass
assert not open_matches,open_matches
write_new(O/'delete-before.json',dict(items=items,accounting=before_account,payload_files_scanned=scanned,
 same_size_non_targets=same_size,open_descriptors=open_matches,scope='six exact manifest-bound FE payloads; other methods and shared assets KEEP'))
free_before=os.statvfs(B).f_bavail*os.statvfs(B).f_frsize
accounting()
for i in items:
 p=Path(i['path']);assert all(info(p)[k]==i[k] for k in info(p));p.unlink();assert not p.exists()
 tomb=dict(instruction='USER-FE-ORIGINAL-W0-RESET-20261011-R1',deleted=i,backup_created=False,recovery='NO_VERIFIED_RECOVERY_COPY; old metadata is historical only')
 write_new(p.parent/'USER_FE_WITHDRAWN_CHECKPOINT_DELETED.json',tomb)
free_after=os.statvfs(B).f_bavail*os.statvfs(B).f_frsize
write_new(O/'delete-after.json',dict(count=len(items),deleted_bytes=sum(i['bytes'] for i in items),
 filesystem_available_delta=free_after-free_before,filesystem_delta_not_exclusive_to_this_task=True,
 paths=[i['path'] for i in items],all_absent=all(not Path(i['path']).exists() for i in items),
 backup_created=False,recovery='NOT_AVAILABLE_FROM_THESE_DELETED_PAYLOADS',raw_source_metadata_KEEP=True))
print(json.dumps(read(O/'delete-after.json')))
