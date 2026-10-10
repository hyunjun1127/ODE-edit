import os,stat
from pathlib import Path
from official.runners.server1.common import read,member
from official.experiments.prepare import write_new
O=Path(__file__).parent;R=Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1')
pairs=[('cf-checkpoint-r1/configs/llama3-cf-memit_fe.json','cf-checkpoint-r1/runs/qualification-memit_fe'),('cf-display-score-repair-r1/preparation-r1/configs/cf-memit_fe.json','cf-display-score-repair-r1/preparation-r1/runs/cf-memit_fe'),('no-gpu-qualification-r1/preparation-r1/configs/zsre-memit_fe.json','no-gpu-qualification-r1/preparation-r1/runs/zsre-memit_fe')]
files=[]
for config,out in pairs:
 c=read(R/config);assert c['method']=='MEMIT_FE'
 for f in (R/out).glob('**/checkpoint/*.pt'):
  assert f.is_file() and not f.is_symlink()
  p=read(f.parent/'latest.json');assert p['file']==f.name
  if out.startswith('cf-checkpoint-r1/'):assert c['qualification_outputs']['MEMIT_FE']==str(R/out)
  else:assert read(R/out/'COMPLETE.json')['identity']['config_sha256']==c['config_sha256']
  s=f.stat();files.append(dict(path=str(f),bytes=s.st_size,inode=s.st_ino,mtime_ns=s.st_mtime_ns,config=member(R/config),pointer=member(f.parent/'latest.json')))
write_new(O/'native-checkpoint-delete-manifest.json',dict(files=files,count=len(files),bytes=sum(x['bytes'] for x in files)))
for m in files:
 p=Path(m['path']);s=p.lstat();assert stat.S_ISREG(s.st_mode) and (s.st_ino,s.st_size,s.st_mtime_ns)==(m['inode'],m['bytes'],m['mtime_ns']);p.unlink()
write_new(O/'native-checkpoint-deleted.json',dict(count=len(files),bytes=sum(x['bytes'] for x in files),all_absent=all(not Path(x['path']).exists() for x in files),backup=False))
print('DELETED_NATIVE',len(files),sum(x['bytes'] for x in files))
