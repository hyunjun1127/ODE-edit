from inventory import *
def run():
 root=R/'local/state/alpha-jv-migration-server4-20260907/tech-r1'
 pub=W/'experiment-reports/servers/server4/alpha-jv-l8-only-sequential1000-review-2026-09-07-v1'
 schemas=read(pub/'checkpoint-integrity.json'); files=[];cp=[]
 for chain in sorted(root.glob('chain-*')):
  t=read(chain/'terminal-receipt.json');assert t['status']=='TERMINAL_VALID'
  for p in sorted((chain/'checkpoints').glob('*.pt')):
   s=p.lstat();assert stat.S_ISREG(s.st_mode) and p.resolve()==p and s.st_nlink==1 and s.st_uid==os.getuid()
   rec=p.with_suffix('.receipt.json');r=read(rec);h=sha(p);assert h==r['sha256'] and s.st_size==r['bytes']
   schema=next(x for x in schemas if x['path']==str(p));assert schema['status']=='CPU_TENSOR_FULL_REHASH_PASS'
   after=p.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
   cp.append(dict(path=str(p),realpath=str(p),owner='janghj',uid=s.st_uid,type='regular',nlink=1,dev=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=h,arm=chain.name,batch=int(p.name.split('-')[0][1:]),schema=schema,reuse=False,retained_path='/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/jvp1k-v1/payload/'+str(p.relative_to(R)),receipt=member(rec)))
   files.append(rec);print('JVP_HASHED',p.name,chain.name,flush=True)
  files.extend([chain/'terminal-receipt.json',chain/'runtime.lock.json'])
  for d in sorted(chain.glob('batch-*')):
   # Essential chronological metadata/targets only. No evaluator raw/log transfer.
   for p in d.iterdir():
    if p.is_file() and ('commit' in p.name or 'context' in p.name or p.name in ['entry.json','fixed-z.pt']):files.append(p)
 for p in root.iterdir():
  if p.is_file() and p.name.endswith('.lock.json'):files.append(p)
 lock=read(root/'source.lock.json');repo=Path(lock['repo'])
 for m in lock['kernel_members']:
  p=repo/m['path'];assert sha(p)==m['sha256'];files.append(p)
 files.append(Path(lock['contract_file']))
 save('jvp-source-manifest.json',dict(members=cp,count=len(cp),bytes=sum(e['bytes'] for e in cp),schema_reuse=member(pub/'checkpoint-integrity.json'),full_GPU_replay=0))
 cm=[dict(**member(p),relative=str(p.relative_to(R)),disposition='COPY_ONLY_SOURCE_PRESERVED') for p in sorted(set(files))]
 save('jvp-companion-manifest.json',dict(members=cm,count=len(cm),bytes=sum(m['bytes'] for m in cm)))
 for n,ls in [('jvp-payload-files.txt',[str(Path(e['path']).relative_to(R)) for e in cp]),('jvp-companion-files.txt',[e['relative'] for e in cm])]:
  with (C/n).open('x') as f:f.write('\n'.join(ls)+'\n')
if __name__=='__main__':run()
