"""Read-only exact checkpoint inventory; no deletion or experiment actions."""
import csv,json,hashlib,os,stat,pwd,time
from pathlib import Path
R=Path('/data/janghj/ODE-edit'); C=R/'local/checkpoint-migration-server2/20260911-v1'
W=R/'local/worktrees/server4-checkpoint-migration-server2-20260911-v1'
P=W/'experiment-reports/servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def save(n,v):
 with (C/n).open('x') as f:json.dump(v,f,indent=2)
def rows(p):return list(csv.DictReader(p.open()))
def member(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def main():
 start=time.time(); candidates=[]; companions={}; hold=[]
 pub=read(P/'analysis-manifest.json')
 for n in ['source-config-compatibility.csv','checkpoint-tensors.csv']:
  assert sha(P/n)==next(m['sha256'] for m in pub['members'] if m['path']==n)
 schema=rows(P/'checkpoint-tensors.csv')
 old=read(R/'local/blue-checkpoint-downstream-transfer/20260909-v1/source-package/checkpoint-manifest.json')
 reuse={e['file']['path']:e for e in old['checkpoints']}
 def addcomp(p):
  if p.exists():
   assert p.resolve()==p and p.is_file() and not p.is_symlink(),p
   companions[str(p)]=p
 def collect(root,arm,job,terminalname,kind,config=None):
  term=root/terminalname;t=read(term);addcomp(term)
  assert t['status']=='TERMINAL_VALID',(arm,t['status'])
  tm=t.get('manifest_members',[])
  cp=[m for m in tm if Path(m['path']).name in ['W-method-state.pt','W-M.pt']]
  assert len(cp)==(12 if kind=='lifelong' else 3),(arm,len(cp))
  rt=root/'runtime.json';addcomp(rt)
  for m in tm:
   rel=Path(m['path'])
   if rel.name in ['commit.json','entry.json','contexts.json','native-layer-targets.pt']:addcomp(root/rel)
  for m in cp:
   p=root/m['path'];batch=int(p.parent.name[1:])
   candidates.append(dict(path=str(p),expected_sha256=m['sha256'],expected_bytes=m['bytes'],arm=arm,job=job,batch=batch,edits=batch*100,bundle=kind,terminal=member(term),schema=[s for s in schema if s['arm']==arm and int(s['batch'])==batch],config=config))
 for c in rows(P/'source-config-compatibility.csv'):
  root=Path(c['raw_root']);attempt=root.parent.parent
  collect(root,c['arm'],c['job'],'terminal.json','lifelong',c)
  for p in [attempt/'execution.lock.json',Path(c['source_archive'])]:addcomp(p)
  lock=read(attempt/'execution.lock.json')
  # Config/source closure only, never shared weights/P/stats or evaluation raw.
  for m in lock.get('members',[]):
   p=Path(m['path'])
   if p.is_relative_to(attempt) and (p.suffix in ['.py','.json','.yaml','.yml','.sh']):addcomp(p)
 one=[('BLUE_1K','38940',R/'local/blue-alphaedit-sequential-comparison/attempt-v1/execution-tech-r2/main-llama'),('BLUE_L4_1K','38997',R/'local/blue-alphaedit-l4-oneshot-sequential/attempt-v1/main-llama'),('BLUE_L8_1K','38988',R/'local/blue-alphaedit-l8-oneshot-sequential/attempt-v1/main-llama')]
 for arm,job,root in one:
  collect(root,arm,job,'terminal.json','blue-1k')
  for p in root.parent.iterdir():
   if p.is_file() and (p.name.endswith('.lock.json') or p.name.endswith('.tar') or p.name.endswith('.tar.gz') or p.suffix=='.py'):addcomp(p)
 # JVP checkpoint schema is separate; inventory first and preserve pending schema review.
 jroot=R/'local/state/alpha-jv-migration-server4-20260907/tech-r1'
 for chain in sorted(jroot.glob('chain-*')):
  terminal=chain/'terminal-receipt.json';t=read(terminal)
  for p in sorted((chain/'checkpoints').glob('*.pt')):
   receipt=p.with_suffix('.receipt.json')
   hold.append(dict(path=str(p),bytes=p.stat().st_size,owner=pwd.getpwuid(p.stat().st_uid).pw_name,status='INVENTORY_JVP_SCHEMA_REVIEW_PENDING',receipt=str(receipt),terminal_status=t['status']))
 # Current file full SHA, stable stats, exact published/terminal binding.
 result=[]
 for i,e in enumerate(candidates):
  p=Path(e['path']);s=p.lstat()
  if not stat.S_ISREG(s.st_mode) or p.resolve()!=p or s.st_uid!=os.getuid() or s.st_nlink!=1:
   hold.append(dict(**e,status='HOLD_PATH_OWNER_LINK'));continue
  h=sha(p);after=p.stat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),p
  assert h==e['expected_sha256'] and s.st_size==e['expected_bytes'],p
  e.update(realpath=str(p.resolve()),owner=pwd.getpwuid(s.st_uid).pw_name,uid=s.st_uid,type='regular',nlink=s.st_nlink,dev=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=h)
  if str(p) in reuse:
   olde=reuse[str(p)];assert olde['file']['sha256']==h
   e['retained_path']='/mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1/'+olde['file']['destination_relative']
   e['reuse']=True
  else:e['retained_path']='/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/new177-v1/payload/'+str(p.relative_to(R));e['reuse']=False
  result.append(e)
  print('HASHED',i+1,len(candidates),e['arm'],e['batch'],flush=True)
 save('source-manifest.json',dict(instruction_id='ODEEDIT-S06-SERVER4-CHECKPOINT-MIGRATION-SERVER2-V1',members=result,count=len(result),bytes=sum(e['bytes'] for e in result),reuse_count=sum(e['reuse'] for e in result),source_report_manifest=member(P/'analysis-manifest.json'),held=hold,seconds=time.time()-start))
 # Companion files remain on source. Their names/content are local-only.
 cm=[]
 for p in sorted(companions.values()):cm.append(dict(**member(p),relative=str(p.relative_to(R)),disposition='COPY_ONLY_SOURCE_PRESERVED'))
 save('companion-manifest.json',dict(members=cm,count=len(cm),bytes=sum(m['bytes'] for m in cm)))
 with (C/'payload-files.txt').open('x') as f:f.write(''.join(str(Path(e['path']).relative_to(R))+'\n' for e in result if not e['reuse']))
 with (C/'companion-files.txt').open('x') as f:f.write(''.join(m['relative']+'\n' for m in cm))
 print('MANIFEST_READY',len(result),sum(e['bytes'] for e in result),'COMPANIONS',len(cm),sum(m['bytes'] for m in cm),flush=True)
if __name__=='__main__':main()
