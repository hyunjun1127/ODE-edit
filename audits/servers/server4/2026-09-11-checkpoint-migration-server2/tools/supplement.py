from inventory import *
def run():
 ms=read(C/'source-manifest.json')['members'];files={};bindings=[]
 for arm in ['BLUE_1K','BLUE_L4_1K','BLUE_L8_1K']:
  e=next(x for x in ms if x['arm']==arm);root=Path(e['path']).parent.parent;lp=root.parent/'execution.lock.json';l=read(lp)
  for m in l.get('members',[])+l.get('adapter_members',[]):
   p=Path(m['path'])
   if p.is_relative_to(R) and p.suffix in ['.py','.json','.yaml','.yml','.sh','.toml','.txt'] and p.stat().st_size<10_000_000 and 'dataset' not in str(p):
    assert sha(p)==m['sha256'];files[str(p)]=p
  files[str(lp)]=lp
  files[str(Path(l['config']))]=Path(l['config'])
  if l.get('source_archive'):files[l['source_archive']['path']]=Path(l['source_archive']['path'])
  bindings.append(dict(arm=arm,lock=member(lp),base_revision=l['revision'],base_snapshot=l['snapshot'],projector=l['projector'],source=l['source_root'],layers=l['hparams']['layers'],rng='NOT_SAVED_IN_1K_CHECKPOINT; seed policy recorded only',restore_scope='selected W + dense M + contexts; GPU continuation not tested'))
 cm=[dict(**member(p),relative=str(p.relative_to(R)),disposition='COPY_ONLY_SOURCE_PRESERVED') for p in sorted(files.values())]
 save('supplement-manifest.json',dict(members=cm,count=len(cm),bytes=sum(m['bytes'] for m in cm),bindings=bindings))
 with (C/'supplement-files.txt').open('x') as f:f.write(''.join(m['relative']+'\n' for m in cm))
 print(len(cm),sum(m['bytes'] for m in cm))
if __name__=='__main__':run()
